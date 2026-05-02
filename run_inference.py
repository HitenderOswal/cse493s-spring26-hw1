import math

from vllm import LLM, SamplingParams
import re

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import time

# helper function to generate prompts
def generate_prompts(dataset, enable_thinking: bool): # Set False for no-thinking condition
    prompts = []
    gold_answers = []

    for i, example in tqdm(enumerate(dataset)):
        problem = example["prompt"][0]["content"]
        gold_answer = int(example["label"])

        messages = [{"role": "system", "content": "You are a careful competition math assistant. Always output your final answer in \\boxed{}."},
                    {"role": "user", "content": problem}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts.append(prompt)
        gold_answers.append(gold_answer)
    
    return prompts, gold_answers


# helper functions for answer extraction
def strip_thinking_trace(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", "", text, flags=re.DOTALL)
    return text.strip()

def extract_answer(text: str, mode="exact_match") -> int | None:
    """Extract an AIME-style integer answer from a model completion."""
    answer_text = strip_thinking_trace(text)
    if not answer_text:
        if mode == "exact_match":
            return None
        else:
            answer_text = text  # fall back to full text


    # 1. Boxed LaTeX answer: \boxed{123}
    if mode == "exact_match":
        boxed = re.findall(r"\\boxed\{(\d+)\}", answer_text)
        if boxed:
            val = int(boxed[-1])
            return val
        else:
            return None

    elif mode == "flexible_extract":
        # 2. "The answer is N" or "answer: N" patterns
        patterns = [
            r"(?:the\s+)?answer\s+is\s+[:\s]*(\d+)",
            r"answer[:\s]+(\d+)",
            r"=\s*(\d+)\s*$",
            r"(?:therefore|thus|so),?\s+(\d+)\s*(?:\.|$)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, answer_text, re.IGNORECASE)
            if matches:
                val = int(matches[-1])
                return val

        # 3. Last integer in [0, 999] in the answer portion
        integers = re.findall(r"\b(\d{1,3})\b", answer_text)
        for candidate in reversed(integers):
            val = int(candidate)
            return val
        return None

# token counting
THINK_OPEN = 151667
THINK_CLOSE = 151668

def get_prompt_token_count(list_ids):
  try:
    start = list_ids.index(THINK_OPEN)
    return start
  except ValueError:
    return 0

def get_think_token_count(list_ids):
  input_len = get_prompt_token_count(list_ids)
  try:
    end = list_ids.index(THINK_CLOSE)
    return (end - input_len)
  except ValueError:
    return (len(list_ids) - input_len)

# inference once
def inference_once(model, prompts, sampling_params):
    return model.generate(prompts, sampling_params)

# inference until we hit the token limit
def inference_seq(model, prompts, sampling_params, wait_token_id):
    # index the prompts so we can keep track of them through reorders
    states = {i: prompt for i, prompt in enumerate(prompts)}
    final_prompts = [None] * len(prompts)
    finished = [None] * len(prompts)
    token_lens = [0] * len(prompts)
    
    answer_params = SamplingParams(
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0
    )
    
    while states:
        items = list(states.items())
        idxs, curr_prompts = zip(*items)
        
        curr_state = model.generate(list(curr_prompts), sampling_params)
        new_states = {}
        answer_states = {}
        
        for i, prompt, res in zip(idxs, curr_prompts, curr_state):
            old_len = token_lens[i]
            token_lens[i] += get_think_token_count(res.outputs[0].token_ids)
            if token_lens[i] >= sampling_params.max_tokens:
                # trim to max tokens, close the think, and make it generate an answer
                # we know the first token is always <think>, so we can just trim to max_tokens and add </think> at the end
                gen = res.outputs[0].token_ids[:sampling_params.max_tokens - old_len] + [THINK_CLOSE]
                answer_states[i] = prompt + tokenizer.decode(gen)
            else:
                # trim ending think off the unfinished generations, and feed back into the model
                gen = res.outputs[0].token_ids
                try:
                    end_idx = gen.index(THINK_CLOSE)
                    gen = gen[:end_idx]
                except ValueError:
                    pass
                gen += [wait_token_id]  # add a wait token to prevent the model from immediately generating more thinking tokens
                new_states[i] = prompt + tokenizer.decode(gen)
        
        answer_idxs, answer_prompts = zip(*list(answer_states.items())) if answer_states else ([], [])
        if answer_prompts:
            answer_results = model.generate(list(answer_prompts), answer_params)
            for i, answer_prompt, res in zip(answer_idxs, answer_prompts, answer_results):
                finished[i] = res
                final_prompts[i] = answer_prompt
        
        states = new_states
        print(f"Current Token lengths: {token_lens}")
    return final_prompts, finished
    
def inference_seq_with_analysis(model, prompts, gold_answers, max_token_lengths, wait_token_id, answer_mode):
    for max_tokens in max_token_lengths:
        print(f"Running inference with max_tokens={max_tokens}...")
        start = time.perf_counter()
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0
        )
        
        final_prompts, outputs = inference_seq(model, prompts, sampling_params, wait_token_id)
        end = time.perf_counter()

        results = [output.outputs[0].text if output is not None else "" for output in outputs]

        records = []
        token_lens = []

        for i in range(len(results)):
            gold_answer = gold_answers[i]
            extracted = extract_answer(results[i], mode=answer_mode)
            correct = extracted == gold_answers[i] if extracted is not None else False
            token_ids = outputs[i].outputs[0].token_ids if outputs[i] is not None else []

            records.append({
                "prompt": prompts[i],
                "gold": gold_answers[i],
                "output": results[i],
                "extracted": extracted,
                "correct": correct
            })
            
            token_lens.append(get_think_token_count(token_ids))

            print(f"[{i+1}/{len(results)}] gold={gold_answer} pred={extracted} correct={correct}")

        results_df = pd.DataFrame(records)
        print(f"Elapsed time for {max_tokens}: {end - start:.6f} seconds")
        print(f"saving results to results_seq_{max_tokens}.csv and final_gens_seq_{max_tokens}.json")
        results_df.to_csv(f"results_seq_{max_tokens}.csv", index=False)
        with open(f"final_gens_seq_{max_tokens}.json", "w") as f:
            json.dump(final_prompts, f)

        print(results_df.to_string())
        print(token_lens)
        print(results_df["correct"].mean())
        
        
def inference_parallel(model, prompts, dups=8):
    prompts = [prompt for prompt in prompts for _ in range(dups)]
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        stop=["</think>"]
    )
    thinks = model.generate(prompts, sampling_params)
    think_texts = [think.outputs[0].text + "</think>" if "</think>" not in think.outputs[0].text else think.outputs[0].text for think in thinks]
    answer_prompts = [prompt + think for prompt, think in zip(prompts, think_texts)]
    
    answer_params = SamplingParams(
        max_tokens=2048,
        temperature=0.6,
        top_p=0.95,
        top_k=50
    )
    answers = model.generate(answer_prompts, answer_params)
    
    text_gens = [think + answer.outputs[0].text for think, answer in zip(think_texts, answers)]
    return text_gens

def inference_parallel_with_analysis(model, prompts, gold_answers, dups=8):
    start = time.perf_counter()
    results = inference_parallel(model, prompts, dups)
    end = time.perf_counter()
    answer_modes = ["exact_match", "flexible_extract"]
    for answer_mode in answer_modes:
        records = []
        for i in range(len(results) // dups):
            corrects = 0
            answers = []
            gold_answer = gold_answers[i]
            for j in range(dups):
                extracted = extract_answer(results[i * dups + j], mode=answer_mode)
                answers.append(extracted)
                correct = extracted == gold_answer if extracted is not None else False
                if correct:
                    corrects += 1
                    
            majority_answer = max(set(answers), key=answers.count)
            majority_correct = majority_answer == gold_answer if majority_answer is not None else False

            any_correct = corrects > 0
            
            records.append({
                "prompt": prompts[i],
                "gold": gold_answer,
                "output": [results[i * dups + j] for j in range(0, dups)],
                "extracted": majority_answer,
                "majority_correct": majority_correct,
                "any_correct": any_correct
            })
            print(f"[{i+1}/{len(results) // dups}] gold={gold_answer} pred={majority_answer} majority_correct={majority_correct} any_correct={any_correct}")
        
        results_df = pd.DataFrame(records)
        print(results_df.to_string())
        print(f"Majority: {results_df["majority_correct"].mean()}")
        print(f"Any: {results_df["any_correct"].mean()}")
        
        print(f"Elapsed time for parallel scaling: {end - start:.6f} seconds")
        print(f"saving results to results_parallel_{dups}_{answer_mode}.csv")
        results_df.to_csv(f"results_parallel_{dups}_{answer_mode}.csv", index=False)

# not used, vllm documentation is not clear enough on how to implement custom logit processors on newer versions
def xtc_logits_processor(logits):
    top_p_threshold=0.5
    penalty=0.2
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    top_p = cumulative_probs <= top_p_threshold
    k = top_p.sum(dim=-1)
    k = torch.clamp(k - 3, min=0) # keep bottom 3 in top-p
    B, V = probs.shape
    rank_idx = torch.arange(V, device=probs.device).unsqueeze(0).expand(B, V)
    selected_idx = rank_idx < k.unsqueeze(1)    
    selected_values = sorted_idx[selected_idx]
    
    row_idx = torch.arange(B, device=probs.device).unsqueeze(1).expand(B, V)
    selected_rows = row_idx[selected_idx]
    
    mask = torch.zeros_like(top_p, dtype=torch.bool)
    mask[selected_rows, selected_values] = True

    logits = logits - penalty * mask.float()
    return logits
        

def improved_inference_parallel(model, prompts, dups=8):
    prompts = [prompt for prompt in prompts for _ in range(dups)]
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        stop=["</think>"],
        logprobs=1
    )
    thinks = model.generate(prompts, sampling_params)
    think_texts = [think.outputs[0].text + "</think>" if "</think>" not in think.outputs[0].text else think.outputs[0].text for think in thinks]
    answer_prompts = [prompt + think for prompt, think in zip(prompts, think_texts)]
    
    answer_params = SamplingParams(
        max_tokens=2048,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        logprobs=1
    )
    answers = model.generate(answer_prompts, answer_params)
    
    text_gens = [think + answer.outputs[0].text for think, answer in zip(think_texts, answers)]
    ids = [think.outputs[0].token_ids + answer.outputs[0].token_ids for think, answer in zip(thinks, answers)]
    logprob_objects = [think.outputs[0].logprobs + answer.outputs[0].logprobs for think, answer in zip(thinks, answers)]
    logprobs = []
    for lp, tids in zip(logprob_objects, ids):
        logprob = []
        for token_id, logp in zip(tids, lp):
            logprob.append(logp[token_id])
        logprobs.append(logprob)
    return text_gens, logprobs

def improved_inference_parallel_with_analysis(model, prompts, gold_answers, dups=8):
    start = time.perf_counter()
    results, logprobs = improved_inference_parallel(model, prompts, dups)
    end = time.perf_counter()
    print(f"Elapsed time for improved parallel scaling: {end - start:.6f} seconds")

    answer_modes = ["exact_match", "flexible_extract"]
    for answer_mode in answer_modes:
        records = []
        for i in range(len(results) // dups):
            answers = []
            scores = []
            gold_answer = gold_answers[i]
            for j in range(dups):
                extracted = extract_answer(results[i * dups + j], mode=answer_mode)
                answers.append(extracted)
                lp = sum(t.logprob for t in logprobs[i * dups + j]) / len(logprobs[i * dups + j])
                score = math.exp(lp)
                scores.append(score)
            votes = {}
            for a, w in zip(answers, scores):
                if a:
                    if a in votes:
                        votes[a] += w
                    else:
                        votes[a] = w
            if votes:
                majority_answer = max(votes, key=votes.get)
            else:
                majority_answer = None
            majority_correct = majority_answer == gold_answer if majority_answer is not None else False

            
            records.append({
                "prompt": prompts[i],
                "gold": gold_answer,
                "output": [results[i * dups + j] for j in range(0, dups)],
                "extracted": majority_answer,
                "majority_correct": majority_correct,
            })
            print(f"[{i+1}/{len(results) // dups}] gold={gold_answer} pred={majority_answer} majority_correct={majority_correct}")
        
        results_df = pd.DataFrame(records)
        print(results_df.to_string())
        print(f"Majority: {results_df["majority_correct"].mean()}")
        
        print(f"saving results to results_improved_parallel_{dups}_{answer_mode}.csv")
        results_df.to_csv(f"results_improved_parallel_{dups}_{answer_mode}.csv", index=False)

# load model and dataset
MODEL_NAME = "Qwen/Qwen3-4B"
# MODEL_NAME = "allenai/Olmo-3-7B-Thinking"
DATASET_NAME = "OpenRLHF/aime-2024"
MAX_NEW_TOKENS = 32768

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = LLM(model=MODEL_NAME)

sampling_params = SamplingParams(
    max_tokens=MAX_NEW_TOKENS,
    temperature=0.0,
    top_p=1.0
)

dataset = load_dataset(DATASET_NAME, split="train")

prompts, gold_answers = generate_prompts(dataset, enable_thinking=True)

WAIT_TOKEN_ID = tokenizer.encode("Wait", add_special_tokens=False)[0]
print(f"Wait token id: {WAIT_TOKEN_ID}")
print(f"Think open token id: {tokenizer.encode('<think>', add_special_tokens=False)[0]}")
print(f"Think close token id: {tokenizer.encode('</think>', add_special_tokens=False)[0]}")

ANSWER_MODE = "exact_match"

# SEQUENTIAL SCALING TEST
# max_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
# inference_seq_with_analysis(model, prompts, gold_answers, max_lengths, WAIT_TOKEN_ID, ANSWER_MODE)

# PARALLEL SCALING TEST
# num_dupes = [1, 2, 4, 8, 16, 32]
# for dups in num_dupes:
#     print(f"Running parallel inference with {dups} duplicates...")
#     inference_parallel_with_analysis(model, prompts, gold_answers, dups=dups)
    
# IMPROVED PARALLEL SCALING TEST
num_dupes = [4, 8] # 16k and 32k token budgets (each run is 4k tokens)
for dups in num_dupes:
    print(f"Running improved parallel inference with {dups} duplicates...")
    improved_inference_parallel_with_analysis(model, prompts, gold_answers, dups=dups)
    # inference_parallel_with_analysis(model, prompts, gold_answers, dups=dups)