import argparse
from pathlib import Path

import torch
import tiktoken

from model import GPT, GPTConfig
from torch.nn import functional as F

def load_tokenizer():
    return tiktoken.encoding_for_model("gpt2")

@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def decode_tokens(tokenizer, token_ids, skip_special_tokens: bool = False):
    if skip_special_tokens:
        token_ids = [token_id for token_id in token_ids if token_id != tokenizer.eot_token]
    return tokenizer.decode(token_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple text-generation debug script.")
    parser.add_argument("--model-path", required=True, help="Path to a checkpoint file or checkpoint directory.")
    parser.add_argument("--prompt", default="I love machine learning", help="Prompt to extend.")
    parser.add_argument("--max-new-tokens", type=int, default=40, help="How many tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. Use 0 for greedy decoding.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tokenizer = load_tokenizer()
    checkpoint = torch.load(args.model_path, map_location=device)

    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device=device, dtype=dtype)
    model.eval()

    input_token_ids = tokenizer.encode(args.prompt, allowed_special={'<|endoftext|>'})
    input_ids = torch.tensor([input_token_ids], dtype=torch.long, device=device)
    print(decode_tokens(tokenizer, input_token_ids))
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    generated = output_ids[0, input_ids.shape[1] :]
    print(decode_tokens(tokenizer, output_ids[0].tolist()))
    if generated.numel() > 0:
        print("\n--- generated only ---")
        print(decode_tokens(tokenizer, generated.tolist(), skip_special_tokens=True))

if __name__ == "__main__":
    main()