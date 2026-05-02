"""
CSE 493S/599S HW2: interface for Part 0 and Part 1.

We will be using an autograder for this part. For ease of grading, please fill in
these functions to evaluate your trained models. Do not rename the functions
or change their signatures.

You may import from other files in your repo. You may add helper functions.
Just make sure the three functions below work as specified.
"""

import torch
from model import GPT, GPTConfig
from pathlib import Path

import tiktoken

def load_model_and_tokenizer(checkpoint_dir: str):
    """
    Load a trained model and its tokenizer from a checkpoint directory.

    Args:
        checkpoint_dir: Path to a directory containing your saved model
            and any tokenizer files you need.

    Returns:
        A tuple (model, tokenizer). The model should be ready for inference
        (in eval mode, on an appropriate device). The tokenizer should be
        whatever object your predict_answer / generate_sanity_check functions
        expect — we do not constrain its type.
    """


    """
    Tokenizer for arithmetic tasks.

    Args:
        c: The character to tokenize.
        p: The modulus (97 or 113).

    Returns:
        The tokenized character.
    """
    def arithmetic_tokenizer(c, p):
        if c == '+':
            return p
        elif c == '-':
            return p + 1
        elif c == '/':
            return p + 2
        elif c == '=':
            return p + 3
        return c

    ckpt_path = Path(checkpoint_dir) / "ckpt.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])

    model.to(device).eval()
    
    if checkpoint["model_config"]["vocab_size"] == 50304:
        tokenizer = tiktoken.encoding_for_model("gpt2")
    else:
        tokenizer = arithmetic_tokenizer

    return model, tokenizer


def get_bos_token(tokenizer=None):
    """
    Get the BOS token for the tokenizer, for part 0 of the assignment.
    """
    return tokenizer.eot_token


def predict_answer(model, tokenizer, a: int, b: int, op: str, p: int) -> int:
    """
    Predict the answer to a modular arithmetic problem.

    Args:
        model: The model returned by load_model_and_tokenizer.
        tokenizer: The tokenizer returned by load_model_and_tokenizer.
        a: First operand, integer in [0, p).
        b: Second operand, integer in [0, p).
        op: One of '+', '-', '/'.
        p: The modulus (97 or 113).

    Returns:
        The model's predicted answer as an integer in [0, p).
        You are responsible for formatting the input according to your
        training scheme and parsing the model's output back to an integer.
    """

    device = next(model.parameters()).device
    inp = torch.tensor([[tokenizer(a, p), tokenizer(op, p), tokenizer(b, p), tokenizer('=', p)]], dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits = model(inp)[:, -1, :]
        pred = torch.argmax(logits, dim=-1).item()

    return int(pred)
