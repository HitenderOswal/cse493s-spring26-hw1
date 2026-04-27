import argparse
from pathlib import Path
import numpy as np

import tiktoken

def main() -> None:
    parser = argparse.ArgumentParser(description="Simple tokenizer script")
    parser.add_argument("--input", required=True, help="Path to input data.")
    parser.add_argument("--output", required=True, help="Path to output data.")
    args = parser.parse_args()

    tokenizer = tiktoken.encoding_for_model("gpt2")

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8").strip()
    
    output_tokens = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    
    output_path = Path(args.output)
    
    seq = np.array([output_tokens], dtype=np.uint16)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seq.tofile(output_path)

    print(tokenizer.decode(output_tokens))

if __name__ == "__main__":
    main()