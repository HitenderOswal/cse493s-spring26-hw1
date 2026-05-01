# produce inferences for arithemetic operations

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from part_0_1_contract import load_model_and_tokenizer, predict_answer

def main():
  parser = argparse.ArgumentParser(description="Addition inference script")
  parser.add_argument("--ckpt-dir", type=str, required=True, help="Checkpoint directory")
  parser.add_argument("--a", type=int, required=True, help="First operand")
  parser.add_argument("--b", type=int, required=True, help="Second operand")
  parser.add_argument("--p", type=int, required=True, help="Modulus: 97 or 113")
  parser.add_argument("--op", type=str, required=True, help="Operator: +, -, /")
  args = parser.parse_args()

  model, tokenizer = load_model_and_tokenizer(args.ckpt_dir)
  pred = predict_answer(model, tokenizer, args.a, args.b, args.op, args.p)
  print(f"{args.a} {args.op} {args.b} % {args.p} = {pred}")

if __name__ == "__main__":
  main()