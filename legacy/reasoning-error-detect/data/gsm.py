import json
import pandas as pd
from glob import glob
import re
import numpy as np
import argparse
from utils import read_jsonl, solve_math_problems
from datasets import load_dataset

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="gsm8k")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--out_file", type=str, default="gsm_test_hf.jsonl")
    args.add_argument("--max_examples", type=int, default=-1)
    args = args.parse_args()

    np.random.seed(args.seed)
    # load hf dataset
    data = load_dataset(args.data_path, "main", split="test")

    new_data = []
    for d in data:
        new_data.append(
            {
                "question": d["question"],
                "answer": solve_math_problems(d["answer"]),
                "solution": d["answer"],
                "options": {},
            }
        )

    # # dump data as jsonl
    with open(args.out_file, "w") as f:
        for d in new_data:
            json.dump(d, f)
            f.write("\n")
