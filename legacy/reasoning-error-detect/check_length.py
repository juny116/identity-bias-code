import json
import numpy as np
import random
from data import read_jsonl, parse_math, parse_gsm, parse_correct


d1 = read_jsonl(f"results/gsm/error-gemini.jsonl")

# print average input length of correct true and false

correct_true = []
correct_false = []

for d in d1:
    if d["gold"] == "True":
        correct_true.append(d["input_tokens"])
    else:
        correct_false.append(d["input_tokens"])

print(np.mean([x for x in correct_true]))
print(np.mean([x for x in correct_false]))

# sort d by input_tokens
d1 = sorted(d1, key=lambda x: x["input_tokens"])
# sample 5 shortest and 5 longest
shortest = d1[:5]
longest = d1[-5:]

# print shortest and longest
print("SHORTEST")
for s in shortest:
    print(s["gold"])
    print(s["input"])
    # print()
    # for p in s["pred"]:
    #     print("PRED")
    #     print(p)
    #     print()
    print()

print("LONGEST")
for l in longest:
    print(l["gold"])
    print(l["input"])
    # print()
    # for p in l["pred"]:
    #     print("PRED")
    #     print(p)
    #     print()
    print()
