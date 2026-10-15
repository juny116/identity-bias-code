import json
import numpy as np
import random
from data import read_jsonl, parse_math, parse_gsm, parse_correct

target = "gsm"
gemini = read_jsonl(f"results/{target}/gemini.jsonl")
llama3 = read_jsonl(f"results/{target}/llama3.jsonl")
gpt4 = read_jsonl(f"results/{target}/gpt-4.jsonl")

list = json.loads(__import__("pathlib").Path(__import__("os").environ["RESAMPLE_INDICES_PATH"]).read_text())

# sample only data with index in list
new_gemini = []
for d in gemini:
    if d["idx"] in list:
        new_gemini.append(d)

new_llama3 = []
for d in llama3:
    if d["idx"] in list:
        new_llama3.append(d)

new_gpt4 = []
for d in gpt4:
    if d["idx"] in list:
        new_gpt4.append(d)

# write new data to jsonl
out_path = f"data/ablation/{target}-gemini.jsonl"
with open(out_path, "w") as f:
    for d in new_gemini:
        f.write(json.dumps(d) + "\n")

out_path = f"data/ablation/{target}-llama3.jsonl"
with open(out_path, "w") as f:
    for d in new_llama3:
        f.write(json.dumps(d) + "\n")

out_path = f"data/ablation/{target}-gpt-4.jsonl"
with open(out_path, "w") as f:
    for d in new_gpt4:
        f.write(json.dumps(d) + "\n")
