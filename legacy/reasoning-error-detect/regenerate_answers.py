from data import read_jsonl, write_jsonl
import json

target = "gemini"
original_path = f"data/fermi.jsonl"
pred_path = f"data/fermi-{target}.jsonl"
output_path = f"data/fermi-{target}-out.jsonl"

data = read_jsonl(original_path)
pred_data = read_jsonl(pred_path)

# new_data = []
# for i, d in enumerate(data):
#     new_data.append(
#         {
#             "correct": not d["error"],
#             "pred": d["pred"],
#             "question": d["question"],
#             "solution": d["solution"],
#             "token_lengths": d["token_lengths"],
#         }
#     )

for d in pred_data:
    d["contexts"] = data[d["idx"]]["contexts"]

write_jsonl(pred_data, output_path)
