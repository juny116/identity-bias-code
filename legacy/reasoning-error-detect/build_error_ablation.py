import json
import numpy as np
import random
import hydra
from omegaconf import DictConfig
from data import read_jsonl, parse_math, parse_gsm, parse_correct, parse_tf


def check_answer(gt, pred_solution, parse_fn):
    pred_answer = parse_fn(pred_solution).strip()
    try:
        pred_answer = str(int(float(pred_answer)))
    except:
        pass

    if pred_answer == gt:
        return 1, pred_answer
    else:
        return 0, pred_answer


@hydra.main(version_base=None, config_path="conf", config_name="sample")
def main(config: DictConfig) -> None:
    # set seed
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    out_path = config["out_path"]
    manual_ban_list = config["manual_ban_list"]
    max_example_num = 50

    parse_fn = parse_tf

    target = "strategy"
    gpt35 = read_jsonl(f"results/{target}/gemini.jsonl")
    llama3 = read_jsonl(f"results/{target}/llama3.jsonl")
    gpt4 = read_jsonl(f"results/{target}/gpt-4.jsonl")
    gpt35_cor = []
    gpt35_inc = []
    llama3_cor = []
    llama3_inc = []
    gpt4_cor = []
    gpt4_inc = []

    # gpt-3.5
    for i, r in enumerate(gpt35):
        if i in manual_ban_list:
            continue
        # if all 5 samples are correct, select the last one
        has_result = False
        for idx, p in enumerate(r["pred"]):
            acc, ans = check_answer(r["gold"], r["pred"][idx], parse_fn)
            # if invalid answer, continue
            if ans == "":
                continue
            # if correct add example and check next sample
            if acc:
                has_result = True
                correct = True
            # if incorrect add to incorrect list and break
            else:
                has_result = True
                correct = False
                gpt35_inc.append(i)
                break
        # if no error and has result, add to correct list
        if correct and has_result:
            gpt35_cor.append(i)

    # llama3-70b
    for i, r in enumerate(llama3):
        if i in manual_ban_list:
            continue
        # if all 5 samples are correct, select the last one
        has_result = False
        for idx, p in enumerate(r["pred"]):
            acc, ans = check_answer(r["gold"], r["pred"][idx], parse_fn)
            # if invalid answer, continue
            if ans == "":
                continue
            # if correct add example and check next sample
            if acc:
                has_result = True
                correct = True
            # if incorrect add to incorrect list and break
            else:
                has_result = True
                correct = False
                llama3_inc.append(i)
                break
        # if no error and has result, add to correct list
        if correct and has_result:
            llama3_cor.append(i)

    # gpt-4o
    for i, r in enumerate(gpt4):
        if i in manual_ban_list:
            continue
        # if all 5 samples are correct, select the last one
        has_result = False
        for idx, p in enumerate(r["pred"]):
            acc, ans = check_answer(r["gold"], r["pred"][idx], parse_fn)
            # if invalid answer, continue
            if ans == "":
                continue
            # if correct add example and check next sample
            if acc:
                has_result = True
                correct = True
            # if incorrect add to incorrect list and break
            else:
                has_result = True
                correct = False
                gpt4_inc.append(i)
                break
        # if no error and has result, add to correct list
        if correct and has_result:
            gpt4_cor.append(i)

    print("gemini correct: ", len(gpt35_cor))
    print("gemini incorrect: ", len(gpt35_inc))
    print("LLAMA3-70b correct: ", len(llama3_cor))
    print("LLAMA3-70b incorrect: ", len(llama3_inc))
    print("GPT-4o correct: ", len(gpt4_cor))
    print("GPT-4o incorrect: ", len(gpt4_inc))
    # get list of indices that are correct in all 3 models
    cor_list = list(set(gpt35_cor) & set(llama3_cor) & set(gpt4_cor))
    print("All correct examples: ", len(cor_list))
    # get list of indices that are incorrect in all 3 models
    inc_list = list(set(gpt35_inc) & set(llama3_inc) & set(gpt4_inc))
    print("All incorrect examples: ", len(inc_list))

    # get first 50 examples from each list
    cor_list = cor_list[:max_example_num]
    inc_list = inc_list[:max_example_num]
    # print(cor_list)
    # print(inc_list)
    # combine and shuffle
    combined_list = cor_list + inc_list
    random.shuffle(combined_list)
    print(combined_list)

    new_gpt35 = []
    for i, r in enumerate(gpt35):
        if i in combined_list:
            if i in cor_list:
                example = {
                    "idx": i,
                    "correct": "True",
                    "gold": r["gold"],
                    "pred": ans,
                    "solution": r["pred"][idx],
                    "question": r["question"],
                }
                if "contexts" in r:
                    example["contexts"] = r["contexts"]
            else:
                example = {
                    "idx": i,
                    "correct": "False",
                    "gold": r["gold"],
                    "pred": ans,
                    "solution": r["pred"][idx],
                    "question": r["question"],
                }
                if "contexts" in r:
                    example["contexts"] = r["contexts"]

            new_gpt35.append(example)

    new_gpt4 = []
    for i, r in enumerate(gpt4):
        if i in combined_list:
            if i in cor_list:
                example = {
                    "idx": i,
                    "correct": "True",
                    "gold": r["gold"],
                    "pred": ans,
                    "solution": r["pred"][idx],
                    "question": r["question"],
                }
                if "contexts" in r:
                    example["contexts"] = r["contexts"]
            else:
                example = {
                    "idx": i,
                    "correct": "False",
                    "gold": r["gold"],
                    "pred": ans,
                    "solution": r["pred"][idx],
                    "question": r["question"],
                }
                if "contexts" in r:
                    example["contexts"] = r["contexts"]

            new_gpt4.append(example)

    new_llama3 = []
    for i, r in enumerate(llama3):
        if i in combined_list:
            if i in cor_list:
                example = {
                    "idx": i,
                    "correct": "True",
                    "gold": r["gold"],
                    "pred": ans,
                    "solution": r["pred"][idx],
                    "question": r["question"],
                }
                if "contexts" in r:
                    example["contexts"] = r["contexts"]
            else:
                example = {
                    "idx": i,
                    "correct": "False",
                    "gold": r["gold"],
                    "pred": ans,
                    "solution": r["pred"][idx],
                    "question": r["question"],
                }
                if "contexts" in r:
                    example["contexts"] = r["contexts"]

            new_llama3.append(example)
    # write
    out_path = f"data/ablation/{target}-gemini.jsonl"
    with open(out_path, "w") as f:
        for d in new_gpt35:
            f.write(json.dumps(d) + "\n")

    out_path = f"data/ablation/{target}-llama3.jsonl"
    with open(out_path, "w") as f:
        for d in new_llama3:
            f.write(json.dumps(d) + "\n")

    out_path = f"data/ablation/{target}-gpt-4.jsonl"
    with open(out_path, "w") as f:
        for d in new_gpt4:
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    main()
