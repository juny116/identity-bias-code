import json
import numpy as np
import random
import hydra
from omegaconf import DictConfig
from data import read_jsonl, parse_math, parse_gsm, parse_tf


def check_answer(gt, pred_solution, parse_fn):
    pred_answer = parse_fn(pred_solution).strip()
    # print("Pred answer: ", pred_answer)
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
    results = read_jsonl(config["result_path"])
    out_path = config["out_path"]
    manual_ban_list = config["manual_ban_list"]
    max_example_num = config["max_example_num"]
    if config["task"] == "gsm":
        parse_fn = parse_gsm
    elif config["task"] == "math" or config["task"] == "fermi":
        parse_fn = parse_math
    elif config["task"] == "strategy":
        parse_fn = parse_tf

    correct_list = []
    incorrect_list = []
    # for i, r in enumerate(results):
    for i, r in enumerate(results):
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
            # if incorrect add to incorrect list and break
            else:
                has_result = True
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
                incorrect_list.append(example)
                break
        # if no error and has result, add to correct list
        if example["correct"] == "True" and has_result:
            correct_list.append(example)

    max_examples = min(len(correct_list), len(incorrect_list), max_example_num)
    print("Correct examples: ", len(correct_list))
    print("Incorrect examples: ", len(incorrect_list))
    print("Total examples: ", len(correct_list) + len(incorrect_list))
    print("Max examples per label: ", max_examples)

    # sample max_examples from correct and incorrect
    correct_list = random.sample(correct_list, int(max_examples))
    incorrect_list = random.sample(incorrect_list, int(max_examples))
    # combine and shuffle
    combined_list = correct_list + incorrect_list

    # check for duplicates
    idx_list = []
    for d in combined_list:
        idx_list.append(d["idx"])
    assert len(set(idx_list)) == len(idx_list)

    random.shuffle(combined_list)

    # write to out_file
    with open(out_path, "w") as f:
        for d in combined_list:
            json.dump(d, f)
            f.write("\n")


if __name__ == "__main__":
    main()
