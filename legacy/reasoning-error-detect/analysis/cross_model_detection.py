import json
import numpy as np
import random
import hydra
from omegaconf import DictConfig
from data import read_jsonl, parse_math, parse_gsm, parse_correct
from response_utils import most_frequent


def compute_accuracy(gt, pred_solutions, parse_fn):
    # print("new")
    # print(pred_solutions)
    if type(pred_solutions) == list:
        pred_answers = []
        for pred_solution in pred_solutions:
            pred_answer = parse_fn(pred_solution)
            if type(pred_answer) == str:
                pred_answer = pred_answer.strip()
            pred_answers.append(pred_answer)
        pred_answer = most_frequent(pred_answers)
    else:
        pred_answer = parse_fn(pred_solutions)
        if type(pred_answer) == str:
            pred_answer = pred_answer.strip()

    if type(gt) == list:
        gt = gt[0]
    # print(gt)
    correct = 0
    # print(pred_answer, gt)
    # check how many answers are correct
    if type(pred_solutions) == list:
        for p in pred_answers:
            if p == gt:
                correct += 1
        avg_score = correct / len(pred_answers)
    else:
        if pred_answer == gt:
            avg_score = 1
        else:
            avg_score = 0

    # return the correctness of the ensemble and the average score
    if pred_answer == gt:
        return 1, avg_score
    else:
        return 0, avg_score


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(config: DictConfig) -> None:
    # set seed
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    results = read_jsonl(config["result_path"])
    gpt35_path = "data/gsm-gpt-3.5.jsonl"
    gpt35 = read_jsonl(gpt35_path)
    gpt35_list = []
    gpt35_r = {}
    for i, v in enumerate(gpt35):
        gpt35_list.append(v["idx"])
        gpt35_r[v["idx"]] = i

    llama3_path = "data/gsm-llama3-70b.jsonl"
    llama3 = read_jsonl(llama3_path)
    llama3_list = []
    llama3_r = {}
    for i, v in enumerate(llama3):
        llama3_list.append(v["idx"])
        llama3_r[v["idx"]] = i

    gpt4_path = "data/gsm-gpt-4o.jsonl"
    gpt4 = read_jsonl(gpt4_path)
    gpt4_list = []
    gpt4_r = {}
    for i, v in enumerate(gpt4):
        gpt4_list.append(v["idx"])
        gpt4_r[v["idx"]] = i

    # get only the indices that are in all 3 datasets
    idx_list = list(set(gpt35_list) & set(llama3_list) & set(gpt4_list))

    out_path = config["out_path"]
    num_ensembles = config["num_ensembles"]
    if config["task"] == "gsm":
        parse_fn = parse_gsm
    elif config["task"] == "math":
        parse_fn = parse_math
    elif config["task"] == "correct":
        parse_fn = parse_correct
    correct_list = []
    reuslt_dict = {}
    avg_task = []
    for i, index in enumerate(idx_list):
        t = gpt35_r[index]
        r = results[t]
        acc, avg = compute_accuracy(r["correct"], r["pred"][:5], parse_fn)
        avg_task.append(avg)
        correct_list.append(acc)
        reuslt_dict[i] = (acc, avg)

    print(len(correct_list))
    print(f"{np.mean(correct_list)}")
    # print(f"{np.mean(error_list)}")

    with open(out_path, "w") as f:
        json.dump(reuslt_dict, f)


if __name__ == "__main__":
    main()
