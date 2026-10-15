import json
import numpy as np
import random
import hydra
from omegaconf import DictConfig
from data import read_jsonl, parse_math, parse_gsm, parse_correct
from utils import most_frequent


def compute_accuracy(gt, pred_solutions, parse_fn):
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
    correct = 0
    error = 0
    # print(pred_answer, gt)
    # check how many answers are correct
    if type(pred_solutions) == list:
        for p in pred_answers:
            if p == gt:
                correct += 1
            elif p == "Error":
                error += 1
        avg_score = correct / len(pred_answers)
        avg_error = error / len(pred_answers)
    else:
        avg_error = 0
        if pred_answer == gt:
            avg_score = 1
        else:
            avg_score = 0
            if pred_answer == "Error":
                avg_error = 1

    # return the correctness of the ensemble and the average score
    if pred_answer == gt:
        return 1, avg_score, avg_error
    else:
        return 0, avg_score, avg_error


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(config: DictConfig) -> None:
    # set seed
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    results = read_jsonl(config["result_path"])
    out_path = config["out_path"]
    num_ensembles = config["num_ensembles"]
    if config["task"] == "gsm" or config["task"] == "fermi":
        parse_fn = parse_gsm
    elif config["task"] == "math":
        parse_fn = parse_math
    elif config["task"] == "correct":
        parse_fn = parse_correct
    correct_list = []
    reuslt_dict = {}
    avg_list = []
    avg_error_list = []
    for i, r in enumerate(results):
        acc, avg, error = compute_accuracy(
            r["gold"], r["pred"][:num_ensembles], parse_fn
        )
        print("INPUT")
        print(r["input"])
        print("OUTPUT")
        print(r["pred"][0])
        print("GOLD", r["gold"])
        correct_list.append(acc)
        avg_list.append(avg)
        avg_error_list.append(error)
        reuslt_dict[i] = (acc, avg)
        input("next")

    # print(len(correct_list))
    # print(f"{np.mean(correct_list)}")
    # print(f"{np.mean(avg_list)}")
    # print(f"{np.mean(avg_error_list)}")
    # with open(out_path, "w") as f:
    #     json.dump(reuslt_dict, f)


if __name__ == "__main__":
    main()
