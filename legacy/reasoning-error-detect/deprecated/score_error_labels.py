import json
import numpy as np
import random
import hydra
from omegaconf import DictConfig
from data import read_jsonl, parse_correct
from response_utils import most_frequent


def compute_accuracy(gt, pred_solutions):
    if type(pred_solutions) == list:
        pred_answers = []
        for pred_solution in pred_solutions:
            pred_answers.append(parse_correct(pred_solution))
        pred_answer = most_frequent(pred_answers)
    else:
        pred_answer = parse_correct(pred_solutions)
    if type(gt) == list:
        gt = gt[0]

    correct = 0
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

    if pred_answer == gt:
        return 1, avg_score
    else:
        return 0, avg_score


@hydra.main(version_base=None, config_path="conf", config_name="eval_error")
def main(config: DictConfig) -> None:
    # set seed
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    results = read_jsonl(config["result_path"])
    out_path = config["out_path"]
    num_ensembles = config["num_ensembles"]
    correct_list = []
    result_dict = {}
    for i, r in enumerate(results):
        acc, avg = compute_accuracy(r["error"], r["pred"][:num_ensembles])
        correct_list.append(acc)
        result_dict[i] = avg

    print(f"{np.mean(correct_list)}")

    with open(out_path, "w") as f:
        json.dump(result_dict, f)


if __name__ == "__main__":
    main()
