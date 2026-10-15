import json
import numpy as np
import random
import hydra
from omegaconf import DictConfig
from data import read_jsonl, parse_math, parse_gsm, parse_correct, parse_tf
from response_utils import most_frequent

global cnt
cnt = 0


def compute_accuracy(gt, pred_solutions, parse_fn):
    global cnt
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
    pred_correct = 0
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
    if pred_answer == "Error":
        print("Error")
        cnt += 1
    if pred_answer == "True":
        pred_correct = 1
    if pred_answer == gt:
        return 1, avg_score, avg_error, pred_correct
    else:
        return 0, avg_score, avg_error, pred_correct


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
    elif config["task"] == "strategy":
        parse_fn = parse_tf

    correct_list = []
    reuslt_dict = {}
    avg_list = []
    avg_error_list = []
    correct_ratio = []
    token_lens = []
    for i, r in enumerate(results):
        token_lens.append(r["input_tokens"])

    # make 4 bins for token lengths where each bin has the same number of examples
    token_lens = np.array(token_lens)
    idx = np.argsort(token_lens)
    token_lens = token_lens[idx]
    bin_size = len(token_lens) // 4
    bins = []
    for i in range(4):
        bins.append(token_lens[i * bin_size : (i + 1) * bin_size])
    # get max token length for the each bin
    max_lens = [max(b) for b in bins]

    b1 = []
    b2 = []
    b3 = []
    b4 = []
    for i, r in enumerate(results):
        acc, avg, error, pred_correct = compute_accuracy(
            r["gold"], r["pred"][:num_ensembles], parse_fn
        )
        if r["input_tokens"] <= max_lens[0]:
            b1.append(acc)
        elif r["input_tokens"] <= max_lens[1]:
            b2.append(acc)
        elif r["input_tokens"] <= max_lens[2]:
            b3.append(acc)
        else:
            b4.append(acc)
        # correct_list.append(acc)
        # avg_list.append(avg)
        # avg_error_list.append(error)
        # correct_ratio.append(pred_correct)
        # reuslt_dict[i] = (acc, avg)

    # print(len(correct_list))
    # print(f"{np.mean(correct_list)}")
    # print(f"{np.mean(avg_list)}")
    # print(f"{np.mean(avg_error_list)}")
    # print(f"{np.mean(correct_ratio)}")
    # with open(out_path, "w") as f:
    #     json.dump(reuslt_dict, f)
    print(f"Bin 1: {np.mean(b1)}")
    print(f"Bin 2: {np.mean(b2)}")
    print(f"Bin 3: {np.mean(b3)}")
    print(f"Bin 4: {np.mean(b4)}")
    print(f"Error count: {cnt}")


if __name__ == "__main__":
    main()
