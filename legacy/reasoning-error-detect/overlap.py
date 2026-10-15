import json
import numpy as np
import random
import hydra
from omegaconf import DictConfig
from data import read_jsonl, parse_math, parse_gsm, parse_correct
from utils import most_frequent
from scipy.stats.stats import pearsonr


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
    # results = read_jsonl(config["result_path"])
    d_name = "gemini"
    # d_name = "llama3-70b"
    # d_name = "gpt-4o"

    t_name = "gsm"
    # t_name = "math"

    m_name = "gemini"
    # m_name = "llama3-70b"
    # m_name = "gpt-4o"

    results = read_jsonl(f"results/{t_name}/{m_name}.jsonl")
    # results = read_jsonl(f"results/{t_name}/llama3-70b/parse/cot.jsonl")

    idx_path = f"data/{t_name}-{d_name}.jsonl"
    idx = read_jsonl(idx_path)
    idx_list = []
    for i in idx:
        idx_list.append(i["idx"])
    # error_path = "results/math/llama3-70b/error/gpt-4o.jsonl"
    error_path = "results/math/gpt-4o/error/gpt-4o.jsonl"
    # error_path = f"results/{t_name}/{m_name}/error/{d_name}.jsonl"
    error = read_jsonl(error_path)

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
    avg_error = []
    avg_task = []
    error_list = []
    overlap = []
    poten = 0
    total_inc = 0
    for i, index in enumerate(idx_list):
        r = results[index]
        error_example = error[i]
        acc, avg = compute_accuracy(r["gold"], r["pred"][:5], parse_fn)
        avg_task.append(avg)
        e_acc, e_avg = compute_accuracy(
            error_example["correct"], error_example["pred"][:5], parse_correct
        )
        error_list.append(e_acc)
        avg_error.append(e_avg)
        correct_list.append(acc)
        reuslt_dict[i] = (acc, avg)
        if acc == 0:
            total_inc += 1
        if acc == 0 and e_acc == 1:
            poten += 1

        if acc == e_acc:
            overlap.append(1)
        else:
            overlap.append(0)
        # print("-------------------")
        # print(idx[i]["question"])
        # print(r["input"])
        # print(error_example["question"])
        # print(avg, e_avg)
        # print(acc, e_acc)
        # print("-------------------")

    print(np.corrcoef(avg_task, avg_error))
    # for i, r in enumerate(results):
    #     if i not in idx_list:
    #         continue
    #     # acc, avg = compute_accuracy(r["gold"], r["pred"][:num_ensembles], parse_fn)
    #     acc, avg = compute_accuracy(r["gold"], r["pred"][:5], parse_fn)
    #     error_example = error[e_cnt]
    #     e_acc, e_avg = compute_accuracy(
    #         error_example["correct"], error_example["pred"][:5], parse_correct
    #     )

    #     correct_list.append(acc)
    #     reuslt_dict[i] = (acc, avg)
    #     print("-------------------")
    #     print(idx[e_cnt]["question"])
    #     print(r["question"])
    #     print(error_example["question"])
    #     print(avg, e_avg)
    #     print("-------------------")
    #     e_cnt += 1

    print(len(correct_list))
    print(f"{np.mean(correct_list)}")
    print(f"{np.mean(overlap)}")
    print(f"{np.mean(poten/total_inc)}")
    # print(f"{np.mean(error_list)}")

    with open(out_path, "w") as f:
        json.dump(reuslt_dict, f)


if __name__ == "__main__":
    main()
