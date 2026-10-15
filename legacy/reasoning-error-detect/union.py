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


gpt4 = read_jsonl("results/math/gpt-4o/parse/cot.jsonl")
llama3 = read_jsonl("results/math/llama3-70b/parse/cot.jsonl")
gpt3 = read_jsonl("results/math/gpt-3.5/parse/cot.jsonl")
parse_fn = parse_math
correct_list = []
correct_g4 = []
correct_g3 = []
correct_l3 = []

for i, (g3, l3, g4) in enumerate(zip(gpt3, llama3, gpt4)):
    g3_acc, g3_avg = compute_accuracy(g3["gold"], g3["pred"][:5], parse_fn)
    l3_acc, l3_avg = compute_accuracy(l3["gold"], l3["pred"][:5], parse_fn)
    g4_acc, g4_avg = compute_accuracy(g4["gold"], g4["pred"][:5], parse_fn)

    correct_g4.append(g4_acc)
    correct_g3.append(g3_acc)
    correct_l3.append(l3_acc)
    if g3_acc == 1 or l3_acc == 1 or g4_acc == 1:
        correct_list.append(1)
    else:
        correct_list.append(0)

print(np.mean(correct_list))
print(np.mean(correct_g4))
print(np.mean(correct_g3))
print(np.mean(correct_l3))
