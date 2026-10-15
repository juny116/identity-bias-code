import json
import numpy as np
from tqdm import tqdm
import random
import hydra
from omegaconf import DictConfig
from glob import glob
from data import read_jsonl, parse_option, parse_answer


def compute_accuracy(gt, pred_solutions):
    if type(pred_solutions) == list:
        pred_answers = []
        for pred_solution in pred_solutions:
            cor = False
            inc = False
            if pred_solution.find("INCORRECT") != -1:
                inc = True
            elif pred_solution.find("CORRECT") != -1:
                cor = True
            if cor and inc:
                pred_answers.append("INVALID")
            elif cor:
                pred_answers.append(False)
            elif inc:
                pred_answers.append(True)
            else:
                pred_answers.append("INVALID")
                # print(pred_solution)
            # pred_answers.append(pred_solution.strip())
        pred_answer = most_frequent(pred_answers)
    else:
        print(pred_solutions)
        pred_answer = pred_solutions.strip()
    if type(gt) == list:
        gt = gt[0]

    # print(pred_answers, pred_answer, gt)
    if pred_answer == gt:
        return 1, pred_answer
    else:
        return 0, pred_answer


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(config: DictConfig) -> None:
    # set seed
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    # results = read_jsonl(config["result_path"])
    out_path = config["out_path"]
    target = "gpt-3.5"
    # load data
    data = read_jsonl(f"data/gsm-{target}.jsonl")
    # gpt-3.5
    with open("results/gsm/gpt-3.5/dict/cot.json", "r") as f:
        gpt_dict = json.load(f)
    with open(f"results/gsm/gpt-3.5/dict/{target}.json", "r") as f:
        gpt_result = json.load(f)
        # dict to list
        gpt_result = [gpt_result[str(i)] for i in range(len(gpt_result))]
    # llama3-70b
    with open("results/gsm/llama3-70b/dict/cot.json", "r") as f:
        llama_dict = json.load(f)
    with open(f"results/gsm/llama3-70b/dict/{target}.json", "r") as f:
        llama_result = json.load(f)
        # dict to list
        llama_result = [llama_result[str(i)] for i in range(len(llama_result))]
    # gpt-4o
    with open("results/gsm/gpt-4o/dict/cot.json", "r") as f:
        gpt4o_dict = json.load(f)
    with open(f"results/gsm/gpt-4o/dict/{target}.json", "r") as f:
        gpt4o_result = json.load(f)
        # dict to list
        gpt4o_result = [gpt4o_result[str(i)] for i in range(len(gpt4o_result))]

    # for d, g3, l3, g4 in zip(data, gpt_result, llama_result, gpt4o_result):
    #     # print(
    #     #     f"GOLD: {not(d['error'])}\tGPT-3.5: {g3}\tACC: {gpt_dict[str(d['idx'])][1]}\tLLAMA3-70b: {l3}\tACC: {llama_dict[str(d['idx'])][1]}\tGPT-4o: {g4}\tACC: {gpt4o_dict[str(d['idx'])][1]}"
    #     # )
    #     gold = 1 if not d["error"] else 0
    #     print(
    #         f"{gold}\t{g3}\t{gpt_dict[str(d['idx'])][1]}\t{l3}\t{llama_dict[str(d['idx'])][1]}\t{g4}\t{gpt4o_dict[str(d['idx'])][1]}"
    #     )

    # gpt-3.5 math
    data = read_jsonl(f"data/math-{target}.jsonl")
    with open("results/math/gpt-3.5/dict/cot.json", "r") as f:
        gpt_dict = json.load(f)
    with open(f"results/math/gpt-3.5/dict/{target}.json", "r") as f:
        gpt_result = json.load(f)
        # dict to list
        gpt_result = [gpt_result[str(i)] for i in range(len(gpt_result))]
    # llama3-70b
    with open("results/math/llama3-70b/dict/cot.json", "r") as f:
        llama_dict = json.load(f)
    with open(f"results/math/llama3-70b/dict/{target}.json", "r") as f:
        llama_result = json.load(f)
        # dict to list
        llama_result = [llama_result[str(i)] for i in range(len(llama_result))]
    # gpt-4o
    with open("results/math/gpt-4o/dict/cot.json", "r") as f:
        gpt4o_dict = json.load(f)
    with open(f"results/math/gpt-4o/dict/{target}.json", "r") as f:
        gpt4o_result = json.load(f)
        # dict to list
        gpt4o_result = [gpt4o_result[str(i)] for i in range(len(gpt4o_result))]

    for d, g3, l3, g4 in zip(data, gpt_result, llama_result, gpt4o_result):
        # print(
        #     f"GOLD: {not(d['error'])}\tGPT-3.5: {g3}\tACC: {gpt_dict[str(d['idx'])][1]}\tLLAMA3-70b: {l3}\tACC: {llama_dict[str(d['idx'])][1]}\tGPT-4o: {g4}\tACC: {gpt4o_dict[str(d['idx'])][1]}"
        # )
        gold = 1 if not d["error"] else 0
        print(
            f"{gold}\t{g3}\t{gpt_dict[str(d['idx'])][1]}\t{l3}\t{llama_dict[str(d['idx'])][1]}\t{g4}\t{gpt4o_dict[str(d['idx'])][1]}"
        )
    # correct_list = []
    # pred_answers = []
    # for i, r in enumerate(results):
    #     acc, pred_answer = compute_accuracy(r["error"], r["pred"][:1])
    #     correct_list.append(acc)
    #     pred_answers.append(pred_answer)

    # print(f"{np.mean(correct_list)}")
    # count where pred answer is correct
    # COR = 0
    # INC = 0
    # for p in pred_answers:
    #     if p == False:
    #         COR += 1
    #     else:
    #         INC += 1
    # print(f"Correct: {COR}, Incorrect: {INC}")


if __name__ == "__main__":
    main()
