from models import AutoModelForAPI
import asyncio
import json
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from data import read_jsonl, write_jsonl
from utils import messages_to_string
import os
import logging

logger = logging.getLogger("httpx")
logger.propagate = False


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    max_examples = config["max_examples"]
    # load model
    model_config = config["model"]
    model = AutoModelForAPI(
        model_type=model_config["model_type"],
        api_key=model_config["api_key"],
        model=model_config["model_name_or_path"],
        base_url=model_config["base_url"],
        organization=model_config["organization"],
        project=model_config["project"],
    )

    instruction = config["template"]["instruction"]
    form = config["template"]["format"]

    prompt = instruction

    test_data = read_jsonl(config["test_path"])
    # if the result file exists, read it
    if os.path.exists(config["out_path"]):
        results = read_jsonl(config["out_path"])
    else:
        results = []
    # if len(results) > 0 continue from last checkpoint
    for i, d in enumerate(tqdm(test_data[:max_examples])):
        if len(results) > 0 and i < len(results):
            continue
        answers = []
        token_lengths = []
        for r in range(0, config["rounds"]):
            context = ""
            if "contexts" in d:
                for k, c in enumerate(d["contexts"]):
                    context += f"Fact {k+1}. {c}\n"
                context += "\n"
            query = (
                form.replace("[QUESTION]", d["question"])
                .replace("[ANSWER]", "")
                .replace("[CONTEXTS]", context)
            )
            # query = (
            #     form.replace("[QUESTION]", d["question"]).replace("[ANSWER]", "").replace("[CONTEXTS]", d["solution"])
            #     # .replace("[SOLUTION]", d["solution"])
            # )
            context = prompt + query
            messages = [
                {
                    "role": "user",
                    "content": context,
                },
            ]
            if i == 0:
                print(messages_to_string(messages))
                print("=====================================")
            result = model(
                messages,
                max_tokens=model_config["max_tokens"],
                temperature=model_config["temperature"],
            )
            token_len = result["tokens"]["completion_tokens"]
            gen_result = result["messages"][-1]["content"]
            token_lengths.append(token_len)
            answers.append(gen_result)

        results.append(
            {
                "gold": d["answer"],
                "pred": answers,
                "token_lengths": token_lengths,
                "input": messages[0]["content"],
                "input_tokens": result["tokens"]["prompt_tokens"],
            }
        )

        # write results to jsonl
        out_path = config["out_path"]
        write_jsonl(results, out_path)


if __name__ == "__main__":
    main()
