import re
import json


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def write_jsonl(data, path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def convert_to_number(string):
    string = string.replace("$", "")
    string = string.replace(",", "")
    try:
        float_n = float(string)
        int_n = int(float_n)
    except ValueError:
        return None

    if float_n == int_n:
        return int_n
    else:
        return float_n


def parse_option(input_str):
    pattern = r"\((\w)\)"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    return solution


def parse_answer(input_str):
    pattern = r"\{([0-9.,$]*)\}"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break

    return solution


def solve_math_problems(input_str):
    # pattern = r"\d+\.?\d*"
    pattern = r"#### .*"
    matches = re.findall(pattern, input_str)
    if matches:
        last = matches[-1]
        ans = re.sub(r"#### ", "", last)
        ans = re.sub(r",", "", ans)
        return ans

    return None


def parse_math(input_str):
    pattern = r"boxed\{.*\}"
    matches = re.findall(pattern, input_str)

    solution = ""

    for match_str in matches[::-1]:
        # solution = re.sub(r"[^0-9.]", "", match_str)
        solution = re.sub(r"boxed\{", "", match_str)
        solution = repl_last(solution, "}", "")
        # solution = match_str
        if solution:
            break

    return solution


def parse_gsm(input_str):
    pattern = r"boxed\{.*\}"
    matches = re.findall(pattern, input_str)

    solution = ""

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break

    solution = solution.strip()

    try:
        solution = str(int(float(solution)))
    except:
        pass

    return solution


def parse_correct(input_str):
    pattern = r"boxed\{.*\}"
    matches = re.findall(pattern, input_str)

    solution = ""

    for match_str in matches[::-1]:
        # solution = re.sub(r"[^0-9.]", "", match_str)
        solution = re.sub(r"boxed\{", "", match_str)
        solution = repl_last(solution, "}", "")
        # solution = match_str
        if solution:
            break
    result = "Error"
    if solution == "correct" or solution == "Correct" or solution == "CORRECT":
        result = "True"
    elif solution == "incorrect" or solution == "Incorrect" or solution == "INCORRECT":
        result = "False"
    if result == "Error":
        print("Error in parse_correct")
    return result


def repl_last(s, sub, repl):
    pattern = sub + "(?!.*" + sub + ")"
    return re.sub(pattern, repl, s, flags=re.DOTALL)


# def parse_correct(input_str):
#     result = ""
#     if input_str.find("INCORRECT") != -1:
#         result = "False"
#     elif input_str.find("CORRECT") != -1:
#         result = "True"
#     elif input_str.find("incorrect") != -1:
#         result = "False"
#     elif input_str.find("correct") != -1:
#         result = "True"

#     # return True if incorrect, False if correct, None if invalid
#     return result


def parse_tf(input_str):
    result = ""
    if input_str.find("TRUE") != -1:
        result = "True"
    elif input_str.find("True") != -1:
        result = "True"
    elif input_str.find("true") != -1:
        result = "True"
    elif input_str.find("FALSE") != -1:
        result = "False"
    elif input_str.find("False") != -1:
        result = "False"
    elif input_str.find("false") != -1:
        result = "False"
    return result
