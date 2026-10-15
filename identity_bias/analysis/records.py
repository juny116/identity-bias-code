"""Load judge records and apply optional choice-answer corrections."""

import pandas as pd

from identity_bias.io.jsonl import read_directory

from .label_corrections import apply


def load_judge_records(directory, *, corrections=None):
    frame = pd.DataFrame(read_directory(directory))
    return apply(frame, corrections=corrections)[0]
