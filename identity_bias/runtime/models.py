"""Load named model profiles from a caller-supplied JSON file."""

import json
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from identity_bias.types import LLMConfig, LLMProvider


def load_model_profiles(path):
    with Path(path).expanduser().open() as stream:
        definitions = json.load(stream)
    profiles = {}
    for name, values in definitions.items():
        values = dict(values)
        values["provider"] = LLMProvider(values["provider"])
        key_env = values.pop("api_key_env", None)
        if key_env:
            values["api_key"] = os.environ.get(key_env)
        profile = LLMConfig(**values)

        def factory(base_url=None, _profile=profile):
            return replace(_profile, **({"base_url": base_url} if base_url else {}))

        profiles[name] = factory
    return profiles


class ModelProfiles(Mapping):
    """Read profiles on demand, so importing a module performs no file I/O."""

    def _load(self):
        path = os.environ.get("IDENTITY_BIAS_MODELS")
        return load_model_profiles(path) if path else {}

    def __getitem__(self, name):
        profiles = self._load()
        if name not in profiles:
            raise ValueError(
                f"Unknown model profile {name!r}; provide --models-file before the command"
            )
        return profiles[name]

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())


MODEL_PRESETS = ModelProfiles()
