from __future__ import annotations

import importlib.util
from typing import Iterable, List


REQUIRED_IMPORTS = [
    ("jax", "jax"),
    ("flax", "flax"),
    ("grain", "grain"),
    ("tensorflow_datasets", "tensorflow_datasets"),
    ("kagglehub", "kagglehub"),
    ("huggingface_hub", "huggingface_hub"),
    ("optax", "optax"),
    ("orbax", "orbax"),
    ("qwix", "qwix"),
    ("tunix", "tunix"),
]


def find_missing(imports: Iterable[tuple[str, str]] = REQUIRED_IMPORTS) -> List[str]:
    missing: List[str] = []
    for module_name, pip_name in imports:
        if importlib.util.find_spec(module_name) is None:
            missing.append(pip_name)
    return missing


def assert_dependencies() -> None:
    missing = find_missing()
    if missing:
        raise RuntimeError(
            "Missing dependencies: "
            + ", ".join(missing)
            + ". Install them in your environment before running."
        )
