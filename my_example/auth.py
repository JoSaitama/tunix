from __future__ import annotations

import os
from typing import Optional

import kagglehub


def get_hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def ensure_kaggle_login() -> None:
    has_user = bool(os.environ.get("KAGGLE_USERNAME"))
    has_key = bool(os.environ.get("KAGGLE_KEY"))
    if not (has_user and has_key):
        kagglehub.login()


def maybe_init_wandb(use_wandb: bool) -> None:
    if not use_wandb:
        return
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not found. Skipping wandb init.")
        return
    import wandb

    wandb.init()
