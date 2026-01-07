from __future__ import annotations

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

SYSTEM_PROMPT = (
    "You are given a problem. First, think about the problem "
    f"and provide your reasoning. Place it between {reasoning_start} and "
    f"{reasoning_end}. Then, provide the final answer (i.e., just one numerical "
    f"value) between {solution_start} and {solution_end}."
)

TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model
"""


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()
