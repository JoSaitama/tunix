from __future__ import annotations

import os
from typing import Any, Tuple

from tqdm.auto import tqdm
from tunix.utils import math_utils


THOUGHT_DELIMITER_END = "</think>"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _extract_answer_from_response(response: Any) -> str | None:
    if response is None or response == "":
        return None
    response_text = str(response)
    if THOUGHT_DELIMITER_END in response_text:
        model_solution = response_text.split(THOUGHT_DELIMITER_END, 1)[1]
    else:
        model_solution = response_text
    return math_utils.extract_answer(model_solution)


def evaluate_correctness(response: Any, ground_truths: Any) -> bool:
    """Align with deepscaler AIME correctness logic."""
    if response is None or response == "":
        return False

    model_answer = _extract_answer_from_response(response)
    if model_answer is None:
        return False
    if ground_truths is None:
        return False

    if isinstance(ground_truths, str | float | int):
        ground_truths = [ground_truths]

    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = math_utils.extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        else:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        return False

    for ground_truth in processed_ground_truths:
        is_correct = math_utils.grade_answer_mathd(
            model_answer, ground_truth
        ) or math_utils.grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            return True
    return False


def evaluate(
    dataset,
    sampler,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    num_passes: int = 1,
    verbose: bool = False,
) -> Tuple[int, int, float]:
    diag_enabled = _env_flag("TUNIX_QWEN_AIME_DIAG")
    response_total = 0
    response_boxed = 0
    response_parseable = 0

    corr = 0
    total = 0

    for batch in tqdm(dataset):
        answers = batch["answer"]
        questions = batch["question"]

        multiple_call_responses = [[] for _ in range(len(questions))]
        for pass_id in range(num_passes):
            responses = sampler.generate(
                questions,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=pass_id,
            )
            for idx, response in enumerate(responses):
                multiple_call_responses[idx].append(response)
                if diag_enabled:
                    response_text = "" if response is None else str(response)
                    response_total += 1
                    if "\\boxed" in response_text:
                        response_boxed += 1
                    if _extract_answer_from_response(response_text) is not None:
                        response_parseable += 1
                if verbose:
                    print(f"Question:\t{questions[idx]}")
                    print(f"Correct Answer:\t{answers[idx]}")
                    print(f"Response:\t{response}")
                    print("-" * 50)

        for question, responses, answer in zip(questions, multiple_call_responses, answers):
            del question  # Not needed below.
            question_correct = False

            for response in responses:
                is_correct = evaluate_correctness(response, answer)
                if is_correct:
                    question_correct = True

                if question_correct:
                    break

            if question_correct:
                corr += 1
            total += 1

            if verbose and total % 10 == 0:
                print(
                    f"===> corr={corr}, total={total}, "
                    f"corr%={corr / total * 100:.2f}"
                )

    if diag_enabled and response_total:
        print(
            "[eval-diag] "
            f"response_boxed_rate={response_boxed / response_total:.4f}, "
            f"response_parseable_rate={response_parseable / response_total:.4f}, "
            f"responses={response_total}"
        )

    return (
        corr,
        total,
        (corr / total * 100) if total else 0.0,
    )
