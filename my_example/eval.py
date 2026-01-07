from __future__ import annotations

from typing import Iterable, Tuple

from tqdm.auto import tqdm

from .prompts import solution_start, solution_end
from .rewards import MATCH_FORMAT, MATCH_NUMBERS


def _extract_number(response: str) -> str | None:
    match = MATCH_NUMBERS.search(response)
    return match.group(1) if match else None


def evaluate(
    dataset,
    sampler,
    temperature: float | None,
    top_k: int,
    top_p: float | None,
    num_passes: int = 1,
    verbose: bool = False,
) -> Tuple[int, int, float, float, float]:
    """Computes accuracy and percentage of outputs matching the format."""
    response_lst = []
    corr = 0
    partially_corr = 0
    corr_format = 0
    total = 0

    for batch in tqdm(dataset):
        answers = batch["answer"]
        questions = batch["question"]

        multiple_call_responses = [[] for _ in range(len(questions))]
        for p in range(num_passes):
            responses = sampler.generate(
                questions, temperature=temperature, top_k=top_k, top_p=top_p, seed=p
            )
            for idx, response in enumerate(responses):
                multiple_call_responses[idx].append(response)
                if verbose:
                    print(f"Question:\t{questions[idx]}")
                    print(f"Correct Answer:\t{answers[idx]}")
                    print(f"Response:\t{response}")
                    print("-" * 50)

        for question, multiple_call_response, answer in zip(
            questions, multiple_call_responses, answers
        ):
            corr_ctr_per_question = 0
            partially_corr_per_question = 0
            corr_format_per_question = 0

            for response in multiple_call_response:
                extracted_response = _extract_number(response) or "-1000000"
                try:
                    if float(extracted_response.strip()) == float(answer.strip()):
                        corr_ctr_per_question += 1

                    ratio = float(extracted_response.strip()) / float(answer.strip())
                    if 0.9 <= ratio <= 1.1:
                        partially_corr_per_question += 1
                except Exception:
                    pass

                if MATCH_FORMAT.search(response) is not None:
                    corr_format_per_question += 1

                if (
                    corr_ctr_per_question > 0
                    and partially_corr_per_question > 0
                    and corr_format_per_question > 0
                ):
                    break

            if corr_ctr_per_question > 0:
                corr += 1
            if partially_corr_per_question > 0:
                partially_corr += 1
            if corr_format_per_question > 0:
                corr_format += 1

            total += 1
            if verbose and total % 10 == 0:
                print(
                    f"===> corr={corr}, total={total}, "
                    f"corr%={corr / total * 100:.2f}, "
                    f"partial%={partially_corr / total * 100:.2f}, "
                    f"format%={corr_format / total * 100:.2f}"
                )

    return (
        corr,
        total,
        corr / total * 100,
        partially_corr / total * 100,
        corr_format / total * 100,
    )
