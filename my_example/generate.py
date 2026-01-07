from __future__ import annotations

from typing import Iterable, List

from tunix.generate import sampler as sampler_lib

from .prompts import SYSTEM_PROMPT, TEMPLATE


class SamplerWrapper:
    def __init__(self, transformer, tokenizer, cache_config, eos_tokens=None):
        self._sampler = sampler_lib.Sampler(
            transformer=transformer,
            tokenizer=tokenizer,
            cache_config=cache_config,
        )
        self._eos_tokens = eos_tokens or []

    def generate(
        self,
        questions,
        temperature: float | None,
        top_k: int,
        top_p: float | None,
        seed: int | None = None,
        eos_tokens: list[int] | None = None,
        max_generation_steps: int = 768,
    ) -> List[str]:
        if isinstance(questions, str):
            input_batch = [
                TEMPLATE.format(
                    system_prompt=SYSTEM_PROMPT,
                    question=questions,
                )
            ]
        else:
            input_batch = [
                TEMPLATE.format(
                    system_prompt=SYSTEM_PROMPT,
                    question=q,
                )
                for q in questions
            ]

        out_data = self._sampler(
            input_strings=input_batch,
            max_generation_steps=max_generation_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            echo=False,
            seed=seed,
            eos_tokens=eos_tokens or self._eos_tokens,
        )
        output = out_data.text
        if isinstance(questions, str):
            return [output[0]]
        return output
