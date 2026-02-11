from __future__ import annotations

from typing import List

from tunix.generate import sampler as sampler_lib

from .prompts import BOXED_INSTRUCTION


class SamplerWrapper:
    def __init__(self, transformer, tokenizer, cache_config, eos_tokens=None):
        self._sampler = sampler_lib.Sampler(
            transformer=transformer,
            tokenizer=tokenizer,
            cache_config=cache_config,
        )
        self._tokenizer = tokenizer
        self._eos_tokens = eos_tokens or []

    def _build_prompt(self, question: str) -> str:
        content = f"{question}\n{BOXED_INSTRUCTION}"
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def generate(
        self,
        questions,
        temperature: float | None,
        top_k: int | None,
        top_p: float | None,
        seed: int | None = None,
        eos_tokens: list[int] | None = None,
        max_generation_steps: int = 2048,
    ) -> List[str]:
        if isinstance(questions, str):
            input_batch = [self._build_prompt(questions)]
        else:
            input_batch = [self._build_prompt(q) for q in questions]

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

