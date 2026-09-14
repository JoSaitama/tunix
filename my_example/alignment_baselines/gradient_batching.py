"""Pure helpers for memory-safe prompt-gradient batching."""

from __future__ import annotations


def feature_completion_slices(
    *,
    prompt_count: int,
    num_rollouts: int,
    promptwise: bool,
) -> tuple[slice, ...]:
    """Returns completion-axis slices without changing prompt/rollout order.

    A generated selector chunk is laid out as contiguous rollout groups:
    ``[prompt_0 rollout_0..G, prompt_1 rollout_0..G, ...]``. LearnAlign
    evaluates one such group at a time to avoid materializing the full
    per-completion gradient tree for every prompt in the generation chunk.
    """
    if prompt_count <= 0:
        raise ValueError("prompt_count must be positive")
    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be positive")
    completion_count = prompt_count * num_rollouts
    if not promptwise:
        return (slice(0, completion_count),)
    return tuple(
        slice(prompt_index * num_rollouts, (prompt_index + 1) * num_rollouts)
        for prompt_index in range(prompt_count)
    )
