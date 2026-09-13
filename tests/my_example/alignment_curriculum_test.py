from __future__ import annotations

import json
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from my_example.alignment_baselines.artifacts import SelectionArtifactWriter
from my_example.alignment_baselines.curriculum import (
    GradAlignCurriculum,
    LearnAlignCurriculum,
)


class _Estimator:
    def estimate(self, examples, *, num_rollouts, apply_mismatch):
        indices = np.asarray([example["index"] for example in examples])
        features = np.stack([indices + 1.0, np.ones_like(indices)], axis=1)
        clean_outcomes = np.zeros(
            (len(examples), num_rollouts), dtype=np.float64
        )
        clean_outcomes[:, : max(1, num_rollouts // 2)] = 1.0
        selector_rewards = clean_outcomes.copy()
        mismatch_selected = np.zeros(len(examples), dtype=np.bool_)
        mismatch_effective = np.zeros(len(examples), dtype=np.bool_)
        if apply_mismatch:
            selector_rewards = selector_rewards[:, ::-1]
            mismatch_selected[:] = True
            mismatch_effective[:] = True
        advantages = selector_rewards - selector_rewards.mean(
            axis=1, keepdims=True
        )
        return SimpleNamespace(
            features=features,
            clean_binary_outcomes=clean_outcomes,
            selector_rewards=selector_rewards,
            selector_advantages=advantages,
            mismatch_selected=mismatch_selected,
            mismatch_effective=mismatch_effective,
        )


class _Cluster:
    def __init__(self):
        self.logged = []

    def log_scalar_immediately(self, name, value, step):
        self.logged.append((name, value, step))


def _examples(count):
    return [
        {
            "prompts": f"prompt-{index}",
            "answer": str(index),
            "index": index,
        }
        for index in range(count)
    ]


class AlignmentCurriculumTest(unittest.TestCase):

    def test_learnalign_preserves_total_update_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            curriculum = LearnAlignCurriculum(
                training_examples=_examples(16),
                train_batch_size=4,
                total_update_steps=7,
                estimator=_Estimator(),
                writer=SelectionArtifactWriter(tmp, "learnalign"),
                rl_cluster=_Cluster(),
                warmup_prompts=8,
                estimation_rollouts=4,
                selection_ratio=4,
                warmup_counts_toward_budget=True,
            )
            batches = list(curriculum)
            self.assertEqual(len(batches), 7)
            self.assertEqual(curriculum.effective_max_steps, 7)
            with open(
                f"{tmp}/learnalign_selection.jsonl", encoding="utf-8"
            ) as stream:
                first_record = json.loads(next(stream))
            self.assertEqual(
                first_record["clean_binary_outcomes"], [1, 1, 0, 0]
            )
            self.assertEqual(first_record["selector_rewards"], [0, 0, 1, 1])
            self.assertEqual(
                first_record["selector_advantages"], [-0.5, -0.5, 0.5, 0.5]
            )
            self.assertTrue(first_record["mismatch_selected"])
            self.assertTrue(first_record["mismatch_effective"])

    def test_gradalign_selects_exact_interval_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            curriculum = GradAlignCurriculum(
                training_examples=_examples(24),
                validation_examples=_examples(4),
                train_batch_size=4,
                total_update_steps=5,
                estimator=_Estimator(),
                writer=SelectionArtifactWriter(tmp, "gradalign"),
                rl_cluster=_Cluster(),
                candidate_rollouts=4,
                validation_rollouts=4,
                selection_ratio=4,
                selection_interval=3,
            )
            batches = list(curriculum)
            self.assertEqual(len(batches), 5)
            self.assertTrue(all(len(batch["prompts"]) == 4 for batch in batches))
            with open(
                f"{tmp}/gradalign_selection.jsonl", encoding="utf-8"
            ) as stream:
                records = [json.loads(line) for line in stream]
            self.assertTrue(
                any(record["record_type"] == "validation" for record in records)
            )
            validation = next(
                record for record in records if record["record_type"] == "validation"
            )
            self.assertEqual(
                validation["clean_binary_outcomes"], [1, 1, 0, 0]
            )
            self.assertEqual(validation["selector_rewards"], [1, 1, 0, 0])
            self.assertEqual(
                validation["selector_advantages"], [0.5, 0.5, -0.5, -0.5]
            )
            self.assertFalse(validation["mismatch_selected"])
            self.assertFalse(validation["mismatch_effective"])
            candidate = next(
                record for record in records if record["record_type"] == "candidate"
            )
            self.assertEqual(
                candidate["clean_binary_outcomes"], [1, 1, 0, 0]
            )
            self.assertEqual(candidate["selector_rewards"], [0, 0, 1, 1])
            self.assertEqual(
                candidate["selector_advantages"], [-0.5, -0.5, 0.5, 0.5]
            )
            self.assertTrue(candidate["mismatch_selected"])
            self.assertTrue(candidate["mismatch_effective"])


if __name__ == "__main__":
    unittest.main()
