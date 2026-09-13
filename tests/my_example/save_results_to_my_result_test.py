"""Tests for the shared GSM8K stdout evaluation-result format."""

from __future__ import annotations

import sys
import types
import unittest

# The desktop test runtime does not bundle TensorBoard. The parser under test is
# independent of EventAccumulator, so provide only the import surface needed to
# load the shared exporter without changing production behavior.
try:
    import tensorboard.backend.event_processing.event_accumulator  # noqa: F401
except ModuleNotFoundError:
    tensorboard = types.ModuleType("tensorboard")
    backend = types.ModuleType("tensorboard.backend")
    event_processing = types.ModuleType("tensorboard.backend.event_processing")
    event_accumulator = types.ModuleType(
        "tensorboard.backend.event_processing.event_accumulator"
    )
    event_accumulator.EventAccumulator = object
    tensorboard.backend = backend
    backend.event_processing = event_processing
    event_processing.event_accumulator = event_accumulator
    sys.modules["tensorboard"] = tensorboard
    sys.modules["tensorboard.backend"] = backend
    sys.modules["tensorboard.backend.event_processing"] = event_processing
    sys.modules[
        "tensorboard.backend.event_processing.event_accumulator"
    ] = event_accumulator

from my_example.save_results_to_my_result import _parse_eval_metrics


class ParseEvalMetricsTest(unittest.TestCase):

    def test_preserves_existing_pre_post_accuracy_structure(self):
        parsed = _parse_eval_metrics(
            "\n".join(
                [
                    "unrelated output",
                    (
                        "pre-train: num_correct=623, total=1319, "
                        "accuracy=47.23275208491281%, "
                        "partial_accuracy=49.96209249431388%, "
                        "format_accuracy=4.169825625473845%"
                    ),
                    (
                        "post-train: num_correct=731, total=1319, "
                        "accuracy=55.420773313116%, "
                        "partial_accuracy=57.9226686884003%, "
                        "format_accuracy=67.32373009855952%"
                    ),
                ]
            )
        )

        self.assertEqual(
            parsed,
            {
                "pre-train": {
                    "num_correct": 623,
                    "total": 1319,
                    "accuracy": 47.23275208491281,
                    "partial_accuracy": 49.96209249431388,
                    "format_accuracy": 4.169825625473845,
                },
                "post-train": {
                    "num_correct": 731,
                    "total": 1319,
                    "accuracy": 55.420773313116,
                    "partial_accuracy": 57.9226686884003,
                    "format_accuracy": 67.32373009855952,
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
