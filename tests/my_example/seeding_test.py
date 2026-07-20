"""Tests for experiment seed compatibility and overrides."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from my_example import seeding


class SeedingTest(unittest.TestCase):

    def test_unset_seed_preserves_legacy_dataset_seed(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(seeding.experiment_seed())
            self.assertEqual(seeding.dataset_shuffle_seed(), 42)

    def test_explicit_zero_matches_legacy_random_values(self):
        with mock.patch.dict(
            os.environ,
            {seeding.EXPERIMENT_SEED_ENV: "0"},
            clear=True,
        ):
            self.assertEqual(seeding.experiment_seed(), 0)
            self.assertEqual(seeding.dataset_shuffle_seed(), 42)

    def test_seed_offsets_dataset_shuffle(self):
        with mock.patch.dict(
            os.environ,
            {seeding.EXPERIMENT_SEED_ENV: "2"},
            clear=True,
        ):
            self.assertEqual(seeding.experiment_seed(), 2)
            self.assertEqual(seeding.dataset_shuffle_seed(), 44)

    def test_invalid_seed_fails_clearly(self):
        with mock.patch.dict(
            os.environ,
            {seeding.EXPERIMENT_SEED_ENV: "not-an-int"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                seeding.experiment_seed()


if __name__ == "__main__":
    unittest.main()
