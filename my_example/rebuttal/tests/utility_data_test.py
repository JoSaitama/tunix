"""Server CPU tests for deterministic partitions and safe portable replay."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from my_example.rebuttal import utility_data as d


class UtilityDataTest(unittest.TestCase):
    def test_independent_prefixes_do_not_move_heldout_when_candidates_change(self):
        a, h = d.partition_indices(40, 24, 4, 8, 0)
        b, k = d.partition_indices(40, 24, 12, 12, 0)
        np.testing.assert_array_equal(a, b[:4])
        np.testing.assert_array_equal(h, k[:8])
        self.assertFalse(set(b) & set(k))
        self.assertTrue((b < 24).all() and (k >= 24).all())

    def test_fixed_partition_rejects_duplicates_leakage_and_bad_counts(self):
        good = {"candidate_indices": [0, 1, 2, 3], "heldout_indices": [8, 9, 10, 11]}
        d.validate_partition(good, 16, 8, 4, 4)
        for patch in ({"candidate_indices": [0, 0, 1, 2]}, {"heldout_indices": [4, 9, 10, 11]},
                      {"heldout_indices": [8, 9]}, {"candidate_indices": [False, 1, 2, 3]}):
            with self.assertRaises(ValueError):
                d.validate_partition(dict(good, **patch), 16, 8, 4, 4)

    def host(self):
        return {"prompt_ids": np.ones((16, 2), dtype=np.int32),
                "completion_ids": np.ones((16, 3), dtype=np.int32),
                "completion_mask": np.ones((16, 3), dtype=np.int32),
                "advantages": np.tile([-1., -.5, .5, 1.], 4).astype(np.float32),
                "clean_rewards": np.arange(16, dtype=np.float64),
                "observed_rewards": np.arange(16, dtype=np.float64),
                "correct": np.zeros(16, dtype=bool), "text": np.array(["x"] * 16),
                "answers": np.array(["1"] * 16), "noise_selected": np.zeros(4, dtype=bool),
                "noise_effective": np.zeros(4, dtype=bool), "rollout_seed": np.array(0),
                "source_indices": np.repeat(np.arange(4), 4),
                "prompt_ids_hash": np.repeat(np.array(["p0", "p1", "p2", "p3"]), 4),
                "old_per_token_logps": np.zeros((16, 3), dtype=np.float32),
                "ref_per_token_logps": np.zeros((16, 3), dtype=np.float32)}

    def test_npz_values_roundtrip_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.npz"
            np.savez_compressed(path, **self.host())
            before = d.sha256_file(path)
            result = d.load_rollout(path, list(range(4)))
            np.testing.assert_array_equal(result["completion_ids"], self.host()["completion_ids"])
            self.assertEqual(before, d.sha256_file(path))

    def test_npz_rejects_malformed_vectors_masks_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.npz"
            for patch in ({"completion_mask": np.full((16, 3), 2)},
                          {"advantages": np.zeros(15)}, {"clean_rewards": np.full(16, np.nan)},
                          {"source_indices": np.zeros(16)}, {"correct": np.zeros(16, dtype=np.int32)}):
                np.savez_compressed(path, **dict(self.host(), **patch))
                with self.assertRaises((ValueError, AssertionError)):
                    d.load_rollout(path, list(range(4)))

    def test_npz_rejects_pickle_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.npz"
            np.savez_compressed(path, **dict(self.host(), text=np.array([object()] * 16)))
            with self.assertRaises(ValueError):
                d.load_rollout(path, list(range(4)))
            link = Path(tmp) / "linked.npz"
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "symlink"):
                d.load_rollout(link, list(range(4)))

    def test_replay_contract_rejects_changed_config_and_production_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hashes = {}
            for name in d.PRODUCTION_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("source")
                hashes[name] = d.sha256_file(path)
            values = {key: 0 for key in d.REPLAY_CONTRACT}
            manifest = {"arguments": values, "audit_source_sha256": hashes}
            d.validate_replay_contract(manifest, SimpleNamespace(**values), root)
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                d.validate_replay_contract(manifest, SimpleNamespace(**dict(values, lora_rank=64)), root)
            (root / d.PRODUCTION_FILES[0]).write_text("different source")
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                d.validate_replay_contract(manifest, SimpleNamespace(**values), root)

    def test_replay_cannot_write_under_or_above_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "old"
            root.mkdir()
            manifest = {"arguments": {}, "data_partition": {}, "restored_lora": {"sha256": "fixture"}}
            (root / "audit_manifest.json").write_text(json.dumps(manifest))
            for target in (root, root / "new", root.parent):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    d.replay_metadata(root, target)
            self.assertEqual(d.replay_metadata(root, root.parent / "new")[0], root)


if __name__ == "__main__":
    unittest.main()
