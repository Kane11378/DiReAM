"""Synthetic operator tests only: no image dataset, GT, AP or CUDA execution."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

# Operator dependencies are supplied by the selected Python environment.


RUN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUN / 'code/feature_adapter'))
from exact_native import (NativeNoRotExactRuntime, profile_spec, select_exact,
                          slot_budget_sum, validate_budget, validate_fit_ids)

READER_CODE = RUN/"code"
PATCHCORE = RUN/"third_party/PatchCore"
RESOURCES = {"upstream": {"patchcore": str(PATCHCORE)}}
torch.set_num_threads(2)


def choose(x, k):
    return select_exact(x, k, RESOURCES, READER_CODE, device="cpu")


class AbsoluteKTests(unittest.TestCase):
    def test_fixed_ratio_basepoint_exact_original_indices_and_vectors(self):
        x = np.random.default_rng(404).normal(size=(250, 384)).astype(np.float32)
        bank, indices, _ = choose(x, 10)
        from coreset_runtime import ExactUsefulSampler
        np.random.seed(0)
        torch.manual_seed(0)
        original = ExactUsefulSampler(.04, torch.device("cpu"), number_of_starting_points=10,
                                      dimension_to_project_features_to=128)
        with torch.no_grad():
            reference = original.run(x)
        self.assertTrue(np.array_equal(indices, original.indices))
        self.assertTrue(np.array_equal(bank, reference))

    def test_arbitrary_k_avoids_ratio_floor_error(self):
        self.assertEqual(int(100 * .29), 28)
        x = np.random.default_rng(505).normal(size=(100, 128)).astype(np.float32)
        bank, indices, meta = choose(x, 29)
        self.assertEqual(bank.shape, (29, 128))
        self.assertEqual(indices.shape, (29,))
        self.assertEqual(meta["selection_count_source"], "absolute_integer_K")

    def test_ties_and_duplicate_indices_are_preserved(self):
        x = np.zeros((20, 128), np.float32)
        bank, indices, meta = choose(x, 7)
        self.assertTrue(np.array_equal(indices, np.zeros(7, np.int64)))
        self.assertEqual(meta["unique_selected_indices"], 1)
        self.assertTrue(np.array_equal(bank, x[indices]))

    def test_seeded_projection_and_prefix_are_equal_for_synthetic_pool(self):
        x = np.random.default_rng(606).normal(size=(80, 384)).astype(np.float32)
        _, short, sm = choose(x, 5)
        _, long, lm = choose(x, 9)
        self.assertTrue(np.array_equal(short, long[:5]))
        self.assertEqual(sm["starting_points"], lm["starting_points"])
        # This test does NOT assert real CUDA/category prefix equivalence.

    def test_k_can_equal_pool_but_cannot_clip_or_coerce(self):
        x = np.zeros((1, 128), np.float32)
        bank, ix, _ = choose(x, 1)
        self.assertEqual(len(bank), 1)
        self.assertEqual(ix.tolist(), [0])
        for k in (0, -1, 2, 1.0, True):
            with self.assertRaises(ValueError):
                validate_budget(1, k)

    def test_invalid_candidate_values_fail(self):
        x = np.zeros((10, 128), np.float32)
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            choose(x, 2)

    def test_unknown_profile_and_five_slot_budget(self):
        with self.assertRaises(ValueError):
            profile_spec("AnomalyDINO_F_coreset01_mislabelled_norot")
        self.assertEqual(slot_budget_sum([7208] * 5, 5, "known_slot"), 36040)
        self.assertEqual(slot_budget_sum([7208] * 5, 5, "pooled"), 36040)
        with self.assertRaises(ValueError):
            slot_budget_sum([36040], 5, "known_slot")

    def test_candidate_roles_reject_leakage(self):
        f = [{"global_identity": "fit-1"}]
        c = [{"global_identity": "cal-1"}]
        q = [{"global_identity": "query-1"}]
        self.assertEqual(validate_fit_ids(f, c, q), ["fit-1"])
        with self.assertRaises(ValueError):
            validate_fit_ids(f, f, q)


class FakeNative:
    def __init__(self, foreground):
        self.reads = []
        self.mask_calls = 0
        self.foreground = foreground
        self.wrapper = self
        self.upstream_utils = self

    def _features(self, im):
        self.reads.append(im)
        return np.ones((4, 384), np.float32), (2, 2)

    def _normalize(self, features):
        return features / np.linalg.norm(features, axis=1, keepdims=True)

    def compute_background_mask(self, f, grid, threshold, masking_type):
        self.mask_calls += 1
        assert threshold == 10 and masking_type is True
        return np.full(4, self.foreground, bool)

    def dists2map(self, d, image_shape):
        return np.full(image_shape[:2], d.mean(), np.float32)


class FakeNN:
    def __init__(self):
        self.calls = 0

    def run(self, k, features):
        self.calls += 1
        assert k == 1
        return np.full((len(features), 1), 8., np.float32), np.zeros((len(features), 1), np.int64)


class NativePathTests(unittest.TestCase):
    def runtime(self, profile, slots, foreground=False):
        r = object.__new__(NativeNoRotExactRuntime)
        r.spec = profile_spec(profile)
        r.profile = profile
        r.model = FakeNative(foreground)
        r.state = {"slots": slots}
        shared_nn = FakeNN()
        r.engines = {s: shared_nn for s in range(slots)}
        return r

    def test_paddy_unmasked_once_per_actual_view_and_half_squared_distances(self):
        r = self.runtime("NativeNoRot_exactK_pooled_U", 5)
        vv = [np.full((7, 9, 3), s, np.uint8) for s in range(5)]
        z = r.predict(vv)
        self.assertEqual(len(r.model.reads), 5)
        self.assertTrue(all(a is b for a, b in zip(vv, r.model.reads)))
        self.assertEqual(r.model.mask_calls, 0)
        self.assertTrue(np.array_equal(z["native_patch_maps"], np.full((5, 2, 2), 4., np.float64)))
        self.assertEqual(float(z["object_score"]), 4.)
        self.assertEqual(int(z["actual_readings_per_view"]), 1)
        self.assertIs(r.engines[0], r.engines[4])

    def test_native_all_zero_foreground_stays_zero_without_nn_fallback(self):
        r = self.runtime("NativeNoRot_exactK_F", 1, foreground=False)
        z = r.predict([np.zeros((7, 9, 3), np.uint8)])
        self.assertEqual(r.model.mask_calls, 1)
        self.assertEqual(r.engines[0].calls, 0)
        self.assertEqual(float(z["object_score"]), 0.)
        self.assertFalse(z["maps256"].any())

    def test_candidate_builder_direct_input_and_no_reference_mask(self):
        r = self.runtime("NativeNoRot_exactK_F", 1, foreground=False)
        r.provenance = {"test_only": "synthetic"}
        im = np.zeros((7, 9, 3), np.uint8)
        r.cm = SimpleNamespace(image=lambda row, source: [im])
        f = [{"global_identity": "synthetic-fit-1", "real_view_boxes": [[0, 0, 9, 7]],
              "source_payload_sha256": "synthetic-test-not-a-source-pin"}]
        c = [{"global_identity": "synthetic-cal-1"}]
        q = [{"global_identity": "synthetic-query-1"}]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as parent:
            out = Path(parent) / "candidate"
            spec = r.build_candidates(f, c, q, "synthetic-no-file", out)
            self.assertIs(r.model.reads[0], im)
            self.assertEqual(r.model.mask_calls, 0)
            self.assertFalse(spec["normal_rotation"])
            self.assertFalse(spec["reference_foreground"])
            self.assertEqual(spec["normal_augmentations_per_view"], 1)
            self.assertEqual(spec["normal_input_path"], "direct_unwarped_rgb")
            self.assertEqual(spec["records"][0]["shape"], [4, 384])


if __name__ == "__main__":
    unittest.main(verbosity=2)
