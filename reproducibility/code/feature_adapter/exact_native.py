"""Exact-budget native matching without normal-image rotation.

The original encoder, normalization, raw-token foreground policy and field
postprocessing are retained. Exact-K selection replaces the sampler loop count
while keeping its projection, distance order and tie behavior.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import importlib
import json
import sys
from typing import Any

import numpy as np

PATCHCORE_SAMPLER_SHA256 = "39612cd2b486865ece304348f740f4bc700a1c2e8684fbab3c2ed016c7801f2d"
CORESET_RUNTIME_SHA256 = "a6aec9ed1762da315d15dd8db6ae589fc37310910e9f7706bfa42cd04d9d7ed3"
COMPLETE_METHODS_SHA256 = "429080c9e3bab770582b606b9646c77e8d9f5938a46809fb3acaa28ecbf67001"
NATIVE_PROFILES = {
    "NativeNoRot_exactK_F": {"foreground": True, "slot_mode": "known_slot"},
    "NativeNoRot_exactK_slot_U": {"foreground": False, "slot_mode": "known_slot"},
    "NativeNoRot_exactK_pooled_U": {"foreground": False, "slot_mode": "pooled"},
    "NativeNoRot_full_F": {"foreground": True, "slot_mode": "known_slot"},
    "NativeNoRot_full_slot_U": {"foreground": False, "slot_mode": "known_slot"},
    "NativeNoRot_full_pooled_U": {"foreground": False, "slot_mode": "pooled"},
}


def sha(path: str | Path) -> str:
    with Path(path).open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def write_json_new(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(payload, out, ensure_ascii=False, indent=2)
        out.write("\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8-sig"))


def profile_spec(profile: str) -> dict[str, Any]:
    try:
        return dict(NATIVE_PROFILES[profile])
    except KeyError as exc:
        raise ValueError(f"Unregistered Native profile: {profile}") from exc


def validate_budget(candidate_rows: int, k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("Absolute K must be an integer")
    if not 0 < k <= candidate_rows:
        raise ValueError(f"Requested exact K={k} outside candidate pool N={candidate_rows}; no clipping")


def slot_budget_sum(k_by_slot: list[int], slot_count: int, mode: str) -> int:
    if mode not in ("known_slot", "pooled"):
        raise ValueError(mode)
    if len(k_by_slot) != slot_count:
        raise ValueError("Supply one original budget per actual acquisition slot")
    if any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0 for k in k_by_slot):
        raise ValueError("Every slot budget must be a positive integer")
    return int(sum(k_by_slot))


def exact_sampler_class(reader_code: Path, patchcore_root: Path):
    """Load pinned source; preserve the original projection and distance order."""
    import torch
    import tqdm

    sampler_path = patchcore_root / "src/patchcore/sampler.py"
    if sha(sampler_path) != PATCHCORE_SAMPLER_SHA256:
        raise ValueError("Pinned PatchCore sampler SHA256 mismatch")
    if sha(reader_code / "coreset_runtime.py") != CORESET_RUNTIME_SHA256:
        raise ValueError("Pinned exact-useful runtime SHA256 mismatch")
    sys.path.insert(0, str(patchcore_root / "src"))
    sys.path.insert(0, str(reader_code))
    from coreset_runtime import ExactUsefulSampler
    from patchcore.sampler import ApproximateGreedyCoresetSampler

    imported_sampler = Path(sys.modules[ApproximateGreedyCoresetSampler.__module__].__file__).resolve()
    if sha(imported_sampler) != PATCHCORE_SAMPLER_SHA256:
        raise ValueError("An already-imported PatchCore sampler has another identity")
    if sha(Path(sys.modules[ExactUsefulSampler.__module__].__file__)) != CORESET_RUNTIME_SHA256:
        raise ValueError("An already-imported coreset runtime has another identity")

    class AbsoluteKSampler(ExactUsefulSampler):
        def __init__(self, k, device):
            # The parent validates percentage in (0,1). It does not determine K.
            super().__init__(0.5, device, number_of_starting_points=10,
                             dimension_to_project_features_to=128)
            self.absolute_k = int(k)

        def _compute_greedy_coreset_indices(self, features):
            # Literal pinned upstream loop, with the sole count substitution
            # num_coreset_samples = self.absolute_k. Duplicate/tie behavior is
            # intentionally retained; there is no mask of previously chosen rows.
            number_of_starting_points = np.clip(self.number_of_starting_points, None, len(features))
            start_points = np.random.choice(len(features), number_of_starting_points, replace=False).tolist()
            self.start_points = list(start_points)
            approximate_distance_matrix = self._compute_batchwise_differences(features, features[start_points])
            approximate_coreset_anchor_distances = torch.mean(approximate_distance_matrix, axis=-1).reshape(-1, 1)
            coreset_indices = []
            num_coreset_samples = self.absolute_k
            with torch.no_grad():
                for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                    select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                    coreset_indices.append(select_idx)
                    coreset_select_distance = self._compute_batchwise_differences(features, features[select_idx:select_idx + 1])
                    approximate_coreset_anchor_distances = torch.cat(
                        [approximate_coreset_anchor_distances, coreset_select_distance], dim=-1)
                    approximate_coreset_anchor_distances = torch.min(
                        approximate_coreset_anchor_distances, dim=1).values.reshape(-1, 1)
            self.indices = np.array(coreset_indices)
            return self.indices

    return AbsoluteKSampler


def select_exact(source: np.ndarray, k: int, resources: dict, reader_code: str | Path,
                 device: str = "cuda") -> tuple[np.ndarray, np.ndarray, dict]:
    """Select from all supplied fit candidates, without ratios or deduplication.

CPU execution supports operator tests. Image inference uses the CUDA backend
and must not silently substitute a different computational device.
"""
    import torch

    source = np.asarray(source)
    if source.ndim != 2 or source.dtype != np.float32 or not np.isfinite(source).all():
        raise ValueError("Finite FP32 N-by-D candidate array required")
    validate_budget(len(source), k)
    cls = exact_sampler_class(Path(reader_code), Path(resources["upstream"]["patchcore"]))
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sampler = cls(k, torch.device(device))
    with torch.no_grad():
        bank = sampler.run(source)
    ix = np.asarray(sampler.indices, dtype=np.int64)
    bank = np.ascontiguousarray(bank, dtype=np.float32)
    if len(ix) != k or len(bank) != k or not np.array_equal(bank, source[ix]):
        raise RuntimeError("Exact-K/index-vector identity failed")
    return bank, ix, {
        "candidate_rows": len(source), "retained_rows": len(bank),
        "unique_selected_indices": len(np.unique(ix)), "duplicate_indices_preserved": True,
        "seed": 0, "projection_dim": 128, "starting_points": sampler.start_points,
        "sampler_source_sha256": PATCHCORE_SAMPLER_SHA256,
        "device": device, "selection_count_source": "absolute_integer_K",
    }


def validate_fit_ids(fit_rows: list[dict], calibration_rows: list[dict], query_rows: list[dict]) -> list[str]:
    ids = [r["global_identity"] for r in fit_rows]
    sets = [set(r["global_identity"] for r in rr) for rr in (fit_rows, calibration_rows, query_rows)]
    if any(len(ss) != len(rr) for ss, rr in zip(sets, (fit_rows, calibration_rows, query_rows))):
        raise ValueError("Duplicate identity within a role")
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Fit/calibration/query identities overlap")
    if not ids:
        raise ValueError("Empty fit role")
    return ids


class NativeNoRotExactRuntime:
    """Original Native kernels with explicit no-rotation bank topology.

    Constructor initializes exactly one original encoder. Candidate generation
    receives all fit identities and explicit calibration/query identities solely
    to reject leakage. Query predict(vv) performs one Native encoding per view.
    """
    def __init__(self, profile: str, reader_root: str | Path, base_reader: str | Path,
                 resources: dict, provenance: dict):
        self.spec = profile_spec(profile)
        self.profile = profile
        self.reader_root = Path(reader_root)
        self.resources = resources
        self.provenance = dict(provenance)
        required = {"weight_sha256", "native_encoder_buffer_sha256", "normalization", "native_geometry"}
        if not required <= set(provenance):
            raise ValueError(f"Missing normal feature provenance: {sorted(required - set(provenance))}")
        if sha(self.reader_root / "code/complete_methods.py") != COMPLETE_METHODS_SHA256:
            raise ValueError("Pinned Native source SHA256 mismatch")
        sys.path.insert(0, str(self.reader_root / "code"))
        self.cm = importlib.import_module("complete_methods")
        if Path(self.cm.__file__).resolve() != (self.reader_root / "code/complete_methods.py").resolve():
            raise ValueError("Imported complete_methods is not the explicitly selected reader")
        self.legacy = self.cm.Runtime("AnomalyDINO_F_norot", base_reader, resources)
        self.model = self.legacy.model
        self.engines = {}

    def build_candidates(self, fit_rows, calibration_rows, query_rows, image_source, destination):
        ids = validate_fit_ids(fit_rows, calibration_rows, query_rows)
        slots = len(fit_rows[0]["real_view_boxes"])
        if any(len(row["real_view_boxes"]) != slots for row in fit_rows):
            raise ValueError("Inconsistent actual acquisition slot count")
        if self.spec["foreground"] and slots != 1:
            raise ValueError('The registered masked Native profile is single-view VisA')
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        records = []
        for i, row in enumerate(fit_rows):
            # The source callback verifies each original RGB payload before decoding.
            vv = image_source(row) if callable(image_source) else self.cm.image(row, image_source)
            for slot, im in enumerate(vv):
                # Direct RGB; no augment_image, rot90, zero-angle warp or mask.
                features, grid = self.model._features(im)
                z = np.ascontiguousarray(self.model._normalize(features), dtype=np.float32)
                if z.ndim != 2 or z.shape[1] != 384 or not np.isfinite(z).all():
                    raise ValueError("Invalid Native feature candidates")
                filename = f"normal_{i:05d}_slot_{slot}.npy"
                np.save(destination / filename, z, allow_pickle=False)
                records.append({"file": filename, "slot_id": slot, "global_identity": ids[i],
                                "source_payload_sha256": row["source_payload_sha256"],
                                "shape": list(z.shape), "grid": list(grid), "sha256": sha(destination / filename)})
        payload = {"schema": 'diream-native-candidates-v1', "status": "COMPLETE",
                   "ordered_normal_ids": ids, "slots": slots, "normal_rotation": False,
                   "normal_augmentations_per_view": 1, "normal_input_path": "direct_unwarped_rgb",
                   "reference_foreground": False, "provenance": self.provenance, "records": records}
        write_json_new(destination / "NATIVE_CANDIDATES.json", payload)
        return payload

    def fit_from_candidates(self, candidates, destination, fit_rows, calibration_rows, query_rows,
                            k_by_slot: list[int] | None):
        import torch

        ids = validate_fit_ids(fit_rows, calibration_rows, query_rows)
        candidates = Path(candidates)
        spec_path = candidates / "NATIVE_CANDIDATES.json"
        cs = read_json(spec_path)
        if cs["status"] != "COMPLETE" or cs["ordered_normal_ids"] != ids or cs["provenance"] != self.provenance:
            raise ValueError("Candidate membership/provenance mismatch")
        if cs["normal_rotation"] is not False or cs["normal_augmentations_per_view"] != 1 or cs["reference_foreground"] is not False or cs["normal_input_path"] != "direct_unwarped_rgb":
            raise ValueError("Candidates are not direct no-normal-rotation Native")
        slots = cs["slots"]
        if slots != len(fit_rows[0]["real_view_boxes"]):
            raise ValueError("Candidate actual slot count does not match fit manifest")
        expected_pairs = [(r["global_identity"], slot) for r in fit_rows for slot in range(slots)]
        actual_pairs = [(r["global_identity"], r["slot_id"]) for r in cs["records"]]
        if actual_pairs != expected_pairs:
            raise ValueError("Candidate records are not exactly the ordered fit-by-actual-slot members")
        rgb_pins = {r["global_identity"]: r["source_payload_sha256"] for r in fit_rows}
        if any(r["source_payload_sha256"] != rgb_pins[r["global_identity"]] for r in cs["records"]):
            raise ValueError("Candidate RGB provenance differs from fit manifest")
        mode = self.spec["slot_mode"]
        is_full = "_full_" in self.profile
        if is_full != (k_by_slot is None):
            raise ValueError("Full profiles require no K; exact profiles require explicit per-slot K")
        if k_by_slot is not None:
            slot_budget_sum(k_by_slot, slots, mode)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        states = []
        group_slots = [[s] for s in range(slots)] if mode == "known_slot" else [list(range(slots))]
        for group_id, selected_slots in enumerate(group_slots):
            rr = [r for r in cs["records"] if r["slot_id"] in selected_slots]
            arrays = []
            for record in rr:
                path = candidates / record["file"]
                if sha(path) != record["sha256"]:
                    raise ValueError(f"Native shard changed: {path}")
                a = np.load(path, allow_pickle=False)
                if list(a.shape) != record["shape"] or a.dtype != np.float32:
                    raise ValueError("Candidate shape/dtype mismatch")
                arrays.append(a)
            source = np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32)
            if is_full:
                bank, ix = source, np.arange(len(source), dtype=np.int64)
                meta = {"candidate_rows": len(source), "retained_rows": len(source), "selection_count_source": "full_no_compression"}
            else:
                k = sum(k_by_slot[s] for s in selected_slots)
                bank, ix, meta = select_exact(source, k, self.resources, self.reader_root / "code", "cuda")
            bankfile = f"bank_{group_id}.npy"
            indexfile = f"indices_{group_id}.npy"
            np.save(destination / bankfile, bank, allow_pickle=False)
            np.save(destination / indexfile, ix, allow_pickle=False)
            states.append({"group_id": group_id, "slots": selected_slots, "bank": bankfile,
                           "indices": indexfile, "bank_sha256": sha(destination / bankfile),
                           "indices_sha256": sha(destination / indexfile), **meta,
                           "feature_bytes": int(bank.nbytes)})
        torch.cuda.synchronize()
        payload = {"schema": 'diream-native-state-v1', "status": "COMPLETE", "profile_id": self.profile,
                   "profile_spec": self.spec, "ordered_normal_ids": ids, "slots": slots,
                   "normal_rotation": False, "normal_augmentations_per_view": 1,
                   "normal_input_path": "direct_unwarped_rgb", "reference_foreground": False,
                   "query_readings_per_view": 1, "provenance": self.provenance,
                   "candidate_manifest": str(spec_path.resolve()), "candidate_manifest_sha256": sha(spec_path),
                   "states": states, "GT_reads": 0}
        write_json_new(destination / "NATIVE_STATE.json", payload)
        return payload

    def load(self, folder):
        from reader_baseline_adapters import TorchExactNN

        folder = Path(folder)
        state = read_json(folder / "NATIVE_STATE.json")
        if state["status"] != "COMPLETE" or state["profile_id"] != self.profile or state["profile_spec"] != self.spec or state["provenance"] != self.provenance:
            raise ValueError("Native state profile/provenance mismatch")
        self.engines = {}
        for row in state["states"]:
            path = folder / row["bank"]
            if sha(path) != row["bank_sha256"] or sha(folder / row["indices"]) != row["indices_sha256"]:
                raise ValueError("Native bank/index hash mismatch")
            bank = np.load(path, allow_pickle=False)
            if bank.shape != (row["retained_rows"], 384) or bank.dtype != np.float32:
                raise ValueError("Native bank shape/dtype mismatch")
            engine = TorchExactNN("cuda", q_chunk=1024, r_chunk=65536)
            engine.fit(bank)
            for slot in row["slots"]:
                self.engines[slot] = engine
        self.state = state

    def predict(self, vv):
        import cv2

        if len(vv) != self.state["slots"]:
            raise ValueError("Actual view count differs from bound state")
        native, maps, scores, patches, masks = [], [], [], [], []
        matching_vectors = search_calls = 0
        for slot, im in enumerate(vv):
            f, grid = self.model._features(im)
            if self.spec["foreground"]:
                np.random.seed(0)
                mask = self.model.wrapper.compute_background_mask(f, grid, threshold=10, masking_type=True)
            else:
                mask = np.ones(len(f), dtype=bool)
            masks.append(mask.copy())
            d = np.zeros(len(f), np.float64)
            if mask.any():
                d[mask] = self.engines[slot].run(1, self.model._normalize(f[mask]))[0][:, 0] / 2
                matching_vectors += int(mask.sum())
                search_calls += 1
            k = int(len(d) * .01)
            scores.append(float(d.max() if k == 0 else np.sort(d)[-k:].mean()))
            patches.append(d.reshape(grid))
            m = self.model.upstream_utils.dists2map(d.reshape(grid), im.shape).astype(np.float32)
            native.append(m)
            maps.append(cv2.resize(m, (256, 256), interpolation=cv2.INTER_LINEAR))
        return {"maps256": np.stack(maps), "native_maps": np.stack(native),
                "native_patch_maps": np.stack(patches), "object_score": np.array(max(scores), np.float64),
                "native_masks": np.stack(masks),
                "matching_query_vectors": np.array(matching_vectors, np.int64),
                "actual_search_invocations": np.array(search_calls, np.int32),
                "actual_readings_per_view": np.array(1, np.int32)}
