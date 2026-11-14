"""Bind directional reading and matching profiles to validated reference states.

The field calculation is in pinned_directional_profiles. This adapter checks
profile, input and state identities; normal-state fitting and image acquisition
are handled by the caller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from pinned_directional_profiles import raw_field as _legacy_raw_field


@dataclass(frozen=True)
class Profile:
    profile_id: str
    legacy_profile: str
    bank_kind: str
    encoder_readings_per_view: int
    distance_queries_per_anchor: int
    mask_policy: str


PROFILES = {
    "N_W_U": Profile("N_W_U", "within_N01_fast", "component", 1, 1, "none"),
    "I_W_U": Profile("I_W_U", "within_I01", "component", 4, 4, "none"),
    "M_W_U": Profile("M_W_U", "within_M05", "mean", 4, 1, "none"),
    "N_W_F0": Profile("N_W_F0", "within_N01_fast_F0", "component", 1, 1, "turn0"),
    "I_W_FC4": Profile("I_W_FC4", "within_I01_FC4", "component", 4, 4, "per_turn_before_mean"),
    "M_W_FC4": Profile("M_W_FC4", "within_M05_FC4", "mean", 4, 1, "mean_mask_after_match"),
}


def profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except (KeyError, TypeError):
        raise ValueError(f"Unknown directional/control profile: {name!r}") from None


def registry() -> list[dict]:
    return [asdict(item) for item in PROFILES.values()]


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


@dataclass(frozen=True)
class StateBinding:
    """One actual acquisition slot. Row budget is per slot, never replicated."""

    slot_id: int
    bank_kind: str
    bank_path: str
    bank_sha256: str
    indices_path: str
    indices_sha256: str
    model_path: str
    model_sha256: str
    candidate_membership_digest: str
    retained_rows: int

    def validate_files(self) -> None:
        if self.bank_kind not in {"component", "mean"} or self.retained_rows < 1:
            raise ValueError("Invalid bank kind or row budget")
        for key in ("bank", "indices", "model"):
            if sha256(getattr(self, key + "_path")) != getattr(self, key + "_sha256"):
                raise ValueError(f"State file changed: {key}")
        bank = np.load(self.bank_path, mmap_mode="r", allow_pickle=False)
        indices = np.load(self.indices_path, allow_pickle=False)
        if bank.shape != (self.retained_rows, 384) or bank.dtype != np.float32:
            raise ValueError("Bank must have exact configured FP32[K,384] shape")
        if indices.shape != (self.retained_rows,) or indices.dtype.kind not in "iu":
            raise ValueError("Selected indices do not have exact configured K")
        if not np.isfinite(bank).all():
            raise ValueError("Nonfinite reference bank")
        with np.load(self.model_path, allow_pickle=False) as src:
            validate_model({name: src[name] for name in src.files})


def validate_shared_component(left: StateBinding, right: StateBinding) -> None:
    """Require N and I to share the literal selected vectors, indices and W."""
    if left.bank_kind != "component" or right.bank_kind != "component":
        raise ValueError("N and I require transformed four-direction component states")
    keys = ("slot_id", "bank_sha256", "indices_sha256", "model_sha256",
            "candidate_membership_digest", "retained_rows")
    if any(getattr(left, key) != getattr(right, key) for key in keys):
        raise ValueError("N/I state identity differs")


def validate_model(model: Mapping[str, np.ndarray]) -> None:
    if set(model) != {"matrix", "center"}:
        raise ValueError("Within-anchor model requires exactly matrix and center")
    if model["center"].shape != (384,) or model["matrix"].shape != (384, 384):
        raise ValueError("Within-anchor model shape differs")
    if not all(np.isfinite(item).all() for item in model.values()):
        raise ValueError("Nonfinite within-anchor model")


def raw_field(components: np.ndarray, profile_id: str, engine: Callable,
              model: Mapping[str, np.ndarray], masks: np.ndarray | None = None):
    """Return original FP32[32,32] field plus operator timings.

    ``engine`` is the original HostOriginalSearch bound to one validated state.
    It returns mean-square distance; unchanged multiplication by 192 yields
    half the squared Euclidean distance for 384-dimensional descriptors.
    Every component direction searches the same unrestricted component bank.
    No second normalization is performed here or by the pinned field operator.
    """
    item = profile(profile_id)
    x = np.asarray(components)
    allowed = ((32, 32, 4, 384),) if item.encoder_readings_per_view == 4 else (
        (32, 32, 4, 384), (32, 32, 384))
    if x.dtype != np.float32 or x.shape not in allowed or not np.isfinite(x).all():
        raise ValueError("Finite aligned FP32 query components are required")
    validate_model(model)
    if item.mask_policy != "none":
        if masks is None or masks.shape != (32, 32, 4) or masks.dtype != bool:
            raise ValueError("Aligned Boolean raw-token masks are required")
    return _legacy_raw_field(x, item.legacy_profile, engine, model, masks)


def encode_view(encoder, image: np.ndarray, profile_id: str, *,
                square_views: Callable, normalize_components: Callable,
                foreground_mask: Callable, device: str = "cuda", _torch_api=None):
    """Read real RGB once per required direction, with no feature/mask cache.

    Pass the pinned reader's square_views, normalize_components and foreground
    primitive explicitly. Masks use RAW unnormalized tokens before inverse
    grid alignment. Each profile call loads only its caller-owned model/state.
    Decode, search and final_fields are the outer request driver's operations.
    """
    # Generated-array tests can supply a tensor API double here.
    if _torch_api is None:
        import torch
    else:
        torch = _torch_api

    item = profile(profile_id)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Decoded uint8 RGB view is required")
    square = square_views(image[None])
    directions = range(item.encoder_readings_per_view)
    mu = torch.tensor([.485, .456, .406], device=device, dtype=torch.float32)[None, :, None, None]
    sd = torch.tensor([.229, .224, .225], device=device, dtype=torch.float32)[None, :, None, None]
    encoded, turn_masks = [], {}
    calls = pca_calls = 0
    for k in directions:
        turn = np.rot90(square, k, axes=(1, 2)).copy()
        x = torch.from_numpy(turn.transpose(0, 3, 1, 2).copy()).to(device).float() / 255
        x = (x - mu) / sd
        with torch.inference_mode(), torch.autocast(device_type=device, enabled=False):
            token = encoder.model.get_intermediate_layers(
                x, n=1, reshape=False, return_class_token=False, norm=True)[0]
        if tuple(token.shape) != (1, 1024, 384) or token.dtype != torch.float32:
            raise ValueError("Pinned S14 raw-token shape/dtype differs")
        raw = token.cpu().numpy()
        aligned = np.rot90(raw.reshape(1, 32, 32, 384), -k, axes=(1, 2)).copy()
        z, _ = normalize_components(aligned, "error")
        encoded.append(z[0])
        calls += 1
        if item.mask_policy != "none":
            mask = foreground_mask(raw[0], (32, 32)).reshape(32, 32)
            turn_masks[k] = np.ascontiguousarray(np.rot90(mask, -k))
            pca_calls += 1
    components = np.stack(encoded, axis=-2) if calls == 4 else encoded[0]
    masks = None
    if turn_masks:
        masks = np.stack([turn_masks.get(k, turn_masks[0]) for k in range(4)], axis=-1)
    return components, masks, dict(actual_encoder_calls=calls, actual_pca_calls=pca_calls,
                                   directions=list(directions), query_feature_cache=False,
                                   query_mask_cache=False)
