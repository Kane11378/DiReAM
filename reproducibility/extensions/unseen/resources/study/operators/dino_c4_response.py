"""Whole-image DINOv2 token-response instrument.

This is neither the PDN RF65 operator nor native AnomalyDINO. The latter remains
an independent full-system baseline with its native short-edge geometry and
eight reference augmentations. Here an explicitly supplied view is square448,
then pixel-rotated by exact C4; the full-image token grid is rotated back before
retaining four384D responses. Transformer self-attention remains global.

Import is inert: no image discovery, data access, model loading, weights fetch,
labels, GPU execution, file writing, or global torch configuration occurs.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np
from PIL import Image
import torch

from host_original_search import HostOriginalSearch, HostOriginalResponseBank

HERE = Path(__file__).resolve().parent
BASELINES = HERE.parent / "baseline_sources"
DINO_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_ROOT = BASELINES / "dinov2" / ("dinov2-" + DINO_COMMIT)
WEIGHT_PATH = BASELINES / "weights/dinov2_vits14_pretrain.pth"
WEIGHT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
SIDE, PATCH, GRID, DIM, ROTATIONS = 448, 14, 32, 384, 4
ARMS = ("single", "normalaug", "independent", "aligned", "cyclic", "querymin")
COSINE_SCALE = np.float32(DIM / 2)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check_views(views):
    a = np.asarray(views)
    if a.ndim != 4 or a.shape[-1] != 3 or a.dtype != np.uint8 or min(a.shape[:3]) < 1:
        raise ValueError("explicit nonempty uint8 [V,H,W,3] real RGB views required")
    return a


def square_views(views):
    """Whole extent ->448square once, PIL BICUBIC, then no crop/pad/rotation resample."""
    vv = _check_views(views)
    return np.stack([np.asarray(Image.fromarray(im).resize((SIDE, SIDE), Image.Resampling.BICUBIC))
                     for im in vv]).astype(np.uint8, copy=False)


def normalize_components(features, zero_policy="error"):
    """Per384D component L2 norm in FP64, output FP32; zero is explicit.

    For ordinary nonzero components, squared L2/2 equals cosine distance up to
    FP32 normalization rounding. With zero_policy='keep', exact zeros stay zero:
    the distance is half squared Euclidean on unit-or-zero vectors, NOT ordinary
    cosine distance for a pair involving zero. No epsilon creates fake directions.
    """
    a = np.asarray(features)
    if a.shape[-1] != DIM or not np.isfinite(a).all():
        raise ValueError("finite384D feature components required")
    if zero_policy not in ("error", "keep"):
        raise ValueError("zero_policy must be 'error' or 'keep'")
    x = np.asarray(a, dtype=np.float32)
    norm = np.sqrt(np.square(x.astype(np.float64)).sum(-1, keepdims=True))
    zero = norm[..., 0] == 0
    if zero.any() and zero_policy == "error":
        raise ValueError(f"{int(zero.sum())} exact zero token components; cosine is undefined")
    out = np.divide(x.astype(np.float64), norm, out=np.zeros_like(norm*x, dtype=np.float64), where=norm > 0).astype(np.float32)
    return np.ascontiguousarray(out), int(zero.sum())


def verify_local_source(baselines=BASELINES):
    """Check fixed local source manifest + weight bytes, without importing a model."""
    base = Path(baselines)
    source_root = base / "dinov2" / ("dinov2-" + DINO_COMMIT)
    records = json.loads((base / "UPSTREAM_FILE_MANIFEST.json").read_text("utf-8-sig"))
    prefix = "dinov2/dinov2-" + DINO_COMMIT + "/"
    relevant = [r for r in records if r["path"].replace("\\", "/").startswith(prefix)]
    if not relevant:
        raise ValueError("DINOv2 source absent from fixed source manifest")
    for r in relevant:
        p = base / r["path"]
        if p.stat().st_size != r["bytes"] or sha256(p) != r["sha256"]:
            raise ValueError(f"pinned source differs: {r['path']}")
    wp = base / "weights/dinov2_vits14_pretrain.pth"
    if sha256(wp) != WEIGHT_SHA256:
        raise ValueError("DINOv2-S14 no-register weight bytes differ")
    return dict(code_commit=DINO_COMMIT, source_root=str(source_root),
                source_manifest_entries_checked=len(relevant), weight_path=str(wp),
                weight_sha256=WEIGHT_SHA256)


def load_fixed_model(device="cuda", baselines=BASELINES):
    """Only explicit invocation loads the pinned local no-reg model; never downloads."""
    identity = verify_local_source(baselines)
    source_root = Path(identity["source_root"]).resolve()
    if "dinov2" in sys.modules:
        present = Path(sys.modules["dinov2"].__file__).resolve()
        if not present.is_relative_to(source_root):
            raise RuntimeError(f"another dinov2 module already loaded from {present}; use an isolated process")
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    hub = importlib.import_module("dinov2.hub.backbones")
    if not Path(hub.__file__).resolve().is_relative_to(source_root):
        raise RuntimeError("pinned DINOv2 import location mismatch")
    model = hub.dinov2_vits14(pretrained=False)
    state = torch.load(identity["weight_path"], map_location="cpu", weights_only=True)
    status = model.load_state_dict(state, strict=True)
    del state
    if model.num_register_tokens != 0 or model.patch_size != PATCH or model.embed_dim != DIM:
        raise ValueError("requires frozen DINOv2 ViT-S/14,384D,no registers")
    model = model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    identity.update(strict_load=str(status), parameters=sum(p.numel() for p in model.parameters()),
                    model="DINOv2 ViT-S/14 no-register, full precision")
    return model, identity


class DinoC4Encoder:
    """Explicit[V,H,W,3] ->[V,32,32,4,384], exact square-image C4 geometry.

    Default batch_size=1 matches native one-image forward geometry/numerics.
    A caller-supplied model is supported for isolated geometry tests; it is marked
    unverified in provenance rather than claimed to be the official descriptor.
    """
    def __init__(self, device="cuda", *, model=None, baselines=BASELINES,
                 batch_size=1, zero_policy="error"):
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.zero_policy = zero_policy
        if self.batch_size < 1:
            raise ValueError("positive batch_size required")
        if model is None:
            model, self.identity = load_fixed_model(device, baselines)
        else:
            self.identity = dict(model="caller-supplied, source/weights unverified; testing or caller responsibility")
            model = model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        self.model = model
        self.protocol = dict(
            name="DINO-S14 square448 whole-image-token C4 instrument",
            source=self.identity, preprocessed_side=SIDE, token_grid=[GRID, GRID],
            component_dimension=DIM, responses=ROTATIONS,
            resize="complete HxW to448x448 PIL BICUBIC; aspect stretch if rectangular; no crop/pad",
            rotation="np.rot90(square,k,axes=(H,W)), CCW k=0,1,2,3, exact after resize",
            token_read="get_intermediate_layers(x,n=1,reshape=False,return_class_token=False,norm=True)[0]",
            token_alignment="row-major32x32 then np.rot90(grid,-k,axes=(row,col))",
            normalization="each384D component FP64 L2norm -> normalized FP32",
            zero_policy=zero_policy, batch_size=self.batch_size,
            pixel_normalization="uint8 /255 then ImageNet mean(.485,.456,.406)/std(.229,.224,.225)",
            transformer_context="whole image, global attention and rotated absolute positional embedding; not RF65",
            distances="GeneralSearch returned MSE *384/2 = mean component half squared L2; cosine only nonzero",
            label_access=False, native_AnomalyDINO=False,
        )
        self.last_encoding = None

    @torch.inference_mode()
    def encode_views(self, views):
        raw = _check_views(views)
        square = square_views(raw)
        output = np.empty((len(square), GRID, GRID, ROTATIONS, DIM), np.float32)
        mean = torch.tensor([.485, .456, .406], dtype=torch.float32, device=self.device)[None, :, None, None]
        std = torch.tensor([.229, .224, .225], dtype=torch.float32, device=self.device)[None, :, None, None]
        zero_counts = []
        for k in range(ROTATIONS):
            rotated = np.rot90(square, k, axes=(1, 2)).copy()
            blocks = []
            for start in range(0, len(square), self.batch_size):
                x = torch.from_numpy(rotated[start:start+self.batch_size].transpose(0, 3, 1, 2).copy()).to(self.device).float()/255
                x = (x-mean)/std
                with torch.autocast(device_type=self.device.type, enabled=False):
                    tokens = self.model.get_intermediate_layers(x, n=1, reshape=False,
                                                               return_class_token=False, norm=True)[0]
                expected = (len(x), GRID*GRID, DIM)
                if tuple(tokens.shape) != expected or tokens.dtype != torch.float32:
                    raise ValueError(f"expected FP32 {expected}, got {tuple(tokens.shape)} {tokens.dtype}")
                block = tokens.cpu().numpy().reshape(-1, GRID, GRID, DIM)
                blocks.append(np.rot90(block, -k, axes=(1, 2)).copy())
            aligned, nz = normalize_components(np.concatenate(blocks), self.zero_policy)
            output[:, :, :, k] = aligned
            zero_counts.append(nz)
        self.last_encoding = dict(input_shape=list(raw.shape), output_shape=list(output.shape),
                                  zero_token_counts_by_rotation=zero_counts, new_oracle_or_labels=False)
        return output


def _check_responses(array, *, zero_policy="error", unit_tolerance=1e-5):
    a = np.asarray(array)
    if a.ndim != 6 or tuple(a.shape[2:]) != (GRID, GRID, ROTATIONS, DIM) or min(a.shape[:2]) < 1 or a.dtype != np.float32:
        raise ValueError("responses must be nonempty FP32 [N,V,32,32,4,384]")
    zeros = 0
    # Bounded validation; avoid FP64 copies of a multi-GB complete response bank.
    for obj in range(len(a)):
        x = np.asarray(a[obj])
        if not np.isfinite(x).all():
            raise ValueError("nonfinite token responses")
        norm = np.sqrt(np.square(x.astype(np.float64)).sum(-1))
        z = norm == 0
        zeros += int(z.sum())
        if z.any() and zero_policy != "keep":
            raise ValueError("zero vectors require explicitly zero_policy='keep'")
        if np.any(np.abs(norm[~z]-1) > unit_tolerance):
            raise ValueError("token components are not unit normalized; normalize at encoding, not here")
    return a, zeros


@torch.inference_mode()
def match_responses(source, query, *, device="cuda", arms=ARMS, return_witness=True,
                    zero_policy="error", reference_ids=None, query_ids=None,
                    progress: Callable | None = None):
    """Six arms, same-slot source32square/full query32square, no fallback/smoothing.

    Inputs are already encoded normalized responses. Match all references in
    supplied order, at all1024 coordinates within the same real acquisition slot.
    This does not impose inter-camera alignment. The source/query geometry is
    identical for every arm; no duplicated views, ANN/coreset, or defect labels.
    Matched donors flatten object*1024+row*32+column. Independent donors further
    flatten physical_row*4+normal_rotation. C4 shift follows the old operator.
    """
    source, sz = _check_responses(source, zero_policy=zero_policy)
    query, qz = _check_responses(query, zero_policy=zero_policy)
    if source.shape[1] != query.shape[1]:
        raise ValueError("matching real-view slot count required")
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError("unique nonempty arms from " + repr(ARMS))
    if reference_ids is not None and (len(reference_ids) != len(source) or len(set(reference_ids)) != len(source)):
        raise ValueError("unique reference ids must match source rows")
    if query_ids is not None and (len(query_ids) != len(query) or len(set(query_ids)) != len(query)):
        raise ValueError("unique query ids must match query rows")
    if reference_ids is not None and query_ids is not None and set(reference_ids) & set(query_ids):
        raise ValueError("source and query identities overlap")
    n, views = query.shape[:2]
    maps = {a: np.empty((n, views, GRID, GRID), np.float32) for a in arms}
    witness = {}
    if return_witness:
        for key, tail, dtype in [("single_reference", (), np.int64),
                                 ("querymin_reference", (), np.int64),
                                 ("querymin_query_rotation", (), np.uint8),
                                 ("normalaug_reference", (), np.int64),
                                 ("independent_reference", (ROTATIONS,), np.int64),
                                 ("aligned_reference", (), np.int64),
                                 ("cyclic_reference", (), np.int64),
                                 ("cyclic_shift", (), np.uint8)]:
            arm = "querymin" if key.startswith("querymin") else "normalaug" if key.startswith("normalaug") else key.split("_")[0]
            if arm in arms:
                witness[key] = np.empty((n, views, GRID, GRID)+tail, dtype=dtype)
    groups = [g for g, need in [("single", {"single", "querymin"}),
                               ("independent", {"normalaug", "independent"}),
                               ("coherent", {"aligned", "cyclic"})] if set(arms) & need]
    for v in range(views):
        bank = np.ascontiguousarray(source[:, v]).reshape(-1, ROTATIONS, DIM)
        for group in groups:
            engine = (HostOriginalResponseBank(bank, device) if group == "coherent" else
                      HostOriginalSearch(bank[:, 0] if group == "single" else bank.reshape(-1, DIM), device))
            for i in range(n):
                z = np.ascontiguousarray(query[i, v]).reshape(-1, ROTATIONS, DIM)
                if group in ("single", "independent"):
                    # Query all four once. q0 is reused as single/normalaug;
                    # reductions retain the declared response identities.
                    values, ids = engine(z.reshape(-1, DIM))
                    values = values.reshape(-1, ROTATIONS)
                    ids = ids.reshape(-1, ROTATIONS)
                    if group == "single":
                        if "single" in arms:
                            maps["single"][i, v] = (values[:, 0]*COSINE_SCALE).reshape(GRID, GRID)
                            if return_witness: witness["single_reference"][i, v] = ids[:, 0].reshape(GRID, GRID)
                        if "querymin" in arms:
                            rotation = values.argmin(1)
                            chosen = np.arange(GRID*GRID)
                            maps["querymin"][i, v] = (values[chosen, rotation]*COSINE_SCALE).reshape(GRID, GRID)
                            if return_witness:
                                witness["querymin_reference"][i, v] = ids[chosen, rotation].reshape(GRID, GRID)
                                witness["querymin_query_rotation"][i, v] = rotation.astype(np.uint8).reshape(GRID, GRID)
                    else:
                        if "normalaug" in arms:
                            maps["normalaug"][i, v] = (values[:, 0]*COSINE_SCALE).reshape(GRID, GRID)
                            if return_witness: witness["normalaug_reference"][i, v] = ids[:, 0].reshape(GRID, GRID)
                        if "independent" in arms:
                            maps["independent"][i, v] = (values.mean(1, dtype=np.float64).astype(np.float32)*COSINE_SCALE).reshape(GRID, GRID)
                            if return_witness: witness["independent_reference"][i, v] = ids.reshape(GRID, GRID, ROTATIONS)
                else:
                    answer = engine(z)
                    for arm, cost, ref in [("aligned", "aligned_cost", "aligned_reference"),
                                           ("cyclic", "cyclic_cost", "cyclic_reference")]:
                        if arm in arms:
                            maps[arm][i, v] = (answer[cost]*COSINE_SCALE).reshape(GRID, GRID)
                            if return_witness: witness[arm+"_reference"][i, v] = answer[ref].reshape(GRID, GRID)
                    if "cyclic" in arms and return_witness:
                        witness["cyclic_shift"][i, v] = answer["cyclic_shift"].reshape(GRID, GRID)
            del engine
            if progress is not None: progress(dict(slot=v, operation=group, objects=n))
    for a in maps.values():
        if not np.isfinite(a).all() or float(a.min()) < -1e-7 or float(a.max()) > 2+1e-4:
            raise ValueError("nonfinite/out-of-range normalized token distance")
    return dict(maps=maps, witness=witness, protocol=dict(
        identity="developmental whole-image-token response operation; not PDN RF65 or native AnomalyDINO",
        arms=list(arms), source_shape=list(source.shape), query_shape=list(query.shape),
        reference_ids=None if reference_ids is None else list(reference_ids),
        query_ids=None if query_ids is None else list(query_ids),
        reference_domain="same physical camera slot, all source objects and all32x32 token coordinates",
        source_zero_components=sz, query_zero_components=qz, zero_policy=zero_policy,
        map_geometry="32square full-image token grid; no fallback, spatial smoothing, crop or semantic mask",
        scoring="half squared unit-vector L2; MSE384 or MSE1536 *192; C4 is mean over4 components",
        numeric="unchanged centeredFP32 screening, selected original FP64 sum castFP32/divide,1024/32768 chunks; first ties",
        module_sha256=sha256(__file__), search_sha256=sha256(HERE/"host_original_search.py"),
        labels_read=False))
