"""Host-stored original dictionary search with device-side FP32 screening.

Screening tensors stay on the selected device; the original uncentred FP32
dictionary stays on the host. Only selected original rows are transferred back
for the original FP64-distance calculation. C4 queries upload once per call.

No labels, discovery, fitting, image encoding, network, or GPU execution occurs
on import. The caller must first use the frozen context_operator.configure()
and unchanged CUBLAS_WORKSPACE_CONFIG / deterministic environment.

Numerical scope: the mean, centered bank, norm, chunk shapes, screening formula,
strict tie rules, and selected-distance arithmetic follow the as-run operators.
CUDA numerical equivalence still requires an actual same-device development
check. This is not an all-query exact-nearest-neighbor certificate: the original
centred FP32 screen can choose differently from direct FP64 at near ties.
"""
from __future__ import annotations

import warnings
import numpy as np
import torch


def _host_tensor_readonly(array: np.ndarray) -> torch.Tensor:
    # Read-only memmaps are safe here because no write through this tensor is
    # performed. CUDA .to() copies; CPU screening also does not mutate original.
    # Suppress only PyTorch's exact read-only *write* warning in this local scope.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable.*", category=UserWarning)
        return torch.from_numpy(array)


class HostOriginalSearch:
    """GeneralSearch-compatible values/indices with GPU-centred-bank residency.

    ``bank`` is [N,D] FP32 or float-convertible. Its row order is scientific state:
    exact screening ties retain the first row. With copy_host=False, the caller
    must keep the underlying bank unchanged for this object's lifetime. A strong
    reference is kept, so closing a caller variable does not release the array.
    A read-only FP32 contiguous memmap is supported without a full host copy.

    Construction computes the origin on ``device`` with the same full-bank
    float64 mean as the old implementation. This initialization cost and peak
    memory must be measured separately from subsequent resident requests.
    No CPU recomputation of the mean substitutes for the legacy CUDA reduction.

    Default chunks are the original 1024/32768. Changing them requires a new
    numerical check. The original dictionary is never retained on the GPU.
    """
    @torch.inference_mode()
    def __init__(self, bank, device="cuda", *, copy_host=False,
                 query_chunk=1024, reference_chunk=32768, validate_finite=True):
        b = np.asarray(bank)
        if b.ndim != 2 or len(b) == 0 or b.shape[1] == 0:
            raise ValueError("bank must be a nonempty [references, features] array")
        if copy_host:
            b = np.array(b, dtype=np.float32, order="C", copy=True)
        else:
            b = np.ascontiguousarray(b, dtype=np.float32)
        self.host_original = b
        self.d = int(b.shape[1])
        self.device = torch.device(device)
        self.query_chunk = int(query_chunk)
        self.reference_chunk = int(reference_chunk)
        if self.query_chunk < 1 or self.reference_chunk < 1:
            raise ValueError("chunk sizes must be positive")
        original = _host_tensor_readonly(b).to(self.device)
        if validate_finite:
            # Same finite-value condition, bounded transient check memory.
            for start in range(0, len(b), self.reference_chunk):
                if not bool(torch.isfinite(original[start:start+self.reference_chunk]).all()):
                    raise ValueError("nonfinite bank")
        self.origin = original.mean(0, dtype=torch.float64).float()
        self.centered = original - self.origin
        self.norm = self.centered.square().sum(1)
        del original

    @property
    def payload(self):
        """Actual array/tensor bytes only; excludes allocator/transient/RSS peak."""
        return {
            "host_original_bytes": int(self.host_original.nbytes),
            "device_centered_bytes": self.centered.numel()*self.centered.element_size(),
            "device_norm_bytes": self.norm.numel()*self.norm.element_size(),
            "device_origin_bytes": self.origin.numel()*self.origin.element_size(),
            "original_retained_on_device": False,
            "query_chunk": self.query_chunk,
            "reference_chunk": self.reference_chunk,
        }

    def _query_tensor(self, query):
        q = np.asarray(query)
        if q.ndim != 2 or len(q) == 0 or q.shape[1] != self.d:
            raise ValueError("query must be a nonempty [queries, bank_features] array")
        if not np.isfinite(q).all():
            raise ValueError("nonfinite query")
        # This is the one host-to-device query transfer for a GeneralSearch call.
        return torch.from_numpy(np.array(q, dtype=np.float32, order="C", copy=True)).to(self.device)

    @torch.inference_mode()
    def _match_tensor(self, q0):
        """q0 is a contiguous FP32 tensor on the prepared bank's device."""
        if q0.ndim != 2 or q0.shape[1] != self.d or q0.device != self.centered.device:
            raise ValueError("query tensor shape/device mismatch")
        centered = q0 - self.origin
        values = torch.empty(len(q0), device=q0.device)
        indices = torch.empty(len(q0), dtype=torch.int64, device=q0.device)
        for st in range(0, len(q0), self.query_chunk):
            a = centered[st:st+self.query_chunk]
            an = a.square().sum(1)
            best = torch.full((len(a),), float("inf"), device=q0.device)
            chosen = torch.full((len(a),), len(self.host_original), dtype=torch.int64, device=q0.device)
            for j in range(0, len(self.host_original), self.reference_chunk):
                stop = j+self.reference_chunk
                # Keep this expression and chunk shape aligned with the old
                # rule; do not replace by cdist, cosine, fp16, or a packed GEMM.
                cost = (an[:, None] + self.norm[None, j:stop]
                        - 2*(a@self.centered[j:stop].T)).clamp_min_(0)
                v, k = cost.min(1)
                take = v < best
                chosen = torch.where(take, k+j, chosen)
                best = torch.minimum(best, v)
            # Only nearest-row indices cross to the host, then selected FP32
            # original vectors cross to the device. No approximate reranking.
            host_indices = chosen.cpu().numpy()
            gathered = np.array(self.host_original[host_indices], dtype=np.float32, order="C", copy=True)
            selected = torch.from_numpy(gathered).to(q0.device)
            delta = q0[st:st+len(a)].double() - selected.double()
            # Preserve cast-then-divide, rather than direct float64 mean.
            values[st:st+len(a)] = delta.square().sum(1).float()/self.d
            indices[st:st+len(a)] = chosen
        return values.cpu().numpy(), indices.cpu().numpy()

    @torch.inference_mode()
    def __call__(self, query):
        return self._match_tensor(self._query_tensor(query))


class HostOriginalResponseBank(HostOriginalSearch):
    """PreparedResponseBank-compatible C4 response query, [N,4,384].

    Same physical reference row and common cyclic shift serve all four channels.
    Shift s is np.roll(query,-s,axis=1); first shift wins equal returned costs.
    This stores only one centred bank. It never expands four shifted banks.
    """
    def __init__(self, responses, device="cuda", **kwargs):
        a = np.asarray(responses)
        if a.ndim != 3 or a.shape[1:] != (4, 384) or not len(a):
            raise ValueError("responses must be nonempty [references,4,384]")
        super().__init__(a.reshape(-1, 1536), device, **kwargs)

    @torch.inference_mode()
    def __call__(self, query):
        q = np.asarray(query)
        if q.ndim != 3 or q.shape[1:] != (4, 384) or not len(q):
            raise ValueError("query must be nonempty [queries,4,384]")
        q0 = self._query_tensor(q.reshape(-1, 1536)).reshape(-1, 4, 384)
        aligned, ai = self._match_tensor(q0.reshape(-1, 1536))
        value, index = aligned.copy(), ai.copy()
        shift = np.zeros(len(q), np.uint8)
        for s in range(1, 4):
            qs = torch.roll(q0, shifts=-s, dims=1).contiguous().reshape(-1, 1536)
            v, i = self._match_tensor(qs)
            take = v < value
            value[take], index[take], shift[take] = v[take], i[take], s
        return {
            "aligned_cost": aligned,
            "aligned_reference": ai,
            "cyclic_cost": value,
            "cyclic_reference": index,
            "cyclic_shift": shift,
        }
