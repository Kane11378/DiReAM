"""Exact forward-only runtime for pinned official approximate-greedy selection.

Only unused gradients and repeated immutable-A norms are eliminated. Original
projection/selection loop/distance order/argmax/tie behavior remain unchanged.
Upstream PatchCore Apache-2.0 source is an explicit separately pinned dependency.
See protocol/coreset_equivalence.json for actual whole-index/vector checks.
"""
import numpy as np
import torch
from patchcore.sampler import ApproximateGreedyCoresetSampler
class ExactUsefulSampler(ApproximateGreedyCoresetSampler):
 def _compute_batchwise_differences(self,matrix_a,matrix_b):
  key=(matrix_a.data_ptr(),tuple(matrix_a.shape),tuple(matrix_a.stride()),str(matrix_a.dtype),str(matrix_a.device),matrix_a._version)
  if getattr(self,'_constant_key',None)!=key:
   # Same literal bmm/reduction as upstream, computed once for immutable A.
   self._a_times_a=matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1,1);self._constant_a=matrix_a;self._constant_key=key;self.norm_computations=getattr(self,'norm_computations',0)+1
  a_times_a=self._a_times_a
  b_times_b=matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1,-1)
  a_times_b=matrix_a.mm(matrix_b.T)
  return (-2*a_times_b+a_times_a+b_times_b).clamp(0,None).sqrt()
 def _compute_greedy_coreset_indices(self,features):
  self.indices=super()._compute_greedy_coreset_indices(features);return self.indices
