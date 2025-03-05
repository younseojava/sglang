import ctypes
import os

import torch

if os.path.exists("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12"):
    ctypes.CDLL(
        "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12",
        mode=ctypes.RTLD_GLOBAL,
    )

from sgl_kernel.ops.activation import (
    apply_rope_with_cos_sin_cache_inplace,
    fused_add_rmsnorm,
    gelu_and_mul,
    gelu_tanh_and_mul,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    rmsnorm,
    silu_and_mul,
)
from sgl_kernel.ops.allreduce import *
from sgl_kernel.ops.attention import lightning_attention_decode
from sgl_kernel.ops.gemm import (
    bmm_fp8,
    cublas_grouped_gemm,
    fp8_blockwise_scaled_mm,
    fp8_scaled_mm,
    int8_scaled_mm,
    sgl_per_token_group_quant_fp8,
)
from sgl_kernel.ops.moe import moe_align_block_size
from sgl_kernel.ops.sampling import (
    min_p_sampling_from_probs,
    top_k_renorm_prob,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_prob,
    top_p_sampling_from_probs,
)
from sgl_kernel.ops.speculative import (
    build_tree_kernel,
    build_tree_kernel_efficient,
    tree_speculative_sampling_target_only,
)
from sgl_kernel.version import __version__
