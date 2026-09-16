# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from typing import Optional
# from .parser import *
# now is fully compatible with the upstream
# tir script
# TODO(lei): remove this import once the
# upstream tir script is fully compatible
from tvm.script.parser.tir import *
from .tir import (
    prim_func,  # noqa: F401
)
from .tir.ir import *  # noqa: F401
from tilelang.layout import Layout, Fragment  # noqa: F401
from .proxy import (
    ptr,  # noqa: F401
    make_tensor,  # noqa: F401
    Buffer,  # noqa: F401
    Tensor,  # noqa: F401
    FragmentBuffer,  # noqa: F401
    SharedBuffer,  # noqa: F401
    LocalBuffer,  # noqa: F401
)
from .parallel import Parallel  # noqa: F401
from .pipeline import Pipelined  # noqa: F401
from .frame import has_let_value, get_let_value  # noqa: F401
from .kernel import (
    Kernel,  # noqa: F401
    KernelLaunchFrame,  # noqa: F401
    get_thread_binding,  # noqa: F401
    get_thread_bindings,  # noqa: F401
    get_block_binding,  # noqa: F401
    get_block_bindings,  # noqa: F401
)
from .allocate import (
    alloc_local,  # noqa: F401
    alloc_shared,  # noqa: F401
    alloc_fragment,  # noqa: F401
    alloc_var,  # noqa: F401
)
from .copy import copy, c2d_im2col  # noqa: F401
from .gemm import GemmWarpPolicy, gemm  # noqa: F401
from .fill import fill, clear  # noqa: F401
from .reduce import (
    reduce,  # noqa: F401
    reduce_max,  # noqa: F401
    reduce_min,  # noqa: F401
    reduce_sum,  # noqa: F401
    reduce_abssum,  # noqa: F401
)
from .print import print  # noqa: F401
from .customize import (
    atomic_add,  # noqa: F401
    atomic_addx2,  # noqa: F401
    dp4a,  # noqa: F401
    clamp,  # noqa: F401
    reshape,  # noqa: F401
    view,  # noqa: F401
    ppl_gemm,  # noqa: F401
    ppl_copy,  # noqa: F401
    ppl_fill,  # noqa: F401
    ppl_add,  # noqa: F401
    ppl_div,  # noqa: F401
    ppl_exp,  # noqa: F401
    ppl_max,  # noqa: F401
    ppl_mul,  # noqa: F401 
    ppl_mul_C,  # noqa: F401
    ppl_reduce_max,  # noqa: F401
    ppl_reduce_sum,  # noqa: F401
    ppl_subtract,  # noqa: F401
    ppl_rsqrt,  # noqa: F401
    ppl_add_C,  # noqa: F401
    ppl_gather,  # noqa: F401
    ppl_embedding,  # noqa: F401
    ppl_topk,  # noqa: F401
)
from .rvt import (  # noqa: F401
    rvt_call, rvt_kernel_start, rvt_set_max_cmd_id, rvt_parallel, rvt_fence, rvt_sync_i,
    rvt_sync_all, rvt_sync_all_gdma, rvt_sync_all_tiu, rvt_cfg_satu, rvt_cfg_lanemask,
    rvt_cfg_round_mode, rvt_fadd, rvt_fsub, rvt_fmul, rvt_fmac, rvt_dma_ld, rvt_dma_st, rvt_dma_cp,
)
from .builtin import *  # noqa: F401

from .memscope import *  # noqa: F401


def symbolic(name: str, dtype: str = "int32"):
    return tir.Var(name, dtype)


def use_swizzle(panel_size: int, order: str = "row", enable: bool = True):
    # If order is row, use rasterization2DRow, otherwise use rasterization2DColumn
    # The panel size is the number of threads in a warp
    # Use to improve the L2 Cache Locality
    device_func = ("rasterization2DRow" if order == "row" else "rasterization2DColumn")
    return attr(None, "threadblock_swizzle_pattern",
                f"tl::{device_func}<{panel_size}>") if enable else None


def annotate_layout(layout_map):
    # layout_map is a dictionary of buffer to layout
    layout_map = {buffer.data: layout for buffer, layout in layout_map.items()}
    return block_attr({"layout_map": layout_map})


def import_source(source: Optional[str] = None):
    # source is the source code to be imported
    return block_attr({"pragma_import_c": source}) if source is not None else None
