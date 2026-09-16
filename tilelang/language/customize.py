# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

import tilelang.language as T
from tvm import ir
from tvm.tir import PrimExpr, Buffer, BufferRegion, BufferLoad
from typing import List, Union
from .copy import buffer_to_tile_region, buffer_region_to_tile_region, buffer_load_to_tile_region

_TPU_LOCAL_SCOPES = {"shared", "shared.dyn", "local", "local.fragment", "local.matrix"}
_TPU_BASE_FLOAT_DTYPES = {"float16", "bfloat16", "float32"}
_TPU_FP8_DTYPES = {"e4m3_float8", "e5m2_float8"}
_TPU_ELEMENTWISE_FLOAT_DTYPES = _TPU_BASE_FLOAT_DTYPES | _TPU_FP8_DTYPES
_TPUV7_EU_ELEMENTS = {
    "e4m3_float8": 64,
    "e5m2_float8": 64,
    "float16": 32,
    "bfloat16": 32,
    "float32": 16,
}
_TPUV7_DESCRIPTOR_DIM_MAX = 65535


def _require_buffer(name, value):
    if not isinstance(value, Buffer):
        raise TypeError(f"{name} must be a TIR Buffer, got {type(value).__name__}")


def _tpu_tensor_region(buffer, access_type):
    """Preserve logical Buffer metadata at the native TPU semantic boundary."""
    return buffer_to_tile_region(buffer, access_type)


def _static_positive_dim(operation, value):
    static_value = value if isinstance(value, int) else getattr(value, "value", None)
    if isinstance(static_value, bool) or not isinstance(static_value, int):
        raise ValueError(f"{operation} requires static integer dimensions, got {value}")
    if static_value <= 0:
        raise ValueError(f"{operation} requires positive dimensions, got {static_value}")
    return static_value


def _require_descriptor_shape(operation, value):
    """Validate one tensor shape before it becomes a TPUv7 ``dim4``."""
    rank = len(value.shape)
    if rank < 1 or rank > 4:
        raise ValueError(f"{operation} supports descriptor ranks 1 through 4, got rank {rank}")
    for axis, dim in enumerate(value.shape):
        extent = _static_positive_dim(f"{operation} axis {axis}", dim)
        if extent > _TPUV7_DESCRIPTOR_DIM_MAX:
            raise ValueError(f"{operation} axis {axis} extent {extent} exceeds the TPUv7 "
                             f"descriptor limit {_TPUV7_DESCRIPTOR_DIM_MAX}")


def _require_local_buffer(name, value, *, allow_matrix=False):
    _require_buffer(name, value)
    if value.scope() not in _TPU_LOCAL_SCOPES:
        raise ValueError(f"{name} must reside in TPU local memory, got scope={value.scope()!r}")
    if value.scope() == "local.matrix" and not allow_matrix:
        raise ValueError(f"{name} uses local.matrix storage, which is reserved for FP32 GEMM")
    _require_descriptor_shape(name, value)


def _require_global_buffer(name, value):
    _require_buffer(name, value)
    if value.scope() != "global":
        raise ValueError(f"{name} must reside in global memory, got scope={value.scope()!r}")


def _require_rank(name, value, rank):
    if len(value.shape) != rank:
        raise ValueError(f"{name} must be rank {rank}, got rank {len(value.shape)}")


def _require_dtype(name, value, supported):
    dtype = str(value.dtype)
    if dtype not in supported:
        expected = ", ".join(sorted(supported))
        raise ValueError(f"{name} dtype must be one of {{{expected}}}, got {dtype}")


def _require_same_dtype(operation, *buffers):
    dtypes = {str(buffer.dtype) for buffer in buffers}
    if len(dtypes) != 1:
        raise ValueError(f"{operation} requires matching buffer dtypes, got {sorted(dtypes)}")


def _require_same_shape(operation, lhs, rhs):
    try:
        ir.assert_structural_equal(lhs.shape, rhs.shape)
    except ValueError as error:
        raise ValueError(
            f"{operation} requires matching shapes, got {lhs.shape} and {rhs.shape}") from error


def _require_elementwise_shapes(operation, out, lhs, rhs):
    """Accept an equal RHS or the one supported W-broadcast form ``(M, 1)``."""
    _require_same_shape(operation, out, lhs)
    try:
        _require_same_shape(operation, out, rhs)
        return
    except ValueError:
        pass
    try:
        ir.assert_structural_equal(rhs.shape[0], out.shape[0])
        ir.assert_structural_equal(rhs.shape[1], 1)
    except ValueError as error:
        raise ValueError(f"{operation} RHS must match {out.shape} or use W-broadcast "
                         f"({out.shape[0]}, 1), got {rhs.shape}") from error


def _require_storage_disjoint(operation, lhs_name, lhs, rhs_name, rhs):
    if lhs.data.same_as(rhs.data):
        raise ValueError(f"{operation} requires {lhs_name} and {rhs_name} to use distinct storage")


def _require_distinct_storage(operation, **buffers):
    items = list(buffers.items())
    for index, (lhs_name, lhs) in enumerate(items):
        for rhs_name, rhs in items[index + 1:]:
            _require_storage_disjoint(operation, lhs_name, lhs, rhs_name, rhs)


def _require_exp_hw_limit(operation, value):
    """Enforce PPL's ``shape.h * shape.w <= 65535`` exp-family contract."""
    dims = [
        _static_positive_dim(f"{operation} axis {axis}", dim)
        for axis, dim in enumerate(value.shape)
    ]
    hw = dims[-1] if len(dims) < 4 else dims[-2] * dims[-1]
    if hw > _TPUV7_DESCRIPTOR_DIM_MAX:
        raise ValueError(f"{operation} requires descriptor h*w <= {_TPUV7_DESCRIPTOR_DIM_MAX}, "
                         f"got {hw}")


def atomic_add(dst: Buffer, value: PrimExpr) -> PrimExpr:
    """Perform an atomic addition operation.

    Args:
        dst (Buffer): Destination buffer where the atomic addition will be performed
        value (PrimExpr): Value to be atomically added

    Returns:
        PrimExpr: Handle to the atomic addition operation
    """
    return T.call_extern("handle", "AtomicAdd", T.address_of(dst), value)


def atomic_addx2(dst: Buffer, value: PrimExpr) -> PrimExpr:
    """Perform an atomic addition operation with double-width operands.

    Args:
        dst (Buffer): Destination buffer where the atomic addition will be performed
        value (PrimExpr): Value to be atomically added (double-width)

    Returns:
        PrimExpr: Handle to the double-width atomic addition operation
    """
    return T.call_extern("handle", "AtomicAddx2", T.address_of(dst), T.address_of(value))


def dp4a(A: Buffer, B: Buffer, C: Buffer) -> PrimExpr:
    """Perform a 4-element dot product with accumulation (DP4A).

    Args:
        A (Buffer): First input buffer
        B (Buffer): Second input buffer
        C (Buffer): Accumulation buffer

    Returns:
        PrimExpr: Handle to the DP4A operation
    """
    return T.call_extern("handle", "DP4A", T.address_of(A), T.address_of(B), T.address_of(C))


def clamp(dst: PrimExpr, min_val: PrimExpr, max_val: PrimExpr) -> PrimExpr:
    """Clamps the input value dst between [min_val, max_val]
    
    Args:
        dst: Input value to be clamped
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Value clamped to the specified range
    """
    dst = T.max(dst, min_val)  # Ensure value is not less than minimum
    dst = T.min(dst, max_val)  # Ensure value is not greater than maximum
    return dst


def reshape(src: Buffer, shape: List[PrimExpr]) -> Buffer:
    """Reshapes the input buffer to the specified shape.
    
    Args:
        src (Buffer): Input buffer to be reshaped
        shape (List[PrimExpr]): New shape for the buffer

    Returns:
        Buffer: A new buffer view with the specified shape
    """
    return T.Tensor(shape, src.dtype, src.data)


def view(src: Buffer,
         shape: Union[List[PrimExpr], None] = None,
         dtype: Union[str, None] = None) -> Buffer:
    """Views the input buffer with optionally modified shape and dtype.
    
    Args:
        src (Buffer): Input buffer to be viewed
        shape (Union[List[PrimExpr], None], optional): New shape for the buffer. Defaults to None.
        dtype (Union[str, None], optional): New dtype for the buffer. Defaults to None.

    Returns:
        Buffer: A new buffer view with the specified shape and dtype
    """
    if shape is None:
        shape = src.shape
    if dtype is None:
        dtype = src.dtype
    return T.Tensor(shape, dtype, src.data)


def ppl_gemm(A, B, C, transpose_A=False, transpose_B=False, *, accumulate):
    """Launch a TPU GEMM on local/shared tiles.

    Args:
        A: Left-hand input tile.
        B: Right-hand input tile.
        C: Output/accumulation tile with shape `(M, N)`.
        transpose_A: Whether `A` should be treated as transposed.
            This is supported only by the native FP32 matrix path.
        transpose_B: Whether `B` should be treated as transposed.
        accumulate: Whether to compute `C += A @ B` instead of overwriting C.
            This keyword is mandatory so the read/write contract of `C` never
            depends on a backend-specific instruction default.

    Returns:
        PrimExpr: Handle to the emitted GEMM extern call.

    Example:
        `T.ppl_gemm(Q_shared, K_shared, acc_s, transpose_B=True, accumulate=False)`

    Notes:
        `K` is inferred from `A` and `B`, and must match.
        The backend contract carries accumulation explicitly. Programming
        model-specific instruction availability is validated after target
        selection; for example, RV Tensor supports an accumulating
        FP16/BF16 right-transpose form while TPU-Kernel does not.
        TPU-Kernel FP8 uses the separately validated
        ``tpu_bdc_fp8_mm_R_trans`` form, whose explicit ``result_add`` flag
        supports accumulation.
        Native FP32 supports NN and TN (`transpose_A=True`) forms. It does not
        support `transpose_B=True`. Lower-precision inputs support NN and NT
        (`transpose_B=True`) forms; TN is not exposed by their instruction
        families.

    Dtype support:
        - TPU-Kernel and RV Tensor: matching FP32 inputs with FP32 output;
          matching FP16/BF16 inputs with either same-dtype output in overwrite
          mode or FP32 output; matching E4M3/E5M2 inputs with FP32 output.
        - FP32 uses ``local.matrix`` and supports NN/TN. Lower-precision paths
          use regular local storage and support NN/NT.
        - TPU-Kernel FP16/BF16 NT cannot accumulate. RV Tensor FP16/BF16 NT
          and both FP8 NT paths can overwrite or accumulate.
    """
    for name, buffer in (("A", A), ("B", B), ("C", C)):
        _require_local_buffer(name, buffer, allow_matrix=True)
        _require_rank(name, buffer, 2)
    _require_same_dtype("ppl_gemm inputs", A, B)
    _require_storage_disjoint("ppl_gemm", "C", C, "A", A)
    _require_storage_disjoint("ppl_gemm", "C", C, "B", B)
    input_dtype = str(A.dtype)
    if input_dtype not in _TPU_BASE_FLOAT_DTYPES | _TPU_FP8_DTYPES:
        raise ValueError("ppl_gemm inputs must use float32, float16, bfloat16, or FP8; "
                         f"got {A.dtype}")
    if not isinstance(transpose_A, bool) or not isinstance(transpose_B, bool):
        raise TypeError("ppl_gemm transpose_A and transpose_B must be Python bools")
    if not isinstance(accumulate, bool):
        raise TypeError("ppl_gemm accumulate must be a Python bool")
    if transpose_A and input_dtype != "float32":
        raise ValueError("ppl_gemm transpose_A=True is supported only for float32 inputs")
    if transpose_A and transpose_B:
        raise ValueError("ppl_gemm cannot transpose both inputs")
    if input_dtype == "float32" or input_dtype in _TPU_FP8_DTYPES:
        if str(C.dtype) != "float32":
            raise ValueError("ppl_gemm FP32/FP8 inputs require a float32 output/accumulator")
    elif str(C.dtype) != "float32" and not (not accumulate and str(C.dtype) == input_dtype):
        raise ValueError("ppl_gemm requires a float32 C tile when accumulate=True; "
                         "overwrite mode also permits C to match the input dtype")
    scopes = {buffer.scope() for buffer in (A, B, C)}
    if input_dtype == "float32":
        if scopes != {"local.matrix"}:
            raise ValueError("ppl_gemm FP32 operands must use local.matrix storage")
    elif "local.matrix" in scopes:
        raise ValueError("ppl_gemm local.matrix storage is reserved for FP32 operands")
    Aptr = _tpu_tensor_region(A, "r")
    Bptr = _tpu_tensor_region(B, "r")
    Cptr = _tpu_tensor_region(C, "rw" if accumulate else "w")
    M = C.shape[0]
    N = C.shape[1]
    K = A.shape[0] if transpose_A else A.shape[1]
    K_B = B.shape[1] if transpose_B else B.shape[0]
    try:
        ir.assert_structural_equal(K, K_B)
    except ValueError as error:
        raise ValueError(f"ppl_gemm K mismatch: A gives {K}, B gives {K_B}") from error
    return T.call_extern("handle", "tl.tpu.gemm", Aptr, Bptr, Cptr, transpose_A, transpose_B, M, N,
                         K, accumulate)


def ppl_copy(
    src,
    dst,
):
    """Copy a tile/region between global and local memory, with optional cast.

    Args:
        src: Source buffer, `BufferRegion`, or `BufferLoad`.
        dst: Destination buffer, `BufferRegion`, or `BufferLoad`.

    Returns:
        PrimExpr: Handle to the emitted copy extern call.

    Example:
        `T.ppl_copy(X[by * block_M, 0], X_shared)`
        `T.ppl_copy(A_shared_fp32, A_shared)`

    Notes:
        This op is commonly used for global-to-shared loads, shared-to-global
        stores, and shared-to-shared copies between temporary tiles.
        When source and destination dtypes differ, this op can also be used as
        a convenient copy-and-convert step.
        The most common TPU usage is copying 2D tiles or simple row/column
        slices.

    Dtype support:
        - TPU-Kernel and RV Tensor same-dtype transport: E4M3, E5M2, FP16,
          BF16, FP32, INT8/16/32, and UINT8/16/32.
        - Local-to-local float conversion: any pair among FP16, BF16, and
          FP32, plus either FP8 format to or from one of those base formats.
          Direct E4M3-to-E5M2 conversion is not exposed.
        - Dtype conversion is local-only. Global DMA copies preserve dtype.
          FP32 ``local.matrix`` copies are global-to-local or local-to-global.
    """

    def _is_one(value):
        return isinstance(value, int) and value == 1 or (hasattr(value, "value") and
                                                         value.value == 1)

    def _merge_extent(src_value, dst_value):
        if _is_one(src_value):
            return dst_value
        if _is_one(dst_value):
            return src_value
        ir.assert_structural_equal(src_value, dst_value)
        return src_value

    def get_extent(data):
        if isinstance(data, Buffer):
            return data.shape
        elif isinstance(data, BufferRegion):
            return [x.extent for x in data.region]
        elif isinstance(data, BufferLoad):
            return [getattr(index, "lanes", 1) for index in data.indices]
        else:
            return None

    supported_operands = (Buffer, BufferRegion, BufferLoad)
    if not isinstance(src, supported_operands):
        raise TypeError("ppl_copy src must be a Buffer, BufferRegion, or BufferLoad, "
                        f"got {type(src).__name__}")
    if not isinstance(dst, supported_operands):
        raise TypeError("ppl_copy dst must be a Buffer, BufferRegion, or BufferLoad, "
                        f"got {type(dst).__name__}")

    src_extent = list(get_extent(src))
    dst_extent = list(get_extent(dst))

    def _to_dim4(values):
        if len(values) == 1:
            return [1, 1, 1, values[0]]
        if len(values) == 2:
            return [1, values[0], 1, values[1]]
        if len(values) == 3:
            return [values[0], values[1], 1, values[2]]
        if len(values) == 4:
            return values
        raise ValueError(f"ppl_copy supports ranks 1 through 4, got rank {len(values)}")

    def _from_dim4(values, rank):
        if rank == 1:
            return [values[3]]
        if rank == 2:
            return [values[1], values[3]]
        if rank == 3:
            return [values[0], values[1], values[3]]
        if rank == 4:
            return values
        raise AssertionError(f"unexpected validated rank {rank}")

    merged_dim4 = [
        _merge_extent(src_value, dst_value) for src_value, dst_value in zip(  # noqa: B905
            _to_dim4(src_extent), _to_dim4(dst_extent))
    ]
    src_region_extent = _from_dim4(merged_dim4, len(src_extent))
    dst_region_extent = _from_dim4(merged_dim4, len(dst_extent))

    def _to_region(data, access_type, region_extent):
        if isinstance(data, Buffer):
            return buffer_to_tile_region(data, access_type)
        elif isinstance(data, BufferRegion):
            return buffer_region_to_tile_region(data, access_type)
        else:
            return buffer_load_to_tile_region(data, access_type, region_extent)

    src = _to_region(src, "r", src_region_extent)
    dst = _to_region(dst, "w", dst_region_extent)
    return T.call_extern("handle", "tl.tpu.copy", src, dst)


def ppl_fill(buffer, value):
    """Fill a local/shared tile with a scalar constant.

    Args:
        buffer: Destination tile to be written.
        value: Scalar literal to broadcast to every element.

    Returns:
        PrimExpr: Handle to the emitted fill extern call.

    Example:
        `T.ppl_fill(C_shared, T.float32(0))`
        `T.ppl_fill(scores_max, -T.infinity(accum_dtype))`

    Notes:
        This is typically used to initialize accumulation buffers, masks,
        or temporary outputs before later elementwise or reduction ops.

    Dtype support:
        - TPU-Kernel: E4M3, E5M2, FP16, BF16, and FP32.
        - RV Tensor: E4M3, E5M2, FP16, BF16, and FP32.
    """
    _require_local_buffer("buffer", buffer)
    _require_dtype("buffer", buffer, _TPU_ELEMENTWISE_FLOAT_DTYPES)
    buffer = _tpu_tensor_region(buffer, "w")
    return T.call_extern("handle", "tl.tpu.fill", buffer, value)


def ppl_subtract(out, inp1, inp2):
    """Compute elementwise subtraction `out = inp1 - inp2`.

    Args:
        out: Output tile.
        inp1: Left-hand input tile.
        inp2: Right-hand input tile.

    Returns:
        PrimExpr: Handle to the emitted subtraction extern call.

    Example:
        `T.ppl_subtract(scores_scale, scores_max_prev, scores_max)`

    Notes:
        The usual usage is that all tiles have the same shape.
        A limited broadcast-style usage is also supported in common cases when
        the second input has shape `(M, 1)`.

    Dtype support:
        - TPU-Kernel: E4M3, E5M2, FP16, BF16, and FP32.
        - RV Tensor: E4M3, E5M2, FP16, BF16, and FP32.
    """
    for name, buffer in (("out", out), ("inp1", inp1), ("inp2", inp2)):
        _require_local_buffer(name, buffer)
        _require_rank(name, buffer, 2)
        _require_dtype(name, buffer, _TPU_ELEMENTWISE_FLOAT_DTYPES)
    _require_same_dtype("ppl_subtract", out, inp1, inp2)
    _require_elementwise_shapes("ppl_subtract", out, inp1, inp2)
    outptr = _tpu_tensor_region(out, "w")
    inpptr1 = _tpu_tensor_region(inp1, "r")
    inpptr2 = _tpu_tensor_region(inp2, "r")
    return T.call_extern("handle", "tl.tpu.sub", outptr, inpptr1, inpptr2)


def ppl_mul_C(out, inp1, value):
    """Compute elementwise scalar multiply `out = inp1 * value`.

    Args:
        out: Output tile.
        inp1: Input tile.
        value: Scalar multiplier.

    Returns:
        PrimExpr: Handle to the emitted multiply-by-constant extern call.

    Example:
        `T.ppl_mul_C(scores_scale, scores_scale, scale)`
        `T.ppl_mul_C(x_neg, in_x, T.float32(-1.0))`

    Notes:
        This is commonly used for scaling, sign flip, and normalization-style
        updates on a local tile.

    Dtype support:
        - TPU-Kernel: E4M3, E5M2, FP16, BF16, and FP32.
        - RV Tensor: E4M3, E5M2, FP16, BF16, and FP32.
    """
    for name, buffer in (("out", out), ("inp1", inp1)):
        _require_local_buffer(name, buffer)
        _require_dtype(name, buffer, _TPU_ELEMENTWISE_FLOAT_DTYPES)
    _require_same_dtype("ppl_mul_C", out, inp1)
    _require_same_shape("ppl_mul_C", out, inp1)
    outptr = _tpu_tensor_region(out, "w")
    inpptr1 = _tpu_tensor_region(inp1, "r")
    return T.call_extern("handle", "tl.tpu.mul_scalar", outptr, inpptr1, value)


def ppl_mul(out, inp1, inp2):
    """Compute elementwise multiplication `out = inp1 * inp2`.

    Args:
        out: Output tile.
        inp1: Left-hand input tile.
        inp2: Right-hand input tile.

    Returns:
        PrimExpr: Handle to the emitted multiply extern call.

    Example:
        `T.ppl_mul(A_pow2, A_shared, A_shared)`
        `T.ppl_mul(out, right, x_neg_exp_1_div)`

    Notes:
        The usual usage is that all tiles have the same shape.
        A limited broadcast-style usage is also supported in common cases when
        the second input has shape `(M, 1)`.

    Dtype support:
        - TPU-Kernel: E4M3, E5M2, FP16, BF16, and FP32.
        - RV Tensor: E4M3, E5M2, FP16, BF16, and FP32.
    """
    for name, buffer in (("out", out), ("inp1", inp1), ("inp2", inp2)):
        _require_local_buffer(name, buffer)
        _require_rank(name, buffer, 2)
        _require_dtype(name, buffer, _TPU_ELEMENTWISE_FLOAT_DTYPES)
    _require_same_dtype("ppl_mul", out, inp1, inp2)
    _require_elementwise_shapes("ppl_mul", out, inp1, inp2)
    outptr = _tpu_tensor_region(out, "w")
    inpptr1 = _tpu_tensor_region(inp1, "r")
    inpptr2 = _tpu_tensor_region(inp2, "r")
    return T.call_extern("handle", "tl.tpu.mul", outptr, inpptr1, inpptr2)


def ppl_max(out, inp1, inp2):
    """Compute elementwise maximum ``out = max(inp1, inp2)``.

    The portable semantic maps to ``tpu_bdc_max`` on TPU-Kernel and
    ``rvt_fmax`` on RV Tensor.  It supports equal rank-2 tiles and the same
    W-dimension broadcast form as the other portable elementwise operations.
    The TPU-Kernel selector uses the generic ``tpu_bdc_max`` entry point whose
    dtype argument distinguishes E4M3 and E5M2.

    Dtype support:
        - TPU-Kernel: E4M3, E5M2, FP16, BF16, and FP32.
        - RV Tensor: E4M3, E5M2, FP16, BF16, and FP32.
    """
    for name, buffer in (("out", out), ("inp1", inp1), ("inp2", inp2)):
        _require_local_buffer(name, buffer)
        _require_rank(name, buffer, 2)
        _require_dtype(name, buffer, _TPU_ELEMENTWISE_FLOAT_DTYPES)
    _require_same_dtype("ppl_max", out, inp1, inp2)
    _require_elementwise_shapes("ppl_max", out, inp1, inp2)
    outptr = _tpu_tensor_region(out, "w")
    inpptr1 = _tpu_tensor_region(inp1, "r")
    inpptr2 = _tpu_tensor_region(inp2, "r")
    return T.call_extern("handle", "tl.tpu.max", outptr, inpptr1, inpptr2)


@T.macro
def _exp_impl(out, work0, work1, coeff):
    T.call_extern("handle", "tl.tpu.exp", _tpu_tensor_region(out, "rw"),
                  _tpu_tensor_region(work0, "rw"), _tpu_tensor_region(work1, "rw"),
                  _tpu_tensor_region(coeff, "rw"))


def ppl_exp(out, work0, work1, coeff):
    """Compute `exp(out)` in place.

    Args:
        out: Input/output tile. The result overwrites this buffer.
        work0: Scratch tile with the same shape as `out`.
        work1: Scratch tile with the same shape as `out`.
        coeff: Coefficient buffer initialized by the TPU-Kernel API.

    Example:
        `T.ppl_exp(scores_scale, work0, work1, coeff)`

    Notes:
        This computes natural exponential `exp(x)`.  It uses the PPL 1.7
        `tpu_bdc_load_fp_exp_coeff` and `tpu_bdc_fp_exp` coefficient-buffer
        contract on TPU-Kernel. RV currently accepts FP32 only, uses a range-
        reduced polynomial, and leaves coeff unused. Subnormal relative
        accuracy is not guaranteed on RV (the device may flush to zero).

    Dtype support:
        - TPU-Kernel: FP16, BF16, and FP32.
        - RV Tensor: FP32 only.
    """
    for name, buffer in (("out", out), ("work0", work0), ("work1", work1), ("coeff", coeff)):
        _require_local_buffer(name, buffer)
        _require_dtype(name, buffer, _TPU_BASE_FLOAT_DTYPES)
    _require_same_dtype("ppl_exp", out, work0, work1, coeff)
    _require_same_shape("ppl_exp", out, work0)
    _require_same_shape("ppl_exp", out, work1)
    _require_distinct_storage("ppl_exp", out=out, work0=work0, work1=work1, coeff=coeff)
    _require_exp_hw_limit("ppl_exp", out)
    _require_rank("coeff", coeff, 2)
    try:
        ir.assert_structural_equal(coeff.shape[0], 64)
        ir.assert_structural_equal(coeff.shape[1], 32)
    except ValueError as error:
        raise ValueError(f"ppl_exp expects coeff shape (64, 32), got {coeff.shape}") from error
    return _exp_impl(out, work0, work1, coeff)


def _prepare_row_gather(operation, output, param, index, param_h):
    """Validate a global row-gather contract and return semantic operands."""
    for name, buffer in (("output", output), ("param", param), ("index", index)):
        _require_global_buffer(name, buffer)
    _require_rank("output", output, 2)
    _require_rank("param", param, 2)
    _require_rank("index", index, 2)
    for name, buffer in (("output", output), ("param", param), ("index", index)):
        _require_descriptor_shape(f"{operation} {name}", buffer)
    _require_distinct_storage(operation, output=output, param=param, index=index)
    _require_same_dtype(f"{operation} payload", output, param)
    _require_dtype("output", output, _TPU_ELEMENTWISE_FLOAT_DTYPES)
    if str(index.dtype) != "uint32":
        raise ValueError(f"{operation} index dtype must be uint32, got {index.dtype}")
    if not isinstance(param_h, int) or param_h <= 0:
        raise ValueError(f"{operation} param_h must be a positive Python integer")
    try:
        ir.assert_structural_equal(param.shape[0], param_h)
        ir.assert_structural_equal(output.shape[1], param.shape[1])
        ir.assert_structural_equal(output.shape[0], index.shape[0])
        ir.assert_structural_equal(index.shape[1], 1)
    except ValueError as error:
        raise ValueError(f"{operation} expects param=(param_h, width), "
                         "output=(count, width), and index=(count, 1)") from error
    return (_tpu_tensor_region(output,
                               "w"), _tpu_tensor_region(param, "r"), _tpu_tensor_region(index, "r"))


def ppl_gather(output, param, index, param_h):
    """Gather complete rows from one global-memory table.

    Args:
        output: Global output buffer with shape ``(count, width)``.
        param: Global source table with shape ``(param_h, width)``.
        index: Global ``uint32`` row indices with shape ``(count, 1)``.
        param_h: Positive static row count, equal to ``param.shape[0]``.

    Returns:
        PrimExpr: Handle to the TPU-Kernel gather semantic operation.

    Notes:
        This is a TPU-Kernel-only system-memory operation. Output, source, and
        index storage must be distinct.

    Dtype support:
        - TPU-Kernel: E4M3, E5M2, FP16, BF16, and FP32 payloads; UINT32 index.
        - RV Tensor: not exposed by this backend-specific op. Use
          ``ppl_embedding`` for a portable row lookup.
    """
    outptr, paramptr, indexptr = _prepare_row_gather("ppl_gather", output, param, index, param_h)
    return T.call_extern("handle", "tl.tpukernel.gather", outptr, paramptr, indexptr, param_h)


def ppl_topk(dst_data, dst_idx, src, K, descended, length):
    """Select the first ``K`` sorted values and their source indices.

    ``src`` has ``length`` elements.  The TPU HAU primitive writes exactly
    ``K`` values and ``K`` indices, so both destination buffers must have
    extent ``K``; no unspecified tail allocation is part of this contract.
    This TPU-Kernel-only operation is supported on BM1690 and is rejected for
    SG2260E by target-specific codegen.

    Dtype support:
        - TPU-Kernel on BM1690: FP32, INT32, or UINT32 values; INT32 indices.
        - TPU-Kernel on SG2260E: unavailable in the current SDK.
        - RV Tensor: unavailable.
    """
    for name, buffer in (("dst_data", dst_data), ("dst_idx", dst_idx), ("src", src)):
        _require_global_buffer(name, buffer)
    _require_same_dtype("ppl_topk payload", dst_data, src)
    _require_dtype("src", src, {"float32", "int32", "uint32"})
    if str(dst_idx.dtype) != "int32":
        raise ValueError(f"ppl_topk dst_idx dtype must be int32, got {dst_idx.dtype}")
    if not isinstance(K, int) or not isinstance(length, int) or K <= 0 or length <= 0:
        raise ValueError("ppl_topk K and length must be positive Python integers")
    if length < K:
        raise ValueError(f"ppl_topk requires K <= length, got K={K}, length={length}")
    if not isinstance(descended, bool):
        raise TypeError("ppl_topk descended must be a Python bool")
    for name, buffer in (("dst_data", dst_data), ("dst_idx", dst_idx), ("src", src)):
        _require_rank(name, buffer, 1)
        _require_descriptor_shape(f"ppl_topk {name}", buffer)
    _require_distinct_storage("ppl_topk", dst_data=dst_data, dst_idx=dst_idx, src=src)
    try:
        ir.assert_structural_equal(src.shape[0], length)
        ir.assert_structural_equal(dst_data.shape[0], K)
        ir.assert_structural_equal(dst_idx.shape[0], K)
    except ValueError as error:
        raise ValueError(
            "ppl_topk expects src=(length,), dst_data=(K,), and dst_idx=(K,)") from error
    dst_data_ptr = _tpu_tensor_region(dst_data, "w")
    dst_idx_ptr = _tpu_tensor_region(dst_idx, "w")
    srcptr = _tpu_tensor_region(src, "r")
    return T.call_extern("handle", "tl.tpukernel.topk", dst_data_ptr, dst_idx_ptr, srcptr, K,
                         descended, length)


def ppl_rsqrt(out, inp):
    """Compute reciprocal square root `out = rsqrt(inp)`.

    Args:
        out: Output tile.
        inp: Input tile.

    Returns:
        PrimExpr: Handle to the emitted rsqrt extern call.

    Example:
        `T.ppl_rsqrt(A_powsum, A_powsum)`

    Notes:
        PPL 1.7 exposes the same generic reciprocal-square-root instruction
        for FP16, BF16, and FP32 on both BM1690 and SG2260E.

    Dtype support:
        - TPU-Kernel: FP16, BF16, and FP32.
        - RV Tensor: FP16, BF16, and FP32.
    """
    for name, buffer in (("out", out), ("inp", inp)):
        _require_local_buffer(name, buffer)
        _require_dtype(name, buffer, _TPU_BASE_FLOAT_DTYPES)
    _require_same_dtype("ppl_rsqrt", out, inp)
    _require_same_shape("ppl_rsqrt", out, inp)
    inpptr = _tpu_tensor_region(inp, "r")
    outptr = _tpu_tensor_region(out, "w")
    return T.call_extern("handle", "tl.tpu.rsqrt", outptr, inpptr)


def ppl_add_C(out, inp1, value):
    """Compute elementwise scalar add `out = inp1 + value`.

    Args:
        out: Output tile.
        inp1: Input tile.
        value: Scalar bias to add.

    Returns:
        PrimExpr: Handle to the emitted add-by-constant extern call.

    Example:
        `T.ppl_add_C(A_powsum, A_powsum, T.float32(1e-12))`

    Notes:
        This is commonly used to add epsilon, bias, or other scalar offsets to
        a local tile.

    Dtype support:
        - TPU-Kernel: E4M3, E5M2, FP16, BF16, and FP32.
        - RV Tensor: E4M3, E5M2, FP16, BF16, and FP32.
    """
    for name, buffer in (("out", out), ("inp1", inp1)):
        _require_local_buffer(name, buffer)
        _require_dtype(name, buffer, _TPU_ELEMENTWISE_FLOAT_DTYPES)
    _require_same_dtype("ppl_add_C", out, inp1)
    _require_same_shape("ppl_add_C", out, inp1)
    outptr = _tpu_tensor_region(out, "w")
    inpptr1 = _tpu_tensor_region(inp1, "r")
    return T.call_extern("handle", "tl.tpu.add_scalar", outptr, inpptr1, value)


def ppl_add(out, inp1, inp2):
    """Compute elementwise addition `out = inp1 + inp2`.

    Args:
        out: Output tile.
        inp1: Left-hand input tile.
        inp2: Right-hand input tile.

    Returns:
        PrimExpr: Handle to the emitted add extern call.

    Example:
        `T.ppl_add(logsum, logsum, scores_sum)`
        `T.ppl_add(x_neg_exp_1, x_neg_exp, ones)`

    Notes:
        The usual usage is that all tiles have the same shape.
        A limited broadcast-style usage is also supported in common cases when
        the second input has shape `(M, 1)`.

    Dtype support:
        - TPU-Kernel: E4M3, E5M2, FP16, BF16, and FP32.
        - RV Tensor: E4M3, E5M2, FP16, BF16, and FP32.
    """
    for name, buffer in (("out", out), ("inp1", inp1), ("inp2", inp2)):
        _require_local_buffer(name, buffer)
        _require_rank(name, buffer, 2)
        _require_dtype(name, buffer, _TPU_ELEMENTWISE_FLOAT_DTYPES)
    _require_same_dtype("ppl_add", out, inp1, inp2)
    _require_elementwise_shapes("ppl_add", out, inp1, inp2)
    outptr = _tpu_tensor_region(out, "w")
    inpptr1 = _tpu_tensor_region(inp1, "r")
    inpptr2 = _tpu_tensor_region(inp2, "r")
    return T.call_extern("handle", "tl.tpu.add", outptr, inpptr1, inpptr2)


def ppl_div(out, inp1, inp2):
    """Compute elementwise division `out = inp1 / inp2`.

    Args:
        out: Output tile.
        inp1: Numerator tile.
        inp2: Denominator tile.

    Returns:
        PrimExpr: Handle to the emitted division extern call.

    Example:
        `T.ppl_div(acc_o, acc_o, logsum)`
        `T.ppl_div(x_neg_exp_1_div, x, x_neg_exp_1)`

    Notes:
        The usual usage is that all tiles have the same shape.
        A limited broadcast-style usage is also supported in common cases when
        the second input has shape `(M, 1)`.

    Dtype support:
        - TPU-Kernel: FP16, BF16, and FP32.
        - RV Tensor: FP16, BF16, and FP32.
    """
    for name, buffer in (("out", out), ("inp1", inp1), ("inp2", inp2)):
        _require_local_buffer(name, buffer)
        _require_rank(name, buffer, 2)
        _require_dtype(name, buffer, _TPU_BASE_FLOAT_DTYPES)
    _require_same_dtype("ppl_div", out, inp1, inp2)
    _require_elementwise_shapes("ppl_div", out, inp1, inp2)
    outptr = _tpu_tensor_region(out, "w")
    inpptr1 = _tpu_tensor_region(inp1, "r")
    inpptr2 = _tpu_tensor_region(inp2, "r")
    return T.call_extern("handle", "tl.tpu.div", outptr, inpptr1, inpptr2)


@T.macro
def _reduce_sum_impl(inp, out, dim, eu_elements):
    """Internal macro backing `ppl_reduce_sum`.

    Prefer calling `ppl_reduce_sum(...)` directly in user kernels.
    """
    with T.block("reduce_sum"):
        tmp_shape = [inp.shape[0], eu_elements]
        tmp_buffer_sum = T.alloc_shared(tmp_shape, inp.dtype)
        eu_num = T.int32(eu_elements)
        channel = T.int32(64)
        align_w = T.ceildiv(inp.shape[1], eu_num) * eu_num
        stride = T.ceildiv(inp.shape[0], channel) * align_w
        # Shared semantic ABI: TPU-Kernel uses padded pooling; RV uses column slices.
        T.call_extern("handle", "tl.tpu.reduce_sum", _tpu_tensor_region(inp, "rw"),
                      _tpu_tensor_region(out, "w"), _tpu_tensor_region(tmp_buffer_sum, "rw"),
                      eu_num, align_w, stride)


def ppl_reduce_sum(inp, out, dim):
    """Reduce a 2D tile along its second dimension with summation.

    Args:
        inp: Input tile, typically shaped `(M, N)`.
        out: Output tile, typically shaped `(M, 1)`.
        dim: Reduction axis. The current TPU path only supports `dim=1`.

    Returns:
        PrimExpr: Handle to the emitted reduction macro call.

    Example:
        `T.ppl_reduce_sum(acc_s, scores_sum, dim=1)`
        `T.ppl_reduce_sum(X_shared, Y_shared, dim=1)`

    Notes:
        This op is intended for 2D tiles and currently only supports
        reduction along `dim=1`.
        The usual output shape is `(inp.shape[0], 1)`.

    Dtype support:
        - TPU-Kernel: FP16, BF16, and FP32.
        - RV Tensor: E4M3, E5M2, FP16, BF16, and FP32. FP8 accumulation rounds
          after every same-format add.
    """
    for name, buffer in (("inp", inp), ("out", out)):
        _require_local_buffer(name, buffer)
        _require_rank(name, buffer, 2)
        _require_dtype(name, buffer, _TPU_ELEMENTWISE_FLOAT_DTYPES)
    _require_same_dtype("ppl_reduce_sum", inp, out)
    _require_storage_disjoint("ppl_reduce_sum", "inp", inp, "out", out)
    if dim != 1:
        raise ValueError(f"ppl_reduce_sum only supports dim=1, got {dim}")
    try:
        ir.assert_structural_equal(inp.shape[0], out.shape[0])
        ir.assert_structural_equal(out.shape[1], 1)
    except ValueError as error:
        raise ValueError(
            f"ppl_reduce_sum expects output shape ({inp.shape[0]}, 1), got {out.shape}") from error
    return _reduce_sum_impl(inp, out, dim, _TPUV7_EU_ELEMENTS[str(inp.dtype)])


@T.macro
def _reduce_max_impl(inp, out, dim, eu_elements):
    """Internal macro backing `ppl_reduce_max`.

    Prefer calling `ppl_reduce_max(...)` directly in user kernels.
    """
    with T.block("reduce_max"):
        tmp_shape = [inp.shape[0], eu_elements]
        tmp_buffer_max = T.alloc_shared(tmp_shape, inp.dtype)
        eu_num = T.int32(eu_elements)
        channel = T.int32(64)
        align_w = T.ceildiv(inp.shape[1], eu_num) * eu_num
        stride = T.ceildiv(inp.shape[0], channel) * align_w
        # Shared semantic ABI: TPU-Kernel uses padded pooling; RV uses column slices.
        T.call_extern("handle", "tl.tpu.reduce_max", _tpu_tensor_region(inp, "rw"),
                      _tpu_tensor_region(out, "w"), _tpu_tensor_region(tmp_buffer_max, "rw"),
                      eu_num, align_w, stride)


def ppl_reduce_max(inp, out, dim):
    """Reduce a 2D tile along its second dimension with max.

    Args:
        inp: Input tile, typically shaped `(M, N)`.
        out: Output tile, typically shaped `(M, 1)`.
        dim: Reduction axis. The current TPU path only supports `dim=1`.

    Returns:
        PrimExpr: Handle to the emitted reduction macro call.

    Example:
        `T.ppl_reduce_max(acc_s, scores_max, dim=1)`

    Notes:
        This op is intended for 2D tiles and currently only supports
        reduction along `dim=1`.
        The usual output shape is `(inp.shape[0], 1)`.
        This operation always overwrites `out`.  Cross-tile accumulation must
        be expressed as a separate max operation; the TPU-Kernel reduction
        sequence does not consume the previous contents of `out`.

    Dtype support:
        - TPU-Kernel: E4M3, E5M2, FP16, BF16, and FP32.
        - RV Tensor: E4M3, E5M2, FP16, BF16, and FP32.
    """
    for name, buffer in (("inp", inp), ("out", out)):
        _require_local_buffer(name, buffer)
        _require_rank(name, buffer, 2)
        _require_dtype(name, buffer, _TPU_ELEMENTWISE_FLOAT_DTYPES)
    _require_same_dtype("ppl_reduce_max", inp, out)
    _require_storage_disjoint("ppl_reduce_max", "inp", inp, "out", out)
    if dim != 1:
        raise ValueError(f"ppl_reduce_max only supports dim=1, got {dim}")
    try:
        ir.assert_structural_equal(inp.shape[0], out.shape[0])
        ir.assert_structural_equal(out.shape[1], 1)
    except ValueError as error:
        raise ValueError(
            f"ppl_reduce_max expects output shape ({inp.shape[0]}, 1), got {out.shape}") from error
    return _reduce_max_impl(inp, out, dim, _TPUV7_EU_ELEMENTS[str(inp.dtype)])


def ppl_embedding(out, weight, indices):
    """Gather embedding rows from a global table.

    Args:
        out: Global output buffer with shape ``(N, D)``.
        weight: Global embedding table with shape ``(V, D)``.
        indices: Global UINT32 token indices with shape ``(N, 1)``.

    Dtype support:
        - TPU-Kernel: E4M3, E5M2, FP16, BF16, and FP32 payloads.
        - RV Tensor: E4M3, E5M2, FP16, BF16, and FP32 payloads.

    Indices must lie in ``[0, V)``. The caller validates token IDs before launch.
    No padding index, negative indexing, or training gradient is implied.
    """
    param_h = _static_positive_dim("ppl_embedding weight axis 0", weight.shape[0])
    outptr, weightptr, indexptr = _prepare_row_gather("ppl_embedding", out, weight, indices,
                                                      param_h)
    return T.call_extern("handle", "tl.tpu.embedding", outptr, weightptr, indexptr, param_h)
