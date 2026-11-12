"""Preserve compact NumPy chunks through elementwise operations and reductions."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

import numpy as np

from dask.array.numpy_compat import normalize_axis_tuple

_NUMERIC_OPERATORS = (
    operator.abs,
    operator.add,
    operator.and_,
    operator.eq,
    operator.floordiv,
    operator.ge,
    operator.gt,
    operator.invert,
    operator.le,
    operator.lshift,
    operator.lt,
    operator.mod,
    operator.mul,
    operator.ne,
    operator.neg,
    operator.or_,
    operator.pos,
    operator.pow,
    operator.rshift,
    operator.sub,
    operator.truediv,
    operator.xor,
)


def can_preserve_broadcast(func: Callable[..., Any]) -> bool:
    # A user-provided ufunc (e.g. frompyfunc) may have side effects. Only
    # NumPy's own elementwise, single-output ufuncs and known operators qualify.
    if isinstance(func, np.ufunc):
        return (
            func.nout == 1
            and func.signature is None
            and getattr(np, func.__name__, None) is func
        )
    return any(func is op for op in _NUMERIC_OPERATORS)


def preserve_broadcast(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Evaluate only the distinct values along shared zero-stride axes.

    Slice instead of extracting scalar values so NumPy's dtype promotion sees
    the same array ranks and scalar types as it would for the original chunks.
    The resulting read-only view has the original broadcast shape, just like
    the chunks created by Dask's ones, zeros, and full.
    """
    arrays = []
    for arg in args:
        if type(arg) is np.ndarray:
            if arg.dtype.hasobject:
                return func(*args, **kwargs)
            arrays.append(arg)
        elif type(arg) in (bool, int, float, complex):
            continue
        elif (
            isinstance(arg, np.generic)
            and type(arg).__module__ == "numpy"
            and not arg.dtype.hasobject
        ):
            continue
        else:
            # Subclasses, other array backends, and object scalars can define
            # their own arithmetic or observe how many elements are visited.
            return func(*args, **kwargs)

    if not any(
        size > 1 and stride == 0
        for array in arrays
        for size, stride in zip(array.shape, array.strides)
    ):
        return func(*args, **kwargs)

    shape = np.broadcast_shapes(*(array.shape for array in arrays))
    if 0 in shape:
        # Empty operations must not evaluate even one element: doing so could
        # introduce arithmetic warnings or exceptions that NumPy does not emit.
        return func(*args, **kwargs)

    compress = [size > 1 for size in shape]
    for array in arrays:
        offset = len(shape) - array.ndim
        for axis, (size, stride) in enumerate(zip(array.shape, array.strides)):
            if size != 1 and stride != 0:
                compress[offset + axis] = False

    if not any(compress):
        return func(*args, **kwargs)

    compact_args = []
    for arg in args:
        if type(arg) is np.ndarray and arg.ndim:
            offset = len(shape) - arg.ndim
            index = tuple(
                slice(0, 1) if compress[offset + axis] else slice(None)
                for axis in range(arg.ndim)
            )
            compact_args.append(arg[index])
        else:
            compact_args.append(arg)

    result = func(*compact_args, **kwargs)
    return np.broadcast_to(result, shape)


def reduce_preserving_broadcast(
    func: Callable[..., Any],
    x: Any,
    axis: int | tuple[int, ...] | None = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Avoid repeated reads in idempotent reductions of NumPy chunks.

    Only min, max, any and all use this helper. Other reductions keep their
    original arithmetic. Masks, initial values, output buffers and positional
    options go directly to NumPy, preserving their full argument semantics.
    """
    if (
        type(x) is not np.ndarray
        or x.dtype.kind not in "biufc"
        or x.size == 0
        or args
        or kwargs.get("out") is not None
        or "where" in kwargs
        or "initial" in kwargs
    ):
        return func(x, axis, *args, **kwargs)

    compress = [size > 1 and stride == 0 for size, stride in zip(x.shape, x.strides)]
    if not any(compress):
        return func(x, axis=axis, **kwargs)

    # Let NumPy itself report unsupported or duplicate axes.
    try:
        axes = (
            tuple(range(x.ndim)) if axis is None else normalize_axis_tuple(axis, x.ndim)
        )
    except (TypeError, ValueError):
        return func(x, axis=axis, **kwargs)

    extrema = func is np.min or func is np.max
    if (
        extrema
        and x.dtype.kind in "fc"
        and any(x.shape[a] > 1 and x.strides[a] != 0 for a in axes)
    ):
        # NumPy's reduction loop can select different signs among tied zeros.
        # Only collapse floating extrema when every reduced axis is constant.
        return func(x, axis=axis, **kwargs)

    compact = x[
        tuple(slice(0, 1) if collapse else slice(None) for collapse in compress)
    ]
    if extrema and x.dtype.kind in "fc" and not np.isfinite(compact).all():
        # Preserve signaling-NaN and other exceptional floating behavior.
        return func(x, axis=axis, **kwargs)

    result = func(compact, axis=axis, **kwargs)
    keepdims = result.ndim == x.ndim and bool(axes)
    shape = (
        tuple(1 if a in axes else size for a, size in enumerate(x.shape))
        if keepdims
        else tuple(size for a, size in enumerate(x.shape) if a not in axes)
    )
    if result.shape != shape:
        return np.broadcast_to(result, shape)
    return result
