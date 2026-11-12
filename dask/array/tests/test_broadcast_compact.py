"""Regressions for compact elementwise chunks (Dask issue 9517)."""

import operator

import numpy as np
import pytest

import dask
import dask.array as da
from dask.array._broadcast import can_preserve_broadcast, preserve_broadcast
from dask.array.utils import assert_eq
from dask.core import flatten
from dask.sizeof import sizeof


def first_chunk(array):
    key = next(flatten(array.__dask_keys__()))
    return dask.get(array.__dask_graph__(), key)


@pytest.mark.parametrize("func", [operator.add, np.add, np.multiply, np.equal])
def test_uniform_elementwise_chunks_remain_compact(func):
    x = da.ones(100_000, chunks=100_000, dtype=np.int64)
    result = func(x, x)
    chunk = first_chunk(result)
    expected = func(np.ones(100_000, dtype=np.int64), 1)
    np.testing.assert_array_equal(chunk, expected)
    assert chunk.shape == (100_000,)
    assert chunk.strides == (0,)
    assert sizeof(chunk) == chunk.dtype.itemsize
    assert_eq(result, expected)


def test_shared_broadcast_axis_with_varying_values():
    x = da.from_array(np.arange(3.0), chunks=3)
    x = da.ones((200, 1), chunks=(200, 1)) * x
    result = np.sin(x) + x
    chunk = first_chunk(result)
    expected = np.broadcast_to(np.sin(np.arange(3.0)) + np.arange(3.0), (200, 3))
    np.testing.assert_array_equal(chunk, expected)
    assert chunk.strides[0] == 0
    assert sizeof(chunk) == 3 * chunk.dtype.itemsize
    assert_eq(result, expected)


def test_distinct_broadcast_axes_are_not_collapsed():
    x = da.ones((4, 1), chunks=(4, 1), dtype=int) * da.from_array(
        np.arange(3), chunks=3
    )
    y = da.from_array(np.arange(4).reshape(4, 1), chunks=(4, 1))
    result = x + y
    expected = np.arange(3) + np.arange(4).reshape(4, 1)
    assert_eq(result, expected)
    assert first_chunk(result).strides[0] != 0


@pytest.mark.parametrize("dtype", [np.int8, np.float32, np.complex64])
def test_compact_arithmetic_preserves_scalar_promotion(dtype):
    x = da.ones(100, chunks=100, dtype=dtype)
    result = x + 1.0
    # The experimental expression planner converts Python scalars to arrays.
    # Compare with its unchanged dense path as well as checking the legacy
    # planner against NumPy\'s weak-scalar promotion.
    dense = np.ones(100, dtype=dtype)
    if da._array_expr_enabled():
        expected = (da.from_array(dense, chunks=100) + 1.0).compute()
    else:
        expected = dense + 1.0
    assert_eq(result, expected)
    assert first_chunk(result).dtype == expected.dtype


def test_dtype_enforcement_precedes_restoring_shape():
    # Exercise the explicit-dtype path without converting the final broadcast
    # view into a dense array.
    from dask.array.core import elemwise

    if da._array_expr_enabled():
        pytest.skip("core.elemwise expects the legacy collection")
    x = da.ones(100, chunks=100, dtype=np.int8)
    result = elemwise(np.add, x, x, dtype=np.int64)
    chunk = first_chunk(result)
    assert chunk.dtype == np.dtype("int64")
    assert sizeof(chunk) == 8
    assert_eq(result, np.full(100, 2, dtype=np.int64))


def test_where_preserves_unselected_out_values():
    if da._array_expr_enabled():
        pytest.skip("array expressions do not implement out/where")
    x = da.ones(6, chunks=6)
    mask = da.from_array(np.array([True, False] * 3), chunks=6)
    out = da.from_array(np.arange(6.0), chunks=6)
    result = np.add(x, x, where=mask, out=out)
    assert_eq(result, np.array([2.0, 1.0, 2.0, 3.0, 2.0, 5.0]))


def test_empty_chunks_do_not_evaluate_a_representative():
    x = np.broadcast_to(np.array(0.0), (0, 100))
    with np.errstate(divide="raise", invalid="raise"):
        result = preserve_broadcast(np.reciprocal, x)
    assert result.shape == x.shape


def test_array_subclasses_keep_their_dispatch():
    x = np.ma.array([1.0, 2.0], mask=[False, True])
    result = preserve_broadcast(np.add, x, x)
    assert isinstance(result, np.ma.MaskedArray)
    np.testing.assert_array_equal(result.mask, x.mask)
    np.testing.assert_array_equal(result.data[~result.mask], [2.0])


def test_object_operations_visit_each_element():
    class Value:
        calls = 0

        def __add__(self, other):
            type(self).calls += 1
            return type(self).calls

    x = np.broadcast_to(np.array(Value(), dtype=object), (6,))
    result = preserve_broadcast(operator.add, x, x)
    np.testing.assert_array_equal(result, np.arange(1, 7))
    assert Value.calls == 6


def test_user_functions_and_generalized_ufuncs_are_not_eligible():
    def custom(x):
        return x

    assert not can_preserve_broadcast(custom)
    assert not can_preserve_broadcast(np.frompyfunc(custom, 1, 1))
    assert not can_preserve_broadcast(np.matmul)
    assert not can_preserve_broadcast(np.modf)


def test_dense_inputs_still_produce_independent_output():
    x = np.arange(6.0)
    result = preserve_broadcast(np.add, x, 1)
    np.testing.assert_array_equal(result, x + 1)
    assert not np.shares_memory(result, x)


@pytest.mark.parametrize("name", ["min", "max", "any", "all"])
def test_idempotent_reductions_restore_nonreduced_axes(name):
    from dask.array import chunk

    x = np.broadcast_to(np.arange(3).reshape(1, 3, 1), (20, 3, 40))
    result = getattr(chunk, name)(x, axis=(-2, -1), keepdims=True)
    expected = getattr(np, name)(x, axis=(-2, -1), keepdims=True)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (20, 1, 1)
    assert sizeof(result) == result.dtype.itemsize


def test_uniform_reductions_use_public_array_path():
    x = da.ones(100_000, chunks=25_000)
    for name in ("min", "max", "any", "all"):
        result = getattr(x, name)().compute()
        expected = getattr(np, name)(np.ones(100_000))
        np.testing.assert_equal(result, expected)


def test_floating_extrema_preserve_signed_zero_selection():
    from dask.array import chunk

    x = np.broadcast_to(np.array([0.0, -0.0]), (100, 2))
    for name in ("min", "max"):
        result = getattr(chunk, name)(x)
        expected = getattr(np, name)(x)
        np.testing.assert_equal(result, expected)
        np.testing.assert_equal(np.signbit(result), np.signbit(expected))


def test_reduction_options_and_empty_inputs_keep_numpy_behavior():
    from dask.array import chunk

    x = np.broadcast_to(np.array(2), (6,))
    mask = np.array([True, False, True, False, False, False])
    np.testing.assert_equal(
        chunk.min(x, initial=1, where=mask),
        np.min(x, initial=1, where=mask),
    )
    out = np.empty((), dtype=x.dtype)
    assert chunk.max(x, None, out) is out
    np.testing.assert_equal(out, 2)
    empty = np.empty((0, 6))
    np.testing.assert_array_equal(chunk.any(empty, axis=0), np.any(empty, axis=0))
    with pytest.raises(ValueError):
        chunk.min(empty, axis=0)
