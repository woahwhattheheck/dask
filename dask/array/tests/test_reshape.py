from __future__ import annotations

import numpy as np
import pytest

import dask
import dask.array as da
from dask.array import from_array
from dask.array.reshape import (
    _smooth_chunks,
    contract_tuple,
    expand_tuple,
    reshape_blockwise,
    reshape_rechunk,
)
from dask.array.utils import assert_eq
from dask.graph_manipulation import bind, clone
from dask.layers import ArrayReshapeLayer


@pytest.mark.parametrize(
    "inshape,outshape,prechunks,inchunks,outchunks",
    [
        ((4,), (4,), ((2, 2),), ((2, 2),), ((2, 2),)),
        ((4,), (2, 2), ((2, 2),), ((2, 2),), ((1, 1), (2,))),
        ((4,), (4, 1), ((2, 2),), ((2, 2),), ((2, 2), (1,))),
        ((4,), (1, 4), ((2, 2),), ((2, 2),), ((1,), (2, 2))),
        ((1, 4), (4,), ((1,), (2, 2)), ((1,), (2, 2)), ((2, 2),)),
        ((4, 1), (4,), ((2, 2), (1,)), ((2, 2), (1,)), ((2, 2),)),
        (
            (4, 1, 4),
            (4, 4),
            ((2, 2), (1,), (2, 2)),
            ((2, 2), (1,), (2, 2)),
            ((2, 2), (2, 2)),
        ),
        ((4, 4), (4, 1, 4), ((2, 2), (2, 2)), ((2, 2), (2, 2)), ((2, 2), (1,), (2, 2))),
        ((2, 2), (4,), ((2,), (2,)), ((2,), (2,)), ((4,),)),
        ((2, 2), (4,), ((1, 1), (2,)), ((1, 1), (2,)), ((2, 2),)),
        ((2, 2), (4,), ((2,), (1, 1)), ((1, 1), (2,)), ((2, 2),)),
        (
            (64,),
            (4, 4, 4),
            ((8, 8, 8, 8, 8, 8, 8, 8),),
            ((8, 8, 8, 8, 8, 8, 8, 8),),
            ((1, 1, 1, 1), (2, 2), (4,)),
        ),
        ((64,), (4, 4, 4), ((32, 32),), ((32, 32),), ((2, 2), (4,), (4,))),
        ((64,), (4, 4, 4), ((16, 48),), ((16, 48),), ((1, 3), (4,), (4,))),
        ((64,), (4, 4, 4), ((20, 44),), ((16, 16, 32),), ((1, 1, 2), (4,), (4,))),
        (
            (64, 4),
            (8, 8, 4),
            ((16, 16, 16, 16), (2, 2)),
            ((16, 16, 16, 16), (2, 2)),
            ((2, 2, 2, 2), (8,), (2, 2)),
        ),
    ],
)
def test_reshape_rechunk(inshape, outshape, prechunks, inchunks, outchunks):
    result_in, result_out, _, _ = reshape_rechunk(inshape, outshape, prechunks)
    assert result_in == inchunks
    assert result_out == outchunks
    assert np.prod(list(map(len, result_in))) == np.prod(list(map(len, result_out)))


def test_expand_tuple():
    assert expand_tuple((2, 4), 2) == (1, 1, 2, 2)
    assert expand_tuple((2, 4), 3) == (1, 1, 1, 1, 2)
    assert expand_tuple((3, 4), 2) == (1, 2, 2, 2)
    assert expand_tuple((7, 4), 3) == (2, 2, 3, 1, 1, 2)


def test_contract_tuple():
    assert contract_tuple((1, 1, 2, 3, 1), 2) == (2, 2, 2, 2)
    assert contract_tuple((1, 1, 2, 5, 1), 2) == (2, 2, 4, 2)
    assert contract_tuple((2, 4), 2) == (2, 4)
    assert contract_tuple((2, 4), 3) == (6,)


def test_reshape_unknown_sizes():
    a = np.random.random((10, 6, 6))
    A = da.from_array(a, chunks=(5, 2, 3))

    a2 = a.reshape((60, -1))
    A2 = A.reshape((60, -1))

    assert A2.shape == (60, 6)
    assert_eq(A2, a2)

    with pytest.raises(ValueError):
        a.reshape((60, -1, -1))
    with pytest.raises(ValueError):
        A.reshape((60, -1, -1))


@pytest.mark.parametrize("shape", [(2, -1), (0,), (0, 6), (2, 0, 1)])
def test_reshape_empty_multiple_chunks(shape):
    # GH#2561: zero-length dimensions do not need general reshape rechunking.
    a = np.ones((2, 0, 0))
    arr = da.from_array(a, chunks=(1, 0, 0))
    result = arr.reshape(shape)

    assert result.npartitions == 1
    assert_eq(result, a.reshape(shape))


@pytest.mark.parametrize("shape", [(0, 2), (2, 0), (-1, 4)])
def test_reshape_known_empty_unknown_axis(shape):
    values = np.empty((4, 0))
    arr = da.from_array(values, chunks=(2, 0))
    mask = da.from_array(np.array([True, False, True, True]), chunks=2)
    result = arr[mask].reshape(shape)

    assert not any(np.isnan(s) for s in result.shape)
    assert_eq(result, values[[True, False, True, True]].reshape(shape))


def test_reshape_different_chunk_plans():
    values = np.arange(120).reshape(4, 5, 6)
    arr = da.from_array(values, chunks=(2, 2, 3))
    merged = arr.reshape((20, 6))
    split = arr.reshape((20, 6), merge_chunks=False)

    assert merged.chunks != split.chunks
    assert merged.name != split.name
    expected = (merged.blocks[0, 0].compute(), split.blocks[0, 0].compute())
    actual = dask.compute(merged.blocks[0, 0], split.blocks[0, 0])
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference)


def test_reshape_symbolic_layer():
    values = np.arange(48).reshape(2, 3, 8)
    arr = da.from_array(values, chunks=(1, 1, 4))
    result = arr.reshape((6, 8))
    layer = result.dask.layers[result.name]

    assert isinstance(layer, ArrayReshapeLayer)
    assert not layer.is_materialized()
    key = (result.name, 2, 1)
    culled, dependencies = layer.cull({key}, ())
    assert len(culled) == 1
    assert dependencies[key] == {(arr.name, 0, 2, 1)}
    assert_eq(result.blocks[2, 1], values.reshape((6, 8))[2:3, 4:])
    assert not layer.is_materialized()


@pytest.mark.parametrize("omit_input", [False, True])
def test_reshape_graph_manipulation(omit_input):
    values = np.arange(48).reshape(2, 3, 8)
    arr = da.from_array(values, chunks=(1, 1, 4))
    result = arr.reshape((6, 8))
    omit = arr if omit_input else None
    assert_eq(clone(result, omit=omit), values.reshape((6, 8)))
    assert_eq(bind(result, arr.sum(), omit=omit), values.reshape((6, 8)))


def test_reshape_empty_masked_multiple_chunks():
    a = np.ma.ones((2, 0, 0))
    arr = da.from_array(a, chunks=(1, 0, 0))
    result = arr.reshape((2, 0))

    assert isinstance(result._meta, np.ma.MaskedArray)
    assert isinstance(result.compute(), np.ma.MaskedArray)
    assert_eq(result, a.reshape((2, 0)))


def test_reshape_empty_scipy_sparse_multiple_chunks():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    a = scipy_sparse.csr_matrix((2, 0))
    arr = da.from_array(a, chunks=(1, 0))
    result = arr.reshape((0, 2))

    assert scipy_sparse.issparse(result._meta)
    actual = result.compute()
    assert scipy_sparse.issparse(actual)
    np.testing.assert_array_equal(actual.toarray(), a.toarray().reshape((0, 2)))


@pytest.mark.parametrize("unknown_axes", [(0,), (1,), (2,), (0, 2), (0, 1, 2)])
@pytest.mark.parametrize("method", ["reshape", "ravel", "flatten"])
def test_flatten_unknown_chunks(unknown_axes, method):
    # GH#3194: combine trailing chunks before flattening to preserve C order.
    expected = np.arange(120).reshape(4, 5, 6)
    arr = da.from_array(expected, chunks=2)
    for axis in unknown_axes:
        keep = np.arange(expected.shape[axis]) >= expected.shape[axis] // 2
        key = (slice(None),) * axis + (keep,)
        expected = expected[key]
        key = (slice(None),) * axis + (da.from_array(keep, chunks=2),)
        arr = arr[key]

    if method == "reshape":
        result = arr.reshape(-1)
    else:
        result = getattr(arr, method)()

    assert np.isnan(result.shape[0])
    assert not result.dask.layers[result.name].is_materialized()
    assert_eq(result, expected.ravel())


@pytest.mark.parametrize(
    "chunks, merge_chunks, output_blocks",
    [((2, 2, 3), True, 2), ((2, 2, 3), False, 40), ((1, 1, 3), True, 40)],
)
def test_flatten_unknown_columns_preserves_chunk_boundaries(
    chunks, merge_chunks, output_blocks
):
    # Independent column blocks can be flattened one row at a time, without
    # concatenating a complete trailing slab for every first-axis block.
    values = np.arange(120).reshape(4, 5, 6)
    arr = da.from_array(values, chunks=chunks)
    keep = np.array([True, False, True, False, True, True])
    filtered = arr[:, :, da.from_array(keep, chunks=3)]
    result = filtered.reshape(-1, merge_chunks=merge_chunks)

    assert len(result.chunks[0]) == output_blocks
    assert_eq(result, values[:, :, keep].ravel())


def test_flatten_unknown_trailing_single_block():
    values = np.arange(120).reshape(4, 5, 6)
    arr = da.from_array(values, chunks=(2, 5, 6))
    keep = np.array([True, False, True, False, True, True])
    filtered = arr[:, :, da.from_array(keep, chunks=6)]
    result = filtered.reshape(-1, merge_chunks=False)

    assert len(result.chunks[0]) == 2
    assert_eq(result, values[:, :, keep].ravel())


def test_flatten_unknown_requires_merge():
    arr = da.from_array(np.arange(24).reshape(4, 6), chunks=(2, 3))
    filtered = arr[arr[:, 0] > 0]
    with pytest.raises(ValueError, match="merge_chunks=False"):
        filtered.reshape(-1, merge_chunks=False)
    assert_eq(filtered.reshape(-1), np.arange(24).reshape(4, 6)[1:].ravel())


@pytest.mark.parametrize("shape", [(0, 6), (4, 0, 6), (4, 6, 0)])
def test_flatten_unknown_with_known_zero(shape):
    arr = da.from_array(np.empty(shape), chunks=2)
    axis = shape.index(6)
    keep = da.from_array(np.array([True, False, True, False, True, True]), chunks=2)
    filtered = arr[(slice(None),) * axis + (keep,)]
    result = filtered.reshape(-1)

    assert result.shape == (0,)
    assert all(c == 0 for c in result.chunks[0])
    assert_eq(result, np.empty(0))


@pytest.mark.parametrize(
    "inshape, inchunks, outshape, outchunks",
    [
        # (2, 3, 4) -> (6, 4)
        ((2, 3, 4), ((1, 1), (1, 2), (2, 2)), (6, 4), ((1, 2, 1, 2), (2, 2))),
        # (1, 2, 3, 4) -> (12, 4)
        ((1, 2, 3, 4), ((1,), (1, 1), (1, 2), (2, 2)), (6, 4), ((1, 2, 1, 2), (2, 2))),
        # (2, 2, 3, 4) -> (12, 4)
        (
            (2, 2, 3, 4),
            ((1, 1), (1, 1), (1, 2), (2, 2)),
            (12, 4),
            ((1, 2, 1, 2, 1, 2, 1, 2), (2, 2)),
        ),
        # (2, 2, 3, 4) -> (4, 3, 4)
        (
            (2, 2, 3, 4),
            ((1, 1), (1, 1), (1, 2), (2, 2)),
            (4, 3, 4),
            ((1, 1, 1, 1), (1, 2), (2, 2)),
        ),
        # (2, 2, 3, 4) -> (4, 3, 4)
        ((2, 2, 3, 4), ((1, 1), (2,), (1, 2), (4,)), (4, 3, 4), ((2, 2), (1, 2), (4,))),
        # (2, 3, 4) -> (24,).
        ((2, 3, 4), ((1, 1), (1, 1, 1), (2, 2)), (24,), ((2,) * 12,)),
        # (2, 3, 4) -> (2, 12)
        ((2, 3, 4), ((1, 1), (1, 1, 1), (4,)), (2, 12), ((1, 1), (4,) * 3)),
    ],
)
def test_reshape_all_chunked_no_merge(inshape, inchunks, outshape, outchunks):
    # https://github.com/dask/dask/issues/5544#issuecomment-712280433
    # When the early axes are completely chunked then we are just moving blocks
    # and can avoid any rechunking. The result inchunks are the same as the
    # input chunks.
    base = np.arange(np.prod(inshape)).reshape(inshape)
    a = da.from_array(base, chunks=inchunks)

    # test directly
    inchunks2, outchunks2, _, _ = reshape_rechunk(a.shape, outshape, inchunks)
    assert inchunks2 == inchunks
    assert outchunks2 == outchunks

    # and via reshape
    result = a.reshape(outshape)
    assert result.chunks == outchunks
    assert_eq(result, base.reshape(outshape))


@pytest.mark.parametrize(
    "inshape, inchunks, expected_inchunks, outshape, outchunks",
    [
        # (2, 3, 4) -> (24,). This does merge, since the second dim isn't fully chunked!
        (
            (2, 3, 4),
            ((1, 1), (1, 2), (2, 2)),
            (
                (1, 1),
                (
                    1,
                    1,
                    1,
                ),
                (4,),
            ),
            (24,),
            ((4, 4, 4, 4, 4, 4),),
        ),
    ],
)
def test_reshape_all_not_chunked_merge(
    inshape, inchunks, expected_inchunks, outshape, outchunks
):
    base = np.arange(np.prod(inshape)).reshape(inshape)
    a = da.from_array(base, chunks=inchunks)

    # test directly
    inchunks2, outchunks2, _, _ = reshape_rechunk(a.shape, outshape, inchunks)
    assert inchunks2 == expected_inchunks
    assert outchunks2 == outchunks

    # and via reshape
    result = a.reshape(outshape)
    assert result.chunks == outchunks
    assert_eq(result, base.reshape(outshape))


@pytest.mark.parametrize(
    "inshape, inchunks, outshape, outchunks",
    [
        # (2, 3, 4) -> (6, 4)
        ((2, 3, 4), ((2,), (1, 2), (2, 2)), (6, 4), ((1, 2, 1, 2), (2, 2))),
        # (1, 2, 3, 4) -> (12, 4)
        ((1, 2, 3, 4), ((1,), (2,), (1, 2), (2, 2)), (6, 4), ((1, 2, 1, 2), (2, 2))),
        # (2, 2, 3, 4) -> (12, 4) (3 cases)
        (
            (2, 2, 3, 4),
            ((1, 1), (2,), (1, 2), (2, 2)),
            (12, 4),
            ((1, 2, 1, 2, 1, 2, 1, 2), (2, 2)),
        ),
        (
            (2, 2, 3, 4),
            ((2,), (1, 1), (1, 2), (2, 2)),
            (12, 4),
            ((1, 2, 1, 2, 1, 2, 1, 2), (2, 2)),
        ),
        (
            (2, 2, 3, 4),
            ((2,), (2,), (1, 2), (2, 2)),
            (12, 4),
            ((1, 2, 1, 2, 1, 2, 1, 2), (2, 2)),
        ),
        # (2, 2, 3, 4) -> (4, 3, 4)
        # TODO: I'm confused about the behavior in this case.
        # (
        #     (2, 2, 3, 4),
        #     ((2,), (2,), (1, 2), (2, 2)),
        #     (4, 3, 4),
        #     ((1, 1, 1, 1), (1, 2), (2, 2)),
        # ),
        # (2, 2, 3, 4) -> (4, 3, 4)
        ((2, 2, 3, 4), ((2,), (2,), (1, 2), (4,)), (4, 3, 4), ((2, 2), (1, 2), (4,))),
    ],
)
def test_reshape_merge_chunks(inshape, inchunks, outshape, outchunks):
    # https://github.com/dask/dask/issues/5544#issuecomment-712280433
    # When the early axes are completely chunked then we are just moving blocks
    # and can avoid any rechunking. The outchunks will always be ...
    base = np.arange(np.prod(inshape)).reshape(inshape)
    a = da.from_array(base, chunks=inchunks)

    # and via reshape
    result = a.reshape(outshape, merge_chunks=False)
    assert result.chunks == outchunks
    assert_eq(result, base.reshape(outshape))

    assert result.chunks != a.reshape(outshape).chunks


def test_smooth_chunks():
    ii = 2
    ileft = 1
    result_inchunks = [None, (1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (20,)]
    result_in = _smooth_chunks(ileft, ii, 2, result_inchunks)
    expected_in = [None, (1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (2, 2, 2, 2, 2, 2, 2, 2, 2, 2)]
    assert result_in == expected_in


def test_smooth_chunks_not_divisible():
    ii = 2
    ileft = 1
    result_inchunks = [None, (1,) * 36, (9,)]
    result_in = _smooth_chunks(ileft, ii, 6, result_inchunks)
    expected_in = [None, (1,) * 36, (4, 5)]
    assert result_in == expected_in


def test_smooth_chunks_first_dimension():
    ii = 1
    ileft = 0
    result_inchunks = [
        (1, 1, 3, 1, 1, 3),
        (6,),
        (
            3,
            3,
        ),
    ]
    result_in = _smooth_chunks(ileft, ii, 12, result_inchunks)
    expected_in = [(1, 1, 1, 2, 1, 1, 1, 2), (6,), (3, 3)]
    assert result_in == expected_in
    result_in = _smooth_chunks(ileft, ii, 6, result_inchunks)
    expected_in = [(1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (6,), (3, 3)]
    assert result_in == expected_in


def test_smooth_chunks_large_dimension():
    # Ensure that we properly smooth 2 dimensions
    ii = 3
    ileft = 1
    result_inchunks = [None, (1, 1), (2,), (400,)]
    result_in = _smooth_chunks(ileft, ii, 4, result_inchunks)
    expected_in = [None, (1, 1), (1, 1), (4,) * 100]
    assert result_in == expected_in


def test_reshape_not_squashing_chunks():
    arr = np.arange(0, 30 * 10 * 20).reshape(30, 10, 20)
    darr = da.from_array(arr, chunks=(30, 2, 1))

    result = darr.reshape(30, 200)

    assert result.chunks == ((30,), (2,) * 100)
    assert np.prod(result.shape) == np.prod(darr.shape)
    assert_eq(result, arr.reshape(30, 200))


@pytest.mark.parametrize(
    "reshaper, chunks",
    [
        ((2, 2, 100), ((1, 1), (1, 1), (20,) * 5)),
        ((2, 200), ((1, 1), (20,) * 10)),
        ((10, 10, 4), ((1,) * 10, (5, 5), (4,))),
        ((2, 25, 8), ((1,) * 2, (2,) * 5 + (3,) * 5, (8,))),
    ],
)
def test_reshape_to_higher_dimension(reshaper, chunks):
    arr = np.arange(0, 4_000).reshape(10, 400)
    darr = da.from_array(arr, chunks=(5, (20,) * 19 + (19, 1)))

    result = darr.reshape(10, *reshaper)

    assert result.chunks == ((5, 5),) + chunks
    assert np.prod(result.shape) == np.prod(darr.shape)
    assert_eq(result, arr.reshape(10, *reshaper))


def test_reshape_lower_dimension():
    arr = da.ones((20, 20, 5), chunks=(10, 10, 5))
    result = arr.reshape(400, 5)
    assert result.chunks == ((100,) * 4, (5,))


def test_reshape_split_out_chunks():
    # too large to assert the actual result
    arr = da.ones(shape=(2, 1000, 100, 4800), chunks=(2, 1000, 100, 83))
    result = arr.reshape(2, 1000, 100, 4, 1200)
    assert result.chunks == (
        (2,),
        (1000,),
        (100,),
        (1, 1, 1, 1),
        (80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80),
    )


def test_reshape_blockwise():
    x = np.arange(0, 54).reshape(6, 3, 3)
    arr = from_array(x, chunks=(3, 2, (2, 1)))
    assert arr.chunks == ((3, 3), (2, 1), (2, 1))
    result = reshape_blockwise(arr, (18, 3))
    assert result.chunks == ((6, 3, 6, 3), (2, 1))
    expected = [
        np.concatenate(
            [x[:3, :2, :2].reshape(6, 2), x[:3, :2, 2:].reshape(6, 1)], axis=1
        ),
        np.concatenate(
            [x[:3, 2:, :2].reshape(3, 2), x[:3, 2:, 2:].reshape(3, 1)], axis=1
        ),
        np.concatenate(
            [x[3:, :2, :2].reshape(6, 2), x[3:, :2, 2:].reshape(6, 1)], axis=1
        ),
        np.concatenate(
            [x[3:, 2:, :2].reshape(3, 2), x[3:, 2:, 2:].reshape(3, 1)], axis=1
        ),
    ]
    expected = np.concatenate(expected, axis=0)
    assert_eq(result, expected)
    result2 = reshape_blockwise(result, (6, 3, 3), chunks=arr.chunks)
    assert_eq(arr, result2)
    assert result2.chunks == arr.chunks

    result = reshape_blockwise(arr, (6, 9))
    assert result.chunks == ((3, 3), (4, 2, 2, 1))
    expected = [
        np.concatenate(
            [
                x[:3, :2, :2].reshape(3, 4),
                x[:3, :2, 2:].reshape(3, 2),
                x[:3, 2:, :2].reshape(3, 2),
                x[:3, 2:, 2:].reshape(3, 1),
            ],
            axis=1,
        ),
        np.concatenate(
            [
                x[3:, :2, :2].reshape(3, 4),
                x[3:, :2, 2:].reshape(3, 2),
                x[3:, 2:, :2].reshape(3, 2),
                x[3:, 2:, 2:].reshape(3, 1),
            ],
            axis=1,
        ),
    ]

    # chunks of first half of the array, i.e. arr[:3, :, :]:
    # [[[0, 1], [3, 4]], [[9, 10], [12, 13]], [[18, 19], [21, 22]]]
    # [[[2], [5]], [[11], [14]], [[20], [23]]]
    # [[[6, 7], [15, 16]], [[24, 25]]]
    # [[[8], [17]], [[26]]]
    # The inner dimensions are squashed, i.e.
    # [[ 0  1  3  4  2  5  6  7  8]
    #  [ 9 10 12 13 11 14 15 16 17]
    #  [18 19 21 22 20 23 24 25 26]]
    expected = np.concatenate(expected, axis=0)
    assert_eq(result, expected)
    result2 = reshape_blockwise(result, (6, 3, 3), chunks=arr.chunks)
    assert_eq(arr, result2)
    assert result2.chunks == arr.chunks

    with pytest.raises(
        ValueError, match="Chunk sizes do not match for the following chunks"
    ):
        reshape_blockwise(result, (6, 3, 3), chunks=((3, 4), (3, 1), (3, 1)))


def test_reshape_blockwise_raises_for_expansion():
    x = np.arange(0, 54).reshape(6, 3, 3)
    arr = from_array(x, chunks=(3, 2, (2, 1)))
    msg = "Need to specify chunks if expanding dimensions."
    with pytest.raises(TypeError, match=msg):
        reshape_blockwise(arr, (2, 3, 3, 3))

    with pytest.raises(
        NotImplementedError,
        match="reshape_blockwise not implemented for expanding dimensions",
    ):
        reshape_blockwise(arr, (3, 2, 9))


def test_reshape_blockwise_dimension_reduction_and_chunks():
    x = np.arange(0, 54).reshape(6, 3, 3)
    arr = from_array(x, chunks=(3, 2, (2, 1)))
    with pytest.raises(ValueError, match="Setting chunks is not allowed when"):
        reshape_blockwise(arr, (18, 3), arr.chunks)


def test_reshape_blockwise_identity():
    x = np.arange(0, 54).reshape(6, 3, 3)
    arr = from_array(x, chunks=(3, 2, (2, 1)))
    result = reshape_blockwise(arr, (6, 3, 3))
    assert len(arr.dask) == len(result.dask)
    assert_eq(result, x)

    result = reshape_blockwise(arr, (6, 3, 3), chunks=arr.chunks)
    assert len(arr.dask) == len(result.dask)
    assert_eq(result, x)


def test_argwhere_reshaping_not_concatenating_chunks():
    # GH#10080
    arr = da.random.random((500, 500, 500), chunks=(100, 100, 100)) < 0
    result = da.argwhere(arr)
    assert result.chunks == ((np.nan,) * 125, (1, 1, 1))


def test_blockwise_reshape_sanity_checks():
    x = np.arange(0, 20).reshape(10, 2)
    arr = from_array(x, chunks=(5, 1))
    with pytest.raises(ValueError, match="total size of new array must be unchanged"):
        reshape_blockwise(arr, (10, 6))

    arr = from_array(x, chunks=(5, 1))
    arr = arr[arr[:, 0] > 0]
    with pytest.raises(ValueError, match="Array chunk size or shape is unknown"):
        reshape_blockwise(arr, (5, 2, 2))


@pytest.mark.parametrize("shape", [(-1, 0), (0, -2), (-1, -2)])
def test_reshape_empty_ambiguous_inference(shape):
    arr = da.ones((2, 0, 0), chunks=(1, 0, 0))
    with pytest.raises(ValueError):
        arr.reshape(shape)


@pytest.mark.parametrize("shape", [(-2,), (2, -2)])
def test_reshape_negative_inferred_dimension(shape):
    expected = np.arange(6)
    arr = da.from_array(expected, chunks=2)
    assert_eq(arr.reshape(shape), expected.reshape(shape))


@pytest.mark.parametrize("shape", [-1, np.int64(-1)])
def test_reshape_function_scalar_shape(shape):
    expected = np.arange(6).reshape((2, 3))
    arr = da.from_array(expected, chunks=1)
    assert_eq(da.reshape(arr, shape), expected.reshape(shape))

    arr = arr[arr[:, 0] > 0]
    assert_eq(da.reshape(arr, shape), expected[expected[:, 0] > 0].reshape(shape))


def test_reshape_inferred_dimension_integer_precision():
    size = 2**53 + 1
    arr = da.empty((size,), chunks=-1)
    assert arr.reshape((-1, 1)).shape == (size, 1)


@pytest.mark.parametrize("axes", [(0,), (0, 1), (0, 2)])
@pytest.mark.parametrize("unknown_axis", [0, 1])
def test_flatten_unknown_chunks_leading_singletons(axes, unknown_axis):
    expected = np.arange(120).reshape(10, 12)
    arr = da.from_array(expected, chunks=(2, 3))
    mask = np.arange(expected.shape[unknown_axis]) % 3 != 0
    key = (slice(None),) * unknown_axis + (da.from_array(mask, chunks=2),)
    arr = arr[key]
    expected = expected[(slice(None),) * unknown_axis + (mask,)]
    expanded = da.expand_dims(arr, axes)
    result = expanded.ravel()

    # Adding leading singleton axes must not merge every row slab into one task.
    assert len(result.chunks[0]) == len(arr.chunks[0])
    assert not result.dask.layers[result.name].is_materialized()
    assert_eq(result, expected.ravel())


@pytest.mark.parametrize("shape, unknown_axis", [((3, 0), 0), ((0, 3), 1)])
def test_flatten_unknown_chunks_known_empty(shape, unknown_axis):
    expected = np.empty(shape)
    arr = da.from_array(expected, chunks=2)
    mask = da.from_array(np.ones(3, dtype=bool), chunks=2)
    arr = arr[(slice(None),) * unknown_axis + (mask,)]
    result = da.expand_dims(arr, 0).ravel()

    assert result.shape == (0,)
    assert all(c == 0 for c in result.chunks[0])
    assert_eq(result, expected.ravel())
