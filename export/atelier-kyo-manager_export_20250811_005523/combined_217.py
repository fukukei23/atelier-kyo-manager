
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\reshaping.py ===
import itertools

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import EABackedBlock


class BaseReshapingTests:
    """Tests for reshaping and concatenation."""

    @pytest.mark.parametrize("in_frame", [True, False])
    def test_concat(self, data, in_frame):
        wrapped = pd.Series(data)
        if in_frame:
            wrapped = pd.DataFrame(wrapped)
        result = pd.concat([wrapped, wrapped], ignore_index=True)

        assert len(result) == len(data) * 2

        if in_frame:
            dtype = result.dtypes[0]
        else:
            dtype = result.dtype

        assert dtype == data.dtype
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result._mgr.blocks[0], EABackedBlock)
        assert isinstance(result._mgr.arrays[0], ExtensionArray)

    @pytest.mark.parametrize("in_frame", [True, False])
    def test_concat_all_na_block(self, data_missing, in_frame):
        valid_block = pd.Series(data_missing.take([1, 1]), index=[0, 1])
        na_block = pd.Series(data_missing.take([0, 0]), index=[2, 3])
        if in_frame:
            valid_block = pd.DataFrame({"a": valid_block})
            na_block = pd.DataFrame({"a": na_block})
        result = pd.concat([valid_block, na_block])
        if in_frame:
            expected = pd.DataFrame({"a": data_missing.take([1, 1, 0, 0])})
            tm.assert_frame_equal(result, expected)
        else:
            expected = pd.Series(data_missing.take([1, 1, 0, 0]))
            tm.assert_series_equal(result, expected)

    def test_concat_mixed_dtypes(self, data):
        # https://github.com/pandas-dev/pandas/issues/20762
        df1 = pd.DataFrame({"A": data[:3]})
        df2 = pd.DataFrame({"A": [1, 2, 3]})
        df3 = pd.DataFrame({"A": ["a", "b", "c"]}).astype("category")
        dfs = [df1, df2, df3]

        # dataframes
        result = pd.concat(dfs)
        expected = pd.concat([x.astype(object) for x in dfs])
        tm.assert_frame_equal(result, expected)

        # series
        result = pd.concat([x["A"] for x in dfs])
        expected = pd.concat([x["A"].astype(object) for x in dfs])
        tm.assert_series_equal(result, expected)

        # simple test for just EA and one other
        result = pd.concat([df1, df2.astype(object)])
        expected = pd.concat([df1.astype("object"), df2.astype("object")])
        tm.assert_frame_equal(result, expected)

        result = pd.concat([df1["A"], df2["A"].astype(object)])
        expected = pd.concat([df1["A"].astype("object"), df2["A"].astype("object")])
        tm.assert_series_equal(result, expected)

    def test_concat_columns(self, data, na_value):
        df1 = pd.DataFrame({"A": data[:3]})
        df2 = pd.DataFrame({"B": [1, 2, 3]})

        expected = pd.DataFrame({"A": data[:3], "B": [1, 2, 3]})
        result = pd.concat([df1, df2], axis=1)
        tm.assert_frame_equal(result, expected)
        result = pd.concat([df1["A"], df2["B"]], axis=1)
        tm.assert_frame_equal(result, expected)

        # non-aligned
        df2 = pd.DataFrame({"B": [1, 2, 3]}, index=[1, 2, 3])
        expected = pd.DataFrame(
            {
                "A": data._from_sequence(list(data[:3]) + [na_value], dtype=data.dtype),
                "B": [np.nan, 1, 2, 3],
            }
        )

        result = pd.concat([df1, df2], axis=1)
        tm.assert_frame_equal(result, expected)
        result = pd.concat([df1["A"], df2["B"]], axis=1)
        tm.assert_frame_equal(result, expected)

    def test_concat_extension_arrays_copy_false(self, data, na_value):
        # GH 20756
        df1 = pd.DataFrame({"A": data[:3]})
        df2 = pd.DataFrame({"B": data[3:7]})
        expected = pd.DataFrame(
            {
                "A": data._from_sequence(list(data[:3]) + [na_value], dtype=data.dtype),
                "B": data[3:7],
            }
        )
        result = pd.concat([df1, df2], axis=1, copy=False)
        tm.assert_frame_equal(result, expected)

    def test_concat_with_reindex(self, data):
        # GH-33027
        a = pd.DataFrame({"a": data[:5]})
        b = pd.DataFrame({"b": data[:5]})
        result = pd.concat([a, b], ignore_index=True)
        expected = pd.DataFrame(
            {
                "a": data.take(list(range(5)) + ([-1] * 5), allow_fill=True),
                "b": data.take(([-1] * 5) + list(range(5)), allow_fill=True),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_align(self, data, na_value):
        a = data[:3]
        b = data[2:5]
        r1, r2 = pd.Series(a).align(pd.Series(b, index=[1, 2, 3]))

        # Assumes that the ctor can take a list of scalars of the type
        e1 = pd.Series(data._from_sequence(list(a) + [na_value], dtype=data.dtype))
        e2 = pd.Series(data._from_sequence([na_value] + list(b), dtype=data.dtype))
        tm.assert_series_equal(r1, e1)
        tm.assert_series_equal(r2, e2)

    def test_align_frame(self, data, na_value):
        a = data[:3]
        b = data[2:5]
        r1, r2 = pd.DataFrame({"A": a}).align(pd.DataFrame({"A": b}, index=[1, 2, 3]))

        # Assumes that the ctor can take a list of scalars of the type
        e1 = pd.DataFrame(
            {"A": data._from_sequence(list(a) + [na_value], dtype=data.dtype)}
        )
        e2 = pd.DataFrame(
            {"A": data._from_sequence([na_value] + list(b), dtype=data.dtype)}
        )
        tm.assert_frame_equal(r1, e1)
        tm.assert_frame_equal(r2, e2)

    def test_align_series_frame(self, data, na_value):
        # https://github.com/pandas-dev/pandas/issues/20576
        ser = pd.Series(data, name="a")
        df = pd.DataFrame({"col": np.arange(len(ser) + 1)})
        r1, r2 = ser.align(df)

        e1 = pd.Series(
            data._from_sequence(list(data) + [na_value], dtype=data.dtype),
            name=ser.name,
        )

        tm.assert_series_equal(r1, e1)
        tm.assert_frame_equal(r2, df)

    def test_set_frame_expand_regular_with_extension(self, data):
        df = pd.DataFrame({"A": [1] * len(data)})
        df["B"] = data
        expected = pd.DataFrame({"A": [1] * len(data), "B": data})
        tm.assert_frame_equal(df, expected)

    def test_set_frame_expand_extension_with_regular(self, data):
        df = pd.DataFrame({"A": data})
        df["B"] = [1] * len(data)
        expected = pd.DataFrame({"A": data, "B": [1] * len(data)})
        tm.assert_frame_equal(df, expected)

    def test_set_frame_overwrite_object(self, data):
        # https://github.com/pandas-dev/pandas/issues/20555
        df = pd.DataFrame({"A": [1] * len(data)}, dtype=object)
        df["A"] = data
        assert df.dtypes["A"] == data.dtype

    def test_merge(self, data, na_value):
        # GH-20743
        df1 = pd.DataFrame({"ext": data[:3], "int1": [1, 2, 3], "key": [0, 1, 2]})
        df2 = pd.DataFrame({"int2": [1, 2, 3, 4], "key": [0, 0, 1, 3]})

        res = pd.merge(df1, df2)
        exp = pd.DataFrame(
            {
                "int1": [1, 1, 2],
                "int2": [1, 2, 3],
                "key": [0, 0, 1],
                "ext": data._from_sequence(
                    [data[0], data[0], data[1]], dtype=data.dtype
                ),
            }
        )
        tm.assert_frame_equal(res, exp[["ext", "int1", "key", "int2"]])

        res = pd.merge(df1, df2, how="outer")
        exp = pd.DataFrame(
            {
                "int1": [1, 1, 2, 3, np.nan],
                "int2": [1, 2, 3, np.nan, 4],
                "key": [0, 0, 1, 2, 3],
                "ext": data._from_sequence(
                    [data[0], data[0], data[1], data[2], na_value], dtype=data.dtype
                ),
            }
        )
        tm.assert_frame_equal(res, exp[["ext", "int1", "key", "int2"]])

    def test_merge_on_extension_array(self, data):
        # GH 23020
        a, b = data[:2]
        key = type(data)._from_sequence([a, b], dtype=data.dtype)

        df = pd.DataFrame({"key": key, "val": [1, 2]})
        result = pd.merge(df, df, on="key")
        expected = pd.DataFrame({"key": key, "val_x": [1, 2], "val_y": [1, 2]})
        tm.assert_frame_equal(result, expected)

        # order
        result = pd.merge(df.iloc[[1, 0]], df, on="key")
        expected = expected.iloc[[1, 0]].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_merge_on_extension_array_duplicates(self, data):
        # GH 23020
        a, b = data[:2]
        key = type(data)._from_sequence([a, b, a], dtype=data.dtype)
        df1 = pd.DataFrame({"key": key, "val": [1, 2, 3]})
        df2 = pd.DataFrame({"key": key, "val": [1, 2, 3]})

        result = pd.merge(df1, df2, on="key")
        expected = pd.DataFrame(
            {
                "key": key.take([0, 0, 1, 2, 2]),
                "val_x": [1, 1, 2, 3, 3],
                "val_y": [1, 3, 2, 1, 3],
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    @pytest.mark.parametrize(
        "columns",
        [
            ["A", "B"],
            pd.MultiIndex.from_tuples(
                [("A", "a"), ("A", "b")], names=["outer", "inner"]
            ),
        ],
    )
    @pytest.mark.parametrize("future_stack", [True, False])
    def test_stack(self, data, columns, future_stack):
        df = pd.DataFrame({"A": data[:5], "B": data[:5]})
        df.columns = columns
        result = df.stack(future_stack=future_stack)
        expected = df.astype(object).stack(future_stack=future_stack)
        # we need a second astype(object), in case the constructor inferred
        # object -> specialized, as is done for period.
        expected = expected.astype(object)

        if isinstance(expected, pd.Series):
            assert result.dtype == df.iloc[:, 0].dtype
        else:
            assert all(result.dtypes == df.iloc[:, 0].dtype)

        result = result.astype(object)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "index",
        [
            # Two levels, uniform.
            pd.MultiIndex.from_product(([["A", "B"], ["a", "b"]]), names=["a", "b"]),
            # non-uniform
            pd.MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "b")]),
            # three levels, non-uniform
            pd.MultiIndex.from_product([("A", "B"), ("a", "b", "c"), (0, 1, 2)]),
            pd.MultiIndex.from_tuples(
                [
                    ("A", "a", 1),
                    ("A", "b", 0),
                    ("A", "a", 0),
                    ("B", "a", 0),
                    ("B", "c", 1),
                ]
            ),
        ],
    )
    @pytest.mark.parametrize("obj", ["series", "frame"])
    def test_unstack(self, data, index, obj):
        data = data[: len(index)]
        if obj == "series":
            ser = pd.Series(data, index=index)
        else:
            ser = pd.DataFrame({"A": data, "B": data}, index=index)

        n = index.nlevels
        levels = list(range(n))
        # [0, 1, 2]
        # [(0,), (1,), (2,), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        combinations = itertools.chain.from_iterable(
            itertools.permutations(levels, i) for i in range(1, n)
        )

        for level in combinations:
            result = ser.unstack(level=level)
            assert all(
                isinstance(result[col].array, type(data)) for col in result.columns
            )

            if obj == "series":
                # We should get the same result with to_frame+unstack+droplevel
                df = ser.to_frame()

                alt = df.unstack(level=level).droplevel(0, axis=1)
                tm.assert_frame_equal(result, alt)

            obj_ser = ser.astype(object)

            expected = obj_ser.unstack(level=level, fill_value=data.dtype.na_value)
            if obj == "series":
                assert (expected.dtypes == object).all()

            result = result.astype(object)
            tm.assert_frame_equal(result, expected)

    def test_ravel(self, data):
        # as long as EA is 1D-only, ravel is a no-op
        result = data.ravel()
        assert type(result) == type(data)

        if data.dtype._is_immutable:
            pytest.skip(f"test_ravel assumes mutability and {data.dtype} is immutable")

        # Check that we have a view, not a copy
        result[0] = result[1]
        assert data[0] == data[1]

    def test_transpose(self, data):
        result = data.transpose()
        assert type(result) == type(data)

        # check we get a new object
        assert result is not data

        # If we ever _did_ support 2D, shape should be reversed
        assert result.shape == data.shape[::-1]

        if data.dtype._is_immutable:
            pytest.skip(
                f"test_transpose assumes mutability and {data.dtype} is immutable"
            )

        # Check that we have a view, not a copy
        result[0] = result[1]
        assert data[0] == data[1]

    def test_transpose_frame(self, data):
        df = pd.DataFrame({"A": data[:4], "B": data[:4]}, index=["a", "b", "c", "d"])
        result = df.T
        expected = pd.DataFrame(
            {
                "a": type(data)._from_sequence([data[0]] * 2, dtype=data.dtype),
                "b": type(data)._from_sequence([data[1]] * 2, dtype=data.dtype),
                "c": type(data)._from_sequence([data[2]] * 2, dtype=data.dtype),
                "d": type(data)._from_sequence([data[3]] * 2, dtype=data.dtype),
            },
            index=["A", "B"],
        )
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(np.transpose(np.transpose(df)), df)
        tm.assert_frame_equal(np.transpose(np.transpose(df[["A"]])), df[["A"]])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\test_compression.py ===
import gzip
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap
import time
import zipfile

import numpy as np
import pytest

from pandas.compat import is_platform_windows

import pandas as pd
import pandas._testing as tm

import pandas.io.common as icom


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_pickle", "to_json", "to_csv"])
def test_compression_size(obj, method, compression_only):
    if compression_only == "tar":
        compression_only = {"method": "tar", "mode": "w:gz"}

    with tm.ensure_clean() as path:
        getattr(obj, method)(path, compression=compression_only)
        compressed_size = os.path.getsize(path)
        getattr(obj, method)(path, compression=None)
        uncompressed_size = os.path.getsize(path)
        assert uncompressed_size > compressed_size


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_csv", "to_json"])
def test_compression_size_fh(obj, method, compression_only):
    with tm.ensure_clean() as path:
        with icom.get_handle(
            path,
            "w:gz" if compression_only == "tar" else "w",
            compression=compression_only,
        ) as handles:
            getattr(obj, method)(handles.handle)
            assert not handles.handle.closed
        compressed_size = os.path.getsize(path)
    with tm.ensure_clean() as path:
        with icom.get_handle(path, "w", compression=None) as handles:
            getattr(obj, method)(handles.handle)
            assert not handles.handle.closed
        uncompressed_size = os.path.getsize(path)
        assert uncompressed_size > compressed_size


@pytest.mark.parametrize(
    "write_method, write_kwargs, read_method",
    [
        ("to_csv", {"index": False}, pd.read_csv),
        ("to_json", {}, pd.read_json),
        ("to_pickle", {}, pd.read_pickle),
    ],
)
def test_dataframe_compression_defaults_to_infer(
    write_method, write_kwargs, read_method, compression_only, compression_to_extension
):
    # GH22004
    input = pd.DataFrame([[1.0, 0, -4], [3.4, 5, 2]], columns=["X", "Y", "Z"])
    extension = compression_to_extension[compression_only]
    with tm.ensure_clean("compressed" + extension) as path:
        getattr(input, write_method)(path, **write_kwargs)
        output = read_method(path, compression=compression_only)
    tm.assert_frame_equal(output, input)


@pytest.mark.parametrize(
    "write_method,write_kwargs,read_method,read_kwargs",
    [
        ("to_csv", {"index": False, "header": True}, pd.read_csv, {"squeeze": True}),
        ("to_json", {}, pd.read_json, {"typ": "series"}),
        ("to_pickle", {}, pd.read_pickle, {}),
    ],
)
def test_series_compression_defaults_to_infer(
    write_method,
    write_kwargs,
    read_method,
    read_kwargs,
    compression_only,
    compression_to_extension,
):
    # GH22004
    input = pd.Series([0, 5, -2, 10], name="X")
    extension = compression_to_extension[compression_only]
    with tm.ensure_clean("compressed" + extension) as path:
        getattr(input, write_method)(path, **write_kwargs)
        if "squeeze" in read_kwargs:
            kwargs = read_kwargs.copy()
            del kwargs["squeeze"]
            output = read_method(path, compression=compression_only, **kwargs).squeeze(
                "columns"
            )
        else:
            output = read_method(path, compression=compression_only, **read_kwargs)
    tm.assert_series_equal(output, input, check_names=False)


def test_compression_warning(compression_only):
    # Assert that passing a file object to to_csv while explicitly specifying a
    # compression protocol triggers a RuntimeWarning, as per GH21227.
    df = pd.DataFrame(
        100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
        columns=["X", "Y", "Z"],
    )
    with tm.ensure_clean() as path:
        with icom.get_handle(path, "w", compression=compression_only) as handles:
            with tm.assert_produces_warning(RuntimeWarning):
                df.to_csv(handles.handle, compression=compression_only)


def test_compression_binary(compression_only):
    """
    Binary file handles support compression.

    GH22555
    """
    df = pd.DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=pd.Index(list("ABCD")),
        index=pd.Index([f"i-{i}" for i in range(30)]),
    )

    # with a file
    with tm.ensure_clean() as path:
        with open(path, mode="wb") as file:
            df.to_csv(file, mode="wb", compression=compression_only)
            file.seek(0)  # file shouldn't be closed
        tm.assert_frame_equal(
            df, pd.read_csv(path, index_col=0, compression=compression_only)
        )

    # with BytesIO
    file = io.BytesIO()
    df.to_csv(file, mode="wb", compression=compression_only)
    file.seek(0)  # file shouldn't be closed
    tm.assert_frame_equal(
        df, pd.read_csv(file, index_col=0, compression=compression_only)
    )


def test_gzip_reproducibility_file_name():
    """
    Gzip should create reproducible archives with mtime.

    Note: Archives created with different filenames will still be different!

    GH 28103
    """
    df = pd.DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=pd.Index(list("ABCD")),
        index=pd.Index([f"i-{i}" for i in range(30)]),
    )
    compression_options = {"method": "gzip", "mtime": 1}

    # test for filename
    with tm.ensure_clean() as path:
        path = Path(path)
        df.to_csv(path, compression=compression_options)
        time.sleep(0.1)
        output = path.read_bytes()
        df.to_csv(path, compression=compression_options)
        assert output == path.read_bytes()


def test_gzip_reproducibility_file_object():
    """
    Gzip should create reproducible archives with mtime.

    GH 28103
    """
    df = pd.DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=pd.Index(list("ABCD")),
        index=pd.Index([f"i-{i}" for i in range(30)]),
    )
    compression_options = {"method": "gzip", "mtime": 1}

    # test for file object
    buffer = io.BytesIO()
    df.to_csv(buffer, compression=compression_options, mode="wb")
    output = buffer.getvalue()
    time.sleep(0.1)
    buffer = io.BytesIO()
    df.to_csv(buffer, compression=compression_options, mode="wb")
    assert output == buffer.getvalue()


@pytest.mark.single_cpu
def test_with_missing_lzma():
    """Tests if import pandas works when lzma is not present."""
    # https://github.com/pandas-dev/pandas/issues/27575
    code = textwrap.dedent(
        """\
        import sys
        sys.modules['lzma'] = None
        import pandas
        """
    )
    subprocess.check_output([sys.executable, "-c", code], stderr=subprocess.PIPE)


@pytest.mark.single_cpu
def test_with_missing_lzma_runtime():
    """Tests if RuntimeError is hit when calling lzma without
    having the module available.
    """
    code = textwrap.dedent(
        """
        import sys
        import pytest
        sys.modules['lzma'] = None
        import pandas as pd
        df = pd.DataFrame()
        with pytest.raises(RuntimeError, match='lzma module'):
            df.to_csv('foo.csv', compression='xz')
        """
    )
    subprocess.check_output([sys.executable, "-c", code], stderr=subprocess.PIPE)


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_pickle", "to_json", "to_csv"])
def test_gzip_compression_level(obj, method):
    # GH33196
    with tm.ensure_clean() as path:
        getattr(obj, method)(path, compression="gzip")
        compressed_size_default = os.path.getsize(path)
        getattr(obj, method)(path, compression={"method": "gzip", "compresslevel": 1})
        compressed_size_fast = os.path.getsize(path)
        assert compressed_size_default < compressed_size_fast


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_pickle", "to_json", "to_csv"])
def test_xz_compression_level_read(obj, method):
    with tm.ensure_clean() as path:
        getattr(obj, method)(path, compression="xz")
        compressed_size_default = os.path.getsize(path)
        getattr(obj, method)(path, compression={"method": "xz", "preset": 1})
        compressed_size_fast = os.path.getsize(path)
        assert compressed_size_default < compressed_size_fast
        if method == "to_csv":
            pd.read_csv(path, compression="xz")


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_pickle", "to_json", "to_csv"])
def test_bzip_compression_level(obj, method):
    """GH33196 bzip needs file size > 100k to show a size difference between
    compression levels, so here we just check if the call works when
    compression is passed as a dict.
    """
    with tm.ensure_clean() as path:
        getattr(obj, method)(path, compression={"method": "bz2", "compresslevel": 1})


@pytest.mark.parametrize(
    "suffix,archive",
    [
        (".zip", zipfile.ZipFile),
        (".tar", tarfile.TarFile),
    ],
)
def test_empty_archive_zip(suffix, archive):
    with tm.ensure_clean(filename=suffix) as path:
        with archive(path, "w"):
            pass
        with pytest.raises(ValueError, match="Zero files found"):
            pd.read_csv(path)


def test_ambiguous_archive_zip():
    with tm.ensure_clean(filename=".zip") as path:
        with zipfile.ZipFile(path, "w") as file:
            file.writestr("a.csv", "foo,bar")
            file.writestr("b.csv", "foo,bar")
        with pytest.raises(ValueError, match="Multiple files found in ZIP file"):
            pd.read_csv(path)


def test_ambiguous_archive_tar(tmp_path):
    csvAPath = tmp_path / "a.csv"
    with open(csvAPath, "w", encoding="utf-8") as a:
        a.write("foo,bar\n")
    csvBPath = tmp_path / "b.csv"
    with open(csvBPath, "w", encoding="utf-8") as b:
        b.write("foo,bar\n")

    tarpath = tmp_path / "archive.tar"
    with tarfile.TarFile(tarpath, "w") as tar:
        tar.add(csvAPath, "a.csv")
        tar.add(csvBPath, "b.csv")

    with pytest.raises(ValueError, match="Multiple files found in TAR archive"):
        pd.read_csv(tarpath)


def test_tar_gz_to_different_filename():
    with tm.ensure_clean(filename=".foo") as file:
        pd.DataFrame(
            [["1", "2"]],
            columns=["foo", "bar"],
        ).to_csv(file, compression={"method": "tar", "mode": "w:gz"}, index=False)
        with gzip.open(file) as uncompressed:
            with tarfile.TarFile(fileobj=uncompressed) as archive:
                members = archive.getmembers()
                assert len(members) == 1
                content = archive.extractfile(members[0]).read().decode("utf8")

                if is_platform_windows():
                    expected = "foo,bar\r\n1,2\r\n"
                else:
                    expected = "foo,bar\n1,2\n"

                assert content == expected


def test_tar_no_error_on_close():
    with io.BytesIO() as buffer:
        with icom._BytesTarFile(fileobj=buffer, mode="w"):
            pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\test_timezones.py ===
from datetime import (
    date,
    timedelta,
)

import numpy as np
import pytest

from pandas._libs.tslibs.timezones import maybe_get_tz
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
    _maybe_remove,
    ensure_clean_store,
)


def _compare_with_tz(a, b):
    tm.assert_frame_equal(a, b)

    # compare the zones on each element
    for c in a.columns:
        for i in a.index:
            a_e = a.loc[i, c]
            b_e = b.loc[i, c]
            if not (a_e == b_e and a_e.tz == b_e.tz):
                raise AssertionError(f"invalid tz comparison [{a_e}] [{b_e}]")


# use maybe_get_tz instead of dateutil.tz.gettz to handle the windows
# filename issues.
gettz_dateutil = lambda x: maybe_get_tz("dateutil/" + x)
gettz_pytz = lambda x: x


@pytest.mark.parametrize("gettz", [gettz_dateutil, gettz_pytz])
def test_append_with_timezones(setup_path, gettz):
    # as columns

    # Single-tzinfo, no DST transition
    df_est = DataFrame(
        {
            "A": [
                Timestamp("20130102 2:00:00", tz=gettz("US/Eastern")).as_unit("ns")
                + timedelta(hours=1) * i
                for i in range(5)
            ]
        }
    )

    # frame with all columns having same tzinfo, but different sides
    #  of DST transition
    df_crosses_dst = DataFrame(
        {
            "A": Timestamp("20130102", tz=gettz("US/Eastern")).as_unit("ns"),
            "B": Timestamp("20130603", tz=gettz("US/Eastern")).as_unit("ns"),
        },
        index=range(5),
    )

    df_mixed_tz = DataFrame(
        {
            "A": Timestamp("20130102", tz=gettz("US/Eastern")).as_unit("ns"),
            "B": Timestamp("20130102", tz=gettz("EET")).as_unit("ns"),
        },
        index=range(5),
    )

    df_different_tz = DataFrame(
        {
            "A": Timestamp("20130102", tz=gettz("US/Eastern")).as_unit("ns"),
            "B": Timestamp("20130102", tz=gettz("CET")).as_unit("ns"),
        },
        index=range(5),
    )

    with ensure_clean_store(setup_path) as store:
        _maybe_remove(store, "df_tz")
        store.append("df_tz", df_est, data_columns=["A"])
        result = store["df_tz"]
        _compare_with_tz(result, df_est)
        tm.assert_frame_equal(result, df_est)

        # select with tz aware
        expected = df_est[df_est.A >= df_est.A[3]]
        result = store.select("df_tz", where="A>=df_est.A[3]")
        _compare_with_tz(result, expected)

        # ensure we include dates in DST and STD time here.
        _maybe_remove(store, "df_tz")
        store.append("df_tz", df_crosses_dst)
        result = store["df_tz"]
        _compare_with_tz(result, df_crosses_dst)
        tm.assert_frame_equal(result, df_crosses_dst)

        msg = (
            r"invalid info for \[values_block_1\] for \[tz\], "
            r"existing_value \[(dateutil/.*)?(US/Eastern|America/New_York)\] "
            r"conflicts with new value \[(dateutil/.*)?EET\]"
        )
        with pytest.raises(ValueError, match=msg):
            store.append("df_tz", df_mixed_tz)

        # this is ok
        _maybe_remove(store, "df_tz")
        store.append("df_tz", df_mixed_tz, data_columns=["A", "B"])
        result = store["df_tz"]
        _compare_with_tz(result, df_mixed_tz)
        tm.assert_frame_equal(result, df_mixed_tz)

        # can't append with diff timezone
        msg = (
            r"invalid info for \[B\] for \[tz\], "
            r"existing_value \[(dateutil/.*)?EET\] "
            r"conflicts with new value \[(dateutil/.*)?CET\]"
        )
        with pytest.raises(ValueError, match=msg):
            store.append("df_tz", df_different_tz)


@pytest.mark.parametrize("gettz", [gettz_dateutil, gettz_pytz])
def test_append_with_timezones_as_index(setup_path, gettz):
    # GH#4098 example

    dti = date_range("2000-1-1", periods=3, freq="h", tz=gettz("US/Eastern"))
    dti = dti._with_freq(None)  # freq doesn't round-trip

    df = DataFrame({"A": Series(range(3), index=dti)})

    with ensure_clean_store(setup_path) as store:
        _maybe_remove(store, "df")
        store.put("df", df)
        result = store.select("df")
        tm.assert_frame_equal(result, df)

        _maybe_remove(store, "df")
        store.append("df", df)
        result = store.select("df")
        tm.assert_frame_equal(result, df)


def test_roundtrip_tz_aware_index(setup_path, unit):
    # GH 17618
    ts = Timestamp("2000-01-01 01:00:00", tz="US/Eastern")
    dti = DatetimeIndex([ts]).as_unit(unit)
    df = DataFrame(data=[0], index=dti)

    with ensure_clean_store(setup_path) as store:
        store.put("frame", df, format="fixed")
        recons = store["frame"]
        tm.assert_frame_equal(recons, df)

    value = recons.index[0]._value
    denom = {"ns": 1, "us": 1000, "ms": 10**6, "s": 10**9}[unit]
    assert value == 946706400000000000 // denom


def test_store_index_name_with_tz(setup_path):
    # GH 13884
    df = DataFrame({"A": [1, 2]})
    df.index = DatetimeIndex([1234567890123456787, 1234567890123456788])
    df.index = df.index.tz_localize("UTC")
    df.index.name = "foo"

    with ensure_clean_store(setup_path) as store:
        store.put("frame", df, format="table")
        recons = store["frame"]
        tm.assert_frame_equal(recons, df)


def test_tseries_select_index_column(setup_path):
    # GH7777
    # selecting a UTC datetimeindex column did
    # not preserve UTC tzinfo set before storing

    # check that no tz still works
    rng = date_range("1/1/2000", "1/30/2000")
    frame = DataFrame(
        np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng
    )

    with ensure_clean_store(setup_path) as store:
        store.append("frame", frame)
        result = store.select_column("frame", "index")
        assert rng.tz == DatetimeIndex(result.values).tz

    # check utc
    rng = date_range("1/1/2000", "1/30/2000", tz="UTC")
    frame = DataFrame(
        np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng
    )

    with ensure_clean_store(setup_path) as store:
        store.append("frame", frame)
        result = store.select_column("frame", "index")
        assert rng.tz == result.dt.tz

    # double check non-utc
    rng = date_range("1/1/2000", "1/30/2000", tz="US/Eastern")
    frame = DataFrame(
        np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng
    )

    with ensure_clean_store(setup_path) as store:
        store.append("frame", frame)
        result = store.select_column("frame", "index")
        assert rng.tz == result.dt.tz


def test_timezones_fixed_format_frame_non_empty(setup_path):
    with ensure_clean_store(setup_path) as store:
        # index
        rng = date_range("1/1/2000", "1/30/2000", tz="US/Eastern")
        rng = rng._with_freq(None)  # freq doesn't round-trip
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng
        )
        store["df"] = df
        result = store["df"]
        tm.assert_frame_equal(result, df)

        # as data
        # GH11411
        _maybe_remove(store, "df")
        df = DataFrame(
            {
                "A": rng,
                "B": rng.tz_convert("UTC").tz_localize(None),
                "C": rng.tz_convert("CET"),
                "D": range(len(rng)),
            },
            index=rng,
        )
        store["df"] = df
        result = store["df"]
        tm.assert_frame_equal(result, df)


def test_timezones_fixed_format_empty(setup_path, tz_aware_fixture, frame_or_series):
    # GH 20594

    dtype = pd.DatetimeTZDtype(tz=tz_aware_fixture)

    obj = Series(dtype=dtype, name="A")
    if frame_or_series is DataFrame:
        obj = obj.to_frame()

    with ensure_clean_store(setup_path) as store:
        store["obj"] = obj
        result = store["obj"]
        tm.assert_equal(result, obj)


def test_timezones_fixed_format_series_nonempty(setup_path, tz_aware_fixture):
    # GH 20594

    dtype = pd.DatetimeTZDtype(tz=tz_aware_fixture)

    with ensure_clean_store(setup_path) as store:
        s = Series([0], dtype=dtype)
        store["s"] = s
        result = store["s"]
        tm.assert_series_equal(result, s)


def test_fixed_offset_tz(setup_path):
    rng = date_range("1/1/2000 00:00:00-07:00", "1/30/2000 00:00:00-07:00")
    frame = DataFrame(
        np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng
    )

    with ensure_clean_store(setup_path) as store:
        store["frame"] = frame
        recons = store["frame"]
        tm.assert_index_equal(recons.index, rng)
        assert rng.tz == recons.index.tz


@td.skip_if_windows
def test_store_timezone(setup_path):
    # GH2852
    # issue storing datetime.date with a timezone as it resets when read
    # back in a new timezone

    # original method
    with ensure_clean_store(setup_path) as store:
        today = date(2013, 9, 10)
        df = DataFrame([1, 2, 3], index=[today, today, today])
        store["obj1"] = df
        result = store["obj1"]
        tm.assert_frame_equal(result, df)

    # with tz setting
    with ensure_clean_store(setup_path) as store:
        with tm.set_timezone("EST5EDT"):
            today = date(2013, 9, 10)
            df = DataFrame([1, 2, 3], index=[today, today, today])
            store["obj1"] = df

        with tm.set_timezone("CST6CDT"):
            result = store["obj1"]

        tm.assert_frame_equal(result, df)


def test_legacy_datetimetz_object(datapath):
    # legacy from < 0.17.0
    # 8260
    expected = DataFrame(
        {
            "A": Timestamp("20130102", tz="US/Eastern").as_unit("ns"),
            "B": Timestamp("20130603", tz="CET").as_unit("ns"),
        },
        index=range(5),
    )
    with ensure_clean_store(
        datapath("io", "data", "legacy_hdf", "datetimetz_object.h5"), mode="r"
    ) as store:
        result = store["df"]
        tm.assert_frame_equal(result, expected)


def test_dst_transitions(setup_path):
    # make sure we are not failing on transitions
    with ensure_clean_store(setup_path) as store:
        times = date_range(
            "2013-10-26 23:00",
            "2013-10-27 01:00",
            tz="Europe/London",
            freq="h",
            ambiguous="infer",
        )
        times = times._with_freq(None)  # freq doesn't round-trip

        for i in [times, times + pd.Timedelta("10min")]:
            _maybe_remove(store, "df")
            df = DataFrame({"A": range(len(i)), "B": i}, index=i)
            store.append("df", df)
            result = store.select("df")
            tm.assert_frame_equal(result, df)


def test_read_with_where_tz_aware_index(tmp_path, setup_path):
    # GH 11926
    periods = 10
    dts = date_range("20151201", periods=periods, freq="D", tz="UTC")
    mi = pd.MultiIndex.from_arrays([dts, range(periods)], names=["DATE", "NO"])
    expected = DataFrame({"MYCOL": 0}, index=mi)

    key = "mykey"
    path = tmp_path / setup_path
    with pd.HDFStore(path) as store:
        store.append(key, expected, format="table", append=True)
    result = pd.read_hdf(path, key, where="DATE > 20151130")
    tm.assert_frame_equal(result, expected)


def test_py2_created_with_datetimez(datapath):
    # The test HDF5 file was created in Python 2, but could not be read in
    # Python 3.
    #
    # GH26443
    index = DatetimeIndex(["2019-01-01T18:00"], dtype="M8[ns, America/New_York]")
    expected = DataFrame({"data": 123}, index=index)
    with ensure_clean_store(
        datapath("io", "data", "legacy_hdf", "gh26443.h5"), mode="r"
    ) as store:
        result = store["key"]
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\tests\test_satask.py ===
from sympy.assumptions.ask import Q
from sympy.assumptions.assume import assuming
from sympy.core.numbers import (I, pi)
from sympy.core.relational import (Eq, Gt)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import Abs
from sympy.logic.boolalg import Implies
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.assumptions.cnf import CNF, Literal
from sympy.assumptions.satask import (satask, extract_predargs,
    get_relevant_clsfacts)

from sympy.testing.pytest import raises, XFAIL


x, y, z = symbols('x y z')


def test_satask():
    # No relevant facts
    assert satask(Q.real(x), Q.real(x)) is True
    assert satask(Q.real(x), ~Q.real(x)) is False
    assert satask(Q.real(x)) is None

    assert satask(Q.real(x), Q.positive(x)) is True
    assert satask(Q.positive(x), Q.real(x)) is None
    assert satask(Q.real(x), ~Q.positive(x)) is None
    assert satask(Q.positive(x), ~Q.real(x)) is False

    raises(ValueError, lambda: satask(Q.real(x), Q.real(x) & ~Q.real(x)))

    with assuming(Q.positive(x)):
        assert satask(Q.real(x)) is True
        assert satask(~Q.positive(x)) is False
        raises(ValueError, lambda: satask(Q.real(x), ~Q.positive(x)))

    assert satask(Q.zero(x), Q.nonzero(x)) is False
    assert satask(Q.positive(x), Q.zero(x)) is False
    assert satask(Q.real(x), Q.zero(x)) is True
    assert satask(Q.zero(x), Q.zero(x*y)) is None
    assert satask(Q.zero(x*y), Q.zero(x))


def test_zero():
    """
    Everything in this test doesn't work with the ask handlers, and most
    things would be very difficult or impossible to make work under that
    model.

    """
    assert satask(Q.zero(x) | Q.zero(y), Q.zero(x*y)) is True
    assert satask(Q.zero(x*y), Q.zero(x) | Q.zero(y)) is True

    assert satask(Implies(Q.zero(x), Q.zero(x*y))) is True

    # This one in particular requires computing the fixed-point of the
    # relevant facts, because going from Q.nonzero(x*y) -> ~Q.zero(x*y) and
    # Q.zero(x*y) -> Equivalent(Q.zero(x*y), Q.zero(x) | Q.zero(y)) takes two
    # steps.
    assert satask(Q.zero(x) | Q.zero(y), Q.nonzero(x*y)) is False

    assert satask(Q.zero(x), Q.zero(x**2)) is True


def test_zero_positive():
    assert satask(Q.zero(x + y), Q.positive(x) & Q.positive(y)) is False
    assert satask(Q.positive(x) & Q.positive(y), Q.zero(x + y)) is False
    assert satask(Q.nonzero(x + y), Q.positive(x) & Q.positive(y)) is True
    assert satask(Q.positive(x) & Q.positive(y), Q.nonzero(x + y)) is None

    # This one requires several levels of forward chaining
    assert satask(Q.zero(x*(x + y)), Q.positive(x) & Q.positive(y)) is False

    assert satask(Q.positive(pi*x*y + 1), Q.positive(x) & Q.positive(y)) is True
    assert satask(Q.positive(pi*x*y - 5), Q.positive(x) & Q.positive(y)) is None


def test_zero_pow():
    assert satask(Q.zero(x**y), Q.zero(x) & Q.positive(y)) is True
    assert satask(Q.zero(x**y), Q.nonzero(x) & Q.zero(y)) is False

    assert satask(Q.zero(x), Q.zero(x**y)) is True

    assert satask(Q.zero(x**y), Q.zero(x)) is None


@XFAIL
# Requires correct Q.square calculation first
def test_invertible():
    A = MatrixSymbol('A', 5, 5)
    B = MatrixSymbol('B', 5, 5)
    assert satask(Q.invertible(A*B), Q.invertible(A) & Q.invertible(B)) is True
    assert satask(Q.invertible(A), Q.invertible(A*B)) is True
    assert satask(Q.invertible(A) & Q.invertible(B), Q.invertible(A*B)) is True


def test_prime():
    assert satask(Q.prime(5)) is True
    assert satask(Q.prime(6)) is False
    assert satask(Q.prime(-5)) is False

    assert satask(Q.prime(x*y), Q.integer(x) & Q.integer(y)) is None
    assert satask(Q.prime(x*y), Q.prime(x) & Q.prime(y)) is False


def test_old_assump():
    assert satask(Q.positive(1)) is True
    assert satask(Q.positive(-1)) is False
    assert satask(Q.positive(0)) is False
    assert satask(Q.positive(I)) is False
    assert satask(Q.positive(pi)) is True

    assert satask(Q.negative(1)) is False
    assert satask(Q.negative(-1)) is True
    assert satask(Q.negative(0)) is False
    assert satask(Q.negative(I)) is False
    assert satask(Q.negative(pi)) is False

    assert satask(Q.zero(1)) is False
    assert satask(Q.zero(-1)) is False
    assert satask(Q.zero(0)) is True
    assert satask(Q.zero(I)) is False
    assert satask(Q.zero(pi)) is False

    assert satask(Q.nonzero(1)) is True
    assert satask(Q.nonzero(-1)) is True
    assert satask(Q.nonzero(0)) is False
    assert satask(Q.nonzero(I)) is False
    assert satask(Q.nonzero(pi)) is True

    assert satask(Q.nonpositive(1)) is False
    assert satask(Q.nonpositive(-1)) is True
    assert satask(Q.nonpositive(0)) is True
    assert satask(Q.nonpositive(I)) is False
    assert satask(Q.nonpositive(pi)) is False

    assert satask(Q.nonnegative(1)) is True
    assert satask(Q.nonnegative(-1)) is False
    assert satask(Q.nonnegative(0)) is True
    assert satask(Q.nonnegative(I)) is False
    assert satask(Q.nonnegative(pi)) is True


def test_rational_irrational():
    assert satask(Q.irrational(2)) is False
    assert satask(Q.rational(2)) is True
    assert satask(Q.irrational(pi)) is True
    assert satask(Q.rational(pi)) is False
    assert satask(Q.irrational(I)) is False
    assert satask(Q.rational(I)) is False

    assert satask(Q.irrational(x*y*z), Q.irrational(x) & Q.irrational(y) &
        Q.rational(z)) is None
    assert satask(Q.irrational(x*y*z), Q.irrational(x) & Q.rational(y) &
        Q.rational(z)) is True
    assert satask(Q.irrational(pi*x*y), Q.rational(x) & Q.rational(y)) is True

    assert satask(Q.irrational(x + y + z), Q.irrational(x) & Q.irrational(y) &
        Q.rational(z)) is None
    assert satask(Q.irrational(x + y + z), Q.irrational(x) & Q.rational(y) &
        Q.rational(z)) is True
    assert satask(Q.irrational(pi + x + y), Q.rational(x) & Q.rational(y)) is True

    assert satask(Q.irrational(x*y*z), Q.rational(x) & Q.rational(y) &
        Q.rational(z)) is False
    assert satask(Q.rational(x*y*z), Q.rational(x) & Q.rational(y) &
        Q.rational(z)) is True

    assert satask(Q.irrational(x + y + z), Q.rational(x) & Q.rational(y) &
        Q.rational(z)) is False
    assert satask(Q.rational(x + y + z), Q.rational(x) & Q.rational(y) &
        Q.rational(z)) is True


def test_even_satask():
    assert satask(Q.even(2)) is True
    assert satask(Q.even(3)) is False

    assert satask(Q.even(x*y), Q.even(x) & Q.odd(y)) is True
    assert satask(Q.even(x*y), Q.even(x) & Q.integer(y)) is True
    assert satask(Q.even(x*y), Q.even(x) & Q.even(y)) is True
    assert satask(Q.even(x*y), Q.odd(x) & Q.odd(y)) is False
    assert satask(Q.even(x*y), Q.even(x)) is None
    assert satask(Q.even(x*y), Q.odd(x) & Q.integer(y)) is None
    assert satask(Q.even(x*y), Q.odd(x) & Q.odd(y)) is False

    assert satask(Q.even(abs(x)), Q.even(x)) is True
    assert satask(Q.even(abs(x)), Q.odd(x)) is False
    assert satask(Q.even(x), Q.even(abs(x))) is None # x could be complex


def test_odd_satask():
    assert satask(Q.odd(2)) is False
    assert satask(Q.odd(3)) is True

    assert satask(Q.odd(x*y), Q.even(x) & Q.odd(y)) is False
    assert satask(Q.odd(x*y), Q.even(x) & Q.integer(y)) is False
    assert satask(Q.odd(x*y), Q.even(x) & Q.even(y)) is False
    assert satask(Q.odd(x*y), Q.odd(x) & Q.odd(y)) is True
    assert satask(Q.odd(x*y), Q.even(x)) is None
    assert satask(Q.odd(x*y), Q.odd(x) & Q.integer(y)) is None
    assert satask(Q.odd(x*y), Q.odd(x) & Q.odd(y)) is True

    assert satask(Q.odd(abs(x)), Q.even(x)) is False
    assert satask(Q.odd(abs(x)), Q.odd(x)) is True
    assert satask(Q.odd(x), Q.odd(abs(x))) is None # x could be complex


def test_integer():
    assert satask(Q.integer(1)) is True
    assert satask(Q.integer(S.Half)) is False

    assert satask(Q.integer(x + y), Q.integer(x) & Q.integer(y)) is True
    assert satask(Q.integer(x + y), Q.integer(x)) is None

    assert satask(Q.integer(x + y), Q.integer(x) & ~Q.integer(y)) is False
    assert satask(Q.integer(x + y + z), Q.integer(x) & Q.integer(y) &
        ~Q.integer(z)) is False
    assert satask(Q.integer(x + y + z), Q.integer(x) & ~Q.integer(y) &
        ~Q.integer(z)) is None
    assert satask(Q.integer(x + y + z), Q.integer(x) & ~Q.integer(y)) is None
    assert satask(Q.integer(x + y), Q.integer(x) & Q.irrational(y)) is False

    assert satask(Q.integer(x*y), Q.integer(x) & Q.integer(y)) is True
    assert satask(Q.integer(x*y), Q.integer(x)) is None

    assert satask(Q.integer(x*y), Q.integer(x) & ~Q.integer(y)) is None
    assert satask(Q.integer(x*y), Q.integer(x) & ~Q.rational(y)) is False
    assert satask(Q.integer(x*y*z), Q.integer(x) & Q.integer(y) &
        ~Q.rational(z)) is False
    assert satask(Q.integer(x*y*z), Q.integer(x) & ~Q.rational(y) &
        ~Q.rational(z)) is None
    assert satask(Q.integer(x*y*z), Q.integer(x) & ~Q.rational(y)) is None
    assert satask(Q.integer(x*y), Q.integer(x) & Q.irrational(y)) is False


def test_abs():
    assert satask(Q.nonnegative(abs(x))) is True
    assert satask(Q.positive(abs(x)), ~Q.zero(x)) is True
    assert satask(Q.zero(x), ~Q.zero(abs(x))) is False
    assert satask(Q.zero(x), Q.zero(abs(x))) is True
    assert satask(Q.nonzero(x), ~Q.zero(abs(x))) is None # x could be complex
    assert satask(Q.zero(abs(x)), Q.zero(x)) is True


def test_imaginary():
    assert satask(Q.imaginary(2*I)) is True
    assert satask(Q.imaginary(x*y), Q.imaginary(x)) is None
    assert satask(Q.imaginary(x*y), Q.imaginary(x) & Q.real(y)) is True
    assert satask(Q.imaginary(x), Q.real(x)) is False
    assert satask(Q.imaginary(1)) is False
    assert satask(Q.imaginary(x*y), Q.real(x) & Q.real(y)) is False
    assert satask(Q.imaginary(x + y), Q.real(x) & Q.real(y)) is False


def test_real():
    assert satask(Q.real(x*y), Q.real(x) & Q.real(y)) is True
    assert satask(Q.real(x + y), Q.real(x) & Q.real(y)) is True
    assert satask(Q.real(x*y*z), Q.real(x) & Q.real(y) & Q.real(z)) is True
    assert satask(Q.real(x*y*z), Q.real(x) & Q.real(y)) is None
    assert satask(Q.real(x*y*z), Q.real(x) & Q.real(y) & Q.imaginary(z)) is False
    assert satask(Q.real(x + y + z), Q.real(x) & Q.real(y) & Q.real(z)) is True
    assert satask(Q.real(x + y + z), Q.real(x) & Q.real(y)) is None


def test_pos_neg():
    assert satask(~Q.positive(x), Q.negative(x)) is True
    assert satask(~Q.negative(x), Q.positive(x)) is True
    assert satask(Q.positive(x + y), Q.positive(x) & Q.positive(y)) is True
    assert satask(Q.negative(x + y), Q.negative(x) & Q.negative(y)) is True
    assert satask(Q.positive(x + y), Q.negative(x) & Q.negative(y)) is False
    assert satask(Q.negative(x + y), Q.positive(x) & Q.positive(y)) is False


def test_pow_pos_neg():
    assert satask(Q.nonnegative(x**2), Q.positive(x)) is True
    assert satask(Q.nonpositive(x**2), Q.positive(x)) is False
    assert satask(Q.positive(x**2), Q.positive(x)) is True
    assert satask(Q.negative(x**2), Q.positive(x)) is False
    assert satask(Q.real(x**2), Q.positive(x)) is True

    assert satask(Q.nonnegative(x**2), Q.negative(x)) is True
    assert satask(Q.nonpositive(x**2), Q.negative(x)) is False
    assert satask(Q.positive(x**2), Q.negative(x)) is True
    assert satask(Q.negative(x**2), Q.negative(x)) is False
    assert satask(Q.real(x**2), Q.negative(x)) is True

    assert satask(Q.nonnegative(x**2), Q.nonnegative(x)) is True
    assert satask(Q.nonpositive(x**2), Q.nonnegative(x)) is None
    assert satask(Q.positive(x**2), Q.nonnegative(x)) is None
    assert satask(Q.negative(x**2), Q.nonnegative(x)) is False
    assert satask(Q.real(x**2), Q.nonnegative(x)) is True

    assert satask(Q.nonnegative(x**2), Q.nonpositive(x)) is True
    assert satask(Q.nonpositive(x**2), Q.nonpositive(x)) is None
    assert satask(Q.positive(x**2), Q.nonpositive(x)) is None
    assert satask(Q.negative(x**2), Q.nonpositive(x)) is False
    assert satask(Q.real(x**2), Q.nonpositive(x)) is True

    assert satask(Q.nonnegative(x**3), Q.positive(x)) is True
    assert satask(Q.nonpositive(x**3), Q.positive(x)) is False
    assert satask(Q.positive(x**3), Q.positive(x)) is True
    assert satask(Q.negative(x**3), Q.positive(x)) is False
    assert satask(Q.real(x**3), Q.positive(x)) is True

    assert satask(Q.nonnegative(x**3), Q.negative(x)) is False
    assert satask(Q.nonpositive(x**3), Q.negative(x)) is True
    assert satask(Q.positive(x**3), Q.negative(x)) is False
    assert satask(Q.negative(x**3), Q.negative(x)) is True
    assert satask(Q.real(x**3), Q.negative(x)) is True

    assert satask(Q.nonnegative(x**3), Q.nonnegative(x)) is True
    assert satask(Q.nonpositive(x**3), Q.nonnegative(x)) is None
    assert satask(Q.positive(x**3), Q.nonnegative(x)) is None
    assert satask(Q.negative(x**3), Q.nonnegative(x)) is False
    assert satask(Q.real(x**3), Q.nonnegative(x)) is True

    assert satask(Q.nonnegative(x**3), Q.nonpositive(x)) is None
    assert satask(Q.nonpositive(x**3), Q.nonpositive(x)) is True
    assert satask(Q.positive(x**3), Q.nonpositive(x)) is False
    assert satask(Q.negative(x**3), Q.nonpositive(x)) is None
    assert satask(Q.real(x**3), Q.nonpositive(x)) is True

    # If x is zero, x**negative is not real.
    assert satask(Q.nonnegative(x**-2), Q.nonpositive(x)) is None
    assert satask(Q.nonpositive(x**-2), Q.nonpositive(x)) is None
    assert satask(Q.positive(x**-2), Q.nonpositive(x)) is None
    assert satask(Q.negative(x**-2), Q.nonpositive(x)) is None
    assert satask(Q.real(x**-2), Q.nonpositive(x)) is None

    # We could deduce things for negative powers if x is nonzero, but it
    # isn't implemented yet.


def test_prime_composite():
    assert satask(Q.prime(x), Q.composite(x)) is False
    assert satask(Q.composite(x), Q.prime(x)) is False
    assert satask(Q.composite(x), ~Q.prime(x)) is None
    assert satask(Q.prime(x), ~Q.composite(x)) is None
    # since 1 is neither prime nor composite the following should hold
    assert satask(Q.prime(x), Q.integer(x) & Q.positive(x) & ~Q.composite(x)) is None
    assert satask(Q.prime(2)) is True
    assert satask(Q.prime(4)) is False
    assert satask(Q.prime(1)) is False
    assert satask(Q.composite(1)) is False


def test_extract_predargs():
    props = CNF.from_prop(Q.zero(Abs(x*y)) & Q.zero(x*y))
    assump = CNF.from_prop(Q.zero(x))
    context = CNF.from_prop(Q.zero(y))
    assert extract_predargs(props) == {Abs(x*y), x*y}
    assert extract_predargs(props, assump) == {Abs(x*y), x*y, x}
    assert extract_predargs(props, assump, context) == {Abs(x*y), x*y, x, y}

    props = CNF.from_prop(Eq(x, y))
    assump = CNF.from_prop(Gt(y, z))
    assert extract_predargs(props, assump) == {x, y, z}


def test_get_relevant_clsfacts():
    exprs = {Abs(x*y)}
    exprs, facts = get_relevant_clsfacts(exprs)
    assert exprs == {x*y}
    assert facts.clauses == \
        {frozenset({Literal(Q.odd(Abs(x*y)), False), Literal(Q.odd(x*y), True)}),
        frozenset({Literal(Q.zero(Abs(x*y)), False), Literal(Q.zero(x*y), True)}),
        frozenset({Literal(Q.even(Abs(x*y)), False), Literal(Q.even(x*y), True)}),
        frozenset({Literal(Q.zero(Abs(x*y)), True), Literal(Q.zero(x*y), False)}),
        frozenset({Literal(Q.even(Abs(x*y)), False),
                    Literal(Q.odd(Abs(x*y)), False),
                    Literal(Q.odd(x*y), True)}),
        frozenset({Literal(Q.even(Abs(x*y)), False),
                    Literal(Q.even(x*y), True),
                    Literal(Q.odd(Abs(x*y)), False)}),
        frozenset({Literal(Q.positive(Abs(x*y)), False),
                    Literal(Q.zero(Abs(x*y)), False)})}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\tests\test_ctypeslib.py ===
import sys
import sysconfig
import weakref
from pathlib import Path

import pytest

import numpy as np
from numpy.ctypeslib import as_array, load_library, ndpointer
from numpy.testing import assert_, assert_array_equal, assert_equal, assert_raises

try:
    import ctypes
except ImportError:
    ctypes = None
else:
    cdll = None
    test_cdll = None
    if hasattr(sys, 'gettotalrefcount'):
        try:
            cdll = load_library(
                '_multiarray_umath_d', np._core._multiarray_umath.__file__
            )
        except OSError:
            pass
        try:
            test_cdll = load_library(
                '_multiarray_tests', np._core._multiarray_tests.__file__
            )
        except OSError:
            pass
    if cdll is None:
        cdll = load_library(
            '_multiarray_umath', np._core._multiarray_umath.__file__)
    if test_cdll is None:
        test_cdll = load_library(
            '_multiarray_tests', np._core._multiarray_tests.__file__
        )

    c_forward_pointer = test_cdll.forward_pointer


@pytest.mark.skipif(ctypes is None,
                    reason="ctypes not available in this python")
@pytest.mark.skipif(sys.platform == 'cygwin',
                    reason="Known to fail on cygwin")
class TestLoadLibrary:
    def test_basic(self):
        loader_path = np._core._multiarray_umath.__file__

        out1 = load_library('_multiarray_umath', loader_path)
        out2 = load_library(Path('_multiarray_umath'), loader_path)
        out3 = load_library('_multiarray_umath', Path(loader_path))
        out4 = load_library(b'_multiarray_umath', loader_path)

        assert isinstance(out1, ctypes.CDLL)
        assert out1 is out2 is out3 is out4

    def test_basic2(self):
        # Regression for #801: load_library with a full library name
        # (including extension) does not work.
        try:
            so_ext = sysconfig.get_config_var('EXT_SUFFIX')
            load_library(f'_multiarray_umath{so_ext}',
                         np._core._multiarray_umath.__file__)
        except ImportError as e:
            msg = ("ctypes is not available on this python: skipping the test"
                   " (import error was: %s)" % str(e))
            print(msg)


class TestNdpointer:
    def test_dtype(self):
        dt = np.intc
        p = ndpointer(dtype=dt)
        assert_(p.from_param(np.array([1], dt)))
        dt = '<i4'
        p = ndpointer(dtype=dt)
        assert_(p.from_param(np.array([1], dt)))
        dt = np.dtype('>i4')
        p = ndpointer(dtype=dt)
        p.from_param(np.array([1], dt))
        assert_raises(TypeError, p.from_param,
                          np.array([1], dt.newbyteorder('swap')))
        dtnames = ['x', 'y']
        dtformats = [np.intc, np.float64]
        dtdescr = {'names': dtnames, 'formats': dtformats}
        dt = np.dtype(dtdescr)
        p = ndpointer(dtype=dt)
        assert_(p.from_param(np.zeros((10,), dt)))
        samedt = np.dtype(dtdescr)
        p = ndpointer(dtype=samedt)
        assert_(p.from_param(np.zeros((10,), dt)))
        dt2 = np.dtype(dtdescr, align=True)
        if dt.itemsize != dt2.itemsize:
            assert_raises(TypeError, p.from_param, np.zeros((10,), dt2))
        else:
            assert_(p.from_param(np.zeros((10,), dt2)))

    def test_ndim(self):
        p = ndpointer(ndim=0)
        assert_(p.from_param(np.array(1)))
        assert_raises(TypeError, p.from_param, np.array([1]))
        p = ndpointer(ndim=1)
        assert_raises(TypeError, p.from_param, np.array(1))
        assert_(p.from_param(np.array([1])))
        p = ndpointer(ndim=2)
        assert_(p.from_param(np.array([[1]])))

    def test_shape(self):
        p = ndpointer(shape=(1, 2))
        assert_(p.from_param(np.array([[1, 2]])))
        assert_raises(TypeError, p.from_param, np.array([[1], [2]]))
        p = ndpointer(shape=())
        assert_(p.from_param(np.array(1)))

    def test_flags(self):
        x = np.array([[1, 2], [3, 4]], order='F')
        p = ndpointer(flags='FORTRAN')
        assert_(p.from_param(x))
        p = ndpointer(flags='CONTIGUOUS')
        assert_raises(TypeError, p.from_param, x)
        p = ndpointer(flags=x.flags.num)
        assert_(p.from_param(x))
        assert_raises(TypeError, p.from_param, np.array([[1, 2], [3, 4]]))

    def test_cache(self):
        assert_(ndpointer(dtype=np.float64) is ndpointer(dtype=np.float64))

        # shapes are normalized
        assert_(ndpointer(shape=2) is ndpointer(shape=(2,)))

        # 1.12 <= v < 1.16 had a bug that made these fail
        assert_(ndpointer(shape=2) is not ndpointer(ndim=2))
        assert_(ndpointer(ndim=2) is not ndpointer(shape=2))

@pytest.mark.skipif(ctypes is None,
                    reason="ctypes not available on this python installation")
class TestNdpointerCFunc:
    def test_arguments(self):
        """ Test that arguments are coerced from arrays """
        c_forward_pointer.restype = ctypes.c_void_p
        c_forward_pointer.argtypes = (ndpointer(ndim=2),)

        c_forward_pointer(np.zeros((2, 3)))
        # too many dimensions
        assert_raises(
            ctypes.ArgumentError, c_forward_pointer, np.zeros((2, 3, 4)))

    @pytest.mark.parametrize(
        'dt', [
            float,
            np.dtype({
                'formats': ['<i4', '<i4'],
                'names': ['a', 'b'],
                'offsets': [0, 2],
                'itemsize': 6
            })
        ], ids=[
            'float',
            'overlapping-fields'
        ]
    )
    def test_return(self, dt):
        """ Test that return values are coerced to arrays """
        arr = np.zeros((2, 3), dt)
        ptr_type = ndpointer(shape=arr.shape, dtype=arr.dtype)

        c_forward_pointer.restype = ptr_type
        c_forward_pointer.argtypes = (ptr_type,)

        # check that the arrays are equivalent views on the same data
        arr2 = c_forward_pointer(arr)
        assert_equal(arr2.dtype, arr.dtype)
        assert_equal(arr2.shape, arr.shape)
        assert_equal(
            arr2.__array_interface__['data'],
            arr.__array_interface__['data']
        )

    def test_vague_return_value(self):
        """ Test that vague ndpointer return values do not promote to arrays """
        arr = np.zeros((2, 3))
        ptr_type = ndpointer(dtype=arr.dtype)

        c_forward_pointer.restype = ptr_type
        c_forward_pointer.argtypes = (ptr_type,)

        ret = c_forward_pointer(arr)
        assert_(isinstance(ret, ptr_type))


@pytest.mark.skipif(ctypes is None,
                    reason="ctypes not available on this python installation")
class TestAsArray:
    def test_array(self):
        from ctypes import c_int

        pair_t = c_int * 2
        a = as_array(pair_t(1, 2))
        assert_equal(a.shape, (2,))
        assert_array_equal(a, np.array([1, 2]))
        a = as_array((pair_t * 3)(pair_t(1, 2), pair_t(3, 4), pair_t(5, 6)))
        assert_equal(a.shape, (3, 2))
        assert_array_equal(a, np.array([[1, 2], [3, 4], [5, 6]]))

    def test_pointer(self):
        from ctypes import POINTER, c_int, cast

        p = cast((c_int * 10)(*range(10)), POINTER(c_int))

        a = as_array(p, shape=(10,))
        assert_equal(a.shape, (10,))
        assert_array_equal(a, np.arange(10))

        a = as_array(p, shape=(2, 5))
        assert_equal(a.shape, (2, 5))
        assert_array_equal(a, np.arange(10).reshape((2, 5)))

        # shape argument is required
        assert_raises(TypeError, as_array, p)

    @pytest.mark.skipif(
            sys.version_info[:2] == (3, 12),
            reason="Broken in 3.12.0rc1, see gh-24399",
    )
    def test_struct_array_pointer(self):
        from ctypes import Structure, c_int16, pointer

        class Struct(Structure):
            _fields_ = [('a', c_int16)]

        Struct3 = 3 * Struct

        c_array = (2 * Struct3)(
            Struct3(Struct(a=1), Struct(a=2), Struct(a=3)),
            Struct3(Struct(a=4), Struct(a=5), Struct(a=6))
        )

        expected = np.array([
            [(1,), (2,), (3,)],
            [(4,), (5,), (6,)],
        ], dtype=[('a', np.int16)])

        def check(x):
            assert_equal(x.dtype, expected.dtype)
            assert_equal(x, expected)

        # all of these should be equivalent
        check(as_array(c_array))
        check(as_array(pointer(c_array), shape=()))
        check(as_array(pointer(c_array[0]), shape=(2,)))
        check(as_array(pointer(c_array[0][0]), shape=(2, 3)))

    def test_reference_cycles(self):
        # related to gh-6511
        import ctypes

        # create array to work with
        # don't use int/long to avoid running into bpo-10746
        N = 100
        a = np.arange(N, dtype=np.short)

        # get pointer to array
        pnt = np.ctypeslib.as_ctypes(a)

        with np.testing.assert_no_gc_cycles():
            # decay the array above to a pointer to its first element
            newpnt = ctypes.cast(pnt, ctypes.POINTER(ctypes.c_short))
            # and construct an array using this data
            b = np.ctypeslib.as_array(newpnt, (N,))
            # now delete both, which should cleanup both objects
            del newpnt, b

    def test_segmentation_fault(self):
        arr = np.zeros((224, 224, 3))
        c_arr = np.ctypeslib.as_ctypes(arr)
        arr_ref = weakref.ref(arr)
        del arr

        # check the reference wasn't cleaned up
        assert_(arr_ref() is not None)

        # check we avoid the segfault
        c_arr[0][0][0]


@pytest.mark.skipif(ctypes is None,
                    reason="ctypes not available on this python installation")
class TestAsCtypesType:
    """ Test conversion from dtypes to ctypes types """
    def test_scalar(self):
        dt = np.dtype('<u2')
        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_equal(ct, ctypes.c_uint16.__ctype_le__)

        dt = np.dtype('>u2')
        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_equal(ct, ctypes.c_uint16.__ctype_be__)

        dt = np.dtype('u2')
        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_equal(ct, ctypes.c_uint16)

    def test_subarray(self):
        dt = np.dtype((np.int32, (2, 3)))
        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_equal(ct, 2 * (3 * ctypes.c_int32))

    def test_structure(self):
        dt = np.dtype([
            ('a', np.uint16),
            ('b', np.uint32),
        ])

        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_(issubclass(ct, ctypes.Structure))
        assert_equal(ctypes.sizeof(ct), dt.itemsize)
        assert_equal(ct._fields_, [
            ('a', ctypes.c_uint16),
            ('b', ctypes.c_uint32),
        ])

    def test_structure_aligned(self):
        dt = np.dtype([
            ('a', np.uint16),
            ('b', np.uint32),
        ], align=True)

        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_(issubclass(ct, ctypes.Structure))
        assert_equal(ctypes.sizeof(ct), dt.itemsize)
        assert_equal(ct._fields_, [
            ('a', ctypes.c_uint16),
            ('', ctypes.c_char * 2),  # padding
            ('b', ctypes.c_uint32),
        ])

    def test_union(self):
        dt = np.dtype({
            'names': ['a', 'b'],
            'offsets': [0, 0],
            'formats': [np.uint16, np.uint32]
        })

        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_(issubclass(ct, ctypes.Union))
        assert_equal(ctypes.sizeof(ct), dt.itemsize)
        assert_equal(ct._fields_, [
            ('a', ctypes.c_uint16),
            ('b', ctypes.c_uint32),
        ])

    def test_padded_union(self):
        dt = np.dtype({
            'names': ['a', 'b'],
            'offsets': [0, 0],
            'formats': [np.uint16, np.uint32],
            'itemsize': 5,
        })

        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_(issubclass(ct, ctypes.Union))
        assert_equal(ctypes.sizeof(ct), dt.itemsize)
        assert_equal(ct._fields_, [
            ('a', ctypes.c_uint16),
            ('b', ctypes.c_uint32),
            ('', ctypes.c_char * 5),  # padding
        ])

    def test_overlapping(self):
        dt = np.dtype({
            'names': ['a', 'b'],
            'offsets': [0, 2],
            'formats': [np.uint32, np.uint32]
        })
        assert_raises(NotImplementedError, np.ctypeslib.as_ctypes_type, dt)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_index_col.py ===
"""
Tests that the specified index column (a.k.a "index_col")
is properly handled or inferred during parsing for all of
the parsers defined in parsers.py
"""
from io import StringIO

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


@pytest.mark.parametrize("with_header", [True, False])
def test_index_col_named(all_parsers, with_header):
    parser = all_parsers
    no_header = """\
KORD1,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD2,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD3,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD4,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD5,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD6,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000"""
    header = "ID,date,NominalTime,ActualTime,TDew,TAir,Windspeed,Precip,WindDir\n"

    if with_header:
        data = header + no_header

        result = parser.read_csv(StringIO(data), index_col="ID")
        expected = parser.read_csv(StringIO(data), header=0).set_index("ID")
        tm.assert_frame_equal(result, expected)
    else:
        data = no_header
        msg = "Index ID invalid"

        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), index_col="ID")


def test_index_col_named2(all_parsers):
    parser = all_parsers
    data = """\
1,2,3,4,hello
5,6,7,8,world
9,10,11,12,foo
"""

    expected = DataFrame(
        {"a": [1, 5, 9], "b": [2, 6, 10], "c": [3, 7, 11], "d": [4, 8, 12]},
        index=Index(["hello", "world", "foo"], name="message"),
    )
    names = ["a", "b", "c", "d", "message"]

    result = parser.read_csv(StringIO(data), names=names, index_col=["message"])
    tm.assert_frame_equal(result, expected)


def test_index_col_is_true(all_parsers):
    # see gh-9798
    data = "a,b\n1,2"
    parser = all_parsers

    msg = "The value of index_col couldn't be 'True'"
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), index_col=True)


@skip_pyarrow  # CSV parse error: Expected 3 columns, got 4
def test_infer_index_col(all_parsers):
    data = """A,B,C
foo,1,2,3
bar,4,5,6
baz,7,8,9
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data))

    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        index=["foo", "bar", "baz"],
        columns=["A", "B", "C"],
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
@pytest.mark.parametrize(
    "index_col,kwargs",
    [
        (None, {"columns": ["x", "y", "z"]}),
        (False, {"columns": ["x", "y", "z"]}),
        (0, {"columns": ["y", "z"], "index": Index([], name="x")}),
        (1, {"columns": ["x", "z"], "index": Index([], name="y")}),
        ("x", {"columns": ["y", "z"], "index": Index([], name="x")}),
        ("y", {"columns": ["x", "z"], "index": Index([], name="y")}),
        (
            [0, 1],
            {
                "columns": ["z"],
                "index": MultiIndex.from_arrays([[]] * 2, names=["x", "y"]),
            },
        ),
        (
            ["x", "y"],
            {
                "columns": ["z"],
                "index": MultiIndex.from_arrays([[]] * 2, names=["x", "y"]),
            },
        ),
        (
            [1, 0],
            {
                "columns": ["z"],
                "index": MultiIndex.from_arrays([[]] * 2, names=["y", "x"]),
            },
        ),
        (
            ["y", "x"],
            {
                "columns": ["z"],
                "index": MultiIndex.from_arrays([[]] * 2, names=["y", "x"]),
            },
        ),
    ],
)
def test_index_col_empty_data(all_parsers, index_col, kwargs):
    data = "x,y,z"
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=index_col)

    expected = DataFrame(**kwargs)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_empty_with_index_col_false(all_parsers):
    # see gh-10413
    data = "x,y"
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=False)

    expected = DataFrame(columns=["x", "y"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "index_names",
    [
        ["", ""],
        ["foo", ""],
        ["", "bar"],
        ["foo", "bar"],
        ["NotReallyUnnamed", "Unnamed: 0"],
    ],
)
def test_multi_index_naming(all_parsers, index_names, request):
    parser = all_parsers

    if parser.engine == "pyarrow" and "" in index_names:
        mark = pytest.mark.xfail(reason="One case raises, others are wrong")
        request.applymarker(mark)

    # We don't want empty index names being replaced with "Unnamed: 0"
    data = ",".join(index_names + ["col\na,c,1\na,d,2\nb,c,3\nb,d,4"])
    result = parser.read_csv(StringIO(data), index_col=[0, 1])

    expected = DataFrame(
        {"col": [1, 2, 3, 4]}, index=MultiIndex.from_product([["a", "b"], ["c", "d"]])
    )
    expected.index.names = [name if name else None for name in index_names]
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: Found non-unique column index
def test_multi_index_naming_not_all_at_beginning(all_parsers):
    parser = all_parsers
    data = ",Unnamed: 2,\na,c,1\na,d,2\nb,c,3\nb,d,4"
    result = parser.read_csv(StringIO(data), index_col=[0, 2])

    expected = DataFrame(
        {"Unnamed: 2": ["c", "d", "c", "d"]},
        index=MultiIndex(
            levels=[["a", "b"], [1, 2, 3, 4]], codes=[[0, 0, 1, 1], [0, 1, 2, 3]]
        ),
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: Found non-unique column index
def test_no_multi_index_level_names_empty(all_parsers):
    # GH 10984
    parser = all_parsers
    midx = MultiIndex.from_tuples([("A", 1, 2), ("A", 1, 2), ("B", 1, 2)])
    expected = DataFrame(
        np.random.default_rng(2).standard_normal((3, 3)),
        index=midx,
        columns=["x", "y", "z"],
    )
    with tm.ensure_clean() as path:
        expected.to_csv(path)
        result = parser.read_csv(path, index_col=[0, 1, 2])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_header_with_index_col(all_parsers):
    # GH 33476
    parser = all_parsers
    data = """
I11,A,A
I12,B,B
I2,1,3
"""
    midx = MultiIndex.from_tuples([("A", "B"), ("A", "B.1")], names=["I11", "I12"])
    idx = Index(["I2"])
    expected = DataFrame([[1, 3]], index=idx, columns=midx)

    result = parser.read_csv(StringIO(data), index_col=0, header=[0, 1])
    tm.assert_frame_equal(result, expected)

    col_idx = Index(["A", "A.1"])
    idx = Index(["I12", "I2"], name="I11")
    expected = DataFrame([["B", "B"], ["1", "3"]], index=idx, columns=col_idx)

    result = parser.read_csv(StringIO(data), index_col="I11", header=0)
    tm.assert_frame_equal(result, expected)


@pytest.mark.slow
def test_index_col_large_csv(all_parsers, monkeypatch):
    # https://github.com/pandas-dev/pandas/issues/37094
    parser = all_parsers

    ARR_LEN = 100
    df = DataFrame(
        {
            "a": range(ARR_LEN + 1),
            "b": np.random.default_rng(2).standard_normal(ARR_LEN + 1),
        }
    )

    with tm.ensure_clean() as path:
        df.to_csv(path, index=False)
        with monkeypatch.context() as m:
            m.setattr("pandas.core.algorithms._MINIMUM_COMP_ARR_LEN", ARR_LEN)
            result = parser.read_csv(path, index_col=[0])

    tm.assert_frame_equal(result, df.set_index("a"))


@xfail_pyarrow  # TypeError: an integer is required
def test_index_col_multiindex_columns_no_data(all_parsers):
    # GH#38292
    parser = all_parsers
    result = parser.read_csv(
        StringIO("a0,a1,a2\nb0,b1,b2\n"), header=[0, 1], index_col=0
    )
    expected = DataFrame(
        [],
        index=Index([]),
        columns=MultiIndex.from_arrays(
            [["a1", "a2"], ["b1", "b2"]], names=["a0", "b0"]
        ),
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_index_col_header_no_data(all_parsers):
    # GH#38292
    parser = all_parsers
    result = parser.read_csv(StringIO("a0,a1,a2\n"), header=[0], index_col=0)
    expected = DataFrame(
        [],
        columns=["a1", "a2"],
        index=Index([], name="a0"),
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_multiindex_columns_no_data(all_parsers):
    # GH#38292
    parser = all_parsers
    result = parser.read_csv(StringIO("a0,a1,a2\nb0,b1,b2\n"), header=[0, 1])
    expected = DataFrame(
        [], columns=MultiIndex.from_arrays([["a0", "a1", "a2"], ["b0", "b1", "b2"]])
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_multiindex_columns_index_col_with_data(all_parsers):
    # GH#38292
    parser = all_parsers
    result = parser.read_csv(
        StringIO("a0,a1,a2\nb0,b1,b2\ndata,data,data"), header=[0, 1], index_col=0
    )
    expected = DataFrame(
        [["data", "data"]],
        columns=MultiIndex.from_arrays(
            [["a1", "a2"], ["b1", "b2"]], names=["a0", "b0"]
        ),
        index=Index(["data"]),
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_infer_types_boolean_sum(all_parsers):
    # GH#44079
    parser = all_parsers
    result = parser.read_csv(
        StringIO("0,1"),
        names=["a", "b"],
        index_col=["a"],
        dtype={"a": "UInt8"},
    )
    expected = DataFrame(
        data={
            "a": [
                0,
            ],
            "b": [1],
        }
    ).set_index("a")
    # Not checking index type now, because the C parser will return a
    # index column of dtype 'object', and the Python parser will return a
    # index column of dtype 'int64'.
    tm.assert_frame_equal(result, expected, check_index_type=False)


@pytest.mark.parametrize("dtype, val", [(object, "01"), ("int64", 1)])
def test_specify_dtype_for_index_col(all_parsers, dtype, val, request):
    # GH#9435
    data = "a,b\n01,2"
    parser = all_parsers
    if dtype == object and parser.engine == "pyarrow":
        request.applymarker(
            pytest.mark.xfail(reason="Cannot disable type-inference for pyarrow engine")
        )
    result = parser.read_csv(StringIO(data), index_col="a", dtype={"a": dtype})
    expected = DataFrame({"b": [2]}, index=Index([val], name="a", dtype=dtype))
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_multiindex_columns_not_leading_index_col(all_parsers):
    # GH#38549
    parser = all_parsers
    data = """a,b,c,d
e,f,g,h
x,y,1,2
"""
    result = parser.read_csv(
        StringIO(data),
        header=[0, 1],
        index_col=1,
    )
    cols = MultiIndex.from_tuples(
        [("a", "e"), ("c", "g"), ("d", "h")], names=["b", "f"]
    )
    expected = DataFrame([["x", 1, 2]], columns=cols, index=["y"])
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\external\tests\test_codegen.py ===
# This tests the compilation and execution of the source code generated with
# utilities.codegen. The compilation takes place in a temporary directory that
# is removed after the test. By default the test directory is always removed,
# but this behavior can be changed by setting the environment variable
# SYMPY_TEST_CLEAN_TEMP to:
#   export SYMPY_TEST_CLEAN_TEMP=always   : the default behavior.
#   export SYMPY_TEST_CLEAN_TEMP=success  : only remove the directories of working tests.
#   export SYMPY_TEST_CLEAN_TEMP=never    : never remove the directories with the test code.
# When a directory is not removed, the necessary information is printed on
# screen to find the files that belong to the (failed) tests. If a test does
# not fail, py.test captures all the output and you will not see the directories
# corresponding to the successful tests. Use the --nocapture option to see all
# the output.

# All tests below have a counterpart in utilities/test/test_codegen.py. In the
# latter file, the resulting code is compared with predefined strings, without
# compilation or execution.

# All the generated Fortran code should conform with the Fortran 95 standard,
# and all the generated C code should be ANSI C, which facilitates the
# incorporation in various projects. The tests below assume that the binary cc
# is somewhere in the path and that it can compile ANSI C code.

from sympy.abc import x, y, z
from sympy.testing.pytest import IS_WASM, skip
from sympy.utilities.codegen import codegen, make_routine, get_code_generator
import sys
import os
import tempfile
import subprocess
from pathlib import Path


# templates for the main program that will test the generated code.

main_template = {}
main_template['F95'] = """
program main
  include "codegen.h"
  integer :: result;
  result = 0

  %(statements)s

  call exit(result)
end program
"""

main_template['C89'] = """
#include "codegen.h"
#include <stdio.h>
#include <math.h>

int main() {
  int result = 0;

  %(statements)s

  return result;
}
"""
main_template['C99'] = main_template['C89']
# templates for the numerical tests

numerical_test_template = {}
numerical_test_template['C89'] = """
  if (fabs(%(call)s)>%(threshold)s) {
    printf("Numerical validation failed: %(call)s=%%e threshold=%(threshold)s\\n", %(call)s);
    result = -1;
  }
"""
numerical_test_template['C99'] = numerical_test_template['C89']

numerical_test_template['F95'] = """
  if (abs(%(call)s)>%(threshold)s) then
    write(6,"('Numerical validation failed:')")
    write(6,"('%(call)s=',e15.5,'threshold=',e15.5)") %(call)s, %(threshold)s
    result = -1;
  end if
"""
# command sequences for supported compilers

compile_commands = {}
compile_commands['cc'] = [
    "cc -c codegen.c -o codegen.o",
    "cc -c main.c -o main.o",
    "cc main.o codegen.o -lm -o test.exe"
]

compile_commands['gfortran'] = [
    "gfortran -c codegen.f90 -o codegen.o",
    "gfortran -ffree-line-length-none -c main.f90 -o main.o",
    "gfortran main.o codegen.o -o test.exe"
]

compile_commands['g95'] = [
    "g95 -c codegen.f90 -o codegen.o",
    "g95 -ffree-line-length-huge -c main.f90 -o main.o",
    "g95 main.o codegen.o -o test.exe"
]

compile_commands['ifort'] = [
    "ifort -c codegen.f90 -o codegen.o",
    "ifort -c main.f90 -o main.o",
    "ifort main.o codegen.o -o test.exe"
]

combinations_lang_compiler = [
    ('C89', 'cc'),
    ('C99', 'cc'),
    ('F95', 'ifort'),
    ('F95', 'gfortran'),
    ('F95', 'g95')
]

def try_run(commands):
    """Run a series of commands and only return True if all ran fine."""
    if IS_WASM:
        return False
    with open(os.devnull, 'w') as null:
        for command in commands:
            retcode = subprocess.call(command, stdout=null, shell=True,
                    stderr=subprocess.STDOUT)
            if retcode != 0:
                return False
    return True


def run_test(label, routines, numerical_tests, language, commands, friendly=True):
    """A driver for the codegen tests.

       This driver assumes that a compiler ifort is present in the PATH and that
       ifort is (at least) a Fortran 90 compiler. The generated code is written in
       a temporary directory, together with a main program that validates the
       generated code. The test passes when the compilation and the validation
       run correctly.
    """

    # Check input arguments before touching the file system
    language = language.upper()
    assert language in main_template
    assert language in numerical_test_template

    # Check that environment variable makes sense
    clean = os.getenv('SYMPY_TEST_CLEAN_TEMP', 'always').lower()
    if clean not in ('always', 'success', 'never'):
        raise ValueError("SYMPY_TEST_CLEAN_TEMP must be one of the following: 'always', 'success' or 'never'.")

    # Do all the magic to compile, run and validate the test code
    # 1) prepare the temporary working directory, switch to that dir
    work = tempfile.mkdtemp("_sympy_%s_test" % language, "%s_" % label)
    oldwork = os.getcwd()
    os.chdir(work)

    # 2) write the generated code
    if friendly:
        # interpret the routines as a name_expr list and call the friendly
        # function codegen
        codegen(routines, language, "codegen", to_files=True)
    else:
        code_gen = get_code_generator(language, "codegen")
        code_gen.write(routines, "codegen", to_files=True)

    # 3) write a simple main program that links to the generated code, and that
    #    includes the numerical tests
    test_strings = []
    for fn_name, args, expected, threshold in numerical_tests:
        call_string = "%s(%s)-(%s)" % (
            fn_name, ",".join(str(arg) for arg in args), expected)
        if language == "F95":
            call_string = fortranize_double_constants(call_string)
            threshold = fortranize_double_constants(str(threshold))
        test_strings.append(numerical_test_template[language] % {
            "call": call_string,
            "threshold": threshold,
        })

    if language == "F95":
        f_name = "main.f90"
    elif language.startswith("C"):
        f_name = "main.c"
    else:
        raise NotImplementedError(
            "FIXME: filename extension unknown for language: %s" % language)

    Path(f_name).write_text(
        main_template[language] % {'statements': "".join(test_strings)})

    # 4) Compile and link
    compiled = try_run(commands)

    # 5) Run if compiled
    if compiled:
        executed = try_run(["./test.exe"])
    else:
        executed = False

    # 6) Clean up stuff
    if clean == 'always' or (clean == 'success' and compiled and executed):
        def safe_remove(filename):
            if os.path.isfile(filename):
                os.remove(filename)
        safe_remove("codegen.f90")
        safe_remove("codegen.c")
        safe_remove("codegen.h")
        safe_remove("codegen.o")
        safe_remove("main.f90")
        safe_remove("main.c")
        safe_remove("main.o")
        safe_remove("test.exe")
        os.chdir(oldwork)
        os.rmdir(work)
    else:
        print("TEST NOT REMOVED: %s" % work, file=sys.stderr)
        os.chdir(oldwork)

    # 7) Do the assertions in the end
    assert compiled, "failed to compile %s code with:\n%s" % (
        language, "\n".join(commands))
    assert executed, "failed to execute %s code from:\n%s" % (
        language, "\n".join(commands))


def fortranize_double_constants(code_string):
    """
    Replaces every literal float with literal doubles
    """
    import re
    pattern_exp = re.compile(r'\d+(\.)?\d*[eE]-?\d+')
    pattern_float = re.compile(r'\d+\.\d*(?!\d*d)')

    def subs_exp(matchobj):
        return re.sub('[eE]', 'd', matchobj.group(0))

    def subs_float(matchobj):
        return "%sd0" % matchobj.group(0)

    code_string = pattern_exp.sub(subs_exp, code_string)
    code_string = pattern_float.sub(subs_float, code_string)

    return code_string


def is_feasible(language, commands):
    # This test should always work, otherwise the compiler is not present.
    routine = make_routine("test", x)
    numerical_tests = [
        ("test", ( 1.0,), 1.0, 1e-15),
        ("test", (-1.0,), -1.0, 1e-15),
    ]
    try:
        run_test("is_feasible", [routine], numerical_tests, language, commands,
                 friendly=False)
        return True
    except AssertionError:
        return False

valid_lang_commands = []
invalid_lang_compilers = []
for lang, compiler in combinations_lang_compiler:
    commands = compile_commands[compiler]
    if is_feasible(lang, commands):
        valid_lang_commands.append((lang, commands))
    else:
        invalid_lang_compilers.append((lang, compiler))

# We test all language-compiler combinations, just to report what is skipped

def test_C89_cc():
    if ("C89", 'cc') in invalid_lang_compilers:
        skip("`cc' command didn't work as expected (C89)")


def test_C99_cc():
    if ("C99", 'cc') in invalid_lang_compilers:
        skip("`cc' command didn't work as expected (C99)")


def test_F95_ifort():
    if ("F95", 'ifort') in invalid_lang_compilers:
        skip("`ifort' command didn't work as expected")


def test_F95_gfortran():
    if ("F95", 'gfortran') in invalid_lang_compilers:
        skip("`gfortran' command didn't work as expected")


def test_F95_g95():
    if ("F95", 'g95') in invalid_lang_compilers:
        skip("`g95' command didn't work as expected")

# Here comes the actual tests


def test_basic_codegen():
    numerical_tests = [
        ("test", (1.0, 6.0, 3.0), 21.0, 1e-15),
        ("test", (-1.0, 2.0, -2.5), -2.5, 1e-15),
    ]
    name_expr = [("test", (x + y)*z)]
    for lang, commands in valid_lang_commands:
        run_test("basic_codegen", name_expr, numerical_tests, lang, commands)


def test_intrinsic_math1_codegen():
    # not included: log10
    from sympy.core.evalf import N
    from sympy.functions import ln
    from sympy.functions.elementary.exponential import log
    from sympy.functions.elementary.hyperbolic import (cosh, sinh, tanh)
    from sympy.functions.elementary.integers import (ceiling, floor)
    from sympy.functions.elementary.miscellaneous import sqrt
    from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, tan)
    name_expr = [
        ("test_fabs", abs(x)),
        ("test_acos", acos(x)),
        ("test_asin", asin(x)),
        ("test_atan", atan(x)),
        ("test_cos", cos(x)),
        ("test_cosh", cosh(x)),
        ("test_log", log(x)),
        ("test_ln", ln(x)),
        ("test_sin", sin(x)),
        ("test_sinh", sinh(x)),
        ("test_sqrt", sqrt(x)),
        ("test_tan", tan(x)),
        ("test_tanh", tanh(x)),
    ]
    numerical_tests = []
    for name, expr in name_expr:
        for xval in 0.2, 0.5, 0.8:
            expected = N(expr.subs(x, xval))
            numerical_tests.append((name, (xval,), expected, 1e-14))
    for lang, commands in valid_lang_commands:
        if lang.startswith("C"):
            name_expr_C = [("test_floor", floor(x)), ("test_ceil", ceiling(x))]
        else:
            name_expr_C = []
        run_test("intrinsic_math1", name_expr + name_expr_C,
                 numerical_tests, lang, commands)


def test_instrinsic_math2_codegen():
    # not included: frexp, ldexp, modf, fmod
    from sympy.core.evalf import N
    from sympy.functions.elementary.trigonometric import atan2
    name_expr = [
        ("test_atan2", atan2(x, y)),
        ("test_pow", x**y),
    ]
    numerical_tests = []
    for name, expr in name_expr:
        for xval, yval in (0.2, 1.3), (0.5, -0.2), (0.8, 0.8):
            expected = N(expr.subs(x, xval).subs(y, yval))
            numerical_tests.append((name, (xval, yval), expected, 1e-14))
    for lang, commands in valid_lang_commands:
        run_test("intrinsic_math2", name_expr, numerical_tests, lang, commands)


def test_complicated_codegen():
    from sympy.core.evalf import N
    from sympy.functions.elementary.trigonometric import (cos, sin, tan)
    name_expr = [
        ("test1", ((sin(x) + cos(y) + tan(z))**7).expand()),
        ("test2", cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))),
    ]
    numerical_tests = []
    for name, expr in name_expr:
        for xval, yval, zval in (0.2, 1.3, -0.3), (0.5, -0.2, 0.0), (0.8, 2.1, 0.8):
            expected = N(expr.subs(x, xval).subs(y, yval).subs(z, zval))
            numerical_tests.append((name, (xval, yval, zval), expected, 1e-12))
    for lang, commands in valid_lang_commands:
        run_test(
            "complicated_codegen", name_expr, numerical_tests, lang, commands)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\tests\test_mutable_ndim_array.py ===
from copy import copy

from sympy.tensor.array.dense_ndim_array import MutableDenseNDimArray
from sympy.core.function import diff
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices import SparseMatrix
from sympy.matrices import Matrix
from sympy.tensor.array.sparse_ndim_array import MutableSparseNDimArray
from sympy.testing.pytest import raises


def test_ndim_array_initiation():
    arr_with_one_element = MutableDenseNDimArray([23])
    assert len(arr_with_one_element) == 1
    assert arr_with_one_element[0] == 23
    assert arr_with_one_element.rank() == 1
    raises(ValueError, lambda: arr_with_one_element[1])

    arr_with_symbol_element = MutableDenseNDimArray([Symbol('x')])
    assert len(arr_with_symbol_element) == 1
    assert arr_with_symbol_element[0] == Symbol('x')
    assert arr_with_symbol_element.rank() == 1

    number5 = 5
    vector = MutableDenseNDimArray.zeros(number5)
    assert len(vector) == number5
    assert vector.shape == (number5,)
    assert vector.rank() == 1
    raises(ValueError, lambda: arr_with_one_element[5])

    vector = MutableSparseNDimArray.zeros(number5)
    assert len(vector) == number5
    assert vector.shape == (number5,)
    assert vector._sparse_array == {}
    assert vector.rank() == 1

    n_dim_array = MutableDenseNDimArray(range(3**4), (3, 3, 3, 3,))
    assert len(n_dim_array) == 3 * 3 * 3 * 3
    assert n_dim_array.shape == (3, 3, 3, 3)
    assert n_dim_array.rank() == 4
    raises(ValueError, lambda: n_dim_array[0, 0, 0, 3])
    raises(ValueError, lambda: n_dim_array[3, 0, 0, 0])
    raises(ValueError, lambda: n_dim_array[3**4])

    array_shape = (3, 3, 3, 3)
    sparse_array = MutableSparseNDimArray.zeros(*array_shape)
    assert len(sparse_array._sparse_array) == 0
    assert len(sparse_array) == 3 * 3 * 3 * 3
    assert n_dim_array.shape == array_shape
    assert n_dim_array.rank() == 4

    one_dim_array = MutableDenseNDimArray([2, 3, 1])
    assert len(one_dim_array) == 3
    assert one_dim_array.shape == (3,)
    assert one_dim_array.rank() == 1
    assert one_dim_array.tolist() == [2, 3, 1]

    shape = (3, 3)
    array_with_many_args = MutableSparseNDimArray.zeros(*shape)
    assert len(array_with_many_args) == 3 * 3
    assert array_with_many_args.shape == shape
    assert array_with_many_args[0, 0] == 0
    assert array_with_many_args.rank() == 2

    shape = (int(3), int(3))
    array_with_long_shape = MutableSparseNDimArray.zeros(*shape)
    assert len(array_with_long_shape) == 3 * 3
    assert array_with_long_shape.shape == shape
    assert array_with_long_shape[int(0), int(0)] == 0
    assert array_with_long_shape.rank() == 2

    vector_with_long_shape = MutableDenseNDimArray(range(5), int(5))
    assert len(vector_with_long_shape) == 5
    assert vector_with_long_shape.shape == (int(5),)
    assert vector_with_long_shape.rank() == 1
    raises(ValueError, lambda: vector_with_long_shape[int(5)])

    from sympy.abc import x
    for ArrayType in [MutableDenseNDimArray, MutableSparseNDimArray]:
        rank_zero_array = ArrayType(x)
        assert len(rank_zero_array) == 1
        assert rank_zero_array.shape == ()
        assert rank_zero_array.rank() == 0
        assert rank_zero_array[()] == x
        raises(ValueError, lambda: rank_zero_array[0])

def test_sympify():
    from sympy.abc import x, y, z, t
    arr = MutableDenseNDimArray([[x, y], [1, z*t]])
    arr_other = sympify(arr)
    assert arr_other.shape == (2, 2)
    assert arr_other == arr


def test_reshape():
    array = MutableDenseNDimArray(range(50), 50)
    assert array.shape == (50,)
    assert array.rank() == 1

    array = array.reshape(5, 5, 2)
    assert array.shape == (5, 5, 2)
    assert array.rank() == 3
    assert len(array) == 50


def test_iterator():
    array = MutableDenseNDimArray(range(4), (2, 2))
    assert array[0] == MutableDenseNDimArray([0, 1])
    assert array[1] == MutableDenseNDimArray([2, 3])

    array = array.reshape(4)
    j = 0
    for i in array:
        assert i == j
        j += 1


def test_getitem():
    for ArrayType in [MutableDenseNDimArray, MutableSparseNDimArray]:
        array = ArrayType(range(24)).reshape(2, 3, 4)
        assert array.tolist() == [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
        assert array[0] == ArrayType([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
        assert array[0, 0] == ArrayType([0, 1, 2, 3])
        value = 0
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    assert array[i, j, k] == value
                    value += 1

    raises(ValueError, lambda: array[3, 4, 5])
    raises(ValueError, lambda: array[3, 4, 5, 6])
    raises(ValueError, lambda: array[3, 4, 5, 3:4])


def test_sparse():
    sparse_array = MutableSparseNDimArray([0, 0, 0, 1], (2, 2))
    assert len(sparse_array) == 2 * 2
    # dictionary where all data is, only non-zero entries are actually stored:
    assert len(sparse_array._sparse_array) == 1

    assert sparse_array.tolist() == [[0, 0], [0, 1]]

    for i, j in zip(sparse_array, [[0, 0], [0, 1]]):
        assert i == MutableSparseNDimArray(j)

    sparse_array[0, 0] = 123
    assert len(sparse_array._sparse_array) == 2
    assert sparse_array[0, 0] == 123
    assert sparse_array/0 == MutableSparseNDimArray([[S.ComplexInfinity, S.NaN], [S.NaN, S.ComplexInfinity]], (2, 2))

    # when element in sparse array become zero it will disappear from
    # dictionary
    sparse_array[0, 0] = 0
    assert len(sparse_array._sparse_array) == 1
    sparse_array[1, 1] = 0
    assert len(sparse_array._sparse_array) == 0
    assert sparse_array[0, 0] == 0

    # test for large scale sparse array
    # equality test
    a = MutableSparseNDimArray.zeros(100000, 200000)
    b = MutableSparseNDimArray.zeros(100000, 200000)
    assert a == b
    a[1, 1] = 1
    b[1, 1] = 2
    assert a != b

    # __mul__ and __rmul__
    assert a * 3 == MutableSparseNDimArray({200001: 3}, (100000, 200000))
    assert 3 * a == MutableSparseNDimArray({200001: 3}, (100000, 200000))
    assert a * 0 == MutableSparseNDimArray({}, (100000, 200000))
    assert 0 * a == MutableSparseNDimArray({}, (100000, 200000))

    # __truediv__
    assert a/3 == MutableSparseNDimArray({200001: Rational(1, 3)}, (100000, 200000))

    # __neg__
    assert -a == MutableSparseNDimArray({200001: -1}, (100000, 200000))


def test_calculation():

    a = MutableDenseNDimArray([1]*9, (3, 3))
    b = MutableDenseNDimArray([9]*9, (3, 3))

    c = a + b
    for i in c:
        assert i == MutableDenseNDimArray([10, 10, 10])

    assert c == MutableDenseNDimArray([10]*9, (3, 3))
    assert c == MutableSparseNDimArray([10]*9, (3, 3))

    c = b - a
    for i in c:
        assert i == MutableSparseNDimArray([8, 8, 8])

    assert c == MutableDenseNDimArray([8]*9, (3, 3))
    assert c == MutableSparseNDimArray([8]*9, (3, 3))


def test_ndim_array_converting():
    dense_array = MutableDenseNDimArray([1, 2, 3, 4], (2, 2))
    alist = dense_array.tolist()

    assert alist == [[1, 2], [3, 4]]

    matrix = dense_array.tomatrix()
    assert (isinstance(matrix, Matrix))

    for i in range(len(dense_array)):
        assert dense_array[dense_array._get_tuple_index(i)] == matrix[i]
    assert matrix.shape == dense_array.shape

    assert MutableDenseNDimArray(matrix) == dense_array
    assert MutableDenseNDimArray(matrix.as_immutable()) == dense_array
    assert MutableDenseNDimArray(matrix.as_mutable()) == dense_array

    sparse_array = MutableSparseNDimArray([1, 2, 3, 4], (2, 2))
    alist = sparse_array.tolist()

    assert alist == [[1, 2], [3, 4]]

    matrix = sparse_array.tomatrix()
    assert(isinstance(matrix, SparseMatrix))

    for i in range(len(sparse_array)):
        assert sparse_array[sparse_array._get_tuple_index(i)] == matrix[i]
    assert matrix.shape == sparse_array.shape

    assert MutableSparseNDimArray(matrix) == sparse_array
    assert MutableSparseNDimArray(matrix.as_immutable()) == sparse_array
    assert MutableSparseNDimArray(matrix.as_mutable()) == sparse_array


def test_converting_functions():
    arr_list = [1, 2, 3, 4]
    arr_matrix = Matrix(((1, 2), (3, 4)))

    # list
    arr_ndim_array = MutableDenseNDimArray(arr_list, (2, 2))
    assert (isinstance(arr_ndim_array, MutableDenseNDimArray))
    assert arr_matrix.tolist() == arr_ndim_array.tolist()

    # Matrix
    arr_ndim_array = MutableDenseNDimArray(arr_matrix)
    assert (isinstance(arr_ndim_array, MutableDenseNDimArray))
    assert arr_matrix.tolist() == arr_ndim_array.tolist()
    assert arr_matrix.shape == arr_ndim_array.shape


def test_equality():
    first_list = [1, 2, 3, 4]
    second_list = [1, 2, 3, 4]
    third_list = [4, 3, 2, 1]
    assert first_list == second_list
    assert first_list != third_list

    first_ndim_array = MutableDenseNDimArray(first_list, (2, 2))
    second_ndim_array = MutableDenseNDimArray(second_list, (2, 2))
    third_ndim_array = MutableDenseNDimArray(third_list, (2, 2))
    fourth_ndim_array = MutableDenseNDimArray(first_list, (2, 2))

    assert first_ndim_array == second_ndim_array
    second_ndim_array[0, 0] = 0
    assert first_ndim_array != second_ndim_array
    assert first_ndim_array != third_ndim_array
    assert first_ndim_array == fourth_ndim_array


def test_arithmetic():
    a = MutableDenseNDimArray([3 for i in range(9)], (3, 3))
    b = MutableDenseNDimArray([7 for i in range(9)], (3, 3))

    c1 = a + b
    c2 = b + a
    assert c1 == c2

    d1 = a - b
    d2 = b - a
    assert d1 == d2 * (-1)

    e1 = a * 5
    e2 = 5 * a
    e3 = copy(a)
    e3 *= 5
    assert e1 == e2 == e3

    f1 = a / 5
    f2 = copy(a)
    f2 /= 5
    assert f1 == f2
    assert f1[0, 0] == f1[0, 1] == f1[0, 2] == f1[1, 0] == f1[1, 1] == \
    f1[1, 2] == f1[2, 0] == f1[2, 1] == f1[2, 2] == Rational(3, 5)

    assert type(a) == type(b) == type(c1) == type(c2) == type(d1) == type(d2) \
        == type(e1) == type(e2) == type(e3) == type(f1)

    z0 = -a
    assert z0 == MutableDenseNDimArray([-3 for i in range(9)], (3, 3))


def test_higher_dimenions():
    m3 = MutableDenseNDimArray(range(10, 34), (2, 3, 4))

    assert m3.tolist() == [[[10, 11, 12, 13],
            [14, 15, 16, 17],
            [18, 19, 20, 21]],

           [[22, 23, 24, 25],
            [26, 27, 28, 29],
            [30, 31, 32, 33]]]

    assert m3._get_tuple_index(0) == (0, 0, 0)
    assert m3._get_tuple_index(1) == (0, 0, 1)
    assert m3._get_tuple_index(4) == (0, 1, 0)
    assert m3._get_tuple_index(12) == (1, 0, 0)

    assert str(m3) == '[[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]]'

    m3_rebuilt = MutableDenseNDimArray([[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]])
    assert m3 == m3_rebuilt

    m3_other = MutableDenseNDimArray([[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]], (2, 3, 4))

    assert m3 == m3_other


def test_slices():
    md = MutableDenseNDimArray(range(10, 34), (2, 3, 4))

    assert md[:] == MutableDenseNDimArray(range(10, 34), (2, 3, 4))
    assert md[:, :, 0].tomatrix() == Matrix([[10, 14, 18], [22, 26, 30]])
    assert md[0, 1:2, :].tomatrix() == Matrix([[14, 15, 16, 17]])
    assert md[0, 1:3, :].tomatrix() == Matrix([[14, 15, 16, 17], [18, 19, 20, 21]])
    assert md[:, :, :] == md

    sd = MutableSparseNDimArray(range(10, 34), (2, 3, 4))
    assert sd == MutableSparseNDimArray(md)

    assert sd[:] == MutableSparseNDimArray(range(10, 34), (2, 3, 4))
    assert sd[:, :, 0].tomatrix() == Matrix([[10, 14, 18], [22, 26, 30]])
    assert sd[0, 1:2, :].tomatrix() == Matrix([[14, 15, 16, 17]])
    assert sd[0, 1:3, :].tomatrix() == Matrix([[14, 15, 16, 17], [18, 19, 20, 21]])
    assert sd[:, :, :] == sd


def test_slices_assign():
    a = MutableDenseNDimArray(range(12), shape=(4, 3))
    b = MutableSparseNDimArray(range(12), shape=(4, 3))

    for i in [a, b]:
        assert i.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        i[0, :] = [2, 2, 2]
        assert i.tolist() == [[2, 2, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        i[0, 1:] = [8, 8]
        assert i.tolist() == [[2, 8, 8], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        i[1:3, 1] = [20, 44]
        assert i.tolist() == [[2, 8, 8], [3, 20, 5], [6, 44, 8], [9, 10, 11]]


def test_diff():
    from sympy.abc import x, y, z
    md = MutableDenseNDimArray([[x, y], [x*z, x*y*z]])
    assert md.diff(x) == MutableDenseNDimArray([[1, 0], [z, y*z]])
    assert diff(md, x) == MutableDenseNDimArray([[1, 0], [z, y*z]])

    sd = MutableSparseNDimArray(md)
    assert sd == MutableSparseNDimArray([x, y, x*z, x*y*z], (2, 2))
    assert sd.diff(x) == MutableSparseNDimArray([[1, 0], [z, y*z]])
    assert diff(sd, x) == MutableSparseNDimArray([[1, 0], [z, y*z]])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_sample.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm
import pandas.core.common as com


class TestSample:
    @pytest.fixture
    def obj(self, frame_or_series):
        if frame_or_series is Series:
            arr = np.random.default_rng(2).standard_normal(10)
        else:
            arr = np.random.default_rng(2).standard_normal((10, 10))
        return frame_or_series(arr, dtype=None)

    @pytest.mark.parametrize("test", list(range(10)))
    def test_sample(self, test, obj):
        # Fixes issue: 2419
        # Check behavior of random_state argument
        # Check for stability when receives seed or random state -- run 10
        # times.

        seed = np.random.default_rng(2).integers(0, 100)
        tm.assert_equal(
            obj.sample(n=4, random_state=seed), obj.sample(n=4, random_state=seed)
        )

        tm.assert_equal(
            obj.sample(frac=0.7, random_state=seed),
            obj.sample(frac=0.7, random_state=seed),
        )

        tm.assert_equal(
            obj.sample(n=4, random_state=np.random.default_rng(test)),
            obj.sample(n=4, random_state=np.random.default_rng(test)),
        )

        tm.assert_equal(
            obj.sample(frac=0.7, random_state=np.random.default_rng(test)),
            obj.sample(frac=0.7, random_state=np.random.default_rng(test)),
        )

        tm.assert_equal(
            obj.sample(
                frac=2,
                replace=True,
                random_state=np.random.default_rng(test),
            ),
            obj.sample(
                frac=2,
                replace=True,
                random_state=np.random.default_rng(test),
            ),
        )

        os1, os2 = [], []
        for _ in range(2):
            os1.append(obj.sample(n=4, random_state=test))
            os2.append(obj.sample(frac=0.7, random_state=test))
        tm.assert_equal(*os1)
        tm.assert_equal(*os2)

    def test_sample_lengths(self, obj):
        # Check lengths are right
        assert len(obj.sample(n=4) == 4)
        assert len(obj.sample(frac=0.34) == 3)
        assert len(obj.sample(frac=0.36) == 4)

    def test_sample_invalid_random_state(self, obj):
        # Check for error when random_state argument invalid.
        msg = (
            "random_state must be an integer, array-like, a BitGenerator, Generator, "
            "a numpy RandomState, or None"
        )
        with pytest.raises(ValueError, match=msg):
            obj.sample(random_state="a_string")

    def test_sample_wont_accept_n_and_frac(self, obj):
        # Giving both frac and N throws error
        msg = "Please enter a value for `frac` OR `n`, not both"
        with pytest.raises(ValueError, match=msg):
            obj.sample(n=3, frac=0.3)

    def test_sample_requires_positive_n_frac(self, obj):
        with pytest.raises(
            ValueError,
            match="A negative number of rows requested. Please provide `n` >= 0",
        ):
            obj.sample(n=-3)
        with pytest.raises(
            ValueError,
            match="A negative number of rows requested. Please provide `frac` >= 0",
        ):
            obj.sample(frac=-0.3)

    def test_sample_requires_integer_n(self, obj):
        # Make sure float values of `n` give error
        with pytest.raises(ValueError, match="Only integers accepted as `n` values"):
            obj.sample(n=3.2)

    def test_sample_invalid_weight_lengths(self, obj):
        # Weight length must be right
        msg = "Weights and axis to be sampled must be of same length"
        with pytest.raises(ValueError, match=msg):
            obj.sample(n=3, weights=[0, 1])

        with pytest.raises(ValueError, match=msg):
            bad_weights = [0.5] * 11
            obj.sample(n=3, weights=bad_weights)

        with pytest.raises(ValueError, match="Fewer non-zero entries in p than size"):
            bad_weight_series = Series([0, 0, 0.2])
            obj.sample(n=4, weights=bad_weight_series)

    def test_sample_negative_weights(self, obj):
        # Check won't accept negative weights
        bad_weights = [-0.1] * 10
        msg = "weight vector many not include negative values"
        with pytest.raises(ValueError, match=msg):
            obj.sample(n=3, weights=bad_weights)

    def test_sample_inf_weights(self, obj):
        # Check inf and -inf throw errors:

        weights_with_inf = [0.1] * 10
        weights_with_inf[0] = np.inf
        msg = "weight vector may not include `inf` values"
        with pytest.raises(ValueError, match=msg):
            obj.sample(n=3, weights=weights_with_inf)

        weights_with_ninf = [0.1] * 10
        weights_with_ninf[0] = -np.inf
        with pytest.raises(ValueError, match=msg):
            obj.sample(n=3, weights=weights_with_ninf)

    def test_sample_zero_weights(self, obj):
        # All zeros raises errors

        zero_weights = [0] * 10
        with pytest.raises(ValueError, match="Invalid weights: weights sum to zero"):
            obj.sample(n=3, weights=zero_weights)

    def test_sample_missing_weights(self, obj):
        # All missing weights

        nan_weights = [np.nan] * 10
        with pytest.raises(ValueError, match="Invalid weights: weights sum to zero"):
            obj.sample(n=3, weights=nan_weights)

    def test_sample_none_weights(self, obj):
        # Check None are also replaced by zeros.
        weights_with_None = [None] * 10
        weights_with_None[5] = 0.5
        tm.assert_equal(
            obj.sample(n=1, axis=0, weights=weights_with_None), obj.iloc[5:6]
        )

    @pytest.mark.parametrize(
        "func_str,arg",
        [
            ("np.array", [2, 3, 1, 0]),
            ("np.random.MT19937", 3),
            ("np.random.PCG64", 11),
        ],
    )
    def test_sample_random_state(self, func_str, arg, frame_or_series):
        # GH#32503
        obj = DataFrame({"col1": range(10, 20), "col2": range(20, 30)})
        obj = tm.get_obj(obj, frame_or_series)
        result = obj.sample(n=3, random_state=eval(func_str)(arg))
        expected = obj.sample(n=3, random_state=com.random_state(eval(func_str)(arg)))
        tm.assert_equal(result, expected)

    def test_sample_generator(self, frame_or_series):
        # GH#38100
        obj = frame_or_series(np.arange(100))
        rng = np.random.default_rng(2)

        # Consecutive calls should advance the seed
        result1 = obj.sample(n=50, random_state=rng)
        result2 = obj.sample(n=50, random_state=rng)
        assert not (result1.index.values == result2.index.values).all()

        # Matching generator initialization must give same result
        # Consecutive calls should advance the seed
        result1 = obj.sample(n=50, random_state=np.random.default_rng(11))
        result2 = obj.sample(n=50, random_state=np.random.default_rng(11))
        tm.assert_equal(result1, result2)

    def test_sample_upsampling_without_replacement(self, frame_or_series):
        # GH#27451

        obj = DataFrame({"A": list("abc")})
        obj = tm.get_obj(obj, frame_or_series)

        msg = (
            "Replace has to be set to `True` when "
            "upsampling the population `frac` > 1."
        )
        with pytest.raises(ValueError, match=msg):
            obj.sample(frac=2, replace=False)


class TestSampleDataFrame:
    # Tests which are relevant only for DataFrame, so these are
    #  as fully parametrized as they can get.

    def test_sample(self):
        # GH#2419
        # additional specific object based tests

        # A few dataframe test with degenerate weights.
        easy_weight_list = [0] * 10
        easy_weight_list[5] = 1

        df = DataFrame(
            {
                "col1": range(10, 20),
                "col2": range(20, 30),
                "colString": ["a"] * 10,
                "easyweights": easy_weight_list,
            }
        )
        sample1 = df.sample(n=1, weights="easyweights")
        tm.assert_frame_equal(sample1, df.iloc[5:6])

        # Ensure proper error if string given as weight for Series or
        # DataFrame with axis = 1.
        ser = Series(range(10))
        msg = "Strings cannot be passed as weights when sampling from a Series."
        with pytest.raises(ValueError, match=msg):
            ser.sample(n=3, weights="weight_column")

        msg = (
            "Strings can only be passed to weights when sampling from rows on a "
            "DataFrame"
        )
        with pytest.raises(ValueError, match=msg):
            df.sample(n=1, weights="weight_column", axis=1)

        # Check weighting key error
        with pytest.raises(
            KeyError, match="'String passed to weights not a valid column'"
        ):
            df.sample(n=3, weights="not_a_real_column_name")

        # Check that re-normalizes weights that don't sum to one.
        weights_less_than_1 = [0] * 10
        weights_less_than_1[0] = 0.5
        tm.assert_frame_equal(df.sample(n=1, weights=weights_less_than_1), df.iloc[:1])

        ###
        # Test axis argument
        ###

        # Test axis argument
        df = DataFrame({"col1": range(10), "col2": ["a"] * 10})
        second_column_weight = [0, 1]
        tm.assert_frame_equal(
            df.sample(n=1, axis=1, weights=second_column_weight), df[["col2"]]
        )

        # Different axis arg types
        tm.assert_frame_equal(
            df.sample(n=1, axis="columns", weights=second_column_weight), df[["col2"]]
        )

        weight = [0] * 10
        weight[5] = 0.5
        tm.assert_frame_equal(df.sample(n=1, axis="rows", weights=weight), df.iloc[5:6])
        tm.assert_frame_equal(
            df.sample(n=1, axis="index", weights=weight), df.iloc[5:6]
        )

        # Check out of range axis values
        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            df.sample(n=1, axis=2)

        msg = "No axis named not_a_name for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            df.sample(n=1, axis="not_a_name")

        ser = Series(range(10))
        with pytest.raises(ValueError, match="No axis named 1 for object type Series"):
            ser.sample(n=1, axis=1)

        # Test weight length compared to correct axis
        msg = "Weights and axis to be sampled must be of same length"
        with pytest.raises(ValueError, match=msg):
            df.sample(n=1, axis=1, weights=[0.5] * 10)

    def test_sample_axis1(self):
        # Check weights with axis = 1
        easy_weight_list = [0] * 3
        easy_weight_list[2] = 1

        df = DataFrame(
            {"col1": range(10, 20), "col2": range(20, 30), "colString": ["a"] * 10}
        )
        sample1 = df.sample(n=1, axis=1, weights=easy_weight_list)
        tm.assert_frame_equal(sample1, df[["colString"]])

        # Test default axes
        tm.assert_frame_equal(
            df.sample(n=3, random_state=42), df.sample(n=3, axis=0, random_state=42)
        )

    def test_sample_aligns_weights_with_frame(self):
        # Test that function aligns weights with frame
        df = DataFrame({"col1": [5, 6, 7], "col2": ["a", "b", "c"]}, index=[9, 5, 3])
        ser = Series([1, 0, 0], index=[3, 5, 9])
        tm.assert_frame_equal(df.loc[[3]], df.sample(1, weights=ser))

        # Weights have index values to be dropped because not in
        # sampled DataFrame
        ser2 = Series([0.001, 0, 10000], index=[3, 5, 10])
        tm.assert_frame_equal(df.loc[[3]], df.sample(1, weights=ser2))

        # Weights have empty values to be filed with zeros
        ser3 = Series([0.01, 0], index=[3, 5])
        tm.assert_frame_equal(df.loc[[3]], df.sample(1, weights=ser3))

        # No overlap in weight and sampled DataFrame indices
        ser4 = Series([1, 0], index=[1, 2])

        with pytest.raises(ValueError, match="Invalid weights: weights sum to zero"):
            df.sample(1, weights=ser4)

    def test_sample_is_copy(self):
        # GH#27357, GH#30784: ensure the result of sample is an actual copy and
        # doesn't track the parent dataframe / doesn't give SettingWithCopy warnings
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=["a", "b", "c"]
        )
        df2 = df.sample(3)

        with tm.assert_produces_warning(None):
            df2["d"] = 1

    def test_sample_does_not_modify_weights(self):
        # GH-42843
        result = np.array([np.nan, 1, np.nan])
        expected = result.copy()
        ser = Series([1, 2, 3])

        # Test numpy array weights won't be modified in place
        ser.sample(weights=result)
        tm.assert_numpy_array_equal(result, expected)

        # Test DataFrame column won't be modified in place
        df = DataFrame({"values": [1, 1, 1], "weights": [1, np.nan, np.nan]})
        expected = df["weights"].copy()

        df.sample(frac=1.0, replace=True, weights="weights")
        result = df["weights"]
        tm.assert_series_equal(result, expected)

    def test_sample_ignore_index(self):
        # GH 38581
        df = DataFrame(
            {"col1": range(10, 20), "col2": range(20, 30), "colString": ["a"] * 10}
        )
        result = df.sample(3, ignore_index=True)
        expected_index = Index(range(3))
        tm.assert_index_equal(result.index, expected_index, exact=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_linearize.py ===
from sympy import symbols, Matrix, cos, sin, atan, sqrt, Rational
from sympy.core.sympify import sympify
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point,\
    dot, cross, inertia, KanesMethod, Particle, RigidBody, Lagrangian,\
    LagrangesMethod
from sympy.testing.pytest import slow


@slow
def test_linearize_rolling_disc_kane():
    # Symbols for time and constant parameters
    t, r, m, g, v = symbols('t r m g v')

    # Configuration variables and their time derivatives
    q1, q2, q3, q4, q5, q6 = q = dynamicsymbols('q1:7')
    q1d, q2d, q3d, q4d, q5d, q6d = qd = [qi.diff(t) for qi in q]

    # Generalized speeds and their time derivatives
    u = dynamicsymbols('u:6')
    u1, u2, u3, u4, u5, u6 = u = dynamicsymbols('u1:7')
    u1d, u2d, u3d, u4d, u5d, u6d = [ui.diff(t) for ui in u]

    # Reference frames
    N = ReferenceFrame('N')                   # Inertial frame
    NO = Point('NO')                          # Inertial origin
    A = N.orientnew('A', 'Axis', [q1, N.z])   # Yaw intermediate frame
    B = A.orientnew('B', 'Axis', [q2, A.x])   # Lean intermediate frame
    C = B.orientnew('C', 'Axis', [q3, B.y])   # Disc fixed frame
    CO = NO.locatenew('CO', q4*N.x + q5*N.y + q6*N.z)      # Disc center

    # Disc angular velocity in N expressed using time derivatives of coordinates
    w_c_n_qd = C.ang_vel_in(N)
    w_b_n_qd = B.ang_vel_in(N)

    # Inertial angular velocity and angular acceleration of disc fixed frame
    C.set_ang_vel(N, u1*B.x + u2*B.y + u3*B.z)

    # Disc center velocity in N expressed using time derivatives of coordinates
    v_co_n_qd = CO.pos_from(NO).dt(N)

    # Disc center velocity in N expressed using generalized speeds
    CO.set_vel(N, u4*C.x + u5*C.y + u6*C.z)

    # Disc Ground Contact Point
    P = CO.locatenew('P', r*B.z)
    P.v2pt_theory(CO, N, C)

    # Configuration constraint
    f_c = Matrix([q6 - dot(CO.pos_from(P), N.z)])

    # Velocity level constraints
    f_v = Matrix([dot(P.vel(N), uv) for uv in C])

    # Kinematic differential equations
    kindiffs = Matrix([dot(w_c_n_qd - C.ang_vel_in(N), uv) for uv in B] +
                        [dot(v_co_n_qd - CO.vel(N), uv) for uv in N])
    qdots = solve(kindiffs, qd)

    # Set angular velocity of remaining frames
    B.set_ang_vel(N, w_b_n_qd.subs(qdots))
    C.set_ang_acc(N, C.ang_vel_in(N).dt(B) + cross(B.ang_vel_in(N), C.ang_vel_in(N)))

    # Active forces
    F_CO = m*g*A.z

    # Create inertia dyadic of disc C about point CO
    I = (m * r**2) / 4
    J = (m * r**2) / 2
    I_C_CO = inertia(C, I, J, I)

    Disc = RigidBody('Disc', CO, C, m, (I_C_CO, CO))
    BL = [Disc]
    FL = [(CO, F_CO)]
    KM = KanesMethod(N, [q1, q2, q3, q4, q5], [u1, u2, u3], kd_eqs=kindiffs,
            q_dependent=[q6], configuration_constraints=f_c,
            u_dependent=[u4, u5, u6], velocity_constraints=f_v)
    (fr, fr_star) = KM.kanes_equations(BL, FL)

    # Test generalized form equations
    linearizer = KM.to_linearizer()
    assert linearizer.f_c == f_c
    assert linearizer.f_v == f_v
    assert linearizer.f_a == f_v.diff(t).subs(KM.kindiffdict())
    sol = solve(linearizer.f_0 + linearizer.f_1, qd)
    for qi in qdots.keys():
        assert sol[qi] == qdots[qi]
    assert simplify(linearizer.f_2 + linearizer.f_3 - fr - fr_star) == Matrix([0, 0, 0])

    # Perform the linearization
    # Precomputed operating point
    q_op = {q6: -r*cos(q2)}
    u_op = {u1: 0,
            u2: sin(q2)*q1d + q3d,
            u3: cos(q2)*q1d,
            u4: -r*(sin(q2)*q1d + q3d)*cos(q3),
            u5: 0,
            u6: -r*(sin(q2)*q1d + q3d)*sin(q3)}
    qd_op = {q2d: 0,
             q4d: -r*(sin(q2)*q1d + q3d)*cos(q1),
             q5d: -r*(sin(q2)*q1d + q3d)*sin(q1),
             q6d: 0}
    ud_op = {u1d: 4*g*sin(q2)/(5*r) + sin(2*q2)*q1d**2/2 + 6*cos(q2)*q1d*q3d/5,
             u2d: 0,
             u3d: 0,
             u4d: r*(sin(q2)*sin(q3)*q1d*q3d + sin(q3)*q3d**2),
             u5d: r*(4*g*sin(q2)/(5*r) + sin(2*q2)*q1d**2/2 + 6*cos(q2)*q1d*q3d/5),
             u6d: -r*(sin(q2)*cos(q3)*q1d*q3d + cos(q3)*q3d**2)}

    A, B = linearizer.linearize(op_point=[q_op, u_op, qd_op, ud_op], A_and_B=True, simplify=True)

    upright_nominal = {q1d: 0, q2: 0, m: 1, r: 1, g: 1}

    # Precomputed solution
    A_sol = Matrix([[0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [sin(q1)*q3d, 0, 0, 0, 0, -sin(q1), -cos(q1), 0],
                    [-cos(q1)*q3d, 0, 0, 0, 0, cos(q1), -sin(q1), 0],
                    [0, Rational(4, 5), 0, 0, 0, 0, 0, 6*q3d/5],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -2*q3d, 0, 0]])
    B_sol = Matrix([])

    # Check that linearization is correct
    assert A.subs(upright_nominal) == A_sol
    assert B.subs(upright_nominal) == B_sol

    # Check eigenvalues at critical speed are all zero:
    assert sympify(A.subs(upright_nominal).subs(q3d, 1/sqrt(3))).eigenvals() == {0: 8}

    # Check whether alternative solvers work
    # symengine doesn't support method='GJ'
    linearizer = KM.to_linearizer(linear_solver='GJ')
    A, B = linearizer.linearize(op_point=[q_op, u_op, qd_op, ud_op],
                                A_and_B=True, simplify=True)
    assert A.subs(upright_nominal) == A_sol
    assert B.subs(upright_nominal) == B_sol

def test_linearize_pendulum_kane_minimal():
    q1 = dynamicsymbols('q1')                     # angle of pendulum
    u1 = dynamicsymbols('u1')                     # Angular velocity
    q1d = dynamicsymbols('q1', 1)                 # Angular velocity
    L, m, t = symbols('L, m, t')
    g = 9.8

    # Compose world frame
    N = ReferenceFrame('N')
    pN = Point('N*')
    pN.set_vel(N, 0)

    # A.x is along the pendulum
    A = N.orientnew('A', 'axis', [q1, N.z])
    A.set_ang_vel(N, u1*N.z)

    # Locate point P relative to the origin N*
    P = pN.locatenew('P', L*A.x)
    P.v2pt_theory(pN, N, A)
    pP = Particle('pP', P, m)

    # Create Kinematic Differential Equations
    kde = Matrix([q1d - u1])

    # Input the force resultant at P
    R = m*g*N.x

    # Solve for eom with kanes method
    KM = KanesMethod(N, q_ind=[q1], u_ind=[u1], kd_eqs=kde)
    (fr, frstar) = KM.kanes_equations([pP], [(P, R)])

    # Linearize
    A, B, inp_vec = KM.linearize(A_and_B=True, simplify=True)

    assert A == Matrix([[0, 1], [-9.8*cos(q1)/L, 0]])
    assert B == Matrix([])

def test_linearize_pendulum_kane_nonminimal():
    # Create generalized coordinates and speeds for this non-minimal realization
    # q1, q2 = N.x and N.y coordinates of pendulum
    # u1, u2 = N.x and N.y velocities of pendulum
    q1, q2 = dynamicsymbols('q1:3')
    q1d, q2d = dynamicsymbols('q1:3', level=1)
    u1, u2 = dynamicsymbols('u1:3')
    u1d, u2d = dynamicsymbols('u1:3', level=1)
    L, m, t = symbols('L, m, t')
    g = 9.8

    # Compose world frame
    N = ReferenceFrame('N')
    pN = Point('N*')
    pN.set_vel(N, 0)

    # A.x is along the pendulum
    theta1 = atan(q2/q1)
    A = N.orientnew('A', 'axis', [theta1, N.z])

    # Locate the pendulum mass
    P = pN.locatenew('P1', q1*N.x + q2*N.y)
    pP = Particle('pP', P, m)

    # Calculate the kinematic differential equations
    kde = Matrix([q1d - u1,
                  q2d - u2])
    dq_dict = solve(kde, [q1d, q2d])

    # Set velocity of point P
    P.set_vel(N, P.pos_from(pN).dt(N).subs(dq_dict))

    # Configuration constraint is length of pendulum
    f_c = Matrix([P.pos_from(pN).magnitude() - L])

    # Velocity constraint is that the velocity in the A.x direction is
    # always zero (the pendulum is never getting longer).
    f_v = Matrix([P.vel(N).express(A).dot(A.x)])
    f_v.simplify()

    # Acceleration constraints is the time derivative of the velocity constraint
    f_a = f_v.diff(t)
    f_a.simplify()

    # Input the force resultant at P
    R = m*g*N.x

    # Derive the equations of motion using the KanesMethod class.
    KM = KanesMethod(N, q_ind=[q2], u_ind=[u2], q_dependent=[q1],
            u_dependent=[u1], configuration_constraints=f_c,
            velocity_constraints=f_v, acceleration_constraints=f_a, kd_eqs=kde)
    (fr, frstar) = KM.kanes_equations([pP], [(P, R)])

    # Set the operating point to be straight down, and non-moving
    q_op = {q1: L, q2: 0}
    u_op = {u1: 0, u2: 0}
    ud_op = {u1d: 0, u2d: 0}

    A, B, inp_vec = KM.linearize(op_point=[q_op, u_op, ud_op], A_and_B=True,
                                 simplify=True)

    assert A.expand() == Matrix([[0, 1], [-9.8/L, 0]])
    assert B == Matrix([])


    # symengine doesn't support method='GJ'
    A, B, inp_vec = KM.linearize(op_point=[q_op, u_op, ud_op], A_and_B=True,
                                simplify=True, linear_solver='GJ')

    assert A.expand() == Matrix([[0, 1], [-9.8/L, 0]])
    assert B == Matrix([])

    A, B, inp_vec = KM.linearize(op_point=[q_op, u_op, ud_op],
                                 A_and_B=True,
                                 simplify=True,
                                 linear_solver=lambda A, b: A.LUsolve(b))

    assert A.expand() == Matrix([[0, 1], [-9.8/L, 0]])
    assert B == Matrix([])


def test_linearize_pendulum_lagrange_minimal():
    q1 = dynamicsymbols('q1')                     # angle of pendulum
    q1d = dynamicsymbols('q1', 1)                 # Angular velocity
    L, m, t = symbols('L, m, t')
    g = 9.8

    # Compose world frame
    N = ReferenceFrame('N')
    pN = Point('N*')
    pN.set_vel(N, 0)

    # A.x is along the pendulum
    A = N.orientnew('A', 'axis', [q1, N.z])
    A.set_ang_vel(N, q1d*N.z)

    # Locate point P relative to the origin N*
    P = pN.locatenew('P', L*A.x)
    P.v2pt_theory(pN, N, A)
    pP = Particle('pP', P, m)

    # Solve for eom with Lagranges method
    Lag = Lagrangian(N, pP)
    LM = LagrangesMethod(Lag, [q1], forcelist=[(P, m*g*N.x)], frame=N)
    LM.form_lagranges_equations()

    # Linearize
    A, B, inp_vec = LM.linearize([q1], [q1d], A_and_B=True)

    assert simplify(A) == Matrix([[0, 1], [-9.8*cos(q1)/L, 0]])
    assert B == Matrix([])

    # Check an alternative solver
    A, B, inp_vec = LM.linearize([q1], [q1d], A_and_B=True, linear_solver='GJ')

    assert simplify(A) == Matrix([[0, 1], [-9.8*cos(q1)/L, 0]])
    assert B == Matrix([])


def test_linearize_pendulum_lagrange_nonminimal():
    q1, q2 = dynamicsymbols('q1:3')
    q1d, q2d = dynamicsymbols('q1:3', level=1)
    L, m, t = symbols('L, m, t')
    g = 9.8
    # Compose World Frame
    N = ReferenceFrame('N')
    pN = Point('N*')
    pN.set_vel(N, 0)
    # A.x is along the pendulum
    theta1 = atan(q2/q1)
    A = N.orientnew('A', 'axis', [theta1, N.z])
    # Create point P, the pendulum mass
    P = pN.locatenew('P1', q1*N.x + q2*N.y)
    P.set_vel(N, P.pos_from(pN).dt(N))
    pP = Particle('pP', P, m)
    # Constraint Equations
    f_c = Matrix([q1**2 + q2**2 - L**2])
    # Calculate the lagrangian, and form the equations of motion
    Lag = Lagrangian(N, pP)
    LM = LagrangesMethod(Lag, [q1, q2], hol_coneqs=f_c, forcelist=[(P, m*g*N.x)], frame=N)
    LM.form_lagranges_equations()
    # Compose operating point
    op_point = {q1: L, q2: 0, q1d: 0, q2d: 0, q1d.diff(t): 0, q2d.diff(t): 0}
    # Solve for multiplier operating point
    lam_op = LM.solve_multipliers(op_point=op_point)
    op_point.update(lam_op)
    # Perform the Linearization
    A, B, inp_vec = LM.linearize([q2], [q2d], [q1], [q1d],
            op_point=op_point, A_and_B=True)
    assert simplify(A) == Matrix([[0, 1], [-9.8/L, 0]])
    assert B == Matrix([])

    # Check if passing a function to linear_solver works
    A, B, inp_vec = LM.linearize([q2], [q2d], [q1], [q1d], op_point=op_point,
                                 A_and_B=True, linear_solver=lambda A, b:
                                 A.LUsolve(b))
    assert simplify(A) == Matrix([[0, 1], [-9.8/L, 0]])
    assert B == Matrix([])

def test_linearize_rolling_disc_lagrange():
    q1, q2, q3 = q = dynamicsymbols('q1 q2 q3')
    q1d, q2d, q3d = qd = dynamicsymbols('q1 q2 q3', 1)
    r, m, g = symbols('r m g')

    N = ReferenceFrame('N')
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    L = Y.orientnew('L', 'Axis', [q2, Y.x])
    R = L.orientnew('R', 'Axis', [q3, L.y])

    C = Point('C')
    C.set_vel(N, 0)
    Dmc = C.locatenew('Dmc', r * L.z)
    Dmc.v2pt_theory(C, N, R)

    I = inertia(L, m / 4 * r**2, m / 2 * r**2, m / 4 * r**2)
    BodyD = RigidBody('BodyD', Dmc, R, m, (I, Dmc))
    BodyD.potential_energy = - m * g * r * cos(q2)

    Lag = Lagrangian(N, BodyD)
    l = LagrangesMethod(Lag, q)
    l.form_lagranges_equations()

    # Linearize about steady-state upright rolling
    op_point = {q1: 0, q2: 0, q3: 0,
                q1d: 0, q2d: 0,
                q1d.diff(): 0, q2d.diff(): 0, q3d.diff(): 0}
    A = l.linearize(q_ind=q, qd_ind=qd, op_point=op_point, A_and_B=True)[0]
    sol = Matrix([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, -6*q3d, 0],
                  [0, -4*g/(5*r), 0, 6*q3d/5, 0, 0],
                  [0, 0, 0, 0, 0, 0]])

    assert A == sol

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\tests\test_sympy_parser.py ===
# -*- coding: utf-8 -*-


import builtins
import types

from sympy.assumptions import Q
from sympy.core import Symbol, Function, Float, Rational, Integer, I, Mul, Pow, Eq, Lt, Le, Gt, Ge, Ne
from sympy.functions import exp, factorial, factorial2, sin, Min, Max
from sympy.logic import And
from sympy.series import Limit
from sympy.testing.pytest import raises

from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, rationalize, TokenError,
    split_symbols, implicit_multiplication, convert_equals_signs,
    convert_xor, function_exponentiation, lambda_notation, auto_symbol,
    repeated_decimals, implicit_multiplication_application,
    auto_number, factorial_notation, implicit_application,
    _transformation, T
    )


def test_sympy_parser():
    x = Symbol('x')
    inputs = {
        '2*x': 2 * x,
        '3.00': Float(3),
        '22/7': Rational(22, 7),
        '2+3j': 2 + 3*I,
        'exp(x)': exp(x),
        'x!': factorial(x),
        'x!!': factorial2(x),
        '(x + 1)! - 1': factorial(x + 1) - 1,
        '3.[3]': Rational(10, 3),
        '.0[3]': Rational(1, 30),
        '3.2[3]': Rational(97, 30),
        '1.3[12]': Rational(433, 330),
        '1 + 3.[3]': Rational(13, 3),
        '1 + .0[3]': Rational(31, 30),
        '1 + 3.2[3]': Rational(127, 30),
        '.[0011]': Rational(1, 909),
        '0.1[00102] + 1': Rational(366697, 333330),
        '1.[0191]': Rational(10190, 9999),
        '10!': 3628800,
        '-(2)': -Integer(2),
        '[-1, -2, 3]': [Integer(-1), Integer(-2), Integer(3)],
        'Symbol("x").free_symbols': x.free_symbols,
        "S('S(3).n(n=3)')": Float(3, 3),
        'factorint(12, visual=True)': Mul(
            Pow(2, 2, evaluate=False),
            Pow(3, 1, evaluate=False),
            evaluate=False),
        'Limit(sin(x), x, 0, dir="-")': Limit(sin(x), x, 0, dir='-'),
        'Q.even(x)': Q.even(x),


    }
    for text, result in inputs.items():
        assert parse_expr(text) == result

    raises(TypeError, lambda:
        parse_expr('x', standard_transformations))
    raises(TypeError, lambda:
        parse_expr('x', transformations=lambda x,y: 1))
    raises(TypeError, lambda:
        parse_expr('x', transformations=(lambda x,y: 1,)))
    raises(TypeError, lambda: parse_expr('x', transformations=((),)))
    raises(TypeError, lambda: parse_expr('x', {}, [], []))
    raises(TypeError, lambda: parse_expr('x', [], [], {}))
    raises(TypeError, lambda: parse_expr('x', [], [], {}))


def test_rationalize():
    inputs = {
        '0.123': Rational(123, 1000)
    }
    transformations = standard_transformations + (rationalize,)
    for text, result in inputs.items():
        assert parse_expr(text, transformations=transformations) == result


def test_factorial_fail():
    inputs = ['x!!!', 'x!!!!', '(!)']


    for text in inputs:
        try:
            parse_expr(text)
            assert False
        except TokenError:
            assert True


def test_repeated_fail():
    inputs = ['1[1]', '.1e1[1]', '0x1[1]', '1.1j[1]', '1.1[1 + 1]',
        '0.1[[1]]', '0x1.1[1]']


    # All are valid Python, so only raise TypeError for invalid indexing
    for text in inputs:
        raises(TypeError, lambda: parse_expr(text))


    inputs = ['0.1[', '0.1[1', '0.1[]']
    for text in inputs:
        raises((TokenError, SyntaxError), lambda: parse_expr(text))


def test_repeated_dot_only():
    assert parse_expr('.[1]') == Rational(1, 9)
    assert parse_expr('1 + .[1]') == Rational(10, 9)


def test_local_dict():
    local_dict = {
        'my_function': lambda x: x + 2
    }
    inputs = {
        'my_function(2)': Integer(4)
    }
    for text, result in inputs.items():
        assert parse_expr(text, local_dict=local_dict) == result


def test_local_dict_split_implmult():
    t = standard_transformations + (split_symbols, implicit_multiplication,)
    w = Symbol('w', real=True)
    y = Symbol('y')
    assert parse_expr('yx', local_dict={'x':w}, transformations=t) == y*w


def test_local_dict_symbol_to_fcn():
    x = Symbol('x')
    d = {'foo': Function('bar')}
    assert parse_expr('foo(x)', local_dict=d) == d['foo'](x)
    d = {'foo': Symbol('baz')}
    raises(TypeError, lambda: parse_expr('foo(x)', local_dict=d))


def test_global_dict():
    global_dict = {
        'Symbol': Symbol
    }
    inputs = {
        'Q & S': And(Symbol('Q'), Symbol('S'))
    }
    for text, result in inputs.items():
        assert parse_expr(text, global_dict=global_dict) == result


def test_no_globals():

    # Replicate creating the default global_dict:
    default_globals = {}
    exec('from sympy import *', default_globals)
    builtins_dict = vars(builtins)
    for name, obj in builtins_dict.items():
        if isinstance(obj, types.BuiltinFunctionType):
            default_globals[name] = obj
    default_globals['max'] = Max
    default_globals['min'] = Min

    # Need to include Symbol or parse_expr will not work:
    default_globals.pop('Symbol')
    global_dict = {'Symbol':Symbol}

    for name in default_globals:
        obj = parse_expr(name, global_dict=global_dict)
        assert obj == Symbol(name)


def test_issue_2515():
    raises(TokenError, lambda: parse_expr('(()'))
    raises(TokenError, lambda: parse_expr('"""'))


def test_issue_7663():
    x = Symbol('x')
    e = '2*(x+1)'
    assert parse_expr(e, evaluate=False) == parse_expr(e, evaluate=False)
    assert parse_expr(e, evaluate=False).equals(2*(x+1))

def test_recursive_evaluate_false_10560():
    inputs = {
        '4*-3' : '4*-3',
        '-4*3' : '(-4)*3',
        "-2*x*y": '(-2)*x*y',
        "x*-4*x": "x*(-4)*x"
    }
    for text, result in inputs.items():
        assert parse_expr(text, evaluate=False) == parse_expr(result, evaluate=False)


def test_function_evaluate_false():
    inputs = [
        'Abs(0)', 'im(0)', 're(0)', 'sign(0)', 'arg(0)', 'conjugate(0)',
        'acos(0)', 'acot(0)', 'acsc(0)', 'asec(0)', 'asin(0)', 'atan(0)',
        'acosh(0)', 'acoth(0)', 'acsch(0)', 'asech(0)', 'asinh(0)', 'atanh(0)',
        'cos(0)', 'cot(0)', 'csc(0)', 'sec(0)', 'sin(0)', 'tan(0)',
        'cosh(0)', 'coth(0)', 'csch(0)', 'sech(0)', 'sinh(0)', 'tanh(0)',
        'exp(0)', 'log(0)', 'sqrt(0)',
    ]
    for case in inputs:
        expr = parse_expr(case, evaluate=False)
        assert case == str(expr) != str(expr.doit())
    assert str(parse_expr('ln(0)', evaluate=False)) == 'log(0)'
    assert str(parse_expr('cbrt(0)', evaluate=False)) == '0**(1/3)'


def test_issue_10773():
    inputs = {
    '-10/5': '(-10)/5',
    '-10/-5' : '(-10)/(-5)',
    }
    for text, result in inputs.items():
        assert parse_expr(text, evaluate=False) == parse_expr(result, evaluate=False)


def test_split_symbols():
    transformations = standard_transformations + \
                      (split_symbols, implicit_multiplication,)
    x = Symbol('x')
    y = Symbol('y')
    xy = Symbol('xy')


    assert parse_expr("xy") == xy
    assert parse_expr("xy", transformations=transformations) == x*y


def test_split_symbols_function():
    transformations = standard_transformations + \
                      (split_symbols, implicit_multiplication,)
    x = Symbol('x')
    y = Symbol('y')
    a = Symbol('a')
    f = Function('f')


    assert parse_expr("ay(x+1)", transformations=transformations) == a*y*(x+1)
    assert parse_expr("af(x+1)", transformations=transformations,
                      local_dict={'f':f}) == a*f(x+1)


def test_functional_exponent():
    t = standard_transformations + (convert_xor, function_exponentiation)
    x = Symbol('x')
    y = Symbol('y')
    a = Symbol('a')
    yfcn = Function('y')
    assert parse_expr("sin^2(x)", transformations=t) == (sin(x))**2
    assert parse_expr("sin^y(x)", transformations=t) == (sin(x))**y
    assert parse_expr("exp^y(x)", transformations=t) == (exp(x))**y
    assert parse_expr("E^y(x)", transformations=t) == exp(yfcn(x))
    assert parse_expr("a^y(x)", transformations=t) == a**(yfcn(x))


def test_match_parentheses_implicit_multiplication():
    transformations = standard_transformations + \
                      (implicit_multiplication,)
    raises(TokenError, lambda: parse_expr('(1,2),(3,4]',transformations=transformations))


def test_convert_equals_signs():
    transformations = standard_transformations + \
                        (convert_equals_signs, )
    x = Symbol('x')
    y = Symbol('y')
    assert parse_expr("1*2=x", transformations=transformations) == Eq(2, x)
    assert parse_expr("y = x", transformations=transformations) == Eq(y, x)
    assert parse_expr("(2*y = x) = False",
        transformations=transformations) == Eq(Eq(2*y, x), False)


def test_parse_function_issue_3539():
    x = Symbol('x')
    f = Function('f')
    assert parse_expr('f(x)') == f(x)

def test_issue_24288():
    assert parse_expr("1 < 2", evaluate=False) == Lt(1, 2, evaluate=False)
    assert parse_expr("1 <= 2", evaluate=False) == Le(1, 2, evaluate=False)
    assert parse_expr("1 > 2", evaluate=False) == Gt(1, 2, evaluate=False)
    assert parse_expr("1 >= 2", evaluate=False) == Ge(1, 2, evaluate=False)
    assert parse_expr("1 != 2", evaluate=False) == Ne(1, 2, evaluate=False)
    assert parse_expr("1 == 2", evaluate=False) == Eq(1, 2, evaluate=False)
    assert parse_expr("1 < 2 < 3", evaluate=False) == And(Lt(1, 2, evaluate=False), Lt(2, 3, evaluate=False), evaluate=False)
    assert parse_expr("1 <= 2 <= 3", evaluate=False) == And(Le(1, 2, evaluate=False), Le(2, 3, evaluate=False), evaluate=False)
    assert parse_expr("1 < 2 <= 3 < 4", evaluate=False) == \
        And(Lt(1, 2, evaluate=False), Le(2, 3, evaluate=False), Lt(3, 4, evaluate=False), evaluate=False)
    # Valid Python relational operators that SymPy does not decide how to handle them yet
    raises(ValueError, lambda: parse_expr("1 in 2", evaluate=False))
    raises(ValueError, lambda: parse_expr("1 is 2", evaluate=False))
    raises(ValueError, lambda: parse_expr("1 not in 2", evaluate=False))
    raises(ValueError, lambda: parse_expr("1 is not 2", evaluate=False))

def test_split_symbols_numeric():
    transformations = (
        standard_transformations +
        (implicit_multiplication_application,))

    n = Symbol('n')
    expr1 = parse_expr('2**n * 3**n')
    expr2 = parse_expr('2**n3**n', transformations=transformations)
    assert expr1 == expr2 == 2**n*3**n

    expr1 = parse_expr('n12n34', transformations=transformations)
    assert expr1 == n*12*n*34


def test_unicode_names():
    assert parse_expr('α') == Symbol('α')


def test_python3_features():
    assert parse_expr("123_456") == 123456
    assert parse_expr("1.2[3_4]") == parse_expr("1.2[34]") == Rational(611, 495)
    assert parse_expr("1.2[012_012]") == parse_expr("1.2[012012]") == Rational(400, 333)
    assert parse_expr('.[3_4]') == parse_expr('.[34]') == Rational(34, 99)
    assert parse_expr('.1[3_4]') == parse_expr('.1[34]') == Rational(133, 990)
    assert parse_expr('123_123.123_123[3_4]') == parse_expr('123123.123123[34]') == Rational(12189189189211, 99000000)


def test_issue_19501():
    x = Symbol('x')
    eq = parse_expr('E**x(1+x)', local_dict={'x': x}, transformations=(
        standard_transformations +
        (implicit_multiplication_application,)))
    assert eq.free_symbols == {x}


def test_parsing_definitions():
    from sympy.abc import x
    assert len(_transformation) == 12  # if this changes, extend below
    assert _transformation[0] == lambda_notation
    assert _transformation[1] == auto_symbol
    assert _transformation[2] == repeated_decimals
    assert _transformation[3] == auto_number
    assert _transformation[4] == factorial_notation
    assert _transformation[5] == implicit_multiplication_application
    assert _transformation[6] == convert_xor
    assert _transformation[7] == implicit_application
    assert _transformation[8] == implicit_multiplication
    assert _transformation[9] == convert_equals_signs
    assert _transformation[10] == function_exponentiation
    assert _transformation[11] == rationalize
    assert T[:5] == T[0,1,2,3,4] == standard_transformations
    t = _transformation
    assert T[-1, 0] == (t[len(t) - 1], t[0])
    assert T[:5, 8] == standard_transformations + (t[8],)
    assert parse_expr('0.3x^2', transformations='all') == 3*x**2/10
    assert parse_expr('sin 3x', transformations='implicit') == sin(3*x)


def test_builtins():
    cases = [
        ('abs(x)', 'Abs(x)'),
        ('max(x, y)', 'Max(x, y)'),
        ('min(x, y)', 'Min(x, y)'),
        ('pow(x, y)', 'Pow(x, y)'),
    ]
    for built_in_func_call, sympy_func_call in cases:
        assert parse_expr(built_in_func_call) == parse_expr(sympy_func_call)
    assert str(parse_expr('pow(38, -1, 97)')) == '23'


def test_issue_22822():
    raises(ValueError, lambda: parse_expr('x', {'': 1}))
    data = {'some_parameter': None}
    assert parse_expr('some_parameter is None', data) is True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\test_downstream.py ===
"""
Testing that we work in the downstream packages
"""
import array
import subprocess
import sys

import numpy as np
import pytest

from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    TimedeltaIndex,
)
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    TimedeltaArray,
)
from pandas.util.version import Version


@pytest.fixture
def df():
    return DataFrame({"A": [1, 2, 3]})


def test_dask(df):
    # dask sets "compute.use_numexpr" to False, so catch the current value
    # and ensure to reset it afterwards to avoid impacting other tests
    olduse = pd.get_option("compute.use_numexpr")

    try:
        pytest.importorskip("toolz")
        dd = pytest.importorskip("dask.dataframe")

        ddf = dd.from_pandas(df, npartitions=3)
        assert ddf.A is not None
        assert ddf.compute() is not None
    finally:
        pd.set_option("compute.use_numexpr", olduse)


def test_dask_ufunc():
    # dask sets "compute.use_numexpr" to False, so catch the current value
    # and ensure to reset it afterwards to avoid impacting other tests
    olduse = pd.get_option("compute.use_numexpr")

    try:
        da = pytest.importorskip("dask.array")
        dd = pytest.importorskip("dask.dataframe")

        s = Series([1.5, 2.3, 3.7, 4.0])
        ds = dd.from_pandas(s, npartitions=2)

        result = da.fix(ds).compute()
        expected = np.fix(s)
        tm.assert_series_equal(result, expected)
    finally:
        pd.set_option("compute.use_numexpr", olduse)


def test_construct_dask_float_array_int_dtype_match_ndarray():
    # GH#40110 make sure we treat a float-dtype dask array with the same
    #  rules we would for an ndarray
    dd = pytest.importorskip("dask.dataframe")

    arr = np.array([1, 2.5, 3])
    darr = dd.from_array(arr)

    res = Series(darr)
    expected = Series(arr)
    tm.assert_series_equal(res, expected)

    # GH#49599 in 2.0 we raise instead of silently ignoring the dtype
    msg = "Trying to coerce float values to integers"
    with pytest.raises(ValueError, match=msg):
        Series(darr, dtype="i8")

    msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
    arr[2] = np.nan
    with pytest.raises(IntCastingNaNError, match=msg):
        Series(darr, dtype="i8")
    # which is the same as we get with a numpy input
    with pytest.raises(IntCastingNaNError, match=msg):
        Series(arr, dtype="i8")


def test_xarray(df):
    pytest.importorskip("xarray")

    assert df.to_xarray() is not None


def test_xarray_cftimeindex_nearest():
    # https://github.com/pydata/xarray/issues/3751
    cftime = pytest.importorskip("cftime")
    xarray = pytest.importorskip("xarray")

    times = xarray.cftime_range("0001", periods=2)
    key = cftime.DatetimeGregorian(2000, 1, 1)
    result = times.get_indexer([key], method="nearest")
    expected = 1
    assert result == expected


@pytest.mark.single_cpu
def test_oo_optimizable():
    # GH 21071
    subprocess.check_call([sys.executable, "-OO", "-c", "import pandas"])


@pytest.mark.single_cpu
def test_oo_optimized_datetime_index_unpickle():
    # GH 42866
    subprocess.check_call(
        [
            sys.executable,
            "-OO",
            "-c",
            (
                "import pandas as pd, pickle; "
                "pickle.loads(pickle.dumps(pd.date_range('2021-01-01', periods=1)))"
            ),
        ]
    )


def test_statsmodels():
    smf = pytest.importorskip("statsmodels.formula.api")

    df = DataFrame(
        {"Lottery": range(5), "Literacy": range(5), "Pop1831": range(100, 105)}
    )
    smf.ols("Lottery ~ Literacy + np.log(Pop1831)", data=df).fit()


def test_scikit_learn():
    pytest.importorskip("sklearn")
    from sklearn import (
        datasets,
        svm,
    )

    digits = datasets.load_digits()
    clf = svm.SVC(gamma=0.001, C=100.0)
    clf.fit(digits.data[:-1], digits.target[:-1])
    clf.predict(digits.data[-1:])


def test_seaborn():
    seaborn = pytest.importorskip("seaborn")
    tips = DataFrame(
        {"day": pd.date_range("2023", freq="D", periods=5), "total_bill": range(5)}
    )
    seaborn.stripplot(x="day", y="total_bill", data=tips)


def test_pandas_datareader():
    pytest.importorskip("pandas_datareader")


@pytest.mark.filterwarnings("ignore:Passing a BlockManager:DeprecationWarning")
def test_pyarrow(df):
    pyarrow = pytest.importorskip("pyarrow")
    table = pyarrow.Table.from_pandas(df)
    result = table.to_pandas()
    tm.assert_frame_equal(result, df)


def test_yaml_dump(df):
    # GH#42748
    yaml = pytest.importorskip("yaml")

    dumped = yaml.dump(df)

    loaded = yaml.load(dumped, Loader=yaml.Loader)
    tm.assert_frame_equal(df, loaded)

    loaded2 = yaml.load(dumped, Loader=yaml.UnsafeLoader)
    tm.assert_frame_equal(df, loaded2)


@pytest.mark.single_cpu
def test_missing_required_dependency():
    # GH 23868
    # To ensure proper isolation, we pass these flags
    # -S : disable site-packages
    # -s : disable user site-packages
    # -E : disable PYTHON* env vars, especially PYTHONPATH
    # https://github.com/MacPython/pandas-wheels/pull/50

    pyexe = sys.executable.replace("\\", "/")

    # We skip this test if pandas is installed as a site package. We first
    # import the package normally and check the path to the module before
    # executing the test which imports pandas with site packages disabled.
    call = [pyexe, "-c", "import pandas;print(pandas.__file__)"]
    output = subprocess.check_output(call).decode()
    if "site-packages" in output:
        pytest.skip("pandas installed as site package")

    # This test will fail if pandas is installed as a site package. The flags
    # prevent pandas being imported and the test will report Failed: DID NOT
    # RAISE <class 'subprocess.CalledProcessError'>
    call = [pyexe, "-sSE", "-c", "import pandas"]

    msg = (
        rf"Command '\['{pyexe}', '-sSE', '-c', 'import pandas'\]' "
        "returned non-zero exit status 1."
    )

    with pytest.raises(subprocess.CalledProcessError, match=msg) as exc:
        subprocess.check_output(call, stderr=subprocess.STDOUT)

    output = exc.value.stdout.decode()
    for name in ["numpy", "pytz", "dateutil"]:
        assert name in output


def test_frame_setitem_dask_array_into_new_col(request):
    # GH#47128

    # dask sets "compute.use_numexpr" to False, so catch the current value
    # and ensure to reset it afterwards to avoid impacting other tests
    olduse = pd.get_option("compute.use_numexpr")

    try:
        dask = pytest.importorskip("dask")
        da = pytest.importorskip("dask.array")
        if Version(dask.__version__) <= Version("2025.1.0") and Version(
            np.__version__
        ) >= Version("2.1"):
            request.applymarker(
                pytest.mark.xfail(reason="loc.__setitem__ incorrectly mutated column c")
            )

        dda = da.array([1, 2])
        df = DataFrame({"a": ["a", "b"]})
        df["b"] = dda
        df["c"] = dda
        df.loc[[False, True], "b"] = 100
        result = df.loc[[1], :]
        expected = DataFrame({"a": ["b"], "b": [100], "c": [2]}, index=[1])
        tm.assert_frame_equal(result, expected)
    finally:
        pd.set_option("compute.use_numexpr", olduse)


def test_pandas_priority():
    # GH#48347

    class MyClass:
        __pandas_priority__ = 5000

        def __radd__(self, other):
            return self

    left = MyClass()
    right = Series(range(3))

    assert right.__add__(left) is NotImplemented
    assert right + left is left


@pytest.fixture(
    params=[
        "memoryview",
        "array",
        pytest.param("dask", marks=td.skip_if_no("dask.array")),
        pytest.param("xarray", marks=td.skip_if_no("xarray")),
    ]
)
def array_likes(request):
    """
    Fixture giving a numpy array and a parametrized 'data' object, which can
    be a memoryview, array, dask or xarray object created from the numpy array.
    """
    # GH#24539 recognize e.g xarray, dask, ...
    arr = np.array([1, 2, 3], dtype=np.int64)

    name = request.param
    if name == "memoryview":
        data = memoryview(arr)
    elif name == "array":
        data = array.array("i", arr)
    elif name == "dask":
        import dask.array

        data = dask.array.array(arr)
    elif name == "xarray":
        import xarray as xr

        data = xr.DataArray(arr)

    return arr, data


@pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
def test_from_obscure_array(dtype, array_likes):
    # GH#24539 recognize e.g xarray, dask, ...
    # Note: we dont do this for PeriodArray bc _from_sequence won't accept
    #  an array of integers
    # TODO: could check with arraylike of Period objects
    arr, data = array_likes

    cls = {"M8[ns]": DatetimeArray, "m8[ns]": TimedeltaArray}[dtype]

    depr_msg = f"{cls.__name__}.__init__ is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        expected = cls(arr)
    result = cls._from_sequence(data, dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    if not isinstance(data, memoryview):
        # FIXME(GH#44431) these raise on memoryview and attempted fix
        #  fails on py3.10
        func = {"M8[ns]": pd.to_datetime, "m8[ns]": pd.to_timedelta}[dtype]
        result = func(arr).array
        expected = func(data).array
        tm.assert_equal(result, expected)

    # Let's check the Indexes while we're here
    idx_cls = {"M8[ns]": DatetimeIndex, "m8[ns]": TimedeltaIndex}[dtype]
    result = idx_cls(arr)
    expected = idx_cls(data)
    tm.assert_index_equal(result, expected)


def test_dataframe_consortium() -> None:
    """
    Test some basic methods of the dataframe consortium standard.

    Full testing is done at https://github.com/data-apis/dataframe-api-compat,
    this is just to check that the entry point works as expected.
    """
    pytest.importorskip("dataframe_api_compat")
    df_pd = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = df_pd.__dataframe_consortium_standard__()
    result_1 = df.get_column_names()
    expected_1 = ["a", "b"]
    assert result_1 == expected_1

    ser = Series([1, 2, 3], name="a")
    col = ser.__column_consortium_standard__()
    assert col.name == "a"


def test_xarray_coerce_unit():
    # GH44053
    xr = pytest.importorskip("xarray")

    arr = xr.DataArray([1, 2, 3])
    result = pd.to_datetime(arr, unit="ns")
    expected = DatetimeIndex(
        [
            "1970-01-01 00:00:00.000000001",
            "1970-01-01 00:00:00.000000002",
            "1970-01-01 00:00:00.000000003",
        ],
        dtype="datetime64[ns]",
        freq=None,
    )
    tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_jax.py ===
from sympy.concrete.summations import Sum
from sympy.core.mod import Mod
from sympy.core.relational import (Equality, Unequality)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.utilities.lambdify import lambdify

from sympy.abc import x, i, j, a, b, c, d
from sympy.core import Function, Pow, Symbol
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.cfunctions import log1p, expm1, hypot, log10, exp2, log2, Sqrt
from sympy.tensor.array import Array
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, \
    PermuteDims, ArrayDiagonal
from sympy.printing.numpy import JaxPrinter, _jax_known_constants, _jax_known_functions
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array

from sympy.testing.pytest import skip, raises
from sympy.external import import_module

# Unlike NumPy which will aggressively promote operands to double precision,
# jax always uses single precision. Double precision in jax can be
# configured before the call to `import jax`, however this must be explicitly
# configured and is not fully supported. Thus, the tests here have been modified
# from the tests in test_numpy.py, only in the fact that they assert lambdify
# function accuracy to only single precision accuracy.
# https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision

jax = import_module('jax')

if jax:
    deafult_float_info = jax.numpy.finfo(jax.numpy.array([]).dtype)
    JAX_DEFAULT_EPSILON = deafult_float_info.eps


def test_jax_piecewise_regression():
    """
    NumPyPrinter needs to print Piecewise()'s choicelist as a list to avoid
    breaking compatibility with numpy 1.8. This is not necessary in numpy 1.9+.
    See gh-9747 and gh-9749 for details.
    """
    printer = JaxPrinter()
    p = Piecewise((1, x < 0), (0, True))
    assert printer.doprint(p) == \
        'jax.numpy.select([jax.numpy.less(x, 0),True], [1,0], default=jax.numpy.nan)'
    assert printer.module_imports == {'jax.numpy': {'select', 'less', 'nan'}}


def test_jax_logaddexp():
    lae = logaddexp(a, b)
    assert JaxPrinter().doprint(lae) == 'jax.numpy.logaddexp(a, b)'
    lae2 = logaddexp2(a, b)
    assert JaxPrinter().doprint(lae2) == 'jax.numpy.logaddexp2(a, b)'


def test_jax_sum():
    if not jax:
        skip("JAX not installed")

    s = Sum(x ** i, (i, a, b))
    f = lambdify((a, b, x), s, 'jax')

    a_, b_ = 0, 10
    x_ = jax.numpy.linspace(-1, +1, 10)
    assert jax.numpy.allclose(f(a_, b_, x_), sum(x_ ** i_ for i_ in range(a_, b_ + 1)))

    s = Sum(i * x, (i, a, b))
    f = lambdify((a, b, x), s, 'jax')

    a_, b_ = 0, 10
    x_ = jax.numpy.linspace(-1, +1, 10)
    assert jax.numpy.allclose(f(a_, b_, x_), sum(i_ * x_ for i_ in range(a_, b_ + 1)))


def test_jax_multiple_sums():
    if not jax:
        skip("JAX not installed")

    s = Sum((x + j) * i, (i, a, b), (j, c, d))
    f = lambdify((a, b, c, d, x), s, 'jax')

    a_, b_ = 0, 10
    c_, d_ = 11, 21
    x_ = jax.numpy.linspace(-1, +1, 10)
    assert jax.numpy.allclose(f(a_, b_, c_, d_, x_),
                       sum((x_ + j_) * i_ for i_ in range(a_, b_ + 1) for j_ in range(c_, d_ + 1)))


def test_jax_codegen_einsum():
    if not jax:
        skip("JAX not installed")

    M = MatrixSymbol("M", 2, 2)
    N = MatrixSymbol("N", 2, 2)

    cg = convert_matrix_to_array(M * N)
    f = lambdify((M, N), cg, 'jax')

    ma = jax.numpy.array([[1, 2], [3, 4]])
    mb = jax.numpy.array([[1,-2], [-1, 3]])
    assert (f(ma, mb) == jax.numpy.matmul(ma, mb)).all()


def test_jax_codegen_extra():
    if not jax:
        skip("JAX not installed")

    M = MatrixSymbol("M", 2, 2)
    N = MatrixSymbol("N", 2, 2)
    P = MatrixSymbol("P", 2, 2)
    Q = MatrixSymbol("Q", 2, 2)
    ma = jax.numpy.array([[1, 2], [3, 4]])
    mb = jax.numpy.array([[1,-2], [-1, 3]])
    mc = jax.numpy.array([[2, 0], [1, 2]])
    md = jax.numpy.array([[1,-1], [4, 7]])

    cg = ArrayTensorProduct(M, N)
    f = lambdify((M, N), cg, 'jax')
    assert (f(ma, mb) == jax.numpy.einsum(ma, [0, 1], mb, [2, 3])).all()

    cg = ArrayAdd(M, N)
    f = lambdify((M, N), cg, 'jax')
    assert (f(ma, mb) == ma+mb).all()

    cg = ArrayAdd(M, N, P)
    f = lambdify((M, N, P), cg, 'jax')
    assert (f(ma, mb, mc) == ma+mb+mc).all()

    cg = ArrayAdd(M, N, P, Q)
    f = lambdify((M, N, P, Q), cg, 'jax')
    assert (f(ma, mb, mc, md) == ma+mb+mc+md).all()

    cg = PermuteDims(M, [1, 0])
    f = lambdify((M,), cg, 'jax')
    assert (f(ma) == ma.T).all()

    cg = PermuteDims(ArrayTensorProduct(M, N), [1, 2, 3, 0])
    f = lambdify((M, N), cg, 'jax')
    assert (f(ma, mb) == jax.numpy.transpose(jax.numpy.einsum(ma, [0, 1], mb, [2, 3]), (1, 2, 3, 0))).all()

    cg = ArrayDiagonal(ArrayTensorProduct(M, N), (1, 2))
    f = lambdify((M, N), cg, 'jax')
    assert (f(ma, mb) == jax.numpy.diagonal(jax.numpy.einsum(ma, [0, 1], mb, [2, 3]), axis1=1, axis2=2)).all()


def test_jax_relational():
    if not jax:
        skip("JAX not installed")

    e = Equality(x, 1)

    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [False, True, False])

    e = Unequality(x, 1)

    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [True, False, True])

    e = (x < 1)

    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [True, False, False])

    e = (x <= 1)

    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [True, True, False])

    e = (x > 1)

    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [False, False, True])

    e = (x >= 1)

    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [False, True, True])

    # Multi-condition expressions
    e = (x >= 1) & (x < 2)
    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [False, True, False])

    e = (x >= 1) | (x < 2)
    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [True, True, True])

def test_jax_mod():
    if not jax:
        skip("JAX not installed")

    e = Mod(a, b)
    f = lambdify((a, b), e, 'jax')

    a_ = jax.numpy.array([0, 1, 2, 3])
    b_ = 2
    assert jax.numpy.array_equal(f(a_, b_), [0, 1, 0, 1])

    a_ = jax.numpy.array([0, 1, 2, 3])
    b_ = jax.numpy.array([2, 2, 2, 2])
    assert jax.numpy.array_equal(f(a_, b_), [0, 1, 0, 1])

    a_ = jax.numpy.array([2, 3, 4, 5])
    b_ = jax.numpy.array([2, 3, 4, 5])
    assert jax.numpy.array_equal(f(a_, b_), [0, 0, 0, 0])


def test_jax_pow():
    if not jax:
        skip('JAX not installed')

    expr = Pow(2, -1, evaluate=False)
    f = lambdify([], expr, 'jax')
    assert f() == 0.5


def test_jax_expm1():
    if not jax:
        skip("JAX not installed")

    f = lambdify((a,), expm1(a), 'jax')
    assert abs(f(1e-10) - 1e-10 - 5e-21) <= 1e-10 * JAX_DEFAULT_EPSILON


def test_jax_log1p():
    if not jax:
        skip("JAX not installed")

    f = lambdify((a,), log1p(a), 'jax')
    assert abs(f(1e-99) - 1e-99) <= 1e-99 * JAX_DEFAULT_EPSILON

def test_jax_hypot():
    if not jax:
        skip("JAX not installed")
    assert abs(lambdify((a, b), hypot(a, b), 'jax')(3, 4) - 5) <= JAX_DEFAULT_EPSILON

def test_jax_log10():
    if not jax:
        skip("JAX not installed")

    assert abs(lambdify((a,), log10(a), 'jax')(100) - 2) <= JAX_DEFAULT_EPSILON


def test_jax_exp2():
    if not jax:
        skip("JAX not installed")
    assert abs(lambdify((a,), exp2(a), 'jax')(5) - 32) <= JAX_DEFAULT_EPSILON


def test_jax_log2():
    if not jax:
        skip("JAX not installed")
    assert abs(lambdify((a,), log2(a), 'jax')(256) - 8) <= JAX_DEFAULT_EPSILON


def test_jax_Sqrt():
    if not jax:
        skip("JAX not installed")
    assert abs(lambdify((a,), Sqrt(a), 'jax')(4) - 2) <= JAX_DEFAULT_EPSILON


def test_jax_sqrt():
    if not jax:
        skip("JAX not installed")
    assert abs(lambdify((a,), sqrt(a), 'jax')(4) - 2) <= JAX_DEFAULT_EPSILON


def test_jax_matsolve():
    if not jax:
        skip("JAX not installed")

    M = MatrixSymbol("M", 3, 3)
    x = MatrixSymbol("x", 3, 1)

    expr = M**(-1) * x + x
    matsolve_expr = MatrixSolve(M, x) + x

    f = lambdify((M, x), expr, 'jax')
    f_matsolve = lambdify((M, x), matsolve_expr, 'jax')

    m0 = jax.numpy.array([[1, 2, 3], [3, 2, 5], [5, 6, 7]])
    assert jax.numpy.linalg.matrix_rank(m0) == 3

    x0 = jax.numpy.array([3, 4, 5])

    assert jax.numpy.allclose(f_matsolve(m0, x0), f(m0, x0))


def test_16857():
    if not jax:
        skip("JAX not installed")

    a_1 = MatrixSymbol('a_1', 10, 3)
    a_2 = MatrixSymbol('a_2', 10, 3)
    a_3 = MatrixSymbol('a_3', 10, 3)
    a_4 = MatrixSymbol('a_4', 10, 3)
    A = BlockMatrix([[a_1, a_2], [a_3, a_4]])
    assert A.shape == (20, 6)

    printer = JaxPrinter()
    assert printer.doprint(A) == 'jax.numpy.block([[a_1, a_2], [a_3, a_4]])'


def test_issue_17006():
    if not jax:
        skip("JAX not installed")

    M = MatrixSymbol("M", 2, 2)

    f = lambdify(M, M + Identity(2), 'jax')
    ma = jax.numpy.array([[1, 2], [3, 4]])
    mr = jax.numpy.array([[2, 2], [3, 5]])

    assert (f(ma) == mr).all()

    from sympy.core.symbol import symbols
    n = symbols('n', integer=True)
    N = MatrixSymbol("M", n, n)
    raises(NotImplementedError, lambda: lambdify(N, N + Identity(n), 'jax'))


def test_jax_array():
    assert JaxPrinter().doprint(Array(((1, 2), (3, 5)))) == 'jax.numpy.array([[1, 2], [3, 5]])'
    assert JaxPrinter().doprint(Array((1, 2))) == 'jax.numpy.array([1, 2])'


def test_jax_known_funcs_consts():
    assert _jax_known_constants['NaN'] == 'jax.numpy.nan'
    assert _jax_known_constants['EulerGamma'] == 'jax.numpy.euler_gamma'

    assert _jax_known_functions['acos'] == 'jax.numpy.arccos'
    assert _jax_known_functions['log'] == 'jax.numpy.log'


def test_jax_print_methods():
    prntr = JaxPrinter()
    assert hasattr(prntr, '_print_acos')
    assert hasattr(prntr, '_print_log')


def test_jax_printmethod():
    printer = JaxPrinter()
    assert hasattr(printer, 'printmethod')
    assert printer.printmethod == '_jaxcode'


def test_jax_custom_print_method():

    class expm1(Function):

        def _jaxcode(self, printer):
            x, = self.args
            function = f'expm1({printer._print(x)})'
            return printer._module_format(printer._module + '.' + function)

    printer = JaxPrinter()
    assert printer.doprint(expm1(Symbol('x'))) == 'jax.numpy.expm1(x)'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\tests\test_http.py ===
# -*- coding: utf-8 -*-
#
import os
import os.path
import socket
import ssl
import unittest

import websocket
from websocket._exceptions import WebSocketProxyException, WebSocketException
from websocket._http import (
    _get_addrinfo_list,
    _start_proxied_socket,
    _tunnel,
    connect,
    proxy_info,
    read_headers,
    HAVE_PYTHON_SOCKS,
)

"""
test_http.py
websocket - WebSocket client library for Python

Copyright 2024 engn33r

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

try:
    from python_socks._errors import ProxyConnectionError, ProxyError, ProxyTimeoutError
except:
    from websocket._http import ProxyConnectionError, ProxyError, ProxyTimeoutError

# Skip test to access the internet unless TEST_WITH_INTERNET == 1
TEST_WITH_INTERNET = os.environ.get("TEST_WITH_INTERNET", "0") == "1"
TEST_WITH_PROXY = os.environ.get("TEST_WITH_PROXY", "0") == "1"
# Skip tests relying on local websockets server unless LOCAL_WS_SERVER_PORT != -1
LOCAL_WS_SERVER_PORT = os.environ.get("LOCAL_WS_SERVER_PORT", "-1")
TEST_WITH_LOCAL_SERVER = LOCAL_WS_SERVER_PORT != "-1"


class SockMock:
    def __init__(self):
        self.data = []
        self.sent = []

    def add_packet(self, data):
        self.data.append(data)

    def gettimeout(self):
        return None

    def recv(self, bufsize):
        if self.data:
            e = self.data.pop(0)
            if isinstance(e, Exception):
                raise e
            if len(e) > bufsize:
                self.data.insert(0, e[bufsize:])
            return e[:bufsize]

    def send(self, data):
        self.sent.append(data)
        return len(data)

    def close(self):
        pass


class HeaderSockMock(SockMock):
    def __init__(self, fname):
        SockMock.__init__(self)
        path = os.path.join(os.path.dirname(__file__), fname)
        with open(path, "rb") as f:
            self.add_packet(f.read())


class OptsList:
    def __init__(self):
        self.timeout = 1
        self.sockopt = []
        self.sslopt = {"cert_reqs": ssl.CERT_NONE}


class HttpTest(unittest.TestCase):
    def test_read_header(self):
        status, header, _ = read_headers(HeaderSockMock("data/header01.txt"))
        self.assertEqual(status, 101)
        self.assertEqual(header["connection"], "Upgrade")
        # header02.txt is intentionally malformed
        self.assertRaises(
            WebSocketException, read_headers, HeaderSockMock("data/header02.txt")
        )

    def test_tunnel(self):
        self.assertRaises(
            WebSocketProxyException,
            _tunnel,
            HeaderSockMock("data/header01.txt"),
            "example.com",
            80,
            ("username", "password"),
        )
        self.assertRaises(
            WebSocketProxyException,
            _tunnel,
            HeaderSockMock("data/header02.txt"),
            "example.com",
            80,
            ("username", "password"),
        )

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def test_connect(self):
        # Not currently testing an actual proxy connection, so just check whether proxy errors are raised. This requires internet for a DNS lookup
        if HAVE_PYTHON_SOCKS:
            # Need this check, otherwise case where python_socks is not installed triggers
            # websocket._exceptions.WebSocketException: Python Socks is needed for SOCKS proxying but is not available
            self.assertRaises(
                (ProxyTimeoutError, OSError),
                _start_proxied_socket,
                "wss://example.com",
                OptsList(),
                proxy_info(
                    http_proxy_host="example.com",
                    http_proxy_port="8080",
                    proxy_type="socks4",
                    http_proxy_timeout=1,
                ),
            )
            self.assertRaises(
                (ProxyTimeoutError, OSError),
                _start_proxied_socket,
                "wss://example.com",
                OptsList(),
                proxy_info(
                    http_proxy_host="example.com",
                    http_proxy_port="8080",
                    proxy_type="socks4a",
                    http_proxy_timeout=1,
                ),
            )
            self.assertRaises(
                (ProxyTimeoutError, OSError),
                _start_proxied_socket,
                "wss://example.com",
                OptsList(),
                proxy_info(
                    http_proxy_host="example.com",
                    http_proxy_port="8080",
                    proxy_type="socks5",
                    http_proxy_timeout=1,
                ),
            )
            self.assertRaises(
                (ProxyTimeoutError, OSError),
                _start_proxied_socket,
                "wss://example.com",
                OptsList(),
                proxy_info(
                    http_proxy_host="example.com",
                    http_proxy_port="8080",
                    proxy_type="socks5h",
                    http_proxy_timeout=1,
                ),
            )
            self.assertRaises(
                ProxyConnectionError,
                connect,
                "wss://example.com",
                OptsList(),
                proxy_info(
                    http_proxy_host="127.0.0.1",
                    http_proxy_port=9999,
                    proxy_type="socks4",
                    http_proxy_timeout=1,
                ),
                None,
            )

        self.assertRaises(
            TypeError,
            _get_addrinfo_list,
            None,
            80,
            True,
            proxy_info(
                http_proxy_host="127.0.0.1", http_proxy_port="9999", proxy_type="http"
            ),
        )
        self.assertRaises(
            TypeError,
            _get_addrinfo_list,
            None,
            80,
            True,
            proxy_info(
                http_proxy_host="127.0.0.1", http_proxy_port="9999", proxy_type="http"
            ),
        )
        self.assertRaises(
            socket.timeout,
            connect,
            "wss://google.com",
            OptsList(),
            proxy_info(
                http_proxy_host="8.8.8.8",
                http_proxy_port=9999,
                proxy_type="http",
                http_proxy_timeout=1,
            ),
            None,
        )
        self.assertEqual(
            connect(
                "wss://google.com",
                OptsList(),
                proxy_info(
                    http_proxy_host="8.8.8.8", http_proxy_port=8080, proxy_type="http"
                ),
                True,
            ),
            (True, ("google.com", 443, "/")),
        )
        # The following test fails on Mac OS with a gaierror, not an OverflowError
        # self.assertRaises(OverflowError, connect, "wss://example.com", OptsList(), proxy_info(http_proxy_host="127.0.0.1", http_proxy_port=99999, proxy_type="socks4", timeout=2), False)

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    @unittest.skipUnless(
        TEST_WITH_PROXY, "This test requires a HTTP proxy to be running on port 8899"
    )
    @unittest.skipUnless(
        TEST_WITH_LOCAL_SERVER, "Tests using local websocket server are disabled"
    )
    def test_proxy_connect(self):
        ws = websocket.WebSocket()
        ws.connect(
            f"ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}",
            http_proxy_host="127.0.0.1",
            http_proxy_port="8899",
            proxy_type="http",
        )
        ws.send("Hello, Server")
        server_response = ws.recv()
        self.assertEqual(server_response, "Hello, Server")
        # self.assertEqual(_start_proxied_socket("wss://api.bitfinex.com/ws/2", OptsList(), proxy_info(http_proxy_host="127.0.0.1", http_proxy_port="8899", proxy_type="http"))[1], ("api.bitfinex.com", 443, '/ws/2'))
        self.assertEqual(
            _get_addrinfo_list(
                "api.bitfinex.com",
                443,
                True,
                proxy_info(
                    http_proxy_host="127.0.0.1",
                    http_proxy_port="8899",
                    proxy_type="http",
                ),
            ),
            (
                socket.getaddrinfo(
                    "127.0.0.1", 8899, 0, socket.SOCK_STREAM, socket.SOL_TCP
                ),
                True,
                None,
            ),
        )
        self.assertEqual(
            connect(
                "wss://api.bitfinex.com/ws/2",
                OptsList(),
                proxy_info(
                    http_proxy_host="127.0.0.1", http_proxy_port=8899, proxy_type="http"
                ),
                None,
            )[1],
            ("api.bitfinex.com", 443, "/ws/2"),
        )
        # TODO: Test SOCKS4 and SOCK5 proxies with unit tests

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def test_sslopt(self):
        ssloptions = {
            "check_hostname": False,
            "server_hostname": "ServerName",
            "ssl_version": ssl.PROTOCOL_TLS_CLIENT,
            "ciphers": "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:\
                        TLS_AES_128_GCM_SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:\
                        ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:\
                        ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:\
                        DHE-RSA-CHACHA20-POLY1305:ECDHE-ECDSA-AES128-GCM-SHA256:\
                        ECDHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES128-GCM-SHA256:\
                        ECDHE-ECDSA-AES256-SHA384:ECDHE-RSA-AES256-SHA384:\
                        DHE-RSA-AES256-SHA256:ECDHE-ECDSA-AES128-SHA256:\
                        ECDHE-RSA-AES128-SHA256:DHE-RSA-AES128-SHA256:\
                        ECDHE-ECDSA-AES256-SHA:ECDHE-RSA-AES256-SHA",
            "ecdh_curve": "prime256v1",
        }
        ws_ssl1 = websocket.WebSocket(sslopt=ssloptions)
        ws_ssl1.connect("wss://api.bitfinex.com/ws/2")
        ws_ssl1.send("Hello")
        ws_ssl1.close()

        ws_ssl2 = websocket.WebSocket(sslopt={"check_hostname": True})
        ws_ssl2.connect("wss://api.bitfinex.com/ws/2")
        ws_ssl2.close

    def test_proxy_info(self):
        self.assertEqual(
            proxy_info(
                http_proxy_host="127.0.0.1", http_proxy_port="8080", proxy_type="http"
            ).proxy_protocol,
            "http",
        )
        self.assertRaises(
            ProxyError,
            proxy_info,
            http_proxy_host="127.0.0.1",
            http_proxy_port="8080",
            proxy_type="badval",
        )
        self.assertEqual(
            proxy_info(
                http_proxy_host="example.com", http_proxy_port="8080", proxy_type="http"
            ).proxy_host,
            "example.com",
        )
        self.assertEqual(
            proxy_info(
                http_proxy_host="127.0.0.1", http_proxy_port="8080", proxy_type="http"
            ).proxy_port,
            "8080",
        )
        self.assertEqual(
            proxy_info(
                http_proxy_host="127.0.0.1", http_proxy_port="8080", proxy_type="http"
            ).auth,
            None,
        )
        self.assertEqual(
            proxy_info(
                http_proxy_host="127.0.0.1",
                http_proxy_port="8080",
                proxy_type="http",
                http_proxy_auth=("my_username123", "my_pass321"),
            ).auth[0],
            "my_username123",
        )
        self.assertEqual(
            proxy_info(
                http_proxy_host="127.0.0.1",
                http_proxy_port="8080",
                proxy_type="http",
                http_proxy_auth=("my_username123", "my_pass321"),
            ).auth[1],
            "my_pass321",
        )


if __name__ == "__main__":
    unittest.main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_longdouble.py ===
import platform
import warnings

import pytest

import numpy as np
from numpy._core.tests._locales import CommaDecimalPointLocale
from numpy.testing import (
    IS_MUSL,
    assert_,
    assert_array_equal,
    assert_equal,
    assert_raises,
    temppath,
)

LD_INFO = np.finfo(np.longdouble)
longdouble_longer_than_double = (LD_INFO.eps < np.finfo(np.double).eps)


_o = 1 + LD_INFO.eps
string_to_longdouble_inaccurate = (_o != np.longdouble(str(_o)))
del _o


def test_scalar_extraction():
    """Confirm that extracting a value doesn't convert to python float"""
    o = 1 + LD_INFO.eps
    a = np.array([o, o, o])
    assert_equal(a[1], o)


# Conversions string -> long double

# 0.1 not exactly representable in base 2 floating point.
repr_precision = len(repr(np.longdouble(0.1)))
# +2 from macro block starting around line 842 in scalartypes.c.src.


@pytest.mark.skipif(IS_MUSL,
                    reason="test flaky on musllinux")
@pytest.mark.skipif(LD_INFO.precision + 2 >= repr_precision,
                    reason="repr precision not enough to show eps")
def test_str_roundtrip():
    # We will only see eps in repr if within printing precision.
    o = 1 + LD_INFO.eps
    assert_equal(np.longdouble(str(o)), o, f"str was {str(o)}")


@pytest.mark.skipif(string_to_longdouble_inaccurate, reason="Need strtold_l")
def test_str_roundtrip_bytes():
    o = 1 + LD_INFO.eps
    assert_equal(np.longdouble(str(o).encode("ascii")), o)


@pytest.mark.skipif(string_to_longdouble_inaccurate, reason="Need strtold_l")
@pytest.mark.parametrize("strtype", (np.str_, np.bytes_, str, bytes))
def test_array_and_stringlike_roundtrip(strtype):
    """
    Test that string representations of long-double roundtrip both
    for array casting and scalar coercion, see also gh-15608.
    """
    o = 1 + LD_INFO.eps

    if strtype in (np.bytes_, bytes):
        o_str = strtype(str(o).encode("ascii"))
    else:
        o_str = strtype(str(o))

    # Test that `o` is correctly coerced from the string-like
    assert o == np.longdouble(o_str)

    # Test that arrays also roundtrip correctly:
    o_strarr = np.asarray([o] * 3, dtype=strtype)
    assert (o == o_strarr.astype(np.longdouble)).all()

    # And array coercion and casting to string give the same as scalar repr:
    assert (o_strarr == o_str).all()
    assert (np.asarray([o] * 3).astype(strtype) == o_str).all()


def test_bogus_string():
    assert_raises(ValueError, np.longdouble, "spam")
    assert_raises(ValueError, np.longdouble, "1.0 flub")


@pytest.mark.skipif(string_to_longdouble_inaccurate, reason="Need strtold_l")
def test_fromstring():
    o = 1 + LD_INFO.eps
    s = (" " + str(o)) * 5
    a = np.array([o] * 5)
    assert_equal(np.fromstring(s, sep=" ", dtype=np.longdouble), a,
                 err_msg=f"reading '{s}'")


def test_fromstring_complex():
    for ctype in ["complex", "cdouble"]:
        # Check spacing between separator
        assert_equal(np.fromstring("1, 2 ,  3  ,4", sep=",", dtype=ctype),
                     np.array([1., 2., 3., 4.]))
        # Real component not specified
        assert_equal(np.fromstring("1j, -2j,  3j, 4e1j", sep=",", dtype=ctype),
                     np.array([1.j, -2.j, 3.j, 40.j]))
        # Both components specified
        assert_equal(np.fromstring("1+1j,2-2j, -3+3j,  -4e1+4j", sep=",", dtype=ctype),
                     np.array([1. + 1.j, 2. - 2.j, - 3. + 3.j, - 40. + 4j]))
        # Spaces at wrong places
        with assert_raises(ValueError):
            np.fromstring("1+2 j,3", dtype=ctype, sep=",")
        with assert_raises(ValueError):
            np.fromstring("1+ 2j,3", dtype=ctype, sep=",")
        with assert_raises(ValueError):
            np.fromstring("1 +2j,3", dtype=ctype, sep=",")
        with assert_raises(ValueError):
            np.fromstring("1+j", dtype=ctype, sep=",")
        with assert_raises(ValueError):
            np.fromstring("1+", dtype=ctype, sep=",")
        with assert_raises(ValueError):
            np.fromstring("1j+1", dtype=ctype, sep=",")


def test_fromstring_bogus():
    with assert_raises(ValueError):
        np.fromstring("1. 2. 3. flop 4.", dtype=float, sep=" ")


def test_fromstring_empty():
    with assert_raises(ValueError):
        np.fromstring("xxxxx", sep="x")


def test_fromstring_missing():
    with assert_raises(ValueError):
        np.fromstring("1xx3x4x5x6", sep="x")


class TestFileBased:

    ldbl = 1 + LD_INFO.eps
    tgt = np.array([ldbl] * 5)
    out = ''.join([str(t) + '\n' for t in tgt])

    def test_fromfile_bogus(self):
        with temppath() as path:
            with open(path, 'w') as f:
                f.write("1. 2. 3. flop 4.\n")

            with assert_raises(ValueError):
                np.fromfile(path, dtype=float, sep=" ")

    def test_fromfile_complex(self):
        for ctype in ["complex", "cdouble"]:
            # Check spacing between separator and only real component specified
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1, 2 ,  3  ,4\n")

                res = np.fromfile(path, dtype=ctype, sep=",")
            assert_equal(res, np.array([1., 2., 3., 4.]))

            # Real component not specified
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1j, -2j,  3j, 4e1j\n")

                res = np.fromfile(path, dtype=ctype, sep=",")
            assert_equal(res, np.array([1.j, -2.j, 3.j, 40.j]))

            # Both components specified
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1+1j,2-2j, -3+3j,  -4e1+4j\n")

                res = np.fromfile(path, dtype=ctype, sep=",")
            assert_equal(res, np.array([1. + 1.j, 2. - 2.j, - 3. + 3.j, - 40. + 4j]))

            # Spaces at wrong places
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1+2 j,3\n")

                with assert_raises(ValueError):
                    np.fromfile(path, dtype=ctype, sep=",")

            # Spaces at wrong places
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1+ 2j,3\n")

                with assert_raises(ValueError):
                    np.fromfile(path, dtype=ctype, sep=",")

            # Spaces at wrong places
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1 +2j,3\n")

                with assert_raises(ValueError):
                    np.fromfile(path, dtype=ctype, sep=",")

            # Wrong sep
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1+j\n")

                with assert_raises(ValueError):
                    np.fromfile(path, dtype=ctype, sep=",")

            # Wrong sep
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1+\n")

                with assert_raises(ValueError):
                    np.fromfile(path, dtype=ctype, sep=",")

            # Wrong sep
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1j+1\n")

                with assert_raises(ValueError):
                    np.fromfile(path, dtype=ctype, sep=",")

    @pytest.mark.skipif(string_to_longdouble_inaccurate,
                        reason="Need strtold_l")
    def test_fromfile(self):
        with temppath() as path:
            with open(path, 'w') as f:
                f.write(self.out)
            res = np.fromfile(path, dtype=np.longdouble, sep="\n")
        assert_equal(res, self.tgt)

    @pytest.mark.skipif(string_to_longdouble_inaccurate,
                        reason="Need strtold_l")
    def test_genfromtxt(self):
        with temppath() as path:
            with open(path, 'w') as f:
                f.write(self.out)
            res = np.genfromtxt(path, dtype=np.longdouble)
        assert_equal(res, self.tgt)

    @pytest.mark.skipif(string_to_longdouble_inaccurate,
                        reason="Need strtold_l")
    def test_loadtxt(self):
        with temppath() as path:
            with open(path, 'w') as f:
                f.write(self.out)
            res = np.loadtxt(path, dtype=np.longdouble)
        assert_equal(res, self.tgt)

    @pytest.mark.skipif(string_to_longdouble_inaccurate,
                        reason="Need strtold_l")
    def test_tofile_roundtrip(self):
        with temppath() as path:
            self.tgt.tofile(path, sep=" ")
            res = np.fromfile(path, dtype=np.longdouble, sep=" ")
        assert_equal(res, self.tgt)


# Conversions long double -> string


def test_str_exact():
    o = 1 + LD_INFO.eps
    assert_(str(o) != '1')


@pytest.mark.skipif(longdouble_longer_than_double, reason="BUG #2376")
@pytest.mark.skipif(string_to_longdouble_inaccurate,
                    reason="Need strtold_l")
def test_format():
    assert_(f"{1 + LD_INFO.eps:.40g}" != '1')


@pytest.mark.skipif(longdouble_longer_than_double, reason="BUG #2376")
@pytest.mark.skipif(string_to_longdouble_inaccurate,
                    reason="Need strtold_l")
def test_percent():
    o = 1 + LD_INFO.eps
    assert_(f"{o:.40g}" != '1')


@pytest.mark.skipif(longdouble_longer_than_double,
                    reason="array repr problem")
@pytest.mark.skipif(string_to_longdouble_inaccurate,
                    reason="Need strtold_l")
def test_array_repr():
    o = 1 + LD_INFO.eps
    a = np.array([o])
    b = np.array([1], dtype=np.longdouble)
    if not np.all(a != b):
        raise ValueError("precision loss creating arrays")
    assert_(repr(a) != repr(b))

#
# Locale tests: scalar types formatting should be independent of the locale
#

class TestCommaDecimalPointLocale(CommaDecimalPointLocale):

    def test_str_roundtrip_foreign(self):
        o = 1.5
        assert_equal(o, np.longdouble(str(o)))

    def test_fromstring_foreign_repr(self):
        f = 1.234
        a = np.fromstring(repr(f), dtype=float, sep=" ")
        assert_equal(a[0], f)

    def test_fromstring_foreign(self):
        s = "1.234"
        a = np.fromstring(s, dtype=np.longdouble, sep=" ")
        assert_equal(a[0], np.longdouble(s))

    def test_fromstring_foreign_sep(self):
        a = np.array([1, 2, 3, 4])
        b = np.fromstring("1,2,3,4,", dtype=np.longdouble, sep=",")
        assert_array_equal(a, b)

    def test_fromstring_foreign_value(self):
        with assert_raises(ValueError):
            np.fromstring("1,234", dtype=np.longdouble, sep=" ")


@pytest.mark.parametrize("int_val", [
    # cases discussed in gh-10723
    # and gh-9968
    2 ** 1024, 0])
def test_longdouble_from_int(int_val):
    # for issue gh-9968
    str_val = str(int_val)
    # we'll expect a RuntimeWarning on platforms
    # with np.longdouble equivalent to np.double
    # for large integer input
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', RuntimeWarning)
        # can be inf==inf on some platforms
        assert np.longdouble(int_val) == np.longdouble(str_val)
        # we can't directly compare the int and
        # max longdouble value on all platforms
        if np.allclose(np.finfo(np.longdouble).max,
                       np.finfo(np.double).max) and w:
            assert w[0].category is RuntimeWarning

@pytest.mark.parametrize("bool_val", [
    True, False])
def test_longdouble_from_bool(bool_val):
    assert np.longdouble(bool_val) == np.longdouble(int(bool_val))


@pytest.mark.skipif(
    not (IS_MUSL and platform.machine() == "x86_64"),
    reason="only need to run on musllinux_x86_64"
)
def test_musllinux_x86_64_signature():
    # this test may fail if you're emulating musllinux_x86_64 on a different
    # architecture, but should pass natively.
    known_sigs = [b'\xcd\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xfb\xbf']
    sig = (np.longdouble(-1.0) / np.longdouble(10.0))
    sig = sig.view(sig.dtype.newbyteorder('<')).tobytes()[:10]
    assert sig in known_sigs


def test_eps_positive():
    # np.finfo('g').eps should be positive on all platforms. If this isn't true
    # then something may have gone wrong with the MachArLike, e.g. if
    # np._core.getlimits._discovered_machar didn't work properly
    assert np.finfo(np.longdouble).eps > 0.

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\interval\test_interval_range.py ===
from datetime import timedelta

import numpy as np
import pytest

from pandas.core.dtypes.common import is_integer

from pandas import (
    DateOffset,
    Interval,
    IntervalIndex,
    Timedelta,
    Timestamp,
    date_range,
    interval_range,
    timedelta_range,
)
import pandas._testing as tm

from pandas.tseries.offsets import Day


@pytest.fixture(params=[None, "foo"])
def name(request):
    return request.param


class TestIntervalRange:
    @pytest.mark.parametrize("freq, periods", [(1, 100), (2.5, 40), (5, 20), (25, 4)])
    def test_constructor_numeric(self, closed, name, freq, periods):
        start, end = 0, 100
        breaks = np.arange(101, step=freq)
        expected = IntervalIndex.from_breaks(breaks, name=name, closed=closed)

        # defined from start/end/freq
        result = interval_range(
            start=start, end=end, freq=freq, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

        # defined from start/periods/freq
        result = interval_range(
            start=start, periods=periods, freq=freq, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

        # defined from end/periods/freq
        result = interval_range(
            end=end, periods=periods, freq=freq, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

        # GH 20976: linspace behavior defined from start/end/periods
        result = interval_range(
            start=start, end=end, periods=periods, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    @pytest.mark.parametrize(
        "freq, periods", [("D", 364), ("2D", 182), ("22D18h", 16), ("ME", 11)]
    )
    def test_constructor_timestamp(self, closed, name, freq, periods, tz):
        start, end = Timestamp("20180101", tz=tz), Timestamp("20181231", tz=tz)
        breaks = date_range(start=start, end=end, freq=freq)
        expected = IntervalIndex.from_breaks(breaks, name=name, closed=closed)

        # defined from start/end/freq
        result = interval_range(
            start=start, end=end, freq=freq, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

        # defined from start/periods/freq
        result = interval_range(
            start=start, periods=periods, freq=freq, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

        # defined from end/periods/freq
        result = interval_range(
            end=end, periods=periods, freq=freq, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

        # GH 20976: linspace behavior defined from start/end/periods
        if not breaks.freq.n == 1 and tz is None:
            result = interval_range(
                start=start, end=end, periods=periods, name=name, closed=closed
            )
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "freq, periods", [("D", 100), ("2D12h", 40), ("5D", 20), ("25D", 4)]
    )
    def test_constructor_timedelta(self, closed, name, freq, periods):
        start, end = Timedelta("0 days"), Timedelta("100 days")
        breaks = timedelta_range(start=start, end=end, freq=freq)
        expected = IntervalIndex.from_breaks(breaks, name=name, closed=closed)

        # defined from start/end/freq
        result = interval_range(
            start=start, end=end, freq=freq, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

        # defined from start/periods/freq
        result = interval_range(
            start=start, periods=periods, freq=freq, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

        # defined from end/periods/freq
        result = interval_range(
            end=end, periods=periods, freq=freq, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

        # GH 20976: linspace behavior defined from start/end/periods
        result = interval_range(
            start=start, end=end, periods=periods, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "start, end, freq, expected_endpoint",
        [
            (0, 10, 3, 9),
            (0, 10, 1.5, 9),
            (0.5, 10, 3, 9.5),
            (Timedelta("0D"), Timedelta("10D"), "2D4h", Timedelta("8D16h")),
            (
                Timestamp("2018-01-01"),
                Timestamp("2018-02-09"),
                "MS",
                Timestamp("2018-02-01"),
            ),
            (
                Timestamp("2018-01-01", tz="US/Eastern"),
                Timestamp("2018-01-20", tz="US/Eastern"),
                "5D12h",
                Timestamp("2018-01-17 12:00:00", tz="US/Eastern"),
            ),
        ],
    )
    def test_early_truncation(self, start, end, freq, expected_endpoint):
        # index truncates early if freq causes end to be skipped
        result = interval_range(start=start, end=end, freq=freq)
        result_endpoint = result.right[-1]
        assert result_endpoint == expected_endpoint

    @pytest.mark.parametrize(
        "start, end, freq",
        [(0.5, None, None), (None, 4.5, None), (0.5, None, 1.5), (None, 6.5, 1.5)],
    )
    def test_no_invalid_float_truncation(self, start, end, freq):
        # GH 21161
        if freq is None:
            breaks = [0.5, 1.5, 2.5, 3.5, 4.5]
        else:
            breaks = [0.5, 2.0, 3.5, 5.0, 6.5]
        expected = IntervalIndex.from_breaks(breaks)

        result = interval_range(start=start, end=end, periods=4, freq=freq)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "start, mid, end",
        [
            (
                Timestamp("2018-03-10", tz="US/Eastern"),
                Timestamp("2018-03-10 23:30:00", tz="US/Eastern"),
                Timestamp("2018-03-12", tz="US/Eastern"),
            ),
            (
                Timestamp("2018-11-03", tz="US/Eastern"),
                Timestamp("2018-11-04 00:30:00", tz="US/Eastern"),
                Timestamp("2018-11-05", tz="US/Eastern"),
            ),
        ],
    )
    def test_linspace_dst_transition(self, start, mid, end):
        # GH 20976: linspace behavior defined from start/end/periods
        # accounts for the hour gained/lost during DST transition
        start = start.as_unit("ns")
        mid = mid.as_unit("ns")
        end = end.as_unit("ns")
        result = interval_range(start=start, end=end, periods=2)
        expected = IntervalIndex.from_breaks([start, mid, end])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("freq", [2, 2.0])
    @pytest.mark.parametrize("end", [10, 10.0])
    @pytest.mark.parametrize("start", [0, 0.0])
    def test_float_subtype(self, start, end, freq):
        # Has float subtype if any of start/end/freq are float, even if all
        # resulting endpoints can safely be upcast to integers

        # defined from start/end/freq
        index = interval_range(start=start, end=end, freq=freq)
        result = index.dtype.subtype
        expected = "int64" if is_integer(start + end + freq) else "float64"
        assert result == expected

        # defined from start/periods/freq
        index = interval_range(start=start, periods=5, freq=freq)
        result = index.dtype.subtype
        expected = "int64" if is_integer(start + freq) else "float64"
        assert result == expected

        # defined from end/periods/freq
        index = interval_range(end=end, periods=5, freq=freq)
        result = index.dtype.subtype
        expected = "int64" if is_integer(end + freq) else "float64"
        assert result == expected

        # GH 20976: linspace behavior defined from start/end/periods
        index = interval_range(start=start, end=end, periods=5)
        result = index.dtype.subtype
        expected = "int64" if is_integer(start + end) else "float64"
        assert result == expected

    def test_interval_range_fractional_period(self):
        # float value for periods
        expected = interval_range(start=0, periods=10)
        msg = "Non-integer 'periods' in pd.date_range, .* pd.interval_range"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = interval_range(start=0, periods=10.5)
        tm.assert_index_equal(result, expected)

    def test_constructor_coverage(self):
        # equivalent timestamp-like start/end
        start, end = Timestamp("2017-01-01"), Timestamp("2017-01-15")
        expected = interval_range(start=start, end=end)

        result = interval_range(start=start.to_pydatetime(), end=end.to_pydatetime())
        tm.assert_index_equal(result, expected)

        result = interval_range(start=start.asm8, end=end.asm8)
        tm.assert_index_equal(result, expected)

        # equivalent freq with timestamp
        equiv_freq = [
            "D",
            Day(),
            Timedelta(days=1),
            timedelta(days=1),
            DateOffset(days=1),
        ]
        for freq in equiv_freq:
            result = interval_range(start=start, end=end, freq=freq)
            tm.assert_index_equal(result, expected)

        # equivalent timedelta-like start/end
        start, end = Timedelta(days=1), Timedelta(days=10)
        expected = interval_range(start=start, end=end)

        result = interval_range(start=start.to_pytimedelta(), end=end.to_pytimedelta())
        tm.assert_index_equal(result, expected)

        result = interval_range(start=start.asm8, end=end.asm8)
        tm.assert_index_equal(result, expected)

        # equivalent freq with timedelta
        equiv_freq = ["D", Day(), Timedelta(days=1), timedelta(days=1)]
        for freq in equiv_freq:
            result = interval_range(start=start, end=end, freq=freq)
            tm.assert_index_equal(result, expected)

    def test_errors(self):
        # not enough params
        msg = (
            "Of the four parameters: start, end, periods, and freq, "
            "exactly three must be specified"
        )

        with pytest.raises(ValueError, match=msg):
            interval_range(start=0)

        with pytest.raises(ValueError, match=msg):
            interval_range(end=5)

        with pytest.raises(ValueError, match=msg):
            interval_range(periods=2)

        with pytest.raises(ValueError, match=msg):
            interval_range()

        # too many params
        with pytest.raises(ValueError, match=msg):
            interval_range(start=0, end=5, periods=6, freq=1.5)

        # mixed units
        msg = "start, end, freq need to be type compatible"
        with pytest.raises(TypeError, match=msg):
            interval_range(start=0, end=Timestamp("20130101"), freq=2)

        with pytest.raises(TypeError, match=msg):
            interval_range(start=0, end=Timedelta("1 day"), freq=2)

        with pytest.raises(TypeError, match=msg):
            interval_range(start=0, end=10, freq="D")

        with pytest.raises(TypeError, match=msg):
            interval_range(start=Timestamp("20130101"), end=10, freq="D")

        with pytest.raises(TypeError, match=msg):
            interval_range(
                start=Timestamp("20130101"), end=Timedelta("1 day"), freq="D"
            )

        with pytest.raises(TypeError, match=msg):
            interval_range(
                start=Timestamp("20130101"), end=Timestamp("20130110"), freq=2
            )

        with pytest.raises(TypeError, match=msg):
            interval_range(start=Timedelta("1 day"), end=10, freq="D")

        with pytest.raises(TypeError, match=msg):
            interval_range(
                start=Timedelta("1 day"), end=Timestamp("20130110"), freq="D"
            )

        with pytest.raises(TypeError, match=msg):
            interval_range(start=Timedelta("1 day"), end=Timedelta("10 days"), freq=2)

        # invalid periods
        msg = "periods must be a number, got foo"
        with pytest.raises(TypeError, match=msg):
            interval_range(start=0, periods="foo")

        # invalid start
        msg = "start must be numeric or datetime-like, got foo"
        with pytest.raises(ValueError, match=msg):
            interval_range(start="foo", periods=10)

        # invalid end
        msg = r"end must be numeric or datetime-like, got \(0, 1\]"
        with pytest.raises(ValueError, match=msg):
            interval_range(end=Interval(0, 1), periods=10)

        # invalid freq for datetime-like
        msg = "freq must be numeric or convertible to DateOffset, got foo"
        with pytest.raises(ValueError, match=msg):
            interval_range(start=0, end=10, freq="foo")

        with pytest.raises(ValueError, match=msg):
            interval_range(start=Timestamp("20130101"), periods=10, freq="foo")

        with pytest.raises(ValueError, match=msg):
            interval_range(end=Timedelta("1 day"), periods=10, freq="foo")

        # mixed tz
        start = Timestamp("2017-01-01", tz="US/Eastern")
        end = Timestamp("2017-01-07", tz="US/Pacific")
        msg = "Start and end cannot both be tz-aware with different timezones"
        with pytest.raises(TypeError, match=msg):
            interval_range(start=start, end=end)

    def test_float_freq(self):
        # GH 54477
        result = interval_range(0, 1, freq=0.1)
        expected = IntervalIndex.from_breaks([0 + 0.1 * n for n in range(11)])
        tm.assert_index_equal(result, expected)

        result = interval_range(0, 1, freq=0.6)
        expected = IntervalIndex.from_breaks([0, 0.6])
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\test_union_categoricals.py ===
import numpy as np
import pytest

from pandas.core.dtypes.concat import union_categoricals

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    Series,
)
import pandas._testing as tm


class TestUnionCategoricals:
    @pytest.mark.parametrize(
        "a, b, combined",
        [
            (list("abc"), list("abd"), list("abcabd")),
            ([0, 1, 2], [2, 3, 4], [0, 1, 2, 2, 3, 4]),
            ([0, 1.2, 2], [2, 3.4, 4], [0, 1.2, 2, 2, 3.4, 4]),
            (
                ["b", "b", np.nan, "a"],
                ["a", np.nan, "c"],
                ["b", "b", np.nan, "a", "a", np.nan, "c"],
            ),
            (
                pd.date_range("2014-01-01", "2014-01-05"),
                pd.date_range("2014-01-06", "2014-01-07"),
                pd.date_range("2014-01-01", "2014-01-07"),
            ),
            (
                pd.date_range("2014-01-01", "2014-01-05", tz="US/Central"),
                pd.date_range("2014-01-06", "2014-01-07", tz="US/Central"),
                pd.date_range("2014-01-01", "2014-01-07", tz="US/Central"),
            ),
            (
                pd.period_range("2014-01-01", "2014-01-05"),
                pd.period_range("2014-01-06", "2014-01-07"),
                pd.period_range("2014-01-01", "2014-01-07"),
            ),
        ],
    )
    @pytest.mark.parametrize("box", [Categorical, CategoricalIndex, Series])
    def test_union_categorical(self, a, b, combined, box):
        # GH 13361
        result = union_categoricals([box(Categorical(a)), box(Categorical(b))])
        expected = Categorical(combined)
        tm.assert_categorical_equal(result, expected)

    def test_union_categorical_ordered_appearance(self):
        # new categories ordered by appearance
        s = Categorical(["x", "y", "z"])
        s2 = Categorical(["a", "b", "c"])
        result = union_categoricals([s, s2])
        expected = Categorical(
            ["x", "y", "z", "a", "b", "c"], categories=["x", "y", "z", "a", "b", "c"]
        )
        tm.assert_categorical_equal(result, expected)

    def test_union_categorical_ordered_true(self):
        s = Categorical([0, 1.2, 2], ordered=True)
        s2 = Categorical([0, 1.2, 2], ordered=True)
        result = union_categoricals([s, s2])
        expected = Categorical([0, 1.2, 2, 0, 1.2, 2], ordered=True)
        tm.assert_categorical_equal(result, expected)

    def test_union_categorical_match_types(self):
        # must exactly match types
        s = Categorical([0, 1.2, 2])
        s2 = Categorical([2, 3, 4])
        msg = "dtype of categories must be the same"
        with pytest.raises(TypeError, match=msg):
            union_categoricals([s, s2])

    def test_union_categorical_empty(self):
        msg = "No Categoricals to union"
        with pytest.raises(ValueError, match=msg):
            union_categoricals([])

    def test_union_categoricals_nan(self):
        # GH 13759
        res = union_categoricals(
            [Categorical([1, 2, np.nan]), Categorical([3, 2, np.nan])]
        )
        exp = Categorical([1, 2, np.nan, 3, 2, np.nan])
        tm.assert_categorical_equal(res, exp)

        res = union_categoricals(
            [Categorical(["A", "B"]), Categorical(["B", "B", np.nan])]
        )
        exp = Categorical(["A", "B", "B", "B", np.nan])
        tm.assert_categorical_equal(res, exp)

        val1 = [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-03-01"), pd.NaT]
        val2 = [pd.NaT, pd.Timestamp("2011-01-01"), pd.Timestamp("2011-02-01")]

        res = union_categoricals([Categorical(val1), Categorical(val2)])
        exp = Categorical(
            val1 + val2,
            categories=[
                pd.Timestamp("2011-01-01"),
                pd.Timestamp("2011-03-01"),
                pd.Timestamp("2011-02-01"),
            ],
        )
        tm.assert_categorical_equal(res, exp)

        # all NaN
        res = union_categoricals(
            [
                Categorical(np.array([np.nan, np.nan], dtype=object)),
                Categorical(["X"], categories=pd.Index(["X"], dtype=object)),
            ]
        )
        exp = Categorical([np.nan, np.nan, "X"])
        tm.assert_categorical_equal(res, exp)

        res = union_categoricals(
            [Categorical([np.nan, np.nan]), Categorical([np.nan, np.nan])]
        )
        exp = Categorical([np.nan, np.nan, np.nan, np.nan])
        tm.assert_categorical_equal(res, exp)

    @pytest.mark.parametrize("val", [[], ["1"]])
    def test_union_categoricals_empty(self, val, request, using_infer_string):
        # GH 13759
        if using_infer_string and val == ["1"]:
            request.applymarker(
                pytest.mark.xfail(
                    reason="TDOD(infer_string) object and strings dont match"
                )
            )
        res = union_categoricals([Categorical([]), Categorical(val)])
        exp = Categorical(val)
        tm.assert_categorical_equal(res, exp)

    def test_union_categorical_same_category(self):
        # check fastpath
        c1 = Categorical([1, 2, 3, 4], categories=[1, 2, 3, 4])
        c2 = Categorical([3, 2, 1, np.nan], categories=[1, 2, 3, 4])
        res = union_categoricals([c1, c2])
        exp = Categorical([1, 2, 3, 4, 3, 2, 1, np.nan], categories=[1, 2, 3, 4])
        tm.assert_categorical_equal(res, exp)

    def test_union_categorical_same_category_str(self):
        c1 = Categorical(["z", "z", "z"], categories=["x", "y", "z"])
        c2 = Categorical(["x", "x", "x"], categories=["x", "y", "z"])
        res = union_categoricals([c1, c2])
        exp = Categorical(["z", "z", "z", "x", "x", "x"], categories=["x", "y", "z"])
        tm.assert_categorical_equal(res, exp)

    def test_union_categorical_same_categories_different_order(self):
        # https://github.com/pandas-dev/pandas/issues/19096
        c1 = Categorical(["a", "b", "c"], categories=["a", "b", "c"])
        c2 = Categorical(["a", "b", "c"], categories=["b", "a", "c"])
        result = union_categoricals([c1, c2])
        expected = Categorical(
            ["a", "b", "c", "a", "b", "c"], categories=["a", "b", "c"]
        )
        tm.assert_categorical_equal(result, expected)

    def test_union_categoricals_ordered(self):
        c1 = Categorical([1, 2, 3], ordered=True)
        c2 = Categorical([1, 2, 3], ordered=False)

        msg = "Categorical.ordered must be the same"
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2])

        res = union_categoricals([c1, c1])
        exp = Categorical([1, 2, 3, 1, 2, 3], ordered=True)
        tm.assert_categorical_equal(res, exp)

        c1 = Categorical([1, 2, 3, np.nan], ordered=True)
        c2 = Categorical([3, 2], categories=[1, 2, 3], ordered=True)

        res = union_categoricals([c1, c2])
        exp = Categorical([1, 2, 3, np.nan, 3, 2], ordered=True)
        tm.assert_categorical_equal(res, exp)

        c1 = Categorical([1, 2, 3], ordered=True)
        c2 = Categorical([1, 2, 3], categories=[3, 2, 1], ordered=True)

        msg = "to union ordered Categoricals, all categories must be the same"
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2])

    def test_union_categoricals_ignore_order(self):
        # GH 15219
        c1 = Categorical([1, 2, 3], ordered=True)
        c2 = Categorical([1, 2, 3], ordered=False)

        res = union_categoricals([c1, c2], ignore_order=True)
        exp = Categorical([1, 2, 3, 1, 2, 3])
        tm.assert_categorical_equal(res, exp)

        msg = "Categorical.ordered must be the same"
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2], ignore_order=False)

        res = union_categoricals([c1, c1], ignore_order=True)
        exp = Categorical([1, 2, 3, 1, 2, 3])
        tm.assert_categorical_equal(res, exp)

        res = union_categoricals([c1, c1], ignore_order=False)
        exp = Categorical([1, 2, 3, 1, 2, 3], categories=[1, 2, 3], ordered=True)
        tm.assert_categorical_equal(res, exp)

        c1 = Categorical([1, 2, 3, np.nan], ordered=True)
        c2 = Categorical([3, 2], categories=[1, 2, 3], ordered=True)

        res = union_categoricals([c1, c2], ignore_order=True)
        exp = Categorical([1, 2, 3, np.nan, 3, 2])
        tm.assert_categorical_equal(res, exp)

        c1 = Categorical([1, 2, 3], ordered=True)
        c2 = Categorical([1, 2, 3], categories=[3, 2, 1], ordered=True)

        res = union_categoricals([c1, c2], ignore_order=True)
        exp = Categorical([1, 2, 3, 1, 2, 3])
        tm.assert_categorical_equal(res, exp)

        res = union_categoricals([c2, c1], ignore_order=True, sort_categories=True)
        exp = Categorical([1, 2, 3, 1, 2, 3], categories=[1, 2, 3])
        tm.assert_categorical_equal(res, exp)

        c1 = Categorical([1, 2, 3], ordered=True)
        c2 = Categorical([4, 5, 6], ordered=True)
        result = union_categoricals([c1, c2], ignore_order=True)
        expected = Categorical([1, 2, 3, 4, 5, 6])
        tm.assert_categorical_equal(result, expected)

        msg = "to union ordered Categoricals, all categories must be the same"
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2], ignore_order=False)

        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2])

    def test_union_categoricals_sort(self):
        # GH 13846
        c1 = Categorical(["x", "y", "z"])
        c2 = Categorical(["a", "b", "c"])
        result = union_categoricals([c1, c2], sort_categories=True)
        expected = Categorical(
            ["x", "y", "z", "a", "b", "c"], categories=["a", "b", "c", "x", "y", "z"]
        )
        tm.assert_categorical_equal(result, expected)

        # fastpath
        c1 = Categorical(["a", "b"], categories=["b", "a", "c"])
        c2 = Categorical(["b", "c"], categories=["b", "a", "c"])
        result = union_categoricals([c1, c2], sort_categories=True)
        expected = Categorical(["a", "b", "b", "c"], categories=["a", "b", "c"])
        tm.assert_categorical_equal(result, expected)

        c1 = Categorical(["a", "b"], categories=["c", "a", "b"])
        c2 = Categorical(["b", "c"], categories=["c", "a", "b"])
        result = union_categoricals([c1, c2], sort_categories=True)
        expected = Categorical(["a", "b", "b", "c"], categories=["a", "b", "c"])
        tm.assert_categorical_equal(result, expected)

        # fastpath - skip resort
        c1 = Categorical(["a", "b"], categories=["a", "b", "c"])
        c2 = Categorical(["b", "c"], categories=["a", "b", "c"])
        result = union_categoricals([c1, c2], sort_categories=True)
        expected = Categorical(["a", "b", "b", "c"], categories=["a", "b", "c"])
        tm.assert_categorical_equal(result, expected)

        c1 = Categorical(["x", np.nan])
        c2 = Categorical([np.nan, "b"])
        result = union_categoricals([c1, c2], sort_categories=True)
        expected = Categorical(["x", np.nan, np.nan, "b"], categories=["b", "x"])
        tm.assert_categorical_equal(result, expected)

        c1 = Categorical([np.nan])
        c2 = Categorical([np.nan])
        result = union_categoricals([c1, c2], sort_categories=True)
        expected = Categorical([np.nan, np.nan])
        tm.assert_categorical_equal(result, expected)

        c1 = Categorical([])
        c2 = Categorical([])
        result = union_categoricals([c1, c2], sort_categories=True)
        expected = Categorical([])
        tm.assert_categorical_equal(result, expected)

        c1 = Categorical(["b", "a"], categories=["b", "a", "c"], ordered=True)
        c2 = Categorical(["a", "c"], categories=["b", "a", "c"], ordered=True)
        msg = "Cannot use sort_categories=True with ordered Categoricals"
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2], sort_categories=True)

    def test_union_categoricals_sort_false(self):
        # GH 13846
        c1 = Categorical(["x", "y", "z"])
        c2 = Categorical(["a", "b", "c"])
        result = union_categoricals([c1, c2], sort_categories=False)
        expected = Categorical(
            ["x", "y", "z", "a", "b", "c"], categories=["x", "y", "z", "a", "b", "c"]
        )
        tm.assert_categorical_equal(result, expected)

    def test_union_categoricals_sort_false_fastpath(self):
        # fastpath
        c1 = Categorical(["a", "b"], categories=["b", "a", "c"])
        c2 = Categorical(["b", "c"], categories=["b", "a", "c"])
        result = union_categoricals([c1, c2], sort_categories=False)
        expected = Categorical(["a", "b", "b", "c"], categories=["b", "a", "c"])
        tm.assert_categorical_equal(result, expected)

    def test_union_categoricals_sort_false_skipresort(self):
        # fastpath - skip resort
        c1 = Categorical(["a", "b"], categories=["a", "b", "c"])
        c2 = Categorical(["b", "c"], categories=["a", "b", "c"])
        result = union_categoricals([c1, c2], sort_categories=False)
        expected = Categorical(["a", "b", "b", "c"], categories=["a", "b", "c"])
        tm.assert_categorical_equal(result, expected)

    def test_union_categoricals_sort_false_one_nan(self):
        c1 = Categorical(["x", np.nan])
        c2 = Categorical([np.nan, "b"])
        result = union_categoricals([c1, c2], sort_categories=False)
        expected = Categorical(["x", np.nan, np.nan, "b"], categories=["x", "b"])
        tm.assert_categorical_equal(result, expected)

    def test_union_categoricals_sort_false_only_nan(self):
        c1 = Categorical([np.nan])
        c2 = Categorical([np.nan])
        result = union_categoricals([c1, c2], sort_categories=False)
        expected = Categorical([np.nan, np.nan])
        tm.assert_categorical_equal(result, expected)

    def test_union_categoricals_sort_false_empty(self):
        c1 = Categorical([])
        c2 = Categorical([])
        result = union_categoricals([c1, c2], sort_categories=False)
        expected = Categorical([])
        tm.assert_categorical_equal(result, expected)

    def test_union_categoricals_sort_false_ordered_true(self):
        c1 = Categorical(["b", "a"], categories=["b", "a", "c"], ordered=True)
        c2 = Categorical(["a", "c"], categories=["b", "a", "c"], ordered=True)
        result = union_categoricals([c1, c2], sort_categories=False)
        expected = Categorical(
            ["b", "a", "a", "c"], categories=["b", "a", "c"], ordered=True
        )
        tm.assert_categorical_equal(result, expected)

    def test_union_categorical_unwrap(self):
        # GH 14173
        c1 = Categorical(["a", "b"])
        c2 = Series(["b", "c"], dtype="category")
        result = union_categoricals([c1, c2])
        expected = Categorical(["a", "b", "b", "c"])
        tm.assert_categorical_equal(result, expected)

        c2 = CategoricalIndex(c2)
        result = union_categoricals([c1, c2])
        tm.assert_categorical_equal(result, expected)

        c1 = Series(c1)
        result = union_categoricals([c1, c2])
        tm.assert_categorical_equal(result, expected)

        msg = "all components to combine must be Categorical"
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, ["a", "b", "c"]])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_unicode.py ===

import numpy as np
from numpy.testing import assert_, assert_array_equal, assert_equal


def buffer_length(arr):
    if isinstance(arr, str):
        if not arr:
            charmax = 0
        else:
            charmax = max(ord(c) for c in arr)
        if charmax < 256:
            size = 1
        elif charmax < 65536:
            size = 2
        else:
            size = 4
        return size * len(arr)
    v = memoryview(arr)
    if v.shape is None:
        return len(v) * v.itemsize
    else:
        return np.prod(v.shape) * v.itemsize


# In both cases below we need to make sure that the byte swapped value (as
# UCS4) is still a valid unicode:
# Value that can be represented in UCS2 interpreters
ucs2_value = '\u0900'
# Value that cannot be represented in UCS2 interpreters (but can in UCS4)
ucs4_value = '\U00100900'


def test_string_cast():
    str_arr = np.array(["1234", "1234\0\0"], dtype='S')
    uni_arr1 = str_arr.astype('>U')
    uni_arr2 = str_arr.astype('<U')

    assert_array_equal(str_arr != uni_arr1, np.ones(2, dtype=bool))
    assert_array_equal(uni_arr1 != str_arr, np.ones(2, dtype=bool))
    assert_array_equal(str_arr == uni_arr1, np.zeros(2, dtype=bool))
    assert_array_equal(uni_arr1 == str_arr, np.zeros(2, dtype=bool))

    assert_array_equal(uni_arr1, uni_arr2)


############################################################
#    Creation tests
############################################################

class CreateZeros:
    """Check the creation of zero-valued arrays"""

    def content_check(self, ua, ua_scalar, nbytes):

        # Check the length of the unicode base type
        assert_(int(ua.dtype.str[2:]) == self.ulen)
        # Check the length of the data buffer
        assert_(buffer_length(ua) == nbytes)
        # Small check that data in array element is ok
        assert_(ua_scalar == '')
        # Encode to ascii and double check
        assert_(ua_scalar.encode('ascii') == b'')
        # Check buffer lengths for scalars
        assert_(buffer_length(ua_scalar) == 0)

    def test_zeros0D(self):
        # Check creation of 0-dimensional objects
        ua = np.zeros((), dtype=f'U{self.ulen}')
        self.content_check(ua, ua[()], 4 * self.ulen)

    def test_zerosSD(self):
        # Check creation of single-dimensional objects
        ua = np.zeros((2,), dtype=f'U{self.ulen}')
        self.content_check(ua, ua[0], 4 * self.ulen * 2)
        self.content_check(ua, ua[1], 4 * self.ulen * 2)

    def test_zerosMD(self):
        # Check creation of multi-dimensional objects
        ua = np.zeros((2, 3, 4), dtype=f'U{self.ulen}')
        self.content_check(ua, ua[0, 0, 0], 4 * self.ulen * 2 * 3 * 4)
        self.content_check(ua, ua[-1, -1, -1], 4 * self.ulen * 2 * 3 * 4)


class TestCreateZeros_1(CreateZeros):
    """Check the creation of zero-valued arrays (size 1)"""
    ulen = 1


class TestCreateZeros_2(CreateZeros):
    """Check the creation of zero-valued arrays (size 2)"""
    ulen = 2


class TestCreateZeros_1009(CreateZeros):
    """Check the creation of zero-valued arrays (size 1009)"""
    ulen = 1009


class CreateValues:
    """Check the creation of unicode arrays with values"""

    def content_check(self, ua, ua_scalar, nbytes):

        # Check the length of the unicode base type
        assert_(int(ua.dtype.str[2:]) == self.ulen)
        # Check the length of the data buffer
        assert_(buffer_length(ua) == nbytes)
        # Small check that data in array element is ok
        assert_(ua_scalar == self.ucs_value * self.ulen)
        # Encode to UTF-8 and double check
        assert_(ua_scalar.encode('utf-8') ==
                        (self.ucs_value * self.ulen).encode('utf-8'))
        # Check buffer lengths for scalars
        if self.ucs_value == ucs4_value:
            # In UCS2, the \U0010FFFF will be represented using a
            # surrogate *pair*
            assert_(buffer_length(ua_scalar) == 2 * 2 * self.ulen)
        else:
            # In UCS2, the \uFFFF will be represented using a
            # regular 2-byte word
            assert_(buffer_length(ua_scalar) == 2 * self.ulen)

    def test_values0D(self):
        # Check creation of 0-dimensional objects with values
        ua = np.array(self.ucs_value * self.ulen, dtype=f'U{self.ulen}')
        self.content_check(ua, ua[()], 4 * self.ulen)

    def test_valuesSD(self):
        # Check creation of single-dimensional objects with values
        ua = np.array([self.ucs_value * self.ulen] * 2, dtype=f'U{self.ulen}')
        self.content_check(ua, ua[0], 4 * self.ulen * 2)
        self.content_check(ua, ua[1], 4 * self.ulen * 2)

    def test_valuesMD(self):
        # Check creation of multi-dimensional objects with values
        ua = np.array([[[self.ucs_value * self.ulen] * 2] * 3] * 4, dtype=f'U{self.ulen}')
        self.content_check(ua, ua[0, 0, 0], 4 * self.ulen * 2 * 3 * 4)
        self.content_check(ua, ua[-1, -1, -1], 4 * self.ulen * 2 * 3 * 4)


class TestCreateValues_1_UCS2(CreateValues):
    """Check the creation of valued arrays (size 1, UCS2 values)"""
    ulen = 1
    ucs_value = ucs2_value


class TestCreateValues_1_UCS4(CreateValues):
    """Check the creation of valued arrays (size 1, UCS4 values)"""
    ulen = 1
    ucs_value = ucs4_value


class TestCreateValues_2_UCS2(CreateValues):
    """Check the creation of valued arrays (size 2, UCS2 values)"""
    ulen = 2
    ucs_value = ucs2_value


class TestCreateValues_2_UCS4(CreateValues):
    """Check the creation of valued arrays (size 2, UCS4 values)"""
    ulen = 2
    ucs_value = ucs4_value


class TestCreateValues_1009_UCS2(CreateValues):
    """Check the creation of valued arrays (size 1009, UCS2 values)"""
    ulen = 1009
    ucs_value = ucs2_value


class TestCreateValues_1009_UCS4(CreateValues):
    """Check the creation of valued arrays (size 1009, UCS4 values)"""
    ulen = 1009
    ucs_value = ucs4_value


############################################################
#    Assignment tests
############################################################

class AssignValues:
    """Check the assignment of unicode arrays with values"""

    def content_check(self, ua, ua_scalar, nbytes):

        # Check the length of the unicode base type
        assert_(int(ua.dtype.str[2:]) == self.ulen)
        # Check the length of the data buffer
        assert_(buffer_length(ua) == nbytes)
        # Small check that data in array element is ok
        assert_(ua_scalar == self.ucs_value * self.ulen)
        # Encode to UTF-8 and double check
        assert_(ua_scalar.encode('utf-8') ==
                        (self.ucs_value * self.ulen).encode('utf-8'))
        # Check buffer lengths for scalars
        if self.ucs_value == ucs4_value:
            # In UCS2, the \U0010FFFF will be represented using a
            # surrogate *pair*
            assert_(buffer_length(ua_scalar) == 2 * 2 * self.ulen)
        else:
            # In UCS2, the \uFFFF will be represented using a
            # regular 2-byte word
            assert_(buffer_length(ua_scalar) == 2 * self.ulen)

    def test_values0D(self):
        # Check assignment of 0-dimensional objects with values
        ua = np.zeros((), dtype=f'U{self.ulen}')
        ua[()] = self.ucs_value * self.ulen
        self.content_check(ua, ua[()], 4 * self.ulen)

    def test_valuesSD(self):
        # Check assignment of single-dimensional objects with values
        ua = np.zeros((2,), dtype=f'U{self.ulen}')
        ua[0] = self.ucs_value * self.ulen
        self.content_check(ua, ua[0], 4 * self.ulen * 2)
        ua[1] = self.ucs_value * self.ulen
        self.content_check(ua, ua[1], 4 * self.ulen * 2)

    def test_valuesMD(self):
        # Check assignment of multi-dimensional objects with values
        ua = np.zeros((2, 3, 4), dtype=f'U{self.ulen}')
        ua[0, 0, 0] = self.ucs_value * self.ulen
        self.content_check(ua, ua[0, 0, 0], 4 * self.ulen * 2 * 3 * 4)
        ua[-1, -1, -1] = self.ucs_value * self.ulen
        self.content_check(ua, ua[-1, -1, -1], 4 * self.ulen * 2 * 3 * 4)


class TestAssignValues_1_UCS2(AssignValues):
    """Check the assignment of valued arrays (size 1, UCS2 values)"""
    ulen = 1
    ucs_value = ucs2_value


class TestAssignValues_1_UCS4(AssignValues):
    """Check the assignment of valued arrays (size 1, UCS4 values)"""
    ulen = 1
    ucs_value = ucs4_value


class TestAssignValues_2_UCS2(AssignValues):
    """Check the assignment of valued arrays (size 2, UCS2 values)"""
    ulen = 2
    ucs_value = ucs2_value


class TestAssignValues_2_UCS4(AssignValues):
    """Check the assignment of valued arrays (size 2, UCS4 values)"""
    ulen = 2
    ucs_value = ucs4_value


class TestAssignValues_1009_UCS2(AssignValues):
    """Check the assignment of valued arrays (size 1009, UCS2 values)"""
    ulen = 1009
    ucs_value = ucs2_value


class TestAssignValues_1009_UCS4(AssignValues):
    """Check the assignment of valued arrays (size 1009, UCS4 values)"""
    ulen = 1009
    ucs_value = ucs4_value


############################################################
#    Byteorder tests
############################################################

class ByteorderValues:
    """Check the byteorder of unicode arrays in round-trip conversions"""

    def test_values0D(self):
        # Check byteorder of 0-dimensional objects
        ua = np.array(self.ucs_value * self.ulen, dtype=f'U{self.ulen}')
        ua2 = ua.view(ua.dtype.newbyteorder())
        # This changes the interpretation of the data region (but not the
        #  actual data), therefore the returned scalars are not
        #  the same (they are byte-swapped versions of each other).
        assert_(ua[()] != ua2[()])
        ua3 = ua2.view(ua2.dtype.newbyteorder())
        # Arrays must be equal after the round-trip
        assert_equal(ua, ua3)

    def test_valuesSD(self):
        # Check byteorder of single-dimensional objects
        ua = np.array([self.ucs_value * self.ulen] * 2, dtype=f'U{self.ulen}')
        ua2 = ua.view(ua.dtype.newbyteorder())
        assert_((ua != ua2).all())
        assert_(ua[-1] != ua2[-1])
        ua3 = ua2.view(ua2.dtype.newbyteorder())
        # Arrays must be equal after the round-trip
        assert_equal(ua, ua3)

    def test_valuesMD(self):
        # Check byteorder of multi-dimensional objects
        ua = np.array([[[self.ucs_value * self.ulen] * 2] * 3] * 4,
                      dtype=f'U{self.ulen}')
        ua2 = ua.view(ua.dtype.newbyteorder())
        assert_((ua != ua2).all())
        assert_(ua[-1, -1, -1] != ua2[-1, -1, -1])
        ua3 = ua2.view(ua2.dtype.newbyteorder())
        # Arrays must be equal after the round-trip
        assert_equal(ua, ua3)

    def test_values_cast(self):
        # Check byteorder of when casting the array for a strided and
        # contiguous array:
        test1 = np.array([self.ucs_value * self.ulen] * 2, dtype=f'U{self.ulen}')
        test2 = np.repeat(test1, 2)[::2]
        for ua in (test1, test2):
            ua2 = ua.astype(dtype=ua.dtype.newbyteorder())
            assert_((ua == ua2).all())
            assert_(ua[-1] == ua2[-1])
            ua3 = ua2.astype(dtype=ua.dtype)
            # Arrays must be equal after the round-trip
            assert_equal(ua, ua3)

    def test_values_updowncast(self):
        # Check byteorder of when casting the array to a longer and shorter
        # string length for strided and contiguous arrays
        test1 = np.array([self.ucs_value * self.ulen] * 2, dtype=f'U{self.ulen}')
        test2 = np.repeat(test1, 2)[::2]
        for ua in (test1, test2):
            # Cast to a longer type with zero padding
            longer_type = np.dtype(f'U{self.ulen + 1}').newbyteorder()
            ua2 = ua.astype(dtype=longer_type)
            assert_((ua == ua2).all())
            assert_(ua[-1] == ua2[-1])
            # Cast back again with truncating:
            ua3 = ua2.astype(dtype=ua.dtype)
            # Arrays must be equal after the round-trip
            assert_equal(ua, ua3)


class TestByteorder_1_UCS2(ByteorderValues):
    """Check the byteorder in unicode (size 1, UCS2 values)"""
    ulen = 1
    ucs_value = ucs2_value


class TestByteorder_1_UCS4(ByteorderValues):
    """Check the byteorder in unicode (size 1, UCS4 values)"""
    ulen = 1
    ucs_value = ucs4_value


class TestByteorder_2_UCS2(ByteorderValues):
    """Check the byteorder in unicode (size 2, UCS2 values)"""
    ulen = 2
    ucs_value = ucs2_value


class TestByteorder_2_UCS4(ByteorderValues):
    """Check the byteorder in unicode (size 2, UCS4 values)"""
    ulen = 2
    ucs_value = ucs4_value


class TestByteorder_1009_UCS2(ByteorderValues):
    """Check the byteorder in unicode (size 1009, UCS2 values)"""
    ulen = 1009
    ucs_value = ucs2_value


class TestByteorder_1009_UCS4(ByteorderValues):
    """Check the byteorder in unicode (size 1009, UCS4 values)"""
    ulen = 1009
    ucs_value = ucs4_value

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\tests\test_powsimp.py ===
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (E, I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import sin
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.simplify.powsimp import (powdenest, powsimp)
from sympy.simplify.simplify import (signsimp, simplify)
from sympy.core.symbol import Str

from sympy.abc import x, y, z, a, b


def test_powsimp():
    x, y, z, n = symbols('x,y,z,n')
    f = Function('f')
    assert powsimp( 4**x * 2**(-x) * 2**(-x) ) == 1
    assert powsimp( (-4)**x * (-2)**(-x) * 2**(-x) ) == 1

    assert powsimp(
        f(4**x * 2**(-x) * 2**(-x)) ) == f(4**x * 2**(-x) * 2**(-x))
    assert powsimp( f(4**x * 2**(-x) * 2**(-x)), deep=True ) == f(1)
    assert exp(x)*exp(y) == exp(x)*exp(y)
    assert powsimp(exp(x)*exp(y)) == exp(x + y)
    assert powsimp(exp(x)*exp(y)*2**x*2**y) == (2*E)**(x + y)
    assert powsimp(exp(x)*exp(y)*2**x*2**y, combine='exp') == \
        exp(x + y)*2**(x + y)
    assert powsimp(exp(x)*exp(y)*exp(2)*sin(x) + sin(y) + 2**x*2**y) == \
        exp(2 + x + y)*sin(x) + sin(y) + 2**(x + y)
    assert powsimp(sin(exp(x)*exp(y))) == sin(exp(x)*exp(y))
    assert powsimp(sin(exp(x)*exp(y)), deep=True) == sin(exp(x + y))
    assert powsimp(x**2*x**y) == x**(2 + y)
    # This should remain factored, because 'exp' with deep=True is supposed
    # to act like old automatic exponent combining.
    assert powsimp((1 + E*exp(E))*exp(-E), combine='exp', deep=True) == \
        (1 + exp(1 + E))*exp(-E)
    assert powsimp((1 + E*exp(E))*exp(-E), deep=True) == \
        (1 + exp(1 + E))*exp(-E)
    assert powsimp((1 + E*exp(E))*exp(-E)) == (1 + exp(1 + E))*exp(-E)
    assert powsimp((1 + E*exp(E))*exp(-E), combine='exp') == \
        (1 + exp(1 + E))*exp(-E)
    assert powsimp((1 + E*exp(E))*exp(-E), combine='base') == \
        (1 + E*exp(E))*exp(-E)
    x, y = symbols('x,y', nonnegative=True)
    n = Symbol('n', real=True)
    assert powsimp(y**n * (y/x)**(-n)) == x**n
    assert powsimp(x**(x**(x*y)*y**(x*y))*y**(x**(x*y)*y**(x*y)), deep=True) \
        == (x*y)**(x*y)**(x*y)
    assert powsimp(2**(2**(2*x)*x), deep=False) == 2**(2**(2*x)*x)
    assert powsimp(2**(2**(2*x)*x), deep=True) == 2**(x*4**x)
    assert powsimp(
        exp(-x + exp(-x)*exp(-x*log(x))), deep=False, combine='exp') == \
        exp(-x + exp(-x)*exp(-x*log(x)))
    assert powsimp(
        exp(-x + exp(-x)*exp(-x*log(x))), deep=False, combine='exp') == \
        exp(-x + exp(-x)*exp(-x*log(x)))
    assert powsimp((x + y)/(3*z), deep=False, combine='exp') == (x + y)/(3*z)
    assert powsimp((x/3 + y/3)/z, deep=True, combine='exp') == (x/3 + y/3)/z
    assert powsimp(exp(x)/(1 + exp(x)*exp(y)), deep=True) == \
        exp(x)/(1 + exp(x + y))
    assert powsimp(x*y**(z**x*z**y), deep=True) == x*y**(z**(x + y))
    assert powsimp((z**x*z**y)**x, deep=True) == (z**(x + y))**x
    assert powsimp(x*(z**x*z**y)**x, deep=True) == x*(z**(x + y))**x
    p = symbols('p', positive=True)
    assert powsimp((1/x)**log(2)/x) == (1/x)**(1 + log(2))
    assert powsimp((1/p)**log(2)/p) == p**(-1 - log(2))

    # coefficient of exponent can only be simplified for positive bases
    assert powsimp(2**(2*x)) == 4**x
    assert powsimp((-1)**(2*x)) == (-1)**(2*x)
    i = symbols('i', integer=True)
    assert powsimp((-1)**(2*i)) == 1
    assert powsimp((-1)**(-x)) != (-1)**x  # could be 1/((-1)**x), but is not
    # force=True overrides assumptions
    assert powsimp((-1)**(2*x), force=True) == 1

    # rational exponents allow combining of negative terms
    w, n, m = symbols('w n m', negative=True)
    e = i/a  # not a rational exponent if `a` is unknown
    ex = w**e*n**e*m**e
    assert powsimp(ex) == m**(i/a)*n**(i/a)*w**(i/a)
    e = i/3
    ex = w**e*n**e*m**e
    assert powsimp(ex) == (-1)**i*(-m*n*w)**(i/3)
    e = (3 + i)/i
    ex = w**e*n**e*m**e
    assert powsimp(ex) == (-1)**(3*e)*(-m*n*w)**e

    eq = x**(a*Rational(2, 3))
    # eq != (x**a)**(2/3) (try x = -1 and a = 3 to see)
    assert powsimp(eq).exp == eq.exp == a*Rational(2, 3)
    # powdenest goes the other direction
    assert powsimp(2**(2*x)) == 4**x

    assert powsimp(exp(p/2)) == exp(p/2)

    # issue 6368
    eq = Mul(*[sqrt(Dummy(imaginary=True)) for i in range(3)])
    assert powsimp(eq) == eq and eq.is_Mul

    assert all(powsimp(e) == e for e in (sqrt(x**a), sqrt(x**2)))

    # issue 8836
    assert str( powsimp(exp(I*pi/3)*root(-1,3)) ) == '(-1)**(2/3)'

    # issue 9183
    assert powsimp(-0.1**x) == -0.1**x

    # issue 10095
    assert powsimp((1/(2*E))**oo) == (exp(-1)/2)**oo

    # PR 13131
    eq = sin(2*x)**2*sin(2.0*x)**2
    assert powsimp(eq) == eq

    # issue 14615
    assert powsimp(x**2*y**3*(x*y**2)**Rational(3, 2)
        ) == x*y*(x*y**2)**Rational(5, 2)

    #issue 27380
    assert powsimp(1.0**(x+1)/1.0**x) == 1.0

def test_powsimp_negated_base():
    assert powsimp((-x + y)/sqrt(x - y)) == -sqrt(x - y)
    assert powsimp((-x + y)*(-z + y)/sqrt(x - y)/sqrt(z - y)) == sqrt(x - y)*sqrt(z - y)
    p = symbols('p', positive=True)
    reps = {p: 2, a: S.Half}
    assert powsimp((-p)**a/p**a).subs(reps) == ((-1)**a).subs(reps)
    assert powsimp((-p)**a*p**a).subs(reps) == ((-p**2)**a).subs(reps)
    n = symbols('n', negative=True)
    reps = {p: -2, a: S.Half}
    assert powsimp((-n)**a/n**a).subs(reps) == (-1)**(-a).subs(a, S.Half)
    assert powsimp((-n)**a*n**a).subs(reps) == ((-n**2)**a).subs(reps)
    # if x is 0 then the lhs is 0**a*oo**a which is not (-1)**a
    eq = (-x)**a/x**a
    assert powsimp(eq) == eq


def test_powsimp_nc():
    x, y, z = symbols('x,y,z')
    A, B, C = symbols('A B C', commutative=False)

    assert powsimp(A**x*A**y, combine='all') == A**(x + y)
    assert powsimp(A**x*A**y, combine='base') == A**x*A**y
    assert powsimp(A**x*A**y, combine='exp') == A**(x + y)

    assert powsimp(A**x*B**x, combine='all') == A**x*B**x
    assert powsimp(A**x*B**x, combine='base') == A**x*B**x
    assert powsimp(A**x*B**x, combine='exp') == A**x*B**x

    assert powsimp(B**x*A**x, combine='all') == B**x*A**x
    assert powsimp(B**x*A**x, combine='base') == B**x*A**x
    assert powsimp(B**x*A**x, combine='exp') == B**x*A**x

    assert powsimp(A**x*A**y*A**z, combine='all') == A**(x + y + z)
    assert powsimp(A**x*A**y*A**z, combine='base') == A**x*A**y*A**z
    assert powsimp(A**x*A**y*A**z, combine='exp') == A**(x + y + z)

    assert powsimp(A**x*B**x*C**x, combine='all') == A**x*B**x*C**x
    assert powsimp(A**x*B**x*C**x, combine='base') == A**x*B**x*C**x
    assert powsimp(A**x*B**x*C**x, combine='exp') == A**x*B**x*C**x

    assert powsimp(B**x*A**x*C**x, combine='all') == B**x*A**x*C**x
    assert powsimp(B**x*A**x*C**x, combine='base') == B**x*A**x*C**x
    assert powsimp(B**x*A**x*C**x, combine='exp') == B**x*A**x*C**x


def test_issue_6440():
    assert powsimp(16*2**a*8**b) == 2**(a + 3*b + 4)


def test_powdenest():
    x, y = symbols('x,y')
    p, q = symbols('p q', positive=True)
    i, j = symbols('i,j', integer=True)

    assert powdenest(x) == x
    assert powdenest(x + 2*(x**(a*Rational(2, 3)))**(3*x)) == (x + 2*(x**(a*Rational(2, 3)))**(3*x))
    assert powdenest((exp(a*Rational(2, 3)))**(3*x))  # -X-> (exp(a/3))**(6*x)
    assert powdenest((x**(a*Rational(2, 3)))**(3*x)) == ((x**(a*Rational(2, 3)))**(3*x))
    assert powdenest(exp(3*x*log(2))) == 2**(3*x)
    assert powdenest(sqrt(p**2)) == p
    eq = p**(2*i)*q**(4*i)
    assert powdenest(eq) == (p*q**2)**(2*i)
    # -X-> (x**x)**i*(x**x)**j == x**(x*(i + j))
    assert powdenest((x**x)**(i + j))
    assert powdenest(exp(3*y*log(x))) == x**(3*y)
    assert powdenest(exp(y*(log(a) + log(b)))) == (a*b)**y
    assert powdenest(exp(3*(log(a) + log(b)))) == a**3*b**3
    assert powdenest(((x**(2*i))**(3*y))**x) == ((x**(2*i))**(3*y))**x
    assert powdenest(((x**(2*i))**(3*y))**x, force=True) == x**(6*i*x*y)
    assert powdenest(((x**(a*Rational(2, 3)))**(3*y/i))**x) == \
        (((x**(a*Rational(2, 3)))**(3*y/i))**x)
    assert powdenest((x**(2*i)*y**(4*i))**z, force=True) == (x*y**2)**(2*i*z)
    assert powdenest((p**(2*i)*q**(4*i))**j) == (p*q**2)**(2*i*j)
    e = ((p**(2*a))**(3*y))**x
    assert powdenest(e) == e
    e = ((x**2*y**4)**a)**(x*y)
    assert powdenest(e) == e
    e = (((x**2*y**4)**a)**(x*y))**3
    assert powdenest(e) == ((x**2*y**4)**a)**(3*x*y)
    assert powdenest((((x**2*y**4)**a)**(x*y)), force=True) == \
        (x*y**2)**(2*a*x*y)
    assert powdenest((((x**2*y**4)**a)**(x*y))**3, force=True) == \
        (x*y**2)**(6*a*x*y)
    assert powdenest((x**2*y**6)**i) != (x*y**3)**(2*i)
    x, y = symbols('x,y', positive=True)
    assert powdenest((x**2*y**6)**i) == (x*y**3)**(2*i)

    assert powdenest((x**(i*Rational(2, 3))*y**(i/2))**(2*i)) == (x**Rational(4, 3)*y)**(i**2)
    assert powdenest(sqrt(x**(2*i)*y**(6*i))) == (x*y**3)**i

    assert powdenest(4**x) == 2**(2*x)
    assert powdenest((4**x)**y) == 2**(2*x*y)
    assert powdenest(4**x*y) == 2**(2*x)*y


def test_powdenest_polar():
    x, y, z = symbols('x y z', polar=True)
    a, b, c = symbols('a b c')
    assert powdenest((x*y*z)**a) == x**a*y**a*z**a
    assert powdenest((x**a*y**b)**c) == x**(a*c)*y**(b*c)
    assert powdenest(((x**a)**b*y**c)**c) == x**(a*b*c)*y**(c**2)


def test_issue_5805():
    arg = ((gamma(x)*hyper((), (), x))*pi)**2
    assert powdenest(arg) == (pi*gamma(x)*hyper((), (), x))**2
    assert arg.is_positive is None


def test_issue_9324_powsimp_on_matrix_symbol():
    M = MatrixSymbol('M', 10, 10)
    expr = powsimp(M, deep=True)
    assert expr == M
    assert expr.args[0] == Str('M')


def test_issue_6367():
    z = -5*sqrt(2)/(2*sqrt(2*sqrt(29) + 29)) + sqrt(-sqrt(29)/29 + S.Half)
    assert Mul(*[powsimp(a) for a in Mul.make_args(z.normal())]) == 0
    assert powsimp(z.normal()) == 0
    assert simplify(z) == 0
    assert powsimp(sqrt(2 + sqrt(3))*sqrt(2 - sqrt(3)) + 1) == 2
    assert powsimp(z) != 0


def test_powsimp_polar():
    from sympy.functions.elementary.complexes import polar_lift
    from sympy.functions.elementary.exponential import exp_polar
    x, y, z = symbols('x y z')
    p, q, r = symbols('p q r', polar=True)

    assert (polar_lift(-1))**(2*x) == exp_polar(2*pi*I*x)
    assert powsimp(p**x * q**x) == (p*q)**x
    assert p**x * (1/p)**x == 1
    assert (1/p)**x == p**(-x)

    assert exp_polar(x)*exp_polar(y) == exp_polar(x)*exp_polar(y)
    assert powsimp(exp_polar(x)*exp_polar(y)) == exp_polar(x + y)
    assert powsimp(exp_polar(x)*exp_polar(y)*p**x*p**y) == \
        (p*exp_polar(1))**(x + y)
    assert powsimp(exp_polar(x)*exp_polar(y)*p**x*p**y, combine='exp') == \
        exp_polar(x + y)*p**(x + y)
    assert powsimp(
        exp_polar(x)*exp_polar(y)*exp_polar(2)*sin(x) + sin(y) + p**x*p**y) \
        == p**(x + y) + sin(x)*exp_polar(2 + x + y) + sin(y)
    assert powsimp(sin(exp_polar(x)*exp_polar(y))) == \
        sin(exp_polar(x)*exp_polar(y))
    assert powsimp(sin(exp_polar(x)*exp_polar(y)), deep=True) == \
        sin(exp_polar(x + y))


def test_issue_5728():
    b = x*sqrt(y)
    a = sqrt(b)
    c = sqrt(sqrt(x)*y)
    assert powsimp(a*b) == sqrt(b)**3
    assert powsimp(a*b**2*sqrt(y)) == sqrt(y)*a**5
    assert powsimp(a*x**2*c**3*y) == c**3*a**5
    assert powsimp(a*x*c**3*y**2) == c**7*a
    assert powsimp(x*c**3*y**2) == c**7
    assert powsimp(x*c**3*y) == x*y*c**3
    assert powsimp(sqrt(x)*c**3*y) == c**5
    assert powsimp(sqrt(x)*a**3*sqrt(y)) == sqrt(x)*sqrt(y)*a**3
    assert powsimp(Mul(sqrt(x)*c**3*sqrt(y), y, evaluate=False)) == \
        sqrt(x)*sqrt(y)**3*c**3
    assert powsimp(a**2*a*x**2*y) == a**7

    # symbolic powers work, too
    b = x**y*y
    a = b*sqrt(b)
    assert a.is_Mul is True
    assert powsimp(a) == sqrt(b)**3

    # as does exp
    a = x*exp(y*Rational(2, 3))
    assert powsimp(a*sqrt(a)) == sqrt(a)**3
    assert powsimp(a**2*sqrt(a)) == sqrt(a)**5
    assert powsimp(a**2*sqrt(sqrt(a))) == sqrt(sqrt(a))**9


def test_issue_from_PR1599():
    n1, n2, n3, n4 = symbols('n1 n2 n3 n4', negative=True)
    assert (powsimp(sqrt(n1)*sqrt(n2)*sqrt(n3)) ==
        -I*sqrt(-n1)*sqrt(-n2)*sqrt(-n3))
    assert (powsimp(root(n1, 3)*root(n2, 3)*root(n3, 3)*root(n4, 3)) ==
        -(-1)**Rational(1, 3)*
        (-n1)**Rational(1, 3)*(-n2)**Rational(1, 3)*(-n3)**Rational(1, 3)*(-n4)**Rational(1, 3))


def test_issue_10195():
    a = Symbol('a', integer=True)
    l = Symbol('l', even=True, nonzero=True)
    n = Symbol('n', odd=True)
    e_x = (-1)**(n/2 - S.Half) - (-1)**(n*Rational(3, 2) - S.Half)
    assert powsimp((-1)**(l/2)) == I**l
    assert powsimp((-1)**(n/2)) == I**n
    assert powsimp((-1)**(n*Rational(3, 2))) == -I**n
    assert powsimp(e_x) == (-1)**(n/2 - S.Half) + (-1)**(n*Rational(3, 2) +
            S.Half)
    assert powsimp((-1)**(a*Rational(3, 2))) == (-I)**a

def test_issue_15709():
    assert powsimp(3**x*Rational(2, 3)) == 2*3**(x-1)
    assert powsimp(2*3**x/3) == 2*3**(x-1)


def test_issue_11981():
    x, y = symbols('x y', commutative=False)
    assert powsimp((x*y)**2 * (y*x)**2) == (x*y)**2 * (y*x)**2


def test_issue_17524():
    a = symbols("a", real=True)
    e = (-1 - a**2)*sqrt(1 + a**2)
    assert signsimp(powsimp(e)) == signsimp(e) == -(a**2 + 1)**(S(3)/2)


def test_issue_19627():
    # if you use force the user must verify
    assert powdenest(sqrt(sin(x)**2), force=True) == sin(x)
    assert powdenest((x**(S.Half/y))**(2*y), force=True) == x
    from sympy.core.function import expand_power_base
    e = 1 - a
    expr = (exp(z/e)*x**(b/e)*y**((1 - b)/e))**e
    assert powdenest(expand_power_base(expr, force=True), force=True
        ) == x**b*y**(1 - b)*exp(z)


def test_issue_22546():
    p1, p2 = symbols('p1, p2', positive=True)
    ref = powsimp(p1**z/p2**z)
    e = z + 1
    ans = ref.subs(z, e)
    assert ans.is_Pow
    assert powsimp(p1**e/p2**e) == ans
    i = symbols('i', integer=True)
    ref = powsimp(x**i/y**i)
    e = i + 1
    ans = ref.subs(i, e)
    assert ans.is_Pow
    assert powsimp(x**e/y**e) == ans

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_expand.py ===
from sympy.core.expr import unchanged
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational as R, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.order import O
from sympy.simplify.radsimp import expand_numer
from sympy.core.function import (expand, expand_multinomial,
    expand_power_base, expand_log)

from sympy.testing.pytest import raises
from sympy.core.random import verify_numerically

from sympy.abc import x, y, z


def test_expand_no_log():
    assert (
        (1 + log(x**4))**2).expand(log=False) == 1 + 2*log(x**4) + log(x**4)**2
    assert ((1 + log(x**4))*(1 + log(x**3))).expand(
        log=False) == 1 + log(x**4) + log(x**3) + log(x**4)*log(x**3)


def test_expand_no_multinomial():
    assert ((1 + x)*(1 + (1 + x)**4)).expand(multinomial=False) == \
        1 + x + (1 + x)**4 + x*(1 + x)**4


def test_expand_negative_integer_powers():
    expr = (x + y)**(-2)
    assert expr.expand() == 1 / (2*x*y + x**2 + y**2)
    assert expr.expand(multinomial=False) == (x + y)**(-2)
    expr = (x + y)**(-3)
    assert expr.expand() == 1 / (3*x*x*y + 3*x*y*y + x**3 + y**3)
    assert expr.expand(multinomial=False) == (x + y)**(-3)
    expr = (x + y)**(2) * (x + y)**(-4)
    assert expr.expand() == 1 / (2*x*y + x**2 + y**2)
    assert expr.expand(multinomial=False) == (x + y)**(-2)


def test_expand_non_commutative():
    A = Symbol('A', commutative=False)
    B = Symbol('B', commutative=False)
    C = Symbol('C', commutative=False)
    a = Symbol('a')
    b = Symbol('b')
    i = Symbol('i', integer=True)
    n = Symbol('n', negative=True)
    m = Symbol('m', negative=True)
    p = Symbol('p', polar=True)
    np = Symbol('p', polar=False)

    assert (C*(A + B)).expand() == C*A + C*B
    assert (C*(A + B)).expand() != A*C + B*C
    assert ((A + B)**2).expand() == A**2 + A*B + B*A + B**2
    assert ((A + B)**3).expand() == (A**2*B + B**2*A + A*B**2 + B*A**2 +
                                     A**3 + B**3 + A*B*A + B*A*B)
    # issue 6219
    assert ((a*A*B*A**-1)**2).expand() == a**2*A*B**2/A
    # Note that (a*A*B*A**-1)**2 is automatically converted to a**2*(A*B*A**-1)**2
    assert ((a*A*B*A**-1)**2).expand(deep=False) == a**2*(A*B*A**-1)**2
    assert ((a*A*B*A**-1)**2).expand() == a**2*(A*B**2*A**-1)
    assert ((a*A*B*A**-1)**2).expand(force=True) == a**2*A*B**2*A**(-1)
    assert ((a*A*B)**2).expand() == a**2*A*B*A*B
    assert ((a*A)**2).expand() == a**2*A**2
    assert ((a*A*B)**i).expand() == a**i*(A*B)**i
    assert ((a*A*(B*(A*B/A)**2))**i).expand() == a**i*(A*B*A*B**2/A)**i
    # issue 6558
    assert (A*B*(A*B)**-1).expand() == 1
    assert ((a*A)**i).expand() == a**i*A**i
    assert ((a*A*B*A**-1)**3).expand() == a**3*A*B**3/A
    assert ((a*A*B*A*B/A)**3).expand() == \
        a**3*A*B*(A*B**2)*(A*B**2)*A*B*A**(-1)
    assert ((a*A*B*A*B/A)**-2).expand() == \
        A*B**-1*A**-1*B**-2*A**-1*B**-1*A**-1/a**2
    assert ((a*b*A*B*A**-1)**i).expand() == a**i*b**i*(A*B/A)**i
    assert ((a*(a*b)**i)**i).expand() == a**i*a**(i**2)*b**(i**2)
    e = Pow(Mul(a, 1/a, A, B, evaluate=False), S(2), evaluate=False)
    assert e.expand() == A*B*A*B
    assert sqrt(a*(A*b)**i).expand() == sqrt(a*b**i*A**i)
    assert (sqrt(-a)**a).expand() == sqrt(-a)**a
    assert expand((-2*n)**(i/3)) == 2**(i/3)*(-n)**(i/3)
    assert expand((-2*n*m)**(i/a)) == (-2)**(i/a)*(-n)**(i/a)*(-m)**(i/a)
    assert expand((-2*a*p)**b) == 2**b*p**b*(-a)**b
    assert expand((-2*a*np)**b) == 2**b*(-a*np)**b
    assert expand(sqrt(A*B)) == sqrt(A*B)
    assert expand(sqrt(-2*a*b)) == sqrt(2)*sqrt(-a*b)


def test_expand_radicals():
    a = (x + y)**R(1, 2)

    assert (a**1).expand() == a
    assert (a**3).expand() == x*a + y*a
    assert (a**5).expand() == x**2*a + 2*x*y*a + y**2*a

    assert (1/a**1).expand() == 1/a
    assert (1/a**3).expand() == 1/(x*a + y*a)
    assert (1/a**5).expand() == 1/(x**2*a + 2*x*y*a + y**2*a)

    a = (x + y)**R(1, 3)

    assert (a**1).expand() == a
    assert (a**2).expand() == a**2
    assert (a**4).expand() == x*a + y*a
    assert (a**5).expand() == x*a**2 + y*a**2
    assert (a**7).expand() == x**2*a + 2*x*y*a + y**2*a


def test_expand_modulus():
    assert ((x + y)**11).expand(modulus=11) == x**11 + y**11
    assert ((x + sqrt(2)*y)**11).expand(modulus=11) == x**11 + 10*sqrt(2)*y**11
    assert (x + y/2).expand(modulus=1) == y/2

    raises(ValueError, lambda: ((x + y)**11).expand(modulus=0))
    raises(ValueError, lambda: ((x + y)**11).expand(modulus=x))


def test_issue_5743():
    assert (x*sqrt(
        x + y)*(1 + sqrt(x + y))).expand() == x**2 + x*y + x*sqrt(x + y)
    assert (x*sqrt(
        x + y)*(1 + x*sqrt(x + y))).expand() == x**3 + x**2*y + x*sqrt(x + y)


def test_expand_frac():
    assert expand((x + y)*y/x/(x + 1), frac=True) == \
        (x*y + y**2)/(x**2 + x)
    assert expand((x + y)*y/x/(x + 1), numer=True) == \
        (x*y + y**2)/(x*(x + 1))
    assert expand((x + y)*y/x/(x + 1), denom=True) == \
        y*(x + y)/(x**2 + x)
    eq = (x + 1)**2/y
    assert expand_numer(eq, multinomial=False) == eq
    # issue 26329
    eq = (exp(x*z) - exp(y*z))/exp(z*(x + y))
    ans = exp(-y*z) - exp(-x*z)
    assert eq.expand(numer=True) != ans
    assert eq.expand(numer=True, exact=True) == ans
    assert expand_numer(eq) != ans
    assert expand_numer(eq, exact=True) == ans


def test_issue_6121():
    eq = -I*exp(-3*I*pi/4)/(4*pi**(S(3)/2)*sqrt(x))
    assert eq.expand(complex=True)  # does not give oo recursion
    eq = -I*exp(-3*I*pi/4)/(4*pi**(R(3, 2))*sqrt(x))
    assert eq.expand(complex=True)  # does not give oo recursion


def test_expand_power_base():
    assert expand_power_base((x*y*z)**4) == x**4*y**4*z**4
    assert expand_power_base((x*y*z)**x).is_Pow
    assert expand_power_base((x*y*z)**x, force=True) == x**x*y**x*z**x
    assert expand_power_base((x*(y*z)**2)**3) == x**3*y**6*z**6

    assert expand_power_base((sin((x*y)**2)*y)**z).is_Pow
    assert expand_power_base(
        (sin((x*y)**2)*y)**z, force=True) == sin((x*y)**2)**z*y**z
    assert expand_power_base(
        (sin((x*y)**2)*y)**z, deep=True) == (sin(x**2*y**2)*y)**z

    assert expand_power_base(exp(x)**2) == exp(2*x)
    assert expand_power_base((exp(x)*exp(y))**2) == exp(2*x)*exp(2*y)

    assert expand_power_base(
        (exp((x*y)**z)*exp(y))**2) == exp(2*(x*y)**z)*exp(2*y)
    assert expand_power_base((exp((x*y)**z)*exp(
        y))**2, deep=True, force=True) == exp(2*x**z*y**z)*exp(2*y)

    assert expand_power_base((exp(x)*exp(y))**z).is_Pow
    assert expand_power_base(
        (exp(x)*exp(y))**z, force=True) == exp(x)**z*exp(y)**z


def test_expand_arit():
    a = Symbol("a")
    b = Symbol("b", positive=True)
    c = Symbol("c")

    p = R(5)
    e = (a + b)*c
    assert e == c*(a + b)
    assert (e.expand() - a*c - b*c) == R(0)
    e = (a + b)*(a + b)
    assert e == (a + b)**2
    assert e.expand() == 2*a*b + a**2 + b**2
    e = (a + b)*(a + b)**R(2)
    assert e == (a + b)**3
    assert e.expand() == 3*b*a**2 + 3*a*b**2 + a**3 + b**3
    assert e.expand() == 3*b*a**2 + 3*a*b**2 + a**3 + b**3
    e = (a + b)*(a + c)*(b + c)
    assert e == (a + c)*(a + b)*(b + c)
    assert e.expand() == 2*a*b*c + b*a**2 + c*a**2 + b*c**2 + a*c**2 + c*b**2 + a*b**2
    e = (a + R(1))**p
    assert e == (1 + a)**5
    assert e.expand() == 1 + 5*a + 10*a**2 + 10*a**3 + 5*a**4 + a**5
    e = (a + b + c)*(a + c + p)
    assert e == (5 + a + c)*(a + b + c)
    assert e.expand() == 5*a + 5*b + 5*c + 2*a*c + b*c + a*b + a**2 + c**2
    x = Symbol("x")
    s = exp(x*x) - 1
    e = s.nseries(x, 0, 6)/x**2
    assert e.expand() == 1 + x**2/2 + O(x**4)

    e = (x*(y + z))**(x*(y + z))*(x + y)
    assert e.expand(power_exp=False, power_base=False) == x*(x*y + x*
                    z)**(x*y + x*z) + y*(x*y + x*z)**(x*y + x*z)
    assert e.expand(power_exp=False, power_base=False, deep=False) == x* \
        (x*(y + z))**(x*(y + z)) + y*(x*(y + z))**(x*(y + z))
    e = x * (x + (y + 1)**2)
    assert e.expand(deep=False) == x**2 + x*(y + 1)**2
    e = (x*(y + z))**z
    assert e.expand(power_base=True, mul=True, deep=True) in [x**z*(y +
                    z)**z, (x*y + x*z)**z]
    assert ((2*y)**z).expand() == 2**z*y**z
    p = Symbol('p', positive=True)
    assert sqrt(-x).expand().is_Pow
    assert sqrt(-x).expand(force=True) == I*sqrt(x)
    assert ((2*y*p)**z).expand() == 2**z*p**z*y**z
    assert ((2*y*p*x)**z).expand() == 2**z*p**z*(x*y)**z
    assert ((2*y*p*x)**z).expand(force=True) == 2**z*p**z*x**z*y**z
    assert ((2*y*p*-pi)**z).expand() == 2**z*pi**z*p**z*(-y)**z
    assert ((2*y*p*-pi*x)**z).expand() == 2**z*pi**z*p**z*(-x*y)**z
    n = Symbol('n', negative=True)
    m = Symbol('m', negative=True)
    assert ((-2*x*y*n)**z).expand() == 2**z*(-n)**z*(x*y)**z
    assert ((-2*x*y*n*m)**z).expand() == 2**z*(-m)**z*(-n)**z*(-x*y)**z
    # issue 5482
    assert sqrt(-2*x*n) == sqrt(2)*sqrt(-n)*sqrt(x)
    # issue 5605 (2)
    assert (cos(x + y)**2).expand(trig=True) in [
        (-sin(x)*sin(y) + cos(x)*cos(y))**2,
        sin(x)**2*sin(y)**2 - 2*sin(x)*sin(y)*cos(x)*cos(y) + cos(x)**2*cos(y)**2
    ]

    # Check that this isn't too slow
    x = Symbol('x')
    W = 1
    for i in range(1, 21):
        W = W * (x - i)
    W = W.expand()
    assert W.has(-1672280820*x**15)

def test_expand_mul():
    # part of issue 20597
    e = Mul(2, 3, evaluate=False)
    assert e.expand() == 6

    e = Mul(2, 3, 1/x, evaluate=False)
    assert e.expand() == 6/x
    e = Mul(2, R(1, 3), evaluate=False)
    assert e.expand() == R(2, 3)

def test_power_expand():
    """Test for Pow.expand()"""
    a = Symbol('a')
    b = Symbol('b')
    p = (a + b)**2
    assert p.expand() == a**2 + b**2 + 2*a*b

    p = (1 + 2*(1 + a))**2
    assert p.expand() == 9 + 4*(a**2) + 12*a

    p = 2**(a + b)
    assert p.expand() == 2**a*2**b

    A = Symbol('A', commutative=False)
    B = Symbol('B', commutative=False)
    assert (2**(A + B)).expand() == 2**(A + B)
    assert (A**(a + b)).expand() != A**(a + b)


def test_issues_5919_6830():
    # issue 5919
    n = -1 + 1/x
    z = n/x/(-n)**2 - 1/n/x
    assert expand(z) == 1/(x**2 - 2*x + 1) - 1/(x - 2 + 1/x) - 1/(-x + 1)

    # issue 6830
    p = (1 + x)**2
    assert expand_multinomial((1 + x*p)**2) == (
        x**2*(x**4 + 4*x**3 + 6*x**2 + 4*x + 1) + 2*x*(x**2 + 2*x + 1) + 1)
    assert expand_multinomial((1 + (y + x)*p)**2) == (
        2*((x + y)*(x**2 + 2*x + 1)) + (x**2 + 2*x*y + y**2)*
        (x**4 + 4*x**3 + 6*x**2 + 4*x + 1) + 1)
    A = Symbol('A', commutative=False)
    p = (1 + A)**2
    assert expand_multinomial((1 + x*p)**2) == (
        x**2*(1 + 4*A + 6*A**2 + 4*A**3 + A**4) + 2*x*(1 + 2*A + A**2) + 1)
    assert expand_multinomial((1 + (y + x)*p)**2) == (
        (x + y)*(1 + 2*A + A**2)*2 + (x**2 + 2*x*y + y**2)*
        (1 + 4*A + 6*A**2 + 4*A**3 + A**4) + 1)
    assert expand_multinomial((1 + (y + x)*p)**3) == (
        (x + y)*(1 + 2*A + A**2)*3 + (x**2 + 2*x*y + y**2)*(1 + 4*A +
        6*A**2 + 4*A**3 + A**4)*3 + (x**3 + 3*x**2*y + 3*x*y**2 + y**3)*(1 + 6*A
        + 15*A**2 + 20*A**3 + 15*A**4 + 6*A**5 + A**6) + 1)
    # unevaluate powers
    eq = (Pow((x + 1)*((A + 1)**2), 2, evaluate=False))
    # - in this case the base is not an Add so no further
    #   expansion is done
    assert expand_multinomial(eq) == \
        (x**2 + 2*x + 1)*(1 + 4*A + 6*A**2 + 4*A**3 + A**4)
    # - but here, the expanded base *is* an Add so it gets expanded
    eq = (Pow(((A + 1)**2), 2, evaluate=False))
    assert expand_multinomial(eq) == 1 + 4*A + 6*A**2 + 4*A**3 + A**4

    # coverage
    def ok(a, b, n):
        e = (a + I*b)**n
        return verify_numerically(e, expand_multinomial(e))

    for a in [2, S.Half]:
        for b in [3, R(1, 3)]:
            for n in range(2, 6):
                assert ok(a, b, n)

    assert expand_multinomial((x + 1 + O(z))**2) == \
        1 + 2*x + x**2 + O(z)
    assert expand_multinomial((x + 1 + O(z))**3) == \
        1 + 3*x + 3*x**2 + x**3 + O(z)

    assert expand_multinomial(3**(x + y + 3)) == 27*3**(x + y)

def test_expand_log():
    t = Symbol('t', positive=True)
    # after first expansion, -2*log(2) + log(4); then 0 after second
    assert expand(log(t**2) - log(t**2/4) - 2*log(2)) == 0
    assert expand_log(log(7*6)/log(6)) == 1 + log(7)/log(6)
    b = factorial(10)
    assert expand_log(log(7*b**4)/log(b)
        ) == 4 + log(7)/log(b)


def test_issue_23952():
    assert (x**(y + z)).expand(force=True) == x**y*x**z
    one = Symbol('1', integer=True, prime=True, odd=True, positive=True)
    two = Symbol('2', integer=True, prime=True, even=True)
    e = two - one
    for b in (0, x):
        # 0**e = 0, 0**-e = zoo; but if expanded then nan
        assert unchanged(Pow, b, e)  # power_exp
        assert unchanged(Pow, b, -e)  # power_exp
        assert unchanged(Pow, b, y - x)  # power_exp
        assert unchanged(Pow, b, 3 - x)  # multinomial
        assert (b**e).expand().is_Pow  # power_exp
        assert (b**-e).expand().is_Pow  # power_exp
        assert (b**(y - x)).expand().is_Pow  # power_exp
        assert (b**(3 - x)).expand().is_Pow  # multinomial
    nn1 = Symbol('nn1', nonnegative=True)
    nn2 = Symbol('nn2', nonnegative=True)
    nn3 = Symbol('nn3', nonnegative=True)
    assert (x**(nn1 + nn2)).expand() == x**nn1*x**nn2
    assert (x**(-nn1 - nn2)).expand() == x**-nn1*x**-nn2
    assert unchanged(Pow, x, nn1 + nn2 - nn3)
    assert unchanged(Pow, x, 1 + nn2 - nn3)
    assert unchanged(Pow, x, nn1 - nn2)
    assert unchanged(Pow, x, 1 - nn2)
    assert unchanged(Pow, x, -1 + nn2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\apply\test_invalid_arg.py ===
# Tests specifically aimed at detecting bad arguments.
# This file is organized by reason for exception.
#     1. always invalid argument values
#     2. missing column(s)
#     3. incompatible ops/dtype/args/kwargs
#     4. invalid result shape/type
# If your test does not fit into one of these categories, add to this list.

from itertools import chain
import re

import numpy as np
import pytest

from pandas.errors import SpecificationError

from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


@pytest.mark.parametrize("result_type", ["foo", 1])
def test_result_type_error(result_type):
    # allowed result_type
    df = DataFrame(
        np.tile(np.arange(3, dtype="int64"), 6).reshape(6, -1) + 1,
        columns=["A", "B", "C"],
    )

    msg = (
        "invalid value for result_type, must be one of "
        "{None, 'reduce', 'broadcast', 'expand'}"
    )
    with pytest.raises(ValueError, match=msg):
        df.apply(lambda x: [1, 2, 3], axis=1, result_type=result_type)


def test_apply_invalid_axis_value():
    df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=["a", "a", "c"])
    msg = "No axis named 2 for object type DataFrame"
    with pytest.raises(ValueError, match=msg):
        df.apply(lambda x: x, 2)


def test_agg_raises():
    # GH 26513
    df = DataFrame({"A": [0, 1], "B": [1, 2]})
    msg = "Must provide"

    with pytest.raises(TypeError, match=msg):
        df.agg()


def test_map_with_invalid_na_action_raises():
    # https://github.com/pandas-dev/pandas/issues/32815
    s = Series([1, 2, 3])
    msg = "na_action must either be 'ignore' or None"
    with pytest.raises(ValueError, match=msg):
        s.map(lambda x: x, na_action="____")


@pytest.mark.parametrize("input_na_action", ["____", True])
def test_map_arg_is_dict_with_invalid_na_action_raises(input_na_action):
    # https://github.com/pandas-dev/pandas/issues/46588
    s = Series([1, 2, 3])
    msg = f"na_action must either be 'ignore' or None, {input_na_action} was passed"
    with pytest.raises(ValueError, match=msg):
        s.map({1: 2}, na_action=input_na_action)


@pytest.mark.parametrize("method", ["apply", "agg", "transform"])
@pytest.mark.parametrize("func", [{"A": {"B": "sum"}}, {"A": {"B": ["sum"]}}])
def test_nested_renamer(frame_or_series, method, func):
    # GH 35964
    obj = frame_or_series({"A": [1]})
    match = "nested renamer is not supported"
    with pytest.raises(SpecificationError, match=match):
        getattr(obj, method)(func)


@pytest.mark.parametrize(
    "renamer",
    [{"foo": ["min", "max"]}, {"foo": ["min", "max"], "bar": ["sum", "mean"]}],
)
def test_series_nested_renamer(renamer):
    s = Series(range(6), dtype="int64", name="series")
    msg = "nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        s.agg(renamer)


def test_apply_dict_depr():
    tsdf = DataFrame(
        np.random.default_rng(2).standard_normal((10, 3)),
        columns=["A", "B", "C"],
        index=date_range("1/1/2000", periods=10),
    )
    msg = "nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        tsdf.A.agg({"foo": ["sum", "mean"]})


@pytest.mark.parametrize("method", ["agg", "transform"])
def test_dict_nested_renaming_depr(method):
    df = DataFrame({"A": range(5), "B": 5})

    # nested renaming
    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        getattr(df, method)({"A": {"foo": "min"}, "B": {"bar": "max"}})


@pytest.mark.parametrize("method", ["apply", "agg", "transform"])
@pytest.mark.parametrize("func", [{"B": "sum"}, {"B": ["sum"]}])
def test_missing_column(method, func):
    # GH 40004
    obj = DataFrame({"A": [1]})
    match = re.escape("Column(s) ['B'] do not exist")
    with pytest.raises(KeyError, match=match):
        getattr(obj, method)(func)


def test_transform_mixed_column_name_dtypes():
    # GH39025
    df = DataFrame({"a": ["1"]})
    msg = r"Column\(s\) \[1, 'b'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        df.transform({"a": int, 1: str, "b": int})


@pytest.mark.parametrize(
    "how, args", [("pct_change", ()), ("nsmallest", (1, ["a", "b"])), ("tail", 1)]
)
def test_apply_str_axis_1_raises(how, args):
    # GH 39211 - some ops don't support axis=1
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    msg = f"Operation {how} does not support axis=1"
    with pytest.raises(ValueError, match=msg):
        df.apply(how, axis=1, args=args)


def test_transform_axis_1_raises():
    # GH 35964
    msg = "No axis named 1 for object type Series"
    with pytest.raises(ValueError, match=msg):
        Series([1]).transform("sum", axis=1)


def test_apply_modify_traceback():
    data = DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "foo",
                "foo",
                "foo",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
                "one",
            ],
            "C": [
                "dull",
                "dull",
                "shiny",
                "dull",
                "dull",
                "shiny",
                "shiny",
                "dull",
                "shiny",
                "shiny",
                "shiny",
            ],
            "D": np.random.default_rng(2).standard_normal(11),
            "E": np.random.default_rng(2).standard_normal(11),
            "F": np.random.default_rng(2).standard_normal(11),
        }
    )

    data.loc[4, "C"] = np.nan

    def transform(row):
        if row["C"].startswith("shin") and row["A"] == "foo":
            row["D"] = 7
        return row

    msg = "'float' object has no attribute 'startswith'"
    with pytest.raises(AttributeError, match=msg):
        data.apply(transform, axis=1)


@pytest.mark.parametrize(
    "df, func, expected",
    tm.get_cython_table_params(
        DataFrame([["a", "b"], ["b", "a"]]), [["cumprod", TypeError]]
    ),
)
def test_agg_cython_table_raises_frame(df, func, expected, axis, using_infer_string):
    # GH 21224
    if using_infer_string:
        expected = (expected, NotImplementedError)

    msg = (
        "can't multiply sequence by non-int of type 'str'"
        "|cannot perform cumprod with type str"  # NotImplementedError python backend
        "|operation 'cumprod' not supported for dtype 'str'"  # TypeError pyarrow
    )
    warn = None if isinstance(func, str) else FutureWarning
    with pytest.raises(expected, match=msg):
        with tm.assert_produces_warning(warn, match="using DataFrame.cumprod"):
            df.agg(func, axis=axis)


@pytest.mark.parametrize(
    "series, func, expected",
    chain(
        tm.get_cython_table_params(
            Series("a b c".split()),
            [
                ("mean", TypeError),  # mean raises TypeError
                ("prod", TypeError),
                ("std", TypeError),
                ("var", TypeError),
                ("median", TypeError),
                ("cumprod", TypeError),
            ],
        )
    ),
)
def test_agg_cython_table_raises_series(series, func, expected, using_infer_string):
    # GH21224
    msg = r"[Cc]ould not convert|can't multiply sequence by non-int of type"
    if func == "median" or func is np.nanmedian or func is np.median:
        msg = r"Cannot convert \['a' 'b' 'c'\] to numeric"

    if using_infer_string and func in ("cumprod", np.cumprod, np.nancumprod):
        expected = (expected, NotImplementedError)

    msg = (
        msg + "|does not support|has no kernel|Cannot perform|cannot perform|operation"
    )
    warn = None if isinstance(func, str) else FutureWarning

    with pytest.raises(expected, match=msg):
        # e.g. Series('a b'.split()).cumprod() will raise
        with tm.assert_produces_warning(warn, match="is currently using Series.*"):
            series.agg(func)


def test_agg_none_to_type():
    # GH 40543
    df = DataFrame({"a": [None]})
    msg = re.escape("int() argument must be a string")
    with pytest.raises(TypeError, match=msg):
        df.agg({"a": lambda x: int(x.iloc[0])})


def test_transform_none_to_type():
    # GH#34377
    df = DataFrame({"a": [None]})
    msg = "argument must be a"
    with pytest.raises(TypeError, match=msg):
        df.transform({"a": lambda x: int(x.iloc[0])})


@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.array([1, 2]).reshape(-1, 2),
        lambda x: [1, 2],
        lambda x: Series([1, 2]),
    ],
)
def test_apply_broadcast_error(func):
    df = DataFrame(
        np.tile(np.arange(3, dtype="int64"), 6).reshape(6, -1) + 1,
        columns=["A", "B", "C"],
    )

    # > 1 ndim
    msg = "too many dims to broadcast|cannot broadcast result"
    with pytest.raises(ValueError, match=msg):
        df.apply(func, axis=1, result_type="broadcast")


def test_transform_and_agg_err_agg(axis, float_frame):
    # cannot both transform and agg
    msg = "cannot combine transform and aggregation operations"
    with pytest.raises(ValueError, match=msg):
        with np.errstate(all="ignore"):
            float_frame.agg(["max", "sqrt"], axis=axis)


@pytest.mark.filterwarnings("ignore::FutureWarning")  # GH53325
@pytest.mark.parametrize(
    "func, msg",
    [
        (["sqrt", "max"], "cannot combine transform and aggregation"),
        (
            {"foo": np.sqrt, "bar": "sum"},
            "cannot perform both aggregation and transformation",
        ),
    ],
)
def test_transform_and_agg_err_series(string_series, func, msg):
    # we are trying to transform with an aggregator
    with pytest.raises(ValueError, match=msg):
        with np.errstate(all="ignore"):
            string_series.agg(func)


@pytest.mark.parametrize("func", [["max", "min"], ["max", "sqrt"]])
def test_transform_wont_agg_frame(axis, float_frame, func):
    # GH 35964
    # cannot both transform and agg
    msg = "Function did not transform"
    with pytest.raises(ValueError, match=msg):
        float_frame.transform(func, axis=axis)


@pytest.mark.parametrize("func", [["min", "max"], ["sqrt", "max"]])
def test_transform_wont_agg_series(string_series, func):
    # GH 35964
    # we are trying to transform with an aggregator
    msg = "Function did not transform"

    with pytest.raises(ValueError, match=msg):
        string_series.transform(func)


@pytest.mark.parametrize(
    "op_wrapper", [lambda x: x, lambda x: [x], lambda x: {"A": x}, lambda x: {"A": [x]}]
)
def test_transform_reducer_raises(all_reductions, frame_or_series, op_wrapper):
    # GH 35964
    op = op_wrapper(all_reductions)

    obj = DataFrame({"A": [1, 2, 3]})
    obj = tm.get_obj(obj, frame_or_series)

    msg = "Function did not transform"
    with pytest.raises(ValueError, match=msg):
        obj.transform(op)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_duplicates.py ===
from itertools import product

import numpy as np
import pytest

from pandas._libs import (
    hashtable,
    index as libindex,
)

from pandas import (
    NA,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm


@pytest.fixture
def idx_dup():
    # compare tests/indexes/multi/conftest.py
    major_axis = Index(["foo", "bar", "baz", "qux"])
    minor_axis = Index(["one", "two"])

    major_codes = np.array([0, 0, 1, 0, 1, 1])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])
    index_names = ["first", "second"]
    mi = MultiIndex(
        levels=[major_axis, minor_axis],
        codes=[major_codes, minor_codes],
        names=index_names,
        verify_integrity=False,
    )
    return mi


@pytest.mark.parametrize("names", [None, ["first", "second"]])
def test_unique(names):
    mi = MultiIndex.from_arrays([[1, 2, 1, 2], [1, 1, 1, 2]], names=names)

    res = mi.unique()
    exp = MultiIndex.from_arrays([[1, 2, 2], [1, 1, 2]], names=mi.names)
    tm.assert_index_equal(res, exp)

    mi = MultiIndex.from_arrays([list("aaaa"), list("abab")], names=names)
    res = mi.unique()
    exp = MultiIndex.from_arrays([list("aa"), list("ab")], names=mi.names)
    tm.assert_index_equal(res, exp)

    mi = MultiIndex.from_arrays([list("aaaa"), list("aaaa")], names=names)
    res = mi.unique()
    exp = MultiIndex.from_arrays([["a"], ["a"]], names=mi.names)
    tm.assert_index_equal(res, exp)

    # GH #20568 - empty MI
    mi = MultiIndex.from_arrays([[], []], names=names)
    res = mi.unique()
    tm.assert_index_equal(mi, res)


def test_unique_datetimelike():
    idx1 = DatetimeIndex(
        ["2015-01-01", "2015-01-01", "2015-01-01", "2015-01-01", "NaT", "NaT"]
    )
    idx2 = DatetimeIndex(
        ["2015-01-01", "2015-01-01", "2015-01-02", "2015-01-02", "NaT", "2015-01-01"],
        tz="Asia/Tokyo",
    )
    result = MultiIndex.from_arrays([idx1, idx2]).unique()

    eidx1 = DatetimeIndex(["2015-01-01", "2015-01-01", "NaT", "NaT"])
    eidx2 = DatetimeIndex(
        ["2015-01-01", "2015-01-02", "NaT", "2015-01-01"], tz="Asia/Tokyo"
    )
    exp = MultiIndex.from_arrays([eidx1, eidx2])
    tm.assert_index_equal(result, exp)


@pytest.mark.parametrize("level", [0, "first", 1, "second"])
def test_unique_level(idx, level):
    # GH #17896 - with level= argument
    result = idx.unique(level=level)
    expected = idx.get_level_values(level).unique()
    tm.assert_index_equal(result, expected)

    # With already unique level
    mi = MultiIndex.from_arrays([[1, 3, 2, 4], [1, 3, 2, 5]], names=["first", "second"])
    result = mi.unique(level=level)
    expected = mi.get_level_values(level)
    tm.assert_index_equal(result, expected)

    # With empty MI
    mi = MultiIndex.from_arrays([[], []], names=["first", "second"])
    result = mi.unique(level=level)
    expected = mi.get_level_values(level)
    tm.assert_index_equal(result, expected)


def test_duplicate_multiindex_codes():
    # GH 17464
    # Make sure that a MultiIndex with duplicate levels throws a ValueError
    msg = r"Level values must be unique: \[[A', ]+\] on level 0"
    with pytest.raises(ValueError, match=msg):
        mi = MultiIndex([["A"] * 10, range(10)], [[0] * 10, range(10)])

    # And that using set_levels with duplicate levels fails
    mi = MultiIndex.from_arrays([["A", "A", "B", "B", "B"], [1, 2, 1, 2, 3]])
    msg = r"Level values must be unique: \[[AB', ]+\] on level 0"
    with pytest.raises(ValueError, match=msg):
        mi.set_levels([["A", "B", "A", "A", "B"], [2, 1, 3, -2, 5]])


@pytest.mark.parametrize("names", [["a", "b", "a"], [1, 1, 2], [1, "a", 1]])
def test_duplicate_level_names(names):
    # GH18872, GH19029
    mi = MultiIndex.from_product([[0, 1]] * 3, names=names)
    assert mi.names == names

    # With .rename()
    mi = MultiIndex.from_product([[0, 1]] * 3)
    mi = mi.rename(names)
    assert mi.names == names

    # With .rename(., level=)
    mi.rename(names[1], level=1, inplace=True)
    mi = mi.rename([names[0], names[2]], level=[0, 2])
    assert mi.names == names


def test_duplicate_meta_data():
    # GH 10115
    mi = MultiIndex(
        levels=[[0, 1], [0, 1, 2]], codes=[[0, 0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 0, 1, 2]]
    )

    for idx in [
        mi,
        mi.set_names([None, None]),
        mi.set_names([None, "Num"]),
        mi.set_names(["Upper", "Num"]),
    ]:
        assert idx.has_duplicates
        assert idx.drop_duplicates().names == idx.names


def test_has_duplicates(idx, idx_dup):
    # see fixtures
    assert idx.is_unique is True
    assert idx.has_duplicates is False
    assert idx_dup.is_unique is False
    assert idx_dup.has_duplicates is True

    mi = MultiIndex(
        levels=[[0, 1], [0, 1, 2]], codes=[[0, 0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 0, 1, 2]]
    )
    assert mi.is_unique is False
    assert mi.has_duplicates is True

    # single instance of NaN
    mi_nan = MultiIndex(
        levels=[["a", "b"], [0, 1]], codes=[[-1, 0, 0, 1, 1], [-1, 0, 1, 0, 1]]
    )
    assert mi_nan.is_unique is True
    assert mi_nan.has_duplicates is False

    # multiple instances of NaN
    mi_nan_dup = MultiIndex(
        levels=[["a", "b"], [0, 1]], codes=[[-1, -1, 0, 0, 1, 1], [-1, -1, 0, 1, 0, 1]]
    )
    assert mi_nan_dup.is_unique is False
    assert mi_nan_dup.has_duplicates is True


def test_has_duplicates_from_tuples():
    # GH 9075
    t = [
        ("x", "out", "z", 5, "y", "in", "z", 169),
        ("x", "out", "z", 7, "y", "in", "z", 119),
        ("x", "out", "z", 9, "y", "in", "z", 135),
        ("x", "out", "z", 13, "y", "in", "z", 145),
        ("x", "out", "z", 14, "y", "in", "z", 158),
        ("x", "out", "z", 16, "y", "in", "z", 122),
        ("x", "out", "z", 17, "y", "in", "z", 160),
        ("x", "out", "z", 18, "y", "in", "z", 180),
        ("x", "out", "z", 20, "y", "in", "z", 143),
        ("x", "out", "z", 21, "y", "in", "z", 128),
        ("x", "out", "z", 22, "y", "in", "z", 129),
        ("x", "out", "z", 25, "y", "in", "z", 111),
        ("x", "out", "z", 28, "y", "in", "z", 114),
        ("x", "out", "z", 29, "y", "in", "z", 121),
        ("x", "out", "z", 31, "y", "in", "z", 126),
        ("x", "out", "z", 32, "y", "in", "z", 155),
        ("x", "out", "z", 33, "y", "in", "z", 123),
        ("x", "out", "z", 12, "y", "in", "z", 144),
    ]

    mi = MultiIndex.from_tuples(t)
    assert not mi.has_duplicates


@pytest.mark.parametrize("nlevels", [4, 8])
@pytest.mark.parametrize("with_nulls", [True, False])
def test_has_duplicates_overflow(nlevels, with_nulls):
    # handle int64 overflow if possible
    # no overflow with 4
    # overflow possible with 8
    codes = np.tile(np.arange(500), 2)
    level = np.arange(500)

    if with_nulls:  # inject some null values
        codes[500] = -1  # common nan value
        codes = [codes.copy() for i in range(nlevels)]
        for i in range(nlevels):
            codes[i][500 + i - nlevels // 2] = -1

        codes += [np.array([-1, 1]).repeat(500)]
    else:
        codes = [codes] * nlevels + [np.arange(2).repeat(500)]

    levels = [level] * nlevels + [[0, 1]]

    # no dups
    mi = MultiIndex(levels=levels, codes=codes)
    assert not mi.has_duplicates

    # with a dup
    if with_nulls:

        def f(a):
            return np.insert(a, 1000, a[0])

        codes = list(map(f, codes))
        mi = MultiIndex(levels=levels, codes=codes)
    else:
        values = mi.values.tolist()
        mi = MultiIndex.from_tuples(values + [values[0]])

    assert mi.has_duplicates


@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", np.array([False, False, False, True, True, False])),
        ("last", np.array([False, True, True, False, False, False])),
        (False, np.array([False, True, True, True, True, False])),
    ],
)
def test_duplicated(idx_dup, keep, expected):
    result = idx_dup.duplicated(keep=keep)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.arm_slow
def test_duplicated_hashtable_impl(keep, monkeypatch):
    # GH 9125
    n, k = 6, 10
    levels = [np.arange(n), [str(i) for i in range(n)], 1000 + np.arange(n)]
    codes = [np.random.default_rng(2).choice(n, k * n) for _ in levels]
    with monkeypatch.context() as m:
        m.setattr(libindex, "_SIZE_CUTOFF", 50)
        mi = MultiIndex(levels=levels, codes=codes)

        result = mi.duplicated(keep=keep)
        expected = hashtable.duplicated(mi.values, keep=keep)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("val", [101, 102])
def test_duplicated_with_nan(val):
    # GH5873
    mi = MultiIndex.from_arrays([[101, val], [3.5, np.nan]])
    assert not mi.has_duplicates

    tm.assert_numpy_array_equal(mi.duplicated(), np.zeros(2, dtype="bool"))


@pytest.mark.parametrize("n", range(1, 6))
@pytest.mark.parametrize("m", range(1, 5))
def test_duplicated_with_nan_multi_shape(n, m):
    # GH5873
    # all possible unique combinations, including nan
    codes = product(range(-1, n), range(-1, m))
    mi = MultiIndex(
        levels=[list("abcde")[:n], list("WXYZ")[:m]],
        codes=np.random.default_rng(2).permutation(list(codes)).T,
    )
    assert len(mi) == (n + 1) * (m + 1)
    assert not mi.has_duplicates

    tm.assert_numpy_array_equal(mi.duplicated(), np.zeros(len(mi), dtype="bool"))


def test_duplicated_drop_duplicates():
    # GH#4060
    idx = MultiIndex.from_arrays(([1, 2, 3, 1, 2, 3], [1, 1, 1, 1, 2, 2]))

    expected = np.array([False, False, False, True, False, False], dtype=bool)
    duplicated = idx.duplicated()
    tm.assert_numpy_array_equal(duplicated, expected)
    assert duplicated.dtype == bool
    expected = MultiIndex.from_arrays(([1, 2, 3, 2, 3], [1, 1, 1, 2, 2]))
    tm.assert_index_equal(idx.drop_duplicates(), expected)

    expected = np.array([True, False, False, False, False, False])
    duplicated = idx.duplicated(keep="last")
    tm.assert_numpy_array_equal(duplicated, expected)
    assert duplicated.dtype == bool
    expected = MultiIndex.from_arrays(([2, 3, 1, 2, 3], [1, 1, 1, 2, 2]))
    tm.assert_index_equal(idx.drop_duplicates(keep="last"), expected)

    expected = np.array([True, False, False, True, False, False])
    duplicated = idx.duplicated(keep=False)
    tm.assert_numpy_array_equal(duplicated, expected)
    assert duplicated.dtype == bool
    expected = MultiIndex.from_arrays(([2, 3, 2, 3], [1, 1, 2, 2]))
    tm.assert_index_equal(idx.drop_duplicates(keep=False), expected)


@pytest.mark.parametrize(
    "dtype",
    [
        np.complex64,
        np.complex128,
    ],
)
def test_duplicated_series_complex_numbers(dtype):
    # GH 17927
    expected = Series(
        [False, False, False, True, False, False, False, True, False, True],
        dtype=bool,
    )
    result = Series(
        [
            np.nan + np.nan * 1j,
            0,
            1j,
            1j,
            1,
            1 + 1j,
            1 + 2j,
            1 + 1j,
            np.nan,
            np.nan + np.nan * 1j,
        ],
        dtype=dtype,
    ).duplicated()
    tm.assert_series_equal(result, expected)


def test_midx_unique_ea_dtype():
    # GH#48335
    vals_a = Series([1, 2, NA, NA], dtype="Int64")
    vals_b = np.array([1, 2, 3, 3])
    midx = MultiIndex.from_arrays([vals_a, vals_b], names=["a", "b"])
    result = midx.unique()

    exp_vals_a = Series([1, 2, NA], dtype="Int64")
    exp_vals_b = np.array([1, 2, 3])
    expected = MultiIndex.from_arrays([exp_vals_a, exp_vals_b], names=["a", "b"])
    tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\test_setops.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    PeriodIndex,
    date_range,
    period_range,
)
import pandas._testing as tm


def _permute(obj):
    return obj.take(np.random.default_rng(2).permutation(len(obj)))


class TestPeriodIndex:
    def test_union(self, sort):
        # union
        other1 = period_range("1/1/2000", freq="D", periods=5)
        rng1 = period_range("1/6/2000", freq="D", periods=5)
        expected1 = PeriodIndex(
            [
                "2000-01-06",
                "2000-01-07",
                "2000-01-08",
                "2000-01-09",
                "2000-01-10",
                "2000-01-01",
                "2000-01-02",
                "2000-01-03",
                "2000-01-04",
                "2000-01-05",
            ],
            freq="D",
        )

        rng2 = period_range("1/1/2000", freq="D", periods=5)
        other2 = period_range("1/4/2000", freq="D", periods=5)
        expected2 = period_range("1/1/2000", freq="D", periods=8)

        rng3 = period_range("1/1/2000", freq="D", periods=5)
        other3 = PeriodIndex([], freq="D")
        expected3 = period_range("1/1/2000", freq="D", periods=5)

        rng4 = period_range("2000-01-01 09:00", freq="h", periods=5)
        other4 = period_range("2000-01-02 09:00", freq="h", periods=5)
        expected4 = PeriodIndex(
            [
                "2000-01-01 09:00",
                "2000-01-01 10:00",
                "2000-01-01 11:00",
                "2000-01-01 12:00",
                "2000-01-01 13:00",
                "2000-01-02 09:00",
                "2000-01-02 10:00",
                "2000-01-02 11:00",
                "2000-01-02 12:00",
                "2000-01-02 13:00",
            ],
            freq="h",
        )

        rng5 = PeriodIndex(
            ["2000-01-01 09:01", "2000-01-01 09:03", "2000-01-01 09:05"], freq="min"
        )
        other5 = PeriodIndex(
            ["2000-01-01 09:01", "2000-01-01 09:05", "2000-01-01 09:08"], freq="min"
        )
        expected5 = PeriodIndex(
            [
                "2000-01-01 09:01",
                "2000-01-01 09:03",
                "2000-01-01 09:05",
                "2000-01-01 09:08",
            ],
            freq="min",
        )

        rng6 = period_range("2000-01-01", freq="M", periods=7)
        other6 = period_range("2000-04-01", freq="M", periods=7)
        expected6 = period_range("2000-01-01", freq="M", periods=10)

        rng7 = period_range("2003-01-01", freq="Y", periods=5)
        other7 = period_range("1998-01-01", freq="Y", periods=8)
        expected7 = PeriodIndex(
            [
                "2003",
                "2004",
                "2005",
                "2006",
                "2007",
                "1998",
                "1999",
                "2000",
                "2001",
                "2002",
            ],
            freq="Y",
        )

        rng8 = PeriodIndex(
            ["1/3/2000", "1/2/2000", "1/1/2000", "1/5/2000", "1/4/2000"], freq="D"
        )
        other8 = period_range("1/6/2000", freq="D", periods=5)
        expected8 = PeriodIndex(
            [
                "1/3/2000",
                "1/2/2000",
                "1/1/2000",
                "1/5/2000",
                "1/4/2000",
                "1/6/2000",
                "1/7/2000",
                "1/8/2000",
                "1/9/2000",
                "1/10/2000",
            ],
            freq="D",
        )

        for rng, other, expected in [
            (rng1, other1, expected1),
            (rng2, other2, expected2),
            (rng3, other3, expected3),
            (rng4, other4, expected4),
            (rng5, other5, expected5),
            (rng6, other6, expected6),
            (rng7, other7, expected7),
            (rng8, other8, expected8),
        ]:
            result_union = rng.union(other, sort=sort)
            if sort is None:
                expected = expected.sort_values()
            tm.assert_index_equal(result_union, expected)

    def test_union_misc(self, sort):
        index = period_range("1/1/2000", "1/20/2000", freq="D")

        result = index[:-5].union(index[10:], sort=sort)
        tm.assert_index_equal(result, index)

        # not in order
        result = _permute(index[:-5]).union(_permute(index[10:]), sort=sort)
        if sort is False:
            tm.assert_index_equal(result.sort_values(), index)
        else:
            tm.assert_index_equal(result, index)

        # cast if different frequencies
        index = period_range("1/1/2000", "1/20/2000", freq="D")
        index2 = period_range("1/1/2000", "1/20/2000", freq="W-WED")
        result = index.union(index2, sort=sort)
        expected = index.astype(object).union(index2.astype(object), sort=sort)
        tm.assert_index_equal(result, expected)

    def test_intersection(self, sort):
        index = period_range("1/1/2000", "1/20/2000", freq="D")

        result = index[:-5].intersection(index[10:], sort=sort)
        tm.assert_index_equal(result, index[10:-5])

        # not in order
        left = _permute(index[:-5])
        right = _permute(index[10:])
        result = left.intersection(right, sort=sort)
        if sort is False:
            tm.assert_index_equal(result.sort_values(), index[10:-5])
        else:
            tm.assert_index_equal(result, index[10:-5])

        # cast if different frequencies
        index = period_range("1/1/2000", "1/20/2000", freq="D")
        index2 = period_range("1/1/2000", "1/20/2000", freq="W-WED")

        result = index.intersection(index2, sort=sort)
        expected = pd.Index([], dtype=object)
        tm.assert_index_equal(result, expected)

        index3 = period_range("1/1/2000", "1/20/2000", freq="2D")
        result = index.intersection(index3, sort=sort)
        tm.assert_index_equal(result, expected)

    def test_intersection_cases(self, sort):
        base = period_range("6/1/2000", "6/30/2000", freq="D", name="idx")

        # if target has the same name, it is preserved
        rng2 = period_range("5/15/2000", "6/20/2000", freq="D", name="idx")
        expected2 = period_range("6/1/2000", "6/20/2000", freq="D", name="idx")

        # if target name is different, it will be reset
        rng3 = period_range("5/15/2000", "6/20/2000", freq="D", name="other")
        expected3 = period_range("6/1/2000", "6/20/2000", freq="D", name=None)

        rng4 = period_range("7/1/2000", "7/31/2000", freq="D", name="idx")
        expected4 = PeriodIndex([], name="idx", freq="D")

        for rng, expected in [
            (rng2, expected2),
            (rng3, expected3),
            (rng4, expected4),
        ]:
            result = base.intersection(rng, sort=sort)
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == expected.freq

        # non-monotonic
        base = PeriodIndex(
            ["2011-01-05", "2011-01-04", "2011-01-02", "2011-01-03"],
            freq="D",
            name="idx",
        )

        rng2 = PeriodIndex(
            ["2011-01-04", "2011-01-02", "2011-02-02", "2011-02-03"],
            freq="D",
            name="idx",
        )
        expected2 = PeriodIndex(["2011-01-04", "2011-01-02"], freq="D", name="idx")

        rng3 = PeriodIndex(
            ["2011-01-04", "2011-01-02", "2011-02-02", "2011-02-03"],
            freq="D",
            name="other",
        )
        expected3 = PeriodIndex(["2011-01-04", "2011-01-02"], freq="D", name=None)

        rng4 = period_range("7/1/2000", "7/31/2000", freq="D", name="idx")
        expected4 = PeriodIndex([], freq="D", name="idx")

        for rng, expected in [
            (rng2, expected2),
            (rng3, expected3),
            (rng4, expected4),
        ]:
            result = base.intersection(rng, sort=sort)
            if sort is None:
                expected = expected.sort_values()
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == "D"

        # empty same freq
        rng = date_range("6/1/2000", "6/15/2000", freq="min")
        result = rng[0:0].intersection(rng)
        assert len(result) == 0

        result = rng.intersection(rng[0:0])
        assert len(result) == 0

    def test_difference(self, sort):
        # diff
        period_rng = ["1/3/2000", "1/2/2000", "1/1/2000", "1/5/2000", "1/4/2000"]
        rng1 = PeriodIndex(period_rng, freq="D")
        other1 = period_range("1/6/2000", freq="D", periods=5)
        expected1 = rng1

        rng2 = PeriodIndex(period_rng, freq="D")
        other2 = period_range("1/4/2000", freq="D", periods=5)
        expected2 = PeriodIndex(["1/3/2000", "1/2/2000", "1/1/2000"], freq="D")

        rng3 = PeriodIndex(period_rng, freq="D")
        other3 = PeriodIndex([], freq="D")
        expected3 = rng3

        period_rng = [
            "2000-01-01 10:00",
            "2000-01-01 09:00",
            "2000-01-01 12:00",
            "2000-01-01 11:00",
            "2000-01-01 13:00",
        ]
        rng4 = PeriodIndex(period_rng, freq="h")
        other4 = period_range("2000-01-02 09:00", freq="h", periods=5)
        expected4 = rng4

        rng5 = PeriodIndex(
            ["2000-01-01 09:03", "2000-01-01 09:01", "2000-01-01 09:05"], freq="min"
        )
        other5 = PeriodIndex(["2000-01-01 09:01", "2000-01-01 09:05"], freq="min")
        expected5 = PeriodIndex(["2000-01-01 09:03"], freq="min")

        period_rng = [
            "2000-02-01",
            "2000-01-01",
            "2000-06-01",
            "2000-07-01",
            "2000-05-01",
            "2000-03-01",
            "2000-04-01",
        ]
        rng6 = PeriodIndex(period_rng, freq="M")
        other6 = period_range("2000-04-01", freq="M", periods=7)
        expected6 = PeriodIndex(["2000-02-01", "2000-01-01", "2000-03-01"], freq="M")

        period_rng = ["2003", "2007", "2006", "2005", "2004"]
        rng7 = PeriodIndex(period_rng, freq="Y")
        other7 = period_range("1998-01-01", freq="Y", periods=8)
        expected7 = PeriodIndex(["2007", "2006"], freq="Y")

        for rng, other, expected in [
            (rng1, other1, expected1),
            (rng2, other2, expected2),
            (rng3, other3, expected3),
            (rng4, other4, expected4),
            (rng5, other5, expected5),
            (rng6, other6, expected6),
            (rng7, other7, expected7),
        ]:
            result_difference = rng.difference(other, sort=sort)
            if sort is None and len(other):
                # We dont sort (yet?) when empty GH#24959
                expected = expected.sort_values()
            tm.assert_index_equal(result_difference, expected)

    def test_difference_freq(self, sort):
        # GH14323: difference of Period MUST preserve frequency
        # but the ability to union results must be preserved

        index = period_range("20160920", "20160925", freq="D")

        other = period_range("20160921", "20160924", freq="D")
        expected = PeriodIndex(["20160920", "20160925"], freq="D")
        idx_diff = index.difference(other, sort)
        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal("freq", idx_diff, expected)

        other = period_range("20160922", "20160925", freq="D")
        idx_diff = index.difference(other, sort)
        expected = PeriodIndex(["20160920", "20160921"], freq="D")
        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal("freq", idx_diff, expected)

    def test_intersection_equal_duplicates(self):
        # GH#38302
        idx = period_range("2011-01-01", periods=2)
        idx_dup = idx.append(idx)
        result = idx_dup.intersection(idx_dup)
        tm.assert_index_equal(result, idx)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_union_duplicates(self):
        # GH#36289
        idx = period_range("2011-01-01", periods=2)
        idx_dup = idx.append(idx)

        idx2 = period_range("2011-01-02", periods=2)
        idx2_dup = idx2.append(idx2)
        result = idx_dup.union(idx2_dup)

        expected = PeriodIndex(
            [
                "2011-01-01",
                "2011-01-01",
                "2011-01-02",
                "2011-01-02",
                "2011-01-03",
                "2011-01-03",
            ],
            freq="D",
        )
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_wrapping_geometry.py ===
"""Tests for the ``sympy.physics.mechanics.wrapping_geometry.py`` module."""

import pytest

from sympy import (
    Integer,
    Rational,
    S,
    Symbol,
    acos,
    cos,
    pi,
    sin,
    sqrt,
)
from sympy.core.relational import Eq
from sympy.physics.mechanics import (
    Point,
    ReferenceFrame,
    WrappingCylinder,
    WrappingSphere,
    dynamicsymbols,
)
from sympy.simplify.simplify import simplify


r = Symbol('r', positive=True)
x = Symbol('x')
q = dynamicsymbols('q')
N = ReferenceFrame('N')


class TestWrappingSphere:

    @staticmethod
    def test_valid_constructor():
        r = Symbol('r', positive=True)
        pO = Point('pO')
        sphere = WrappingSphere(r, pO)
        assert isinstance(sphere, WrappingSphere)
        assert hasattr(sphere, 'radius')
        assert sphere.radius == r
        assert hasattr(sphere, 'point')
        assert sphere.point == pO

    @staticmethod
    @pytest.mark.parametrize('position', [S.Zero, Integer(2)*r*N.x])
    def test_geodesic_length_point_not_on_surface_invalid(position):
        r = Symbol('r', positive=True)
        pO = Point('pO')
        sphere = WrappingSphere(r, pO)

        p1 = Point('p1')
        p1.set_pos(pO, position)
        p2 = Point('p2')
        p2.set_pos(pO, position)

        error_msg = r'point .* does not lie on the surface of'
        with pytest.raises(ValueError, match=error_msg):
            sphere.geodesic_length(p1, p2)

    @staticmethod
    @pytest.mark.parametrize(
        'position_1, position_2, expected',
        [
            (r*N.x, r*N.x, S.Zero),
            (r*N.x, r*N.y, S.Half*pi*r),
            (r*N.x, r*-N.x, pi*r),
            (r*-N.x, r*N.x, pi*r),
            (r*N.x, r*sqrt(2)*S.Half*(N.x + N.y), Rational(1, 4)*pi*r),
            (
                r*sqrt(2)*S.Half*(N.x + N.y),
                r*sqrt(3)*Rational(1, 3)*(N.x + N.y + N.z),
                r*acos(sqrt(6)*Rational(1, 3)),
            ),
        ]
    )
    def test_geodesic_length(position_1, position_2, expected):
        r = Symbol('r', positive=True)
        pO = Point('pO')
        sphere = WrappingSphere(r, pO)

        p1 = Point('p1')
        p1.set_pos(pO, position_1)
        p2 = Point('p2')
        p2.set_pos(pO, position_2)

        assert simplify(Eq(sphere.geodesic_length(p1, p2), expected))

    @staticmethod
    @pytest.mark.parametrize(
        'position_1, position_2, vector_1, vector_2',
        [
            (r * N.x, r * N.y, N.y, N.x),
            (r * N.x, -r * N.y, -N.y, N.x),
            (
                r * N.y,
                sqrt(2)/2 * r * N.x - sqrt(2)/2 * r * N.y,
                N.x,
                sqrt(2)/2 * N.x + sqrt(2)/2 * N.y,
            ),
            (
                r * N.x,
                r / 2 * N.x + sqrt(3)/2 * r * N.y,
                N.y,
                sqrt(3)/2 * N.x - 1/2 * N.y,
            ),
            (
                r * N.x,
                sqrt(2)/2 * r * N.x + sqrt(2)/2 * r * N.y,
                N.y,
                sqrt(2)/2 * N.x - sqrt(2)/2 * N.y,
            ),
        ]
    )
    def test_geodesic_end_vectors(position_1, position_2, vector_1, vector_2):
        r = Symbol('r', positive=True)
        pO = Point('pO')
        sphere = WrappingSphere(r, pO)

        p1 = Point('p1')
        p1.set_pos(pO, position_1)
        p2 = Point('p2')
        p2.set_pos(pO, position_2)

        expected = (vector_1, vector_2)

        assert sphere.geodesic_end_vectors(p1, p2) == expected

    @staticmethod
    @pytest.mark.parametrize(
        'position',
        [r * N.x, r * cos(q) * N.x + r * sin(q) * N.y]
    )
    def test_geodesic_end_vectors_invalid_coincident(position):
        r = Symbol('r', positive=True)
        pO = Point('pO')
        sphere = WrappingSphere(r, pO)

        p1 = Point('p1')
        p1.set_pos(pO, position)
        p2 = Point('p2')
        p2.set_pos(pO, position)

        with pytest.raises(ValueError):
            _ = sphere.geodesic_end_vectors(p1, p2)

    @staticmethod
    @pytest.mark.parametrize(
        'position_1, position_2',
        [
            (r * N.x, -r * N.x),
            (-r * N.y, r * N.y),
            (
                r * cos(q) * N.x + r * sin(q) * N.y,
                -r * cos(q) * N.x - r * sin(q) * N.y,
            )
        ]
    )
    def test_geodesic_end_vectors_invalid_diametrically_opposite(
        position_1,
        position_2,
    ):
        r = Symbol('r', positive=True)
        pO = Point('pO')
        sphere = WrappingSphere(r, pO)

        p1 = Point('p1')
        p1.set_pos(pO, position_1)
        p2 = Point('p2')
        p2.set_pos(pO, position_2)

        with pytest.raises(ValueError):
            _ = sphere.geodesic_end_vectors(p1, p2)


class TestWrappingCylinder:

    @staticmethod
    def test_valid_constructor():
        N = ReferenceFrame('N')
        r = Symbol('r', positive=True)
        pO = Point('pO')
        cylinder = WrappingCylinder(r, pO, N.x)
        assert isinstance(cylinder, WrappingCylinder)
        assert hasattr(cylinder, 'radius')
        assert cylinder.radius == r
        assert hasattr(cylinder, 'point')
        assert cylinder.point == pO
        assert hasattr(cylinder, 'axis')
        assert cylinder.axis == N.x

    @staticmethod
    @pytest.mark.parametrize(
        'position, expected',
        [
            (S.Zero, False),
            (r*N.y, True),
            (r*N.z, True),
            (r*(N.y + N.z).normalize(), True),
            (Integer(2)*r*N.y, False),
            (r*(N.x + N.y), True),
            (r*(Integer(2)*N.x + N.y), True),
            (Integer(2)*N.x + r*(Integer(2)*N.y + N.z).normalize(), True),
            (r*(cos(q)*N.y + sin(q)*N.z), True)
        ]
    )
    def test_point_is_on_surface(position, expected):
        r = Symbol('r', positive=True)
        pO = Point('pO')
        cylinder = WrappingCylinder(r, pO, N.x)

        p1 = Point('p1')
        p1.set_pos(pO, position)

        assert cylinder.point_on_surface(p1) is expected

    @staticmethod
    @pytest.mark.parametrize('position', [S.Zero, Integer(2)*r*N.y])
    def test_geodesic_length_point_not_on_surface_invalid(position):
        r = Symbol('r', positive=True)
        pO = Point('pO')
        cylinder = WrappingCylinder(r, pO, N.x)

        p1 = Point('p1')
        p1.set_pos(pO, position)
        p2 = Point('p2')
        p2.set_pos(pO, position)

        error_msg = r'point .* does not lie on the surface of'
        with pytest.raises(ValueError, match=error_msg):
            cylinder.geodesic_length(p1, p2)

    @staticmethod
    @pytest.mark.parametrize(
        'axis, position_1, position_2, expected',
        [
            (N.x, r*N.y, r*N.y, S.Zero),
            (N.x, r*N.y, N.x + r*N.y, S.One),
            (N.x, r*N.y, -x*N.x + r*N.y, sqrt(x**2)),
            (-N.x, r*N.y, x*N.x + r*N.y, sqrt(x**2)),
            (N.x, r*N.y, r*N.z, S.Half*pi*sqrt(r**2)),
            (-N.x, r*N.y, r*N.z, Integer(3)*S.Half*pi*sqrt(r**2)),
            (N.x, r*N.z, r*N.y, Integer(3)*S.Half*pi*sqrt(r**2)),
            (-N.x, r*N.z, r*N.y, S.Half*pi*sqrt(r**2)),
            (N.x, r*N.y, r*(cos(q)*N.y + sin(q)*N.z), sqrt(r**2*q**2)),
            (
                -N.x, r*N.y,
                r*(cos(q)*N.y + sin(q)*N.z),
                sqrt(r**2*(Integer(2)*pi - q)**2),
            ),
        ]
    )
    def test_geodesic_length(axis, position_1, position_2, expected):
        r = Symbol('r', positive=True)
        pO = Point('pO')
        cylinder = WrappingCylinder(r, pO, axis)

        p1 = Point('p1')
        p1.set_pos(pO, position_1)
        p2 = Point('p2')
        p2.set_pos(pO, position_2)

        assert simplify(Eq(cylinder.geodesic_length(p1, p2), expected))

    @staticmethod
    @pytest.mark.parametrize(
        'axis, position_1, position_2, vector_1, vector_2',
        [
            (N.z, r * N.x, r * N.y, N.y, N.x),
            (N.z, r * N.x, -r * N.x, N.y, N.y),
            (N.z, -r * N.x, r * N.x, -N.y, -N.y),
            (-N.z, r * N.x, -r * N.x, -N.y, -N.y),
            (-N.z, -r * N.x, r * N.x, N.y, N.y),
            (N.z, r * N.x, -r * N.y, N.y, -N.x),
            (
                N.z,
                r * N.y,
                sqrt(2)/2 * r * N.x - sqrt(2)/2 * r * N.y,
                - N.x,
                - sqrt(2)/2 * N.x - sqrt(2)/2 * N.y,
            ),
            (
                N.z,
                r * N.x,
                r / 2 * N.x + sqrt(3)/2 * r * N.y,
                N.y,
                sqrt(3)/2 * N.x - 1/2 * N.y,
            ),
            (
                N.z,
                r * N.x,
                sqrt(2)/2 * r * N.x + sqrt(2)/2 * r * N.y,
                N.y,
                sqrt(2)/2 * N.x - sqrt(2)/2 * N.y,
            ),
            (
                N.z,
                r * N.x,
                r * N.x + N.z,
                N.z,
                -N.z,
            ),
            (
                N.z,
                r * N.x,
                r * N.y + pi/2 * r * N.z,
                sqrt(2)/2 * N.y + sqrt(2)/2 * N.z,
                sqrt(2)/2 * N.x - sqrt(2)/2 * N.z,
            ),
            (
                N.z,
                r * N.x,
                r * cos(q) * N.x + r * sin(q) * N.y,
                N.y,
                sin(q) * N.x - cos(q) * N.y,
            ),
        ]
    )
    def test_geodesic_end_vectors(
        axis,
        position_1,
        position_2,
        vector_1,
        vector_2,
    ):
        r = Symbol('r', positive=True)
        pO = Point('pO')
        cylinder = WrappingCylinder(r, pO, axis)

        p1 = Point('p1')
        p1.set_pos(pO, position_1)
        p2 = Point('p2')
        p2.set_pos(pO, position_2)

        expected = (vector_1, vector_2)
        end_vectors = tuple(
            end_vector.simplify()
            for end_vector in cylinder.geodesic_end_vectors(p1, p2)
        )

        assert end_vectors == expected

    @staticmethod
    @pytest.mark.parametrize(
        'axis, position',
        [
            (N.z, r * N.x),
            (N.z, r * cos(q) * N.x + r * sin(q) * N.y + N.z),
        ]
    )
    def test_geodesic_end_vectors_invalid_coincident(axis, position):
        r = Symbol('r', positive=True)
        pO = Point('pO')
        cylinder = WrappingCylinder(r, pO, axis)

        p1 = Point('p1')
        p1.set_pos(pO, position)
        p2 = Point('p2')
        p2.set_pos(pO, position)

        with pytest.raises(ValueError):
            _ = cylinder.geodesic_end_vectors(p1, p2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_rust.py ===
from sympy.core import (S, pi, oo, symbols, Rational, Integer,
                        GoldenRatio, EulerGamma, Catalan, Lambda, Dummy,
                        Eq, Ne, Le, Lt, Gt, Ge, Mod)
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
                             sign, floor)
from sympy.logic import ITE
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import MatrixSymbol, SparseMatrix, Matrix

from sympy.printing.codeprinter import rust_code

x, y, z = symbols('x,y,z', integer=False, real=True)
k, m, n = symbols('k,m,n', integer=True)


def test_Integer():
    assert rust_code(Integer(42)) == "42"
    assert rust_code(Integer(-56)) == "-56"


def test_Relational():
    assert rust_code(Eq(x, y)) == "x == y"
    assert rust_code(Ne(x, y)) == "x != y"
    assert rust_code(Le(x, y)) == "x <= y"
    assert rust_code(Lt(x, y)) == "x < y"
    assert rust_code(Gt(x, y)) == "x > y"
    assert rust_code(Ge(x, y)) == "x >= y"


def test_Rational():
    assert rust_code(Rational(3, 7)) == "3_f64/7.0"
    assert rust_code(Rational(18, 9)) == "2"
    assert rust_code(Rational(3, -7)) == "-3_f64/7.0"
    assert rust_code(Rational(-3, -7)) == "3_f64/7.0"
    assert rust_code(x + Rational(3, 7)) == "x + 3_f64/7.0"
    assert rust_code(Rational(3, 7)*x) == "(3_f64/7.0)*x"


def test_basic_ops():
    assert rust_code(x + y) == "x + y"
    assert rust_code(x - y) == "x - y"
    assert rust_code(x * y) == "x*y"
    assert rust_code(x / y) == "x*y.recip()"
    assert rust_code(-x) == "-x"
    assert rust_code(2 * x) == "2.0*x"
    assert rust_code(y + 2) == "y + 2.0"
    assert rust_code(x + n) == "n as f64 + x"

def test_printmethod():
    class fabs(Abs):
        def _rust_code(self, printer):
            return "%s.fabs()" % printer._print(self.args[0])
    assert rust_code(fabs(x)) == "x.fabs()"
    a = MatrixSymbol("a", 1, 3)
    assert rust_code(a[0,0]) == 'a[0]'


def test_Functions():
    assert rust_code(sin(x) ** cos(x)) == "x.sin().powf(x.cos())"
    assert rust_code(abs(x)) == "x.abs()"
    assert rust_code(ceiling(x)) == "x.ceil()"
    assert rust_code(floor(x)) == "x.floor()"

    # Automatic rewrite
    assert rust_code(Mod(x, 3)) == 'x - 3.0*((1_f64/3.0)*x).floor()'


def test_Pow():
    assert rust_code(1/x) == "x.recip()"
    assert rust_code(x**-1) == rust_code(x**-1.0) == "x.recip()"
    assert rust_code(sqrt(x)) == "x.sqrt()"
    assert rust_code(x**S.Half) == rust_code(x**0.5) == "x.sqrt()"

    assert rust_code(1/sqrt(x)) == "x.sqrt().recip()"
    assert rust_code(x**-S.Half) == rust_code(x**-0.5) == "x.sqrt().recip()"

    assert rust_code(1/pi) == "PI.recip()"
    assert rust_code(pi**-1) == rust_code(pi**-1.0) == "PI.recip()"
    assert rust_code(pi**-0.5) == "PI.sqrt().recip()"

    assert rust_code(x**Rational(1, 3)) == "x.cbrt()"
    assert rust_code(2**x) == "x.exp2()"
    assert rust_code(exp(x)) == "x.exp()"
    assert rust_code(x**3) == "x.powi(3)"
    assert rust_code(x**(y**3)) == "x.powf(y.powi(3))"
    assert rust_code(x**Rational(2, 3)) == "x.powf(2_f64/3.0)"

    g = implemented_function('g', Lambda(x, 2*x))
    assert rust_code(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "(3.5*2.0*x).powf(-x + y.powf(x))/(x.powi(2) + y)"
    _cond_cfunc = [(lambda base, exp: exp.is_integer, "dpowi", 1),
                   (lambda base, exp: not exp.is_integer, "pow", 1)]
    assert rust_code(x**3, user_functions={'Pow': _cond_cfunc}) == 'x.dpowi(3)'
    assert rust_code(x**3.2, user_functions={'Pow': _cond_cfunc}) == 'x.pow(3.2)'


def test_constants():
    assert rust_code(pi) == "PI"
    assert rust_code(oo) == "INFINITY"
    assert rust_code(S.Infinity) == "INFINITY"
    assert rust_code(-oo) == "NEG_INFINITY"
    assert rust_code(S.NegativeInfinity) == "NEG_INFINITY"
    assert rust_code(S.NaN) == "NAN"
    assert rust_code(exp(1)) == "E"
    assert rust_code(S.Exp1) == "E"


def test_constants_other():
    assert rust_code(2*GoldenRatio) == "const GoldenRatio: f64 = %s;\n2.0*GoldenRatio" % GoldenRatio.evalf(17)
    assert rust_code(
            2*Catalan) == "const Catalan: f64 = %s;\n2.0*Catalan" % Catalan.evalf(17)
    assert rust_code(2*EulerGamma) == "const EulerGamma: f64 = %s;\n2.0*EulerGamma" % EulerGamma.evalf(17)


def test_boolean():
    assert rust_code(True) == "true"
    assert rust_code(S.true) == "true"
    assert rust_code(False) == "false"
    assert rust_code(S.false) == "false"
    assert rust_code(k & m) == "k && m"
    assert rust_code(k | m) == "k || m"
    assert rust_code(~k) == "!k"
    assert rust_code(k & m & n) == "k && m && n"
    assert rust_code(k | m | n) == "k || m || n"
    assert rust_code((k & m) | n) == "n || k && m"
    assert rust_code((k | m) & n) == "n && (k || m)"


def test_Piecewise():
    expr = Piecewise((x, x < 1), (x + 2, True))
    assert rust_code(expr) == (
            "if (x < 1.0) {\n"
            "    x\n"
            "} else {\n"
            "    x + 2.0\n"
            "}")
    assert rust_code(expr, assign_to="r") == (
        "r = if (x < 1.0) {\n"
        "    x\n"
        "} else {\n"
        "    x + 2.0\n"
        "};")
    assert rust_code(expr, assign_to="r", inline=True) == (
        "r = if (x < 1.0) { x } else { x + 2.0 };")
    expr = Piecewise((x, x < 1), (x + 1, x < 5), (x + 2, True))
    assert rust_code(expr, inline=True) == (
        "if (x < 1.0) { x } else if (x < 5.0) { x + 1.0 } else { x + 2.0 }")
    assert rust_code(expr, assign_to="r", inline=True) == (
        "r = if (x < 1.0) { x } else if (x < 5.0) { x + 1.0 } else { x + 2.0 };")
    assert rust_code(expr, assign_to="r") == (
        "r = if (x < 1.0) {\n"
        "    x\n"
        "} else if (x < 5.0) {\n"
        "    x + 1.0\n"
        "} else {\n"
        "    x + 2.0\n"
        "};")
    expr = 2*Piecewise((x, x < 1), (x + 1, x < 5), (x + 2, True))
    assert rust_code(expr, inline=True) == (
        "2.0*if (x < 1.0) { x } else if (x < 5.0) { x + 1.0 } else { x + 2.0 }")
    expr = 2*Piecewise((x, x < 1), (x + 1, x < 5), (x + 2, True)) - 42
    assert rust_code(expr, inline=True) == (
        "2.0*if (x < 1.0) { x } else if (x < 5.0) { x + 1.0 } else { x + 2.0 } - 42.0")
    # Check that Piecewise without a True (default) condition error
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: rust_code(expr))


def test_dereference_printing():
    expr = x + y + sin(z) + z
    assert rust_code(expr, dereference=[z]) == "x + y + (*z) + (*z).sin()"


def test_sign():
    expr = sign(x) * y
    assert rust_code(expr) == "y*(if (x == 0.0) { 0.0 } else { (x).signum() }) as f64"
    assert rust_code(expr, assign_to='r') == "r = y*(if (x == 0.0) { 0.0 } else { (x).signum() }) as f64;"

    expr = sign(x + y) + 42
    assert rust_code(expr) == "(if (x + y == 0.0) { 0.0 } else { (x + y).signum() }) + 42"
    assert rust_code(expr, assign_to='r') == "r = (if (x + y == 0.0) { 0.0 } else { (x + y).signum() }) + 42;"

    expr = sign(cos(x))
    assert rust_code(expr) == "(if (x.cos() == 0.0) { 0.0 } else { (x.cos()).signum() })"


def test_reserved_words():

    x, y = symbols("x if")

    expr = sin(y)
    assert rust_code(expr) == "if_.sin()"
    assert rust_code(expr, dereference=[y]) == "(*if_).sin()"
    assert rust_code(expr, reserved_word_suffix='_unreserved') == "if_unreserved.sin()"

    with raises(ValueError):
        rust_code(expr, error_on_reserved=True)


def test_ITE():
    ekpr = ITE(k < 1, m, n)
    assert rust_code(ekpr) == (
            "if (k < 1) {\n"
            "    m\n"
            "} else {\n"
            "    n\n"
            "}")


def test_Indexed():
    n, m, o = symbols('n m o', integer=True)
    i, j, k = Idx('i', n), Idx('j', m), Idx('k', o)

    x = IndexedBase('x')[j]
    assert rust_code(x) == "x[j]"

    A = IndexedBase('A')[i, j]
    assert rust_code(A) == "A[m*i + j]"

    B = IndexedBase('B')[i, j, k]
    assert rust_code(B) == "B[m*o*i + o*j + k]"


def test_dummy_loops():
    i, m = symbols('i m', integer=True, cls=Dummy)
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx(i, m)

    assert rust_code(x[i], assign_to=y[i]) == (
        "for i in 0..m {\n"
        "    y[i] = x[i];\n"
        "}")


def test_loops():
    m, n = symbols('m n', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')
    i = Idx('i', m)
    j = Idx('j', n)

    assert rust_code(A[i, j]*x[j], assign_to=y[i]) == (
        "for i in 0..m {\n"
        "    y[i] = 0;\n"
        "}\n"
        "for i in 0..m {\n"
        "    for j in 0..n {\n"
        "        y[i] = A[n*i + j]*x[j] + y[i];\n"
        "    }\n"
        "}")

    assert rust_code(A[i, j]*x[j] + x[i] + z[i], assign_to=y[i]) == (
        "for i in 0..m {\n"
        "    y[i] = x[i] + z[i];\n"
        "}\n"
        "for i in 0..m {\n"
        "    for j in 0..n {\n"
        "        y[i] = A[n*i + j]*x[j] + y[i];\n"
        "    }\n"
        "}")


def test_loops_multiple_contractions():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    assert rust_code(b[j, k, l]*a[i, j, k, l], assign_to=y[i]) == (
        "for i in 0..m {\n"
        "    y[i] = 0;\n"
        "}\n"
        "for i in 0..m {\n"
        "    for j in 0..n {\n"
        "        for k in 0..o {\n"
        "            for l in 0..p {\n"
        "                y[i] = a[%s]*b[%s] + y[i];\n" % (i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\
        "            }\n"
        "        }\n"
        "    }\n"
        "}")


def test_loops_addfactor():
    m, n, o, p = symbols('m n o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    code = rust_code((a[i, j, k, l] + b[i, j, k, l])*c[j, k, l], assign_to=y[i])
    assert code == (
        "for i in 0..m {\n"
        "    y[i] = 0;\n"
        "}\n"
        "for i in 0..m {\n"
        "    for j in 0..n {\n"
        "        for k in 0..o {\n"
        "            for l in 0..p {\n"
        "                y[i] = (a[%s] + b[%s])*c[%s] + y[i];\n" % (i*n*o*p + j*o*p + k*p + l, i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\
        "            }\n"
        "        }\n"
        "    }\n"
        "}")


def test_settings():
    raises(TypeError, lambda: rust_code(sin(x), method="garbage"))


def test_inline_function():
    x = symbols('x')
    g = implemented_function('g', Lambda(x, 2*x))
    assert rust_code(g(x)) == "2*x"

    g = implemented_function('g', Lambda(x, 2*x/Catalan))
    assert rust_code(g(x)) == (
        "const Catalan: f64 = %s;\n2.0*x/Catalan" % Catalan.evalf(17))

    A = IndexedBase('A')
    i = Idx('i', symbols('n', integer=True))
    g = implemented_function('g', Lambda(x, x*(1 + x)*(2 + x)))
    assert rust_code(g(A[i]), assign_to=A[i]) == (
        "for i in 0..n {\n"
        "    A[i] = (A[i] + 1)*(A[i] + 2)*A[i];\n"
        "}")


def test_user_functions():
    x = symbols('x', integer=False)
    n = symbols('n', integer=True)
    custom_functions = {
        "ceiling": "ceil",
        "Abs": [(lambda x: not x.is_integer, "fabs", 4), (lambda x: x.is_integer, "abs", 4)],
    }
    assert rust_code(ceiling(x), user_functions=custom_functions) == "x.ceil()"
    assert rust_code(Abs(x), user_functions=custom_functions) == "fabs(x)"
    assert rust_code(Abs(n), user_functions=custom_functions) == "abs(n)"


def test_matrix():
    assert rust_code(Matrix([1, 2, 3])) == '[1, 2, 3]'
    with raises(ValueError):
        rust_code(Matrix([[1, 2, 3]]))


def test_sparse_matrix():
    # gh-15791
    with raises(NotImplementedError):
        rust_code(SparseMatrix([[1, 2, 3]]))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\tests\test_arrayop.py ===
import itertools
import random

from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import _af_invert
from sympy.testing.pytest import raises

from sympy.core.function import diff
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import (adjoint, conjugate, transpose)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.tensor.array import Array, ImmutableDenseNDimArray, ImmutableSparseNDimArray, MutableSparseNDimArray

from sympy.tensor.array.arrayop import tensorproduct, tensorcontraction, derive_by_array, permutedims, Flatten, \
    tensordiagonal


def test_import_NDimArray():
    from sympy.tensor.array import NDimArray
    del NDimArray


def test_tensorproduct():
    x,y,z,t = symbols('x y z t')
    from sympy.abc import a,b,c,d
    assert tensorproduct() == 1
    assert tensorproduct([x]) == Array([x])
    assert tensorproduct([x], [y]) == Array([[x*y]])
    assert tensorproduct([x], [y], [z]) == Array([[[x*y*z]]])
    assert tensorproduct([x], [y], [z], [t]) == Array([[[[x*y*z*t]]]])

    assert tensorproduct(x) == x
    assert tensorproduct(x, y) == x*y
    assert tensorproduct(x, y, z) == x*y*z
    assert tensorproduct(x, y, z, t) == x*y*z*t

    for ArrayType in [ImmutableDenseNDimArray, ImmutableSparseNDimArray]:
        A = ArrayType([x, y])
        B = ArrayType([1, 2, 3])
        C = ArrayType([a, b, c, d])

        assert tensorproduct(A, B, C) == ArrayType([[[a*x, b*x, c*x, d*x], [2*a*x, 2*b*x, 2*c*x, 2*d*x], [3*a*x, 3*b*x, 3*c*x, 3*d*x]],
                                                    [[a*y, b*y, c*y, d*y], [2*a*y, 2*b*y, 2*c*y, 2*d*y], [3*a*y, 3*b*y, 3*c*y, 3*d*y]]])

        assert tensorproduct([x, y], [1, 2, 3]) == tensorproduct(A, B)

        assert tensorproduct(A, 2) == ArrayType([2*x, 2*y])
        assert tensorproduct(A, [2]) == ArrayType([[2*x], [2*y]])
        assert tensorproduct([2], A) == ArrayType([[2*x, 2*y]])
        assert tensorproduct(a, A) == ArrayType([a*x, a*y])
        assert tensorproduct(a, A, B) == ArrayType([[a*x, 2*a*x, 3*a*x], [a*y, 2*a*y, 3*a*y]])
        assert tensorproduct(A, B, a) == ArrayType([[a*x, 2*a*x, 3*a*x], [a*y, 2*a*y, 3*a*y]])
        assert tensorproduct(B, a, A) == ArrayType([[a*x, a*y], [2*a*x, 2*a*y], [3*a*x, 3*a*y]])

    # tests for large scale sparse array
    for SparseArrayType in [ImmutableSparseNDimArray, MutableSparseNDimArray]:
        a = SparseArrayType({1:2, 3:4},(1000, 2000))
        b = SparseArrayType({1:2, 3:4},(1000, 2000))
        assert tensorproduct(a, b) == ImmutableSparseNDimArray({2000001: 4, 2000003: 8, 6000001: 8, 6000003: 16}, (1000, 2000, 1000, 2000))


def test_tensorcontraction():
    from sympy.abc import a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x
    B = Array(range(18), (2, 3, 3))
    assert tensorcontraction(B, (1, 2)) == Array([12, 39])
    C1 = Array([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x], (2, 3, 2, 2))

    assert tensorcontraction(C1, (0, 2)) == Array([[a + o, b + p], [e + s, f + t], [i + w, j + x]])
    assert tensorcontraction(C1, (0, 2, 3)) == Array([a + p, e + t, i + x])
    assert tensorcontraction(C1, (2, 3)) == Array([[a + d, e + h, i + l], [m + p, q + t, u + x]])


def test_derivative_by_array():
    from sympy.abc import i, j, t, x, y, z

    bexpr = x*y**2*exp(z)*log(t)
    sexpr = sin(bexpr)
    cexpr = cos(bexpr)

    a = Array([sexpr])

    assert derive_by_array(sexpr, t) == x*y**2*exp(z)*cos(x*y**2*exp(z)*log(t))/t
    assert derive_by_array(sexpr, [x, y, z]) == Array([bexpr/x*cexpr, 2*y*bexpr/y**2*cexpr, bexpr*cexpr])
    assert derive_by_array(a, [x, y, z]) == Array([[bexpr/x*cexpr], [2*y*bexpr/y**2*cexpr], [bexpr*cexpr]])

    assert derive_by_array(sexpr, [[x, y], [z, t]]) == Array([[bexpr/x*cexpr, 2*y*bexpr/y**2*cexpr], [bexpr*cexpr, bexpr/log(t)/t*cexpr]])
    assert derive_by_array(a, [[x, y], [z, t]]) == Array([[[bexpr/x*cexpr], [2*y*bexpr/y**2*cexpr]], [[bexpr*cexpr], [bexpr/log(t)/t*cexpr]]])
    assert derive_by_array([[x, y], [z, t]], [x, y]) == Array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]])
    assert derive_by_array([[x, y], [z, t]], [[x, y], [z, t]]) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
                                                                         [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])

    assert diff(sexpr, t) == x*y**2*exp(z)*cos(x*y**2*exp(z)*log(t))/t
    assert diff(sexpr, Array([x, y, z])) == Array([bexpr/x*cexpr, 2*y*bexpr/y**2*cexpr, bexpr*cexpr])
    assert diff(a, Array([x, y, z])) == Array([[bexpr/x*cexpr], [2*y*bexpr/y**2*cexpr], [bexpr*cexpr]])

    assert diff(sexpr, Array([[x, y], [z, t]])) == Array([[bexpr/x*cexpr, 2*y*bexpr/y**2*cexpr], [bexpr*cexpr, bexpr/log(t)/t*cexpr]])
    assert diff(a, Array([[x, y], [z, t]])) == Array([[[bexpr/x*cexpr], [2*y*bexpr/y**2*cexpr]], [[bexpr*cexpr], [bexpr/log(t)/t*cexpr]]])
    assert diff(Array([[x, y], [z, t]]), Array([x, y])) == Array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]])
    assert diff(Array([[x, y], [z, t]]), Array([[x, y], [z, t]])) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
                                                                         [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])

    # test for large scale sparse array
    for SparseArrayType in [ImmutableSparseNDimArray, MutableSparseNDimArray]:
        b = MutableSparseNDimArray({0:i, 1:j}, (10000, 20000))
        assert derive_by_array(b, i) == ImmutableSparseNDimArray({0: 1}, (10000, 20000))
        assert derive_by_array(b, (i, j)) == ImmutableSparseNDimArray({0: 1, 200000001: 1}, (2, 10000, 20000))

    #https://github.com/sympy/sympy/issues/20655
    U = Array([x, y, z])
    E = 2
    assert derive_by_array(E, U) ==  ImmutableDenseNDimArray([0, 0, 0])


def test_issue_emerged_while_discussing_10972():
    ua = Array([-1,0])
    Fa = Array([[0, 1], [-1, 0]])
    po = tensorproduct(Fa, ua, Fa, ua)
    assert tensorcontraction(po, (1, 2), (4, 5)) == Array([[0, 0], [0, 1]])

    sa = symbols('a0:144')
    po = Array(sa, [2, 2, 3, 3, 2, 2])
    assert tensorcontraction(po, (0, 1), (2, 3), (4, 5)) == sa[0] + sa[108] + sa[111] + sa[124] + sa[127] + sa[140] + sa[143] + sa[16] + sa[19] + sa[3] + sa[32] + sa[35]
    assert tensorcontraction(po, (0, 1, 4, 5), (2, 3)) == sa[0] + sa[111] + sa[127] + sa[143] + sa[16] + sa[32]
    assert tensorcontraction(po, (0, 1), (4, 5)) == Array([[sa[0] + sa[108] + sa[111] + sa[3], sa[112] + sa[115] + sa[4] + sa[7],
                                                             sa[11] + sa[116] + sa[119] + sa[8]], [sa[12] + sa[120] + sa[123] + sa[15],
                                                             sa[124] + sa[127] + sa[16] + sa[19], sa[128] + sa[131] + sa[20] + sa[23]],
                                                            [sa[132] + sa[135] + sa[24] + sa[27], sa[136] + sa[139] + sa[28] + sa[31],
                                                             sa[140] + sa[143] + sa[32] + sa[35]]])
    assert tensorcontraction(po, (0, 1), (2, 3)) == Array([[sa[0] + sa[108] + sa[124] + sa[140] + sa[16] + sa[32], sa[1] + sa[109] + sa[125] + sa[141] + sa[17] + sa[33]],
                                                           [sa[110] + sa[126] + sa[142] + sa[18] + sa[2] + sa[34], sa[111] + sa[127] + sa[143] + sa[19] + sa[3] + sa[35]]])


def test_array_permutedims():
    sa = symbols('a0:144')

    for ArrayType in [ImmutableDenseNDimArray, ImmutableSparseNDimArray]:
        m1 = ArrayType(sa[:6], (2, 3))
        assert permutedims(m1, (1, 0)) == transpose(m1)
        assert m1.tomatrix().T == permutedims(m1, (1, 0)).tomatrix()

        assert m1.tomatrix().T == transpose(m1).tomatrix()
        assert m1.tomatrix().C == conjugate(m1).tomatrix()
        assert m1.tomatrix().H == adjoint(m1).tomatrix()

        assert m1.tomatrix().T == m1.transpose().tomatrix()
        assert m1.tomatrix().C == m1.conjugate().tomatrix()
        assert m1.tomatrix().H == m1.adjoint().tomatrix()

        raises(ValueError, lambda: permutedims(m1, (0,)))
        raises(ValueError, lambda: permutedims(m1, (0, 0)))
        raises(ValueError, lambda: permutedims(m1, (1, 2, 0)))

        # Some tests with random arrays:
        dims = 6
        shape = [random.randint(1,5) for i in range(dims)]
        elems = [random.random() for i in range(tensorproduct(*shape))]
        ra = ArrayType(elems, shape)
        perm = list(range(dims))
        # Randomize the permutation:
        random.shuffle(perm)
        # Test inverse permutation:
        assert permutedims(permutedims(ra, perm), _af_invert(perm)) == ra
        # Test that permuted shape corresponds to action by `Permutation`:
        assert permutedims(ra, perm).shape == tuple(Permutation(perm)(shape))

        z = ArrayType.zeros(4,5,6,7)

        assert permutedims(z, (2, 3, 1, 0)).shape == (6, 7, 5, 4)
        assert permutedims(z, [2, 3, 1, 0]).shape == (6, 7, 5, 4)
        assert permutedims(z, Permutation([2, 3, 1, 0])).shape == (6, 7, 5, 4)

        po = ArrayType(sa, [2, 2, 3, 3, 2, 2])

        raises(ValueError, lambda: permutedims(po, (1, 1)))
        raises(ValueError, lambda: po.transpose())
        raises(ValueError, lambda: po.adjoint())

        assert permutedims(po, reversed(range(po.rank()))) == ArrayType(
            [[[[[[sa[0], sa[72]], [sa[36], sa[108]]], [[sa[12], sa[84]], [sa[48], sa[120]]], [[sa[24],
                                                                                               sa[96]], [sa[60], sa[132]]]],
               [[[sa[4], sa[76]], [sa[40], sa[112]]], [[sa[16],
                                                        sa[88]], [sa[52], sa[124]]],
                [[sa[28], sa[100]], [sa[64], sa[136]]]],
               [[[sa[8],
                  sa[80]], [sa[44], sa[116]]], [[sa[20], sa[92]], [sa[56], sa[128]]], [[sa[32],
                                                                                        sa[104]], [sa[68], sa[140]]]]],
              [[[[sa[2], sa[74]], [sa[38], sa[110]]], [[sa[14],
                                                        sa[86]], [sa[50], sa[122]]], [[sa[26], sa[98]], [sa[62], sa[134]]]],
               [[[sa[6],
                  sa[78]], [sa[42], sa[114]]], [[sa[18], sa[90]], [sa[54], sa[126]]], [[sa[30],
                                                                                        sa[102]], [sa[66], sa[138]]]],
               [[[sa[10], sa[82]], [sa[46], sa[118]]], [[sa[22],
                                                         sa[94]], [sa[58], sa[130]]],
                [[sa[34], sa[106]], [sa[70], sa[142]]]]]],
             [[[[[sa[1],
                  sa[73]], [sa[37], sa[109]]], [[sa[13], sa[85]], [sa[49], sa[121]]], [[sa[25],
                                                                                        sa[97]], [sa[61], sa[133]]]],
               [[[sa[5], sa[77]], [sa[41], sa[113]]], [[sa[17],
                                                        sa[89]], [sa[53], sa[125]]],
                [[sa[29], sa[101]], [sa[65], sa[137]]]],
               [[[sa[9],
                  sa[81]], [sa[45], sa[117]]], [[sa[21], sa[93]], [sa[57], sa[129]]], [[sa[33],
                                                                                        sa[105]], [sa[69], sa[141]]]]],
              [[[[sa[3], sa[75]], [sa[39], sa[111]]], [[sa[15],
                                                        sa[87]], [sa[51], sa[123]]], [[sa[27], sa[99]], [sa[63], sa[135]]]],
               [[[sa[7],
                  sa[79]], [sa[43], sa[115]]], [[sa[19], sa[91]], [sa[55], sa[127]]], [[sa[31],
                                                                                        sa[103]], [sa[67], sa[139]]]],
               [[[sa[11], sa[83]], [sa[47], sa[119]]], [[sa[23],
                                                         sa[95]], [sa[59], sa[131]]],
                [[sa[35], sa[107]], [sa[71], sa[143]]]]]]])

        assert permutedims(po, (1, 0, 2, 3, 4, 5)) == ArrayType(
            [[[[[[sa[0], sa[1]], [sa[2], sa[3]]], [[sa[4], sa[5]], [sa[6], sa[7]]], [[sa[8], sa[9]], [sa[10],
                                                                                                      sa[11]]]],
               [[[sa[12], sa[13]], [sa[14], sa[15]]], [[sa[16], sa[17]], [sa[18],
                                                                          sa[19]]], [[sa[20], sa[21]], [sa[22], sa[23]]]],
               [[[sa[24], sa[25]], [sa[26],
                                    sa[27]]], [[sa[28], sa[29]], [sa[30], sa[31]]], [[sa[32], sa[33]], [sa[34],
                                                                                                        sa[35]]]]],
              [[[[sa[72], sa[73]], [sa[74], sa[75]]], [[sa[76], sa[77]], [sa[78],
                                                                          sa[79]]], [[sa[80], sa[81]], [sa[82], sa[83]]]],
               [[[sa[84], sa[85]], [sa[86],
                                    sa[87]]], [[sa[88], sa[89]], [sa[90], sa[91]]], [[sa[92], sa[93]], [sa[94],
                                                                                                        sa[95]]]],
               [[[sa[96], sa[97]], [sa[98], sa[99]]], [[sa[100], sa[101]], [sa[102],
                                                                            sa[103]]],
                [[sa[104], sa[105]], [sa[106], sa[107]]]]]], [[[[[sa[36], sa[37]], [sa[38],
                                                                                    sa[39]]],
                                                                [[sa[40], sa[41]], [sa[42], sa[43]]],
                                                                [[sa[44], sa[45]], [sa[46],
                                                                                    sa[47]]]],
                                                               [[[sa[48], sa[49]], [sa[50], sa[51]]],
                                                                [[sa[52], sa[53]], [sa[54],
                                                                                    sa[55]]],
                                                                [[sa[56], sa[57]], [sa[58], sa[59]]]],
                                                               [[[sa[60], sa[61]], [sa[62],
                                                                                    sa[63]]],
                                                                [[sa[64], sa[65]], [sa[66], sa[67]]],
                                                                [[sa[68], sa[69]], [sa[70],
                                                                                    sa[71]]]]], [
                                                                  [[[sa[108], sa[109]], [sa[110], sa[111]]],
                                                                   [[sa[112], sa[113]], [sa[114],
                                                                                         sa[115]]],
                                                                   [[sa[116], sa[117]], [sa[118], sa[119]]]],
                                                                  [[[sa[120], sa[121]], [sa[122],
                                                                                         sa[123]]],
                                                                   [[sa[124], sa[125]], [sa[126], sa[127]]],
                                                                   [[sa[128], sa[129]], [sa[130],
                                                                                         sa[131]]]],
                                                                  [[[sa[132], sa[133]], [sa[134], sa[135]]],
                                                                   [[sa[136], sa[137]], [sa[138],
                                                                                         sa[139]]],
                                                                   [[sa[140], sa[141]], [sa[142], sa[143]]]]]]])

        assert permutedims(po, (0, 2, 1, 4, 3, 5)) == ArrayType(
            [[[[[[sa[0], sa[1]], [sa[4], sa[5]], [sa[8], sa[9]]], [[sa[2], sa[3]], [sa[6], sa[7]], [sa[10],
                                                                                                    sa[11]]]],
               [[[sa[36], sa[37]], [sa[40], sa[41]], [sa[44], sa[45]]], [[sa[38],
                                                                          sa[39]], [sa[42], sa[43]], [sa[46], sa[47]]]]],
              [[[[sa[12], sa[13]], [sa[16],
                                    sa[17]], [sa[20], sa[21]]], [[sa[14], sa[15]], [sa[18], sa[19]], [sa[22],
                                                                                                      sa[23]]]],
               [[[sa[48], sa[49]], [sa[52], sa[53]], [sa[56], sa[57]]], [[sa[50],
                                                                          sa[51]], [sa[54], sa[55]], [sa[58], sa[59]]]]],
              [[[[sa[24], sa[25]], [sa[28],
                                    sa[29]], [sa[32], sa[33]]], [[sa[26], sa[27]], [sa[30], sa[31]], [sa[34],
                                                                                                      sa[35]]]],
               [[[sa[60], sa[61]], [sa[64], sa[65]], [sa[68], sa[69]]], [[sa[62],
                                                                          sa[63]], [sa[66], sa[67]], [sa[70], sa[71]]]]]],
             [[[[[sa[72], sa[73]], [sa[76],
                                    sa[77]], [sa[80], sa[81]]], [[sa[74], sa[75]], [sa[78], sa[79]], [sa[82],
                                                                                                      sa[83]]]],
               [[[sa[108], sa[109]], [sa[112], sa[113]], [sa[116], sa[117]]], [[sa[110],
                                                                                sa[111]], [sa[114], sa[115]],
                                                                               [sa[118], sa[119]]]]],
              [[[[sa[84], sa[85]], [sa[88],
                                    sa[89]], [sa[92], sa[93]]], [[sa[86], sa[87]], [sa[90], sa[91]], [sa[94],
                                                                                                      sa[95]]]],
               [[[sa[120], sa[121]], [sa[124], sa[125]], [sa[128], sa[129]]], [[sa[122],
                                                                                sa[123]], [sa[126], sa[127]],
                                                                               [sa[130], sa[131]]]]],
              [[[[sa[96], sa[97]], [sa[100],
                                    sa[101]], [sa[104], sa[105]]], [[sa[98], sa[99]], [sa[102], sa[103]], [sa[106],
                                                                                                           sa[107]]]],
               [[[sa[132], sa[133]], [sa[136], sa[137]], [sa[140], sa[141]]], [[sa[134],
                                                                                sa[135]], [sa[138], sa[139]],
                                                                               [sa[142], sa[143]]]]]]])

        po2 = po.reshape(4, 9, 2, 2)
        assert po2 == ArrayType([[[[sa[0], sa[1]], [sa[2], sa[3]]], [[sa[4], sa[5]], [sa[6], sa[7]]], [[sa[8], sa[9]], [sa[10], sa[11]]], [[sa[12], sa[13]], [sa[14], sa[15]]], [[sa[16], sa[17]], [sa[18], sa[19]]], [[sa[20], sa[21]], [sa[22], sa[23]]], [[sa[24], sa[25]], [sa[26], sa[27]]], [[sa[28], sa[29]], [sa[30], sa[31]]], [[sa[32], sa[33]], [sa[34], sa[35]]]], [[[sa[36], sa[37]], [sa[38], sa[39]]], [[sa[40], sa[41]], [sa[42], sa[43]]], [[sa[44], sa[45]], [sa[46], sa[47]]], [[sa[48], sa[49]], [sa[50], sa[51]]], [[sa[52], sa[53]], [sa[54], sa[55]]], [[sa[56], sa[57]], [sa[58], sa[59]]], [[sa[60], sa[61]], [sa[62], sa[63]]], [[sa[64], sa[65]], [sa[66], sa[67]]], [[sa[68], sa[69]], [sa[70], sa[71]]]], [[[sa[72], sa[73]], [sa[74], sa[75]]], [[sa[76], sa[77]], [sa[78], sa[79]]], [[sa[80], sa[81]], [sa[82], sa[83]]], [[sa[84], sa[85]], [sa[86], sa[87]]], [[sa[88], sa[89]], [sa[90], sa[91]]], [[sa[92], sa[93]], [sa[94], sa[95]]], [[sa[96], sa[97]], [sa[98], sa[99]]], [[sa[100], sa[101]], [sa[102], sa[103]]], [[sa[104], sa[105]], [sa[106], sa[107]]]], [[[sa[108], sa[109]], [sa[110], sa[111]]], [[sa[112], sa[113]], [sa[114], sa[115]]], [[sa[116], sa[117]], [sa[118], sa[119]]], [[sa[120], sa[121]], [sa[122], sa[123]]], [[sa[124], sa[125]], [sa[126], sa[127]]], [[sa[128], sa[129]], [sa[130], sa[131]]], [[sa[132], sa[133]], [sa[134], sa[135]]], [[sa[136], sa[137]], [sa[138], sa[139]]], [[sa[140], sa[141]], [sa[142], sa[143]]]]])

        assert permutedims(po2, (3, 2, 0, 1)) == ArrayType([[[[sa[0], sa[4], sa[8], sa[12], sa[16], sa[20], sa[24], sa[28], sa[32]], [sa[36], sa[40], sa[44], sa[48], sa[52], sa[56], sa[60], sa[64], sa[68]], [sa[72], sa[76], sa[80], sa[84], sa[88], sa[92], sa[96], sa[100], sa[104]], [sa[108], sa[112], sa[116], sa[120], sa[124], sa[128], sa[132], sa[136], sa[140]]], [[sa[2], sa[6], sa[10], sa[14], sa[18], sa[22], sa[26], sa[30], sa[34]], [sa[38], sa[42], sa[46], sa[50], sa[54], sa[58], sa[62], sa[66], sa[70]], [sa[74], sa[78], sa[82], sa[86], sa[90], sa[94], sa[98], sa[102], sa[106]], [sa[110], sa[114], sa[118], sa[122], sa[126], sa[130], sa[134], sa[138], sa[142]]]], [[[sa[1], sa[5], sa[9], sa[13], sa[17], sa[21], sa[25], sa[29], sa[33]], [sa[37], sa[41], sa[45], sa[49], sa[53], sa[57], sa[61], sa[65], sa[69]], [sa[73], sa[77], sa[81], sa[85], sa[89], sa[93], sa[97], sa[101], sa[105]], [sa[109], sa[113], sa[117], sa[121], sa[125], sa[129], sa[133], sa[137], sa[141]]], [[sa[3], sa[7], sa[11], sa[15], sa[19], sa[23], sa[27], sa[31], sa[35]], [sa[39], sa[43], sa[47], sa[51], sa[55], sa[59], sa[63], sa[67], sa[71]], [sa[75], sa[79], sa[83], sa[87], sa[91], sa[95], sa[99], sa[103], sa[107]], [sa[111], sa[115], sa[119], sa[123], sa[127], sa[131], sa[135], sa[139], sa[143]]]]])

    # test for large scale sparse array
    for SparseArrayType in [ImmutableSparseNDimArray, MutableSparseNDimArray]:
        A = SparseArrayType({1:1, 10000:2}, (10000, 20000, 10000))
        assert permutedims(A, (0, 1, 2)) == A
        assert permutedims(A, (1, 0, 2)) == SparseArrayType({1: 1, 100000000: 2}, (20000, 10000, 10000))
        B = SparseArrayType({1:1, 20000:2}, (10000, 20000))
        assert B.transpose() == SparseArrayType({10000: 1, 1: 2}, (20000, 10000))


def test_permutedims_with_indices():
    A = Array(range(32)).reshape(2, 2, 2, 2, 2)
    indices_new = list("abcde")
    indices_old = list("ebdac")
    new_A = permutedims(A, index_order_new=indices_new, index_order_old=indices_old)
    for a, b, c, d, e in itertools.product(range(2), range(2), range(2), range(2), range(2)):
        assert new_A[a, b, c, d, e] == A[e, b, d, a, c]
    indices_old = list("cabed")
    new_A = permutedims(A, index_order_new=indices_new, index_order_old=indices_old)
    for a, b, c, d, e in itertools.product(range(2), range(2), range(2), range(2), range(2)):
        assert new_A[a, b, c, d, e] == A[c, a, b, e, d]
    raises(ValueError, lambda: permutedims(A, index_order_old=list("aacde"), index_order_new=list("abcde")))
    raises(ValueError, lambda: permutedims(A, index_order_old=list("abcde"), index_order_new=list("abcce")))
    raises(ValueError, lambda: permutedims(A, index_order_old=list("abcde"), index_order_new=list("abce")))
    raises(ValueError, lambda: permutedims(A, index_order_old=list("abce"), index_order_new=list("abce")))
    raises(ValueError, lambda: permutedims(A, [2, 1, 0, 3, 4], index_order_old=list("abcde")))
    raises(ValueError, lambda: permutedims(A, [2, 1, 0, 3, 4], index_order_new=list("abcde")))


def test_flatten():
    from sympy.matrices.dense import Matrix
    for ArrayType in [ImmutableDenseNDimArray, ImmutableSparseNDimArray, Matrix]:
        A = ArrayType(range(24)).reshape(4, 6)
        assert list(Flatten(A)) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

        for i, v in enumerate(Flatten(A)):
            assert i == v


def test_tensordiagonal():
    from sympy.matrices.dense import eye
    expr = Array(range(9)).reshape(3, 3)
    raises(ValueError, lambda: tensordiagonal(expr, [0], [1]))
    raises(ValueError, lambda: tensordiagonal(expr, [0, 0]))
    assert tensordiagonal(eye(3), [0, 1]) == Array([1, 1, 1])
    assert tensordiagonal(expr, [0, 1]) == Array([0, 4, 8])
    x, y, z = symbols("x y z")
    expr2 = tensorproduct([x, y, z], expr)
    assert tensordiagonal(expr2, [1, 2]) == Array([[0, 4*x, 8*x], [0, 4*y, 8*y], [0, 4*z, 8*z]])
    assert tensordiagonal(expr2, [0, 1]) == Array([[0, 3*y, 6*z], [x, 4*y, 7*z], [2*x, 5*y, 8*z]])
    assert tensordiagonal(expr2, [0, 1, 2]) == Array([0, 4*y, 8*z])
    # assert tensordiagonal(expr2, [0]) == permutedims(expr2, [1, 2, 0])
    # assert tensordiagonal(expr2, [1]) == permutedims(expr2, [0, 2, 1])
    # assert tensordiagonal(expr2, [2]) == expr2
    # assert tensordiagonal(expr2, [1], [2]) == expr2
    # assert tensordiagonal(expr2, [0], [1]) == permutedims(expr2, [2, 0, 1])

    a, b, c, X, Y, Z = symbols("a b c X Y Z")
    expr3 = tensorproduct([x, y, z], [1, 2, 3], [a, b, c], [X, Y, Z])
    assert tensordiagonal(expr3, [0, 1, 2, 3]) == Array([x*a*X, 2*y*b*Y, 3*z*c*Z])
    assert tensordiagonal(expr3, [0, 1], [2, 3]) == tensorproduct([x, 2*y, 3*z], [a*X, b*Y, c*Z])

    # assert tensordiagonal(expr3, [0], [1, 2], [3]) == tensorproduct([x, y, z], [a, 2*b, 3*c], [X, Y, Z])
    assert tensordiagonal(tensordiagonal(expr3, [2, 3]), [0, 1]) == tensorproduct([a*X, b*Y, c*Z], [x, 2*y, 3*z])

    raises(ValueError, lambda: tensordiagonal([[1, 2, 3], [4, 5, 6]], [0, 1]))
    raises(ValueError, lambda: tensordiagonal(expr3.reshape(3, 3, 9), [1, 2]))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\matrixlib\tests\test_interaction.py ===
"""Tests of interaction of matrix with other parts of numpy.

Note that tests with MaskedArray and linalg are done in separate files.
"""
import textwrap
import warnings

import pytest

import numpy as np
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
)


def test_fancy_indexing():
    # The matrix class messes with the shape. While this is always
    # weird (getitem is not used, it does not have setitem nor knows
    # about fancy indexing), this tests gh-3110
    # 2018-04-29: moved here from core.tests.test_index.
    m = np.matrix([[1, 2], [3, 4]])

    assert_(isinstance(m[[0, 1, 0], :], np.matrix))

    # gh-3110. Note the transpose currently because matrices do *not*
    # support dimension fixing for fancy indexing correctly.
    x = np.asmatrix(np.arange(50).reshape(5, 10))
    assert_equal(x[:2, np.array(-1)], x[:2, -1].T)


def test_polynomial_mapdomain():
    # test that polynomial preserved matrix subtype.
    # 2018-04-29: moved here from polynomial.tests.polyutils.
    dom1 = [0, 4]
    dom2 = [1, 3]
    x = np.matrix([dom1, dom1])
    res = np.polynomial.polyutils.mapdomain(x, dom1, dom2)
    assert_(isinstance(res, np.matrix))


def test_sort_matrix_none():
    # 2018-04-29: moved here from core.tests.test_multiarray
    a = np.matrix([[2, 1, 0]])
    actual = np.sort(a, axis=None)
    expected = np.matrix([[0, 1, 2]])
    assert_equal(actual, expected)
    assert_(type(expected) is np.matrix)


def test_partition_matrix_none():
    # gh-4301
    # 2018-04-29: moved here from core.tests.test_multiarray
    a = np.matrix([[2, 1, 0]])
    actual = np.partition(a, 1, axis=None)
    expected = np.matrix([[0, 1, 2]])
    assert_equal(actual, expected)
    assert_(type(expected) is np.matrix)


def test_dot_scalar_and_matrix_of_objects():
    # Ticket #2469
    # 2018-04-29: moved here from core.tests.test_multiarray
    arr = np.matrix([1, 2], dtype=object)
    desired = np.matrix([[3, 6]], dtype=object)
    assert_equal(np.dot(arr, 3), desired)
    assert_equal(np.dot(3, arr), desired)


def test_inner_scalar_and_matrix():
    # 2018-04-29: moved here from core.tests.test_multiarray
    for dt in np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?':
        sca = np.array(3, dtype=dt)[()]
        arr = np.matrix([[1, 2], [3, 4]], dtype=dt)
        desired = np.matrix([[3, 6], [9, 12]], dtype=dt)
        assert_equal(np.inner(arr, sca), desired)
        assert_equal(np.inner(sca, arr), desired)


def test_inner_scalar_and_matrix_of_objects():
    # Ticket #4482
    # 2018-04-29: moved here from core.tests.test_multiarray
    arr = np.matrix([1, 2], dtype=object)
    desired = np.matrix([[3, 6]], dtype=object)
    assert_equal(np.inner(arr, 3), desired)
    assert_equal(np.inner(3, arr), desired)


def test_iter_allocate_output_subtype():
    # Make sure that the subtype with priority wins
    # 2018-04-29: moved here from core.tests.test_nditer, given the
    # matrix specific shape test.

    # matrix vs ndarray
    a = np.matrix([[1, 2], [3, 4]])
    b = np.arange(4).reshape(2, 2).T
    i = np.nditer([a, b, None], [],
                  [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    assert_(type(i.operands[2]) is np.matrix)
    assert_(type(i.operands[2]) is not np.ndarray)
    assert_equal(i.operands[2].shape, (2, 2))

    # matrix always wants things to be 2D
    b = np.arange(4).reshape(1, 2, 2)
    assert_raises(RuntimeError, np.nditer, [a, b, None], [],
                  [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    # but if subtypes are disabled, the result can still work
    i = np.nditer([a, b, None], [],
                  [['readonly'], ['readonly'],
                   ['writeonly', 'allocate', 'no_subtype']])
    assert_(type(i.operands[2]) is np.ndarray)
    assert_(type(i.operands[2]) is not np.matrix)
    assert_equal(i.operands[2].shape, (1, 2, 2))


def like_function():
    # 2018-04-29: moved here from core.tests.test_numeric
    a = np.matrix([[1, 2], [3, 4]])
    for like_function in np.zeros_like, np.ones_like, np.empty_like:
        b = like_function(a)
        assert_(type(b) is np.matrix)

        c = like_function(a, subok=False)
        assert_(type(c) is not np.matrix)


def test_array_astype():
    # 2018-04-29: copied here from core.tests.test_api
    # subok=True passes through a matrix
    a = np.matrix([[0, 1, 2], [3, 4, 5]], dtype='f4')
    b = a.astype('f4', subok=True, copy=False)
    assert_(a is b)

    # subok=True is default, and creates a subtype on a cast
    b = a.astype('i4', copy=False)
    assert_equal(a, b)
    assert_equal(type(b), np.matrix)

    # subok=False never returns a matrix
    b = a.astype('f4', subok=False, copy=False)
    assert_equal(a, b)
    assert_(not (a is b))
    assert_(type(b) is not np.matrix)


def test_stack():
    # 2018-04-29: copied here from core.tests.test_shape_base
    # check np.matrix cannot be stacked
    m = np.matrix([[1, 2], [3, 4]])
    assert_raises_regex(ValueError, 'shape too large to be a matrix',
                        np.stack, [m, m])


def test_object_scalar_multiply():
    # Tickets #2469 and #4482
    # 2018-04-29: moved here from core.tests.test_ufunc
    arr = np.matrix([1, 2], dtype=object)
    desired = np.matrix([[3, 6]], dtype=object)
    assert_equal(np.multiply(arr, 3), desired)
    assert_equal(np.multiply(3, arr), desired)


def test_nanfunctions_matrices():
    # Check that it works and that type and
    # shape are preserved
    # 2018-04-29: moved here from core.tests.test_nanfunctions
    mat = np.matrix(np.eye(3))
    for f in [np.nanmin, np.nanmax]:
        res = f(mat, axis=0)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (1, 3))
        res = f(mat, axis=1)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (3, 1))
        res = f(mat)
        assert_(np.isscalar(res))
    # check that rows of nan are dealt with for subclasses (#4628)
    mat[1] = np.nan
    for f in [np.nanmin, np.nanmax]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            res = f(mat, axis=0)
            assert_(isinstance(res, np.matrix))
            assert_(not np.any(np.isnan(res)))
            assert_(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            res = f(mat, axis=1)
            assert_(isinstance(res, np.matrix))
            assert_(np.isnan(res[1, 0]) and not np.isnan(res[0, 0])
                    and not np.isnan(res[2, 0]))
            assert_(len(w) == 1, 'no warning raised')
            assert_(issubclass(w[0].category, RuntimeWarning))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            res = f(mat)
            assert_(np.isscalar(res))
            assert_(res != np.nan)
            assert_(len(w) == 0)


def test_nanfunctions_matrices_general():
    # Check that it works and that type and
    # shape are preserved
    # 2018-04-29: moved here from core.tests.test_nanfunctions
    mat = np.matrix(np.eye(3))
    for f in (np.nanargmin, np.nanargmax, np.nansum, np.nanprod,
              np.nanmean, np.nanvar, np.nanstd):
        res = f(mat, axis=0)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (1, 3))
        res = f(mat, axis=1)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (3, 1))
        res = f(mat)
        assert_(np.isscalar(res))

    for f in np.nancumsum, np.nancumprod:
        res = f(mat, axis=0)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (3, 3))
        res = f(mat, axis=1)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (3, 3))
        res = f(mat)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (1, 3 * 3))


def test_average_matrix():
    # 2018-04-29: moved here from core.tests.test_function_base.
    y = np.matrix(np.random.rand(5, 5))
    assert_array_equal(y.mean(0), np.average(y, 0))

    a = np.matrix([[1, 2], [3, 4]])
    w = np.matrix([[1, 2], [3, 4]])

    r = np.average(a, axis=0, weights=w)
    assert_equal(type(r), np.matrix)
    assert_equal(r, [[2.5, 10.0 / 3]])


def test_dot_matrix():
    # Test to make sure matrices give the same answer as ndarrays
    # 2018-04-29: moved here from core.tests.test_function_base.
    x = np.linspace(0, 5)
    y = np.linspace(-5, 0)
    mx = np.matrix(x)
    my = np.matrix(y)
    r = np.dot(x, y)
    mr = np.dot(mx, my.T)
    assert_almost_equal(mr, r)


def test_ediff1d_matrix():
    # 2018-04-29: moved here from core.tests.test_arraysetops.
    assert isinstance(np.ediff1d(np.matrix(1)), np.matrix)
    assert isinstance(np.ediff1d(np.matrix(1), to_begin=1), np.matrix)


def test_apply_along_axis_matrix():
    # this test is particularly malicious because matrix
    # refuses to become 1d
    # 2018-04-29: moved here from core.tests.test_shape_base.
    def double(row):
        return row * 2

    m = np.matrix([[0, 1], [2, 3]])
    expected = np.matrix([[0, 2], [4, 6]])

    result = np.apply_along_axis(double, 0, m)
    assert_(isinstance(result, np.matrix))
    assert_array_equal(result, expected)

    result = np.apply_along_axis(double, 1, m)
    assert_(isinstance(result, np.matrix))
    assert_array_equal(result, expected)


def test_kron_matrix():
    # 2018-04-29: moved here from core.tests.test_shape_base.
    a = np.ones([2, 2])
    m = np.asmatrix(a)
    assert_equal(type(np.kron(a, a)), np.ndarray)
    assert_equal(type(np.kron(m, m)), np.matrix)
    assert_equal(type(np.kron(a, m)), np.matrix)
    assert_equal(type(np.kron(m, a)), np.matrix)


class TestConcatenatorMatrix:
    # 2018-04-29: moved here from core.tests.test_index_tricks.
    def test_matrix(self):
        a = [1, 2]
        b = [3, 4]

        ab_r = np.r_['r', a, b]
        ab_c = np.r_['c', a, b]

        assert_equal(type(ab_r), np.matrix)
        assert_equal(type(ab_c), np.matrix)

        assert_equal(np.array(ab_r), [[1, 2, 3, 4]])
        assert_equal(np.array(ab_c), [[1], [2], [3], [4]])

        assert_raises(ValueError, lambda: np.r_['rc', a, b])

    def test_matrix_scalar(self):
        r = np.r_['r', [1, 2], 3]
        assert_equal(type(r), np.matrix)
        assert_equal(np.array(r), [[1, 2, 3]])

    def test_matrix_builder(self):
        a = np.array([1])
        b = np.array([2])
        c = np.array([3])
        d = np.array([4])
        actual = np.r_['a, b; c, d']
        expected = np.bmat([[a, b], [c, d]])

        assert_equal(actual, expected)
        assert_equal(type(actual), type(expected))


def test_array_equal_error_message_matrix():
    # 2018-04-29: moved here from testing.tests.test_utils.
    with pytest.raises(AssertionError) as exc_info:
        assert_equal(np.array([1, 2]), np.matrix([1, 2]))
    msg = str(exc_info.value)
    msg_reference = textwrap.dedent("""\

    Arrays are not equal

    (shapes (2,), (1, 2) mismatch)
     ACTUAL: array([1, 2])
     DESIRED: matrix([[1, 2]])""")
    assert_equal(msg, msg_reference)


def test_array_almost_equal_matrix():
    # Matrix slicing keeps things 2-D, while array does not necessarily.
    # See gh-8452.
    # 2018-04-29: moved here from testing.tests.test_utils.
    m1 = np.matrix([[1., 2.]])
    m2 = np.matrix([[1., np.nan]])
    m3 = np.matrix([[1., -np.inf]])
    m4 = np.matrix([[np.nan, np.inf]])
    m5 = np.matrix([[1., 2.], [np.nan, np.inf]])
    for assert_func in assert_array_almost_equal, assert_almost_equal:
        for m in m1, m2, m3, m4, m5:
            assert_func(m, m)
            a = np.array(m)
            assert_func(a, m)
            assert_func(m, a)