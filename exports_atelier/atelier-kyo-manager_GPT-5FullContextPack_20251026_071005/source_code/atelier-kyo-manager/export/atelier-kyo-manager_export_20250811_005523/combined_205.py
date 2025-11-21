
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\test_pickle.py ===
"""
manage legacy pickle tests

How to add pickle tests:

1. Install pandas version intended to output the pickle.

2. Execute "generate_legacy_storage_files.py" to create the pickle.
$ python generate_legacy_storage_files.py <output_dir> pickle

3. Move the created pickle to "data/legacy_pickle/<version>" directory.
"""
from __future__ import annotations

from array import array
import bz2
import datetime
import functools
from functools import partial
import gzip
import io
import os
from pathlib import Path
import pickle
import shutil
import tarfile
from typing import Any
import uuid
import zipfile

import numpy as np
import pytest

from pandas.compat import (
    get_lzma_file,
    is_platform_little_endian,
)
from pandas.compat._optional import import_optional_dependency
from pandas.compat.compressors import flatten_buffer
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    period_range,
)
import pandas._testing as tm
from pandas.tests.io.generate_legacy_storage_files import create_pickle_data

import pandas.io.common as icom
from pandas.tseries.offsets import (
    Day,
    MonthEnd,
)


# ---------------------
# comparison functions
# ---------------------
def compare_element(result, expected, typ):
    if isinstance(expected, Index):
        tm.assert_index_equal(expected, result)
        return

    if typ.startswith("sp_"):
        tm.assert_equal(result, expected)
    elif typ == "timestamp":
        if expected is pd.NaT:
            assert result is pd.NaT
        else:
            assert result == expected
    else:
        comparator = getattr(tm, f"assert_{typ}_equal", tm.assert_almost_equal)
        comparator(result, expected)


# ---------------------
# tests
# ---------------------


@pytest.mark.parametrize(
    "data",
    [
        b"123",
        b"123456",
        bytearray(b"123"),
        memoryview(b"123"),
        pickle.PickleBuffer(b"123"),
        array("I", [1, 2, 3]),
        memoryview(b"123456").cast("B", (3, 2)),
        memoryview(b"123456").cast("B", (3, 2))[::2],
        np.arange(12).reshape((3, 4), order="C"),
        np.arange(12).reshape((3, 4), order="F"),
        np.arange(12).reshape((3, 4), order="C")[:, ::2],
    ],
)
def test_flatten_buffer(data):
    result = flatten_buffer(data)
    expected = memoryview(data).tobytes("A")
    assert result == expected
    if isinstance(data, (bytes, bytearray)):
        assert result is data
    elif isinstance(result, memoryview):
        assert result.ndim == 1
        assert result.format == "B"
        assert result.contiguous
        assert result.shape == (result.nbytes,)


def test_pickles(datapath):
    if not is_platform_little_endian():
        pytest.skip("known failure on non-little endian")

    # For loop for compat with --strict-data-files
    for legacy_pickle in Path(__file__).parent.glob("data/legacy_pickle/*/*.p*kl*"):
        legacy_pickle = datapath(legacy_pickle)

        data = pd.read_pickle(legacy_pickle)

        for typ, dv in data.items():
            for dt, result in dv.items():
                expected = data[typ][dt]

                if typ == "series" and dt == "ts":
                    # GH 7748
                    tm.assert_series_equal(result, expected)
                    assert result.index.freq == expected.index.freq
                    assert not result.index.freq.normalize
                    tm.assert_series_equal(result > 0, expected > 0)

                    # GH 9291
                    freq = result.index.freq
                    assert freq + Day(1) == Day(2)

                    res = freq + pd.Timedelta(hours=1)
                    assert isinstance(res, pd.Timedelta)
                    assert res == pd.Timedelta(days=1, hours=1)

                    res = freq + pd.Timedelta(nanoseconds=1)
                    assert isinstance(res, pd.Timedelta)
                    assert res == pd.Timedelta(days=1, nanoseconds=1)
                elif typ == "index" and dt == "period":
                    tm.assert_index_equal(result, expected)
                    assert isinstance(result.freq, MonthEnd)
                    assert result.freq == MonthEnd()
                    assert result.freqstr == "M"
                    tm.assert_index_equal(result.shift(2), expected.shift(2))
                elif typ == "series" and dt in ("dt_tz", "cat"):
                    tm.assert_series_equal(result, expected)
                elif typ == "frame" and dt in (
                    "dt_mixed_tzs",
                    "cat_onecol",
                    "cat_and_float",
                ):
                    tm.assert_frame_equal(result, expected)
                else:
                    compare_element(result, expected, typ)


def python_pickler(obj, path):
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=-1)


def python_unpickler(path):
    with open(path, "rb") as fh:
        fh.seek(0)
        return pickle.load(fh)


def flatten(data: dict) -> list[tuple[str, Any]]:
    """Flatten create_pickle_data"""
    return [
        (typ, example)
        for typ, examples in data.items()
        for example in examples.values()
    ]


@pytest.mark.parametrize(
    "pickle_writer",
    [
        pytest.param(python_pickler, id="python"),
        pytest.param(pd.to_pickle, id="pandas_proto_default"),
        pytest.param(
            functools.partial(pd.to_pickle, protocol=pickle.HIGHEST_PROTOCOL),
            id="pandas_proto_highest",
        ),
        pytest.param(functools.partial(pd.to_pickle, protocol=4), id="pandas_proto_4"),
        pytest.param(
            functools.partial(pd.to_pickle, protocol=5),
            id="pandas_proto_5",
        ),
    ],
)
@pytest.mark.parametrize("writer", [pd.to_pickle, python_pickler])
@pytest.mark.parametrize("typ, expected", flatten(create_pickle_data()))
def test_round_trip_current(typ, expected, pickle_writer, writer):
    with tm.ensure_clean() as path:
        # test writing with each pickler
        pickle_writer(expected, path)

        # test reading with each unpickler
        result = pd.read_pickle(path)
        compare_element(result, expected, typ)

        result = python_unpickler(path)
        compare_element(result, expected, typ)

        # and the same for file objects (GH 35679)
        with open(path, mode="wb") as handle:
            writer(expected, path)
            handle.seek(0)  # shouldn't close file handle
        with open(path, mode="rb") as handle:
            result = pd.read_pickle(handle)
            handle.seek(0)  # shouldn't close file handle
        compare_element(result, expected, typ)


def test_pickle_path_pathlib():
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    result = tm.round_trip_pathlib(df.to_pickle, pd.read_pickle)
    tm.assert_frame_equal(df, result)


def test_pickle_path_localpath():
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    result = tm.round_trip_localpath(df.to_pickle, pd.read_pickle)
    tm.assert_frame_equal(df, result)


# ---------------------
# test pickle compression
# ---------------------


@pytest.fixture
def get_random_path():
    return f"__{uuid.uuid4()}__.pickle"


class TestCompression:
    _extension_to_compression = icom.extension_to_compression

    def compress_file(self, src_path, dest_path, compression):
        if compression is None:
            shutil.copyfile(src_path, dest_path)
            return

        if compression == "gzip":
            f = gzip.open(dest_path, "w")
        elif compression == "bz2":
            f = bz2.BZ2File(dest_path, "w")
        elif compression == "zip":
            with zipfile.ZipFile(dest_path, "w", compression=zipfile.ZIP_DEFLATED) as f:
                f.write(src_path, os.path.basename(src_path))
        elif compression == "tar":
            with open(src_path, "rb") as fh:
                with tarfile.open(dest_path, mode="w") as tar:
                    tarinfo = tar.gettarinfo(src_path, os.path.basename(src_path))
                    tar.addfile(tarinfo, fh)
        elif compression == "xz":
            f = get_lzma_file()(dest_path, "w")
        elif compression == "zstd":
            f = import_optional_dependency("zstandard").open(dest_path, "wb")
        else:
            msg = f"Unrecognized compression type: {compression}"
            raise ValueError(msg)

        if compression not in ["zip", "tar"]:
            with open(src_path, "rb") as fh:
                with f:
                    f.write(fh.read())

    def test_write_explicit(self, compression, get_random_path):
        base = get_random_path
        path1 = base + ".compressed"
        path2 = base + ".raw"

        with tm.ensure_clean(path1) as p1, tm.ensure_clean(path2) as p2:
            df = DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=Index(list("ABCD"), dtype=object),
                index=Index([f"i-{i}" for i in range(30)], dtype=object),
            )

            # write to compressed file
            df.to_pickle(p1, compression=compression)

            # decompress
            with tm.decompress_file(p1, compression=compression) as f:
                with open(p2, "wb") as fh:
                    fh.write(f.read())

            # read decompressed file
            df2 = pd.read_pickle(p2, compression=None)

            tm.assert_frame_equal(df, df2)

    @pytest.mark.parametrize("compression", ["", "None", "bad", "7z"])
    def test_write_explicit_bad(self, compression, get_random_path):
        with pytest.raises(ValueError, match="Unrecognized compression type"):
            with tm.ensure_clean(get_random_path) as path:
                df = DataFrame(
                    1.1 * np.arange(120).reshape((30, 4)),
                    columns=Index(list("ABCD"), dtype=object),
                    index=Index([f"i-{i}" for i in range(30)], dtype=object),
                )
                df.to_pickle(path, compression=compression)

    def test_write_infer(self, compression_ext, get_random_path):
        base = get_random_path
        path1 = base + compression_ext
        path2 = base + ".raw"
        compression = self._extension_to_compression.get(compression_ext.lower())

        with tm.ensure_clean(path1) as p1, tm.ensure_clean(path2) as p2:
            df = DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=Index(list("ABCD"), dtype=object),
                index=Index([f"i-{i}" for i in range(30)], dtype=object),
            )

            # write to compressed file by inferred compression method
            df.to_pickle(p1)

            # decompress
            with tm.decompress_file(p1, compression=compression) as f:
                with open(p2, "wb") as fh:
                    fh.write(f.read())

            # read decompressed file
            df2 = pd.read_pickle(p2, compression=None)

            tm.assert_frame_equal(df, df2)

    def test_read_explicit(self, compression, get_random_path):
        base = get_random_path
        path1 = base + ".raw"
        path2 = base + ".compressed"

        with tm.ensure_clean(path1) as p1, tm.ensure_clean(path2) as p2:
            df = DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=Index(list("ABCD"), dtype=object),
                index=Index([f"i-{i}" for i in range(30)], dtype=object),
            )

            # write to uncompressed file
            df.to_pickle(p1, compression=None)

            # compress
            self.compress_file(p1, p2, compression=compression)

            # read compressed file
            df2 = pd.read_pickle(p2, compression=compression)
            tm.assert_frame_equal(df, df2)

    def test_read_infer(self, compression_ext, get_random_path):
        base = get_random_path
        path1 = base + ".raw"
        path2 = base + compression_ext
        compression = self._extension_to_compression.get(compression_ext.lower())

        with tm.ensure_clean(path1) as p1, tm.ensure_clean(path2) as p2:
            df = DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=Index(list("ABCD"), dtype=object),
                index=Index([f"i-{i}" for i in range(30)], dtype=object),
            )

            # write to uncompressed file
            df.to_pickle(p1, compression=None)

            # compress
            self.compress_file(p1, p2, compression=compression)

            # read compressed file by inferred compression method
            df2 = pd.read_pickle(p2)
            tm.assert_frame_equal(df, df2)


# ---------------------
# test pickle compression
# ---------------------


class TestProtocol:
    @pytest.mark.parametrize("protocol", [-1, 0, 1, 2])
    def test_read(self, protocol, get_random_path):
        with tm.ensure_clean(get_random_path) as path:
            df = DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=Index(list("ABCD"), dtype=object),
                index=Index([f"i-{i}" for i in range(30)], dtype=object),
            )
            df.to_pickle(path, protocol=protocol)
            df2 = pd.read_pickle(path)
            tm.assert_frame_equal(df, df2)


@pytest.mark.parametrize(
    ["pickle_file", "excols"],
    [
        ("test_py27.pkl", Index(["a", "b", "c"], dtype=object)),
        (
            "test_mi_py27.pkl",
            pd.MultiIndex(
                [
                    Index(["a", "b", "c"], dtype=object),
                    Index(["A", "B", "C"], dtype=object),
                ],
                [np.array([0, 1, 2]), np.array([0, 1, 2])],
            ),
        ),
    ],
)
def test_unicode_decode_error(datapath, pickle_file, excols):
    # pickle file written with py27, should be readable without raising
    #  UnicodeDecodeError, see GH#28645 and GH#31988
    path = datapath("io", "data", "pickle", pickle_file)
    df = pd.read_pickle(path)

    # just test the columns are correct since the values are random
    tm.assert_index_equal(df.columns, excols)


# ---------------------
# tests for buffer I/O
# ---------------------


def test_pickle_buffer_roundtrip():
    with tm.ensure_clean() as path:
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        with open(path, "wb") as fh:
            df.to_pickle(fh)
        with open(path, "rb") as fh:
            result = pd.read_pickle(fh)
        tm.assert_frame_equal(df, result)


# ---------------------
# tests for URL I/O
# ---------------------


@pytest.mark.parametrize(
    "mockurl", ["http://url.com", "ftp://test.com", "http://gzip.com"]
)
def test_pickle_generalurl_read(monkeypatch, mockurl):
    def python_pickler(obj, path):
        with open(path, "wb") as fh:
            pickle.dump(obj, fh, protocol=-1)

    class MockReadResponse:
        def __init__(self, path) -> None:
            self.file = open(path, "rb")
            if "gzip" in path:
                self.headers = {"Content-Encoding": "gzip"}
            else:
                self.headers = {"Content-Encoding": ""}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def read(self):
            return self.file.read()

        def close(self):
            return self.file.close()

    with tm.ensure_clean() as path:

        def mock_urlopen_read(*args, **kwargs):
            return MockReadResponse(path)

        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        python_pickler(df, path)
        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen_read)
        result = pd.read_pickle(mockurl)
        tm.assert_frame_equal(df, result)


def test_pickle_fsspec_roundtrip():
    pytest.importorskip("fsspec")
    with tm.ensure_clean():
        mockurl = "memory://mockfile"
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        df.to_pickle(mockurl)
        result = pd.read_pickle(mockurl)
        tm.assert_frame_equal(df, result)


class MyTz(datetime.tzinfo):
    def __init__(self) -> None:
        pass


def test_read_pickle_with_subclass():
    # GH 12163
    expected = Series(dtype=object), MyTz()
    result = tm.round_trip_pickle(expected)

    tm.assert_series_equal(result[0], expected[0])
    assert isinstance(result[1], MyTz)


def test_pickle_binary_object_compression(compression):
    """
    Read/write from binary file-objects w/wo compression.

    GH 26237, GH 29054, and GH 29570
    """
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )

    # reference for compression
    with tm.ensure_clean() as path:
        df.to_pickle(path, compression=compression)
        reference = Path(path).read_bytes()

    # write
    buffer = io.BytesIO()
    df.to_pickle(buffer, compression=compression)
    buffer.seek(0)

    # gzip  and zip safe the filename: cannot compare the compressed content
    assert buffer.getvalue() == reference or compression in ("gzip", "zip", "tar")

    # read
    read_df = pd.read_pickle(buffer, compression=compression)
    buffer.seek(0)
    tm.assert_frame_equal(df, read_df)


def test_pickle_dataframe_with_multilevel_index(
    multiindex_year_month_day_dataframe_random_data,
    multiindex_dataframe_random_data,
):
    ymd = multiindex_year_month_day_dataframe_random_data
    frame = multiindex_dataframe_random_data

    def _test_roundtrip(frame):
        unpickled = tm.round_trip_pickle(frame)
        tm.assert_frame_equal(frame, unpickled)

    _test_roundtrip(frame)
    _test_roundtrip(frame.T)
    _test_roundtrip(ymd)
    _test_roundtrip(ymd.T)


def test_pickle_timeseries_periodindex():
    # GH#2891
    prng = period_range("1/1/2011", "1/1/2012", freq="M")
    ts = Series(np.random.default_rng(2).standard_normal(len(prng)), prng)
    new_ts = tm.round_trip_pickle(ts)
    assert new_ts.index.freqstr == "M"


@pytest.mark.parametrize(
    "name", [777, 777.0, "name", datetime.datetime(2001, 11, 11), (1, 2)]
)
def test_pickle_preserve_name(name):
    unpickled = tm.round_trip_pickle(Series(np.arange(10, dtype=np.float64), name=name))
    assert unpickled.name == name


def test_pickle_datetimes(datetime_series):
    unp_ts = tm.round_trip_pickle(datetime_series)
    tm.assert_series_equal(unp_ts, datetime_series)


def test_pickle_strings(string_series):
    unp_series = tm.round_trip_pickle(string_series)
    tm.assert_series_equal(unp_series, string_series)


@td.skip_array_manager_invalid_test
def test_pickle_preserves_block_ndim():
    # GH#37631
    ser = Series(list("abc")).astype("category").iloc[[0]]
    res = tm.round_trip_pickle(ser)

    assert res._mgr.blocks[0].ndim == 1
    assert res._mgr.blocks[0].shape == (1,)

    # GH#37631 OP issue was about indexing, underlying problem was pickle
    tm.assert_series_equal(res[[True]], ser)


@pytest.mark.parametrize("protocol", [pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL])
def test_pickle_big_dataframe_compression(protocol, compression):
    # GH#39002
    df = DataFrame(range(100000))
    result = tm.round_trip_pathlib(
        partial(df.to_pickle, protocol=protocol, compression=compression),
        partial(pd.read_pickle, compression=compression),
    )
    tm.assert_frame_equal(df, result)


def test_pickle_frame_v124_unpickle_130(datapath):
    # GH#42345 DataFrame created in 1.2.x, unpickle in 1.3.x
    path = datapath(
        Path(__file__).parent,
        "data",
        "legacy_pickle",
        "1.2.4",
        "empty_frame_v1_2_4-GH#42345.pkl",
    )
    with open(path, "rb") as fd:
        df = pickle.load(fd)

    expected = DataFrame(index=[], columns=[])
    tm.assert_frame_equal(df, expected)


def test_pickle_pos_args_deprecation():
    # GH-54229
    df = DataFrame({"a": [1, 2, 3]})
    msg = (
        r"Starting with pandas version 3.0 all arguments of to_pickle except for the "
        r"argument 'path' will be keyword-only."
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        buffer = io.BytesIO()
        df.to_pickle(buffer, "infer")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_fiscal.py ===
"""
Tests for Fiscal Year and Fiscal Quarter offset classes
"""
from datetime import datetime

from dateutil.relativedelta import relativedelta
import pytest

from pandas import Timestamp
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
    WeekDay,
    assert_is_on_offset,
    assert_offset_equal,
)

from pandas.tseries.offsets import (
    FY5253,
    FY5253Quarter,
)


def makeFY5253LastOfMonthQuarter(*args, **kwds):
    return FY5253Quarter(*args, variation="last", **kwds)


def makeFY5253NearestEndMonthQuarter(*args, **kwds):
    return FY5253Quarter(*args, variation="nearest", **kwds)


def makeFY5253NearestEndMonth(*args, **kwds):
    return FY5253(*args, variation="nearest", **kwds)


def makeFY5253LastOfMonth(*args, **kwds):
    return FY5253(*args, variation="last", **kwds)


def test_get_offset_name():
    assert (
        makeFY5253LastOfMonthQuarter(
            weekday=1, startingMonth=3, qtr_with_extra_week=4
        ).freqstr
        == "REQ-L-MAR-TUE-4"
    )
    assert (
        makeFY5253NearestEndMonthQuarter(
            weekday=1, startingMonth=3, qtr_with_extra_week=3
        ).freqstr
        == "REQ-N-MAR-TUE-3"
    )


class TestFY5253LastOfMonth:
    offset_lom_sat_aug = makeFY5253LastOfMonth(1, startingMonth=8, weekday=WeekDay.SAT)
    offset_lom_sat_sep = makeFY5253LastOfMonth(1, startingMonth=9, weekday=WeekDay.SAT)

    on_offset_cases = [
        # From Wikipedia (see:
        # https://en.wikipedia.org/wiki/4%E2%80%934%E2%80%935_calendar#Last_Saturday_of_the_month_at_fiscal_year_end)
        (offset_lom_sat_aug, datetime(2006, 8, 26), True),
        (offset_lom_sat_aug, datetime(2007, 8, 25), True),
        (offset_lom_sat_aug, datetime(2008, 8, 30), True),
        (offset_lom_sat_aug, datetime(2009, 8, 29), True),
        (offset_lom_sat_aug, datetime(2010, 8, 28), True),
        (offset_lom_sat_aug, datetime(2011, 8, 27), True),
        (offset_lom_sat_aug, datetime(2012, 8, 25), True),
        (offset_lom_sat_aug, datetime(2013, 8, 31), True),
        (offset_lom_sat_aug, datetime(2014, 8, 30), True),
        (offset_lom_sat_aug, datetime(2015, 8, 29), True),
        (offset_lom_sat_aug, datetime(2016, 8, 27), True),
        (offset_lom_sat_aug, datetime(2017, 8, 26), True),
        (offset_lom_sat_aug, datetime(2018, 8, 25), True),
        (offset_lom_sat_aug, datetime(2019, 8, 31), True),
        (offset_lom_sat_aug, datetime(2006, 8, 27), False),
        (offset_lom_sat_aug, datetime(2007, 8, 28), False),
        (offset_lom_sat_aug, datetime(2008, 8, 31), False),
        (offset_lom_sat_aug, datetime(2009, 8, 30), False),
        (offset_lom_sat_aug, datetime(2010, 8, 29), False),
        (offset_lom_sat_aug, datetime(2011, 8, 28), False),
        (offset_lom_sat_aug, datetime(2006, 8, 25), False),
        (offset_lom_sat_aug, datetime(2007, 8, 24), False),
        (offset_lom_sat_aug, datetime(2008, 8, 29), False),
        (offset_lom_sat_aug, datetime(2009, 8, 28), False),
        (offset_lom_sat_aug, datetime(2010, 8, 27), False),
        (offset_lom_sat_aug, datetime(2011, 8, 26), False),
        (offset_lom_sat_aug, datetime(2019, 8, 30), False),
        # From GMCR (see for example:
        # http://yahoo.brand.edgar-online.com/Default.aspx?
        # companyid=3184&formtypeID=7)
        (offset_lom_sat_sep, datetime(2010, 9, 25), True),
        (offset_lom_sat_sep, datetime(2011, 9, 24), True),
        (offset_lom_sat_sep, datetime(2012, 9, 29), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)

    def test_apply(self):
        offset_lom_aug_sat = makeFY5253LastOfMonth(startingMonth=8, weekday=WeekDay.SAT)
        offset_lom_aug_sat_1 = makeFY5253LastOfMonth(
            n=1, startingMonth=8, weekday=WeekDay.SAT
        )

        date_seq_lom_aug_sat = [
            datetime(2006, 8, 26),
            datetime(2007, 8, 25),
            datetime(2008, 8, 30),
            datetime(2009, 8, 29),
            datetime(2010, 8, 28),
            datetime(2011, 8, 27),
            datetime(2012, 8, 25),
            datetime(2013, 8, 31),
            datetime(2014, 8, 30),
            datetime(2015, 8, 29),
            datetime(2016, 8, 27),
        ]

        tests = [
            (offset_lom_aug_sat, date_seq_lom_aug_sat),
            (offset_lom_aug_sat_1, date_seq_lom_aug_sat),
            (offset_lom_aug_sat, [datetime(2006, 8, 25)] + date_seq_lom_aug_sat),
            (offset_lom_aug_sat_1, [datetime(2006, 8, 27)] + date_seq_lom_aug_sat[1:]),
            (
                makeFY5253LastOfMonth(n=-1, startingMonth=8, weekday=WeekDay.SAT),
                list(reversed(date_seq_lom_aug_sat)),
            ),
        ]
        for test in tests:
            offset, data = test
            current = data[0]
            for datum in data[1:]:
                current = current + offset
                assert current == datum


class TestFY5253NearestEndMonth:
    def test_get_year_end(self):
        assert makeFY5253NearestEndMonth(
            startingMonth=8, weekday=WeekDay.SAT
        ).get_year_end(datetime(2013, 1, 1)) == datetime(2013, 8, 31)
        assert makeFY5253NearestEndMonth(
            startingMonth=8, weekday=WeekDay.SUN
        ).get_year_end(datetime(2013, 1, 1)) == datetime(2013, 9, 1)
        assert makeFY5253NearestEndMonth(
            startingMonth=8, weekday=WeekDay.FRI
        ).get_year_end(datetime(2013, 1, 1)) == datetime(2013, 8, 30)

        offset_n = FY5253(weekday=WeekDay.TUE, startingMonth=12, variation="nearest")
        assert offset_n.get_year_end(datetime(2012, 1, 1)) == datetime(2013, 1, 1)
        assert offset_n.get_year_end(datetime(2012, 1, 10)) == datetime(2013, 1, 1)

        assert offset_n.get_year_end(datetime(2013, 1, 1)) == datetime(2013, 12, 31)
        assert offset_n.get_year_end(datetime(2013, 1, 2)) == datetime(2013, 12, 31)
        assert offset_n.get_year_end(datetime(2013, 1, 3)) == datetime(2013, 12, 31)
        assert offset_n.get_year_end(datetime(2013, 1, 10)) == datetime(2013, 12, 31)

        JNJ = FY5253(n=1, startingMonth=12, weekday=6, variation="nearest")
        assert JNJ.get_year_end(datetime(2006, 1, 1)) == datetime(2006, 12, 31)

    offset_lom_aug_sat = makeFY5253NearestEndMonth(
        1, startingMonth=8, weekday=WeekDay.SAT
    )
    offset_lom_aug_thu = makeFY5253NearestEndMonth(
        1, startingMonth=8, weekday=WeekDay.THU
    )
    offset_n = FY5253(weekday=WeekDay.TUE, startingMonth=12, variation="nearest")

    on_offset_cases = [
        #    From Wikipedia (see:
        #    https://en.wikipedia.org/wiki/4%E2%80%934%E2%80%935_calendar
        #    #Saturday_nearest_the_end_of_month)
        #    2006-09-02   2006 September 2
        #    2007-09-01   2007 September 1
        #    2008-08-30   2008 August 30    (leap year)
        #    2009-08-29   2009 August 29
        #    2010-08-28   2010 August 28
        #    2011-09-03   2011 September 3
        #    2012-09-01   2012 September 1  (leap year)
        #    2013-08-31   2013 August 31
        #    2014-08-30   2014 August 30
        #    2015-08-29   2015 August 29
        #    2016-09-03   2016 September 3  (leap year)
        #    2017-09-02   2017 September 2
        #    2018-09-01   2018 September 1
        #    2019-08-31   2019 August 31
        (offset_lom_aug_sat, datetime(2006, 9, 2), True),
        (offset_lom_aug_sat, datetime(2007, 9, 1), True),
        (offset_lom_aug_sat, datetime(2008, 8, 30), True),
        (offset_lom_aug_sat, datetime(2009, 8, 29), True),
        (offset_lom_aug_sat, datetime(2010, 8, 28), True),
        (offset_lom_aug_sat, datetime(2011, 9, 3), True),
        (offset_lom_aug_sat, datetime(2016, 9, 3), True),
        (offset_lom_aug_sat, datetime(2017, 9, 2), True),
        (offset_lom_aug_sat, datetime(2018, 9, 1), True),
        (offset_lom_aug_sat, datetime(2019, 8, 31), True),
        (offset_lom_aug_sat, datetime(2006, 8, 27), False),
        (offset_lom_aug_sat, datetime(2007, 8, 28), False),
        (offset_lom_aug_sat, datetime(2008, 8, 31), False),
        (offset_lom_aug_sat, datetime(2009, 8, 30), False),
        (offset_lom_aug_sat, datetime(2010, 8, 29), False),
        (offset_lom_aug_sat, datetime(2011, 8, 28), False),
        (offset_lom_aug_sat, datetime(2006, 8, 25), False),
        (offset_lom_aug_sat, datetime(2007, 8, 24), False),
        (offset_lom_aug_sat, datetime(2008, 8, 29), False),
        (offset_lom_aug_sat, datetime(2009, 8, 28), False),
        (offset_lom_aug_sat, datetime(2010, 8, 27), False),
        (offset_lom_aug_sat, datetime(2011, 8, 26), False),
        (offset_lom_aug_sat, datetime(2019, 8, 30), False),
        # From Micron, see:
        # http://google.brand.edgar-online.com/?sym=MU&formtypeID=7
        (offset_lom_aug_thu, datetime(2012, 8, 30), True),
        (offset_lom_aug_thu, datetime(2011, 9, 1), True),
        (offset_n, datetime(2012, 12, 31), False),
        (offset_n, datetime(2013, 1, 1), True),
        (offset_n, datetime(2013, 1, 2), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)

    def test_apply(self):
        date_seq_nem_8_sat = [
            datetime(2006, 9, 2),
            datetime(2007, 9, 1),
            datetime(2008, 8, 30),
            datetime(2009, 8, 29),
            datetime(2010, 8, 28),
            datetime(2011, 9, 3),
        ]

        JNJ = [
            datetime(2005, 1, 2),
            datetime(2006, 1, 1),
            datetime(2006, 12, 31),
            datetime(2007, 12, 30),
            datetime(2008, 12, 28),
            datetime(2010, 1, 3),
            datetime(2011, 1, 2),
            datetime(2012, 1, 1),
            datetime(2012, 12, 30),
        ]

        DEC_SAT = FY5253(n=-1, startingMonth=12, weekday=5, variation="nearest")

        tests = [
            (
                makeFY5253NearestEndMonth(startingMonth=8, weekday=WeekDay.SAT),
                date_seq_nem_8_sat,
            ),
            (
                makeFY5253NearestEndMonth(n=1, startingMonth=8, weekday=WeekDay.SAT),
                date_seq_nem_8_sat,
            ),
            (
                makeFY5253NearestEndMonth(startingMonth=8, weekday=WeekDay.SAT),
                [datetime(2006, 9, 1)] + date_seq_nem_8_sat,
            ),
            (
                makeFY5253NearestEndMonth(n=1, startingMonth=8, weekday=WeekDay.SAT),
                [datetime(2006, 9, 3)] + date_seq_nem_8_sat[1:],
            ),
            (
                makeFY5253NearestEndMonth(n=-1, startingMonth=8, weekday=WeekDay.SAT),
                list(reversed(date_seq_nem_8_sat)),
            ),
            (
                makeFY5253NearestEndMonth(n=1, startingMonth=12, weekday=WeekDay.SUN),
                JNJ,
            ),
            (
                makeFY5253NearestEndMonth(n=-1, startingMonth=12, weekday=WeekDay.SUN),
                list(reversed(JNJ)),
            ),
            (
                makeFY5253NearestEndMonth(n=1, startingMonth=12, weekday=WeekDay.SUN),
                [datetime(2005, 1, 2), datetime(2006, 1, 1)],
            ),
            (
                makeFY5253NearestEndMonth(n=1, startingMonth=12, weekday=WeekDay.SUN),
                [datetime(2006, 1, 2), datetime(2006, 12, 31)],
            ),
            (DEC_SAT, [datetime(2013, 1, 15), datetime(2012, 12, 29)]),
        ]
        for test in tests:
            offset, data = test
            current = data[0]
            for datum in data[1:]:
                current = current + offset
                assert current == datum


class TestFY5253LastOfMonthQuarter:
    def test_is_anchored(self):
        msg = "FY5253Quarter.is_anchored is deprecated "

        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert makeFY5253LastOfMonthQuarter(
                startingMonth=1, weekday=WeekDay.SAT, qtr_with_extra_week=4
            ).is_anchored()
            assert makeFY5253LastOfMonthQuarter(
                weekday=WeekDay.SAT, startingMonth=3, qtr_with_extra_week=4
            ).is_anchored()
            assert not makeFY5253LastOfMonthQuarter(
                2, startingMonth=1, weekday=WeekDay.SAT, qtr_with_extra_week=4
            ).is_anchored()

    def test_equality(self):
        assert makeFY5253LastOfMonthQuarter(
            startingMonth=1, weekday=WeekDay.SAT, qtr_with_extra_week=4
        ) == makeFY5253LastOfMonthQuarter(
            startingMonth=1, weekday=WeekDay.SAT, qtr_with_extra_week=4
        )
        assert makeFY5253LastOfMonthQuarter(
            startingMonth=1, weekday=WeekDay.SAT, qtr_with_extra_week=4
        ) != makeFY5253LastOfMonthQuarter(
            startingMonth=1, weekday=WeekDay.SUN, qtr_with_extra_week=4
        )
        assert makeFY5253LastOfMonthQuarter(
            startingMonth=1, weekday=WeekDay.SAT, qtr_with_extra_week=4
        ) != makeFY5253LastOfMonthQuarter(
            startingMonth=2, weekday=WeekDay.SAT, qtr_with_extra_week=4
        )

    def test_offset(self):
        offset = makeFY5253LastOfMonthQuarter(
            1, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4
        )
        offset2 = makeFY5253LastOfMonthQuarter(
            2, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4
        )
        offset4 = makeFY5253LastOfMonthQuarter(
            4, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4
        )

        offset_neg1 = makeFY5253LastOfMonthQuarter(
            -1, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4
        )
        offset_neg2 = makeFY5253LastOfMonthQuarter(
            -2, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4
        )

        GMCR = [
            datetime(2010, 3, 27),
            datetime(2010, 6, 26),
            datetime(2010, 9, 25),
            datetime(2010, 12, 25),
            datetime(2011, 3, 26),
            datetime(2011, 6, 25),
            datetime(2011, 9, 24),
            datetime(2011, 12, 24),
            datetime(2012, 3, 24),
            datetime(2012, 6, 23),
            datetime(2012, 9, 29),
            datetime(2012, 12, 29),
            datetime(2013, 3, 30),
            datetime(2013, 6, 29),
        ]

        assert_offset_equal(offset, base=GMCR[0], expected=GMCR[1])
        assert_offset_equal(
            offset, base=GMCR[0] + relativedelta(days=-1), expected=GMCR[0]
        )
        assert_offset_equal(offset, base=GMCR[1], expected=GMCR[2])

        assert_offset_equal(offset2, base=GMCR[0], expected=GMCR[2])
        assert_offset_equal(offset4, base=GMCR[0], expected=GMCR[4])

        assert_offset_equal(offset_neg1, base=GMCR[-1], expected=GMCR[-2])
        assert_offset_equal(
            offset_neg1, base=GMCR[-1] + relativedelta(days=+1), expected=GMCR[-1]
        )
        assert_offset_equal(offset_neg2, base=GMCR[-1], expected=GMCR[-3])

        date = GMCR[0] + relativedelta(days=-1)
        for expected in GMCR:
            assert_offset_equal(offset, date, expected)
            date = date + offset

        date = GMCR[-1] + relativedelta(days=+1)
        for expected in reversed(GMCR):
            assert_offset_equal(offset_neg1, date, expected)
            date = date + offset_neg1

    lomq_aug_sat_4 = makeFY5253LastOfMonthQuarter(
        1, startingMonth=8, weekday=WeekDay.SAT, qtr_with_extra_week=4
    )
    lomq_sep_sat_4 = makeFY5253LastOfMonthQuarter(
        1, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4
    )

    on_offset_cases = [
        # From Wikipedia
        (lomq_aug_sat_4, datetime(2006, 8, 26), True),
        (lomq_aug_sat_4, datetime(2007, 8, 25), True),
        (lomq_aug_sat_4, datetime(2008, 8, 30), True),
        (lomq_aug_sat_4, datetime(2009, 8, 29), True),
        (lomq_aug_sat_4, datetime(2010, 8, 28), True),
        (lomq_aug_sat_4, datetime(2011, 8, 27), True),
        (lomq_aug_sat_4, datetime(2019, 8, 31), True),
        (lomq_aug_sat_4, datetime(2006, 8, 27), False),
        (lomq_aug_sat_4, datetime(2007, 8, 28), False),
        (lomq_aug_sat_4, datetime(2008, 8, 31), False),
        (lomq_aug_sat_4, datetime(2009, 8, 30), False),
        (lomq_aug_sat_4, datetime(2010, 8, 29), False),
        (lomq_aug_sat_4, datetime(2011, 8, 28), False),
        (lomq_aug_sat_4, datetime(2006, 8, 25), False),
        (lomq_aug_sat_4, datetime(2007, 8, 24), False),
        (lomq_aug_sat_4, datetime(2008, 8, 29), False),
        (lomq_aug_sat_4, datetime(2009, 8, 28), False),
        (lomq_aug_sat_4, datetime(2010, 8, 27), False),
        (lomq_aug_sat_4, datetime(2011, 8, 26), False),
        (lomq_aug_sat_4, datetime(2019, 8, 30), False),
        # From GMCR
        (lomq_sep_sat_4, datetime(2010, 9, 25), True),
        (lomq_sep_sat_4, datetime(2011, 9, 24), True),
        (lomq_sep_sat_4, datetime(2012, 9, 29), True),
        (lomq_sep_sat_4, datetime(2013, 6, 29), True),
        (lomq_sep_sat_4, datetime(2012, 6, 23), True),
        (lomq_sep_sat_4, datetime(2012, 6, 30), False),
        (lomq_sep_sat_4, datetime(2013, 3, 30), True),
        (lomq_sep_sat_4, datetime(2012, 3, 24), True),
        (lomq_sep_sat_4, datetime(2012, 12, 29), True),
        (lomq_sep_sat_4, datetime(2011, 12, 24), True),
        # INTC (extra week in Q1)
        # See: http://www.intc.com/releasedetail.cfm?ReleaseID=542844
        (
            makeFY5253LastOfMonthQuarter(
                1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
            ),
            datetime(2011, 4, 2),
            True,
        ),
        # see: http://google.brand.edgar-online.com/?sym=INTC&formtypeID=7
        (
            makeFY5253LastOfMonthQuarter(
                1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
            ),
            datetime(2012, 12, 29),
            True,
        ),
        (
            makeFY5253LastOfMonthQuarter(
                1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
            ),
            datetime(2011, 12, 31),
            True,
        ),
        (
            makeFY5253LastOfMonthQuarter(
                1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
            ),
            datetime(2010, 12, 25),
            True,
        ),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)

    def test_year_has_extra_week(self):
        # End of long Q1
        assert makeFY5253LastOfMonthQuarter(
            1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
        ).year_has_extra_week(datetime(2011, 4, 2))

        # Start of long Q1
        assert makeFY5253LastOfMonthQuarter(
            1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
        ).year_has_extra_week(datetime(2010, 12, 26))

        # End of year before year with long Q1
        assert not makeFY5253LastOfMonthQuarter(
            1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
        ).year_has_extra_week(datetime(2010, 12, 25))

        for year in [
            x for x in range(1994, 2011 + 1) if x not in [2011, 2005, 2000, 1994]
        ]:
            assert not makeFY5253LastOfMonthQuarter(
                1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
            ).year_has_extra_week(datetime(year, 4, 2))

        # Other long years
        assert makeFY5253LastOfMonthQuarter(
            1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
        ).year_has_extra_week(datetime(2005, 4, 2))

        assert makeFY5253LastOfMonthQuarter(
            1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
        ).year_has_extra_week(datetime(2000, 4, 2))

        assert makeFY5253LastOfMonthQuarter(
            1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
        ).year_has_extra_week(datetime(1994, 4, 2))

    def test_get_weeks(self):
        sat_dec_1 = makeFY5253LastOfMonthQuarter(
            1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
        )
        sat_dec_4 = makeFY5253LastOfMonthQuarter(
            1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=4
        )

        assert sat_dec_1.get_weeks(datetime(2011, 4, 2)) == [14, 13, 13, 13]
        assert sat_dec_4.get_weeks(datetime(2011, 4, 2)) == [13, 13, 13, 14]
        assert sat_dec_1.get_weeks(datetime(2010, 12, 25)) == [13, 13, 13, 13]


class TestFY5253NearestEndMonthQuarter:
    offset_nem_sat_aug_4 = makeFY5253NearestEndMonthQuarter(
        1, startingMonth=8, weekday=WeekDay.SAT, qtr_with_extra_week=4
    )
    offset_nem_thu_aug_4 = makeFY5253NearestEndMonthQuarter(
        1, startingMonth=8, weekday=WeekDay.THU, qtr_with_extra_week=4
    )
    offset_n = FY5253(weekday=WeekDay.TUE, startingMonth=12, variation="nearest")

    on_offset_cases = [
        # From Wikipedia
        (offset_nem_sat_aug_4, datetime(2006, 9, 2), True),
        (offset_nem_sat_aug_4, datetime(2007, 9, 1), True),
        (offset_nem_sat_aug_4, datetime(2008, 8, 30), True),
        (offset_nem_sat_aug_4, datetime(2009, 8, 29), True),
        (offset_nem_sat_aug_4, datetime(2010, 8, 28), True),
        (offset_nem_sat_aug_4, datetime(2011, 9, 3), True),
        (offset_nem_sat_aug_4, datetime(2016, 9, 3), True),
        (offset_nem_sat_aug_4, datetime(2017, 9, 2), True),
        (offset_nem_sat_aug_4, datetime(2018, 9, 1), True),
        (offset_nem_sat_aug_4, datetime(2019, 8, 31), True),
        (offset_nem_sat_aug_4, datetime(2006, 8, 27), False),
        (offset_nem_sat_aug_4, datetime(2007, 8, 28), False),
        (offset_nem_sat_aug_4, datetime(2008, 8, 31), False),
        (offset_nem_sat_aug_4, datetime(2009, 8, 30), False),
        (offset_nem_sat_aug_4, datetime(2010, 8, 29), False),
        (offset_nem_sat_aug_4, datetime(2011, 8, 28), False),
        (offset_nem_sat_aug_4, datetime(2006, 8, 25), False),
        (offset_nem_sat_aug_4, datetime(2007, 8, 24), False),
        (offset_nem_sat_aug_4, datetime(2008, 8, 29), False),
        (offset_nem_sat_aug_4, datetime(2009, 8, 28), False),
        (offset_nem_sat_aug_4, datetime(2010, 8, 27), False),
        (offset_nem_sat_aug_4, datetime(2011, 8, 26), False),
        (offset_nem_sat_aug_4, datetime(2019, 8, 30), False),
        # From Micron, see:
        # http://google.brand.edgar-online.com/?sym=MU&formtypeID=7
        (offset_nem_thu_aug_4, datetime(2012, 8, 30), True),
        (offset_nem_thu_aug_4, datetime(2011, 9, 1), True),
        # See: http://google.brand.edgar-online.com/?sym=MU&formtypeID=13
        (offset_nem_thu_aug_4, datetime(2013, 5, 30), True),
        (offset_nem_thu_aug_4, datetime(2013, 2, 28), True),
        (offset_nem_thu_aug_4, datetime(2012, 11, 29), True),
        (offset_nem_thu_aug_4, datetime(2012, 5, 31), True),
        (offset_nem_thu_aug_4, datetime(2007, 3, 1), True),
        (offset_nem_thu_aug_4, datetime(1994, 3, 3), True),
        (offset_n, datetime(2012, 12, 31), False),
        (offset_n, datetime(2013, 1, 1), True),
        (offset_n, datetime(2013, 1, 2), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)

    def test_offset(self):
        offset = makeFY5253NearestEndMonthQuarter(
            1, startingMonth=8, weekday=WeekDay.THU, qtr_with_extra_week=4
        )

        MU = [
            datetime(2012, 5, 31),
            datetime(2012, 8, 30),
            datetime(2012, 11, 29),
            datetime(2013, 2, 28),
            datetime(2013, 5, 30),
        ]

        date = MU[0] + relativedelta(days=-1)
        for expected in MU:
            assert_offset_equal(offset, date, expected)
            date = date + offset

        assert_offset_equal(offset, datetime(2012, 5, 31), datetime(2012, 8, 30))
        assert_offset_equal(offset, datetime(2012, 5, 30), datetime(2012, 5, 31))

        offset2 = FY5253Quarter(
            weekday=5, startingMonth=12, variation="last", qtr_with_extra_week=4
        )

        assert_offset_equal(offset2, datetime(2013, 1, 15), datetime(2013, 3, 30))


def test_bunched_yearends():
    # GH#14774 cases with two fiscal year-ends in the same calendar-year
    fy = FY5253(n=1, weekday=5, startingMonth=12, variation="nearest")
    dt = Timestamp("2004-01-01")
    assert fy.rollback(dt) == Timestamp("2002-12-28")
    assert (-fy)._apply(dt) == Timestamp("2002-12-28")
    assert dt - fy == Timestamp("2002-12-28")

    assert fy.rollforward(dt) == Timestamp("2004-01-03")
    assert fy._apply(dt) == Timestamp("2004-01-03")
    assert fy + dt == Timestamp("2004-01-03")
    assert dt + fy == Timestamp("2004-01-03")

    # Same thing, but starting from a Timestamp in the previous year.
    dt = Timestamp("2003-12-31")
    assert fy.rollback(dt) == Timestamp("2002-12-28")
    assert (-fy)._apply(dt) == Timestamp("2002-12-28")
    assert dt - fy == Timestamp("2002-12-28")


def test_fy5253_last_onoffset():
    # GH#18877 dates on the year-end but not normalized to midnight
    offset = FY5253(n=-5, startingMonth=5, variation="last", weekday=0)
    ts = Timestamp("1984-05-28 06:29:43.955911354+0200", tz="Europe/San_Marino")
    fast = offset.is_on_offset(ts)
    slow = (ts + offset) - offset == ts
    assert fast == slow


def test_fy5253_nearest_onoffset():
    # GH#18877 dates on the year-end but not normalized to midnight
    offset = FY5253(n=3, startingMonth=7, variation="nearest", weekday=2)
    ts = Timestamp("2032-07-28 00:12:59.035729419+0000", tz="Africa/Dakar")
    fast = offset.is_on_offset(ts)
    slow = (ts + offset) - offset == ts
    assert fast == slow


def test_fy5253qtr_onoffset_nearest():
    # GH#19036
    ts = Timestamp("1985-09-02 23:57:46.232550356-0300", tz="Atlantic/Bermuda")
    offset = FY5253Quarter(
        n=3, qtr_with_extra_week=1, startingMonth=2, variation="nearest", weekday=0
    )
    fast = offset.is_on_offset(ts)
    slow = (ts + offset) - offset == ts
    assert fast == slow


def test_fy5253qtr_onoffset_last():
    # GH#19036
    offset = FY5253Quarter(
        n=-2, qtr_with_extra_week=1, startingMonth=7, variation="last", weekday=2
    )
    ts = Timestamp("2011-01-26 19:03:40.331096129+0200", tz="Africa/Windhoek")
    slow = (ts + offset) - offset == ts
    fast = offset.is_on_offset(ts)
    assert fast == slow

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\combinatorial\tests\test_comb_factorials.py ===
from sympy.concrete.products import Product
from sympy.core.function import expand_func
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core import EulerGamma
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import (ff, rf, binomial, factorial, factorial2)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.polys.polytools import Poly
from sympy.series.order import O
from sympy.simplify.simplify import simplify
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.functions.combinatorial.factorials import subfactorial
from sympy.functions.special.gamma_functions import uppergamma
from sympy.testing.pytest import XFAIL, raises, slow

#Solves and Fixes Issue #10388 - This is the updated test for the same solved issue

def test_rf_eval_apply():
    x, y = symbols('x,y')
    n, k = symbols('n k', integer=True)
    m = Symbol('m', integer=True, nonnegative=True)

    assert rf(nan, y) is nan
    assert rf(x, nan) is nan

    assert unchanged(rf, x, y)

    assert rf(oo, 0) == 1
    assert rf(-oo, 0) == 1

    assert rf(oo, 6) is oo
    assert rf(-oo, 7) is -oo
    assert rf(-oo, 6) is oo

    assert rf(oo, -6) is oo
    assert rf(-oo, -7) is oo

    assert rf(-1, pi) == 0
    assert rf(-5, 1 + I) == 0

    assert unchanged(rf, -3, k)
    assert unchanged(rf, x, Symbol('k', integer=False))
    assert rf(-3, Symbol('k', integer=False)) == 0
    assert rf(Symbol('x', negative=True, integer=True), Symbol('k', integer=False)) == 0

    assert rf(x, 0) == 1
    assert rf(x, 1) == x
    assert rf(x, 2) == x*(x + 1)
    assert rf(x, 3) == x*(x + 1)*(x + 2)
    assert rf(x, 5) == x*(x + 1)*(x + 2)*(x + 3)*(x + 4)

    assert rf(x, -1) == 1/(x - 1)
    assert rf(x, -2) == 1/((x - 1)*(x - 2))
    assert rf(x, -3) == 1/((x - 1)*(x - 2)*(x - 3))

    assert rf(1, 100) == factorial(100)

    assert rf(x**2 + 3*x, 2) == (x**2 + 3*x)*(x**2 + 3*x + 1)
    assert isinstance(rf(x**2 + 3*x, 2), Mul)
    assert rf(x**3 + x, -2) == 1/((x**3 + x - 1)*(x**3 + x - 2))

    assert rf(Poly(x**2 + 3*x, x), 2) == Poly(x**4 + 8*x**3 + 19*x**2 + 12*x, x)
    assert isinstance(rf(Poly(x**2 + 3*x, x), 2), Poly)
    raises(ValueError, lambda: rf(Poly(x**2 + 3*x, x, y), 2))
    assert rf(Poly(x**3 + x, x), -2) == 1/(x**6 - 9*x**5 + 35*x**4 - 75*x**3 + 94*x**2 - 66*x + 20)
    raises(ValueError, lambda: rf(Poly(x**3 + x, x, y), -2))

    assert rf(x, m).is_integer is None
    assert rf(n, k).is_integer is None
    assert rf(n, m).is_integer is True
    assert rf(n, k + pi).is_integer is False
    assert rf(n, m + pi).is_integer is False
    assert rf(pi, m).is_integer is False

    def check(x, k, o, n):
        a, b = Dummy(), Dummy()
        r = lambda x, k: o(a, b).rewrite(n).subs({a:x,b:k})
        for i in range(-5,5):
            for j in range(-5,5):
                assert o(i, j) == r(i, j), (o, n, i, j)
    check(x, k, rf, ff)
    check(x, k, rf, binomial)
    check(n, k, rf, factorial)
    check(x, y, rf, factorial)
    check(x, y, rf, binomial)

    assert rf(x, k).rewrite(ff) == ff(x + k - 1, k)
    assert rf(x, k).rewrite(gamma) == Piecewise(
        (gamma(k + x)/gamma(x), x > 0),
        ((-1)**k*gamma(1 - x)/gamma(-k - x + 1), True))
    assert rf(5, k).rewrite(gamma) == gamma(k + 5)/24
    assert rf(x, k).rewrite(binomial) == factorial(k)*binomial(x + k - 1, k)
    assert rf(n, k).rewrite(factorial) == Piecewise(
        (factorial(k + n - 1)/factorial(n - 1), n > 0),
        ((-1)**k*factorial(-n)/factorial(-k - n), True))
    assert rf(5, k).rewrite(factorial) == factorial(k + 4)/24
    assert rf(x, y).rewrite(factorial) == rf(x, y)
    assert rf(x, y).rewrite(binomial) == rf(x, y)

    import random
    from mpmath import rf as mpmath_rf
    for i in range(100):
        x = -500 + 500 * random.random()
        k = -500 + 500 * random.random()
        assert (abs(mpmath_rf(x, k) - rf(x, k)) < 10**(-15))


def test_ff_eval_apply():
    x, y = symbols('x,y')
    n, k = symbols('n k', integer=True)
    m = Symbol('m', integer=True, nonnegative=True)

    assert ff(nan, y) is nan
    assert ff(x, nan) is nan

    assert unchanged(ff, x, y)

    assert ff(oo, 0) == 1
    assert ff(-oo, 0) == 1

    assert ff(oo, 6) is oo
    assert ff(-oo, 7) is -oo
    assert ff(-oo, 6) is oo

    assert ff(oo, -6) is oo
    assert ff(-oo, -7) is oo

    assert ff(x, 0) == 1
    assert ff(x, 1) == x
    assert ff(x, 2) == x*(x - 1)
    assert ff(x, 3) == x*(x - 1)*(x - 2)
    assert ff(x, 5) == x*(x - 1)*(x - 2)*(x - 3)*(x - 4)

    assert ff(x, -1) == 1/(x + 1)
    assert ff(x, -2) == 1/((x + 1)*(x + 2))
    assert ff(x, -3) == 1/((x + 1)*(x + 2)*(x + 3))

    assert ff(100, 100) == factorial(100)

    assert ff(2*x**2 - 5*x, 2) == (2*x**2  - 5*x)*(2*x**2 - 5*x - 1)
    assert isinstance(ff(2*x**2 - 5*x, 2), Mul)
    assert ff(x**2 + 3*x, -2) == 1/((x**2 + 3*x + 1)*(x**2 + 3*x + 2))

    assert ff(Poly(2*x**2 - 5*x, x), 2) == Poly(4*x**4 - 28*x**3 + 59*x**2 - 35*x, x)
    assert isinstance(ff(Poly(2*x**2 - 5*x, x), 2), Poly)
    raises(ValueError, lambda: ff(Poly(2*x**2 - 5*x, x, y), 2))
    assert ff(Poly(x**2 + 3*x, x), -2) == 1/(x**4 + 12*x**3 + 49*x**2 + 78*x + 40)
    raises(ValueError, lambda: ff(Poly(x**2 + 3*x, x, y), -2))


    assert ff(x, m).is_integer is None
    assert ff(n, k).is_integer is None
    assert ff(n, m).is_integer is True
    assert ff(n, k + pi).is_integer is False
    assert ff(n, m + pi).is_integer is False
    assert ff(pi, m).is_integer is False

    assert isinstance(ff(x, x), ff)
    assert ff(n, n) == factorial(n)

    def check(x, k, o, n):
        a, b = Dummy(), Dummy()
        r = lambda x, k: o(a, b).rewrite(n).subs({a:x,b:k})
        for i in range(-5,5):
            for j in range(-5,5):
                assert o(i, j) == r(i, j), (o, n)
    check(x, k, ff, rf)
    check(x, k, ff, gamma)
    check(n, k, ff, factorial)
    check(x, k, ff, binomial)
    check(x, y, ff, factorial)
    check(x, y, ff, binomial)

    assert ff(x, k).rewrite(rf) == rf(x - k + 1, k)
    assert ff(x, k).rewrite(gamma) == Piecewise(
        (gamma(x + 1)/gamma(-k + x + 1), x >= 0),
        ((-1)**k*gamma(k - x)/gamma(-x), True))
    assert ff(5, k).rewrite(gamma) == 120/gamma(6 - k)
    assert ff(n, k).rewrite(factorial) == Piecewise(
        (factorial(n)/factorial(-k + n), n >= 0),
        ((-1)**k*factorial(k - n - 1)/factorial(-n - 1), True))
    assert ff(5, k).rewrite(factorial) == 120/factorial(5 - k)
    assert ff(x, k).rewrite(binomial) == factorial(k) * binomial(x, k)
    assert ff(x, y).rewrite(factorial) == ff(x, y)
    assert ff(x, y).rewrite(binomial) == ff(x, y)

    import random
    from mpmath import ff as mpmath_ff
    for i in range(100):
        x = -500 + 500 * random.random()
        k = -500 + 500 * random.random()
        a = mpmath_ff(x, k)
        b = ff(x, k)
        assert (abs(a - b) < abs(a) * 10**(-15))


def test_rf_ff_eval_hiprec():
    maple = Float('6.9109401292234329956525265438452')
    us = ff(18, Rational(2, 3)).evalf(32)
    assert abs(us - maple)/us < 1e-31

    maple = Float('6.8261540131125511557924466355367')
    us = rf(18, Rational(2, 3)).evalf(32)
    assert abs(us - maple)/us < 1e-31

    maple = Float('34.007346127440197150854651814225')
    us = rf(Float('4.4', 32), Float('2.2', 32))
    assert abs(us - maple)/us < 1e-31


def test_rf_lambdify_mpmath():
    from sympy.utilities.lambdify import lambdify
    x, y = symbols('x,y')
    f = lambdify((x,y), rf(x, y), 'mpmath')
    maple = Float('34.007346127440197')
    us = f(4.4, 2.2)
    assert abs(us - maple)/us < 1e-15


def test_factorial():
    x = Symbol('x')
    n = Symbol('n', integer=True)
    k = Symbol('k', integer=True, nonnegative=True)
    r = Symbol('r', integer=False)
    s = Symbol('s', integer=False, negative=True)
    t = Symbol('t', nonnegative=True)
    u = Symbol('u', noninteger=True)

    assert factorial(-2) is zoo
    assert factorial(0) == 1
    assert factorial(7) == 5040
    assert factorial(19) == 121645100408832000
    assert factorial(31) == 8222838654177922817725562880000000
    assert factorial(n).func == factorial
    assert factorial(2*n).func == factorial

    assert factorial(x).is_integer is None
    assert factorial(n).is_integer is None
    assert factorial(k).is_integer
    assert factorial(r).is_integer is None

    assert factorial(n).is_positive is None
    assert factorial(k).is_positive

    assert factorial(x).is_real is None
    assert factorial(n).is_real is None
    assert factorial(k).is_real is True
    assert factorial(r).is_real is None
    assert factorial(s).is_real is True
    assert factorial(t).is_real is True
    assert factorial(u).is_real is True

    assert factorial(x).is_composite is None
    assert factorial(n).is_composite is None
    assert factorial(k).is_composite is None
    assert factorial(k + 3).is_composite is True
    assert factorial(r).is_composite is None
    assert factorial(s).is_composite is None
    assert factorial(t).is_composite is None
    assert factorial(u).is_composite is None

    assert factorial(oo) is oo


def test_factorial_Mod():
    pr = Symbol('pr', prime=True)
    p, q = 10**9 + 9, 10**9 + 33 # prime modulo
    r, s = 10**7 + 5, 33333333 # composite modulo
    assert Mod(factorial(pr - 1), pr) == pr - 1
    assert Mod(factorial(pr - 1), -pr) == -1
    assert Mod(factorial(r - 1, evaluate=False), r) == 0
    assert Mod(factorial(s - 1, evaluate=False), s) == 0
    assert Mod(factorial(p - 1, evaluate=False), p) == p - 1
    assert Mod(factorial(q - 1, evaluate=False), q) == q - 1
    assert Mod(factorial(p - 50, evaluate=False), p) == 854928834
    assert Mod(factorial(q - 1800, evaluate=False), q) == 905504050
    assert Mod(factorial(153, evaluate=False), r) == Mod(factorial(153), r)
    assert Mod(factorial(255, evaluate=False), s) == Mod(factorial(255), s)
    assert Mod(factorial(4, evaluate=False), 3) == S.Zero
    assert Mod(factorial(5, evaluate=False), 6) == S.Zero


def test_factorial_diff():
    n = Symbol('n', integer=True)

    assert factorial(n).diff(n) == \
        gamma(1 + n)*polygamma(0, 1 + n)
    assert factorial(n**2).diff(n) == \
        2*n*gamma(1 + n**2)*polygamma(0, 1 + n**2)
    raises(ArgumentIndexError, lambda: factorial(n**2).fdiff(2))


def test_factorial_series():
    n = Symbol('n', integer=True)

    assert factorial(n).series(n, 0, 3) == \
        1 - n*EulerGamma + n**2*(EulerGamma**2/2 + pi**2/12) + O(n**3)


def test_factorial_rewrite():
    n = Symbol('n', integer=True)
    k = Symbol('k', integer=True, nonnegative=True)

    assert factorial(n).rewrite(gamma) == gamma(n + 1)
    _i = Dummy('i')
    assert factorial(k).rewrite(Product).dummy_eq(Product(_i, (_i, 1, k)))
    assert factorial(n).rewrite(Product) == factorial(n)


def test_factorial2():
    n = Symbol('n', integer=True)

    assert factorial2(-1) == 1
    assert factorial2(0) == 1
    assert factorial2(7) == 105
    assert factorial2(8) == 384

    # The following is exhaustive
    tt = Symbol('tt', integer=True, nonnegative=True)
    tte = Symbol('tte', even=True, nonnegative=True)
    tpe = Symbol('tpe', even=True, positive=True)
    tto = Symbol('tto', odd=True, nonnegative=True)
    tf = Symbol('tf', integer=True, nonnegative=False)
    tfe = Symbol('tfe', even=True, nonnegative=False)
    tfo = Symbol('tfo', odd=True, nonnegative=False)
    ft = Symbol('ft', integer=False, nonnegative=True)
    ff = Symbol('ff', integer=False, nonnegative=False)
    fn = Symbol('fn', integer=False)
    nt = Symbol('nt', nonnegative=True)
    nf = Symbol('nf', nonnegative=False)
    nn = Symbol('nn')
    z = Symbol('z', zero=True)
    #Solves and Fixes Issue #10388 - This is the updated test for the same solved issue
    raises(ValueError, lambda: factorial2(oo))
    raises(ValueError, lambda: factorial2(Rational(5, 2)))
    raises(ValueError, lambda: factorial2(-4))
    assert factorial2(n).is_integer is None
    assert factorial2(tt - 1).is_integer
    assert factorial2(tte - 1).is_integer
    assert factorial2(tpe - 3).is_integer
    assert factorial2(tto - 4).is_integer
    assert factorial2(tto - 2).is_integer
    assert factorial2(tf).is_integer is None
    assert factorial2(tfe).is_integer is None
    assert factorial2(tfo).is_integer is None
    assert factorial2(ft).is_integer is None
    assert factorial2(ff).is_integer is None
    assert factorial2(fn).is_integer is None
    assert factorial2(nt).is_integer is None
    assert factorial2(nf).is_integer is None
    assert factorial2(nn).is_integer is None

    assert factorial2(n).is_positive is None
    assert factorial2(tt - 1).is_positive is True
    assert factorial2(tte - 1).is_positive is True
    assert factorial2(tpe - 3).is_positive is True
    assert factorial2(tpe - 1).is_positive is True
    assert factorial2(tto - 2).is_positive is True
    assert factorial2(tto - 1).is_positive is True
    assert factorial2(tf).is_positive is None
    assert factorial2(tfe).is_positive is None
    assert factorial2(tfo).is_positive is None
    assert factorial2(ft).is_positive is None
    assert factorial2(ff).is_positive is None
    assert factorial2(fn).is_positive is None
    assert factorial2(nt).is_positive is None
    assert factorial2(nf).is_positive is None
    assert factorial2(nn).is_positive is None

    assert factorial2(tt).is_even is None
    assert factorial2(tt).is_odd is None
    assert factorial2(tte).is_even is None
    assert factorial2(tte).is_odd is None
    assert factorial2(tte + 2).is_even is True
    assert factorial2(tpe).is_even is True
    assert factorial2(tpe).is_odd is False
    assert factorial2(tto).is_odd is True
    assert factorial2(tf).is_even is None
    assert factorial2(tf).is_odd is None
    assert factorial2(tfe).is_even is None
    assert factorial2(tfe).is_odd is None
    assert factorial2(tfo).is_even is False
    assert factorial2(tfo).is_odd is None
    assert factorial2(z).is_even is False
    assert factorial2(z).is_odd is True


def test_factorial2_rewrite():
    n = Symbol('n', integer=True)
    assert factorial2(n).rewrite(gamma) == \
        2**(n/2)*Piecewise((1, Eq(Mod(n, 2), 0)), (sqrt(2)/sqrt(pi), Eq(Mod(n, 2), 1)))*gamma(n/2 + 1)
    assert factorial2(2*n).rewrite(gamma) == 2**n*gamma(n + 1)
    assert factorial2(2*n + 1).rewrite(gamma) == \
        sqrt(2)*2**(n + S.Half)*gamma(n + Rational(3, 2))/sqrt(pi)


def test_binomial():
    x = Symbol('x')
    n = Symbol('n', integer=True)
    nz = Symbol('nz', integer=True, nonzero=True)
    k = Symbol('k', integer=True)
    kp = Symbol('kp', integer=True, positive=True)
    kn = Symbol('kn', integer=True, negative=True)
    u = Symbol('u', negative=True)
    v = Symbol('v', nonnegative=True)
    p = Symbol('p', positive=True)
    z = Symbol('z', zero=True)
    nt = Symbol('nt', integer=False)
    kt = Symbol('kt', integer=False)
    a = Symbol('a', integer=True, nonnegative=True)
    b = Symbol('b', integer=True, nonnegative=True)

    assert binomial(0, 0) == 1
    assert binomial(1, 1) == 1
    assert binomial(10, 10) == 1
    assert binomial(n, z) == 1
    assert binomial(1, 2) == 0
    assert binomial(-1, 2) == 1
    assert binomial(1, -1) == 0
    assert binomial(-1, 1) == -1
    assert binomial(-1, -1) == 0
    assert binomial(S.Half, S.Half) == 1
    assert binomial(-10, 1) == -10
    assert binomial(-10, 7) == -11440
    assert binomial(n, -1) == 0 # holds for all integers (negative, zero, positive)
    assert binomial(kp, -1) == 0
    assert binomial(nz, 0) == 1
    assert expand_func(binomial(n, 1)) == n
    assert expand_func(binomial(n, 2)) == n*(n - 1)/2
    assert expand_func(binomial(n, n - 2)) == n*(n - 1)/2
    assert expand_func(binomial(n, n - 1)) == n
    assert binomial(n, 3).func == binomial
    assert binomial(n, 3).expand(func=True) ==  n**3/6 - n**2/2 + n/3
    assert expand_func(binomial(n, 3)) ==  n*(n - 2)*(n - 1)/6
    assert binomial(n, n).func == binomial # e.g. (-1, -1) == 0, (2, 2) == 1
    assert binomial(n, n + 1).func == binomial  # e.g. (-1, 0) == 1
    assert binomial(kp, kp + 1) == 0
    assert binomial(kn, kn) == 0 # issue #14529
    assert binomial(n, u).func == binomial
    assert binomial(kp, u).func == binomial
    assert binomial(n, p).func == binomial
    assert binomial(n, k).func == binomial
    assert binomial(n, n + p).func == binomial
    assert binomial(kp, kp + p).func == binomial

    assert expand_func(binomial(n, n - 3)) == n*(n - 2)*(n - 1)/6

    assert binomial(n, k).is_integer
    assert binomial(nt, k).is_integer is None
    assert binomial(x, nt).is_integer is False

    assert binomial(gamma(25), 6) == 79232165267303928292058750056084441948572511312165380965440075720159859792344339983120618959044048198214221915637090855535036339620413440000
    assert binomial(1324, 47) == 906266255662694632984994480774946083064699457235920708992926525848438478406790323869952
    assert binomial(1735, 43) == 190910140420204130794758005450919715396159959034348676124678207874195064798202216379800
    assert binomial(2512, 53) == 213894469313832631145798303740098720367984955243020898718979538096223399813295457822575338958939834177325304000
    assert binomial(3383, 52) == 27922807788818096863529701501764372757272890613101645521813434902890007725667814813832027795881839396839287659777235
    assert binomial(4321, 51) == 124595639629264868916081001263541480185227731958274383287107643816863897851139048158022599533438936036467601690983780576

    assert binomial(a, b).is_nonnegative is True
    assert binomial(-1, 2, evaluate=False).is_nonnegative is True
    assert binomial(10, 5, evaluate=False).is_nonnegative is True
    assert binomial(10, -3, evaluate=False).is_nonnegative is True
    assert binomial(-10, -3, evaluate=False).is_nonnegative is True
    assert binomial(-10, 2, evaluate=False).is_nonnegative is True
    assert binomial(-10, 1, evaluate=False).is_nonnegative is False
    assert binomial(-10, 7, evaluate=False).is_nonnegative is False

    # issue #14625
    for _ in (pi, -pi, nt, v, a):
        assert binomial(_, _) == 1
        assert binomial(_, _ - 1) == _
    assert isinstance(binomial(u, u), binomial)
    assert isinstance(binomial(u, u - 1), binomial)
    assert isinstance(binomial(x, x), binomial)
    assert isinstance(binomial(x, x - 1), binomial)

    #issue #18802
    assert expand_func(binomial(x + 1, x)) == x + 1
    assert expand_func(binomial(x, x - 1)) == x
    assert expand_func(binomial(x + 1, x - 1)) == x*(x + 1)/2
    assert expand_func(binomial(x**2 + 1, x**2)) == x**2 + 1

    # issue #13980 and #13981
    assert binomial(-7, -5) == 0
    assert binomial(-23, -12) == 0
    assert binomial(Rational(13, 2), -10) == 0
    assert binomial(-49, -51) == 0

    assert binomial(19, Rational(-7, 2)) == S(-68719476736)/(911337863661225*pi)
    assert binomial(0, Rational(3, 2)) == S(-2)/(3*pi)
    assert binomial(-3, Rational(-7, 2)) is zoo
    assert binomial(kn, kt) is zoo

    assert binomial(nt, kt).func == binomial
    assert binomial(nt, Rational(15, 6)) == 8*gamma(nt + 1)/(15*sqrt(pi)*gamma(nt - Rational(3, 2)))
    assert binomial(Rational(20, 3), Rational(-10, 8)) == gamma(Rational(23, 3))/(gamma(Rational(-1, 4))*gamma(Rational(107, 12)))
    assert binomial(Rational(19, 2), Rational(-7, 2)) == Rational(-1615, 8388608)
    assert binomial(Rational(-13, 5), Rational(-7, 8)) == gamma(Rational(-8, 5))/(gamma(Rational(-29, 40))*gamma(Rational(1, 8)))
    assert binomial(Rational(-19, 8), Rational(-13, 5)) == gamma(Rational(-11, 8))/(gamma(Rational(-8, 5))*gamma(Rational(49, 40)))

    # binomial for complexes
    assert binomial(I, Rational(-89, 8)) == gamma(1 + I)/(gamma(Rational(-81, 8))*gamma(Rational(97, 8) + I))
    assert binomial(I, 2*I) == gamma(1 + I)/(gamma(1 - I)*gamma(1 + 2*I))
    assert binomial(-7, I) is zoo
    assert binomial(Rational(-7, 6), I) == gamma(Rational(-1, 6))/(gamma(Rational(-1, 6) - I)*gamma(1 + I))
    assert binomial((1+2*I), (1+3*I)) == gamma(2 + 2*I)/(gamma(1 - I)*gamma(2 + 3*I))
    assert binomial(I, 5) == Rational(1, 3) - I/S(12)
    assert binomial((2*I + 3), 7) == -13*I/S(63)
    assert isinstance(binomial(I, n), binomial)
    assert expand_func(binomial(3, 2, evaluate=False)) == 3
    assert expand_func(binomial(n, 0, evaluate=False)) == 1
    assert expand_func(binomial(n, -2, evaluate=False)) == 0
    assert expand_func(binomial(n, k)) == binomial(n, k)


def test_binomial_Mod():
    p, q = 10**5 + 3, 10**9 + 33 # prime modulo
    r = 10**7 + 5 # composite modulo

    # A few tests to get coverage
    # Lucas Theorem
    assert Mod(binomial(156675, 4433, evaluate=False), p) == Mod(binomial(156675, 4433), p)

    # factorial Mod
    assert Mod(binomial(1234, 432, evaluate=False), q) == Mod(binomial(1234, 432), q)

    # binomial factorize
    assert Mod(binomial(253, 113, evaluate=False), r) == Mod(binomial(253, 113), r)

    # using Granville's generalisation of Lucas' Theorem
    assert Mod(binomial(10**18, 10**12, evaluate=False), p*p) == 3744312326


@slow
def test_binomial_Mod_slow():
    p, q = 10**5 + 3, 10**9 + 33 # prime modulo
    r, s = 10**7 + 5, 33333333 # composite modulo

    n, k, m = symbols('n k m')
    assert (binomial(n, k) % q).subs({n: s, k: p}) == Mod(binomial(s, p), q)
    assert (binomial(n, k) % m).subs({n: 8, k: 5, m: 13}) == 4
    assert (binomial(9, k) % 7).subs(k, 2) == 1

    # Lucas Theorem
    assert Mod(binomial(123456, 43253, evaluate=False), p) == Mod(binomial(123456, 43253), p)
    assert Mod(binomial(-178911, 237, evaluate=False), p) == Mod(-binomial(178911 + 237 - 1, 237), p)
    assert Mod(binomial(-178911, 238, evaluate=False), p) == Mod(binomial(178911 + 238 - 1, 238), p)

    # factorial Mod
    assert Mod(binomial(9734, 451, evaluate=False), q) == Mod(binomial(9734, 451), q)
    assert Mod(binomial(-10733, 4459, evaluate=False), q) == Mod(binomial(-10733, 4459), q)
    assert Mod(binomial(-15733, 4458, evaluate=False), q) == Mod(binomial(-15733, 4458), q)
    assert Mod(binomial(23, -38, evaluate=False), q) is S.Zero
    assert Mod(binomial(23, 38, evaluate=False), q) is S.Zero

    # binomial factorize
    assert Mod(binomial(753, 119, evaluate=False), r) == Mod(binomial(753, 119), r)
    assert Mod(binomial(3781, 948, evaluate=False), s) == Mod(binomial(3781, 948), s)
    assert Mod(binomial(25773, 1793, evaluate=False), s) == Mod(binomial(25773, 1793), s)
    assert Mod(binomial(-753, 118, evaluate=False), r) == Mod(binomial(-753, 118), r)
    assert Mod(binomial(-25773, 1793, evaluate=False), s) == Mod(binomial(-25773, 1793), s)


def test_binomial_diff():
    n = Symbol('n', integer=True)
    k = Symbol('k', integer=True)

    assert binomial(n, k).diff(n) == \
        (-polygamma(0, 1 + n - k) + polygamma(0, 1 + n))*binomial(n, k)
    assert binomial(n**2, k**3).diff(n) == \
        2*n*(-polygamma(
            0, 1 + n**2 - k**3) + polygamma(0, 1 + n**2))*binomial(n**2, k**3)

    assert binomial(n, k).diff(k) == \
        (-polygamma(0, 1 + k) + polygamma(0, 1 + n - k))*binomial(n, k)
    assert binomial(n**2, k**3).diff(k) == \
        3*k**2*(-polygamma(
            0, 1 + k**3) + polygamma(0, 1 + n**2 - k**3))*binomial(n**2, k**3)
    raises(ArgumentIndexError, lambda: binomial(n, k).fdiff(3))


def test_binomial_rewrite():
    n = Symbol('n', integer=True)
    k = Symbol('k', integer=True)
    x = Symbol('x')

    assert binomial(n, k).rewrite(
        factorial) == factorial(n)/(factorial(k)*factorial(n - k))
    assert binomial(
        n, k).rewrite(gamma) == gamma(n + 1)/(gamma(k + 1)*gamma(n - k + 1))
    assert binomial(n, k).rewrite(ff) == ff(n, k) / factorial(k)
    assert binomial(n, x).rewrite(ff) == binomial(n, x)


@XFAIL
def test_factorial_simplify_fail():
    # simplify(factorial(x + 1).diff(x) - ((x + 1)*factorial(x)).diff(x))) == 0
    from sympy.abc import x
    assert simplify(x*polygamma(0, x + 1) - x*polygamma(0, x + 2) +
                    polygamma(0, x + 1) - polygamma(0, x + 2) + 1) == 0


def test_subfactorial():
    assert all(subfactorial(i) == ans for i, ans in enumerate(
        [1, 0, 1, 2, 9, 44, 265, 1854, 14833, 133496]))
    assert subfactorial(oo) is oo
    assert subfactorial(nan) is nan
    assert subfactorial(23) == 9510425471055777937262
    assert unchanged(subfactorial, 2.2)

    x = Symbol('x')
    assert subfactorial(x).rewrite(uppergamma) == uppergamma(x + 1, -1)/S.Exp1

    tt = Symbol('tt', integer=True, nonnegative=True)
    tf = Symbol('tf', integer=True, nonnegative=False)
    tn = Symbol('tf', integer=True)
    ft = Symbol('ft', integer=False, nonnegative=True)
    ff = Symbol('ff', integer=False, nonnegative=False)
    fn = Symbol('ff', integer=False)
    nt = Symbol('nt', nonnegative=True)
    nf = Symbol('nf', nonnegative=False)
    nn = Symbol('nf')
    te = Symbol('te', even=True, nonnegative=True)
    to = Symbol('to', odd=True, nonnegative=True)
    assert subfactorial(tt).is_integer
    assert subfactorial(tf).is_integer is None
    assert subfactorial(tn).is_integer is None
    assert subfactorial(ft).is_integer is None
    assert subfactorial(ff).is_integer is None
    assert subfactorial(fn).is_integer is None
    assert subfactorial(nt).is_integer is None
    assert subfactorial(nf).is_integer is None
    assert subfactorial(nn).is_integer is None
    assert subfactorial(tt).is_nonnegative
    assert subfactorial(tf).is_nonnegative is None
    assert subfactorial(tn).is_nonnegative is None
    assert subfactorial(ft).is_nonnegative is None
    assert subfactorial(ff).is_nonnegative is None
    assert subfactorial(fn).is_nonnegative is None
    assert subfactorial(nt).is_nonnegative is None
    assert subfactorial(nf).is_nonnegative is None
    assert subfactorial(nn).is_nonnegative is None
    assert subfactorial(tt).is_even is None
    assert subfactorial(tt).is_odd is None
    assert subfactorial(te).is_odd is True
    assert subfactorial(to).is_even is True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\test_chaining_and_caching.py ===
from string import ascii_letters

import numpy as np
import pytest

from pandas.errors import (
    SettingWithCopyError,
    SettingWithCopyWarning,
)
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
    date_range,
    option_context,
)
import pandas._testing as tm

msg = "A value is trying to be set on a copy of a slice from a DataFrame"


def random_text(nobs=100):
    # Construct a DataFrame where each row is a random slice from 'letters'
    idxs = np.random.default_rng(2).integers(len(ascii_letters), size=(nobs, 2))
    idxs.sort(axis=1)
    strings = [ascii_letters[x[0] : x[1]] for x in idxs]

    return DataFrame(strings, columns=["letters"])


class TestCaching:
    def test_slice_consolidate_invalidate_item_cache(self, using_copy_on_write):
        # this is chained assignment, but will 'work'
        with option_context("chained_assignment", None):
            # #3970
            df = DataFrame({"aa": np.arange(5), "bb": [2.2] * 5})

            # Creates a second float block
            df["cc"] = 0.0

            # caches a reference to the 'bb' series
            df["bb"]

            # Assignment to wrong series
            with tm.raises_chained_assignment_error():
                df["bb"].iloc[0] = 0.17
            df._clear_item_cache()
            if not using_copy_on_write:
                tm.assert_almost_equal(df["bb"][0], 0.17)
            else:
                # with ArrayManager, parent is not mutated with chained assignment
                tm.assert_almost_equal(df["bb"][0], 2.2)

    @pytest.mark.parametrize("do_ref", [True, False])
    def test_setitem_cache_updating(self, do_ref):
        # GH 5424
        cont = ["one", "two", "three", "four", "five", "six", "seven"]

        df = DataFrame({"a": cont, "b": cont[3:] + cont[:3], "c": np.arange(7)})

        # ref the cache
        if do_ref:
            df.loc[0, "c"]

        # set it
        df.loc[7, "c"] = 1

        assert df.loc[0, "c"] == 0.0
        assert df.loc[7, "c"] == 1.0

    def test_setitem_cache_updating_slices(
        self, using_copy_on_write, warn_copy_on_write
    ):
        # GH 7084
        # not updating cache on series setting with slices
        expected = DataFrame(
            {"A": [600, 600, 600]}, index=date_range("5/7/2014", "5/9/2014")
        )
        out = DataFrame({"A": [0, 0, 0]}, index=date_range("5/7/2014", "5/9/2014"))
        df = DataFrame({"C": ["A", "A", "A"], "D": [100, 200, 300]})

        # loop through df to update out
        six = Timestamp("5/7/2014")
        eix = Timestamp("5/9/2014")
        for ix, row in df.iterrows():
            out.loc[six:eix, row["C"]] = out.loc[six:eix, row["C"]] + row["D"]

        tm.assert_frame_equal(out, expected)
        tm.assert_series_equal(out["A"], expected["A"])

        # try via a chain indexing
        # this actually works
        out = DataFrame({"A": [0, 0, 0]}, index=date_range("5/7/2014", "5/9/2014"))
        out_original = out.copy()
        for ix, row in df.iterrows():
            v = out[row["C"]][six:eix] + row["D"]
            with tm.raises_chained_assignment_error(
                (ix == 0) or warn_copy_on_write or using_copy_on_write
            ):
                out[row["C"]][six:eix] = v

        if not using_copy_on_write:
            tm.assert_frame_equal(out, expected)
            tm.assert_series_equal(out["A"], expected["A"])
        else:
            tm.assert_frame_equal(out, out_original)
            tm.assert_series_equal(out["A"], out_original["A"])

        out = DataFrame({"A": [0, 0, 0]}, index=date_range("5/7/2014", "5/9/2014"))
        for ix, row in df.iterrows():
            out.loc[six:eix, row["C"]] += row["D"]

        tm.assert_frame_equal(out, expected)
        tm.assert_series_equal(out["A"], expected["A"])

    def test_altering_series_clears_parent_cache(
        self, using_copy_on_write, warn_copy_on_write
    ):
        # GH #33675
        df = DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["A", "B"])
        ser = df["A"]

        if using_copy_on_write or warn_copy_on_write:
            assert "A" not in df._item_cache
        else:
            assert "A" in df._item_cache

        # Adding a new entry to ser swaps in a new array, so "A" needs to
        #  be removed from df._item_cache
        ser["c"] = 5
        assert len(ser) == 3
        assert "A" not in df._item_cache
        assert df["A"] is not ser
        assert len(df["A"]) == 2


class TestChaining:
    def test_setitem_chained_setfault(self, using_copy_on_write):
        # GH6026
        data = ["right", "left", "left", "left", "right", "left", "timeout"]
        mdata = ["right", "left", "left", "left", "right", "left", "none"]

        df = DataFrame({"response": np.array(data)})
        mask = df.response == "timeout"
        with tm.raises_chained_assignment_error():
            df.response[mask] = "none"
        if using_copy_on_write:
            tm.assert_frame_equal(df, DataFrame({"response": data}))
        else:
            tm.assert_frame_equal(df, DataFrame({"response": mdata}))

        recarray = np.rec.fromarrays([data], names=["response"])
        df = DataFrame(recarray)
        mask = df.response == "timeout"
        with tm.raises_chained_assignment_error():
            df.response[mask] = "none"
        if using_copy_on_write:
            tm.assert_frame_equal(df, DataFrame({"response": data}))
        else:
            tm.assert_frame_equal(df, DataFrame({"response": mdata}))

        df = DataFrame({"response": data, "response1": data})
        df_original = df.copy()
        mask = df.response == "timeout"
        with tm.raises_chained_assignment_error():
            df.response[mask] = "none"
        if using_copy_on_write:
            tm.assert_frame_equal(df, df_original)
        else:
            tm.assert_frame_equal(df, DataFrame({"response": mdata, "response1": data}))

        # GH 6056
        expected = DataFrame({"A": [np.nan, "bar", "bah", "foo", "bar"]})
        df = DataFrame({"A": np.array(["foo", "bar", "bah", "foo", "bar"])})
        with tm.raises_chained_assignment_error():
            df["A"].iloc[0] = np.nan
        if using_copy_on_write:
            expected = DataFrame({"A": ["foo", "bar", "bah", "foo", "bar"]})
        else:
            expected = DataFrame({"A": [np.nan, "bar", "bah", "foo", "bar"]})
        result = df.head()
        tm.assert_frame_equal(result, expected)

        df = DataFrame({"A": np.array(["foo", "bar", "bah", "foo", "bar"])})
        with tm.raises_chained_assignment_error():
            df.A.iloc[0] = np.nan
        result = df.head()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.arm_slow
    def test_detect_chained_assignment(self, using_copy_on_write):
        with option_context("chained_assignment", "raise"):
            # work with the chain
            expected = DataFrame([[-5, 1], [-6, 3]], columns=list("AB"))
            df = DataFrame(
                np.arange(4).reshape(2, 2), columns=list("AB"), dtype="int64"
            )
            df_original = df.copy()
            assert df._is_copy is None

            with tm.raises_chained_assignment_error():
                df["A"][0] = -5
            with tm.raises_chained_assignment_error():
                df["A"][1] = -6
            if using_copy_on_write:
                tm.assert_frame_equal(df, df_original)
            else:
                tm.assert_frame_equal(df, expected)

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_raises(
        self, using_array_manager, using_copy_on_write, warn_copy_on_write
    ):
        # test with the chaining
        df = DataFrame(
            {
                "A": Series(range(2), dtype="int64"),
                "B": np.array(np.arange(2, 4), dtype=np.float64),
            }
        )
        df_original = df.copy()
        assert df._is_copy is None

        if using_copy_on_write:
            with tm.raises_chained_assignment_error():
                df["A"][0] = -5
            with tm.raises_chained_assignment_error():
                df["A"][1] = -6
            tm.assert_frame_equal(df, df_original)
        elif warn_copy_on_write:
            with tm.raises_chained_assignment_error():
                df["A"][0] = -5
            with tm.raises_chained_assignment_error():
                df["A"][1] = np.nan
        elif not using_array_manager:
            with pytest.raises(SettingWithCopyError, match=msg):
                with tm.raises_chained_assignment_error():
                    df["A"][0] = -5

            with pytest.raises(SettingWithCopyError, match=msg):
                with tm.raises_chained_assignment_error():
                    df["A"][1] = np.nan

            assert df["A"]._is_copy is None
        else:
            # INFO(ArrayManager) for ArrayManager it doesn't matter that it's
            # a mixed dataframe
            df["A"][0] = -5
            df["A"][1] = -6
            expected = DataFrame([[-5, 2], [-6, 3]], columns=list("AB"))
            expected["B"] = expected["B"].astype("float64")
            tm.assert_frame_equal(df, expected)

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_fails(
        self, using_copy_on_write, warn_copy_on_write
    ):
        # Using a copy (the chain), fails
        df = DataFrame(
            {
                "A": Series(range(2), dtype="int64"),
                "B": np.array(np.arange(2, 4), dtype=np.float64),
            }
        )

        if using_copy_on_write or warn_copy_on_write:
            with tm.raises_chained_assignment_error():
                df.loc[0]["A"] = -5
        else:
            with pytest.raises(SettingWithCopyError, match=msg):
                df.loc[0]["A"] = -5

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_doc_example(
        self, using_copy_on_write, warn_copy_on_write
    ):
        # Doc example
        df = DataFrame(
            {
                "a": ["one", "one", "two", "three", "two", "one", "six"],
                "c": Series(range(7), dtype="int64"),
            }
        )
        assert df._is_copy is None

        indexer = df.a.str.startswith("o")
        if using_copy_on_write or warn_copy_on_write:
            with tm.raises_chained_assignment_error():
                df[indexer]["c"] = 42
        else:
            with pytest.raises(SettingWithCopyError, match=msg):
                df[indexer]["c"] = 42

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_object_dtype(
        self, using_array_manager, using_copy_on_write, warn_copy_on_write
    ):
        expected = DataFrame({"A": [111, "bbb", "ccc"], "B": [1, 2, 3]})
        df = DataFrame(
            {"A": Series(["aaa", "bbb", "ccc"], dtype=object), "B": [1, 2, 3]}
        )
        df_original = df.copy()

        if not using_copy_on_write and not warn_copy_on_write:
            with pytest.raises(SettingWithCopyError, match=msg):
                df.loc[0]["A"] = 111

        if using_copy_on_write:
            with tm.raises_chained_assignment_error():
                df["A"][0] = 111
            tm.assert_frame_equal(df, df_original)
        elif warn_copy_on_write:
            with tm.raises_chained_assignment_error():
                df["A"][0] = 111
            tm.assert_frame_equal(df, expected)
        elif not using_array_manager:
            with pytest.raises(SettingWithCopyError, match=msg):
                with tm.raises_chained_assignment_error():
                    df["A"][0] = 111

            df.loc[0, "A"] = 111
            tm.assert_frame_equal(df, expected)
        else:
            # INFO(ArrayManager) for ArrayManager it doesn't matter that it's
            # a mixed dataframe
            df["A"][0] = 111
            tm.assert_frame_equal(df, expected)

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_is_copy_pickle(self):
        # gh-5475: Make sure that is_copy is picked up reconstruction
        df = DataFrame({"A": [1, 2]})
        assert df._is_copy is None

        with tm.ensure_clean("__tmp__pickle") as path:
            df.to_pickle(path)
            df2 = pd.read_pickle(path)
            df2["B"] = df2["A"]
            df2["B"] = df2["A"]

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_setting_entire_column(self):
        # gh-5597: a spurious raise as we are setting the entire column here

        df = random_text(100000)

        # Always a copy
        x = df.iloc[[0, 1, 2]]
        assert x._is_copy is not None

        x = df.iloc[[0, 1, 2, 4]]
        assert x._is_copy is not None

        # Explicitly copy
        indexer = df.letters.apply(lambda x: len(x) > 10)
        df = df.loc[indexer].copy()

        assert df._is_copy is None
        df["letters"] = df["letters"].apply(str.lower)

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_implicit_take(self):
        # Implicitly take
        df = random_text(100000)
        indexer = df.letters.apply(lambda x: len(x) > 10)
        df = df.loc[indexer]

        assert df._is_copy is not None
        df["letters"] = df["letters"].apply(str.lower)

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_implicit_take2(
        self, using_copy_on_write, warn_copy_on_write
    ):
        if using_copy_on_write or warn_copy_on_write:
            pytest.skip("_is_copy is not always set for CoW")
        # Implicitly take 2
        df = random_text(100000)
        indexer = df.letters.apply(lambda x: len(x) > 10)

        df = df.loc[indexer]
        assert df._is_copy is not None
        df.loc[:, "letters"] = df["letters"].apply(str.lower)

        # with the enforcement of #45333 in 2.0, the .loc[:, letters] setting
        #  is inplace, so df._is_copy remains non-None.
        assert df._is_copy is not None

        df["letters"] = df["letters"].apply(str.lower)
        assert df._is_copy is None

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_str(self):
        df = random_text(100000)
        indexer = df.letters.apply(lambda x: len(x) > 10)
        df.loc[indexer, "letters"] = df.loc[indexer, "letters"].apply(str.lower)

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_is_copy(self):
        # an identical take, so no copy
        df = DataFrame({"a": [1]}).dropna()
        assert df._is_copy is None
        df["a"] += 1

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_sorting(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        ser = df.iloc[:, 0].sort_values()

        tm.assert_series_equal(ser, df.iloc[:, 0].sort_values())
        tm.assert_series_equal(ser, df[0].sort_values())

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_false_positives(self):
        # see gh-6025: false positives
        df = DataFrame({"column1": ["a", "a", "a"], "column2": [4, 8, 9]})
        str(df)

        df["column1"] = df["column1"] + "b"
        str(df)

        df = df[df["column2"] != 8]
        str(df)

        df["column1"] = df["column1"] + "c"
        str(df)

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_undefined_column(
        self, using_copy_on_write, warn_copy_on_write
    ):
        # from SO:
        # https://stackoverflow.com/questions/24054495/potential-bug-setting-value-for-undefined-column-using-iloc
        df = DataFrame(np.arange(0, 9), columns=["count"])
        df["group"] = "b"
        df_original = df.copy()

        if using_copy_on_write:
            with tm.raises_chained_assignment_error():
                df.iloc[0:5]["group"] = "a"
            tm.assert_frame_equal(df, df_original)
        elif warn_copy_on_write:
            with tm.raises_chained_assignment_error():
                df.iloc[0:5]["group"] = "a"
        else:
            with pytest.raises(SettingWithCopyError, match=msg):
                with tm.raises_chained_assignment_error():
                    df.iloc[0:5]["group"] = "a"

    @pytest.mark.arm_slow
    def test_detect_chained_assignment_changing_dtype(
        self, using_array_manager, using_copy_on_write, warn_copy_on_write
    ):
        # Mixed type setting but same dtype & changing dtype
        df = DataFrame(
            {
                "A": date_range("20130101", periods=5),
                "B": np.random.default_rng(2).standard_normal(5),
                "C": np.arange(5, dtype="int64"),
                "D": ["a", "b", "c", "d", "e"],
            }
        )
        df_original = df.copy()

        if using_copy_on_write or warn_copy_on_write:
            with tm.raises_chained_assignment_error():
                df.loc[2]["D"] = "foo"
            with tm.raises_chained_assignment_error():
                df.loc[2]["C"] = "foo"
            tm.assert_frame_equal(df, df_original)
            with tm.raises_chained_assignment_error(extra_warnings=(FutureWarning,)):
                df["C"][2] = "foo"
            if using_copy_on_write:
                tm.assert_frame_equal(df, df_original)
            else:
                assert df.loc[2, "C"] == "foo"
        else:
            with pytest.raises(SettingWithCopyError, match=msg):
                df.loc[2]["D"] = "foo"

            with pytest.raises(SettingWithCopyError, match=msg):
                df.loc[2]["C"] = "foo"

            if not using_array_manager:
                with pytest.raises(SettingWithCopyError, match=msg):
                    with tm.raises_chained_assignment_error():
                        df["C"][2] = "foo"
            else:
                # INFO(ArrayManager) for ArrayManager it doesn't matter if it's
                # changing the dtype or not
                df["C"][2] = "foo"
                assert df.loc[2, "C"] == "foo"

    def test_setting_with_copy_bug(self, using_copy_on_write, warn_copy_on_write):
        # operating on a copy
        df = DataFrame(
            {"a": list(range(4)), "b": list("ab.."), "c": ["a", "b", np.nan, "d"]}
        )
        df_original = df.copy()
        mask = pd.isna(df.c)

        if using_copy_on_write:
            with tm.raises_chained_assignment_error():
                df[["c"]][mask] = df[["b"]][mask]
            tm.assert_frame_equal(df, df_original)
        elif warn_copy_on_write:
            with tm.raises_chained_assignment_error():
                df[["c"]][mask] = df[["b"]][mask]
        else:
            with pytest.raises(SettingWithCopyError, match=msg):
                df[["c"]][mask] = df[["b"]][mask]

    def test_setting_with_copy_bug_no_warning(self):
        # invalid warning as we are returning a new object
        # GH 8730
        df1 = DataFrame({"x": Series(["a", "b", "c"]), "y": Series(["d", "e", "f"])})
        df2 = df1[["x"]]

        # this should not raise
        df2["y"] = ["g", "h", "i"]

    def test_detect_chained_assignment_warnings_errors(
        self, using_copy_on_write, warn_copy_on_write
    ):
        df = DataFrame({"A": ["aaa", "bbb", "ccc"], "B": [1, 2, 3]})
        if using_copy_on_write or warn_copy_on_write:
            with tm.raises_chained_assignment_error():
                df.loc[0]["A"] = 111
            return

        with option_context("chained_assignment", "warn"):
            with tm.assert_produces_warning(SettingWithCopyWarning):
                df.loc[0]["A"] = 111

        with option_context("chained_assignment", "raise"):
            with pytest.raises(SettingWithCopyError, match=msg):
                df.loc[0]["A"] = 111

    @pytest.mark.parametrize("rhs", [3, DataFrame({0: [1, 2, 3, 4]})])
    def test_detect_chained_assignment_warning_stacklevel(
        self, rhs, using_copy_on_write, warn_copy_on_write
    ):
        # GH#42570
        df = DataFrame(np.arange(25).reshape(5, 5))
        df_original = df.copy()
        chained = df.loc[:3]
        with option_context("chained_assignment", "warn"):
            if not using_copy_on_write and not warn_copy_on_write:
                with tm.assert_produces_warning(SettingWithCopyWarning) as t:
                    chained[2] = rhs
                    assert t[0].filename == __file__
            else:
                # INFO(CoW) no warning, and original dataframe not changed
                chained[2] = rhs
                tm.assert_frame_equal(df, df_original)

    # TODO(ArrayManager) fast_xs with array-like scalars is not yet working
    @td.skip_array_manager_not_yet_implemented
    def test_chained_getitem_with_lists(self):
        # GH6394
        # Regression in chained getitem indexing with embedded list-like from
        # 0.12

        df = DataFrame({"A": 5 * [np.zeros(3)], "B": 5 * [np.ones(3)]})
        expected = df["A"].iloc[2]
        result = df.loc[2, "A"]
        tm.assert_numpy_array_equal(result, expected)
        result2 = df.iloc[2]["A"]
        tm.assert_numpy_array_equal(result2, expected)
        result3 = df["A"].loc[2]
        tm.assert_numpy_array_equal(result3, expected)
        result4 = df["A"].iloc[2]
        tm.assert_numpy_array_equal(result4, expected)

    def test_cache_updating(self):
        # GH 4939, make sure to update the cache on setitem

        df = DataFrame(
            np.zeros((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
        )
        df["A"]  # cache series
        df.loc["Hello Friend"] = df.iloc[0]
        assert "Hello Friend" in df["A"].index
        assert "Hello Friend" in df["B"].index

    def test_cache_updating2(self, using_copy_on_write):
        # 10264
        df = DataFrame(
            np.zeros((5, 5), dtype="int64"),
            columns=["a", "b", "c", "d", "e"],
            index=range(5),
        )
        df["f"] = 0
        df_orig = df.copy()
        if using_copy_on_write:
            with pytest.raises(ValueError, match="read-only"):
                df.f.values[3] = 1
            tm.assert_frame_equal(df, df_orig)
            return

        df.f.values[3] = 1

        df.f.values[3] = 2
        expected = DataFrame(
            np.zeros((5, 6), dtype="int64"),
            columns=["a", "b", "c", "d", "e", "f"],
            index=range(5),
        )
        expected.at[3, "f"] = 2
        tm.assert_frame_equal(df, expected)
        expected = Series([0, 0, 0, 2, 0], name="f")
        tm.assert_series_equal(df.f, expected)

    def test_iloc_setitem_chained_assignment(self, using_copy_on_write):
        # GH#3970
        with option_context("chained_assignment", None):
            df = DataFrame({"aa": range(5), "bb": [2.2] * 5})
            df["cc"] = 0.0

            ck = [True] * len(df)

            with tm.raises_chained_assignment_error():
                df["bb"].iloc[0] = 0.13

            # GH#3970 this lookup used to break the chained setting to 0.15
            df.iloc[ck]

            with tm.raises_chained_assignment_error():
                df["bb"].iloc[0] = 0.15

            if not using_copy_on_write:
                assert df["bb"].iloc[0] == 0.15
            else:
                assert df["bb"].iloc[0] == 2.2

    def test_getitem_loc_assignment_slice_state(self):
        # GH 13569
        df = DataFrame({"a": [10, 20, 30]})
        with tm.raises_chained_assignment_error():
            df["a"].loc[4] = 40
        tm.assert_frame_equal(df, DataFrame({"a": [10, 20, 30]}))
        tm.assert_series_equal(df["a"], Series([10, 20, 30], name="a"))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_c_parser_only.py ===
"""
Tests that apply specifically to the CParser. Unless specifically stated
as a CParser-specific issue, the goal is to eventually move as many of
these tests out of this module as soon as the Python parser can accept
further arguments when parsing.
"""
from decimal import Decimal
from io import (
    BytesIO,
    StringIO,
    TextIOWrapper,
)
import mmap
import os
import tarfile

import numpy as np
import pytest

from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import (
    ParserError,
    ParserWarning,
)
import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    concat,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "malformed",
    ["1\r1\r1\r 1\r 1\r", "1\r1\r1\r 1\r 1\r11\r", "1\r1\r1\r 1\r 1\r11\r1\r"],
    ids=["words pointer", "stream pointer", "lines pointer"],
)
def test_buffer_overflow(c_parser_only, malformed):
    # see gh-9205: test certain malformed input files that cause
    # buffer overflows in tokenizer.c
    msg = "Buffer overflow caught - possible malformed input file."
    parser = c_parser_only

    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(malformed))


def test_delim_whitespace_custom_terminator(c_parser_only):
    # See gh-12912
    data = "a b c~1 2 3~4 5 6~7 8 9"
    parser = c_parser_only

    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        df = parser.read_csv(StringIO(data), lineterminator="~", delim_whitespace=True)
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"])
    tm.assert_frame_equal(df, expected)


def test_dtype_and_names_error(c_parser_only):
    # see gh-8833: passing both dtype and names
    # resulting in an error reporting issue
    parser = c_parser_only
    data = """
1.0 1
2.0 2
3.0 3
"""
    # base cases
    result = parser.read_csv(StringIO(data), sep=r"\s+", header=None)
    expected = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3]])
    tm.assert_frame_equal(result, expected)

    result = parser.read_csv(StringIO(data), sep=r"\s+", header=None, names=["a", "b"])
    expected = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)

    # fallback casting
    result = parser.read_csv(
        StringIO(data), sep=r"\s+", header=None, names=["a", "b"], dtype={"a": np.int32}
    )
    expected = DataFrame([[1, 1], [2, 2], [3, 3]], columns=["a", "b"])
    expected["a"] = expected["a"].astype(np.int32)
    tm.assert_frame_equal(result, expected)

    data = """
1.0 1
nan 2
3.0 3
"""
    # fallback casting, but not castable
    warning = RuntimeWarning if np_version_gte1p24 else None
    with pytest.raises(ValueError, match="cannot safely convert"):
        with tm.assert_produces_warning(warning, check_stacklevel=False):
            parser.read_csv(
                StringIO(data),
                sep=r"\s+",
                header=None,
                names=["a", "b"],
                dtype={"a": np.int32},
            )


@pytest.mark.parametrize(
    "match,kwargs",
    [
        # For each of these cases, all of the dtypes are valid, just unsupported.
        (
            (
                "the dtype datetime64 is not supported for parsing, "
                "pass this column using parse_dates instead"
            ),
            {"dtype": {"A": "datetime64", "B": "float64"}},
        ),
        (
            (
                "the dtype datetime64 is not supported for parsing, "
                "pass this column using parse_dates instead"
            ),
            {"dtype": {"A": "datetime64", "B": "float64"}, "parse_dates": ["B"]},
        ),
        (
            "the dtype timedelta64 is not supported for parsing",
            {"dtype": {"A": "timedelta64", "B": "float64"}},
        ),
        (
            f"the dtype {tm.ENDIAN}U8 is not supported for parsing",
            {"dtype": {"A": "U8"}},
        ),
    ],
    ids=["dt64-0", "dt64-1", "td64", f"{tm.ENDIAN}U8"],
)
def test_unsupported_dtype(c_parser_only, match, kwargs):
    parser = c_parser_only
    df = DataFrame(
        np.random.default_rng(2).random((5, 2)),
        columns=list("AB"),
        index=["1A", "1B", "1C", "1D", "1E"],
    )

    with tm.ensure_clean("__unsupported_dtype__.csv") as path:
        df.to_csv(path)

        with pytest.raises(TypeError, match=match):
            parser.read_csv(path, index_col=0, **kwargs)


@td.skip_if_32bit
@pytest.mark.slow
# test numbers between 1 and 2
@pytest.mark.parametrize("num", np.linspace(1.0, 2.0, num=21))
def test_precise_conversion(c_parser_only, num):
    parser = c_parser_only

    normal_errors = []
    precise_errors = []

    def error(val: float, actual_val: Decimal) -> Decimal:
        return abs(Decimal(f"{val:.100}") - actual_val)

    # 25 decimal digits of precision
    text = f"a\n{num:.25}"

    normal_val = float(
        parser.read_csv(StringIO(text), float_precision="legacy")["a"][0]
    )
    precise_val = float(parser.read_csv(StringIO(text), float_precision="high")["a"][0])
    roundtrip_val = float(
        parser.read_csv(StringIO(text), float_precision="round_trip")["a"][0]
    )
    actual_val = Decimal(text[2:])

    normal_errors.append(error(normal_val, actual_val))
    precise_errors.append(error(precise_val, actual_val))

    # round-trip should match float()
    assert roundtrip_val == float(text[2:])

    assert sum(precise_errors) <= sum(normal_errors)
    assert max(precise_errors) <= max(normal_errors)


def test_usecols_dtypes(c_parser_only, using_infer_string):
    parser = c_parser_only
    data = """\
1,2,3
4,5,6
7,8,9
10,11,12"""

    result = parser.read_csv(
        StringIO(data),
        usecols=(0, 1, 2),
        names=("a", "b", "c"),
        header=None,
        converters={"a": str},
        dtype={"b": int, "c": float},
    )
    result2 = parser.read_csv(
        StringIO(data),
        usecols=(0, 2),
        names=("a", "b", "c"),
        header=None,
        converters={"a": str},
        dtype={"b": int, "c": float},
    )

    if using_infer_string:
        assert (result.dtypes == ["string", int, float]).all()
        assert (result2.dtypes == ["string", float]).all()
    else:
        assert (result.dtypes == [object, int, float]).all()
        assert (result2.dtypes == [object, float]).all()


def test_disable_bool_parsing(c_parser_only):
    # see gh-2090

    parser = c_parser_only
    data = """A,B,C
Yes,No,Yes
No,Yes,Yes
Yes,,Yes
No,No,No"""

    result = parser.read_csv(StringIO(data), dtype=object)
    assert (result.dtypes == object).all()

    result = parser.read_csv(StringIO(data), dtype=object, na_filter=False)
    assert result["B"][2] == ""


def test_custom_lineterminator(c_parser_only):
    parser = c_parser_only
    data = "a,b,c~1,2,3~4,5,6"

    result = parser.read_csv(StringIO(data), lineterminator="~")
    expected = parser.read_csv(StringIO(data.replace("~", "\n")))

    tm.assert_frame_equal(result, expected)


def test_parse_ragged_csv(c_parser_only):
    parser = c_parser_only
    data = """1,2,3
1,2,3,4
1,2,3,4,5
1,2
1,2,3,4"""

    nice_data = """1,2,3,,
1,2,3,4,
1,2,3,4,5
1,2,,,
1,2,3,4,"""
    result = parser.read_csv(
        StringIO(data), header=None, names=["a", "b", "c", "d", "e"]
    )

    expected = parser.read_csv(
        StringIO(nice_data), header=None, names=["a", "b", "c", "d", "e"]
    )

    tm.assert_frame_equal(result, expected)

    # too many columns, cause segfault if not careful
    data = "1,2\n3,4,5"

    result = parser.read_csv(StringIO(data), header=None, names=range(50))
    expected = parser.read_csv(StringIO(data), header=None, names=range(3)).reindex(
        columns=range(50)
    )

    tm.assert_frame_equal(result, expected)


def test_tokenize_CR_with_quoting(c_parser_only):
    # see gh-3453
    parser = c_parser_only
    data = ' a,b,c\r"a,b","e,d","f,f"'

    result = parser.read_csv(StringIO(data), header=None)
    expected = parser.read_csv(StringIO(data.replace("\r", "\n")), header=None)
    tm.assert_frame_equal(result, expected)

    result = parser.read_csv(StringIO(data))
    expected = parser.read_csv(StringIO(data.replace("\r", "\n")))
    tm.assert_frame_equal(result, expected)


@pytest.mark.slow
@pytest.mark.parametrize("count", [3 * 2**n for n in range(6)])
def test_grow_boundary_at_cap(c_parser_only, count):
    # See gh-12494
    #
    # Cause of error was that the C parser
    # was not increasing the buffer size when
    # the desired space would fill the buffer
    # to capacity, which would later cause a
    # buffer overflow error when checking the
    # EOF terminator of the CSV stream.
    # 3 * 2^n commas was observed to break the parser
    parser = c_parser_only

    with StringIO("," * count) as s:
        expected = DataFrame(columns=[f"Unnamed: {i}" for i in range(count + 1)])
        df = parser.read_csv(s)
    tm.assert_frame_equal(df, expected)


@pytest.mark.slow
@pytest.mark.parametrize("encoding", [None, "utf-8"])
def test_parse_trim_buffers(c_parser_only, encoding):
    # This test is part of a bugfix for gh-13703. It attempts to
    # to stress the system memory allocator, to cause it to move the
    # stream buffer and either let the OS reclaim the region, or let
    # other memory requests of parser otherwise modify the contents
    # of memory space, where it was formally located.
    # This test is designed to cause a `segfault` with unpatched
    # `tokenizer.c`. Sometimes the test fails on `segfault`, other
    # times it fails due to memory corruption, which causes the
    # loaded DataFrame to differ from the expected one.

    # Also force 'utf-8' encoding, so that `_string_convert` would take
    # a different execution branch.

    parser = c_parser_only

    # Generate a large mixed-type CSV file on-the-fly (one record is
    # approx 1.5KiB).
    record_ = (
        """9999-9,99:99,,,,ZZ,ZZ,,,ZZZ-ZZZZ,.Z-ZZZZ,-9.99,,,9.99,Z"""
        """ZZZZ,,-99,9,ZZZ-ZZZZ,ZZ-ZZZZ,,9.99,ZZZ-ZZZZZ,ZZZ-ZZZZZ,"""
        """ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,9"""
        """99,ZZZ-ZZZZ,,ZZ-ZZZZ,,,,,ZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZ,,,9,9,"""
        """9,9,99,99,999,999,ZZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZ,9,ZZ-ZZZZ,9."""
        """99,ZZ-ZZZZ,ZZ-ZZZZ,,,,ZZZZ,,,ZZ,ZZ,,,,,,,,,,,,,9,,,999."""
        """99,999.99,,,ZZZZZ,,,Z9,,,,,,,ZZZ,ZZZ,,,,,,,,,,,ZZZZZ,ZZ"""
        """ZZZ,ZZZ-ZZZZZZ,ZZZ-ZZZZZZ,ZZ-ZZZZ,ZZ-ZZZZ,ZZ-ZZZZ,ZZ-ZZ"""
        """ZZ,,,999999,999999,ZZZ,ZZZ,,,ZZZ,ZZZ,999.99,999.99,,,,Z"""
        """ZZ-ZZZ,ZZZ-ZZZ,-9.99,-9.99,9,9,,99,,9.99,9.99,9,9,9.99,"""
        """9.99,,,,9.99,9.99,,99,,99,9.99,9.99,,,ZZZ,ZZZ,,999.99,,"""
        """999.99,ZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,,,ZZZZZ,ZZZZZ,ZZZ,ZZZ,9,9,"""
        """,,,,,ZZZ-ZZZZ,ZZZ999Z,,,999.99,,999.99,ZZZ-ZZZZ,,,9.999"""
        """,9.999,9.999,9.999,-9.999,-9.999,-9.999,-9.999,9.999,9."""
        """999,9.999,9.999,9.999,9.999,9.999,9.999,99999,ZZZ-ZZZZ,"""
        """,9.99,ZZZ,,,,,,,,ZZZ,,,,,9,,,,9,,,,,,,,,,ZZZ-ZZZZ,ZZZ-Z"""
        """ZZZ,,ZZZZZ,ZZZZZ,ZZZZZ,ZZZZZ,,,9.99,,ZZ-ZZZZ,ZZ-ZZZZ,ZZ"""
        """,999,,,,ZZ-ZZZZ,ZZZ,ZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,,,99.99,99.99"""
        """,,,9.99,9.99,9.99,9.99,ZZZ-ZZZZ,,,ZZZ-ZZZZZ,,,,,-9.99,-"""
        """9.99,-9.99,-9.99,,,,,,,,,ZZZ-ZZZZ,,9,9.99,9.99,99ZZ,,-9"""
        """.99,-9.99,ZZZ-ZZZZ,,,,,,,ZZZ-ZZZZ,9.99,9.99,9999,,,,,,,"""
        """,,,-9.9,Z/Z-ZZZZ,999.99,9.99,,999.99,ZZ-ZZZZ,ZZ-ZZZZ,9."""
        """99,9.99,9.99,9.99,9.99,9.99,,ZZZ-ZZZZZ,ZZZ-ZZZZZ,ZZZ-ZZ"""
        """ZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZZ,ZZZ,ZZZ,ZZZ,ZZZ,9.99,,,-9.99,ZZ"""
        """-ZZZZ,-999.99,,-9999,,999.99,,,,999.99,99.99,,,ZZ-ZZZZZ"""
        """ZZZ,ZZ-ZZZZ-ZZZZZZZ,,,,ZZ-ZZ-ZZZZZZZZ,ZZZZZZZZ,ZZZ-ZZZZ"""
        """,9999,999.99,ZZZ-ZZZZ,-9.99,-9.99,ZZZ-ZZZZ,99:99:99,,99"""
        """,99,,9.99,,-99.99,,,,,,9.99,ZZZ-ZZZZ,-9.99,-9.99,9.99,9"""
        """.99,,ZZZ,,,,,,,ZZZ,ZZZ,,,,,"""
    )

    # Set the number of lines so that a call to `parser_trim_buffers`
    # is triggered: after a couple of full chunks are consumed a
    # relatively small 'residual' chunk would cause reallocation
    # within the parser.
    chunksize, n_lines = 128, 2 * 128 + 15
    csv_data = "\n".join([record_] * n_lines) + "\n"

    # We will use StringIO to load the CSV from this text buffer.
    # pd.read_csv() will iterate over the file in chunks and will
    # finally read a residual chunk of really small size.

    # Generate the expected output: manually create the dataframe
    # by splitting by comma and repeating the `n_lines` times.
    row = tuple(val_ if val_ else np.nan for val_ in record_.split(","))
    expected = DataFrame(
        [row for _ in range(n_lines)], dtype=object, columns=None, index=None
    )

    # Iterate over the CSV file in chunks of `chunksize` lines
    with parser.read_csv(
        StringIO(csv_data),
        header=None,
        dtype=object,
        chunksize=chunksize,
        encoding=encoding,
    ) as chunks_:
        result = concat(chunks_, axis=0, ignore_index=True)

    # Check for data corruption if there was no segfault
    tm.assert_frame_equal(result, expected)


def test_internal_null_byte(c_parser_only):
    # see gh-14012
    #
    # The null byte ('\x00') should not be used as a
    # true line terminator, escape character, or comment
    # character, only as a placeholder to indicate that
    # none was specified.
    #
    # This test should be moved to test_common.py ONLY when
    # Python's csv class supports parsing '\x00'.
    parser = c_parser_only

    names = ["a", "b", "c"]
    data = "1,2,3\n4,\x00,6\n7,8,9"
    expected = DataFrame([[1, 2.0, 3], [4, np.nan, 6], [7, 8, 9]], columns=names)

    result = parser.read_csv(StringIO(data), names=names)
    tm.assert_frame_equal(result, expected)


def test_read_nrows_large(c_parser_only):
    # gh-7626 - Read only nrows of data in for large inputs (>262144b)
    parser = c_parser_only
    header_narrow = "\t".join(["COL_HEADER_" + str(i) for i in range(10)]) + "\n"
    data_narrow = "\t".join(["somedatasomedatasomedata1" for _ in range(10)]) + "\n"
    header_wide = "\t".join(["COL_HEADER_" + str(i) for i in range(15)]) + "\n"
    data_wide = "\t".join(["somedatasomedatasomedata2" for _ in range(15)]) + "\n"
    test_input = header_narrow + data_narrow * 1050 + header_wide + data_wide * 2

    df = parser.read_csv(StringIO(test_input), sep="\t", nrows=1010)

    assert df.size == 1010 * 10


def test_float_precision_round_trip_with_text(c_parser_only):
    # see gh-15140
    parser = c_parser_only
    df = parser.read_csv(StringIO("a"), header=None, float_precision="round_trip")
    tm.assert_frame_equal(df, DataFrame({0: ["a"]}))


def test_large_difference_in_columns(c_parser_only):
    # see gh-14125
    parser = c_parser_only

    count = 10000
    large_row = ("X," * count)[:-1] + "\n"
    normal_row = "XXXXXX XXXXXX,111111111111111\n"
    test_input = (large_row + normal_row * 6)[:-1]

    result = parser.read_csv(StringIO(test_input), header=None, usecols=[0])
    rows = test_input.split("\n")

    expected = DataFrame([row.split(",")[0] for row in rows])
    tm.assert_frame_equal(result, expected)


def test_data_after_quote(c_parser_only):
    # see gh-15910
    parser = c_parser_only

    data = 'a\n1\n"b"a'
    result = parser.read_csv(StringIO(data))

    expected = DataFrame({"a": ["1", "ba"]})
    tm.assert_frame_equal(result, expected)


def test_comment_whitespace_delimited(c_parser_only):
    parser = c_parser_only
    test_input = """\
1 2
2 2 3
3 2 3 # 3 fields
4 2 3# 3 fields
5 2 # 2 fields
6 2# 2 fields
7 # 1 field, NaN
8# 1 field, NaN
9 2 3 # skipped line
# comment"""
    with tm.assert_produces_warning(
        ParserWarning, match="Skipping line", check_stacklevel=False
    ):
        df = parser.read_csv(
            StringIO(test_input),
            comment="#",
            header=None,
            delimiter="\\s+",
            skiprows=0,
            on_bad_lines="warn",
        )
    expected = DataFrame([[1, 2], [5, 2], [6, 2], [7, np.nan], [8, np.nan]])
    tm.assert_frame_equal(df, expected)


def test_file_like_no_next(c_parser_only):
    # gh-16530: the file-like need not have a "next" or "__next__"
    # attribute despite having an "__iter__" attribute.
    #
    # NOTE: This is only true for the C engine, not Python engine.
    class NoNextBuffer(StringIO):
        def __next__(self):
            raise AttributeError("No next method")

        next = __next__

    parser = c_parser_only
    data = "a\n1"

    expected = DataFrame({"a": [1]})
    result = parser.read_csv(NoNextBuffer(data))

    tm.assert_frame_equal(result, expected)


def test_buffer_rd_bytes_bad_unicode(c_parser_only):
    # see gh-22748
    t = BytesIO(b"\xB0")
    t = TextIOWrapper(t, encoding="ascii", errors="surrogateescape")
    msg = "'utf-8' codec can't encode character"
    with pytest.raises(UnicodeError, match=msg):
        c_parser_only.read_csv(t, encoding="UTF-8")


@pytest.mark.parametrize("tar_suffix", [".tar", ".tar.gz"])
def test_read_tarfile(c_parser_only, csv_dir_path, tar_suffix):
    # see gh-16530
    #
    # Unfortunately, Python's CSV library can't handle
    # tarfile objects (expects string, not bytes when
    # iterating through a file-like).
    parser = c_parser_only
    tar_path = os.path.join(csv_dir_path, "tar_csv" + tar_suffix)

    with tarfile.open(tar_path, "r") as tar:
        data_file = tar.extractfile("tar_data.csv")

        out = parser.read_csv(data_file)
        expected = DataFrame({"a": [1]})
        tm.assert_frame_equal(out, expected)


def test_chunk_whitespace_on_boundary(c_parser_only):
    # see gh-9735: this issue is C parser-specific (bug when
    # parsing whitespace and characters at chunk boundary)
    #
    # This test case has a field too large for the Python parser / CSV library.
    parser = c_parser_only

    chunk1 = "a" * (1024 * 256 - 2) + "\na"
    chunk2 = "\n a"
    result = parser.read_csv(StringIO(chunk1 + chunk2), header=None)

    expected = DataFrame(["a" * (1024 * 256 - 2), "a", " a"])
    tm.assert_frame_equal(result, expected)


def test_file_handles_mmap(c_parser_only, csv1):
    # gh-14418
    #
    # Don't close user provided file handles.
    parser = c_parser_only

    with open(csv1, encoding="utf-8") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            parser.read_csv(m)
            assert not m.closed


def test_file_binary_mode(c_parser_only):
    # see gh-23779
    parser = c_parser_only
    expected = DataFrame([[1, 2, 3], [4, 5, 6]])

    with tm.ensure_clean() as path:
        with open(path, "w", encoding="utf-8") as f:
            f.write("1,2,3\n4,5,6")

        with open(path, "rb") as f:
            result = parser.read_csv(f, header=None)
            tm.assert_frame_equal(result, expected)


def test_unix_style_breaks(c_parser_only):
    # GH 11020
    parser = c_parser_only
    with tm.ensure_clean() as path:
        with open(path, "w", newline="\n", encoding="utf-8") as f:
            f.write("blah\n\ncol_1,col_2,col_3\n\n")
        result = parser.read_csv(path, skiprows=2, encoding="utf-8", engine="c")
    expected = DataFrame(columns=["col_1", "col_2", "col_3"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("float_precision", [None, "legacy", "high", "round_trip"])
@pytest.mark.parametrize(
    "data,thousands,decimal",
    [
        (
            """A|B|C
1|2,334.01|5
10|13|10.
""",
            ",",
            ".",
        ),
        (
            """A|B|C
1|2.334,01|5
10|13|10,
""",
            ".",
            ",",
        ),
    ],
)
def test_1000_sep_with_decimal(
    c_parser_only, data, thousands, decimal, float_precision
):
    parser = c_parser_only
    expected = DataFrame({"A": [1, 10], "B": [2334.01, 13], "C": [5, 10.0]})

    result = parser.read_csv(
        StringIO(data),
        sep="|",
        thousands=thousands,
        decimal=decimal,
        float_precision=float_precision,
    )
    tm.assert_frame_equal(result, expected)


def test_float_precision_options(c_parser_only):
    # GH 17154, 36228
    parser = c_parser_only
    s = "foo\n243.164\n"
    df = parser.read_csv(StringIO(s))
    df2 = parser.read_csv(StringIO(s), float_precision="high")

    tm.assert_frame_equal(df, df2)

    df3 = parser.read_csv(StringIO(s), float_precision="legacy")

    assert not df.iloc[0, 0] == df3.iloc[0, 0]

    msg = "Unrecognized float_precision option: junk"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(s), float_precision="junk")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\dtypes\test_dtypes_basic.py ===
"""
Tests dtype specification during parsing
for all of the parsers defined in parsers.py
"""
from collections import defaultdict
from io import StringIO

import numpy as np
import pytest

from pandas.errors import ParserWarning

import pandas as pd
from pandas import (
    DataFrame,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.arrays import IntegerArray

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")


@pytest.mark.parametrize("dtype", [str, object])
@pytest.mark.parametrize("check_orig", [True, False])
@pytest.mark.usefixtures("pyarrow_xfail")
def test_dtype_all_columns(all_parsers, dtype, check_orig, using_infer_string):
    # see gh-3795, gh-6607
    parser = all_parsers

    df = DataFrame(
        np.random.default_rng(2).random((5, 2)).round(4),
        columns=list("AB"),
        index=["1A", "1B", "1C", "1D", "1E"],
    )

    with tm.ensure_clean("__passing_str_as_dtype__.csv") as path:
        df.to_csv(path)

        result = parser.read_csv(path, dtype=dtype, index_col=0)

        if check_orig:
            expected = df.copy()
            result = result.astype(float)
        elif using_infer_string and dtype is str:
            expected = df.astype(str)
        else:
            expected = df.astype(str).astype(object)

        tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
def test_dtype_per_column(all_parsers):
    parser = all_parsers
    data = """\
one,two
1,2.5
2,3.5
3,4.5
4,5.5"""
    expected = DataFrame(
        [[1, "2.5"], [2, "3.5"], [3, "4.5"], [4, "5.5"]], columns=["one", "two"]
    )
    expected["one"] = expected["one"].astype(np.float64)

    result = parser.read_csv(StringIO(data), dtype={"one": np.float64, 1: str})
    tm.assert_frame_equal(result, expected)


def test_invalid_dtype_per_column(all_parsers):
    parser = all_parsers
    data = """\
one,two
1,2.5
2,3.5
3,4.5
4,5.5"""

    with pytest.raises(TypeError, match="data type [\"']foo[\"'] not understood"):
        parser.read_csv(StringIO(data), dtype={"one": "foo", 1: "int"})


def test_raise_on_passed_int_dtype_with_nas(all_parsers):
    # see gh-2631
    parser = all_parsers
    data = """YEAR, DOY, a
2001,106380451,10
2001,,11
2001,106380451,67"""

    if parser.engine == "c":
        msg = "Integer column has NA values"
    elif parser.engine == "pyarrow":
        msg = "The 'skipinitialspace' option is not supported with the 'pyarrow' engine"
    else:
        msg = "Unable to convert column DOY"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), dtype={"DOY": np.int64}, skipinitialspace=True)


def test_dtype_with_converters(all_parsers):
    parser = all_parsers
    data = """a,b
1.1,2.2
1.2,2.3"""

    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data), dtype={"a": "i8"}, converters={"a": lambda x: str(x)}
            )
        return

    # Dtype spec ignored if converted specified.
    result = parser.read_csv_check_warnings(
        ParserWarning,
        "Both a converter and dtype were specified for column a "
        "- only the converter will be used.",
        StringIO(data),
        dtype={"a": "i8"},
        converters={"a": lambda x: str(x)},
    )
    expected = DataFrame({"a": ["1.1", "1.2"], "b": [2.2, 2.3]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype", list(np.typecodes["AllInteger"] + np.typecodes["Float"])
)
def test_numeric_dtype(all_parsers, dtype):
    data = "0\n1"
    parser = all_parsers
    expected = DataFrame([0, 1], dtype=dtype)

    result = parser.read_csv(StringIO(data), header=None, dtype=dtype)
    tm.assert_frame_equal(expected, result)


@pytest.mark.usefixtures("pyarrow_xfail")
def test_boolean_dtype(all_parsers):
    parser = all_parsers
    data = "\n".join(
        [
            "a",
            "True",
            "TRUE",
            "true",
            "1",
            "1.0",
            "False",
            "FALSE",
            "false",
            "0",
            "0.0",
            "NaN",
            "nan",
            "NA",
            "null",
            "NULL",
        ]
    )

    result = parser.read_csv(StringIO(data), dtype="boolean")
    expected = DataFrame(
        {
            "a": pd.array(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                dtype="boolean",
            )
        }
    )

    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
def test_delimiter_with_usecols_and_parse_dates(all_parsers):
    # GH#35873
    result = all_parsers.read_csv(
        StringIO('"dump","-9,1","-9,1",20101010'),
        engine="python",
        names=["col", "col1", "col2", "col3"],
        usecols=["col1", "col2", "col3"],
        parse_dates=["col3"],
        decimal=",",
    )
    expected = DataFrame(
        {"col1": [-9.1], "col2": [-9.1], "col3": [Timestamp("2010-10-10")]}
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("thousands", ["_", None])
def test_decimal_and_exponential(
    request, python_parser_only, numeric_decimal, thousands
):
    # GH#31920
    decimal_number_check(request, python_parser_only, numeric_decimal, thousands, None)


@pytest.mark.parametrize("thousands", ["_", None])
@pytest.mark.parametrize("float_precision", [None, "legacy", "high", "round_trip"])
def test_1000_sep_decimal_float_precision(
    request, c_parser_only, numeric_decimal, float_precision, thousands
):
    # test decimal and thousand sep handling in across 'float_precision'
    # parsers
    decimal_number_check(
        request, c_parser_only, numeric_decimal, thousands, float_precision
    )
    text, value = numeric_decimal
    text = " " + text + " "
    if isinstance(value, str):  # the negative cases (parse as text)
        value = " " + value + " "
    decimal_number_check(
        request, c_parser_only, (text, value), thousands, float_precision
    )


def decimal_number_check(request, parser, numeric_decimal, thousands, float_precision):
    # GH#31920
    value = numeric_decimal[0]
    if thousands is None and value in ("1_,", "1_234,56", "1_234,56e0"):
        request.applymarker(
            pytest.mark.xfail(reason=f"thousands={thousands} and sep is in {value}")
        )
    df = parser.read_csv(
        StringIO(value),
        float_precision=float_precision,
        sep="|",
        thousands=thousands,
        decimal=",",
        header=None,
    )
    val = df.iloc[0, 0]
    assert val == numeric_decimal[1]


@pytest.mark.parametrize("float_precision", [None, "legacy", "high", "round_trip"])
def test_skip_whitespace(c_parser_only, float_precision):
    DATA = """id\tnum\t
1\t1.2 \t
1\t 2.1\t
2\t 1\t
2\t 1.2 \t
"""
    df = c_parser_only.read_csv(
        StringIO(DATA),
        float_precision=float_precision,
        sep="\t",
        header=0,
        dtype={1: np.float64},
    )
    tm.assert_series_equal(df.iloc[:, 1], pd.Series([1.2, 2.1, 1.0, 1.2], name="num"))


@pytest.mark.usefixtures("pyarrow_xfail")
def test_true_values_cast_to_bool(all_parsers):
    # GH#34655
    text = """a,b
yes,xxx
no,yyy
1,zzz
0,aaa
    """
    parser = all_parsers
    result = parser.read_csv(
        StringIO(text),
        true_values=["yes"],
        false_values=["no"],
        dtype={"a": "boolean"},
    )
    expected = DataFrame(
        {"a": [True, False, True, False], "b": ["xxx", "yyy", "zzz", "aaa"]}
    )
    expected["a"] = expected["a"].astype("boolean")
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
@pytest.mark.parametrize("dtypes, exp_value", [({}, "1"), ({"a.1": "int64"}, 1)])
def test_dtype_mangle_dup_cols(all_parsers, dtypes, exp_value):
    # GH#35211
    parser = all_parsers
    data = """a,a\n1,1"""
    dtype_dict = {"a": str, **dtypes}
    # GH#42462
    dtype_dict_copy = dtype_dict.copy()
    result = parser.read_csv(StringIO(data), dtype=dtype_dict)
    expected = DataFrame({"a": ["1"], "a.1": [exp_value]})
    assert dtype_dict == dtype_dict_copy, "dtype dict changed"
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
def test_dtype_mangle_dup_cols_single_dtype(all_parsers):
    # GH#42022
    parser = all_parsers
    data = """a,a\n1,1"""
    result = parser.read_csv(StringIO(data), dtype=str)
    expected = DataFrame({"a": ["1"], "a.1": ["1"]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
def test_dtype_multi_index(all_parsers):
    # GH 42446
    parser = all_parsers
    data = "A,B,B\nX,Y,Z\n1,2,3"

    result = parser.read_csv(
        StringIO(data),
        header=list(range(2)),
        dtype={
            ("A", "X"): np.int32,
            ("B", "Y"): np.int32,
            ("B", "Z"): np.float32,
        },
    )

    expected = DataFrame(
        {
            ("A", "X"): np.int32([1]),
            ("B", "Y"): np.int32([2]),
            ("B", "Z"): np.float32([3]),
        }
    )

    tm.assert_frame_equal(result, expected)


def test_nullable_int_dtype(all_parsers, any_int_ea_dtype):
    # GH 25472
    parser = all_parsers
    dtype = any_int_ea_dtype

    data = """a,b,c
,3,5
1,,6
2,4,"""
    expected = DataFrame(
        {
            "a": pd.array([pd.NA, 1, 2], dtype=dtype),
            "b": pd.array([3, pd.NA, 4], dtype=dtype),
            "c": pd.array([5, 6, pd.NA], dtype=dtype),
        }
    )
    actual = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(actual, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
@pytest.mark.parametrize("default", ["float", "float64"])
def test_dtypes_defaultdict(all_parsers, default):
    # GH#41574
    data = """a,b
1,2
"""
    dtype = defaultdict(lambda: default, a="int64")
    parser = all_parsers
    result = parser.read_csv(StringIO(data), dtype=dtype)
    expected = DataFrame({"a": [1], "b": 2.0})
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
def test_dtypes_defaultdict_mangle_dup_cols(all_parsers):
    # GH#41574
    data = """a,b,a,b,b.1
1,2,3,4,5
"""
    dtype = defaultdict(lambda: "float64", a="int64")
    dtype["b.1"] = "int64"
    parser = all_parsers
    result = parser.read_csv(StringIO(data), dtype=dtype)
    expected = DataFrame({"a": [1], "b": [2.0], "a.1": [3], "b.2": [4.0], "b.1": [5]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures("pyarrow_xfail")
def test_dtypes_defaultdict_invalid(all_parsers):
    # GH#41574
    data = """a,b
1,2
"""
    dtype = defaultdict(lambda: "invalid_dtype", a="int64")
    parser = all_parsers
    with pytest.raises(TypeError, match="not understood"):
        parser.read_csv(StringIO(data), dtype=dtype)


def test_dtype_backend(all_parsers):
    # GH#36712

    parser = all_parsers

    data = """a,b,c,d,e,f,g,h,i,j
1,2.5,True,a,,,,,12-31-2019,
3,4.5,False,b,6,7.5,True,a,12-31-2019,
"""
    result = parser.read_csv(
        StringIO(data), dtype_backend="numpy_nullable", parse_dates=["i"]
    )
    expected = DataFrame(
        {
            "a": pd.Series([1, 3], dtype="Int64"),
            "b": pd.Series([2.5, 4.5], dtype="Float64"),
            "c": pd.Series([True, False], dtype="boolean"),
            "d": pd.Series(["a", "b"], dtype="string"),
            "e": pd.Series([pd.NA, 6], dtype="Int64"),
            "f": pd.Series([pd.NA, 7.5], dtype="Float64"),
            "g": pd.Series([pd.NA, True], dtype="boolean"),
            "h": pd.Series([pd.NA, "a"], dtype="string"),
            "i": pd.Series([Timestamp("2019-12-31")] * 2),
            "j": pd.Series([pd.NA, pd.NA], dtype="Int64"),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_dtype_backend_and_dtype(all_parsers):
    # GH#36712

    parser = all_parsers

    data = """a,b
1,2.5
,
"""
    result = parser.read_csv(
        StringIO(data), dtype_backend="numpy_nullable", dtype="float64"
    )
    expected = DataFrame({"a": [1.0, np.nan], "b": [2.5, np.nan]})
    tm.assert_frame_equal(result, expected)


def test_dtype_backend_string(all_parsers, string_storage):
    # GH#36712
    with pd.option_context("mode.string_storage", string_storage):
        parser = all_parsers

        data = """a,b
a,x
b,
"""
        result = parser.read_csv(StringIO(data), dtype_backend="numpy_nullable")

        expected = DataFrame(
            {
                "a": pd.array(["a", "b"], dtype=pd.StringDtype(string_storage)),
                "b": pd.array(["x", pd.NA], dtype=pd.StringDtype(string_storage)),
            },
        )
    tm.assert_frame_equal(result, expected)


def test_dtype_backend_ea_dtype_specified(all_parsers):
    # GH#491496
    data = """a,b
1,2
"""
    parser = all_parsers
    result = parser.read_csv(
        StringIO(data), dtype="Int64", dtype_backend="numpy_nullable"
    )
    expected = DataFrame({"a": [1], "b": 2}, dtype="Int64")
    tm.assert_frame_equal(result, expected)


def test_dtype_backend_pyarrow(all_parsers, request):
    # GH#36712
    pa = pytest.importorskip("pyarrow")
    parser = all_parsers

    data = """a,b,c,d,e,f,g,h,i,j
1,2.5,True,a,,,,,12-31-2019,
3,4.5,False,b,6,7.5,True,a,12-31-2019,
"""
    result = parser.read_csv(StringIO(data), dtype_backend="pyarrow", parse_dates=["i"])
    expected = DataFrame(
        {
            "a": pd.Series([1, 3], dtype="int64[pyarrow]"),
            "b": pd.Series([2.5, 4.5], dtype="float64[pyarrow]"),
            "c": pd.Series([True, False], dtype="bool[pyarrow]"),
            "d": pd.Series(["a", "b"], dtype=pd.ArrowDtype(pa.string())),
            "e": pd.Series([pd.NA, 6], dtype="int64[pyarrow]"),
            "f": pd.Series([pd.NA, 7.5], dtype="float64[pyarrow]"),
            "g": pd.Series([pd.NA, True], dtype="bool[pyarrow]"),
            "h": pd.Series(
                [pd.NA, "a"],
                dtype=pd.ArrowDtype(pa.string()),
            ),
            "i": pd.Series([Timestamp("2019-12-31")] * 2),
            "j": pd.Series([pd.NA, pd.NA], dtype="null[pyarrow]"),
        }
    )
    tm.assert_frame_equal(result, expected)


# pyarrow engine failing:
# https://github.com/pandas-dev/pandas/issues/56136
@pytest.mark.usefixtures("pyarrow_xfail")
def test_ea_int_avoid_overflow(all_parsers):
    # GH#32134
    parser = all_parsers
    data = """a,b
1,1
,1
1582218195625938945,1
"""
    result = parser.read_csv(StringIO(data), dtype={"a": "Int64"})
    expected = DataFrame(
        {
            "a": IntegerArray(
                np.array([1, 1, 1582218195625938945]), np.array([False, True, False])
            ),
            "b": 1,
        }
    )
    tm.assert_frame_equal(result, expected)


def test_string_inference(all_parsers):
    # GH#54430
    dtype = pd.StringDtype(na_value=np.nan)

    data = """a,b
x,1
y,2
,3"""
    parser = all_parsers
    with pd.option_context("future.infer_string", True):
        result = parser.read_csv(StringIO(data))

    expected = DataFrame(
        {"a": pd.Series(["x", "y", None], dtype=dtype), "b": [1, 2, 3]},
        columns=pd.Index(["a", "b"], dtype=dtype),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["O", object, "object", np.object_, str, np.str_])
def test_string_inference_object_dtype(all_parsers, dtype, using_infer_string):
    # GH#56047
    data = """a,b
x,a
y,a
z,a"""
    parser = all_parsers
    with pd.option_context("future.infer_string", True):
        result = parser.read_csv(StringIO(data), dtype=dtype)

    expected_dtype = pd.StringDtype(na_value=np.nan) if dtype is str else object
    expected = DataFrame(
        {
            "a": pd.Series(["x", "y", "z"], dtype=expected_dtype),
            "b": pd.Series(["a", "a", "a"], dtype=expected_dtype),
        },
        columns=pd.Index(["a", "b"], dtype=pd.StringDtype(na_value=np.nan)),
    )
    tm.assert_frame_equal(result, expected)

    with pd.option_context("future.infer_string", True):
        result = parser.read_csv(StringIO(data), dtype={"a": dtype})

    expected = DataFrame(
        {
            "a": pd.Series(["x", "y", "z"], dtype=expected_dtype),
            "b": pd.Series(["a", "a", "a"], dtype=pd.StringDtype(na_value=np.nan)),
        },
        columns=pd.Index(["a", "b"], dtype=pd.StringDtype(na_value=np.nan)),
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_accurate_parsing_of_large_integers(all_parsers):
    # GH#52505
    data = """SYMBOL,MOMENT,ID,ID_DEAL
AAPL,20230301181139587,1925036343869802844,
AAPL,20230301181139587,2023552585717889863,2023552585717263358
NVDA,20230301181139587,2023552585717889863,2023552585717263359
AMC,20230301181139587,2023552585717889863,2023552585717263360
AMZN,20230301181139587,2023552585717889759,2023552585717263360
MSFT,20230301181139587,2023552585717889863,2023552585717263361
NVDA,20230301181139587,2023552585717889827,2023552585717263361"""
    orders = all_parsers.read_csv(StringIO(data), dtype={"ID_DEAL": pd.Int64Dtype()})
    assert len(orders.loc[orders["ID_DEAL"] == 2023552585717263358, "ID_DEAL"]) == 1
    assert len(orders.loc[orders["ID_DEAL"] == 2023552585717263359, "ID_DEAL"]) == 1
    assert len(orders.loc[orders["ID_DEAL"] == 2023552585717263360, "ID_DEAL"]) == 2
    assert len(orders.loc[orders["ID_DEAL"] == 2023552585717263361, "ID_DEAL"]) == 2


def test_dtypes_with_usecols(all_parsers):
    # GH#54868

    parser = all_parsers
    data = """a,b,c
1,2,3
4,5,6"""

    result = parser.read_csv(StringIO(data), usecols=["a", "c"], dtype={"a": object})
    if parser.engine == "pyarrow":
        values = [1, 4]
    else:
        values = ["1", "4"]
    expected = DataFrame({"a": pd.Series(values, dtype=object), "c": [3, 6]})
    tm.assert_frame_equal(result, expected)


def test_index_col_with_dtype_no_rangeindex(all_parsers):
    data = StringIO("345.5,519.5,0\n519.5,726.5,1")
    result = all_parsers.read_csv(
        data,
        header=None,
        names=["start", "stop", "bin_id"],
        dtype={"start": np.float32, "stop": np.float32, "bin_id": np.uint32},
        index_col="bin_id",
    ).index
    expected = pd.Index([0, 1], dtype=np.uint32, name="bin_id")
    tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_character.py ===
import textwrap

import pytest

import numpy as np
from numpy.f2py.tests import util
from numpy.testing import assert_array_equal, assert_equal, assert_raises


@pytest.mark.slow
class TestCharacterString(util.F2PyTest):
    # options = ['--debug-capi', '--build-dir', '/tmp/test-build-f2py']
    suffix = '.f90'
    fprefix = 'test_character_string'
    length_list = ['1', '3', 'star']

    code = ''
    for length in length_list:
        fsuffix = length
        clength = {'star': '(*)'}.get(length, length)

        code += textwrap.dedent(f"""

        subroutine {fprefix}_input_{fsuffix}(c, o, n)
          character*{clength}, intent(in) :: c
          integer n
          !f2py integer, depend(c), intent(hide) :: n = slen(c)
          integer*1, dimension(n) :: o
          !f2py intent(out) o
          o = transfer(c, o)
        end subroutine {fprefix}_input_{fsuffix}

        subroutine {fprefix}_output_{fsuffix}(c, o, n)
          character*{clength}, intent(out) :: c
          integer n
          integer*1, dimension(n), intent(in) :: o
          !f2py integer, depend(o), intent(hide) :: n = len(o)
          c = transfer(o, c)
        end subroutine {fprefix}_output_{fsuffix}

        subroutine {fprefix}_array_input_{fsuffix}(c, o, m, n)
          integer m, i, n
          character*{clength}, intent(in), dimension(m) :: c
          !f2py integer, depend(c), intent(hide) :: m = len(c)
          !f2py integer, depend(c), intent(hide) :: n = f2py_itemsize(c)
          integer*1, dimension(m, n), intent(out) :: o
          do i=1,m
            o(i, :) = transfer(c(i), o(i, :))
          end do
        end subroutine {fprefix}_array_input_{fsuffix}

        subroutine {fprefix}_array_output_{fsuffix}(c, o, m, n)
          character*{clength}, intent(out), dimension(m) :: c
          integer n
          integer*1, dimension(m, n), intent(in) :: o
          !f2py character(f2py_len=n) :: c
          !f2py integer, depend(o), intent(hide) :: m = len(o)
          !f2py integer, depend(o), intent(hide) :: n = shape(o, 1)
          do i=1,m
            c(i) = transfer(o(i, :), c(i))
          end do
        end subroutine {fprefix}_array_output_{fsuffix}

        subroutine {fprefix}_2d_array_input_{fsuffix}(c, o, m1, m2, n)
          integer m1, m2, i, j, n
          character*{clength}, intent(in), dimension(m1, m2) :: c
          !f2py integer, depend(c), intent(hide) :: m1 = len(c)
          !f2py integer, depend(c), intent(hide) :: m2 = shape(c, 1)
          !f2py integer, depend(c), intent(hide) :: n = f2py_itemsize(c)
          integer*1, dimension(m1, m2, n), intent(out) :: o
          do i=1,m1
            do j=1,m2
              o(i, j, :) = transfer(c(i, j), o(i, j, :))
            end do
          end do
        end subroutine {fprefix}_2d_array_input_{fsuffix}
        """)

    @pytest.mark.parametrize("length", length_list)
    def test_input(self, length):
        fsuffix = {'(*)': 'star'}.get(length, length)
        f = getattr(self.module, self.fprefix + '_input_' + fsuffix)

        a = {'1': 'a', '3': 'abc', 'star': 'abcde' * 3}[length]

        assert_array_equal(f(a), np.array(list(map(ord, a)), dtype='u1'))

    @pytest.mark.parametrize("length", length_list[:-1])
    def test_output(self, length):
        fsuffix = length
        f = getattr(self.module, self.fprefix + '_output_' + fsuffix)

        a = {'1': 'a', '3': 'abc'}[length]

        assert_array_equal(f(np.array(list(map(ord, a)), dtype='u1')),
                           a.encode())

    @pytest.mark.parametrize("length", length_list)
    def test_array_input(self, length):
        fsuffix = length
        f = getattr(self.module, self.fprefix + '_array_input_' + fsuffix)

        a = np.array([{'1': 'a', '3': 'abc', 'star': 'abcde' * 3}[length],
                      {'1': 'A', '3': 'ABC', 'star': 'ABCDE' * 3}[length],
                      ], dtype='S')

        expected = np.array([list(s) for s in a], dtype='u1')
        assert_array_equal(f(a), expected)

    @pytest.mark.parametrize("length", length_list)
    def test_array_output(self, length):
        fsuffix = length
        f = getattr(self.module, self.fprefix + '_array_output_' + fsuffix)

        expected = np.array(
            [{'1': 'a', '3': 'abc', 'star': 'abcde' * 3}[length],
             {'1': 'A', '3': 'ABC', 'star': 'ABCDE' * 3}[length]], dtype='S')

        a = np.array([list(s) for s in expected], dtype='u1')
        assert_array_equal(f(a), expected)

    @pytest.mark.parametrize("length", length_list)
    def test_2d_array_input(self, length):
        fsuffix = length
        f = getattr(self.module, self.fprefix + '_2d_array_input_' + fsuffix)

        a = np.array([[{'1': 'a', '3': 'abc', 'star': 'abcde' * 3}[length],
                       {'1': 'A', '3': 'ABC', 'star': 'ABCDE' * 3}[length]],
                      [{'1': 'f', '3': 'fgh', 'star': 'fghij' * 3}[length],
                       {'1': 'F', '3': 'FGH', 'star': 'FGHIJ' * 3}[length]]],
                     dtype='S')
        expected = np.array([[list(item) for item in row] for row in a],
                            dtype='u1', order='F')
        assert_array_equal(f(a), expected)


class TestCharacter(util.F2PyTest):
    # options = ['--debug-capi', '--build-dir', '/tmp/test-build-f2py']
    suffix = '.f90'
    fprefix = 'test_character'

    code = textwrap.dedent(f"""
       subroutine {fprefix}_input(c, o)
          character, intent(in) :: c
          integer*1 o
          !f2py intent(out) o
          o = transfer(c, o)
       end subroutine {fprefix}_input

       subroutine {fprefix}_output(c, o)
          character :: c
          integer*1, intent(in) :: o
          !f2py intent(out) c
          c = transfer(o, c)
       end subroutine {fprefix}_output

       subroutine {fprefix}_input_output(c, o)
          character, intent(in) :: c
          character o
          !f2py intent(out) o
          o = c
       end subroutine {fprefix}_input_output

       subroutine {fprefix}_inout(c, n)
          character :: c, n
          !f2py intent(in) n
          !f2py intent(inout) c
          c = n
       end subroutine {fprefix}_inout

       function {fprefix}_return(o) result (c)
          character :: c
          character, intent(in) :: o
          c = transfer(o, c)
       end function {fprefix}_return

       subroutine {fprefix}_array_input(c, o)
          character, intent(in) :: c(3)
          integer*1 o(3)
          !f2py intent(out) o
          integer i
          do i=1,3
            o(i) = transfer(c(i), o(i))
          end do
       end subroutine {fprefix}_array_input

       subroutine {fprefix}_2d_array_input(c, o)
          character, intent(in) :: c(2, 3)
          integer*1 o(2, 3)
          !f2py intent(out) o
          integer i, j
          do i=1,2
            do j=1,3
              o(i, j) = transfer(c(i, j), o(i, j))
            end do
          end do
       end subroutine {fprefix}_2d_array_input

       subroutine {fprefix}_array_output(c, o)
          character :: c(3)
          integer*1, intent(in) :: o(3)
          !f2py intent(out) c
          do i=1,3
            c(i) = transfer(o(i), c(i))
          end do
       end subroutine {fprefix}_array_output

       subroutine {fprefix}_array_inout(c, n)
          character :: c(3), n(3)
          !f2py intent(in) n(3)
          !f2py intent(inout) c(3)
          do i=1,3
            c(i) = n(i)
          end do
       end subroutine {fprefix}_array_inout

       subroutine {fprefix}_2d_array_inout(c, n)
          character :: c(2, 3), n(2, 3)
          !f2py intent(in) n(2, 3)
          !f2py intent(inout) c(2. 3)
          integer i, j
          do i=1,2
            do j=1,3
              c(i, j) = n(i, j)
            end do
          end do
       end subroutine {fprefix}_2d_array_inout

       function {fprefix}_array_return(o) result (c)
          character, dimension(3) :: c
          character, intent(in) :: o(3)
          do i=1,3
            c(i) = o(i)
          end do
       end function {fprefix}_array_return

       function {fprefix}_optional(o) result (c)
          character, intent(in) :: o
          !f2py character o = "a"
          character :: c
          c = o
       end function {fprefix}_optional
    """)

    @pytest.mark.parametrize("dtype", ['c', 'S1'])
    def test_input(self, dtype):
        f = getattr(self.module, self.fprefix + '_input')

        assert_equal(f(np.array('a', dtype=dtype)), ord('a'))
        assert_equal(f(np.array(b'a', dtype=dtype)), ord('a'))
        assert_equal(f(np.array(['a'], dtype=dtype)), ord('a'))
        assert_equal(f(np.array('abc', dtype=dtype)), ord('a'))
        assert_equal(f(np.array([['a']], dtype=dtype)), ord('a'))

    def test_input_varia(self):
        f = getattr(self.module, self.fprefix + '_input')

        assert_equal(f('a'), ord('a'))
        assert_equal(f(b'a'), ord(b'a'))
        assert_equal(f(''), 0)
        assert_equal(f(b''), 0)
        assert_equal(f(b'\0'), 0)
        assert_equal(f('ab'), ord('a'))
        assert_equal(f(b'ab'), ord('a'))
        assert_equal(f(['a']), ord('a'))

        assert_equal(f(np.array(b'a')), ord('a'))
        assert_equal(f(np.array([b'a'])), ord('a'))
        a = np.array('a')
        assert_equal(f(a), ord('a'))
        a = np.array(['a'])
        assert_equal(f(a), ord('a'))

        try:
            f([])
        except IndexError as msg:
            if not str(msg).endswith(' got 0-list'):
                raise
        else:
            raise SystemError(f'{f.__name__} should have failed on empty list')

        try:
            f(97)
        except TypeError as msg:
            if not str(msg).endswith(' got int instance'):
                raise
        else:
            raise SystemError(f'{f.__name__} should have failed on int value')

    @pytest.mark.parametrize("dtype", ['c', 'S1', 'U1'])
    def test_array_input(self, dtype):
        f = getattr(self.module, self.fprefix + '_array_input')

        assert_array_equal(f(np.array(['a', 'b', 'c'], dtype=dtype)),
                           np.array(list(map(ord, 'abc')), dtype='i1'))
        assert_array_equal(f(np.array([b'a', b'b', b'c'], dtype=dtype)),
                           np.array(list(map(ord, 'abc')), dtype='i1'))

    def test_array_input_varia(self):
        f = getattr(self.module, self.fprefix + '_array_input')
        assert_array_equal(f(['a', 'b', 'c']),
                           np.array(list(map(ord, 'abc')), dtype='i1'))
        assert_array_equal(f([b'a', b'b', b'c']),
                           np.array(list(map(ord, 'abc')), dtype='i1'))

        try:
            f(['a', 'b', 'c', 'd'])
        except ValueError as msg:
            if not str(msg).endswith(
                    'th dimension must be fixed to 3 but got 4'):
                raise
        else:
            raise SystemError(
                f'{f.__name__} should have failed on wrong input')

    @pytest.mark.parametrize("dtype", ['c', 'S1', 'U1'])
    def test_2d_array_input(self, dtype):
        f = getattr(self.module, self.fprefix + '_2d_array_input')

        a = np.array([['a', 'b', 'c'],
                      ['d', 'e', 'f']], dtype=dtype, order='F')
        expected = a.view(np.uint32 if dtype == 'U1' else np.uint8)
        assert_array_equal(f(a), expected)

    def test_output(self):
        f = getattr(self.module, self.fprefix + '_output')

        assert_equal(f(ord(b'a')), b'a')
        assert_equal(f(0), b'\0')

    def test_array_output(self):
        f = getattr(self.module, self.fprefix + '_array_output')

        assert_array_equal(f(list(map(ord, 'abc'))),
                           np.array(list('abc'), dtype='S1'))

    def test_input_output(self):
        f = getattr(self.module, self.fprefix + '_input_output')

        assert_equal(f(b'a'), b'a')
        assert_equal(f('a'), b'a')
        assert_equal(f(''), b'\0')

    @pytest.mark.parametrize("dtype", ['c', 'S1'])
    def test_inout(self, dtype):
        f = getattr(self.module, self.fprefix + '_inout')

        a = np.array(list('abc'), dtype=dtype)
        f(a, 'A')
        assert_array_equal(a, np.array(list('Abc'), dtype=a.dtype))
        f(a[1:], 'B')
        assert_array_equal(a, np.array(list('ABc'), dtype=a.dtype))

        a = np.array(['abc'], dtype=dtype)
        f(a, 'A')
        assert_array_equal(a, np.array(['Abc'], dtype=a.dtype))

    def test_inout_varia(self):
        f = getattr(self.module, self.fprefix + '_inout')
        a = np.array('abc', dtype='S3')
        f(a, 'A')
        assert_array_equal(a, np.array('Abc', dtype=a.dtype))

        a = np.array(['abc'], dtype='S3')
        f(a, 'A')
        assert_array_equal(a, np.array(['Abc'], dtype=a.dtype))

        try:
            f('abc', 'A')
        except ValueError as msg:
            if not str(msg).endswith(' got 3-str'):
                raise
        else:
            raise SystemError(f'{f.__name__} should have failed on str value')

    @pytest.mark.parametrize("dtype", ['c', 'S1'])
    def test_array_inout(self, dtype):
        f = getattr(self.module, self.fprefix + '_array_inout')
        n = np.array(['A', 'B', 'C'], dtype=dtype, order='F')

        a = np.array(['a', 'b', 'c'], dtype=dtype, order='F')
        f(a, n)
        assert_array_equal(a, n)

        a = np.array(['a', 'b', 'c', 'd'], dtype=dtype)
        f(a[1:], n)
        assert_array_equal(a, np.array(['a', 'A', 'B', 'C'], dtype=dtype))

        a = np.array([['a', 'b', 'c']], dtype=dtype, order='F')
        f(a, n)
        assert_array_equal(a, np.array([['A', 'B', 'C']], dtype=dtype))

        a = np.array(['a', 'b', 'c', 'd'], dtype=dtype, order='F')
        try:
            f(a, n)
        except ValueError as msg:
            if not str(msg).endswith(
                    'th dimension must be fixed to 3 but got 4'):
                raise
        else:
            raise SystemError(
                f'{f.__name__} should have failed on wrong input')

    @pytest.mark.parametrize("dtype", ['c', 'S1'])
    def test_2d_array_inout(self, dtype):
        f = getattr(self.module, self.fprefix + '_2d_array_inout')
        n = np.array([['A', 'B', 'C'],
                      ['D', 'E', 'F']],
                     dtype=dtype, order='F')
        a = np.array([['a', 'b', 'c'],
                      ['d', 'e', 'f']],
                     dtype=dtype, order='F')
        f(a, n)
        assert_array_equal(a, n)

    def test_return(self):
        f = getattr(self.module, self.fprefix + '_return')

        assert_equal(f('a'), b'a')

    @pytest.mark.skip('fortran function returning array segfaults')
    def test_array_return(self):
        f = getattr(self.module, self.fprefix + '_array_return')

        a = np.array(list('abc'), dtype='S1')
        assert_array_equal(f(a), a)

    def test_optional(self):
        f = getattr(self.module, self.fprefix + '_optional')

        assert_equal(f(), b"a")
        assert_equal(f(b'B'), b"B")


class TestMiscCharacter(util.F2PyTest):
    # options = ['--debug-capi', '--build-dir', '/tmp/test-build-f2py']
    suffix = '.f90'
    fprefix = 'test_misc_character'

    code = textwrap.dedent(f"""
       subroutine {fprefix}_gh18684(x, y, m)
         character(len=5), dimension(m), intent(in) :: x
         character*5, dimension(m), intent(out) :: y
         integer i, m
         !f2py integer, intent(hide), depend(x) :: m = f2py_len(x)
         do i=1,m
           y(i) = x(i)
         end do
       end subroutine {fprefix}_gh18684

       subroutine {fprefix}_gh6308(x, i)
         integer i
         !f2py check(i>=0 && i<12) i
         character*5 name, x
         common name(12)
         name(i + 1) = x
       end subroutine {fprefix}_gh6308

       subroutine {fprefix}_gh4519(x)
         character(len=*), intent(in) :: x(:)
         !f2py intent(out) x
         integer :: i
         ! Uncomment for debug printing:
         !do i=1, size(x)
         !   print*, "x(",i,")=", x(i)
         !end do
       end subroutine {fprefix}_gh4519

       pure function {fprefix}_gh3425(x) result (y)
         character(len=*), intent(in) :: x
         character(len=len(x)) :: y
         integer :: i
         do i = 1, len(x)
           j = iachar(x(i:i))
           if (j>=iachar("a") .and. j<=iachar("z") ) then
             y(i:i) = achar(j-32)
           else
             y(i:i) = x(i:i)
           endif
         end do
       end function {fprefix}_gh3425

       subroutine {fprefix}_character_bc_new(x, y, z)
         character, intent(in) :: x
         character, intent(out) :: y
         !f2py character, depend(x) :: y = x
         !f2py character, dimension((x=='a'?1:2)), depend(x), intent(out) :: z
         character, dimension(*) :: z
         !f2py character, optional, check(x == 'a' || x == 'b') :: x = 'a'
         !f2py callstatement (*f2py_func)(&x, &y, z)
         !f2py callprotoargument character*, character*, character*
         if (y.eq.x) then
           y = x
         else
           y = 'e'
         endif
         z(1) = 'c'
       end subroutine {fprefix}_character_bc_new

       subroutine {fprefix}_character_bc_old(x, y, z)
         character, intent(in) :: x
         character, intent(out) :: y
         !f2py character, depend(x) :: y = x[0]
         !f2py character, dimension((*x=='a'?1:2)), depend(x), intent(out) :: z
         character, dimension(*) :: z
         !f2py character, optional, check(*x == 'a' || x[0] == 'b') :: x = 'a'
         !f2py callstatement (*f2py_func)(x, y, z)
         !f2py callprotoargument char*, char*, char*
          if (y.eq.x) then
           y = x
         else
           y = 'e'
         endif
         z(1) = 'c'
       end subroutine {fprefix}_character_bc_old
    """)

    @pytest.mark.slow
    def test_gh18684(self):
        # Test character(len=5) and character*5 usages
        f = getattr(self.module, self.fprefix + '_gh18684')
        x = np.array(["abcde", "fghij"], dtype='S5')
        y = f(x)

        assert_array_equal(x, y)

    def test_gh6308(self):
        # Test character string array in a common block
        f = getattr(self.module, self.fprefix + '_gh6308')

        assert_equal(self.module._BLNK_.name.dtype, np.dtype('S5'))
        assert_equal(len(self.module._BLNK_.name), 12)
        f("abcde", 0)
        assert_equal(self.module._BLNK_.name[0], b"abcde")
        f("12345", 5)
        assert_equal(self.module._BLNK_.name[5], b"12345")

    def test_gh4519(self):
        # Test array of assumed length strings
        f = getattr(self.module, self.fprefix + '_gh4519')

        for x, expected in [
                ('a', {'shape': (), 'dtype': np.dtype('S1')}),
                ('text', {'shape': (), 'dtype': np.dtype('S4')}),
                (np.array(['1', '2', '3'], dtype='S1'),
                 {'shape': (3,), 'dtype': np.dtype('S1')}),
                (['1', '2', '34'],
                 {'shape': (3,), 'dtype': np.dtype('S2')}),
                (['', ''], {'shape': (2,), 'dtype': np.dtype('S1')})]:
            r = f(x)
            for k, v in expected.items():
                assert_equal(getattr(r, k), v)

    def test_gh3425(self):
        # Test returning a copy of assumed length string
        f = getattr(self.module, self.fprefix + '_gh3425')
        # f is equivalent to bytes.upper

        assert_equal(f('abC'), b'ABC')
        assert_equal(f(''), b'')
        assert_equal(f('abC12d'), b'ABC12D')

    @pytest.mark.parametrize("state", ['new', 'old'])
    def test_character_bc(self, state):
        f = getattr(self.module, self.fprefix + '_character_bc_' + state)

        c, a = f()
        assert_equal(c, b'a')
        assert_equal(len(a), 1)

        c, a = f(b'b')
        assert_equal(c, b'b')
        assert_equal(len(a), 2)

        assert_raises(Exception, lambda: f(b'c'))


class TestStringScalarArr(util.F2PyTest):
    sources = [util.getpath("tests", "src", "string", "scalar_string.f90")]

    def test_char(self):
        for out in (self.module.string_test.string,
                    self.module.string_test.string77):
            expected = ()
            assert out.shape == expected
            expected = '|S8'
            assert out.dtype == expected

    def test_char_arr(self):
        for out in (self.module.string_test.strarr,
                    self.module.string_test.strarr77):
            expected = (5, 7)
            assert out.shape == expected
            expected = '|S12'
            assert out.dtype == expected

class TestStringAssumedLength(util.F2PyTest):
    sources = [util.getpath("tests", "src", "string", "gh24008.f")]

    def test_gh24008(self):
        self.module.greet("joe", "bob")

@pytest.mark.slow
class TestStringOptionalInOut(util.F2PyTest):
    sources = [util.getpath("tests", "src", "string", "gh24662.f90")]

    def test_gh24662(self):
        self.module.string_inout_optional()
        a = np.array('hi', dtype='S32')
        self.module.string_inout_optional(a)
        assert "output string" in a.tobytes().decode()
        with pytest.raises(Exception):  # noqa: B017
            aa = "Hi"
            self.module.string_inout_optional(aa)


@pytest.mark.slow
class TestNewCharHandling(util.F2PyTest):
    # from v1.24 onwards, gh-19388
    sources = [
        util.getpath("tests", "src", "string", "gh25286.pyf"),
        util.getpath("tests", "src", "string", "gh25286.f90")
    ]
    module_name = "_char_handling_test"

    def test_gh25286(self):
        info = self.module.charint('T')
        assert info == 2

@pytest.mark.slow
class TestBCCharHandling(util.F2PyTest):
    # SciPy style, "incorrect" bindings with a hook
    sources = [
        util.getpath("tests", "src", "string", "gh25286_bc.pyf"),
        util.getpath("tests", "src", "string", "gh25286.f90")
    ]
    module_name = "_char_handling_test"

    def test_gh25286(self):
        info = self.module.charint('T')
        assert info == 2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flatbuffers\__init__.py ===
# Copyright 2014 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .builder import Builder
from .table import Table
from .compat import range_func as compat_range
from ._version import __version__
from . import util

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_typing\_nbit.py ===
"""A module with the precisions of platform-specific `~numpy.number`s."""

from typing import TypeAlias

from ._nbit_base import _8Bit, _16Bit, _32Bit, _64Bit, _96Bit, _128Bit

# To-be replaced with a `npt.NBitBase` subclass by numpy's mypy plugin
_NBitByte: TypeAlias = _8Bit
_NBitShort: TypeAlias = _16Bit
_NBitIntC: TypeAlias = _32Bit
_NBitIntP: TypeAlias = _32Bit | _64Bit
_NBitInt: TypeAlias = _NBitIntP
_NBitLong: TypeAlias = _32Bit | _64Bit
_NBitLongLong: TypeAlias = _64Bit

_NBitHalf: TypeAlias = _16Bit
_NBitSingle: TypeAlias = _32Bit
_NBitDouble: TypeAlias = _64Bit
_NBitLongDouble: TypeAlias = _64Bit | _96Bit | _128Bit

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\__init__.py ===
from .calibrate import (  # noqa: F401
    CalibraterBase,
    CalibrationDataReader,
    CalibrationMethod,
    MinMaxCalibrater,
    create_calibrator,
)
from .qdq_quantizer import QDQQuantizer  # noqa: F401
from .quant_utils import QuantFormat, QuantType, write_calibration_table  # noqa: F401
from .quantize import (
    DynamicQuantConfig,  # noqa: F401
    QuantizationMode,  # noqa: F401
    StaticQuantConfig,  # noqa: F401
    get_qdq_config,  # noqa: F401
    quantize,  # noqa: F401
    quantize_dynamic,  # noqa: F401
    quantize_static,  # noqa: F401
)
from .shape_inference import quant_pre_process  # noqa: F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\__init__.py ===
# Copyright (c) 2010-2024 openpyxl

DEBUG = False

from openpyxl.compat.numbers import NUMPY
from openpyxl.xml import DEFUSEDXML, LXML
from openpyxl.workbook import Workbook
from openpyxl.reader.excel import load_workbook as open
from openpyxl.reader.excel import load_workbook
import openpyxl._constants as constants

# Expose constants especially the version number

__author__ = constants.__author__
__author_email__ = constants.__author_email__
__license__ = constants.__license__
__maintainer_email__ = constants.__maintainer_email__
__url__ = constants.__url__
__version__ = constants.__version__

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\__init__.py ===
# Copyright (c) 2010-2024 openpyxl

from .area_chart import AreaChart, AreaChart3D
from .bar_chart import BarChart, BarChart3D
from .bubble_chart import BubbleChart
from .line_chart import LineChart, LineChart3D
from .pie_chart import (
    PieChart,
    PieChart3D,
    DoughnutChart,
    ProjectedPieChart
)
from .radar_chart import RadarChart
from .scatter_chart import ScatterChart
from .stock_chart import StockChart
from .surface_chart import SurfaceChart, SurfaceChart3D

from .series_factory import SeriesFactory as Series
from .reference import Reference

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\sparse\__init__.py ===
from pandas.core.arrays.sparse.accessor import (
    SparseAccessor,
    SparseFrameAccessor,
)
from pandas.core.arrays.sparse.array import (
    BlockIndex,
    IntIndex,
    SparseArray,
    make_sparse_index,
)

__all__ = [
    "BlockIndex",
    "IntIndex",
    "make_sparse_index",
    "SparseAccessor",
    "SparseArray",
    "SparseFrameAccessor",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\excel\__init__.py ===
from pandas.io.excel._base import (
    ExcelFile,
    ExcelWriter,
    read_excel,
)
from pandas.io.excel._odswriter import ODSWriter as _ODSWriter
from pandas.io.excel._openpyxl import OpenpyxlWriter as _OpenpyxlWriter
from pandas.io.excel._util import register_writer
from pandas.io.excel._xlsxwriter import XlsxWriter as _XlsxWriter

__all__ = ["read_excel", "ExcelWriter", "ExcelFile"]


register_writer(_OpenpyxlWriter)

register_writer(_XlsxWriter)


register_writer(_ODSWriter)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\_timer.py ===
"""
Timer context manager, only used in debug.

"""

from time import time

import contextlib
from typing import Generator


@contextlib.contextmanager
def timer(subject: str = "time") -> Generator[None, None, None]:
    """print the elapsed time. (only used in debugging)"""
    start = time()
    yield
    elapsed = time() - start
    elapsed_ms = elapsed * 1000
    print(f"{subject} elapsed {elapsed_ms:.1f}ms")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\clipboard\no_clipboard.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

GLOBAL_CLIPBOARD_BUFFER = ""


def get_clipboard_text() -> str:
    return GLOBAL_CLIPBOARD_BUFFER


def set_clipboard_text(text: str) -> None:
    global GLOBAL_CLIPBOARD_BUFFER
    GLOBAL_CLIPBOARD_BUFFER = text

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\__init__.py ===
# Licensed to the Software Freedom Conservancy (SFC) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The SFC licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


__version__ = "4.33.0"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\support\events.py ===
# Licensed to the Software Freedom Conservancy (SFC) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The SFC licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from .abstract_event_listener import AbstractEventListener  # noqa
from .event_firing_webdriver import EventFiringWebDriver  # noqa

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\engine\strategies.py ===
# engine/strategies.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Deprecated mock engine strategy used by Alembic.


"""

from __future__ import annotations

from .mock import MockConnection  # noqa


class MockEngineStrategy:
    MockConnection = MockConnection

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\suite\__init__.py ===
# testing/suite/__init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
from .test_cte import *  # noqa
from .test_ddl import *  # noqa
from .test_deprecations import *  # noqa
from .test_dialect import *  # noqa
from .test_insert import *  # noqa
from .test_reflection import *  # noqa
from .test_results import *  # noqa
from .test_rowcount import *  # noqa
from .test_select import *  # noqa
from .test_sequence import *  # noqa
from .test_types import *  # noqa
from .test_unicode_ddl import *  # noqa
from .test_update_delete import *  # noqa

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\diffgeom\__init__.py ===
from .diffgeom import (
    BaseCovarDerivativeOp, BaseScalarField, BaseVectorField, Commutator,
    contravariant_order, CoordSystem, CoordinateSymbol,
    CovarDerivativeOp, covariant_order, Differential, intcurve_diffequ,
    intcurve_series, LieDerivative, Manifold, metric_to_Christoffel_1st,
    metric_to_Christoffel_2nd, metric_to_Ricci_components,
    metric_to_Riemann_components, Patch, Point, TensorProduct, twoform_to_matrix,
    vectors_in_basis, WedgeProduct,
)

__all__ = [
    'BaseCovarDerivativeOp', 'BaseScalarField', 'BaseVectorField', 'Commutator',
    'contravariant_order', 'CoordSystem', 'CoordinateSymbol',
    'CovarDerivativeOp', 'covariant_order', 'Differential', 'intcurve_diffequ',
    'intcurve_series', 'LieDerivative', 'Manifold', 'metric_to_Christoffel_1st',
    'metric_to_Christoffel_2nd', 'metric_to_Ricci_components',
    'metric_to_Riemann_components', 'Patch', 'Point', 'TensorProduct',
    'twoform_to_matrix', 'vectors_in_basis', 'WedgeProduct',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_theanocode.py ===
"""
Important note on tests in this module - the Theano printing functions use a
global cache by default, which means that tests using it will modify global
state and thus not be independent from each other. Instead of using the "cache"
keyword argument each time, this module uses the theano_code_ and
theano_function_ functions defined below which default to using a new, empty
cache instead.
"""

import logging

from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy

theanologger = logging.getLogger('theano.configdefaults')
theanologger.setLevel(logging.CRITICAL)
theano = import_module('theano')
theanologger.setLevel(logging.WARNING)


if theano:
    import numpy as np
    ts = theano.scalar
    tt = theano.tensor
    xt, yt, zt = [tt.scalar(name, 'floatX') for name in 'xyz']
    Xt, Yt, Zt = [tt.tensor('floatX', (False, False), name=n) for n in 'XYZ']
else:
    #bin/test will not execute any tests now
    disabled = True

import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
        theano_function)


# Default set of matrix symbols for testing - make square so we can both
# multiply and perform elementwise operations between them.
X, Y, Z = [sy.MatrixSymbol(n, 4, 4) for n in 'XYZ']

# For testing AppliedUndef
f_t = sy.Function('f')(t)


def theano_code_(expr, **kwargs):
    """ Wrapper for theano_code that uses a new, empty cache by default. """
    kwargs.setdefault('cache', {})
    with warns_deprecated_sympy():
        return theano_code(expr, **kwargs)

def theano_function_(inputs, outputs, **kwargs):
    """ Wrapper for theano_function that uses a new, empty cache by default. """
    kwargs.setdefault('cache', {})
    with warns_deprecated_sympy():
        return theano_function(inputs, outputs, **kwargs)


def fgraph_of(*exprs):
    """ Transform SymPy expressions into Theano Computation.

    Parameters
    ==========
    exprs
        SymPy expressions

    Returns
    =======
    theano.gof.FunctionGraph
    """
    outs = list(map(theano_code_, exprs))
    ins = theano.gof.graph.inputs(outs)
    ins, outs = theano.gof.graph.clone(ins, outs)
    return theano.gof.FunctionGraph(ins, outs)


def theano_simplify(fgraph):
    """ Simplify a Theano Computation.

    Parameters
    ==========
    fgraph : theano.gof.FunctionGraph

    Returns
    =======
    theano.gof.FunctionGraph
    """
    mode = theano.compile.get_default_mode().excluding("fusion")
    fgraph = fgraph.clone()
    mode.optimizer.optimize(fgraph)
    return fgraph


def theq(a, b):
    """ Test two Theano objects for equality.

    Also accepts numeric types and lists/tuples of supported types.

    Note - debugprint() has a bug where it will accept numeric types but does
    not respect the "file" argument and in this case and instead prints the number
    to stdout and returns an empty string. This can lead to tests passing where
    they should fail because any two numbers will always compare as equal. To
    prevent this we treat numbers as a separate case.
    """
    numeric_types = (int, float, np.number)
    a_is_num = isinstance(a, numeric_types)
    b_is_num = isinstance(b, numeric_types)

    # Compare numeric types using regular equality
    if a_is_num or b_is_num:
        if not (a_is_num and b_is_num):
            return False

        return a == b

    # Compare sequences element-wise
    a_is_seq = isinstance(a, (tuple, list))
    b_is_seq = isinstance(b, (tuple, list))

    if a_is_seq or b_is_seq:
        if not (a_is_seq and b_is_seq) or type(a) != type(b):
            return False

        return list(map(theq, a)) == list(map(theq, b))

    # Otherwise, assume debugprint() can handle it
    astr = theano.printing.debugprint(a, file='str')
    bstr = theano.printing.debugprint(b, file='str')

    # Check for bug mentioned above
    for argname, argval, argstr in [('a', a, astr), ('b', b, bstr)]:
        if argstr == '':
            raise TypeError(
                'theano.printing.debugprint(%s) returned empty string '
                '(%s is instance of %r)'
                % (argname, argname, type(argval))
            )

    return astr == bstr


def test_example_symbols():
    """
    Check that the example symbols in this module print to their Theano
    equivalents, as many of the other tests depend on this.
    """
    assert theq(xt, theano_code_(x))
    assert theq(yt, theano_code_(y))
    assert theq(zt, theano_code_(z))
    assert theq(Xt, theano_code_(X))
    assert theq(Yt, theano_code_(Y))
    assert theq(Zt, theano_code_(Z))


def test_Symbol():
    """ Test printing a Symbol to a theano variable. """
    xx = theano_code_(x)
    assert isinstance(xx, (tt.TensorVariable, ts.ScalarVariable))
    assert xx.broadcastable == ()
    assert xx.name == x.name

    xx2 = theano_code_(x, broadcastables={x: (False,)})
    assert xx2.broadcastable == (False,)
    assert xx2.name == x.name

def test_MatrixSymbol():
    """ Test printing a MatrixSymbol to a theano variable. """
    XX = theano_code_(X)
    assert isinstance(XX, tt.TensorVariable)
    assert XX.broadcastable == (False, False)

@SKIP  # TODO - this is currently not checked but should be implemented
def test_MatrixSymbol_wrong_dims():
    """ Test MatrixSymbol with invalid broadcastable. """
    bcs = [(), (False,), (True,), (True, False), (False, True,), (True, True)]
    for bc in bcs:
        with raises(ValueError):
            theano_code_(X, broadcastables={X: bc})

def test_AppliedUndef():
    """ Test printing AppliedUndef instance, which works similarly to Symbol. """
    ftt = theano_code_(f_t)
    assert isinstance(ftt, tt.TensorVariable)
    assert ftt.broadcastable == ()
    assert ftt.name == 'f_t'


def test_add():
    expr = x + y
    comp = theano_code_(expr)
    assert comp.owner.op == theano.tensor.add

def test_trig():
    assert theq(theano_code_(sy.sin(x)), tt.sin(xt))
    assert theq(theano_code_(sy.tan(x)), tt.tan(xt))

def test_many():
    """ Test printing a complex expression with multiple symbols. """
    expr = sy.exp(x**2 + sy.cos(y)) * sy.log(2*z)
    comp = theano_code_(expr)
    expected = tt.exp(xt**2 + tt.cos(yt)) * tt.log(2*zt)
    assert theq(comp, expected)


def test_dtype():
    """ Test specifying specific data types through the dtype argument. """
    for dtype in ['float32', 'float64', 'int8', 'int16', 'int32', 'int64']:
        assert theano_code_(x, dtypes={x: dtype}).type.dtype == dtype

    # "floatX" type
    assert theano_code_(x, dtypes={x: 'floatX'}).type.dtype in ('float32', 'float64')

    # Type promotion
    assert theano_code_(x + 1, dtypes={x: 'float32'}).type.dtype == 'float32'
    assert theano_code_(x + y, dtypes={x: 'float64', y: 'float32'}).type.dtype == 'float64'


def test_broadcastables():
    """ Test the "broadcastables" argument when printing symbol-like objects. """

    # No restrictions on shape
    for s in [x, f_t]:
        for bc in [(), (False,), (True,), (False, False), (True, False)]:
            assert theano_code_(s, broadcastables={s: bc}).broadcastable == bc

    # TODO - matrix broadcasting?

def test_broadcasting():
    """ Test "broadcastable" attribute after applying element-wise binary op. """

    expr = x + y

    cases = [
        [(), (), ()],
        [(False,), (False,), (False,)],
        [(True,), (False,), (False,)],
        [(False, True), (False, False), (False, False)],
        [(True, False), (False, False), (False, False)],
    ]

    for bc1, bc2, bc3 in cases:
        comp = theano_code_(expr, broadcastables={x: bc1, y: bc2})
        assert comp.broadcastable == bc3


def test_MatMul():
    expr = X*Y*Z
    expr_t = theano_code_(expr)
    assert isinstance(expr_t.owner.op, tt.Dot)
    assert theq(expr_t, Xt.dot(Yt).dot(Zt))

def test_Transpose():
    assert isinstance(theano_code_(X.T).owner.op, tt.DimShuffle)

def test_MatAdd():
    expr = X+Y+Z
    assert isinstance(theano_code_(expr).owner.op, tt.Elemwise)


def test_Rationals():
    assert theq(theano_code_(sy.Integer(2) / 3), tt.true_div(2, 3))
    assert theq(theano_code_(S.Half), tt.true_div(1, 2))

def test_Integers():
    assert theano_code_(sy.Integer(3)) == 3

def test_factorial():
    n = sy.Symbol('n')
    assert theano_code_(sy.factorial(n))

def test_Derivative():
    simp = lambda expr: theano_simplify(fgraph_of(expr))
    assert theq(simp(theano_code_(sy.Derivative(sy.sin(x), x, evaluate=False))),
                simp(theano.grad(tt.sin(xt), xt)))


def test_theano_function_simple():
    """ Test theano_function() with single output. """
    f = theano_function_([x, y], [x+y])
    assert f(2, 3) == 5

def test_theano_function_multi():
    """ Test theano_function() with multiple outputs. """
    f = theano_function_([x, y], [x+y, x-y])
    o1, o2 = f(2, 3)
    assert o1 == 5
    assert o2 == -1

def test_theano_function_numpy():
    """ Test theano_function() vs Numpy implementation. """
    f = theano_function_([x, y], [x+y], dim=1,
                         dtypes={x: 'float64', y: 'float64'})
    assert np.linalg.norm(f([1, 2], [3, 4]) - np.asarray([4, 6])) < 1e-9

    f = theano_function_([x, y], [x+y], dtypes={x: 'float64', y: 'float64'},
                         dim=1)
    xx = np.arange(3).astype('float64')
    yy = 2*np.arange(3).astype('float64')
    assert np.linalg.norm(f(xx, yy) - 3*np.arange(3)) < 1e-9


def test_theano_function_matrix():
    m = sy.Matrix([[x, y], [z, x + y + z]])
    expected = np.array([[1.0, 2.0], [3.0, 1.0 + 2.0 + 3.0]])
    f = theano_function_([x, y, z], [m])
    np.testing.assert_allclose(f(1.0, 2.0, 3.0), expected)
    f = theano_function_([x, y, z], [m], scalar=True)
    np.testing.assert_allclose(f(1.0, 2.0, 3.0), expected)
    f = theano_function_([x, y, z], [m, m])
    assert isinstance(f(1.0, 2.0, 3.0), type([]))
    np.testing.assert_allclose(f(1.0, 2.0, 3.0)[0], expected)
    np.testing.assert_allclose(f(1.0, 2.0, 3.0)[1], expected)

def test_dim_handling():
    assert dim_handling([x], dim=2) == {x: (False, False)}
    assert dim_handling([x, y], dims={x: 1, y: 2}) == {x: (False, True),
                                                       y: (False, False)}
    assert dim_handling([x], broadcastables={x: (False,)}) == {x: (False,)}

def test_theano_function_kwargs():
    """
    Test passing additional kwargs from theano_function() to theano.function().
    """
    import numpy as np
    f = theano_function_([x, y, z], [x+y], dim=1, on_unused_input='ignore',
            dtypes={x: 'float64', y: 'float64', z: 'float64'})
    assert np.linalg.norm(f([1, 2], [3, 4], [0, 0]) - np.asarray([4, 6])) < 1e-9

    f = theano_function_([x, y, z], [x+y],
                        dtypes={x: 'float64', y: 'float64', z: 'float64'},
                        dim=1, on_unused_input='ignore')
    xx = np.arange(3).astype('float64')
    yy = 2*np.arange(3).astype('float64')
    zz = 2*np.arange(3).astype('float64')
    assert np.linalg.norm(f(xx, yy, zz) - 3*np.arange(3)) < 1e-9

def test_theano_function_scalar():
    """ Test the "scalar" argument to theano_function(). """

    args = [
        ([x, y], [x + y], None, [0]),  # Single 0d output
        ([X, Y], [X + Y], None, [2]),  # Single 2d output
        ([x, y], [x + y], {x: 0, y: 1}, [1]),  # Single 1d output
        ([x, y], [x + y, x - y], None, [0, 0]),  # Two 0d outputs
        ([x, y, X, Y], [x + y, X + Y], None, [0, 2]),  # One 0d output, one 2d
    ]

    # Create and test functions with and without the scalar setting
    for inputs, outputs, in_dims, out_dims in args:
        for scalar in [False, True]:

            f = theano_function_(inputs, outputs, dims=in_dims, scalar=scalar)

            # Check the theano_function attribute is set whether wrapped or not
            assert isinstance(f.theano_function, theano.compile.function_module.Function)

            # Feed in inputs of the appropriate size and get outputs
            in_values = [
                np.ones([1 if bc else 5 for bc in i.type.broadcastable])
                for i in f.theano_function.input_storage
            ]
            out_values = f(*in_values)
            if not isinstance(out_values, list):
                out_values = [out_values]

            # Check output types and shapes
            assert len(out_dims) == len(out_values)
            for d, value in zip(out_dims, out_values):

                if scalar and d == 0:
                    # Should have been converted to a scalar value
                    assert isinstance(value, np.number)

                else:
                    # Otherwise should be an array
                    assert isinstance(value, np.ndarray)
                    assert value.ndim == d

def test_theano_function_bad_kwarg():
    """
    Passing an unknown keyword argument to theano_function() should raise an
    exception.
    """
    raises(Exception, lambda : theano_function_([x], [x+1], foobar=3))


def test_slice():
    assert theano_code_(slice(1, 2, 3)) == slice(1, 2, 3)

    def theq_slice(s1, s2):
        for attr in ['start', 'stop', 'step']:
            a1 = getattr(s1, attr)
            a2 = getattr(s2, attr)
            if a1 is None or a2 is None:
                if not (a1 is None or a2 is None):
                    return False
            elif not theq(a1, a2):
                return False
        return True

    dtypes = {x: 'int32', y: 'int32'}
    assert theq_slice(theano_code_(slice(x, y), dtypes=dtypes), slice(xt, yt))
    assert theq_slice(theano_code_(slice(1, x, 3), dtypes=dtypes), slice(1, xt, 3))

def test_MatrixSlice():
    from theano import Constant

    cache = {}

    n = sy.Symbol('n', integer=True)
    X = sy.MatrixSymbol('X', n, n)

    Y = X[1:2:3, 4:5:6]
    Yt = theano_code_(Y, cache=cache)

    s = ts.Scalar('int64')
    assert tuple(Yt.owner.op.idx_list) == (slice(s, s, s), slice(s, s, s))
    assert Yt.owner.inputs[0] == theano_code_(X, cache=cache)
    # == doesn't work in theano like it does in SymPy. You have to use
    # equals.
    assert all(Yt.owner.inputs[i].equals(Constant(s, i)) for i in range(1, 7))

    k = sy.Symbol('k')
    theano_code_(k, dtypes={k: 'int32'})
    start, stop, step = 4, k, 2
    Y = X[start:stop:step]
    Yt = theano_code_(Y, dtypes={n: 'int32', k: 'int32'})
    # assert Yt.owner.op.idx_list[0].stop == kt

def test_BlockMatrix():
    n = sy.Symbol('n', integer=True)
    A, B, C, D = [sy.MatrixSymbol(name, n, n) for name in 'ABCD']
    At, Bt, Ct, Dt = map(theano_code_, (A, B, C, D))
    Block = sy.BlockMatrix([[A, B], [C, D]])
    Blockt = theano_code_(Block)
    solutions = [tt.join(0, tt.join(1, At, Bt), tt.join(1, Ct, Dt)),
                 tt.join(1, tt.join(0, At, Ct), tt.join(0, Bt, Dt))]
    assert any(theq(Blockt, solution) for solution in solutions)

@SKIP
def test_BlockMatrix_Inverse_execution():
    k, n = 2, 4
    dtype = 'float32'
    A = sy.MatrixSymbol('A', n, k)
    B = sy.MatrixSymbol('B', n, n)
    inputs = A, B
    output = B.I*A

    cutsizes = {A: [(n//2, n//2), (k//2, k//2)],
                B: [(n//2, n//2), (n//2, n//2)]}
    cutinputs = [sy.blockcut(i, *cutsizes[i]) for i in inputs]
    cutoutput = output.subs(dict(zip(inputs, cutinputs)))

    dtypes = dict(zip(inputs, [dtype]*len(inputs)))
    f = theano_function_(inputs, [output], dtypes=dtypes, cache={})
    fblocked = theano_function_(inputs, [sy.block_collapse(cutoutput)],
                                dtypes=dtypes, cache={})

    ninputs = [np.random.rand(*x.shape).astype(dtype) for x in inputs]
    ninputs = [np.arange(n*k).reshape(A.shape).astype(dtype),
               np.eye(n).astype(dtype)]
    ninputs[1] += np.ones(B.shape)*1e-5

    assert np.allclose(f(*ninputs), fblocked(*ninputs), rtol=1e-5)

def test_DenseMatrix():
    t = sy.Symbol('theta')
    for MatrixType in [sy.Matrix, sy.ImmutableMatrix]:
        X = MatrixType([[sy.cos(t), -sy.sin(t)], [sy.sin(t), sy.cos(t)]])
        tX = theano_code_(X)
        assert isinstance(tX, tt.TensorVariable)
        assert tX.owner.op == tt.join_


def test_cache_basic():
    """ Test single symbol-like objects are cached when printed by themselves. """

    # Pairs of objects which should be considered equivalent with respect to caching
    pairs = [
        (x, sy.Symbol('x')),
        (X, sy.MatrixSymbol('X', *X.shape)),
        (f_t, sy.Function('f')(sy.Symbol('t'))),
    ]

    for s1, s2 in pairs:
        cache = {}
        st = theano_code_(s1, cache=cache)

        # Test hit with same instance
        assert theano_code_(s1, cache=cache) is st

        # Test miss with same instance but new cache
        assert theano_code_(s1, cache={}) is not st

        # Test hit with different but equivalent instance
        assert theano_code_(s2, cache=cache) is st

def test_global_cache():
    """ Test use of the global cache. """
    from sympy.printing.theanocode import global_cache

    backup = dict(global_cache)
    try:
        # Temporarily empty global cache
        global_cache.clear()

        for s in [x, X, f_t]:
            with warns_deprecated_sympy():
                st = theano_code(s)
                assert theano_code(s) is st

    finally:
        # Restore global cache
        global_cache.update(backup)

def test_cache_types_distinct():
    """
    Test that symbol-like objects of different types (Symbol, MatrixSymbol,
    AppliedUndef) are distinguished by the cache even if they have the same
    name.
    """
    symbols = [sy.Symbol('f_t'), sy.MatrixSymbol('f_t', 4, 4), f_t]

    cache = {}  # Single shared cache
    printed = {}

    for s in symbols:
        st = theano_code_(s, cache=cache)
        assert st not in printed.values()
        printed[s] = st

    # Check all printed objects are distinct
    assert len(set(map(id, printed.values()))) == len(symbols)

    # Check retrieving
    for s, st in printed.items():
        with warns_deprecated_sympy():
            assert theano_code(s, cache=cache) is st

def test_symbols_are_created_once():
    """
    Test that a symbol is cached and reused when it appears in an expression
    more than once.
    """
    expr = sy.Add(x, x, evaluate=False)
    comp = theano_code_(expr)

    assert theq(comp, xt + xt)
    assert not theq(comp, xt + theano_code_(x))

def test_cache_complex():
    """
    Test caching on a complicated expression with multiple symbols appearing
    multiple times.
    """
    expr = x ** 2 + (y - sy.exp(x)) * sy.sin(z - x * y)
    symbol_names = {s.name for s in expr.free_symbols}
    expr_t = theano_code_(expr)

    # Iterate through variables in the Theano computational graph that the
    # printed expression depends on
    seen = set()
    for v in theano.gof.graph.ancestors([expr_t]):
        # Owner-less, non-constant variables should be our symbols
        if v.owner is None and not isinstance(v, theano.gof.graph.Constant):
            # Check it corresponds to a symbol and appears only once
            assert v.name in symbol_names
            assert v.name not in seen
            seen.add(v.name)

    # Check all were present
    assert seen == symbol_names


def test_Piecewise():
    # A piecewise linear
    expr = sy.Piecewise((0, x<0), (x, x<2), (1, True))  # ___/III
    result = theano_code_(expr)
    assert result.owner.op == tt.switch

    expected = tt.switch(xt<0, 0, tt.switch(xt<2, xt, 1))
    assert theq(result, expected)

    expr = sy.Piecewise((x, x < 0))
    result = theano_code_(expr)
    expected = tt.switch(xt < 0, xt, np.nan)
    assert theq(result, expected)

    expr = sy.Piecewise((0, sy.And(x>0, x<2)), \
        (x, sy.Or(x>2, x<0)))
    result = theano_code_(expr)
    expected = tt.switch(tt.and_(xt>0,xt<2), 0, \
        tt.switch(tt.or_(xt>2, xt<0), xt, np.nan))
    assert theq(result, expected)


def test_Relationals():
    assert theq(theano_code_(sy.Eq(x, y)), tt.eq(xt, yt))
    # assert theq(theano_code_(sy.Ne(x, y)), tt.neq(xt, yt))  # TODO - implement
    assert theq(theano_code_(x > y), xt > yt)
    assert theq(theano_code_(x < y), xt < yt)
    assert theq(theano_code_(x >= y), xt >= yt)
    assert theq(theano_code_(x <= y), xt <= yt)


def test_complexfunctions():
    with warns_deprecated_sympy():
        xt, yt = theano_code_(x, dtypes={x:'complex128'}), theano_code_(y, dtypes={y: 'complex128'})
    from sympy.functions.elementary.complexes import conjugate
    from theano.tensor import as_tensor_variable as atv
    from theano.tensor import complex as cplx
    with warns_deprecated_sympy():
        assert theq(theano_code_(y*conjugate(x)), yt*(xt.conj()))
        assert theq(theano_code_((1+2j)*x), xt*(atv(1.0)+atv(2.0)*cplx(0,1)))


def test_constantfunctions():
    with warns_deprecated_sympy():
        tf = theano_function_([],[1+1j])
    assert(tf()==1+1j)


def test_Exp1():
    """
    Test that exp(1) prints without error and evaluates close to SymPy's E
    """
    # sy.exp(1) should yield same instance of E as sy.E (singleton), but extra
    # check added for sanity
    e_a = sy.exp(1)
    e_b = sy.E

    np.testing.assert_allclose(float(e_a), np.e)
    np.testing.assert_allclose(float(e_b), np.e)

    e = theano_code_(e_a)
    np.testing.assert_allclose(float(e_a), e.eval())

    e = theano_code_(e_b)
    np.testing.assert_allclose(float(e_b), e.eval())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\testing\randtest.py ===
"""
.. deprecated:: 1.10

   ``sympy.testing.randtest`` functions have been moved to
   :mod:`sympy.core.random`.

"""
from sympy.utilities.exceptions import sympy_deprecation_warning

sympy_deprecation_warning("The sympy.testing.randtest submodule is deprecated. Use sympy.core.random instead.",
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-sympy-testing-randtest")

from sympy.core.random import (  # noqa:F401
    random_complex_number,
    verify_numerically,
    test_derivative_numerically,
    _randrange,
    _randint)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\test_transforms.py ===
from sympy.integrals.transforms import (
    mellin_transform, inverse_mellin_transform,
    fourier_transform, inverse_fourier_transform,
    sine_transform, inverse_sine_transform,
    cosine_transform, inverse_cosine_transform,
    hankel_transform, inverse_hankel_transform,
    FourierTransform, SineTransform, CosineTransform, InverseFourierTransform,
    InverseSineTransform, InverseCosineTransform, IntegralTransformError)
from sympy.integrals.laplace import (
    laplace_transform, inverse_laplace_transform)
from sympy.core.function import Function, expand_mul
from sympy.core import EulerGamma
from sympy.core.numbers import I, Rational, oo, pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import re, unpolarify
from sympy.functions.elementary.exponential import exp, exp_polar, log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan, cos, sin, tan
from sympy.functions.special.bessel import besseli, besselj, besselk, bessely
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.error_functions import erf, expint
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import meijerg
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.trigsimp import trigsimp
from sympy.testing.pytest import XFAIL, slow, skip, raises
from sympy.abc import x, s, a, b, c, d


nu, beta, rho = symbols('nu beta rho')


def test_undefined_function():
    from sympy.integrals.transforms import MellinTransform
    f = Function('f')
    assert mellin_transform(f(x), x, s) == MellinTransform(f(x), x, s)
    assert mellin_transform(f(x) + exp(-x), x, s) == \
        (MellinTransform(f(x), x, s) + gamma(s + 1)/s, (0, oo), True)


def test_free_symbols():
    f = Function('f')
    assert mellin_transform(f(x), x, s).free_symbols == {s}
    assert mellin_transform(f(x)*a, x, s).free_symbols == {s, a}


def test_as_integral():
    from sympy.integrals.integrals import Integral
    f = Function('f')
    assert mellin_transform(f(x), x, s).rewrite('Integral') == \
        Integral(x**(s - 1)*f(x), (x, 0, oo))
    assert fourier_transform(f(x), x, s).rewrite('Integral') == \
        Integral(f(x)*exp(-2*I*pi*s*x), (x, -oo, oo))
    assert laplace_transform(f(x), x, s, noconds=True).rewrite('Integral') == \
        Integral(f(x)*exp(-s*x), (x, 0, oo))
    assert str(2*pi*I*inverse_mellin_transform(f(s), s, x, (a, b)).rewrite('Integral')) \
        == "Integral(f(s)/x**s, (s, _c - oo*I, _c + oo*I))"
    assert str(2*pi*I*inverse_laplace_transform(f(s), s, x).rewrite('Integral')) == \
        "Integral(f(s)*exp(s*x), (s, _c - oo*I, _c + oo*I))"
    assert inverse_fourier_transform(f(s), s, x).rewrite('Integral') == \
        Integral(f(s)*exp(2*I*pi*s*x), (s, -oo, oo))

# NOTE this is stuck in risch because meijerint cannot handle it


@slow
@XFAIL
def test_mellin_transform_fail():
    skip("Risch takes forever.")

    MT = mellin_transform

    bpos = symbols('b', positive=True)
    # bneg = symbols('b', negative=True)

    expr = (sqrt(x + b**2) + b)**a/sqrt(x + b**2)
    # TODO does not work with bneg, argument wrong. Needs changes to matching.
    assert MT(expr.subs(b, -bpos), x, s) == \
        ((-1)**(a + 1)*2**(a + 2*s)*bpos**(a + 2*s - 1)*gamma(a + s)
         *gamma(1 - a - 2*s)/gamma(1 - s),
            (-re(a), -re(a)/2 + S.Half), True)

    expr = (sqrt(x + b**2) + b)**a
    assert MT(expr.subs(b, -bpos), x, s) == \
        (
            2**(a + 2*s)*a*bpos**(a + 2*s)*gamma(-a - 2*
                   s)*gamma(a + s)/gamma(-s + 1),
            (-re(a), -re(a)/2), True)

    # Test exponent 1:
    assert MT(expr.subs({b: -bpos, a: 1}), x, s) == \
        (-bpos**(2*s + 1)*gamma(s)*gamma(-s - S.Half)/(2*sqrt(pi)),
            (-1, Rational(-1, 2)), True)


def test_mellin_transform():
    from sympy.functions.elementary.miscellaneous import (Max, Min)
    MT = mellin_transform

    bpos = symbols('b', positive=True)

    # 8.4.2
    assert MT(x**nu*Heaviside(x - 1), x, s) == \
        (-1/(nu + s), (-oo, -re(nu)), True)
    assert MT(x**nu*Heaviside(1 - x), x, s) == \
        (1/(nu + s), (-re(nu), oo), True)

    assert MT((1 - x)**(beta - 1)*Heaviside(1 - x), x, s) == \
        (gamma(beta)*gamma(s)/gamma(beta + s), (0, oo), re(beta) > 0)
    assert MT((x - 1)**(beta - 1)*Heaviside(x - 1), x, s) == \
        (gamma(beta)*gamma(1 - beta - s)/gamma(1 - s),
            (-oo, 1 - re(beta)), re(beta) > 0)

    assert MT((1 + x)**(-rho), x, s) == \
        (gamma(s)*gamma(rho - s)/gamma(rho), (0, re(rho)), True)

    assert MT(abs(1 - x)**(-rho), x, s) == (
        2*sin(pi*rho/2)*gamma(1 - rho)*
        cos(pi*(s - rho/2))*gamma(s)*gamma(rho-s)/pi,
        (0, re(rho)), re(rho) < 1)
    mt = MT((1 - x)**(beta - 1)*Heaviside(1 - x)
            + a*(x - 1)**(beta - 1)*Heaviside(x - 1), x, s)
    assert mt[1], mt[2] == ((0, -re(beta) + 1), re(beta) > 0)

    assert MT((x**a - b**a)/(x - b), x, s)[0] == \
        pi*b**(a + s - 1)*sin(pi*a)/(sin(pi*s)*sin(pi*(a + s)))
    assert MT((x**a - bpos**a)/(x - bpos), x, s) == \
        (pi*bpos**(a + s - 1)*sin(pi*a)/(sin(pi*s)*sin(pi*(a + s))),
            (Max(0, -re(a)), Min(1, 1 - re(a))), True)

    expr = (sqrt(x + b**2) + b)**a
    assert MT(expr.subs(b, bpos), x, s) == \
        (-a*(2*bpos)**(a + 2*s)*gamma(s)*gamma(-a - 2*s)/gamma(-a - s + 1),
            (0, -re(a)/2), True)

    expr = (sqrt(x + b**2) + b)**a/sqrt(x + b**2)
    assert MT(expr.subs(b, bpos), x, s) == \
        (2**(a + 2*s)*bpos**(a + 2*s - 1)*gamma(s)
                                         *gamma(1 - a - 2*s)/gamma(1 - a - s),
            (0, -re(a)/2 + S.Half), True)

    # 8.4.2
    assert MT(exp(-x), x, s) == (gamma(s), (0, oo), True)
    assert MT(exp(-1/x), x, s) == (gamma(-s), (-oo, 0), True)

    # 8.4.5
    assert MT(log(x)**4*Heaviside(1 - x), x, s) == (24/s**5, (0, oo), True)
    assert MT(log(x)**3*Heaviside(x - 1), x, s) == (6/s**4, (-oo, 0), True)
    assert MT(log(x + 1), x, s) == (pi/(s*sin(pi*s)), (-1, 0), True)
    assert MT(log(1/x + 1), x, s) == (pi/(s*sin(pi*s)), (0, 1), True)
    assert MT(log(abs(1 - x)), x, s) == (pi/(s*tan(pi*s)), (-1, 0), True)
    assert MT(log(abs(1 - 1/x)), x, s) == (pi/(s*tan(pi*s)), (0, 1), True)

    # 8.4.14
    assert MT(erf(sqrt(x)), x, s) == \
        (-gamma(s + S.Half)/(sqrt(pi)*s), (Rational(-1, 2), 0), True)


def test_mellin_transform2():
    MT = mellin_transform
    # TODO we cannot currently do these (needs summation of 3F2(-1))
    #      this also implies that they cannot be written as a single g-function
    #      (although this is possible)
    mt = MT(log(x)/(x + 1), x, s)
    assert mt[1:] == ((0, 1), True)
    assert not hyperexpand(mt[0], allow_hyper=True).has(meijerg)
    mt = MT(log(x)**2/(x + 1), x, s)
    assert mt[1:] == ((0, 1), True)
    assert not hyperexpand(mt[0], allow_hyper=True).has(meijerg)
    mt = MT(log(x)/(x + 1)**2, x, s)
    assert mt[1:] == ((0, 2), True)
    assert not hyperexpand(mt[0], allow_hyper=True).has(meijerg)


@slow
def test_mellin_transform_bessel():
    from sympy.functions.elementary.miscellaneous import Max
    MT = mellin_transform

    # 8.4.19
    assert MT(besselj(a, 2*sqrt(x)), x, s) == \
        (gamma(a/2 + s)/gamma(a/2 - s + 1), (-re(a)/2, Rational(3, 4)), True)
    assert MT(sin(sqrt(x))*besselj(a, sqrt(x)), x, s) == \
        (2**a*gamma(-2*s + S.Half)*gamma(a/2 + s + S.Half)/(
        gamma(-a/2 - s + 1)*gamma(a - 2*s + 1)), (
        -re(a)/2 - S.Half, Rational(1, 4)), True)
    assert MT(cos(sqrt(x))*besselj(a, sqrt(x)), x, s) == \
        (2**a*gamma(a/2 + s)*gamma(-2*s + S.Half)/(
        gamma(-a/2 - s + S.Half)*gamma(a - 2*s + 1)), (
        -re(a)/2, Rational(1, 4)), True)
    assert MT(besselj(a, sqrt(x))**2, x, s) == \
        (gamma(a + s)*gamma(S.Half - s)
         / (sqrt(pi)*gamma(1 - s)*gamma(1 + a - s)),
            (-re(a), S.Half), True)
    assert MT(besselj(a, sqrt(x))*besselj(-a, sqrt(x)), x, s) == \
        (gamma(s)*gamma(S.Half - s)
         / (sqrt(pi)*gamma(1 - a - s)*gamma(1 + a - s)),
            (0, S.Half), True)
    # NOTE: prudnikov gives the strip below as (1/2 - re(a), 1). As far as
    #       I can see this is wrong (since besselj(z) ~ 1/sqrt(z) for z large)
    assert MT(besselj(a - 1, sqrt(x))*besselj(a, sqrt(x)), x, s) == \
        (gamma(1 - s)*gamma(a + s - S.Half)
         / (sqrt(pi)*gamma(Rational(3, 2) - s)*gamma(a - s + S.Half)),
            (S.Half - re(a), S.Half), True)
    assert MT(besselj(a, sqrt(x))*besselj(b, sqrt(x)), x, s) == \
        (4**s*gamma(1 - 2*s)*gamma((a + b)/2 + s)
         / (gamma(1 - s + (b - a)/2)*gamma(1 - s + (a - b)/2)
            *gamma( 1 - s + (a + b)/2)),
            (-(re(a) + re(b))/2, S.Half), True)
    assert MT(besselj(a, sqrt(x))**2 + besselj(-a, sqrt(x))**2, x, s)[1:] == \
        ((Max(re(a), -re(a)), S.Half), True)

    # Section 8.4.20
    assert MT(bessely(a, 2*sqrt(x)), x, s) == \
        (-cos(pi*(a/2 - s))*gamma(s - a/2)*gamma(s + a/2)/pi,
            (Max(-re(a)/2, re(a)/2), Rational(3, 4)), True)
    assert MT(sin(sqrt(x))*bessely(a, sqrt(x)), x, s) == \
        (-4**s*sin(pi*(a/2 - s))*gamma(S.Half - 2*s)
         * gamma((1 - a)/2 + s)*gamma((1 + a)/2 + s)
         / (sqrt(pi)*gamma(1 - s - a/2)*gamma(1 - s + a/2)),
            (Max(-(re(a) + 1)/2, (re(a) - 1)/2), Rational(1, 4)), True)
    assert MT(cos(sqrt(x))*bessely(a, sqrt(x)), x, s) == \
        (-4**s*cos(pi*(a/2 - s))*gamma(s - a/2)*gamma(s + a/2)*gamma(S.Half - 2*s)
         / (sqrt(pi)*gamma(S.Half - s - a/2)*gamma(S.Half - s + a/2)),
            (Max(-re(a)/2, re(a)/2), Rational(1, 4)), True)
    assert MT(besselj(a, sqrt(x))*bessely(a, sqrt(x)), x, s) == \
        (-cos(pi*s)*gamma(s)*gamma(a + s)*gamma(S.Half - s)
         / (pi**S('3/2')*gamma(1 + a - s)),
            (Max(-re(a), 0), S.Half), True)
    assert MT(besselj(a, sqrt(x))*bessely(b, sqrt(x)), x, s) == \
        (-4**s*cos(pi*(a/2 - b/2 + s))*gamma(1 - 2*s)
         * gamma(a/2 - b/2 + s)*gamma(a/2 + b/2 + s)
         / (pi*gamma(a/2 - b/2 - s + 1)*gamma(a/2 + b/2 - s + 1)),
            (Max((-re(a) + re(b))/2, (-re(a) - re(b))/2), S.Half), True)
    # NOTE bessely(a, sqrt(x))**2 and bessely(a, sqrt(x))*bessely(b, sqrt(x))
    # are a mess (no matter what way you look at it ...)
    assert MT(bessely(a, sqrt(x))**2, x, s)[1:] == \
             ((Max(-re(a), 0, re(a)), S.Half), True)

    # Section 8.4.22
    # TODO we can't do any of these (delicate cancellation)

    # Section 8.4.23
    assert MT(besselk(a, 2*sqrt(x)), x, s) == \
        (gamma(
         s - a/2)*gamma(s + a/2)/2, (Max(-re(a)/2, re(a)/2), oo), True)
    assert MT(besselj(a, 2*sqrt(2*sqrt(x)))*besselk(
        a, 2*sqrt(2*sqrt(x))), x, s) == (4**(-s)*gamma(2*s)*
        gamma(a/2 + s)/(2*gamma(a/2 - s + 1)), (Max(0, -re(a)/2), oo), True)
    # TODO bessely(a, x)*besselk(a, x) is a mess
    assert MT(besseli(a, sqrt(x))*besselk(a, sqrt(x)), x, s) == \
        (gamma(s)*gamma(
        a + s)*gamma(-s + S.Half)/(2*sqrt(pi)*gamma(a - s + 1)),
        (Max(-re(a), 0), S.Half), True)
    assert MT(besseli(b, sqrt(x))*besselk(a, sqrt(x)), x, s) == \
        (2**(2*s - 1)*gamma(-2*s + 1)*gamma(-a/2 + b/2 + s)* \
        gamma(a/2 + b/2 + s)/(gamma(-a/2 + b/2 - s + 1)* \
        gamma(a/2 + b/2 - s + 1)), (Max(-re(a)/2 - re(b)/2, \
        re(a)/2 - re(b)/2), S.Half), True)

    # TODO products of besselk are a mess

    mt = MT(exp(-x/2)*besselk(a, x/2), x, s)
    mt0 = gammasimp(trigsimp(gammasimp(mt[0].expand(func=True))))
    assert mt0 == 2*pi**Rational(3, 2)*cos(pi*s)*gamma(S.Half - s)/(
        (cos(2*pi*a) - cos(2*pi*s))*gamma(-a - s + 1)*gamma(a - s + 1))
    assert mt[1:] == ((Max(-re(a), re(a)), oo), True)
    # TODO exp(x/2)*besselk(a, x/2) [etc] cannot currently be done
    # TODO various strange products of special orders


@slow
def test_expint():
    from sympy.functions.elementary.miscellaneous import Max
    from sympy.functions.special.error_functions import Ci, E1, Si
    from sympy.simplify.simplify import simplify

    aneg = Symbol('a', negative=True)
    u = Symbol('u', polar=True)

    assert mellin_transform(E1(x), x, s) == (gamma(s)/s, (0, oo), True)
    assert inverse_mellin_transform(gamma(s)/s, s, x,
              (0, oo)).rewrite(expint).expand() == E1(x)
    assert mellin_transform(expint(a, x), x, s) == \
        (gamma(s)/(a + s - 1), (Max(1 - re(a), 0), oo), True)
    # XXX IMT has hickups with complicated strips ...
    assert simplify(unpolarify(
                    inverse_mellin_transform(gamma(s)/(aneg + s - 1), s, x,
                  (1 - aneg, oo)).rewrite(expint).expand(func=True))) == \
        expint(aneg, x)

    assert mellin_transform(Si(x), x, s) == \
        (-2**s*sqrt(pi)*gamma(s/2 + S.Half)/(
        2*s*gamma(-s/2 + 1)), (-1, 0), True)
    assert inverse_mellin_transform(-2**s*sqrt(pi)*gamma((s + 1)/2)
                                    /(2*s*gamma(-s/2 + 1)), s, x, (-1, 0)) \
        == Si(x)

    assert mellin_transform(Ci(sqrt(x)), x, s) == \
        (-2**(2*s - 1)*sqrt(pi)*gamma(s)/(s*gamma(-s + S.Half)), (0, 1), True)
    assert inverse_mellin_transform(
        -4**s*sqrt(pi)*gamma(s)/(2*s*gamma(-s + S.Half)),
        s, u, (0, 1)).expand() == Ci(sqrt(u))


@slow
def test_inverse_mellin_transform():
    from sympy.core.function import expand
    from sympy.functions.elementary.miscellaneous import (Max, Min)
    from sympy.functions.elementary.trigonometric import cot
    from sympy.simplify.powsimp import powsimp
    from sympy.simplify.simplify import simplify
    IMT = inverse_mellin_transform

    assert IMT(gamma(s), s, x, (0, oo)) == exp(-x)
    assert IMT(gamma(-s), s, x, (-oo, 0)) == exp(-1/x)
    assert simplify(IMT(s/(2*s**2 - 2), s, x, (2, oo))) == \
        (x**2 + 1)*Heaviside(1 - x)/(4*x)

    # test passing "None"
    assert IMT(1/(s**2 - 1), s, x, (-1, None)) == \
        -x*Heaviside(-x + 1)/2 - Heaviside(x - 1)/(2*x)
    assert IMT(1/(s**2 - 1), s, x, (None, 1)) == \
        -x*Heaviside(-x + 1)/2 - Heaviside(x - 1)/(2*x)

    # test expansion of sums
    assert IMT(gamma(s) + gamma(s - 1), s, x, (1, oo)) == (x + 1)*exp(-x)/x

    # test factorisation of polys
    r = symbols('r', real=True)
    assert IMT(1/(s**2 + 1), s, exp(-x), (None, oo)
              ).subs(x, r).rewrite(sin).simplify() \
        == sin(r)*Heaviside(1 - exp(-r))

    # test multiplicative substitution
    _a, _b = symbols('a b', positive=True)
    assert IMT(_b**(-s/_a)*factorial(s/_a)/s, s, x, (0, oo)) == exp(-_b*x**_a)
    assert IMT(factorial(_a/_b + s/_b)/(_a + s), s, x, (-_a, oo)) == x**_a*exp(-x**_b)

    def simp_pows(expr):
        return simplify(powsimp(expand_mul(expr, deep=False), force=True)).replace(exp_polar, exp)

    # Now test the inverses of all direct transforms tested above

    # Section 8.4.2
    nu = symbols('nu', real=True)
    assert IMT(-1/(nu + s), s, x, (-oo, None)) == x**nu*Heaviside(x - 1)
    assert IMT(1/(nu + s), s, x, (None, oo)) == x**nu*Heaviside(1 - x)
    assert simp_pows(IMT(gamma(beta)*gamma(s)/gamma(s + beta), s, x, (0, oo))) \
        == (1 - x)**(beta - 1)*Heaviside(1 - x)
    assert simp_pows(IMT(gamma(beta)*gamma(1 - beta - s)/gamma(1 - s),
                         s, x, (-oo, None))) \
        == (x - 1)**(beta - 1)*Heaviside(x - 1)
    assert simp_pows(IMT(gamma(s)*gamma(rho - s)/gamma(rho), s, x, (0, None))) \
        == (1/(x + 1))**rho
    expr = IMT(d**c*d**(s - 1)*sin(pi*c)
                         *gamma(s)*gamma(s + c)*gamma(1 - s)*gamma(1 - s - c)/pi,
                         s, x, (Max(-re(c), 0), Min(1 - re(c), 1)))
    assert powsimp(expand_mul(expr, deep=False)).replace(exp_polar, exp).simplify() \
        == (-d**c + x**c)/(-d + x)

    assert simplify(IMT(1/sqrt(pi)*(-c/2)*gamma(s)*gamma((1 - c)/2 - s)
                        *gamma(-c/2 - s)/gamma(1 - c - s),
                        s, x, (0, -re(c)/2))) == \
        (1 + sqrt(x + 1))**c
    assert simplify(IMT(2**(a + 2*s)*b**(a + 2*s - 1)*gamma(s)*gamma(1 - a - 2*s)
                        /gamma(1 - a - s), s, x, (0, (-re(a) + 1)/2))) == \
        b**(a - 1)*(b**2*(sqrt(1 + x/b**2) + 1)**a + x*(sqrt(1 + x/b**2) + 1
        )**(a - 1))/(b**2 + x)
    assert simplify(IMT(-2**(c + 2*s)*c*b**(c + 2*s)*gamma(s)*gamma(-c - 2*s)
                        / gamma(-c - s + 1), s, x, (0, -re(c)/2))) == \
        b**c*(sqrt(1 + x/b**2) + 1)**c

    # Section 8.4.5
    assert IMT(24/s**5, s, x, (0, oo)) == log(x)**4*Heaviside(1 - x)
    assert expand(IMT(6/s**4, s, x, (-oo, 0)), force=True) == \
        log(x)**3*Heaviside(x - 1)
    assert IMT(pi/(s*sin(pi*s)), s, x, (-1, 0)) == log(x + 1)
    assert IMT(pi/(s*sin(pi*s/2)), s, x, (-2, 0)) == log(x**2 + 1)
    assert IMT(pi/(s*sin(2*pi*s)), s, x, (Rational(-1, 2), 0)) == log(sqrt(x) + 1)
    assert IMT(pi/(s*sin(pi*s)), s, x, (0, 1)) == log(1 + 1/x)

    # TODO
    def mysimp(expr):
        from sympy.core.function import expand
        from sympy.simplify.powsimp import powsimp
        from sympy.simplify.simplify import logcombine
        return expand(
            powsimp(logcombine(expr, force=True), force=True, deep=True),
            force=True).replace(exp_polar, exp)

    assert mysimp(mysimp(IMT(pi/(s*tan(pi*s)), s, x, (-1, 0)))) in [
        log(1 - x)*Heaviside(1 - x) + log(x - 1)*Heaviside(x - 1),
        log(x)*Heaviside(x - 1) + log(1 - 1/x)*Heaviside(x - 1) + log(-x +
        1)*Heaviside(-x + 1)]
    # test passing cot
    assert mysimp(IMT(pi*cot(pi*s)/s, s, x, (0, 1))) in [
        log(1/x - 1)*Heaviside(1 - x) + log(1 - 1/x)*Heaviside(x - 1),
        -log(x)*Heaviside(-x + 1) + log(1 - 1/x)*Heaviside(x - 1) + log(-x +
        1)*Heaviside(-x + 1), ]

    # 8.4.14
    assert IMT(-gamma(s + S.Half)/(sqrt(pi)*s), s, x, (Rational(-1, 2), 0)) == \
        erf(sqrt(x))

    # 8.4.19
    assert simplify(IMT(gamma(a/2 + s)/gamma(a/2 - s + 1), s, x, (-re(a)/2, Rational(3, 4)))) \
        == besselj(a, 2*sqrt(x))
    assert simplify(IMT(2**a*gamma(S.Half - 2*s)*gamma(s + (a + 1)/2)
                      / (gamma(1 - s - a/2)*gamma(1 - 2*s + a)),
                      s, x, (-(re(a) + 1)/2, Rational(1, 4)))) == \
        sin(sqrt(x))*besselj(a, sqrt(x))
    assert simplify(IMT(2**a*gamma(a/2 + s)*gamma(S.Half - 2*s)
                      / (gamma(S.Half - s - a/2)*gamma(1 - 2*s + a)),
                      s, x, (-re(a)/2, Rational(1, 4)))) == \
        cos(sqrt(x))*besselj(a, sqrt(x))
    # TODO this comes out as an amazing mess, but simplifies nicely
    assert simplify(IMT(gamma(a + s)*gamma(S.Half - s)
                      / (sqrt(pi)*gamma(1 - s)*gamma(1 + a - s)),
                      s, x, (-re(a), S.Half))) == \
        besselj(a, sqrt(x))**2
    assert simplify(IMT(gamma(s)*gamma(S.Half - s)
                      / (sqrt(pi)*gamma(1 - s - a)*gamma(1 + a - s)),
                      s, x, (0, S.Half))) == \
        besselj(-a, sqrt(x))*besselj(a, sqrt(x))
    assert simplify(IMT(4**s*gamma(-2*s + 1)*gamma(a/2 + b/2 + s)
                      / (gamma(-a/2 + b/2 - s + 1)*gamma(a/2 - b/2 - s + 1)
                         *gamma(a/2 + b/2 - s + 1)),
                      s, x, (-(re(a) + re(b))/2, S.Half))) == \
        besselj(a, sqrt(x))*besselj(b, sqrt(x))

    # Section 8.4.20
    # TODO this can be further simplified!
    assert simplify(IMT(-2**(2*s)*cos(pi*a/2 - pi*b/2 + pi*s)*gamma(-2*s + 1) *
                    gamma(a/2 - b/2 + s)*gamma(a/2 + b/2 + s) /
                    (pi*gamma(a/2 - b/2 - s + 1)*gamma(a/2 + b/2 - s + 1)),
                    s, x,
                    (Max(-re(a)/2 - re(b)/2, -re(a)/2 + re(b)/2), S.Half))) == \
                    besselj(a, sqrt(x))*-(besselj(-b, sqrt(x)) -
                    besselj(b, sqrt(x))*cos(pi*b))/sin(pi*b)
    # TODO more

    # for coverage

    assert IMT(pi/cos(pi*s), s, x, (0, S.Half)) == sqrt(x)/(x + 1)


def test_fourier_transform():
    from sympy.core.function import (expand, expand_complex, expand_trig)
    from sympy.polys.polytools import factor
    from sympy.simplify.simplify import simplify
    FT = fourier_transform
    IFT = inverse_fourier_transform

    def simp(x):
        return simplify(expand_trig(expand_complex(expand(x))))

    def sinc(x):
        return sin(pi*x)/(pi*x)
    k = symbols('k', real=True)
    f = Function("f")

    # TODO for this to work with real a, need to expand abs(a*x) to abs(a)*abs(x)
    a = symbols('a', positive=True)
    b = symbols('b', positive=True)

    posk = symbols('posk', positive=True)

    # Test unevaluated form
    assert fourier_transform(f(x), x, k) == FourierTransform(f(x), x, k)
    assert inverse_fourier_transform(
        f(k), k, x) == InverseFourierTransform(f(k), k, x)

    # basic examples from wikipedia
    assert simp(FT(Heaviside(1 - abs(2*a*x)), x, k)) == sinc(k/a)/a
    # TODO IFT is a *mess*
    assert simp(FT(Heaviside(1 - abs(a*x))*(1 - abs(a*x)), x, k)) == sinc(k/a)**2/a
    # TODO IFT

    assert factor(FT(exp(-a*x)*Heaviside(x), x, k), extension=I) == \
        1/(a + 2*pi*I*k)
    # NOTE: the ift comes out in pieces
    assert IFT(1/(a + 2*pi*I*x), x, posk,
            noconds=False) == (exp(-a*posk), True)
    assert IFT(1/(a + 2*pi*I*x), x, -posk,
            noconds=False) == (0, True)
    assert IFT(1/(a + 2*pi*I*x), x, symbols('k', negative=True),
            noconds=False) == (0, True)
    # TODO IFT without factoring comes out as meijer g

    assert factor(FT(x*exp(-a*x)*Heaviside(x), x, k), extension=I) == \
        1/(a + 2*pi*I*k)**2
    assert FT(exp(-a*x)*sin(b*x)*Heaviside(x), x, k) == \
        b/(b**2 + (a + 2*I*pi*k)**2)

    assert FT(exp(-a*x**2), x, k) == sqrt(pi)*exp(-pi**2*k**2/a)/sqrt(a)
    assert IFT(sqrt(pi/a)*exp(-(pi*k)**2/a), k, x) == exp(-a*x**2)
    assert FT(exp(-a*abs(x)), x, k) == 2*a/(a**2 + 4*pi**2*k**2)
    # TODO IFT (comes out as meijer G)

    # TODO besselj(n, x), n an integer > 0 actually can be done...

    # TODO are there other common transforms (no distributions!)?


def test_sine_transform():
    t = symbols("t")
    w = symbols("w")
    a = symbols("a")
    f = Function("f")

    # Test unevaluated form
    assert sine_transform(f(t), t, w) == SineTransform(f(t), t, w)
    assert inverse_sine_transform(
        f(w), w, t) == InverseSineTransform(f(w), w, t)

    assert sine_transform(1/sqrt(t), t, w) == 1/sqrt(w)
    assert inverse_sine_transform(1/sqrt(w), w, t) == 1/sqrt(t)

    assert sine_transform((1/sqrt(t))**3, t, w) == 2*sqrt(w)

    assert sine_transform(t**(-a), t, w) == 2**(
        -a + S.Half)*w**(a - 1)*gamma(-a/2 + 1)/gamma((a + 1)/2)
    assert inverse_sine_transform(2**(-a + S(
        1)/2)*w**(a - 1)*gamma(-a/2 + 1)/gamma(a/2 + S.Half), w, t) == t**(-a)

    assert sine_transform(
        exp(-a*t), t, w) == sqrt(2)*w/(sqrt(pi)*(a**2 + w**2))
    assert inverse_sine_transform(
        sqrt(2)*w/(sqrt(pi)*(a**2 + w**2)), w, t) == exp(-a*t)

    assert sine_transform(
        log(t)/t, t, w) == sqrt(2)*sqrt(pi)*-(log(w**2) + 2*EulerGamma)/4

    assert sine_transform(
        t*exp(-a*t**2), t, w) == sqrt(2)*w*exp(-w**2/(4*a))/(4*a**Rational(3, 2))
    assert inverse_sine_transform(
        sqrt(2)*w*exp(-w**2/(4*a))/(4*a**Rational(3, 2)), w, t) == t*exp(-a*t**2)


def test_cosine_transform():
    from sympy.functions.special.error_functions import (Ci, Si)

    t = symbols("t")
    w = symbols("w")
    a = symbols("a")
    f = Function("f")

    # Test unevaluated form
    assert cosine_transform(f(t), t, w) == CosineTransform(f(t), t, w)
    assert inverse_cosine_transform(
        f(w), w, t) == InverseCosineTransform(f(w), w, t)

    assert cosine_transform(1/sqrt(t), t, w) == 1/sqrt(w)
    assert inverse_cosine_transform(1/sqrt(w), w, t) == 1/sqrt(t)

    assert cosine_transform(1/(
        a**2 + t**2), t, w) == sqrt(2)*sqrt(pi)*exp(-a*w)/(2*a)

    assert cosine_transform(t**(
        -a), t, w) == 2**(-a + S.Half)*w**(a - 1)*gamma((-a + 1)/2)/gamma(a/2)
    assert inverse_cosine_transform(2**(-a + S(
        1)/2)*w**(a - 1)*gamma(-a/2 + S.Half)/gamma(a/2), w, t) == t**(-a)

    assert cosine_transform(
        exp(-a*t), t, w) == sqrt(2)*a/(sqrt(pi)*(a**2 + w**2))
    assert inverse_cosine_transform(
        sqrt(2)*a/(sqrt(pi)*(a**2 + w**2)), w, t) == exp(-a*t)

    assert cosine_transform(exp(-a*sqrt(t))*cos(a*sqrt(
        t)), t, w) == a*exp(-a**2/(2*w))/(2*w**Rational(3, 2))

    assert cosine_transform(1/(a + t), t, w) == sqrt(2)*(
        (-2*Si(a*w) + pi)*sin(a*w)/2 - cos(a*w)*Ci(a*w))/sqrt(pi)
    assert inverse_cosine_transform(sqrt(2)*meijerg(((S.Half, 0), ()), (
        (S.Half, 0, 0), (S.Half,)), a**2*w**2/4)/(2*pi), w, t) == 1/(a + t)

    assert cosine_transform(1/sqrt(a**2 + t**2), t, w) == sqrt(2)*meijerg(
        ((S.Half,), ()), ((0, 0), (S.Half,)), a**2*w**2/4)/(2*sqrt(pi))
    assert inverse_cosine_transform(sqrt(2)*meijerg(((S.Half,), ()), ((0, 0), (S.Half,)), a**2*w**2/4)/(2*sqrt(pi)), w, t) == 1/(t*sqrt(a**2/t**2 + 1))


def test_hankel_transform():
    r = Symbol("r")
    k = Symbol("k")
    nu = Symbol("nu")
    m = Symbol("m")
    a = symbols("a")

    assert hankel_transform(1/r, r, k, 0) == 1/k
    assert inverse_hankel_transform(1/k, k, r, 0) == 1/r

    assert hankel_transform(
        1/r**m, r, k, 0) == 2**(-m + 1)*k**(m - 2)*gamma(-m/2 + 1)/gamma(m/2)
    assert inverse_hankel_transform(
        2**(-m + 1)*k**(m - 2)*gamma(-m/2 + 1)/gamma(m/2), k, r, 0) == r**(-m)

    assert hankel_transform(1/r**m, r, k, nu) == (
        2*2**(-m)*k**(m - 2)*gamma(-m/2 + nu/2 + 1)/gamma(m/2 + nu/2))
    assert inverse_hankel_transform(2**(-m + 1)*k**(
        m - 2)*gamma(-m/2 + nu/2 + 1)/gamma(m/2 + nu/2), k, r, nu) == r**(-m)

    assert hankel_transform(r**nu*exp(-a*r), r, k, nu) == \
        2**(nu + 1)*a*k**(-nu - 3)*(a**2/k**2 + 1)**(-nu - S(
                                                     3)/2)*gamma(nu + Rational(3, 2))/sqrt(pi)
    assert inverse_hankel_transform(
        2**(nu + 1)*a*k**(-nu - 3)*(a**2/k**2 + 1)**(-nu - Rational(3, 2))*gamma(
        nu + Rational(3, 2))/sqrt(pi), k, r, nu) == r**nu*exp(-a*r)


def test_issue_7181():
    assert mellin_transform(1/(1 - x), x, s) != None


def test_issue_8882():
    # This is the original test.
    # from sympy import diff, Integral, integrate
    # r = Symbol('r')
    # psi = 1/r*sin(r)*exp(-(a0*r))
    # h = -1/2*diff(psi, r, r) - 1/r*psi
    # f = 4*pi*psi*h*r**2
    # assert integrate(f, (r, -oo, 3), meijerg=True).has(Integral) == True

    # To save time, only the critical part is included.
    F = -a**(-s + 1)*(4 + 1/a**2)**(-s/2)*sqrt(1/a**2)*exp(-s*I*pi)* \
        sin(s*atan(sqrt(1/a**2)/2))*gamma(s)
    raises(IntegralTransformError, lambda:
        inverse_mellin_transform(F, s, x, (-1, oo),
        **{'as_meijerg': True, 'needeval': True}))


def test_issue_12591():
    x, y = symbols("x y", real=True)
    assert fourier_transform(exp(x), x, y) == FourierTransform(exp(x), x, y)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_filters.py ===
from string import ascii_lowercase

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    Timestamp,
)
import pandas._testing as tm


def test_filter_series():
    s = Series([1, 3, 20, 5, 22, 24, 7])
    expected_odd = Series([1, 3, 5, 7], index=[0, 1, 3, 6])
    expected_even = Series([20, 22, 24], index=[2, 4, 5])
    grouper = s.apply(lambda x: x % 2)
    grouped = s.groupby(grouper)
    tm.assert_series_equal(grouped.filter(lambda x: x.mean() < 10), expected_odd)
    tm.assert_series_equal(grouped.filter(lambda x: x.mean() > 10), expected_even)
    # Test dropna=False.
    tm.assert_series_equal(
        grouped.filter(lambda x: x.mean() < 10, dropna=False),
        expected_odd.reindex(s.index),
    )
    tm.assert_series_equal(
        grouped.filter(lambda x: x.mean() > 10, dropna=False),
        expected_even.reindex(s.index),
    )


def test_filter_single_column_df():
    df = DataFrame([1, 3, 20, 5, 22, 24, 7])
    expected_odd = DataFrame([1, 3, 5, 7], index=[0, 1, 3, 6])
    expected_even = DataFrame([20, 22, 24], index=[2, 4, 5])
    grouper = df[0].apply(lambda x: x % 2)
    grouped = df.groupby(grouper)
    tm.assert_frame_equal(grouped.filter(lambda x: x.mean() < 10), expected_odd)
    tm.assert_frame_equal(grouped.filter(lambda x: x.mean() > 10), expected_even)
    # Test dropna=False.
    tm.assert_frame_equal(
        grouped.filter(lambda x: x.mean() < 10, dropna=False),
        expected_odd.reindex(df.index),
    )
    tm.assert_frame_equal(
        grouped.filter(lambda x: x.mean() > 10, dropna=False),
        expected_even.reindex(df.index),
    )


def test_filter_multi_column_df():
    df = DataFrame({"A": [1, 12, 12, 1], "B": [1, 1, 1, 1]})
    grouper = df["A"].apply(lambda x: x % 2)
    grouped = df.groupby(grouper)
    expected = DataFrame({"A": [12, 12], "B": [1, 1]}, index=[1, 2])
    tm.assert_frame_equal(
        grouped.filter(lambda x: x["A"].sum() - x["B"].sum() > 10), expected
    )


def test_filter_mixed_df():
    df = DataFrame({"A": [1, 12, 12, 1], "B": "a b c d".split()})
    grouper = df["A"].apply(lambda x: x % 2)
    grouped = df.groupby(grouper)
    expected = DataFrame({"A": [12, 12], "B": ["b", "c"]}, index=[1, 2])
    tm.assert_frame_equal(grouped.filter(lambda x: x["A"].sum() > 10), expected)


def test_filter_out_all_groups():
    s = Series([1, 3, 20, 5, 22, 24, 7])
    grouper = s.apply(lambda x: x % 2)
    grouped = s.groupby(grouper)
    tm.assert_series_equal(grouped.filter(lambda x: x.mean() > 1000), s[[]])
    df = DataFrame({"A": [1, 12, 12, 1], "B": "a b c d".split()})
    grouper = df["A"].apply(lambda x: x % 2)
    grouped = df.groupby(grouper)
    tm.assert_frame_equal(grouped.filter(lambda x: x["A"].sum() > 1000), df.loc[[]])


def test_filter_out_no_groups():
    s = Series([1, 3, 20, 5, 22, 24, 7])
    grouper = s.apply(lambda x: x % 2)
    grouped = s.groupby(grouper)
    filtered = grouped.filter(lambda x: x.mean() > 0)
    tm.assert_series_equal(filtered, s)
    df = DataFrame({"A": [1, 12, 12, 1], "B": "a b c d".split()})
    grouper = df["A"].apply(lambda x: x % 2)
    grouped = df.groupby(grouper)
    filtered = grouped.filter(lambda x: x["A"].mean() > 0)
    tm.assert_frame_equal(filtered, df)


def test_filter_out_all_groups_in_df():
    # GH12768
    df = DataFrame({"a": [1, 1, 2], "b": [1, 2, 0]})
    res = df.groupby("a")
    res = res.filter(lambda x: x["b"].sum() > 5, dropna=False)
    expected = DataFrame({"a": [np.nan] * 3, "b": [np.nan] * 3})
    tm.assert_frame_equal(expected, res)

    df = DataFrame({"a": [1, 1, 2], "b": [1, 2, 0]})
    res = df.groupby("a")
    res = res.filter(lambda x: x["b"].sum() > 5, dropna=True)
    expected = DataFrame({"a": [], "b": []}, dtype="int64")
    tm.assert_frame_equal(expected, res)


def test_filter_condition_raises():
    def raise_if_sum_is_zero(x):
        if x.sum() == 0:
            raise ValueError
        return x.sum() > 0

    s = Series([-1, 0, 1, 2])
    grouper = s.apply(lambda x: x % 2)
    grouped = s.groupby(grouper)
    msg = "the filter must return a boolean result"
    with pytest.raises(TypeError, match=msg):
        grouped.filter(raise_if_sum_is_zero)


def test_filter_with_axis_in_groupby():
    # issue 11041
    index = pd.MultiIndex.from_product([range(10), [0, 1]])
    data = DataFrame(np.arange(100).reshape(-1, 20), columns=index, dtype="int64")

    msg = "DataFrame.groupby with axis=1"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = data.groupby(level=0, axis=1)
    result = gb.filter(lambda x: x.iloc[0, 0] > 10)
    expected = data.iloc[:, 12:20]
    tm.assert_frame_equal(result, expected)


def test_filter_bad_shapes():
    df = DataFrame({"A": np.arange(8), "B": list("aabbbbcc"), "C": np.arange(8)})
    s = df["B"]
    g_df = df.groupby("B")
    g_s = s.groupby(s)

    f = lambda x: x
    msg = "filter function returned a DataFrame, but expected a scalar bool"
    with pytest.raises(TypeError, match=msg):
        g_df.filter(f)
    msg = "the filter must return a boolean result"
    with pytest.raises(TypeError, match=msg):
        g_s.filter(f)

    f = lambda x: x == 1
    msg = "filter function returned a DataFrame, but expected a scalar bool"
    with pytest.raises(TypeError, match=msg):
        g_df.filter(f)
    msg = "the filter must return a boolean result"
    with pytest.raises(TypeError, match=msg):
        g_s.filter(f)

    f = lambda x: np.outer(x, x)
    msg = "can't multiply sequence by non-int of type 'str'"
    with pytest.raises(TypeError, match=msg):
        g_df.filter(f)
    msg = "the filter must return a boolean result"
    with pytest.raises(TypeError, match=msg):
        g_s.filter(f)


def test_filter_nan_is_false():
    df = DataFrame({"A": np.arange(8), "B": list("aabbbbcc"), "C": np.arange(8)})
    s = df["B"]
    g_df = df.groupby(df["B"])
    g_s = s.groupby(s)

    f = lambda x: np.nan
    tm.assert_frame_equal(g_df.filter(f), df.loc[[]])
    tm.assert_series_equal(g_s.filter(f), s[[]])


def test_filter_pdna_is_false():
    # in particular, dont raise in filter trying to call bool(pd.NA)
    df = DataFrame({"A": np.arange(8), "B": list("aabbbbcc"), "C": np.arange(8)})
    ser = df["B"]
    g_df = df.groupby(df["B"])
    g_s = ser.groupby(ser)

    func = lambda x: pd.NA
    res = g_df.filter(func)
    tm.assert_frame_equal(res, df.loc[[]])
    res = g_s.filter(func)
    tm.assert_series_equal(res, ser[[]])


def test_filter_against_workaround_ints():
    # Series of ints
    s = Series(np.random.default_rng(2).integers(0, 100, 100))
    grouper = s.apply(lambda x: np.round(x, -1))
    grouped = s.groupby(grouper)
    f = lambda x: x.mean() > 10

    old_way = s[grouped.transform(f).astype("bool")]
    new_way = grouped.filter(f)
    tm.assert_series_equal(new_way.sort_values(), old_way.sort_values())


def test_filter_against_workaround_floats():
    # Series of floats
    s = 100 * Series(np.random.default_rng(2).random(100))
    grouper = s.apply(lambda x: np.round(x, -1))
    grouped = s.groupby(grouper)
    f = lambda x: x.mean() > 10
    old_way = s[grouped.transform(f).astype("bool")]
    new_way = grouped.filter(f)
    tm.assert_series_equal(new_way.sort_values(), old_way.sort_values())


def test_filter_against_workaround_dataframe():
    # Set up DataFrame of ints, floats, strings.
    letters = np.array(list(ascii_lowercase))
    N = 100
    random_letters = letters.take(
        np.random.default_rng(2).integers(0, 26, N, dtype=int)
    )
    df = DataFrame(
        {
            "ints": Series(np.random.default_rng(2).integers(0, 100, N)),
            "floats": N / 10 * Series(np.random.default_rng(2).random(N)),
            "letters": Series(random_letters),
        }
    )

    # Group by ints; filter on floats.
    grouped = df.groupby("ints")
    old_way = df[grouped.floats.transform(lambda x: x.mean() > N / 20).astype("bool")]
    new_way = grouped.filter(lambda x: x["floats"].mean() > N / 20)
    tm.assert_frame_equal(new_way, old_way)

    # Group by floats (rounded); filter on strings.
    grouper = df.floats.apply(lambda x: np.round(x, -1))
    grouped = df.groupby(grouper)
    old_way = df[grouped.letters.transform(lambda x: len(x) < N / 10).astype("bool")]
    new_way = grouped.filter(lambda x: len(x.letters) < N / 10)
    tm.assert_frame_equal(new_way, old_way)

    # Group by strings; filter on ints.
    grouped = df.groupby("letters")
    old_way = df[grouped.ints.transform(lambda x: x.mean() > N / 20).astype("bool")]
    new_way = grouped.filter(lambda x: x["ints"].mean() > N / 20)
    tm.assert_frame_equal(new_way, old_way)


def test_filter_using_len():
    # BUG GH4447
    df = DataFrame({"A": np.arange(8), "B": list("aabbbbcc"), "C": np.arange(8)})
    grouped = df.groupby("B")
    actual = grouped.filter(lambda x: len(x) > 2)
    expected = DataFrame(
        {"A": np.arange(2, 6), "B": list("bbbb"), "C": np.arange(2, 6)},
        index=np.arange(2, 6, dtype=np.int64),
    )
    tm.assert_frame_equal(actual, expected)

    actual = grouped.filter(lambda x: len(x) > 4)
    expected = df.loc[[]]
    tm.assert_frame_equal(actual, expected)

    # Series have always worked properly, but we'll test anyway.
    s = df["B"]
    grouped = s.groupby(s)
    actual = grouped.filter(lambda x: len(x) > 2)
    expected = Series(4 * ["b"], index=np.arange(2, 6, dtype=np.int64), name="B")
    tm.assert_series_equal(actual, expected)

    actual = grouped.filter(lambda x: len(x) > 4)
    expected = s[[]]
    tm.assert_series_equal(actual, expected)


def test_filter_maintains_ordering():
    # Simple case: index is sequential. #4621
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]}
    )
    s = df["pid"]
    grouped = df.groupby("tag")
    actual = grouped.filter(lambda x: len(x) > 1)
    expected = df.iloc[[1, 2, 4, 7]]
    tm.assert_frame_equal(actual, expected)

    grouped = s.groupby(df["tag"])
    actual = grouped.filter(lambda x: len(x) > 1)
    expected = s.iloc[[1, 2, 4, 7]]
    tm.assert_series_equal(actual, expected)

    # Now index is sequentially decreasing.
    df.index = np.arange(len(df) - 1, -1, -1)
    s = df["pid"]
    grouped = df.groupby("tag")
    actual = grouped.filter(lambda x: len(x) > 1)
    expected = df.iloc[[1, 2, 4, 7]]
    tm.assert_frame_equal(actual, expected)

    grouped = s.groupby(df["tag"])
    actual = grouped.filter(lambda x: len(x) > 1)
    expected = s.iloc[[1, 2, 4, 7]]
    tm.assert_series_equal(actual, expected)

    # Index is shuffled.
    SHUFFLED = [4, 6, 7, 2, 1, 0, 5, 3]
    df.index = df.index[SHUFFLED]
    s = df["pid"]
    grouped = df.groupby("tag")
    actual = grouped.filter(lambda x: len(x) > 1)
    expected = df.iloc[[1, 2, 4, 7]]
    tm.assert_frame_equal(actual, expected)

    grouped = s.groupby(df["tag"])
    actual = grouped.filter(lambda x: len(x) > 1)
    expected = s.iloc[[1, 2, 4, 7]]
    tm.assert_series_equal(actual, expected)


def test_filter_multiple_timestamp():
    # GH 10114
    df = DataFrame(
        {
            "A": np.arange(5, dtype="int64"),
            "B": ["foo", "bar", "foo", "bar", "bar"],
            "C": Timestamp("20130101"),
        }
    )

    grouped = df.groupby(["B", "C"])

    result = grouped["A"].filter(lambda x: True)
    tm.assert_series_equal(df["A"], result)

    result = grouped["A"].transform(len)
    expected = Series([2, 3, 2, 3, 3], name="A")
    tm.assert_series_equal(result, expected)

    result = grouped.filter(lambda x: True)
    tm.assert_frame_equal(df, result)

    result = grouped.transform("sum")
    expected = DataFrame({"A": [2, 8, 2, 8, 8]})
    tm.assert_frame_equal(result, expected)

    result = grouped.transform(len)
    expected = DataFrame({"A": [2, 3, 2, 3, 3]})
    tm.assert_frame_equal(result, expected)


def test_filter_and_transform_with_non_unique_int_index():
    # GH4620
    index = [1, 1, 1, 2, 1, 1, 0, 1]
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]},
        index=index,
    )
    grouped_df = df.groupby("tag")
    ser = df["pid"]
    grouped_ser = ser.groupby(df["tag"])
    expected_indexes = [1, 2, 4, 7]

    # Filter DataFrame
    actual = grouped_df.filter(lambda x: len(x) > 1)
    expected = df.iloc[expected_indexes]
    tm.assert_frame_equal(actual, expected)

    actual = grouped_df.filter(lambda x: len(x) > 1, dropna=False)
    # Cast to avoid upcast when setting nan below
    expected = df.copy().astype("float64")
    expected.iloc[[0, 3, 5, 6]] = np.nan
    tm.assert_frame_equal(actual, expected)

    # Filter Series
    actual = grouped_ser.filter(lambda x: len(x) > 1)
    expected = ser.take(expected_indexes)
    tm.assert_series_equal(actual, expected)

    actual = grouped_ser.filter(lambda x: len(x) > 1, dropna=False)
    expected = Series([np.nan, 1, 1, np.nan, 2, np.nan, np.nan, 3], index, name="pid")
    # ^ made manually because this can get confusing!
    tm.assert_series_equal(actual, expected)

    # Transform Series
    actual = grouped_ser.transform(len)
    expected = Series([1, 2, 2, 1, 2, 1, 1, 2], index, name="pid")
    tm.assert_series_equal(actual, expected)

    # Transform (a column from) DataFrameGroupBy
    actual = grouped_df.pid.transform(len)
    tm.assert_series_equal(actual, expected)


def test_filter_and_transform_with_multiple_non_unique_int_index():
    # GH4620
    index = [1, 1, 1, 2, 0, 0, 0, 1]
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]},
        index=index,
    )
    grouped_df = df.groupby("tag")
    ser = df["pid"]
    grouped_ser = ser.groupby(df["tag"])
    expected_indexes = [1, 2, 4, 7]

    # Filter DataFrame
    actual = grouped_df.filter(lambda x: len(x) > 1)
    expected = df.iloc[expected_indexes]
    tm.assert_frame_equal(actual, expected)

    actual = grouped_df.filter(lambda x: len(x) > 1, dropna=False)
    # Cast to avoid upcast when setting nan below
    expected = df.copy().astype("float64")
    expected.iloc[[0, 3, 5, 6]] = np.nan
    tm.assert_frame_equal(actual, expected)

    # Filter Series
    actual = grouped_ser.filter(lambda x: len(x) > 1)
    expected = ser.take(expected_indexes)
    tm.assert_series_equal(actual, expected)

    actual = grouped_ser.filter(lambda x: len(x) > 1, dropna=False)
    expected = Series([np.nan, 1, 1, np.nan, 2, np.nan, np.nan, 3], index, name="pid")
    # ^ made manually because this can get confusing!
    tm.assert_series_equal(actual, expected)

    # Transform Series
    actual = grouped_ser.transform(len)
    expected = Series([1, 2, 2, 1, 2, 1, 1, 2], index, name="pid")
    tm.assert_series_equal(actual, expected)

    # Transform (a column from) DataFrameGroupBy
    actual = grouped_df.pid.transform(len)
    tm.assert_series_equal(actual, expected)


def test_filter_and_transform_with_non_unique_float_index():
    # GH4620
    index = np.array([1, 1, 1, 2, 1, 1, 0, 1], dtype=float)
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]},
        index=index,
    )
    grouped_df = df.groupby("tag")
    ser = df["pid"]
    grouped_ser = ser.groupby(df["tag"])
    expected_indexes = [1, 2, 4, 7]

    # Filter DataFrame
    actual = grouped_df.filter(lambda x: len(x) > 1)
    expected = df.iloc[expected_indexes]
    tm.assert_frame_equal(actual, expected)

    actual = grouped_df.filter(lambda x: len(x) > 1, dropna=False)
    # Cast to avoid upcast when setting nan below
    expected = df.copy().astype("float64")
    expected.iloc[[0, 3, 5, 6]] = np.nan
    tm.assert_frame_equal(actual, expected)

    # Filter Series
    actual = grouped_ser.filter(lambda x: len(x) > 1)
    expected = ser.take(expected_indexes)
    tm.assert_series_equal(actual, expected)

    actual = grouped_ser.filter(lambda x: len(x) > 1, dropna=False)
    expected = Series([np.nan, 1, 1, np.nan, 2, np.nan, np.nan, 3], index, name="pid")
    # ^ made manually because this can get confusing!
    tm.assert_series_equal(actual, expected)

    # Transform Series
    actual = grouped_ser.transform(len)
    expected = Series([1, 2, 2, 1, 2, 1, 1, 2], index, name="pid")
    tm.assert_series_equal(actual, expected)

    # Transform (a column from) DataFrameGroupBy
    actual = grouped_df.pid.transform(len)
    tm.assert_series_equal(actual, expected)


def test_filter_and_transform_with_non_unique_timestamp_index():
    # GH4620
    t0 = Timestamp("2013-09-30 00:05:00")
    t1 = Timestamp("2013-10-30 00:05:00")
    t2 = Timestamp("2013-11-30 00:05:00")
    index = [t1, t1, t1, t2, t1, t1, t0, t1]
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]},
        index=index,
    )
    grouped_df = df.groupby("tag")
    ser = df["pid"]
    grouped_ser = ser.groupby(df["tag"])
    expected_indexes = [1, 2, 4, 7]

    # Filter DataFrame
    actual = grouped_df.filter(lambda x: len(x) > 1)
    expected = df.iloc[expected_indexes]
    tm.assert_frame_equal(actual, expected)

    actual = grouped_df.filter(lambda x: len(x) > 1, dropna=False)
    # Cast to avoid upcast when setting nan below
    expected = df.copy().astype("float64")
    expected.iloc[[0, 3, 5, 6]] = np.nan
    tm.assert_frame_equal(actual, expected)

    # Filter Series
    actual = grouped_ser.filter(lambda x: len(x) > 1)
    expected = ser.take(expected_indexes)
    tm.assert_series_equal(actual, expected)

    actual = grouped_ser.filter(lambda x: len(x) > 1, dropna=False)
    expected = Series([np.nan, 1, 1, np.nan, 2, np.nan, np.nan, 3], index, name="pid")
    # ^ made manually because this can get confusing!
    tm.assert_series_equal(actual, expected)

    # Transform Series
    actual = grouped_ser.transform(len)
    expected = Series([1, 2, 2, 1, 2, 1, 1, 2], index, name="pid")
    tm.assert_series_equal(actual, expected)

    # Transform (a column from) DataFrameGroupBy
    actual = grouped_df.pid.transform(len)
    tm.assert_series_equal(actual, expected)


def test_filter_and_transform_with_non_unique_string_index():
    # GH4620
    index = list("bbbcbbab")
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]},
        index=index,
    )
    grouped_df = df.groupby("tag")
    ser = df["pid"]
    grouped_ser = ser.groupby(df["tag"])
    expected_indexes = [1, 2, 4, 7]

    # Filter DataFrame
    actual = grouped_df.filter(lambda x: len(x) > 1)
    expected = df.iloc[expected_indexes]
    tm.assert_frame_equal(actual, expected)

    actual = grouped_df.filter(lambda x: len(x) > 1, dropna=False)
    # Cast to avoid upcast when setting nan below
    expected = df.copy().astype("float64")
    expected.iloc[[0, 3, 5, 6]] = np.nan
    tm.assert_frame_equal(actual, expected)

    # Filter Series
    actual = grouped_ser.filter(lambda x: len(x) > 1)
    expected = ser.take(expected_indexes)
    tm.assert_series_equal(actual, expected)

    actual = grouped_ser.filter(lambda x: len(x) > 1, dropna=False)
    expected = Series([np.nan, 1, 1, np.nan, 2, np.nan, np.nan, 3], index, name="pid")
    # ^ made manually because this can get confusing!
    tm.assert_series_equal(actual, expected)

    # Transform Series
    actual = grouped_ser.transform(len)
    expected = Series([1, 2, 2, 1, 2, 1, 1, 2], index, name="pid")
    tm.assert_series_equal(actual, expected)

    # Transform (a column from) DataFrameGroupBy
    actual = grouped_df.pid.transform(len)
    tm.assert_series_equal(actual, expected)


def test_filter_has_access_to_grouped_cols():
    df = DataFrame([[1, 2], [1, 3], [5, 6]], columns=["A", "B"])
    g = df.groupby("A")
    # previously didn't have access to col A #????
    filt = g.filter(lambda x: x["A"].sum() == 2)
    tm.assert_frame_equal(filt, df.iloc[[0, 1]])


def test_filter_enforces_scalarness():
    df = DataFrame(
        [
            ["best", "a", "x"],
            ["worst", "b", "y"],
            ["best", "c", "x"],
            ["best", "d", "y"],
            ["worst", "d", "y"],
            ["worst", "d", "y"],
            ["best", "d", "z"],
        ],
        columns=["a", "b", "c"],
    )
    with pytest.raises(TypeError, match="filter function returned a.*"):
        df.groupby("c").filter(lambda g: g["a"] == "best")


def test_filter_non_bool_raises():
    df = DataFrame(
        [
            ["best", "a", 1],
            ["worst", "b", 1],
            ["best", "c", 1],
            ["best", "d", 1],
            ["worst", "d", 1],
            ["worst", "d", 1],
            ["best", "d", 1],
        ],
        columns=["a", "b", "c"],
    )
    with pytest.raises(TypeError, match="filter function returned a.*"):
        df.groupby("a").filter(lambda g: g.c.mean())


def test_filter_dropna_with_empty_groups():
    # GH 10780
    data = Series(np.random.default_rng(2).random(9), index=np.repeat([1, 2, 3], 3))
    grouped = data.groupby(level=0)
    result_false = grouped.filter(lambda x: x.mean() > 1, dropna=False)
    expected_false = Series([np.nan] * 9, index=np.repeat([1, 2, 3], 3))
    tm.assert_series_equal(result_false, expected_false)

    result_true = grouped.filter(lambda x: x.mean() > 1, dropna=True)
    expected_true = Series(index=pd.Index([], dtype=int), dtype=np.float64)
    tm.assert_series_equal(result_true, expected_true)


def test_filter_consistent_result_before_after_agg_func():
    # GH 17091
    df = DataFrame({"data": range(6), "key": list("ABCABC")})
    grouper = df.groupby("key")
    result = grouper.filter(lambda x: True)
    expected = DataFrame({"data": range(6), "key": list("ABCABC")})
    tm.assert_frame_equal(result, expected)

    grouper.sum()
    result = grouper.filter(lambda x: True)
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_aesaracode.py ===
"""
Important note on tests in this module - the Aesara printing functions use a
global cache by default, which means that tests using it will modify global
state and thus not be independent from each other. Instead of using the "cache"
keyword argument each time, this module uses the aesara_code_ and
aesara_function_ functions defined below which default to using a new, empty
cache instead.
"""

import logging

from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy

from sympy.utilities.exceptions import ignore_warnings


aesaralogger = logging.getLogger('aesara.configdefaults')
aesaralogger.setLevel(logging.CRITICAL)
aesara = import_module('aesara')
aesaralogger.setLevel(logging.WARNING)


if aesara:
    import numpy as np
    aet = aesara.tensor
    from aesara.scalar.basic import ScalarType
    from aesara.graph.basic import Variable
    from aesara.tensor.var import TensorVariable
    from aesara.tensor.elemwise import Elemwise, DimShuffle
    from aesara.tensor.math import Dot

    from sympy.printing.aesaracode import true_divide

    xt, yt, zt = [aet.scalar(name, 'floatX') for name in 'xyz']
    Xt, Yt, Zt = [aet.tensor('floatX', (False, False), name=n) for n in 'XYZ']
else:
    #bin/test will not execute any tests now
    disabled = True

import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.aesaracode import (aesara_code, dim_handling,
        aesara_function)


# Default set of matrix symbols for testing - make square so we can both
# multiply and perform elementwise operations between them.
X, Y, Z = [sy.MatrixSymbol(n, 4, 4) for n in 'XYZ']

# For testing AppliedUndef
f_t = sy.Function('f')(t)


def aesara_code_(expr, **kwargs):
    """ Wrapper for aesara_code that uses a new, empty cache by default. """
    kwargs.setdefault('cache', {})
    with warns_deprecated_sympy():
        return aesara_code(expr, **kwargs)

def aesara_function_(inputs, outputs, **kwargs):
    """ Wrapper for aesara_function that uses a new, empty cache by default. """
    kwargs.setdefault('cache', {})
    with warns_deprecated_sympy():
        return aesara_function(inputs, outputs, **kwargs)


def fgraph_of(*exprs):
    """ Transform SymPy expressions into Aesara Computation.

    Parameters
    ==========
    exprs
        SymPy expressions

    Returns
    =======
    aesara.graph.fg.FunctionGraph
    """
    outs = list(map(aesara_code_, exprs))
    ins = list(aesara.graph.basic.graph_inputs(outs))
    ins, outs = aesara.graph.basic.clone(ins, outs)
    return aesara.graph.fg.FunctionGraph(ins, outs)


def aesara_simplify(fgraph):
    """ Simplify a Aesara Computation.

    Parameters
    ==========
    fgraph : aesara.graph.fg.FunctionGraph

    Returns
    =======
    aesara.graph.fg.FunctionGraph
    """
    mode = aesara.compile.get_default_mode().excluding("fusion")
    fgraph = fgraph.clone()
    mode.optimizer.rewrite(fgraph)
    return fgraph


def theq(a, b):
    """ Test two Aesara objects for equality.

    Also accepts numeric types and lists/tuples of supported types.

    Note - debugprint() has a bug where it will accept numeric types but does
    not respect the "file" argument and in this case and instead prints the number
    to stdout and returns an empty string. This can lead to tests passing where
    they should fail because any two numbers will always compare as equal. To
    prevent this we treat numbers as a separate case.
    """
    numeric_types = (int, float, np.number)
    a_is_num = isinstance(a, numeric_types)
    b_is_num = isinstance(b, numeric_types)

    # Compare numeric types using regular equality
    if a_is_num or b_is_num:
        if not (a_is_num and b_is_num):
            return False

        return a == b

    # Compare sequences element-wise
    a_is_seq = isinstance(a, (tuple, list))
    b_is_seq = isinstance(b, (tuple, list))

    if a_is_seq or b_is_seq:
        if not (a_is_seq and b_is_seq) or type(a) != type(b):
            return False

        return list(map(theq, a)) == list(map(theq, b))

    # Otherwise, assume debugprint() can handle it
    astr = aesara.printing.debugprint(a, file='str')
    bstr = aesara.printing.debugprint(b, file='str')

    # Check for bug mentioned above
    for argname, argval, argstr in [('a', a, astr), ('b', b, bstr)]:
        if argstr == '':
            raise TypeError(
                'aesara.printing.debugprint(%s) returned empty string '
                '(%s is instance of %r)'
                % (argname, argname, type(argval))
            )

    return astr == bstr


def test_example_symbols():
    """
    Check that the example symbols in this module print to their Aesara
    equivalents, as many of the other tests depend on this.
    """
    assert theq(xt, aesara_code_(x))
    assert theq(yt, aesara_code_(y))
    assert theq(zt, aesara_code_(z))
    assert theq(Xt, aesara_code_(X))
    assert theq(Yt, aesara_code_(Y))
    assert theq(Zt, aesara_code_(Z))


def test_Symbol():
    """ Test printing a Symbol to a aesara variable. """
    xx = aesara_code_(x)
    assert isinstance(xx, Variable)
    assert xx.broadcastable == ()
    assert xx.name == x.name

    xx2 = aesara_code_(x, broadcastables={x: (False,)})
    assert xx2.broadcastable == (False,)
    assert xx2.name == x.name

def test_MatrixSymbol():
    """ Test printing a MatrixSymbol to a aesara variable. """
    XX = aesara_code_(X)
    assert isinstance(XX, TensorVariable)
    assert XX.broadcastable == (False, False)

@SKIP  # TODO - this is currently not checked but should be implemented
def test_MatrixSymbol_wrong_dims():
    """ Test MatrixSymbol with invalid broadcastable. """
    bcs = [(), (False,), (True,), (True, False), (False, True,), (True, True)]
    for bc in bcs:
        with raises(ValueError):
            aesara_code_(X, broadcastables={X: bc})

def test_AppliedUndef():
    """ Test printing AppliedUndef instance, which works similarly to Symbol. """
    ftt = aesara_code_(f_t)
    assert isinstance(ftt, TensorVariable)
    assert ftt.broadcastable == ()
    assert ftt.name == 'f_t'


def test_add():
    expr = x + y
    comp = aesara_code_(expr)
    assert comp.owner.op == aesara.tensor.add

def test_trig():
    assert theq(aesara_code_(sy.sin(x)), aet.sin(xt))
    assert theq(aesara_code_(sy.tan(x)), aet.tan(xt))

def test_many():
    """ Test printing a complex expression with multiple symbols. """
    expr = sy.exp(x**2 + sy.cos(y)) * sy.log(2*z)
    comp = aesara_code_(expr)
    expected = aet.exp(xt**2 + aet.cos(yt)) * aet.log(2*zt)
    assert theq(comp, expected)


def test_dtype():
    """ Test specifying specific data types through the dtype argument. """
    for dtype in ['float32', 'float64', 'int8', 'int16', 'int32', 'int64']:
        assert aesara_code_(x, dtypes={x: dtype}).type.dtype == dtype

    # "floatX" type
    assert aesara_code_(x, dtypes={x: 'floatX'}).type.dtype in ('float32', 'float64')

    # Type promotion
    assert aesara_code_(x + 1, dtypes={x: 'float32'}).type.dtype == 'float32'
    assert aesara_code_(x + y, dtypes={x: 'float64', y: 'float32'}).type.dtype == 'float64'


def test_broadcastables():
    """ Test the "broadcastables" argument when printing symbol-like objects. """

    # No restrictions on shape
    for s in [x, f_t]:
        for bc in [(), (False,), (True,), (False, False), (True, False)]:
            assert aesara_code_(s, broadcastables={s: bc}).broadcastable == bc

    # TODO - matrix broadcasting?

def test_broadcasting():
    """ Test "broadcastable" attribute after applying element-wise binary op. """

    expr = x + y

    cases = [
        [(), (), ()],
        [(False,), (False,), (False,)],
        [(True,), (False,), (False,)],
        [(False, True), (False, False), (False, False)],
        [(True, False), (False, False), (False, False)],
    ]

    for bc1, bc2, bc3 in cases:
        comp = aesara_code_(expr, broadcastables={x: bc1, y: bc2})
        assert comp.broadcastable == bc3


def test_MatMul():
    expr = X*Y*Z
    expr_t = aesara_code_(expr)
    assert isinstance(expr_t.owner.op, Dot)
    assert theq(expr_t, Xt.dot(Yt).dot(Zt))

def test_Transpose():
    assert isinstance(aesara_code_(X.T).owner.op, DimShuffle)

def test_MatAdd():
    expr = X+Y+Z
    assert isinstance(aesara_code_(expr).owner.op, Elemwise)


def test_Rationals():
    assert theq(aesara_code_(sy.Integer(2) / 3), true_divide(2, 3))
    assert theq(aesara_code_(S.Half), true_divide(1, 2))

def test_Integers():
    assert aesara_code_(sy.Integer(3)) == 3

def test_factorial():
    n = sy.Symbol('n')
    assert aesara_code_(sy.factorial(n))

def test_Derivative():
    with ignore_warnings(UserWarning):
        simp = lambda expr: aesara_simplify(fgraph_of(expr))
        assert theq(simp(aesara_code_(sy.Derivative(sy.sin(x), x, evaluate=False))),
                    simp(aesara.grad(aet.sin(xt), xt)))


def test_aesara_function_simple():
    """ Test aesara_function() with single output. """
    f = aesara_function_([x, y], [x+y])
    assert f(2, 3) == 5

def test_aesara_function_multi():
    """ Test aesara_function() with multiple outputs. """
    f = aesara_function_([x, y], [x+y, x-y])
    o1, o2 = f(2, 3)
    assert o1 == 5
    assert o2 == -1

def test_aesara_function_numpy():
    """ Test aesara_function() vs Numpy implementation. """
    f = aesara_function_([x, y], [x+y], dim=1,
                         dtypes={x: 'float64', y: 'float64'})
    assert np.linalg.norm(f([1, 2], [3, 4]) - np.asarray([4, 6])) < 1e-9

    f = aesara_function_([x, y], [x+y], dtypes={x: 'float64', y: 'float64'},
                         dim=1)
    xx = np.arange(3).astype('float64')
    yy = 2*np.arange(3).astype('float64')
    assert np.linalg.norm(f(xx, yy) - 3*np.arange(3)) < 1e-9


def test_aesara_function_matrix():
    m = sy.Matrix([[x, y], [z, x + y + z]])
    expected = np.array([[1.0, 2.0], [3.0, 1.0 + 2.0 + 3.0]])
    f = aesara_function_([x, y, z], [m])
    np.testing.assert_allclose(f(1.0, 2.0, 3.0), expected)
    f = aesara_function_([x, y, z], [m], scalar=True)
    np.testing.assert_allclose(f(1.0, 2.0, 3.0), expected)
    f = aesara_function_([x, y, z], [m, m])
    assert isinstance(f(1.0, 2.0, 3.0), type([]))
    np.testing.assert_allclose(f(1.0, 2.0, 3.0)[0], expected)
    np.testing.assert_allclose(f(1.0, 2.0, 3.0)[1], expected)

def test_dim_handling():
    assert dim_handling([x], dim=2) == {x: (False, False)}
    assert dim_handling([x, y], dims={x: 1, y: 2}) == {x: (False, True),
                                                       y: (False, False)}
    assert dim_handling([x], broadcastables={x: (False,)}) == {x: (False,)}

def test_aesara_function_kwargs():
    """
    Test passing additional kwargs from aesara_function() to aesara.function().
    """
    import numpy as np
    f = aesara_function_([x, y, z], [x+y], dim=1, on_unused_input='ignore',
            dtypes={x: 'float64', y: 'float64', z: 'float64'})
    assert np.linalg.norm(f([1, 2], [3, 4], [0, 0]) - np.asarray([4, 6])) < 1e-9

    f = aesara_function_([x, y, z], [x+y],
                        dtypes={x: 'float64', y: 'float64', z: 'float64'},
                        dim=1, on_unused_input='ignore')
    xx = np.arange(3).astype('float64')
    yy = 2*np.arange(3).astype('float64')
    zz = 2*np.arange(3).astype('float64')
    assert np.linalg.norm(f(xx, yy, zz) - 3*np.arange(3)) < 1e-9

def test_aesara_function_scalar():
    """ Test the "scalar" argument to aesara_function(). """
    from aesara.compile.function.types import Function

    args = [
        ([x, y], [x + y], None, [0]),  # Single 0d output
        ([X, Y], [X + Y], None, [2]),  # Single 2d output
        ([x, y], [x + y], {x: 0, y: 1}, [1]),  # Single 1d output
        ([x, y], [x + y, x - y], None, [0, 0]),  # Two 0d outputs
        ([x, y, X, Y], [x + y, X + Y], None, [0, 2]),  # One 0d output, one 2d
    ]

    # Create and test functions with and without the scalar setting
    for inputs, outputs, in_dims, out_dims in args:
        for scalar in [False, True]:

            f = aesara_function_(inputs, outputs, dims=in_dims, scalar=scalar)

            # Check the aesara_function attribute is set whether wrapped or not
            assert isinstance(f.aesara_function, Function)

            # Feed in inputs of the appropriate size and get outputs
            in_values = [
                np.ones([1 if bc else 5 for bc in i.type.broadcastable])
                for i in f.aesara_function.input_storage
            ]
            out_values = f(*in_values)
            if not isinstance(out_values, list):
                out_values = [out_values]

            # Check output types and shapes
            assert len(out_dims) == len(out_values)
            for d, value in zip(out_dims, out_values):

                if scalar and d == 0:
                    # Should have been converted to a scalar value
                    assert isinstance(value, np.number)

                else:
                    # Otherwise should be an array
                    assert isinstance(value, np.ndarray)
                    assert value.ndim == d

def test_aesara_function_bad_kwarg():
    """
    Passing an unknown keyword argument to aesara_function() should raise an
    exception.
    """
    raises(Exception, lambda : aesara_function_([x], [x+1], foobar=3))


def test_slice():
    assert aesara_code_(slice(1, 2, 3)) == slice(1, 2, 3)

    def theq_slice(s1, s2):
        for attr in ['start', 'stop', 'step']:
            a1 = getattr(s1, attr)
            a2 = getattr(s2, attr)
            if a1 is None or a2 is None:
                if not (a1 is None or a2 is None):
                    return False
            elif not theq(a1, a2):
                return False
        return True

    dtypes = {x: 'int32', y: 'int32'}
    assert theq_slice(aesara_code_(slice(x, y), dtypes=dtypes), slice(xt, yt))
    assert theq_slice(aesara_code_(slice(1, x, 3), dtypes=dtypes), slice(1, xt, 3))

def test_MatrixSlice():
    cache = {}

    n = sy.Symbol('n', integer=True)
    X = sy.MatrixSymbol('X', n, n)

    Y = X[1:2:3, 4:5:6]
    Yt = aesara_code_(Y, cache=cache)

    s = ScalarType('int64')
    assert tuple(Yt.owner.op.idx_list) == (slice(s, s, s), slice(s, s, s))
    assert Yt.owner.inputs[0] == aesara_code_(X, cache=cache)
    # == doesn't work in Aesara like it does in SymPy. You have to use
    # equals.
    assert all(Yt.owner.inputs[i].data == i for i in range(1, 7))

    k = sy.Symbol('k')
    aesara_code_(k, dtypes={k: 'int32'})
    start, stop, step = 4, k, 2
    Y = X[start:stop:step]
    Yt = aesara_code_(Y, dtypes={n: 'int32', k: 'int32'})
    # assert Yt.owner.op.idx_list[0].stop == kt

def test_BlockMatrix():
    n = sy.Symbol('n', integer=True)
    A, B, C, D = [sy.MatrixSymbol(name, n, n) for name in 'ABCD']
    At, Bt, Ct, Dt = map(aesara_code_, (A, B, C, D))
    Block = sy.BlockMatrix([[A, B], [C, D]])
    Blockt = aesara_code_(Block)
    solutions = [aet.join(0, aet.join(1, At, Bt), aet.join(1, Ct, Dt)),
                 aet.join(1, aet.join(0, At, Ct), aet.join(0, Bt, Dt))]
    assert any(theq(Blockt, solution) for solution in solutions)

@SKIP
def test_BlockMatrix_Inverse_execution():
    k, n = 2, 4
    dtype = 'float32'
    A = sy.MatrixSymbol('A', n, k)
    B = sy.MatrixSymbol('B', n, n)
    inputs = A, B
    output = B.I*A

    cutsizes = {A: [(n//2, n//2), (k//2, k//2)],
                B: [(n//2, n//2), (n//2, n//2)]}
    cutinputs = [sy.blockcut(i, *cutsizes[i]) for i in inputs]
    cutoutput = output.subs(dict(zip(inputs, cutinputs)))

    dtypes = dict(zip(inputs, [dtype]*len(inputs)))
    f = aesara_function_(inputs, [output], dtypes=dtypes, cache={})
    fblocked = aesara_function_(inputs, [sy.block_collapse(cutoutput)],
                                dtypes=dtypes, cache={})

    ninputs = [np.random.rand(*x.shape).astype(dtype) for x in inputs]
    ninputs = [np.arange(n*k).reshape(A.shape).astype(dtype),
               np.eye(n).astype(dtype)]
    ninputs[1] += np.ones(B.shape)*1e-5

    assert np.allclose(f(*ninputs), fblocked(*ninputs), rtol=1e-5)

def test_DenseMatrix():
    from aesara.tensor.basic import Join

    t = sy.Symbol('theta')
    for MatrixType in [sy.Matrix, sy.ImmutableMatrix]:
        X = MatrixType([[sy.cos(t), -sy.sin(t)], [sy.sin(t), sy.cos(t)]])
        tX = aesara_code_(X)
        assert isinstance(tX, TensorVariable)
        assert isinstance(tX.owner.op, Join)


def test_cache_basic():
    """ Test single symbol-like objects are cached when printed by themselves. """

    # Pairs of objects which should be considered equivalent with respect to caching
    pairs = [
        (x, sy.Symbol('x')),
        (X, sy.MatrixSymbol('X', *X.shape)),
        (f_t, sy.Function('f')(sy.Symbol('t'))),
    ]

    for s1, s2 in pairs:
        cache = {}
        st = aesara_code_(s1, cache=cache)

        # Test hit with same instance
        assert aesara_code_(s1, cache=cache) is st

        # Test miss with same instance but new cache
        assert aesara_code_(s1, cache={}) is not st

        # Test hit with different but equivalent instance
        assert aesara_code_(s2, cache=cache) is st

def test_global_cache():
    """ Test use of the global cache. """
    from sympy.printing.aesaracode import global_cache

    backup = dict(global_cache)
    try:
        # Temporarily empty global cache
        global_cache.clear()

        for s in [x, X, f_t]:
            with warns_deprecated_sympy():
                st = aesara_code(s)
                assert aesara_code(s) is st

    finally:
        # Restore global cache
        global_cache.update(backup)

def test_cache_types_distinct():
    """
    Test that symbol-like objects of different types (Symbol, MatrixSymbol,
    AppliedUndef) are distinguished by the cache even if they have the same
    name.
    """
    symbols = [sy.Symbol('f_t'), sy.MatrixSymbol('f_t', 4, 4), f_t]

    cache = {}  # Single shared cache
    printed = {}

    for s in symbols:
        st = aesara_code_(s, cache=cache)
        assert st not in printed.values()
        printed[s] = st

    # Check all printed objects are distinct
    assert len(set(map(id, printed.values()))) == len(symbols)

    # Check retrieving
    for s, st in printed.items():
        with warns_deprecated_sympy():
            assert aesara_code(s, cache=cache) is st

def test_symbols_are_created_once():
    """
    Test that a symbol is cached and reused when it appears in an expression
    more than once.
    """
    expr = sy.Add(x, x, evaluate=False)
    comp = aesara_code_(expr)

    assert theq(comp, xt + xt)
    assert not theq(comp, xt + aesara_code_(x))

def test_cache_complex():
    """
    Test caching on a complicated expression with multiple symbols appearing
    multiple times.
    """
    expr = x ** 2 + (y - sy.exp(x)) * sy.sin(z - x * y)
    symbol_names = {s.name for s in expr.free_symbols}
    expr_t = aesara_code_(expr)

    # Iterate through variables in the Aesara computational graph that the
    # printed expression depends on
    seen = set()
    for v in aesara.graph.basic.ancestors([expr_t]):
        # Owner-less, non-constant variables should be our symbols
        if v.owner is None and not isinstance(v, aesara.graph.basic.Constant):
            # Check it corresponds to a symbol and appears only once
            assert v.name in symbol_names
            assert v.name not in seen
            seen.add(v.name)

    # Check all were present
    assert seen == symbol_names


def test_Piecewise():
    # A piecewise linear
    expr = sy.Piecewise((0, x<0), (x, x<2), (1, True))  # ___/III
    result = aesara_code_(expr)
    assert result.owner.op == aet.switch

    expected = aet.switch(xt<0, 0, aet.switch(xt<2, xt, 1))
    assert theq(result, expected)

    expr = sy.Piecewise((x, x < 0))
    result = aesara_code_(expr)
    expected = aet.switch(xt < 0, xt, np.nan)
    assert theq(result, expected)

    expr = sy.Piecewise((0, sy.And(x>0, x<2)), \
        (x, sy.Or(x>2, x<0)))
    result = aesara_code_(expr)
    expected = aet.switch(aet.and_(xt>0,xt<2), 0, \
        aet.switch(aet.or_(xt>2, xt<0), xt, np.nan))
    assert theq(result, expected)


def test_Relationals():
    assert theq(aesara_code_(sy.Eq(x, y)), aet.eq(xt, yt))
    # assert theq(aesara_code_(sy.Ne(x, y)), aet.neq(xt, yt))  # TODO - implement
    assert theq(aesara_code_(x > y), xt > yt)
    assert theq(aesara_code_(x < y), xt < yt)
    assert theq(aesara_code_(x >= y), xt >= yt)
    assert theq(aesara_code_(x <= y), xt <= yt)


def test_complexfunctions():
    dtypes = {x:'complex128', y:'complex128'}
    with warns_deprecated_sympy():
        xt, yt = aesara_code(x, dtypes=dtypes), aesara_code(y, dtypes=dtypes)
    from sympy.functions.elementary.complexes import conjugate
    from aesara.tensor import as_tensor_variable as atv
    from aesara.tensor import complex as cplx
    with warns_deprecated_sympy():
        assert theq(aesara_code(y*conjugate(x), dtypes=dtypes), yt*(xt.conj()))
        assert theq(aesara_code((1+2j)*x), xt*(atv(1.0)+atv(2.0)*cplx(0,1)))


def test_constantfunctions():
    with warns_deprecated_sympy():
        tf = aesara_function([],[1+1j])
    assert(tf()==1+1j)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\test_intpoly.py ===
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt

from sympy.core import S, Rational

from sympy.integrals.intpoly import (decompose, best_origin, distance_to_side,
                                     polytope_integrate, point_sort,
                                     hyperplane_parameters, main_integrate3d,
                                     main_integrate, polygon_integrate,
                                     lineseg_integrate, integration_reduction,
                                     integration_reduction_dynamic, is_vertex)

from sympy.geometry.line import Segment2D
from sympy.geometry.polygon import Polygon
from sympy.geometry.point import Point, Point2D
from sympy.abc import x, y, z

from sympy.testing.pytest import slow


def test_decompose():
    assert decompose(x) == {1: x}
    assert decompose(x**2) == {2: x**2}
    assert decompose(x*y) == {2: x*y}
    assert decompose(x + y) == {1: x + y}
    assert decompose(x**2 + y) == {1: y, 2: x**2}
    assert decompose(8*x**2 + 4*y + 7) == {0: 7, 1: 4*y, 2: 8*x**2}
    assert decompose(x**2 + 3*y*x) == {2: x**2 + 3*x*y}
    assert decompose(9*x**2 + y + 4*x + x**3 + y**2*x + 3) ==\
        {0: 3, 1: 4*x + y, 2: 9*x**2, 3: x**3 + x*y**2}

    assert decompose(x, True) == {x}
    assert decompose(x ** 2, True) == {x**2}
    assert decompose(x * y, True) == {x * y}
    assert decompose(x + y, True) == {x, y}
    assert decompose(x ** 2 + y, True) == {y, x ** 2}
    assert decompose(8 * x ** 2 + 4 * y + 7, True) == {7, 4*y, 8*x**2}
    assert decompose(x ** 2 + 3 * y * x, True) == {x ** 2, 3 * x * y}
    assert decompose(9 * x ** 2 + y + 4 * x + x ** 3 + y ** 2 * x + 3, True) == \
           {3, y, 4*x, 9*x**2, x*y**2, x**3}


def test_best_origin():
    expr1 = y ** 2 * x ** 5 + y ** 5 * x ** 7 + 7 * x + x ** 12 + y ** 7 * x

    l1 = Segment2D(Point(0, 3), Point(1, 1))
    l2 = Segment2D(Point(S(3) / 2, 0), Point(S(3) / 2, 3))
    l3 = Segment2D(Point(0, S(3) / 2), Point(3, S(3) / 2))
    l4 = Segment2D(Point(0, 2), Point(2, 0))
    l5 = Segment2D(Point(0, 2), Point(1, 1))
    l6 = Segment2D(Point(2, 0), Point(1, 1))

    assert best_origin((2, 1), 3, l1, expr1) == (0, 3)
    # XXX: Should these return exact Rational output? Maybe best_origin should
    #      sympify its arguments...
    assert best_origin((2, 0), 3, l2, x ** 7) == (1.5, 0)
    assert best_origin((0, 2), 3, l3, x ** 7) == (0, 1.5)
    assert best_origin((1, 1), 2, l4, x ** 7 * y ** 3) == (0, 2)
    assert best_origin((1, 1), 2, l4, x ** 3 * y ** 7) == (2, 0)
    assert best_origin((1, 1), 2, l5, x ** 2 * y ** 9) == (0, 2)
    assert best_origin((1, 1), 2, l6, x ** 9 * y ** 2) == (2, 0)


@slow
def test_polytope_integrate():
    #  Convex 2-Polytopes
    #  Vertex representation
    assert polytope_integrate(Polygon(Point(0, 0), Point(0, 2),
                                      Point(4, 0)), 1) == 4
    assert polytope_integrate(Polygon(Point(0, 0), Point(0, 1),
                                      Point(1, 1), Point(1, 0)), x * y) ==\
                                      Rational(1, 4)
    assert polytope_integrate(Polygon(Point(0, 3), Point(5, 3), Point(1, 1)),
                              6*x**2 - 40*y) == Rational(-935, 3)

    assert polytope_integrate(Polygon(Point(0, 0), Point(0, sqrt(3)),
                                      Point(sqrt(3), sqrt(3)),
                                      Point(sqrt(3), 0)), 1) == 3

    hexagon = Polygon(Point(0, 0), Point(-sqrt(3) / 2, S.Half),
                      Point(-sqrt(3) / 2, S(3) / 2), Point(0, 2),
                      Point(sqrt(3) / 2, S(3) / 2), Point(sqrt(3) / 2, S.Half))

    assert polytope_integrate(hexagon, 1) == S(3*sqrt(3)) / 2

    #  Hyperplane representation
    assert polytope_integrate([((-1, 0), 0), ((1, 2), 4),
                               ((0, -1), 0)], 1) == 4
    assert polytope_integrate([((-1, 0), 0), ((0, 1), 1),
                               ((1, 0), 1), ((0, -1), 0)], x * y) == Rational(1, 4)
    assert polytope_integrate([((0, 1), 3), ((1, -2), -1),
                               ((-2, -1), -3)], 6*x**2 - 40*y) == Rational(-935, 3)
    assert polytope_integrate([((-1, 0), 0), ((0, sqrt(3)), 3),
                               ((sqrt(3), 0), 3), ((0, -1), 0)], 1) == 3

    hexagon = [((Rational(-1, 2), -sqrt(3) / 2), 0),
               ((-1, 0), sqrt(3) / 2),
               ((Rational(-1, 2), sqrt(3) / 2), sqrt(3)),
               ((S.Half, sqrt(3) / 2), sqrt(3)),
               ((1, 0), sqrt(3) / 2),
               ((S.Half, -sqrt(3) / 2), 0)]
    assert polytope_integrate(hexagon, 1) == S(3*sqrt(3)) / 2

    #  Non-convex polytopes
    #  Vertex representation
    assert polytope_integrate(Polygon(Point(-1, -1), Point(-1, 1),
                                      Point(1, 1), Point(0, 0),
                                      Point(1, -1)), 1) == 3
    assert polytope_integrate(Polygon(Point(-1, -1), Point(-1, 1),
                                      Point(0, 0), Point(1, 1),
                                      Point(1, -1), Point(0, 0)), 1) == 2
    #  Hyperplane representation
    assert polytope_integrate([((-1, 0), 1), ((0, 1), 1), ((1, -1), 0),
                               ((1, 1), 0), ((0, -1), 1)], 1) == 3
    assert polytope_integrate([((-1, 0), 1), ((1, 1), 0), ((-1, 1), 0),
                               ((1, 0), 1), ((-1, -1), 0),
                               ((1, -1), 0)], 1) == 2

    #  Tests for 2D polytopes mentioned in Chin et al(Page 10):
    #  http://dilbert.engr.ucdavis.edu/~suku/quadrature/cls-integration.pdf
    fig1 = Polygon(Point(1.220, -0.827), Point(-1.490, -4.503),
                   Point(-3.766, -1.622), Point(-4.240, -0.091),
                   Point(-3.160, 4), Point(-0.981, 4.447),
                   Point(0.132, 4.027))
    assert polytope_integrate(fig1, x**2 + x*y + y**2) ==\
        S(2031627344735367)/(8*10**12)

    fig2 = Polygon(Point(4.561, 2.317), Point(1.491, -1.315),
                   Point(-3.310, -3.164), Point(-4.845, -3.110),
                   Point(-4.569, 1.867))
    assert polytope_integrate(fig2, x**2 + x*y + y**2) ==\
        S(517091313866043)/(16*10**11)

    fig3 = Polygon(Point(-2.740, -1.888), Point(-3.292, 4.233),
                   Point(-2.723, -0.697), Point(-0.643, -3.151))
    assert polytope_integrate(fig3, x**2 + x*y + y**2) ==\
        S(147449361647041)/(8*10**12)

    fig4 = Polygon(Point(0.211, -4.622), Point(-2.684, 3.851),
                   Point(0.468, 4.879), Point(4.630, -1.325),
                   Point(-0.411, -1.044))
    assert polytope_integrate(fig4, x**2 + x*y + y**2) ==\
        S(180742845225803)/(10**12)

    #  Tests for many polynomials with maximum degree given(2D case).
    tri = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    polys = []
    expr1 = x**9*y + x**7*y**3 + 2*x**2*y**8
    expr2 = x**6*y**4 + x**5*y**5 + 2*y**10
    expr3 = x**10 + x**9*y + x**8*y**2 + x**5*y**5
    polys.extend((expr1, expr2, expr3))
    result_dict = polytope_integrate(tri, polys, max_degree=10)
    assert result_dict[expr1] == Rational(615780107, 594)
    assert result_dict[expr2] == Rational(13062161, 27)
    assert result_dict[expr3] == Rational(1946257153, 924)

    tri = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    expr1 = x**7*y**1 + 2*x**2*y**6
    expr2 = x**6*y**4 + x**5*y**5 + 2*y**10
    expr3 = x**10 + x**9*y + x**8*y**2 + x**5*y**5
    polys.extend((expr1, expr2, expr3))
    assert polytope_integrate(tri, polys, max_degree=9) == \
        {x**7*y + 2*x**2*y**6: Rational(489262, 9)}

    #  Tests when all integral of all monomials up to a max_degree is to be
    #  calculated.
    assert polytope_integrate(Polygon(Point(0, 0), Point(0, 1),
                                      Point(1, 1), Point(1, 0)),
                              max_degree=4) == {0: 0, 1: 1, x: S.Half,
                                                x ** 2 * y ** 2: S.One / 9,
                                                x ** 4: S.One / 5,
                                                y ** 4: S.One / 5,
                                                y: S.Half,
                                                x * y ** 2: S.One / 6,
                                                y ** 2: S.One / 3,
                                                x ** 3: S.One / 4,
                                                x ** 2 * y: S.One / 6,
                                                x ** 3 * y: S.One / 8,
                                                x * y: S.One / 4,
                                                y ** 3: S.One / 4,
                                                x ** 2: S.One / 3,
                                                x * y ** 3: S.One / 8}

    #  Tests for 3D polytopes
    cube1 = [[(0, 0, 0), (0, 6, 6), (6, 6, 6), (3, 6, 0),
              (0, 6, 0), (6, 0, 6), (3, 0, 0), (0, 0, 6)],
             [1, 2, 3, 4], [3, 2, 5, 6], [1, 7, 5, 2], [0, 6, 5, 7],
             [1, 4, 0, 7], [0, 4, 3, 6]]
    assert polytope_integrate(cube1, 1) == S(162)

    #  3D Test cases in Chin et al(2015)
    cube2 = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),
             (5, 0, 5), (5, 5, 0), (5, 5, 5)],
             [3, 7, 6, 2], [1, 5, 7, 3], [5, 4, 6, 7], [0, 4, 5, 1],
             [2, 0, 1, 3], [2, 6, 4, 0]]

    cube3 = [[(0, 0, 0), (5, 0, 0), (5, 4, 0), (3, 2, 0), (3, 5, 0),
              (0, 5, 0), (0, 0, 5), (5, 0, 5), (5, 4, 5), (3, 2, 5),
              (3, 5, 5), (0, 5, 5)],
             [6, 11, 5, 0], [1, 7, 6, 0], [5, 4, 3, 2, 1, 0], [11, 10, 4, 5],
             [10, 9, 3, 4], [9, 8, 2, 3], [8, 7, 1, 2], [7, 8, 9, 10, 11, 6]]

    cube4 = [[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
              (S.One / 4, S.One / 4, S.One / 4)],
             [0, 2, 1], [1, 3, 0], [4, 2, 3], [4, 3, 1],
             [0, 1, 2], [2, 4, 1], [0, 3, 2]]

    assert polytope_integrate(cube2, x ** 2 + y ** 2 + x * y + z ** 2) ==\
           Rational(15625, 4)
    assert polytope_integrate(cube3, x ** 2 + y ** 2 + x * y + z ** 2) ==\
           S(33835) / 12
    assert polytope_integrate(cube4, x ** 2 + y ** 2 + x * y + z ** 2) ==\
           S(37) / 960

    #  Test cases from Mathematica's PolyhedronData library
    octahedron = [[(S.NegativeOne / sqrt(2), 0, 0), (0, S.One / sqrt(2), 0),
                   (0, 0, S.NegativeOne / sqrt(2)), (0, 0, S.One / sqrt(2)),
                   (0, S.NegativeOne / sqrt(2), 0), (S.One / sqrt(2), 0, 0)],
                  [3, 4, 5], [3, 5, 1], [3, 1, 0], [3, 0, 4], [4, 0, 2],
                  [4, 2, 5], [2, 0, 1], [5, 2, 1]]

    assert polytope_integrate(octahedron, 1) == sqrt(2) / 3

    great_stellated_dodecahedron =\
        [[(-0.32491969623290634095, 0, 0.42532540417601993887),
          (0.32491969623290634095, 0, -0.42532540417601993887),
          (-0.52573111211913359231, 0, 0.10040570794311363956),
          (0.52573111211913359231, 0, -0.10040570794311363956),
          (-0.10040570794311363956, -0.3090169943749474241, 0.42532540417601993887),
          (-0.10040570794311363956, 0.30901699437494742410, 0.42532540417601993887),
          (0.10040570794311363956, -0.3090169943749474241, -0.42532540417601993887),
          (0.10040570794311363956, 0.30901699437494742410, -0.42532540417601993887),
          (-0.16245984811645317047, -0.5, 0.10040570794311363956),
          (-0.16245984811645317047,  0.5, 0.10040570794311363956),
          (0.16245984811645317047,  -0.5, -0.10040570794311363956),
          (0.16245984811645317047,   0.5, -0.10040570794311363956),
          (-0.42532540417601993887, -0.3090169943749474241, -0.10040570794311363956),
          (-0.42532540417601993887, 0.30901699437494742410, -0.10040570794311363956),
          (-0.26286555605956679615, 0.1909830056250525759, -0.42532540417601993887),
          (-0.26286555605956679615, -0.1909830056250525759, -0.42532540417601993887),
          (0.26286555605956679615, 0.1909830056250525759, 0.42532540417601993887),
          (0.26286555605956679615, -0.1909830056250525759, 0.42532540417601993887),
          (0.42532540417601993887, -0.3090169943749474241, 0.10040570794311363956),
          (0.42532540417601993887, 0.30901699437494742410, 0.10040570794311363956)],
         [12, 3, 0, 6, 16], [17, 7, 0, 3, 13],
         [9, 6, 0, 7, 8], [18, 2, 1, 4, 14],
         [15, 5, 1, 2, 19], [11, 4, 1, 5, 10],
         [8, 19, 2, 18, 9], [10, 13, 3, 12, 11],
         [16, 14, 4, 11, 12], [13, 10, 5, 15, 17],
         [14, 16, 6, 9, 18], [19, 8, 7, 17, 15]]
    #  Actual volume is : 0.163118960624632
    assert Abs(polytope_integrate(great_stellated_dodecahedron, 1) -\
        0.163118960624632) < 1e-12

    expr = x **2 + y ** 2 + z ** 2
    octahedron_five_compound = [[(0, -0.7071067811865475244, 0),
                                 (0, 0.70710678118654752440, 0),
                                 (0.1148764602736805918,
                                  -0.35355339059327376220, -0.60150095500754567366),
                                 (0.1148764602736805918, 0.35355339059327376220,
                                     -0.60150095500754567366),
                                 (0.18587401723009224507,
                                  -0.57206140281768429760, 0.37174803446018449013),
                                 (0.18587401723009224507,  0.57206140281768429760,
                                  0.37174803446018449013),
                                 (0.30075047750377283683, -0.21850801222441053540,
                                  0.60150095500754567366),
                                 (0.30075047750377283683, 0.21850801222441053540,
                                  0.60150095500754567366),
                                 (0.48662449473386508189, -0.35355339059327376220,
                                  -0.37174803446018449013),
                                 (0.48662449473386508189, 0.35355339059327376220,
                                  -0.37174803446018449013),
                                 (-0.60150095500754567366, 0, -0.37174803446018449013),
                                 (-0.30075047750377283683, -0.21850801222441053540,
                                  -0.60150095500754567366),
                                 (-0.30075047750377283683, 0.21850801222441053540,
                                  -0.60150095500754567366),
                                 (0.60150095500754567366, 0, 0.37174803446018449013),
                                 (0.4156269377774534286, -0.57206140281768429760, 0),
                                 (0.4156269377774534286, 0.57206140281768429760, 0),
                                 (0.37174803446018449013, 0, -0.60150095500754567366),
                                 (-0.4156269377774534286, -0.57206140281768429760, 0),
                                 (-0.4156269377774534286, 0.57206140281768429760, 0),
                                 (-0.67249851196395732696, -0.21850801222441053540, 0),
                                 (-0.67249851196395732696, 0.21850801222441053540, 0),
                                 (0.67249851196395732696, -0.21850801222441053540, 0),
                                 (0.67249851196395732696, 0.21850801222441053540, 0),
                                 (-0.37174803446018449013, 0, 0.60150095500754567366),
                                 (-0.48662449473386508189, -0.35355339059327376220,
                                 0.37174803446018449013),
                                 (-0.48662449473386508189, 0.35355339059327376220,
                                  0.37174803446018449013),
                                 (-0.18587401723009224507, -0.57206140281768429760,
                                  -0.37174803446018449013),
                                 (-0.18587401723009224507, 0.57206140281768429760,
                                  -0.37174803446018449013),
                                 (-0.11487646027368059176, -0.35355339059327376220,
                                  0.60150095500754567366),
                                 (-0.11487646027368059176, 0.35355339059327376220,
                                 0.60150095500754567366)],
                                 [0, 10, 16], [23, 10, 0], [16, 13, 0],
                                 [0, 13, 23], [16, 10, 1], [1, 10, 23],
                                 [1, 13, 16], [23, 13, 1], [2, 4, 19],
                                 [22, 4, 2], [2, 19, 27], [27, 22, 2],
                                 [20, 5, 3], [3, 5, 21], [26, 20, 3],
                                 [3, 21, 26], [29, 19, 4], [4, 22, 29],
                                 [5, 20, 28], [28, 21, 5], [6, 8, 15],
                                 [17, 8, 6], [6, 15, 25], [25, 17, 6],
                                 [14, 9, 7], [7, 9, 18], [24, 14, 7],
                                 [7, 18, 24], [8, 12, 15], [17, 12, 8],
                                 [14, 11, 9], [9, 11, 18], [11, 14, 24],
                                 [24, 18, 11], [25, 15, 12], [12, 17, 25],
                                 [29, 27, 19], [20, 26, 28], [28, 26, 21],
                                 [22, 27, 29]]
    assert Abs(polytope_integrate(octahedron_five_compound, expr)) - 0.353553\
        < 1e-6

    cube_five_compound = [[(-0.1624598481164531631, -0.5, -0.6881909602355867691),
                           (-0.1624598481164531631, 0.5, -0.6881909602355867691),
                           (0.1624598481164531631, -0.5, 0.68819096023558676910),
                           (0.1624598481164531631, 0.5, 0.68819096023558676910),
                          (-0.52573111211913359231, 0, -0.6881909602355867691),
                          (0.52573111211913359231, 0, 0.68819096023558676910),
                          (-0.26286555605956679615, -0.8090169943749474241,
                           -0.1624598481164531631),
                          (-0.26286555605956679615, 0.8090169943749474241,
                           -0.1624598481164531631),
                          (0.26286555605956680301, -0.8090169943749474241,
                           0.1624598481164531631),
                          (0.26286555605956680301, 0.8090169943749474241,
                           0.1624598481164531631),
                          (-0.42532540417601993887, -0.3090169943749474241,
                           0.68819096023558676910),
                          (-0.42532540417601993887, 0.30901699437494742410,
                           0.68819096023558676910),
                          (0.42532540417601996609, -0.3090169943749474241,
                           -0.6881909602355867691),
                          (0.42532540417601996609, 0.30901699437494742410,
                           -0.6881909602355867691),
                          (-0.6881909602355867691, -0.5, 0.1624598481164531631),
                          (-0.6881909602355867691, 0.5,  0.1624598481164531631),
                          (0.68819096023558676910, -0.5, -0.1624598481164531631),
                          (0.68819096023558676910, 0.5, -0.1624598481164531631),
                          (-0.85065080835203998877, 0, -0.1624598481164531631),
                          (0.85065080835203993218, 0, 0.1624598481164531631)],
                          [18, 10, 3, 7], [13, 19, 8, 0], [18, 0, 8, 10],
                          [3, 19, 13, 7], [18, 7, 13, 0], [8, 19, 3, 10],
                          [6, 2, 11, 18], [1, 9, 19, 12], [11, 9, 1, 18],
                          [6, 12, 19, 2], [1, 12, 6, 18], [11, 2, 19, 9],
                          [4, 14, 11, 7], [17, 5, 8, 12], [4, 12, 8, 14],
                          [11, 5, 17, 7], [4, 7, 17, 12], [8, 5, 11, 14],
                          [6, 10, 15, 4], [13, 9, 5, 16], [15, 9, 13, 4],
                          [6, 16, 5, 10], [13, 16, 6, 4], [15, 10, 5, 9],
                          [14, 15, 1, 0], [16, 17, 3, 2], [14, 2, 3, 15],
                          [1, 17, 16, 0], [14, 0, 16, 2], [3, 17, 1, 15]]
    assert Abs(polytope_integrate(cube_five_compound, expr) - 1.25) < 1e-12

    echidnahedron = [[(0, 0, -2.4898982848827801995),
                      (0, 0, 2.4898982848827802734),
                      (0, -4.2360679774997896964, -2.4898982848827801995),
                      (0, -4.2360679774997896964, 2.4898982848827802734),
                      (0, 4.2360679774997896964, -2.4898982848827801995),
                      (0, 4.2360679774997896964, 2.4898982848827802734),
                      (-4.0287400534704067567, -1.3090169943749474241, -2.4898982848827801995),
                      (-4.0287400534704067567, -1.3090169943749474241, 2.4898982848827802734),
                      (-4.0287400534704067567, 1.3090169943749474241, -2.4898982848827801995),
                      (-4.0287400534704067567, 1.3090169943749474241, 2.4898982848827802734),
                      (4.0287400534704069747, -1.3090169943749474241, -2.4898982848827801995),
                      (4.0287400534704069747, -1.3090169943749474241, 2.4898982848827802734),
                      (4.0287400534704069747, 1.3090169943749474241, -2.4898982848827801995),
                      (4.0287400534704069747, 1.3090169943749474241, 2.4898982848827802734),
                      (-2.4898982848827801995, -3.4270509831248422723, -2.4898982848827801995),
                      (-2.4898982848827801995, -3.4270509831248422723, 2.4898982848827802734),
                      (-2.4898982848827801995, 3.4270509831248422723, -2.4898982848827801995),
                      (-2.4898982848827801995, 3.4270509831248422723, 2.4898982848827802734),
                      (2.4898982848827802734, -3.4270509831248422723, -2.4898982848827801995),
                      (2.4898982848827802734, -3.4270509831248422723, 2.4898982848827802734),
                      (2.4898982848827802734, 3.4270509831248422723, -2.4898982848827801995),
                      (2.4898982848827802734, 3.4270509831248422723, 2.4898982848827802734),
                      (-4.7169310137059934362, -0.8090169943749474241, -1.1135163644116066184),
                      (-4.7169310137059934362, 0.8090169943749474241, -1.1135163644116066184),
                      (4.7169310137059937438, -0.8090169943749474241, 1.11351636441160673519),
                      (4.7169310137059937438, 0.8090169943749474241, 1.11351636441160673519),
                      (-4.2916056095299737777, -2.1180339887498948482, 1.11351636441160673519),
                      (-4.2916056095299737777, 2.1180339887498948482, 1.11351636441160673519),
                      (4.2916056095299737777, -2.1180339887498948482, -1.1135163644116066184),
                      (4.2916056095299737777, 2.1180339887498948482, -1.1135163644116066184),
                      (-3.6034146492943870399, 0, -3.3405490932348205213),
                      (3.6034146492943870399, 0, 3.3405490932348202056),
                      (-3.3405490932348205213, -3.4270509831248422723, 1.11351636441160673519),
                      (-3.3405490932348205213, 3.4270509831248422723, 1.11351636441160673519),
                      (3.3405490932348202056, -3.4270509831248422723, -1.1135163644116066184),
                      (3.3405490932348202056, 3.4270509831248422723, -1.1135163644116066184),
                      (-2.9152236890588002395, -2.1180339887498948482, 3.3405490932348202056),
                      (-2.9152236890588002395, 2.1180339887498948482, 3.3405490932348202056),
                      (2.9152236890588002395, -2.1180339887498948482, -3.3405490932348205213),
                      (2.9152236890588002395, 2.1180339887498948482, -3.3405490932348205213),
                      (-2.2270327288232132368, 0, -1.1135163644116066184),
                      (-2.2270327288232132368, -4.2360679774997896964, -1.1135163644116066184),
                      (-2.2270327288232132368, 4.2360679774997896964, -1.1135163644116066184),
                      (2.2270327288232134704, 0, 1.11351636441160673519),
                      (2.2270327288232134704, -4.2360679774997896964, 1.11351636441160673519),
                      (2.2270327288232134704, 4.2360679774997896964, 1.11351636441160673519),
                      (-1.8017073246471935200, -1.3090169943749474241, 1.11351636441160673519),
                      (-1.8017073246471935200, 1.3090169943749474241, 1.11351636441160673519),
                      (1.8017073246471935043, -1.3090169943749474241, -1.1135163644116066184),
                      (1.8017073246471935043, 1.3090169943749474241, -1.1135163644116066184),
                      (-1.3763819204711735382, 0, -4.7169310137059934362),
                      (-1.3763819204711735382, 0, 0.26286555605956679615),
                      (1.37638192047117353821, 0, 4.7169310137059937438),
                      (1.37638192047117353821, 0, -0.26286555605956679615),
                      (-1.1135163644116066184, -3.4270509831248422723, -3.3405490932348205213),
                      (-1.1135163644116066184, -0.8090169943749474241, 4.7169310137059937438),
                      (-1.1135163644116066184, -0.8090169943749474241, -0.26286555605956679615),
                      (-1.1135163644116066184, 0.8090169943749474241, 4.7169310137059937438),
                      (-1.1135163644116066184, 0.8090169943749474241, -0.26286555605956679615),
                      (-1.1135163644116066184, 3.4270509831248422723, -3.3405490932348205213),
                      (1.11351636441160673519, -3.4270509831248422723, 3.3405490932348202056),
                      (1.11351636441160673519, -0.8090169943749474241, -4.7169310137059934362),
                      (1.11351636441160673519, -0.8090169943749474241, 0.26286555605956679615),
                      (1.11351636441160673519, 0.8090169943749474241, -4.7169310137059934362),
                      (1.11351636441160673519, 0.8090169943749474241, 0.26286555605956679615),
                      (1.11351636441160673519, 3.4270509831248422723, 3.3405490932348202056),
                      (-0.85065080835203998877, 0, 1.11351636441160673519),
                      (0.85065080835203993218, 0, -1.1135163644116066184),
                      (-0.6881909602355867691, -0.5, -1.1135163644116066184),
                      (-0.6881909602355867691, 0.5, -1.1135163644116066184),
                      (-0.6881909602355867691, -4.7360679774997896964, -1.1135163644116066184),
                      (-0.6881909602355867691, -2.1180339887498948482, -1.1135163644116066184),
                      (-0.6881909602355867691, 2.1180339887498948482, -1.1135163644116066184),
                      (-0.6881909602355867691, 4.7360679774997896964, -1.1135163644116066184),
                      (0.68819096023558676910, -0.5, 1.11351636441160673519),
                      (0.68819096023558676910, 0.5, 1.11351636441160673519),
                      (0.68819096023558676910, -4.7360679774997896964, 1.11351636441160673519),
                      (0.68819096023558676910, -2.1180339887498948482, 1.11351636441160673519),
                      (0.68819096023558676910, 2.1180339887498948482, 1.11351636441160673519),
                      (0.68819096023558676910, 4.7360679774997896964, 1.11351636441160673519),
                      (-0.42532540417601993887, -1.3090169943749474241, -4.7169310137059934362),
                      (-0.42532540417601993887, -1.3090169943749474241, 0.26286555605956679615),
                      (-0.42532540417601993887, 1.3090169943749474241, -4.7169310137059934362),
                      (-0.42532540417601993887, 1.3090169943749474241, 0.26286555605956679615),
                      (-0.26286555605956679615, -0.8090169943749474241, 1.11351636441160673519),
                      (-0.26286555605956679615, 0.8090169943749474241, 1.11351636441160673519),
                      (0.26286555605956679615, -0.8090169943749474241, -1.1135163644116066184),
                      (0.26286555605956679615, 0.8090169943749474241, -1.1135163644116066184),
                      (0.42532540417601996609, -1.3090169943749474241, 4.7169310137059937438),
                      (0.42532540417601996609, -1.3090169943749474241, -0.26286555605956679615),
                      (0.42532540417601996609, 1.3090169943749474241, 4.7169310137059937438),
                      (0.42532540417601996609, 1.3090169943749474241, -0.26286555605956679615)],
                      [9, 66, 47], [44, 62, 77], [20, 91, 49], [33, 47, 83],
                      [3, 77, 84], [12, 49, 53], [36, 84, 66], [28, 53, 62],
                      [73, 83, 91], [15, 84, 46], [25, 64, 43], [16, 58, 72],
                      [26, 46, 51], [11, 43, 74], [4, 72, 91], [60, 74, 84],
                      [35, 91, 64], [23, 51, 58], [19, 74, 77], [79, 83, 78],
                      [6, 56, 40], [76, 77, 81], [21, 78, 75], [8, 40, 58],
                      [31, 75, 74], [42, 58, 83], [41, 81, 56], [13, 75, 43],
                      [27, 51, 47], [2, 89, 71], [24, 43, 62], [17, 47, 85],
                      [14, 71, 56], [65, 85, 75], [22, 56, 51], [34, 62, 89],
                      [5, 85, 78], [32, 81, 46], [10, 53, 48], [45, 78, 64],
                      [7, 46, 66], [18, 48, 89], [37, 66, 85], [70, 89, 81],
                      [29, 64, 53], [88, 74, 1], [38, 67, 48], [42, 83, 72],
                      [57, 1, 85], [34, 48, 62], [59, 72, 87], [19, 62, 74],
                      [63, 87, 67], [17, 85, 83], [52, 75, 1], [39, 87, 49],
                      [22, 51, 40], [55, 1, 66], [29, 49, 64], [30, 40, 69],
                      [13, 64, 75], [82, 69, 87], [7, 66, 51], [90, 85, 1],
                      [59, 69, 72], [70, 81, 71], [88, 1, 84], [73, 72, 83],
                      [54, 71, 68], [5, 83, 85], [50, 68, 69], [3, 84, 81],
                      [57, 66, 1], [30, 68, 40], [28, 62, 48], [52, 1, 74],
                      [23, 40, 51], [38, 48, 86], [9, 51, 66], [80, 86, 68],
                      [11, 74, 62], [55, 84, 1], [54, 86, 71], [35, 64, 49],
                      [90, 1, 75], [41, 71, 81], [39, 49, 67], [15, 81, 84],
                      [61, 67, 86], [21, 75, 64], [24, 53, 43], [50, 69, 0],
                      [37, 85, 47], [31, 43, 75], [61, 0, 67], [27, 47, 58],
                      [10, 67, 53], [8, 58, 69], [90, 75, 85], [45, 91, 78],
                      [80, 68, 0], [36, 66, 46], [65, 78, 85], [63, 0, 87],
                      [32, 46, 56], [20, 87, 91], [14, 56, 68], [57, 85, 66],
                      [33, 58, 47], [61, 86, 0], [60, 84, 77], [37, 47, 66],
                      [82, 0, 69], [44, 77, 89], [16, 69, 58], [18, 89, 86],
                      [55, 66, 84], [26, 56, 46], [63, 67, 0], [31, 74, 43],
                      [36, 46, 84], [50, 0, 68], [25, 43, 53], [6, 68, 56],
                      [12, 53, 67], [88, 84, 74], [76, 89, 77], [82, 87, 0],
                      [65, 75, 78], [60, 77, 74], [80, 0, 86], [79, 78, 91],
                      [2, 86, 89], [4, 91, 87], [52, 74, 75], [21, 64, 78],
                      [18, 86, 48], [23, 58, 40], [5, 78, 83], [28, 48, 53],
                      [6, 40, 68], [25, 53, 64], [54, 68, 86], [33, 83, 58],
                      [17, 83, 47], [12, 67, 49], [41, 56, 71], [9, 47, 51],
                      [35, 49, 91], [2, 71, 86], [79, 91, 83], [38, 86, 67],
                      [26, 51, 56], [7, 51, 46], [4, 87, 72], [34, 89, 48],
                      [15, 46, 81], [42, 72, 58], [10, 48, 67], [27, 58, 51],
                      [39, 67, 87], [76, 81, 89], [3, 81, 77], [8, 69, 40],
                      [29, 53, 49], [19, 77, 62], [22, 40, 56], [20, 49, 87],
                      [32, 56, 81], [59, 87, 69], [24, 62, 53], [11, 62, 43],
                      [14, 68, 71], [73, 91, 72], [13, 43, 64], [70, 71, 89],
                      [16, 72, 69], [44, 89, 62], [30, 69, 68], [45, 64, 91]]
    #  Actual volume is : 51.405764746872634
    assert Abs(polytope_integrate(echidnahedron, 1) - 51.4057647468726) < 1e-12
    assert Abs(polytope_integrate(echidnahedron, expr) - 253.569603474519) <\
    1e-12

    #  Tests for many polynomials with maximum degree given(2D case).
    assert polytope_integrate(cube2, [x**2, y*z], max_degree=2) == \
        {y * z: 3125 / S(4), x ** 2: 3125 / S(3)}

    assert polytope_integrate(cube2, max_degree=2) == \
        {1: 125, x: 625 / S(2), x * z: 3125 / S(4), y: 625 / S(2),
         y * z: 3125 / S(4), z ** 2: 3125 / S(3), y ** 2: 3125 / S(3),
         z: 625 / S(2), x * y: 3125 / S(4), x ** 2: 3125 / S(3)}

def test_point_sort():
    assert point_sort([Point(0, 0), Point(1, 0), Point(1, 1)]) == \
        [Point2D(1, 1), Point2D(1, 0), Point2D(0, 0)]

    fig6 = Polygon((0, 0), (1, 0), (1, 1))
    assert polytope_integrate(fig6, x*y) == Rational(-1, 8)
    assert polytope_integrate(fig6, x*y, clockwise = True) == Rational(1, 8)


def test_polytopes_intersecting_sides():
    fig5 = Polygon(Point(-4.165, -0.832), Point(-3.668, 1.568),
                   Point(-3.266, 1.279), Point(-1.090, -2.080),
                   Point(3.313, -0.683), Point(3.033, -4.845),
                   Point(-4.395, 4.840), Point(-1.007, -3.328))
    assert polytope_integrate(fig5, x**2 + x*y + y**2) ==\
        S(1633405224899363)/(24*10**12)

    fig6 = Polygon(Point(-3.018, -4.473), Point(-0.103, 2.378),
                   Point(-1.605, -2.308), Point(4.516, -0.771),
                   Point(4.203, 0.478))
    assert polytope_integrate(fig6, x**2 + x*y + y**2) ==\
        S(88161333955921)/(3*10**12)


def test_max_degree():
    polygon = Polygon((0, 0), (0, 1), (1, 1), (1, 0))
    polys = [1, x, y, x*y, x**2*y, x*y**2]
    assert polytope_integrate(polygon, polys, max_degree=3) == \
        {1: 1, x: S.Half, y: S.Half, x*y: Rational(1, 4), x**2*y: Rational(1, 6), x*y**2: Rational(1, 6)}
    assert polytope_integrate(polygon, polys, max_degree=2) == \
        {1: 1, x: S.Half, y: S.Half, x*y: Rational(1, 4)}
    assert polytope_integrate(polygon, polys, max_degree=1) == \
        {1: 1, x: S.Half, y: S.Half}


def test_main_integrate3d():
    cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
             (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
            [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
            [3, 1, 0, 2], [0, 4, 6, 2]]
    vertices = cube[0]
    faces = cube[1:]
    hp_params = hyperplane_parameters(faces, vertices)
    assert main_integrate3d(1, faces, vertices, hp_params) == -125
    assert main_integrate3d(1, faces, vertices, hp_params, max_degree=1) == \
        {1: -125, y: Rational(-625, 2), z: Rational(-625, 2), x: Rational(-625, 2)}


def test_main_integrate():
    triangle = Polygon((0, 3), (5, 3), (1, 1))
    facets = triangle.sides
    hp_params = hyperplane_parameters(triangle)
    assert main_integrate(x**2 + y**2, facets, hp_params) == Rational(325, 6)
    assert main_integrate(x**2 + y**2, facets, hp_params, max_degree=1) == \
        {0: 0, 1: 5, y: Rational(35, 3), x: 10}


def test_polygon_integrate():
    cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
             (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
            [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
            [3, 1, 0, 2], [0, 4, 6, 2]]
    facet = cube[1]
    facets = cube[1:]
    vertices = cube[0]
    assert polygon_integrate(facet, [(0, 1, 0), 5], 0, facets, vertices, 1, 0) == -25


def test_distance_to_side():
    point = (0, 0, 0)
    assert distance_to_side(point, [(0, 0, 1), (0, 1, 0)], (1, 0, 0)) == -sqrt(2)/2


def test_lineseg_integrate():
    polygon = [(0, 5, 0), (5, 5, 0), (5, 5, 5), (0, 5, 5)]
    line_seg = [(0, 5, 0), (5, 5, 0)]
    assert lineseg_integrate(polygon, 0, line_seg, 1, 0) == 5
    assert lineseg_integrate(polygon, 0, line_seg, 0, 0) == 0


def test_integration_reduction():
    triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    facets = triangle.sides
    a, b = hyperplane_parameters(triangle)[0]
    assert integration_reduction(facets, 0, a, b, 1, (x, y), 0) == 5
    assert integration_reduction(facets, 0, a, b, 0, (x, y), 0) == 0


def test_integration_reduction_dynamic():
    triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    facets = triangle.sides
    a, b = hyperplane_parameters(triangle)[0]
    x0 = facets[0].points[0]
    monomial_values = [[0, 0, 0, 0], [1, 0, 0, 5],\
                       [y, 0, 1, 15], [x, 1, 0, None]]

    assert integration_reduction_dynamic(facets, 0, a, b, x, 1, (x, y), 1,\
                                         0, 1, x0, monomial_values, 3) == Rational(25, 2)
    assert integration_reduction_dynamic(facets, 0, a, b, 0, 1, (x, y), 1,\
                                         0, 1, x0, monomial_values, 3) == 0


def test_is_vertex():
    assert is_vertex(2) is False
    assert is_vertex((2, 3)) is True
    assert is_vertex(Point(2, 3)) is True
    assert is_vertex((2, 3, 4)) is True
    assert is_vertex((2, 3, 4, 5)) is False


def test_issue_19234():
    polygon = Polygon(Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))
    polys =  [ 1, x, y, x*y, x**2*y, x*y**2]
    assert polytope_integrate(polygon, polys) == \
        {1: 1, x: S.Half, y: S.Half, x*y: Rational(1, 4), x**2*y: Rational(1, 6), x*y**2: Rational(1, 6)}
    polys =  [ 1, x, y, x*y, 3 + x**2*y, x + x*y**2]
    assert polytope_integrate(polygon, polys) == \
        {1: 1, x: S.Half, y: S.Half, x*y: Rational(1, 4), x**2*y + 3: Rational(19, 6), x*y**2 + x: Rational(2, 3)}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_umath_complex.py ===
import platform
import sys

# import the c-extension module directly since _arg is not exported via umath
import numpy._core._multiarray_umath as ncu
import pytest

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_array_max_ulp,
    assert_equal,
    assert_raises,
)

# TODO: branch cuts (use Pauli code)
# TODO: conj 'symmetry'
# TODO: FPU exceptions

# At least on Windows the results of many complex functions are not conforming
# to the C99 standard. See ticket 1574.
# Ditto for Solaris (ticket 1642) and OS X on PowerPC.
# FIXME: this will probably change when we require full C99 compatibility
with np.errstate(all='ignore'):
    functions_seem_flaky = ((np.exp(complex(np.inf, 0)).imag != 0)
                            or (np.log(complex(ncu.NZERO, 0)).imag != np.pi))
# TODO: replace with a check on whether platform-provided C99 funcs are used
xfail_complex_tests = (not sys.platform.startswith('linux') or functions_seem_flaky)

# TODO This can be xfail when the generator functions are got rid of.
platform_skip = pytest.mark.skipif(xfail_complex_tests,
                                   reason="Inadequate C99 complex support")


class TestCexp:
    def test_simple(self):
        check = check_complex_value
        f = np.exp

        check(f, 1, 0, np.exp(1), 0, False)
        check(f, 0, 1, np.cos(1), np.sin(1), False)

        ref = np.exp(1) * complex(np.cos(1), np.sin(1))
        check(f, 1, 1, ref.real, ref.imag, False)

    @platform_skip
    def test_special_values(self):
        # C99: Section G 6.3.1

        check = check_complex_value
        f = np.exp

        # cexp(+-0 + 0i) is 1 + 0i
        check(f, ncu.PZERO, 0, 1, 0, False)
        check(f, ncu.NZERO, 0, 1, 0, False)

        # cexp(x + infi) is nan + nani for finite x and raises 'invalid' FPU
        # exception
        check(f,  1, np.inf, np.nan, np.nan)
        check(f, -1, np.inf, np.nan, np.nan)
        check(f,  0, np.inf, np.nan, np.nan)

        # cexp(inf + 0i) is inf + 0i
        check(f,  np.inf, 0, np.inf, 0)

        # cexp(-inf + yi) is +0 * (cos(y) + i sin(y)) for finite y
        check(f, -np.inf, 1, ncu.PZERO, ncu.PZERO)
        check(f, -np.inf, 0.75 * np.pi, ncu.NZERO, ncu.PZERO)

        # cexp(inf + yi) is +inf * (cos(y) + i sin(y)) for finite y
        check(f,  np.inf, 1, np.inf, np.inf)
        check(f,  np.inf, 0.75 * np.pi, -np.inf, np.inf)

        # cexp(-inf + inf i) is +-0 +- 0i (signs unspecified)
        def _check_ninf_inf(dummy):
            msgform = "cexp(-inf, inf) is (%f, %f), expected (+-0, +-0)"
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(-np.inf, np.inf)))
                if z.real != 0 or z.imag != 0:
                    raise AssertionError(msgform % (z.real, z.imag))

        _check_ninf_inf(None)

        # cexp(inf + inf i) is +-inf + NaNi and raised invalid FPU ex.
        def _check_inf_inf(dummy):
            msgform = "cexp(inf, inf) is (%f, %f), expected (+-inf, nan)"
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(np.inf, np.inf)))
                if not np.isinf(z.real) or not np.isnan(z.imag):
                    raise AssertionError(msgform % (z.real, z.imag))

        _check_inf_inf(None)

        # cexp(-inf + nan i) is +-0 +- 0i
        def _check_ninf_nan(dummy):
            msgform = "cexp(-inf, nan) is (%f, %f), expected (+-0, +-0)"
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(-np.inf, np.nan)))
                if z.real != 0 or z.imag != 0:
                    raise AssertionError(msgform % (z.real, z.imag))

        _check_ninf_nan(None)

        # cexp(inf + nan i) is +-inf + nan
        def _check_inf_nan(dummy):
            msgform = "cexp(-inf, nan) is (%f, %f), expected (+-inf, nan)"
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(np.inf, np.nan)))
                if not np.isinf(z.real) or not np.isnan(z.imag):
                    raise AssertionError(msgform % (z.real, z.imag))

        _check_inf_nan(None)

        # cexp(nan + yi) is nan + nani for y != 0 (optional: raises invalid FPU
        # ex)
        check(f, np.nan, 1, np.nan, np.nan)
        check(f, np.nan, -1, np.nan, np.nan)

        check(f, np.nan,  np.inf, np.nan, np.nan)
        check(f, np.nan, -np.inf, np.nan, np.nan)

        # cexp(nan + nani) is nan + nani
        check(f, np.nan, np.nan, np.nan, np.nan)

    # TODO This can be xfail when the generator functions are got rid of.
    @pytest.mark.skip(reason="cexp(nan + 0I) is wrong on most platforms")
    def test_special_values2(self):
        # XXX: most implementations get it wrong here (including glibc <= 2.10)
        # cexp(nan + 0i) is nan + 0i
        check = check_complex_value
        f = np.exp

        check(f, np.nan, 0, np.nan, 0)

class TestClog:
    def test_simple(self):
        x = np.array([1 + 0j, 1 + 2j])
        y_r = np.log(np.abs(x)) + 1j * np.angle(x)
        y = np.log(x)
        assert_almost_equal(y, y_r)

    @platform_skip
    @pytest.mark.skipif(platform.machine() == "armv5tel", reason="See gh-413.")
    def test_special_values(self):
        xl = []
        yl = []

        # From C99 std (Sec 6.3.2)
        # XXX: check exceptions raised
        # --- raise for invalid fails.

        # clog(-0 + i0) returns -inf + i pi and raises the 'divide-by-zero'
        # floating-point exception.
        with np.errstate(divide='raise'):
            x = np.array([ncu.NZERO], dtype=complex)
            y = complex(-np.inf, np.pi)
            assert_raises(FloatingPointError, np.log, x)
        with np.errstate(divide='ignore'):
            assert_almost_equal(np.log(x), y)

        xl.append(x)
        yl.append(y)

        # clog(+0 + i0) returns -inf + i0 and raises the 'divide-by-zero'
        # floating-point exception.
        with np.errstate(divide='raise'):
            x = np.array([0], dtype=complex)
            y = complex(-np.inf, 0)
            assert_raises(FloatingPointError, np.log, x)
        with np.errstate(divide='ignore'):
            assert_almost_equal(np.log(x), y)

        xl.append(x)
        yl.append(y)

        # clog(x + i inf returns +inf + i pi /2, for finite x.
        x = np.array([complex(1, np.inf)], dtype=complex)
        y = complex(np.inf, 0.5 * np.pi)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        x = np.array([complex(-1, np.inf)], dtype=complex)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(x + iNaN) returns NaN + iNaN and optionally raises the
        # 'invalid' floating- point exception, for finite x.
        with np.errstate(invalid='raise'):
            x = np.array([complex(1., np.nan)], dtype=complex)
            y = complex(np.nan, np.nan)
            #assert_raises(FloatingPointError, np.log, x)
        with np.errstate(invalid='ignore'):
            assert_almost_equal(np.log(x), y)

        xl.append(x)
        yl.append(y)

        with np.errstate(invalid='raise'):
            x = np.array([np.inf + 1j * np.nan], dtype=complex)
            #assert_raises(FloatingPointError, np.log, x)
        with np.errstate(invalid='ignore'):
            assert_almost_equal(np.log(x), y)

        xl.append(x)
        yl.append(y)

        # clog(- inf + iy) returns +inf + ipi , for finite positive-signed y.
        x = np.array([-np.inf + 1j], dtype=complex)
        y = complex(np.inf, np.pi)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(+ inf + iy) returns +inf + i0, for finite positive-signed y.
        x = np.array([np.inf + 1j], dtype=complex)
        y = complex(np.inf, 0)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(- inf + i inf) returns +inf + i3pi /4.
        x = np.array([complex(-np.inf, np.inf)], dtype=complex)
        y = complex(np.inf, 0.75 * np.pi)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(+ inf + i inf) returns +inf + ipi /4.
        x = np.array([complex(np.inf, np.inf)], dtype=complex)
        y = complex(np.inf, 0.25 * np.pi)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(+/- inf + iNaN) returns +inf + iNaN.
        x = np.array([complex(np.inf, np.nan)], dtype=complex)
        y = complex(np.inf, np.nan)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        x = np.array([complex(-np.inf, np.nan)], dtype=complex)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(NaN + iy) returns NaN + iNaN and optionally raises the
        # 'invalid' floating-point exception, for finite y.
        x = np.array([complex(np.nan, 1)], dtype=complex)
        y = complex(np.nan, np.nan)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(NaN + i inf) returns +inf + iNaN.
        x = np.array([complex(np.nan, np.inf)], dtype=complex)
        y = complex(np.inf, np.nan)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(NaN + iNaN) returns NaN + iNaN.
        x = np.array([complex(np.nan, np.nan)], dtype=complex)
        y = complex(np.nan, np.nan)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(conj(z)) = conj(clog(z)).
        xa = np.array(xl, dtype=complex)
        ya = np.array(yl, dtype=complex)
        with np.errstate(divide='ignore'):
            for i in range(len(xa)):
                assert_almost_equal(np.log(xa[i].conj()), ya[i].conj())


class TestCsqrt:

    def test_simple(self):
        # sqrt(1)
        check_complex_value(np.sqrt, 1, 0, 1, 0)

        # sqrt(1i)
        rres = 0.5 * np.sqrt(2)
        ires = rres
        check_complex_value(np.sqrt, 0, 1, rres, ires, False)

        # sqrt(-1)
        check_complex_value(np.sqrt, -1, 0, 0, 1)

    def test_simple_conjugate(self):
        ref = np.conj(np.sqrt(complex(1, 1)))

        def f(z):
            return np.sqrt(np.conj(z))

        check_complex_value(f, 1, 1, ref.real, ref.imag, False)

    #def test_branch_cut(self):
    #    _check_branch_cut(f, -1, 0, 1, -1)

    @platform_skip
    def test_special_values(self):
        # C99: Sec G 6.4.2

        check = check_complex_value
        f = np.sqrt

        # csqrt(+-0 + 0i) is 0 + 0i
        check(f, ncu.PZERO, 0, 0, 0)
        check(f, ncu.NZERO, 0, 0, 0)

        # csqrt(x + infi) is inf + infi for any x (including NaN)
        check(f,  1, np.inf, np.inf, np.inf)
        check(f, -1, np.inf, np.inf, np.inf)

        check(f, ncu.PZERO, np.inf, np.inf, np.inf)
        check(f, ncu.NZERO, np.inf, np.inf, np.inf)
        check(f,    np.inf, np.inf, np.inf, np.inf)
        check(f,   -np.inf, np.inf, np.inf, np.inf)  # noqa: E221
        check(f,   -np.nan, np.inf, np.inf, np.inf)  # noqa: E221

        # csqrt(x + nani) is nan + nani for any finite x
        check(f,  1, np.nan, np.nan, np.nan)
        check(f, -1, np.nan, np.nan, np.nan)
        check(f,  0, np.nan, np.nan, np.nan)

        # csqrt(-inf + yi) is +0 + infi for any finite y > 0
        check(f, -np.inf, 1, ncu.PZERO, np.inf)

        # csqrt(inf + yi) is +inf + 0i for any finite y > 0
        check(f, np.inf, 1, np.inf, ncu.PZERO)

        # csqrt(-inf + nani) is nan +- infi (both +i infi are valid)
        def _check_ninf_nan(dummy):
            msgform = "csqrt(-inf, nan) is (%f, %f), expected (nan, +-inf)"
            z = np.sqrt(np.array(complex(-np.inf, np.nan)))
            # FIXME: ugly workaround for isinf bug.
            with np.errstate(invalid='ignore'):
                if not (np.isnan(z.real) and np.isinf(z.imag)):
                    raise AssertionError(msgform % (z.real, z.imag))

        _check_ninf_nan(None)

        # csqrt(+inf + nani) is inf + nani
        check(f, np.inf, np.nan, np.inf, np.nan)

        # csqrt(nan + yi) is nan + nani for any finite y (infinite handled in x
        # + nani)
        check(f, np.nan,       0, np.nan, np.nan)
        check(f, np.nan,       1, np.nan, np.nan)
        check(f, np.nan,  np.nan, np.nan, np.nan)

        # XXX: check for conj(csqrt(z)) == csqrt(conj(z)) (need to fix branch
        # cuts first)

class TestCpow:
    def setup_method(self):
        self.olderr = np.seterr(invalid='ignore')

    def teardown_method(self):
        np.seterr(**self.olderr)

    def test_simple(self):
        x = np.array([1 + 1j, 0 + 2j, 1 + 2j, np.inf, np.nan])
        y_r = x ** 2
        y = np.power(x, 2)
        assert_almost_equal(y, y_r)

    def test_scalar(self):
        x = np.array([1, 1j,         2,  2.5 + .37j, np.inf, np.nan])
        y = np.array([1, 1j, -0.5 + 1.5j, -0.5 + 1.5j,      2,      3])
        lx = list(range(len(x)))

        # Hardcode the expected `builtins.complex` values,
        # as complex exponentiation is broken as of bpo-44698
        p_r = [
            1 + 0j,
            0.20787957635076193 + 0j,
            0.35812203996480685 + 0.6097119028618724j,
            0.12659112128185032 + 0.48847676699581527j,
            complex(np.inf, np.nan),
            complex(np.nan, np.nan),
        ]

        n_r = [x[i] ** y[i] for i in lx]
        for i in lx:
            assert_almost_equal(n_r[i], p_r[i], err_msg='Loop %d\n' % i)

    def test_array(self):
        x = np.array([1, 1j,         2,  2.5 + .37j, np.inf, np.nan])
        y = np.array([1, 1j, -0.5 + 1.5j, -0.5 + 1.5j,      2,      3])
        lx = list(range(len(x)))

        # Hardcode the expected `builtins.complex` values,
        # as complex exponentiation is broken as of bpo-44698
        p_r = [
            1 + 0j,
            0.20787957635076193 + 0j,
            0.35812203996480685 + 0.6097119028618724j,
            0.12659112128185032 + 0.48847676699581527j,
            complex(np.inf, np.nan),
            complex(np.nan, np.nan),
        ]

        n_r = x ** y
        for i in lx:
            assert_almost_equal(n_r[i], p_r[i], err_msg='Loop %d\n' % i)

class TestCabs:
    def setup_method(self):
        self.olderr = np.seterr(invalid='ignore')

    def teardown_method(self):
        np.seterr(**self.olderr)

    def test_simple(self):
        x = np.array([1 + 1j, 0 + 2j, 1 + 2j, np.inf, np.nan])
        y_r = np.array([np.sqrt(2.), 2, np.sqrt(5), np.inf, np.nan])
        y = np.abs(x)
        assert_almost_equal(y, y_r)

    def test_fabs(self):
        # Test that np.abs(x +- 0j) == np.abs(x) (as mandated by C99 for cabs)
        x = np.array([1 + 0j], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))

        x = np.array([complex(1, ncu.NZERO)], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))

        x = np.array([complex(np.inf, ncu.NZERO)], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))

        x = np.array([complex(np.nan, ncu.NZERO)], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))

    def test_cabs_inf_nan(self):
        x, y = [], []

        # cabs(+-nan + nani) returns nan
        x.append(np.nan)
        y.append(np.nan)
        check_real_value(np.abs,  np.nan, np.nan, np.nan)

        x.append(np.nan)
        y.append(-np.nan)
        check_real_value(np.abs, -np.nan, np.nan, np.nan)

        # According to C99 standard, if exactly one of the real/part is inf and
        # the other nan, then cabs should return inf
        x.append(np.inf)
        y.append(np.nan)
        check_real_value(np.abs,  np.inf, np.nan, np.inf)

        x.append(-np.inf)
        y.append(np.nan)
        check_real_value(np.abs, -np.inf, np.nan, np.inf)

        # cabs(conj(z)) == conj(cabs(z)) (= cabs(z))
        def f(a):
            return np.abs(np.conj(a))

        def g(a, b):
            return np.abs(complex(a, b))

        xa = np.array(x, dtype=complex)
        assert len(xa) == len(x) == len(y)
        for xi, yi in zip(x, y):
            ref = g(xi, yi)
            check_real_value(f, xi, yi, ref)

class TestCarg:
    def test_simple(self):
        check_real_value(ncu._arg, 1, 0, 0, False)
        check_real_value(ncu._arg, 0, 1, 0.5 * np.pi, False)

        check_real_value(ncu._arg, 1, 1, 0.25 * np.pi, False)
        check_real_value(ncu._arg, ncu.PZERO, ncu.PZERO, ncu.PZERO)

    # TODO This can be xfail when the generator functions are got rid of.
    @pytest.mark.skip(
        reason="Complex arithmetic with signed zero fails on most platforms")
    def test_zero(self):
        # carg(-0 +- 0i) returns +- pi
        check_real_value(ncu._arg, ncu.NZERO, ncu.PZERO,  np.pi, False)
        check_real_value(ncu._arg, ncu.NZERO, ncu.NZERO, -np.pi, False)

        # carg(+0 +- 0i) returns +- 0
        check_real_value(ncu._arg, ncu.PZERO, ncu.PZERO, ncu.PZERO)
        check_real_value(ncu._arg, ncu.PZERO, ncu.NZERO, ncu.NZERO)

        # carg(x +- 0i) returns +- 0 for x > 0
        check_real_value(ncu._arg, 1, ncu.PZERO, ncu.PZERO, False)
        check_real_value(ncu._arg, 1, ncu.NZERO, ncu.NZERO, False)

        # carg(x +- 0i) returns +- pi for x < 0
        check_real_value(ncu._arg, -1, ncu.PZERO,  np.pi, False)
        check_real_value(ncu._arg, -1, ncu.NZERO, -np.pi, False)

        # carg(+- 0 + yi) returns pi/2 for y > 0
        check_real_value(ncu._arg, ncu.PZERO, 1, 0.5 * np.pi, False)
        check_real_value(ncu._arg, ncu.NZERO, 1, 0.5 * np.pi, False)

        # carg(+- 0 + yi) returns -pi/2 for y < 0
        check_real_value(ncu._arg, ncu.PZERO, -1, 0.5 * np.pi, False)
        check_real_value(ncu._arg, ncu.NZERO, -1, -0.5 * np.pi, False)

    #def test_branch_cuts(self):
    #    _check_branch_cut(ncu._arg, -1, 1j, -1, 1)

    def test_special_values(self):
        # carg(-np.inf +- yi) returns +-pi for finite y > 0
        check_real_value(ncu._arg, -np.inf,  1,  np.pi, False)
        check_real_value(ncu._arg, -np.inf, -1, -np.pi, False)

        # carg(np.inf +- yi) returns +-0 for finite y > 0
        check_real_value(ncu._arg, np.inf,  1, ncu.PZERO, False)
        check_real_value(ncu._arg, np.inf, -1, ncu.NZERO, False)

        # carg(x +- np.infi) returns +-pi/2 for finite x
        check_real_value(ncu._arg, 1,  np.inf,  0.5 * np.pi, False)
        check_real_value(ncu._arg, 1, -np.inf, -0.5 * np.pi, False)

        # carg(-np.inf +- np.infi) returns +-3pi/4
        check_real_value(ncu._arg, -np.inf,  np.inf,  0.75 * np.pi, False)
        check_real_value(ncu._arg, -np.inf, -np.inf, -0.75 * np.pi, False)

        # carg(np.inf +- np.infi) returns +-pi/4
        check_real_value(ncu._arg, np.inf,  np.inf,  0.25 * np.pi, False)
        check_real_value(ncu._arg, np.inf, -np.inf, -0.25 * np.pi, False)

        # carg(x + yi) returns np.nan if x or y is nan
        check_real_value(ncu._arg, np.nan,      0, np.nan, False)
        check_real_value(ncu._arg,      0, np.nan, np.nan, False)

        check_real_value(ncu._arg, np.nan, np.inf, np.nan, False)
        check_real_value(ncu._arg, np.inf, np.nan, np.nan, False)


def check_real_value(f, x1, y1, x, exact=True):
    z1 = np.array([complex(x1, y1)])
    if exact:
        assert_equal(f(z1), x)
    else:
        assert_almost_equal(f(z1), x)


def check_complex_value(f, x1, y1, x2, y2, exact=True):
    z1 = np.array([complex(x1, y1)])
    z2 = complex(x2, y2)
    with np.errstate(invalid='ignore'):
        if exact:
            assert_equal(f(z1), z2)
        else:
            assert_almost_equal(f(z1), z2)

class TestSpecialComplexAVX:
    @pytest.mark.parametrize("stride", [-4, -2, -1, 1, 2, 4])
    @pytest.mark.parametrize("astype", [np.complex64, np.complex128])
    def test_array(self, stride, astype):
        arr = np.array([complex(np.nan, np.nan),
                        complex(np.nan, np.inf),
                        complex(np.inf, np.nan),
                        complex(np.inf, np.inf),
                        complex(0.,     np.inf),
                        complex(np.inf, 0.),
                        complex(0.,     0.),
                        complex(0.,     np.nan),
                        complex(np.nan, 0.)], dtype=astype)
        abs_true = np.array([np.nan, np.inf, np.inf, np.inf, np.inf, np.inf, 0., np.nan, np.nan], dtype=arr.real.dtype)
        sq_true = np.array([complex(np.nan,  np.nan),
                            complex(np.nan,  np.nan),
                            complex(np.nan,  np.nan),
                            complex(np.nan,  np.inf),
                            complex(-np.inf, np.nan),
                            complex(np.inf,  np.nan),
                            complex(0.,      0.),
                            complex(np.nan,  np.nan),
                            complex(np.nan,  np.nan)], dtype=astype)
        with np.errstate(invalid='ignore'):
            assert_equal(np.abs(arr[::stride]), abs_true[::stride])
            assert_equal(np.square(arr[::stride]), sq_true[::stride])

class TestComplexAbsoluteAVX:
    @pytest.mark.parametrize("arraysize", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 18, 19])
    @pytest.mark.parametrize("stride", [-4, -3, -2, -1, 1, 2, 3, 4])
    @pytest.mark.parametrize("astype", [np.complex64, np.complex128])
    # test to ensure masking and strides work as intended in the AVX implementation
    def test_array(self, arraysize, stride, astype):
        arr = np.ones(arraysize, dtype=astype)
        abs_true = np.ones(arraysize, dtype=arr.real.dtype)
        assert_equal(np.abs(arr[::stride]), abs_true[::stride])

# Testcase taken as is from https://github.com/numpy/numpy/issues/16660
class TestComplexAbsoluteMixedDTypes:
    @pytest.mark.parametrize("stride", [-4, -3, -2, -1, 1, 2, 3, 4])
    @pytest.mark.parametrize("astype", [np.complex64, np.complex128])
    @pytest.mark.parametrize("func", ['abs', 'square', 'conjugate'])
    def test_array(self, stride, astype, func):
        dtype = [('template_id', '<i8'), ('bank_chisq', '<f4'),
                 ('bank_chisq_dof', '<i8'), ('chisq', '<f4'), ('chisq_dof', '<i8'),
                 ('cont_chisq', '<f4'), ('psd_var_val', '<f4'), ('sg_chisq', '<f4'),
                 ('mycomplex', astype), ('time_index', '<i8')]
        vec = np.array([
                (0, 0., 0, -31.666483, 200, 0., 0.,  1.      ,  3.0 + 4.0j  ,  613090),   # noqa: E203,E501
                (1, 0., 0, 260.91525 ,  42, 0., 0.,  1.      ,  5.0 + 12.0j ,  787315),   # noqa: E203,E501
                (1, 0., 0,  52.15155 ,  42, 0., 0.,  1.      ,  8.0 + 15.0j ,  806641),   # noqa: E203,E501
                (1, 0., 0,  52.430195,  42, 0., 0.,  1.      ,  7.0 + 24.0j , 1363540),   # noqa: E203,E501
                (2, 0., 0, 304.43646 ,  58, 0., 0.,  1.      ,  20.0 + 21.0j,  787323),   # noqa: E203,E501
                (3, 0., 0, 299.42108 ,  52, 0., 0.,  1.      ,  12.0 + 35.0j,  787332),   # noqa: E203,E501
                (4, 0., 0,  39.4836  ,  28, 0., 0.,  9.182192,  9.0 + 40.0j ,  787304),   # noqa: E203,E501
                (4, 0., 0,  76.83787 ,  28, 0., 0.,  1.      ,  28.0 + 45.0j, 1321869),   # noqa: E203,E501
                (5, 0., 0, 143.26366 ,  24, 0., 0., 10.996129,  11.0 + 60.0j,  787299)],  # noqa: E203,E501
            dtype=dtype)
        myfunc = getattr(np, func)
        a = vec['mycomplex']
        g = myfunc(a[::stride])

        b = vec['mycomplex'].copy()
        h = myfunc(b[::stride])

        assert_array_max_ulp(h.real, g.real, 1)
        assert_array_max_ulp(h.imag, g.imag, 1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\tests\test_chebyshev.py ===
"""Tests for chebyshev module.

"""
from functools import reduce

import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)


def trim(x):
    return cheb.chebtrim(x, tol=1e-6)


T0 = [1]
T1 = [0, 1]
T2 = [-1, 0, 2]
T3 = [0, -3, 0, 4]
T4 = [1, 0, -8, 0, 8]
T5 = [0, 5, 0, -20, 0, 16]
T6 = [-1, 0, 18, 0, -48, 0, 32]
T7 = [0, -7, 0, 56, 0, -112, 0, 64]
T8 = [1, 0, -32, 0, 160, 0, -256, 0, 128]
T9 = [0, 9, 0, -120, 0, 432, 0, -576, 0, 256]

Tlist = [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9]


class TestPrivate:

    def test__cseries_to_zseries(self):
        for i in range(5):
            inp = np.array([2] + [1] * i, np.double)
            tgt = np.array([.5] * i + [2] + [.5] * i, np.double)
            res = cheb._cseries_to_zseries(inp)
            assert_equal(res, tgt)

    def test__zseries_to_cseries(self):
        for i in range(5):
            inp = np.array([.5] * i + [2] + [.5] * i, np.double)
            tgt = np.array([2] + [1] * i, np.double)
            res = cheb._zseries_to_cseries(inp)
            assert_equal(res, tgt)


class TestConstants:

    def test_chebdomain(self):
        assert_equal(cheb.chebdomain, [-1, 1])

    def test_chebzero(self):
        assert_equal(cheb.chebzero, [0])

    def test_chebone(self):
        assert_equal(cheb.chebone, [1])

    def test_chebx(self):
        assert_equal(cheb.chebx, [0, 1])


class TestArithmetic:

    def test_chebadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = cheb.chebadd([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = cheb.chebsub([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebmulx(self):
        assert_equal(cheb.chebmulx([0]), [0])
        assert_equal(cheb.chebmulx([1]), [0, 1])
        for i in range(1, 5):
            ser = [0] * i + [1]
            tgt = [0] * (i - 1) + [.5, 0, .5]
            assert_equal(cheb.chebmulx(ser), tgt)

    def test_chebmul(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(i + j + 1)
                tgt[i + j] += .5
                tgt[abs(i - j)] += .5
                res = cheb.chebmul([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebdiv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0] * i + [1]
                cj = [0] * j + [1]
                tgt = cheb.chebadd(ci, cj)
                quo, rem = cheb.chebdiv(tgt, ci)
                res = cheb.chebadd(cheb.chebmul(quo, ci), rem)
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebpow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = np.arange(i + 1)
                tgt = reduce(cheb.chebmul, [c] * j, np.array([1]))
                res = cheb.chebpow(c, j)
                assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = np.array([2.5, 2., 1.5])
    c2d = np.einsum('i,j->ij', c1d, c1d)
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = np.random.random((3, 5)) * 2 - 1
    y = polyval(x, [1., 2., 3.])

    def test_chebval(self):
        # check empty input
        assert_equal(cheb.chebval([], [1]).size, 0)

        # check normal input)
        x = np.linspace(-1, 1)
        y = [polyval(x, c) for c in Tlist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = cheb.chebval(x, [0] * i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)

        # check that shape is preserved
        for i in range(3):
            dims = [2] * i
            x = np.zeros(dims)
            assert_equal(cheb.chebval(x, [1]).shape, dims)
            assert_equal(cheb.chebval(x, [1, 0]).shape, dims)
            assert_equal(cheb.chebval(x, [1, 0, 0]).shape, dims)

    def test_chebval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        assert_raises(ValueError, cheb.chebval2d, x1, x2[:2], self.c2d)

        # test values
        tgt = y1 * y2
        res = cheb.chebval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = cheb.chebval2d(z, z, self.c2d)
        assert_(res.shape == (2, 3))

    def test_chebval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        assert_raises(ValueError, cheb.chebval3d, x1, x2, x3[:2], self.c3d)

        # test values
        tgt = y1 * y2 * y3
        res = cheb.chebval3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = cheb.chebval3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3))

    def test_chebgrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = np.einsum('i,j->ij', y1, y2)
        res = cheb.chebgrid2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = cheb.chebgrid2d(z, z, self.c2d)
        assert_(res.shape == (2, 3) * 2)

    def test_chebgrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        res = cheb.chebgrid3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = cheb.chebgrid3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3) * 3)


class TestIntegral:

    def test_chebint(self):
        # check exceptions
        assert_raises(TypeError, cheb.chebint, [0], .5)
        assert_raises(ValueError, cheb.chebint, [0], -1)
        assert_raises(ValueError, cheb.chebint, [0], 1, [0, 0])
        assert_raises(ValueError, cheb.chebint, [0], lbnd=[0])
        assert_raises(ValueError, cheb.chebint, [0], scl=[0])
        assert_raises(TypeError, cheb.chebint, [0], axis=.5)

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = cheb.chebint([0], m=i, k=k)
            assert_almost_equal(res, [0, 1])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            chebpol = cheb.poly2cheb(pol)
            chebint = cheb.chebint(chebpol, m=1, k=[i])
            res = cheb.cheb2poly(chebint)
            assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            chebpol = cheb.poly2cheb(pol)
            chebint = cheb.chebint(chebpol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(cheb.chebval(-1, chebint), i)

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            chebpol = cheb.poly2cheb(pol)
            chebint = cheb.chebint(chebpol, m=1, k=[i], scl=2)
            res = cheb.cheb2poly(chebint)
            assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = cheb.chebint(tgt, m=1)
                res = cheb.chebint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = cheb.chebint(tgt, m=1, k=[k])
                res = cheb.chebint(pol, m=j, k=list(range(j)))
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = cheb.chebint(tgt, m=1, k=[k], lbnd=-1)
                res = cheb.chebint(pol, m=j, k=list(range(j)), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = cheb.chebint(tgt, m=1, k=[k], scl=2)
                res = cheb.chebint(pol, m=j, k=list(range(j)), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_chebint_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([cheb.chebint(c) for c in c2d.T]).T
        res = cheb.chebint(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([cheb.chebint(c) for c in c2d])
        res = cheb.chebint(c2d, axis=1)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([cheb.chebint(c, k=3) for c in c2d])
        res = cheb.chebint(c2d, k=3, axis=1)
        assert_almost_equal(res, tgt)


class TestDerivative:

    def test_chebder(self):
        # check exceptions
        assert_raises(TypeError, cheb.chebder, [0], .5)
        assert_raises(ValueError, cheb.chebder, [0], -1)

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0] * i + [1]
            res = cheb.chebder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = cheb.chebder(cheb.chebint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = cheb.chebder(cheb.chebint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_chebder_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([cheb.chebder(c) for c in c2d.T]).T
        res = cheb.chebder(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([cheb.chebder(c) for c in c2d])
        res = cheb.chebder(c2d, axis=1)
        assert_almost_equal(res, tgt)


class TestVander:
    # some random values in [-1, 1)
    x = np.random.random((3, 5)) * 2 - 1

    def test_chebvander(self):
        # check for 1d x
        x = np.arange(3)
        v = cheb.chebvander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            assert_almost_equal(v[..., i], cheb.chebval(x, coef))

        # check for 2d x
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = cheb.chebvander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            assert_almost_equal(v[..., i], cheb.chebval(x, coef))

    def test_chebvander2d(self):
        # also tests chebval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3))
        van = cheb.chebvander2d(x1, x2, [1, 2])
        tgt = cheb.chebval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = cheb.chebvander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))

    def test_chebvander3d(self):
        # also tests chebval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3, 4))
        van = cheb.chebvander3d(x1, x2, x3, [1, 2, 3])
        tgt = cheb.chebval3d(x1, x2, x3, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = cheb.chebvander3d([x1], [x2], [x3], [1, 2, 3])
        assert_(van.shape == (1, 5, 24))


class TestFitting:

    def test_chebfit(self):
        def f(x):
            return x * (x - 1) * (x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        # Test exceptions
        assert_raises(ValueError, cheb.chebfit, [1], [1], -1)
        assert_raises(TypeError, cheb.chebfit, [[1]], [1], 0)
        assert_raises(TypeError, cheb.chebfit, [], [1], 0)
        assert_raises(TypeError, cheb.chebfit, [1], [[[1]]], 0)
        assert_raises(TypeError, cheb.chebfit, [1, 2], [1], 0)
        assert_raises(TypeError, cheb.chebfit, [1], [1, 2], 0)
        assert_raises(TypeError, cheb.chebfit, [1], [1], 0, w=[[1]])
        assert_raises(TypeError, cheb.chebfit, [1], [1], 0, w=[1, 1])
        assert_raises(ValueError, cheb.chebfit, [1], [1], [-1,])
        assert_raises(ValueError, cheb.chebfit, [1], [1], [2, -1, 6])
        assert_raises(TypeError, cheb.chebfit, [1], [1], [])

        # Test fit
        x = np.linspace(0, 2)
        y = f(x)
        #
        coef3 = cheb.chebfit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(cheb.chebval(x, coef3), y)
        coef3 = cheb.chebfit(x, y, [0, 1, 2, 3])
        assert_equal(len(coef3), 4)
        assert_almost_equal(cheb.chebval(x, coef3), y)
        #
        coef4 = cheb.chebfit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(cheb.chebval(x, coef4), y)
        coef4 = cheb.chebfit(x, y, [0, 1, 2, 3, 4])
        assert_equal(len(coef4), 5)
        assert_almost_equal(cheb.chebval(x, coef4), y)
        # check things still work if deg is not in strict increasing
        coef4 = cheb.chebfit(x, y, [2, 3, 4, 1, 0])
        assert_equal(len(coef4), 5)
        assert_almost_equal(cheb.chebval(x, coef4), y)
        #
        coef2d = cheb.chebfit(x, np.array([y, y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        coef2d = cheb.chebfit(x, np.array([y, y]).T, [0, 1, 2, 3])
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        # test weighting
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = cheb.chebfit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        wcoef3 = cheb.chebfit(x, yw, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = cheb.chebfit(x, np.array([yw, yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        wcoef2d = cheb.chebfit(x, np.array([yw, yw]).T, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        assert_almost_equal(cheb.chebfit(x, x, 1), [0, 1])
        assert_almost_equal(cheb.chebfit(x, x, [0, 1]), [0, 1])
        # test fitting only even polynomials
        x = np.linspace(-1, 1)
        y = f2(x)
        coef1 = cheb.chebfit(x, y, 4)
        assert_almost_equal(cheb.chebval(x, coef1), y)
        coef2 = cheb.chebfit(x, y, [0, 2, 4])
        assert_almost_equal(cheb.chebval(x, coef2), y)
        assert_almost_equal(coef1, coef2)


class TestInterpolate:

    def f(self, x):
        return x * (x - 1) * (x - 2)

    def test_raises(self):
        assert_raises(ValueError, cheb.chebinterpolate, self.f, -1)
        assert_raises(TypeError, cheb.chebinterpolate, self.f, 10.)

    def test_dimensions(self):
        for deg in range(1, 5):
            assert_(cheb.chebinterpolate(self.f, deg).shape == (deg + 1,))

    def test_approximation(self):

        def powx(x, p):
            return x**p

        x = np.linspace(-1, 1, 10)
        for deg in range(10):
            for p in range(deg + 1):
                c = cheb.chebinterpolate(powx, deg, (p,))
                assert_almost_equal(cheb.chebval(x, c), powx(x, p), decimal=12)


class TestCompanion:

    def test_raises(self):
        assert_raises(ValueError, cheb.chebcompanion, [])
        assert_raises(ValueError, cheb.chebcompanion, [1])

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0] * i + [1]
            assert_(cheb.chebcompanion(coef).shape == (i, i))

    def test_linear_root(self):
        assert_(cheb.chebcompanion([1, 2])[0, 0] == -.5)


class TestGauss:

    def test_100(self):
        x, w = cheb.chebgauss(100)

        # test orthogonality. Note that the results need to be normalized,
        # otherwise the huge values that can arise from fast growing
        # functions like Laguerre can be very confusing.
        v = cheb.chebvander(x, 99)
        vv = np.dot(v.T * w, v)
        vd = 1 / np.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        assert_almost_equal(vv, np.eye(100))

        # check that the integral of 1 is correct
        tgt = np.pi
        assert_almost_equal(w.sum(), tgt)


class TestMisc:

    def test_chebfromroots(self):
        res = cheb.chebfromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = np.cos(np.linspace(-np.pi, 0, 2 * i + 1)[1::2])
            tgt = [0] * i + [1]
            res = cheb.chebfromroots(roots) * 2**(i - 1)
            assert_almost_equal(trim(res), trim(tgt))

    def test_chebroots(self):
        assert_almost_equal(cheb.chebroots([1]), [])
        assert_almost_equal(cheb.chebroots([1, 2]), [-.5])
        for i in range(2, 5):
            tgt = np.linspace(-1, 1, i)
            res = cheb.chebroots(cheb.chebfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_chebtrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        assert_raises(ValueError, cheb.chebtrim, coef, -1)

        # Test results
        assert_equal(cheb.chebtrim(coef), coef[:-1])
        assert_equal(cheb.chebtrim(coef, 1), coef[:-3])
        assert_equal(cheb.chebtrim(coef, 2), [0])

    def test_chebline(self):
        assert_equal(cheb.chebline(3, 4), [3, 4])

    def test_cheb2poly(self):
        for i in range(10):
            assert_almost_equal(cheb.cheb2poly([0] * i + [1]), Tlist[i])

    def test_poly2cheb(self):
        for i in range(10):
            assert_almost_equal(cheb.poly2cheb(Tlist[i]), [0] * i + [1])

    def test_weight(self):
        x = np.linspace(-1, 1, 11)[1:-1]
        tgt = 1. / (np.sqrt(1 + x) * np.sqrt(1 - x))
        res = cheb.chebweight(x)
        assert_almost_equal(res, tgt)

    def test_chebpts1(self):
        # test exceptions
        assert_raises(ValueError, cheb.chebpts1, 1.5)
        assert_raises(ValueError, cheb.chebpts1, 0)

        # test points
        tgt = [0]
        assert_almost_equal(cheb.chebpts1(1), tgt)
        tgt = [-0.70710678118654746, 0.70710678118654746]
        assert_almost_equal(cheb.chebpts1(2), tgt)
        tgt = [-0.86602540378443871, 0, 0.86602540378443871]
        assert_almost_equal(cheb.chebpts1(3), tgt)
        tgt = [-0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325]
        assert_almost_equal(cheb.chebpts1(4), tgt)

    def test_chebpts2(self):
        # test exceptions
        assert_raises(ValueError, cheb.chebpts2, 1.5)
        assert_raises(ValueError, cheb.chebpts2, 1)

        # test points
        tgt = [-1, 1]
        assert_almost_equal(cheb.chebpts2(2), tgt)
        tgt = [-1, 0, 1]
        assert_almost_equal(cheb.chebpts2(3), tgt)
        tgt = [-1, -0.5, .5, 1]
        assert_almost_equal(cheb.chebpts2(4), tgt)
        tgt = [-1.0, -0.707106781187, 0, 0.707106781187, 1.0]
        assert_almost_equal(cheb.chebpts2(5), tgt)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_numerictypes.py ===
import itertools
import sys

import pytest

import numpy as np
import numpy._core.numerictypes as nt
from numpy._core.numerictypes import issctype, maximum_sctype, sctype2char, sctypes
from numpy.testing import (
    IS_PYPY,
    assert_,
    assert_equal,
    assert_raises,
    assert_raises_regex,
)

# This is the structure of the table used for plain objects:
#
# +-+-+-+
# |x|y|z|
# +-+-+-+

# Structure of a plain array description:
Pdescr = [
    ('x', 'i4', (2,)),
    ('y', 'f8', (2, 2)),
    ('z', 'u1')]

# A plain list of tuples with values for testing:
PbufferT = [
    # x     y                  z
    ([3, 2], [[6., 4.], [6., 4.]], 8),
    ([4, 3], [[7., 5.], [7., 5.]], 9),
    ]


# This is the structure of the table used for nested objects (DON'T PANIC!):
#
# +-+---------------------------------+-----+----------+-+-+
# |x|Info                             |color|info      |y|z|
# | +-----+--+----------------+----+--+     +----+-----+ | |
# | |value|y2|Info2           |name|z2|     |Name|Value| | |
# | |     |  +----+-----+--+--+    |  |     |    |     | | |
# | |     |  |name|value|y3|z3|    |  |     |    |     | | |
# +-+-----+--+----+-----+--+--+----+--+-----+----+-----+-+-+
#

# The corresponding nested array description:
Ndescr = [
    ('x', 'i4', (2,)),
    ('Info', [
        ('value', 'c16'),
        ('y2', 'f8'),
        ('Info2', [
            ('name', 'S2'),
            ('value', 'c16', (2,)),
            ('y3', 'f8', (2,)),
            ('z3', 'u4', (2,))]),
        ('name', 'S2'),
        ('z2', 'b1')]),
    ('color', 'S2'),
    ('info', [
        ('Name', 'U8'),
        ('Value', 'c16')]),
    ('y', 'f8', (2, 2)),
    ('z', 'u1')]

NbufferT = [
    # x     Info                                                color info        y                  z
    #       value y2 Info2                            name z2         Name Value
    #                name   value    y3       z3
    ([3, 2], (6j, 6., (b'nn', [6j, 4j], [6., 4.], [1, 2]), b'NN', True),
     b'cc', ('NN', 6j), [[6., 4.], [6., 4.]], 8),
    ([4, 3], (7j, 7., (b'oo', [7j, 5j], [7., 5.], [2, 1]), b'OO', False),
     b'dd', ('OO', 7j), [[7., 5.], [7., 5.]], 9),
    ]


byteorder = {'little': '<', 'big': '>'}[sys.byteorder]

def normalize_descr(descr):
    "Normalize a description adding the platform byteorder."

    out = []
    for item in descr:
        dtype = item[1]
        if isinstance(dtype, str):
            if dtype[0] not in ['|', '<', '>']:
                onebyte = dtype[1:] == "1"
                if onebyte or dtype[0] in ['S', 'V', 'b']:
                    dtype = "|" + dtype
                else:
                    dtype = byteorder + dtype
            if len(item) > 2 and np.prod(item[2]) > 1:
                nitem = (item[0], dtype, item[2])
            else:
                nitem = (item[0], dtype)
            out.append(nitem)
        elif isinstance(dtype, list):
            l = normalize_descr(dtype)
            out.append((item[0], l))
        else:
            raise ValueError(f"Expected a str or list and got {type(item)}")
    return out


############################################################
#    Creation tests
############################################################

class CreateZeros:
    """Check the creation of heterogeneous arrays zero-valued"""

    def test_zeros0D(self):
        """Check creation of 0-dimensional objects"""
        h = np.zeros((), dtype=self._descr)
        assert_(normalize_descr(self._descr) == h.dtype.descr)
        assert_(h.dtype.fields['x'][0].name[:4] == 'void')
        assert_(h.dtype.fields['x'][0].char == 'V')
        assert_(h.dtype.fields['x'][0].type == np.void)
        # A small check that data is ok
        assert_equal(h['z'], np.zeros((), dtype='u1'))

    def test_zerosSD(self):
        """Check creation of single-dimensional objects"""
        h = np.zeros((2,), dtype=self._descr)
        assert_(normalize_descr(self._descr) == h.dtype.descr)
        assert_(h.dtype['y'].name[:4] == 'void')
        assert_(h.dtype['y'].char == 'V')
        assert_(h.dtype['y'].type == np.void)
        # A small check that data is ok
        assert_equal(h['z'], np.zeros((2,), dtype='u1'))

    def test_zerosMD(self):
        """Check creation of multi-dimensional objects"""
        h = np.zeros((2, 3), dtype=self._descr)
        assert_(normalize_descr(self._descr) == h.dtype.descr)
        assert_(h.dtype['z'].name == 'uint8')
        assert_(h.dtype['z'].char == 'B')
        assert_(h.dtype['z'].type == np.uint8)
        # A small check that data is ok
        assert_equal(h['z'], np.zeros((2, 3), dtype='u1'))


class TestCreateZerosPlain(CreateZeros):
    """Check the creation of heterogeneous arrays zero-valued (plain)"""
    _descr = Pdescr

class TestCreateZerosNested(CreateZeros):
    """Check the creation of heterogeneous arrays zero-valued (nested)"""
    _descr = Ndescr


class CreateValues:
    """Check the creation of heterogeneous arrays with values"""

    def test_tuple(self):
        """Check creation from tuples"""
        h = np.array(self._buffer, dtype=self._descr)
        assert_(normalize_descr(self._descr) == h.dtype.descr)
        if self.multiple_rows:
            assert_(h.shape == (2,))
        else:
            assert_(h.shape == ())

    def test_list_of_tuple(self):
        """Check creation from list of tuples"""
        h = np.array([self._buffer], dtype=self._descr)
        assert_(normalize_descr(self._descr) == h.dtype.descr)
        if self.multiple_rows:
            assert_(h.shape == (1, 2))
        else:
            assert_(h.shape == (1,))

    def test_list_of_list_of_tuple(self):
        """Check creation from list of list of tuples"""
        h = np.array([[self._buffer]], dtype=self._descr)
        assert_(normalize_descr(self._descr) == h.dtype.descr)
        if self.multiple_rows:
            assert_(h.shape == (1, 1, 2))
        else:
            assert_(h.shape == (1, 1))


class TestCreateValuesPlainSingle(CreateValues):
    """Check the creation of heterogeneous arrays (plain, single row)"""
    _descr = Pdescr
    multiple_rows = 0
    _buffer = PbufferT[0]

class TestCreateValuesPlainMultiple(CreateValues):
    """Check the creation of heterogeneous arrays (plain, multiple rows)"""
    _descr = Pdescr
    multiple_rows = 1
    _buffer = PbufferT

class TestCreateValuesNestedSingle(CreateValues):
    """Check the creation of heterogeneous arrays (nested, single row)"""
    _descr = Ndescr
    multiple_rows = 0
    _buffer = NbufferT[0]

class TestCreateValuesNestedMultiple(CreateValues):
    """Check the creation of heterogeneous arrays (nested, multiple rows)"""
    _descr = Ndescr
    multiple_rows = 1
    _buffer = NbufferT


############################################################
#    Reading tests
############################################################

class ReadValuesPlain:
    """Check the reading of values in heterogeneous arrays (plain)"""

    def test_access_fields(self):
        h = np.array(self._buffer, dtype=self._descr)
        if not self.multiple_rows:
            assert_(h.shape == ())
            assert_equal(h['x'], np.array(self._buffer[0], dtype='i4'))
            assert_equal(h['y'], np.array(self._buffer[1], dtype='f8'))
            assert_equal(h['z'], np.array(self._buffer[2], dtype='u1'))
        else:
            assert_(len(h) == 2)
            assert_equal(h['x'], np.array([self._buffer[0][0],
                                             self._buffer[1][0]], dtype='i4'))
            assert_equal(h['y'], np.array([self._buffer[0][1],
                                             self._buffer[1][1]], dtype='f8'))
            assert_equal(h['z'], np.array([self._buffer[0][2],
                                             self._buffer[1][2]], dtype='u1'))


class TestReadValuesPlainSingle(ReadValuesPlain):
    """Check the creation of heterogeneous arrays (plain, single row)"""
    _descr = Pdescr
    multiple_rows = 0
    _buffer = PbufferT[0]

class TestReadValuesPlainMultiple(ReadValuesPlain):
    """Check the values of heterogeneous arrays (plain, multiple rows)"""
    _descr = Pdescr
    multiple_rows = 1
    _buffer = PbufferT

class ReadValuesNested:
    """Check the reading of values in heterogeneous arrays (nested)"""

    def test_access_top_fields(self):
        """Check reading the top fields of a nested array"""
        h = np.array(self._buffer, dtype=self._descr)
        if not self.multiple_rows:
            assert_(h.shape == ())
            assert_equal(h['x'], np.array(self._buffer[0], dtype='i4'))
            assert_equal(h['y'], np.array(self._buffer[4], dtype='f8'))
            assert_equal(h['z'], np.array(self._buffer[5], dtype='u1'))
        else:
            assert_(len(h) == 2)
            assert_equal(h['x'], np.array([self._buffer[0][0],
                                           self._buffer[1][0]], dtype='i4'))
            assert_equal(h['y'], np.array([self._buffer[0][4],
                                           self._buffer[1][4]], dtype='f8'))
            assert_equal(h['z'], np.array([self._buffer[0][5],
                                           self._buffer[1][5]], dtype='u1'))

    def test_nested1_acessors(self):
        """Check reading the nested fields of a nested array (1st level)"""
        h = np.array(self._buffer, dtype=self._descr)
        if not self.multiple_rows:
            assert_equal(h['Info']['value'],
                         np.array(self._buffer[1][0], dtype='c16'))
            assert_equal(h['Info']['y2'],
                         np.array(self._buffer[1][1], dtype='f8'))
            assert_equal(h['info']['Name'],
                         np.array(self._buffer[3][0], dtype='U2'))
            assert_equal(h['info']['Value'],
                         np.array(self._buffer[3][1], dtype='c16'))
        else:
            assert_equal(h['Info']['value'],
                         np.array([self._buffer[0][1][0],
                                self._buffer[1][1][0]],
                                dtype='c16'))
            assert_equal(h['Info']['y2'],
                         np.array([self._buffer[0][1][1],
                                self._buffer[1][1][1]],
                                dtype='f8'))
            assert_equal(h['info']['Name'],
                         np.array([self._buffer[0][3][0],
                                self._buffer[1][3][0]],
                               dtype='U2'))
            assert_equal(h['info']['Value'],
                         np.array([self._buffer[0][3][1],
                                self._buffer[1][3][1]],
                               dtype='c16'))

    def test_nested2_acessors(self):
        """Check reading the nested fields of a nested array (2nd level)"""
        h = np.array(self._buffer, dtype=self._descr)
        if not self.multiple_rows:
            assert_equal(h['Info']['Info2']['value'],
                         np.array(self._buffer[1][2][1], dtype='c16'))
            assert_equal(h['Info']['Info2']['z3'],
                         np.array(self._buffer[1][2][3], dtype='u4'))
        else:
            assert_equal(h['Info']['Info2']['value'],
                         np.array([self._buffer[0][1][2][1],
                                self._buffer[1][1][2][1]],
                               dtype='c16'))
            assert_equal(h['Info']['Info2']['z3'],
                         np.array([self._buffer[0][1][2][3],
                                self._buffer[1][1][2][3]],
                               dtype='u4'))

    def test_nested1_descriptor(self):
        """Check access nested descriptors of a nested array (1st level)"""
        h = np.array(self._buffer, dtype=self._descr)
        assert_(h.dtype['Info']['value'].name == 'complex128')
        assert_(h.dtype['Info']['y2'].name == 'float64')
        assert_(h.dtype['info']['Name'].name == 'str256')
        assert_(h.dtype['info']['Value'].name == 'complex128')

    def test_nested2_descriptor(self):
        """Check access nested descriptors of a nested array (2nd level)"""
        h = np.array(self._buffer, dtype=self._descr)
        assert_(h.dtype['Info']['Info2']['value'].name == 'void256')
        assert_(h.dtype['Info']['Info2']['z3'].name == 'void64')


class TestReadValuesNestedSingle(ReadValuesNested):
    """Check the values of heterogeneous arrays (nested, single row)"""
    _descr = Ndescr
    multiple_rows = False
    _buffer = NbufferT[0]

class TestReadValuesNestedMultiple(ReadValuesNested):
    """Check the values of heterogeneous arrays (nested, multiple rows)"""
    _descr = Ndescr
    multiple_rows = True
    _buffer = NbufferT

class TestEmptyField:
    def test_assign(self):
        a = np.arange(10, dtype=np.float32)
        a.dtype = [("int",   "<0i4"), ("float", "<2f4")]
        assert_(a['int'].shape == (5, 0))
        assert_(a['float'].shape == (5, 2))


class TestMultipleFields:
    def setup_method(self):
        self.ary = np.array([(1, 2, 3, 4), (5, 6, 7, 8)], dtype='i4,f4,i2,c8')

    def _bad_call(self):
        return self.ary['f0', 'f1']

    def test_no_tuple(self):
        assert_raises(IndexError, self._bad_call)

    def test_return(self):
        res = self.ary[['f0', 'f2']].tolist()
        assert_(res == [(1, 3), (5, 7)])


class TestIsSubDType:
    # scalar types can be promoted into dtypes
    wrappers = [np.dtype, lambda x: x]

    def test_both_abstract(self):
        assert_(np.issubdtype(np.floating, np.inexact))
        assert_(not np.issubdtype(np.inexact, np.floating))

    def test_same(self):
        for cls in (np.float32, np.int32):
            for w1, w2 in itertools.product(self.wrappers, repeat=2):
                assert_(np.issubdtype(w1(cls), w2(cls)))

    def test_subclass(self):
        # note we cannot promote floating to a dtype, as it would turn into a
        # concrete type
        for w in self.wrappers:
            assert_(np.issubdtype(w(np.float32), np.floating))
            assert_(np.issubdtype(w(np.float64), np.floating))

    def test_subclass_backwards(self):
        for w in self.wrappers:
            assert_(not np.issubdtype(np.floating, w(np.float32)))
            assert_(not np.issubdtype(np.floating, w(np.float64)))

    def test_sibling_class(self):
        for w1, w2 in itertools.product(self.wrappers, repeat=2):
            assert_(not np.issubdtype(w1(np.float32), w2(np.float64)))
            assert_(not np.issubdtype(w1(np.float64), w2(np.float32)))

    def test_nondtype_nonscalartype(self):
        # See gh-14619 and gh-9505 which introduced the deprecation to fix
        # this. These tests are directly taken from gh-9505
        assert not np.issubdtype(np.float32, 'float64')
        assert not np.issubdtype(np.float32, 'f8')
        assert not np.issubdtype(np.int32, str)
        assert not np.issubdtype(np.int32, 'int64')
        assert not np.issubdtype(np.str_, 'void')
        # for the following the correct spellings are
        # np.integer, np.floating, or np.complexfloating respectively:
        assert not np.issubdtype(np.int8, int)  # np.int8 is never np.int_
        assert not np.issubdtype(np.float32, float)
        assert not np.issubdtype(np.complex64, complex)
        assert not np.issubdtype(np.float32, "float")
        assert not np.issubdtype(np.float64, "f")

        # Test the same for the correct first datatype and abstract one
        # in the case of int, float, complex:
        assert np.issubdtype(np.float64, 'float64')
        assert np.issubdtype(np.float64, 'f8')
        assert np.issubdtype(np.str_, str)
        assert np.issubdtype(np.int64, 'int64')
        assert np.issubdtype(np.void, 'void')
        assert np.issubdtype(np.int8, np.integer)
        assert np.issubdtype(np.float32, np.floating)
        assert np.issubdtype(np.complex64, np.complexfloating)
        assert np.issubdtype(np.float64, "float")
        assert np.issubdtype(np.float32, "f")


class TestIsDType:
    """
    Check correctness of `np.isdtype`. The test considers different argument
    configurations: `np.isdtype(dtype, k1)` and `np.isdtype(dtype, (k1, k2))`
    with concrete dtypes and dtype groups.
    """
    dtype_group_dict = {
        "signed integer": sctypes["int"],
        "unsigned integer": sctypes["uint"],
        "integral": sctypes["int"] + sctypes["uint"],
        "real floating": sctypes["float"],
        "complex floating": sctypes["complex"],
        "numeric": (
            sctypes["int"] + sctypes["uint"] + sctypes["float"] +
            sctypes["complex"]
        )
    }

    @pytest.mark.parametrize(
        "dtype,close_dtype",
        [
            (np.int64, np.int32), (np.uint64, np.uint32),
            (np.float64, np.float32), (np.complex128, np.complex64)
        ]
    )
    @pytest.mark.parametrize(
        "dtype_group",
        [
            None, "signed integer", "unsigned integer", "integral",
            "real floating", "complex floating", "numeric"
        ]
    )
    def test_isdtype(self, dtype, close_dtype, dtype_group):
        # First check if same dtypes return `true` and different ones
        # give `false` (even if they're close in the dtype hierarchy!)
        if dtype_group is None:
            assert np.isdtype(dtype, dtype)
            assert not np.isdtype(dtype, close_dtype)
            assert np.isdtype(dtype, (dtype, close_dtype))

        # Check that dtype and a dtype group that it belongs to
        # return `true`, and `false` otherwise.
        elif dtype in self.dtype_group_dict[dtype_group]:
            assert np.isdtype(dtype, dtype_group)
            assert np.isdtype(dtype, (close_dtype, dtype_group))
        else:
            assert not np.isdtype(dtype, dtype_group)

    def test_isdtype_invalid_args(self):
        with assert_raises_regex(TypeError, r".*must be a NumPy dtype.*"):
            np.isdtype("int64", np.int64)
        with assert_raises_regex(TypeError, r".*kind argument must.*"):
            np.isdtype(np.int64, 1)
        with assert_raises_regex(ValueError, r".*not a known kind name.*"):
            np.isdtype(np.int64, "int64")

    def test_sctypes_complete(self):
        # issue 26439: int32/intc were masking each other on 32-bit builds
        assert np.int32 in sctypes['int']
        assert np.intc in sctypes['int']
        assert np.int64 in sctypes['int']
        assert np.uint32 in sctypes['uint']
        assert np.uintc in sctypes['uint']
        assert np.uint64 in sctypes['uint']

class TestSctypeDict:
    def test_longdouble(self):
        assert_(np._core.sctypeDict['float64'] is not np.longdouble)
        assert_(np._core.sctypeDict['complex128'] is not np.clongdouble)

    def test_ulong(self):
        assert np._core.sctypeDict['ulong'] is np.ulong
        assert np.dtype(np.ulong) is np.dtype("ulong")
        assert np.dtype(np.ulong).itemsize == np.dtype(np.long).itemsize


@pytest.mark.filterwarnings("ignore:.*maximum_sctype.*:DeprecationWarning")
class TestMaximumSctype:

    # note that parametrizing with sctype['int'] and similar would skip types
    # with the same size (gh-11923)

    @pytest.mark.parametrize(
        't', [np.byte, np.short, np.intc, np.long, np.longlong]
    )
    def test_int(self, t):
        assert_equal(maximum_sctype(t), np._core.sctypes['int'][-1])

    @pytest.mark.parametrize(
        't', [np.ubyte, np.ushort, np.uintc, np.ulong, np.ulonglong]
    )
    def test_uint(self, t):
        assert_equal(maximum_sctype(t), np._core.sctypes['uint'][-1])

    @pytest.mark.parametrize('t', [np.half, np.single, np.double, np.longdouble])
    def test_float(self, t):
        assert_equal(maximum_sctype(t), np._core.sctypes['float'][-1])

    @pytest.mark.parametrize('t', [np.csingle, np.cdouble, np.clongdouble])
    def test_complex(self, t):
        assert_equal(maximum_sctype(t), np._core.sctypes['complex'][-1])

    @pytest.mark.parametrize('t', [np.bool, np.object_, np.str_, np.bytes_,
                                   np.void])
    def test_other(self, t):
        assert_equal(maximum_sctype(t), t)


class Test_sctype2char:
    # This function is old enough that we're really just documenting the quirks
    # at this point.

    def test_scalar_type(self):
        assert_equal(sctype2char(np.double), 'd')
        assert_equal(sctype2char(np.long), 'l')
        assert_equal(sctype2char(np.int_), np.array(0).dtype.char)
        assert_equal(sctype2char(np.str_), 'U')
        assert_equal(sctype2char(np.bytes_), 'S')

    def test_other_type(self):
        assert_equal(sctype2char(float), 'd')
        assert_equal(sctype2char(list), 'O')
        assert_equal(sctype2char(np.ndarray), 'O')

    def test_third_party_scalar_type(self):
        from numpy._core._rational_tests import rational
        assert_raises(KeyError, sctype2char, rational)
        assert_raises(KeyError, sctype2char, rational(1))

    def test_array_instance(self):
        assert_equal(sctype2char(np.array([1.0, 2.0])), 'd')

    def test_abstract_type(self):
        assert_raises(KeyError, sctype2char, np.floating)

    def test_non_type(self):
        assert_raises(ValueError, sctype2char, 1)

@pytest.mark.parametrize("rep, expected", [
    (np.int32, True),
    (list, False),
    (1.1, False),
    (str, True),
    (np.dtype(np.float64), True),
    (np.dtype((np.int16, (3, 4))), True),
    (np.dtype([('a', np.int8)]), True),
    ])
def test_issctype(rep, expected):
    # ensure proper identification of scalar
    # data-types by issctype()
    actual = issctype(rep)
    assert type(actual) is bool
    assert_equal(actual, expected)


@pytest.mark.skipif(sys.flags.optimize > 1,
                    reason="no docstrings present to inspect when PYTHONOPTIMIZE/Py_OptimizeFlag > 1")
@pytest.mark.xfail(IS_PYPY,
                   reason="PyPy cannot modify tp_doc after PyType_Ready")
class TestDocStrings:
    def test_platform_dependent_aliases(self):
        if np.int64 is np.int_:
            assert_('int64' in np.int_.__doc__)
        elif np.int64 is np.longlong:
            assert_('int64' in np.longlong.__doc__)


class TestScalarTypeNames:
    # gh-9799

    numeric_types = [
        np.byte, np.short, np.intc, np.long, np.longlong,
        np.ubyte, np.ushort, np.uintc, np.ulong, np.ulonglong,
        np.half, np.single, np.double, np.longdouble,
        np.csingle, np.cdouble, np.clongdouble,
    ]

    def test_names_are_unique(self):
        # none of the above may be aliases for each other
        assert len(set(self.numeric_types)) == len(self.numeric_types)

        # names must be unique
        names = [t.__name__ for t in self.numeric_types]
        assert len(set(names)) == len(names)

    @pytest.mark.parametrize('t', numeric_types)
    def test_names_reflect_attributes(self, t):
        """ Test that names correspond to where the type is under ``np.`` """
        assert getattr(np, t.__name__) is t

    @pytest.mark.parametrize('t', numeric_types)
    def test_names_are_undersood_by_dtype(self, t):
        """ Test the dtype constructor maps names back to the type """
        assert np.dtype(t.__name__).type is t


class TestBoolDefinition:
    def test_bool_definition(self):
        assert nt.bool is np.bool

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\ranges\test_range.py ===
import numpy as np
import pytest

from pandas.core.dtypes.common import ensure_platform_int

import pandas as pd
from pandas import (
    Index,
    RangeIndex,
)
import pandas._testing as tm


class TestRangeIndex:
    @pytest.fixture
    def simple_index(self):
        return RangeIndex(start=0, stop=20, step=2)

    def test_constructor_unwraps_index(self):
        result = RangeIndex(1, 3)
        expected = np.array([1, 2], dtype=np.int64)
        tm.assert_numpy_array_equal(result._data, expected)

    def test_can_hold_identifiers(self, simple_index):
        idx = simple_index
        key = idx[0]
        assert idx._can_hold_identifiers_and_holds_name(key) is False

    def test_too_many_names(self, simple_index):
        index = simple_index
        with pytest.raises(ValueError, match="^Length"):
            index.names = ["roger", "harold"]

    @pytest.mark.parametrize(
        "index, start, stop, step",
        [
            (RangeIndex(5), 0, 5, 1),
            (RangeIndex(0, 5), 0, 5, 1),
            (RangeIndex(5, step=2), 0, 5, 2),
            (RangeIndex(1, 5, 2), 1, 5, 2),
        ],
    )
    def test_start_stop_step_attrs(self, index, start, stop, step):
        # GH 25710
        assert index.start == start
        assert index.stop == stop
        assert index.step == step

    def test_copy(self):
        i = RangeIndex(5, name="Foo")
        i_copy = i.copy()
        assert i_copy is not i
        assert i_copy.identical(i)
        assert i_copy._range == range(0, 5, 1)
        assert i_copy.name == "Foo"

    def test_repr(self):
        i = RangeIndex(5, name="Foo")
        result = repr(i)
        expected = "RangeIndex(start=0, stop=5, step=1, name='Foo')"
        assert result == expected

        result = eval(result)
        tm.assert_index_equal(result, i, exact=True)

        i = RangeIndex(5, 0, -1)
        result = repr(i)
        expected = "RangeIndex(start=5, stop=0, step=-1)"
        assert result == expected

        result = eval(result)
        tm.assert_index_equal(result, i, exact=True)

    def test_insert(self):
        idx = RangeIndex(5, name="Foo")
        result = idx[1:4]

        # test 0th element
        tm.assert_index_equal(idx[0:4], result.insert(0, idx[0]), exact="equiv")

        # GH 18295 (test missing)
        expected = Index([0, np.nan, 1, 2, 3, 4], dtype=np.float64)
        for na in [np.nan, None, pd.NA]:
            result = RangeIndex(5).insert(1, na)
            tm.assert_index_equal(result, expected)

        result = RangeIndex(5).insert(1, pd.NaT)
        expected = Index([0, pd.NaT, 1, 2, 3, 4], dtype=object)
        tm.assert_index_equal(result, expected)

    def test_insert_edges_preserves_rangeindex(self):
        idx = Index(range(4, 9, 2))

        result = idx.insert(0, 2)
        expected = Index(range(2, 9, 2))
        tm.assert_index_equal(result, expected, exact=True)

        result = idx.insert(3, 10)
        expected = Index(range(4, 11, 2))
        tm.assert_index_equal(result, expected, exact=True)

    def test_insert_middle_preserves_rangeindex(self):
        # insert in the middle
        idx = Index(range(0, 3, 2))
        result = idx.insert(1, 1)
        expected = Index(range(3))
        tm.assert_index_equal(result, expected, exact=True)

        idx = idx * 2
        result = idx.insert(1, 2)
        expected = expected * 2
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete(self):
        idx = RangeIndex(5, name="Foo")
        expected = idx[1:]
        result = idx.delete(0)
        tm.assert_index_equal(result, expected, exact=True)
        assert result.name == expected.name

        expected = idx[:-1]
        result = idx.delete(-1)
        tm.assert_index_equal(result, expected, exact=True)
        assert result.name == expected.name

        msg = "index 5 is out of bounds for axis 0 with size 5"
        with pytest.raises((IndexError, ValueError), match=msg):
            # either depending on numpy version
            result = idx.delete(len(idx))

    def test_delete_preserves_rangeindex(self):
        idx = Index(range(2), name="foo")

        result = idx.delete([1])
        expected = Index(range(1), name="foo")
        tm.assert_index_equal(result, expected, exact=True)

        result = idx.delete(1)
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete_preserves_rangeindex_middle(self):
        idx = Index(range(3), name="foo")
        result = idx.delete(1)
        expected = idx[::2]
        tm.assert_index_equal(result, expected, exact=True)

        result = idx.delete(-2)
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete_preserves_rangeindex_list_at_end(self):
        idx = RangeIndex(0, 6, 1)

        loc = [2, 3, 4, 5]
        result = idx.delete(loc)
        expected = idx[:2]
        tm.assert_index_equal(result, expected, exact=True)

        result = idx.delete(loc[::-1])
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete_preserves_rangeindex_list_middle(self):
        idx = RangeIndex(0, 6, 1)

        loc = [1, 2, 3, 4]
        result = idx.delete(loc)
        expected = RangeIndex(0, 6, 5)
        tm.assert_index_equal(result, expected, exact=True)

        result = idx.delete(loc[::-1])
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete_all_preserves_rangeindex(self):
        idx = RangeIndex(0, 6, 1)

        loc = [0, 1, 2, 3, 4, 5]
        result = idx.delete(loc)
        expected = idx[:0]
        tm.assert_index_equal(result, expected, exact=True)

        result = idx.delete(loc[::-1])
        tm.assert_index_equal(result, expected, exact=True)

    def test_delete_not_preserving_rangeindex(self):
        idx = RangeIndex(0, 6, 1)

        loc = [0, 3, 5]
        result = idx.delete(loc)
        expected = Index([1, 2, 4])
        tm.assert_index_equal(result, expected, exact=True)

        result = idx.delete(loc[::-1])
        tm.assert_index_equal(result, expected, exact=True)

    def test_view(self):
        i = RangeIndex(0, name="Foo")
        i_view = i.view()
        assert i_view.name == "Foo"

        i_view = i.view("i8")
        tm.assert_numpy_array_equal(i.values, i_view)

        msg = "Passing a type in RangeIndex.view is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            i_view = i.view(RangeIndex)
        tm.assert_index_equal(i, i_view)

    def test_dtype(self, simple_index):
        index = simple_index
        assert index.dtype == np.int64

    def test_cache(self):
        # GH 26565, GH26617, GH35432, GH53387
        # This test checks whether _cache has been set.
        # Calling RangeIndex._cache["_data"] creates an int64 array of the same length
        # as the RangeIndex and stores it in _cache.
        idx = RangeIndex(0, 100, 10)

        assert idx._cache == {}

        repr(idx)
        assert idx._cache == {}

        str(idx)
        assert idx._cache == {}

        idx.get_loc(20)
        assert idx._cache == {}

        90 in idx  # True
        assert idx._cache == {}

        91 in idx  # False
        assert idx._cache == {}

        idx.all()
        assert idx._cache == {}

        idx.any()
        assert idx._cache == {}

        for _ in idx:
            pass
        assert idx._cache == {}

        msg = "RangeIndex.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            idx.format()
        assert idx._cache == {}

        df = pd.DataFrame({"a": range(10)}, index=idx)

        # df.__repr__ should not populate index cache
        str(df)
        assert idx._cache == {}

        df.loc[50]
        assert idx._cache == {}

        with pytest.raises(KeyError, match="51"):
            df.loc[51]
        assert idx._cache == {}

        df.loc[10:50]
        assert idx._cache == {}

        df.iloc[5:10]
        assert idx._cache == {}

        # after calling take, _cache may contain other keys, but not "_data"
        idx.take([3, 0, 1])
        assert "_data" not in idx._cache

        df.loc[[50]]
        assert "_data" not in idx._cache

        df.iloc[[5, 6, 7, 8, 9]]
        assert "_data" not in idx._cache

        # idx._cache should contain a _data entry after call to idx._data
        idx._data
        assert isinstance(idx._data, np.ndarray)
        assert idx._data is idx._data  # check cached value is reused
        assert "_data" in idx._cache
        expected = np.arange(0, 100, 10, dtype="int64")
        tm.assert_numpy_array_equal(idx._cache["_data"], expected)

    def test_is_monotonic(self):
        index = RangeIndex(0, 20, 2)
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is False
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is False

        index = RangeIndex(4, 0, -1)
        assert index.is_monotonic_increasing is False
        assert index._is_strictly_monotonic_increasing is False
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_decreasing is True

        index = RangeIndex(1, 2)
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True

        index = RangeIndex(2, 1)
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True

        index = RangeIndex(1, 1)
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True

    @pytest.mark.parametrize(
        "left,right",
        [
            (RangeIndex(0, 9, 2), RangeIndex(0, 10, 2)),
            (RangeIndex(0), RangeIndex(1, -1, 3)),
            (RangeIndex(1, 2, 3), RangeIndex(1, 3, 4)),
            (RangeIndex(0, -9, -2), RangeIndex(0, -10, -2)),
        ],
    )
    def test_equals_range(self, left, right):
        assert left.equals(right)
        assert right.equals(left)

    def test_logical_compat(self, simple_index):
        idx = simple_index
        assert idx.all() == idx.values.all()
        assert idx.any() == idx.values.any()

    def test_identical(self, simple_index):
        index = simple_index
        i = Index(index.copy())
        assert i.identical(index)

        # we don't allow object dtype for RangeIndex
        if isinstance(index, RangeIndex):
            return

        same_values_different_type = Index(i, dtype=object)
        assert not i.identical(same_values_different_type)

        i = index.copy(dtype=object)
        i = i.rename("foo")
        same_values = Index(i, dtype=object)
        assert same_values.identical(index.copy(dtype=object))

        assert not i.identical(index)
        assert Index(same_values, name="foo", dtype=object).identical(i)

        assert not index.copy(dtype=object).identical(index.copy(dtype="int64"))

    def test_nbytes(self):
        # memory savings vs int index
        idx = RangeIndex(0, 1000)
        assert idx.nbytes < Index(idx._values).nbytes / 10

        # constant memory usage
        i2 = RangeIndex(0, 10)
        assert idx.nbytes == i2.nbytes

    @pytest.mark.parametrize(
        "start,stop,step",
        [
            # can't
            ("foo", "bar", "baz"),
            # shouldn't
            ("0", "1", "2"),
        ],
    )
    def test_cant_or_shouldnt_cast(self, start, stop, step):
        msg = f"Wrong type {type(start)} for value {start}"
        with pytest.raises(TypeError, match=msg):
            RangeIndex(start, stop, step)

    def test_view_index(self, simple_index):
        index = simple_index
        msg = "Passing a type in RangeIndex.view is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            index.view(Index)

    def test_prevent_casting(self, simple_index):
        index = simple_index
        result = index.astype("O")
        assert result.dtype == np.object_

    def test_repr_roundtrip(self, simple_index):
        index = simple_index
        tm.assert_index_equal(eval(repr(index)), index)

    def test_slice_keep_name(self):
        idx = RangeIndex(1, 2, name="asdf")
        assert idx.name == idx[1:].name

    @pytest.mark.parametrize(
        "index",
        [
            RangeIndex(start=0, stop=20, step=2, name="foo"),
            RangeIndex(start=18, stop=-1, step=-2, name="bar"),
        ],
        ids=["index_inc", "index_dec"],
    )
    def test_has_duplicates(self, index):
        assert index.is_unique
        assert not index.has_duplicates

    def test_extended_gcd(self, simple_index):
        index = simple_index
        result = index._extended_gcd(6, 10)
        assert result[0] == result[1] * 6 + result[2] * 10
        assert 2 == result[0]

        result = index._extended_gcd(10, 6)
        assert 2 == result[1] * 10 + result[2] * 6
        assert 2 == result[0]

    def test_min_fitting_element(self):
        result = RangeIndex(0, 20, 2)._min_fitting_element(1)
        assert 2 == result

        result = RangeIndex(1, 6)._min_fitting_element(1)
        assert 1 == result

        result = RangeIndex(18, -2, -2)._min_fitting_element(1)
        assert 2 == result

        result = RangeIndex(5, 0, -1)._min_fitting_element(1)
        assert 1 == result

        big_num = 500000000000000000000000

        result = RangeIndex(5, big_num * 2, 1)._min_fitting_element(big_num)
        assert big_num == result

    def test_slice_specialised(self, simple_index):
        index = simple_index
        index.name = "foo"

        # scalar indexing
        res = index[1]
        expected = 2
        assert res == expected

        res = index[-1]
        expected = 18
        assert res == expected

        # slicing
        # slice value completion
        index_slice = index[:]
        expected = index
        tm.assert_index_equal(index_slice, expected)

        # positive slice values
        index_slice = index[7:10:2]
        expected = Index([14, 18], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        # negative slice values
        index_slice = index[-1:-5:-2]
        expected = Index([18, 14], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        # stop overshoot
        index_slice = index[2:100:4]
        expected = Index([4, 12], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        # reverse
        index_slice = index[::-1]
        expected = Index(index.values[::-1], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        index_slice = index[-8::-1]
        expected = Index([4, 2, 0], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        index_slice = index[-40::-1]
        expected = Index(np.array([], dtype=np.int64), name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        index_slice = index[40::-1]
        expected = Index(index.values[40::-1], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

        index_slice = index[10::-1]
        expected = Index(index.values[::-1], name="foo")
        tm.assert_index_equal(index_slice, expected, exact="equiv")

    @pytest.mark.parametrize("step", set(range(-5, 6)) - {0})
    def test_len_specialised(self, step):
        # make sure that our len is the same as np.arange calc
        start, stop = (0, 5) if step > 0 else (5, 0)

        arr = np.arange(start, stop, step)
        index = RangeIndex(start, stop, step)
        assert len(index) == len(arr)

        index = RangeIndex(stop, start, step)
        assert len(index) == 0

    @pytest.mark.parametrize(
        "indices, expected",
        [
            ([RangeIndex(1, 12, 5)], RangeIndex(1, 12, 5)),
            ([RangeIndex(0, 6, 4)], RangeIndex(0, 6, 4)),
            ([RangeIndex(1, 3), RangeIndex(3, 7)], RangeIndex(1, 7)),
            ([RangeIndex(1, 5, 2), RangeIndex(5, 6)], RangeIndex(1, 6, 2)),
            ([RangeIndex(1, 3, 2), RangeIndex(4, 7, 3)], RangeIndex(1, 7, 3)),
            ([RangeIndex(-4, 3, 2), RangeIndex(4, 7, 2)], RangeIndex(-4, 7, 2)),
            ([RangeIndex(-4, -8), RangeIndex(-8, -12)], RangeIndex(0, 0)),
            ([RangeIndex(-4, -8), RangeIndex(3, -4)], RangeIndex(0, 0)),
            ([RangeIndex(-4, -8), RangeIndex(3, 5)], RangeIndex(3, 5)),
            ([RangeIndex(-4, -2), RangeIndex(3, 5)], Index([-4, -3, 3, 4])),
            ([RangeIndex(-2), RangeIndex(3, 5)], RangeIndex(3, 5)),
            ([RangeIndex(2), RangeIndex(2)], Index([0, 1, 0, 1])),
            ([RangeIndex(2), RangeIndex(2, 5), RangeIndex(5, 8, 4)], RangeIndex(0, 6)),
            (
                [RangeIndex(2), RangeIndex(3, 5), RangeIndex(5, 8, 4)],
                Index([0, 1, 3, 4, 5]),
            ),
            (
                [RangeIndex(-2, 2), RangeIndex(2, 5), RangeIndex(5, 8, 4)],
                RangeIndex(-2, 6),
            ),
            ([RangeIndex(3), Index([-1, 3, 15])], Index([0, 1, 2, -1, 3, 15])),
            ([RangeIndex(3), Index([-1, 3.1, 15.0])], Index([0, 1, 2, -1, 3.1, 15.0])),
            ([RangeIndex(3), Index(["a", None, 14])], Index([0, 1, 2, "a", None, 14])),
            ([RangeIndex(3, 1), Index(["a", None, 14])], Index(["a", None, 14])),
        ],
    )
    def test_append(self, indices, expected):
        # GH16212
        result = indices[0].append(indices[1:])
        tm.assert_index_equal(result, expected, exact=True)

        if len(indices) == 2:
            # Append single item rather than list
            result2 = indices[0].append(indices[1])
            tm.assert_index_equal(result2, expected, exact=True)

    def test_engineless_lookup(self):
        # GH 16685
        # Standard lookup on RangeIndex should not require the engine to be
        # created
        idx = RangeIndex(2, 10, 3)

        assert idx.get_loc(5) == 1
        tm.assert_numpy_array_equal(
            idx.get_indexer([2, 8]), ensure_platform_int(np.array([0, 2]))
        )
        with pytest.raises(KeyError, match="3"):
            idx.get_loc(3)

        assert "_engine" not in idx._cache

        # Different types of scalars can be excluded immediately, no need to
        #  use the _engine
        with pytest.raises(KeyError, match="'a'"):
            idx.get_loc("a")

        assert "_engine" not in idx._cache

    def test_format_empty(self):
        # GH35712
        empty_idx = RangeIndex(0)
        msg = r"RangeIndex\.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert empty_idx.format() == []
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert empty_idx.format(name=True) == [""]

    @pytest.mark.parametrize(
        "ri",
        [
            RangeIndex(0, -1, -1),
            RangeIndex(0, 1, 1),
            RangeIndex(1, 3, 2),
            RangeIndex(0, -1, -2),
            RangeIndex(-3, -5, -2),
        ],
    )
    def test_append_len_one(self, ri):
        # GH39401
        result = ri.append([])
        tm.assert_index_equal(result, ri, exact=True)

    @pytest.mark.parametrize("base", [RangeIndex(0, 2), Index([0, 1])])
    def test_isin_range(self, base):
        # GH#41151
        values = RangeIndex(0, 1)
        result = base.isin(values)
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_sort_values_key(self):
        # GH#43666, GH#52764
        sort_order = {8: 2, 6: 0, 4: 8, 2: 10, 0: 12}
        values = RangeIndex(0, 10, 2)
        result = values.sort_values(key=lambda x: x.map(sort_order))
        expected = Index([6, 8, 4, 2, 0], dtype="int64")
        tm.assert_index_equal(result, expected, check_exact=True)

        # check this matches the Series.sort_values behavior
        ser = values.to_series()
        result2 = ser.sort_values(key=lambda x: x.map(sort_order))
        tm.assert_series_equal(result2, expected.to_series(), check_exact=True)

    def test_range_index_rsub_by_const(self):
        # GH#53255
        result = 3 - RangeIndex(0, 4, 1)
        expected = RangeIndex(3, -1, -1)
        tm.assert_index_equal(result, expected)