
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\test_format.py ===
"""
Tests for the file pandas.io.formats.format, *not* tests for general formatting
of pandas objects.
"""
from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size

import numpy as np
import pytest

from pandas._config import using_string_dtype

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    NaT,
    Series,
    Timestamp,
    date_range,
    get_option,
    option_context,
    read_csv,
    reset_option,
)

from pandas.io.formats import printing
import pandas.io.formats.format as fmt


@pytest.fixture(params=["string", "pathlike", "buffer"])
def filepath_or_buffer_id(request):
    """
    A fixture yielding test ids for filepath_or_buffer testing.
    """
    return request.param


@pytest.fixture
def filepath_or_buffer(filepath_or_buffer_id, tmp_path):
    """
    A fixture yielding a string representing a filepath, a path-like object
    and a StringIO buffer. Also checks that buffer is not closed.
    """
    if filepath_or_buffer_id == "buffer":
        buf = StringIO()
        yield buf
        assert not buf.closed
    else:
        assert isinstance(tmp_path, Path)
        if filepath_or_buffer_id == "pathlike":
            yield tmp_path / "foo"
        else:
            yield str(tmp_path / "foo")


@pytest.fixture
def assert_filepath_or_buffer_equals(
    filepath_or_buffer, filepath_or_buffer_id, encoding
):
    """
    Assertion helper for checking filepath_or_buffer.
    """
    if encoding is None:
        encoding = "utf-8"

    def _assert_filepath_or_buffer_equals(expected):
        if filepath_or_buffer_id == "string":
            with open(filepath_or_buffer, encoding=encoding) as f:
                result = f.read()
        elif filepath_or_buffer_id == "pathlike":
            result = filepath_or_buffer.read_text(encoding=encoding)
        elif filepath_or_buffer_id == "buffer":
            result = filepath_or_buffer.getvalue()
        assert result == expected

    return _assert_filepath_or_buffer_equals


def has_info_repr(df):
    r = repr(df)
    c1 = r.split("\n")[0].startswith("<class")
    c2 = r.split("\n")[0].startswith(r"&lt;class")  # _repr_html_
    return c1 or c2


def has_non_verbose_info_repr(df):
    has_info = has_info_repr(df)
    r = repr(df)

    # 1. <class>
    # 2. Index
    # 3. Columns
    # 4. dtype
    # 5. memory usage
    # 6. trailing newline
    nv = len(r.split("\n")) == 6
    return has_info and nv


def has_horizontally_truncated_repr(df):
    try:  # Check header row
        fst_line = np.array(repr(df).splitlines()[0].split())
        cand_col = np.where(fst_line == "...")[0][0]
    except IndexError:
        return False
    # Make sure each row has this ... in the same place
    r = repr(df)
    for ix, _ in enumerate(r.splitlines()):
        if not r.split()[cand_col] == "...":
            return False
    return True


def has_vertically_truncated_repr(df):
    r = repr(df)
    only_dot_row = False
    for row in r.splitlines():
        if re.match(r"^[\.\ ]+$", row):
            only_dot_row = True
    return only_dot_row


def has_truncated_repr(df):
    return has_horizontally_truncated_repr(df) or has_vertically_truncated_repr(df)


def has_doubly_truncated_repr(df):
    return has_horizontally_truncated_repr(df) and has_vertically_truncated_repr(df)


def has_expanded_repr(df):
    r = repr(df)
    for line in r.split("\n"):
        if line.endswith("\\"):
            return True
    return False


class TestDataFrameFormatting:
    def test_repr_truncation(self):
        max_len = 20
        with option_context("display.max_colwidth", max_len):
            df = DataFrame(
                {
                    "A": np.random.default_rng(2).standard_normal(10),
                    "B": [
                        "a"
                        * np.random.default_rng(2).integers(max_len - 1, max_len + 1)
                        for _ in range(10)
                    ],
                }
            )
            r = repr(df)
            r = r[r.find("\n") + 1 :]

            adj = printing.get_adjustment()

            for line, value in zip(r.split("\n"), df["B"]):
                if adj.len(value) + 1 > max_len:
                    assert "..." in line
                else:
                    assert "..." not in line

        with option_context("display.max_colwidth", 999999):
            assert "..." not in repr(df)

        with option_context("display.max_colwidth", max_len + 2):
            assert "..." not in repr(df)

    def test_repr_truncation_preserves_na(self):
        # https://github.com/pandas-dev/pandas/issues/55630
        df = DataFrame({"a": [pd.NA for _ in range(10)]})
        with option_context("display.max_rows", 2, "display.show_dimensions", False):
            assert repr(df) == "       a\n0   <NA>\n..   ...\n9   <NA>"

    def test_max_colwidth_negative_int_raises(self):
        # Deprecation enforced from:
        # https://github.com/pandas-dev/pandas/issues/31532
        with pytest.raises(
            ValueError, match="Value must be a nonnegative integer or None"
        ):
            with option_context("display.max_colwidth", -1):
                pass

    def test_repr_chop_threshold(self):
        df = DataFrame([[0.1, 0.5], [0.5, -0.1]])
        reset_option("display.chop_threshold")  # default None
        assert repr(df) == "     0    1\n0  0.1  0.5\n1  0.5 -0.1"

        with option_context("display.chop_threshold", 0.2):
            assert repr(df) == "     0    1\n0  0.0  0.5\n1  0.5  0.0"

        with option_context("display.chop_threshold", 0.6):
            assert repr(df) == "     0    1\n0  0.0  0.0\n1  0.0  0.0"

        with option_context("display.chop_threshold", None):
            assert repr(df) == "     0    1\n0  0.1  0.5\n1  0.5 -0.1"

    def test_repr_chop_threshold_column_below(self):
        # GH 6839: validation case

        df = DataFrame([[10, 20, 30, 40], [8e-10, -1e-11, 2e-9, -2e-11]]).T

        with option_context("display.chop_threshold", 0):
            assert repr(df) == (
                "      0             1\n"
                "0  10.0  8.000000e-10\n"
                "1  20.0 -1.000000e-11\n"
                "2  30.0  2.000000e-09\n"
                "3  40.0 -2.000000e-11"
            )

        with option_context("display.chop_threshold", 1e-8):
            assert repr(df) == (
                "      0             1\n"
                "0  10.0  0.000000e+00\n"
                "1  20.0  0.000000e+00\n"
                "2  30.0  0.000000e+00\n"
                "3  40.0  0.000000e+00"
            )

        with option_context("display.chop_threshold", 5e-11):
            assert repr(df) == (
                "      0             1\n"
                "0  10.0  8.000000e-10\n"
                "1  20.0  0.000000e+00\n"
                "2  30.0  2.000000e-09\n"
                "3  40.0  0.000000e+00"
            )

    def test_repr_no_backslash(self):
        with option_context("mode.sim_interactive", True):
            df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
            assert "\\" not in repr(df)

    def test_expand_frame_repr(self):
        df_small = DataFrame("hello", index=[0], columns=[0])
        df_wide = DataFrame("hello", index=[0], columns=range(10))
        df_tall = DataFrame("hello", index=range(30), columns=range(5))

        with option_context("mode.sim_interactive", True):
            with option_context(
                "display.max_columns",
                10,
                "display.width",
                20,
                "display.max_rows",
                20,
                "display.show_dimensions",
                True,
            ):
                with option_context("display.expand_frame_repr", True):
                    assert not has_truncated_repr(df_small)
                    assert not has_expanded_repr(df_small)
                    assert not has_truncated_repr(df_wide)
                    assert has_expanded_repr(df_wide)
                    assert has_vertically_truncated_repr(df_tall)
                    assert has_expanded_repr(df_tall)

                with option_context("display.expand_frame_repr", False):
                    assert not has_truncated_repr(df_small)
                    assert not has_expanded_repr(df_small)
                    assert not has_horizontally_truncated_repr(df_wide)
                    assert not has_expanded_repr(df_wide)
                    assert has_vertically_truncated_repr(df_tall)
                    assert not has_expanded_repr(df_tall)

    def test_repr_non_interactive(self):
        # in non interactive mode, there can be no dependency on the
        # result of terminal auto size detection
        df = DataFrame("hello", index=range(1000), columns=range(5))

        with option_context(
            "mode.sim_interactive", False, "display.width", 0, "display.max_rows", 5000
        ):
            assert not has_truncated_repr(df)
            assert not has_expanded_repr(df)

    def test_repr_truncates_terminal_size(self, monkeypatch):
        # see gh-21180

        terminal_size = (118, 96)
        monkeypatch.setattr(
            "pandas.io.formats.format.get_terminal_size", lambda: terminal_size
        )

        index = range(5)
        columns = MultiIndex.from_tuples(
            [
                ("This is a long title with > 37 chars.", "cat"),
                ("This is a loooooonger title with > 43 chars.", "dog"),
            ]
        )
        df = DataFrame(1, index=index, columns=columns)

        result = repr(df)

        h1, h2 = result.split("\n")[:2]
        assert "long" in h1
        assert "loooooonger" in h1
        assert "cat" in h2
        assert "dog" in h2

        # regular columns
        df2 = DataFrame({"A" * 41: [1, 2], "B" * 41: [1, 2]})
        result = repr(df2)

        assert df2.columns[0] in result.split("\n")[0]

    def test_repr_truncates_terminal_size_full(self, monkeypatch):
        # GH 22984 ensure entire window is filled
        terminal_size = (80, 24)
        df = DataFrame(np.random.default_rng(2).random((1, 7)))

        monkeypatch.setattr(
            "pandas.io.formats.format.get_terminal_size", lambda: terminal_size
        )
        assert "..." not in str(df)

    def test_repr_truncation_column_size(self):
        # dataframe with last column very wide -> check it is not used to
        # determine size of truncation (...) column
        df = DataFrame(
            {
                "a": [108480, 30830],
                "b": [12345, 12345],
                "c": [12345, 12345],
                "d": [12345, 12345],
                "e": ["a" * 50] * 2,
            }
        )
        assert "..." in str(df)
        assert "    ...    " not in str(df)

    def test_repr_max_columns_max_rows(self):
        term_width, term_height = get_terminal_size()
        if term_width < 10 or term_height < 10:
            pytest.skip(f"terminal size too small, {term_width} x {term_height}")

        def mkframe(n):
            index = [f"{i:05d}" for i in range(n)]
            return DataFrame(0, index, index)

        df6 = mkframe(6)
        df10 = mkframe(10)
        with option_context("mode.sim_interactive", True):
            with option_context("display.width", term_width * 2):
                with option_context("display.max_rows", 5, "display.max_columns", 5):
                    assert not has_expanded_repr(mkframe(4))
                    assert not has_expanded_repr(mkframe(5))
                    assert not has_expanded_repr(df6)
                    assert has_doubly_truncated_repr(df6)

                with option_context("display.max_rows", 20, "display.max_columns", 10):
                    # Out off max_columns boundary, but no extending
                    # since not exceeding width
                    assert not has_expanded_repr(df6)
                    assert not has_truncated_repr(df6)

                with option_context("display.max_rows", 9, "display.max_columns", 10):
                    # out vertical bounds can not result in expanded repr
                    assert not has_expanded_repr(df10)
                    assert has_vertically_truncated_repr(df10)

            # width=None in terminal, auto detection
            with option_context(
                "display.max_columns",
                100,
                "display.max_rows",
                term_width * 20,
                "display.width",
                None,
            ):
                df = mkframe((term_width // 7) - 2)
                assert not has_expanded_repr(df)
                df = mkframe((term_width // 7) + 2)
                printing.pprint_thing(df._repr_fits_horizontal_())
                assert has_expanded_repr(df)

    def test_repr_min_rows(self):
        df = DataFrame({"a": range(20)})

        # default setting no truncation even if above min_rows
        assert ".." not in repr(df)
        assert ".." not in df._repr_html_()

        df = DataFrame({"a": range(61)})

        # default of max_rows 60 triggers truncation if above
        assert ".." in repr(df)
        assert ".." in df._repr_html_()

        with option_context("display.max_rows", 10, "display.min_rows", 4):
            # truncated after first two rows
            assert ".." in repr(df)
            assert "2  " not in repr(df)
            assert "..." in df._repr_html_()
            assert "<td>2</td>" not in df._repr_html_()

        with option_context("display.max_rows", 12, "display.min_rows", None):
            # when set to None, follow value of max_rows
            assert "5    5" in repr(df)
            assert "<td>5</td>" in df._repr_html_()

        with option_context("display.max_rows", 10, "display.min_rows", 12):
            # when set value higher as max_rows, use the minimum
            assert "5    5" not in repr(df)
            assert "<td>5</td>" not in df._repr_html_()

        with option_context("display.max_rows", None, "display.min_rows", 12):
            # max_rows of None -> never truncate
            assert ".." not in repr(df)
            assert ".." not in df._repr_html_()

    def test_str_max_colwidth(self):
        # GH 7856
        df = DataFrame(
            [
                {
                    "a": "foo",
                    "b": "bar",
                    "c": "uncomfortably long line with lots of stuff",
                    "d": 1,
                },
                {"a": "foo", "b": "bar", "c": "stuff", "d": 1},
            ]
        )
        df.set_index(["a", "b", "c"])
        assert str(df) == (
            "     a    b                                           c  d\n"
            "0  foo  bar  uncomfortably long line with lots of stuff  1\n"
            "1  foo  bar                                       stuff  1"
        )
        with option_context("max_colwidth", 20):
            assert str(df) == (
                "     a    b                    c  d\n"
                "0  foo  bar  uncomfortably lo...  1\n"
                "1  foo  bar                stuff  1"
            )

    def test_auto_detect(self):
        term_width, term_height = get_terminal_size()
        fac = 1.05  # Arbitrary large factor to exceed term width
        cols = range(int(term_width * fac))
        index = range(10)
        df = DataFrame(index=index, columns=cols)
        with option_context("mode.sim_interactive", True):
            with option_context("display.max_rows", None):
                with option_context("display.max_columns", None):
                    # Wrap around with None
                    assert has_expanded_repr(df)
            with option_context("display.max_rows", 0):
                with option_context("display.max_columns", 0):
                    # Truncate with auto detection.
                    assert has_horizontally_truncated_repr(df)

            index = range(int(term_height * fac))
            df = DataFrame(index=index, columns=cols)
            with option_context("display.max_rows", 0):
                with option_context("display.max_columns", None):
                    # Wrap around with None
                    assert has_expanded_repr(df)
                    # Truncate vertically
                    assert has_vertically_truncated_repr(df)

            with option_context("display.max_rows", None):
                with option_context("display.max_columns", 0):
                    assert has_horizontally_truncated_repr(df)

    def test_to_string_repr_unicode2(self):
        idx = Index(["abc", "\u03c3a", "aegdvg"])
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        rs = repr(ser).split("\n")
        line_len = len(rs[0])
        for line in rs[1:]:
            try:
                line = line.decode(get_option("display.encoding"))
            except AttributeError:
                pass
            if not line.startswith("dtype:"):
                assert len(line) == line_len

    def test_east_asian_unicode_false(self):
        # not aligned properly because of east asian width

        # mid col
        df = DataFrame(
            {"a": ["あ", "いいい", "う", "ええええええ"], "b": [1, 222, 33333, 4]},
            index=["a", "bb", "c", "ddd"],
        )
        expected = (
            "          a      b\na         あ      1\n"
            "bb      いいい    222\nc         う  33333\n"
            "ddd  ええええええ      4"
        )
        assert repr(df) == expected

        # last col
        df = DataFrame(
            {"a": [1, 222, 33333, 4], "b": ["あ", "いいい", "う", "ええええええ"]},
            index=["a", "bb", "c", "ddd"],
        )
        expected = (
            "         a       b\na        1       あ\n"
            "bb     222     いいい\nc    33333       う\n"
            "ddd      4  ええええええ"
        )
        assert repr(df) == expected

        # all col
        df = DataFrame(
            {
                "a": ["あああああ", "い", "う", "えええ"],
                "b": ["あ", "いいい", "う", "ええええええ"],
            },
            index=["a", "bb", "c", "ddd"],
        )
        expected = (
            "         a       b\na    あああああ       あ\n"
            "bb       い     いいい\nc        う       う\n"
            "ddd    えええ  ええええええ"
        )
        assert repr(df) == expected

        # column name
        df = DataFrame(
            {
                "b": ["あ", "いいい", "う", "ええええええ"],
                "あああああ": [1, 222, 33333, 4],
            },
            index=["a", "bb", "c", "ddd"],
        )
        expected = (
            "          b  あああああ\na         あ      1\n"
            "bb      いいい    222\nc         う  33333\n"
            "ddd  ええええええ      4"
        )
        assert repr(df) == expected

        # index
        df = DataFrame(
            {
                "a": ["あああああ", "い", "う", "えええ"],
                "b": ["あ", "いいい", "う", "ええええええ"],
            },
            index=["あああ", "いいいいいい", "うう", "え"],
        )
        expected = (
            "            a       b\nあああ     あああああ       あ\n"
            "いいいいいい      い     いいい\nうう          う       う\n"
            "え         えええ  ええええええ"
        )
        assert repr(df) == expected

        # index name
        df = DataFrame(
            {
                "a": ["あああああ", "い", "う", "えええ"],
                "b": ["あ", "いいい", "う", "ええええええ"],
            },
            index=Index(["あ", "い", "うう", "え"], name="おおおお"),
        )
        expected = (
            "          a       b\n"
            "おおおお               \n"
            "あ     あああああ       あ\n"
            "い         い     いいい\n"
            "うう        う       う\n"
            "え       えええ  ええええええ"
        )
        assert repr(df) == expected

        # all
        df = DataFrame(
            {
                "あああ": ["あああ", "い", "う", "えええええ"],
                "いいいいい": ["あ", "いいい", "う", "ええ"],
            },
            index=Index(["あ", "いいい", "うう", "え"], name="お"),
        )
        expected = (
            "       あああ いいいいい\n"
            "お               \n"
            "あ      あああ     あ\n"
            "いいい      い   いいい\n"
            "うう       う     う\n"
            "え    えええええ    ええ"
        )
        assert repr(df) == expected

        # MultiIndex
        idx = MultiIndex.from_tuples(
            [("あ", "いい"), ("う", "え"), ("おおお", "かかかか"), ("き", "くく")]
        )
        df = DataFrame(
            {
                "a": ["あああああ", "い", "う", "えええ"],
                "b": ["あ", "いいい", "う", "ええええええ"],
            },
            index=idx,
        )
        expected = (
            "              a       b\n"
            "あ   いい    あああああ       あ\n"
            "う   え         い     いいい\n"
            "おおお かかかか      う       う\n"
            "き   くく      えええ  ええええええ"
        )
        assert repr(df) == expected

        # truncate
        with option_context("display.max_rows", 3, "display.max_columns", 3):
            df = DataFrame(
                {
                    "a": ["あああああ", "い", "う", "えええ"],
                    "b": ["あ", "いいい", "う", "ええええええ"],
                    "c": ["お", "か", "ききき", "くくくくくく"],
                    "ああああ": ["さ", "し", "す", "せ"],
                },
                columns=["a", "b", "c", "ああああ"],
            )

            expected = (
                "        a  ... ああああ\n0   あああああ  ...    さ\n"
                "..    ...  ...  ...\n3     えええ  ...    せ\n"
                "\n[4 rows x 4 columns]"
            )
            assert repr(df) == expected

            df.index = ["あああ", "いいいい", "う", "aaa"]
            expected = (
                "         a  ... ああああ\nあああ  あああああ  ...    さ\n"
                "..     ...  ...  ...\naaa    えええ  ...    せ\n"
                "\n[4 rows x 4 columns]"
            )
            assert repr(df) == expected

    def test_east_asian_unicode_true(self):
        # Enable Unicode option -----------------------------------------
        with option_context("display.unicode.east_asian_width", True):
            # mid col
            df = DataFrame(
                {"a": ["あ", "いいい", "う", "ええええええ"], "b": [1, 222, 33333, 4]},
                index=["a", "bb", "c", "ddd"],
            )
            expected = (
                "                a      b\na              あ      1\n"
                "bb         いいい    222\nc              う  33333\n"
                "ddd  ええええええ      4"
            )
            assert repr(df) == expected

            # last col
            df = DataFrame(
                {"a": [1, 222, 33333, 4], "b": ["あ", "いいい", "う", "ええええええ"]},
                index=["a", "bb", "c", "ddd"],
            )
            expected = (
                "         a             b\na        1            あ\n"
                "bb     222        いいい\nc    33333            う\n"
                "ddd      4  ええええええ"
            )
            assert repr(df) == expected

            # all col
            df = DataFrame(
                {
                    "a": ["あああああ", "い", "う", "えええ"],
                    "b": ["あ", "いいい", "う", "ええええええ"],
                },
                index=["a", "bb", "c", "ddd"],
            )
            expected = (
                "              a             b\n"
                "a    あああああ            あ\n"
                "bb           い        いいい\n"
                "c            う            う\n"
                "ddd      えええ  ええええええ"
            )
            assert repr(df) == expected

            # column name
            df = DataFrame(
                {
                    "b": ["あ", "いいい", "う", "ええええええ"],
                    "あああああ": [1, 222, 33333, 4],
                },
                index=["a", "bb", "c", "ddd"],
            )
            expected = (
                "                b  あああああ\n"
                "a              あ           1\n"
                "bb         いいい         222\n"
                "c              う       33333\n"
                "ddd  ええええええ           4"
            )
            assert repr(df) == expected

            # index
            df = DataFrame(
                {
                    "a": ["あああああ", "い", "う", "えええ"],
                    "b": ["あ", "いいい", "う", "ええええええ"],
                },
                index=["あああ", "いいいいいい", "うう", "え"],
            )
            expected = (
                "                       a             b\n"
                "あああ        あああああ            あ\n"
                "いいいいいい          い        いいい\n"
                "うう                  う            う\n"
                "え                えええ  ええええええ"
            )
            assert repr(df) == expected

            # index name
            df = DataFrame(
                {
                    "a": ["あああああ", "い", "う", "えええ"],
                    "b": ["あ", "いいい", "う", "ええええええ"],
                },
                index=Index(["あ", "い", "うう", "え"], name="おおおお"),
            )
            expected = (
                "                   a             b\n"
                "おおおお                          \n"
                "あ        あああああ            あ\n"
                "い                い        いいい\n"
                "うう              う            う\n"
                "え            えええ  ええええええ"
            )
            assert repr(df) == expected

            # all
            df = DataFrame(
                {
                    "あああ": ["あああ", "い", "う", "えええええ"],
                    "いいいいい": ["あ", "いいい", "う", "ええ"],
                },
                index=Index(["あ", "いいい", "うう", "え"], name="お"),
            )
            expected = (
                "            あああ いいいいい\n"
                "お                           \n"
                "あ          あああ         あ\n"
                "いいい          い     いいい\n"
                "うう            う         う\n"
                "え      えええええ       ええ"
            )
            assert repr(df) == expected

            # MultiIndex
            idx = MultiIndex.from_tuples(
                [("あ", "いい"), ("う", "え"), ("おおお", "かかかか"), ("き", "くく")]
            )
            df = DataFrame(
                {
                    "a": ["あああああ", "い", "う", "えええ"],
                    "b": ["あ", "いいい", "う", "ええええええ"],
                },
                index=idx,
            )
            expected = (
                "                          a             b\n"
                "あ     いい      あああああ            あ\n"
                "う     え                い        いいい\n"
                "おおお かかかか          う            う\n"
                "き     くく          えええ  ええええええ"
            )
            assert repr(df) == expected

            # truncate
            with option_context("display.max_rows", 3, "display.max_columns", 3):
                df = DataFrame(
                    {
                        "a": ["あああああ", "い", "う", "えええ"],
                        "b": ["あ", "いいい", "う", "ええええええ"],
                        "c": ["お", "か", "ききき", "くくくくくく"],
                        "ああああ": ["さ", "し", "す", "せ"],
                    },
                    columns=["a", "b", "c", "ああああ"],
                )

                expected = (
                    "             a  ... ああああ\n"
                    "0   あああああ  ...       さ\n"
                    "..         ...  ...      ...\n"
                    "3       えええ  ...       せ\n"
                    "\n[4 rows x 4 columns]"
                )
                assert repr(df) == expected

                df.index = ["あああ", "いいいい", "う", "aaa"]
                expected = (
                    "                 a  ... ああああ\n"
                    "あああ  あああああ  ...       さ\n"
                    "...            ...  ...      ...\n"
                    "aaa         えええ  ...       せ\n"
                    "\n[4 rows x 4 columns]"
                )
                assert repr(df) == expected

            # ambiguous unicode
            df = DataFrame(
                {
                    "b": ["あ", "いいい", "¡¡", "ええええええ"],
                    "あああああ": [1, 222, 33333, 4],
                },
                index=["a", "bb", "c", "¡¡¡"],
            )
            expected = (
                "                b  あああああ\n"
                "a              あ           1\n"
                "bb         いいい         222\n"
                "c              ¡¡       33333\n"
                "¡¡¡  ええええええ           4"
            )
            assert repr(df) == expected

    def test_to_string_buffer_all_unicode(self):
        buf = StringIO()

        empty = DataFrame({"c/\u03c3": Series(dtype=object)})
        nonempty = DataFrame({"c/\u03c3": Series([1, 2, 3])})

        print(empty, file=buf)
        print(nonempty, file=buf)

        # this should work
        buf.getvalue()

    @pytest.mark.parametrize(
        "index_scalar",
        [
            "a" * 10,
            1,
            Timestamp(2020, 1, 1),
            pd.Period("2020-01-01"),
        ],
    )
    @pytest.mark.parametrize("h", [10, 20])
    @pytest.mark.parametrize("w", [10, 20])
    def test_to_string_truncate_indices(self, index_scalar, h, w):
        with option_context("display.expand_frame_repr", False):
            df = DataFrame(
                index=[index_scalar] * h, columns=[str(i) * 10 for i in range(w)]
            )
            with option_context("display.max_rows", 15):
                if h == 20:
                    assert has_vertically_truncated_repr(df)
                else:
                    assert not has_vertically_truncated_repr(df)
            with option_context("display.max_columns", 15):
                if w == 20:
                    assert has_horizontally_truncated_repr(df)
                else:
                    assert not has_horizontally_truncated_repr(df)
            with option_context("display.max_rows", 15, "display.max_columns", 15):
                if h == 20 and w == 20:
                    assert has_doubly_truncated_repr(df)
                else:
                    assert not has_doubly_truncated_repr(df)

    def test_to_string_truncate_multilevel(self):
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        df = DataFrame(index=arrays, columns=arrays)
        with option_context("display.max_rows", 7, "display.max_columns", 7):
            assert has_doubly_truncated_repr(df)

    @pytest.mark.parametrize("dtype", ["object", "datetime64[us]"])
    def test_truncate_with_different_dtypes(self, dtype):
        # 11594, 12045
        # when truncated the dtypes of the splits can differ

        # 11594
        ser = Series(
            [datetime(2012, 1, 1)] * 10
            + [datetime(1012, 1, 2)]
            + [datetime(2012, 1, 3)] * 10,
            dtype=dtype,
        )

        with option_context("display.max_rows", 8):
            result = str(ser)
        assert dtype in result

    def test_truncate_with_different_dtypes2(self):
        # 12045
        df = DataFrame({"text": ["some words"] + [None] * 9}, dtype=object)

        with option_context("display.max_rows", 8, "display.max_columns", 3):
            result = str(df)
            assert "None" in result
            assert "NaN" not in result

    def test_truncate_with_different_dtypes_multiindex(self):
        # GH#13000
        df = DataFrame({"Vals": range(100)})
        frame = pd.concat([df], keys=["Sweep"], names=["Sweep", "Index"])
        result = repr(frame)

        result2 = repr(frame.iloc[:5])
        assert result.startswith(result2)

    def test_datetimelike_frame(self):
        # GH 12211
        df = DataFrame({"date": [Timestamp("20130101").tz_localize("UTC")] + [NaT] * 5})

        with option_context("display.max_rows", 5):
            result = str(df)
            assert "2013-01-01 00:00:00+00:00" in result
            assert "NaT" in result
            assert "..." in result
            assert "[6 rows x 1 columns]" in result

        dts = [Timestamp("2011-01-01", tz="US/Eastern")] * 5 + [NaT] * 5
        df = DataFrame({"dt": dts, "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        with option_context("display.max_rows", 5):
            expected = (
                "                          dt   x\n"
                "0  2011-01-01 00:00:00-05:00   1\n"
                "1  2011-01-01 00:00:00-05:00   2\n"
                "..                       ...  ..\n"
                "8                        NaT   9\n"
                "9                        NaT  10\n\n"
                "[10 rows x 2 columns]"
            )
            assert repr(df) == expected

        dts = [NaT] * 5 + [Timestamp("2011-01-01", tz="US/Eastern")] * 5
        df = DataFrame({"dt": dts, "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        with option_context("display.max_rows", 5):
            expected = (
                "                          dt   x\n"
                "0                        NaT   1\n"
                "1                        NaT   2\n"
                "..                       ...  ..\n"
                "8  2011-01-01 00:00:00-05:00   9\n"
                "9  2011-01-01 00:00:00-05:00  10\n\n"
                "[10 rows x 2 columns]"
            )
            assert repr(df) == expected

        dts = [Timestamp("2011-01-01", tz="Asia/Tokyo")] * 5 + [
            Timestamp("2011-01-01", tz="US/Eastern")
        ] * 5
        df = DataFrame({"dt": dts, "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        with option_context("display.max_rows", 5):
            expected = (
                "                           dt   x\n"
                "0   2011-01-01 00:00:00+09:00   1\n"
                "1   2011-01-01 00:00:00+09:00   2\n"
                "..                        ...  ..\n"
                "8   2011-01-01 00:00:00-05:00   9\n"
                "9   2011-01-01 00:00:00-05:00  10\n\n"
                "[10 rows x 2 columns]"
            )
            assert repr(df) == expected

    @pytest.mark.parametrize(
        "start_date",
        [
            "2017-01-01 23:59:59.999999999",
            "2017-01-01 23:59:59.99999999",
            "2017-01-01 23:59:59.9999999",
            "2017-01-01 23:59:59.999999",
            "2017-01-01 23:59:59.99999",
            "2017-01-01 23:59:59.9999",
        ],
    )
    def test_datetimeindex_highprecision(self, start_date):
        # GH19030
        # Check that high-precision time values for the end of day are
        # included in repr for DatetimeIndex
        df = DataFrame({"A": date_range(start=start_date, freq="D", periods=5)})
        result = str(df)
        assert start_date in result

        dti = date_range(start=start_date, freq="D", periods=5)
        df = DataFrame({"A": range(5)}, index=dti)
        result = str(df.index)
        assert start_date in result

    def test_string_repr_encoding(self, datapath):
        filepath = datapath("io", "parser", "data", "unicode_series.csv")
        df = read_csv(filepath, header=None, encoding="latin1")
        repr(df)
        repr(df[1])

    def test_repr_corner(self):
        # representing infs poses no problems
        df = DataFrame({"foo": [-np.inf, np.inf]})
        repr(df)

    def test_frame_info_encoding(self):
        index = ["'Til There Was You (1997)", "ldum klaka (Cold Fever) (1994)"]
        with option_context("display.max_rows", 1):
            df = DataFrame(columns=["a", "b", "c"], index=index)
            repr(df)
            repr(df.T)

    def test_wide_repr(self):
        with option_context(
            "mode.sim_interactive",
            True,
            "display.show_dimensions",
            True,
            "display.max_columns",
            20,
        ):
            max_cols = get_option("display.max_columns")
            df = DataFrame([["a" * 25] * (max_cols - 1)] * 10)
            with option_context("display.expand_frame_repr", False):
                rep_str = repr(df)

            assert f"10 rows x {max_cols - 1} columns" in rep_str
            with option_context("display.expand_frame_repr", True):
                wide_repr = repr(df)
            assert rep_str != wide_repr

            with option_context("display.width", 120):
                wider_repr = repr(df)
                assert len(wider_repr) < len(wide_repr)

    def test_wide_repr_wide_columns(self):
        with option_context("mode.sim_interactive", True, "display.max_columns", 20):
            df = DataFrame(
                np.random.default_rng(2).standard_normal((5, 3)),
                columns=["a" * 90, "b" * 90, "c" * 90],
            )
            rep_str = repr(df)

            assert len(rep_str.splitlines()) == 20

    def test_wide_repr_named(self):
        with option_context("mode.sim_interactive", True, "display.max_columns", 20):
            max_cols = get_option("display.max_columns")
            df = DataFrame([["a" * 25] * (max_cols - 1)] * 10)
            df.index.name = "DataFrame Index"
            with option_context("display.expand_frame_repr", False):
                rep_str = repr(df)
            with option_context("display.expand_frame_repr", True):
                wide_repr = repr(df)
            assert rep_str != wide_repr

            with option_context("display.width", 150):
                wider_repr = repr(df)
                assert len(wider_repr) < len(wide_repr)

            for line in wide_repr.splitlines()[1::13]:
                assert "DataFrame Index" in line

    def test_wide_repr_multiindex(self):
        with option_context("mode.sim_interactive", True, "display.max_columns", 20):
            midx = MultiIndex.from_arrays([["a" * 5] * 10] * 2)
            max_cols = get_option("display.max_columns")
            df = DataFrame([["a" * 25] * (max_cols - 1)] * 10, index=midx)
            df.index.names = ["Level 0", "Level 1"]
            with option_context("display.expand_frame_repr", False):
                rep_str = repr(df)
            with option_context("display.expand_frame_repr", True):
                wide_repr = repr(df)
            assert rep_str != wide_repr

            with option_context("display.width", 150):
                wider_repr = repr(df)
                assert len(wider_repr) < len(wide_repr)

            for line in wide_repr.splitlines()[1::13]:
                assert "Level 0 Level 1" in line

    def test_wide_repr_multiindex_cols(self):
        with option_context("mode.sim_interactive", True, "display.max_columns", 20):
            max_cols = get_option("display.max_columns")
            midx = MultiIndex.from_arrays([["a" * 5] * 10] * 2)
            mcols = MultiIndex.from_arrays([["b" * 3] * (max_cols - 1)] * 2)
            df = DataFrame(
                [["c" * 25] * (max_cols - 1)] * 10, index=midx, columns=mcols
            )
            df.index.names = ["Level 0", "Level 1"]
            with option_context("display.expand_frame_repr", False):
                rep_str = repr(df)
            with option_context("display.expand_frame_repr", True):
                wide_repr = repr(df)
            assert rep_str != wide_repr

        with option_context("display.width", 150, "display.max_columns", 20):
            wider_repr = repr(df)
            assert len(wider_repr) < len(wide_repr)

    def test_wide_repr_unicode(self):
        with option_context("mode.sim_interactive", True, "display.max_columns", 20):
            max_cols = 20
            df = DataFrame([["a" * 25] * 10] * (max_cols - 1))
            with option_context("display.expand_frame_repr", False):
                rep_str = repr(df)
            with option_context("display.expand_frame_repr", True):
                wide_repr = repr(df)
            assert rep_str != wide_repr

            with option_context("display.width", 150):
                wider_repr = repr(df)
                assert len(wider_repr) < len(wide_repr)

    def test_wide_repr_wide_long_columns(self):
        with option_context("mode.sim_interactive", True):
            df = DataFrame({"a": ["a" * 30, "b" * 30], "b": ["c" * 70, "d" * 80]})

            result = repr(df)
            assert "ccccc" in result
            assert "ddddd" in result

    def test_long_series(self):
        n = 1000
        s = Series(
            np.random.default_rng(2).integers(-50, 50, n),
            index=[f"s{x:04d}" for x in range(n)],
            dtype="int64",
        )

        str_rep = str(s)
        nmatches = len(re.findall("dtype", str_rep))
        assert nmatches == 1

    def test_to_string_ascii_error(self):
        data = [
            (
                "0  ",
                "                        .gitignore ",
                "     5 ",
                " \xe2\x80\xa2\xe2\x80\xa2\xe2\x80\xa2\xe2\x80\xa2\xe2\x80\xa2",
            )
        ]
        df = DataFrame(data)

        # it works!
        repr(df)

    def test_show_dimensions(self):
        df = DataFrame(123, index=range(10, 15), columns=range(30))

        with option_context(
            "display.max_rows",
            10,
            "display.max_columns",
            40,
            "display.width",
            500,
            "display.expand_frame_repr",
            "info",
            "display.show_dimensions",
            True,
        ):
            assert "5 rows" in str(df)
            assert "5 rows" in df._repr_html_()
        with option_context(
            "display.max_rows",
            10,
            "display.max_columns",
            40,
            "display.width",
            500,
            "display.expand_frame_repr",
            "info",
            "display.show_dimensions",
            False,
        ):
            assert "5 rows" not in str(df)
            assert "5 rows" not in df._repr_html_()
        with option_context(
            "display.max_rows",
            2,
            "display.max_columns",
            2,
            "display.width",
            500,
            "display.expand_frame_repr",
            "info",
            "display.show_dimensions",
            "truncate",
        ):
            assert "5 rows" in str(df)
            assert "5 rows" in df._repr_html_()
        with option_context(
            "display.max_rows",
            10,
            "display.max_columns",
            40,
            "display.width",
            500,
            "display.expand_frame_repr",
            "info",
            "display.show_dimensions",
            "truncate",
        ):
            assert "5 rows" not in str(df)
            assert "5 rows" not in df._repr_html_()

    def test_info_repr(self):
        # GH#21746 For tests inside a terminal (i.e. not CI) we need to detect
        # the terminal size to ensure that we try to print something "too big"
        term_width, term_height = get_terminal_size()

        max_rows = 60
        max_cols = 20 + (max(term_width, 80) - 80) // 4
        # Long
        h, w = max_rows + 1, max_cols - 1
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        assert has_vertically_truncated_repr(df)
        with option_context("display.large_repr", "info"):
            assert has_info_repr(df)

        # Wide
        h, w = max_rows - 1, max_cols + 1
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        assert has_horizontally_truncated_repr(df)
        with option_context(
            "display.large_repr", "info", "display.max_columns", max_cols
        ):
            assert has_info_repr(df)

    def test_info_repr_max_cols(self):
        # GH #6939
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        with option_context(
            "display.large_repr",
            "info",
            "display.max_columns",
            1,
            "display.max_info_columns",
            4,
        ):
            assert has_non_verbose_info_repr(df)

        with option_context(
            "display.large_repr",
            "info",
            "display.max_columns",
            1,
            "display.max_info_columns",
            5,
        ):
            assert not has_non_verbose_info_repr(df)

        # FIXME: don't leave commented-out
        # test verbose overrides
        # set_option('display.max_info_columns', 4)  # exceeded

    def test_pprint_pathological_object(self):
        """
        If the test fails, it at least won't hang.
        """

        class A:
            def __getitem__(self, key):
                return 3  # obviously simplified

        df = DataFrame([A()])
        repr(df)  # just don't die

    def test_float_trim_zeros(self):
        vals = [
            2.08430917305e10,
            3.52205017305e10,
            2.30674817305e10,
            2.03954217305e10,
            5.59897817305e10,
        ]
        skip = True
        for line in repr(DataFrame({"A": vals})).split("\n")[:-2]:
            if line.startswith("dtype:"):
                continue
            if _three_digit_exp():
                assert ("+010" in line) or skip
            else:
                assert ("+10" in line) or skip
            skip = False

    @pytest.mark.parametrize(
        "data, expected",
        [
            (["3.50"], "0    3.50\ndtype: object"),
            ([1.20, "1.00"], "0     1.2\n1    1.00\ndtype: object"),
            ([np.nan], "0   NaN\ndtype: float64"),
            ([None], "0    None\ndtype: object"),
            (["3.50", np.nan], "0    3.50\n1     NaN\ndtype: object"),
            ([3.50, np.nan], "0    3.5\n1    NaN\ndtype: float64"),
            ([3.50, np.nan, "3.50"], "0     3.5\n1     NaN\n2    3.50\ndtype: object"),
            ([3.50, None, "3.50"], "0     3.5\n1    None\n2    3.50\ndtype: object"),
        ],
    )
    def test_repr_str_float_truncation(self, data, expected, using_infer_string):
        # GH#38708
        series = Series(data, dtype=object if "3.50" in data else None)
        result = repr(series)
        assert result == expected

    @pytest.mark.parametrize(
        "float_format,expected",
        [
            ("{:,.0f}".format, "0   1,000\n1    test\ndtype: object"),
            ("{:.4f}".format, "0   1000.0000\n1        test\ndtype: object"),
        ],
    )
    def test_repr_float_format_in_object_col(self, float_format, expected):
        # GH#40024
        df = Series([1000.0, "test"])
        with option_context("display.float_format", float_format):
            result = repr(df)

        assert result == expected

    def test_period(self):
        # GH 12615
        df = DataFrame(
            {
                "A": pd.period_range("2013-01", periods=4, freq="M"),
                "B": [
                    pd.Period("2011-01", freq="M"),
                    pd.Period("2011-02-01", freq="D"),
                    pd.Period("2011-03-01 09:00", freq="h"),
                    pd.Period("2011-04", freq="M"),
                ],
                "C": list("abcd"),
            }
        )
        exp = (
            "         A                 B  C\n"
            "0  2013-01           2011-01  a\n"
            "1  2013-02        2011-02-01  b\n"
            "2  2013-03  2011-03-01 09:00  c\n"
            "3  2013-04           2011-04  d"
        )
        assert str(df) == exp

    @pytest.mark.parametrize(
        "length, max_rows, min_rows, expected",
        [
            (10, 10, 10, 10),
            (10, 10, None, 10),
            (10, 8, None, 8),
            (20, 30, 10, 30),  # max_rows > len(frame), hence max_rows
            (50, 30, 10, 10),  # max_rows < len(frame), hence min_rows
            (100, 60, 10, 10),  # same
            (60, 60, 10, 60),  # edge case
            (61, 60, 10, 10),  # edge case
        ],
    )
    def test_max_rows_fitted(self, length, min_rows, max_rows, expected):
        """Check that display logic is correct.

        GH #37359

        See description here:
        https://pandas.pydata.org/docs/dev/user_guide/options.html#frequently-used-options
        """
        formatter = fmt.DataFrameFormatter(
            DataFrame(np.random.default_rng(2).random((length, 3))),
            max_rows=max_rows,
            min_rows=min_rows,
        )
        result = formatter.max_rows_fitted
        assert result == expected


def gen_series_formatting():
    s1 = Series(["a"] * 100)
    s2 = Series(["ab"] * 100)
    s3 = Series(["a", "ab", "abc", "abcd", "abcde", "abcdef"])
    s4 = s3[::-1]
    test_sers = {"onel": s1, "twol": s2, "asc": s3, "desc": s4}
    return test_sers


class TestSeriesFormatting:
    def test_freq_name_separation(self):
        s = Series(
            np.random.default_rng(2).standard_normal(10),
            index=date_range("1/1/2000", periods=10),
            name=0,
        )

        result = repr(s)
        assert "Freq: D, Name: 0" in result

    def test_unicode_name_in_footer(self):
        s = Series([1, 2], name="\u05e2\u05d1\u05e8\u05d9\u05ea")
        sf = fmt.SeriesFormatter(s, name="\u05e2\u05d1\u05e8\u05d9\u05ea")
        sf._get_footer()  # should not raise exception

    @pytest.mark.xfail(using_string_dtype(), reason="Fixup when arrow is default")
    def test_east_asian_unicode_series(self):
        # not aligned properly because of east asian width

        # unicode index
        s = Series(["a", "bb", "CCC", "D"], index=["あ", "いい", "ううう", "ええええ"])
        expected = "".join(
            [
                "あ         a\n",
                "いい       bb\n",
                "ううう     CCC\n",
                "ええええ      D\ndtype: object",
            ]
        )
        assert repr(s) == expected

        # unicode values
        s = Series(["あ", "いい", "ううう", "ええええ"], index=["a", "bb", "c", "ddd"])
        expected = "".join(
            [
                "a         あ\n",
                "bb       いい\n",
                "c       ううう\n",
                "ddd    ええええ\n",
                "dtype: object",
            ]
        )

        assert repr(s) == expected

        # both
        s = Series(
            ["あ", "いい", "ううう", "ええええ"],
            index=["ああ", "いいいい", "う", "えええ"],
        )
        expected = "".join(
            [
                "ああ         あ\n",
                "いいいい      いい\n",
                "う        ううう\n",
                "えええ     ええええ\n",
                "dtype: object",
            ]
        )

        assert repr(s) == expected

        # unicode footer
        s = Series(
            ["あ", "いい", "ううう", "ええええ"],
            index=["ああ", "いいいい", "う", "えええ"],
            name="おおおおおおお",
        )
        expected = (
            "ああ         あ\nいいいい      いい\nう        ううう\n"
            "えええ     ええええ\nName: おおおおおおお, dtype: object"
        )
        assert repr(s) == expected

        # MultiIndex
        idx = MultiIndex.from_tuples(
            [("あ", "いい"), ("う", "え"), ("おおお", "かかかか"), ("き", "くく")]
        )
        s = Series([1, 22, 3333, 44444], index=idx)
        expected = (
            "あ    いい          1\n"
            "う    え          22\n"
            "おおお  かかかか     3333\n"
            "き    くく      44444\ndtype: int64"
        )
        assert repr(s) == expected

        # object dtype, shorter than unicode repr
        s = Series([1, 22, 3333, 44444], index=[1, "AB", np.nan, "あああ"])
        expected = (
            "1          1\nAB        22\nNaN     3333\nあああ    44444\ndtype: int64"
        )
        assert repr(s) == expected

        # object dtype, longer than unicode repr
        s = Series(
            [1, 22, 3333, 44444], index=[1, "AB", Timestamp("2011-01-01"), "あああ"]
        )
        expected = (
            "1                          1\n"
            "AB                        22\n"
            "2011-01-01 00:00:00     3333\n"
            "あああ                    44444\ndtype: int64"
        )
        assert repr(s) == expected

        # truncate
        with option_context("display.max_rows", 3):
            s = Series(["あ", "いい", "ううう", "ええええ"], name="おおおおおおお")

            expected = (
                "0       あ\n     ... \n"
                "3    ええええ\n"
                "Name: おおおおおおお, Length: 4, dtype: object"
            )
            assert repr(s) == expected

            s.index = ["ああ", "いいいい", "う", "えええ"]
            expected = (
                "ああ        あ\n       ... \n"
                "えええ    ええええ\n"
                "Name: おおおおおおお, Length: 4, dtype: object"
            )
            assert repr(s) == expected

        # Enable Unicode option -----------------------------------------
        with option_context("display.unicode.east_asian_width", True):
            # unicode index
            s = Series(
                ["a", "bb", "CCC", "D"],
                index=["あ", "いい", "ううう", "ええええ"],
            )
            expected = (
                "あ            a\nいい         bb\nううう      CCC\n"
                "ええええ      D\ndtype: object"
            )
            assert repr(s) == expected

            # unicode values
            s = Series(
                ["あ", "いい", "ううう", "ええええ"],
                index=["a", "bb", "c", "ddd"],
            )
            expected = (
                "a            あ\nbb         いい\nc        ううう\n"
                "ddd    ええええ\ndtype: object"
            )
            assert repr(s) == expected
            # both
            s = Series(
                ["あ", "いい", "ううう", "ええええ"],
                index=["ああ", "いいいい", "う", "えええ"],
            )
            expected = (
                "ああ              あ\n"
                "いいいい        いい\n"
                "う            ううう\n"
                "えええ      ええええ\ndtype: object"
            )
            assert repr(s) == expected

            # unicode footer
            s = Series(
                ["あ", "いい", "ううう", "ええええ"],
                index=["ああ", "いいいい", "う", "えええ"],
                name="おおおおおおお",
            )
            expected = (
                "ああ              あ\n"
                "いいいい        いい\n"
                "う            ううう\n"
                "えええ      ええええ\n"
                "Name: おおおおおおお, dtype: object"
            )
            assert repr(s) == expected

            # MultiIndex
            idx = MultiIndex.from_tuples(
                [("あ", "いい"), ("う", "え"), ("おおお", "かかかか"), ("き", "くく")]
            )
            s = Series([1, 22, 3333, 44444], index=idx)
            expected = (
                "あ      いい            1\n"
                "う      え             22\n"
                "おおお  かかかか     3333\n"
                "き      くく        44444\n"
                "dtype: int64"
            )
            assert repr(s) == expected

            # object dtype, shorter than unicode repr
            s = Series([1, 22, 3333, 44444], index=[1, "AB", np.nan, "あああ"])
            expected = (
                "1             1\nAB           22\nNaN        3333\n"
                "あああ    44444\ndtype: int64"
            )
            assert repr(s) == expected

            # object dtype, longer than unicode repr
            s = Series(
                [1, 22, 3333, 44444],
                index=[1, "AB", Timestamp("2011-01-01"), "あああ"],
            )
            expected = (
                "1                          1\n"
                "AB                        22\n"
                "2011-01-01 00:00:00     3333\n"
                "あああ                 44444\ndtype: int64"
            )
            assert repr(s) == expected

            # truncate
            with option_context("display.max_rows", 3):
                s = Series(["あ", "いい", "ううう", "ええええ"], name="おおおおおおお")
                expected = (
                    "0          あ\n       ...   \n"
                    "3    ええええ\n"
                    "Name: おおおおおおお, Length: 4, dtype: object"
                )
                assert repr(s) == expected

                s.index = ["ああ", "いいいい", "う", "えええ"]
                expected = (
                    "ああ            あ\n"
                    "            ...   \n"
                    "えええ    ええええ\n"
                    "Name: おおおおおおお, Length: 4, dtype: object"
                )
                assert repr(s) == expected

            # ambiguous unicode
            s = Series(
                ["¡¡", "い¡¡", "ううう", "ええええ"],
                index=["ああ", "¡¡¡¡いい", "¡¡", "えええ"],
            )
            expected = (
                "ああ              ¡¡\n"
                "¡¡¡¡いい        い¡¡\n"
                "¡¡            ううう\n"
                "えええ      ええええ\ndtype: object"
            )
            assert repr(s) == expected

    def test_float_trim_zeros(self):
        vals = [
            2.08430917305e10,
            3.52205017305e10,
            2.30674817305e10,
            2.03954217305e10,
            5.59897817305e10,
        ]
        for line in repr(Series(vals)).split("\n"):
            if line.startswith("dtype:"):
                continue
            if _three_digit_exp():
                assert "+010" in line
            else:
                assert "+10" in line

    @pytest.mark.parametrize(
        "start_date",
        [
            "2017-01-01 23:59:59.999999999",
            "2017-01-01 23:59:59.99999999",
            "2017-01-01 23:59:59.9999999",
            "2017-01-01 23:59:59.999999",
            "2017-01-01 23:59:59.99999",
            "2017-01-01 23:59:59.9999",
        ],
    )
    def test_datetimeindex_highprecision(self, start_date):
        # GH19030
        # Check that high-precision time values for the end of day are
        # included in repr for DatetimeIndex
        s1 = Series(date_range(start=start_date, freq="D", periods=5))
        result = str(s1)
        assert start_date in result

        dti = date_range(start=start_date, freq="D", periods=5)
        s2 = Series(3, index=dti)
        result = str(s2.index)
        assert start_date in result

    def test_mixed_datetime64(self):
        df = DataFrame({"A": [1, 2], "B": ["2012-01-01", "2012-01-02"]})
        df["B"] = pd.to_datetime(df.B)

        result = repr(df.loc[0])
        assert "2012-01-01" in result

    def test_period(self):
        # GH 12615
        index = pd.period_range("2013-01", periods=6, freq="M")
        s = Series(np.arange(6, dtype="int64"), index=index)
        exp = (
            "2013-01    0\n"
            "2013-02    1\n"
            "2013-03    2\n"
            "2013-04    3\n"
            "2013-05    4\n"
            "2013-06    5\n"
            "Freq: M, dtype: int64"
        )
        assert str(s) == exp

        s = Series(index)
        exp = (
            "0    2013-01\n"
            "1    2013-02\n"
            "2    2013-03\n"
            "3    2013-04\n"
            "4    2013-05\n"
            "5    2013-06\n"
            "dtype: period[M]"
        )
        assert str(s) == exp

        # periods with mixed freq
        s = Series(
            [
                pd.Period("2011-01", freq="M"),
                pd.Period("2011-02-01", freq="D"),
                pd.Period("2011-03-01 09:00", freq="h"),
            ]
        )
        exp = (
            "0             2011-01\n1          2011-02-01\n"
            "2    2011-03-01 09:00\ndtype: object"
        )
        assert str(s) == exp

    def test_max_multi_index_display(self):
        # GH 7101

        # doc example (indexing.rst)

        # multi-index
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        tuples = list(zip(*arrays))
        index = MultiIndex.from_tuples(tuples, names=["first", "second"])
        s = Series(np.random.default_rng(2).standard_normal(8), index=index)

        with option_context("display.max_rows", 10):
            assert len(str(s).split("\n")) == 10
        with option_context("display.max_rows", 3):
            assert len(str(s).split("\n")) == 5
        with option_context("display.max_rows", 2):
            assert len(str(s).split("\n")) == 5
        with option_context("display.max_rows", 1):
            assert len(str(s).split("\n")) == 4
        with option_context("display.max_rows", 0):
            assert len(str(s).split("\n")) == 10

        # index
        s = Series(np.random.default_rng(2).standard_normal(8), None)

        with option_context("display.max_rows", 10):
            assert len(str(s).split("\n")) == 9
        with option_context("display.max_rows", 3):
            assert len(str(s).split("\n")) == 4
        with option_context("display.max_rows", 2):
            assert len(str(s).split("\n")) == 4
        with option_context("display.max_rows", 1):
            assert len(str(s).split("\n")) == 3
        with option_context("display.max_rows", 0):
            assert len(str(s).split("\n")) == 9

    # Make sure #8532 is fixed
    def test_consistent_format(self):
        s = Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9999, 1, 1] * 10)
        with option_context("display.max_rows", 10, "display.show_dimensions", False):
            res = repr(s)
        exp = (
            "0      1.0000\n1      1.0000\n2      1.0000\n3      "
            "1.0000\n4      1.0000\n        ...  \n125    "
            "1.0000\n126    1.0000\n127    0.9999\n128    "
            "1.0000\n129    1.0000\ndtype: float64"
        )
        assert res == exp

    def chck_ncols(self, s):
        lines = [
            line for line in repr(s).split("\n") if not re.match(r"[^\.]*\.+", line)
        ][:-1]
        ncolsizes = len({len(line.strip()) for line in lines})
        assert ncolsizes == 1

    @pytest.mark.xfail(using_string_dtype(), reason="change when arrow is default")
    def test_format_explicit(self):
        test_sers = gen_series_formatting()
        with option_context("display.max_rows", 4, "display.show_dimensions", False):
            res = repr(test_sers["onel"])
            exp = "0     a\n1     a\n     ..\n98    a\n99    a\ndtype: object"
            assert exp == res
            res = repr(test_sers["twol"])
            exp = "0     ab\n1     ab\n      ..\n98    ab\n99    ab\ndtype: object"
            assert exp == res
            res = repr(test_sers["asc"])
            exp = (
                "0         a\n1        ab\n      ...  \n4     abcde\n5    "
                "abcdef\ndtype: object"
            )
            assert exp == res
            res = repr(test_sers["desc"])
            exp = (
                "5    abcdef\n4     abcde\n      ...  \n1        ab\n0         "
                "a\ndtype: object"
            )
            assert exp == res

    def test_ncols(self):
        test_sers = gen_series_formatting()
        for s in test_sers.values():
            self.chck_ncols(s)

    def test_max_rows_eq_one(self):
        s = Series(range(10), dtype="int64")
        with option_context("display.max_rows", 1):
            strrepr = repr(s).split("\n")
        exp1 = ["0", "0"]
        res1 = strrepr[0].split()
        assert exp1 == res1
        exp2 = [".."]
        res2 = strrepr[1].split()
        assert exp2 == res2

    def test_truncate_ndots(self):
        def getndots(s):
            return len(re.match(r"[^\.]*(\.*)", s).groups()[0])

        s = Series([0, 2, 3, 6])
        with option_context("display.max_rows", 2):
            strrepr = repr(s).replace("\n", "")
        assert getndots(strrepr) == 2

        s = Series([0, 100, 200, 400])
        with option_context("display.max_rows", 2):
            strrepr = repr(s).replace("\n", "")
        assert getndots(strrepr) == 3

    def test_show_dimensions(self):
        # gh-7117
        s = Series(range(5))

        assert "Length" not in repr(s)

        with option_context("display.max_rows", 4):
            assert "Length" in repr(s)

        with option_context("display.show_dimensions", True):
            assert "Length" in repr(s)

        with option_context("display.max_rows", 4, "display.show_dimensions", False):
            assert "Length" not in repr(s)

    def test_repr_min_rows(self):
        s = Series(range(20))

        # default setting no truncation even if above min_rows
        assert ".." not in repr(s)

        s = Series(range(61))

        # default of max_rows 60 triggers truncation if above
        assert ".." in repr(s)

        with option_context("display.max_rows", 10, "display.min_rows", 4):
            # truncated after first two rows
            assert ".." in repr(s)
            assert "2  " not in repr(s)

        with option_context("display.max_rows", 12, "display.min_rows", None):
            # when set to None, follow value of max_rows
            assert "5      5" in repr(s)

        with option_context("display.max_rows", 10, "display.min_rows", 12):
            # when set value higher as max_rows, use the minimum
            assert "5      5" not in repr(s)

        with option_context("display.max_rows", None, "display.min_rows", 12):
            # max_rows of None -> never truncate
            assert ".." not in repr(s)


class TestGenericArrayFormatter:
    def test_1d_array(self):
        # _GenericArrayFormatter is used on types for which there isn't a dedicated
        # formatter. np.bool_ is one of those types.
        obj = fmt._GenericArrayFormatter(np.array([True, False]))
        res = obj.get_result()
        assert len(res) == 2
        # Results should be right-justified.
        assert res[0] == "  True"
        assert res[1] == " False"

    def test_2d_array(self):
        obj = fmt._GenericArrayFormatter(np.array([[True, False], [False, True]]))
        res = obj.get_result()
        assert len(res) == 2
        assert res[0] == " [True, False]"
        assert res[1] == " [False, True]"

    def test_3d_array(self):
        obj = fmt._GenericArrayFormatter(
            np.array([[[True, True], [False, False]], [[False, True], [True, False]]])
        )
        res = obj.get_result()
        assert len(res) == 2
        assert res[0] == " [[True, True], [False, False]]"
        assert res[1] == " [[False, True], [True, False]]"

    def test_2d_extension_type(self):
        # GH 33770

        # Define a stub extension type with just enough code to run Series.__repr__()
        class DtypeStub(pd.api.extensions.ExtensionDtype):
            @property
            def type(self):
                return np.ndarray

            @property
            def name(self):
                return "DtypeStub"

        class ExtTypeStub(pd.api.extensions.ExtensionArray):
            def __len__(self) -> int:
                return 2

            def __getitem__(self, ix):
                return [ix == 1, ix == 0]

            @property
            def dtype(self):
                return DtypeStub()

        series = Series(ExtTypeStub(), copy=False)
        res = repr(series)  # This line crashed before #33770 was fixed.
        expected = "\n".join(
            ["0    [False True]", "1    [True False]", "dtype: DtypeStub"]
        )
        assert res == expected


def _three_digit_exp():
    return f"{1.7e8:.4g}" == "1.7e+008"


class TestFloatArrayFormatter:
    def test_misc(self):
        obj = fmt.FloatArrayFormatter(np.array([], dtype=np.float64))
        result = obj.get_result()
        assert len(result) == 0

    def test_format(self):
        obj = fmt.FloatArrayFormatter(np.array([12, 0], dtype=np.float64))
        result = obj.get_result()
        assert result[0] == " 12.0"
        assert result[1] == "  0.0"

    def test_output_display_precision_trailing_zeroes(self):
        # Issue #20359: trimming zeros while there is no decimal point

        # Happens when display precision is set to zero
        with option_context("display.precision", 0):
            s = Series([840.0, 4200.0])
            expected_output = "0     840\n1    4200\ndtype: float64"
            assert str(s) == expected_output

    @pytest.mark.parametrize(
        "value,expected",
        [
            ([9.4444], "   0\n0  9"),
            ([0.49], "       0\n0  5e-01"),
            ([10.9999], "    0\n0  11"),
            ([9.5444, 9.6], "    0\n0  10\n1  10"),
            ([0.46, 0.78, -9.9999], "       0\n0  5e-01\n1  8e-01\n2 -1e+01"),
        ],
    )
    def test_set_option_precision(self, value, expected):
        # Issue #30122
        # Precision was incorrectly shown

        with option_context("display.precision", 0):
            df_value = DataFrame(value)
            assert str(df_value) == expected

    def test_output_significant_digits(self):
        # Issue #9764

        # In case default display precision changes:
        with option_context("display.precision", 6):
            # DataFrame example from issue #9764
            d = DataFrame(
                {
                    "col1": [
                        9.999e-8,
                        1e-7,
                        1.0001e-7,
                        2e-7,
                        4.999e-7,
                        5e-7,
                        5.0001e-7,
                        6e-7,
                        9.999e-7,
                        1e-6,
                        1.0001e-6,
                        2e-6,
                        4.999e-6,
                        5e-6,
                        5.0001e-6,
                        6e-6,
                    ]
                }
            )

            expected_output = {
                (0, 6): "           col1\n"
                "0  9.999000e-08\n"
                "1  1.000000e-07\n"
                "2  1.000100e-07\n"
                "3  2.000000e-07\n"
                "4  4.999000e-07\n"
                "5  5.000000e-07",
                (1, 6): "           col1\n"
                "1  1.000000e-07\n"
                "2  1.000100e-07\n"
                "3  2.000000e-07\n"
                "4  4.999000e-07\n"
                "5  5.000000e-07",
                (1, 8): "           col1\n"
                "1  1.000000e-07\n"
                "2  1.000100e-07\n"
                "3  2.000000e-07\n"
                "4  4.999000e-07\n"
                "5  5.000000e-07\n"
                "6  5.000100e-07\n"
                "7  6.000000e-07",
                (8, 16): "            col1\n"
                "8   9.999000e-07\n"
                "9   1.000000e-06\n"
                "10  1.000100e-06\n"
                "11  2.000000e-06\n"
                "12  4.999000e-06\n"
                "13  5.000000e-06\n"
                "14  5.000100e-06\n"
                "15  6.000000e-06",
                (9, 16): "        col1\n"
                "9   0.000001\n"
                "10  0.000001\n"
                "11  0.000002\n"
                "12  0.000005\n"
                "13  0.000005\n"
                "14  0.000005\n"
                "15  0.000006",
            }

            for (start, stop), v in expected_output.items():
                assert str(d[start:stop]) == v

    def test_too_long(self):
        # GH 10451
        with option_context("display.precision", 4):
            # need both a number > 1e6 and something that normally formats to
            # having length > display.precision + 6
            df = DataFrame({"x": [12345.6789]})
            assert str(df) == "            x\n0  12345.6789"
            df = DataFrame({"x": [2e6]})
            assert str(df) == "           x\n0  2000000.0"
            df = DataFrame({"x": [12345.6789, 2e6]})
            assert str(df) == "            x\n0  1.2346e+04\n1  2.0000e+06"


class TestTimedelta64Formatter:
    def test_days(self):
        x = pd.to_timedelta(list(range(5)) + [NaT], unit="D")._values
        result = fmt._Timedelta64Formatter(x).get_result()
        assert result[0].strip() == "0 days"
        assert result[1].strip() == "1 days"

        result = fmt._Timedelta64Formatter(x[1:2]).get_result()
        assert result[0].strip() == "1 days"

        result = fmt._Timedelta64Formatter(x).get_result()
        assert result[0].strip() == "0 days"
        assert result[1].strip() == "1 days"

        result = fmt._Timedelta64Formatter(x[1:2]).get_result()
        assert result[0].strip() == "1 days"

    def test_days_neg(self):
        x = pd.to_timedelta(list(range(5)) + [NaT], unit="D")._values
        result = fmt._Timedelta64Formatter(-x).get_result()
        assert result[0].strip() == "0 days"
        assert result[1].strip() == "-1 days"

    def test_subdays(self):
        y = pd.to_timedelta(list(range(5)) + [NaT], unit="s")._values
        result = fmt._Timedelta64Formatter(y).get_result()
        assert result[0].strip() == "0 days 00:00:00"
        assert result[1].strip() == "0 days 00:00:01"

    def test_subdays_neg(self):
        y = pd.to_timedelta(list(range(5)) + [NaT], unit="s")._values
        result = fmt._Timedelta64Formatter(-y).get_result()
        assert result[0].strip() == "0 days 00:00:00"
        assert result[1].strip() == "-1 days +23:59:59"

    def test_zero(self):
        x = pd.to_timedelta(list(range(1)) + [NaT], unit="D")._values
        result = fmt._Timedelta64Formatter(x).get_result()
        assert result[0].strip() == "0 days"

        x = pd.to_timedelta(list(range(1)), unit="D")._values
        result = fmt._Timedelta64Formatter(x).get_result()
        assert result[0].strip() == "0 days"


class TestDatetime64Formatter:
    def test_mixed(self):
        x = Series([datetime(2013, 1, 1), datetime(2013, 1, 1, 12), NaT])._values
        result = fmt._Datetime64Formatter(x).get_result()
        assert result[0].strip() == "2013-01-01 00:00:00"
        assert result[1].strip() == "2013-01-01 12:00:00"

    def test_dates(self):
        x = Series([datetime(2013, 1, 1), datetime(2013, 1, 2), NaT])._values
        result = fmt._Datetime64Formatter(x).get_result()
        assert result[0].strip() == "2013-01-01"
        assert result[1].strip() == "2013-01-02"

    def test_date_nanos(self):
        x = Series([Timestamp(200)])._values
        result = fmt._Datetime64Formatter(x).get_result()
        assert result[0].strip() == "1970-01-01 00:00:00.000000200"

    def test_dates_display(self):
        # 10170
        # make sure that we are consistently display date formatting
        x = Series(date_range("20130101 09:00:00", periods=5, freq="D"))
        x.iloc[1] = np.nan
        result = fmt._Datetime64Formatter(x._values).get_result()
        assert result[0].strip() == "2013-01-01 09:00:00"
        assert result[1].strip() == "NaT"
        assert result[4].strip() == "2013-01-05 09:00:00"

        x = Series(date_range("20130101 09:00:00", periods=5, freq="s"))
        x.iloc[1] = np.nan
        result = fmt._Datetime64Formatter(x._values).get_result()
        assert result[0].strip() == "2013-01-01 09:00:00"
        assert result[1].strip() == "NaT"
        assert result[4].strip() == "2013-01-01 09:00:04"

        x = Series(date_range("20130101 09:00:00", periods=5, freq="ms"))
        x.iloc[1] = np.nan
        result = fmt._Datetime64Formatter(x._values).get_result()
        assert result[0].strip() == "2013-01-01 09:00:00.000"
        assert result[1].strip() == "NaT"
        assert result[4].strip() == "2013-01-01 09:00:00.004"

        x = Series(date_range("20130101 09:00:00", periods=5, freq="us"))
        x.iloc[1] = np.nan
        result = fmt._Datetime64Formatter(x._values).get_result()
        assert result[0].strip() == "2013-01-01 09:00:00.000000"
        assert result[1].strip() == "NaT"
        assert result[4].strip() == "2013-01-01 09:00:00.000004"

        x = Series(date_range("20130101 09:00:00", periods=5, freq="ns"))
        x.iloc[1] = np.nan
        result = fmt._Datetime64Formatter(x._values).get_result()
        assert result[0].strip() == "2013-01-01 09:00:00.000000000"
        assert result[1].strip() == "NaT"
        assert result[4].strip() == "2013-01-01 09:00:00.000000004"

    def test_datetime64formatter_yearmonth(self):
        x = Series([datetime(2016, 1, 1), datetime(2016, 2, 2)])._values

        def format_func(x):
            return x.strftime("%Y-%m")

        formatter = fmt._Datetime64Formatter(x, formatter=format_func)
        result = formatter.get_result()
        assert result == ["2016-01", "2016-02"]

    def test_datetime64formatter_hoursecond(self):
        x = Series(
            pd.to_datetime(["10:10:10.100", "12:12:12.120"], format="%H:%M:%S.%f")
        )._values

        def format_func(x):
            return x.strftime("%H:%M")

        formatter = fmt._Datetime64Formatter(x, formatter=format_func)
        result = formatter.get_result()
        assert result == ["10:10", "12:12"]

    def test_datetime64formatter_tz_ms(self):
        x = (
            Series(
                np.array(["2999-01-01", "2999-01-02", "NaT"], dtype="datetime64[ms]")
            )
            .dt.tz_localize("US/Pacific")
            ._values
        )
        result = fmt._Datetime64TZFormatter(x).get_result()
        assert result[0].strip() == "2999-01-01 00:00:00-08:00"
        assert result[1].strip() == "2999-01-02 00:00:00-08:00"


class TestFormatPercentiles:
    @pytest.mark.parametrize(
        "percentiles, expected",
        [
            (
                [0.01999, 0.02001, 0.5, 0.666666, 0.9999],
                ["1.999%", "2.001%", "50%", "66.667%", "99.99%"],
            ),
            (
                [0, 0.5, 0.02001, 0.5, 0.666666, 0.9999],
                ["0%", "50%", "2.0%", "50%", "66.67%", "99.99%"],
            ),
            ([0.281, 0.29, 0.57, 0.58], ["28.1%", "29%", "57%", "58%"]),
            ([0.28, 0.29, 0.57, 0.58], ["28%", "29%", "57%", "58%"]),
            (
                [0.9, 0.99, 0.999, 0.9999, 0.99999],
                ["90%", "99%", "99.9%", "99.99%", "99.999%"],
            ),
        ],
    )
    def test_format_percentiles(self, percentiles, expected):
        result = fmt.format_percentiles(percentiles)
        assert result == expected

    @pytest.mark.parametrize(
        "percentiles",
        [
            ([0.1, np.nan, 0.5]),
            ([-0.001, 0.1, 0.5]),
            ([2, 0.1, 0.5]),
            ([0.1, 0.5, "a"]),
        ],
    )
    def test_error_format_percentiles(self, percentiles):
        msg = r"percentiles should all be in the interval \[0,1\]"
        with pytest.raises(ValueError, match=msg):
            fmt.format_percentiles(percentiles)

    def test_format_percentiles_integer_idx(self):
        # Issue #26660
        result = fmt.format_percentiles(np.linspace(0, 1, 10 + 1))
        expected = [
            "0%",
            "10%",
            "20%",
            "30%",
            "40%",
            "50%",
            "60%",
            "70%",
            "80%",
            "90%",
            "100%",
        ]
        assert result == expected


@pytest.mark.parametrize("method", ["to_string", "to_html", "to_latex"])
@pytest.mark.parametrize(
    "encoding, data",
    [(None, "abc"), ("utf-8", "abc"), ("gbk", "造成输出中文显示乱码"), ("foo", "abc")],
)
def test_filepath_or_buffer_arg(
    method,
    filepath_or_buffer,
    assert_filepath_or_buffer_equals,
    encoding,
    data,
    filepath_or_buffer_id,
):
    df = DataFrame([data])
    if method in ["to_latex"]:  # uses styler implementation
        pytest.importorskip("jinja2")

    if filepath_or_buffer_id not in ["string", "pathlike"] and encoding is not None:
        with pytest.raises(
            ValueError, match="buf is not a file name and encoding is specified."
        ):
            getattr(df, method)(buf=filepath_or_buffer, encoding=encoding)
    elif encoding == "foo":
        with pytest.raises(LookupError, match="unknown encoding"):
            getattr(df, method)(buf=filepath_or_buffer, encoding=encoding)
    else:
        expected = getattr(df, method)()
        getattr(df, method)(buf=filepath_or_buffer, encoding=encoding)
        assert_filepath_or_buffer_equals(expected)


@pytest.mark.parametrize("method", ["to_string", "to_html", "to_latex"])
def test_filepath_or_buffer_bad_arg_raises(float_frame, method):
    if method in ["to_latex"]:  # uses styler implementation
        pytest.importorskip("jinja2")
    msg = "buf is not a file name and it has no write method"
    with pytest.raises(TypeError, match=msg):
        getattr(float_frame, method)(buf=object())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\control\tests\test_lti.py ===
from sympy.core.add import Add
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, pi, Rational, oo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan
from sympy.matrices.dense import eye
from sympy.physics.control.lti import SISOLinearTimeInvariant
from sympy.polys.polytools import factor
from sympy.polys.rootoftools import CRootOf
from sympy.simplify.simplify import simplify
from sympy.core.containers import Tuple
from sympy.matrices import ImmutableMatrix, Matrix, ShapeError
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.physics.control import (TransferFunction, PIDController, Series, Parallel,
    Feedback, TransferFunctionMatrix, MIMOSeries, MIMOParallel, MIMOFeedback,
    StateSpace, gbt, bilinear, forward_diff, backward_diff, phase_margin, gain_margin)
from sympy.testing.pytest import raises

a, x, b, c, s, g, d, p, k, tau, zeta, wn, T = symbols('a, x, b, c, s, g, d, p, k,\
    tau, zeta, wn, T')
a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3 = symbols('a0:4,\
    b0:4, c0:4, d0:4')
TF1 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
TF2 = TransferFunction(k, 1, s)
TF3 = TransferFunction(a2*p - s, a2*s + p, s)


def test_TransferFunction_construction():
    tf = TransferFunction(s + 1, s**2 + s + 1, s)
    assert tf.num == (s + 1)
    assert tf.den == (s**2 + s + 1)
    assert tf.args == (s + 1, s**2 + s + 1, s)

    tf1 = TransferFunction(s + 4, s - 5, s)
    assert tf1.num == (s + 4)
    assert tf1.den == (s - 5)
    assert tf1.args == (s + 4, s - 5, s)

    # using different polynomial variables.
    tf2 = TransferFunction(p + 3, p**2 - 9, p)
    assert tf2.num == (p + 3)
    assert tf2.den == (p**2 - 9)
    assert tf2.args == (p + 3, p**2 - 9, p)

    tf3 = TransferFunction(p**3 + 5*p**2 + 4, p**4 + 3*p + 1, p)
    assert tf3.args == (p**3 + 5*p**2 + 4, p**4 + 3*p + 1, p)

    # no pole-zero cancellation on its own.
    tf4 = TransferFunction((s + 3)*(s - 1), (s - 1)*(s + 5), s)
    assert tf4.den == (s - 1)*(s + 5)
    assert tf4.args == ((s + 3)*(s - 1), (s - 1)*(s + 5), s)

    tf4_ = TransferFunction(p + 2, p + 2, p)
    assert tf4_.args == (p + 2, p + 2, p)

    tf5 = TransferFunction(s - 1, 4 - p, s)
    assert tf5.args == (s - 1, 4 - p, s)

    tf5_ = TransferFunction(s - 1, s - 1, s)
    assert tf5_.args == (s - 1, s - 1, s)

    tf6 = TransferFunction(5, 6, s)
    assert tf6.num == 5
    assert tf6.den == 6
    assert tf6.args == (5, 6, s)

    tf6_ = TransferFunction(1/2, 4, s)
    assert tf6_.num == 0.5
    assert tf6_.den == 4
    assert tf6_.args == (0.500000000000000, 4, s)

    tf7 = TransferFunction(3*s**2 + 2*p + 4*s, 8*p**2 + 7*s, s)
    tf8 = TransferFunction(3*s**2 + 2*p + 4*s, 8*p**2 + 7*s, p)
    assert not tf7 == tf8

    tf7_ = TransferFunction(a0*s + a1*s**2 + a2*s**3, b0*p - b1*s, s)
    tf8_ = TransferFunction(a0*s + a1*s**2 + a2*s**3, b0*p - b1*s, s)
    assert tf7_ == tf8_
    assert -(-tf7_) == tf7_ == -(-(-(-tf7_)))

    tf9 = TransferFunction(a*s**3 + b*s**2 + g*s + d, d*p + g*p**2 + g*s, s)
    assert tf9.args == (a*s**3 + b*s**2 + d + g*s, d*p + g*p**2 + g*s, s)

    tf10 = TransferFunction(p**3 + d, g*s**2 + d*s + a, p)
    tf10_ = TransferFunction(p**3 + d, g*s**2 + d*s + a, p)
    assert tf10.args == (d + p**3, a + d*s + g*s**2, p)
    assert tf10_ == tf10

    tf11 = TransferFunction(a1*s + a0, b2*s**2 + b1*s + b0, s)
    assert tf11.num == (a0 + a1*s)
    assert tf11.den == (b0 + b1*s + b2*s**2)
    assert tf11.args == (a0 + a1*s, b0 + b1*s + b2*s**2, s)

    # when just the numerator is 0, leave the denominator alone.
    tf12 = TransferFunction(0, p**2 - p + 1, p)
    assert tf12.args == (0, p**2 - p + 1, p)

    tf13 = TransferFunction(0, 1, s)
    assert tf13.args == (0, 1, s)

    # float exponents
    tf14 = TransferFunction(a0*s**0.5 + a2*s**0.6 - a1, a1*p**(-8.7), s)
    assert tf14.args == (a0*s**0.5 - a1 + a2*s**0.6, a1*p**(-8.7), s)

    tf15 = TransferFunction(a2**2*p**(1/4) + a1*s**(-4/5), a0*s - p, p)
    assert tf15.args == (a1*s**(-0.8) + a2**2*p**0.25, a0*s - p, p)

    omega_o, k_p, k_o, k_i = symbols('omega_o, k_p, k_o, k_i')
    tf18 = TransferFunction((k_p + k_o*s + k_i/s), s**2 + 2*omega_o*s + omega_o**2, s)
    assert tf18.num == k_i/s + k_o*s + k_p
    assert tf18.args == (k_i/s + k_o*s + k_p, omega_o**2 + 2*omega_o*s + s**2, s)

    # ValueError when denominator is zero.
    raises(ValueError, lambda: TransferFunction(4, 0, s))
    raises(ValueError, lambda: TransferFunction(s, 0, s))
    raises(ValueError, lambda: TransferFunction(0, 0, s))

    raises(TypeError, lambda: TransferFunction(Matrix([1, 2, 3]), s, s))

    raises(TypeError, lambda: TransferFunction(s**2 + 2*s - 1, s + 3, 3))
    raises(TypeError, lambda: TransferFunction(p + 1, 5 - p, 4))
    raises(TypeError, lambda: TransferFunction(3, 4, 8))


def test_TransferFunction_functions():
    # classmethod from_rational_expression
    expr_1 = Mul(0, Pow(s, -1, evaluate=False), evaluate=False)
    expr_2 = s/0
    expr_3 = (p*s**2 + 5*s)/(s + 1)**3
    expr_4 = 6
    expr_5 = ((2 + 3*s)*(5 + 2*s))/((9 + 3*s)*(5 + 2*s**2))
    expr_6 = (9*s**4 + 4*s**2 + 8)/((s + 1)*(s + 9))
    tf = TransferFunction(s + 1, s**2 + 2, s)
    delay = exp(-s/tau)
    expr_7 = delay*tf.to_expr()
    H1 = TransferFunction.from_rational_expression(expr_7, s)
    H2 = TransferFunction(s + 1, (s**2 + 2)*exp(s/tau), s)
    expr_8 = Add(2,  3*s/(s**2 + 1), evaluate=False)

    assert TransferFunction.from_rational_expression(expr_1) == TransferFunction(0, s, s)
    raises(ZeroDivisionError, lambda: TransferFunction.from_rational_expression(expr_2))
    raises(ValueError, lambda: TransferFunction.from_rational_expression(expr_3))
    assert TransferFunction.from_rational_expression(expr_3, s) == TransferFunction((p*s**2 + 5*s), (s + 1)**3, s)
    assert TransferFunction.from_rational_expression(expr_3, p) == TransferFunction((p*s**2 + 5*s), (s + 1)**3, p)
    raises(ValueError, lambda: TransferFunction.from_rational_expression(expr_4))
    assert TransferFunction.from_rational_expression(expr_4, s) == TransferFunction(6, 1, s)
    assert TransferFunction.from_rational_expression(expr_5, s) == \
        TransferFunction((2 + 3*s)*(5 + 2*s), (9 + 3*s)*(5 + 2*s**2), s)
    assert TransferFunction.from_rational_expression(expr_6, s) == \
        TransferFunction((9*s**4 + 4*s**2 + 8), (s + 1)*(s + 9), s)
    assert H1 == H2
    assert TransferFunction.from_rational_expression(expr_8, s) == \
        TransferFunction(2*s**2 + 3*s + 2, s**2 + 1, s)

    # classmethod from_coeff_lists
    tf1 = TransferFunction.from_coeff_lists([1, 2], [3, 4, 5], s)
    num2 = [p**2, 2*p]
    den2 = [p**3, p + 1, 4]
    tf2 = TransferFunction.from_coeff_lists(num2, den2, s)
    num3 = [1, 2, 3]
    den3 = [0, 0]

    assert tf1 == TransferFunction(s + 2, 3*s**2 + 4*s + 5, s)
    assert tf2 == TransferFunction(p**2*s + 2*p, p**3*s**2 + s*(p + 1) + 4, s)
    raises(ZeroDivisionError, lambda: TransferFunction.from_coeff_lists(num3, den3, s))

    # classmethod from_zpk
    zeros = [4]
    poles = [-1+2j, -1-2j]
    gain = 3
    tf1 = TransferFunction.from_zpk(zeros, poles, gain, s)

    assert tf1 == TransferFunction(3*s - 12, (s + 1.0 - 2.0*I)*(s + 1.0 + 2.0*I), s)

    # explicitly cancel poles and zeros.
    tf0 = TransferFunction(s**5 + s**3 + s, s - s**2, s)
    a = TransferFunction(-(s**4 + s**2 + 1), s - 1, s)
    assert tf0.simplify() == simplify(tf0) == a

    tf1 = TransferFunction((p + 3)*(p - 1), (p - 1)*(p + 5), p)
    b = TransferFunction(p + 3, p + 5, p)
    assert tf1.simplify() == simplify(tf1) == b

    # expand the numerator and the denominator.
    G1 = TransferFunction((1 - s)**2, (s**2 + 1)**2, s)
    G2 = TransferFunction(1, -3, p)
    c = (a2*s**p + a1*s**s + a0*p**p)*(p**s + s**p)
    d = (b0*s**s + b1*p**s)*(b2*s*p + p**p)
    e = a0*p**p*p**s + a0*p**p*s**p + a1*p**s*s**s + a1*s**p*s**s + a2*p**s*s**p + a2*s**(2*p)
    f = b0*b2*p*s*s**s + b0*p**p*s**s + b1*b2*p*p**s*s + b1*p**p*p**s
    g = a1*a2*s*s**p + a1*p*s + a2*b1*p*s*s**p + b1*p**2*s
    G3 = TransferFunction(c, d, s)
    G4 = TransferFunction(a0*s**s - b0*p**p, (a1*s + b1*s*p)*(a2*s**p + p), p)

    assert G1.expand() == TransferFunction(s**2 - 2*s + 1, s**4 + 2*s**2 + 1, s)
    assert tf1.expand() == TransferFunction(p**2 + 2*p - 3, p**2 + 4*p - 5, p)
    assert G2.expand() == G2
    assert G3.expand() == TransferFunction(e, f, s)
    assert G4.expand() == TransferFunction(a0*s**s - b0*p**p, g, p)

    # purely symbolic polynomials.
    p1 = a1*s + a0
    p2 = b2*s**2 + b1*s + b0
    SP1 = TransferFunction(p1, p2, s)
    expect1 = TransferFunction(2.0*s + 1.0, 5.0*s**2 + 4.0*s + 3.0, s)
    expect1_ = TransferFunction(2*s + 1, 5*s**2 + 4*s + 3, s)
    assert SP1.subs({a0: 1, a1: 2, b0: 3, b1: 4, b2: 5}) == expect1_
    assert SP1.subs({a0: 1, a1: 2, b0: 3, b1: 4, b2: 5}).evalf() == expect1
    assert expect1_.evalf() == expect1

    c1, d0, d1, d2 = symbols('c1, d0:3')
    p3, p4 = c1*p, d2*p**3 + d1*p**2 - d0
    SP2 = TransferFunction(p3, p4, p)
    expect2 = TransferFunction(2.0*p, 5.0*p**3 + 2.0*p**2 - 3.0, p)
    expect2_ = TransferFunction(2*p, 5*p**3 + 2*p**2 - 3, p)
    assert SP2.subs({c1: 2, d0: 3, d1: 2, d2: 5}) == expect2_
    assert SP2.subs({c1: 2, d0: 3, d1: 2, d2: 5}).evalf() == expect2
    assert expect2_.evalf() == expect2

    SP3 = TransferFunction(a0*p**3 + a1*s**2 - b0*s + b1, a1*s + p, s)
    expect3 = TransferFunction(2.0*p**3 + 4.0*s**2 - s + 5.0, p + 4.0*s, s)
    expect3_ = TransferFunction(2*p**3 + 4*s**2 - s + 5, p + 4*s, s)
    assert SP3.subs({a0: 2, a1: 4, b0: 1, b1: 5}) == expect3_
    assert SP3.subs({a0: 2, a1: 4, b0: 1, b1: 5}).evalf() == expect3
    assert expect3_.evalf() == expect3

    SP4 = TransferFunction(s - a1*p**3, a0*s + p, p)
    expect4 = TransferFunction(7.0*p**3 + s, p - s, p)
    expect4_ = TransferFunction(7*p**3 + s, p - s, p)
    assert SP4.subs({a0: -1, a1: -7}) == expect4_
    assert SP4.subs({a0: -1, a1: -7}).evalf() == expect4
    assert expect4_.evalf() == expect4

    # evaluate the transfer function at particular frequencies.
    assert tf1.eval_frequency(wn) == wn**2/(wn**2 + 4*wn - 5) + 2*wn/(wn**2 + 4*wn - 5) - 3/(wn**2 + 4*wn - 5)
    assert G1.eval_frequency(1 + I) == S(3)/25 + S(4)*I/25
    assert G4.eval_frequency(S(5)/3) == \
        a0*s**s/(a1*a2*s**(S(8)/3) + S(5)*a1*s/3 + 5*a2*b1*s**(S(8)/3)/3 + S(25)*b1*s/9) - 5*3**(S(1)/3)*5**(S(2)/3)*b0/(9*a1*a2*s**(S(8)/3) + 15*a1*s + 15*a2*b1*s**(S(8)/3) + 25*b1*s)

    # Low-frequency (or DC) gain.
    assert tf0.dc_gain() == 1
    assert tf1.dc_gain() == Rational(3, 5)
    assert SP2.dc_gain() == 0
    assert expect4.dc_gain() == -1
    assert expect2_.dc_gain() == 0
    assert TransferFunction(1, s, s).dc_gain() == oo

    # Poles of a transfer function.
    tf_ = TransferFunction(x**3 - k, k, x)
    _tf = TransferFunction(k, x**4 - k, x)
    TF_ = TransferFunction(x**2, x**10 + x + x**2, x)
    _TF = TransferFunction(x**10 + x + x**2, x**2, x)
    assert G1.poles() == [I, I, -I, -I]
    assert G2.poles() == []
    assert tf1.poles() == [-5, 1]
    assert expect4_.poles() == [s]
    assert SP4.poles() == [-a0*s]
    assert expect3.poles() == [-0.25*p]
    assert str(expect2.poles()) == str([0.729001428685125, -0.564500714342563 - 0.710198984796332*I, -0.564500714342563 + 0.710198984796332*I])
    assert str(expect1.poles()) == str([-0.4 - 0.66332495807108*I, -0.4 + 0.66332495807108*I])
    assert _tf.poles() == [k**(Rational(1, 4)), -k**(Rational(1, 4)), I*k**(Rational(1, 4)), -I*k**(Rational(1, 4))]
    assert TF_.poles() == [CRootOf(x**9 + x + 1, 0), 0, CRootOf(x**9 + x + 1, 1), CRootOf(x**9 + x + 1, 2),
        CRootOf(x**9 + x + 1, 3), CRootOf(x**9 + x + 1, 4), CRootOf(x**9 + x + 1, 5), CRootOf(x**9 + x + 1, 6),
        CRootOf(x**9 + x + 1, 7), CRootOf(x**9 + x + 1, 8)]
    raises(NotImplementedError, lambda: TransferFunction(x**2, a0*x**10 + x + x**2, x).poles())

    # Stability of a transfer function.
    q, r = symbols('q, r', negative=True)
    t = symbols('t', positive=True)
    TF_ = TransferFunction(s**2 + a0 - a1*p, q*s - r, s)
    stable_tf = TransferFunction(s**2 + a0 - a1*p, q*s - 1, s)
    stable_tf_ = TransferFunction(s**2 + a0 - a1*p, q*s - t, s)

    assert G1.is_stable() is False
    assert G2.is_stable() is True
    assert tf1.is_stable() is False   # as one pole is +ve, and the other is -ve.
    assert expect2.is_stable() is False
    assert expect1.is_stable() is True
    assert stable_tf.is_stable() is True
    assert stable_tf_.is_stable() is True
    assert TF_.is_stable() is False
    assert expect4_.is_stable() is None   # no assumption provided for the only pole 's'.
    assert SP4.is_stable() is None

    # Zeros of a transfer function.
    assert G1.zeros() == [1, 1]
    assert G2.zeros() == []
    assert tf1.zeros() == [-3, 1]
    assert expect4_.zeros() == [7**(Rational(2, 3))*(-s)**(Rational(1, 3))/7, -7**(Rational(2, 3))*(-s)**(Rational(1, 3))/14 -
        sqrt(3)*7**(Rational(2, 3))*I*(-s)**(Rational(1, 3))/14, -7**(Rational(2, 3))*(-s)**(Rational(1, 3))/14 + sqrt(3)*7**(Rational(2, 3))*I*(-s)**(Rational(1, 3))/14]
    assert SP4.zeros() == [(s/a1)**(Rational(1, 3)), -(s/a1)**(Rational(1, 3))/2 - sqrt(3)*I*(s/a1)**(Rational(1, 3))/2,
        -(s/a1)**(Rational(1, 3))/2 + sqrt(3)*I*(s/a1)**(Rational(1, 3))/2]
    assert str(expect3.zeros()) == str([0.125 - 1.11102430216445*sqrt(-0.405063291139241*p**3 - 1.0),
        1.11102430216445*sqrt(-0.405063291139241*p**3 - 1.0) + 0.125])
    assert tf_.zeros() == [k**(Rational(1, 3)), -k**(Rational(1, 3))/2 - sqrt(3)*I*k**(Rational(1, 3))/2,
        -k**(Rational(1, 3))/2 + sqrt(3)*I*k**(Rational(1, 3))/2]
    assert _TF.zeros() == [CRootOf(x**9 + x + 1, 0), 0, CRootOf(x**9 + x + 1, 1), CRootOf(x**9 + x + 1, 2),
        CRootOf(x**9 + x + 1, 3), CRootOf(x**9 + x + 1, 4), CRootOf(x**9 + x + 1, 5), CRootOf(x**9 + x + 1, 6),
        CRootOf(x**9 + x + 1, 7), CRootOf(x**9 + x + 1, 8)]
    raises(NotImplementedError, lambda: TransferFunction(a0*x**10 + x + x**2, x**2, x).zeros())

    # negation of TF.
    tf2 = TransferFunction(s + 3, s**2 - s**3 + 9, s)
    tf3 = TransferFunction(-3*p + 3, 1 - p, p)
    assert -tf2 == TransferFunction(-s - 3, s**2 - s**3 + 9, s)
    assert -tf3 == TransferFunction(3*p - 3, 1 - p, p)

    # taking power of a TF.
    tf4 = TransferFunction(p + 4, p - 3, p)
    tf5 = TransferFunction(s**2 + 1, 1 - s, s)
    expect2 = TransferFunction((s**2 + 1)**3, (1 - s)**3, s)
    expect1 = TransferFunction((p + 4)**2, (p - 3)**2, p)
    assert (tf4*tf4).doit() == tf4**2 == pow(tf4, 2) == expect1
    assert (tf5*tf5*tf5).doit() == tf5**3 == pow(tf5, 3) == expect2
    assert tf5**0 == pow(tf5, 0) == TransferFunction(1, 1, s)
    assert Series(tf4).doit()**-1 == tf4**-1 == pow(tf4, -1) == TransferFunction(p - 3, p + 4, p)
    assert (tf5*tf5).doit()**-1 == tf5**-2 == pow(tf5, -2) == TransferFunction((1 - s)**2, (s**2 + 1)**2, s)

    raises(ValueError, lambda: tf4**(s**2 + s - 1))
    raises(ValueError, lambda: tf5**s)
    raises(ValueError, lambda: tf4**tf5)

    # SymPy's own functions.
    tf = TransferFunction(s - 1, s**2 - 2*s + 1, s)
    tf6 = TransferFunction(s + p, p**2 - 5, s)
    assert factor(tf) == TransferFunction(s - 1, (s - 1)**2, s)
    assert tf.num.subs(s, 2) == tf.den.subs(s, 2) == 1
    # subs & xreplace
    assert tf.subs(s, 2) == TransferFunction(s - 1, s**2 - 2*s + 1, s)
    assert tf6.subs(p, 3) == TransferFunction(s + 3, 4, s)
    assert tf3.xreplace({p: s}) == TransferFunction(-3*s + 3, 1 - s, s)
    raises(TypeError, lambda: tf3.xreplace({p: exp(2)}))
    assert tf3.subs(p, exp(2)) == tf3

    tf7 = TransferFunction(a0*s**p + a1*p**s, a2*p - s, s)
    assert tf7.xreplace({s: k}) == TransferFunction(a0*k**p + a1*p**k, a2*p - k, k)
    assert tf7.subs(s, k) == TransferFunction(a0*s**p + a1*p**s, a2*p - s, s)

    # Conversion to Expr with to_expr()
    tf8 = TransferFunction(a0*s**5 + 5*s**2 + 3, s**6 - 3, s)
    tf9 = TransferFunction((5 + s), (5 + s)*(6 + s), s)
    tf10 = TransferFunction(0, 1, s)
    tf11 = TransferFunction(1, 1, s)
    assert tf8.to_expr() == Mul((a0*s**5 + 5*s**2 + 3), Pow((s**6 - 3), -1, evaluate=False), evaluate=False)
    assert tf9.to_expr() == Mul((s + 5), Pow((5 + s)*(6 + s), -1, evaluate=False), evaluate=False)
    assert tf10.to_expr() == Mul(S(0), Pow(1, -1, evaluate=False), evaluate=False)
    assert tf11.to_expr() == Pow(1, -1, evaluate=False)


def test_TransferFunction_addition_and_subtraction():
    tf1 = TransferFunction(s + 6, s - 5, s)
    tf2 = TransferFunction(s + 3, s + 1, s)
    tf3 = TransferFunction(s + 1, s**2 + s + 1, s)
    tf4 = TransferFunction(p, 2 - p, p)

    # addition
    assert tf1 + tf2 == Parallel(tf1, tf2)
    assert tf3 + tf1 == Parallel(tf3, tf1)
    assert -tf1 + tf2 + tf3 == Parallel(-tf1, tf2, tf3)
    assert tf1 + (tf2 + tf3) == Parallel(tf1, tf2, tf3)

    c = symbols("c", commutative=False)
    raises(ValueError, lambda: tf1 + Matrix([1, 2, 3]))
    raises(ValueError, lambda: tf2 + c)
    raises(ValueError, lambda: tf3 + tf4)
    raises(ValueError, lambda: tf1 + (s - 1))
    raises(ValueError, lambda: tf1 + 8)
    raises(ValueError, lambda: (1 - p**3) + tf1)

    # subtraction
    assert tf1 - tf2 == Parallel(tf1, -tf2)
    assert tf3 - tf2 == Parallel(tf3, -tf2)
    assert -tf1 - tf3 == Parallel(-tf1, -tf3)
    assert tf1 - tf2 + tf3 == Parallel(tf1, -tf2, tf3)

    raises(ValueError, lambda: tf1 - Matrix([1, 2, 3]))
    raises(ValueError, lambda: tf3 - tf4)
    raises(ValueError, lambda: tf1 - (s - 1))
    raises(ValueError, lambda: tf1 - 8)
    raises(ValueError, lambda: (s + 5) - tf2)
    raises(ValueError, lambda: (1 + p**4) - tf1)


def test_TransferFunction_multiplication_and_division():
    G1 = TransferFunction(s + 3, -s**3 + 9, s)
    G2 = TransferFunction(s + 1, s - 5, s)
    G3 = TransferFunction(p, p**4 - 6, p)
    G4 = TransferFunction(p + 4, p - 5, p)
    G5 = TransferFunction(s + 6, s - 5, s)
    G6 = TransferFunction(s + 3, s + 1, s)
    G7 = TransferFunction(1, 1, s)

    # multiplication
    assert G1*G2 == Series(G1, G2)
    assert -G1*G5 == Series(-G1, G5)
    assert -G2*G5*-G6 == Series(-G2, G5, -G6)
    assert -G1*-G2*-G5*-G6 == Series(-G1, -G2, -G5, -G6)
    assert G3*G4 == Series(G3, G4)
    assert (G1*G2)*-(G5*G6) == \
        Series(G1, G2, TransferFunction(-1, 1, s), Series(G5, G6))
    assert G1*G2*(G5 + G6) == Series(G1, G2, Parallel(G5, G6))

    # division - See ``test_Feedback_functions()`` for division by Parallel objects.
    assert G5/G6 == Series(G5, pow(G6, -1))
    assert -G3/G4 == Series(-G3, pow(G4, -1))
    assert (G5*G6)/G7 == Series(G5, G6, pow(G7, -1))

    c = symbols("c", commutative=False)
    raises(ValueError, lambda: G3 * Matrix([1, 2, 3]))
    raises(ValueError, lambda: G1 * c)
    raises(ValueError, lambda: G3 * G5)
    raises(ValueError, lambda: G5 * (s - 1))
    raises(ValueError, lambda: 9 * G5)

    raises(ValueError, lambda: G3 / Matrix([1, 2, 3]))
    raises(ValueError, lambda: G6 / 0)
    raises(ValueError, lambda: G3 / G5)
    raises(ValueError, lambda: G5 / 2)
    raises(ValueError, lambda: G5 / s**2)
    raises(ValueError, lambda: (s - 4*s**2) / G2)
    raises(ValueError, lambda: 0 / G4)
    raises(ValueError, lambda: G7 / (1 + G6))
    raises(ValueError, lambda: G7 / (G5 * G6))
    raises(ValueError, lambda: G7 / (G7 + (G5 + G6)))


def test_TransferFunction_is_proper():
    omega_o, zeta, tau = symbols('omega_o, zeta, tau')
    G1 = TransferFunction(omega_o**2, s**2 + p*omega_o*zeta*s + omega_o**2, omega_o)
    G2 = TransferFunction(tau - s**3, tau + p**4, tau)
    G3 = TransferFunction(a*b*s**3 + s**2 - a*p + s, b - s*p**2, p)
    G4 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)
    assert G1.is_proper
    assert G2.is_proper
    assert G3.is_proper
    assert not G4.is_proper


def test_TransferFunction_is_strictly_proper():
    omega_o, zeta, tau = symbols('omega_o, zeta, tau')
    tf1 = TransferFunction(omega_o**2, s**2 + p*omega_o*zeta*s + omega_o**2, omega_o)
    tf2 = TransferFunction(tau - s**3, tau + p**4, tau)
    tf3 = TransferFunction(a*b*s**3 + s**2 - a*p + s, b - s*p**2, p)
    tf4 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)
    assert not tf1.is_strictly_proper
    assert not tf2.is_strictly_proper
    assert tf3.is_strictly_proper
    assert not tf4.is_strictly_proper


def test_TransferFunction_is_biproper():
    tau, omega_o, zeta = symbols('tau, omega_o, zeta')
    tf1 = TransferFunction(omega_o**2, s**2 + p*omega_o*zeta*s + omega_o**2, omega_o)
    tf2 = TransferFunction(tau - s**3, tau + p**4, tau)
    tf3 = TransferFunction(a*b*s**3 + s**2 - a*p + s, b - s*p**2, p)
    tf4 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)
    assert tf1.is_biproper
    assert tf2.is_biproper
    assert not tf3.is_biproper
    assert not tf4.is_biproper


def test_PIDController():
    kp, ki, kd, tf = symbols("kp ki kd tf")
    p1 = PIDController(kp, ki, kd, tf)
    p2 = PIDController()

    # Type Checking
    assert isinstance(p1, PIDController)
    assert isinstance(p1, TransferFunction)

    # Properties checking
    assert p1 == PIDController(kp, ki, kd, tf, s)
    assert p2 == PIDController(kp, ki, kd, 0, s)
    assert p1.num == kd*s**2 + ki*s*tf + ki + kp*s**2*tf + kp*s
    assert p1.den == s**2*tf + s
    assert p1.var == s
    assert p1.kp == kp
    assert p1.ki == ki
    assert p1.kd == kd
    assert p1.tf == tf

    # Functionality checking
    assert p1.doit() == TransferFunction(kd*s**2 + ki*s*tf + ki + kp*s**2*tf + kp*s, s**2*tf + s, s)
    assert p1.is_proper == True
    assert p1.is_biproper == True
    assert p1.is_strictly_proper == False
    assert p2.doit() == TransferFunction(kd*s**2 + ki + kp*s, s, s)

    # Using PIDController with TransferFunction
    tf1 = TransferFunction(s, s + 1, s)
    par1 = Parallel(p1, tf1)
    ser1 = Series(p1, tf1)
    fed1 = Feedback(p1, tf1)
    assert par1 == Parallel(PIDController(kp, ki, kd, tf, s), TransferFunction(s, s + 1, s))
    assert ser1 == Series(PIDController(kp, ki, kd, tf, s), TransferFunction(s, s + 1, s))
    assert fed1 == Feedback(PIDController(kp, ki, kd, tf, s), TransferFunction(s, s + 1, s))
    assert par1.doit() == TransferFunction(s*(s**2*tf + s) + (s + 1)*(kd*s**2 + ki*s*tf + ki + kp*s**2*tf + kp*s),
                                           (s + 1)*(s**2*tf + s), s)
    assert ser1.doit() == TransferFunction(s*(kd*s**2 + ki*s*tf + ki + kp*s**2*tf + kp*s),
                                           (s + 1)*(s**2*tf + s), s)
    assert fed1.doit() == TransferFunction((s + 1)*(s**2*tf + s)*(kd*s**2 + ki*s*tf + ki + kp*s**2*tf + kp*s),
                                           (s*(kd*s**2 + ki*s*tf + ki + kp*s**2*tf + kp*s) + (s + 1)*(s**2*tf + s))*(s**2*tf + s), s)


def test_Series_construction():
    tf = TransferFunction(a0*s**3 + a1*s**2 - a2*s, b0*p**4 + b1*p**3 - b2*s*p, s)
    tf2 = TransferFunction(a2*p - s, a2*s + p, s)
    tf3 = TransferFunction(a0*p + p**a1 - s, p, p)
    tf4 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    inp = Function('X_d')(s)
    out = Function('X')(s)

    s0 = Series(tf, tf2)
    assert s0.args == (tf, tf2)
    assert s0.var == s

    s1 = Series(Parallel(tf, -tf2), tf2)
    assert s1.args == (Parallel(tf, -tf2), tf2)
    assert s1.var == s

    tf3_ = TransferFunction(inp, 1, s)
    tf4_ = TransferFunction(-out, 1, s)
    s2 = Series(tf, Parallel(tf3_, tf4_), tf2)
    assert s2.args == (tf, Parallel(tf3_, tf4_), tf2)

    s3 = Series(tf, tf2, tf4)
    assert s3.args == (tf, tf2, tf4)

    s4 = Series(tf3_, tf4_)
    assert s4.args == (tf3_, tf4_)
    assert s4.var == s

    s6 = Series(tf2, tf4, Parallel(tf2, -tf), tf4)
    assert s6.args == (tf2, tf4, Parallel(tf2, -tf), tf4)

    s7 = Series(tf, tf2)
    assert s0 == s7
    assert not s0 == s2

    raises(ValueError, lambda: Series(tf, tf3))
    raises(ValueError, lambda: Series(tf, tf2, tf3, tf4))
    raises(ValueError, lambda: Series(-tf3, tf2))
    raises(TypeError, lambda: Series(2, tf, tf4))
    raises(TypeError, lambda: Series(s**2 + p*s, tf3, tf2))
    raises(TypeError, lambda: Series(tf3, Matrix([1, 2, 3, 4])))


def test_MIMOSeries_construction():
    tf_1 = TransferFunction(a0*s**3 + a1*s**2 - a2*s, b0*p**4 + b1*p**3 - b2*s*p, s)
    tf_2 = TransferFunction(a2*p - s, a2*s + p, s)
    tf_3 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)

    tfm_1 = TransferFunctionMatrix([[tf_1, tf_2, tf_3], [-tf_3, -tf_2, tf_1]])
    tfm_2 = TransferFunctionMatrix([[-tf_2], [-tf_2], [-tf_3]])
    tfm_3 = TransferFunctionMatrix([[-tf_3]])
    tfm_4 = TransferFunctionMatrix([[TF3], [TF2], [-TF1]])
    tfm_5 = TransferFunctionMatrix.from_Matrix(Matrix([1/p]), p)

    s8 = MIMOSeries(tfm_2, tfm_1)
    assert s8.args == (tfm_2, tfm_1)
    assert s8.var == s
    assert s8.shape == (s8.num_outputs, s8.num_inputs) == (2, 1)

    s9 = MIMOSeries(tfm_3, tfm_2, tfm_1)
    assert s9.args == (tfm_3, tfm_2, tfm_1)
    assert s9.var == s
    assert s9.shape == (s9.num_outputs, s9.num_inputs) == (2, 1)

    s11 = MIMOSeries(tfm_3, MIMOParallel(-tfm_2, -tfm_4), tfm_1)
    assert s11.args == (tfm_3, MIMOParallel(-tfm_2, -tfm_4), tfm_1)
    assert s11.shape == (s11.num_outputs, s11.num_inputs) == (2, 1)

    # arg cannot be empty tuple.
    raises(ValueError, lambda: MIMOSeries())

    # arg cannot contain SISO as well as MIMO systems.
    raises(TypeError, lambda: MIMOSeries(tfm_1, tf_1))

    # for all the adjacent transfer function matrices:
    # no. of inputs of first TFM must be equal to the no. of outputs of the second TFM.
    raises(ValueError, lambda: MIMOSeries(tfm_1, tfm_2, -tfm_1))

    # all the TFMs must use the same complex variable.
    raises(ValueError, lambda: MIMOSeries(tfm_3, tfm_5))

    # Number or expression not allowed in the arguments.
    raises(TypeError, lambda: MIMOSeries(2, tfm_2, tfm_3))
    raises(TypeError, lambda: MIMOSeries(s**2 + p*s, -tfm_2, tfm_3))
    raises(TypeError, lambda: MIMOSeries(Matrix([1/p]), tfm_3))


def test_Series_functions():
    tf1 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    tf2 = TransferFunction(k, 1, s)
    tf3 = TransferFunction(a2*p - s, a2*s + p, s)
    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)

    assert tf1*tf2*tf3 == Series(tf1, tf2, tf3) == Series(Series(tf1, tf2), tf3) \
        == Series(tf1, Series(tf2, tf3))
    assert tf1*(tf2 + tf3) == Series(tf1, Parallel(tf2, tf3))
    assert tf1*tf2 + tf5 == Parallel(Series(tf1, tf2), tf5)
    assert tf1*tf2 - tf5 == Parallel(Series(tf1, tf2), -tf5)
    assert tf1*tf2 + tf3 + tf5 == Parallel(Series(tf1, tf2), tf3, tf5)
    assert tf1*tf2 - tf3 - tf5 == Parallel(Series(tf1, tf2), -tf3, -tf5)
    assert tf1*tf2 - tf3 + tf5 == Parallel(Series(tf1, tf2), -tf3, tf5)
    assert tf1*tf2 + tf3*tf5 == Parallel(Series(tf1, tf2), Series(tf3, tf5))
    assert tf1*tf2 - tf3*tf5 == Parallel(Series(tf1, tf2), Series(TransferFunction(-1, 1, s), Series(tf3, tf5)))
    assert tf2*tf3*(tf2 - tf1)*tf3 == Series(tf2, tf3, Parallel(tf2, -tf1), tf3)
    assert -tf1*tf2 == Series(-tf1, tf2)
    assert -(tf1*tf2) == Series(TransferFunction(-1, 1, s), Series(tf1, tf2))
    raises(ValueError, lambda: tf1*tf2*tf4)
    raises(ValueError, lambda: tf1*(tf2 - tf4))
    raises(ValueError, lambda: tf3*Matrix([1, 2, 3]))

    # evaluate=True -> doit()
    assert Series(tf1, tf2, evaluate=True) == Series(tf1, tf2).doit() == \
        TransferFunction(k, s**2 + 2*s*wn*zeta + wn**2, s)
    assert Series(tf1, tf2, Parallel(tf1, -tf3), evaluate=True) == Series(tf1, tf2, Parallel(tf1, -tf3)).doit() == \
        TransferFunction(k*(a2*s + p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2)), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2)**2, s)
    assert Series(tf2, tf1, -tf3, evaluate=True) == Series(tf2, tf1, -tf3).doit() == \
        TransferFunction(k*(-a2*p + s), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert not Series(tf1, -tf2, evaluate=False) == Series(tf1, -tf2).doit()

    assert Series(Parallel(tf1, tf2), Parallel(tf2, -tf3)).doit() == \
        TransferFunction((k*(s**2 + 2*s*wn*zeta + wn**2) + 1)*(-a2*p + k*(a2*s + p) + s), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert Series(-tf1, -tf2, -tf3).doit() == \
        TransferFunction(k*(-a2*p + s), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert -Series(tf1, tf2, tf3).doit() == \
        TransferFunction(-k*(a2*p - s), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert Series(tf2, tf3, Parallel(tf2, -tf1), tf3).doit() == \
        TransferFunction(k*(a2*p - s)**2*(k*(s**2 + 2*s*wn*zeta + wn**2) - 1), (a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2), s)

    assert Series(tf1, tf2).rewrite(TransferFunction) == TransferFunction(k, s**2 + 2*s*wn*zeta + wn**2, s)
    assert Series(tf2, tf1, -tf3).rewrite(TransferFunction) == \
        TransferFunction(k*(-a2*p + s), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)

    S1 = Series(Parallel(tf1, tf2), Parallel(tf2, -tf3))
    assert S1.is_proper
    assert not S1.is_strictly_proper
    assert S1.is_biproper

    S2 = Series(tf1, tf2, tf3)
    assert S2.is_proper
    assert S2.is_strictly_proper
    assert not S2.is_biproper

    S3 = Series(tf1, -tf2, Parallel(tf1, -tf3))
    assert S3.is_proper
    assert S3.is_strictly_proper
    assert not S3.is_biproper


def test_MIMOSeries_functions():
    tfm1 = TransferFunctionMatrix([[TF1, TF2, TF3], [-TF3, -TF2, TF1]])
    tfm2 = TransferFunctionMatrix([[-TF1], [-TF2], [-TF3]])
    tfm3 = TransferFunctionMatrix([[-TF1]])
    tfm4 = TransferFunctionMatrix([[-TF2, -TF3], [-TF1, TF2]])
    tfm5 = TransferFunctionMatrix([[TF2, -TF2], [-TF3, -TF2]])
    tfm6 = TransferFunctionMatrix([[-TF3], [TF1]])
    tfm7 = TransferFunctionMatrix([[TF1], [-TF2]])

    assert tfm1*tfm2 + tfm6 == MIMOParallel(MIMOSeries(tfm2, tfm1), tfm6)
    assert tfm1*tfm2 + tfm7 + tfm6 == MIMOParallel(MIMOSeries(tfm2, tfm1), tfm7, tfm6)
    assert tfm1*tfm2 - tfm6 - tfm7 == MIMOParallel(MIMOSeries(tfm2, tfm1), -tfm6, -tfm7)
    assert tfm4*tfm5 + (tfm4 - tfm5) == MIMOParallel(MIMOSeries(tfm5, tfm4), tfm4, -tfm5)
    assert tfm4*-tfm6 + (-tfm4*tfm6) == MIMOParallel(MIMOSeries(-tfm6, tfm4), MIMOSeries(tfm6, -tfm4))

    raises(ValueError, lambda: tfm1*tfm2 + TF1)
    raises(TypeError, lambda: tfm1*tfm2 + a0)
    raises(TypeError, lambda: tfm4*tfm6 - (s - 1))
    raises(TypeError, lambda: tfm4*-tfm6 - 8)
    raises(TypeError, lambda: (-1 + p**5) + tfm1*tfm2)

    # Shape criteria.

    raises(TypeError, lambda: -tfm1*tfm2 + tfm4)
    raises(TypeError, lambda: tfm1*tfm2 - tfm4 + tfm5)
    raises(TypeError, lambda: tfm1*tfm2 - tfm4*tfm5)

    assert tfm1*tfm2*-tfm3 == MIMOSeries(-tfm3, tfm2, tfm1)
    assert (tfm1*-tfm2)*tfm3 == MIMOSeries(tfm3, -tfm2, tfm1)

    # Multiplication of a Series object with a SISO TF not allowed.

    raises(ValueError, lambda: tfm4*tfm5*TF1)
    raises(TypeError, lambda: tfm4*tfm5*a1)
    raises(TypeError, lambda: tfm4*-tfm5*(s - 2))
    raises(TypeError, lambda: tfm5*tfm4*9)
    raises(TypeError, lambda: (-p**3 + 1)*tfm5*tfm4)

    # Transfer function matrix in the arguments.
    assert (MIMOSeries(tfm2, tfm1, evaluate=True) == MIMOSeries(tfm2, tfm1).doit()
        == TransferFunctionMatrix(((TransferFunction(-k**2*(a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2 + (-a2*p + s)*(a2*p - s)*(s**2 + 2*s*wn*zeta + wn**2)**2 - (a2*s + p)**2,
        (a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2, s),),
        (TransferFunction(k**2*(a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2 + (-a2*p + s)*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) + (a2*p - s)*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2),
        (a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2, s),))))

    # doit() should not cancel poles and zeros.
    mat_1 = Matrix([[1/(1+s), (1+s)/(1+s**2+2*s)**3]])
    mat_2 = Matrix([[(1+s)], [(1+s**2+2*s)**3/(1+s)]])
    tm_1, tm_2 = TransferFunctionMatrix.from_Matrix(mat_1, s), TransferFunctionMatrix.from_Matrix(mat_2, s)
    assert (MIMOSeries(tm_2, tm_1).doit()
        == TransferFunctionMatrix(((TransferFunction(2*(s + 1)**2*(s**2 + 2*s + 1)**3, (s + 1)**2*(s**2 + 2*s + 1)**3, s),),)))
    assert MIMOSeries(tm_2, tm_1).doit().simplify() == TransferFunctionMatrix(((TransferFunction(2, 1, s),),))

    # calling doit() will expand the internal Series and Parallel objects.
    assert (MIMOSeries(-tfm3, -tfm2, tfm1, evaluate=True)
        == MIMOSeries(-tfm3, -tfm2, tfm1).doit()
        == TransferFunctionMatrix(((TransferFunction(k**2*(a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2 + (a2*p - s)**2*(s**2 + 2*s*wn*zeta + wn**2)**2 + (a2*s + p)**2,
        (a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**3, s),),
        (TransferFunction(-k**2*(a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2 + (-a2*p + s)*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) + (a2*p - s)*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2),
        (a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**3, s),))))
    assert (MIMOSeries(MIMOParallel(tfm4, tfm5), tfm5, evaluate=True)
        == MIMOSeries(MIMOParallel(tfm4, tfm5), tfm5).doit()
        == TransferFunctionMatrix(((TransferFunction(-k*(-a2*s - p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2)), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s), TransferFunction(k*(-a2*p - \
            k*(a2*s + p) + s), a2*s + p, s)), (TransferFunction(-k*(-a2*s - p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2)), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s), \
            TransferFunction((-a2*p + s)*(-a2*p - k*(a2*s + p) + s), (a2*s + p)**2, s)))) == MIMOSeries(MIMOParallel(tfm4, tfm5), tfm5).rewrite(TransferFunctionMatrix))


def test_Parallel_construction():
    tf = TransferFunction(a0*s**3 + a1*s**2 - a2*s, b0*p**4 + b1*p**3 - b2*s*p, s)
    tf2 = TransferFunction(a2*p - s, a2*s + p, s)
    tf3 = TransferFunction(a0*p + p**a1 - s, p, p)
    tf4 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    inp = Function('X_d')(s)
    out = Function('X')(s)

    p0 = Parallel(tf, tf2)
    assert p0.args == (tf, tf2)
    assert p0.var == s

    p1 = Parallel(Series(tf, -tf2), tf2)
    assert p1.args == (Series(tf, -tf2), tf2)
    assert p1.var == s

    tf3_ = TransferFunction(inp, 1, s)
    tf4_ = TransferFunction(-out, 1, s)
    p2 = Parallel(tf, Series(tf3_, -tf4_), tf2)
    assert p2.args == (tf, Series(tf3_, -tf4_), tf2)

    p3 = Parallel(tf, tf2, tf4)
    assert p3.args == (tf, tf2, tf4)

    p4 = Parallel(tf3_, tf4_)
    assert p4.args == (tf3_, tf4_)
    assert p4.var == s

    p5 = Parallel(tf, tf2)
    assert p0 == p5
    assert not p0 == p1

    p6 = Parallel(tf2, tf4, Series(tf2, -tf4))
    assert p6.args == (tf2, tf4, Series(tf2, -tf4))

    p7 = Parallel(tf2, tf4, Series(tf2, -tf), tf4)
    assert p7.args == (tf2, tf4, Series(tf2, -tf), tf4)

    raises(ValueError, lambda: Parallel(tf, tf3))
    raises(ValueError, lambda: Parallel(tf, tf2, tf3, tf4))
    raises(ValueError, lambda: Parallel(-tf3, tf4))
    raises(TypeError, lambda: Parallel(2, tf, tf4))
    raises(TypeError, lambda: Parallel(s**2 + p*s, tf3, tf2))
    raises(TypeError, lambda: Parallel(tf3, Matrix([1, 2, 3, 4])))


def test_MIMOParallel_construction():
    tfm1 = TransferFunctionMatrix([[TF1], [TF2], [TF3]])
    tfm2 = TransferFunctionMatrix([[-TF3], [TF2], [TF1]])
    tfm3 = TransferFunctionMatrix([[TF1]])
    tfm4 = TransferFunctionMatrix([[TF2], [TF1], [TF3]])
    tfm5 = TransferFunctionMatrix([[TF1, TF2], [TF2, TF1]])
    tfm6 = TransferFunctionMatrix([[TF2, TF1], [TF1, TF2]])
    tfm7 = TransferFunctionMatrix.from_Matrix(Matrix([[1/p]]), p)

    p8 = MIMOParallel(tfm1, tfm2)
    assert p8.args == (tfm1, tfm2)
    assert p8.var == s
    assert p8.shape == (p8.num_outputs, p8.num_inputs) == (3, 1)

    p9 = MIMOParallel(MIMOSeries(tfm3, tfm1), tfm2)
    assert p9.args == (MIMOSeries(tfm3, tfm1), tfm2)
    assert p9.var == s
    assert p9.shape == (p9.num_outputs, p9.num_inputs) == (3, 1)

    p10 = MIMOParallel(tfm1, MIMOSeries(tfm3, tfm4), tfm2)
    assert p10.args == (tfm1, MIMOSeries(tfm3, tfm4), tfm2)
    assert p10.var == s
    assert p10.shape == (p10.num_outputs, p10.num_inputs) == (3, 1)

    p11 = MIMOParallel(tfm2, tfm1, tfm4)
    assert p11.args == (tfm2, tfm1, tfm4)
    assert p11.shape == (p11.num_outputs, p11.num_inputs) == (3, 1)

    p12 = MIMOParallel(tfm6, tfm5)
    assert p12.args == (tfm6, tfm5)
    assert p12.shape == (p12.num_outputs, p12.num_inputs) == (2, 2)

    p13 = MIMOParallel(tfm2, tfm4, MIMOSeries(-tfm3, tfm4), -tfm4)
    assert p13.args == (tfm2, tfm4, MIMOSeries(-tfm3, tfm4), -tfm4)
    assert p13.shape == (p13.num_outputs, p13.num_inputs) == (3, 1)

    # arg cannot be empty tuple.
    raises(TypeError, lambda: MIMOParallel(()))

    # arg cannot contain SISO as well as MIMO systems.
    raises(TypeError, lambda: MIMOParallel(tfm1, tfm2, TF1))

    # all TFMs must have same shapes.
    raises(TypeError, lambda: MIMOParallel(tfm1, tfm3, tfm4))

    # all TFMs must be using the same complex variable.
    raises(ValueError, lambda: MIMOParallel(tfm3, tfm7))

    # Number or expression not allowed in the arguments.
    raises(TypeError, lambda: MIMOParallel(2, tfm1, tfm4))
    raises(TypeError, lambda: MIMOParallel(s**2 + p*s, -tfm4, tfm2))


def test_Parallel_functions():
    tf1 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    tf2 = TransferFunction(k, 1, s)
    tf3 = TransferFunction(a2*p - s, a2*s + p, s)
    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)

    assert tf1 + tf2 + tf3 == Parallel(tf1, tf2, tf3)
    assert tf1 + tf2 + tf3 + tf5 == Parallel(tf1, tf2, tf3, tf5)
    assert tf1 + tf2 - tf3 - tf5 == Parallel(tf1, tf2, -tf3, -tf5)
    assert tf1 + tf2*tf3 == Parallel(tf1, Series(tf2, tf3))
    assert tf1 - tf2*tf3 == Parallel(tf1, -Series(tf2,tf3))
    assert -tf1 - tf2 == Parallel(-tf1, -tf2)
    assert -(tf1 + tf2) == Series(TransferFunction(-1, 1, s), Parallel(tf1, tf2))
    assert (tf2 + tf3)*tf1 == Series(Parallel(tf2, tf3), tf1)
    assert (tf1 + tf2)*(tf3*tf5) == Series(Parallel(tf1, tf2), tf3, tf5)
    assert -(tf2 + tf3)*-tf5 == Series(TransferFunction(-1, 1, s), Parallel(tf2, tf3), -tf5)
    assert tf2 + tf3 + tf2*tf1 + tf5 == Parallel(tf2, tf3, Series(tf2, tf1), tf5)
    assert tf2 + tf3 + tf2*tf1 - tf3 == Parallel(tf2, tf3, Series(tf2, tf1), -tf3)
    assert (tf1 + tf2 + tf5)*(tf3 + tf5) == Series(Parallel(tf1, tf2, tf5), Parallel(tf3, tf5))
    raises(ValueError, lambda: tf1 + tf2 + tf4)
    raises(ValueError, lambda: tf1 - tf2*tf4)
    raises(ValueError, lambda: tf3 + Matrix([1, 2, 3]))

    # evaluate=True -> doit()
    assert Parallel(tf1, tf2, evaluate=True) == Parallel(tf1, tf2).doit() == \
        TransferFunction(k*(s**2 + 2*s*wn*zeta + wn**2) + 1, s**2 + 2*s*wn*zeta + wn**2, s)
    assert Parallel(tf1, tf2, Series(-tf1, tf3), evaluate=True) == \
        Parallel(tf1, tf2, Series(-tf1, tf3)).doit() == TransferFunction(k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2)**2 + \
            (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2) + (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), (a2*s + p)*(s**2 + \
                2*s*wn*zeta + wn**2)**2, s)
    assert Parallel(tf2, tf1, -tf3, evaluate=True) == Parallel(tf2, tf1, -tf3).doit() == \
        TransferFunction(a2*s + k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) + p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2) \
            , (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert not Parallel(tf1, -tf2, evaluate=False) == Parallel(tf1, -tf2).doit()

    assert Parallel(Series(tf1, tf2), Series(tf2, tf3)).doit() == \
        TransferFunction(k*(a2*p - s)*(s**2 + 2*s*wn*zeta + wn**2) + k*(a2*s + p), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert Parallel(-tf1, -tf2, -tf3).doit() == \
        TransferFunction(-a2*s - k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) - p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2), \
            (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert -Parallel(tf1, tf2, tf3).doit() == \
        TransferFunction(-a2*s - k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) - p - (a2*p - s)*(s**2 + 2*s*wn*zeta + wn**2), \
            (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert Parallel(tf2, tf3, Series(tf2, -tf1), tf3).doit() == \
        TransferFunction(k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) - k*(a2*s + p) + (2*a2*p - 2*s)*(s**2 + 2*s*wn*zeta \
            + wn**2), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)

    assert Parallel(tf1, tf2).rewrite(TransferFunction) == \
        TransferFunction(k*(s**2 + 2*s*wn*zeta + wn**2) + 1, s**2 + 2*s*wn*zeta + wn**2, s)
    assert Parallel(tf2, tf1, -tf3).rewrite(TransferFunction) == \
        TransferFunction(a2*s + k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) + p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + \
             wn**2), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)

    assert Parallel(tf1, Parallel(tf2, tf3)) == Parallel(tf1, tf2, tf3) == Parallel(Parallel(tf1, tf2), tf3)

    P1 = Parallel(Series(tf1, tf2), Series(tf2, tf3))
    assert P1.is_proper
    assert not P1.is_strictly_proper
    assert P1.is_biproper

    P2 = Parallel(tf1, -tf2, -tf3)
    assert P2.is_proper
    assert not P2.is_strictly_proper
    assert P2.is_biproper

    P3 = Parallel(tf1, -tf2, Series(tf1, tf3))
    assert P3.is_proper
    assert not P3.is_strictly_proper
    assert P3.is_biproper


def test_MIMOParallel_functions():
    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)

    tfm1 = TransferFunctionMatrix([[TF1], [TF2], [TF3]])
    tfm2 = TransferFunctionMatrix([[-TF2], [tf5], [-TF1]])
    tfm3 = TransferFunctionMatrix([[tf5], [-tf5], [TF2]])
    tfm4 = TransferFunctionMatrix([[TF2, -tf5], [TF1, tf5]])
    tfm5 = TransferFunctionMatrix([[TF1, TF2], [TF3, -tf5]])
    tfm6 = TransferFunctionMatrix([[-TF2]])
    tfm7 = TransferFunctionMatrix([[tf4], [-tf4], [tf4]])

    assert tfm1 + tfm2 + tfm3 == MIMOParallel(tfm1, tfm2, tfm3) == MIMOParallel(MIMOParallel(tfm1, tfm2), tfm3)
    assert tfm2 - tfm1 - tfm3 == MIMOParallel(tfm2, -tfm1, -tfm3)
    assert tfm2 - tfm3 + (-tfm1*tfm6*-tfm6) == MIMOParallel(tfm2, -tfm3, MIMOSeries(-tfm6, tfm6, -tfm1))
    assert tfm1 + tfm1 - (-tfm1*tfm6) == MIMOParallel(tfm1, tfm1, -MIMOSeries(tfm6, -tfm1))
    assert tfm2 - tfm3 - tfm1 + tfm2 == MIMOParallel(tfm2, -tfm3, -tfm1, tfm2)
    assert tfm1 + tfm2 - tfm3 - tfm1 == MIMOParallel(tfm1, tfm2, -tfm3, -tfm1)
    raises(ValueError, lambda: tfm1 + tfm2 + TF2)
    raises(TypeError, lambda: tfm1 - tfm2 - a1)
    raises(TypeError, lambda: tfm2 - tfm3 - (s - 1))
    raises(TypeError, lambda: -tfm3 - tfm2 - 9)
    raises(TypeError, lambda: (1 - p**3) - tfm3 - tfm2)
    # All TFMs must use the same complex var. tfm7 uses 'p'.
    raises(ValueError, lambda: tfm3 - tfm2 - tfm7)
    raises(ValueError, lambda: tfm2 - tfm1 + tfm7)
    # (tfm1 +/- tfm2) has (3, 1) shape while tfm4 has (2, 2) shape.
    raises(TypeError, lambda: tfm1 + tfm2 + tfm4)
    raises(TypeError, lambda: (tfm1 - tfm2) - tfm4)

    assert (tfm1 + tfm2)*tfm6 == MIMOSeries(tfm6, MIMOParallel(tfm1, tfm2))
    assert (tfm2 - tfm3)*tfm6*-tfm6 == MIMOSeries(-tfm6, tfm6, MIMOParallel(tfm2, -tfm3))
    assert (tfm2 - tfm1 - tfm3)*(tfm6 + tfm6) == MIMOSeries(MIMOParallel(tfm6, tfm6), MIMOParallel(tfm2, -tfm1, -tfm3))
    raises(ValueError, lambda: (tfm4 + tfm5)*TF1)
    raises(TypeError, lambda: (tfm2 - tfm3)*a2)
    raises(TypeError, lambda: (tfm3 + tfm2)*(s - 6))
    raises(TypeError, lambda: (tfm1 + tfm2 + tfm3)*0)
    raises(TypeError, lambda: (1 - p**3)*(tfm1 + tfm3))

    # (tfm3 - tfm2) has (3, 1) shape while tfm4*tfm5 has (2, 2) shape.
    raises(ValueError, lambda: (tfm3 - tfm2)*tfm4*tfm5)
    # (tfm1 - tfm2) has (3, 1) shape while tfm5 has (2, 2) shape.
    raises(ValueError, lambda: (tfm1 - tfm2)*tfm5)

    # TFM in the arguments.
    assert (MIMOParallel(tfm1, tfm2, evaluate=True) == MIMOParallel(tfm1, tfm2).doit()
    == MIMOParallel(tfm1, tfm2).rewrite(TransferFunctionMatrix)
    == TransferFunctionMatrix(((TransferFunction(-k*(s**2 + 2*s*wn*zeta + wn**2) + 1, s**2 + 2*s*wn*zeta + wn**2, s),), \
        (TransferFunction(-a0 + a1*s**2 + a2*s + k*(a0 + s), a0 + s, s),), (TransferFunction(-a2*s - p + (a2*p - s)* \
        (s**2 + 2*s*wn*zeta + wn**2), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s),))))


def test_Feedback_construction():
    tf1 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    tf2 = TransferFunction(k, 1, s)
    tf3 = TransferFunction(a2*p - s, a2*s + p, s)
    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)
    tf6 = TransferFunction(s - p, p + s, p)

    f1 = Feedback(TransferFunction(1, 1, s), tf1*tf2*tf3)
    assert f1.args == (TransferFunction(1, 1, s), Series(tf1, tf2, tf3), -1)
    assert f1.sys1 == TransferFunction(1, 1, s)
    assert f1.sys2 == Series(tf1, tf2, tf3)
    assert f1.var == s

    f2 = Feedback(tf1, tf2*tf3)
    assert f2.args == (tf1, Series(tf2, tf3), -1)
    assert f2.sys1 == tf1
    assert f2.sys2 == Series(tf2, tf3)
    assert f2.var == s

    f3 = Feedback(tf1*tf2, tf5)
    assert f3.args == (Series(tf1, tf2), tf5, -1)
    assert f3.sys1 == Series(tf1, tf2)

    f4 = Feedback(tf4, tf6)
    assert f4.args == (tf4, tf6, -1)
    assert f4.sys1 == tf4
    assert f4.var == p

    f5 = Feedback(tf5, TransferFunction(1, 1, s))
    assert f5.args == (tf5, TransferFunction(1, 1, s), -1)
    assert f5.var == s
    assert f5 == Feedback(tf5)  # When sys2 is not passed explicitly, it is assumed to be unit tf.

    f6 = Feedback(TransferFunction(1, 1, p), tf4)
    assert f6.args == (TransferFunction(1, 1, p), tf4, -1)
    assert f6.var == p

    f7 = -Feedback(tf4*tf6, TransferFunction(1, 1, p))
    assert f7.args == (Series(TransferFunction(-1, 1, p), Series(tf4, tf6)), -TransferFunction(1, 1, p), -1)
    assert f7.sys1 == Series(TransferFunction(-1, 1, p), Series(tf4, tf6))

    # denominator can't be a Parallel instance
    raises(TypeError, lambda: Feedback(tf1, tf2 + tf3))
    raises(TypeError, lambda: Feedback(tf1, Matrix([1, 2, 3])))
    raises(TypeError, lambda: Feedback(TransferFunction(1, 1, s), s - 1))
    raises(TypeError, lambda: Feedback(1, 1))
    # raises(ValueError, lambda: Feedback(TransferFunction(1, 1, s), TransferFunction(1, 1, s)))
    raises(ValueError, lambda: Feedback(tf2, tf4*tf5))
    raises(ValueError, lambda: Feedback(tf2, tf1, 1.5))  # `sign` can only be -1 or 1
    raises(ValueError, lambda: Feedback(tf1, -tf1**-1))  # denominator can't be zero
    raises(ValueError, lambda: Feedback(tf4, tf5))  # Both systems should use the same `var`


def test_Feedback_functions():
    tf = TransferFunction(1, 1, s)
    tf1 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    tf2 = TransferFunction(k, 1, s)
    tf3 = TransferFunction(a2*p - s, a2*s + p, s)
    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)
    tf6 = TransferFunction(s - p, p + s, p)

    assert (tf1*tf2*tf3 / tf3*tf5) == Series(tf1, tf2, tf3, pow(tf3, -1), tf5)
    assert (tf1*tf2*tf3) / (tf3*tf5) == Series((tf1*tf2*tf3).doit(), pow((tf3*tf5).doit(),-1))
    assert tf / (tf + tf1) == Feedback(tf, tf1)
    assert tf / (tf + tf1*tf2*tf3) == Feedback(tf, tf1*tf2*tf3)
    assert tf1 / (tf + tf1*tf2*tf3) == Feedback(tf1, tf2*tf3)
    assert (tf1*tf2) / (tf + tf1*tf2) == Feedback(tf1*tf2, tf)
    assert (tf1*tf2) / (tf + tf1*tf2*tf5) == Feedback(tf1*tf2, tf5)
    assert (tf1*tf2) / (tf + tf1*tf2*tf5*tf3) in (Feedback(tf1*tf2, tf5*tf3), Feedback(tf1*tf2, tf3*tf5))
    assert tf4 / (TransferFunction(1, 1, p) + tf4*tf6) == Feedback(tf4, tf6)
    assert tf5 / (tf + tf5) == Feedback(tf5, tf)

    raises(TypeError, lambda: tf1*tf2*tf3 / (1 + tf1*tf2*tf3))
    raises(ValueError, lambda: tf2*tf3 / (tf + tf2*tf3*tf4))

    assert Feedback(tf, tf1*tf2*tf3).doit() == \
        TransferFunction((a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), k*(a2*p - s) + \
        (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert Feedback(tf, tf1*tf2*tf3).sensitivity == \
        1/(k*(a2*p - s)/((a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2)) + 1)
    assert Feedback(tf1, tf2*tf3).doit() == \
        TransferFunction((a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), (k*(a2*p - s) + \
        (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2))*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert Feedback(tf1, tf2*tf3).sensitivity == \
        1/(k*(a2*p - s)/((a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2)) + 1)
    assert Feedback(tf1*tf2, tf5).doit() == \
        TransferFunction(k*(a0 + s)*(s**2 + 2*s*wn*zeta + wn**2), (k*(-a0 + a1*s**2 + a2*s) + \
        (a0 + s)*(s**2 + 2*s*wn*zeta + wn**2))*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert Feedback(tf1*tf2, tf5, 1).sensitivity == \
        1/(-k*(-a0 + a1*s**2 + a2*s)/((a0 + s)*(s**2 + 2*s*wn*zeta + wn**2)) + 1)
    assert Feedback(tf4, tf6).doit() == \
        TransferFunction(p*(p + s)*(a0*p + p**a1 - s), p*(p*(p + s) + (-p + s)*(a0*p + p**a1 - s)), p)
    assert -Feedback(tf4*tf6, TransferFunction(1, 1, p)).doit() == \
        TransferFunction(-p*(-p + s)*(p + s)*(a0*p + p**a1 - s), p*(p + s)*(p*(p + s) + (-p + s)*(a0*p + p**a1 - s)), p)
    assert Feedback(tf, tf).doit() == TransferFunction(1, 2, s)

    assert Feedback(tf1, tf2*tf5).rewrite(TransferFunction) == \
        TransferFunction((a0 + s)*(s**2 + 2*s*wn*zeta + wn**2), (k*(-a0 + a1*s**2 + a2*s) + \
        (a0 + s)*(s**2 + 2*s*wn*zeta + wn**2))*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert Feedback(TransferFunction(1, 1, p), tf4).rewrite(TransferFunction) == \
        TransferFunction(p, a0*p + p + p**a1 - s, p)


def test_Feedback_with_Series():
    # Solves issue https://github.com/sympy/sympy/issues/26161
    tf1 = TransferFunction(s+1, 1, s)
    tf2 = TransferFunction(s+2, 1, s)
    fd1 = Feedback(tf1, tf2, -1) # Negative Feedback system
    fd2 = Feedback(tf1, tf2, 1) # Positive Feedback system
    unit = TransferFunction(1, 1, s)

    # Checking the type
    assert isinstance(fd1, SISOLinearTimeInvariant)
    assert isinstance(fd1, Feedback)

    # Testing the numerator and denominator
    assert fd1.num == tf1
    assert fd2.num == tf1
    assert fd1.den == Parallel(unit, Series(tf2, tf1))
    assert fd2.den == Parallel(unit, -Series(tf2, tf1))

    # Testing the Series and Parallel Combination with Feedback and TransferFunction
    s1 = Series(tf1, fd1)
    p1 = Parallel(tf1, fd1)
    assert tf1 * fd1 == s1
    assert tf1 + fd1 == p1
    assert s1.doit() == TransferFunction((s + 1)**2, (s + 1)*(s + 2) + 1, s)
    assert p1.doit() == TransferFunction(s + (s + 1)*((s + 1)*(s + 2) + 1) + 1, (s + 1)*(s + 2) + 1, s)

    # Testing the use of Feedback and TransferFunction with Feedback
    fd3 = Feedback(tf1*fd1, tf2, -1)
    assert fd3 == Feedback(Series(tf1, fd1), tf2)
    assert fd3.num == tf1 * fd1
    assert fd3.den == Parallel(unit, Series(tf2, Series(tf1, fd1)))

    # Testing the use of Feedback and TransferFunction with TransferFunction
    tf3 = TransferFunction(tf1*fd1, tf2, s)
    assert tf3 == TransferFunction(Series(tf1, fd1), tf2, s)
    assert tf3.num == tf1*fd1


def test_issue_26161():
    # Issue https://github.com/sympy/sympy/issues/26161
    Ib, Is, m, h, l2, l1 = symbols('I_b, I_s, m, h, l2, l1',
                                            real=True, nonnegative=True)
    KD, KP, v = symbols('K_D, K_P, v', real=True)

    tau1_sq = (Ib + m * h ** 2) / m / g / h
    tau2 = l2 / v
    tau3 = v / (l1 + l2)
    K = v ** 2 / g / (l1 + l2)

    Gtheta = TransferFunction(-K * (tau2 * s + 1), tau1_sq * s ** 2 - 1, s)
    Gdelta = TransferFunction(1, Is * s ** 2 + c * s, s)
    Gpsi = TransferFunction(1, tau3 * s, s)
    Dcont = TransferFunction(KD * s, 1, s)
    PIcont = TransferFunction(KP, s, s)
    Gunity = TransferFunction(1, 1, s)

    Ginner = Feedback(Dcont * Gdelta, Gtheta)
    Gouter = Feedback(PIcont * Ginner * Gpsi, Gunity)
    assert Gouter == Feedback(Series(PIcont, Series(Ginner, Gpsi)), Gunity)
    assert Gouter.num == Series(PIcont, Series(Ginner, Gpsi))
    assert Gouter.den == Parallel(Gunity, Series(Gunity, Series(PIcont, Series(Ginner, Gpsi))))
    expr = (KD*KP*g*s**3*v**2*(l1 + l2)*(Is*s**2 + c*s)**2*(-g*h*m + s**2*(Ib + h**2*m))*(-KD*g*h*m*s*v**2*(l2*s + v) + \
            g*v*(l1 + l2)*(Is*s**2 + c*s)*(-g*h*m + s**2*(Ib + h**2*m))))/((s**2*v*(Is*s**2 + c*s)*(-KD*g*h*m*s*v**2* \
            (l2*s + v) + g*v*(l1 + l2)*(Is*s**2 + c*s)*(-g*h*m + s**2*(Ib + h**2*m)))*(KD*KP*g*s*v*(l1 + l2)**2* \
            (Is*s**2 + c*s)*(-g*h*m + s**2*(Ib + h**2*m)) + s**2*v*(Is*s**2 + c*s)*(-KD*g*h*m*s*v**2*(l2*s + v) + \
            g*v*(l1 + l2)*(Is*s**2 + c*s)*(-g*h*m + s**2*(Ib + h**2*m))))/(l1 + l2)))

    assert (Gouter.to_expr() - expr).simplify() == 0


def test_MIMOFeedback_construction():
    tf1 = TransferFunction(1, s, s)
    tf2 = TransferFunction(s, s**3 - 1, s)
    tf3 = TransferFunction(s, s + 1, s)
    tf4 = TransferFunction(s, s**2 + 1, s)

    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
    tfm_2 = TransferFunctionMatrix([[tf2, tf3], [tf4, tf1]])
    tfm_3 = TransferFunctionMatrix([[tf3, tf4], [tf1, tf2]])

    f1 = MIMOFeedback(tfm_1, tfm_2)
    assert f1.args == (tfm_1, tfm_2, -1)
    assert f1.sys1 == tfm_1
    assert f1.sys2 == tfm_2
    assert f1.var == s
    assert f1.sign == -1
    assert -(-f1) == f1

    f2 = MIMOFeedback(tfm_2, tfm_1, 1)
    assert f2.args == (tfm_2, tfm_1, 1)
    assert f2.sys1 == tfm_2
    assert f2.sys2 == tfm_1
    assert f2.var == s
    assert f2.sign == 1

    f3 = MIMOFeedback(tfm_1, MIMOSeries(tfm_3, tfm_2))
    assert f3.args == (tfm_1, MIMOSeries(tfm_3, tfm_2), -1)
    assert f3.sys1 == tfm_1
    assert f3.sys2 == MIMOSeries(tfm_3, tfm_2)
    assert f3.var == s
    assert f3.sign == -1

    mat = Matrix([[1, 1/s], [0, 1]])
    sys1 = controller = TransferFunctionMatrix.from_Matrix(mat, s)
    f4 = MIMOFeedback(sys1, controller)
    assert f4.args == (sys1, controller, -1)
    assert f4.sys1 == f4.sys2 == sys1


def test_MIMOFeedback_errors():
    tf1 = TransferFunction(1, s, s)
    tf2 = TransferFunction(s, s**3 - 1, s)
    tf3 = TransferFunction(s, s - 1, s)
    tf4 = TransferFunction(s, s**2 + 1, s)
    tf5 = TransferFunction(1, 1, s)
    tf6 = TransferFunction(-1, s - 1, s)

    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
    tfm_2 = TransferFunctionMatrix([[tf2, tf3], [tf4, tf1]])
    tfm_3 = TransferFunctionMatrix.from_Matrix(eye(2), var=s)
    tfm_4 = TransferFunctionMatrix([[tf1, tf5], [tf5, tf5]])
    tfm_5 = TransferFunctionMatrix([[-tf3, tf3], [tf3, tf6]])
    # tfm_4 is inverse of tfm_5. Therefore tfm_5*tfm_4 = I
    tfm_6 = TransferFunctionMatrix([[-tf3]])
    tfm_7 = TransferFunctionMatrix([[tf3, tf4]])

    # Unsupported Types
    raises(TypeError, lambda: MIMOFeedback(tf1, tf2))
    raises(TypeError, lambda: MIMOFeedback(MIMOParallel(tfm_1, tfm_2), tfm_3))
    # Shape Errors
    raises(ValueError, lambda: MIMOFeedback(tfm_1, tfm_6, 1))
    raises(ValueError, lambda: MIMOFeedback(tfm_7, tfm_7))
    # sign not 1/-1
    raises(ValueError, lambda: MIMOFeedback(tfm_1, tfm_2, -2))
    # Non-Invertible Systems
    raises(ValueError, lambda: MIMOFeedback(tfm_5, tfm_4, 1))
    raises(ValueError, lambda: MIMOFeedback(tfm_4, -tfm_5))
    raises(ValueError, lambda: MIMOFeedback(tfm_3, tfm_3, 1))
    # Variable not same in both the systems
    tfm_8 = TransferFunctionMatrix.from_Matrix(eye(2), var=p)
    raises(ValueError, lambda: MIMOFeedback(tfm_1, tfm_8, 1))


def test_MIMOFeedback_functions():
    tf1 = TransferFunction(1, s, s)
    tf2 = TransferFunction(s, s - 1, s)
    tf3 = TransferFunction(1, 1, s)
    tf4 = TransferFunction(-1, s - 1, s)

    tfm_1 = TransferFunctionMatrix.from_Matrix(eye(2), var=s)
    tfm_2 = TransferFunctionMatrix([[tf1, tf3], [tf3, tf3]])
    tfm_3 = TransferFunctionMatrix([[-tf2, tf2], [tf2, tf4]])
    tfm_4 = TransferFunctionMatrix([[tf1, tf2], [-tf2, tf1]])

    # sensitivity, doit(), rewrite()
    F_1 = MIMOFeedback(tfm_2, tfm_3)
    F_2 = MIMOFeedback(tfm_2, MIMOSeries(tfm_4, -tfm_1), 1)

    assert F_1.sensitivity == Matrix([[S.Half, 0], [0, S.Half]])
    assert F_2.sensitivity == Matrix([[(-2*s**4 + s**2)/(s**2 - s + 1),
        (2*s**3 - s**2)/(s**2 - s + 1)], [-s**2, s]])

    assert F_1.doit() == \
        TransferFunctionMatrix(((TransferFunction(1, 2*s, s),
        TransferFunction(1, 2, s)), (TransferFunction(1, 2, s),
        TransferFunction(1, 2, s)))) == F_1.rewrite(TransferFunctionMatrix)
    assert F_2.doit(cancel=False, expand=True) == \
        TransferFunctionMatrix(((TransferFunction(-s**5 + 2*s**4 - 2*s**3 + s**2, s**5 - 2*s**4 + 3*s**3 - 2*s**2 + s, s),
        TransferFunction(-2*s**4 + 2*s**3, s**2 - s + 1, s)), (TransferFunction(0, 1, s), TransferFunction(-s**2 + s, 1, s))))
    assert F_2.doit(cancel=False) == \
        TransferFunctionMatrix(((TransferFunction(s*(2*s**3 - s**2)*(s**2 - s + 1) + \
        (-2*s**4 + s**2)*(s**2 - s + 1), s*(s**2 - s + 1)**2, s), TransferFunction(-2*s**4 + 2*s**3, s**2 - s + 1, s)),
        (TransferFunction(0, 1, s), TransferFunction(-s**2 + s, 1, s))))
    assert F_2.doit() == \
        TransferFunctionMatrix(((TransferFunction(s*(-2*s**2 + s*(2*s - 1) + 1), s**2 - s + 1, s),
        TransferFunction(-2*s**3*(s - 1), s**2 - s + 1, s)), (TransferFunction(0, 1, s), TransferFunction(s*(1 - s), 1, s))))
    assert F_2.doit(expand=True) == \
        TransferFunctionMatrix(((TransferFunction(-s**2 + s, s**2 - s + 1, s), TransferFunction(-2*s**4 + 2*s**3, s**2 - s + 1, s)),
        (TransferFunction(0, 1, s), TransferFunction(-s**2 + s, 1, s))))

    assert -(F_1.doit()) == (-F_1).doit()  # First negating then calculating vs calculating then negating.


def test_TransferFunctionMatrix_construction():
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)
    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)

    tfm3_ = TransferFunctionMatrix([[-TF3]])
    assert tfm3_.shape == (tfm3_.num_outputs, tfm3_.num_inputs) == (1, 1)
    assert tfm3_.args == Tuple(Tuple(Tuple(-TF3)))
    assert tfm3_.var == s

    tfm5 = TransferFunctionMatrix([[TF1, -TF2], [TF3, tf5]])
    assert tfm5.shape == (tfm5.num_outputs, tfm5.num_inputs) == (2, 2)
    assert tfm5.args == Tuple(Tuple(Tuple(TF1, -TF2), Tuple(TF3, tf5)))
    assert tfm5.var == s

    tfm7 = TransferFunctionMatrix([[TF1, TF2], [TF3, -tf5], [-tf5, TF2]])
    assert tfm7.shape == (tfm7.num_outputs, tfm7.num_inputs) == (3, 2)
    assert tfm7.args == Tuple(Tuple(Tuple(TF1, TF2), Tuple(TF3, -tf5), Tuple(-tf5, TF2)))
    assert tfm7.var == s

    # all transfer functions will use the same complex variable. tf4 uses 'p'.
    raises(ValueError, lambda: TransferFunctionMatrix([[TF1], [TF2], [tf4]]))
    raises(ValueError, lambda: TransferFunctionMatrix([[TF1, tf4], [TF3, tf5]]))

    # length of all the lists in the TFM should be equal.
    raises(ValueError, lambda: TransferFunctionMatrix([[TF1], [TF3, tf5]]))
    raises(ValueError, lambda: TransferFunctionMatrix([[TF1, TF3], [tf5]]))

    # lists should only support transfer functions in them.
    raises(TypeError, lambda: TransferFunctionMatrix([[TF1, TF2], [TF3, Matrix([1, 2])]]))
    raises(TypeError, lambda: TransferFunctionMatrix([[TF1, Matrix([1, 2])], [TF3, TF2]]))

    # `arg` should strictly be nested list of TransferFunction
    raises(ValueError, lambda: TransferFunctionMatrix([TF1, TF2, tf5]))
    raises(ValueError, lambda: TransferFunctionMatrix([TF1]))

def test_TransferFunctionMatrix_functions():
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)

    #  Classmethod (from_matrix)

    mat_1 = ImmutableMatrix([
        [s*(s + 1)*(s - 3)/(s**4 + 1), 2],
        [p, p*(s + 1)/(s*(s**1 + 1))]
        ])
    mat_2 = ImmutableMatrix([[(2*s + 1)/(s**2 - 9)]])
    mat_3 = ImmutableMatrix([[1, 2], [3, 4]])
    assert TransferFunctionMatrix.from_Matrix(mat_1, s) == \
        TransferFunctionMatrix([[TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(2, 1, s)],
        [TransferFunction(p, 1, s), TransferFunction(p, s, s)]])
    assert TransferFunctionMatrix.from_Matrix(mat_2, s) == \
        TransferFunctionMatrix([[TransferFunction(2*s + 1, s**2 - 9, s)]])
    assert TransferFunctionMatrix.from_Matrix(mat_3, p) == \
        TransferFunctionMatrix([[TransferFunction(1, 1, p), TransferFunction(2, 1, p)],
        [TransferFunction(3, 1, p), TransferFunction(4, 1, p)]])

    #  Negating a TFM

    tfm1 = TransferFunctionMatrix([[TF1], [TF2]])
    assert -tfm1 == TransferFunctionMatrix([[-TF1], [-TF2]])

    tfm2 = TransferFunctionMatrix([[TF1, TF2, TF3], [tf5, -TF1, -TF3]])
    assert -tfm2 == TransferFunctionMatrix([[-TF1, -TF2, -TF3], [-tf5, TF1, TF3]])

    # subs()

    H_1 = TransferFunctionMatrix.from_Matrix(mat_1, s)
    H_2 = TransferFunctionMatrix([[TransferFunction(a*p*s, k*s**2, s), TransferFunction(p*s, k*(s**2 - a), s)]])
    assert H_1.subs(p, 1) == TransferFunctionMatrix([[TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(2, 1, s)], [TransferFunction(1, 1, s), TransferFunction(1, s, s)]])
    assert H_1.subs({p: 1}) == TransferFunctionMatrix([[TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(2, 1, s)], [TransferFunction(1, 1, s), TransferFunction(1, s, s)]])
    assert H_1.subs({p: 1, s: 1}) == TransferFunctionMatrix([[TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(2, 1, s)], [TransferFunction(1, 1, s), TransferFunction(1, s, s)]]) # This should ignore `s` as it is `var`
    assert H_2.subs(p, 2) == TransferFunctionMatrix([[TransferFunction(2*a*s, k*s**2, s), TransferFunction(2*s, k*(-a + s**2), s)]])
    assert H_2.subs(k, 1) == TransferFunctionMatrix([[TransferFunction(a*p*s, s**2, s), TransferFunction(p*s, -a + s**2, s)]])
    assert H_2.subs(a, 0) == TransferFunctionMatrix([[TransferFunction(0, k*s**2, s), TransferFunction(p*s, k*s**2, s)]])
    assert H_2.subs({p: 1, k: 1, a: a0}) == TransferFunctionMatrix([[TransferFunction(a0*s, s**2, s), TransferFunction(s, -a0 + s**2, s)]])

    # eval_frequency()
    assert H_2.eval_frequency(S(1)/2 + I) == Matrix([[2*a*p/(5*k) - 4*I*a*p/(5*k), I*p/(-a*k - 3*k/4 + I*k) + p/(-2*a*k - 3*k/2 + 2*I*k)]])

    # transpose()

    assert H_1.transpose() == TransferFunctionMatrix([[TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(p, 1, s)], [TransferFunction(2, 1, s), TransferFunction(p, s, s)]])
    assert H_2.transpose() == TransferFunctionMatrix([[TransferFunction(a*p*s, k*s**2, s)], [TransferFunction(p*s, k*(-a + s**2), s)]])
    assert H_1.transpose().transpose() == H_1
    assert H_2.transpose().transpose() == H_2

    # elem_poles()

    assert H_1.elem_poles() == [[[-sqrt(2)/2 - sqrt(2)*I/2, -sqrt(2)/2 + sqrt(2)*I/2, sqrt(2)/2 - sqrt(2)*I/2, sqrt(2)/2 + sqrt(2)*I/2], []],
        [[], [0]]]
    assert H_2.elem_poles() == [[[0, 0], [sqrt(a), -sqrt(a)]]]
    assert tfm2.elem_poles() == [[[wn*(-zeta + sqrt((zeta - 1)*(zeta + 1))), wn*(-zeta - sqrt((zeta - 1)*(zeta + 1)))], [], [-p/a2]],
        [[-a0], [wn*(-zeta + sqrt((zeta - 1)*(zeta + 1))), wn*(-zeta - sqrt((zeta - 1)*(zeta + 1)))], [-p/a2]]]

    # elem_zeros()

    assert H_1.elem_zeros() == [[[-1, 0, 3], []], [[], []]]
    assert H_2.elem_zeros() == [[[0], [0]]]
    assert tfm2.elem_zeros() == [[[], [], [a2*p]],
        [[-a2/(2*a1) - sqrt(4*a0*a1 + a2**2)/(2*a1), -a2/(2*a1) + sqrt(4*a0*a1 + a2**2)/(2*a1)], [], [a2*p]]]

    # doit()

    H_3 = TransferFunctionMatrix([[Series(TransferFunction(1, s**3 - 3, s), TransferFunction(s**2 - 2*s + 5, 1, s), TransferFunction(1, s, s))]])
    H_4 = TransferFunctionMatrix([[Parallel(TransferFunction(s**3 - 3, 4*s**4 - s**2 - 2*s + 5, s), TransferFunction(4 - s**3, 4*s**4 - s**2 - 2*s + 5, s))]])

    assert H_3.doit() == TransferFunctionMatrix([[TransferFunction(s**2 - 2*s + 5, s*(s**3 - 3), s)]])
    assert H_4.doit() == TransferFunctionMatrix([[TransferFunction(1, 4*s**4 - s**2 - 2*s + 5, s)]])

    # _flat()

    assert H_1._flat() == [TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(2, 1, s), TransferFunction(p, 1, s), TransferFunction(p, s, s)]
    assert H_2._flat() == [TransferFunction(a*p*s, k*s**2, s), TransferFunction(p*s, k*(-a + s**2), s)]
    assert H_3._flat() == [Series(TransferFunction(1, s**3 - 3, s), TransferFunction(s**2 - 2*s + 5, 1, s), TransferFunction(1, s, s))]
    assert H_4._flat() == [Parallel(TransferFunction(s**3 - 3, 4*s**4 - s**2 - 2*s + 5, s), TransferFunction(4 - s**3, 4*s**4 - s**2 - 2*s + 5, s))]

    # evalf()

    assert H_1.evalf() == \
        TransferFunctionMatrix(((TransferFunction(s*(s - 3.0)*(s + 1.0), s**4 + 1.0, s), TransferFunction(2.0, 1, s)), (TransferFunction(1.0*p, 1, s), TransferFunction(p, s, s))))
    assert H_2.subs({a:3.141, p:2.88, k:2}).evalf() == \
        TransferFunctionMatrix(((TransferFunction(4.5230399999999999494093572138808667659759521484375, s, s),
        TransferFunction(2.87999999999999989341858963598497211933135986328125*s, 2.0*s**2 - 6.282000000000000028421709430404007434844970703125, s)),))

    # simplify()

    H_5 = TransferFunctionMatrix([[TransferFunction(s**5 + s**3 + s, s - s**2, s),
        TransferFunction((s + 3)*(s - 1), (s - 1)*(s + 5), s)]])

    assert H_5.simplify() == simplify(H_5) == \
        TransferFunctionMatrix(((TransferFunction(-s**4 - s**2 - 1, s - 1, s), TransferFunction(s + 3, s + 5, s)),))

    # expand()

    assert (H_1.expand()
            == TransferFunctionMatrix(((TransferFunction(s**3 - 2*s**2 - 3*s, s**4 + 1, s), TransferFunction(2, 1, s)),
            (TransferFunction(p, 1, s), TransferFunction(p, s, s)))))
    assert H_5.expand() == \
        TransferFunctionMatrix(((TransferFunction(s**5 + s**3 + s, -s**2 + s, s), TransferFunction(s**2 + 2*s - 3, s**2 + 4*s - 5, s)),))

def test_TransferFunction_gbt():
    # simple transfer function, e.g. ohms law
    tf = TransferFunction(1, a*s+b, s)
    numZ, denZ = gbt(tf, T, 0.5)
    # discretized transfer function with coefs from tf.gbt()
    tf_test_bilinear = TransferFunction(s * numZ[0] + numZ[1], s * denZ[0] + denZ[1], s)
    # corresponding tf with manually calculated coefs
    tf_test_manual = TransferFunction(s * T/(2*(a + b*T/2)) + T/(2*(a + b*T/2)), s + (-a + b*T/2)/(a + b*T/2), s)

    assert S.Zero == (tf_test_bilinear.simplify()-tf_test_manual.simplify()).simplify().num

    tf = TransferFunction(1, a*s+b, s)
    numZ, denZ = gbt(tf, T, 0)
    # discretized transfer function with coefs from tf.gbt()
    tf_test_forward = TransferFunction(numZ[0], s*denZ[0]+denZ[1], s)
    # corresponding tf with manually calculated coefs
    tf_test_manual = TransferFunction(T/a, s + (-a + b*T)/a, s)

    assert S.Zero == (tf_test_forward.simplify()-tf_test_manual.simplify()).simplify().num

    tf = TransferFunction(1, a*s+b, s)
    numZ, denZ = gbt(tf, T, 1)
    # discretized transfer function with coefs from tf.gbt()
    tf_test_backward = TransferFunction(s*numZ[0], s*denZ[0]+denZ[1], s)
    # corresponding tf with manually calculated coefs
    tf_test_manual = TransferFunction(s * T/(a + b*T), s - a/(a + b*T), s)

    assert S.Zero == (tf_test_backward.simplify()-tf_test_manual.simplify()).simplify().num

    tf = TransferFunction(1, a*s+b, s)
    numZ, denZ = gbt(tf, T, 0.3)
    # discretized transfer function with coefs from tf.gbt()
    tf_test_gbt = TransferFunction(s*numZ[0]+numZ[1], s*denZ[0]+denZ[1], s)
    # corresponding tf with manually calculated coefs
    tf_test_manual = TransferFunction(s*3*T/(10*(a + 3*b*T/10)) + 7*T/(10*(a + 3*b*T/10)), s + (-a + 7*b*T/10)/(a + 3*b*T/10), s)

    assert S.Zero == (tf_test_gbt.simplify()-tf_test_manual.simplify()).simplify().num

def test_TransferFunction_bilinear():
    # simple transfer function, e.g. ohms law
    tf = TransferFunction(1, a*s+b, s)
    numZ, denZ = bilinear(tf, T)
    # discretized transfer function with coefs from tf.bilinear()
    tf_test_bilinear = TransferFunction(s*numZ[0]+numZ[1], s*denZ[0]+denZ[1], s)
    # corresponding tf with manually calculated coefs
    tf_test_manual = TransferFunction(s * T/(2*(a + b*T/2)) + T/(2*(a + b*T/2)), s + (-a + b*T/2)/(a + b*T/2), s)

    assert S.Zero == (tf_test_bilinear.simplify()-tf_test_manual.simplify()).simplify().num

def test_TransferFunction_forward_diff():
    # simple transfer function, e.g. ohms law
    tf = TransferFunction(1, a*s+b, s)
    numZ, denZ = forward_diff(tf, T)
    # discretized transfer function with coefs from tf.forward_diff()
    tf_test_forward = TransferFunction(numZ[0], s*denZ[0]+denZ[1], s)
    # corresponding tf with manually calculated coefs
    tf_test_manual = TransferFunction(T/a, s + (-a + b*T)/a, s)

    assert S.Zero == (tf_test_forward.simplify()-tf_test_manual.simplify()).simplify().num

def test_TransferFunction_backward_diff():
    # simple transfer function, e.g. ohms law
    tf = TransferFunction(1, a*s+b, s)
    numZ, denZ = backward_diff(tf, T)
    # discretized transfer function with coefs from tf.backward_diff()
    tf_test_backward = TransferFunction(s*numZ[0]+numZ[1], s*denZ[0]+denZ[1], s)
    # corresponding tf with manually calculated coefs
    tf_test_manual = TransferFunction(s * T/(a + b*T), s - a/(a + b*T), s)

    assert S.Zero == (tf_test_backward.simplify()-tf_test_manual.simplify()).simplify().num

def test_TransferFunction_phase_margin():
    # Test for phase margin
    tf1 = TransferFunction(10, p**3 + 1, p)
    tf2 = TransferFunction(s**2, 10, s)
    tf3 = TransferFunction(1, a*s+b, s)
    tf4 = TransferFunction((s + 1)*exp(s/tau), s**2 + 2, s)
    tf_m = TransferFunctionMatrix([[tf2],[tf3]])

    assert phase_margin(tf1) == -180 + 180*atan(3*sqrt(11))/pi
    assert phase_margin(tf2) == 0

    raises(NotImplementedError, lambda: phase_margin(tf4))
    raises(ValueError, lambda: phase_margin(tf3))
    raises(ValueError, lambda: phase_margin(MIMOSeries(tf_m)))

def test_TransferFunction_gain_margin():
    # Test for gain margin
    tf1 = TransferFunction(s**2, 5*(s+1)*(s-5)*(s-10), s)
    tf2 = TransferFunction(s**2 + 2*s + 1, 1, s)
    tf3 = TransferFunction(1, a*s+b, s)
    tf4 = TransferFunction((s + 1)*exp(s/tau), s**2 + 2, s)
    tf_m = TransferFunctionMatrix([[tf2],[tf3]])

    assert gain_margin(tf1) == -20*log(S(7)/540)/log(10)
    assert gain_margin(tf2) == oo

    raises(NotImplementedError, lambda: gain_margin(tf4))
    raises(ValueError, lambda: gain_margin(tf3))
    raises(ValueError, lambda: gain_margin(MIMOSeries(tf_m)))


def test_StateSpace_construction():
    # using different numbers for a SISO system.
    A1 = Matrix([[0, 1], [1, 0]])
    B1 = Matrix([1, 0])
    C1 = Matrix([[0, 1]])
    D1 = Matrix([0])
    ss1 = StateSpace(A1, B1, C1, D1)

    assert ss1.state_matrix == Matrix([[0, 1], [1, 0]])
    assert ss1.input_matrix == Matrix([1, 0])
    assert ss1.output_matrix == Matrix([[0, 1]])
    assert ss1.feedforward_matrix == Matrix([0])
    assert ss1.args == (Matrix([[0, 1], [1, 0]]), Matrix([[1], [0]]), Matrix([[0, 1]]), Matrix([[0]]))

    # using different symbols for a SISO system.
    ss2 = StateSpace(Matrix([a0]), Matrix([a1]),
                    Matrix([a2]), Matrix([a3]))

    assert ss2.state_matrix == Matrix([[a0]])
    assert ss2.input_matrix == Matrix([[a1]])
    assert ss2.output_matrix == Matrix([[a2]])
    assert ss2.feedforward_matrix == Matrix([[a3]])
    assert ss2.args == (Matrix([[a0]]), Matrix([[a1]]), Matrix([[a2]]), Matrix([[a3]]))

    # using different numbers for a MIMO system.
    ss3 = StateSpace(Matrix([[-1.5, -2], [1, 0]]),
                    Matrix([[0.5, 0], [0, 1]]),
                    Matrix([[0, 1], [0, 2]]),
                    Matrix([[2, 2], [1, 1]]))

    assert ss3.state_matrix == Matrix([[-1.5, -2], [1,  0]])
    assert ss3.input_matrix == Matrix([[0.5, 0], [0, 1]])
    assert ss3.output_matrix == Matrix([[0, 1], [0, 2]])
    assert ss3.feedforward_matrix == Matrix([[2, 2], [1, 1]])
    assert ss3.args == (Matrix([[-1.5, -2],
                                [1,  0]]),
                        Matrix([[0.5, 0],
                                [0, 1]]),
                        Matrix([[0, 1],
                                [0, 2]]),
                        Matrix([[2, 2],
                                [1, 1]]))

    # using different symbols for a MIMO system.
    A4 = Matrix([[a0, a1], [a2, a3]])
    B4 = Matrix([[b0, b1], [b2, b3]])
    C4 = Matrix([[c0, c1], [c2, c3]])
    D4 = Matrix([[d0, d1], [d2, d3]])
    ss4 = StateSpace(A4, B4, C4, D4)

    assert ss4.state_matrix == Matrix([[a0, a1], [a2, a3]])
    assert ss4.input_matrix == Matrix([[b0, b1], [b2, b3]])
    assert ss4.output_matrix == Matrix([[c0, c1], [c2, c3]])
    assert ss4.feedforward_matrix == Matrix([[d0, d1], [d2, d3]])
    assert ss4.args == (Matrix([[a0, a1],
                                [a2, a3]]),
                        Matrix([[b0, b1],
                                [b2, b3]]),
                        Matrix([[c0, c1],
                                [c2, c3]]),
                        Matrix([[d0, d1],
                                [d2, d3]]))

    # using less matrices. Rest will be filled with a minimum of zeros.
    ss5 = StateSpace()
    assert ss5.args == (Matrix([[0]]), Matrix([[0]]), Matrix([[0]]), Matrix([[0]]))

    A6 = Matrix([[0, 1], [1, 0]])
    B6 = Matrix([1, 1])
    ss6 = StateSpace(A6, B6)

    assert ss6.state_matrix == Matrix([[0, 1], [1, 0]])
    assert ss6.input_matrix ==  Matrix([1, 1])
    assert ss6.output_matrix == Matrix([[0, 0]])
    assert ss6.feedforward_matrix == Matrix([[0]])
    assert ss6.args == (Matrix([[0, 1],
                                [1, 0]]),
                        Matrix([[1],
                                [1]]),
                        Matrix([[0, 0]]),
                        Matrix([[0]]))

    # Check if the system is SISO or MIMO.
    # If system is not SISO, then it is definitely MIMO.

    assert ss1.is_SISO == True
    assert ss2.is_SISO == True
    assert ss3.is_SISO == False
    assert ss4.is_SISO == False
    assert ss5.is_SISO == True
    assert ss6.is_SISO == True

    # ShapeError if matrices do not fit.
    raises(ShapeError, lambda: StateSpace(Matrix([s, (s+1)**2]), Matrix([s+1]),
                                          Matrix([s**2 - 1]), Matrix([2*s])))
    raises(ShapeError, lambda: StateSpace(Matrix([s]), Matrix([s+1, s**3 + 1]),
                                          Matrix([s**2 - 1]), Matrix([2*s])))
    raises(ShapeError, lambda: StateSpace(Matrix([s]), Matrix([s+1]),
                                          Matrix([[s**2 - 1], [s**2 + 2*s + 1]]), Matrix([2*s])))
    raises(ShapeError, lambda: StateSpace(Matrix([[-s, -s], [s, 0]]),
                                                Matrix([[s/2, 0], [0, s]]),
                                                Matrix([[0, s]]),
                                                Matrix([[2*s, 2*s], [s, s]])))

    # TypeError if arguments are not sympy matrices.
    raises(TypeError, lambda: StateSpace(s**2, s+1, 2*s, 1))
    raises(TypeError, lambda: StateSpace(Matrix([2, 0.5]), Matrix([-1]),
                                         Matrix([1]), 0))
def test_StateSpace_add():
    A1 = Matrix([[4, 1],[2, -3]])
    B1 = Matrix([[5, 2],[-3, -3]])
    C1 = Matrix([[2, -4],[0, 1]])
    D1 = Matrix([[3, 2],[1, -1]])
    ss1 = StateSpace(A1, B1, C1, D1)

    A2 = Matrix([[-3, 4, 2],[-1, -3, 0],[2, 5, 3]])
    B2 = Matrix([[1, 4],[-3, -3],[-2, 1]])
    C2 = Matrix([[4, 2, -3],[1, 4, 3]])
    D2 = Matrix([[-2, 4],[0, 1]])
    ss2 = StateSpace(A2, B2, C2, D2)
    ss3 = StateSpace()
    ss4 = StateSpace(Matrix([1]), Matrix([2]), Matrix([3]), Matrix([4]))

    expected_add = \
        StateSpace(
        Matrix([
        [4,  1,  0,  0, 0],
        [2, -3,  0,  0, 0],
        [0,  0, -3,  4, 2],
        [0,  0, -1, -3, 0],
        [0,  0,  2,  5, 3]]),
        Matrix([
        [ 5,  2],
        [-3, -3],
        [ 1,  4],
        [-3, -3],
        [-2,  1]]),
        Matrix([
        [2, -4, 4, 2, -3],
        [0,  1, 1, 4,  3]]),
        Matrix([
        [1, 6],
        [1, 0]]))

    expected_mul = \
        StateSpace(
        Matrix([
        [ -3,   4,  2, 0,  0],
        [ -1,  -3,  0, 0,  0],
        [  2,   5,  3, 0,  0],
        [ 22,  18, -9, 4,  1],
        [-15, -18,  0, 2, -3]]),
        Matrix([
        [  1,   4],
        [ -3,  -3],
        [ -2,   1],
        [-10,  22],
        [  6, -15]]),
        Matrix([
        [14, 14, -3, 2, -4],
        [ 3, -2, -6, 0,  1]]),
        Matrix([
        [-6, 14],
        [-2,  3]]))

    assert ss1 + ss2 == expected_add
    assert ss1*ss2 == expected_mul
    assert ss3 + 1/2 == StateSpace(Matrix([[0]]), Matrix([[0]]), Matrix([[0]]), Matrix([[0.5]]))
    assert ss4*1.5 == StateSpace(Matrix([[1]]), Matrix([[2]]), Matrix([[4.5]]), Matrix([[6.0]]))
    assert 1.5*ss4 == StateSpace(Matrix([[1]]), Matrix([[3.0]]), Matrix([[3]]), Matrix([[6.0]]))
    raises(ShapeError, lambda: ss1 + ss3)
    raises(ShapeError, lambda: ss2*ss4)

def test_StateSpace_negation():
    A = Matrix([[a0, a1], [a2, a3]])
    B = Matrix([[b0, b1], [b2, b3]])
    C = Matrix([[c0, c1], [c1, c2], [c2, c3]])
    D = Matrix([[d0, d1], [d1, d2], [d2, d3]])
    SS = StateSpace(A, B, C, D)
    SS_neg = -SS

    state_mat = Matrix([[-1, 1], [1, -1]])
    input_mat = Matrix([1, -1])
    output_mat = Matrix([[-1, 1]])
    feedforward_mat = Matrix([1])
    system = StateSpace(state_mat, input_mat, output_mat, feedforward_mat)

    assert SS_neg == \
        StateSpace(Matrix([[a0, a1],
                           [a2, a3]]),
                   Matrix([[b0, b1],
                           [b2, b3]]),
                   Matrix([[-c0, -c1],
                           [-c1, -c2],
                           [-c2, -c3]]),
                   Matrix([[-d0, -d1],
                           [-d1, -d2],
                           [-d2, -d3]]))
    assert -system == \
        StateSpace(Matrix([[-1,  1],
                           [ 1, -1]]),
                   Matrix([[ 1],[-1]]),
                   Matrix([[1, -1]]),
                   Matrix([[-1]]))
    assert -SS_neg == SS
    assert -(-(-(-system))) == system

def test_SymPy_substitution_functions():
    # subs
    ss1 = StateSpace(Matrix([s]), Matrix([(s + 1)**2]), Matrix([s**2 - 1]), Matrix([2*s]))
    ss2 = StateSpace(Matrix([s + p]), Matrix([(s + 1)*(p - 1)]), Matrix([p**3 - s**3]), Matrix([s - p]))

    assert ss1.subs({s:5}) == StateSpace(Matrix([[5]]), Matrix([[36]]), Matrix([[24]]), Matrix([[10]]))
    assert ss2.subs({p:1}) == StateSpace(Matrix([[s + 1]]), Matrix([[0]]), Matrix([[1 - s**3]]), Matrix([[s - 1]]))

    # xreplace
    assert ss1.xreplace({s:p}) == \
        StateSpace(Matrix([[p]]), Matrix([[(p + 1)**2]]), Matrix([[p**2 - 1]]), Matrix([[2*p]]))
    assert ss2.xreplace({s:a, p:b}) == \
        StateSpace(Matrix([[a + b]]), Matrix([[(a + 1)*(b - 1)]]), Matrix([[-a**3 + b**3]]), Matrix([[a - b]]))

    # evalf
    p1 = a1*s + a0
    p2 = b2*s**2 + b1*s + b0
    G = StateSpace(Matrix([p1]), Matrix([p2]))
    expect = StateSpace(Matrix([[2*s + 1]]), Matrix([[5*s**2 + 4*s + 3]]), Matrix([[0]]), Matrix([[0]]))
    expect_ = StateSpace(Matrix([[2.0*s + 1.0]]), Matrix([[5.0*s**2 + 4.0*s + 3.0]]), Matrix([[0]]), Matrix([[0]]))
    assert G.subs({a0: 1, a1: 2, b0: 3, b1: 4, b2: 5}) == expect
    assert G.subs({a0: 1, a1: 2, b0: 3, b1: 4, b2: 5}).evalf() == expect_
    assert expect.evalf() == expect_

def test_conversion():
    # StateSpace to TransferFunction for SISO
    A1 = Matrix([[-5, -1], [3, -1]])
    B1 = Matrix([2, 5])
    C1 = Matrix([[1, 2]])
    D1 = Matrix([0])
    H1 = StateSpace(A1, B1, C1, D1)
    H3 = StateSpace(Matrix([[a0, a1], [a2, a3]]), B = Matrix([[b1], [b2]]), C = Matrix([[c1, c2]]))
    tm1 = H1.rewrite(TransferFunction)
    tm2 = (-H1).rewrite(TransferFunction)

    tf1 = tm1[0][0]
    tf2 = tm2[0][0]

    assert tf1 == TransferFunction(12*s + 59, s**2 + 6*s + 8, s)
    assert tf2.num == -tf1.num
    assert tf2.den == tf1.den

    # StateSpace to TransferFunction for MIMO
    A2 = Matrix([[-1.5, -2, 3], [1, 0, 1], [2, 1, 1]])
    B2 = Matrix([[0.5, 0, 1], [0, 1, 2], [2, 2, 3]])
    C2 = Matrix([[0, 1, 0], [0, 2, 1], [1, 0, 2]])
    D2 = Matrix([[2, 2, 0], [1, 1, 1], [3, 2, 1]])
    H2 = StateSpace(A2, B2, C2, D2)
    tm3 = H2.rewrite(TransferFunction)

    # outputs for input i obtained at Index i-1. Consider input 1
    assert tm3[0][0] == TransferFunction(2.0*s**3 + 1.0*s**2 - 10.5*s + 4.5, 1.0*s**3 + 0.5*s**2 - 6.5*s - 2.5, s)
    assert tm3[0][1] == TransferFunction(2.0*s**3 + 2.0*s**2 - 10.5*s - 3.5, 1.0*s**3 + 0.5*s**2 - 6.5*s - 2.5, s)
    assert tm3[0][2] == TransferFunction(2.0*s**2 + 5.0*s - 0.5, 1.0*s**3 + 0.5*s**2 - 6.5*s - 2.5, s)
    assert H3.rewrite(TransferFunction) == [[TransferFunction(-c1*(a1*b2 - a3*b1 + b1*s) - c2*(-a0*b2 + a2*b1 + b2*s),
                                                              -a0*a3 + a0*s + a1*a2 + a3*s - s**2, s)]]
    # TransferFunction to StateSpace
    SS = TF1.rewrite(StateSpace)
    assert SS == \
        StateSpace(Matrix([[     0,          1],
                           [-wn**2, -2*wn*zeta]]),
                   Matrix([[0],
                           [1]]),
                   Matrix([[1, 0]]),
                   Matrix([[0]]))
    assert SS.rewrite(TransferFunction)[0][0] == TF1

    # Transfer function has to be proper
    raises(ValueError, lambda: TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s).rewrite(StateSpace))


def test_StateSpace_dsolve():
    # https://web.mit.edu/2.14/www/Handouts/StateSpaceResponse.pdf
    # https://lpsa.swarthmore.edu/Transient/TransMethSS.html
    A1 = Matrix([[0, 1], [-2, -3]])
    B1 = Matrix([[0], [1]])
    C1 = Matrix([[1, -1]])
    D1 = Matrix([0])
    I1 = Matrix([[1], [2]])
    t = symbols('t')
    ss1 = StateSpace(A1, B1, C1, D1)

    # Zero input and Zero initial conditions
    assert ss1.dsolve() == Matrix([[0]])
    assert ss1.dsolve(initial_conditions=I1) == Matrix([[8*exp(-t) - 9*exp(-2*t)]])

    A2 = Matrix([[-2, 0], [1, -1]])
    C2 = eye(2,2)
    I2 = Matrix([2, 3])
    ss2 = StateSpace(A=A2, C=C2)
    assert ss2.dsolve(initial_conditions=I2) == Matrix([[2*exp(-2*t)], [5*exp(-t) - 2*exp(-2*t)]])

    A3 = Matrix([[-1, 1], [-4, -4]])
    B3 = Matrix([[0], [4]])
    C3 = Matrix([[0, 1]])
    D3 = Matrix([0])
    U3 = Matrix([10])
    ss3 = StateSpace(A3, B3, C3, D3)
    op = ss3.dsolve(input_vector=U3, var=t)
    assert str(op.simplify().expand().evalf()[0]) == str(5.0 + 20.7880460155075*exp(-5*t/2)*sin(sqrt(7)*t/2)
                                            - 5.0*exp(-5*t/2)*cos(sqrt(7)*t/2))

    # Test with Heaviside as input
    A4 = Matrix([[-1, 1], [-4, -4]])
    B4 = Matrix([[0], [4]])
    C4 = Matrix([[0, 1]])
    U4 = Matrix([[10*Heaviside(t)]])
    ss4 = StateSpace(A4, B4, C4)
    op4 = str(ss4.dsolve(var=t, input_vector=U4)[0].simplify().expand().evalf())
    assert op4 == str(5.0*Heaviside(t) + 20.7880460155075*exp(-5*t/2)*sin(sqrt(7)*t/2)*Heaviside(t)
                                            - 5.0*exp(-5*t/2)*cos(sqrt(7)*t/2)*Heaviside(t))

    # Test with Symbolic Matrices
    m, a, x0 = symbols('m a x_0')
    A5 = Matrix([[0, 1], [0, 0]])
    B5 = Matrix([[0], [1 / m]])
    C5 = Matrix([[1, 0]])
    I5 = Matrix([[x0], [0]])
    U5 = Matrix([[exp(-a * t)]])
    ss5 = StateSpace(A5, B5, C5)
    op5 = ss5.dsolve(initial_conditions=I5, input_vector=U5, var=t).simplify()
    assert op5[0].args[0][0] == x0 + t/(a*m) - 1/(a**2*m) + exp(-a*t)/(a**2*m)
    a11, a12, a21, a22, b1, b2, c1, c2, i1, i2 = symbols('a_11 a_12 a_21 a_22 b_1 b_2 c_1 c_2 i_1 i_2')
    A6 = Matrix([[a11, a12], [a21, a22]])
    B6 = Matrix([b1, b2])
    C6 = Matrix([[c1, c2]])
    I6 = Matrix([i1, i2])
    ss6 = StateSpace(A6, B6, C6)
    expr6 = ss6.dsolve(initial_conditions=I6)[0]
    expr6 = expr6.subs([(a11, 0), (a12, 1), (a21, -2), (a22, -3), (b1, 0), (b2, 1), (c1, 1), (c2, -1), (i1, 1), (i2, 2)])
    assert expr6 == 8*exp(-t) - 9*exp(-2*t)


def test_StateSpace_functions():
    # https://in.mathworks.com/help/control/ref/statespacemodel.obsv.html

    A_mat = Matrix([[-1.5, -2], [1, 0]])
    B_mat = Matrix([0.5, 0])
    C_mat = Matrix([[0, 1]])
    D_mat = Matrix([1])
    SS1 = StateSpace(A_mat, B_mat, C_mat, D_mat)
    SS2 = StateSpace(Matrix([[1, 1], [4, -2]]),Matrix([[0, 1], [0, 2]]),Matrix([[-1, 1], [1, -1]]))
    SS3 = StateSpace(Matrix([[1, 1], [4, -2]]),Matrix([[1, -1], [1, -1]]))
    SS4 = StateSpace(Matrix([[a0, a1], [a2, a3]]), Matrix([[b1], [b2]]), Matrix([[c1, c2]]))

    # Observability
    assert SS1.is_observable() == True
    assert SS2.is_observable() == False
    assert SS1.observability_matrix() == Matrix([[0, 1], [1, 0]])
    assert SS2.observability_matrix() == Matrix([[-1,  1], [ 1, -1], [ 3, -3], [-3,  3]])
    assert SS1.observable_subspace() == [Matrix([[0], [1]]), Matrix([[1], [0]])]
    assert SS2.observable_subspace() == [Matrix([[-1], [ 1], [ 3], [-3]])]
    Qo = SS4.observability_matrix().subs([(a0, 0), (a1, -6), (a2, 1), (a3, -5), (c1, 0), (c2, 1)])
    assert Qo == Matrix([[0, 1], [1, -5]])

    # Controllability
    assert SS1.is_controllable() == True
    assert SS3.is_controllable() == False
    assert SS1.controllability_matrix() ==  Matrix([[0.5, -0.75], [  0,   0.5]])
    assert SS3.controllability_matrix() == Matrix([[1, -1, 2, -2], [1, -1, 2, -2]])
    assert SS1.controllable_subspace() == [Matrix([[0.5], [  0]]), Matrix([[-0.75], [  0.5]])]
    assert SS3.controllable_subspace() == [Matrix([[1], [1]])]
    assert SS4.controllable_subspace() == [Matrix([
                                          [b1],
                                          [b2]]), Matrix([
                                          [a0*b1 + a1*b2],
                                          [a2*b1 + a3*b2]])]
    Qc = SS4.controllability_matrix().subs([(a0, 0), (a1, 1), (a2, -6), (a3, -5), (b1, 0), (b2, 1)])
    assert Qc == Matrix([[0, 1], [1, -5]])

    # Append
    A1 = Matrix([[0, 1], [1, 0]])
    B1 = Matrix([[0], [1]])
    C1 = Matrix([[0, 1]])
    D1 = Matrix([[0]])
    ss1 = StateSpace(A1, B1, C1, D1)
    ss2 = StateSpace(Matrix([[1, 0], [0, 1]]), Matrix([[1], [0]]), Matrix([[1, 0]]), Matrix([[1]]))
    ss3 = ss1.append(ss2)
    ss4 = SS4.append(ss1)

    assert ss3.num_states == ss1.num_states + ss2.num_states
    assert ss3.num_inputs == ss1.num_inputs + ss2.num_inputs
    assert ss3.num_outputs == ss1.num_outputs + ss2.num_outputs
    assert ss3.state_matrix == Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert ss3.input_matrix == Matrix([[0, 0], [1, 0], [0, 1], [0, 0]])
    assert ss3.output_matrix == Matrix([[0, 1, 0, 0], [0, 0, 1, 0]])
    assert ss3.feedforward_matrix == Matrix([[0, 0], [0, 1]])

    # Using symbolic matrices
    assert ss4.num_states == SS4.num_states + ss1.num_states
    assert ss4.num_inputs == SS4.num_inputs + ss1.num_inputs
    assert ss4.num_outputs == SS4.num_outputs + ss1.num_outputs
    assert ss4.state_matrix == Matrix([[a0, a1, 0, 0], [a2, a3, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert ss4.input_matrix == Matrix([[b1, 0], [b2, 0], [0, 0], [0, 1]])
    assert ss4.output_matrix == Matrix([[c1, c2, 0, 0], [0, 0, 0, 1]])
    assert ss4.feedforward_matrix == Matrix([[0, 0], [0, 0]])


def test_StateSpace_series():
    # For SISO Systems
    a1 = Matrix([[0, 1], [1, 0]])
    b1 = Matrix([[0], [1]])
    c1 = Matrix([[0, 1]])
    d1 = Matrix([[0]])
    a2 = Matrix([[1, 0], [0, 1]])
    b2 = Matrix([[1], [0]])
    c2 = Matrix([[1, 0]])
    d2 = Matrix([[1]])

    ss1 = StateSpace(a1, b1, c1, d1)
    ss2 = StateSpace(a2, b2, c2, d2)
    tf1 = TransferFunction(s, s+1, s)
    ser1 = Series(ss1, ss2)
    assert ser1 == Series(StateSpace(Matrix([
                            [0, 1],
                            [1, 0]]), Matrix([
                            [0],
                            [1]]), Matrix([[0, 1]]), Matrix([[0]])), StateSpace(Matrix([
                            [1, 0],
                            [0, 1]]), Matrix([
                            [1],
                            [0]]), Matrix([[1, 0]]), Matrix([[1]])))
    assert ser1.doit() == StateSpace(
                            Matrix([
                            [0, 1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 1, 1, 0],
                            [0, 0, 0, 1]]),
                            Matrix([
                            [0],
                            [1],
                            [0],
                            [0]]),
                            Matrix([[0, 1, 1, 0]]),
                            Matrix([[0]]))

    assert ser1.num_inputs == 1
    assert ser1.num_outputs == 1
    assert ser1.rewrite(TransferFunction) == TransferFunction(s**2, s**3 - s**2 - s + 1, s)
    ser2 = Series(ss1)
    ser3 = Series(ser2, ss2)
    assert ser3.doit() == ser1.doit()

    # TransferFunction interconnection with StateSpace
    ser_tf = Series(tf1, ss1)
    assert ser_tf == Series(TransferFunction(s, s + 1, s), StateSpace(Matrix([
                            [0, 1],
                            [1, 0]]), Matrix([
                            [0],
                            [1]]), Matrix([[0, 1]]), Matrix([[0]])))
    assert ser_tf.doit() == StateSpace(
                            Matrix([
                            [-1, 0,  0],
                            [0, 0,  1],
                            [-1, 1, 0]]),
                            Matrix([
                            [1],
                            [0],
                            [1]]),
                            Matrix([[0, 0, 1]]),
                            Matrix([[0]]))
    assert ser_tf.rewrite(TransferFunction) == TransferFunction(s**2, s**3 + s**2 - s - 1, s)

    # For MIMO Systems
    a3 = Matrix([[4, 1], [2, -3]])
    b3 = Matrix([[5, 2], [-3, -3]])
    c3 = Matrix([[2, -4], [0, 1]])
    d3 = Matrix([[3, 2], [1, -1]])
    a4 = Matrix([[-3, 4, 2], [-1, -3, 0], [2, 5, 3]])
    b4 = Matrix([[1, 4], [-3, -3], [-2, 1]])
    c4 = Matrix([[4, 2, -3], [1, 4, 3]])
    d4 = Matrix([[-2, 4], [0, 1]])
    ss3 = StateSpace(a3, b3, c3, d3)
    ss4 = StateSpace(a4, b4, c4, d4)
    ser4 = MIMOSeries(ss3, ss4)
    assert ser4 == MIMOSeries(StateSpace(Matrix([
                    [4,  1],
                    [2, -3]]), Matrix([
                    [ 5,  2],
                    [-3, -3]]), Matrix([
                    [2, -4],
                    [0,  1]]), Matrix([
                    [3,  2],
                    [1, -1]])), StateSpace(Matrix([
                    [-3,  4, 2],
                    [-1, -3, 0],
                    [ 2,  5, 3]]), Matrix([
                    [ 1,  4],
                    [-3, -3],
                    [-2,  1]]), Matrix([
                    [4, 2, -3],
                    [1, 4,  3]]), Matrix([
                    [-2, 4],
                    [ 0, 1]])))
    assert ser4.doit() == StateSpace(
                        Matrix([
                        [4,   1,  0, 0,  0],
                        [2,  -3,  0, 0,  0],
                        [2,   0,  -3, 4,  2],
                        [-6,  9, -1, -3,  0],
                        [-4, 9,  2, 5, 3]]),
                        Matrix([
                        [5,   2],
                        [-3,  -3],
                        [7,   -2],
                        [-12,  -3],
                        [-5, -5]]),
                        Matrix([
                        [-4, 12, 4, 2, -3],
                        [0, 1, 1, 4, 3]]),
                        Matrix([
                        [-2, -8],
                        [1, -1]]))
    assert ser4.num_inputs == ss3.num_inputs
    assert ser4.num_outputs == ss4.num_outputs
    ser5 = MIMOSeries(ss3)
    ser6 = MIMOSeries(ser5, ss4)
    assert ser6.doit() == ser4.doit()
    assert ser6.rewrite(TransferFunctionMatrix) == ser4.rewrite(TransferFunctionMatrix)
    tf2 = TransferFunction(1, s, s)
    tf3 = TransferFunction(1, s+1, s)
    tf4 = TransferFunction(s, s+2, s)
    tfm = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
    ser6 = MIMOSeries(ss3, tfm)
    assert ser6 == MIMOSeries(StateSpace(Matrix([
                        [4,  1],
                        [2, -3]]), Matrix([
                        [ 5,  2],
                        [-3, -3]]), Matrix([
                        [2, -4],
                        [0,  1]]), Matrix([
                        [3,  2],
                        [1, -1]])), TransferFunctionMatrix((
                        (TransferFunction(s, s + 1, s), TransferFunction(1, s, s)),
                        (TransferFunction(1, s + 1, s), TransferFunction(s, s + 2, s)))))


def test_StateSpace_parallel():
    # For SISO system
    a1 = Matrix([[0, 1], [1, 0]])
    b1 = Matrix([[0], [1]])
    c1 = Matrix([[0, 1]])
    d1 = Matrix([[0]])
    a2 = Matrix([[1, 0], [0, 1]])
    b2 = Matrix([[1], [0]])
    c2 = Matrix([[1, 0]])
    d2 = Matrix([[1]])
    ss1 = StateSpace(a1, b1, c1, d1)
    ss2 = StateSpace(a2, b2, c2, d2)
    p1 = Parallel(ss1, ss2)
    assert p1 == Parallel(StateSpace(Matrix([[0, 1], [1, 0]]), Matrix([[0], [1]]), Matrix([[0, 1]]), Matrix([[0]])),
                          StateSpace(Matrix([[1, 0],[0, 1]]), Matrix([[1],[0]]), Matrix([[1, 0]]), Matrix([[1]])))
    assert p1.doit() == StateSpace(Matrix([
                        [0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]),
                        Matrix([
                        [0],
                        [1],
                        [1],
                        [0]]),
                        Matrix([[0, 1, 1, 0]]),
                        Matrix([[1]]))
    assert p1.rewrite(TransferFunction) == TransferFunction(s*(s + 2), s**2 - 1, s)

    # Connecting StateSpace with TransferFunction
    tf1 = TransferFunction(s, s+1, s)
    p2 = Parallel(ss1, tf1)
    assert p2 == Parallel(StateSpace(Matrix([
                        [0, 1],
                        [1, 0]]), Matrix([
                        [0],
                        [1]]), Matrix([[0, 1]]), Matrix([[0]])), TransferFunction(s, s + 1, s))
    assert p2.doit() == StateSpace(
                        Matrix([
                        [0, 1,  0],
                        [1, 0,  0],
                        [0, 0, -1]]),
                        Matrix([
                        [0],
                        [1],
                        [1]]),
                        Matrix([[0, 1, -1]]),
                        Matrix([[1]]))
    assert p2.rewrite(TransferFunction) == TransferFunction(s**2, s**2 - 1, s)

    # For MIMO
    a3 = Matrix([[4, 1], [2, -3]])
    b3 = Matrix([[5, 2], [-3, -3]])
    c3 = Matrix([[2, -4], [0, 1]])
    d3 = Matrix([[3, 2], [1, -1]])
    a4 = Matrix([[-3, 4, 2], [-1, -3, 0], [2, 5, 3]])
    b4 = Matrix([[1, 4], [-3, -3], [-2, 1]])
    c4 = Matrix([[4, 2, -3], [1, 4, 3]])
    d4 = Matrix([[-2, 4], [0, 1]])
    ss3 = StateSpace(a3, b3, c3, d3)
    ss4 = StateSpace(a4, b4, c4, d4)
    p3 = MIMOParallel(ss3, ss4)
    assert p3 == MIMOParallel(StateSpace(Matrix([
                        [4,  1],
                        [2, -3]]), Matrix([
                        [ 5,  2],
                        [-3, -3]]), Matrix([
                        [2, -4],
                        [0,  1]]), Matrix([
                        [3,  2],
                        [1, -1]])), StateSpace(Matrix([
                        [-3,  4, 2],
                        [-1, -3, 0],
                        [ 2,  5, 3]]), Matrix([
                        [ 1,  4],
                        [-3, -3],
                        [-2,  1]]), Matrix([
                        [4, 2, -3],
                        [1, 4,  3]]), Matrix([
                        [-2, 4],
                        [ 0, 1]])))
    assert p3.doit() == StateSpace(Matrix([
                        [4, 1, 0, 0, 0],
                        [2, -3, 0, 0, 0],
                        [0, 0, -3, 4, 2],
                        [0, 0, -1, -3, 0],
                        [0, 0, 2, 5, 3]]),
                        Matrix([
                        [5, 2],
                        [-3, -3],
                        [1, 4],
                        [-3, -3],
                        [-2, 1]]),
                        Matrix([
                        [2, -4, 4, 2, -3],
                        [0, 1, 1, 4, 3]]),
                        Matrix([
                        [1, 6],
                        [1, 0]]))

    # Using StateSpace with MIMOParallel.
    tf2 = TransferFunction(1, s, s)
    tf3 = TransferFunction(1, s + 1, s)
    tf4 = TransferFunction(s, s + 2, s)
    tfm = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
    p4 = MIMOParallel(tfm, ss3)
    assert p4 == MIMOParallel(TransferFunctionMatrix((
                        (TransferFunction(s, s + 1, s), TransferFunction(1, s, s)),
                        (TransferFunction(1, s + 1, s), TransferFunction(s, s + 2, s)))),
                        StateSpace(Matrix([
                        [4, 1],
                        [2, -3]]), Matrix([
                        [5, 2],
                        [-3, -3]]), Matrix([
                        [2, -4],
                        [0, 1]]), Matrix([
                        [3, 2],
                        [1, -1]])))


def test_StateSpace_feedback():
    # For SISO
    a1 = Matrix([[0, 1], [1, 0]])
    b1 = Matrix([[0], [1]])
    c1 = Matrix([[0, 1]])
    d1 = Matrix([[0]])
    a2 = Matrix([[1, 0], [0, 1]])
    b2 = Matrix([[1], [0]])
    c2 = Matrix([[1, 0]])
    d2 = Matrix([[1]])
    ss1 = StateSpace(a1, b1, c1, d1)
    ss2 = StateSpace(a2, b2, c2, d2)
    fd1 = Feedback(ss1, ss2)

    # Negative feedback
    assert fd1 == Feedback(StateSpace(Matrix([[0, 1], [1, 0]]), Matrix([[0], [1]]), Matrix([[0, 1]]), Matrix([[0]])),
                          StateSpace(Matrix([[1, 0],[0, 1]]), Matrix([[1],[0]]), Matrix([[1, 0]]), Matrix([[1]])), -1)
    assert fd1.doit() == StateSpace(Matrix([
                            [0,  1,  0, 0],
                            [1, -1, -1, 0],
                            [0,  1,  1, 0],
                            [0,  0,  0, 1]]), Matrix([
                            [0],
                            [1],
                            [0],
                            [0]]), Matrix(
                            [[0, 1, 0, 0]]), Matrix(
                            [[0]]))
    assert fd1.rewrite(TransferFunction) == TransferFunction(s*(s - 1), s**3 - s + 1, s)

    # Positive Feedback
    fd2 = Feedback(ss1, ss2, 1)
    assert fd2.doit() == StateSpace(Matrix([
                            [0, 1, 0, 0],
                            [1, 1, 1, 0],
                            [0, 1, 1, 0],
                            [0, 0, 0, 1]]), Matrix([
                            [0],
                            [1],
                            [0],
                            [0]]), Matrix(
                            [[0, 1, 0, 0]]), Matrix(
                            [[0]]))
    assert fd2.rewrite(TransferFunction) == TransferFunction(s*(s - 1), s**3 - 2*s**2 - s + 1, s)

    # Connection with TransferFunction
    tf1 = TransferFunction(s, s+1, s)
    fd3 = Feedback(ss1, tf1)
    assert fd3 == Feedback(StateSpace(Matrix([
                            [0, 1],
                            [1, 0]]), Matrix([
                            [0],
                            [1]]), Matrix([[0, 1]]), Matrix([[0]])),
                            TransferFunction(s, s + 1, s), -1)
    assert fd3.doit() == StateSpace (Matrix([
                            [0,  1,  0],
                            [1, -1,  1],
                            [0,  1, -1]]), Matrix([
                            [0],
                            [1],
                            [0]]), Matrix(
                            [[0, 1, 0]]), Matrix(
                            [[0]]))

    # For MIMO
    a3 = Matrix([[4, 1], [2, -3]])
    b3 = Matrix([[5, 2], [-3, -3]])
    c3 = Matrix([[2, -4], [0, 1]])
    d3 = Matrix([[3, 2], [1, -1]])
    a4 = Matrix([[-3, 4, 2], [-1, -3, 0], [2, 5, 3]])
    b4 = Matrix([[1, 4], [-3, -3], [-2, 1]])
    c4 = Matrix([[4, 2, -3], [1, 4, 3]])
    d4 = Matrix([[-2, 4], [0, 1]])
    ss3 = StateSpace(a3, b3, c3, d3)
    ss4 = StateSpace(a4, b4, c4, d4)

    # Negative Feedback
    fd4 = MIMOFeedback(ss3, ss4)
    assert fd4 == MIMOFeedback(StateSpace(Matrix([
                            [4,  1],
                            [2, -3]]), Matrix([
                            [ 5,  2],
                            [-3, -3]]), Matrix([
                            [2, -4],
                            [0,  1]]), Matrix([
                            [3,  2],
                            [1, -1]])), StateSpace(Matrix([
                            [-3,  4, 2],
                            [-1, -3, 0],
                            [ 2,  5, 3]]), Matrix([
                            [ 1,  4],
                            [-3, -3],
                            [-2,  1]]), Matrix([
                            [4, 2, -3],
                            [1, 4,  3]]), Matrix([
                            [-2, 4],
                            [ 0, 1]])), -1)
    assert fd4.doit() == StateSpace(Matrix([
                            [Rational(3), Rational(-3, 4), Rational(-15, 4), Rational(-37, 2), Rational(-15)],
                            [Rational(7, 2), Rational(-39, 8), Rational(9, 8), Rational(39, 4), Rational(9)],
                            [Rational(3), Rational(-41, 4), Rational(-45, 4), Rational(-51, 2), Rational(-19)],
                            [Rational(-9, 2), Rational(129, 8), Rational(73, 8), Rational(171, 4), Rational(36)],
                            [Rational(-3, 2), Rational(47, 8), Rational(31, 8), Rational(85, 4), Rational(18)]]), Matrix([
                            [Rational(-1, 4), Rational(19, 4)],
                            [Rational(3, 8), Rational(-21, 8)],
                            [Rational(1, 4), Rational(29, 4)],
                            [Rational(3, 8), Rational(-93, 8)],
                            [Rational(5, 8), Rational(-35, 8)]]), Matrix([
                            [Rational(1), Rational(-15, 4), Rational(-7, 4), Rational(-21, 2), Rational(-9)],
                            [Rational(1, 2), Rational(-13, 8), Rational(-13, 8), Rational(-19, 4), Rational(-3)]]), Matrix([
                            [Rational(-1, 4), Rational(11, 4)],
                            [Rational(1, 8), Rational(9, 8)]]))

    # Positive Feedback
    fd5 = MIMOFeedback(ss3, ss4, 1)
    assert fd5.doit() == StateSpace(Matrix([
                            [Rational(4, 7), Rational(62, 7), Rational(1), Rational(-8), Rational(-69, 7)],
                            [Rational(32, 7), Rational(-135, 14), Rational(-3, 2), Rational(3), Rational(36, 7)],
                            [Rational(-10, 7), Rational(41, 7), Rational(-4), Rational(-12), Rational(-97, 7)],
                            [Rational(12, 7), Rational(-111, 14), Rational(-5, 2), Rational(18), Rational(171, 7)],
                            [Rational(2, 7), Rational(-29, 14), Rational(-1, 2), Rational(10), Rational(81, 7)]]), Matrix([
                            [Rational(6, 7), Rational(-17, 7)],
                            [Rational(-9, 14), Rational(15, 14)],
                            [Rational(6, 7), Rational(-31, 7)],
                            [Rational(-27, 14), Rational(87, 14)],
                            [Rational(-15, 14), Rational(25, 14)]]), Matrix([
                            [Rational(-2, 7), Rational(11, 7), Rational(1), Rational(-4), Rational(-39, 7)],
                            [Rational(-2, 7), Rational(15, 14), Rational(-1, 2), Rational(-3), Rational(-18, 7)]]), Matrix([
                            [Rational(4, 7), Rational(-9, 7)],
                            [Rational(1, 14), Rational(-11, 14)]]))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\t5\t5_encoder.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

import logging
import random

import torch
from transformers import MT5Config, T5Config

logger = logging.getLogger(__name__)


class T5Encoder(torch.nn.Module):
    """T5 encoder outputs only the last hidden state"""

    def __init__(self, encoder, config: T5Config | MT5Config):
        super().__init__()
        self.encoder = encoder
        self.config = config

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids, attention_mask)[0]


class T5EncoderInputs:
    def __init__(self, input_ids, attention_mask):
        self.input_ids: torch.LongTensor = input_ids
        self.attention_mask: torch.LongTensor = attention_mask

    @staticmethod
    def create_dummy(
        batch_size: int,
        sequence_length: int,
        vocab_size: int,
        device: torch.device,
        use_int32_inputs: bool = False,
    ):  # -> T5EncoderInputs
        """Create dummy inputs for T5 encoder.

        Args:
            batch_size (int): batch size
            sequence_length (int): sequence length
            vocab_size (int): vocabulary size
            device (torch.device): device of output tensors

        Returns:
            T5EncoderInputs: dummy inputs for encoder
        """
        dtype = torch.int32 if use_int32_inputs else torch.int64

        input_ids = torch.randint(
            low=0,
            high=vocab_size - 1,
            size=(batch_size, sequence_length),
            dtype=dtype,
            device=device,
        )

        attention_mask = torch.ones([batch_size, sequence_length], dtype=dtype, device=device)
        if sequence_length >= 2:
            for i in range(batch_size):
                padding_position = random.randint(0, sequence_length - 1)
                attention_mask[i, :padding_position] = 0
        return T5EncoderInputs(input_ids, attention_mask)

    def to_list(self) -> list:
        input_list = [v for v in [self.input_ids, self.attention_mask] if v is not None]
        return input_list

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\copier.py ===
# Copyright (c) 2010-2024 openpyxl

#standard lib imports
from copy import copy

from .worksheet import Worksheet


class WorksheetCopy:
    """
    Copy the values, styles, dimensions, merged cells, margins, and
    print/page setup from one worksheet to another within the same
    workbook.
    """

    def __init__(self, source_worksheet, target_worksheet):
        self.source = source_worksheet
        self.target = target_worksheet
        self._verify_resources()


    def _verify_resources(self):

        if (not isinstance(self.source, Worksheet)
            and not isinstance(self.target, Worksheet)):
            raise TypeError("Can only copy worksheets")

        if self.source is self.target:
            raise ValueError("Cannot copy a worksheet to itself")

        if self.source.parent != self.target.parent:
            raise ValueError('Cannot copy between worksheets from different workbooks')


    def copy_worksheet(self):
        self._copy_cells()
        self._copy_dimensions()

        self.target.sheet_format = copy(self.source.sheet_format)
        self.target.sheet_properties = copy(self.source.sheet_properties)
        self.target.merged_cells = copy(self.source.merged_cells)
        self.target.page_margins = copy(self.source.page_margins)
        self.target.page_setup = copy(self.source.page_setup)
        self.target.print_options = copy(self.source.print_options)


    def _copy_cells(self):
        for (row, col), source_cell  in self.source._cells.items():
            target_cell = self.target.cell(column=col, row=row)

            target_cell._value = source_cell._value
            target_cell.data_type = source_cell.data_type

            if source_cell.has_style:
                target_cell._style = copy(source_cell._style)

            if source_cell.hyperlink:
                target_cell._hyperlink = copy(source_cell.hyperlink)

            if source_cell.comment:
                target_cell.comment = copy(source_cell.comment)


    def _copy_dimensions(self):
        for attr in ('row_dimensions', 'column_dimensions'):
            src = getattr(self.source, attr)
            target = getattr(self.target, attr)
            for key, dim in src.items():
                target[key] = copy(dim)
                target[key].worksheet = self.target

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\cachecontrol\_cmd.py ===
# SPDX-FileCopyrightText: 2015 Eric Larson
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from argparse import ArgumentParser
from typing import TYPE_CHECKING

from pip._vendor import requests

from pip._vendor.cachecontrol.adapter import CacheControlAdapter
from pip._vendor.cachecontrol.cache import DictCache
from pip._vendor.cachecontrol.controller import logger

if TYPE_CHECKING:
    from argparse import Namespace

    from pip._vendor.cachecontrol.controller import CacheController


def setup_logging() -> None:
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    logger.addHandler(handler)


def get_session() -> requests.Session:
    adapter = CacheControlAdapter(
        DictCache(), cache_etags=True, serializer=None, heuristic=None
    )
    sess = requests.Session()
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)

    sess.cache_controller = adapter.controller  # type: ignore[attr-defined]
    return sess


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("url", help="The URL to try and cache")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    sess = get_session()

    # Make a request to get a response
    resp = sess.get(args.url)

    # Turn on logging
    setup_logging()

    # try setting the cache
    cache_controller: CacheController = (
        sess.cache_controller  # type: ignore[attr-defined]
    )
    cache_controller.cache_response(resp.request, resp.raw)

    # Now try to get it
    if cache_controller.cached_request(resp.request):
        print("Cached!")
    else:
        print("Not cached :(")


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\console.py ===
"""
    pygments.console
    ~~~~~~~~~~~~~~~~

    Format colored console output.

    :copyright: Copyright 2006-2025 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

esc = "\x1b["

codes = {}
codes[""] = ""
codes["reset"] = esc + "39;49;00m"

codes["bold"] = esc + "01m"
codes["faint"] = esc + "02m"
codes["standout"] = esc + "03m"
codes["underline"] = esc + "04m"
codes["blink"] = esc + "05m"
codes["overline"] = esc + "06m"

dark_colors = ["black", "red", "green", "yellow", "blue",
               "magenta", "cyan", "gray"]
light_colors = ["brightblack", "brightred", "brightgreen", "brightyellow", "brightblue",
                "brightmagenta", "brightcyan", "white"]

x = 30
for dark, light in zip(dark_colors, light_colors):
    codes[dark] = esc + "%im" % x
    codes[light] = esc + "%im" % (60 + x)
    x += 1

del dark, light, x

codes["white"] = codes["bold"]


def reset_color():
    return codes["reset"]


def colorize(color_key, text):
    return codes[color_key] + text + codes["reset"]


def ansiformat(attr, text):
    """
    Format ``text`` with a color and/or some attributes::

        color       normal color
        *color*     bold color
        _color_     underlined color
        +color+     blinking color
    """
    result = []
    if attr[:1] == attr[-1:] == '+':
        result.append(codes['blink'])
        attr = attr[1:-1]
    if attr[:1] == attr[-1:] == '*':
        result.append(codes['bold'])
        attr = attr[1:-1]
    if attr[:1] == attr[-1:] == '_':
        result.append(codes['underline'])
        attr = attr[1:-1]
    result.append(codes[attr])
    result.append(text)
    result.append(codes['reset'])
    return ''.join(result)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\filter.py ===
"""
    pygments.filter
    ~~~~~~~~~~~~~~~

    Module that implements the default filter.

    :copyright: Copyright 2006-2025 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""


def apply_filters(stream, filters, lexer=None):
    """
    Use this method to apply an iterable of filters to
    a stream. If lexer is given it's forwarded to the
    filter, otherwise the filter receives `None`.
    """
    def _apply(filter_, stream):
        yield from filter_.filter(lexer, stream)
    for filter_ in filters:
        stream = _apply(filter_, stream)
    return stream


def simplefilter(f):
    """
    Decorator that converts a function into a filter::

        @simplefilter
        def lowercase(self, lexer, stream, options):
            for ttype, value in stream:
                yield ttype, value.lower()
    """
    return type(f.__name__, (FunctionFilter,), {
        '__module__': getattr(f, '__module__'),
        '__doc__': f.__doc__,
        'function': f,
    })


class Filter:
    """
    Default filter. Subclass this class or use the `simplefilter`
    decorator to create own filters.
    """

    def __init__(self, **options):
        self.options = options

    def filter(self, lexer, stream):
        raise NotImplementedError()


class FunctionFilter(Filter):
    """
    Abstract class used by `simplefilter` to create simple
    function filters on the fly. The `simplefilter` decorator
    automatically creates subclasses of this class for
    functions passed to it.
    """
    function = None

    def __init__(self, **options):
        if not hasattr(self, 'function'):
            raise TypeError(f'{self.__class__.__name__!r} used without bound function')
        Filter.__init__(self, **options)

    def filter(self, lexer, stream):
        # pylint: disable=not-callable
        yield from self.function(lexer, stream, self.options)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_lambdify.py ===
from itertools import product
import math
import inspect
import linecache
import gc

import mpmath
import cmath

from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.concrete.summations import Sum
from sympy.core.function import (Function, Lambda, diff)
from sympy.core.numbers import (E, Float, I, Rational, all_close, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.factorials import (RisingFactorial, factorial)
from sympy.functions.combinatorial.numbers import bernoulli, harmonic
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import asinh,acosh,atanh
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (asin, acos, atan, cos, cot, sin,
                                                      sinc, tan)
from sympy.functions import sinh,cosh,tanh
from sympy.functions.special.bessel import (besseli, besselj, besselk, bessely, jn, yn)
from sympy.functions.special.beta_functions import (beta, betainc, betainc_regularized)
from sympy.functions.special.delta_functions import (Heaviside)
from sympy.functions.special.error_functions import (Ei, erf, erfc, fresnelc, fresnels, Si, Ci)
from sympy.functions.special.gamma_functions import (digamma, gamma, loggamma, polygamma)
from sympy.functions.special.zeta_functions import zeta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, false, ITE, Not, Or, true)
from sympy.matrices.expressions.dotproduct import DotProduct
from sympy.simplify.cse_main import cse
from sympy.tensor.array import derive_by_array, Array
from sympy.tensor.array.expressions import ArraySymbol
from sympy.tensor.indexed import IndexedBase, Idx
from sympy.utilities.lambdify import lambdify
from sympy.utilities.iterables import numbered_symbols
from sympy.vector import CoordSys3D
from sympy.core.expr import UnevaluatedExpr
from sympy.codegen.cfunctions import expm1, log1p, exp2, log2, log10, hypot, isnan, isinf
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2, amin, amax, minimum, maximum
from sympy.codegen.scipy_nodes import cosm1, powm1
from sympy.functions.elementary.complexes import re, im, arg
from sympy.functions.special.polynomials import \
    chebyshevt, chebyshevu, legendre, hermite, laguerre, gegenbauer, \
    assoc_legendre, assoc_laguerre, jacobi
from sympy.matrices import Matrix, MatrixSymbol, SparseMatrix
from sympy.printing.codeprinter import PrintMethodNotImplementedError
from sympy.printing.lambdarepr import LambdaPrinter
from sympy.printing.numpy import NumPyPrinter
from sympy.utilities.lambdify import implemented_function, lambdastr
from sympy.testing.pytest import skip
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.utilities.exceptions import ignore_warnings
from sympy.external import import_module
from sympy.functions.special.gamma_functions import uppergamma, lowergamma


import sympy


MutableDenseMatrix = Matrix

numpy = import_module('numpy')
scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})
numexpr = import_module('numexpr')
tensorflow = import_module('tensorflow')
cupy = import_module('cupy')
jax = import_module('jax')
numba = import_module('numba')

if tensorflow:
    # Hide Tensorflow warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w, x, y, z = symbols('w,x,y,z')

#================== Test different arguments =======================


def test_no_args():
    f = lambdify([], 1)
    raises(TypeError, lambda: f(-1))
    assert f() == 1


def test_single_arg():
    f = lambdify(x, 2*x)
    assert f(1) == 2


def test_list_args():
    f = lambdify([x, y], x + y)
    assert f(1, 2) == 3


def test_nested_args():
    f1 = lambdify([[w, x]], [w, x])
    assert f1([91, 2]) == [91, 2]
    raises(TypeError, lambda: f1(1, 2))

    f2 = lambdify([(w, x), (y, z)], [w, x, y, z])
    assert f2((18, 12), (73, 4)) == [18, 12, 73, 4]
    raises(TypeError, lambda: f2(3, 4))

    f3 = lambdify([w, [[[x]], y], z], [w, x, y, z])
    assert f3(10, [[[52]], 31], 44) == [10, 52, 31, 44]


def test_str_args():
    f = lambdify('x,y,z', 'z,y,x')
    assert f(3, 2, 1) == (1, 2, 3)
    assert f(1.0, 2.0, 3.0) == (3.0, 2.0, 1.0)
    # make sure correct number of args required
    raises(TypeError, lambda: f(0))


def test_own_namespace_1():
    myfunc = lambda x: 1
    f = lambdify(x, sin(x), {"sin": myfunc})
    assert f(0.1) == 1
    assert f(100) == 1


def test_own_namespace_2():
    def myfunc(x):
        return 1
    f = lambdify(x, sin(x), {'sin': myfunc})
    assert f(0.1) == 1
    assert f(100) == 1


def test_own_module():
    f = lambdify(x, sin(x), math)
    assert f(0) == 0.0

    p, q, r = symbols("p q r", real=True)
    ae = abs(exp(p+UnevaluatedExpr(q+r)))
    f = lambdify([p, q, r], [ae, ae], modules=math)
    results = f(1.0, 1e18, -1e18)
    refvals = [math.exp(1.0)]*2
    for res, ref in zip(results, refvals):
        assert abs((res-ref)/ref) < 1e-15


def test_bad_args():
    # no vargs given
    raises(TypeError, lambda: lambdify(1))
    # same with vector exprs
    raises(TypeError, lambda: lambdify([1, 2]))


def test_atoms():
    # Non-Symbol atoms should not be pulled out from the expression namespace
    f = lambdify(x, pi + x, {"pi": 3.14})
    assert f(0) == 3.14
    f = lambdify(x, I + x, {"I": 1j})
    assert f(1) == 1 + 1j

#================== Test different modules =========================

# high precision output of sin(0.2*pi) is used to detect if precision is lost unwanted


@conserve_mpmath_dps
def test_sympy_lambda():
    mpmath.mp.dps = 50
    sin02 = mpmath.mpf("0.19866933079506121545941262711838975037020672954020")
    f = lambdify(x, sin(x), "sympy")
    assert f(x) == sin(x)
    prec = 1e-15
    assert -prec < f(Rational(1, 5)).evalf() - Float(str(sin02)) < prec
    # arctan is in numpy module and should not be available
    # The arctan below gives NameError. What is this supposed to test?
    # raises(NameError, lambda: lambdify(x, arctan(x), "sympy"))


@conserve_mpmath_dps
def test_math_lambda():
    mpmath.mp.dps = 50
    sin02 = mpmath.mpf("0.19866933079506121545941262711838975037020672954020")
    f = lambdify(x, sin(x), "math")
    prec = 1e-15
    assert -prec < f(0.2) - sin02 < prec
    raises(TypeError, lambda: f(x))
           # if this succeeds, it can't be a Python math function


@conserve_mpmath_dps
def test_mpmath_lambda():
    mpmath.mp.dps = 50
    sin02 = mpmath.mpf("0.19866933079506121545941262711838975037020672954020")
    f = lambdify(x, sin(x), "mpmath")
    prec = 1e-49  # mpmath precision is around 50 decimal places
    assert -prec < f(mpmath.mpf("0.2")) - sin02 < prec
    raises(TypeError, lambda: f(x))
           # if this succeeds, it can't be a mpmath function

    ref2 = (mpmath.mpf("1e-30")
            - mpmath.mpf("1e-45")/2
            + 5*mpmath.mpf("1e-60")/6
            - 3*mpmath.mpf("1e-75")/4
            + 33*mpmath.mpf("1e-90")/40
            )
    f2a = lambdify((x, y), x**y - 1, "mpmath")
    f2b = lambdify((x, y), powm1(x, y), "mpmath")
    f2c = lambdify((x,), expm1(x*log1p(x)), "mpmath")
    ans2a = f2a(mpmath.mpf("1")+mpmath.mpf("1e-15"), mpmath.mpf("1e-15"))
    ans2b = f2b(mpmath.mpf("1")+mpmath.mpf("1e-15"), mpmath.mpf("1e-15"))
    ans2c = f2c(mpmath.mpf("1e-15"))
    assert abs(ans2a - ref2) < 1e-51
    assert abs(ans2b - ref2) < 1e-67
    assert abs(ans2c - ref2) < 1e-80


@conserve_mpmath_dps
def test_number_precision():
    mpmath.mp.dps = 50
    sin02 = mpmath.mpf("0.19866933079506121545941262711838975037020672954020")
    f = lambdify(x, sin02, "mpmath")
    prec = 1e-49  # mpmath precision is around 50 decimal places
    assert -prec < f(0) - sin02 < prec

@conserve_mpmath_dps
def test_mpmath_precision():
    mpmath.mp.dps = 100
    assert str(lambdify((), pi.evalf(100), 'mpmath')()) == str(pi.evalf(100))

#================== Test Translations ==============================
# We can only check if all translated functions are valid. It has to be checked
# by hand if they are complete.


def test_math_transl():
    from sympy.utilities.lambdify import MATH_TRANSLATIONS
    for sym, mat in MATH_TRANSLATIONS.items():
        assert sym in sympy.__dict__
        assert mat in math.__dict__


def test_mpmath_transl():
    from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
    for sym, mat in MPMATH_TRANSLATIONS.items():
        assert sym in sympy.__dict__ or sym == 'Matrix'
        assert mat in mpmath.__dict__


def test_numpy_transl():
    if not numpy:
        skip("numpy not installed.")

    from sympy.utilities.lambdify import NUMPY_TRANSLATIONS
    for sym, nump in NUMPY_TRANSLATIONS.items():
        assert sym in sympy.__dict__
        assert nump in numpy.__dict__


def test_scipy_transl():
    if not scipy:
        skip("scipy not installed.")

    from sympy.utilities.lambdify import SCIPY_TRANSLATIONS
    for sym, scip in SCIPY_TRANSLATIONS.items():
        assert sym in sympy.__dict__
        assert scip in scipy.__dict__ or scip in scipy.special.__dict__


def test_numpy_translation_abs():
    if not numpy:
        skip("numpy not installed.")

    f = lambdify(x, Abs(x), "numpy")
    assert f(-1) == 1
    assert f(1) == 1


def test_numexpr_printer():
    if not numexpr:
        skip("numexpr not installed.")

    # if translation/printing is done incorrectly then evaluating
    # a lambdified numexpr expression will throw an exception
    from sympy.printing.lambdarepr import NumExprPrinter

    blacklist = ('where', 'complex', 'contains')
    arg_tuple = (x, y, z) # some functions take more than one argument
    for sym in NumExprPrinter._numexpr_functions.keys():
        if sym in blacklist:
            continue
        ssym = S(sym)
        if hasattr(ssym, '_nargs'):
            nargs = ssym._nargs[0]
        else:
            nargs = 1
        args = arg_tuple[:nargs]
        f = lambdify(args, ssym(*args), modules='numexpr')
        assert f(*(1, )*nargs) is not None


def test_cmath_sqrt():
    f = lambdify(x, sqrt(x), "cmath")
    assert f(0) == 0
    assert f(1) == 1
    assert f(4) == 2
    assert abs(f(2) - 1.414) < 0.001
    assert f(-1) == 1j
    assert f(-4) == 2j


def test_cmath_log():
    f = lambdify(x, log(x), "cmath")
    assert abs(f(1) - 0) < 1e-15
    assert abs(f(cmath.e) - 1) < 1e-15
    assert abs(f(-1) - cmath.log(-1)) < 1e-15


def test_cmath_sinh():
    f = lambdify(x, sinh(x), "cmath")
    assert abs(f(0) - cmath.sinh(0)) < 1e-15
    assert abs(f(pi) - cmath.sinh(pi)) < 1e-15
    assert abs(f(-pi) - cmath.sinh(-pi)) < 1e-15
    assert abs(f(1j) - cmath.sinh(1j)) < 1e-15


def test_cmath_cosh():
    f = lambdify(x, cosh(x), "cmath")
    assert abs(f(0) - cmath.cosh(0)) < 1e-15
    assert abs(f(pi) - cmath.cosh(pi)) < 1e-15
    assert abs(f(-pi) - cmath.cosh(-pi)) < 1e-15
    assert abs(f(1j) - cmath.cosh(1j)) < 1e-15


def test_cmath_tanh():
    f = lambdify(x, tanh(x), "cmath")
    assert abs(f(0) - cmath.tanh(0)) < 1e-15
    assert abs(f(pi) - cmath.tanh(pi)) < 1e-15
    assert abs(f(-pi) - cmath.tanh(-pi)) < 1e-15
    assert abs(f(1j) - cmath.tanh(1j)) < 1e-15


def test_cmath_sin():
    f = lambdify(x, sin(x), "cmath")
    assert abs(f(0) - cmath.sin(0)) < 1e-15
    assert abs(f(pi) - cmath.sin(pi)) < 1e-15
    assert abs(f(-pi) - cmath.sin(-pi)) < 1e-15
    assert abs(f(1j) - cmath.sin(1j)) < 1e-15


def test_cmath_cos():
    f = lambdify(x, cos(x), "cmath")
    assert abs(f(0) - cmath.cos(0)) < 1e-15
    assert abs(f(pi) - cmath.cos(pi)) < 1e-15
    assert abs(f(-pi) - cmath.cos(-pi)) < 1e-15
    assert abs(f(1j) - cmath.cos(1j)) < 1e-15


def test_cmath_tan():
    f = lambdify(x, tan(x), "cmath")
    assert abs(f(0) - cmath.tan(0)) < 1e-15
    assert abs(f(1j) - cmath.tan(1j)) < 1e-15


def test_cmath_asin():
    f = lambdify(x, asin(x), "cmath")
    assert abs(f(0) - cmath.asin(0)) < 1e-15
    assert abs(f(1) - cmath.asin(1)) < 1e-15
    assert abs(f(-1) - cmath.asin(-1)) < 1e-15
    assert abs(f(2) - cmath.asin(2)) < 1e-15
    assert abs(f(1j) - cmath.asin(1j)) < 1e-15


def test_cmath_acos():
    f = lambdify(x, acos(x), "cmath")
    assert abs(f(1) - cmath.acos(1)) < 1e-15
    assert abs(f(-1) - cmath.acos(-1)) < 1e-15
    assert abs(f(2) - cmath.acos(2)) < 1e-15
    assert abs(f(1j) - cmath.acos(1j)) < 1e-15


def test_cmath_atan():
    f = lambdify(x, atan(x), "cmath")
    assert abs(f(0) - cmath.atan(0)) < 1e-15
    assert abs(f(1) - cmath.atan(1)) < 1e-15
    assert abs(f(-1) - cmath.atan(-1)) < 1e-15
    assert abs(f(2) - cmath.atan(2)) < 1e-15
    assert abs(f(2j) - cmath.atan(2j)) < 1e-15


def test_cmath_asinh():
    f = lambdify(x, asinh(x), "cmath")
    assert abs(f(0) - cmath.asinh(0)) < 1e-15
    assert abs(f(1) - cmath.asinh(1)) < 1e-15
    assert abs(f(-1) - cmath.asinh(-1)) < 1e-15
    assert abs(f(2) - cmath.asinh(2)) < 1e-15
    assert abs(f(2j) - cmath.asinh(2j)) < 1e-15


def test_cmath_acosh():
    f = lambdify(x, acosh(x), "cmath")
    assert abs(f(1) - cmath.acosh(1)) < 1e-15
    assert abs(f(2) - cmath.acosh(2)) < 1e-15
    assert abs(f(-1) - cmath.acosh(-1)) < 1e-15
    assert abs(f(2j) - cmath.acosh(2j)) < 1e-15


def test_cmath_atanh():
    f = lambdify(x, atanh(x), "cmath")
    assert abs(f(0) - cmath.atanh(0)) < 1e-15
    assert abs(f(0.5) - cmath.atanh(0.5)) < 1e-15
    assert abs(f(-0.5) - cmath.atanh(-0.5)) < 1e-15
    assert abs(f(2) - cmath.atanh(2)) < 1e-15
    assert abs(f(-2) - cmath.atanh(-2)) < 1e-15
    assert abs(f(2j) - cmath.atanh(2j)) < 1e-15


def test_cmath_complex_identities():
    # Define symbol
    z = symbols('z')

    # Trigonometric identity using re(z) and im(z)
    expr = cos(z) - cos(re(z)) * cosh(im(z)) + I * sin(re(z)) * sinh(im(z))
    func = lambdify([z], expr, modules=["cmath", "math"])
    hpi = math.pi / 2
    assert abs(func(hpi + 1j * hpi)) < 4e-16

    # Euler's Formula: e^(i*z) = cos(z) + i*sin(z)
    func = lambdify([z], exp(I * z) - (cos(z) + I * sin(z)), modules=["cmath", "math"])
    assert abs(func(hpi)) < 4e-16

    # Exponential Identity: e^z = e^(Re(z)) * (cos(Im(z)) + i*sin(Im(z)))
    func_exp = lambdify([z], exp(z) - exp(re(z)) * (cos(im(z)) + I * sin(im(z))),
                        modules=["cmath", "math"])
    assert abs(func_exp(hpi + 1j * hpi)) < 4e-16

    # Complex Cosine Identity: cos(z) = cos(Re(z)) * cosh(Im(z)) - i*sin(Re(z)) * sinh(Im(z))
    func_cos = lambdify([z], cos(z) - (cos(re(z)) * cosh(im(z)) - I * sin(re(z)) * sinh(im(z))),
                        modules=["cmath", "math"])
    assert abs(func_cos(hpi + 1j * hpi)) < 4e-16

    # Complex Sine Identity: sin(z) = sin(Re(z)) * cosh(Im(z)) + i*cos(Re(z)) * sinh(Im(z))
    func_sin = lambdify([z], sin(z) - (sin(re(z)) * cosh(im(z)) + I * cos(re(z)) * sinh(im(z))),
                        modules=["cmath", "math"])
    assert abs(func_sin(hpi + 1j * hpi)) < 4e-16

    # Complex Hyperbolic Cosine Identity: cosh(z) = cosh(Re(z)) * cos(Im(z)) + i*sinh(Re(z)) * sin(Im(z))
    func_cosh_1 = lambdify([z], cosh(z) - (cosh(re(z)) * cos(im(z)) + I * sinh(re(z)) * sin(im(z))),
                         modules=["cmath", "math"])
    assert abs(func_cosh_1(hpi + 1j * hpi)) < 4e-16

    # Complex Hyperbolic Sine Identity: sinh(z) = sinh(Re(z)) * cos(Im(z)) + i*cosh(Re(z)) * sin(Im(z))
    func_sinh = lambdify([z], sinh(z) - (sinh(re(z)) * cos(im(z)) + I * cosh(re(z)) * sin(im(z))),
                         modules=["cmath", "math"])
    assert abs(func_sinh(hpi + 1j * hpi)) < 4e-16

    # cosh(z) = (e^z + e^(-z)) / 2
    func_cosh_2 = lambdify([z], cosh(z) - (exp(z) + exp(-z)) / 2, modules=["cmath", "math"])
    assert abs(func_cosh_2(hpi)) < 4e-16

    # Additional expressions testing log and exp with real and imaginary parts
    expr1 = log(re(z)) + log(im(z)) - log(re(z) * im(z))
    expr2 = exp(re(z)) * exp(im(z) * I) - exp(z)
    expr3 = log(exp(re(z))) - re(z)
    expr4 = exp(log(re(z))) - re(z)
    expr5 = log(exp(re(z) + im(z))) - (re(z) + im(z))
    expr6 = exp(log(re(z) + im(z))) - (re(z) + im(z))
    func1 = lambdify([z], expr1, modules=["cmath", "math"])
    func2 = lambdify([z], expr2, modules=["cmath", "math"])
    func3 = lambdify([z], expr3, modules=["cmath", "math"])
    func4 = lambdify([z], expr4, modules=["cmath", "math"])
    func5 = lambdify([z], expr5, modules=["cmath", "math"])
    func6 = lambdify([z], expr6, modules=["cmath", "math"])
    test_value = 3 + 4j
    assert abs(func1(test_value)) < 4e-16
    assert abs(func2(test_value)) < 4e-16
    assert abs(func3(test_value)) < 4e-16
    assert abs(func4(test_value)) < 4e-16
    assert abs(func5(test_value)) < 4e-16
    assert abs(func6(test_value)) < 4e-16


def test_issue_9334():
    if not numexpr:
        skip("numexpr not installed.")
    if not numpy:
        skip("numpy not installed.")
    expr = S('b*a - sqrt(a**2)')
    a, b = sorted(expr.free_symbols, key=lambda s: s.name)
    func_numexpr = lambdify((a,b), expr, modules=[numexpr], dummify=False)
    foo, bar = numpy.random.random((2, 4))
    func_numexpr(foo, bar)


def test_issue_12984():
    if not numexpr:
        skip("numexpr not installed.")
    func_numexpr = lambdify((x,y,z), Piecewise((y, x >= 0), (z, x > -1)), numexpr)
    with ignore_warnings(RuntimeWarning):
        assert func_numexpr(1, 24, 42) == 24
        assert str(func_numexpr(-1, 24, 42)) == 'nan'


def test_empty_modules():
    x, y = symbols('x y')
    expr = -(x % y)

    no_modules = lambdify([x, y], expr)
    empty_modules = lambdify([x, y], expr, modules=[])
    assert no_modules(3, 7) == empty_modules(3, 7)
    assert no_modules(3, 7) == -3


def test_exponentiation():
    f = lambdify(x, x**2)
    assert f(-1) == 1
    assert f(0) == 0
    assert f(1) == 1
    assert f(-2) == 4
    assert f(2) == 4
    assert f(2.5) == 6.25


def test_sqrt():
    f = lambdify(x, sqrt(x))
    assert f(0) == 0.0
    assert f(1) == 1.0
    assert f(4) == 2.0
    assert abs(f(2) - 1.414) < 0.001
    assert f(6.25) == 2.5


def test_trig():
    f = lambdify([x], [cos(x), sin(x)], 'math')
    d = f(pi)
    prec = 1e-11
    assert -prec < d[0] + 1 < prec
    assert -prec < d[1] < prec
    d = f(3.14159)
    prec = 1e-5
    assert -prec < d[0] + 1 < prec
    assert -prec < d[1] < prec


def test_integral():
    if numpy and not scipy:
        skip("scipy not installed.")
    f = Lambda(x, exp(-x**2))
    l = lambdify(y, Integral(f(x), (x, y, oo)))
    d = l(-oo)
    assert 1.77245385 < d < 1.772453851


def test_double_integral():
    if numpy and not scipy:
        skip("scipy not installed.")
    # example from http://mpmath.org/doc/current/calculus/integration.html
    i = Integral(1/(1 - x**2*y**2), (x, 0, 1), (y, 0, z))
    l = lambdify([z], i)
    d = l(1)
    assert 1.23370055 < d < 1.233700551

def test_spherical_bessel():
    if numpy and not scipy:
        skip("scipy not installed.")
    test_point = 4.2 #randomly selected
    x = symbols("x")
    jtest = jn(2, x)
    assert abs(lambdify(x,jtest)(test_point) -
            jtest.subs(x,test_point).evalf()) < 1e-8
    ytest = yn(2, x)
    assert abs(lambdify(x,ytest)(test_point) -
            ytest.subs(x,test_point).evalf()) < 1e-8


#================== Test vectors ===================================


def test_vector_simple():
    f = lambdify((x, y, z), (z, y, x))
    assert f(3, 2, 1) == (1, 2, 3)
    assert f(1.0, 2.0, 3.0) == (3.0, 2.0, 1.0)
    # make sure correct number of args required
    raises(TypeError, lambda: f(0))


def test_vector_discontinuous():
    f = lambdify(x, (-1/x, 1/x))
    raises(ZeroDivisionError, lambda: f(0))
    assert f(1) == (-1.0, 1.0)
    assert f(2) == (-0.5, 0.5)
    assert f(-2) == (0.5, -0.5)


def test_trig_symbolic():
    f = lambdify([x], [cos(x), sin(x)], 'math')
    d = f(pi)
    assert abs(d[0] + 1) < 0.0001
    assert abs(d[1] - 0) < 0.0001


def test_trig_float():
    f = lambdify([x], [cos(x), sin(x)])
    d = f(3.14159)
    assert abs(d[0] + 1) < 0.0001
    assert abs(d[1] - 0) < 0.0001


def test_docs():
    f = lambdify(x, x**2)
    assert f(2) == 4
    f = lambdify([x, y, z], [z, y, x])
    assert f(1, 2, 3) == [3, 2, 1]
    f = lambdify(x, sqrt(x))
    assert f(4) == 2.0
    f = lambdify((x, y), sin(x*y)**2)
    assert f(0, 5) == 0


def test_math():
    f = lambdify((x, y), sin(x), modules="math")
    assert f(0, 5) == 0


def test_sin():
    f = lambdify(x, sin(x)**2)
    assert isinstance(f(2), float)
    f = lambdify(x, sin(x)**2, modules="math")
    assert isinstance(f(2), float)


def test_matrix():
    A = Matrix([[x, x*y], [sin(z) + 4, x**z]])
    sol = Matrix([[1, 2], [sin(3) + 4, 1]])
    f = lambdify((x, y, z), A, modules="sympy")
    assert f(1, 2, 3) == sol
    f = lambdify((x, y, z), (A, [A]), modules="sympy")
    assert f(1, 2, 3) == (sol, [sol])
    J = Matrix((x, x + y)).jacobian((x, y))
    v = Matrix((x, y))
    sol = Matrix([[1, 0], [1, 1]])
    assert lambdify(v, J, modules='sympy')(1, 2) == sol
    assert lambdify(v.T, J, modules='sympy')(1, 2) == sol


def test_numpy_matrix():
    if not numpy:
        skip("numpy not installed.")
    A = Matrix([[x, x*y], [sin(z) + 4, x**z]])
    sol_arr = numpy.array([[1, 2], [numpy.sin(3) + 4, 1]])
    #Lambdify array first, to ensure return to array as default
    f = lambdify((x, y, z), A, ['numpy'])
    numpy.testing.assert_allclose(f(1, 2, 3), sol_arr)
    #Check that the types are arrays and matrices
    assert isinstance(f(1, 2, 3), numpy.ndarray)

    # gh-15071
    class dot(Function):
        pass
    x_dot_mtx = dot(x, Matrix([[2], [1], [0]]))
    f_dot1 = lambdify(x, x_dot_mtx)
    inp = numpy.zeros((17, 3))
    assert numpy.all(f_dot1(inp) == 0)

    strict_kw = {"allow_unknown_functions": False, "inline": True, "fully_qualified_modules": False}
    p2 = NumPyPrinter(dict(user_functions={'dot': 'dot'}, **strict_kw))
    f_dot2 = lambdify(x, x_dot_mtx, printer=p2)
    assert numpy.all(f_dot2(inp) == 0)

    p3 = NumPyPrinter(strict_kw)
    # The line below should probably fail upon construction (before calling with "(inp)"):
    raises(Exception, lambda: lambdify(x, x_dot_mtx, printer=p3)(inp))


def test_numpy_transpose():
    if not numpy:
        skip("numpy not installed.")
    A = Matrix([[1, x], [0, 1]])
    f = lambdify((x), A.T, modules="numpy")
    numpy.testing.assert_array_equal(f(2), numpy.array([[1, 0], [2, 1]]))


def test_numpy_dotproduct():
    if not numpy:
        skip("numpy not installed")
    A = Matrix([x, y, z])
    f1 = lambdify([x, y, z], DotProduct(A, A), modules='numpy')
    f2 = lambdify([x, y, z], DotProduct(A, A.T), modules='numpy')
    f3 = lambdify([x, y, z], DotProduct(A.T, A), modules='numpy')
    f4 = lambdify([x, y, z], DotProduct(A, A.T), modules='numpy')

    assert f1(1, 2, 3) == \
           f2(1, 2, 3) == \
           f3(1, 2, 3) == \
           f4(1, 2, 3) == \
           numpy.array([14])


def test_numpy_inverse():
    if not numpy:
        skip("numpy not installed.")
    A = Matrix([[1, x], [0, 1]])
    f = lambdify((x), A**-1, modules="numpy")
    numpy.testing.assert_array_equal(f(2), numpy.array([[1, -2], [0,  1]]))


def test_numpy_old_matrix():
    if not numpy:
        skip("numpy not installed.")
    A = Matrix([[x, x*y], [sin(z) + 4, x**z]])
    sol_arr = numpy.array([[1, 2], [numpy.sin(3) + 4, 1]])
    f = lambdify((x, y, z), A, [{'ImmutableDenseMatrix': numpy.matrix}, 'numpy'])
    with ignore_warnings(PendingDeprecationWarning):
        numpy.testing.assert_allclose(f(1, 2, 3), sol_arr)
        assert isinstance(f(1, 2, 3), numpy.matrix)


def test_scipy_sparse_matrix():
    if not scipy:
        skip("scipy not installed.")
    A = SparseMatrix([[x, 0], [0, y]])
    f = lambdify((x, y), A, modules="scipy")
    B = f(1, 2)
    assert isinstance(B, scipy.sparse.coo_matrix)


def test_python_div_zero_issue_11306():
    if not numpy:
        skip("numpy not installed.")
    p = Piecewise((1 / x, y < -1), (x, y < 1), (1 / x, True))
    f = lambdify([x, y], p, modules='numpy')
    with numpy.errstate(divide='ignore'):
        assert float(f(numpy.array(0), numpy.array(0.5))) == 0
        assert float(f(numpy.array(0), numpy.array(1))) == float('inf')


def test_issue9474():
    mods = [None, 'math']
    if numpy:
        mods.append('numpy')
    if mpmath:
        mods.append('mpmath')
    for mod in mods:
        f = lambdify(x, S.One/x, modules=mod)
        assert f(2) == 0.5
        f = lambdify(x, floor(S.One/x), modules=mod)
        assert f(2) == 0

    for absfunc, modules in product([Abs, abs], mods):
        f = lambdify(x, absfunc(x), modules=modules)
        assert f(-1) == 1
        assert f(1) == 1
        assert f(3+4j) == 5


def test_issue_9871():
    if not numexpr:
        skip("numexpr not installed.")
    if not numpy:
        skip("numpy not installed.")

    r = sqrt(x**2 + y**2)
    expr = diff(1/r, x)

    xn = yn = numpy.linspace(1, 10, 16)
    # expr(xn, xn) = -xn/(sqrt(2)*xn)^3
    fv_exact = -numpy.sqrt(2.)**-3 * xn**-2

    fv_numpy = lambdify((x, y), expr, modules='numpy')(xn, yn)
    fv_numexpr = lambdify((x, y), expr, modules='numexpr')(xn, yn)
    numpy.testing.assert_allclose(fv_numpy, fv_exact, rtol=1e-10)
    numpy.testing.assert_allclose(fv_numexpr, fv_exact, rtol=1e-10)


def test_numpy_piecewise():
    if not numpy:
        skip("numpy not installed.")
    pieces = Piecewise((x, x < 3), (x**2, x > 5), (0, True))
    f = lambdify(x, pieces, modules="numpy")
    numpy.testing.assert_array_equal(f(numpy.arange(10)),
                                     numpy.array([0, 1, 2, 0, 0, 0, 36, 49, 64, 81]))
    # If we evaluate somewhere all conditions are False, we should get back NaN
    nodef_func = lambdify(x, Piecewise((x, x > 0), (-x, x < 0)))
    numpy.testing.assert_array_equal(nodef_func(numpy.array([-1, 0, 1])),
                                     numpy.array([1, numpy.nan, 1]))


def test_numpy_logical_ops():
    if not numpy:
        skip("numpy not installed.")
    and_func = lambdify((x, y), And(x, y), modules="numpy")
    and_func_3 = lambdify((x, y, z), And(x, y, z), modules="numpy")
    or_func = lambdify((x, y), Or(x, y), modules="numpy")
    or_func_3 = lambdify((x, y, z), Or(x, y, z), modules="numpy")
    not_func = lambdify((x), Not(x), modules="numpy")
    arr1 = numpy.array([True, True])
    arr2 = numpy.array([False, True])
    arr3 = numpy.array([True, False])
    numpy.testing.assert_array_equal(and_func(arr1, arr2), numpy.array([False, True]))
    numpy.testing.assert_array_equal(and_func_3(arr1, arr2, arr3), numpy.array([False, False]))
    numpy.testing.assert_array_equal(or_func(arr1, arr2), numpy.array([True, True]))
    numpy.testing.assert_array_equal(or_func_3(arr1, arr2, arr3), numpy.array([True, True]))
    numpy.testing.assert_array_equal(not_func(arr2), numpy.array([True, False]))


def test_numpy_matmul():
    if not numpy:
        skip("numpy not installed.")
    xmat = Matrix([[x, y], [z, 1+z]])
    ymat = Matrix([[x**2], [Abs(x)]])
    mat_func = lambdify((x, y, z), xmat*ymat, modules="numpy")
    numpy.testing.assert_array_equal(mat_func(0.5, 3, 4), numpy.array([[1.625], [3.5]]))
    numpy.testing.assert_array_equal(mat_func(-0.5, 3, 4), numpy.array([[1.375], [3.5]]))
    # Multiple matrices chained together in multiplication
    f = lambdify((x, y, z), xmat*xmat*xmat, modules="numpy")
    numpy.testing.assert_array_equal(f(0.5, 3, 4), numpy.array([[72.125, 119.25],
                                                                [159, 251]]))


def test_numpy_numexpr():
    if not numpy:
        skip("numpy not installed.")
    if not numexpr:
        skip("numexpr not installed.")
    a, b, c = numpy.random.randn(3, 128, 128)
    # ensure that numpy and numexpr return same value for complicated expression
    expr = sin(x) + cos(y) + tan(z)**2 + Abs(z-y)*acos(sin(y*z)) + \
           Abs(y-z)*acosh(2+exp(y-x))- sqrt(x**2+I*y**2)
    npfunc = lambdify((x, y, z), expr, modules='numpy')
    nefunc = lambdify((x, y, z), expr, modules='numexpr')
    assert numpy.allclose(npfunc(a, b, c), nefunc(a, b, c))


def test_numexpr_userfunctions():
    if not numpy:
        skip("numpy not installed.")
    if not numexpr:
        skip("numexpr not installed.")
    a, b = numpy.random.randn(2, 10)
    uf = type('uf', (Function, ),
              {'eval' : classmethod(lambda x, y : y**2+1)})
    func = lambdify(x, 1-uf(x), modules='numexpr')
    assert numpy.allclose(func(a), -(a**2))

    uf = implemented_function(Function('uf'), lambda x, y : 2*x*y+1)
    func = lambdify((x, y), uf(x, y), modules='numexpr')
    assert numpy.allclose(func(a, b), 2*a*b+1)


def test_tensorflow_basic_math():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = Max(sin(x), Abs(1/(x+2)))
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        a = tensorflow.constant(0, dtype=tensorflow.float32)
        assert func(a).eval(session=s) == 0.5


def test_tensorflow_placeholders():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = Max(sin(x), Abs(1/(x+2)))
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        a = tensorflow.compat.v1.placeholder(dtype=tensorflow.float32)
        assert func(a).eval(session=s, feed_dict={a: 0}) == 0.5


def test_tensorflow_variables():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = Max(sin(x), Abs(1/(x+2)))
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        a = tensorflow.Variable(0, dtype=tensorflow.float32)
        s.run(a.initializer)
        assert func(a).eval(session=s, feed_dict={a: 0}) == 0.5


def test_tensorflow_logical_operations():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = Not(And(Or(x, y), y))
    func = lambdify([x, y], expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        assert func(False, True).eval(session=s) == False


def test_tensorflow_piecewise():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = Piecewise((0, Eq(x,0)), (-1, x < 0), (1, x > 0))
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        assert func(-1).eval(session=s) == -1
        assert func(0).eval(session=s) == 0
        assert func(1).eval(session=s) == 1


def test_tensorflow_multi_max():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = Max(x, -x, x**2)
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        assert func(-2).eval(session=s) == 4


def test_tensorflow_multi_min():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = Min(x, -x, x**2)
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        assert func(-2).eval(session=s) == -2


def test_tensorflow_relational():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = x >= 0
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        assert func(1).eval(session=s) == True


def test_tensorflow_complexes():
    if not tensorflow:
        skip("tensorflow not installed")

    func1 = lambdify(x, re(x), modules="tensorflow")
    func2 = lambdify(x, im(x), modules="tensorflow")
    func3 = lambdify(x, Abs(x), modules="tensorflow")
    func4 = lambdify(x, arg(x), modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        # For versions before
        # https://github.com/tensorflow/tensorflow/issues/30029
        # resolved, using Python numeric types may not work
        a = tensorflow.constant(1+2j)
        assert func1(a).eval(session=s) == 1
        assert func2(a).eval(session=s) == 2

        tensorflow_result = func3(a).eval(session=s)
        sympy_result = Abs(1 + 2j).evalf()
        assert abs(tensorflow_result-sympy_result) < 10**-6

        tensorflow_result = func4(a).eval(session=s)
        sympy_result = arg(1 + 2j).evalf()
        assert abs(tensorflow_result-sympy_result) < 10**-6


def test_tensorflow_array_arg():
    # Test for issue 14655 (tensorflow part)
    if not tensorflow:
        skip("tensorflow not installed.")

    f = lambdify([[x, y]], x*x + y, 'tensorflow')

    with tensorflow.compat.v1.Session() as s:
        fcall = f(tensorflow.constant([2.0, 1.0]))
        assert fcall.eval(session=s) == 5.0


#================== Test symbolic ==================================


def test_sym_single_arg():
    f = lambdify(x, x * y)
    assert f(z) == z * y


def test_sym_list_args():
    f = lambdify([x, y], x + y + z)
    assert f(1, 2) == 3 + z


def test_sym_integral():
    f = Lambda(x, exp(-x**2))
    l = lambdify(x, Integral(f(x), (x, -oo, oo)), modules="sympy")
    assert l(y) == Integral(exp(-y**2), (y, -oo, oo))
    assert l(y).doit() == sqrt(pi)


def test_namespace_order():
    # lambdify had a bug, such that module dictionaries or cached module
    # dictionaries would pull earlier namespaces into themselves.
    # Because the module dictionaries form the namespace of the
    # generated lambda, this meant that the behavior of a previously
    # generated lambda function could change as a result of later calls
    # to lambdify.
    n1 = {'f': lambda x: 'first f'}
    n2 = {'f': lambda x: 'second f',
          'g': lambda x: 'function g'}
    f = sympy.Function('f')
    g = sympy.Function('g')
    if1 = lambdify(x, f(x), modules=(n1, "sympy"))
    assert if1(1) == 'first f'
    if2 = lambdify(x, g(x), modules=(n2, "sympy"))
    # previously gave 'second f'
    assert if1(1) == 'first f'

    assert if2(1) == 'function g'


def test_imps():
    # Here we check if the default returned functions are anonymous - in
    # the sense that we can have more than one function with the same name
    f = implemented_function('f', lambda x: 2*x)
    g = implemented_function('f', lambda x: math.sqrt(x))
    l1 = lambdify(x, f(x))
    l2 = lambdify(x, g(x))
    assert str(f(x)) == str(g(x))
    assert l1(3) == 6
    assert l2(3) == math.sqrt(3)
    # check that we can pass in a Function as input
    func = sympy.Function('myfunc')
    assert not hasattr(func, '_imp_')
    my_f = implemented_function(func, lambda x: 2*x)
    assert hasattr(my_f, '_imp_')
    # Error for functions with same name and different implementation
    f2 = implemented_function("f", lambda x: x + 101)
    raises(ValueError, lambda: lambdify(x, f(f2(x))))


def test_imps_errors():
    # Test errors that implemented functions can return, and still be able to
    # form expressions.
    # See: https://github.com/sympy/sympy/issues/10810
    #
    # XXX: Removed AttributeError here. This test was added due to issue 10810
    # but that issue was about ValueError. It doesn't seem reasonable to
    # "support" catching AttributeError in the same context...
    for val, error_class in product((0, 0., 2, 2.0), (TypeError, ValueError)):

        def myfunc(a):
            if a == 0:
                raise error_class
            return 1

        f = implemented_function('f', myfunc)
        expr = f(val)
        assert expr == f(val)


def test_imps_wrong_args():
    raises(ValueError, lambda: implemented_function(sin, lambda x: x))


def test_lambdify_imps():
    # Test lambdify with implemented functions
    # first test basic (sympy) lambdify
    f = sympy.cos
    assert lambdify(x, f(x))(0) == 1
    assert lambdify(x, 1 + f(x))(0) == 2
    assert lambdify((x, y), y + f(x))(0, 1) == 2
    # make an implemented function and test
    f = implemented_function("f", lambda x: x + 100)
    assert lambdify(x, f(x))(0) == 100
    assert lambdify(x, 1 + f(x))(0) == 101
    assert lambdify((x, y), y + f(x))(0, 1) == 101
    # Can also handle tuples, lists, dicts as expressions
    lam = lambdify(x, (f(x), x))
    assert lam(3) == (103, 3)
    lam = lambdify(x, [f(x), x])
    assert lam(3) == [103, 3]
    lam = lambdify(x, [f(x), (f(x), x)])
    assert lam(3) == [103, (103, 3)]
    lam = lambdify(x, {f(x): x})
    assert lam(3) == {103: 3}
    lam = lambdify(x, {f(x): x})
    assert lam(3) == {103: 3}
    lam = lambdify(x, {x: f(x)})
    assert lam(3) == {3: 103}
    # Check that imp preferred to other namespaces by default
    d = {'f': lambda x: x + 99}
    lam = lambdify(x, f(x), d)
    assert lam(3) == 103
    # Unless flag passed
    lam = lambdify(x, f(x), d, use_imps=False)
    assert lam(3) == 102


def test_dummification():
    t = symbols('t')
    F = Function('F')
    G = Function('G')
    #"\alpha" is not a valid Python variable name
    #lambdify should sub in a dummy for it, and return
    #without a syntax error
    alpha = symbols(r'\alpha')
    some_expr = 2 * F(t)**2 / G(t)
    lam = lambdify((F(t), G(t)), some_expr)
    assert lam(3, 9) == 2
    lam = lambdify(sin(t), 2 * sin(t)**2)
    assert lam(F(t)) == 2 * F(t)**2
    #Test that \alpha was properly dummified
    lam = lambdify((alpha, t), 2*alpha + t)
    assert lam(2, 1) == 5
    raises(SyntaxError, lambda: lambdify(F(t) * G(t), F(t) * G(t) + 5))
    raises(SyntaxError, lambda: lambdify(2 * F(t), 2 * F(t) + 5))
    raises(SyntaxError, lambda: lambdify(2 * F(t), 4 * F(t) + 5))


def test_lambdify__arguments_with_invalid_python_identifiers():
    # see sympy/sympy#26690
    N = CoordSys3D('N')
    xn, yn, zn = N.base_scalars()
    expr = xn + yn
    f = lambdify([xn, yn], expr)
    res = f(0.2, 0.3)
    ref = 0.2 + 0.3
    assert abs(res-ref) < 1e-15


def test_curly_matrix_symbol():
    # Issue #15009
    curlyv = sympy.MatrixSymbol("{v}", 2, 1)
    lam = lambdify(curlyv, curlyv)
    assert lam(1)==1
    lam = lambdify(curlyv, curlyv, dummify=True)
    assert lam(1)==1


def test_python_keywords():
    # Test for issue 7452. The automatic dummification should ensure use of
    # Python reserved keywords as symbol names will create valid lambda
    # functions. This is an additional regression test.
    python_if = symbols('if')
    expr = python_if / 2
    f = lambdify(python_if, expr)
    assert f(4.0) == 2.0


def test_lambdify_docstring():
    func = lambdify((w, x, y, z), w + x + y + z)
    ref = (
        "Created with lambdify. Signature:\n\n"
        "func(w, x, y, z)\n\n"
        "Expression:\n\n"
        "w + x + y + z"
    ).splitlines()
    assert func.__doc__.splitlines()[:len(ref)] == ref
    syms = symbols('a1:26')
    func = lambdify(syms, sum(syms))
    ref = (
        "Created with lambdify. Signature:\n\n"
        "func(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,\n"
        "        a16, a17, a18, a19, a20, a21, a22, a23, a24, a25)\n\n"
        "Expression:\n\n"
        "a1 + a10 + a11 + a12 + a13 + a14 + a15 + a16 + a17 + a18 + a19 + a2 + a20 +..."
    ).splitlines()
    assert func.__doc__.splitlines()[:len(ref)] == ref


def test_lambdify_linecache():
    func = lambdify(x, x + 1)
    source = 'def _lambdifygenerated(x):\n    return x + 1\n'
    assert inspect.getsource(func) == source
    filename = inspect.getsourcefile(func)
    assert filename.startswith('<lambdifygenerated-')
    assert filename in linecache.cache
    assert linecache.cache[filename] == (len(source), None, source.splitlines(True), filename)
    del func
    gc.collect()
    assert filename not in linecache.cache

#================== Test special printers ==========================


def test_special_printers():
    from sympy.printing.lambdarepr import IntervalPrinter

    def intervalrepr(expr):
        return IntervalPrinter().doprint(expr)

    expr = sqrt(sqrt(2) + sqrt(3)) + S.Half

    func0 = lambdify((), expr, modules="mpmath", printer=intervalrepr)
    func1 = lambdify((), expr, modules="mpmath", printer=IntervalPrinter)
    func2 = lambdify((), expr, modules="mpmath", printer=IntervalPrinter())

    mpi = type(mpmath.mpi(1, 2))

    assert isinstance(func0(), mpi)
    assert isinstance(func1(), mpi)
    assert isinstance(func2(), mpi)

    # To check Is lambdify loggamma works for mpmath or not
    exp1 = lambdify(x, loggamma(x), 'mpmath')(5)
    exp2 = lambdify(x, loggamma(x), 'mpmath')(1.8)
    exp3 = lambdify(x, loggamma(x), 'mpmath')(15)
    exp_ls = [exp1, exp2, exp3]

    sol1 = mpmath.loggamma(5)
    sol2 = mpmath.loggamma(1.8)
    sol3 = mpmath.loggamma(15)
    sol_ls = [sol1, sol2, sol3]

    assert exp_ls == sol_ls


def test_true_false():
    # We want exact is comparison here, not just ==
    assert lambdify([], true)() is True
    assert lambdify([], false)() is False


def test_issue_2790():
    assert lambdify((x, (y, z)), x + y)(1, (2, 4)) == 3
    assert lambdify((x, (y, (w, z))), w + x + y + z)(1, (2, (3, 4))) == 10
    assert lambdify(x, x + 1, dummify=False)(1) == 2


def test_issue_12092():
    f = implemented_function('f', lambda x: x**2)
    assert f(f(2)).evalf() == Float(16)


def test_issue_14911():
    class Variable(sympy.Symbol):
        def _sympystr(self, printer):
            return printer.doprint(self.name)

        _lambdacode = _sympystr
        _numpycode = _sympystr

    x = Variable('x')
    y = 2 * x
    code = LambdaPrinter().doprint(y)
    assert code.replace(' ', '') == '2*x'


def test_ITE():
    assert lambdify((x, y, z), ITE(x, y, z))(True, 5, 3) == 5
    assert lambdify((x, y, z), ITE(x, y, z))(False, 5, 3) == 3


def test_Min_Max():
    # see gh-10375
    assert lambdify((x, y, z), Min(x, y, z))(1, 2, 3) == 1
    assert lambdify((x, y, z), Max(x, y, z))(1, 2, 3) == 3


def test_amin_amax_minimum_maximum():
    if not numpy:
        skip("numpy not installed")

    a234 = numpy.array([2, 3, 4])
    a152 = numpy.array([1, 5, 2])

    a254 = numpy.array([2, 5, 4])
    a132 = numpy.array([1, 3, 2])
    # 2 args
    assert numpy.all(lambdify((x, y), maximum(x, y))(a234, a152) == a254)
    assert numpy.all(lambdify((x, y), minimum(x, y))(a234, a152) == a132)

    # 3 args
    assert numpy.all(lambdify((x, y, z), maximum(x, y, z))(a234, a152, a234) == a254)
    assert numpy.all(lambdify((x, y, z), minimum(x, y, z))(a234, a152, a234) == a132)

    # 1 arg
    assert numpy.all(lambdify((x,), maximum(x))(a234) == a234)
    assert numpy.all(lambdify((x,), minimum(x))(a234) == a234)

    # 4 args, mixed length
    assert numpy.all(lambdify((x, y, z, w), maximum(x, y, z, w))(a234, a152, a234, 3) == [3, 5, 4])
    assert numpy.all(lambdify((x, y, z, w), minimum(x, y, z, w))(a234, a152, a234, 2) == [1, 2, 2])

    # amin & amax
    assert lambdify((x, y), [amin(x), amax(y)])(a234, a152) == [2, 5]
    A = numpy.array([
        [0, 4, 8],
        [1, 5, 9],
        [2, 6, 10],
    ])
    min_, max_ = lambdify((x,), [amin(x, axis=0), amax(x, axis=1)])(A)
    assert numpy.all(min_ == numpy.amin(A, axis=0))
    assert numpy.all(max_ == numpy.amax(A, axis=1))

    # see gh-25659
    assert numpy.all(lambdify((x, y), Max(x, y))([1, 2, 3], [3, 2, 1]) == [3, 2, 3])
    assert numpy.all(lambdify((x), Min(2, x))([1, 2, 3]) == [1, 2, 2])



def test_Indexed():
    # Issue #10934
    if not numpy:
        skip("numpy not installed")

    a = IndexedBase('a')
    i, j = symbols('i j')
    b = numpy.array([[1, 2], [3, 4]])
    assert lambdify(a, Sum(a[x, y], (x, 0, 1), (y, 0, 1)))(b) == 10

def test_Sum():
    e = Sum(z, (y, 0, x), (x, 0, 10))
    ref = 66*z
    assert e.doit() == ref
    assert lambdify([z], e)(7) == ref.subs(z, 7)

def test_Idx():
    # Issue 26888
    a = IndexedBase('a')
    i = Idx('i')
    b = [1,2,3]
    assert lambdify([a, i], a[i])(b, 2) == 3


def test_issue_12173():
    #test for issue 12173
    expr1 = lambdify((x, y), uppergamma(x, y),"mpmath")(1, 2)
    expr2 = lambdify((x, y), lowergamma(x, y),"mpmath")(1, 2)
    assert expr1 == uppergamma(1, 2).evalf()
    assert expr2 == lowergamma(1, 2).evalf()


def test_issue_13642():
    if not numpy:
        skip("numpy not installed")
    f = lambdify(x, sinc(x))
    assert Abs(f(1) - sinc(1)).n() < 1e-15


def test_sinc_mpmath():
    f = lambdify(x, sinc(x), "mpmath")
    assert Abs(f(1) - sinc(1)).n() < 1e-15


def test_lambdify_dummy_arg():
    d1 = Dummy()
    f1 = lambdify(d1, d1 + 1, dummify=False)
    assert f1(2) == 3
    f1b = lambdify(d1, d1 + 1)
    assert f1b(2) == 3
    d2 = Dummy('x')
    f2 = lambdify(d2, d2 + 1)
    assert f2(2) == 3
    f3 = lambdify([[d2]], d2 + 1)
    assert f3([2]) == 3


def test_lambdify_mixed_symbol_dummy_args():
    d = Dummy()
    # Contrived example of name clash
    dsym = symbols(str(d))
    f = lambdify([d, dsym], d - dsym)
    assert f(4, 1) == 3


def test_numpy_array_arg():
    # Test for issue 14655 (numpy part)
    if not numpy:
        skip("numpy not installed")

    f = lambdify([[x, y]], x*x + y, 'numpy')

    assert f(numpy.array([2.0, 1.0])) == 5


def test_scipy_fns():
    if not scipy:
        skip("scipy not installed")

    single_arg_sympy_fns = [Ei, erf, erfc, factorial, gamma, loggamma, digamma, Si, Ci]
    single_arg_scipy_fns = [scipy.special.expi, scipy.special.erf, scipy.special.erfc,
        scipy.special.factorial, scipy.special.gamma, scipy.special.gammaln,
                            scipy.special.psi, scipy.special.sici, scipy.special.sici]
    numpy.random.seed(0)
    for (sympy_fn, scipy_fn) in zip(single_arg_sympy_fns, single_arg_scipy_fns):
        f = lambdify(x, sympy_fn(x), modules="scipy")
        for i in range(20):
            tv = numpy.random.uniform(-10, 10) + 1j*numpy.random.uniform(-5, 5)
            # SciPy thinks that factorial(z) is 0 when re(z) < 0 and
            # does not support complex numbers.
            # SymPy does not think so.
            if sympy_fn == factorial:
                tv = numpy.abs(tv)
            # SciPy supports gammaln for real arguments only,
            # and there is also a branch cut along the negative real axis
            if sympy_fn == loggamma:
                tv = numpy.abs(tv)
            # SymPy's digamma evaluates as polygamma(0, z)
            # which SciPy supports for real arguments only
            if sympy_fn == digamma:
                tv = numpy.real(tv)
            sympy_result = sympy_fn(tv).evalf()
            scipy_result = scipy_fn(tv)
            # SciPy's sici returns a tuple with both Si and Ci present in it
            # which needs to be unpacked
            if sympy_fn == Si:
                scipy_result = scipy_fn(tv)[0]
            if sympy_fn == Ci:
                scipy_result = scipy_fn(tv)[1]
            assert abs(f(tv) - sympy_result) < 1e-13*(1 + abs(sympy_result))
            assert abs(f(tv) - scipy_result) < 1e-13*(1 + abs(sympy_result))

    double_arg_sympy_fns = [RisingFactorial, besselj, bessely, besseli,
                            besselk, polygamma]
    double_arg_scipy_fns = [scipy.special.poch, scipy.special.jv,
                            scipy.special.yv, scipy.special.iv, scipy.special.kv, scipy.special.polygamma]
    for (sympy_fn, scipy_fn) in zip(double_arg_sympy_fns, double_arg_scipy_fns):
        f = lambdify((x, y), sympy_fn(x, y), modules="scipy")
        for i in range(20):
            # SciPy supports only real orders of Bessel functions
            tv1 = numpy.random.uniform(-10, 10)
            tv2 = numpy.random.uniform(-10, 10) + 1j*numpy.random.uniform(-5, 5)
            # SciPy requires a real valued 2nd argument for: poch, polygamma
            if sympy_fn in (RisingFactorial, polygamma):
                tv2 = numpy.real(tv2)
            if sympy_fn == polygamma:
                tv1 = abs(int(tv1))  # first argument to polygamma must be a non-negative integer.
            sympy_result = sympy_fn(tv1, tv2).evalf()
            assert abs(f(tv1, tv2) - sympy_result) < 1e-13*(1 + abs(sympy_result))
            assert abs(f(tv1, tv2) - scipy_fn(tv1, tv2)) < 1e-13*(1 + abs(sympy_result))


def test_scipy_polys():
    if not scipy:
        skip("scipy not installed")
    numpy.random.seed(0)

    params = symbols('n k a b')
    # list polynomials with the number of parameters
    polys = [
        (chebyshevt, 1),
        (chebyshevu, 1),
        (legendre, 1),
        (hermite, 1),
        (laguerre, 1),
        (gegenbauer, 2),
        (assoc_legendre, 2),
        (assoc_laguerre, 2),
        (jacobi, 3)
    ]

    msg = \
        "The random test of the function {func} with the arguments " \
        "{args} had failed because the SymPy result {sympy_result} " \
        "and SciPy result {scipy_result} had failed to converge " \
        "within the tolerance {tol} " \
        "(Actual absolute difference : {diff})"

    for sympy_fn, num_params in polys:
        args = params[:num_params] + (x,)
        f = lambdify(args, sympy_fn(*args))
        for _ in range(10):
            tn = numpy.random.randint(3, 10)
            tparams = tuple(numpy.random.uniform(0, 5, size=num_params-1))
            tv = numpy.random.uniform(-10, 10) + 1j*numpy.random.uniform(-5, 5)
            # SciPy supports hermite for real arguments only
            if sympy_fn == hermite:
                tv = numpy.real(tv)
            # assoc_legendre needs x in (-1, 1) and integer param at most n
            if sympy_fn == assoc_legendre:
                tv = numpy.random.uniform(-1, 1)
                tparams = tuple(numpy.random.randint(1, tn, size=1))

            vals = (tn,) + tparams + (tv,)
            scipy_result = f(*vals)
            sympy_result = sympy_fn(*vals).evalf()
            atol = 1e-9*(1 + abs(sympy_result))
            diff = abs(scipy_result - sympy_result)
            try:
                assert diff < atol
            except TypeError:
                raise AssertionError(
                    msg.format(
                        func=repr(sympy_fn),
                        args=repr(vals),
                        sympy_result=repr(sympy_result),
                        scipy_result=repr(scipy_result),
                        diff=diff,
                        tol=atol)
                    )


def test_lambdify_inspect():
    f = lambdify(x, x**2)
    # Test that inspect.getsource works but don't hard-code implementation
    # details
    assert 'x**2' in inspect.getsource(f)


def test_issue_14941():
    x, y = Dummy(), Dummy()

    # test dict
    f1 = lambdify([x, y], {x: 3, y: 3}, 'sympy')
    assert f1(2, 3) == {2: 3, 3: 3}

    # test tuple
    f2 = lambdify([x, y], (y, x), 'sympy')
    assert f2(2, 3) == (3, 2)
    f2b = lambdify([], (1,))  # gh-23224
    assert f2b() == (1,)

    # test list
    f3 = lambdify([x, y], [y, x], 'sympy')
    assert f3(2, 3) == [3, 2]


def test_lambdify_Derivative_arg_issue_16468():
    f = Function('f')(x)
    fx = f.diff()
    assert lambdify((f, fx), f + fx)(10, 5) == 15
    assert eval(lambdastr((f, fx), f/fx))(10, 5) == 2
    raises(Exception, lambda:
        eval(lambdastr((f, fx), f/fx, dummify=False)))
    assert eval(lambdastr((f, fx), f/fx, dummify=True))(10, 5) == 2
    assert eval(lambdastr((fx, f), f/fx, dummify=True))(S(10), 5) == S.Half
    assert lambdify(fx, 1 + fx)(41) == 42
    assert eval(lambdastr(fx, 1 + fx, dummify=True))(41) == 42


def test_lambdify_Derivative_zeta():
    # This is related to gh-11802 (and to lesser extent gh-26663)
    expr1 = zeta(x).diff(x, evaluate=False)
    f1 = lambdify(x, expr1, modules=['mpmath'])
    ans1 = f1(2)
    ref1 = (zeta(2+1e-8).evalf()-zeta(2).evalf())/1e-8
    assert abs(ans1 - ref1)/abs(ref1) < 1e-7

    expr2 = zeta(x**2).diff(x)
    f2 = lambdify(x, expr2, modules=['mpmath'])
    ans2 = f2(2**0.5)
    ref2 = 2*2**0.5*ref1
    assert abs(ans2-ref2)/abs(ref2) < 1e-7


def test_lambdify_Derivative_custom_printer():
    func1 = Function('func1')
    func2 = Function('func2')

    class MyPrinter(NumPyPrinter):

        def _print_Derivative_func1(self, args, seq_orders):
            arg, = args
            order, = seq_orders
            return '42'

    expr1 = func1(x).diff(x)
    raises(PrintMethodNotImplementedError, lambda: lambdify([x], expr1))
    f1 = lambdify([x], expr1, printer=MyPrinter)
    assert f1(7) == 42

    expr2 = func2(x).diff(x)
    raises(PrintMethodNotImplementedError, lambda: lambdify([x], expr2, printer=MyPrinter))


def test_lambdify_derivative_and_functions_as_arguments():
    # see: https://github.com/sympy/sympy/issues/26663#issuecomment-2157179517
    t, a, b = symbols('t, a, b')
    f = Function('f')(t)
    args = f.diff(t, 2), f.diff(t), f, a, b
    expr1 = a*f.diff(t, 2) + b*f.diff(t) + a*b*f + a**2
    num_args = 2.0, 3.0, 4.0, 5.0, 6.0
    ref1 = 5*2 + 6*3 + 5*6*4 + 5**2

    expr2 = a*f.diff(t, 2) + b*f.diff(t) - a*b*f + b**2 - a**2
    ref2 = 5*2 + 6*3 - 5*6*4 + 6**2 - 5**2

    for dummify, _cse in product([False, None, True], [False, True]):
        func1 = lambdify(args, expr1, cse=_cse, dummify=dummify)
        res1 = func1(*num_args)
        assert abs(res1 - ref1) < 1e-12

        func12 = lambdify(args, [expr1, expr2], cse=_cse, dummify=dummify)
        res12 = func12(*num_args)
        assert len(res12) == 2
        assert abs(res12[0] - ref1) < 1e-12
        assert abs(res12[1] - ref2) < 1e-12


def test_imag_real():
    f_re = lambdify([z], sympy.re(z))
    val = 3+2j
    assert f_re(val) == val.real

    f_im = lambdify([z], sympy.im(z))  # see #15400
    assert f_im(val) == val.imag


def test_MatrixSymbol_issue_15578():
    if not numpy:
        skip("numpy not installed")
    A = MatrixSymbol('A', 2, 2)
    A0 = numpy.array([[1, 2], [3, 4]])
    f = lambdify(A, A**(-1))
    assert numpy.allclose(f(A0), numpy.array([[-2., 1.], [1.5, -0.5]]))
    g = lambdify(A, A**3)
    assert numpy.allclose(g(A0), numpy.array([[37, 54], [81, 118]]))


def test_issue_15654():
    if not scipy:
        skip("scipy not installed")
    from sympy.abc import n, l, r, Z
    from sympy.physics import hydrogen
    nv, lv, rv, Zv = 1, 0, 3, 1
    sympy_value = hydrogen.R_nl(nv, lv, rv, Zv).evalf()
    f = lambdify((n, l, r, Z), hydrogen.R_nl(n, l, r, Z))
    scipy_value = f(nv, lv, rv, Zv)
    assert abs(sympy_value - scipy_value) < 1e-15


def test_issue_15827():
    if not numpy:
        skip("numpy not installed")
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 2, 3)
    C = MatrixSymbol("C", 3, 4)
    D = MatrixSymbol("D", 4, 5)
    k=symbols("k")
    f = lambdify(A, (2*k)*A)
    g = lambdify(A, (2+k)*A)
    h = lambdify(A, 2*A)
    i = lambdify((B, C, D), 2*B*C*D)
    assert numpy.array_equal(f(numpy.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])), \
    numpy.array([[2*k, 4*k, 6*k], [2*k, 4*k, 6*k], [2*k, 4*k, 6*k]], dtype=object))

    assert numpy.array_equal(g(numpy.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])), \
    numpy.array([[k + 2, 2*k + 4, 3*k + 6], [k + 2, 2*k + 4, 3*k + 6], \
    [k + 2, 2*k + 4, 3*k + 6]], dtype=object))

    assert numpy.array_equal(h(numpy.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])), \
    numpy.array([[2, 4, 6], [2, 4, 6], [2, 4, 6]]))

    assert numpy.array_equal(i(numpy.array([[1, 2, 3], [1, 2, 3]]), numpy.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]), \
    numpy.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])), numpy.array([[ 120, 240, 360, 480, 600], \
    [ 120, 240, 360, 480, 600]]))


def test_issue_16930():
    if not scipy:
        skip("scipy not installed")

    x = symbols("x")
    f = lambda x:  S.GoldenRatio * x**2
    f_ = lambdify(x, f(x), modules='scipy')
    assert f_(1) == scipy.constants.golden_ratio

def test_issue_17898():
    if not scipy:
        skip("scipy not installed")
    x = symbols("x")
    f_ = lambdify([x], sympy.LambertW(x,-1), modules='scipy')
    assert f_(0.1) == mpmath.lambertw(0.1, -1)

def test_issue_13167_21411():
    if not numpy:
        skip("numpy not installed")
    f1 = lambdify(x, sympy.Heaviside(x))
    f2 = lambdify(x, sympy.Heaviside(x, 1))
    res1 = f1([-1, 0, 1])
    res2 = f2([-1, 0, 1])
    assert Abs(res1[0]).n() < 1e-15        # First functionality: only one argument passed
    assert Abs(res1[1] - 1/2).n() < 1e-15
    assert Abs(res1[2] - 1).n() < 1e-15
    assert Abs(res2[0]).n() < 1e-15        # Second functionality: two arguments passed
    assert Abs(res2[1] - 1).n() < 1e-15
    assert Abs(res2[2] - 1).n() < 1e-15

def test_single_e():
    f = lambdify(x, E)
    assert f(23) == exp(1.0)

def test_issue_16536():
    if not scipy:
        skip("scipy not installed")

    a = symbols('a')
    f1 = lowergamma(a, x)
    F = lambdify((a, x), f1, modules='scipy')
    assert abs(lowergamma(1, 3) - F(1, 3)) <= 1e-10

    f2 = uppergamma(a, x)
    F = lambdify((a, x), f2, modules='scipy')
    assert abs(uppergamma(1, 3) - F(1, 3)) <= 1e-10


def test_issue_22726():
    if not numpy:
        skip("numpy not installed")

    x1, x2 = symbols('x1 x2')
    f = Max(S.Zero, Min(x1, x2))
    g = derive_by_array(f, (x1, x2))
    G = lambdify((x1, x2), g, modules='numpy')
    point = {x1: 1, x2: 2}
    assert (abs(g.subs(point) - G(*point.values())) <= 1e-10).all()


def test_issue_22739():
    if not numpy:
        skip("numpy not installed")

    x1, x2 = symbols('x1 x2')
    f = Heaviside(Min(x1, x2))
    F = lambdify((x1, x2), f, modules='numpy')
    point = {x1: 1, x2: 2}
    assert abs(f.subs(point) - F(*point.values())) <= 1e-10


def test_issue_22992():
    if not numpy:
        skip("numpy not installed")

    a, t = symbols('a t')
    expr = a*(log(cot(t/2)) - cos(t))
    F = lambdify([a, t], expr, 'numpy')

    point = {a: 10, t: 2}

    assert abs(expr.subs(point) - F(*point.values())) <= 1e-10

    # Standard math
    F = lambdify([a, t], expr)

    assert abs(expr.subs(point) - F(*point.values())) <= 1e-10


def test_issue_19764():
    if not numpy:
        skip("numpy not installed")

    expr = Array([x, x**2])
    f = lambdify(x, expr, 'numpy')

    assert f(1).__class__ == numpy.ndarray

def test_issue_20070():
    if not numba:
        skip("numba not installed")

    f = lambdify(x, sin(x), 'numpy')
    assert numba.jit(f, nopython=True)(1)==0.8414709848078965


def test_fresnel_integrals_scipy():
    if not scipy:
        skip("scipy not installed")

    f1 = fresnelc(x)
    f2 = fresnels(x)
    F1 = lambdify(x, f1, modules='scipy')
    F2 = lambdify(x, f2, modules='scipy')

    assert abs(fresnelc(1.3) - F1(1.3)) <= 1e-10
    assert abs(fresnels(1.3) - F2(1.3)) <= 1e-10


def test_beta_scipy():
    if not scipy:
        skip("scipy not installed")

    f = beta(x, y)
    F = lambdify((x, y), f, modules='scipy')

    assert abs(beta(1.3, 2.3) - F(1.3, 2.3)) <= 1e-10


def test_beta_math():
    f = beta(x, y)
    F = lambdify((x, y), f, modules='math')

    assert abs(beta(1.3, 2.3) - F(1.3, 2.3)) <= 1e-10


def test_betainc_scipy():
    if not scipy:
        skip("scipy not installed")

    f = betainc(w, x, y, z)
    F = lambdify((w, x, y, z), f, modules='scipy')

    assert abs(betainc(1.4, 3.1, 0.1, 0.5) - F(1.4, 3.1, 0.1, 0.5)) <= 1e-10


def test_betainc_regularized_scipy():
    if not scipy:
        skip("scipy not installed")

    f = betainc_regularized(w, x, y, z)
    F = lambdify((w, x, y, z), f, modules='scipy')

    assert abs(betainc_regularized(0.2, 3.5, 0.1, 1) - F(0.2, 3.5, 0.1, 1)) <= 1e-10


def test_numpy_special_math():
    if not numpy:
        skip("numpy not installed")

    funcs = [expm1, log1p, exp2, log2, log10, hypot, logaddexp, logaddexp2]
    for func in funcs:
        if 2 in func.nargs:
            expr = func(x, y)
            args = (x, y)
            num_args = (0.3, 0.4)
        elif 1 in func.nargs:
            expr = func(x)
            args = (x,)
            num_args = (0.3,)
        else:
            raise NotImplementedError("Need to handle other than unary & binary functions in test")
        f = lambdify(args, expr)
        result = f(*num_args)
        reference = expr.subs(dict(zip(args, num_args))).evalf()
        assert numpy.allclose(result, float(reference))

    lae2 = lambdify((x, y), logaddexp2(log2(x), log2(y)))
    assert abs(2.0**lae2(1e-50, 2.5e-50) - 3.5e-50) < 1e-62  # from NumPy's docstring


def test_scipy_special_math():
    if not scipy:
        skip("scipy not installed")

    cm1 = lambdify((x,), cosm1(x), modules='scipy')
    assert abs(cm1(1e-20) + 5e-41) < 1e-200

    have_scipy_1_10plus = tuple(map(int, scipy.version.version.split('.')[:2])) >= (1, 10)

    if have_scipy_1_10plus:
        cm2 = lambdify((x, y), powm1(x, y), modules='scipy')
        assert abs(cm2(1.2, 1e-9) - 1.82321557e-10)  < 1e-17


def test_scipy_bernoulli():
    if not scipy:
        skip("scipy not installed")

    bern = lambdify((x,), bernoulli(x), modules='scipy')
    assert bern(1) == 0.5


def test_scipy_harmonic():
    if not scipy:
        skip("scipy not installed")

    hn = lambdify((x,), harmonic(x), modules='scipy')
    assert hn(2) == 1.5
    hnm = lambdify((x, y), harmonic(x, y), modules='scipy')
    assert hnm(2, 2) == 1.25


def test_cupy_array_arg():
    if not cupy:
        skip("CuPy not installed")

    f = lambdify([[x, y]], x*x + y, 'cupy')
    result = f(cupy.array([2.0, 1.0]))
    assert result == 5
    assert "cupy" in str(type(result))


def test_cupy_array_arg_using_numpy():
    # numpy functions can be run on cupy arrays
    # unclear if we can "officially" support this,
    # depends on numpy __array_function__ support
    if not cupy:
        skip("CuPy not installed")

    f = lambdify([[x, y]], x*x + y, 'numpy')
    result = f(cupy.array([2.0, 1.0]))
    assert result == 5
    assert "cupy" in str(type(result))

def test_cupy_dotproduct():
    if not cupy:
        skip("CuPy not installed")

    A = Matrix([x, y, z])
    f1 = lambdify([x, y, z], DotProduct(A, A), modules='cupy')
    f2 = lambdify([x, y, z], DotProduct(A, A.T), modules='cupy')
    f3 = lambdify([x, y, z], DotProduct(A.T, A), modules='cupy')
    f4 = lambdify([x, y, z], DotProduct(A, A.T), modules='cupy')

    assert f1(1, 2, 3) == \
        f2(1, 2, 3) == \
        f3(1, 2, 3) == \
        f4(1, 2, 3) == \
        cupy.array([14])


def test_jax_array_arg():
    if not jax:
        skip("JAX not installed")

    f = lambdify([[x, y]], x*x + y, 'jax')
    result = f(jax.numpy.array([2.0, 1.0]))
    assert result == 5
    assert "jax" in str(type(result))


def test_jax_array_arg_using_numpy():
    if not jax:
        skip("JAX not installed")

    f = lambdify([[x, y]], x*x + y, 'numpy')
    result = f(jax.numpy.array([2.0, 1.0]))
    assert result == 5
    assert "jax" in str(type(result))


def test_jax_dotproduct():
    if not jax:
        skip("JAX not installed")

    A = Matrix([x, y, z])
    f1 = lambdify([x, y, z], DotProduct(A, A), modules='jax')
    f2 = lambdify([x, y, z], DotProduct(A, A.T), modules='jax')
    f3 = lambdify([x, y, z], DotProduct(A.T, A), modules='jax')
    f4 = lambdify([x, y, z], DotProduct(A, A.T), modules='jax')

    assert f1(1, 2, 3) == \
        f2(1, 2, 3) == \
        f3(1, 2, 3) == \
        f4(1, 2, 3) == \
        jax.numpy.array([14])


def test_lambdify_cse():
    def no_op_cse(exprs):
        return (), exprs

    def dummy_cse(exprs):
        from sympy.simplify.cse_main import cse
        return cse(exprs, symbols=numbered_symbols(cls=Dummy))

    def minmem(exprs):
        from sympy.simplify.cse_main import cse_release_variables, cse
        return cse(exprs, postprocess=cse_release_variables)

    class Case:
        def __init__(self, *, args, exprs, num_args, requires_numpy=False):
            self.args = args
            self.exprs = exprs
            self.num_args = num_args
            subs_dict = dict(zip(self.args, self.num_args))
            self.ref = [e.subs(subs_dict).evalf() for e in exprs]
            self.requires_numpy = requires_numpy

        def lambdify(self, *, cse):
            return lambdify(self.args, self.exprs, cse=cse)

        def assertAllClose(self, result, *, abstol=1e-15, reltol=1e-15):
            if self.requires_numpy:
                assert all(numpy.allclose(result[i], numpy.asarray(r, dtype=float),
                                          rtol=reltol, atol=abstol)
                           for i, r in enumerate(self.ref))
                return

            for i, r in enumerate(self.ref):
                abs_err = abs(result[i] - r)
                if r == 0:
                    assert abs_err < abstol
                else:
                    assert abs_err/abs(r) < reltol

    cases = [
        Case(
            args=(x, y, z),
            exprs=[
             x + y + z,
             x + y - z,
             2*x + 2*y - z,
             (x+y)**2 + (y+z)**2,
            ],
            num_args=(2., 3., 4.)
        ),
        Case(
            args=(x, y, z),
            exprs=[
            x + sympy.Heaviside(x),
            y + sympy.Heaviside(x),
            z + sympy.Heaviside(x, 1),
            z/sympy.Heaviside(x, 1)
            ],
            num_args=(0., 3., 4.)
        ),
        Case(
            args=(x, y, z),
            exprs=[
            x + sinc(y),
            y + sinc(y),
            z - sinc(y)
            ],
            num_args=(0.1, 0.2, 0.3)
        ),
        Case(
            args=(x, y, z),
            exprs=[
                Matrix([[x, x*y], [sin(z) + 4, x**z]]),
                x*y+sin(z)-x**z,
                Matrix([x*x, sin(z), x**z])
            ],
            num_args=(1.,2.,3.),
            requires_numpy=True
        ),
        Case(
            args=(x, y),
            exprs=[(x + y - 1)**2, x, x + y,
            (x + y)/(2*x + 1) + (x + y - 1)**2, (2*x + 1)**(x + y)],
            num_args=(1,2)
        )
    ]
    for case in cases:
        if not numpy and case.requires_numpy:
            continue
        for _cse in [False, True, minmem, no_op_cse, dummy_cse]:
            f = case.lambdify(cse=_cse)
            result = f(*case.num_args)
            case.assertAllClose(result)

def test_issue_25288():
    syms = numbered_symbols(cls=Dummy)
    ok = lambdify(x, [x**2, sin(x**2)], cse=lambda e: cse(e, symbols=syms))(2)
    assert ok


def test_deprecated_set():
    with warns_deprecated_sympy():
        lambdify({x, y}, x + y)

def test_issue_13881():
    if not numpy:
        skip("numpy not installed.")

    X = MatrixSymbol('X', 3, 1)

    f = lambdify(X, X.T*X, 'numpy')
    assert f(numpy.array([1, 2, 3])) == 14
    assert f(numpy.array([3, 2, 1])) == 14

    f = lambdify(X, X*X.T, 'numpy')
    assert f(numpy.array([1, 2, 3])) == 14
    assert f(numpy.array([3, 2, 1])) == 14

    f = lambdify(X, (X*X.T)*X, 'numpy')
    arr1 = numpy.array([[1], [2], [3]])
    arr2 = numpy.array([[14],[28],[42]])

    assert numpy.array_equal(f(arr1), arr2)


def test_23536_lambdify_cse_dummy():

    f = Function('x')(y)
    g = Function('w')(y)
    expr = z + (f**4 + g**5)*(f**3 + (g*f)**3)
    expr = expr.expand()
    eval_expr = lambdify(((f, g), z), expr, cse=True)
    ans = eval_expr((1.0, 2.0), 3.0)  # shouldn't raise NameError
    assert ans == 300.0  # not a list and value is 300


class LambdifyDocstringTestCase:
    SIGNATURE = None
    EXPR = None
    SRC = None

    def __init__(self, docstring_limit, expected_redacted):
        self.docstring_limit = docstring_limit
        self.expected_redacted = expected_redacted

    @property
    def expected_expr(self):
        expr_redacted_msg = "EXPRESSION REDACTED DUE TO LENGTH, (see lambdify's `docstring_limit`)"
        return self.EXPR if not self.expected_redacted else expr_redacted_msg

    @property
    def expected_src(self):
        src_redacted_msg = "SOURCE CODE REDACTED DUE TO LENGTH, (see lambdify's `docstring_limit`)"
        return self.SRC if not self.expected_redacted else src_redacted_msg

    @property
    def expected_docstring(self):
        expected_docstring = (
            f'Created with lambdify. Signature:\n\n'
            f'func({self.SIGNATURE})\n\n'
            f'Expression:\n\n'
            f'{self.expected_expr}\n\n'
            f'Source code:\n\n'
            f'{self.expected_src}\n\n'
            f'Imported modules:\n\n'
        )
        return expected_docstring

    def __len__(self):
        return len(self.expected_docstring)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'docstring_limit={self.docstring_limit}, '
            f'expected_redacted={self.expected_redacted})'
        )


def test_lambdify_docstring_size_limit_simple_symbol():

    class SimpleSymbolTestCase(LambdifyDocstringTestCase):
        SIGNATURE = 'x'
        EXPR = 'x'
        SRC = (
            'def _lambdifygenerated(x):\n'
            '    return x\n'
        )

    x = symbols('x')

    test_cases = (
        SimpleSymbolTestCase(docstring_limit=None, expected_redacted=False),
        SimpleSymbolTestCase(docstring_limit=100, expected_redacted=False),
        SimpleSymbolTestCase(docstring_limit=1, expected_redacted=False),
        SimpleSymbolTestCase(docstring_limit=0, expected_redacted=True),
        SimpleSymbolTestCase(docstring_limit=-1, expected_redacted=True),
    )
    for test_case in test_cases:
        lambdified_expr = lambdify(
            [x],
            x,
            'sympy',
            docstring_limit=test_case.docstring_limit,
        )
        assert lambdified_expr.__doc__ == test_case.expected_docstring


def test_lambdify_docstring_size_limit_nested_expr():

    class ExprListTestCase(LambdifyDocstringTestCase):
        SIGNATURE = 'x, y, z'
        EXPR = (
            '[x, [y], z, x**3 + 3*x**2*y + 3*x**2*z + 3*x*y**2 + 6*x*y*z '
            '+ 3*x*z**2 +...'
        )
        SRC = (
            'def _lambdifygenerated(x, y, z):\n'
            '    return [x, [y], z, x**3 + 3*x**2*y + 3*x**2*z + 3*x*y**2 '
            '+ 6*x*y*z + 3*x*z**2 + y**3 + 3*y**2*z + 3*y*z**2 + z**3]\n'
        )

    x, y, z = symbols('x, y, z')
    expr = [x, [y], z, ((x + y + z)**3).expand()]

    test_cases = (
        ExprListTestCase(docstring_limit=None, expected_redacted=False),
        ExprListTestCase(docstring_limit=200, expected_redacted=False),
        ExprListTestCase(docstring_limit=50, expected_redacted=True),
        ExprListTestCase(docstring_limit=0, expected_redacted=True),
        ExprListTestCase(docstring_limit=-1, expected_redacted=True),
    )
    for test_case in test_cases:
        lambdified_expr = lambdify(
            [x, y, z],
            expr,
            'sympy',
            docstring_limit=test_case.docstring_limit,
        )
        assert lambdified_expr.__doc__ == test_case.expected_docstring


def test_lambdify_docstring_size_limit_matrix():

    class MatrixTestCase(LambdifyDocstringTestCase):
        SIGNATURE = 'x, y, z'
        EXPR = (
            'Matrix([[0, x], [x + y + z, x**3 + 3*x**2*y + 3*x**2*z + 3*x*y**2 '
            '+ 6*x*y*z...'
        )
        SRC = (
            'def _lambdifygenerated(x, y, z):\n'
            '    return ImmutableDenseMatrix([[0, x], [x + y + z, x**3 '
            '+ 3*x**2*y + 3*x**2*z + 3*x*y**2 + 6*x*y*z + 3*x*z**2 + y**3 '
            '+ 3*y**2*z + 3*y*z**2 + z**3]])\n'
        )

    x, y, z = symbols('x, y, z')
    expr = Matrix([[S.Zero, x], [x + y + z, ((x + y + z)**3).expand()]])

    test_cases = (
        MatrixTestCase(docstring_limit=None, expected_redacted=False),
        MatrixTestCase(docstring_limit=200, expected_redacted=False),
        MatrixTestCase(docstring_limit=50, expected_redacted=True),
        MatrixTestCase(docstring_limit=0, expected_redacted=True),
        MatrixTestCase(docstring_limit=-1, expected_redacted=True),
    )
    for test_case in test_cases:
        lambdified_expr = lambdify(
            [x, y, z],
            expr,
            'sympy',
            docstring_limit=test_case.docstring_limit,
        )
        assert lambdified_expr.__doc__ == test_case.expected_docstring


def test_lambdify_empty_tuple():
    a = symbols("a")
    expr = ((), (a,))
    f = lambdify(a, expr)
    result = f(1)
    assert result == ((), (1,)), "Lambdify did not handle the empty tuple correctly."


def test_assoc_legendre_numerical_evaluation():

    tol = 1e-10

    sympy_result_integer = assoc_legendre(1, 1/2, 0.1).evalf()
    sympy_result_complex = assoc_legendre(2, 1, 3).evalf()
    mpmath_result_integer = -0.474572528387641
    mpmath_result_complex = -25.45584412271571*I

    assert all_close(sympy_result_integer, mpmath_result_integer, tol)
    assert all_close(sympy_result_complex, mpmath_result_complex, tol)


def test_Piecewise():

    modules = [math]
    if numpy:
        modules.append('numpy')

    for mod in modules:
        # test isinf
        f = lambdify(x, Piecewise((7.0, isinf(x)), (3.0, True)), mod)
        assert f(+float('inf')) == +7.0
        assert f(-float('inf')) == +7.0
        assert f(42.) == 3.0

        f2 = lambdify(x, Piecewise((7.0*sign(x), isinf(x)), (3.0, True)), mod)
        assert f2(+float('inf')) == +7.0
        assert f2(-float('inf')) == -7.0
        assert f2(42.) == 3.0

        # test isnan (gh-26784)
        g = lambdify(x, Piecewise((7.0, isnan(x)), (3.0, True)), mod)
        assert g(float('nan')) == 7.0
        assert g(42.) == 3.0


def test_array_symbol():
    if not numpy:
        skip("numpy not installed.")
    a = ArraySymbol('a', (3,))
    f = lambdify((a), a)
    assert numpy.all(f(numpy.array([1,2,3])) == numpy.array([1,2,3]))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\attrs\__init__.py ===
# SPDX-License-Identifier: MIT

from attr import (
    NOTHING,
    Attribute,
    AttrsInstance,
    Converter,
    Factory,
    NothingType,
    _make_getattr,
    assoc,
    cmp_using,
    define,
    evolve,
    field,
    fields,
    fields_dict,
    frozen,
    has,
    make_class,
    mutable,
    resolve_types,
    validate,
)
from attr._next_gen import asdict, astuple

from . import converters, exceptions, filters, setters, validators


__all__ = [
    "NOTHING",
    "Attribute",
    "AttrsInstance",
    "Converter",
    "Factory",
    "NothingType",
    "__author__",
    "__copyright__",
    "__description__",
    "__doc__",
    "__email__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
    "asdict",
    "assoc",
    "astuple",
    "cmp_using",
    "converters",
    "define",
    "evolve",
    "exceptions",
    "field",
    "fields",
    "fields_dict",
    "filters",
    "frozen",
    "has",
    "make_class",
    "mutable",
    "resolve_types",
    "setters",
    "validate",
    "validators",
]

__getattr__ = _make_getattr(__name__)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\_null_file.py ===
from types import TracebackType
from typing import IO, Iterable, Iterator, List, Optional, Type


class NullFile(IO[str]):
    def close(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def read(self, __n: int = 1) -> str:
        return ""

    def readable(self) -> bool:
        return False

    def readline(self, __limit: int = 1) -> str:
        return ""

    def readlines(self, __hint: int = 1) -> List[str]:
        return []

    def seek(self, __offset: int, __whence: int = 1) -> int:
        return 0

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        return 0

    def truncate(self, __size: Optional[int] = 1) -> int:
        return 0

    def writable(self) -> bool:
        return False

    def writelines(self, __lines: Iterable[str]) -> None:
        pass

    def __next__(self) -> str:
        return ""

    def __iter__(self) -> Iterator[str]:
        return iter([""])

    def __enter__(self) -> IO[str]:
        return self

    def __exit__(
        self,
        __t: Optional[Type[BaseException]],
        __value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> None:
        pass

    def write(self, text: str) -> int:
        return 0

    def flush(self) -> None:
        pass

    def fileno(self) -> int:
        return -1


NULL_FILE = NullFile()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\remote\fedcm.py ===
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

from typing import List, Optional

from .command import Command


class FedCM:
    def __init__(self, driver) -> None:
        self._driver = driver

    @property
    def title(self) -> str:
        """Gets the title of the dialog."""
        return self._driver.execute(Command.GET_FEDCM_TITLE)["value"].get("title")

    @property
    def subtitle(self) -> Optional[str]:
        """Gets the subtitle of the dialog."""
        return self._driver.execute(Command.GET_FEDCM_TITLE)["value"].get("subtitle")

    @property
    def dialog_type(self) -> str:
        """Gets the type of the dialog currently being shown."""
        return self._driver.execute(Command.GET_FEDCM_DIALOG_TYPE).get("value")

    @property
    def account_list(self) -> List[dict]:
        """Gets the list of accounts shown in the dialog."""
        return self._driver.execute(Command.GET_FEDCM_ACCOUNT_LIST).get("value")

    def select_account(self, index: int) -> None:
        """Selects an account from the dialog by index."""
        self._driver.execute(Command.SELECT_FEDCM_ACCOUNT, {"accountIndex": index})

    def accept(self) -> None:
        """Clicks the continue button in the dialog."""
        self._driver.execute(Command.CLICK_FEDCM_DIALOG_BUTTON, {"dialogButton": "ConfirmIdpLoginContinue"})

    def dismiss(self) -> None:
        """Cancels/dismisses the FedCM dialog."""
        self._driver.execute(Command.CANCEL_FEDCM_DIALOG)

    def enable_delay(self) -> None:
        """Re-enables the promise rejection delay for FedCM."""
        self._driver.execute(Command.SET_FEDCM_DELAY, {"enabled": True})

    def disable_delay(self) -> None:
        """Disables the promise rejection delay for FedCM."""
        self._driver.execute(Command.SET_FEDCM_DELAY, {"enabled": False})

    def reset_cooldown(self) -> None:
        """Resets the FedCM dialog cooldown, allowing immediate retriggers."""
        self._driver.execute(Command.RESET_FEDCM_COOLDOWN)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\logic\utilities\dimacs.py ===
"""For reading in DIMACS file format

www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/satformat.ps

"""

from sympy.core import Symbol
from sympy.logic.boolalg import And, Or
import re
from pathlib import Path


def load(s):
    """Loads a boolean expression from a string.

    Examples
    ========

    >>> from sympy.logic.utilities.dimacs import load
    >>> load('1')
    cnf_1
    >>> load('1 2')
    cnf_1 | cnf_2
    >>> load('1 \\n 2')
    cnf_1 & cnf_2
    >>> load('1 2 \\n 3')
    cnf_3 & (cnf_1 | cnf_2)
    """
    clauses = []

    lines = s.split('\n')

    pComment = re.compile(r'c.*')
    pStats = re.compile(r'p\s*cnf\s*(\d*)\s*(\d*)')

    while len(lines) > 0:
        line = lines.pop(0)

        # Only deal with lines that aren't comments
        if not pComment.match(line):
            m = pStats.match(line)

            if not m:
                nums = line.rstrip('\n').split(' ')
                list = []
                for lit in nums:
                    if lit != '':
                        if int(lit) == 0:
                            continue
                        num = abs(int(lit))
                        sign = True
                        if int(lit) < 0:
                            sign = False

                        if sign:
                            list.append(Symbol("cnf_%s" % num))
                        else:
                            list.append(~Symbol("cnf_%s" % num))

                if len(list) > 0:
                    clauses.append(Or(*list))

    return And(*clauses)


def load_file(location):
    """Loads a boolean expression from a file."""
    s = Path(location).read_text()
    return load(s)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\testing\_checkpoints.py ===
from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING

from .. import _core

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def _assert_yields_or_not(expected: bool) -> Generator[None, None, None]:
    """Check if checkpoints are executed in a block of code."""
    __tracebackhide__ = True
    task = _core.current_task()
    orig_cancel = task._cancel_points
    orig_schedule = task._schedule_points
    try:
        yield
        if expected and (
            task._cancel_points == orig_cancel or task._schedule_points == orig_schedule
        ):
            raise AssertionError("assert_checkpoints block did not yield!")
    finally:
        if not expected and (
            task._cancel_points != orig_cancel or task._schedule_points != orig_schedule
        ):
            raise AssertionError("assert_no_checkpoints block yielded!")


def assert_checkpoints() -> AbstractContextManager[None]:
    """Use as a context manager to check that the code inside the ``with``
    block either exits with an exception or executes at least one
    :ref:`checkpoint <checkpoints>`.

    Raises:
      AssertionError: if no checkpoint was executed.

    Example:
      Check that :func:`trio.sleep` is a checkpoint, even if it doesn't
      block::

         with trio.testing.assert_checkpoints():
             await trio.sleep(0)

    """
    __tracebackhide__ = True
    return _assert_yields_or_not(True)


def assert_no_checkpoints() -> AbstractContextManager[None]:
    """Use as a context manager to check that the code inside the ``with``
    block does not execute any :ref:`checkpoints <checkpoints>`.

    Raises:
      AssertionError: if a checkpoint was executed.

    Example:
      Synchronous code never contains any checkpoints, but we can double-check
      that::

         send_channel, receive_channel = trio.open_memory_channel(10)
         with trio.testing.assert_no_checkpoints():
             send_channel.send_nowait(None)

    """
    __tracebackhide__ = True
    return _assert_yields_or_not(False)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\tests\test_trigonometric.py ===
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.add import Add
from sympy.core.function import (Lambda, diff)
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (arg, conjugate, im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acoth, asinh, atanh, cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin, atan, atan2,
                                                      cos, cot, csc, sec, sin, sinc, tan)
from sympy.functions.special.bessel import (besselj, jn)
from sympy.functions.special.delta_functions import Heaviside
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (cancel, gcd)
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.series.series import series
from sympy.sets.fancysets import ImageSet
from sympy.sets.sets import (FiniteSet, Interval)
from sympy.simplify.simplify import simplify
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError, PoleError
from sympy.core.relational import Ne, Eq
from sympy.functions.elementary.piecewise import Piecewise
from sympy.sets.setexpr import SetExpr
from sympy.testing.pytest import XFAIL, slow, raises


x, y, z = symbols('x y z')
r = Symbol('r', real=True)
k, m = symbols('k m', integer=True)
p = Symbol('p', positive=True)
n = Symbol('n', negative=True)
np = Symbol('p', nonpositive=True)
nn = Symbol('n', nonnegative=True)
nz = Symbol('nz', nonzero=True)
ep = Symbol('ep', extended_positive=True)
en = Symbol('en', extended_negative=True)
enp = Symbol('ep', extended_nonpositive=True)
enn = Symbol('en', extended_nonnegative=True)
enz = Symbol('enz', extended_nonzero=True)
a = Symbol('a', algebraic=True)
na = Symbol('na', nonzero=True, algebraic=True)


def test_sin():
    x, y = symbols('x y')
    z = symbols('z', imaginary=True)

    assert sin.nargs == FiniteSet(1)
    assert sin(nan) is nan
    assert sin(zoo) is nan

    assert sin(oo) == AccumBounds(-1, 1)
    assert sin(oo) - sin(oo) == AccumBounds(-2, 2)
    assert sin(oo*I) == oo*I
    assert sin(-oo*I) == -oo*I
    assert 0*sin(oo) is S.Zero
    assert 0/sin(oo) is S.Zero
    assert 0 + sin(oo) == AccumBounds(-1, 1)
    assert 5 + sin(oo) == AccumBounds(4, 6)

    assert sin(0) == 0

    assert sin(z*I) == I*sinh(z)
    assert sin(asin(x)) == x
    assert sin(atan(x)) == x / sqrt(1 + x**2)
    assert sin(acos(x)) == sqrt(1 - x**2)
    assert sin(acot(x)) == 1 / (sqrt(1 + 1 / x**2) * x)
    assert sin(acsc(x)) == 1 / x
    assert sin(asec(x)) == sqrt(1 - 1 / x**2)
    assert sin(atan2(y, x)) == y / sqrt(x**2 + y**2)

    assert sin(pi*I) == sinh(pi)*I
    assert sin(-pi*I) == -sinh(pi)*I
    assert sin(-2*I) == -sinh(2)*I

    assert sin(pi) == 0
    assert sin(-pi) == 0
    assert sin(2*pi) == 0
    assert sin(-2*pi) == 0
    assert sin(-3*10**73*pi) == 0
    assert sin(7*10**103*pi) == 0

    assert sin(pi/2) == 1
    assert sin(-pi/2) == -1
    assert sin(pi*Rational(5, 2)) == 1
    assert sin(pi*Rational(7, 2)) == -1

    ne = symbols('ne', integer=True, even=False)
    e = symbols('e', even=True)
    assert sin(pi*ne/2) == (-1)**(ne/2 - S.Half)
    assert sin(pi*k/2).func == sin
    assert sin(pi*e/2) == 0
    assert sin(pi*k) == 0
    assert sin(pi*k).subs(k, 3) == sin(pi*k/2).subs(k, 6)  # issue 8298

    assert sin(pi/3) == S.Half*sqrt(3)
    assert sin(pi*Rational(-2, 3)) == Rational(-1, 2)*sqrt(3)

    assert sin(pi/4) == S.Half*sqrt(2)
    assert sin(-pi/4) == Rational(-1, 2)*sqrt(2)
    assert sin(pi*Rational(17, 4)) == S.Half*sqrt(2)
    assert sin(pi*Rational(-3, 4)) == Rational(-1, 2)*sqrt(2)

    assert sin(pi/6) == S.Half
    assert sin(-pi/6) == Rational(-1, 2)
    assert sin(pi*Rational(7, 6)) == Rational(-1, 2)
    assert sin(pi*Rational(-5, 6)) == Rational(-1, 2)

    assert sin(pi*Rational(1, 5)) == sqrt((5 - sqrt(5)) / 8)
    assert sin(pi*Rational(2, 5)) == sqrt((5 + sqrt(5)) / 8)
    assert sin(pi*Rational(3, 5)) == sin(pi*Rational(2, 5))
    assert sin(pi*Rational(4, 5)) == sin(pi*Rational(1, 5))
    assert sin(pi*Rational(6, 5)) == -sin(pi*Rational(1, 5))
    assert sin(pi*Rational(8, 5)) == -sin(pi*Rational(2, 5))

    assert sin(pi*Rational(-1273, 5)) == -sin(pi*Rational(2, 5))

    assert sin(pi/8) == sqrt((2 - sqrt(2))/4)

    assert sin(pi/10) == Rational(-1, 4) + sqrt(5)/4

    assert sin(pi/12) == -sqrt(2)/4 + sqrt(6)/4
    assert sin(pi*Rational(5, 12)) == sqrt(2)/4 + sqrt(6)/4
    assert sin(pi*Rational(-7, 12)) == -sqrt(2)/4 - sqrt(6)/4
    assert sin(pi*Rational(-11, 12)) == sqrt(2)/4 - sqrt(6)/4

    assert sin(pi*Rational(104, 105)) == sin(pi/105)
    assert sin(pi*Rational(106, 105)) == -sin(pi/105)

    assert sin(pi*Rational(-104, 105)) == -sin(pi/105)
    assert sin(pi*Rational(-106, 105)) == sin(pi/105)

    assert sin(x*I) == sinh(x)*I

    assert sin(k*pi) == 0
    assert sin(17*k*pi) == 0
    assert sin(2*k*pi + 4) == sin(4)
    assert sin(2*k*pi + m*pi + 1) == (-1)**(m + 2*k)*sin(1)

    assert sin(k*pi*I) == sinh(k*pi)*I

    assert sin(r).is_real is True

    assert sin(0, evaluate=False).is_algebraic
    assert sin(a).is_algebraic is None
    assert sin(na).is_algebraic is False
    q = Symbol('q', rational=True)
    assert sin(pi*q).is_algebraic
    qn = Symbol('qn', rational=True, nonzero=True)
    assert sin(qn).is_rational is False
    assert sin(q).is_rational is None  # issue 8653

    assert isinstance(sin( re(x) - im(y)), sin) is True
    assert isinstance(sin(-re(x) + im(y)), sin) is False

    assert sin(SetExpr(Interval(0, 1))) == SetExpr(ImageSet(Lambda(x, sin(x)),
                       Interval(0, 1)))

    for d in list(range(1, 22)) + [60, 85]:
        for n in range(d*2 + 1):
            x = n*pi/d
            e = abs( float(sin(x)) - sin(float(x)) )
            assert e < 1e-12

    assert sin(0, evaluate=False).is_zero is True
    assert sin(k*pi, evaluate=False).is_zero is True

    assert sin(Add(1, -1, evaluate=False), evaluate=False).is_zero is True


def test_sin_cos():
    for d in [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 24, 30, 40, 60, 120]:  # list is not exhaustive...
        for n in range(-2*d, d*2):
            x = n*pi/d
            assert sin(x + pi/2) == cos(x), "fails for %d*pi/%d" % (n, d)
            assert sin(x - pi/2) == -cos(x), "fails for %d*pi/%d" % (n, d)
            assert sin(x) == cos(x - pi/2), "fails for %d*pi/%d" % (n, d)
            assert -sin(x) == cos(x + pi/2), "fails for %d*pi/%d" % (n, d)


def test_sin_series():
    assert sin(x).series(x, 0, 9) == \
        x - x**3/6 + x**5/120 - x**7/5040 + O(x**9)


def test_sin_rewrite():
    assert sin(x).rewrite(exp) == -I*(exp(I*x) - exp(-I*x))/2
    assert sin(x).rewrite(tan) == 2*tan(x/2)/(1 + tan(x/2)**2)
    assert sin(x).rewrite(cot) == \
        Piecewise((0, Eq(im(x), 0) & Eq(Mod(x, pi), 0)),
                  (2*cot(x/2)/(cot(x/2)**2 + 1), True))
    assert sin(sinh(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, sinh(3)).n()
    assert sin(cosh(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, cosh(3)).n()
    assert sin(tanh(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, tanh(3)).n()
    assert sin(coth(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, coth(3)).n()
    assert sin(sin(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, sin(3)).n()
    assert sin(cos(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, cos(3)).n()
    assert sin(tan(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, tan(3)).n()
    assert sin(cot(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, cot(3)).n()
    assert sin(log(x)).rewrite(Pow) == I*x**-I / 2 - I*x**I /2
    assert sin(x).rewrite(csc) == 1/csc(x)
    assert sin(x).rewrite(cos) == cos(x - pi / 2, evaluate=False)
    assert sin(x).rewrite(sec) == 1 / sec(x - pi / 2, evaluate=False)
    assert sin(cos(x)).rewrite(Pow) == sin(cos(x))
    assert sin(x).rewrite(besselj) == sqrt(pi*x/2)*besselj(S.Half, x)
    assert sin(x).rewrite(besselj).subs(x, 0) == sin(0)


def _test_extrig(f, i, e):
    from sympy.core.function import expand_trig
    assert unchanged(f, i)
    assert expand_trig(f(i)) == f(i)
    # testing directly instead of with .expand(trig=True)
    # because the other expansions undo the unevaluated Mul
    assert expand_trig(f(Mul(i, 1, evaluate=False))) == e
    assert abs(f(i) - e).n() < 1e-10


def test_sin_expansion():
    # Note: these formulas are not unique.  The ones here come from the
    # Chebyshev formulas.
    assert sin(x + y).expand(trig=True) == sin(x)*cos(y) + cos(x)*sin(y)
    assert sin(x - y).expand(trig=True) == sin(x)*cos(y) - cos(x)*sin(y)
    assert sin(y - x).expand(trig=True) == cos(x)*sin(y) - sin(x)*cos(y)
    assert sin(2*x).expand(trig=True) == 2*sin(x)*cos(x)
    assert sin(3*x).expand(trig=True) == -4*sin(x)**3 + 3*sin(x)
    assert sin(4*x).expand(trig=True) == -8*sin(x)**3*cos(x) + 4*sin(x)*cos(x)
    assert sin(2*pi/17).expand(trig=True) == sin(2*pi/17, evaluate=False)
    assert sin(x+pi/17).expand(trig=True) == sin(pi/17)*cos(x) + cos(pi/17)*sin(x)
    _test_extrig(sin, 2, 2*sin(1)*cos(1))
    _test_extrig(sin, 3, -4*sin(1)**3 + 3*sin(1))


def test_sin_AccumBounds():
    assert sin(AccumBounds(-oo, oo)) == AccumBounds(-1, 1)
    assert sin(AccumBounds(0, oo)) == AccumBounds(-1, 1)
    assert sin(AccumBounds(-oo, 0)) == AccumBounds(-1, 1)
    assert sin(AccumBounds(0, 2*S.Pi)) == AccumBounds(-1, 1)
    assert sin(AccumBounds(0, S.Pi*Rational(3, 4))) == AccumBounds(0, 1)
    assert sin(AccumBounds(S.Pi*Rational(3, 4), S.Pi*Rational(7, 4))) == AccumBounds(-1, sin(S.Pi*Rational(3, 4)))
    assert sin(AccumBounds(S.Pi/4, S.Pi/3)) == AccumBounds(sin(S.Pi/4), sin(S.Pi/3))
    assert sin(AccumBounds(S.Pi*Rational(3, 4), S.Pi*Rational(5, 6))) == AccumBounds(sin(S.Pi*Rational(5, 6)), sin(S.Pi*Rational(3, 4)))


def test_sin_fdiff():
    assert sin(x).fdiff() == cos(x)
    raises(ArgumentIndexError, lambda: sin(x).fdiff(2))


def test_trig_symmetry():
    assert sin(-x) == -sin(x)
    assert cos(-x) == cos(x)
    assert tan(-x) == -tan(x)
    assert cot(-x) == -cot(x)
    assert sin(x + pi) == -sin(x)
    assert sin(x + 2*pi) == sin(x)
    assert sin(x + 3*pi) == -sin(x)
    assert sin(x + 4*pi) == sin(x)
    assert sin(x - 5*pi) == -sin(x)
    assert cos(x + pi) == -cos(x)
    assert cos(x + 2*pi) == cos(x)
    assert cos(x + 3*pi) == -cos(x)
    assert cos(x + 4*pi) == cos(x)
    assert cos(x - 5*pi) == -cos(x)
    assert tan(x + pi) == tan(x)
    assert tan(x - 3*pi) == tan(x)
    assert cot(x + pi) == cot(x)
    assert cot(x - 3*pi) == cot(x)
    assert sin(pi/2 - x) == cos(x)
    assert sin(pi*Rational(3, 2) - x) == -cos(x)
    assert sin(pi*Rational(5, 2) - x) == cos(x)
    assert cos(pi/2 - x) == sin(x)
    assert cos(pi*Rational(3, 2) - x) == -sin(x)
    assert cos(pi*Rational(5, 2) - x) == sin(x)
    assert tan(pi/2 - x) == cot(x)
    assert tan(pi*Rational(3, 2) - x) == cot(x)
    assert tan(pi*Rational(5, 2) - x) == cot(x)
    assert cot(pi/2 - x) == tan(x)
    assert cot(pi*Rational(3, 2) - x) == tan(x)
    assert cot(pi*Rational(5, 2) - x) == tan(x)
    assert sin(pi/2 + x) == cos(x)
    assert cos(pi/2 + x) == -sin(x)
    assert tan(pi/2 + x) == -cot(x)
    assert cot(pi/2 + x) == -tan(x)


def test_cos():
    x, y = symbols('x y')

    assert cos.nargs == FiniteSet(1)
    assert cos(nan) is nan

    assert cos(oo) == AccumBounds(-1, 1)
    assert cos(oo) - cos(oo) == AccumBounds(-2, 2)
    assert cos(oo*I) is oo
    assert cos(-oo*I) is oo
    assert cos(zoo) is nan

    assert cos(0) == 1

    assert cos(acos(x)) == x
    assert cos(atan(x)) == 1 / sqrt(1 + x**2)
    assert cos(asin(x)) == sqrt(1 - x**2)
    assert cos(acot(x)) == 1 / sqrt(1 + 1 / x**2)
    assert cos(acsc(x)) == sqrt(1 - 1 / x**2)
    assert cos(asec(x)) == 1 / x
    assert cos(atan2(y, x)) == x / sqrt(x**2 + y**2)

    assert cos(pi*I) == cosh(pi)
    assert cos(-pi*I) == cosh(pi)
    assert cos(-2*I) == cosh(2)

    assert cos(pi/2) == 0
    assert cos(-pi/2) == 0
    assert cos(pi/2) == 0
    assert cos(-pi/2) == 0
    assert cos((-3*10**73 + 1)*pi/2) == 0
    assert cos((7*10**103 + 1)*pi/2) == 0

    n = symbols('n', integer=True, even=False)
    e = symbols('e', even=True)
    assert cos(pi*n/2) == 0
    assert cos(pi*e/2) == (-1)**(e/2)

    assert cos(pi) == -1
    assert cos(-pi) == -1
    assert cos(2*pi) == 1
    assert cos(5*pi) == -1
    assert cos(8*pi) == 1

    assert cos(pi/3) == S.Half
    assert cos(pi*Rational(-2, 3)) == Rational(-1, 2)

    assert cos(pi/4) == S.Half*sqrt(2)
    assert cos(-pi/4) == S.Half*sqrt(2)
    assert cos(pi*Rational(11, 4)) == Rational(-1, 2)*sqrt(2)
    assert cos(pi*Rational(-3, 4)) == Rational(-1, 2)*sqrt(2)

    assert cos(pi/6) == S.Half*sqrt(3)
    assert cos(-pi/6) == S.Half*sqrt(3)
    assert cos(pi*Rational(7, 6)) == Rational(-1, 2)*sqrt(3)
    assert cos(pi*Rational(-5, 6)) == Rational(-1, 2)*sqrt(3)

    assert cos(pi*Rational(1, 5)) == (sqrt(5) + 1)/4
    assert cos(pi*Rational(2, 5)) == (sqrt(5) - 1)/4
    assert cos(pi*Rational(3, 5)) == -cos(pi*Rational(2, 5))
    assert cos(pi*Rational(4, 5)) == -cos(pi*Rational(1, 5))
    assert cos(pi*Rational(6, 5)) == -cos(pi*Rational(1, 5))
    assert cos(pi*Rational(8, 5)) == cos(pi*Rational(2, 5))

    assert cos(pi*Rational(-1273, 5)) == -cos(pi*Rational(2, 5))

    assert cos(pi/8) == sqrt((2 + sqrt(2))/4)

    assert cos(pi/12) == sqrt(2)/4 + sqrt(6)/4
    assert cos(pi*Rational(5, 12)) == -sqrt(2)/4 + sqrt(6)/4
    assert cos(pi*Rational(7, 12)) == sqrt(2)/4 - sqrt(6)/4
    assert cos(pi*Rational(11, 12)) == -sqrt(2)/4 - sqrt(6)/4

    assert cos(pi*Rational(104, 105)) == -cos(pi/105)
    assert cos(pi*Rational(106, 105)) == -cos(pi/105)

    assert cos(pi*Rational(-104, 105)) == -cos(pi/105)
    assert cos(pi*Rational(-106, 105)) == -cos(pi/105)

    assert cos(x*I) == cosh(x)
    assert cos(k*pi*I) == cosh(k*pi)

    assert cos(r).is_real is True

    assert cos(0, evaluate=False).is_algebraic
    assert cos(a).is_algebraic is None
    assert cos(na).is_algebraic is False
    q = Symbol('q', rational=True)
    assert cos(pi*q).is_algebraic
    assert cos(pi*Rational(2, 7)).is_algebraic

    assert cos(k*pi) == (-1)**k
    assert cos(2*k*pi) == 1
    assert cos(0, evaluate=False).is_zero is False
    assert cos(Rational(1, 2)).is_zero is False
    # The following test will return None as the result, but really it should
    # be True even if it is not always possible to resolve an assumptions query.
    assert cos(asin(-1, evaluate=False), evaluate=False).is_zero is None
    for d in list(range(1, 22)) + [60, 85]:
        for n in range(2*d + 1):
            x = n*pi/d
            e = abs( float(cos(x)) - cos(float(x)) )
            assert e < 1e-12


def test_issue_6190():
    c = Float('123456789012345678901234567890.25', '')
    for cls in [sin, cos, tan, cot]:
        assert cls(c*pi) == cls(pi/4)
        assert cls(4.125*pi) == cls(pi/8)
        assert cls(4.7*pi) == cls((4.7 % 2)*pi)


def test_cos_series():
    assert cos(x).series(x, 0, 9) == \
        1 - x**2/2 + x**4/24 - x**6/720 + x**8/40320 + O(x**9)


def test_cos_rewrite():
    assert cos(x).rewrite(exp) == exp(I*x)/2 + exp(-I*x)/2
    assert cos(x).rewrite(tan) == (1 - tan(x/2)**2)/(1 + tan(x/2)**2)
    assert cos(x).rewrite(cot) == \
        Piecewise((1, Eq(im(x), 0) & Eq(Mod(x, 2*pi), 0)),
                  ((cot(x/2)**2 - 1)/(cot(x/2)**2 + 1), True))
    assert cos(sinh(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, sinh(3)).n()
    assert cos(cosh(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, cosh(3)).n()
    assert cos(tanh(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, tanh(3)).n()
    assert cos(coth(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, coth(3)).n()
    assert cos(sin(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, sin(3)).n()
    assert cos(cos(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, cos(3)).n()
    assert cos(tan(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, tan(3)).n()
    assert cos(cot(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, cot(3)).n()
    assert cos(log(x)).rewrite(Pow) == x**I/2 + x**-I/2
    assert cos(x).rewrite(sec) == 1/sec(x)
    assert cos(x).rewrite(sin) == sin(x + pi/2, evaluate=False)
    assert cos(x).rewrite(csc) == 1/csc(-x + pi/2, evaluate=False)
    assert cos(sin(x)).rewrite(Pow) == cos(sin(x))
    assert cos(x).rewrite(besselj) == Piecewise(
                (sqrt(pi*x/2)*besselj(-S.Half, x), Ne(x, 0)),
                (1, True)
            )
    assert cos(x).rewrite(besselj).subs(x, 0) == cos(0)


def test_cos_expansion():
    assert cos(x + y).expand(trig=True) == cos(x)*cos(y) - sin(x)*sin(y)
    assert cos(x - y).expand(trig=True) == cos(x)*cos(y) + sin(x)*sin(y)
    assert cos(y - x).expand(trig=True) == cos(x)*cos(y) + sin(x)*sin(y)
    assert cos(2*x).expand(trig=True) == 2*cos(x)**2 - 1
    assert cos(3*x).expand(trig=True) == 4*cos(x)**3 - 3*cos(x)
    assert cos(4*x).expand(trig=True) == 8*cos(x)**4 - 8*cos(x)**2 + 1
    assert cos(2*pi/17).expand(trig=True) == cos(2*pi/17, evaluate=False)
    assert cos(x+pi/17).expand(trig=True) == cos(pi/17)*cos(x) - sin(pi/17)*sin(x)
    _test_extrig(cos, 2, 2*cos(1)**2 - 1)
    _test_extrig(cos, 3, 4*cos(1)**3 - 3*cos(1))


def test_cos_AccumBounds():
    assert cos(AccumBounds(-oo, oo)) == AccumBounds(-1, 1)
    assert cos(AccumBounds(0, oo)) == AccumBounds(-1, 1)
    assert cos(AccumBounds(-oo, 0)) == AccumBounds(-1, 1)
    assert cos(AccumBounds(0, 2*S.Pi)) == AccumBounds(-1, 1)
    assert cos(AccumBounds(-S.Pi/3, S.Pi/4)) == AccumBounds(cos(-S.Pi/3), 1)
    assert cos(AccumBounds(S.Pi*Rational(3, 4), S.Pi*Rational(5, 4))) == AccumBounds(-1, cos(S.Pi*Rational(3, 4)))
    assert cos(AccumBounds(S.Pi*Rational(5, 4), S.Pi*Rational(4, 3))) == AccumBounds(cos(S.Pi*Rational(5, 4)), cos(S.Pi*Rational(4, 3)))
    assert cos(AccumBounds(S.Pi/4, S.Pi/3)) == AccumBounds(cos(S.Pi/3), cos(S.Pi/4))


def test_cos_fdiff():
    assert cos(x).fdiff() == -sin(x)
    raises(ArgumentIndexError, lambda: cos(x).fdiff(2))


def test_tan():
    assert tan(nan) is nan

    assert tan(zoo) is nan
    assert tan(oo) == AccumBounds(-oo, oo)
    assert tan(oo) - tan(oo) == AccumBounds(-oo, oo)
    assert tan.nargs == FiniteSet(1)
    assert tan(oo*I) == I
    assert tan(-oo*I) == -I

    assert tan(0) == 0

    assert tan(atan(x)) == x
    assert tan(asin(x)) == x / sqrt(1 - x**2)
    assert tan(acos(x)) == sqrt(1 - x**2) / x
    assert tan(acot(x)) == 1 / x
    assert tan(acsc(x)) == 1 / (sqrt(1 - 1 / x**2) * x)
    assert tan(asec(x)) == sqrt(1 - 1 / x**2) * x
    assert tan(atan2(y, x)) == y/x

    assert tan(pi*I) == tanh(pi)*I
    assert tan(-pi*I) == -tanh(pi)*I
    assert tan(-2*I) == -tanh(2)*I

    assert tan(pi) == 0
    assert tan(-pi) == 0
    assert tan(2*pi) == 0
    assert tan(-2*pi) == 0
    assert tan(-3*10**73*pi) == 0

    assert tan(pi/2) is zoo
    assert tan(pi*Rational(3, 2)) is zoo

    assert tan(pi/3) == sqrt(3)
    assert tan(pi*Rational(-2, 3)) == sqrt(3)

    assert tan(pi/4) is S.One
    assert tan(-pi/4) is S.NegativeOne
    assert tan(pi*Rational(17, 4)) is S.One
    assert tan(pi*Rational(-3, 4)) is S.One

    assert tan(pi/5) == sqrt(5 - 2*sqrt(5))
    assert tan(pi*Rational(2, 5)) == sqrt(5 + 2*sqrt(5))
    assert tan(pi*Rational(18, 5)) == -sqrt(5 + 2*sqrt(5))
    assert tan(pi*Rational(-16, 5)) == -sqrt(5 - 2*sqrt(5))

    assert tan(pi/6) == 1/sqrt(3)
    assert tan(-pi/6) == -1/sqrt(3)
    assert tan(pi*Rational(7, 6)) == 1/sqrt(3)
    assert tan(pi*Rational(-5, 6)) == 1/sqrt(3)

    assert tan(pi/8) == -1 + sqrt(2)
    assert tan(pi*Rational(3, 8)) == 1 + sqrt(2)  # issue 15959
    assert tan(pi*Rational(5, 8)) == -1 - sqrt(2)
    assert tan(pi*Rational(7, 8)) == 1 - sqrt(2)

    assert tan(pi/10) == sqrt(1 - 2*sqrt(5)/5)
    assert tan(pi*Rational(3, 10)) == sqrt(1 + 2*sqrt(5)/5)
    assert tan(pi*Rational(17, 10)) == -sqrt(1 + 2*sqrt(5)/5)
    assert tan(pi*Rational(-31, 10)) == -sqrt(1 - 2*sqrt(5)/5)

    assert tan(pi/12) == -sqrt(3) + 2
    assert tan(pi*Rational(5, 12)) == sqrt(3) + 2
    assert tan(pi*Rational(7, 12)) == -sqrt(3) - 2
    assert tan(pi*Rational(11, 12)) == sqrt(3) - 2

    assert tan(pi/24).radsimp() == -2 - sqrt(3) + sqrt(2) + sqrt(6)
    assert tan(pi*Rational(5, 24)).radsimp() == -2 + sqrt(3) - sqrt(2) + sqrt(6)
    assert tan(pi*Rational(7, 24)).radsimp() == 2 - sqrt(3) - sqrt(2) + sqrt(6)
    assert tan(pi*Rational(11, 24)).radsimp() == 2 + sqrt(3) + sqrt(2) + sqrt(6)
    assert tan(pi*Rational(13, 24)).radsimp() == -2 - sqrt(3) - sqrt(2) - sqrt(6)
    assert tan(pi*Rational(17, 24)).radsimp() == -2 + sqrt(3) + sqrt(2) - sqrt(6)
    assert tan(pi*Rational(19, 24)).radsimp() == 2 - sqrt(3) + sqrt(2) - sqrt(6)
    assert tan(pi*Rational(23, 24)).radsimp() == 2 + sqrt(3) - sqrt(2) - sqrt(6)

    assert tan(x*I) == tanh(x)*I

    assert tan(k*pi) == 0
    assert tan(17*k*pi) == 0

    assert tan(k*pi*I) == tanh(k*pi)*I

    assert tan(r).is_real is None
    assert tan(r).is_extended_real is True

    assert tan(0, evaluate=False).is_algebraic
    assert tan(a).is_algebraic is None
    assert tan(na).is_algebraic is False

    assert tan(pi*Rational(10, 7)) == tan(pi*Rational(3, 7))
    assert tan(pi*Rational(11, 7)) == -tan(pi*Rational(3, 7))
    assert tan(pi*Rational(-11, 7)) == tan(pi*Rational(3, 7))

    assert tan(pi*Rational(15, 14)) == tan(pi/14)
    assert tan(pi*Rational(-15, 14)) == -tan(pi/14)

    assert tan(r).is_finite is None
    assert tan(I*r).is_finite is True

    # https://github.com/sympy/sympy/issues/21177
    f = tan(pi*(x + S(3)/2))/(3*x)
    assert f.as_leading_term(x) == -1/(3*pi*x**2)


def test_tan_series():
    assert tan(x).series(x, 0, 9) == \
        x + x**3/3 + 2*x**5/15 + 17*x**7/315 + O(x**9)


def test_tan_rewrite():
    neg_exp, pos_exp = exp(-x*I), exp(x*I)
    assert tan(x).rewrite(exp) == I*(neg_exp - pos_exp)/(neg_exp + pos_exp)
    assert tan(x).rewrite(sin) == 2*sin(x)**2/sin(2*x)
    assert tan(x).rewrite(cos) == cos(x - S.Pi/2, evaluate=False)/cos(x)
    assert tan(x).rewrite(cot) == 1/cot(x)
    assert tan(sinh(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, sinh(3)).n()
    assert tan(cosh(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, cosh(3)).n()
    assert tan(tanh(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, tanh(3)).n()
    assert tan(coth(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, coth(3)).n()
    assert tan(sin(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, sin(3)).n()
    assert tan(cos(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, cos(3)).n()
    assert tan(tan(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, tan(3)).n()
    assert tan(cot(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, cot(3)).n()
    assert tan(log(x)).rewrite(Pow) == I*(x**-I - x**I)/(x**-I + x**I)
    assert tan(x).rewrite(sec) == sec(x)/sec(x - pi/2, evaluate=False)
    assert tan(x).rewrite(csc) == csc(-x + pi/2, evaluate=False)/csc(x)
    assert tan(sin(x)).rewrite(Pow) == tan(sin(x))
    assert tan(pi*Rational(2, 5), evaluate=False).rewrite(sqrt) == sqrt(sqrt(5)/8 +
               Rational(5, 8))/(Rational(-1, 4) + sqrt(5)/4)
    assert tan(x).rewrite(besselj) == besselj(S.Half, x)/besselj(-S.Half, x)
    assert tan(x).rewrite(besselj).subs(x, 0) == tan(0)


@slow
def test_tan_rewrite_slow():
    assert 0 == (cos(pi/34)*tan(pi/34) - sin(pi/34)).rewrite(pow)
    assert 0 == (cos(pi/17)*tan(pi/17) - sin(pi/17)).rewrite(pow)
    assert tan(pi/19).rewrite(pow) == tan(pi/19)
    assert tan(pi*Rational(8, 19)).rewrite(sqrt) == tan(pi*Rational(8, 19))
    assert tan(pi*Rational(2, 5), evaluate=False).rewrite(sqrt) == sqrt(sqrt(5)/8 +
               Rational(5, 8))/(Rational(-1, 4) + sqrt(5)/4)


def test_tan_subs():
    assert tan(x).subs(tan(x), y) == y
    assert tan(x).subs(x, y) == tan(y)
    assert tan(x).subs(x, S.Pi/2) is zoo
    assert tan(x).subs(x, S.Pi*Rational(3, 2)) is zoo


def test_tan_expansion():
    assert tan(x + y).expand(trig=True) == ((tan(x) + tan(y))/(1 - tan(x)*tan(y))).expand()
    assert tan(x - y).expand(trig=True) == ((tan(x) - tan(y))/(1 + tan(x)*tan(y))).expand()
    assert tan(x + y + z).expand(trig=True) == (
        (tan(x) + tan(y) + tan(z) - tan(x)*tan(y)*tan(z))/
        (1 - tan(x)*tan(y) - tan(x)*tan(z) - tan(y)*tan(z))).expand()
    assert 0 == tan(2*x).expand(trig=True).rewrite(tan).subs([(tan(x), Rational(1, 7))])*24 - 7
    assert 0 == tan(3*x).expand(trig=True).rewrite(tan).subs([(tan(x), Rational(1, 5))])*55 - 37
    assert 0 == tan(4*x - pi/4).expand(trig=True).rewrite(tan).subs([(tan(x), Rational(1, 5))])*239 - 1
    _test_extrig(tan, 2, 2*tan(1)/(1 - tan(1)**2))
    _test_extrig(tan, 3, (-tan(1)**3 + 3*tan(1))/(1 - 3*tan(1)**2))


def test_tan_AccumBounds():
    assert tan(AccumBounds(-oo, oo)) == AccumBounds(-oo, oo)
    assert tan(AccumBounds(S.Pi/3, S.Pi*Rational(2, 3))) == AccumBounds(-oo, oo)
    assert tan(AccumBounds(S.Pi/6, S.Pi/3)) == AccumBounds(tan(S.Pi/6), tan(S.Pi/3))


def test_tan_fdiff():
    assert tan(x).fdiff() == tan(x)**2 + 1
    raises(ArgumentIndexError, lambda: tan(x).fdiff(2))


def test_cot():
    assert cot(nan) is nan

    assert cot.nargs == FiniteSet(1)
    assert cot(oo*I) == -I
    assert cot(-oo*I) == I
    assert cot(zoo) is nan

    assert cot(0) is zoo
    assert cot(2*pi) is zoo

    assert cot(acot(x)) == x
    assert cot(atan(x)) == 1 / x
    assert cot(asin(x)) == sqrt(1 - x**2) / x
    assert cot(acos(x)) == x / sqrt(1 - x**2)
    assert cot(acsc(x)) == sqrt(1 - 1 / x**2) * x
    assert cot(asec(x)) == 1 / (sqrt(1 - 1 / x**2) * x)
    assert cot(atan2(y, x)) == x/y

    assert cot(pi*I) == -coth(pi)*I
    assert cot(-pi*I) == coth(pi)*I
    assert cot(-2*I) == coth(2)*I

    assert cot(pi) == cot(2*pi) == cot(3*pi)
    assert cot(-pi) == cot(-2*pi) == cot(-3*pi)

    assert cot(pi/2) == 0
    assert cot(-pi/2) == 0
    assert cot(pi*Rational(5, 2)) == 0
    assert cot(pi*Rational(7, 2)) == 0

    assert cot(pi/3) == 1/sqrt(3)
    assert cot(pi*Rational(-2, 3)) == 1/sqrt(3)

    assert cot(pi/4) is S.One
    assert cot(-pi/4) is S.NegativeOne
    assert cot(pi*Rational(17, 4)) is S.One
    assert cot(pi*Rational(-3, 4)) is S.One

    assert cot(pi/6) == sqrt(3)
    assert cot(-pi/6) == -sqrt(3)
    assert cot(pi*Rational(7, 6)) == sqrt(3)
    assert cot(pi*Rational(-5, 6)) == sqrt(3)

    assert cot(pi/8) == 1 + sqrt(2)
    assert cot(pi*Rational(3, 8)) == -1 + sqrt(2)
    assert cot(pi*Rational(5, 8)) == 1 - sqrt(2)
    assert cot(pi*Rational(7, 8)) == -1 - sqrt(2)

    assert cot(pi/12) == sqrt(3) + 2
    assert cot(pi*Rational(5, 12)) == -sqrt(3) + 2
    assert cot(pi*Rational(7, 12)) == sqrt(3) - 2
    assert cot(pi*Rational(11, 12)) == -sqrt(3) - 2

    assert cot(pi/24).radsimp() == sqrt(2) + sqrt(3) + 2 + sqrt(6)
    assert cot(pi*Rational(5, 24)).radsimp() == -sqrt(2) - sqrt(3) + 2 + sqrt(6)
    assert cot(pi*Rational(7, 24)).radsimp() == -sqrt(2) + sqrt(3) - 2 + sqrt(6)
    assert cot(pi*Rational(11, 24)).radsimp() == sqrt(2) - sqrt(3) - 2 + sqrt(6)
    assert cot(pi*Rational(13, 24)).radsimp() == -sqrt(2) + sqrt(3) + 2 - sqrt(6)
    assert cot(pi*Rational(17, 24)).radsimp() == sqrt(2) - sqrt(3) + 2 - sqrt(6)
    assert cot(pi*Rational(19, 24)).radsimp() == sqrt(2) + sqrt(3) - 2 - sqrt(6)
    assert cot(pi*Rational(23, 24)).radsimp() == -sqrt(2) - sqrt(3) - 2 - sqrt(6)

    assert cot(x*I) == -coth(x)*I
    assert cot(k*pi*I) == -coth(k*pi)*I

    assert cot(r).is_real is None
    assert cot(r).is_extended_real is True

    assert cot(a).is_algebraic is None
    assert cot(na).is_algebraic is False

    assert cot(pi*Rational(10, 7)) == cot(pi*Rational(3, 7))
    assert cot(pi*Rational(11, 7)) == -cot(pi*Rational(3, 7))
    assert cot(pi*Rational(-11, 7)) == cot(pi*Rational(3, 7))

    assert cot(pi*Rational(39, 34)) == cot(pi*Rational(5, 34))
    assert cot(pi*Rational(-41, 34)) == -cot(pi*Rational(7, 34))

    assert cot(x).is_finite is None
    assert cot(r).is_finite is None
    i = Symbol('i', imaginary=True)
    assert cot(i).is_finite is True

    assert cot(x).subs(x, 3*pi) is zoo

    # https://github.com/sympy/sympy/issues/21177
    f = cot(pi*(x + 4))/(3*x)
    assert f.as_leading_term(x) == 1/(3*pi*x**2)


def test_tan_cot_sin_cos_evalf():
    assert abs((tan(pi*Rational(8, 15))*cos(pi*Rational(8, 15))/sin(pi*Rational(8, 15)) - 1).evalf()) < 1e-14
    assert abs((cot(pi*Rational(4, 15))*sin(pi*Rational(4, 15))/cos(pi*Rational(4, 15)) - 1).evalf()) < 1e-14

@XFAIL
def test_tan_cot_sin_cos_ratsimp():
    assert 1 == (tan(pi*Rational(8, 15))*cos(pi*Rational(8, 15))/sin(pi*Rational(8, 15))).ratsimp()
    assert 1 == (cot(pi*Rational(4, 15))*sin(pi*Rational(4, 15))/cos(pi*Rational(4, 15))).ratsimp()


def test_cot_series():
    assert cot(x).series(x, 0, 9) == \
        1/x - x/3 - x**3/45 - 2*x**5/945 - x**7/4725 + O(x**9)
    # issue 6210
    assert cot(x**4 + x**5).series(x, 0, 1) == \
        x**(-4) - 1/x**3 + x**(-2) - 1/x + 1 + O(x)
    assert cot(pi*(1-x)).series(x, 0, 3) == -1/(pi*x) + pi*x/3 + O(x**3)
    assert cot(x).taylor_term(0, x) == 1/x
    assert cot(x).taylor_term(2, x) is S.Zero
    assert cot(x).taylor_term(3, x) == -x**3/45


def test_cot_rewrite():
    neg_exp, pos_exp = exp(-x*I), exp(x*I)
    assert cot(x).rewrite(exp) == I*(pos_exp + neg_exp)/(pos_exp - neg_exp)
    assert cot(x).rewrite(sin) == sin(2*x)/(2*(sin(x)**2))
    assert cot(x).rewrite(cos) == cos(x)/cos(x - pi/2, evaluate=False)
    assert cot(x).rewrite(tan) == 1/tan(x)
    def check(func):
        z = cot(func(x)).rewrite(exp) - cot(x).rewrite(exp).subs(x, func(x))
        assert z.rewrite(exp).expand() == 0
    check(sinh)
    check(cosh)
    check(tanh)
    check(coth)
    check(sin)
    check(cos)
    check(tan)
    assert cot(log(x)).rewrite(Pow) == -I*(x**-I + x**I)/(x**-I - x**I)
    assert cot(x).rewrite(sec) == sec(x - pi / 2, evaluate=False) / sec(x)
    assert cot(x).rewrite(csc) == csc(x) / csc(- x + pi / 2, evaluate=False)
    assert cot(sin(x)).rewrite(Pow) == cot(sin(x))
    assert cot(pi*Rational(2, 5), evaluate=False).rewrite(sqrt) == (Rational(-1, 4) + sqrt(5)/4)/\
                                                        sqrt(sqrt(5)/8 + Rational(5, 8))
    assert cot(x).rewrite(besselj) == besselj(-S.Half, x)/besselj(S.Half, x)
    assert cot(x).rewrite(besselj).subs(x, 0) == cot(0)


@slow
def test_cot_rewrite_slow():
    assert cot(pi*Rational(4, 34)).rewrite(pow).ratsimp() == \
        (cos(pi*Rational(4, 34))/sin(pi*Rational(4, 34))).rewrite(pow).ratsimp()
    assert cot(pi*Rational(4, 17)).rewrite(pow) == \
        (cos(pi*Rational(4, 17))/sin(pi*Rational(4, 17))).rewrite(pow)
    assert cot(pi/19).rewrite(pow) == cot(pi/19)
    assert cot(pi/19).rewrite(sqrt) == cot(pi/19)
    assert cot(pi*Rational(2, 5), evaluate=False).rewrite(sqrt) == \
        (Rational(-1, 4) + sqrt(5)/4) / sqrt(sqrt(5)/8 + Rational(5, 8))


def test_cot_subs():
    assert cot(x).subs(cot(x), y) == y
    assert cot(x).subs(x, y) == cot(y)
    assert cot(x).subs(x, 0) is zoo
    assert cot(x).subs(x, S.Pi) is zoo


def test_cot_expansion():
    assert cot(x + y).expand(trig=True).together() == (
        (cot(x)*cot(y) - 1)/(cot(x) + cot(y)))
    assert cot(x - y).expand(trig=True).together() == (
        cot(x)*cot(-y) - 1)/(cot(x) + cot(-y))
    assert cot(x + y + z).expand(trig=True).together() == (
        (cot(x)*cot(y)*cot(z) - cot(x) - cot(y) - cot(z))/
        (-1 + cot(x)*cot(y) + cot(x)*cot(z) + cot(y)*cot(z)))
    assert cot(3*x).expand(trig=True).together() == (
        (cot(x)**2 - 3)*cot(x)/(3*cot(x)**2 - 1))
    assert cot(2*x).expand(trig=True) == cot(x)/2 - 1/(2*cot(x))
    assert cot(3*x).expand(trig=True).together() == (
        cot(x)**2 - 3)*cot(x)/(3*cot(x)**2 - 1)
    assert cot(4*x - pi/4).expand(trig=True).cancel() == (
        -tan(x)**4 + 4*tan(x)**3 + 6*tan(x)**2 - 4*tan(x) - 1
        )/(tan(x)**4 + 4*tan(x)**3 - 6*tan(x)**2 - 4*tan(x) + 1)
    _test_extrig(cot, 2, (-1 + cot(1)**2)/(2*cot(1)))
    _test_extrig(cot, 3, (-3*cot(1) + cot(1)**3)/(-1 + 3*cot(1)**2))


def test_cot_AccumBounds():
    assert cot(AccumBounds(-oo, oo)) == AccumBounds(-oo, oo)
    assert cot(AccumBounds(-S.Pi/3, S.Pi/3)) == AccumBounds(-oo, oo)
    assert cot(AccumBounds(S.Pi/6, S.Pi/3)) == AccumBounds(cot(S.Pi/3), cot(S.Pi/6))


def test_cot_fdiff():
    assert cot(x).fdiff() == -cot(x)**2 - 1
    raises(ArgumentIndexError, lambda: cot(x).fdiff(2))


def test_sinc():
    assert isinstance(sinc(x), sinc)

    s = Symbol('s', zero=True)
    assert sinc(s) is S.One
    assert sinc(S.Infinity) is S.Zero
    assert sinc(S.NegativeInfinity) is S.Zero
    assert sinc(S.NaN) is S.NaN
    assert sinc(S.ComplexInfinity) is S.NaN

    n = Symbol('n', integer=True, nonzero=True)
    assert sinc(n*pi) is S.Zero
    assert sinc(-n*pi) is S.Zero
    assert sinc(pi/2) == 2 / pi
    assert sinc(-pi/2) == 2 / pi
    assert sinc(pi*Rational(5, 2)) == 2 / (5*pi)
    assert sinc(pi*Rational(7, 2)) == -2 / (7*pi)

    assert sinc(-x) == sinc(x)

    assert sinc(x).diff(x) == cos(x)/x - sin(x)/x**2
    assert sinc(x).diff(x) == (sin(x)/x).diff(x)
    assert sinc(x).diff(x, x) == (-sin(x) - 2*cos(x)/x + 2*sin(x)/x**2)/x
    assert sinc(x).diff(x, x) == (sin(x)/x).diff(x, x)
    assert limit(sinc(x).diff(x), x, 0) == 0
    assert limit(sinc(x).diff(x, x), x, 0) == -S(1)/3

    # https://github.com/sympy/sympy/issues/11402
    #
    # assert sinc(x).diff(x) == Piecewise(((x*cos(x) - sin(x)) / x**2, Ne(x, 0)), (0, True))
    #
    # assert sinc(x).diff(x).equals(sinc(x).rewrite(sin).diff(x))
    #
    # assert sinc(x).diff(x).subs(x, 0) is S.Zero

    assert sinc(x).series() == 1 - x**2/6 + x**4/120 + O(x**6)

    assert sinc(x).rewrite(jn) == jn(0, x)
    assert sinc(x).rewrite(sin) == Piecewise((sin(x)/x, Ne(x, 0)), (1, True))
    assert sinc(pi, evaluate=False).is_zero is True
    assert sinc(0, evaluate=False).is_zero is False
    assert sinc(n*pi, evaluate=False).is_zero is True
    assert sinc(x).is_zero is None
    xr = Symbol('xr', real=True, nonzero=True)
    assert sinc(x).is_real is None
    assert sinc(xr).is_real is True
    assert sinc(I*xr).is_real is True
    assert sinc(I*100).is_real is True
    assert sinc(x).is_finite is None
    assert sinc(xr).is_finite is True


def test_asin():
    assert asin(nan) is nan

    assert asin.nargs == FiniteSet(1)
    assert asin(oo) == -I*oo
    assert asin(-oo) == I*oo
    assert asin(zoo) is zoo

    # Note: asin(-x) = - asin(x)
    assert asin(0) == 0
    assert asin(1) == pi/2
    assert asin(-1) == -pi/2
    assert asin(sqrt(3)/2) == pi/3
    assert asin(-sqrt(3)/2) == -pi/3
    assert asin(sqrt(2)/2) == pi/4
    assert asin(-sqrt(2)/2) == -pi/4
    assert asin(sqrt((5 - sqrt(5))/8)) == pi/5
    assert asin(-sqrt((5 - sqrt(5))/8)) == -pi/5
    assert asin(S.Half) == pi/6
    assert asin(Rational(-1, 2)) == -pi/6
    assert asin((sqrt(2 - sqrt(2)))/2) == pi/8
    assert asin(-(sqrt(2 - sqrt(2)))/2) == -pi/8
    assert asin((sqrt(5) - 1)/4) == pi/10
    assert asin(-(sqrt(5) - 1)/4) == -pi/10
    assert asin((sqrt(3) - 1)/sqrt(2**3)) == pi/12
    assert asin(-(sqrt(3) - 1)/sqrt(2**3)) == -pi/12

    # check round-trip for exact values:
    for d in [5, 6, 8, 10, 12]:
        for n in range(-(d//2), d//2 + 1):
            if gcd(n, d) == 1:
                assert asin(sin(n*pi/d)) == n*pi/d

    assert asin(x).diff(x) == 1/sqrt(1 - x**2)

    assert asin(0.2, evaluate=False).is_real is True
    assert asin(-2).is_real is False
    assert asin(r).is_real is None

    assert asin(-2*I) == -I*asinh(2)

    assert asin(Rational(1, 7), evaluate=False).is_positive is True
    assert asin(Rational(-1, 7), evaluate=False).is_positive is False
    assert asin(p).is_positive is None
    assert asin(sin(Rational(7, 2))) == Rational(-7, 2) + pi
    assert asin(sin(Rational(-7, 4))) == Rational(7, 4) - pi
    assert unchanged(asin, cos(x))


def test_asin_series():
    assert asin(x).series(x, 0, 9) == \
        x + x**3/6 + 3*x**5/40 + 5*x**7/112 + O(x**9)
    t5 = asin(x).taylor_term(5, x)
    assert t5 == 3*x**5/40
    assert asin(x).taylor_term(7, x, t5, 0) == 5*x**7/112


def test_asin_leading_term():
    assert asin(x).as_leading_term(x) == x
    # Tests concerning branch points
    assert asin(x + 1).as_leading_term(x) == pi/2
    assert asin(x - 1).as_leading_term(x) == -pi/2
    assert asin(1/x).as_leading_term(x, cdir=1) == I*log(x) + pi/2 - I*log(2)
    assert asin(1/x).as_leading_term(x, cdir=-1) == -I*log(x) - 3*pi/2 + I*log(2)
    # Tests concerning points lying on branch cuts
    assert asin(I*x + 2).as_leading_term(x, cdir=1) == pi - asin(2)
    assert asin(-I*x + 2).as_leading_term(x, cdir=1) == asin(2)
    assert asin(I*x - 2).as_leading_term(x, cdir=1) == -asin(2)
    assert asin(-I*x - 2).as_leading_term(x, cdir=1) == -pi + asin(2)
    # Tests concerning im(ndir) == 0
    assert asin(-I*x**2 + x - 2).as_leading_term(x, cdir=1) == -pi/2 + I*log(2 - sqrt(3))
    assert asin(-I*x**2 + x - 2).as_leading_term(x, cdir=-1) == -pi/2 + I*log(2 - sqrt(3))


def test_asin_rewrite():
    assert asin(x).rewrite(log) == -I*log(I*x + sqrt(1 - x**2))
    assert asin(x).rewrite(atan) == 2*atan(x/(1 + sqrt(1 - x**2)))
    assert asin(x).rewrite(acos) == S.Pi/2 - acos(x)
    assert asin(x).rewrite(acot) == 2*acot((sqrt(-x**2 + 1) + 1)/x)
    assert asin(x).rewrite(asec) == -asec(1/x) + pi/2
    assert asin(x).rewrite(acsc) == acsc(1/x)


def test_asin_fdiff():
    assert asin(x).fdiff() == 1/sqrt(1 - x**2)
    raises(ArgumentIndexError, lambda: asin(x).fdiff(2))


def test_acos():
    assert acos(nan) is nan
    assert acos(zoo) is zoo

    assert acos.nargs == FiniteSet(1)
    assert acos(oo) == I*oo
    assert acos(-oo) == -I*oo

    # Note: acos(-x) = pi - acos(x)
    assert acos(0) == pi/2
    assert acos(S.Half) == pi/3
    assert acos(Rational(-1, 2)) == pi*Rational(2, 3)
    assert acos(1) == 0
    assert acos(-1) == pi
    assert acos(sqrt(2)/2) == pi/4
    assert acos(-sqrt(2)/2) == pi*Rational(3, 4)

    # check round-trip for exact values:
    for d in range(5, 13):
        for num in range(d):
            if gcd(num, d) == 1:
                assert acos(cos(num*pi/d)) == num*pi/d
                assert acos(-cos(num*pi/d)) == pi - num*pi/d
                assert acos(sin(num*pi/d)) == pi/2 - asin(sin(num*pi/d))
                assert acos(-sin(num*pi/d)) == pi/2 - asin(-sin(num*pi/d))

    assert acos(2*I) == pi/2 - asin(2*I)

    assert acos(x).diff(x) == -1/sqrt(1 - x**2)

    assert acos(0.2).is_real is True
    assert acos(-2).is_real is False
    assert acos(r).is_real is None

    assert acos(Rational(1, 7), evaluate=False).is_positive is True
    assert acos(Rational(-1, 7), evaluate=False).is_positive is True
    assert acos(Rational(3, 2), evaluate=False).is_positive is False
    assert acos(p).is_positive is None

    assert acos(2 + p).conjugate() != acos(10 + p)
    assert acos(-3 + n).conjugate() != acos(-3 + n)
    assert acos(Rational(1, 3)).conjugate() == acos(Rational(1, 3))
    assert acos(Rational(-1, 3)).conjugate() == acos(Rational(-1, 3))
    assert acos(p + n*I).conjugate() == acos(p - n*I)
    assert acos(z).conjugate() != acos(conjugate(z))


def test_acos_leading_term():
    assert acos(x).as_leading_term(x) == pi/2
    # Tests concerning branch points
    assert acos(x + 1).as_leading_term(x) == sqrt(2)*sqrt(-x)
    assert acos(x - 1).as_leading_term(x) == pi
    assert acos(1/x).as_leading_term(x, cdir=1) == -I*log(x) + I*log(2)
    assert acos(1/x).as_leading_term(x, cdir=-1) == I*log(x) + 2*pi - I*log(2)
    # Tests concerning points lying on branch cuts
    assert acos(I*x + 2).as_leading_term(x, cdir=1) == -acos(2)
    assert acos(-I*x + 2).as_leading_term(x, cdir=1) == acos(2)
    assert acos(I*x - 2).as_leading_term(x, cdir=1) == acos(-2)
    assert acos(-I*x - 2).as_leading_term(x, cdir=1) == 2*pi - acos(-2)
    # Tests concerning im(ndir) == 0
    assert acos(-I*x**2 + x - 2).as_leading_term(x, cdir=1) == pi + I*log(sqrt(3) + 2)
    assert acos(-I*x**2 + x - 2).as_leading_term(x, cdir=-1) == pi + I*log(sqrt(3) + 2)


def test_acos_series():
    assert acos(x).series(x, 0, 8) == \
        pi/2 - x - x**3/6 - 3*x**5/40 - 5*x**7/112 + O(x**8)
    assert acos(x).series(x, 0, 8) == pi/2 - asin(x).series(x, 0, 8)
    t5 = acos(x).taylor_term(5, x)
    assert t5 == -3*x**5/40
    assert acos(x).taylor_term(7, x, t5, 0) == -5*x**7/112
    assert acos(x).taylor_term(0, x) == pi/2
    assert acos(x).taylor_term(2, x) is S.Zero


def test_acos_rewrite():
    assert acos(x).rewrite(log) == pi/2 + I*log(I*x + sqrt(1 - x**2))
    assert acos(x).rewrite(atan) == pi*(-x*sqrt(x**(-2)) + 1)/2 + atan(sqrt(1 - x**2)/x)
    assert acos(0).rewrite(atan) == S.Pi/2
    assert acos(0.5).rewrite(atan) == acos(0.5).rewrite(log)
    assert acos(x).rewrite(asin) == S.Pi/2 - asin(x)
    assert acos(x).rewrite(acot) == -2*acot((sqrt(-x**2 + 1) + 1)/x) + pi/2
    assert acos(x).rewrite(asec) == asec(1/x)
    assert acos(x).rewrite(acsc) == -acsc(1/x) + pi/2


def test_acos_fdiff():
    assert acos(x).fdiff() == -1/sqrt(1 - x**2)
    raises(ArgumentIndexError, lambda: acos(x).fdiff(2))


def test_atan():
    assert atan(nan) is nan

    assert atan.nargs == FiniteSet(1)
    assert atan(oo) == pi/2
    assert atan(-oo) == -pi/2
    assert atan(zoo) == AccumBounds(-pi/2, pi/2)

    assert atan(0) == 0
    assert atan(1) == pi/4
    assert atan(sqrt(3)) == pi/3
    assert atan(-(1 + sqrt(2))) == pi*Rational(-3, 8)
    assert atan(sqrt(5 - 2 * sqrt(5))) == pi/5
    assert atan(-sqrt(1 - 2 * sqrt(5)/ 5)) == -pi/10
    assert atan(sqrt(1 + 2 * sqrt(5) / 5)) == pi*Rational(3, 10)
    assert atan(-2 + sqrt(3)) == -pi/12
    assert atan(2 + sqrt(3)) == pi*Rational(5, 12)
    assert atan(-2 - sqrt(3)) == pi*Rational(-5, 12)

    # check round-trip for exact values:
    for d in [5, 6, 8, 10, 12]:
        for num in range(-(d//2), d//2 + 1):
            if gcd(num, d) == 1:
                assert atan(tan(num*pi/d)) == num*pi/d

    assert atan(oo) == pi/2
    assert atan(x).diff(x) == 1/(1 + x**2)

    assert atan(r).is_real is True

    assert atan(-2*I) == -I*atanh(2)
    assert unchanged(atan, cot(x))
    assert atan(cot(Rational(1, 4))) == Rational(-1, 4) + pi/2
    assert acot(Rational(1, 4)).is_rational is False

    for s in (x, p, n, np, nn, nz, ep, en, enp, enn, enz):
        if s.is_real or s.is_extended_real is None:
            assert s.is_nonzero is atan(s).is_nonzero
            assert s.is_positive is atan(s).is_positive
            assert s.is_negative is atan(s).is_negative
            assert s.is_nonpositive is atan(s).is_nonpositive
            assert s.is_nonnegative is atan(s).is_nonnegative
        else:
            assert s.is_extended_nonzero is atan(s).is_nonzero
            assert s.is_extended_positive is atan(s).is_positive
            assert s.is_extended_negative is atan(s).is_negative
            assert s.is_extended_nonpositive is atan(s).is_nonpositive
            assert s.is_extended_nonnegative is atan(s).is_nonnegative
        assert s.is_extended_nonzero is atan(s).is_extended_nonzero
        assert s.is_extended_positive is atan(s).is_extended_positive
        assert s.is_extended_negative is atan(s).is_extended_negative
        assert s.is_extended_nonpositive is atan(s).is_extended_nonpositive
        assert s.is_extended_nonnegative is atan(s).is_extended_nonnegative


def test_atan_rewrite():
    assert atan(x).rewrite(log) == I*(log(1 - I*x)-log(1 + I*x))/2
    assert atan(x).rewrite(asin) == (-asin(1/sqrt(x**2 + 1)) + pi/2)*sqrt(x**2)/x
    assert atan(x).rewrite(acos) == sqrt(x**2)*acos(1/sqrt(x**2 + 1))/x
    assert atan(x).rewrite(acot) == acot(1/x)
    assert atan(x).rewrite(asec) == sqrt(x**2)*asec(sqrt(x**2 + 1))/x
    assert atan(x).rewrite(acsc) == (-acsc(sqrt(x**2 + 1)) + pi/2)*sqrt(x**2)/x

    assert atan(-5*I).evalf() == atan(x).rewrite(log).evalf(subs={x:-5*I})
    assert atan(5*I).evalf() == atan(x).rewrite(log).evalf(subs={x:5*I})


def test_atan_fdiff():
    assert atan(x).fdiff() == 1/(x**2 + 1)
    raises(ArgumentIndexError, lambda: atan(x).fdiff(2))


def test_atan_leading_term():
    assert atan(x).as_leading_term(x) == x
    assert atan(1/x).as_leading_term(x, cdir=1) == pi/2
    assert atan(1/x).as_leading_term(x, cdir=-1) == -pi/2
    # Tests concerning branch points
    assert atan(x + I).as_leading_term(x, cdir=1) == -I*log(x)/2 + pi/4 + I*log(2)/2
    assert atan(x + I).as_leading_term(x, cdir=-1) == -I*log(x)/2 - 3*pi/4 + I*log(2)/2
    assert atan(x - I).as_leading_term(x, cdir=1) == I*log(x)/2 + pi/4 - I*log(2)/2
    assert atan(x - I).as_leading_term(x, cdir=-1) == I*log(x)/2 + pi/4 - I*log(2)/2
    # Tests concerning points lying on branch cuts
    assert atan(x + 2*I).as_leading_term(x, cdir=1) == I*atanh(2)
    assert atan(x + 2*I).as_leading_term(x, cdir=-1) == -pi + I*atanh(2)
    assert atan(x - 2*I).as_leading_term(x, cdir=1) == pi - I*atanh(2)
    assert atan(x - 2*I).as_leading_term(x, cdir=-1) == -I*atanh(2)
    # Tests concerning re(ndir) == 0
    assert atan(2*I - I*x - x**2).as_leading_term(x, cdir=1) == -pi/2 + I*log(3)/2
    assert atan(2*I - I*x - x**2).as_leading_term(x, cdir=-1) == -pi/2 + I*log(3)/2


def test_atan2():
    assert atan2.nargs == FiniteSet(2)
    assert atan2(0, 0) is S.NaN
    assert atan2(0, 1) == 0
    assert atan2(1, 1) == pi/4
    assert atan2(1, 0) == pi/2
    assert atan2(1, -1) == pi*Rational(3, 4)
    assert atan2(0, -1) == pi
    assert atan2(-1, -1) == pi*Rational(-3, 4)
    assert atan2(-1, 0) == -pi/2
    assert atan2(-1, 1) == -pi/4
    i = symbols('i', imaginary=True)
    r = symbols('r', real=True)
    eq = atan2(r, i)
    ans = -I*log((i + I*r)/sqrt(i**2 + r**2))
    reps = ((r, 2), (i, I))
    assert eq.subs(reps) == ans.subs(reps)

    x = Symbol('x', negative=True)
    y = Symbol('y', negative=True)
    assert atan2(y, x) == atan(y/x) - pi
    y = Symbol('y', nonnegative=True)
    assert atan2(y, x) == atan(y/x) + pi
    y = Symbol('y')
    assert atan2(y, x) == atan2(y, x, evaluate=False)

    u = Symbol("u", positive=True)
    assert atan2(0, u) == 0
    u = Symbol("u", negative=True)
    assert atan2(0, u) == pi

    assert atan2(y, oo) ==  0
    assert atan2(y, -oo)==  2*pi*Heaviside(re(y), S.Half) - pi

    assert atan2(y, x).rewrite(log) == -I*log((x + I*y)/sqrt(x**2 + y**2))
    assert atan2(0, 0) is S.NaN

    ex = atan2(y, x) - arg(x + I*y)
    assert ex.subs({x:2, y:3}).rewrite(arg) == 0
    assert ex.subs({x:2, y:3*I}).rewrite(arg) == -pi - I*log(sqrt(5)*I/5)
    assert ex.subs({x:2*I, y:3}).rewrite(arg) == -pi/2 - I*log(sqrt(5)*I)
    assert ex.subs({x:2*I, y:3*I}).rewrite(arg) == -pi + atan(Rational(2, 3)) + atan(Rational(3, 2))
    i = symbols('i', imaginary=True)
    r = symbols('r', real=True)
    e = atan2(i, r)
    rewrite = e.rewrite(arg)
    reps = {i: I, r: -2}
    assert rewrite == -I*log(abs(I*i + r)/sqrt(abs(i**2 + r**2))) + arg((I*i + r)/sqrt(i**2 + r**2))
    assert (e - rewrite).subs(reps).equals(0)

    assert atan2(0, x).rewrite(atan) == Piecewise((pi, re(x) < 0),
                                            (0, Ne(x, 0)),
                                            (nan, True))
    assert atan2(0, r).rewrite(atan) == Piecewise((pi, r < 0), (0, Ne(r, 0)), (S.NaN, True))
    assert atan2(0, i),rewrite(atan) == 0
    assert atan2(0, r + i).rewrite(atan) == Piecewise((pi, r < 0), (0, True))

    assert atan2(y, x).rewrite(atan) == Piecewise(
            (2*atan(y/(x + sqrt(x**2 + y**2))), Ne(y, 0)),
            (pi, re(x) < 0),
            (0, (re(x) > 0) | Ne(im(x), 0)),
            (nan, True))
    assert conjugate(atan2(x, y)) == atan2(conjugate(x), conjugate(y))

    assert diff(atan2(y, x), x) == -y/(x**2 + y**2)
    assert diff(atan2(y, x), y) == x/(x**2 + y**2)

    assert simplify(diff(atan2(y, x).rewrite(log), x)) == -y/(x**2 + y**2)
    assert simplify(diff(atan2(y, x).rewrite(log), y)) ==  x/(x**2 + y**2)

    assert str(atan2(1, 2).evalf(5)) == '0.46365'
    raises(ArgumentIndexError, lambda: atan2(x, y).fdiff(3))

def test_issue_17461():
    class A(Symbol):
        is_extended_real = True

        def _eval_evalf(self, prec):
            return Float(5.0)

    x = A('X')
    y = A('Y')
    assert abs(atan2(x, y).evalf() - 0.785398163397448) <= 1e-10

def test_acot():
    assert acot(nan) is nan

    assert acot.nargs == FiniteSet(1)
    assert acot(-oo) == 0
    assert acot(oo) == 0
    assert acot(zoo) == 0
    assert acot(1) == pi/4
    assert acot(0) == pi/2
    assert acot(sqrt(3)/3) == pi/3
    assert acot(1/sqrt(3)) == pi/3
    assert acot(-1/sqrt(3)) == -pi/3
    assert acot(x).diff(x) == -1/(1 + x**2)

    assert acot(r).is_extended_real is True

    assert acot(I*pi) == -I*acoth(pi)
    assert acot(-2*I) == I*acoth(2)
    assert acot(x).is_positive is None
    assert acot(n).is_positive is False
    assert acot(p).is_positive is True
    assert acot(I).is_positive is False
    assert acot(Rational(1, 4)).is_rational is False
    assert unchanged(acot, cot(x))
    assert unchanged(acot, tan(x))
    assert acot(cot(Rational(1, 4))) == Rational(1, 4)
    assert acot(tan(Rational(-1, 4))) == Rational(1, 4) - pi/2


def test_acot_rewrite():
    assert acot(x).rewrite(log) == I*(log(1 - I/x)-log(1 + I/x))/2
    assert acot(x).rewrite(asin) == x*(-asin(sqrt(-x**2)/sqrt(-x**2 - 1)) + pi/2)*sqrt(x**(-2))
    assert acot(x).rewrite(acos) == x*sqrt(x**(-2))*acos(sqrt(-x**2)/sqrt(-x**2 - 1))
    assert acot(x).rewrite(atan) == atan(1/x)
    assert acot(x).rewrite(asec) == x*sqrt(x**(-2))*asec(sqrt((x**2 + 1)/x**2))
    assert acot(x).rewrite(acsc) == x*(-acsc(sqrt((x**2 + 1)/x**2)) + pi/2)*sqrt(x**(-2))

    assert acot(-I/5).evalf() == acot(x).rewrite(log).evalf(subs={x:-I/5})
    assert acot(I/5).evalf() == acot(x).rewrite(log).evalf(subs={x:I/5})


def test_acot_fdiff():
    assert acot(x).fdiff() == -1/(x**2 + 1)
    raises(ArgumentIndexError, lambda: acot(x).fdiff(2))

def test_acot_leading_term():
    assert acot(1/x).as_leading_term(x) == x
    # Tests concerning branch points
    assert acot(x + I).as_leading_term(x, cdir=1) == I*log(x)/2 + pi/4 - I*log(2)/2
    assert acot(x + I).as_leading_term(x, cdir=-1) == I*log(x)/2 + pi/4 - I*log(2)/2
    assert acot(x - I).as_leading_term(x, cdir=1) == -I*log(x)/2 + pi/4 + I*log(2)/2
    assert acot(x - I).as_leading_term(x, cdir=-1) == -I*log(x)/2 - 3*pi/4 + I*log(2)/2
    # Tests concerning points lying on branch cuts
    assert acot(x).as_leading_term(x, cdir=1) == pi/2
    assert acot(x).as_leading_term(x, cdir=-1) == -pi/2
    assert acot(x + I/2).as_leading_term(x, cdir=1) == pi - I*acoth(S(1)/2)
    assert acot(x + I/2).as_leading_term(x, cdir=-1) == -I*acoth(S(1)/2)
    assert acot(x - I/2).as_leading_term(x, cdir=1) == I*acoth(S(1)/2)
    assert acot(x - I/2).as_leading_term(x, cdir=-1) == -pi + I*acoth(S(1)/2)
    # Tests concerning re(ndir) == 0
    assert acot(I/2 - I*x - x**2).as_leading_term(x, cdir=1) == -pi/2 - I*log(3)/2
    assert acot(I/2 - I*x - x**2).as_leading_term(x, cdir=-1) == -pi/2 - I*log(3)/2


def test_attributes():
    assert sin(x).args == (x,)


def test_sincos_rewrite():
    assert sin(pi/2 - x) == cos(x)
    assert sin(pi - x) == sin(x)
    assert cos(pi/2 - x) == sin(x)
    assert cos(pi - x) == -cos(x)


def _check_even_rewrite(func, arg):
    """Checks that the expr has been rewritten using f(-x) -> f(x)
    arg : -x
    """
    return func(arg).args[0] == -arg


def _check_odd_rewrite(func, arg):
    """Checks that the expr has been rewritten using f(-x) -> -f(x)
    arg : -x
    """
    return func(arg).func.is_Mul


def _check_no_rewrite(func, arg):
    """Checks that the expr is not rewritten"""
    return func(arg).args[0] == arg


def test_evenodd_rewrite():
    a = cos(2)  # negative
    b = sin(1)  # positive
    even = [cos]
    odd = [sin, tan, cot, asin, atan, acot]
    with_minus = [-1, -2**1024 * E, -pi/105, -x*y, -x - y]
    for func in even:
        for expr in with_minus:
            assert _check_even_rewrite(func, expr)
        assert _check_no_rewrite(func, a*b)
        assert func(
            x - y) == func(y - x)  # it doesn't matter which form is canonical
    for func in odd:
        for expr in with_minus:
            assert _check_odd_rewrite(func, expr)
        assert _check_no_rewrite(func, a*b)
        assert func(
            x - y) == -func(y - x)  # it doesn't matter which form is canonical


def test_as_leading_term_issue_5272():
    assert sin(x).as_leading_term(x) == x
    assert cos(x).as_leading_term(x) == 1
    assert tan(x).as_leading_term(x) == x
    assert cot(x).as_leading_term(x) == 1/x


def test_leading_terms():
    assert sin(1/x).as_leading_term(x) == AccumBounds(-1, 1)
    assert sin(S.Half).as_leading_term(x) == sin(S.Half)
    assert cos(1/x).as_leading_term(x) == AccumBounds(-1, 1)
    assert cos(S.Half).as_leading_term(x) == cos(S.Half)
    assert sec(1/x).as_leading_term(x) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert csc(1/x).as_leading_term(x) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert tan(1/x).as_leading_term(x) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert cot(1/x).as_leading_term(x) == AccumBounds(S.NegativeInfinity, S.Infinity)

    # https://github.com/sympy/sympy/issues/21038
    f = sin(pi*(x + 4))/(3*x)
    assert f.as_leading_term(x) == pi/3


def test_atan2_expansion():
    assert cancel(atan2(x**2, x + 1).diff(x) - atan(x**2/(x + 1)).diff(x)) == 0
    assert cancel(atan(y/x).series(y, 0, 5) - atan2(y, x).series(y, 0, 5)
                  + atan2(0, x) - atan(0)) == O(y**5)
    assert cancel(atan(y/x).series(x, 1, 4) - atan2(y, x).series(x, 1, 4)
                  + atan2(y, 1) - atan(y)) == O((x - 1)**4, (x, 1))
    assert cancel(atan((y + x)/x).series(x, 1, 3) - atan2(y + x, x).series(x, 1, 3)
                  + atan2(1 + y, 1) - atan(1 + y)) == O((x - 1)**3, (x, 1))
    assert Matrix([atan2(y, x)]).jacobian([y, x]) == \
        Matrix([[x/(y**2 + x**2), -y/(y**2 + x**2)]])


def test_aseries():
    def t(n, v, d, e):
        assert abs(
            n(1/v).evalf() - n(1/x).series(x, dir=d).removeO().subs(x, v)) < e
    t(atan, 0.1, '+', 1e-5)
    t(atan, -0.1, '-', 1e-5)
    t(acot, 0.1, '+', 1e-5)
    t(acot, -0.1, '-', 1e-5)


def test_issue_4420():
    i = Symbol('i', integer=True)
    e = Symbol('e', even=True)
    o = Symbol('o', odd=True)

    # unknown parity for variable
    assert cos(4*i*pi) == 1
    assert sin(4*i*pi) == 0
    assert tan(4*i*pi) == 0
    assert cot(4*i*pi) is zoo

    assert cos(3*i*pi) == cos(pi*i)  # +/-1
    assert sin(3*i*pi) == 0
    assert tan(3*i*pi) == 0
    assert cot(3*i*pi) is zoo

    assert cos(4.0*i*pi) == 1
    assert sin(4.0*i*pi) == 0
    assert tan(4.0*i*pi) == 0
    assert cot(4.0*i*pi) is zoo

    assert cos(3.0*i*pi) == cos(pi*i)  # +/-1
    assert sin(3.0*i*pi) == 0
    assert tan(3.0*i*pi) == 0
    assert cot(3.0*i*pi) is zoo

    assert cos(4.5*i*pi) == cos(0.5*pi*i)
    assert sin(4.5*i*pi) == sin(0.5*pi*i)
    assert tan(4.5*i*pi) == tan(0.5*pi*i)
    assert cot(4.5*i*pi) == cot(0.5*pi*i)

    # parity of variable is known
    assert cos(4*e*pi) == 1
    assert sin(4*e*pi) == 0
    assert tan(4*e*pi) == 0
    assert cot(4*e*pi) is zoo

    assert cos(3*e*pi) == 1
    assert sin(3*e*pi) == 0
    assert tan(3*e*pi) == 0
    assert cot(3*e*pi) is zoo

    assert cos(4.0*e*pi) == 1
    assert sin(4.0*e*pi) == 0
    assert tan(4.0*e*pi) == 0
    assert cot(4.0*e*pi) is zoo

    assert cos(3.0*e*pi) == 1
    assert sin(3.0*e*pi) == 0
    assert tan(3.0*e*pi) == 0
    assert cot(3.0*e*pi) is zoo

    assert cos(4.5*e*pi) == cos(0.5*pi*e)
    assert sin(4.5*e*pi) == sin(0.5*pi*e)
    assert tan(4.5*e*pi) == tan(0.5*pi*e)
    assert cot(4.5*e*pi) == cot(0.5*pi*e)

    assert cos(4*o*pi) == 1
    assert sin(4*o*pi) == 0
    assert tan(4*o*pi) == 0
    assert cot(4*o*pi) is zoo

    assert cos(3*o*pi) == -1
    assert sin(3*o*pi) == 0
    assert tan(3*o*pi) == 0
    assert cot(3*o*pi) is zoo

    assert cos(4.0*o*pi) == 1
    assert sin(4.0*o*pi) == 0
    assert tan(4.0*o*pi) == 0
    assert cot(4.0*o*pi) is zoo

    assert cos(3.0*o*pi) == -1
    assert sin(3.0*o*pi) == 0
    assert tan(3.0*o*pi) == 0
    assert cot(3.0*o*pi) is zoo

    assert cos(4.5*o*pi) == cos(0.5*pi*o)
    assert sin(4.5*o*pi) == sin(0.5*pi*o)
    assert tan(4.5*o*pi) == tan(0.5*pi*o)
    assert cot(4.5*o*pi) == cot(0.5*pi*o)

    # x could be imaginary
    assert cos(4*x*pi) == cos(4*pi*x)
    assert sin(4*x*pi) == sin(4*pi*x)
    assert tan(4*x*pi) == tan(4*pi*x)
    assert cot(4*x*pi) == cot(4*pi*x)

    assert cos(3*x*pi) == cos(3*pi*x)
    assert sin(3*x*pi) == sin(3*pi*x)
    assert tan(3*x*pi) == tan(3*pi*x)
    assert cot(3*x*pi) == cot(3*pi*x)

    assert cos(4.0*x*pi) == cos(4.0*pi*x)
    assert sin(4.0*x*pi) == sin(4.0*pi*x)
    assert tan(4.0*x*pi) == tan(4.0*pi*x)
    assert cot(4.0*x*pi) == cot(4.0*pi*x)

    assert cos(3.0*x*pi) == cos(3.0*pi*x)
    assert sin(3.0*x*pi) == sin(3.0*pi*x)
    assert tan(3.0*x*pi) == tan(3.0*pi*x)
    assert cot(3.0*x*pi) == cot(3.0*pi*x)

    assert cos(4.5*x*pi) == cos(4.5*pi*x)
    assert sin(4.5*x*pi) == sin(4.5*pi*x)
    assert tan(4.5*x*pi) == tan(4.5*pi*x)
    assert cot(4.5*x*pi) == cot(4.5*pi*x)


def test_inverses():
    raises(AttributeError, lambda: sin(x).inverse())
    raises(AttributeError, lambda: cos(x).inverse())
    assert tan(x).inverse() == atan
    assert cot(x).inverse() == acot
    raises(AttributeError, lambda: csc(x).inverse())
    raises(AttributeError, lambda: sec(x).inverse())
    assert asin(x).inverse() == sin
    assert acos(x).inverse() == cos
    assert atan(x).inverse() == tan
    assert acot(x).inverse() == cot


def test_real_imag():
    a, b = symbols('a b', real=True)
    z = a + b*I
    for deep in [True, False]:
        assert sin(
            z).as_real_imag(deep=deep) == (sin(a)*cosh(b), cos(a)*sinh(b))
        assert cos(
            z).as_real_imag(deep=deep) == (cos(a)*cosh(b), -sin(a)*sinh(b))
        assert tan(z).as_real_imag(deep=deep) == (sin(2*a)/(cos(2*a) +
            cosh(2*b)), sinh(2*b)/(cos(2*a) + cosh(2*b)))
        assert cot(z).as_real_imag(deep=deep) == (-sin(2*a)/(cos(2*a) -
            cosh(2*b)), sinh(2*b)/(cos(2*a) - cosh(2*b)))
        assert sin(a).as_real_imag(deep=deep) == (sin(a), 0)
        assert cos(a).as_real_imag(deep=deep) == (cos(a), 0)
        assert tan(a).as_real_imag(deep=deep) == (tan(a), 0)
        assert cot(a).as_real_imag(deep=deep) == (cot(a), 0)


@slow
def test_sincos_rewrite_sqrt():
    # equivalent to testing rewrite(pow)
    for p in [1, 3, 5, 17]:
        for t in [1, 8]:
            n = t*p
            # The vertices `exp(i*pi/n)` of a regular `n`-gon can
            # be expressed by means of nested square roots if and
            # only if `n` is a product of Fermat primes, `p`, and
            # powers of 2, `t'. The code aims to check all vertices
            # not belonging to an `m`-gon for `m < n`(`gcd(i, n) == 1`).
            # For large `n` this makes the test too slow, therefore
            # the vertices are limited to those of index `i < 10`.
            for i in range(1, min((n + 1)//2 + 1, 10)):
                if 1 == gcd(i, n):
                    x = i*pi/n
                    s1 = sin(x).rewrite(sqrt)
                    c1 = cos(x).rewrite(sqrt)
                    assert not s1.has(cos, sin), "fails for %d*pi/%d" % (i, n)
                    assert not c1.has(cos, sin), "fails for %d*pi/%d" % (i, n)
                    assert 1e-3 > abs(sin(x.evalf(5)) - s1.evalf(2)), "fails for %d*pi/%d" % (i, n)
                    assert 1e-3 > abs(cos(x.evalf(5)) - c1.evalf(2)), "fails for %d*pi/%d" % (i, n)
    assert cos(pi/14).rewrite(sqrt) == sqrt(cos(pi/7)/2 + S.Half)
    assert cos(pi*Rational(-15, 2)/11, evaluate=False).rewrite(
        sqrt) == -sqrt(-cos(pi*Rational(4, 11))/2 + S.Half)
    assert cos(Mul(2, pi, S.Half, evaluate=False), evaluate=False).rewrite(
        sqrt) == -1
    e = cos(pi/3/17)  # don't use pi/15 since that is caught at instantiation
    a = (
        -3*sqrt(-sqrt(17) + 17)*sqrt(sqrt(17) + 17)/64 -
        3*sqrt(34)*sqrt(sqrt(17) + 17)/128 - sqrt(sqrt(17) +
        17)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) + 17)
        + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/64 - sqrt(-sqrt(17)
        + 17)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/128 - Rational(1, 32) +
        sqrt(2)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/64 +
        3*sqrt(2)*sqrt(sqrt(17) + 17)/128 + sqrt(34)*sqrt(-sqrt(17) + 17)/128
        + 13*sqrt(2)*sqrt(-sqrt(17) + 17)/128 + sqrt(17)*sqrt(-sqrt(17) +
        17)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) + 17)
        + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/128 + 5*sqrt(17)/32
        + sqrt(3)*sqrt(-sqrt(2)*sqrt(sqrt(17) + 17)*sqrt(sqrt(17)/32 +
        sqrt(2)*sqrt(-sqrt(17) + 17)/32 +
        sqrt(2)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/32 + Rational(15, 32))/8 -
        5*sqrt(2)*sqrt(sqrt(17)/32 + sqrt(2)*sqrt(-sqrt(17) + 17)/32 +
        sqrt(2)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/32 +
        Rational(15, 32))*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/64 -
        3*sqrt(2)*sqrt(-sqrt(17) + 17)*sqrt(sqrt(17)/32 +
        sqrt(2)*sqrt(-sqrt(17) + 17)/32 +
        sqrt(2)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/32 + Rational(15, 32))/32
        + sqrt(34)*sqrt(sqrt(17)/32 + sqrt(2)*sqrt(-sqrt(17) + 17)/32 +
        sqrt(2)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/32 +
        Rational(15, 32))*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/64 +
        sqrt(sqrt(17)/32 + sqrt(2)*sqrt(-sqrt(17) + 17)/32 +
        sqrt(2)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/32 + Rational(15, 32))/2 +
        S.Half + sqrt(-sqrt(17) + 17)*sqrt(sqrt(17)/32 + sqrt(2)*sqrt(-sqrt(17) +
        17)/32 + sqrt(2)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) -
        sqrt(2)*sqrt(-sqrt(17) + 17) + sqrt(34)*sqrt(-sqrt(17) + 17) +
        6*sqrt(17) + 34)/32 + Rational(15, 32))*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) -
        sqrt(2)*sqrt(-sqrt(17) + 17) + sqrt(34)*sqrt(-sqrt(17) + 17) +
        6*sqrt(17) + 34)/32 + sqrt(34)*sqrt(-sqrt(17) + 17)*sqrt(sqrt(17)/32 +
        sqrt(2)*sqrt(-sqrt(17) + 17)/32 +
        sqrt(2)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/32 +
        Rational(15, 32))/32)/2)
    assert e.rewrite(sqrt) == a
    assert e.n() == a.n()
    # coverage of fermatCoords: multiplicity > 1; the following could be
    # different but that portion of the code should be tested in some way
    assert cos(pi/9/17).rewrite(sqrt) == \
        sin(pi/9)*sin(pi*Rational(2, 17)) + cos(pi/9)*cos(pi*Rational(2, 17))


@slow
def test_sincos_rewrite_sqrt_257():
    assert cos(pi/257).rewrite(sqrt).evalf(64) == cos(pi/257).evalf(64)


@slow
def test_tancot_rewrite_sqrt():
    # equivalent to testing rewrite(pow)
    for p in [1, 3, 5, 17]:
        for t in [1, 8]:
            n = t*p
            for i in range(1, min((n + 1)//2 + 1, 10)):
                if 1 == gcd(i, n):
                    x = i*pi/n
                    if  2*i != n and 3*i != 2*n:
                        t1 = tan(x).rewrite(sqrt)
                        assert not t1.has(cot, tan), "fails for %d*pi/%d" % (i, n)
                        assert 1e-3 > abs( tan(x.evalf(7)) - t1.evalf(4) ), "fails for %d*pi/%d" % (i, n)
                    if  i != 0 and i != n:
                        c1 = cot(x).rewrite(sqrt)
                        assert not c1.has(cot, tan), "fails for %d*pi/%d" % (i, n)
                        assert 1e-3 > abs( cot(x.evalf(7)) - c1.evalf(4) ), "fails for %d*pi/%d" % (i, n)


def test_sec():
    x = symbols('x', real=True)
    z = symbols('z')

    assert sec.nargs == FiniteSet(1)

    assert sec(zoo) is nan
    assert sec(0) == 1
    assert sec(pi) == -1
    assert sec(pi/2) is zoo
    assert sec(-pi/2) is zoo
    assert sec(pi/6) == 2*sqrt(3)/3
    assert sec(pi/3) == 2
    assert sec(pi*Rational(5, 2)) is zoo
    assert sec(pi*Rational(9, 7)) == -sec(pi*Rational(2, 7))
    assert sec(pi*Rational(3, 4)) == -sqrt(2)  # issue 8421
    assert sec(I) == 1/cosh(1)
    assert sec(x*I) == 1/cosh(x)
    assert sec(-x) == sec(x)

    assert sec(asec(x)) == x

    assert sec(z).conjugate() == sec(conjugate(z))

    assert (sec(z).as_real_imag() ==
    (cos(re(z))*cosh(im(z))/(sin(re(z))**2*sinh(im(z))**2 +
                             cos(re(z))**2*cosh(im(z))**2),
     sin(re(z))*sinh(im(z))/(sin(re(z))**2*sinh(im(z))**2 +
                             cos(re(z))**2*cosh(im(z))**2)))

    assert sec(x).expand(trig=True) == 1/cos(x)
    assert sec(2*x).expand(trig=True) == 1/(2*cos(x)**2 - 1)

    assert sec(x).is_extended_real == True
    assert sec(z).is_real == None

    assert sec(a).is_algebraic is None
    assert sec(na).is_algebraic is False

    assert sec(x).as_leading_term() == sec(x)

    assert sec(0, evaluate=False).is_finite == True
    assert sec(x).is_finite == None
    assert sec(pi/2, evaluate=False).is_finite == False

    assert series(sec(x), x, x0=0, n=6) == 1 + x**2/2 + 5*x**4/24 + O(x**6)

    # https://github.com/sympy/sympy/issues/7166
    assert series(sqrt(sec(x))) == 1 + x**2/4 + 7*x**4/96 + O(x**6)

    # https://github.com/sympy/sympy/issues/7167
    assert (series(sqrt(sec(x)), x, x0=pi*3/2, n=4) ==
            1/sqrt(x - pi*Rational(3, 2)) + (x - pi*Rational(3, 2))**Rational(3, 2)/12 +
            (x - pi*Rational(3, 2))**Rational(7, 2)/160 + O((x - pi*Rational(3, 2))**4, (x, pi*Rational(3, 2))))

    assert sec(x).diff(x) == tan(x)*sec(x)

    # Taylor Term checks
    assert sec(z).taylor_term(4, z) == 5*z**4/24
    assert sec(z).taylor_term(6, z) == 61*z**6/720
    assert sec(z).taylor_term(5, z) == 0


def test_sec_rewrite():
    assert sec(x).rewrite(exp) == 1/(exp(I*x)/2 + exp(-I*x)/2)
    assert sec(x).rewrite(cos) == 1/cos(x)
    assert sec(x).rewrite(tan) == (tan(x/2)**2 + 1)/(-tan(x/2)**2 + 1)
    assert sec(x).rewrite(pow) == sec(x)
    assert sec(x).rewrite(sqrt) == sec(x)
    assert sec(z).rewrite(cot) == (cot(z/2)**2 + 1)/(cot(z/2)**2 - 1)
    assert sec(x).rewrite(sin) == 1 / sin(x + pi / 2, evaluate=False)
    assert sec(x).rewrite(tan) == (tan(x / 2)**2 + 1) / (-tan(x / 2)**2 + 1)
    assert sec(x).rewrite(csc) == csc(-x + pi/2, evaluate=False)
    assert sec(x).rewrite(besselj) == Piecewise(
                (sqrt(2)/(sqrt(pi*x)*besselj(-S.Half, x)), Ne(x, 0)),
                (1, True)
            )
    assert sec(x).rewrite(besselj).subs(x, 0) == sec(0)


def test_sec_fdiff():
    assert sec(x).fdiff() == tan(x)*sec(x)
    raises(ArgumentIndexError, lambda: sec(x).fdiff(2))


def test_csc():
    x = symbols('x', real=True)
    z = symbols('z')

    # https://github.com/sympy/sympy/issues/6707
    cosecant = csc('x')
    alternate = 1/sin('x')
    assert cosecant.equals(alternate) == True
    assert alternate.equals(cosecant) == True

    assert csc.nargs == FiniteSet(1)

    assert csc(0) is zoo
    assert csc(pi) is zoo
    assert csc(zoo) is nan

    assert csc(pi/2) == 1
    assert csc(-pi/2) == -1
    assert csc(pi/6) == 2
    assert csc(pi/3) == 2*sqrt(3)/3
    assert csc(pi*Rational(5, 2)) == 1
    assert csc(pi*Rational(9, 7)) == -csc(pi*Rational(2, 7))
    assert csc(pi*Rational(3, 4)) == sqrt(2)  # issue 8421
    assert csc(I) == -I/sinh(1)
    assert csc(x*I) == -I/sinh(x)
    assert csc(-x) == -csc(x)

    assert csc(acsc(x)) == x

    assert csc(z).conjugate() == csc(conjugate(z))

    assert (csc(z).as_real_imag() ==
            (sin(re(z))*cosh(im(z))/(sin(re(z))**2*cosh(im(z))**2 +
                                     cos(re(z))**2*sinh(im(z))**2),
             -cos(re(z))*sinh(im(z))/(sin(re(z))**2*cosh(im(z))**2 +
                          cos(re(z))**2*sinh(im(z))**2)))

    assert csc(x).expand(trig=True) == 1/sin(x)
    assert csc(2*x).expand(trig=True) == 1/(2*sin(x)*cos(x))

    assert csc(x).is_extended_real == True
    assert csc(z).is_real == None

    assert csc(a).is_algebraic is None
    assert csc(na).is_algebraic is False

    assert csc(x).as_leading_term() == csc(x)

    assert csc(0, evaluate=False).is_finite == False
    assert csc(x).is_finite == None
    assert csc(pi/2, evaluate=False).is_finite == True

    assert series(csc(x), x, x0=pi/2, n=6) == \
        1 + (x - pi/2)**2/2 + 5*(x - pi/2)**4/24 + O((x - pi/2)**6, (x, pi/2))
    assert series(csc(x), x, x0=0, n=6) == \
            1/x + x/6 + 7*x**3/360 + 31*x**5/15120 + O(x**6)

    assert csc(x).diff(x) == -cot(x)*csc(x)

    assert csc(x).taylor_term(2, x) == 0
    assert csc(x).taylor_term(3, x) == 7*x**3/360
    assert csc(x).taylor_term(5, x) == 31*x**5/15120
    raises(ArgumentIndexError, lambda: csc(x).fdiff(2))


def test_asec():
    z = Symbol('z', zero=True)
    assert asec(z) is zoo
    assert asec(nan) is nan
    assert asec(1) == 0
    assert asec(-1) == pi
    assert asec(oo) == pi/2
    assert asec(-oo) == pi/2
    assert asec(zoo) == pi/2

    assert asec(sec(pi*Rational(13, 4))) == pi*Rational(3, 4)
    assert asec(1 + sqrt(5)) == pi*Rational(2, 5)
    assert asec(2/sqrt(3)) == pi/6
    assert asec(sqrt(4 - 2*sqrt(2))) == pi/8
    assert asec(-sqrt(4 + 2*sqrt(2))) == pi*Rational(5, 8)
    assert asec(sqrt(2 + 2*sqrt(5)/5)) == pi*Rational(3, 10)
    assert asec(-sqrt(2 + 2*sqrt(5)/5)) == pi*Rational(7, 10)
    assert asec(sqrt(2) - sqrt(6)) == pi*Rational(11, 12)

    for d in [3, 4, 6]:
        for num in range(d):
            if gcd(num, d) == 1:
                assert asec(sec(num*pi/d)) == num*pi/d
                assert asec(-sec(num*pi/d)) == pi - num*pi/d
                assert asec(csc(num*pi/d)) == pi/2 - acsc(csc(num*pi/d))
                assert asec(-csc(num*pi/d)) == pi/2 - acsc(-csc(num*pi/d))

    assert asec(x).diff(x) == 1/(x**2*sqrt(1 - 1/x**2))

    assert asec(x).rewrite(log) == I*log(sqrt(1 - 1/x**2) + I/x) + pi/2
    assert asec(x).rewrite(asin) == -asin(1/x) + pi/2
    assert asec(x).rewrite(acos) == acos(1/x)
    assert asec(x).rewrite(atan) == \
        pi*(1 - sqrt(x**2)/x)/2 + sqrt(x**2)*atan(sqrt(x**2 - 1))/x
    assert asec(x).rewrite(acot) == \
        pi*(1 - sqrt(x**2)/x)/2 + sqrt(x**2)*acot(1/sqrt(x**2 - 1))/x
    assert asec(x).rewrite(acsc) == -acsc(x) + pi/2
    raises(ArgumentIndexError, lambda: asec(x).fdiff(2))


def test_asec_is_real():
    assert asec(S.Half).is_real is False
    n = Symbol('n', positive=True, integer=True)
    assert asec(n).is_extended_real is True
    assert asec(x).is_real is None
    assert asec(r).is_real is None
    t = Symbol('t', real=False, finite=True)
    assert asec(t).is_real is False


def test_asec_leading_term():
    assert asec(1/x).as_leading_term(x) == pi/2
    # Tests concerning branch points
    assert asec(x + 1).as_leading_term(x) == sqrt(2)*sqrt(x)
    assert asec(x - 1).as_leading_term(x) == pi
    # Tests concerning points lying on branch cuts
    assert asec(x).as_leading_term(x, cdir=1) == -I*log(x) + I*log(2)
    assert asec(x).as_leading_term(x, cdir=-1) == I*log(x) + 2*pi - I*log(2)
    assert asec(I*x + 1/2).as_leading_term(x, cdir=1) == asec(1/2)
    assert asec(-I*x + 1/2).as_leading_term(x, cdir=1) == -asec(1/2)
    assert asec(I*x - 1/2).as_leading_term(x, cdir=1) == 2*pi - asec(-1/2)
    assert asec(-I*x - 1/2).as_leading_term(x, cdir=1) == asec(-1/2)
    # Tests concerning im(ndir) == 0
    assert asec(-I*x**2 + x - S(1)/2).as_leading_term(x, cdir=1) == pi + I*log(2 - sqrt(3))
    assert asec(-I*x**2 + x - S(1)/2).as_leading_term(x, cdir=-1) == pi + I*log(2 - sqrt(3))


def test_asec_series():
    assert asec(x).series(x, 0, 9) == \
        I*log(2) - I*log(x) - I*x**2/4 - 3*I*x**4/32 \
        - 5*I*x**6/96 - 35*I*x**8/1024 + O(x**9)
    t4 = asec(x).taylor_term(4, x)
    assert t4 == -3*I*x**4/32
    assert asec(x).taylor_term(6, x, t4, 0) == -5*I*x**6/96


def test_acsc():
    assert acsc(nan) is nan
    assert acsc(1) == pi/2
    assert acsc(-1) == -pi/2
    assert acsc(oo) == 0
    assert acsc(-oo) == 0
    assert acsc(zoo) == 0
    assert acsc(0) is zoo

    assert acsc(csc(3)) == -3 + pi
    assert acsc(csc(4)) == -4 + pi
    assert acsc(csc(6)) == 6 - 2*pi
    assert unchanged(acsc, csc(x))
    assert unchanged(acsc, sec(x))

    assert acsc(2/sqrt(3)) == pi/3
    assert acsc(csc(pi*Rational(13, 4))) == -pi/4
    assert acsc(sqrt(2 + 2*sqrt(5)/5)) == pi/5
    assert acsc(-sqrt(2 + 2*sqrt(5)/5)) == -pi/5
    assert acsc(-2) == -pi/6
    assert acsc(-sqrt(4 + 2*sqrt(2))) == -pi/8
    assert acsc(sqrt(4 - 2*sqrt(2))) == pi*Rational(3, 8)
    assert acsc(1 + sqrt(5)) == pi/10
    assert acsc(sqrt(2) - sqrt(6)) == pi*Rational(-5, 12)

    assert acsc(x).diff(x) == -1/(x**2*sqrt(1 - 1/x**2))

    assert acsc(x).rewrite(log) == -I*log(sqrt(1 - 1/x**2) + I/x)
    assert acsc(x).rewrite(asin) == asin(1/x)
    assert acsc(x).rewrite(acos) == -acos(1/x) + pi/2
    assert acsc(x).rewrite(atan) == \
        (-atan(sqrt(x**2 - 1)) + pi/2)*sqrt(x**2)/x
    assert acsc(x).rewrite(acot) == (-acot(1/sqrt(x**2 - 1)) + pi/2)*sqrt(x**2)/x
    assert acsc(x).rewrite(asec) == -asec(x) + pi/2
    raises(ArgumentIndexError, lambda: acsc(x).fdiff(2))


def test_csc_rewrite():
    assert csc(x).rewrite(pow) == csc(x)
    assert csc(x).rewrite(sqrt) == csc(x)

    assert csc(x).rewrite(exp) == 2*I/(exp(I*x) - exp(-I*x))
    assert csc(x).rewrite(sin) == 1/sin(x)
    assert csc(x).rewrite(tan) == (tan(x/2)**2 + 1)/(2*tan(x/2))
    assert csc(x).rewrite(cot) == (cot(x/2)**2 + 1)/(2*cot(x/2))
    assert csc(x).rewrite(cos) == 1/cos(x - pi/2, evaluate=False)
    assert csc(x).rewrite(sec) == sec(-x + pi/2, evaluate=False)

    # issue 17349
    assert csc(1 - exp(-besselj(I, I))).rewrite(cos) == \
           -1/cos(-pi/2 - 1 + cos(I*besselj(I, I)) +
                  I*cos(-pi/2 + I*besselj(I, I), evaluate=False), evaluate=False)
    assert csc(x).rewrite(besselj) == sqrt(2)/(sqrt(pi*x)*besselj(S.Half, x))
    assert csc(x).rewrite(besselj).subs(x, 0) == csc(0)


def test_acsc_leading_term():
    assert acsc(1/x).as_leading_term(x) == x
    # Tests concerning branch points
    assert acsc(x + 1).as_leading_term(x) == pi/2
    assert acsc(x - 1).as_leading_term(x) == -pi/2
    # Tests concerning points lying on branch cuts
    assert acsc(x).as_leading_term(x, cdir=1) == I*log(x) + pi/2 - I*log(2)
    assert acsc(x).as_leading_term(x, cdir=-1) == -I*log(x) - 3*pi/2 + I*log(2)
    assert acsc(I*x + 1/2).as_leading_term(x, cdir=1) == acsc(1/2)
    assert acsc(-I*x + 1/2).as_leading_term(x, cdir=1) == pi - acsc(1/2)
    assert acsc(I*x - 1/2).as_leading_term(x, cdir=1) == -pi - acsc(-1/2)
    assert acsc(-I*x - 1/2).as_leading_term(x, cdir=1) == -acsc(1/2)
    # Tests concerning im(ndir) == 0
    assert acsc(-I*x**2 + x - S(1)/2).as_leading_term(x, cdir=1) == -pi/2 + I*log(sqrt(3) + 2)
    assert acsc(-I*x**2 + x - S(1)/2).as_leading_term(x, cdir=-1) == -pi/2 + I*log(sqrt(3) + 2)


def test_acsc_series():
    assert acsc(x).series(x, 0, 9) == \
        -I*log(2) + pi/2 + I*log(x) + I*x**2/4 \
        + 3*I*x**4/32 + 5*I*x**6/96 + 35*I*x**8/1024 + O(x**9)
    t6 = acsc(x).taylor_term(6, x)
    assert t6 == 5*I*x**6/96
    assert acsc(x).taylor_term(8, x, t6, 0) == 35*I*x**8/1024


def test_asin_nseries():
    assert asin(x + 2)._eval_nseries(x, 4, None, I) == -asin(2) + pi + \
    sqrt(3)*I*x/3 - sqrt(3)*I*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    assert asin(x + 2)._eval_nseries(x, 4, None, -I) == asin(2) - \
    sqrt(3)*I*x/3 + sqrt(3)*I*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert asin(x - 2)._eval_nseries(x, 4, None, I) == -asin(2) - \
    sqrt(3)*I*x/3 - sqrt(3)*I*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert asin(x - 2)._eval_nseries(x, 4, None, -I) == asin(2) - pi + \
    sqrt(3)*I*x/3 + sqrt(3)*I*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    # testing nseries for asin at branch points
    assert asin(1 + x)._eval_nseries(x, 3, None) == pi/2 - sqrt(2)*sqrt(-x) - \
    sqrt(2)*(-x)**(S(3)/2)/12 - 3*sqrt(2)*(-x)**(S(5)/2)/160 + O(x**3)
    assert asin(-1 + x)._eval_nseries(x, 3, None) == -pi/2 + sqrt(2)*sqrt(x) + \
    sqrt(2)*x**(S(3)/2)/12 + 3*sqrt(2)*x**(S(5)/2)/160 + O(x**3)
    assert asin(exp(x))._eval_nseries(x, 3, None) == pi/2 - sqrt(2)*sqrt(-x) + \
    sqrt(2)*(-x)**(S(3)/2)/6 - sqrt(2)*(-x)**(S(5)/2)/120 + O(x**3)
    assert asin(-exp(x))._eval_nseries(x, 3, None) == -pi/2 + sqrt(2)*sqrt(-x) - \
    sqrt(2)*(-x)**(S(3)/2)/6 + sqrt(2)*(-x)**(S(5)/2)/120 + O(x**3)


def test_acos_nseries():
    assert acos(x + 2)._eval_nseries(x, 4, None, I) == -acos(2) - sqrt(3)*I*x/3 + \
    sqrt(3)*I*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert acos(x + 2)._eval_nseries(x, 4, None, -I) == acos(2) + sqrt(3)*I*x/3 - \
    sqrt(3)*I*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    assert acos(x - 2)._eval_nseries(x, 4, None, I) == acos(-2) + sqrt(3)*I*x/3 + \
    sqrt(3)*I*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    assert acos(x - 2)._eval_nseries(x, 4, None, -I) == -acos(-2) + 2*pi - \
    sqrt(3)*I*x/3 - sqrt(3)*I*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    # testing nseries for acos at branch points
    assert acos(1 + x)._eval_nseries(x, 3, None) == sqrt(2)*sqrt(-x) + \
    sqrt(2)*(-x)**(S(3)/2)/12 + 3*sqrt(2)*(-x)**(S(5)/2)/160 + O(x**3)
    assert acos(-1 + x)._eval_nseries(x, 3, None) == pi - sqrt(2)*sqrt(x) - \
    sqrt(2)*x**(S(3)/2)/12 - 3*sqrt(2)*x**(S(5)/2)/160 + O(x**3)
    assert acos(exp(x))._eval_nseries(x, 3, None) == sqrt(2)*sqrt(-x) - \
    sqrt(2)*(-x)**(S(3)/2)/6 + sqrt(2)*(-x)**(S(5)/2)/120 + O(x**3)
    assert acos(-exp(x))._eval_nseries(x, 3, None) == pi - sqrt(2)*sqrt(-x) + \
    sqrt(2)*(-x)**(S(3)/2)/6 - sqrt(2)*(-x)**(S(5)/2)/120 + O(x**3)


def test_atan_nseries():
    assert atan(x + 2*I)._eval_nseries(x, 4, None, 1) == I*atanh(2) - x/3 - \
    2*I*x**2/9 + 13*x**3/81 + O(x**4)
    assert atan(x + 2*I)._eval_nseries(x, 4, None, -1) == I*atanh(2) - pi - \
    x/3 - 2*I*x**2/9 + 13*x**3/81 + O(x**4)
    assert atan(x - 2*I)._eval_nseries(x, 4, None, 1) == -I*atanh(2) + pi - \
    x/3 + 2*I*x**2/9 + 13*x**3/81 + O(x**4)
    assert atan(x - 2*I)._eval_nseries(x, 4, None, -1) == -I*atanh(2) - x/3 + \
    2*I*x**2/9 + 13*x**3/81 + O(x**4)
    assert atan(1/x)._eval_nseries(x, 2, None, 1) == pi/2 - x + O(x**2)
    assert atan(1/x)._eval_nseries(x, 2, None, -1) == -pi/2 - x + O(x**2)
    # testing nseries for atan at branch points
    assert atan(x + I)._eval_nseries(x, 4, None) == I*log(2)/2 + pi/4 - \
    I*log(x)/2 + x/4 + I*x**2/16 - x**3/48 + O(x**4)
    assert atan(x - I)._eval_nseries(x, 4, None) == -I*log(2)/2 + pi/4 + \
    I*log(x)/2 + x/4 - I*x**2/16 - x**3/48 + O(x**4)


def test_acot_nseries():
    assert acot(x + S(1)/2*I)._eval_nseries(x, 4, None, 1) == -I*acoth(S(1)/2) + \
    pi - 4*x/3 + 8*I*x**2/9 + 112*x**3/81 + O(x**4)
    assert acot(x + S(1)/2*I)._eval_nseries(x, 4, None, -1) == -I*acoth(S(1)/2) - \
    4*x/3 + 8*I*x**2/9 + 112*x**3/81 + O(x**4)
    assert acot(x - S(1)/2*I)._eval_nseries(x, 4, None, 1) == I*acoth(S(1)/2) - \
    4*x/3 - 8*I*x**2/9 + 112*x**3/81 + O(x**4)
    assert acot(x - S(1)/2*I)._eval_nseries(x, 4, None, -1) == I*acoth(S(1)/2) - \
    pi - 4*x/3 - 8*I*x**2/9 + 112*x**3/81 + O(x**4)
    assert acot(x)._eval_nseries(x, 2, None, 1) == pi/2 - x + O(x**2)
    assert acot(x)._eval_nseries(x, 2, None, -1) == -pi/2 - x + O(x**2)
    # testing nseries for acot at branch points
    assert acot(x + I)._eval_nseries(x, 4, None) == -I*log(2)/2 + pi/4 + \
    I*log(x)/2 - x/4 - I*x**2/16 + x**3/48 + O(x**4)
    assert acot(x - I)._eval_nseries(x, 4, None) == I*log(2)/2 + pi/4 - \
    I*log(x)/2 - x/4 + I*x**2/16 + x**3/48 + O(x**4)


def test_asec_nseries():
    assert asec(x + S(1)/2)._eval_nseries(x, 4, None, I) == asec(S(1)/2) - \
    4*sqrt(3)*I*x/3 + 8*sqrt(3)*I*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert asec(x + S(1)/2)._eval_nseries(x, 4, None, -I) == -asec(S(1)/2) + \
    4*sqrt(3)*I*x/3 - 8*sqrt(3)*I*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert asec(x - S(1)/2)._eval_nseries(x, 4, None, I) == -asec(-S(1)/2) + \
    2*pi + 4*sqrt(3)*I*x/3 + 8*sqrt(3)*I*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert asec(x - S(1)/2)._eval_nseries(x, 4, None, -I) == asec(-S(1)/2) - \
    4*sqrt(3)*I*x/3 - 8*sqrt(3)*I*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)
    # testing nseries for asec at branch points
    assert asec(1 + x)._eval_nseries(x, 3, None) == sqrt(2)*sqrt(x) - \
    5*sqrt(2)*x**(S(3)/2)/12 + 43*sqrt(2)*x**(S(5)/2)/160 + O(x**3)
    assert asec(-1 + x)._eval_nseries(x, 3, None) == pi - sqrt(2)*sqrt(-x) + \
    5*sqrt(2)*(-x)**(S(3)/2)/12 - 43*sqrt(2)*(-x)**(S(5)/2)/160 + O(x**3)
    assert asec(exp(x))._eval_nseries(x, 3, None) == sqrt(2)*sqrt(x) - \
    sqrt(2)*x**(S(3)/2)/6 + sqrt(2)*x**(S(5)/2)/120 + O(x**3)
    assert asec(-exp(x))._eval_nseries(x, 3, None) == pi - sqrt(2)*sqrt(x) + \
    sqrt(2)*x**(S(3)/2)/6 - sqrt(2)*x**(S(5)/2)/120 + O(x**3)


def test_acsc_nseries():
    assert acsc(x + S(1)/2)._eval_nseries(x, 4, None, I) == acsc(S(1)/2) + \
    4*sqrt(3)*I*x/3 - 8*sqrt(3)*I*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert acsc(x + S(1)/2)._eval_nseries(x, 4, None, -I) == -acsc(S(1)/2) + \
    pi - 4*sqrt(3)*I*x/3 + 8*sqrt(3)*I*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert acsc(x - S(1)/2)._eval_nseries(x, 4, None, I) == acsc(S(1)/2) - pi -\
    4*sqrt(3)*I*x/3 - 8*sqrt(3)*I*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert acsc(x - S(1)/2)._eval_nseries(x, 4, None, -I) == -acsc(S(1)/2) + \
    4*sqrt(3)*I*x/3 + 8*sqrt(3)*I*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)
    # testing nseries for acsc at branch points
    assert acsc(1 + x)._eval_nseries(x, 3, None) == pi/2 - sqrt(2)*sqrt(x) + \
    5*sqrt(2)*x**(S(3)/2)/12 - 43*sqrt(2)*x**(S(5)/2)/160 + O(x**3)
    assert acsc(-1 + x)._eval_nseries(x, 3, None) == -pi/2 + sqrt(2)*sqrt(-x) - \
    5*sqrt(2)*(-x)**(S(3)/2)/12 + 43*sqrt(2)*(-x)**(S(5)/2)/160 + O(x**3)
    assert acsc(exp(x))._eval_nseries(x, 3, None) == pi/2 - sqrt(2)*sqrt(x) + \
    sqrt(2)*x**(S(3)/2)/6 - sqrt(2)*x**(S(5)/2)/120 + O(x**3)
    assert acsc(-exp(x))._eval_nseries(x, 3, None) == -pi/2 + sqrt(2)*sqrt(x) - \
    sqrt(2)*x**(S(3)/2)/6 + sqrt(2)*x**(S(5)/2)/120 + O(x**3)


def test_issue_8653():
    n = Symbol('n', integer=True)
    assert sin(n).is_irrational is None
    assert cos(n).is_irrational is None
    assert tan(n).is_irrational is None


def test_issue_9157():
    n = Symbol('n', integer=True, positive=True)
    assert atan(n - 1).is_nonnegative is True


def test_trig_period():
    x, y = symbols('x, y')

    assert sin(x).period() == 2*pi
    assert cos(x).period() == 2*pi
    assert tan(x).period() == pi
    assert cot(x).period() == pi
    assert sec(x).period() == 2*pi
    assert csc(x).period() == 2*pi
    assert sin(2*x).period() == pi
    assert cot(4*x - 6).period() == pi/4
    assert cos((-3)*x).period() == pi*Rational(2, 3)
    assert cos(x*y).period(x) == 2*pi/abs(y)
    assert sin(3*x*y + 2*pi).period(y) == 2*pi/abs(3*x)
    assert tan(3*x).period(y) is S.Zero
    raises(NotImplementedError, lambda: sin(x**2).period(x))


def test_issue_7171():
    assert sin(x).rewrite(sqrt) == sin(x)
    assert sin(x).rewrite(pow) == sin(x)


def test_issue_11864():
    w, k = symbols('w, k', real=True)
    F = Piecewise((1, Eq(2*pi*k, 0)), (sin(pi*k)/(pi*k), True))
    soln = Piecewise((1, Eq(2*pi*k, 0)), (sinc(pi*k), True))
    assert F.rewrite(sinc) == soln

def test_real_assumptions():
    z = Symbol('z', real=False, finite=True)
    assert sin(z).is_real is None
    assert cos(z).is_real is None
    assert tan(z).is_real is False
    assert sec(z).is_real is None
    assert csc(z).is_real is None
    assert cot(z).is_real is False
    assert asin(p).is_real is None
    assert asin(n).is_real is None
    assert asec(p).is_real is None
    assert asec(n).is_real is None
    assert acos(p).is_real is None
    assert acos(n).is_real is None
    assert acsc(p).is_real is None
    assert acsc(n).is_real is None
    assert atan(p).is_positive is True
    assert atan(n).is_negative is True
    assert acot(p).is_positive is True
    assert acot(n).is_negative is True

def test_issue_14320():
    assert asin(sin(2)) == -2 + pi and (-pi/2 <= -2 + pi <= pi/2) and sin(2) == sin(-2 + pi)
    assert asin(cos(2)) == -2 + pi/2 and (-pi/2 <= -2 + pi/2 <= pi/2) and cos(2) == sin(-2 + pi/2)
    assert acos(sin(2)) == -pi/2 + 2 and (0 <= -pi/2 + 2 <= pi) and sin(2) == cos(-pi/2 + 2)
    assert acos(cos(20)) == -6*pi + 20 and (0 <= -6*pi + 20 <= pi) and cos(20) == cos(-6*pi + 20)
    assert acos(cos(30)) == -30 + 10*pi and (0 <= -30 + 10*pi <= pi) and cos(30) == cos(-30 + 10*pi)

    assert atan(tan(17)) == -5*pi + 17 and (-pi/2 < -5*pi + 17 < pi/2) and tan(17) == tan(-5*pi + 17)
    assert atan(tan(15)) == -5*pi + 15 and (-pi/2 < -5*pi + 15 < pi/2) and tan(15) == tan(-5*pi + 15)
    assert atan(cot(12)) == -12 + pi*Rational(7, 2) and (-pi/2 < -12 + pi*Rational(7, 2) < pi/2) and cot(12) == tan(-12 + pi*Rational(7, 2))
    assert acot(cot(15)) == -5*pi + 15 and (-pi/2 < -5*pi + 15 <= pi/2) and cot(15) == cot(-5*pi + 15)
    assert acot(tan(19)) == -19 + pi*Rational(13, 2) and (-pi/2 < -19 + pi*Rational(13, 2) <= pi/2) and tan(19) == cot(-19 + pi*Rational(13, 2))

    assert asec(sec(11)) == -11 + 4*pi and (0 <= -11 + 4*pi <= pi) and cos(11) == cos(-11 + 4*pi)
    assert asec(csc(13)) == -13 + pi*Rational(9, 2) and (0 <= -13 + pi*Rational(9, 2) <= pi) and sin(13) == cos(-13 + pi*Rational(9, 2))
    assert acsc(csc(14)) == -4*pi + 14 and (-pi/2 <= -4*pi + 14 <= pi/2) and sin(14) == sin(-4*pi + 14)
    assert acsc(sec(10)) == pi*Rational(-7, 2) + 10 and (-pi/2 <= pi*Rational(-7, 2) + 10 <= pi/2) and cos(10) == sin(pi*Rational(-7, 2) + 10)

def test_issue_14543():
    assert sec(2*pi + 11) == sec(11)
    assert sec(2*pi - 11) == sec(11)
    assert sec(pi + 11) == -sec(11)
    assert sec(pi - 11) == -sec(11)

    assert csc(2*pi + 17) == csc(17)
    assert csc(2*pi - 17) == -csc(17)
    assert csc(pi + 17) == -csc(17)
    assert csc(pi - 17) == csc(17)

    x = Symbol('x')
    assert csc(pi/2 + x) == sec(x)
    assert csc(pi/2 - x) == sec(x)
    assert csc(pi*Rational(3, 2) + x) == -sec(x)
    assert csc(pi*Rational(3, 2) - x) == -sec(x)

    assert sec(pi/2 - x) == csc(x)
    assert sec(pi/2 + x) == -csc(x)
    assert sec(pi*Rational(3, 2) + x) == csc(x)
    assert sec(pi*Rational(3, 2) - x) == -csc(x)


def test_as_real_imag():
    # This is for https://github.com/sympy/sympy/issues/17142
    # If it start failing again in irrelevant builds or in the master
    # please open up the issue again.
    expr = atan(I/(I + I*tan(1)))
    assert expr.as_real_imag() == (expr, 0)


def test_issue_18746():
    e3 = cos(S.Pi*(x/4 + 1/4))
    assert e3.period() == 8


def test_issue_25833():
    assert limit(atan(x**2), x, oo) == pi/2
    assert limit(atan(x**2 - 1), x, oo) == pi/2
    assert limit(atan(log(2**x)/log(2*x)), x, oo) == pi/2


def test_issue_25847():
    #atan
    assert atan(sin(x)/x).as_leading_term(x) == pi/4
    raises(PoleError, lambda: atan(exp(1/x)).as_leading_term(x))

    #asin
    assert asin(sin(x)/x).as_leading_term(x) == pi/2
    raises(PoleError, lambda: asin(exp(1/x)).as_leading_term(x))

    #acos
    assert acos(sin(x)/x).as_leading_term(x) == 0
    raises(PoleError, lambda: acos(exp(1/x)).as_leading_term(x))

    #acot
    assert acot(sin(x)/x).as_leading_term(x) == pi/4
    raises(PoleError, lambda: acot(exp(1/x)).as_leading_term(x))

    #asec
    assert asec(sin(x)/x).as_leading_term(x) == 0
    raises(PoleError, lambda: asec(exp(1/x)).as_leading_term(x))

    #acsc
    assert acsc(sin(x)/x).as_leading_term(x) == pi/2
    raises(PoleError, lambda: acsc(exp(1/x)).as_leading_term(x))

def test_issue_23843():
    #atan
    assert atan(x + I).series(x, oo) == -16/(5*x**5) - 2*I/x**4 + 4/(3*x**3) + I/x**2 - 1/x + pi/2 + O(x**(-6), (x, oo))
    assert atan(x + I).series(x, -oo) == -16/(5*x**5) - 2*I/x**4 + 4/(3*x**3) + I/x**2 - 1/x - pi/2 + O(x**(-6), (x, -oo))
    assert atan(x - I).series(x, oo) == -16/(5*x**5) + 2*I/x**4 + 4/(3*x**3) - I/x**2 - 1/x + pi/2 + O(x**(-6), (x, oo))
    assert atan(x - I).series(x, -oo) == -16/(5*x**5) + 2*I/x**4 + 4/(3*x**3) - I/x**2 - 1/x - pi/2 + O(x**(-6), (x, -oo))

    #acot
    assert acot(x + I).series(x, oo) == 16/(5*x**5) + 2*I/x**4 - 4/(3*x**3) - I/x**2 + 1/x + O(x**(-6), (x, oo))
    assert acot(x + I).series(x, -oo) == 16/(5*x**5) + 2*I/x**4 - 4/(3*x**3) - I/x**2 + 1/x + O(x**(-6), (x, -oo))
    assert acot(x - I).series(x, oo) == 16/(5*x**5) - 2*I/x**4 - 4/(3*x**3) + I/x**2 + 1/x + O(x**(-6), (x, oo))
    assert acot(x - I).series(x, -oo) == 16/(5*x**5) - 2*I/x**4 - 4/(3*x**3) + I/x**2 + 1/x + O(x**(-6), (x, -oo))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\resample\test_datetime_index.py ===
from datetime import datetime
from functools import partial

import numpy as np
import pytest
import pytz

from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    Timedelta,
    Timestamp,
    isna,
    notna,
)
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
    Period,
    period_range,
)
from pandas.core.resample import (
    DatetimeIndex,
    _get_timestamp_range_edges,
)

from pandas.tseries import offsets
from pandas.tseries.offsets import Minute


@pytest.fixture()
def _index_factory():
    return date_range


@pytest.fixture
def _index_freq():
    return "Min"


@pytest.fixture
def _static_values(index):
    return np.random.default_rng(2).random(len(index))


@pytest.fixture(params=["s", "ms", "us", "ns"])
def unit(request):
    return request.param


@pytest.fixture
def simple_date_range_series():
    """
    Series with date range index and random data for test purposes.
    """

    def _simple_date_range_series(start, end, freq="D"):
        rng = date_range(start, end, freq=freq)
        return Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    return _simple_date_range_series


def test_custom_grouper(index, unit):
    dti = index.as_unit(unit)
    s = Series(np.array([1] * len(dti)), index=dti, dtype="int64")

    b = Grouper(freq=Minute(5))
    g = s.groupby(b)

    # check all cython functions work
    g.ohlc()  # doesn't use _cython_agg_general
    funcs = ["sum", "mean", "prod", "min", "max", "var"]
    for f in funcs:
        g._cython_agg_general(f, alt=None, numeric_only=True)

    b = Grouper(freq=Minute(5), closed="right", label="right")
    g = s.groupby(b)
    # check all cython functions work
    g.ohlc()  # doesn't use _cython_agg_general
    funcs = ["sum", "mean", "prod", "min", "max", "var"]
    for f in funcs:
        g._cython_agg_general(f, alt=None, numeric_only=True)

    assert g.ngroups == 2593
    assert notna(g.mean()).all()

    # construct expected val
    arr = [1] + [5] * 2592
    idx = dti[0:-1:5]
    idx = idx.append(dti[-1:])
    idx = DatetimeIndex(idx, freq="5min").as_unit(unit)
    expect = Series(arr, index=idx)

    # GH2763 - return input dtype if we can
    result = g.agg("sum")
    tm.assert_series_equal(result, expect)


def test_custom_grouper_df(index, unit):
    b = Grouper(freq=Minute(5), closed="right", label="right")
    dti = index.as_unit(unit)
    df = DataFrame(
        np.random.default_rng(2).random((len(dti), 10)), index=dti, dtype="float64"
    )
    r = df.groupby(b).agg("sum")

    assert len(r.columns) == 10
    assert len(r.index) == 2593


@pytest.mark.parametrize(
    "_index_start,_index_end,_index_name",
    [("1/1/2000 00:00:00", "1/1/2000 00:13:00", "index")],
)
@pytest.mark.parametrize(
    "closed, expected",
    [
        (
            "right",
            lambda s: Series(
                [s.iloc[0], s[1:6].mean(), s[6:11].mean(), s[11:].mean()],
                index=date_range("1/1/2000", periods=4, freq="5min", name="index"),
            ),
        ),
        (
            "left",
            lambda s: Series(
                [s[:5].mean(), s[5:10].mean(), s[10:].mean()],
                index=date_range(
                    "1/1/2000 00:05", periods=3, freq="5min", name="index"
                ),
            ),
        ),
    ],
)
def test_resample_basic(series, closed, expected, unit):
    s = series
    s.index = s.index.as_unit(unit)
    expected = expected(s)
    expected.index = expected.index.as_unit(unit)
    result = s.resample("5min", closed=closed, label="right").mean()
    tm.assert_series_equal(result, expected)


def test_resample_integerarray(unit):
    # GH 25580, resample on IntegerArray
    ts = Series(
        range(9),
        index=date_range("1/1/2000", periods=9, freq="min").as_unit(unit),
        dtype="Int64",
    )
    result = ts.resample("3min").sum()
    expected = Series(
        [3, 12, 21],
        index=date_range("1/1/2000", periods=3, freq="3min").as_unit(unit),
        dtype="Int64",
    )
    tm.assert_series_equal(result, expected)

    result = ts.resample("3min").mean()
    expected = Series(
        [1, 4, 7],
        index=date_range("1/1/2000", periods=3, freq="3min").as_unit(unit),
        dtype="Float64",
    )
    tm.assert_series_equal(result, expected)


def test_resample_basic_grouper(series, unit):
    s = series
    s.index = s.index.as_unit(unit)
    result = s.resample("5Min").last()
    grouper = Grouper(freq=Minute(5), closed="left", label="left")
    expected = s.groupby(grouper).agg(lambda x: x.iloc[-1])
    tm.assert_series_equal(result, expected)


@pytest.mark.filterwarnings(
    "ignore:The 'convention' keyword in Series.resample:FutureWarning"
)
@pytest.mark.parametrize(
    "_index_start,_index_end,_index_name",
    [("1/1/2000 00:00:00", "1/1/2000 00:13:00", "index")],
)
@pytest.mark.parametrize(
    "keyword,value",
    [("label", "righttt"), ("closed", "righttt"), ("convention", "starttt")],
)
def test_resample_string_kwargs(series, keyword, value, unit):
    # see gh-19303
    # Check that wrong keyword argument strings raise an error
    series.index = series.index.as_unit(unit)
    msg = f"Unsupported value {value} for `{keyword}`"
    with pytest.raises(ValueError, match=msg):
        series.resample("5min", **({keyword: value}))


@pytest.mark.parametrize(
    "_index_start,_index_end,_index_name",
    [("1/1/2000 00:00:00", "1/1/2000 00:13:00", "index")],
)
def test_resample_how(series, downsample_method, unit):
    if downsample_method == "ohlc":
        pytest.skip("covered by test_resample_how_ohlc")

    s = series
    s.index = s.index.as_unit(unit)
    grouplist = np.ones_like(s)
    grouplist[0] = 0
    grouplist[1:6] = 1
    grouplist[6:11] = 2
    grouplist[11:] = 3
    expected = s.groupby(grouplist).agg(downsample_method)
    expected.index = date_range(
        "1/1/2000", periods=4, freq="5min", name="index"
    ).as_unit(unit)

    result = getattr(
        s.resample("5min", closed="right", label="right"), downsample_method
    )()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "_index_start,_index_end,_index_name",
    [("1/1/2000 00:00:00", "1/1/2000 00:13:00", "index")],
)
def test_resample_how_ohlc(series, unit):
    s = series
    s.index = s.index.as_unit(unit)
    grouplist = np.ones_like(s)
    grouplist[0] = 0
    grouplist[1:6] = 1
    grouplist[6:11] = 2
    grouplist[11:] = 3

    def _ohlc(group):
        if isna(group).all():
            return np.repeat(np.nan, 4)
        return [group.iloc[0], group.max(), group.min(), group.iloc[-1]]

    expected = DataFrame(
        s.groupby(grouplist).agg(_ohlc).values.tolist(),
        index=date_range("1/1/2000", periods=4, freq="5min", name="index").as_unit(
            unit
        ),
        columns=["open", "high", "low", "close"],
    )

    result = s.resample("5min", closed="right", label="right").ohlc()
    tm.assert_frame_equal(result, expected)


def test_resample_how_callables(unit):
    # GH#7929
    data = np.arange(5, dtype=np.int64)
    ind = date_range(start="2014-01-01", periods=len(data), freq="d").as_unit(unit)
    df = DataFrame({"A": data, "B": data}, index=ind)

    def fn(x, a=1):
        return str(type(x))

    class FnClass:
        def __call__(self, x):
            return str(type(x))

    df_standard = df.resample("ME").apply(fn)
    df_lambda = df.resample("ME").apply(lambda x: str(type(x)))
    df_partial = df.resample("ME").apply(partial(fn))
    df_partial2 = df.resample("ME").apply(partial(fn, a=2))
    df_class = df.resample("ME").apply(FnClass())

    tm.assert_frame_equal(df_standard, df_lambda)
    tm.assert_frame_equal(df_standard, df_partial)
    tm.assert_frame_equal(df_standard, df_partial2)
    tm.assert_frame_equal(df_standard, df_class)


def test_resample_rounding(unit):
    # GH 8371
    # odd results when rounding is needed

    ts = [
        "2014-11-08 00:00:01",
        "2014-11-08 00:00:02",
        "2014-11-08 00:00:02",
        "2014-11-08 00:00:03",
        "2014-11-08 00:00:07",
        "2014-11-08 00:00:07",
        "2014-11-08 00:00:08",
        "2014-11-08 00:00:08",
        "2014-11-08 00:00:08",
        "2014-11-08 00:00:09",
        "2014-11-08 00:00:10",
        "2014-11-08 00:00:11",
        "2014-11-08 00:00:11",
        "2014-11-08 00:00:13",
        "2014-11-08 00:00:14",
        "2014-11-08 00:00:15",
        "2014-11-08 00:00:17",
        "2014-11-08 00:00:20",
        "2014-11-08 00:00:21",
    ]
    df = DataFrame({"value": [1] * 19}, index=pd.to_datetime(ts))
    df.index = df.index.as_unit(unit)

    result = df.resample("6s").sum()
    expected = DataFrame(
        {"value": [4, 9, 4, 2]},
        index=date_range("2014-11-08", freq="6s", periods=4).as_unit(unit),
    )
    tm.assert_frame_equal(result, expected)

    result = df.resample("7s").sum()
    expected = DataFrame(
        {"value": [4, 10, 4, 1]},
        index=date_range("2014-11-08", freq="7s", periods=4).as_unit(unit),
    )
    tm.assert_frame_equal(result, expected)

    result = df.resample("11s").sum()
    expected = DataFrame(
        {"value": [11, 8]},
        index=date_range("2014-11-08", freq="11s", periods=2).as_unit(unit),
    )
    tm.assert_frame_equal(result, expected)

    result = df.resample("13s").sum()
    expected = DataFrame(
        {"value": [13, 6]},
        index=date_range("2014-11-08", freq="13s", periods=2).as_unit(unit),
    )
    tm.assert_frame_equal(result, expected)

    result = df.resample("17s").sum()
    expected = DataFrame(
        {"value": [16, 3]},
        index=date_range("2014-11-08", freq="17s", periods=2).as_unit(unit),
    )
    tm.assert_frame_equal(result, expected)


def test_resample_basic_from_daily(unit):
    # from daily
    dti = date_range(
        start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq="D", name="index"
    ).as_unit(unit)

    s = Series(np.random.default_rng(2).random(len(dti)), dti)

    # to weekly
    result = s.resample("w-sun").last()

    assert len(result) == 3
    assert (result.index.dayofweek == [6, 6, 6]).all()
    assert result.iloc[0] == s["1/2/2005"]
    assert result.iloc[1] == s["1/9/2005"]
    assert result.iloc[2] == s.iloc[-1]

    result = s.resample("W-MON").last()
    assert len(result) == 2
    assert (result.index.dayofweek == [0, 0]).all()
    assert result.iloc[0] == s["1/3/2005"]
    assert result.iloc[1] == s["1/10/2005"]

    result = s.resample("W-TUE").last()
    assert len(result) == 2
    assert (result.index.dayofweek == [1, 1]).all()
    assert result.iloc[0] == s["1/4/2005"]
    assert result.iloc[1] == s["1/10/2005"]

    result = s.resample("W-WED").last()
    assert len(result) == 2
    assert (result.index.dayofweek == [2, 2]).all()
    assert result.iloc[0] == s["1/5/2005"]
    assert result.iloc[1] == s["1/10/2005"]

    result = s.resample("W-THU").last()
    assert len(result) == 2
    assert (result.index.dayofweek == [3, 3]).all()
    assert result.iloc[0] == s["1/6/2005"]
    assert result.iloc[1] == s["1/10/2005"]

    result = s.resample("W-FRI").last()
    assert len(result) == 2
    assert (result.index.dayofweek == [4, 4]).all()
    assert result.iloc[0] == s["1/7/2005"]
    assert result.iloc[1] == s["1/10/2005"]

    # to biz day
    result = s.resample("B").last()
    assert len(result) == 7
    assert (result.index.dayofweek == [4, 0, 1, 2, 3, 4, 0]).all()

    assert result.iloc[0] == s["1/2/2005"]
    assert result.iloc[1] == s["1/3/2005"]
    assert result.iloc[5] == s["1/9/2005"]
    assert result.index.name == "index"


def test_resample_upsampling_picked_but_not_correct(unit):
    # Test for issue #3020
    dates = date_range("01-Jan-2014", "05-Jan-2014", freq="D").as_unit(unit)
    series = Series(1, index=dates)

    result = series.resample("D").mean()
    assert result.index[0] == dates[0]

    # GH 5955
    # incorrect deciding to upsample when the axis frequency matches the
    # resample frequency

    s = Series(
        np.arange(1.0, 6), index=[datetime(1975, 1, i, 12, 0) for i in range(1, 6)]
    )
    s.index = s.index.as_unit(unit)
    expected = Series(
        np.arange(1.0, 6),
        index=date_range("19750101", periods=5, freq="D").as_unit(unit),
    )

    result = s.resample("D").count()
    tm.assert_series_equal(result, Series(1, index=expected.index))

    result1 = s.resample("D").sum()
    result2 = s.resample("D").mean()
    tm.assert_series_equal(result1, expected)
    tm.assert_series_equal(result2, expected)


@pytest.mark.parametrize("f", ["sum", "mean", "prod", "min", "max", "var"])
def test_resample_frame_basic_cy_funcs(f, unit):
    df = DataFrame(
        np.random.default_rng(2).standard_normal((50, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=50, freq="B"),
    )
    df.index = df.index.as_unit(unit)

    b = Grouper(freq="ME")
    g = df.groupby(b)

    # check all cython functions work
    g._cython_agg_general(f, alt=None, numeric_only=True)


@pytest.mark.parametrize("freq", ["YE", "ME"])
def test_resample_frame_basic_M_A(freq, unit):
    df = DataFrame(
        np.random.default_rng(2).standard_normal((50, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=50, freq="B"),
    )
    df.index = df.index.as_unit(unit)
    result = df.resample(freq).mean()
    tm.assert_series_equal(result["A"], df["A"].resample(freq).mean())


@pytest.mark.parametrize("freq", ["W-WED", "ME"])
def test_resample_frame_basic_kind(freq, unit):
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    df.index = df.index.as_unit(unit)
    msg = "The 'kind' keyword in DataFrame.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.resample(freq, kind="period").mean()


def test_resample_upsample(unit):
    # from daily
    dti = date_range(
        start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq="D", name="index"
    ).as_unit(unit)

    s = Series(np.random.default_rng(2).random(len(dti)), dti)

    # to minutely, by padding
    result = s.resample("Min").ffill()
    assert len(result) == 12961
    assert result.iloc[0] == s.iloc[0]
    assert result.iloc[-1] == s.iloc[-1]

    assert result.index.name == "index"


def test_resample_how_method(unit):
    # GH9915
    s = Series(
        [11, 22],
        index=[
            Timestamp("2015-03-31 21:48:52.672000"),
            Timestamp("2015-03-31 21:49:52.739000"),
        ],
    )
    s.index = s.index.as_unit(unit)
    expected = Series(
        [11, np.nan, np.nan, np.nan, np.nan, np.nan, 22],
        index=DatetimeIndex(
            [
                Timestamp("2015-03-31 21:48:50"),
                Timestamp("2015-03-31 21:49:00"),
                Timestamp("2015-03-31 21:49:10"),
                Timestamp("2015-03-31 21:49:20"),
                Timestamp("2015-03-31 21:49:30"),
                Timestamp("2015-03-31 21:49:40"),
                Timestamp("2015-03-31 21:49:50"),
            ],
            freq="10s",
        ),
    )
    expected.index = expected.index.as_unit(unit)
    tm.assert_series_equal(s.resample("10s").mean(), expected)


def test_resample_extra_index_point(unit):
    # GH#9756
    index = date_range(start="20150101", end="20150331", freq="BME").as_unit(unit)
    expected = DataFrame({"A": Series([21, 41, 63], index=index)})

    index = date_range(start="20150101", end="20150331", freq="B").as_unit(unit)
    df = DataFrame({"A": Series(range(len(index)), index=index)}, dtype="int64")
    result = df.resample("BME").last()
    tm.assert_frame_equal(result, expected)


def test_upsample_with_limit(unit):
    rng = date_range("1/1/2000", periods=3, freq="5min").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)

    result = ts.resample("min").ffill(limit=2)
    expected = ts.reindex(result.index, method="ffill", limit=2)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("freq", ["1D", "10h", "5Min", "10s"])
@pytest.mark.parametrize("rule", ["YE", "3ME", "15D", "30h", "15Min", "30s"])
def test_nearest_upsample_with_limit(tz_aware_fixture, freq, rule, unit):
    # GH 33939
    rng = date_range("1/1/2000", periods=3, freq=freq, tz=tz_aware_fixture).as_unit(
        unit
    )
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)

    result = ts.resample(rule).nearest(limit=2)
    expected = ts.reindex(result.index, method="nearest", limit=2)
    tm.assert_series_equal(result, expected)


def test_resample_ohlc(series, unit):
    s = series
    s.index = s.index.as_unit(unit)

    grouper = Grouper(freq=Minute(5))
    expect = s.groupby(grouper).agg(lambda x: x.iloc[-1])
    result = s.resample("5Min").ohlc()

    assert len(result) == len(expect)
    assert len(result.columns) == 4

    xs = result.iloc[-2]
    assert xs["open"] == s.iloc[-6]
    assert xs["high"] == s[-6:-1].max()
    assert xs["low"] == s[-6:-1].min()
    assert xs["close"] == s.iloc[-2]

    xs = result.iloc[0]
    assert xs["open"] == s.iloc[0]
    assert xs["high"] == s[:5].max()
    assert xs["low"] == s[:5].min()
    assert xs["close"] == s.iloc[4]


def test_resample_ohlc_result(unit):
    # GH 12332
    index = date_range("1-1-2000", "2-15-2000", freq="h").as_unit(unit)
    index = index.union(date_range("4-15-2000", "5-15-2000", freq="h").as_unit(unit))
    s = Series(range(len(index)), index=index)

    a = s.loc[:"4-15-2000"].resample("30min").ohlc()
    assert isinstance(a, DataFrame)

    b = s.loc[:"4-14-2000"].resample("30min").ohlc()
    assert isinstance(b, DataFrame)


def test_resample_ohlc_result_odd_period(unit):
    # GH12348
    # raising on odd period
    rng = date_range("2013-12-30", "2014-01-07").as_unit(unit)
    index = rng.drop(
        [
            Timestamp("2014-01-01"),
            Timestamp("2013-12-31"),
            Timestamp("2014-01-04"),
            Timestamp("2014-01-05"),
        ]
    )
    df = DataFrame(data=np.arange(len(index)), index=index)
    result = df.resample("B").mean()
    expected = df.reindex(index=date_range(rng[0], rng[-1], freq="B").as_unit(unit))
    tm.assert_frame_equal(result, expected)


def test_resample_ohlc_dataframe(unit):
    df = (
        DataFrame(
            {
                "PRICE": {
                    Timestamp("2011-01-06 10:59:05", tz=None): 24990,
                    Timestamp("2011-01-06 12:43:33", tz=None): 25499,
                    Timestamp("2011-01-06 12:54:09", tz=None): 25499,
                },
                "VOLUME": {
                    Timestamp("2011-01-06 10:59:05", tz=None): 1500000000,
                    Timestamp("2011-01-06 12:43:33", tz=None): 5000000000,
                    Timestamp("2011-01-06 12:54:09", tz=None): 100000000,
                },
            }
        )
    ).reindex(["VOLUME", "PRICE"], axis=1)
    df.index = df.index.as_unit(unit)
    df.columns.name = "Cols"
    res = df.resample("h").ohlc()
    exp = pd.concat(
        [df["VOLUME"].resample("h").ohlc(), df["PRICE"].resample("h").ohlc()],
        axis=1,
        keys=df.columns,
    )
    assert exp.columns.names[0] == "Cols"
    tm.assert_frame_equal(exp, res)

    df.columns = [["a", "b"], ["c", "d"]]
    res = df.resample("h").ohlc()
    exp.columns = pd.MultiIndex.from_tuples(
        [
            ("a", "c", "open"),
            ("a", "c", "high"),
            ("a", "c", "low"),
            ("a", "c", "close"),
            ("b", "d", "open"),
            ("b", "d", "high"),
            ("b", "d", "low"),
            ("b", "d", "close"),
        ]
    )
    tm.assert_frame_equal(exp, res)

    # dupe columns fail atm
    # df.columns = ['PRICE', 'PRICE']


def test_resample_dup_index():
    # GH 4812
    # dup columns with resample raising
    df = DataFrame(
        np.random.default_rng(2).standard_normal((4, 12)),
        index=[2000, 2000, 2000, 2000],
        columns=[Period(year=2000, month=i + 1, freq="M") for i in range(12)],
    )
    df.iloc[3, :] = np.nan
    warning_msg = "DataFrame.resample with axis=1 is deprecated."
    with tm.assert_produces_warning(FutureWarning, match=warning_msg):
        result = df.resample("QE", axis=1).mean()

    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.groupby(lambda x: int((x.month - 1) / 3), axis=1).mean()
    expected.columns = [Period(year=2000, quarter=i + 1, freq="Q") for i in range(4)]
    tm.assert_frame_equal(result, expected)


def test_resample_reresample(unit):
    dti = date_range(
        start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq="D"
    ).as_unit(unit)
    s = Series(np.random.default_rng(2).random(len(dti)), dti)
    bs = s.resample("B", closed="right", label="right").mean()
    result = bs.resample("8h").mean()
    assert len(result) == 25
    assert isinstance(result.index.freq, offsets.DateOffset)
    assert result.index.freq == offsets.Hour(8)


@pytest.mark.parametrize(
    "freq, expected_kwargs",
    [
        ["YE-DEC", {"start": "1990", "end": "2000", "freq": "Y-DEC"}],
        ["YE-JUN", {"start": "1990", "end": "2000", "freq": "Y-JUN"}],
        ["ME", {"start": "1990-01", "end": "2000-01", "freq": "M"}],
    ],
)
def test_resample_timestamp_to_period(
    simple_date_range_series, freq, expected_kwargs, unit
):
    ts = simple_date_range_series("1/1/1990", "1/1/2000")
    ts.index = ts.index.as_unit(unit)

    msg = "The 'kind' keyword in Series.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ts.resample(freq, kind="period").mean()
    expected = ts.resample(freq).mean()
    expected.index = period_range(**expected_kwargs)
    tm.assert_series_equal(result, expected)


def test_ohlc_5min(unit):
    def _ohlc(group):
        if isna(group).all():
            return np.repeat(np.nan, 4)
        return [group.iloc[0], group.max(), group.min(), group.iloc[-1]]

    rng = date_range("1/1/2000 00:00:00", "1/1/2000 5:59:50", freq="10s").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    resampled = ts.resample("5min", closed="right", label="right").ohlc()

    assert (resampled.loc["1/1/2000 00:00"] == ts.iloc[0]).all()

    exp = _ohlc(ts[1:31])
    assert (resampled.loc["1/1/2000 00:05"] == exp).all()

    exp = _ohlc(ts["1/1/2000 5:55:01":])
    assert (resampled.loc["1/1/2000 6:00:00"] == exp).all()


def test_downsample_non_unique(unit):
    rng = date_range("1/1/2000", "2/29/2000").as_unit(unit)
    rng2 = rng.repeat(5).values
    ts = Series(np.random.default_rng(2).standard_normal(len(rng2)), index=rng2)

    result = ts.resample("ME").mean()

    expected = ts.groupby(lambda x: x.month).mean()
    assert len(result) == 2
    tm.assert_almost_equal(result.iloc[0], expected[1])
    tm.assert_almost_equal(result.iloc[1], expected[2])


def test_asfreq_non_unique(unit):
    # GH #1077
    rng = date_range("1/1/2000", "2/29/2000").as_unit(unit)
    rng2 = rng.repeat(2).values
    ts = Series(np.random.default_rng(2).standard_normal(len(rng2)), index=rng2)

    msg = "cannot reindex on an axis with duplicate labels"
    with pytest.raises(ValueError, match=msg):
        ts.asfreq("B")


def test_resample_axis1(unit):
    rng = date_range("1/1/2000", "2/29/2000").as_unit(unit)
    df = DataFrame(
        np.random.default_rng(2).standard_normal((3, len(rng))),
        columns=rng,
        index=["a", "b", "c"],
    )

    warning_msg = "DataFrame.resample with axis=1 is deprecated."
    with tm.assert_produces_warning(FutureWarning, match=warning_msg):
        result = df.resample("ME", axis=1).mean()
    expected = df.T.resample("ME").mean().T
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("freq", ["min", "5min", "15min", "30min", "4h", "12h"])
def test_resample_anchored_ticks(freq, unit):
    # If a fixed delta (5 minute, 4 hour) evenly divides a day, we should
    # "anchor" the origin at midnight so we get regular intervals rather
    # than starting from the first timestamp which might start in the
    # middle of a desired interval

    rng = date_range("1/1/2000 04:00:00", periods=86400, freq="s").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    ts[:2] = np.nan  # so results are the same
    result = ts[2:].resample(freq, closed="left", label="left").mean()
    expected = ts.resample(freq, closed="left", label="left").mean()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("end", [1, 2])
def test_resample_single_group(end, unit):
    mysum = lambda x: x.sum()

    rng = date_range("2000-1-1", f"2000-{end}-10", freq="D").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    tm.assert_series_equal(ts.resample("ME").sum(), ts.resample("ME").apply(mysum))


def test_resample_single_group_std(unit):
    # GH 3849
    s = Series(
        [30.1, 31.6],
        index=[Timestamp("20070915 15:30:00"), Timestamp("20070915 15:40:00")],
    )
    s.index = s.index.as_unit(unit)
    expected = Series(
        [0.75], index=DatetimeIndex([Timestamp("20070915")], freq="D").as_unit(unit)
    )
    result = s.resample("D").apply(lambda x: np.std(x))
    tm.assert_series_equal(result, expected)


def test_resample_offset(unit):
    # GH 31809

    rng = date_range("1/1/2000 00:00:00", "1/1/2000 02:00", freq="s").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    resampled = ts.resample("5min", offset="2min").mean()
    exp_rng = date_range("12/31/1999 23:57:00", "1/1/2000 01:57", freq="5min").as_unit(
        unit
    )
    tm.assert_index_equal(resampled.index, exp_rng)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"origin": "1999-12-31 23:57:00"},
        {"origin": Timestamp("1970-01-01 00:02:00")},
        {"origin": "epoch", "offset": "2m"},
        # origin of '1999-31-12 12:02:00' should be equivalent for this case
        {"origin": "1999-12-31 12:02:00"},
        {"offset": "-3m"},
    ],
)
def test_resample_origin(kwargs, unit):
    # GH 31809
    rng = date_range("2000-01-01 00:00:00", "2000-01-01 02:00", freq="s").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    exp_rng = date_range(
        "1999-12-31 23:57:00", "2000-01-01 01:57", freq="5min"
    ).as_unit(unit)

    resampled = ts.resample("5min", **kwargs).mean()
    tm.assert_index_equal(resampled.index, exp_rng)


@pytest.mark.parametrize(
    "origin", ["invalid_value", "epch", "startday", "startt", "2000-30-30", object()]
)
def test_resample_bad_origin(origin, unit):
    rng = date_range("2000-01-01 00:00:00", "2000-01-01 02:00", freq="s").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    msg = (
        "'origin' should be equal to 'epoch', 'start', 'start_day', "
        "'end', 'end_day' or should be a Timestamp convertible type. Got "
        f"'{origin}' instead."
    )
    with pytest.raises(ValueError, match=msg):
        ts.resample("5min", origin=origin)


@pytest.mark.parametrize("offset", ["invalid_value", "12dayys", "2000-30-30", object()])
def test_resample_bad_offset(offset, unit):
    rng = date_range("2000-01-01 00:00:00", "2000-01-01 02:00", freq="s").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    msg = f"'offset' should be a Timedelta convertible type. Got '{offset}' instead."
    with pytest.raises(ValueError, match=msg):
        ts.resample("5min", offset=offset)


def test_resample_origin_prime_freq(unit):
    # GH 31809
    start, end = "2000-10-01 23:30:00", "2000-10-02 00:30:00"
    rng = date_range(start, end, freq="7min").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    exp_rng = date_range(
        "2000-10-01 23:14:00", "2000-10-02 00:22:00", freq="17min"
    ).as_unit(unit)
    resampled = ts.resample("17min").mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    resampled = ts.resample("17min", origin="start_day").mean()
    tm.assert_index_equal(resampled.index, exp_rng)

    exp_rng = date_range(
        "2000-10-01 23:30:00", "2000-10-02 00:21:00", freq="17min"
    ).as_unit(unit)
    resampled = ts.resample("17min", origin="start").mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    resampled = ts.resample("17min", offset="23h30min").mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    resampled = ts.resample("17min", origin="start_day", offset="23h30min").mean()
    tm.assert_index_equal(resampled.index, exp_rng)

    exp_rng = date_range(
        "2000-10-01 23:18:00", "2000-10-02 00:26:00", freq="17min"
    ).as_unit(unit)
    resampled = ts.resample("17min", origin="epoch").mean()
    tm.assert_index_equal(resampled.index, exp_rng)

    exp_rng = date_range(
        "2000-10-01 23:24:00", "2000-10-02 00:15:00", freq="17min"
    ).as_unit(unit)
    resampled = ts.resample("17min", origin="2000-01-01").mean()
    tm.assert_index_equal(resampled.index, exp_rng)


def test_resample_origin_with_tz(unit):
    # GH 31809
    msg = "The origin must have the same timezone as the index."

    tz = "Europe/Paris"
    rng = date_range(
        "2000-01-01 00:00:00", "2000-01-01 02:00", freq="s", tz=tz
    ).as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    exp_rng = date_range(
        "1999-12-31 23:57:00", "2000-01-01 01:57", freq="5min", tz=tz
    ).as_unit(unit)
    resampled = ts.resample("5min", origin="1999-12-31 23:57:00+00:00").mean()
    tm.assert_index_equal(resampled.index, exp_rng)

    # origin of '1999-31-12 12:02:00+03:00' should be equivalent for this case
    resampled = ts.resample("5min", origin="1999-12-31 12:02:00+03:00").mean()
    tm.assert_index_equal(resampled.index, exp_rng)

    resampled = ts.resample("5min", origin="epoch", offset="2m").mean()
    tm.assert_index_equal(resampled.index, exp_rng)

    with pytest.raises(ValueError, match=msg):
        ts.resample("5min", origin="12/31/1999 23:57:00").mean()

    # if the series is not tz aware, origin should not be tz aware
    rng = date_range("2000-01-01 00:00:00", "2000-01-01 02:00", freq="s").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    with pytest.raises(ValueError, match=msg):
        ts.resample("5min", origin="12/31/1999 23:57:00+03:00").mean()


def test_resample_origin_epoch_with_tz_day_vs_24h(unit):
    # GH 34474
    start, end = "2000-10-01 23:30:00+0500", "2000-12-02 00:30:00+0500"
    rng = date_range(start, end, freq="7min").as_unit(unit)
    random_values = np.random.default_rng(2).standard_normal(len(rng))
    ts_1 = Series(random_values, index=rng)

    result_1 = ts_1.resample("D", origin="epoch").mean()
    result_2 = ts_1.resample("24h", origin="epoch").mean()
    tm.assert_series_equal(result_1, result_2)

    # check that we have the same behavior with epoch even if we are not timezone aware
    ts_no_tz = ts_1.tz_localize(None)
    result_3 = ts_no_tz.resample("D", origin="epoch").mean()
    result_4 = ts_no_tz.resample("24h", origin="epoch").mean()
    tm.assert_series_equal(result_1, result_3.tz_localize(rng.tz), check_freq=False)
    tm.assert_series_equal(result_1, result_4.tz_localize(rng.tz), check_freq=False)

    # check that we have the similar results with two different timezones (+2H and +5H)
    start, end = "2000-10-01 23:30:00+0200", "2000-12-02 00:30:00+0200"
    rng = date_range(start, end, freq="7min").as_unit(unit)
    ts_2 = Series(random_values, index=rng)
    result_5 = ts_2.resample("D", origin="epoch").mean()
    result_6 = ts_2.resample("24h", origin="epoch").mean()
    tm.assert_series_equal(result_1.tz_localize(None), result_5.tz_localize(None))
    tm.assert_series_equal(result_1.tz_localize(None), result_6.tz_localize(None))


def test_resample_origin_with_day_freq_on_dst(unit):
    # GH 31809
    tz = "America/Chicago"

    def _create_series(values, timestamps, freq="D"):
        return Series(
            values,
            index=DatetimeIndex(
                [Timestamp(t, tz=tz) for t in timestamps], freq=freq, ambiguous=True
            ).as_unit(unit),
        )

    # test classical behavior of origin in a DST context
    start = Timestamp("2013-11-02", tz=tz)
    end = Timestamp("2013-11-03 23:59", tz=tz)
    rng = date_range(start, end, freq="1h").as_unit(unit)
    ts = Series(np.ones(len(rng)), index=rng)

    expected = _create_series([24.0, 25.0], ["2013-11-02", "2013-11-03"])
    for origin in ["epoch", "start", "start_day", start, None]:
        result = ts.resample("D", origin=origin).sum()
        tm.assert_series_equal(result, expected)

    # test complex behavior of origin/offset in a DST context
    start = Timestamp("2013-11-03", tz=tz)
    end = Timestamp("2013-11-03 23:59", tz=tz)
    rng = date_range(start, end, freq="1h").as_unit(unit)
    ts = Series(np.ones(len(rng)), index=rng)

    expected_ts = ["2013-11-02 22:00-05:00", "2013-11-03 22:00-06:00"]
    expected = _create_series([23.0, 2.0], expected_ts)
    result = ts.resample("D", origin="start", offset="-2h").sum()
    tm.assert_series_equal(result, expected)

    expected_ts = ["2013-11-02 22:00-05:00", "2013-11-03 21:00-06:00"]
    expected = _create_series([22.0, 3.0], expected_ts, freq="24h")
    result = ts.resample("24h", origin="start", offset="-2h").sum()
    tm.assert_series_equal(result, expected)

    expected_ts = ["2013-11-02 02:00-05:00", "2013-11-03 02:00-06:00"]
    expected = _create_series([3.0, 22.0], expected_ts)
    result = ts.resample("D", origin="start", offset="2h").sum()
    tm.assert_series_equal(result, expected)

    expected_ts = ["2013-11-02 23:00-05:00", "2013-11-03 23:00-06:00"]
    expected = _create_series([24.0, 1.0], expected_ts)
    result = ts.resample("D", origin="start", offset="-1h").sum()
    tm.assert_series_equal(result, expected)

    expected_ts = ["2013-11-02 01:00-05:00", "2013-11-03 01:00:00-0500"]
    expected = _create_series([1.0, 24.0], expected_ts)
    result = ts.resample("D", origin="start", offset="1h").sum()
    tm.assert_series_equal(result, expected)


def test_resample_daily_anchored(unit):
    rng = date_range("1/1/2000 0:00:00", periods=10000, freq="min").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    ts[:2] = np.nan  # so results are the same

    result = ts[2:].resample("D", closed="left", label="left").mean()
    expected = ts.resample("D", closed="left", label="left").mean()
    tm.assert_series_equal(result, expected)


def test_resample_to_period_monthly_buglet(unit):
    # GH #1259

    rng = date_range("1/1/2000", "12/31/2000").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    msg = "The 'kind' keyword in Series.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ts.resample("ME", kind="period").mean()
    exp_index = period_range("Jan-2000", "Dec-2000", freq="M")
    tm.assert_index_equal(result.index, exp_index)


def test_period_with_agg():
    # aggregate a period resampler with a lambda
    s2 = Series(
        np.random.default_rng(2).integers(0, 5, 50),
        index=period_range("2012-01-01", freq="h", periods=50),
        dtype="float64",
    )

    expected = s2.to_timestamp().resample("D").mean().to_period()
    msg = "Resampling with a PeriodIndex is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = s2.resample("D")
    result = rs.agg(lambda x: x.mean())
    tm.assert_series_equal(result, expected)


def test_resample_segfault(unit):
    # GH 8573
    # segfaulting in older versions
    all_wins_and_wagers = [
        (1, datetime(2013, 10, 1, 16, 20), 1, 0),
        (2, datetime(2013, 10, 1, 16, 10), 1, 0),
        (2, datetime(2013, 10, 1, 18, 15), 1, 0),
        (2, datetime(2013, 10, 1, 16, 10, 31), 1, 0),
    ]

    df = DataFrame.from_records(
        all_wins_and_wagers, columns=("ID", "timestamp", "A", "B")
    ).set_index("timestamp")
    df.index = df.index.as_unit(unit)
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("ID").resample("5min").sum()
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.groupby("ID").apply(lambda x: x.resample("5min").sum())
    tm.assert_frame_equal(result, expected)


def test_resample_dtype_preservation(unit):
    # GH 12202
    # validation tests for dtype preservation

    df = DataFrame(
        {
            "date": date_range(start="2016-01-01", periods=4, freq="W").as_unit(unit),
            "group": [1, 1, 2, 2],
            "val": Series([5, 6, 7, 8], dtype="int32"),
        }
    ).set_index("date")

    result = df.resample("1D").ffill()
    assert result.val.dtype == np.int32

    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("group").resample("1D").ffill()
    assert result.val.dtype == np.int32


def test_resample_dtype_coercion(unit):
    pytest.importorskip("scipy.interpolate")

    # GH 16361
    df = {"a": [1, 3, 1, 4]}
    df = DataFrame(df, index=date_range("2017-01-01", "2017-01-04").as_unit(unit))

    expected = df.astype("float64").resample("h").mean()["a"].interpolate("cubic")

    result = df.resample("h")["a"].mean().interpolate("cubic")
    tm.assert_series_equal(result, expected)

    result = df.resample("h").mean()["a"].interpolate("cubic")
    tm.assert_series_equal(result, expected)


def test_weekly_resample_buglet(unit):
    # #1327
    rng = date_range("1/1/2000", freq="B", periods=20).as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    resampled = ts.resample("W").mean()
    expected = ts.resample("W-SUN").mean()
    tm.assert_series_equal(resampled, expected)


def test_monthly_resample_error(unit):
    # #1451
    dates = date_range("4/16/2012 20:00", periods=5000, freq="h").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(dates)), index=dates)
    # it works!
    ts.resample("ME")


def test_nanosecond_resample_error():
    # GH 12307 - Values falls after last bin when
    # Resampling using pd.tseries.offsets.Nano as period
    start = 1443707890427
    exp_start = 1443707890400
    indx = date_range(start=pd.to_datetime(start), periods=10, freq="100ns")
    ts = Series(range(len(indx)), index=indx)
    r = ts.resample(pd.tseries.offsets.Nano(100))
    result = r.agg("mean")

    exp_indx = date_range(start=pd.to_datetime(exp_start), periods=10, freq="100ns")
    exp = Series(range(len(exp_indx)), index=exp_indx, dtype=float)

    tm.assert_series_equal(result, exp)


def test_resample_anchored_intraday(unit):
    # #1471, #1458

    rng = date_range("1/1/2012", "4/1/2012", freq="100min").as_unit(unit)
    df = DataFrame(rng.month, index=rng)

    result = df.resample("ME").mean()
    msg = "The 'kind' keyword in DataFrame.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.resample("ME", kind="period").mean().to_timestamp(how="end")
    expected.index += Timedelta(1, "ns") - Timedelta(1, "D")
    expected.index = expected.index.as_unit(unit)._with_freq("infer")
    assert expected.index.freq == "ME"
    tm.assert_frame_equal(result, expected)

    result = df.resample("ME", closed="left").mean()
    msg = "The 'kind' keyword in DataFrame.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        exp = df.shift(1, freq="D").resample("ME", kind="period").mean()
    exp = exp.to_timestamp(how="end")

    exp.index = exp.index + Timedelta(1, "ns") - Timedelta(1, "D")
    exp.index = exp.index.as_unit(unit)._with_freq("infer")
    assert exp.index.freq == "ME"
    tm.assert_frame_equal(result, exp)


def test_resample_anchored_intraday2(unit):
    rng = date_range("1/1/2012", "4/1/2012", freq="100min").as_unit(unit)
    df = DataFrame(rng.month, index=rng)

    result = df.resample("QE").mean()
    msg = "The 'kind' keyword in DataFrame.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.resample("QE", kind="period").mean().to_timestamp(how="end")
    expected.index += Timedelta(1, "ns") - Timedelta(1, "D")
    expected.index._data.freq = "QE"
    expected.index._freq = lib.no_default
    expected.index = expected.index.as_unit(unit)
    tm.assert_frame_equal(result, expected)

    result = df.resample("QE", closed="left").mean()
    msg = "The 'kind' keyword in DataFrame.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = (
            df.shift(1, freq="D").resample("QE", kind="period", closed="left").mean()
        )
    expected = expected.to_timestamp(how="end")
    expected.index += Timedelta(1, "ns") - Timedelta(1, "D")
    expected.index._data.freq = "QE"
    expected.index._freq = lib.no_default
    expected.index = expected.index.as_unit(unit)
    tm.assert_frame_equal(result, expected)


def test_resample_anchored_intraday3(simple_date_range_series, unit):
    ts = simple_date_range_series("2012-04-29 23:00", "2012-04-30 5:00", freq="h")
    ts.index = ts.index.as_unit(unit)
    resampled = ts.resample("ME").mean()
    assert len(resampled) == 1


@pytest.mark.parametrize("freq", ["MS", "BMS", "QS-MAR", "YS-DEC", "YS-JUN"])
def test_resample_anchored_monthstart(simple_date_range_series, freq, unit):
    ts = simple_date_range_series("1/1/2000", "12/31/2002")
    ts.index = ts.index.as_unit(unit)
    ts.resample(freq).mean()


@pytest.mark.parametrize("label, sec", [[None, 2.0], ["right", "4.2"]])
def test_resample_anchored_multiday(label, sec):
    # When resampling a range spanning multiple days, ensure that the
    # start date gets used to determine the offset.  Fixes issue where
    # a one day period is not a multiple of the frequency.
    #
    # See: https://github.com/pandas-dev/pandas/issues/8683

    index1 = date_range("2014-10-14 23:06:23.206", periods=3, freq="400ms")
    index2 = date_range("2014-10-15 23:00:00", periods=2, freq="2200ms")
    index = index1.union(index2)

    s = Series(np.random.default_rng(2).standard_normal(5), index=index)

    # Ensure left closing works
    result = s.resample("2200ms", label=label).mean()
    assert result.index[-1] == Timestamp(f"2014-10-15 23:00:{sec}00")


def test_corner_cases(unit):
    # miscellaneous test coverage

    rng = date_range("1/1/2000", periods=12, freq="min").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    result = ts.resample("5min", closed="right", label="left").mean()
    ex_index = date_range("1999-12-31 23:55", periods=4, freq="5min").as_unit(unit)
    tm.assert_index_equal(result.index, ex_index)


def test_corner_cases_date(simple_date_range_series, unit):
    # resample to periods
    ts = simple_date_range_series("2000-04-28", "2000-04-30 11:00", freq="h")
    ts.index = ts.index.as_unit(unit)
    msg = "The 'kind' keyword in Series.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ts.resample("ME", kind="period").mean()
    assert len(result) == 1
    assert result.index[0] == Period("2000-04", freq="M")


def test_anchored_lowercase_buglet(unit):
    dates = date_range("4/16/2012 20:00", periods=50000, freq="s").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(dates)), index=dates)
    # it works!
    ts.resample("d").mean()


def test_upsample_apply_functions(unit):
    # #1596
    rng = date_range("2012-06-12", periods=4, freq="h").as_unit(unit)

    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    result = ts.resample("20min").aggregate(["mean", "sum"])
    assert isinstance(result, DataFrame)


def test_resample_not_monotonic(unit):
    rng = date_range("2012-06-12", periods=200, freq="h").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    ts = ts.take(np.random.default_rng(2).permutation(len(ts)))

    result = ts.resample("D").sum()
    exp = ts.sort_index().resample("D").sum()
    tm.assert_series_equal(result, exp)


@pytest.mark.parametrize(
    "dtype",
    [
        "int64",
        "int32",
        "float64",
        pytest.param(
            "float32",
            marks=pytest.mark.xfail(
                reason="Empty groups cause x.mean() to return float64"
            ),
        ),
    ],
)
def test_resample_median_bug_1688(dtype, unit):
    # GH#55958
    dti = DatetimeIndex(
        [datetime(2012, 1, 1, 0, 0, 0), datetime(2012, 1, 1, 0, 5, 0)]
    ).as_unit(unit)
    df = DataFrame(
        [1, 2],
        index=dti,
        dtype=dtype,
    )

    result = df.resample("min").apply(lambda x: x.mean())
    exp = df.asfreq("min")
    tm.assert_frame_equal(result, exp)

    result = df.resample("min").median()
    exp = df.asfreq("min")
    tm.assert_frame_equal(result, exp)


def test_how_lambda_functions(simple_date_range_series, unit):
    ts = simple_date_range_series("1/1/2000", "4/1/2000")
    ts.index = ts.index.as_unit(unit)

    result = ts.resample("ME").apply(lambda x: x.mean())
    exp = ts.resample("ME").mean()
    tm.assert_series_equal(result, exp)

    foo_exp = ts.resample("ME").mean()
    foo_exp.name = "foo"
    bar_exp = ts.resample("ME").std()
    bar_exp.name = "bar"

    result = ts.resample("ME").apply([lambda x: x.mean(), lambda x: x.std(ddof=1)])
    result.columns = ["foo", "bar"]
    tm.assert_series_equal(result["foo"], foo_exp)
    tm.assert_series_equal(result["bar"], bar_exp)

    # this is a MI Series, so comparing the names of the results
    # doesn't make sense
    result = ts.resample("ME").aggregate(
        {"foo": lambda x: x.mean(), "bar": lambda x: x.std(ddof=1)}
    )
    tm.assert_series_equal(result["foo"], foo_exp, check_names=False)
    tm.assert_series_equal(result["bar"], bar_exp, check_names=False)


def test_resample_unequal_times(unit):
    # #1772
    start = datetime(1999, 3, 1, 5)
    # end hour is less than start
    end = datetime(2012, 7, 31, 4)
    bad_ind = date_range(start, end, freq="30min").as_unit(unit)
    df = DataFrame({"close": 1}, index=bad_ind)

    # it works!
    df.resample("YS").sum()


def test_resample_consistency(unit):
    # GH 6418
    # resample with bfill / limit / reindex consistency

    i30 = date_range("2002-02-02", periods=4, freq="30min").as_unit(unit)
    s = Series(np.arange(4.0), index=i30)
    s.iloc[2] = np.nan

    # Upsample by factor 3 with reindex() and resample() methods:
    i10 = date_range(i30[0], i30[-1], freq="10min").as_unit(unit)

    s10 = s.reindex(index=i10, method="bfill")
    s10_2 = s.reindex(index=i10, method="bfill", limit=2)
    rl = s.reindex_like(s10, method="bfill", limit=2)
    r10_2 = s.resample("10Min").bfill(limit=2)
    r10 = s.resample("10Min").bfill()

    # s10_2, r10, r10_2, rl should all be equal
    tm.assert_series_equal(s10_2, r10)
    tm.assert_series_equal(s10_2, r10_2)
    tm.assert_series_equal(s10_2, rl)


dates1: list[DatetimeNaTType] = [
    datetime(2014, 10, 1),
    datetime(2014, 9, 3),
    datetime(2014, 11, 5),
    datetime(2014, 9, 5),
    datetime(2014, 10, 8),
    datetime(2014, 7, 15),
]

dates2: list[DatetimeNaTType] = (
    dates1[:2] + [pd.NaT] + dates1[2:4] + [pd.NaT] + dates1[4:]
)
dates3 = [pd.NaT] + dates1 + [pd.NaT]


@pytest.mark.parametrize("dates", [dates1, dates2, dates3])
def test_resample_timegrouper(dates, unit):
    # GH 7227
    dates = DatetimeIndex(dates).as_unit(unit)
    df = DataFrame({"A": dates, "B": np.arange(len(dates))})
    result = df.set_index("A").resample("ME").count()
    exp_idx = DatetimeIndex(
        ["2014-07-31", "2014-08-31", "2014-09-30", "2014-10-31", "2014-11-30"],
        freq="ME",
        name="A",
    ).as_unit(unit)
    expected = DataFrame({"B": [1, 0, 2, 2, 1]}, index=exp_idx)
    if df["A"].isna().any():
        expected.index = expected.index._with_freq(None)
    tm.assert_frame_equal(result, expected)

    result = df.groupby(Grouper(freq="ME", key="A")).count()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dates", [dates1, dates2, dates3])
def test_resample_timegrouper2(dates, unit):
    dates = DatetimeIndex(dates).as_unit(unit)

    df = DataFrame({"A": dates, "B": np.arange(len(dates)), "C": np.arange(len(dates))})
    result = df.set_index("A").resample("ME").count()

    exp_idx = DatetimeIndex(
        ["2014-07-31", "2014-08-31", "2014-09-30", "2014-10-31", "2014-11-30"],
        freq="ME",
        name="A",
    ).as_unit(unit)
    expected = DataFrame(
        {"B": [1, 0, 2, 2, 1], "C": [1, 0, 2, 2, 1]},
        index=exp_idx,
        columns=["B", "C"],
    )
    if df["A"].isna().any():
        expected.index = expected.index._with_freq(None)
    tm.assert_frame_equal(result, expected)

    result = df.groupby(Grouper(freq="ME", key="A")).count()
    tm.assert_frame_equal(result, expected)


def test_resample_nunique(unit):
    # GH 12352
    df = DataFrame(
        {
            "ID": {
                Timestamp("2015-06-05 00:00:00"): "0010100903",
                Timestamp("2015-06-08 00:00:00"): "0010150847",
            },
            "DATE": {
                Timestamp("2015-06-05 00:00:00"): "2015-06-05",
                Timestamp("2015-06-08 00:00:00"): "2015-06-08",
            },
        }
    )
    df.index = df.index.as_unit(unit)
    r = df.resample("D")
    g = df.groupby(Grouper(freq="D"))
    expected = df.groupby(Grouper(freq="D")).ID.apply(lambda x: x.nunique())
    assert expected.name == "ID"

    for t in [r, g]:
        result = t.ID.nunique()
        tm.assert_series_equal(result, expected)

    result = df.ID.resample("D").nunique()
    tm.assert_series_equal(result, expected)

    result = df.ID.groupby(Grouper(freq="D")).nunique()
    tm.assert_series_equal(result, expected)


def test_resample_nunique_preserves_column_level_names(unit):
    # see gh-23222
    df = DataFrame(
        np.random.default_rng(2).standard_normal((5, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=5, freq="D"),
    ).abs()
    df.index = df.index.as_unit(unit)
    df.columns = pd.MultiIndex.from_arrays(
        [df.columns.tolist()] * 2, names=["lev0", "lev1"]
    )
    result = df.resample("1h").nunique()
    tm.assert_index_equal(df.columns, result.columns)


@pytest.mark.parametrize(
    "func",
    [
        lambda x: x.nunique(),
        lambda x: x.agg(Series.nunique),
        lambda x: x.agg("nunique"),
    ],
    ids=["nunique", "series_nunique", "nunique_str"],
)
def test_resample_nunique_with_date_gap(func, unit):
    # GH 13453
    # Since all elements are unique, these should all be the same
    index = date_range("1-1-2000", "2-15-2000", freq="h").as_unit(unit)
    index2 = date_range("4-15-2000", "5-15-2000", freq="h").as_unit(unit)
    index3 = index.append(index2)
    s = Series(range(len(index3)), index=index3, dtype="int64")
    r = s.resample("ME")
    result = r.count()
    expected = func(r)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("n", [10000, 100000])
@pytest.mark.parametrize("k", [10, 100, 1000])
def test_resample_group_info(n, k, unit):
    # GH10914

    # use a fixed seed to always have the same uniques
    prng = np.random.default_rng(2)

    dr = date_range(start="2015-08-27", periods=n // 10, freq="min").as_unit(unit)
    ts = Series(prng.integers(0, n // k, n).astype("int64"), index=prng.choice(dr, n))

    left = ts.resample("30min").nunique()
    ix = date_range(start=ts.index.min(), end=ts.index.max(), freq="30min").as_unit(
        unit
    )

    vals = ts.values
    bins = np.searchsorted(ix.values, ts.index, side="right")

    sorter = np.lexsort((vals, bins))
    vals, bins = vals[sorter], bins[sorter]

    mask = np.r_[True, vals[1:] != vals[:-1]]
    mask |= np.r_[True, bins[1:] != bins[:-1]]

    arr = np.bincount(bins[mask] - 1, minlength=len(ix)).astype("int64", copy=False)
    right = Series(arr, index=ix)

    tm.assert_series_equal(left, right)


def test_resample_size(unit):
    n = 10000
    dr = date_range("2015-09-19", periods=n, freq="min").as_unit(unit)
    ts = Series(
        np.random.default_rng(2).standard_normal(n),
        index=np.random.default_rng(2).choice(dr, n),
    )

    left = ts.resample("7min").size()
    ix = date_range(start=left.index.min(), end=ts.index.max(), freq="7min").as_unit(
        unit
    )

    bins = np.searchsorted(ix.values, ts.index.values, side="right")
    val = np.bincount(bins, minlength=len(ix) + 1)[1:].astype("int64", copy=False)

    right = Series(val, index=ix)
    tm.assert_series_equal(left, right)


def test_resample_across_dst():
    # The test resamples a DatetimeIndex with values before and after a
    # DST change
    # Issue: 14682

    # The DatetimeIndex we will start with
    # (note that DST happens at 03:00+02:00 -> 02:00+01:00)
    # 2016-10-30 02:23:00+02:00, 2016-10-30 02:23:00+01:00
    df1 = DataFrame([1477786980, 1477790580], columns=["ts"])
    dti1 = DatetimeIndex(
        pd.to_datetime(df1.ts, unit="s")
        .dt.tz_localize("UTC")
        .dt.tz_convert("Europe/Madrid")
    )

    # The expected DatetimeIndex after resampling.
    # 2016-10-30 02:00:00+02:00, 2016-10-30 02:00:00+01:00
    df2 = DataFrame([1477785600, 1477789200], columns=["ts"])
    dti2 = DatetimeIndex(
        pd.to_datetime(df2.ts, unit="s")
        .dt.tz_localize("UTC")
        .dt.tz_convert("Europe/Madrid"),
        freq="h",
    )
    df = DataFrame([5, 5], index=dti1)

    result = df.resample(rule="h").sum()
    expected = DataFrame([5, 5], index=dti2)

    tm.assert_frame_equal(result, expected)


def test_groupby_with_dst_time_change(unit):
    # GH 24972
    index = (
        DatetimeIndex([1478064900001000000, 1480037118776792000], tz="UTC")
        .tz_convert("America/Chicago")
        .as_unit(unit)
    )

    df = DataFrame([1, 2], index=index)
    result = df.groupby(Grouper(freq="1d")).last()
    expected_index_values = date_range(
        "2016-11-02", "2016-11-24", freq="d", tz="America/Chicago"
    ).as_unit(unit)

    index = DatetimeIndex(expected_index_values)
    expected = DataFrame([1.0] + ([np.nan] * 21) + [2.0], index=index)
    tm.assert_frame_equal(result, expected)


def test_resample_dst_anchor(unit):
    # 5172
    dti = DatetimeIndex([datetime(2012, 11, 4, 23)], tz="US/Eastern").as_unit(unit)
    df = DataFrame([5], index=dti)

    dti = DatetimeIndex(df.index.normalize(), freq="D").as_unit(unit)
    expected = DataFrame([5], index=dti)
    tm.assert_frame_equal(df.resample(rule="D").sum(), expected)
    df.resample(rule="MS").sum()
    tm.assert_frame_equal(
        df.resample(rule="MS").sum(),
        DataFrame(
            [5],
            index=DatetimeIndex(
                [datetime(2012, 11, 1)], tz="US/Eastern", freq="MS"
            ).as_unit(unit),
        ),
    )


def test_resample_dst_anchor2(unit):
    dti = date_range(
        "2013-09-30", "2013-11-02", freq="30Min", tz="Europe/Paris"
    ).as_unit(unit)
    values = range(dti.size)
    df = DataFrame({"a": values, "b": values, "c": values}, index=dti, dtype="int64")
    how = {"a": "min", "b": "max", "c": "count"}

    rs = df.resample("W-MON")
    result = rs.agg(how)[["a", "b", "c"]]
    expected = DataFrame(
        {
            "a": [0, 48, 384, 720, 1056, 1394],
            "b": [47, 383, 719, 1055, 1393, 1586],
            "c": [48, 336, 336, 336, 338, 193],
        },
        index=date_range(
            "9/30/2013", "11/4/2013", freq="W-MON", tz="Europe/Paris"
        ).as_unit(unit),
    )
    tm.assert_frame_equal(
        result,
        expected,
        "W-MON Frequency",
    )

    rs2 = df.resample("2W-MON")
    result2 = rs2.agg(how)[["a", "b", "c"]]
    expected2 = DataFrame(
        {
            "a": [0, 48, 720, 1394],
            "b": [47, 719, 1393, 1586],
            "c": [48, 672, 674, 193],
        },
        index=date_range(
            "9/30/2013", "11/11/2013", freq="2W-MON", tz="Europe/Paris"
        ).as_unit(unit),
    )
    tm.assert_frame_equal(
        result2,
        expected2,
        "2W-MON Frequency",
    )

    rs3 = df.resample("MS")
    result3 = rs3.agg(how)[["a", "b", "c"]]
    expected3 = DataFrame(
        {"a": [0, 48, 1538], "b": [47, 1537, 1586], "c": [48, 1490, 49]},
        index=date_range("9/1/2013", "11/1/2013", freq="MS", tz="Europe/Paris").as_unit(
            unit
        ),
    )
    tm.assert_frame_equal(
        result3,
        expected3,
        "MS Frequency",
    )

    rs4 = df.resample("2MS")
    result4 = rs4.agg(how)[["a", "b", "c"]]
    expected4 = DataFrame(
        {"a": [0, 1538], "b": [1537, 1586], "c": [1538, 49]},
        index=date_range(
            "9/1/2013", "11/1/2013", freq="2MS", tz="Europe/Paris"
        ).as_unit(unit),
    )
    tm.assert_frame_equal(
        result4,
        expected4,
        "2MS Frequency",
    )

    df_daily = df["10/26/2013":"10/29/2013"]
    rs_d = df_daily.resample("D")
    result_d = rs_d.agg({"a": "min", "b": "max", "c": "count"})[["a", "b", "c"]]
    expected_d = DataFrame(
        {
            "a": [1248, 1296, 1346, 1394],
            "b": [1295, 1345, 1393, 1441],
            "c": [48, 50, 48, 48],
        },
        index=date_range(
            "10/26/2013", "10/29/2013", freq="D", tz="Europe/Paris"
        ).as_unit(unit),
    )
    tm.assert_frame_equal(
        result_d,
        expected_d,
        "D Frequency",
    )


def test_downsample_across_dst(unit):
    # GH 8531
    tz = pytz.timezone("Europe/Berlin")
    dt = datetime(2014, 10, 26)
    dates = date_range(tz.localize(dt), periods=4, freq="2h").as_unit(unit)
    result = Series(5, index=dates).resample("h").mean()
    expected = Series(
        [5.0, np.nan] * 3 + [5.0],
        index=date_range(tz.localize(dt), periods=7, freq="h").as_unit(unit),
    )
    tm.assert_series_equal(result, expected)


def test_downsample_across_dst_weekly(unit):
    # GH 9119, GH 21459
    df = DataFrame(
        index=DatetimeIndex(
            ["2017-03-25", "2017-03-26", "2017-03-27", "2017-03-28", "2017-03-29"],
            tz="Europe/Amsterdam",
        ).as_unit(unit),
        data=[11, 12, 13, 14, 15],
    )
    result = df.resample("1W").sum()
    expected = DataFrame(
        [23, 42],
        index=DatetimeIndex(
            ["2017-03-26", "2017-04-02"], tz="Europe/Amsterdam", freq="W"
        ).as_unit(unit),
    )
    tm.assert_frame_equal(result, expected)


def test_downsample_across_dst_weekly_2(unit):
    # GH 9119, GH 21459
    idx = date_range("2013-04-01", "2013-05-01", tz="Europe/London", freq="h").as_unit(
        unit
    )
    s = Series(index=idx, dtype=np.float64)
    result = s.resample("W").mean()
    expected = Series(
        index=date_range("2013-04-07", freq="W", periods=5, tz="Europe/London").as_unit(
            unit
        ),
        dtype=np.float64,
    )
    tm.assert_series_equal(result, expected)


def test_downsample_dst_at_midnight(unit):
    # GH 25758
    start = datetime(2018, 11, 3, 12)
    end = datetime(2018, 11, 5, 12)
    index = date_range(start, end, freq="1h").as_unit(unit)
    index = index.tz_localize("UTC").tz_convert("America/Havana")
    data = list(range(len(index)))
    dataframe = DataFrame(data, index=index)
    result = dataframe.groupby(Grouper(freq="1D")).mean()

    dti = date_range("2018-11-03", periods=3).tz_localize(
        "America/Havana", ambiguous=True
    )
    dti = DatetimeIndex(dti, freq="D").as_unit(unit)
    expected = DataFrame([7.5, 28.0, 44.5], index=dti)
    tm.assert_frame_equal(result, expected)


def test_resample_with_nat(unit):
    # GH 13020
    index = DatetimeIndex(
        [
            pd.NaT,
            "1970-01-01 00:00:00",
            pd.NaT,
            "1970-01-01 00:00:01",
            "1970-01-01 00:00:02",
        ]
    ).as_unit(unit)
    frame = DataFrame([2, 3, 5, 7, 11], index=index)

    index_1s = DatetimeIndex(
        ["1970-01-01 00:00:00", "1970-01-01 00:00:01", "1970-01-01 00:00:02"]
    ).as_unit(unit)
    frame_1s = DataFrame([3.0, 7.0, 11.0], index=index_1s)
    tm.assert_frame_equal(frame.resample("1s").mean(), frame_1s)

    index_2s = DatetimeIndex(["1970-01-01 00:00:00", "1970-01-01 00:00:02"]).as_unit(
        unit
    )
    frame_2s = DataFrame([5.0, 11.0], index=index_2s)
    tm.assert_frame_equal(frame.resample("2s").mean(), frame_2s)

    index_3s = DatetimeIndex(["1970-01-01 00:00:00"]).as_unit(unit)
    frame_3s = DataFrame([7.0], index=index_3s)
    tm.assert_frame_equal(frame.resample("3s").mean(), frame_3s)

    tm.assert_frame_equal(frame.resample("60s").mean(), frame_3s)


def test_resample_datetime_values(unit):
    # GH 13119
    # check that datetime dtype is preserved when NaT values are
    # introduced by the resampling

    dates = [datetime(2016, 1, 15), datetime(2016, 1, 19)]
    df = DataFrame({"timestamp": dates}, index=dates)
    df.index = df.index.as_unit(unit)

    exp = Series(
        [datetime(2016, 1, 15), pd.NaT, datetime(2016, 1, 19)],
        index=date_range("2016-01-15", periods=3, freq="2D").as_unit(unit),
        name="timestamp",
    )

    res = df.resample("2D").first()["timestamp"]
    tm.assert_series_equal(res, exp)
    res = df["timestamp"].resample("2D").first()
    tm.assert_series_equal(res, exp)


def test_resample_apply_with_additional_args(series, unit):
    # GH 14615
    def f(data, add_arg):
        return np.mean(data) * add_arg

    series.index = series.index.as_unit(unit)

    multiplier = 10
    result = series.resample("D").apply(f, multiplier)
    expected = series.resample("D").mean().multiply(multiplier)
    tm.assert_series_equal(result, expected)

    # Testing as kwarg
    result = series.resample("D").apply(f, add_arg=multiplier)
    expected = series.resample("D").mean().multiply(multiplier)
    tm.assert_series_equal(result, expected)


def test_resample_apply_with_additional_args2():
    # Testing dataframe
    def f(data, add_arg):
        return np.mean(data) * add_arg

    multiplier = 10

    df = DataFrame({"A": 1, "B": 2}, index=date_range("2017", periods=10))
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A").resample("D").agg(f, multiplier).astype(float)
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.groupby("A").resample("D").mean().multiply(multiplier)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize(
    "n1, freq1, n2, freq2",
    [
        (30, "s", 0.5, "Min"),
        (60, "s", 1, "Min"),
        (3600, "s", 1, "h"),
        (60, "Min", 1, "h"),
        (21600, "s", 0.25, "D"),
        (86400, "s", 1, "D"),
        (43200, "s", 0.5, "D"),
        (1440, "Min", 1, "D"),
        (12, "h", 0.5, "D"),
        (24, "h", 1, "D"),
    ],
)
def test_resample_equivalent_offsets(n1, freq1, n2, freq2, k, unit):
    # GH 24127
    n1_ = n1 * k
    n2_ = n2 * k
    dti = date_range("1991-09-05", "1991-09-12", freq=freq1).as_unit(unit)
    ser = Series(range(len(dti)), index=dti)

    result1 = ser.resample(str(n1_) + freq1).mean()
    result2 = ser.resample(str(n2_) + freq2).mean()
    tm.assert_series_equal(result1, result2)


@pytest.mark.parametrize(
    "first,last,freq,exp_first,exp_last",
    [
        ("19910905", "19920406", "D", "19910905", "19920407"),
        ("19910905 00:00", "19920406 06:00", "D", "19910905", "19920407"),
        ("19910905 06:00", "19920406 06:00", "h", "19910905 06:00", "19920406 07:00"),
        ("19910906", "19920406", "ME", "19910831", "19920430"),
        ("19910831", "19920430", "ME", "19910831", "19920531"),
        ("1991-08", "1992-04", "ME", "19910831", "19920531"),
    ],
)
def test_get_timestamp_range_edges(first, last, freq, exp_first, exp_last, unit):
    first = Period(first)
    first = first.to_timestamp(first.freq).as_unit(unit)
    last = Period(last)
    last = last.to_timestamp(last.freq).as_unit(unit)

    exp_first = Timestamp(exp_first)
    exp_last = Timestamp(exp_last)

    freq = pd.tseries.frequencies.to_offset(freq)
    result = _get_timestamp_range_edges(first, last, freq, unit="ns")
    expected = (exp_first, exp_last)
    assert result == expected


@pytest.mark.parametrize("duplicates", [True, False])
def test_resample_apply_product(duplicates, unit):
    # GH 5586
    index = date_range(start="2012-01-31", freq="ME", periods=12).as_unit(unit)

    ts = Series(range(12), index=index)
    df = DataFrame({"A": ts, "B": ts + 2})
    if duplicates:
        df.columns = ["A", "A"]

    msg = "using DatetimeIndexResampler.prod"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.resample("QE").apply(np.prod)
    expected = DataFrame(
        np.array([[0, 24], [60, 210], [336, 720], [990, 1716]], dtype=np.int64),
        index=DatetimeIndex(
            ["2012-03-31", "2012-06-30", "2012-09-30", "2012-12-31"], freq="QE-DEC"
        ).as_unit(unit),
        columns=df.columns,
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "first,last,freq_in,freq_out,exp_last",
    [
        (
            "2020-03-28",
            "2020-03-31",
            "D",
            "24h",
            "2020-03-30 01:00",
        ),  # includes transition into DST
        (
            "2020-03-28",
            "2020-10-27",
            "D",
            "24h",
            "2020-10-27 00:00",
        ),  # includes transition into and out of DST
        (
            "2020-10-25",
            "2020-10-27",
            "D",
            "24h",
            "2020-10-26 23:00",
        ),  # includes transition out of DST
        (
            "2020-03-28",
            "2020-03-31",
            "24h",
            "D",
            "2020-03-30 00:00",
        ),  # same as above, but from 24H to D
        ("2020-03-28", "2020-10-27", "24h", "D", "2020-10-27 00:00"),
        ("2020-10-25", "2020-10-27", "24h", "D", "2020-10-26 00:00"),
    ],
)
def test_resample_calendar_day_with_dst(
    first: str, last: str, freq_in: str, freq_out: str, exp_last: str, unit
):
    # GH 35219
    ts = Series(
        1.0, date_range(first, last, freq=freq_in, tz="Europe/Amsterdam").as_unit(unit)
    )
    result = ts.resample(freq_out).ffill()
    expected = Series(
        1.0,
        date_range(first, exp_last, freq=freq_out, tz="Europe/Amsterdam").as_unit(unit),
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["min", "max", "first", "last"])
def test_resample_aggregate_functions_min_count(func, unit):
    # GH#37768
    index = date_range(start="2020", freq="ME", periods=3).as_unit(unit)
    ser = Series([1, np.nan, np.nan], index)
    result = getattr(ser.resample("QE"), func)(min_count=2)
    expected = Series(
        [np.nan],
        index=DatetimeIndex(["2020-03-31"], freq="QE-DEC").as_unit(unit),
    )
    tm.assert_series_equal(result, expected)


def test_resample_unsigned_int(any_unsigned_int_numpy_dtype, unit):
    # gh-43329
    df = DataFrame(
        index=date_range(start="2000-01-01", end="2000-01-03 23", freq="12h").as_unit(
            unit
        ),
        columns=["x"],
        data=[0, 1, 0] * 2,
        dtype=any_unsigned_int_numpy_dtype,
    )
    df = df.loc[(df.index < "2000-01-02") | (df.index > "2000-01-03"), :]

    result = df.resample("D").max()

    expected = DataFrame(
        [1, np.nan, 0],
        columns=["x"],
        index=date_range(start="2000-01-01", end="2000-01-03 23", freq="D").as_unit(
            unit
        ),
    )
    tm.assert_frame_equal(result, expected)


def test_long_rule_non_nano():
    # https://github.com/pandas-dev/pandas/issues/51024
    idx = date_range("0300-01-01", "2000-01-01", unit="s", freq="100YE")
    ser = Series([1, 4, 2, 8, 5, 7, 1, 4, 2, 8, 5, 7, 1, 4, 2, 8, 5], index=idx)
    result = ser.resample("200YE").mean()
    expected_idx = DatetimeIndex(
        np.array(
            [
                "0300-12-31",
                "0500-12-31",
                "0700-12-31",
                "0900-12-31",
                "1100-12-31",
                "1300-12-31",
                "1500-12-31",
                "1700-12-31",
                "1900-12-31",
            ]
        ).astype("datetime64[s]"),
        freq="200YE-DEC",
    )
    expected = Series([1.0, 3.0, 6.5, 4.0, 3.0, 6.5, 4.0, 3.0, 6.5], index=expected_idx)
    tm.assert_series_equal(result, expected)


def test_resample_empty_series_with_tz():
    # GH#53664
    df = DataFrame({"ts": [], "values": []}).astype(
        {"ts": "datetime64[ns, Atlantic/Faroe]"}
    )
    result = df.resample("2MS", on="ts", closed="left", label="left", origin="start")[
        "values"
    ].sum()

    expected_idx = DatetimeIndex(
        [], freq="2MS", name="ts", dtype="datetime64[ns, Atlantic/Faroe]"
    )
    expected = Series([], index=expected_idx, name="values", dtype="float64")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "freq, freq_depr",
    [
        ("2ME", "2M"),
        ("2QE", "2Q"),
        ("2QE-SEP", "2Q-SEP"),
        ("1YE", "1Y"),
        ("2YE-MAR", "2Y-MAR"),
        ("1YE", "1A"),
        ("2YE-MAR", "2A-MAR"),
    ],
)
def test_resample_M_Q_Y_A_deprecated(freq, freq_depr):
    # GH#9586
    depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed "
    f"in a future version, please use '{freq[1:]}' instead."

    s = Series(range(10), index=date_range("20130101", freq="d", periods=10))
    expected = s.resample(freq).mean()
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        result = s.resample(freq_depr).mean()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "freq, freq_depr",
    [
        ("2BME", "2BM"),
        ("2BQE", "2BQ"),
        ("2BQE-MAR", "2BQ-MAR"),
    ],
)
def test_resample_BM_BQ_deprecated(freq, freq_depr):
    # GH#52064
    depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed "
    f"in a future version, please use '{freq[1:]}' instead."

    s = Series(range(10), index=date_range("20130101", freq="d", periods=10))
    expected = s.resample(freq).mean()
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        result = s.resample(freq_depr).mean()
    tm.assert_series_equal(result, expected)


def test_resample_ms_closed_right(unit):
    # https://github.com/pandas-dev/pandas/issues/55271
    dti = date_range(start="2020-01-31", freq="1min", periods=6000, unit=unit)
    df = DataFrame({"ts": dti}, index=dti)
    grouped = df.resample("MS", closed="right")
    result = grouped.last()
    exp_dti = DatetimeIndex(
        [datetime(2020, 1, 1), datetime(2020, 2, 1)], freq="MS"
    ).as_unit(unit)
    expected = DataFrame(
        {"ts": [datetime(2020, 2, 1), datetime(2020, 2, 4, 3, 59)]},
        index=exp_dti,
    ).astype(f"M8[{unit}]")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("freq", ["B", "C"])
def test_resample_c_b_closed_right(freq: str, unit):
    # https://github.com/pandas-dev/pandas/issues/55281
    dti = date_range(start="2020-01-31", freq="1min", periods=6000, unit=unit)
    df = DataFrame({"ts": dti}, index=dti)
    grouped = df.resample(freq, closed="right")
    result = grouped.last()

    exp_dti = DatetimeIndex(
        [
            datetime(2020, 1, 30),
            datetime(2020, 1, 31),
            datetime(2020, 2, 3),
            datetime(2020, 2, 4),
        ],
        freq=freq,
    ).as_unit(unit)
    expected = DataFrame(
        {
            "ts": [
                datetime(2020, 1, 31),
                datetime(2020, 2, 3),
                datetime(2020, 2, 4),
                datetime(2020, 2, 4, 3, 59),
            ]
        },
        index=exp_dti,
    ).astype(f"M8[{unit}]")
    tm.assert_frame_equal(result, expected)


def test_resample_b_55282(unit):
    # https://github.com/pandas-dev/pandas/issues/55282
    dti = date_range("2023-09-26", periods=6, freq="12h", unit=unit)
    ser = Series([1, 2, 3, 4, 5, 6], index=dti)
    result = ser.resample("B", closed="right", label="right").mean()

    exp_dti = DatetimeIndex(
        [
            datetime(2023, 9, 26),
            datetime(2023, 9, 27),
            datetime(2023, 9, 28),
            datetime(2023, 9, 29),
        ],
        freq="B",
    ).as_unit(unit)
    expected = Series(
        [1.0, 2.5, 4.5, 6.0],
        index=exp_dti,
    )
    tm.assert_series_equal(result, expected)


@td.skip_if_no("pyarrow")
@pytest.mark.parametrize(
    "tz",
    [
        None,
        pytest.param(
            "UTC",
            marks=pytest.mark.xfail(
                condition=is_platform_windows(),
                reason="TODO: Set ARROW_TIMEZONE_DATABASE env var in CI",
            ),
        ),
    ],
)
def test_arrow_timestamp_resample(tz):
    # GH 56371
    idx = Series(date_range("2020-01-01", periods=5), dtype="timestamp[ns][pyarrow]")
    if tz is not None:
        idx = idx.dt.tz_localize(tz)
    expected = Series(np.arange(5, dtype=np.float64), index=idx)
    result = expected.resample("1D").mean()
    tm.assert_series_equal(result, expected)