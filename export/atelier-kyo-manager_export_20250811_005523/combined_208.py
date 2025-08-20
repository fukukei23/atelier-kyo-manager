
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\usecols\test_usecols_basic.py ===
"""
Tests the usecols functionality during parsing
for all of the parsers defined in parsers.py
"""
from io import StringIO

import numpy as np
import pytest

from pandas.errors import ParserError

from pandas import (
    DataFrame,
    Index,
    array,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

_msg_validate_usecols_arg = (
    "'usecols' must either be list-like "
    "of all strings, all unicode, all "
    "integers or a callable."
)
_msg_validate_usecols_names = (
    "Usecols do not match columns, columns expected but not found: {0}"
)
_msg_pyarrow_requires_names = (
    "The pyarrow engine does not allow 'usecols' to be integer column "
    "positions. Pass a list of string column names instead."
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame is deprecated:DeprecationWarning"
)


def test_raise_on_mixed_dtype_usecols(all_parsers):
    # See gh-12678
    data = """a,b,c
        1000,2000,3000
        4000,5000,6000
        """
    usecols = [0, "b", 2]
    parser = all_parsers

    with pytest.raises(ValueError, match=_msg_validate_usecols_arg):
        parser.read_csv(StringIO(data), usecols=usecols)


@pytest.mark.parametrize("usecols", [(1, 2), ("b", "c")])
def test_usecols(all_parsers, usecols, request):
    data = """\
a,b,c
1,2,3
4,5,6
7,8,9
10,11,12"""
    parser = all_parsers
    if parser.engine == "pyarrow" and isinstance(usecols[0], int):
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), usecols=usecols)
        return

    result = parser.read_csv(StringIO(data), usecols=usecols)

    expected = DataFrame([[2, 3], [5, 6], [8, 9], [11, 12]], columns=["b", "c"])
    tm.assert_frame_equal(result, expected)


def test_usecols_with_names(all_parsers):
    data = """\
a,b,c
1,2,3
4,5,6
7,8,9
10,11,12"""
    parser = all_parsers
    names = ["foo", "bar"]

    if parser.engine == "pyarrow":
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), names=names, usecols=[1, 2], header=0)
        return

    result = parser.read_csv(StringIO(data), names=names, usecols=[1, 2], header=0)

    expected = DataFrame([[2, 3], [5, 6], [8, 9], [11, 12]], columns=names)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "names,usecols", [(["b", "c"], [1, 2]), (["a", "b", "c"], ["b", "c"])]
)
def test_usecols_relative_to_names(all_parsers, names, usecols):
    data = """\
1,2,3
4,5,6
7,8,9
10,11,12"""
    parser = all_parsers
    if parser.engine == "pyarrow" and not isinstance(usecols[0], int):
        # ArrowKeyError: Column 'fb' in include_columns does not exist
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    result = parser.read_csv(StringIO(data), names=names, header=None, usecols=usecols)

    expected = DataFrame([[2, 3], [5, 6], [8, 9], [11, 12]], columns=["b", "c"])
    tm.assert_frame_equal(result, expected)


def test_usecols_relative_to_names2(all_parsers):
    # see gh-5766
    data = """\
1,2,3
4,5,6
7,8,9
10,11,12"""
    parser = all_parsers

    result = parser.read_csv(
        StringIO(data), names=["a", "b"], header=None, usecols=[0, 1]
    )

    expected = DataFrame([[1, 2], [4, 5], [7, 8], [10, 11]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)


# regex mismatch: "Length mismatch: Expected axis has 1 elements"
@xfail_pyarrow
def test_usecols_name_length_conflict(all_parsers):
    data = """\
1,2,3
4,5,6
7,8,9
10,11,12"""
    parser = all_parsers
    msg = "Number of passed names did not match number of header fields in the file"
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), names=["a", "b"], header=None, usecols=[1])


def test_usecols_single_string(all_parsers):
    # see gh-20558
    parser = all_parsers
    data = """foo, bar, baz
1000, 2000, 3000
4000, 5000, 6000"""

    with pytest.raises(ValueError, match=_msg_validate_usecols_arg):
        parser.read_csv(StringIO(data), usecols="foo")


@skip_pyarrow  # CSV parse error in one case, AttributeError in another
@pytest.mark.parametrize(
    "data", ["a,b,c,d\n1,2,3,4\n5,6,7,8", "a,b,c,d\n1,2,3,4,\n5,6,7,8,"]
)
def test_usecols_index_col_false(all_parsers, data):
    # see gh-9082
    parser = all_parsers
    usecols = ["a", "c", "d"]
    expected = DataFrame({"a": [1, 5], "c": [3, 7], "d": [4, 8]})

    result = parser.read_csv(StringIO(data), usecols=usecols, index_col=False)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("index_col", ["b", 0])
@pytest.mark.parametrize("usecols", [["b", "c"], [1, 2]])
def test_usecols_index_col_conflict(all_parsers, usecols, index_col, request):
    # see gh-4201: test that index_col as integer reflects usecols
    parser = all_parsers
    data = "a,b,c,d\nA,a,1,one\nB,b,2,two"

    if parser.engine == "pyarrow" and isinstance(usecols[0], int):
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), usecols=usecols, index_col=index_col)
        return

    expected = DataFrame({"c": [1, 2]}, index=Index(["a", "b"], name="b"))

    result = parser.read_csv(StringIO(data), usecols=usecols, index_col=index_col)
    tm.assert_frame_equal(result, expected)


def test_usecols_index_col_conflict2(all_parsers):
    # see gh-4201: test that index_col as integer reflects usecols
    parser = all_parsers
    data = "a,b,c,d\nA,a,1,one\nB,b,2,two"

    expected = DataFrame({"b": ["a", "b"], "c": [1, 2], "d": ("one", "two")})
    expected = expected.set_index(["b", "c"])

    result = parser.read_csv(
        StringIO(data), usecols=["b", "c", "d"], index_col=["b", "c"]
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Expected 3 columns, got 4
def test_usecols_implicit_index_col(all_parsers):
    # see gh-2654
    parser = all_parsers
    data = "a,b,c\n4,apple,bat,5.7\n8,orange,cow,10"

    result = parser.read_csv(StringIO(data), usecols=["a", "b"])
    expected = DataFrame({"a": ["apple", "orange"], "b": ["bat", "cow"]}, index=[4, 8])
    tm.assert_frame_equal(result, expected)


def test_usecols_index_col_middle(all_parsers):
    # GH#9098
    parser = all_parsers
    data = """a,b,c,d
1,2,3,4
"""
    result = parser.read_csv(StringIO(data), usecols=["b", "c", "d"], index_col="c")
    expected = DataFrame({"b": [2], "d": [4]}, index=Index([3], name="c"))
    tm.assert_frame_equal(result, expected)


def test_usecols_index_col_end(all_parsers):
    # GH#9098
    parser = all_parsers
    data = """a,b,c,d
1,2,3,4
"""
    result = parser.read_csv(StringIO(data), usecols=["b", "c", "d"], index_col="d")
    expected = DataFrame({"b": [2], "c": [3]}, index=Index([4], name="d"))
    tm.assert_frame_equal(result, expected)


def test_usecols_regex_sep(all_parsers):
    # see gh-2733
    parser = all_parsers
    data = "a  b  c\n4  apple  bat  5.7\n8  orange  cow  10"

    if parser.engine == "pyarrow":
        msg = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep=r"\s+", usecols=("a", "b"))
        return

    result = parser.read_csv(StringIO(data), sep=r"\s+", usecols=("a", "b"))

    expected = DataFrame({"a": ["apple", "orange"], "b": ["bat", "cow"]}, index=[4, 8])
    tm.assert_frame_equal(result, expected)


def test_usecols_with_whitespace(all_parsers):
    parser = all_parsers
    data = "a  b  c\n4  apple  bat  5.7\n8  orange  cow  10"

    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"

    if parser.engine == "pyarrow":
        msg = "The 'delim_whitespace' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(
                FutureWarning, match=depr_msg, check_stacklevel=False
            ):
                parser.read_csv(
                    StringIO(data), delim_whitespace=True, usecols=("a", "b")
                )
        return

    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        result = parser.read_csv(
            StringIO(data), delim_whitespace=True, usecols=("a", "b")
        )
    expected = DataFrame({"a": ["apple", "orange"], "b": ["bat", "cow"]}, index=[4, 8])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "usecols,expected",
    [
        # Column selection by index.
        ([0, 1], DataFrame(data=[[1000, 2000], [4000, 5000]], columns=["2", "0"])),
        # Column selection by name.
        (
            ["0", "1"],
            DataFrame(data=[[2000, 3000], [5000, 6000]], columns=["0", "1"]),
        ),
    ],
)
def test_usecols_with_integer_like_header(all_parsers, usecols, expected, request):
    parser = all_parsers
    data = """2,0,1
1000,2000,3000
4000,5000,6000"""

    if parser.engine == "pyarrow" and isinstance(usecols[0], int):
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), usecols=usecols)
        return

    result = parser.read_csv(StringIO(data), usecols=usecols)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # mismatched shape
def test_empty_usecols(all_parsers):
    data = "a,b,c\n1,2,3\n4,5,6"
    expected = DataFrame(columns=Index([]))
    parser = all_parsers

    result = parser.read_csv(StringIO(data), usecols=set())
    tm.assert_frame_equal(result, expected)


def test_np_array_usecols(all_parsers):
    # see gh-12546
    parser = all_parsers
    data = "a,b,c\n1,2,3"
    usecols = np.array(["a", "b"])

    expected = DataFrame([[1, 2]], columns=usecols)
    result = parser.read_csv(StringIO(data), usecols=usecols)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "usecols,expected",
    [
        (
            lambda x: x.upper() in ["AAA", "BBB", "DDD"],
            DataFrame(
                {
                    "AaA": {
                        0: 0.056674972999999997,
                        1: 2.6132309819999997,
                        2: 3.5689350380000002,
                    },
                    "bBb": {0: 8, 1: 2, 2: 7},
                    "ddd": {0: "a", 1: "b", 2: "a"},
                }
            ),
        ),
        (lambda x: False, DataFrame(columns=Index([]))),
    ],
)
def test_callable_usecols(all_parsers, usecols, expected):
    # see gh-14154
    data = """AaA,bBb,CCC,ddd
0.056674973,8,True,a
2.613230982,2,False,b
3.568935038,7,False,a"""
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine does not allow 'usecols' to be a callable"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), usecols=usecols)
        return

    result = parser.read_csv(StringIO(data), usecols=usecols)
    tm.assert_frame_equal(result, expected)


# ArrowKeyError: Column 'fa' in include_columns does not exist in CSV file
@skip_pyarrow
@pytest.mark.parametrize("usecols", [["a", "c"], lambda x: x in ["a", "c"]])
def test_incomplete_first_row(all_parsers, usecols):
    # see gh-6710
    data = "1,2\n1,2,3"
    parser = all_parsers
    names = ["a", "b", "c"]
    expected = DataFrame({"a": [1, 1], "c": [np.nan, 3]})

    result = parser.read_csv(StringIO(data), names=names, usecols=usecols)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Expected 3 columns, got 4
@pytest.mark.parametrize(
    "data,usecols,kwargs,expected",
    [
        # see gh-8985
        (
            "19,29,39\n" * 2 + "10,20,30,40",
            [0, 1, 2],
            {"header": None},
            DataFrame([[19, 29, 39], [19, 29, 39], [10, 20, 30]]),
        ),
        # see gh-9549
        (
            ("A,B,C\n1,2,3\n3,4,5\n1,2,4,5,1,6\n1,2,3,,,1,\n1,2,3\n5,6,7"),
            ["A", "B", "C"],
            {},
            DataFrame(
                {
                    "A": [1, 3, 1, 1, 1, 5],
                    "B": [2, 4, 2, 2, 2, 6],
                    "C": [3, 5, 4, 3, 3, 7],
                }
            ),
        ),
    ],
)
def test_uneven_length_cols(all_parsers, data, usecols, kwargs, expected):
    # see gh-8985
    parser = all_parsers
    result = parser.read_csv(StringIO(data), usecols=usecols, **kwargs)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "usecols,kwargs,expected,msg",
    [
        (
            ["a", "b", "c", "d"],
            {},
            DataFrame({"a": [1, 5], "b": [2, 6], "c": [3, 7], "d": [4, 8]}),
            None,
        ),
        (
            ["a", "b", "c", "f"],
            {},
            None,
            _msg_validate_usecols_names.format(r"\['f'\]"),
        ),
        (["a", "b", "f"], {}, None, _msg_validate_usecols_names.format(r"\['f'\]")),
        (
            ["a", "b", "f", "g"],
            {},
            None,
            _msg_validate_usecols_names.format(r"\[('f', 'g'|'g', 'f')\]"),
        ),
        # see gh-14671
        (
            None,
            {"header": 0, "names": ["A", "B", "C", "D"]},
            DataFrame({"A": [1, 5], "B": [2, 6], "C": [3, 7], "D": [4, 8]}),
            None,
        ),
        (
            ["A", "B", "C", "f"],
            {"header": 0, "names": ["A", "B", "C", "D"]},
            None,
            _msg_validate_usecols_names.format(r"\['f'\]"),
        ),
        (
            ["A", "B", "f"],
            {"names": ["A", "B", "C", "D"]},
            None,
            _msg_validate_usecols_names.format(r"\['f'\]"),
        ),
    ],
)
def test_raises_on_usecols_names_mismatch(
    all_parsers, usecols, kwargs, expected, msg, request
):
    data = "a,b,c,d\n1,2,3,4\n5,6,7,8"
    kwargs.update(usecols=usecols)
    parser = all_parsers

    if parser.engine == "pyarrow" and not (
        usecols is not None and expected is not None
    ):
        # everything but the first case
        # ArrowKeyError: Column 'f' in include_columns does not exist in CSV file
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    if expected is None:
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
    else:
        result = parser.read_csv(StringIO(data), **kwargs)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("usecols", [["A", "C"], [0, 2]])
def test_usecols_subset_names_mismatch_orig_columns(all_parsers, usecols, request):
    data = "a,b,c,d\n1,2,3,4\n5,6,7,8"
    names = ["A", "B", "C", "D"]
    parser = all_parsers

    if parser.engine == "pyarrow":
        if isinstance(usecols[0], int):
            with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
                parser.read_csv(StringIO(data), header=0, names=names, usecols=usecols)
            return
        # "pyarrow.lib.ArrowKeyError: Column 'A' in include_columns does not exist"
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    result = parser.read_csv(StringIO(data), header=0, names=names, usecols=usecols)
    expected = DataFrame({"A": [1, 5], "C": [3, 7]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("names", [None, ["a", "b"]])
def test_usecols_indices_out_of_bounds(all_parsers, names):
    # GH#25623 & GH 41130; enforced in 2.0
    parser = all_parsers
    data = """
a,b
1,2
    """

    err = ParserError
    msg = "Defining usecols with out-of-bounds"
    if parser.engine == "pyarrow":
        err = ValueError
        msg = _msg_pyarrow_requires_names

    with pytest.raises(err, match=msg):
        parser.read_csv(StringIO(data), usecols=[0, 2], names=names, header=0)


def test_usecols_additional_columns(all_parsers):
    # GH#46997
    parser = all_parsers
    usecols = lambda header: header.strip() in ["a", "b", "c"]

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine does not allow 'usecols' to be a callable"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO("a,b\nx,y,z"), index_col=False, usecols=usecols)
        return
    result = parser.read_csv(StringIO("a,b\nx,y,z"), index_col=False, usecols=usecols)
    expected = DataFrame({"a": ["x"], "b": "y"})
    tm.assert_frame_equal(result, expected)


def test_usecols_additional_columns_integer_columns(all_parsers):
    # GH#46997
    parser = all_parsers
    usecols = lambda header: header.strip() in ["0", "1"]
    if parser.engine == "pyarrow":
        msg = "The pyarrow engine does not allow 'usecols' to be a callable"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO("0,1\nx,y,z"), index_col=False, usecols=usecols)
        return
    result = parser.read_csv(StringIO("0,1\nx,y,z"), index_col=False, usecols=usecols)
    expected = DataFrame({"0": ["x"], "1": "y"})
    tm.assert_frame_equal(result, expected)


def test_usecols_dtype(all_parsers):
    parser = all_parsers
    data = """
col1,col2,col3
a,1,x
b,2,y
"""
    result = parser.read_csv(
        StringIO(data),
        usecols=["col1", "col2"],
        dtype={"col1": "string", "col2": "uint8", "col3": "string"},
    )
    expected = DataFrame(
        {"col1": array(["a", "b"]), "col2": np.array([1, 2], dtype="uint8")}
    )
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\common.py ===
"""
Module consolidating common testing functions for checking plotting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas.core.dtypes.api import is_list_like

import pandas as pd
from pandas import Series
import pandas._testing as tm

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes


def _check_legend_labels(axes, labels=None, visible=True):
    """
    Check each axes has expected legend labels

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like
    labels : list-like
        expected legend labels
    visible : bool
        expected legend visibility. labels are checked only when visible is
        True
    """
    if visible and (labels is None):
        raise ValueError("labels must be specified when visible is True")
    axes = _flatten_visible(axes)
    for ax in axes:
        if visible:
            assert ax.get_legend() is not None
            _check_text_labels(ax.get_legend().get_texts(), labels)
        else:
            assert ax.get_legend() is None


def _check_legend_marker(ax, expected_markers=None, visible=True):
    """
    Check ax has expected legend markers

    Parameters
    ----------
    ax : matplotlib Axes object
    expected_markers : list-like
        expected legend markers
    visible : bool
        expected legend visibility. labels are checked only when visible is
        True
    """
    if visible and (expected_markers is None):
        raise ValueError("Markers must be specified when visible is True")
    if visible:
        handles, _ = ax.get_legend_handles_labels()
        markers = [handle.get_marker() for handle in handles]
        assert markers == expected_markers
    else:
        assert ax.get_legend() is None


def _check_data(xp, rs):
    """
    Check each axes has identical lines

    Parameters
    ----------
    xp : matplotlib Axes object
    rs : matplotlib Axes object
    """
    import matplotlib.pyplot as plt

    xp_lines = xp.get_lines()
    rs_lines = rs.get_lines()

    assert len(xp_lines) == len(rs_lines)
    for xpl, rsl in zip(xp_lines, rs_lines):
        xpdata = xpl.get_xydata()
        rsdata = rsl.get_xydata()
        tm.assert_almost_equal(xpdata, rsdata)

    plt.close("all")


def _check_visible(collections, visible=True):
    """
    Check each artist is visible or not

    Parameters
    ----------
    collections : matplotlib Artist or its list-like
        target Artist or its list or collection
    visible : bool
        expected visibility
    """
    from matplotlib.collections import Collection

    if not isinstance(collections, Collection) and not is_list_like(collections):
        collections = [collections]

    for patch in collections:
        assert patch.get_visible() == visible


def _check_patches_all_filled(axes: Axes | Sequence[Axes], filled: bool = True) -> None:
    """
    Check for each artist whether it is filled or not

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like
    filled : bool
        expected filling
    """

    axes = _flatten_visible(axes)
    for ax in axes:
        for patch in ax.patches:
            assert patch.fill == filled


def _get_colors_mapped(series, colors):
    unique = series.unique()
    # unique and colors length can be differed
    # depending on slice value
    mapped = dict(zip(unique, colors))
    return [mapped[v] for v in series.values]


def _check_colors(collections, linecolors=None, facecolors=None, mapping=None):
    """
    Check each artist has expected line colors and face colors

    Parameters
    ----------
    collections : list-like
        list or collection of target artist
    linecolors : list-like which has the same length as collections
        list of expected line colors
    facecolors : list-like which has the same length as collections
        list of expected face colors
    mapping : Series
        Series used for color grouping key
        used for andrew_curves, parallel_coordinates, radviz test
    """
    from matplotlib import colors
    from matplotlib.collections import (
        Collection,
        LineCollection,
        PolyCollection,
    )
    from matplotlib.lines import Line2D

    conv = colors.ColorConverter
    if linecolors is not None:
        if mapping is not None:
            linecolors = _get_colors_mapped(mapping, linecolors)
            linecolors = linecolors[: len(collections)]

        assert len(collections) == len(linecolors)
        for patch, color in zip(collections, linecolors):
            if isinstance(patch, Line2D):
                result = patch.get_color()
                # Line2D may contains string color expression
                result = conv.to_rgba(result)
            elif isinstance(patch, (PolyCollection, LineCollection)):
                result = tuple(patch.get_edgecolor()[0])
            else:
                result = patch.get_edgecolor()

            expected = conv.to_rgba(color)
            assert result == expected

    if facecolors is not None:
        if mapping is not None:
            facecolors = _get_colors_mapped(mapping, facecolors)
            facecolors = facecolors[: len(collections)]

        assert len(collections) == len(facecolors)
        for patch, color in zip(collections, facecolors):
            if isinstance(patch, Collection):
                # returned as list of np.array
                result = patch.get_facecolor()[0]
            else:
                result = patch.get_facecolor()

            if isinstance(result, np.ndarray):
                result = tuple(result)

            expected = conv.to_rgba(color)
            assert result == expected


def _check_text_labels(texts, expected):
    """
    Check each text has expected labels

    Parameters
    ----------
    texts : matplotlib Text object, or its list-like
        target text, or its list
    expected : str or list-like which has the same length as texts
        expected text label, or its list
    """
    if not is_list_like(texts):
        assert texts.get_text() == expected
    else:
        labels = [t.get_text() for t in texts]
        assert len(labels) == len(expected)
        for label, e in zip(labels, expected):
            assert label == e


def _check_ticks_props(axes, xlabelsize=None, xrot=None, ylabelsize=None, yrot=None):
    """
    Check each axes has expected tick properties

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like
    xlabelsize : number
        expected xticks font size
    xrot : number
        expected xticks rotation
    ylabelsize : number
        expected yticks font size
    yrot : number
        expected yticks rotation
    """
    from matplotlib.ticker import NullFormatter

    axes = _flatten_visible(axes)
    for ax in axes:
        if xlabelsize is not None or xrot is not None:
            if isinstance(ax.xaxis.get_minor_formatter(), NullFormatter):
                # If minor ticks has NullFormatter, rot / fontsize are not
                # retained
                labels = ax.get_xticklabels()
            else:
                labels = ax.get_xticklabels() + ax.get_xticklabels(minor=True)

            for label in labels:
                if xlabelsize is not None:
                    tm.assert_almost_equal(label.get_fontsize(), xlabelsize)
                if xrot is not None:
                    tm.assert_almost_equal(label.get_rotation(), xrot)

        if ylabelsize is not None or yrot is not None:
            if isinstance(ax.yaxis.get_minor_formatter(), NullFormatter):
                labels = ax.get_yticklabels()
            else:
                labels = ax.get_yticklabels() + ax.get_yticklabels(minor=True)

            for label in labels:
                if ylabelsize is not None:
                    tm.assert_almost_equal(label.get_fontsize(), ylabelsize)
                if yrot is not None:
                    tm.assert_almost_equal(label.get_rotation(), yrot)


def _check_ax_scales(axes, xaxis="linear", yaxis="linear"):
    """
    Check each axes has expected scales

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like
    xaxis : {'linear', 'log'}
        expected xaxis scale
    yaxis : {'linear', 'log'}
        expected yaxis scale
    """
    axes = _flatten_visible(axes)
    for ax in axes:
        assert ax.xaxis.get_scale() == xaxis
        assert ax.yaxis.get_scale() == yaxis


def _check_axes_shape(axes, axes_num=None, layout=None, figsize=None):
    """
    Check expected number of axes is drawn in expected layout

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like
    axes_num : number
        expected number of axes. Unnecessary axes should be set to
        invisible.
    layout : tuple
        expected layout, (expected number of rows , columns)
    figsize : tuple
        expected figsize. default is matplotlib default
    """
    from pandas.plotting._matplotlib.tools import flatten_axes

    if figsize is None:
        figsize = (6.4, 4.8)
    visible_axes = _flatten_visible(axes)

    if axes_num is not None:
        assert len(visible_axes) == axes_num
        for ax in visible_axes:
            # check something drawn on visible axes
            assert len(ax.get_children()) > 0

    if layout is not None:
        x_set = set()
        y_set = set()
        for ax in flatten_axes(axes):
            # check axes coordinates to estimate layout
            points = ax.get_position().get_points()
            x_set.add(points[0][0])
            y_set.add(points[0][1])
        result = (len(y_set), len(x_set))
        assert result == layout

    tm.assert_numpy_array_equal(
        visible_axes[0].figure.get_size_inches(),
        np.array(figsize, dtype=np.float64),
    )


def _flatten_visible(axes: Axes | Sequence[Axes]) -> Sequence[Axes]:
    """
    Flatten axes, and filter only visible

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like

    """
    from pandas.plotting._matplotlib.tools import flatten_axes

    axes_ndarray = flatten_axes(axes)
    axes = [ax for ax in axes_ndarray if ax.get_visible()]
    return axes


def _check_has_errorbars(axes, xerr=0, yerr=0):
    """
    Check axes has expected number of errorbars

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like
    xerr : number
        expected number of x errorbar
    yerr : number
        expected number of y errorbar
    """
    axes = _flatten_visible(axes)
    for ax in axes:
        containers = ax.containers
        xerr_count = 0
        yerr_count = 0
        for c in containers:
            has_xerr = getattr(c, "has_xerr", False)
            has_yerr = getattr(c, "has_yerr", False)
            if has_xerr:
                xerr_count += 1
            if has_yerr:
                yerr_count += 1
        assert xerr == xerr_count
        assert yerr == yerr_count


def _check_box_return_type(
    returned, return_type, expected_keys=None, check_ax_title=True
):
    """
    Check box returned type is correct

    Parameters
    ----------
    returned : object to be tested, returned from boxplot
    return_type : str
        return_type passed to boxplot
    expected_keys : list-like, optional
        group labels in subplot case. If not passed,
        the function checks assuming boxplot uses single ax
    check_ax_title : bool
        Whether to check the ax.title is the same as expected_key
        Intended to be checked by calling from ``boxplot``.
        Normal ``plot`` doesn't attach ``ax.title``, it must be disabled.
    """
    from matplotlib.axes import Axes

    types = {"dict": dict, "axes": Axes, "both": tuple}
    if expected_keys is None:
        # should be fixed when the returning default is changed
        if return_type is None:
            return_type = "dict"

        assert isinstance(returned, types[return_type])
        if return_type == "both":
            assert isinstance(returned.ax, Axes)
            assert isinstance(returned.lines, dict)
    else:
        # should be fixed when the returning default is changed
        if return_type is None:
            for r in _flatten_visible(returned):
                assert isinstance(r, Axes)
            return

        assert isinstance(returned, Series)

        assert sorted(returned.keys()) == sorted(expected_keys)
        for key, value in returned.items():
            assert isinstance(value, types[return_type])
            # check returned dict has correct mapping
            if return_type == "axes":
                if check_ax_title:
                    assert value.get_title() == key
            elif return_type == "both":
                if check_ax_title:
                    assert value.ax.get_title() == key
                assert isinstance(value.ax, Axes)
                assert isinstance(value.lines, dict)
            elif return_type == "dict":
                line = value["medians"][0]
                axes = line.axes
                if check_ax_title:
                    assert axes.get_title() == key
            else:
                raise AssertionError


def _check_grid_settings(obj, kinds, kws={}):
    # Make sure plot defaults to rcParams['axes.grid'] setting, GH 9792

    import matplotlib as mpl

    def is_grid_on():
        xticks = mpl.pyplot.gca().xaxis.get_major_ticks()
        yticks = mpl.pyplot.gca().yaxis.get_major_ticks()
        xoff = all(not g.gridline.get_visible() for g in xticks)
        yoff = all(not g.gridline.get_visible() for g in yticks)

        return not (xoff and yoff)

    spndx = 1
    for kind in kinds:
        mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
        spndx += 1
        mpl.rc("axes", grid=False)
        obj.plot(kind=kind, **kws)
        assert not is_grid_on()
        mpl.pyplot.clf()

        mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
        spndx += 1
        mpl.rc("axes", grid=True)
        obj.plot(kind=kind, grid=False, **kws)
        assert not is_grid_on()
        mpl.pyplot.clf()

        if kind not in ["pie", "hexbin", "scatter"]:
            mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
            spndx += 1
            mpl.rc("axes", grid=True)
            obj.plot(kind=kind, **kws)
            assert is_grid_on()
            mpl.pyplot.clf()

            mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
            spndx += 1
            mpl.rc("axes", grid=False)
            obj.plot(kind=kind, grid=True, **kws)
            assert is_grid_on()
            mpl.pyplot.clf()


def _unpack_cycler(rcParams, field="color"):
    """
    Auxiliary function for correctly unpacking cycler after MPL >= 1.5
    """
    return [v[field] for v in rcParams["axes.prop_cycle"]]


def get_x_axis(ax):
    return ax._shared_axes["x"]


def get_y_axis(ax):
    return ax._shared_axes["y"]


def _check_plot_works(f, default_axes=False, **kwargs):
    """
    Create plot and ensure that plot return object is valid.

    Parameters
    ----------
    f : func
        Plotting function.
    default_axes : bool, optional
        If False (default):
            - If `ax` not in `kwargs`, then create subplot(211) and plot there
            - Create new subplot(212) and plot there as well
            - Mind special corner case for bootstrap_plot (see `_gen_two_subplots`)
        If True:
            - Simply run plotting function with kwargs provided
            - All required axes instances will be created automatically
            - It is recommended to use it when the plotting function
            creates multiple axes itself. It helps avoid warnings like
            'UserWarning: To output multiple subplots,
            the figure containing the passed axes is being cleared'
    **kwargs
        Keyword arguments passed to the plotting function.

    Returns
    -------
    Plot object returned by the last plotting.
    """
    import matplotlib.pyplot as plt

    if default_axes:
        gen_plots = _gen_default_plot
    else:
        gen_plots = _gen_two_subplots

    ret = None
    try:
        fig = kwargs.get("figure", plt.gcf())
        plt.clf()

        for ret in gen_plots(f, fig, **kwargs):
            tm.assert_is_valid_plot_return_object(ret)

    finally:
        plt.close(fig)

    return ret


def _gen_default_plot(f, fig, **kwargs):
    """
    Create plot in a default way.
    """
    yield f(**kwargs)


def _gen_two_subplots(f, fig, **kwargs):
    """
    Create plot on two subplots forcefully created.
    """
    if "ax" not in kwargs:
        fig.add_subplot(211)
    yield f(**kwargs)

    if f is pd.plotting.bootstrap_plot:
        assert "ax" not in kwargs
    else:
        kwargs["ax"] = fig.add_subplot(212)
    yield f(**kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_rank.py ===
from itertools import chain
import operator

import numpy as np
import pytest

from pandas._libs.algos import (
    Infinity,
    NegInfinity,
)
import pandas.util._test_decorators as td

from pandas import (
    NA,
    NaT,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.api.types import CategoricalDtype


@pytest.fixture
def ser():
    return Series([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3])


@pytest.fixture(
    params=[
        ["average", np.array([1.5, 5.5, 7.0, 3.5, np.nan, 3.5, 1.5, 8.0, np.nan, 5.5])],
        ["min", np.array([1, 5, 7, 3, np.nan, 3, 1, 8, np.nan, 5])],
        ["max", np.array([2, 6, 7, 4, np.nan, 4, 2, 8, np.nan, 6])],
        ["first", np.array([1, 5, 7, 3, np.nan, 4, 2, 8, np.nan, 6])],
        ["dense", np.array([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3])],
    ],
    ids=lambda x: x[0],
)
def results(request):
    return request.param


@pytest.fixture(
    params=[
        "object",
        "float64",
        "int64",
        "Float64",
        "Int64",
        pytest.param("float64[pyarrow]", marks=td.skip_if_no("pyarrow")),
        pytest.param("int64[pyarrow]", marks=td.skip_if_no("pyarrow")),
        pytest.param("string[pyarrow]", marks=td.skip_if_no("pyarrow")),
        "string[python]",
        "str",
    ]
)
def dtype(request):
    return request.param


def expected_dtype(dtype, method, pct=False):
    exp_dtype = "float64"
    # elif dtype in ["Int64", "Float64", "string[pyarrow]", "string[python]"]:
    if dtype in ["string[pyarrow]"]:
        exp_dtype = "Float64"
    elif dtype in ["float64[pyarrow]", "int64[pyarrow]"]:
        if method == "average" or pct:
            exp_dtype = "double[pyarrow]"
        else:
            exp_dtype = "uint64[pyarrow]"

    return exp_dtype


class TestSeriesRank:
    def test_rank(self, datetime_series):
        sp_stats = pytest.importorskip("scipy.stats")

        datetime_series[::2] = np.nan
        datetime_series[:10:3] = 4.0

        ranks = datetime_series.rank()
        oranks = datetime_series.astype("O").rank()

        tm.assert_series_equal(ranks, oranks)

        mask = np.isnan(datetime_series)
        filled = datetime_series.fillna(np.inf)

        # rankdata returns a ndarray
        exp = Series(sp_stats.rankdata(filled), index=filled.index, name="ts")
        exp[mask] = np.nan

        tm.assert_series_equal(ranks, exp)

        iseries = Series(np.arange(5).repeat(2))

        iranks = iseries.rank()
        exp = iseries.astype(float).rank()
        tm.assert_series_equal(iranks, exp)
        iseries = Series(np.arange(5)) + 1.0
        exp = iseries / 5.0
        iranks = iseries.rank(pct=True)

        tm.assert_series_equal(iranks, exp)

        iseries = Series(np.repeat(1, 100))
        exp = Series(np.repeat(0.505, 100))
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        # Explicit cast to float to avoid implicit cast when setting nan
        iseries = iseries.astype("float")
        iseries[1] = np.nan
        exp = Series(np.repeat(50.0 / 99.0, 100))
        exp[1] = np.nan
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        iseries = Series(np.arange(5)) + 1.0
        iseries[4] = np.nan
        exp = iseries / 4.0
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        iseries = Series(np.repeat(np.nan, 100))
        exp = iseries.copy()
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        # Explicit cast to float to avoid implicit cast when setting nan
        iseries = Series(np.arange(5), dtype="float") + 1
        iseries[4] = np.nan
        exp = iseries / 4.0
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        rng = date_range("1/1/1990", periods=5)
        # Explicit cast to float to avoid implicit cast when setting nan
        iseries = Series(np.arange(5), rng, dtype="float") + 1
        iseries.iloc[4] = np.nan
        exp = iseries / 4.0
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        iseries = Series([1e-50, 1e-100, 1e-20, 1e-2, 1e-20 + 1e-30, 1e-1])
        exp = Series([2, 1, 3, 5, 4, 6.0])
        iranks = iseries.rank()
        tm.assert_series_equal(iranks, exp)

        # GH 5968
        iseries = Series(["3 day", "1 day 10m", "-2 day", NaT], dtype="m8[ns]")
        exp = Series([3, 2, 1, np.nan])
        iranks = iseries.rank()
        tm.assert_series_equal(iranks, exp)

        values = np.array(
            [-50, -1, -1e-20, -1e-25, -1e-50, 0, 1e-40, 1e-20, 1e-10, 2, 40],
            dtype="float64",
        )
        random_order = np.random.default_rng(2).permutation(len(values))
        iseries = Series(values[random_order])
        exp = Series(random_order + 1.0, dtype="float64")
        iranks = iseries.rank()
        tm.assert_series_equal(iranks, exp)

    def test_rank_categorical(self):
        # GH issue #15420 rank incorrectly orders ordered categories

        # Test ascending/descending ranking for ordered categoricals
        exp = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        exp_desc = Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        ordered = Series(
            ["first", "second", "third", "fourth", "fifth", "sixth"]
        ).astype(
            CategoricalDtype(
                categories=["first", "second", "third", "fourth", "fifth", "sixth"],
                ordered=True,
            )
        )
        tm.assert_series_equal(ordered.rank(), exp)
        tm.assert_series_equal(ordered.rank(ascending=False), exp_desc)

        # Unordered categoricals should be ranked as objects
        unordered = Series(
            ["first", "second", "third", "fourth", "fifth", "sixth"]
        ).astype(
            CategoricalDtype(
                categories=["first", "second", "third", "fourth", "fifth", "sixth"],
                ordered=False,
            )
        )
        exp_unordered = Series([2.0, 4.0, 6.0, 3.0, 1.0, 5.0])
        res = unordered.rank()
        tm.assert_series_equal(res, exp_unordered)

        unordered1 = Series([1, 2, 3, 4, 5, 6]).astype(
            CategoricalDtype([1, 2, 3, 4, 5, 6], False)
        )
        exp_unordered1 = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        res1 = unordered1.rank()
        tm.assert_series_equal(res1, exp_unordered1)

        # Test na_option for rank data
        na_ser = Series(
            ["first", "second", "third", "fourth", "fifth", "sixth", np.nan]
        ).astype(
            CategoricalDtype(
                ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"],
                True,
            )
        )

        exp_top = Series([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0])
        exp_bot = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        exp_keep = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan])

        tm.assert_series_equal(na_ser.rank(na_option="top"), exp_top)
        tm.assert_series_equal(na_ser.rank(na_option="bottom"), exp_bot)
        tm.assert_series_equal(na_ser.rank(na_option="keep"), exp_keep)

        # Test na_option for rank data with ascending False
        exp_top = Series([7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        exp_bot = Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 7.0])
        exp_keep = Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0, np.nan])

        tm.assert_series_equal(na_ser.rank(na_option="top", ascending=False), exp_top)
        tm.assert_series_equal(
            na_ser.rank(na_option="bottom", ascending=False), exp_bot
        )
        tm.assert_series_equal(na_ser.rank(na_option="keep", ascending=False), exp_keep)

        # Test invalid values for na_option
        msg = "na_option must be one of 'keep', 'top', or 'bottom'"

        with pytest.raises(ValueError, match=msg):
            na_ser.rank(na_option="bad", ascending=False)

        # invalid type
        with pytest.raises(ValueError, match=msg):
            na_ser.rank(na_option=True, ascending=False)

        # Test with pct=True
        na_ser = Series(["first", "second", "third", "fourth", np.nan]).astype(
            CategoricalDtype(["first", "second", "third", "fourth"], True)
        )
        exp_top = Series([0.4, 0.6, 0.8, 1.0, 0.2])
        exp_bot = Series([0.2, 0.4, 0.6, 0.8, 1.0])
        exp_keep = Series([0.25, 0.5, 0.75, 1.0, np.nan])

        tm.assert_series_equal(na_ser.rank(na_option="top", pct=True), exp_top)
        tm.assert_series_equal(na_ser.rank(na_option="bottom", pct=True), exp_bot)
        tm.assert_series_equal(na_ser.rank(na_option="keep", pct=True), exp_keep)

    def test_rank_signature(self):
        s = Series([0, 1])
        s.rank(method="average")
        msg = "No axis named average for object type Series"
        with pytest.raises(ValueError, match=msg):
            s.rank("average")

    def test_rank_tie_methods(self, ser, results, dtype, using_infer_string):
        method, exp = results
        if (
            dtype == "int64"
            or dtype == "Int64"
            or (not using_infer_string and dtype == "str")
        ):
            pytest.skip("int64/str does not support NaN")

        ser = ser if dtype is None else ser.astype(dtype)
        result = ser.rank(method=method)
        tm.assert_series_equal(result, Series(exp, dtype=expected_dtype(dtype, method)))

    @pytest.mark.parametrize("ascending", [True, False])
    @pytest.mark.parametrize("method", ["average", "min", "max", "first", "dense"])
    @pytest.mark.parametrize("na_option", ["top", "bottom", "keep"])
    @pytest.mark.parametrize(
        "dtype, na_value, pos_inf, neg_inf",
        [
            ("object", None, Infinity(), NegInfinity()),
            ("float64", np.nan, np.inf, -np.inf),
            ("Float64", NA, np.inf, -np.inf),
            pytest.param(
                "float64[pyarrow]",
                NA,
                np.inf,
                -np.inf,
                marks=td.skip_if_no("pyarrow"),
            ),
        ],
    )
    def test_rank_tie_methods_on_infs_nans(
        self, method, na_option, ascending, dtype, na_value, pos_inf, neg_inf
    ):
        pytest.importorskip("scipy")
        if dtype == "float64[pyarrow]":
            if method == "average":
                exp_dtype = "float64[pyarrow]"
            else:
                exp_dtype = "uint64[pyarrow]"
        else:
            exp_dtype = "float64"

        chunk = 3
        in_arr = [neg_inf] * chunk + [na_value] * chunk + [pos_inf] * chunk
        iseries = Series(in_arr, dtype=dtype)
        exp_ranks = {
            "average": ([2, 2, 2], [5, 5, 5], [8, 8, 8]),
            "min": ([1, 1, 1], [4, 4, 4], [7, 7, 7]),
            "max": ([3, 3, 3], [6, 6, 6], [9, 9, 9]),
            "first": ([1, 2, 3], [4, 5, 6], [7, 8, 9]),
            "dense": ([1, 1, 1], [2, 2, 2], [3, 3, 3]),
        }
        ranks = exp_ranks[method]
        if na_option == "top":
            order = [ranks[1], ranks[0], ranks[2]]
        elif na_option == "bottom":
            order = [ranks[0], ranks[2], ranks[1]]
        else:
            order = [ranks[0], [np.nan] * chunk, ranks[1]]
        expected = order if ascending else order[::-1]
        expected = list(chain.from_iterable(expected))
        result = iseries.rank(method=method, na_option=na_option, ascending=ascending)
        tm.assert_series_equal(result, Series(expected, dtype=exp_dtype))

    def test_rank_desc_mix_nans_infs(self):
        # GH 19538
        # check descending ranking when mix nans and infs
        iseries = Series([1, np.nan, np.inf, -np.inf, 25])
        result = iseries.rank(ascending=False)
        exp = Series([3, np.nan, 1, 4, 2], dtype="float64")
        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize("method", ["average", "min", "max", "first", "dense"])
    @pytest.mark.parametrize(
        "op, value",
        [
            [operator.add, 0],
            [operator.add, 1e6],
            [operator.mul, 1e-6],
        ],
    )
    def test_rank_methods_series(self, method, op, value):
        sp_stats = pytest.importorskip("scipy.stats")

        xs = np.random.default_rng(2).standard_normal(9)
        xs = np.concatenate([xs[i:] for i in range(0, 9, 2)])  # add duplicates
        np.random.default_rng(2).shuffle(xs)

        index = [chr(ord("a") + i) for i in range(len(xs))]
        vals = op(xs, value)
        ts = Series(vals, index=index)
        result = ts.rank(method=method)
        sprank = sp_stats.rankdata(vals, method if method != "first" else "ordinal")
        expected = Series(sprank, index=index).astype("float64")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ser, exp",
        [
            ([1], [1]),
            ([2], [1]),
            ([0], [1]),
            ([2, 2], [1, 1]),
            ([1, 2, 3], [1, 2, 3]),
            ([4, 2, 1], [3, 2, 1]),
            ([1, 1, 5, 5, 3], [1, 1, 3, 3, 2]),
            ([-5, -4, -3, -2, -1], [1, 2, 3, 4, 5]),
        ],
    )
    def test_rank_dense_method(self, dtype, ser, exp):
        if ser[0] < 0 and dtype.startswith("str"):
            exp = exp[::-1]
        s = Series(ser).astype(dtype)
        result = s.rank(method="dense")
        expected = Series(exp).astype(expected_dtype(dtype, "dense"))
        tm.assert_series_equal(result, expected)

    def test_rank_descending(self, ser, results, dtype, using_infer_string):
        method, _ = results
        if dtype == "int64" or (not using_infer_string and dtype == "str"):
            s = ser.dropna()
        else:
            s = ser.astype(dtype)

        res = s.rank(ascending=False)
        if dtype.startswith("str"):
            expected = (s.astype("float64").max() - s.astype("float64")).rank()
        else:
            expected = (s.max() - s).rank()
        tm.assert_series_equal(res, expected.astype(expected_dtype(dtype, "average")))

        if dtype.startswith("str"):
            expected = (s.astype("float64").max() - s.astype("float64")).rank(
                method=method
            )
        else:
            expected = (s.max() - s).rank(method=method)
        res2 = s.rank(method=method, ascending=False)
        tm.assert_series_equal(res2, expected.astype(expected_dtype(dtype, method)))

    def test_rank_int(self, ser, results):
        method, exp = results
        s = ser.dropna().astype("i8")

        result = s.rank(method=method)
        expected = Series(exp).dropna()
        expected.index = result.index
        tm.assert_series_equal(result, expected)

    def test_rank_object_bug(self):
        # GH 13445

        # smoke tests
        Series([np.nan] * 32).astype(object).rank(ascending=True)
        Series([np.nan] * 32).astype(object).rank(ascending=False)

    def test_rank_modify_inplace(self):
        # GH 18521
        # Check rank does not mutate series
        s = Series([Timestamp("2017-01-05 10:20:27.569000"), NaT])
        expected = s.copy()

        s.rank()
        result = s
        tm.assert_series_equal(result, expected)

    def test_rank_ea_small_values(self):
        # GH#52471
        ser = Series(
            [5.4954145e29, -9.791984e-21, 9.3715776e-26, NA, 1.8790257e-28],
            dtype="Float64",
        )
        result = ser.rank(method="min")
        expected = Series([4, 1, 3, np.nan, 2])
        tm.assert_series_equal(result, expected)


# GH15630, pct should be on 100% basis when method='dense'


@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.0, 1.0]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 2, 2.0 / 2, 2.0 / 2]),
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),
        ([1, 1, 5, 5, 3], [1.0 / 3, 1.0 / 3, 3.0 / 3, 3.0 / 3, 2.0 / 3]),
        ([1, 1, 3, 3, 5, 5], [1.0 / 3, 1.0 / 3, 2.0 / 3, 2.0 / 3, 3.0 / 3, 3.0 / 3]),
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),
    ],
)
def test_rank_dense_pct(dtype, ser, exp):
    if ser[0] < 0 and dtype.startswith("str"):
        exp = exp[::-1]
    s = Series(ser).astype(dtype)
    result = s.rank(method="dense", pct=True)
    expected = Series(exp).astype(expected_dtype(dtype, "dense", pct=True))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.0 / 2, 1.0 / 2]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 3, 2.0 / 3, 2.0 / 3]),
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),
        ([1, 1, 5, 5, 3], [1.0 / 5, 1.0 / 5, 4.0 / 5, 4.0 / 5, 3.0 / 5]),
        ([1, 1, 3, 3, 5, 5], [1.0 / 6, 1.0 / 6, 3.0 / 6, 3.0 / 6, 5.0 / 6, 5.0 / 6]),
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),
    ],
)
def test_rank_min_pct(dtype, ser, exp):
    if ser[0] < 0 and dtype.startswith("str"):
        exp = exp[::-1]
    s = Series(ser).astype(dtype)
    result = s.rank(method="min", pct=True)
    expected = Series(exp).astype(expected_dtype(dtype, "min", pct=True))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.0, 1.0]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 3, 3.0 / 3, 3.0 / 3]),
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),
        ([1, 1, 5, 5, 3], [2.0 / 5, 2.0 / 5, 5.0 / 5, 5.0 / 5, 3.0 / 5]),
        ([1, 1, 3, 3, 5, 5], [2.0 / 6, 2.0 / 6, 4.0 / 6, 4.0 / 6, 6.0 / 6, 6.0 / 6]),
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),
    ],
)
def test_rank_max_pct(dtype, ser, exp):
    if ser[0] < 0 and dtype.startswith("str"):
        exp = exp[::-1]
    s = Series(ser).astype(dtype)
    result = s.rank(method="max", pct=True)
    expected = Series(exp).astype(expected_dtype(dtype, "max", pct=True))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.5 / 2, 1.5 / 2]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 3, 2.5 / 3, 2.5 / 3]),
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),
        ([1, 1, 5, 5, 3], [1.5 / 5, 1.5 / 5, 4.5 / 5, 4.5 / 5, 3.0 / 5]),
        ([1, 1, 3, 3, 5, 5], [1.5 / 6, 1.5 / 6, 3.5 / 6, 3.5 / 6, 5.5 / 6, 5.5 / 6]),
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),
    ],
)
def test_rank_average_pct(dtype, ser, exp):
    if ser[0] < 0 and dtype.startswith("str"):
        exp = exp[::-1]
    s = Series(ser).astype(dtype)
    result = s.rank(method="average", pct=True)
    expected = Series(exp).astype(expected_dtype(dtype, "average", pct=True))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.0 / 2, 2.0 / 2.0]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),
        ([1, 1, 5, 5, 3], [1.0 / 5, 2.0 / 5, 4.0 / 5, 5.0 / 5, 3.0 / 5]),
        ([1, 1, 3, 3, 5, 5], [1.0 / 6, 2.0 / 6, 3.0 / 6, 4.0 / 6, 5.0 / 6, 6.0 / 6]),
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),
    ],
)
def test_rank_first_pct(dtype, ser, exp):
    if ser[0] < 0 and dtype.startswith("str"):
        exp = exp[::-1]
    s = Series(ser).astype(dtype)
    result = s.rank(method="first", pct=True)
    expected = Series(exp).astype(expected_dtype(dtype, "first", pct=True))
    tm.assert_series_equal(result, expected)


@pytest.mark.single_cpu
def test_pct_max_many_rows():
    # GH 18271
    s = Series(np.arange(2**24 + 1))
    result = s.rank(pct=True).max()
    assert result == 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\style\test_format.py ===
import numpy as np
import pytest

from pandas import (
    NA,
    DataFrame,
    IndexSlice,
    MultiIndex,
    NaT,
    Timestamp,
    option_context,
)

pytest.importorskip("jinja2")
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape


@pytest.fixture
def df():
    return DataFrame(
        data=[[0, -0.609], [1, -1.228]],
        columns=["A", "B"],
        index=["x", "y"],
    )


@pytest.fixture
def styler(df):
    return Styler(df, uuid_len=0)


@pytest.fixture
def df_multi():
    return DataFrame(
        data=np.arange(16).reshape(4, 4),
        columns=MultiIndex.from_product([["A", "B"], ["a", "b"]]),
        index=MultiIndex.from_product([["X", "Y"], ["x", "y"]]),
    )


@pytest.fixture
def styler_multi(df_multi):
    return Styler(df_multi, uuid_len=0)


def test_display_format(styler):
    ctx = styler.format("{:0.1f}")._translate(True, True)
    assert all(["display_value" in c for c in row] for row in ctx["body"])
    assert all([len(c["display_value"]) <= 3 for c in row[1:]] for row in ctx["body"])
    assert len(ctx["body"][0][1]["display_value"].lstrip("-")) <= 3


@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize("columns", [True, False])
def test_display_format_index(styler, index, columns):
    exp_index = ["x", "y"]
    if index:
        styler.format_index(lambda v: v.upper(), axis=0)  # test callable
        exp_index = ["X", "Y"]

    exp_columns = ["A", "B"]
    if columns:
        styler.format_index("*{}*", axis=1)  # test string
        exp_columns = ["*A*", "*B*"]

    ctx = styler._translate(True, True)

    for r, row in enumerate(ctx["body"]):
        assert row[0]["display_value"] == exp_index[r]

    for c, col in enumerate(ctx["head"][1:]):
        assert col["display_value"] == exp_columns[c]


def test_format_dict(styler):
    ctx = styler.format({"A": "{:0.1f}", "B": "{0:.2%}"})._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "0.0"
    assert ctx["body"][0][2]["display_value"] == "-60.90%"


def test_format_index_dict(styler):
    ctx = styler.format_index({0: lambda v: v.upper()})._translate(True, True)
    for i, val in enumerate(["X", "Y"]):
        assert ctx["body"][i][0]["display_value"] == val


def test_format_string(styler):
    ctx = styler.format("{:.2f}")._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "0.00"
    assert ctx["body"][0][2]["display_value"] == "-0.61"
    assert ctx["body"][1][1]["display_value"] == "1.00"
    assert ctx["body"][1][2]["display_value"] == "-1.23"


def test_format_callable(styler):
    ctx = styler.format(lambda v: "neg" if v < 0 else "pos")._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "pos"
    assert ctx["body"][0][2]["display_value"] == "neg"
    assert ctx["body"][1][1]["display_value"] == "pos"
    assert ctx["body"][1][2]["display_value"] == "neg"


def test_format_with_na_rep():
    # GH 21527 28358
    df = DataFrame([[None, None], [1.1, 1.2]], columns=["A", "B"])

    ctx = df.style.format(None, na_rep="-")._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "-"
    assert ctx["body"][0][2]["display_value"] == "-"

    ctx = df.style.format("{:.2%}", na_rep="-")._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "-"
    assert ctx["body"][0][2]["display_value"] == "-"
    assert ctx["body"][1][1]["display_value"] == "110.00%"
    assert ctx["body"][1][2]["display_value"] == "120.00%"

    ctx = df.style.format("{:.2%}", na_rep="-", subset=["B"])._translate(True, True)
    assert ctx["body"][0][2]["display_value"] == "-"
    assert ctx["body"][1][2]["display_value"] == "120.00%"


def test_format_index_with_na_rep():
    df = DataFrame([[1, 2, 3, 4, 5]], columns=["A", None, np.nan, NaT, NA])
    ctx = df.style.format_index(None, na_rep="--", axis=1)._translate(True, True)
    assert ctx["head"][0][1]["display_value"] == "A"
    for i in [2, 3, 4, 5]:
        assert ctx["head"][0][i]["display_value"] == "--"


def test_format_non_numeric_na():
    # GH 21527 28358
    df = DataFrame(
        {
            "object": [None, np.nan, "foo"],
            "datetime": [None, NaT, Timestamp("20120101")],
        }
    )
    ctx = df.style.format(None, na_rep="-")._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "-"
    assert ctx["body"][0][2]["display_value"] == "-"
    assert ctx["body"][1][1]["display_value"] == "-"
    assert ctx["body"][1][2]["display_value"] == "-"


@pytest.mark.parametrize(
    "func, attr, kwargs",
    [
        ("format", "_display_funcs", {}),
        ("format_index", "_display_funcs_index", {"axis": 0}),
        ("format_index", "_display_funcs_columns", {"axis": 1}),
    ],
)
def test_format_clear(styler, func, attr, kwargs):
    assert (0, 0) not in getattr(styler, attr)  # using default
    getattr(styler, func)("{:.2f}", **kwargs)
    assert (0, 0) in getattr(styler, attr)  # formatter is specified
    getattr(styler, func)(**kwargs)
    assert (0, 0) not in getattr(styler, attr)  # formatter cleared to default


@pytest.mark.parametrize(
    "escape, exp",
    [
        ("html", "&lt;&gt;&amp;&#34;%$#_{}~^\\~ ^ \\ "),
        (
            "latex",
            '<>\\&"\\%\\$\\#\\_\\{\\}\\textasciitilde \\textasciicircum '
            "\\textbackslash \\textasciitilde \\space \\textasciicircum \\space "
            "\\textbackslash \\space ",
        ),
    ],
)
def test_format_escape_html(escape, exp):
    chars = '<>&"%$#_{}~^\\~ ^ \\ '
    df = DataFrame([[chars]])

    s = Styler(df, uuid_len=0).format("&{0}&", escape=None)
    expected = f'<td id="T__row0_col0" class="data row0 col0" >&{chars}&</td>'
    assert expected in s.to_html()

    # only the value should be escaped before passing to the formatter
    s = Styler(df, uuid_len=0).format("&{0}&", escape=escape)
    expected = f'<td id="T__row0_col0" class="data row0 col0" >&{exp}&</td>'
    assert expected in s.to_html()

    # also test format_index()
    styler = Styler(DataFrame(columns=[chars]), uuid_len=0)
    styler.format_index("&{0}&", escape=None, axis=1)
    assert styler._translate(True, True)["head"][0][1]["display_value"] == f"&{chars}&"
    styler.format_index("&{0}&", escape=escape, axis=1)
    assert styler._translate(True, True)["head"][0][1]["display_value"] == f"&{exp}&"


@pytest.mark.parametrize(
    "chars, expected",
    [
        (
            r"$ \$&%#_{}~^\ $ &%#_{}~^\ $",
            "".join(
                [
                    r"$ \$&%#_{}~^\ $ ",
                    r"\&\%\#\_\{\}\textasciitilde \textasciicircum ",
                    r"\textbackslash \space \$",
                ]
            ),
        ),
        (
            r"\( &%#_{}~^\ \) &%#_{}~^\ \(",
            "".join(
                [
                    r"\( &%#_{}~^\ \) ",
                    r"\&\%\#\_\{\}\textasciitilde \textasciicircum ",
                    r"\textbackslash \space \textbackslash (",
                ]
            ),
        ),
        (
            r"$\&%#_{}^\$",
            r"\$\textbackslash \&\%\#\_\{\}\textasciicircum \textbackslash \$",
        ),
        (
            r"$ \frac{1}{2} $ \( \frac{1}{2} \)",
            "".join(
                [
                    r"$ \frac{1}{2} $",
                    r" \textbackslash ( \textbackslash frac\{1\}\{2\} \textbackslash )",
                ]
            ),
        ),
    ],
)
def test_format_escape_latex_math(chars, expected):
    # GH 51903
    # latex-math escape works for each DataFrame cell separately. If we have
    # a combination of dollar signs and brackets, the dollar sign would apply.
    df = DataFrame([[chars]])
    s = df.style.format("{0}", escape="latex-math")
    assert s._translate(True, True)["body"][0][1]["display_value"] == expected


def test_format_escape_na_rep():
    # tests the na_rep is not escaped
    df = DataFrame([['<>&"', None]])
    s = Styler(df, uuid_len=0).format("X&{0}>X", escape="html", na_rep="&")
    ex = '<td id="T__row0_col0" class="data row0 col0" >X&&lt;&gt;&amp;&#34;>X</td>'
    expected2 = '<td id="T__row0_col1" class="data row0 col1" >&</td>'
    assert ex in s.to_html()
    assert expected2 in s.to_html()

    # also test for format_index()
    df = DataFrame(columns=['<>&"', None])
    styler = Styler(df, uuid_len=0)
    styler.format_index("X&{0}>X", escape="html", na_rep="&", axis=1)
    ctx = styler._translate(True, True)
    assert ctx["head"][0][1]["display_value"] == "X&&lt;&gt;&amp;&#34;>X"
    assert ctx["head"][0][2]["display_value"] == "&"


def test_format_escape_floats(styler):
    # test given formatter for number format is not impacted by escape
    s = styler.format("{:.1f}", escape="html")
    for expected in [">0.0<", ">1.0<", ">-1.2<", ">-0.6<"]:
        assert expected in s.to_html()
    # tests precision of floats is not impacted by escape
    s = styler.format(precision=1, escape="html")
    for expected in [">0<", ">1<", ">-1.2<", ">-0.6<"]:
        assert expected in s.to_html()


@pytest.mark.parametrize("formatter", [5, True, [2.0]])
@pytest.mark.parametrize("func", ["format", "format_index"])
def test_format_raises(styler, formatter, func):
    with pytest.raises(TypeError, match="expected str or callable"):
        getattr(styler, func)(formatter)


@pytest.mark.parametrize(
    "precision, expected",
    [
        (1, ["1.0", "2.0", "3.2", "4.6"]),
        (2, ["1.00", "2.01", "3.21", "4.57"]),
        (3, ["1.000", "2.009", "3.212", "4.566"]),
    ],
)
def test_format_with_precision(precision, expected):
    # Issue #13257
    df = DataFrame([[1.0, 2.0090, 3.2121, 4.566]], columns=[1.0, 2.0090, 3.2121, 4.566])
    styler = Styler(df)
    styler.format(precision=precision)
    styler.format_index(precision=precision, axis=1)

    ctx = styler._translate(True, True)
    for col, exp in enumerate(expected):
        assert ctx["body"][0][col + 1]["display_value"] == exp  # format test
        assert ctx["head"][0][col + 1]["display_value"] == exp  # format_index test


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "level, expected",
    [
        (0, ["X", "X", "_", "_"]),  # level int
        ("zero", ["X", "X", "_", "_"]),  # level name
        (1, ["_", "_", "X", "X"]),  # other level int
        ("one", ["_", "_", "X", "X"]),  # other level name
        ([0, 1], ["X", "X", "X", "X"]),  # both levels
        ([0, "zero"], ["X", "X", "_", "_"]),  # level int and name simultaneous
        ([0, "one"], ["X", "X", "X", "X"]),  # both levels as int and name
        (["one", "zero"], ["X", "X", "X", "X"]),  # both level names, reversed
    ],
)
def test_format_index_level(axis, level, expected):
    midx = MultiIndex.from_arrays([["_", "_"], ["_", "_"]], names=["zero", "one"])
    df = DataFrame([[1, 2], [3, 4]])
    if axis == 0:
        df.index = midx
    else:
        df.columns = midx

    styler = df.style.format_index(lambda v: "X", level=level, axis=axis)
    ctx = styler._translate(True, True)

    if axis == 0:  # compare index
        result = [ctx["body"][s][0]["display_value"] for s in range(2)]
        result += [ctx["body"][s][1]["display_value"] for s in range(2)]
    else:  # compare columns
        result = [ctx["head"][0][s + 1]["display_value"] for s in range(2)]
        result += [ctx["head"][1][s + 1]["display_value"] for s in range(2)]

    assert expected == result


def test_format_subset():
    df = DataFrame([[0.1234, 0.1234], [1.1234, 1.1234]], columns=["a", "b"])
    ctx = df.style.format(
        {"a": "{:0.1f}", "b": "{0:.2%}"}, subset=IndexSlice[0, :]
    )._translate(True, True)
    expected = "0.1"
    raw_11 = "1.123400"
    assert ctx["body"][0][1]["display_value"] == expected
    assert ctx["body"][1][1]["display_value"] == raw_11
    assert ctx["body"][0][2]["display_value"] == "12.34%"

    ctx = df.style.format("{:0.1f}", subset=IndexSlice[0, :])._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == expected
    assert ctx["body"][1][1]["display_value"] == raw_11

    ctx = df.style.format("{:0.1f}", subset=IndexSlice["a"])._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == expected
    assert ctx["body"][0][2]["display_value"] == "0.123400"

    ctx = df.style.format("{:0.1f}", subset=IndexSlice[0, "a"])._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == expected
    assert ctx["body"][1][1]["display_value"] == raw_11

    ctx = df.style.format("{:0.1f}", subset=IndexSlice[[0, 1], ["a"]])._translate(
        True, True
    )
    assert ctx["body"][0][1]["display_value"] == expected
    assert ctx["body"][1][1]["display_value"] == "1.1"
    assert ctx["body"][0][2]["display_value"] == "0.123400"
    assert ctx["body"][1][2]["display_value"] == raw_11


@pytest.mark.parametrize("formatter", [None, "{:,.1f}"])
@pytest.mark.parametrize("decimal", [".", "*"])
@pytest.mark.parametrize("precision", [None, 2])
@pytest.mark.parametrize("func, col", [("format", 1), ("format_index", 0)])
def test_format_thousands(formatter, decimal, precision, func, col):
    styler = DataFrame([[1000000.123456789]], index=[1000000.123456789]).style
    result = getattr(styler, func)(  # testing float
        thousands="_", formatter=formatter, decimal=decimal, precision=precision
    )._translate(True, True)
    assert "1_000_000" in result["body"][0][col]["display_value"]

    styler = DataFrame([[1000000]], index=[1000000]).style
    result = getattr(styler, func)(  # testing int
        thousands="_", formatter=formatter, decimal=decimal, precision=precision
    )._translate(True, True)
    assert "1_000_000" in result["body"][0][col]["display_value"]

    styler = DataFrame([[1 + 1000000.123456789j]], index=[1 + 1000000.123456789j]).style
    result = getattr(styler, func)(  # testing complex
        thousands="_", formatter=formatter, decimal=decimal, precision=precision
    )._translate(True, True)
    assert "1_000_000" in result["body"][0][col]["display_value"]


@pytest.mark.parametrize("formatter", [None, "{:,.4f}"])
@pytest.mark.parametrize("thousands", [None, ",", "*"])
@pytest.mark.parametrize("precision", [None, 4])
@pytest.mark.parametrize("func, col", [("format", 1), ("format_index", 0)])
def test_format_decimal(formatter, thousands, precision, func, col):
    styler = DataFrame([[1000000.123456789]], index=[1000000.123456789]).style
    result = getattr(styler, func)(  # testing float
        decimal="_", formatter=formatter, thousands=thousands, precision=precision
    )._translate(True, True)
    assert "000_123" in result["body"][0][col]["display_value"]

    styler = DataFrame([[1 + 1000000.123456789j]], index=[1 + 1000000.123456789j]).style
    result = getattr(styler, func)(  # testing complex
        decimal="_", formatter=formatter, thousands=thousands, precision=precision
    )._translate(True, True)
    assert "000_123" in result["body"][0][col]["display_value"]


def test_str_escape_error():
    msg = "`escape` only permitted in {'html', 'latex', 'latex-math'}, got "
    with pytest.raises(ValueError, match=msg):
        _str_escape("text", "bad_escape")

    with pytest.raises(ValueError, match=msg):
        _str_escape("text", [])

    _str_escape(2.00, "bad_escape")  # OK since dtype is float


def test_long_int_formatting():
    df = DataFrame(data=[[1234567890123456789]], columns=["test"])
    styler = df.style
    ctx = styler._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "1234567890123456789"

    styler = df.style.format(thousands="_")
    ctx = styler._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "1_234_567_890_123_456_789"


def test_format_options():
    df = DataFrame({"int": [2000, 1], "float": [1.009, None], "str": ["&<", "&~"]})
    ctx = df.style._translate(True, True)

    # test option: na_rep
    assert ctx["body"][1][2]["display_value"] == "nan"
    with option_context("styler.format.na_rep", "MISSING"):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op["body"][1][2]["display_value"] == "MISSING"

    # test option: decimal and precision
    assert ctx["body"][0][2]["display_value"] == "1.009000"
    with option_context("styler.format.decimal", "_"):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op["body"][0][2]["display_value"] == "1_009000"
    with option_context("styler.format.precision", 2):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op["body"][0][2]["display_value"] == "1.01"

    # test option: thousands
    assert ctx["body"][0][1]["display_value"] == "2000"
    with option_context("styler.format.thousands", "_"):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op["body"][0][1]["display_value"] == "2_000"

    # test option: escape
    assert ctx["body"][0][3]["display_value"] == "&<"
    assert ctx["body"][1][3]["display_value"] == "&~"
    with option_context("styler.format.escape", "html"):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op["body"][0][3]["display_value"] == "&amp;&lt;"
    with option_context("styler.format.escape", "latex"):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op["body"][1][3]["display_value"] == "\\&\\textasciitilde "
    with option_context("styler.format.escape", "latex-math"):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op["body"][1][3]["display_value"] == "\\&\\textasciitilde "

    # test option: formatter
    with option_context("styler.format.formatter", {"int": "{:,.2f}"}):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op["body"][0][1]["display_value"] == "2,000.00"


def test_precision_zero(df):
    styler = Styler(df, precision=0)
    ctx = styler._translate(True, True)
    assert ctx["body"][0][2]["display_value"] == "-1"
    assert ctx["body"][1][2]["display_value"] == "-1"


@pytest.mark.parametrize(
    "formatter, exp",
    [
        (lambda x: f"{x:.3f}", "9.000"),
        ("{:.2f}", "9.00"),
        ({0: "{:.1f}"}, "9.0"),
        (None, "9"),
    ],
)
def test_formatter_options_validator(formatter, exp):
    df = DataFrame([[9]])
    with option_context("styler.format.formatter", formatter):
        assert f" {exp} " in df.style.to_latex()


def test_formatter_options_raises():
    msg = "Value must be an instance of"
    with pytest.raises(ValueError, match=msg):
        with option_context("styler.format.formatter", ["bad", "type"]):
            DataFrame().style.to_latex()


def test_1level_multiindex():
    # GH 43383
    midx = MultiIndex.from_product([[1, 2]], names=[""])
    df = DataFrame(-1, index=midx, columns=[0, 1])
    ctx = df.style._translate(True, True)
    assert ctx["body"][0][0]["display_value"] == "1"
    assert ctx["body"][0][0]["is_visible"] is True
    assert ctx["body"][1][0]["display_value"] == "2"
    assert ctx["body"][1][0]["is_visible"] is True


def test_boolean_format():
    # gh 46384: booleans do not collapse to integer representation on display
    df = DataFrame([[True, False]])
    ctx = df.style._translate(True, True)
    assert ctx["body"][0][1]["display_value"] is True
    assert ctx["body"][0][2]["display_value"] is False


@pytest.mark.parametrize(
    "hide, labels",
    [
        (False, [1, 2]),
        (True, [1, 2, 3, 4]),
    ],
)
def test_relabel_raise_length(styler_multi, hide, labels):
    if hide:
        styler_multi.hide(axis=0, subset=[("X", "x"), ("Y", "y")])
    with pytest.raises(ValueError, match="``labels`` must be of length equal"):
        styler_multi.relabel_index(labels=labels)


def test_relabel_index(styler_multi):
    labels = [(1, 2), (3, 4)]
    styler_multi.hide(axis=0, subset=[("X", "x"), ("Y", "y")])
    styler_multi.relabel_index(labels=labels)
    ctx = styler_multi._translate(True, True)
    assert {"value": "X", "display_value": 1}.items() <= ctx["body"][0][0].items()
    assert {"value": "y", "display_value": 2}.items() <= ctx["body"][0][1].items()
    assert {"value": "Y", "display_value": 3}.items() <= ctx["body"][1][0].items()
    assert {"value": "x", "display_value": 4}.items() <= ctx["body"][1][1].items()


def test_relabel_columns(styler_multi):
    labels = [(1, 2), (3, 4)]
    styler_multi.hide(axis=1, subset=[("A", "a"), ("B", "b")])
    styler_multi.relabel_index(axis=1, labels=labels)
    ctx = styler_multi._translate(True, True)
    assert {"value": "A", "display_value": 1}.items() <= ctx["head"][0][3].items()
    assert {"value": "B", "display_value": 3}.items() <= ctx["head"][0][4].items()
    assert {"value": "b", "display_value": 2}.items() <= ctx["head"][1][3].items()
    assert {"value": "a", "display_value": 4}.items() <= ctx["head"][1][4].items()


def test_relabel_roundtrip(styler):
    styler.relabel_index(["{}", "{}"])
    ctx = styler._translate(True, True)
    assert {"value": "x", "display_value": "x"}.items() <= ctx["body"][0][0].items()
    assert {"value": "y", "display_value": "y"}.items() <= ctx["body"][1][0].items()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\crypto\tests\test_crypto.py ===
from sympy.core import symbols
from sympy.crypto.crypto import (cycle_list,
      encipher_shift, encipher_affine, encipher_substitution,
      check_and_join, encipher_vigenere, decipher_vigenere,
      encipher_hill, decipher_hill, encipher_bifid5, encipher_bifid6,
      bifid5_square, bifid6_square, bifid5, bifid6,
      decipher_bifid5, decipher_bifid6, encipher_kid_rsa,
      decipher_kid_rsa, kid_rsa_private_key, kid_rsa_public_key,
      decipher_rsa, rsa_private_key, rsa_public_key, encipher_rsa,
      lfsr_connection_polynomial, lfsr_autocorrelation, lfsr_sequence,
      encode_morse, decode_morse, elgamal_private_key, elgamal_public_key,
      encipher_elgamal, decipher_elgamal, dh_private_key, dh_public_key,
      dh_shared_key, decipher_shift, decipher_affine, encipher_bifid,
      decipher_bifid, bifid_square, padded_key, uniq, decipher_gm,
      encipher_gm, gm_public_key, gm_private_key, encipher_bg, decipher_bg,
      bg_private_key, bg_public_key, encipher_rot13, decipher_rot13,
      encipher_atbash, decipher_atbash, NonInvertibleCipherWarning,
      encipher_railfence, decipher_railfence)
from sympy.external.gmpy import gcd
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF

from sympy.testing.pytest import raises, warns

from sympy.core.random import randrange

def test_encipher_railfence():
    assert encipher_railfence("hello world",2) == "hlowrdel ol"
    assert encipher_railfence("hello world",3) == "horel ollwd"
    assert encipher_railfence("hello world",4) == "hwe olordll"

def test_decipher_railfence():
    assert decipher_railfence("hlowrdel ol",2) == "hello world"
    assert decipher_railfence("horel ollwd",3) == "hello world"
    assert decipher_railfence("hwe olordll",4) == "hello world"


def test_cycle_list():
    assert cycle_list(3, 4) == [3, 0, 1, 2]
    assert cycle_list(-1, 4) == [3, 0, 1, 2]
    assert cycle_list(1, 4) == [1, 2, 3, 0]


def test_encipher_shift():
    assert encipher_shift("ABC", 0) == "ABC"
    assert encipher_shift("ABC", 1) == "BCD"
    assert encipher_shift("ABC", -1) == "ZAB"
    assert decipher_shift("ZAB", -1) == "ABC"

def test_encipher_rot13():
    assert encipher_rot13("ABC") == "NOP"
    assert encipher_rot13("NOP") == "ABC"
    assert decipher_rot13("ABC") == "NOP"
    assert decipher_rot13("NOP") == "ABC"


def test_encipher_affine():
    assert encipher_affine("ABC", (1, 0)) == "ABC"
    assert encipher_affine("ABC", (1, 1)) == "BCD"
    assert encipher_affine("ABC", (-1, 0)) == "AZY"
    assert encipher_affine("ABC", (-1, 1), symbols="ABCD") == "BAD"
    assert encipher_affine("123", (-1, 1), symbols="1234") == "214"
    assert encipher_affine("ABC", (3, 16)) == "QTW"
    assert decipher_affine("QTW", (3, 16)) == "ABC"

def test_encipher_atbash():
    assert encipher_atbash("ABC") == "ZYX"
    assert encipher_atbash("ZYX") == "ABC"
    assert decipher_atbash("ABC") == "ZYX"
    assert decipher_atbash("ZYX") == "ABC"

def test_encipher_substitution():
    assert encipher_substitution("ABC", "BAC", "ABC") == "BAC"
    assert encipher_substitution("123", "1243", "1234") == "124"


def test_check_and_join():
    assert check_and_join("abc") == "abc"
    assert check_and_join(uniq("aaabc")) == "abc"
    assert check_and_join("ab c".split()) == "abc"
    assert check_and_join("abc", "a", filter=True) == "a"
    raises(ValueError, lambda: check_and_join('ab', 'a'))


def test_encipher_vigenere():
    assert encipher_vigenere("ABC", "ABC") == "ACE"
    assert encipher_vigenere("ABC", "ABC", symbols="ABCD") == "ACA"
    assert encipher_vigenere("ABC", "AB", symbols="ABCD") == "ACC"
    assert encipher_vigenere("AB", "ABC", symbols="ABCD") == "AC"
    assert encipher_vigenere("A", "ABC", symbols="ABCD") == "A"


def test_decipher_vigenere():
    assert decipher_vigenere("ABC", "ABC") == "AAA"
    assert decipher_vigenere("ABC", "ABC", symbols="ABCD") == "AAA"
    assert decipher_vigenere("ABC", "AB", symbols="ABCD") == "AAC"
    assert decipher_vigenere("AB", "ABC", symbols="ABCD") == "AA"
    assert decipher_vigenere("A", "ABC", symbols="ABCD") == "A"


def test_encipher_hill():
    A = Matrix(2, 2, [1, 2, 3, 5])
    assert encipher_hill("ABCD", A) == "CFIV"
    A = Matrix(2, 2, [1, 0, 0, 1])
    assert encipher_hill("ABCD", A) == "ABCD"
    assert encipher_hill("ABCD", A, symbols="ABCD") == "ABCD"
    A = Matrix(2, 2, [1, 2, 3, 5])
    assert encipher_hill("ABCD", A, symbols="ABCD") == "CBAB"
    assert encipher_hill("AB", A, symbols="ABCD") == "CB"
    # message length, n, does not need to be a multiple of k;
    # it is padded
    assert encipher_hill("ABA", A) == "CFGC"
    assert encipher_hill("ABA", A, pad="Z") == "CFYV"


def test_decipher_hill():
    A = Matrix(2, 2, [1, 2, 3, 5])
    assert decipher_hill("CFIV", A) == "ABCD"
    A = Matrix(2, 2, [1, 0, 0, 1])
    assert decipher_hill("ABCD", A) == "ABCD"
    assert decipher_hill("ABCD", A, symbols="ABCD") == "ABCD"
    A = Matrix(2, 2, [1, 2, 3, 5])
    assert decipher_hill("CBAB", A, symbols="ABCD") == "ABCD"
    assert decipher_hill("CB", A, symbols="ABCD") == "AB"
    # n does not need to be a multiple of k
    assert decipher_hill("CFA", A) == "ABAA"


def test_encipher_bifid5():
    assert encipher_bifid5("AB", "AB") == "AB"
    assert encipher_bifid5("AB", "CD") == "CO"
    assert encipher_bifid5("ab", "c") == "CH"
    assert encipher_bifid5("a bc", "b") == "BAC"


def test_bifid5_square():
    A = bifid5
    f = lambda i, j: symbols(A[5*i + j])
    M = Matrix(5, 5, f)
    assert bifid5_square("") == M


def test_decipher_bifid5():
    assert decipher_bifid5("AB", "AB") == "AB"
    assert decipher_bifid5("CO", "CD") == "AB"
    assert decipher_bifid5("ch", "c") == "AB"
    assert decipher_bifid5("b ac", "b") == "ABC"


def test_encipher_bifid6():
    assert encipher_bifid6("AB", "AB") == "AB"
    assert encipher_bifid6("AB", "CD") == "CP"
    assert encipher_bifid6("ab", "c") == "CI"
    assert encipher_bifid6("a bc", "b") == "BAC"


def test_decipher_bifid6():
    assert decipher_bifid6("AB", "AB") == "AB"
    assert decipher_bifid6("CP", "CD") == "AB"
    assert decipher_bifid6("ci", "c") == "AB"
    assert decipher_bifid6("b ac", "b") == "ABC"


def test_bifid6_square():
    A = bifid6
    f = lambda i, j: symbols(A[6*i + j])
    M = Matrix(6, 6, f)
    assert bifid6_square("") == M


def test_rsa_public_key():
    assert rsa_public_key(2, 3, 1) == (6, 1)
    assert rsa_public_key(5, 3, 3) == (15, 3)

    with warns(NonInvertibleCipherWarning):
        assert rsa_public_key(2, 2, 1) == (4, 1)
        assert rsa_public_key(8, 8, 8) is False


def test_rsa_private_key():
    assert rsa_private_key(2, 3, 1) == (6, 1)
    assert rsa_private_key(5, 3, 3) == (15, 3)
    assert rsa_private_key(23,29,5) == (667,493)

    with warns(NonInvertibleCipherWarning):
        assert rsa_private_key(2, 2, 1) == (4, 1)
        assert rsa_private_key(8, 8, 8) is False


def test_rsa_large_key():
    # Sample from
    # http://www.herongyang.com/Cryptography/JCE-Public-Key-RSA-Private-Public-Key-Pair-Sample.html
    p = int('101565610013301240713207239558950144682174355406589305284428666'\
        '903702505233009')
    q = int('894687191887545488935455605955948413812376003053143521429242133'\
        '12069293984003')
    e = int('65537')
    d = int('893650581832704239530398858744759129594796235440844479456143566'\
        '6999402846577625762582824202269399672579058991442587406384754958587'\
        '400493169361356902030209')
    assert rsa_public_key(p, q, e) == (p*q, e)
    assert rsa_private_key(p, q, e) == (p*q, d)


def test_encipher_rsa():
    puk = rsa_public_key(2, 3, 1)
    assert encipher_rsa(2, puk) == 2
    puk = rsa_public_key(5, 3, 3)
    assert encipher_rsa(2, puk) == 8

    with warns(NonInvertibleCipherWarning):
        puk = rsa_public_key(2, 2, 1)
        assert encipher_rsa(2, puk) == 2


def test_decipher_rsa():
    prk = rsa_private_key(2, 3, 1)
    assert decipher_rsa(2, prk) == 2
    prk = rsa_private_key(5, 3, 3)
    assert decipher_rsa(8, prk) == 2

    with warns(NonInvertibleCipherWarning):
        prk = rsa_private_key(2, 2, 1)
        assert decipher_rsa(2, prk) == 2


def test_mutltiprime_rsa_full_example():
    # Test example from
    # https://iopscience.iop.org/article/10.1088/1742-6596/995/1/012030
    puk = rsa_public_key(2, 3, 5, 7, 11, 13, 7)
    prk = rsa_private_key(2, 3, 5, 7, 11, 13, 7)
    assert puk == (30030, 7)
    assert prk == (30030, 823)

    msg = 10
    encrypted = encipher_rsa(2 * msg - 15, puk)
    assert encrypted == 18065
    decrypted = (decipher_rsa(encrypted, prk) + 15) / 2
    assert decrypted == msg

    # Test example from
    # https://www.scirp.org/pdf/JCC_2018032215502008.pdf
    puk1 = rsa_public_key(53, 41, 43, 47, 41)
    prk1 = rsa_private_key(53, 41, 43, 47, 41)
    puk2 = rsa_public_key(53, 41, 43, 47, 97)
    prk2 = rsa_private_key(53, 41, 43, 47, 97)

    assert puk1 == (4391633, 41)
    assert prk1 == (4391633, 294041)
    assert puk2 == (4391633, 97)
    assert prk2 == (4391633, 455713)

    msg = 12321
    encrypted = encipher_rsa(encipher_rsa(msg, puk1), puk2)
    assert encrypted == 1081588
    decrypted = decipher_rsa(decipher_rsa(encrypted, prk2), prk1)
    assert decrypted == msg


def test_rsa_crt_extreme():
    p = int(
        '10177157607154245068023861503693082120906487143725062283406501' \
        '54082258226204046999838297167140821364638180697194879500245557' \
        '65445186962893346463841419427008800341257468600224049986260471' \
        '92257248163014468841725476918639415726709736077813632961290911' \
        '0256421232977833028677441206049309220354796014376698325101693')

    q = int(
        '28752342353095132872290181526607275886182793241660805077850801' \
        '75689512797754286972952273553128181861830576836289738668745250' \
        '34028199691128870676414118458442900035778874482624765513861643' \
        '27966696316822188398336199002306588703902894100476186823849595' \
        '103239410527279605442148285816149368667083114802852804976893')

    r = int(
        '17698229259868825776879500736350186838850961935956310134378261' \
        '89771862186717463067541369694816245225291921138038800171125596' \
        '07315449521981157084370187887650624061033066022458512942411841' \
        '18747893789972315277160085086164119879536041875335384844820566' \
        '0287479617671726408053319619892052000850883994343378882717849')

    s = int(
        '68925428438585431029269182233502611027091755064643742383515623' \
        '64321310582896893395529367074942808353187138794422745718419645' \
        '28291231865157212604266903677599180789896916456120289112752835' \
        '98502265889669730331688206825220074713977607415178738015831030' \
        '364290585369150502819743827343552098197095520550865360159439'
    )

    t = int(
        '69035483433453632820551311892368908779778144568711455301541094' \
        '31487047642322695357696860925747923189635033183069823820910521' \
        '71172909106797748883261493224162414050106920442445896819806600' \
        '15448444826108008217972129130625571421904893252804729877353352' \
        '739420480574842850202181462656251626522910618936534699566291'
    )

    e = 65537
    puk = rsa_public_key(p, q, r, s, t, e)
    prk = rsa_private_key(p, q, r, s, t, e)

    plaintext = 1000
    ciphertext_1 = encipher_rsa(plaintext, puk)
    ciphertext_2 = encipher_rsa(plaintext, puk, [p, q, r, s, t])
    assert ciphertext_1 == ciphertext_2
    assert decipher_rsa(ciphertext_1, prk) == \
        decipher_rsa(ciphertext_1, prk, [p, q, r, s, t])


def test_rsa_exhaustive():
    p, q = 61, 53
    e = 17
    puk = rsa_public_key(p, q, e, totient='Carmichael')
    prk = rsa_private_key(p, q, e, totient='Carmichael')

    for msg in range(puk[0]):
        encrypted = encipher_rsa(msg, puk)
        decrypted = decipher_rsa(encrypted, prk)
        try:
            assert decrypted == msg
        except AssertionError:
            raise AssertionError(
                "The RSA is not correctly decrypted " \
                "(Original : {}, Encrypted : {}, Decrypted : {})" \
                .format(msg, encrypted, decrypted)
                )


def test_rsa_multiprime_exhanstive():
    primes = [3, 5, 7, 11]
    e = 7
    args = primes + [e]
    puk = rsa_public_key(*args, totient='Carmichael')
    prk = rsa_private_key(*args, totient='Carmichael')
    n = puk[0]

    for msg in range(n):
        encrypted = encipher_rsa(msg, puk)
        decrypted = decipher_rsa(encrypted, prk)
        try:
            assert decrypted == msg
        except AssertionError:
            raise AssertionError(
                "The RSA is not correctly decrypted " \
                "(Original : {}, Encrypted : {}, Decrypted : {})" \
                .format(msg, encrypted, decrypted)
                )


def test_rsa_multipower_exhanstive():
    primes = [5, 5, 7]
    e = 7
    args = primes + [e]
    puk = rsa_public_key(*args, multipower=True)
    prk = rsa_private_key(*args, multipower=True)
    n = puk[0]

    for msg in range(n):
        if gcd(msg, n) != 1:
            continue

        encrypted = encipher_rsa(msg, puk)
        decrypted = decipher_rsa(encrypted, prk)
        try:
            assert decrypted == msg
        except AssertionError:
            raise AssertionError(
                "The RSA is not correctly decrypted " \
                "(Original : {}, Encrypted : {}, Decrypted : {})" \
                .format(msg, encrypted, decrypted)
                )


def test_kid_rsa_public_key():
    assert kid_rsa_public_key(1, 2, 1, 1) == (5, 2)
    assert kid_rsa_public_key(1, 2, 2, 1) == (8, 3)
    assert kid_rsa_public_key(1, 2, 1, 2) == (7, 2)


def test_kid_rsa_private_key():
    assert kid_rsa_private_key(1, 2, 1, 1) == (5, 3)
    assert kid_rsa_private_key(1, 2, 2, 1) == (8, 3)
    assert kid_rsa_private_key(1, 2, 1, 2) == (7, 4)


def test_encipher_kid_rsa():
    assert encipher_kid_rsa(1, (5, 2)) == 2
    assert encipher_kid_rsa(1, (8, 3)) == 3
    assert encipher_kid_rsa(1, (7, 2)) == 2


def test_decipher_kid_rsa():
    assert decipher_kid_rsa(2, (5, 3)) == 1
    assert decipher_kid_rsa(3, (8, 3)) == 1
    assert decipher_kid_rsa(2, (7, 4)) == 1


def test_encode_morse():
    assert encode_morse('ABC') == '.-|-...|-.-.'
    assert encode_morse('SMS ') == '...|--|...||'
    assert encode_morse('SMS\n') == '...|--|...||'
    assert encode_morse('') == ''
    assert encode_morse(' ') == '||'
    assert encode_morse(' ', sep='`') == '``'
    assert encode_morse(' ', sep='``') == '````'
    assert encode_morse('!@#$%^&*()_+') == '-.-.--|.--.-.|...-..-|-.--.|-.--.-|..--.-|.-.-.'
    assert encode_morse('12345') == '.----|..---|...--|....-|.....'
    assert encode_morse('67890') == '-....|--...|---..|----.|-----'


def test_decode_morse():
    assert decode_morse('-.-|.|-.--') == 'KEY'
    assert decode_morse('.-.|..-|-.||') == 'RUN'
    raises(KeyError, lambda: decode_morse('.....----'))


def test_lfsr_sequence():
    raises(TypeError, lambda: lfsr_sequence(1, [1], 1))
    raises(TypeError, lambda: lfsr_sequence([1], 1, 1))
    F = FF(2)
    assert lfsr_sequence([F(1)], [F(1)], 2) == [F(1), F(1)]
    assert lfsr_sequence([F(0)], [F(1)], 2) == [F(1), F(0)]
    F = FF(3)
    assert lfsr_sequence([F(1)], [F(1)], 2) == [F(1), F(1)]
    assert lfsr_sequence([F(0)], [F(2)], 2) == [F(2), F(0)]
    assert lfsr_sequence([F(1)], [F(2)], 2) == [F(2), F(2)]


def test_lfsr_autocorrelation():
    raises(TypeError, lambda: lfsr_autocorrelation(1, 2, 3))
    F = FF(2)
    s = lfsr_sequence([F(1), F(0)], [F(0), F(1)], 5)
    assert lfsr_autocorrelation(s, 2, 0) == 1
    assert lfsr_autocorrelation(s, 2, 1) == -1


def test_lfsr_connection_polynomial():
    F = FF(2)
    x = symbols("x")
    s = lfsr_sequence([F(1), F(0)], [F(0), F(1)], 5)
    assert lfsr_connection_polynomial(s) == x**2 + 1
    s = lfsr_sequence([F(1), F(1)], [F(0), F(1)], 5)
    assert lfsr_connection_polynomial(s) == x**2 + x + 1


def test_elgamal_private_key():
    a, b, _ = elgamal_private_key(digit=100)
    assert isprime(a)
    assert is_primitive_root(b, a)
    assert len(bin(a)) >= 102


def test_elgamal():
    dk = elgamal_private_key(5)
    ek = elgamal_public_key(dk)
    P = ek[0]
    assert P - 1 == decipher_elgamal(encipher_elgamal(P - 1, ek), dk)
    raises(ValueError, lambda: encipher_elgamal(P, dk))
    raises(ValueError, lambda: encipher_elgamal(-1, dk))


def test_dh_private_key():
    p, g, _ = dh_private_key(digit = 100)
    assert isprime(p)
    assert is_primitive_root(g, p)
    assert len(bin(p)) >= 102


def test_dh_public_key():
    p1, g1, a = dh_private_key(digit = 100)
    p2, g2, ga = dh_public_key((p1, g1, a))
    assert p1 == p2
    assert g1 == g2
    assert ga == pow(g1, a, p1)


def test_dh_shared_key():
    prk = dh_private_key(digit = 100)
    p, _, ga = dh_public_key(prk)
    b = randrange(2, p)
    sk = dh_shared_key((p, _, ga), b)
    assert sk == pow(ga, b, p)
    raises(ValueError, lambda: dh_shared_key((1031, 14, 565), 2000))


def test_padded_key():
    assert padded_key('b', 'ab') == 'ba'
    raises(ValueError, lambda: padded_key('ab', 'ace'))
    raises(ValueError, lambda: padded_key('ab', 'abba'))


def test_bifid():
    raises(ValueError, lambda: encipher_bifid('abc', 'b', 'abcde'))
    assert encipher_bifid('abc', 'b', 'abcd') == 'bdb'
    raises(ValueError, lambda: decipher_bifid('bdb', 'b', 'abcde'))
    assert encipher_bifid('bdb', 'b', 'abcd') == 'abc'
    raises(ValueError, lambda: bifid_square('abcde'))
    assert bifid5_square("B") == \
        bifid5_square('BACDEFGHIKLMNOPQRSTUVWXYZ')
    assert bifid6_square('B0') == \
        bifid6_square('B0ACDEFGHIJKLMNOPQRSTUVWXYZ123456789')


def test_encipher_decipher_gm():
    ps = [131, 137, 139, 149, 151, 157, 163, 167,
          173, 179, 181, 191, 193, 197, 199]
    qs = [89, 97, 101, 103, 107, 109, 113, 127,
          131, 137, 139, 149, 151, 157, 47]
    messages = [
        0, 32855, 34303, 14805, 1280, 75859, 38368,
        724, 60356, 51675, 76697, 61854, 18661,
    ]
    for p, q in zip(ps, qs):
        pri = gm_private_key(p, q)
        for msg in messages:
            pub = gm_public_key(p, q)
            enc = encipher_gm(msg, pub)
            dec = decipher_gm(enc, pri)
            assert dec == msg


def test_gm_private_key():
    raises(ValueError, lambda: gm_public_key(13, 15))
    raises(ValueError, lambda: gm_public_key(0, 0))
    raises(ValueError, lambda: gm_public_key(0, 5))
    assert 17, 19 == gm_public_key(17, 19)


def test_gm_public_key():
    assert 323 == gm_public_key(17, 19)[1]
    assert 15  == gm_public_key(3, 5)[1]
    raises(ValueError, lambda: gm_public_key(15, 19))

def test_encipher_decipher_bg():
    ps = [67, 7, 71, 103, 11, 43, 107, 47,
          79, 19, 83, 23, 59, 127, 31]
    qs = qs = [7, 71, 103, 11, 43, 107, 47,
               79, 19, 83, 23, 59, 127, 31, 67]
    messages = [
        0, 328, 343, 148, 1280, 758, 383,
        724, 603, 516, 766, 618, 186,
    ]

    for p, q in zip(ps, qs):
        pri = bg_private_key(p, q)
        for msg in messages:
            pub = bg_public_key(p, q)
            enc = encipher_bg(msg, pub)
            dec = decipher_bg(enc, pri)
            assert dec == msg

def test_bg_private_key():
    raises(ValueError, lambda: bg_private_key(8, 16))
    raises(ValueError, lambda: bg_private_key(8, 8))
    raises(ValueError, lambda: bg_private_key(13, 17))
    assert 23, 31 == bg_private_key(23, 31)

def test_bg_public_key():
    assert 5293 == bg_public_key(67, 79)
    assert 713 == bg_public_key(23, 31)
    raises(ValueError, lambda: bg_private_key(13, 17))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_tensor_can.py ===
from sympy.combinatorics.permutations import Permutation, Perm
from sympy.combinatorics.tensor_can import (perm_af_direct_product, dummy_sgs,
    riemann_bsgs, get_symmetric_group_sgs, canonicalize, bsgs_direct_product)
from sympy.combinatorics.testutil import canonicalize_naive, graph_certificate
from sympy.testing.pytest import skip, XFAIL

def test_perm_af_direct_product():
    gens1 = [[1,0,2,3], [0,1,3,2]]
    gens2 = [[1,0]]
    assert perm_af_direct_product(gens1, gens2, 0) == [[1, 0, 2, 3, 4, 5], [0, 1, 3, 2, 4, 5], [0, 1, 2, 3, 5, 4]]
    gens1 = [[1,0,2,3,5,4], [0,1,3,2,4,5]]
    gens2 = [[1,0,2,3]]
    assert [[1, 0, 2, 3, 4, 5, 7, 6], [0, 1, 3, 2, 4, 5, 6, 7], [0, 1, 2, 3, 5, 4, 6, 7]]

def test_dummy_sgs():
    a = dummy_sgs([1,2], 0, 4)
    assert a == [[0,2,1,3,4,5]]
    a = dummy_sgs([2,3,4,5], 0, 8)
    assert a == [x._array_form for x in [Perm(9)(2,3), Perm(9)(4,5),
        Perm(9)(2,4)(3,5)]]

    a = dummy_sgs([2,3,4,5], 1, 8)
    assert a == [x._array_form for x in [Perm(2,3)(8,9), Perm(4,5)(8,9),
        Perm(9)(2,4)(3,5)]]

def test_get_symmetric_group_sgs():
    assert get_symmetric_group_sgs(2) == ([0], [Permutation(3)(0,1)])
    assert get_symmetric_group_sgs(2, 1) == ([0], [Permutation(0,1)(2,3)])
    assert get_symmetric_group_sgs(3) == ([0,1], [Permutation(4)(0,1), Permutation(4)(1,2)])
    assert get_symmetric_group_sgs(3, 1) == ([0,1], [Permutation(0,1)(3,4), Permutation(1,2)(3,4)])
    assert get_symmetric_group_sgs(4) == ([0,1,2], [Permutation(5)(0,1), Permutation(5)(1,2), Permutation(5)(2,3)])
    assert get_symmetric_group_sgs(4, 1) == ([0,1,2], [Permutation(0,1)(4,5), Permutation(1,2)(4,5), Permutation(2,3)(4,5)])


def test_canonicalize_no_slot_sym():
    # cases in which there is no slot symmetry after fixing the
    # free indices; here and in the following if the symmetry of the
    # metric is not specified, it is assumed to be symmetric.
    # If it is not specified, tensors are commuting.

    # A_d0 * B^d0; g = [1,0, 2,3]; T_c = A^d0*B_d0; can = [0,1,2,3]
    base1, gens1 = get_symmetric_group_sgs(1)
    dummies = [0, 1]
    g = Permutation([1,0,2,3])
    can = canonicalize(g, dummies, 0, (base1,gens1,1,0), (base1,gens1,1,0))
    assert can == [0,1,2,3]
    # equivalently
    can = canonicalize(g, dummies, 0, (base1, gens1, 2, None))
    assert can == [0,1,2,3]

    # with antisymmetric metric; T_c = -A^d0*B_d0; can = [0,1,3,2]
    can = canonicalize(g, dummies, 1, (base1,gens1,1,0), (base1,gens1,1,0))
    assert can == [0,1,3,2]

    # A^a * B^b; ord = [a,b]; g = [0,1,2,3]; can = g
    g = Permutation([0,1,2,3])
    dummies = []
    t0 = t1 = (base1, gens1, 1, 0)
    can = canonicalize(g, dummies, 0, t0, t1)
    assert can == [0,1,2,3]
    # B^b * A^a
    g = Permutation([1,0,2,3])
    can = canonicalize(g, dummies, 0, t0, t1)
    assert can == [1,0,2,3]

    # A symmetric
    # A^{b}_{d0}*A^{d0, a} order a,b,d0,-d0; T_c = A^{a d0}*A{b}_{d0}
    # g = [1,3,2,0,4,5]; can = [0,2,1,3,4,5]
    base2, gens2 = get_symmetric_group_sgs(2)
    dummies = [2,3]
    g = Permutation([1,3,2,0,4,5])
    can = canonicalize(g, dummies, 0, (base2, gens2, 2, 0))
    assert can == [0, 2, 1, 3, 4, 5]
    # with antisymmetric metric
    can = canonicalize(g, dummies, 1, (base2, gens2, 2, 0))
    assert can == [0, 2, 1, 3, 4, 5]
    # A^{a}_{d0}*A^{d0, b}
    g = Permutation([0,3,2,1,4,5])
    can = canonicalize(g, dummies, 1, (base2, gens2, 2, 0))
    assert can == [0, 2, 1, 3, 5, 4]

    # A, B symmetric
    # A^b_d0*B^{d0,a}; g=[1,3,2,0,4,5]
    # T_c = A^{b,d0}*B_{a,d0}; can = [1,2,0,3,4,5]
    dummies = [2,3]
    g = Permutation([1,3,2,0,4,5])
    can = canonicalize(g, dummies, 0, (base2,gens2,1,0), (base2,gens2,1,0))
    assert can == [1,2,0,3,4,5]
    # same with antisymmetric metric
    can = canonicalize(g, dummies, 1, (base2,gens2,1,0), (base2,gens2,1,0))
    assert can == [1,2,0,3,5,4]

    # A^{d1}_{d0}*B^d0*C_d1 ord=[d0,-d0,d1,-d1]; g = [2,1,0,3,4,5]
    # T_c = A^{d0 d1}*B_d0*C_d1; can = [0,2,1,3,4,5]
    base1, gens1 = get_symmetric_group_sgs(1)
    base2, gens2 = get_symmetric_group_sgs(2)
    g = Permutation([2,1,0,3,4,5])
    dummies = [0,1,2,3]
    t0 = (base2, gens2, 1, 0)
    t1 = t2 = (base1, gens1, 1, 0)
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    assert can == [0, 2, 1, 3, 4, 5]

    # A without symmetry
    # A^{d1}_{d0}*B^d0*C_d1 ord=[d0,-d0,d1,-d1]; g = [2,1,0,3,4,5]
    # T_c = A^{d0 d1}*B_d1*C_d0; can = [0,2,3,1,4,5]
    g = Permutation([2,1,0,3,4,5])
    dummies = [0,1,2,3]
    t0 = ([], [Permutation(list(range(4)))], 1, 0)
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    assert can == [0,2,3,1,4,5]
    # A, B without symmetry
    # A^{d1}_{d0}*B_{d1}^{d0}; g = [2,1,3,0,4,5]
    # T_c = A^{d0 d1}*B_{d0 d1}; can = [0,2,1,3,4,5]
    t0 = t1 = ([], [Permutation(list(range(4)))], 1, 0)
    dummies = [0,1,2,3]
    g = Permutation([2,1,3,0,4,5])
    can = canonicalize(g, dummies, 0, t0, t1)
    assert can == [0, 2, 1, 3, 4, 5]
    # A_{d0}^{d1}*B_{d1}^{d0}; g = [1,2,3,0,4,5]
    # T_c = A^{d0 d1}*B_{d1 d0}; can = [0,2,3,1,4,5]
    g = Permutation([1,2,3,0,4,5])
    can = canonicalize(g, dummies, 0, t0, t1)
    assert can == [0,2,3,1,4,5]

    # A, B, C without symmetry
    # A^{d1 d0}*B_{a d0}*C_{d1 b} ord=[a,b,d0,-d0,d1,-d1]
    # g=[4,2,0,3,5,1,6,7]
    # T_c=A^{d0 d1}*B_{a d1}*C_{d0 b}; can = [2,4,0,5,3,1,6,7]
    t0 = t1 = t2 = ([], [Permutation(list(range(4)))], 1, 0)
    dummies = [2,3,4,5]
    g = Permutation([4,2,0,3,5,1,6,7])
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    assert can == [2,4,0,5,3,1,6,7]

    # A symmetric, B and C without symmetry
    # A^{d1 d0}*B_{a d0}*C_{d1 b} ord=[a,b,d0,-d0,d1,-d1]
    # g=[4,2,0,3,5,1,6,7]
    # T_c = A^{d0 d1}*B_{a d0}*C_{d1 b}; can = [2,4,0,3,5,1,6,7]
    t0 = (base2,gens2,1,0)
    t1 = t2 = ([], [Permutation(list(range(4)))], 1, 0)
    dummies = [2,3,4,5]
    g = Permutation([4,2,0,3,5,1,6,7])
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    assert can == [2,4,0,3,5,1,6,7]

    # A and C symmetric, B without symmetry
    # A^{d1 d0}*B_{a d0}*C_{d1 b} ord=[a,b,d0,-d0,d1,-d1]
    # g=[4,2,0,3,5,1,6,7]
    # T_c = A^{d0 d1}*B_{a d0}*C_{b d1}; can = [2,4,0,3,1,5,6,7]
    t0 = t2 = (base2,gens2,1,0)
    t1 = ([], [Permutation(list(range(4)))], 1, 0)
    dummies = [2,3,4,5]
    g = Permutation([4,2,0,3,5,1,6,7])
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    assert can == [2,4,0,3,1,5,6,7]

    # A symmetric, B without symmetry, C antisymmetric
    # A^{d1 d0}*B_{a d0}*C_{d1 b} ord=[a,b,d0,-d0,d1,-d1]
    # g=[4,2,0,3,5,1,6,7]
    # T_c = -A^{d0 d1}*B_{a d0}*C_{b d1}; can = [2,4,0,3,1,5,7,6]
    t0 = (base2,gens2, 1, 0)
    t1 = ([], [Permutation(list(range(4)))], 1, 0)
    base2a, gens2a = get_symmetric_group_sgs(2, 1)
    t2 = (base2a, gens2a, 1, 0)
    dummies = [2,3,4,5]
    g = Permutation([4,2,0,3,5,1,6,7])
    can = canonicalize(g, dummies, 0, t0, t1, t2)
    assert can == [2,4,0,3,1,5,7,6]


def test_canonicalize_no_dummies():
    base1, gens1 = get_symmetric_group_sgs(1)
    base2, gens2 = get_symmetric_group_sgs(2)
    base2a, gens2a = get_symmetric_group_sgs(2, 1)

    # A commuting
    # A^c A^b A^a; ord = [a,b,c]; g = [2,1,0,3,4]
    # T_c = A^a A^b A^c; can = list(range(5))
    g = Permutation([2,1,0,3,4])
    can = canonicalize(g, [], 0, (base1, gens1, 3, 0))
    assert can == list(range(5))

    # A anticommuting
    # A^c A^b A^a; ord = [a,b,c]; g = [2,1,0,3,4]
    # T_c = -A^a A^b A^c; can = [0,1,2,4,3]
    g = Permutation([2,1,0,3,4])
    can = canonicalize(g, [], 0, (base1, gens1, 3, 1))
    assert can == [0,1,2,4,3]

    # A commuting and symmetric
    # A^{b,d}*A^{c,a}; ord = [a,b,c,d]; g = [1,3,2,0,4,5]
    # T_c = A^{a c}*A^{b d}; can = [0,2,1,3,4,5]
    g = Permutation([1,3,2,0,4,5])
    can = canonicalize(g, [], 0, (base2, gens2, 2, 0))
    assert can == [0,2,1,3,4,5]

    # A anticommuting and symmetric
    # A^{b,d}*A^{c,a}; ord = [a,b,c,d]; g = [1,3,2,0,4,5]
    # T_c = -A^{a c}*A^{b d}; can = [0,2,1,3,5,4]
    g = Permutation([1,3,2,0,4,5])
    can = canonicalize(g, [], 0, (base2, gens2, 2, 1))
    assert can == [0,2,1,3,5,4]
    # A^{c,a}*A^{b,d} ; g = [2,0,1,3,4,5]
    # T_c = A^{a c}*A^{b d}; can = [0,2,1,3,4,5]
    g = Permutation([2,0,1,3,4,5])
    can = canonicalize(g, [], 0, (base2, gens2, 2, 1))
    assert can == [0,2,1,3,4,5]

def test_no_metric_symmetry():
    # no metric symmetry
    # A^d1_d0 * A^d0_d1; ord = [d0,-d0,d1,-d1]; g= [2,1,0,3,4,5]
    # T_c = A^d0_d1 * A^d1_d0; can = [0,3,2,1,4,5]
    g = Permutation([2,1,0,3,4,5])
    can = canonicalize(g, list(range(4)), None, [[], [Permutation(list(range(4)))], 2, 0])
    assert can == [0,3,2,1,4,5]

    # A^d1_d2 * A^d0_d3 * A^d2_d1 * A^d3_d0
    # ord = [d0,-d0,d1,-d1,d2,-d2,d3,-d3]
    #        0    1  2  3  4   5   6   7
    # g = [2,5,0,7,4,3,6,1,8,9]
    # T_c = A^d0_d1 * A^d1_d0 * A^d2_d3 * A^d3_d2
    # can = [0,3,2,1,4,7,6,5,8,9]
    g = Permutation([2,5,0,7,4,3,6,1,8,9])
    #can = canonicalize(g, list(range(8)), 0, [[], [list(range(4))], 4, 0])
    #assert can == [0, 2, 3, 1, 4, 6, 7, 5, 8, 9]
    can = canonicalize(g, list(range(8)), None, [[], [Permutation(list(range(4)))], 4, 0])
    assert can == [0, 3, 2, 1, 4, 7, 6, 5, 8, 9]

    # A^d0_d2 * A^d1_d3 * A^d3_d0 * A^d2_d1
    # g = [0,5,2,7,6,1,4,3,8,9]
    # T_c = A^d0_d1 * A^d1_d2 * A^d2_d3 * A^d3_d0
    # can = [0,3,2,5,4,7,6,1,8,9]
    g = Permutation([0,5,2,7,6,1,4,3,8,9])
    can = canonicalize(g, list(range(8)), None, [[], [Permutation(list(range(4)))], 4, 0])
    assert can == [0,3,2,5,4,7,6,1,8,9]

    g = Permutation([12,7,10,3,14,13,4,11,6,1,2,9,0,15,8,5,16,17])
    can = canonicalize(g, list(range(16)), None, [[], [Permutation(list(range(4)))], 8, 0])
    assert can == [0,3,2,5,4,7,6,1,8,11,10,13,12,15,14,9,16,17]

def test_canonical_free():
    # t = A^{d0 a1}*A_d0^a0
    # ord = [a0,a1,d0,-d0];  g = [2,1,3,0,4,5]; dummies = [[2,3]]
    # t_c = A_d0^a0*A^{d0 a1}
    # can = [3,0, 2,1, 4,5]
    g = Permutation([2,1,3,0,4,5])
    dummies = [[2,3]]
    can = canonicalize(g, dummies, [None], ([], [Permutation(3)], 2, 0))
    assert can == [3,0, 2,1, 4,5]

def test_canonicalize1():
    base1, gens1 = get_symmetric_group_sgs(1)
    base1a, gens1a = get_symmetric_group_sgs(1, 1)
    base2, gens2 = get_symmetric_group_sgs(2)
    base3, gens3 = get_symmetric_group_sgs(3)
    base2a, gens2a = get_symmetric_group_sgs(2, 1)
    base3a, gens3a = get_symmetric_group_sgs(3, 1)

    # A_d0*A^d0; ord = [d0,-d0]; g = [1,0,2,3]
    # T_c = A^d0*A_d0; can = [0,1,2,3]
    g = Permutation([1,0,2,3])
    can = canonicalize(g, [0, 1], 0, (base1, gens1, 2, 0))
    assert can == list(range(4))

    # A commuting
    # A_d0*A_d1*A_d2*A^d2*A^d1*A^d0; ord=[d0,-d0,d1,-d1,d2,-d2]
    # g = [1,3,5,4,2,0,6,7]
    # T_c = A^d0*A_d0*A^d1*A_d1*A^d2*A_d2; can = list(range(8))
    g = Permutation([1,3,5,4,2,0,6,7])
    can = canonicalize(g, list(range(6)), 0, (base1, gens1, 6, 0))
    assert can == list(range(8))

    # A anticommuting
    # A_d0*A_d1*A_d2*A^d2*A^d1*A^d0; ord=[d0,-d0,d1,-d1,d2,-d2]
    # g = [1,3,5,4,2,0,6,7]
    # T_c 0;  can = 0
    g = Permutation([1,3,5,4,2,0,6,7])
    can = canonicalize(g, list(range(6)), 0, (base1, gens1, 6, 1))
    assert can == 0
    can1 = canonicalize_naive(g, list(range(6)), 0, (base1, gens1, 6, 1))
    assert can1 == 0

    # A commuting symmetric
    # A^{d0 b}*A^a_d1*A^d1_d0; ord=[a,b,d0,-d0,d1,-d1]
    # g = [2,1,0,5,4,3,6,7]
    # T_c = A^{a d0}*A^{b d1}*A_{d0 d1}; can = [0,2,1,4,3,5,6,7]
    g = Permutation([2,1,0,5,4,3,6,7])
    can = canonicalize(g, list(range(2,6)), 0, (base2, gens2, 3, 0))
    assert can == [0,2,1,4,3,5,6,7]

    # A, B commuting symmetric
    # A^{d0 b}*A^d1_d0*B^a_d1; ord=[a,b,d0,-d0,d1,-d1]
    # g = [2,1,4,3,0,5,6,7]
    # T_c = A^{b d0}*A_d0^d1*B^a_d1; can = [1,2,3,4,0,5,6,7]
    g = Permutation([2,1,4,3,0,5,6,7])
    can = canonicalize(g, list(range(2,6)), 0, (base2,gens2,2,0), (base2,gens2,1,0))
    assert can == [1,2,3,4,0,5,6,7]

    # A commuting symmetric
    # A^{d1 d0 b}*A^{a}_{d1 d0}; ord=[a,b, d0,-d0,d1,-d1]
    # g = [4,2,1,0,5,3,6,7]
    # T_c = A^{a d0 d1}*A^{b}_{d0 d1}; can = [0,2,4,1,3,5,6,7]
    g = Permutation([4,2,1,0,5,3,6,7])
    can = canonicalize(g, list(range(2,6)), 0, (base3, gens3, 2, 0))
    assert can == [0,2,4,1,3,5,6,7]


    # A^{d3 d0 d2}*A^a0_{d1 d2}*A^d1_d3^a1*A^{a2 a3}_d0
    # ord = [a0,a1,a2,a3,d0,-d0,d1,-d1,d2,-d2,d3,-d3]
    #        0   1  2  3  4  5  6   7  8   9  10  11
    # g = [10,4,8, 0,7,9, 6,11,1, 2,3,5, 12,13]
    # T_c = A^{a0 d0 d1}*A^a1_d0^d2*A^{a2 a3 d3}*A_{d1 d2 d3}
    # can = [0,4,6, 1,5,8, 2,3,10, 7,9,11, 12,13]
    g = Permutation([10,4,8, 0,7,9, 6,11,1, 2,3,5, 12,13])
    can = canonicalize(g, list(range(4,12)), 0, (base3, gens3, 4, 0))
    assert can == [0,4,6, 1,5,8, 2,3,10, 7,9,11, 12,13]

    # A commuting symmetric, B antisymmetric
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # ord = [d0,-d0,d1,-d1,d2,-d2,d3,-d3]
    # g = [0,2,4,5,7,3,1,6,8,9]
    # in this esxample and in the next three,
    # renaming dummy indices and using symmetry of A,
    # T = A^{d0 d1 d2} * A_{d0 d1 d3} * B_d2^d3
    # can = 0
    g = Permutation([0,2,4,5,7,3,1,6,8,9])
    can = canonicalize(g, list(range(8)), 0, (base3, gens3,2,0), (base2a,gens2a,1,0))
    assert can == 0
    # A anticommuting symmetric, B anticommuting
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # T_c = A^{d0 d1 d2} * A_{d0 d1}^d3 * B_{d2 d3}
    # can = [0,2,4, 1,3,6, 5,7, 8,9]
    can = canonicalize(g, list(range(8)), 0, (base3, gens3,2,1), (base2a,gens2a,1,0))
    assert can == [0,2,4, 1,3,6, 5,7, 8,9]
    # A anticommuting symmetric, B antisymmetric commuting, antisymmetric metric
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # T_c = -A^{d0 d1 d2} * A_{d0 d1}^d3 * B_{d2 d3}
    # can = [0,2,4, 1,3,6, 5,7, 9,8]
    can = canonicalize(g, list(range(8)), 1, (base3, gens3,2,1), (base2a,gens2a,1,0))
    assert can == [0,2,4, 1,3,6, 5,7, 9,8]

    # A anticommuting symmetric, B anticommuting anticommuting,
    # no metric symmetry
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # T_c = A^{d0 d1 d2} * A_{d0 d1 d3} * B_d2^d3
    # can = [0,2,4, 1,3,7, 5,6, 8,9]
    can = canonicalize(g, list(range(8)), None, (base3, gens3,2,1), (base2a,gens2a,1,0))
    assert can == [0,2,4,1,3,7,5,6,8,9]

    # Gamma anticommuting
    # Gamma_{mu nu} * gamma^rho * Gamma^{nu mu alpha}
    # ord = [alpha, rho, mu,-mu,nu,-nu]
    # g = [3,5,1,4,2,0,6,7]
    # T_c = -Gamma^{mu nu} * gamma^rho * Gamma_{alpha mu nu}
    # can = [2,4,1,0,3,5,7,6]]
    g = Permutation([3,5,1,4,2,0,6,7])
    t0 = (base2a, gens2a, 1, None)
    t1 = (base1, gens1, 1, None)
    t2 = (base3a, gens3a, 1, None)
    can = canonicalize(g, list(range(2, 6)), 0, t0, t1, t2)
    assert can == [2,4,1,0,3,5,7,6]

    # Gamma_{mu nu} * Gamma^{gamma beta} * gamma_rho * Gamma^{nu mu alpha}
    # ord = [alpha, beta, gamma, -rho, mu,-mu,nu,-nu]
    #         0      1      2     3    4   5   6  7
    # g = [5,7,2,1,3,6,4,0,8,9]
    # T_c = Gamma^{mu nu} * Gamma^{beta gamma} * gamma_rho * Gamma^alpha_{mu nu}    # can = [4,6,1,2,3,0,5,7,8,9]
    t0 = (base2a, gens2a, 2, None)
    g = Permutation([5,7,2,1,3,6,4,0,8,9])
    can = canonicalize(g, list(range(4, 8)), 0, t0, t1, t2)
    assert can == [4,6,1,2,3,0,5,7,8,9]

    # f^a_{b,c} antisymmetric in b,c; A_mu^a no symmetry
    # f^c_{d a} * f_{c e b} * A_mu^d * A_nu^a * A^{nu e} * A^{mu b}
    # ord = [mu,-mu,nu,-nu,a,-a,b,-b,c,-c,d,-d, e, -e]
    #         0  1  2   3  4  5 6  7 8  9 10 11 12 13
    # g = [8,11,5, 9,13,7, 1,10, 3,4, 2,12, 0,6, 14,15]
    # T_c = -f^{a b c} * f_a^{d e} * A^mu_b * A_{mu d} * A^nu_c * A_{nu e}
    # can = [4,6,8,       5,10,12,   0,7,     1,11,       2,9,    3,13, 15,14]
    g = Permutation([8,11,5, 9,13,7, 1,10, 3,4, 2,12, 0,6, 14,15])
    base_f, gens_f = bsgs_direct_product(base1, gens1, base2a, gens2a)
    base_A, gens_A = bsgs_direct_product(base1, gens1, base1, gens1)
    t0 = (base_f, gens_f, 2, 0)
    t1 = (base_A, gens_A, 4, 0)
    can = canonicalize(g, [list(range(4)), list(range(4, 14))], [0, 0], t0, t1)
    assert can == [4,6,8, 5,10,12, 0,7, 1,11, 2,9, 3,13, 15,14]


def test_riemann_invariants():
    baser, gensr = riemann_bsgs
    # R^{d0 d1}_{d1 d0}; ord = [d0,-d0,d1,-d1]; g = [0,2,3,1,4,5]
    # T_c = -R^{d0 d1}_{d0 d1}; can = [0,2,1,3,5,4]
    g = Permutation([0,2,3,1,4,5])
    can = canonicalize(g, list(range(2, 4)), 0, (baser, gensr, 1, 0))
    assert can == [0,2,1,3,5,4]
    # use a non minimal BSGS
    can = canonicalize(g, list(range(2, 4)), 0, ([2, 0], [Permutation([1,0,2,3,5,4]), Permutation([2,3,0,1,4,5])], 1, 0))
    assert can == [0,2,1,3,5,4]

    """
    The following tests in test_riemann_invariants and in
    test_riemann_invariants1 have been checked using xperm.c from XPerm in
    in [1] and with an older version contained in [2]

    [1] xperm.c part of xPerm written by J. M. Martin-Garcia
        http://www.xact.es/index.html
    [2] test_xperm.cc in cadabra by Kasper Peeters, http://cadabra.phi-sci.com/
    """
    # R_d11^d1_d0^d5 * R^{d6 d4 d0}_d5 * R_{d7 d2 d8 d9} *
    # R_{d10 d3 d6 d4} * R^{d2 d7 d11}_d1 * R^{d8 d9 d3 d10}
    # ord: contravariant d_k ->2*k, covariant d_k -> 2*k+1
    # T_c = R^{d0 d1 d2 d3} * R_{d0 d1}^{d4 d5} * R_{d2 d3}^{d6 d7} *
    # R_{d4 d5}^{d8 d9} * R_{d6 d7}^{d10 d11} * R_{d8 d9 d10 d11}
    g = Permutation([23,2,1,10,12,8,0,11,15,5,17,19,21,7,13,9,4,14,22,3,16,18,6,20,24,25])
    can = canonicalize(g, list(range(24)), 0, (baser, gensr, 6, 0))
    assert can == [0,2,4,6,1,3,8,10,5,7,12,14,9,11,16,18,13,15,20,22,17,19,21,23,24,25]

    # use a non minimal BSGS
    can = canonicalize(g, list(range(24)), 0, ([2, 0], [Permutation([1,0,2,3,5,4]), Permutation([2,3,0,1,4,5])], 6, 0))
    assert can == [0,2,4,6,1,3,8,10,5,7,12,14,9,11,16,18,13,15,20,22,17,19,21,23,24,25]

    g = Permutation([0,2,5,7,4,6,9,11,8,10,13,15,12,14,17,19,16,18,21,23,20,22,25,27,24,26,29,31,28,30,33,35,32,34,37,39,36,38,1,3,40,41])
    can = canonicalize(g, list(range(40)), 0, (baser, gensr, 10, 0))
    assert can == [0,2,4,6,1,3,8,10,5,7,12,14,9,11,16,18,13,15,20,22,17,19,24,26,21,23,28,30,25,27,32,34,29,31,36,38,33,35,37,39,40,41]


@XFAIL
def test_riemann_invariants1():
    skip('takes too much time')
    baser, gensr = riemann_bsgs
    g = Permutation([17, 44, 11, 3, 0, 19, 23, 15, 38, 4, 25, 27, 43, 36, 22, 14, 8, 30, 41, 20, 2, 10, 12, 28, 18, 1, 29, 13, 37, 42, 33, 7, 9, 31, 24, 26, 39, 5, 34, 47, 32, 6, 21, 40, 35, 46, 45, 16, 48, 49])
    can = canonicalize(g, list(range(48)), 0, (baser, gensr, 12, 0))
    assert can == [0, 2, 4, 6, 1, 3, 8, 10, 5, 7, 12, 14, 9, 11, 16, 18, 13, 15, 20, 22, 17, 19, 24, 26, 21, 23, 28, 30, 25, 27, 32, 34, 29, 31, 36, 38, 33, 35, 40, 42, 37, 39, 44, 46, 41, 43, 45, 47, 48, 49]

    g = Permutation([0,2,4,6, 7,8,10,12, 14,16,18,20, 19,22,24,26, 5,21,28,30, 32,34,36,38, 40,42,44,46, 13,48,50,52, 15,49,54,56, 17,33,41,58, 9,23,60,62, 29,35,63,64, 3,45,66,68, 25,37,47,57, 11,31,69,70, 27,39,53,72, 1,59,73,74, 55,61,67,76, 43,65,75,78, 51,71,77,79, 80,81])
    can = canonicalize(g, list(range(80)), 0, (baser, gensr, 20, 0))
    assert can == [0,2,4,6, 1,8,10,12, 3,14,16,18, 5,20,22,24, 7,26,28,30, 9,15,32,34, 11,36,23,38, 13,40,42,44, 17,39,29,46, 19,48,43,50, 21,45,52,54, 25,56,33,58, 27,60,53,62, 31,51,64,66, 35,65,47,68, 37,70,49,72, 41,74,57,76, 55,67,59,78, 61,69,71,75, 63,79,73,77, 80,81]


def test_riemann_products():
    baser, gensr = riemann_bsgs
    base1, gens1 = get_symmetric_group_sgs(1)
    base2, gens2 = get_symmetric_group_sgs(2)
    base2a, gens2a = get_symmetric_group_sgs(2, 1)

    # R^{a b d0}_d0 = 0
    g = Permutation([0,1,2,3,4,5])
    can = canonicalize(g, list(range(2,4)), 0, (baser, gensr, 1, 0))
    assert can == 0

    # R^{d0 b a}_d0 ; ord = [a,b,d0,-d0}; g = [2,1,0,3,4,5]
    # T_c = -R^{a d0 b}_d0;  can = [0,2,1,3,5,4]
    g = Permutation([2,1,0,3,4,5])
    can = canonicalize(g, list(range(2, 4)), 0, (baser, gensr, 1, 0))
    assert can == [0,2,1,3,5,4]

    # R^d1_d2^b_d0 * R^{d0 a}_d1^d2; ord=[a,b,d0,-d0,d1,-d1,d2,-d2]
    # g = [4,7,1,3,2,0,5,6,8,9]
    # T_c = -R^{a d0 d1 d2}* R^b_{d0 d1 d2}
    # can = [0,2,4,6,1,3,5,7,9,8]
    g = Permutation([4,7,1,3,2,0,5,6,8,9])
    can = canonicalize(g, list(range(2,8)), 0, (baser, gensr, 2, 0))
    assert can == [0,2,4,6,1,3,5,7,9,8]
    can1 = canonicalize_naive(g, list(range(2,8)), 0, (baser, gensr, 2, 0))
    assert can == can1

    # A symmetric commuting
    # R^{d6 d5}_d2^d1 * R^{d4 d0 d2 d3} * A_{d6 d0} A_{d3 d1} * A_{d4 d5}
    # g = [12,10,5,2, 8,0,4,6, 13,1, 7,3, 9,11,14,15]
    # T_c = -R^{d0 d1 d2 d3} * R_d0^{d4 d5 d6} * A_{d1 d4}*A_{d2 d5}*A_{d3 d6}

    g = Permutation([12,10,5,2,8,0,4,6,13,1,7,3,9,11,14,15])
    can = canonicalize(g, list(range(14)), 0, ((baser,gensr,2,0)), (base2,gens2,3,0))
    assert can == [0, 2, 4, 6, 1, 8, 10, 12, 3, 9, 5, 11, 7, 13, 15, 14]

    # R^{d2 a0 a2 d0} * R^d1_d2^{a1 a3} * R^{a4 a5}_{d0 d1}
    # ord = [a0,a1,a2,a3,a4,a5,d0,-d0,d1,-d1,d2,-d2]
    #         0  1  2  3 4  5  6   7  8   9  10  11
    # can = [0, 6, 2, 8, 1, 3, 7, 10, 4, 5, 9, 11, 12, 13]
    # T_c = R^{a0 d0 a2 d1}*R^{a1 a3}_d0^d2*R^{a4 a5}_{d1 d2}
    g = Permutation([10,0,2,6,8,11,1,3,4,5,7,9,12,13])
    can = canonicalize(g, list(range(6,12)), 0, (baser, gensr, 3, 0))
    assert can == [0, 6, 2, 8, 1, 3, 7, 10, 4, 5, 9, 11, 12, 13]
    #can1 = canonicalize_naive(g, list(range(6,12)), 0, (baser, gensr, 3, 0))
    #assert can == can1

    # A^n_{i, j} antisymmetric in i,j
    # A_m0^d0_a1 * A_m1^a0_d0; ord = [m0,m1,a0,a1,d0,-d0]
    # g = [0,4,3,1,2,5,6,7]
    # T_c = -A_{m a1}^d0 * A_m1^a0_d0
    # can = [0,3,4,1,2,5,7,6]
    base, gens = bsgs_direct_product(base1, gens1, base2a, gens2a)
    dummies = list(range(4, 6))
    g = Permutation([0,4,3,1,2,5,6,7])
    can = canonicalize(g, dummies, 0, (base, gens, 2, 0))
    assert can == [0, 3, 4, 1, 2, 5, 7, 6]


    # A^n_{i, j} symmetric in i,j
    # A^m0_a0^d2 * A^n0_d2^d1 * A^n1_d1^d0 * A_{m0 d0}^a1
    # ordering: first the free indices; then first n, then d
    # ord=[n0,n1,a0,a1, m0,-m0,d0,-d0,d1,-d1,d2,-d2]
    #      0  1   2  3   4  5  6   7  8   9  10  11]
    # g = [4,2,10, 0,11,8, 1,9,6, 5,7,3, 12,13]
    # if the dummy indices m_i and d_i were separated,
    # one gets
    # T_c = A^{n0 d0 d1} * A^n1_d0^d2 * A^m0^a0_d1 * A_m0^a1_d2
    # can = [0, 6, 8, 1, 7, 10, 4, 2, 9, 5, 3, 11, 12, 13]
    # If they are not, so can is
    # T_c = A^{n0 m0 d0} A^n1_m0^d1 A^{d2 a0}_d0 A_d2^a1_d1
    # can = [0, 4, 6, 1, 5, 8, 10, 2, 7, 11, 3, 9, 12, 13]
    # case with single type of indices

    base, gens = bsgs_direct_product(base1, gens1, base2, gens2)
    dummies = list(range(4, 12))
    g = Permutation([4,2,10, 0,11,8, 1,9,6, 5,7,3, 12,13])
    can = canonicalize(g, dummies, 0, (base, gens, 4, 0))
    assert can == [0, 4, 6, 1, 5, 8, 10, 2, 7, 11, 3, 9, 12, 13]
    # case with separated indices
    dummies = [list(range(4, 6)), list(range(6,12))]
    sym = [0, 0]
    can = canonicalize(g, dummies, sym, (base, gens, 4, 0))
    assert can == [0, 6, 8, 1, 7, 10, 4, 2, 9, 5, 3, 11, 12, 13]
    # case with separated indices with the second type of index
    # with antisymmetric metric: there is a sign change
    sym = [0, 1]
    can = canonicalize(g, dummies, sym, (base, gens, 4, 0))
    assert can == [0, 6, 8, 1, 7, 10, 4, 2, 9, 5, 3, 11, 13, 12]

def test_graph_certificate():
    # test tensor invariants constructed from random regular graphs;
    # checked graph isomorphism with networkx
    import random
    def randomize_graph(size, g):
        p = list(range(size))
        random.shuffle(p)
        g1a = {}
        for k, v in g1.items():
            g1a[p[k]] = [p[i] for i in v]
        return g1a

    g1 = {0: [2, 3, 7], 1: [4, 5, 7], 2: [0, 4, 6], 3: [0, 6, 7], 4: [1, 2, 5], 5: [1, 4, 6], 6: [2, 3, 5], 7: [0, 1, 3]}
    g2 = {0: [2, 3, 7], 1: [2, 4, 5], 2: [0, 1, 5], 3: [0, 6, 7], 4: [1, 5, 6], 5: [1, 2, 4], 6: [3, 4, 7], 7: [0, 3, 6]}

    c1 = graph_certificate(g1)
    c2 = graph_certificate(g2)
    assert c1 != c2
    g1a = randomize_graph(8, g1)
    c1a = graph_certificate(g1a)
    assert c1 == c1a

    g1 = {0: [8, 1, 9, 7], 1: [0, 9, 3, 4], 2: [3, 4, 6, 7], 3: [1, 2, 5, 6], 4: [8, 1, 2, 5], 5: [9, 3, 4, 7], 6: [8, 2, 3, 7], 7: [0, 2, 5, 6], 8: [0, 9, 4, 6], 9: [8, 0, 5, 1]}
    g2 = {0: [1, 2, 5, 6], 1: [0, 9, 5, 7], 2: [0, 4, 6, 7], 3: [8, 9, 6, 7], 4: [8, 2, 6, 7], 5: [0, 9, 8, 1], 6: [0, 2, 3, 4], 7: [1, 2, 3, 4], 8: [9, 3, 4, 5], 9: [8, 1, 3, 5]}
    c1 = graph_certificate(g1)
    c2 = graph_certificate(g2)
    assert c1 != c2
    g1a = randomize_graph(10, g1)
    c1a = graph_certificate(g1a)
    assert c1 == c1a

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\tests\test_hermite_e.py ===
"""Tests for hermite_e module.

"""
from functools import reduce

import numpy as np
import numpy.polynomial.hermite_e as herme
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

He0 = np.array([1])
He1 = np.array([0, 1])
He2 = np.array([-1, 0, 1])
He3 = np.array([0, -3, 0, 1])
He4 = np.array([3, 0, -6, 0, 1])
He5 = np.array([0, 15, 0, -10, 0, 1])
He6 = np.array([-15, 0, 45, 0, -15, 0, 1])
He7 = np.array([0, -105, 0, 105, 0, -21, 0, 1])
He8 = np.array([105, 0, -420, 0, 210, 0, -28, 0, 1])
He9 = np.array([0, 945, 0, -1260, 0, 378, 0, -36, 0, 1])

Helist = [He0, He1, He2, He3, He4, He5, He6, He7, He8, He9]


def trim(x):
    return herme.hermetrim(x, tol=1e-6)


class TestConstants:

    def test_hermedomain(self):
        assert_equal(herme.hermedomain, [-1, 1])

    def test_hermezero(self):
        assert_equal(herme.hermezero, [0])

    def test_hermeone(self):
        assert_equal(herme.hermeone, [1])

    def test_hermex(self):
        assert_equal(herme.hermex, [0, 1])


class TestArithmetic:
    x = np.linspace(-3, 3, 100)

    def test_hermeadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = herme.hermeadd([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermesub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = herme.hermesub([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermemulx(self):
        assert_equal(herme.hermemulx([0]), [0])
        assert_equal(herme.hermemulx([1]), [0, 1])
        for i in range(1, 5):
            ser = [0] * i + [1]
            tgt = [0] * (i - 1) + [i, 0, 1]
            assert_equal(herme.hermemulx(ser), tgt)

    def test_hermemul(self):
        # check values of result
        for i in range(5):
            pol1 = [0] * i + [1]
            val1 = herme.hermeval(self.x, pol1)
            for j in range(5):
                msg = f"At i={i}, j={j}"
                pol2 = [0] * j + [1]
                val2 = herme.hermeval(self.x, pol2)
                pol3 = herme.hermemul(pol1, pol2)
                val3 = herme.hermeval(self.x, pol3)
                assert_(len(pol3) == i + j + 1, msg)
                assert_almost_equal(val3, val1 * val2, err_msg=msg)

    def test_hermediv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0] * i + [1]
                cj = [0] * j + [1]
                tgt = herme.hermeadd(ci, cj)
                quo, rem = herme.hermediv(tgt, ci)
                res = herme.hermeadd(herme.hermemul(quo, ci), rem)
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermepow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = np.arange(i + 1)
                tgt = reduce(herme.hermemul, [c] * j, np.array([1]))
                res = herme.hermepow(c, j)
                assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = np.array([4., 2., 3.])
    c2d = np.einsum('i,j->ij', c1d, c1d)
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = np.random.random((3, 5)) * 2 - 1
    y = polyval(x, [1., 2., 3.])

    def test_hermeval(self):
        # check empty input
        assert_equal(herme.hermeval([], [1]).size, 0)

        # check normal input)
        x = np.linspace(-1, 1)
        y = [polyval(x, c) for c in Helist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = herme.hermeval(x, [0] * i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)

        # check that shape is preserved
        for i in range(3):
            dims = [2] * i
            x = np.zeros(dims)
            assert_equal(herme.hermeval(x, [1]).shape, dims)
            assert_equal(herme.hermeval(x, [1, 0]).shape, dims)
            assert_equal(herme.hermeval(x, [1, 0, 0]).shape, dims)

    def test_hermeval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        assert_raises(ValueError, herme.hermeval2d, x1, x2[:2], self.c2d)

        # test values
        tgt = y1 * y2
        res = herme.hermeval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = herme.hermeval2d(z, z, self.c2d)
        assert_(res.shape == (2, 3))

    def test_hermeval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        assert_raises(ValueError, herme.hermeval3d, x1, x2, x3[:2], self.c3d)

        # test values
        tgt = y1 * y2 * y3
        res = herme.hermeval3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = herme.hermeval3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3))

    def test_hermegrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = np.einsum('i,j->ij', y1, y2)
        res = herme.hermegrid2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = herme.hermegrid2d(z, z, self.c2d)
        assert_(res.shape == (2, 3) * 2)

    def test_hermegrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        res = herme.hermegrid3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = herme.hermegrid3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3) * 3)


class TestIntegral:

    def test_hermeint(self):
        # check exceptions
        assert_raises(TypeError, herme.hermeint, [0], .5)
        assert_raises(ValueError, herme.hermeint, [0], -1)
        assert_raises(ValueError, herme.hermeint, [0], 1, [0, 0])
        assert_raises(ValueError, herme.hermeint, [0], lbnd=[0])
        assert_raises(ValueError, herme.hermeint, [0], scl=[0])
        assert_raises(TypeError, herme.hermeint, [0], axis=.5)

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = herme.hermeint([0], m=i, k=k)
            assert_almost_equal(res, [0, 1])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            hermepol = herme.poly2herme(pol)
            hermeint = herme.hermeint(hermepol, m=1, k=[i])
            res = herme.herme2poly(hermeint)
            assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            hermepol = herme.poly2herme(pol)
            hermeint = herme.hermeint(hermepol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(herme.hermeval(-1, hermeint), i)

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            hermepol = herme.poly2herme(pol)
            hermeint = herme.hermeint(hermepol, m=1, k=[i], scl=2)
            res = herme.herme2poly(hermeint)
            assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = herme.hermeint(tgt, m=1)
                res = herme.hermeint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = herme.hermeint(tgt, m=1, k=[k])
                res = herme.hermeint(pol, m=j, k=list(range(j)))
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = herme.hermeint(tgt, m=1, k=[k], lbnd=-1)
                res = herme.hermeint(pol, m=j, k=list(range(j)), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = herme.hermeint(tgt, m=1, k=[k], scl=2)
                res = herme.hermeint(pol, m=j, k=list(range(j)), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_hermeint_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([herme.hermeint(c) for c in c2d.T]).T
        res = herme.hermeint(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([herme.hermeint(c) for c in c2d])
        res = herme.hermeint(c2d, axis=1)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([herme.hermeint(c, k=3) for c in c2d])
        res = herme.hermeint(c2d, k=3, axis=1)
        assert_almost_equal(res, tgt)


class TestDerivative:

    def test_hermeder(self):
        # check exceptions
        assert_raises(TypeError, herme.hermeder, [0], .5)
        assert_raises(ValueError, herme.hermeder, [0], -1)

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0] * i + [1]
            res = herme.hermeder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = herme.hermeder(herme.hermeint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = herme.hermeder(
                    herme.hermeint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_hermeder_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([herme.hermeder(c) for c in c2d.T]).T
        res = herme.hermeder(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([herme.hermeder(c) for c in c2d])
        res = herme.hermeder(c2d, axis=1)
        assert_almost_equal(res, tgt)


class TestVander:
    # some random values in [-1, 1)
    x = np.random.random((3, 5)) * 2 - 1

    def test_hermevander(self):
        # check for 1d x
        x = np.arange(3)
        v = herme.hermevander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            assert_almost_equal(v[..., i], herme.hermeval(x, coef))

        # check for 2d x
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = herme.hermevander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            assert_almost_equal(v[..., i], herme.hermeval(x, coef))

    def test_hermevander2d(self):
        # also tests hermeval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3))
        van = herme.hermevander2d(x1, x2, [1, 2])
        tgt = herme.hermeval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = herme.hermevander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))

    def test_hermevander3d(self):
        # also tests hermeval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3, 4))
        van = herme.hermevander3d(x1, x2, x3, [1, 2, 3])
        tgt = herme.hermeval3d(x1, x2, x3, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = herme.hermevander3d([x1], [x2], [x3], [1, 2, 3])
        assert_(van.shape == (1, 5, 24))


class TestFitting:

    def test_hermefit(self):
        def f(x):
            return x * (x - 1) * (x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        # Test exceptions
        assert_raises(ValueError, herme.hermefit, [1], [1], -1)
        assert_raises(TypeError, herme.hermefit, [[1]], [1], 0)
        assert_raises(TypeError, herme.hermefit, [], [1], 0)
        assert_raises(TypeError, herme.hermefit, [1], [[[1]]], 0)
        assert_raises(TypeError, herme.hermefit, [1, 2], [1], 0)
        assert_raises(TypeError, herme.hermefit, [1], [1, 2], 0)
        assert_raises(TypeError, herme.hermefit, [1], [1], 0, w=[[1]])
        assert_raises(TypeError, herme.hermefit, [1], [1], 0, w=[1, 1])
        assert_raises(ValueError, herme.hermefit, [1], [1], [-1,])
        assert_raises(ValueError, herme.hermefit, [1], [1], [2, -1, 6])
        assert_raises(TypeError, herme.hermefit, [1], [1], [])

        # Test fit
        x = np.linspace(0, 2)
        y = f(x)
        #
        coef3 = herme.hermefit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(herme.hermeval(x, coef3), y)
        coef3 = herme.hermefit(x, y, [0, 1, 2, 3])
        assert_equal(len(coef3), 4)
        assert_almost_equal(herme.hermeval(x, coef3), y)
        #
        coef4 = herme.hermefit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(herme.hermeval(x, coef4), y)
        coef4 = herme.hermefit(x, y, [0, 1, 2, 3, 4])
        assert_equal(len(coef4), 5)
        assert_almost_equal(herme.hermeval(x, coef4), y)
        # check things still work if deg is not in strict increasing
        coef4 = herme.hermefit(x, y, [2, 3, 4, 1, 0])
        assert_equal(len(coef4), 5)
        assert_almost_equal(herme.hermeval(x, coef4), y)
        #
        coef2d = herme.hermefit(x, np.array([y, y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        coef2d = herme.hermefit(x, np.array([y, y]).T, [0, 1, 2, 3])
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        # test weighting
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = herme.hermefit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        wcoef3 = herme.hermefit(x, yw, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = herme.hermefit(x, np.array([yw, yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        wcoef2d = herme.hermefit(x, np.array([yw, yw]).T, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        assert_almost_equal(herme.hermefit(x, x, 1), [0, 1])
        assert_almost_equal(herme.hermefit(x, x, [0, 1]), [0, 1])
        # test fitting only even Legendre polynomials
        x = np.linspace(-1, 1)
        y = f2(x)
        coef1 = herme.hermefit(x, y, 4)
        assert_almost_equal(herme.hermeval(x, coef1), y)
        coef2 = herme.hermefit(x, y, [0, 2, 4])
        assert_almost_equal(herme.hermeval(x, coef2), y)
        assert_almost_equal(coef1, coef2)


class TestCompanion:

    def test_raises(self):
        assert_raises(ValueError, herme.hermecompanion, [])
        assert_raises(ValueError, herme.hermecompanion, [1])

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0] * i + [1]
            assert_(herme.hermecompanion(coef).shape == (i, i))

    def test_linear_root(self):
        assert_(herme.hermecompanion([1, 2])[0, 0] == -.5)


class TestGauss:

    def test_100(self):
        x, w = herme.hermegauss(100)

        # test orthogonality. Note that the results need to be normalized,
        # otherwise the huge values that can arise from fast growing
        # functions like Laguerre can be very confusing.
        v = herme.hermevander(x, 99)
        vv = np.dot(v.T * w, v)
        vd = 1 / np.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        assert_almost_equal(vv, np.eye(100))

        # check that the integral of 1 is correct
        tgt = np.sqrt(2 * np.pi)
        assert_almost_equal(w.sum(), tgt)


class TestMisc:

    def test_hermefromroots(self):
        res = herme.hermefromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = np.cos(np.linspace(-np.pi, 0, 2 * i + 1)[1::2])
            pol = herme.hermefromroots(roots)
            res = herme.hermeval(roots, pol)
            tgt = 0
            assert_(len(pol) == i + 1)
            assert_almost_equal(herme.herme2poly(pol)[-1], 1)
            assert_almost_equal(res, tgt)

    def test_hermeroots(self):
        assert_almost_equal(herme.hermeroots([1]), [])
        assert_almost_equal(herme.hermeroots([1, 1]), [-1])
        for i in range(2, 5):
            tgt = np.linspace(-1, 1, i)
            res = herme.hermeroots(herme.hermefromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_hermetrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        assert_raises(ValueError, herme.hermetrim, coef, -1)

        # Test results
        assert_equal(herme.hermetrim(coef), coef[:-1])
        assert_equal(herme.hermetrim(coef, 1), coef[:-3])
        assert_equal(herme.hermetrim(coef, 2), [0])

    def test_hermeline(self):
        assert_equal(herme.hermeline(3, 4), [3, 4])

    def test_herme2poly(self):
        for i in range(10):
            assert_almost_equal(herme.herme2poly([0] * i + [1]), Helist[i])

    def test_poly2herme(self):
        for i in range(10):
            assert_almost_equal(herme.poly2herme(Helist[i]), [0] * i + [1])

    def test_weight(self):
        x = np.linspace(-5, 5, 11)
        tgt = np.exp(-.5 * x**2)
        res = herme.hermeweight(x)
        assert_almost_equal(res, tgt)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\tests\test_hermite.py ===
"""Tests for hermite module.

"""
from functools import reduce

import numpy as np
import numpy.polynomial.hermite as herm
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

H0 = np.array([1])
H1 = np.array([0, 2])
H2 = np.array([-2, 0, 4])
H3 = np.array([0, -12, 0, 8])
H4 = np.array([12, 0, -48, 0, 16])
H5 = np.array([0, 120, 0, -160, 0, 32])
H6 = np.array([-120, 0, 720, 0, -480, 0, 64])
H7 = np.array([0, -1680, 0, 3360, 0, -1344, 0, 128])
H8 = np.array([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256])
H9 = np.array([0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512])

Hlist = [H0, H1, H2, H3, H4, H5, H6, H7, H8, H9]


def trim(x):
    return herm.hermtrim(x, tol=1e-6)


class TestConstants:

    def test_hermdomain(self):
        assert_equal(herm.hermdomain, [-1, 1])

    def test_hermzero(self):
        assert_equal(herm.hermzero, [0])

    def test_hermone(self):
        assert_equal(herm.hermone, [1])

    def test_hermx(self):
        assert_equal(herm.hermx, [0, .5])


class TestArithmetic:
    x = np.linspace(-3, 3, 100)

    def test_hermadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = herm.hermadd([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = herm.hermsub([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermmulx(self):
        assert_equal(herm.hermmulx([0]), [0])
        assert_equal(herm.hermmulx([1]), [0, .5])
        for i in range(1, 5):
            ser = [0] * i + [1]
            tgt = [0] * (i - 1) + [i, 0, .5]
            assert_equal(herm.hermmulx(ser), tgt)

    def test_hermmul(self):
        # check values of result
        for i in range(5):
            pol1 = [0] * i + [1]
            val1 = herm.hermval(self.x, pol1)
            for j in range(5):
                msg = f"At i={i}, j={j}"
                pol2 = [0] * j + [1]
                val2 = herm.hermval(self.x, pol2)
                pol3 = herm.hermmul(pol1, pol2)
                val3 = herm.hermval(self.x, pol3)
                assert_(len(pol3) == i + j + 1, msg)
                assert_almost_equal(val3, val1 * val2, err_msg=msg)

    def test_hermdiv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0] * i + [1]
                cj = [0] * j + [1]
                tgt = herm.hermadd(ci, cj)
                quo, rem = herm.hermdiv(tgt, ci)
                res = herm.hermadd(herm.hermmul(quo, ci), rem)
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermpow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = np.arange(i + 1)
                tgt = reduce(herm.hermmul, [c] * j, np.array([1]))
                res = herm.hermpow(c, j)
                assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = np.array([2.5, 1., .75])
    c2d = np.einsum('i,j->ij', c1d, c1d)
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = np.random.random((3, 5)) * 2 - 1
    y = polyval(x, [1., 2., 3.])

    def test_hermval(self):
        # check empty input
        assert_equal(herm.hermval([], [1]).size, 0)

        # check normal input)
        x = np.linspace(-1, 1)
        y = [polyval(x, c) for c in Hlist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = herm.hermval(x, [0] * i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)

        # check that shape is preserved
        for i in range(3):
            dims = [2] * i
            x = np.zeros(dims)
            assert_equal(herm.hermval(x, [1]).shape, dims)
            assert_equal(herm.hermval(x, [1, 0]).shape, dims)
            assert_equal(herm.hermval(x, [1, 0, 0]).shape, dims)

    def test_hermval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        assert_raises(ValueError, herm.hermval2d, x1, x2[:2], self.c2d)

        # test values
        tgt = y1 * y2
        res = herm.hermval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = herm.hermval2d(z, z, self.c2d)
        assert_(res.shape == (2, 3))

    def test_hermval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        assert_raises(ValueError, herm.hermval3d, x1, x2, x3[:2], self.c3d)

        # test values
        tgt = y1 * y2 * y3
        res = herm.hermval3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = herm.hermval3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3))

    def test_hermgrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = np.einsum('i,j->ij', y1, y2)
        res = herm.hermgrid2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = herm.hermgrid2d(z, z, self.c2d)
        assert_(res.shape == (2, 3) * 2)

    def test_hermgrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        res = herm.hermgrid3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = herm.hermgrid3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3) * 3)


class TestIntegral:

    def test_hermint(self):
        # check exceptions
        assert_raises(TypeError, herm.hermint, [0], .5)
        assert_raises(ValueError, herm.hermint, [0], -1)
        assert_raises(ValueError, herm.hermint, [0], 1, [0, 0])
        assert_raises(ValueError, herm.hermint, [0], lbnd=[0])
        assert_raises(ValueError, herm.hermint, [0], scl=[0])
        assert_raises(TypeError, herm.hermint, [0], axis=.5)

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = herm.hermint([0], m=i, k=k)
            assert_almost_equal(res, [0, .5])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            hermpol = herm.poly2herm(pol)
            hermint = herm.hermint(hermpol, m=1, k=[i])
            res = herm.herm2poly(hermint)
            assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            hermpol = herm.poly2herm(pol)
            hermint = herm.hermint(hermpol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(herm.hermval(-1, hermint), i)

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            hermpol = herm.poly2herm(pol)
            hermint = herm.hermint(hermpol, m=1, k=[i], scl=2)
            res = herm.herm2poly(hermint)
            assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = herm.hermint(tgt, m=1)
                res = herm.hermint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = herm.hermint(tgt, m=1, k=[k])
                res = herm.hermint(pol, m=j, k=list(range(j)))
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = herm.hermint(tgt, m=1, k=[k], lbnd=-1)
                res = herm.hermint(pol, m=j, k=list(range(j)), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = herm.hermint(tgt, m=1, k=[k], scl=2)
                res = herm.hermint(pol, m=j, k=list(range(j)), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_hermint_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([herm.hermint(c) for c in c2d.T]).T
        res = herm.hermint(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([herm.hermint(c) for c in c2d])
        res = herm.hermint(c2d, axis=1)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([herm.hermint(c, k=3) for c in c2d])
        res = herm.hermint(c2d, k=3, axis=1)
        assert_almost_equal(res, tgt)


class TestDerivative:

    def test_hermder(self):
        # check exceptions
        assert_raises(TypeError, herm.hermder, [0], .5)
        assert_raises(ValueError, herm.hermder, [0], -1)

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0] * i + [1]
            res = herm.hermder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = herm.hermder(herm.hermint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = herm.hermder(herm.hermint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_hermder_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([herm.hermder(c) for c in c2d.T]).T
        res = herm.hermder(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([herm.hermder(c) for c in c2d])
        res = herm.hermder(c2d, axis=1)
        assert_almost_equal(res, tgt)


class TestVander:
    # some random values in [-1, 1)
    x = np.random.random((3, 5)) * 2 - 1

    def test_hermvander(self):
        # check for 1d x
        x = np.arange(3)
        v = herm.hermvander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            assert_almost_equal(v[..., i], herm.hermval(x, coef))

        # check for 2d x
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = herm.hermvander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            assert_almost_equal(v[..., i], herm.hermval(x, coef))

    def test_hermvander2d(self):
        # also tests hermval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3))
        van = herm.hermvander2d(x1, x2, [1, 2])
        tgt = herm.hermval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = herm.hermvander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))

    def test_hermvander3d(self):
        # also tests hermval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3, 4))
        van = herm.hermvander3d(x1, x2, x3, [1, 2, 3])
        tgt = herm.hermval3d(x1, x2, x3, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = herm.hermvander3d([x1], [x2], [x3], [1, 2, 3])
        assert_(van.shape == (1, 5, 24))


class TestFitting:

    def test_hermfit(self):
        def f(x):
            return x * (x - 1) * (x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        # Test exceptions
        assert_raises(ValueError, herm.hermfit, [1], [1], -1)
        assert_raises(TypeError, herm.hermfit, [[1]], [1], 0)
        assert_raises(TypeError, herm.hermfit, [], [1], 0)
        assert_raises(TypeError, herm.hermfit, [1], [[[1]]], 0)
        assert_raises(TypeError, herm.hermfit, [1, 2], [1], 0)
        assert_raises(TypeError, herm.hermfit, [1], [1, 2], 0)
        assert_raises(TypeError, herm.hermfit, [1], [1], 0, w=[[1]])
        assert_raises(TypeError, herm.hermfit, [1], [1], 0, w=[1, 1])
        assert_raises(ValueError, herm.hermfit, [1], [1], [-1,])
        assert_raises(ValueError, herm.hermfit, [1], [1], [2, -1, 6])
        assert_raises(TypeError, herm.hermfit, [1], [1], [])

        # Test fit
        x = np.linspace(0, 2)
        y = f(x)
        #
        coef3 = herm.hermfit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(herm.hermval(x, coef3), y)
        coef3 = herm.hermfit(x, y, [0, 1, 2, 3])
        assert_equal(len(coef3), 4)
        assert_almost_equal(herm.hermval(x, coef3), y)
        #
        coef4 = herm.hermfit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(herm.hermval(x, coef4), y)
        coef4 = herm.hermfit(x, y, [0, 1, 2, 3, 4])
        assert_equal(len(coef4), 5)
        assert_almost_equal(herm.hermval(x, coef4), y)
        # check things still work if deg is not in strict increasing
        coef4 = herm.hermfit(x, y, [2, 3, 4, 1, 0])
        assert_equal(len(coef4), 5)
        assert_almost_equal(herm.hermval(x, coef4), y)
        #
        coef2d = herm.hermfit(x, np.array([y, y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        coef2d = herm.hermfit(x, np.array([y, y]).T, [0, 1, 2, 3])
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        # test weighting
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = herm.hermfit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        wcoef3 = herm.hermfit(x, yw, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = herm.hermfit(x, np.array([yw, yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        wcoef2d = herm.hermfit(x, np.array([yw, yw]).T, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        assert_almost_equal(herm.hermfit(x, x, 1), [0, .5])
        assert_almost_equal(herm.hermfit(x, x, [0, 1]), [0, .5])
        # test fitting only even Legendre polynomials
        x = np.linspace(-1, 1)
        y = f2(x)
        coef1 = herm.hermfit(x, y, 4)
        assert_almost_equal(herm.hermval(x, coef1), y)
        coef2 = herm.hermfit(x, y, [0, 2, 4])
        assert_almost_equal(herm.hermval(x, coef2), y)
        assert_almost_equal(coef1, coef2)


class TestCompanion:

    def test_raises(self):
        assert_raises(ValueError, herm.hermcompanion, [])
        assert_raises(ValueError, herm.hermcompanion, [1])

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0] * i + [1]
            assert_(herm.hermcompanion(coef).shape == (i, i))

    def test_linear_root(self):
        assert_(herm.hermcompanion([1, 2])[0, 0] == -.25)


class TestGauss:

    def test_100(self):
        x, w = herm.hermgauss(100)

        # test orthogonality. Note that the results need to be normalized,
        # otherwise the huge values that can arise from fast growing
        # functions like Laguerre can be very confusing.
        v = herm.hermvander(x, 99)
        vv = np.dot(v.T * w, v)
        vd = 1 / np.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        assert_almost_equal(vv, np.eye(100))

        # check that the integral of 1 is correct
        tgt = np.sqrt(np.pi)
        assert_almost_equal(w.sum(), tgt)


class TestMisc:

    def test_hermfromroots(self):
        res = herm.hermfromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = np.cos(np.linspace(-np.pi, 0, 2 * i + 1)[1::2])
            pol = herm.hermfromroots(roots)
            res = herm.hermval(roots, pol)
            tgt = 0
            assert_(len(pol) == i + 1)
            assert_almost_equal(herm.herm2poly(pol)[-1], 1)
            assert_almost_equal(res, tgt)

    def test_hermroots(self):
        assert_almost_equal(herm.hermroots([1]), [])
        assert_almost_equal(herm.hermroots([1, 1]), [-.5])
        for i in range(2, 5):
            tgt = np.linspace(-1, 1, i)
            res = herm.hermroots(herm.hermfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_hermtrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        assert_raises(ValueError, herm.hermtrim, coef, -1)

        # Test results
        assert_equal(herm.hermtrim(coef), coef[:-1])
        assert_equal(herm.hermtrim(coef, 1), coef[:-3])
        assert_equal(herm.hermtrim(coef, 2), [0])

    def test_hermline(self):
        assert_equal(herm.hermline(3, 4), [3, 2])

    def test_herm2poly(self):
        for i in range(10):
            assert_almost_equal(herm.herm2poly([0] * i + [1]), Hlist[i])

    def test_poly2herm(self):
        for i in range(10):
            assert_almost_equal(herm.poly2herm(Hlist[i]), [0] * i + [1])

    def test_weight(self):
        x = np.linspace(-5, 5, 11)
        tgt = np.exp(-x**2)
        res = herm.hermweight(x)
        assert_almost_equal(res, tgt)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\frequencies\test_inference.py ===
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas._libs.tslibs.ccalendar import (
    DAYS,
    MONTHS,
)
from pandas._libs.tslibs.offsets import _get_offset
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.compat import is_platform_windows

from pandas import (
    DatetimeIndex,
    Index,
    RangeIndex,
    Series,
    Timestamp,
    date_range,
    period_range,
)
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    TimedeltaArray,
)
from pandas.core.tools.datetimes import to_datetime

from pandas.tseries import (
    frequencies,
    offsets,
)


@pytest.fixture(
    params=[
        (timedelta(1), "D"),
        (timedelta(hours=1), "h"),
        (timedelta(minutes=1), "min"),
        (timedelta(seconds=1), "s"),
        (np.timedelta64(1, "ns"), "ns"),
        (timedelta(microseconds=1), "us"),
        (timedelta(microseconds=1000), "ms"),
    ]
)
def base_delta_code_pair(request):
    return request.param


freqs = (
    [f"QE-{month}" for month in MONTHS]
    + [f"{annual}-{month}" for annual in ["YE", "BYE"] for month in MONTHS]
    + ["ME", "BME", "BMS"]
    + [f"WOM-{count}{day}" for count in range(1, 5) for day in DAYS]
    + [f"W-{day}" for day in DAYS]
)


@pytest.mark.parametrize("freq", freqs)
@pytest.mark.parametrize("periods", [5, 7])
def test_infer_freq_range(periods, freq):
    freq = freq.upper()

    gen = date_range("1/1/2000", periods=periods, freq=freq)
    index = DatetimeIndex(gen.values)

    if not freq.startswith("QE-"):
        assert frequencies.infer_freq(index) == gen.freqstr
    else:
        inf_freq = frequencies.infer_freq(index)
        is_dec_range = inf_freq == "QE-DEC" and gen.freqstr in (
            "QE",
            "QE-DEC",
            "QE-SEP",
            "QE-JUN",
            "QE-MAR",
        )
        is_nov_range = inf_freq == "QE-NOV" and gen.freqstr in (
            "QE-NOV",
            "QE-AUG",
            "QE-MAY",
            "QE-FEB",
        )
        is_oct_range = inf_freq == "QE-OCT" and gen.freqstr in (
            "QE-OCT",
            "QE-JUL",
            "QE-APR",
            "QE-JAN",
        )
        assert is_dec_range or is_nov_range or is_oct_range


def test_raise_if_period_index():
    index = period_range(start="1/1/1990", periods=20, freq="M")
    msg = "Check the `freq` attribute instead of using infer_freq"

    with pytest.raises(TypeError, match=msg):
        frequencies.infer_freq(index)


def test_raise_if_too_few():
    index = DatetimeIndex(["12/31/1998", "1/3/1999"])
    msg = "Need at least 3 dates to infer frequency"

    with pytest.raises(ValueError, match=msg):
        frequencies.infer_freq(index)


def test_business_daily():
    index = DatetimeIndex(["01/01/1999", "1/4/1999", "1/5/1999"])
    assert frequencies.infer_freq(index) == "B"


def test_business_daily_look_alike():
    # see gh-16624
    #
    # Do not infer "B when "weekend" (2-day gap) in wrong place.
    index = DatetimeIndex(["12/31/1998", "1/3/1999", "1/4/1999"])
    assert frequencies.infer_freq(index) is None


def test_day_corner():
    index = DatetimeIndex(["1/1/2000", "1/2/2000", "1/3/2000"])
    assert frequencies.infer_freq(index) == "D"


def test_non_datetime_index():
    dates = to_datetime(["1/1/2000", "1/2/2000", "1/3/2000"])
    assert frequencies.infer_freq(dates) == "D"


def test_fifth_week_of_month_infer():
    # see gh-9425
    #
    # Only attempt to infer up to WOM-4.
    index = DatetimeIndex(["2014-03-31", "2014-06-30", "2015-03-30"])
    assert frequencies.infer_freq(index) is None


def test_week_of_month_fake():
    # All of these dates are on same day
    # of week and are 4 or 5 weeks apart.
    index = DatetimeIndex(["2013-08-27", "2013-10-01", "2013-10-29", "2013-11-26"])
    assert frequencies.infer_freq(index) != "WOM-4TUE"


def test_fifth_week_of_month():
    # see gh-9425
    #
    # Only supports freq up to WOM-4.
    msg = (
        "Of the four parameters: start, end, periods, "
        "and freq, exactly three must be specified"
    )

    with pytest.raises(ValueError, match=msg):
        date_range("2014-01-01", freq="WOM-5MON")


def test_monthly_ambiguous():
    rng = DatetimeIndex(["1/31/2000", "2/29/2000", "3/31/2000"])
    assert rng.inferred_freq == "ME"


def test_annual_ambiguous():
    rng = DatetimeIndex(["1/31/2000", "1/31/2001", "1/31/2002"])
    assert rng.inferred_freq == "YE-JAN"


@pytest.mark.parametrize("count", range(1, 5))
def test_infer_freq_delta(base_delta_code_pair, count):
    b = Timestamp(datetime.now())
    base_delta, code = base_delta_code_pair

    inc = base_delta * count
    index = DatetimeIndex([b + inc * j for j in range(3)])

    exp_freq = f"{count:d}{code}" if count > 1 else code
    assert frequencies.infer_freq(index) == exp_freq


@pytest.mark.parametrize(
    "constructor",
    [
        lambda now, delta: DatetimeIndex(
            [now + delta * 7] + [now + delta * j for j in range(3)]
        ),
        lambda now, delta: DatetimeIndex(
            [now + delta * j for j in range(3)] + [now + delta * 7]
        ),
    ],
)
def test_infer_freq_custom(base_delta_code_pair, constructor):
    b = Timestamp(datetime.now())
    base_delta, _ = base_delta_code_pair

    index = constructor(b, base_delta)
    assert frequencies.infer_freq(index) is None


@pytest.mark.parametrize(
    "freq,expected", [("Q", "QE-DEC"), ("Q-NOV", "QE-NOV"), ("Q-OCT", "QE-OCT")]
)
def test_infer_freq_index(freq, expected):
    rng = period_range("1959Q2", "2009Q3", freq=freq)
    with tm.assert_produces_warning(FutureWarning, match="Dtype inference"):
        rng = Index(rng.to_timestamp("D", how="e").astype(object))

    assert rng.inferred_freq == expected


@pytest.mark.parametrize(
    "expected,dates",
    list(
        {
            "YS-JAN": ["2009-01-01", "2010-01-01", "2011-01-01", "2012-01-01"],
            "QE-OCT": ["2009-01-31", "2009-04-30", "2009-07-31", "2009-10-31"],
            "ME": ["2010-11-30", "2010-12-31", "2011-01-31", "2011-02-28"],
            "W-SAT": ["2010-12-25", "2011-01-01", "2011-01-08", "2011-01-15"],
            "D": ["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"],
            "h": [
                "2011-12-31 22:00",
                "2011-12-31 23:00",
                "2012-01-01 00:00",
                "2012-01-01 01:00",
            ],
        }.items()
    ),
)
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_infer_freq_tz(tz_naive_fixture, expected, dates, unit):
    # see gh-7310, GH#55609
    tz = tz_naive_fixture
    idx = DatetimeIndex(dates, tz=tz).as_unit(unit)
    assert idx.inferred_freq == expected


def test_infer_freq_tz_series(tz_naive_fixture):
    # infer_freq should work with both tz-naive and tz-aware series. See gh-52456
    tz = tz_naive_fixture
    idx = date_range("2021-01-01", "2021-01-04", tz=tz)
    series = idx.to_series().reset_index(drop=True)
    inferred_freq = frequencies.infer_freq(series)
    assert inferred_freq == "D"


@pytest.mark.parametrize(
    "date_pair",
    [
        ["2013-11-02", "2013-11-5"],  # Fall DST
        ["2014-03-08", "2014-03-11"],  # Spring DST
        ["2014-01-01", "2014-01-03"],  # Regular Time
    ],
)
@pytest.mark.parametrize(
    "freq",
    ["h", "3h", "10min", "3601s", "3600001ms", "3600000001us", "3600000000001ns"],
)
def test_infer_freq_tz_transition(tz_naive_fixture, date_pair, freq):
    # see gh-8772
    tz = tz_naive_fixture
    idx = date_range(date_pair[0], date_pair[1], freq=freq, tz=tz)
    assert idx.inferred_freq == freq


def test_infer_freq_tz_transition_custom():
    index = date_range("2013-11-03", periods=5, freq="3h").tz_localize(
        "America/Chicago"
    )
    assert index.inferred_freq is None


@pytest.mark.parametrize(
    "data,expected",
    [
        # Hourly freq in a day must result in "h"
        (
            [
                "2014-07-01 09:00",
                "2014-07-01 10:00",
                "2014-07-01 11:00",
                "2014-07-01 12:00",
                "2014-07-01 13:00",
                "2014-07-01 14:00",
            ],
            "h",
        ),
        (
            [
                "2014-07-01 09:00",
                "2014-07-01 10:00",
                "2014-07-01 11:00",
                "2014-07-01 12:00",
                "2014-07-01 13:00",
                "2014-07-01 14:00",
                "2014-07-01 15:00",
                "2014-07-01 16:00",
                "2014-07-02 09:00",
                "2014-07-02 10:00",
                "2014-07-02 11:00",
            ],
            "bh",
        ),
        (
            [
                "2014-07-04 09:00",
                "2014-07-04 10:00",
                "2014-07-04 11:00",
                "2014-07-04 12:00",
                "2014-07-04 13:00",
                "2014-07-04 14:00",
                "2014-07-04 15:00",
                "2014-07-04 16:00",
                "2014-07-07 09:00",
                "2014-07-07 10:00",
                "2014-07-07 11:00",
            ],
            "bh",
        ),
        (
            [
                "2014-07-04 09:00",
                "2014-07-04 10:00",
                "2014-07-04 11:00",
                "2014-07-04 12:00",
                "2014-07-04 13:00",
                "2014-07-04 14:00",
                "2014-07-04 15:00",
                "2014-07-04 16:00",
                "2014-07-07 09:00",
                "2014-07-07 10:00",
                "2014-07-07 11:00",
                "2014-07-07 12:00",
                "2014-07-07 13:00",
                "2014-07-07 14:00",
                "2014-07-07 15:00",
                "2014-07-07 16:00",
                "2014-07-08 09:00",
                "2014-07-08 10:00",
                "2014-07-08 11:00",
                "2014-07-08 12:00",
                "2014-07-08 13:00",
                "2014-07-08 14:00",
                "2014-07-08 15:00",
                "2014-07-08 16:00",
            ],
            "bh",
        ),
    ],
)
def test_infer_freq_business_hour(data, expected):
    # see gh-7905
    idx = DatetimeIndex(data)
    assert idx.inferred_freq == expected


def test_not_monotonic():
    rng = DatetimeIndex(["1/31/2000", "1/31/2001", "1/31/2002"])
    rng = rng[::-1]

    assert rng.inferred_freq == "-1YE-JAN"


def test_non_datetime_index2():
    rng = DatetimeIndex(["1/31/2000", "1/31/2001", "1/31/2002"])
    vals = rng.to_pydatetime()

    result = frequencies.infer_freq(vals)
    assert result == rng.inferred_freq


@pytest.mark.parametrize(
    "idx",
    [
        Index(np.arange(5), dtype=np.int64),
        Index(np.arange(5), dtype=np.float64),
        period_range("2020-01-01", periods=5),
        RangeIndex(5),
    ],
)
def test_invalid_index_types(idx):
    # see gh-48439
    msg = "|".join(
        [
            "cannot infer freq from a non-convertible",
            "Check the `freq` attribute instead of using infer_freq",
        ]
    )

    with pytest.raises(TypeError, match=msg):
        frequencies.infer_freq(idx)


@pytest.mark.skipif(is_platform_windows(), reason="see gh-10822: Windows issue")
def test_invalid_index_types_unicode():
    # see gh-10822
    #
    # Odd error message on conversions to datetime for unicode.
    msg = "Unknown datetime string format"

    with pytest.raises(ValueError, match=msg):
        frequencies.infer_freq(Index(["ZqgszYBfuL"]))


def test_string_datetime_like_compat():
    # see gh-6463
    data = ["2004-01", "2004-02", "2004-03", "2004-04"]

    expected = frequencies.infer_freq(data)
    result = frequencies.infer_freq(Index(data))

    assert result == expected


def test_series():
    # see gh-6407
    s = Series(date_range("20130101", "20130110"))
    inferred = frequencies.infer_freq(s)
    assert inferred == "D"


@pytest.mark.parametrize("end", [10, 10.0])
def test_series_invalid_type(end):
    # see gh-6407
    msg = "cannot infer freq from a non-convertible dtype on a Series"
    s = Series(np.arange(end))

    with pytest.raises(TypeError, match=msg):
        frequencies.infer_freq(s)


def test_series_inconvertible_string(using_infer_string):
    # see gh-6407
    if using_infer_string:
        msg = "cannot infer freq from"

        with pytest.raises(TypeError, match=msg):
            frequencies.infer_freq(Series(["foo", "bar"]))
    else:
        msg = "Unknown datetime string format"

        with pytest.raises(ValueError, match=msg):
            frequencies.infer_freq(Series(["foo", "bar"]))


@pytest.mark.parametrize("freq", [None, "ms"])
def test_series_period_index(freq):
    # see gh-6407
    #
    # Cannot infer on PeriodIndex
    msg = "cannot infer freq from a non-convertible dtype on a Series"
    s = Series(period_range("2013", periods=10, freq=freq))

    with pytest.raises(TypeError, match=msg):
        frequencies.infer_freq(s)


@pytest.mark.parametrize("freq", ["ME", "ms", "s"])
def test_series_datetime_index(freq):
    s = Series(date_range("20130101", periods=10, freq=freq))
    inferred = frequencies.infer_freq(s)
    assert inferred == freq


@pytest.mark.parametrize(
    "offset_func",
    [
        _get_offset,
        lambda freq: date_range("2011-01-01", periods=5, freq=freq),
    ],
)
@pytest.mark.parametrize(
    "freq",
    [
        "WEEKDAY",
        "EOM",
        "W@MON",
        "W@TUE",
        "W@WED",
        "W@THU",
        "W@FRI",
        "W@SAT",
        "W@SUN",
        "QE@JAN",
        "QE@FEB",
        "QE@MAR",
        "YE@JAN",
        "YE@FEB",
        "YE@MAR",
        "YE@APR",
        "YE@MAY",
        "YE@JUN",
        "YE@JUL",
        "YE@AUG",
        "YE@SEP",
        "YE@OCT",
        "YE@NOV",
        "YE@DEC",
        "YE@JAN",
        "WOM@1MON",
        "WOM@2MON",
        "WOM@3MON",
        "WOM@4MON",
        "WOM@1TUE",
        "WOM@2TUE",
        "WOM@3TUE",
        "WOM@4TUE",
        "WOM@1WED",
        "WOM@2WED",
        "WOM@3WED",
        "WOM@4WED",
        "WOM@1THU",
        "WOM@2THU",
        "WOM@3THU",
        "WOM@4THU",
        "WOM@1FRI",
        "WOM@2FRI",
        "WOM@3FRI",
        "WOM@4FRI",
    ],
)
def test_legacy_offset_warnings(offset_func, freq):
    with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
        offset_func(freq)


def test_ms_vs_capital_ms():
    left = _get_offset("ms")
    right = _get_offset("MS")

    assert left == offsets.Milli()
    assert right == offsets.MonthBegin()


def test_infer_freq_non_nano():
    arr = np.arange(10).astype(np.int64).view("M8[s]")
    dta = DatetimeArray._simple_new(arr, dtype=arr.dtype)
    res = frequencies.infer_freq(dta)
    assert res == "s"

    arr2 = arr.view("m8[ms]")
    tda = TimedeltaArray._simple_new(arr2, dtype=arr2.dtype)
    res2 = frequencies.infer_freq(tda)
    assert res2 == "ms"


def test_infer_freq_non_nano_tzaware(tz_aware_fixture):
    tz = tz_aware_fixture

    dti = date_range("2016-01-01", periods=365, freq="B", tz=tz)
    dta = dti._data.as_unit("s")

    res = frequencies.infer_freq(dta)
    assert res == "B"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\tests\test_ddm.py ===
from sympy.testing.pytest import raises
from sympy.external.gmpy import GROUND_TYPES

from sympy.polys import ZZ, QQ

from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (
    DMShapeError, DMNonInvertibleMatrixError, DMDomainError,
    DMBadInputError)


def test_DDM_init():
    items = [[ZZ(0), ZZ(1), ZZ(2)], [ZZ(3), ZZ(4), ZZ(5)]]
    shape = (2, 3)
    ddm = DDM(items, shape, ZZ)
    assert ddm.shape == shape
    assert ddm.rows == 2
    assert ddm.cols == 3
    assert ddm.domain == ZZ

    raises(DMBadInputError, lambda: DDM([[ZZ(2), ZZ(3)]], (2, 2), ZZ))
    raises(DMBadInputError, lambda: DDM([[ZZ(1)], [ZZ(2), ZZ(3)]], (2, 2), ZZ))


def test_DDM_getsetitem():
    ddm = DDM([[ZZ(2), ZZ(3)], [ZZ(4), ZZ(5)]], (2, 2), ZZ)

    assert ddm[0][0] == ZZ(2)
    assert ddm[0][1] == ZZ(3)
    assert ddm[1][0] == ZZ(4)
    assert ddm[1][1] == ZZ(5)

    raises(IndexError, lambda: ddm[2][0])
    raises(IndexError, lambda: ddm[0][2])

    ddm[0][0] = ZZ(-1)
    assert ddm[0][0] == ZZ(-1)


def test_DDM_str():
    ddm = DDM([[ZZ(0), ZZ(1)], [ZZ(2), ZZ(3)]], (2, 2), ZZ)
    if GROUND_TYPES == 'gmpy': # pragma: no cover
        assert str(ddm) == '[[0, 1], [2, 3]]'
        assert repr(ddm) == 'DDM([[mpz(0), mpz(1)], [mpz(2), mpz(3)]], (2, 2), ZZ)'
    else:        # pragma: no cover
        assert repr(ddm) == 'DDM([[0, 1], [2, 3]], (2, 2), ZZ)'
        assert str(ddm) == '[[0, 1], [2, 3]]'


def test_DDM_eq():
    items = [[ZZ(0), ZZ(1)], [ZZ(2), ZZ(3)]]
    ddm1 = DDM(items, (2, 2), ZZ)
    ddm2 = DDM(items, (2, 2), ZZ)

    assert (ddm1 == ddm1) is True
    assert (ddm1 == items) is False
    assert (items == ddm1) is False
    assert (ddm1 == ddm2) is True
    assert (ddm2 == ddm1) is True

    assert (ddm1 != ddm1) is False
    assert (ddm1 != items) is True
    assert (items != ddm1) is True
    assert (ddm1 != ddm2) is False
    assert (ddm2 != ddm1) is False

    ddm3 = DDM([[ZZ(0), ZZ(1)], [ZZ(3), ZZ(3)]], (2, 2), ZZ)
    ddm3 = DDM(items, (2, 2), QQ)

    assert (ddm1 == ddm3) is False
    assert (ddm3 == ddm1) is False
    assert (ddm1 != ddm3) is True
    assert (ddm3 != ddm1) is True


def test_DDM_convert_to():
    ddm = DDM([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    assert ddm.convert_to(ZZ) == ddm
    ddmq = ddm.convert_to(QQ)
    assert ddmq.domain == QQ


def test_DDM_zeros():
    ddmz = DDM.zeros((3, 4), QQ)
    assert list(ddmz) == [[QQ(0)] * 4] * 3
    assert ddmz.shape == (3, 4)
    assert ddmz.domain == QQ

def test_DDM_ones():
    ddmone = DDM.ones((2, 3), QQ)
    assert list(ddmone) == [[QQ(1)] * 3] * 2
    assert ddmone.shape == (2, 3)
    assert ddmone.domain == QQ

def test_DDM_eye():
    ddmz = DDM.eye(3, QQ)
    f = lambda i, j: QQ(1) if i == j else QQ(0)
    assert list(ddmz) == [[f(i, j) for i in range(3)] for j in range(3)]
    assert ddmz.shape == (3, 3)
    assert ddmz.domain == QQ


def test_DDM_copy():
    ddm1 = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    ddm2 = ddm1.copy()
    assert (ddm1 == ddm2) is True
    ddm1[0][0] = QQ(-1)
    assert (ddm1 == ddm2) is False
    ddm2[0][0] = QQ(-1)
    assert (ddm1 == ddm2) is True


def test_DDM_transpose():
    ddm = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    ddmT = DDM([[QQ(1), QQ(2)]], (1, 2), QQ)
    assert ddm.transpose() == ddmT
    ddm02 = DDM([], (0, 2), QQ)
    ddm02T = DDM([[], []], (2, 0), QQ)
    assert ddm02.transpose() == ddm02T
    assert ddm02T.transpose() == ddm02
    ddm0 = DDM([], (0, 0), QQ)
    assert ddm0.transpose() == ddm0


def test_DDM_add():
    A = DDM([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    B = DDM([[ZZ(3)], [ZZ(4)]], (2, 1), ZZ)
    C = DDM([[ZZ(4)], [ZZ(6)]], (2, 1), ZZ)
    AQ = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    assert A + B == A.add(B) == C

    raises(DMShapeError, lambda: A + DDM([[ZZ(5)]], (1, 1), ZZ))
    raises(TypeError, lambda: A + ZZ(1))
    raises(TypeError, lambda: ZZ(1) + A)
    raises(DMDomainError, lambda: A + AQ)
    raises(DMDomainError, lambda: AQ + A)


def test_DDM_sub():
    A = DDM([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    B = DDM([[ZZ(3)], [ZZ(4)]], (2, 1), ZZ)
    C = DDM([[ZZ(-2)], [ZZ(-2)]], (2, 1), ZZ)
    AQ = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    D = DDM([[ZZ(5)]], (1, 1), ZZ)
    assert A - B == A.sub(B) == C

    raises(TypeError, lambda: A - ZZ(1))
    raises(TypeError, lambda: ZZ(1) - A)
    raises(DMShapeError, lambda: A - D)
    raises(DMShapeError, lambda: D - A)
    raises(DMShapeError, lambda: A.sub(D))
    raises(DMShapeError, lambda: D.sub(A))
    raises(DMDomainError, lambda: A - AQ)
    raises(DMDomainError, lambda: AQ - A)
    raises(DMDomainError, lambda: A.sub(AQ))
    raises(DMDomainError, lambda: AQ.sub(A))


def test_DDM_neg():
    A = DDM([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    An = DDM([[ZZ(-1)], [ZZ(-2)]], (2, 1), ZZ)
    assert -A == A.neg() == An
    assert -An == An.neg() == A


def test_DDM_mul():
    A = DDM([[ZZ(1)]], (1, 1), ZZ)
    A2 = DDM([[ZZ(2)]], (1, 1), ZZ)
    assert A * ZZ(2) == A2
    assert ZZ(2) * A == A2
    raises(TypeError, lambda: [[1]] * A)
    raises(TypeError, lambda: A * [[1]])


def test_DDM_matmul():
    A = DDM([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    B = DDM([[ZZ(3), ZZ(4)]], (1, 2), ZZ)
    AB = DDM([[ZZ(3), ZZ(4)], [ZZ(6), ZZ(8)]], (2, 2), ZZ)
    BA = DDM([[ZZ(11)]], (1, 1), ZZ)

    assert A @ B == A.matmul(B) == AB
    assert B @ A == B.matmul(A) == BA

    raises(TypeError, lambda: A @ 1)
    raises(TypeError, lambda: A @ [[3, 4]])

    Bq = DDM([[QQ(3), QQ(4)]], (1, 2), QQ)

    raises(DMDomainError, lambda: A @ Bq)
    raises(DMDomainError, lambda: Bq @ A)

    C = DDM([[ZZ(1)]], (1, 1), ZZ)

    assert A @ C == A.matmul(C) == A

    raises(DMShapeError, lambda: C @ A)
    raises(DMShapeError, lambda: C.matmul(A))

    Z04 = DDM([], (0, 4), ZZ)
    Z40 = DDM([[]]*4, (4, 0), ZZ)
    Z50 = DDM([[]]*5, (5, 0), ZZ)
    Z05 = DDM([], (0, 5), ZZ)
    Z45 = DDM([[0] * 5] * 4, (4, 5), ZZ)
    Z54 = DDM([[0] * 4] * 5, (5, 4), ZZ)
    Z00 = DDM([], (0, 0), ZZ)

    assert Z04 @ Z45 == Z04.matmul(Z45) == Z05
    assert Z45 @ Z50 == Z45.matmul(Z50) == Z40
    assert Z00 @ Z04 == Z00.matmul(Z04) == Z04
    assert Z50 @ Z00 == Z50.matmul(Z00) == Z50
    assert Z00 @ Z00 == Z00.matmul(Z00) == Z00
    assert Z50 @ Z04 == Z50.matmul(Z04) == Z54

    raises(DMShapeError, lambda: Z05 @ Z40)
    raises(DMShapeError, lambda: Z05.matmul(Z40))


def test_DDM_hstack():
    A = DDM([[ZZ(1), ZZ(2), ZZ(3)]], (1, 3), ZZ)
    B = DDM([[ZZ(4), ZZ(5)]], (1, 2), ZZ)
    C = DDM([[ZZ(6)]], (1, 1), ZZ)

    Ah = A.hstack(B)
    assert Ah.shape == (1, 5)
    assert Ah.domain == ZZ
    assert Ah == DDM([[ZZ(1), ZZ(2), ZZ(3), ZZ(4), ZZ(5)]], (1, 5), ZZ)

    Ah = A.hstack(B, C)
    assert Ah.shape == (1, 6)
    assert Ah.domain == ZZ
    assert Ah == DDM([[ZZ(1), ZZ(2), ZZ(3), ZZ(4), ZZ(5), ZZ(6)]], (1, 6), ZZ)


def test_DDM_vstack():
    A = DDM([[ZZ(1)], [ZZ(2)], [ZZ(3)]], (3, 1), ZZ)
    B = DDM([[ZZ(4)], [ZZ(5)]], (2, 1), ZZ)
    C = DDM([[ZZ(6)]], (1, 1), ZZ)

    Ah = A.vstack(B)
    assert Ah.shape == (5, 1)
    assert Ah.domain == ZZ
    assert Ah == DDM([[ZZ(1)], [ZZ(2)], [ZZ(3)], [ZZ(4)], [ZZ(5)]], (5, 1), ZZ)

    Ah = A.vstack(B, C)
    assert Ah.shape == (6, 1)
    assert Ah.domain == ZZ
    assert Ah == DDM([[ZZ(1)], [ZZ(2)], [ZZ(3)], [ZZ(4)], [ZZ(5)], [ZZ(6)]], (6, 1), ZZ)


def test_DDM_applyfunc():
    A = DDM([[ZZ(1), ZZ(2), ZZ(3)]], (1, 3), ZZ)
    B = DDM([[ZZ(2), ZZ(4), ZZ(6)]], (1, 3), ZZ)
    assert A.applyfunc(lambda x: 2*x, ZZ) == B

def test_DDM_rref():

    A = DDM([], (0, 4), QQ)
    assert A.rref() == (A, [])

    A = DDM([[QQ(0), QQ(1)], [QQ(1), QQ(1)]], (2, 2), QQ)
    Ar = DDM([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
    pivots = [0, 1]
    assert A.rref() == (Ar, pivots)

    A = DDM([[QQ(1), QQ(2), QQ(1)], [QQ(3), QQ(4), QQ(1)]], (2, 3), QQ)
    Ar = DDM([[QQ(1), QQ(0), QQ(-1)], [QQ(0), QQ(1), QQ(1)]], (2, 3), QQ)
    pivots = [0, 1]
    assert A.rref() == (Ar, pivots)

    A = DDM([[QQ(3), QQ(4), QQ(1)], [QQ(1), QQ(2), QQ(1)]], (2, 3), QQ)
    Ar = DDM([[QQ(1), QQ(0), QQ(-1)], [QQ(0), QQ(1), QQ(1)]], (2, 3), QQ)
    pivots = [0, 1]
    assert A.rref() == (Ar, pivots)

    A = DDM([[QQ(1), QQ(0)], [QQ(1), QQ(3)], [QQ(0), QQ(1)]], (3, 2), QQ)
    Ar = DDM([[QQ(1), QQ(0)], [QQ(0), QQ(1)], [QQ(0), QQ(0)]], (3, 2), QQ)
    pivots = [0, 1]
    assert A.rref() == (Ar, pivots)

    A = DDM([[QQ(1), QQ(0), QQ(1)], [QQ(3), QQ(0), QQ(1)]], (2, 3), QQ)
    Ar = DDM([[QQ(1), QQ(0), QQ(0)], [QQ(0), QQ(0), QQ(1)]], (2, 3), QQ)
    pivots = [0, 2]
    assert A.rref() == (Ar, pivots)


def test_DDM_nullspace():
    # more tests are in test_nullspace.py
    A = DDM([[QQ(1), QQ(1)], [QQ(1), QQ(1)]], (2, 2), QQ)
    Anull = DDM([[QQ(-1), QQ(1)]], (1, 2), QQ)
    nonpivots = [1]
    assert A.nullspace() == (Anull, nonpivots)


def test_DDM_particular():
    A = DDM([[QQ(1), QQ(0)]], (1, 2), QQ)
    assert A.particular() == DDM.zeros((1, 1), QQ)


def test_DDM_det():
    # 0x0 case
    A = DDM([], (0, 0), ZZ)
    assert A.det() == ZZ(1)

    # 1x1 case
    A = DDM([[ZZ(2)]], (1, 1), ZZ)
    assert A.det() == ZZ(2)

    # 2x2 case
    A = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.det() == ZZ(-2)

    # 3x3 with swap
    A = DDM([[ZZ(1), ZZ(2), ZZ(3)], [ZZ(1), ZZ(2), ZZ(4)], [ZZ(1), ZZ(2), ZZ(5)]], (3, 3), ZZ)
    assert A.det() == ZZ(0)

    # 2x2 QQ case
    A = DDM([[QQ(1, 2), QQ(1, 2)], [QQ(1, 3), QQ(1, 4)]], (2, 2), QQ)
    assert A.det() == QQ(-1, 24)

    # Nonsquare error
    A = DDM([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    raises(DMShapeError, lambda: A.det())

    # Nonsquare error with empty matrix
    A = DDM([], (0, 1), ZZ)
    raises(DMShapeError, lambda: A.det())


def test_DDM_inv():
    A = DDM([[QQ(1, 1), QQ(2, 1)], [QQ(3, 1), QQ(4, 1)]], (2, 2), QQ)
    Ainv = DDM([[QQ(-2, 1), QQ(1, 1)], [QQ(3, 2), QQ(-1, 2)]], (2, 2), QQ)
    assert A.inv() == Ainv

    A = DDM([[QQ(1), QQ(2)]], (1, 2), QQ)
    raises(DMShapeError, lambda: A.inv())

    A = DDM([[ZZ(2)]], (1, 1), ZZ)
    raises(DMDomainError, lambda: A.inv())

    A = DDM([], (0, 0), QQ)
    assert A.inv() == A

    A = DDM([[QQ(1), QQ(2)], [QQ(2), QQ(4)]], (2, 2), QQ)
    raises(DMNonInvertibleMatrixError, lambda: A.inv())


def test_DDM_lu():
    A = DDM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    L, U, swaps = A.lu()
    assert L == DDM([[QQ(1), QQ(0)], [QQ(3), QQ(1)]], (2, 2), QQ)
    assert U == DDM([[QQ(1), QQ(2)], [QQ(0), QQ(-2)]], (2, 2), QQ)
    assert swaps == []

    A = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 2]]
    Lexp = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]
    Uexp = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
    to_dom = lambda rows, dom: [[dom(e) for e in row] for row in rows]
    A = DDM(to_dom(A, QQ), (4, 4), QQ)
    Lexp = DDM(to_dom(Lexp, QQ), (4, 4), QQ)
    Uexp = DDM(to_dom(Uexp, QQ), (4, 4), QQ)
    L, U, swaps = A.lu()
    assert L == Lexp
    assert U == Uexp
    assert swaps == []


def test_DDM_lu_solve():
    # Basic example
    A = DDM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    b = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    x = DDM([[QQ(0)], [QQ(1, 2)]], (2, 1), QQ)
    assert A.lu_solve(b) == x

    # Example with swaps
    A = DDM([[QQ(0), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    assert A.lu_solve(b) == x

    # Overdetermined, consistent
    A = DDM([[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(5), QQ(6)]], (3, 2), QQ)
    b = DDM([[QQ(1)], [QQ(2)], [QQ(3)]], (3, 1), QQ)
    assert A.lu_solve(b) == x

    # Overdetermined, inconsistent
    b = DDM([[QQ(1)], [QQ(2)], [QQ(4)]], (3, 1), QQ)
    raises(DMNonInvertibleMatrixError, lambda: A.lu_solve(b))

    # Square, noninvertible
    A = DDM([[QQ(1), QQ(2)], [QQ(1), QQ(2)]], (2, 2), QQ)
    b = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    raises(DMNonInvertibleMatrixError, lambda: A.lu_solve(b))

    # Underdetermined
    A = DDM([[QQ(1), QQ(2)]], (1, 2), QQ)
    b = DDM([[QQ(3)]], (1, 1), QQ)
    raises(NotImplementedError, lambda: A.lu_solve(b))

    # Domain mismatch
    bz = DDM([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    raises(DMDomainError, lambda: A.lu_solve(bz))

    # Shape mismatch
    b3 = DDM([[QQ(1)], [QQ(2)], [QQ(3)]], (3, 1), QQ)
    raises(DMShapeError, lambda: A.lu_solve(b3))


def test_DDM_charpoly():
    A = DDM([], (0, 0), ZZ)
    assert A.charpoly() == [ZZ(1)]

    A = DDM([
        [ZZ(1), ZZ(2), ZZ(3)],
        [ZZ(4), ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)
    Avec = [ZZ(1), ZZ(-15), ZZ(-18), ZZ(0)]
    assert A.charpoly() == Avec

    A = DDM([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMShapeError, lambda: A.charpoly())


def test_DDM_getitem():
    dm = DDM([
        [ZZ(1), ZZ(2), ZZ(3)],
        [ZZ(4), ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)

    assert dm.getitem(1, 1) == ZZ(5)
    assert dm.getitem(1, -2) == ZZ(5)
    assert dm.getitem(-1, -3) == ZZ(7)

    raises(IndexError, lambda: dm.getitem(3, 3))


def test_DDM_setitem():
    dm = DDM.zeros((3, 3), ZZ)
    dm.setitem(0, 0, 1)
    dm.setitem(1, -2, 1)
    dm.setitem(-1, -1, 1)
    assert dm == DDM.eye(3, ZZ)

    raises(IndexError, lambda: dm.setitem(3, 3, 0))


def test_DDM_extract_slice():
    dm = DDM([
        [ZZ(1), ZZ(2), ZZ(3)],
        [ZZ(4), ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)

    assert dm.extract_slice(slice(0, 3), slice(0, 3)) == dm
    assert dm.extract_slice(slice(1, 3), slice(-2)) == DDM([[4], [7]], (2, 1), ZZ)
    assert dm.extract_slice(slice(1, 3), slice(-2)) == DDM([[4], [7]], (2, 1), ZZ)
    assert dm.extract_slice(slice(2, 3), slice(-2)) == DDM([[ZZ(7)]], (1, 1), ZZ)
    assert dm.extract_slice(slice(0, 2), slice(-2)) == DDM([[1], [4]], (2, 1), ZZ)
    assert dm.extract_slice(slice(-1), slice(-1)) == DDM([[1, 2], [4, 5]], (2, 2), ZZ)

    assert dm.extract_slice(slice(2), slice(3, 4)) == DDM([[], []], (2, 0), ZZ)
    assert dm.extract_slice(slice(3, 4), slice(2)) == DDM([], (0, 2), ZZ)
    assert dm.extract_slice(slice(3, 4), slice(3, 4)) == DDM([], (0, 0), ZZ)


def test_DDM_extract():
    dm1 = DDM([
        [ZZ(1), ZZ(2), ZZ(3)],
        [ZZ(4), ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)
    dm2 = DDM([
        [ZZ(6), ZZ(4)],
        [ZZ(3), ZZ(1)]], (2, 2), ZZ)
    assert dm1.extract([1, 0], [2, 0]) == dm2
    assert dm1.extract([-2, 0], [-1, 0]) == dm2

    assert dm1.extract([], []) == DDM.zeros((0, 0), ZZ)
    assert dm1.extract([1], []) == DDM.zeros((1, 0), ZZ)
    assert dm1.extract([], [1]) == DDM.zeros((0, 1), ZZ)

    raises(IndexError, lambda: dm2.extract([2], [0]))
    raises(IndexError, lambda: dm2.extract([0], [2]))
    raises(IndexError, lambda: dm2.extract([-3], [0]))
    raises(IndexError, lambda: dm2.extract([0], [-3]))


def test_DDM_flat():
    dm = DDM([
        [ZZ(6), ZZ(4)],
        [ZZ(3), ZZ(1)]], (2, 2), ZZ)
    assert dm.flat() == [ZZ(6), ZZ(4), ZZ(3), ZZ(1)]


def test_DDM_is_zero_matrix():
    A = DDM([[QQ(1), QQ(0)], [QQ(0), QQ(0)]], (2, 2), QQ)
    Azero = DDM.zeros((1, 2), QQ)
    assert A.is_zero_matrix() is False
    assert Azero.is_zero_matrix() is True


def test_DDM_is_upper():
    # Wide matrices:
    A = DDM([
        [QQ(1), QQ(2), QQ(3), QQ(4)],
        [QQ(0), QQ(5), QQ(6), QQ(7)],
        [QQ(0), QQ(0), QQ(8), QQ(9)]
    ], (3, 4), QQ)
    B = DDM([
        [QQ(1), QQ(2), QQ(3), QQ(4)],
        [QQ(0), QQ(5), QQ(6), QQ(7)],
        [QQ(0), QQ(7), QQ(8), QQ(9)]
    ], (3, 4), QQ)
    assert A.is_upper() is True
    assert B.is_upper() is False

    # Tall matrices:
    A = DDM([
        [QQ(1), QQ(2), QQ(3)],
        [QQ(0), QQ(5), QQ(6)],
        [QQ(0), QQ(0), QQ(8)],
        [QQ(0), QQ(0), QQ(0)]
    ], (4, 3), QQ)
    B = DDM([
        [QQ(1), QQ(2), QQ(3)],
        [QQ(0), QQ(5), QQ(6)],
        [QQ(0), QQ(0), QQ(8)],
        [QQ(0), QQ(0), QQ(10)]
    ], (4, 3), QQ)
    assert A.is_upper() is True
    assert B.is_upper() is False


def test_DDM_is_lower():
    # Tall matrices:
    A = DDM([
        [QQ(1), QQ(2), QQ(3), QQ(4)],
        [QQ(0), QQ(5), QQ(6), QQ(7)],
        [QQ(0), QQ(0), QQ(8), QQ(9)]
    ], (3, 4), QQ).transpose()
    B = DDM([
        [QQ(1), QQ(2), QQ(3), QQ(4)],
        [QQ(0), QQ(5), QQ(6), QQ(7)],
        [QQ(0), QQ(7), QQ(8), QQ(9)]
    ], (3, 4), QQ).transpose()
    assert A.is_lower() is True
    assert B.is_lower() is False

    # Wide matrices:
    A = DDM([
        [QQ(1), QQ(2), QQ(3)],
        [QQ(0), QQ(5), QQ(6)],
        [QQ(0), QQ(0), QQ(8)],
        [QQ(0), QQ(0), QQ(0)]
    ], (4, 3), QQ).transpose()
    B = DDM([
        [QQ(1), QQ(2), QQ(3)],
        [QQ(0), QQ(5), QQ(6)],
        [QQ(0), QQ(0), QQ(8)],
        [QQ(0), QQ(0), QQ(10)]
    ], (4, 3), QQ).transpose()
    assert A.is_lower() is True
    assert B.is_lower() is False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\tests\test_nseries.py ===
from sympy.calculus.util import AccumBounds
from sympy.core.function import (Derivative, PoleError)
from sympy.core.numbers import (E, I, Integer, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, asinh, atanh, cosh, coth, sinh, tanh)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, cot, sin, tan)
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.abc import x, y, z

from sympy.testing.pytest import raises, XFAIL


def test_simple_1():
    assert x.nseries(x, n=5) == x
    assert y.nseries(x, n=5) == y
    assert (1/(x*y)).nseries(y, n=5) == 1/(x*y)
    assert Rational(3, 4).nseries(x, n=5) == Rational(3, 4)
    assert x.nseries() == x


def test_mul_0():
    assert (x*log(x)).nseries(x, n=5) == x*log(x)


def test_mul_1():
    assert (x*log(2 + x)).nseries(x, n=5) == x*log(2) + x**2/2 - x**3/8 + \
        x**4/24 + O(x**5)
    assert (x*log(1 + x)).nseries(
        x, n=5) == x**2 - x**3/2 + x**4/3 + O(x**5)


def test_pow_0():
    assert (x**2).nseries(x, n=5) == x**2
    assert (1/x).nseries(x, n=5) == 1/x
    assert (1/x**2).nseries(x, n=5) == 1/x**2
    assert (x**Rational(2, 3)).nseries(x, n=5) == (x**Rational(2, 3))
    assert (sqrt(x)**3).nseries(x, n=5) == (sqrt(x)**3)


def test_pow_1():
    assert ((1 + x)**2).nseries(x, n=5) == x**2 + 2*x + 1

    # https://github.com/sympy/sympy/issues/21075
    assert ((sqrt(x) + 1)**2).nseries(x) == 2*sqrt(x) + x + 1
    assert ((sqrt(x) + cbrt(x))**2).nseries(x) == 2*x**Rational(5, 6)\
        + x**Rational(2, 3) + x


def test_geometric_1():
    assert (1/(1 - x)).nseries(x, n=5) == 1 + x + x**2 + x**3 + x**4 + O(x**5)
    assert (x/(1 - x)).nseries(x, n=6) == x + x**2 + x**3 + x**4 + x**5 + O(x**6)
    assert (x**3/(1 - x)).nseries(x, n=8) == x**3 + x**4 + x**5 + x**6 + \
        x**7 + O(x**8)


def test_sqrt_1():
    assert sqrt(1 + x).nseries(x, n=5) == 1 + x/2 - x**2/8 + x**3/16 - 5*x**4/128 + O(x**5)


def test_exp_1():
    assert exp(x).nseries(x, n=5) == 1 + x + x**2/2 + x**3/6 + x**4/24 + O(x**5)
    assert exp(x).nseries(x, n=12) == 1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 +  \
        x**6/720 + x**7/5040 + x**8/40320 + x**9/362880 + x**10/3628800 +  \
        x**11/39916800 + O(x**12)
    assert exp(1/x).nseries(x, n=5) == exp(1/x)
    assert exp(1/(1 + x)).nseries(x, n=4) ==  \
        (E*(1 - x - 13*x**3/6 + 3*x**2/2)).expand() + O(x**4)
    assert exp(2 + x).nseries(x, n=5) ==  \
        (exp(2)*(1 + x + x**2/2 + x**3/6 + x**4/24)).expand() + O(x**5)


def test_exp_sqrt_1():
    assert exp(1 + sqrt(x)).nseries(x, n=3) ==  \
        (exp(1)*(1 + sqrt(x) + x/2 + sqrt(x)*x/6)).expand() + O(sqrt(x)**3)


def test_power_x_x1():
    assert (exp(x*log(x))).nseries(x, n=4) == \
        1 + x*log(x) + x**2*log(x)**2/2 + x**3*log(x)**3/6 + O(x**4*log(x)**4)


def test_power_x_x2():
    assert (x**x).nseries(x, n=4) == \
        1 + x*log(x) + x**2*log(x)**2/2 + x**3*log(x)**3/6 + O(x**4*log(x)**4)


def test_log_singular1():
    assert log(1 + 1/x).nseries(x, n=5) == x - log(x) - x**2/2 + x**3/3 - \
        x**4/4 + O(x**5)


def test_log_power1():
    e = 1 / (1/x + x ** (log(3)/log(2)))
    assert e.nseries(x, n=5) == -x**(log(3)/log(2) + 2) + x + O(x**5)


def test_log_series():
    l = Symbol('l')
    e = 1/(1 - log(x))
    assert e.nseries(x, n=5, logx=l) == 1/(1 - l)


def test_log2():
    e = log(-1/x)
    assert e.nseries(x, n=5) == -log(x) + log(-1)


def test_log3():
    l = Symbol('l')
    e = 1/log(-1/x)
    assert e.nseries(x, n=4, logx=l) == 1/(-l + log(-1))


def test_series1():
    e = sin(x)
    assert e.nseries(x, 0, 0) != 0
    assert e.nseries(x, 0, 0) == O(1, x)
    assert e.nseries(x, 0, 1) == O(x, x)
    assert e.nseries(x, 0, 2) == x + O(x**2, x)
    assert e.nseries(x, 0, 3) == x + O(x**3, x)
    assert e.nseries(x, 0, 4) == x - x**3/6 + O(x**4, x)

    e = (exp(x) - 1)/x
    assert e.nseries(x, 0, 3) == 1 + x/2 + x**2/6 + O(x**3)

    assert x.nseries(x, 0, 2) == x


@XFAIL
def test_series1_failing():
    assert x.nseries(x, 0, 0) == O(1, x)
    assert x.nseries(x, 0, 1) == O(x, x)


def test_seriesbug1():
    assert (1/x).nseries(x, 0, 3) == 1/x
    assert (x + 1/x).nseries(x, 0, 3) == x + 1/x


def test_series2x():
    assert ((x + 1)**(-2)).nseries(x, 0, 4) == 1 - 2*x + 3*x**2 - 4*x**3 + O(x**4, x)
    assert ((x + 1)**(-1)).nseries(x, 0, 4) == 1 - x + x**2 - x**3 + O(x**4, x)
    assert ((x + 1)**0).nseries(x, 0, 3) == 1
    assert ((x + 1)**1).nseries(x, 0, 3) == 1 + x
    assert ((x + 1)**2).nseries(x, 0, 3) == x**2 + 2*x + 1
    assert ((x + 1)**3).nseries(x, 0, 3) == 1 + 3*x + 3*x**2 + O(x**3)

    assert (1/(1 + x)).nseries(x, 0, 4) == 1 - x + x**2 - x**3 + O(x**4, x)
    assert (x + 3/(1 + 2*x)).nseries(x, 0, 4) == 3 - 5*x + 12*x**2 - 24*x**3 + O(x**4, x)

    assert ((1/x + 1)**3).nseries(x, 0, 3) == 1 + 3/x + 3/x**2 + x**(-3)
    assert (1/(1 + 1/x)).nseries(x, 0, 4) == x - x**2 + x**3 - O(x**4, x)
    assert (1/(1 + 1/x**2)).nseries(x, 0, 6) == x**2 - x**4 + O(x**6, x)


def test_bug2():  # 1/log(0)*log(0) problem
    w = Symbol("w")
    e = (w**(-1) + w**(
        -log(3)*log(2)**(-1)))**(-1)*(3*w**(-log(3)*log(2)**(-1)) + 2*w**(-1))
    e = e.expand()
    assert e.nseries(w, 0, 4).subs(w, 0) == 3


def test_exp():
    e = (1 + x)**(1/x)
    assert e.nseries(x, n=3) == exp(1) - x*exp(1)/2 + 11*exp(1)*x**2/24 + O(x**3)


def test_exp2():
    w = Symbol("w")
    e = w**(1 - log(x)/(log(2) + log(x)))
    logw = Symbol("logw")
    assert e.nseries(
        w, 0, 1, logx=logw) == exp(logw*log(2)/(log(x) + log(2)))


def test_bug3():
    e = (2/x + 3/x**2)/(1/x + 1/x**2)
    assert e.nseries(x, n=3) == 3 - x + x**2 + O(x**3)


def test_generalexponent():
    p = 2
    e = (2/x + 3/x**p)/(1/x + 1/x**p)
    assert e.nseries(x, 0, 3) == 3 - x + x**2 + O(x**3)
    p = S.Half
    e = (2/x + 3/x**p)/(1/x + 1/x**p)
    assert e.nseries(x, 0, 2) == 2 - x + sqrt(x) + x**(S(3)/2) + O(x**2)

    e = 1 + sqrt(x)
    assert e.nseries(x, 0, 4) == 1 + sqrt(x)

# more complicated example


def test_genexp_x():
    e = 1/(1 + sqrt(x))
    assert e.nseries(x, 0, 2) == \
        1 + x - sqrt(x) - sqrt(x)**3 + O(x**2, x)

# more complicated example


def test_genexp_x2():
    p = Rational(3, 2)
    e = (2/x + 3/x**p)/(1/x + 1/x**p)
    assert e.nseries(x, 0, 3) == 3 + x + x**2 - sqrt(x) - x**(S(3)/2) - x**(S(5)/2) + O(x**3)


def test_seriesbug2():
    w = Symbol("w")
    #simple case (1):
    e = ((2*w)/w)**(1 + w)
    assert e.nseries(w, 0, 1) == 2 + O(w, w)
    assert e.nseries(w, 0, 1).subs(w, 0) == 2


def test_seriesbug2b():
    w = Symbol("w")
    #test sin
    e = sin(2*w)/w
    assert e.nseries(w, 0, 3) == 2 - 4*w**2/3 + O(w**3)


def test_seriesbug2d():
    w = Symbol("w", real=True)
    e = log(sin(2*w)/w)
    assert e.series(w, n=5) == log(2) - 2*w**2/3 - 4*w**4/45 + O(w**5)


def test_seriesbug2c():
    w = Symbol("w", real=True)
    #more complicated case, but sin(x)~x, so the result is the same as in (1)
    e = (sin(2*w)/w)**(1 + w)
    assert e.series(w, 0, 1) == 2 + O(w)
    assert e.series(w, 0, 3) == 2 + 2*w*log(2) + \
        w**2*(Rational(-4, 3) + log(2)**2) + O(w**3)
    assert e.series(w, 0, 2).subs(w, 0) == 2


def test_expbug4():
    x = Symbol("x", real=True)
    assert (log(
        sin(2*x)/x)*(1 + x)).series(x, 0, 2) == log(2) + x*log(2) + O(x**2, x)
    assert exp(
        log(sin(2*x)/x)*(1 + x)).series(x, 0, 2) == 2 + 2*x*log(2) + O(x**2)

    assert exp(log(2) + O(x)).nseries(x, 0, 2) == 2 + O(x)
    assert ((2 + O(x))**(1 + x)).nseries(x, 0, 2) == 2 + O(x)


def test_logbug4():
    assert log(2 + O(x)).nseries(x, 0, 2) == log(2) + O(x, x)


def test_expbug5():
    assert exp(log(1 + x)/x).nseries(x, n=3) == exp(1) + -exp(1)*x/2 + 11*exp(1)*x**2/24 + O(x**3)

    assert exp(O(x)).nseries(x, 0, 2) == 1 + O(x)


def test_sinsinbug():
    assert sin(sin(x)).nseries(x, 0, 8) == x - x**3/3 + x**5/10 - 8*x**7/315 + O(x**8)


def test_issue_3258():
    a = x/(exp(x) - 1)
    assert a.nseries(x, 0, 5) == 1 - x/2 - x**4/720 + x**2/12 + O(x**5)


def test_issue_3204():
    x = Symbol("x", nonnegative=True)
    f = sin(x**3)**Rational(1, 3)
    assert f.nseries(x, 0, 17) == x - x**7/18 - x**13/3240 + O(x**17)


def test_issue_3224():
    f = sqrt(1 - sqrt(y))
    assert f.nseries(y, 0, 2) == 1 - sqrt(y)/2 - y/8 - sqrt(y)**3/16 + O(y**2)


def test_issue_3463():
    w, i = symbols('w,i')
    r = log(5)/log(3)
    p = w**(-1 + r)
    e = 1/x*(-log(w**(1 + r)) + log(w + w**r))
    e_ser = -r*log(w)/x + p/x - p**2/(2*x) + O(w)
    assert e.nseries(w, n=1) == e_ser


def test_sin():
    assert sin(8*x).nseries(x, n=4) == 8*x - 256*x**3/3 + O(x**4)
    assert sin(x + y).nseries(x, n=1) == sin(y) + O(x)
    assert sin(x + y).nseries(x, n=2) == sin(y) + cos(y)*x + O(x**2)
    assert sin(x + y).nseries(x, n=5) == sin(y) + cos(y)*x - sin(y)*x**2/2 - \
        cos(y)*x**3/6 + sin(y)*x**4/24 + O(x**5)


def test_issue_3515():
    e = sin(8*x)/x
    assert e.nseries(x, n=6) == 8 - 256*x**2/3 + 4096*x**4/15 + O(x**6)


def test_issue_3505():
    e = sin(x)**(-4)*(sqrt(cos(x))*sin(x)**2 -
        cos(x)**Rational(1, 3)*sin(x)**2)
    assert e.nseries(x, n=9) == Rational(-1, 12) - 7*x**2/288 - \
        43*x**4/10368 - 1123*x**6/2488320 + 377*x**8/29859840 + O(x**9)


def test_issue_3501():
    a = Symbol("a")
    e = x**(-2)*(x*sin(a + x) - x*sin(a))
    assert e.nseries(x, n=6) == cos(a) - sin(a)*x/2 - cos(a)*x**2/6 + \
        x**3*sin(a)/24 + x**4*cos(a)/120 - x**5*sin(a)/720 + O(x**6)
    e = x**(-2)*(x*cos(a + x) - x*cos(a))
    assert e.nseries(x, n=6) == -sin(a) - cos(a)*x/2 + sin(a)*x**2/6 + \
        cos(a)*x**3/24 - x**4*sin(a)/120 - x**5*cos(a)/720 + O(x**6)


def test_issue_3502():
    e = sin(5*x)/sin(2*x)
    assert e.nseries(x, n=2) == Rational(5, 2) + O(x**2)
    assert e.nseries(x, n=6) == \
        Rational(5, 2) - 35*x**2/4 + 329*x**4/48 + O(x**6)


def test_issue_3503():
    e = sin(2 + x)/(2 + x)
    assert e.nseries(x, n=2) == sin(2)/2 + x*cos(2)/2 - x*sin(2)/4 + O(x**2)


def test_issue_3506():
    e = (x + sin(3*x))**(-2)*(x*(x + sin(3*x)) - (x + sin(3*x))*sin(2*x))
    assert e.nseries(x, n=7) == \
        Rational(-1, 4) + 5*x**2/96 + 91*x**4/768 + 11117*x**6/129024 + O(x**7)


def test_issue_3508():
    x = Symbol("x", real=True)
    assert log(sin(x)).series(x, n=5) == log(x) - x**2/6 - x**4/180 + O(x**5)
    e = -log(x) + x*(-log(x) + log(sin(2*x))) + log(sin(2*x))
    assert e.series(x, n=5) == \
        log(2) + log(2)*x - 2*x**2/3 - 2*x**3/3 - 4*x**4/45 + O(x**5)


def test_issue_3507():
    e = x**(-4)*(x**2 - x**2*sqrt(cos(x)))
    assert e.nseries(x, n=9) == \
        Rational(1, 4) + x**2/96 + 19*x**4/5760 + 559*x**6/645120 + 29161*x**8/116121600 + O(x**9)


def test_issue_3639():
    assert sin(cos(x)).nseries(x, n=5) == \
        sin(1) - x**2*cos(1)/2 - x**4*sin(1)/8 + x**4*cos(1)/24 + O(x**5)


def test_hyperbolic():
    assert sinh(x).nseries(x, n=6) == x + x**3/6 + x**5/120 + O(x**6)
    assert cosh(x).nseries(x, n=5) == 1 + x**2/2 + x**4/24 + O(x**5)
    assert tanh(x).nseries(x, n=6) == x - x**3/3 + 2*x**5/15 + O(x**6)
    assert coth(x).nseries(x, n=6) == \
        1/x - x**3/45 + x/3 + 2*x**5/945 + O(x**6)
    assert asinh(x).nseries(x, n=6) == x - x**3/6 + 3*x**5/40 + O(x**6)
    assert acosh(x).nseries(x, n=6) == \
        pi*I/2 - I*x - 3*I*x**5/40 - I*x**3/6 + O(x**6)
    assert atanh(x).nseries(x, n=6) == x + x**3/3 + x**5/5 + O(x**6)
    assert acoth(x).nseries(x, n=6) == -I*pi/2 + x + x**3/3 + x**5/5 + O(x**6)


def test_series2():
    w = Symbol("w", real=True)
    x = Symbol("x", real=True)
    e = w**(-2)*(w*exp(1/x - w) - w*exp(1/x))
    assert e.nseries(w, n=4) == -exp(1/x) + w*exp(1/x)/2 - w**2*exp(1/x)/6 + w**3*exp(1/x)/24 + O(w**4)


def test_series3():
    w = Symbol("w", real=True)
    e = w**(-6)*(w**3*tan(w) - w**3*sin(w))
    assert e.nseries(w, n=8) == Integer(1)/2 + w**2/8 + 13*w**4/240 + 529*w**6/24192 + O(w**8)


def test_bug4():
    w = Symbol("w")
    e = x/(w**4 + x**2*w**4 + 2*x*w**4)*w**4
    assert e.nseries(w, n=2).removeO().expand() in [x/(1 + 2*x + x**2),
        1/(1 + x/2 + 1/x/2)/2, 1/x/(1 + 2/x + x**(-2))]


def test_bug5():
    w = Symbol("w")
    l = Symbol('l')
    e = (-log(w) + log(1 + w*log(x)))**(-2)*w**(-2)*((-log(w) +
        log(1 + x*w))*(-log(w) + log(1 + w*log(x)))*w - x*(-log(w) +
        log(1 + w*log(x)))*w)
    assert e.nseries(w, n=0, logx=l) == x/w/l + 1/w + O(1, w)
    assert e.nseries(w, n=1, logx=l) == x/w/l + 1/w - x/l + 1/l*log(x) \
        + x*log(x)/l**2 + O(w)


def test_issue_4115():
    assert (sin(x)/(1 - cos(x))).nseries(x, n=1) == 2/x + O(x)
    assert (sin(x)**2/(1 - cos(x))).nseries(x, n=1) == 2 + O(x)


def test_pole():
    raises(PoleError, lambda: sin(1/x).series(x, 0, 5))
    raises(PoleError, lambda: sin(1 + 1/x).series(x, 0, 5))
    raises(PoleError, lambda: (x*sin(1/x)).series(x, 0, 5))


def test_expsinbug():
    assert exp(sin(x)).series(x, 0, 0) == O(1, x)
    assert exp(sin(x)).series(x, 0, 1) == 1 + O(x)
    assert exp(sin(x)).series(x, 0, 2) == 1 + x + O(x**2)
    assert exp(sin(x)).series(x, 0, 3) == 1 + x + x**2/2 + O(x**3)
    assert exp(sin(x)).series(x, 0, 4) == 1 + x + x**2/2 + O(x**4)
    assert exp(sin(x)).series(x, 0, 5) == 1 + x + x**2/2 - x**4/8 + O(x**5)


def test_floor():
    x = Symbol('x')
    assert floor(x).series(x) == 0
    assert floor(-x).series(x) == -1
    assert floor(sin(x)).series(x) == 0
    assert floor(sin(-x)).series(x) == -1
    assert floor(x**3).series(x) == 0
    assert floor(-x**3).series(x) == -1
    assert floor(cos(x)).series(x) == 0
    assert floor(cos(-x)).series(x) == 0
    assert floor(5 + sin(x)).series(x) == 5
    assert floor(5 + sin(-x)).series(x) == 4

    assert floor(x).series(x, 2) == 2
    assert floor(-x).series(x, 2) == -3

    x = Symbol('x', negative=True)
    assert floor(x + 1.5).series(x) == 1


def test_frac():
    assert frac(x).series(x, cdir=1) == x
    assert frac(x).series(x, cdir=-1) == 1 + x
    assert frac(2*x + 1).series(x, cdir=1) == 2*x
    assert frac(2*x + 1).series(x, cdir=-1) == 1 + 2*x
    assert frac(x**2).series(x, cdir=1) == x**2
    assert frac(x**2).series(x, cdir=-1) == x**2
    assert frac(sin(x) + 5).series(x, cdir=1) == x - x**3/6 + x**5/120 + O(x**6)
    assert frac(sin(x) + 5).series(x, cdir=-1) == 1 + x - x**3/6 + x**5/120 + O(x**6)
    assert frac(sin(x) + S.Half).series(x) == S.Half + x - x**3/6 + x**5/120 + O(x**6)
    assert frac(x**8).series(x, cdir=1) == O(x**6)
    assert frac(1/x).series(x) == AccumBounds(0, 1) + O(x**6)


def test_ceiling():
    assert ceiling(x).series(x) == 1
    assert ceiling(-x).series(x) == 0
    assert ceiling(sin(x)).series(x) == 1
    assert ceiling(sin(-x)).series(x) == 0
    assert ceiling(1 - cos(x)).series(x) == 1
    assert ceiling(1 - cos(-x)).series(x) == 1
    assert ceiling(x).series(x, 2) == 3
    assert ceiling(-x).series(x, 2) == -2


def test_abs():
    a = Symbol('a')
    assert abs(x).nseries(x, n=4) == x
    assert abs(-x).nseries(x, n=4) == x
    assert abs(x + 1).nseries(x, n=4) == x + 1
    assert abs(sin(x)).nseries(x, n=4) == x - Rational(1, 6)*x**3 + O(x**4)
    assert abs(sin(-x)).nseries(x, n=4) == x - Rational(1, 6)*x**3 + O(x**4)
    assert abs(x - a).nseries(x, 1) == -a*sign(1 - a) + (x - 1)*sign(1 - a) + sign(1 - a)


def test_dir():
    assert abs(x).series(x, 0, dir="+") == x
    assert abs(x).series(x, 0, dir="-") == -x
    assert floor(x + 2).series(x, 0, dir='+') == 2
    assert floor(x + 2).series(x, 0, dir='-') == 1
    assert floor(x + 2.2).series(x, 0, dir='-') == 2
    assert ceiling(x + 2.2).series(x, 0, dir='-') == 3
    assert sin(x + y).series(x, 0, dir='-') == sin(x + y).series(x, 0, dir='+')


def test_cdir():
    assert abs(x).series(x, 0, cdir=1) == x
    assert abs(x).series(x, 0, cdir=-1) == -x
    assert floor(x + 2).series(x, 0, cdir=1) == 2
    assert floor(x + 2).series(x, 0, cdir=-1) == 1
    assert floor(x + 2.2).series(x, 0, cdir=1) == 2
    assert ceiling(x + 2.2).series(x, 0, cdir=-1) == 3
    assert sin(x + y).series(x, 0, cdir=-1) == sin(x + y).series(x, 0, cdir=1)


def test_issue_3504():
    a = Symbol("a")
    e = asin(a*x)/x
    assert e.series(x, 4, n=2).removeO() == \
        (x - 4)*(a/(4*sqrt(-16*a**2 + 1)) - asin(4*a)/16) + asin(4*a)/4


def test_issue_4441():
    a, b = symbols('a,b')
    f = 1/(1 + a*x)
    assert f.series(x, 0, 5) == 1 - a*x + a**2*x**2 - a**3*x**3 + \
        a**4*x**4 + O(x**5)
    f = 1/(1 + (a + b)*x)
    assert f.series(x, 0, 3) == 1 + x*(-a - b)\
        + x**2*(a + b)**2 + O(x**3)


def test_issue_4329():
    assert tan(x).series(x, pi/2, n=3).removeO() == \
        -pi/6 + x/3 - 1/(x - pi/2)
    assert cot(x).series(x, pi, n=3).removeO() == \
        -x/3 + pi/3 + 1/(x - pi)
    assert limit(tan(x)**tan(2*x), x, pi/4) == exp(-1)


def test_issue_5183():
    assert abs(x + x**2).series(n=1) == O(x)
    assert abs(x + x**2).series(n=2) == x + O(x**2)
    assert ((1 + x)**2).series(x, n=6) == x**2 + 2*x + 1
    assert (1 + 1/x).series() == 1 + 1/x
    assert Derivative(exp(x).series(), x).doit() == \
        1 + x + x**2/2 + x**3/6 + x**4/24 + O(x**5)


def test_issue_5654():
    a = Symbol('a')
    assert (1/(x**2+a**2)**2).nseries(x, x0=I*a, n=0) == \
        -I/(4*a**3*(-I*a + x)) - 1/(4*a**2*(-I*a + x)**2) + O(1, (x, I*a))
    assert (1/(x**2+a**2)**2).nseries(x, x0=I*a, n=1) == 3/(16*a**4) \
        -I/(4*a**3*(-I*a + x)) - 1/(4*a**2*(-I*a + x)**2) + O(-I*a + x, (x, I*a))


def test_issue_5925():
    sx = sqrt(x + z).series(z, 0, 1)
    sxy = sqrt(x + y + z).series(z, 0, 1)
    s1, s2 = sx.subs(x, x + y), sxy
    assert (s1 - s2).expand().removeO().simplify() == 0

    sx = sqrt(x + z).series(z, 0, 1)
    sxy = sqrt(x + y + z).series(z, 0, 1)
    assert sxy.subs({x:1, y:2}) == sx.subs(x, 3)


def test_exp_2():
    assert exp(x**3).nseries(x, 0, 14) == 1 + x**3 + x**6/2 + x**9/6 + x**12/24 + O(x**14)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_combine_first.py ===
from datetime import datetime

import numpy as np
import pytest

from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm


class TestDataFrameCombineFirst:
    def test_combine_first_mixed(self):
        a = Series(["a", "b"], index=range(2))
        b = Series(range(2), index=range(2))
        f = DataFrame({"A": a, "B": b})

        a = Series(["a", "b"], index=range(5, 7))
        b = Series(range(2), index=range(5, 7))
        g = DataFrame({"A": a, "B": b})

        exp = DataFrame({"A": list("abab"), "B": [0, 1, 0, 1]}, index=[0, 1, 5, 6])
        combined = f.combine_first(g)
        tm.assert_frame_equal(combined, exp)

    def test_combine_first(self, float_frame, using_infer_string):
        # disjoint
        head, tail = float_frame[:5], float_frame[5:]

        combined = head.combine_first(tail)
        reordered_frame = float_frame.reindex(combined.index)
        tm.assert_frame_equal(combined, reordered_frame)
        tm.assert_index_equal(combined.columns, float_frame.columns)
        tm.assert_series_equal(combined["A"], reordered_frame["A"])

        # same index
        fcopy = float_frame.copy()
        fcopy["A"] = 1
        del fcopy["C"]

        fcopy2 = float_frame.copy()
        fcopy2["B"] = 0
        del fcopy2["D"]

        combined = fcopy.combine_first(fcopy2)

        assert (combined["A"] == 1).all()
        tm.assert_series_equal(combined["B"], fcopy["B"])
        tm.assert_series_equal(combined["C"], fcopy2["C"])
        tm.assert_series_equal(combined["D"], fcopy["D"])

        # overlap
        head, tail = reordered_frame[:10].copy(), reordered_frame
        head["A"] = 1

        combined = head.combine_first(tail)
        assert (combined["A"][:10] == 1).all()

        # reverse overlap
        tail.iloc[:10, tail.columns.get_loc("A")] = 0
        combined = tail.combine_first(head)
        assert (combined["A"][:10] == 0).all()

        # no overlap
        f = float_frame[:10]
        g = float_frame[10:]
        combined = f.combine_first(g)
        tm.assert_series_equal(combined["A"].reindex(f.index), f["A"])
        tm.assert_series_equal(combined["A"].reindex(g.index), g["A"])

        # corner cases
        warning = FutureWarning if using_infer_string else None
        with tm.assert_produces_warning(warning, match="empty entries"):
            comb = float_frame.combine_first(DataFrame())
        tm.assert_frame_equal(comb, float_frame)

        comb = DataFrame().combine_first(float_frame)
        tm.assert_frame_equal(comb, float_frame.sort_index())

        comb = float_frame.combine_first(DataFrame(index=["faz", "boo"]))
        assert "faz" in comb.index

        # #2525
        df = DataFrame({"a": [1]}, index=[datetime(2012, 1, 1)])
        df2 = DataFrame(columns=["b"])
        result = df.combine_first(df2)
        assert "b" in result

    def test_combine_first_mixed_bug(self):
        idx = Index(["a", "b", "c", "e"])
        ser1 = Series([5.0, -9.0, 4.0, 100.0], index=idx)
        ser2 = Series(["a", "b", "c", "e"], index=idx)
        ser3 = Series([12, 4, 5, 97], index=idx)

        frame1 = DataFrame({"col0": ser1, "col2": ser2, "col3": ser3})

        idx = Index(["a", "b", "c", "f"])
        ser1 = Series([5.0, -9.0, 4.0, 100.0], index=idx)
        ser2 = Series(["a", "b", "c", "f"], index=idx)
        ser3 = Series([12, 4, 5, 97], index=idx)

        frame2 = DataFrame({"col1": ser1, "col2": ser2, "col5": ser3})

        combined = frame1.combine_first(frame2)
        assert len(combined.columns) == 5

    def test_combine_first_same_as_in_update(self):
        # gh 3016 (same as in update)
        df = DataFrame(
            [[1.0, 2.0, False, True], [4.0, 5.0, True, False]],
            columns=["A", "B", "bool1", "bool2"],
        )

        other = DataFrame([[45, 45]], index=[0], columns=["A", "B"])
        result = df.combine_first(other)
        tm.assert_frame_equal(result, df)

        df.loc[0, "A"] = np.nan
        result = df.combine_first(other)
        df.loc[0, "A"] = 45
        tm.assert_frame_equal(result, df)

    def test_combine_first_doc_example(self):
        # doc example
        df1 = DataFrame(
            {"A": [1.0, np.nan, 3.0, 5.0, np.nan], "B": [np.nan, 2.0, 3.0, np.nan, 6.0]}
        )

        df2 = DataFrame(
            {
                "A": [5.0, 2.0, 4.0, np.nan, 3.0, 7.0],
                "B": [np.nan, np.nan, 3.0, 4.0, 6.0, 8.0],
            }
        )

        result = df1.combine_first(df2)
        expected = DataFrame({"A": [1, 2, 3, 5, 3, 7.0], "B": [np.nan, 2, 3, 4, 6, 8]})
        tm.assert_frame_equal(result, expected)

    def test_combine_first_return_obj_type_with_bools(self):
        # GH3552

        df1 = DataFrame(
            [[np.nan, 3.0, True], [-4.6, np.nan, True], [np.nan, 7.0, False]]
        )
        df2 = DataFrame([[-42.6, np.nan, True], [-5.0, 1.6, False]], index=[1, 2])

        expected = Series([True, True, False], name=2, dtype=bool)

        result_12 = df1.combine_first(df2)[2]
        tm.assert_series_equal(result_12, expected)

        result_21 = df2.combine_first(df1)[2]
        tm.assert_series_equal(result_21, expected)

    @pytest.mark.parametrize(
        "data1, data2, data_expected",
        (
            (
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
                [pd.NaT, pd.NaT, pd.NaT],
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
            ),
            (
                [pd.NaT, pd.NaT, pd.NaT],
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
            ),
            (
                [datetime(2000, 1, 2), pd.NaT, pd.NaT],
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
                [datetime(2000, 1, 2), datetime(2000, 1, 2), datetime(2000, 1, 3)],
            ),
            (
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
                [datetime(2000, 1, 2), pd.NaT, pd.NaT],
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
            ),
        ),
    )
    def test_combine_first_convert_datatime_correctly(
        self, data1, data2, data_expected
    ):
        # GH 3593

        df1, df2 = DataFrame({"a": data1}), DataFrame({"a": data2})
        result = df1.combine_first(df2)
        expected = DataFrame({"a": data_expected})
        tm.assert_frame_equal(result, expected)

    def test_combine_first_align_nan(self):
        # GH 7509 (not fixed)
        dfa = DataFrame([[pd.Timestamp("2011-01-01"), 2]], columns=["a", "b"])
        dfb = DataFrame([[4], [5]], columns=["b"])
        assert dfa["a"].dtype == "datetime64[ns]"
        assert dfa["b"].dtype == "int64"

        res = dfa.combine_first(dfb)
        exp = DataFrame(
            {"a": [pd.Timestamp("2011-01-01"), pd.NaT], "b": [2, 5]},
            columns=["a", "b"],
        )
        tm.assert_frame_equal(res, exp)
        assert res["a"].dtype == "datetime64[ns]"
        # TODO: this must be int64
        assert res["b"].dtype == "int64"

        res = dfa.iloc[:0].combine_first(dfb)
        exp = DataFrame({"a": [np.nan, np.nan], "b": [4, 5]}, columns=["a", "b"])
        tm.assert_frame_equal(res, exp)
        # TODO: this must be datetime64
        assert res["a"].dtype == "float64"
        # TODO: this must be int64
        assert res["b"].dtype == "int64"

    def test_combine_first_timezone(self, unit):
        # see gh-7630
        data1 = pd.to_datetime("20100101 01:01").tz_localize("UTC").as_unit(unit)
        df1 = DataFrame(
            columns=["UTCdatetime", "abc"],
            data=data1,
            index=pd.date_range("20140627", periods=1),
        )
        data2 = pd.to_datetime("20121212 12:12").tz_localize("UTC").as_unit(unit)
        df2 = DataFrame(
            columns=["UTCdatetime", "xyz"],
            data=data2,
            index=pd.date_range("20140628", periods=1),
        )
        res = df2[["UTCdatetime"]].combine_first(df1)
        exp = DataFrame(
            {
                "UTCdatetime": [
                    pd.Timestamp("2010-01-01 01:01", tz="UTC"),
                    pd.Timestamp("2012-12-12 12:12", tz="UTC"),
                ],
                "abc": [pd.Timestamp("2010-01-01 01:01:00", tz="UTC"), pd.NaT],
            },
            columns=["UTCdatetime", "abc"],
            index=pd.date_range("20140627", periods=2, freq="D"),
            dtype=f"datetime64[{unit}, UTC]",
        )
        assert res["UTCdatetime"].dtype == f"datetime64[{unit}, UTC]"
        assert res["abc"].dtype == f"datetime64[{unit}, UTC]"

        tm.assert_frame_equal(res, exp)

    def test_combine_first_timezone2(self, unit):
        # see gh-10567
        dts1 = pd.date_range("2015-01-01", "2015-01-05", tz="UTC", unit=unit)
        df1 = DataFrame({"DATE": dts1})
        dts2 = pd.date_range("2015-01-03", "2015-01-05", tz="UTC", unit=unit)
        df2 = DataFrame({"DATE": dts2})

        res = df1.combine_first(df2)
        tm.assert_frame_equal(res, df1)
        assert res["DATE"].dtype == f"datetime64[{unit}, UTC]"

    def test_combine_first_timezone3(self, unit):
        dts1 = pd.DatetimeIndex(
            ["2011-01-01", "NaT", "2011-01-03", "2011-01-04"], tz="US/Eastern"
        ).as_unit(unit)
        df1 = DataFrame({"DATE": dts1}, index=[1, 3, 5, 7])
        dts2 = pd.DatetimeIndex(
            ["2012-01-01", "2012-01-02", "2012-01-03"], tz="US/Eastern"
        ).as_unit(unit)
        df2 = DataFrame({"DATE": dts2}, index=[2, 4, 5])

        res = df1.combine_first(df2)
        exp_dts = pd.DatetimeIndex(
            [
                "2011-01-01",
                "2012-01-01",
                "NaT",
                "2012-01-02",
                "2011-01-03",
                "2011-01-04",
            ],
            tz="US/Eastern",
        ).as_unit(unit)
        exp = DataFrame({"DATE": exp_dts}, index=[1, 2, 3, 4, 5, 7])
        tm.assert_frame_equal(res, exp)

    # FIXME: parametrizing over unit breaks on non-nano
    def test_combine_first_timezone4(self):
        # different tz
        dts1 = pd.date_range("2015-01-01", "2015-01-05", tz="US/Eastern")
        df1 = DataFrame({"DATE": dts1})
        dts2 = pd.date_range("2015-01-03", "2015-01-05")
        df2 = DataFrame({"DATE": dts2})

        # if df1 doesn't have NaN, keep its dtype
        res = df1.combine_first(df2)
        tm.assert_frame_equal(res, df1)
        assert res["DATE"].dtype == "datetime64[ns, US/Eastern]"

    def test_combine_first_timezone5(self, unit):
        dts1 = pd.date_range("2015-01-01", "2015-01-02", tz="US/Eastern", unit=unit)
        df1 = DataFrame({"DATE": dts1})
        dts2 = pd.date_range("2015-01-01", "2015-01-03", unit=unit)
        df2 = DataFrame({"DATE": dts2})

        res = df1.combine_first(df2)
        exp_dts = [
            pd.Timestamp("2015-01-01", tz="US/Eastern"),
            pd.Timestamp("2015-01-02", tz="US/Eastern"),
            pd.Timestamp("2015-01-03"),
        ]
        exp = DataFrame({"DATE": exp_dts})
        tm.assert_frame_equal(res, exp)
        assert res["DATE"].dtype == "object"

    def test_combine_first_timedelta(self):
        data1 = pd.TimedeltaIndex(["1 day", "NaT", "3 day", "4day"])
        df1 = DataFrame({"TD": data1}, index=[1, 3, 5, 7])
        data2 = pd.TimedeltaIndex(["10 day", "11 day", "12 day"])
        df2 = DataFrame({"TD": data2}, index=[2, 4, 5])

        res = df1.combine_first(df2)
        exp_dts = pd.TimedeltaIndex(
            ["1 day", "10 day", "NaT", "11 day", "3 day", "4 day"]
        )
        exp = DataFrame({"TD": exp_dts}, index=[1, 2, 3, 4, 5, 7])
        tm.assert_frame_equal(res, exp)
        assert res["TD"].dtype == "timedelta64[ns]"

    def test_combine_first_period(self):
        data1 = pd.PeriodIndex(["2011-01", "NaT", "2011-03", "2011-04"], freq="M")
        df1 = DataFrame({"P": data1}, index=[1, 3, 5, 7])
        data2 = pd.PeriodIndex(["2012-01-01", "2012-02", "2012-03"], freq="M")
        df2 = DataFrame({"P": data2}, index=[2, 4, 5])

        res = df1.combine_first(df2)
        exp_dts = pd.PeriodIndex(
            ["2011-01", "2012-01", "NaT", "2012-02", "2011-03", "2011-04"], freq="M"
        )
        exp = DataFrame({"P": exp_dts}, index=[1, 2, 3, 4, 5, 7])
        tm.assert_frame_equal(res, exp)
        assert res["P"].dtype == data1.dtype

        # different freq
        dts2 = pd.PeriodIndex(["2012-01-01", "2012-01-02", "2012-01-03"], freq="D")
        df2 = DataFrame({"P": dts2}, index=[2, 4, 5])

        res = df1.combine_first(df2)
        exp_dts = [
            pd.Period("2011-01", freq="M"),
            pd.Period("2012-01-01", freq="D"),
            pd.NaT,
            pd.Period("2012-01-02", freq="D"),
            pd.Period("2011-03", freq="M"),
            pd.Period("2011-04", freq="M"),
        ]
        exp = DataFrame({"P": exp_dts}, index=[1, 2, 3, 4, 5, 7])
        tm.assert_frame_equal(res, exp)
        assert res["P"].dtype == "object"

    def test_combine_first_int(self):
        # GH14687 - integer series that do no align exactly

        df1 = DataFrame({"a": [0, 1, 3, 5]}, dtype="int64")
        df2 = DataFrame({"a": [1, 4]}, dtype="int64")

        result_12 = df1.combine_first(df2)
        expected_12 = DataFrame({"a": [0, 1, 3, 5]})
        tm.assert_frame_equal(result_12, expected_12)

        result_21 = df2.combine_first(df1)
        expected_21 = DataFrame({"a": [1, 4, 3, 5]})
        tm.assert_frame_equal(result_21, expected_21)

    @pytest.mark.parametrize("val", [1, 1.0])
    def test_combine_first_with_asymmetric_other(self, val):
        # see gh-20699
        df1 = DataFrame({"isNum": [val]})
        df2 = DataFrame({"isBool": [True]})

        res = df1.combine_first(df2)
        exp = DataFrame({"isBool": [True], "isNum": [val]})

        tm.assert_frame_equal(res, exp)

    def test_combine_first_string_dtype_only_na(self, nullable_string_dtype):
        # GH: 37519
        df = DataFrame(
            {"a": ["962", "85"], "b": [pd.NA] * 2}, dtype=nullable_string_dtype
        )
        df2 = DataFrame({"a": ["85"], "b": [pd.NA]}, dtype=nullable_string_dtype)
        df.set_index(["a", "b"], inplace=True)
        df2.set_index(["a", "b"], inplace=True)
        result = df.combine_first(df2)
        expected = DataFrame(
            {"a": ["962", "85"], "b": [pd.NA] * 2}, dtype=nullable_string_dtype
        ).set_index(["a", "b"])
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "scalar1, scalar2",
    [
        (datetime(2020, 1, 1), datetime(2020, 1, 2)),
        (pd.Period("2020-01-01", "D"), pd.Period("2020-01-02", "D")),
        (pd.Timedelta("89 days"), pd.Timedelta("60 min")),
        (pd.Interval(left=0, right=1), pd.Interval(left=2, right=3, closed="left")),
    ],
)
def test_combine_first_timestamp_bug(scalar1, scalar2, nulls_fixture):
    # GH28481
    na_value = nulls_fixture

    frame = DataFrame([[na_value, na_value]], columns=["a", "b"])
    other = DataFrame([[scalar1, scalar2]], columns=["b", "c"])

    common_dtype = find_common_type([frame.dtypes["b"], other.dtypes["b"]])

    if is_dtype_equal(common_dtype, "object") or frame.dtypes["b"] == other.dtypes["b"]:
        val = scalar1
    else:
        val = na_value

    result = frame.combine_first(other)

    expected = DataFrame([[na_value, val, scalar2]], columns=["a", "b", "c"])

    expected["b"] = expected["b"].astype(common_dtype)

    tm.assert_frame_equal(result, expected)


def test_combine_first_timestamp_bug_NaT():
    # GH28481
    frame = DataFrame([[pd.NaT, pd.NaT]], columns=["a", "b"])
    other = DataFrame(
        [[datetime(2020, 1, 1), datetime(2020, 1, 2)]], columns=["b", "c"]
    )

    result = frame.combine_first(other)
    expected = DataFrame(
        [[pd.NaT, datetime(2020, 1, 1), datetime(2020, 1, 2)]], columns=["a", "b", "c"]
    )

    tm.assert_frame_equal(result, expected)


def test_combine_first_with_nan_multiindex():
    # gh-36562

    mi1 = MultiIndex.from_arrays(
        [["b", "b", "c", "a", "b", np.nan], [1, 2, 3, 4, 5, 6]], names=["a", "b"]
    )
    df = DataFrame({"c": [1, 1, 1, 1, 1, 1]}, index=mi1)
    mi2 = MultiIndex.from_arrays(
        [["a", "b", "c", "a", "b", "d"], [1, 1, 1, 1, 1, 1]], names=["a", "b"]
    )
    s = Series([1, 2, 3, 4, 5, 6], index=mi2)
    res = df.combine_first(DataFrame({"d": s}))
    mi_expected = MultiIndex.from_arrays(
        [
            ["a", "a", "a", "b", "b", "b", "b", "c", "c", "d", np.nan],
            [1, 1, 4, 1, 1, 2, 5, 1, 3, 1, 6],
        ],
        names=["a", "b"],
    )
    expected = DataFrame(
        {
            "c": [np.nan, np.nan, 1, 1, 1, 1, 1, np.nan, 1, np.nan, 1],
            "d": [1.0, 4.0, np.nan, 2.0, 5.0, np.nan, np.nan, 3.0, np.nan, 6.0, np.nan],
        },
        index=mi_expected,
    )
    tm.assert_frame_equal(res, expected)


def test_combine_preserve_dtypes():
    # GH7509
    a_column = Series(["a", "b"], index=range(2))
    b_column = Series(range(2), index=range(2))
    df1 = DataFrame({"A": a_column, "B": b_column})

    c_column = Series(["a", "b"], index=range(5, 7))
    b_column = Series(range(-1, 1), index=range(5, 7))
    df2 = DataFrame({"B": b_column, "C": c_column})

    expected = DataFrame(
        {
            "A": ["a", "b", np.nan, np.nan],
            "B": [0, 1, -1, 0],
            "C": [np.nan, np.nan, "a", "b"],
        },
        index=[0, 1, 5, 6],
    )
    combined = df1.combine_first(df2)
    tm.assert_frame_equal(combined, expected)


def test_combine_first_duplicates_rows_for_nan_index_values():
    # GH39881
    df1 = DataFrame(
        {"x": [9, 10, 11]},
        index=MultiIndex.from_arrays([[1, 2, 3], [np.nan, 5, 6]], names=["a", "b"]),
    )

    df2 = DataFrame(
        {"y": [12, 13, 14]},
        index=MultiIndex.from_arrays([[1, 2, 4], [np.nan, 5, 7]], names=["a", "b"]),
    )

    expected = DataFrame(
        {
            "x": [9.0, 10.0, 11.0, np.nan],
            "y": [12.0, 13.0, np.nan, 14.0],
        },
        index=MultiIndex.from_arrays(
            [[1, 2, 3, 4], [np.nan, 5, 6, 7]], names=["a", "b"]
        ),
    )
    combined = df1.combine_first(df2)
    tm.assert_frame_equal(combined, expected)


def test_combine_first_int64_not_cast_to_float64():
    # GH 28613
    df_1 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df_2 = DataFrame({"A": [1, 20, 30], "B": [40, 50, 60], "C": [12, 34, 65]})
    result = df_1.combine_first(df_2)
    expected = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [12, 34, 65]})
    tm.assert_frame_equal(result, expected)


def test_midx_losing_dtype():
    # GH#49830
    midx = MultiIndex.from_arrays([[0, 0], [np.nan, np.nan]])
    midx2 = MultiIndex.from_arrays([[1, 1], [np.nan, np.nan]])
    df1 = DataFrame({"a": [None, 4]}, index=midx)
    df2 = DataFrame({"a": [3, 3]}, index=midx2)
    result = df1.combine_first(df2)
    expected_midx = MultiIndex.from_arrays(
        [[0, 0, 1, 1], [np.nan, np.nan, np.nan, np.nan]]
    )
    expected = DataFrame({"a": [np.nan, 4, 3, 3]}, index=expected_midx)
    tm.assert_frame_equal(result, expected)


def test_combine_first_empty_columns():
    left = DataFrame(columns=["a", "b"])
    right = DataFrame(columns=["a", "c"])
    result = left.combine_first(right)
    expected = DataFrame(columns=["a", "b", "c"])
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\tests\test_printing.py ===
from decimal import Decimal

# For testing polynomial printing with object arrays
from fractions import Fraction
from math import inf, nan

import pytest

import numpy.polynomial as poly
from numpy._core import arange, array, printoptions
from numpy.testing import assert_, assert_equal


class TestStrUnicodeSuperSubscripts:

    @pytest.fixture(scope='class', autouse=True)
    def use_unicode(self):
        poly.set_default_printstyle('unicode')

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0·x + 3.0·x²"),
        ([-1, 0, 3, -1], "-1.0 + 0.0·x + 3.0·x² - 1.0·x³"),
        (arange(12), ("0.0 + 1.0·x + 2.0·x² + 3.0·x³ + 4.0·x⁴ + 5.0·x⁵ + "
                      "6.0·x⁶ + 7.0·x⁷ +\n8.0·x⁸ + 9.0·x⁹ + 10.0·x¹⁰ + "
                      "11.0·x¹¹")),
    ))
    def test_polynomial_str(self, inp, tgt):
        p = poly.Polynomial(inp)
        res = str(p)
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0·T₁(x) + 3.0·T₂(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0·T₁(x) + 3.0·T₂(x) - 1.0·T₃(x)"),
        (arange(12), ("0.0 + 1.0·T₁(x) + 2.0·T₂(x) + 3.0·T₃(x) + 4.0·T₄(x) + "
                      "5.0·T₅(x) +\n6.0·T₆(x) + 7.0·T₇(x) + 8.0·T₈(x) + "
                      "9.0·T₉(x) + 10.0·T₁₀(x) + 11.0·T₁₁(x)")),
    ))
    def test_chebyshev_str(self, inp, tgt):
        res = str(poly.Chebyshev(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0·P₁(x) + 3.0·P₂(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0·P₁(x) + 3.0·P₂(x) - 1.0·P₃(x)"),
        (arange(12), ("0.0 + 1.0·P₁(x) + 2.0·P₂(x) + 3.0·P₃(x) + 4.0·P₄(x) + "
                      "5.0·P₅(x) +\n6.0·P₆(x) + 7.0·P₇(x) + 8.0·P₈(x) + "
                      "9.0·P₉(x) + 10.0·P₁₀(x) + 11.0·P₁₁(x)")),
    ))
    def test_legendre_str(self, inp, tgt):
        res = str(poly.Legendre(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0·H₁(x) + 3.0·H₂(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0·H₁(x) + 3.0·H₂(x) - 1.0·H₃(x)"),
        (arange(12), ("0.0 + 1.0·H₁(x) + 2.0·H₂(x) + 3.0·H₃(x) + 4.0·H₄(x) + "
                      "5.0·H₅(x) +\n6.0·H₆(x) + 7.0·H₇(x) + 8.0·H₈(x) + "
                      "9.0·H₉(x) + 10.0·H₁₀(x) + 11.0·H₁₁(x)")),
    ))
    def test_hermite_str(self, inp, tgt):
        res = str(poly.Hermite(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0·He₁(x) + 3.0·He₂(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0·He₁(x) + 3.0·He₂(x) - 1.0·He₃(x)"),
        (arange(12), ("0.0 + 1.0·He₁(x) + 2.0·He₂(x) + 3.0·He₃(x) + "
                      "4.0·He₄(x) + 5.0·He₅(x) +\n6.0·He₆(x) + 7.0·He₇(x) + "
                      "8.0·He₈(x) + 9.0·He₉(x) + 10.0·He₁₀(x) +\n"
                      "11.0·He₁₁(x)")),
    ))
    def test_hermiteE_str(self, inp, tgt):
        res = str(poly.HermiteE(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0·L₁(x) + 3.0·L₂(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0·L₁(x) + 3.0·L₂(x) - 1.0·L₃(x)"),
        (arange(12), ("0.0 + 1.0·L₁(x) + 2.0·L₂(x) + 3.0·L₃(x) + 4.0·L₄(x) + "
                      "5.0·L₅(x) +\n6.0·L₆(x) + 7.0·L₇(x) + 8.0·L₈(x) + "
                      "9.0·L₉(x) + 10.0·L₁₀(x) + 11.0·L₁₁(x)")),
    ))
    def test_laguerre_str(self, inp, tgt):
        res = str(poly.Laguerre(inp))
        assert_equal(res, tgt)

    def test_polynomial_str_domains(self):
        res = str(poly.Polynomial([0, 1]))
        tgt = '0.0 + 1.0·x'
        assert_equal(res, tgt)

        res = str(poly.Polynomial([0, 1], domain=[1, 2]))
        tgt = '0.0 + 1.0·(-3.0 + 2.0x)'
        assert_equal(res, tgt)

class TestStrAscii:

    @pytest.fixture(scope='class', autouse=True)
    def use_ascii(self):
        poly.set_default_printstyle('ascii')

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0 x + 3.0 x**2"),
        ([-1, 0, 3, -1], "-1.0 + 0.0 x + 3.0 x**2 - 1.0 x**3"),
        (arange(12), ("0.0 + 1.0 x + 2.0 x**2 + 3.0 x**3 + 4.0 x**4 + "
                      "5.0 x**5 + 6.0 x**6 +\n7.0 x**7 + 8.0 x**8 + "
                      "9.0 x**9 + 10.0 x**10 + 11.0 x**11")),
    ))
    def test_polynomial_str(self, inp, tgt):
        res = str(poly.Polynomial(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0 T_1(x) + 3.0 T_2(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0 T_1(x) + 3.0 T_2(x) - 1.0 T_3(x)"),
        (arange(12), ("0.0 + 1.0 T_1(x) + 2.0 T_2(x) + 3.0 T_3(x) + "
                      "4.0 T_4(x) + 5.0 T_5(x) +\n6.0 T_6(x) + 7.0 T_7(x) + "
                      "8.0 T_8(x) + 9.0 T_9(x) + 10.0 T_10(x) +\n"
                      "11.0 T_11(x)")),
    ))
    def test_chebyshev_str(self, inp, tgt):
        res = str(poly.Chebyshev(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0 P_1(x) + 3.0 P_2(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0 P_1(x) + 3.0 P_2(x) - 1.0 P_3(x)"),
        (arange(12), ("0.0 + 1.0 P_1(x) + 2.0 P_2(x) + 3.0 P_3(x) + "
                      "4.0 P_4(x) + 5.0 P_5(x) +\n6.0 P_6(x) + 7.0 P_7(x) + "
                      "8.0 P_8(x) + 9.0 P_9(x) + 10.0 P_10(x) +\n"
                      "11.0 P_11(x)")),
    ))
    def test_legendre_str(self, inp, tgt):
        res = str(poly.Legendre(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0 H_1(x) + 3.0 H_2(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0 H_1(x) + 3.0 H_2(x) - 1.0 H_3(x)"),
        (arange(12), ("0.0 + 1.0 H_1(x) + 2.0 H_2(x) + 3.0 H_3(x) + "
                      "4.0 H_4(x) + 5.0 H_5(x) +\n6.0 H_6(x) + 7.0 H_7(x) + "
                      "8.0 H_8(x) + 9.0 H_9(x) + 10.0 H_10(x) +\n"
                      "11.0 H_11(x)")),
    ))
    def test_hermite_str(self, inp, tgt):
        res = str(poly.Hermite(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0 He_1(x) + 3.0 He_2(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0 He_1(x) + 3.0 He_2(x) - 1.0 He_3(x)"),
        (arange(12), ("0.0 + 1.0 He_1(x) + 2.0 He_2(x) + 3.0 He_3(x) + "
                      "4.0 He_4(x) +\n5.0 He_5(x) + 6.0 He_6(x) + "
                      "7.0 He_7(x) + 8.0 He_8(x) + 9.0 He_9(x) +\n"
                      "10.0 He_10(x) + 11.0 He_11(x)")),
    ))
    def test_hermiteE_str(self, inp, tgt):
        res = str(poly.HermiteE(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0 L_1(x) + 3.0 L_2(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0 L_1(x) + 3.0 L_2(x) - 1.0 L_3(x)"),
        (arange(12), ("0.0 + 1.0 L_1(x) + 2.0 L_2(x) + 3.0 L_3(x) + "
                      "4.0 L_4(x) + 5.0 L_5(x) +\n6.0 L_6(x) + 7.0 L_7(x) + "
                      "8.0 L_8(x) + 9.0 L_9(x) + 10.0 L_10(x) +\n"
                      "11.0 L_11(x)")),
    ))
    def test_laguerre_str(self, inp, tgt):
        res = str(poly.Laguerre(inp))
        assert_equal(res, tgt)

    def test_polynomial_str_domains(self):
        res = str(poly.Polynomial([0, 1]))
        tgt = '0.0 + 1.0 x'
        assert_equal(res, tgt)

        res = str(poly.Polynomial([0, 1], domain=[1, 2]))
        tgt = '0.0 + 1.0 (-3.0 + 2.0x)'
        assert_equal(res, tgt)

class TestLinebreaking:

    @pytest.fixture(scope='class', autouse=True)
    def use_ascii(self):
        poly.set_default_printstyle('ascii')

    def test_single_line_one_less(self):
        # With 'ascii' style, len(str(p)) is default linewidth - 1 (i.e. 74)
        p = poly.Polynomial([12345678, 12345678, 12345678, 12345678, 123])
        assert_equal(len(str(p)), 74)
        assert_equal(str(p), (
            '12345678.0 + 12345678.0 x + 12345678.0 x**2 + '
            '12345678.0 x**3 + 123.0 x**4'
        ))

    def test_num_chars_is_linewidth(self):
        # len(str(p)) == default linewidth == 75
        p = poly.Polynomial([12345678, 12345678, 12345678, 12345678, 1234])
        assert_equal(len(str(p)), 75)
        assert_equal(str(p), (
            '12345678.0 + 12345678.0 x + 12345678.0 x**2 + '
            '12345678.0 x**3 +\n1234.0 x**4'
        ))

    def test_first_linebreak_multiline_one_less_than_linewidth(self):
        # Multiline str where len(first_line) + len(next_term) == lw - 1 == 74
        p = poly.Polynomial(
                [12345678, 12345678, 12345678, 12345678, 1, 12345678]
            )
        assert_equal(len(str(p).split('\n')[0]), 74)
        assert_equal(str(p), (
            '12345678.0 + 12345678.0 x + 12345678.0 x**2 + '
            '12345678.0 x**3 + 1.0 x**4 +\n12345678.0 x**5'
        ))

    def test_first_linebreak_multiline_on_linewidth(self):
        # First line is one character longer than previous test
        p = poly.Polynomial(
                [12345678, 12345678, 12345678, 12345678.12, 1, 12345678]
            )
        assert_equal(str(p), (
            '12345678.0 + 12345678.0 x + 12345678.0 x**2 + '
            '12345678.12 x**3 +\n1.0 x**4 + 12345678.0 x**5'
        ))

    @pytest.mark.parametrize(('lw', 'tgt'), (
        (75, ('0.0 + 10.0 x + 200.0 x**2 + 3000.0 x**3 + 40000.0 x**4 + '
              '500000.0 x**5 +\n600000.0 x**6 + 70000.0 x**7 + 8000.0 x**8 + '
              '900.0 x**9')),
        (45, ('0.0 + 10.0 x + 200.0 x**2 + 3000.0 x**3 +\n40000.0 x**4 + '
              '500000.0 x**5 +\n600000.0 x**6 + 70000.0 x**7 + 8000.0 x**8 +\n'
              '900.0 x**9')),
        (132, ('0.0 + 10.0 x + 200.0 x**2 + 3000.0 x**3 + 40000.0 x**4 + '
               '500000.0 x**5 + 600000.0 x**6 + 70000.0 x**7 + 8000.0 x**8 + '
               '900.0 x**9')),
    ))
    def test_linewidth_printoption(self, lw, tgt):
        p = poly.Polynomial(
            [0, 10, 200, 3000, 40000, 500000, 600000, 70000, 8000, 900]
        )
        with printoptions(linewidth=lw):
            assert_equal(str(p), tgt)
            for line in str(p).split('\n'):
                assert_(len(line) < lw)


def test_set_default_printoptions():
    p = poly.Polynomial([1, 2, 3])
    c = poly.Chebyshev([1, 2, 3])
    poly.set_default_printstyle('ascii')
    assert_equal(str(p), "1.0 + 2.0 x + 3.0 x**2")
    assert_equal(str(c), "1.0 + 2.0 T_1(x) + 3.0 T_2(x)")
    poly.set_default_printstyle('unicode')
    assert_equal(str(p), "1.0 + 2.0·x + 3.0·x²")
    assert_equal(str(c), "1.0 + 2.0·T₁(x) + 3.0·T₂(x)")
    with pytest.raises(ValueError):
        poly.set_default_printstyle('invalid_input')


def test_complex_coefficients():
    """Test both numpy and built-in complex."""
    coefs = [0 + 1j, 1 + 1j, -2 + 2j, 3 + 0j]
    # numpy complex
    p1 = poly.Polynomial(coefs)
    # Python complex
    p2 = poly.Polynomial(array(coefs, dtype=object))
    poly.set_default_printstyle('unicode')
    assert_equal(str(p1), "1j + (1+1j)·x - (2-2j)·x² + (3+0j)·x³")
    assert_equal(str(p2), "1j + (1+1j)·x + (-2+2j)·x² + (3+0j)·x³")
    poly.set_default_printstyle('ascii')
    assert_equal(str(p1), "1j + (1+1j) x - (2-2j) x**2 + (3+0j) x**3")
    assert_equal(str(p2), "1j + (1+1j) x + (-2+2j) x**2 + (3+0j) x**3")


@pytest.mark.parametrize(('coefs', 'tgt'), (
    (array([Fraction(1, 2), Fraction(3, 4)], dtype=object), (
        "1/2 + 3/4·x"
    )),
    (array([1, 2, Fraction(5, 7)], dtype=object), (
        "1 + 2·x + 5/7·x²"
    )),
    (array([Decimal('1.00'), Decimal('2.2'), 3], dtype=object), (
        "1.00 + 2.2·x + 3·x²"
    )),
))
def test_numeric_object_coefficients(coefs, tgt):
    p = poly.Polynomial(coefs)
    poly.set_default_printstyle('unicode')
    assert_equal(str(p), tgt)


@pytest.mark.parametrize(('coefs', 'tgt'), (
    (array([1, 2, 'f'], dtype=object), '1 + 2·x + f·x²'),
    (array([1, 2, [3, 4]], dtype=object), '1 + 2·x + [3, 4]·x²'),
))
def test_nonnumeric_object_coefficients(coefs, tgt):
    """
    Test coef fallback for object arrays of non-numeric coefficients.
    """
    p = poly.Polynomial(coefs)
    poly.set_default_printstyle('unicode')
    assert_equal(str(p), tgt)


class TestFormat:
    def test_format_unicode(self):
        poly.set_default_printstyle('ascii')
        p = poly.Polynomial([1, 2, 0, -1])
        assert_equal(format(p, 'unicode'), "1.0 + 2.0·x + 0.0·x² - 1.0·x³")

    def test_format_ascii(self):
        poly.set_default_printstyle('unicode')
        p = poly.Polynomial([1, 2, 0, -1])
        assert_equal(
            format(p, 'ascii'), "1.0 + 2.0 x + 0.0 x**2 - 1.0 x**3"
        )

    def test_empty_formatstr(self):
        poly.set_default_printstyle('ascii')
        p = poly.Polynomial([1, 2, 3])
        assert_equal(format(p), "1.0 + 2.0 x + 3.0 x**2")
        assert_equal(f"{p}", "1.0 + 2.0 x + 3.0 x**2")

    def test_bad_formatstr(self):
        p = poly.Polynomial([1, 2, 0, -1])
        with pytest.raises(ValueError):
            format(p, '.2f')


@pytest.mark.parametrize(('poly', 'tgt'), (
    (poly.Polynomial, '1.0 + 2.0·z + 3.0·z²'),
    (poly.Chebyshev, '1.0 + 2.0·T₁(z) + 3.0·T₂(z)'),
    (poly.Hermite, '1.0 + 2.0·H₁(z) + 3.0·H₂(z)'),
    (poly.HermiteE, '1.0 + 2.0·He₁(z) + 3.0·He₂(z)'),
    (poly.Laguerre, '1.0 + 2.0·L₁(z) + 3.0·L₂(z)'),
    (poly.Legendre, '1.0 + 2.0·P₁(z) + 3.0·P₂(z)'),
))
def test_symbol(poly, tgt):
    p = poly([1, 2, 3], symbol='z')
    assert_equal(f"{p:unicode}", tgt)


class TestRepr:
    def test_polynomial_repr(self):
        res = repr(poly.Polynomial([0, 1]))
        tgt = (
            "Polynomial([0., 1.], domain=[-1.,  1.], window=[-1.,  1.], "
            "symbol='x')"
        )
        assert_equal(res, tgt)

    def test_chebyshev_repr(self):
        res = repr(poly.Chebyshev([0, 1]))
        tgt = (
            "Chebyshev([0., 1.], domain=[-1.,  1.], window=[-1.,  1.], "
            "symbol='x')"
        )
        assert_equal(res, tgt)

    def test_legendre_repr(self):
        res = repr(poly.Legendre([0, 1]))
        tgt = (
            "Legendre([0., 1.], domain=[-1.,  1.], window=[-1.,  1.], "
            "symbol='x')"
        )
        assert_equal(res, tgt)

    def test_hermite_repr(self):
        res = repr(poly.Hermite([0, 1]))
        tgt = (
            "Hermite([0., 1.], domain=[-1.,  1.], window=[-1.,  1.], "
            "symbol='x')"
        )
        assert_equal(res, tgt)

    def test_hermiteE_repr(self):
        res = repr(poly.HermiteE([0, 1]))
        tgt = (
            "HermiteE([0., 1.], domain=[-1.,  1.], window=[-1.,  1.], "
            "symbol='x')"
        )
        assert_equal(res, tgt)

    def test_laguerre_repr(self):
        res = repr(poly.Laguerre([0, 1]))
        tgt = (
            "Laguerre([0., 1.], domain=[0., 1.], window=[0., 1.], "
            "symbol='x')"
        )
        assert_equal(res, tgt)


class TestLatexRepr:
    """Test the latex repr used by Jupyter"""

    @staticmethod
    def as_latex(obj):
        # right now we ignore the formatting of scalars in our tests, since
        # it makes them too verbose. Ideally, the formatting of scalars will
        # be fixed such that tests below continue to pass
        obj._repr_latex_scalar = lambda x, parens=False: str(x)
        try:
            return obj._repr_latex_()
        finally:
            del obj._repr_latex_scalar

    def test_simple_polynomial(self):
        # default input
        p = poly.Polynomial([1, 2, 3])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0 + 2.0\,x + 3.0\,x^{2}$')

        # translated input
        p = poly.Polynomial([1, 2, 3], domain=[-2, 0])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0 + 2.0\,\left(1.0 + x\right) + 3.0\,\left(1.0 + x\right)^{2}$')  # noqa: E501

        # scaled input
        p = poly.Polynomial([1, 2, 3], domain=[-0.5, 0.5])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0 + 2.0\,\left(2.0x\right) + 3.0\,\left(2.0x\right)^{2}$')

        # affine input
        p = poly.Polynomial([1, 2, 3], domain=[-1, 0])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0 + 2.0\,\left(1.0 + 2.0x\right) + 3.0\,\left(1.0 + 2.0x\right)^{2}$')  # noqa: E501

    def test_basis_func(self):
        p = poly.Chebyshev([1, 2, 3])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0\,{T}_{0}(x) + 2.0\,{T}_{1}(x) + 3.0\,{T}_{2}(x)$')
        # affine input - check no surplus parens are added
        p = poly.Chebyshev([1, 2, 3], domain=[-1, 0])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0\,{T}_{0}(1.0 + 2.0x) + 2.0\,{T}_{1}(1.0 + 2.0x) + 3.0\,{T}_{2}(1.0 + 2.0x)$')  # noqa: E501

    def test_multichar_basis_func(self):
        p = poly.HermiteE([1, 2, 3])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0\,{He}_{0}(x) + 2.0\,{He}_{1}(x) + 3.0\,{He}_{2}(x)$')

    def test_symbol_basic(self):
        # default input
        p = poly.Polynomial([1, 2, 3], symbol='z')
        assert_equal(self.as_latex(p),
            r'$z \mapsto 1.0 + 2.0\,z + 3.0\,z^{2}$')

        # translated input
        p = poly.Polynomial([1, 2, 3], domain=[-2, 0], symbol='z')
        assert_equal(
            self.as_latex(p),
            (
                r'$z \mapsto 1.0 + 2.0\,\left(1.0 + z\right) + 3.0\,'
                r'\left(1.0 + z\right)^{2}$'
            ),
        )

        # scaled input
        p = poly.Polynomial([1, 2, 3], domain=[-0.5, 0.5], symbol='z')
        assert_equal(
            self.as_latex(p),
            (
                r'$z \mapsto 1.0 + 2.0\,\left(2.0z\right) + 3.0\,'
                r'\left(2.0z\right)^{2}$'
            ),
        )

        # affine input
        p = poly.Polynomial([1, 2, 3], domain=[-1, 0], symbol='z')
        assert_equal(
            self.as_latex(p),
            (
                r'$z \mapsto 1.0 + 2.0\,\left(1.0 + 2.0z\right) + 3.0\,'
                r'\left(1.0 + 2.0z\right)^{2}$'
            ),
        )

    def test_numeric_object_coefficients(self):
        coefs = array([Fraction(1, 2), Fraction(1)])
        p = poly.Polynomial(coefs)
        assert_equal(self.as_latex(p), '$x \\mapsto 1/2 + 1\\,x$')


SWITCH_TO_EXP = (
    '1.0 + (1.0e-01) x + (1.0e-02) x**2',
    '1.2 + (1.2e-01) x + (1.2e-02) x**2',
    '1.23 + 0.12 x + (1.23e-02) x**2 + (1.23e-03) x**3',
    '1.235 + 0.123 x + (1.235e-02) x**2 + (1.235e-03) x**3',
    '1.2346 + 0.1235 x + 0.0123 x**2 + (1.2346e-03) x**3 + (1.2346e-04) x**4',
    '1.23457 + 0.12346 x + 0.01235 x**2 + (1.23457e-03) x**3 + '
    '(1.23457e-04) x**4',
    '1.234568 + 0.123457 x + 0.012346 x**2 + 0.001235 x**3 + '
    '(1.234568e-04) x**4 + (1.234568e-05) x**5',
    '1.2345679 + 0.1234568 x + 0.0123457 x**2 + 0.0012346 x**3 + '
    '(1.2345679e-04) x**4 + (1.2345679e-05) x**5')

class TestPrintOptions:
    """
    Test the output is properly configured via printoptions.
    The exponential notation is enabled automatically when the values
    are too small or too large.
    """

    @pytest.fixture(scope='class', autouse=True)
    def use_ascii(self):
        poly.set_default_printstyle('ascii')

    def test_str(self):
        p = poly.Polynomial([1 / 2, 1 / 7, 1 / 7 * 10**8, 1 / 7 * 10**9])
        assert_equal(str(p), '0.5 + 0.14285714 x + 14285714.28571429 x**2 '
                             '+ (1.42857143e+08) x**3')

        with printoptions(precision=3):
            assert_equal(str(p), '0.5 + 0.143 x + 14285714.286 x**2 '
                                 '+ (1.429e+08) x**3')

    def test_latex(self):
        p = poly.Polynomial([1 / 2, 1 / 7, 1 / 7 * 10**8, 1 / 7 * 10**9])
        assert_equal(p._repr_latex_(),
            r'$x \mapsto \text{0.5} + \text{0.14285714}\,x + '
            r'\text{14285714.28571429}\,x^{2} + '
            r'\text{(1.42857143e+08)}\,x^{3}$')

        with printoptions(precision=3):
            assert_equal(p._repr_latex_(),
                r'$x \mapsto \text{0.5} + \text{0.143}\,x + '
                r'\text{14285714.286}\,x^{2} + \text{(1.429e+08)}\,x^{3}$')

    def test_fixed(self):
        p = poly.Polynomial([1 / 2])
        assert_equal(str(p), '0.5')

        with printoptions(floatmode='fixed'):
            assert_equal(str(p), '0.50000000')

        with printoptions(floatmode='fixed', precision=4):
            assert_equal(str(p), '0.5000')

    def test_switch_to_exp(self):
        for i, s in enumerate(SWITCH_TO_EXP):
            with printoptions(precision=i):
                p = poly.Polynomial([1.23456789 * 10**-i
                                     for i in range(i // 2 + 3)])
                assert str(p).replace('\n', ' ') == s

    def test_non_finite(self):
        p = poly.Polynomial([nan, inf])
        assert str(p) == 'nan + inf x'
        assert p._repr_latex_() == r'$x \mapsto \text{nan} + \text{inf}\,x$'  # noqa: RUF027
        with printoptions(nanstr='NAN', infstr='INF'):
            assert str(p) == 'NAN + INF x'
            assert p._repr_latex_() == \
                r'$x \mapsto \text{NAN} + \text{INF}\,x$'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\numeric\test_numeric.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    Series,
)
import pandas._testing as tm


class TestFloatNumericIndex:
    @pytest.fixture(params=[np.float64, np.float32])
    def dtype(self, request):
        return request.param

    @pytest.fixture
    def simple_index(self, dtype):
        values = np.arange(5, dtype=dtype)
        return Index(values)

    @pytest.fixture(
        params=[
            [1.5, 2, 3, 4, 5],
            [0.0, 2.5, 5.0, 7.5, 10.0],
            [5, 4, 3, 2, 1.5],
            [10.0, 7.5, 5.0, 2.5, 0.0],
        ],
        ids=["mixed", "float", "mixed_dec", "float_dec"],
    )
    def index(self, request, dtype):
        return Index(request.param, dtype=dtype)

    @pytest.fixture
    def mixed_index(self, dtype):
        return Index([1.5, 2, 3, 4, 5], dtype=dtype)

    @pytest.fixture
    def float_index(self, dtype):
        return Index([0.0, 2.5, 5.0, 7.5, 10.0], dtype=dtype)

    def test_repr_roundtrip(self, index):
        tm.assert_index_equal(eval(repr(index)), index, exact=True)

    def check_coerce(self, a, b, is_float_index=True):
        assert a.equals(b)
        tm.assert_index_equal(a, b, exact=False)
        if is_float_index:
            assert isinstance(b, Index)
        else:
            assert type(b) is Index

    def test_constructor_from_list_no_dtype(self):
        index = Index([1.5, 2.5, 3.5])
        assert index.dtype == np.float64

    def test_constructor(self, dtype):
        index_cls = Index

        # explicit construction
        index = index_cls([1, 2, 3, 4, 5], dtype=dtype)

        assert isinstance(index, index_cls)
        assert index.dtype == dtype

        expected = np.array([1, 2, 3, 4, 5], dtype=dtype)
        tm.assert_numpy_array_equal(index.values, expected)

        index = index_cls(np.array([1, 2, 3, 4, 5]), dtype=dtype)
        assert isinstance(index, index_cls)
        assert index.dtype == dtype

        index = index_cls([1.0, 2, 3, 4, 5], dtype=dtype)
        assert isinstance(index, index_cls)
        assert index.dtype == dtype

        index = index_cls(np.array([1.0, 2, 3, 4, 5]), dtype=dtype)
        assert isinstance(index, index_cls)
        assert index.dtype == dtype

        index = index_cls([1.0, 2, 3, 4, 5], dtype=dtype)
        assert isinstance(index, index_cls)
        assert index.dtype == dtype

        index = index_cls(np.array([1.0, 2, 3, 4, 5]), dtype=dtype)
        assert isinstance(index, index_cls)
        assert index.dtype == dtype

        # nan handling
        result = index_cls([np.nan, np.nan], dtype=dtype)
        assert pd.isna(result.values).all()

        result = index_cls(np.array([np.nan]), dtype=dtype)
        assert pd.isna(result.values).all()

    def test_constructor_invalid(self):
        index_cls = Index
        cls_name = index_cls.__name__
        # invalid
        msg = (
            rf"{cls_name}\(\.\.\.\) must be called with a collection of "
            r"some kind, 0\.0 was passed"
        )
        with pytest.raises(TypeError, match=msg):
            index_cls(0.0)

    def test_constructor_coerce(self, mixed_index, float_index):
        self.check_coerce(mixed_index, Index([1.5, 2, 3, 4, 5]))
        self.check_coerce(float_index, Index(np.arange(5) * 2.5))

        result = Index(np.array(np.arange(5) * 2.5, dtype=object))
        assert result.dtype == object  # as of 2.0 to match Series
        self.check_coerce(float_index, result.astype("float64"))

    def test_constructor_explicit(self, mixed_index, float_index):
        # these don't auto convert
        self.check_coerce(
            float_index, Index((np.arange(5) * 2.5), dtype=object), is_float_index=False
        )
        self.check_coerce(
            mixed_index, Index([1.5, 2, 3, 4, 5], dtype=object), is_float_index=False
        )

    def test_type_coercion_fail(self, any_int_numpy_dtype):
        # see gh-15832
        msg = "Trying to coerce float values to integers"
        with pytest.raises(ValueError, match=msg):
            Index([1, 2, 3.5], dtype=any_int_numpy_dtype)

    def test_equals_numeric(self):
        index_cls = Index

        idx = index_cls([1.0, 2.0])
        assert idx.equals(idx)
        assert idx.identical(idx)

        idx2 = index_cls([1.0, 2.0])
        assert idx.equals(idx2)

        idx = index_cls([1.0, np.nan])
        assert idx.equals(idx)
        assert idx.identical(idx)

        idx2 = index_cls([1.0, np.nan])
        assert idx.equals(idx2)

    @pytest.mark.parametrize(
        "other",
        (
            Index([1, 2], dtype=np.int64),
            Index([1.0, 2.0], dtype=object),
            Index([1, 2], dtype=object),
        ),
    )
    def test_equals_numeric_other_index_type(self, other):
        idx = Index([1.0, 2.0])
        assert idx.equals(other)
        assert other.equals(idx)

    @pytest.mark.parametrize(
        "vals",
        [
            pd.date_range("2016-01-01", periods=3),
            pd.timedelta_range("1 Day", periods=3),
        ],
    )
    def test_lookups_datetimelike_values(self, vals, dtype):
        # If we have datetime64 or timedelta64 values, make sure they are
        #  wrapped correctly  GH#31163
        ser = Series(vals, index=range(3, 6))
        ser.index = ser.index.astype(dtype)

        expected = vals[1]

        result = ser[4.0]
        assert isinstance(result, type(expected)) and result == expected
        result = ser[4]
        assert isinstance(result, type(expected)) and result == expected

        result = ser.loc[4.0]
        assert isinstance(result, type(expected)) and result == expected
        result = ser.loc[4]
        assert isinstance(result, type(expected)) and result == expected

        result = ser.at[4.0]
        assert isinstance(result, type(expected)) and result == expected
        # GH#31329 .at[4] should cast to 4.0, matching .loc behavior
        result = ser.at[4]
        assert isinstance(result, type(expected)) and result == expected

        result = ser.iloc[1]
        assert isinstance(result, type(expected)) and result == expected

        result = ser.iat[1]
        assert isinstance(result, type(expected)) and result == expected

    def test_doesnt_contain_all_the_things(self):
        idx = Index([np.nan])
        assert not idx.isin([0]).item()
        assert not idx.isin([1]).item()
        assert idx.isin([np.nan]).item()

    def test_nan_multiple_containment(self):
        index_cls = Index

        idx = index_cls([1.0, np.nan])
        tm.assert_numpy_array_equal(idx.isin([1.0]), np.array([True, False]))
        tm.assert_numpy_array_equal(idx.isin([2.0, np.pi]), np.array([False, False]))
        tm.assert_numpy_array_equal(idx.isin([np.nan]), np.array([False, True]))
        tm.assert_numpy_array_equal(idx.isin([1.0, np.nan]), np.array([True, True]))
        idx = index_cls([1.0, 2.0])
        tm.assert_numpy_array_equal(idx.isin([np.nan]), np.array([False, False]))

    def test_fillna_float64(self):
        index_cls = Index
        # GH 11343
        idx = Index([1.0, np.nan, 3.0], dtype=float, name="x")
        # can't downcast
        exp = Index([1.0, 0.1, 3.0], name="x")
        tm.assert_index_equal(idx.fillna(0.1), exp, exact=True)

        # downcast
        exp = index_cls([1.0, 2.0, 3.0], name="x")
        tm.assert_index_equal(idx.fillna(2), exp)

        # object
        exp = Index([1.0, "obj", 3.0], name="x")
        tm.assert_index_equal(idx.fillna("obj"), exp, exact=True)

    def test_logical_compat(self, simple_index):
        idx = simple_index
        assert idx.all() == idx.values.all()
        assert idx.any() == idx.values.any()

        assert idx.all() == idx.to_series().all()
        assert idx.any() == idx.to_series().any()


class TestNumericInt:
    @pytest.fixture(params=[np.int64, np.int32, np.int16, np.int8, np.uint64])
    def dtype(self, request):
        return request.param

    @pytest.fixture
    def simple_index(self, dtype):
        return Index(range(0, 20, 2), dtype=dtype)

    def test_is_monotonic(self):
        index_cls = Index

        index = index_cls([1, 2, 3, 4])
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index.is_monotonic_decreasing is False
        assert index._is_strictly_monotonic_decreasing is False

        index = index_cls([4, 3, 2, 1])
        assert index.is_monotonic_increasing is False
        assert index._is_strictly_monotonic_increasing is False
        assert index._is_strictly_monotonic_decreasing is True

        index = index_cls([1])
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True

    def test_is_strictly_monotonic(self):
        index_cls = Index

        index = index_cls([1, 1, 2, 3])
        assert index.is_monotonic_increasing is True
        assert index._is_strictly_monotonic_increasing is False

        index = index_cls([3, 2, 1, 1])
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_decreasing is False

        index = index_cls([1, 1])
        assert index.is_monotonic_increasing
        assert index.is_monotonic_decreasing
        assert not index._is_strictly_monotonic_increasing
        assert not index._is_strictly_monotonic_decreasing

    def test_logical_compat(self, simple_index):
        idx = simple_index
        assert idx.all() == idx.values.all()
        assert idx.any() == idx.values.any()

    def test_identical(self, simple_index, dtype):
        index = simple_index

        idx = Index(index.copy())
        assert idx.identical(index)

        same_values_different_type = Index(idx, dtype=object)
        assert not idx.identical(same_values_different_type)

        idx = index.astype(dtype=object)
        idx = idx.rename("foo")
        same_values = Index(idx, dtype=object)
        assert same_values.identical(idx)

        assert not idx.identical(index)
        assert Index(same_values, name="foo", dtype=object).identical(idx)

        assert not index.astype(dtype=object).identical(index.astype(dtype=dtype))

    def test_cant_or_shouldnt_cast(self, dtype):
        msg = r"invalid literal for int\(\) with base 10: 'foo'"

        # can't
        data = ["foo", "bar", "baz"]
        with pytest.raises(ValueError, match=msg):
            Index(data, dtype=dtype)

    def test_view_index(self, simple_index):
        index = simple_index
        msg = "Passing a type in .*Index.view is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            index.view(Index)

    def test_prevent_casting(self, simple_index):
        index = simple_index
        result = index.astype("O")
        assert result.dtype == np.object_


class TestIntNumericIndex:
    @pytest.fixture(params=[np.int64, np.int32, np.int16, np.int8])
    def dtype(self, request):
        return request.param

    def test_constructor_from_list_no_dtype(self):
        index = Index([1, 2, 3])
        assert index.dtype == np.int64

    def test_constructor(self, dtype):
        index_cls = Index

        # scalar raise Exception
        msg = (
            rf"{index_cls.__name__}\(\.\.\.\) must be called with a collection of some "
            "kind, 5 was passed"
        )
        with pytest.raises(TypeError, match=msg):
            index_cls(5)

        # copy
        # pass list, coerce fine
        index = index_cls([-5, 0, 1, 2], dtype=dtype)
        arr = index.values.copy()
        new_index = index_cls(arr, copy=True)
        tm.assert_index_equal(new_index, index, exact=True)
        val = int(arr[0]) + 3000

        # this should not change index
        if dtype != np.int8:
            # NEP 50 won't allow assignment that would overflow
            arr[0] = val
            assert new_index[0] != val

        if dtype == np.int64:
            # pass list, coerce fine
            index = index_cls([-5, 0, 1, 2], dtype=dtype)
            expected = Index([-5, 0, 1, 2], dtype=dtype)
            tm.assert_index_equal(index, expected)

            # from iterable
            index = index_cls(iter([-5, 0, 1, 2]), dtype=dtype)
            expected = index_cls([-5, 0, 1, 2], dtype=dtype)
            tm.assert_index_equal(index, expected, exact=True)

            # interpret list-like
            expected = index_cls([5, 0], dtype=dtype)
            for cls in [Index, index_cls]:
                for idx in [
                    cls([5, 0], dtype=dtype),
                    cls(np.array([5, 0]), dtype=dtype),
                    cls(Series([5, 0]), dtype=dtype),
                ]:
                    tm.assert_index_equal(idx, expected)

    def test_constructor_corner(self, dtype):
        index_cls = Index

        arr = np.array([1, 2, 3, 4], dtype=object)

        index = index_cls(arr, dtype=dtype)
        assert index.values.dtype == index.dtype
        if dtype == np.int64:
            without_dtype = Index(arr)
            # as of 2.0 we do not infer a dtype when we get an object-dtype
            #  ndarray of numbers, matching Series behavior
            assert without_dtype.dtype == object

            tm.assert_index_equal(index, without_dtype.astype(np.int64))

        # preventing casting
        arr = np.array([1, "2", 3, "4"], dtype=object)
        msg = "Trying to coerce float values to integers"
        with pytest.raises(ValueError, match=msg):
            index_cls(arr, dtype=dtype)

    def test_constructor_coercion_signed_to_unsigned(
        self,
        any_unsigned_int_numpy_dtype,
    ):
        # see gh-15832
        msg = "|".join(
            [
                "Trying to coerce negative values to unsigned integers",
                "The elements provided in the data cannot all be casted",
            ]
        )
        with pytest.raises(OverflowError, match=msg):
            Index([-1], dtype=any_unsigned_int_numpy_dtype)

    def test_constructor_np_signed(self, any_signed_int_numpy_dtype):
        # GH#47475
        scalar = np.dtype(any_signed_int_numpy_dtype).type(1)
        result = Index([scalar])
        expected = Index([1], dtype=any_signed_int_numpy_dtype)
        tm.assert_index_equal(result, expected, exact=True)

    def test_constructor_np_unsigned(self, any_unsigned_int_numpy_dtype):
        # GH#47475
        scalar = np.dtype(any_unsigned_int_numpy_dtype).type(1)
        result = Index([scalar])
        expected = Index([1], dtype=any_unsigned_int_numpy_dtype)
        tm.assert_index_equal(result, expected, exact=True)

    def test_coerce_list(self):
        # coerce things
        arr = Index([1, 2, 3, 4])
        assert isinstance(arr, Index)

        # but not if explicit dtype passed
        arr = Index([1, 2, 3, 4], dtype=object)
        assert type(arr) is Index


class TestFloat16Index:
    # float 16 indexes not supported
    # GH 49535
    def test_constructor(self):
        index_cls = Index
        dtype = np.float16

        msg = "float16 indexes are not supported"

        # explicit construction
        with pytest.raises(NotImplementedError, match=msg):
            index_cls([1, 2, 3, 4, 5], dtype=dtype)

        with pytest.raises(NotImplementedError, match=msg):
            index_cls(np.array([1, 2, 3, 4, 5]), dtype=dtype)

        with pytest.raises(NotImplementedError, match=msg):
            index_cls([1.0, 2, 3, 4, 5], dtype=dtype)

        with pytest.raises(NotImplementedError, match=msg):
            index_cls(np.array([1.0, 2, 3, 4, 5]), dtype=dtype)

        with pytest.raises(NotImplementedError, match=msg):
            index_cls([1.0, 2, 3, 4, 5], dtype=dtype)

        with pytest.raises(NotImplementedError, match=msg):
            index_cls(np.array([1.0, 2, 3, 4, 5]), dtype=dtype)

        # nan handling
        with pytest.raises(NotImplementedError, match=msg):
            index_cls([np.nan, np.nan], dtype=dtype)

        with pytest.raises(NotImplementedError, match=msg):
            index_cls(np.array([np.nan]), dtype=dtype)


@pytest.mark.parametrize(
    "box",
    [list, lambda x: np.array(x, dtype=object), lambda x: Index(x, dtype=object)],
)
def test_uint_index_does_not_convert_to_float64(box):
    # https://github.com/pandas-dev/pandas/issues/28279
    # https://github.com/pandas-dev/pandas/issues/28023
    series = Series(
        [0, 1, 2, 3, 4, 5],
        index=[
            7606741985629028552,
            17876870360202815256,
            17876870360202815256,
            13106359306506049338,
            8991270399732411471,
            8991270399732411472,
        ],
    )

    result = series.loc[box([7606741985629028552, 17876870360202815256])]

    expected = Index(
        [7606741985629028552, 17876870360202815256, 17876870360202815256],
        dtype="uint64",
    )
    tm.assert_index_equal(result.index, expected)

    tm.assert_equal(result, series.iloc[:3])


def test_float64_index_equals():
    # https://github.com/pandas-dev/pandas/issues/35217
    float_index = Index([1.0, 2, 3])
    string_index = Index(["1", "2", "3"])

    result = float_index.equals(string_index)
    assert result is False

    result = string_index.equals(float_index)
    assert result is False


def test_map_dtype_inference_unsigned_to_signed():
    # GH#44609 cases where we don't retain dtype
    idx = Index([1, 2, 3], dtype=np.uint64)
    result = idx.map(lambda x: -x)
    expected = Index([-1, -2, -3], dtype=np.int64)
    tm.assert_index_equal(result, expected)


def test_map_dtype_inference_overflows():
    # GH#44609 case where we have to upcast
    idx = Index(np.array([1, 2, 3], dtype=np.int8))
    result = idx.map(lambda x: x * 1000)
    # TODO: we could plausibly try to infer down to int16 here
    expected = Index([1000, 2000, 3000], dtype=np.int64)
    tm.assert_index_equal(result, expected)


def test_view_to_datetimelike():
    # GH#55710
    idx = Index([1, 2, 3])
    res = idx.view("m8[s]")
    expected = pd.TimedeltaIndex(idx.values.view("m8[s]"))
    tm.assert_index_equal(res, expected)

    res2 = idx.view("m8[D]")
    expected2 = idx.values.view("m8[D]")
    tm.assert_numpy_array_equal(res2, expected2)

    res3 = idx.view("M8[h]")
    expected3 = idx.values.view("M8[h]")
    tm.assert_numpy_array_equal(res3, expected3)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_kane.py ===
from sympy import solve
from sympy import (cos, expand, Matrix, sin, symbols, tan, sqrt, S,
                                zeros, eye)
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
                                     RigidBody, KanesMethod, inertia, Particle,
                                     dot, find_dynamicsymbols)
from sympy.testing.pytest import raises


def test_invalid_coordinates():
    # Simple pendulum, but use symbols instead of dynamicsymbols
    l, m, g = symbols('l m g')
    q, u = symbols('q u')  # Generalized coordinate
    kd = [q.diff(dynamicsymbols._t) - u]
    N, O = ReferenceFrame('N'), Point('O')
    O.set_vel(N, 0)
    P = Particle('P', Point('P'), m)
    P.point.set_pos(O, l * (sin(q) * N.x - cos(q) * N.y))
    F = (P.point, -m * g * N.y)
    raises(ValueError, lambda: KanesMethod(N, [q], [u], kd, bodies=[P],
                                           forcelist=[F]))


def test_one_dof():
    # This is for a 1 dof spring-mass-damper case.
    # It is described in more detail in the KanesMethod docstring.
    q, u = dynamicsymbols('q u')
    qd, ud = dynamicsymbols('q u', 1)
    m, c, k = symbols('m c k')
    N = ReferenceFrame('N')
    P = Point('P')
    P.set_vel(N, u * N.x)

    kd = [qd - u]
    FL = [(P, (-k * q - c * u) * N.x)]
    pa = Particle('pa', P, m)
    BL = [pa]

    KM = KanesMethod(N, [q], [u], kd)
    KM.kanes_equations(BL, FL)

    assert KM.bodies == BL
    assert KM.loads == FL

    MM = KM.mass_matrix
    forcing = KM.forcing
    rhs = MM.inv() * forcing
    assert expand(rhs[0]) == expand(-(q * k + u * c) / m)

    assert simplify(KM.rhs() -
                    KM.mass_matrix_full.LUsolve(KM.forcing_full)) == zeros(2, 1)

    assert (KM.linearize(A_and_B=True, )[0] == Matrix([[0, 1], [-k/m, -c/m]]))


def test_two_dof():
    # This is for a 2 d.o.f., 2 particle spring-mass-damper.
    # The first coordinate is the displacement of the first particle, and the
    # second is the relative displacement between the first and second
    # particles. Speeds are defined as the time derivatives of the particles.
    q1, q2, u1, u2 = dynamicsymbols('q1 q2 u1 u2')
    q1d, q2d, u1d, u2d = dynamicsymbols('q1 q2 u1 u2', 1)
    m, c1, c2, k1, k2 = symbols('m c1 c2 k1 k2')
    N = ReferenceFrame('N')
    P1 = Point('P1')
    P2 = Point('P2')
    P1.set_vel(N, u1 * N.x)
    P2.set_vel(N, (u1 + u2) * N.x)
    # Note we multiply the kinematic equation by an arbitrary factor
    # to test the implicit vs explicit kinematics attribute
    kd = [q1d/2 - u1/2, 2*q2d - 2*u2]

    # Now we create the list of forces, then assign properties to each
    # particle, then create a list of all particles.
    FL = [(P1, (-k1 * q1 - c1 * u1 + k2 * q2 + c2 * u2) * N.x), (P2, (-k2 *
        q2 - c2 * u2) * N.x)]
    pa1 = Particle('pa1', P1, m)
    pa2 = Particle('pa2', P2, m)
    BL = [pa1, pa2]

    # Finally we create the KanesMethod object, specify the inertial frame,
    # pass relevant information, and form Fr & Fr*. Then we calculate the mass
    # matrix and forcing terms, and finally solve for the udots.
    KM = KanesMethod(N, q_ind=[q1, q2], u_ind=[u1, u2], kd_eqs=kd)
    KM.kanes_equations(BL, FL)
    MM = KM.mass_matrix
    forcing = KM.forcing
    rhs = MM.inv() * forcing
    assert expand(rhs[0]) == expand((-k1 * q1 - c1 * u1 + k2 * q2 + c2 * u2)/m)
    assert expand(rhs[1]) == expand((k1 * q1 + c1 * u1 - 2 * k2 * q2 - 2 *
                                    c2 * u2) / m)

    # Check that the explicit form is the default and kinematic mass matrix is identity
    assert KM.explicit_kinematics
    assert KM.mass_matrix_kin == eye(2)

    # Check that for the implicit form the mass matrix is not identity
    KM.explicit_kinematics = False
    assert KM.mass_matrix_kin == Matrix([[S(1)/2, 0], [0, 2]])

    # Check that whether using implicit or explicit kinematics the RHS
    # equations are consistent with the matrix form
    for explicit_kinematics in [False, True]:
        KM.explicit_kinematics = explicit_kinematics
        assert simplify(KM.rhs() -
                        KM.mass_matrix_full.LUsolve(KM.forcing_full)) == zeros(4, 1)

    # Make sure an error is raised if nonlinear kinematic differential
    # equations are supplied.
    kd = [q1d - u1**2, sin(q2d) - cos(u2)]
    raises(ValueError, lambda: KanesMethod(N, q_ind=[q1, q2],
                                           u_ind=[u1, u2], kd_eqs=kd))

def test_pend():
    q, u = dynamicsymbols('q u')
    qd, ud = dynamicsymbols('q u', 1)
    m, l, g = symbols('m l g')
    N = ReferenceFrame('N')
    P = Point('P')
    P.set_vel(N, -l * u * sin(q) * N.x + l * u * cos(q) * N.y)
    kd = [qd - u]

    FL = [(P, m * g * N.x)]
    pa = Particle('pa', P, m)
    BL = [pa]

    KM = KanesMethod(N, [q], [u], kd)
    KM.kanes_equations(BL, FL)
    MM = KM.mass_matrix
    forcing = KM.forcing
    rhs = MM.inv() * forcing
    rhs.simplify()
    assert expand(rhs[0]) == expand(-g / l * sin(q))
    assert simplify(KM.rhs() -
                    KM.mass_matrix_full.LUsolve(KM.forcing_full)) == zeros(2, 1)


def test_rolling_disc():
    # Rolling Disc Example
    # Here the rolling disc is formed from the contact point up, removing the
    # need to introduce generalized speeds. Only 3 configuration and three
    # speed variables are need to describe this system, along with the disc's
    # mass and radius, and the local gravity (note that mass will drop out).
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1 q2 q3 u1 u2 u3')
    q1d, q2d, q3d, u1d, u2d, u3d = dynamicsymbols('q1 q2 q3 u1 u2 u3', 1)
    r, m, g = symbols('r m g')

    # The kinematics are formed by a series of simple rotations. Each simple
    # rotation creates a new frame, and the next rotation is defined by the new
    # frame's basis vectors. This example uses a 3-1-2 series of rotations, or
    # Z, X, Y series of rotations. Angular velocity for this is defined using
    # the second frame's basis (the lean frame).
    N = ReferenceFrame('N')
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    L = Y.orientnew('L', 'Axis', [q2, Y.x])
    R = L.orientnew('R', 'Axis', [q3, L.y])
    w_R_N_qd = R.ang_vel_in(N)
    R.set_ang_vel(N, u1 * L.x + u2 * L.y + u3 * L.z)

    # This is the translational kinematics. We create a point with no velocity
    # in N; this is the contact point between the disc and ground. Next we form
    # the position vector from the contact point to the disc's center of mass.
    # Finally we form the velocity and acceleration of the disc.
    C = Point('C')
    C.set_vel(N, 0)
    Dmc = C.locatenew('Dmc', r * L.z)
    Dmc.v2pt_theory(C, N, R)

    # This is a simple way to form the inertia dyadic.
    I = inertia(L, m / 4 * r**2, m / 2 * r**2, m / 4 * r**2)

    # Kinematic differential equations; how the generalized coordinate time
    # derivatives relate to generalized speeds.
    kd = [dot(R.ang_vel_in(N) - w_R_N_qd, uv) for uv in L]

    # Creation of the force list; it is the gravitational force at the mass
    # center of the disc. Then we create the disc by assigning a Point to the
    # center of mass attribute, a ReferenceFrame to the frame attribute, and mass
    # and inertia. Then we form the body list.
    ForceList = [(Dmc, - m * g * Y.z)]
    BodyD = RigidBody('BodyD', Dmc, R, m, (I, Dmc))
    BodyList = [BodyD]

    # Finally we form the equations of motion, using the same steps we did
    # before. Specify inertial frame, supply generalized speeds, supply
    # kinematic differential equation dictionary, compute Fr from the force
    # list and Fr* from the body list, compute the mass matrix and forcing
    # terms, then solve for the u dots (time derivatives of the generalized
    # speeds).
    KM = KanesMethod(N, q_ind=[q1, q2, q3], u_ind=[u1, u2, u3], kd_eqs=kd)
    KM.kanes_equations(BodyList, ForceList)
    MM = KM.mass_matrix
    forcing = KM.forcing
    rhs = MM.inv() * forcing
    kdd = KM.kindiffdict()
    rhs = rhs.subs(kdd)
    rhs.simplify()
    assert rhs.expand() == Matrix([(6*u2*u3*r - u3**2*r*tan(q2) +
        4*g*sin(q2))/(5*r), -2*u1*u3/3, u1*(-2*u2 + u3*tan(q2))]).expand()
    assert simplify(KM.rhs() -
                    KM.mass_matrix_full.LUsolve(KM.forcing_full)) == zeros(6, 1)

    # This code tests our output vs. benchmark values. When r=g=m=1, the
    # critical speed (where all eigenvalues of the linearized equations are 0)
    # is 1 / sqrt(3) for the upright case.
    A = KM.linearize(A_and_B=True)[0]
    A_upright = A.subs({r: 1, g: 1, m: 1}).subs({q1: 0, q2: 0, q3: 0, u1: 0, u3: 0})
    import sympy
    assert sympy.sympify(A_upright.subs({u2: 1 / sqrt(3)})).eigenvals() == {S.Zero: 6}


def test_aux():
    # Same as above, except we have 2 auxiliary speeds for the ground contact
    # point, which is known to be zero. In one case, we go through then
    # substitute the aux. speeds in at the end (they are zero, as well as their
    # derivative), in the other case, we use the built-in auxiliary speed part
    # of KanesMethod. The equations from each should be the same.
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1 q2 q3 u1 u2 u3')
    q1d, q2d, q3d, u1d, u2d, u3d = dynamicsymbols('q1 q2 q3 u1 u2 u3', 1)
    u4, u5, f1, f2 = dynamicsymbols('u4, u5, f1, f2')
    u4d, u5d = dynamicsymbols('u4, u5', 1)
    r, m, g = symbols('r m g')

    N = ReferenceFrame('N')
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    L = Y.orientnew('L', 'Axis', [q2, Y.x])
    R = L.orientnew('R', 'Axis', [q3, L.y])
    w_R_N_qd = R.ang_vel_in(N)
    R.set_ang_vel(N, u1 * L.x + u2 * L.y + u3 * L.z)

    C = Point('C')
    C.set_vel(N, u4 * L.x + u5 * (Y.z ^ L.x))
    Dmc = C.locatenew('Dmc', r * L.z)
    Dmc.v2pt_theory(C, N, R)
    Dmc.a2pt_theory(C, N, R)

    I = inertia(L, m / 4 * r**2, m / 2 * r**2, m / 4 * r**2)

    kd = [dot(R.ang_vel_in(N) - w_R_N_qd, uv) for uv in L]

    ForceList = [(Dmc, - m * g * Y.z), (C, f1 * L.x + f2 * (Y.z ^ L.x))]
    BodyD = RigidBody('BodyD', Dmc, R, m, (I, Dmc))
    BodyList = [BodyD]

    KM = KanesMethod(N, q_ind=[q1, q2, q3], u_ind=[u1, u2, u3, u4, u5],
                     kd_eqs=kd)
    (fr, frstar) = KM.kanes_equations(BodyList, ForceList)
    fr = fr.subs({u4d: 0, u5d: 0}).subs({u4: 0, u5: 0})
    frstar = frstar.subs({u4d: 0, u5d: 0}).subs({u4: 0, u5: 0})

    KM2 = KanesMethod(N, q_ind=[q1, q2, q3], u_ind=[u1, u2, u3], kd_eqs=kd,
                      u_auxiliary=[u4, u5])
    (fr2, frstar2) = KM2.kanes_equations(BodyList, ForceList)
    fr2 = fr2.subs({u4d: 0, u5d: 0}).subs({u4: 0, u5: 0})
    frstar2 = frstar2.subs({u4d: 0, u5d: 0}).subs({u4: 0, u5: 0})

    frstar.simplify()
    frstar2.simplify()

    assert (fr - fr2).expand() == Matrix([0, 0, 0, 0, 0])
    assert (frstar - frstar2).expand() == Matrix([0, 0, 0, 0, 0])


def test_parallel_axis():
    # This is for a 2 dof inverted pendulum on a cart.
    # This tests the parallel axis code in KanesMethod. The inertia of the
    # pendulum is defined about the hinge, not about the center of mass.

    # Defining the constants and knowns of the system
    gravity = symbols('g')
    k, ls = symbols('k ls')
    a, mA, mC = symbols('a mA mC')
    F = dynamicsymbols('F')
    Ix, Iy, Iz = symbols('Ix Iy Iz')

    # Declaring the Generalized coordinates and speeds
    q1, q2 = dynamicsymbols('q1 q2')
    q1d, q2d = dynamicsymbols('q1 q2', 1)
    u1, u2 = dynamicsymbols('u1 u2')
    u1d, u2d = dynamicsymbols('u1 u2', 1)

    # Creating reference frames
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')

    A.orient(N, 'Axis', [-q2, N.z])
    A.set_ang_vel(N, -u2 * N.z)

    # Origin of Newtonian reference frame
    O = Point('O')

    # Creating and Locating the positions of the cart, C, and the
    # center of mass of the pendulum, A
    C = O.locatenew('C', q1 * N.x)
    Ao = C.locatenew('Ao', a * A.y)

    # Defining velocities of the points
    O.set_vel(N, 0)
    C.set_vel(N, u1 * N.x)
    Ao.v2pt_theory(C, N, A)
    Cart = Particle('Cart', C, mC)
    Pendulum = RigidBody('Pendulum', Ao, A, mA, (inertia(A, Ix, Iy, Iz), C))

    # kinematical differential equations

    kindiffs = [q1d - u1, q2d - u2]

    bodyList = [Cart, Pendulum]

    forceList = [(Ao, -N.y * gravity * mA),
                 (C, -N.y * gravity * mC),
                 (C, -N.x * k * (q1 - ls)),
                 (C, N.x * F)]

    km = KanesMethod(N, [q1, q2], [u1, u2], kindiffs)
    (fr, frstar) = km.kanes_equations(bodyList, forceList)
    mm = km.mass_matrix_full
    assert mm[3, 3] == Iz

def test_input_format():
    # 1 dof problem from test_one_dof
    q, u = dynamicsymbols('q u')
    qd, ud = dynamicsymbols('q u', 1)
    m, c, k = symbols('m c k')
    N = ReferenceFrame('N')
    P = Point('P')
    P.set_vel(N, u * N.x)

    kd = [qd - u]
    FL = [(P, (-k * q - c * u) * N.x)]
    pa = Particle('pa', P, m)
    BL = [pa]

    KM = KanesMethod(N, [q], [u], kd)
    # test for input format kane.kanes_equations((body1, body2, particle1))
    assert KM.kanes_equations(BL)[0] == Matrix([0])
    # test for input format kane.kanes_equations(bodies=(body1, body 2), loads=(load1,load2))
    assert KM.kanes_equations(bodies=BL, loads=None)[0] == Matrix([0])
    # test for input format kane.kanes_equations(bodies=(body1, body 2), loads=None)
    assert KM.kanes_equations(BL, loads=None)[0] == Matrix([0])
    # test for input format kane.kanes_equations(bodies=(body1, body 2))
    assert KM.kanes_equations(BL)[0] == Matrix([0])
    # test for input format kane.kanes_equations(bodies=(body1, body2), loads=[])
    assert KM.kanes_equations(BL, [])[0] == Matrix([0])
    # test for error raised when a wrong force list (in this case a string) is provided
    raises(ValueError, lambda: KM._form_fr('bad input'))

    # 1 dof problem from test_one_dof with FL & BL in instance
    KM = KanesMethod(N, [q], [u], kd, bodies=BL, forcelist=FL)
    assert KM.kanes_equations()[0] == Matrix([-c*u - k*q])

    # 2 dof problem from test_two_dof
    q1, q2, u1, u2 = dynamicsymbols('q1 q2 u1 u2')
    q1d, q2d, u1d, u2d = dynamicsymbols('q1 q2 u1 u2', 1)
    m, c1, c2, k1, k2 = symbols('m c1 c2 k1 k2')
    N = ReferenceFrame('N')
    P1 = Point('P1')
    P2 = Point('P2')
    P1.set_vel(N, u1 * N.x)
    P2.set_vel(N, (u1 + u2) * N.x)
    kd = [q1d - u1, q2d - u2]

    FL = ((P1, (-k1 * q1 - c1 * u1 + k2 * q2 + c2 * u2) * N.x), (P2, (-k2 *
        q2 - c2 * u2) * N.x))
    pa1 = Particle('pa1', P1, m)
    pa2 = Particle('pa2', P2, m)
    BL = (pa1, pa2)

    KM = KanesMethod(N, q_ind=[q1, q2], u_ind=[u1, u2], kd_eqs=kd)
    # test for input format
    # kane.kanes_equations((body1, body2), (load1, load2))
    KM.kanes_equations(BL, FL)
    MM = KM.mass_matrix
    forcing = KM.forcing
    rhs = MM.inv() * forcing
    assert expand(rhs[0]) == expand((-k1 * q1 - c1 * u1 + k2 * q2 + c2 * u2)/m)
    assert expand(rhs[1]) == expand((k1 * q1 + c1 * u1 - 2 * k2 * q2 - 2 *
                                    c2 * u2) / m)


def test_implicit_kinematics():
    # Test that implicit kinematics can handle complicated
    # equations that explicit form struggles with
    # See https://github.com/sympy/sympy/issues/22626

    # Inertial frame
    NED = ReferenceFrame('NED')
    NED_o = Point('NED_o')
    NED_o.set_vel(NED, 0)

    # body frame
    q_att = dynamicsymbols('lambda_0:4', real=True)
    B = NED.orientnew('B', 'Quaternion', q_att)

    # Generalized coordinates
    q_pos = dynamicsymbols('B_x:z')
    B_cm = NED_o.locatenew('B_cm', q_pos[0]*B.x + q_pos[1]*B.y + q_pos[2]*B.z)

    q_ind = q_att[1:] + q_pos
    q_dep = [q_att[0]]

    kinematic_eqs = []

    # Generalized velocities
    B_ang_vel = B.ang_vel_in(NED)
    P, Q, R = dynamicsymbols('P Q R')
    B.set_ang_vel(NED, P*B.x + Q*B.y + R*B.z)

    B_ang_vel_kd = (B.ang_vel_in(NED) - B_ang_vel).simplify()

    # Equating the two gives us the kinematic equation
    kinematic_eqs += [
        B_ang_vel_kd & B.x,
        B_ang_vel_kd & B.y,
        B_ang_vel_kd & B.z
    ]

    B_cm_vel = B_cm.vel(NED)
    U, V, W = dynamicsymbols('U V W')
    B_cm.set_vel(NED, U*B.x + V*B.y + W*B.z)

    # Compute the velocity of the point using the two methods
    B_ref_vel_kd = (B_cm.vel(NED) - B_cm_vel)

    # taking dot product with unit vectors to get kinematic equations
    # relating body coordinates and velocities

    # Note, there is a choice to dot with NED.xyz here. That makes
    # the implicit form have some bigger terms but is still fine, the
    # explicit form still struggles though
    kinematic_eqs += [
                      B_ref_vel_kd & B.x,
                      B_ref_vel_kd & B.y,
                      B_ref_vel_kd & B.z,
                     ]

    u_ind = [U, V, W, P, Q, R]

    # constraints
    q_att_vec = Matrix(q_att)
    config_cons = [(q_att_vec.T*q_att_vec)[0] - 1] #unit norm
    kinematic_eqs = kinematic_eqs + [(q_att_vec.T * q_att_vec.diff())[0]]

    try:
        KM = KanesMethod(NED, q_ind, u_ind,
          q_dependent= q_dep,
          kd_eqs = kinematic_eqs,
          configuration_constraints = config_cons,
          velocity_constraints= [],
          u_dependent= [], #no dependent speeds
          u_auxiliary = [], # No auxiliary speeds
          explicit_kinematics = False # implicit kinematics
        )
    except Exception as e:
        raise e

    # mass and inertia dyadic relative to CM
    M_B = symbols('M_B')
    J_B = inertia(B, *[S(f'J_B_{ax}')*(1 if ax[0] == ax[1] else -1)
            for ax in ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']])
    J_B = J_B.subs({S('J_B_xy'): 0, S('J_B_yz'): 0})
    RB = RigidBody('RB', B_cm, B, M_B, (J_B, B_cm))

    rigid_bodies = [RB]
    # Forces
    force_list = [
        #gravity pointing down
        (RB.masscenter, RB.mass*S('g')*NED.z),
        #generic forces and torques in body frame(inputs)
        (RB.frame, dynamicsymbols('T_z')*B.z),
        (RB.masscenter, dynamicsymbols('F_z')*B.z)
    ]

    KM.kanes_equations(rigid_bodies, force_list)

    # Expecting implicit form to be less than 5% of the flops
    n_ops_implicit = sum(
        [x.count_ops() for x in KM.forcing_full] +
        [x.count_ops() for x in KM.mass_matrix_full]
    )
    # Save implicit kinematic matrices to use later
    mass_matrix_kin_implicit = KM.mass_matrix_kin
    forcing_kin_implicit = KM.forcing_kin

    KM.explicit_kinematics = True
    n_ops_explicit = sum(
        [x.count_ops() for x in KM.forcing_full] +
        [x.count_ops() for x in KM.mass_matrix_full]
    )
    forcing_kin_explicit = KM.forcing_kin

    assert n_ops_implicit / n_ops_explicit < .05

    # Ideally we would check that implicit and explicit equations give the same result as done in test_one_dof
    # But the whole raison-d'etre of the implicit equations is to deal with problems such
    # as this one where the explicit form is too complicated to handle, especially the angular part
    # (i.e. tests would be too slow)
    # Instead, we check that the kinematic equations are correct using more fundamental tests:
    #
    # (1) that we recover the kinematic equations we have provided
    assert (mass_matrix_kin_implicit * KM.q.diff() - forcing_kin_implicit) == Matrix(kinematic_eqs)

    # (2) that rate of quaternions matches what 'textbook' solutions give
    # Note that we just use the explicit kinematics for the linear velocities
    # as they are not as complicated as the angular ones
    qdot_candidate = forcing_kin_explicit

    quat_dot_textbook = Matrix([
        [0, -P, -Q, -R],
        [P,  0,  R, -Q],
        [Q, -R,  0,  P],
        [R,  Q, -P,  0],
    ]) * q_att_vec / 2

    # Again, if we don't use this "textbook" solution
    # sympy will struggle to deal with the terms related to quaternion rates
    # due to the number of operations involved
    qdot_candidate[-1] = quat_dot_textbook[0] # lambda_0, note the [-1] as sympy's Kane puts the dependent coordinate last
    qdot_candidate[0]  = quat_dot_textbook[1] # lambda_1
    qdot_candidate[1]  = quat_dot_textbook[2] # lambda_2
    qdot_candidate[2]  = quat_dot_textbook[3] # lambda_3

    # sub the config constraint in the candidate solution and compare to the implicit rhs
    lambda_0_sol = solve(config_cons[0], q_att_vec[0])[1]
    lhs_candidate = simplify(mass_matrix_kin_implicit * qdot_candidate).subs({q_att_vec[0]: lambda_0_sol})
    assert lhs_candidate == forcing_kin_implicit

def test_issue_24887():
    # Spherical pendulum
    g, l, m, c = symbols('g l m c')
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1:4 u1:4')
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    A.orient_body_fixed(N, (q1, q2, q3), 'zxy')
    N_w_A = A.ang_vel_in(N)
    # A.set_ang_vel(N, u1 * A.x + u2 * A.y + u3 * A.z)
    kdes = [N_w_A.dot(A.x) - u1, N_w_A.dot(A.y) - u2, N_w_A.dot(A.z) - u3]
    O = Point('O')
    O.set_vel(N, 0)
    Po = O.locatenew('Po', -l * A.y)
    Po.set_vel(A, 0)
    P = Particle('P', Po, m)
    kane = KanesMethod(N, [q1, q2, q3], [u1, u2, u3], kdes, bodies=[P],
                       forcelist=[(Po, -m * g * N.y)])
    kane.kanes_equations()
    expected_md = m * l ** 2 * Matrix([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    expected_fd = Matrix([
        [l*m*(g*(sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3)) - l*u2*u3)],
        [0], [l*m*(-g*(sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1)) + l*u1*u2)]])
    assert find_dynamicsymbols(kane.forcing).issubset({q1, q2, q3, u1, u2, u3})
    assert simplify(kane.mass_matrix - expected_md) == zeros(3, 3)
    assert simplify(kane.forcing - expected_fd) == zeros(3, 1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_smtlib.py ===
import contextlib
import itertools
import re
import typing
from enum import Enum
from typing import Callable

import sympy
from sympy import Add, Implies, sqrt
from sympy.core import Mul, Pow
from sympy.core import (S, pi, symbols, Function, Rational, Integer,
                        Symbol, Eq, Ne, Le, Lt, Gt, Ge)
from sympy.functions import Piecewise, exp, sin, cos
from sympy.assumptions.ask import Q
from sympy.printing.smtlib import smtlib_code
from sympy.testing.pytest import raises, Failed

x, y, z = symbols('x,y,z')


class _W(Enum):
    DEFAULTING_TO_FLOAT = re.compile("Could not infer type of `.+`. Defaulting to float.", re.IGNORECASE)
    WILL_NOT_DECLARE = re.compile("Non-Symbol/Function `.+` will not be declared.", re.IGNORECASE)
    WILL_NOT_ASSERT = re.compile("Non-Boolean expression `.+` will not be asserted. Converting to SMTLib verbatim.", re.IGNORECASE)


@contextlib.contextmanager
def _check_warns(expected: typing.Iterable[_W]):
    warns: typing.List[str] = []
    log_warn = warns.append
    yield log_warn

    errors = []
    for i, (w, e) in enumerate(itertools.zip_longest(warns, expected)):
        if not e:
            errors += [f"[{i}] Received unexpected warning `{w}`."]
        elif not w:
            errors += [f"[{i}] Did not receive expected warning `{e.name}`."]
        elif not e.value.match(w):
            errors += [f"[{i}] Warning `{w}` does not match expected {e.name}."]

    if errors: raise Failed('\n'.join(errors))


def test_Integer():
    with _check_warns([_W.WILL_NOT_ASSERT] * 2) as w:
        assert smtlib_code(Integer(67), log_warn=w) == "67"
        assert smtlib_code(Integer(-1), log_warn=w) == "-1"
    with _check_warns([]) as w:
        assert smtlib_code(Integer(67)) == "67"
        assert smtlib_code(Integer(-1)) == "-1"


def test_Rational():
    with _check_warns([_W.WILL_NOT_ASSERT] * 4) as w:
        assert smtlib_code(Rational(3, 7), log_warn=w) == "(/ 3 7)"
        assert smtlib_code(Rational(18, 9), log_warn=w) == "2"
        assert smtlib_code(Rational(3, -7), log_warn=w) == "(/ -3 7)"
        assert smtlib_code(Rational(-3, -7), log_warn=w) == "(/ 3 7)"

    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT] * 2) as w:
        assert smtlib_code(x + Rational(3, 7), auto_declare=False, log_warn=w) == "(+ (/ 3 7) x)"
        assert smtlib_code(Rational(3, 7) * x, log_warn=w) == "(declare-const x Real)\n" \
                                                              "(* (/ 3 7) x)"


def test_Relational():
    with _check_warns([_W.DEFAULTING_TO_FLOAT] * 12) as w:
        assert smtlib_code(Eq(x, y), auto_declare=False, log_warn=w) == "(assert (= x y))"
        assert smtlib_code(Ne(x, y), auto_declare=False, log_warn=w) == "(assert (not (= x y)))"
        assert smtlib_code(Le(x, y), auto_declare=False, log_warn=w) == "(assert (<= x y))"
        assert smtlib_code(Lt(x, y), auto_declare=False, log_warn=w) == "(assert (< x y))"
        assert smtlib_code(Gt(x, y), auto_declare=False, log_warn=w) == "(assert (> x y))"
        assert smtlib_code(Ge(x, y), auto_declare=False, log_warn=w) == "(assert (>= x y))"


def test_AppliedBinaryRelation():
    with _check_warns([_W.DEFAULTING_TO_FLOAT] * 12) as w:
        assert smtlib_code(Q.eq(x, y), auto_declare=False, log_warn=w) == "(assert (= x y))"
        assert smtlib_code(Q.ne(x, y), auto_declare=False, log_warn=w) == "(assert (not (= x y)))"
        assert smtlib_code(Q.lt(x, y), auto_declare=False, log_warn=w) == "(assert (< x y))"
        assert smtlib_code(Q.le(x, y), auto_declare=False, log_warn=w) == "(assert (<= x y))"
        assert smtlib_code(Q.gt(x, y), auto_declare=False, log_warn=w) == "(assert (> x y))"
        assert smtlib_code(Q.ge(x, y), auto_declare=False, log_warn=w) == "(assert (>= x y))"

    raises(ValueError, lambda: smtlib_code(Q.complex(x), log_warn=w))


def test_AppliedPredicate():
    with _check_warns([_W.DEFAULTING_TO_FLOAT] * 6) as w:
        assert smtlib_code(Q.positive(x), auto_declare=False, log_warn=w) == "(assert (> x 0))"
        assert smtlib_code(Q.negative(x), auto_declare=False, log_warn=w) == "(assert (< x 0))"
        assert smtlib_code(Q.zero(x), auto_declare=False, log_warn=w) == "(assert (= x 0))"
        assert smtlib_code(Q.nonpositive(x), auto_declare=False, log_warn=w) == "(assert (<= x 0))"
        assert smtlib_code(Q.nonnegative(x), auto_declare=False, log_warn=w) == "(assert (>= x 0))"
        assert smtlib_code(Q.nonzero(x), auto_declare=False, log_warn=w) == "(assert (not (= x 0)))"

def test_Function():
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(sin(x) ** cos(x), auto_declare=False, log_warn=w) == "(pow (sin x) (cos x))"

    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(
            abs(x),
            symbol_table={x: int, y: bool},
            known_types={int: "INTEGER_TYPE"},
            known_functions={sympy.Abs: "ABSOLUTE_VALUE_OF"},
            log_warn=w
        ) == "(declare-const x INTEGER_TYPE)\n" \
             "(ABSOLUTE_VALUE_OF x)"

    my_fun1 = Function('f1')
    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(
            my_fun1(x),
            symbol_table={my_fun1: Callable[[bool], float]},
            log_warn=w
        ) == "(declare-const x Bool)\n" \
             "(declare-fun f1 (Bool) Real)\n" \
             "(f1 x)"

    with _check_warns([]) as w:
        assert smtlib_code(
            my_fun1(x),
            symbol_table={my_fun1: Callable[[bool], bool]},
            log_warn=w
        ) == "(declare-const x Bool)\n" \
             "(declare-fun f1 (Bool) Bool)\n" \
             "(assert (f1 x))"

        assert smtlib_code(
            Eq(my_fun1(x, z), y),
            symbol_table={my_fun1: Callable[[int, bool], bool]},
            log_warn=w
        ) == "(declare-const x Int)\n" \
             "(declare-const y Bool)\n" \
             "(declare-const z Bool)\n" \
             "(declare-fun f1 (Int Bool) Bool)\n" \
             "(assert (= (f1 x z) y))"

        assert smtlib_code(
            Eq(my_fun1(x, z), y),
            symbol_table={my_fun1: Callable[[int, bool], bool]},
            known_functions={my_fun1: "MY_KNOWN_FUN", Eq: '=='},
            log_warn=w
        ) == "(declare-const x Int)\n" \
             "(declare-const y Bool)\n" \
             "(declare-const z Bool)\n" \
             "(assert (== (MY_KNOWN_FUN x z) y))"

    with _check_warns([_W.DEFAULTING_TO_FLOAT] * 3) as w:
        assert smtlib_code(
            Eq(my_fun1(x, z), y),
            known_functions={my_fun1: "MY_KNOWN_FUN", Eq: '=='},
            log_warn=w
        ) == "(declare-const x Real)\n" \
             "(declare-const y Real)\n" \
             "(declare-const z Real)\n" \
             "(assert (== (MY_KNOWN_FUN x z) y))"


def test_Pow():
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(x ** 3, auto_declare=False, log_warn=w) == "(pow x 3)"
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(x ** (y ** 3), auto_declare=False, log_warn=w) == "(pow x (pow y 3))"
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(x ** Rational(2, 3), auto_declare=False, log_warn=w) == '(pow x (/ 2 3))'

        a = Symbol('a', integer=True)
        b = Symbol('b', real=True)
        c = Symbol('c')

        def g(x): return 2 * x

        # if x=1, y=2, then expr=2.333...
        expr = 1 / (g(a) * 3.5) ** (a - b ** a) / (a ** 2 + b)

    with _check_warns([]) as w:
        assert smtlib_code(
            [
                Eq(a < 2, c),
                Eq(b > a, c),
                c & True,
                Eq(expr, 2 + Rational(1, 3))
            ],
            log_warn=w
        ) == '(declare-const a Int)\n' \
             '(declare-const b Real)\n' \
             '(declare-const c Bool)\n' \
             '(assert (= (< a 2) c))\n' \
             '(assert (= (> b a) c))\n' \
             '(assert c)\n' \
             '(assert (= ' \
             '(* (pow (* 7.0 a) (+ (pow b a) (* -1 a))) (pow (+ b (pow a 2)) -1)) ' \
             '(/ 7 3)' \
             '))'

    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(
            Mul(-2, c, Pow(Mul(b, b, evaluate=False), -1, evaluate=False), evaluate=False),
            log_warn=w
        ) == '(declare-const b Real)\n' \
             '(declare-const c Real)\n' \
             '(* -2 c (pow (* b b) -1))'


def test_basic_ops():
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(x * y, auto_declare=False, log_warn=w) == "(* x y)"

    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(x + y, auto_declare=False, log_warn=w) == "(+ x y)"

    # with _check_warns([_SmtlibWarnings.DEFAULTING_TO_FLOAT, _SmtlibWarnings.DEFAULTING_TO_FLOAT, _SmtlibWarnings.WILL_NOT_ASSERT]) as w:
    # todo: implement re-write, currently does '(+ x (* -1 y))' instead
    # assert smtlib_code(x - y, auto_declare=False, log_warn=w) == "(- x y)"

    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(-x, auto_declare=False, log_warn=w) == "(* -1 x)"


def test_quantifier_extensions():
    from sympy.logic.boolalg import Boolean
    from sympy import Interval, Tuple, sympify

    # start For-all quantifier class example
    class ForAll(Boolean):
        def _smtlib(self, printer):
            bound_symbol_declarations = [
                printer._s_expr(sym.name, [
                    printer._known_types[printer.symbol_table[sym]],
                    Interval(start, end)
                ]) for sym, start, end in self.limits
            ]
            return printer._s_expr('forall', [
                printer._s_expr('', bound_symbol_declarations),
                self.function
            ])

        @property
        def bound_symbols(self):
            return {s for s, _, _ in self.limits}

        @property
        def free_symbols(self):
            bound_symbol_names = {s.name for s in self.bound_symbols}
            return {
                s for s in self.function.free_symbols
                if s.name not in bound_symbol_names
            }

        def __new__(cls, *args):
            limits = [sympify(a) for a in args if isinstance(a, (tuple, Tuple))]
            function = [sympify(a) for a in args if isinstance(a, Boolean)]
            assert len(limits) + len(function) == len(args)
            assert len(function) == 1
            function = function[0]

            if isinstance(function, ForAll): return ForAll.__new__(
                ForAll, *(limits + function.limits), function.function
            )
            inst = Boolean.__new__(cls)
            inst._args = tuple(limits + [function])
            inst.limits = limits
            inst.function = function
            return inst

    # end For-All Quantifier class example

    f = Function('f')
    with _check_warns([_W.DEFAULTING_TO_FLOAT]) as w:
        assert smtlib_code(
            ForAll((x, -42, +21), Eq(f(x), f(x))),
            symbol_table={f: Callable[[float], float]},
            log_warn=w
        ) == '(assert (forall ( (x Real [-42, 21])) true))'

    with _check_warns([_W.DEFAULTING_TO_FLOAT] * 2) as w:
        assert smtlib_code(
            ForAll(
                (x, -42, +21), (y, -100, 3),
                Implies(Eq(x, y), Eq(f(x), f(y)))
            ),
            symbol_table={f: Callable[[float], float]},
            log_warn=w
        ) == '(declare-fun f (Real) Real)\n' \
             '(assert (' \
             'forall ( (x Real [-42, 21]) (y Real [-100, 3])) ' \
             '(=> (= x y) (= (f x) (f y)))' \
             '))'

    a = Symbol('a', integer=True)
    b = Symbol('b', real=True)
    c = Symbol('c')

    with _check_warns([]) as w:
        assert smtlib_code(
            ForAll(
                (a, 2, 100), ForAll(
                    (b, 2, 100),
                    Implies(a < b, sqrt(a) < b) | c
                )),
            log_warn=w
        ) == '(declare-const c Bool)\n' \
             '(assert (forall ( (a Int [2, 100]) (b Real [2, 100])) ' \
             '(or c (=> (< a b) (< (pow a (/ 1 2)) b)))' \
             '))'


def test_mix_number_mult_symbols():
    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(
            1 / pi,
            known_constants={pi: "MY_PI"},
            log_warn=w
        ) == '(pow MY_PI -1)'

    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(
            [
                Eq(pi, 3.14, evaluate=False),
                1 / pi,
            ],
            known_constants={pi: "MY_PI"},
            log_warn=w
        ) == '(assert (= MY_PI 3.14))\n' \
             '(pow MY_PI -1)'

    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(
            Add(S.Zero, S.One, S.NegativeOne, S.Half,
                S.Exp1, S.Pi, S.GoldenRatio, evaluate=False),
            known_constants={
                S.Pi: 'p', S.GoldenRatio: 'g',
                S.Exp1: 'e'
            },
            known_functions={
                Add: 'plus',
                exp: 'exp'
            },
            precision=3,
            log_warn=w
        ) == '(plus 0 1 -1 (/ 1 2) (exp 1) p g)'

    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(
            Add(S.Zero, S.One, S.NegativeOne, S.Half,
                S.Exp1, S.Pi, S.GoldenRatio, evaluate=False),
            known_constants={
                S.Pi: 'p'
            },
            known_functions={
                Add: 'plus',
                exp: 'exp'
            },
            precision=3,
            log_warn=w
        ) == '(plus 0 1 -1 (/ 1 2) (exp 1) p 1.62)'

    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(
            Add(S.Zero, S.One, S.NegativeOne, S.Half,
                S.Exp1, S.Pi, S.GoldenRatio, evaluate=False),
            known_functions={Add: 'plus'},
            precision=3,
            log_warn=w
        ) == '(plus 0 1 -1 (/ 1 2) 2.72 3.14 1.62)'

    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(
            Add(S.Zero, S.One, S.NegativeOne, S.Half,
                S.Exp1, S.Pi, S.GoldenRatio, evaluate=False),
            known_constants={S.Exp1: 'e'},
            known_functions={Add: 'plus'},
            precision=3,
            log_warn=w
        ) == '(plus 0 1 -1 (/ 1 2) e 3.14 1.62)'


def test_boolean():
    with _check_warns([]) as w:
        assert smtlib_code(x & y, log_warn=w) == '(declare-const x Bool)\n' \
                                                 '(declare-const y Bool)\n' \
                                                 '(assert (and x y))'
        assert smtlib_code(x | y, log_warn=w) == '(declare-const x Bool)\n' \
                                                 '(declare-const y Bool)\n' \
                                                 '(assert (or x y))'
        assert smtlib_code(~x, log_warn=w) == '(declare-const x Bool)\n' \
                                              '(assert (not x))'
        assert smtlib_code(x & y & z, log_warn=w) == '(declare-const x Bool)\n' \
                                                     '(declare-const y Bool)\n' \
                                                     '(declare-const z Bool)\n' \
                                                     '(assert (and x y z))'

    with _check_warns([_W.DEFAULTING_TO_FLOAT]) as w:
        assert smtlib_code((x & ~y) | (z > 3), log_warn=w) == '(declare-const x Bool)\n' \
                                                              '(declare-const y Bool)\n' \
                                                              '(declare-const z Real)\n' \
                                                              '(assert (or (> z 3) (and x (not y))))'

    f = Function('f')
    g = Function('g')
    h = Function('h')
    with _check_warns([_W.DEFAULTING_TO_FLOAT]) as w:
        assert smtlib_code(
            [Gt(f(x), y),
             Lt(y, g(z))],
            symbol_table={
                f: Callable[[bool], int], g: Callable[[bool], int],
            }, log_warn=w
        ) == '(declare-const x Bool)\n' \
             '(declare-const y Real)\n' \
             '(declare-const z Bool)\n' \
             '(declare-fun f (Bool) Int)\n' \
             '(declare-fun g (Bool) Int)\n' \
             '(assert (> (f x) y))\n' \
             '(assert (< y (g z)))'

    with _check_warns([]) as w:
        assert smtlib_code(
            [Eq(f(x), y),
             Lt(y, g(z))],
            symbol_table={
                f: Callable[[bool], int], g: Callable[[bool], int],
            }, log_warn=w
        ) == '(declare-const x Bool)\n' \
             '(declare-const y Int)\n' \
             '(declare-const z Bool)\n' \
             '(declare-fun f (Bool) Int)\n' \
             '(declare-fun g (Bool) Int)\n' \
             '(assert (= (f x) y))\n' \
             '(assert (< y (g z)))'

    with _check_warns([]) as w:
        assert smtlib_code(
            [Eq(f(x), y),
             Eq(g(f(x)), z),
             Eq(h(g(f(x))), x)],
            symbol_table={
                f: Callable[[float], int],
                g: Callable[[int], bool],
                h: Callable[[bool], float]
            },
            log_warn=w
        ) == '(declare-const x Real)\n' \
             '(declare-const y Int)\n' \
             '(declare-const z Bool)\n' \
             '(declare-fun f (Real) Int)\n' \
             '(declare-fun g (Int) Bool)\n' \
             '(declare-fun h (Bool) Real)\n' \
             '(assert (= (f x) y))\n' \
             '(assert (= (g (f x)) z))\n' \
             '(assert (= (h (g (f x))) x))'


# todo: make smtlib_code support arrays
# def test_containers():
#     assert julia_code([1, 2, 3, [4, 5, [6, 7]], 8, [9, 10], 11]) == \
#            "Any[1, 2, 3, Any[4, 5, Any[6, 7]], 8, Any[9, 10], 11]"
#     assert julia_code((1, 2, (3, 4))) == "(1, 2, (3, 4))"
#     assert julia_code([1]) == "Any[1]"
#     assert julia_code((1,)) == "(1,)"
#     assert julia_code(Tuple(*[1, 2, 3])) == "(1, 2, 3)"
#     assert julia_code((1, x * y, (3, x ** 2))) == "(1, x .* y, (3, x .^ 2))"
#     # scalar, matrix, empty matrix and empty list
#     assert julia_code((1, eye(3), Matrix(0, 0, []), [])) == "(1, [1 0 0;\n0 1 0;\n0 0 1], zeros(0, 0), Any[])"

def test_smtlib_piecewise():
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(
            Piecewise((x, x < 1),
                      (x ** 2, True)),
            auto_declare=False,
            log_warn=w
        ) == '(ite (< x 1) x (pow x 2))'

    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(
            Piecewise((x ** 2, x < 1),
                      (x ** 3, x < 2),
                      (x ** 4, x < 3),
                      (x ** 5, True)),
            auto_declare=False,
            log_warn=w
        ) == '(ite (< x 1) (pow x 2) ' \
             '(ite (< x 2) (pow x 3) ' \
             '(ite (< x 3) (pow x 4) ' \
             '(pow x 5))))'

    # Check that Piecewise without a True (default) condition error
    expr = Piecewise((x, x < 1), (x ** 2, x > 1), (sin(x), x > 0))
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        raises(AssertionError, lambda: smtlib_code(expr, log_warn=w))


def test_smtlib_piecewise_times_const():
    pw = Piecewise((x, x < 1), (x ** 2, True))
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(2 * pw, log_warn=w) == '(declare-const x Real)\n(* 2 (ite (< x 1) x (pow x 2)))'
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(pw / x, log_warn=w) == '(declare-const x Real)\n(* (pow x -1) (ite (< x 1) x (pow x 2)))'
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(pw / (x * y), log_warn=w) == '(declare-const x Real)\n(declare-const y Real)\n(* (pow x -1) (pow y -1) (ite (< x 1) x (pow x 2)))'
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(pw / 3, log_warn=w) == '(declare-const x Real)\n(* (/ 1 3) (ite (< x 1) x (pow x 2)))'


# todo: make smtlib_code support arrays / matrices ?
# def test_smtlib_matrix_assign_to():
#     A = Matrix([[1, 2, 3]])
#     assert smtlib_code(A, assign_to='a') == "a = [1 2 3]"
#     A = Matrix([[1, 2], [3, 4]])
#     assert smtlib_code(A, assign_to='A') == "A = [1 2;\n3 4]"

# def test_julia_matrix_1x1():
#     A = Matrix([[3]])
#     B = MatrixSymbol('B', 1, 1)
#     C = MatrixSymbol('C', 1, 2)
#     assert julia_code(A, assign_to=B) == "B = [3]"
#     raises(ValueError, lambda: julia_code(A, assign_to=C))

# def test_julia_matrix_elements():
#     A = Matrix([[x, 2, x * y]])
#     assert julia_code(A[0, 0] ** 2 + A[0, 1] + A[0, 2]) == "x .^ 2 + x .* y + 2"
#     A = MatrixSymbol('AA', 1, 3)
#     assert julia_code(A) == "AA"
#     assert julia_code(A[0, 0] ** 2 + sin(A[0, 1]) + A[0, 2]) == \
#            "sin(AA[1,2]) + AA[1,1] .^ 2 + AA[1,3]"
#     assert julia_code(sum(A)) == "AA[1,1] + AA[1,2] + AA[1,3]"

def test_smtlib_boolean():
    with _check_warns([]) as w:
        assert smtlib_code(True, auto_assert=False, log_warn=w) == 'true'
        assert smtlib_code(True, log_warn=w) == '(assert true)'
        assert smtlib_code(S.true, log_warn=w) == '(assert true)'
        assert smtlib_code(S.false, log_warn=w) == '(assert false)'
        assert smtlib_code(False, log_warn=w) == '(assert false)'
        assert smtlib_code(False, auto_assert=False, log_warn=w) == 'false'


def test_not_supported():
    f = Function('f')
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        raises(KeyError, lambda: smtlib_code(f(x).diff(x), symbol_table={f: Callable[[float], float]}, log_warn=w))
    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        raises(KeyError, lambda: smtlib_code(S.ComplexInfinity, log_warn=w))


def test_Float():
    assert smtlib_code(0.0) == "0.0"
    assert smtlib_code(0.000000000000000003) == '(* 3.0 (pow 10 -18))'
    assert smtlib_code(5.3) == "5.3"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_interpolate.py ===
import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    NaT,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDataFrameInterpolate:
    def test_interpolate_complex(self):
        # GH#53635
        ser = Series([complex("1+1j"), float("nan"), complex("2+2j")])
        assert ser.dtype.kind == "c"

        res = ser.interpolate()
        expected = Series([ser[0], ser[0] * 1.5, ser[2]])
        tm.assert_series_equal(res, expected)

        df = ser.to_frame()
        res = df.interpolate()
        expected = expected.to_frame()
        tm.assert_frame_equal(res, expected)

    def test_interpolate_datetimelike_values(self, frame_or_series):
        # GH#11312, GH#51005
        orig = Series(date_range("2012-01-01", periods=5))
        ser = orig.copy()
        ser[2] = NaT

        res = frame_or_series(ser).interpolate()
        expected = frame_or_series(orig)
        tm.assert_equal(res, expected)

        # datetime64tz cast
        ser_tz = ser.dt.tz_localize("US/Pacific")
        res_tz = frame_or_series(ser_tz).interpolate()
        expected_tz = frame_or_series(orig.dt.tz_localize("US/Pacific"))
        tm.assert_equal(res_tz, expected_tz)

        # timedelta64 cast
        ser_td = ser - ser[0]
        res_td = frame_or_series(ser_td).interpolate()
        expected_td = frame_or_series(orig - orig[0])
        tm.assert_equal(res_td, expected_td)

    def test_interpolate_inplace(self, frame_or_series, using_array_manager, request):
        # GH#44749
        if using_array_manager and frame_or_series is DataFrame:
            mark = pytest.mark.xfail(reason=".values-based in-place check is invalid")
            request.applymarker(mark)

        obj = frame_or_series([1, np.nan, 2])
        orig = obj.values

        obj.interpolate(inplace=True)
        expected = frame_or_series([1, 1.5, 2])
        tm.assert_equal(obj, expected)

        # check we operated *actually* inplace
        assert np.shares_memory(orig, obj.values)
        assert orig.squeeze()[1] == 1.5

    def test_interp_basic(self, using_copy_on_write, using_infer_string):
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 4],
                "B": [1, 4, 9, np.nan],
                "C": [1, 2, 3, 5],
                "D": list("abcd"),
            }
        )
        expected = DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0],
                "B": [1.0, 4.0, 9.0, 9.0],
                "C": [1, 2, 3, 5],
                "D": list("abcd"),
            }
        )
        if using_infer_string:
            dtype = "str" if using_infer_string else "object"
            msg = f"[Cc]annot interpolate with {dtype} dtype"
            with pytest.raises(TypeError, match=msg):
                df.interpolate()
            return

        msg = "DataFrame.interpolate with object dtype"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.interpolate()
        tm.assert_frame_equal(result, expected)

        # check we didn't operate inplace GH#45791
        cvalues = df["C"]._values
        dvalues = df["D"].values
        if using_copy_on_write:
            assert np.shares_memory(cvalues, result["C"]._values)
            assert np.shares_memory(dvalues, result["D"]._values)
        else:
            assert not np.shares_memory(cvalues, result["C"]._values)
            assert not np.shares_memory(dvalues, result["D"]._values)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = df.interpolate(inplace=True)
        assert res is None
        tm.assert_frame_equal(df, expected)

        # check we DID operate inplace
        assert tm.shares_memory(df["C"]._values, cvalues)
        assert tm.shares_memory(df["D"]._values, dvalues)

    @pytest.mark.xfail(
        using_string_dtype(), reason="interpolate doesn't work for string"
    )
    def test_interp_basic_with_non_range_index(self, using_infer_string):
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 4],
                "B": [1, 4, 9, np.nan],
                "C": [1, 2, 3, 5],
                "D": list("abcd"),
            }
        )

        msg = "DataFrame.interpolate with object dtype"
        warning = FutureWarning if not using_infer_string else None
        with tm.assert_produces_warning(warning, match=msg):
            result = df.set_index("C").interpolate()
        expected = df.set_index("C")
        expected.loc[3, "A"] = 3
        expected.loc[5, "B"] = 9
        tm.assert_frame_equal(result, expected)

    def test_interp_empty(self):
        # https://github.com/pandas-dev/pandas/issues/35598
        df = DataFrame()
        result = df.interpolate()
        assert result is not df
        expected = df
        tm.assert_frame_equal(result, expected)

    def test_interp_bad_method(self):
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 4],
                "B": [1, 4, 9, np.nan],
                "C": [1, 2, 3, 5],
            }
        )
        msg = (
            r"method must be one of \['linear', 'time', 'index', 'values', "
            r"'nearest', 'zero', 'slinear', 'quadratic', 'cubic', "
            r"'barycentric', 'krogh', 'spline', 'polynomial', "
            r"'from_derivatives', 'piecewise_polynomial', 'pchip', 'akima', "
            r"'cubicspline'\]. Got 'not_a_method' instead."
        )
        with pytest.raises(ValueError, match=msg):
            df.interpolate(method="not_a_method")

    def test_interp_combo(self):
        df = DataFrame(
            {
                "A": [1.0, 2.0, np.nan, 4.0],
                "B": [1, 4, 9, np.nan],
                "C": [1, 2, 3, 5],
                "D": list("abcd"),
            }
        )

        result = df["A"].interpolate()
        expected = Series([1.0, 2.0, 3.0, 4.0], name="A")
        tm.assert_series_equal(result, expected)

        msg = "The 'downcast' keyword in Series.interpolate is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df["A"].interpolate(downcast="infer")
        expected = Series([1, 2, 3, 4], name="A")
        tm.assert_series_equal(result, expected)

    def test_inerpolate_invalid_downcast(self):
        # GH#53103
        df = DataFrame(
            {
                "A": [1.0, 2.0, np.nan, 4.0],
                "B": [1, 4, 9, np.nan],
                "C": [1, 2, 3, 5],
                "D": list("abcd"),
            }
        )

        msg = "downcast must be either None or 'infer'"
        msg2 = "The 'downcast' keyword in DataFrame.interpolate is deprecated"
        msg3 = "The 'downcast' keyword in Series.interpolate is deprecated"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=msg2):
                df.interpolate(downcast="int64")
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=msg3):
                df["A"].interpolate(downcast="int64")

    def test_interp_nan_idx(self):
        df = DataFrame({"A": [1, 2, np.nan, 4], "B": [np.nan, 2, 3, 4]})
        df = df.set_index("A")
        msg = (
            "Interpolation with NaNs in the index has not been implemented. "
            "Try filling those NaNs before interpolating."
        )
        with pytest.raises(NotImplementedError, match=msg):
            df.interpolate(method="values")

    def test_interp_various(self):
        pytest.importorskip("scipy")
        df = DataFrame(
            {"A": [1, 2, np.nan, 4, 5, np.nan, 7], "C": [1, 2, 3, 5, 8, 13, 21]}
        )
        df = df.set_index("C")
        expected = df.copy()
        result = df.interpolate(method="polynomial", order=1)

        expected.loc[3, "A"] = 2.66666667
        expected.loc[13, "A"] = 5.76923076
        tm.assert_frame_equal(result, expected)

        result = df.interpolate(method="cubic")
        # GH #15662.
        expected.loc[3, "A"] = 2.81547781
        expected.loc[13, "A"] = 5.52964175
        tm.assert_frame_equal(result, expected)

        result = df.interpolate(method="nearest")
        expected.loc[3, "A"] = 2
        expected.loc[13, "A"] = 5
        tm.assert_frame_equal(result, expected, check_dtype=False)

        result = df.interpolate(method="quadratic")
        expected.loc[3, "A"] = 2.82150771
        expected.loc[13, "A"] = 6.12648668
        tm.assert_frame_equal(result, expected)

        result = df.interpolate(method="slinear")
        expected.loc[3, "A"] = 2.66666667
        expected.loc[13, "A"] = 5.76923077
        tm.assert_frame_equal(result, expected)

        result = df.interpolate(method="zero")
        expected.loc[3, "A"] = 2.0
        expected.loc[13, "A"] = 5
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_interp_alt_scipy(self):
        pytest.importorskip("scipy")
        df = DataFrame(
            {"A": [1, 2, np.nan, 4, 5, np.nan, 7], "C": [1, 2, 3, 5, 8, 13, 21]}
        )
        result = df.interpolate(method="barycentric")
        expected = df.copy()
        expected.loc[2, "A"] = 3
        expected.loc[5, "A"] = 6
        tm.assert_frame_equal(result, expected)

        msg = "The 'downcast' keyword in DataFrame.interpolate is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.interpolate(method="barycentric", downcast="infer")
        tm.assert_frame_equal(result, expected.astype(np.int64))

        result = df.interpolate(method="krogh")
        expectedk = df.copy()
        expectedk["A"] = expected["A"]
        tm.assert_frame_equal(result, expectedk)

        result = df.interpolate(method="pchip")
        expected.loc[2, "A"] = 3
        expected.loc[5, "A"] = 6.0

        tm.assert_frame_equal(result, expected)

    def test_interp_rowwise(self):
        df = DataFrame(
            {
                0: [1, 2, np.nan, 4],
                1: [2, 3, 4, np.nan],
                2: [np.nan, 4, 5, 6],
                3: [4, np.nan, 6, 7],
                4: [1, 2, 3, 4],
            }
        )
        result = df.interpolate(axis=1)
        expected = df.copy()
        expected.loc[3, 1] = 5
        expected.loc[0, 2] = 3
        expected.loc[1, 3] = 3
        expected[4] = expected[4].astype(np.float64)
        tm.assert_frame_equal(result, expected)

        result = df.interpolate(axis=1, method="values")
        tm.assert_frame_equal(result, expected)

        result = df.interpolate(axis=0)
        expected = df.interpolate()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "axis_name, axis_number",
        [
            pytest.param("rows", 0, id="rows_0"),
            pytest.param("index", 0, id="index_0"),
            pytest.param("columns", 1, id="columns_1"),
        ],
    )
    def test_interp_axis_names(self, axis_name, axis_number):
        # GH 29132: test axis names
        data = {0: [0, np.nan, 6], 1: [1, np.nan, 7], 2: [2, 5, 8]}

        df = DataFrame(data, dtype=np.float64)
        result = df.interpolate(axis=axis_name, method="linear")
        expected = df.interpolate(axis=axis_number, method="linear")
        tm.assert_frame_equal(result, expected)

    def test_rowwise_alt(self):
        df = DataFrame(
            {
                0: [0, 0.5, 1.0, np.nan, 4, 8, np.nan, np.nan, 64],
                1: [1, 2, 3, 4, 3, 2, 1, 0, -1],
            }
        )
        df.interpolate(axis=0)
        # TODO: assert something?

    @pytest.mark.parametrize(
        "check_scipy", [False, pytest.param(True, marks=td.skip_if_no("scipy"))]
    )
    def test_interp_leading_nans(self, check_scipy):
        df = DataFrame(
            {"A": [np.nan, np.nan, 0.5, 0.25, 0], "B": [np.nan, -3, -3.5, np.nan, -4]}
        )
        result = df.interpolate()
        expected = df.copy()
        expected.loc[3, "B"] = -3.75
        tm.assert_frame_equal(result, expected)

        if check_scipy:
            result = df.interpolate(method="polynomial", order=1)
            tm.assert_frame_equal(result, expected)

    def test_interp_raise_on_only_mixed(self, axis):
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 4],
                "B": ["a", "b", "c", "d"],
                "C": [np.nan, 2, 5, 7],
                "D": [np.nan, np.nan, 9, 9],
                "E": [1, 2, 3, 4],
            }
        )
        msg = (
            "Cannot interpolate with all object-dtype columns "
            "in the DataFrame. Try setting at least one "
            "column to a numeric dtype."
        )
        with pytest.raises(TypeError, match=msg):
            df.astype("object").interpolate(axis=axis)

    def test_interp_raise_on_all_object_dtype(self):
        # GH 22985
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, dtype="object")
        msg = (
            "Cannot interpolate with all object-dtype columns "
            "in the DataFrame. Try setting at least one "
            "column to a numeric dtype."
        )
        with pytest.raises(TypeError, match=msg):
            df.interpolate()

    def test_interp_inplace(self, using_copy_on_write):
        df = DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        expected = DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        expected_cow = df.copy()
        result = df.copy()

        if using_copy_on_write:
            with tm.raises_chained_assignment_error():
                return_value = result["a"].interpolate(inplace=True)
            assert return_value is None
            tm.assert_frame_equal(result, expected_cow)
        else:
            with tm.assert_produces_warning(FutureWarning, match="inplace method"):
                return_value = result["a"].interpolate(inplace=True)
            assert return_value is None
            tm.assert_frame_equal(result, expected)

        result = df.copy()
        msg = "The 'downcast' keyword in Series.interpolate is deprecated"

        if using_copy_on_write:
            with tm.assert_produces_warning(
                (FutureWarning, ChainedAssignmentError), match=msg
            ):
                return_value = result["a"].interpolate(inplace=True, downcast="infer")
            assert return_value is None
            tm.assert_frame_equal(result, expected_cow)
        else:
            with tm.assert_produces_warning(FutureWarning, match=msg):
                return_value = result["a"].interpolate(inplace=True, downcast="infer")
            assert return_value is None
            tm.assert_frame_equal(result, expected.astype("int64"))

    def test_interp_inplace_row(self):
        # GH 10395
        result = DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0], "b": [np.nan, 2.0, 3.0, 4.0], "c": [3, 2, 2, 2]}
        )
        expected = result.interpolate(method="linear", axis=1, inplace=False)
        return_value = result.interpolate(method="linear", axis=1, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected)

    def test_interp_ignore_all_good(self):
        # GH
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 4],
                "B": [1, 2, 3, 4],
                "C": [1.0, 2.0, np.nan, 4.0],
                "D": [1.0, 2.0, 3.0, 4.0],
            }
        )
        expected = DataFrame(
            {
                "A": np.array([1, 2, 3, 4], dtype="float64"),
                "B": np.array([1, 2, 3, 4], dtype="int64"),
                "C": np.array([1.0, 2.0, 3, 4.0], dtype="float64"),
                "D": np.array([1.0, 2.0, 3.0, 4.0], dtype="float64"),
            }
        )

        msg = "The 'downcast' keyword in DataFrame.interpolate is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.interpolate(downcast=None)
        tm.assert_frame_equal(result, expected)

        # all good
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df[["B", "D"]].interpolate(downcast=None)
        tm.assert_frame_equal(result, df[["B", "D"]])

    def test_interp_time_inplace_axis(self):
        # GH 9687
        periods = 5
        idx = date_range(start="2014-01-01", periods=periods)
        data = np.random.default_rng(2).random((periods, periods))
        data[data < 0.5] = np.nan
        expected = DataFrame(index=idx, columns=idx, data=data)

        result = expected.interpolate(axis=0, method="time")
        return_value = expected.interpolate(axis=0, method="time", inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("axis_name, axis_number", [("index", 0), ("columns", 1)])
    def test_interp_string_axis(self, axis_name, axis_number):
        # https://github.com/pandas-dev/pandas/issues/25190
        x = np.linspace(0, 100, 1000)
        y = np.sin(x)
        df = DataFrame(
            data=np.tile(y, (10, 1)), index=np.arange(10), columns=x
        ).reindex(columns=x * 1.005)
        result = df.interpolate(method="linear", axis=axis_name)
        expected = df.interpolate(method="linear", axis=axis_number)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("multiblock", [True, False])
    @pytest.mark.parametrize("method", ["ffill", "bfill", "pad"])
    def test_interp_fillna_methods(
        self, request, axis, multiblock, method, using_array_manager
    ):
        # GH 12918
        if using_array_manager and axis in (1, "columns"):
            # TODO(ArrayManager) support axis=1
            td.mark_array_manager_not_yet_implemented(request)

        df = DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0, np.nan, 5.0],
                "B": [2.0, 4.0, 6.0, np.nan, 8.0, 10.0],
                "C": [3.0, 6.0, 9.0, np.nan, np.nan, 30.0],
            }
        )
        if multiblock:
            df["D"] = np.nan
            df["E"] = 1.0

        method2 = method if method != "pad" else "ffill"
        expected = getattr(df, method2)(axis=axis)
        msg = f"DataFrame.interpolate with method={method} is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.interpolate(method=method, axis=axis)
        tm.assert_frame_equal(result, expected)

    def test_interpolate_empty_df(self):
        # GH#53199
        df = DataFrame()
        expected = df.copy()
        result = df.interpolate(inplace=True)
        assert result is None
        tm.assert_frame_equal(df, expected)

    def test_interpolate_ea(self, any_int_ea_dtype):
        # GH#55347
        df = DataFrame({"a": [1, None, None, None, 3]}, dtype=any_int_ea_dtype)
        orig = df.copy()
        result = df.interpolate(limit=2)
        expected = DataFrame({"a": [1, 1.5, 2.0, None, 3]}, dtype="Float64")
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(df, orig)

    @pytest.mark.parametrize(
        "dtype",
        [
            "Float64",
            "Float32",
            pytest.param("float32[pyarrow]", marks=td.skip_if_no("pyarrow")),
            pytest.param("float64[pyarrow]", marks=td.skip_if_no("pyarrow")),
        ],
    )
    def test_interpolate_ea_float(self, dtype):
        # GH#55347
        df = DataFrame({"a": [1, None, None, None, 3]}, dtype=dtype)
        orig = df.copy()
        result = df.interpolate(limit=2)
        expected = DataFrame({"a": [1, 1.5, 2.0, None, 3]}, dtype=dtype)
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(df, orig)

    @pytest.mark.parametrize(
        "dtype",
        ["int64", "uint64", "int32", "int16", "int8", "uint32", "uint16", "uint8"],
    )
    def test_interpolate_arrow(self, dtype):
        # GH#55347
        pytest.importorskip("pyarrow")
        df = DataFrame({"a": [1, None, None, None, 3]}, dtype=dtype + "[pyarrow]")
        result = df.interpolate(limit=2)
        expected = DataFrame({"a": [1, 1.5, 2.0, None, 3]}, dtype="float64[pyarrow]")
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\sparse\test_libsparse.py ===
import operator

import numpy as np
import pytest

import pandas._libs.sparse as splib
import pandas.util._test_decorators as td

from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import (
    BlockIndex,
    IntIndex,
    make_sparse_index,
)


@pytest.fixture
def test_length():
    return 20


@pytest.fixture(
    params=[
        [
            [0, 7, 15],
            [3, 5, 5],
            [2, 9, 14],
            [2, 3, 5],
            [2, 9, 15],
            [1, 3, 4],
        ],
        [
            [0, 5],
            [4, 4],
            [1],
            [4],
            [1],
            [3],
        ],
        [
            [0],
            [10],
            [0, 5],
            [3, 7],
            [0, 5],
            [3, 5],
        ],
        [
            [10],
            [5],
            [0, 12],
            [5, 3],
            [12],
            [3],
        ],
        [
            [0, 10],
            [4, 6],
            [5, 17],
            [4, 2],
            [],
            [],
        ],
        [
            [0],
            [5],
            [],
            [],
            [],
            [],
        ],
    ],
    ids=[
        "plain_case",
        "delete_blocks",
        "split_blocks",
        "skip_block",
        "no_intersect",
        "one_empty",
    ],
)
def cases(request):
    return request.param


class TestSparseIndexUnion:
    @pytest.mark.parametrize(
        "xloc, xlen, yloc, ylen, eloc, elen",
        [
            [[0], [5], [5], [4], [0], [9]],
            [[0, 10], [5, 5], [2, 17], [5, 2], [0, 10, 17], [7, 5, 2]],
            [[1], [5], [3], [5], [1], [7]],
            [[2, 10], [4, 4], [4], [8], [2], [12]],
            [[0, 5], [3, 5], [0], [7], [0], [10]],
            [[2, 10], [4, 4], [4, 13], [8, 4], [2], [15]],
            [[2], [15], [4, 9, 14], [3, 2, 2], [2], [15]],
            [[0, 10], [3, 3], [5, 15], [2, 2], [0, 5, 10, 15], [3, 2, 3, 2]],
        ],
    )
    def test_index_make_union(self, xloc, xlen, yloc, ylen, eloc, elen, test_length):
        # Case 1
        # x: ----
        # y:     ----
        # r: --------
        # Case 2
        # x: -----     -----
        # y:   -----          --
        # Case 3
        # x: ------
        # y:    -------
        # r: ----------
        # Case 4
        # x: ------  -----
        # y:    -------
        # r: -------------
        # Case 5
        # x: ---  -----
        # y: -------
        # r: -------------
        # Case 6
        # x: ------  -----
        # y:    -------  ---
        # r: -------------
        # Case 7
        # x: ----------------------
        # y:   ----  ----   ---
        # r: ----------------------
        # Case 8
        # x: ----       ---
        # y:       ---       ---
        xindex = BlockIndex(test_length, xloc, xlen)
        yindex = BlockIndex(test_length, yloc, ylen)
        bresult = xindex.make_union(yindex)
        assert isinstance(bresult, BlockIndex)
        tm.assert_numpy_array_equal(bresult.blocs, np.array(eloc, dtype=np.int32))
        tm.assert_numpy_array_equal(bresult.blengths, np.array(elen, dtype=np.int32))

        ixindex = xindex.to_int_index()
        iyindex = yindex.to_int_index()
        iresult = ixindex.make_union(iyindex)
        assert isinstance(iresult, IntIndex)
        tm.assert_numpy_array_equal(iresult.indices, bresult.to_int_index().indices)

    def test_int_index_make_union(self):
        a = IntIndex(5, np.array([0, 3, 4], dtype=np.int32))
        b = IntIndex(5, np.array([0, 2], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([0, 2, 3, 4], np.int32))
        assert res.equals(exp)

        a = IntIndex(5, np.array([], dtype=np.int32))
        b = IntIndex(5, np.array([0, 2], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([0, 2], np.int32))
        assert res.equals(exp)

        a = IntIndex(5, np.array([], dtype=np.int32))
        b = IntIndex(5, np.array([], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([], np.int32))
        assert res.equals(exp)

        a = IntIndex(5, np.array([0, 1, 2, 3, 4], dtype=np.int32))
        b = IntIndex(5, np.array([0, 1, 2, 3, 4], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([0, 1, 2, 3, 4], np.int32))
        assert res.equals(exp)

        a = IntIndex(5, np.array([0, 1], dtype=np.int32))
        b = IntIndex(4, np.array([0, 1], dtype=np.int32))

        msg = "Indices must reference same underlying length"
        with pytest.raises(ValueError, match=msg):
            a.make_union(b)


class TestSparseIndexIntersect:
    @td.skip_if_windows
    def test_intersect(self, cases, test_length):
        xloc, xlen, yloc, ylen, eloc, elen = cases
        xindex = BlockIndex(test_length, xloc, xlen)
        yindex = BlockIndex(test_length, yloc, ylen)
        expected = BlockIndex(test_length, eloc, elen)
        longer_index = BlockIndex(test_length + 1, yloc, ylen)

        result = xindex.intersect(yindex)
        assert result.equals(expected)
        result = xindex.to_int_index().intersect(yindex.to_int_index())
        assert result.equals(expected.to_int_index())

        msg = "Indices must reference same underlying length"
        with pytest.raises(Exception, match=msg):
            xindex.intersect(longer_index)
        with pytest.raises(Exception, match=msg):
            xindex.to_int_index().intersect(longer_index.to_int_index())

    def test_intersect_empty(self):
        xindex = IntIndex(4, np.array([], dtype=np.int32))
        yindex = IntIndex(4, np.array([2, 3], dtype=np.int32))
        assert xindex.intersect(yindex).equals(xindex)
        assert yindex.intersect(xindex).equals(xindex)

        xindex = xindex.to_block_index()
        yindex = yindex.to_block_index()
        assert xindex.intersect(yindex).equals(xindex)
        assert yindex.intersect(xindex).equals(xindex)

    @pytest.mark.parametrize(
        "case",
        [
            # Argument 2 to "IntIndex" has incompatible type "ndarray[Any,
            # dtype[signedinteger[_32Bit]]]"; expected "Sequence[int]"
            IntIndex(5, np.array([1, 2], dtype=np.int32)),  # type: ignore[arg-type]
            IntIndex(5, np.array([0, 2, 4], dtype=np.int32)),  # type: ignore[arg-type]
            IntIndex(0, np.array([], dtype=np.int32)),  # type: ignore[arg-type]
            IntIndex(5, np.array([], dtype=np.int32)),  # type: ignore[arg-type]
        ],
    )
    def test_intersect_identical(self, case):
        assert case.intersect(case).equals(case)
        case = case.to_block_index()
        assert case.intersect(case).equals(case)


class TestSparseIndexCommon:
    def test_int_internal(self):
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind="integer")
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.indices, np.array([2, 3], dtype=np.int32))

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind="integer")
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.indices, np.array([], dtype=np.int32))

        idx = make_sparse_index(
            4, np.array([0, 1, 2, 3], dtype=np.int32), kind="integer"
        )
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.indices, np.array([0, 1, 2, 3], dtype=np.int32))

    def test_block_internal(self):
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.blocs, np.array([2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([2], dtype=np.int32))

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.blocs, np.array([], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([], dtype=np.int32))

        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.blocs, np.array([0], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([4], dtype=np.int32))

        idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 3
        tm.assert_numpy_array_equal(idx.blocs, np.array([0, 2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([1, 2], dtype=np.int32))

    @pytest.mark.parametrize("kind", ["integer", "block"])
    def test_lookup(self, kind):
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind=kind)
        assert idx.lookup(-1) == -1
        assert idx.lookup(0) == -1
        assert idx.lookup(1) == -1
        assert idx.lookup(2) == 0
        assert idx.lookup(3) == 1
        assert idx.lookup(4) == -1

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind=kind)

        for i in range(-1, 5):
            assert idx.lookup(i) == -1

        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind=kind)
        assert idx.lookup(-1) == -1
        assert idx.lookup(0) == 0
        assert idx.lookup(1) == 1
        assert idx.lookup(2) == 2
        assert idx.lookup(3) == 3
        assert idx.lookup(4) == -1

        idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind=kind)
        assert idx.lookup(-1) == -1
        assert idx.lookup(0) == 0
        assert idx.lookup(1) == -1
        assert idx.lookup(2) == 1
        assert idx.lookup(3) == 2
        assert idx.lookup(4) == -1

    @pytest.mark.parametrize("kind", ["integer", "block"])
    def test_lookup_array(self, kind):
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind=kind)

        res = idx.lookup_array(np.array([-1, 0, 2], dtype=np.int32))
        exp = np.array([-1, -1, 0], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)

        res = idx.lookup_array(np.array([4, 2, 1, 3], dtype=np.int32))
        exp = np.array([-1, 0, -1, 1], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind=kind)
        res = idx.lookup_array(np.array([-1, 0, 2, 4], dtype=np.int32))
        exp = np.array([-1, -1, -1, -1], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)

        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind=kind)
        res = idx.lookup_array(np.array([-1, 0, 2], dtype=np.int32))
        exp = np.array([-1, 0, 2], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)

        res = idx.lookup_array(np.array([4, 2, 1, 3], dtype=np.int32))
        exp = np.array([-1, 2, 1, 3], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)

        idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind=kind)
        res = idx.lookup_array(np.array([2, 1, 3, 0], dtype=np.int32))
        exp = np.array([1, -1, 2, 0], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)

        res = idx.lookup_array(np.array([1, 4, 2, 5], dtype=np.int32))
        exp = np.array([-1, -1, 1, -1], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)

    @pytest.mark.parametrize(
        "idx, expected",
        [
            [0, -1],
            [5, 0],
            [7, 2],
            [8, -1],
            [9, -1],
            [10, -1],
            [11, -1],
            [12, 3],
            [17, 8],
            [18, -1],
        ],
    )
    def test_lookup_basics(self, idx, expected):
        bindex = BlockIndex(20, [5, 12], [3, 6])
        assert bindex.lookup(idx) == expected

        iindex = bindex.to_int_index()
        assert iindex.lookup(idx) == expected


class TestBlockIndex:
    def test_block_internal(self):
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.blocs, np.array([2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([2], dtype=np.int32))

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.blocs, np.array([], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([], dtype=np.int32))

        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.blocs, np.array([0], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([4], dtype=np.int32))

        idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 3
        tm.assert_numpy_array_equal(idx.blocs, np.array([0, 2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([1, 2], dtype=np.int32))

    @pytest.mark.parametrize("i", [5, 10, 100, 101])
    def test_make_block_boundary(self, i):
        idx = make_sparse_index(i, np.arange(0, i, 2, dtype=np.int32), kind="block")

        exp = np.arange(0, i, 2, dtype=np.int32)
        tm.assert_numpy_array_equal(idx.blocs, exp)
        tm.assert_numpy_array_equal(idx.blengths, np.ones(len(exp), dtype=np.int32))

    def test_equals(self):
        index = BlockIndex(10, [0, 4], [2, 5])

        assert index.equals(index)
        assert not index.equals(BlockIndex(10, [0, 4], [2, 6]))

    def test_check_integrity(self):
        locs = []
        lengths = []

        # 0-length OK
        BlockIndex(0, locs, lengths)

        # also OK even though empty
        BlockIndex(1, locs, lengths)

        msg = "Block 0 extends beyond end"
        with pytest.raises(ValueError, match=msg):
            BlockIndex(10, [5], [10])

        msg = "Block 0 overlaps"
        with pytest.raises(ValueError, match=msg):
            BlockIndex(10, [2, 5], [5, 3])

    def test_to_int_index(self):
        locs = [0, 10]
        lengths = [4, 6]
        exp_inds = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15]

        block = BlockIndex(20, locs, lengths)
        dense = block.to_int_index()

        tm.assert_numpy_array_equal(dense.indices, np.array(exp_inds, dtype=np.int32))

    def test_to_block_index(self):
        index = BlockIndex(10, [0, 5], [4, 5])
        assert index.to_block_index() is index


class TestIntIndex:
    def test_check_integrity(self):
        # Too many indices than specified in self.length
        msg = "Too many indices"

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=1, indices=[1, 2, 3])

        # No index can be negative.
        msg = "No index can be less than zero"

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, -2, 3])

        # No index can be negative.
        msg = "No index can be less than zero"

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, -2, 3])

        # All indices must be less than the length.
        msg = "All indices must be less than the length"

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 2, 5])

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 2, 6])

        # Indices must be strictly ascending.
        msg = "Indices must be strictly increasing"

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 3, 2])

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 3, 3])

    def test_int_internal(self):
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind="integer")
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.indices, np.array([2, 3], dtype=np.int32))

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind="integer")
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.indices, np.array([], dtype=np.int32))

        idx = make_sparse_index(
            4, np.array([0, 1, 2, 3], dtype=np.int32), kind="integer"
        )
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.indices, np.array([0, 1, 2, 3], dtype=np.int32))

    def test_equals(self):
        index = IntIndex(10, [0, 1, 2, 3, 4])
        assert index.equals(index)
        assert not index.equals(IntIndex(10, [0, 1, 2, 3]))

    def test_to_block_index(self, cases, test_length):
        xloc, xlen, yloc, ylen, _, _ = cases
        xindex = BlockIndex(test_length, xloc, xlen)
        yindex = BlockIndex(test_length, yloc, ylen)

        # see if survive the round trip
        xbindex = xindex.to_int_index().to_block_index()
        ybindex = yindex.to_int_index().to_block_index()
        assert isinstance(xbindex, BlockIndex)
        assert xbindex.equals(xindex)
        assert ybindex.equals(yindex)

    def test_to_int_index(self):
        index = IntIndex(10, [2, 3, 4, 5, 6])
        assert index.to_int_index() is index


class TestSparseOperators:
    @pytest.mark.parametrize("opname", ["add", "sub", "mul", "truediv", "floordiv"])
    def test_op(self, opname, cases, test_length):
        xloc, xlen, yloc, ylen, _, _ = cases
        sparse_op = getattr(splib, f"sparse_{opname}_float64")
        python_op = getattr(operator, opname)

        xindex = BlockIndex(test_length, xloc, xlen)
        yindex = BlockIndex(test_length, yloc, ylen)

        xdindex = xindex.to_int_index()
        ydindex = yindex.to_int_index()

        x = np.arange(xindex.npoints) * 10.0 + 1
        y = np.arange(yindex.npoints) * 100.0 + 1

        xfill = 0
        yfill = 2

        result_block_vals, rb_index, bfill = sparse_op(
            x, xindex, xfill, y, yindex, yfill
        )
        result_int_vals, ri_index, ifill = sparse_op(
            x, xdindex, xfill, y, ydindex, yfill
        )

        assert rb_index.to_int_index().equals(ri_index)
        tm.assert_numpy_array_equal(result_block_vals, result_int_vals)
        assert bfill == ifill

        # check versus Series...
        xseries = Series(x, xdindex.indices)
        xseries = xseries.reindex(np.arange(test_length)).fillna(xfill)

        yseries = Series(y, ydindex.indices)
        yseries = yseries.reindex(np.arange(test_length)).fillna(yfill)

        series_result = python_op(xseries, yseries)
        series_result = series_result.reindex(ri_index.indices)

        tm.assert_numpy_array_equal(result_block_vals, series_result.values)
        tm.assert_numpy_array_equal(result_int_vals, series_result.values)