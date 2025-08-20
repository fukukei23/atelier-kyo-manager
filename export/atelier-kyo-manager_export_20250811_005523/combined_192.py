
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\style\test_to_latex.py ===
from textwrap import dedent

import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    option_context,
)

pytest.importorskip("jinja2")
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
    _parse_latex_cell_styles,
    _parse_latex_css_conversion,
    _parse_latex_header_span,
    _parse_latex_table_styles,
    _parse_latex_table_wrapping,
)


@pytest.fixture
def df():
    return DataFrame(
        {"A": [0, 1], "B": [-0.61, -1.22], "C": Series(["ab", "cd"], dtype=object)}
    )


@pytest.fixture
def df_ext():
    return DataFrame(
        {"A": [0, 1, 2], "B": [-0.61, -1.22, -2.22], "C": ["ab", "cd", "de"]}
    )


@pytest.fixture
def styler(df):
    return Styler(df, uuid_len=0, precision=2)


def test_minimal_latex_tabular(styler):
    expected = dedent(
        """\
        \\begin{tabular}{lrrl}
         & A & B & C \\\\
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\end{tabular}
        """
    )
    assert styler.to_latex() == expected


def test_tabular_hrules(styler):
    expected = dedent(
        """\
        \\begin{tabular}{lrrl}
        \\toprule
         & A & B & C \\\\
        \\midrule
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\bottomrule
        \\end{tabular}
        """
    )
    assert styler.to_latex(hrules=True) == expected


def test_tabular_custom_hrules(styler):
    styler.set_table_styles(
        [
            {"selector": "toprule", "props": ":hline"},
            {"selector": "bottomrule", "props": ":otherline"},
        ]
    )  # no midrule
    expected = dedent(
        """\
        \\begin{tabular}{lrrl}
        \\hline
         & A & B & C \\\\
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\otherline
        \\end{tabular}
        """
    )
    assert styler.to_latex() == expected


def test_column_format(styler):
    # default setting is already tested in `test_latex_minimal_tabular`
    styler.set_table_styles([{"selector": "column_format", "props": ":cccc"}])

    assert "\\begin{tabular}{rrrr}" in styler.to_latex(column_format="rrrr")
    styler.set_table_styles([{"selector": "column_format", "props": ":r|r|cc"}])
    assert "\\begin{tabular}{r|r|cc}" in styler.to_latex()


def test_siunitx_cols(styler):
    expected = dedent(
        """\
        \\begin{tabular}{lSSl}
        {} & {A} & {B} & {C} \\\\
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\end{tabular}
        """
    )
    assert styler.to_latex(siunitx=True) == expected


def test_position(styler):
    assert "\\begin{table}[h!]" in styler.to_latex(position="h!")
    assert "\\end{table}" in styler.to_latex(position="h!")
    styler.set_table_styles([{"selector": "position", "props": ":b!"}])
    assert "\\begin{table}[b!]" in styler.to_latex()
    assert "\\end{table}" in styler.to_latex()


@pytest.mark.parametrize("env", [None, "longtable"])
def test_label(styler, env):
    assert "\n\\label{text}" in styler.to_latex(label="text", environment=env)
    styler.set_table_styles([{"selector": "label", "props": ":{more §text}"}])
    assert "\n\\label{more :text}" in styler.to_latex(environment=env)


def test_position_float_raises(styler):
    msg = "`position_float` should be one of 'raggedright', 'raggedleft', 'centering',"
    with pytest.raises(ValueError, match=msg):
        styler.to_latex(position_float="bad_string")

    msg = "`position_float` cannot be used in 'longtable' `environment`"
    with pytest.raises(ValueError, match=msg):
        styler.to_latex(position_float="centering", environment="longtable")


@pytest.mark.parametrize("label", [(None, ""), ("text", "\\label{text}")])
@pytest.mark.parametrize("position", [(None, ""), ("h!", "{table}[h!]")])
@pytest.mark.parametrize("caption", [(None, ""), ("text", "\\caption{text}")])
@pytest.mark.parametrize("column_format", [(None, ""), ("rcrl", "{tabular}{rcrl}")])
@pytest.mark.parametrize("position_float", [(None, ""), ("centering", "\\centering")])
def test_kwargs_combinations(
    styler, label, position, caption, column_format, position_float
):
    result = styler.to_latex(
        label=label[0],
        position=position[0],
        caption=caption[0],
        column_format=column_format[0],
        position_float=position_float[0],
    )
    assert label[1] in result
    assert position[1] in result
    assert caption[1] in result
    assert column_format[1] in result
    assert position_float[1] in result


def test_custom_table_styles(styler):
    styler.set_table_styles(
        [
            {"selector": "mycommand", "props": ":{myoptions}"},
            {"selector": "mycommand2", "props": ":{myoptions2}"},
        ]
    )
    expected = dedent(
        """\
        \\begin{table}
        \\mycommand{myoptions}
        \\mycommand2{myoptions2}
        """
    )
    assert expected in styler.to_latex()


def test_cell_styling(styler):
    styler.highlight_max(props="itshape:;Huge:--wrap;")
    expected = dedent(
        """\
        \\begin{tabular}{lrrl}
         & A & B & C \\\\
        0 & 0 & \\itshape {\\Huge -0.61} & ab \\\\
        1 & \\itshape {\\Huge 1} & -1.22 & \\itshape {\\Huge cd} \\\\
        \\end{tabular}
        """
    )
    assert expected == styler.to_latex()


def test_multiindex_columns(df):
    cidx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df.columns = cidx
    expected = dedent(
        """\
        \\begin{tabular}{lrrl}
         & \\multicolumn{2}{r}{A} & B \\\\
         & a & b & c \\\\
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\end{tabular}
        """
    )
    s = df.style.format(precision=2)
    assert expected == s.to_latex()

    # non-sparse
    expected = dedent(
        """\
        \\begin{tabular}{lrrl}
         & A & A & B \\\\
         & a & b & c \\\\
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\end{tabular}
        """
    )
    s = df.style.format(precision=2)
    assert expected == s.to_latex(sparse_columns=False)


def test_multiindex_row(df_ext):
    ridx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df_ext.index = ridx
    expected = dedent(
        """\
        \\begin{tabular}{llrrl}
         &  & A & B & C \\\\
        \\multirow[c]{2}{*}{A} & a & 0 & -0.61 & ab \\\\
         & b & 1 & -1.22 & cd \\\\
        B & c & 2 & -2.22 & de \\\\
        \\end{tabular}
        """
    )
    styler = df_ext.style.format(precision=2)
    result = styler.to_latex()
    assert expected == result

    # non-sparse
    expected = dedent(
        """\
        \\begin{tabular}{llrrl}
         &  & A & B & C \\\\
        A & a & 0 & -0.61 & ab \\\\
        A & b & 1 & -1.22 & cd \\\\
        B & c & 2 & -2.22 & de \\\\
        \\end{tabular}
        """
    )
    result = styler.to_latex(sparse_index=False)
    assert expected == result


def test_multirow_naive(df_ext):
    ridx = MultiIndex.from_tuples([("X", "x"), ("X", "y"), ("Y", "z")])
    df_ext.index = ridx
    expected = dedent(
        """\
        \\begin{tabular}{llrrl}
         &  & A & B & C \\\\
        X & x & 0 & -0.61 & ab \\\\
         & y & 1 & -1.22 & cd \\\\
        Y & z & 2 & -2.22 & de \\\\
        \\end{tabular}
        """
    )
    styler = df_ext.style.format(precision=2)
    result = styler.to_latex(multirow_align="naive")
    assert expected == result


def test_multiindex_row_and_col(df_ext):
    cidx = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    ridx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df_ext.index, df_ext.columns = ridx, cidx
    expected = dedent(
        """\
        \\begin{tabular}{llrrl}
         &  & \\multicolumn{2}{l}{Z} & Y \\\\
         &  & a & b & c \\\\
        \\multirow[b]{2}{*}{A} & a & 0 & -0.61 & ab \\\\
         & b & 1 & -1.22 & cd \\\\
        B & c & 2 & -2.22 & de \\\\
        \\end{tabular}
        """
    )
    styler = df_ext.style.format(precision=2)
    result = styler.to_latex(multirow_align="b", multicol_align="l")
    assert result == expected

    # non-sparse
    expected = dedent(
        """\
        \\begin{tabular}{llrrl}
         &  & Z & Z & Y \\\\
         &  & a & b & c \\\\
        A & a & 0 & -0.61 & ab \\\\
        A & b & 1 & -1.22 & cd \\\\
        B & c & 2 & -2.22 & de \\\\
        \\end{tabular}
        """
    )
    result = styler.to_latex(sparse_index=False, sparse_columns=False)
    assert result == expected


@pytest.mark.parametrize(
    "multicol_align, siunitx, header",
    [
        ("naive-l", False, " & A & &"),
        ("naive-r", False, " & & & A"),
        ("naive-l", True, "{} & {A} & {} & {}"),
        ("naive-r", True, "{} & {} & {} & {A}"),
    ],
)
def test_multicol_naive(df, multicol_align, siunitx, header):
    ridx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("A", "c")])
    df.columns = ridx
    level1 = " & a & b & c" if not siunitx else "{} & {a} & {b} & {c}"
    col_format = "lrrl" if not siunitx else "lSSl"
    expected = dedent(
        f"""\
        \\begin{{tabular}}{{{col_format}}}
        {header} \\\\
        {level1} \\\\
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\end{{tabular}}
        """
    )
    styler = df.style.format(precision=2)
    result = styler.to_latex(multicol_align=multicol_align, siunitx=siunitx)
    assert expected == result


def test_multi_options(df_ext):
    cidx = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    ridx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df_ext.index, df_ext.columns = ridx, cidx
    styler = df_ext.style.format(precision=2)

    expected = dedent(
        """\
     &  & \\multicolumn{2}{r}{Z} & Y \\\\
     &  & a & b & c \\\\
    \\multirow[c]{2}{*}{A} & a & 0 & -0.61 & ab \\\\
    """
    )
    result = styler.to_latex()
    assert expected in result

    with option_context("styler.latex.multicol_align", "l"):
        assert " &  & \\multicolumn{2}{l}{Z} & Y \\\\" in styler.to_latex()

    with option_context("styler.latex.multirow_align", "b"):
        assert "\\multirow[b]{2}{*}{A} & a & 0 & -0.61 & ab \\\\" in styler.to_latex()


def test_multiindex_columns_hidden():
    df = DataFrame([[1, 2, 3, 4]])
    df.columns = MultiIndex.from_tuples([("A", 1), ("A", 2), ("A", 3), ("B", 1)])
    s = df.style
    assert "{tabular}{lrrrr}" in s.to_latex()
    s.set_table_styles([])  # reset the position command
    s.hide([("A", 2)], axis="columns")
    assert "{tabular}{lrrr}" in s.to_latex()


@pytest.mark.parametrize(
    "option, value",
    [
        ("styler.sparse.index", True),
        ("styler.sparse.index", False),
        ("styler.sparse.columns", True),
        ("styler.sparse.columns", False),
    ],
)
def test_sparse_options(df_ext, option, value):
    cidx = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    ridx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df_ext.index, df_ext.columns = ridx, cidx
    styler = df_ext.style

    latex1 = styler.to_latex()
    with option_context(option, value):
        latex2 = styler.to_latex()
    assert (latex1 == latex2) is value


def test_hidden_index(styler):
    styler.hide(axis="index")
    expected = dedent(
        """\
        \\begin{tabular}{rrl}
        A & B & C \\\\
        0 & -0.61 & ab \\\\
        1 & -1.22 & cd \\\\
        \\end{tabular}
        """
    )
    assert styler.to_latex() == expected


@pytest.mark.parametrize("environment", ["table", "figure*", None])
def test_comprehensive(df_ext, environment):
    # test as many low level features simultaneously as possible
    cidx = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    ridx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df_ext.index, df_ext.columns = ridx, cidx
    stlr = df_ext.style
    stlr.set_caption("mycap")
    stlr.set_table_styles(
        [
            {"selector": "label", "props": ":{fig§item}"},
            {"selector": "position", "props": ":h!"},
            {"selector": "position_float", "props": ":centering"},
            {"selector": "column_format", "props": ":rlrlr"},
            {"selector": "toprule", "props": ":toprule"},
            {"selector": "midrule", "props": ":midrule"},
            {"selector": "bottomrule", "props": ":bottomrule"},
            {"selector": "rowcolors", "props": ":{3}{pink}{}"},  # custom command
        ]
    )
    stlr.highlight_max(axis=0, props="textbf:--rwrap;cellcolor:[rgb]{1,1,0.6}--rwrap")
    stlr.highlight_max(axis=None, props="Huge:--wrap;", subset=[("Z", "a"), ("Z", "b")])

    expected = (
        """\
\\begin{table}[h!]
\\centering
\\caption{mycap}
\\label{fig:item}
\\rowcolors{3}{pink}{}
\\begin{tabular}{rlrlr}
\\toprule
 &  & \\multicolumn{2}{r}{Z} & Y \\\\
 &  & a & b & c \\\\
\\midrule
\\multirow[c]{2}{*}{A} & a & 0 & \\textbf{\\cellcolor[rgb]{1,1,0.6}{-0.61}} & ab \\\\
 & b & 1 & -1.22 & cd \\\\
B & c & \\textbf{\\cellcolor[rgb]{1,1,0.6}{{\\Huge 2}}} & -2.22 & """
        """\
\\textbf{\\cellcolor[rgb]{1,1,0.6}{de}} \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
    ).replace("table", environment if environment else "table")
    result = stlr.format(precision=2).to_latex(environment=environment)
    assert result == expected


def test_environment_option(styler):
    with option_context("styler.latex.environment", "bar-env"):
        assert "\\begin{bar-env}" in styler.to_latex()
        assert "\\begin{foo-env}" in styler.to_latex(environment="foo-env")


def test_parse_latex_table_styles(styler):
    styler.set_table_styles(
        [
            {"selector": "foo", "props": [("attr", "value")]},
            {"selector": "bar", "props": [("attr", "overwritten")]},
            {"selector": "bar", "props": [("attr", "baz"), ("attr2", "ignored")]},
            {"selector": "label", "props": [("", "{fig§item}")]},
        ]
    )
    assert _parse_latex_table_styles(styler.table_styles, "bar") == "baz"

    # test '§' replaced by ':' [for CSS compatibility]
    assert _parse_latex_table_styles(styler.table_styles, "label") == "{fig:item}"


def test_parse_latex_cell_styles_basic():  # test nesting
    cell_style = [("itshape", "--rwrap"), ("cellcolor", "[rgb]{0,1,1}--rwrap")]
    expected = "\\itshape{\\cellcolor[rgb]{0,1,1}{text}}"
    assert _parse_latex_cell_styles(cell_style, "text") == expected


@pytest.mark.parametrize(
    "wrap_arg, expected",
    [  # test wrapping
        ("", "\\<command><options> <display_value>"),
        ("--wrap", "{\\<command><options> <display_value>}"),
        ("--nowrap", "\\<command><options> <display_value>"),
        ("--lwrap", "{\\<command><options>} <display_value>"),
        ("--dwrap", "{\\<command><options>}{<display_value>}"),
        ("--rwrap", "\\<command><options>{<display_value>}"),
    ],
)
def test_parse_latex_cell_styles_braces(wrap_arg, expected):
    cell_style = [("<command>", f"<options>{wrap_arg}")]
    assert _parse_latex_cell_styles(cell_style, "<display_value>") == expected


def test_parse_latex_header_span():
    cell = {"attributes": 'colspan="3"', "display_value": "text", "cellstyle": []}
    expected = "\\multicolumn{3}{Y}{text}"
    assert _parse_latex_header_span(cell, "X", "Y") == expected

    cell = {"attributes": 'rowspan="5"', "display_value": "text", "cellstyle": []}
    expected = "\\multirow[X]{5}{*}{text}"
    assert _parse_latex_header_span(cell, "X", "Y") == expected

    cell = {"display_value": "text", "cellstyle": []}
    assert _parse_latex_header_span(cell, "X", "Y") == "text"

    cell = {"display_value": "text", "cellstyle": [("bfseries", "--rwrap")]}
    assert _parse_latex_header_span(cell, "X", "Y") == "\\bfseries{text}"


def test_parse_latex_table_wrapping(styler):
    styler.set_table_styles(
        [
            {"selector": "toprule", "props": ":value"},
            {"selector": "bottomrule", "props": ":value"},
            {"selector": "midrule", "props": ":value"},
            {"selector": "column_format", "props": ":value"},
        ]
    )
    assert _parse_latex_table_wrapping(styler.table_styles, styler.caption) is False
    assert _parse_latex_table_wrapping(styler.table_styles, "some caption") is True
    styler.set_table_styles(
        [
            {"selector": "not-ignored", "props": ":value"},
        ],
        overwrite=False,
    )
    assert _parse_latex_table_wrapping(styler.table_styles, None) is True


def test_short_caption(styler):
    result = styler.to_latex(caption=("full cap", "short cap"))
    assert "\\caption[short cap]{full cap}" in result


@pytest.mark.parametrize(
    "css, expected",
    [
        ([("color", "red")], [("color", "{red}")]),  # test color and input format types
        (
            [("color", "rgb(128, 128, 128 )")],
            [("color", "[rgb]{0.502, 0.502, 0.502}")],
        ),
        (
            [("color", "rgb(128, 50%, 25% )")],
            [("color", "[rgb]{0.502, 0.500, 0.250}")],
        ),
        (
            [("color", "rgba(128,128,128,1)")],
            [("color", "[rgb]{0.502, 0.502, 0.502}")],
        ),
        ([("color", "#FF00FF")], [("color", "[HTML]{FF00FF}")]),
        ([("color", "#F0F")], [("color", "[HTML]{FF00FF}")]),
        ([("font-weight", "bold")], [("bfseries", "")]),  # test font-weight and types
        ([("font-weight", "bolder")], [("bfseries", "")]),
        ([("font-weight", "normal")], []),
        ([("background-color", "red")], [("cellcolor", "{red}--lwrap")]),
        (
            [("background-color", "#FF00FF")],  # test background-color command and wrap
            [("cellcolor", "[HTML]{FF00FF}--lwrap")],
        ),
        ([("font-style", "italic")], [("itshape", "")]),  # test font-style and types
        ([("font-style", "oblique")], [("slshape", "")]),
        ([("font-style", "normal")], []),
        ([("color", "red /*--dwrap*/")], [("color", "{red}--dwrap")]),  # css comments
        ([("background-color", "red /* --dwrap */")], [("cellcolor", "{red}--dwrap")]),
    ],
)
def test_parse_latex_css_conversion(css, expected):
    result = _parse_latex_css_conversion(css)
    assert result == expected


@pytest.mark.parametrize(
    "env, inner_env",
    [
        (None, "tabular"),
        ("table", "tabular"),
        ("longtable", "longtable"),
    ],
)
@pytest.mark.parametrize(
    "convert, exp", [(True, "bfseries"), (False, "font-weightbold")]
)
def test_parse_latex_css_convert_minimal(styler, env, inner_env, convert, exp):
    # parameters ensure longtable template is also tested
    styler.highlight_max(props="font-weight:bold;")
    result = styler.to_latex(convert_css=convert, environment=env)
    expected = dedent(
        f"""\
        0 & 0 & \\{exp} -0.61 & ab \\\\
        1 & \\{exp} 1 & -1.22 & \\{exp} cd \\\\
        \\end{{{inner_env}}}
    """
    )
    assert expected in result


def test_parse_latex_css_conversion_option():
    css = [("command", "option--latex--wrap")]
    expected = [("command", "option--wrap")]
    result = _parse_latex_css_conversion(css)
    assert result == expected


def test_styler_object_after_render(styler):
    # GH 42320
    pre_render = styler._copy(deepcopy=True)
    styler.to_latex(
        column_format="rllr",
        position="h",
        position_float="centering",
        hrules=True,
        label="my lab",
        caption="my cap",
    )

    assert pre_render.table_styles == styler.table_styles
    assert pre_render.caption == styler.caption


def test_longtable_comprehensive(styler):
    result = styler.to_latex(
        environment="longtable", hrules=True, label="fig:A", caption=("full", "short")
    )
    expected = dedent(
        """\
        \\begin{longtable}{lrrl}
        \\caption[short]{full} \\label{fig:A} \\\\
        \\toprule
         & A & B & C \\\\
        \\midrule
        \\endfirsthead
        \\caption[]{full} \\\\
        \\toprule
         & A & B & C \\\\
        \\midrule
        \\endhead
        \\midrule
        \\multicolumn{4}{r}{Continued on next page} \\\\
        \\midrule
        \\endfoot
        \\bottomrule
        \\endlastfoot
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\end{longtable}
    """
    )
    assert result == expected


def test_longtable_minimal(styler):
    result = styler.to_latex(environment="longtable")
    expected = dedent(
        """\
        \\begin{longtable}{lrrl}
         & A & B & C \\\\
        \\endfirsthead
         & A & B & C \\\\
        \\endhead
        \\multicolumn{4}{r}{Continued on next page} \\\\
        \\endfoot
        \\endlastfoot
        0 & 0 & -0.61 & ab \\\\
        1 & 1 & -1.22 & cd \\\\
        \\end{longtable}
    """
    )
    assert result == expected


@pytest.mark.parametrize(
    "sparse, exp, siunitx",
    [
        (True, "{} & \\multicolumn{2}{r}{A} & {B}", True),
        (False, "{} & {A} & {A} & {B}", True),
        (True, " & \\multicolumn{2}{r}{A} & B", False),
        (False, " & A & A & B", False),
    ],
)
def test_longtable_multiindex_columns(df, sparse, exp, siunitx):
    cidx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df.columns = cidx
    with_si = "{} & {a} & {b} & {c} \\\\"
    without_si = " & a & b & c \\\\"
    expected = dedent(
        f"""\
        \\begin{{longtable}}{{l{"SS" if siunitx else "rr"}l}}
        {exp} \\\\
        {with_si if siunitx else without_si}
        \\endfirsthead
        {exp} \\\\
        {with_si if siunitx else without_si}
        \\endhead
        """
    )
    result = df.style.to_latex(
        environment="longtable", sparse_columns=sparse, siunitx=siunitx
    )
    assert expected in result


@pytest.mark.parametrize(
    "caption, cap_exp",
    [
        ("full", ("{full}", "")),
        (("full", "short"), ("{full}", "[short]")),
    ],
)
@pytest.mark.parametrize("label, lab_exp", [(None, ""), ("tab:A", " \\label{tab:A}")])
def test_longtable_caption_label(styler, caption, cap_exp, label, lab_exp):
    cap_exp1 = f"\\caption{cap_exp[1]}{cap_exp[0]}"
    cap_exp2 = f"\\caption[]{cap_exp[0]}"

    expected = dedent(
        f"""\
        {cap_exp1}{lab_exp} \\\\
         & A & B & C \\\\
        \\endfirsthead
        {cap_exp2} \\\\
        """
    )
    assert expected in styler.to_latex(
        environment="longtable", caption=caption, label=label
    )


@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize(
    "columns, siunitx",
    [
        (True, True),
        (True, False),
        (False, False),
    ],
)
def test_apply_map_header_render_mi(df_ext, index, columns, siunitx):
    cidx = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    ridx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df_ext.index, df_ext.columns = ridx, cidx
    styler = df_ext.style

    func = lambda v: "bfseries: --rwrap" if "A" in v or "Z" in v or "c" in v else None

    if index:
        styler.map_index(func, axis="index")
    if columns:
        styler.map_index(func, axis="columns")

    result = styler.to_latex(siunitx=siunitx)

    expected_index = dedent(
        """\
    \\multirow[c]{2}{*}{\\bfseries{A}} & a & 0 & -0.610000 & ab \\\\
    \\bfseries{} & b & 1 & -1.220000 & cd \\\\
    B & \\bfseries{c} & 2 & -2.220000 & de \\\\
    """
    )
    assert (expected_index in result) is index

    exp_cols_si = dedent(
        """\
    {} & {} & \\multicolumn{2}{r}{\\bfseries{Z}} & {Y} \\\\
    {} & {} & {a} & {b} & {\\bfseries{c}} \\\\
    """
    )
    exp_cols_no_si = """\
 &  & \\multicolumn{2}{r}{\\bfseries{Z}} & Y \\\\
 &  & a & b & \\bfseries{c} \\\\
"""
    assert ((exp_cols_si if siunitx else exp_cols_no_si) in result) is columns


def test_repr_option(styler):
    assert "<style" in styler._repr_html_()[:6]
    assert styler._repr_latex_() is None
    with option_context("styler.render.repr", "latex"):
        assert "\\begin{tabular}" in styler._repr_latex_()[:15]
        assert styler._repr_html_() is None


@pytest.mark.parametrize("option", ["hrules"])
def test_bool_options(styler, option):
    with option_context(f"styler.latex.{option}", False):
        latex_false = styler.to_latex()
    with option_context(f"styler.latex.{option}", True):
        latex_true = styler.to_latex()
    assert latex_false != latex_true  # options are reactive under to_latex(*no_args)


def test_siunitx_basic_headers(styler):
    assert "{} & {A} & {B} & {C} \\\\" in styler.to_latex(siunitx=True)
    assert " & A & B & C \\\\" in styler.to_latex()  # default siunitx=False


@pytest.mark.parametrize("axis", ["index", "columns"])
def test_css_convert_apply_index(styler, axis):
    styler.map_index(lambda x: "font-weight: bold;", axis=axis)
    for label in getattr(styler, axis):
        assert f"\\bfseries {label}" in styler.to_latex(convert_css=True)


def test_hide_index_latex(styler):
    # GH 43637
    styler.hide([0], axis=0)
    result = styler.to_latex()
    expected = dedent(
        """\
    \\begin{tabular}{lrrl}
     & A & B & C \\\\
    1 & 1 & -1.22 & cd \\\\
    \\end{tabular}
    """
    )
    assert expected == result


def test_latex_hiding_index_columns_multiindex_alignment():
    # gh 43644
    midx = MultiIndex.from_product(
        [["i0", "j0"], ["i1"], ["i2", "j2"]], names=["i-0", "i-1", "i-2"]
    )
    cidx = MultiIndex.from_product(
        [["c0"], ["c1", "d1"], ["c2", "d2"]], names=["c-0", "c-1", "c-2"]
    )
    df = DataFrame(np.arange(16).reshape(4, 4), index=midx, columns=cidx)
    styler = Styler(df, uuid_len=0)
    styler.hide(level=1, axis=0).hide(level=0, axis=1)
    styler.hide([("i0", "i1", "i2")], axis=0)
    styler.hide([("c0", "c1", "c2")], axis=1)
    styler.map(lambda x: "color:{red};" if x == 5 else "")
    styler.map_index(lambda x: "color:{blue};" if "j" in x else "")
    result = styler.to_latex()
    expected = dedent(
        """\
        \\begin{tabular}{llrrr}
         & c-1 & c1 & \\multicolumn{2}{r}{d1} \\\\
         & c-2 & d2 & c2 & d2 \\\\
        i-0 & i-2 &  &  &  \\\\
        i0 & \\color{blue} j2 & \\color{red} 5 & 6 & 7 \\\\
        \\multirow[c]{2}{*}{\\color{blue} j0} & i2 & 9 & 10 & 11 \\\\
        \\color{blue}  & \\color{blue} j2 & 13 & 14 & 15 \\\\
        \\end{tabular}
        """
    )
    assert result == expected


def test_rendered_links():
    # note the majority of testing is done in test_html.py: test_rendered_links
    # these test only the alternative latex format is functional
    df = DataFrame(["text www.domain.com text"])
    result = df.style.format(hyperlinks="latex").to_latex()
    assert r"text \href{www.domain.com}{www.domain.com} text" in result


def test_apply_index_hidden_levels():
    # gh 45156
    styler = DataFrame(
        [[1]],
        index=MultiIndex.from_tuples([(0, 1)], names=["l0", "l1"]),
        columns=MultiIndex.from_tuples([(0, 1)], names=["c0", "c1"]),
    ).style
    styler.hide(level=1)
    styler.map_index(lambda v: "color: red;", level=0, axis=1)
    result = styler.to_latex(convert_css=True)
    expected = dedent(
        """\
        \\begin{tabular}{lr}
        c0 & \\color{red} 0 \\\\
        c1 & 1 \\\\
        l0 &  \\\\
        0 & 1 \\\\
        \\end{tabular}
        """
    )
    assert result == expected


@pytest.mark.parametrize("clines", ["bad", "index", "skip-last", "all", "data"])
def test_clines_validation(clines, styler):
    msg = f"`clines` value of {clines} is invalid."
    with pytest.raises(ValueError, match=msg):
        styler.to_latex(clines=clines)


@pytest.mark.parametrize(
    "clines, exp",
    [
        ("all;index", "\n\\cline{1-1}"),
        ("all;data", "\n\\cline{1-2}"),
        ("skip-last;index", ""),
        ("skip-last;data", ""),
        (None, ""),
    ],
)
@pytest.mark.parametrize("env", ["table", "longtable"])
def test_clines_index(clines, exp, env):
    df = DataFrame([[1], [2], [3], [4]])
    result = df.style.to_latex(clines=clines, environment=env)
    expected = f"""\
0 & 1 \\\\{exp}
1 & 2 \\\\{exp}
2 & 3 \\\\{exp}
3 & 4 \\\\{exp}
"""
    assert expected in result


@pytest.mark.parametrize(
    "clines, expected",
    [
        (
            None,
            dedent(
                """\
            \\multirow[c]{2}{*}{A} & X & 1 \\\\
             & Y & 2 \\\\
            \\multirow[c]{2}{*}{B} & X & 3 \\\\
             & Y & 4 \\\\
            """
            ),
        ),
        (
            "skip-last;index",
            dedent(
                """\
            \\multirow[c]{2}{*}{A} & X & 1 \\\\
             & Y & 2 \\\\
            \\cline{1-2}
            \\multirow[c]{2}{*}{B} & X & 3 \\\\
             & Y & 4 \\\\
            \\cline{1-2}
            """
            ),
        ),
        (
            "skip-last;data",
            dedent(
                """\
            \\multirow[c]{2}{*}{A} & X & 1 \\\\
             & Y & 2 \\\\
            \\cline{1-3}
            \\multirow[c]{2}{*}{B} & X & 3 \\\\
             & Y & 4 \\\\
            \\cline{1-3}
            """
            ),
        ),
        (
            "all;index",
            dedent(
                """\
            \\multirow[c]{2}{*}{A} & X & 1 \\\\
            \\cline{2-2}
             & Y & 2 \\\\
            \\cline{1-2} \\cline{2-2}
            \\multirow[c]{2}{*}{B} & X & 3 \\\\
            \\cline{2-2}
             & Y & 4 \\\\
            \\cline{1-2} \\cline{2-2}
            """
            ),
        ),
        (
            "all;data",
            dedent(
                """\
            \\multirow[c]{2}{*}{A} & X & 1 \\\\
            \\cline{2-3}
             & Y & 2 \\\\
            \\cline{1-3} \\cline{2-3}
            \\multirow[c]{2}{*}{B} & X & 3 \\\\
            \\cline{2-3}
             & Y & 4 \\\\
            \\cline{1-3} \\cline{2-3}
            """
            ),
        ),
    ],
)
@pytest.mark.parametrize("env", ["table"])
def test_clines_multiindex(clines, expected, env):
    # also tests simultaneously with hidden rows and a hidden multiindex level
    midx = MultiIndex.from_product([["A", "-", "B"], [0], ["X", "Y"]])
    df = DataFrame([[1], [2], [99], [99], [3], [4]], index=midx)
    styler = df.style
    styler.hide([("-", 0, "X"), ("-", 0, "Y")])
    styler.hide(level=1)
    result = styler.to_latex(clines=clines, environment=env)
    assert expected in result


def test_col_format_len(styler):
    # gh 46037
    result = styler.to_latex(environment="longtable", column_format="lrr{10cm}")
    expected = r"\multicolumn{4}{r}{Continued on next page} \\"
    assert expected in result


def test_concat(styler):
    result = styler.concat(styler.data.agg(["sum"]).style).to_latex()
    expected = dedent(
        """\
    \\begin{tabular}{lrrl}
     & A & B & C \\\\
    0 & 0 & -0.61 & ab \\\\
    1 & 1 & -1.22 & cd \\\\
    sum & 1 & -1.830000 & abcd \\\\
    \\end{tabular}
    """
    )
    assert result == expected


def test_concat_recursion():
    # tests hidden row recursion and applied styles
    styler1 = DataFrame([[1], [9]]).style.hide([1]).highlight_min(color="red")
    styler2 = DataFrame([[9], [2]]).style.hide([0]).highlight_min(color="green")
    styler3 = DataFrame([[3], [9]]).style.hide([1]).highlight_min(color="blue")

    result = styler1.concat(styler2.concat(styler3)).to_latex(convert_css=True)
    expected = dedent(
        """\
    \\begin{tabular}{lr}
     & 0 \\\\
    0 & {\\cellcolor{red}} 1 \\\\
    1 & {\\cellcolor{green}} 2 \\\\
    0 & {\\cellcolor{blue}} 3 \\\\
    \\end{tabular}
    """
    )
    assert result == expected


def test_concat_chain():
    # tests hidden row recursion and applied styles
    styler1 = DataFrame([[1], [9]]).style.hide([1]).highlight_min(color="red")
    styler2 = DataFrame([[9], [2]]).style.hide([0]).highlight_min(color="green")
    styler3 = DataFrame([[3], [9]]).style.hide([1]).highlight_min(color="blue")

    result = styler1.concat(styler2).concat(styler3).to_latex(convert_css=True)
    expected = dedent(
        """\
    \\begin{tabular}{lr}
     & 0 \\\\
    0 & {\\cellcolor{red}} 1 \\\\
    1 & {\\cellcolor{green}} 2 \\\\
    0 & {\\cellcolor{blue}} 3 \\\\
    \\end{tabular}
    """
    )
    assert result == expected


@pytest.mark.parametrize(
    "df, expected",
    [
        (
            DataFrame(),
            dedent(
                """\
            \\begin{tabular}{l}
            \\end{tabular}
            """
            ),
        ),
        (
            DataFrame(columns=["a", "b", "c"]),
            dedent(
                """\
            \\begin{tabular}{llll}
             & a & b & c \\\\
            \\end{tabular}
            """
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "clines", [None, "all;data", "all;index", "skip-last;data", "skip-last;index"]
)
def test_empty_clines(df: DataFrame, expected: str, clines: str):
    # GH 47203
    result = df.style.to_latex(clines=clines)
    assert result == expected

# === atelier-kyo-manager/multi_lang_crawler.py ===
from googletrans import Translator
from icrawler.builtin import BingImageCrawler, BaiduImageCrawler
import os

translator = Translator()

product_list = [
    "アディダス ショルダーバッグ",
    "ナイキ シューズ"
]

base_dir = './images'

for product in product_list:
    # 多言語変換
    jp = product
    en = translator.translate(product, src='ja', dest='en').text
    zh = translator.translate(product, src='ja', dest='zh-cn').text

    print(f"日本語: {jp}, 英語: {en}, 中国語: {zh}")

    # Bing（日本語・英語）
    for keyword in [jp, en]:
        folder = os.path.join(base_dir, 'bing', keyword.replace(' ', '_'))
        os.makedirs(folder, exist_ok=True)
        crawler = BingImageCrawler(storage={'root_dir': folder}, downloader_threads=8)
        crawler.crawl(keyword=keyword, max_num=10)

    # Baidu（中国語推奨）
    folder = os.path.join(base_dir, 'baidu', zh.replace(' ', '_'))
    os.makedirs(folder, exist_ok=True)
    crawler = BaiduImageCrawler(storage={'root_dir': folder}, downloader_threads=8)
    crawler.crawl(keyword=zh, max_num=10)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\__init__.py ===
"""
The `numpy.core` submodule exists solely for backward compatibility
purposes. The original `core` was renamed to `_core` and made private.
`numpy.core` will be removed in the future.
"""
from numpy import _core

from ._utils import _raise_warning


# We used to use `np.core._ufunc_reconstruct` to unpickle.
# This is unnecessary, but old pickles saved before 1.20 will be using it,
# and there is no reason to break loading them.
def _ufunc_reconstruct(module, name):
    # The `fromlist` kwarg is required to ensure that `mod` points to the
    # inner-most module rather than the parent package when module name is
    # nested. This makes it possible to pickle non-toplevel ufuncs such as
    # scipy.special.expit for instance.
    mod = __import__(module, fromlist=[name])
    return getattr(mod, name)


# force lazy-loading of submodules to ensure a warning is printed

__all__ = ["arrayprint", "defchararray", "_dtype_ctypes", "_dtype",  # noqa: F822
           "einsumfunc", "fromnumeric", "function_base", "getlimits",
           "_internal", "multiarray", "_multiarray_umath", "numeric",
           "numerictypes", "overrides", "records", "shape_base", "umath"]

def __getattr__(attr_name):
    attr = getattr(_core, attr_name)
    _raise_warning(attr_name)
    return attr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\capi\_pybind_state.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Ensure that dependencies are available and then load the extension module.
"""
import os
import platform
import warnings

from . import _ld_preload  # noqa: F401

if platform.system() == "Windows":
    from . import version_info

    # If on Windows, check if this import error is caused by the user not installing the 2019 VC Runtime
    # The VC Redist installer usually puts the VC Runtime dlls in the System32 folder, but it may also be found
    # in some other locations.
    # TODO, we may want to try to load the VC Runtime dlls instead of checking if the hardcoded file path
    # is valid, and raise ImportError if the load fails
    if version_info.vs2019 and platform.architecture()[0] == "64bit":
        system_root = os.getenv("SystemRoot") or "C:\\Windows"
        if not os.path.isfile(os.path.join(system_root, "System32", "vcruntime140_1.dll")):
            warnings.warn("Please install the 2019 Visual C++ runtime and then try again. "
                          "If you've installed the runtime in a non-standard location "
                          "(other than %SystemRoot%\\System32), "
                          "make sure it can be found by setting the correct path.")



from .onnxruntime_pybind11_state import *  # noqa


# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\sam2\nvtx_helper.py ===
# -------------------------------------------------------------------------
# Copyright (R) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import nvtx
from cuda import cudart


class NvtxHelper:
    def __init__(self, stages):
        self.stages = stages
        self.events = {}
        for stage in stages:
            for marker in ["start", "stop"]:
                self.events[stage + "-" + marker] = cudart.cudaEventCreate()[1]
        self.markers = {}

    def start_profile(self, stage, color="blue"):
        self.markers[stage] = nvtx.start_range(message=stage, color=color)
        event_name = stage + "-start"
        if event_name in self.events:
            cudart.cudaEventRecord(self.events[event_name], 0)

    def stop_profile(self, stage):
        event_name = stage + "-stop"
        if event_name in self.events:
            cudart.cudaEventRecord(self.events[event_name], 0)
        nvtx.end_range(self.markers[stage])

    def print_latency(self):
        for stage in self.stages:
            latency = cudart.cudaEventElapsedTime(self.events[f"{stage}-start"], self.events[f"{stage}-stop"])[1]
            print(f"{stage}: {latency:.2f} ms")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\drawing\xdr.py ===
# Copyright (c) 2010-2024 openpyxl

"""
Spreadsheet Drawing has some copies of Drawing ML elements
"""

from .geometry import Point2D, PositiveSize2D, Transform2D


class XDRPoint2D(Point2D):

    namespace = None
    x = Point2D.x
    y = Point2D.y


class XDRPositiveSize2D(PositiveSize2D):

    namespace = None
    cx = PositiveSize2D.cx
    cy = PositiveSize2D.cy


class XDRTransform2D(Transform2D):

    namespace = None
    rot = Transform2D.rot
    flipH = Transform2D.flipH
    flipV = Transform2D.flipV
    off = Transform2D.off
    ext = Transform2D.ext
    chOff = Transform2D.chOff
    chExt = Transform2D.chExt

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\outcome\_util.py ===
from typing import Any, Dict


class AlreadyUsedError(RuntimeError):
    """An Outcome can only be unwrapped once."""
    pass


def fixup_module_metadata(
        module_name: str,
        namespace: Dict[str, object],
) -> None:
    def fix_one(obj: object) -> None:
        mod = getattr(obj, "__module__", None)
        if mod is not None and mod.startswith("outcome."):
            obj.__module__ = module_name
            if isinstance(obj, type):
                for attr_value in obj.__dict__.values():
                    fix_one(attr_value)

    all_list = namespace["__all__"]
    assert isinstance(all_list, (tuple, list)), repr(all_list)
    for objname in all_list:
        obj = namespace[objname]
        fix_one(obj)


def remove_tb_frames(exc: BaseException, n: int) -> BaseException:
    tb = exc.__traceback__
    for _ in range(n):
        assert tb is not None
        tb = tb.tb_next
    return exc.with_traceback(tb)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\api\extensions\__init__.py ===
"""
Public API for extending pandas objects.
"""

from pandas._libs.lib import no_default

from pandas.core.dtypes.base import (
    ExtensionDtype,
    register_extension_dtype,
)

from pandas.core.accessor import (
    register_dataframe_accessor,
    register_index_accessor,
    register_series_accessor,
)
from pandas.core.algorithms import take
from pandas.core.arrays import (
    ExtensionArray,
    ExtensionScalarOpsMixin,
)

__all__ = [
    "no_default",
    "ExtensionDtype",
    "register_extension_dtype",
    "register_dataframe_accessor",
    "register_index_accessor",
    "register_series_accessor",
    "take",
    "ExtensionArray",
    "ExtensionScalarOpsMixin",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\json\test_ujson.py ===
import calendar
import datetime
import decimal
import json
import locale
import math
import re
import time

import dateutil
import numpy as np
import pytest
import pytz

import pandas._libs.json as ujson
from pandas.compat import IS64

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    NaT,
    PeriodIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm


def _clean_dict(d):
    """
    Sanitize dictionary for JSON by converting all keys to strings.

    Parameters
    ----------
    d : dict
        The dictionary to convert.

    Returns
    -------
    cleaned_dict : dict
    """
    return {str(k): v for k, v in d.items()}


@pytest.fixture(
    params=[None, "split", "records", "values", "index"]  # Column indexed by default.
)
def orient(request):
    return request.param


class TestUltraJSONTests:
    @pytest.mark.skipif(not IS64, reason="not compliant on 32-bit, xref #15865")
    def test_encode_decimal(self):
        sut = decimal.Decimal("1337.1337")
        encoded = ujson.ujson_dumps(sut, double_precision=15)
        decoded = ujson.ujson_loads(encoded)
        assert decoded == 1337.1337

        sut = decimal.Decimal("0.95")
        encoded = ujson.ujson_dumps(sut, double_precision=1)
        assert encoded == "1.0"

        decoded = ujson.ujson_loads(encoded)
        assert decoded == 1.0

        sut = decimal.Decimal("0.94")
        encoded = ujson.ujson_dumps(sut, double_precision=1)
        assert encoded == "0.9"

        decoded = ujson.ujson_loads(encoded)
        assert decoded == 0.9

        sut = decimal.Decimal("1.95")
        encoded = ujson.ujson_dumps(sut, double_precision=1)
        assert encoded == "2.0"

        decoded = ujson.ujson_loads(encoded)
        assert decoded == 2.0

        sut = decimal.Decimal("-1.95")
        encoded = ujson.ujson_dumps(sut, double_precision=1)
        assert encoded == "-2.0"

        decoded = ujson.ujson_loads(encoded)
        assert decoded == -2.0

        sut = decimal.Decimal("0.995")
        encoded = ujson.ujson_dumps(sut, double_precision=2)
        assert encoded == "1.0"

        decoded = ujson.ujson_loads(encoded)
        assert decoded == 1.0

        sut = decimal.Decimal("0.9995")
        encoded = ujson.ujson_dumps(sut, double_precision=3)
        assert encoded == "1.0"

        decoded = ujson.ujson_loads(encoded)
        assert decoded == 1.0

        sut = decimal.Decimal("0.99999999999999944")
        encoded = ujson.ujson_dumps(sut, double_precision=15)
        assert encoded == "1.0"

        decoded = ujson.ujson_loads(encoded)
        assert decoded == 1.0

    @pytest.mark.parametrize("ensure_ascii", [True, False])
    def test_encode_string_conversion(self, ensure_ascii):
        string_input = "A string \\ / \b \f \n \r \t </script> &"
        not_html_encoded = '"A string \\\\ \\/ \\b \\f \\n \\r \\t <\\/script> &"'
        html_encoded = (
            '"A string \\\\ \\/ \\b \\f \\n \\r \\t \\u003c\\/script\\u003e \\u0026"'
        )

        def helper(expected_output, **encode_kwargs):
            output = ujson.ujson_dumps(
                string_input, ensure_ascii=ensure_ascii, **encode_kwargs
            )

            assert output == expected_output
            assert string_input == json.loads(output)
            assert string_input == ujson.ujson_loads(output)

        # Default behavior assumes encode_html_chars=False.
        helper(not_html_encoded)

        # Make sure explicit encode_html_chars=False works.
        helper(not_html_encoded, encode_html_chars=False)

        # Make sure explicit encode_html_chars=True does the encoding.
        helper(html_encoded, encode_html_chars=True)

    @pytest.mark.parametrize(
        "long_number", [-4342969734183514, -12345678901234.56789012, -528656961.4399388]
    )
    def test_double_long_numbers(self, long_number):
        sut = {"a": long_number}
        encoded = ujson.ujson_dumps(sut, double_precision=15)

        decoded = ujson.ujson_loads(encoded)
        assert sut == decoded

    def test_encode_non_c_locale(self):
        lc_category = locale.LC_NUMERIC

        # We just need one of these locales to work.
        for new_locale in ("it_IT.UTF-8", "Italian_Italy"):
            if tm.can_set_locale(new_locale, lc_category):
                with tm.set_locale(new_locale, lc_category):
                    assert ujson.ujson_loads(ujson.ujson_dumps(4.78e60)) == 4.78e60
                    assert ujson.ujson_loads("4.78", precise_float=True) == 4.78
                break

    def test_decimal_decode_test_precise(self):
        sut = {"a": 4.56}
        encoded = ujson.ujson_dumps(sut)
        decoded = ujson.ujson_loads(encoded, precise_float=True)
        assert sut == decoded

    def test_encode_double_tiny_exponential(self):
        num = 1e-40
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = 1e-100
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = -1e-45
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = -1e-145
        assert np.allclose(num, ujson.ujson_loads(ujson.ujson_dumps(num)))

    @pytest.mark.parametrize("unicode_key", ["key1", "بن"])
    def test_encode_dict_with_unicode_keys(self, unicode_key):
        unicode_dict = {unicode_key: "value1"}
        assert unicode_dict == ujson.ujson_loads(ujson.ujson_dumps(unicode_dict))

    @pytest.mark.parametrize(
        "double_input", [math.pi, -math.pi]  # Should work with negatives too.
    )
    def test_encode_double_conversion(self, double_input):
        output = ujson.ujson_dumps(double_input)
        assert round(double_input, 5) == round(json.loads(output), 5)
        assert round(double_input, 5) == round(ujson.ujson_loads(output), 5)

    def test_encode_with_decimal(self):
        decimal_input = 1.0
        output = ujson.ujson_dumps(decimal_input)

        assert output == "1.0"

    def test_encode_array_of_nested_arrays(self):
        nested_input = [[[[]]]] * 20
        output = ujson.ujson_dumps(nested_input)

        assert nested_input == json.loads(output)
        assert nested_input == ujson.ujson_loads(output)

    def test_encode_array_of_doubles(self):
        doubles_input = [31337.31337, 31337.31337, 31337.31337, 31337.31337] * 10
        output = ujson.ujson_dumps(doubles_input)

        assert doubles_input == json.loads(output)
        assert doubles_input == ujson.ujson_loads(output)

    def test_double_precision(self):
        double_input = 30.012345678901234
        output = ujson.ujson_dumps(double_input, double_precision=15)

        assert double_input == json.loads(output)
        assert double_input == ujson.ujson_loads(output)

        for double_precision in (3, 9):
            output = ujson.ujson_dumps(double_input, double_precision=double_precision)
            rounded_input = round(double_input, double_precision)

            assert rounded_input == json.loads(output)
            assert rounded_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize(
        "invalid_val",
        [
            20,
            -1,
            "9",
            None,
        ],
    )
    def test_invalid_double_precision(self, invalid_val):
        double_input = 30.12345678901234567890
        expected_exception = ValueError if isinstance(invalid_val, int) else TypeError
        msg = (
            r"Invalid value '.*' for option 'double_precision', max is '15'|"
            r"an integer is required \(got type |"
            r"object cannot be interpreted as an integer"
        )
        with pytest.raises(expected_exception, match=msg):
            ujson.ujson_dumps(double_input, double_precision=invalid_val)

    def test_encode_string_conversion2(self):
        string_input = "A string \\ / \b \f \n \r \t"
        output = ujson.ujson_dumps(string_input)

        assert string_input == json.loads(output)
        assert string_input == ujson.ujson_loads(output)
        assert output == '"A string \\\\ \\/ \\b \\f \\n \\r \\t"'

    @pytest.mark.parametrize(
        "unicode_input",
        ["Räksmörgås اسامة بن محمد بن عوض بن لادن", "\xe6\x97\xa5\xd1\x88"],
    )
    def test_encode_unicode_conversion(self, unicode_input):
        enc = ujson.ujson_dumps(unicode_input)
        dec = ujson.ujson_loads(enc)

        assert enc == json.dumps(unicode_input)
        assert dec == json.loads(enc)

    def test_encode_control_escaping(self):
        escaped_input = "\x19"
        enc = ujson.ujson_dumps(escaped_input)
        dec = ujson.ujson_loads(enc)

        assert escaped_input == dec
        assert enc == json.dumps(escaped_input)

    def test_encode_unicode_surrogate_pair(self):
        surrogate_input = "\xf0\x90\x8d\x86"
        enc = ujson.ujson_dumps(surrogate_input)
        dec = ujson.ujson_loads(enc)

        assert enc == json.dumps(surrogate_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_4bytes_utf8(self):
        four_bytes_input = "\xf0\x91\x80\xb0TRAILINGNORMAL"
        enc = ujson.ujson_dumps(four_bytes_input)
        dec = ujson.ujson_loads(enc)

        assert enc == json.dumps(four_bytes_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_4bytes_utf8highest(self):
        four_bytes_input = "\xf3\xbf\xbf\xbfTRAILINGNORMAL"
        enc = ujson.ujson_dumps(four_bytes_input)

        dec = ujson.ujson_loads(enc)

        assert enc == json.dumps(four_bytes_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_error(self):
        string = "'\udac0'"
        msg = (
            r"'utf-8' codec can't encode character '\\udac0' "
            r"in position 1: surrogates not allowed"
        )
        with pytest.raises(UnicodeEncodeError, match=msg):
            ujson.ujson_dumps([string])

    def test_encode_array_in_array(self):
        arr_in_arr_input = [[[[]]]]
        output = ujson.ujson_dumps(arr_in_arr_input)

        assert arr_in_arr_input == json.loads(output)
        assert output == json.dumps(arr_in_arr_input)
        assert arr_in_arr_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize(
        "num_input",
        [
            31337,
            -31337,  # Negative number.
            -9223372036854775808,  # Large negative number.
        ],
    )
    def test_encode_num_conversion(self, num_input):
        output = ujson.ujson_dumps(num_input)
        assert num_input == json.loads(output)
        assert output == json.dumps(num_input)
        assert num_input == ujson.ujson_loads(output)

    def test_encode_list_conversion(self):
        list_input = [1, 2, 3, 4]
        output = ujson.ujson_dumps(list_input)

        assert list_input == json.loads(output)
        assert list_input == ujson.ujson_loads(output)

    def test_encode_dict_conversion(self):
        dict_input = {"k1": 1, "k2": 2, "k3": 3, "k4": 4}
        output = ujson.ujson_dumps(dict_input)

        assert dict_input == json.loads(output)
        assert dict_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize("builtin_value", [None, True, False])
    def test_encode_builtin_values_conversion(self, builtin_value):
        output = ujson.ujson_dumps(builtin_value)
        assert builtin_value == json.loads(output)
        assert output == json.dumps(builtin_value)
        assert builtin_value == ujson.ujson_loads(output)

    def test_encode_datetime_conversion(self):
        datetime_input = datetime.datetime.fromtimestamp(time.time())
        output = ujson.ujson_dumps(datetime_input, date_unit="s")
        expected = calendar.timegm(datetime_input.utctimetuple())

        assert int(expected) == json.loads(output)
        assert int(expected) == ujson.ujson_loads(output)

    def test_encode_date_conversion(self):
        date_input = datetime.date.fromtimestamp(time.time())
        output = ujson.ujson_dumps(date_input, date_unit="s")

        tup = (date_input.year, date_input.month, date_input.day, 0, 0, 0)
        expected = calendar.timegm(tup)

        assert int(expected) == json.loads(output)
        assert int(expected) == ujson.ujson_loads(output)

    @pytest.mark.parametrize(
        "test",
        [datetime.time(), datetime.time(1, 2, 3), datetime.time(10, 12, 15, 343243)],
    )
    def test_encode_time_conversion_basic(self, test):
        output = ujson.ujson_dumps(test)
        expected = f'"{test.isoformat()}"'
        assert expected == output

    def test_encode_time_conversion_pytz(self):
        # see gh-11473: to_json segfaults with timezone-aware datetimes
        test = datetime.time(10, 12, 15, 343243, pytz.utc)
        output = ujson.ujson_dumps(test)
        expected = f'"{test.isoformat()}"'
        assert expected == output

    def test_encode_time_conversion_dateutil(self):
        # see gh-11473: to_json segfaults with timezone-aware datetimes
        test = datetime.time(10, 12, 15, 343243, dateutil.tz.tzutc())
        output = ujson.ujson_dumps(test)
        expected = f'"{test.isoformat()}"'
        assert expected == output

    @pytest.mark.parametrize(
        "decoded_input", [NaT, np.datetime64("NaT"), np.nan, np.inf, -np.inf]
    )
    def test_encode_as_null(self, decoded_input):
        assert ujson.ujson_dumps(decoded_input) == "null", "Expected null"

    def test_datetime_units(self):
        val = datetime.datetime(2013, 8, 17, 21, 17, 12, 215504)
        stamp = Timestamp(val).as_unit("ns")

        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit="s"))
        assert roundtrip == stamp._value // 10**9

        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit="ms"))
        assert roundtrip == stamp._value // 10**6

        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit="us"))
        assert roundtrip == stamp._value // 10**3

        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit="ns"))
        assert roundtrip == stamp._value

        msg = "Invalid value 'foo' for option 'date_unit'"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_dumps(val, date_unit="foo")

    def test_encode_to_utf8(self):
        unencoded = "\xe6\x97\xa5\xd1\x88"

        enc = ujson.ujson_dumps(unencoded, ensure_ascii=False)
        dec = ujson.ujson_loads(enc)

        assert enc == json.dumps(unencoded, ensure_ascii=False)
        assert dec == json.loads(enc)

    def test_decode_from_unicode(self):
        unicode_input = '{"obj": 31337}'

        dec1 = ujson.ujson_loads(unicode_input)
        dec2 = ujson.ujson_loads(str(unicode_input))

        assert dec1 == dec2

    def test_encode_recursion_max(self):
        # 8 is the max recursion depth

        class O2:
            member = 0

        class O1:
            member = 0

        decoded_input = O1()
        decoded_input.member = O2()
        decoded_input.member.member = decoded_input

        with pytest.raises(OverflowError, match="Maximum recursion level reached"):
            ujson.ujson_dumps(decoded_input)

    def test_decode_jibberish(self):
        jibberish = "fdsa sda v9sa fdsa"
        msg = "Unexpected character found when decoding 'false'"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(jibberish)

    @pytest.mark.parametrize(
        "broken_json",
        [
            "[",  # Broken array start.
            "{",  # Broken object start.
            "]",  # Broken array end.
            "}",  # Broken object end.
        ],
    )
    def test_decode_broken_json(self, broken_json):
        msg = "Expected object or value"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(broken_json)

    @pytest.mark.parametrize("too_big_char", ["[", "{"])
    def test_decode_depth_too_big(self, too_big_char):
        with pytest.raises(ValueError, match="Reached object decoding depth limit"):
            ujson.ujson_loads(too_big_char * (1024 * 1024))

    @pytest.mark.parametrize(
        "bad_string",
        [
            '"TESTING',  # Unterminated.
            '"TESTING\\"',  # Unterminated escape.
            "tru",  # Broken True.
            "fa",  # Broken False.
            "n",  # Broken None.
        ],
    )
    def test_decode_bad_string(self, bad_string):
        msg = (
            "Unexpected character found when decoding|"
            "Unmatched ''\"' when when decoding 'string'"
        )
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(bad_string)

    @pytest.mark.parametrize(
        "broken_json, err_msg",
        [
            (
                '{{1337:""}}',
                "Key name of object must be 'string' when decoding 'object'",
            ),
            ('{{"key":"}', "Unmatched ''\"' when when decoding 'string'"),
            ("[[[true", "Unexpected character found when decoding array value (2)"),
        ],
    )
    def test_decode_broken_json_leak(self, broken_json, err_msg):
        for _ in range(1000):
            with pytest.raises(ValueError, match=re.escape(err_msg)):
                ujson.ujson_loads(broken_json)

    @pytest.mark.parametrize(
        "invalid_dict",
        [
            "{{{{31337}}}}",  # No key.
            '{{{{"key":}}}}',  # No value.
            '{{{{"key"}}}}',  # No colon or value.
        ],
    )
    def test_decode_invalid_dict(self, invalid_dict):
        msg = (
            "Key name of object must be 'string' when decoding 'object'|"
            "No ':' found when decoding object value|"
            "Expected object or value"
        )
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(invalid_dict)

    @pytest.mark.parametrize(
        "numeric_int_as_str", ["31337", "-31337"]  # Should work with negatives.
    )
    def test_decode_numeric_int(self, numeric_int_as_str):
        assert int(numeric_int_as_str) == ujson.ujson_loads(numeric_int_as_str)

    def test_encode_null_character(self):
        wrapped_input = "31337 \x00 1337"
        output = ujson.ujson_dumps(wrapped_input)

        assert wrapped_input == json.loads(output)
        assert output == json.dumps(wrapped_input)
        assert wrapped_input == ujson.ujson_loads(output)

        alone_input = "\x00"
        output = ujson.ujson_dumps(alone_input)

        assert alone_input == json.loads(output)
        assert output == json.dumps(alone_input)
        assert alone_input == ujson.ujson_loads(output)
        assert '"  \\u0000\\r\\n "' == ujson.ujson_dumps("  \u0000\r\n ")

    def test_decode_null_character(self):
        wrapped_input = '"31337 \\u0000 31337"'
        assert ujson.ujson_loads(wrapped_input) == json.loads(wrapped_input)

    def test_encode_list_long_conversion(self):
        long_input = [
            9223372036854775807,
            9223372036854775807,
            9223372036854775807,
            9223372036854775807,
            9223372036854775807,
            9223372036854775807,
        ]
        output = ujson.ujson_dumps(long_input)

        assert long_input == json.loads(output)
        assert long_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize("long_input", [9223372036854775807, 18446744073709551615])
    def test_encode_long_conversion(self, long_input):
        output = ujson.ujson_dumps(long_input)

        assert long_input == json.loads(output)
        assert output == json.dumps(long_input)
        assert long_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize("bigNum", [2**64, -(2**63) - 1])
    def test_dumps_ints_larger_than_maxsize(self, bigNum):
        encoding = ujson.ujson_dumps(bigNum)
        assert str(bigNum) == encoding

        with pytest.raises(
            ValueError,
            match="Value is too big|Value is too small",
        ):
            assert ujson.ujson_loads(encoding) == bigNum

    @pytest.mark.parametrize(
        "int_exp", ["1337E40", "1.337E40", "1337E+9", "1.337e+40", "1.337E-4"]
    )
    def test_decode_numeric_int_exp(self, int_exp):
        assert ujson.ujson_loads(int_exp) == json.loads(int_exp)

    def test_loads_non_str_bytes_raises(self):
        msg = "a bytes-like object is required, not 'NoneType'"
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_loads(None)

    @pytest.mark.parametrize("val", [3590016419, 2**31, 2**32, (2**32) - 1])
    def test_decode_number_with_32bit_sign_bit(self, val):
        # Test that numbers that fit within 32 bits but would have the
        # sign bit set (2**31 <= x < 2**32) are decoded properly.
        doc = f'{{"id": {val}}}'
        assert ujson.ujson_loads(doc)["id"] == val

    def test_encode_big_escape(self):
        # Make sure no Exception is raised.
        for _ in range(10):
            base = "\u00e5".encode()
            escape_input = base * 1024 * 1024 * 2
            ujson.ujson_dumps(escape_input)

    def test_decode_big_escape(self):
        # Make sure no Exception is raised.
        for _ in range(10):
            base = "\u00e5".encode()
            quote = b'"'

            escape_input = quote + (base * 1024 * 1024 * 2) + quote
            ujson.ujson_loads(escape_input)

    def test_to_dict(self):
        d = {"key": 31337}

        class DictTest:
            def toDict(self):
                return d

        o = DictTest()
        output = ujson.ujson_dumps(o)

        dec = ujson.ujson_loads(output)
        assert dec == d

    def test_default_handler(self):
        class _TestObject:
            def __init__(self, val) -> None:
                self.val = val

            @property
            def recursive_attr(self):
                return _TestObject("recursive_attr")

            def __str__(self) -> str:
                return str(self.val)

        msg = "Maximum recursion level reached"
        with pytest.raises(OverflowError, match=msg):
            ujson.ujson_dumps(_TestObject("foo"))
        assert '"foo"' == ujson.ujson_dumps(_TestObject("foo"), default_handler=str)

        def my_handler(_):
            return "foobar"

        assert '"foobar"' == ujson.ujson_dumps(
            _TestObject("foo"), default_handler=my_handler
        )

        def my_handler_raises(_):
            raise TypeError("I raise for anything")

        with pytest.raises(TypeError, match="I raise for anything"):
            ujson.ujson_dumps(_TestObject("foo"), default_handler=my_handler_raises)

        def my_int_handler(_):
            return 42

        assert (
            ujson.ujson_loads(
                ujson.ujson_dumps(_TestObject("foo"), default_handler=my_int_handler)
            )
            == 42
        )

        def my_obj_handler(_):
            return datetime.datetime(2013, 2, 3)

        assert ujson.ujson_loads(
            ujson.ujson_dumps(datetime.datetime(2013, 2, 3))
        ) == ujson.ujson_loads(
            ujson.ujson_dumps(_TestObject("foo"), default_handler=my_obj_handler)
        )

        obj_list = [_TestObject("foo"), _TestObject("bar")]
        assert json.loads(json.dumps(obj_list, default=str)) == ujson.ujson_loads(
            ujson.ujson_dumps(obj_list, default_handler=str)
        )

    def test_encode_object(self):
        class _TestObject:
            def __init__(self, a, b, _c, d) -> None:
                self.a = a
                self.b = b
                self._c = _c
                self.d = d

            def e(self):
                return 5

        # JSON keys should be all non-callable non-underscore attributes, see GH-42768
        test_object = _TestObject(a=1, b=2, _c=3, d=4)
        assert ujson.ujson_loads(ujson.ujson_dumps(test_object)) == {
            "a": 1,
            "b": 2,
            "d": 4,
        }

    def test_ujson__name__(self):
        # GH 52898
        assert ujson.__name__ == "pandas._libs.json"


class TestNumpyJSONTests:
    @pytest.mark.parametrize("bool_input", [True, False])
    def test_bool(self, bool_input):
        b = bool(bool_input)
        assert ujson.ujson_loads(ujson.ujson_dumps(b)) == b

    def test_bool_array(self):
        bool_array = np.array(
            [True, False, True, True, False, True, False, False], dtype=bool
        )
        output = np.array(ujson.ujson_loads(ujson.ujson_dumps(bool_array)), dtype=bool)
        tm.assert_numpy_array_equal(bool_array, output)

    def test_int(self, any_int_numpy_dtype):
        klass = np.dtype(any_int_numpy_dtype).type
        num = klass(1)

        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    def test_int_array(self, any_int_numpy_dtype):
        arr = np.arange(100, dtype=int)
        arr_input = arr.astype(any_int_numpy_dtype)

        arr_output = np.array(
            ujson.ujson_loads(ujson.ujson_dumps(arr_input)), dtype=any_int_numpy_dtype
        )
        tm.assert_numpy_array_equal(arr_input, arr_output)

    def test_int_max(self, any_int_numpy_dtype):
        if any_int_numpy_dtype in ("int64", "uint64") and not IS64:
            pytest.skip("Cannot test 64-bit integer on 32-bit platform")

        klass = np.dtype(any_int_numpy_dtype).type

        # uint64 max will always overflow,
        # as it's encoded to signed.
        if any_int_numpy_dtype == "uint64":
            num = np.iinfo("int64").max
        else:
            num = np.iinfo(any_int_numpy_dtype).max

        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    def test_float(self, float_numpy_dtype):
        klass = np.dtype(float_numpy_dtype).type
        num = klass(256.2013)

        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    def test_float_array(self, float_numpy_dtype):
        arr = np.arange(12.5, 185.72, 1.7322, dtype=float)
        float_input = arr.astype(float_numpy_dtype)

        float_output = np.array(
            ujson.ujson_loads(ujson.ujson_dumps(float_input, double_precision=15)),
            dtype=float_numpy_dtype,
        )
        tm.assert_almost_equal(float_input, float_output)

    def test_float_max(self, float_numpy_dtype):
        klass = np.dtype(float_numpy_dtype).type
        num = klass(np.finfo(float_numpy_dtype).max / 10)

        tm.assert_almost_equal(
            klass(ujson.ujson_loads(ujson.ujson_dumps(num, double_precision=15))), num
        )

    def test_array_basic(self):
        arr = np.arange(96)
        arr = arr.reshape((2, 2, 2, 2, 3, 2))

        tm.assert_numpy_array_equal(
            np.array(ujson.ujson_loads(ujson.ujson_dumps(arr))), arr
        )

    @pytest.mark.parametrize("shape", [(10, 10), (5, 5, 4), (100, 1)])
    def test_array_reshaped(self, shape):
        arr = np.arange(100)
        arr = arr.reshape(shape)

        tm.assert_numpy_array_equal(
            np.array(ujson.ujson_loads(ujson.ujson_dumps(arr))), arr
        )

    def test_array_list(self):
        arr_list = [
            "a",
            [],
            {},
            {},
            [],
            42,
            97.8,
            ["a", "b"],
            {"key": "val"},
        ]
        arr = np.array(arr_list, dtype=object)
        result = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr)), dtype=object)
        tm.assert_numpy_array_equal(result, arr)

    def test_array_float(self):
        dtype = np.float32

        arr = np.arange(100.202, 200.202, 1, dtype=dtype)
        arr = arr.reshape((5, 5, 4))

        arr_out = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr)), dtype=dtype)
        tm.assert_almost_equal(arr, arr_out)

    def test_0d_array(self):
        # gh-18878
        msg = re.escape(
            "array(1) (numpy-scalar) is not JSON serializable at the moment"
        )
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_dumps(np.array(1))

    def test_array_long_double(self):
        msg = re.compile(
            "1234.5.* \\(numpy-scalar\\) is not JSON serializable at the moment"
        )
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_dumps(np.longdouble(1234.5))


class TestPandasJSONTests:
    def test_dataframe(self, orient):
        dtype = np.int64

        df = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            index=["a", "b"],
            columns=["x", "y", "z"],
            dtype=dtype,
        )
        encode_kwargs = {} if orient is None else {"orient": orient}
        assert (df.dtypes == dtype).all()

        output = ujson.ujson_loads(ujson.ujson_dumps(df, **encode_kwargs))
        assert (df.dtypes == dtype).all()

        # Ensure proper DataFrame initialization.
        if orient == "split":
            dec = _clean_dict(output)
            output = DataFrame(**dec)
        else:
            output = DataFrame(output)

        # Corrections to enable DataFrame comparison.
        if orient == "values":
            df.columns = [0, 1, 2]
            df.index = [0, 1]
        elif orient == "records":
            df.index = [0, 1]
        elif orient == "index":
            df = df.transpose()

        assert (df.dtypes == dtype).all()
        tm.assert_frame_equal(output, df)

    def test_dataframe_nested(self, orient):
        df = DataFrame(
            [[1, 2, 3], [4, 5, 6]], index=["a", "b"], columns=["x", "y", "z"]
        )

        nested = {"df1": df, "df2": df.copy()}
        kwargs = {} if orient is None else {"orient": orient}

        exp = {
            "df1": ujson.ujson_loads(ujson.ujson_dumps(df, **kwargs)),
            "df2": ujson.ujson_loads(ujson.ujson_dumps(df, **kwargs)),
        }
        assert ujson.ujson_loads(ujson.ujson_dumps(nested, **kwargs)) == exp

    def test_series(self, orient):
        dtype = np.int64
        s = Series(
            [10, 20, 30, 40, 50, 60],
            name="series",
            index=[6, 7, 8, 9, 10, 15],
            dtype=dtype,
        ).sort_values()
        assert s.dtype == dtype

        encode_kwargs = {} if orient is None else {"orient": orient}

        output = ujson.ujson_loads(ujson.ujson_dumps(s, **encode_kwargs))
        assert s.dtype == dtype

        if orient == "split":
            dec = _clean_dict(output)
            output = Series(**dec)
        else:
            output = Series(output)

        if orient in (None, "index"):
            s.name = None
            output = output.sort_values()
            s.index = ["6", "7", "8", "9", "10", "15"]
        elif orient in ("records", "values"):
            s.name = None
            s.index = [0, 1, 2, 3, 4, 5]

        assert s.dtype == dtype
        tm.assert_series_equal(output, s)

    def test_series_nested(self, orient):
        s = Series(
            [10, 20, 30, 40, 50, 60], name="series", index=[6, 7, 8, 9, 10, 15]
        ).sort_values()
        nested = {"s1": s, "s2": s.copy()}
        kwargs = {} if orient is None else {"orient": orient}

        exp = {
            "s1": ujson.ujson_loads(ujson.ujson_dumps(s, **kwargs)),
            "s2": ujson.ujson_loads(ujson.ujson_dumps(s, **kwargs)),
        }
        assert ujson.ujson_loads(ujson.ujson_dumps(nested, **kwargs)) == exp

    def test_index(self):
        i = Index([23, 45, 18, 98, 43, 11], name="index")

        # Column indexed.
        output = Index(ujson.ujson_loads(ujson.ujson_dumps(i)), name="index")
        tm.assert_index_equal(i, output)

        dec = _clean_dict(ujson.ujson_loads(ujson.ujson_dumps(i, orient="split")))
        output = Index(**dec)

        tm.assert_index_equal(i, output)
        assert i.name == output.name

        tm.assert_index_equal(i, output)
        assert i.name == output.name

        output = Index(
            ujson.ujson_loads(ujson.ujson_dumps(i, orient="values")), name="index"
        )
        tm.assert_index_equal(i, output)

        output = Index(
            ujson.ujson_loads(ujson.ujson_dumps(i, orient="records")), name="index"
        )
        tm.assert_index_equal(i, output)

        output = Index(
            ujson.ujson_loads(ujson.ujson_dumps(i, orient="index")), name="index"
        )
        tm.assert_index_equal(i, output)

    def test_datetime_index(self):
        date_unit = "ns"

        # freq doesn't round-trip
        rng = DatetimeIndex(list(date_range("1/1/2000", periods=20)), freq=None)
        encoded = ujson.ujson_dumps(rng, date_unit=date_unit)

        decoded = DatetimeIndex(np.array(ujson.ujson_loads(encoded)))
        tm.assert_index_equal(rng, decoded)

        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        decoded = Series(ujson.ujson_loads(ujson.ujson_dumps(ts, date_unit=date_unit)))

        idx_values = decoded.index.values.astype(np.int64)
        decoded.index = DatetimeIndex(idx_values)
        tm.assert_series_equal(ts, decoded)

    @pytest.mark.parametrize(
        "invalid_arr",
        [
            "[31337,]",  # Trailing comma.
            "[,31337]",  # Leading comma.
            "[]]",  # Unmatched bracket.
            "[,]",  # Only comma.
        ],
    )
    def test_decode_invalid_array(self, invalid_arr):
        msg = (
            "Expected object or value|Trailing data|"
            "Unexpected character found when decoding array value"
        )
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(invalid_arr)

    @pytest.mark.parametrize("arr", [[], [31337]])
    def test_decode_array(self, arr):
        assert arr == ujson.ujson_loads(str(arr))

    @pytest.mark.parametrize("extreme_num", [9223372036854775807, -9223372036854775808])
    def test_decode_extreme_numbers(self, extreme_num):
        assert extreme_num == ujson.ujson_loads(str(extreme_num))

    @pytest.mark.parametrize("too_extreme_num", [f"{2**64}", f"{-2**63-1}"])
    def test_decode_too_extreme_numbers(self, too_extreme_num):
        with pytest.raises(
            ValueError,
            match="Value is too big|Value is too small",
        ):
            ujson.ujson_loads(too_extreme_num)

    def test_decode_with_trailing_whitespaces(self):
        assert {} == ujson.ujson_loads("{}\n\t ")

    def test_decode_with_trailing_non_whitespaces(self):
        with pytest.raises(ValueError, match="Trailing data"):
            ujson.ujson_loads("{}\n\t a")

    @pytest.mark.parametrize("value", [f"{2**64}", f"{-2**63-1}"])
    def test_decode_array_with_big_int(self, value):
        with pytest.raises(
            ValueError,
            match="Value is too big|Value is too small",
        ):
            ujson.ujson_loads(value)

    @pytest.mark.parametrize(
        "float_number",
        [
            1.1234567893,
            1.234567893,
            1.34567893,
            1.4567893,
            1.567893,
            1.67893,
            1.7893,
            1.893,
            1.3,
        ],
    )
    @pytest.mark.parametrize("sign", [-1, 1])
    def test_decode_floating_point(self, sign, float_number):
        float_number *= sign
        tm.assert_almost_equal(
            float_number, ujson.ujson_loads(str(float_number)), rtol=1e-15
        )

    def test_encode_big_set(self):
        s = set()

        for x in range(100000):
            s.add(x)

        # Make sure no Exception is raised.
        ujson.ujson_dumps(s)

    def test_encode_empty_set(self):
        assert "[]" == ujson.ujson_dumps(set())

    def test_encode_set(self):
        s = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        enc = ujson.ujson_dumps(s)
        dec = ujson.ujson_loads(enc)

        for v in dec:
            assert v in s

    @pytest.mark.parametrize(
        "td",
        [
            Timedelta(days=366),
            Timedelta(days=-1),
            Timedelta(hours=13, minutes=5, seconds=5),
            Timedelta(hours=13, minutes=20, seconds=30),
            Timedelta(days=-1, nanoseconds=5),
            Timedelta(nanoseconds=1),
            Timedelta(microseconds=1, nanoseconds=1),
            Timedelta(milliseconds=1, microseconds=1, nanoseconds=1),
            Timedelta(milliseconds=999, microseconds=999, nanoseconds=999),
        ],
    )
    def test_encode_timedelta_iso(self, td):
        # GH 28256
        result = ujson.ujson_dumps(td, iso_dates=True)
        expected = f'"{td.isoformat()}"'

        assert result == expected

    def test_encode_periodindex(self):
        # GH 46683
        p = PeriodIndex(["2022-04-06", "2022-04-07"], freq="D")
        df = DataFrame(index=p)
        assert df.to_json() == "{}"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\distlib\__init__.py ===
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012-2023 Vinay Sajip.
# Licensed to the Python Software Foundation under a contributor agreement.
# See LICENSE.txt and CONTRIBUTORS.txt.
#
import logging

__version__ = '0.3.9'


class DistlibException(Exception):
    pass


try:
    from logging import NullHandler
except ImportError:  # pragma: no cover

    class NullHandler(logging.Handler):

        def handle(self, record):
            pass

        def emit(self, record):
            pass

        def createLock(self):
            self.lock = None


logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\requests\hooks.py ===
"""
requests.hooks
~~~~~~~~~~~~~~

This module provides the capabilities for the Requests hooks system.

Available hooks:

``response``:
    The response generated from a Request.
"""
HOOKS = ["response"]


def default_hooks():
    return {event: [] for event in HOOKS}


# TODO: response is the only one


def dispatch_hook(key, hooks, hook_data, **kwargs):
    """Dispatches a hook dictionary on a given piece of data."""
    hooks = hooks or {}
    hooks = hooks.get(key)
    if hooks:
        if hasattr(hooks, "__call__"):
            hooks = [hooks]
        for hook in hooks:
            _hook_data = hook(hook_data, **kwargs)
            if _hook_data is not None:
                hook_data = _hook_data
    return hook_data

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\abc.py ===
from abc import ABC


class RichRenderable(ABC):
    """An abstract base class for Rich renderables.

    Note that there is no need to extend this class, the intended use is to check if an
    object supports the Rich renderable protocol. For example::

        if isinstance(my_object, RichRenderable):
            console.print(my_object)

    """

    @classmethod
    def __subclasshook__(cls, other: type) -> bool:
        """Check if this class supports the rich render protocol."""
        return hasattr(other, "__rich_console__") or hasattr(other, "__rich__")


if __name__ == "__main__":  # pragma: no cover
    from pip._vendor.rich.text import Text

    t = Text()
    print(isinstance(Text, RichRenderable))
    print(isinstance(t, RichRenderable))

    class Foo:
        pass

    f = Foo()
    print(isinstance(f, RichRenderable))
    print(isinstance("", RichRenderable))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\requests\hooks.py ===
"""
requests.hooks
~~~~~~~~~~~~~~

This module provides the capabilities for the Requests hooks system.

Available hooks:

``response``:
    The response generated from a Request.
"""
HOOKS = ["response"]


def default_hooks():
    return {event: [] for event in HOOKS}


# TODO: response is the only one


def dispatch_hook(key, hooks, hook_data, **kwargs):
    """Dispatches a hook dictionary on a given piece of data."""
    hooks = hooks or {}
    hooks = hooks.get(key)
    if hooks:
        if hasattr(hooks, "__call__"):
            hooks = [hooks]
        for hook in hooks:
            _hook_data = hook(hook_data, **kwargs)
            if _hook_data is not None:
                hook_data = _hook_data
    return hook_data

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\remote\script_key.py ===
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

import uuid


class ScriptKey:
    def __init__(self, id=None):
        self._id = id or uuid.uuid4()

    @property
    def id(self):
        return self._id

    def __eq__(self, other):
        return self._id == other

    def __repr__(self) -> str:
        return f"ScriptKey(id={self.id})"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\categories\__init__.py ===
"""
Category Theory module.

Provides some of the fundamental category-theory-related classes,
including categories, morphisms, diagrams.  Functors are not
implemented yet.

The general reference work this module tries to follow is

  [JoyOfCats] J. Adamek, H. Herrlich. G. E. Strecker: Abstract and
              Concrete Categories. The Joy of Cats.

The latest version of this book should be available for free download
from

   katmat.math.uni-bremen.de/acc/acc.pdf

"""

from .baseclasses import (Object, Morphism, IdentityMorphism,
                         NamedMorphism, CompositeMorphism, Category,
                         Diagram)

from .diagram_drawing import (DiagramGrid, XypicDiagramDrawer,
                             xypic_draw_diagram, preview_diagram)

__all__ = [
    'Object', 'Morphism', 'IdentityMorphism', 'NamedMorphism',
    'CompositeMorphism', 'Category', 'Diagram',

    'DiagramGrid', 'XypicDiagramDrawer', 'xypic_draw_diagram',
    'preview_diagram',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\test-examples\ruletest5.py ===
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

x, y = _me.dynamicsymbols('x y')
x_d, y_d = _me.dynamicsymbols('x_ y_', 1)
e1 = (x+y)**2+(x-y)**3
e2 = (x-y)**2
e3 = x**2+y**2+2*x*y
m1 = _sm.Matrix([e1,e2]).reshape(2, 1)
m2 = _sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)
m3 = m1+_sm.Matrix([x,y]).reshape(2, 1)
am = _sm.Matrix([i.expand() for i in m1]).reshape((m1).shape[0], (m1).shape[1])
cm = _sm.Matrix([i.expand() for i in _sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)]).reshape((_sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)).shape[0], (_sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)).shape[1])
em = _sm.Matrix([i.expand() for i in m1+_sm.Matrix([x,y]).reshape(2, 1)]).reshape((m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[0], (m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[1])
f = (e1).expand()
g = (e2).expand()
a = _sm.factor((e3), x)
bm = _sm.Matrix([_sm.factor(i, x) for i in m1]).reshape((m1).shape[0], (m1).shape[1])
cm = _sm.Matrix([_sm.factor(i, x) for i in m1+_sm.Matrix([x,y]).reshape(2, 1)]).reshape((m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[0], (m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[1])
a = (e3).diff(x)
b = (e3).diff(y)
cm = _sm.Matrix([i.diff(x) for i in m2]).reshape((m2).shape[0], (m2).shape[1])
dm = _sm.Matrix([i.diff(x) for i in m1+_sm.Matrix([x,y]).reshape(2, 1)]).reshape((m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[0], (m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[1])
frame_a = _me.ReferenceFrame('a')
frame_b = _me.ReferenceFrame('b')
frame_b.orient(frame_a, 'DCM', _sm.Matrix([1,0,0,1,0,0,1,0,0]).reshape(3, 3))
v1 = x*frame_a.x+y*frame_a.y+x*y*frame_a.z
e = (v1).diff(x, frame_b)
fm = _sm.Matrix([i.diff(_sm.Symbol('t')) for i in m1]).reshape((m1).shape[0], (m1).shape[1])
gm = _sm.Matrix([i.diff(_sm.Symbol('t')) for i in _sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)]).reshape((_sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)).shape[0], (_sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)).shape[1])
h = (v1).dt(frame_b)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\pkgdata.py ===
# This module is deprecated and will be removed.

import sys
import os
from io import StringIO

from sympy.utilities.decorator import deprecated


@deprecated(
    """
    The sympy.utilities.pkgdata module and its get_resource function are
    deprecated. Use the stdlib importlib.resources module instead.
    """,
    deprecated_since_version="1.12",
    active_deprecations_target="pkgdata",
)
def get_resource(identifier, pkgname=__name__):

    mod = sys.modules[pkgname]
    fn = getattr(mod, '__file__', None)
    if fn is None:
        raise OSError("%r has no __file__!")
    path = os.path.join(os.path.dirname(fn), identifier)
    loader = getattr(mod, '__loader__', None)
    if loader is not None:
        try:
            data = loader.get_data(path)
        except (OSError, AttributeError):
            pass
        else:
            return StringIO(data.decode('utf-8'))
    return open(os.path.normpath(path), 'rb')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_actuator.py ===
"""Tests for the ``sympy.physics.mechanics.actuator.py`` module."""

import pytest

from sympy import (
    S,
    Matrix,
    Symbol,
    SympifyError,
    sqrt,
    Abs,
    symbols,
    exp,
    sign,
)
from sympy.physics.mechanics import (
    ActuatorBase,
    Force,
    ForceActuator,
    KanesMethod,
    LinearDamper,
    LinearPathway,
    LinearSpring,
    Particle,
    PinJoint,
    Point,
    ReferenceFrame,
    RigidBody,
    TorqueActuator,
    Vector,
    dynamicsymbols,
    DuffingSpring,
    CoulombKineticFriction,
)

from sympy.core.expr import Expr as ExprType

target = RigidBody('target')
reaction = RigidBody('reaction')


class TestForceActuator:

    @pytest.fixture(autouse=True)
    def _linear_pathway_fixture(self):
        self.force = Symbol('F')
        self.pA = Point('pA')
        self.pB = Point('pB')
        self.pathway = LinearPathway(self.pA, self.pB)
        self.q1 = dynamicsymbols('q1')
        self.q2 = dynamicsymbols('q2')
        self.q3 = dynamicsymbols('q3')
        self.q1d = dynamicsymbols('q1', 1)
        self.q2d = dynamicsymbols('q2', 1)
        self.q3d = dynamicsymbols('q3', 1)
        self.N = ReferenceFrame('N')

    def test_is_actuator_base_subclass(self):
        assert issubclass(ForceActuator, ActuatorBase)

    @pytest.mark.parametrize(
        'force, expected_force',
        [
            (1, S.One),
            (S.One, S.One),
            (Symbol('F'), Symbol('F')),
            (dynamicsymbols('F'), dynamicsymbols('F')),
            (Symbol('F')**2 + Symbol('F'), Symbol('F')**2 + Symbol('F')),
        ]
    )
    def test_valid_constructor_force(self, force, expected_force):
        instance = ForceActuator(force, self.pathway)
        assert isinstance(instance, ForceActuator)
        assert hasattr(instance, 'force')
        assert isinstance(instance.force, ExprType)
        assert instance.force == expected_force

    @pytest.mark.parametrize('force', [None, 'F'])
    def test_invalid_constructor_force_not_sympifyable(self, force):
        with pytest.raises(SympifyError):
            _ = ForceActuator(force, self.pathway)

    @pytest.mark.parametrize(
        'pathway',
        [
            LinearPathway(Point('pA'), Point('pB')),
        ]
    )
    def test_valid_constructor_pathway(self, pathway):
        instance = ForceActuator(self.force, pathway)
        assert isinstance(instance, ForceActuator)
        assert hasattr(instance, 'pathway')
        assert isinstance(instance.pathway, LinearPathway)
        assert instance.pathway == pathway

    def test_invalid_constructor_pathway_not_pathway_base(self):
        with pytest.raises(TypeError):
            _ = ForceActuator(self.force, None)

    @pytest.mark.parametrize(
        'property_name, fixture_attr_name',
        [
            ('force', 'force'),
            ('pathway', 'pathway'),
        ]
    )
    def test_properties_are_immutable(self, property_name, fixture_attr_name):
        instance = ForceActuator(self.force, self.pathway)
        value = getattr(self, fixture_attr_name)
        with pytest.raises(AttributeError):
            setattr(instance, property_name, value)

    def test_repr(self):
        actuator = ForceActuator(self.force, self.pathway)
        expected = "ForceActuator(F, LinearPathway(pA, pB))"
        assert repr(actuator) == expected

    def test_to_loads_static_pathway(self):
        self.pB.set_pos(self.pA, 2*self.N.x)
        actuator = ForceActuator(self.force, self.pathway)
        expected = [
            (self.pA, - self.force*self.N.x),
            (self.pB, self.force*self.N.x),
        ]
        assert actuator.to_loads() == expected

    def test_to_loads_2D_pathway(self):
        self.pB.set_pos(self.pA, 2*self.q1*self.N.x)
        actuator = ForceActuator(self.force, self.pathway)
        expected = [
            (self.pA, - self.force*(self.q1/sqrt(self.q1**2))*self.N.x),
            (self.pB, self.force*(self.q1/sqrt(self.q1**2))*self.N.x),
        ]
        assert actuator.to_loads() == expected

    def test_to_loads_3D_pathway(self):
        self.pB.set_pos(
            self.pA,
            self.q1*self.N.x - self.q2*self.N.y + 2*self.q3*self.N.z,
        )
        actuator = ForceActuator(self.force, self.pathway)
        length = sqrt(self.q1**2 + self.q2**2 + 4*self.q3**2)
        pO_force = (
            - self.force*self.q1*self.N.x/length
            + self.force*self.q2*self.N.y/length
            - 2*self.force*self.q3*self.N.z/length
        )
        pI_force = (
            self.force*self.q1*self.N.x/length
            - self.force*self.q2*self.N.y/length
            + 2*self.force*self.q3*self.N.z/length
        )
        expected = [
            (self.pA, pO_force),
            (self.pB, pI_force),
        ]
        assert actuator.to_loads() == expected


class TestLinearSpring:

    @pytest.fixture(autouse=True)
    def _linear_spring_fixture(self):
        self.stiffness = Symbol('k')
        self.l = Symbol('l')
        self.pA = Point('pA')
        self.pB = Point('pB')
        self.pathway = LinearPathway(self.pA, self.pB)
        self.q = dynamicsymbols('q')
        self.N = ReferenceFrame('N')

    def test_is_force_actuator_subclass(self):
        assert issubclass(LinearSpring, ForceActuator)

    def test_is_actuator_base_subclass(self):
        assert issubclass(LinearSpring, ActuatorBase)

    @pytest.mark.parametrize(
        (
            'stiffness, '
            'expected_stiffness, '
            'equilibrium_length, '
            'expected_equilibrium_length, '
            'force'
        ),
        [
            (
                1,
                S.One,
                0,
                S.Zero,
                -sqrt(dynamicsymbols('q')**2),
            ),
            (
                Symbol('k'),
                Symbol('k'),
                0,
                S.Zero,
                -Symbol('k')*sqrt(dynamicsymbols('q')**2),
            ),
            (
                Symbol('k'),
                Symbol('k'),
                S.Zero,
                S.Zero,
                -Symbol('k')*sqrt(dynamicsymbols('q')**2),
            ),
            (
                Symbol('k'),
                Symbol('k'),
                Symbol('l'),
                Symbol('l'),
                -Symbol('k')*(sqrt(dynamicsymbols('q')**2) - Symbol('l')),
            ),
        ]
    )
    def test_valid_constructor(
        self,
        stiffness,
        expected_stiffness,
        equilibrium_length,
        expected_equilibrium_length,
        force,
    ):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        spring = LinearSpring(stiffness, self.pathway, equilibrium_length)

        assert isinstance(spring, LinearSpring)

        assert hasattr(spring, 'stiffness')
        assert isinstance(spring.stiffness, ExprType)
        assert spring.stiffness == expected_stiffness

        assert hasattr(spring, 'pathway')
        assert isinstance(spring.pathway, LinearPathway)
        assert spring.pathway == self.pathway

        assert hasattr(spring, 'equilibrium_length')
        assert isinstance(spring.equilibrium_length, ExprType)
        assert spring.equilibrium_length == expected_equilibrium_length

        assert hasattr(spring, 'force')
        assert isinstance(spring.force, ExprType)
        assert spring.force == force

    @pytest.mark.parametrize('stiffness', [None, 'k'])
    def test_invalid_constructor_stiffness_not_sympifyable(self, stiffness):
        with pytest.raises(SympifyError):
            _ = LinearSpring(stiffness, self.pathway, self.l)

    def test_invalid_constructor_pathway_not_pathway_base(self):
        with pytest.raises(TypeError):
            _ = LinearSpring(self.stiffness, None, self.l)

    @pytest.mark.parametrize('equilibrium_length', [None, 'l'])
    def test_invalid_constructor_equilibrium_length_not_sympifyable(
        self,
        equilibrium_length,
    ):
        with pytest.raises(SympifyError):
            _ = LinearSpring(self.stiffness, self.pathway, equilibrium_length)

    @pytest.mark.parametrize(
        'property_name, fixture_attr_name',
        [
            ('stiffness', 'stiffness'),
            ('pathway', 'pathway'),
            ('equilibrium_length', 'l'),
        ]
    )
    def test_properties_are_immutable(self, property_name, fixture_attr_name):
        spring = LinearSpring(self.stiffness, self.pathway, self.l)
        value = getattr(self, fixture_attr_name)
        with pytest.raises(AttributeError):
            setattr(spring, property_name, value)

    @pytest.mark.parametrize(
        'equilibrium_length, expected',
        [
            (S.Zero, 'LinearSpring(k, LinearPathway(pA, pB))'),
            (
                Symbol('l'),
                'LinearSpring(k, LinearPathway(pA, pB), equilibrium_length=l)',
            ),
        ]
    )
    def test_repr(self, equilibrium_length, expected):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        spring = LinearSpring(self.stiffness, self.pathway, equilibrium_length)
        assert repr(spring) == expected

    def test_to_loads(self):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        spring = LinearSpring(self.stiffness, self.pathway, self.l)
        normal = self.q/sqrt(self.q**2)*self.N.x
        pA_force = self.stiffness*(sqrt(self.q**2) - self.l)*normal
        pB_force = -self.stiffness*(sqrt(self.q**2) - self.l)*normal
        expected = [Force(self.pA, pA_force), Force(self.pB, pB_force)]
        loads = spring.to_loads()

        for load, (point, vector) in zip(loads, expected):
            assert isinstance(load, Force)
            assert load.point == point
            assert (load.vector - vector).simplify() == 0


class TestLinearDamper:

    @pytest.fixture(autouse=True)
    def _linear_damper_fixture(self):
        self.damping = Symbol('c')
        self.l = Symbol('l')
        self.pA = Point('pA')
        self.pB = Point('pB')
        self.pathway = LinearPathway(self.pA, self.pB)
        self.q = dynamicsymbols('q')
        self.dq = dynamicsymbols('q', 1)
        self.u = dynamicsymbols('u')
        self.N = ReferenceFrame('N')

    def test_is_force_actuator_subclass(self):
        assert issubclass(LinearDamper, ForceActuator)

    def test_is_actuator_base_subclass(self):
        assert issubclass(LinearDamper, ActuatorBase)

    def test_valid_constructor(self):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        damper = LinearDamper(self.damping, self.pathway)

        assert isinstance(damper, LinearDamper)

        assert hasattr(damper, 'damping')
        assert isinstance(damper.damping, ExprType)
        assert damper.damping == self.damping

        assert hasattr(damper, 'pathway')
        assert isinstance(damper.pathway, LinearPathway)
        assert damper.pathway == self.pathway

    def test_valid_constructor_force(self):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        damper = LinearDamper(self.damping, self.pathway)

        expected_force = -self.damping*sqrt(self.q**2)*self.dq/self.q
        assert hasattr(damper, 'force')
        assert isinstance(damper.force, ExprType)
        assert damper.force == expected_force

    @pytest.mark.parametrize('damping', [None, 'c'])
    def test_invalid_constructor_damping_not_sympifyable(self, damping):
        with pytest.raises(SympifyError):
            _ = LinearDamper(damping, self.pathway)

    def test_invalid_constructor_pathway_not_pathway_base(self):
        with pytest.raises(TypeError):
            _ = LinearDamper(self.damping, None)

    @pytest.mark.parametrize(
        'property_name, fixture_attr_name',
        [
            ('damping', 'damping'),
            ('pathway', 'pathway'),
        ]
    )
    def test_properties_are_immutable(self, property_name, fixture_attr_name):
        damper = LinearDamper(self.damping, self.pathway)
        value = getattr(self, fixture_attr_name)
        with pytest.raises(AttributeError):
            setattr(damper, property_name, value)

    def test_repr(self):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        damper = LinearDamper(self.damping, self.pathway)
        expected = 'LinearDamper(c, LinearPathway(pA, pB))'
        assert repr(damper) == expected

    def test_to_loads(self):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        damper = LinearDamper(self.damping, self.pathway)
        direction = self.q**2/self.q**2*self.N.x
        pA_force = self.damping*self.dq*direction
        pB_force = -self.damping*self.dq*direction
        expected = [Force(self.pA, pA_force), Force(self.pB, pB_force)]
        assert damper.to_loads() == expected


class TestForcedMassSpringDamperModel():
    r"""A single degree of freedom translational forced mass-spring-damper.

    Notes
    =====

    This system is well known to have the governing equation:

    .. math::
        m \ddot{x} = F - k x - c \dot{x}

    where $F$ is an externally applied force, $m$ is the mass of the particle
    to which the spring and damper are attached, $k$ is the spring's stiffness,
    $c$ is the dampers damping coefficient, and $x$ is the generalized
    coordinate representing the system's single (translational) degree of
    freedom.

    """

    @pytest.fixture(autouse=True)
    def _force_mass_spring_damper_model_fixture(self):
        self.m = Symbol('m')
        self.k = Symbol('k')
        self.c = Symbol('c')
        self.F = Symbol('F')

        self.q = dynamicsymbols('q')
        self.dq = dynamicsymbols('q', 1)
        self.u = dynamicsymbols('u')

        self.frame = ReferenceFrame('N')
        self.origin = Point('pO')
        self.origin.set_vel(self.frame, 0)

        self.attachment = Point('pA')
        self.attachment.set_pos(self.origin, self.q*self.frame.x)

        self.mass = Particle('mass', self.attachment, self.m)
        self.pathway = LinearPathway(self.origin, self.attachment)

        self.kanes_method = KanesMethod(
            self.frame,
            q_ind=[self.q],
            u_ind=[self.u],
            kd_eqs=[self.dq - self.u],
        )
        self.bodies = [self.mass]

        self.mass_matrix = Matrix([[self.m]])
        self.forcing = Matrix([[self.F - self.c*self.u - self.k*self.q]])

    def test_force_acuator(self):
        stiffness = -self.k*self.pathway.length
        spring = ForceActuator(stiffness, self.pathway)
        damping = -self.c*self.pathway.extension_velocity
        damper = ForceActuator(damping, self.pathway)

        loads = [
            (self.attachment, self.F*self.frame.x),
            *spring.to_loads(),
            *damper.to_loads(),
        ]
        self.kanes_method.kanes_equations(self.bodies, loads)

        assert self.kanes_method.mass_matrix == self.mass_matrix
        assert self.kanes_method.forcing == self.forcing

    def test_linear_spring_linear_damper(self):
        spring = LinearSpring(self.k, self.pathway)
        damper = LinearDamper(self.c, self.pathway)

        loads = [
            (self.attachment, self.F*self.frame.x),
            *spring.to_loads(),
            *damper.to_loads(),
        ]
        self.kanes_method.kanes_equations(self.bodies, loads)

        assert self.kanes_method.mass_matrix == self.mass_matrix
        assert self.kanes_method.forcing == self.forcing


class TestTorqueActuator:

    @pytest.fixture(autouse=True)
    def _torque_actuator_fixture(self):
        self.torque = Symbol('T')
        self.N = ReferenceFrame('N')
        self.A = ReferenceFrame('A')
        self.axis = self.N.z
        self.target = RigidBody('target', frame=self.N)
        self.reaction = RigidBody('reaction', frame=self.A)

    def test_is_actuator_base_subclass(self):
        assert issubclass(TorqueActuator, ActuatorBase)

    @pytest.mark.parametrize(
        'torque',
        [
            Symbol('T'),
            dynamicsymbols('T'),
            Symbol('T')**2 + Symbol('T'),
        ]
    )
    @pytest.mark.parametrize(
        'target_frame, reaction_frame',
        [
            (target.frame, reaction.frame),
            (target, reaction.frame),
            (target.frame, reaction),
            (target, reaction),
        ]
    )
    def test_valid_constructor_with_reaction(
        self,
        torque,
        target_frame,
        reaction_frame,
    ):
        instance = TorqueActuator(
            torque,
            self.axis,
            target_frame,
            reaction_frame,
        )
        assert isinstance(instance, TorqueActuator)

        assert hasattr(instance, 'torque')
        assert isinstance(instance.torque, ExprType)
        assert instance.torque == torque

        assert hasattr(instance, 'axis')
        assert isinstance(instance.axis, Vector)
        assert instance.axis == self.axis

        assert hasattr(instance, 'target_frame')
        assert isinstance(instance.target_frame, ReferenceFrame)
        assert instance.target_frame == target.frame

        assert hasattr(instance, 'reaction_frame')
        assert isinstance(instance.reaction_frame, ReferenceFrame)
        assert instance.reaction_frame == reaction.frame

    @pytest.mark.parametrize(
        'torque',
        [
            Symbol('T'),
            dynamicsymbols('T'),
            Symbol('T')**2 + Symbol('T'),
        ]
    )
    @pytest.mark.parametrize('target_frame', [target.frame, target])
    def test_valid_constructor_without_reaction(self, torque, target_frame):
        instance = TorqueActuator(torque, self.axis, target_frame)
        assert isinstance(instance, TorqueActuator)

        assert hasattr(instance, 'torque')
        assert isinstance(instance.torque, ExprType)
        assert instance.torque == torque

        assert hasattr(instance, 'axis')
        assert isinstance(instance.axis, Vector)
        assert instance.axis == self.axis

        assert hasattr(instance, 'target_frame')
        assert isinstance(instance.target_frame, ReferenceFrame)
        assert instance.target_frame == target.frame

        assert hasattr(instance, 'reaction_frame')
        assert instance.reaction_frame is None

    @pytest.mark.parametrize('torque', [None, 'T'])
    def test_invalid_constructor_torque_not_sympifyable(self, torque):
        with pytest.raises(SympifyError):
            _ = TorqueActuator(torque, self.axis, self.target)

    @pytest.mark.parametrize('axis', [Symbol('a'), dynamicsymbols('a')])
    def test_invalid_constructor_axis_not_vector(self, axis):
        with pytest.raises(TypeError):
            _ = TorqueActuator(self.torque, axis, self.target, self.reaction)

    @pytest.mark.parametrize(
        'frames',
        [
            (None, ReferenceFrame('child')),
            (ReferenceFrame('parent'), True),
            (None, RigidBody('child')),
            (RigidBody('parent'), True),
        ]
    )
    def test_invalid_constructor_frames_not_frame(self, frames):
        with pytest.raises(TypeError):
            _ = TorqueActuator(self.torque, self.axis, *frames)

    @pytest.mark.parametrize(
        'property_name, fixture_attr_name',
        [
            ('torque', 'torque'),
            ('axis', 'axis'),
            ('target_frame', 'target'),
            ('reaction_frame', 'reaction'),
        ]
    )
    def test_properties_are_immutable(self, property_name, fixture_attr_name):
        actuator = TorqueActuator(
            self.torque,
            self.axis,
            self.target,
            self.reaction,
        )
        value = getattr(self, fixture_attr_name)
        with pytest.raises(AttributeError):
            setattr(actuator, property_name, value)

    def test_repr_without_reaction(self):
        actuator = TorqueActuator(self.torque, self.axis, self.target)
        expected = 'TorqueActuator(T, axis=N.z, target_frame=N)'
        assert repr(actuator) == expected

    def test_repr_with_reaction(self):
        actuator = TorqueActuator(
            self.torque,
            self.axis,
            self.target,
            self.reaction,
        )
        expected = 'TorqueActuator(T, axis=N.z, target_frame=N, reaction_frame=A)'
        assert repr(actuator) == expected

    def test_at_pin_joint_constructor(self):
        pin_joint = PinJoint(
            'pin',
            self.target,
            self.reaction,
            coordinates=dynamicsymbols('q'),
            speeds=dynamicsymbols('u'),
            parent_interframe=self.N,
            joint_axis=self.axis,
        )
        instance = TorqueActuator.at_pin_joint(self.torque, pin_joint)
        assert isinstance(instance, TorqueActuator)

        assert hasattr(instance, 'torque')
        assert isinstance(instance.torque, ExprType)
        assert instance.torque == self.torque

        assert hasattr(instance, 'axis')
        assert isinstance(instance.axis, Vector)
        assert instance.axis == self.axis

        assert hasattr(instance, 'target_frame')
        assert isinstance(instance.target_frame, ReferenceFrame)
        assert instance.target_frame == self.A

        assert hasattr(instance, 'reaction_frame')
        assert isinstance(instance.reaction_frame, ReferenceFrame)
        assert instance.reaction_frame == self.N

    def test_at_pin_joint_pin_joint_not_pin_joint_invalid(self):
        with pytest.raises(TypeError):
            _ = TorqueActuator.at_pin_joint(self.torque, Symbol('pin'))

    def test_to_loads_without_reaction(self):
        actuator = TorqueActuator(self.torque, self.axis, self.target)
        expected = [
            (self.N, self.torque*self.axis),
        ]
        assert actuator.to_loads() == expected

    def test_to_loads_with_reaction(self):
        actuator = TorqueActuator(
            self.torque,
            self.axis,
            self.target,
            self.reaction,
        )
        expected = [
            (self.N, self.torque*self.axis),
            (self.A, - self.torque*self.axis),
        ]
        assert actuator.to_loads() == expected


class NonSympifyable:
    pass


class TestDuffingSpring:
    @pytest.fixture(autouse=True)
    # Set up common variables that will be used in multiple tests
    def _duffing_spring_fixture(self):
        self.linear_stiffness = Symbol('beta')
        self.nonlinear_stiffness = Symbol('alpha')
        self.equilibrium_length = Symbol('l')
        self.pA = Point('pA')
        self.pB = Point('pB')
        self.pathway = LinearPathway(self.pA, self.pB)
        self.q = dynamicsymbols('q')
        self.N = ReferenceFrame('N')

    # Simples tests to check that DuffingSpring is a subclass of ForceActuator and ActuatorBase
    def test_is_force_actuator_subclass(self):
        assert issubclass(DuffingSpring, ForceActuator)

    def test_is_actuator_base_subclass(self):
        assert issubclass(DuffingSpring, ActuatorBase)

    @pytest.mark.parametrize(
    # Create parametrized tests that allows running the same test function multiple times with different sets of arguments
    (
        'linear_stiffness,  '
        'expected_linear_stiffness,  '
        'nonlinear_stiffness,   '
        'expected_nonlinear_stiffness,  '
        'equilibrium_length,    '
        'expected_equilibrium_length,   '
        'force'
    ),
    [
            (
                1,
                S.One,
                1,
                S.One,
                0,
                S.Zero,
                -sqrt(dynamicsymbols('q')**2)-(sqrt(dynamicsymbols('q')**2))**3,
            ),
            (
                Symbol('beta'),
                Symbol('beta'),
                Symbol('alpha'),
                Symbol('alpha'),
                0,
                S.Zero,
                -Symbol('beta')*sqrt(dynamicsymbols('q')**2)-Symbol('alpha')*(sqrt(dynamicsymbols('q')**2))**3,
            ),
            (
                Symbol('beta'),
                Symbol('beta'),
                Symbol('alpha'),
                Symbol('alpha'),
                S.Zero,
                S.Zero,
                -Symbol('beta')*sqrt(dynamicsymbols('q')**2)-Symbol('alpha')*(sqrt(dynamicsymbols('q')**2))**3,
            ),
            (
                Symbol('beta'),
                Symbol('beta'),
                Symbol('alpha'),
                Symbol('alpha'),
                Symbol('l'),
                Symbol('l'),
                -Symbol('beta') * (sqrt(dynamicsymbols('q')**2) - Symbol('l')) - Symbol('alpha') * (sqrt(dynamicsymbols('q')**2) - Symbol('l'))**3,
            ),
        ]
    )

    # Check if DuffingSpring correctly initializes its attributes
    # It tests various combinations of linear & nonlinear stiffness, equilibriun length, and the resulting force expression
    def test_valid_constructor(
        self,
        linear_stiffness,
        expected_linear_stiffness,
        nonlinear_stiffness,
        expected_nonlinear_stiffness,
        equilibrium_length,
        expected_equilibrium_length,
        force,
    ):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        spring = DuffingSpring(linear_stiffness, nonlinear_stiffness, self.pathway, equilibrium_length)

        assert isinstance(spring, DuffingSpring)

        assert hasattr(spring, 'linear_stiffness')
        assert isinstance(spring.linear_stiffness, ExprType)
        assert spring.linear_stiffness == expected_linear_stiffness

        assert hasattr(spring, 'nonlinear_stiffness')
        assert isinstance(spring.nonlinear_stiffness, ExprType)
        assert spring.nonlinear_stiffness == expected_nonlinear_stiffness

        assert hasattr(spring, 'pathway')
        assert isinstance(spring.pathway, LinearPathway)
        assert spring.pathway == self.pathway

        assert hasattr(spring, 'equilibrium_length')
        assert isinstance(spring.equilibrium_length, ExprType)
        assert spring.equilibrium_length == expected_equilibrium_length

        assert hasattr(spring, 'force')
        assert isinstance(spring.force, ExprType)
        assert spring.force == force

    @pytest.mark.parametrize('linear_stiffness', [None, NonSympifyable()])
    def test_invalid_constructor_linear_stiffness_not_sympifyable(self, linear_stiffness):
        with pytest.raises(SympifyError):
            _ = DuffingSpring(linear_stiffness, self.nonlinear_stiffness, self.pathway, self.equilibrium_length)

    @pytest.mark.parametrize('nonlinear_stiffness', [None, NonSympifyable()])
    def test_invalid_constructor_nonlinear_stiffness_not_sympifyable(self, nonlinear_stiffness):
        with pytest.raises(SympifyError):
            _ = DuffingSpring(self.linear_stiffness, nonlinear_stiffness, self.pathway, self.equilibrium_length)

    def test_invalid_constructor_pathway_not_pathway_base(self):
        with pytest.raises(TypeError):
            _ = DuffingSpring(self.linear_stiffness, self.nonlinear_stiffness, NonSympifyable(), self.equilibrium_length)

    @pytest.mark.parametrize('equilibrium_length', [None, NonSympifyable()])
    def test_invalid_constructor_equilibrium_length_not_sympifyable(self, equilibrium_length):
        with pytest.raises(SympifyError):
            _ = DuffingSpring(self.linear_stiffness, self.nonlinear_stiffness, self.pathway, equilibrium_length)

    @pytest.mark.parametrize(
        'property_name, fixture_attr_name',
        [
            ('linear_stiffness', 'linear_stiffness'),
            ('nonlinear_stiffness', 'nonlinear_stiffness'),
            ('pathway', 'pathway'),
            ('equilibrium_length', 'equilibrium_length')
        ]
    )
    # Check if certain properties of DuffingSpring object are immutable after initialization
    # Ensure that once DuffingSpring is created, its key properties cannot be changed
    def test_properties_are_immutable(self, property_name, fixture_attr_name):
        spring = DuffingSpring(self.linear_stiffness, self.nonlinear_stiffness, self.pathway, self.equilibrium_length)
        with pytest.raises(AttributeError):
            setattr(spring, property_name, getattr(self, fixture_attr_name))

    @pytest.mark.parametrize(
        'equilibrium_length, expected',
        [
            (0, 'DuffingSpring(beta, alpha, LinearPathway(pA, pB), equilibrium_length=0)'),
            (Symbol('l'), 'DuffingSpring(beta, alpha, LinearPathway(pA, pB), equilibrium_length=l)'),
        ]
    )
    # Check the __repr__ method of DuffingSpring class
    # Check if the actual string representation of DuffingSpring instance matches the expected string for each provided parameter values
    def test_repr(self, equilibrium_length, expected):
        spring = DuffingSpring(self.linear_stiffness, self.nonlinear_stiffness, self.pathway, equilibrium_length)
        assert repr(spring) == expected

    def test_to_loads(self):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        spring = DuffingSpring(self.linear_stiffness, self.nonlinear_stiffness, self.pathway, self.equilibrium_length)

        # Calculate the displacement from the equilibrium length
        displacement = self.q - self.equilibrium_length

        # Make sure this matches the computation in DuffingSpring class
        force = -self.linear_stiffness * displacement - self.nonlinear_stiffness * displacement**3

        # The expected loads on pA and pB due to the spring
        expected_loads = [Force(self.pA, force * self.N.x), Force(self.pB, -force * self.N.x)]

        # Compare expected loads to what is returned from DuffingSpring.to_loads()
        calculated_loads = spring.to_loads()
        for calculated, expected in zip(calculated_loads, expected_loads):
            assert calculated.point == expected.point
            for dim in self.N:  # Assuming self.N is the reference frame
                calculated_component = calculated.vector.dot(dim)
                expected_component = expected.vector.dot(dim)
                # Substitute all symbols with numeric values
                substitutions = {self.q: 1, Symbol('l'): 1, Symbol('alpha'): 1, Symbol('beta'): 1}  # Add other necessary symbols as needed
                diff = (calculated_component - expected_component).subs(substitutions).evalf()
                # Check if the absolute value of the difference is below a threshold
                assert Abs(diff) < 1e-9, f"The forces do not match. Difference: {diff}"

class TestCoulombKineticFriction:
    @pytest.fixture(autouse=True)
    def _block_on_surface(self):
        """A block sliding on a surface.

        Notes
        =====
        This test validates the correctness of the CoulombKineticFriction by simulating
        a block sliding on a surface with the Coulomb kinetic friction force.
        The test covers scenarios with both positive and negative velocities.

        """

        # Mass, gravity constant, friction coefficient, coefficient of Stribeck friction, viscous_coefficient
        self.m, self.g, self.mu_k, self.mu_s, self.v_s, self.sigma, self.F = symbols('m g mu_k mu_s v_s sigma F', real=True)

    def test_block_on_surface_default(self):
        # General Case
        q = dynamicsymbols('q')

        N = ReferenceFrame('N')
        O = Point('O')
        P = O.locatenew('P', q * N.x)
        O.set_vel(N, 0)
        P.set_vel(N, q.diff() * N.x)

        pathway = LinearPathway(O, P)
        friction = CoulombKineticFriction(self.mu_k, self.m * self.g, pathway)
        expected_general = [Force(point=O, force=self.g * self.m * self.mu_k * q * sign(sqrt(q**2) * q.diff()/q)/sqrt(q**2) * N.x),
                            Force(point=P, force=-self.g * self.m * self.mu_k * q * sign(sqrt(q**2) * q.diff()/q)/sqrt(q**2) * N.x)]

        assert friction.to_loads() == expected_general

        # Positive
        q = dynamicsymbols('q', positive=True)

        N = ReferenceFrame('N')
        O = Point('O')
        P = O.locatenew('P', q * N.x)
        O.set_vel(N, 0)
        P.set_vel(N, q.diff() * N.x)

        pathway = LinearPathway(O, P)
        friction = CoulombKineticFriction(self.mu_k, self.m * self.g, pathway)
        expected_positive = [Force(point=O, force=self.g * self.m * self.mu_k * sign(q.diff()) * N.x),
                             Force(point=P, force=-self.g * self.m * self.mu_k * sign(q.diff()) * N.x)]

        assert friction.to_loads() == expected_positive

        # Negative
        q = dynamicsymbols('q', positive=False)

        N = ReferenceFrame('N')
        O = Point('O')
        P = O.locatenew('P', q * N.x)
        O.set_vel(N, 0)
        P.set_vel(N, q.diff() * N.x)

        pathway = LinearPathway(O, P)
        friction = CoulombKineticFriction(self.mu_k, self.m * self.g, pathway)
        expected_negative = [Force(point=O, force=self.g * self.m * self.mu_k * q * sign(sqrt(q**2) * q.diff()/q)/sqrt(q**2)*N.x),
                             Force(point=P, force=-self.g * self.m * self.mu_k * q * sign(sqrt(q**2) * q.diff()/q)/sqrt(q**2)*N.x)]

        assert friction.to_loads() == expected_negative

    def test_block_on_surface_viscous(self):
        # General Case
        q = dynamicsymbols('q')

        N = ReferenceFrame('N')
        O = Point('O')
        P = O.locatenew('P', q * N.x)
        O.set_vel(N, 0)
        P.set_vel(N, q.diff() * N.x)

        pathway = LinearPathway(O, P)
        friction = CoulombKineticFriction(self.mu_k, self.m * self.g, pathway, sigma=self.sigma)
        expected_general = [Force(point=O, force=(self.g * self.m * self.mu_k * sign(sqrt(q**2) * q.diff()/q) + self.sigma * sqrt(q**2) * q.diff()/q) * q/sqrt(q**2) * N.x),
                            Force(point=P, force=(-self.g * self.m * self.mu_k * sign(sqrt(q**2) * q.diff()/q) - self.sigma * sqrt(q**2) * q.diff()/q) * q/sqrt(q**2) * N.x)]

        assert friction.to_loads() == expected_general

        # Positive
        q = dynamicsymbols('q', positive=True)

        N = ReferenceFrame('N')
        O = Point('O')
        P = O.locatenew('P', q * N.x)
        O.set_vel(N, 0)
        P.set_vel(N, q.diff() * N.x)

        pathway = LinearPathway(O, P)
        friction = CoulombKineticFriction(self.mu_k, self.m * self.g, pathway, sigma=self.sigma)
        expected_positive = [Force(point=O, force=(self.g * self.m * self.mu_k * sign(q.diff()) + self.sigma * q.diff()) * N.x),
                             Force(point=P, force=(-self.g * self.m * self.mu_k * sign(q.diff()) - self.sigma * q.diff()) * N.x)]

        assert friction.to_loads() == expected_positive

        # Negative
        q = dynamicsymbols('q', positive=False)

        N = ReferenceFrame('N')
        O = Point('O')
        P = O.locatenew('P', q * N.x)
        O.set_vel(N, 0)
        P.set_vel(N, q.diff() * N.x)

        pathway = LinearPathway(O, P)
        friction = CoulombKineticFriction(self.mu_k, self.m * self.g, pathway, sigma=self.sigma)
        expected_negative = [Force(point=O, force=(self.g * self.m * self.mu_k * sign(sqrt(q**2) * q.diff()/q) + self.sigma * sqrt(q**2) * q.diff()/q) * q/sqrt(q**2) * N.x),
                             Force(point=P, force=(-self.g * self.m * self.mu_k * sign(sqrt(q**2) * q.diff()/q) - self.sigma * sqrt(q**2) * q.diff()/q) * q/sqrt(q**2) * N.x)]

        assert friction.to_loads() == expected_negative

    def test_block_on_surface_stribeck(self):
        # General Case
        q = dynamicsymbols('q')

        N = ReferenceFrame('N')
        O = Point('O')
        P = O.locatenew('P', q * N.x)
        O.set_vel(N, 0)
        P.set_vel(N, q.diff() * N.x)

        pathway = LinearPathway(O, P)
        friction = CoulombKineticFriction(self.mu_k, self.m * self.g, pathway, v_s=self.v_s, mu_s=self.mu_s)
        expected_general = [Force(point=O, force=(self.g * self.m * self.mu_k + (-self.g * self.m * self.mu_k + self.g * self.m * self.mu_s) * exp(-q.diff()**2/self.v_s**2)) * q * sign(sqrt(q**2) * q.diff()/q)/sqrt(q**2) * N.x),
                            Force(point=P, force=- (self.g * self.m * self.mu_k + (-self.g * self.m * self.mu_k + self.g * self.m * self.mu_s) * exp(-q.diff()**2/self.v_s**2)) * q * sign(sqrt(q**2) * q.diff()/q)/sqrt(q**2) * N.x)]

        assert friction.to_loads() == expected_general

        # Positive
        q = dynamicsymbols('q', positive=True)

        N = ReferenceFrame('N')
        O = Point('O')
        P = O.locatenew('P', q * N.x)
        O.set_vel(N, 0)
        P.set_vel(N, q.diff() * N.x)

        pathway = LinearPathway(O, P)
        friction = CoulombKineticFriction(self.mu_k, self.m * self.g, pathway, v_s=self.v_s, mu_s=self.mu_s)
        expected_positive = [Force(point=O, force=(self.g * self.m * self.mu_k + (-self.g * self.m * self.mu_k + self.g * self.m * self.mu_s) * exp(-q.diff()**2/self.v_s**2)) * sign(q.diff()) * N.x),
                             Force(point=P, force=- (self.g * self.m * self.mu_k + (-self.g * self.m * self.mu_k + self.g * self.m * self.mu_s) * exp(-q.diff()**2/self.v_s**2)) * sign(q.diff()) * N.x)]

        assert friction.to_loads() == expected_positive

        # Negative
        q = dynamicsymbols('q', positive=False)

        N = ReferenceFrame('N')
        O = Point('O')
        P = O.locatenew('P', q * N.x)
        O.set_vel(N, 0)
        P.set_vel(N, q.diff() * N.x)

        pathway = LinearPathway(O, P)
        friction = CoulombKineticFriction(self.mu_k, self.m * self.g, pathway, v_s=self.v_s, mu_s=self.mu_s)
        expected_negative = [Force(point=O, force=(self.g * self.m * self.mu_k + (-self.g * self.m * self.mu_k + self.g * self.m * self.mu_s) * exp(-q.diff()**2/self.v_s**2)) * q * sign(sqrt(q**2) * q.diff()/q)/sqrt(q**2) * N.x),
                             Force(point=P, force=- (self.g * self.m * self.mu_k + (-self.g * self.m * self.mu_k + self.g * self.m * self.mu_s) * exp(-q.diff()**2/self.v_s**2)) * q * sign(sqrt(q**2) * q.diff()/q)/sqrt(q**2) * N.x)]

        assert friction.to_loads() == expected_negative

    def test_block_on_surface_all(self):
        # General Case
        q = dynamicsymbols('q')

        N = ReferenceFrame('N')
        O = Point('O')
        P = O.locatenew('P', q * N.x)
        O.set_vel(N, 0)
        P.set_vel(N, q.diff() * N.x)

        pathway = LinearPathway(O, P)
        friction = CoulombKineticFriction(self.mu_k, self.m * self.g, pathway, v_s=self.v_s, sigma=self.sigma, mu_s=self.mu_s)
        expected_general = [Force(point=O, force=(self.sigma * sqrt(q**2) * q.diff()/q + (self.g * self.m * self.mu_k + (-self.g * self.m * self.mu_k + self.g * self.m * self.mu_s) * exp(-q.diff()**2/self.v_s**2)) * sign(sqrt(q**2) * q.diff()/q)) * q/sqrt(q**2) * N.x),
                            Force(point=P, force=(-self.sigma * sqrt(q**2) * q.diff()/q - (self.g * self.m * self.mu_k + (-self.g * self.m * self.mu_k + self.g * self.m * self.mu_s) * exp(-q.diff()**2/self.v_s**2)) * sign(sqrt(q**2) * q.diff()/q)) * q/sqrt(q**2) * N.x)]

        assert friction.to_loads() == expected_general

        # Positive
        q = dynamicsymbols('q', positive=True)

        N = ReferenceFrame('N')
        O = Point('O')
        P = O.locatenew('P', q * N.x)
        O.set_vel(N, 0)
        P.set_vel(N, q.diff() * N.x)

        pathway = LinearPathway(O, P)
        friction = CoulombKineticFriction(self.mu_k, self.m * self.g, pathway, v_s=self.v_s, sigma=self.sigma, mu_s=self.mu_s)
        expected_positive = [Force(point=O, force=(self.sigma * q.diff() + (self.g * self.m * self.mu_k + (-self.g * self.m * self.mu_k + self.g * self.m * self.mu_s) * exp(-q.diff()**2/self.v_s**2)) * sign(q.diff())) * N.x),
                             Force(point=P, force=(-self.sigma * q.diff() - (self.g * self.m * self.mu_k + (-self.g * self.m * self.mu_k + self.g * self.m * self.mu_s) * exp(-q.diff()**2/self.v_s**2)) * sign(q.diff())) * N.x)]

        assert friction.to_loads() == expected_positive

        # Negative
        q = dynamicsymbols('q', positive=False)

        N = ReferenceFrame('N')
        O = Point('O')
        P = O.locatenew('P', q * N.x)
        O.set_vel(N, 0)
        P.set_vel(N, q.diff() * N.x)

        pathway = LinearPathway(O, P)
        friction = CoulombKineticFriction(self.mu_k, self.m * self.g, pathway, v_s=self.v_s, sigma=self.sigma, mu_s=self.mu_s)
        expected_negative = [Force(point=O, force=(self.sigma * sqrt(q**2) * q.diff()/q + (self.g * self.m * self.mu_k + (-self.g * self.m * self.mu_k + self.g * self.m * self.mu_s) * exp(-q.diff()**2/self.v_s**2)) * sign(sqrt(q**2) * q.diff()/q)) * q/sqrt(q**2) * N.x),
                             Force(point=P, force=(-self.sigma * sqrt(q**2) * q.diff()/q - (self.g * self.m * self.mu_k + (-self.g * self.m * self.mu_k + self.g * self.m * self.mu_s) * exp(-q.diff()**2/self.v_s**2)) * sign(sqrt(q**2) * q.diff()/q)) * q/sqrt(q**2) * N.x)]

        assert friction.to_loads() == expected_negative

    def test_normal_force_zero(self):
        q = dynamicsymbols('q')

        N = ReferenceFrame('N')
        O = Point('O')
        P = O.locatenew('P', q * N.x)
        O.set_vel(N, 0)
        P.set_vel(N, q.diff() * N.x)

        pathway = LinearPathway(O, P)
        friction = CoulombKineticFriction(
            self.mu_k,
            0,
            pathway
        )
        assert friction.force == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\diophantine\tests\test_diophantine.py ===
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix
from sympy.ntheory.factor_ import factorint
from sympy.simplify.powsimp import powsimp
from sympy.core.function import _mexpand
from sympy.core.sorting import default_sort_key, ordered
from sympy.functions.elementary.trigonometric import sin
from sympy.solvers.diophantine import diophantine
from sympy.solvers.diophantine.diophantine import (diop_DN,
    diop_solve, diop_ternary_quadratic_normal,
    diop_general_pythagorean, diop_ternary_quadratic, diop_linear,
    diop_quadratic, diop_general_sum_of_squares, diop_general_sum_of_even_powers,
    descent, diop_bf_DN, divisible, equivalent, find_DN, ldescent, length,
    reconstruct, partition, power_representation,
    prime_as_sum_of_two_squares, square_factor, sum_of_four_squares,
    sum_of_three_squares, transformation_to_DN, transformation_to_normal,
    classify_diop, base_solution_linear, cornacchia, sqf_normal, gaussian_reduce, holzer,
    check_param, parametrize_ternary_quadratic, sum_of_powers, sum_of_squares,
    _diop_ternary_quadratic_normal, _nint_or_floor,
    _odd, _even, _remove_gcd, _can_do_sum_of_squares, DiophantineSolutionSet, GeneralPythagorean,
    BinaryQuadratic)

from sympy.testing.pytest import slow, raises, XFAIL
from sympy.utilities.iterables import (
        signed_permutations)

a, b, c, d, p, q, x, y, z, w, t, u, v, X, Y, Z = symbols(
    "a, b, c, d, p, q, x, y, z, w, t, u, v, X, Y, Z", integer=True)
t_0, t_1, t_2, t_3, t_4, t_5, t_6 = symbols("t_:7", integer=True)
m1, m2, m3 = symbols('m1:4', integer=True)
n1 = symbols('n1', integer=True)


def diop_simplify(eq):
    return _mexpand(powsimp(_mexpand(eq)))


def test_input_format():
    raises(TypeError, lambda: diophantine(sin(x)))
    raises(TypeError, lambda: diophantine(x/pi - 3))


def test_nosols():
    # diophantine should sympify eq so that these are equivalent
    assert diophantine(3) == set()
    assert diophantine(S(3)) == set()


def test_univariate():
    assert diop_solve((x - 1)*(x - 2)**2) == {(1,), (2,)}
    assert diop_solve((x - 1)*(x - 2)) == {(1,), (2,)}


def test_classify_diop():
    raises(TypeError, lambda: classify_diop(x**2/3 - 1))
    raises(ValueError, lambda: classify_diop(1))
    raises(NotImplementedError, lambda: classify_diop(w*x*y*z - 1))
    raises(NotImplementedError, lambda: classify_diop(x**3 + y**3 + z**4 - 90))
    assert classify_diop(14*x**2 + 15*x - 42) == (
        [x], {1: -42, x: 15, x**2: 14}, 'univariate')
    assert classify_diop(x*y + z) == (
        [x, y, z], {x*y: 1, z: 1}, 'inhomogeneous_ternary_quadratic')
    assert classify_diop(x*y + z + w + x**2) == (
        [w, x, y, z], {x*y: 1, w: 1, x**2: 1, z: 1}, 'inhomogeneous_general_quadratic')
    assert classify_diop(x*y + x*z + x**2 + 1) == (
        [x, y, z], {x*y: 1, x*z: 1, x**2: 1, 1: 1}, 'inhomogeneous_general_quadratic')
    assert classify_diop(x*y + z + w + 42) == (
        [w, x, y, z], {x*y: 1, w: 1, 1: 42, z: 1}, 'inhomogeneous_general_quadratic')
    assert classify_diop(x*y + z*w) == (
        [w, x, y, z], {x*y: 1, w*z: 1}, 'homogeneous_general_quadratic')
    assert classify_diop(x*y**2 + 1) == (
        [x, y], {x*y**2: 1, 1: 1}, 'cubic_thue')
    assert classify_diop(x**4 + y**4 + z**4 - (1 + 16 + 81)) == (
        [x, y, z], {1: -98, x**4: 1, z**4: 1, y**4: 1}, 'general_sum_of_even_powers')
    assert classify_diop(x**2 + y**2 + z**2) == (
        [x, y, z], {x**2: 1, y**2: 1, z**2: 1}, 'homogeneous_ternary_quadratic_normal')


def test_linear():
    assert diop_solve(x) == (0,)
    assert diop_solve(1*x) == (0,)
    assert diop_solve(3*x) == (0,)
    assert diop_solve(x + 1) == (-1,)
    assert diop_solve(2*x + 1) == (None,)
    assert diop_solve(2*x + 4) == (-2,)
    assert diop_solve(y + x) == (t_0, -t_0)
    assert diop_solve(y + x + 0) == (t_0, -t_0)
    assert diop_solve(y + x - 0) == (t_0, -t_0)
    assert diop_solve(0*x - y - 5) == (-5,)
    assert diop_solve(3*y + 2*x - 5) == (3*t_0 - 5, -2*t_0 + 5)
    assert diop_solve(2*x - 3*y - 5) == (3*t_0 - 5, 2*t_0 - 5)
    assert diop_solve(-2*x - 3*y - 5) == (3*t_0 + 5, -2*t_0 - 5)
    assert diop_solve(7*x + 5*y) == (5*t_0, -7*t_0)
    assert diop_solve(2*x + 4*y) == (-2*t_0, t_0)
    assert diop_solve(4*x + 6*y - 4) == (3*t_0 - 2, -2*t_0 + 2)
    assert diop_solve(4*x + 6*y - 3) == (None, None)
    assert diop_solve(0*x + 3*y - 4*z + 5) == (4*t_0 + 5, 3*t_0 + 5)
    assert diop_solve(4*x + 3*y - 4*z + 5) == (t_0, 8*t_0 + 4*t_1 + 5, 7*t_0 + 3*t_1 + 5)
    assert diop_solve(4*x + 3*y - 4*z + 5, None) == (0, 5, 5)
    assert diop_solve(4*x + 2*y + 8*z - 5) == (None, None, None)
    assert diop_solve(5*x + 7*y - 2*z - 6) == (t_0, -3*t_0 + 2*t_1 + 6, -8*t_0 + 7*t_1 + 18)
    assert diop_solve(3*x - 6*y + 12*z - 9) == (2*t_0 + 3, t_0 + 2*t_1, t_1)
    assert diop_solve(6*w + 9*x + 20*y - z) == (t_0, t_1, t_1 + t_2, 6*t_0 + 29*t_1 + 20*t_2)

    # to ignore constant factors, use diophantine
    raises(TypeError, lambda: diop_solve(x/2))


def test_quadratic_simple_hyperbolic_case():
    # Simple Hyperbolic case: A = C = 0 and B != 0
    assert diop_solve(3*x*y + 34*x - 12*y + 1) == \
           {(-133, -11), (5, -57)}
    assert diop_solve(6*x*y + 2*x + 3*y + 1) == set()
    assert diop_solve(-13*x*y + 2*x - 4*y - 54) == {(27, 0)}
    assert diop_solve(-27*x*y - 30*x - 12*y - 54) == {(-14, -1)}
    assert diop_solve(2*x*y + 5*x + 56*y + 7) == {(-161, -3), (-47, -6), (-35, -12),
                                                  (-29, -69), (-27, 64), (-21, 7),
                                                  (-9, 1), (105, -2)}
    assert diop_solve(6*x*y + 9*x + 2*y + 3) == set()
    assert diop_solve(x*y + x + y + 1) == {(-1, t), (t, -1)}
    assert diophantine(48*x*y)


def test_quadratic_elliptical_case():
    # Elliptical case: B**2 - 4AC < 0

    assert diop_solve(42*x**2 + 8*x*y + 15*y**2 + 23*x + 17*y - 4915) == {(-11, -1)}
    assert diop_solve(4*x**2 + 3*y**2 + 5*x - 11*y + 12) == set()
    assert diop_solve(x**2 + y**2 + 2*x + 2*y + 2) == {(-1, -1)}
    assert diop_solve(15*x**2 - 9*x*y + 14*y**2 - 23*x - 14*y - 4950) == {(-15, 6)}
    assert diop_solve(10*x**2 + 12*x*y + 12*y**2 - 34) == \
           {(-1, -1), (-1, 2), (1, -2), (1, 1)}


def test_quadratic_parabolic_case():
    # Parabolic case: B**2 - 4AC = 0
    assert check_solutions(8*x**2 - 24*x*y + 18*y**2 + 5*x + 7*y + 16)
    assert check_solutions(8*x**2 - 24*x*y + 18*y**2 + 6*x + 12*y - 6)
    assert check_solutions(8*x**2 + 24*x*y + 18*y**2 + 4*x + 6*y - 7)
    assert check_solutions(-4*x**2 + 4*x*y - y**2 + 2*x - 3)
    assert check_solutions(x**2 + 2*x*y + y**2 + 2*x + 2*y + 1)
    assert check_solutions(x**2 - 2*x*y + y**2 + 2*x + 2*y + 1)
    assert check_solutions(y**2 - 41*x + 40)


def test_quadratic_perfect_square():
    # B**2 - 4*A*C > 0
    # B**2 - 4*A*C is a perfect square
    assert check_solutions(48*x*y)
    assert check_solutions(4*x**2 - 5*x*y + y**2 + 2)
    assert check_solutions(-2*x**2 - 3*x*y + 2*y**2 -2*x - 17*y + 25)
    assert check_solutions(12*x**2 + 13*x*y + 3*y**2 - 2*x + 3*y - 12)
    assert check_solutions(8*x**2 + 10*x*y + 2*y**2 - 32*x - 13*y - 23)
    assert check_solutions(4*x**2 - 4*x*y - 3*y- 8*x - 3)
    assert check_solutions(- 4*x*y - 4*y**2 - 3*y- 5*x - 10)
    assert check_solutions(x**2 - y**2 - 2*x - 2*y)
    assert check_solutions(x**2 - 9*y**2 - 2*x - 6*y)
    assert check_solutions(4*x**2 - 9*y**2 - 4*x - 12*y - 3)


def test_quadratic_non_perfect_square():
    # B**2 - 4*A*C is not a perfect square
    # Used check_solutions() since the solutions are complex expressions involving
    # square roots and exponents
    assert check_solutions(x**2 - 2*x - 5*y**2)
    assert check_solutions(3*x**2 - 2*y**2 - 2*x - 2*y)
    assert check_solutions(x**2 - x*y - y**2 - 3*y)
    assert check_solutions(x**2 - 9*y**2 - 2*x - 6*y)
    assert BinaryQuadratic(x**2 + y**2 + 2*x + 2*y + 2).solve() == {(-1, -1)}


def test_issue_9106():
    eq = -48 - 2*x*(3*x - 1) + y*(3*y - 1)
    v = (x, y)
    for sol in diophantine(eq):
        assert not diop_simplify(eq.xreplace(dict(zip(v, sol))))


def test_issue_18138():
    eq = x**2 - x - y**2
    v = (x, y)
    for sol in diophantine(eq):
        assert not diop_simplify(eq.xreplace(dict(zip(v, sol))))


@slow
def test_quadratic_non_perfect_slow():
    assert check_solutions(8*x**2 + 10*x*y - 2*y**2 - 32*x - 13*y - 23)
    # This leads to very large numbers.
    # assert check_solutions(5*x**2 - 13*x*y + y**2 - 4*x - 4*y - 15)
    assert check_solutions(-3*x**2 - 2*x*y + 7*y**2 - 5*x - 7)
    assert check_solutions(-4 - x + 4*x**2 - y - 3*x*y - 4*y**2)
    assert check_solutions(1 + 2*x + 2*x**2 + 2*y + x*y - 2*y**2)


def test_DN():
    # Most of the test cases were adapted from,
    # Solving the generalized Pell equation x**2 - D*y**2 = N, John P. Robertson, July 31, 2004.
    # https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf
    # others are verified using Wolfram Alpha.

    # Covers cases where D <= 0 or D > 0 and D is a square or N = 0
    # Solutions are straightforward in these cases.
    assert diop_DN(3, 0) == [(0, 0)]
    assert diop_DN(-17, -5) == []
    assert diop_DN(-19, 23) == [(2, 1)]
    assert diop_DN(-13, 17) == [(2, 1)]
    assert diop_DN(-15, 13) == []
    assert diop_DN(0, 5) == []
    assert diop_DN(0, 9) == [(3, t)]
    assert diop_DN(9, 0) == [(3*t, t)]
    assert diop_DN(16, 24) == []
    assert diop_DN(9, 180) == [(18, 4)]
    assert diop_DN(9, -180) == [(12, 6)]
    assert diop_DN(7, 0) == [(0, 0)]

    # When equation is x**2 + y**2 = N
    # Solutions are interchangeable
    assert diop_DN(-1, 5) == [(2, 1), (1, 2)]
    assert diop_DN(-1, 169) == [(12, 5), (5, 12), (13, 0), (0, 13)]

    # D > 0 and D is not a square

    # N = 1
    assert diop_DN(13, 1) == [(649, 180)]
    assert diop_DN(980, 1) == [(51841, 1656)]
    assert diop_DN(981, 1) == [(158070671986249, 5046808151700)]
    assert diop_DN(986, 1) == [(49299, 1570)]
    assert diop_DN(991, 1) == [(379516400906811930638014896080, 12055735790331359447442538767)]
    assert diop_DN(17, 1) == [(33, 8)]
    assert diop_DN(19, 1) == [(170, 39)]

    # N = -1
    assert diop_DN(13, -1) == [(18, 5)]
    assert diop_DN(991, -1) == []
    assert diop_DN(41, -1) == [(32, 5)]
    assert diop_DN(290, -1) == [(17, 1)]
    assert diop_DN(21257, -1) == [(13913102721304, 95427381109)]
    assert diop_DN(32, -1) == []

    # |N| > 1
    # Some tests were created using calculator at
    # http://www.numbertheory.org/php/patz.html

    assert diop_DN(13, -4) == [(3, 1), (393, 109), (36, 10)]
    # Source I referred returned (3, 1), (393, 109) and (-3, 1) as fundamental solutions
    # So (-3, 1) and (393, 109) should be in the same equivalent class
    assert equivalent(-3, 1, 393, 109, 13, -4) == True

    assert diop_DN(13, 27) == [(220, 61), (40, 11), (768, 213), (12, 3)]
    assert set(diop_DN(157, 12)) == {(13, 1), (10663, 851), (579160, 46222),
                                     (483790960, 38610722), (26277068347, 2097138361),
                                     (21950079635497, 1751807067011)}
    assert diop_DN(13, 25) == [(3245, 900)]
    assert diop_DN(192, 18) == []
    assert diop_DN(23, 13) == [(-6, 1), (6, 1)]
    assert diop_DN(167, 2) == [(13, 1)]
    assert diop_DN(167, -2) == []

    assert diop_DN(123, -2) == [(11, 1)]
    # One calculator returned [(11, 1), (-11, 1)] but both of these are in
    # the same equivalence class
    assert equivalent(11, 1, -11, 1, 123, -2)

    assert diop_DN(123, -23) == [(-10, 1), (10, 1)]

    assert diop_DN(0, 0, t) == [(0, t)]
    assert diop_DN(0, -1, t) == []


def test_bf_pell():
    assert diop_bf_DN(13, -4) == [(3, 1), (-3, 1), (36, 10)]
    assert diop_bf_DN(13, 27) == [(12, 3), (-12, 3), (40, 11), (-40, 11)]
    assert diop_bf_DN(167, -2) == []
    assert diop_bf_DN(1729, 1) == [(44611924489705, 1072885712316)]
    assert diop_bf_DN(89, -8) == [(9, 1), (-9, 1)]
    assert diop_bf_DN(21257, -1) == [(13913102721304, 95427381109)]
    assert diop_bf_DN(340, -4) == [(756, 41)]
    assert diop_bf_DN(-1, 0, t) == [(0, 0)]
    assert diop_bf_DN(0, 0, t) == [(0, t)]
    assert diop_bf_DN(4, 0, t) == [(2*t, t), (-2*t, t)]
    assert diop_bf_DN(3, 0, t) == [(0, 0)]
    assert diop_bf_DN(1, -2, t) == []


def test_length():
    assert length(2, 1, 0) == 1
    assert length(-2, 4, 5) == 3
    assert length(-5, 4, 17) == 4
    assert length(0, 4, 13) == 6
    assert length(7, 13, 11) == 23
    assert length(1, 6, 4) == 2


def is_pell_transformation_ok(eq):
    """
    Test whether X*Y, X, or Y terms are present in the equation
    after transforming the equation using the transformation returned
    by transformation_to_pell(). If they are not present we are good.
    Moreover, coefficient of X**2 should be a divisor of coefficient of
    Y**2 and the constant term.
    """
    A, B = transformation_to_DN(eq)
    u = (A*Matrix([X, Y]) + B)[0]
    v = (A*Matrix([X, Y]) + B)[1]
    simplified = diop_simplify(eq.subs(zip((x, y), (u, v))))

    coeff = dict([reversed(t.as_independent(*[X, Y])) for t in simplified.args])

    for term in [X*Y, X, Y]:
        if term in coeff.keys():
            return False

    for term in [X**2, Y**2, 1]:
        if term not in coeff.keys():
            coeff[term] = 0

    if coeff[X**2] != 0:
        return divisible(coeff[Y**2], coeff[X**2]) and \
        divisible(coeff[1], coeff[X**2])

    return True


def test_transformation_to_pell():
    assert is_pell_transformation_ok(-13*x**2 - 7*x*y + y**2 + 2*x - 2*y - 14)
    assert is_pell_transformation_ok(-17*x**2 + 19*x*y - 7*y**2 - 5*x - 13*y - 23)
    assert is_pell_transformation_ok(x**2 - y**2 + 17)
    assert is_pell_transformation_ok(-x**2 + 7*y**2 - 23)
    assert is_pell_transformation_ok(25*x**2 - 45*x*y + 5*y**2 - 5*x - 10*y + 5)
    assert is_pell_transformation_ok(190*x**2 + 30*x*y + y**2 - 3*y - 170*x - 130)
    assert is_pell_transformation_ok(x**2 - 2*x*y -190*y**2 - 7*y - 23*x - 89)
    assert is_pell_transformation_ok(15*x**2 - 9*x*y + 14*y**2 - 23*x - 14*y - 4950)


def test_find_DN():
    assert find_DN(x**2 - 2*x - y**2) == (1, 1)
    assert find_DN(x**2 - 3*y**2 - 5) == (3, 5)
    assert find_DN(x**2 - 2*x*y - 4*y**2 - 7) == (5, 7)
    assert find_DN(4*x**2 - 8*x*y - y**2 - 9) == (20, 36)
    assert find_DN(7*x**2 - 2*x*y - y**2 - 12) == (8, 84)
    assert find_DN(-3*x**2 + 4*x*y -y**2) == (1, 0)
    assert find_DN(-13*x**2 - 7*x*y + y**2 + 2*x - 2*y -14) == (101, -7825480)


def test_ldescent():
    # Equations which have solutions
    u = ([(13, 23), (3, -11), (41, -113), (4, -7), (-7, 4), (91, -3), (1, 1), (1, -1),
        (4, 32), (17, 13), (123689, 1), (19, -570)])
    for a, b in u:
        w, x, y = ldescent(a, b)
        assert a*x**2 + b*y**2 == w**2
    assert ldescent(-1, -1) is None
    assert ldescent(2, 6) is None


def test_diop_ternary_quadratic_normal():
    assert check_solutions(234*x**2 - 65601*y**2 - z**2)
    assert check_solutions(23*x**2 + 616*y**2 - z**2)
    assert check_solutions(5*x**2 + 4*y**2 - z**2)
    assert check_solutions(3*x**2 + 6*y**2 - 3*z**2)
    assert check_solutions(x**2 + 3*y**2 - z**2)
    assert check_solutions(4*x**2 + 5*y**2 - z**2)
    assert check_solutions(x**2 + y**2 - z**2)
    assert check_solutions(16*x**2 + y**2 - 25*z**2)
    assert check_solutions(6*x**2 - y**2 + 10*z**2)
    assert check_solutions(213*x**2 + 12*y**2 - 9*z**2)
    assert check_solutions(34*x**2 - 3*y**2 - 301*z**2)
    assert check_solutions(124*x**2 - 30*y**2 - 7729*z**2)


def is_normal_transformation_ok(eq):
    A = transformation_to_normal(eq)
    X, Y, Z = A*Matrix([x, y, z])
    simplified = diop_simplify(eq.subs(zip((x, y, z), (X, Y, Z))))

    coeff = dict([reversed(t.as_independent(*[X, Y, Z])) for t in simplified.args])
    for term in [X*Y, Y*Z, X*Z]:
        if term in coeff.keys():
            return False

    return True


def test_transformation_to_normal():
    assert is_normal_transformation_ok(x**2 + 3*y**2 + z**2 - 13*x*y - 16*y*z + 12*x*z)
    assert is_normal_transformation_ok(x**2 + 3*y**2 - 100*z**2)
    assert is_normal_transformation_ok(x**2 + 23*y*z)
    assert is_normal_transformation_ok(3*y**2 - 100*z**2 - 12*x*y)
    assert is_normal_transformation_ok(x**2 + 23*x*y - 34*y*z + 12*x*z)
    assert is_normal_transformation_ok(z**2 + 34*x*y - 23*y*z + x*z)
    assert is_normal_transformation_ok(x**2 + y**2 + z**2 - x*y - y*z - x*z)
    assert is_normal_transformation_ok(x**2 + 2*y*z + 3*z**2)
    assert is_normal_transformation_ok(x*y + 2*x*z + 3*y*z)
    assert is_normal_transformation_ok(2*x*z + 3*y*z)


def test_diop_ternary_quadratic():
    assert check_solutions(2*x**2 + z**2 + y**2 - 4*x*y)
    assert check_solutions(x**2 - y**2 - z**2 - x*y - y*z)
    assert check_solutions(3*x**2 - x*y - y*z - x*z)
    assert check_solutions(x**2 - y*z - x*z)
    assert check_solutions(5*x**2 - 3*x*y - x*z)
    assert check_solutions(4*x**2 - 5*y**2 - x*z)
    assert check_solutions(3*x**2 + 2*y**2 - z**2 - 2*x*y + 5*y*z - 7*y*z)
    assert check_solutions(8*x**2 - 12*y*z)
    assert check_solutions(45*x**2 - 7*y**2 - 8*x*y - z**2)
    assert check_solutions(x**2 - 49*y**2 - z**2 + 13*z*y -8*x*y)
    assert check_solutions(90*x**2 + 3*y**2 + 5*x*y + 2*z*y + 5*x*z)
    assert check_solutions(x**2 + 3*y**2 + z**2 - x*y - 17*y*z)
    assert check_solutions(x**2 + 3*y**2 + z**2 - x*y - 16*y*z + 12*x*z)
    assert check_solutions(x**2 + 3*y**2 + z**2 - 13*x*y - 16*y*z + 12*x*z)
    assert check_solutions(x*y - 7*y*z + 13*x*z)

    assert diop_ternary_quadratic_normal(x**2 + y**2 + z**2) == (None, None, None)
    assert diop_ternary_quadratic_normal(x**2 + y**2) is None
    raises(ValueError, lambda:
        _diop_ternary_quadratic_normal((x, y, z),
        {x*y: 1, x**2: 2, y**2: 3, z**2: 0}))
    eq = -2*x*y - 6*x*z + 7*y**2 - 3*y*z + 4*z**2
    assert diop_ternary_quadratic(eq) == (7, 2, 0)
    assert diop_ternary_quadratic_normal(4*x**2 + 5*y**2 - z**2) == \
        (1, 0, 2)
    assert diop_ternary_quadratic(x*y + 2*y*z) == \
        (-2, 0, n1)
    eq = -5*x*y - 8*x*z - 3*y*z + 8*z**2
    assert parametrize_ternary_quadratic(eq) == \
        (8*p**2 - 3*p*q, -8*p*q + 8*q**2, 5*p*q)
    # this cannot be tested with diophantine because it will
    # factor into a product
    assert diop_solve(x*y + 2*y*z) == (-2*p*q, -n1*p**2 + p**2, p*q)


def test_square_factor():
    assert square_factor(1) == square_factor(-1) == 1
    assert square_factor(0) == 1
    assert square_factor(5) == square_factor(-5) == 1
    assert square_factor(4) == square_factor(-4) == 2
    assert square_factor(12) == square_factor(-12) == 2
    assert square_factor(6) == 1
    assert square_factor(18) == 3
    assert square_factor(52) == 2
    assert square_factor(49) == 7
    assert square_factor(392) == 14
    assert square_factor(factorint(-12)) == 2


def test_parametrize_ternary_quadratic():
    assert check_solutions(x**2 + y**2 - z**2)
    assert check_solutions(x**2 + 2*x*y + z**2)
    assert check_solutions(234*x**2 - 65601*y**2 - z**2)
    assert check_solutions(3*x**2 + 2*y**2 - z**2 - 2*x*y + 5*y*z - 7*y*z)
    assert check_solutions(x**2 - y**2 - z**2)
    assert check_solutions(x**2 - 49*y**2 - z**2 + 13*z*y - 8*x*y)
    assert check_solutions(8*x*y + z**2)
    assert check_solutions(124*x**2 - 30*y**2 - 7729*z**2)
    assert check_solutions(236*x**2 - 225*y**2 - 11*x*y - 13*y*z - 17*x*z)
    assert check_solutions(90*x**2 + 3*y**2 + 5*x*y + 2*z*y + 5*x*z)
    assert check_solutions(124*x**2 - 30*y**2 - 7729*z**2)


def test_no_square_ternary_quadratic():
    assert check_solutions(2*x*y + y*z - 3*x*z)
    assert check_solutions(189*x*y - 345*y*z - 12*x*z)
    assert check_solutions(23*x*y + 34*y*z)
    assert check_solutions(x*y + y*z + z*x)
    assert check_solutions(23*x*y + 23*y*z + 23*x*z)


def test_descent():

    u = ([(13, 23), (3, -11), (41, -113), (91, -3), (1, 1), (1, -1), (17, 13), (123689, 1), (19, -570)])
    for a, b in u:
        w, x, y = descent(a, b)
        assert a*x**2 + b*y**2 == w**2
    # the docstring warns against bad input, so these are expected results
    # - can't both be negative
    raises(TypeError, lambda: descent(-1, -3))
    # A can't be zero unless B != 1
    raises(ZeroDivisionError, lambda: descent(0, 3))
    # supposed to be square-free
    raises(TypeError, lambda: descent(4, 3))


def test_diophantine():
    assert check_solutions((x - y)*(y - z)*(z - x))
    assert check_solutions((x - y)*(x**2 + y**2 - z**2))
    assert check_solutions((x - 3*y + 7*z)*(x**2 + y**2 - z**2))
    assert check_solutions(x**2 - 3*y**2 - 1)
    assert check_solutions(y**2 + 7*x*y)
    assert check_solutions(x**2 - 3*x*y + y**2)
    assert check_solutions(z*(x**2 - y**2 - 15))
    assert check_solutions(x*(2*y - 2*z + 5))
    assert check_solutions((x**2 - 3*y**2 - 1)*(x**2 - y**2 - 15))
    assert check_solutions((x**2 - 3*y**2 - 1)*(y - 7*z))
    assert check_solutions((x**2 + y**2 - z**2)*(x - 7*y - 3*z + 4*w))
    # Following test case caused problems in parametric representation
    # But this can be solved by factoring out y.
    # No need to use methods for ternary quadratic equations.
    assert check_solutions(y**2 - 7*x*y + 4*y*z)
    assert check_solutions(x**2 - 2*x + 1)

    assert diophantine(x - y) == diophantine(Eq(x, y))
    # 18196
    eq = x**4 + y**4 - 97
    assert diophantine(eq, permute=True) == diophantine(-eq, permute=True)
    assert diophantine(3*x*pi - 2*y*pi) == {(2*t_0, 3*t_0)}
    eq = x**2 + y**2 + z**2 - 14
    base_sol = {(1, 2, 3)}
    assert diophantine(eq) == base_sol
    complete_soln = set(signed_permutations(base_sol.pop()))
    assert diophantine(eq, permute=True) == complete_soln

    assert diophantine(x**2 + x*Rational(15, 14) - 3) == set()
    # test issue 11049
    eq = 92*x**2 - 99*y**2 - z**2
    coeff = eq.as_coefficients_dict()
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == \
           {(9, 7, 51)}
    assert diophantine(eq) == {(
        891*p**2 + 9*q**2, -693*p**2 - 102*p*q + 7*q**2,
        5049*p**2 - 1386*p*q - 51*q**2)}
    eq = 2*x**2 + 2*y**2 - z**2
    coeff = eq.as_coefficients_dict()
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == \
           {(1, 1, 2)}
    assert diophantine(eq) == {(
        2*p**2 - q**2, -2*p**2 + 4*p*q - q**2,
        4*p**2 - 4*p*q + 2*q**2)}
    eq = 411*x**2+57*y**2-221*z**2
    coeff = eq.as_coefficients_dict()
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == \
           {(2021, 2645, 3066)}
    assert diophantine(eq) == \
           {(115197*p**2 - 446641*q**2, -150765*p**2 + 1355172*p*q -
             584545*q**2, 174762*p**2 - 301530*p*q + 677586*q**2)}
    eq = 573*x**2+267*y**2-984*z**2
    coeff = eq.as_coefficients_dict()
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == \
           {(49, 233, 127)}
    assert diophantine(eq) == \
           {(4361*p**2 - 16072*q**2, -20737*p**2 + 83312*p*q - 76424*q**2,
             11303*p**2 - 41474*p*q + 41656*q**2)}
    # this produces factors during reconstruction
    eq = x**2 + 3*y**2 - 12*z**2
    coeff = eq.as_coefficients_dict()
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == \
           {(0, 2, 1)}
    assert diophantine(eq) == \
           {(24*p*q, 2*p**2 - 24*q**2, p**2 + 12*q**2)}
    # solvers have not been written for every type
    raises(NotImplementedError, lambda: diophantine(x*y**2 + 1))

    # rational expressions
    assert diophantine(1/x) == set()
    assert diophantine(1/x + 1/y - S.Half) == {(6, 3), (-2, 1), (4, 4), (1, -2), (3, 6)}
    assert diophantine(x**2 + y**2 +3*x- 5, permute=True) == \
           {(-1, 1), (-4, -1), (1, -1), (1, 1), (-4, 1), (-1, -1), (4, 1), (4, -1)}


    #test issue 18186
    assert diophantine(y**4 + x**4 - 2**4 - 3**4, syms=(x, y), permute=True) == \
           {(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)}
    assert diophantine(y**4 + x**4 - 2**4 - 3**4, syms=(y, x), permute=True) == \
           {(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)}

    # issue 18122
    assert check_solutions(x**2 - y)
    assert check_solutions(y**2 - x)
    assert diophantine((x**2 - y), t) == {(t, t**2)}
    assert diophantine((y**2 - x), t) == {(t**2, t)}


def test_general_pythagorean():
    from sympy.abc import a, b, c, d, e

    assert check_solutions(a**2 + b**2 + c**2 - d**2)
    assert check_solutions(a**2 + 4*b**2 + 4*c**2 - d**2)
    assert check_solutions(9*a**2 + 4*b**2 + 4*c**2 - d**2)
    assert check_solutions(9*a**2 + 4*b**2 - 25*d**2 + 4*c**2 )
    assert check_solutions(9*a**2 - 16*d**2 + 4*b**2 + 4*c**2)
    assert check_solutions(-e**2 + 9*a**2 + 4*b**2 + 4*c**2 + 25*d**2)
    assert check_solutions(16*a**2 - b**2 + 9*c**2 + d**2 + 25*e**2)

    assert GeneralPythagorean(a**2 + b**2 + c**2 - d**2).solve(parameters=[x, y, z]) == \
           {(x**2 + y**2 - z**2, 2*x*z, 2*y*z, x**2 + y**2 + z**2)}


def test_diop_general_sum_of_squares_quick():
    for i in range(3, 10):
        assert check_solutions(sum(i**2 for i in symbols(':%i' % i)) - i)

    assert diop_general_sum_of_squares(x**2 + y**2 - 2) is None
    assert diop_general_sum_of_squares(x**2 + y**2 + z**2 + 2) == set()
    eq = x**2 + y**2 + z**2 - (1 + 4 + 9)
    assert diop_general_sum_of_squares(eq) == \
           {(1, 2, 3)}
    eq = u**2 + v**2 + x**2 + y**2 + z**2 - 1313
    assert len(diop_general_sum_of_squares(eq, 3)) == 3
    # issue 11016
    var = symbols(':5') + (symbols('6', negative=True),)
    eq = Add(*[i**2 for i in var]) - 112

    base_soln = {(0, 1, 1, 5, 6, -7), (1, 1, 1, 3, 6, -8), (2, 3, 3, 4, 5, -7), (0, 1, 1, 1, 3, -10),
                 (0, 0, 4, 4, 4, -8), (1, 2, 3, 3, 5, -8), (0, 1, 2, 3, 7, -7), (2, 2, 4, 4, 6, -6),
                 (1, 1, 3, 4, 6, -7), (0, 2, 3, 3, 3, -9), (0, 0, 2, 2, 2, -10), (1, 1, 2, 3, 4, -9),
                 (0, 1, 1, 2, 5, -9), (0, 0, 2, 6, 6, -6), (1, 3, 4, 5, 5, -6), (0, 2, 2, 2, 6, -8),
                 (0, 3, 3, 3, 6, -7), (0, 2, 3, 5, 5, -7), (0, 1, 5, 5, 5, -6)}
    assert diophantine(eq) == base_soln
    assert len(diophantine(eq, permute=True)) == 196800

    # handle negated squares with signsimp
    assert diophantine(12 - x**2 - y**2 - z**2) == {(2, 2, 2)}
    # diophantine handles simplification, so classify_diop should
    # not have to look for additional patterns that are removed
    # by diophantine
    eq = a**2 + b**2 + c**2 + d**2 - 4
    raises(NotImplementedError, lambda: classify_diop(-eq))


def test_issue_23807():
    # fixes recursion error
    eq = x**2 + y**2 + z**2 - 1000000
    base_soln = {(0, 0, 1000), (0, 352, 936), (480, 600, 640), (24, 640, 768), (192, 640, 744),
                 (192, 480, 856), (168, 224, 960), (0, 600, 800), (280, 576, 768), (152, 480, 864),
                 (0, 280, 960), (352, 360, 864), (424, 480, 768), (360, 480, 800), (224, 600, 768),
                 (96, 360, 928), (168, 576, 800), (96, 480, 872)}

    assert diophantine(eq) == base_soln


def test_diop_partition():
    for n in [8, 10]:
        for k in range(1, 8):
            for p in partition(n, k):
                assert len(p) == k
    assert list(partition(3, 5)) == []
    assert [list(p) for p in partition(3, 5, 1)] == [
        [0, 0, 0, 0, 3], [0, 0, 0, 1, 2], [0, 0, 1, 1, 1]]
    assert list(partition(0)) == [()]
    assert list(partition(1, 0)) == [()]
    assert [list(i) for i in partition(3)] == [[1, 1, 1], [1, 2], [3]]


def test_prime_as_sum_of_two_squares():
    for i in [5, 13, 17, 29, 37, 41, 2341, 3557, 34841, 64601]:
        a, b = prime_as_sum_of_two_squares(i)
        assert a**2 + b**2 == i
    assert prime_as_sum_of_two_squares(7) is None
    ans = prime_as_sum_of_two_squares(800029)
    assert ans == (450, 773) and type(ans[0]) is int


def test_sum_of_three_squares():
    for i in [0, 1, 2, 34, 123, 34304595905, 34304595905394941, 343045959052344,
              800, 801, 802, 803, 804, 805, 806]:
        a, b, c = sum_of_three_squares(i)
        assert a**2 + b**2 + c**2 == i
        assert a >= 0

    # error
    raises(ValueError, lambda: sum_of_three_squares(-1))

    assert sum_of_three_squares(7) is None
    assert sum_of_three_squares((4**5)*15) is None
    # if there are two zeros, there might be a solution
    # with only one zero, e.g. 25 => (0, 3, 4) or
    # with no zeros, e.g. 49 => (2, 3, 6)
    assert sum_of_three_squares(25) == (0, 0, 5)
    assert sum_of_three_squares(4) == (0, 0, 2)


def test_sum_of_four_squares():
    from sympy.core.random import randint

    # this should never fail
    n = randint(1, 100000000000000)
    assert sum(i**2 for i in sum_of_four_squares(n)) == n

    # error
    raises(ValueError, lambda: sum_of_four_squares(-1))

    for n in range(1000):
        result = sum_of_four_squares(n)
        assert len(result) == 4
        assert all(r >= 0 for r in result)
        assert sum(r**2 for r in result) == n
        assert list(result) == sorted(result)


def test_power_representation():
    tests = [(1729, 3, 2), (234, 2, 4), (2, 1, 2), (3, 1, 3), (5, 2, 2), (12352, 2, 4),
             (32760, 2, 3)]

    for test in tests:
        n, p, k = test
        f = power_representation(n, p, k)

        while True:
            try:
                l = next(f)
                assert len(l) == k

                chk_sum = 0
                for l_i in l:
                    chk_sum = chk_sum + l_i**p
                assert chk_sum == n

            except StopIteration:
                break

    assert list(power_representation(20, 2, 4, True)) == \
        [(1, 1, 3, 3), (0, 0, 2, 4)]
    raises(ValueError, lambda: list(power_representation(1.2, 2, 2)))
    raises(ValueError, lambda: list(power_representation(2, 0, 2)))
    raises(ValueError, lambda: list(power_representation(2, 2, 0)))
    assert list(power_representation(-1, 2, 2)) == []
    assert list(power_representation(1, 1, 1)) == [(1,)]
    assert list(power_representation(3, 2, 1)) == []
    assert list(power_representation(4, 2, 1)) == [(2,)]
    assert list(power_representation(3**4, 4, 6, zeros=True)) == \
        [(1, 2, 2, 2, 2, 2), (0, 0, 0, 0, 0, 3)]
    assert list(power_representation(3**4, 4, 5, zeros=False)) == []
    assert list(power_representation(-2, 3, 2)) == [(-1, -1)]
    assert list(power_representation(-2, 4, 2)) == []
    assert list(power_representation(0, 3, 2, True)) == [(0, 0)]
    assert list(power_representation(0, 3, 2, False)) == []
    # when we are dealing with squares, do feasibility checks
    assert len(list(power_representation(4**10*(8*10 + 7), 2, 3))) == 0
    # there will be a recursion error if these aren't recognized
    big = 2**30
    for i in [13, 10, 7, 5, 4, 2, 1]:
        assert list(sum_of_powers(big, 2, big - i)) == []


def test_assumptions():
    """
    Test whether diophantine respects the assumptions.
    """
    #Test case taken from the below so question regarding assumptions in diophantine module
    #https://stackoverflow.com/questions/23301941/how-can-i-declare-natural-symbols-with-sympy
    m, n = symbols('m n', integer=True, positive=True)
    diof = diophantine(n**2 + m*n - 500)
    assert diof == {(5, 20), (40, 10), (95, 5), (121, 4), (248, 2), (499, 1)}

    a, b = symbols('a b', integer=True, positive=False)
    diof = diophantine(a*b + 2*a + 3*b - 6)
    assert diof == {(-15, -3), (-9, -4), (-7, -5), (-6, -6), (-5, -8), (-4, -14)}

    x, y = symbols('x y', integer=True)
    diof = diophantine(10*x**2 + 5*x*y - 3*y)
    assert diof == {(1, -5), (-3, 5), (0, 0)}

    x, y = symbols('x y', integer=True, positive=True)
    diof = diophantine(10*x**2 + 5*x*y - 3*y)
    assert diof == set()

    x, y = symbols('x y', integer=True, negative=False)
    diof = diophantine(10*x**2 + 5*x*y - 3*y)
    assert diof == {(0, 0)}


def check_solutions(eq):
    """
    Determines whether solutions returned by diophantine() satisfy the original
    equation. Hope to generalize this so we can remove functions like check_ternay_quadratic,
    check_solutions_normal, check_solutions()
    """
    s = diophantine(eq)

    factors = Mul.make_args(eq)

    var = list(eq.free_symbols)
    var.sort(key=default_sort_key)

    while s:
        solution = s.pop()
        for f in factors:
            if diop_simplify(f.subs(zip(var, solution))) == 0:
                break
        else:
            return False
    return True


def test_diopcoverage():
    eq = (2*x + y + 1)**2
    assert diop_solve(eq) == {(t_0, -2*t_0 - 1)}
    eq = 2*x**2 + 6*x*y + 12*x + 4*y**2 + 18*y + 18
    assert diop_solve(eq) == {(t, -t - 3), (-2*t - 3, t)}
    assert diop_quadratic(x + y**2 - 3) == {(-t**2 + 3, t)}

    assert diop_linear(x + y - 3) == (t_0, 3 - t_0)

    assert base_solution_linear(0, 1, 2, t=None) == (0, 0)
    ans = (3*t - 1, -2*t + 1)
    assert base_solution_linear(4, 8, 12, t) == ans
    assert base_solution_linear(4, 8, 12, t=None) == tuple(_.subs(t, 0) for _ in ans)

    assert cornacchia(1, 1, 20) == set()
    assert cornacchia(1, 1, 5) == {(2, 1)}
    assert cornacchia(1, 2, 17) == {(3, 2)}

    raises(ValueError, lambda: reconstruct(4, 20, 1))

    assert gaussian_reduce(4, 1, 3) == (1, 1)
    eq = -w**2 - x**2 - y**2 + z**2

    assert diop_general_pythagorean(eq) == \
        diop_general_pythagorean(-eq) == \
            (m1**2 + m2**2 - m3**2, 2*m1*m3,
            2*m2*m3, m1**2 + m2**2 + m3**2)

    assert len(check_param(S(3) + x/3, S(4) + x/2, S(2), [x])) == 0
    assert len(check_param(Rational(3, 2), S(4) + x, S(2), [x])) == 0
    assert len(check_param(S(4) + x, Rational(3, 2), S(2), [x])) == 0

    assert _nint_or_floor(16, 10) == 2
    assert _odd(1) == (not _even(1)) == True
    assert _odd(0) == (not _even(0)) == False
    assert _remove_gcd(2, 4, 6) == (1, 2, 3)
    raises(TypeError, lambda: _remove_gcd((2, 4, 6)))
    assert sqf_normal(2*3**2*5, 2*5*11, 2*7**2*11)  == \
        (11, 1, 5)

    # it's ok if these pass some day when the solvers are implemented
    raises(NotImplementedError, lambda: diophantine(x**2 + y**2 + x*y + 2*y*z - 12))
    raises(NotImplementedError, lambda: diophantine(x**3 + y**2))
    assert diop_quadratic(x**2 + y**2 - 1**2 - 3**4) == \
           {(-9, -1), (-9, 1), (-1, -9), (-1, 9), (1, -9), (1, 9), (9, -1), (9, 1)}


def test_holzer():
    # if the input is good, don't let it diverge in holzer()
    # (but see test_fail_holzer below)
    assert holzer(2, 7, 13, 4, 79, 23) == (2, 7, 13)

    # None in uv condition met; solution is not Holzer reduced
    # so this will hopefully change but is here for coverage
    assert holzer(2, 6, 2, 1, 1, 10) == (2, 6, 2)

    raises(ValueError, lambda: holzer(2, 7, 14, 4, 79, 23))


@XFAIL
def test_fail_holzer():
    eq = lambda x, y, z: a*x**2 + b*y**2 - c*z**2
    a, b, c = 4, 79, 23
    x, y, z = xyz = 26, 1, 11
    X, Y, Z = ans = 2, 7, 13
    assert eq(*xyz) == 0
    assert eq(*ans) == 0
    assert max(a*x**2, b*y**2, c*z**2) <= a*b*c
    assert max(a*X**2, b*Y**2, c*Z**2) <= a*b*c
    h = holzer(x, y, z, a, b, c)
    assert h == ans  # it would be nice to get the smaller soln


def test_issue_9539():
    assert diophantine(6*w + 9*y + 20*x - z) == \
           {(t_0, t_1, t_1 + t_2, 6*t_0 + 29*t_1 + 9*t_2)}


def test_issue_8943():
    assert diophantine(
        3*(x**2 + y**2 + z**2) - 14*(x*y + y*z + z*x)) == \
           {(0, 0, 0)}


def test_diop_sum_of_even_powers():
    eq = x**4 + y**4 + z**4 - 2673
    assert diop_solve(eq) == {(3, 6, 6), (2, 4, 7)}
    assert diop_general_sum_of_even_powers(eq, 2) == {(3, 6, 6), (2, 4, 7)}
    raises(NotImplementedError, lambda: diop_general_sum_of_even_powers(-eq, 2))
    neg = symbols('neg', negative=True)
    eq = x**4 + y**4 + neg**4 - 2673
    assert diop_general_sum_of_even_powers(eq) == {(-3, 6, 6)}
    assert diophantine(x**4 + y**4 + 2) == set()
    assert diop_general_sum_of_even_powers(x**4 + y**4 - 2, limit=0) == set()


def test_sum_of_squares_powers():
    tru = {(0, 0, 1, 1, 11), (0, 0, 5, 7, 7), (0, 1, 3, 7, 8), (0, 1, 4, 5, 9), (0, 3, 4, 7, 7), (0, 3, 5, 5, 8),
           (1, 1, 2, 6, 9), (1, 1, 6, 6, 7), (1, 2, 3, 3, 10), (1, 3, 4, 4, 9), (1, 5, 5, 6, 6), (2, 2, 3, 5, 9),
           (2, 3, 5, 6, 7), (3, 3, 4, 5, 8)}
    eq = u**2 + v**2 + x**2 + y**2 + z**2 - 123
    ans = diop_general_sum_of_squares(eq, oo)  # allow oo to be used
    assert len(ans) == 14
    assert ans == tru

    raises(ValueError, lambda: list(sum_of_squares(10, -1)))
    assert list(sum_of_squares(1, 1)) == [(1,)]
    assert list(sum_of_squares(1, 2)) == []
    assert list(sum_of_squares(1, 2, True)) == [(0, 1)]
    assert list(sum_of_squares(-10, 2)) == []
    assert list(sum_of_squares(2, 3)) == []
    assert list(sum_of_squares(0, 3, True)) == [(0, 0, 0)]
    assert list(sum_of_squares(0, 3)) == []
    assert list(sum_of_squares(4, 1)) == [(2,)]
    assert list(sum_of_squares(5, 1)) == []
    assert list(sum_of_squares(50, 2)) == [(5, 5), (1, 7)]
    assert list(sum_of_squares(11, 5, True)) == [
        (1, 1, 1, 2, 2), (0, 0, 1, 1, 3)]
    assert list(sum_of_squares(8, 8)) == [(1, 1, 1, 1, 1, 1, 1, 1)]

    assert [len(list(sum_of_squares(i, 5, True))) for i in range(30)] == [
        1, 1, 1, 1, 2,
        2, 1, 1, 2, 2,
        2, 2, 2, 3, 2,
        1, 3, 3, 3, 3,
        4, 3, 3, 2, 2,
        4, 4, 4, 4, 5]
    assert [len(list(sum_of_squares(i, 5))) for i in range(30)] == [
        0, 0, 0, 0, 0,
        1, 0, 0, 1, 0,
        0, 1, 0, 1, 1,
        0, 1, 1, 0, 1,
        2, 1, 1, 1, 1,
        1, 1, 1, 1, 3]
    for i in range(30):
        s1 = set(sum_of_squares(i, 5, True))
        assert not s1 or all(sum(j**2 for j in t) == i for t in s1)
        s2 = set(sum_of_squares(i, 5))
        assert all(sum(j**2 for j in t) == i for t in s2)

    raises(ValueError, lambda: list(sum_of_powers(2, -1, 1)))
    raises(ValueError, lambda: list(sum_of_powers(2, 1, -1)))
    assert list(sum_of_powers(-2, 3, 2)) == [(-1, -1)]
    assert list(sum_of_powers(-2, 4, 2)) == []
    assert list(sum_of_powers(2, 1, 1)) == [(2,)]
    assert list(sum_of_powers(2, 1, 3, True)) == [(0, 0, 2), (0, 1, 1)]
    assert list(sum_of_powers(5, 1, 2, True)) == [(0, 5), (1, 4), (2, 3)]
    assert list(sum_of_powers(6, 2, 2)) == []
    assert list(sum_of_powers(3**5, 3, 1)) == []
    assert list(sum_of_powers(3**6, 3, 1)) == [(9,)] and (9**3 == 3**6)
    assert list(sum_of_powers(2**1000, 5, 2)) == []


def test__can_do_sum_of_squares():
    assert _can_do_sum_of_squares(3, -1) is False
    assert _can_do_sum_of_squares(-3, 1) is False
    assert _can_do_sum_of_squares(0, 1)
    assert _can_do_sum_of_squares(4, 1)
    assert _can_do_sum_of_squares(1, 2)
    assert _can_do_sum_of_squares(2, 2)
    assert _can_do_sum_of_squares(3, 2) is False


def test_diophantine_permute_sign():
    from sympy.abc import a, b, c, d, e
    eq = a**4 + b**4 - (2**4 + 3**4)
    base_sol = {(2, 3)}
    assert diophantine(eq) == base_sol
    complete_soln = set(signed_permutations(base_sol.pop()))
    assert diophantine(eq, permute=True) == complete_soln

    eq = a**2 + b**2 + c**2 + d**2 + e**2 - 234
    assert len(diophantine(eq)) == 35
    assert len(diophantine(eq, permute=True)) == 62000
    soln = {(-1, -1), (-1, 2), (1, -2), (1, 1)}
    assert diophantine(10*x**2 + 12*x*y + 12*y**2 - 34, permute=True) == soln


@XFAIL
def test_not_implemented():
    eq = x**2 + y**4 - 1**2 - 3**4
    assert diophantine(eq, syms=[x, y]) == {(9, 1), (1, 3)}


def test_issue_9538():
    eq = x - 3*y + 2
    assert diophantine(eq, syms=[y,x]) == {(t_0, 3*t_0 - 2)}
    raises(TypeError, lambda: diophantine(eq, syms={y, x}))


def test_ternary_quadratic():
    # solution with 3 parameters
    s = diophantine(2*x**2 + y**2 - 2*z**2)
    p, q, r = ordered(S(s).free_symbols)
    assert s == {(
        p**2 - 2*q**2,
        -2*p**2 + 4*p*q - 4*p*r - 4*q**2,
        p**2 - 4*p*q + 2*q**2 - 4*q*r)}
    # solution with Mul in solution
    s = diophantine(x**2 + 2*y**2 - 2*z**2)
    assert s == {(4*p*q, p**2 - 2*q**2, p**2 + 2*q**2)}
    # solution with no Mul in solution
    s = diophantine(2*x**2 + 2*y**2 - z**2)
    assert s == {(2*p**2 - q**2, -2*p**2 + 4*p*q - q**2,
        4*p**2 - 4*p*q + 2*q**2)}
    # reduced form when parametrized
    s = diophantine(3*x**2 + 72*y**2 - 27*z**2)
    assert s == {(24*p**2 - 9*q**2, 6*p*q, 8*p**2 + 3*q**2)}
    assert parametrize_ternary_quadratic(
        3*x**2 + 2*y**2 - z**2 - 2*x*y + 5*y*z - 7*y*z) == (
        2*p**2 - 2*p*q - q**2, 2*p**2 + 2*p*q - q**2, 2*p**2 -
        2*p*q + 3*q**2)
    assert parametrize_ternary_quadratic(
        124*x**2 - 30*y**2 - 7729*z**2) == (
        -1410*p**2 - 363263*q**2, 2700*p**2 + 30916*p*q -
        695610*q**2, -60*p**2 + 5400*p*q + 15458*q**2)


def test_diophantine_solution_set():
    s1 = DiophantineSolutionSet([], [])
    assert set(s1) == set()
    assert s1.symbols == ()
    assert s1.parameters == ()
    raises(ValueError, lambda: s1.add((x,)))
    assert list(s1.dict_iterator()) == []

    s2 = DiophantineSolutionSet([x, y], [t, u])
    assert s2.symbols == (x, y)
    assert s2.parameters == (t, u)
    raises(ValueError, lambda: s2.add((1,)))
    s2.add((3, 4))
    assert set(s2) == {(3, 4)}
    s2.update((3, 4), (-1, u))
    assert set(s2) == {(3, 4), (-1, u)}
    raises(ValueError, lambda: s1.update(s2))
    assert list(s2.dict_iterator()) == [{x: -1, y: u}, {x: 3, y: 4}]

    s3 = DiophantineSolutionSet([x, y, z], [t, u])
    assert len(s3.parameters) == 2
    s3.add((t**2 + u, t - u, 1))
    assert set(s3) == {(t**2 + u, t - u, 1)}
    assert s3.subs(t, 2) == {(u + 4, 2 - u, 1)}
    assert s3(2) == {(u + 4, 2 - u, 1)}
    assert s3.subs({t: 7, u: 8}) == {(57, -1, 1)}
    assert s3(7, 8) == {(57, -1, 1)}
    assert s3.subs({t: 5}) == {(u + 25, 5 - u, 1)}
    assert s3(5) == {(u + 25, 5 - u, 1)}
    assert s3.subs(u, -3) == {(t**2 - 3, t + 3, 1)}
    assert s3(None, -3) == {(t**2 - 3, t + 3, 1)}
    assert s3.subs({t: 2, u: 8}) == {(12, -6, 1)}
    assert s3(2, 8) == {(12, -6, 1)}
    assert s3.subs({t: 5, u: -3}) == {(22, 8, 1)}
    assert s3(5, -3) == {(22, 8, 1)}
    raises(TypeError, lambda: s3.subs(x=1))
    raises(TypeError, lambda: s3.subs(1, 2, 3))
    raises(ValueError, lambda: s3.add(()))
    raises(ValueError, lambda: s3.add((1, 2, 3, 4)))
    raises(ValueError, lambda: s3.add((1, 2)))
    raises(ValueError, lambda: s3(1, 2, 3))
    raises(TypeError, lambda: s3(t=1))

    s4 = DiophantineSolutionSet([x, y], [t, u])
    s4.add((t, 11*t))
    s4.add((-t, 22*t))
    assert s4(0, 0) == {(0, 0)}


def test_quadratic_parameter_passing():
    eq = -33*x*y + 3*y**2
    solution = BinaryQuadratic(eq).solve(parameters=[t, u])
    # test that parameters are passed all the way to the final solution
    assert solution == {(t, 11*t), (t, -22*t)}
    assert solution(0, 0) == {(0, 0)}

def test_issue_18628():
    eq1 = x**2 - 15*x + y**2 - 8*y
    sol = diophantine(eq1)
    assert sol == {(15, 0), (15, 8), (-1, 4), (0, 0), (0, 8), (16, 4)}
    eq2 = 2*x**2 - 9*x + 4*y**2 - 8*y + 14
    sol = diophantine(eq2)
    assert sol == {(2, 1)}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timestamp\test_constructors.py ===
import calendar
from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)
import zoneinfo

import dateutil.tz
from dateutil.tz import (
    gettz,
    tzoffset,
    tzutc,
)
import numpy as np
import pytest
import pytz

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.compat import PY310
from pandas.errors import OutOfBoundsDatetime

from pandas import (
    NA,
    NaT,
    Period,
    Timedelta,
    Timestamp,
)


class TestTimestampConstructorUnitKeyword:
    @pytest.mark.parametrize("typ", [int, float])
    def test_constructor_int_float_with_YM_unit(self, typ):
        # GH#47266 avoid the conversions in cast_from_unit
        val = typ(150)

        ts = Timestamp(val, unit="Y")
        expected = Timestamp("2120-01-01")
        assert ts == expected

        ts = Timestamp(val, unit="M")
        expected = Timestamp("1982-07-01")
        assert ts == expected

    @pytest.mark.parametrize("typ", [int, float])
    def test_construct_from_int_float_with_unit_out_of_bound_raises(self, typ):
        # GH#50870  make sure we get a OutOfBoundsDatetime instead of OverflowError
        val = typ(150000000000000)

        msg = f"cannot convert input {val} with the unit 'D'"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(val, unit="D")

    def test_constructor_float_not_round_with_YM_unit_raises(self):
        # GH#47267 avoid the conversions in cast_from-unit

        msg = "Conversion of non-round float with unit=[MY] is ambiguous"
        with pytest.raises(ValueError, match=msg):
            Timestamp(150.5, unit="Y")

        with pytest.raises(ValueError, match=msg):
            Timestamp(150.5, unit="M")

    @pytest.mark.parametrize(
        "value, check_kwargs",
        [
            [946688461000000000, {}],
            [946688461000000000 / 1000, {"unit": "us"}],
            [946688461000000000 / 1_000_000, {"unit": "ms"}],
            [946688461000000000 / 1_000_000_000, {"unit": "s"}],
            [10957, {"unit": "D", "h": 0}],
            [
                (946688461000000000 + 500000) / 1000000000,
                {"unit": "s", "us": 499, "ns": 964},
            ],
            [
                (946688461000000000 + 500000000) / 1000000000,
                {"unit": "s", "us": 500000},
            ],
            [(946688461000000000 + 500000) / 1000000, {"unit": "ms", "us": 500}],
            [(946688461000000000 + 500000) / 1000, {"unit": "us", "us": 500}],
            [(946688461000000000 + 500000000) / 1000000, {"unit": "ms", "us": 500000}],
            [946688461000000000 / 1000.0 + 5, {"unit": "us", "us": 5}],
            [946688461000000000 / 1000.0 + 5000, {"unit": "us", "us": 5000}],
            [946688461000000000 / 1000000.0 + 0.5, {"unit": "ms", "us": 500}],
            [946688461000000000 / 1000000.0 + 0.005, {"unit": "ms", "us": 5, "ns": 5}],
            [946688461000000000 / 1000000000.0 + 0.5, {"unit": "s", "us": 500000}],
            [10957 + 0.5, {"unit": "D", "h": 12}],
        ],
    )
    def test_construct_with_unit(self, value, check_kwargs):
        def check(value, unit=None, h=1, s=1, us=0, ns=0):
            stamp = Timestamp(value, unit=unit)
            assert stamp.year == 2000
            assert stamp.month == 1
            assert stamp.day == 1
            assert stamp.hour == h
            if unit != "D":
                assert stamp.minute == 1
                assert stamp.second == s
                assert stamp.microsecond == us
            else:
                assert stamp.minute == 0
                assert stamp.second == 0
                assert stamp.microsecond == 0
            assert stamp.nanosecond == ns

        check(value, **check_kwargs)


class TestTimestampConstructorFoldKeyword:
    def test_timestamp_constructor_invalid_fold_raise(self):
        # Test for GH#25057
        # Valid fold values are only [None, 0, 1]
        msg = "Valid values for the fold argument are None, 0, or 1."
        with pytest.raises(ValueError, match=msg):
            Timestamp(123, fold=2)

    def test_timestamp_constructor_pytz_fold_raise(self):
        # Test for GH#25057
        # pytz doesn't support fold. Check that we raise
        # if fold is passed with pytz
        msg = "pytz timezones do not support fold. Please use dateutil timezones."
        tz = pytz.timezone("Europe/London")
        with pytest.raises(ValueError, match=msg):
            Timestamp(datetime(2019, 10, 27, 0, 30, 0, 0), tz=tz, fold=0)

    @pytest.mark.parametrize("fold", [0, 1])
    @pytest.mark.parametrize(
        "ts_input",
        [
            1572136200000000000,
            1572136200000000000.0,
            np.datetime64(1572136200000000000, "ns"),
            "2019-10-27 01:30:00+01:00",
            datetime(2019, 10, 27, 0, 30, 0, 0, tzinfo=timezone.utc),
        ],
    )
    def test_timestamp_constructor_fold_conflict(self, ts_input, fold):
        # Test for GH#25057
        # Check that we raise on fold conflict
        msg = (
            "Cannot pass fold with possibly unambiguous input: int, float, "
            "numpy.datetime64, str, or timezone-aware datetime-like. "
            "Pass naive datetime-like or build Timestamp from components."
        )
        with pytest.raises(ValueError, match=msg):
            Timestamp(ts_input=ts_input, fold=fold)

    @pytest.mark.parametrize("tz", ["dateutil/Europe/London", None])
    @pytest.mark.parametrize("fold", [0, 1])
    def test_timestamp_constructor_retain_fold(self, tz, fold):
        # Test for GH#25057
        # Check that we retain fold
        ts = Timestamp(year=2019, month=10, day=27, hour=1, minute=30, tz=tz, fold=fold)
        result = ts.fold
        expected = fold
        assert result == expected

    try:
        _tzs = [
            "dateutil/Europe/London",
            zoneinfo.ZoneInfo("Europe/London"),
        ]
    except zoneinfo.ZoneInfoNotFoundError:
        _tzs = ["dateutil/Europe/London"]

    @pytest.mark.parametrize("tz", _tzs)
    @pytest.mark.parametrize(
        "ts_input,fold_out",
        [
            (1572136200000000000, 0),
            (1572139800000000000, 1),
            ("2019-10-27 01:30:00+01:00", 0),
            ("2019-10-27 01:30:00+00:00", 1),
            (datetime(2019, 10, 27, 1, 30, 0, 0, fold=0), 0),
            (datetime(2019, 10, 27, 1, 30, 0, 0, fold=1), 1),
        ],
    )
    def test_timestamp_constructor_infer_fold_from_value(self, tz, ts_input, fold_out):
        # Test for GH#25057
        # Check that we infer fold correctly based on timestamps since utc
        # or strings
        ts = Timestamp(ts_input, tz=tz)
        result = ts.fold
        expected = fold_out
        assert result == expected

    @pytest.mark.parametrize("tz", ["dateutil/Europe/London"])
    @pytest.mark.parametrize(
        "ts_input,fold,value_out",
        [
            (datetime(2019, 10, 27, 1, 30, 0, 0), 0, 1572136200000000),
            (datetime(2019, 10, 27, 1, 30, 0, 0), 1, 1572139800000000),
        ],
    )
    def test_timestamp_constructor_adjust_value_for_fold(
        self, tz, ts_input, fold, value_out
    ):
        # Test for GH#25057
        # Check that we adjust value for fold correctly
        # based on timestamps since utc
        ts = Timestamp(ts_input, tz=tz, fold=fold)
        result = ts._value
        expected = value_out
        assert result == expected


class TestTimestampConstructorPositionalAndKeywordSupport:
    def test_constructor_positional(self):
        # see GH#10758
        msg = (
            "'NoneType' object cannot be interpreted as an integer"
            if PY310
            else "an integer is required"
        )
        with pytest.raises(TypeError, match=msg):
            Timestamp(2000, 1)

        msg = "month must be in 1..12"
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 0, 1)
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 13, 1)

        msg = "day is out of range for month"
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 1, 0)
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 1, 32)

        # see gh-11630
        assert repr(Timestamp(2015, 11, 12)) == repr(Timestamp("20151112"))
        assert repr(Timestamp(2015, 11, 12, 1, 2, 3, 999999)) == repr(
            Timestamp("2015-11-12 01:02:03.999999")
        )

    def test_constructor_keyword(self):
        # GH#10758
        msg = "function missing required argument 'day'|Required argument 'day'"
        with pytest.raises(TypeError, match=msg):
            Timestamp(year=2000, month=1)

        msg = "month must be in 1..12"
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=0, day=1)
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=13, day=1)

        msg = "day is out of range for month"
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=1, day=0)
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=1, day=32)

        assert repr(Timestamp(year=2015, month=11, day=12)) == repr(
            Timestamp("20151112")
        )

        assert repr(
            Timestamp(
                year=2015,
                month=11,
                day=12,
                hour=1,
                minute=2,
                second=3,
                microsecond=999999,
            )
        ) == repr(Timestamp("2015-11-12 01:02:03.999999"))

    @pytest.mark.parametrize(
        "arg",
        [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "microsecond",
            "nanosecond",
        ],
    )
    def test_invalid_date_kwarg_with_string_input(self, arg):
        kwarg = {arg: 1}
        msg = "Cannot pass a date attribute keyword argument"
        with pytest.raises(ValueError, match=msg):
            Timestamp("2010-10-10 12:59:59.999999999", **kwarg)

    @pytest.mark.parametrize("kwargs", [{}, {"year": 2020}, {"year": 2020, "month": 1}])
    def test_constructor_missing_keyword(self, kwargs):
        # GH#31200

        # The exact error message of datetime() depends on its version
        msg1 = r"function missing required argument '(year|month|day)' \(pos [123]\)"
        msg2 = r"Required argument '(year|month|day)' \(pos [123]\) not found"
        msg = "|".join([msg1, msg2])

        with pytest.raises(TypeError, match=msg):
            Timestamp(**kwargs)

    def test_constructor_positional_with_tzinfo(self):
        # GH#31929
        ts = Timestamp(2020, 12, 31, tzinfo=timezone.utc)
        expected = Timestamp("2020-12-31", tzinfo=timezone.utc)
        assert ts == expected

    @pytest.mark.parametrize("kwd", ["nanosecond", "microsecond", "second", "minute"])
    def test_constructor_positional_keyword_mixed_with_tzinfo(self, kwd, request):
        # TODO: if we passed microsecond with a keyword we would mess up
        #  xref GH#45307
        if kwd != "nanosecond":
            # nanosecond is keyword-only as of 2.0, others are not
            mark = pytest.mark.xfail(reason="GH#45307")
            request.applymarker(mark)

        kwargs = {kwd: 4}
        ts = Timestamp(2020, 12, 31, tzinfo=timezone.utc, **kwargs)

        td_kwargs = {kwd + "s": 4}
        td = Timedelta(**td_kwargs)
        expected = Timestamp("2020-12-31", tz=timezone.utc) + td
        assert ts == expected


class TestTimestampClassMethodConstructors:
    # Timestamp constructors other than __new__

    def test_constructor_strptime(self):
        # GH#25016
        # Test support for Timestamp.strptime
        fmt = "%Y%m%d-%H%M%S-%f%z"
        ts = "20190129-235348-000001+0000"
        msg = r"Timestamp.strptime\(\) is not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            Timestamp.strptime(ts, fmt)

    def test_constructor_fromisocalendar(self):
        # GH#30395
        expected_timestamp = Timestamp("2000-01-03 00:00:00")
        expected_stdlib = datetime.fromisocalendar(2000, 1, 1)
        result = Timestamp.fromisocalendar(2000, 1, 1)
        assert result == expected_timestamp
        assert result == expected_stdlib
        assert isinstance(result, Timestamp)

    def test_constructor_fromordinal(self):
        base = datetime(2000, 1, 1)

        ts = Timestamp.fromordinal(base.toordinal())
        assert base == ts
        assert base.toordinal() == ts.toordinal()

        ts = Timestamp.fromordinal(base.toordinal(), tz="US/Eastern")
        assert Timestamp("2000-01-01", tz="US/Eastern") == ts
        assert base.toordinal() == ts.toordinal()

        # GH#3042
        dt = datetime(2011, 4, 16, 0, 0)
        ts = Timestamp.fromordinal(dt.toordinal())
        assert ts.to_pydatetime() == dt

        # with a tzinfo
        stamp = Timestamp("2011-4-16", tz="US/Eastern")
        dt_tz = stamp.to_pydatetime()
        ts = Timestamp.fromordinal(dt_tz.toordinal(), tz="US/Eastern")
        assert ts.to_pydatetime() == dt_tz

    def test_now(self):
        # GH#9000
        ts_from_string = Timestamp("now")
        ts_from_method = Timestamp.now()
        ts_datetime = datetime.now()

        ts_from_string_tz = Timestamp("now", tz="US/Eastern")
        ts_from_method_tz = Timestamp.now(tz="US/Eastern")

        # Check that the delta between the times is less than 1s (arbitrarily
        # small)
        delta = Timedelta(seconds=1)
        assert abs(ts_from_method - ts_from_string) < delta
        assert abs(ts_datetime - ts_from_method) < delta
        assert abs(ts_from_method_tz - ts_from_string_tz) < delta
        assert (
            abs(
                ts_from_string_tz.tz_localize(None)
                - ts_from_method_tz.tz_localize(None)
            )
            < delta
        )

    def test_today(self):
        ts_from_string = Timestamp("today")
        ts_from_method = Timestamp.today()
        ts_datetime = datetime.today()

        ts_from_string_tz = Timestamp("today", tz="US/Eastern")
        ts_from_method_tz = Timestamp.today(tz="US/Eastern")

        # Check that the delta between the times is less than 1s (arbitrarily
        # small)
        delta = Timedelta(seconds=1)
        assert abs(ts_from_method - ts_from_string) < delta
        assert abs(ts_datetime - ts_from_method) < delta
        assert abs(ts_from_method_tz - ts_from_string_tz) < delta
        assert (
            abs(
                ts_from_string_tz.tz_localize(None)
                - ts_from_method_tz.tz_localize(None)
            )
            < delta
        )


class TestTimestampResolutionInference:
    def test_construct_from_time_unit(self):
        # GH#54097 only passing a time component, no date
        ts = Timestamp("01:01:01.111")
        assert ts.unit == "ms"

    def test_constructor_str_infer_reso(self):
        # non-iso8601 path

        # _parse_delimited_date path
        ts = Timestamp("01/30/2023")
        assert ts.unit == "s"

        # _parse_dateabbr_string path
        ts = Timestamp("2015Q1")
        assert ts.unit == "s"

        # dateutil_parse path
        ts = Timestamp("2016-01-01 1:30:01 PM")
        assert ts.unit == "s"

        ts = Timestamp("2016 June 3 15:25:01.345")
        assert ts.unit == "ms"

        ts = Timestamp("300-01-01")
        assert ts.unit == "s"

        ts = Timestamp("300 June 1:30:01.300")
        assert ts.unit == "ms"

        # dateutil path -> don't drop trailing zeros
        ts = Timestamp("01-01-2013T00:00:00.000000000+0000")
        assert ts.unit == "ns"

        ts = Timestamp("2016/01/02 03:04:05.001000 UTC")
        assert ts.unit == "us"

        # higher-than-nanosecond -> we drop the trailing bits
        ts = Timestamp("01-01-2013T00:00:00.000000002100+0000")
        assert ts == Timestamp("01-01-2013T00:00:00.000000002+0000")
        assert ts.unit == "ns"

        # GH#56208 minute reso through the ISO8601 path with tz offset
        ts = Timestamp("2020-01-01 00:00+00:00")
        assert ts.unit == "s"

        ts = Timestamp("2020-01-01 00+00:00")
        assert ts.unit == "s"

    @pytest.mark.parametrize("method", ["now", "today"])
    def test_now_today_unit(self, method):
        # GH#55879
        ts_from_method = getattr(Timestamp, method)()
        ts_from_string = Timestamp(method)
        assert ts_from_method.unit == ts_from_string.unit == "us"


class TestTimestampConstructors:
    def test_weekday_but_no_day_raises(self):
        # GH#52659
        msg = "Parsing datetimes with weekday but no day information is not supported"
        with pytest.raises(ValueError, match=msg):
            Timestamp("2023 Sept Thu")

    def test_construct_from_string_invalid_raises(self):
        # dateutil (weirdly) parses "200622-12-31" as
        #  datetime(2022, 6, 20, 12, 0, tzinfo=tzoffset(None, -111600)
        #  which besides being mis-parsed, is a tzoffset that will cause
        #  str(ts) to raise ValueError.  Ensure we raise in the constructor
        #  instead.
        # see test_to_datetime_malformed_raise for analogous to_datetime test
        with pytest.raises(ValueError, match="gives an invalid tzoffset"):
            Timestamp("200622-12-31")

    def test_constructor_from_iso8601_str_with_offset_reso(self):
        # GH#49737
        ts = Timestamp("2016-01-01 04:05:06-01:00")
        assert ts.unit == "s"

        ts = Timestamp("2016-01-01 04:05:06.000-01:00")
        assert ts.unit == "ms"

        ts = Timestamp("2016-01-01 04:05:06.000000-01:00")
        assert ts.unit == "us"

        ts = Timestamp("2016-01-01 04:05:06.000000001-01:00")
        assert ts.unit == "ns"

    def test_constructor_from_date_second_reso(self):
        # GH#49034 constructing from a pydate object gets lowest supported
        #  reso, i.e. seconds
        obj = date(2012, 9, 1)
        ts = Timestamp(obj)
        assert ts.unit == "s"

    def test_constructor_datetime64_with_tz(self):
        # GH#42288, GH#24559
        dt = np.datetime64("1970-01-01 05:00:00")
        tzstr = "UTC+05:00"

        # pre-2.0 this interpreted dt as a UTC time. in 2.0 this is treated
        #  as a wall-time, consistent with DatetimeIndex behavior
        ts = Timestamp(dt, tz=tzstr)

        alt = Timestamp(dt).tz_localize(tzstr)
        assert ts == alt
        assert ts.hour == 5

    def test_constructor(self):
        base_str = "2014-07-01 09:00"
        base_dt = datetime(2014, 7, 1, 9)
        base_expected = 1_404_205_200_000_000_000

        # confirm base representation is correct
        assert calendar.timegm(base_dt.timetuple()) * 1_000_000_000 == base_expected

        tests = [
            (base_str, base_dt, base_expected),
            (
                "2014-07-01 10:00",
                datetime(2014, 7, 1, 10),
                base_expected + 3600 * 1_000_000_000,
            ),
            (
                "2014-07-01 09:00:00.000008000",
                datetime(2014, 7, 1, 9, 0, 0, 8),
                base_expected + 8000,
            ),
            (
                "2014-07-01 09:00:00.000000005",
                Timestamp("2014-07-01 09:00:00.000000005"),
                base_expected + 5,
            ),
        ]

        timezones = [
            (None, 0),
            ("UTC", 0),
            (pytz.utc, 0),
            ("Asia/Tokyo", 9),
            ("US/Eastern", -4),
            ("dateutil/US/Pacific", -7),
            (pytz.FixedOffset(-180), -3),
            (dateutil.tz.tzoffset(None, 18000), 5),
        ]

        for date_str, date_obj, expected in tests:
            for result in [Timestamp(date_str), Timestamp(date_obj)]:
                result = result.as_unit("ns")  # test originally written before non-nano
                # only with timestring
                assert result.as_unit("ns")._value == expected

                # re-creation shouldn't affect to internal value
                result = Timestamp(result)
                assert result.as_unit("ns")._value == expected

            # with timezone
            for tz, offset in timezones:
                for result in [Timestamp(date_str, tz=tz), Timestamp(date_obj, tz=tz)]:
                    result = result.as_unit(
                        "ns"
                    )  # test originally written before non-nano
                    expected_tz = expected - offset * 3600 * 1_000_000_000
                    assert result.as_unit("ns")._value == expected_tz

                    # should preserve tz
                    result = Timestamp(result)
                    assert result.as_unit("ns")._value == expected_tz

                    # should convert to UTC
                    if tz is not None:
                        result = Timestamp(result).tz_convert("UTC")
                    else:
                        result = Timestamp(result, tz="UTC")
                    expected_utc = expected - offset * 3600 * 1_000_000_000
                    assert result.as_unit("ns")._value == expected_utc

    def test_constructor_with_stringoffset(self):
        # GH 7833
        base_str = "2014-07-01 11:00:00+02:00"
        base_dt = datetime(2014, 7, 1, 9)
        base_expected = 1_404_205_200_000_000_000

        # confirm base representation is correct
        assert calendar.timegm(base_dt.timetuple()) * 1_000_000_000 == base_expected

        tests = [
            (base_str, base_expected),
            ("2014-07-01 12:00:00+02:00", base_expected + 3600 * 1_000_000_000),
            ("2014-07-01 11:00:00.000008000+02:00", base_expected + 8000),
            ("2014-07-01 11:00:00.000000005+02:00", base_expected + 5),
        ]

        timezones = [
            (None, 0),
            ("UTC", 0),
            (pytz.utc, 0),
            ("Asia/Tokyo", 9),
            ("US/Eastern", -4),
            ("dateutil/US/Pacific", -7),
            (pytz.FixedOffset(-180), -3),
            (dateutil.tz.tzoffset(None, 18000), 5),
        ]

        for date_str, expected in tests:
            for result in [Timestamp(date_str)]:
                # only with timestring
                assert result.as_unit("ns")._value == expected

                # re-creation shouldn't affect to internal value
                result = Timestamp(result)
                assert result.as_unit("ns")._value == expected

            # with timezone
            for tz, offset in timezones:
                result = Timestamp(date_str, tz=tz)
                expected_tz = expected
                assert result.as_unit("ns")._value == expected_tz

                # should preserve tz
                result = Timestamp(result)
                assert result.as_unit("ns")._value == expected_tz

                # should convert to UTC
                result = Timestamp(result).tz_convert("UTC")
                expected_utc = expected
                assert result.as_unit("ns")._value == expected_utc

        # This should be 2013-11-01 05:00 in UTC
        # converted to Chicago tz
        result = Timestamp("2013-11-01 00:00:00-0500", tz="America/Chicago")
        assert result._value == Timestamp("2013-11-01 05:00")._value
        expected = "Timestamp('2013-11-01 00:00:00-0500', tz='America/Chicago')"
        assert repr(result) == expected
        assert result == eval(repr(result))

        # This should be 2013-11-01 05:00 in UTC
        # converted to Tokyo tz (+09:00)
        result = Timestamp("2013-11-01 00:00:00-0500", tz="Asia/Tokyo")
        assert result._value == Timestamp("2013-11-01 05:00")._value
        expected = "Timestamp('2013-11-01 14:00:00+0900', tz='Asia/Tokyo')"
        assert repr(result) == expected
        assert result == eval(repr(result))

        # GH11708
        # This should be 2015-11-18 10:00 in UTC
        # converted to Asia/Katmandu
        result = Timestamp("2015-11-18 15:45:00+05:45", tz="Asia/Katmandu")
        assert result._value == Timestamp("2015-11-18 10:00")._value
        expected = "Timestamp('2015-11-18 15:45:00+0545', tz='Asia/Katmandu')"
        assert repr(result) == expected
        assert result == eval(repr(result))

        # This should be 2015-11-18 10:00 in UTC
        # converted to Asia/Kolkata
        result = Timestamp("2015-11-18 15:30:00+05:30", tz="Asia/Kolkata")
        assert result._value == Timestamp("2015-11-18 10:00")._value
        expected = "Timestamp('2015-11-18 15:30:00+0530', tz='Asia/Kolkata')"
        assert repr(result) == expected
        assert result == eval(repr(result))

    def test_constructor_invalid(self):
        msg = "Cannot convert input"
        with pytest.raises(TypeError, match=msg):
            Timestamp(slice(2))
        msg = "Cannot convert Period"
        with pytest.raises(ValueError, match=msg):
            Timestamp(Period("1000-01-01"))

    def test_constructor_invalid_tz(self):
        # GH#17690
        msg = (
            "Argument 'tzinfo' has incorrect type "
            r"\(expected datetime.tzinfo, got str\)"
        )
        with pytest.raises(TypeError, match=msg):
            Timestamp("2017-10-22", tzinfo="US/Eastern")

        msg = "at most one of"
        with pytest.raises(ValueError, match=msg):
            Timestamp("2017-10-22", tzinfo=pytz.utc, tz="UTC")

        msg = "Cannot pass a date attribute keyword argument when passing a date string"
        with pytest.raises(ValueError, match=msg):
            # GH#5168
            # case where user tries to pass tz as an arg, not kwarg, gets
            # interpreted as `year`
            Timestamp("2012-01-01", "US/Pacific")

    def test_constructor_tz_or_tzinfo(self):
        # GH#17943, GH#17690, GH#5168
        stamps = [
            Timestamp(year=2017, month=10, day=22, tz="UTC"),
            Timestamp(year=2017, month=10, day=22, tzinfo=pytz.utc),
            Timestamp(year=2017, month=10, day=22, tz=pytz.utc),
            Timestamp(datetime(2017, 10, 22), tzinfo=pytz.utc),
            Timestamp(datetime(2017, 10, 22), tz="UTC"),
            Timestamp(datetime(2017, 10, 22), tz=pytz.utc),
        ]
        assert all(ts == stamps[0] for ts in stamps)

    @pytest.mark.parametrize(
        "result",
        [
            Timestamp(datetime(2000, 1, 2, 3, 4, 5, 6), nanosecond=1),
            Timestamp(
                year=2000,
                month=1,
                day=2,
                hour=3,
                minute=4,
                second=5,
                microsecond=6,
                nanosecond=1,
            ),
            Timestamp(
                year=2000,
                month=1,
                day=2,
                hour=3,
                minute=4,
                second=5,
                microsecond=6,
                nanosecond=1,
                tz="UTC",
            ),
            Timestamp(2000, 1, 2, 3, 4, 5, 6, None, nanosecond=1),
            Timestamp(2000, 1, 2, 3, 4, 5, 6, tz=pytz.UTC, nanosecond=1),
        ],
    )
    def test_constructor_nanosecond(self, result):
        # GH 18898
        # As of 2.0 (GH 49416), nanosecond should not be accepted positionally
        expected = Timestamp(datetime(2000, 1, 2, 3, 4, 5, 6), tz=result.tz)
        expected = expected + Timedelta(nanoseconds=1)
        assert result == expected

    @pytest.mark.parametrize("z", ["Z0", "Z00"])
    def test_constructor_invalid_Z0_isostring(self, z):
        # GH 8910
        msg = f"Unknown datetime string format, unable to parse: 2014-11-02 01:00{z}"
        with pytest.raises(ValueError, match=msg):
            Timestamp(f"2014-11-02 01:00{z}")

    def test_out_of_bounds_integer_value(self):
        # GH#26651 check that we raise OutOfBoundsDatetime, not OverflowError
        msg = str(Timestamp.max._value * 2)
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(Timestamp.max._value * 2)
        msg = str(Timestamp.min._value * 2)
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(Timestamp.min._value * 2)

    def test_out_of_bounds_value(self):
        one_us = np.timedelta64(1).astype("timedelta64[us]")

        # By definition we can't go out of bounds in [ns], so we
        # convert the datetime64s to [us] so we can go out of bounds
        min_ts_us = np.datetime64(Timestamp.min).astype("M8[us]") + one_us
        max_ts_us = np.datetime64(Timestamp.max).astype("M8[us]")

        # No error for the min/max datetimes
        Timestamp(min_ts_us)
        Timestamp(max_ts_us)

        # We used to raise on these before supporting non-nano
        us_val = NpyDatetimeUnit.NPY_FR_us.value
        assert Timestamp(min_ts_us - one_us)._creso == us_val
        assert Timestamp(max_ts_us + one_us)._creso == us_val

        # https://github.com/numpy/numpy/issues/22346 for why
        #  we can't use the same construction as above with minute resolution

        # too_low, too_high are the _just_ outside the range of M8[s]
        too_low = np.datetime64("-292277022657-01-27T08:29", "m")
        too_high = np.datetime64("292277026596-12-04T15:31", "m")

        msg = "Out of bounds"
        # One us less than the minimum is an error
        with pytest.raises(ValueError, match=msg):
            Timestamp(too_low)

        # One us more than the maximum is an error
        with pytest.raises(ValueError, match=msg):
            Timestamp(too_high)

    def test_out_of_bounds_string(self):
        msg = "Cannot cast .* to unit='ns' without overflow"
        with pytest.raises(ValueError, match=msg):
            Timestamp("1676-01-01").as_unit("ns")
        with pytest.raises(ValueError, match=msg):
            Timestamp("2263-01-01").as_unit("ns")

        ts = Timestamp("2263-01-01")
        assert ts.unit == "s"

        ts = Timestamp("1676-01-01")
        assert ts.unit == "s"

    def test_barely_out_of_bounds(self):
        # GH#19529
        # GH#19382 close enough to bounds that dropping nanos would result
        # in an in-bounds datetime
        msg = "Out of bounds nanosecond timestamp: 2262-04-11 23:47:16"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp("2262-04-11 23:47:16.854775808")

    @pytest.mark.skip_ubsan
    def test_bounds_with_different_units(self):
        out_of_bounds_dates = ("1677-09-21", "2262-04-12")

        time_units = ("D", "h", "m", "s", "ms", "us")

        for date_string in out_of_bounds_dates:
            for unit in time_units:
                dt64 = np.datetime64(date_string, unit)
                ts = Timestamp(dt64)
                if unit in ["s", "ms", "us"]:
                    # We can preserve the input unit
                    assert ts._value == dt64.view("i8")
                else:
                    # we chose the closest unit that we _do_ support
                    assert ts._creso == NpyDatetimeUnit.NPY_FR_s.value

        # With more extreme cases, we can't even fit inside second resolution
        info = np.iinfo(np.int64)
        msg = "Out of bounds second timestamp:"
        for value in [info.min + 1, info.max]:
            for unit in ["D", "h", "m"]:
                dt64 = np.datetime64(value, unit)
                with pytest.raises(OutOfBoundsDatetime, match=msg):
                    Timestamp(dt64)

        in_bounds_dates = ("1677-09-23", "2262-04-11")

        for date_string in in_bounds_dates:
            for unit in time_units:
                dt64 = np.datetime64(date_string, unit)
                Timestamp(dt64)

    @pytest.mark.parametrize("arg", ["001-01-01", "0001-01-01"])
    def test_out_of_bounds_string_consistency(self, arg):
        # GH 15829
        msg = "Cannot cast 0001-01-01 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(arg).as_unit("ns")

        ts = Timestamp(arg)
        assert ts.unit == "s"
        assert ts.year == ts.month == ts.day == 1

    def test_min_valid(self):
        # Ensure that Timestamp.min is a valid Timestamp
        Timestamp(Timestamp.min)

    def test_max_valid(self):
        # Ensure that Timestamp.max is a valid Timestamp
        Timestamp(Timestamp.max)

    @pytest.mark.parametrize("offset", ["+0300", "+0200"])
    def test_construct_timestamp_near_dst(self, offset):
        # GH 20854
        expected = Timestamp(f"2016-10-30 03:00:00{offset}", tz="Europe/Helsinki")
        result = Timestamp(expected).tz_convert("Europe/Helsinki")
        assert result == expected

    @pytest.mark.parametrize(
        "arg", ["2013/01/01 00:00:00+09:00", "2013-01-01 00:00:00+09:00"]
    )
    def test_construct_with_different_string_format(self, arg):
        # GH 12064
        result = Timestamp(arg)
        expected = Timestamp(datetime(2013, 1, 1), tz=pytz.FixedOffset(540))
        assert result == expected

    @pytest.mark.parametrize("box", [datetime, Timestamp])
    def test_raise_tz_and_tzinfo_in_datetime_input(self, box):
        # GH 23579
        kwargs = {"year": 2018, "month": 1, "day": 1, "tzinfo": pytz.utc}
        msg = "Cannot pass a datetime or Timestamp"
        with pytest.raises(ValueError, match=msg):
            Timestamp(box(**kwargs), tz="US/Pacific")
        msg = "Cannot pass a datetime or Timestamp"
        with pytest.raises(ValueError, match=msg):
            Timestamp(box(**kwargs), tzinfo=pytz.timezone("US/Pacific"))

    def test_dont_convert_dateutil_utc_to_pytz_utc(self):
        result = Timestamp(datetime(2018, 1, 1), tz=tzutc())
        expected = Timestamp(datetime(2018, 1, 1)).tz_localize(tzutc())
        assert result == expected

    def test_constructor_subclassed_datetime(self):
        # GH 25851
        # ensure that subclassed datetime works for
        # Timestamp creation
        class SubDatetime(datetime):
            pass

        data = SubDatetime(2000, 1, 1)
        result = Timestamp(data)
        expected = Timestamp(2000, 1, 1)
        assert result == expected

    def test_timestamp_constructor_tz_utc(self):
        utc_stamp = Timestamp("3/11/2012 05:00", tz="utc")
        assert utc_stamp.tzinfo is timezone.utc
        assert utc_stamp.hour == 5

        utc_stamp = Timestamp("3/11/2012 05:00").tz_localize("utc")
        assert utc_stamp.hour == 5

    def test_timestamp_to_datetime_tzoffset(self):
        tzinfo = tzoffset(None, 7200)
        expected = Timestamp("3/11/2012 04:00", tz=tzinfo)
        result = Timestamp(expected.to_pydatetime())
        assert expected == result

    def test_timestamp_constructor_near_dst_boundary(self):
        # GH#11481 & GH#15777
        # Naive string timestamps were being localized incorrectly
        # with tz_convert_from_utc_single instead of tz_localize_to_utc

        for tz in ["Europe/Brussels", "Europe/Prague"]:
            result = Timestamp("2015-10-25 01:00", tz=tz)
            expected = Timestamp("2015-10-25 01:00").tz_localize(tz)
            assert result == expected

            msg = "Cannot infer dst time from 2015-10-25 02:00:00"
            with pytest.raises(pytz.AmbiguousTimeError, match=msg):
                Timestamp("2015-10-25 02:00", tz=tz)

        result = Timestamp("2017-03-26 01:00", tz="Europe/Paris")
        expected = Timestamp("2017-03-26 01:00").tz_localize("Europe/Paris")
        assert result == expected

        msg = "2017-03-26 02:00"
        with pytest.raises(pytz.NonExistentTimeError, match=msg):
            Timestamp("2017-03-26 02:00", tz="Europe/Paris")

        # GH#11708
        naive = Timestamp("2015-11-18 10:00:00")
        result = naive.tz_localize("UTC").tz_convert("Asia/Kolkata")
        expected = Timestamp("2015-11-18 15:30:00+0530", tz="Asia/Kolkata")
        assert result == expected

        # GH#15823
        result = Timestamp("2017-03-26 00:00", tz="Europe/Paris")
        expected = Timestamp("2017-03-26 00:00:00+0100", tz="Europe/Paris")
        assert result == expected

        result = Timestamp("2017-03-26 01:00", tz="Europe/Paris")
        expected = Timestamp("2017-03-26 01:00:00+0100", tz="Europe/Paris")
        assert result == expected

        msg = "2017-03-26 02:00"
        with pytest.raises(pytz.NonExistentTimeError, match=msg):
            Timestamp("2017-03-26 02:00", tz="Europe/Paris")

        result = Timestamp("2017-03-26 02:00:00+0100", tz="Europe/Paris")
        naive = Timestamp(result.as_unit("ns")._value)
        expected = naive.tz_localize("UTC").tz_convert("Europe/Paris")
        assert result == expected

        result = Timestamp("2017-03-26 03:00", tz="Europe/Paris")
        expected = Timestamp("2017-03-26 03:00:00+0200", tz="Europe/Paris")
        assert result == expected

    @pytest.mark.parametrize(
        "tz",
        [
            pytz.timezone("US/Eastern"),
            gettz("US/Eastern"),
            "US/Eastern",
            "dateutil/US/Eastern",
        ],
    )
    def test_timestamp_constructed_by_date_and_tz(self, tz):
        # GH#2993, Timestamp cannot be constructed by datetime.date
        # and tz correctly

        result = Timestamp(date(2012, 3, 11), tz=tz)

        expected = Timestamp("3/11/2012", tz=tz)
        assert result.hour == expected.hour
        assert result == expected


def test_constructor_ambiguous_dst():
    # GH 24329
    # Make sure that calling Timestamp constructor
    # on Timestamp created from ambiguous time
    # doesn't change Timestamp.value
    ts = Timestamp(1382835600000000000, tz="dateutil/Europe/London")
    expected = ts._value
    result = Timestamp(ts)._value
    assert result == expected


@pytest.mark.parametrize("epoch", [1552211999999999872, 1552211999999999999])
def test_constructor_before_dst_switch(epoch):
    # GH 31043
    # Make sure that calling Timestamp constructor
    # on time just before DST switch doesn't lead to
    # nonexistent time or value change
    ts = Timestamp(epoch, tz="dateutil/America/Los_Angeles")
    result = ts.tz.dst(ts)
    expected = timedelta(seconds=0)
    assert Timestamp(ts)._value == epoch
    assert result == expected


def test_timestamp_constructor_identity():
    # Test for #30543
    expected = Timestamp("2017-01-01T12")
    result = Timestamp(expected)
    assert result is expected


@pytest.mark.parametrize("nano", [-1, 1000])
def test_timestamp_nano_range(nano):
    # GH 48255
    with pytest.raises(ValueError, match="nanosecond must be in 0..999"):
        Timestamp(year=2022, month=1, day=1, nanosecond=nano)


def test_non_nano_value():
    # https://github.com/pandas-dev/pandas/issues/49076
    result = Timestamp("1800-01-01", unit="s").value
    # `.value` shows nanoseconds, even though unit is 's'
    assert result == -5364662400000000000

    # out-of-nanoseconds-bounds `.value` raises informative message
    msg = (
        r"Cannot convert Timestamp to nanoseconds without overflow. "
        r"Use `.asm8.view\('i8'\)` to cast represent Timestamp in its "
        r"own unit \(here, s\).$"
    )
    ts = Timestamp("0300-01-01")
    with pytest.raises(OverflowError, match=msg):
        ts.value
    # check that the suggested workaround actually works
    result = ts.asm8.view("i8")
    assert result == -52700112000


@pytest.mark.parametrize("na_value", [None, np.nan, np.datetime64("NaT"), NaT, NA])
def test_timestamp_constructor_na_value(na_value):
    # GH45481
    result = Timestamp(na_value)
    expected = NaT
    assert result is expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\tests\test_hyperexpand.py ===
from sympy.core.random import randrange

from sympy.simplify.hyperexpand import (ShiftA, ShiftB, UnShiftA, UnShiftB,
                       MeijerShiftA, MeijerShiftB, MeijerShiftC, MeijerShiftD,
                       MeijerUnShiftA, MeijerUnShiftB, MeijerUnShiftC,
                       MeijerUnShiftD,
                       ReduceOrder, reduce_order, apply_operators,
                       devise_plan, make_derivative_operator, Formula,
                       hyperexpand, Hyper_Function, G_Function,
                       reduce_order_meijer,
                       build_hypergeometric_formula)
from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.abc import z, a, b, c
from sympy.testing.pytest import XFAIL, raises, slow, tooslow
from sympy.core.random import verify_numerically as tn

from sympy.core.numbers import (Rational, pi)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import atanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.functions.special.bessel import besseli
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import (gamma, lowergamma)


def test_branch_bug():
    assert hyperexpand(hyper((Rational(-1, 3), S.Half), (Rational(2, 3), Rational(3, 2)), -z)) == \
        -z**S('1/3')*lowergamma(exp_polar(I*pi)/3, z)/5 \
        + sqrt(pi)*erf(sqrt(z))/(5*sqrt(z))
    assert hyperexpand(meijerg([Rational(7, 6), 1], [], [Rational(2, 3)], [Rational(1, 6), 0], z)) == \
        2*z**S('2/3')*(2*sqrt(pi)*erf(sqrt(z))/sqrt(z) - 2*lowergamma(
                       Rational(2, 3), z)/z**S('2/3'))*gamma(Rational(2, 3))/gamma(Rational(5, 3))


def test_hyperexpand():
    # Luke, Y. L. (1969), The Special Functions and Their Approximations,
    # Volume 1, section 6.2

    assert hyperexpand(hyper([], [], z)) == exp(z)
    assert hyperexpand(hyper([1, 1], [2], -z)*z) == log(1 + z)
    assert hyperexpand(hyper([], [S.Half], -z**2/4)) == cos(z)
    assert hyperexpand(z*hyper([], [S('3/2')], -z**2/4)) == sin(z)
    assert hyperexpand(hyper([S('1/2'), S('1/2')], [S('3/2')], z**2)*z) \
        == asin(z)
    assert isinstance(Sum(binomial(2, z)*z**2, (z, 0, a)).doit(), Expr)


def can_do(ap, bq, numerical=True, div=1, lowerplane=False):
    r = hyperexpand(hyper(ap, bq, z))
    if r.has(hyper):
        return False
    if not numerical:
        return True
    repl = {}
    randsyms = r.free_symbols - {z}
    while randsyms:
        # Only randomly generated parameters are checked.
        for n, ai in enumerate(randsyms):
            repl[ai] = randcplx(n)/div
        if not any(b.is_Integer and b <= 0 for b in Tuple(*bq).subs(repl)):
            break
    [a, b, c, d] = [2, -1, 3, 1]
    if lowerplane:
        [a, b, c, d] = [2, -2, 3, -1]
    return tn(
        hyper(ap, bq, z).subs(repl),
        r.replace(exp_polar, exp).subs(repl),
        z, a=a, b=b, c=c, d=d)


def test_roach():
    # Kelly B. Roach.  Meijer G Function Representations.
    # Section "Gallery"
    assert can_do([S.Half], [Rational(9, 2)])
    assert can_do([], [1, Rational(5, 2), 4])
    assert can_do([Rational(-1, 2), 1, 2], [3, 4])
    assert can_do([Rational(1, 3)], [Rational(-2, 3), Rational(-1, 2), S.Half, 1])
    assert can_do([Rational(-3, 2), Rational(-1, 2)], [Rational(-5, 2), 1])
    assert can_do([Rational(-3, 2), ], [Rational(-1, 2), S.Half])  # shine-integral
    assert can_do([Rational(-3, 2), Rational(-1, 2)], [2])  # elliptic integrals


@XFAIL
def test_roach_fail():
    assert can_do([Rational(-1, 2), 1], [Rational(1, 4), S.Half, Rational(3, 4)])  # PFDD
    assert can_do([Rational(3, 2)], [Rational(5, 2), 5])  # struve function
    assert can_do([Rational(-1, 2), S.Half, 1], [Rational(3, 2), Rational(5, 2)])  # polylog, pfdd
    assert can_do([1, 2, 3], [S.Half, 4])  # XXX ?
    assert can_do([S.Half], [Rational(-1, 3), Rational(-1, 2), Rational(-2, 3)])  # PFDD ?

# For the long table tests, see end of file


def test_polynomial():
    from sympy.core.numbers import oo
    assert hyperexpand(hyper([], [-1], z)) is oo
    assert hyperexpand(hyper([-2], [-1], z)) is oo
    assert hyperexpand(hyper([0, 0], [-1], z)) == 1
    assert can_do([-5, -2, randcplx(), randcplx()], [-10, randcplx()])
    assert hyperexpand(hyper((-1, 1), (-2,), z)) == 1 + z/2


def test_hyperexpand_bases():
    assert hyperexpand(hyper([2], [a], z)) == \
        a + z**(-a + 1)*(-a**2 + 3*a + z*(a - 1) - 2)*exp(z)* \
        lowergamma(a - 1, z) - 1
    # TODO [a+1, aRational(-1, 2)], [2*a]
    assert hyperexpand(hyper([1, 2], [3], z)) == -2/z - 2*log(-z + 1)/z**2
    assert hyperexpand(hyper([S.Half, 2], [Rational(3, 2)], z)) == \
        -1/(2*z - 2) + atanh(sqrt(z))/sqrt(z)/2
    assert hyperexpand(hyper([S.Half, S.Half], [Rational(5, 2)], z)) == \
        (-3*z + 3)/4/(z*sqrt(-z + 1)) \
        + (6*z - 3)*asin(sqrt(z))/(4*z**Rational(3, 2))
    assert hyperexpand(hyper([1, 2], [Rational(3, 2)], z)) == -1/(2*z - 2) \
        - asin(sqrt(z))/(sqrt(z)*(2*z - 2)*sqrt(-z + 1))
    assert hyperexpand(hyper([Rational(-1, 2) - 1, 1, 2], [S.Half, 3], z)) == \
        sqrt(z)*(z*Rational(6, 7) - Rational(6, 5))*atanh(sqrt(z)) \
        + (-30*z**2 + 32*z - 6)/35/z - 6*log(-z + 1)/(35*z**2)
    assert hyperexpand(hyper([1 + S.Half, 1, 1], [2, 2], z)) == \
        -4*log(sqrt(-z + 1)/2 + S.Half)/z
    # TODO hyperexpand(hyper([a], [2*a + 1], z))
    # TODO [S.Half, a], [Rational(3, 2), a+1]
    assert hyperexpand(hyper([2], [b, 1], z)) == \
        z**(-b/2 + S.Half)*besseli(b - 1, 2*sqrt(z))*gamma(b) \
        + z**(-b/2 + 1)*besseli(b, 2*sqrt(z))*gamma(b)
    # TODO [a], [a - S.Half, 2*a]


def test_hyperexpand_parametric():
    assert hyperexpand(hyper([a, S.Half + a], [S.Half], z)) \
        == (1 + sqrt(z))**(-2*a)/2 + (1 - sqrt(z))**(-2*a)/2
    assert hyperexpand(hyper([a, Rational(-1, 2) + a], [2*a], z)) \
        == 2**(2*a - 1)*((-z + 1)**S.Half + 1)**(-2*a + 1)


def test_shifted_sum():
    from sympy.simplify.simplify import simplify
    assert simplify(hyperexpand(z**4*hyper([2], [3, S('3/2')], -z**2))) \
        == z*sin(2*z) + (-z**2 + S.Half)*cos(2*z) - S.Half


def _randrat():
    """ Steer clear of integers. """
    return S(randrange(25) + 10)/50


def randcplx(offset=-1):
    """ Polys is not good with real coefficients. """
    return _randrat() + I*_randrat() + I*(1 + offset)


@slow
def test_formulae():
    from sympy.simplify.hyperexpand import FormulaCollection
    formulae = FormulaCollection().formulae
    for formula in formulae:
        h = formula.func(formula.z)
        rep = {}
        for n, sym in enumerate(formula.symbols):
            rep[sym] = randcplx(n)

        # NOTE hyperexpand returns truly branched functions. We know we are
        #      on the main sheet, but numerical evaluation can still go wrong
        #      (e.g. if exp_polar cannot be evalf'd).
        #      Just replace all exp_polar by exp, this usually works.

        # first test if the closed-form is actually correct
        h = h.subs(rep)
        closed_form = formula.closed_form.subs(rep).rewrite('nonrepsmall')
        z = formula.z
        assert tn(h, closed_form.replace(exp_polar, exp), z)

        # now test the computed matrix
        cl = (formula.C * formula.B)[0].subs(rep).rewrite('nonrepsmall')
        assert tn(closed_form.replace(
            exp_polar, exp), cl.replace(exp_polar, exp), z)
        deriv1 = z*formula.B.applyfunc(lambda t: t.rewrite(
            'nonrepsmall')).diff(z)
        deriv2 = formula.M * formula.B
        for d1, d2 in zip(deriv1, deriv2):
            assert tn(d1.subs(rep).replace(exp_polar, exp),
                      d2.subs(rep).rewrite('nonrepsmall').replace(exp_polar, exp), z)


def test_meijerg_formulae():
    from sympy.simplify.hyperexpand import MeijerFormulaCollection
    formulae = MeijerFormulaCollection().formulae
    for sig in formulae:
        for formula in formulae[sig]:
            g = meijerg(formula.func.an, formula.func.ap,
                        formula.func.bm, formula.func.bq,
                        formula.z)
            rep = {}
            for sym in formula.symbols:
                rep[sym] = randcplx()

            # first test if the closed-form is actually correct
            g = g.subs(rep)
            closed_form = formula.closed_form.subs(rep)
            z = formula.z
            assert tn(g, closed_form, z)

            # now test the computed matrix
            cl = (formula.C * formula.B)[0].subs(rep)
            assert tn(closed_form, cl, z)
            deriv1 = z*formula.B.diff(z)
            deriv2 = formula.M * formula.B
            for d1, d2 in zip(deriv1, deriv2):
                assert tn(d1.subs(rep), d2.subs(rep), z)


def op(f):
    return z*f.diff(z)


def test_plan():
    assert devise_plan(Hyper_Function([0], ()),
            Hyper_Function([0], ()), z) == []
    with raises(ValueError):
        devise_plan(Hyper_Function([1], ()), Hyper_Function((), ()), z)
    with raises(ValueError):
        devise_plan(Hyper_Function([2], [1]), Hyper_Function([2], [2]), z)
    with raises(ValueError):
        devise_plan(Hyper_Function([2], []), Hyper_Function([S("1/2")], []), z)

    # We cannot use pi/(10000 + n) because polys is insanely slow.
    a1, a2, b1 = (randcplx(n) for n in range(3))
    b1 += 2*I
    h = hyper([a1, a2], [b1], z)

    h2 = hyper((a1 + 1, a2), [b1], z)
    assert tn(apply_operators(h,
        devise_plan(Hyper_Function((a1 + 1, a2), [b1]),
            Hyper_Function((a1, a2), [b1]), z), op),
        h2, z)

    h2 = hyper((a1 + 1, a2 - 1), [b1], z)
    assert tn(apply_operators(h,
        devise_plan(Hyper_Function((a1 + 1, a2 - 1), [b1]),
            Hyper_Function((a1, a2), [b1]), z), op),
        h2, z)


def test_plan_derivatives():
    a1, a2, a3 = 1, 2, S('1/2')
    b1, b2 = 3, S('5/2')
    h = Hyper_Function((a1, a2, a3), (b1, b2))
    h2 = Hyper_Function((a1 + 1, a2 + 1, a3 + 2), (b1 + 1, b2 + 1))
    ops = devise_plan(h2, h, z)
    f = Formula(h, z, h(z), [])
    deriv = make_derivative_operator(f.M, z)
    assert tn((apply_operators(f.C, ops, deriv)*f.B)[0], h2(z), z)

    h2 = Hyper_Function((a1, a2 - 1, a3 - 2), (b1 - 1, b2 - 1))
    ops = devise_plan(h2, h, z)
    assert tn((apply_operators(f.C, ops, deriv)*f.B)[0], h2(z), z)


def test_reduction_operators():
    a1, a2, b1 = (randcplx(n) for n in range(3))
    h = hyper([a1], [b1], z)

    assert ReduceOrder(2, 0) is None
    assert ReduceOrder(2, -1) is None
    assert ReduceOrder(1, S('1/2')) is None

    h2 = hyper((a1, a2), (b1, a2), z)
    assert tn(ReduceOrder(a2, a2).apply(h, op), h2, z)

    h2 = hyper((a1, a2 + 1), (b1, a2), z)
    assert tn(ReduceOrder(a2 + 1, a2).apply(h, op), h2, z)

    h2 = hyper((a2 + 4, a1), (b1, a2), z)
    assert tn(ReduceOrder(a2 + 4, a2).apply(h, op), h2, z)

    # test several step order reduction
    ap = (a2 + 4, a1, b1 + 1)
    bq = (a2, b1, b1)
    func, ops = reduce_order(Hyper_Function(ap, bq))
    assert func.ap == (a1,)
    assert func.bq == (b1,)
    assert tn(apply_operators(h, ops, op), hyper(ap, bq, z), z)


def test_shift_operators():
    a1, a2, b1, b2, b3 = (randcplx(n) for n in range(5))
    h = hyper((a1, a2), (b1, b2, b3), z)

    raises(ValueError, lambda: ShiftA(0))
    raises(ValueError, lambda: ShiftB(1))

    assert tn(ShiftA(a1).apply(h, op), hyper((a1 + 1, a2), (b1, b2, b3), z), z)
    assert tn(ShiftA(a2).apply(h, op), hyper((a1, a2 + 1), (b1, b2, b3), z), z)
    assert tn(ShiftB(b1).apply(h, op), hyper((a1, a2), (b1 - 1, b2, b3), z), z)
    assert tn(ShiftB(b2).apply(h, op), hyper((a1, a2), (b1, b2 - 1, b3), z), z)
    assert tn(ShiftB(b3).apply(h, op), hyper((a1, a2), (b1, b2, b3 - 1), z), z)


def test_ushift_operators():
    a1, a2, b1, b2, b3 = (randcplx(n) for n in range(5))
    h = hyper((a1, a2), (b1, b2, b3), z)

    raises(ValueError, lambda: UnShiftA((1,), (), 0, z))
    raises(ValueError, lambda: UnShiftB((), (-1,), 0, z))
    raises(ValueError, lambda: UnShiftA((1,), (0, -1, 1), 0, z))
    raises(ValueError, lambda: UnShiftB((0, 1), (1,), 0, z))

    s = UnShiftA((a1, a2), (b1, b2, b3), 0, z)
    assert tn(s.apply(h, op), hyper((a1 - 1, a2), (b1, b2, b3), z), z)
    s = UnShiftA((a1, a2), (b1, b2, b3), 1, z)
    assert tn(s.apply(h, op), hyper((a1, a2 - 1), (b1, b2, b3), z), z)

    s = UnShiftB((a1, a2), (b1, b2, b3), 0, z)
    assert tn(s.apply(h, op), hyper((a1, a2), (b1 + 1, b2, b3), z), z)
    s = UnShiftB((a1, a2), (b1, b2, b3), 1, z)
    assert tn(s.apply(h, op), hyper((a1, a2), (b1, b2 + 1, b3), z), z)
    s = UnShiftB((a1, a2), (b1, b2, b3), 2, z)
    assert tn(s.apply(h, op), hyper((a1, a2), (b1, b2, b3 + 1), z), z)


def can_do_meijer(a1, a2, b1, b2, numeric=True):
    """
    This helper function tries to hyperexpand() the meijer g-function
    corresponding to the parameters a1, a2, b1, b2.
    It returns False if this expansion still contains g-functions.
    If numeric is True, it also tests the so-obtained formula numerically
    (at random values) and returns False if the test fails.
    Else it returns True.
    """
    from sympy.core.function import expand
    from sympy.functions.elementary.complexes import unpolarify
    r = hyperexpand(meijerg(a1, a2, b1, b2, z))
    if r.has(meijerg):
        return False
    # NOTE hyperexpand() returns a truly branched function, whereas numerical
    #      evaluation only works on the main branch. Since we are evaluating on
    #      the main branch, this should not be a problem, but expressions like
    #      exp_polar(I*pi/2*x)**a are evaluated incorrectly. We thus have to get
    #      rid of them. The expand heuristically does this...
    r = unpolarify(expand(r, force=True, power_base=True, power_exp=False,
                          mul=False, log=False, multinomial=False, basic=False))

    if not numeric:
        return True

    repl = {}
    for n, ai in enumerate(meijerg(a1, a2, b1, b2, z).free_symbols - {z}):
        repl[ai] = randcplx(n)
    return tn(meijerg(a1, a2, b1, b2, z).subs(repl), r.subs(repl), z)


@slow
def test_meijerg_expand():
    from sympy.simplify.gammasimp import gammasimp
    from sympy.simplify.simplify import simplify
    # from mpmath docs
    assert hyperexpand(meijerg([[], []], [[0], []], -z)) == exp(z)

    assert hyperexpand(meijerg([[1, 1], []], [[1], [0]], z)) == \
        log(z + 1)
    assert hyperexpand(meijerg([[1, 1], []], [[1], [1]], z)) == \
        z/(z + 1)
    assert hyperexpand(meijerg([[], []], [[S.Half], [0]], (z/2)**2)) \
        == sin(z)/sqrt(pi)
    assert hyperexpand(meijerg([[], []], [[0], [S.Half]], (z/2)**2)) \
        == cos(z)/sqrt(pi)
    assert can_do_meijer([], [a], [a - 1, a - S.Half], [])
    assert can_do_meijer([], [], [a/2], [-a/2], False)  # branches...
    assert can_do_meijer([a], [b], [a], [b, a - 1])

    # wikipedia
    assert hyperexpand(meijerg([1], [], [], [0], z)) == \
        Piecewise((0, abs(z) < 1), (1, abs(1/z) < 1),
                 (meijerg([1], [], [], [0], z), True))
    assert hyperexpand(meijerg([], [1], [0], [], z)) == \
        Piecewise((1, abs(z) < 1), (0, abs(1/z) < 1),
                 (meijerg([], [1], [0], [], z), True))

    # The Special Functions and their Approximations
    assert can_do_meijer([], [], [a + b/2], [a, a - b/2, a + S.Half])
    assert can_do_meijer(
        [], [], [a], [b], False)  # branches only agree for small z
    assert can_do_meijer([], [S.Half], [a], [-a])
    assert can_do_meijer([], [], [a, b], [])
    assert can_do_meijer([], [], [a, b], [])
    assert can_do_meijer([], [], [a, a + S.Half], [b, b + S.Half])
    assert can_do_meijer([], [], [a, -a], [0, S.Half], False)  # dito
    assert can_do_meijer([], [], [a, a + S.Half, b, b + S.Half], [])
    assert can_do_meijer([S.Half], [], [0], [a, -a])
    assert can_do_meijer([S.Half], [], [a], [0, -a], False)  # dito
    assert can_do_meijer([], [a - S.Half], [a, b], [a - S.Half], False)
    assert can_do_meijer([], [a + S.Half], [a + b, a - b, a], [], False)
    assert can_do_meijer([a + S.Half], [], [b, 2*a - b, a], [], False)

    # This for example is actually zero.
    assert can_do_meijer([], [], [], [a, b])

    # Testing a bug:
    assert hyperexpand(meijerg([0, 2], [], [], [-1, 1], z)) == \
        Piecewise((0, abs(z) < 1),
                  (z*(1 - 1/z**2)/2, abs(1/z) < 1),
                  (meijerg([0, 2], [], [], [-1, 1], z), True))

    # Test that the simplest possible answer is returned:
    assert gammasimp(simplify(hyperexpand(
        meijerg([1], [1 - a], [-a/2, -a/2 + S.Half], [], 1/z)))) == \
        -2*sqrt(pi)*(sqrt(z + 1) + 1)**a/a

    # Test that hyper is returned
    assert hyperexpand(meijerg([1], [], [a], [0, 0], z)) == hyper(
        (a,), (a + 1, a + 1), z*exp_polar(I*pi))*z**a*gamma(a)/gamma(a + 1)**2

    # Test place option
    f = meijerg(((0, 1), ()), ((S.Half,), (0,)), z**2)
    assert hyperexpand(f) == sqrt(pi)/sqrt(1 + z**(-2))
    assert hyperexpand(f, place=0) == sqrt(pi)*z/sqrt(z**2 + 1)


def test_meijerg_lookup():
    from sympy.functions.special.error_functions import (Ci, Si)
    from sympy.functions.special.gamma_functions import uppergamma
    assert hyperexpand(meijerg([a], [], [b, a], [], z)) == \
        z**b*exp(z)*gamma(-a + b + 1)*uppergamma(a - b, z)
    assert hyperexpand(meijerg([0], [], [0, 0], [], z)) == \
        exp(z)*uppergamma(0, z)
    assert can_do_meijer([a], [], [b, a + 1], [])
    assert can_do_meijer([a], [], [b + 2, a], [])
    assert can_do_meijer([a], [], [b - 2, a], [])

    assert hyperexpand(meijerg([a], [], [a, a, a - S.Half], [], z)) == \
        -sqrt(pi)*z**(a - S.Half)*(2*cos(2*sqrt(z))*(Si(2*sqrt(z)) - pi/2)
                                   - 2*sin(2*sqrt(z))*Ci(2*sqrt(z))) == \
        hyperexpand(meijerg([a], [], [a, a - S.Half, a], [], z)) == \
        hyperexpand(meijerg([a], [], [a - S.Half, a, a], [], z))
    assert can_do_meijer([a - 1], [], [a + 2, a - Rational(3, 2), a + 1], [])


@XFAIL
def test_meijerg_expand_fail():
    # These basically test hyper([], [1/2 - a, 1/2 + 1, 1/2], z),
    # which is *very* messy. But since the meijer g actually yields a
    # sum of bessel functions, things can sometimes be simplified a lot and
    # are then put into tables...
    assert can_do_meijer([], [], [a + S.Half], [a, a - b/2, a + b/2])
    assert can_do_meijer([], [], [0, S.Half], [a, -a])
    assert can_do_meijer([], [], [3*a - S.Half, a, -a - S.Half], [a - S.Half])
    assert can_do_meijer([], [], [0, a - S.Half, -a - S.Half], [S.Half])
    assert can_do_meijer([], [], [a, b + S.Half, b], [2*b - a])
    assert can_do_meijer([], [], [a, b + S.Half, b, 2*b - a])
    assert can_do_meijer([S.Half], [], [-a, a], [0])


@slow
def test_meijerg():
    # carefully set up the parameters.
    # NOTE: this used to fail sometimes. I believe it is fixed, but if you
    #       hit an inexplicable test failure here, please let me know the seed.
    a1, a2 = (randcplx(n) - 5*I - n*I for n in range(2))
    b1, b2 = (randcplx(n) + 5*I + n*I for n in range(2))
    b3, b4, b5, a3, a4, a5 = (randcplx() for n in range(6))
    g = meijerg([a1], [a3, a4], [b1], [b3, b4], z)

    assert ReduceOrder.meijer_minus(3, 4) is None
    assert ReduceOrder.meijer_plus(4, 3) is None

    g2 = meijerg([a1, a2], [a3, a4], [b1], [b3, b4, a2], z)
    assert tn(ReduceOrder.meijer_plus(a2, a2).apply(g, op), g2, z)

    g2 = meijerg([a1, a2], [a3, a4], [b1], [b3, b4, a2 + 1], z)
    assert tn(ReduceOrder.meijer_plus(a2, a2 + 1).apply(g, op), g2, z)

    g2 = meijerg([a1, a2 - 1], [a3, a4], [b1], [b3, b4, a2 + 2], z)
    assert tn(ReduceOrder.meijer_plus(a2 - 1, a2 + 2).apply(g, op), g2, z)

    g2 = meijerg([a1], [a3, a4, b2 - 1], [b1, b2 + 2], [b3, b4], z)
    assert tn(ReduceOrder.meijer_minus(
        b2 + 2, b2 - 1).apply(g, op), g2, z, tol=1e-6)

    # test several-step reduction
    an = [a1, a2]
    bq = [b3, b4, a2 + 1]
    ap = [a3, a4, b2 - 1]
    bm = [b1, b2 + 1]
    niq, ops = reduce_order_meijer(G_Function(an, ap, bm, bq))
    assert niq.an == (a1,)
    assert set(niq.ap) == {a3, a4}
    assert niq.bm == (b1,)
    assert set(niq.bq) == {b3, b4}
    assert tn(apply_operators(g, ops, op), meijerg(an, ap, bm, bq, z), z)


def test_meijerg_shift_operators():
    # carefully set up the parameters. XXX this still fails sometimes
    a1, a2, a3, a4, a5, b1, b2, b3, b4, b5 = (randcplx(n) for n in range(10))
    g = meijerg([a1], [a3, a4], [b1], [b3, b4], z)

    assert tn(MeijerShiftA(b1).apply(g, op),
              meijerg([a1], [a3, a4], [b1 + 1], [b3, b4], z), z)
    assert tn(MeijerShiftB(a1).apply(g, op),
              meijerg([a1 - 1], [a3, a4], [b1], [b3, b4], z), z)
    assert tn(MeijerShiftC(b3).apply(g, op),
              meijerg([a1], [a3, a4], [b1], [b3 + 1, b4], z), z)
    assert tn(MeijerShiftD(a3).apply(g, op),
              meijerg([a1], [a3 - 1, a4], [b1], [b3, b4], z), z)

    s = MeijerUnShiftA([a1], [a3, a4], [b1], [b3, b4], 0, z)
    assert tn(
        s.apply(g, op), meijerg([a1], [a3, a4], [b1 - 1], [b3, b4], z), z)

    s = MeijerUnShiftC([a1], [a3, a4], [b1], [b3, b4], 0, z)
    assert tn(
        s.apply(g, op), meijerg([a1], [a3, a4], [b1], [b3 - 1, b4], z), z)

    s = MeijerUnShiftB([a1], [a3, a4], [b1], [b3, b4], 0, z)
    assert tn(
        s.apply(g, op), meijerg([a1 + 1], [a3, a4], [b1], [b3, b4], z), z)

    s = MeijerUnShiftD([a1], [a3, a4], [b1], [b3, b4], 0, z)
    assert tn(
        s.apply(g, op), meijerg([a1], [a3 + 1, a4], [b1], [b3, b4], z), z)


@slow
def test_meijerg_confluence():
    def t(m, a, b):
        from sympy.core.sympify import sympify
        a, b = sympify([a, b])
        m_ = m
        m = hyperexpand(m)
        if not m == Piecewise((a, abs(z) < 1), (b, abs(1/z) < 1), (m_, True)):
            return False
        if not (m.args[0].args[0] == a and m.args[1].args[0] == b):
            return False
        z0 = randcplx()/10
        if abs(m.subs(z, z0).n() - a.subs(z, z0).n()).n() > 1e-10:
            return False
        if abs(m.subs(z, 1/z0).n() - b.subs(z, 1/z0).n()).n() > 1e-10:
            return False
        return True

    assert t(meijerg([], [1, 1], [0, 0], [], z), -log(z), 0)
    assert t(meijerg(
        [], [3, 1], [0, 0], [], z), -z**2/4 + z - log(z)/2 - Rational(3, 4), 0)
    assert t(meijerg([], [3, 1], [-1, 0], [], z),
             z**2/12 - z/2 + log(z)/2 + Rational(1, 4) + 1/(6*z), 0)
    assert t(meijerg([], [1, 1, 1, 1], [0, 0, 0, 0], [], z), -log(z)**3/6, 0)
    assert t(meijerg([1, 1], [], [], [0, 0], z), 0, -log(1/z))
    assert t(meijerg([1, 1], [2, 2], [1, 1], [0, 0], z),
             -z*log(z) + 2*z, -log(1/z) + 2)
    assert t(meijerg([S.Half], [1, 1], [0, 0], [Rational(3, 2)], z), log(z)/2 - 1, 0)

    def u(an, ap, bm, bq):
        m = meijerg(an, ap, bm, bq, z)
        m2 = hyperexpand(m, allow_hyper=True)
        if m2.has(meijerg) and not (m2.is_Piecewise and len(m2.args) == 3):
            return False
        return tn(m, m2, z)
    assert u([], [1], [0, 0], [])
    assert u([1, 1], [], [], [0])
    assert u([1, 1], [2, 2, 5], [1, 1, 6], [0, 0])
    assert u([1, 1], [2, 2, 5], [1, 1, 6], [0])


def test_meijerg_with_Floats():
    # see issue #10681
    from sympy.polys.domains.realfield import RR
    f = meijerg(((3.0, 1), ()), ((Rational(3, 2),), (0,)), z)
    a = -2.3632718012073
    g = a*z**Rational(3, 2)*hyper((-0.5, Rational(3, 2)), (Rational(5, 2),), z*exp_polar(I*pi))
    assert RR.almosteq((hyperexpand(f)/g).n(), 1.0, 1e-12)


def test_lerchphi():
    from sympy.functions.special.zeta_functions import (lerchphi, polylog)
    from sympy.simplify.gammasimp import gammasimp
    assert hyperexpand(hyper([1, a], [a + 1], z)/a) == lerchphi(z, 1, a)
    assert hyperexpand(
        hyper([1, a, a], [a + 1, a + 1], z)/a**2) == lerchphi(z, 2, a)
    assert hyperexpand(hyper([1, a, a, a], [a + 1, a + 1, a + 1], z)/a**3) == \
        lerchphi(z, 3, a)
    assert hyperexpand(hyper([1] + [a]*10, [a + 1]*10, z)/a**10) == \
        lerchphi(z, 10, a)
    assert gammasimp(hyperexpand(meijerg([0, 1 - a], [], [0],
        [-a], exp_polar(-I*pi)*z))) == lerchphi(z, 1, a)
    assert gammasimp(hyperexpand(meijerg([0, 1 - a, 1 - a], [], [0],
        [-a, -a], exp_polar(-I*pi)*z))) == lerchphi(z, 2, a)
    assert gammasimp(hyperexpand(meijerg([0, 1 - a, 1 - a, 1 - a], [], [0],
        [-a, -a, -a], exp_polar(-I*pi)*z))) == lerchphi(z, 3, a)

    assert hyperexpand(z*hyper([1, 1], [2], z)) == -log(1 + -z)
    assert hyperexpand(z*hyper([1, 1, 1], [2, 2], z)) == polylog(2, z)
    assert hyperexpand(z*hyper([1, 1, 1, 1], [2, 2, 2], z)) == polylog(3, z)

    assert hyperexpand(hyper([1, a, 1 + S.Half], [a + 1, S.Half], z)) == \
        -2*a/(z - 1) + (-2*a**2 + a)*lerchphi(z, 1, a)

    # Now numerical tests. These make sure reductions etc are carried out
    # correctly

    # a rational function (polylog at negative integer order)
    assert can_do([2, 2, 2], [1, 1])

    # NOTE these contain log(1-x) etc ... better make sure we have |z| < 1
    # reduction of order for polylog
    assert can_do([1, 1, 1, b + 5], [2, 2, b], div=10)

    # reduction of order for lerchphi
    # XXX lerchphi in mpmath is flaky
    assert can_do(
        [1, a, a, a, b + 5], [a + 1, a + 1, a + 1, b], numerical=False)

    # test a bug
    from sympy.functions.elementary.complexes import Abs
    assert hyperexpand(hyper([S.Half, S.Half, S.Half, 1],
                             [Rational(3, 2), Rational(3, 2), Rational(3, 2)], Rational(1, 4))) == \
        Abs(-polylog(3, exp_polar(I*pi)/2) + polylog(3, S.Half))


def test_partial_simp():
    # First test that hypergeometric function formulae work.
    a, b, c, d, e = (randcplx() for _ in range(5))
    for func in [Hyper_Function([a, b, c], [d, e]),
            Hyper_Function([], [a, b, c, d, e])]:
        f = build_hypergeometric_formula(func)
        z = f.z
        assert f.closed_form == func(z)
        deriv1 = f.B.diff(z)*z
        deriv2 = f.M*f.B
        for func1, func2 in zip(deriv1, deriv2):
            assert tn(func1, func2, z)

    # Now test that formulae are partially simplified.
    a, b, z = symbols('a b z')
    assert hyperexpand(hyper([3, a], [1, b], z)) == \
        (-a*b/2 + a*z/2 + 2*a)*hyper([a + 1], [b], z) \
        + (a*b/2 - 2*a + 1)*hyper([a], [b], z)
    assert tn(
        hyperexpand(hyper([3, d], [1, e], z)), hyper([3, d], [1, e], z), z)
    assert hyperexpand(hyper([3], [1, a, b], z)) == \
        hyper((), (a, b), z) \
        + z*hyper((), (a + 1, b), z)/(2*a) \
        - z*(b - 4)*hyper((), (a + 1, b + 1), z)/(2*a*b)
    assert tn(
        hyperexpand(hyper([3], [1, d, e], z)), hyper([3], [1, d, e], z), z)


def test_hyperexpand_special():
    assert hyperexpand(hyper([a, b], [c], 1)) == \
        gamma(c)*gamma(c - a - b)/gamma(c - a)/gamma(c - b)
    assert hyperexpand(hyper([a, b], [1 + a - b], -1)) == \
        gamma(1 + a/2)*gamma(1 + a - b)/gamma(1 + a)/gamma(1 + a/2 - b)
    assert hyperexpand(hyper([a, b], [1 + b - a], -1)) == \
        gamma(1 + b/2)*gamma(1 + b - a)/gamma(1 + b)/gamma(1 + b/2 - a)
    assert hyperexpand(meijerg([1 - z - a/2], [1 - z + a/2], [b/2], [-b/2], 1)) == \
        gamma(1 - 2*z)*gamma(z + a/2 + b/2)/gamma(1 - z + a/2 - b/2) \
        /gamma(1 - z - a/2 + b/2)/gamma(1 - z + a/2 + b/2)
    assert hyperexpand(hyper([a], [b], 0)) == 1
    assert hyper([a], [b], 0) != 0


def test_Mod1_behavior():
    from sympy.core.symbol import Symbol
    from sympy.simplify.simplify import simplify
    n = Symbol('n', integer=True)
    # Note: this should not hang.
    assert simplify(hyperexpand(meijerg([1], [], [n + 1], [0], z))) == \
        lowergamma(n + 1, z)


@slow
def test_prudnikov_misc():
    assert can_do([1, (3 + I)/2, (3 - I)/2], [Rational(3, 2), 2])
    assert can_do([S.Half, a - 1], [Rational(3, 2), a + 1], lowerplane=True)
    assert can_do([], [b + 1])
    assert can_do([a], [a - 1, b + 1])

    assert can_do([a], [a - S.Half, 2*a])
    assert can_do([a], [a - S.Half, 2*a + 1])
    assert can_do([a], [a - S.Half, 2*a - 1])
    assert can_do([a], [a + S.Half, 2*a])
    assert can_do([a], [a + S.Half, 2*a + 1])
    assert can_do([a], [a + S.Half, 2*a - 1])
    assert can_do([S.Half], [b, 2 - b])
    assert can_do([S.Half], [b, 3 - b])
    assert can_do([1], [2, b])

    assert can_do([a, a + S.Half], [2*a, b, 2*a - b + 1])
    assert can_do([a, a + S.Half], [S.Half, 2*a, 2*a + S.Half])
    assert can_do([a], [a + 1], lowerplane=True)  # lowergamma


def test_prudnikov_1():
    # A. P. Prudnikov, Yu. A. Brychkov and O. I. Marichev (1990).
    # Integrals and Series: More Special Functions, Vol. 3,.
    # Gordon and Breach Science Publisher

    # 7.3.1
    assert can_do([a, -a], [S.Half])
    assert can_do([a, 1 - a], [S.Half])
    assert can_do([a, 1 - a], [Rational(3, 2)])
    assert can_do([a, 2 - a], [S.Half])
    assert can_do([a, 2 - a], [Rational(3, 2)])
    assert can_do([a, 2 - a], [Rational(3, 2)])
    assert can_do([a, a + S.Half], [2*a - 1])
    assert can_do([a, a + S.Half], [2*a])
    assert can_do([a, a + S.Half], [2*a + 1])
    assert can_do([a, a + S.Half], [S.Half])
    assert can_do([a, a + S.Half], [Rational(3, 2)])
    assert can_do([a, a/2 + 1], [a/2])
    assert can_do([1, b], [2])
    assert can_do([1, b], [b + 1], numerical=False)  # Lerch Phi
             # NOTE: branches are complicated for |z| > 1

    assert can_do([a], [2*a])
    assert can_do([a], [2*a + 1])
    assert can_do([a], [2*a - 1])


@slow
def test_prudnikov_2():
    h = S.Half
    assert can_do([-h, -h], [h])
    assert can_do([-h, h], [3*h])
    assert can_do([-h, h], [5*h])
    assert can_do([-h, h], [7*h])
    assert can_do([-h, 1], [h])

    for p in [-h, h]:
        for n in [-h, h, 1, 3*h, 2, 5*h, 3, 7*h, 4]:
            for m in [-h, h, 3*h, 5*h, 7*h]:
                assert can_do([p, n], [m])
        for n in [1, 2, 3, 4]:
            for m in [1, 2, 3, 4]:
                assert can_do([p, n], [m])


def test_prudnikov_3():
    h = S.Half
    assert can_do([Rational(1, 4), Rational(3, 4)], [h])
    assert can_do([Rational(1, 4), Rational(3, 4)], [3*h])
    assert can_do([Rational(1, 3), Rational(2, 3)], [3*h])
    assert can_do([Rational(3, 4), Rational(5, 4)], [h])
    assert can_do([Rational(3, 4), Rational(5, 4)], [3*h])


@tooslow
def test_prudnikov_3_slow():
    # XXX: This is marked as tooslow and hence skipped in CI. None of the
    # individual cases below fails or hangs. Some cases are slow and the loops
    # below generate 280 different cases. Is it really necessary to test all
    # 280 cases here?
    h = S.Half
    for p in [1, 2, 3, 4]:
        for n in [-h, h, 1, 3*h, 2, 5*h, 3, 7*h, 4, 9*h]:
            for m in [1, 3*h, 2, 5*h, 3, 7*h, 4]:
                assert can_do([p, m], [n])


@slow
def test_prudnikov_4():
    h = S.Half
    for p in [3*h, 5*h, 7*h]:
        for n in [-h, h, 3*h, 5*h, 7*h]:
            for m in [3*h, 2, 5*h, 3, 7*h, 4]:
                assert can_do([p, m], [n])
        for n in [1, 2, 3, 4]:
            for m in [2, 3, 4]:
                assert can_do([p, m], [n])


@slow
def test_prudnikov_5():
    h = S.Half

    for p in [1, 2, 3]:
        for q in range(p, 4):
            for r in [1, 2, 3]:
                for s in range(r, 4):
                    assert can_do([-h, p, q], [r, s])

    for p in [h, 1, 3*h, 2, 5*h, 3]:
        for q in [h, 3*h, 5*h]:
            for r in [h, 3*h, 5*h]:
                for s in [h, 3*h, 5*h]:
                    if s <= q and s <= r:
                        assert can_do([-h, p, q], [r, s])

    for p in [h, 1, 3*h, 2, 5*h, 3]:
        for q in [1, 2, 3]:
            for r in [h, 3*h, 5*h]:
                for s in [1, 2, 3]:
                    assert can_do([-h, p, q], [r, s])


@slow
def test_prudnikov_6():
    h = S.Half

    for m in [3*h, 5*h]:
        for n in [1, 2, 3]:
            for q in [h, 1, 2]:
                for p in [1, 2, 3]:
                    assert can_do([h, q, p], [m, n])
            for q in [1, 2, 3]:
                for p in [3*h, 5*h]:
                    assert can_do([h, q, p], [m, n])

    for q in [1, 2]:
        for p in [1, 2, 3]:
            for m in [1, 2, 3]:
                for n in [1, 2, 3]:
                    assert can_do([h, q, p], [m, n])

    assert can_do([h, h, 5*h], [3*h, 3*h])
    assert can_do([h, 1, 5*h], [3*h, 3*h])
    assert can_do([h, 2, 2], [1, 3])

    # pages 435 to 457 contain more PFDD and stuff like this


@slow
def test_prudnikov_7():
    assert can_do([3], [6])

    h = S.Half
    for n in [h, 3*h, 5*h, 7*h]:
        assert can_do([-h], [n])
    for m in [-h, h, 1, 3*h, 2, 5*h, 3, 7*h, 4]:  # HERE
        for n in [-h, h, 3*h, 5*h, 7*h, 1, 2, 3, 4]:
            assert can_do([m], [n])


@slow
def test_prudnikov_8():
    h = S.Half

    # 7.12.2
    for ai in [1, 2, 3]:
        for bi in [1, 2, 3]:
            for ci in range(1, ai + 1):
                for di in [h, 1, 3*h, 2, 5*h, 3]:
                    assert can_do([ai, bi], [ci, di])
        for bi in [3*h, 5*h]:
            for ci in [h, 1, 3*h, 2, 5*h, 3]:
                for di in [1, 2, 3]:
                    assert can_do([ai, bi], [ci, di])

    for ai in [-h, h, 3*h, 5*h]:
        for bi in [1, 2, 3]:
            for ci in [h, 1, 3*h, 2, 5*h, 3]:
                for di in [1, 2, 3]:
                    assert can_do([ai, bi], [ci, di])
        for bi in [h, 3*h, 5*h]:
            for ci in [h, 3*h, 5*h, 3]:
                for di in [h, 1, 3*h, 2, 5*h, 3]:
                    if ci <= bi:
                        assert can_do([ai, bi], [ci, di])


def test_prudnikov_9():
    # 7.13.1 [we have a general formula ... so this is a bit pointless]
    for i in range(9):
        assert can_do([], [(S(i) + 1)/2])
    for i in range(5):
        assert can_do([], [-(2*S(i) + 1)/2])


@slow
def test_prudnikov_10():
    # 7.14.2
    h = S.Half
    for p in [-h, h, 1, 3*h, 2, 5*h, 3, 7*h, 4]:
        for m in [1, 2, 3, 4]:
            for n in range(m, 5):
                assert can_do([p], [m, n])

    for p in [1, 2, 3, 4]:
        for n in [h, 3*h, 5*h, 7*h]:
            for m in [1, 2, 3, 4]:
                assert can_do([p], [n, m])

    for p in [3*h, 5*h, 7*h]:
        for m in [h, 1, 2, 5*h, 3, 7*h, 4]:
            assert can_do([p], [h, m])
            assert can_do([p], [3*h, m])

    for m in [h, 1, 2, 5*h, 3, 7*h, 4]:
        assert can_do([7*h], [5*h, m])

    assert can_do([Rational(-1, 2)], [S.Half, S.Half])  # shine-integral shi


def test_prudnikov_11():
    # 7.15
    assert can_do([a, a + S.Half], [2*a, b, 2*a - b])
    assert can_do([a, a + S.Half], [Rational(3, 2), 2*a, 2*a - S.Half])

    assert can_do([Rational(1, 4), Rational(3, 4)], [S.Half, S.Half, 1])
    assert can_do([Rational(5, 4), Rational(3, 4)], [Rational(3, 2), S.Half, 2])
    assert can_do([Rational(5, 4), Rational(3, 4)], [Rational(3, 2), Rational(3, 2), 1])
    assert can_do([Rational(5, 4), Rational(7, 4)], [Rational(3, 2), Rational(5, 2), 2])

    assert can_do([1, 1], [Rational(3, 2), 2, 2])  # cosh-integral chi


def test_prudnikov_12():
    # 7.16
    assert can_do(
        [], [a, a + S.Half, 2*a], False)  # branches only agree for some z!
    assert can_do([], [a, a + S.Half, 2*a + 1], False)  # dito
    assert can_do([], [S.Half, a, a + S.Half])
    assert can_do([], [Rational(3, 2), a, a + S.Half])

    assert can_do([], [Rational(1, 4), S.Half, Rational(3, 4)])
    assert can_do([], [S.Half, S.Half, 1])
    assert can_do([], [S.Half, Rational(3, 2), 1])
    assert can_do([], [Rational(3, 4), Rational(3, 2), Rational(5, 4)])
    assert can_do([], [1, 1, Rational(3, 2)])
    assert can_do([], [1, 2, Rational(3, 2)])
    assert can_do([], [1, Rational(3, 2), Rational(3, 2)])
    assert can_do([], [Rational(5, 4), Rational(3, 2), Rational(7, 4)])
    assert can_do([], [2, Rational(3, 2), Rational(3, 2)])


@slow
def test_prudnikov_2F1():
    h = S.Half
    # Elliptic integrals
    for p in [-h, h]:
        for m in [h, 3*h, 5*h, 7*h]:
            for n in [1, 2, 3, 4]:
                assert can_do([p, m], [n])


@XFAIL
def test_prudnikov_fail_2F1():
    assert can_do([a, b], [b + 1])  # incomplete beta function
    assert can_do([-1, b], [c])    # Poly. also -2, -3 etc

    # TODO polys

    # Legendre functions:
    assert can_do([a, b], [a + b + S.Half])
    assert can_do([a, b], [a + b - S.Half])
    assert can_do([a, b], [a + b + Rational(3, 2)])
    assert can_do([a, b], [(a + b + 1)/2])
    assert can_do([a, b], [(a + b)/2 + 1])
    assert can_do([a, b], [a - b + 1])
    assert can_do([a, b], [a - b + 2])
    assert can_do([a, b], [2*b])
    assert can_do([a, b], [S.Half])
    assert can_do([a, b], [Rational(3, 2)])
    assert can_do([a, 1 - a], [c])
    assert can_do([a, 2 - a], [c])
    assert can_do([a, 3 - a], [c])
    assert can_do([a, a + S.Half], [c])
    assert can_do([1, b], [c])
    assert can_do([1, b], [Rational(3, 2)])

    assert can_do([Rational(1, 4), Rational(3, 4)], [1])

    # PFDD
    o = S.One
    assert can_do([o/8, 1], [o/8*9])
    assert can_do([o/6, 1], [o/6*7])
    assert can_do([o/6, 1], [o/6*13])
    assert can_do([o/5, 1], [o/5*6])
    assert can_do([o/5, 1], [o/5*11])
    assert can_do([o/4, 1], [o/4*5])
    assert can_do([o/4, 1], [o/4*9])
    assert can_do([o/3, 1], [o/3*4])
    assert can_do([o/3, 1], [o/3*7])
    assert can_do([o/8*3, 1], [o/8*11])
    assert can_do([o/5*2, 1], [o/5*7])
    assert can_do([o/5*2, 1], [o/5*12])
    assert can_do([o/5*3, 1], [o/5*8])
    assert can_do([o/5*3, 1], [o/5*13])
    assert can_do([o/8*5, 1], [o/8*13])
    assert can_do([o/4*3, 1], [o/4*7])
    assert can_do([o/4*3, 1], [o/4*11])
    assert can_do([o/3*2, 1], [o/3*5])
    assert can_do([o/3*2, 1], [o/3*8])
    assert can_do([o/5*4, 1], [o/5*9])
    assert can_do([o/5*4, 1], [o/5*14])
    assert can_do([o/6*5, 1], [o/6*11])
    assert can_do([o/6*5, 1], [o/6*17])
    assert can_do([o/8*7, 1], [o/8*15])


@XFAIL
def test_prudnikov_fail_3F2():
    assert can_do([a, a + Rational(1, 3), a + Rational(2, 3)], [Rational(1, 3), Rational(2, 3)])
    assert can_do([a, a + Rational(1, 3), a + Rational(2, 3)], [Rational(2, 3), Rational(4, 3)])
    assert can_do([a, a + Rational(1, 3), a + Rational(2, 3)], [Rational(4, 3), Rational(5, 3)])

    # page 421
    assert can_do([a, a + Rational(1, 3), a + Rational(2, 3)], [a*Rational(3, 2), (3*a + 1)/2])

    # pages 422 ...
    assert can_do([Rational(-1, 2), S.Half, S.Half], [1, 1])  # elliptic integrals
    assert can_do([Rational(-1, 2), S.Half, 1], [Rational(3, 2), Rational(3, 2)])
    # TODO LOTS more

    # PFDD
    assert can_do([Rational(1, 8), Rational(3, 8), 1], [Rational(9, 8), Rational(11, 8)])
    assert can_do([Rational(1, 8), Rational(5, 8), 1], [Rational(9, 8), Rational(13, 8)])
    assert can_do([Rational(1, 8), Rational(7, 8), 1], [Rational(9, 8), Rational(15, 8)])
    assert can_do([Rational(1, 6), Rational(1, 3), 1], [Rational(7, 6), Rational(4, 3)])
    assert can_do([Rational(1, 6), Rational(2, 3), 1], [Rational(7, 6), Rational(5, 3)])
    assert can_do([Rational(1, 6), Rational(2, 3), 1], [Rational(5, 3), Rational(13, 6)])
    assert can_do([S.Half, 1, 1], [Rational(1, 4), Rational(3, 4)])
    # LOTS more


@XFAIL
def test_prudnikov_fail_other():
    # 7.11.2

    # 7.12.1
    assert can_do([1, a], [b, 1 - 2*a + b])  # ???

    # 7.14.2
    assert can_do([Rational(-1, 2)], [S.Half, 1])  # struve
    assert can_do([1], [S.Half, S.Half])  # struve
    assert can_do([Rational(1, 4)], [S.Half, Rational(5, 4)])  # PFDD
    assert can_do([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)])  # PFDD
    assert can_do([1], [Rational(1, 4), Rational(3, 4)])  # PFDD
    assert can_do([1], [Rational(3, 4), Rational(5, 4)])  # PFDD
    assert can_do([1], [Rational(5, 4), Rational(7, 4)])  # PFDD
    # TODO LOTS more

    # 7.15.2
    assert can_do([S.Half, 1], [Rational(3, 4), Rational(5, 4), Rational(3, 2)])  # PFDD
    assert can_do([S.Half, 1], [Rational(7, 4), Rational(5, 4), Rational(3, 2)])  # PFDD

    # 7.16.1
    assert can_do([], [Rational(1, 3), S(2/3)])  # PFDD
    assert can_do([], [Rational(2, 3), S(4/3)])  # PFDD
    assert can_do([], [Rational(5, 3), S(4/3)])  # PFDD

    # XXX this does not *evaluate* right??
    assert can_do([], [a, a + S.Half, 2*a - 1])


def test_bug():
    h = hyper([-1, 1], [z], -1)
    assert hyperexpand(h) == (z + 1)/z


def test_omgissue_203():
    h = hyper((-5, -3, -4), (-6, -6), 1)
    assert hyperexpand(h) == Rational(1, 30)
    h = hyper((-6, -7, -5), (-6, -6), 1)
    assert hyperexpand(h) == Rational(-1, 6)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\test_old_base.py ===
from __future__ import annotations

from datetime import datetime
import weakref

import numpy as np
import pytest

from pandas._libs.tslibs import Timestamp

from pandas.core.dtypes.common import (
    is_integer_dtype,
    is_numeric_dtype,
)
from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    CategoricalIndex,
    DatetimeIndex,
    DatetimeTZDtype,
    Index,
    IntervalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    StringDtype,
    TimedeltaIndex,
    isna,
    period_range,
)
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import BaseMaskedArray


class TestBase:
    @pytest.fixture(
        params=[
            RangeIndex(start=0, stop=20, step=2),
            Index(np.arange(5, dtype=np.float64)),
            Index(np.arange(5, dtype=np.float32)),
            Index(np.arange(5, dtype=np.uint64)),
            Index(range(0, 20, 2), dtype=np.int64),
            Index(range(0, 20, 2), dtype=np.int32),
            Index(range(0, 20, 2), dtype=np.int16),
            Index(range(0, 20, 2), dtype=np.int8),
            Index(list("abcde")),
            Index([0, "a", 1, "b", 2, "c"]),
            period_range("20130101", periods=5, freq="D"),
            TimedeltaIndex(
                [
                    "0 days 01:00:00",
                    "1 days 01:00:00",
                    "2 days 01:00:00",
                    "3 days 01:00:00",
                    "4 days 01:00:00",
                ],
                dtype="timedelta64[ns]",
                freq="D",
            ),
            DatetimeIndex(
                ["2013-01-01", "2013-01-02", "2013-01-03", "2013-01-04", "2013-01-05"],
                dtype="datetime64[ns]",
                freq="D",
            ),
            IntervalIndex.from_breaks(range(11), closed="right"),
        ]
    )
    def simple_index(self, request):
        return request.param

    def test_pickle_compat_construction(self, simple_index):
        # need an object to create with
        if isinstance(simple_index, RangeIndex):
            pytest.skip("RangeIndex() is a valid constructor")
        msg = "|".join(
            [
                r"Index\(\.\.\.\) must be called with a collection of some "
                r"kind, None was passed",
                r"DatetimeIndex\(\) must be called with a collection of some "
                r"kind, None was passed",
                r"TimedeltaIndex\(\) must be called with a collection of some "
                r"kind, None was passed",
                r"__new__\(\) missing 1 required positional argument: 'data'",
                r"__new__\(\) takes at least 2 arguments \(1 given\)",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            type(simple_index)()

    def test_shift(self, simple_index):
        # GH8083 test the base class for shift
        if isinstance(simple_index, (DatetimeIndex, TimedeltaIndex, PeriodIndex)):
            pytest.skip("Tested in test_ops/test_arithmetic")
        idx = simple_index
        msg = (
            f"This method is only implemented for DatetimeIndex, PeriodIndex and "
            f"TimedeltaIndex; Got type {type(idx).__name__}"
        )
        with pytest.raises(NotImplementedError, match=msg):
            idx.shift(1)
        with pytest.raises(NotImplementedError, match=msg):
            idx.shift(1, 2)

    def test_constructor_name_unhashable(self, simple_index):
        # GH#29069 check that name is hashable
        # See also same-named test in tests.series.test_constructors
        idx = simple_index
        with pytest.raises(TypeError, match="Index.name must be a hashable type"):
            type(idx)(idx, name=[])

    def test_create_index_existing_name(self, simple_index):
        # GH11193, when an existing index is passed, and a new name is not
        # specified, the new index should inherit the previous object name
        expected = simple_index.copy()
        if not isinstance(expected, MultiIndex):
            expected.name = "foo"
            result = Index(expected)
            tm.assert_index_equal(result, expected)

            result = Index(expected, name="bar")
            expected.name = "bar"
            tm.assert_index_equal(result, expected)
        else:
            expected.names = ["foo", "bar"]
            result = Index(expected)
            tm.assert_index_equal(
                result,
                Index(
                    Index(
                        [
                            ("foo", "one"),
                            ("foo", "two"),
                            ("bar", "one"),
                            ("baz", "two"),
                            ("qux", "one"),
                            ("qux", "two"),
                        ],
                        dtype="object",
                    ),
                    names=["foo", "bar"],
                ),
            )

            result = Index(expected, names=["A", "B"])
            tm.assert_index_equal(
                result,
                Index(
                    Index(
                        [
                            ("foo", "one"),
                            ("foo", "two"),
                            ("bar", "one"),
                            ("baz", "two"),
                            ("qux", "one"),
                            ("qux", "two"),
                        ],
                        dtype="object",
                    ),
                    names=["A", "B"],
                ),
            )

    def test_numeric_compat(self, simple_index):
        idx = simple_index
        # Check that this doesn't cover MultiIndex case, if/when it does,
        #  we can remove multi.test_compat.test_numeric_compat
        assert not isinstance(idx, MultiIndex)
        if type(idx) is Index:
            pytest.skip("Not applicable for Index")
        if is_numeric_dtype(simple_index.dtype) or isinstance(
            simple_index, TimedeltaIndex
        ):
            pytest.skip("Tested elsewhere.")

        typ = type(idx._data).__name__
        cls = type(idx).__name__
        lmsg = "|".join(
            [
                rf"unsupported operand type\(s\) for \*: '{typ}' and 'int'",
                "cannot perform (__mul__|__truediv__|__floordiv__) with "
                f"this index type: ({cls}|{typ})",
            ]
        )
        with pytest.raises(TypeError, match=lmsg):
            idx * 1
        rmsg = "|".join(
            [
                rf"unsupported operand type\(s\) for \*: 'int' and '{typ}'",
                "cannot perform (__rmul__|__rtruediv__|__rfloordiv__) with "
                f"this index type: ({cls}|{typ})",
            ]
        )
        with pytest.raises(TypeError, match=rmsg):
            1 * idx

        div_err = lmsg.replace("*", "/")
        with pytest.raises(TypeError, match=div_err):
            idx / 1
        div_err = rmsg.replace("*", "/")
        with pytest.raises(TypeError, match=div_err):
            1 / idx

        floordiv_err = lmsg.replace("*", "//")
        with pytest.raises(TypeError, match=floordiv_err):
            idx // 1
        floordiv_err = rmsg.replace("*", "//")
        with pytest.raises(TypeError, match=floordiv_err):
            1 // idx

    def test_logical_compat(self, simple_index):
        if simple_index.dtype in (object, "string"):
            pytest.skip("Tested elsewhere.")
        idx = simple_index
        if idx.dtype.kind in "iufcbm":
            assert idx.all() == idx._values.all()
            assert idx.all() == idx.to_series().all()
            assert idx.any() == idx._values.any()
            assert idx.any() == idx.to_series().any()
        else:
            msg = "cannot perform (any|all)"
            if isinstance(idx, IntervalIndex):
                msg = (
                    r"'IntervalArray' with dtype interval\[.*\] does "
                    "not support reduction '(any|all)'"
                )
            with pytest.raises(TypeError, match=msg):
                idx.all()
            with pytest.raises(TypeError, match=msg):
                idx.any()

    def test_repr_roundtrip(self, simple_index):
        if isinstance(simple_index, IntervalIndex):
            pytest.skip(f"Not a valid repr for {type(simple_index).__name__}")
        idx = simple_index
        tm.assert_index_equal(eval(repr(idx)), idx)

    def test_repr_max_seq_item_setting(self, simple_index):
        # GH10182
        if isinstance(simple_index, IntervalIndex):
            pytest.skip(f"Not a valid repr for {type(simple_index).__name__}")
        idx = simple_index
        idx = idx.repeat(50)
        with pd.option_context("display.max_seq_items", None):
            repr(idx)
            assert "..." not in str(idx)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_ensure_copied_data(self, index):
        # Check the "copy" argument of each Index.__new__ is honoured
        # GH12309
        init_kwargs = {}
        if isinstance(index, PeriodIndex):
            # Needs "freq" specification:
            init_kwargs["freq"] = index.freq
        elif isinstance(index, (RangeIndex, MultiIndex, CategoricalIndex)):
            pytest.skip(
                "RangeIndex cannot be initialized from data, "
                "MultiIndex and CategoricalIndex are tested separately"
            )
        elif index.dtype == object and index.inferred_type in ["boolean", "string"]:
            init_kwargs["dtype"] = index.dtype

        index_type = type(index)
        result = index_type(index.values, copy=True, **init_kwargs)
        if isinstance(index.dtype, DatetimeTZDtype):
            result = result.tz_localize("UTC").tz_convert(index.tz)
        if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
            index = index._with_freq(None)

        tm.assert_index_equal(index, result)

        if isinstance(index, PeriodIndex):
            # .values an object array of Period, thus copied
            depr_msg = "The 'ordinal' keyword in PeriodIndex is deprecated"
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                result = index_type(ordinal=index.asi8, copy=False, **init_kwargs)
            tm.assert_numpy_array_equal(index.asi8, result.asi8, check_same="same")
        elif isinstance(index, IntervalIndex):
            # checked in test_interval.py
            pass
        elif type(index) is Index and not isinstance(index.dtype, np.dtype):
            result = index_type(index.values, copy=False, **init_kwargs)
            tm.assert_index_equal(result, index)

            if isinstance(index._values, BaseMaskedArray):
                assert np.shares_memory(index._values._data, result._values._data)
                tm.assert_numpy_array_equal(
                    index._values._data, result._values._data, check_same="same"
                )
                assert np.shares_memory(index._values._mask, result._values._mask)
                tm.assert_numpy_array_equal(
                    index._values._mask, result._values._mask, check_same="same"
                )
            elif (
                isinstance(index.dtype, StringDtype) and index.dtype.storage == "python"
            ):
                assert np.shares_memory(index._values._ndarray, result._values._ndarray)
                tm.assert_numpy_array_equal(
                    index._values._ndarray, result._values._ndarray, check_same="same"
                )
            elif (
                isinstance(index.dtype, StringDtype)
                and index.dtype.storage == "pyarrow"
            ):
                assert tm.shares_memory(result._values, index._values)
            else:
                raise NotImplementedError(index.dtype)
        else:
            result = index_type(index.values, copy=False, **init_kwargs)
            tm.assert_numpy_array_equal(index.values, result.values, check_same="same")

    def test_memory_usage(self, index):
        index._engine.clear_mapping()
        result = index.memory_usage()
        if index.empty:
            # we report 0 for no-length
            assert result == 0
            return

        # non-zero length
        index.get_loc(index[0])
        result2 = index.memory_usage()
        result3 = index.memory_usage(deep=True)

        # RangeIndex, IntervalIndex
        # don't have engines
        # Index[EA] has engine but it does not have a Hashtable .mapping
        if not isinstance(index, (RangeIndex, IntervalIndex)) and not (
            type(index) is Index and not isinstance(index.dtype, np.dtype)
        ):
            assert result2 > result

        if index.inferred_type == "object":
            assert result3 > result2

    def test_argsort(self, index):
        if isinstance(index, CategoricalIndex):
            pytest.skip(f"{type(self).__name__} separately tested")

        result = index.argsort()
        expected = np.array(index).argsort()
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    def test_numpy_argsort(self, index):
        result = np.argsort(index)
        expected = index.argsort()
        tm.assert_numpy_array_equal(result, expected)

        result = np.argsort(index, kind="mergesort")
        expected = index.argsort(kind="mergesort")
        tm.assert_numpy_array_equal(result, expected)

        # these are the only two types that perform
        # pandas compatibility input validation - the
        # rest already perform separate (or no) such
        # validation via their 'values' attribute as
        # defined in pandas.core.indexes/base.py - they
        # cannot be changed at the moment due to
        # backwards compatibility concerns
        if isinstance(index, (CategoricalIndex, RangeIndex)):
            msg = "the 'axis' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.argsort(index, axis=1)

            msg = "the 'order' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.argsort(index, order=("a", "b"))

    def test_repeat(self, simple_index):
        rep = 2
        idx = simple_index.copy()
        new_index_cls = idx._constructor
        expected = new_index_cls(idx.values.repeat(rep), name=idx.name)
        tm.assert_index_equal(idx.repeat(rep), expected)

        idx = simple_index
        rep = np.arange(len(idx))
        expected = new_index_cls(idx.values.repeat(rep), name=idx.name)
        tm.assert_index_equal(idx.repeat(rep), expected)

    def test_numpy_repeat(self, simple_index):
        rep = 2
        idx = simple_index
        expected = idx.repeat(rep)
        tm.assert_index_equal(np.repeat(idx, rep), expected)

        msg = "the 'axis' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.repeat(idx, rep, axis=0)

    def test_where(self, listlike_box, simple_index):
        if isinstance(simple_index, (IntervalIndex, PeriodIndex)) or is_numeric_dtype(
            simple_index.dtype
        ):
            pytest.skip("Tested elsewhere.")
        klass = listlike_box

        idx = simple_index
        if isinstance(idx, (DatetimeIndex, TimedeltaIndex)):
            # where does not preserve freq
            idx = idx._with_freq(None)

        cond = [True] * len(idx)
        result = idx.where(klass(cond))
        expected = idx
        tm.assert_index_equal(result, expected)

        cond = [False] + [True] * len(idx[1:])
        expected = Index([idx._na_value] + idx[1:].tolist(), dtype=idx.dtype)
        result = idx.where(klass(cond))
        tm.assert_index_equal(result, expected)

    def test_insert_base(self, index):
        trimmed = index[1:4]

        if not len(index):
            pytest.skip("Not applicable for empty index")

        # test 0th element
        warn = None
        if index.dtype == object and index.inferred_type == "boolean":
            # GH#51363
            warn = FutureWarning
        msg = "The behavior of Index.insert with object-dtype is deprecated"
        with tm.assert_produces_warning(warn, match=msg):
            result = trimmed.insert(0, index[0])
        assert index[0:4].equals(result)

    def test_insert_out_of_bounds(self, index, using_infer_string):
        # TypeError/IndexError matches what np.insert raises in these cases

        if len(index) > 0:
            err = TypeError
        else:
            err = IndexError
        if len(index) == 0:
            # 0 vs 0.5 in error message varies with numpy version
            msg = "index (0|0.5) is out of bounds for axis 0 with size 0"
        else:
            msg = "slice indices must be integers or None or have an __index__ method"

        if using_infer_string and (
            index.dtype == "string" or index.dtype == "category"  # noqa: PLR1714
        ):
            msg = "loc must be an integer between"

        with pytest.raises(err, match=msg):
            index.insert(0.5, "foo")

        msg = "|".join(
            [
                r"index -?\d+ is out of bounds for axis 0 with size \d+",
                "loc must be an integer between",
            ]
        )
        with pytest.raises(IndexError, match=msg):
            index.insert(len(index) + 1, 1)

        with pytest.raises(IndexError, match=msg):
            index.insert(-len(index) - 1, 1)

    def test_delete_base(self, index):
        if not len(index):
            pytest.skip("Not applicable for empty index")

        if isinstance(index, RangeIndex):
            # tested in class
            pytest.skip(f"{type(self).__name__} tested elsewhere")

        expected = index[1:]
        result = index.delete(0)
        assert result.equals(expected)
        assert result.name == expected.name

        expected = index[:-1]
        result = index.delete(-1)
        assert result.equals(expected)
        assert result.name == expected.name

        length = len(index)
        msg = f"index {length} is out of bounds for axis 0 with size {length}"
        with pytest.raises(IndexError, match=msg):
            index.delete(length)

    @pytest.mark.filterwarnings(r"ignore:Dtype inference:FutureWarning")
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_equals(self, index):
        if isinstance(index, IntervalIndex):
            pytest.skip(f"{type(index).__name__} tested elsewhere")

        is_ea_idx = type(index) is Index and not isinstance(index.dtype, np.dtype)

        assert index.equals(index)
        assert index.equals(index.copy())
        if not is_ea_idx:
            # doesn't hold for e.g. IntegerDtype
            assert index.equals(index.astype(object))

        assert not index.equals(list(index))
        assert not index.equals(np.array(index))

        # Cannot pass in non-int64 dtype to RangeIndex
        if not isinstance(index, RangeIndex) and not is_ea_idx:
            same_values = Index(index, dtype=object)
            assert index.equals(same_values)
            assert same_values.equals(index)

        if index.nlevels == 1:
            # do not test MultiIndex
            assert not index.equals(Series(index))

    def test_equals_op(self, simple_index):
        # GH9947, GH10637
        index_a = simple_index

        n = len(index_a)
        index_b = index_a[0:-1]
        index_c = index_a[0:-1].append(index_a[-2:-1])
        index_d = index_a[0:1]

        msg = "Lengths must match|could not be broadcast"
        with pytest.raises(ValueError, match=msg):
            index_a == index_b
        expected1 = np.array([True] * n)
        expected2 = np.array([True] * (n - 1) + [False])
        tm.assert_numpy_array_equal(index_a == index_a, expected1)
        tm.assert_numpy_array_equal(index_a == index_c, expected2)

        # test comparisons with numpy arrays
        array_a = np.array(index_a)
        array_b = np.array(index_a[0:-1])
        array_c = np.array(index_a[0:-1].append(index_a[-2:-1]))
        array_d = np.array(index_a[0:1])
        with pytest.raises(ValueError, match=msg):
            index_a == array_b
        tm.assert_numpy_array_equal(index_a == array_a, expected1)
        tm.assert_numpy_array_equal(index_a == array_c, expected2)

        # test comparisons with Series
        series_a = Series(array_a)
        series_b = Series(array_b)
        series_c = Series(array_c)
        series_d = Series(array_d)
        with pytest.raises(ValueError, match=msg):
            index_a == series_b

        tm.assert_numpy_array_equal(index_a == series_a, expected1)
        tm.assert_numpy_array_equal(index_a == series_c, expected2)

        # cases where length is 1 for one of them
        with pytest.raises(ValueError, match="Lengths must match"):
            index_a == index_d
        with pytest.raises(ValueError, match="Lengths must match"):
            index_a == series_d
        with pytest.raises(ValueError, match="Lengths must match"):
            index_a == array_d
        msg = "Can only compare identically-labeled Series objects"
        with pytest.raises(ValueError, match=msg):
            series_a == series_d
        with pytest.raises(ValueError, match="Lengths must match"):
            series_a == array_d

        # comparing with a scalar should broadcast; note that we are excluding
        # MultiIndex because in this case each item in the index is a tuple of
        # length 2, and therefore is considered an array of length 2 in the
        # comparison instead of a scalar
        if not isinstance(index_a, MultiIndex):
            expected3 = np.array([False] * (len(index_a) - 2) + [True, False])
            # assuming the 2nd to last item is unique in the data
            item = index_a[-2]
            tm.assert_numpy_array_equal(index_a == item, expected3)
            tm.assert_series_equal(series_a == item, Series(expected3))

    def test_format(self, simple_index):
        # GH35439
        if is_numeric_dtype(simple_index.dtype) or isinstance(
            simple_index, DatetimeIndex
        ):
            pytest.skip("Tested elsewhere.")
        idx = simple_index
        expected = [str(x) for x in idx]
        msg = r"Index\.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert idx.format() == expected

    def test_format_empty(self, simple_index):
        # GH35712
        if isinstance(simple_index, (PeriodIndex, RangeIndex)):
            pytest.skip("Tested elsewhere")
        empty_idx = type(simple_index)([])
        msg = r"Index\.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert empty_idx.format() == []
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert empty_idx.format(name=True) == [""]

    def test_fillna(self, index):
        # GH 11343
        if len(index) == 0:
            pytest.skip("Not relevant for empty index")
        elif index.dtype == bool:
            pytest.skip(f"{index.dtype} cannot hold NAs")
        elif isinstance(index, Index) and is_integer_dtype(index.dtype):
            pytest.skip(f"Not relevant for Index with {index.dtype}")
        elif isinstance(index, MultiIndex):
            idx = index.copy(deep=True)
            msg = "isna is not defined for MultiIndex"
            with pytest.raises(NotImplementedError, match=msg):
                idx.fillna(idx[0])
        else:
            idx = index.copy(deep=True)
            result = idx.fillna(idx[0])
            tm.assert_index_equal(result, idx)
            assert result is not idx

            msg = "'value' must be a scalar, passed: "
            with pytest.raises(TypeError, match=msg):
                idx.fillna([idx[0]])

            idx = index.copy(deep=True)
            values = idx._values

            values[1] = np.nan

            idx = type(index)(values)

            msg = "does not support 'downcast'"
            msg2 = r"The 'downcast' keyword in .*Index\.fillna is deprecated"
            with tm.assert_produces_warning(FutureWarning, match=msg2):
                with pytest.raises(NotImplementedError, match=msg):
                    # For now at least, we only raise if there are NAs present
                    idx.fillna(idx[0], downcast="infer")

            expected = np.array([False] * len(idx), dtype=bool)
            expected[1] = True
            tm.assert_numpy_array_equal(idx._isnan, expected)
            assert idx.hasnans is True

    def test_nulls(self, index):
        # this is really a smoke test for the methods
        # as these are adequately tested for function elsewhere
        if len(index) == 0:
            tm.assert_numpy_array_equal(index.isna(), np.array([], dtype=bool))
        elif isinstance(index, MultiIndex):
            idx = index.copy()
            msg = "isna is not defined for MultiIndex"
            with pytest.raises(NotImplementedError, match=msg):
                idx.isna()
        elif not index.hasnans:
            tm.assert_numpy_array_equal(index.isna(), np.zeros(len(index), dtype=bool))
            tm.assert_numpy_array_equal(index.notna(), np.ones(len(index), dtype=bool))
        else:
            result = isna(index)
            tm.assert_numpy_array_equal(index.isna(), result)
            tm.assert_numpy_array_equal(index.notna(), ~result)

    def test_empty(self, simple_index):
        # GH 15270
        idx = simple_index
        assert not idx.empty
        assert idx[:0].empty

    def test_join_self_unique(self, join_type, simple_index):
        idx = simple_index
        if idx.is_unique:
            joined = idx.join(idx, how=join_type)
            expected = simple_index
            if join_type == "outer":
                expected = algos.safe_sort(expected)
            tm.assert_index_equal(joined, expected)

    def test_map(self, simple_index):
        # callable
        if isinstance(simple_index, (TimedeltaIndex, PeriodIndex)):
            pytest.skip("Tested elsewhere.")
        idx = simple_index

        result = idx.map(lambda x: x)
        # RangeIndex are equivalent to the similar Index with int64 dtype
        tm.assert_index_equal(result, idx, exact="equiv")

    @pytest.mark.parametrize(
        "mapper",
        [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: Series(values, index),
        ],
    )
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_map_dictlike(self, mapper, simple_index, request):
        idx = simple_index
        if isinstance(idx, (DatetimeIndex, TimedeltaIndex, PeriodIndex)):
            pytest.skip("Tested elsewhere.")

        identity = mapper(idx.values, idx)

        result = idx.map(identity)
        # RangeIndex are equivalent to the similar Index with int64 dtype
        tm.assert_index_equal(result, idx, exact="equiv")

        # empty mappable
        dtype = None
        if idx.dtype.kind == "f":
            dtype = idx.dtype

        expected = Index([np.nan] * len(idx), dtype=dtype)
        result = idx.map(mapper(expected, idx))
        tm.assert_index_equal(result, expected)

    def test_map_str(self, simple_index):
        # GH 31202
        if isinstance(simple_index, CategoricalIndex):
            pytest.skip("See test_map.py")
        idx = simple_index
        result = idx.map(str)
        expected = Index([str(x) for x in idx])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("copy", [True, False])
    @pytest.mark.parametrize("name", [None, "foo"])
    @pytest.mark.parametrize("ordered", [True, False])
    def test_astype_category(self, copy, name, ordered, simple_index):
        # GH 18630
        idx = simple_index
        if name:
            idx = idx.rename(name)

        # standard categories
        dtype = CategoricalDtype(ordered=ordered)
        result = idx.astype(dtype, copy=copy)
        expected = CategoricalIndex(idx, name=name, ordered=ordered)
        tm.assert_index_equal(result, expected, exact=True)

        # non-standard categories
        dtype = CategoricalDtype(idx.unique().tolist()[:-1], ordered)
        result = idx.astype(dtype, copy=copy)
        expected = CategoricalIndex(idx, name=name, dtype=dtype)
        tm.assert_index_equal(result, expected, exact=True)

        if ordered is False:
            # dtype='category' defaults to ordered=False, so only test once
            result = idx.astype("category", copy=copy)
            expected = CategoricalIndex(idx, name=name)
            tm.assert_index_equal(result, expected, exact=True)

    def test_is_unique(self, simple_index):
        # initialize a unique index
        index = simple_index.drop_duplicates()
        assert index.is_unique is True

        # empty index should be unique
        index_empty = index[:0]
        assert index_empty.is_unique is True

        # test basic dupes
        index_dup = index.insert(0, index[0])
        assert index_dup.is_unique is False

        # single NA should be unique
        index_na = index.insert(0, np.nan)
        assert index_na.is_unique is True

        # multiple NA should not be unique
        index_na_dup = index_na.insert(0, np.nan)
        assert index_na_dup.is_unique is False

    @pytest.mark.arm_slow
    def test_engine_reference_cycle(self, simple_index):
        # GH27585
        index = simple_index.copy()
        ref = weakref.ref(index)
        index._engine
        del index
        assert ref() is None

    def test_getitem_2d_deprecated(self, simple_index):
        # GH#30588, GH#31479
        if isinstance(simple_index, IntervalIndex):
            pytest.skip("Tested elsewhere")
        idx = simple_index
        msg = "Multi-dimensional indexing|too many|only"
        with pytest.raises((ValueError, IndexError), match=msg):
            idx[:, None]

        if not isinstance(idx, RangeIndex):
            # GH#44051 RangeIndex already raised pre-2.0 with a different message
            with pytest.raises((ValueError, IndexError), match=msg):
                idx[True]
            with pytest.raises((ValueError, IndexError), match=msg):
                idx[False]
        else:
            msg = "only integers, slices"
            with pytest.raises(IndexError, match=msg):
                idx[True]
            with pytest.raises(IndexError, match=msg):
                idx[False]

    def test_copy_shares_cache(self, simple_index):
        # GH32898, GH36840
        idx = simple_index
        idx.get_loc(idx[0])  # populates the _cache.
        copy = idx.copy()

        assert copy._cache is idx._cache

    def test_shallow_copy_shares_cache(self, simple_index):
        # GH32669, GH36840
        idx = simple_index
        idx.get_loc(idx[0])  # populates the _cache.
        shallow_copy = idx._view()

        assert shallow_copy._cache is idx._cache

        shallow_copy = idx._shallow_copy(idx._data)
        assert shallow_copy._cache is not idx._cache
        assert shallow_copy._cache == {}

    def test_index_groupby(self, simple_index):
        idx = simple_index[:5]
        to_groupby = np.array([1, 2, np.nan, 2, 1])
        tm.assert_dict_equal(
            idx.groupby(to_groupby), {1.0: idx[[0, 4]], 2.0: idx[[1, 3]]}
        )

        to_groupby = DatetimeIndex(
            [
                datetime(2011, 11, 1),
                datetime(2011, 12, 1),
                pd.NaT,
                datetime(2011, 12, 1),
                datetime(2011, 11, 1),
            ],
            tz="UTC",
        ).values

        ex_keys = [Timestamp("2011-11-01"), Timestamp("2011-12-01")]
        expected = {ex_keys[0]: idx[[0, 4]], ex_keys[1]: idx[[1, 3]]}
        tm.assert_dict_equal(idx.groupby(to_groupby), expected)

    def test_append_preserves_dtype(self, simple_index):
        # In particular Index with dtype float32
        index = simple_index
        N = len(index)

        result = index.append(index)
        assert result.dtype == index.dtype
        tm.assert_index_equal(result[:N], index, check_exact=True)
        tm.assert_index_equal(result[N:], index, check_exact=True)

        alt = index.take(list(range(N)) * 2)
        tm.assert_index_equal(result, alt, check_exact=True)

    def test_inv(self, simple_index, using_infer_string):
        idx = simple_index

        if idx.dtype.kind in ["i", "u"]:
            res = ~idx
            expected = Index(~idx.values, name=idx.name)
            tm.assert_index_equal(res, expected)

            # check that we are matching Series behavior
            res2 = ~Series(idx)
            tm.assert_series_equal(res2, Series(expected))
        else:
            if idx.dtype.kind == "f":
                msg = "ufunc 'invert' not supported for the input types"
            else:
                msg = "bad operand|__invert__ is not supported for string dtype"
            with pytest.raises(TypeError, match=msg):
                ~idx

            # check that we get the same behavior with Series
            with pytest.raises(TypeError, match=msg):
                ~Series(idx)

    def test_is_boolean_is_deprecated(self, simple_index):
        # GH50042
        idx = simple_index
        with tm.assert_produces_warning(FutureWarning):
            idx.is_boolean()

    def test_is_floating_is_deprecated(self, simple_index):
        # GH50042
        idx = simple_index
        with tm.assert_produces_warning(FutureWarning):
            idx.is_floating()

    def test_is_integer_is_deprecated(self, simple_index):
        # GH50042
        idx = simple_index
        with tm.assert_produces_warning(FutureWarning):
            idx.is_integer()

    def test_holds_integer_deprecated(self, simple_index):
        # GH50243
        idx = simple_index
        msg = f"{type(idx).__name__}.holds_integer is deprecated. "
        with tm.assert_produces_warning(FutureWarning, match=msg):
            idx.holds_integer()

    def test_is_numeric_is_deprecated(self, simple_index):
        # GH50042
        idx = simple_index
        with tm.assert_produces_warning(
            FutureWarning,
            match=f"{type(idx).__name__}.is_numeric is deprecated. ",
        ):
            idx.is_numeric()

    def test_is_categorical_is_deprecated(self, simple_index):
        # GH50042
        idx = simple_index
        with tm.assert_produces_warning(
            FutureWarning,
            match=r"Use pandas\.api\.types\.is_categorical_dtype instead",
        ):
            idx.is_categorical()

    def test_is_interval_is_deprecated(self, simple_index):
        # GH50042
        idx = simple_index
        with tm.assert_produces_warning(FutureWarning):
            idx.is_interval()

    def test_is_object_is_deprecated(self, simple_index):
        # GH50042
        idx = simple_index
        with tm.assert_produces_warning(FutureWarning):
            idx.is_object()


class TestNumericBase:
    @pytest.fixture(
        params=[
            RangeIndex(start=0, stop=20, step=2),
            Index(np.arange(5, dtype=np.float64)),
            Index(np.arange(5, dtype=np.float32)),
            Index(np.arange(5, dtype=np.uint64)),
            Index(range(0, 20, 2), dtype=np.int64),
            Index(range(0, 20, 2), dtype=np.int32),
            Index(range(0, 20, 2), dtype=np.int16),
            Index(range(0, 20, 2), dtype=np.int8),
        ]
    )
    def simple_index(self, request):
        return request.param

    def test_constructor_unwraps_index(self, simple_index):
        if isinstance(simple_index, RangeIndex):
            pytest.skip("Tested elsewhere.")
        index_cls = type(simple_index)
        dtype = simple_index.dtype

        idx = Index([1, 2], dtype=dtype)
        result = index_cls(idx)
        expected = np.array([1, 2], dtype=idx.dtype)
        tm.assert_numpy_array_equal(result._data, expected)

    def test_can_hold_identifiers(self, simple_index):
        idx = simple_index
        key = idx[0]
        assert idx._can_hold_identifiers_and_holds_name(key) is False

    def test_view(self, simple_index):
        if isinstance(simple_index, RangeIndex):
            pytest.skip("Tested elsewhere.")
        index_cls = type(simple_index)
        dtype = simple_index.dtype

        idx = index_cls([], dtype=dtype, name="Foo")
        idx_view = idx.view()
        assert idx_view.name == "Foo"

        idx_view = idx.view(dtype)
        tm.assert_index_equal(idx, index_cls(idx_view, name="Foo"), exact=True)

        msg = "Passing a type in .*Index.view is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            idx_view = idx.view(index_cls)
        tm.assert_index_equal(idx, index_cls(idx_view, name="Foo"), exact=True)

    def test_format(self, simple_index):
        # GH35439
        if isinstance(simple_index, DatetimeIndex):
            pytest.skip("Tested elsewhere")
        idx = simple_index
        max_width = max(len(str(x)) for x in idx)
        expected = [str(x).ljust(max_width) for x in idx]
        msg = r"Index\.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert idx.format() == expected

    def test_insert_non_na(self, simple_index):
        # GH#43921 inserting an element that we know we can hold should
        #  not change dtype or type (except for RangeIndex)
        index = simple_index

        result = index.insert(0, index[0])

        expected = Index([index[0]] + list(index), dtype=index.dtype)
        tm.assert_index_equal(result, expected, exact=True)

    def test_insert_na(self, nulls_fixture, simple_index):
        # GH 18295 (test missing)
        index = simple_index
        na_val = nulls_fixture

        if na_val is pd.NaT:
            expected = Index([index[0], pd.NaT] + list(index[1:]), dtype=object)
        else:
            expected = Index([index[0], np.nan] + list(index[1:]))
            # GH#43921 we preserve float dtype
            if index.dtype.kind == "f":
                expected = Index(expected, dtype=index.dtype)

        result = index.insert(1, na_val)
        tm.assert_index_equal(result, expected, exact=True)

    def test_arithmetic_explicit_conversions(self, simple_index):
        # GH 8608
        # add/sub are overridden explicitly for Float/Int Index
        index_cls = type(simple_index)
        if index_cls is RangeIndex:
            idx = RangeIndex(5)
        else:
            idx = index_cls(np.arange(5, dtype="int64"))

        # float conversions
        arr = np.arange(5, dtype="int64") * 3.2
        expected = Index(arr, dtype=np.float64)
        fidx = idx * 3.2
        tm.assert_index_equal(fidx, expected)
        fidx = 3.2 * idx
        tm.assert_index_equal(fidx, expected)

        # interops with numpy arrays
        expected = Index(arr, dtype=np.float64)
        a = np.zeros(5, dtype="float64")
        result = fidx - a
        tm.assert_index_equal(result, expected)

        expected = Index(-arr, dtype=np.float64)
        a = np.zeros(5, dtype="float64")
        result = a - fidx
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("complex_dtype", [np.complex64, np.complex128])
    def test_astype_to_complex(self, complex_dtype, simple_index):
        result = simple_index.astype(complex_dtype)

        assert type(result) is Index and result.dtype == complex_dtype

    def test_cast_string(self, simple_index):
        if isinstance(simple_index, RangeIndex):
            pytest.skip("casting of strings not relevant for RangeIndex")
        result = type(simple_index)(["0", "1", "2"], dtype=simple_index.dtype)
        expected = type(simple_index)([0, 1, 2], dtype=simple_index.dtype)
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\functions\signals.py ===
from .functions import defun_wrapped

@defun_wrapped
def squarew(ctx, t, amplitude=1, period=1):
    P = period
    A = amplitude
    return A*((-1)**ctx.floor(2*t/P))

@defun_wrapped
def trianglew(ctx, t, amplitude=1, period=1):
    A = amplitude
    P = period

    return 2*A*(0.5 - ctx.fabs(1 - 2*ctx.frac(t/P + 0.25)))

@defun_wrapped
def sawtoothw(ctx, t, amplitude=1, period=1):
    A = amplitude
    P = period
    return A*ctx.frac(t/P)

@defun_wrapped
def unit_triangle(ctx, t, amplitude=1):
    A = amplitude
    if t <= -1 or t >= 1:
        return ctx.zero
    return A*(-ctx.fabs(t) + 1)

@defun_wrapped
def sigmoid(ctx, t, amplitude=1):
    A = amplitude
    return A / (1 + ctx.exp(-t))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\printoptions.py ===
"""
Stores and defines the low-level format_options context variable.

This is defined in its own file outside of the arrayprint module
so we can import it from C while initializing the multiarray
C module during import without introducing circular dependencies.
"""

import sys
from contextvars import ContextVar

__all__ = ["format_options"]

default_format_options_dict = {
    "edgeitems": 3,  # repr N leading and trailing items of each dimension
    "threshold": 1000,  # total items > triggers array summarization
    "floatmode": "maxprec",
    "precision": 8,  # precision of floating point representations
    "suppress": False,  # suppress printing small floating values in exp format
    "linewidth": 75,
    "nanstr": "nan",
    "infstr": "inf",
    "sign": "-",
    "formatter": None,
    # Internally stored as an int to simplify comparisons; converted from/to
    # str/False on the way in/out.
    'legacy': sys.maxsize,
    'override_repr': None,
}

format_options = ContextVar(
    "format_options", default=default_format_options_dict)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\EdgeEnd.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class EdgeEnd(object):
    __slots__ = ['_tab']

    @classmethod
    def SizeOf(cls):
        return 12

    # EdgeEnd
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # EdgeEnd
    def NodeIndex(self): return self._tab.Get(flatbuffers.number_types.Uint32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(0))
    # EdgeEnd
    def SrcArgIndex(self): return self._tab.Get(flatbuffers.number_types.Int32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(4))
    # EdgeEnd
    def DstArgIndex(self): return self._tab.Get(flatbuffers.number_types.Int32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(8))

def CreateEdgeEnd(builder, nodeIndex, srcArgIndex, dstArgIndex):
    builder.Prep(4, 12)
    builder.PrependInt32(dstArgIndex)
    builder.PrependInt32(srcArgIndex)
    builder.PrependUint32(nodeIndex)
    return builder.Offset()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\onnx_model_conformer.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

from fusion_attention import AttentionMask
from fusion_conformer_attention import FusionConformerAttention
from fusion_options import FusionOptions
from onnx_model_bert import BertOnnxModel

logger = logging.getLogger(__name__)


class ConformerOnnxModel(BertOnnxModel):
    def __init__(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)
        self.attention_mask = AttentionMask(self)
        self.attention_fusion = FusionConformerAttention(self, self.hidden_size, self.num_heads, self.attention_mask)

    def optimize(self, options: FusionOptions | None = None, add_dynamic_axes: bool = False):
        self.attention_fusion.use_multi_head_attention = False if options is None else options.use_multi_head_attention
        self.attention_fusion.disable_multi_head_attention_bias = (
            False if options is None else options.disable_multi_head_attention_bias
        )
        super().optimize(options, add_dynamic_axes)

    def fuse_attention(self):
        self.attention_fusion.apply()

    def preprocess(self):
        self.adjust_reshape_and_expand()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\reader.py ===
# Copyright (c) 2010-2024 openpyxl

"""
Read a chart
"""

def read_chart(chartspace):
    cs = chartspace
    plot = cs.chart.plotArea

    chart = plot._charts[0]
    chart._charts = plot._charts

    chart.title = cs.chart.title
    chart.display_blanks = cs.chart.dispBlanksAs
    chart.visible_cells_only = cs.chart.plotVisOnly
    chart.layout = plot.layout
    chart.legend = cs.chart.legend

    # 3d attributes
    chart.floor = cs.chart.floor
    chart.sideWall = cs.chart.sideWall
    chart.backWall = cs.chart.backWall
    chart.pivotSource = cs.pivotSource
    chart.pivotFormats = cs.chart.pivotFmts
    chart.idx_base = min((s.idx for s in chart.series), default=0)
    chart._reindex()

    # Border, fill, etc.
    chart.graphical_properties = cs.graphical_properties

    return chart

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\_emoji_replace.py ===
from typing import Callable, Match, Optional
import re

from ._emoji_codes import EMOJI


_ReStringMatch = Match[str]  # regex match object
_ReSubCallable = Callable[[_ReStringMatch], str]  # Callable invoked by re.sub
_EmojiSubMethod = Callable[[_ReSubCallable, str], str]  # Sub method of a compiled re


def _emoji_replace(
    text: str,
    default_variant: Optional[str] = None,
    _emoji_sub: _EmojiSubMethod = re.compile(r"(:(\S*?)(?:(?:\-)(emoji|text))?:)").sub,
) -> str:
    """Replace emoji code in text."""
    get_emoji = EMOJI.__getitem__
    variants = {"text": "\uFE0E", "emoji": "\uFE0F"}
    get_variant = variants.get
    default_variant_code = variants.get(default_variant, "") if default_variant else ""

    def do_replace(match: Match[str]) -> str:
        emoji_code, emoji_name, variant = match.groups()
        try:
            return get_emoji(emoji_name.lower()) + get_variant(
                variant, default_variant_code
            )
        except KeyError:
            return emoji_code

    return _emoji_sub(do_replace, text)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\webdriver_manager\core\logger.py ===
import logging

from webdriver_manager.core.config import wdm_log_level

__logger = logging.getLogger("WDM")
__logger.addHandler(logging.NullHandler())


def log(text):
    """Emitting the log message."""
    __logger.log(wdm_log_level(), text)


def set_logger(logger):
    """
    Set the global logger.

    Parameters
    ----------
    logger : logging.Logger
        The custom logger to use.

    Returns None
    """

    # Check if the logger is a valid logger
    if not isinstance(logger, logging.Logger):
        raise ValueError("The logger must be an instance of logging.Logger")

    # Bind the logger input to the global logger
    global __logger
    __logger = logger

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\test_select.py ===
import numpy as np
import pytest

from pandas._libs.tslibs import Timestamp

import pandas as pd
from pandas import (
    DataFrame,
    HDFStore,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
    bdate_range,
    concat,
    date_range,
    isna,
    read_hdf,
)
from pandas.tests.io.pytables.common import (
    _maybe_remove,
    ensure_clean_store,
)

from pandas.io.pytables import Term

pytestmark = [pytest.mark.single_cpu]


def test_select_columns_in_where(setup_path):
    # GH 6169
    # recreate multi-indexes when columns is passed
    # in the `where` argument
    index = MultiIndex(
        levels=[["foo", "bar", "baz", "qux"], ["one", "two", "three"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["foo_name", "bar_name"],
    )

    # With a DataFrame
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 3)),
        index=index,
        columns=["A", "B", "C"],
    )

    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table")
        expected = df[["A"]]

        tm.assert_frame_equal(store.select("df", columns=["A"]), expected)

        tm.assert_frame_equal(store.select("df", where="columns=['A']"), expected)

    # With a Series
    s = Series(np.random.default_rng(2).standard_normal(10), index=index, name="A")
    with ensure_clean_store(setup_path) as store:
        store.put("s", s, format="table")
        tm.assert_series_equal(store.select("s", where="columns=['A']"), s)


def test_select_with_dups(setup_path):
    # single dtypes
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)), columns=["A", "A", "B", "B"]
    )
    df.index = date_range("20130101 9:30", periods=10, freq="min")

    with ensure_clean_store(setup_path) as store:
        store.append("df", df)

        result = store.select("df")
        expected = df
        tm.assert_frame_equal(result, expected, by_blocks=True)

        result = store.select("df", columns=df.columns)
        expected = df
        tm.assert_frame_equal(result, expected, by_blocks=True)

        result = store.select("df", columns=["A"])
        expected = df.loc[:, ["A"]]
        tm.assert_frame_equal(result, expected)

    # dups across dtypes
    df = concat(
        [
            DataFrame(
                np.random.default_rng(2).standard_normal((10, 4)),
                columns=["A", "A", "B", "B"],
            ),
            DataFrame(
                np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2),
                columns=["A", "C"],
            ),
        ],
        axis=1,
    )
    df.index = date_range("20130101 9:30", periods=10, freq="min")

    with ensure_clean_store(setup_path) as store:
        store.append("df", df)

        result = store.select("df")
        expected = df
        tm.assert_frame_equal(result, expected, by_blocks=True)

        result = store.select("df", columns=df.columns)
        expected = df
        tm.assert_frame_equal(result, expected, by_blocks=True)

        expected = df.loc[:, ["A"]]
        result = store.select("df", columns=["A"])
        tm.assert_frame_equal(result, expected, by_blocks=True)

        expected = df.loc[:, ["B", "A"]]
        result = store.select("df", columns=["B", "A"])
        tm.assert_frame_equal(result, expected, by_blocks=True)

    # duplicates on both index and columns
    with ensure_clean_store(setup_path) as store:
        store.append("df", df)
        store.append("df", df)

        expected = df.loc[:, ["B", "A"]]
        expected = concat([expected, expected])
        result = store.select("df", columns=["B", "A"])
        tm.assert_frame_equal(result, expected, by_blocks=True)


def test_select(setup_path):
    with ensure_clean_store(setup_path) as store:
        # select with columns=
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        _maybe_remove(store, "df")
        store.append("df", df)
        result = store.select("df", columns=["A", "B"])
        expected = df.reindex(columns=["A", "B"])
        tm.assert_frame_equal(expected, result)

        # equivalently
        result = store.select("df", [("columns=['A', 'B']")])
        expected = df.reindex(columns=["A", "B"])
        tm.assert_frame_equal(expected, result)

        # with a data column
        _maybe_remove(store, "df")
        store.append("df", df, data_columns=["A"])
        result = store.select("df", ["A > 0"], columns=["A", "B"])
        expected = df[df.A > 0].reindex(columns=["A", "B"])
        tm.assert_frame_equal(expected, result)

        # all a data columns
        _maybe_remove(store, "df")
        store.append("df", df, data_columns=True)
        result = store.select("df", ["A > 0"], columns=["A", "B"])
        expected = df[df.A > 0].reindex(columns=["A", "B"])
        tm.assert_frame_equal(expected, result)

        # with a data column, but different columns
        _maybe_remove(store, "df")
        store.append("df", df, data_columns=["A"])
        result = store.select("df", ["A > 0"], columns=["C", "D"])
        expected = df[df.A > 0].reindex(columns=["C", "D"])
        tm.assert_frame_equal(expected, result)


def test_select_dtypes(setup_path):
    with ensure_clean_store(setup_path) as store:
        # with a Timestamp data column (GH #2637)
        df = DataFrame(
            {
                "ts": bdate_range("2012-01-01", periods=300),
                "A": np.random.default_rng(2).standard_normal(300),
            }
        )
        _maybe_remove(store, "df")
        store.append("df", df, data_columns=["ts", "A"])

        result = store.select("df", "ts>=Timestamp('2012-02-01')")
        expected = df[df.ts >= Timestamp("2012-02-01")]
        tm.assert_frame_equal(expected, result)

        # bool columns (GH #2849)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=["A", "B"]
        )
        df["object"] = "foo"
        df.loc[4:5, "object"] = "bar"
        df["boolv"] = df["A"] > 0
        _maybe_remove(store, "df")
        store.append("df", df, data_columns=True)

        expected = df[df.boolv == True].reindex(columns=["A", "boolv"])  # noqa: E712
        for v in [True, "true", 1]:
            result = store.select("df", f"boolv == {v}", columns=["A", "boolv"])
            tm.assert_frame_equal(expected, result)

        expected = df[df.boolv == False].reindex(columns=["A", "boolv"])  # noqa: E712
        for v in [False, "false", 0]:
            result = store.select("df", f"boolv == {v}", columns=["A", "boolv"])
            tm.assert_frame_equal(expected, result)

        # integer index
        df = DataFrame(
            {
                "A": np.random.default_rng(2).random(20),
                "B": np.random.default_rng(2).random(20),
            }
        )
        _maybe_remove(store, "df_int")
        store.append("df_int", df)
        result = store.select("df_int", "index<10 and columns=['A']")
        expected = df.reindex(index=list(df.index)[0:10], columns=["A"])
        tm.assert_frame_equal(expected, result)

        # float index
        df = DataFrame(
            {
                "A": np.random.default_rng(2).random(20),
                "B": np.random.default_rng(2).random(20),
                "index": np.arange(20, dtype="f8"),
            }
        )
        _maybe_remove(store, "df_float")
        store.append("df_float", df)
        result = store.select("df_float", "index<10.0 and columns=['A']")
        expected = df.reindex(index=list(df.index)[0:10], columns=["A"])
        tm.assert_frame_equal(expected, result)

    with ensure_clean_store(setup_path) as store:
        # floats w/o NaN
        df = DataFrame({"cols": range(11), "values": range(11)}, dtype="float64")
        df["cols"] = (df["cols"] + 10).apply(str)

        store.append("df1", df, data_columns=True)
        result = store.select("df1", where="values>2.0")
        expected = df[df["values"] > 2.0]
        tm.assert_frame_equal(expected, result)

        # floats with NaN
        df.iloc[0] = np.nan
        expected = df[df["values"] > 2.0]

        store.append("df2", df, data_columns=True, index=False)
        result = store.select("df2", where="values>2.0")
        tm.assert_frame_equal(expected, result)

        # https://github.com/PyTables/PyTables/issues/282
        # bug in selection when 0th row has a np.nan and an index
        # store.append('df3',df,data_columns=True)
        # result = store.select(
        #    'df3', where='values>2.0')
        # tm.assert_frame_equal(expected, result)

        # not in first position float with NaN ok too
        df = DataFrame({"cols": range(11), "values": range(11)}, dtype="float64")
        df["cols"] = (df["cols"] + 10).apply(str)

        df.iloc[1] = np.nan
        expected = df[df["values"] > 2.0]

        store.append("df4", df, data_columns=True)
        result = store.select("df4", where="values>2.0")
        tm.assert_frame_equal(expected, result)

    # test selection with comparison against numpy scalar
    # GH 11283
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD")),
            index=Index([f"i-{i}" for i in range(30)]),
        )

        expected = df[df["A"] > 0]

        store.append("df", df, data_columns=True)
        np_zero = np.float64(0)  # noqa: F841
        result = store.select("df", where=["A>np_zero"])
        tm.assert_frame_equal(expected, result)


def test_select_with_many_inputs(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(
            {
                "ts": bdate_range("2012-01-01", periods=300),
                "A": np.random.default_rng(2).standard_normal(300),
                "B": range(300),
                "users": ["a"] * 50
                + ["b"] * 50
                + ["c"] * 100
                + [f"a{i:03d}" for i in range(100)],
            }
        )
        _maybe_remove(store, "df")
        store.append("df", df, data_columns=["ts", "A", "B", "users"])

        # regular select
        result = store.select("df", "ts>=Timestamp('2012-02-01')")
        expected = df[df.ts >= Timestamp("2012-02-01")]
        tm.assert_frame_equal(expected, result)

        # small selector
        result = store.select("df", "ts>=Timestamp('2012-02-01') & users=['a','b','c']")
        expected = df[
            (df.ts >= Timestamp("2012-02-01")) & df.users.isin(["a", "b", "c"])
        ]
        tm.assert_frame_equal(expected, result)

        # big selector along the columns
        selector = ["a", "b", "c"] + [f"a{i:03d}" for i in range(60)]
        result = store.select("df", "ts>=Timestamp('2012-02-01') and users=selector")
        expected = df[(df.ts >= Timestamp("2012-02-01")) & df.users.isin(selector)]
        tm.assert_frame_equal(expected, result)

        selector = range(100, 200)
        result = store.select("df", "B=selector")
        expected = df[df.B.isin(selector)]
        tm.assert_frame_equal(expected, result)
        assert len(result) == 100

        # big selector along the index
        selector = Index(df.ts[0:100].values)
        result = store.select("df", "ts=selector")
        expected = df[df.ts.isin(selector.values)]
        tm.assert_frame_equal(expected, result)
        assert len(result) == 100


def test_select_iterator(tmp_path, setup_path):
    # single table
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        _maybe_remove(store, "df")
        store.append("df", df)

        expected = store.select("df")

        results = list(store.select("df", iterator=True))
        result = concat(results)
        tm.assert_frame_equal(expected, result)

        results = list(store.select("df", chunksize=2))
        assert len(results) == 5
        result = concat(results)
        tm.assert_frame_equal(expected, result)

        results = list(store.select("df", chunksize=2))
        result = concat(results)
        tm.assert_frame_equal(result, expected)

    path = tmp_path / setup_path

    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    df.to_hdf(path, key="df_non_table")

    msg = "can only use an iterator or chunksize on a table"
    with pytest.raises(TypeError, match=msg):
        read_hdf(path, "df_non_table", chunksize=2)

    with pytest.raises(TypeError, match=msg):
        read_hdf(path, "df_non_table", iterator=True)

    path = tmp_path / setup_path

    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    df.to_hdf(path, key="df", format="table")

    results = list(read_hdf(path, "df", chunksize=2))
    result = concat(results)

    assert len(results) == 5
    tm.assert_frame_equal(result, df)
    tm.assert_frame_equal(result, read_hdf(path, "df"))

    # multiple

    with ensure_clean_store(setup_path) as store:
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        store.append("df1", df1, data_columns=True)
        df2 = df1.copy().rename(columns="{}_2".format)
        df2["foo"] = "bar"
        store.append("df2", df2)

        df = concat([df1, df2], axis=1)

        # full selection
        expected = store.select_as_multiple(["df1", "df2"], selector="df1")
        results = list(
            store.select_as_multiple(["df1", "df2"], selector="df1", chunksize=2)
        )
        result = concat(results)
        tm.assert_frame_equal(expected, result)


def test_select_iterator_complete_8014(setup_path):
    # GH 8014
    # using iterator and where clause
    chunksize = 1e4

    # no iterator
    with ensure_clean_store(setup_path) as store:
        expected = DataFrame(
            np.random.default_rng(2).standard_normal((100064, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=100064, freq="s"),
        )
        _maybe_remove(store, "df")
        store.append("df", expected)

        beg_dt = expected.index[0]
        end_dt = expected.index[-1]

        # select w/o iteration and no where clause works
        result = store.select("df")
        tm.assert_frame_equal(expected, result)

        # select w/o iterator and where clause, single term, begin
        # of range, works
        where = f"index >= '{beg_dt}'"
        result = store.select("df", where=where)
        tm.assert_frame_equal(expected, result)

        # select w/o iterator and where clause, single term, end
        # of range, works
        where = f"index <= '{end_dt}'"
        result = store.select("df", where=where)
        tm.assert_frame_equal(expected, result)

        # select w/o iterator and where clause, inclusive range,
        # works
        where = f"index >= '{beg_dt}' & index <= '{end_dt}'"
        result = store.select("df", where=where)
        tm.assert_frame_equal(expected, result)

    # with iterator, full range
    with ensure_clean_store(setup_path) as store:
        expected = DataFrame(
            np.random.default_rng(2).standard_normal((100064, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=100064, freq="s"),
        )
        _maybe_remove(store, "df")
        store.append("df", expected)

        beg_dt = expected.index[0]
        end_dt = expected.index[-1]

        # select w/iterator and no where clause works
        results = list(store.select("df", chunksize=chunksize))
        result = concat(results)
        tm.assert_frame_equal(expected, result)

        # select w/iterator and where clause, single term, begin of range
        where = f"index >= '{beg_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))
        result = concat(results)
        tm.assert_frame_equal(expected, result)

        # select w/iterator and where clause, single term, end of range
        where = f"index <= '{end_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))
        result = concat(results)
        tm.assert_frame_equal(expected, result)

        # select w/iterator and where clause, inclusive range
        where = f"index >= '{beg_dt}' & index <= '{end_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))
        result = concat(results)
        tm.assert_frame_equal(expected, result)


def test_select_iterator_non_complete_8014(setup_path):
    # GH 8014
    # using iterator and where clause
    chunksize = 1e4

    # with iterator, non complete range
    with ensure_clean_store(setup_path) as store:
        expected = DataFrame(
            np.random.default_rng(2).standard_normal((100064, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=100064, freq="s"),
        )
        _maybe_remove(store, "df")
        store.append("df", expected)

        beg_dt = expected.index[1]
        end_dt = expected.index[-2]

        # select w/iterator and where clause, single term, begin of range
        where = f"index >= '{beg_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))
        result = concat(results)
        rexpected = expected[expected.index >= beg_dt]
        tm.assert_frame_equal(rexpected, result)

        # select w/iterator and where clause, single term, end of range
        where = f"index <= '{end_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))
        result = concat(results)
        rexpected = expected[expected.index <= end_dt]
        tm.assert_frame_equal(rexpected, result)

        # select w/iterator and where clause, inclusive range
        where = f"index >= '{beg_dt}' & index <= '{end_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))
        result = concat(results)
        rexpected = expected[(expected.index >= beg_dt) & (expected.index <= end_dt)]
        tm.assert_frame_equal(rexpected, result)

    # with iterator, empty where
    with ensure_clean_store(setup_path) as store:
        expected = DataFrame(
            np.random.default_rng(2).standard_normal((100064, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=100064, freq="s"),
        )
        _maybe_remove(store, "df")
        store.append("df", expected)

        end_dt = expected.index[-1]

        # select w/iterator and where clause, single term, begin of range
        where = f"index > '{end_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))
        assert 0 == len(results)


def test_select_iterator_many_empty_frames(setup_path):
    # GH 8014
    # using iterator and where clause can return many empty
    # frames.
    chunksize = 10_000

    # with iterator, range limited to the first chunk
    with ensure_clean_store(setup_path) as store:
        expected = DataFrame(
            np.random.default_rng(2).standard_normal((100064, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=100064, freq="s"),
        )
        _maybe_remove(store, "df")
        store.append("df", expected)

        beg_dt = expected.index[0]
        end_dt = expected.index[chunksize - 1]

        # select w/iterator and where clause, single term, begin of range
        where = f"index >= '{beg_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))
        result = concat(results)
        rexpected = expected[expected.index >= beg_dt]
        tm.assert_frame_equal(rexpected, result)

        # select w/iterator and where clause, single term, end of range
        where = f"index <= '{end_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))

        assert len(results) == 1
        result = concat(results)
        rexpected = expected[expected.index <= end_dt]
        tm.assert_frame_equal(rexpected, result)

        # select w/iterator and where clause, inclusive range
        where = f"index >= '{beg_dt}' & index <= '{end_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))

        # should be 1, is 10
        assert len(results) == 1
        result = concat(results)
        rexpected = expected[(expected.index >= beg_dt) & (expected.index <= end_dt)]
        tm.assert_frame_equal(rexpected, result)

        # select w/iterator and where clause which selects
        # *nothing*.
        #
        # To be consistent with Python idiom I suggest this should
        # return [] e.g. `for e in []: print True` never prints
        # True.

        where = f"index <= '{beg_dt}' & index >= '{end_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))

        # should be []
        assert len(results) == 0


def test_frame_select(setup_path):
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )

    with ensure_clean_store(setup_path) as store:
        store.put("frame", df, format="table")
        date = df.index[len(df) // 2]

        crit1 = Term("index>=date")
        assert crit1.env.scope["date"] == date

        crit2 = "columns=['A', 'D']"
        crit3 = "columns=A"

        result = store.select("frame", [crit1, crit2])
        expected = df.loc[date:, ["A", "D"]]
        tm.assert_frame_equal(result, expected)

        result = store.select("frame", [crit3])
        expected = df.loc[:, ["A"]]
        tm.assert_frame_equal(result, expected)

        # invalid terms
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        store.append("df_time", df)
        msg = "day is out of range for month: 0"
        with pytest.raises(ValueError, match=msg):
            store.select("df_time", "index>0")

        # can't select if not written as table
        # store['frame'] = df
        # with pytest.raises(ValueError):
        #     store.select('frame', [crit1, crit2])


def test_frame_select_complex(setup_path):
    # select via complex criteria

    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    df["string"] = "foo"
    df.loc[df.index[0:4], "string"] = "bar"

    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", data_columns=["string"])

        # empty
        result = store.select("df", 'index>df.index[3] & string="bar"')
        expected = df.loc[(df.index > df.index[3]) & (df.string == "bar")]
        tm.assert_frame_equal(result, expected)

        result = store.select("df", 'index>df.index[3] & string="foo"')
        expected = df.loc[(df.index > df.index[3]) & (df.string == "foo")]
        tm.assert_frame_equal(result, expected)

        # or
        result = store.select("df", 'index>df.index[3] | string="bar"')
        expected = df.loc[(df.index > df.index[3]) | (df.string == "bar")]
        tm.assert_frame_equal(result, expected)

        result = store.select(
            "df", '(index>df.index[3] & index<=df.index[6]) | string="bar"'
        )
        expected = df.loc[
            ((df.index > df.index[3]) & (df.index <= df.index[6]))
            | (df.string == "bar")
        ]
        tm.assert_frame_equal(result, expected)

        # invert
        result = store.select("df", 'string!="bar"')
        expected = df.loc[df.string != "bar"]
        tm.assert_frame_equal(result, expected)

        # invert not implemented in numexpr :(
        msg = "cannot use an invert condition when passing to numexpr"
        with pytest.raises(NotImplementedError, match=msg):
            store.select("df", '~(string="bar")')

        # invert ok for filters
        result = store.select("df", "~(columns=['A','B'])")
        expected = df.loc[:, df.columns.difference(["A", "B"])]
        tm.assert_frame_equal(result, expected)

        # in
        result = store.select("df", "index>df.index[3] & columns in ['A','B']")
        expected = df.loc[df.index > df.index[3]].reindex(columns=["A", "B"])
        tm.assert_frame_equal(result, expected)


def test_frame_select_complex2(tmp_path):
    pp = tmp_path / "params.hdf"
    hh = tmp_path / "hist.hdf"

    # use non-trivial selection criteria
    params = DataFrame({"A": [1, 1, 2, 2, 3]})
    params.to_hdf(pp, key="df", mode="w", format="table", data_columns=["A"])

    selection = read_hdf(pp, "df", where="A=[2,3]")
    hist = DataFrame(
        np.random.default_rng(2).standard_normal((25, 1)),
        columns=["data"],
        index=MultiIndex.from_tuples(
            [(i, j) for i in range(5) for j in range(5)], names=["l1", "l2"]
        ),
    )

    hist.to_hdf(hh, key="df", mode="w", format="table")

    expected = read_hdf(hh, "df", where="l1=[2, 3, 4]")

    # scope with list like
    l0 = selection.index.tolist()  # noqa: F841
    with HDFStore(hh) as store:
        result = store.select("df", where="l1=l0")
        tm.assert_frame_equal(result, expected)

    result = read_hdf(hh, "df", where="l1=l0")
    tm.assert_frame_equal(result, expected)

    # index
    index = selection.index  # noqa: F841
    result = read_hdf(hh, "df", where="l1=index")
    tm.assert_frame_equal(result, expected)

    result = read_hdf(hh, "df", where="l1=selection.index")
    tm.assert_frame_equal(result, expected)

    result = read_hdf(hh, "df", where="l1=selection.index.tolist()")
    tm.assert_frame_equal(result, expected)

    result = read_hdf(hh, "df", where="l1=list(selection.index)")
    tm.assert_frame_equal(result, expected)

    # scope with index
    with HDFStore(hh) as store:
        result = store.select("df", where="l1=index")
        tm.assert_frame_equal(result, expected)

        result = store.select("df", where="l1=selection.index")
        tm.assert_frame_equal(result, expected)

        result = store.select("df", where="l1=selection.index.tolist()")
        tm.assert_frame_equal(result, expected)

        result = store.select("df", where="l1=list(selection.index)")
        tm.assert_frame_equal(result, expected)


def test_invalid_filtering(setup_path):
    # can't use more than one filter (atm)

    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )

    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table")

        msg = "unable to collapse Joint Filters"
        # not implemented
        with pytest.raises(NotImplementedError, match=msg):
            store.select("df", "columns=['A'] | columns=['B']")

        # in theory we could deal with this
        with pytest.raises(NotImplementedError, match=msg):
            store.select("df", "columns=['A','B'] & columns=['C']")


def test_string_select(setup_path):
    # GH 2973
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )

        # test string ==/!=
        df["x"] = "none"
        df.loc[df.index[2:7], "x"] = ""

        store.append("df", df, data_columns=["x"])

        result = store.select("df", "x=none")
        expected = df[df.x == "none"]
        tm.assert_frame_equal(result, expected)

        result = store.select("df", "x!=none")
        expected = df[df.x != "none"]
        tm.assert_frame_equal(result, expected)

        df2 = df.copy()
        df2.loc[df2.x == "", "x"] = np.nan

        store.append("df2", df2, data_columns=["x"])
        result = store.select("df2", "x!=none")
        expected = df2[isna(df2.x)]
        tm.assert_frame_equal(result, expected)

        # int ==/!=
        df["int"] = 1
        df.loc[df.index[2:7], "int"] = 2

        store.append("df3", df, data_columns=["int"])

        result = store.select("df3", "int=2")
        expected = df[df.int == 2]
        tm.assert_frame_equal(result, expected)

        result = store.select("df3", "int!=2")
        expected = df[df.int != 2]
        tm.assert_frame_equal(result, expected)


def test_select_as_multiple(setup_path):
    df1 = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    df2 = df1.copy().rename(columns="{}_2".format)
    df2["foo"] = "bar"

    with ensure_clean_store(setup_path) as store:
        msg = "keys must be a list/tuple"
        # no tables stored
        with pytest.raises(TypeError, match=msg):
            store.select_as_multiple(None, where=["A>0", "B>0"], selector="df1")

        store.append("df1", df1, data_columns=["A", "B"])
        store.append("df2", df2)

        # exceptions
        with pytest.raises(TypeError, match=msg):
            store.select_as_multiple(None, where=["A>0", "B>0"], selector="df1")

        with pytest.raises(TypeError, match=msg):
            store.select_as_multiple([None], where=["A>0", "B>0"], selector="df1")

        msg = "'No object named df3 in the file'"
        with pytest.raises(KeyError, match=msg):
            store.select_as_multiple(
                ["df1", "df3"], where=["A>0", "B>0"], selector="df1"
            )

        with pytest.raises(KeyError, match=msg):
            store.select_as_multiple(["df3"], where=["A>0", "B>0"], selector="df1")

        with pytest.raises(KeyError, match="'No object named df4 in the file'"):
            store.select_as_multiple(
                ["df1", "df2"], where=["A>0", "B>0"], selector="df4"
            )

        # default select
        result = store.select("df1", ["A>0", "B>0"])
        expected = store.select_as_multiple(
            ["df1"], where=["A>0", "B>0"], selector="df1"
        )
        tm.assert_frame_equal(result, expected)
        expected = store.select_as_multiple("df1", where=["A>0", "B>0"], selector="df1")
        tm.assert_frame_equal(result, expected)

        # multiple
        result = store.select_as_multiple(
            ["df1", "df2"], where=["A>0", "B>0"], selector="df1"
        )
        expected = concat([df1, df2], axis=1)
        expected = expected[(expected.A > 0) & (expected.B > 0)]
        tm.assert_frame_equal(result, expected, check_freq=False)
        # FIXME: 2021-01-20 this is failing with freq None vs 4B on some builds

        # multiple (diff selector)
        result = store.select_as_multiple(
            ["df1", "df2"], where="index>df2.index[4]", selector="df2"
        )
        expected = concat([df1, df2], axis=1)
        expected = expected[5:]
        tm.assert_frame_equal(result, expected)

        # test exception for diff rows
        df3 = df1.copy().head(2)
        store.append("df3", df3)
        msg = "all tables must have exactly the same nrows!"
        with pytest.raises(ValueError, match=msg):
            store.select_as_multiple(
                ["df1", "df3"], where=["A>0", "B>0"], selector="df1"
            )


def test_nan_selection_bug_4858(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame({"cols": range(6), "values": range(6)}, dtype="float64")
        df["cols"] = (df["cols"] + 10).apply(str)
        df.iloc[0] = np.nan

        expected = DataFrame(
            {"cols": ["13.0", "14.0", "15.0"], "values": [3.0, 4.0, 5.0]},
            index=[3, 4, 5],
        )

        # write w/o the index on that particular column
        store.append("df", df, data_columns=True, index=["cols"])
        result = store.select("df", where="values>2.0")
        tm.assert_frame_equal(result, expected)


def test_query_with_nested_special_character(setup_path):
    df = DataFrame(
        {
            "a": ["a", "a", "c", "b", "test & test", "c", "b", "e"],
            "b": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    expected = df[df.a == "test & test"]
    with ensure_clean_store(setup_path) as store:
        store.append("test", df, format="table", data_columns=True)
        result = store.select("test", 'a = "test & test"')
    tm.assert_frame_equal(expected, result)


def test_query_long_float_literal(setup_path):
    # GH 14241
    df = DataFrame({"A": [1000000000.0009, 1000000000.0011, 1000000000.0015]})

    with ensure_clean_store(setup_path) as store:
        store.append("test", df, format="table", data_columns=True)

        cutoff = 1000000000.0006
        result = store.select("test", f"A < {cutoff:.4f}")
        assert result.empty

        cutoff = 1000000000.0010
        result = store.select("test", f"A > {cutoff:.4f}")
        expected = df.loc[[1, 2], :]
        tm.assert_frame_equal(expected, result)

        exact = 1000000000.0011
        result = store.select("test", f"A == {exact:.4f}")
        expected = df.loc[[1], :]
        tm.assert_frame_equal(expected, result)


def test_query_compare_column_type(setup_path):
    # GH 15492
    df = DataFrame(
        {
            "date": ["2014-01-01", "2014-01-02"],
            "real_date": date_range("2014-01-01", periods=2),
            "float": [1.1, 1.2],
            "int": [1, 2],
        },
        columns=["date", "real_date", "float", "int"],
    )

    with ensure_clean_store(setup_path) as store:
        store.append("test", df, format="table", data_columns=True)

        ts = Timestamp("2014-01-01")  # noqa: F841
        result = store.select("test", where="real_date > ts")
        expected = df.loc[[1], :]
        tm.assert_frame_equal(expected, result)

        for op in ["<", ">", "=="]:
            # non strings to string column always fail
            for v in [2.1, True, Timestamp("2014-01-01"), pd.Timedelta(1, "s")]:
                query = f"date {op} v"
                msg = f"Cannot compare {v} of type {type(v)} to string column"
                with pytest.raises(TypeError, match=msg):
                    store.select("test", where=query)

            # strings to other columns must be convertible to type
            v = "a"
            for col in ["int", "float", "real_date"]:
                query = f"{col} {op} v"
                if col == "real_date":
                    msg = 'Given date string "a" not likely a datetime'
                else:
                    msg = "could not convert string to"
                with pytest.raises(ValueError, match=msg):
                    store.select("test", where=query)

            for v, col in zip(
                ["1", "1.1", "2014-01-01"], ["int", "float", "real_date"]
            ):
                query = f"{col} {op} v"
                result = store.select("test", where=query)

                if op == "==":
                    expected = df.loc[[0], :]
                elif op == ">":
                    expected = df.loc[[1], :]
                else:
                    expected = df.loc[[], :]
                tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize("where", ["", (), (None,), [], [None]])
def test_select_empty_where(tmp_path, where):
    # GH26610

    df = DataFrame([1, 2, 3])
    path = tmp_path / "empty_where.h5"
    with HDFStore(path) as store:
        store.put("df", df, "t")
        result = read_hdf(store, "df", where=where)
        tm.assert_frame_equal(result, df)


def test_select_large_integer(tmp_path):
    path = tmp_path / "large_int.h5"

    df = DataFrame(
        zip(
            ["a", "b", "c", "d"],
            [-9223372036854775801, -9223372036854775802, -9223372036854775803, 123],
        ),
        columns=["x", "y"],
    )
    with HDFStore(path) as s:
        s.append("data", df, data_columns=True, index=False)
        result = s.select("data", where="y==-9223372036854775801").get("y").get(0)
    expected = df["y"][0]

    assert expected == result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_read_fwf.py ===
"""
Tests the 'read_fwf' function in parsers.py. This
test suite is independent of the others because the
engine is set to 'python-fwf' internally.
"""

from datetime import datetime
from io import (
    BytesIO,
    StringIO,
)
from pathlib import Path

import numpy as np
import pytest

from pandas.errors import EmptyDataError

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
)
import pandas._testing as tm

from pandas.io.common import urlopen
from pandas.io.parsers import (
    read_csv,
    read_fwf,
)


def test_basic():
    data = """\
A         B            C            D
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4
201160    364.136849   183.628767   11806.2
201161    413.836124   184.375703   11916.8
201162    502.953953   173.237159   12468.3
"""
    result = read_fwf(StringIO(data))
    expected = DataFrame(
        [
            [201158, 360.242940, 149.910199, 11950.7],
            [201159, 444.953632, 166.985655, 11788.4],
            [201160, 364.136849, 183.628767, 11806.2],
            [201161, 413.836124, 184.375703, 11916.8],
            [201162, 502.953953, 173.237159, 12468.3],
        ],
        columns=["A", "B", "C", "D"],
    )
    tm.assert_frame_equal(result, expected)


def test_colspecs():
    data = """\
A   B     C            D            E
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4
201160    364.136849   183.628767   11806.2
201161    413.836124   184.375703   11916.8
201162    502.953953   173.237159   12468.3
"""
    colspecs = [(0, 4), (4, 8), (8, 20), (21, 33), (34, 43)]
    result = read_fwf(StringIO(data), colspecs=colspecs)

    expected = DataFrame(
        [
            [2011, 58, 360.242940, 149.910199, 11950.7],
            [2011, 59, 444.953632, 166.985655, 11788.4],
            [2011, 60, 364.136849, 183.628767, 11806.2],
            [2011, 61, 413.836124, 184.375703, 11916.8],
            [2011, 62, 502.953953, 173.237159, 12468.3],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    tm.assert_frame_equal(result, expected)


def test_widths():
    data = """\
A    B    C            D            E
2011 58   360.242940   149.910199   11950.7
2011 59   444.953632   166.985655   11788.4
2011 60   364.136849   183.628767   11806.2
2011 61   413.836124   184.375703   11916.8
2011 62   502.953953   173.237159   12468.3
"""
    result = read_fwf(StringIO(data), widths=[5, 5, 13, 13, 7])

    expected = DataFrame(
        [
            [2011, 58, 360.242940, 149.910199, 11950.7],
            [2011, 59, 444.953632, 166.985655, 11788.4],
            [2011, 60, 364.136849, 183.628767, 11806.2],
            [2011, 61, 413.836124, 184.375703, 11916.8],
            [2011, 62, 502.953953, 173.237159, 12468.3],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    tm.assert_frame_equal(result, expected)


def test_non_space_filler():
    # From Thomas Kluyver:
    #
    # Apparently, some non-space filler characters can be seen, this is
    # supported by specifying the 'delimiter' character:
    #
    # http://publib.boulder.ibm.com/infocenter/dmndhelp/v6r1mx/index.jsp?topic=/com.ibm.wbit.612.help.config.doc/topics/rfixwidth.html
    data = """\
A~~~~B~~~~C~~~~~~~~~~~~D~~~~~~~~~~~~E
201158~~~~360.242940~~~149.910199~~~11950.7
201159~~~~444.953632~~~166.985655~~~11788.4
201160~~~~364.136849~~~183.628767~~~11806.2
201161~~~~413.836124~~~184.375703~~~11916.8
201162~~~~502.953953~~~173.237159~~~12468.3
"""
    colspecs = [(0, 4), (4, 8), (8, 20), (21, 33), (34, 43)]
    result = read_fwf(StringIO(data), colspecs=colspecs, delimiter="~")

    expected = DataFrame(
        [
            [2011, 58, 360.242940, 149.910199, 11950.7],
            [2011, 59, 444.953632, 166.985655, 11788.4],
            [2011, 60, 364.136849, 183.628767, 11806.2],
            [2011, 61, 413.836124, 184.375703, 11916.8],
            [2011, 62, 502.953953, 173.237159, 12468.3],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    tm.assert_frame_equal(result, expected)


def test_over_specified():
    data = """\
A   B     C            D            E
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4
201160    364.136849   183.628767   11806.2
201161    413.836124   184.375703   11916.8
201162    502.953953   173.237159   12468.3
"""
    colspecs = [(0, 4), (4, 8), (8, 20), (21, 33), (34, 43)]

    with pytest.raises(ValueError, match="must specify only one of"):
        read_fwf(StringIO(data), colspecs=colspecs, widths=[6, 10, 10, 7])


def test_under_specified():
    data = """\
A   B     C            D            E
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4
201160    364.136849   183.628767   11806.2
201161    413.836124   184.375703   11916.8
201162    502.953953   173.237159   12468.3
"""
    with pytest.raises(ValueError, match="Must specify either"):
        read_fwf(StringIO(data), colspecs=None, widths=None)


def test_read_csv_compat():
    csv_data = """\
A,B,C,D,E
2011,58,360.242940,149.910199,11950.7
2011,59,444.953632,166.985655,11788.4
2011,60,364.136849,183.628767,11806.2
2011,61,413.836124,184.375703,11916.8
2011,62,502.953953,173.237159,12468.3
"""
    expected = read_csv(StringIO(csv_data), engine="python")

    fwf_data = """\
A   B     C            D            E
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4
201160    364.136849   183.628767   11806.2
201161    413.836124   184.375703   11916.8
201162    502.953953   173.237159   12468.3
"""
    colspecs = [(0, 4), (4, 8), (8, 20), (21, 33), (34, 43)]
    result = read_fwf(StringIO(fwf_data), colspecs=colspecs)
    tm.assert_frame_equal(result, expected)


def test_bytes_io_input():
    data = BytesIO("שלום\nשלום".encode())  # noqa: RUF001
    result = read_fwf(data, widths=[2, 2], encoding="utf8")
    expected = DataFrame([["של", "ום"]], columns=["של", "ום"])
    tm.assert_frame_equal(result, expected)


def test_fwf_colspecs_is_list_or_tuple():
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""

    msg = "column specifications must be a list or tuple.+"

    with pytest.raises(TypeError, match=msg):
        read_fwf(StringIO(data), colspecs={"a": 1}, delimiter=",")


def test_fwf_colspecs_is_list_or_tuple_of_two_element_tuples():
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""

    msg = "Each column specification must be.+"

    with pytest.raises(TypeError, match=msg):
        read_fwf(StringIO(data), colspecs=[("a", 1)])


@pytest.mark.parametrize(
    "colspecs,exp_data",
    [
        ([(0, 3), (3, None)], [[123, 456], [456, 789]]),
        ([(None, 3), (3, 6)], [[123, 456], [456, 789]]),
        ([(0, None), (3, None)], [[123456, 456], [456789, 789]]),
        ([(None, None), (3, 6)], [[123456, 456], [456789, 789]]),
    ],
)
def test_fwf_colspecs_none(colspecs, exp_data):
    # see gh-7079
    data = """\
123456
456789
"""
    expected = DataFrame(exp_data)

    result = read_fwf(StringIO(data), colspecs=colspecs, header=None)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "infer_nrows,exp_data",
    [
        # infer_nrows --> colspec == [(2, 3), (5, 6)]
        (1, [[1, 2], [3, 8]]),
        # infer_nrows > number of rows
        (10, [[1, 2], [123, 98]]),
    ],
)
def test_fwf_colspecs_infer_nrows(infer_nrows, exp_data):
    # see gh-15138
    data = """\
  1  2
123 98
"""
    expected = DataFrame(exp_data)

    result = read_fwf(StringIO(data), infer_nrows=infer_nrows, header=None)
    tm.assert_frame_equal(result, expected)


def test_fwf_regression():
    # see gh-3594
    #
    # Turns out "T060" is parsable as a datetime slice!
    tz_list = [1, 10, 20, 30, 60, 80, 100]
    widths = [16] + [8] * len(tz_list)
    names = ["SST"] + [f"T{z:03d}" for z in tz_list[1:]]

    data = """  2009164202000   9.5403  9.4105  8.6571  7.8372  6.0612  5.8843  5.5192
2009164203000   9.5435  9.2010  8.6167  7.8176  6.0804  5.8728  5.4869
2009164204000   9.5873  9.1326  8.4694  7.5889  6.0422  5.8526  5.4657
2009164205000   9.5810  9.0896  8.4009  7.4652  6.0322  5.8189  5.4379
2009164210000   9.6034  9.0897  8.3822  7.4905  6.0908  5.7904  5.4039
"""

    with tm.assert_produces_warning(FutureWarning, match="use 'date_format' instead"):
        result = read_fwf(
            StringIO(data),
            index_col=0,
            header=None,
            names=names,
            widths=widths,
            parse_dates=True,
            date_parser=lambda s: datetime.strptime(s, "%Y%j%H%M%S"),
        )
    expected = DataFrame(
        [
            [9.5403, 9.4105, 8.6571, 7.8372, 6.0612, 5.8843, 5.5192],
            [9.5435, 9.2010, 8.6167, 7.8176, 6.0804, 5.8728, 5.4869],
            [9.5873, 9.1326, 8.4694, 7.5889, 6.0422, 5.8526, 5.4657],
            [9.5810, 9.0896, 8.4009, 7.4652, 6.0322, 5.8189, 5.4379],
            [9.6034, 9.0897, 8.3822, 7.4905, 6.0908, 5.7904, 5.4039],
        ],
        index=DatetimeIndex(
            [
                "2009-06-13 20:20:00",
                "2009-06-13 20:30:00",
                "2009-06-13 20:40:00",
                "2009-06-13 20:50:00",
                "2009-06-13 21:00:00",
            ]
        ),
        columns=["SST", "T010", "T020", "T030", "T060", "T080", "T100"],
    )
    tm.assert_frame_equal(result, expected)
    result = read_fwf(
        StringIO(data),
        index_col=0,
        header=None,
        names=names,
        widths=widths,
        parse_dates=True,
        date_format="%Y%j%H%M%S",
    )
    tm.assert_frame_equal(result, expected)


def test_fwf_for_uint8():
    data = """1421302965.213420    PRI=3 PGN=0xef00      DST=0x17 SRC=0x28    04 154 00 00 00 00 00 127
1421302964.226776    PRI=6 PGN=0xf002               SRC=0x47    243 00 00 255 247 00 00 71"""  # noqa: E501
    df = read_fwf(
        StringIO(data),
        colspecs=[(0, 17), (25, 26), (33, 37), (49, 51), (58, 62), (63, 1000)],
        names=["time", "pri", "pgn", "dst", "src", "data"],
        converters={
            "pgn": lambda x: int(x, 16),
            "src": lambda x: int(x, 16),
            "dst": lambda x: int(x, 16),
            "data": lambda x: len(x.split(" ")),
        },
    )

    expected = DataFrame(
        [
            [1421302965.213420, 3, 61184, 23, 40, 8],
            [1421302964.226776, 6, 61442, None, 71, 8],
        ],
        columns=["time", "pri", "pgn", "dst", "src", "data"],
    )
    expected["dst"] = expected["dst"].astype(object)
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("comment", ["#", "~", "!"])
def test_fwf_comment(comment):
    data = """\
  1   2.   4  #hello world
  5  NaN  10.0
"""
    data = data.replace("#", comment)

    colspecs = [(0, 3), (4, 9), (9, 25)]
    expected = DataFrame([[1, 2.0, 4], [5, np.nan, 10.0]])

    result = read_fwf(StringIO(data), colspecs=colspecs, header=None, comment=comment)
    tm.assert_almost_equal(result, expected)


def test_fwf_skip_blank_lines():
    data = """

A         B            C            D

201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4


201162    502.953953   173.237159   12468.3

"""
    result = read_fwf(StringIO(data), skip_blank_lines=True)
    expected = DataFrame(
        [
            [201158, 360.242940, 149.910199, 11950.7],
            [201159, 444.953632, 166.985655, 11788.4],
            [201162, 502.953953, 173.237159, 12468.3],
        ],
        columns=["A", "B", "C", "D"],
    )
    tm.assert_frame_equal(result, expected)

    data = """\
A         B            C            D
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4


201162    502.953953   173.237159   12468.3
"""
    result = read_fwf(StringIO(data), skip_blank_lines=False)
    expected = DataFrame(
        [
            [201158, 360.242940, 149.910199, 11950.7],
            [201159, 444.953632, 166.985655, 11788.4],
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [201162, 502.953953, 173.237159, 12468.3],
        ],
        columns=["A", "B", "C", "D"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("thousands", [",", "#", "~"])
def test_fwf_thousands(thousands):
    data = """\
 1 2,334.0    5
10   13     10.
"""
    data = data.replace(",", thousands)

    colspecs = [(0, 3), (3, 11), (12, 16)]
    expected = DataFrame([[1, 2334.0, 5], [10, 13, 10.0]])

    result = read_fwf(
        StringIO(data), header=None, colspecs=colspecs, thousands=thousands
    )
    tm.assert_almost_equal(result, expected)


@pytest.mark.parametrize("header", [True, False])
def test_bool_header_arg(header):
    # see gh-6114
    data = """\
MyColumn
   a
   b
   a
   b"""

    msg = "Passing a bool to header is invalid"
    with pytest.raises(TypeError, match=msg):
        read_fwf(StringIO(data), header=header)


def test_full_file():
    # File with all values.
    test = """index                             A    B    C
2000-01-03T00:00:00  0.980268513777    3  foo
2000-01-04T00:00:00  1.04791624281    -4  bar
2000-01-05T00:00:00  0.498580885705   73  baz
2000-01-06T00:00:00  1.12020151869     1  foo
2000-01-07T00:00:00  0.487094399463    0  bar
2000-01-10T00:00:00  0.836648671666    2  baz
2000-01-11T00:00:00  0.157160753327   34  foo"""
    colspecs = ((0, 19), (21, 35), (38, 40), (42, 45))
    expected = read_fwf(StringIO(test), colspecs=colspecs)

    result = read_fwf(StringIO(test))
    tm.assert_frame_equal(result, expected)


def test_full_file_with_missing():
    # File with missing values.
    test = """index                             A    B    C
2000-01-03T00:00:00  0.980268513777    3  foo
2000-01-04T00:00:00  1.04791624281    -4  bar
                     0.498580885705   73  baz
2000-01-06T00:00:00  1.12020151869     1  foo
2000-01-07T00:00:00                    0  bar
2000-01-10T00:00:00  0.836648671666    2  baz
                                      34"""
    colspecs = ((0, 19), (21, 35), (38, 40), (42, 45))
    expected = read_fwf(StringIO(test), colspecs=colspecs)

    result = read_fwf(StringIO(test))
    tm.assert_frame_equal(result, expected)


def test_full_file_with_spaces():
    # File with spaces in columns.
    test = """
Account                 Name  Balance     CreditLimit   AccountCreated
101     Keanu Reeves          9315.45     10000.00           1/17/1998
312     Gerard Butler         90.00       1000.00             8/6/2003
868     Jennifer Love Hewitt  0           17000.00           5/25/1985
761     Jada Pinkett-Smith    49654.87    100000.00          12/5/2006
317     Bill Murray           789.65      5000.00             2/5/2007
""".strip(
        "\r\n"
    )
    colspecs = ((0, 7), (8, 28), (30, 38), (42, 53), (56, 70))
    expected = read_fwf(StringIO(test), colspecs=colspecs)

    result = read_fwf(StringIO(test))
    tm.assert_frame_equal(result, expected)


def test_full_file_with_spaces_and_missing():
    # File with spaces and missing values in columns.
    test = """
Account               Name    Balance     CreditLimit   AccountCreated
101                           10000.00                       1/17/1998
312     Gerard Butler         90.00       1000.00             8/6/2003
868                                                          5/25/1985
761     Jada Pinkett-Smith    49654.87    100000.00          12/5/2006
317     Bill Murray           789.65
""".strip(
        "\r\n"
    )
    colspecs = ((0, 7), (8, 28), (30, 38), (42, 53), (56, 70))
    expected = read_fwf(StringIO(test), colspecs=colspecs)

    result = read_fwf(StringIO(test))
    tm.assert_frame_equal(result, expected)


def test_messed_up_data():
    # Completely messed up file.
    test = """
   Account          Name             Balance     Credit Limit   Account Created
       101                           10000.00                       1/17/1998
       312     Gerard Butler         90.00       1000.00

       761     Jada Pinkett-Smith    49654.87    100000.00          12/5/2006
  317          Bill Murray           789.65
""".strip(
        "\r\n"
    )
    colspecs = ((2, 10), (15, 33), (37, 45), (49, 61), (64, 79))
    expected = read_fwf(StringIO(test), colspecs=colspecs)

    result = read_fwf(StringIO(test))
    tm.assert_frame_equal(result, expected)


def test_multiple_delimiters():
    test = r"""
col1~~~~~col2  col3++++++++++++++++++col4
~~22.....11.0+++foo~~~~~~~~~~Keanu Reeves
  33+++122.33\\\bar.........Gerard Butler
++44~~~~12.01   baz~~Jennifer Love Hewitt
~~55       11+++foo++++Jada Pinkett-Smith
..66++++++.03~~~bar           Bill Murray
""".strip(
        "\r\n"
    )
    delimiter = " +~.\\"
    colspecs = ((0, 4), (7, 13), (15, 19), (21, 41))
    expected = read_fwf(StringIO(test), colspecs=colspecs, delimiter=delimiter)

    result = read_fwf(StringIO(test), delimiter=delimiter)
    tm.assert_frame_equal(result, expected)


def test_variable_width_unicode():
    data = """
שלום שלום
ום   שלל
של   ום
""".strip(
        "\r\n"
    )
    encoding = "utf8"
    kwargs = {"header": None, "encoding": encoding}

    expected = read_fwf(
        BytesIO(data.encode(encoding)), colspecs=[(0, 4), (5, 9)], **kwargs
    )
    result = read_fwf(BytesIO(data.encode(encoding)), **kwargs)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", [{}, {"a": "float64", "b": str, "c": "int32"}])
def test_dtype(dtype):
    data = """ a    b    c
1    2    3.2
3    4    5.2
"""
    colspecs = [(0, 5), (5, 10), (10, None)]
    result = read_fwf(StringIO(data), colspecs=colspecs, dtype=dtype)

    expected = DataFrame(
        {"a": [1, 3], "b": [2, 4], "c": [3.2, 5.2]}, columns=["a", "b", "c"]
    )

    for col, dt in dtype.items():
        expected[col] = expected[col].astype(dt)

    tm.assert_frame_equal(result, expected)


def test_skiprows_inference():
    # see gh-11256
    data = """
Text contained in the file header

DataCol1   DataCol2
     0.0        1.0
   101.6      956.1
""".strip()
    skiprows = 2

    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        expected = read_csv(StringIO(data), skiprows=skiprows, delim_whitespace=True)

    result = read_fwf(StringIO(data), skiprows=skiprows)
    tm.assert_frame_equal(result, expected)


def test_skiprows_by_index_inference():
    data = """
To be skipped
Not  To  Be  Skipped
Once more to be skipped
123  34   8      123
456  78   9      456
""".strip()
    skiprows = [0, 2]

    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        expected = read_csv(StringIO(data), skiprows=skiprows, delim_whitespace=True)

    result = read_fwf(StringIO(data), skiprows=skiprows)
    tm.assert_frame_equal(result, expected)


def test_skiprows_inference_empty():
    data = """
AA   BBB  C
12   345  6
78   901  2
""".strip()

    msg = "No rows from which to infer column width"
    with pytest.raises(EmptyDataError, match=msg):
        read_fwf(StringIO(data), skiprows=3)


def test_whitespace_preservation():
    # see gh-16772
    header = None
    csv_data = """
 a ,bbb
 cc,dd """

    fwf_data = """
 a bbb
 ccdd """
    result = read_fwf(
        StringIO(fwf_data), widths=[3, 3], header=header, skiprows=[0], delimiter="\n\t"
    )
    expected = read_csv(StringIO(csv_data), header=header)
    tm.assert_frame_equal(result, expected)


def test_default_delimiter():
    header = None
    csv_data = """
a,bbb
cc,dd"""

    fwf_data = """
a \tbbb
cc\tdd """
    result = read_fwf(StringIO(fwf_data), widths=[3, 3], header=header, skiprows=[0])
    expected = read_csv(StringIO(csv_data), header=header)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("infer", [True, False])
def test_fwf_compression(compression_only, infer, compression_to_extension):
    data = """1111111111
    2222222222
    3333333333""".strip()

    compression = compression_only
    extension = compression_to_extension[compression]

    kwargs = {"widths": [5, 5], "names": ["one", "two"]}
    expected = read_fwf(StringIO(data), **kwargs)

    data = bytes(data, encoding="utf-8")

    with tm.ensure_clean(filename="tmp." + extension) as path:
        tm.write_to_compressed(compression, path, data)

        if infer is not None:
            kwargs["compression"] = "infer" if infer else compression

        result = read_fwf(path, **kwargs)
        tm.assert_frame_equal(result, expected)


def test_binary_mode():
    """
    read_fwf supports opening files in binary mode.

    GH 18035.
    """
    data = """aaa aaa aaa
bba bab b a"""
    df_reference = DataFrame(
        [["bba", "bab", "b a"]], columns=["aaa", "aaa.1", "aaa.2"], index=[0]
    )
    with tm.ensure_clean() as path:
        Path(path).write_text(data, encoding="utf-8")
        with open(path, "rb") as file:
            df = read_fwf(file)
            file.seek(0)
            tm.assert_frame_equal(df, df_reference)


@pytest.mark.parametrize("memory_map", [True, False])
def test_encoding_mmap(memory_map):
    """
    encoding should be working, even when using a memory-mapped file.

    GH 23254.
    """
    encoding = "iso8859_1"
    with tm.ensure_clean() as path:
        Path(path).write_bytes(" 1 A Ä 2\n".encode(encoding))
        df = read_fwf(
            path,
            header=None,
            widths=[2, 2, 2, 2],
            encoding=encoding,
            memory_map=memory_map,
        )
    df_reference = DataFrame([[1, "A", "Ä", 2]])
    tm.assert_frame_equal(df, df_reference)


@pytest.mark.parametrize(
    "colspecs, names, widths, index_col",
    [
        (
            [(0, 6), (6, 12), (12, 18), (18, None)],
            list("abcde"),
            None,
            None,
        ),
        (
            None,
            list("abcde"),
            [6] * 4,
            None,
        ),
        (
            [(0, 6), (6, 12), (12, 18), (18, None)],
            list("abcde"),
            None,
            True,
        ),
        (
            None,
            list("abcde"),
            [6] * 4,
            False,
        ),
        (
            None,
            list("abcde"),
            [6] * 4,
            True,
        ),
        (
            [(0, 6), (6, 12), (12, 18), (18, None)],
            list("abcde"),
            None,
            False,
        ),
    ],
)
def test_len_colspecs_len_names(colspecs, names, widths, index_col):
    # GH#40830
    data = """col1  col2  col3  col4
    bab   ba    2"""
    msg = "Length of colspecs must match length of names"
    with pytest.raises(ValueError, match=msg):
        read_fwf(
            StringIO(data),
            colspecs=colspecs,
            names=names,
            widths=widths,
            index_col=index_col,
        )


@pytest.mark.parametrize(
    "colspecs, names, widths, index_col, expected",
    [
        (
            [(0, 6), (6, 12), (12, 18), (18, None)],
            list("abc"),
            None,
            0,
            DataFrame(
                index=["col1", "ba"],
                columns=["a", "b", "c"],
                data=[["col2", "col3", "col4"], ["b   ba", "2", np.nan]],
            ),
        ),
        (
            [(0, 6), (6, 12), (12, 18), (18, None)],
            list("ab"),
            None,
            [0, 1],
            DataFrame(
                index=[["col1", "ba"], ["col2", "b   ba"]],
                columns=["a", "b"],
                data=[["col3", "col4"], ["2", np.nan]],
            ),
        ),
        (
            [(0, 6), (6, 12), (12, 18), (18, None)],
            list("a"),
            None,
            [0, 1, 2],
            DataFrame(
                index=[["col1", "ba"], ["col2", "b   ba"], ["col3", "2"]],
                columns=["a"],
                data=[["col4"], [np.nan]],
            ),
        ),
        (
            None,
            list("abc"),
            [6] * 4,
            0,
            DataFrame(
                index=["col1", "ba"],
                columns=["a", "b", "c"],
                data=[["col2", "col3", "col4"], ["b   ba", "2", np.nan]],
            ),
        ),
        (
            None,
            list("ab"),
            [6] * 4,
            [0, 1],
            DataFrame(
                index=[["col1", "ba"], ["col2", "b   ba"]],
                columns=["a", "b"],
                data=[["col3", "col4"], ["2", np.nan]],
            ),
        ),
        (
            None,
            list("a"),
            [6] * 4,
            [0, 1, 2],
            DataFrame(
                index=[["col1", "ba"], ["col2", "b   ba"], ["col3", "2"]],
                columns=["a"],
                data=[["col4"], [np.nan]],
            ),
        ),
    ],
)
def test_len_colspecs_len_names_with_index_col(
    colspecs, names, widths, index_col, expected
):
    # GH#40830
    data = """col1  col2  col3  col4
    bab   ba    2"""
    result = read_fwf(
        StringIO(data),
        colspecs=colspecs,
        names=names,
        widths=widths,
        index_col=index_col,
    )
    tm.assert_frame_equal(result, expected)


def test_colspecs_with_comment():
    # GH 14135
    result = read_fwf(
        StringIO("#\nA1K\n"), colspecs=[(1, 2), (2, 3)], comment="#", header=None
    )
    expected = DataFrame([[1, "K"]], columns=[0, 1])
    tm.assert_frame_equal(result, expected)


def test_skip_rows_and_n_rows():
    # GH#44021
    data = """a\tb
1\t a
2\t b
3\t c
4\t d
5\t e
6\t f
    """
    result = read_fwf(StringIO(data), nrows=4, skiprows=[2, 4])
    expected = DataFrame({"a": [1, 3, 5, 6], "b": ["a", "c", "e", "f"]})
    tm.assert_frame_equal(result, expected)


def test_skiprows_with_iterator():
    # GH#10261, GH#56323
    data = """0
1
2
3
4
5
6
7
8
9
    """
    df_iter = read_fwf(
        StringIO(data),
        colspecs=[(0, 2)],
        names=["a"],
        iterator=True,
        chunksize=2,
        skiprows=[0, 1, 2, 6, 9],
    )
    expected_frames = [
        DataFrame({"a": [3, 4]}),
        DataFrame({"a": [5, 7]}, index=[2, 3]),
        DataFrame({"a": [8]}, index=[4]),
    ]
    for i, result in enumerate(df_iter):
        tm.assert_frame_equal(result, expected_frames[i])


def test_names_and_infer_colspecs():
    # GH#45337
    data = """X   Y   Z
      959.0    345   22.2
    """
    result = read_fwf(StringIO(data), skiprows=1, usecols=[0, 2], names=["a", "b"])
    expected = DataFrame({"a": [959.0], "b": 22.2})
    tm.assert_frame_equal(result, expected)


def test_widths_and_usecols():
    # GH#46580
    data = """0  1    n -0.4100.1
0  2    p  0.2 90.1
0  3    n -0.3140.4"""
    result = read_fwf(
        StringIO(data),
        header=None,
        usecols=(0, 1, 3),
        widths=(3, 5, 1, 5, 5),
        index_col=False,
        names=("c0", "c1", "c3"),
    )
    expected = DataFrame(
        {
            "c0": 0,
            "c1": [1, 2, 3],
            "c3": [-0.4, 0.2, -0.3],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_dtype_backend(string_storage, dtype_backend):
    # GH#50289
    data = """a  b    c      d  e     f  g    h  i
1  2.5  True  a
3  4.5  False b  True  6  7.5  a"""
    with pd.option_context("mode.string_storage", string_storage):
        result = read_fwf(StringIO(data), dtype_backend=dtype_backend)

    if dtype_backend == "pyarrow":
        pa = pytest.importorskip("pyarrow")
        string_dtype = pd.ArrowDtype(pa.string())
    else:
        string_dtype = pd.StringDtype(string_storage)

    expected = DataFrame(
        {
            "a": pd.Series([1, 3], dtype="Int64"),
            "b": pd.Series([2.5, 4.5], dtype="Float64"),
            "c": pd.Series([True, False], dtype="boolean"),
            "d": pd.Series(["a", "b"], dtype=string_dtype),
            "e": pd.Series([pd.NA, True], dtype="boolean"),
            "f": pd.Series([pd.NA, 6], dtype="Int64"),
            "g": pd.Series([pd.NA, 7.5], dtype="Float64"),
            "h": pd.Series([None, "a"], dtype=string_dtype),
            "i": pd.Series([pd.NA, pd.NA], dtype="Int64"),
        }
    )
    if dtype_backend == "pyarrow":
        pa = pytest.importorskip("pyarrow")
        from pandas.arrays import ArrowExtensionArray

        expected = DataFrame(
            {
                col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                for col in expected.columns
            }
        )
        expected["i"] = ArrowExtensionArray(pa.array([None, None]))

    # the storage of the str columns' Index is also affected by the
    # string_storage setting -> ignore that for checking the result
    tm.assert_frame_equal(result, expected, check_column_type=False)


def test_invalid_dtype_backend():
    msg = (
        "dtype_backend numpy is invalid, only 'numpy_nullable' and "
        "'pyarrow' are allowed."
    )
    with pytest.raises(ValueError, match=msg):
        read_fwf("test", dtype_backend="numpy")


@pytest.mark.network
@pytest.mark.single_cpu
def test_url_urlopen(httpserver):
    data = """\
A         B            C            D
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4
201160    364.136849   183.628767   11806.2
201161    413.836124   184.375703   11916.8
201162    502.953953   173.237159   12468.3
"""
    httpserver.serve_content(content=data)
    expected = pd.Index(list("ABCD"))
    with urlopen(httpserver.url) as f:
        result = read_fwf(f).columns

    tm.assert_index_equal(result, expected)