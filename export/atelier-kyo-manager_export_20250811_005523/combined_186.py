
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\test_to_latex.py ===
import codecs
from datetime import datetime
from textwrap import dedent

import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm

pytest.importorskip("jinja2")


def _dedent(string):
    """Dedent without new line in the beginning.

    Built-in textwrap.dedent would keep new line character in the beginning
    of multi-line string starting from the new line.
    This version drops the leading new line character.
    """
    return dedent(string).lstrip()


@pytest.fixture
def df_short():
    """Short dataframe for testing table/tabular/longtable LaTeX env."""
    return DataFrame({"a": [1, 2], "b": ["b1", "b2"]})


class TestToLatex:
    def test_to_latex_to_file(self, float_frame):
        with tm.ensure_clean("test.tex") as path:
            float_frame.to_latex(path)
            with open(path, encoding="utf-8") as f:
                assert float_frame.to_latex() == f.read()

    def test_to_latex_to_file_utf8_with_encoding(self):
        # test with utf-8 and encoding option (GH 7061)
        df = DataFrame([["au\xdfgangen"]])
        with tm.ensure_clean("test.tex") as path:
            df.to_latex(path, encoding="utf-8")
            with codecs.open(path, "r", encoding="utf-8") as f:
                assert df.to_latex() == f.read()

    def test_to_latex_to_file_utf8_without_encoding(self):
        # test with utf-8 without encoding option
        df = DataFrame([["au\xdfgangen"]])
        with tm.ensure_clean("test.tex") as path:
            df.to_latex(path)
            with codecs.open(path, "r", encoding="utf-8") as f:
                assert df.to_latex() == f.read()

    def test_to_latex_tabular_with_index(self):
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex()
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_tabular_without_index(self):
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(index=False)
        expected = _dedent(
            r"""
            \begin{tabular}{rl}
            \toprule
            a & b \\
            \midrule
            1 & b1 \\
            2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    @pytest.mark.parametrize(
        "bad_column_format",
        [5, 1.2, ["l", "r"], ("r", "c"), {"r", "c", "l"}, {"a": "r", "b": "l"}],
    )
    def test_to_latex_bad_column_format(self, bad_column_format):
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        msg = r"`column_format` must be str or unicode"
        with pytest.raises(ValueError, match=msg):
            df.to_latex(column_format=bad_column_format)

    def test_to_latex_column_format_just_works(self, float_frame):
        # GH Bug #9402
        float_frame.to_latex(column_format="lcr")

    def test_to_latex_column_format(self):
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(column_format="lcr")
        expected = _dedent(
            r"""
            \begin{tabular}{lcr}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_float_format_object_col(self):
        # GH#40024
        ser = Series([1000.0, "test"])
        result = ser.to_latex(float_format="{:,.0f}".format)
        expected = _dedent(
            r"""
            \begin{tabular}{ll}
            \toprule
             & 0 \\
            \midrule
            0 & 1,000 \\
            1 & test \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_empty_tabular(self):
        df = DataFrame()
        result = df.to_latex()
        expected = _dedent(
            r"""
            \begin{tabular}{l}
            \toprule
            \midrule
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_series(self):
        s = Series(["a", "b", "c"])
        result = s.to_latex()
        expected = _dedent(
            r"""
            \begin{tabular}{ll}
            \toprule
             & 0 \\
            \midrule
            0 & a \\
            1 & b \\
            2 & c \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_midrule_location(self):
        # GH 18326
        df = DataFrame({"a": [1, 2]})
        df.index.name = "foo"
        result = df.to_latex(index_names=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lr}
            \toprule
             & a \\
            \midrule
            0 & 1 \\
            1 & 2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_pos_args_deprecation(self):
        # GH-54229
        df = DataFrame(
            {
                "name": ["Raphael", "Donatello"],
                "age": [26, 45],
                "height": [181.23, 177.65],
            }
        )
        msg = (
            r"Starting with pandas version 3.0 all arguments of to_latex except for "
            r"the argument 'buf' will be keyword-only."
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df.to_latex(None, None)


class TestToLatexLongtable:
    def test_to_latex_empty_longtable(self):
        df = DataFrame()
        result = df.to_latex(longtable=True)
        expected = _dedent(
            r"""
            \begin{longtable}{l}
            \toprule
            \midrule
            \endfirsthead
            \toprule
            \midrule
            \endhead
            \midrule
            \multicolumn{0}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            \end{longtable}
            """
        )
        assert result == expected

    def test_to_latex_longtable_with_index(self):
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(longtable=True)
        expected = _dedent(
            r"""
            \begin{longtable}{lrl}
            \toprule
             & a & b \\
            \midrule
            \endfirsthead
            \toprule
             & a & b \\
            \midrule
            \endhead
            \midrule
            \multicolumn{3}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \end{longtable}
            """
        )
        assert result == expected

    def test_to_latex_longtable_without_index(self):
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(index=False, longtable=True)
        expected = _dedent(
            r"""
            \begin{longtable}{rl}
            \toprule
            a & b \\
            \midrule
            \endfirsthead
            \toprule
            a & b \\
            \midrule
            \endhead
            \midrule
            \multicolumn{2}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            1 & b1 \\
            2 & b2 \\
            \end{longtable}
            """
        )
        assert result == expected

    @pytest.mark.parametrize(
        "df, expected_number",
        [
            (DataFrame({"a": [1, 2]}), 1),
            (DataFrame({"a": [1, 2], "b": [3, 4]}), 2),
            (DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}), 3),
        ],
    )
    def test_to_latex_longtable_continued_on_next_page(self, df, expected_number):
        result = df.to_latex(index=False, longtable=True)
        assert rf"\multicolumn{{{expected_number}}}" in result


class TestToLatexHeader:
    def test_to_latex_no_header_with_index(self):
        # GH 7124
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(header=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_no_header_without_index(self):
        # GH 7124
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(index=False, header=False)
        expected = _dedent(
            r"""
            \begin{tabular}{rl}
            \toprule
            \midrule
            1 & b1 \\
            2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_specified_header_with_index(self):
        # GH 7124
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(header=["AA", "BB"])
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & AA & BB \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_specified_header_without_index(self):
        # GH 7124
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(header=["AA", "BB"], index=False)
        expected = _dedent(
            r"""
            \begin{tabular}{rl}
            \toprule
            AA & BB \\
            \midrule
            1 & b1 \\
            2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    @pytest.mark.parametrize(
        "header, num_aliases",
        [
            (["A"], 1),
            (("B",), 1),
            (("Col1", "Col2", "Col3"), 3),
            (("Col1", "Col2", "Col3", "Col4"), 4),
        ],
    )
    def test_to_latex_number_of_items_in_header_missmatch_raises(
        self,
        header,
        num_aliases,
    ):
        # GH 7124
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        msg = f"Writing 2 cols but got {num_aliases} aliases"
        with pytest.raises(ValueError, match=msg):
            df.to_latex(header=header)

    def test_to_latex_decimal(self):
        # GH 12031
        df = DataFrame({"a": [1.0, 2.1], "b": ["b1", "b2"]})
        result = df.to_latex(decimal=",")
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1,000000 & b1 \\
            1 & 2,100000 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected


class TestToLatexBold:
    def test_to_latex_bold_rows(self):
        # GH 16707
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(bold_rows=True)
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            \textbf{0} & 1 & b1 \\
            \textbf{1} & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_no_bold_rows(self):
        # GH 16707
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(bold_rows=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected


class TestToLatexCaptionLabel:
    @pytest.fixture
    def caption_table(self):
        """Caption for table/tabular LaTeX environment."""
        return "a table in a \\texttt{table/tabular} environment"

    @pytest.fixture
    def short_caption(self):
        """Short caption for testing \\caption[short_caption]{full_caption}."""
        return "a table"

    @pytest.fixture
    def label_table(self):
        """Label for table/tabular LaTeX environment."""
        return "tab:table_tabular"

    @pytest.fixture
    def caption_longtable(self):
        """Caption for longtable LaTeX environment."""
        return "a table in a \\texttt{longtable} environment"

    @pytest.fixture
    def label_longtable(self):
        """Label for longtable LaTeX environment."""
        return "tab:longtable"

    def test_to_latex_caption_only(self, df_short, caption_table):
        # GH 25436
        result = df_short.to_latex(caption=caption_table)
        expected = _dedent(
            r"""
            \begin{table}
            \caption{a table in a \texttt{table/tabular} environment}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        assert result == expected

    def test_to_latex_label_only(self, df_short, label_table):
        # GH 25436
        result = df_short.to_latex(label=label_table)
        expected = _dedent(
            r"""
            \begin{table}
            \label{tab:table_tabular}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        assert result == expected

    def test_to_latex_caption_and_label(self, df_short, caption_table, label_table):
        # GH 25436
        result = df_short.to_latex(caption=caption_table, label=label_table)
        expected = _dedent(
            r"""
            \begin{table}
            \caption{a table in a \texttt{table/tabular} environment}
            \label{tab:table_tabular}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        assert result == expected

    def test_to_latex_caption_and_shortcaption(
        self,
        df_short,
        caption_table,
        short_caption,
    ):
        result = df_short.to_latex(caption=(caption_table, short_caption))
        expected = _dedent(
            r"""
            \begin{table}
            \caption[a table]{a table in a \texttt{table/tabular} environment}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        assert result == expected

    def test_to_latex_caption_and_shortcaption_list_is_ok(self, df_short):
        caption = ("Long-long-caption", "Short")
        result_tuple = df_short.to_latex(caption=caption)
        result_list = df_short.to_latex(caption=list(caption))
        assert result_tuple == result_list

    def test_to_latex_caption_shortcaption_and_label(
        self,
        df_short,
        caption_table,
        short_caption,
        label_table,
    ):
        # test when the short_caption is provided alongside caption and label
        result = df_short.to_latex(
            caption=(caption_table, short_caption),
            label=label_table,
        )
        expected = _dedent(
            r"""
            \begin{table}
            \caption[a table]{a table in a \texttt{table/tabular} environment}
            \label{tab:table_tabular}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        assert result == expected

    @pytest.mark.parametrize(
        "bad_caption",
        [
            ("full_caption", "short_caption", "extra_string"),
            ("full_caption", "short_caption", 1),
            ("full_caption", "short_caption", None),
            ("full_caption",),
            (None,),
        ],
    )
    def test_to_latex_bad_caption_raises(self, bad_caption):
        # test that wrong number of params is raised
        df = DataFrame({"a": [1]})
        msg = "`caption` must be either a string or 2-tuple of strings"
        with pytest.raises(ValueError, match=msg):
            df.to_latex(caption=bad_caption)

    def test_to_latex_two_chars_caption(self, df_short):
        # test that two chars caption is handled correctly
        # it must not be unpacked into long_caption, short_caption.
        result = df_short.to_latex(caption="xy")
        expected = _dedent(
            r"""
            \begin{table}
            \caption{xy}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        assert result == expected

    def test_to_latex_longtable_caption_only(self, df_short, caption_longtable):
        # GH 25436
        # test when no caption and no label is provided
        # is performed by test_to_latex_longtable()
        result = df_short.to_latex(longtable=True, caption=caption_longtable)
        expected = _dedent(
            r"""
            \begin{longtable}{lrl}
            \caption{a table in a \texttt{longtable} environment} \\
            \toprule
             & a & b \\
            \midrule
            \endfirsthead
            \caption[]{a table in a \texttt{longtable} environment} \\
            \toprule
             & a & b \\
            \midrule
            \endhead
            \midrule
            \multicolumn{3}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \end{longtable}
            """
        )
        assert result == expected

    def test_to_latex_longtable_label_only(self, df_short, label_longtable):
        # GH 25436
        result = df_short.to_latex(longtable=True, label=label_longtable)
        expected = _dedent(
            r"""
            \begin{longtable}{lrl}
            \label{tab:longtable} \\
            \toprule
             & a & b \\
            \midrule
            \endfirsthead
            \toprule
             & a & b \\
            \midrule
            \endhead
            \midrule
            \multicolumn{3}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \end{longtable}
            """
        )
        assert result == expected

    def test_to_latex_longtable_caption_and_label(
        self,
        df_short,
        caption_longtable,
        label_longtable,
    ):
        # GH 25436
        result = df_short.to_latex(
            longtable=True,
            caption=caption_longtable,
            label=label_longtable,
        )
        expected = _dedent(
            r"""
        \begin{longtable}{lrl}
        \caption{a table in a \texttt{longtable} environment} \label{tab:longtable} \\
        \toprule
         & a & b \\
        \midrule
        \endfirsthead
        \caption[]{a table in a \texttt{longtable} environment} \\
        \toprule
         & a & b \\
        \midrule
        \endhead
        \midrule
        \multicolumn{3}{r}{Continued on next page} \\
        \midrule
        \endfoot
        \bottomrule
        \endlastfoot
        0 & 1 & b1 \\
        1 & 2 & b2 \\
        \end{longtable}
        """
        )
        assert result == expected

    def test_to_latex_longtable_caption_shortcaption_and_label(
        self,
        df_short,
        caption_longtable,
        short_caption,
        label_longtable,
    ):
        # test when the caption, the short_caption and the label are provided
        result = df_short.to_latex(
            longtable=True,
            caption=(caption_longtable, short_caption),
            label=label_longtable,
        )
        expected = _dedent(
            r"""
\begin{longtable}{lrl}
\caption[a table]{a table in a \texttt{longtable} environment} \label{tab:longtable} \\
\toprule
 & a & b \\
\midrule
\endfirsthead
\caption[]{a table in a \texttt{longtable} environment} \\
\toprule
 & a & b \\
\midrule
\endhead
\midrule
\multicolumn{3}{r}{Continued on next page} \\
\midrule
\endfoot
\bottomrule
\endlastfoot
0 & 1 & b1 \\
1 & 2 & b2 \\
\end{longtable}
"""
        )
        assert result == expected


class TestToLatexEscape:
    @pytest.fixture
    def df_with_symbols(self):
        """Dataframe with special characters for testing chars escaping."""
        a = "a"
        b = "b"
        yield DataFrame({"co$e^x$": {a: "a", b: "b"}, "co^l1": {a: "a", b: "b"}})

    def test_to_latex_escape_false(self, df_with_symbols):
        result = df_with_symbols.to_latex(escape=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lll}
            \toprule
             & co$e^x$ & co^l1 \\
            \midrule
            a & a & a \\
            b & b & b \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_escape_default(self, df_with_symbols):
        # gh50871: in v2.0 escape is False by default (styler.format.escape=None)
        default = df_with_symbols.to_latex()
        specified_true = df_with_symbols.to_latex(escape=True)
        assert default != specified_true

    def test_to_latex_special_escape(self):
        df = DataFrame([r"a\b\c", r"^a^b^c", r"~a~b~c"])
        result = df.to_latex(escape=True)
        expected = _dedent(
            r"""
            \begin{tabular}{ll}
            \toprule
             & 0 \\
            \midrule
            0 & a\textbackslash b\textbackslash c \\
            1 & \textasciicircum a\textasciicircum b\textasciicircum c \\
            2 & \textasciitilde a\textasciitilde b\textasciitilde c \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_escape_special_chars(self):
        special_characters = ["&", "%", "$", "#", "_", "{", "}", "~", "^", "\\"]
        df = DataFrame(data=special_characters)
        result = df.to_latex(escape=True)
        expected = _dedent(
            r"""
            \begin{tabular}{ll}
            \toprule
             & 0 \\
            \midrule
            0 & \& \\
            1 & \% \\
            2 & \$ \\
            3 & \# \\
            4 & \_ \\
            5 & \{ \\
            6 & \} \\
            7 & \textasciitilde  \\
            8 & \textasciicircum  \\
            9 & \textbackslash  \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_specified_header_special_chars_without_escape(self):
        # GH 7124
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(header=["$A$", "$B$"], escape=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & $A$ & $B$ \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected


class TestToLatexPosition:
    def test_to_latex_position(self):
        the_position = "h"
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(position=the_position)
        expected = _dedent(
            r"""
            \begin{table}[h]
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        assert result == expected

    def test_to_latex_longtable_position(self):
        the_position = "t"
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(longtable=True, position=the_position)
        expected = _dedent(
            r"""
            \begin{longtable}[t]{lrl}
            \toprule
             & a & b \\
            \midrule
            \endfirsthead
            \toprule
             & a & b \\
            \midrule
            \endhead
            \midrule
            \multicolumn{3}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \end{longtable}
            """
        )
        assert result == expected


class TestToLatexFormatters:
    def test_to_latex_with_formatters(self):
        df = DataFrame(
            {
                "datetime64": [
                    datetime(2016, 1, 1),
                    datetime(2016, 2, 5),
                    datetime(2016, 3, 3),
                ],
                "float": [1.0, 2.0, 3.0],
                "int": [1, 2, 3],
                "object": [(1, 2), True, False],
            }
        )

        formatters = {
            "datetime64": lambda x: x.strftime("%Y-%m"),
            "float": lambda x: f"[{x: 4.1f}]",
            "int": lambda x: f"0x{x:x}",
            "object": lambda x: f"-{x!s}-",
            "__index__": lambda x: f"index: {x}",
        }
        result = df.to_latex(formatters=dict(formatters))

        expected = _dedent(
            r"""
            \begin{tabular}{llrrl}
            \toprule
             & datetime64 & float & int & object \\
            \midrule
            index: 0 & 2016-01 & [ 1.0] & 0x1 & -(1, 2)- \\
            index: 1 & 2016-02 & [ 2.0] & 0x2 & -True- \\
            index: 2 & 2016-03 & [ 3.0] & 0x3 & -False- \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_float_format_no_fixed_width_3decimals(self):
        # GH 21625
        df = DataFrame({"x": [0.19999]})
        result = df.to_latex(float_format="%.3f")
        expected = _dedent(
            r"""
            \begin{tabular}{lr}
            \toprule
             & x \\
            \midrule
            0 & 0.200 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_float_format_no_fixed_width_integer(self):
        # GH 22270
        df = DataFrame({"x": [100.0]})
        result = df.to_latex(float_format="%.0f")
        expected = _dedent(
            r"""
            \begin{tabular}{lr}
            \toprule
             & x \\
            \midrule
            0 & 100 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    @pytest.mark.parametrize("na_rep", ["NaN", "Ted"])
    def test_to_latex_na_rep_and_float_format(self, na_rep):
        df = DataFrame(
            [
                ["A", 1.2225],
                ["A", None],
            ],
            columns=["Group", "Data"],
        )
        result = df.to_latex(na_rep=na_rep, float_format="{:.2f}".format)
        expected = _dedent(
            rf"""
            \begin{{tabular}}{{llr}}
            \toprule
             & Group & Data \\
            \midrule
            0 & A & 1.22 \\
            1 & A & {na_rep} \\
            \bottomrule
            \end{{tabular}}
            """
        )
        assert result == expected


class TestToLatexMultiindex:
    @pytest.fixture
    def multiindex_frame(self):
        """Multiindex dataframe for testing multirow LaTeX macros."""
        yield DataFrame.from_dict(
            {
                ("c1", 0): Series({x: x for x in range(4)}),
                ("c1", 1): Series({x: x + 4 for x in range(4)}),
                ("c2", 0): Series({x: x for x in range(4)}),
                ("c2", 1): Series({x: x + 4 for x in range(4)}),
                ("c3", 0): Series({x: x for x in range(4)}),
            }
        ).T

    @pytest.fixture
    def multicolumn_frame(self):
        """Multicolumn dataframe for testing multicolumn LaTeX macros."""
        yield DataFrame(
            {
                ("c1", 0): {x: x for x in range(5)},
                ("c1", 1): {x: x + 5 for x in range(5)},
                ("c2", 0): {x: x for x in range(5)},
                ("c2", 1): {x: x + 5 for x in range(5)},
                ("c3", 0): {x: x for x in range(5)},
            }
        )

    def test_to_latex_multindex_header(self):
        # GH 16718
        df = DataFrame({"a": [0], "b": [1], "c": [2], "d": [3]})
        df = df.set_index(["a", "b"])
        observed = df.to_latex(header=["r1", "r2"], multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{llrr}
            \toprule
             &  & r1 & r2 \\
            a & b &  &  \\
            \midrule
            0 & 1 & 2 & 3 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert observed == expected

    def test_to_latex_multiindex_empty_name(self):
        # GH 18669
        mi = pd.MultiIndex.from_product([[1, 2]], names=[""])
        df = DataFrame(-1, index=mi, columns=range(4))
        observed = df.to_latex()
        expected = _dedent(
            r"""
            \begin{tabular}{lrrrr}
            \toprule
             & 0 & 1 & 2 & 3 \\
             &  &  &  &  \\
            \midrule
            1 & -1 & -1 & -1 & -1 \\
            2 & -1 & -1 & -1 & -1 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert observed == expected

    def test_to_latex_multiindex_column_tabular(self):
        df = DataFrame({("x", "y"): ["a"]})
        result = df.to_latex()
        expected = _dedent(
            r"""
            \begin{tabular}{ll}
            \toprule
             & x \\
             & y \\
            \midrule
            0 & a \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multiindex_small_tabular(self):
        df = DataFrame({("x", "y"): ["a"]}).T
        result = df.to_latex(multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lll}
            \toprule
             &  & 0 \\
            \midrule
            x & y & a \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multiindex_tabular(self, multiindex_frame):
        result = multiindex_frame.to_latex(multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{llrrrr}
            \toprule
             &  & 0 & 1 & 2 & 3 \\
            \midrule
            c1 & 0 & 0 & 1 & 2 & 3 \\
             & 1 & 4 & 5 & 6 & 7 \\
            c2 & 0 & 0 & 1 & 2 & 3 \\
             & 1 & 4 & 5 & 6 & 7 \\
            c3 & 0 & 0 & 1 & 2 & 3 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multicolumn_tabular(self, multiindex_frame):
        # GH 14184
        df = multiindex_frame.T
        df.columns.names = ["a", "b"]
        result = df.to_latex(multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lrrrrr}
            \toprule
            a & \multicolumn{2}{r}{c1} & \multicolumn{2}{r}{c2} & c3 \\
            b & 0 & 1 & 0 & 1 & 0 \\
            \midrule
            0 & 0 & 4 & 0 & 4 & 0 \\
            1 & 1 & 5 & 1 & 5 & 1 \\
            2 & 2 & 6 & 2 & 6 & 2 \\
            3 & 3 & 7 & 3 & 7 & 3 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_index_has_name_tabular(self):
        # GH 10660
        df = DataFrame({"a": [0, 0, 1, 1], "b": list("abab"), "c": [1, 2, 3, 4]})
        result = df.set_index(["a", "b"]).to_latex(multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{llr}
            \toprule
             &  & c \\
            a & b &  \\
            \midrule
            0 & a & 1 \\
             & b & 2 \\
            1 & a & 3 \\
             & b & 4 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_groupby_tabular(self):
        # GH 10660
        df = DataFrame({"a": [0, 0, 1, 1], "b": list("abab"), "c": [1, 2, 3, 4]})
        result = (
            df.groupby("a")
            .describe()
            .to_latex(float_format="{:.1f}".format, escape=True)
        )
        expected = _dedent(
            r"""
            \begin{tabular}{lrrrrrrrr}
            \toprule
             & \multicolumn{8}{r}{c} \\
             & count & mean & std & min & 25\% & 50\% & 75\% & max \\
            a &  &  &  &  &  &  &  &  \\
            \midrule
            0 & 2.0 & 1.5 & 0.7 & 1.0 & 1.2 & 1.5 & 1.8 & 2.0 \\
            1 & 2.0 & 3.5 & 0.7 & 3.0 & 3.2 & 3.5 & 3.8 & 4.0 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multiindex_dupe_level(self):
        # see gh-14484
        #
        # If an index is repeated in subsequent rows, it should be
        # replaced with a blank in the created table. This should
        # ONLY happen if all higher order indices (to the left) are
        # equal too. In this test, 'c' has to be printed both times
        # because the higher order index 'A' != 'B'.
        df = DataFrame(
            index=pd.MultiIndex.from_tuples([("A", "c"), ("B", "c")]), columns=["col"]
        )
        result = df.to_latex(multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lll}
            \toprule
             &  & col \\
            \midrule
            A & c & NaN \\
            B & c & NaN \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multicolumn_default(self, multicolumn_frame):
        result = multicolumn_frame.to_latex()
        expected = _dedent(
            r"""
            \begin{tabular}{lrrrrr}
            \toprule
             & \multicolumn{2}{r}{c1} & \multicolumn{2}{r}{c2} & c3 \\
             & 0 & 1 & 0 & 1 & 0 \\
            \midrule
            0 & 0 & 5 & 0 & 5 & 0 \\
            1 & 1 & 6 & 1 & 6 & 1 \\
            2 & 2 & 7 & 2 & 7 & 2 \\
            3 & 3 & 8 & 3 & 8 & 3 \\
            4 & 4 & 9 & 4 & 9 & 4 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multicolumn_false(self, multicolumn_frame):
        result = multicolumn_frame.to_latex(multicolumn=False, multicolumn_format="l")
        expected = _dedent(
            r"""
            \begin{tabular}{lrrrrr}
            \toprule
             & c1 & & c2 & & c3 \\
             & 0 & 1 & 0 & 1 & 0 \\
            \midrule
            0 & 0 & 5 & 0 & 5 & 0 \\
            1 & 1 & 6 & 1 & 6 & 1 \\
            2 & 2 & 7 & 2 & 7 & 2 \\
            3 & 3 & 8 & 3 & 8 & 3 \\
            4 & 4 & 9 & 4 & 9 & 4 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multirow_true(self, multicolumn_frame):
        result = multicolumn_frame.T.to_latex(multirow=True)
        expected = _dedent(
            r"""
            \begin{tabular}{llrrrrr}
            \toprule
             &  & 0 & 1 & 2 & 3 & 4 \\
            \midrule
            \multirow[t]{2}{*}{c1} & 0 & 0 & 1 & 2 & 3 & 4 \\
             & 1 & 5 & 6 & 7 & 8 & 9 \\
            \cline{1-7}
            \multirow[t]{2}{*}{c2} & 0 & 0 & 1 & 2 & 3 & 4 \\
             & 1 & 5 & 6 & 7 & 8 & 9 \\
            \cline{1-7}
            c3 & 0 & 0 & 1 & 2 & 3 & 4 \\
            \cline{1-7}
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multicolumnrow_with_multicol_format(self, multicolumn_frame):
        multicolumn_frame.index = multicolumn_frame.T.index
        result = multicolumn_frame.T.to_latex(
            multirow=True,
            multicolumn=True,
            multicolumn_format="c",
        )
        expected = _dedent(
            r"""
            \begin{tabular}{llrrrrr}
            \toprule
             &  & \multicolumn{2}{c}{c1} & \multicolumn{2}{c}{c2} & c3 \\
             &  & 0 & 1 & 0 & 1 & 0 \\
            \midrule
            \multirow[t]{2}{*}{c1} & 0 & 0 & 1 & 2 & 3 & 4 \\
             & 1 & 5 & 6 & 7 & 8 & 9 \\
            \cline{1-7}
            \multirow[t]{2}{*}{c2} & 0 & 0 & 1 & 2 & 3 & 4 \\
             & 1 & 5 & 6 & 7 & 8 & 9 \\
            \cline{1-7}
            c3 & 0 & 0 & 1 & 2 & 3 & 4 \\
            \cline{1-7}
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    @pytest.mark.parametrize("name0", [None, "named0"])
    @pytest.mark.parametrize("name1", [None, "named1"])
    @pytest.mark.parametrize("axes", [[0], [1], [0, 1]])
    def test_to_latex_multiindex_names(self, name0, name1, axes):
        # GH 18667
        names = [name0, name1]
        mi = pd.MultiIndex.from_product([[1, 2], [3, 4]])
        df = DataFrame(-1, index=mi.copy(), columns=mi.copy())
        for idx in axes:
            df.axes[idx].names = names

        idx_names = tuple(n or "" for n in names)
        idx_names_row = (
            f"{idx_names[0]} & {idx_names[1]} &  &  &  &  \\\\\n"
            if (0 in axes and any(names))
            else ""
        )
        col_names = [n if (bool(n) and 1 in axes) else "" for n in names]
        observed = df.to_latex(multirow=False)
        # pylint: disable-next=consider-using-f-string
        expected = r"""\begin{tabular}{llrrrr}
\toprule
 & %s & \multicolumn{2}{r}{1} & \multicolumn{2}{r}{2} \\
 & %s & 3 & 4 & 3 & 4 \\
%s\midrule
1 & 3 & -1 & -1 & -1 & -1 \\
 & 4 & -1 & -1 & -1 & -1 \\
2 & 3 & -1 & -1 & -1 & -1 \\
 & 4 & -1 & -1 & -1 & -1 \\
\bottomrule
\end{tabular}
""" % tuple(
            list(col_names) + [idx_names_row]
        )
        assert observed == expected

    @pytest.mark.parametrize("one_row", [True, False])
    def test_to_latex_multiindex_nans(self, one_row):
        # GH 14249
        df = DataFrame({"a": [None, 1], "b": [2, 3], "c": [4, 5]})
        if one_row:
            df = df.iloc[[0]]
        observed = df.set_index(["a", "b"]).to_latex(multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{llr}
            \toprule
             &  & c \\
            a & b &  \\
            \midrule
            NaN & 2 & 4 \\
            """
        )
        if not one_row:
            expected += r"""1.000000 & 3 & 5 \\
"""
        expected += r"""\bottomrule
\end{tabular}
"""
        assert observed == expected

    def test_to_latex_non_string_index(self):
        # GH 19981
        df = DataFrame([[1, 2, 3]] * 2).set_index([0, 1])
        result = df.to_latex(multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{llr}
            \toprule
             &  & 2 \\
            0 & 1 &  \\
            \midrule
            1 & 2 & 3 \\
             & 2 & 3 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multiindex_multirow(self):
        # GH 16719
        mi = pd.MultiIndex.from_product(
            [[0.0, 1.0], [3.0, 2.0, 1.0], ["0", "1"]], names=["i", "val0", "val1"]
        )
        df = DataFrame(index=mi)
        result = df.to_latex(multirow=True, escape=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lll}
            \toprule
            i & val0 & val1 \\
            \midrule
            \multirow[t]{6}{*}{0.000000} & \multirow[t]{2}{*}{3.000000} & 0 \\
             &  & 1 \\
            \cline{2-3}
             & \multirow[t]{2}{*}{2.000000} & 0 \\
             &  & 1 \\
            \cline{2-3}
             & \multirow[t]{2}{*}{1.000000} & 0 \\
             &  & 1 \\
            \cline{1-3} \cline{2-3}
            \multirow[t]{6}{*}{1.000000} & \multirow[t]{2}{*}{3.000000} & 0 \\
             &  & 1 \\
            \cline{2-3}
             & \multirow[t]{2}{*}{2.000000} & 0 \\
             &  & 1 \\
            \cline{2-3}
             & \multirow[t]{2}{*}{1.000000} & 0 \\
             &  & 1 \\
            \cline{1-3} \cline{2-3}
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\internals\test_internals.py ===
from datetime import (
    date,
    datetime,
)
import itertools
import re

import numpy as np
import pytest

from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
import pandas.util._test_decorators as td

from pandas.core.dtypes.common import is_scalar

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalIndex,
    Series,
    Timedelta,
    Timestamp,
    period_range,
)
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
    DatetimeArray,
    SparseArray,
    TimedeltaArray,
)
from pandas.core.internals import (
    BlockManager,
    SingleBlockManager,
    make_block,
)
from pandas.core.internals.blocks import (
    ensure_block_shape,
    maybe_coerce_values,
    new_block,
)

# this file contains BlockManager specific tests
# TODO(ArrayManager) factor out interleave_dtype tests
pytestmark = td.skip_array_manager_invalid_test


@pytest.fixture(params=[new_block, make_block])
def block_maker(request):
    """
    Fixture to test both the internal new_block and pseudo-public make_block.
    """
    return request.param


@pytest.fixture
def mgr():
    return create_mgr(
        "a: f8; b: object; c: f8; d: object; e: f8;"
        "f: bool; g: i8; h: complex; i: datetime-1; j: datetime-2;"
        "k: M8[ns, US/Eastern]; l: M8[ns, CET];"
    )


def assert_block_equal(left, right):
    tm.assert_numpy_array_equal(left.values, right.values)
    assert left.dtype == right.dtype
    assert isinstance(left.mgr_locs, BlockPlacement)
    assert isinstance(right.mgr_locs, BlockPlacement)
    tm.assert_numpy_array_equal(left.mgr_locs.as_array, right.mgr_locs.as_array)


def get_numeric_mat(shape):
    arr = np.arange(shape[0])
    return np.lib.stride_tricks.as_strided(
        x=arr, shape=shape, strides=(arr.itemsize,) + (0,) * (len(shape) - 1)
    ).copy()


N = 10


def create_block(typestr, placement, item_shape=None, num_offset=0, maker=new_block):
    """
    Supported typestr:

        * float, f8, f4, f2
        * int, i8, i4, i2, i1
        * uint, u8, u4, u2, u1
        * complex, c16, c8
        * bool
        * object, string, O
        * datetime, dt, M8[ns], M8[ns, tz]
        * timedelta, td, m8[ns]
        * sparse (SparseArray with fill_value=0.0)
        * sparse_na (SparseArray with fill_value=np.nan)
        * category, category2

    """
    placement = BlockPlacement(placement)
    num_items = len(placement)

    if item_shape is None:
        item_shape = (N,)

    shape = (num_items,) + item_shape

    mat = get_numeric_mat(shape)

    if typestr in (
        "float",
        "f8",
        "f4",
        "f2",
        "int",
        "i8",
        "i4",
        "i2",
        "i1",
        "uint",
        "u8",
        "u4",
        "u2",
        "u1",
    ):
        values = mat.astype(typestr) + num_offset
    elif typestr in ("complex", "c16", "c8"):
        values = 1.0j * (mat.astype(typestr) + num_offset)
    elif typestr in ("object", "string", "O"):
        values = np.reshape([f"A{i:d}" for i in mat.ravel() + num_offset], shape)
    elif typestr in ("b", "bool"):
        values = np.ones(shape, dtype=np.bool_)
    elif typestr in ("datetime", "dt", "M8[ns]"):
        values = (mat * 1e9).astype("M8[ns]")
    elif typestr.startswith("M8[ns"):
        # datetime with tz
        m = re.search(r"M8\[ns,\s*(\w+\/?\w*)\]", typestr)
        assert m is not None, f"incompatible typestr -> {typestr}"
        tz = m.groups()[0]
        assert num_items == 1, "must have only 1 num items for a tz-aware"
        values = DatetimeIndex(np.arange(N) * 10**9, tz=tz)._data
        values = ensure_block_shape(values, ndim=len(shape))
    elif typestr in ("timedelta", "td", "m8[ns]"):
        values = (mat * 1).astype("m8[ns]")
    elif typestr in ("category",):
        values = Categorical([1, 1, 2, 2, 3, 3, 3, 3, 4, 4])
    elif typestr in ("category2",):
        values = Categorical(["a", "a", "a", "a", "b", "b", "c", "c", "c", "d"])
    elif typestr in ("sparse", "sparse_na"):
        if shape[-1] != 10:
            # We also are implicitly assuming this in the category cases above
            raise NotImplementedError

        assert all(s == 1 for s in shape[:-1])
        if typestr.endswith("_na"):
            fill_value = np.nan
        else:
            fill_value = 0.0
        values = SparseArray(
            [fill_value, fill_value, 1, 2, 3, fill_value, 4, 5, fill_value, 6],
            fill_value=fill_value,
        )
        arr = values.sp_values.view()
        arr += num_offset - 1
    else:
        raise ValueError(f'Unsupported typestr: "{typestr}"')

    values = maybe_coerce_values(values)
    return maker(values, placement=placement, ndim=len(shape))


def create_single_mgr(typestr, num_rows=None):
    if num_rows is None:
        num_rows = N

    return SingleBlockManager(
        create_block(typestr, placement=slice(0, num_rows), item_shape=()),
        Index(np.arange(num_rows)),
    )


def create_mgr(descr, item_shape=None):
    """
    Construct BlockManager from string description.

    String description syntax looks similar to np.matrix initializer.  It looks
    like this::

        a,b,c: f8; d,e,f: i8

    Rules are rather simple:

    * see list of supported datatypes in `create_block` method
    * components are semicolon-separated
    * each component is `NAME,NAME,NAME: DTYPE_ID`
    * whitespace around colons & semicolons are removed
    * components with same DTYPE_ID are combined into single block
    * to force multiple blocks with same dtype, use '-SUFFIX'::

        'a:f8-1; b:f8-2; c:f8-foobar'

    """
    if item_shape is None:
        item_shape = (N,)

    offset = 0
    mgr_items = []
    block_placements = {}
    for d in descr.split(";"):
        d = d.strip()
        if not len(d):
            continue
        names, blockstr = d.partition(":")[::2]
        blockstr = blockstr.strip()
        names = names.strip().split(",")

        mgr_items.extend(names)
        placement = list(np.arange(len(names)) + offset)
        try:
            block_placements[blockstr].extend(placement)
        except KeyError:
            block_placements[blockstr] = placement
        offset += len(names)

    mgr_items = Index(mgr_items)

    blocks = []
    num_offset = 0
    for blockstr, placement in block_placements.items():
        typestr = blockstr.split("-")[0]
        blocks.append(
            create_block(
                typestr, placement, item_shape=item_shape, num_offset=num_offset
            )
        )
        num_offset += len(placement)

    sblocks = sorted(blocks, key=lambda b: b.mgr_locs[0])
    return BlockManager(
        tuple(sblocks),
        [mgr_items] + [Index(np.arange(n)) for n in item_shape],
    )


@pytest.fixture
def fblock():
    return create_block("float", [0, 2, 4])


class TestBlock:
    def test_constructor(self):
        int32block = create_block("i4", [0])
        assert int32block.dtype == np.int32

    @pytest.mark.parametrize(
        "typ, data",
        [
            ["float", [0, 2, 4]],
            ["complex", [7]],
            ["object", [1, 3]],
            ["bool", [5]],
        ],
    )
    def test_pickle(self, typ, data):
        blk = create_block(typ, data)
        assert_block_equal(tm.round_trip_pickle(blk), blk)

    def test_mgr_locs(self, fblock):
        assert isinstance(fblock.mgr_locs, BlockPlacement)
        tm.assert_numpy_array_equal(
            fblock.mgr_locs.as_array, np.array([0, 2, 4], dtype=np.intp)
        )

    def test_attrs(self, fblock):
        assert fblock.shape == fblock.values.shape
        assert fblock.dtype == fblock.values.dtype
        assert len(fblock) == len(fblock.values)

    def test_copy(self, fblock):
        cop = fblock.copy()
        assert cop is not fblock
        assert_block_equal(fblock, cop)

    def test_delete(self, fblock):
        newb = fblock.copy()
        locs = newb.mgr_locs
        nb = newb.delete(0)[0]
        assert newb.mgr_locs is locs

        assert nb is not newb

        tm.assert_numpy_array_equal(
            nb.mgr_locs.as_array, np.array([2, 4], dtype=np.intp)
        )
        assert not (newb.values[0] == 1).all()
        assert (nb.values[0] == 1).all()

        newb = fblock.copy()
        locs = newb.mgr_locs
        nb = newb.delete(1)
        assert len(nb) == 2
        assert newb.mgr_locs is locs

        tm.assert_numpy_array_equal(
            nb[0].mgr_locs.as_array, np.array([0], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            nb[1].mgr_locs.as_array, np.array([4], dtype=np.intp)
        )
        assert not (newb.values[1] == 2).all()
        assert (nb[1].values[0] == 2).all()

        newb = fblock.copy()
        nb = newb.delete(2)
        assert len(nb) == 1
        tm.assert_numpy_array_equal(
            nb[0].mgr_locs.as_array, np.array([0, 2], dtype=np.intp)
        )
        assert (nb[0].values[1] == 1).all()

        newb = fblock.copy()

        with pytest.raises(IndexError, match=None):
            newb.delete(3)

    def test_delete_datetimelike(self):
        # dont use np.delete on values, as that will coerce from DTA/TDA to ndarray
        arr = np.arange(20, dtype="i8").reshape(5, 4).view("m8[ns]")
        df = DataFrame(arr)
        blk = df._mgr.blocks[0]
        assert isinstance(blk.values, TimedeltaArray)

        nb = blk.delete(1)
        assert len(nb) == 2
        assert isinstance(nb[0].values, TimedeltaArray)
        assert isinstance(nb[1].values, TimedeltaArray)

        df = DataFrame(arr.view("M8[ns]"))
        blk = df._mgr.blocks[0]
        assert isinstance(blk.values, DatetimeArray)

        nb = blk.delete([1, 3])
        assert len(nb) == 2
        assert isinstance(nb[0].values, DatetimeArray)
        assert isinstance(nb[1].values, DatetimeArray)

    def test_split(self):
        # GH#37799
        values = np.random.default_rng(2).standard_normal((3, 4))
        blk = new_block(values, placement=BlockPlacement([3, 1, 6]), ndim=2)
        result = blk._split()

        # check that we get views, not copies
        values[:] = -9999
        assert (blk.values == -9999).all()

        assert len(result) == 3
        expected = [
            new_block(values[[0]], placement=BlockPlacement([3]), ndim=2),
            new_block(values[[1]], placement=BlockPlacement([1]), ndim=2),
            new_block(values[[2]], placement=BlockPlacement([6]), ndim=2),
        ]
        for res, exp in zip(result, expected):
            assert_block_equal(res, exp)


class TestBlockManager:
    def test_attrs(self):
        mgr = create_mgr("a,b,c: f8-1; d,e,f: f8-2")
        assert mgr.nblocks == 2
        assert len(mgr) == 6

    def test_duplicate_ref_loc_failure(self):
        tmp_mgr = create_mgr("a:bool; a: f8")

        axes, blocks = tmp_mgr.axes, tmp_mgr.blocks

        blocks[0].mgr_locs = BlockPlacement(np.array([0]))
        blocks[1].mgr_locs = BlockPlacement(np.array([0]))

        # test trying to create block manager with overlapping ref locs

        msg = "Gaps in blk ref_locs"

        with pytest.raises(AssertionError, match=msg):
            mgr = BlockManager(blocks, axes)
            mgr._rebuild_blknos_and_blklocs()

        blocks[0].mgr_locs = BlockPlacement(np.array([0]))
        blocks[1].mgr_locs = BlockPlacement(np.array([1]))
        mgr = BlockManager(blocks, axes)
        mgr.iget(1)

    def test_pickle(self, mgr):
        mgr2 = tm.round_trip_pickle(mgr)
        tm.assert_frame_equal(
            DataFrame._from_mgr(mgr, axes=mgr.axes),
            DataFrame._from_mgr(mgr2, axes=mgr2.axes),
        )

        # GH2431
        assert hasattr(mgr2, "_is_consolidated")
        assert hasattr(mgr2, "_known_consolidated")

        # reset to False on load
        assert not mgr2._is_consolidated
        assert not mgr2._known_consolidated

    @pytest.mark.parametrize("mgr_string", ["a,a,a:f8", "a: f8; a: i8"])
    def test_non_unique_pickle(self, mgr_string):
        mgr = create_mgr(mgr_string)
        mgr2 = tm.round_trip_pickle(mgr)
        tm.assert_frame_equal(
            DataFrame._from_mgr(mgr, axes=mgr.axes),
            DataFrame._from_mgr(mgr2, axes=mgr2.axes),
        )

    def test_categorical_block_pickle(self):
        mgr = create_mgr("a: category")
        mgr2 = tm.round_trip_pickle(mgr)
        tm.assert_frame_equal(
            DataFrame._from_mgr(mgr, axes=mgr.axes),
            DataFrame._from_mgr(mgr2, axes=mgr2.axes),
        )

        smgr = create_single_mgr("category")
        smgr2 = tm.round_trip_pickle(smgr)
        tm.assert_series_equal(
            Series()._constructor_from_mgr(smgr, axes=smgr.axes),
            Series()._constructor_from_mgr(smgr2, axes=smgr2.axes),
        )

    def test_iget(self):
        cols = Index(list("abc"))
        values = np.random.default_rng(2).random((3, 3))
        block = new_block(
            values=values.copy(),
            placement=BlockPlacement(np.arange(3, dtype=np.intp)),
            ndim=values.ndim,
        )
        mgr = BlockManager(blocks=(block,), axes=[cols, Index(np.arange(3))])

        tm.assert_almost_equal(mgr.iget(0).internal_values(), values[0])
        tm.assert_almost_equal(mgr.iget(1).internal_values(), values[1])
        tm.assert_almost_equal(mgr.iget(2).internal_values(), values[2])

    def test_set(self):
        mgr = create_mgr("a,b,c: int", item_shape=(3,))

        mgr.insert(len(mgr.items), "d", np.array(["foo"] * 3))
        mgr.iset(1, np.array(["bar"] * 3))
        tm.assert_numpy_array_equal(mgr.iget(0).internal_values(), np.array([0] * 3))
        tm.assert_numpy_array_equal(
            mgr.iget(1).internal_values(), np.array(["bar"] * 3, dtype=np.object_)
        )
        tm.assert_numpy_array_equal(mgr.iget(2).internal_values(), np.array([2] * 3))
        tm.assert_numpy_array_equal(
            mgr.iget(3).internal_values(), np.array(["foo"] * 3, dtype=np.object_)
        )

    def test_set_change_dtype(self, mgr):
        mgr.insert(len(mgr.items), "baz", np.zeros(N, dtype=bool))

        mgr.iset(mgr.items.get_loc("baz"), np.repeat("foo", N))
        idx = mgr.items.get_loc("baz")
        assert mgr.iget(idx).dtype == np.object_

        mgr2 = mgr.consolidate()
        mgr2.iset(mgr2.items.get_loc("baz"), np.repeat("foo", N))
        idx = mgr2.items.get_loc("baz")
        assert mgr2.iget(idx).dtype == np.object_

        mgr2.insert(
            len(mgr2.items),
            "quux",
            np.random.default_rng(2).standard_normal(N).astype(int),
        )
        idx = mgr2.items.get_loc("quux")
        assert mgr2.iget(idx).dtype == np.dtype(int)

        mgr2.iset(
            mgr2.items.get_loc("quux"), np.random.default_rng(2).standard_normal(N)
        )
        assert mgr2.iget(idx).dtype == np.float64

    def test_copy(self, mgr):
        cp = mgr.copy(deep=False)
        for blk, cp_blk in zip(mgr.blocks, cp.blocks):
            # view assertion
            tm.assert_equal(cp_blk.values, blk.values)
            if isinstance(blk.values, np.ndarray):
                assert cp_blk.values.base is blk.values.base
            else:
                # DatetimeTZBlock has DatetimeIndex values
                assert cp_blk.values._ndarray.base is blk.values._ndarray.base

        # copy(deep=True) consolidates, so the block-wise assertions will
        #  fail is mgr is not consolidated
        mgr._consolidate_inplace()
        cp = mgr.copy(deep=True)
        for blk, cp_blk in zip(mgr.blocks, cp.blocks):
            bvals = blk.values
            cpvals = cp_blk.values

            tm.assert_equal(cpvals, bvals)

            if isinstance(cpvals, np.ndarray):
                lbase = cpvals.base
                rbase = bvals.base
            else:
                lbase = cpvals._ndarray.base
                rbase = bvals._ndarray.base

            # copy assertion we either have a None for a base or in case of
            # some blocks it is an array (e.g. datetimetz), but was copied
            if isinstance(cpvals, DatetimeArray):
                assert (lbase is None and rbase is None) or (lbase is not rbase)
            elif not isinstance(cpvals, np.ndarray):
                assert lbase is not rbase
            else:
                assert lbase is None and rbase is None

    def test_sparse(self):
        mgr = create_mgr("a: sparse-1; b: sparse-2")
        assert mgr.as_array().dtype == np.float64

    def test_sparse_mixed(self):
        mgr = create_mgr("a: sparse-1; b: sparse-2; c: f8")
        assert len(mgr.blocks) == 3
        assert isinstance(mgr, BlockManager)

    @pytest.mark.parametrize(
        "mgr_string, dtype",
        [("c: f4; d: f2", np.float32), ("c: f4; d: f2; e: f8", np.float64)],
    )
    def test_as_array_float(self, mgr_string, dtype):
        mgr = create_mgr(mgr_string)
        assert mgr.as_array().dtype == dtype

    @pytest.mark.parametrize(
        "mgr_string, dtype",
        [
            ("a: bool-1; b: bool-2", np.bool_),
            ("a: i8-1; b: i8-2; c: i4; d: i2; e: u1", np.int64),
            ("c: i4; d: i2; e: u1", np.int32),
        ],
    )
    def test_as_array_int_bool(self, mgr_string, dtype):
        mgr = create_mgr(mgr_string)
        assert mgr.as_array().dtype == dtype

    def test_as_array_datetime(self):
        mgr = create_mgr("h: datetime-1; g: datetime-2")
        assert mgr.as_array().dtype == "M8[ns]"

    def test_as_array_datetime_tz(self):
        mgr = create_mgr("h: M8[ns, US/Eastern]; g: M8[ns, CET]")
        assert mgr.iget(0).dtype == "datetime64[ns, US/Eastern]"
        assert mgr.iget(1).dtype == "datetime64[ns, CET]"
        assert mgr.as_array().dtype == "object"

    @pytest.mark.parametrize("t", ["float16", "float32", "float64", "int32", "int64"])
    def test_astype(self, t):
        # coerce all
        mgr = create_mgr("c: f4; d: f2; e: f8")

        t = np.dtype(t)
        tmgr = mgr.astype(t)
        assert tmgr.iget(0).dtype.type == t
        assert tmgr.iget(1).dtype.type == t
        assert tmgr.iget(2).dtype.type == t

        # mixed
        mgr = create_mgr("a,b: object; c: bool; d: datetime; e: f4; f: f2; g: f8")

        t = np.dtype(t)
        tmgr = mgr.astype(t, errors="ignore")
        assert tmgr.iget(2).dtype.type == t
        assert tmgr.iget(4).dtype.type == t
        assert tmgr.iget(5).dtype.type == t
        assert tmgr.iget(6).dtype.type == t

        assert tmgr.iget(0).dtype.type == np.object_
        assert tmgr.iget(1).dtype.type == np.object_
        if t != np.int64:
            assert tmgr.iget(3).dtype.type == np.datetime64
        else:
            assert tmgr.iget(3).dtype.type == t

    def test_convert(self, using_infer_string):
        def _compare(old_mgr, new_mgr):
            """compare the blocks, numeric compare ==, object don't"""
            old_blocks = set(old_mgr.blocks)
            new_blocks = set(new_mgr.blocks)
            assert len(old_blocks) == len(new_blocks)

            # compare non-numeric
            for b in old_blocks:
                found = False
                for nb in new_blocks:
                    if (b.values == nb.values).all():
                        found = True
                        break
                assert found

            for b in new_blocks:
                found = False
                for ob in old_blocks:
                    if (b.values == ob.values).all():
                        found = True
                        break
                assert found

        # noops
        mgr = create_mgr("f: i8; g: f8")
        new_mgr = mgr.convert(copy=True)
        _compare(mgr, new_mgr)

        # convert
        mgr = create_mgr("a,b,foo: object; f: i8; g: f8")
        mgr.iset(0, np.array(["1"] * N, dtype=np.object_))
        mgr.iset(1, np.array(["2."] * N, dtype=np.object_))
        mgr.iset(2, np.array(["foo."] * N, dtype=np.object_))
        new_mgr = mgr.convert(copy=True)
        dtype = "str" if using_infer_string else np.object_
        assert new_mgr.iget(0).dtype == dtype
        assert new_mgr.iget(1).dtype == dtype
        assert new_mgr.iget(2).dtype == dtype
        assert new_mgr.iget(3).dtype == np.int64
        assert new_mgr.iget(4).dtype == np.float64

        mgr = create_mgr(
            "a,b,foo: object; f: i4; bool: bool; dt: datetime; i: i8; g: f8; h: f2"
        )
        mgr.iset(0, np.array(["1"] * N, dtype=np.object_))
        mgr.iset(1, np.array(["2."] * N, dtype=np.object_))
        mgr.iset(2, np.array(["foo."] * N, dtype=np.object_))
        new_mgr = mgr.convert(copy=True)
        assert new_mgr.iget(0).dtype == dtype
        assert new_mgr.iget(1).dtype == dtype
        assert new_mgr.iget(2).dtype == dtype
        assert new_mgr.iget(3).dtype == np.int32
        assert new_mgr.iget(4).dtype == np.bool_
        assert new_mgr.iget(5).dtype.type, np.datetime64
        assert new_mgr.iget(6).dtype == np.int64
        assert new_mgr.iget(7).dtype == np.float64
        assert new_mgr.iget(8).dtype == np.float16

    def test_interleave(self):
        # self
        for dtype in ["f8", "i8", "object", "bool", "complex", "M8[ns]", "m8[ns]"]:
            mgr = create_mgr(f"a: {dtype}")
            assert mgr.as_array().dtype == dtype
            mgr = create_mgr(f"a: {dtype}; b: {dtype}")
            assert mgr.as_array().dtype == dtype

    @pytest.mark.parametrize(
        "mgr_string, dtype",
        [
            ("a: category", "i8"),
            ("a: category; b: category", "i8"),
            ("a: category; b: category2", "object"),
            ("a: category2", "object"),
            ("a: category2; b: category2", "object"),
            ("a: f8", "f8"),
            ("a: f8; b: i8", "f8"),
            ("a: f4; b: i8", "f8"),
            ("a: f4; b: i8; d: object", "object"),
            ("a: bool; b: i8", "object"),
            ("a: complex", "complex"),
            ("a: f8; b: category", "object"),
            ("a: M8[ns]; b: category", "object"),
            ("a: M8[ns]; b: bool", "object"),
            ("a: M8[ns]; b: i8", "object"),
            ("a: m8[ns]; b: bool", "object"),
            ("a: m8[ns]; b: i8", "object"),
            ("a: M8[ns]; b: m8[ns]", "object"),
        ],
    )
    def test_interleave_dtype(self, mgr_string, dtype):
        # will be converted according the actual dtype of the underlying
        mgr = create_mgr("a: category")
        assert mgr.as_array().dtype == "i8"
        mgr = create_mgr("a: category; b: category2")
        assert mgr.as_array().dtype == "object"
        mgr = create_mgr("a: category2")
        assert mgr.as_array().dtype == "object"

        # combinations
        mgr = create_mgr("a: f8")
        assert mgr.as_array().dtype == "f8"
        mgr = create_mgr("a: f8; b: i8")
        assert mgr.as_array().dtype == "f8"
        mgr = create_mgr("a: f4; b: i8")
        assert mgr.as_array().dtype == "f8"
        mgr = create_mgr("a: f4; b: i8; d: object")
        assert mgr.as_array().dtype == "object"
        mgr = create_mgr("a: bool; b: i8")
        assert mgr.as_array().dtype == "object"
        mgr = create_mgr("a: complex")
        assert mgr.as_array().dtype == "complex"
        mgr = create_mgr("a: f8; b: category")
        assert mgr.as_array().dtype == "f8"
        mgr = create_mgr("a: M8[ns]; b: category")
        assert mgr.as_array().dtype == "object"
        mgr = create_mgr("a: M8[ns]; b: bool")
        assert mgr.as_array().dtype == "object"
        mgr = create_mgr("a: M8[ns]; b: i8")
        assert mgr.as_array().dtype == "object"
        mgr = create_mgr("a: m8[ns]; b: bool")
        assert mgr.as_array().dtype == "object"
        mgr = create_mgr("a: m8[ns]; b: i8")
        assert mgr.as_array().dtype == "object"
        mgr = create_mgr("a: M8[ns]; b: m8[ns]")
        assert mgr.as_array().dtype == "object"

    def test_consolidate_ordering_issues(self, mgr):
        mgr.iset(mgr.items.get_loc("f"), np.random.default_rng(2).standard_normal(N))
        mgr.iset(mgr.items.get_loc("d"), np.random.default_rng(2).standard_normal(N))
        mgr.iset(mgr.items.get_loc("b"), np.random.default_rng(2).standard_normal(N))
        mgr.iset(mgr.items.get_loc("g"), np.random.default_rng(2).standard_normal(N))
        mgr.iset(mgr.items.get_loc("h"), np.random.default_rng(2).standard_normal(N))

        # we have datetime/tz blocks in mgr
        cons = mgr.consolidate()
        assert cons.nblocks == 4
        cons = mgr.consolidate().get_numeric_data()
        assert cons.nblocks == 1
        assert isinstance(cons.blocks[0].mgr_locs, BlockPlacement)
        tm.assert_numpy_array_equal(
            cons.blocks[0].mgr_locs.as_array, np.arange(len(cons.items), dtype=np.intp)
        )

    def test_reindex_items(self):
        # mgr is not consolidated, f8 & f8-2 blocks
        mgr = create_mgr("a: f8; b: i8; c: f8; d: i8; e: f8; f: bool; g: f8-2")

        reindexed = mgr.reindex_axis(["g", "c", "a", "d"], axis=0)
        # reindex_axis does not consolidate_inplace, as that risks failing to
        #  invalidate _item_cache
        assert not reindexed.is_consolidated()

        tm.assert_index_equal(reindexed.items, Index(["g", "c", "a", "d"]))
        tm.assert_almost_equal(
            mgr.iget(6).internal_values(), reindexed.iget(0).internal_values()
        )
        tm.assert_almost_equal(
            mgr.iget(2).internal_values(), reindexed.iget(1).internal_values()
        )
        tm.assert_almost_equal(
            mgr.iget(0).internal_values(), reindexed.iget(2).internal_values()
        )
        tm.assert_almost_equal(
            mgr.iget(3).internal_values(), reindexed.iget(3).internal_values()
        )

    def test_get_numeric_data(self, using_copy_on_write):
        mgr = create_mgr(
            "int: int; float: float; complex: complex;"
            "str: object; bool: bool; obj: object; dt: datetime",
            item_shape=(3,),
        )
        mgr.iset(5, np.array([1, 2, 3], dtype=np.object_))

        numeric = mgr.get_numeric_data()
        tm.assert_index_equal(numeric.items, Index(["int", "float", "complex", "bool"]))
        tm.assert_almost_equal(
            mgr.iget(mgr.items.get_loc("float")).internal_values(),
            numeric.iget(numeric.items.get_loc("float")).internal_values(),
        )

        # Check sharing
        numeric.iset(
            numeric.items.get_loc("float"),
            np.array([100.0, 200.0, 300.0]),
            inplace=True,
        )
        if using_copy_on_write:
            tm.assert_almost_equal(
                mgr.iget(mgr.items.get_loc("float")).internal_values(),
                np.array([1.0, 1.0, 1.0]),
            )
        else:
            tm.assert_almost_equal(
                mgr.iget(mgr.items.get_loc("float")).internal_values(),
                np.array([100.0, 200.0, 300.0]),
            )

    def test_get_bool_data(self, using_copy_on_write):
        mgr = create_mgr(
            "int: int; float: float; complex: complex;"
            "str: object; bool: bool; obj: object; dt: datetime",
            item_shape=(3,),
        )
        mgr.iset(6, np.array([True, False, True], dtype=np.object_))

        bools = mgr.get_bool_data()
        tm.assert_index_equal(bools.items, Index(["bool"]))
        tm.assert_almost_equal(
            mgr.iget(mgr.items.get_loc("bool")).internal_values(),
            bools.iget(bools.items.get_loc("bool")).internal_values(),
        )

        bools.iset(0, np.array([True, False, True]), inplace=True)
        if using_copy_on_write:
            tm.assert_numpy_array_equal(
                mgr.iget(mgr.items.get_loc("bool")).internal_values(),
                np.array([True, True, True]),
            )
        else:
            tm.assert_numpy_array_equal(
                mgr.iget(mgr.items.get_loc("bool")).internal_values(),
                np.array([True, False, True]),
            )

    def test_unicode_repr_doesnt_raise(self):
        repr(create_mgr("b,\u05d0: object"))

    @pytest.mark.parametrize(
        "mgr_string", ["a,b,c: i8-1; d,e,f: i8-2", "a,a,a: i8-1; b,b,b: i8-2"]
    )
    def test_equals(self, mgr_string):
        # unique items
        bm1 = create_mgr(mgr_string)
        bm2 = BlockManager(bm1.blocks[::-1], bm1.axes)
        assert bm1.equals(bm2)

    @pytest.mark.parametrize(
        "mgr_string",
        [
            "a:i8;b:f8",  # basic case
            "a:i8;b:f8;c:c8;d:b",  # many types
            "a:i8;e:dt;f:td;g:string",  # more types
            "a:i8;b:category;c:category2",  # categories
            "c:sparse;d:sparse_na;b:f8",  # sparse
        ],
    )
    def test_equals_block_order_different_dtypes(self, mgr_string):
        # GH 9330
        bm = create_mgr(mgr_string)
        block_perms = itertools.permutations(bm.blocks)
        for bm_perm in block_perms:
            bm_this = BlockManager(tuple(bm_perm), bm.axes)
            assert bm.equals(bm_this)
            assert bm_this.equals(bm)

    def test_single_mgr_ctor(self):
        mgr = create_single_mgr("f8", num_rows=5)
        assert mgr.external_values().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]

    @pytest.mark.parametrize("value", [1, "True", [1, 2, 3], 5.0])
    def test_validate_bool_args(self, value):
        bm1 = create_mgr("a,b,c: i8-1; d,e,f: i8-2")

        msg = (
            'For argument "inplace" expected type bool, '
            f"received type {type(value).__name__}."
        )
        with pytest.raises(ValueError, match=msg):
            bm1.replace_list([1], [2], inplace=value)

    def test_iset_split_block(self):
        bm = create_mgr("a,b,c: i8; d: f8")
        bm._iset_split_block(0, np.array([0]))
        tm.assert_numpy_array_equal(
            bm.blklocs, np.array([0, 0, 1, 0], dtype="int64" if IS64 else "int32")
        )
        # First indexer currently does not have a block associated with it in case
        tm.assert_numpy_array_equal(
            bm.blknos, np.array([0, 0, 0, 1], dtype="int64" if IS64 else "int32")
        )
        assert len(bm.blocks) == 2

    def test_iset_split_block_values(self):
        bm = create_mgr("a,b,c: i8; d: f8")
        bm._iset_split_block(0, np.array([0]), np.array([list(range(10))]))
        tm.assert_numpy_array_equal(
            bm.blklocs, np.array([0, 0, 1, 0], dtype="int64" if IS64 else "int32")
        )
        # First indexer currently does not have a block associated with it in case
        tm.assert_numpy_array_equal(
            bm.blknos, np.array([0, 2, 2, 1], dtype="int64" if IS64 else "int32")
        )
        assert len(bm.blocks) == 3


def _as_array(mgr):
    if mgr.ndim == 1:
        return mgr.external_values()
    return mgr.as_array().T


class TestIndexing:
    # Nosetests-style data-driven tests.
    #
    # This test applies different indexing routines to block managers and
    # compares the outcome to the result of same operations on np.ndarray.
    #
    # NOTE: sparse (SparseBlock with fill_value != np.nan) fail a lot of tests
    #       and are disabled.

    MANAGERS = [
        create_single_mgr("f8", N),
        create_single_mgr("i8", N),
        # 2-dim
        create_mgr("a,b,c,d,e,f: f8", item_shape=(N,)),
        create_mgr("a,b,c,d,e,f: i8", item_shape=(N,)),
        create_mgr("a,b: f8; c,d: i8; e,f: string", item_shape=(N,)),
        create_mgr("a,b: f8; c,d: i8; e,f: f8", item_shape=(N,)),
    ]

    @pytest.mark.parametrize("mgr", MANAGERS)
    def test_get_slice(self, mgr):
        def assert_slice_ok(mgr, axis, slobj):
            mat = _as_array(mgr)

            # we maybe using an ndarray to test slicing and
            # might not be the full length of the axis
            if isinstance(slobj, np.ndarray):
                ax = mgr.axes[axis]
                if len(ax) and len(slobj) and len(slobj) != len(ax):
                    slobj = np.concatenate(
                        [slobj, np.zeros(len(ax) - len(slobj), dtype=bool)]
                    )

            if isinstance(slobj, slice):
                sliced = mgr.get_slice(slobj, axis=axis)
            elif (
                mgr.ndim == 1
                and axis == 0
                and isinstance(slobj, np.ndarray)
                and slobj.dtype == bool
            ):
                sliced = mgr.get_rows_with_mask(slobj)
            else:
                # BlockManager doesn't support non-slice, SingleBlockManager
                #  doesn't support axis > 0
                raise TypeError(slobj)

            mat_slobj = (slice(None),) * axis + (slobj,)
            tm.assert_numpy_array_equal(
                mat[mat_slobj], _as_array(sliced), check_dtype=False
            )
            tm.assert_index_equal(mgr.axes[axis][slobj], sliced.axes[axis])

        assert mgr.ndim <= 2, mgr.ndim
        for ax in range(mgr.ndim):
            # slice
            assert_slice_ok(mgr, ax, slice(None))
            assert_slice_ok(mgr, ax, slice(3))
            assert_slice_ok(mgr, ax, slice(100))
            assert_slice_ok(mgr, ax, slice(1, 4))
            assert_slice_ok(mgr, ax, slice(3, 0, -2))

            if mgr.ndim < 2:
                # 2D only support slice objects

                # boolean mask
                assert_slice_ok(mgr, ax, np.ones(mgr.shape[ax], dtype=np.bool_))
                assert_slice_ok(mgr, ax, np.zeros(mgr.shape[ax], dtype=np.bool_))

                if mgr.shape[ax] >= 3:
                    assert_slice_ok(mgr, ax, np.arange(mgr.shape[ax]) % 3 == 0)
                    assert_slice_ok(
                        mgr, ax, np.array([True, True, False], dtype=np.bool_)
                    )

    @pytest.mark.parametrize("mgr", MANAGERS)
    def test_take(self, mgr):
        def assert_take_ok(mgr, axis, indexer):
            mat = _as_array(mgr)
            taken = mgr.take(indexer, axis)
            tm.assert_numpy_array_equal(
                np.take(mat, indexer, axis), _as_array(taken), check_dtype=False
            )
            tm.assert_index_equal(mgr.axes[axis].take(indexer), taken.axes[axis])

        for ax in range(mgr.ndim):
            # take/fancy indexer
            assert_take_ok(mgr, ax, indexer=np.array([], dtype=np.intp))
            assert_take_ok(mgr, ax, indexer=np.array([0, 0, 0], dtype=np.intp))
            assert_take_ok(
                mgr, ax, indexer=np.array(list(range(mgr.shape[ax])), dtype=np.intp)
            )

            if mgr.shape[ax] >= 3:
                assert_take_ok(mgr, ax, indexer=np.array([0, 1, 2], dtype=np.intp))
                assert_take_ok(mgr, ax, indexer=np.array([-1, -2, -3], dtype=np.intp))

    @pytest.mark.parametrize("mgr", MANAGERS)
    @pytest.mark.parametrize("fill_value", [None, np.nan, 100.0])
    def test_reindex_axis(self, fill_value, mgr):
        def assert_reindex_axis_is_ok(mgr, axis, new_labels, fill_value):
            mat = _as_array(mgr)
            indexer = mgr.axes[axis].get_indexer_for(new_labels)

            reindexed = mgr.reindex_axis(new_labels, axis, fill_value=fill_value)
            tm.assert_numpy_array_equal(
                algos.take_nd(mat, indexer, axis, fill_value=fill_value),
                _as_array(reindexed),
                check_dtype=False,
            )
            tm.assert_index_equal(reindexed.axes[axis], new_labels)

        for ax in range(mgr.ndim):
            assert_reindex_axis_is_ok(mgr, ax, Index([]), fill_value)
            assert_reindex_axis_is_ok(mgr, ax, mgr.axes[ax], fill_value)
            assert_reindex_axis_is_ok(mgr, ax, mgr.axes[ax][[0, 0, 0]], fill_value)
            assert_reindex_axis_is_ok(mgr, ax, Index(["foo", "bar", "baz"]), fill_value)
            assert_reindex_axis_is_ok(
                mgr, ax, Index(["foo", mgr.axes[ax][0], "baz"]), fill_value
            )

            if mgr.shape[ax] >= 3:
                assert_reindex_axis_is_ok(mgr, ax, mgr.axes[ax][:-3], fill_value)
                assert_reindex_axis_is_ok(mgr, ax, mgr.axes[ax][-3::-1], fill_value)
                assert_reindex_axis_is_ok(
                    mgr, ax, mgr.axes[ax][[0, 1, 2, 0, 1, 2]], fill_value
                )

    @pytest.mark.parametrize("mgr", MANAGERS)
    @pytest.mark.parametrize("fill_value", [None, np.nan, 100.0])
    def test_reindex_indexer(self, fill_value, mgr):
        def assert_reindex_indexer_is_ok(mgr, axis, new_labels, indexer, fill_value):
            mat = _as_array(mgr)
            reindexed_mat = algos.take_nd(mat, indexer, axis, fill_value=fill_value)
            reindexed = mgr.reindex_indexer(
                new_labels, indexer, axis, fill_value=fill_value
            )
            tm.assert_numpy_array_equal(
                reindexed_mat, _as_array(reindexed), check_dtype=False
            )
            tm.assert_index_equal(reindexed.axes[axis], new_labels)

        for ax in range(mgr.ndim):
            assert_reindex_indexer_is_ok(
                mgr, ax, Index([]), np.array([], dtype=np.intp), fill_value
            )
            assert_reindex_indexer_is_ok(
                mgr, ax, mgr.axes[ax], np.arange(mgr.shape[ax]), fill_value
            )
            assert_reindex_indexer_is_ok(
                mgr,
                ax,
                Index(["foo"] * mgr.shape[ax]),
                np.arange(mgr.shape[ax]),
                fill_value,
            )
            assert_reindex_indexer_is_ok(
                mgr, ax, mgr.axes[ax][::-1], np.arange(mgr.shape[ax]), fill_value
            )
            assert_reindex_indexer_is_ok(
                mgr, ax, mgr.axes[ax], np.arange(mgr.shape[ax])[::-1], fill_value
            )
            assert_reindex_indexer_is_ok(
                mgr, ax, Index(["foo", "bar", "baz"]), np.array([0, 0, 0]), fill_value
            )
            assert_reindex_indexer_is_ok(
                mgr, ax, Index(["foo", "bar", "baz"]), np.array([-1, 0, -1]), fill_value
            )
            assert_reindex_indexer_is_ok(
                mgr,
                ax,
                Index(["foo", mgr.axes[ax][0], "baz"]),
                np.array([-1, -1, -1]),
                fill_value,
            )

            if mgr.shape[ax] >= 3:
                assert_reindex_indexer_is_ok(
                    mgr,
                    ax,
                    Index(["foo", "bar", "baz"]),
                    np.array([0, 1, 2]),
                    fill_value,
                )


class TestBlockPlacement:
    @pytest.mark.parametrize(
        "slc, expected",
        [
            (slice(0, 4), 4),
            (slice(0, 4, 2), 2),
            (slice(0, 3, 2), 2),
            (slice(0, 1, 2), 1),
            (slice(1, 0, -1), 1),
        ],
    )
    def test_slice_len(self, slc, expected):
        assert len(BlockPlacement(slc)) == expected

    @pytest.mark.parametrize("slc", [slice(1, 1, 0), slice(1, 2, 0)])
    def test_zero_step_raises(self, slc):
        msg = "slice step cannot be zero"
        with pytest.raises(ValueError, match=msg):
            BlockPlacement(slc)

    def test_slice_canonize_negative_stop(self):
        # GH#37524 negative stop is OK with negative step and positive start
        slc = slice(3, -1, -2)

        bp = BlockPlacement(slc)
        assert bp.indexer == slice(3, None, -2)

    @pytest.mark.parametrize(
        "slc",
        [
            slice(None, None),
            slice(10, None),
            slice(None, None, -1),
            slice(None, 10, -1),
            # These are "unbounded" because negative index will
            #  change depending on container shape.
            slice(-1, None),
            slice(None, -1),
            slice(-1, -1),
            slice(-1, None, -1),
            slice(None, -1, -1),
            slice(-1, -1, -1),
        ],
    )
    def test_unbounded_slice_raises(self, slc):
        msg = "unbounded slice"
        with pytest.raises(ValueError, match=msg):
            BlockPlacement(slc)

    @pytest.mark.parametrize(
        "slc",
        [
            slice(0, 0),
            slice(100, 0),
            slice(100, 100),
            slice(100, 100, -1),
            slice(0, 100, -1),
        ],
    )
    def test_not_slice_like_slices(self, slc):
        assert not BlockPlacement(slc).is_slice_like

    @pytest.mark.parametrize(
        "arr, slc",
        [
            ([0], slice(0, 1, 1)),
            ([100], slice(100, 101, 1)),
            ([0, 1, 2], slice(0, 3, 1)),
            ([0, 5, 10], slice(0, 15, 5)),
            ([0, 100], slice(0, 200, 100)),
            ([2, 1], slice(2, 0, -1)),
        ],
    )
    def test_array_to_slice_conversion(self, arr, slc):
        assert BlockPlacement(arr).as_slice == slc

    @pytest.mark.parametrize(
        "arr",
        [
            [],
            [-1],
            [-1, -2, -3],
            [-10],
            [-1],
            [-1, 0, 1, 2],
            [-2, 0, 2, 4],
            [1, 0, -1],
            [1, 1, 1],
        ],
    )
    def test_not_slice_like_arrays(self, arr):
        assert not BlockPlacement(arr).is_slice_like

    @pytest.mark.parametrize(
        "slc, expected",
        [(slice(0, 3), [0, 1, 2]), (slice(0, 0), []), (slice(3, 0), [])],
    )
    def test_slice_iter(self, slc, expected):
        assert list(BlockPlacement(slc)) == expected

    @pytest.mark.parametrize(
        "slc, arr",
        [
            (slice(0, 3), [0, 1, 2]),
            (slice(0, 0), []),
            (slice(3, 0), []),
            (slice(3, 0, -1), [3, 2, 1]),
        ],
    )
    def test_slice_to_array_conversion(self, slc, arr):
        tm.assert_numpy_array_equal(
            BlockPlacement(slc).as_array, np.asarray(arr, dtype=np.intp)
        )

    def test_blockplacement_add(self):
        bpl = BlockPlacement(slice(0, 5))
        assert bpl.add(1).as_slice == slice(1, 6, 1)
        assert bpl.add(np.arange(5)).as_slice == slice(0, 10, 2)
        assert list(bpl.add(np.arange(5, 0, -1))) == [5, 5, 5, 5, 5]

    @pytest.mark.parametrize(
        "val, inc, expected",
        [
            (slice(0, 0), 0, []),
            (slice(1, 4), 0, [1, 2, 3]),
            (slice(3, 0, -1), 0, [3, 2, 1]),
            ([1, 2, 4], 0, [1, 2, 4]),
            (slice(0, 0), 10, []),
            (slice(1, 4), 10, [11, 12, 13]),
            (slice(3, 0, -1), 10, [13, 12, 11]),
            ([1, 2, 4], 10, [11, 12, 14]),
            (slice(0, 0), -1, []),
            (slice(1, 4), -1, [0, 1, 2]),
            ([1, 2, 4], -1, [0, 1, 3]),
        ],
    )
    def test_blockplacement_add_int(self, val, inc, expected):
        assert list(BlockPlacement(val).add(inc)) == expected

    @pytest.mark.parametrize("val", [slice(1, 4), [1, 2, 4]])
    def test_blockplacement_add_int_raises(self, val):
        msg = "iadd causes length change"
        with pytest.raises(ValueError, match=msg):
            BlockPlacement(val).add(-10)


class TestCanHoldElement:
    @pytest.fixture(
        params=[
            lambda x: x,
            lambda x: x.to_series(),
            lambda x: x._data,
            lambda x: list(x),
            lambda x: x.astype(object),
            lambda x: np.asarray(x),
            lambda x: x[0],
            lambda x: x[:0],
        ]
    )
    def element(self, request):
        """
        Functions that take an Index and return an element that should have
        blk._can_hold_element(element) for a Block with this index's dtype.
        """
        return request.param

    def test_datetime_block_can_hold_element(self):
        block = create_block("datetime", [0])

        assert block._can_hold_element([])

        # We will check that block._can_hold_element iff arr.__setitem__ works
        arr = pd.array(block.values.ravel())

        # coerce None
        assert block._can_hold_element(None)
        arr[0] = None
        assert arr[0] is pd.NaT

        # coerce different types of datetime objects
        vals = [np.datetime64("2010-10-10"), datetime(2010, 10, 10)]
        for val in vals:
            assert block._can_hold_element(val)
            arr[0] = val

        val = date(2010, 10, 10)
        assert not block._can_hold_element(val)

        msg = (
            "value should be a 'Timestamp', 'NaT', "
            "or array of those. Got 'date' instead."
        )
        with pytest.raises(TypeError, match=msg):
            arr[0] = val

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    def test_interval_can_hold_element_emptylist(self, dtype, element):
        arr = np.array([1, 3, 4], dtype=dtype)
        ii = IntervalIndex.from_breaks(arr)
        blk = new_block(ii._data, BlockPlacement([1]), ndim=2)

        assert blk._can_hold_element([])
        # TODO: check this holds for all blocks

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    def test_interval_can_hold_element(self, dtype, element):
        arr = np.array([1, 3, 4, 9], dtype=dtype)
        ii = IntervalIndex.from_breaks(arr)
        blk = new_block(ii._data, BlockPlacement([1]), ndim=2)

        elem = element(ii)
        self.check_series_setitem(elem, ii, True)
        assert blk._can_hold_element(elem)

        # Careful: to get the expected Series-inplace behavior we need
        # `elem` to not have the same length as `arr`
        ii2 = IntervalIndex.from_breaks(arr[:-1], closed="neither")
        elem = element(ii2)
        with tm.assert_produces_warning(FutureWarning):
            self.check_series_setitem(elem, ii, False)
        assert not blk._can_hold_element(elem)

        ii3 = IntervalIndex.from_breaks([Timestamp(1), Timestamp(3), Timestamp(4)])
        elem = element(ii3)
        with tm.assert_produces_warning(FutureWarning):
            self.check_series_setitem(elem, ii, False)
        assert not blk._can_hold_element(elem)

        ii4 = IntervalIndex.from_breaks([Timedelta(1), Timedelta(3), Timedelta(4)])
        elem = element(ii4)
        with tm.assert_produces_warning(FutureWarning):
            self.check_series_setitem(elem, ii, False)
        assert not blk._can_hold_element(elem)

    def test_period_can_hold_element_emptylist(self):
        pi = period_range("2016", periods=3, freq="Y")
        blk = new_block(pi._data.reshape(1, 3), BlockPlacement([1]), ndim=2)

        assert blk._can_hold_element([])

    def test_period_can_hold_element(self, element):
        pi = period_range("2016", periods=3, freq="Y")

        elem = element(pi)
        self.check_series_setitem(elem, pi, True)

        # Careful: to get the expected Series-inplace behavior we need
        # `elem` to not have the same length as `arr`
        pi2 = pi.asfreq("D")[:-1]
        elem = element(pi2)
        with tm.assert_produces_warning(FutureWarning):
            self.check_series_setitem(elem, pi, False)

        dti = pi.to_timestamp("s")[:-1]
        elem = element(dti)
        with tm.assert_produces_warning(FutureWarning):
            self.check_series_setitem(elem, pi, False)

    def check_can_hold_element(self, obj, elem, inplace: bool):
        blk = obj._mgr.blocks[0]
        if inplace:
            assert blk._can_hold_element(elem)
        else:
            assert not blk._can_hold_element(elem)

    def check_series_setitem(self, elem, index: Index, inplace: bool):
        arr = index._data.copy()
        ser = Series(arr, copy=False)

        self.check_can_hold_element(ser, elem, inplace)

        if is_scalar(elem):
            ser[0] = elem
        else:
            ser[: len(elem)] = elem

        if inplace:
            assert ser.array is arr  # i.e. setting was done inplace
        else:
            assert ser.dtype == object


class TestShouldStore:
    def test_should_store_categorical(self):
        cat = Categorical(["A", "B", "C"])
        df = DataFrame(cat)
        blk = df._mgr.blocks[0]

        # matching dtype
        assert blk.should_store(cat)
        assert blk.should_store(cat[:-1])

        # different dtype
        assert not blk.should_store(cat.as_ordered())

        # ndarray instead of Categorical
        assert not blk.should_store(np.asarray(cat))


def test_validate_ndim():
    values = np.array([1.0, 2.0])
    placement = BlockPlacement(slice(2))
    msg = r"Wrong number of dimensions. values.ndim != ndim \[1 != 2\]"

    with pytest.raises(ValueError, match=msg):
        make_block(values, placement, ndim=2)


def test_block_shape():
    idx = Index([0, 1, 2, 3, 4])
    a = Series([1, 2, 3]).reindex(idx)
    b = Series(Categorical([1, 2, 3])).reindex(idx)

    assert a._mgr.blocks[0].mgr_locs.indexer == b._mgr.blocks[0].mgr_locs.indexer


def test_make_block_no_pandas_array(block_maker):
    # https://github.com/pandas-dev/pandas/pull/24866
    arr = pd.arrays.NumpyExtensionArray(np.array([1, 2]))

    # NumpyExtensionArray, no dtype
    result = block_maker(arr, BlockPlacement(slice(len(arr))), ndim=arr.ndim)
    assert result.dtype.kind in ["i", "u"]

    if block_maker is make_block:
        # new_block requires caller to unwrap NumpyExtensionArray
        assert result.is_extension is False

        # NumpyExtensionArray, NumpyEADtype
        result = block_maker(arr, slice(len(arr)), dtype=arr.dtype, ndim=arr.ndim)
        assert result.dtype.kind in ["i", "u"]
        assert result.is_extension is False

        # new_block no longer taked dtype keyword
        # ndarray, NumpyEADtype
        result = block_maker(
            arr.to_numpy(), slice(len(arr)), dtype=arr.dtype, ndim=arr.ndim
        )
        assert result.dtype.kind in ["i", "u"]
        assert result.is_extension is False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dateutil\_common.py ===
"""
Common code used in multiple modules.
"""


class weekday(object):
    __slots__ = ["weekday", "n"]

    def __init__(self, weekday, n=None):
        self.weekday = weekday
        self.n = n

    def __call__(self, n):
        if n == self.n:
            return self
        else:
            return self.__class__(self.weekday, n)

    def __eq__(self, other):
        try:
            if self.weekday != other.weekday or self.n != other.n:
                return False
        except AttributeError:
            return False
        return True

    def __hash__(self):
        return hash((
          self.weekday,
          self.n,
        ))

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        s = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")[self.weekday]
        if not self.n:
            return s
        else:
            return "%s(%+d)" % (s, self.n)

# vim:ts=4:sw=4:et

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask_wtf\recaptcha\widgets.py ===
from urllib.parse import urlencode

from flask import current_app
from markupsafe import Markup

RECAPTCHA_SCRIPT_DEFAULT = "https://www.google.com/recaptcha/api.js"
RECAPTCHA_DIV_CLASS_DEFAULT = "g-recaptcha"
RECAPTCHA_TEMPLATE = """
<script src='%s' async defer></script>
<div class="%s" %s></div>
"""

__all__ = ["RecaptchaWidget"]


class RecaptchaWidget:
    def recaptcha_html(self, public_key):
        html = current_app.config.get("RECAPTCHA_HTML")
        if html:
            return Markup(html)
        params = current_app.config.get("RECAPTCHA_PARAMETERS")
        script = current_app.config.get("RECAPTCHA_SCRIPT")
        if not script:
            script = RECAPTCHA_SCRIPT_DEFAULT
        if params:
            script += "?" + urlencode(params)
        attrs = current_app.config.get("RECAPTCHA_DATA_ATTRS", {})
        attrs["sitekey"] = public_key
        snippet = " ".join(f'data-{k}="{attrs[k]}"' for k in attrs)  # noqa: B028, B907
        div_class = current_app.config.get("RECAPTCHA_DIV_CLASS")
        if not div_class:
            div_class = RECAPTCHA_DIV_CLASS_DEFAULT
        return Markup(RECAPTCHA_TEMPLATE % (script, div_class, snippet))

    def __call__(self, field, error=None, **kwargs):
        """Returns the recaptcha input HTML."""

        try:
            public_key = current_app.config["RECAPTCHA_PUBLIC_KEY"]
        except KeyError:
            raise RuntimeError("RECAPTCHA_PUBLIC_KEY config not set") from None

        return self.recaptcha_html(public_key)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flatbuffers\util.py ===
# Copyright 2017 Google Inc. All rights reserved.
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

from . import encode
from . import number_types
from . import packer

def GetSizePrefix(buf, offset):
	"""Extract the size prefix from a buffer."""
	return encode.Get(packer.int32, buf, offset)

def GetBufferIdentifier(buf, offset, size_prefixed=False):
        """Extract the file_identifier from a buffer"""
        if size_prefixed:
            # increase offset by size of UOffsetTFlags
            offset += number_types.UOffsetTFlags.bytewidth
        # increase offset by size of root table pointer
        offset += number_types.UOffsetTFlags.bytewidth
        # end of FILE_IDENTIFIER
        end = offset + encode.FILE_IDENTIFIER_LENGTH
        return buf[offset:end]

def BufferHasIdentifier(buf, offset, file_identifier, size_prefixed=False):
        got = GetBufferIdentifier(buf, offset, size_prefixed=size_prefixed)
        return got == file_identifier

def RemoveSizePrefix(buf, offset):
	"""
	Create a slice of a size-prefixed buffer that has
	its position advanced just past the size prefix.
	"""
	return buf, offset + number_types.Int32Flags.bytewidth

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\api_pb2.py ===
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: google/protobuf/api.proto
# Protobuf Python Version: 6.31.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    31,
    1,
    '',
    'google/protobuf/api.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import source_context_pb2 as google_dot_protobuf_dot_source__context__pb2
from google.protobuf import type_pb2 as google_dot_protobuf_dot_type__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x19google/protobuf/api.proto\x12\x0fgoogle.protobuf\x1a$google/protobuf/source_context.proto\x1a\x1agoogle/protobuf/type.proto\"\xc1\x02\n\x03\x41pi\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x31\n\x07methods\x18\x02 \x03(\x0b\x32\x17.google.protobuf.MethodR\x07methods\x12\x31\n\x07options\x18\x03 \x03(\x0b\x32\x17.google.protobuf.OptionR\x07options\x12\x18\n\x07version\x18\x04 \x01(\tR\x07version\x12\x45\n\x0esource_context\x18\x05 \x01(\x0b\x32\x1e.google.protobuf.SourceContextR\rsourceContext\x12.\n\x06mixins\x18\x06 \x03(\x0b\x32\x16.google.protobuf.MixinR\x06mixins\x12/\n\x06syntax\x18\x07 \x01(\x0e\x32\x17.google.protobuf.SyntaxR\x06syntax\"\xb2\x02\n\x06Method\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12(\n\x10request_type_url\x18\x02 \x01(\tR\x0erequestTypeUrl\x12+\n\x11request_streaming\x18\x03 \x01(\x08R\x10requestStreaming\x12*\n\x11response_type_url\x18\x04 \x01(\tR\x0fresponseTypeUrl\x12-\n\x12response_streaming\x18\x05 \x01(\x08R\x11responseStreaming\x12\x31\n\x07options\x18\x06 \x03(\x0b\x32\x17.google.protobuf.OptionR\x07options\x12/\n\x06syntax\x18\x07 \x01(\x0e\x32\x17.google.protobuf.SyntaxR\x06syntax\"/\n\x05Mixin\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x12\n\x04root\x18\x02 \x01(\tR\x04rootBv\n\x13\x63om.google.protobufB\x08\x41piProtoP\x01Z,google.golang.org/protobuf/types/known/apipb\xa2\x02\x03GPB\xaa\x02\x1eGoogle.Protobuf.WellKnownTypesb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'google.protobuf.api_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\023com.google.protobufB\010ApiProtoP\001Z,google.golang.org/protobuf/types/known/apipb\242\002\003GPB\252\002\036Google.Protobuf.WellKnownTypes'
  _globals['_API']._serialized_start=113
  _globals['_API']._serialized_end=434
  _globals['_METHOD']._serialized_start=437
  _globals['_METHOD']._serialized_end=743
  _globals['_MIXIN']._serialized_start=745
  _globals['_MIXIN']._serialized_end=792
# @@protoc_insertion_point(module_scope)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\humanfriendly\decorators.py ===
# Human friendly input/output in Python.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: March 2, 2020
# URL: https://humanfriendly.readthedocs.io

"""Simple function decorators to make Python programming easier."""

# Standard library modules.
import functools

# Public identifiers that require documentation.
__all__ = ('RESULTS_ATTRIBUTE', 'cached')

RESULTS_ATTRIBUTE = 'cached_results'
"""The name of the property used to cache the return values of functions (a string)."""


def cached(function):
    """
    Rudimentary caching decorator for functions.

    :param function: The function whose return value should be cached.
    :returns: The decorated function.

    The given function will only be called once, the first time the wrapper
    function is called. The return value is cached by the wrapper function as
    an attribute of the given function and returned on each subsequent call.

    .. note:: Currently no function arguments are supported because only a
              single return value can be cached. Accepting any function
              arguments at all would imply that the cache is parametrized on
              function arguments, which is not currently the case.
    """
    @functools.wraps(function)
    def wrapper():
        try:
            return getattr(wrapper, RESULTS_ATTRIBUTE)
        except AttributeError:
            result = function()
            setattr(wrapper, RESULTS_ATTRIBUTE, result)
            return result
    return wrapper

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\descriptors.py ===
# Copyright (c) 2010-2024 openpyxl



from openpyxl.descriptors.nested import (
    NestedMinMax
    )

from openpyxl.descriptors import Typed

from .data_source import NumFmt

"""
Utility descriptors for the chart module.
For convenience but also clarity.
"""

class NestedGapAmount(NestedMinMax):

    allow_none = True
    min = 0
    max = 500


class NestedOverlap(NestedMinMax):

    allow_none = True
    min = -100
    max = 100


class NumberFormatDescriptor(Typed):
    """
    Allow direct assignment of format code
    """

    expected_type = NumFmt
    allow_none = True

    def __set__(self, instance, value):
        if isinstance(value, str):
            value = NumFmt(value)
        super().__set__(instance, value)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\compat\numbers.py ===
# Copyright (c) 2010-2024 openpyxl

from decimal import Decimal

NUMERIC_TYPES = (int, float, Decimal)


try:
    import numpy
    NUMPY = True
except ImportError:
    NUMPY = False


if NUMPY:
    NUMERIC_TYPES = NUMERIC_TYPES + (numpy.short,
                                     numpy.ushort,
                                     numpy.intc,
                                     numpy.uintc,
                                     numpy.int_,
                                     numpy.uint,
                                     numpy.longlong,
                                     numpy.ulonglong,
                                     numpy.half,
                                     numpy.float16,
                                     numpy.single,
                                     numpy.double,
                                     numpy.longdouble,
                                     numpy.int8,
                                     numpy.int16,
                                     numpy.int32,
                                     numpy.int64,
                                     numpy.uint8,
                                     numpy.uint16,
                                     numpy.uint32,
                                     numpy.uint64,
                                     numpy.intp,
                                     numpy.uintp,
                                     numpy.float32,
                                     numpy.float64,
                                     numpy.bool_,
                                     numpy.floating,
                                     numpy.integer)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\utils\escape.py ===
# Copyright (c) 2010-2024 openpyxl

"""
OOXML has non-standard escaping for characters < \031
"""

import re


def escape(value):
    r"""
    Convert ASCII < 31 to OOXML: \n == _x + hex(ord(\n)) + _
    """

    CHAR_REGEX = re.compile(r"[\001-\031]")

    def _sub(match):
        """
        Callback to escape chars
        """
        return "_x{:0>4x}_".format(ord(match.group(0)))

    return CHAR_REGEX.sub(_sub, value)


def unescape(value):
    r"""
    Convert escaped strings to ASCIII: _x000a_ == \n
    """


    ESCAPED_REGEX = re.compile("_x([0-9A-Fa-f]{4})_")

    def _sub(match):
        """
        Callback to unescape chars
        """
        return chr(int(match.group(1), 16))

    if "_x" in value:
        value = ESCAPED_REGEX.sub(_sub, value)

    return value

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\__init__.py ===
from pandas.core.arrays.arrow import ArrowExtensionArray
from pandas.core.arrays.base import (
    ExtensionArray,
    ExtensionOpsMixin,
    ExtensionScalarOpsMixin,
)
from pandas.core.arrays.boolean import BooleanArray
from pandas.core.arrays.categorical import Categorical
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.floating import FloatingArray
from pandas.core.arrays.integer import IntegerArray
from pandas.core.arrays.interval import IntervalArray
from pandas.core.arrays.masked import BaseMaskedArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.arrays.period import (
    PeriodArray,
    period_array,
)
from pandas.core.arrays.sparse import SparseArray
from pandas.core.arrays.string_ import StringArray
from pandas.core.arrays.string_arrow import ArrowStringArray
from pandas.core.arrays.timedeltas import TimedeltaArray

__all__ = [
    "ArrowExtensionArray",
    "ExtensionArray",
    "ExtensionOpsMixin",
    "ExtensionScalarOpsMixin",
    "ArrowStringArray",
    "BaseMaskedArray",
    "BooleanArray",
    "Categorical",
    "DatetimeArray",
    "FloatingArray",
    "IntegerArray",
    "IntervalArray",
    "NumpyExtensionArray",
    "PeriodArray",
    "period_array",
    "SparseArray",
    "StringArray",
    "TimedeltaArray",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\utils\packaging.py ===
import functools
import logging
from typing import Optional, Tuple

from pip._vendor.packaging import specifiers, version
from pip._vendor.packaging.requirements import Requirement

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=32)
def check_requires_python(
    requires_python: Optional[str], version_info: Tuple[int, ...]
) -> bool:
    """
    Check if the given Python version matches a "Requires-Python" specifier.

    :param version_info: A 3-tuple of ints representing a Python
        major-minor-micro version to check (e.g. `sys.version_info[:3]`).

    :return: `True` if the given Python version satisfies the requirement.
        Otherwise, return `False`.

    :raises InvalidSpecifier: If `requires_python` has an invalid format.
    """
    if requires_python is None:
        # The package provides no information
        return True
    requires_python_specifier = specifiers.SpecifierSet(requires_python)

    python_version = version.parse(".".join(map(str, version_info)))
    return python_version in requires_python_specifier


@functools.lru_cache(maxsize=10000)
def get_requirement(req_string: str) -> Requirement:
    """Construct a packaging.Requirement object with caching"""
    # Parsing requirement strings is expensive, and is also expected to happen
    # with a low diversity of different arguments (at least relative the number
    # constructed). This method adds a cache to requirement object creation to
    # minimize repeated parsing of the same string to construct equivalent
    # Requirement objects.
    return Requirement(req_string)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\cachecontrol\wrapper.py ===
# SPDX-FileCopyrightText: 2015 Eric Larson
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Collection

from pip._vendor.cachecontrol.adapter import CacheControlAdapter
from pip._vendor.cachecontrol.cache import DictCache

if TYPE_CHECKING:
    from pip._vendor import requests

    from pip._vendor.cachecontrol.cache import BaseCache
    from pip._vendor.cachecontrol.controller import CacheController
    from pip._vendor.cachecontrol.heuristics import BaseHeuristic
    from pip._vendor.cachecontrol.serialize import Serializer


def CacheControl(
    sess: requests.Session,
    cache: BaseCache | None = None,
    cache_etags: bool = True,
    serializer: Serializer | None = None,
    heuristic: BaseHeuristic | None = None,
    controller_class: type[CacheController] | None = None,
    adapter_class: type[CacheControlAdapter] | None = None,
    cacheable_methods: Collection[str] | None = None,
) -> requests.Session:
    cache = DictCache() if cache is None else cache
    adapter_class = adapter_class or CacheControlAdapter
    adapter = adapter_class(
        cache,
        cache_etags=cache_etags,
        serializer=serializer,
        heuristic=heuristic,
        controller_class=controller_class,
        cacheable_methods=cacheable_methods,
    )
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)

    return sess

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\modeline.py ===
"""
    pygments.modeline
    ~~~~~~~~~~~~~~~~~

    A simple modeline parser (based on pymodeline).

    :copyright: Copyright 2006-2025 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import re

__all__ = ['get_filetype_from_buffer']


modeline_re = re.compile(r'''
    (?: vi | vim | ex ) (?: [<=>]? \d* )? :
    .* (?: ft | filetype | syn | syntax ) = ( [^:\s]+ )
''', re.VERBOSE)


def get_filetype_from_line(l): # noqa: E741
    m = modeline_re.search(l)
    if m:
        return m.group(1)


def get_filetype_from_buffer(buf, max_lines=5):
    """
    Scan the buffer for modelines and return filetype if one is found.
    """
    lines = buf.splitlines()
    for line in lines[-1:-max_lines-1:-1]:
        ret = get_filetype_from_line(line)
        if ret:
            return ret
    for i in range(max_lines, -1, -1):
        if i < len(lines):
            ret = get_filetype_from_line(lines[i])
            if ret:
                return ret

    return None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\_loop.py ===
from typing import Iterable, Tuple, TypeVar

T = TypeVar("T")


def loop_first(values: Iterable[T]) -> Iterable[Tuple[bool, T]]:
    """Iterate and generate a tuple with a flag for first value."""
    iter_values = iter(values)
    try:
        value = next(iter_values)
    except StopIteration:
        return
    yield True, value
    for value in iter_values:
        yield False, value


def loop_last(values: Iterable[T]) -> Iterable[Tuple[bool, T]]:
    """Iterate and generate a tuple with a flag for last value."""
    iter_values = iter(values)
    try:
        previous_value = next(iter_values)
    except StopIteration:
        return
    for value in iter_values:
        yield False, previous_value
        previous_value = value
    yield True, previous_value


def loop_first_last(values: Iterable[T]) -> Iterable[Tuple[bool, bool, T]]:
    """Iterate and generate a tuple with a flag for first and last value."""
    iter_values = iter(values)
    try:
        previous_value = next(iter_values)
    except StopIteration:
        return
    first = True
    for value in iter_values:
        yield first, False, previous_value
        first = False
        previous_value = value
    yield first, True, previous_value

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\device_orientation.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: DeviceOrientation (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

def clear_device_orientation_override() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears the overridden Device Orientation.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'DeviceOrientation.clearDeviceOrientationOverride',
    }
    json = yield cmd_dict


def set_device_orientation_override(
        alpha: float,
        beta: float,
        gamma: float
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Overrides the Device Orientation.

    :param alpha: Mock alpha
    :param beta: Mock beta
    :param gamma: Mock gamma
    '''
    params: T_JSON_DICT = dict()
    params['alpha'] = alpha
    params['beta'] = beta
    params['gamma'] = gamma
    cmd_dict: T_JSON_DICT = {
        'method': 'DeviceOrientation.setDeviceOrientationOverride',
        'params': params,
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\device_orientation.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: DeviceOrientation (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

def clear_device_orientation_override() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears the overridden Device Orientation.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'DeviceOrientation.clearDeviceOrientationOverride',
    }
    json = yield cmd_dict


def set_device_orientation_override(
        alpha: float,
        beta: float,
        gamma: float
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Overrides the Device Orientation.

    :param alpha: Mock alpha
    :param beta: Mock beta
    :param gamma: Mock gamma
    '''
    params: T_JSON_DICT = dict()
    params['alpha'] = alpha
    params['beta'] = beta
    params['gamma'] = gamma
    cmd_dict: T_JSON_DICT = {
        'method': 'DeviceOrientation.setDeviceOrientationOverride',
        'params': params,
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\device_orientation.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: DeviceOrientation (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

def clear_device_orientation_override() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears the overridden Device Orientation.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'DeviceOrientation.clearDeviceOrientationOverride',
    }
    json = yield cmd_dict


def set_device_orientation_override(
        alpha: float,
        beta: float,
        gamma: float
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Overrides the Device Orientation.

    :param alpha: Mock alpha
    :param beta: Mock beta
    :param gamma: Mock gamma
    '''
    params: T_JSON_DICT = dict()
    params['alpha'] = alpha
    params['beta'] = beta
    params['gamma'] = gamma
    cmd_dict: T_JSON_DICT = {
        'method': 'DeviceOrientation.setDeviceOrientationOverride',
        'params': params,
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\__init__.py ===
from sympy.combinatorics.permutations import Permutation, Cycle
from sympy.combinatorics.prufer import Prufer
from sympy.combinatorics.generators import cyclic, alternating, symmetric, dihedral
from sympy.combinatorics.subsets import Subset
from sympy.combinatorics.partitions import (Partition, IntegerPartition,
    RGS_rank, RGS_unrank, RGS_enum)
from sympy.combinatorics.polyhedron import (Polyhedron, tetrahedron, cube,
    octahedron, dodecahedron, icosahedron)
from sympy.combinatorics.perm_groups import PermutationGroup, Coset, SymmetricPermutationGroup
from sympy.combinatorics.group_constructs import DirectProduct
from sympy.combinatorics.graycode import GrayCode
from sympy.combinatorics.named_groups import (SymmetricGroup, DihedralGroup,
    CyclicGroup, AlternatingGroup, AbelianGroup, RubikGroup)
from sympy.combinatorics.pc_groups import PolycyclicGroup, Collector
from sympy.combinatorics.free_groups import free_group

__all__ = [
    'Permutation', 'Cycle',

    'Prufer',

    'cyclic', 'alternating', 'symmetric', 'dihedral',

    'Subset',

    'Partition', 'IntegerPartition', 'RGS_rank', 'RGS_unrank', 'RGS_enum',

    'Polyhedron', 'tetrahedron', 'cube', 'octahedron', 'dodecahedron',
    'icosahedron',

    'PermutationGroup', 'Coset', 'SymmetricPermutationGroup',

    'DirectProduct',

    'GrayCode',

    'SymmetricGroup', 'DihedralGroup', 'CyclicGroup', 'AlternatingGroup',
    'AbelianGroup', 'RubikGroup',

    'PolycyclicGroup', 'Collector',

    'free_group',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\benchmarks\bench_arit.py ===
from sympy.core import Add, Mul, symbols

x, y, z = symbols('x,y,z')


def timeit_neg():
    -x


def timeit_Add_x1():
    x + 1


def timeit_Add_1x():
    1 + x


def timeit_Add_x05():
    x + 0.5


def timeit_Add_xy():
    x + y


def timeit_Add_xyz():
    Add(*[x, y, z])


def timeit_Mul_xy():
    x*y


def timeit_Mul_xyz():
    Mul(*[x, y, z])


def timeit_Div_xy():
    x/y


def timeit_Div_2y():
    2/y

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\definitions\dimension_definitions.py ===
from sympy.physics.units import Dimension


angle: Dimension = Dimension(name="angle")

# base dimensions (MKS)
length = Dimension(name="length", symbol="L")
mass = Dimension(name="mass", symbol="M")
time = Dimension(name="time", symbol="T")

# base dimensions (MKSA not in MKS)
current: Dimension = Dimension(name='current', symbol='I')

# other base dimensions:
temperature: Dimension = Dimension("temperature", "T")
amount_of_substance: Dimension = Dimension("amount_of_substance")
luminous_intensity: Dimension = Dimension("luminous_intensity")

# derived dimensions (MKS)
velocity = Dimension(name="velocity")
acceleration = Dimension(name="acceleration")
momentum = Dimension(name="momentum")
force = Dimension(name="force", symbol="F")
energy = Dimension(name="energy", symbol="E")
power = Dimension(name="power")
pressure = Dimension(name="pressure")
frequency = Dimension(name="frequency", symbol="f")
action = Dimension(name="action", symbol="A")
area = Dimension("area")
volume = Dimension("volume")

# derived dimensions (MKSA not in MKS)
voltage: Dimension = Dimension(name='voltage', symbol='U')
impedance: Dimension = Dimension(name='impedance', symbol='Z')
conductance: Dimension = Dimension(name='conductance', symbol='G')
capacitance: Dimension = Dimension(name='capacitance')
inductance: Dimension = Dimension(name='inductance')
charge: Dimension = Dimension(name='charge', symbol='Q')
magnetic_density: Dimension = Dimension(name='magnetic_density', symbol='B')
magnetic_flux: Dimension = Dimension(name='magnetic_flux')

# Dimensions in information theory:
information: Dimension = Dimension(name='information')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\util\proxy.py ===
from __future__ import annotations

import typing

from .url import Url

if typing.TYPE_CHECKING:
    from ..connection import ProxyConfig


def connection_requires_http_tunnel(
    proxy_url: Url | None = None,
    proxy_config: ProxyConfig | None = None,
    destination_scheme: str | None = None,
) -> bool:
    """
    Returns True if the connection requires an HTTP CONNECT through the proxy.

    :param URL proxy_url:
        URL of the proxy.
    :param ProxyConfig proxy_config:
        Proxy configuration from poolmanager.py
    :param str destination_scheme:
        The scheme of the destination. (i.e https, http, etc)
    """
    # If we're not using a proxy, no way to use a tunnel.
    if proxy_url is None:
        return False

    # HTTP destinations never require tunneling, we always forward.
    if destination_scheme == "http":
        return False

    # Support for forwarding with HTTPS proxies and HTTPS destinations.
    if (
        proxy_url.scheme == "https"
        and proxy_config
        and proxy_config.use_forwarding_for_https
    ):
        return False

    # Otherwise always use a tunnel.
    return True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_to_csv.py ===
import csv
from io import StringIO
import os

import numpy as np
import pytest

from pandas.errors import ParserError

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    NaT,
    Series,
    Timestamp,
    date_range,
    period_range,
    read_csv,
    to_datetime,
)
import pandas._testing as tm
import pandas.core.common as com

from pandas.io.common import get_handle


class TestDataFrameToCSV:
    def read_csv(self, path, **kwargs):
        params = {"index_col": 0}
        params.update(**kwargs)

        return read_csv(path, **params)

    def test_to_csv_from_csv1(self, float_frame, datetime_frame):
        with tm.ensure_clean("__tmp_to_csv_from_csv1__") as path:
            float_frame.iloc[:5, float_frame.columns.get_loc("A")] = np.nan

            float_frame.to_csv(path)
            float_frame.to_csv(path, columns=["A", "B"])
            float_frame.to_csv(path, header=False)
            float_frame.to_csv(path, index=False)

            # test roundtrip
            # freq does not roundtrip
            datetime_frame.index = datetime_frame.index._with_freq(None)
            datetime_frame.to_csv(path)
            recons = self.read_csv(path, parse_dates=True)
            tm.assert_frame_equal(datetime_frame, recons)

            datetime_frame.to_csv(path, index_label="index")
            recons = self.read_csv(path, index_col=None, parse_dates=True)

            assert len(recons.columns) == len(datetime_frame.columns) + 1

            # no index
            datetime_frame.to_csv(path, index=False)
            recons = self.read_csv(path, index_col=None, parse_dates=True)
            tm.assert_almost_equal(datetime_frame.values, recons.values)

            # corner case
            dm = DataFrame(
                {
                    "s1": Series(range(3), index=np.arange(3, dtype=np.int64)),
                    "s2": Series(range(2), index=np.arange(2, dtype=np.int64)),
                }
            )
            dm.to_csv(path)

            recons = self.read_csv(path)
            tm.assert_frame_equal(dm, recons)

    def test_to_csv_from_csv2(self, float_frame):
        with tm.ensure_clean("__tmp_to_csv_from_csv2__") as path:
            # duplicate index
            df = DataFrame(
                np.random.default_rng(2).standard_normal((3, 3)),
                index=["a", "a", "b"],
                columns=["x", "y", "z"],
            )
            df.to_csv(path)
            result = self.read_csv(path)
            tm.assert_frame_equal(result, df)

            midx = MultiIndex.from_tuples([("A", 1, 2), ("A", 1, 2), ("B", 1, 2)])
            df = DataFrame(
                np.random.default_rng(2).standard_normal((3, 3)),
                index=midx,
                columns=["x", "y", "z"],
            )

            df.to_csv(path)
            result = self.read_csv(path, index_col=[0, 1, 2], parse_dates=False)
            tm.assert_frame_equal(result, df, check_names=False)

            # column aliases
            col_aliases = Index(["AA", "X", "Y", "Z"])
            float_frame.to_csv(path, header=col_aliases)

            rs = self.read_csv(path)
            xp = float_frame.copy()
            xp.columns = col_aliases
            tm.assert_frame_equal(xp, rs)

            msg = "Writing 4 cols but got 2 aliases"
            with pytest.raises(ValueError, match=msg):
                float_frame.to_csv(path, header=["AA", "X"])

    def test_to_csv_from_csv3(self):
        with tm.ensure_clean("__tmp_to_csv_from_csv3__") as path:
            df1 = DataFrame(np.random.default_rng(2).standard_normal((3, 1)))
            df2 = DataFrame(np.random.default_rng(2).standard_normal((3, 1)))

            df1.to_csv(path)
            df2.to_csv(path, mode="a", header=False)
            xp = pd.concat([df1, df2])
            rs = read_csv(path, index_col=0)
            rs.columns = [int(label) for label in rs.columns]
            xp.columns = [int(label) for label in xp.columns]
            tm.assert_frame_equal(xp, rs)

    def test_to_csv_from_csv4(self):
        with tm.ensure_clean("__tmp_to_csv_from_csv4__") as path:
            # GH 10833 (TimedeltaIndex formatting)
            dt = pd.Timedelta(seconds=1)
            df = DataFrame(
                {"dt_data": [i * dt for i in range(3)]},
                index=Index([i * dt for i in range(3)], name="dt_index"),
            )
            df.to_csv(path)

            result = read_csv(path, index_col="dt_index")
            result.index = pd.to_timedelta(result.index)
            result["dt_data"] = pd.to_timedelta(result["dt_data"])

            tm.assert_frame_equal(df, result, check_index_type=True)

    def test_to_csv_from_csv5(self, timezone_frame):
        # tz, 8260
        with tm.ensure_clean("__tmp_to_csv_from_csv5__") as path:
            timezone_frame.to_csv(path)
            result = read_csv(path, index_col=0, parse_dates=["A"])

            converter = (
                lambda c: to_datetime(result[c])
                .dt.tz_convert("UTC")
                .dt.tz_convert(timezone_frame[c].dt.tz)
            )
            result["B"] = converter("B")
            result["C"] = converter("C")
            tm.assert_frame_equal(result, timezone_frame)

    def test_to_csv_cols_reordering(self):
        # GH3454
        chunksize = 5
        N = int(chunksize * 2.5)

        df = DataFrame(
            np.ones((N, 3)),
            index=Index([f"i-{i}" for i in range(N)], name="a"),
            columns=Index([f"i-{i}" for i in range(3)], name="a"),
        )
        cs = df.columns
        cols = [cs[2], cs[0]]

        with tm.ensure_clean() as path:
            df.to_csv(path, columns=cols, chunksize=chunksize)
            rs_c = read_csv(path, index_col=0)

        tm.assert_frame_equal(df[cols], rs_c, check_names=False)

    @pytest.mark.parametrize("cols", [None, ["b", "a"]])
    def test_to_csv_new_dupe_cols(self, cols):
        chunksize = 5
        N = int(chunksize * 2.5)

        # dupe cols
        df = DataFrame(
            np.ones((N, 3)),
            index=Index([f"i-{i}" for i in range(N)], name="a"),
            columns=["a", "a", "b"],
        )
        with tm.ensure_clean() as path:
            df.to_csv(path, columns=cols, chunksize=chunksize)
            rs_c = read_csv(path, index_col=0)

            # we wrote them in a different order
            # so compare them in that order
            if cols is not None:
                if df.columns.is_unique:
                    rs_c.columns = cols
                else:
                    indexer, missing = df.columns.get_indexer_non_unique(cols)
                    rs_c.columns = df.columns.take(indexer)

                for c in cols:
                    obj_df = df[c]
                    obj_rs = rs_c[c]
                    if isinstance(obj_df, Series):
                        tm.assert_series_equal(obj_df, obj_rs)
                    else:
                        tm.assert_frame_equal(obj_df, obj_rs, check_names=False)

            # wrote in the same order
            else:
                rs_c.columns = df.columns
                tm.assert_frame_equal(df, rs_c, check_names=False)

    @pytest.mark.slow
    def test_to_csv_dtnat(self):
        # GH3437
        def make_dtnat_arr(n, nnat=None):
            if nnat is None:
                nnat = int(n * 0.1)  # 10%
            s = list(date_range("2000", freq="5min", periods=n))
            if nnat:
                for i in np.random.default_rng(2).integers(0, len(s), nnat):
                    s[i] = NaT
                i = np.random.default_rng(2).integers(100)
                s[-i] = NaT
                s[i] = NaT
            return s

        chunksize = 1000
        s1 = make_dtnat_arr(chunksize + 5)
        s2 = make_dtnat_arr(chunksize + 5, 0)

        with tm.ensure_clean("1.csv") as pth:
            df = DataFrame({"a": s1, "b": s2})
            df.to_csv(pth, chunksize=chunksize)

            recons = self.read_csv(pth).apply(to_datetime)
            tm.assert_frame_equal(df, recons, check_names=False)

    def _return_result_expected(
        self,
        df,
        chunksize,
        r_dtype=None,
        c_dtype=None,
        rnlvl=None,
        cnlvl=None,
        dupe_col=False,
    ):
        kwargs = {"parse_dates": False}
        if cnlvl:
            if rnlvl is not None:
                kwargs["index_col"] = list(range(rnlvl))
            kwargs["header"] = list(range(cnlvl))

            with tm.ensure_clean("__tmp_to_csv_moar__") as path:
                df.to_csv(path, encoding="utf8", chunksize=chunksize)
                recons = self.read_csv(path, **kwargs)
        else:
            kwargs["header"] = 0

            with tm.ensure_clean("__tmp_to_csv_moar__") as path:
                df.to_csv(path, encoding="utf8", chunksize=chunksize)
                recons = self.read_csv(path, **kwargs)

        def _to_uni(x):
            if not isinstance(x, str):
                return x.decode("utf8")
            return x

        if dupe_col:
            # read_Csv disambiguates the columns by
            # labeling them dupe.1,dupe.2, etc'. monkey patch columns
            recons.columns = df.columns
        if rnlvl and not cnlvl:
            delta_lvl = [recons.iloc[:, i].values for i in range(rnlvl - 1)]
            ix = MultiIndex.from_arrays([list(recons.index)] + delta_lvl)
            recons.index = ix
            recons = recons.iloc[:, rnlvl - 1 :]

        type_map = {"i": "i", "f": "f", "s": "O", "u": "O", "dt": "O", "p": "O"}
        if r_dtype:
            if r_dtype == "u":  # unicode
                r_dtype = "O"
                recons.index = np.array(
                    [_to_uni(label) for label in recons.index], dtype=r_dtype
                )
                df.index = np.array(
                    [_to_uni(label) for label in df.index], dtype=r_dtype
                )
            elif r_dtype == "dt":  # unicode
                r_dtype = "O"
                recons.index = np.array(
                    [Timestamp(label) for label in recons.index], dtype=r_dtype
                )
                df.index = np.array(
                    [Timestamp(label) for label in df.index], dtype=r_dtype
                )
            elif r_dtype == "p":
                r_dtype = "O"
                idx_list = to_datetime(recons.index)
                recons.index = np.array(
                    [Timestamp(label) for label in idx_list], dtype=r_dtype
                )
                df.index = np.array(
                    list(map(Timestamp, df.index.to_timestamp())), dtype=r_dtype
                )
            else:
                r_dtype = type_map.get(r_dtype)
                recons.index = np.array(recons.index, dtype=r_dtype)
                df.index = np.array(df.index, dtype=r_dtype)
        if c_dtype:
            if c_dtype == "u":
                c_dtype = "O"
                recons.columns = np.array(
                    [_to_uni(label) for label in recons.columns], dtype=c_dtype
                )
                df.columns = np.array(
                    [_to_uni(label) for label in df.columns], dtype=c_dtype
                )
            elif c_dtype == "dt":
                c_dtype = "O"
                recons.columns = np.array(
                    [Timestamp(label) for label in recons.columns], dtype=c_dtype
                )
                df.columns = np.array(
                    [Timestamp(label) for label in df.columns], dtype=c_dtype
                )
            elif c_dtype == "p":
                c_dtype = "O"
                col_list = to_datetime(recons.columns)
                recons.columns = np.array(
                    [Timestamp(label) for label in col_list], dtype=c_dtype
                )
                col_list = df.columns.to_timestamp()
                df.columns = np.array(
                    [Timestamp(label) for label in col_list], dtype=c_dtype
                )
            else:
                c_dtype = type_map.get(c_dtype)
                recons.columns = np.array(recons.columns, dtype=c_dtype)
                df.columns = np.array(df.columns, dtype=c_dtype)
        return df, recons

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "nrows", [2, 10, 99, 100, 101, 102, 198, 199, 200, 201, 202, 249, 250, 251]
    )
    def test_to_csv_nrows(self, nrows):
        df = DataFrame(
            np.ones((nrows, 4)),
            index=date_range("2020-01-01", periods=nrows),
            columns=Index(list("abcd"), dtype=object),
        )
        result, expected = self._return_result_expected(df, 1000, "dt", "s")
        tm.assert_frame_equal(result, expected, check_names=False)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "nrows", [2, 10, 99, 100, 101, 102, 198, 199, 200, 201, 202, 249, 250, 251]
    )
    @pytest.mark.parametrize(
        "r_idx_type, c_idx_type", [("i", "i"), ("s", "s"), ("s", "dt"), ("p", "p")]
    )
    @pytest.mark.parametrize("ncols", [1, 2, 3, 4])
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_to_csv_idx_types(self, nrows, r_idx_type, c_idx_type, ncols):
        axes = {
            "i": lambda n: Index(np.arange(n), dtype=np.int64),
            "s": lambda n: Index([f"{i}_{chr(i)}" for i in range(97, 97 + n)]),
            "dt": lambda n: date_range("2020-01-01", periods=n),
            "p": lambda n: period_range("2020-01-01", periods=n, freq="D"),
        }
        df = DataFrame(
            np.ones((nrows, ncols)),
            index=axes[r_idx_type](nrows),
            columns=axes[c_idx_type](ncols),
        )
        result, expected = self._return_result_expected(
            df,
            1000,
            r_idx_type,
            c_idx_type,
        )
        tm.assert_frame_equal(result, expected, check_names=False)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "nrows", [10, 98, 99, 100, 101, 102, 198, 199, 200, 201, 202, 249, 250, 251]
    )
    @pytest.mark.parametrize("ncols", [1, 2, 3, 4])
    def test_to_csv_idx_ncols(self, nrows, ncols):
        df = DataFrame(
            np.ones((nrows, ncols)),
            index=Index([f"i-{i}" for i in range(nrows)], name="a"),
            columns=Index([f"i-{i}" for i in range(ncols)], name="a"),
        )
        result, expected = self._return_result_expected(df, 1000)
        tm.assert_frame_equal(result, expected, check_names=False)

    @pytest.mark.slow
    @pytest.mark.parametrize("nrows", [10, 98, 99, 100, 101, 102])
    def test_to_csv_dup_cols(self, nrows):
        df = DataFrame(
            np.ones((nrows, 3)),
            index=Index([f"i-{i}" for i in range(nrows)], name="a"),
            columns=Index([f"i-{i}" for i in range(3)], name="a"),
        )

        cols = list(df.columns)
        cols[:2] = ["dupe", "dupe"]
        cols[-2:] = ["dupe", "dupe"]
        ix = list(df.index)
        ix[:2] = ["rdupe", "rdupe"]
        ix[-2:] = ["rdupe", "rdupe"]
        df.index = ix
        df.columns = cols
        result, expected = self._return_result_expected(df, 1000, dupe_col=True)
        tm.assert_frame_equal(result, expected, check_names=False)

    @pytest.mark.slow
    def test_to_csv_empty(self):
        df = DataFrame(index=np.arange(10, dtype=np.int64))
        result, expected = self._return_result_expected(df, 1000)
        tm.assert_frame_equal(result, expected, check_column_type=False)

    @pytest.mark.slow
    def test_to_csv_chunksize(self):
        chunksize = 1000
        rows = chunksize // 2 + 1
        df = DataFrame(
            np.ones((rows, 2)),
            columns=Index(list("ab")),
            index=MultiIndex.from_arrays([range(rows) for _ in range(2)]),
        )
        result, expected = self._return_result_expected(df, chunksize, rnlvl=2)
        tm.assert_frame_equal(result, expected, check_names=False)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "nrows", [2, 10, 99, 100, 101, 102, 198, 199, 200, 201, 202, 249, 250, 251]
    )
    @pytest.mark.parametrize("ncols", [2, 3, 4])
    @pytest.mark.parametrize(
        "df_params, func_params",
        [
            [{"r_idx_nlevels": 2}, {"rnlvl": 2}],
            [{"c_idx_nlevels": 2}, {"cnlvl": 2}],
            [{"r_idx_nlevels": 2, "c_idx_nlevels": 2}, {"rnlvl": 2, "cnlvl": 2}],
        ],
    )
    def test_to_csv_params(self, nrows, df_params, func_params, ncols):
        if df_params.get("r_idx_nlevels"):
            index = MultiIndex.from_arrays(
                [f"i-{i}" for i in range(nrows)]
                for _ in range(df_params["r_idx_nlevels"])
            )
        else:
            index = None

        if df_params.get("c_idx_nlevels"):
            columns = MultiIndex.from_arrays(
                [f"i-{i}" for i in range(ncols)]
                for _ in range(df_params["c_idx_nlevels"])
            )
        else:
            columns = Index([f"i-{i}" for i in range(ncols)])
        df = DataFrame(np.ones((nrows, ncols)), index=index, columns=columns)
        result, expected = self._return_result_expected(df, 1000, **func_params)
        tm.assert_frame_equal(result, expected, check_names=False)

    def test_to_csv_from_csv_w_some_infs(self, float_frame):
        # test roundtrip with inf, -inf, nan, as full columns and mix
        float_frame["G"] = np.nan
        f = lambda x: [np.inf, np.nan][np.random.default_rng(2).random() < 0.5]
        float_frame["h"] = float_frame.index.map(f)

        with tm.ensure_clean() as path:
            float_frame.to_csv(path)
            recons = self.read_csv(path)

            tm.assert_frame_equal(float_frame, recons)
            tm.assert_frame_equal(np.isinf(float_frame), np.isinf(recons))

    def test_to_csv_from_csv_w_all_infs(self, float_frame):
        # test roundtrip with inf, -inf, nan, as full columns and mix
        float_frame["E"] = np.inf
        float_frame["F"] = -np.inf

        with tm.ensure_clean() as path:
            float_frame.to_csv(path)
            recons = self.read_csv(path)

            tm.assert_frame_equal(float_frame, recons)
            tm.assert_frame_equal(np.isinf(float_frame), np.isinf(recons))

    def test_to_csv_no_index(self):
        # GH 3624, after appending columns, to_csv fails
        with tm.ensure_clean("__tmp_to_csv_no_index__") as path:
            df = DataFrame({"c1": [1, 2, 3], "c2": [4, 5, 6]})
            df.to_csv(path, index=False)
            result = read_csv(path)
            tm.assert_frame_equal(df, result)
            df["c3"] = Series([7, 8, 9], dtype="int64")
            df.to_csv(path, index=False)
            result = read_csv(path)
            tm.assert_frame_equal(df, result)

    def test_to_csv_with_mix_columns(self):
        # gh-11637: incorrect output when a mix of integer and string column
        # names passed as columns parameter in to_csv

        df = DataFrame({0: ["a", "b", "c"], 1: ["aa", "bb", "cc"]})
        df["test"] = "txt"
        assert df.to_csv() == df.to_csv(columns=[0, 1, "test"])

    def test_to_csv_headers(self):
        # GH6186, the presence or absence of `index` incorrectly
        # causes to_csv to have different header semantics.
        from_df = DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        to_df = DataFrame([[1, 2], [3, 4]], columns=["X", "Y"])
        with tm.ensure_clean("__tmp_to_csv_headers__") as path:
            from_df.to_csv(path, header=["X", "Y"])
            recons = self.read_csv(path)

            tm.assert_frame_equal(to_df, recons)

            from_df.to_csv(path, index=False, header=["X", "Y"])
            recons = self.read_csv(path)

            return_value = recons.reset_index(inplace=True)
            assert return_value is None
            tm.assert_frame_equal(to_df, recons)

    def test_to_csv_multiindex(self, float_frame, datetime_frame):
        frame = float_frame
        old_index = frame.index
        arrays = np.arange(len(old_index) * 2, dtype=np.int64).reshape(2, -1)
        new_index = MultiIndex.from_arrays(arrays, names=["first", "second"])
        frame.index = new_index

        with tm.ensure_clean("__tmp_to_csv_multiindex__") as path:
            frame.to_csv(path, header=False)
            frame.to_csv(path, columns=["A", "B"])

            # round trip
            frame.to_csv(path)

            df = self.read_csv(path, index_col=[0, 1], parse_dates=False)

            # TODO to_csv drops column name
            tm.assert_frame_equal(frame, df, check_names=False)
            assert frame.index.names == df.index.names

            # needed if setUp becomes a class method
            float_frame.index = old_index

            # try multiindex with dates
            tsframe = datetime_frame
            old_index = tsframe.index
            new_index = [old_index, np.arange(len(old_index), dtype=np.int64)]
            tsframe.index = MultiIndex.from_arrays(new_index)

            tsframe.to_csv(path, index_label=["time", "foo"])
            with tm.assert_produces_warning(
                UserWarning, match="Could not infer format"
            ):
                recons = self.read_csv(path, index_col=[0, 1], parse_dates=True)

            # TODO to_csv drops column name
            tm.assert_frame_equal(tsframe, recons, check_names=False)

            # do not load index
            tsframe.to_csv(path)
            recons = self.read_csv(path, index_col=None)
            assert len(recons.columns) == len(tsframe.columns) + 2

            # no index
            tsframe.to_csv(path, index=False)
            recons = self.read_csv(path, index_col=None)
            tm.assert_almost_equal(recons.values, datetime_frame.values)

            # needed if setUp becomes class method
            datetime_frame.index = old_index

        with tm.ensure_clean("__tmp_to_csv_multiindex__") as path:
            # GH3571, GH1651, GH3141

            def _make_frame(names=None):
                if names is True:
                    names = ["first", "second"]
                return DataFrame(
                    np.random.default_rng(2).integers(0, 10, size=(3, 3)),
                    columns=MultiIndex.from_tuples(
                        [("bah", "foo"), ("bah", "bar"), ("ban", "baz")], names=names
                    ),
                    dtype="int64",
                )

            # column & index are multi-index
            df = DataFrame(
                np.ones((5, 3)),
                columns=MultiIndex.from_arrays(
                    [[f"i-{i}" for i in range(3)] for _ in range(4)], names=list("abcd")
                ),
                index=MultiIndex.from_arrays(
                    [[f"i-{i}" for i in range(5)] for _ in range(2)], names=list("ab")
                ),
            )
            df.to_csv(path)
            result = read_csv(path, header=[0, 1, 2, 3], index_col=[0, 1])
            tm.assert_frame_equal(df, result)

            # column is mi
            df = DataFrame(
                np.ones((5, 3)),
                columns=MultiIndex.from_arrays(
                    [[f"i-{i}" for i in range(3)] for _ in range(4)], names=list("abcd")
                ),
            )
            df.to_csv(path)
            result = read_csv(path, header=[0, 1, 2, 3], index_col=0)
            tm.assert_frame_equal(df, result)

            # dup column names?
            df = DataFrame(
                np.ones((5, 3)),
                columns=MultiIndex.from_arrays(
                    [[f"i-{i}" for i in range(3)] for _ in range(4)], names=list("abcd")
                ),
                index=MultiIndex.from_arrays(
                    [[f"i-{i}" for i in range(5)] for _ in range(3)], names=list("abc")
                ),
            )
            df.to_csv(path)
            result = read_csv(path, header=[0, 1, 2, 3], index_col=[0, 1, 2])
            tm.assert_frame_equal(df, result)

            # writing with no index
            df = _make_frame()
            df.to_csv(path, index=False)
            result = read_csv(path, header=[0, 1])
            tm.assert_frame_equal(df, result)

            # we lose the names here
            df = _make_frame(True)
            df.to_csv(path, index=False)
            result = read_csv(path, header=[0, 1])
            assert com.all_none(*result.columns.names)
            result.columns.names = df.columns.names
            tm.assert_frame_equal(df, result)

            # whatsnew example
            df = _make_frame()
            df.to_csv(path)
            result = read_csv(path, header=[0, 1], index_col=[0])
            tm.assert_frame_equal(df, result)

            df = _make_frame(True)
            df.to_csv(path)
            result = read_csv(path, header=[0, 1], index_col=[0])
            tm.assert_frame_equal(df, result)

            # invalid options
            df = _make_frame(True)
            df.to_csv(path)

            for i in [6, 7]:
                msg = f"len of {i}, but only 5 lines in file"
                with pytest.raises(ParserError, match=msg):
                    read_csv(path, header=list(range(i)), index_col=0)

            # write with cols
            msg = "cannot specify cols with a MultiIndex"
            with pytest.raises(TypeError, match=msg):
                df.to_csv(path, columns=["foo", "bar"])

        with tm.ensure_clean("__tmp_to_csv_multiindex__") as path:
            # empty
            tsframe[:0].to_csv(path)
            recons = self.read_csv(path)

            exp = tsframe[:0]
            exp.index = []

            tm.assert_index_equal(recons.columns, exp.columns)
            assert len(recons) == 0

    def test_to_csv_interval_index(self, using_infer_string):
        # GH 28210
        df = DataFrame({"A": list("abc"), "B": range(3)}, index=pd.interval_range(0, 3))

        with tm.ensure_clean("__tmp_to_csv_interval_index__.csv") as path:
            df.to_csv(path)
            result = self.read_csv(path, index_col=0)

            # can't roundtrip intervalindex via read_csv so check string repr (GH 23595)
            expected = df.copy()
            expected.index = expected.index.astype("str")

            tm.assert_frame_equal(result, expected)

    def test_to_csv_float32_nanrep(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((1, 4)).astype(np.float32)
        )
        df[1] = np.nan

        with tm.ensure_clean("__tmp_to_csv_float32_nanrep__.csv") as path:
            df.to_csv(path, na_rep=999)

            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
                assert lines[1].split(",")[2] == "999"

    def test_to_csv_withcommas(self):
        # Commas inside fields should be correctly escaped when saving as CSV.
        df = DataFrame({"A": [1, 2, 3], "B": ["5,6", "7,8", "9,0"]})

        with tm.ensure_clean("__tmp_to_csv_withcommas__.csv") as path:
            df.to_csv(path)
            df2 = self.read_csv(path)
            tm.assert_frame_equal(df2, df)

    def test_to_csv_mixed(self):
        def create_cols(name):
            return [f"{name}{i:03d}" for i in range(5)]

        df_float = DataFrame(
            np.random.default_rng(2).standard_normal((100, 5)),
            dtype="float64",
            columns=create_cols("float"),
        )
        df_int = DataFrame(
            np.random.default_rng(2).standard_normal((100, 5)).astype("int64"),
            dtype="int64",
            columns=create_cols("int"),
        )
        df_bool = DataFrame(True, index=df_float.index, columns=create_cols("bool"))
        df_object = DataFrame(
            "foo", index=df_float.index, columns=create_cols("object"), dtype="object"
        )
        df_dt = DataFrame(
            Timestamp("20010101").as_unit("ns"),
            index=df_float.index,
            columns=create_cols("date"),
        )

        # add in some nans
        df_float.iloc[30:50, 1:3] = np.nan
        df_dt.iloc[30:50, 1:3] = np.nan

        df = pd.concat([df_float, df_int, df_bool, df_object, df_dt], axis=1)

        # dtype
        dtypes = {}
        for n, dtype in [
            ("float", np.float64),
            ("int", np.int64),
            ("bool", np.bool_),
            ("object", object),
        ]:
            for c in create_cols(n):
                dtypes[c] = dtype

        with tm.ensure_clean() as filename:
            df.to_csv(filename)
            rs = read_csv(
                filename, index_col=0, dtype=dtypes, parse_dates=create_cols("date")
            )
            tm.assert_frame_equal(rs, df)

    def test_to_csv_dups_cols(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((1000, 30)),
            columns=list(range(15)) + list(range(15)),
            dtype="float64",
        )

        with tm.ensure_clean() as filename:
            df.to_csv(filename)  # single dtype, fine
            result = read_csv(filename, index_col=0)
            result.columns = df.columns
            tm.assert_frame_equal(result, df)

        df_float = DataFrame(
            np.random.default_rng(2).standard_normal((1000, 3)), dtype="float64"
        )
        df_int = DataFrame(np.random.default_rng(2).standard_normal((1000, 3))).astype(
            "int64"
        )
        df_bool = DataFrame(True, index=df_float.index, columns=range(3))
        df_object = DataFrame("foo", index=df_float.index, columns=range(3))
        df_dt = DataFrame(
            Timestamp("20010101").as_unit("ns"), index=df_float.index, columns=range(3)
        )
        df = pd.concat(
            [df_float, df_int, df_bool, df_object, df_dt], axis=1, ignore_index=True
        )

        df.columns = [0, 1, 2] * 5

        with tm.ensure_clean() as filename:
            df.to_csv(filename)
            result = read_csv(filename, index_col=0)

            # date cols
            for i in ["0.4", "1.4", "2.4"]:
                result[i] = to_datetime(result[i])

            result.columns = df.columns
            tm.assert_frame_equal(result, df)

    def test_to_csv_dups_cols2(self):
        # GH3457
        df = DataFrame(
            np.ones((5, 3)),
            index=Index([f"i-{i}" for i in range(5)], name="foo"),
            columns=Index(["a", "a", "b"]),
        )

        with tm.ensure_clean() as filename:
            df.to_csv(filename)

            # read_csv will rename the dups columns
            result = read_csv(filename, index_col=0)
            result = result.rename(columns={"a.1": "a"})
            tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize("chunksize", [10000, 50000, 100000])
    def test_to_csv_chunking(self, chunksize):
        aa = DataFrame({"A": range(100000)})
        aa["B"] = aa.A + 1.0
        aa["C"] = aa.A + 2.0
        aa["D"] = aa.A + 3.0

        with tm.ensure_clean() as filename:
            aa.to_csv(filename, chunksize=chunksize)
            rs = read_csv(filename, index_col=0)
            tm.assert_frame_equal(rs, aa)

    @pytest.mark.slow
    def test_to_csv_wide_frame_formatting(self, monkeypatch):
        # Issue #8621
        chunksize = 100
        df = DataFrame(
            np.random.default_rng(2).standard_normal((1, chunksize + 10)),
            columns=None,
            index=None,
        )
        with tm.ensure_clean() as filename:
            with monkeypatch.context() as m:
                m.setattr("pandas.io.formats.csvs._DEFAULT_CHUNKSIZE_CELLS", chunksize)
                df.to_csv(filename, header=False, index=False)
            rs = read_csv(filename, header=None)
        tm.assert_frame_equal(rs, df)

    def test_to_csv_bug(self):
        f1 = StringIO("a,1.0\nb,2.0")
        df = self.read_csv(f1, header=None)
        newdf = DataFrame({"t": df[df.columns[0]]})

        with tm.ensure_clean() as path:
            newdf.to_csv(path)

            recons = read_csv(path, index_col=0)
            # don't check_names as t != 1
            tm.assert_frame_equal(recons, newdf, check_names=False)

    def test_to_csv_unicode(self):
        df = DataFrame({"c/\u03c3": [1, 2, 3]})
        with tm.ensure_clean() as path:
            df.to_csv(path, encoding="UTF-8")
            df2 = read_csv(path, index_col=0, encoding="UTF-8")
            tm.assert_frame_equal(df, df2)

            df.to_csv(path, encoding="UTF-8", index=False)
            df2 = read_csv(path, index_col=None, encoding="UTF-8")
            tm.assert_frame_equal(df, df2)

    def test_to_csv_unicode_index_col(self):
        buf = StringIO("")
        df = DataFrame(
            [["\u05d0", "d2", "d3", "d4"], ["a1", "a2", "a3", "a4"]],
            columns=["\u05d0", "\u05d1", "\u05d2", "\u05d3"],
            index=["\u05d0", "\u05d1"],
        )

        df.to_csv(buf, encoding="UTF-8")
        buf.seek(0)

        df2 = read_csv(buf, index_col=0, encoding="UTF-8")
        tm.assert_frame_equal(df, df2)

    def test_to_csv_stringio(self, float_frame):
        buf = StringIO()
        float_frame.to_csv(buf)
        buf.seek(0)
        recons = read_csv(buf, index_col=0)
        tm.assert_frame_equal(recons, float_frame)

    def test_to_csv_float_format(self):
        df = DataFrame(
            [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            index=["A", "B"],
            columns=["X", "Y", "Z"],
        )

        with tm.ensure_clean() as filename:
            df.to_csv(filename, float_format="%.2f")

            rs = read_csv(filename, index_col=0)
            xp = DataFrame(
                [[0.12, 0.23, 0.57], [12.32, 123123.20, 321321.20]],
                index=["A", "B"],
                columns=["X", "Y", "Z"],
            )
            tm.assert_frame_equal(rs, xp)

    def test_to_csv_float_format_over_decimal(self):
        # GH#47436
        df = DataFrame({"a": [0.5, 1.0]})
        result = df.to_csv(
            decimal=",",
            float_format=lambda x: np.format_float_positional(x, trim="-"),
            index=False,
        )
        expected_rows = ["a", "0.5", "1"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

    def test_to_csv_unicodewriter_quoting(self):
        df = DataFrame({"A": [1, 2, 3], "B": ["foo", "bar", "baz"]})

        buf = StringIO()
        df.to_csv(buf, index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

        result = buf.getvalue()
        expected_rows = ['"A","B"', '1,"foo"', '2,"bar"', '3,"baz"']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

    @pytest.mark.parametrize("encoding", [None, "utf-8"])
    def test_to_csv_quote_none(self, encoding):
        # GH4328
        df = DataFrame({"A": ["hello", '{"hello"}']})
        buf = StringIO()
        df.to_csv(buf, quoting=csv.QUOTE_NONE, encoding=encoding, index=False)

        result = buf.getvalue()
        expected_rows = ["A", "hello", '{"hello"}']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

    def test_to_csv_index_no_leading_comma(self):
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["one", "two", "three"])

        buf = StringIO()
        df.to_csv(buf, index_label=False)

        expected_rows = ["A,B", "one,1,4", "two,2,5", "three,3,6"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert buf.getvalue() == expected

    def test_to_csv_lineterminators(self):
        # see gh-20353
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["one", "two", "three"])

        with tm.ensure_clean() as path:
            # case 1: CRLF as line terminator
            df.to_csv(path, lineterminator="\r\n")
            expected = b",A,B\r\none,1,4\r\ntwo,2,5\r\nthree,3,6\r\n"

            with open(path, mode="rb") as f:
                assert f.read() == expected

        with tm.ensure_clean() as path:
            # case 2: LF as line terminator
            df.to_csv(path, lineterminator="\n")
            expected = b",A,B\none,1,4\ntwo,2,5\nthree,3,6\n"

            with open(path, mode="rb") as f:
                assert f.read() == expected

        with tm.ensure_clean() as path:
            # case 3: The default line terminator(=os.linesep)(gh-21406)
            df.to_csv(path)
            os_linesep = os.linesep.encode("utf-8")
            expected = (
                b",A,B"
                + os_linesep
                + b"one,1,4"
                + os_linesep
                + b"two,2,5"
                + os_linesep
                + b"three,3,6"
                + os_linesep
            )

            with open(path, mode="rb") as f:
                assert f.read() == expected

    def test_to_csv_from_csv_categorical(self):
        # CSV with categoricals should result in the same output
        # as when one would add a "normal" Series/DataFrame.
        s = Series(pd.Categorical(["a", "b", "b", "a", "a", "c", "c", "c"]))
        s2 = Series(["a", "b", "b", "a", "a", "c", "c", "c"])
        res = StringIO()

        s.to_csv(res, header=False)
        exp = StringIO()

        s2.to_csv(exp, header=False)
        assert res.getvalue() == exp.getvalue()

        df = DataFrame({"s": s})
        df2 = DataFrame({"s": s2})

        res = StringIO()
        df.to_csv(res)

        exp = StringIO()
        df2.to_csv(exp)

        assert res.getvalue() == exp.getvalue()

    def test_to_csv_path_is_none(self, float_frame):
        # GH 8215
        # Make sure we return string for consistency with
        # Series.to_csv()
        csv_str = float_frame.to_csv(path_or_buf=None)
        assert isinstance(csv_str, str)
        recons = read_csv(StringIO(csv_str), index_col=0)
        tm.assert_frame_equal(float_frame, recons)

    @pytest.mark.parametrize(
        "df,encoding",
        [
            (
                DataFrame(
                    [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
                    index=["A", "B"],
                    columns=["X", "Y", "Z"],
                ),
                None,
            ),
            # GH 21241, 21118
            (DataFrame([["abc", "def", "ghi"]], columns=["X", "Y", "Z"]), "ascii"),
            (DataFrame(5 * [[123, "你好", "世界"]], columns=["X", "Y", "Z"]), "gb2312"),
            (
                DataFrame(
                    5 * [[123, "Γειά σου", "Κόσμε"]],  # noqa: RUF001
                    columns=["X", "Y", "Z"],
                ),
                "cp737",
            ),
        ],
    )
    def test_to_csv_compression(self, df, encoding, compression):
        with tm.ensure_clean() as filename:
            df.to_csv(filename, compression=compression, encoding=encoding)
            # test the round trip - to_csv -> read_csv
            result = read_csv(
                filename, compression=compression, index_col=0, encoding=encoding
            )
            tm.assert_frame_equal(df, result)

            # test the round trip using file handle - to_csv -> read_csv
            with get_handle(
                filename, "w", compression=compression, encoding=encoding
            ) as handles:
                df.to_csv(handles.handle, encoding=encoding)
                assert not handles.handle.closed

            result = read_csv(
                filename,
                compression=compression,
                encoding=encoding,
                index_col=0,
            ).squeeze("columns")
            tm.assert_frame_equal(df, result)

            # explicitly make sure file is compressed
            with tm.decompress_file(filename, compression) as fh:
                text = fh.read().decode(encoding or "utf8")
                for col in df.columns:
                    assert col in text

            with tm.decompress_file(filename, compression) as fh:
                tm.assert_frame_equal(df, read_csv(fh, index_col=0, encoding=encoding))

    def test_to_csv_date_format(self, datetime_frame):
        with tm.ensure_clean("__tmp_to_csv_date_format__") as path:
            dt_index = datetime_frame.index
            datetime_frame = DataFrame(
                {"A": dt_index, "B": dt_index.shift(1)}, index=dt_index
            )
            datetime_frame.to_csv(path, date_format="%Y%m%d")

            # Check that the data was put in the specified format
            test = read_csv(path, index_col=0)

            datetime_frame_int = datetime_frame.map(lambda x: int(x.strftime("%Y%m%d")))
            datetime_frame_int.index = datetime_frame_int.index.map(
                lambda x: int(x.strftime("%Y%m%d"))
            )

            tm.assert_frame_equal(test, datetime_frame_int)

            datetime_frame.to_csv(path, date_format="%Y-%m-%d")

            # Check that the data was put in the specified format
            test = read_csv(path, index_col=0)
            datetime_frame_str = datetime_frame.map(lambda x: x.strftime("%Y-%m-%d"))
            datetime_frame_str.index = datetime_frame_str.index.map(
                lambda x: x.strftime("%Y-%m-%d")
            )

            tm.assert_frame_equal(test, datetime_frame_str)

            # Check that columns get converted
            datetime_frame_columns = datetime_frame.T
            datetime_frame_columns.to_csv(path, date_format="%Y%m%d")

            test = read_csv(path, index_col=0)

            datetime_frame_columns = datetime_frame_columns.map(
                lambda x: int(x.strftime("%Y%m%d"))
            )
            # Columns don't get converted to ints by read_csv
            datetime_frame_columns.columns = datetime_frame_columns.columns.map(
                lambda x: x.strftime("%Y%m%d")
            )

            tm.assert_frame_equal(test, datetime_frame_columns)

            # test NaTs
            nat_index = to_datetime(
                ["NaT"] * 10 + ["2000-01-01", "2000-01-01", "2000-01-01"]
            )
            nat_frame = DataFrame({"A": nat_index}, index=nat_index)
            nat_frame.to_csv(path, date_format="%Y-%m-%d")

            test = read_csv(path, parse_dates=[0, 1], index_col=0)

            tm.assert_frame_equal(test, nat_frame)

    @pytest.mark.parametrize("td", [pd.Timedelta(0), pd.Timedelta("10s")])
    def test_to_csv_with_dst_transitions(self, td):
        with tm.ensure_clean("csv_date_format_with_dst") as path:
            # make sure we are not failing on transitions
            times = date_range(
                "2013-10-26 23:00",
                "2013-10-27 01:00",
                tz="Europe/London",
                freq="h",
                ambiguous="infer",
            )
            i = times + td
            i = i._with_freq(None)  # freq is not preserved by read_csv
            time_range = np.array(range(len(i)), dtype="int64")
            df = DataFrame({"A": time_range}, index=i)
            df.to_csv(path, index=True)
            # we have to reconvert the index as we
            # don't parse the tz's
            result = read_csv(path, index_col=0)
            result.index = to_datetime(result.index, utc=True).tz_convert(
                "Europe/London"
            )
            tm.assert_frame_equal(result, df)

    def test_to_csv_with_dst_transitions_with_pickle(self):
        # GH11619
        idx = date_range("2015-01-01", "2015-12-31", freq="h", tz="Europe/Paris")
        idx = idx._with_freq(None)  # freq does not round-trip
        idx._data._freq = None  # otherwise there is trouble on unpickle
        df = DataFrame({"values": 1, "idx": idx}, index=idx)
        with tm.ensure_clean("csv_date_format_with_dst") as path:
            df.to_csv(path, index=True)
            result = read_csv(path, index_col=0)
            result.index = to_datetime(result.index, utc=True).tz_convert(
                "Europe/Paris"
            )
            result["idx"] = to_datetime(result["idx"], utc=True).astype(
                "datetime64[ns, Europe/Paris]"
            )
            tm.assert_frame_equal(result, df)

        # assert working
        df.astype(str)

        with tm.ensure_clean("csv_date_format_with_dst") as path:
            df.to_pickle(path)
            result = pd.read_pickle(path)
            tm.assert_frame_equal(result, df)

    def test_to_csv_quoting(self):
        df = DataFrame(
            {
                "c_bool": [True, False],
                "c_float": [1.0, 3.2],
                "c_int": [42, np.nan],
                "c_string": ["a", "b,c"],
            }
        )

        expected_rows = [
            ",c_bool,c_float,c_int,c_string",
            "0,True,1.0,42.0,a",
            '1,False,3.2,,"b,c"',
        ]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)

        result = df.to_csv()
        assert result == expected

        result = df.to_csv(quoting=None)
        assert result == expected

        expected_rows = [
            ",c_bool,c_float,c_int,c_string",
            "0,True,1.0,42.0,a",
            '1,False,3.2,,"b,c"',
        ]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)

        result = df.to_csv(quoting=csv.QUOTE_MINIMAL)
        assert result == expected

        expected_rows = [
            '"","c_bool","c_float","c_int","c_string"',
            '"0","True","1.0","42.0","a"',
            '"1","False","3.2","","b,c"',
        ]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)

        result = df.to_csv(quoting=csv.QUOTE_ALL)
        assert result == expected

        # see gh-12922, gh-13259: make sure changes to
        # the formatters do not break this behaviour
        expected_rows = [
            '"","c_bool","c_float","c_int","c_string"',
            '0,True,1.0,42.0,"a"',
            '1,False,3.2,"","b,c"',
        ]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        result = df.to_csv(quoting=csv.QUOTE_NONNUMERIC)
        assert result == expected

        msg = "need to escape, but no escapechar set"
        with pytest.raises(csv.Error, match=msg):
            df.to_csv(quoting=csv.QUOTE_NONE)

        with pytest.raises(csv.Error, match=msg):
            df.to_csv(quoting=csv.QUOTE_NONE, escapechar=None)

        expected_rows = [
            ",c_bool,c_float,c_int,c_string",
            "0,True,1.0,42.0,a",
            "1,False,3.2,,b!,c",
        ]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        result = df.to_csv(quoting=csv.QUOTE_NONE, escapechar="!")
        assert result == expected

        expected_rows = [
            ",c_bool,c_ffloat,c_int,c_string",
            "0,True,1.0,42.0,a",
            "1,False,3.2,,bf,c",
        ]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        result = df.to_csv(quoting=csv.QUOTE_NONE, escapechar="f")
        assert result == expected

        # see gh-3503: quoting Windows line terminators
        # presents with encoding?
        text_rows = ["a,b,c", '1,"test \r\n",3']
        text = tm.convert_rows_list_to_csv_str(text_rows)
        df = read_csv(StringIO(text))

        buf = StringIO()
        df.to_csv(buf, encoding="utf-8", index=False)
        assert buf.getvalue() == text

        # xref gh-7791: make sure the quoting parameter is passed through
        # with multi-indexes
        df = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        df = df.set_index(["a", "b"])

        expected_rows = ['"a","b","c"', '"1","3","5"', '"2","4","6"']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv(quoting=csv.QUOTE_ALL) == expected

    def test_period_index_date_overflow(self):
        # see gh-15982

        dates = ["1990-01-01", "2000-01-01", "3005-01-01"]
        index = pd.PeriodIndex(dates, freq="D")

        df = DataFrame([4, 5, 6], index=index)
        result = df.to_csv()

        expected_rows = [",0", "1990-01-01,4", "2000-01-01,5", "3005-01-01,6"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

        date_format = "%m-%d-%Y"
        result = df.to_csv(date_format=date_format)

        expected_rows = [",0", "01-01-1990,4", "01-01-2000,5", "01-01-3005,6"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

        # Overflow with pd.NaT
        dates = ["1990-01-01", NaT, "3005-01-01"]
        index = pd.PeriodIndex(dates, freq="D")

        df = DataFrame([4, 5, 6], index=index)
        result = df.to_csv()

        expected_rows = [",0", "1990-01-01,4", ",5", "3005-01-01,6"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

    def test_multi_index_header(self):
        # see gh-5539
        columns = MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)])
        df = DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]])
        df.columns = columns

        header = ["a", "b", "c", "d"]
        result = df.to_csv(header=header)

        expected_rows = [",a,b,c,d", "0,1,2,3,4", "1,5,6,7,8"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

    def test_to_csv_single_level_multi_index(self):
        # see gh-26303
        index = Index([(1,), (2,), (3,)])
        df = DataFrame([[1, 2, 3]], columns=index)
        df = df.reindex(columns=[(1,), (3,)])
        expected = ",1,3\n0,1,3\n"
        result = df.to_csv(lineterminator="\n")
        tm.assert_almost_equal(result, expected)

    def test_gz_lineend(self):
        # GH 25311
        df = DataFrame({"a": [1, 2]})
        expected_rows = ["a", "1", "2"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        with tm.ensure_clean("__test_gz_lineend.csv.gz") as path:
            df.to_csv(path, index=False)
            with tm.decompress_file(path, compression="gzip") as f:
                result = f.read().decode("utf-8")

        assert result == expected

    def test_to_csv_numpy_16_bug(self):
        frame = DataFrame({"a": date_range("1/1/2000", periods=10)})

        buf = StringIO()
        frame.to_csv(buf)

        result = buf.getvalue()
        assert "2000-01-01" in result

    def test_to_csv_na_quoting(self):
        # GH 15891
        # Normalize carriage return for Windows OS
        result = (
            DataFrame([None, None])
            .to_csv(None, header=False, index=False, na_rep="")
            .replace("\r\n", "\n")
        )
        expected = '""\n""\n'
        assert result == expected

    def test_to_csv_categorical_and_ea(self):
        # GH#46812
        df = DataFrame({"a": "x", "b": [1, pd.NA]})
        df["b"] = df["b"].astype("Int16")
        df["b"] = df["b"].astype("category")
        result = df.to_csv()
        expected_rows = [",a,b", "0,x,1", "1,x,"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

    def test_to_csv_categorical_and_interval(self):
        # GH#46297
        df = DataFrame(
            {
                "a": [
                    pd.Interval(
                        Timestamp("2020-01-01"),
                        Timestamp("2020-01-02"),
                        closed="both",
                    )
                ]
            }
        )
        df["a"] = df["a"].astype("category")
        result = df.to_csv()
        expected_rows = [",a", '0,"[2020-01-01 00:00:00, 2020-01-02 00:00:00]"']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\tests\test_domainmatrix.py ===
from sympy.external.gmpy import GROUND_TYPES

from sympy import Integer, Rational, S, sqrt, Matrix, symbols
from sympy import FF, ZZ, QQ, QQ_I, EXRAW

from sympy.polys.matrices.domainmatrix import DomainMatrix, DomainScalar, DM
from sympy.polys.matrices.exceptions import (
    DMBadInputError, DMDomainError, DMShapeError, DMFormatError, DMNotAField,
    DMNonSquareMatrixError, DMNonInvertibleMatrixError,
)
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.sdm import SDM

from sympy.testing.pytest import raises


def test_DM():
    ddm = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A = DM([[1, 2], [3, 4]], ZZ)
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        assert A.rep == ddm.to_dfm()
    assert A.shape == (2, 2)
    assert A.domain == ZZ


def test_DomainMatrix_init():
    lol = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]
    dod = {0: {0: ZZ(1), 1:ZZ(2)}, 1: {0:ZZ(3), 1:ZZ(4)}}
    ddm = DDM(lol, (2, 2), ZZ)
    sdm = SDM(dod, (2, 2), ZZ)

    A = DomainMatrix(lol, (2, 2), ZZ)
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        assert A.rep == ddm.to_dfm()
    assert A.shape == (2, 2)
    assert A.domain == ZZ

    A = DomainMatrix(dod, (2, 2), ZZ)
    assert A.rep == sdm
    assert A.shape == (2, 2)
    assert A.domain == ZZ

    raises(TypeError, lambda: DomainMatrix(ddm, (2, 2), ZZ))
    raises(TypeError, lambda: DomainMatrix(sdm, (2, 2), ZZ))
    raises(TypeError, lambda: DomainMatrix(Matrix([[1]]), (1, 1), ZZ))

    for fmt, rep in [('sparse', sdm), ('dense', ddm)]:
        if fmt == 'dense' and GROUND_TYPES == 'flint':
            rep = rep.to_dfm()
        A = DomainMatrix(lol, (2, 2), ZZ, fmt=fmt)
        assert A.rep == rep
        A = DomainMatrix(dod, (2, 2), ZZ, fmt=fmt)
        assert A.rep == rep

    raises(ValueError, lambda: DomainMatrix(lol, (2, 2), ZZ, fmt='invalid'))

    raises(DMBadInputError, lambda: DomainMatrix([[ZZ(1), ZZ(2)]], (2, 2), ZZ))

    # uses copy
    was = [i.copy() for i in lol]
    A[0,0] = ZZ(42)
    assert was == lol


def test_DomainMatrix_from_rep():
    ddm = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A = DomainMatrix.from_rep(ddm)
    # XXX: Should from_rep convert to DFM?
    assert A.rep == ddm
    assert A.shape == (2, 2)
    assert A.domain == ZZ

    sdm = SDM({0: {0: ZZ(1), 1:ZZ(2)}, 1: {0:ZZ(3), 1:ZZ(4)}}, (2, 2), ZZ)
    A = DomainMatrix.from_rep(sdm)
    assert A.rep == sdm
    assert A.shape == (2, 2)
    assert A.domain == ZZ

    A = DomainMatrix([[ZZ(1)]], (1, 1), ZZ)
    raises(TypeError, lambda: DomainMatrix.from_rep(A))


def test_DomainMatrix_from_list():
    ddm = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A = DomainMatrix.from_list([[1, 2], [3, 4]], ZZ)
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        assert A.rep == ddm.to_dfm()
    assert A.shape == (2, 2)
    assert A.domain == ZZ

    dom = FF(7)
    ddm = DDM([[dom(1), dom(2)], [dom(3), dom(4)]], (2, 2), dom)
    A = DomainMatrix.from_list([[1, 2], [3, 4]], dom)
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        assert A.rep == ddm.to_dfm()
    assert A.shape == (2, 2)
    assert A.domain == dom

    dom = FF(2**127-1)
    ddm = DDM([[dom(1), dom(2)], [dom(3), dom(4)]], (2, 2), dom)
    A = DomainMatrix.from_list([[1, 2], [3, 4]], dom)
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        assert A.rep == ddm.to_dfm()
    assert A.shape == (2, 2)
    assert A.domain == dom

    ddm = DDM([[QQ(1, 2), QQ(3, 1)], [QQ(1, 4), QQ(5, 1)]], (2, 2), QQ)
    A = DomainMatrix.from_list([[(1, 2), (3, 1)], [(1, 4), (5, 1)]], QQ)
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        assert A.rep == ddm.to_dfm()
    assert A.shape == (2, 2)
    assert A.domain == QQ


def test_DomainMatrix_from_list_sympy():
    ddm = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A = DomainMatrix.from_list_sympy(2, 2, [[1, 2], [3, 4]])
    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        assert A.rep == ddm.to_dfm()
    assert A.shape == (2, 2)
    assert A.domain == ZZ

    K = QQ.algebraic_field(sqrt(2))
    ddm = DDM(
        [[K.convert(1 + sqrt(2)), K.convert(2 + sqrt(2))],
         [K.convert(3 + sqrt(2)), K.convert(4 + sqrt(2))]],
        (2, 2),
        K
    )
    A = DomainMatrix.from_list_sympy(
        2, 2, [[1 + sqrt(2), 2 + sqrt(2)], [3 + sqrt(2), 4 + sqrt(2)]],
        extension=True)
    assert A.rep == ddm
    assert A.shape == (2, 2)
    assert A.domain == K


def test_DomainMatrix_from_dict_sympy():
    sdm = SDM({0: {0: QQ(1, 2)}, 1: {1: QQ(2, 3)}}, (2, 2), QQ)
    sympy_dict = {0: {0: Rational(1, 2)}, 1: {1: Rational(2, 3)}}
    A = DomainMatrix.from_dict_sympy(2, 2, sympy_dict)
    assert A.rep == sdm
    assert A.shape == (2, 2)
    assert A.domain == QQ

    fds = DomainMatrix.from_dict_sympy
    raises(DMBadInputError, lambda: fds(2, 2, {3: {0: Rational(1, 2)}}))
    raises(DMBadInputError, lambda: fds(2, 2, {0: {3: Rational(1, 2)}}))


def test_DomainMatrix_from_Matrix():
    sdm = SDM({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}, (2, 2), ZZ)
    A = DomainMatrix.from_Matrix(Matrix([[1, 2], [3, 4]]))
    assert A.rep == sdm
    assert A.shape == (2, 2)
    assert A.domain == ZZ

    K = QQ.algebraic_field(sqrt(2))
    sdm = SDM(
        {0: {0: K.convert(1 + sqrt(2)), 1: K.convert(2 + sqrt(2))},
         1: {0: K.convert(3 + sqrt(2)), 1: K.convert(4 + sqrt(2))}},
        (2, 2),
        K
    )
    A = DomainMatrix.from_Matrix(
        Matrix([[1 + sqrt(2), 2 + sqrt(2)], [3 + sqrt(2), 4 + sqrt(2)]]),
        extension=True)
    assert A.rep == sdm
    assert A.shape == (2, 2)
    assert A.domain == K

    A = DomainMatrix.from_Matrix(Matrix([[QQ(1, 2), QQ(3, 4)], [QQ(0, 1), QQ(0, 1)]]), fmt='dense')
    ddm = DDM([[QQ(1, 2), QQ(3, 4)], [QQ(0, 1), QQ(0, 1)]], (2, 2), QQ)

    if GROUND_TYPES != 'flint':
        assert A.rep == ddm
    else:
        assert A.rep == ddm.to_dfm()
    assert A.shape == (2, 2)
    assert A.domain == QQ


def test_DomainMatrix_eq():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A == A
    B = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(1)]], (2, 2), ZZ)
    assert A != B
    C = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]
    assert A != C


def test_DomainMatrix_unify_eq():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B1 = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    B2 = DomainMatrix([[QQ(1), QQ(3)], [QQ(3), QQ(4)]], (2, 2), QQ)
    B3 = DomainMatrix([[ZZ(1)]], (1, 1), ZZ)
    assert A.unify_eq(B1) is True
    assert A.unify_eq(B2) is False
    assert A.unify_eq(B3) is False


def test_DomainMatrix_get_domain():
    K, items = DomainMatrix.get_domain([1, 2, 3, 4])
    assert items == [ZZ(1), ZZ(2), ZZ(3), ZZ(4)]
    assert K == ZZ

    K, items = DomainMatrix.get_domain([1, 2, 3, Rational(1, 2)])
    assert items == [QQ(1), QQ(2), QQ(3), QQ(1, 2)]
    assert K == QQ


def test_DomainMatrix_convert_to():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aq = A.convert_to(QQ)
    assert Aq == DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)


def test_DomainMatrix_choose_domain():
    A = [[1, 2], [3, 0]]
    assert DM(A, QQ).choose_domain() == DM(A, ZZ)
    assert DM(A, QQ).choose_domain(field=True) == DM(A, QQ)
    assert DM(A, ZZ).choose_domain(field=True) == DM(A, QQ)

    x = symbols('x')
    B = [[1, x], [x**2, x**3]]
    assert DM(B, QQ[x]).choose_domain(field=True) == DM(B, ZZ.frac_field(x))


def test_DomainMatrix_to_flat_nz():
    Adm = DM([[1, 2], [3, 0]], ZZ)
    Addm = Adm.rep.to_ddm()
    Asdm = Adm.rep.to_sdm()
    for A in [Adm, Addm, Asdm]:
        elems, data = A.to_flat_nz()
        assert A.from_flat_nz(elems, data, A.domain) == A
        elemsq = [QQ(e) for e in elems]
        assert A.from_flat_nz(elemsq, data, QQ) == A.convert_to(QQ)
        elems2 = [2*e for e in elems]
        assert A.from_flat_nz(elems2, data, A.domain) == 2*A


def test_DomainMatrix_to_sympy():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.to_sympy() == A.convert_to(EXRAW)


def test_DomainMatrix_to_field():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aq = A.to_field()
    assert Aq == DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)


def test_DomainMatrix_to_sparse():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A_sparse = A.to_sparse()
    assert A_sparse.rep == {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}


def test_DomainMatrix_to_dense():
    A = DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}, (2, 2), ZZ)
    A_dense = A.to_dense()
    ddm = DDM([[1, 2], [3, 4]], (2, 2), ZZ)
    if GROUND_TYPES != 'flint':
        assert A_dense.rep == ddm
    else:
        assert A_dense.rep == ddm.to_dfm()


def test_DomainMatrix_unify():
    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aq = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    assert Az.unify(Az) == (Az, Az)
    assert Az.unify(Aq) == (Aq, Aq)
    assert Aq.unify(Az) == (Aq, Aq)
    assert Aq.unify(Aq) == (Aq, Aq)

    As = DomainMatrix({0: {1: ZZ(1)}, 1:{0:ZZ(2)}}, (2, 2), ZZ)
    Ad = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)

    assert As.unify(As) == (As, As)
    assert Ad.unify(Ad) == (Ad, Ad)

    Bs, Bd = As.unify(Ad, fmt='dense')
    assert Bs.rep == DDM([[0, 1], [2, 0]], (2, 2), ZZ).to_dfm_or_ddm()
    assert Bd.rep == DDM([[1, 2],[3, 4]], (2, 2), ZZ).to_dfm_or_ddm()

    Bs, Bd = As.unify(Ad, fmt='sparse')
    assert Bs.rep == SDM({0: {1: 1}, 1: {0: 2}}, (2, 2), ZZ)
    assert Bd.rep == SDM({0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}, (2, 2), ZZ)

    raises(ValueError, lambda: As.unify(Ad, fmt='invalid'))


def test_DomainMatrix_to_Matrix():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A_Matrix = Matrix([[1, 2], [3, 4]])
    assert A.to_Matrix() == A_Matrix
    assert A.to_sparse().to_Matrix() == A_Matrix
    assert A.convert_to(QQ).to_Matrix() == A_Matrix
    assert A.convert_to(QQ.algebraic_field(sqrt(2))).to_Matrix() == A_Matrix


def test_DomainMatrix_to_list():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.to_list() == [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]


def test_DomainMatrix_to_list_flat():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.to_list_flat() == [ZZ(1), ZZ(2), ZZ(3), ZZ(4)]


def test_DomainMatrix_flat():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.flat() == [ZZ(1), ZZ(2), ZZ(3), ZZ(4)]


def test_DomainMatrix_from_list_flat():
    nums = [ZZ(1), ZZ(2), ZZ(3), ZZ(4)]
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)

    assert DomainMatrix.from_list_flat(nums, (2, 2), ZZ) == A
    assert DDM.from_list_flat(nums, (2, 2), ZZ) == A.rep.to_ddm()
    assert SDM.from_list_flat(nums, (2, 2), ZZ) == A.rep.to_sdm()

    assert A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)

    raises(DMBadInputError, DomainMatrix.from_list_flat, nums, (2, 3), ZZ)
    raises(DMBadInputError, DDM.from_list_flat, nums, (2, 3), ZZ)
    raises(DMBadInputError, SDM.from_list_flat, nums, (2, 3), ZZ)


def test_DomainMatrix_to_dod():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.to_dod() == {0: {0: ZZ(1), 1:ZZ(2)}, 1: {0:ZZ(3), 1:ZZ(4)}}
    A = DomainMatrix([[ZZ(1), ZZ(0)], [ZZ(0), ZZ(4)]], (2, 2), ZZ)
    assert A.to_dod() == {0: {0: ZZ(1)}, 1: {1: ZZ(4)}}


def test_DomainMatrix_from_dod():
    items = {0: {0: ZZ(1), 1:ZZ(2)}, 1: {0:ZZ(3), 1:ZZ(4)}}
    A = DM([[1, 2], [3, 4]], ZZ)
    assert DomainMatrix.from_dod(items, (2, 2), ZZ) == A.to_sparse()
    assert A.from_dod_like(items) == A
    assert A.from_dod_like(items, QQ) == A.convert_to(QQ)


def test_DomainMatrix_to_dok():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.to_dok() == {(0, 0):ZZ(1), (0, 1):ZZ(2), (1, 0):ZZ(3), (1, 1):ZZ(4)}
    A = DomainMatrix([[ZZ(1), ZZ(0)], [ZZ(0), ZZ(4)]], (2, 2), ZZ)
    dok = {(0, 0):ZZ(1), (1, 1):ZZ(4)}
    assert A.to_dok() == dok
    assert A.to_dense().to_dok() == dok
    assert A.to_sparse().to_dok() == dok
    assert A.rep.to_ddm().to_dok() == dok
    assert A.rep.to_sdm().to_dok() == dok


def test_DomainMatrix_from_dok():
    items = {(0, 0): ZZ(1), (1, 1): ZZ(2)}
    A = DM([[1, 0], [0, 2]], ZZ)
    assert DomainMatrix.from_dok(items, (2, 2), ZZ) == A.to_sparse()
    assert DDM.from_dok(items, (2, 2), ZZ) == A.rep.to_ddm()
    assert SDM.from_dok(items, (2, 2), ZZ) == A.rep.to_sdm()


def test_DomainMatrix_repr():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert repr(A) == 'DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)'


def test_DomainMatrix_transpose():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    AT = DomainMatrix([[ZZ(1), ZZ(3)], [ZZ(2), ZZ(4)]], (2, 2), ZZ)
    assert A.transpose() == AT


def test_DomainMatrix_is_zero_matrix():
    A = DomainMatrix([[ZZ(1)]], (1, 1), ZZ)
    B = DomainMatrix([[ZZ(0)]], (1, 1), ZZ)
    assert A.is_zero_matrix is False
    assert B.is_zero_matrix is True


def test_DomainMatrix_is_upper():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(0), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.is_upper is True
    assert B.is_upper is False


def test_DomainMatrix_is_lower():
    A = DomainMatrix([[ZZ(1), ZZ(0)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.is_lower is True
    assert B.is_lower is False


def test_DomainMatrix_is_diagonal():
    A = DM([[1, 0], [0, 4]], ZZ)
    B = DM([[1, 2], [3, 4]], ZZ)
    assert A.is_diagonal is A.to_sparse().is_diagonal is True
    assert B.is_diagonal is B.to_sparse().is_diagonal is False


def test_DomainMatrix_is_square():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)], [ZZ(5), ZZ(6)]], (3, 2), ZZ)
    assert A.is_square is True
    assert B.is_square is False


def test_DomainMatrix_diagonal():
    A = DM([[1, 2], [3, 4]], ZZ)
    assert A.diagonal() == A.to_sparse().diagonal() == [ZZ(1), ZZ(4)]
    A = DM([[1, 2], [3, 4], [5, 6]], ZZ)
    assert A.diagonal() == A.to_sparse().diagonal() == [ZZ(1), ZZ(4)]
    A = DM([[1, 2, 3], [4, 5, 6]], ZZ)
    assert A.diagonal() == A.to_sparse().diagonal() == [ZZ(1), ZZ(5)]


def test_DomainMatrix_rank():
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(6), QQ(8)]], (3, 2), QQ)
    assert A.rank() == 2


def test_DomainMatrix_add():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(2), ZZ(4)], [ZZ(6), ZZ(8)]], (2, 2), ZZ)
    assert A + A == A.add(A) == B

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    L = [[2, 3], [3, 4]]
    raises(TypeError, lambda: A + L)
    raises(TypeError, lambda: L + A)

    A1 = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A2 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMShapeError, lambda: A1 + A2)
    raises(DMShapeError, lambda: A2 + A1)
    raises(DMShapeError, lambda: A1.add(A2))
    raises(DMShapeError, lambda: A2.add(A1))

    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aq = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    Asum = DomainMatrix([[QQ(2), QQ(4)], [QQ(6), QQ(8)]], (2, 2), QQ)
    assert Az + Aq == Asum
    assert Aq + Az == Asum
    raises(DMDomainError, lambda: Az.add(Aq))
    raises(DMDomainError, lambda: Aq.add(Az))

    As = DomainMatrix({0: {1: ZZ(1)}, 1: {0: ZZ(2)}}, (2, 2), ZZ)
    Ad = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)

    Asd = As + Ad
    Ads = Ad + As
    assert Asd == DomainMatrix([[1, 3], [5, 4]], (2, 2), ZZ)
    assert Asd.rep == DDM([[1, 3], [5, 4]], (2, 2), ZZ).to_dfm_or_ddm()
    assert Ads == DomainMatrix([[1, 3], [5, 4]], (2, 2), ZZ)
    assert Ads.rep == DDM([[1, 3], [5, 4]], (2, 2), ZZ).to_dfm_or_ddm()
    raises(DMFormatError, lambda: As.add(Ad))


def test_DomainMatrix_sub():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(0), ZZ(0)], [ZZ(0), ZZ(0)]], (2, 2), ZZ)
    assert A - A == A.sub(A) == B

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    L = [[2, 3], [3, 4]]
    raises(TypeError, lambda: A - L)
    raises(TypeError, lambda: L - A)

    A1 = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A2 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMShapeError, lambda: A1 - A2)
    raises(DMShapeError, lambda: A2 - A1)
    raises(DMShapeError, lambda: A1.sub(A2))
    raises(DMShapeError, lambda: A2.sub(A1))

    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aq = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    Adiff = DomainMatrix([[QQ(0), QQ(0)], [QQ(0), QQ(0)]], (2, 2), QQ)
    assert Az - Aq == Adiff
    assert Aq - Az == Adiff
    raises(DMDomainError, lambda: Az.sub(Aq))
    raises(DMDomainError, lambda: Aq.sub(Az))

    As = DomainMatrix({0: {1: ZZ(1)}, 1: {0: ZZ(2)}}, (2, 2), ZZ)
    Ad = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)

    Asd = As - Ad
    Ads = Ad - As
    assert Asd == DomainMatrix([[-1, -1], [-1, -4]], (2, 2), ZZ)
    assert Asd.rep == DDM([[-1, -1], [-1, -4]], (2, 2), ZZ).to_dfm_or_ddm()
    assert Asd == -Ads
    assert Asd.rep == -Ads.rep


def test_DomainMatrix_neg():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aneg = DomainMatrix([[ZZ(-1), ZZ(-2)], [ZZ(-3), ZZ(-4)]], (2, 2), ZZ)
    assert -A == A.neg() == Aneg


def test_DomainMatrix_mul():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A2 = DomainMatrix([[ZZ(7), ZZ(10)], [ZZ(15), ZZ(22)]], (2, 2), ZZ)
    assert A*A == A.matmul(A) == A2

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    L = [[1, 2], [3, 4]]
    raises(TypeError, lambda: A * L)
    raises(TypeError, lambda: L * A)

    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aq = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    Aprod = DomainMatrix([[QQ(7), QQ(10)], [QQ(15), QQ(22)]], (2, 2), QQ)
    assert Az * Aq == Aprod
    assert Aq * Az == Aprod
    raises(DMDomainError, lambda: Az.matmul(Aq))
    raises(DMDomainError, lambda: Aq.matmul(Az))

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    AA = DomainMatrix([[ZZ(2), ZZ(4)], [ZZ(6), ZZ(8)]], (2, 2), ZZ)
    x = ZZ(2)
    assert A * x == x * A == A.mul(x) == AA

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    AA = DomainMatrix.zeros((2, 2), ZZ)
    x = ZZ(0)
    assert A * x == x * A == A.mul(x).to_sparse() == AA

    As = DomainMatrix({0: {1: ZZ(1)}, 1: {0: ZZ(2)}}, (2, 2), ZZ)
    Ad = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)

    Asd = As * Ad
    Ads = Ad * As
    assert Asd == DomainMatrix([[3, 4], [2, 4]], (2, 2), ZZ)
    assert Asd.rep == DDM([[3, 4], [2, 4]], (2, 2), ZZ).to_dfm_or_ddm()
    assert Ads == DomainMatrix([[4, 1], [8, 3]], (2, 2), ZZ)
    assert Ads.rep == DDM([[4, 1], [8, 3]], (2, 2), ZZ).to_dfm_or_ddm()


def test_DomainMatrix_mul_elementwise():
    A = DomainMatrix([[ZZ(2), ZZ(2)], [ZZ(0), ZZ(0)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(4), ZZ(0)], [ZZ(3), ZZ(0)]], (2, 2), ZZ)
    C = DomainMatrix([[ZZ(8), ZZ(0)], [ZZ(0), ZZ(0)]], (2, 2), ZZ)
    assert A.mul_elementwise(B) == C
    assert B.mul_elementwise(A) == C


def test_DomainMatrix_pow():
    eye = DomainMatrix.eye(2, ZZ)
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A2 = DomainMatrix([[ZZ(7), ZZ(10)], [ZZ(15), ZZ(22)]], (2, 2), ZZ)
    A3 = DomainMatrix([[ZZ(37), ZZ(54)], [ZZ(81), ZZ(118)]], (2, 2), ZZ)
    assert A**0 == A.pow(0) == eye
    assert A**1 == A.pow(1) == A
    assert A**2 == A.pow(2) == A2
    assert A**3 == A.pow(3) == A3

    raises(TypeError, lambda: A ** Rational(1, 2))
    raises(NotImplementedError, lambda: A ** -1)
    raises(NotImplementedError, lambda: A.pow(-1))

    A = DomainMatrix.zeros((2, 1), ZZ)
    raises(DMNonSquareMatrixError, lambda: A ** 1)


def test_DomainMatrix_clear_denoms():
    A = DM([[(1,2),(1,3)],[(1,4),(1,5)]], QQ)

    den_Z = DomainScalar(ZZ(60), ZZ)
    Anum_Z = DM([[30, 20], [15, 12]], ZZ)
    Anum_Q = Anum_Z.convert_to(QQ)

    assert A.clear_denoms() == (den_Z, Anum_Q)
    assert A.clear_denoms(convert=True) == (den_Z, Anum_Z)
    assert A * den_Z == Anum_Q
    assert A == Anum_Q / den_Z


def test_DomainMatrix_clear_denoms_rowwise():
    A = DM([[(1,2),(1,3)],[(1,4),(1,5)]], QQ)

    den_Z = DM([[6, 0], [0, 20]], ZZ).to_sparse()
    Anum_Z = DM([[3, 2], [5, 4]], ZZ)
    Anum_Q = DM([[3, 2], [5, 4]], QQ)

    assert A.clear_denoms_rowwise() == (den_Z, Anum_Q)
    assert A.clear_denoms_rowwise(convert=True) == (den_Z, Anum_Z)
    assert den_Z * A == Anum_Q
    assert A == den_Z.to_field().inv() * Anum_Q

    A = DM([[(1,2),(1,3),0,0],[0,0,0,0], [(1,4),(1,5),(1,6),(1,7)]], QQ)
    den_Z = DM([[6, 0, 0], [0, 1, 0], [0, 0, 420]], ZZ).to_sparse()
    Anum_Z = DM([[3, 2, 0, 0], [0, 0, 0, 0], [105, 84, 70, 60]], ZZ)
    Anum_Q = Anum_Z.convert_to(QQ)

    assert A.clear_denoms_rowwise() == (den_Z, Anum_Q)
    assert A.clear_denoms_rowwise(convert=True) == (den_Z, Anum_Z)
    assert den_Z * A == Anum_Q
    assert A == den_Z.to_field().inv() * Anum_Q


def test_DomainMatrix_cancel_denom():
    A = DM([[2, 4], [6, 8]], ZZ)
    assert A.cancel_denom(ZZ(1)) == (DM([[2, 4], [6, 8]], ZZ), ZZ(1))
    assert A.cancel_denom(ZZ(3)) == (DM([[2, 4], [6, 8]], ZZ), ZZ(3))
    assert A.cancel_denom(ZZ(4)) == (DM([[1, 2], [3, 4]], ZZ), ZZ(2))

    A = DM([[1, 2], [3, 4]], ZZ)
    assert A.cancel_denom(ZZ(2)) == (A, ZZ(2))
    assert A.cancel_denom(ZZ(-2)) == (-A, ZZ(2))

    # Test canonicalization of denominator over Gaussian rationals.
    A = DM([[1, 2], [3, 4]], QQ_I)
    assert A.cancel_denom(QQ_I(0,2)) == (QQ_I(0,-1)*A, QQ_I(2))

    raises(ZeroDivisionError, lambda: A.cancel_denom(ZZ(0)))


def test_DomainMatrix_cancel_denom_elementwise():
    A = DM([[2, 4], [6, 8]], ZZ)
    numers, denoms = A.cancel_denom_elementwise(ZZ(1))
    assert numers == DM([[2, 4], [6, 8]], ZZ)
    assert denoms == DM([[1, 1], [1, 1]], ZZ)
    numers, denoms = A.cancel_denom_elementwise(ZZ(4))
    assert numers == DM([[1, 1], [3, 2]], ZZ)
    assert denoms == DM([[2, 1], [2, 1]], ZZ)

    raises(ZeroDivisionError, lambda: A.cancel_denom_elementwise(ZZ(0)))


def test_DomainMatrix_content_primitive():
    A = DM([[2, 4], [6, 8]], ZZ)
    A_primitive = DM([[1, 2], [3, 4]], ZZ)
    A_content = ZZ(2)
    assert A.content() == A_content
    assert A.primitive() == (A_content, A_primitive)


def test_DomainMatrix_scc():
    Ad = DomainMatrix([[ZZ(1), ZZ(2), ZZ(3)],
                       [ZZ(0), ZZ(1), ZZ(0)],
                       [ZZ(2), ZZ(0), ZZ(4)]], (3, 3), ZZ)
    As = Ad.to_sparse()
    Addm = Ad.rep
    Asdm = As.rep
    for A in [Ad, As, Addm, Asdm]:
        assert Ad.scc() == [[1], [0, 2]]

    A = DM([[ZZ(1), ZZ(2), ZZ(3)]], ZZ)
    raises(DMNonSquareMatrixError, lambda: A.scc())


def test_DomainMatrix_rref():
    # More tests in test_rref.py
    A = DomainMatrix([], (0, 1), QQ)
    assert A.rref() == (A, ())

    A = DomainMatrix([[QQ(1)]], (1, 1), QQ)
    assert A.rref() == (A, (0,))

    A = DomainMatrix([[QQ(0)]], (1, 1), QQ)
    assert A.rref() == (A, ())

    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    Ar, pivots = A.rref()
    assert Ar == DomainMatrix([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
    assert pivots == (0, 1)

    A = DomainMatrix([[QQ(0), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    Ar, pivots = A.rref()
    assert Ar == DomainMatrix([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
    assert pivots == (0, 1)

    A = DomainMatrix([[QQ(0), QQ(2)], [QQ(0), QQ(4)]], (2, 2), QQ)
    Ar, pivots = A.rref()
    assert Ar == DomainMatrix([[QQ(0), QQ(1)], [QQ(0), QQ(0)]], (2, 2), QQ)
    assert pivots == (1,)

    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Ar, pivots = Az.rref()
    assert Ar == DomainMatrix([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
    assert pivots == (0, 1)

    methods = ('auto', 'GJ', 'FF', 'CD', 'GJ_dense', 'FF_dense', 'CD_dense')
    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    for method in methods:
        Ar, pivots = Az.rref(method=method)
        assert Ar == DomainMatrix([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
        assert pivots == (0, 1)

    raises(ValueError, lambda: Az.rref(method='foo'))
    raises(ValueError, lambda: Az.rref_den(method='foo'))


def test_DomainMatrix_columnspace():
    A = DomainMatrix([[QQ(1), QQ(-1), QQ(1)], [QQ(2), QQ(-2), QQ(3)]], (2, 3), QQ)
    Acol = DomainMatrix([[QQ(1), QQ(1)], [QQ(2), QQ(3)]], (2, 2), QQ)
    assert A.columnspace() == Acol

    Az = DomainMatrix([[ZZ(1), ZZ(-1), ZZ(1)], [ZZ(2), ZZ(-2), ZZ(3)]], (2, 3), ZZ)
    raises(DMNotAField, lambda: Az.columnspace())

    A = DomainMatrix([[QQ(1), QQ(-1), QQ(1)], [QQ(2), QQ(-2), QQ(3)]], (2, 3), QQ, fmt='sparse')
    Acol = DomainMatrix({0: {0: QQ(1), 1: QQ(1)}, 1: {0: QQ(2), 1: QQ(3)}}, (2, 2), QQ)
    assert A.columnspace() == Acol


def test_DomainMatrix_rowspace():
    A = DomainMatrix([[QQ(1), QQ(-1), QQ(1)], [QQ(2), QQ(-2), QQ(3)]], (2, 3), QQ)
    assert A.rowspace() == A

    Az = DomainMatrix([[ZZ(1), ZZ(-1), ZZ(1)], [ZZ(2), ZZ(-2), ZZ(3)]], (2, 3), ZZ)
    raises(DMNotAField, lambda: Az.rowspace())

    A = DomainMatrix([[QQ(1), QQ(-1), QQ(1)], [QQ(2), QQ(-2), QQ(3)]], (2, 3), QQ, fmt='sparse')
    assert A.rowspace() == A


def test_DomainMatrix_nullspace():
    A = DomainMatrix([[QQ(1), QQ(1)], [QQ(1), QQ(1)]], (2, 2), QQ)
    Anull = DomainMatrix([[QQ(-1), QQ(1)]], (1, 2), QQ)
    assert A.nullspace() == Anull

    A = DomainMatrix([[ZZ(1), ZZ(1)], [ZZ(1), ZZ(1)]], (2, 2), ZZ)
    Anull = DomainMatrix([[ZZ(-1), ZZ(1)]], (1, 2), ZZ)
    assert A.nullspace() == Anull

    raises(DMNotAField, lambda: A.nullspace(divide_last=True))

    A = DomainMatrix([[ZZ(2), ZZ(2)], [ZZ(2), ZZ(2)]], (2, 2), ZZ)
    Anull = DomainMatrix([[ZZ(-2), ZZ(2)]], (1, 2), ZZ)

    Arref, den, pivots = A.rref_den()
    assert den == ZZ(2)
    assert Arref.nullspace_from_rref() == Anull
    assert Arref.nullspace_from_rref(pivots) == Anull
    assert Arref.to_sparse().nullspace_from_rref() == Anull.to_sparse()
    assert Arref.to_sparse().nullspace_from_rref(pivots) == Anull.to_sparse()


def test_DomainMatrix_solve():
    # XXX: Maybe the _solve method should be changed...
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(2), QQ(4)]], (2, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    particular = DomainMatrix([[1, 0]], (1, 2), QQ)
    nullspace = DomainMatrix([[-2, 1]], (1, 2), QQ)
    assert A._solve(b) == (particular, nullspace)

    b3 = DomainMatrix([[QQ(1)], [QQ(1)], [QQ(1)]], (3, 1), QQ)
    raises(DMShapeError, lambda: A._solve(b3))

    bz = DomainMatrix([[ZZ(1)], [ZZ(1)]], (2, 1), ZZ)
    raises(DMNotAField, lambda: A._solve(bz))


def test_DomainMatrix_inv():
    A = DomainMatrix([], (0, 0), QQ)
    assert A.inv() == A

    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    Ainv = DomainMatrix([[QQ(-2), QQ(1)], [QQ(3, 2), QQ(-1, 2)]], (2, 2), QQ)
    assert A.inv() == Ainv

    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    raises(DMNotAField, lambda: Az.inv())

    Ans = DomainMatrix([[QQ(1), QQ(2)]], (1, 2), QQ)
    raises(DMNonSquareMatrixError, lambda: Ans.inv())

    Aninv = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(6)]], (2, 2), QQ)
    raises(DMNonInvertibleMatrixError, lambda: Aninv.inv())

    Z3 = FF(3)
    assert DM([[1, 2], [3, 4]], Z3).inv() == DM([[1, 1], [0, 1]], Z3)

    Z6 = FF(6)
    raises(DMNotAField, lambda: DM([[1, 2], [3, 4]], Z6).inv())


def test_DomainMatrix_det():
    A = DomainMatrix([], (0, 0), ZZ)
    assert A.det() == 1

    A = DomainMatrix([[1]], (1, 1), ZZ)
    assert A.det() == 1

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.det() == ZZ(-2)

    A = DomainMatrix([[ZZ(1), ZZ(2), ZZ(3)], [ZZ(1), ZZ(2), ZZ(4)], [ZZ(1), ZZ(3), ZZ(5)]], (3, 3), ZZ)
    assert A.det() == ZZ(-1)

    A = DomainMatrix([[ZZ(1), ZZ(2), ZZ(3)], [ZZ(1), ZZ(2), ZZ(4)], [ZZ(1), ZZ(2), ZZ(5)]], (3, 3), ZZ)
    assert A.det() == ZZ(0)

    Ans = DomainMatrix([[QQ(1), QQ(2)]], (1, 2), QQ)
    raises(DMNonSquareMatrixError, lambda: Ans.det())

    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    assert A.det() == QQ(-2)


def test_DomainMatrix_eval_poly():
    dM = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    p = [ZZ(1), ZZ(2), ZZ(3)]
    result = DomainMatrix([[ZZ(12), ZZ(14)], [ZZ(21), ZZ(33)]], (2, 2), ZZ)
    assert dM.eval_poly(p) == result == p[0]*dM**2 + p[1]*dM + p[2]*dM**0
    assert dM.eval_poly([]) == dM.zeros(dM.shape, dM.domain)
    assert dM.eval_poly([ZZ(2)]) == 2*dM.eye(2, dM.domain)

    dM2 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMNonSquareMatrixError, lambda: dM2.eval_poly([ZZ(1)]))


def test_DomainMatrix_eval_poly_mul():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    p = [ZZ(1), ZZ(2), ZZ(3)]
    result = DomainMatrix([[ZZ(40)], [ZZ(87)]], (2, 1), ZZ)
    assert A.eval_poly_mul(p, b) == result == p[0]*A**2*b + p[1]*A*b + p[2]*b

    dM = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    dM1 = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    raises(DMNonSquareMatrixError, lambda: dM1.eval_poly_mul([ZZ(1)], b))
    b1 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMShapeError, lambda: dM.eval_poly_mul([ZZ(1)], b1))
    bq = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    raises(DMDomainError, lambda: dM.eval_poly_mul([ZZ(1)], bq))


def _check_solve_den(A, b, xnum, xden):
    # Examples for solve_den, solve_den_charpoly, solve_den_rref should use
    # this so that all methods and types are tested.

    case1 = (A, xnum, b)
    case2 = (A.to_sparse(), xnum.to_sparse(), b.to_sparse())

    for Ai, xnum_i, b_i in [case1, case2]:
        # The key invariant for solve_den:
        assert Ai*xnum_i == xden*b_i

        # solve_den_rref can differ at least by a minus sign
        answers = [(xnum_i, xden), (-xnum_i, -xden)]
        assert Ai.solve_den(b) in answers
        assert Ai.solve_den(b, method='rref') in answers
        assert Ai.solve_den_rref(b) in answers

        # charpoly can only be used if A is square and guarantees to return the
        # actual determinant as a denominator.
        m, n = Ai.shape
        if m == n:
            assert Ai.solve_den(b_i, method='charpoly') == (xnum_i, xden)
            assert Ai.solve_den_charpoly(b_i) == (xnum_i, xden)
        else:
            raises(DMNonSquareMatrixError, lambda: Ai.solve_den_charpoly(b))
            raises(DMNonSquareMatrixError, lambda: Ai.solve_den(b, method='charpoly'))


def test_DomainMatrix_solve_den():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    result = DomainMatrix([[ZZ(0)], [ZZ(-1)]], (2, 1), ZZ)
    den = ZZ(-2)
    _check_solve_den(A, b, result, den)

    A = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(3)],
        [ZZ(1), ZZ(2), ZZ(4)],
        [ZZ(1), ZZ(3), ZZ(5)]], (3, 3), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(2)], [ZZ(3)]], (3, 1), ZZ)
    result = DomainMatrix([[ZZ(2)], [ZZ(0)], [ZZ(-1)]], (3, 1), ZZ)
    den = ZZ(-1)
    _check_solve_den(A, b, result, den)

    A = DomainMatrix([[ZZ(2)], [ZZ(2)]], (2, 1), ZZ)
    b = DomainMatrix([[ZZ(3)], [ZZ(3)]], (2, 1), ZZ)
    result = DomainMatrix([[ZZ(3)]], (1, 1), ZZ)
    den = ZZ(2)
    _check_solve_den(A, b, result, den)


def test_DomainMatrix_solve_den_charpoly():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    A1 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMNonSquareMatrixError, lambda: A1.solve_den_charpoly(b))
    b1 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMShapeError, lambda: A.solve_den_charpoly(b1))
    bq = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    raises(DMDomainError, lambda: A.solve_den_charpoly(bq))


def test_DomainMatrix_solve_den_charpoly_check():
    # Test check
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(2), ZZ(4)]], (2, 2), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(3)]], (2, 1), ZZ)
    raises(DMNonInvertibleMatrixError, lambda: A.solve_den_charpoly(b))
    adjAb = DomainMatrix([[ZZ(-2)], [ZZ(1)]], (2, 1), ZZ)
    assert A.adjugate() * b == adjAb
    assert A.solve_den_charpoly(b, check=False) == (adjAb, ZZ(0))


def test_DomainMatrix_solve_den_errors():
    A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    raises(DMShapeError, lambda: A.solve_den(b))
    raises(DMShapeError, lambda: A.solve_den_rref(b))

    A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    b = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMShapeError, lambda: A.solve_den(b))
    raises(DMShapeError, lambda: A.solve_den_rref(b))

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    b1 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMShapeError, lambda: A.solve_den(b1))

    A = DomainMatrix([[ZZ(2)]], (1, 1), ZZ)
    b = DomainMatrix([[ZZ(2)]], (1, 1), ZZ)
    raises(DMBadInputError, lambda: A.solve_den(b1, method='invalid'))

    A = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    raises(DMNonSquareMatrixError, lambda: A.solve_den_charpoly(b))


def test_DomainMatrix_solve_den_rref_underdetermined():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(1), ZZ(2)]], (2, 2), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(1)]], (2, 1), ZZ)
    raises(DMNonInvertibleMatrixError, lambda: A.solve_den(b))
    raises(DMNonInvertibleMatrixError, lambda: A.solve_den_rref(b))


def test_DomainMatrix_adj_poly_det():
    A = DM([[ZZ(1), ZZ(2), ZZ(3)],
            [ZZ(4), ZZ(5), ZZ(6)],
            [ZZ(7), ZZ(8), ZZ(9)]], ZZ)
    p, detA = A.adj_poly_det()
    assert p == [ZZ(1), ZZ(-15), ZZ(-18)]
    assert A.adjugate() == p[0]*A**2 + p[1]*A**1 + p[2]*A**0 == A.eval_poly(p)
    assert A.det() == detA

    A = DM([[ZZ(1), ZZ(2), ZZ(3)],
            [ZZ(7), ZZ(8), ZZ(9)]], ZZ)
    raises(DMNonSquareMatrixError, lambda: A.adj_poly_det())


def test_DomainMatrix_inv_den():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    den = ZZ(-2)
    result = DomainMatrix([[ZZ(4), ZZ(-2)], [ZZ(-3), ZZ(1)]], (2, 2), ZZ)
    assert A.inv_den() == (result, den)


def test_DomainMatrix_adjugate():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    result = DomainMatrix([[ZZ(4), ZZ(-2)], [ZZ(-3), ZZ(1)]], (2, 2), ZZ)
    assert A.adjugate() == result


def test_DomainMatrix_adj_det():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    adjA = DomainMatrix([[ZZ(4), ZZ(-2)], [ZZ(-3), ZZ(1)]], (2, 2), ZZ)
    assert A.adj_det() == (adjA, ZZ(-2))


def test_DomainMatrix_lu():
    A = DomainMatrix([], (0, 0), QQ)
    assert A.lu() == (A, A, [])

    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    L = DomainMatrix([[QQ(1), QQ(0)], [QQ(3), QQ(1)]], (2, 2), QQ)
    U = DomainMatrix([[QQ(1), QQ(2)], [QQ(0), QQ(-2)]], (2, 2), QQ)
    swaps = []
    assert A.lu() == (L, U, swaps)

    A = DomainMatrix([[QQ(0), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    L = DomainMatrix([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
    U = DomainMatrix([[QQ(3), QQ(4)], [QQ(0), QQ(2)]], (2, 2), QQ)
    swaps = [(0, 1)]
    assert A.lu() == (L, U, swaps)

    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(2), QQ(4)]], (2, 2), QQ)
    L = DomainMatrix([[QQ(1), QQ(0)], [QQ(2), QQ(1)]], (2, 2), QQ)
    U = DomainMatrix([[QQ(1), QQ(2)], [QQ(0), QQ(0)]], (2, 2), QQ)
    swaps = []
    assert A.lu() == (L, U, swaps)

    A = DomainMatrix([[QQ(0), QQ(2)], [QQ(0), QQ(4)]], (2, 2), QQ)
    L = DomainMatrix([[QQ(1), QQ(0)], [QQ(0), QQ(1)]], (2, 2), QQ)
    U = DomainMatrix([[QQ(0), QQ(2)], [QQ(0), QQ(4)]], (2, 2), QQ)
    swaps = []
    assert A.lu() == (L, U, swaps)

    A = DomainMatrix([[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(5), QQ(6)]], (2, 3), QQ)
    L = DomainMatrix([[QQ(1), QQ(0)], [QQ(4), QQ(1)]], (2, 2), QQ)
    U = DomainMatrix([[QQ(1), QQ(2), QQ(3)], [QQ(0), QQ(-3), QQ(-6)]], (2, 3), QQ)
    swaps = []
    assert A.lu() == (L, U, swaps)

    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(5), QQ(6)]], (3, 2), QQ)
    L = DomainMatrix([
        [QQ(1), QQ(0), QQ(0)],
        [QQ(3), QQ(1), QQ(0)],
        [QQ(5), QQ(2), QQ(1)]], (3, 3), QQ)
    U = DomainMatrix([[QQ(1), QQ(2)], [QQ(0), QQ(-2)], [QQ(0), QQ(0)]], (3, 2), QQ)
    swaps = []
    assert A.lu() == (L, U, swaps)

    A = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 2]]
    L = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]
    U = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
    to_dom = lambda rows, dom: [[dom(e) for e in row] for row in rows]
    A = DomainMatrix(to_dom(A, QQ), (4, 4), QQ)
    L = DomainMatrix(to_dom(L, QQ), (4, 4), QQ)
    U = DomainMatrix(to_dom(U, QQ), (4, 4), QQ)
    assert A.lu() == (L, U, [])

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    raises(DMNotAField, lambda: A.lu())


def test_DomainMatrix_lu_solve():
    # Base case
    A = b = x = DomainMatrix([], (0, 0), QQ)
    assert A.lu_solve(b) == x

    # Basic example
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    x = DomainMatrix([[QQ(0)], [QQ(1, 2)]], (2, 1), QQ)
    assert A.lu_solve(b) == x

    # Example with swaps
    A = DomainMatrix([[QQ(0), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    x = DomainMatrix([[QQ(0)], [QQ(1, 2)]], (2, 1), QQ)
    assert A.lu_solve(b) == x

    # Non-invertible
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(2), QQ(4)]], (2, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    raises(DMNonInvertibleMatrixError, lambda: A.lu_solve(b))

    # Overdetermined, consistent
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(5), QQ(6)]], (3, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)], [QQ(3)]], (3, 1), QQ)
    x = DomainMatrix([[QQ(0)], [QQ(1, 2)]], (2, 1), QQ)
    assert A.lu_solve(b) == x

    # Overdetermined, inconsistent
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(5), QQ(6)]], (3, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)], [QQ(4)]], (3, 1), QQ)
    raises(DMNonInvertibleMatrixError, lambda: A.lu_solve(b))

    # Underdetermined
    A = DomainMatrix([[QQ(1), QQ(2)]], (1, 2), QQ)
    b = DomainMatrix([[QQ(1)]], (1, 1), QQ)
    raises(NotImplementedError, lambda: A.lu_solve(b))

    # Non-field
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    b = DomainMatrix([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    raises(DMNotAField, lambda: A.lu_solve(b))

    # Shape mismatch
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    b = DomainMatrix([[QQ(1), QQ(2)]], (1, 2), QQ)
    raises(DMShapeError, lambda: A.lu_solve(b))


def test_DomainMatrix_charpoly():
    A = DomainMatrix([], (0, 0), ZZ)
    p = [ZZ(1)]
    assert A.charpoly() == p
    assert A.to_sparse().charpoly() == p

    A = DomainMatrix([[1]], (1, 1), ZZ)
    p = [ZZ(1), ZZ(-1)]
    assert A.charpoly() == p
    assert A.to_sparse().charpoly() == p

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    p = [ZZ(1), ZZ(-5), ZZ(-2)]
    assert A.charpoly() == p
    assert A.to_sparse().charpoly() == p

    A = DomainMatrix([[ZZ(1), ZZ(2), ZZ(3)], [ZZ(4), ZZ(5), ZZ(6)], [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)
    p = [ZZ(1), ZZ(-15), ZZ(-18), ZZ(0)]
    assert A.charpoly() == p
    assert A.to_sparse().charpoly() == p

    A = DomainMatrix([[ZZ(0), ZZ(1), ZZ(0)],
                      [ZZ(1), ZZ(0), ZZ(1)],
                      [ZZ(0), ZZ(1), ZZ(0)]], (3, 3), ZZ)
    p = [ZZ(1), ZZ(0), ZZ(-2), ZZ(0)]
    assert A.charpoly() == p
    assert A.to_sparse().charpoly() == p

    A = DM([[17, 0, 30,  0,  0,  0, 0,  0, 0, 0],
            [ 0, 0,  0,  0,  0,  0, 0,  0, 0, 0],
            [69, 0,  0,  0,  0, 86, 0,  0, 0, 0],
            [23, 0,  0,  0,  0,  0, 0,  0, 0, 0],
            [ 0, 0,  0,  0,  0,  0, 0,  0, 0, 0],
            [ 0, 0,  0, 13,  0,  0, 0,  0, 0, 0],
            [ 0, 0,  0,  0,  0,  0, 0, 32, 0, 0],
            [ 0, 0,  0,  0, 37, 67, 0,  0, 0, 0],
            [ 0, 0,  0,  0,  0,  0, 0,  0, 0, 0],
            [ 0, 0,  0,  0,  0,  0, 0,  0, 0, 0]], ZZ)
    p = ZZ.map([1, -17, -2070, 0, -771420, 0, 0, 0, 0, 0, 0])
    assert A.charpoly() == p
    assert A.to_sparse().charpoly() == p

    Ans = DomainMatrix([[QQ(1), QQ(2)]], (1, 2), QQ)
    raises(DMNonSquareMatrixError, lambda: Ans.charpoly())


def test_DomainMatrix_charpoly_factor_list():
    A = DomainMatrix([], (0, 0), ZZ)
    assert A.charpoly_factor_list() == []

    A = DM([[1]], ZZ)
    assert A.charpoly_factor_list() == [
        ([ZZ(1), ZZ(-1)], 1)
    ]

    A = DM([[1, 2], [3, 4]], ZZ)
    assert A.charpoly_factor_list() == [
        ([ZZ(1), ZZ(-5), ZZ(-2)], 1)
    ]

    A = DM([[1, 2, 0], [3, 4, 0], [0, 0, 1]], ZZ)
    assert A.charpoly_factor_list() == [
        ([ZZ(1), ZZ(-1)], 1),
        ([ZZ(1), ZZ(-5), ZZ(-2)], 1)
    ]


def test_DomainMatrix_eye():
    A = DomainMatrix.eye(3, QQ)
    assert A.rep == SDM.eye((3, 3), QQ)
    assert A.shape == (3, 3)
    assert A.domain == QQ


def test_DomainMatrix_zeros():
    A = DomainMatrix.zeros((1, 2), QQ)
    assert A.rep == SDM.zeros((1, 2), QQ)
    assert A.shape == (1, 2)
    assert A.domain == QQ


def test_DomainMatrix_ones():
    A = DomainMatrix.ones((2, 3), QQ)
    if GROUND_TYPES != 'flint':
        assert A.rep == DDM.ones((2, 3), QQ)
    else:
        assert A.rep == SDM.ones((2, 3), QQ).to_dfm()
    assert A.shape == (2, 3)
    assert A.domain == QQ


def test_DomainMatrix_diag():
    A = DomainMatrix({0:{0:ZZ(2)}, 1:{1:ZZ(3)}}, (2, 2), ZZ)
    assert DomainMatrix.diag([ZZ(2), ZZ(3)], ZZ) == A

    A = DomainMatrix({0:{0:ZZ(2)}, 1:{1:ZZ(3)}}, (3, 4), ZZ)
    assert DomainMatrix.diag([ZZ(2), ZZ(3)], ZZ, (3, 4)) == A


def test_DomainMatrix_hstack():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
    C = DomainMatrix([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)

    AB = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(5), ZZ(6)],
        [ZZ(3), ZZ(4), ZZ(7), ZZ(8)]], (2, 4), ZZ)
    ABC = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(5), ZZ(6), ZZ(9), ZZ(10)],
        [ZZ(3), ZZ(4), ZZ(7), ZZ(8), ZZ(11), ZZ(12)]], (2, 6), ZZ)
    assert A.hstack(B) == AB
    assert A.hstack(B, C) == ABC


def test_DomainMatrix_vstack():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
    C = DomainMatrix([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)

    AB = DomainMatrix([
        [ZZ(1), ZZ(2)],
        [ZZ(3), ZZ(4)],
        [ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8)]], (4, 2), ZZ)
    ABC = DomainMatrix([
        [ZZ(1), ZZ(2)],
        [ZZ(3), ZZ(4)],
        [ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8)],
        [ZZ(9), ZZ(10)],
        [ZZ(11), ZZ(12)]], (6, 2), ZZ)
    assert A.vstack(B) == AB
    assert A.vstack(B, C) == ABC


def test_DomainMatrix_applyfunc():
    A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    B = DomainMatrix([[ZZ(2), ZZ(4)]], (1, 2), ZZ)
    assert A.applyfunc(lambda x: 2*x) == B


def test_DomainMatrix_scalarmul():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    lamda = DomainScalar(QQ(3)/QQ(2), QQ)
    assert A * lamda == DomainMatrix([[QQ(3, 2), QQ(3)], [QQ(9, 2), QQ(6)]], (2, 2), QQ)
    assert A * 2 == DomainMatrix([[ZZ(2), ZZ(4)], [ZZ(6), ZZ(8)]], (2, 2), ZZ)
    assert 2 * A == DomainMatrix([[ZZ(2), ZZ(4)], [ZZ(6), ZZ(8)]], (2, 2), ZZ)
    assert A * DomainScalar(ZZ(0), ZZ) == DomainMatrix({}, (2, 2), ZZ)
    assert A * DomainScalar(ZZ(1), ZZ) == A

    raises(TypeError, lambda: A * 1.5)


def test_DomainMatrix_truediv():
    A = DomainMatrix.from_Matrix(Matrix([[1, 2], [3, 4]]))
    lamda = DomainScalar(QQ(3)/QQ(2), QQ)
    assert A / lamda == DomainMatrix({0: {0: QQ(2, 3), 1: QQ(4, 3)}, 1: {0: QQ(2), 1: QQ(8, 3)}}, (2, 2), QQ)
    b = DomainScalar(ZZ(1), ZZ)
    assert A / b == DomainMatrix({0: {0: QQ(1), 1: QQ(2)}, 1: {0: QQ(3), 1: QQ(4)}}, (2, 2), QQ)

    assert A / 1 == DomainMatrix({0: {0: QQ(1), 1: QQ(2)}, 1: {0: QQ(3), 1: QQ(4)}}, (2, 2), QQ)
    assert A / 2 == DomainMatrix({0: {0: QQ(1, 2), 1: QQ(1)}, 1: {0: QQ(3, 2), 1: QQ(2)}}, (2, 2), QQ)

    raises(ZeroDivisionError, lambda: A / 0)
    raises(TypeError, lambda: A / 1.5)
    raises(ZeroDivisionError, lambda: A / DomainScalar(ZZ(0), ZZ))

    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.to_field() / 2 == DomainMatrix([[QQ(1, 2), QQ(1)], [QQ(3, 2), QQ(2)]], (2, 2), QQ)
    assert A / 2 == DomainMatrix([[QQ(1, 2), QQ(1)], [QQ(3, 2), QQ(2)]], (2, 2), QQ)
    assert A.to_field() / QQ(2,3) == DomainMatrix([[QQ(3, 2), QQ(3)], [QQ(9, 2), QQ(6)]], (2, 2), QQ)


def test_DomainMatrix_getitem():
    dM = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(3)],
        [ZZ(4), ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)

    assert dM[1:,:-2] == DomainMatrix([[ZZ(4)], [ZZ(7)]], (2, 1), ZZ)
    assert dM[2,:-2] == DomainMatrix([[ZZ(7)]], (1, 1), ZZ)
    assert dM[:-2,:-2] == DomainMatrix([[ZZ(1)]], (1, 1), ZZ)
    assert dM[:-1,0:2] == DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(4), ZZ(5)]], (2, 2), ZZ)
    assert dM[:, -1] == DomainMatrix([[ZZ(3)], [ZZ(6)], [ZZ(9)]], (3, 1), ZZ)
    assert dM[-1, :] == DomainMatrix([[ZZ(7), ZZ(8), ZZ(9)]], (1, 3), ZZ)
    assert dM[::-1, :] == DomainMatrix([
                            [ZZ(7), ZZ(8), ZZ(9)],
                            [ZZ(4), ZZ(5), ZZ(6)],
                            [ZZ(1), ZZ(2), ZZ(3)]], (3, 3), ZZ)

    raises(IndexError, lambda: dM[4, :-2])
    raises(IndexError, lambda: dM[:-2, 4])

    assert dM[1, 2] == DomainScalar(ZZ(6), ZZ)
    assert dM[-2, 2] == DomainScalar(ZZ(6), ZZ)
    assert dM[1, -2] == DomainScalar(ZZ(5), ZZ)
    assert dM[-1, -3] == DomainScalar(ZZ(7), ZZ)

    raises(IndexError, lambda: dM[3, 3])
    raises(IndexError, lambda: dM[1, 4])
    raises(IndexError, lambda: dM[-1, -4])

    dM = DomainMatrix({0: {0: ZZ(1)}}, (10, 10), ZZ)
    assert dM[5, 5] == DomainScalar(ZZ(0), ZZ)
    assert dM[0, 0] == DomainScalar(ZZ(1), ZZ)

    dM = DomainMatrix({1: {0: 1}}, (2,1), ZZ)
    assert dM[0:, 0] == DomainMatrix({1: {0: 1}}, (2, 1), ZZ)
    raises(IndexError, lambda: dM[3, 0])

    dM = DomainMatrix({2: {2: ZZ(1)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    assert dM[:2,:2] == DomainMatrix({}, (2, 2), ZZ)
    assert dM[2:,2:] == DomainMatrix({0: {0: 1}, 2: {2: 1}}, (3, 3), ZZ)
    assert dM[3:,3:] == DomainMatrix({1: {1: 1}}, (2, 2), ZZ)
    assert dM[2:, 6:] == DomainMatrix({}, (3, 0), ZZ)


def test_DomainMatrix_getitem_sympy():
    dM = DomainMatrix({2: {2: ZZ(2)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    val1 = dM.getitem_sympy(0, 0)
    assert val1 is S.Zero
    val2 = dM.getitem_sympy(2, 2)
    assert val2 == 2 and isinstance(val2, Integer)


def test_DomainMatrix_extract():
    dM1 = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(3)],
        [ZZ(4), ZZ(5), ZZ(6)],
        [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)
    dM2 = DomainMatrix([
        [ZZ(1), ZZ(3)],
        [ZZ(7), ZZ(9)]], (2, 2), ZZ)
    assert dM1.extract([0, 2], [0, 2]) == dM2
    assert dM1.to_sparse().extract([0, 2], [0, 2]) == dM2.to_sparse()
    assert dM1.extract([0, -1], [0, -1]) == dM2
    assert dM1.to_sparse().extract([0, -1], [0, -1]) == dM2.to_sparse()

    dM3 = DomainMatrix([
        [ZZ(1), ZZ(2), ZZ(2)],
        [ZZ(4), ZZ(5), ZZ(5)],
        [ZZ(4), ZZ(5), ZZ(5)]], (3, 3), ZZ)
    assert dM1.extract([0, 1, 1], [0, 1, 1]) == dM3
    assert dM1.to_sparse().extract([0, 1, 1], [0, 1, 1]) == dM3.to_sparse()

    empty = [
        ([], [], (0, 0)),
        ([1], [], (1, 0)),
        ([], [1], (0, 1)),
    ]
    for rows, cols, size in empty:
        assert dM1.extract(rows, cols) == DomainMatrix.zeros(size, ZZ).to_dense()
        assert dM1.to_sparse().extract(rows, cols) == DomainMatrix.zeros(size, ZZ)

    dM = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    bad_indices = [([2], [0]), ([0], [2]), ([-3], [0]), ([0], [-3])]
    for rows, cols in bad_indices:
        raises(IndexError, lambda: dM.extract(rows, cols))
        raises(IndexError, lambda: dM.to_sparse().extract(rows, cols))


def test_DomainMatrix_setitem():
    dM = DomainMatrix({2: {2: ZZ(1)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    dM[2, 2] = ZZ(2)
    assert dM == DomainMatrix({2: {2: ZZ(2)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    def setitem(i, j, val):
        dM[i, j] = val
    raises(TypeError, lambda: setitem(2, 2, QQ(1, 2)))
    raises(NotImplementedError, lambda: setitem(slice(1, 2), 2, ZZ(1)))


def test_DomainMatrix_pickling():
    import pickle
    dM = DomainMatrix({2: {2: ZZ(1)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    assert pickle.loads(pickle.dumps(dM)) == dM
    dM = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert pickle.loads(pickle.dumps(dM)) == dM


def test_DomainMatrix_fflu():
    A = DM([[1, 2], [3, 4]], ZZ)
    P, L, D, U = A.fflu()
    assert P.shape == A.shape
    assert L.shape == A.shape
    assert D.shape == A.shape
    assert U.shape == A.shape
    assert P == DM([[1, 0], [0, 1]], ZZ)
    assert L == DM([[1, 0], [3, -2]], ZZ)
    assert D == DM([[1, 0], [0, -2]], ZZ)
    assert U == DM([[1, 2], [0, -2]], ZZ)
    di, d = D.inv_den()
    assert P.matmul(A).rmul(d) == L.matmul(di).matmul(U)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flatbuffers\encode.py ===
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

from . import number_types as N
from . import packer
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature

np = import_numpy()

FILE_IDENTIFIER_LENGTH=4

def Get(packer_type, buf, head):
    """ Get decodes a value at buf[head] using `packer_type`. """
    return packer_type.unpack_from(memoryview_type(buf), head)[0]


def GetVectorAsNumpy(numpy_type, buf, count, offset):
    """ GetVecAsNumpy decodes values starting at buf[head] as
    `numpy_type`, where `numpy_type` is a numpy dtype. """
    if np is not None:
        # TODO: could set .flags.writeable = False to make users jump through
        #       hoops before modifying...
        return np.frombuffer(buf, dtype=numpy_type, count=count, offset=offset)
    else:
        raise NumpyRequiredForThisFeature('Numpy was not found.')


def Write(packer_type, buf, head, n):
    """ Write encodes `n` at buf[head] using `packer_type`. """
    packer_type.pack_into(buf, head, n)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flatbuffers\packer.py ===
# Copyright 2016 Google Inc. All rights reserved.
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

"""
Provide pre-compiled struct packers for encoding and decoding.

See: https://docs.python.org/2/library/struct.html#format-characters
"""

import struct
from . import compat


boolean = struct.Struct(compat.struct_bool_decl)

uint8 = struct.Struct("<B")
uint16 = struct.Struct("<H")
uint32 = struct.Struct("<I")
uint64 = struct.Struct("<Q")

int8 = struct.Struct("<b")
int16 = struct.Struct("<h")
int32 = struct.Struct("<i")
int64 = struct.Struct("<q")

float32 = struct.Struct("<f")
float64 = struct.Struct("<d")

uoffset = uint32
soffset = int32
voffset = uint16

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\onnx_model_clip.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

from fusion_attention_clip import FusionAttentionClip
from onnx import ModelProto
from onnx_model_bert import BertOnnxModel

logger = getLogger(__name__)


class ClipOnnxModel(BertOnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, hidden_size: int = 0):
        super().__init__(model, num_heads=num_heads, hidden_size=hidden_size)
        self.clip_attention_fusion = FusionAttentionClip(self, self.hidden_size, self.num_heads)

    def get_fused_operator_statistics(self):
        """
        Returns node count of fused operators.
        """
        op_count = {}
        ops = [
            "Attention",
            "FastGelu",
            "Gelu",
            "LayerNormalization",
            "QuickGelu",
            "BiasGelu",
            "SkipLayerNormalization",
        ]
        for op in ops:
            nodes = self.get_nodes_by_op_type(op)
            op_count[op] = len(nodes)

        logger.info(f"Optimized operators:{op_count}")
        return op_count

    def fuse_attention(self):
        self.clip_attention_fusion.apply()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\onnx_model_vae.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

from fusion_attention_vae import FusionAttentionVae
from fusion_options import FusionOptions
from onnx import ModelProto
from onnx_model_unet import UnetOnnxModel

logger = getLogger(__name__)


class VaeOnnxModel(UnetOnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, hidden_size: int = 0):
        assert (num_heads == 0 and hidden_size == 0) or (num_heads > 0 and hidden_size % num_heads == 0)
        super().__init__(model, num_heads=num_heads, hidden_size=hidden_size)

    def fuse_multi_head_attention(self, options: FusionOptions | None = None):
        # Self Attention
        self_attention_fusion = FusionAttentionVae(self, self.hidden_size, self.num_heads)
        self_attention_fusion.apply()

    def get_fused_operator_statistics(self):
        """
        Returns node count of fused operators.
        """
        op_count = {}
        ops = [
            "Attention",
            "GroupNorm",
            "SkipGroupNorm",
            "NhwcConv",
        ]
        for op in ops:
            nodes = self.get_nodes_by_op_type(op)
            op_count[op] = len(nodes)

        logger.info(f"Optimized operators:{op_count}")
        return op_count

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\xml\__init__.py ===
# Copyright (c) 2010-2024 openpyxl


"""Collection of XML resources compatible across different Python versions"""
import os


def lxml_available():
    try:
        from lxml.etree import LXML_VERSION
        LXML = LXML_VERSION >= (3, 3, 1, 0)
        if not LXML:
            import warnings
            warnings.warn("The installed version of lxml is too old to be used with openpyxl")
            return False  # we have it, but too old
        else:
            return True  # we have it, and recent enough
    except ImportError:
        return False  # we don't even have it


def lxml_env_set():
    return os.environ.get("OPENPYXL_LXML", "True") == "True"


LXML = lxml_available() and lxml_env_set()


def defusedxml_available():
    try:
        import defusedxml # noqa
    except ImportError:
        return False
    else:
        return True


def defusedxml_env_set():
    return os.environ.get("OPENPYXL_DEFUSEDXML", "True") == "True"


DEFUSEDXML = defusedxml_available() and defusedxml_env_set()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\xml\test_to_xml.py ===
from __future__ import annotations

from io import (
    BytesIO,
    StringIO,
)
import os

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    NA,
    DataFrame,
    Index,
)
import pandas._testing as tm

from pandas.io.common import get_handle
from pandas.io.xml import read_xml

# CHECKLIST

# [x] - ValueError: "Values for parser can only be lxml or etree."

# etree
# [x] - ImportError: "lxml not found, please install or use the etree parser."
# [X] - TypeError: "...is not a valid type for attr_cols"
# [X] - TypeError: "...is not a valid type for elem_cols"
# [X] - LookupError: "unknown encoding"
# [X] - KeyError: "...is not included in namespaces"
# [X] - KeyError: "no valid column"
# [X] - ValueError: "To use stylesheet, you need lxml installed..."
# []  - OSError: (NEED PERMISSOIN ISSUE, DISK FULL, ETC.)
# [X] - FileNotFoundError: "No such file or directory"
# [X] - PermissionError: "Forbidden"

# lxml
# [X] - TypeError: "...is not a valid type for attr_cols"
# [X] - TypeError: "...is not a valid type for elem_cols"
# [X] - LookupError: "unknown encoding"
# []  - OSError: (NEED PERMISSOIN ISSUE, DISK FULL, ETC.)
# [X] - FileNotFoundError: "No such file or directory"
# [X] - KeyError: "...is not included in namespaces"
# [X] - KeyError: "no valid column"
# [X] - ValueError: "stylesheet is not a url, file, or xml string."
# []  - LookupError: (NEED WRONG ENCODING FOR FILE OUTPUT)
# []  - URLError: (USUALLY DUE TO NETWORKING)
# []  - HTTPError: (NEED AN ONLINE STYLESHEET)
# [X] - OSError: "failed to load external entity"
# [X] - XMLSyntaxError: "Opening and ending tag mismatch"
# [X] - XSLTApplyError: "Cannot resolve URI"
# [X] - XSLTParseError: "failed to compile"
# [X] - PermissionError: "Forbidden"


@pytest.fixture
def geom_df():
    return DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4, np.nan, 3],
        }
    )


@pytest.fixture
def planet_df():
    return DataFrame(
        {
            "planet": [
                "Mercury",
                "Venus",
                "Earth",
                "Mars",
                "Jupiter",
                "Saturn",
                "Uranus",
                "Neptune",
            ],
            "type": [
                "terrestrial",
                "terrestrial",
                "terrestrial",
                "terrestrial",
                "gas giant",
                "gas giant",
                "ice giant",
                "ice giant",
            ],
            "location": [
                "inner",
                "inner",
                "inner",
                "inner",
                "outer",
                "outer",
                "outer",
                "outer",
            ],
            "mass": [
                0.330114,
                4.86747,
                5.97237,
                0.641712,
                1898.187,
                568.3174,
                86.8127,
                102.4126,
            ],
        }
    )


@pytest.fixture
def from_file_expected():
    return """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <index>0</index>
    <category>cooking</category>
    <title>Everyday Italian</title>
    <author>Giada De Laurentiis</author>
    <year>2005</year>
    <price>30.0</price>
  </row>
  <row>
    <index>1</index>
    <category>children</category>
    <title>Harry Potter</title>
    <author>J K. Rowling</author>
    <year>2005</year>
    <price>29.99</price>
  </row>
  <row>
    <index>2</index>
    <category>web</category>
    <title>Learning XML</title>
    <author>Erik T. Ray</author>
    <year>2003</year>
    <price>39.95</price>
  </row>
</data>"""


def equalize_decl(doc):
    # etree and lxml differ on quotes and case in xml declaration
    if doc is not None:
        doc = doc.replace(
            '<?xml version="1.0" encoding="utf-8"?',
            "<?xml version='1.0' encoding='utf-8'?",
        )
    return doc


@pytest.fixture(params=["rb", "r"])
def mode(request):
    return request.param


@pytest.fixture(params=[pytest.param("lxml", marks=td.skip_if_no("lxml")), "etree"])
def parser(request):
    return request.param


# FILE OUTPUT


def test_file_output_str_read(xml_books, parser, from_file_expected):
    df_file = read_xml(xml_books, parser=parser)

    with tm.ensure_clean("test.xml") as path:
        df_file.to_xml(path, parser=parser)
        with open(path, "rb") as f:
            output = f.read().decode("utf-8").strip()

        output = equalize_decl(output)

        assert output == from_file_expected


def test_file_output_bytes_read(xml_books, parser, from_file_expected):
    df_file = read_xml(xml_books, parser=parser)

    with tm.ensure_clean("test.xml") as path:
        df_file.to_xml(path, parser=parser)
        with open(path, "rb") as f:
            output = f.read().decode("utf-8").strip()

        output = equalize_decl(output)

        assert output == from_file_expected


def test_str_output(xml_books, parser, from_file_expected):
    df_file = read_xml(xml_books, parser=parser)

    output = df_file.to_xml(parser=parser)
    output = equalize_decl(output)

    assert output == from_file_expected


def test_wrong_file_path(parser, geom_df):
    path = "/my/fake/path/output.xml"

    with pytest.raises(
        OSError,
        match=(r"Cannot save file into a non-existent directory: .*path"),
    ):
        geom_df.to_xml(path, parser=parser)


# INDEX


def test_index_false(xml_books, parser):
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <category>cooking</category>
    <title>Everyday Italian</title>
    <author>Giada De Laurentiis</author>
    <year>2005</year>
    <price>30.0</price>
  </row>
  <row>
    <category>children</category>
    <title>Harry Potter</title>
    <author>J K. Rowling</author>
    <year>2005</year>
    <price>29.99</price>
  </row>
  <row>
    <category>web</category>
    <title>Learning XML</title>
    <author>Erik T. Ray</author>
    <year>2003</year>
    <price>39.95</price>
  </row>
</data>"""

    df_file = read_xml(xml_books, parser=parser)

    with tm.ensure_clean("test.xml") as path:
        df_file.to_xml(path, index=False, parser=parser)
        with open(path, "rb") as f:
            output = f.read().decode("utf-8").strip()

        output = equalize_decl(output)

        assert output == expected


def test_index_false_rename_row_root(xml_books, parser):
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<books>
  <book>
    <category>cooking</category>
    <title>Everyday Italian</title>
    <author>Giada De Laurentiis</author>
    <year>2005</year>
    <price>30.0</price>
  </book>
  <book>
    <category>children</category>
    <title>Harry Potter</title>
    <author>J K. Rowling</author>
    <year>2005</year>
    <price>29.99</price>
  </book>
  <book>
    <category>web</category>
    <title>Learning XML</title>
    <author>Erik T. Ray</author>
    <year>2003</year>
    <price>39.95</price>
  </book>
</books>"""

    df_file = read_xml(xml_books, parser=parser)

    with tm.ensure_clean("test.xml") as path:
        df_file.to_xml(
            path, index=False, root_name="books", row_name="book", parser=parser
        )
        with open(path, "rb") as f:
            output = f.read().decode("utf-8").strip()

        output = equalize_decl(output)

        assert output == expected


@pytest.mark.parametrize(
    "offset_index", [list(range(10, 13)), [str(i) for i in range(10, 13)]]
)
def test_index_false_with_offset_input_index(parser, offset_index, geom_df):
    """
    Tests that the output does not contain the `<index>` field when the index of the
    input Dataframe has an offset.

    This is a regression test for issue #42458.
    """

    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4.0</sides>
  </row>
  <row>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides/>
  </row>
  <row>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""

    offset_geom_df = geom_df.copy()
    offset_geom_df.index = Index(offset_index)
    output = offset_geom_df.to_xml(index=False, parser=parser)
    output = equalize_decl(output)

    assert output == expected


# NA_REP

na_expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <index>0</index>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4.0</sides>
  </row>
  <row>
    <index>1</index>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides/>
  </row>
  <row>
    <index>2</index>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""


def test_na_elem_output(parser, geom_df):
    output = geom_df.to_xml(parser=parser)
    output = equalize_decl(output)

    assert output == na_expected


def test_na_empty_str_elem_option(parser, geom_df):
    output = geom_df.to_xml(na_rep="", parser=parser)
    output = equalize_decl(output)

    assert output == na_expected


def test_na_empty_elem_option(parser, geom_df):
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <index>0</index>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4.0</sides>
  </row>
  <row>
    <index>1</index>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides>0.0</sides>
  </row>
  <row>
    <index>2</index>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""

    output = geom_df.to_xml(na_rep="0.0", parser=parser)
    output = equalize_decl(output)

    assert output == expected


# ATTR_COLS


def test_attrs_cols_nan_output(parser, geom_df):
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row index="0" shape="square" degrees="360" sides="4.0"/>
  <row index="1" shape="circle" degrees="360"/>
  <row index="2" shape="triangle" degrees="180" sides="3.0"/>
</data>"""

    output = geom_df.to_xml(attr_cols=["shape", "degrees", "sides"], parser=parser)
    output = equalize_decl(output)

    assert output == expected


def test_attrs_cols_prefix(parser, geom_df):
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<doc:data xmlns:doc="http://example.xom">
  <doc:row doc:index="0" doc:shape="square" \
doc:degrees="360" doc:sides="4.0"/>
  <doc:row doc:index="1" doc:shape="circle" \
doc:degrees="360"/>
  <doc:row doc:index="2" doc:shape="triangle" \
doc:degrees="180" doc:sides="3.0"/>
</doc:data>"""

    output = geom_df.to_xml(
        attr_cols=["index", "shape", "degrees", "sides"],
        namespaces={"doc": "http://example.xom"},
        prefix="doc",
        parser=parser,
    )
    output = equalize_decl(output)

    assert output == expected


def test_attrs_unknown_column(parser, geom_df):
    with pytest.raises(KeyError, match=("no valid column")):
        geom_df.to_xml(attr_cols=["shape", "degree", "sides"], parser=parser)


def test_attrs_wrong_type(parser, geom_df):
    with pytest.raises(TypeError, match=("is not a valid type for attr_cols")):
        geom_df.to_xml(attr_cols='"shape", "degree", "sides"', parser=parser)


# ELEM_COLS


def test_elems_cols_nan_output(parser, geom_df):
    elems_cols_expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <degrees>360</degrees>
    <sides>4.0</sides>
    <shape>square</shape>
  </row>
  <row>
    <degrees>360</degrees>
    <sides/>
    <shape>circle</shape>
  </row>
  <row>
    <degrees>180</degrees>
    <sides>3.0</sides>
    <shape>triangle</shape>
  </row>
</data>"""

    output = geom_df.to_xml(
        index=False, elem_cols=["degrees", "sides", "shape"], parser=parser
    )
    output = equalize_decl(output)

    assert output == elems_cols_expected


def test_elems_unknown_column(parser, geom_df):
    with pytest.raises(KeyError, match=("no valid column")):
        geom_df.to_xml(elem_cols=["shape", "degree", "sides"], parser=parser)


def test_elems_wrong_type(parser, geom_df):
    with pytest.raises(TypeError, match=("is not a valid type for elem_cols")):
        geom_df.to_xml(elem_cols='"shape", "degree", "sides"', parser=parser)


def test_elems_and_attrs_cols(parser, geom_df):
    elems_cols_expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row shape="square">
    <degrees>360</degrees>
    <sides>4.0</sides>
  </row>
  <row shape="circle">
    <degrees>360</degrees>
    <sides/>
  </row>
  <row shape="triangle">
    <degrees>180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""

    output = geom_df.to_xml(
        index=False,
        elem_cols=["degrees", "sides"],
        attr_cols=["shape"],
        parser=parser,
    )
    output = equalize_decl(output)

    assert output == elems_cols_expected


# HIERARCHICAL COLUMNS


def test_hierarchical_columns(parser, planet_df):
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <location>inner</location>
    <type>terrestrial</type>
    <count_mass>4</count_mass>
    <sum_mass>11.81</sum_mass>
    <mean_mass>2.95</mean_mass>
  </row>
  <row>
    <location>outer</location>
    <type>gas giant</type>
    <count_mass>2</count_mass>
    <sum_mass>2466.5</sum_mass>
    <mean_mass>1233.25</mean_mass>
  </row>
  <row>
    <location>outer</location>
    <type>ice giant</type>
    <count_mass>2</count_mass>
    <sum_mass>189.23</sum_mass>
    <mean_mass>94.61</mean_mass>
  </row>
  <row>
    <location>All</location>
    <type/>
    <count_mass>8</count_mass>
    <sum_mass>2667.54</sum_mass>
    <mean_mass>333.44</mean_mass>
  </row>
</data>"""

    pvt = planet_df.pivot_table(
        index=["location", "type"],
        values="mass",
        aggfunc=["count", "sum", "mean"],
        margins=True,
    ).round(2)

    output = pvt.to_xml(parser=parser)
    output = equalize_decl(output)

    assert output == expected


def test_hierarchical_attrs_columns(parser, planet_df):
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row location="inner" type="terrestrial" count_mass="4" \
sum_mass="11.81" mean_mass="2.95"/>
  <row location="outer" type="gas giant" count_mass="2" \
sum_mass="2466.5" mean_mass="1233.25"/>
  <row location="outer" type="ice giant" count_mass="2" \
sum_mass="189.23" mean_mass="94.61"/>
  <row location="All" type="" count_mass="8" \
sum_mass="2667.54" mean_mass="333.44"/>
</data>"""

    pvt = planet_df.pivot_table(
        index=["location", "type"],
        values="mass",
        aggfunc=["count", "sum", "mean"],
        margins=True,
    ).round(2)

    output = pvt.to_xml(attr_cols=list(pvt.reset_index().columns.values), parser=parser)
    output = equalize_decl(output)

    assert output == expected


# MULTIINDEX


def test_multi_index(parser, planet_df):
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <location>inner</location>
    <type>terrestrial</type>
    <count>4</count>
    <sum>11.81</sum>
    <mean>2.95</mean>
  </row>
  <row>
    <location>outer</location>
    <type>gas giant</type>
    <count>2</count>
    <sum>2466.5</sum>
    <mean>1233.25</mean>
  </row>
  <row>
    <location>outer</location>
    <type>ice giant</type>
    <count>2</count>
    <sum>189.23</sum>
    <mean>94.61</mean>
  </row>
</data>"""

    agg = (
        planet_df.groupby(["location", "type"])["mass"]
        .agg(["count", "sum", "mean"])
        .round(2)
    )

    output = agg.to_xml(parser=parser)
    output = equalize_decl(output)

    assert output == expected


def test_multi_index_attrs_cols(parser, planet_df):
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row location="inner" type="terrestrial" count="4" \
sum="11.81" mean="2.95"/>
  <row location="outer" type="gas giant" count="2" \
sum="2466.5" mean="1233.25"/>
  <row location="outer" type="ice giant" count="2" \
sum="189.23" mean="94.61"/>
</data>"""

    agg = (
        planet_df.groupby(["location", "type"])["mass"]
        .agg(["count", "sum", "mean"])
        .round(2)
    )
    output = agg.to_xml(attr_cols=list(agg.reset_index().columns.values), parser=parser)
    output = equalize_decl(output)

    assert output == expected


# NAMESPACE


def test_default_namespace(parser, geom_df):
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data xmlns="http://example.com">
  <row>
    <index>0</index>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4.0</sides>
  </row>
  <row>
    <index>1</index>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides/>
  </row>
  <row>
    <index>2</index>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""

    output = geom_df.to_xml(namespaces={"": "http://example.com"}, parser=parser)
    output = equalize_decl(output)

    assert output == expected


def test_unused_namespaces(parser, geom_df):
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<data xmlns:oth="http://other.org" xmlns:ex="http://example.com">
  <row>
    <index>0</index>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4.0</sides>
  </row>
  <row>
    <index>1</index>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides/>
  </row>
  <row>
    <index>2</index>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""

    output = geom_df.to_xml(
        namespaces={"oth": "http://other.org", "ex": "http://example.com"},
        parser=parser,
    )
    output = equalize_decl(output)

    assert output == expected


# PREFIX


def test_namespace_prefix(parser, geom_df):
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<doc:data xmlns:doc="http://example.com">
  <doc:row>
    <doc:index>0</doc:index>
    <doc:shape>square</doc:shape>
    <doc:degrees>360</doc:degrees>
    <doc:sides>4.0</doc:sides>
  </doc:row>
  <doc:row>
    <doc:index>1</doc:index>
    <doc:shape>circle</doc:shape>
    <doc:degrees>360</doc:degrees>
    <doc:sides/>
  </doc:row>
  <doc:row>
    <doc:index>2</doc:index>
    <doc:shape>triangle</doc:shape>
    <doc:degrees>180</doc:degrees>
    <doc:sides>3.0</doc:sides>
  </doc:row>
</doc:data>"""

    output = geom_df.to_xml(
        namespaces={"doc": "http://example.com"}, prefix="doc", parser=parser
    )
    output = equalize_decl(output)

    assert output == expected


def test_missing_prefix_in_nmsp(parser, geom_df):
    with pytest.raises(KeyError, match=("doc is not included in namespaces")):
        geom_df.to_xml(
            namespaces={"": "http://example.com"}, prefix="doc", parser=parser
        )


def test_namespace_prefix_and_default(parser, geom_df):
    expected = """\
<?xml version='1.0' encoding='utf-8'?>
<doc:data xmlns:doc="http://other.org" xmlns="http://example.com">
  <doc:row>
    <doc:index>0</doc:index>
    <doc:shape>square</doc:shape>
    <doc:degrees>360</doc:degrees>
    <doc:sides>4.0</doc:sides>
  </doc:row>
  <doc:row>
    <doc:index>1</doc:index>
    <doc:shape>circle</doc:shape>
    <doc:degrees>360</doc:degrees>
    <doc:sides/>
  </doc:row>
  <doc:row>
    <doc:index>2</doc:index>
    <doc:shape>triangle</doc:shape>
    <doc:degrees>180</doc:degrees>
    <doc:sides>3.0</doc:sides>
  </doc:row>
</doc:data>"""

    output = geom_df.to_xml(
        namespaces={"": "http://example.com", "doc": "http://other.org"},
        prefix="doc",
        parser=parser,
    )
    output = equalize_decl(output)

    assert output == expected


# ENCODING

encoding_expected = """\
<?xml version='1.0' encoding='ISO-8859-1'?>
<data>
  <row>
    <index>0</index>
    <rank>1</rank>
    <malename>José</malename>
    <femalename>Sofía</femalename>
  </row>
  <row>
    <index>1</index>
    <rank>2</rank>
    <malename>Luis</malename>
    <femalename>Valentina</femalename>
  </row>
  <row>
    <index>2</index>
    <rank>3</rank>
    <malename>Carlos</malename>
    <femalename>Isabella</femalename>
  </row>
  <row>
    <index>3</index>
    <rank>4</rank>
    <malename>Juan</malename>
    <femalename>Camila</femalename>
  </row>
  <row>
    <index>4</index>
    <rank>5</rank>
    <malename>Jorge</malename>
    <femalename>Valeria</femalename>
  </row>
</data>"""


def test_encoding_option_str(xml_baby_names, parser):
    df_file = read_xml(xml_baby_names, parser=parser, encoding="ISO-8859-1").head(5)

    output = df_file.to_xml(encoding="ISO-8859-1", parser=parser)

    if output is not None:
        # etree and lxml differ on quotes and case in xml declaration
        output = output.replace(
            '<?xml version="1.0" encoding="ISO-8859-1"?',
            "<?xml version='1.0' encoding='ISO-8859-1'?",
        )

    assert output == encoding_expected


def test_correct_encoding_file(xml_baby_names):
    pytest.importorskip("lxml")
    df_file = read_xml(xml_baby_names, encoding="ISO-8859-1", parser="lxml")

    with tm.ensure_clean("test.xml") as path:
        df_file.to_xml(path, index=False, encoding="ISO-8859-1", parser="lxml")


@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16", "ISO-8859-1"])
def test_wrong_encoding_option_lxml(xml_baby_names, parser, encoding):
    pytest.importorskip("lxml")
    df_file = read_xml(xml_baby_names, encoding="ISO-8859-1", parser="lxml")

    with tm.ensure_clean("test.xml") as path:
        df_file.to_xml(path, index=False, encoding=encoding, parser=parser)


def test_misspelled_encoding(parser, geom_df):
    with pytest.raises(LookupError, match=("unknown encoding")):
        geom_df.to_xml(encoding="uft-8", parser=parser)


# PRETTY PRINT


def test_xml_declaration_pretty_print(geom_df):
    pytest.importorskip("lxml")
    expected = """\
<data>
  <row>
    <index>0</index>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4.0</sides>
  </row>
  <row>
    <index>1</index>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides/>
  </row>
  <row>
    <index>2</index>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""

    output = geom_df.to_xml(xml_declaration=False)

    assert output == expected


def test_no_pretty_print_with_decl(parser, geom_df):
    expected = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<data><row><index>0</index><shape>square</shape>"
        "<degrees>360</degrees><sides>4.0</sides></row><row>"
        "<index>1</index><shape>circle</shape><degrees>360"
        "</degrees><sides/></row><row><index>2</index><shape>"
        "triangle</shape><degrees>180</degrees><sides>3.0</sides>"
        "</row></data>"
    )

    output = geom_df.to_xml(pretty_print=False, parser=parser)
    output = equalize_decl(output)

    # etree adds space for closed tags
    if output is not None:
        output = output.replace(" />", "/>")

    assert output == expected


def test_no_pretty_print_no_decl(parser, geom_df):
    expected = (
        "<data><row><index>0</index><shape>square</shape>"
        "<degrees>360</degrees><sides>4.0</sides></row><row>"
        "<index>1</index><shape>circle</shape><degrees>360"
        "</degrees><sides/></row><row><index>2</index><shape>"
        "triangle</shape><degrees>180</degrees><sides>3.0</sides>"
        "</row></data>"
    )

    output = geom_df.to_xml(xml_declaration=False, pretty_print=False, parser=parser)

    # etree adds space for closed tags
    if output is not None:
        output = output.replace(" />", "/>")

    assert output == expected


# PARSER


@td.skip_if_installed("lxml")
def test_default_parser_no_lxml(geom_df):
    with pytest.raises(
        ImportError, match=("lxml not found, please install or use the etree parser.")
    ):
        geom_df.to_xml()


def test_unknown_parser(geom_df):
    with pytest.raises(
        ValueError, match=("Values for parser can only be lxml or etree.")
    ):
        geom_df.to_xml(parser="bs4")


# STYLESHEET

xsl_expected = """\
<?xml version="1.0" encoding="utf-8"?>
<data>
  <row>
    <field field="index">0</field>
    <field field="shape">square</field>
    <field field="degrees">360</field>
    <field field="sides">4.0</field>
  </row>
  <row>
    <field field="index">1</field>
    <field field="shape">circle</field>
    <field field="degrees">360</field>
    <field field="sides"/>
  </row>
  <row>
    <field field="index">2</field>
    <field field="shape">triangle</field>
    <field field="degrees">180</field>
    <field field="sides">3.0</field>
  </row>
</data>"""


def test_stylesheet_file_like(xsl_row_field_output, mode, geom_df):
    pytest.importorskip("lxml")
    with open(
        xsl_row_field_output, mode, encoding="utf-8" if mode == "r" else None
    ) as f:
        assert geom_df.to_xml(stylesheet=f) == xsl_expected


def test_stylesheet_io(xsl_row_field_output, mode, geom_df):
    # note: By default the bodies of untyped functions are not checked,
    # consider using --check-untyped-defs
    pytest.importorskip("lxml")
    xsl_obj: BytesIO | StringIO  # type: ignore[annotation-unchecked]

    with open(
        xsl_row_field_output, mode, encoding="utf-8" if mode == "r" else None
    ) as f:
        if mode == "rb":
            xsl_obj = BytesIO(f.read())
        else:
            xsl_obj = StringIO(f.read())

    output = geom_df.to_xml(stylesheet=xsl_obj)

    assert output == xsl_expected


def test_stylesheet_buffered_reader(xsl_row_field_output, mode, geom_df):
    pytest.importorskip("lxml")
    with open(
        xsl_row_field_output, mode, encoding="utf-8" if mode == "r" else None
    ) as f:
        xsl_obj = f.read()

    output = geom_df.to_xml(stylesheet=xsl_obj)

    assert output == xsl_expected


def test_stylesheet_wrong_path(geom_df):
    lxml_etree = pytest.importorskip("lxml.etree")

    xsl = os.path.join("data", "xml", "row_field_output.xslt")

    with pytest.raises(
        lxml_etree.XMLSyntaxError,
        match=("Start tag expected, '<' not found"),
    ):
        geom_df.to_xml(stylesheet=xsl)


@pytest.mark.parametrize("val", ["", b""])
def test_empty_string_stylesheet(val, geom_df):
    lxml_etree = pytest.importorskip("lxml.etree")

    msg = "|".join(
        [
            "Document is empty",
            "Start tag expected, '<' not found",
            # Seen on Mac with lxml 4.9.1
            r"None \(line 0\)",
        ]
    )

    with pytest.raises(lxml_etree.XMLSyntaxError, match=msg):
        geom_df.to_xml(stylesheet=val)


def test_incorrect_xsl_syntax(geom_df):
    lxml_etree = pytest.importorskip("lxml.etree")

    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="xml" encoding="utf-8" indent="yes" >
    <xsl:strip-space elements="*"/>

    <xsl:template match="@*|node()">
        <xsl:copy>
            <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
    </xsl:template>

    <xsl:template match="row/*">
        <field>
            <xsl:attribute name="field">
                <xsl:value-of select="name()"/>
            </xsl:attribute>
            <xsl:value-of select="text()"/>
        </field>
    </xsl:template>
</xsl:stylesheet>"""

    with pytest.raises(
        lxml_etree.XMLSyntaxError, match=("Opening and ending tag mismatch")
    ):
        geom_df.to_xml(stylesheet=xsl)


def test_incorrect_xsl_eval(geom_df):
    lxml_etree = pytest.importorskip("lxml.etree")

    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="xml" encoding="utf-8" indent="yes" />
    <xsl:strip-space elements="*"/>

    <xsl:template match="@*|node(*)">
        <xsl:copy>
            <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
    </xsl:template>

    <xsl:template match="row/*">
        <field>
            <xsl:attribute name="field">
                <xsl:value-of select="name()"/>
            </xsl:attribute>
            <xsl:value-of select="text()"/>
        </field>
    </xsl:template>
</xsl:stylesheet>"""

    with pytest.raises(lxml_etree.XSLTParseError, match=("failed to compile")):
        geom_df.to_xml(stylesheet=xsl)


def test_incorrect_xsl_apply(geom_df):
    lxml_etree = pytest.importorskip("lxml.etree")

    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="xml" encoding="utf-8" indent="yes" />
    <xsl:strip-space elements="*"/>

    <xsl:template match="@*|node()">
        <xsl:copy>
            <xsl:copy-of select="document('non_existent.xml')/*"/>
        </xsl:copy>
    </xsl:template>
</xsl:stylesheet>"""

    with pytest.raises(lxml_etree.XSLTApplyError, match=("Cannot resolve URI")):
        with tm.ensure_clean("test.xml") as path:
            geom_df.to_xml(path, stylesheet=xsl)


def test_stylesheet_with_etree(geom_df):
    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="xml" encoding="utf-8" indent="yes" />
    <xsl:strip-space elements="*"/>

    <xsl:template match="@*|node(*)">
        <xsl:copy>
            <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
    </xsl:template>"""

    with pytest.raises(
        ValueError, match=("To use stylesheet, you need lxml installed")
    ):
        geom_df.to_xml(parser="etree", stylesheet=xsl)


def test_style_to_csv(geom_df):
    pytest.importorskip("lxml")
    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="text" indent="yes" />
    <xsl:strip-space elements="*"/>

    <xsl:param name="delim">,</xsl:param>
    <xsl:template match="/data">
        <xsl:text>,shape,degrees,sides&#xa;</xsl:text>
        <xsl:apply-templates select="row"/>
    </xsl:template>

    <xsl:template match="row">
        <xsl:value-of select="concat(index, $delim, shape, $delim,
                                     degrees, $delim, sides)"/>
         <xsl:text>&#xa;</xsl:text>
    </xsl:template>
</xsl:stylesheet>"""

    out_csv = geom_df.to_csv(lineterminator="\n")

    if out_csv is not None:
        out_csv = out_csv.strip()
    out_xml = geom_df.to_xml(stylesheet=xsl)

    assert out_csv == out_xml


def test_style_to_string(geom_df):
    pytest.importorskip("lxml")
    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="text" indent="yes" />
    <xsl:strip-space elements="*"/>

    <xsl:param name="delim"><xsl:text>               </xsl:text></xsl:param>
    <xsl:template match="/data">
        <xsl:text>      shape  degrees  sides&#xa;</xsl:text>
        <xsl:apply-templates select="row"/>
    </xsl:template>

    <xsl:template match="row">
        <xsl:value-of select="concat(index, ' ',
                                     substring($delim, 1, string-length('triangle')
                                               - string-length(shape) + 1),
                                     shape,
                                     substring($delim, 1, string-length(name(degrees))
                                               - string-length(degrees) + 2),
                                     degrees,
                                     substring($delim, 1, string-length(name(sides))
                                               - string-length(sides) + 2),
                                     sides)"/>
         <xsl:text>&#xa;</xsl:text>
    </xsl:template>
</xsl:stylesheet>"""

    out_str = geom_df.to_string()
    out_xml = geom_df.to_xml(na_rep="NaN", stylesheet=xsl)

    assert out_xml == out_str


def test_style_to_json(geom_df):
    pytest.importorskip("lxml")
    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="text" indent="yes" />
    <xsl:strip-space elements="*"/>

    <xsl:param name="quot">"</xsl:param>

    <xsl:template match="/data">
        <xsl:text>{"shape":{</xsl:text>
        <xsl:apply-templates select="descendant::row/shape"/>
        <xsl:text>},"degrees":{</xsl:text>
        <xsl:apply-templates select="descendant::row/degrees"/>
        <xsl:text>},"sides":{</xsl:text>
        <xsl:apply-templates select="descendant::row/sides"/>
        <xsl:text>}}</xsl:text>
    </xsl:template>

    <xsl:template match="shape|degrees|sides">
        <xsl:variable name="val">
            <xsl:if test = ".=''">
                <xsl:value-of select="'null'"/>
            </xsl:if>
            <xsl:if test = "number(text()) = text()">
                <xsl:value-of select="text()"/>
            </xsl:if>
            <xsl:if test = "number(text()) != text()">
                <xsl:value-of select="concat($quot, text(), $quot)"/>
            </xsl:if>
        </xsl:variable>
        <xsl:value-of select="concat($quot, preceding-sibling::index,
                                     $quot,':', $val)"/>
        <xsl:if test="preceding-sibling::index != //row[last()]/index">
            <xsl:text>,</xsl:text>
        </xsl:if>
    </xsl:template>
</xsl:stylesheet>"""

    out_json = geom_df.to_json()
    out_xml = geom_df.to_xml(stylesheet=xsl)

    assert out_json == out_xml


# COMPRESSION


geom_xml = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <index>0</index>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4.0</sides>
  </row>
  <row>
    <index>1</index>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides/>
  </row>
  <row>
    <index>2</index>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""


def test_compression_output(parser, compression_only, geom_df):
    with tm.ensure_clean() as path:
        geom_df.to_xml(path, parser=parser, compression=compression_only)

        with get_handle(
            path,
            "r",
            compression=compression_only,
        ) as handle_obj:
            output = handle_obj.handle.read()

    output = equalize_decl(output)

    assert geom_xml == output.strip()


def test_filename_and_suffix_comp(
    parser, compression_only, geom_df, compression_to_extension
):
    compfile = "xml." + compression_to_extension[compression_only]
    with tm.ensure_clean(filename=compfile) as path:
        geom_df.to_xml(path, parser=parser, compression=compression_only)

        with get_handle(
            path,
            "r",
            compression=compression_only,
        ) as handle_obj:
            output = handle_obj.handle.read()

    output = equalize_decl(output)

    assert geom_xml == output.strip()


def test_ea_dtypes(any_numeric_ea_dtype, parser):
    # GH#43903
    expected = """<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <index>0</index>
    <a/>
  </row>
</data>"""
    df = DataFrame({"a": [NA]}).astype(any_numeric_ea_dtype)
    result = df.to_xml(parser=parser)
    assert equalize_decl(result).strip() == expected


def test_unsuported_compression(parser, geom_df):
    with pytest.raises(ValueError, match="Unrecognized compression type"):
        with tm.ensure_clean() as path:
            geom_df.to_xml(path, parser=parser, compression="7z")


# STORAGE OPTIONS


@pytest.mark.single_cpu
def test_s3_permission_output(parser, s3_public_bucket, geom_df):
    s3fs = pytest.importorskip("s3fs")
    pytest.importorskip("lxml")

    with tm.external_error_raised((PermissionError, FileNotFoundError)):
        fs = s3fs.S3FileSystem(anon=True)
        fs.ls(s3_public_bucket.name)

        geom_df.to_xml(
            f"s3://{s3_public_bucket.name}/geom.xml", compression="zip", parser=parser
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\distributions\wheel.py ===
from typing import TYPE_CHECKING, Optional

from pip._vendor.packaging.utils import canonicalize_name

from pip._internal.distributions.base import AbstractDistribution
from pip._internal.metadata import (
    BaseDistribution,
    FilesystemWheel,
    get_wheel_distribution,
)

if TYPE_CHECKING:
    from pip._internal.index.package_finder import PackageFinder


class WheelDistribution(AbstractDistribution):
    """Represents a wheel distribution.

    This does not need any preparation as wheels can be directly unpacked.
    """

    @property
    def build_tracker_id(self) -> Optional[str]:
        return None

    def get_metadata_distribution(self) -> BaseDistribution:
        """Loads the metadata from the wheel file into memory and returns a
        Distribution that uses it, not relying on the wheel file or
        requirement.
        """
        assert self.req.local_file_path, "Set as part of preparation during download"
        assert self.req.name, "Wheels are never unnamed"
        wheel = FilesystemWheel(self.req.local_file_path)
        return get_wheel_distribution(wheel, canonicalize_name(self.req.name))

    def prepare_distribution_metadata(
        self,
        finder: "PackageFinder",
        build_isolation: bool,
        check_build_deps: bool,
    ) -> None:
        pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\utils\retry.py ===
import functools
from time import perf_counter, sleep
from typing import Callable, TypeVar

from pip._vendor.typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


def retry(
    wait: float, stop_after_delay: float
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to automatically retry a function on error.

    If the function raises, the function is recalled with the same arguments
    until it returns or the time limit is reached. When the time limit is
    surpassed, the last exception raised is reraised.

    :param wait: The time to wait after an error before retrying, in seconds.
    :param stop_after_delay: The time limit after which retries will cease,
        in seconds.
    """

    def wrapper(func: Callable[P, T]) -> Callable[P, T]:

        @functools.wraps(func)
        def retry_wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            # The performance counter is monotonic on all platforms we care
            # about and has much better resolution than time.monotonic().
            start_time = perf_counter()
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if perf_counter() - start_time > stop_after_delay:
                        raise
                    sleep(wait)

        return retry_wrapped

    return wrapper

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\protocol.py ===
from typing import Any, cast, Set, TYPE_CHECKING
from inspect import isclass

if TYPE_CHECKING:
    from pip._vendor.rich.console import RenderableType

_GIBBERISH = """aihwerij235234ljsdnp34ksodfipwoe234234jlskjdf"""


def is_renderable(check_object: Any) -> bool:
    """Check if an object may be rendered by Rich."""
    return (
        isinstance(check_object, str)
        or hasattr(check_object, "__rich__")
        or hasattr(check_object, "__rich_console__")
    )


def rich_cast(renderable: object) -> "RenderableType":
    """Cast an object to a renderable by calling __rich__ if present.

    Args:
        renderable (object): A potentially renderable object

    Returns:
        object: The result of recursively calling __rich__.
    """
    from pip._vendor.rich.console import RenderableType

    rich_visited_set: Set[type] = set()  # Prevent potential infinite loop
    while hasattr(renderable, "__rich__") and not isclass(renderable):
        # Detect object which claim to have all the attributes
        if hasattr(renderable, _GIBBERISH):
            return repr(renderable)
        cast_method = getattr(renderable, "__rich__")
        renderable = cast_method()
        renderable_type = type(renderable)
        if renderable_type in rich_visited_set:
            break
        rich_visited_set.add(renderable_type)

    return cast(RenderableType, renderable)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\styled.py ===
from typing import TYPE_CHECKING

from .measure import Measurement
from .segment import Segment
from .style import StyleType

if TYPE_CHECKING:
    from .console import Console, ConsoleOptions, RenderResult, RenderableType


class Styled:
    """Apply a style to a renderable.

    Args:
        renderable (RenderableType): Any renderable.
        style (StyleType): A style to apply across the entire renderable.
    """

    def __init__(self, renderable: "RenderableType", style: "StyleType") -> None:
        self.renderable = renderable
        self.style = style

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        style = console.get_style(self.style)
        rendered_segments = console.render(self.renderable, options)
        segments = Segment.apply_style(rendered_segments, style)
        return segments

    def __rich_measure__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> Measurement:
        return Measurement.get(console, options, self.renderable)


if __name__ == "__main__":  # pragma: no cover
    from pip._vendor.rich import print
    from pip._vendor.rich.panel import Panel

    panel = Styled(Panel("hello"), "on blue")
    print(panel)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\chrome\remote_connection.py ===
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

from typing import Optional

from selenium.webdriver import DesiredCapabilities
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
from selenium.webdriver.remote.client_config import ClientConfig


class ChromeRemoteConnection(ChromiumRemoteConnection):
    browser_name = DesiredCapabilities.CHROME["browserName"]

    def __init__(
        self,
        remote_server_addr: str,
        keep_alive: bool = True,
        ignore_proxy: Optional[bool] = False,
        client_config: Optional[ClientConfig] = None,
    ) -> None:
        super().__init__(
            remote_server_addr=remote_server_addr,
            vendor_prefix="goog",
            browser_name=ChromeRemoteConnection.browser_name,
            keep_alive=keep_alive,
            ignore_proxy=ignore_proxy,
            client_config=client_config,
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\edge\remote_connection.py ===
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

from typing import Optional

from selenium.webdriver import DesiredCapabilities
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
from selenium.webdriver.remote.client_config import ClientConfig


class EdgeRemoteConnection(ChromiumRemoteConnection):
    browser_name = DesiredCapabilities.EDGE["browserName"]

    def __init__(
        self,
        remote_server_addr: str,
        keep_alive: bool = True,
        ignore_proxy: Optional[bool] = False,
        client_config: Optional[ClientConfig] = None,
    ) -> None:
        super().__init__(
            remote_server_addr=remote_server_addr,
            vendor_prefix="goog",
            browser_name=EdgeRemoteConnection.browser_name,
            keep_alive=keep_alive,
            ignore_proxy=ignore_proxy,
            client_config=client_config,
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\util\util.py ===
from __future__ import annotations

import typing
from types import TracebackType


def to_bytes(
    x: str | bytes, encoding: str | None = None, errors: str | None = None
) -> bytes:
    if isinstance(x, bytes):
        return x
    elif not isinstance(x, str):
        raise TypeError(f"not expecting type {type(x).__name__}")
    if encoding or errors:
        return x.encode(encoding or "utf-8", errors=errors or "strict")
    return x.encode()


def to_str(
    x: str | bytes, encoding: str | None = None, errors: str | None = None
) -> str:
    if isinstance(x, str):
        return x
    elif not isinstance(x, bytes):
        raise TypeError(f"not expecting type {type(x).__name__}")
    if encoding or errors:
        return x.decode(encoding or "utf-8", errors=errors or "strict")
    return x.decode()


def reraise(
    tp: type[BaseException] | None,
    value: BaseException,
    tb: TracebackType | None = None,
) -> typing.NoReturn:
    try:
        if value.__traceback__ is not tb:
            raise value.with_traceback(tb)
        raise value
    finally:
        value = None  # type: ignore[assignment]
        tb = None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\util\__init__.py ===
# For backwards compatibility, provide imports that used to be here.
from __future__ import annotations

from .connection import is_connection_dropped
from .request import SKIP_HEADER, SKIPPABLE_HEADERS, make_headers
from .response import is_fp_closed
from .retry import Retry
from .ssl_ import (
    ALPN_PROTOCOLS,
    IS_PYOPENSSL,
    SSLContext,
    assert_fingerprint,
    create_urllib3_context,
    resolve_cert_reqs,
    resolve_ssl_version,
    ssl_wrap_socket,
)
from .timeout import Timeout
from .url import Url, parse_url
from .wait import wait_for_read, wait_for_write

__all__ = (
    "IS_PYOPENSSL",
    "SSLContext",
    "ALPN_PROTOCOLS",
    "Retry",
    "Timeout",
    "Url",
    "assert_fingerprint",
    "create_urllib3_context",
    "is_connection_dropped",
    "is_fp_closed",
    "parse_url",
    "make_headers",
    "resolve_cert_reqs",
    "resolve_ssl_version",
    "ssl_wrap_socket",
    "wait_for_read",
    "wait_for_write",
    "SKIP_HEADER",
    "SKIPPABLE_HEADERS",
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\webdriver_manager\core\download_manager.py ===
import os
from abc import ABC

from webdriver_manager.core.file_manager import File
from webdriver_manager.core.http import WDMHttpClient
from webdriver_manager.core.logger import log


class DownloadManager(ABC):
    def __init__(self, http_client):
        self._http_client = http_client

    def download_file(self, url: str) -> File:
        raise NotImplementedError

    @property
    def http_client(self):
        return self._http_client


class WDMDownloadManager(DownloadManager):
    def __init__(self, http_client=None):
        if http_client is None:
            http_client = WDMHttpClient()
        super().__init__(http_client)

    def download_file(self, url: str) -> File:
        log(f"About to download new driver from {url}")
        response = self._http_client.get(url)
        log(f"Driver downloading response is {response.status_code}")
        file_name = self.extract_filename_from_url(url)
        return File(response, file_name)

    @staticmethod
    def extract_filename_from_url(url):
        # Split the URL by '/'
        url_parts = url.split('/')
        # Get the last part of the URL, which should be the filename
        filename = url_parts[-1]
        # Decode the URL-encoded filename
        filename = os.path.basename(filename)
        return filename

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\logic\tests\test_boolalg.py ===
from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.numbers import oo
from sympy.core.relational import Equality, Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.sets.sets import Interval, Union
from sympy.sets.contains import Contains
from sympy.simplify.simplify import simplify
from sympy.logic.boolalg import (
    And, Boolean, Equivalent, ITE, Implies, Nand, Nor, Not, Or,
    POSform, SOPform, Xor, Xnor, conjuncts, disjuncts,
    distribute_or_over_and, distribute_and_over_or,
    eliminate_implications, is_nnf, is_cnf, is_dnf, simplify_logic,
    to_nnf, to_cnf, to_dnf, to_int_repr, bool_map, true, false,
    BooleanAtom, is_literal, term_to_integer,
    truth_table, as_Boolean, to_anf, is_anf, distribute_xor_over_and,
    anf_coeffs, ANFform, bool_minterm, bool_maxterm, bool_monomial,
    _check_pair, _convert_to_varsSOP, _convert_to_varsPOS, Exclusive,
    gateinputcount)
from sympy.assumptions.cnf import CNF

from sympy.testing.pytest import raises, XFAIL, slow

from itertools import combinations, permutations, product

A, B, C, D = symbols('A:D')
a, b, c, d, e, w, x, y, z = symbols('a:e w:z')


def test_overloading():
    """Test that |, & are overloaded as expected"""

    assert A & B == And(A, B)
    assert A | B == Or(A, B)
    assert (A & B) | C == Or(And(A, B), C)
    assert A >> B == Implies(A, B)
    assert A << B == Implies(B, A)
    assert ~A == Not(A)
    assert A ^ B == Xor(A, B)


def test_And():
    assert And() is true
    assert And(A) == A
    assert And(True) is true
    assert And(False) is false
    assert And(True, True) is true
    assert And(True, False) is false
    assert And(False, False) is false
    assert And(True, A) == A
    assert And(False, A) is false
    assert And(True, True, True) is true
    assert And(True, True, A) == A
    assert And(True, False, A) is false
    assert And(1, A) == A
    raises(TypeError, lambda: And(2, A))
    assert And(A < 1, A >= 1) is false
    e = A > 1
    assert And(e, e.canonical) == e.canonical
    g, l, ge, le = A > B, B < A, A >= B, B <= A
    assert And(g, l, ge, le) == And(ge, g)
    assert {And(*i) for i in permutations((l, g, le, ge))} == {And(ge, g)}
    assert And(And(Eq(a, 0), Eq(b, 0)), And(Ne(a, 0), Eq(c, 0))) is false


def test_Or():
    assert Or() is false
    assert Or(A) == A
    assert Or(True) is true
    assert Or(False) is false
    assert Or(True, True) is true
    assert Or(True, False) is true
    assert Or(False, False) is false
    assert Or(True, A) is true
    assert Or(False, A) == A
    assert Or(True, False, False) is true
    assert Or(True, False, A) is true
    assert Or(False, False, A) == A
    assert Or(1, A) is true
    raises(TypeError, lambda: Or(2, A))
    assert Or(A < 1, A >= 1) is true
    e = A > 1
    assert Or(e, e.canonical) == e
    g, l, ge, le = A > B, B < A, A >= B, B <= A
    assert Or(g, l, ge, le) == Or(g, ge)


def test_Xor():
    assert Xor() is false
    assert Xor(A) == A
    assert Xor(A, A) is false
    assert Xor(True, A, A) is true
    assert Xor(A, A, A, A, A) == A
    assert Xor(True, False, False, A, B) == ~Xor(A, B)
    assert Xor(True) is true
    assert Xor(False) is false
    assert Xor(True, True) is false
    assert Xor(True, False) is true
    assert Xor(False, False) is false
    assert Xor(True, A) == ~A
    assert Xor(False, A) == A
    assert Xor(True, False, False) is true
    assert Xor(True, False, A) == ~A
    assert Xor(False, False, A) == A
    assert isinstance(Xor(A, B), Xor)
    assert Xor(A, B, Xor(C, D)) == Xor(A, B, C, D)
    assert Xor(A, B, Xor(B, C)) == Xor(A, C)
    assert Xor(A < 1, A >= 1, B) == Xor(0, 1, B) == Xor(1, 0, B)
    e = A > 1
    assert Xor(e, e.canonical) == Xor(0, 0) == Xor(1, 1)


def test_rewrite_as_And():
    expr = x ^ y
    assert expr.rewrite(And) == (x | y) & (~x | ~y)


def test_rewrite_as_Or():
    expr = x ^ y
    assert expr.rewrite(Or) == (x & ~y) | (y & ~x)


def test_rewrite_as_Nand():
    expr = (y & z) | (z & ~w)
    assert expr.rewrite(Nand) == ~(~(y & z) & ~(z & ~w))


def test_rewrite_as_Nor():
    expr = z & (y | ~w)
    assert expr.rewrite(Nor) == ~(~z | ~(y | ~w))


def test_Not():
    raises(TypeError, lambda: Not(True, False))
    assert Not(True) is false
    assert Not(False) is true
    assert Not(0) is true
    assert Not(1) is false
    assert Not(2) is false


def test_Nand():
    assert Nand() is false
    assert Nand(A) == ~A
    assert Nand(True) is false
    assert Nand(False) is true
    assert Nand(True, True) is false
    assert Nand(True, False) is true
    assert Nand(False, False) is true
    assert Nand(True, A) == ~A
    assert Nand(False, A) is true
    assert Nand(True, True, True) is false
    assert Nand(True, True, A) == ~A
    assert Nand(True, False, A) is true


def test_Nor():
    assert Nor() is true
    assert Nor(A) == ~A
    assert Nor(True) is false
    assert Nor(False) is true
    assert Nor(True, True) is false
    assert Nor(True, False) is false
    assert Nor(False, False) is true
    assert Nor(True, A) is false
    assert Nor(False, A) == ~A
    assert Nor(True, True, True) is false
    assert Nor(True, True, A) is false
    assert Nor(True, False, A) is false


def test_Xnor():
    assert Xnor() is true
    assert Xnor(A) == ~A
    assert Xnor(A, A) is true
    assert Xnor(True, A, A) is false
    assert Xnor(A, A, A, A, A) == ~A
    assert Xnor(True) is false
    assert Xnor(False) is true
    assert Xnor(True, True) is true
    assert Xnor(True, False) is false
    assert Xnor(False, False) is true
    assert Xnor(True, A) == A
    assert Xnor(False, A) == ~A
    assert Xnor(True, False, False) is false
    assert Xnor(True, False, A) == A
    assert Xnor(False, False, A) == ~A


def test_Implies():
    raises(ValueError, lambda: Implies(A, B, C))
    assert Implies(True, True) is true
    assert Implies(True, False) is false
    assert Implies(False, True) is true
    assert Implies(False, False) is true
    assert Implies(0, A) is true
    assert Implies(1, 1) is true
    assert Implies(1, 0) is false
    assert A >> B == B << A
    assert (A < 1) >> (A >= 1) == (A >= 1)
    assert (A < 1) >> (S.One > A) is true
    assert A >> A is true


def test_Equivalent():
    assert Equivalent(A, B) == Equivalent(B, A) == Equivalent(A, B, A)
    assert Equivalent() is true
    assert Equivalent(A, A) == Equivalent(A) is true
    assert Equivalent(True, True) == Equivalent(False, False) is true
    assert Equivalent(True, False) == Equivalent(False, True) is false
    assert Equivalent(A, True) == A
    assert Equivalent(A, False) == Not(A)
    assert Equivalent(A, B, True) == A & B
    assert Equivalent(A, B, False) == ~A & ~B
    assert Equivalent(1, A) == A
    assert Equivalent(0, A) == Not(A)
    assert Equivalent(A, Equivalent(B, C)) != Equivalent(Equivalent(A, B), C)
    assert Equivalent(A < 1, A >= 1) is false
    assert Equivalent(A < 1, A >= 1, 0) is false
    assert Equivalent(A < 1, A >= 1, 1) is false
    assert Equivalent(A < 1, S.One > A) == Equivalent(1, 1) == Equivalent(0, 0)
    assert Equivalent(Equality(A, B), Equality(B, A)) is true


def test_Exclusive():
    assert Exclusive(False, False, False) is true
    assert Exclusive(True, False, False) is true
    assert Exclusive(True, True, False) is false
    assert Exclusive(True, True, True) is false


def test_equals():
    assert Not(Or(A, B)).equals(And(Not(A), Not(B))) is True
    assert Equivalent(A, B).equals((A >> B) & (B >> A)) is True
    assert ((A | ~B) & (~A | B)).equals((~A & ~B) | (A & B)) is True
    assert (A >> B).equals(~A >> ~B) is False
    assert (A >> (B >> A)).equals(A >> (C >> A)) is False
    raises(NotImplementedError, lambda: (A & B).equals(A > B))


def test_simplification_boolalg():
    """
    Test working of simplification methods.
    """
    set1 = [[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]]
    set2 = [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]]
    assert SOPform([x, y, z], set1) == Or(And(Not(x), z), And(Not(z), x))
    assert Not(SOPform([x, y, z], set2)) == \
           Not(Or(And(Not(x), Not(z)), And(x, z)))
    assert POSform([x, y, z], set1 + set2) is true
    assert SOPform([x, y, z], set1 + set2) is true
    assert SOPform([Dummy(), Dummy(), Dummy()], set1 + set2) is true

    minterms = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1],
                [1, 1, 1, 1]]
    dontcares = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]]
    assert (
        SOPform([w, x, y, z], minterms, dontcares) ==
        Or(And(y, z), And(Not(w), Not(x))))
    assert POSform([w, x, y, z], minterms, dontcares) == And(Or(Not(w), y), z)

    minterms = [1, 3, 7, 11, 15]
    dontcares = [0, 2, 5]
    assert (
        SOPform([w, x, y, z], minterms, dontcares) ==
        Or(And(y, z), And(Not(w), Not(x))))
    assert POSform([w, x, y, z], minterms, dontcares) == And(Or(Not(w), y), z)

    minterms = [1, [0, 0, 1, 1], 7, [1, 0, 1, 1],
                [1, 1, 1, 1]]
    dontcares = [0, [0, 0, 1, 0], 5]
    assert (
        SOPform([w, x, y, z], minterms, dontcares) ==
        Or(And(y, z), And(Not(w), Not(x))))
    assert POSform([w, x, y, z], minterms, dontcares) == And(Or(Not(w), y), z)

    minterms = [1, {y: 1, z: 1}]
    dontcares = [0, [0, 0, 1, 0], 5]
    assert (
        SOPform([w, x, y, z], minterms, dontcares) ==
        Or(And(y, z), And(Not(w), Not(x))))
    assert POSform([w, x, y, z], minterms, dontcares) == And(Or(Not(w), y), z)

    minterms = [{y: 1, z: 1}, 1]
    dontcares = [[0, 0, 0, 0]]

    minterms = [[0, 0, 0]]
    raises(ValueError, lambda: SOPform([w, x, y, z], minterms))
    raises(ValueError, lambda: POSform([w, x, y, z], minterms))

    raises(TypeError, lambda: POSform([w, x, y, z], ["abcdefg"]))

    # test simplification
    ans = And(A, Or(B, C))
    assert simplify_logic(A & (B | C)) == ans
    assert simplify_logic((A & B) | (A & C)) == ans
    assert simplify_logic(Implies(A, B)) == Or(Not(A), B)
    assert simplify_logic(Equivalent(A, B)) == \
           Or(And(A, B), And(Not(A), Not(B)))
    assert simplify_logic(And(Equality(A, 2), C)) == And(Equality(A, 2), C)
    assert simplify_logic(And(Equality(A, 2), A)) == And(Equality(A, 2), A)
    assert simplify_logic(And(Equality(A, B), C)) == And(Equality(A, B), C)
    assert simplify_logic(Or(And(Equality(A, 3), B), And(Equality(A, 3), C))) \
           == And(Equality(A, 3), Or(B, C))
    b = (~x & ~y & ~z) | (~x & ~y & z)
    e = And(A, b)
    assert simplify_logic(e) == A & ~x & ~y
    raises(ValueError, lambda: simplify_logic(A & (B | C), form='blabla'))
    assert simplify(Or(x <= y, And(x < y, z))) == (x <= y)
    assert simplify(Or(x <= y, And(y > x, z))) == (x <= y)
    assert simplify(Or(x >= y, And(y < x, z))) == (x >= y)

    # Check that expressions with nine variables or more are not simplified
    # (without the force-flag)
    a, b, c, d, e, f, g, h, j = symbols('a b c d e f g h j')
    expr = a & b & c & d & e & f & g & h & j | \
           a & b & c & d & e & f & g & h & ~j
    # This expression can be simplified to get rid of the j variables
    assert simplify_logic(expr) == expr

    # Test dontcare
    assert simplify_logic((a & b) | c | d, dontcare=(a & b)) == c | d

    # check input
    ans = SOPform([x, y], [[1, 0]])
    assert SOPform([x, y], [[1, 0]]) == ans
    assert POSform([x, y], [[1, 0]]) == ans

    raises(ValueError, lambda: SOPform([x], [[1]], [[1]]))
    assert SOPform([x], [[1]], [[0]]) is true
    assert SOPform([x], [[0]], [[1]]) is true
    assert SOPform([x], [], []) is false

    raises(ValueError, lambda: POSform([x], [[1]], [[1]]))
    assert POSform([x], [[1]], [[0]]) is true
    assert POSform([x], [[0]], [[1]]) is true
    assert POSform([x], [], []) is false

    # check working of simplify
    assert simplify((A & B) | (A & C)) == And(A, Or(B, C))
    assert simplify(And(x, Not(x))) == False
    assert simplify(Or(x, Not(x))) == True
    assert simplify(And(Eq(x, 0), Eq(x, y))) == And(Eq(x, 0), Eq(y, 0))
    assert And(Eq(x - 1, 0), Eq(x, y)).simplify() == And(Eq(x, 1), Eq(y, 1))
    assert And(Ne(x - 1, 0), Ne(x, y)).simplify() == And(Ne(x, 1), Ne(x, y))
    assert And(Eq(x - 1, 0), Ne(x, y)).simplify() == And(Eq(x, 1), Ne(y, 1))
    assert And(Eq(x - 1, 0), Eq(x, z + y), Eq(y + x, 0)).simplify(
    ) == And(Eq(x, 1), Eq(y, -1), Eq(z, 2))
    assert And(Eq(x - 1, 0), Eq(x + 2, 3)).simplify() == Eq(x, 1)
    assert And(Ne(x - 1, 0), Ne(x + 2, 3)).simplify() == Ne(x, 1)
    assert And(Eq(x - 1, 0), Eq(x + 2, 2)).simplify() == False
    assert And(Ne(x - 1, 0), Ne(x + 2, 2)).simplify(
    ) == And(Ne(x, 1), Ne(x, 0))
    assert simplify(Xor(x, ~x)) == True


def test_bool_map():
    """
    Test working of bool_map function.
    """

    minterms = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1],
                [1, 1, 1, 1]]
    assert bool_map(Not(Not(a)), a) == (a, {a: a})
    assert bool_map(SOPform([w, x, y, z], minterms),
                    POSform([w, x, y, z], minterms)) == \
           (And(Or(Not(w), y), Or(Not(x), y), z), {x: x, w: w, z: z, y: y})
    assert bool_map(SOPform([x, z, y], [[1, 0, 1]]),
                    SOPform([a, b, c], [[1, 0, 1]])) != False
    function1 = SOPform([x, z, y], [[1, 0, 1], [0, 0, 1]])
    function2 = SOPform([a, b, c], [[1, 0, 1], [1, 0, 0]])
    assert bool_map(function1, function2) == \
           (function1, {y: a, z: b})
    assert bool_map(Xor(x, y), ~Xor(x, y)) == False
    assert bool_map(And(x, y), Or(x, y)) is None
    assert bool_map(And(x, y), And(x, y, z)) is None
    # issue 16179
    assert bool_map(Xor(x, y, z), ~Xor(x, y, z)) == False
    assert bool_map(Xor(a, x, y, z), ~Xor(a, x, y, z)) == False


def test_bool_symbol():
    """Test that mixing symbols with boolean values
    works as expected"""

    assert And(A, True) == A
    assert And(A, True, True) == A
    assert And(A, False) is false
    assert And(A, True, False) is false
    assert Or(A, True) is true
    assert Or(A, False) == A


def test_is_boolean():
    assert isinstance(True, Boolean) is False
    assert isinstance(true, Boolean) is True
    assert 1 == True
    assert 1 != true
    assert (1 == true) is False
    assert 0 == False
    assert 0 != false
    assert (0 == false) is False
    assert true.is_Boolean is True
    assert (A & B).is_Boolean
    assert (A | B).is_Boolean
    assert (~A).is_Boolean
    assert (A ^ B).is_Boolean
    assert A.is_Boolean != isinstance(A, Boolean)
    assert isinstance(A, Boolean)


def test_subs():
    assert (A & B).subs(A, True) == B
    assert (A & B).subs(A, False) is false
    assert (A & B).subs(B, True) == A
    assert (A & B).subs(B, False) is false
    assert (A & B).subs({A: True, B: True}) is true
    assert (A | B).subs(A, True) is true
    assert (A | B).subs(A, False) == B
    assert (A | B).subs(B, True) is true
    assert (A | B).subs(B, False) == A
    assert (A | B).subs({A: True, B: True}) is true


"""
we test for axioms of boolean algebra
see https://en.wikipedia.org/wiki/Boolean_algebra_(structure)
"""


def test_commutative():
    """Test for commutativity of And and Or"""
    A, B = map(Boolean, symbols('A,B'))

    assert A & B == B & A
    assert A | B == B | A


def test_and_associativity():
    """Test for associativity of And"""

    assert (A & B) & C == A & (B & C)


def test_or_assicativity():
    assert ((A | B) | C) == (A | (B | C))


def test_double_negation():
    a = Boolean()
    assert ~(~a) == a


# test methods

def test_eliminate_implications():
    assert eliminate_implications(Implies(A, B, evaluate=False)) == (~A) | B
    assert eliminate_implications(
        A >> (C >> Not(B))) == Or(Or(Not(B), Not(C)), Not(A))
    assert eliminate_implications(Equivalent(A, B, C, D)) == \
           (~A | B) & (~B | C) & (~C | D) & (~D | A)


def test_conjuncts():
    assert conjuncts(A & B & C) == {A, B, C}
    assert conjuncts((A | B) & C) == {A | B, C}
    assert conjuncts(A) == {A}
    assert conjuncts(True) == {True}
    assert conjuncts(False) == {False}


def test_disjuncts():
    assert disjuncts(A | B | C) == {A, B, C}
    assert disjuncts((A | B) & C) == {(A | B) & C}
    assert disjuncts(A) == {A}
    assert disjuncts(True) == {True}
    assert disjuncts(False) == {False}


def test_distribute():
    assert distribute_and_over_or(Or(And(A, B), C)) == And(Or(A, C), Or(B, C))
    assert distribute_or_over_and(And(A, Or(B, C))) == Or(And(A, B), And(A, C))
    assert distribute_xor_over_and(And(A, Xor(B, C))) == Xor(And(A, B), And(A, C))


def test_to_anf():
    x, y, z = symbols('x,y,z')
    assert to_anf(And(x, y)) == And(x, y)
    assert to_anf(Or(x, y)) == Xor(x, y, And(x, y))
    assert to_anf(Or(Implies(x, y), And(x, y), y)) == \
           Xor(x, True, x & y, remove_true=False)
    assert to_anf(Or(Nand(x, y), Nor(x, y), Xnor(x, y), Implies(x, y))) == True
    assert to_anf(Or(x, Not(y), Nor(x, z), And(x, y), Nand(y, z))) == \
           Xor(True, And(y, z), And(x, y, z), remove_true=False)
    assert to_anf(Xor(x, y)) == Xor(x, y)
    assert to_anf(Not(x)) == Xor(x, True, remove_true=False)
    assert to_anf(Nand(x, y)) == Xor(True, And(x, y), remove_true=False)
    assert to_anf(Nor(x, y)) == Xor(x, y, True, And(x, y), remove_true=False)
    assert to_anf(Implies(x, y)) == Xor(x, True, And(x, y), remove_true=False)
    assert to_anf(Equivalent(x, y)) == Xor(x, y, True, remove_true=False)
    assert to_anf(Nand(x | y, x >> y), deep=False) == \
           Xor(True, And(Or(x, y), Implies(x, y)), remove_true=False)
    assert to_anf(Nor(x ^ y, x & y), deep=False) == \
           Xor(True, Or(Xor(x, y), And(x, y)), remove_true=False)
    # issue 25218
    assert to_anf(x ^ ~(x ^ y ^ ~y)) == False


def test_to_nnf():
    assert to_nnf(true) is true
    assert to_nnf(false) is false
    assert to_nnf(A) == A
    assert to_nnf(A | ~A | B) is true
    assert to_nnf(A & ~A & B) is false
    assert to_nnf(A >> B) == ~A | B
    assert to_nnf(Equivalent(A, B, C)) == (~A | B) & (~B | C) & (~C | A)
    assert to_nnf(A ^ B ^ C) == \
           (A | B | C) & (~A | ~B | C) & (A | ~B | ~C) & (~A | B | ~C)
    assert to_nnf(ITE(A, B, C)) == (~A | B) & (A | C)
    assert to_nnf(Not(A | B | C)) == ~A & ~B & ~C
    assert to_nnf(Not(A & B & C)) == ~A | ~B | ~C
    assert to_nnf(Not(A >> B)) == A & ~B
    assert to_nnf(Not(Equivalent(A, B, C))) == And(Or(A, B, C), Or(~A, ~B, ~C))
    assert to_nnf(Not(A ^ B ^ C)) == \
           (~A | B | C) & (A | ~B | C) & (A | B | ~C) & (~A | ~B | ~C)
    assert to_nnf(Not(ITE(A, B, C))) == (~A | ~B) & (A | ~C)
    assert to_nnf((A >> B) ^ (B >> A)) == (A & ~B) | (~A & B)
    assert to_nnf((A >> B) ^ (B >> A), False) == \
           (~A | ~B | A | B) & ((A & ~B) | (~A & B))
    assert ITE(A, 1, 0).to_nnf() == A
    assert ITE(A, 0, 1).to_nnf() == ~A
    # although ITE can hold non-Boolean, it will complain if
    # an attempt is made to convert the ITE to Boolean nnf
    raises(TypeError, lambda: ITE(A < 1, [1], B).to_nnf())


def test_to_cnf():
    assert to_cnf(~(B | C)) == And(Not(B), Not(C))
    assert to_cnf((A & B) | C) == And(Or(A, C), Or(B, C))
    assert to_cnf(A >> B) == (~A) | B
    assert to_cnf(A >> (B & C)) == (~A | B) & (~A | C)
    assert to_cnf(A & (B | C) | ~A & (B | C), True) == B | C
    assert to_cnf(A & B) == And(A, B)

    assert to_cnf(Equivalent(A, B)) == And(Or(A, Not(B)), Or(B, Not(A)))
    assert to_cnf(Equivalent(A, B & C)) == \
           (~A | B) & (~A | C) & (~B | ~C | A)
    assert to_cnf(Equivalent(A, B | C), True) == \
           And(Or(Not(B), A), Or(Not(C), A), Or(B, C, Not(A)))
    assert to_cnf(A + 1) == A + 1


def test_issue_18904():
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 = symbols('x1:16')
    eq = ((x1 & x2 & x3 & x4 & x5 & x6 & x7 & x8 & x9) |
          (x1 & x2 & x3 & x4 & x5 & x6 & x7 & x10 & x9) |
          (x1 & x11 & x3 & x12 & x5 & x13 & x14 & x15 & x9))
    assert is_cnf(to_cnf(eq))
    raises(ValueError, lambda: to_cnf(eq, simplify=True))
    for f, t in zip((And, Or), (to_cnf, to_dnf)):
        eq = f(x1, x2, x3, x4, x5, x6, x7, x8, x9)
        raises(ValueError, lambda: to_cnf(eq, simplify=True))
        assert t(eq, simplify=True, force=True) == eq


def test_issue_9949():
    assert is_cnf(to_cnf((b > -5) | (a > 2) & (a < 4)))


def test_to_CNF():
    assert CNF.CNF_to_cnf(CNF.to_CNF(~(B | C))) == to_cnf(~(B | C))
    assert CNF.CNF_to_cnf(CNF.to_CNF((A & B) | C)) == to_cnf((A & B) | C)
    assert CNF.CNF_to_cnf(CNF.to_CNF(A >> B)) == to_cnf(A >> B)
    assert CNF.CNF_to_cnf(CNF.to_CNF(A >> (B & C))) == to_cnf(A >> (B & C))
    assert CNF.CNF_to_cnf(CNF.to_CNF(A & (B | C) | ~A & (B | C))) == to_cnf(A & (B | C) | ~A & (B | C))
    assert CNF.CNF_to_cnf(CNF.to_CNF(A & B)) == to_cnf(A & B)


def test_to_dnf():
    assert to_dnf(~(B | C)) == And(Not(B), Not(C))
    assert to_dnf(A & (B | C)) == Or(And(A, B), And(A, C))
    assert to_dnf(A >> B) == (~A) | B
    assert to_dnf(A >> (B & C)) == (~A) | (B & C)
    assert to_dnf(A | B) == A | B

    assert to_dnf(Equivalent(A, B), True) == \
           Or(And(A, B), And(Not(A), Not(B)))
    assert to_dnf(Equivalent(A, B & C), True) == \
           Or(And(A, B, C), And(Not(A), Not(B)), And(Not(A), Not(C)))
    assert to_dnf(A + 1) == A + 1


def test_to_int_repr():
    x, y, z = map(Boolean, symbols('x,y,z'))

    def sorted_recursive(arg):
        try:
            return sorted(sorted_recursive(x) for x in arg)
        except TypeError:  # arg is not a sequence
            return arg

    assert sorted_recursive(to_int_repr([x | y, z | x], [x, y, z])) == \
           sorted_recursive([[1, 2], [1, 3]])
    assert sorted_recursive(to_int_repr([x | y, z | ~x], [x, y, z])) == \
           sorted_recursive([[1, 2], [3, -1]])


def test_is_anf():
    x, y = symbols('x,y')
    assert is_anf(true) is True
    assert is_anf(false) is True
    assert is_anf(x) is True
    assert is_anf(And(x, y)) is True
    assert is_anf(Xor(x, y, And(x, y))) is True
    assert is_anf(Xor(x, y, Or(x, y))) is False
    assert is_anf(Xor(Not(x), y)) is False


def test_is_nnf():
    assert is_nnf(true) is True
    assert is_nnf(A) is True
    assert is_nnf(~A) is True
    assert is_nnf(A & B) is True
    assert is_nnf((A & B) | (~A & A) | (~B & B) | (~A & ~B), False) is True
    assert is_nnf((A | B) & (~A | ~B)) is True
    assert is_nnf(Not(Or(A, B))) is False
    assert is_nnf(A ^ B) is False
    assert is_nnf((A & B) | (~A & A) | (~B & B) | (~A & ~B), True) is False


def test_is_cnf():
    assert is_cnf(x) is True
    assert is_cnf(x | y | z) is True
    assert is_cnf(x & y & z) is True
    assert is_cnf((x | y) & z) is True
    assert is_cnf((x & y) | z) is False
    assert is_cnf(~(x & y) | z) is False


def test_is_dnf():
    assert is_dnf(x) is True
    assert is_dnf(x | y | z) is True
    assert is_dnf(x & y & z) is True
    assert is_dnf((x & y) | z) is True
    assert is_dnf((x | y) & z) is False
    assert is_dnf(~(x | y) & z) is False


def test_ITE():
    A, B, C = symbols('A:C')
    assert ITE(True, False, True) is false
    assert ITE(True, True, False) is true
    assert ITE(False, True, False) is false
    assert ITE(False, False, True) is true
    assert isinstance(ITE(A, B, C), ITE)

    A = True
    assert ITE(A, B, C) == B
    A = False
    assert ITE(A, B, C) == C
    B = True
    assert ITE(And(A, B), B, C) == C
    assert ITE(Or(A, False), And(B, True), False) is false
    assert ITE(x, A, B) == Not(x)
    assert ITE(x, B, A) == x
    assert ITE(1, x, y) == x
    assert ITE(0, x, y) == y
    raises(TypeError, lambda: ITE(2, x, y))
    raises(TypeError, lambda: ITE(1, [], y))
    raises(TypeError, lambda: ITE(1, (), y))
    raises(TypeError, lambda: ITE(1, y, []))
    assert ITE(1, 1, 1) is S.true
    assert isinstance(ITE(1, 1, 1, evaluate=False), ITE)

    assert ITE(Eq(x, True), y, x) == ITE(x, y, x)
    assert ITE(Eq(x, False), y, x) == ITE(~x, y, x)
    assert ITE(Ne(x, True), y, x) == ITE(~x, y, x)
    assert ITE(Ne(x, False), y, x) == ITE(x, y, x)
    assert ITE(Eq(S.true, x), y, x) == ITE(x, y, x)
    assert ITE(Eq(S.false, x), y, x) == ITE(~x, y, x)
    assert ITE(Ne(S.true, x), y, x) == ITE(~x, y, x)
    assert ITE(Ne(S.false, x), y, x) == ITE(x, y, x)
    # 0 and 1 in the context are not treated as True/False
    # so the equality must always be False since dissimilar
    # objects cannot be equal
    assert ITE(Eq(x, 0), y, x) == x
    assert ITE(Eq(x, 1), y, x) == x
    assert ITE(Ne(x, 0), y, x) == y
    assert ITE(Ne(x, 1), y, x) == y
    assert ITE(Eq(x, 0), y, z).subs(x, 0) == y
    assert ITE(Eq(x, 0), y, z).subs(x, 1) == z
    raises(ValueError, lambda: ITE(x > 1, y, x, z))


def test_is_literal():
    assert is_literal(True) is True
    assert is_literal(False) is True
    assert is_literal(A) is True
    assert is_literal(~A) is True
    assert is_literal(Or(A, B)) is False
    assert is_literal(Q.zero(A)) is True
    assert is_literal(Not(Q.zero(A))) is True
    assert is_literal(Or(A, B)) is False
    assert is_literal(And(Q.zero(A), Q.zero(B))) is False
    assert is_literal(x < 3)
    assert not is_literal(x + y < 3)


def test_operators():
    # Mostly test __and__, __rand__, and so on
    assert True & A == A & True == A
    assert False & A == A & False == False
    assert A & B == And(A, B)
    assert True | A == A | True == True
    assert False | A == A | False == A
    assert A | B == Or(A, B)
    assert ~A == Not(A)
    assert True >> A == A << True == A
    assert False >> A == A << False == True
    assert A >> True == True << A == True
    assert A >> False == False << A == ~A
    assert A >> B == B << A == Implies(A, B)
    assert True ^ A == A ^ True == ~A
    assert False ^ A == A ^ False == A
    assert A ^ B == Xor(A, B)


def test_true_false():
    assert true is S.true
    assert false is S.false
    assert true is not True
    assert false is not False
    assert true
    assert not false
    assert true == True
    assert false == False
    assert not (true == False)
    assert not (false == True)
    assert not (true == false)

    assert hash(true) == hash(True)
    assert hash(false) == hash(False)
    assert len({true, True}) == len({false, False}) == 1

    assert isinstance(true, BooleanAtom)
    assert isinstance(false, BooleanAtom)
    # We don't want to subclass from bool, because bool subclasses from
    # int. But operators like &, |, ^, <<, >>, and ~ act differently on 0 and
    # 1 then we want them to on true and false.  See the docstrings of the
    # various And, Or, etc. functions for examples.
    assert not isinstance(true, bool)
    assert not isinstance(false, bool)

    # Note: using 'is' comparison is important here. We want these to return
    # true and false, not True and False

    assert Not(true) is false
    assert Not(True) is false
    assert Not(false) is true
    assert Not(False) is true
    assert ~true is false
    assert ~false is true

    for T, F in product((True, true), (False, false)):
        assert And(T, F) is false
        assert And(F, T) is false
        assert And(F, F) is false
        assert And(T, T) is true
        assert And(T, x) == x
        assert And(F, x) is false
        if not (T is True and F is False):
            assert T & F is false
            assert F & T is false
        if F is not False:
            assert F & F is false
        if T is not True:
            assert T & T is true

        assert Or(T, F) is true
        assert Or(F, T) is true
        assert Or(F, F) is false
        assert Or(T, T) is true
        assert Or(T, x) is true
        assert Or(F, x) == x
        if not (T is True and F is False):
            assert T | F is true
            assert F | T is true
        if F is not False:
            assert F | F is false
        if T is not True:
            assert T | T is true

        assert Xor(T, F) is true
        assert Xor(F, T) is true
        assert Xor(F, F) is false
        assert Xor(T, T) is false
        assert Xor(T, x) == ~x
        assert Xor(F, x) == x
        if not (T is True and F is False):
            assert T ^ F is true
            assert F ^ T is true
        if F is not False:
            assert F ^ F is false
        if T is not True:
            assert T ^ T is false

        assert Nand(T, F) is true
        assert Nand(F, T) is true
        assert Nand(F, F) is true
        assert Nand(T, T) is false
        assert Nand(T, x) == ~x
        assert Nand(F, x) is true

        assert Nor(T, F) is false
        assert Nor(F, T) is false
        assert Nor(F, F) is true
        assert Nor(T, T) is false
        assert Nor(T, x) is false
        assert Nor(F, x) == ~x

        assert Implies(T, F) is false
        assert Implies(F, T) is true
        assert Implies(F, F) is true
        assert Implies(T, T) is true
        assert Implies(T, x) == x
        assert Implies(F, x) is true
        assert Implies(x, T) is true
        assert Implies(x, F) == ~x
        if not (T is True and F is False):
            assert T >> F is false
            assert F << T is false
            assert F >> T is true
            assert T << F is true
        if F is not False:
            assert F >> F is true
            assert F << F is true
        if T is not True:
            assert T >> T is true
            assert T << T is true

        assert Equivalent(T, F) is false
        assert Equivalent(F, T) is false
        assert Equivalent(F, F) is true
        assert Equivalent(T, T) is true
        assert Equivalent(T, x) == x
        assert Equivalent(F, x) == ~x
        assert Equivalent(x, T) == x
        assert Equivalent(x, F) == ~x

        assert ITE(T, T, T) is true
        assert ITE(T, T, F) is true
        assert ITE(T, F, T) is false
        assert ITE(T, F, F) is false
        assert ITE(F, T, T) is true
        assert ITE(F, T, F) is false
        assert ITE(F, F, T) is true
        assert ITE(F, F, F) is false

    assert all(i.simplify(1, 2) is i for i in (S.true, S.false))


def test_bool_as_set():
    assert ITE(y <= 0, False, y >= 1).as_set() == Interval(1, oo)
    assert And(x <= 2, x >= -2).as_set() == Interval(-2, 2)
    assert Or(x >= 2, x <= -2).as_set() == Interval(-oo, -2) + Interval(2, oo)
    assert Not(x > 2).as_set() == Interval(-oo, 2)
    # issue 10240
    assert Not(And(x > 2, x < 3)).as_set() == \
           Union(Interval(-oo, 2), Interval(3, oo))
    assert true.as_set() == S.UniversalSet
    assert false.as_set() is S.EmptySet
    assert x.as_set() == S.UniversalSet
    assert And(Or(x < 1, x > 3), x < 2).as_set() == Interval.open(-oo, 1)
    assert And(x < 1, sin(x) < 3).as_set() == (x < 1).as_set()
    raises(NotImplementedError, lambda: (sin(x) < 1).as_set())
    # watch for object morph in as_set
    assert Eq(-1, cos(2 * x) ** 2 / sin(2 * x) ** 2).as_set() is S.EmptySet


@XFAIL
def test_multivariate_bool_as_set():
    x, y = symbols('x,y')

    assert And(x >= 0, y >= 0).as_set() == Interval(0, oo) * Interval(0, oo)
    assert Or(x >= 0, y >= 0).as_set() == S.Reals * S.Reals - \
           Interval(-oo, 0, True, True) * Interval(-oo, 0, True, True)


def test_all_or_nothing():
    x = symbols('x', extended_real=True)
    args = x >= -oo, x <= oo
    v = And(*args)
    if v.func is And:
        assert len(v.args) == len(args) - args.count(S.true)
    else:
        assert v == True
    v = Or(*args)
    if v.func is Or:
        assert len(v.args) == 2
    else:
        assert v == True


def test_canonical_atoms():
    assert true.canonical == true
    assert false.canonical == false


def test_negated_atoms():
    assert true.negated == false
    assert false.negated == true


def test_issue_8777():
    assert And(x > 2, x < oo).as_set() == Interval(2, oo, left_open=True)
    assert And(x >= 1, x < oo).as_set() == Interval(1, oo)
    assert (x < oo).as_set() == Interval(-oo, oo)
    assert (x > -oo).as_set() == Interval(-oo, oo)


def test_issue_8975():
    assert Or(And(-oo < x, x <= -2), And(2 <= x, x < oo)).as_set() == \
           Interval(-oo, -2) + Interval(2, oo)


def test_term_to_integer():
    assert term_to_integer([1, 0, 1, 0, 0, 1, 0]) == 82
    assert term_to_integer('0010101000111001') == 10809


def test_issue_21971():
    a, b, c, d = symbols('a b c d')
    f = a & b & c | a & c
    assert f.subs(a & c, d) == b & d | d
    assert f.subs(a & b & c, d) == a & c | d

    f = (a | b | c) & (a | c)
    assert f.subs(a | c, d) == (b | d) & d
    assert f.subs(a | b | c, d) == (a | c) & d

    f = (a ^ b ^ c) & (a ^ c)
    assert f.subs(a ^ c, d) == (b ^ d) & d
    assert f.subs(a ^ b ^ c, d) == (a ^ c) & d


def test_truth_table():
    assert list(truth_table(And(x, y), [x, y], input=False)) == \
           [False, False, False, True]
    assert list(truth_table(x | y, [x, y], input=False)) == \
           [False, True, True, True]
    assert list(truth_table(x >> y, [x, y], input=False)) == \
           [True, True, False, True]
    assert list(truth_table(And(x, y), [x, y])) == \
           [([0, 0], False), ([0, 1], False), ([1, 0], False), ([1, 1], True)]


def test_issue_8571():
    for t in (S.true, S.false):
        raises(TypeError, lambda: +t)
        raises(TypeError, lambda: -t)
        raises(TypeError, lambda: abs(t))
        # use int(bool(t)) to get 0 or 1
        raises(TypeError, lambda: int(t))

        for o in [S.Zero, S.One, x]:
            for _ in range(2):
                raises(TypeError, lambda: o + t)
                raises(TypeError, lambda: o - t)
                raises(TypeError, lambda: o % t)
                raises(TypeError, lambda: o * t)
                raises(TypeError, lambda: o / t)
                raises(TypeError, lambda: o ** t)
                o, t = t, o  # do again in reversed order


def test_expand_relational():
    n = symbols('n', negative=True)
    p, q = symbols('p q', positive=True)
    r = ((n + q * (-n / q + 1)) / (q * (-n / q + 1)) < 0)
    assert r is not S.false
    assert r.expand() is S.false
    assert (q > 0).expand() is S.true


def test_issue_12717():
    assert S.true.is_Atom == True
    assert S.false.is_Atom == True


def test_as_Boolean():
    nz = symbols('nz', nonzero=True)
    assert all(as_Boolean(i) is S.true for i in (True, S.true, 1, nz))
    z = symbols('z', zero=True)
    assert all(as_Boolean(i) is S.false for i in (False, S.false, 0, z))
    assert all(as_Boolean(i) == i for i in (x, x < 0))
    for i in (2, S(2), x + 1, []):
        raises(TypeError, lambda: as_Boolean(i))


def test_binary_symbols():
    assert ITE(x < 1, y, z).binary_symbols == {y, z}
    for f in (Eq, Ne):
        assert f(x, 1).binary_symbols == set()
        assert f(x, True).binary_symbols == {x}
        assert f(x, False).binary_symbols == {x}
    assert S.true.binary_symbols == set()
    assert S.false.binary_symbols == set()
    assert x.binary_symbols == {x}
    assert And(x, Eq(y, False), Eq(z, 1)).binary_symbols == {x, y}
    assert Q.prime(x).binary_symbols == set()
    assert Q.lt(x, 1).binary_symbols == set()
    assert Q.is_true(x).binary_symbols == {x}
    assert Q.eq(x, True).binary_symbols == {x}
    assert Q.prime(x).binary_symbols == set()


def test_BooleanFunction_diff():
    assert And(x, y).diff(x) == Piecewise((0, Eq(y, False)), (1, True))


def test_issue_14700():
    A, B, C, D, E, F, G, H = symbols('A B C D E F G H')
    q = ((B & D & H & ~F) | (B & H & ~C & ~D) | (B & H & ~C & ~F) |
         (B & H & ~D & ~G) | (B & H & ~F & ~G) | (C & G & ~B & ~D) |
         (C & G & ~D & ~H) | (C & G & ~F & ~H) | (D & F & H & ~B) |
         (D & F & ~G & ~H) | (B & D & F & ~C & ~H) | (D & E & F & ~B & ~C) |
         (D & F & ~A & ~B & ~C) | (D & F & ~A & ~C & ~H) |
         (A & B & D & F & ~E & ~H))
    soldnf = ((B & D & H & ~F) | (D & F & H & ~B) | (B & H & ~C & ~D) |
              (B & H & ~D & ~G) | (C & G & ~B & ~D) | (C & G & ~D & ~H) |
              (C & G & ~F & ~H) | (D & F & ~G & ~H) | (D & E & F & ~C & ~H) |
              (D & F & ~A & ~C & ~H) | (A & B & D & F & ~E & ~H))
    solcnf = ((B | C | D) & (B | D | G) & (C | D | H) & (C | F | H) &
              (D | G | H) & (F | G | H) & (B | F | ~D | ~H) &
              (~B | ~D | ~F | ~H) & (D | ~B | ~C | ~G | ~H) &
              (A | H | ~C | ~D | ~F | ~G) & (H | ~C | ~D | ~E | ~F | ~G) &
              (B | E | H | ~A | ~D | ~F | ~G))
    assert simplify_logic(q, "dnf") == soldnf
    assert simplify_logic(q, "cnf") == solcnf

    minterms = [[0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
                [0, 0, 1, 1], [1, 0, 1, 1]]
    dontcares = [[1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [1, 1, 0, 1]]
    assert SOPform([w, x, y, z], minterms) == (x & ~w) | (y & z & ~x)
    # Should not be more complicated with don't cares
    assert SOPform([w, x, y, z], minterms, dontcares) == \
           (x & ~w) | (y & z & ~x)


def test_issue_25115():
    cond = Contains(x, S.Integers)
    # Previously this raised an exception:
    assert simplify_logic(cond) == cond


def test_relational_simplification():
    w, x, y, z = symbols('w x y z', real=True)
    d, e = symbols('d e', real=False)
    # Test all combinations or sign and order
    assert Or(x >= y, x < y).simplify() == S.true
    assert Or(x >= y, y > x).simplify() == S.true
    assert Or(x >= y, -x > -y).simplify() == S.true
    assert Or(x >= y, -y < -x).simplify() == S.true
    assert Or(-x <= -y, x < y).simplify() == S.true
    assert Or(-x <= -y, -x > -y).simplify() == S.true
    assert Or(-x <= -y, y > x).simplify() == S.true
    assert Or(-x <= -y, -y < -x).simplify() == S.true
    assert Or(y <= x, x < y).simplify() == S.true
    assert Or(y <= x, y > x).simplify() == S.true
    assert Or(y <= x, -x > -y).simplify() == S.true
    assert Or(y <= x, -y < -x).simplify() == S.true
    assert Or(-y >= -x, x < y).simplify() == S.true
    assert Or(-y >= -x, y > x).simplify() == S.true
    assert Or(-y >= -x, -x > -y).simplify() == S.true
    assert Or(-y >= -x, -y < -x).simplify() == S.true

    assert Or(x < y, x >= y).simplify() == S.true
    assert Or(y > x, x >= y).simplify() == S.true
    assert Or(-x > -y, x >= y).simplify() == S.true
    assert Or(-y < -x, x >= y).simplify() == S.true
    assert Or(x < y, -x <= -y).simplify() == S.true
    assert Or(-x > -y, -x <= -y).simplify() == S.true
    assert Or(y > x, -x <= -y).simplify() == S.true
    assert Or(-y < -x, -x <= -y).simplify() == S.true
    assert Or(x < y, y <= x).simplify() == S.true
    assert Or(y > x, y <= x).simplify() == S.true
    assert Or(-x > -y, y <= x).simplify() == S.true
    assert Or(-y < -x, y <= x).simplify() == S.true
    assert Or(x < y, -y >= -x).simplify() == S.true
    assert Or(y > x, -y >= -x).simplify() == S.true
    assert Or(-x > -y, -y >= -x).simplify() == S.true
    assert Or(-y < -x, -y >= -x).simplify() == S.true

    # Some other tests
    assert Or(x >= y, w < z, x <= y).simplify() == S.true
    assert And(x >= y, x < y).simplify() == S.false
    assert Or(x >= y, Eq(y, x)).simplify() == (x >= y)
    assert And(x >= y, Eq(y, x)).simplify() == Eq(x, y)
    assert And(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y).simplify() == \
           (Eq(x, y) & (x >= 1) & (y >= 5) & (y > z))
    assert Or(Eq(x, y), x >= y, w < y, z < y).simplify() == \
           (x >= y) | (y > z) | (w < y)
    assert And(Eq(x, y), x >= y, w < y, y >= z, z < y).simplify() == \
           Eq(x, y) & (y > z) & (w < y)
    # assert And(Eq(x, y), x >= y, w < y, y >= z, z < y).simplify(relational_minmax=True) == \
    #    And(Eq(x, y), y > Max(w, z))
    # assert Or(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y).simplify(relational_minmax=True) == \
    #    (Eq(x, y) | (x >= 1) | (y > Min(2, z)))
    assert And(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y).simplify() == \
           (Eq(x, y) & (x >= 1) & (y >= 5) & (y > z))
    assert (Eq(x, y) & Eq(d, e) & (x >= y) & (d >= e)).simplify() == \
           (Eq(x, y) & Eq(d, e) & (d >= e))
    assert And(Eq(x, y), Eq(x, -y)).simplify() == And(Eq(x, 0), Eq(y, 0))
    assert Xor(x >= y, x <= y).simplify() == Ne(x, y)
    assert And(x > 1, x < -1, Eq(x, y)).simplify() == S.false
    # From #16690
    assert And(x >= y, Eq(y, 0)).simplify() == And(x >= 0, Eq(y, 0))
    assert Or(Ne(x, 1), Ne(x, 2)).simplify() == S.true
    assert And(Eq(x, 1), Ne(2, x)).simplify() == Eq(x, 1)
    assert Or(Eq(x, 1), Ne(2, x)).simplify() == Ne(x, 2)


def test_issue_8373():
    x = symbols('x', real=True)
    assert Or(x < 1, x > -1).simplify() == S.true
    assert Or(x < 1, x >= 1).simplify() == S.true
    assert And(x < 1, x >= 1).simplify() == S.false
    assert Or(x <= 1, x >= 1).simplify() == S.true


def test_issue_7950():
    x = symbols('x', real=True)
    assert And(Eq(x, 1), Eq(x, 2)).simplify() == S.false


@slow
def test_relational_simplification_numerically():
    def test_simplification_numerically_function(original, simplified):
        symb = original.free_symbols
        n = len(symb)
        valuelist = list(set(combinations(list(range(-(n - 1), n)) * n, n)))
        for values in valuelist:
            sublist = dict(zip(symb, values))
            originalvalue = original.subs(sublist)
            simplifiedvalue = simplified.subs(sublist)
            assert originalvalue == simplifiedvalue, "Original: {}\nand" \
                                                     " simplified: {}\ndo not evaluate to the same value for {}" \
                                                     "".format(original, simplified, sublist)

    w, x, y, z = symbols('w x y z', real=True)
    d, e = symbols('d e', real=False)

    expressions = (And(Eq(x, y), x >= y, w < y, y >= z, z < y),
                   And(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y),
                   Or(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y),
                   And(x >= y, Eq(y, x)),
                   Or(And(Eq(x, y), x >= y, w < y, Or(y >= z, z < y)),
                      And(Eq(x, y), x >= 1, 2 < y, y >= -1, z < y)),
                   (Eq(x, y) & Eq(d, e) & (x >= y) & (d >= e)),
                   )

    for expression in expressions:
        test_simplification_numerically_function(expression,
                                                 expression.simplify())


def test_relational_simplification_patterns_numerically():
    from sympy.core import Wild
    from sympy.logic.boolalg import _simplify_patterns_and, \
        _simplify_patterns_or, _simplify_patterns_xor
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    symb = [a, b, c]
    patternlists = [[And, _simplify_patterns_and()],
                    [Or, _simplify_patterns_or()],
                    [Xor, _simplify_patterns_xor()]]
    valuelist = list(set(combinations(list(range(-2, 3)) * 3, 3)))
    # Skip combinations of +/-2 and 0, except for all 0
    valuelist = [v for v in valuelist if any(w % 2 for w in v) or not any(v)]
    for func, patternlist in patternlists:
        for pattern in patternlist:
            original = func(*pattern[0].args)
            simplified = pattern[1]
            for values in valuelist:
                sublist = dict(zip(symb, values))
                originalvalue = original.xreplace(sublist)
                simplifiedvalue = simplified.xreplace(sublist)
                assert originalvalue == simplifiedvalue, "Original: {}\nand" \
                                                         " simplified: {}\ndo not evaluate to the same value for" \
                                                         "{}".format(pattern[0], simplified, sublist)


def test_issue_16803():
    n = symbols('n')
    # No simplification done, but should not raise an exception
    assert ((n > 3) | (n < 0) | ((n > 0) & (n < 3))).simplify() == \
           (n > 3) | (n < 0) | ((n > 0) & (n < 3))


def test_issue_17530():
    r = {x: oo, y: oo}
    assert Or(x + y > 0, x - y < 0).subs(r)
    assert not And(x + y < 0, x - y < 0).subs(r)
    raises(TypeError, lambda: Or(x + y < 0, x - y < 0).subs(r))
    raises(TypeError, lambda: And(x + y > 0, x - y < 0).subs(r))
    raises(TypeError, lambda: And(x + y > 0, x - y < 0).subs(r))


def test_anf_coeffs():
    assert anf_coeffs([1, 0]) == [1, 1]
    assert anf_coeffs([0, 0, 0, 1]) == [0, 0, 0, 1]
    assert anf_coeffs([0, 1, 1, 1]) == [0, 1, 1, 1]
    assert anf_coeffs([1, 1, 1, 0]) == [1, 0, 0, 1]
    assert anf_coeffs([1, 0, 0, 0]) == [1, 1, 1, 1]
    assert anf_coeffs([1, 0, 0, 1]) == [1, 1, 1, 0]
    assert anf_coeffs([1, 1, 0, 1]) == [1, 0, 1, 1]


def test_ANFform():
    x, y = symbols('x,y')
    assert ANFform([x], [1, 1]) == True
    assert ANFform([x], [0, 0]) == False
    assert ANFform([x], [1, 0]) == Xor(x, True, remove_true=False)
    assert ANFform([x, y], [1, 1, 1, 0]) == \
           Xor(True, And(x, y), remove_true=False)


def test_bool_minterm():
    x, y = symbols('x,y')
    assert bool_minterm(3, [x, y]) == And(x, y)
    assert bool_minterm([1, 0], [x, y]) == And(Not(y), x)


def test_bool_maxterm():
    x, y = symbols('x,y')
    assert bool_maxterm(2, [x, y]) == Or(Not(x), y)
    assert bool_maxterm([0, 1], [x, y]) == Or(Not(y), x)


def test_bool_monomial():
    x, y = symbols('x,y')
    assert bool_monomial(1, [x, y]) == y
    assert bool_monomial([1, 1], [x, y]) == And(x, y)


def test_check_pair():
    assert _check_pair([0, 1, 0], [0, 1, 1]) == 2
    assert _check_pair([0, 1, 0], [1, 1, 1]) == -1


def test_issue_19114():
    expr = (B & C) | (A & ~C) | (~A & ~B)
    # Expression is minimal, but there are multiple minimal forms possible
    res1 = (A & B) | (C & ~A) | (~B & ~C)
    result = to_dnf(expr, simplify=True)
    assert result in (expr, res1)


def test_issue_20870():
    result = SOPform([a, b, c, d], [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 14, 15])
    expected = ((d & ~b) | (a & b & c) | (a & ~c & ~d) |
                (b & ~a & ~c) | (c & ~a & ~d))
    assert result == expected


def test_convert_to_varsSOP():
    assert _convert_to_varsSOP([0, 1, 0], [x, y, z]) == And(Not(x), y, Not(z))
    assert _convert_to_varsSOP([3, 1, 0], [x, y, z]) == And(y, Not(z))


def test_convert_to_varsPOS():
    assert _convert_to_varsPOS([0, 1, 0], [x, y, z]) == Or(x, Not(y), z)
    assert _convert_to_varsPOS([3, 1, 0], [x, y, z]) == Or(Not(y), z)


def test_gateinputcount():
    a, b, c, d, e = symbols('a:e')
    assert gateinputcount(And(a, b)) == 2
    assert gateinputcount(a | b & c & d ^ (e | a)) == 9
    assert gateinputcount(And(a, True)) == 0
    raises(TypeError, lambda: gateinputcount(a * b))


def test_refine():
    # relational
    assert not refine(x < 0, ~(x < 0))
    assert refine(x < 0, (x < 0))
    assert refine(x < 0, (0 > x)) is S.true
    assert refine(x < 0, (y < 0)) == (x < 0)
    assert not refine(x <= 0, ~(x <= 0))
    assert refine(x <= 0, (x <= 0))
    assert refine(x <= 0, (0 >= x)) is S.true
    assert refine(x <= 0, (y <= 0)) == (x <= 0)
    assert not refine(x > 0, ~(x > 0))
    assert refine(x > 0, (x > 0))
    assert refine(x > 0, (0 < x)) is S.true
    assert refine(x > 0, (y > 0)) == (x > 0)
    assert not refine(x >= 0, ~(x >= 0))
    assert refine(x >= 0, (x >= 0))
    assert refine(x >= 0, (0 <= x)) is S.true
    assert refine(x >= 0, (y >= 0)) == (x >= 0)
    assert not refine(Eq(x, 0), ~(Eq(x, 0)))
    assert refine(Eq(x, 0), (Eq(x, 0)))
    assert refine(Eq(x, 0), (Eq(0, x))) is S.true
    assert refine(Eq(x, 0), (Eq(y, 0))) == Eq(x, 0)
    assert not refine(Ne(x, 0), ~(Ne(x, 0)))
    assert refine(Ne(x, 0), (Ne(0, x))) is S.true
    assert refine(Ne(x, 0), (Ne(x, 0)))
    assert refine(Ne(x, 0), (Ne(y, 0))) == (Ne(x, 0))

    # boolean functions
    assert refine(And(x > 0, y > 0), (x > 0)) == (y > 0)
    assert refine(And(x > 0, y > 0), (x > 0) & (y > 0)) is S.true

    # predicates
    assert refine(Q.positive(x), Q.positive(x)) is S.true
    assert refine(Q.positive(x), Q.negative(x)) is S.false
    assert refine(Q.positive(x), Q.real(x)) == Q.positive(x)


def test_relational_threeterm_simplification_patterns_numerically():
    from sympy.core import Wild
    from sympy.logic.boolalg import _simplify_patterns_and3
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    symb = [a, b, c]
    patternlists = [[And, _simplify_patterns_and3()]]
    valuelist = list(set(combinations(list(range(-2, 3)) * 3, 3)))
    # Skip combinations of +/-2 and 0, except for all 0
    valuelist = [v for v in valuelist if any(w % 2 for w in v) or not any(v)]
    for func, patternlist in patternlists:
        for pattern in patternlist:
            original = func(*pattern[0].args)
            simplified = pattern[1]
            for values in valuelist:
                sublist = dict(zip(symb, values))
                originalvalue = original.xreplace(sublist)
                simplifiedvalue = simplified.xreplace(sublist)
                assert originalvalue == simplifiedvalue, "Original: {}\nand" \
                                                         " simplified: {}\ndo not evaluate to the same value for" \
                                                         "{}".format(pattern[0], simplified, sublist)


def test_issue_25451():
    x = Or(And(a, c), Eq(a, b))
    assert isinstance(x, Or)
    assert set(x.args) == {And(a, c), Eq(a, b)}


def test_issue_26985():
    a, b, c, d = symbols('a b c d')

    # Expression before applying to_anf
    x = Xor(c, And(a, b), And(a, c))
    y = Xor(a, b, And(a, c))

    # Applying to_anf
    result = Xor(Xor(d, And(x, y)), And(x, y))
    result_anf = to_anf(Xor(to_anf(Xor(d, And(x, y))), And(x, y)))

    assert result_anf == d
    assert result == d

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\test_datetimelike.py ===
from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

from pandas._libs import (
    NaT,
    OutOfBoundsDatetime,
    Timestamp,
)
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas.compat.numpy import np_version_gt2

import pandas as pd
from pandas import (
    DatetimeIndex,
    Period,
    PeriodIndex,
    TimedeltaIndex,
)
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    NumpyExtensionArray,
    PeriodArray,
    TimedeltaArray,
)


# TODO: more freq variants
@pytest.fixture(params=["D", "B", "W", "ME", "QE", "YE"])
def freqstr(request):
    """Fixture returning parametrized frequency in string format."""
    return request.param


@pytest.fixture
def period_index(freqstr):
    """
    A fixture to provide PeriodIndex objects with different frequencies.

    Most PeriodArray behavior is already tested in PeriodIndex tests,
    so here we just test that the PeriodArray behavior matches
    the PeriodIndex behavior.
    """
    # TODO: non-monotone indexes; NaTs, different start dates
    with warnings.catch_warnings():
        # suppress deprecation of Period[B]
        warnings.filterwarnings(
            "ignore", message="Period with BDay freq", category=FutureWarning
        )
        freqstr = freq_to_period_freqstr(1, freqstr)
        pi = pd.period_range(start=Timestamp("2000-01-01"), periods=100, freq=freqstr)
    return pi


@pytest.fixture
def datetime_index(freqstr):
    """
    A fixture to provide DatetimeIndex objects with different frequencies.

    Most DatetimeArray behavior is already tested in DatetimeIndex tests,
    so here we just test that the DatetimeArray behavior matches
    the DatetimeIndex behavior.
    """
    # TODO: non-monotone indexes; NaTs, different start dates, timezones
    dti = pd.date_range(start=Timestamp("2000-01-01"), periods=100, freq=freqstr)
    return dti


@pytest.fixture
def timedelta_index():
    """
    A fixture to provide TimedeltaIndex objects with different frequencies.
     Most TimedeltaArray behavior is already tested in TimedeltaIndex tests,
    so here we just test that the TimedeltaArray behavior matches
    the TimedeltaIndex behavior.
    """
    # TODO: flesh this out
    return TimedeltaIndex(["1 Day", "3 Hours", "NaT"])


class SharedTests:
    index_cls: type[DatetimeIndex | PeriodIndex | TimedeltaIndex]

    @pytest.fixture
    def arr1d(self):
        """Fixture returning DatetimeArray with daily frequency."""
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, freq="D")
        else:
            arr = self.index_cls(data, freq="D")._data
        return arr

    def test_compare_len1_raises(self, arr1d):
        # make sure we raise when comparing with different lengths, specific
        #  to the case where one has length-1, which numpy would broadcast
        arr = arr1d
        idx = self.index_cls(arr)

        with pytest.raises(ValueError, match="Lengths must match"):
            arr == arr[:1]

        # test the index classes while we're at it, GH#23078
        with pytest.raises(ValueError, match="Lengths must match"):
            idx <= idx[[0]]

    @pytest.mark.parametrize(
        "result",
        [
            pd.date_range("2020", periods=3),
            pd.date_range("2020", periods=3, tz="UTC"),
            pd.timedelta_range("0 days", periods=3),
            pd.period_range("2020Q1", periods=3, freq="Q"),
        ],
    )
    def test_compare_with_Categorical(self, result):
        expected = pd.Categorical(result)
        assert all(result == expected)
        assert not any(result != expected)

    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize("as_index", [True, False])
    def test_compare_categorical_dtype(self, arr1d, as_index, reverse, ordered):
        other = pd.Categorical(arr1d, ordered=ordered)
        if as_index:
            other = pd.CategoricalIndex(other)

        left, right = arr1d, other
        if reverse:
            left, right = right, left

        ones = np.ones(arr1d.shape, dtype=bool)
        zeros = ~ones

        result = left == right
        tm.assert_numpy_array_equal(result, ones)

        result = left != right
        tm.assert_numpy_array_equal(result, zeros)

        if not reverse and not as_index:
            # Otherwise Categorical raises TypeError bc it is not ordered
            # TODO: we should probably get the same behavior regardless?
            result = left < right
            tm.assert_numpy_array_equal(result, zeros)

            result = left <= right
            tm.assert_numpy_array_equal(result, ones)

            result = left > right
            tm.assert_numpy_array_equal(result, zeros)

            result = left >= right
            tm.assert_numpy_array_equal(result, ones)

    def test_take(self):
        data = np.arange(100, dtype="i8") * 24 * 3600 * 10**9
        np.random.default_rng(2).shuffle(data)

        if self.array_cls is PeriodArray:
            arr = PeriodArray(data, dtype="period[D]")
        else:
            arr = self.index_cls(data)._data
        idx = self.index_cls._simple_new(arr)

        takers = [1, 4, 94]
        result = arr.take(takers)
        expected = idx.take(takers)

        tm.assert_index_equal(self.index_cls(result), expected)

        takers = np.array([1, 4, 94])
        result = arr.take(takers)
        expected = idx.take(takers)

        tm.assert_index_equal(self.index_cls(result), expected)

    @pytest.mark.parametrize("fill_value", [2, 2.0, Timestamp(2021, 1, 1, 12).time])
    def test_take_fill_raises(self, fill_value, arr1d):
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            arr1d.take([0, 1], allow_fill=True, fill_value=fill_value)

    def test_take_fill(self, arr1d):
        arr = arr1d

        result = arr.take([-1, 1], allow_fill=True, fill_value=None)
        assert result[0] is NaT

        result = arr.take([-1, 1], allow_fill=True, fill_value=np.nan)
        assert result[0] is NaT

        result = arr.take([-1, 1], allow_fill=True, fill_value=NaT)
        assert result[0] is NaT

    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_take_fill_str(self, arr1d):
        # Cast str fill_value matching other fill_value-taking methods
        result = arr1d.take([-1, 1], allow_fill=True, fill_value=str(arr1d[-1]))
        expected = arr1d[[-1, 1]]
        tm.assert_equal(result, expected)

        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            arr1d.take([-1, 1], allow_fill=True, fill_value="foo")

    def test_concat_same_type(self, arr1d):
        arr = arr1d
        idx = self.index_cls(arr)
        idx = idx.insert(0, NaT)
        arr = arr1d

        result = arr._concat_same_type([arr[:-1], arr[1:], arr])
        arr2 = arr.astype(object)
        expected = self.index_cls(np.concatenate([arr2[:-1], arr2[1:], arr2]))

        tm.assert_index_equal(self.index_cls(result), expected)

    def test_unbox_scalar(self, arr1d):
        result = arr1d._unbox_scalar(arr1d[0])
        expected = arr1d._ndarray.dtype.type
        assert isinstance(result, expected)

        result = arr1d._unbox_scalar(NaT)
        assert isinstance(result, expected)

        msg = f"'value' should be a {self.scalar_type.__name__}."
        with pytest.raises(ValueError, match=msg):
            arr1d._unbox_scalar("foo")

    def test_check_compatible_with(self, arr1d):
        arr1d._check_compatible_with(arr1d[0])
        arr1d._check_compatible_with(arr1d[:1])
        arr1d._check_compatible_with(NaT)

    def test_scalar_from_string(self, arr1d):
        result = arr1d._scalar_from_string(str(arr1d[0]))
        assert result == arr1d[0]

    def test_reduce_invalid(self, arr1d):
        msg = "does not support reduction 'not a method'"
        with pytest.raises(TypeError, match=msg):
            arr1d._reduce("not a method")

    @pytest.mark.parametrize("method", ["pad", "backfill"])
    def test_fillna_method_doesnt_change_orig(self, method):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, dtype="period[D]")
        else:
            arr = self.array_cls._from_sequence(data)
        arr[4] = NaT

        fill_value = arr[3] if method == "pad" else arr[5]

        result = arr._pad_or_backfill(method=method)
        assert result[4] == fill_value

        # check that the original was not changed
        assert arr[4] is NaT

    def test_searchsorted(self):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, dtype="period[D]")
        else:
            arr = self.array_cls._from_sequence(data)

        # scalar
        result = arr.searchsorted(arr[1])
        assert result == 1

        result = arr.searchsorted(arr[2], side="right")
        assert result == 3

        # own-type
        result = arr.searchsorted(arr[1:3])
        expected = np.array([1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        result = arr.searchsorted(arr[1:3], side="right")
        expected = np.array([2, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        # GH#29884 match numpy convention on whether NaT goes
        #  at the end or the beginning
        result = arr.searchsorted(NaT)
        assert result == 10

    @pytest.mark.parametrize("box", [None, "index", "series"])
    def test_searchsorted_castable_strings(self, arr1d, box, string_storage):
        arr = arr1d
        if box is None:
            pass
        elif box == "index":
            # Test the equivalent Index.searchsorted method while we're here
            arr = self.index_cls(arr)
        else:
            # Test the equivalent Series.searchsorted method while we're here
            arr = pd.Series(arr)

        # scalar
        result = arr.searchsorted(str(arr[1]))
        assert result == 1

        result = arr.searchsorted(str(arr[2]), side="right")
        assert result == 3

        result = arr.searchsorted([str(x) for x in arr[1:3]])
        expected = np.array([1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', "
                "or array of those. Got 'str' instead."
            ),
        ):
            arr.searchsorted("foo")

        with pd.option_context("string_storage", string_storage):
            with pytest.raises(
                TypeError,
                match=re.escape(
                    f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', "
                    "or array of those. Got string array instead."
                ),
            ):
                arr.searchsorted([str(arr[1]), "baz"])

    def test_getitem_near_implementation_bounds(self):
        # We only check tz-naive for DTA bc the bounds are slightly different
        #  for other tzs
        i8vals = np.asarray([NaT._value + n for n in range(1, 5)], dtype="i8")
        if self.array_cls is PeriodArray:
            arr = self.array_cls(i8vals, dtype="period[ns]")
        else:
            arr = self.index_cls(i8vals, freq="ns")._data
        arr[0]  # should not raise OutOfBoundsDatetime

        index = pd.Index(arr)
        index[0]  # should not raise OutOfBoundsDatetime

        ser = pd.Series(arr)
        ser[0]  # should not raise OutOfBoundsDatetime

    def test_getitem_2d(self, arr1d):
        # 2d slicing on a 1D array
        expected = type(arr1d)._simple_new(
            arr1d._ndarray[:, np.newaxis], dtype=arr1d.dtype
        )
        result = arr1d[:, np.newaxis]
        tm.assert_equal(result, expected)

        # Lookup on a 2D array
        arr2d = expected
        expected = type(arr2d)._simple_new(arr2d._ndarray[:3, 0], dtype=arr2d.dtype)
        result = arr2d[:3, 0]
        tm.assert_equal(result, expected)

        # Scalar lookup
        result = arr2d[-1, 0]
        expected = arr1d[-1]
        assert result == expected

    def test_iter_2d(self, arr1d):
        data2d = arr1d._ndarray[:3, np.newaxis]
        arr2d = type(arr1d)._simple_new(data2d, dtype=arr1d.dtype)
        result = list(arr2d)
        assert len(result) == 3
        for x in result:
            assert isinstance(x, type(arr1d))
            assert x.ndim == 1
            assert x.dtype == arr1d.dtype

    def test_repr_2d(self, arr1d):
        data2d = arr1d._ndarray[:3, np.newaxis]
        arr2d = type(arr1d)._simple_new(data2d, dtype=arr1d.dtype)

        result = repr(arr2d)

        if isinstance(arr2d, TimedeltaArray):
            expected = (
                f"<{type(arr2d).__name__}>\n"
                "[\n"
                f"['{arr1d[0]._repr_base()}'],\n"
                f"['{arr1d[1]._repr_base()}'],\n"
                f"['{arr1d[2]._repr_base()}']\n"
                "]\n"
                f"Shape: (3, 1), dtype: {arr1d.dtype}"
            )
        else:
            expected = (
                f"<{type(arr2d).__name__}>\n"
                "[\n"
                f"['{arr1d[0]}'],\n"
                f"['{arr1d[1]}'],\n"
                f"['{arr1d[2]}']\n"
                "]\n"
                f"Shape: (3, 1), dtype: {arr1d.dtype}"
            )

        assert result == expected

    def test_setitem(self):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, dtype="period[D]")
        else:
            arr = self.index_cls(data, freq="D")._data

        arr[0] = arr[1]
        expected = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        expected[0] = expected[1]

        tm.assert_numpy_array_equal(arr.asi8, expected)

        arr[:2] = arr[-2:]
        expected[:2] = expected[-2:]
        tm.assert_numpy_array_equal(arr.asi8, expected)

    @pytest.mark.parametrize(
        "box",
        [
            pd.Index,
            pd.Series,
            np.array,
            list,
            NumpyExtensionArray,
        ],
    )
    def test_setitem_object_dtype(self, box, arr1d):
        expected = arr1d.copy()[::-1]
        if expected.dtype.kind in ["m", "M"]:
            expected = expected._with_freq(None)

        vals = expected
        if box is list:
            vals = list(vals)
        elif box is np.array:
            # if we do np.array(x).astype(object) then dt64 and td64 cast to ints
            vals = np.array(vals.astype(object))
        elif box is NumpyExtensionArray:
            vals = box(np.asarray(vals, dtype=object))
        else:
            vals = box(vals).astype(object)

        arr1d[:] = vals

        tm.assert_equal(arr1d, expected)

    def test_setitem_strs(self, arr1d):
        # Check that we parse strs in both scalar and listlike

        # Setting list-like of strs
        expected = arr1d.copy()
        expected[[0, 1]] = arr1d[-2:]

        result = arr1d.copy()
        result[:2] = [str(x) for x in arr1d[-2:]]
        tm.assert_equal(result, expected)

        # Same thing but now for just a scalar str
        expected = arr1d.copy()
        expected[0] = arr1d[-1]

        result = arr1d.copy()
        result[0] = str(arr1d[-1])
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("as_index", [True, False])
    def test_setitem_categorical(self, arr1d, as_index):
        expected = arr1d.copy()[::-1]
        if not isinstance(expected, PeriodArray):
            expected = expected._with_freq(None)

        cat = pd.Categorical(arr1d)
        if as_index:
            cat = pd.CategoricalIndex(cat)

        arr1d[:] = cat[::-1]

        tm.assert_equal(arr1d, expected)

    def test_setitem_raises(self, arr1d):
        arr = arr1d[:10]
        val = arr[0]

        with pytest.raises(IndexError, match="index 12 is out of bounds"):
            arr[12] = val

        with pytest.raises(TypeError, match="value should be a.* 'object'"):
            arr[0] = object()

        msg = "cannot set using a list-like indexer with a different length"
        with pytest.raises(ValueError, match=msg):
            # GH#36339
            arr[[]] = [arr[1]]

        msg = "cannot set using a slice indexer with a different length than"
        with pytest.raises(ValueError, match=msg):
            # GH#36339
            arr[1:1] = arr[:3]

    @pytest.mark.parametrize("box", [list, np.array, pd.Index, pd.Series])
    def test_setitem_numeric_raises(self, arr1d, box):
        # We dont case e.g. int64 to our own dtype for setitem

        msg = (
            f"value should be a '{arr1d._scalar_type.__name__}', "
            "'NaT', or array of those. Got"
        )
        with pytest.raises(TypeError, match=msg):
            arr1d[:2] = box([0, 1])

        with pytest.raises(TypeError, match=msg):
            arr1d[:2] = box([0.0, 1.0])

    def test_inplace_arithmetic(self):
        # GH#24115 check that iadd and isub are actually in-place
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, dtype="period[D]")
        else:
            arr = self.index_cls(data, freq="D")._data

        expected = arr + pd.Timedelta(days=1)
        arr += pd.Timedelta(days=1)
        tm.assert_equal(arr, expected)

        expected = arr - pd.Timedelta(days=1)
        arr -= pd.Timedelta(days=1)
        tm.assert_equal(arr, expected)

    def test_shift_fill_int_deprecated(self, arr1d):
        # GH#31971, enforced in 2.0
        with pytest.raises(TypeError, match="value should be a"):
            arr1d.shift(1, fill_value=1)

    def test_median(self, arr1d):
        arr = arr1d
        if len(arr) % 2 == 0:
            # make it easier to define `expected`
            arr = arr[:-1]

        expected = arr[len(arr) // 2]

        result = arr.median()
        assert type(result) is type(expected)
        assert result == expected

        arr[len(arr) // 2] = NaT
        if not isinstance(expected, Period):
            expected = arr[len(arr) // 2 - 1 : len(arr) // 2 + 2].mean()

        assert arr.median(skipna=False) is NaT

        result = arr.median()
        assert type(result) is type(expected)
        assert result == expected

        assert arr[:0].median() is NaT
        assert arr[:0].median(skipna=False) is NaT

        # 2d Case
        arr2 = arr.reshape(-1, 1)

        result = arr2.median(axis=None)
        assert type(result) is type(expected)
        assert result == expected

        assert arr2.median(axis=None, skipna=False) is NaT

        result = arr2.median(axis=0)
        expected2 = type(arr)._from_sequence([expected], dtype=arr.dtype)
        tm.assert_equal(result, expected2)

        result = arr2.median(axis=0, skipna=False)
        expected2 = type(arr)._from_sequence([NaT], dtype=arr.dtype)
        tm.assert_equal(result, expected2)

        result = arr2.median(axis=1)
        tm.assert_equal(result, arr)

        result = arr2.median(axis=1, skipna=False)
        tm.assert_equal(result, arr)

    def test_from_integer_array(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        data = pd.array(arr, dtype="Int64")
        if self.array_cls is PeriodArray:
            expected = self.array_cls(arr, dtype=self.example_dtype)
            result = self.array_cls(data, dtype=self.example_dtype)
        else:
            expected = self.array_cls._from_sequence(arr, dtype=self.example_dtype)
            result = self.array_cls._from_sequence(data, dtype=self.example_dtype)

        tm.assert_extension_array_equal(result, expected)


class TestDatetimeArray(SharedTests):
    index_cls = DatetimeIndex
    array_cls = DatetimeArray
    scalar_type = Timestamp
    example_dtype = "M8[ns]"

    @pytest.fixture
    def arr1d(self, tz_naive_fixture, freqstr):
        """
        Fixture returning DatetimeArray with parametrized frequency and
        timezones
        """
        tz = tz_naive_fixture
        dti = pd.date_range("2016-01-01 01:01:00", periods=5, freq=freqstr, tz=tz)
        dta = dti._data
        return dta

    def test_round(self, arr1d):
        # GH#24064
        dti = self.index_cls(arr1d)

        result = dti.round(freq="2min")
        expected = dti - pd.Timedelta(minutes=1)
        expected = expected._with_freq(None)
        tm.assert_index_equal(result, expected)

        dta = dti._data
        result = dta.round(freq="2min")
        expected = expected._data._with_freq(None)
        tm.assert_datetime_array_equal(result, expected)

    def test_array_interface(self, datetime_index):
        arr = datetime_index._data
        copy_false = None if np_version_gt2 else False

        # default asarray gives the same underlying data (for tz naive)
        result = np.asarray(arr)
        expected = arr._ndarray
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, copy=copy_false)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)

        # specifying M8[ns] gives the same result as default
        result = np.asarray(arr, dtype="datetime64[ns]")
        expected = arr._ndarray
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype="datetime64[ns]", copy=copy_false)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype="datetime64[ns]")
        if not np_version_gt2:
            # TODO: GH 57739
            assert result is not expected
        tm.assert_numpy_array_equal(result, expected)

        # to object dtype
        result = np.asarray(arr, dtype=object)
        expected = np.array(list(arr), dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        # to other dtype always copies
        result = np.asarray(arr, dtype="int64")
        assert result is not arr.asi8
        assert not np.may_share_memory(arr, result)
        expected = arr.asi8.copy()
        tm.assert_numpy_array_equal(result, expected)

        # other dtypes handled by numpy
        for dtype in ["float64", str]:
            result = np.asarray(arr, dtype=dtype)
            expected = np.asarray(arr).astype(dtype)
            tm.assert_numpy_array_equal(result, expected)

    def test_array_object_dtype(self, arr1d):
        # GH#23524
        arr = arr1d
        dti = self.index_cls(arr1d)

        expected = np.array(list(dti))

        result = np.array(arr, dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        # also test the DatetimeIndex method while we're at it
        result = np.array(dti, dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_array_tz(self, arr1d):
        # GH#23524
        arr = arr1d
        dti = self.index_cls(arr1d)
        copy_false = None if np_version_gt2 else False

        expected = dti.asi8.view("M8[ns]")
        result = np.array(arr, dtype="M8[ns]")
        tm.assert_numpy_array_equal(result, expected)

        result = np.array(arr, dtype="datetime64[ns]")
        tm.assert_numpy_array_equal(result, expected)

        # check that we are not making copies when setting copy=copy_false
        result = np.array(arr, dtype="M8[ns]", copy=copy_false)
        assert result.base is expected.base
        assert result.base is not None
        result = np.array(arr, dtype="datetime64[ns]", copy=copy_false)
        assert result.base is expected.base
        assert result.base is not None

    def test_array_i8_dtype(self, arr1d):
        arr = arr1d
        dti = self.index_cls(arr1d)
        copy_false = None if np_version_gt2 else False

        expected = dti.asi8
        result = np.array(arr, dtype="i8")
        tm.assert_numpy_array_equal(result, expected)

        result = np.array(arr, dtype=np.int64)
        tm.assert_numpy_array_equal(result, expected)

        # check that we are still making copies when setting copy=copy_false
        result = np.array(arr, dtype="i8", copy=copy_false)
        assert result.base is not expected.base
        assert result.base is None

    def test_from_array_keeps_base(self):
        # Ensure that DatetimeArray._ndarray.base isn't lost.
        arr = np.array(["2000-01-01", "2000-01-02"], dtype="M8[ns]")
        dta = DatetimeArray._from_sequence(arr)

        assert dta._ndarray is arr
        dta = DatetimeArray._from_sequence(arr[:0])
        assert dta._ndarray.base is arr

    def test_from_dti(self, arr1d):
        arr = arr1d
        dti = self.index_cls(arr1d)
        assert list(dti) == list(arr)

        # Check that Index.__new__ knows what to do with DatetimeArray
        dti2 = pd.Index(arr)
        assert isinstance(dti2, DatetimeIndex)
        assert list(dti2) == list(arr)

    def test_astype_object(self, arr1d):
        arr = arr1d
        dti = self.index_cls(arr1d)

        asobj = arr.astype("O")
        assert isinstance(asobj, np.ndarray)
        assert asobj.dtype == "O"
        assert list(asobj) == list(dti)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_to_period(self, datetime_index, freqstr):
        dti = datetime_index
        arr = dti._data

        freqstr = freq_to_period_freqstr(1, freqstr)
        expected = dti.to_period(freq=freqstr)
        result = arr.to_period(freq=freqstr)
        assert isinstance(result, PeriodArray)

        tm.assert_equal(result, expected._data)

    def test_to_period_2d(self, arr1d):
        arr2d = arr1d.reshape(1, -1)

        warn = None if arr1d.tz is None else UserWarning
        with tm.assert_produces_warning(warn):
            result = arr2d.to_period("D")
            expected = arr1d.to_period("D").reshape(1, -1)
        tm.assert_period_array_equal(result, expected)

    @pytest.mark.parametrize("propname", DatetimeArray._bool_ops)
    def test_bool_properties(self, arr1d, propname):
        # in this case _bool_ops is just `is_leap_year`
        dti = self.index_cls(arr1d)
        arr = arr1d
        assert dti.freq == arr.freq

        result = getattr(arr, propname)
        expected = np.array(getattr(dti, propname), dtype=result.dtype)

        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("propname", DatetimeArray._field_ops)
    def test_int_properties(self, arr1d, propname):
        dti = self.index_cls(arr1d)
        arr = arr1d

        result = getattr(arr, propname)
        expected = np.array(getattr(dti, propname), dtype=result.dtype)

        tm.assert_numpy_array_equal(result, expected)

    def test_take_fill_valid(self, arr1d, fixed_now_ts):
        arr = arr1d
        dti = self.index_cls(arr1d)

        now = fixed_now_ts.tz_localize(dti.tz)
        result = arr.take([-1, 1], allow_fill=True, fill_value=now)
        assert result[0] == now

        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            # fill_value Timedelta invalid
            arr.take([-1, 1], allow_fill=True, fill_value=now - now)

        with pytest.raises(TypeError, match=msg):
            # fill_value Period invalid
            arr.take([-1, 1], allow_fill=True, fill_value=Period("2014Q1"))

        tz = None if dti.tz is not None else "US/Eastern"
        now = fixed_now_ts.tz_localize(tz)
        msg = "Cannot compare tz-naive and tz-aware datetime-like objects"
        with pytest.raises(TypeError, match=msg):
            # Timestamp with mismatched tz-awareness
            arr.take([-1, 1], allow_fill=True, fill_value=now)

        value = NaT._value
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            # require NaT, not iNaT, as it could be confused with an integer
            arr.take([-1, 1], allow_fill=True, fill_value=value)

        value = np.timedelta64("NaT", "ns")
        with pytest.raises(TypeError, match=msg):
            # require appropriate-dtype if we have a NA value
            arr.take([-1, 1], allow_fill=True, fill_value=value)

        if arr.tz is not None:
            # GH#37356
            # Assuming here that arr1d fixture does not include Australia/Melbourne
            value = fixed_now_ts.tz_localize("Australia/Melbourne")
            result = arr.take([-1, 1], allow_fill=True, fill_value=value)

            expected = arr.take(
                [-1, 1],
                allow_fill=True,
                fill_value=value.tz_convert(arr.dtype.tz),
            )
            tm.assert_equal(result, expected)

    def test_concat_same_type_invalid(self, arr1d):
        # different timezones
        arr = arr1d

        if arr.tz is None:
            other = arr.tz_localize("UTC")
        else:
            other = arr.tz_localize(None)

        with pytest.raises(ValueError, match="to_concat must have the same"):
            arr._concat_same_type([arr, other])

    def test_concat_same_type_different_freq(self, unit):
        # we *can* concatenate DTI with different freqs.
        a = pd.date_range("2000", periods=2, freq="D", tz="US/Central", unit=unit)._data
        b = pd.date_range("2000", periods=2, freq="h", tz="US/Central", unit=unit)._data
        result = DatetimeArray._concat_same_type([a, b])
        expected = (
            pd.to_datetime(
                [
                    "2000-01-01 00:00:00",
                    "2000-01-02 00:00:00",
                    "2000-01-01 00:00:00",
                    "2000-01-01 01:00:00",
                ]
            )
            .tz_localize("US/Central")
            .as_unit(unit)
            ._data
        )

        tm.assert_datetime_array_equal(result, expected)

    def test_strftime(self, arr1d, using_infer_string):
        arr = arr1d

        result = arr.strftime("%Y %b")
        expected = np.array([ts.strftime("%Y %b") for ts in arr], dtype=object)
        if using_infer_string:
            expected = pd.array(expected, dtype=pd.StringDtype(na_value=np.nan))
        tm.assert_equal(result, expected)

    def test_strftime_nat(self, using_infer_string):
        # GH 29578
        arr = DatetimeIndex(["2019-01-01", NaT])._data

        result = arr.strftime("%Y-%m-%d")
        expected = np.array(["2019-01-01", np.nan], dtype=object)
        if using_infer_string:
            expected = pd.array(expected, dtype=pd.StringDtype(na_value=np.nan))
        tm.assert_equal(result, expected)


class TestTimedeltaArray(SharedTests):
    index_cls = TimedeltaIndex
    array_cls = TimedeltaArray
    scalar_type = pd.Timedelta
    example_dtype = "m8[ns]"

    def test_from_tdi(self):
        tdi = TimedeltaIndex(["1 Day", "3 Hours"])
        arr = tdi._data
        assert list(arr) == list(tdi)

        # Check that Index.__new__ knows what to do with TimedeltaArray
        tdi2 = pd.Index(arr)
        assert isinstance(tdi2, TimedeltaIndex)
        assert list(tdi2) == list(arr)

    def test_astype_object(self):
        tdi = TimedeltaIndex(["1 Day", "3 Hours"])
        arr = tdi._data
        asobj = arr.astype("O")
        assert isinstance(asobj, np.ndarray)
        assert asobj.dtype == "O"
        assert list(asobj) == list(tdi)

    def test_to_pytimedelta(self, timedelta_index):
        tdi = timedelta_index
        arr = tdi._data

        expected = tdi.to_pytimedelta()
        result = arr.to_pytimedelta()

        tm.assert_numpy_array_equal(result, expected)

    def test_total_seconds(self, timedelta_index):
        tdi = timedelta_index
        arr = tdi._data

        expected = tdi.total_seconds()
        result = arr.total_seconds()

        tm.assert_numpy_array_equal(result, expected.values)

    @pytest.mark.parametrize("propname", TimedeltaArray._field_ops)
    def test_int_properties(self, timedelta_index, propname):
        tdi = timedelta_index
        arr = tdi._data

        result = getattr(arr, propname)
        expected = np.array(getattr(tdi, propname), dtype=result.dtype)

        tm.assert_numpy_array_equal(result, expected)

    def test_array_interface(self, timedelta_index):
        arr = timedelta_index._data
        copy_false = None if np_version_gt2 else False

        # default asarray gives the same underlying data
        result = np.asarray(arr)
        expected = arr._ndarray
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, copy=copy_false)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)

        # specifying m8[ns] gives the same result as default
        result = np.asarray(arr, dtype="timedelta64[ns]")
        expected = arr._ndarray
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype="timedelta64[ns]", copy=copy_false)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype="timedelta64[ns]")
        if not np_version_gt2:
            # TODO: GH 57739
            assert result is not expected
        tm.assert_numpy_array_equal(result, expected)

        # to object dtype
        result = np.asarray(arr, dtype=object)
        expected = np.array(list(arr), dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        # to other dtype always copies
        result = np.asarray(arr, dtype="int64")
        assert result is not arr.asi8
        assert not np.may_share_memory(arr, result)
        expected = arr.asi8.copy()
        tm.assert_numpy_array_equal(result, expected)

        # other dtypes handled by numpy
        for dtype in ["float64", str]:
            result = np.asarray(arr, dtype=dtype)
            expected = np.asarray(arr).astype(dtype)
            tm.assert_numpy_array_equal(result, expected)

    def test_take_fill_valid(self, timedelta_index, fixed_now_ts):
        tdi = timedelta_index
        arr = tdi._data

        td1 = pd.Timedelta(days=1)
        result = arr.take([-1, 1], allow_fill=True, fill_value=td1)
        assert result[0] == td1

        value = fixed_now_ts
        msg = f"value should be a '{arr._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            # fill_value Timestamp invalid
            arr.take([0, 1], allow_fill=True, fill_value=value)

        value = fixed_now_ts.to_period("D")
        with pytest.raises(TypeError, match=msg):
            # fill_value Period invalid
            arr.take([0, 1], allow_fill=True, fill_value=value)

        value = np.datetime64("NaT", "ns")
        with pytest.raises(TypeError, match=msg):
            # require appropriate-dtype if we have a NA value
            arr.take([-1, 1], allow_fill=True, fill_value=value)


@pytest.mark.filterwarnings(r"ignore:Period with BDay freq is deprecated:FutureWarning")
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
class TestPeriodArray(SharedTests):
    index_cls = PeriodIndex
    array_cls = PeriodArray
    scalar_type = Period
    example_dtype = PeriodIndex([], freq="W").dtype

    @pytest.fixture
    def arr1d(self, period_index):
        """
        Fixture returning DatetimeArray from parametrized PeriodIndex objects
        """
        return period_index._data

    def test_from_pi(self, arr1d):
        pi = self.index_cls(arr1d)
        arr = arr1d
        assert list(arr) == list(pi)

        # Check that Index.__new__ knows what to do with PeriodArray
        pi2 = pd.Index(arr)
        assert isinstance(pi2, PeriodIndex)
        assert list(pi2) == list(arr)

    def test_astype_object(self, arr1d):
        pi = self.index_cls(arr1d)
        arr = arr1d
        asobj = arr.astype("O")
        assert isinstance(asobj, np.ndarray)
        assert asobj.dtype == "O"
        assert list(asobj) == list(pi)

    def test_take_fill_valid(self, arr1d):
        arr = arr1d

        value = NaT._value
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            # require NaT, not iNaT, as it could be confused with an integer
            arr.take([-1, 1], allow_fill=True, fill_value=value)

        value = np.timedelta64("NaT", "ns")
        with pytest.raises(TypeError, match=msg):
            # require appropriate-dtype if we have a NA value
            arr.take([-1, 1], allow_fill=True, fill_value=value)

    @pytest.mark.parametrize("how", ["S", "E"])
    def test_to_timestamp(self, how, arr1d):
        pi = self.index_cls(arr1d)
        arr = arr1d

        expected = DatetimeIndex(pi.to_timestamp(how=how))._data
        result = arr.to_timestamp(how=how)
        assert isinstance(result, DatetimeArray)

        tm.assert_equal(result, expected)

    def test_to_timestamp_roundtrip_bday(self):
        # Case where infer_freq inside would choose "D" instead of "B"
        dta = pd.date_range("2021-10-18", periods=3, freq="B")._data
        parr = dta.to_period()
        result = parr.to_timestamp()
        assert result.freq == "B"
        tm.assert_extension_array_equal(result, dta)

        dta2 = dta[::2]
        parr2 = dta2.to_period()
        result2 = parr2.to_timestamp()
        assert result2.freq == "2B"
        tm.assert_extension_array_equal(result2, dta2)

        parr3 = dta.to_period("2B")
        result3 = parr3.to_timestamp()
        assert result3.freq == "B"
        tm.assert_extension_array_equal(result3, dta)

    def test_to_timestamp_out_of_bounds(self):
        # GH#19643 previously overflowed silently
        pi = pd.period_range("1500", freq="Y", periods=3)
        msg = "Out of bounds nanosecond timestamp: 1500-01-01 00:00:00"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            pi.to_timestamp()

        with pytest.raises(OutOfBoundsDatetime, match=msg):
            pi._data.to_timestamp()

    @pytest.mark.parametrize("propname", PeriodArray._bool_ops)
    def test_bool_properties(self, arr1d, propname):
        # in this case _bool_ops is just `is_leap_year`
        pi = self.index_cls(arr1d)
        arr = arr1d

        result = getattr(arr, propname)
        expected = np.array(getattr(pi, propname))

        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("propname", PeriodArray._field_ops)
    def test_int_properties(self, arr1d, propname):
        pi = self.index_cls(arr1d)
        arr = arr1d

        result = getattr(arr, propname)
        expected = np.array(getattr(pi, propname))

        tm.assert_numpy_array_equal(result, expected)

    def test_array_interface(self, arr1d):
        arr = arr1d

        # default asarray gives objects
        result = np.asarray(arr)
        expected = np.array(list(arr), dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        # to object dtype (same as default)
        result = np.asarray(arr, dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        # to int64 gives the underlying representation
        result = np.asarray(arr, dtype="int64")
        tm.assert_numpy_array_equal(result, arr.asi8)

        result2 = np.asarray(arr, dtype="int64")
        assert np.may_share_memory(result, result2)

        result_copy1 = np.array(arr, dtype="int64", copy=True)
        result_copy2 = np.array(arr, dtype="int64", copy=True)
        assert not np.may_share_memory(result_copy1, result_copy2)

        # to other dtypes
        msg = r"float\(\) argument must be a string or a( real)? number, not 'Period'"
        with pytest.raises(TypeError, match=msg):
            np.asarray(arr, dtype="float64")

        result = np.asarray(arr, dtype="S20")
        expected = np.asarray(arr).astype("S20")
        tm.assert_numpy_array_equal(result, expected)

    def test_strftime(self, arr1d, using_infer_string):
        arr = arr1d

        result = arr.strftime("%Y")
        expected = np.array([per.strftime("%Y") for per in arr], dtype=object)
        if using_infer_string:
            expected = pd.array(expected, dtype=pd.StringDtype(na_value=np.nan))
        tm.assert_equal(result, expected)

    def test_strftime_nat(self, using_infer_string):
        # GH 29578
        arr = PeriodArray(PeriodIndex(["2019-01-01", NaT], dtype="period[D]"))

        result = arr.strftime("%Y-%m-%d")
        expected = np.array(["2019-01-01", np.nan], dtype=object)
        if using_infer_string:
            expected = pd.array(expected, dtype=pd.StringDtype(na_value=np.nan))
        tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "arr,casting_nats",
    [
        (
            TimedeltaIndex(["1 Day", "3 Hours", "NaT"])._data,
            (NaT, np.timedelta64("NaT", "ns")),
        ),
        (
            pd.date_range("2000-01-01", periods=3, freq="D")._data,
            (NaT, np.datetime64("NaT", "ns")),
        ),
        (pd.period_range("2000-01-01", periods=3, freq="D")._data, (NaT,)),
    ],
    ids=lambda x: type(x).__name__,
)
def test_casting_nat_setitem_array(arr, casting_nats):
    expected = type(arr)._from_sequence([NaT, arr[1], arr[2]], dtype=arr.dtype)

    for nat in casting_nats:
        arr = arr.copy()
        arr[0] = nat
        tm.assert_equal(arr, expected)


@pytest.mark.parametrize(
    "arr,non_casting_nats",
    [
        (
            TimedeltaIndex(["1 Day", "3 Hours", "NaT"])._data,
            (np.datetime64("NaT", "ns"), NaT._value),
        ),
        (
            pd.date_range("2000-01-01", periods=3, freq="D")._data,
            (np.timedelta64("NaT", "ns"), NaT._value),
        ),
        (
            pd.period_range("2000-01-01", periods=3, freq="D")._data,
            (np.datetime64("NaT", "ns"), np.timedelta64("NaT", "ns"), NaT._value),
        ),
    ],
    ids=lambda x: type(x).__name__,
)
def test_invalid_nat_setitem_array(arr, non_casting_nats):
    msg = (
        "value should be a '(Timestamp|Timedelta|Period)', 'NaT', or array of those. "
        "Got '(timedelta64|datetime64|int)' instead."
    )

    for nat in non_casting_nats:
        with pytest.raises(TypeError, match=msg):
            arr[0] = nat


@pytest.mark.parametrize(
    "arr",
    [
        pd.date_range("2000", periods=4).array,
        pd.timedelta_range("2000", periods=4).array,
    ],
)
def test_to_numpy_extra(arr):
    arr[0] = NaT
    original = arr.copy()

    result = arr.to_numpy()
    assert np.isnan(result[0])

    result = arr.to_numpy(dtype="int64")
    assert result[0] == -9223372036854775808

    result = arr.to_numpy(dtype="int64", na_value=0)
    assert result[0] == 0

    result = arr.to_numpy(na_value=arr[1].to_numpy())
    assert result[0] == result[1]

    result = arr.to_numpy(na_value=arr[1].to_numpy(copy=False))
    assert result[0] == result[1]

    tm.assert_equal(arr, original)


@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize(
    "values",
    [
        pd.to_datetime(["2020-01-01", "2020-02-01"]),
        pd.to_timedelta([1, 2], unit="D"),
        PeriodIndex(["2020-01-01", "2020-02-01"], freq="D"),
    ],
)
@pytest.mark.parametrize(
    "klass",
    [
        list,
        np.array,
        pd.array,
        pd.Series,
        pd.Index,
        pd.Categorical,
        pd.CategoricalIndex,
    ],
)
def test_searchsorted_datetimelike_with_listlike(values, klass, as_index):
    # https://github.com/pandas-dev/pandas/issues/32762
    if not as_index:
        values = values._data

    result = values.searchsorted(klass(values))
    expected = np.array([0, 1], dtype=result.dtype)

    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        pd.to_datetime(["2020-01-01", "2020-02-01"]),
        pd.to_timedelta([1, 2], unit="D"),
        PeriodIndex(["2020-01-01", "2020-02-01"], freq="D"),
    ],
)
@pytest.mark.parametrize(
    "arg", [[1, 2], ["a", "b"], [Timestamp("2020-01-01", tz="Europe/London")] * 2]
)
def test_searchsorted_datetimelike_with_listlike_invalid_dtype(values, arg):
    # https://github.com/pandas-dev/pandas/issues/32762
    msg = "[Unexpected type|Cannot compare]"
    with pytest.raises(TypeError, match=msg):
        values.searchsorted(arg)


@pytest.mark.parametrize("klass", [list, tuple, np.array, pd.Series])
def test_period_index_construction_from_strings(klass):
    # https://github.com/pandas-dev/pandas/issues/26109
    strings = ["2020Q1", "2020Q2"] * 2
    data = klass(strings)
    result = PeriodIndex(data, freq="Q")
    expected = PeriodIndex([Period(s) for s in strings])
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
def test_from_pandas_array(dtype):
    # GH#24615
    data = np.array([1, 2, 3], dtype=dtype)
    arr = NumpyExtensionArray(data)

    cls = {"M8[ns]": DatetimeArray, "m8[ns]": TimedeltaArray}[dtype]

    depr_msg = f"{cls.__name__}.__init__ is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        result = cls(arr)
        expected = cls(data)
    tm.assert_extension_array_equal(result, expected)

    result = cls._from_sequence(arr, dtype=dtype)
    expected = cls._from_sequence(data, dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    func = {"M8[ns]": pd.to_datetime, "m8[ns]": pd.to_timedelta}[dtype]
    result = func(arr).array
    expected = func(data).array
    tm.assert_equal(result, expected)

    # Let's check the Indexes while we're here
    idx_cls = {"M8[ns]": DatetimeIndex, "m8[ns]": TimedeltaIndex}[dtype]
    result = idx_cls(arr)
    expected = idx_cls(data)
    tm.assert_index_equal(result, expected)