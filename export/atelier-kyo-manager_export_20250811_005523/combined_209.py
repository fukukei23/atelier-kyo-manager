
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\categorical\test_repr.py ===
import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas import (
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    Index,
    Series,
    date_range,
    option_context,
    period_range,
    timedelta_range,
)


class TestCategoricalReprWithFactor:
    def test_print(self, using_infer_string):
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"], ordered=True)
        if using_infer_string:
            expected = [
                "['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c']",
                "Categories (3, str): [a < b < c]",
            ]
        else:
            expected = [
                "['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c']",
                "Categories (3, object): ['a' < 'b' < 'c']",
            ]
        expected = "\n".join(expected)
        actual = repr(factor)
        assert actual == expected


class TestCategoricalRepr:
    def test_big_print(self):
        codes = np.array([0, 1, 2, 0, 1, 2] * 100)
        dtype = CategoricalDtype(categories=Index(["a", "b", "c"], dtype=object))
        factor = Categorical.from_codes(codes, dtype=dtype)
        expected = [
            "['a', 'b', 'c', 'a', 'b', ..., 'b', 'c', 'a', 'b', 'c']",
            "Length: 600",
            "Categories (3, object): ['a', 'b', 'c']",
        ]
        expected = "\n".join(expected)

        actual = repr(factor)

        assert actual == expected

    def test_empty_print(self):
        factor = Categorical([], Index(["a", "b", "c"], dtype=object))
        expected = "[], Categories (3, object): ['a', 'b', 'c']"
        actual = repr(factor)
        assert actual == expected

        assert expected == actual
        factor = Categorical([], Index(["a", "b", "c"], dtype=object), ordered=True)
        expected = "[], Categories (3, object): ['a' < 'b' < 'c']"
        actual = repr(factor)
        assert expected == actual

        factor = Categorical([], [])
        expected = "[], Categories (0, object): []"
        assert expected == repr(factor)

    def test_print_none_width(self):
        # GH10087
        a = Series(Categorical([1, 2, 3, 4]))
        exp = (
            "0    1\n1    2\n2    3\n3    4\n"
            "dtype: category\nCategories (4, int64): [1, 2, 3, 4]"
        )

        with option_context("display.width", None):
            assert exp == repr(a)

    @pytest.mark.skipif(
        using_string_dtype(),
        reason="Change once infer_string is set to True by default",
    )
    def test_unicode_print(self):
        c = Categorical(["aaaaa", "bb", "cccc"] * 20)
        expected = """\
['aaaaa', 'bb', 'cccc', 'aaaaa', 'bb', ..., 'bb', 'cccc', 'aaaaa', 'bb', 'cccc']
Length: 60
Categories (3, object): ['aaaaa', 'bb', 'cccc']"""

        assert repr(c) == expected

        c = Categorical(["ああああ", "いいいいい", "ううううううう"] * 20)
        expected = """\
['ああああ', 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', ..., 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', 'ううううううう']
Length: 60
Categories (3, object): ['ああああ', 'いいいいい', 'ううううううう']"""  # noqa: E501

        assert repr(c) == expected

        # unicode option should not affect to Categorical, as it doesn't care
        # the repr width
        with option_context("display.unicode.east_asian_width", True):
            c = Categorical(["ああああ", "いいいいい", "ううううううう"] * 20)
            expected = """['ああああ', 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', ..., 'いいいいい', 'ううううううう', 'ああああ', 'いいいいい', 'ううううううう']
Length: 60
Categories (3, object): ['ああああ', 'いいいいい', 'ううううううう']"""  # noqa: E501

            assert repr(c) == expected

    def test_categorical_repr(self):
        c = Categorical([1, 2, 3])
        exp = """[1, 2, 3]
Categories (3, int64): [1, 2, 3]"""

        assert repr(c) == exp

        c = Categorical([1, 2, 3, 1, 2, 3], categories=[1, 2, 3])
        exp = """[1, 2, 3, 1, 2, 3]
Categories (3, int64): [1, 2, 3]"""

        assert repr(c) == exp

        c = Categorical([1, 2, 3, 4, 5] * 10)
        exp = """[1, 2, 3, 4, 5, ..., 1, 2, 3, 4, 5]
Length: 50
Categories (5, int64): [1, 2, 3, 4, 5]"""

        assert repr(c) == exp

        c = Categorical(np.arange(20, dtype=np.int64))
        exp = """[0, 1, 2, 3, 4, ..., 15, 16, 17, 18, 19]
Length: 20
Categories (20, int64): [0, 1, 2, 3, ..., 16, 17, 18, 19]"""

        assert repr(c) == exp

    def test_categorical_repr_ordered(self):
        c = Categorical([1, 2, 3], ordered=True)
        exp = """[1, 2, 3]
Categories (3, int64): [1 < 2 < 3]"""

        assert repr(c) == exp

        c = Categorical([1, 2, 3, 1, 2, 3], categories=[1, 2, 3], ordered=True)
        exp = """[1, 2, 3, 1, 2, 3]
Categories (3, int64): [1 < 2 < 3]"""

        assert repr(c) == exp

        c = Categorical([1, 2, 3, 4, 5] * 10, ordered=True)
        exp = """[1, 2, 3, 4, 5, ..., 1, 2, 3, 4, 5]
Length: 50
Categories (5, int64): [1 < 2 < 3 < 4 < 5]"""

        assert repr(c) == exp

        c = Categorical(np.arange(20, dtype=np.int64), ordered=True)
        exp = """[0, 1, 2, 3, 4, ..., 15, 16, 17, 18, 19]
Length: 20
Categories (20, int64): [0 < 1 < 2 < 3 ... 16 < 17 < 18 < 19]"""

        assert repr(c) == exp

    def test_categorical_repr_datetime(self):
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        c = Categorical(idx)

        exp = (
            "[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, "
            "2011-01-01 12:00:00, 2011-01-01 13:00:00]\n"
            "Categories (5, datetime64[ns]): [2011-01-01 09:00:00, "
            "2011-01-01 10:00:00, 2011-01-01 11:00:00,\n"
            "                                 2011-01-01 12:00:00, "
            "2011-01-01 13:00:00]"
            ""
        )
        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = (
            "[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, "
            "2011-01-01 12:00:00, 2011-01-01 13:00:00, 2011-01-01 09:00:00, "
            "2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, "
            "2011-01-01 13:00:00]\n"
            "Categories (5, datetime64[ns]): [2011-01-01 09:00:00, "
            "2011-01-01 10:00:00, 2011-01-01 11:00:00,\n"
            "                                 2011-01-01 12:00:00, "
            "2011-01-01 13:00:00]"
        )

        assert repr(c) == exp

        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        c = Categorical(idx)
        exp = (
            "[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, "
            "2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, "
            "2011-01-01 13:00:00-05:00]\n"
            "Categories (5, datetime64[ns, US/Eastern]): "
            "[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00,\n"
            "                                             "
            "2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00,\n"
            "                                             "
            "2011-01-01 13:00:00-05:00]"
        )

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = (
            "[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, "
            "2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, "
            "2011-01-01 13:00:00-05:00, 2011-01-01 09:00:00-05:00, "
            "2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, "
            "2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00]\n"
            "Categories (5, datetime64[ns, US/Eastern]): "
            "[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00,\n"
            "                                             "
            "2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00,\n"
            "                                             "
            "2011-01-01 13:00:00-05:00]"
        )

        assert repr(c) == exp

    def test_categorical_repr_datetime_ordered(self):
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        c = Categorical(idx, ordered=True)
        exp = """[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00]
Categories (5, datetime64[ns]): [2011-01-01 09:00:00 < 2011-01-01 10:00:00 < 2011-01-01 11:00:00 <
                                 2011-01-01 12:00:00 < 2011-01-01 13:00:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00, 2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00]
Categories (5, datetime64[ns]): [2011-01-01 09:00:00 < 2011-01-01 10:00:00 < 2011-01-01 11:00:00 <
                                 2011-01-01 12:00:00 < 2011-01-01 13:00:00]"""  # noqa: E501

        assert repr(c) == exp

        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        c = Categorical(idx, ordered=True)
        exp = """[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00]
Categories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00 < 2011-01-01 10:00:00-05:00 <
                                             2011-01-01 11:00:00-05:00 < 2011-01-01 12:00:00-05:00 <
                                             2011-01-01 13:00:00-05:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00, 2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00]
Categories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00 < 2011-01-01 10:00:00-05:00 <
                                             2011-01-01 11:00:00-05:00 < 2011-01-01 12:00:00-05:00 <
                                             2011-01-01 13:00:00-05:00]"""  # noqa: E501

        assert repr(c) == exp

    def test_categorical_repr_int_with_nan(self):
        c = Categorical([1, 2, np.nan])
        c_exp = """[1, 2, NaN]\nCategories (2, int64): [1, 2]"""
        assert repr(c) == c_exp

        s = Series([1, 2, np.nan], dtype="object").astype("category")
        s_exp = """0      1\n1      2\n2    NaN
dtype: category
Categories (2, int64): [1, 2]"""
        assert repr(s) == s_exp

    def test_categorical_repr_period(self):
        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        c = Categorical(idx)
        exp = """[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00]
Categories (5, period[h]): [2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00,
                            2011-01-01 13:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = """[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00, 2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00]
Categories (5, period[h]): [2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00,
                            2011-01-01 13:00]"""  # noqa: E501

        assert repr(c) == exp

        idx = period_range("2011-01", freq="M", periods=5)
        c = Categorical(idx)
        exp = """[2011-01, 2011-02, 2011-03, 2011-04, 2011-05]
Categories (5, period[M]): [2011-01, 2011-02, 2011-03, 2011-04, 2011-05]"""

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = """[2011-01, 2011-02, 2011-03, 2011-04, 2011-05, 2011-01, 2011-02, 2011-03, 2011-04, 2011-05]
Categories (5, period[M]): [2011-01, 2011-02, 2011-03, 2011-04, 2011-05]"""  # noqa: E501

        assert repr(c) == exp

    def test_categorical_repr_period_ordered(self):
        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        c = Categorical(idx, ordered=True)
        exp = """[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00]
Categories (5, period[h]): [2011-01-01 09:00 < 2011-01-01 10:00 < 2011-01-01 11:00 < 2011-01-01 12:00 <
                            2011-01-01 13:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00, 2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00]
Categories (5, period[h]): [2011-01-01 09:00 < 2011-01-01 10:00 < 2011-01-01 11:00 < 2011-01-01 12:00 <
                            2011-01-01 13:00]"""  # noqa: E501

        assert repr(c) == exp

        idx = period_range("2011-01", freq="M", periods=5)
        c = Categorical(idx, ordered=True)
        exp = """[2011-01, 2011-02, 2011-03, 2011-04, 2011-05]
Categories (5, period[M]): [2011-01 < 2011-02 < 2011-03 < 2011-04 < 2011-05]"""

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[2011-01, 2011-02, 2011-03, 2011-04, 2011-05, 2011-01, 2011-02, 2011-03, 2011-04, 2011-05]
Categories (5, period[M]): [2011-01 < 2011-02 < 2011-03 < 2011-04 < 2011-05]"""  # noqa: E501

        assert repr(c) == exp

    def test_categorical_repr_timedelta(self):
        idx = timedelta_range("1 days", periods=5)
        c = Categorical(idx)
        exp = """[1 days, 2 days, 3 days, 4 days, 5 days]
Categories (5, timedelta64[ns]): [1 days, 2 days, 3 days, 4 days, 5 days]"""

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = """[1 days, 2 days, 3 days, 4 days, 5 days, 1 days, 2 days, 3 days, 4 days, 5 days]
Categories (5, timedelta64[ns]): [1 days, 2 days, 3 days, 4 days, 5 days]"""  # noqa: E501

        assert repr(c) == exp

        idx = timedelta_range("1 hours", periods=20)
        c = Categorical(idx)
        exp = """[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, 4 days 01:00:00, ..., 15 days 01:00:00, 16 days 01:00:00, 17 days 01:00:00, 18 days 01:00:00, 19 days 01:00:00]
Length: 20
Categories (20, timedelta64[ns]): [0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00,
                                   3 days 01:00:00, ..., 16 days 01:00:00, 17 days 01:00:00,
                                   18 days 01:00:00, 19 days 01:00:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx)
        exp = """[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, 4 days 01:00:00, ..., 15 days 01:00:00, 16 days 01:00:00, 17 days 01:00:00, 18 days 01:00:00, 19 days 01:00:00]
Length: 40
Categories (20, timedelta64[ns]): [0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00,
                                   3 days 01:00:00, ..., 16 days 01:00:00, 17 days 01:00:00,
                                   18 days 01:00:00, 19 days 01:00:00]"""  # noqa: E501

        assert repr(c) == exp

    def test_categorical_repr_timedelta_ordered(self):
        idx = timedelta_range("1 days", periods=5)
        c = Categorical(idx, ordered=True)
        exp = """[1 days, 2 days, 3 days, 4 days, 5 days]
Categories (5, timedelta64[ns]): [1 days < 2 days < 3 days < 4 days < 5 days]"""

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[1 days, 2 days, 3 days, 4 days, 5 days, 1 days, 2 days, 3 days, 4 days, 5 days]
Categories (5, timedelta64[ns]): [1 days < 2 days < 3 days < 4 days < 5 days]"""  # noqa: E501

        assert repr(c) == exp

        idx = timedelta_range("1 hours", periods=20)
        c = Categorical(idx, ordered=True)
        exp = """[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, 4 days 01:00:00, ..., 15 days 01:00:00, 16 days 01:00:00, 17 days 01:00:00, 18 days 01:00:00, 19 days 01:00:00]
Length: 20
Categories (20, timedelta64[ns]): [0 days 01:00:00 < 1 days 01:00:00 < 2 days 01:00:00 <
                                   3 days 01:00:00 ... 16 days 01:00:00 < 17 days 01:00:00 <
                                   18 days 01:00:00 < 19 days 01:00:00]"""  # noqa: E501

        assert repr(c) == exp

        c = Categorical(idx.append(idx), categories=idx, ordered=True)
        exp = """[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, 4 days 01:00:00, ..., 15 days 01:00:00, 16 days 01:00:00, 17 days 01:00:00, 18 days 01:00:00, 19 days 01:00:00]
Length: 40
Categories (20, timedelta64[ns]): [0 days 01:00:00 < 1 days 01:00:00 < 2 days 01:00:00 <
                                   3 days 01:00:00 ... 16 days 01:00:00 < 17 days 01:00:00 <
                                   18 days 01:00:00 < 19 days 01:00:00]"""  # noqa: E501

        assert repr(c) == exp

    def test_categorical_index_repr(self):
        idx = CategoricalIndex(Categorical([1, 2, 3]))
        exp = """CategoricalIndex([1, 2, 3], categories=[1, 2, 3], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(idx) == exp

        i = CategoricalIndex(Categorical(np.arange(10, dtype=np.int64)))
        exp = """CategoricalIndex([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], categories=[0, 1, 2, 3, ..., 6, 7, 8, 9], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

    def test_categorical_index_repr_ordered(self):
        i = CategoricalIndex(Categorical([1, 2, 3], ordered=True))
        exp = """CategoricalIndex([1, 2, 3], categories=[1, 2, 3], ordered=True, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        i = CategoricalIndex(Categorical(np.arange(10, dtype=np.int64), ordered=True))
        exp = """CategoricalIndex([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], categories=[0, 1, 2, 3, ..., 6, 7, 8, 9], ordered=True, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

    def test_categorical_index_repr_datetime(self):
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00:00', '2011-01-01 10:00:00',
                  '2011-01-01 11:00:00', '2011-01-01 12:00:00',
                  '2011-01-01 13:00:00'],
                 categories=[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00:00-05:00', '2011-01-01 10:00:00-05:00',
                  '2011-01-01 11:00:00-05:00', '2011-01-01 12:00:00-05:00',
                  '2011-01-01 13:00:00-05:00'],
                 categories=[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

    def test_categorical_index_repr_datetime_ordered(self):
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['2011-01-01 09:00:00', '2011-01-01 10:00:00',
                  '2011-01-01 11:00:00', '2011-01-01 12:00:00',
                  '2011-01-01 13:00:00'],
                 categories=[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00], ordered=True, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['2011-01-01 09:00:00-05:00', '2011-01-01 10:00:00-05:00',
                  '2011-01-01 11:00:00-05:00', '2011-01-01 12:00:00-05:00',
                  '2011-01-01 13:00:00-05:00'],
                 categories=[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00], ordered=True, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        i = CategoricalIndex(Categorical(idx.append(idx), ordered=True))
        exp = """CategoricalIndex(['2011-01-01 09:00:00-05:00', '2011-01-01 10:00:00-05:00',
                  '2011-01-01 11:00:00-05:00', '2011-01-01 12:00:00-05:00',
                  '2011-01-01 13:00:00-05:00', '2011-01-01 09:00:00-05:00',
                  '2011-01-01 10:00:00-05:00', '2011-01-01 11:00:00-05:00',
                  '2011-01-01 12:00:00-05:00', '2011-01-01 13:00:00-05:00'],
                 categories=[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00], ordered=True, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

    def test_categorical_index_repr_period(self):
        # test all length
        idx = period_range("2011-01-01 09:00", freq="h", periods=1)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00'], categories=[2011-01-01 09:00], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        idx = period_range("2011-01-01 09:00", freq="h", periods=2)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00'], categories=[2011-01-01 09:00, 2011-01-01 10:00], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        idx = period_range("2011-01-01 09:00", freq="h", periods=3)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00'], categories=[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00',
                  '2011-01-01 12:00', '2011-01-01 13:00'],
                 categories=[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        i = CategoricalIndex(Categorical(idx.append(idx)))
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00',
                  '2011-01-01 12:00', '2011-01-01 13:00', '2011-01-01 09:00',
                  '2011-01-01 10:00', '2011-01-01 11:00', '2011-01-01 12:00',
                  '2011-01-01 13:00'],
                 categories=[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        idx = period_range("2011-01", freq="M", periods=5)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['2011-01', '2011-02', '2011-03', '2011-04', '2011-05'], categories=[2011-01, 2011-02, 2011-03, 2011-04, 2011-05], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

    def test_categorical_index_repr_period_ordered(self):
        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00',
                  '2011-01-01 12:00', '2011-01-01 13:00'],
                 categories=[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00], ordered=True, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

        idx = period_range("2011-01", freq="M", periods=5)
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['2011-01', '2011-02', '2011-03', '2011-04', '2011-05'], categories=[2011-01, 2011-02, 2011-03, 2011-04, 2011-05], ordered=True, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

    def test_categorical_index_repr_timedelta(self):
        idx = timedelta_range("1 days", periods=5)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['1 days', '2 days', '3 days', '4 days', '5 days'], categories=[1 days, 2 days, 3 days, 4 days, 5 days], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        idx = timedelta_range("1 hours", periods=10)
        i = CategoricalIndex(Categorical(idx))
        exp = """CategoricalIndex(['0 days 01:00:00', '1 days 01:00:00', '2 days 01:00:00',
                  '3 days 01:00:00', '4 days 01:00:00', '5 days 01:00:00',
                  '6 days 01:00:00', '7 days 01:00:00', '8 days 01:00:00',
                  '9 days 01:00:00'],
                 categories=[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, ..., 6 days 01:00:00, 7 days 01:00:00, 8 days 01:00:00, 9 days 01:00:00], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

    def test_categorical_index_repr_timedelta_ordered(self):
        idx = timedelta_range("1 days", periods=5)
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['1 days', '2 days', '3 days', '4 days', '5 days'], categories=[1 days, 2 days, 3 days, 4 days, 5 days], ordered=True, dtype='category')"""  # noqa: E501
        assert repr(i) == exp

        idx = timedelta_range("1 hours", periods=10)
        i = CategoricalIndex(Categorical(idx, ordered=True))
        exp = """CategoricalIndex(['0 days 01:00:00', '1 days 01:00:00', '2 days 01:00:00',
                  '3 days 01:00:00', '4 days 01:00:00', '5 days 01:00:00',
                  '6 days 01:00:00', '7 days 01:00:00', '8 days 01:00:00',
                  '9 days 01:00:00'],
                 categories=[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, ..., 6 days 01:00:00, 7 days 01:00:00, 8 days 01:00:00, 9 days 01:00:00], ordered=True, dtype='category')"""  # noqa: E501

        assert repr(i) == exp

    def test_categorical_str_repr(self):
        # GH 33676
        result = repr(Categorical([1, "2", 3, 4]))
        expected = "[1, '2', 3, 4]\nCategories (4, object): [1, 3, 4, '2']"
        assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_drop.py ===
import re

import numpy as np
import pytest

from pandas.errors import PerformanceWarning

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "msg,labels,level",
    [
        (r"labels \[4\] not found in level", 4, "a"),
        (r"labels \[7\] not found in level", 7, "b"),
    ],
)
def test_drop_raise_exception_if_labels_not_in_level(msg, labels, level):
    # GH 8594
    mi = MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]], names=["a", "b"])
    s = Series([10, 20, 30], index=mi)
    df = DataFrame([10, 20, 30], index=mi)

    with pytest.raises(KeyError, match=msg):
        s.drop(labels, level=level)
    with pytest.raises(KeyError, match=msg):
        df.drop(labels, level=level)


@pytest.mark.parametrize("labels,level", [(4, "a"), (7, "b")])
def test_drop_errors_ignore(labels, level):
    # GH 8594
    mi = MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]], names=["a", "b"])
    s = Series([10, 20, 30], index=mi)
    df = DataFrame([10, 20, 30], index=mi)

    expected_s = s.drop(labels, level=level, errors="ignore")
    tm.assert_series_equal(s, expected_s)

    expected_df = df.drop(labels, level=level, errors="ignore")
    tm.assert_frame_equal(df, expected_df)


def test_drop_with_non_unique_datetime_index_and_invalid_keys():
    # GH 30399

    # define dataframe with unique datetime index
    df = DataFrame(
        np.random.default_rng(2).standard_normal((5, 3)),
        columns=["a", "b", "c"],
        index=pd.date_range("2012", freq="h", periods=5),
    )
    # create dataframe with non-unique datetime index
    df = df.iloc[[0, 2, 2, 3]].copy()

    with pytest.raises(KeyError, match="not found in axis"):
        df.drop(["a", "b"])  # Dropping with labels not exist in the index


class TestDataFrameDrop:
    def test_drop_names(self):
        df = DataFrame(
            [[1, 2, 3], [3, 4, 5], [5, 6, 7]],
            index=["a", "b", "c"],
            columns=["d", "e", "f"],
        )
        df.index.name, df.columns.name = "first", "second"
        df_dropped_b = df.drop("b")
        df_dropped_e = df.drop("e", axis=1)
        df_inplace_b, df_inplace_e = df.copy(), df.copy()
        return_value = df_inplace_b.drop("b", inplace=True)
        assert return_value is None
        return_value = df_inplace_e.drop("e", axis=1, inplace=True)
        assert return_value is None
        for obj in (df_dropped_b, df_dropped_e, df_inplace_b, df_inplace_e):
            assert obj.index.name == "first"
            assert obj.columns.name == "second"
        assert list(df.columns) == ["d", "e", "f"]

        msg = r"\['g'\] not found in axis"
        with pytest.raises(KeyError, match=msg):
            df.drop(["g"])
        with pytest.raises(KeyError, match=msg):
            df.drop(["g"], axis=1)

        # errors = 'ignore'
        dropped = df.drop(["g"], errors="ignore")
        expected = Index(["a", "b", "c"], name="first")
        tm.assert_index_equal(dropped.index, expected)

        dropped = df.drop(["b", "g"], errors="ignore")
        expected = Index(["a", "c"], name="first")
        tm.assert_index_equal(dropped.index, expected)

        dropped = df.drop(["g"], axis=1, errors="ignore")
        expected = Index(["d", "e", "f"], name="second")
        tm.assert_index_equal(dropped.columns, expected)

        dropped = df.drop(["d", "g"], axis=1, errors="ignore")
        expected = Index(["e", "f"], name="second")
        tm.assert_index_equal(dropped.columns, expected)

        # GH 16398
        dropped = df.drop([], errors="ignore")
        expected = Index(["a", "b", "c"], name="first")
        tm.assert_index_equal(dropped.index, expected)

    def test_drop(self):
        simple = DataFrame({"A": [1, 2, 3, 4], "B": [0, 1, 2, 3]})
        tm.assert_frame_equal(simple.drop("A", axis=1), simple[["B"]])
        tm.assert_frame_equal(simple.drop(["A", "B"], axis="columns"), simple[[]])
        tm.assert_frame_equal(simple.drop([0, 1, 3], axis=0), simple.loc[[2], :])
        tm.assert_frame_equal(simple.drop([0, 3], axis="index"), simple.loc[[1, 2], :])

        with pytest.raises(KeyError, match=r"\[5\] not found in axis"):
            simple.drop(5)
        with pytest.raises(KeyError, match=r"\['C'\] not found in axis"):
            simple.drop("C", axis=1)
        with pytest.raises(KeyError, match=r"\[5\] not found in axis"):
            simple.drop([1, 5])
        with pytest.raises(KeyError, match=r"\['C'\] not found in axis"):
            simple.drop(["A", "C"], axis=1)

        # GH 42881
        with pytest.raises(KeyError, match=r"\['C', 'D', 'F'\] not found in axis"):
            simple.drop(["C", "D", "F"], axis=1)

        # errors = 'ignore'
        tm.assert_frame_equal(simple.drop(5, errors="ignore"), simple)
        tm.assert_frame_equal(
            simple.drop([0, 5], errors="ignore"), simple.loc[[1, 2, 3], :]
        )
        tm.assert_frame_equal(simple.drop("C", axis=1, errors="ignore"), simple)
        tm.assert_frame_equal(
            simple.drop(["A", "C"], axis=1, errors="ignore"), simple[["B"]]
        )

        # non-unique - wheee!
        nu_df = DataFrame(
            list(zip(range(3), range(-3, 1), list("abc"))), columns=["a", "a", "b"]
        )
        tm.assert_frame_equal(nu_df.drop("a", axis=1), nu_df[["b"]])
        tm.assert_frame_equal(nu_df.drop("b", axis="columns"), nu_df["a"])
        tm.assert_frame_equal(nu_df.drop([]), nu_df)  # GH 16398

        nu_df = nu_df.set_index(Index(["X", "Y", "X"]))
        nu_df.columns = list("abc")
        tm.assert_frame_equal(nu_df.drop("X", axis="rows"), nu_df.loc[["Y"], :])
        tm.assert_frame_equal(nu_df.drop(["X", "Y"], axis=0), nu_df.loc[[], :])

        # inplace cache issue
        # GH#5628
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=list("abc")
        )
        expected = df[~(df.b > 0)]
        return_value = df.drop(labels=df[df.b > 0].index, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(df, expected)

    def test_drop_multiindex_not_lexsorted(self):
        # GH#11640

        # define the lexsorted version
        lexsorted_mi = MultiIndex.from_tuples(
            [("a", ""), ("b1", "c1"), ("b2", "c2")], names=["b", "c"]
        )
        lexsorted_df = DataFrame([[1, 3, 4]], columns=lexsorted_mi)
        assert lexsorted_df.columns._is_lexsorted()

        # define the non-lexsorted version
        not_lexsorted_df = DataFrame(
            columns=["a", "b", "c", "d"], data=[[1, "b1", "c1", 3], [1, "b2", "c2", 4]]
        )
        not_lexsorted_df = not_lexsorted_df.pivot_table(
            index="a", columns=["b", "c"], values="d"
        )
        not_lexsorted_df = not_lexsorted_df.reset_index()
        assert not not_lexsorted_df.columns._is_lexsorted()

        expected = lexsorted_df.drop("a", axis=1).astype(float)
        with tm.assert_produces_warning(PerformanceWarning):
            result = not_lexsorted_df.drop("a", axis=1)

        tm.assert_frame_equal(result, expected)

    def test_drop_api_equivalence(self):
        # equivalence of the labels/axis and index/columns API's (GH#12392)
        df = DataFrame(
            [[1, 2, 3], [3, 4, 5], [5, 6, 7]],
            index=["a", "b", "c"],
            columns=["d", "e", "f"],
        )

        res1 = df.drop("a")
        res2 = df.drop(index="a")
        tm.assert_frame_equal(res1, res2)

        res1 = df.drop("d", axis=1)
        res2 = df.drop(columns="d")
        tm.assert_frame_equal(res1, res2)

        res1 = df.drop(labels="e", axis=1)
        res2 = df.drop(columns="e")
        tm.assert_frame_equal(res1, res2)

        res1 = df.drop(["a"], axis=0)
        res2 = df.drop(index=["a"])
        tm.assert_frame_equal(res1, res2)

        res1 = df.drop(["a"], axis=0).drop(["d"], axis=1)
        res2 = df.drop(index=["a"], columns=["d"])
        tm.assert_frame_equal(res1, res2)

        msg = "Cannot specify both 'labels' and 'index'/'columns'"
        with pytest.raises(ValueError, match=msg):
            df.drop(labels="a", index="b")

        with pytest.raises(ValueError, match=msg):
            df.drop(labels="a", columns="b")

        msg = "Need to specify at least one of 'labels', 'index' or 'columns'"
        with pytest.raises(ValueError, match=msg):
            df.drop(axis=1)

    data = [[1, 2, 3], [1, 2, 3]]

    @pytest.mark.parametrize(
        "actual",
        [
            DataFrame(data=data, index=["a", "a"]),
            DataFrame(data=data, index=["a", "b"]),
            DataFrame(data=data, index=["a", "b"]).set_index([0, 1]),
            DataFrame(data=data, index=["a", "a"]).set_index([0, 1]),
        ],
    )
    def test_raise_on_drop_duplicate_index(self, actual):
        # GH#19186
        level = 0 if isinstance(actual.index, MultiIndex) else None
        msg = re.escape("\"['c'] not found in axis\"")
        with pytest.raises(KeyError, match=msg):
            actual.drop("c", level=level, axis=0)
        with pytest.raises(KeyError, match=msg):
            actual.T.drop("c", level=level, axis=1)
        expected_no_err = actual.drop("c", axis=0, level=level, errors="ignore")
        tm.assert_frame_equal(expected_no_err, actual)
        expected_no_err = actual.T.drop("c", axis=1, level=level, errors="ignore")
        tm.assert_frame_equal(expected_no_err.T, actual)

    @pytest.mark.parametrize("index", [[1, 2, 3], [1, 1, 2]])
    @pytest.mark.parametrize("drop_labels", [[], [1], [2]])
    def test_drop_empty_list(self, index, drop_labels):
        # GH#21494
        expected_index = [i for i in index if i not in drop_labels]
        frame = DataFrame(index=index).drop(drop_labels)
        tm.assert_frame_equal(frame, DataFrame(index=expected_index))

    @pytest.mark.parametrize("index", [[1, 2, 3], [1, 2, 2]])
    @pytest.mark.parametrize("drop_labels", [[1, 4], [4, 5]])
    def test_drop_non_empty_list(self, index, drop_labels):
        # GH# 21494
        with pytest.raises(KeyError, match="not found in axis"):
            DataFrame(index=index).drop(drop_labels)

    @pytest.mark.parametrize(
        "empty_listlike",
        [
            [],
            {},
            np.array([]),
            Series([], dtype="datetime64[ns]"),
            Index([]),
            DatetimeIndex([]),
        ],
    )
    def test_drop_empty_listlike_non_unique_datetime_index(self, empty_listlike):
        # GH#27994
        data = {"column_a": [5, 10], "column_b": ["one", "two"]}
        index = [Timestamp("2021-01-01"), Timestamp("2021-01-01")]
        df = DataFrame(data, index=index)

        # Passing empty list-like should return the same DataFrame.
        expected = df.copy()
        result = df.drop(empty_listlike)
        tm.assert_frame_equal(result, expected)

    def test_mixed_depth_drop(self):
        arrays = [
            ["a", "top", "top", "routine1", "routine1", "routine2"],
            ["", "OD", "OD", "result1", "result2", "result1"],
            ["", "wx", "wy", "", "", ""],
        ]

        tuples = sorted(zip(*arrays))
        index = MultiIndex.from_tuples(tuples)
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 6)), columns=index)

        result = df.drop("a", axis=1)
        expected = df.drop([("a", "", "")], axis=1)
        tm.assert_frame_equal(expected, result)

        result = df.drop(["top"], axis=1)
        expected = df.drop([("top", "OD", "wx")], axis=1)
        expected = expected.drop([("top", "OD", "wy")], axis=1)
        tm.assert_frame_equal(expected, result)

        result = df.drop(("top", "OD", "wx"), axis=1)
        expected = df.drop([("top", "OD", "wx")], axis=1)
        tm.assert_frame_equal(expected, result)

        expected = df.drop([("top", "OD", "wy")], axis=1)
        expected = df.drop("top", axis=1)

        result = df.drop("result1", level=1, axis=1)
        expected = df.drop(
            [("routine1", "result1", ""), ("routine2", "result1", "")], axis=1
        )
        tm.assert_frame_equal(expected, result)

    def test_drop_multiindex_other_level_nan(self):
        # GH#12754
        df = (
            DataFrame(
                {
                    "A": ["one", "one", "two", "two"],
                    "B": [np.nan, 0.0, 1.0, 2.0],
                    "C": ["a", "b", "c", "c"],
                    "D": [1, 2, 3, 4],
                }
            )
            .set_index(["A", "B", "C"])
            .sort_index()
        )
        result = df.drop("c", level="C")
        expected = DataFrame(
            [2, 1],
            columns=["D"],
            index=MultiIndex.from_tuples(
                [("one", 0.0, "b"), ("one", np.nan, "a")], names=["A", "B", "C"]
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_drop_nonunique(self):
        df = DataFrame(
            [
                ["x-a", "x", "a", 1.5],
                ["x-a", "x", "a", 1.2],
                ["z-c", "z", "c", 3.1],
                ["x-a", "x", "a", 4.1],
                ["x-b", "x", "b", 5.1],
                ["x-b", "x", "b", 4.1],
                ["x-b", "x", "b", 2.2],
                ["y-a", "y", "a", 1.2],
                ["z-b", "z", "b", 2.1],
            ],
            columns=["var1", "var2", "var3", "var4"],
        )

        grp_size = df.groupby("var1").size()
        drop_idx = grp_size.loc[grp_size == 1]

        idf = df.set_index(["var1", "var2", "var3"])

        # it works! GH#2101
        result = idf.drop(drop_idx.index, level=0).reset_index()
        expected = df[-df.var1.isin(drop_idx.index)]

        result.index = expected.index

        tm.assert_frame_equal(result, expected)

    def test_drop_level(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data

        result = frame.drop(["bar", "qux"], level="first")
        expected = frame.iloc[[0, 1, 2, 5, 6]]
        tm.assert_frame_equal(result, expected)

        result = frame.drop(["two"], level="second")
        expected = frame.iloc[[0, 2, 3, 6, 7, 9]]
        tm.assert_frame_equal(result, expected)

        result = frame.T.drop(["bar", "qux"], axis=1, level="first")
        expected = frame.iloc[[0, 1, 2, 5, 6]].T
        tm.assert_frame_equal(result, expected)

        result = frame.T.drop(["two"], axis=1, level="second")
        expected = frame.iloc[[0, 2, 3, 6, 7, 9]].T
        tm.assert_frame_equal(result, expected)

    def test_drop_level_nonunique_datetime(self):
        # GH#12701
        idx = Index([2, 3, 4, 4, 5], name="id")
        idxdt = pd.to_datetime(
            [
                "2016-03-23 14:00",
                "2016-03-23 15:00",
                "2016-03-23 16:00",
                "2016-03-23 16:00",
                "2016-03-23 17:00",
            ]
        )
        df = DataFrame(np.arange(10).reshape(5, 2), columns=list("ab"), index=idx)
        df["tstamp"] = idxdt
        df = df.set_index("tstamp", append=True)
        ts = Timestamp("201603231600")
        assert df.index.is_unique is False

        result = df.drop(ts, level="tstamp")
        expected = df.loc[idx != 4]
        tm.assert_frame_equal(result, expected)

    def test_drop_tz_aware_timestamp_across_dst(self, frame_or_series):
        # GH#21761
        start = Timestamp("2017-10-29", tz="Europe/Berlin")
        end = Timestamp("2017-10-29 04:00:00", tz="Europe/Berlin")
        index = pd.date_range(start, end, freq="15min")
        data = frame_or_series(data=[1] * len(index), index=index)
        result = data.drop(start)
        expected_start = Timestamp("2017-10-29 00:15:00", tz="Europe/Berlin")
        expected_idx = pd.date_range(expected_start, end, freq="15min")
        expected = frame_or_series(data=[1] * len(expected_idx), index=expected_idx)
        tm.assert_equal(result, expected)

    def test_drop_preserve_names(self):
        index = MultiIndex.from_arrays(
            [[0, 0, 0, 1, 1, 1], [1, 2, 3, 1, 2, 3]], names=["one", "two"]
        )

        df = DataFrame(np.random.default_rng(2).standard_normal((6, 3)), index=index)

        result = df.drop([(0, 2)])
        assert result.index.names == ("one", "two")

    @pytest.mark.parametrize(
        "operation", ["__iadd__", "__isub__", "__imul__", "__ipow__"]
    )
    @pytest.mark.parametrize("inplace", [False, True])
    def test_inplace_drop_and_operation(self, operation, inplace):
        # GH#30484
        df = DataFrame({"x": range(5)})
        expected = df.copy()
        df["y"] = range(5)
        y = df["y"]

        with tm.assert_produces_warning(None):
            if inplace:
                df.drop("y", axis=1, inplace=inplace)
            else:
                df = df.drop("y", axis=1, inplace=inplace)

            # Perform operation and check result
            getattr(y, operation)(1)
            tm.assert_frame_equal(df, expected)

    def test_drop_with_non_unique_multiindex(self):
        # GH#36293
        mi = MultiIndex.from_arrays([["x", "y", "x"], ["i", "j", "i"]])
        df = DataFrame([1, 2, 3], index=mi)
        result = df.drop(index="x")
        expected = DataFrame([2], index=MultiIndex.from_arrays([["y"], ["j"]]))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("indexer", [("a", "a"), [("a", "a")]])
    def test_drop_tuple_with_non_unique_multiindex(self, indexer):
        # GH#42771
        idx = MultiIndex.from_product([["a", "b"], ["a", "a"]])
        df = DataFrame({"x": range(len(idx))}, index=idx)
        result = df.drop(index=[("a", "a")])
        expected = DataFrame(
            {"x": [2, 3]}, index=MultiIndex.from_tuples([("b", "a"), ("b", "a")])
        )
        tm.assert_frame_equal(result, expected)

    def test_drop_with_duplicate_columns(self):
        df = DataFrame(
            [[1, 5, 7.0], [1, 5, 7.0], [1, 5, 7.0]], columns=["bar", "a", "a"]
        )
        result = df.drop(["a"], axis=1)
        expected = DataFrame([[1], [1], [1]], columns=["bar"])
        tm.assert_frame_equal(result, expected)
        result = df.drop("a", axis=1)
        tm.assert_frame_equal(result, expected)

    def test_drop_with_duplicate_columns2(self):
        # drop buggy GH#6240
        df = DataFrame(
            {
                "A": np.random.default_rng(2).standard_normal(5),
                "B": np.random.default_rng(2).standard_normal(5),
                "C": np.random.default_rng(2).standard_normal(5),
                "D": ["a", "b", "c", "d", "e"],
            }
        )

        expected = df.take([0, 1, 1], axis=1)
        df2 = df.take([2, 0, 1, 2, 1], axis=1)
        result = df2.drop("C", axis=1)
        tm.assert_frame_equal(result, expected)

    def test_drop_inplace_no_leftover_column_reference(self):
        # GH 13934
        df = DataFrame({"a": [1, 2, 3]}, columns=Index(["a"], dtype="object"))
        a = df.a
        df.drop(["a"], axis=1, inplace=True)
        tm.assert_index_equal(df.columns, Index([], dtype="object"))
        a -= a.mean()
        tm.assert_index_equal(df.columns, Index([], dtype="object"))

    def test_drop_level_missing_label_multiindex(self):
        # GH 18561
        df = DataFrame(index=MultiIndex.from_product([range(3), range(3)]))
        with pytest.raises(KeyError, match="labels \\[5\\] not found in level"):
            df.drop(5, level=0)

    @pytest.mark.parametrize("idx, level", [(["a", "b"], 0), (["a"], None)])
    def test_drop_index_ea_dtype(self, any_numeric_ea_dtype, idx, level):
        # GH#45860
        df = DataFrame(
            {"a": [1, 2, 2, pd.NA], "b": 100}, dtype=any_numeric_ea_dtype
        ).set_index(idx)
        result = df.drop(Index([2, pd.NA]), level=level)
        expected = DataFrame(
            {"a": [1], "b": 100}, dtype=any_numeric_ea_dtype
        ).set_index(idx)
        tm.assert_frame_equal(result, expected)

    def test_drop_parse_strings_datetime_index(self):
        # GH #5355
        df = DataFrame(
            {"a": [1, 2], "b": [1, 2]},
            index=[Timestamp("2000-01-03"), Timestamp("2000-01-04")],
        )
        result = df.drop("2000-01-03", axis=0)
        expected = DataFrame({"a": [2], "b": [2]}, index=[Timestamp("2000-01-04")])
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_records.py ===
import collections.abc
import pickle
import textwrap
from io import BytesIO
from os import path
from pathlib import Path

import pytest

import numpy as np
from numpy.testing import (
    assert_,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
    temppath,
)


class TestFromrecords:
    def test_fromrecords(self):
        r = np.rec.fromrecords([[456, 'dbe', 1.2], [2, 'de', 1.3]],
                            names='col1,col2,col3')
        assert_equal(r[0].item(), (456, 'dbe', 1.2))
        assert_equal(r['col1'].dtype.kind, 'i')
        assert_equal(r['col2'].dtype.kind, 'U')
        assert_equal(r['col2'].dtype.itemsize, 12)
        assert_equal(r['col3'].dtype.kind, 'f')

    def test_fromrecords_0len(self):
        """ Verify fromrecords works with a 0-length input """
        dtype = [('a', float), ('b', float)]
        r = np.rec.fromrecords([], dtype=dtype)
        assert_equal(r.shape, (0,))

    def test_fromrecords_2d(self):
        data = [
            [(1, 2), (3, 4), (5, 6)],
            [(6, 5), (4, 3), (2, 1)]
        ]
        expected_a = [[1, 3, 5], [6, 4, 2]]
        expected_b = [[2, 4, 6], [5, 3, 1]]

        # try with dtype
        r1 = np.rec.fromrecords(data, dtype=[('a', int), ('b', int)])
        assert_equal(r1['a'], expected_a)
        assert_equal(r1['b'], expected_b)

        # try with names
        r2 = np.rec.fromrecords(data, names=['a', 'b'])
        assert_equal(r2['a'], expected_a)
        assert_equal(r2['b'], expected_b)

        assert_equal(r1, r2)

    def test_method_array(self):
        r = np.rec.array(
            b'abcdefg' * 100, formats='i2,S3,i4', shape=3, byteorder='big'
        )
        assert_equal(r[1].item(), (25444, b'efg', 1633837924))

    def test_method_array2(self):
        r = np.rec.array(
            [
                (1, 11, 'a'), (2, 22, 'b'), (3, 33, 'c'), (4, 44, 'd'),
                (5, 55, 'ex'), (6, 66, 'f'), (7, 77, 'g')
            ],
            formats='u1,f4,S1'
        )
        assert_equal(r[1].item(), (2, 22.0, b'b'))

    def test_recarray_slices(self):
        r = np.rec.array(
            [
                (1, 11, 'a'), (2, 22, 'b'), (3, 33, 'c'), (4, 44, 'd'),
                (5, 55, 'ex'), (6, 66, 'f'), (7, 77, 'g')
            ],
            formats='u1,f4,S1'
        )
        assert_equal(r[1::2][1].item(), (4, 44.0, b'd'))

    def test_recarray_fromarrays(self):
        x1 = np.array([1, 2, 3, 4])
        x2 = np.array(['a', 'dd', 'xyz', '12'])
        x3 = np.array([1.1, 2, 3, 4])
        r = np.rec.fromarrays([x1, x2, x3], names='a,b,c')
        assert_equal(r[1].item(), (2, 'dd', 2.0))
        x1[1] = 34
        assert_equal(r.a, np.array([1, 2, 3, 4]))

    def test_recarray_fromfile(self):
        data_dir = path.join(path.dirname(__file__), 'data')
        filename = path.join(data_dir, 'recarray_from_file.fits')
        fd = open(filename, 'rb')
        fd.seek(2880 * 2)
        r1 = np.rec.fromfile(fd, formats='f8,i4,S5', shape=3, byteorder='big')
        fd.seek(2880 * 2)
        r2 = np.rec.array(fd, formats='f8,i4,S5', shape=3, byteorder='big')
        fd.seek(2880 * 2)
        bytes_array = BytesIO()
        bytes_array.write(fd.read())
        bytes_array.seek(0)
        r3 = np.rec.fromfile(
            bytes_array, formats='f8,i4,S5', shape=3, byteorder='big'
        )
        fd.close()
        assert_equal(r1, r2)
        assert_equal(r2, r3)

    def test_recarray_from_obj(self):
        count = 10
        a = np.zeros(count, dtype='O')
        b = np.zeros(count, dtype='f8')
        c = np.zeros(count, dtype='f8')
        for i in range(len(a)):
            a[i] = list(range(1, 10))

        mine = np.rec.fromarrays([a, b, c], names='date,data1,data2')
        for i in range(len(a)):
            assert_(mine.date[i] == list(range(1, 10)))
            assert_(mine.data1[i] == 0.0)
            assert_(mine.data2[i] == 0.0)

    def test_recarray_repr(self):
        a = np.array([(1, 0.1), (2, 0.2)],
                     dtype=[('foo', '<i4'), ('bar', '<f8')])
        a = np.rec.array(a)
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            rec.array([(1, 0.1), (2, 0.2)],
                      dtype=[('foo', '<i4'), ('bar', '<f8')])""")
        )

        # make sure non-structured dtypes also show up as rec.array
        a = np.array(np.ones(4, dtype='f8'))
        assert_(repr(np.rec.array(a)).startswith('rec.array'))

        # check that the 'np.record' part of the dtype isn't shown
        a = np.rec.array(np.ones(3, dtype='i4,i4'))
        assert_equal(repr(a).find('numpy.record'), -1)
        a = np.rec.array(np.ones(3, dtype='i4'))
        assert_(repr(a).find('dtype=int32') != -1)

    def test_0d_recarray_repr(self):
        arr_0d = np.rec.array((1, 2.0, '2003'), dtype='<i4,<f8,<M8[Y]')
        assert_equal(repr(arr_0d), textwrap.dedent("""\
            rec.array((1, 2., '2003'),
                      dtype=[('f0', '<i4'), ('f1', '<f8'), ('f2', '<M8[Y]')])"""))

        record = arr_0d[()]
        assert_equal(repr(record),
            "np.record((1, 2.0, '2003'), "
            "dtype=[('f0', '<i4'), ('f1', '<f8'), ('f2', '<M8[Y]')])")
        # 1.13 converted to python scalars before the repr
        try:
            np.set_printoptions(legacy='1.13')
            assert_equal(repr(record), '(1, 2.0, datetime.date(2003, 1, 1))')
        finally:
            np.set_printoptions(legacy=False)

    def test_recarray_from_repr(self):
        a = np.array([(1, 'ABC'), (2, "DEF")],
                     dtype=[('foo', int), ('bar', 'S4')])
        recordarr = np.rec.array(a)
        recarr = a.view(np.recarray)
        recordview = a.view(np.dtype((np.record, a.dtype)))

        recordarr_r = eval("np." + repr(recordarr), {'np': np})
        recarr_r = eval("np." + repr(recarr), {'np': np})
        # Prints the type `numpy.record` as part of the dtype:
        recordview_r = eval("np." + repr(recordview), {'np': np, 'numpy': np})

        assert_equal(type(recordarr_r), np.recarray)
        assert_equal(recordarr_r.dtype.type, np.record)
        assert_equal(recordarr, recordarr_r)

        assert_equal(type(recarr_r), np.recarray)
        assert_equal(recarr_r.dtype.type, np.record)
        assert_equal(recarr, recarr_r)

        assert_equal(type(recordview_r), np.ndarray)
        assert_equal(recordview.dtype.type, np.record)
        assert_equal(recordview, recordview_r)

    def test_recarray_views(self):
        a = np.array([(1, 'ABC'), (2, "DEF")],
                     dtype=[('foo', int), ('bar', 'S4')])
        b = np.array([1, 2, 3, 4, 5], dtype=np.int64)

        # check that np.rec.array gives right dtypes
        assert_equal(np.rec.array(a).dtype.type, np.record)
        assert_equal(type(np.rec.array(a)), np.recarray)
        assert_equal(np.rec.array(b).dtype.type, np.int64)
        assert_equal(type(np.rec.array(b)), np.recarray)

        # check that viewing as recarray does the same
        assert_equal(a.view(np.recarray).dtype.type, np.record)
        assert_equal(type(a.view(np.recarray)), np.recarray)
        assert_equal(b.view(np.recarray).dtype.type, np.int64)
        assert_equal(type(b.view(np.recarray)), np.recarray)

        # check that view to non-structured dtype preserves type=np.recarray
        r = np.rec.array(np.ones(4, dtype="f4,i4"))
        rv = r.view('f8').view('f4,i4')
        assert_equal(type(rv), np.recarray)
        assert_equal(rv.dtype.type, np.record)

        # check that getitem also preserves np.recarray and np.record
        r = np.rec.array(np.ones(4, dtype=[('a', 'i4'), ('b', 'i4'),
                                           ('c', 'i4,i4')]))
        assert_equal(r['c'].dtype.type, np.record)
        assert_equal(type(r['c']), np.recarray)

        # and that it preserves subclasses (gh-6949)
        class C(np.recarray):
            pass

        c = r.view(C)
        assert_equal(type(c['c']), C)

        # check that accessing nested structures keep record type, but
        # not for subarrays, non-void structures, non-structured voids
        test_dtype = [('a', 'f4,f4'), ('b', 'V8'), ('c', ('f4', 2)),
                      ('d', ('i8', 'i4,i4'))]
        r = np.rec.array([((1, 1), b'11111111', [1, 1], 1),
                          ((1, 1), b'11111111', [1, 1], 1)], dtype=test_dtype)
        assert_equal(r.a.dtype.type, np.record)
        assert_equal(r.b.dtype.type, np.void)
        assert_equal(r.c.dtype.type, np.float32)
        assert_equal(r.d.dtype.type, np.int64)
        # check the same, but for views
        r = np.rec.array(np.ones(4, dtype='i4,i4'))
        assert_equal(r.view('f4,f4').dtype.type, np.record)
        assert_equal(r.view(('i4', 2)).dtype.type, np.int32)
        assert_equal(r.view('V8').dtype.type, np.void)
        assert_equal(r.view(('i8', 'i4,i4')).dtype.type, np.int64)

        # check that we can undo the view
        arrs = [np.ones(4, dtype='f4,i4'), np.ones(4, dtype='f8')]
        for arr in arrs:
            rec = np.rec.array(arr)
            # recommended way to view as an ndarray:
            arr2 = rec.view(rec.dtype.fields or rec.dtype, np.ndarray)
            assert_equal(arr2.dtype.type, arr.dtype.type)
            assert_equal(type(arr2), type(arr))

    def test_recarray_from_names(self):
        ra = np.rec.array([
            (1, 'abc', 3.7000002861022949, 0),
            (2, 'xy', 6.6999998092651367, 1),
            (0, ' ', 0.40000000596046448, 0)],
                       names='c1, c2, c3, c4')
        pa = np.rec.fromrecords([
            (1, 'abc', 3.7000002861022949, 0),
            (2, 'xy', 6.6999998092651367, 1),
            (0, ' ', 0.40000000596046448, 0)],
                       names='c1, c2, c3, c4')
        assert_(ra.dtype == pa.dtype)
        assert_(ra.shape == pa.shape)
        for k in range(len(ra)):
            assert_(ra[k].item() == pa[k].item())

    def test_recarray_conflict_fields(self):
        ra = np.rec.array([(1, 'abc', 2.3), (2, 'xyz', 4.2),
                        (3, 'wrs', 1.3)],
                       names='field, shape, mean')
        ra.mean = [1.1, 2.2, 3.3]
        assert_array_almost_equal(ra['mean'], [1.1, 2.2, 3.3])
        assert_(type(ra.mean) is type(ra.var))
        ra.shape = (1, 3)
        assert_(ra.shape == (1, 3))
        ra.shape = ['A', 'B', 'C']
        assert_array_equal(ra['shape'], [['A', 'B', 'C']])
        ra.field = 5
        assert_array_equal(ra['field'], [[5, 5, 5]])
        assert_(isinstance(ra.field, collections.abc.Callable))

    def test_fromrecords_with_explicit_dtype(self):
        a = np.rec.fromrecords([(1, 'a'), (2, 'bbb')],
                                dtype=[('a', int), ('b', object)])
        assert_equal(a.a, [1, 2])
        assert_equal(a[0].a, 1)
        assert_equal(a.b, ['a', 'bbb'])
        assert_equal(a[-1].b, 'bbb')
        #
        ndtype = np.dtype([('a', int), ('b', object)])
        a = np.rec.fromrecords([(1, 'a'), (2, 'bbb')], dtype=ndtype)
        assert_equal(a.a, [1, 2])
        assert_equal(a[0].a, 1)
        assert_equal(a.b, ['a', 'bbb'])
        assert_equal(a[-1].b, 'bbb')

    def test_recarray_stringtypes(self):
        # Issue #3993
        a = np.array([('abc ', 1), ('abc', 2)],
                     dtype=[('foo', 'S4'), ('bar', int)])
        a = a.view(np.recarray)
        assert_equal(a.foo[0] == a.foo[1], False)

    def test_recarray_returntypes(self):
        qux_fields = {'C': (np.dtype('S5'), 0), 'D': (np.dtype('S5'), 6)}
        a = np.rec.array([('abc ', (1, 1), 1, ('abcde', 'fgehi')),
                          ('abc', (2, 3), 1, ('abcde', 'jklmn'))],
                         dtype=[('foo', 'S4'),
                                ('bar', [('A', int), ('B', int)]),
                                ('baz', int), ('qux', qux_fields)])
        assert_equal(type(a.foo), np.ndarray)
        assert_equal(type(a['foo']), np.ndarray)
        assert_equal(type(a.bar), np.recarray)
        assert_equal(type(a['bar']), np.recarray)
        assert_equal(a.bar.dtype.type, np.record)
        assert_equal(type(a['qux']), np.recarray)
        assert_equal(a.qux.dtype.type, np.record)
        assert_equal(dict(a.qux.dtype.fields), qux_fields)
        assert_equal(type(a.baz), np.ndarray)
        assert_equal(type(a['baz']), np.ndarray)
        assert_equal(type(a[0].bar), np.record)
        assert_equal(type(a[0]['bar']), np.record)
        assert_equal(a[0].bar.A, 1)
        assert_equal(a[0].bar['A'], 1)
        assert_equal(a[0]['bar'].A, 1)
        assert_equal(a[0]['bar']['A'], 1)
        assert_equal(a[0].qux.D, b'fgehi')
        assert_equal(a[0].qux['D'], b'fgehi')
        assert_equal(a[0]['qux'].D, b'fgehi')
        assert_equal(a[0]['qux']['D'], b'fgehi')

    def test_zero_width_strings(self):
        # Test for #6430, based on the test case from #1901

        cols = [['test'] * 3, [''] * 3]
        rec = np.rec.fromarrays(cols)
        assert_equal(rec['f0'], ['test', 'test', 'test'])
        assert_equal(rec['f1'], ['', '', ''])

        dt = np.dtype([('f0', '|S4'), ('f1', '|S')])
        rec = np.rec.fromarrays(cols, dtype=dt)
        assert_equal(rec.itemsize, 4)
        assert_equal(rec['f0'], [b'test', b'test', b'test'])
        assert_equal(rec['f1'], [b'', b'', b''])


class TestPathUsage:
    # Test that pathlib.Path can be used
    def test_tofile_fromfile(self):
        with temppath(suffix='.bin') as path:
            path = Path(path)
            np.random.seed(123)
            a = np.random.rand(10).astype('f8,i4,S5')
            a[5] = (0.5, 10, 'abcde')
            with path.open("wb") as fd:
                a.tofile(fd)
            x = np._core.records.fromfile(
                path, formats='f8,i4,S5', shape=10
            )
            assert_array_equal(x, a)


class TestRecord:
    def setup_method(self):
        self.data = np.rec.fromrecords([(1, 2, 3), (4, 5, 6)],
                            dtype=[("col1", "<i4"),
                                   ("col2", "<i4"),
                                   ("col3", "<i4")])

    def test_assignment1(self):
        a = self.data
        assert_equal(a.col1[0], 1)
        a[0].col1 = 0
        assert_equal(a.col1[0], 0)

    def test_assignment2(self):
        a = self.data
        assert_equal(a.col1[0], 1)
        a.col1[0] = 0
        assert_equal(a.col1[0], 0)

    def test_invalid_assignment(self):
        a = self.data

        def assign_invalid_column(x):
            x[0].col5 = 1

        assert_raises(AttributeError, assign_invalid_column, a)

    def test_nonwriteable_setfield(self):
        # gh-8171
        r = np.rec.array([(0,), (1,)], dtype=[('f', 'i4')])
        r.flags.writeable = False
        with assert_raises(ValueError):
            r.f = [2, 3]
        with assert_raises(ValueError):
            r.setfield([2, 3], *r.dtype.fields['f'])

    def test_out_of_order_fields(self):
        # names in the same order, padding added to descr
        x = self.data[['col1', 'col2']]
        assert_equal(x.dtype.names, ('col1', 'col2'))
        assert_equal(x.dtype.descr,
                     [('col1', '<i4'), ('col2', '<i4'), ('', '|V4')])

        # names change order to match indexing, as of 1.14 - descr can't
        # represent that
        y = self.data[['col2', 'col1']]
        assert_equal(y.dtype.names, ('col2', 'col1'))
        assert_raises(ValueError, lambda: y.dtype.descr)

    def test_pickle_1(self):
        # Issue #1529
        a = np.array([(1, [])], dtype=[('a', np.int32), ('b', np.int32, 0)])
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            assert_equal(a, pickle.loads(pickle.dumps(a, protocol=proto)))
            assert_equal(a[0], pickle.loads(pickle.dumps(a[0],
                                                         protocol=proto)))

    def test_pickle_2(self):
        a = self.data
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            assert_equal(a, pickle.loads(pickle.dumps(a, protocol=proto)))
            assert_equal(a[0], pickle.loads(pickle.dumps(a[0],
                                                         protocol=proto)))

    def test_pickle_3(self):
        # Issue #7140
        a = self.data
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            pa = pickle.loads(pickle.dumps(a[0], protocol=proto))
            assert_(pa.flags.c_contiguous)
            assert_(pa.flags.f_contiguous)
            assert_(pa.flags.writeable)
            assert_(pa.flags.aligned)

    def test_pickle_void(self):
        # issue gh-13593
        dt = np.dtype([('obj', 'O'), ('int', 'i')])
        a = np.empty(1, dtype=dt)
        data = (bytearray(b'eman'),)
        a['obj'] = data
        a['int'] = 42
        ctor, args = a[0].__reduce__()
        # check the constructor is what we expect before interpreting the arguments
        assert ctor is np._core.multiarray.scalar
        dtype, obj = args
        # make sure we did not pickle the address
        assert not isinstance(obj, bytes)

        assert_raises(RuntimeError, ctor, dtype, 13)

        # Test roundtrip:
        dump = pickle.dumps(a[0])
        unpickled = pickle.loads(dump)
        assert a[0] == unpickled

        # Also check the similar (impossible) "object scalar" path:
        with assert_raises(TypeError):
            ctor(np.dtype("O"), data)

    def test_objview_record(self):
        # https://github.com/numpy/numpy/issues/2599
        dt = np.dtype([('foo', 'i8'), ('bar', 'O')])
        r = np.zeros((1, 3), dtype=dt).view(np.recarray)
        r.foo = np.array([1, 2, 3])  # TypeError?

        # https://github.com/numpy/numpy/issues/3256
        ra = np.recarray(
            (2,), dtype=[('x', object), ('y', float), ('z', int)]
        )
        ra[['x', 'y']]  # TypeError?

    def test_record_scalar_setitem(self):
        # https://github.com/numpy/numpy/issues/3561
        rec = np.recarray(1, dtype=[('x', float, 5)])
        rec[0].x = 1
        assert_equal(rec[0].x, np.ones(5))

    def test_missing_field(self):
        # https://github.com/numpy/numpy/issues/4806
        arr = np.zeros((3,), dtype=[('x', int), ('y', int)])
        assert_raises(KeyError, lambda: arr[['nofield']])

    def test_fromarrays_nested_structured_arrays(self):
        arrays = [
            np.arange(10),
            np.ones(10, dtype=[('a', '<u2'), ('b', '<f4')]),
        ]
        arr = np.rec.fromarrays(arrays)  # ValueError?

    @pytest.mark.parametrize('nfields', [0, 1, 2])
    def test_assign_dtype_attribute(self, nfields):
        dt = np.dtype([('a', np.uint8), ('b', np.uint8), ('c', np.uint8)][:nfields])
        data = np.zeros(3, dt).view(np.recarray)

        # the original and resulting dtypes differ on whether they are records
        assert data.dtype.type == np.record
        assert dt.type != np.record

        # ensure that the dtype remains a record even when assigned
        data.dtype = dt
        assert data.dtype.type == np.record

    @pytest.mark.parametrize('nfields', [0, 1, 2])
    def test_nested_fields_are_records(self, nfields):
        """ Test that nested structured types are treated as records too """
        dt = np.dtype([('a', np.uint8), ('b', np.uint8), ('c', np.uint8)][:nfields])
        dt_outer = np.dtype([('inner', dt)])

        data = np.zeros(3, dt_outer).view(np.recarray)
        assert isinstance(data, np.recarray)
        assert isinstance(data['inner'], np.recarray)

        data0 = data[0]
        assert isinstance(data0, np.record)
        assert isinstance(data0['inner'], np.record)

    def test_nested_dtype_padding(self):
        """ test that trailing padding is preserved """
        # construct a dtype with padding at the end
        dt = np.dtype([('a', np.uint8), ('b', np.uint8), ('c', np.uint8)])
        dt_padded_end = dt[['a', 'b']]
        assert dt_padded_end.itemsize == dt.itemsize

        dt_outer = np.dtype([('inner', dt_padded_end)])

        data = np.zeros(3, dt_outer).view(np.recarray)
        assert_equal(data['inner'].dtype, dt_padded_end)

        data0 = data[0]
        assert_equal(data0['inner'].dtype, dt_padded_end)


def test_find_duplicate():
    l1 = [1, 2, 3, 4, 5, 6]
    assert_(np.rec.find_duplicate(l1) == [])

    l2 = [1, 2, 1, 4, 5, 6]
    assert_(np.rec.find_duplicate(l2) == [1])

    l3 = [1, 2, 1, 4, 1, 6, 2, 3]
    assert_(np.rec.find_duplicate(l3) == [1, 2])

    l3 = [2, 2, 1, 4, 1, 6, 2, 3]
    assert_(np.rec.find_duplicate(l3) == [2, 1])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask_sqlalchemy\cli.py ===
from __future__ import annotations

import typing as t

from flask import current_app


def add_models_to_shell() -> dict[str, t.Any]:
    """Registered with :meth:`~flask.Flask.shell_context_processor` if
    ``add_models_to_shell`` is enabled. Adds the ``db`` instance and all model classes
    to ``flask shell``.
    """
    db = current_app.extensions["sqlalchemy"]
    out = {m.class_.__name__: m.class_ for m in db.Model._sa_registry.mappers}
    out["db"] = db
    return out

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask_wtf\__init__.py ===
from .csrf import CSRFProtect
from .form import FlaskForm
from .form import Form
from .recaptcha import Recaptcha
from .recaptcha import RecaptchaField
from .recaptcha import RecaptchaWidget

__version__ = "1.2.2"
__all__ = [
    "CSRFProtect",
    "FlaskForm",
    "Form",
    "Recaptcha",
    "RecaptchaField",
    "RecaptchaWidget",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\h11\_version.py ===
# This file must be kept very simple, because it is consumed from several
# places -- it is imported by h11/__init__.py, execfile'd by setup.py, etc.

# We use a simple scheme:
#   1.0.0 -> 1.0.0+dev -> 1.1.0 -> 1.1.0+dev
# where the +dev versions are never released into the wild, they're just what
# we stick into the VCS in between releases.
#
# This is compatible with PEP 440:
#   http://legacy.python.org/dev/peps/pep-0440/
# via the use of the "local suffix" "+dev", which is disallowed on index
# servers and causes 1.0.0+dev to sort after plain 1.0.0, which is what we
# want. (Contrast with the special suffix 1.0.0.dev, which sorts *before*
# 1.0.0.)

__version__ = "0.16.0"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\api\__init__.py ===
""" public toolkit API """
from pandas.api import (
    extensions,
    indexers,
    interchange,
    types,
    typing,
)

__all__ = [
    "interchange",
    "extensions",
    "indexers",
    "types",
    "typing",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\json\test_readlines.py ===
from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    read_json,
)
import pandas._testing as tm

from pandas.io.json._json import JsonReader

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.fixture
def lines_json_df():
    df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    return df.to_json(lines=True, orient="records")


@pytest.fixture(params=["ujson", "pyarrow"])
def engine(request):
    if request.param == "pyarrow":
        pytest.importorskip("pyarrow.json")
    return request.param


def test_read_jsonl():
    # GH9180
    result = read_json(StringIO('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n'), lines=True)
    expected = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)


def test_read_jsonl_engine_pyarrow(datapath, engine):
    result = read_json(
        datapath("io", "json", "data", "line_delimited.json"),
        lines=True,
        engine=engine,
    )
    expected = DataFrame({"a": [1, 3, 5], "b": [2, 4, 6]})
    tm.assert_frame_equal(result, expected)


def test_read_datetime(request, engine):
    # GH33787
    if engine == "pyarrow":
        # GH 48893
        reason = "Pyarrow only supports a file path as an input and line delimited json"
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    df = DataFrame(
        [([1, 2], ["2020-03-05", "2020-04-08T09:58:49+00:00"], "hector")],
        columns=["accounts", "date", "name"],
    )
    json_line = df.to_json(lines=True, orient="records")

    if engine == "pyarrow":
        result = read_json(StringIO(json_line), engine=engine)
    else:
        result = read_json(StringIO(json_line), engine=engine)
    expected = DataFrame(
        [[1, "2020-03-05", "hector"], [2, "2020-04-08T09:58:49+00:00", "hector"]],
        columns=["accounts", "date", "name"],
    )
    tm.assert_frame_equal(result, expected)


def test_read_jsonl_unicode_chars():
    # GH15132: non-ascii unicode characters
    # \u201d == RIGHT DOUBLE QUOTATION MARK

    # simulate file handle
    json = '{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n'
    json = StringIO(json)
    result = read_json(json, lines=True)
    expected = DataFrame([["foo\u201d", "bar"], ["foo", "bar"]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)

    # simulate string
    json = '{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n'
    result = read_json(StringIO(json), lines=True)
    expected = DataFrame([["foo\u201d", "bar"], ["foo", "bar"]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)


def test_to_jsonl():
    # GH9180
    df = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    result = df.to_json(orient="records", lines=True)
    expected = '{"a":1,"b":2}\n{"a":1,"b":2}\n'
    assert result == expected

    df = DataFrame([["foo}", "bar"], ['foo"', "bar"]], columns=["a", "b"])
    result = df.to_json(orient="records", lines=True)
    expected = '{"a":"foo}","b":"bar"}\n{"a":"foo\\"","b":"bar"}\n'
    assert result == expected
    tm.assert_frame_equal(read_json(StringIO(result), lines=True), df)

    # GH15096: escaped characters in columns and data
    df = DataFrame([["foo\\", "bar"], ['foo"', "bar"]], columns=["a\\", "b"])
    result = df.to_json(orient="records", lines=True)
    expected = '{"a\\\\":"foo\\\\","b":"bar"}\n{"a\\\\":"foo\\"","b":"bar"}\n'
    assert result == expected
    tm.assert_frame_equal(read_json(StringIO(result), lines=True), df)


def test_to_jsonl_count_new_lines():
    # GH36888
    df = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    actual_new_lines_count = df.to_json(orient="records", lines=True).count("\n")
    expected_new_lines_count = 2
    assert actual_new_lines_count == expected_new_lines_count


@pytest.mark.parametrize("chunksize", [1, 1.0])
def test_readjson_chunks(request, lines_json_df, chunksize, engine):
    # Basic test that read_json(chunks=True) gives the same result as
    # read_json(chunks=False)
    # GH17048: memory usage when lines=True

    if engine == "pyarrow":
        # GH 48893
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    unchunked = read_json(StringIO(lines_json_df), lines=True)
    with read_json(
        StringIO(lines_json_df), lines=True, chunksize=chunksize, engine=engine
    ) as reader:
        chunked = pd.concat(reader)

    tm.assert_frame_equal(chunked, unchunked)


def test_readjson_chunksize_requires_lines(lines_json_df, engine):
    msg = "chunksize can only be passed if lines=True"
    with pytest.raises(ValueError, match=msg):
        with read_json(
            StringIO(lines_json_df), lines=False, chunksize=2, engine=engine
        ) as _:
            pass


def test_readjson_chunks_series(request, engine):
    if engine == "pyarrow":
        # GH 48893
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason))

    # Test reading line-format JSON to Series with chunksize param
    s = pd.Series({"A": 1, "B": 2})

    strio = StringIO(s.to_json(lines=True, orient="records"))
    unchunked = read_json(strio, lines=True, typ="Series", engine=engine)

    strio = StringIO(s.to_json(lines=True, orient="records"))
    with read_json(
        strio, lines=True, typ="Series", chunksize=1, engine=engine
    ) as reader:
        chunked = pd.concat(reader)

    tm.assert_series_equal(chunked, unchunked)


def test_readjson_each_chunk(request, lines_json_df, engine):
    if engine == "pyarrow":
        # GH 48893
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    # Other tests check that the final result of read_json(chunksize=True)
    # is correct. This checks the intermediate chunks.
    with read_json(
        StringIO(lines_json_df), lines=True, chunksize=2, engine=engine
    ) as reader:
        chunks = list(reader)
    assert chunks[0].shape == (2, 2)
    assert chunks[1].shape == (1, 2)


def test_readjson_chunks_from_file(request, engine):
    if engine == "pyarrow":
        # GH 48893
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    with tm.ensure_clean("test.json") as path:
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df.to_json(path, lines=True, orient="records")
        with read_json(path, lines=True, chunksize=1, engine=engine) as reader:
            chunked = pd.concat(reader)
        unchunked = read_json(path, lines=True, engine=engine)
        tm.assert_frame_equal(unchunked, chunked)


@pytest.mark.parametrize("chunksize", [None, 1])
def test_readjson_chunks_closes(chunksize):
    with tm.ensure_clean("test.json") as path:
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df.to_json(path, lines=True, orient="records")
        reader = JsonReader(
            path,
            orient=None,
            typ="frame",
            dtype=True,
            convert_axes=True,
            convert_dates=True,
            keep_default_dates=True,
            precise_float=False,
            date_unit=None,
            encoding=None,
            lines=True,
            chunksize=chunksize,
            compression=None,
            nrows=None,
        )
        with reader:
            reader.read()
        assert (
            reader.handles.handle.closed
        ), f"didn't close stream with chunksize = {chunksize}"


@pytest.mark.parametrize("chunksize", [0, -1, 2.2, "foo"])
def test_readjson_invalid_chunksize(lines_json_df, chunksize, engine):
    msg = r"'chunksize' must be an integer >=1"

    with pytest.raises(ValueError, match=msg):
        with read_json(
            StringIO(lines_json_df), lines=True, chunksize=chunksize, engine=engine
        ) as _:
            pass


@pytest.mark.parametrize("chunksize", [None, 1, 2])
def test_readjson_chunks_multiple_empty_lines(chunksize):
    j = """

    {"A":1,"B":4}



    {"A":2,"B":5}







    {"A":3,"B":6}
    """
    orig = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    test = read_json(StringIO(j), lines=True, chunksize=chunksize)
    if chunksize is not None:
        with test:
            test = pd.concat(test)
    tm.assert_frame_equal(orig, test, obj=f"chunksize: {chunksize}")


def test_readjson_unicode(request, monkeypatch, engine):
    if engine == "pyarrow":
        # GH 48893
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    with tm.ensure_clean("test.json") as path:
        monkeypatch.setattr("locale.getpreferredencoding", lambda do_setlocale: "cp949")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"£©µÀÆÖÞßéöÿ":["АБВГДабвгд가"]}')

        result = read_json(path, engine=engine)
        expected = DataFrame({"£©µÀÆÖÞßéöÿ": ["АБВГДабвгд가"]})
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows", [1, 2])
def test_readjson_nrows(nrows, engine):
    # GH 33916
    # Test reading line-format JSON to Series with nrows param
    jsonl = """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}"""
    result = read_json(StringIO(jsonl), lines=True, nrows=nrows)
    expected = DataFrame({"a": [1, 3, 5, 7], "b": [2, 4, 6, 8]}).iloc[:nrows]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows,chunksize", [(2, 2), (4, 2)])
def test_readjson_nrows_chunks(request, nrows, chunksize, engine):
    # GH 33916
    # Test reading line-format JSON to Series with nrows and chunksize param
    if engine == "pyarrow":
        # GH 48893
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    jsonl = """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}"""

    if engine != "pyarrow":
        with read_json(
            StringIO(jsonl), lines=True, nrows=nrows, chunksize=chunksize, engine=engine
        ) as reader:
            chunked = pd.concat(reader)
    else:
        with read_json(
            jsonl, lines=True, nrows=nrows, chunksize=chunksize, engine=engine
        ) as reader:
            chunked = pd.concat(reader)
    expected = DataFrame({"a": [1, 3, 5, 7], "b": [2, 4, 6, 8]}).iloc[:nrows]
    tm.assert_frame_equal(chunked, expected)


def test_readjson_nrows_requires_lines(engine):
    # GH 33916
    # Test ValueError raised if nrows is set without setting lines in read_json
    jsonl = """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}"""
    msg = "nrows can only be passed if lines=True"
    with pytest.raises(ValueError, match=msg):
        read_json(jsonl, lines=False, nrows=2, engine=engine)


def test_readjson_lines_chunks_fileurl(request, datapath, engine):
    # GH 27135
    # Test reading line-format JSON from file url
    if engine == "pyarrow":
        # GH 48893
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    df_list_expected = [
        DataFrame([[1, 2]], columns=["a", "b"], index=[0]),
        DataFrame([[3, 4]], columns=["a", "b"], index=[1]),
        DataFrame([[5, 6]], columns=["a", "b"], index=[2]),
    ]
    os_path = datapath("io", "json", "data", "line_delimited.json")
    file_url = Path(os_path).as_uri()
    with read_json(file_url, lines=True, chunksize=1, engine=engine) as url_reader:
        for index, chuck in enumerate(url_reader):
            tm.assert_frame_equal(chuck, df_list_expected[index])


def test_chunksize_is_incremental():
    # See https://github.com/pandas-dev/pandas/issues/34548
    jsonl = (
        """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}\n"""
        * 1000
    )

    class MyReader:
        def __init__(self, contents) -> None:
            self.read_count = 0
            self.stringio = StringIO(contents)

        def read(self, *args):
            self.read_count += 1
            return self.stringio.read(*args)

        def __iter__(self) -> Iterator:
            self.read_count += 1
            return iter(self.stringio)

    reader = MyReader(jsonl)
    assert len(list(read_json(reader, lines=True, chunksize=100))) > 1
    assert reader.read_count > 10


@pytest.mark.parametrize("orient_", ["split", "index", "table"])
def test_to_json_append_orient(orient_):
    # GH 35849
    # Test ValueError when orient is not 'records'
    df = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    msg = (
        r"mode='a' \(append\) is only supported when "
        "lines is True and orient is 'records'"
    )
    with pytest.raises(ValueError, match=msg):
        df.to_json(mode="a", orient=orient_)


def test_to_json_append_lines():
    # GH 35849
    # Test ValueError when lines is not True
    df = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    msg = (
        r"mode='a' \(append\) is only supported when "
        "lines is True and orient is 'records'"
    )
    with pytest.raises(ValueError, match=msg):
        df.to_json(mode="a", lines=False, orient="records")


@pytest.mark.parametrize("mode_", ["r", "x"])
def test_to_json_append_mode(mode_):
    # GH 35849
    # Test ValueError when mode is not supported option
    df = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    msg = (
        f"mode={mode_} is not a valid option."
        "Only 'w' and 'a' are currently supported."
    )
    with pytest.raises(ValueError, match=msg):
        df.to_json(mode=mode_, lines=False, orient="records")


def test_to_json_append_output_consistent_columns():
    # GH 35849
    # Testing that resulting output reads in as expected.
    # Testing same columns, new rows
    df1 = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    df2 = DataFrame({"col1": [3, 4], "col2": ["c", "d"]})

    expected = DataFrame({"col1": [1, 2, 3, 4], "col2": ["a", "b", "c", "d"]})
    with tm.ensure_clean("test.json") as path:
        # Save dataframes to the same file
        df1.to_json(path, lines=True, orient="records")
        df2.to_json(path, mode="a", lines=True, orient="records")

        # Read path file
        result = read_json(path, lines=True)
        tm.assert_frame_equal(result, expected)


def test_to_json_append_output_inconsistent_columns():
    # GH 35849
    # Testing that resulting output reads in as expected.
    # Testing one new column, one old column, new rows
    df1 = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    df3 = DataFrame({"col2": ["e", "f"], "col3": ["!", "#"]})

    expected = DataFrame(
        {
            "col1": [1, 2, None, None],
            "col2": ["a", "b", "e", "f"],
            "col3": [np.nan, np.nan, "!", "#"],
        }
    )
    with tm.ensure_clean("test.json") as path:
        # Save dataframes to the same file
        df1.to_json(path, mode="a", lines=True, orient="records")
        df3.to_json(path, mode="a", lines=True, orient="records")

        # Read path file
        result = read_json(path, lines=True)
        tm.assert_frame_equal(result, expected)


def test_to_json_append_output_different_columns():
    # GH 35849
    # Testing that resulting output reads in as expected.
    # Testing same, differing and new columns
    df1 = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    df2 = DataFrame({"col1": [3, 4], "col2": ["c", "d"]})
    df3 = DataFrame({"col2": ["e", "f"], "col3": ["!", "#"]})
    df4 = DataFrame({"col4": [True, False]})

    expected = DataFrame(
        {
            "col1": [1, 2, 3, 4, None, None, None, None],
            "col2": ["a", "b", "c", "d", "e", "f", np.nan, np.nan],
            "col3": [np.nan, np.nan, np.nan, np.nan, "!", "#", np.nan, np.nan],
            "col4": [None, None, None, None, None, None, True, False],
        }
    ).astype({"col4": "float"})
    with tm.ensure_clean("test.json") as path:
        # Save dataframes to the same file
        df1.to_json(path, mode="a", lines=True, orient="records")
        df2.to_json(path, mode="a", lines=True, orient="records")
        df3.to_json(path, mode="a", lines=True, orient="records")
        df4.to_json(path, mode="a", lines=True, orient="records")

        # Read path file
        result = read_json(path, lines=True)
        tm.assert_frame_equal(result, expected)


def test_to_json_append_output_different_columns_reordered():
    # GH 35849
    # Testing that resulting output reads in as expected.
    # Testing specific result column order.
    df1 = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    df2 = DataFrame({"col1": [3, 4], "col2": ["c", "d"]})
    df3 = DataFrame({"col2": ["e", "f"], "col3": ["!", "#"]})
    df4 = DataFrame({"col4": [True, False]})

    # df4, df3, df2, df1 (in that order)
    expected = DataFrame(
        {
            "col4": [True, False, None, None, None, None, None, None],
            "col2": [np.nan, np.nan, "e", "f", "c", "d", "a", "b"],
            "col3": [np.nan, np.nan, "!", "#", np.nan, np.nan, np.nan, np.nan],
            "col1": [None, None, None, None, 3, 4, 1, 2],
        }
    ).astype({"col4": "float"})
    with tm.ensure_clean("test.json") as path:
        # Save dataframes to the same file
        df4.to_json(path, mode="a", lines=True, orient="records")
        df3.to_json(path, mode="a", lines=True, orient="records")
        df2.to_json(path, mode="a", lines=True, orient="records")
        df1.to_json(path, mode="a", lines=True, orient="records")

        # Read path file
        result = read_json(path, lines=True)
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\_stack.py ===
from typing import List, TypeVar

T = TypeVar("T")


class Stack(List[T]):
    """A small shim over builtin list."""

    @property
    def top(self) -> T:
        """Get top of stack."""
        return self[-1]

    def push(self, item: T) -> None:
        """Push an item on to the stack (append in stack nomenclature)."""
        self.append(item)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\error.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2006-2020  Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


class ReadlineError(Exception):
    pass


class GetSetError(ReadlineError):
    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\clipboard\__init__.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

from .api import set_clipboard_text
from .get_clipboard_text_and_convert import get_clipboard_text_and_convert

__all__ = [
    "set_clipboard_text",
    "get_clipboard_text_and_convert",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\console\__init__.py ===
from pyreadline3.py3k_compat import is_ironpython

if is_ironpython:
    try:
        from .ironpython_console import *
    except ImportError as x:
        raise ImportError(
            "Could not find a console implementation for local " "ironpython version"
        ) from x
else:
    try:
        from .console import *
    except ImportError as x:
        raise ImportError(
            "Could not find a console implementation for local " "python version"
        ) from x

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\logger\null_handler.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

from logging import Handler, LogRecord


class NULLHandler(Handler):

    def emit(self, record: LogRecord) -> None:
        pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\chrome\__init__.py ===
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\chromium\__init__.py ===
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\__init__.py ===
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\actions\__init__.py ===
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\bidi\__init__.py ===
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\fedcm\__init__.py ===
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\edge\__init__.py ===
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\firefox\__init__.py ===
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\ie\__init__.py ===
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\remote\__init__.py ===
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\safari\__init__.py ===
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\support\__init__.py ===
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\webkitgtk\__init__.py ===
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\wpewebkit\__init__.py ===
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\future\__init__.py ===
# future/__init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""2.0 API features.

this module is legacy as 2.0 APIs are now standard.

"""
from .engine import Connection as Connection
from .engine import create_engine as create_engine
from .engine import Engine as Engine
from ..sql._selectable_constructors import select as select

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\gmpyfinitefield.py ===
"""Implementation of :class:`GMPYFiniteField` class. """


from sympy.polys.domains.finitefield import FiniteField
from sympy.polys.domains.gmpyintegerring import GMPYIntegerRing

from sympy.utilities import public

@public
class GMPYFiniteField(FiniteField):
    """Finite field based on GMPY integers. """

    alias = 'FF_gmpy'

    def __init__(self, mod, symmetric=True):
        super().__init__(mod, GMPYIntegerRing(), symmetric)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\pythonfinitefield.py ===
"""Implementation of :class:`PythonFiniteField` class. """


from sympy.polys.domains.finitefield import FiniteField
from sympy.polys.domains.pythonintegerring import PythonIntegerRing

from sympy.utilities import public

@public
class PythonFiniteField(FiniteField):
    """Finite field based on Python's integers. """

    alias = 'FF_python'

    def __init__(self, mod, symmetric=True):
        super().__init__(mod, PythonIntegerRing(), symmetric)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\_typing.py ===
from typing import TypeVar, Protocol


T = TypeVar('T')


class RingElement(Protocol):
    """A ring element.

    Must support ``+``, ``-``, ``*``, ``**`` and ``-``.
    """
    def __add__(self: T, other: T, /) -> T: ...
    def __sub__(self: T, other: T, /) -> T: ...
    def __mul__(self: T, other: T, /) -> T: ...
    def __pow__(self: T, other: int, /) -> T: ...
    def __neg__(self: T, /) -> T: ...

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\gtk.py ===
from sympy.printing.mathml import mathml
from sympy.utilities.mathml import c2p
import tempfile
import subprocess


def print_gtk(x, start_viewer=True):
    """Print to Gtkmathview, a gtk widget capable of rendering MathML.

    Needs libgtkmathview-bin"""
    with tempfile.NamedTemporaryFile('w') as file:
        file.write(c2p(mathml(x), simple=True))
        file.flush()

        if start_viewer:
            subprocess.check_call(('mathmlviewer', file.name))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\ode\__init__.py ===
from .ode import (allhints, checkinfsol, classify_ode,
        constantsimp, dsolve, homogeneous_order)

from .lie_group import infinitesimals

from .subscheck import checkodesol

from .systems import (canonical_odes, linear_ode_to_matrix,
        linodesolve)


__all__ = [
    'allhints', 'checkinfsol', 'checkodesol', 'classify_ode', 'constantsimp',
    'dsolve', 'homogeneous_order', 'infinitesimals', 'canonical_odes', 'linear_ode_to_matrix',
    'linodesolve'
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\contrib\emscripten\__init__.py ===
from __future__ import annotations

import urllib3.connection

from ...connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from .connection import EmscriptenHTTPConnection, EmscriptenHTTPSConnection


def inject_into_urllib3() -> None:
    # override connection classes to use emscripten specific classes
    # n.b. mypy complains about the overriding of classes below
    # if it isn't ignored
    HTTPConnectionPool.ConnectionCls = EmscriptenHTTPConnection
    HTTPSConnectionPool.ConnectionCls = EmscriptenHTTPSConnection
    urllib3.connection.HTTPConnection = EmscriptenHTTPConnection  # type: ignore[misc,assignment]
    urllib3.connection.HTTPSConnection = EmscriptenHTTPSConnection  # type: ignore[misc,assignment]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\tests\test_laguerre.py ===
"""Tests for laguerre module.

"""
from functools import reduce

import numpy as np
import numpy.polynomial.laguerre as lag
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

L0 = np.array([1]) / 1
L1 = np.array([1, -1]) / 1
L2 = np.array([2, -4, 1]) / 2
L3 = np.array([6, -18, 9, -1]) / 6
L4 = np.array([24, -96, 72, -16, 1]) / 24
L5 = np.array([120, -600, 600, -200, 25, -1]) / 120
L6 = np.array([720, -4320, 5400, -2400, 450, -36, 1]) / 720

Llist = [L0, L1, L2, L3, L4, L5, L6]


def trim(x):
    return lag.lagtrim(x, tol=1e-6)


class TestConstants:

    def test_lagdomain(self):
        assert_equal(lag.lagdomain, [0, 1])

    def test_lagzero(self):
        assert_equal(lag.lagzero, [0])

    def test_lagone(self):
        assert_equal(lag.lagone, [1])

    def test_lagx(self):
        assert_equal(lag.lagx, [1, -1])


class TestArithmetic:
    x = np.linspace(-3, 3, 100)

    def test_lagadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = lag.lagadd([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_lagsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = lag.lagsub([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_lagmulx(self):
        assert_equal(lag.lagmulx([0]), [0])
        assert_equal(lag.lagmulx([1]), [1, -1])
        for i in range(1, 5):
            ser = [0] * i + [1]
            tgt = [0] * (i - 1) + [-i, 2 * i + 1, -(i + 1)]
            assert_almost_equal(lag.lagmulx(ser), tgt)

    def test_lagmul(self):
        # check values of result
        for i in range(5):
            pol1 = [0] * i + [1]
            val1 = lag.lagval(self.x, pol1)
            for j in range(5):
                msg = f"At i={i}, j={j}"
                pol2 = [0] * j + [1]
                val2 = lag.lagval(self.x, pol2)
                pol3 = lag.lagmul(pol1, pol2)
                val3 = lag.lagval(self.x, pol3)
                assert_(len(pol3) == i + j + 1, msg)
                assert_almost_equal(val3, val1 * val2, err_msg=msg)

    def test_lagdiv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0] * i + [1]
                cj = [0] * j + [1]
                tgt = lag.lagadd(ci, cj)
                quo, rem = lag.lagdiv(tgt, ci)
                res = lag.lagadd(lag.lagmul(quo, ci), rem)
                assert_almost_equal(trim(res), trim(tgt), err_msg=msg)

    def test_lagpow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = np.arange(i + 1)
                tgt = reduce(lag.lagmul, [c] * j, np.array([1]))
                res = lag.lagpow(c, j)
                assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = np.array([9., -14., 6.])
    c2d = np.einsum('i,j->ij', c1d, c1d)
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = np.random.random((3, 5)) * 2 - 1
    y = polyval(x, [1., 2., 3.])

    def test_lagval(self):
        # check empty input
        assert_equal(lag.lagval([], [1]).size, 0)

        # check normal input)
        x = np.linspace(-1, 1)
        y = [polyval(x, c) for c in Llist]
        for i in range(7):
            msg = f"At i={i}"
            tgt = y[i]
            res = lag.lagval(x, [0] * i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)

        # check that shape is preserved
        for i in range(3):
            dims = [2] * i
            x = np.zeros(dims)
            assert_equal(lag.lagval(x, [1]).shape, dims)
            assert_equal(lag.lagval(x, [1, 0]).shape, dims)
            assert_equal(lag.lagval(x, [1, 0, 0]).shape, dims)

    def test_lagval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        assert_raises(ValueError, lag.lagval2d, x1, x2[:2], self.c2d)

        # test values
        tgt = y1 * y2
        res = lag.lagval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = lag.lagval2d(z, z, self.c2d)
        assert_(res.shape == (2, 3))

    def test_lagval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        assert_raises(ValueError, lag.lagval3d, x1, x2, x3[:2], self.c3d)

        # test values
        tgt = y1 * y2 * y3
        res = lag.lagval3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = lag.lagval3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3))

    def test_laggrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = np.einsum('i,j->ij', y1, y2)
        res = lag.laggrid2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = lag.laggrid2d(z, z, self.c2d)
        assert_(res.shape == (2, 3) * 2)

    def test_laggrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        res = lag.laggrid3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = lag.laggrid3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3) * 3)


class TestIntegral:

    def test_lagint(self):
        # check exceptions
        assert_raises(TypeError, lag.lagint, [0], .5)
        assert_raises(ValueError, lag.lagint, [0], -1)
        assert_raises(ValueError, lag.lagint, [0], 1, [0, 0])
        assert_raises(ValueError, lag.lagint, [0], lbnd=[0])
        assert_raises(ValueError, lag.lagint, [0], scl=[0])
        assert_raises(TypeError, lag.lagint, [0], axis=.5)

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = lag.lagint([0], m=i, k=k)
            assert_almost_equal(res, [1, -1])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            lagpol = lag.poly2lag(pol)
            lagint = lag.lagint(lagpol, m=1, k=[i])
            res = lag.lag2poly(lagint)
            assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            lagpol = lag.poly2lag(pol)
            lagint = lag.lagint(lagpol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(lag.lagval(-1, lagint), i)

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            lagpol = lag.poly2lag(pol)
            lagint = lag.lagint(lagpol, m=1, k=[i], scl=2)
            res = lag.lag2poly(lagint)
            assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = lag.lagint(tgt, m=1)
                res = lag.lagint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = lag.lagint(tgt, m=1, k=[k])
                res = lag.lagint(pol, m=j, k=list(range(j)))
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = lag.lagint(tgt, m=1, k=[k], lbnd=-1)
                res = lag.lagint(pol, m=j, k=list(range(j)), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = lag.lagint(tgt, m=1, k=[k], scl=2)
                res = lag.lagint(pol, m=j, k=list(range(j)), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_lagint_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([lag.lagint(c) for c in c2d.T]).T
        res = lag.lagint(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([lag.lagint(c) for c in c2d])
        res = lag.lagint(c2d, axis=1)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([lag.lagint(c, k=3) for c in c2d])
        res = lag.lagint(c2d, k=3, axis=1)
        assert_almost_equal(res, tgt)


class TestDerivative:

    def test_lagder(self):
        # check exceptions
        assert_raises(TypeError, lag.lagder, [0], .5)
        assert_raises(ValueError, lag.lagder, [0], -1)

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0] * i + [1]
            res = lag.lagder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = lag.lagder(lag.lagint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = lag.lagder(lag.lagint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_lagder_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([lag.lagder(c) for c in c2d.T]).T
        res = lag.lagder(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([lag.lagder(c) for c in c2d])
        res = lag.lagder(c2d, axis=1)
        assert_almost_equal(res, tgt)


class TestVander:
    # some random values in [-1, 1)
    x = np.random.random((3, 5)) * 2 - 1

    def test_lagvander(self):
        # check for 1d x
        x = np.arange(3)
        v = lag.lagvander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            assert_almost_equal(v[..., i], lag.lagval(x, coef))

        # check for 2d x
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = lag.lagvander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            assert_almost_equal(v[..., i], lag.lagval(x, coef))

    def test_lagvander2d(self):
        # also tests lagval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3))
        van = lag.lagvander2d(x1, x2, [1, 2])
        tgt = lag.lagval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = lag.lagvander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))

    def test_lagvander3d(self):
        # also tests lagval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3, 4))
        van = lag.lagvander3d(x1, x2, x3, [1, 2, 3])
        tgt = lag.lagval3d(x1, x2, x3, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = lag.lagvander3d([x1], [x2], [x3], [1, 2, 3])
        assert_(van.shape == (1, 5, 24))


class TestFitting:

    def test_lagfit(self):
        def f(x):
            return x * (x - 1) * (x - 2)

        # Test exceptions
        assert_raises(ValueError, lag.lagfit, [1], [1], -1)
        assert_raises(TypeError, lag.lagfit, [[1]], [1], 0)
        assert_raises(TypeError, lag.lagfit, [], [1], 0)
        assert_raises(TypeError, lag.lagfit, [1], [[[1]]], 0)
        assert_raises(TypeError, lag.lagfit, [1, 2], [1], 0)
        assert_raises(TypeError, lag.lagfit, [1], [1, 2], 0)
        assert_raises(TypeError, lag.lagfit, [1], [1], 0, w=[[1]])
        assert_raises(TypeError, lag.lagfit, [1], [1], 0, w=[1, 1])
        assert_raises(ValueError, lag.lagfit, [1], [1], [-1,])
        assert_raises(ValueError, lag.lagfit, [1], [1], [2, -1, 6])
        assert_raises(TypeError, lag.lagfit, [1], [1], [])

        # Test fit
        x = np.linspace(0, 2)
        y = f(x)
        #
        coef3 = lag.lagfit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(lag.lagval(x, coef3), y)
        coef3 = lag.lagfit(x, y, [0, 1, 2, 3])
        assert_equal(len(coef3), 4)
        assert_almost_equal(lag.lagval(x, coef3), y)
        #
        coef4 = lag.lagfit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(lag.lagval(x, coef4), y)
        coef4 = lag.lagfit(x, y, [0, 1, 2, 3, 4])
        assert_equal(len(coef4), 5)
        assert_almost_equal(lag.lagval(x, coef4), y)
        #
        coef2d = lag.lagfit(x, np.array([y, y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        coef2d = lag.lagfit(x, np.array([y, y]).T, [0, 1, 2, 3])
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        # test weighting
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = lag.lagfit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        wcoef3 = lag.lagfit(x, yw, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = lag.lagfit(x, np.array([yw, yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        wcoef2d = lag.lagfit(x, np.array([yw, yw]).T, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        assert_almost_equal(lag.lagfit(x, x, 1), [1, -1])
        assert_almost_equal(lag.lagfit(x, x, [0, 1]), [1, -1])


class TestCompanion:

    def test_raises(self):
        assert_raises(ValueError, lag.lagcompanion, [])
        assert_raises(ValueError, lag.lagcompanion, [1])

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0] * i + [1]
            assert_(lag.lagcompanion(coef).shape == (i, i))

    def test_linear_root(self):
        assert_(lag.lagcompanion([1, 2])[0, 0] == 1.5)


class TestGauss:

    def test_100(self):
        x, w = lag.laggauss(100)

        # test orthogonality. Note that the results need to be normalized,
        # otherwise the huge values that can arise from fast growing
        # functions like Laguerre can be very confusing.
        v = lag.lagvander(x, 99)
        vv = np.dot(v.T * w, v)
        vd = 1 / np.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        assert_almost_equal(vv, np.eye(100))

        # check that the integral of 1 is correct
        tgt = 1.0
        assert_almost_equal(w.sum(), tgt)


class TestMisc:

    def test_lagfromroots(self):
        res = lag.lagfromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = np.cos(np.linspace(-np.pi, 0, 2 * i + 1)[1::2])
            pol = lag.lagfromroots(roots)
            res = lag.lagval(roots, pol)
            tgt = 0
            assert_(len(pol) == i + 1)
            assert_almost_equal(lag.lag2poly(pol)[-1], 1)
            assert_almost_equal(res, tgt)

    def test_lagroots(self):
        assert_almost_equal(lag.lagroots([1]), [])
        assert_almost_equal(lag.lagroots([0, 1]), [1])
        for i in range(2, 5):
            tgt = np.linspace(0, 3, i)
            res = lag.lagroots(lag.lagfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_lagtrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        assert_raises(ValueError, lag.lagtrim, coef, -1)

        # Test results
        assert_equal(lag.lagtrim(coef), coef[:-1])
        assert_equal(lag.lagtrim(coef, 1), coef[:-3])
        assert_equal(lag.lagtrim(coef, 2), [0])

    def test_lagline(self):
        assert_equal(lag.lagline(3, 4), [7, -4])

    def test_lag2poly(self):
        for i in range(7):
            assert_almost_equal(lag.lag2poly([0] * i + [1]), Llist[i])

    def test_poly2lag(self):
        for i in range(7):
            assert_almost_equal(lag.poly2lag(Llist[i]), [0] * i + [1])

    def test_weight(self):
        x = np.linspace(0, 10, 11)
        tgt = np.exp(-x)
        res = lag.lagweight(x)
        assert_almost_equal(res, tgt)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\tests\test_css.py ===
import pytest
import types

from bs4 import (
    BeautifulSoup,
    ResultSet,
)

from typing import (
    Any,
    List,
    Tuple,
    Type,
)

from packaging.version import Version

from . import (
    SoupTest,
    SOUP_SIEVE_PRESENT,
)

SOUPSIEVE_EXCEPTION_ON_UNSUPPORTED_PSEUDOCLASS: Type[Exception]
if SOUP_SIEVE_PRESENT:
    from soupsieve import __version__, SelectorSyntaxError

    # Some behavior changes in soupsieve 2.6 that affects one of our
    # tests.  For the test to run under all versions of Python
    # supported by Beautiful Soup (which includes versions of Python
    # not supported by soupsieve 2.6) we need to check both behaviors.
    SOUPSIEVE_EXCEPTION_ON_UNSUPPORTED_PSEUDOCLASS = SelectorSyntaxError
    if Version(__version__) < Version("2.6"):
        SOUPSIEVE_EXCEPTION_ON_UNSUPPORTED_PSEUDOCLASS = NotImplementedError


@pytest.mark.skipif(not SOUP_SIEVE_PRESENT, reason="Soup Sieve not installed")
class TestCSSSelectors(SoupTest):
    """Test basic CSS selector functionality.

    This functionality is implemented in soupsieve, which has a much
    more comprehensive test suite, so this is basically an extra check
    that soupsieve works as expected.
    """

    HTML = """
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN"
"http://www.w3.org/TR/html4/strict.dtd">
<html>
<head>
<title>The title</title>
<link rel="stylesheet" href="blah.css" type="text/css" id="l1">
</head>
<body>
<custom-dashed-tag class="dashed" id="dash1">Hello there.</custom-dashed-tag>
<div id="main" class="fancy">
<div id="inner">
<h1 id="header1">An H1</h1>
<p>Some text</p>
<p class="onep" id="p1">Some more text</p>
<h2 id="header2">An H2</h2>
<p class="class1 class2 class3" id="pmulti">Another</p>
<a href="http://bob.example.org/" rel="friend met" id="bob">Bob</a>
<h2 id="header3">Another H2</h2>
<a id="me" href="http://simonwillison.net/" rel="me">me</a>
<span class="s1">
<a href="#" id="s1a1">span1a1</a>
<a href="#" id="s1a2">span1a2 <span id="s1a2s1">test</span></a>
<span class="span2">
<a href="#" id="s2a1">span2a1</a>
</span>
<span class="span3"></span>
<custom-dashed-tag class="dashed" id="dash2"/>
<div data-tag="dashedvalue" id="data1"/>
</span>
</div>
<x id="xid">
<z id="zida"/>
<z id="zidab"/>
<z id="zidac"/>
</x>
<y id="yid">
<z id="zidb"/>
</y>
<p lang="en" id="lang-en">English</p>
<p lang="en-gb" id="lang-en-gb">English UK</p>
<p lang="en-us" id="lang-en-us">English US</p>
<p lang="fr" id="lang-fr">French</p>
</div>

<div id="footer">
</div>
"""

    def setup_method(self):
        self._soup = BeautifulSoup(self.HTML, "html.parser")

    def assert_css_selects(
        self, selector: str, expected_ids: List[str], **kwargs: Any
    ) -> None:
        results = self._soup.select(selector, **kwargs)
        assert isinstance(results, ResultSet)
        el_ids = [el["id"] for el in results]
        el_ids.sort()
        expected_ids.sort()
        assert expected_ids == el_ids, "Selector %s, expected [%s], got [%s]" % (
            selector,
            ", ".join(expected_ids),
            ", ".join(el_ids),
        )

    assertSelect = assert_css_selects

    def assert_css_select_multiple(self, *tests: Tuple[str, List[str]]):
        for selector, expected_ids in tests:
            self.assert_css_selects(selector, expected_ids)

    def test_precompiled(self):
        sel = self._soup.css.compile("div")

        els = self._soup.select(sel)
        assert len(els) == 4
        for div in els:
            assert div.name == "div"

        el = self._soup.select_one(sel)
        assert "main" == el["id"]

    def test_one_tag_one(self):
        els = self._soup.select("title")
        assert len(els) == 1
        assert els[0].name == "title"
        assert els[0].contents == ["The title"]

    def test_one_tag_many(self):
        els = self._soup.select("div")
        assert len(els) == 4
        for div in els:
            assert div.name == "div"

        el = self._soup.select_one("div")
        assert "main" == el["id"]

    def test_select_one_returns_none_if_no_match(self):
        match = self._soup.select_one("nonexistenttag")
        assert None is match

    def test_tag_in_tag_one(self):
        self.assert_css_selects("div div", ["inner", "data1"])

    def test_tag_in_tag_many(self):
        for selector in ("html div", "html body div", "body div"):
            self.assert_css_selects(selector, ["data1", "main", "inner", "footer"])

    def test_limit(self):
        self.assert_css_selects("html div", ["main"], limit=1)
        self.assert_css_selects("html body div", ["inner", "main"], limit=2)
        self.assert_css_selects(
            "body div", ["data1", "main", "inner", "footer"], limit=10
        )

    def test_tag_no_match(self):
        assert len(self._soup.select("del")) == 0

    def test_invalid_tag(self):
        with pytest.raises(SelectorSyntaxError):
            self._soup.select("tag%t")

    def test_select_dashed_tag_ids(self):
        self.assert_css_selects("custom-dashed-tag", ["dash1", "dash2"])

    def test_select_dashed_by_id(self):
        dashed = self._soup.select('custom-dashed-tag[id="dash2"]')
        assert dashed[0].name == "custom-dashed-tag"
        assert dashed[0]["id"] == "dash2"

    def test_dashed_tag_text(self):
        assert self._soup.select("body > custom-dashed-tag")[0].text == "Hello there."

    def test_select_dashed_matches_find_all(self):
        assert self._soup.select("custom-dashed-tag") == self._soup.find_all(
            "custom-dashed-tag"
        )

    def test_header_tags(self):
        self.assert_css_select_multiple(
            ("h1", ["header1"]),
            ("h2", ["header2", "header3"]),
        )

    def test_class_one(self):
        for selector in (".onep", "p.onep", "html p.onep"):
            els = self._soup.select(selector)
            assert len(els) == 1
            assert els[0].name == "p"
            assert els[0]["class"] == ["onep"]

    def test_class_mismatched_tag(self):
        els = self._soup.select("div.onep")
        assert len(els) == 0

    def test_one_id(self):
        for selector in ("div#inner", "#inner", "div div#inner"):
            self.assert_css_selects(selector, ["inner"])

    def test_bad_id(self):
        els = self._soup.select("#doesnotexist")
        assert len(els) == 0

    def test_items_in_id(self):
        els = self._soup.select("div#inner p")
        assert len(els) == 3
        for el in els:
            assert el.name == "p"
        assert els[1]["class"] == ["onep"]
        assert not els[0].has_attr("class")

    def test_a_bunch_of_emptys(self):
        for selector in ("div#main del", "div#main div.oops", "div div#main"):
            assert len(self._soup.select(selector)) == 0

    def test_multi_class_support(self):
        for selector in (
            ".class1",
            "p.class1",
            ".class2",
            "p.class2",
            ".class3",
            "p.class3",
            "html p.class2",
            "div#inner .class2",
        ):
            self.assert_css_selects(selector, ["pmulti"])

    def test_multi_class_selection(self):
        for selector in (".class1.class3", ".class3.class2", ".class1.class2.class3"):
            self.assert_css_selects(selector, ["pmulti"])

    def test_child_selector(self):
        self.assert_css_selects(".s1 > a", ["s1a1", "s1a2"])
        self.assert_css_selects(".s1 > a span", ["s1a2s1"])

    def test_child_selector_id(self):
        self.assert_css_selects(".s1 > a#s1a2 span", ["s1a2s1"])

    def test_attribute_equals(self):
        self.assert_css_select_multiple(
            ('p[class="onep"]', ["p1"]),
            ('p[id="p1"]', ["p1"]),
            ('[class="onep"]', ["p1"]),
            ('[id="p1"]', ["p1"]),
            ('link[rel="stylesheet"]', ["l1"]),
            ('link[type="text/css"]', ["l1"]),
            ('link[href="blah.css"]', ["l1"]),
            ('link[href="no-blah.css"]', []),
            ('[rel="stylesheet"]', ["l1"]),
            ('[type="text/css"]', ["l1"]),
            ('[href="blah.css"]', ["l1"]),
            ('[href="no-blah.css"]', []),
            ('p[href="no-blah.css"]', []),
            ('[href="no-blah.css"]', []),
        )

    def test_attribute_tilde(self):
        self.assert_css_select_multiple(
            ('p[class~="class1"]', ["pmulti"]),
            ('p[class~="class2"]', ["pmulti"]),
            ('p[class~="class3"]', ["pmulti"]),
            ('[class~="class1"]', ["pmulti"]),
            ('[class~="class2"]', ["pmulti"]),
            ('[class~="class3"]', ["pmulti"]),
            ('a[rel~="friend"]', ["bob"]),
            ('a[rel~="met"]', ["bob"]),
            ('[rel~="friend"]', ["bob"]),
            ('[rel~="met"]', ["bob"]),
        )

    def test_attribute_startswith(self):
        self.assert_css_select_multiple(
            ('[rel^="style"]', ["l1"]),
            ('link[rel^="style"]', ["l1"]),
            ('notlink[rel^="notstyle"]', []),
            ('[rel^="notstyle"]', []),
            ('link[rel^="notstyle"]', []),
            ('link[href^="bla"]', ["l1"]),
            ('a[href^="http://"]', ["bob", "me"]),
            ('[href^="http://"]', ["bob", "me"]),
            ('[id^="p"]', ["pmulti", "p1"]),
            ('[id^="m"]', ["me", "main"]),
            ('div[id^="m"]', ["main"]),
            ('a[id^="m"]', ["me"]),
            ('div[data-tag^="dashed"]', ["data1"]),
        )

    def test_attribute_endswith(self):
        self.assert_css_select_multiple(
            ('[href$=".css"]', ["l1"]),
            ('link[href$=".css"]', ["l1"]),
            ('link[id$="1"]', ["l1"]),
            (
                '[id$="1"]',
                ["data1", "l1", "p1", "header1", "s1a1", "s2a1", "s1a2s1", "dash1"],
            ),
            ('div[id$="1"]', ["data1"]),
            ('[id$="noending"]', []),
        )

    def test_attribute_contains(self):
        self.assert_css_select_multiple(
            # From test_attribute_startswith
            ('[rel*="style"]', ["l1"]),
            ('link[rel*="style"]', ["l1"]),
            ('notlink[rel*="notstyle"]', []),
            ('[rel*="notstyle"]', []),
            ('link[rel*="notstyle"]', []),
            ('link[href*="bla"]', ["l1"]),
            ('[href*="http://"]', ["bob", "me"]),
            ('[id*="p"]', ["pmulti", "p1"]),
            ('div[id*="m"]', ["main"]),
            ('a[id*="m"]', ["me"]),
            # From test_attribute_endswith
            ('[href*=".css"]', ["l1"]),
            ('link[href*=".css"]', ["l1"]),
            ('link[id*="1"]', ["l1"]),
            (
                '[id*="1"]',
                [
                    "data1",
                    "l1",
                    "p1",
                    "header1",
                    "s1a1",
                    "s1a2",
                    "s2a1",
                    "s1a2s1",
                    "dash1",
                ],
            ),
            ('div[id*="1"]', ["data1"]),
            ('[id*="noending"]', []),
            # New for this test
            ('[href*="."]', ["bob", "me", "l1"]),
            ('a[href*="."]', ["bob", "me"]),
            ('link[href*="."]', ["l1"]),
            ('div[id*="n"]', ["main", "inner"]),
            ('div[id*="nn"]', ["inner"]),
            ('div[data-tag*="edval"]', ["data1"]),
        )

    def test_attribute_exact_or_hypen(self):
        self.assert_css_select_multiple(
            ('p[lang|="en"]', ["lang-en", "lang-en-gb", "lang-en-us"]),
            ('[lang|="en"]', ["lang-en", "lang-en-gb", "lang-en-us"]),
            ('p[lang|="fr"]', ["lang-fr"]),
            ('p[lang|="gb"]', []),
        )

    def test_attribute_exists(self):
        self.assert_css_select_multiple(
            ("[rel]", ["l1", "bob", "me"]),
            ("link[rel]", ["l1"]),
            ("a[rel]", ["bob", "me"]),
            ("[lang]", ["lang-en", "lang-en-gb", "lang-en-us", "lang-fr"]),
            ("p[class]", ["p1", "pmulti"]),
            ("[blah]", []),
            ("p[blah]", []),
            ("div[data-tag]", ["data1"]),
        )

    def test_quoted_space_in_selector_name(self):
        html = """<div style="display: wrong">nope</div>
        <div style="display: right">yes</div>
        """
        soup = BeautifulSoup(html, "html.parser")
        [chosen] = soup.select('div[style="display: right"]')
        assert "yes" == chosen.string

    def test_unsupported_pseudoclass(self):
        with pytest.raises(SOUPSIEVE_EXCEPTION_ON_UNSUPPORTED_PSEUDOCLASS):
            self._soup.select("a:no-such-pseudoclass")

        with pytest.raises(SelectorSyntaxError):
            self._soup.select("a:nth-of-type(a)")

    def test_nth_of_type(self):
        # Try to select first paragraph
        els = self._soup.select("div#inner p:nth-of-type(1)")
        assert len(els) == 1
        assert els[0].string == "Some text"

        # Try to select third paragraph
        els = self._soup.select("div#inner p:nth-of-type(3)")
        assert len(els) == 1
        assert els[0].string == "Another"

        # Try to select (non-existent!) fourth paragraph
        els = self._soup.select("div#inner p:nth-of-type(4)")
        assert len(els) == 0

        # Zero will select no tags.
        els = self._soup.select("div p:nth-of-type(0)")
        assert len(els) == 0

    def test_nth_of_type_direct_descendant(self):
        els = self._soup.select("div#inner > p:nth-of-type(1)")
        assert len(els) == 1
        assert els[0].string == "Some text"

    def test_id_child_selector_nth_of_type(self):
        self.assert_css_selects("#inner > p:nth-of-type(2)", ["p1"])

    def test_select_on_element(self):
        # Other tests operate on the tree; this operates on an element
        # within the tree.
        inner = self._soup.find("div", id="main")
        selected = inner.select("div")
        # The <div id="inner"> tag was selected. The <div id="footer">
        # tag was not.
        self.assert_selects_ids(selected, ["inner", "data1"])

    def test_overspecified_child_id(self):
        self.assert_css_selects(".fancy #inner", ["inner"])
        self.assert_css_selects(".normal #inner", [])

    def test_adjacent_sibling_selector(self):
        self.assert_css_selects("#p1 + h2", ["header2"])
        self.assert_css_selects("#p1 + h2 + p", ["pmulti"])
        self.assert_css_selects("#p1 + #header2 + .class1", ["pmulti"])
        assert [] == self._soup.select("#p1 + p")

    def test_general_sibling_selector(self):
        self.assert_css_selects("#p1 ~ h2", ["header2", "header3"])
        self.assert_css_selects("#p1 ~ #header2", ["header2"])
        self.assert_css_selects("#p1 ~ h2 + a", ["me"])
        self.assert_css_selects('#p1 ~ h2 + [rel="me"]', ["me"])
        assert [] == self._soup.select("#inner ~ h2")

    def test_dangling_combinator(self):
        with pytest.raises(SelectorSyntaxError):
            self._soup.select("h1 >")

    def test_sibling_combinator_wont_select_same_tag_twice(self):
        self.assert_css_selects("p[lang] ~ p", ["lang-en-gb", "lang-en-us", "lang-fr"])

    # Test the selector grouping operator (the comma)
    def test_multiple_select(self):
        self.assert_css_selects("x, y", ["xid", "yid"])

    def test_multiple_select_with_no_space(self):
        self.assert_css_selects("x,y", ["xid", "yid"])

    def test_multiple_select_with_more_space(self):
        self.assert_css_selects("x,    y", ["xid", "yid"])

    def test_multiple_select_duplicated(self):
        self.assert_css_selects("x, x", ["xid"])

    def test_multiple_select_sibling(self):
        self.assert_css_selects("x, y ~ p[lang=fr]", ["xid", "lang-fr"])

    def test_multiple_select_tag_and_direct_descendant(self):
        self.assert_css_selects("x, y > z", ["xid", "zidb"])

    def test_multiple_select_direct_descendant_and_tags(self):
        self.assert_css_selects(
            "div > x, y, z", ["xid", "yid", "zida", "zidb", "zidab", "zidac"]
        )

    def test_multiple_select_indirect_descendant(self):
        self.assert_css_selects(
            "div x,y,  z", ["xid", "yid", "zida", "zidb", "zidab", "zidac"]
        )

    def test_invalid_multiple_select(self):
        with pytest.raises(SelectorSyntaxError):
            self._soup.select(",x, y")
        with pytest.raises(SelectorSyntaxError):
            self._soup.select("x,,y")

    def test_multiple_select_attrs(self):
        self.assert_css_selects("p[lang=en], p[lang=en-gb]", ["lang-en", "lang-en-gb"])

    def test_multiple_select_ids(self):
        self.assert_css_selects(
            "x, y > z[id=zida], z[id=zidab], z[id=zidb]", ["xid", "zidb", "zidab"]
        )

    def test_multiple_select_nested(self):
        self.assert_css_selects("body > div > x, y > z", ["xid", "zidb"])

    def test_select_duplicate_elements(self):
        # When markup contains duplicate elements, a multiple select
        # will find all of them.
        markup = '<div class="c1"/><div class="c2"/><div class="c1"/>'
        soup = BeautifulSoup(markup, "html.parser")
        selected = soup.select(".c1, .c2")
        assert 3 == len(selected)

        # Verify that find_all finds the same elements, though because
        # of an implementation detail it finds them in a different
        # order.
        for element in soup.find_all(class_=["c1", "c2"]):
            assert element in selected

    def test_closest(self):
        inner = self._soup.find("div", id="inner")
        closest = inner.css.closest("div[id=main]")
        assert closest == self._soup.find("div", id="main")

    def test_match(self):
        inner = self._soup.find("div", id="inner")
        main = self._soup.find("div", id="main")
        assert inner.css.match("div[id=main]") is False
        assert main.css.match("div[id=main]") is True

    def test_iselect(self):
        gen = self._soup.css.iselect("h2")
        assert isinstance(gen, types.GeneratorType)
        [header2, header3] = gen
        assert header2["id"] == "header2"
        assert header3["id"] == "header3"

    def test_filter(self):
        inner = self._soup.find("div", id="inner")
        results = inner.css.filter("h2")
        assert len(inner.css.filter("h2")) == 2

        results = inner.css.filter("h2[id=header3]")
        assert isinstance(results, ResultSet)
        [result] = results
        assert result["id"] == "header3"

    def test_escape(self):
        m = self._soup.css.escape
        assert m(".foo#bar") == "\\.foo\\#bar"
        assert m("()[]{}") == "\\(\\)\\[\\]\\{\\}"
        assert m(".foo") == self._soup.css.escape(".foo")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_to_dict.py ===
from collections import (
    OrderedDict,
    defaultdict,
)
from datetime import datetime

import numpy as np
import pytest
import pytz

from pandas import (
    NA,
    DataFrame,
    Index,
    Interval,
    MultiIndex,
    Period,
    Series,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm


class TestDataFrameToDict:
    def test_to_dict_timestamp(self):
        # GH#11247
        # split/records producing np.datetime64 rather than Timestamps
        # on datetime64[ns] dtypes only

        tsmp = Timestamp("20130101")
        test_data = DataFrame({"A": [tsmp, tsmp], "B": [tsmp, tsmp]})
        test_data_mixed = DataFrame({"A": [tsmp, tsmp], "B": [1, 2]})

        expected_records = [{"A": tsmp, "B": tsmp}, {"A": tsmp, "B": tsmp}]
        expected_records_mixed = [{"A": tsmp, "B": 1}, {"A": tsmp, "B": 2}]

        assert test_data.to_dict(orient="records") == expected_records
        assert test_data_mixed.to_dict(orient="records") == expected_records_mixed

        expected_series = {
            "A": Series([tsmp, tsmp], name="A"),
            "B": Series([tsmp, tsmp], name="B"),
        }
        expected_series_mixed = {
            "A": Series([tsmp, tsmp], name="A"),
            "B": Series([1, 2], name="B"),
        }

        tm.assert_dict_equal(test_data.to_dict(orient="series"), expected_series)
        tm.assert_dict_equal(
            test_data_mixed.to_dict(orient="series"), expected_series_mixed
        )

        expected_split = {
            "index": [0, 1],
            "data": [[tsmp, tsmp], [tsmp, tsmp]],
            "columns": ["A", "B"],
        }
        expected_split_mixed = {
            "index": [0, 1],
            "data": [[tsmp, 1], [tsmp, 2]],
            "columns": ["A", "B"],
        }

        tm.assert_dict_equal(test_data.to_dict(orient="split"), expected_split)
        tm.assert_dict_equal(
            test_data_mixed.to_dict(orient="split"), expected_split_mixed
        )

    def test_to_dict_index_not_unique_with_index_orient(self):
        # GH#22801
        # Data loss when indexes are not unique. Raise ValueError.
        df = DataFrame({"a": [1, 2], "b": [0.5, 0.75]}, index=["A", "A"])
        msg = "DataFrame index must be unique for orient='index'"
        with pytest.raises(ValueError, match=msg):
            df.to_dict(orient="index")

    def test_to_dict_invalid_orient(self):
        df = DataFrame({"A": [0, 1]})
        msg = "orient 'xinvalid' not understood"
        with pytest.raises(ValueError, match=msg):
            df.to_dict(orient="xinvalid")

    @pytest.mark.parametrize("orient", ["d", "l", "r", "sp", "s", "i"])
    def test_to_dict_short_orient_raises(self, orient):
        # GH#32515
        df = DataFrame({"A": [0, 1]})
        with pytest.raises(ValueError, match="not understood"):
            df.to_dict(orient=orient)

    @pytest.mark.parametrize("mapping", [dict, defaultdict(list), OrderedDict])
    def test_to_dict(self, mapping):
        # orient= should only take the listed options
        # see GH#32515
        test_data = {"A": {"1": 1, "2": 2}, "B": {"1": "1", "2": "2", "3": "3"}}

        # GH#16122
        recons_data = DataFrame(test_data).to_dict(into=mapping)

        for k, v in test_data.items():
            for k2, v2 in v.items():
                assert v2 == recons_data[k][k2]

        recons_data = DataFrame(test_data).to_dict("list", into=mapping)

        for k, v in test_data.items():
            for k2, v2 in v.items():
                assert v2 == recons_data[k][int(k2) - 1]

        recons_data = DataFrame(test_data).to_dict("series", into=mapping)

        for k, v in test_data.items():
            for k2, v2 in v.items():
                assert v2 == recons_data[k][k2]

        recons_data = DataFrame(test_data).to_dict("split", into=mapping)
        expected_split = {
            "columns": ["A", "B"],
            "index": ["1", "2", "3"],
            "data": [[1.0, "1"], [2.0, "2"], [np.nan, "3"]],
        }
        tm.assert_dict_equal(recons_data, expected_split)

        recons_data = DataFrame(test_data).to_dict("records", into=mapping)
        expected_records = [
            {"A": 1.0, "B": "1"},
            {"A": 2.0, "B": "2"},
            {"A": np.nan, "B": "3"},
        ]
        assert isinstance(recons_data, list)
        assert len(recons_data) == 3
        for left, right in zip(recons_data, expected_records):
            tm.assert_dict_equal(left, right)

        # GH#10844
        recons_data = DataFrame(test_data).to_dict("index")

        for k, v in test_data.items():
            for k2, v2 in v.items():
                assert v2 == recons_data[k2][k]

        df = DataFrame(test_data)
        df["duped"] = df[df.columns[0]]
        recons_data = df.to_dict("index")
        comp_data = test_data.copy()
        comp_data["duped"] = comp_data[df.columns[0]]
        for k, v in comp_data.items():
            for k2, v2 in v.items():
                assert v2 == recons_data[k2][k]

    @pytest.mark.parametrize("mapping", [list, defaultdict, []])
    def test_to_dict_errors(self, mapping):
        # GH#16122
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)))
        msg = "|".join(
            [
                "unsupported type: <class 'list'>",
                r"to_dict\(\) only accepts initialized defaultdicts",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            df.to_dict(into=mapping)

    def test_to_dict_not_unique_warning(self):
        # GH#16927: When converting to a dict, if a column has a non-unique name
        # it will be dropped, throwing a warning.
        df = DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
        with tm.assert_produces_warning(UserWarning):
            df.to_dict()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize(
        "orient,expected",
        [
            ("list", {"A": [2, 5], "B": [3, 6]}),
            ("dict", {"A": {0: 2, 1: 5}, "B": {0: 3, 1: 6}}),
        ],
    )
    def test_to_dict_not_unique(self, orient, expected):
        # GH#54824: This is to make sure that dataframes with non-unique column
        # would have uniform behavior throughout different orients
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "A", "B"])
        result = df.to_dict(orient)
        assert result == expected

    # orient - orient argument to to_dict function
    # item_getter - function for extracting value from
    # the resulting dict using column name and index
    @pytest.mark.parametrize(
        "orient,item_getter",
        [
            ("dict", lambda d, col, idx: d[col][idx]),
            ("records", lambda d, col, idx: d[idx][col]),
            ("list", lambda d, col, idx: d[col][idx]),
            ("split", lambda d, col, idx: d["data"][idx][d["columns"].index(col)]),
            ("index", lambda d, col, idx: d[idx][col]),
        ],
    )
    def test_to_dict_box_scalars(self, orient, item_getter):
        # GH#14216, GH#23753
        # make sure that we are boxing properly
        df = DataFrame({"a": [1, 2], "b": [0.1, 0.2]})
        result = df.to_dict(orient=orient)
        assert isinstance(item_getter(result, "a", 0), int)
        assert isinstance(item_getter(result, "b", 0), float)

    def test_to_dict_tz(self):
        # GH#18372 When converting to dict with orient='records' columns of
        # datetime that are tz-aware were not converted to required arrays
        data = [
            (datetime(2017, 11, 18, 21, 53, 0, 219225, tzinfo=pytz.utc),),
            (datetime(2017, 11, 18, 22, 6, 30, 61810, tzinfo=pytz.utc),),
        ]
        df = DataFrame(list(data), columns=["d"])

        result = df.to_dict(orient="records")
        expected = [
            {"d": Timestamp("2017-11-18 21:53:00.219225+0000", tz=pytz.utc)},
            {"d": Timestamp("2017-11-18 22:06:30.061810+0000", tz=pytz.utc)},
        ]
        tm.assert_dict_equal(result[0], expected[0])
        tm.assert_dict_equal(result[1], expected[1])

    @pytest.mark.parametrize(
        "into, expected",
        [
            (
                dict,
                {
                    0: {"int_col": 1, "float_col": 1.0},
                    1: {"int_col": 2, "float_col": 2.0},
                    2: {"int_col": 3, "float_col": 3.0},
                },
            ),
            (
                OrderedDict,
                OrderedDict(
                    [
                        (0, {"int_col": 1, "float_col": 1.0}),
                        (1, {"int_col": 2, "float_col": 2.0}),
                        (2, {"int_col": 3, "float_col": 3.0}),
                    ]
                ),
            ),
            (
                defaultdict(dict),
                defaultdict(
                    dict,
                    {
                        0: {"int_col": 1, "float_col": 1.0},
                        1: {"int_col": 2, "float_col": 2.0},
                        2: {"int_col": 3, "float_col": 3.0},
                    },
                ),
            ),
        ],
    )
    def test_to_dict_index_dtypes(self, into, expected):
        # GH#18580
        # When using to_dict(orient='index') on a dataframe with int
        # and float columns only the int columns were cast to float

        df = DataFrame({"int_col": [1, 2, 3], "float_col": [1.0, 2.0, 3.0]})

        result = df.to_dict(orient="index", into=into)
        cols = ["int_col", "float_col"]
        result = DataFrame.from_dict(result, orient="index")[cols]
        expected = DataFrame.from_dict(expected, orient="index")[cols]
        tm.assert_frame_equal(result, expected)

    def test_to_dict_numeric_names(self):
        # GH#24940
        df = DataFrame({str(i): [i] for i in range(5)})
        result = set(df.to_dict("records")[0].keys())
        expected = set(df.columns)
        assert result == expected

    def test_to_dict_wide(self):
        # GH#24939
        df = DataFrame({(f"A_{i:d}"): [i] for i in range(256)})
        result = df.to_dict("records")[0]
        expected = {f"A_{i:d}": i for i in range(256)}
        assert result == expected

    @pytest.mark.parametrize(
        "data,dtype",
        (
            ([True, True, False], bool),
            [
                [
                    datetime(2018, 1, 1),
                    datetime(2019, 2, 2),
                    datetime(2020, 3, 3),
                ],
                Timestamp,
            ],
            [[1.0, 2.0, 3.0], float],
            [[1, 2, 3], int],
            [["X", "Y", "Z"], str],
        ),
    )
    def test_to_dict_orient_dtype(self, data, dtype):
        # GH22620 & GH21256

        df = DataFrame({"a": data})
        d = df.to_dict(orient="records")
        assert all(type(record["a"]) is dtype for record in d)

    @pytest.mark.parametrize(
        "data,expected_dtype",
        (
            [np.uint64(2), int],
            [np.int64(-9), int],
            [np.float64(1.1), float],
            [np.bool_(True), bool],
            [np.datetime64("2005-02-25"), Timestamp],
        ),
    )
    def test_to_dict_scalar_constructor_orient_dtype(self, data, expected_dtype):
        # GH22620 & GH21256

        df = DataFrame({"a": data}, index=[0])
        d = df.to_dict(orient="records")
        result = type(d[0]["a"])
        assert result is expected_dtype

    def test_to_dict_mixed_numeric_frame(self):
        # GH 12859
        df = DataFrame({"a": [1.0], "b": [9.0]})
        result = df.reset_index().to_dict("records")
        expected = [{"index": 0, "a": 1.0, "b": 9.0}]
        assert result == expected

    @pytest.mark.parametrize(
        "index",
        [
            None,
            Index(["aa", "bb"]),
            Index(["aa", "bb"], name="cc"),
            MultiIndex.from_tuples([("a", "b"), ("a", "c")]),
            MultiIndex.from_tuples([("a", "b"), ("a", "c")], names=["n1", "n2"]),
        ],
    )
    @pytest.mark.parametrize(
        "columns",
        [
            ["x", "y"],
            Index(["x", "y"]),
            Index(["x", "y"], name="z"),
            MultiIndex.from_tuples([("x", 1), ("y", 2)]),
            MultiIndex.from_tuples([("x", 1), ("y", 2)], names=["z1", "z2"]),
        ],
    )
    def test_to_dict_orient_tight(self, index, columns):
        df = DataFrame.from_records(
            [[1, 3], [2, 4]],
            columns=columns,
            index=index,
        )
        roundtrip = DataFrame.from_dict(df.to_dict(orient="tight"), orient="tight")

        tm.assert_frame_equal(df, roundtrip)

    @pytest.mark.parametrize(
        "orient",
        ["dict", "list", "split", "records", "index", "tight"],
    )
    @pytest.mark.parametrize(
        "data,expected_types",
        (
            (
                {
                    "a": [np.int64(1), 1, np.int64(3)],
                    "b": [np.float64(1.0), 2.0, np.float64(3.0)],
                    "c": [np.float64(1.0), 2, np.int64(3)],
                    "d": [np.float64(1.0), "a", np.int64(3)],
                    "e": [np.float64(1.0), ["a"], np.int64(3)],
                    "f": [np.float64(1.0), ("a",), np.int64(3)],
                },
                {
                    "a": [int, int, int],
                    "b": [float, float, float],
                    "c": [float, float, float],
                    "d": [float, str, int],
                    "e": [float, list, int],
                    "f": [float, tuple, int],
                },
            ),
            (
                {
                    "a": [1, 2, 3],
                    "b": [1.1, 2.2, 3.3],
                },
                {
                    "a": [int, int, int],
                    "b": [float, float, float],
                },
            ),
            (  # Make sure we have one df which is all object type cols
                {
                    "a": [1, "hello", 3],
                    "b": [1.1, "world", 3.3],
                },
                {
                    "a": [int, str, int],
                    "b": [float, str, float],
                },
            ),
        ),
    )
    def test_to_dict_returns_native_types(self, orient, data, expected_types):
        # GH 46751
        # Tests we get back native types for all orient types
        df = DataFrame(data)
        result = df.to_dict(orient)
        if orient == "dict":
            assertion_iterator = (
                (i, key, value)
                for key, index_value_map in result.items()
                for i, value in index_value_map.items()
            )
        elif orient == "list":
            assertion_iterator = (
                (i, key, value)
                for key, values in result.items()
                for i, value in enumerate(values)
            )
        elif orient in {"split", "tight"}:
            assertion_iterator = (
                (i, key, result["data"][i][j])
                for i in result["index"]
                for j, key in enumerate(result["columns"])
            )
        elif orient == "records":
            assertion_iterator = (
                (i, key, value)
                for i, record in enumerate(result)
                for key, value in record.items()
            )
        elif orient == "index":
            assertion_iterator = (
                (i, key, value)
                for i, record in result.items()
                for key, value in record.items()
            )

        for i, key, value in assertion_iterator:
            assert value == data[key][i]
            assert type(value) is expected_types[key][i]

    @pytest.mark.parametrize("orient", ["dict", "list", "series", "records", "index"])
    def test_to_dict_index_false_error(self, orient):
        # GH#46398
        df = DataFrame({"col1": [1, 2], "col2": [3, 4]}, index=["row1", "row2"])
        msg = "'index=False' is only valid when 'orient' is 'split' or 'tight'"
        with pytest.raises(ValueError, match=msg):
            df.to_dict(orient=orient, index=False)

    @pytest.mark.parametrize(
        "orient, expected",
        [
            ("split", {"columns": ["col1", "col2"], "data": [[1, 3], [2, 4]]}),
            (
                "tight",
                {
                    "columns": ["col1", "col2"],
                    "data": [[1, 3], [2, 4]],
                    "column_names": [None],
                },
            ),
        ],
    )
    def test_to_dict_index_false(self, orient, expected):
        # GH#46398
        df = DataFrame({"col1": [1, 2], "col2": [3, 4]}, index=["row1", "row2"])
        result = df.to_dict(orient=orient, index=False)
        tm.assert_dict_equal(result, expected)

    @pytest.mark.parametrize(
        "orient, expected",
        [
            ("dict", {"a": {0: 1, 1: None}}),
            ("list", {"a": [1, None]}),
            ("split", {"index": [0, 1], "columns": ["a"], "data": [[1], [None]]}),
            (
                "tight",
                {
                    "index": [0, 1],
                    "columns": ["a"],
                    "data": [[1], [None]],
                    "index_names": [None],
                    "column_names": [None],
                },
            ),
            ("records", [{"a": 1}, {"a": None}]),
            ("index", {0: {"a": 1}, 1: {"a": None}}),
        ],
    )
    def test_to_dict_na_to_none(self, orient, expected):
        # GH#50795
        df = DataFrame({"a": [1, NA]}, dtype="Int64")
        result = df.to_dict(orient=orient)
        assert result == expected

    def test_to_dict_masked_native_python(self):
        # GH#34665
        df = DataFrame({"a": Series([1, 2], dtype="Int64"), "B": 1})
        result = df.to_dict(orient="records")
        assert isinstance(result[0]["a"], int)

        df = DataFrame({"a": Series([1, NA], dtype="Int64"), "B": 1})
        result = df.to_dict(orient="records")
        assert isinstance(result[0]["a"], int)

    def test_to_dict_pos_args_deprecation(self):
        # GH-54229
        df = DataFrame({"a": [1, 2, 3]})
        msg = (
            r"Starting with pandas version 3.0 all arguments of to_dict except for the "
            r"argument 'orient' will be keyword-only."
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df.to_dict("records", {})


@pytest.mark.parametrize(
    "val", [Timestamp(2020, 1, 1), Timedelta(1), Period("2020"), Interval(1, 2)]
)
def test_to_dict_list_pd_scalars(val):
    # GH 54824
    df = DataFrame({"a": [val]})
    result = df.to_dict(orient="list")
    expected = {"a": [val]}
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\interval\test_constructors.py ===
from functools import partial

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas.core.dtypes.common import is_unsigned_integer_dtype
from pandas.core.dtypes.dtypes import IntervalDtype

from pandas import (
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    Index,
    Interval,
    IntervalIndex,
    date_range,
    notna,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
import pandas.core.common as com


@pytest.fixture(params=[None, "foo"])
def name(request):
    return request.param


class ConstructorTests:
    """
    Common tests for all variations of IntervalIndex construction. Input data
    to be supplied in breaks format, then converted by the subclass method
    get_kwargs_from_breaks to the expected format.
    """

    @pytest.fixture(
        params=[
            ([3, 14, 15, 92, 653], np.int64),
            (np.arange(10, dtype="int64"), np.int64),
            (Index(np.arange(-10, 11, dtype=np.int64)), np.int64),
            (Index(np.arange(10, 31, dtype=np.uint64)), np.uint64),
            (Index(np.arange(20, 30, 0.5), dtype=np.float64), np.float64),
            (date_range("20180101", periods=10), "<M8[ns]"),
            (
                date_range("20180101", periods=10, tz="US/Eastern"),
                "datetime64[ns, US/Eastern]",
            ),
            (timedelta_range("1 day", periods=10), "<m8[ns]"),
        ]
    )
    def breaks_and_expected_subtype(self, request):
        return request.param

    def test_constructor(self, constructor, breaks_and_expected_subtype, closed, name):
        breaks, expected_subtype = breaks_and_expected_subtype

        result_kwargs = self.get_kwargs_from_breaks(breaks, closed)

        result = constructor(closed=closed, name=name, **result_kwargs)

        assert result.closed == closed
        assert result.name == name
        assert result.dtype.subtype == expected_subtype
        tm.assert_index_equal(result.left, Index(breaks[:-1], dtype=expected_subtype))
        tm.assert_index_equal(result.right, Index(breaks[1:], dtype=expected_subtype))

    @pytest.mark.parametrize(
        "breaks, subtype",
        [
            (Index([0, 1, 2, 3, 4], dtype=np.int64), "float64"),
            (Index([0, 1, 2, 3, 4], dtype=np.int64), "datetime64[ns]"),
            (Index([0, 1, 2, 3, 4], dtype=np.int64), "timedelta64[ns]"),
            (Index([0, 1, 2, 3, 4], dtype=np.float64), "int64"),
            (date_range("2017-01-01", periods=5), "int64"),
            (timedelta_range("1 day", periods=5), "int64"),
        ],
    )
    def test_constructor_dtype(self, constructor, breaks, subtype):
        # GH 19262: conversion via dtype parameter
        expected_kwargs = self.get_kwargs_from_breaks(breaks.astype(subtype))
        expected = constructor(**expected_kwargs)

        result_kwargs = self.get_kwargs_from_breaks(breaks)
        iv_dtype = IntervalDtype(subtype, "right")
        for dtype in (iv_dtype, str(iv_dtype)):
            result = constructor(dtype=dtype, **result_kwargs)
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "breaks",
        [
            Index([0, 1, 2, 3, 4], dtype=np.int64),
            Index([0, 1, 2, 3, 4], dtype=np.uint64),
            Index([0, 1, 2, 3, 4], dtype=np.float64),
            date_range("2017-01-01", periods=5),
            timedelta_range("1 day", periods=5),
        ],
    )
    def test_constructor_pass_closed(self, constructor, breaks):
        # not passing closed to IntervalDtype, but to IntervalArray constructor
        iv_dtype = IntervalDtype(breaks.dtype)

        result_kwargs = self.get_kwargs_from_breaks(breaks)

        for dtype in (iv_dtype, str(iv_dtype)):
            with tm.assert_produces_warning(None):
                result = constructor(dtype=dtype, closed="left", **result_kwargs)
            assert result.dtype.closed == "left"

    @pytest.mark.parametrize("breaks", [[np.nan] * 2, [np.nan] * 4, [np.nan] * 50])
    def test_constructor_nan(self, constructor, breaks, closed):
        # GH 18421
        result_kwargs = self.get_kwargs_from_breaks(breaks)
        result = constructor(closed=closed, **result_kwargs)

        expected_subtype = np.float64
        expected_values = np.array(breaks[:-1], dtype=object)

        assert result.closed == closed
        assert result.dtype.subtype == expected_subtype
        tm.assert_numpy_array_equal(np.array(result), expected_values)

    @pytest.mark.parametrize(
        "breaks",
        [
            [],
            np.array([], dtype="int64"),
            np.array([], dtype="uint64"),
            np.array([], dtype="float64"),
            np.array([], dtype="datetime64[ns]"),
            np.array([], dtype="timedelta64[ns]"),
        ],
    )
    def test_constructor_empty(self, constructor, breaks, closed):
        # GH 18421
        result_kwargs = self.get_kwargs_from_breaks(breaks)
        result = constructor(closed=closed, **result_kwargs)

        expected_values = np.array([], dtype=object)
        expected_subtype = getattr(breaks, "dtype", np.int64)

        assert result.empty
        assert result.closed == closed
        assert result.dtype.subtype == expected_subtype
        tm.assert_numpy_array_equal(np.array(result), expected_values)

    @pytest.mark.parametrize(
        "breaks",
        [
            tuple("0123456789"),
            list("abcdefghij"),
            np.array(list("abcdefghij"), dtype=object),
            np.array(list("abcdefghij"), dtype="<U1"),
        ],
    )
    def test_constructor_string(self, constructor, breaks):
        # GH 19016
        msg = (
            "category, object, and string subtypes are not supported "
            "for IntervalIndex"
        )
        with pytest.raises(TypeError, match=msg):
            constructor(**self.get_kwargs_from_breaks(breaks))

    @pytest.mark.parametrize("cat_constructor", [Categorical, CategoricalIndex])
    def test_constructor_categorical_valid(self, constructor, cat_constructor):
        # GH 21243/21253

        breaks = np.arange(10, dtype="int64")
        expected = IntervalIndex.from_breaks(breaks)

        cat_breaks = cat_constructor(breaks)
        result_kwargs = self.get_kwargs_from_breaks(cat_breaks)
        result = constructor(**result_kwargs)
        tm.assert_index_equal(result, expected)

    def test_generic_errors(self, constructor):
        # filler input data to be used when supplying invalid kwargs
        filler = self.get_kwargs_from_breaks(range(10))

        # invalid closed
        msg = "closed must be one of 'right', 'left', 'both', 'neither'"
        with pytest.raises(ValueError, match=msg):
            constructor(closed="invalid", **filler)

        # unsupported dtype
        msg = "dtype must be an IntervalDtype, got int64"
        with pytest.raises(TypeError, match=msg):
            constructor(dtype="int64", **filler)

        # invalid dtype
        msg = "data type [\"']invalid[\"'] not understood"
        with pytest.raises(TypeError, match=msg):
            constructor(dtype="invalid", **filler)

        # no point in nesting periods in an IntervalIndex
        periods = period_range("2000-01-01", periods=10)
        periods_kwargs = self.get_kwargs_from_breaks(periods)
        msg = "Period dtypes are not supported, use a PeriodIndex instead"
        with pytest.raises(ValueError, match=msg):
            constructor(**periods_kwargs)

        # decreasing values
        decreasing_kwargs = self.get_kwargs_from_breaks(range(10, -1, -1))
        msg = "left side of interval must be <= right side"
        with pytest.raises(ValueError, match=msg):
            constructor(**decreasing_kwargs)


class TestFromArrays(ConstructorTests):
    """Tests specific to IntervalIndex.from_arrays"""

    @pytest.fixture
    def constructor(self):
        return IntervalIndex.from_arrays

    def get_kwargs_from_breaks(self, breaks, closed="right"):
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by IntervalIndex.from_arrays
        """
        return {"left": breaks[:-1], "right": breaks[1:]}

    def test_constructor_errors(self):
        # GH 19016: categorical data
        data = Categorical(list("01234abcde"), ordered=True)
        msg = (
            "category, object, and string subtypes are not supported "
            "for IntervalIndex"
        )
        with pytest.raises(TypeError, match=msg):
            IntervalIndex.from_arrays(data[:-1], data[1:])

        # unequal length
        left = [0, 1, 2]
        right = [2, 3]
        msg = "left and right must have the same length"
        with pytest.raises(ValueError, match=msg):
            IntervalIndex.from_arrays(left, right)

    @pytest.mark.parametrize(
        "left_subtype, right_subtype", [(np.int64, np.float64), (np.float64, np.int64)]
    )
    def test_mixed_float_int(self, left_subtype, right_subtype):
        """mixed int/float left/right results in float for both sides"""
        left = np.arange(9, dtype=left_subtype)
        right = np.arange(1, 10, dtype=right_subtype)
        result = IntervalIndex.from_arrays(left, right)

        expected_left = Index(left, dtype=np.float64)
        expected_right = Index(right, dtype=np.float64)
        expected_subtype = np.float64

        tm.assert_index_equal(result.left, expected_left)
        tm.assert_index_equal(result.right, expected_right)
        assert result.dtype.subtype == expected_subtype

    @pytest.mark.parametrize("interval_cls", [IntervalArray, IntervalIndex])
    def test_from_arrays_mismatched_datetimelike_resos(self, interval_cls):
        # GH#55714
        left = date_range("2016-01-01", periods=3, unit="s")
        right = date_range("2017-01-01", periods=3, unit="ms")
        result = interval_cls.from_arrays(left, right)
        expected = interval_cls.from_arrays(left.as_unit("ms"), right)
        tm.assert_equal(result, expected)

        # td64
        left2 = left - left[0]
        right2 = right - left[0]
        result2 = interval_cls.from_arrays(left2, right2)
        expected2 = interval_cls.from_arrays(left2.as_unit("ms"), right2)
        tm.assert_equal(result2, expected2)

        # dt64tz
        left3 = left.tz_localize("UTC")
        right3 = right.tz_localize("UTC")
        result3 = interval_cls.from_arrays(left3, right3)
        expected3 = interval_cls.from_arrays(left3.as_unit("ms"), right3)
        tm.assert_equal(result3, expected3)


class TestFromBreaks(ConstructorTests):
    """Tests specific to IntervalIndex.from_breaks"""

    @pytest.fixture
    def constructor(self):
        return IntervalIndex.from_breaks

    def get_kwargs_from_breaks(self, breaks, closed="right"):
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by IntervalIndex.from_breaks
        """
        return {"breaks": breaks}

    def test_constructor_errors(self):
        # GH 19016: categorical data
        data = Categorical(list("01234abcde"), ordered=True)
        msg = (
            "category, object, and string subtypes are not supported "
            "for IntervalIndex"
        )
        with pytest.raises(TypeError, match=msg):
            IntervalIndex.from_breaks(data)

    def test_length_one(self):
        """breaks of length one produce an empty IntervalIndex"""
        breaks = [0]
        result = IntervalIndex.from_breaks(breaks)
        expected = IntervalIndex.from_breaks([])
        tm.assert_index_equal(result, expected)

    def test_left_right_dont_share_data(self):
        # GH#36310
        breaks = np.arange(5)
        result = IntervalIndex.from_breaks(breaks)._data
        assert result._left.base is None or result._left.base is not result._right.base


class TestFromTuples(ConstructorTests):
    """Tests specific to IntervalIndex.from_tuples"""

    @pytest.fixture
    def constructor(self):
        return IntervalIndex.from_tuples

    def get_kwargs_from_breaks(self, breaks, closed="right"):
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by IntervalIndex.from_tuples
        """
        if is_unsigned_integer_dtype(breaks):
            pytest.skip(f"{breaks.dtype} not relevant IntervalIndex.from_tuples tests")

        if len(breaks) == 0:
            return {"data": breaks}

        tuples = list(zip(breaks[:-1], breaks[1:]))
        if isinstance(breaks, (list, tuple)):
            return {"data": tuples}
        elif isinstance(getattr(breaks, "dtype", None), CategoricalDtype):
            return {"data": breaks._constructor(tuples)}
        return {"data": com.asarray_tuplesafe(tuples)}

    def test_constructor_errors(self):
        # non-tuple
        tuples = [(0, 1), 2, (3, 4)]
        msg = "IntervalIndex.from_tuples received an invalid item, 2"
        with pytest.raises(TypeError, match=msg.format(t=tuples)):
            IntervalIndex.from_tuples(tuples)

        # too few/many items
        tuples = [(0, 1), (2,), (3, 4)]
        msg = "IntervalIndex.from_tuples requires tuples of length 2, got {t}"
        with pytest.raises(ValueError, match=msg.format(t=tuples)):
            IntervalIndex.from_tuples(tuples)

        tuples = [(0, 1), (2, 3, 4), (5, 6)]
        with pytest.raises(ValueError, match=msg.format(t=tuples)):
            IntervalIndex.from_tuples(tuples)

    def test_na_tuples(self):
        # tuple (NA, NA) evaluates the same as NA as an element
        na_tuple = [(0, 1), (np.nan, np.nan), (2, 3)]
        idx_na_tuple = IntervalIndex.from_tuples(na_tuple)
        idx_na_element = IntervalIndex.from_tuples([(0, 1), np.nan, (2, 3)])
        tm.assert_index_equal(idx_na_tuple, idx_na_element)


class TestClassConstructors(ConstructorTests):
    """Tests specific to the IntervalIndex/Index constructors"""

    @pytest.fixture(
        params=[IntervalIndex, partial(Index, dtype="interval")],
        ids=["IntervalIndex", "Index"],
    )
    def klass(self, request):
        # We use a separate fixture here to include Index.__new__ with dtype kwarg
        return request.param

    @pytest.fixture
    def constructor(self):
        return IntervalIndex

    def get_kwargs_from_breaks(self, breaks, closed="right"):
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by the IntervalIndex/Index constructors
        """
        if is_unsigned_integer_dtype(breaks):
            pytest.skip(f"{breaks.dtype} not relevant for class constructor tests")

        if len(breaks) == 0:
            return {"data": breaks}

        ivs = [
            Interval(left, right, closed) if notna(left) else left
            for left, right in zip(breaks[:-1], breaks[1:])
        ]

        if isinstance(breaks, list):
            return {"data": ivs}
        elif isinstance(getattr(breaks, "dtype", None), CategoricalDtype):
            return {"data": breaks._constructor(ivs)}
        return {"data": np.array(ivs, dtype=object)}

    def test_generic_errors(self, constructor):
        """
        override the base class implementation since errors are handled
        differently; checks unnecessary since caught at the Interval level
        """

    def test_constructor_string(self):
        # GH23013
        # When forming the interval from breaks,
        # the interval of strings is already forbidden.
        pass

    def test_constructor_errors(self, klass):
        # mismatched closed within intervals with no constructor override
        ivs = [Interval(0, 1, closed="right"), Interval(2, 3, closed="left")]
        msg = "intervals must all be closed on the same side"
        with pytest.raises(ValueError, match=msg):
            klass(ivs)

        # scalar
        msg = (
            r"(IntervalIndex|Index)\(...\) must be called with a collection of "
            "some kind, 5 was passed"
        )
        with pytest.raises(TypeError, match=msg):
            klass(5)

        # not an interval; dtype depends on 32bit/windows builds
        msg = "type <class 'numpy.int(32|64)'> with value 0 is not an interval"
        with pytest.raises(TypeError, match=msg):
            klass([0, 1])

    @pytest.mark.parametrize(
        "data, closed",
        [
            ([], "both"),
            ([np.nan, np.nan], "neither"),
            (
                [Interval(0, 3, closed="neither"), Interval(2, 5, closed="neither")],
                "left",
            ),
            (
                [Interval(0, 3, closed="left"), Interval(2, 5, closed="right")],
                "neither",
            ),
            (IntervalIndex.from_breaks(range(5), closed="both"), "right"),
        ],
    )
    def test_override_inferred_closed(self, constructor, data, closed):
        # GH 19370
        if isinstance(data, IntervalIndex):
            tuples = data.to_tuples()
        else:
            tuples = [(iv.left, iv.right) if notna(iv) else iv for iv in data]
        expected = IntervalIndex.from_tuples(tuples, closed=closed)
        result = constructor(data, closed=closed)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "values_constructor", [list, np.array, IntervalIndex, IntervalArray]
    )
    def test_index_object_dtype(self, values_constructor):
        # Index(intervals, dtype=object) is an Index (not an IntervalIndex)
        intervals = [Interval(0, 1), Interval(1, 2), Interval(2, 3)]
        values = values_constructor(intervals)
        result = Index(values, dtype=object)

        assert type(result) is Index
        tm.assert_numpy_array_equal(result.values, np.array(values))

    def test_index_mixed_closed(self):
        # GH27172
        intervals = [
            Interval(0, 1, closed="left"),
            Interval(1, 2, closed="right"),
            Interval(2, 3, closed="neither"),
            Interval(3, 4, closed="both"),
        ]
        result = Index(intervals)
        expected = Index(intervals, dtype=object)
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("timezone", ["UTC", "US/Pacific", "GMT"])
def test_interval_index_subtype(timezone, inclusive_endpoints_fixture):
    # GH#46999
    dates = date_range("2022", periods=3, tz=timezone)
    dtype = f"interval[datetime64[ns, {timezone}], {inclusive_endpoints_fixture}]"
    result = IntervalIndex.from_arrays(
        ["2022-01-01", "2022-01-02"],
        ["2022-01-02", "2022-01-03"],
        closed=inclusive_endpoints_fixture,
        dtype=dtype,
    )
    expected = IntervalIndex.from_arrays(
        dates[:-1], dates[1:], closed=inclusive_endpoints_fixture
    )
    tm.assert_index_equal(result, expected)


def test_dtype_closed_mismatch():
    # GH#38394 closed specified in both dtype and IntervalIndex constructor

    dtype = IntervalDtype(np.int64, "left")

    msg = "closed keyword does not match dtype.closed"
    with pytest.raises(ValueError, match=msg):
        IntervalIndex([], dtype=dtype, closed="neither")

    with pytest.raises(ValueError, match=msg):
        IntervalArray([], dtype=dtype, closed="neither")


@pytest.mark.parametrize(
    "dtype",
    ["Float64", pytest.param("float64[pyarrow]", marks=td.skip_if_no("pyarrow"))],
)
def test_ea_dtype(dtype):
    # GH#56765
    bins = [(0.0, 0.4), (0.4, 0.6)]
    interval_dtype = IntervalDtype(subtype=dtype, closed="left")
    result = IntervalIndex.from_tuples(bins, closed="left", dtype=interval_dtype)
    assert result.dtype == interval_dtype
    expected = IntervalIndex.from_tuples(bins, closed="left").astype(interval_dtype)
    tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_groebnertools.py ===
"""Tests for Groebner bases. """

from sympy.polys.groebnertools import (
    groebner, sig, sig_key,
    lbp, lbp_key, critical_pair,
    cp_key, is_rewritable_or_comparable,
    Sign, Polyn, Num, s_poly, f5_reduce,
    groebner_lcm, groebner_gcd, is_groebner,
    is_reduced
)

from sympy.polys.fglmtools import _representing_matrices
from sympy.polys.orderings import lex, grlex

from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ

from sympy.testing.pytest import slow
from sympy.polys import polyconfig as config

def _do_test_groebner():
    R, x,y = ring("x,y", QQ, lex)
    f = x**2 + 2*x*y**2
    g = x*y + 2*y**3 - 1

    assert groebner([f, g], R) == [x, y**3 - QQ(1,2)]

    R, y,x = ring("y,x", QQ, lex)
    f = 2*x**2*y + y**2
    g = 2*x**3 + x*y - 1

    assert groebner([f, g], R) == [y, x**3 - QQ(1,2)]

    R, x,y,z = ring("x,y,z", QQ, lex)
    f = x - z**2
    g = y - z**3

    assert groebner([f, g], R) == [f, g]

    R, x,y = ring("x,y", QQ, grlex)
    f = x**3 - 2*x*y
    g = x**2*y + x - 2*y**2

    assert groebner([f, g], R) == [x**2, x*y, -QQ(1,2)*x + y**2]

    R, x,y,z = ring("x,y,z", QQ, lex)
    f = -x**2 + y
    g = -x**3 + z

    assert groebner([f, g], R) == [x**2 - y, x*y - z, x*z - y**2, y**3 - z**2]

    R, x,y,z = ring("x,y,z", QQ, grlex)
    f = -x**2 + y
    g = -x**3 + z

    assert groebner([f, g], R) == [y**3 - z**2, x**2 - y, x*y - z, x*z - y**2]

    R, x,y,z = ring("x,y,z", QQ, lex)
    f = -x**2 + z
    g = -x**3 + y

    assert groebner([f, g], R) == [x**2 - z, x*y - z**2, x*z - y, y**2 - z**3]

    R, x,y,z = ring("x,y,z", QQ, grlex)
    f = -x**2 + z
    g = -x**3 + y

    assert groebner([f, g], R) == [-y**2 + z**3, x**2 - z, x*y - z**2, x*z - y]

    R, x,y,z = ring("x,y,z", QQ, lex)
    f = x - y**2
    g = -y**3 + z

    assert groebner([f, g], R) == [x - y**2, y**3 - z]

    R, x,y,z = ring("x,y,z", QQ, grlex)
    f = x - y**2
    g = -y**3 + z

    assert groebner([f, g], R) == [x**2 - y*z, x*y - z, -x + y**2]

    R, x,y,z = ring("x,y,z", QQ, lex)
    f = x - z**2
    g = y - z**3

    assert groebner([f, g], R) == [x - z**2, y - z**3]

    R, x,y,z = ring("x,y,z", QQ, grlex)
    f = x - z**2
    g = y - z**3

    assert groebner([f, g], R) == [x**2 - y*z, x*z - y, -x + z**2]

    R, x,y,z = ring("x,y,z", QQ, lex)
    f = -y**2 + z
    g = x - y**3

    assert groebner([f, g], R) == [x - y*z, y**2 - z]

    R, x,y,z = ring("x,y,z", QQ, grlex)
    f = -y**2 + z
    g = x - y**3

    assert groebner([f, g], R) == [-x**2 + z**3, x*y - z**2, y**2 - z, -x + y*z]

    R, x,y,z = ring("x,y,z", QQ, lex)
    f = y - z**2
    g = x - z**3

    assert groebner([f, g], R) == [x - z**3, y - z**2]

    R, x,y,z = ring("x,y,z", QQ, grlex)
    f = y - z**2
    g = x - z**3

    assert groebner([f, g], R) == [-x**2 + y**3, x*z - y**2, -x + y*z, -y + z**2]

    R, x,y,z = ring("x,y,z", QQ, lex)
    f = 4*x**2*y**2 + 4*x*y + 1
    g = x**2 + y**2 - 1

    assert groebner([f, g], R) == [
        x - 4*y**7 + 8*y**5 - 7*y**3 + 3*y,
        y**8 - 2*y**6 + QQ(3,2)*y**4 - QQ(1,2)*y**2 + QQ(1,16),
    ]

def test_groebner_buchberger():
    with config.using(groebner='buchberger'):
        _do_test_groebner()

def test_groebner_f5b():
    with config.using(groebner='f5b'):
        _do_test_groebner()

def _do_test_benchmark_minpoly():
    R, x,y,z = ring("x,y,z", QQ, lex)

    F = [x**3 + x + 1, y**2 + y + 1, (x + y) * z - (x**2 + y)]
    G = [x + QQ(155,2067)*z**5 - QQ(355,689)*z**4 + QQ(6062,2067)*z**3 - QQ(3687,689)*z**2 + QQ(6878,2067)*z - QQ(25,53),
         y + QQ(4,53)*z**5 - QQ(91,159)*z**4 + QQ(523,159)*z**3 - QQ(387,53)*z**2 + QQ(1043,159)*z - QQ(308,159),
         z**6 - 7*z**5 + 41*z**4 - 82*z**3 + 89*z**2 - 46*z + 13]

    assert groebner(F, R) == G

def test_benchmark_minpoly_buchberger():
    with config.using(groebner='buchberger'):
        _do_test_benchmark_minpoly()

def test_benchmark_minpoly_f5b():
    with config.using(groebner='f5b'):
        _do_test_benchmark_minpoly()


def test_benchmark_coloring():
    V = range(1, 12 + 1)
    E = [(1, 2), (2, 3), (1, 4), (1, 6), (1, 12), (2, 5), (2, 7), (3, 8), (3, 10),
         (4, 11), (4, 9), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),
         (11, 12), (5, 12), (5, 9), (6, 10), (7, 11), (8, 12), (3, 4)]

    R, V = xring([ "x%d" % v for v in V ], QQ, lex)
    E = [(V[i - 1], V[j - 1]) for i, j in E]

    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = V

    I3 = [x**3 - 1 for x in V]
    Ig = [x**2 + x*y + y**2 for x, y in E]

    I = I3 + Ig

    assert groebner(I[:-1], R) == [
        x1 + x11 + x12,
        x2 - x11,
        x3 - x12,
        x4 - x12,
        x5 + x11 + x12,
        x6 - x11,
        x7 - x12,
        x8 + x11 + x12,
        x9 - x11,
        x10 + x11 + x12,
        x11**2 + x11*x12 + x12**2,
        x12**3 - 1,
    ]

    assert groebner(I, R) == [1]


def _do_test_benchmark_katsura_3():
    R, x0,x1,x2 = ring("x:3", ZZ, lex)
    I = [x0 + 2*x1 + 2*x2 - 1,
         x0**2 + 2*x1**2 + 2*x2**2 - x0,
         2*x0*x1 + 2*x1*x2 - x1]

    assert groebner(I, R) == [
        -7 + 7*x0 + 8*x2 + 158*x2**2 - 420*x2**3,
        7*x1 + 3*x2 - 79*x2**2 + 210*x2**3,
        x2 + x2**2 - 40*x2**3 + 84*x2**4,
    ]

    R, x0,x1,x2 = ring("x:3", ZZ, grlex)
    I = [ i.set_ring(R) for i in I ]

    assert groebner(I, R) == [
        7*x1 + 3*x2 - 79*x2**2 + 210*x2**3,
        -x1 + x2 - 3*x2**2 + 5*x1**2,
        -x1 - 4*x2 + 10*x1*x2 + 12*x2**2,
        -1 + x0 + 2*x1 + 2*x2,
    ]

def test_benchmark_katsura3_buchberger():
    with config.using(groebner='buchberger'):
        _do_test_benchmark_katsura_3()

def test_benchmark_katsura3_f5b():
    with config.using(groebner='f5b'):
        _do_test_benchmark_katsura_3()

def _do_test_benchmark_katsura_4():
    R, x0,x1,x2,x3 = ring("x:4", ZZ, lex)
    I = [x0 + 2*x1 + 2*x2 + 2*x3 - 1,
         x0**2 + 2*x1**2 + 2*x2**2 + 2*x3**2 - x0,
         2*x0*x1 + 2*x1*x2 + 2*x2*x3 - x1,
         x1**2 + 2*x0*x2 + 2*x1*x3 - x2]

    assert groebner(I, R) == [
        5913075*x0 - 159690237696*x3**7 + 31246269696*x3**6 + 27439610544*x3**5 - 6475723368*x3**4 - 838935856*x3**3 + 275119624*x3**2 + 4884038*x3 - 5913075,
        1971025*x1 - 97197721632*x3**7 + 73975630752*x3**6 - 12121915032*x3**5 - 2760941496*x3**4 + 814792828*x3**3 - 1678512*x3**2 - 9158924*x3,
        5913075*x2 + 371438283744*x3**7 - 237550027104*x3**6 + 22645939824*x3**5 + 11520686172*x3**4 - 2024910556*x3**3 - 132524276*x3**2 + 30947828*x3,
        128304*x3**8 - 93312*x3**7 + 15552*x3**6 + 3144*x3**5 -
        1120*x3**4 + 36*x3**3 + 15*x3**2 - x3,
    ]

    R, x0,x1,x2,x3 = ring("x:4", ZZ, grlex)
    I = [ i.set_ring(R) for i in I ]

    assert groebner(I, R) == [
        393*x1 - 4662*x2**2 + 4462*x2*x3 - 59*x2 + 224532*x3**4 - 91224*x3**3 - 678*x3**2 + 2046*x3,
        -x1 + 196*x2**3 - 21*x2**2 + 60*x2*x3 - 18*x2 - 168*x3**3 + 83*x3**2 - 9*x3,
        -6*x1 + 1134*x2**2*x3 - 189*x2**2 - 466*x2*x3 + 32*x2 - 630*x3**3 + 57*x3**2 + 51*x3,
        33*x1 + 63*x2**2 + 2268*x2*x3**2 - 188*x2*x3 + 34*x2 + 2520*x3**3 - 849*x3**2 + 3*x3,
        7*x1**2 - x1 - 7*x2**2 - 24*x2*x3 + 3*x2 - 15*x3**2 + 5*x3,
        14*x1*x2 - x1 + 14*x2**2 + 18*x2*x3 - 4*x2 + 6*x3**2 - 2*x3,
        14*x1*x3 - x1 + 7*x2**2 + 32*x2*x3 - 4*x2 + 27*x3**2 - 9*x3,
        x0 + 2*x1 + 2*x2 + 2*x3 - 1,
    ]

def test_benchmark_kastura_4_buchberger():
    with config.using(groebner='buchberger'):
        _do_test_benchmark_katsura_4()

def test_benchmark_kastura_4_f5b():
    with config.using(groebner='f5b'):
        _do_test_benchmark_katsura_4()

def _do_test_benchmark_czichowski():
    R, x,t = ring("x,t", ZZ, lex)
    I = [9*x**8 + 36*x**7 - 32*x**6 - 252*x**5 - 78*x**4 + 468*x**3 + 288*x**2 - 108*x + 9,
         (-72 - 72*t)*x**7 + (-256 - 252*t)*x**6 + (192 + 192*t)*x**5 + (1280 + 1260*t)*x**4 + (312 + 312*t)*x**3 + (-404*t)*x**2 + (-576 - 576*t)*x + 96 + 108*t]

    assert groebner(I, R) == [
        3725588592068034903797967297424801242396746870413359539263038139343329273586196480000*x -
        160420835591776763325581422211936558925462474417709511019228211783493866564923546661604487873*t**7 -
        1406108495478033395547109582678806497509499966197028487131115097902188374051595011248311352864*t**6 -
        5241326875850889518164640374668786338033653548841427557880599579174438246266263602956254030352*t**5 -
        10758917262823299139373269714910672770004760114329943852726887632013485035262879510837043892416*t**4 -
        13119383576444715672578819534846747735372132018341964647712009275306635391456880068261130581248*t**3 -
        9491412317016197146080450036267011389660653495578680036574753839055748080962214787557853941760*t**2 -
        3767520915562795326943800040277726397326609797172964377014046018280260848046603967211258368000*t -
        632314652371226552085897259159210286886724229880266931574701654721512325555116066073245696000,
        610733380717522355121*t**8 +
        6243748742141230639968*t**7 +
        27761407182086143225024*t**6 +
        70066148869420956398592*t**5 +
        109701225644313784229376*t**4 +
        109009005495588442152960*t**3 +
        67072101084384786432000*t**2 +
        23339979742629593088000*t +
        3513592776846090240000,
    ]

    R, x,t = ring("x,t", ZZ, grlex)
    I = [ i.set_ring(R) for i in I ]

    assert groebner(I, R) == [
        16996618586000601590732959134095643086442*t**3*x -
        32936701459297092865176560282688198064839*t**3 +
        78592411049800639484139414821529525782364*t**2*x -
        120753953358671750165454009478961405619916*t**2 +
        120988399875140799712152158915653654637280*t*x -
        144576390266626470824138354942076045758736*t +
        60017634054270480831259316163620768960*x**2 +
        61976058033571109604821862786675242894400*x -
        56266268491293858791834120380427754600960,
        576689018321912327136790519059646508441672750656050290242749*t**4 +
        2326673103677477425562248201573604572527893938459296513327336*t**3 +
        110743790416688497407826310048520299245819959064297990236000*t**2*x +
        3308669114229100853338245486174247752683277925010505284338016*t**2 +
        323150205645687941261103426627818874426097912639158572428800*t*x +
        1914335199925152083917206349978534224695445819017286960055680*t +
        861662882561803377986838989464278045397192862768588480000*x**2 +
        235296483281783440197069672204341465480107019878814196672000*x +
        361850798943225141738895123621685122544503614946436727532800,
       -117584925286448670474763406733005510014188341867*t**3 +
        68566565876066068463853874568722190223721653044*t**2*x -
        435970731348366266878180788833437896139920683940*t**2 +
        196297602447033751918195568051376792491869233408*t*x -
        525011527660010557871349062870980202067479780112*t +
        517905853447200553360289634770487684447317120*x**3 +
        569119014870778921949288951688799397569321920*x**2 +
        138877356748142786670127389526667463202210102080*x -
        205109210539096046121625447192779783475018619520,
       -3725142681462373002731339445216700112264527*t**3 +
        583711207282060457652784180668273817487940*t**2*x -
        12381382393074485225164741437227437062814908*t**2 +
        151081054097783125250959636747516827435040*t*x**2 +
        1814103857455163948531448580501928933873280*t*x -
        13353115629395094645843682074271212731433648*t +
        236415091385250007660606958022544983766080*x**2 +
        1390443278862804663728298060085399578417600*x -
        4716885828494075789338754454248931750698880,
    ]

# NOTE: This is very slow (> 2 minutes on 3.4 GHz) without GMPY
@slow
def test_benchmark_czichowski_buchberger():
    with config.using(groebner='buchberger'):
        _do_test_benchmark_czichowski()

def test_benchmark_czichowski_f5b():
    with config.using(groebner='f5b'):
        _do_test_benchmark_czichowski()

def _do_test_benchmark_cyclic_4():
    R, a,b,c,d = ring("a,b,c,d", ZZ, lex)

    I = [a + b + c + d,
         a*b + a*d + b*c + b*d,
         a*b*c + a*b*d + a*c*d + b*c*d,
         a*b*c*d - 1]

    assert groebner(I, R) == [
        4*a + 3*d**9 - 4*d**5 - 3*d,
        4*b + 4*c - 3*d**9 + 4*d**5 + 7*d,
        4*c**2 + 3*d**10 - 4*d**6 - 3*d**2,
        4*c*d**4 + 4*c - d**9 + 4*d**5 + 5*d, d**12 - d**8 - d**4 + 1
    ]

    R, a,b,c,d = ring("a,b,c,d", ZZ, grlex)
    I = [ i.set_ring(R) for i in I ]

    assert groebner(I, R) == [
        3*b*c - c**2 + d**6 - 3*d**2,
        -b + 3*c**2*d**3 - c - d**5 - 4*d,
        -b + 3*c*d**4 + 2*c + 2*d**5 + 2*d,
        c**4 + 2*c**2*d**2 - d**4 - 2,
        c**3*d + c*d**3 + d**4 + 1,
        b*c**2 - c**3 - c**2*d - 2*c*d**2 - d**3,
        b**2 - c**2, b*d + c**2 + c*d + d**2,
        a + b + c + d
    ]

def test_benchmark_cyclic_4_buchberger():
    with config.using(groebner='buchberger'):
        _do_test_benchmark_cyclic_4()

def test_benchmark_cyclic_4_f5b():
    with config.using(groebner='f5b'):
        _do_test_benchmark_cyclic_4()

def test_sig_key():
    s1 = sig((0,) * 3, 2)
    s2 = sig((1,) * 3, 4)
    s3 = sig((2,) * 3, 2)

    assert sig_key(s1, lex) > sig_key(s2, lex)
    assert sig_key(s2, lex) < sig_key(s3, lex)


def test_lbp_key():
    R, x,y,z,t = ring("x,y,z,t", ZZ, lex)

    p1 = lbp(sig((0,) * 4, 3), R.zero, 12)
    p2 = lbp(sig((0,) * 4, 4), R.zero, 13)
    p3 = lbp(sig((0,) * 4, 4), R.zero, 12)

    assert lbp_key(p1) > lbp_key(p2)
    assert lbp_key(p2) < lbp_key(p3)


def test_critical_pair():
    # from cyclic4 with grlex
    R, x,y,z,t = ring("x,y,z,t", QQ, grlex)

    p1 = (((0, 0, 0, 0), 4), y*z*t**2 + z**2*t**2 - t**4 - 1, 4)
    q1 = (((0, 0, 0, 0), 2), -y**2 - y*t - z*t - t**2, 2)

    p2 = (((0, 0, 0, 2), 3), z**3*t**2 + z**2*t**3 - z - t, 5)
    q2 = (((0, 0, 2, 2), 2), y*z + z*t**5 + z*t + t**6, 13)

    assert critical_pair(p1, q1, R) == (
        ((0, 0, 1, 2), 2), ((0, 0, 1, 2), QQ(-1, 1)), (((0, 0, 0, 0), 2), -y**2 - y*t - z*t - t**2, 2),
        ((0, 1, 0, 0), 4), ((0, 1, 0, 0), QQ(1, 1)), (((0, 0, 0, 0), 4), y*z*t**2 + z**2*t**2 - t**4 - 1, 4)
    )
    assert critical_pair(p2, q2, R) == (
        ((0, 0, 4, 2), 2), ((0, 0, 2, 0), QQ(1, 1)), (((0, 0, 2, 2), 2), y*z + z*t**5 + z*t + t**6, 13),
        ((0, 0, 0, 5), 3), ((0, 0, 0, 3), QQ(1, 1)), (((0, 0, 0, 2), 3), z**3*t**2 + z**2*t**3 - z - t, 5)
    )

def test_cp_key():
    # from cyclic4 with grlex
    R, x,y,z,t = ring("x,y,z,t", QQ, grlex)

    p1 = (((0, 0, 0, 0), 4), y*z*t**2 + z**2*t**2 - t**4 - 1, 4)
    q1 = (((0, 0, 0, 0), 2), -y**2 - y*t - z*t - t**2, 2)

    p2 = (((0, 0, 0, 2), 3), z**3*t**2 + z**2*t**3 - z - t, 5)
    q2 = (((0, 0, 2, 2), 2), y*z + z*t**5 + z*t + t**6, 13)

    cp1 = critical_pair(p1, q1, R)
    cp2 = critical_pair(p2, q2, R)

    assert cp_key(cp1, R) < cp_key(cp2, R)

    cp1 = critical_pair(p1, p2, R)
    cp2 = critical_pair(q1, q2, R)

    assert cp_key(cp1, R) < cp_key(cp2, R)


def test_is_rewritable_or_comparable():
    # from katsura4 with grlex
    R, x,y,z,t = ring("x,y,z,t", QQ, grlex)

    p = lbp(sig((0, 0, 2, 1), 2), R.zero, 2)
    B = [lbp(sig((0, 0, 0, 1), 2), QQ(2,45)*y**2 + QQ(1,5)*y*z + QQ(5,63)*y*t + z**2*t + QQ(4,45)*z**2 + QQ(76,35)*z*t**2 - QQ(32,105)*z*t + QQ(13,7)*t**3 - QQ(13,21)*t**2, 6)]

    # rewritable:
    assert is_rewritable_or_comparable(Sign(p), Num(p), B) is True

    p = lbp(sig((0, 1, 1, 0), 2), R.zero, 7)
    B = [lbp(sig((0, 0, 0, 0), 3), QQ(10,3)*y*z + QQ(4,3)*y*t - QQ(1,3)*y + 4*z**2 + QQ(22,3)*z*t - QQ(4,3)*z + 4*t**2 - QQ(4,3)*t, 3)]

    # comparable:
    assert is_rewritable_or_comparable(Sign(p), Num(p), B) is True


def test_f5_reduce():
    # katsura3 with lex
    R, x,y,z = ring("x,y,z", QQ, lex)

    F = [(((0, 0, 0), 1), x + 2*y + 2*z - 1, 1),
         (((0, 0, 0), 2), 6*y**2 + 8*y*z - 2*y + 6*z**2 - 2*z, 2),
         (((0, 0, 0), 3), QQ(10,3)*y*z - QQ(1,3)*y + 4*z**2 - QQ(4,3)*z, 3),
         (((0, 0, 1), 2), y + 30*z**3 - QQ(79,7)*z**2 + QQ(3,7)*z, 4),
         (((0, 0, 2), 2), z**4 - QQ(10,21)*z**3 + QQ(1,84)*z**2 + QQ(1,84)*z, 5)]

    cp = critical_pair(F[0], F[1], R)
    s = s_poly(cp)

    assert f5_reduce(s, F) == (((0, 2, 0), 1), R.zero, 1)

    s = lbp(sig(Sign(s)[0], 100), Polyn(s), Num(s))
    assert f5_reduce(s, F) == s


def test_representing_matrices():
    R, x,y = ring("x,y", QQ, grlex)

    basis = [(0, 0), (0, 1), (1, 0), (1, 1)]
    F = [x**2 - x - 3*y + 1, -2*x + y**2 + y - 1]

    assert _representing_matrices(basis, F, R) == [
        [[QQ(0, 1), QQ(0, 1),-QQ(1, 1), QQ(3, 1)],
         [QQ(0, 1), QQ(0, 1), QQ(3, 1),-QQ(4, 1)],
         [QQ(1, 1), QQ(0, 1), QQ(1, 1), QQ(6, 1)],
         [QQ(0, 1), QQ(1, 1), QQ(0, 1), QQ(1, 1)]],
        [[QQ(0, 1), QQ(1, 1), QQ(0, 1),-QQ(2, 1)],
         [QQ(1, 1),-QQ(1, 1), QQ(0, 1), QQ(6, 1)],
         [QQ(0, 1), QQ(2, 1), QQ(0, 1), QQ(3, 1)],
         [QQ(0, 1), QQ(0, 1), QQ(1, 1),-QQ(1, 1)]]]

def test_groebner_lcm():
    R, x,y,z = ring("x,y,z", ZZ)

    assert groebner_lcm(x**2 - y**2, x - y) == x**2 - y**2
    assert groebner_lcm(2*x**2 - 2*y**2, 2*x - 2*y) == 2*x**2 - 2*y**2

    R, x,y,z = ring("x,y,z", QQ)

    assert groebner_lcm(x**2 - y**2, x - y) == x**2 - y**2
    assert groebner_lcm(2*x**2 - 2*y**2, 2*x - 2*y) == 2*x**2 - 2*y**2

    R, x,y = ring("x,y", ZZ)

    assert groebner_lcm(x**2*y, x*y**2) == x**2*y**2

    f = 2*x*y**5 - 3*x*y**4 - 2*x*y**3 + 3*x*y**2
    g = y**5 - 2*y**3 + y
    h = 2*x*y**7 - 3*x*y**6 - 4*x*y**5 + 6*x*y**4 + 2*x*y**3 - 3*x*y**2

    assert groebner_lcm(f, g) == h

    f = x**3 - 3*x**2*y - 9*x*y**2 - 5*y**3
    g = x**4 + 6*x**3*y + 12*x**2*y**2 + 10*x*y**3 + 3*y**4
    h = x**5 + x**4*y - 18*x**3*y**2 - 50*x**2*y**3 - 47*x*y**4 - 15*y**5

    assert groebner_lcm(f, g) == h

def test_groebner_gcd():
    R, x,y,z = ring("x,y,z", ZZ)

    assert groebner_gcd(x**2 - y**2, x - y) == x - y
    assert groebner_gcd(2*x**2 - 2*y**2, 2*x - 2*y) == 2*x - 2*y

    R, x,y,z = ring("x,y,z", QQ)

    assert groebner_gcd(x**2 - y**2, x - y) == x - y
    assert groebner_gcd(2*x**2 - 2*y**2, 2*x - 2*y) == x - y

def test_is_groebner():
    R, x,y = ring("x,y", QQ, grlex)
    valid_groebner = [x**2, x*y, -QQ(1,2)*x + y**2]
    invalid_groebner = [x**3, x*y, -QQ(1,2)*x + y**2]
    assert is_groebner(valid_groebner, R) is True
    assert is_groebner(invalid_groebner, R) is False

def test_is_reduced():
    R, x, y = ring("x,y", QQ, lex)
    f = x**2 + 2*x*y**2
    g = x*y + 2*y**3 - 1
    assert is_reduced([f, g], R) ==  False
    G = groebner([f, g], R)
    assert is_reduced(G, R) == True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_numeric_only.py ===
import re

import numpy as np
import pytest

from pandas._libs import lib

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args


class TestNumericOnly:
    # make sure that we are passing thru kwargs to our agg functions

    @pytest.fixture
    def df(self):
        # GH3668
        # GH5724
        df = DataFrame(
            {
                "group": [1, 1, 2],
                "int": [1, 2, 3],
                "float": [4.0, 5.0, 6.0],
                "string": Series(["a", "b", "c"], dtype="str"),
                "object": Series(["a", "b", "c"], dtype=object),
                "category_string": Series(list("abc")).astype("category"),
                "category_int": [7, 8, 9],
                "datetime": date_range("20130101", periods=3),
                "datetimetz": date_range("20130101", periods=3, tz="US/Eastern"),
                "timedelta": pd.timedelta_range("1 s", periods=3, freq="s"),
            },
            columns=[
                "group",
                "int",
                "float",
                "string",
                "object",
                "category_string",
                "category_int",
                "datetime",
                "datetimetz",
                "timedelta",
            ],
        )
        return df

    @pytest.mark.parametrize("method", ["mean", "median"])
    def test_averages(self, df, method):
        # mean / median
        expected_columns_numeric = Index(["int", "float", "category_int"])

        gb = df.groupby("group")
        expected = DataFrame(
            {
                "category_int": [7.5, 9],
                "float": [4.5, 6.0],
                "timedelta": [pd.Timedelta("1.5s"), pd.Timedelta("3s")],
                "int": [1.5, 3],
                "datetime": [
                    Timestamp("2013-01-01 12:00:00"),
                    Timestamp("2013-01-03 00:00:00"),
                ],
                "datetimetz": [
                    Timestamp("2013-01-01 12:00:00", tz="US/Eastern"),
                    Timestamp("2013-01-03 00:00:00", tz="US/Eastern"),
                ],
            },
            index=Index([1, 2], name="group"),
            columns=[
                "int",
                "float",
                "category_int",
            ],
        )

        result = getattr(gb, method)(numeric_only=True)
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        expected_columns = expected.columns

        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize("method", ["min", "max"])
    def test_extrema(self, df, method):
        # TODO: min, max *should* handle
        # categorical (ordered) dtype

        expected_columns = Index(
            [
                "int",
                "float",
                "string",
                "category_int",
                "datetime",
                "datetimetz",
                "timedelta",
            ]
        )
        expected_columns_numeric = expected_columns

        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize("method", ["first", "last"])
    def test_first_last(self, df, method):
        expected_columns = Index(
            [
                "int",
                "float",
                "string",
                "object",
                "category_string",
                "category_int",
                "datetime",
                "datetimetz",
                "timedelta",
            ]
        )
        expected_columns_numeric = expected_columns

        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize("method", ["sum", "cumsum"])
    def test_sum_cumsum(self, df, method):
        expected_columns_numeric = Index(["int", "float", "category_int"])
        expected_columns = Index(
            ["int", "float", "string", "category_int", "timedelta"]
        )
        if method == "cumsum":
            # cumsum loses string
            expected_columns = Index(["int", "float", "category_int", "timedelta"])

        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize("method", ["prod", "cumprod"])
    def test_prod_cumprod(self, df, method):
        expected_columns = Index(["int", "float", "category_int"])
        expected_columns_numeric = expected_columns

        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize("method", ["cummin", "cummax"])
    def test_cummin_cummax(self, df, method):
        # like min, max, but don't include strings
        expected_columns = Index(
            ["int", "float", "category_int", "datetime", "datetimetz", "timedelta"]
        )

        # GH#15561: numeric_only=False set by default like min/max
        expected_columns_numeric = expected_columns

        self._check(df, method, expected_columns, expected_columns_numeric)

    def _check(self, df, method, expected_columns, expected_columns_numeric):
        gb = df.groupby("group")

        # object dtypes for transformations are not implemented in Cython and
        # have no Python fallback
        exception = (
            (NotImplementedError, TypeError) if method.startswith("cum") else TypeError
        )

        if method in ("min", "max", "cummin", "cummax", "cumsum", "cumprod"):
            # The methods default to numeric_only=False and raise TypeError
            msg = "|".join(
                [
                    "Categorical is not ordered",
                    f"Cannot perform {method} with non-ordered Categorical",
                    re.escape(f"agg function failed [how->{method},dtype->object]"),
                    # cumsum/cummin/cummax/cumprod
                    "function is not implemented for this dtype",
                    f"dtype 'str' does not support operation '{method}'",
                ]
            )
            with pytest.raises(exception, match=msg):
                getattr(gb, method)()
        elif method in ("sum", "mean", "median", "prod"):
            msg = "|".join(
                [
                    "category type does not support sum operations",
                    re.escape(f"agg function failed [how->{method},dtype->object]"),
                    re.escape(f"agg function failed [how->{method},dtype->string]"),
                    f"dtype 'str' does not support operation '{method}'",
                ]
            )
            with pytest.raises(exception, match=msg):
                getattr(gb, method)()
        else:
            result = getattr(gb, method)()
            tm.assert_index_equal(result.columns, expected_columns_numeric)

        if method not in ("first", "last"):
            msg = "|".join(
                [
                    "Categorical is not ordered",
                    "category type does not support",
                    "function is not implemented for this dtype",
                    f"Cannot perform {method} with non-ordered Categorical",
                    re.escape(f"agg function failed [how->{method},dtype->object]"),
                    re.escape(f"agg function failed [how->{method},dtype->string]"),
                    f"dtype 'str' does not support operation '{method}'",
                ]
            )
            with pytest.raises(exception, match=msg):
                getattr(gb, method)(numeric_only=False)
        else:
            result = getattr(gb, method)(numeric_only=False)
            tm.assert_index_equal(result.columns, expected_columns)


@pytest.mark.parametrize("numeric_only", [True, False, None])
def test_axis1_numeric_only(request, groupby_func, numeric_only, using_infer_string):
    if groupby_func in ("idxmax", "idxmin"):
        pytest.skip("idxmax and idx_min tested in test_idxmin_idxmax_axis1")
    if groupby_func in ("corrwith", "skew"):
        msg = "GH#47723 groupby.corrwith and skew do not correctly implement axis=1"
        request.applymarker(pytest.mark.xfail(reason=msg))

    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)), columns=["A", "B", "C", "D"]
    )
    df["E"] = "x"
    groups = [1, 2, 3, 1, 2, 3, 1, 2, 3, 4]
    gb = df.groupby(groups)
    method = getattr(gb, groupby_func)
    args = get_groupby_method_args(groupby_func, df)
    kwargs = {"axis": 1}
    if numeric_only is not None:
        # when numeric_only is None we don't pass any argument
        kwargs["numeric_only"] = numeric_only

    # Functions without numeric_only and axis args
    no_args = ("cumprod", "cumsum", "diff", "fillna", "pct_change", "rank", "shift")
    # Functions with axis args
    has_axis = (
        "cumprod",
        "cumsum",
        "diff",
        "pct_change",
        "rank",
        "shift",
        "cummax",
        "cummin",
        "idxmin",
        "idxmax",
        "fillna",
    )
    warn_msg = f"DataFrameGroupBy.{groupby_func} with axis=1 is deprecated"
    if numeric_only is not None and groupby_func in no_args:
        msg = "got an unexpected keyword argument 'numeric_only'"
        if groupby_func in ["cumprod", "cumsum"]:
            with pytest.raises(TypeError, match=msg):
                with tm.assert_produces_warning(FutureWarning, match=warn_msg):
                    method(*args, **kwargs)
        else:
            with pytest.raises(TypeError, match=msg):
                method(*args, **kwargs)
    elif groupby_func not in has_axis:
        msg = "got an unexpected keyword argument 'axis'"
        with pytest.raises(TypeError, match=msg):
            method(*args, **kwargs)
    # fillna and shift are successful even on object dtypes
    elif (numeric_only is None or not numeric_only) and groupby_func not in (
        "fillna",
        "shift",
    ):
        msgs = (
            # cummax, cummin, rank
            "not supported between instances of",
            # cumprod
            "can't multiply sequence by non-int of type 'float'",
            # cumsum, diff, pct_change
            "unsupported operand type",
            "has no kernel",
            "operation 'sub' not supported for dtype 'str' with dtype 'float64'",
        )
        if using_infer_string:
            pa = pytest.importorskip("pyarrow")

            errs = (TypeError, pa.lib.ArrowNotImplementedError)
        else:
            errs = TypeError
        with pytest.raises(errs, match=f"({'|'.join(msgs)})"):
            with tm.assert_produces_warning(FutureWarning, match=warn_msg):
                method(*args, **kwargs)
    else:
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            result = method(*args, **kwargs)

        df_expected = df.drop(columns="E").T if numeric_only else df.T
        expected = getattr(df_expected, groupby_func)(*args).T
        if groupby_func == "shift" and not numeric_only:
            # shift with axis=1 leaves the leftmost column as numeric
            # but transposing for expected gives us object dtype
            expected = expected.astype(float)

        tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "kernel, has_arg",
    [
        ("all", False),
        ("any", False),
        ("bfill", False),
        ("corr", True),
        ("corrwith", True),
        ("cov", True),
        ("cummax", True),
        ("cummin", True),
        ("cumprod", True),
        ("cumsum", True),
        ("diff", False),
        ("ffill", False),
        ("fillna", False),
        ("first", True),
        ("idxmax", True),
        ("idxmin", True),
        ("last", True),
        ("max", True),
        ("mean", True),
        ("median", True),
        ("min", True),
        ("nth", False),
        ("nunique", False),
        ("pct_change", False),
        ("prod", True),
        ("quantile", True),
        ("sem", True),
        ("skew", True),
        ("std", True),
        ("sum", True),
        ("var", True),
    ],
)
@pytest.mark.parametrize("numeric_only", [True, False, lib.no_default])
@pytest.mark.parametrize("keys", [["a1"], ["a1", "a2"]])
def test_numeric_only(kernel, has_arg, numeric_only, keys):
    # GH#46072
    # drops_nuisance: Whether the op drops nuisance columns even when numeric_only=False
    # has_arg: Whether the op has a numeric_only arg
    df = DataFrame({"a1": [1, 1], "a2": [2, 2], "a3": [5, 6], "b": 2 * [object]})

    args = get_groupby_method_args(kernel, df)
    kwargs = {} if numeric_only is lib.no_default else {"numeric_only": numeric_only}

    gb = df.groupby(keys)
    method = getattr(gb, kernel)
    if has_arg and numeric_only is True:
        # Cases where b does not appear in the result
        result = method(*args, **kwargs)
        assert "b" not in result.columns
    elif (
        # kernels that work on any dtype and have numeric_only arg
        kernel in ("first", "last")
        or (
            # kernels that work on any dtype and don't have numeric_only arg
            kernel in ("any", "all", "bfill", "ffill", "fillna", "nth", "nunique")
            and numeric_only is lib.no_default
        )
    ):
        warn = FutureWarning if kernel == "fillna" else None
        msg = "DataFrameGroupBy.fillna is deprecated"
        with tm.assert_produces_warning(warn, match=msg):
            result = method(*args, **kwargs)
        assert "b" in result.columns
    elif has_arg:
        assert numeric_only is not True
        # kernels that are successful on any dtype were above; this will fail

        # object dtypes for transformations are not implemented in Cython and
        # have no Python fallback
        exception = NotImplementedError if kernel.startswith("cum") else TypeError

        msg = "|".join(
            [
                "not allowed for this dtype",
                "cannot be performed against 'object' dtypes",
                # On PY39 message is "a number"; on PY310 and after is "a real number"
                "must be a string or a.* number",
                "unsupported operand type",
                "function is not implemented for this dtype",
                re.escape(f"agg function failed [how->{kernel},dtype->object]"),
            ]
        )
        if kernel == "quantile":
            msg = "dtype 'object' does not support operation 'quantile'"
        elif kernel == "idxmin":
            msg = "'<' not supported between instances of 'type' and 'type'"
        elif kernel == "idxmax":
            msg = "'>' not supported between instances of 'type' and 'type'"
        with pytest.raises(exception, match=msg):
            method(*args, **kwargs)
    elif not has_arg and numeric_only is not lib.no_default:
        with pytest.raises(
            TypeError, match="got an unexpected keyword argument 'numeric_only'"
        ):
            method(*args, **kwargs)
    else:
        assert kernel in ("diff", "pct_change")
        assert numeric_only is lib.no_default
        # Doesn't have numeric_only argument and fails on nuisance columns
        with pytest.raises(TypeError, match=r"unsupported operand type"):
            method(*args, **kwargs)


@pytest.mark.filterwarnings("ignore:Downcasting object dtype arrays:FutureWarning")
@pytest.mark.parametrize("dtype", [bool, int, float, object])
def test_deprecate_numeric_only_series(dtype, groupby_func, request):
    # GH#46560
    grouper = [0, 0, 1]

    ser = Series([1, 0, 0], dtype=dtype)
    gb = ser.groupby(grouper)

    if groupby_func == "corrwith":
        # corrwith is not implemented on SeriesGroupBy
        assert not hasattr(gb, groupby_func)
        return

    method = getattr(gb, groupby_func)

    expected_ser = Series([1, 0, 0])
    expected_gb = expected_ser.groupby(grouper)
    expected_method = getattr(expected_gb, groupby_func)

    args = get_groupby_method_args(groupby_func, ser)

    fails_on_numeric_object = (
        "corr",
        "cov",
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
        "quantile",
    )
    # ops that give an object result on object input
    obj_result = (
        "first",
        "last",
        "nth",
        "bfill",
        "ffill",
        "shift",
        "sum",
        "diff",
        "pct_change",
        "var",
        "mean",
        "median",
        "min",
        "max",
        "prod",
        "skew",
    )

    # Test default behavior; kernels that fail may be enabled in the future but kernels
    # that succeed should not be allowed to fail (without deprecation, at least)
    if groupby_func in fails_on_numeric_object and dtype is object:
        if groupby_func == "quantile":
            msg = "dtype 'object' does not support operation 'quantile'"
        else:
            msg = "is not supported for object dtype"
        warn = FutureWarning if groupby_func == "fillna" else None
        warn_msg = "DataFrameGroupBy.fillna is deprecated"
        with tm.assert_produces_warning(warn, match=warn_msg):
            with pytest.raises(TypeError, match=msg):
                method(*args)
    elif dtype is object:
        warn = FutureWarning if groupby_func == "fillna" else None
        warn_msg = "SeriesGroupBy.fillna is deprecated"
        with tm.assert_produces_warning(warn, match=warn_msg):
            result = method(*args)
        with tm.assert_produces_warning(warn, match=warn_msg):
            expected = expected_method(*args)
        if groupby_func in obj_result:
            expected = expected.astype(object)
        tm.assert_series_equal(result, expected)

    has_numeric_only = (
        "first",
        "last",
        "max",
        "mean",
        "median",
        "min",
        "prod",
        "quantile",
        "sem",
        "skew",
        "std",
        "sum",
        "var",
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
    )
    if groupby_func not in has_numeric_only:
        msg = "got an unexpected keyword argument 'numeric_only'"
        with pytest.raises(TypeError, match=msg):
            method(*args, numeric_only=True)
    elif dtype is object:
        msg = "|".join(
            [
                "SeriesGroupBy.sem called with numeric_only=True and dtype object",
                "Series.skew does not allow numeric_only=True with non-numeric",
                "cum(sum|prod|min|max) is not supported for object dtype",
                r"Cannot use numeric_only=True with SeriesGroupBy\..* and non-numeric",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            method(*args, numeric_only=True)
    elif dtype == bool and groupby_func == "quantile":
        msg = "Allowing bool dtype in SeriesGroupBy.quantile"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # GH#51424
            result = method(*args, numeric_only=True)
            expected = method(*args, numeric_only=False)
        tm.assert_series_equal(result, expected)
    else:
        result = method(*args, numeric_only=True)
        expected = method(*args, numeric_only=False)
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\test_rolling_functions.py ===
from datetime import datetime

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    concat,
    isna,
    notna,
)
import pandas._testing as tm

from pandas.tseries import offsets


@pytest.mark.parametrize(
    "compare_func, roll_func, kwargs",
    [
        [np.mean, "mean", {}],
        [np.nansum, "sum", {}],
        [
            lambda x: np.isfinite(x).astype(float).sum(),
            "count",
            {},
        ],
        [np.median, "median", {}],
        [np.min, "min", {}],
        [np.max, "max", {}],
        [lambda x: np.std(x, ddof=1), "std", {}],
        [lambda x: np.std(x, ddof=0), "std", {"ddof": 0}],
        [lambda x: np.var(x, ddof=1), "var", {}],
        [lambda x: np.var(x, ddof=0), "var", {"ddof": 0}],
    ],
)
def test_series(series, compare_func, roll_func, kwargs, step):
    result = getattr(series.rolling(50, step=step), roll_func)(**kwargs)
    assert isinstance(result, Series)
    end = range(0, len(series), step or 1)[-1] + 1
    tm.assert_almost_equal(result.iloc[-1], compare_func(series[end - 50 : end]))


@pytest.mark.parametrize(
    "compare_func, roll_func, kwargs",
    [
        [np.mean, "mean", {}],
        [np.nansum, "sum", {}],
        [
            lambda x: np.isfinite(x).astype(float).sum(),
            "count",
            {},
        ],
        [np.median, "median", {}],
        [np.min, "min", {}],
        [np.max, "max", {}],
        [lambda x: np.std(x, ddof=1), "std", {}],
        [lambda x: np.std(x, ddof=0), "std", {"ddof": 0}],
        [lambda x: np.var(x, ddof=1), "var", {}],
        [lambda x: np.var(x, ddof=0), "var", {"ddof": 0}],
    ],
)
def test_frame(raw, frame, compare_func, roll_func, kwargs, step):
    result = getattr(frame.rolling(50, step=step), roll_func)(**kwargs)
    assert isinstance(result, DataFrame)
    end = range(0, len(frame), step or 1)[-1] + 1
    tm.assert_series_equal(
        result.iloc[-1, :],
        frame.iloc[end - 50 : end, :].apply(compare_func, axis=0, raw=raw),
        check_names=False,
    )


@pytest.mark.parametrize(
    "compare_func, roll_func, kwargs, minp",
    [
        [np.mean, "mean", {}, 10],
        [np.nansum, "sum", {}, 10],
        [lambda x: np.isfinite(x).astype(float).sum(), "count", {}, 0],
        [np.median, "median", {}, 10],
        [np.min, "min", {}, 10],
        [np.max, "max", {}, 10],
        [lambda x: np.std(x, ddof=1), "std", {}, 10],
        [lambda x: np.std(x, ddof=0), "std", {"ddof": 0}, 10],
        [lambda x: np.var(x, ddof=1), "var", {}, 10],
        [lambda x: np.var(x, ddof=0), "var", {"ddof": 0}, 10],
    ],
)
def test_time_rule_series(series, compare_func, roll_func, kwargs, minp):
    win = 25
    ser = series[::2].resample("B").mean()
    series_result = getattr(ser.rolling(window=win, min_periods=minp), roll_func)(
        **kwargs
    )
    last_date = series_result.index[-1]
    prev_date = last_date - 24 * offsets.BDay()

    trunc_series = series[::2].truncate(prev_date, last_date)
    tm.assert_almost_equal(series_result.iloc[-1], compare_func(trunc_series))


@pytest.mark.parametrize(
    "compare_func, roll_func, kwargs, minp",
    [
        [np.mean, "mean", {}, 10],
        [np.nansum, "sum", {}, 10],
        [lambda x: np.isfinite(x).astype(float).sum(), "count", {}, 0],
        [np.median, "median", {}, 10],
        [np.min, "min", {}, 10],
        [np.max, "max", {}, 10],
        [lambda x: np.std(x, ddof=1), "std", {}, 10],
        [lambda x: np.std(x, ddof=0), "std", {"ddof": 0}, 10],
        [lambda x: np.var(x, ddof=1), "var", {}, 10],
        [lambda x: np.var(x, ddof=0), "var", {"ddof": 0}, 10],
    ],
)
def test_time_rule_frame(raw, frame, compare_func, roll_func, kwargs, minp):
    win = 25
    frm = frame[::2].resample("B").mean()
    frame_result = getattr(frm.rolling(window=win, min_periods=minp), roll_func)(
        **kwargs
    )
    last_date = frame_result.index[-1]
    prev_date = last_date - 24 * offsets.BDay()

    trunc_frame = frame[::2].truncate(prev_date, last_date)
    tm.assert_series_equal(
        frame_result.xs(last_date),
        trunc_frame.apply(compare_func, raw=raw),
        check_names=False,
    )


@pytest.mark.parametrize(
    "compare_func, roll_func, kwargs",
    [
        [np.mean, "mean", {}],
        [np.nansum, "sum", {}],
        [np.median, "median", {}],
        [np.min, "min", {}],
        [np.max, "max", {}],
        [lambda x: np.std(x, ddof=1), "std", {}],
        [lambda x: np.std(x, ddof=0), "std", {"ddof": 0}],
        [lambda x: np.var(x, ddof=1), "var", {}],
        [lambda x: np.var(x, ddof=0), "var", {"ddof": 0}],
    ],
)
def test_nans(compare_func, roll_func, kwargs):
    obj = Series(np.random.default_rng(2).standard_normal(50))
    obj[:10] = np.nan
    obj[-10:] = np.nan

    result = getattr(obj.rolling(50, min_periods=30), roll_func)(**kwargs)
    tm.assert_almost_equal(result.iloc[-1], compare_func(obj[10:-10]))

    # min_periods is working correctly
    result = getattr(obj.rolling(20, min_periods=15), roll_func)(**kwargs)
    assert isna(result.iloc[23])
    assert not isna(result.iloc[24])

    assert not isna(result.iloc[-6])
    assert isna(result.iloc[-5])

    obj2 = Series(np.random.default_rng(2).standard_normal(20))
    result = getattr(obj2.rolling(10, min_periods=5), roll_func)(**kwargs)
    assert isna(result.iloc[3])
    assert notna(result.iloc[4])

    if roll_func != "sum":
        result0 = getattr(obj.rolling(20, min_periods=0), roll_func)(**kwargs)
        result1 = getattr(obj.rolling(20, min_periods=1), roll_func)(**kwargs)
        tm.assert_almost_equal(result0, result1)


def test_nans_count():
    obj = Series(np.random.default_rng(2).standard_normal(50))
    obj[:10] = np.nan
    obj[-10:] = np.nan
    result = obj.rolling(50, min_periods=30).count()
    tm.assert_almost_equal(
        result.iloc[-1], np.isfinite(obj[10:-10]).astype(float).sum()
    )


@pytest.mark.parametrize(
    "roll_func, kwargs",
    [
        ["mean", {}],
        ["sum", {}],
        ["median", {}],
        ["min", {}],
        ["max", {}],
        ["std", {}],
        ["std", {"ddof": 0}],
        ["var", {}],
        ["var", {"ddof": 0}],
    ],
)
@pytest.mark.parametrize("minp", [0, 99, 100])
def test_min_periods(series, minp, roll_func, kwargs, step):
    result = getattr(
        series.rolling(len(series) + 1, min_periods=minp, step=step), roll_func
    )(**kwargs)
    expected = getattr(
        series.rolling(len(series), min_periods=minp, step=step), roll_func
    )(**kwargs)
    nan_mask = isna(result)
    tm.assert_series_equal(nan_mask, isna(expected))

    nan_mask = ~nan_mask
    tm.assert_almost_equal(result[nan_mask], expected[nan_mask])


def test_min_periods_count(series, step):
    result = series.rolling(len(series) + 1, min_periods=0, step=step).count()
    expected = series.rolling(len(series), min_periods=0, step=step).count()
    nan_mask = isna(result)
    tm.assert_series_equal(nan_mask, isna(expected))

    nan_mask = ~nan_mask
    tm.assert_almost_equal(result[nan_mask], expected[nan_mask])


@pytest.mark.parametrize(
    "roll_func, kwargs, minp",
    [
        ["mean", {}, 15],
        ["sum", {}, 15],
        ["count", {}, 0],
        ["median", {}, 15],
        ["min", {}, 15],
        ["max", {}, 15],
        ["std", {}, 15],
        ["std", {"ddof": 0}, 15],
        ["var", {}, 15],
        ["var", {"ddof": 0}, 15],
    ],
)
def test_center(roll_func, kwargs, minp):
    obj = Series(np.random.default_rng(2).standard_normal(50))
    obj[:10] = np.nan
    obj[-10:] = np.nan

    result = getattr(obj.rolling(20, min_periods=minp, center=True), roll_func)(
        **kwargs
    )
    expected = (
        getattr(
            concat([obj, Series([np.nan] * 9)]).rolling(20, min_periods=minp), roll_func
        )(**kwargs)
        .iloc[9:]
        .reset_index(drop=True)
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "roll_func, kwargs, minp, fill_value",
    [
        ["mean", {}, 10, None],
        ["sum", {}, 10, None],
        ["count", {}, 0, 0],
        ["median", {}, 10, None],
        ["min", {}, 10, None],
        ["max", {}, 10, None],
        ["std", {}, 10, None],
        ["std", {"ddof": 0}, 10, None],
        ["var", {}, 10, None],
        ["var", {"ddof": 0}, 10, None],
    ],
)
def test_center_reindex_series(series, roll_func, kwargs, minp, fill_value):
    # shifter index
    s = [f"x{x:d}" for x in range(12)]

    series_xp = (
        getattr(
            series.reindex(list(series.index) + s).rolling(window=25, min_periods=minp),
            roll_func,
        )(**kwargs)
        .shift(-12)
        .reindex(series.index)
    )
    series_rs = getattr(
        series.rolling(window=25, min_periods=minp, center=True), roll_func
    )(**kwargs)
    if fill_value is not None:
        series_xp = series_xp.fillna(fill_value)
    tm.assert_series_equal(series_xp, series_rs)


@pytest.mark.parametrize(
    "roll_func, kwargs, minp, fill_value",
    [
        ["mean", {}, 10, None],
        ["sum", {}, 10, None],
        ["count", {}, 0, 0],
        ["median", {}, 10, None],
        ["min", {}, 10, None],
        ["max", {}, 10, None],
        ["std", {}, 10, None],
        ["std", {"ddof": 0}, 10, None],
        ["var", {}, 10, None],
        ["var", {"ddof": 0}, 10, None],
    ],
)
def test_center_reindex_frame(frame, roll_func, kwargs, minp, fill_value):
    # shifter index
    s = [f"x{x:d}" for x in range(12)]

    frame_xp = (
        getattr(
            frame.reindex(list(frame.index) + s).rolling(window=25, min_periods=minp),
            roll_func,
        )(**kwargs)
        .shift(-12)
        .reindex(frame.index)
    )
    frame_rs = getattr(
        frame.rolling(window=25, min_periods=minp, center=True), roll_func
    )(**kwargs)
    if fill_value is not None:
        frame_xp = frame_xp.fillna(fill_value)
    tm.assert_frame_equal(frame_xp, frame_rs)


@pytest.mark.parametrize(
    "f",
    [
        lambda x: x.rolling(window=10, min_periods=5).cov(x, pairwise=False),
        lambda x: x.rolling(window=10, min_periods=5).corr(x, pairwise=False),
        lambda x: x.rolling(window=10, min_periods=5).max(),
        lambda x: x.rolling(window=10, min_periods=5).min(),
        lambda x: x.rolling(window=10, min_periods=5).sum(),
        lambda x: x.rolling(window=10, min_periods=5).mean(),
        lambda x: x.rolling(window=10, min_periods=5).std(),
        lambda x: x.rolling(window=10, min_periods=5).var(),
        lambda x: x.rolling(window=10, min_periods=5).skew(),
        lambda x: x.rolling(window=10, min_periods=5).kurt(),
        lambda x: x.rolling(window=10, min_periods=5).quantile(q=0.5),
        lambda x: x.rolling(window=10, min_periods=5).median(),
        lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=False),
        lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=True),
        pytest.param(
            lambda x: x.rolling(win_type="boxcar", window=10, min_periods=5).mean(),
            marks=td.skip_if_no("scipy"),
        ),
    ],
)
def test_rolling_functions_window_non_shrinkage(f):
    # GH 7764
    s = Series(range(4))
    s_expected = Series(np.nan, index=s.index)
    df = DataFrame([[1, 5], [3, 2], [3, 9], [-1, 0]], columns=["A", "B"])
    df_expected = DataFrame(np.nan, index=df.index, columns=df.columns)

    s_result = f(s)
    tm.assert_series_equal(s_result, s_expected)

    df_result = f(df)
    tm.assert_frame_equal(df_result, df_expected)


def test_rolling_max_gh6297(step):
    """Replicate result expected in GH #6297"""
    indices = [datetime(1975, 1, i) for i in range(1, 6)]
    # So that we can have 2 datapoints on one of the days
    indices.append(datetime(1975, 1, 3, 6, 0))
    series = Series(range(1, 7), index=indices)
    # Use floats instead of ints as values
    series = series.map(lambda x: float(x))
    # Sort chronologically
    series = series.sort_index()

    expected = Series(
        [1.0, 2.0, 6.0, 4.0, 5.0],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )[::step]
    x = series.resample("D").max().rolling(window=1, step=step).max()
    tm.assert_series_equal(expected, x)


def test_rolling_max_resample(step):
    indices = [datetime(1975, 1, i) for i in range(1, 6)]
    # So that we can have 3 datapoints on last day (4, 10, and 20)
    indices.append(datetime(1975, 1, 5, 1))
    indices.append(datetime(1975, 1, 5, 2))
    series = Series(list(range(5)) + [10, 20], index=indices)
    # Use floats instead of ints as values
    series = series.map(lambda x: float(x))
    # Sort chronologically
    series = series.sort_index()

    # Default how should be max
    expected = Series(
        [0.0, 1.0, 2.0, 3.0, 20.0],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )[::step]
    x = series.resample("D").max().rolling(window=1, step=step).max()
    tm.assert_series_equal(expected, x)

    # Now specify median (10.0)
    expected = Series(
        [0.0, 1.0, 2.0, 3.0, 10.0],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )[::step]
    x = series.resample("D").median().rolling(window=1, step=step).max()
    tm.assert_series_equal(expected, x)

    # Now specify mean (4+10+20)/3
    v = (4.0 + 10.0 + 20.0) / 3.0
    expected = Series(
        [0.0, 1.0, 2.0, 3.0, v],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )[::step]
    x = series.resample("D").mean().rolling(window=1, step=step).max()
    tm.assert_series_equal(expected, x)


def test_rolling_min_resample(step):
    indices = [datetime(1975, 1, i) for i in range(1, 6)]
    # So that we can have 3 datapoints on last day (4, 10, and 20)
    indices.append(datetime(1975, 1, 5, 1))
    indices.append(datetime(1975, 1, 5, 2))
    series = Series(list(range(5)) + [10, 20], index=indices)
    # Use floats instead of ints as values
    series = series.map(lambda x: float(x))
    # Sort chronologically
    series = series.sort_index()

    # Default how should be min
    expected = Series(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )[::step]
    r = series.resample("D").min().rolling(window=1, step=step)
    tm.assert_series_equal(expected, r.min())


def test_rolling_median_resample():
    indices = [datetime(1975, 1, i) for i in range(1, 6)]
    # So that we can have 3 datapoints on last day (4, 10, and 20)
    indices.append(datetime(1975, 1, 5, 1))
    indices.append(datetime(1975, 1, 5, 2))
    series = Series(list(range(5)) + [10, 20], index=indices)
    # Use floats instead of ints as values
    series = series.map(lambda x: float(x))
    # Sort chronologically
    series = series.sort_index()

    # Default how should be median
    expected = Series(
        [0.0, 1.0, 2.0, 3.0, 10],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )
    x = series.resample("D").median().rolling(window=1).median()
    tm.assert_series_equal(expected, x)


def test_rolling_median_memory_error():
    # GH11722
    n = 20000
    Series(np.random.default_rng(2).standard_normal(n)).rolling(
        window=2, center=False
    ).median()
    Series(np.random.default_rng(2).standard_normal(n)).rolling(
        window=2, center=False
    ).median()


@pytest.mark.parametrize(
    "data_type",
    [np.dtype(f"f{width}") for width in [4, 8]]
    + [np.dtype(f"{sign}{width}") for width in [1, 2, 4, 8] for sign in "ui"],
)
def test_rolling_min_max_numeric_types(data_type):
    # GH12373

    # Just testing that these don't throw exceptions and that
    # the return type is float64. Other tests will cover quantitative
    # correctness
    result = DataFrame(np.arange(20, dtype=data_type)).rolling(window=5).max()
    assert result.dtypes[0] == np.dtype("f8")
    result = DataFrame(np.arange(20, dtype=data_type)).rolling(window=5).min()
    assert result.dtypes[0] == np.dtype("f8")


@pytest.mark.parametrize(
    "f",
    [
        lambda x: x.rolling(window=10, min_periods=0).count(),
        lambda x: x.rolling(window=10, min_periods=5).cov(x, pairwise=False),
        lambda x: x.rolling(window=10, min_periods=5).corr(x, pairwise=False),
        lambda x: x.rolling(window=10, min_periods=5).max(),
        lambda x: x.rolling(window=10, min_periods=5).min(),
        lambda x: x.rolling(window=10, min_periods=5).sum(),
        lambda x: x.rolling(window=10, min_periods=5).mean(),
        lambda x: x.rolling(window=10, min_periods=5).std(),
        lambda x: x.rolling(window=10, min_periods=5).var(),
        lambda x: x.rolling(window=10, min_periods=5).skew(),
        lambda x: x.rolling(window=10, min_periods=5).kurt(),
        lambda x: x.rolling(window=10, min_periods=5).quantile(0.5),
        lambda x: x.rolling(window=10, min_periods=5).median(),
        lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=False),
        lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=True),
        pytest.param(
            lambda x: x.rolling(win_type="boxcar", window=10, min_periods=5).mean(),
            marks=td.skip_if_no("scipy"),
        ),
    ],
)
def test_moment_functions_zero_length(f):
    # GH 8056
    s = Series(dtype=np.float64)
    s_expected = s
    df1 = DataFrame()
    df1_expected = df1
    df2 = DataFrame(columns=["a"])
    df2["a"] = df2["a"].astype("float64")
    df2_expected = df2

    s_result = f(s)
    tm.assert_series_equal(s_result, s_expected)

    df1_result = f(df1)
    tm.assert_frame_equal(df1_result, df1_expected)

    df2_result = f(df2)
    tm.assert_frame_equal(df2_result, df2_expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_torch.py ===
import random
import math

from sympy import symbols, Derivative
from sympy.printing.pytorch import torch_code
from sympy import (eye, MatrixSymbol, Matrix)
from sympy.tensor.array import NDimArray
from sympy.tensor.array.expressions.array_expressions import (
    ArrayTensorProduct, ArrayAdd,
    PermuteDims, ArrayDiagonal, _CodegenArrayAbstract)
from sympy.utilities.lambdify import lambdify
from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
from sympy.functions import \
    Abs, ceiling, exp, floor, sign, sin, asin, cos, \
    acos, tan, atan, atan2, cosh, acosh, sinh, asinh, tanh, atanh, \
    re, im, arg, erf, loggamma, sqrt
from sympy.testing.pytest import skip
from sympy.external import import_module
from sympy.matrices.expressions import \
    Determinant, HadamardProduct, Inverse, Trace
from sympy.matrices import randMatrix
from sympy.matrices import Identity, ZeroMatrix, OneMatrix
from sympy import conjugate, I
from sympy import Heaviside, gamma, polygamma



torch = import_module("torch")

M = MatrixSymbol("M", 3, 3)
N = MatrixSymbol("N", 3, 3)
P = MatrixSymbol("P", 3, 3)
Q = MatrixSymbol("Q", 3, 3)

x, y, z, t = symbols("x y z t")

if torch is not None:
    llo = [list(range(i, i + 3)) for i in range(0, 9, 3)]
    m3x3 = torch.tensor(llo, dtype=torch.float64)
    m3x3sympy = Matrix(llo)


def _compare_torch_matrix(variables, expr):
    f = lambdify(variables, expr, 'torch')

    random_matrices = [randMatrix(i.shape[0], i.shape[1]) for i in variables]
    random_variables = [torch.tensor(i.tolist(), dtype=torch.float64) for i in random_matrices]
    r = f(*random_variables)
    e = expr.subs(dict(zip(variables, random_matrices))).doit()

    if isinstance(e, _CodegenArrayAbstract):
        e = e.doit()

    if hasattr(e, 'is_number') and e.is_number:
        if isinstance(r, torch.Tensor) and r.dim() == 0:
            r = r.item()
            e = float(e)
            assert abs(r - e) < 1e-6
            return

    if e.is_Matrix or isinstance(e, NDimArray):
        e = torch.tensor(e.tolist(), dtype=torch.float64)
        assert torch.allclose(r, e, atol=1e-6)
    else:
        raise TypeError(f"Cannot compare {type(r)} with {type(e)}")


def _compare_torch_scalar(variables, expr, rng=lambda: random.uniform(-5, 5)):
    f = lambdify(variables, expr, 'torch')
    rvs = [rng() for v in variables]
    t_rvs = [torch.tensor(i, dtype=torch.float64) for i in rvs]
    r = f(*t_rvs)
    if isinstance(r, torch.Tensor):
        r = r.item()
    e = expr.subs(dict(zip(variables, rvs))).doit()
    assert abs(r - e) < 1e-6


def _compare_torch_relational(variables, expr, rng=lambda: random.randint(0, 10)):
    f = lambdify(variables, expr, 'torch')
    rvs = [rng() for v in variables]
    t_rvs = [torch.tensor(i, dtype=torch.float64) for i in rvs]
    r = f(*t_rvs)
    e = bool(expr.subs(dict(zip(variables, rvs))).doit())
    assert r.item() == e


def test_torch_math():
    if not torch:
        skip("PyTorch not installed")

    expr = Abs(x)
    assert torch_code(expr) == "torch.abs(x)"
    f = lambdify(x, expr, 'torch')
    ma = torch.tensor([[-1, 2, -3, -4]], dtype=torch.float64)
    y_abs = f(ma)
    c = torch.abs(ma)
    assert torch.all(y_abs == c)

    expr = sign(x)
    assert torch_code(expr) == "torch.sign(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-10, 10))

    expr = ceiling(x)
    assert torch_code(expr) == "torch.ceil(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.random())

    expr = floor(x)
    assert torch_code(expr) == "torch.floor(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.random())

    expr = exp(x)
    assert torch_code(expr) == "torch.exp(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-2, 2))

    expr = sqrt(x)
    assert torch_code(expr) == "torch.sqrt(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.random())

    expr = x ** 4
    assert torch_code(expr) == "torch.pow(x, 4)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.random())

    expr = cos(x)
    assert torch_code(expr) == "torch.cos(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.random())

    expr = acos(x)
    assert torch_code(expr) == "torch.acos(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-0.99, 0.99))

    expr = sin(x)
    assert torch_code(expr) == "torch.sin(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.random())

    expr = asin(x)
    assert torch_code(expr) == "torch.asin(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-0.99, 0.99))

    expr = tan(x)
    assert torch_code(expr) == "torch.tan(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-1.5, 1.5))

    expr = atan(x)
    assert torch_code(expr) == "torch.atan(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-5, 5))

    expr = atan2(y, x)
    assert torch_code(expr) == "torch.atan2(y, x)"
    _compare_torch_scalar((y, x), expr, rng=lambda: random.uniform(-5, 5))

    expr = cosh(x)
    assert torch_code(expr) == "torch.cosh(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-2, 2))

    expr = acosh(x)
    assert torch_code(expr) == "torch.acosh(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(1.1, 5))

    expr = sinh(x)
    assert torch_code(expr) == "torch.sinh(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-2, 2))

    expr = asinh(x)
    assert torch_code(expr) == "torch.asinh(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-5, 5))

    expr = tanh(x)
    assert torch_code(expr) == "torch.tanh(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-2, 2))

    expr = atanh(x)
    assert torch_code(expr) == "torch.atanh(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-0.9, 0.9))

    expr = erf(x)
    assert torch_code(expr) == "torch.erf(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(-2, 2))

    expr = loggamma(x)
    assert torch_code(expr) == "torch.lgamma(x)"
    _compare_torch_scalar((x,), expr, rng=lambda: random.uniform(0.5, 5))


def test_torch_complexes():
    assert torch_code(re(x)) == "torch.real(x)"
    assert torch_code(im(x)) == "torch.imag(x)"
    assert torch_code(arg(x)) == "torch.angle(x)"


def test_torch_relational():
    if not torch:
        skip("PyTorch not installed")

    expr = Eq(x, y)
    assert torch_code(expr) == "torch.eq(x, y)"
    _compare_torch_relational((x, y), expr)

    expr = Ne(x, y)
    assert torch_code(expr) == "torch.ne(x, y)"
    _compare_torch_relational((x, y), expr)

    expr = Ge(x, y)
    assert torch_code(expr) == "torch.ge(x, y)"
    _compare_torch_relational((x, y), expr)

    expr = Gt(x, y)
    assert torch_code(expr) == "torch.gt(x, y)"
    _compare_torch_relational((x, y), expr)

    expr = Le(x, y)
    assert torch_code(expr) == "torch.le(x, y)"
    _compare_torch_relational((x, y), expr)

    expr = Lt(x, y)
    assert torch_code(expr) == "torch.lt(x, y)"
    _compare_torch_relational((x, y), expr)


def test_torch_matrix():
    if torch is None:
        skip("PyTorch not installed")

    expr = M
    assert torch_code(expr) == "M"
    f = lambdify((M,), expr, "torch")
    eye_mat = eye(3)
    eye_tensor = torch.tensor(eye_mat.tolist(), dtype=torch.float64)
    assert torch.allclose(f(eye_tensor), eye_tensor)

    expr = M * N
    assert torch_code(expr) == "torch.matmul(M, N)"
    _compare_torch_matrix((M, N), expr)

    expr = M ** 3
    assert torch_code(expr) == "torch.mm(torch.mm(M, M), M)"
    _compare_torch_matrix((M,), expr)

    expr = M * N * P * Q
    assert torch_code(expr) == "torch.matmul(torch.matmul(torch.matmul(M, N), P), Q)"
    _compare_torch_matrix((M, N, P, Q), expr)

    expr = Trace(M)
    assert torch_code(expr) == "torch.trace(M)"
    _compare_torch_matrix((M,), expr)

    expr = Determinant(M)
    assert torch_code(expr) == "torch.det(M)"
    _compare_torch_matrix((M,), expr)

    expr = HadamardProduct(M, N)
    assert torch_code(expr) == "torch.mul(M, N)"
    _compare_torch_matrix((M, N), expr)

    expr = Inverse(M)
    assert torch_code(expr) == "torch.linalg.inv(M)"

    # For inverse, use a matrix that's guaranteed to be invertible
    eye_mat = eye(3)
    eye_tensor = torch.tensor(eye_mat.tolist(), dtype=torch.float64)
    f = lambdify((M,), expr, "torch")
    result = f(eye_tensor)
    expected = torch.linalg.inv(eye_tensor)
    assert torch.allclose(result, expected)


def test_torch_array_operations():
    if not torch:
        skip("PyTorch not installed")

    M = MatrixSymbol("M", 2, 2)
    N = MatrixSymbol("N", 2, 2)
    P = MatrixSymbol("P", 2, 2)
    Q = MatrixSymbol("Q", 2, 2)

    ma = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    mb = torch.tensor([[1., -2.], [-1., 3.]], dtype=torch.float64)
    mc = torch.tensor([[2., 0.], [1., 2.]], dtype=torch.float64)
    md = torch.tensor([[1., -1.], [4., 7.]], dtype=torch.float64)

    cg = ArrayTensorProduct(M, N)
    assert torch_code(cg) == 'torch.einsum("ab,cd", M, N)'
    f = lambdify((M, N), cg, 'torch')
    y = f(ma, mb)
    c = torch.einsum("ij,kl", ma, mb)
    assert torch.allclose(y, c)

    cg = ArrayAdd(M, N)
    assert torch_code(cg) == 'torch.add(M, N)'
    f = lambdify((M, N), cg, 'torch')
    y = f(ma, mb)
    c = ma + mb
    assert torch.allclose(y, c)

    cg = ArrayAdd(M, N, P)
    assert torch_code(cg) == 'torch.add(torch.add(M, N), P)'
    f = lambdify((M, N, P), cg, 'torch')
    y = f(ma, mb, mc)
    c = ma + mb + mc
    assert torch.allclose(y, c)

    cg = ArrayAdd(M, N, P, Q)
    assert torch_code(cg) == 'torch.add(torch.add(torch.add(M, N), P), Q)'
    f = lambdify((M, N, P, Q), cg, 'torch')
    y = f(ma, mb, mc, md)
    c = ma + mb + mc + md
    assert torch.allclose(y, c)

    cg = PermuteDims(M, [1, 0])
    assert torch_code(cg) == 'M.permute(1, 0)'
    f = lambdify((M,), cg, 'torch')
    y = f(ma)
    c = ma.T
    assert torch.allclose(y, c)

    cg = PermuteDims(ArrayTensorProduct(M, N), [1, 2, 3, 0])
    assert torch_code(cg) == 'torch.einsum("ab,cd", M, N).permute(1, 2, 3, 0)'
    f = lambdify((M, N), cg, 'torch')
    y = f(ma, mb)
    c = torch.einsum("ab,cd", ma, mb).permute(1, 2, 3, 0)
    assert torch.allclose(y, c)

    cg = ArrayDiagonal(ArrayTensorProduct(M, N), (1, 2))
    assert torch_code(cg) == 'torch.einsum("ab,bc->acb", M, N)'
    f = lambdify((M, N), cg, 'torch')
    y = f(ma, mb)
    c = torch.einsum("ab,bc->acb", ma, mb)
    assert torch.allclose(y, c)


def test_torch_derivative():
    """Test derivative handling."""
    expr = Derivative(sin(x), x)
    assert torch_code(expr) == 'torch.autograd.grad(torch.sin(x), x)[0]'


def test_torch_printing_dtype():
    if not torch:
        skip("PyTorch not installed")

    # matrix printing with default dtype
    expr = Matrix([[x, sin(y)], [exp(z), -t]])
    assert "dtype=torch.float64" in torch_code(expr)

    # explicit dtype
    assert "dtype=torch.float32" in torch_code(expr, dtype="torch.float32")

    # with requires_grad
    result = torch_code(expr, requires_grad=True)
    assert "requires_grad=True" in result
    assert "dtype=torch.float64" in result

    # both
    result = torch_code(expr, requires_grad=True, dtype="torch.float32")
    assert "requires_grad=True" in result
    assert "dtype=torch.float32" in result


def test_requires_grad():
    if not torch:
        skip("PyTorch not installed")

    expr = sin(x) + cos(y)
    f = lambdify([x, y], expr, 'torch')

    # make sure the gradients flow
    x_val = torch.tensor(1.0, requires_grad=True)
    y_val = torch.tensor(2.0, requires_grad=True)
    result = f(x_val, y_val)
    assert result.requires_grad
    result.backward()

    # x_val.grad should be cos(x_val) which is close to cos(1.0)
    assert abs(x_val.grad.item() - float(cos(1.0).evalf())) < 1e-6

    # y_val.grad should be -sin(y_val) which is close to -sin(2.0)
    assert abs(y_val.grad.item() - float(-sin(2.0).evalf())) < 1e-6


def test_torch_multi_variable_derivatives():
    if not torch:
        skip("PyTorch not installed")

    x, y, z = symbols("x y z")

    expr = Derivative(sin(x), x)
    assert torch_code(expr) == "torch.autograd.grad(torch.sin(x), x)[0]"

    expr = Derivative(sin(x), (x, 2))
    assert torch_code(
        expr) == "torch.autograd.grad(torch.autograd.grad(torch.sin(x), x, create_graph=True)[0], x, create_graph=True)[0]"

    expr = Derivative(sin(x * y), x, y)
    result = torch_code(expr)
    expected = "torch.autograd.grad(torch.autograd.grad(torch.sin(x*y), x, create_graph=True)[0], y, create_graph=True)[0]"
    normalized_result = result.replace(" ", "")
    normalized_expected = expected.replace(" ", "")
    assert normalized_result == normalized_expected

    expr = Derivative(sin(x), x, x)
    result = torch_code(expr)
    expected = "torch.autograd.grad(torch.autograd.grad(torch.sin(x), x, create_graph=True)[0], x, create_graph=True)[0]"
    assert result == expected

    expr = Derivative(sin(x * y * z), x, (y, 2), z)
    result = torch_code(expr)
    expected = "torch.autograd.grad(torch.autograd.grad(torch.autograd.grad(torch.autograd.grad(torch.sin(x*y*z), x, create_graph=True)[0], y, create_graph=True)[0], y, create_graph=True)[0], z, create_graph=True)[0]"
    normalized_result = result.replace(" ", "")
    normalized_expected = expected.replace(" ", "")
    assert normalized_result == normalized_expected


def test_torch_derivative_lambdify():
    if not torch:
        skip("PyTorch not installed")

    x = symbols("x")
    y = symbols("y")

    expr = Derivative(x ** 2, x)
    f = lambdify(x, expr, 'torch')
    x_val = torch.tensor(2.0, requires_grad=True)
    result = f(x_val)
    assert torch.isclose(result, torch.tensor(4.0))

    expr = Derivative(sin(x), (x, 2))
    f = lambdify(x, expr, 'torch')
    # Second derivative of sin(x) at x=0 is 0, not -1
    x_val = torch.tensor(0.0, requires_grad=True)
    result = f(x_val)
    assert torch.isclose(result, torch.tensor(0.0), atol=1e-5)

    x_val = torch.tensor(math.pi / 2, requires_grad=True)
    result = f(x_val)
    assert torch.isclose(result, torch.tensor(-1.0), atol=1e-5)

    expr = Derivative(x * y ** 2, x, y)
    f = lambdify((x, y), expr, 'torch')
    x_val = torch.tensor(2.0, requires_grad=True)
    y_val = torch.tensor(3.0, requires_grad=True)
    result = f(x_val, y_val)
    assert torch.isclose(result, torch.tensor(6.0))


def test_torch_special_matrices():
    if not torch:
        skip("PyTorch not installed")

    expr = Identity(3)
    assert torch_code(expr) == "torch.eye(3)"

    n = symbols("n")
    expr = Identity(n)
    assert torch_code(expr) == "torch.eye(n, n)"

    expr = ZeroMatrix(2, 3)
    assert torch_code(expr) == "torch.zeros((2, 3))"

    m, n = symbols("m n")
    expr = ZeroMatrix(m, n)
    assert torch_code(expr) == "torch.zeros((m, n))"

    expr = OneMatrix(2, 3)
    assert torch_code(expr) == "torch.ones((2, 3))"

    expr = OneMatrix(m, n)
    assert torch_code(expr) == "torch.ones((m, n))"


def test_torch_special_matrices_lambdify():
    if not torch:
        skip("PyTorch not installed")

    expr = Identity(3)
    f = lambdify([], expr, 'torch')
    result = f()
    expected = torch.eye(3)
    assert torch.allclose(result, expected)

    expr = ZeroMatrix(2, 3)
    f = lambdify([], expr, 'torch')
    result = f()
    expected = torch.zeros((2, 3))
    assert torch.allclose(result, expected)

    expr = OneMatrix(2, 3)
    f = lambdify([], expr, 'torch')
    result = f()
    expected = torch.ones((2, 3))
    assert torch.allclose(result, expected)


def test_torch_complex_operations():
    if not torch:
        skip("PyTorch not installed")

    expr = conjugate(x)
    assert torch_code(expr) == "torch.conj(x)"

    # SymPy distributes conjugate over addition and applies specific rules for each term
    expr = conjugate(sin(x) + I * cos(y))
    assert torch_code(expr) == "torch.sin(torch.conj(x)) - 1j*torch.cos(torch.conj(y))"

    expr = I
    assert torch_code(expr) == "1j"

    expr = 2 * I + x
    assert torch_code(expr) == "x + 2*1j"

    expr = exp(I * x)
    assert torch_code(expr) == "torch.exp(1j*x)"


def test_torch_special_functions():
    if not torch:
        skip("PyTorch not installed")

    expr = Heaviside(x)
    assert torch_code(expr) == "torch.heaviside(x, 1/2)"

    expr = Heaviside(x, 0)
    assert torch_code(expr) == "torch.heaviside(x, 0)"

    expr = gamma(x)
    assert torch_code(expr) == "torch.special.gamma(x)"

    expr = polygamma(0, x)  # Use polygamma instead of digamma because sympy will default to that anyway
    assert torch_code(expr) == "torch.special.digamma(x)"

    expr = gamma(sin(x))
    assert torch_code(expr) == "torch.special.gamma(torch.sin(x))"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\cast\test_promote.py ===
"""
These test the method maybe_promote from core/dtypes/cast.py
"""

import datetime
from decimal import Decimal

import numpy as np
import pytest

from pandas._libs.tslibs import NaT

from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna

import pandas as pd


def _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar=None):
    """
    Auxiliary function to unify testing of scalar/array promotion.

    Parameters
    ----------
    dtype : dtype
        The value to pass on as the first argument to maybe_promote.
    fill_value : scalar
        The value to pass on as the second argument to maybe_promote as
        a scalar.
    expected_dtype : dtype
        The expected dtype returned by maybe_promote (by design this is the
        same regardless of whether fill_value was passed as a scalar or in an
        array!).
    exp_val_for_scalar : scalar
        The expected value for the (potentially upcast) fill_value returned by
        maybe_promote.
    """
    assert is_scalar(fill_value)

    # here, we pass on fill_value as a scalar directly; the expected value
    # returned from maybe_promote is fill_value, potentially upcast to the
    # returned dtype.
    result_dtype, result_fill_value = maybe_promote(dtype, fill_value)
    expected_fill_value = exp_val_for_scalar

    assert result_dtype == expected_dtype
    _assert_match(result_fill_value, expected_fill_value)


def _assert_match(result_fill_value, expected_fill_value):
    # GH#23982/25425 require the same type in addition to equality/NA-ness
    res_type = type(result_fill_value)
    ex_type = type(expected_fill_value)

    if hasattr(result_fill_value, "dtype"):
        # Compare types in a way that is robust to platform-specific
        #  idiosyncrasies where e.g. sometimes we get "ulonglong" as an alias
        #  for "uint64" or "intc" as an alias for "int32"
        assert result_fill_value.dtype.kind == expected_fill_value.dtype.kind
        assert result_fill_value.dtype.itemsize == expected_fill_value.dtype.itemsize
    else:
        # On some builds, type comparison fails, e.g. np.int32 != np.int32
        assert res_type == ex_type or res_type.__name__ == ex_type.__name__

    match_value = result_fill_value == expected_fill_value
    if match_value is pd.NA:
        match_value = False

    # Note: type check above ensures that we have the _same_ NA value
    # for missing values, None == None (which is checked
    # through match_value above), but np.nan != np.nan and pd.NaT != pd.NaT
    match_missing = isna(result_fill_value) and isna(expected_fill_value)

    assert match_value or match_missing


@pytest.mark.parametrize(
    "dtype, fill_value, expected_dtype",
    [
        # size 8
        ("int8", 1, "int8"),
        ("int8", np.iinfo("int8").max + 1, "int16"),
        ("int8", np.iinfo("int16").max + 1, "int32"),
        ("int8", np.iinfo("int32").max + 1, "int64"),
        ("int8", np.iinfo("int64").max + 1, "object"),
        ("int8", -1, "int8"),
        ("int8", np.iinfo("int8").min - 1, "int16"),
        ("int8", np.iinfo("int16").min - 1, "int32"),
        ("int8", np.iinfo("int32").min - 1, "int64"),
        ("int8", np.iinfo("int64").min - 1, "object"),
        # keep signed-ness as long as possible
        ("uint8", 1, "uint8"),
        ("uint8", np.iinfo("int8").max + 1, "uint8"),
        ("uint8", np.iinfo("uint8").max + 1, "uint16"),
        ("uint8", np.iinfo("int16").max + 1, "uint16"),
        ("uint8", np.iinfo("uint16").max + 1, "uint32"),
        ("uint8", np.iinfo("int32").max + 1, "uint32"),
        ("uint8", np.iinfo("uint32").max + 1, "uint64"),
        ("uint8", np.iinfo("int64").max + 1, "uint64"),
        ("uint8", np.iinfo("uint64").max + 1, "object"),
        # max of uint8 cannot be contained in int8
        ("uint8", -1, "int16"),
        ("uint8", np.iinfo("int8").min - 1, "int16"),
        ("uint8", np.iinfo("int16").min - 1, "int32"),
        ("uint8", np.iinfo("int32").min - 1, "int64"),
        ("uint8", np.iinfo("int64").min - 1, "object"),
        # size 16
        ("int16", 1, "int16"),
        ("int16", np.iinfo("int8").max + 1, "int16"),
        ("int16", np.iinfo("int16").max + 1, "int32"),
        ("int16", np.iinfo("int32").max + 1, "int64"),
        ("int16", np.iinfo("int64").max + 1, "object"),
        ("int16", -1, "int16"),
        ("int16", np.iinfo("int8").min - 1, "int16"),
        ("int16", np.iinfo("int16").min - 1, "int32"),
        ("int16", np.iinfo("int32").min - 1, "int64"),
        ("int16", np.iinfo("int64").min - 1, "object"),
        ("uint16", 1, "uint16"),
        ("uint16", np.iinfo("int8").max + 1, "uint16"),
        ("uint16", np.iinfo("uint8").max + 1, "uint16"),
        ("uint16", np.iinfo("int16").max + 1, "uint16"),
        ("uint16", np.iinfo("uint16").max + 1, "uint32"),
        ("uint16", np.iinfo("int32").max + 1, "uint32"),
        ("uint16", np.iinfo("uint32").max + 1, "uint64"),
        ("uint16", np.iinfo("int64").max + 1, "uint64"),
        ("uint16", np.iinfo("uint64").max + 1, "object"),
        ("uint16", -1, "int32"),
        ("uint16", np.iinfo("int8").min - 1, "int32"),
        ("uint16", np.iinfo("int16").min - 1, "int32"),
        ("uint16", np.iinfo("int32").min - 1, "int64"),
        ("uint16", np.iinfo("int64").min - 1, "object"),
        # size 32
        ("int32", 1, "int32"),
        ("int32", np.iinfo("int8").max + 1, "int32"),
        ("int32", np.iinfo("int16").max + 1, "int32"),
        ("int32", np.iinfo("int32").max + 1, "int64"),
        ("int32", np.iinfo("int64").max + 1, "object"),
        ("int32", -1, "int32"),
        ("int32", np.iinfo("int8").min - 1, "int32"),
        ("int32", np.iinfo("int16").min - 1, "int32"),
        ("int32", np.iinfo("int32").min - 1, "int64"),
        ("int32", np.iinfo("int64").min - 1, "object"),
        ("uint32", 1, "uint32"),
        ("uint32", np.iinfo("int8").max + 1, "uint32"),
        ("uint32", np.iinfo("uint8").max + 1, "uint32"),
        ("uint32", np.iinfo("int16").max + 1, "uint32"),
        ("uint32", np.iinfo("uint16").max + 1, "uint32"),
        ("uint32", np.iinfo("int32").max + 1, "uint32"),
        ("uint32", np.iinfo("uint32").max + 1, "uint64"),
        ("uint32", np.iinfo("int64").max + 1, "uint64"),
        ("uint32", np.iinfo("uint64").max + 1, "object"),
        ("uint32", -1, "int64"),
        ("uint32", np.iinfo("int8").min - 1, "int64"),
        ("uint32", np.iinfo("int16").min - 1, "int64"),
        ("uint32", np.iinfo("int32").min - 1, "int64"),
        ("uint32", np.iinfo("int64").min - 1, "object"),
        # size 64
        ("int64", 1, "int64"),
        ("int64", np.iinfo("int8").max + 1, "int64"),
        ("int64", np.iinfo("int16").max + 1, "int64"),
        ("int64", np.iinfo("int32").max + 1, "int64"),
        ("int64", np.iinfo("int64").max + 1, "object"),
        ("int64", -1, "int64"),
        ("int64", np.iinfo("int8").min - 1, "int64"),
        ("int64", np.iinfo("int16").min - 1, "int64"),
        ("int64", np.iinfo("int32").min - 1, "int64"),
        ("int64", np.iinfo("int64").min - 1, "object"),
        ("uint64", 1, "uint64"),
        ("uint64", np.iinfo("int8").max + 1, "uint64"),
        ("uint64", np.iinfo("uint8").max + 1, "uint64"),
        ("uint64", np.iinfo("int16").max + 1, "uint64"),
        ("uint64", np.iinfo("uint16").max + 1, "uint64"),
        ("uint64", np.iinfo("int32").max + 1, "uint64"),
        ("uint64", np.iinfo("uint32").max + 1, "uint64"),
        ("uint64", np.iinfo("int64").max + 1, "uint64"),
        ("uint64", np.iinfo("uint64").max + 1, "object"),
        ("uint64", -1, "object"),
        ("uint64", np.iinfo("int8").min - 1, "object"),
        ("uint64", np.iinfo("int16").min - 1, "object"),
        ("uint64", np.iinfo("int32").min - 1, "object"),
        ("uint64", np.iinfo("int64").min - 1, "object"),
    ],
)
def test_maybe_promote_int_with_int(dtype, fill_value, expected_dtype):
    dtype = np.dtype(dtype)
    expected_dtype = np.dtype(expected_dtype)

    # output is not a generic int, but corresponds to expected_dtype
    exp_val_for_scalar = np.array([fill_value], dtype=expected_dtype)[0]

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_int_with_float(any_int_numpy_dtype, float_numpy_dtype):
    dtype = np.dtype(any_int_numpy_dtype)
    fill_dtype = np.dtype(float_numpy_dtype)

    # create array of given dtype; casts "1" to correct dtype
    fill_value = np.array([1], dtype=fill_dtype)[0]

    # filling int with float always upcasts to float64
    expected_dtype = np.float64
    # fill_value can be different float type
    exp_val_for_scalar = np.float64(fill_value)

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_float_with_int(float_numpy_dtype, any_int_numpy_dtype):
    dtype = np.dtype(float_numpy_dtype)
    fill_dtype = np.dtype(any_int_numpy_dtype)

    # create array of given dtype; casts "1" to correct dtype
    fill_value = np.array([1], dtype=fill_dtype)[0]

    # filling float with int always keeps float dtype
    # because: np.finfo('float32').max > np.iinfo('uint64').max
    expected_dtype = dtype
    # output is not a generic float, but corresponds to expected_dtype
    exp_val_for_scalar = np.array([fill_value], dtype=expected_dtype)[0]

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


@pytest.mark.parametrize(
    "dtype, fill_value, expected_dtype",
    [
        # float filled with float
        ("float32", 1, "float32"),
        ("float32", float(np.finfo("float32").max) * 1.1, "float64"),
        ("float64", 1, "float64"),
        ("float64", float(np.finfo("float32").max) * 1.1, "float64"),
        # complex filled with float
        ("complex64", 1, "complex64"),
        ("complex64", float(np.finfo("float32").max) * 1.1, "complex128"),
        ("complex128", 1, "complex128"),
        ("complex128", float(np.finfo("float32").max) * 1.1, "complex128"),
        # float filled with complex
        ("float32", 1 + 1j, "complex64"),
        ("float32", float(np.finfo("float32").max) * (1.1 + 1j), "complex128"),
        ("float64", 1 + 1j, "complex128"),
        ("float64", float(np.finfo("float32").max) * (1.1 + 1j), "complex128"),
        # complex filled with complex
        ("complex64", 1 + 1j, "complex64"),
        ("complex64", float(np.finfo("float32").max) * (1.1 + 1j), "complex128"),
        ("complex128", 1 + 1j, "complex128"),
        ("complex128", float(np.finfo("float32").max) * (1.1 + 1j), "complex128"),
    ],
)
def test_maybe_promote_float_with_float(dtype, fill_value, expected_dtype):
    dtype = np.dtype(dtype)
    expected_dtype = np.dtype(expected_dtype)

    # output is not a generic float, but corresponds to expected_dtype
    exp_val_for_scalar = np.array([fill_value], dtype=expected_dtype)[0]

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_bool_with_any(any_numpy_dtype):
    dtype = np.dtype(bool)
    fill_dtype = np.dtype(any_numpy_dtype)

    # create array of given dtype; casts "1" to correct dtype
    fill_value = np.array([1], dtype=fill_dtype)[0]

    # filling bool with anything but bool casts to object
    expected_dtype = np.dtype(object) if fill_dtype != bool else fill_dtype
    exp_val_for_scalar = fill_value

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_any_with_bool(any_numpy_dtype):
    dtype = np.dtype(any_numpy_dtype)
    fill_value = True

    # filling anything but bool with bool casts to object
    expected_dtype = np.dtype(object) if dtype != bool else dtype
    # output is not a generic bool, but corresponds to expected_dtype
    exp_val_for_scalar = np.array([fill_value], dtype=expected_dtype)[0]

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_bytes_with_any(bytes_dtype, any_numpy_dtype):
    dtype = np.dtype(bytes_dtype)
    fill_dtype = np.dtype(any_numpy_dtype)

    # create array of given dtype; casts "1" to correct dtype
    fill_value = np.array([1], dtype=fill_dtype)[0]

    # we never use bytes dtype internally, always promote to object
    expected_dtype = np.dtype(np.object_)
    exp_val_for_scalar = fill_value

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_any_with_bytes(any_numpy_dtype):
    dtype = np.dtype(any_numpy_dtype)

    # create array of given dtype
    fill_value = b"abc"

    # we never use bytes dtype internally, always promote to object
    expected_dtype = np.dtype(np.object_)
    # output is not a generic bytes, but corresponds to expected_dtype
    exp_val_for_scalar = np.array([fill_value], dtype=expected_dtype)[0]

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_datetime64_with_any(datetime64_dtype, any_numpy_dtype):
    dtype = np.dtype(datetime64_dtype)
    fill_dtype = np.dtype(any_numpy_dtype)

    # create array of given dtype; casts "1" to correct dtype
    fill_value = np.array([1], dtype=fill_dtype)[0]

    # filling datetime with anything but datetime casts to object
    if fill_dtype.kind == "M":
        expected_dtype = dtype
        # for datetime dtypes, scalar values get cast to to_datetime64
        exp_val_for_scalar = pd.Timestamp(fill_value).to_datetime64()
    else:
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


@pytest.mark.parametrize(
    "fill_value",
    [
        pd.Timestamp("now"),
        np.datetime64("now"),
        datetime.datetime.now(),
        datetime.date.today(),
    ],
    ids=["pd.Timestamp", "np.datetime64", "datetime.datetime", "datetime.date"],
)
def test_maybe_promote_any_with_datetime64(any_numpy_dtype, fill_value):
    dtype = np.dtype(any_numpy_dtype)

    # filling datetime with anything but datetime casts to object
    if dtype.kind == "M":
        expected_dtype = dtype
        # for datetime dtypes, scalar values get cast to pd.Timestamp.value
        exp_val_for_scalar = pd.Timestamp(fill_value).to_datetime64()
    else:
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value

    if type(fill_value) is datetime.date and dtype.kind == "M":
        # Casting date to dt64 is deprecated, in 2.0 enforced to cast to object
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


@pytest.mark.parametrize(
    "fill_value",
    [
        pd.Timestamp(2023, 1, 1),
        np.datetime64("2023-01-01"),
        datetime.datetime(2023, 1, 1),
        datetime.date(2023, 1, 1),
    ],
    ids=["pd.Timestamp", "np.datetime64", "datetime.datetime", "datetime.date"],
)
def test_maybe_promote_any_numpy_dtype_with_datetimetz(
    any_numpy_dtype, tz_aware_fixture, fill_value
):
    dtype = np.dtype(any_numpy_dtype)
    fill_dtype = DatetimeTZDtype(tz=tz_aware_fixture)

    fill_value = pd.Series([fill_value], dtype=fill_dtype)[0]

    # filling any numpy dtype with datetimetz casts to object
    expected_dtype = np.dtype(object)
    exp_val_for_scalar = fill_value

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_timedelta64_with_any(timedelta64_dtype, any_numpy_dtype):
    dtype = np.dtype(timedelta64_dtype)
    fill_dtype = np.dtype(any_numpy_dtype)

    # create array of given dtype; casts "1" to correct dtype
    fill_value = np.array([1], dtype=fill_dtype)[0]

    # filling timedelta with anything but timedelta casts to object
    if fill_dtype.kind == "m":
        expected_dtype = dtype
        # for timedelta dtypes, scalar values get cast to pd.Timedelta.value
        exp_val_for_scalar = pd.Timedelta(fill_value).to_timedelta64()
    else:
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


@pytest.mark.parametrize(
    "fill_value",
    [pd.Timedelta(days=1), np.timedelta64(24, "h"), datetime.timedelta(1)],
    ids=["pd.Timedelta", "np.timedelta64", "datetime.timedelta"],
)
def test_maybe_promote_any_with_timedelta64(any_numpy_dtype, fill_value):
    dtype = np.dtype(any_numpy_dtype)

    # filling anything but timedelta with timedelta casts to object
    if dtype.kind == "m":
        expected_dtype = dtype
        # for timedelta dtypes, scalar values get cast to pd.Timedelta.value
        exp_val_for_scalar = pd.Timedelta(fill_value).to_timedelta64()
    else:
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_string_with_any(string_dtype, any_numpy_dtype):
    dtype = np.dtype(string_dtype)
    fill_dtype = np.dtype(any_numpy_dtype)

    # create array of given dtype; casts "1" to correct dtype
    fill_value = np.array([1], dtype=fill_dtype)[0]

    # filling string with anything casts to object
    expected_dtype = np.dtype(object)
    exp_val_for_scalar = fill_value

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_any_with_string(any_numpy_dtype):
    dtype = np.dtype(any_numpy_dtype)

    # create array of given dtype
    fill_value = "abc"

    # filling anything with a string casts to object
    expected_dtype = np.dtype(object)
    exp_val_for_scalar = fill_value

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_object_with_any(object_dtype, any_numpy_dtype):
    dtype = np.dtype(object_dtype)
    fill_dtype = np.dtype(any_numpy_dtype)

    # create array of given dtype; casts "1" to correct dtype
    fill_value = np.array([1], dtype=fill_dtype)[0]

    # filling object with anything stays object
    expected_dtype = np.dtype(object)
    exp_val_for_scalar = fill_value

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_any_with_object(any_numpy_dtype):
    dtype = np.dtype(any_numpy_dtype)

    # create array of object dtype from a scalar value (i.e. passing
    # dtypes.common.is_scalar), which can however not be cast to int/float etc.
    fill_value = pd.DateOffset(1)

    # filling object with anything stays object
    expected_dtype = np.dtype(object)
    exp_val_for_scalar = fill_value

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_any_numpy_dtype_with_na(any_numpy_dtype, nulls_fixture):
    fill_value = nulls_fixture
    dtype = np.dtype(any_numpy_dtype)

    if isinstance(fill_value, Decimal):
        # Subject to change, but ATM (When Decimal(NAN) is being added to nulls_fixture)
        #  this is the existing behavior in maybe_promote,
        #  hinges on is_valid_na_for_dtype
        if dtype.kind in "iufc":
            if dtype.kind in "iu":
                expected_dtype = np.dtype(np.float64)
            else:
                expected_dtype = dtype
            exp_val_for_scalar = np.nan
        else:
            expected_dtype = np.dtype(object)
            exp_val_for_scalar = fill_value
    elif dtype.kind in "iu" and fill_value is not NaT:
        # integer + other missing value (np.nan / None) casts to float
        expected_dtype = np.float64
        exp_val_for_scalar = np.nan
    elif dtype == object and fill_value is NaT:
        # inserting into object does not cast the value
        # but *does* cast None to np.nan
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value
    elif dtype.kind in "mM":
        # datetime / timedelta cast all missing values to dtyped-NaT
        expected_dtype = dtype
        exp_val_for_scalar = dtype.type("NaT", "ns")
    elif fill_value is NaT:
        # NaT upcasts everything that's not datetime/timedelta to object
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = NaT
    elif dtype.kind in "fc":
        # float / complex + missing value (!= NaT) stays the same
        expected_dtype = dtype
        exp_val_for_scalar = np.nan
    else:
        # all other cases cast to object, and use np.nan as missing value
        expected_dtype = np.dtype(object)
        if fill_value is pd.NA:
            exp_val_for_scalar = pd.NA
        else:
            exp_val_for_scalar = np.nan

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\indexing\test_indexing.py ===
""" test get/set & misc """
from datetime import timedelta
import re

import numpy as np
import pytest

from pandas.errors import IndexingError

from pandas import (
    NA,
    DataFrame,
    Index,
    IndexSlice,
    MultiIndex,
    NaT,
    Series,
    Timedelta,
    Timestamp,
    concat,
    date_range,
    isna,
    period_range,
    timedelta_range,
)
import pandas._testing as tm


def test_basic_indexing():
    s = Series(
        np.random.default_rng(2).standard_normal(5), index=["a", "b", "a", "a", "b"]
    )

    warn_msg = "Series.__[sg]etitem__ treating keys as positions is deprecated"
    msg = "index 5 is out of bounds for axis 0 with size 5"
    with pytest.raises(IndexError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            s[5]
    with pytest.raises(IndexError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            s[5] = 0

    with pytest.raises(KeyError, match=r"^'c'$"):
        s["c"]

    s = s.sort_index()

    with pytest.raises(IndexError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            s[5]
    msg = r"index 5 is out of bounds for axis (0|1) with size 5|^5$"
    with pytest.raises(IndexError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            s[5] = 0


def test_getitem_numeric_should_not_fallback_to_positional(any_numeric_dtype):
    # GH51053
    dtype = any_numeric_dtype
    idx = Index([1, 0, 1], dtype=dtype)
    ser = Series(range(3), index=idx)
    result = ser[1]
    expected = Series([0, 2], index=Index([1, 1], dtype=dtype))
    tm.assert_series_equal(result, expected, check_exact=True)


def test_setitem_numeric_should_not_fallback_to_positional(any_numeric_dtype):
    # GH51053
    dtype = any_numeric_dtype
    idx = Index([1, 0, 1], dtype=dtype)
    ser = Series(range(3), index=idx)
    ser[1] = 10
    expected = Series([10, 1, 10], index=idx)
    tm.assert_series_equal(ser, expected, check_exact=True)


def test_basic_getitem_with_labels(datetime_series):
    indices = datetime_series.index[[5, 10, 15]]

    result = datetime_series[indices]
    expected = datetime_series.reindex(indices)
    tm.assert_series_equal(result, expected)

    result = datetime_series[indices[0] : indices[2]]
    expected = datetime_series.loc[indices[0] : indices[2]]
    tm.assert_series_equal(result, expected)


def test_basic_getitem_dt64tz_values():
    # GH12089
    # with tz for values
    ser = Series(
        date_range("2011-01-01", periods=3, tz="US/Eastern"), index=["a", "b", "c"]
    )
    expected = Timestamp("2011-01-01", tz="US/Eastern")
    result = ser.loc["a"]
    assert result == expected
    result = ser.iloc[0]
    assert result == expected
    result = ser["a"]
    assert result == expected


def test_getitem_setitem_ellipsis(using_copy_on_write, warn_copy_on_write):
    s = Series(np.random.default_rng(2).standard_normal(10))

    result = s[...]
    tm.assert_series_equal(result, s)

    with tm.assert_cow_warning(warn_copy_on_write):
        s[...] = 5
    if not using_copy_on_write:
        assert (result == 5).all()


@pytest.mark.parametrize(
    "result_1, duplicate_item, expected_1",
    [
        [
            Series({1: 12, 2: [1, 2, 2, 3]}),
            Series({1: 313}),
            Series({1: 12}, dtype=object),
        ],
        [
            Series({1: [1, 2, 3], 2: [1, 2, 2, 3]}),
            Series({1: [1, 2, 3]}),
            Series({1: [1, 2, 3]}),
        ],
    ],
)
def test_getitem_with_duplicates_indices(result_1, duplicate_item, expected_1):
    # GH 17610
    result = result_1._append(duplicate_item)
    expected = expected_1._append(duplicate_item)
    tm.assert_series_equal(result[1], expected)
    assert result[2] == result_1[2]


def test_getitem_setitem_integers():
    # caused bug without test
    s = Series([1, 2, 3], ["a", "b", "c"])

    assert s.iloc[0] == s["a"]
    s.iloc[0] = 5
    tm.assert_almost_equal(s["a"], 5)


def test_series_box_timestamp():
    rng = date_range("20090415", "20090519", freq="B")
    ser = Series(rng)
    assert isinstance(ser[0], Timestamp)
    assert isinstance(ser.at[1], Timestamp)
    assert isinstance(ser.iat[2], Timestamp)
    assert isinstance(ser.loc[3], Timestamp)
    assert isinstance(ser.iloc[4], Timestamp)

    ser = Series(rng, index=rng)
    msg = "Series.__getitem__ treating keys as positions is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert isinstance(ser[0], Timestamp)
    assert isinstance(ser.at[rng[1]], Timestamp)
    assert isinstance(ser.iat[2], Timestamp)
    assert isinstance(ser.loc[rng[3]], Timestamp)
    assert isinstance(ser.iloc[4], Timestamp)


def test_series_box_timedelta():
    rng = timedelta_range("1 day 1 s", periods=5, freq="h")
    ser = Series(rng)
    assert isinstance(ser[0], Timedelta)
    assert isinstance(ser.at[1], Timedelta)
    assert isinstance(ser.iat[2], Timedelta)
    assert isinstance(ser.loc[3], Timedelta)
    assert isinstance(ser.iloc[4], Timedelta)


def test_getitem_ambiguous_keyerror(indexer_sl):
    ser = Series(range(10), index=list(range(0, 20, 2)))
    with pytest.raises(KeyError, match=r"^1$"):
        indexer_sl(ser)[1]


def test_getitem_dups_with_missing(indexer_sl):
    # breaks reindex, so need to use .loc internally
    # GH 4246
    ser = Series([1, 2, 3, 4], ["foo", "bar", "foo", "bah"])
    with pytest.raises(KeyError, match=re.escape("['bam'] not in index")):
        indexer_sl(ser)[["foo", "bar", "bah", "bam"]]


def test_setitem_ambiguous_keyerror(indexer_sl):
    s = Series(range(10), index=list(range(0, 20, 2)))

    # equivalent of an append
    s2 = s.copy()
    indexer_sl(s2)[1] = 5
    expected = concat([s, Series([5], index=[1])])
    tm.assert_series_equal(s2, expected)


def test_setitem(datetime_series):
    datetime_series[datetime_series.index[5]] = np.nan
    datetime_series.iloc[[1, 2, 17]] = np.nan
    datetime_series.iloc[6] = np.nan
    assert np.isnan(datetime_series.iloc[6])
    assert np.isnan(datetime_series.iloc[2])
    datetime_series[np.isnan(datetime_series)] = 5
    assert not np.isnan(datetime_series.iloc[2])


def test_setslice(datetime_series):
    sl = datetime_series[5:20]
    assert len(sl) == len(sl.index)
    assert sl.index.is_unique is True


def test_basic_getitem_setitem_corner(datetime_series):
    # invalid tuples, e.g. td.ts[:, None] vs. td.ts[:, 2]
    msg = "key of type tuple not found and not a MultiIndex"
    with pytest.raises(KeyError, match=msg):
        datetime_series[:, 2]
    with pytest.raises(KeyError, match=msg):
        datetime_series[:, 2] = 2

    # weird lists. [slice(0, 5)] raises but not two slices
    msg = "Indexing with a single-item list"
    with pytest.raises(ValueError, match=msg):
        # GH#31299
        datetime_series[[slice(None, 5)]]

    # but we're OK with a single-element tuple
    result = datetime_series[(slice(None, 5),)]
    expected = datetime_series[:5]
    tm.assert_series_equal(result, expected)

    # OK
    msg = r"unhashable type(: 'slice')?"
    with pytest.raises(TypeError, match=msg):
        datetime_series[[5, [None, None]]]
    with pytest.raises(TypeError, match=msg):
        datetime_series[[5, [None, None]]] = 2


def test_slice(string_series, object_series, using_copy_on_write, warn_copy_on_write):
    original = string_series.copy()
    numSlice = string_series[10:20]
    numSliceEnd = string_series[-10:]
    objSlice = object_series[10:20]

    assert string_series.index[9] not in numSlice.index
    assert object_series.index[9] not in objSlice.index

    assert len(numSlice) == len(numSlice.index)
    assert string_series[numSlice.index[0]] == numSlice[numSlice.index[0]]

    assert numSlice.index[1] == string_series.index[11]
    tm.assert_numpy_array_equal(np.array(numSliceEnd), np.array(string_series)[-10:])

    # Test return view.
    sl = string_series[10:20]
    with tm.assert_cow_warning(warn_copy_on_write):
        sl[:] = 0

    if using_copy_on_write:
        # Doesn't modify parent (CoW)
        tm.assert_series_equal(string_series, original)
    else:
        assert (string_series[10:20] == 0).all()


def test_timedelta_assignment():
    # GH 8209
    s = Series([], dtype=object)
    s.loc["B"] = timedelta(1)
    expected = Series(
        Timedelta("1 days"), dtype="timedelta64[ns]", index=Index(["B"], dtype=object)
    )
    tm.assert_series_equal(s, expected)

    s = s.reindex(s.index.insert(0, "A"))
    expected = Series(
        [np.nan, Timedelta("1 days")],
        dtype="timedelta64[ns]",
        index=Index(["A", "B"], dtype=object),
    )
    tm.assert_series_equal(s, expected)

    s.loc["A"] = timedelta(1)
    expected = Series(
        Timedelta("1 days"),
        dtype="timedelta64[ns]",
        index=Index(["A", "B"], dtype=object),
    )
    tm.assert_series_equal(s, expected)


def test_underlying_data_conversion(using_copy_on_write):
    # GH 4080
    df = DataFrame({c: [1, 2, 3] for c in ["a", "b", "c"]})
    return_value = df.set_index(["a", "b", "c"], inplace=True)
    assert return_value is None
    s = Series([1], index=[(2, 2, 2)])
    df["val"] = 0
    df_original = df.copy()
    df

    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df["val"].update(s)
        expected = df_original
    else:
        with tm.assert_produces_warning(FutureWarning, match="inplace method"):
            df["val"].update(s)
        expected = DataFrame(
            {"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3], "val": [0, 1, 0]}
        )
        return_value = expected.set_index(["a", "b", "c"], inplace=True)
        assert return_value is None
    tm.assert_frame_equal(df, expected)


def test_preserve_refs(datetime_series):
    seq = datetime_series.iloc[[5, 10, 15]]
    seq.iloc[1] = np.nan
    assert not np.isnan(datetime_series.iloc[10])


def test_multilevel_preserve_name(lexsorted_two_level_string_multiindex, indexer_sl):
    index = lexsorted_two_level_string_multiindex
    ser = Series(
        np.random.default_rng(2).standard_normal(len(index)), index=index, name="sth"
    )

    result = indexer_sl(ser)["foo"]
    assert result.name == ser.name


# miscellaneous methods


@pytest.mark.parametrize(
    "index",
    [
        date_range("2014-01-01", periods=20, freq="MS"),
        period_range("2014-01", periods=20, freq="M"),
        timedelta_range("0", periods=20, freq="h"),
    ],
)
def test_slice_with_negative_step(index):
    keystr1 = str(index[9])
    keystr2 = str(index[13])

    ser = Series(np.arange(20), index)
    SLC = IndexSlice

    for key in [keystr1, index[9]]:
        tm.assert_indexing_slices_equivalent(ser, SLC[key::-1], SLC[9::-1])
        tm.assert_indexing_slices_equivalent(ser, SLC[:key:-1], SLC[:8:-1])

        for key2 in [keystr2, index[13]]:
            tm.assert_indexing_slices_equivalent(ser, SLC[key2:key:-1], SLC[13:8:-1])
            tm.assert_indexing_slices_equivalent(ser, SLC[key:key2:-1], SLC[0:0:-1])


def test_tuple_index():
    # GH 35534 - Selecting values when a Series has an Index of tuples
    s = Series([1, 2], index=[("a",), ("b",)])
    assert s[("a",)] == 1
    assert s[("b",)] == 2
    s[("b",)] = 3
    assert s[("b",)] == 3


def test_frozenset_index():
    # GH35747 - Selecting values when a Series has an Index of frozenset
    idx0, idx1 = frozenset("a"), frozenset("b")
    s = Series([1, 2], index=[idx0, idx1])
    assert s[idx0] == 1
    assert s[idx1] == 2
    s[idx1] = 3
    assert s[idx1] == 3


def test_loc_setitem_all_false_indexer():
    # GH#45778
    ser = Series([1, 2], index=["a", "b"])
    expected = ser.copy()
    rhs = Series([6, 7], index=["a", "b"])
    ser.loc[ser > 100] = rhs
    tm.assert_series_equal(ser, expected)


def test_loc_boolean_indexer_non_matching_index():
    # GH#46551
    ser = Series([1])
    result = ser.loc[Series([NA, False], dtype="boolean")]
    expected = Series([], dtype="int64")
    tm.assert_series_equal(result, expected)


def test_loc_boolean_indexer_miss_matching_index():
    # GH#46551
    ser = Series([1])
    indexer = Series([NA, False], dtype="boolean", index=[1, 2])
    with pytest.raises(IndexingError, match="Unalignable"):
        ser.loc[indexer]


def test_loc_setitem_nested_data_enlargement():
    # GH#48614
    df = DataFrame({"a": [1]})
    ser = Series({"label": df})
    ser.loc["new_label"] = df
    expected = Series({"label": df, "new_label": df})
    tm.assert_series_equal(ser, expected)


def test_loc_ea_numeric_index_oob_slice_end():
    # GH#50161
    ser = Series(1, index=Index([0, 1, 2], dtype="Int64"))
    result = ser.loc[2:3]
    expected = Series(1, index=Index([2], dtype="Int64"))
    tm.assert_series_equal(result, expected)


def test_getitem_bool_int_key():
    # GH#48653
    ser = Series({True: 1, False: 0})
    with pytest.raises(KeyError, match="0"):
        ser.loc[0]


@pytest.mark.parametrize("val", [{}, {"b": "x"}])
@pytest.mark.parametrize("indexer", [[], [False, False], slice(0, -1), np.array([])])
def test_setitem_empty_indexer(indexer, val):
    # GH#45981
    df = DataFrame({"a": [1, 2], **val})
    expected = df.copy()
    df.loc[indexer] = 1.5
    tm.assert_frame_equal(df, expected)


class TestDeprecatedIndexers:
    @pytest.mark.parametrize("key", [{1}, {1: 1}])
    def test_getitem_dict_and_set_deprecated(self, key):
        # GH#42825 enforced in 2.0
        ser = Series([1, 2])
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            ser.loc[key]

    @pytest.mark.parametrize("key", [{1}, {1: 1}, ({1}, 2), ({1: 1}, 2)])
    def test_getitem_dict_and_set_deprecated_multiindex(self, key):
        # GH#42825 enforced in 2.0
        ser = Series([1, 2], index=MultiIndex.from_tuples([(1, 2), (3, 4)]))
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            ser.loc[key]

    @pytest.mark.parametrize("key", [{1}, {1: 1}])
    def test_setitem_dict_and_set_disallowed(self, key):
        # GH#42825 enforced in 2.0
        ser = Series([1, 2])
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            ser.loc[key] = 1

    @pytest.mark.parametrize("key", [{1}, {1: 1}, ({1}, 2), ({1: 1}, 2)])
    def test_setitem_dict_and_set_disallowed_multiindex(self, key):
        # GH#42825 enforced in 2.0
        ser = Series([1, 2], index=MultiIndex.from_tuples([(1, 2), (3, 4)]))
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            ser.loc[key] = 1


class TestSetitemValidation:
    # This is adapted from pandas/tests/arrays/masked/test_indexing.py
    # but checks for warnings instead of errors.
    def _check_setitem_invalid(self, ser, invalid, indexer, warn):
        msg = "Setting an item of incompatible dtype is deprecated"
        msg = re.escape(msg)

        orig_ser = ser.copy()

        with tm.assert_produces_warning(warn, match=msg):
            ser[indexer] = invalid
            ser = orig_ser.copy()

        with tm.assert_produces_warning(warn, match=msg):
            ser.iloc[indexer] = invalid
            ser = orig_ser.copy()

        with tm.assert_produces_warning(warn, match=msg):
            ser.loc[indexer] = invalid
            ser = orig_ser.copy()

        with tm.assert_produces_warning(warn, match=msg):
            ser[:] = invalid

    _invalid_scalars = [
        1 + 2j,
        "True",
        "1",
        "1.0",
        NaT,
        np.datetime64("NaT"),
        np.timedelta64("NaT"),
    ]
    _indexers = [0, [0], slice(0, 1), [True, False, False], slice(None, None, None)]

    @pytest.mark.parametrize(
        "invalid", _invalid_scalars + [1, 1.0, np.int64(1), np.float64(1)]
    )
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_bool(self, invalid, indexer):
        ser = Series([True, False, False], dtype="bool")
        self._check_setitem_invalid(ser, invalid, indexer, FutureWarning)

    @pytest.mark.parametrize("invalid", _invalid_scalars + [True, 1.5, np.float64(1.5)])
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_int(self, invalid, any_int_numpy_dtype, indexer):
        ser = Series([1, 2, 3], dtype=any_int_numpy_dtype)
        if isna(invalid) and invalid is not NaT and not np.isnat(invalid):
            warn = None
        else:
            warn = FutureWarning
        self._check_setitem_invalid(ser, invalid, indexer, warn)

    @pytest.mark.parametrize("invalid", _invalid_scalars + [True])
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_float(self, invalid, float_numpy_dtype, indexer):
        ser = Series([1, 2, None], dtype=float_numpy_dtype)
        self._check_setitem_invalid(ser, invalid, indexer, FutureWarning)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_to_records.py ===
from collections import abc
import email
from email.parser import Parser

import numpy as np
import pytest

from pandas import (
    CategoricalDtype,
    DataFrame,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestDataFrameToRecords:
    def test_to_records_timeseries(self):
        index = date_range("1/1/2000", periods=10)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)),
            index=index,
            columns=["a", "b", "c"],
        )

        result = df.to_records()
        assert result["index"].dtype == "M8[ns]"

        result = df.to_records(index=False)

    def test_to_records_dt64(self):
        df = DataFrame(
            [["one", "two", "three"], ["four", "five", "six"]],
            index=date_range("2012-01-01", "2012-01-02"),
        )

        expected = df.index.values[0]
        result = df.to_records()["index"][0]
        assert expected == result

    def test_to_records_dt64tz_column(self):
        # GH#32535 dont less tz in to_records
        df = DataFrame({"A": date_range("2012-01-01", "2012-01-02", tz="US/Eastern")})

        result = df.to_records()

        assert result.dtype["A"] == object
        val = result[0][1]
        assert isinstance(val, Timestamp)
        assert val == df.loc[0, "A"]

    def test_to_records_with_multindex(self):
        # GH#3189
        index = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        data = np.zeros((8, 4))
        df = DataFrame(data, index=index)
        r = df.to_records(index=True)["level_0"]
        assert "bar" in r
        assert "one" not in r

    def test_to_records_with_Mapping_type(self):
        abc.Mapping.register(email.message.Message)

        headers = Parser().parsestr(
            "From: <user@example.com>\n"
            "To: <someone_else@example.com>\n"
            "Subject: Test message\n"
            "\n"
            "Body would go here\n"
        )

        frame = DataFrame.from_records([headers])
        all(x in frame for x in ["Type", "Subject", "From"])

    def test_to_records_floats(self):
        df = DataFrame(np.random.default_rng(2).random((10, 10)))
        df.to_records()

    def test_to_records_index_name(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)))
        df.index.name = "X"
        rs = df.to_records()
        assert "X" in rs.dtype.fields

        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)))
        rs = df.to_records()
        assert "index" in rs.dtype.fields

        df.index = MultiIndex.from_tuples([("a", "x"), ("a", "y"), ("b", "z")])
        df.index.names = ["A", None]
        result = df.to_records()
        expected = np.rec.fromarrays(
            [np.array(["a", "a", "b"]), np.array(["x", "y", "z"])]
            + [np.asarray(df.iloc[:, i]) for i in range(3)],
            dtype={
                "names": ["A", "level_1", "0", "1", "2"],
                "formats": [
                    "O",
                    "O",
                    f"{tm.ENDIAN}f8",
                    f"{tm.ENDIAN}f8",
                    f"{tm.ENDIAN}f8",
                ],
            },
        )
        tm.assert_numpy_array_equal(result, expected)

    def test_to_records_with_unicode_index(self):
        # GH#13172
        # unicode_literals conflict with to_records
        result = DataFrame([{"a": "x", "b": "y"}]).set_index("a").to_records()
        expected = np.rec.array([("x", "y")], dtype=[("a", "O"), ("b", "O")])
        tm.assert_almost_equal(result, expected)

    def test_to_records_index_dtype(self):
        # GH 47263: consistent data types for Index and MultiIndex
        df = DataFrame(
            {
                1: date_range("2022-01-01", periods=2),
                2: date_range("2022-01-01", periods=2),
                3: date_range("2022-01-01", periods=2),
            }
        )

        expected = np.rec.array(
            [
                ("2022-01-01", "2022-01-01", "2022-01-01"),
                ("2022-01-02", "2022-01-02", "2022-01-02"),
            ],
            dtype=[
                ("1", f"{tm.ENDIAN}M8[ns]"),
                ("2", f"{tm.ENDIAN}M8[ns]"),
                ("3", f"{tm.ENDIAN}M8[ns]"),
            ],
        )

        result = df.to_records(index=False)
        tm.assert_almost_equal(result, expected)

        result = df.set_index(1).to_records(index=True)
        tm.assert_almost_equal(result, expected)

        result = df.set_index([1, 2]).to_records(index=True)
        tm.assert_almost_equal(result, expected)

    def test_to_records_with_unicode_column_names(self):
        # xref issue: https://github.com/numpy/numpy/issues/2407
        # Issue GH#11879. to_records used to raise an exception when used
        # with column names containing non-ascii characters in Python 2
        result = DataFrame(data={"accented_name_é": [1.0]}).to_records()

        # Note that numpy allows for unicode field names but dtypes need
        # to be specified using dictionary instead of list of tuples.
        expected = np.rec.array(
            [(0, 1.0)],
            dtype={"names": ["index", "accented_name_é"], "formats": ["=i8", "=f8"]},
        )
        tm.assert_almost_equal(result, expected)

    def test_to_records_with_categorical(self):
        # GH#8626

        # dict creation
        df = DataFrame({"A": list("abc")}, dtype="category")
        expected = Series(list("abc"), dtype="category", name="A")
        tm.assert_series_equal(df["A"], expected)

        # list-like creation
        df = DataFrame(list("abc"), dtype="category")
        expected = Series(list("abc"), dtype="category", name=0)
        tm.assert_series_equal(df[0], expected)

        # to record array
        # this coerces
        result = df.to_records()
        expected = np.rec.array(
            [(0, "a"), (1, "b"), (2, "c")], dtype=[("index", "=i8"), ("0", "O")]
        )
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            # No dtypes --> default to array dtypes.
            (
                {},
                np.rec.array(
                    [(0, 1, 0.2, "a"), (1, 2, 1.5, "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}i8"),
                        ("A", f"{tm.ENDIAN}i8"),
                        ("B", f"{tm.ENDIAN}f8"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Should have no effect in this case.
            (
                {"index": True},
                np.rec.array(
                    [(0, 1, 0.2, "a"), (1, 2, 1.5, "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}i8"),
                        ("A", f"{tm.ENDIAN}i8"),
                        ("B", f"{tm.ENDIAN}f8"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Column dtype applied across the board. Index unaffected.
            (
                {"column_dtypes": f"{tm.ENDIAN}U4"},
                np.rec.array(
                    [("0", "1", "0.2", "a"), ("1", "2", "1.5", "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}i8"),
                        ("A", f"{tm.ENDIAN}U4"),
                        ("B", f"{tm.ENDIAN}U4"),
                        ("C", f"{tm.ENDIAN}U4"),
                    ],
                ),
            ),
            # Index dtype applied across the board. Columns unaffected.
            (
                {"index_dtypes": f"{tm.ENDIAN}U1"},
                np.rec.array(
                    [("0", 1, 0.2, "a"), ("1", 2, 1.5, "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}U1"),
                        ("A", f"{tm.ENDIAN}i8"),
                        ("B", f"{tm.ENDIAN}f8"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Pass in a type instance.
            (
                {"column_dtypes": str},
                np.rec.array(
                    [("0", "1", "0.2", "a"), ("1", "2", "1.5", "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}i8"),
                        ("A", f"{tm.ENDIAN}U"),
                        ("B", f"{tm.ENDIAN}U"),
                        ("C", f"{tm.ENDIAN}U"),
                    ],
                ),
            ),
            # Pass in a dtype instance.
            (
                {"column_dtypes": np.dtype(np.str_)},
                np.rec.array(
                    [("0", "1", "0.2", "a"), ("1", "2", "1.5", "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}i8"),
                        ("A", f"{tm.ENDIAN}U"),
                        ("B", f"{tm.ENDIAN}U"),
                        ("C", f"{tm.ENDIAN}U"),
                    ],
                ),
            ),
            # Pass in a dictionary (name-only).
            (
                {
                    "column_dtypes": {
                        "A": np.int8,
                        "B": np.float32,
                        "C": f"{tm.ENDIAN}U2",
                    }
                },
                np.rec.array(
                    [("0", "1", "0.2", "a"), ("1", "2", "1.5", "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}i8"),
                        ("A", "i1"),
                        ("B", f"{tm.ENDIAN}f4"),
                        ("C", f"{tm.ENDIAN}U2"),
                    ],
                ),
            ),
            # Pass in a dictionary (indices-only).
            (
                {"index_dtypes": {0: "int16"}},
                np.rec.array(
                    [(0, 1, 0.2, "a"), (1, 2, 1.5, "bc")],
                    dtype=[
                        ("index", "i2"),
                        ("A", f"{tm.ENDIAN}i8"),
                        ("B", f"{tm.ENDIAN}f8"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Ignore index mappings if index is not True.
            (
                {"index": False, "index_dtypes": f"{tm.ENDIAN}U2"},
                np.rec.array(
                    [(1, 0.2, "a"), (2, 1.5, "bc")],
                    dtype=[
                        ("A", f"{tm.ENDIAN}i8"),
                        ("B", f"{tm.ENDIAN}f8"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Non-existent names / indices in mapping should not error.
            (
                {"index_dtypes": {0: "int16", "not-there": "float32"}},
                np.rec.array(
                    [(0, 1, 0.2, "a"), (1, 2, 1.5, "bc")],
                    dtype=[
                        ("index", "i2"),
                        ("A", f"{tm.ENDIAN}i8"),
                        ("B", f"{tm.ENDIAN}f8"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Names / indices not in mapping default to array dtype.
            (
                {"column_dtypes": {"A": np.int8, "B": np.float32}},
                np.rec.array(
                    [("0", "1", "0.2", "a"), ("1", "2", "1.5", "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}i8"),
                        ("A", "i1"),
                        ("B", f"{tm.ENDIAN}f4"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Names / indices not in dtype mapping default to array dtype.
            (
                {"column_dtypes": {"A": np.dtype("int8"), "B": np.dtype("float32")}},
                np.rec.array(
                    [("0", "1", "0.2", "a"), ("1", "2", "1.5", "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}i8"),
                        ("A", "i1"),
                        ("B", f"{tm.ENDIAN}f4"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Mixture of everything.
            (
                {
                    "column_dtypes": {"A": np.int8, "B": np.float32},
                    "index_dtypes": f"{tm.ENDIAN}U2",
                },
                np.rec.array(
                    [("0", "1", "0.2", "a"), ("1", "2", "1.5", "bc")],
                    dtype=[
                        ("index", f"{tm.ENDIAN}U2"),
                        ("A", "i1"),
                        ("B", f"{tm.ENDIAN}f4"),
                        ("C", "O"),
                    ],
                ),
            ),
            # Invalid dype values.
            (
                {"index": False, "column_dtypes": []},
                (ValueError, "Invalid dtype \\[\\] specified for column A"),
            ),
            (
                {"index": False, "column_dtypes": {"A": "int32", "B": 5}},
                (ValueError, "Invalid dtype 5 specified for column B"),
            ),
            # Numpy can't handle EA types, so check error is raised
            (
                {
                    "index": False,
                    "column_dtypes": {"A": "int32", "B": CategoricalDtype(["a", "b"])},
                },
                (ValueError, "Invalid dtype category specified for column B"),
            ),
            # Check that bad types raise
            (
                {"index": False, "column_dtypes": {"A": "int32", "B": "foo"}},
                (TypeError, "data type [\"']foo[\"'] not understood"),
            ),
        ],
    )
    def test_to_records_dtype(self, kwargs, expected):
        # see GH#18146
        df = DataFrame({"A": [1, 2], "B": [0.2, 1.5], "C": ["a", "bc"]})

        if not isinstance(expected, np.rec.recarray):
            with pytest.raises(expected[0], match=expected[1]):
                df.to_records(**kwargs)
        else:
            result = df.to_records(**kwargs)
            tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize(
        "df,kwargs,expected",
        [
            # MultiIndex in the index.
            (
                DataFrame(
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=list("abc")
                ).set_index(["a", "b"]),
                {"column_dtypes": "float64", "index_dtypes": {0: "int32", 1: "int8"}},
                np.rec.array(
                    [(1, 2, 3.0), (4, 5, 6.0), (7, 8, 9.0)],
                    dtype=[
                        ("a", f"{tm.ENDIAN}i4"),
                        ("b", "i1"),
                        ("c", f"{tm.ENDIAN}f8"),
                    ],
                ),
            ),
            # MultiIndex in the columns.
            (
                DataFrame(
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    columns=MultiIndex.from_tuples(
                        [("a", "d"), ("b", "e"), ("c", "f")]
                    ),
                ),
                {
                    "column_dtypes": {0: f"{tm.ENDIAN}U1", 2: "float32"},
                    "index_dtypes": "float32",
                },
                np.rec.array(
                    [(0.0, "1", 2, 3.0), (1.0, "4", 5, 6.0), (2.0, "7", 8, 9.0)],
                    dtype=[
                        ("index", f"{tm.ENDIAN}f4"),
                        ("('a', 'd')", f"{tm.ENDIAN}U1"),
                        ("('b', 'e')", f"{tm.ENDIAN}i8"),
                        ("('c', 'f')", f"{tm.ENDIAN}f4"),
                    ],
                ),
            ),
            # MultiIndex in both the columns and index.
            (
                DataFrame(
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    columns=MultiIndex.from_tuples(
                        [("a", "d"), ("b", "e"), ("c", "f")], names=list("ab")
                    ),
                    index=MultiIndex.from_tuples(
                        [("d", -4), ("d", -5), ("f", -6)], names=list("cd")
                    ),
                ),
                {
                    "column_dtypes": "float64",
                    "index_dtypes": {0: f"{tm.ENDIAN}U2", 1: "int8"},
                },
                np.rec.array(
                    [
                        ("d", -4, 1.0, 2.0, 3.0),
                        ("d", -5, 4.0, 5.0, 6.0),
                        ("f", -6, 7, 8, 9.0),
                    ],
                    dtype=[
                        ("c", f"{tm.ENDIAN}U2"),
                        ("d", "i1"),
                        ("('a', 'd')", f"{tm.ENDIAN}f8"),
                        ("('b', 'e')", f"{tm.ENDIAN}f8"),
                        ("('c', 'f')", f"{tm.ENDIAN}f8"),
                    ],
                ),
            ),
        ],
    )
    def test_to_records_dtype_mi(self, df, kwargs, expected):
        # see GH#18146
        result = df.to_records(**kwargs)
        tm.assert_almost_equal(result, expected)

    def test_to_records_dict_like(self):
        # see GH#18146
        class DictLike:
            def __init__(self, **kwargs) -> None:
                self.d = kwargs.copy()

            def __getitem__(self, key):
                return self.d.__getitem__(key)

            def __contains__(self, key) -> bool:
                return key in self.d

            def keys(self):
                return self.d.keys()

        df = DataFrame({"A": [1, 2], "B": [0.2, 1.5], "C": ["a", "bc"]})

        dtype_mappings = {
            "column_dtypes": DictLike(A=np.int8, B=np.float32),
            "index_dtypes": f"{tm.ENDIAN}U2",
        }

        result = df.to_records(**dtype_mappings)
        expected = np.rec.array(
            [("0", "1", "0.2", "a"), ("1", "2", "1.5", "bc")],
            dtype=[
                ("index", f"{tm.ENDIAN}U2"),
                ("A", "i1"),
                ("B", f"{tm.ENDIAN}f4"),
                ("C", "O"),
            ],
        )
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("tz", ["UTC", "GMT", "US/Eastern"])
    def test_to_records_datetimeindex_with_tz(self, tz):
        # GH#13937
        dr = date_range("2016-01-01", periods=10, freq="s", tz=tz)

        df = DataFrame({"datetime": dr}, index=dr)

        expected = df.to_records()
        result = df.tz_convert("UTC").to_records()

        # both converted to UTC, so they are equal
        tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\tests\test_trigsimp.py ===
from itertools import product
from sympy.core.function import (Subs, count_ops, diff, expand)
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.functions.elementary.trigonometric import (asec, acsc)
from sympy.functions.elementary.trigonometric import (acot, atan)
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import (exptrigsimp, trigsimp)

from sympy.testing.pytest import XFAIL

from sympy.abc import x, y



def test_trigsimp1():
    x, y = symbols('x,y')

    assert trigsimp(1 - sin(x)**2) == cos(x)**2
    assert trigsimp(1 - cos(x)**2) == sin(x)**2
    assert trigsimp(sin(x)**2 + cos(x)**2) == 1
    assert trigsimp(1 + tan(x)**2) == 1/cos(x)**2
    assert trigsimp(1/cos(x)**2 - 1) == tan(x)**2
    assert trigsimp(1/cos(x)**2 - tan(x)**2) == 1
    assert trigsimp(1 + cot(x)**2) == 1/sin(x)**2
    assert trigsimp(1/sin(x)**2 - 1) == 1/tan(x)**2
    assert trigsimp(1/sin(x)**2 - cot(x)**2) == 1

    assert trigsimp(5*cos(x)**2 + 5*sin(x)**2) == 5
    assert trigsimp(5*cos(x/2)**2 + 2*sin(x/2)**2) == 3*cos(x)/2 + Rational(7, 2)

    assert trigsimp(sin(x)/cos(x)) == tan(x)
    assert trigsimp(2*tan(x)*cos(x)) == 2*sin(x)
    assert trigsimp(cot(x)**3*sin(x)**3) == cos(x)**3
    assert trigsimp(y*tan(x)**2/sin(x)**2) == y/cos(x)**2
    assert trigsimp(cot(x)/cos(x)) == 1/sin(x)

    assert trigsimp(sin(x + y) + sin(x - y)) == 2*sin(x)*cos(y)
    assert trigsimp(sin(x + y) - sin(x - y)) == 2*sin(y)*cos(x)
    assert trigsimp(cos(x + y) + cos(x - y)) == 2*cos(x)*cos(y)
    assert trigsimp(cos(x + y) - cos(x - y)) == -2*sin(x)*sin(y)
    assert trigsimp(tan(x + y) - tan(x)/(1 - tan(x)*tan(y))) == \
        sin(y)/(-sin(y)*tan(x) + cos(y))  # -tan(y)/(tan(x)*tan(y) - 1)

    assert trigsimp(sinh(x + y) + sinh(x - y)) == 2*sinh(x)*cosh(y)
    assert trigsimp(sinh(x + y) - sinh(x - y)) == 2*sinh(y)*cosh(x)
    assert trigsimp(cosh(x + y) + cosh(x - y)) == 2*cosh(x)*cosh(y)
    assert trigsimp(cosh(x + y) - cosh(x - y)) == 2*sinh(x)*sinh(y)
    assert trigsimp(tanh(x + y) - tanh(x)/(1 + tanh(x)*tanh(y))) == \
        sinh(y)/(sinh(y)*tanh(x) + cosh(y))

    assert trigsimp(cos(0.12345)**2 + sin(0.12345)**2) == 1.0
    e = 2*sin(x)**2 + 2*cos(x)**2
    assert trigsimp(log(e)) == log(2)


def test_trigsimp1a():
    assert trigsimp(sin(2)**2*cos(3)*exp(2)/cos(2)**2) == tan(2)**2*cos(3)*exp(2)
    assert trigsimp(tan(2)**2*cos(3)*exp(2)*cos(2)**2) == sin(2)**2*cos(3)*exp(2)
    assert trigsimp(cot(2)*cos(3)*exp(2)*sin(2)) == cos(3)*exp(2)*cos(2)
    assert trigsimp(tan(2)*cos(3)*exp(2)/sin(2)) == cos(3)*exp(2)/cos(2)
    assert trigsimp(cot(2)*cos(3)*exp(2)/cos(2)) == cos(3)*exp(2)/sin(2)
    assert trigsimp(cot(2)*cos(3)*exp(2)*tan(2)) == cos(3)*exp(2)
    assert trigsimp(sinh(2)*cos(3)*exp(2)/cosh(2)) == tanh(2)*cos(3)*exp(2)
    assert trigsimp(tanh(2)*cos(3)*exp(2)*cosh(2)) == sinh(2)*cos(3)*exp(2)
    assert trigsimp(coth(2)*cos(3)*exp(2)*sinh(2)) == cosh(2)*cos(3)*exp(2)
    assert trigsimp(tanh(2)*cos(3)*exp(2)/sinh(2)) == cos(3)*exp(2)/cosh(2)
    assert trigsimp(coth(2)*cos(3)*exp(2)/cosh(2)) == cos(3)*exp(2)/sinh(2)
    assert trigsimp(coth(2)*cos(3)*exp(2)*tanh(2)) == cos(3)*exp(2)


def test_trigsimp2():
    x, y = symbols('x,y')
    assert trigsimp(cos(x)**2*sin(y)**2 + cos(x)**2*cos(y)**2 + sin(x)**2,
            recursive=True) == 1
    assert trigsimp(sin(x)**2*sin(y)**2 + sin(x)**2*cos(y)**2 + cos(x)**2,
            recursive=True) == 1
    assert trigsimp(
        Subs(x, x, sin(y)**2 + cos(y)**2)) == Subs(x, x, 1)


def test_issue_4373():
    x = Symbol("x")
    assert abs(trigsimp(2.0*sin(x)**2 + 2.0*cos(x)**2) - 2.0) < 1e-10


def test_trigsimp3():
    x, y = symbols('x,y')
    assert trigsimp(sin(x)/cos(x)) == tan(x)
    assert trigsimp(sin(x)**2/cos(x)**2) == tan(x)**2
    assert trigsimp(sin(x)**3/cos(x)**3) == tan(x)**3
    assert trigsimp(sin(x)**10/cos(x)**10) == tan(x)**10

    assert trigsimp(cos(x)/sin(x)) == 1/tan(x)
    assert trigsimp(cos(x)**2/sin(x)**2) == 1/tan(x)**2
    assert trigsimp(cos(x)**10/sin(x)**10) == 1/tan(x)**10

    assert trigsimp(tan(x)) == trigsimp(sin(x)/cos(x))


def test_issue_4661():
    a, x, y = symbols('a x y')
    eq = -4*sin(x)**4 + 4*cos(x)**4 - 8*cos(x)**2
    assert trigsimp(eq) == -4
    n = sin(x)**6 + 4*sin(x)**4*cos(x)**2 + 5*sin(x)**2*cos(x)**4 + 2*cos(x)**6
    d = -sin(x)**2 - 2*cos(x)**2
    assert simplify(n/d) == -1
    assert trigsimp(-2*cos(x)**2 + cos(x)**4 - sin(x)**4) == -1
    eq = (- sin(x)**3/4)*cos(x) + (cos(x)**3/4)*sin(x) - sin(2*x)*cos(2*x)/8
    assert trigsimp(eq) == 0


def test_issue_4494():
    a, b = symbols('a b')
    eq = sin(a)**2*sin(b)**2 + cos(a)**2*cos(b)**2*tan(a)**2 + cos(a)**2
    assert trigsimp(eq) == 1


def test_issue_5948():
    a, x, y = symbols('a x y')
    assert trigsimp(diff(integrate(cos(x)/sin(x)**7, x), x)) == \
           cos(x)/sin(x)**7


def test_issue_4775():
    a, x, y = symbols('a x y')
    assert trigsimp(sin(x)*cos(y)+cos(x)*sin(y)) == sin(x + y)
    assert trigsimp(sin(x)*cos(y)+cos(x)*sin(y)+3) == sin(x + y) + 3


def test_issue_4280():
    a, x, y = symbols('a x y')
    assert trigsimp(cos(x)**2 + cos(y)**2*sin(x)**2 + sin(y)**2*sin(x)**2) == 1
    assert trigsimp(a**2*sin(x)**2 + a**2*cos(y)**2*cos(x)**2 + a**2*cos(x)**2*sin(y)**2) == a**2
    assert trigsimp(a**2*cos(y)**2*sin(x)**2 + a**2*sin(y)**2*sin(x)**2) == a**2*sin(x)**2


def test_issue_3210():
    eqs = (sin(2)*cos(3) + sin(3)*cos(2),
        -sin(2)*sin(3) + cos(2)*cos(3),
        sin(2)*cos(3) - sin(3)*cos(2),
        sin(2)*sin(3) + cos(2)*cos(3),
        sin(2)*sin(3) + cos(2)*cos(3) + cos(2),
        sinh(2)*cosh(3) + sinh(3)*cosh(2),
        sinh(2)*sinh(3) + cosh(2)*cosh(3),
        )
    assert [trigsimp(e) for e in eqs] == [
        sin(5),
        cos(5),
        -sin(1),
        cos(1),
        cos(1) + cos(2),
        sinh(5),
        cosh(5),
        ]


def test_trigsimp_issues():
    a, x, y = symbols('a x y')

    # issue 4625 - factor_terms works, too
    assert trigsimp(sin(x)**3 + cos(x)**2*sin(x)) == sin(x)

    # issue 5948
    assert trigsimp(diff(integrate(cos(x)/sin(x)**3, x), x)) == \
        cos(x)/sin(x)**3
    assert trigsimp(diff(integrate(sin(x)/cos(x)**3, x), x)) == \
        sin(x)/cos(x)**3

    # check integer exponents
    e = sin(x)**y/cos(x)**y
    assert trigsimp(e) == e
    assert trigsimp(e.subs(y, 2)) == tan(x)**2
    assert trigsimp(e.subs(x, 1)) == tan(1)**y

    # check for multiple patterns
    assert (cos(x)**2/sin(x)**2*cos(y)**2/sin(y)**2).trigsimp() == \
        1/tan(x)**2/tan(y)**2
    assert trigsimp(cos(x)/sin(x)*cos(x+y)/sin(x+y)) == \
        1/(tan(x)*tan(x + y))

    eq = cos(2)*(cos(3) + 1)**2/(cos(3) - 1)**2
    assert trigsimp(eq) == eq.factor()  # factor makes denom (-1 + cos(3))**2
    assert trigsimp(cos(2)*(cos(3) + 1)**2*(cos(3) - 1)**2) == \
        cos(2)*sin(3)**4

    # issue 6789; this generates an expression that formerly caused
    # trigsimp to hang
    assert cot(x).equals(tan(x)) is False

    # nan or the unchanged expression is ok, but not sin(1)
    z = cos(x)**2 + sin(x)**2 - 1
    z1 = tan(x)**2 - 1/cot(x)**2
    n = (1 + z1/z)
    assert trigsimp(sin(n)) != sin(1)
    eq = x*(n - 1) - x*n
    assert trigsimp(eq) is S.NaN
    assert trigsimp(eq, recursive=True) is S.NaN
    assert trigsimp(1).is_Integer

    assert trigsimp(-sin(x)**4 - 2*sin(x)**2*cos(x)**2 - cos(x)**4) == -1


def test_trigsimp_issue_2515():
    x = Symbol('x')
    assert trigsimp(x*cos(x)*tan(x)) == x*sin(x)
    assert trigsimp(-sin(x) + cos(x)*tan(x)) == 0


def test_trigsimp_issue_3826():
    assert trigsimp(tan(2*x).expand(trig=True)) == tan(2*x)


def test_trigsimp_issue_4032():
    n = Symbol('n', integer=True, positive=True)
    assert trigsimp(2**(n/2)*cos(pi*n/4)/2 + 2**(n - 1)/2) == \
        2**(n/2)*cos(pi*n/4)/2 + 2**n/4


def test_trigsimp_issue_7761():
    assert trigsimp(cosh(pi/4)) == cosh(pi/4)


def test_trigsimp_noncommutative():
    x, y = symbols('x,y')
    A, B = symbols('A,B', commutative=False)

    assert trigsimp(A - A*sin(x)**2) == A*cos(x)**2
    assert trigsimp(A - A*cos(x)**2) == A*sin(x)**2
    assert trigsimp(A*sin(x)**2 + A*cos(x)**2) == A
    assert trigsimp(A + A*tan(x)**2) == A/cos(x)**2
    assert trigsimp(A/cos(x)**2 - A) == A*tan(x)**2
    assert trigsimp(A/cos(x)**2 - A*tan(x)**2) == A
    assert trigsimp(A + A*cot(x)**2) == A/sin(x)**2
    assert trigsimp(A/sin(x)**2 - A) == A/tan(x)**2
    assert trigsimp(A/sin(x)**2 - A*cot(x)**2) == A

    assert trigsimp(y*A*cos(x)**2 + y*A*sin(x)**2) == y*A

    assert trigsimp(A*sin(x)/cos(x)) == A*tan(x)
    assert trigsimp(A*tan(x)*cos(x)) == A*sin(x)
    assert trigsimp(A*cot(x)**3*sin(x)**3) == A*cos(x)**3
    assert trigsimp(y*A*tan(x)**2/sin(x)**2) == y*A/cos(x)**2
    assert trigsimp(A*cot(x)/cos(x)) == A/sin(x)

    assert trigsimp(A*sin(x + y) + A*sin(x - y)) == 2*A*sin(x)*cos(y)
    assert trigsimp(A*sin(x + y) - A*sin(x - y)) == 2*A*sin(y)*cos(x)
    assert trigsimp(A*cos(x + y) + A*cos(x - y)) == 2*A*cos(x)*cos(y)
    assert trigsimp(A*cos(x + y) - A*cos(x - y)) == -2*A*sin(x)*sin(y)

    assert trigsimp(A*sinh(x + y) + A*sinh(x - y)) == 2*A*sinh(x)*cosh(y)
    assert trigsimp(A*sinh(x + y) - A*sinh(x - y)) == 2*A*sinh(y)*cosh(x)
    assert trigsimp(A*cosh(x + y) + A*cosh(x - y)) == 2*A*cosh(x)*cosh(y)
    assert trigsimp(A*cosh(x + y) - A*cosh(x - y)) == 2*A*sinh(x)*sinh(y)

    assert trigsimp(A*cos(0.12345)**2 + A*sin(0.12345)**2) == 1.0*A


def test_hyperbolic_simp():
    x, y = symbols('x,y')

    assert trigsimp(sinh(x)**2 + 1) == cosh(x)**2
    assert trigsimp(cosh(x)**2 - 1) == sinh(x)**2
    assert trigsimp(cosh(x)**2 - sinh(x)**2) == 1
    assert trigsimp(1 - tanh(x)**2) == 1/cosh(x)**2
    assert trigsimp(1 - 1/cosh(x)**2) == tanh(x)**2
    assert trigsimp(tanh(x)**2 + 1/cosh(x)**2) == 1
    assert trigsimp(coth(x)**2 - 1) == 1/sinh(x)**2
    assert trigsimp(1/sinh(x)**2 + 1) == 1/tanh(x)**2
    assert trigsimp(coth(x)**2 - 1/sinh(x)**2) == 1

    assert trigsimp(5*cosh(x)**2 - 5*sinh(x)**2) == 5
    assert trigsimp(5*cosh(x/2)**2 - 2*sinh(x/2)**2) == 3*cosh(x)/2 + Rational(7, 2)

    assert trigsimp(sinh(x)/cosh(x)) == tanh(x)
    assert trigsimp(tanh(x)) == trigsimp(sinh(x)/cosh(x))
    assert trigsimp(cosh(x)/sinh(x)) == 1/tanh(x)
    assert trigsimp(2*tanh(x)*cosh(x)) == 2*sinh(x)
    assert trigsimp(coth(x)**3*sinh(x)**3) == cosh(x)**3
    assert trigsimp(y*tanh(x)**2/sinh(x)**2) == y/cosh(x)**2
    assert trigsimp(coth(x)/cosh(x)) == 1/sinh(x)

    for a in (pi/6*I, pi/4*I, pi/3*I):
        assert trigsimp(sinh(a)*cosh(x) + cosh(a)*sinh(x)) == sinh(x + a)
        assert trigsimp(-sinh(a)*cosh(x) + cosh(a)*sinh(x)) == sinh(x - a)

    e = 2*cosh(x)**2 - 2*sinh(x)**2
    assert trigsimp(log(e)) == log(2)

    # issue 19535:
    assert trigsimp(sqrt(cosh(x)**2 - 1)) == sqrt(sinh(x)**2)

    assert trigsimp(cosh(x)**2*cosh(y)**2 - cosh(x)**2*sinh(y)**2 - sinh(x)**2,
            recursive=True) == 1
    assert trigsimp(sinh(x)**2*sinh(y)**2 - sinh(x)**2*cosh(y)**2 + cosh(x)**2,
            recursive=True) == 1

    assert abs(trigsimp(2.0*cosh(x)**2 - 2.0*sinh(x)**2) - 2.0) < 1e-10

    assert trigsimp(sinh(x)**2/cosh(x)**2) == tanh(x)**2
    assert trigsimp(sinh(x)**3/cosh(x)**3) == tanh(x)**3
    assert trigsimp(sinh(x)**10/cosh(x)**10) == tanh(x)**10
    assert trigsimp(cosh(x)**3/sinh(x)**3) == 1/tanh(x)**3

    assert trigsimp(cosh(x)/sinh(x)) == 1/tanh(x)
    assert trigsimp(cosh(x)**2/sinh(x)**2) == 1/tanh(x)**2
    assert trigsimp(cosh(x)**10/sinh(x)**10) == 1/tanh(x)**10

    assert trigsimp(x*cosh(x)*tanh(x)) == x*sinh(x)
    assert trigsimp(-sinh(x) + cosh(x)*tanh(x)) == 0

    assert tan(x) != 1/cot(x)  # cot doesn't auto-simplify

    assert trigsimp(tan(x) - 1/cot(x)) == 0
    assert trigsimp(3*tanh(x)**7 - 2/coth(x)**7) == tanh(x)**7


def test_trigsimp_groebner():
    from sympy.simplify.trigsimp import trigsimp_groebner

    c = cos(x)
    s = sin(x)
    ex = (4*s*c + 12*s + 5*c**3 + 21*c**2 + 23*c + 15)/(
        -s*c**2 + 2*s*c + 15*s + 7*c**3 + 31*c**2 + 37*c + 21)
    resnum = (5*s - 5*c + 1)
    resdenom = (8*s - 6*c)
    results = [resnum/resdenom, (-resnum)/(-resdenom)]
    assert trigsimp_groebner(ex) in results
    assert trigsimp_groebner(s/c, hints=[tan]) == tan(x)
    assert trigsimp_groebner(c*s) == c*s
    assert trigsimp((-s + 1)/c + c/(-s + 1),
                    method='groebner') == 2/c
    assert trigsimp((-s + 1)/c + c/(-s + 1),
                    method='groebner', polynomial=True) == 2/c

    # Test quick=False works
    assert trigsimp_groebner(ex, hints=[2]) in results
    assert trigsimp_groebner(ex, hints=[int(2)]) in results

    # test "I"
    assert trigsimp_groebner(sin(I*x)/cos(I*x), hints=[tanh]) == I*tanh(x)

    # test hyperbolic / sums
    assert trigsimp_groebner((tanh(x)+tanh(y))/(1+tanh(x)*tanh(y)),
                             hints=[(tanh, x, y)]) == tanh(x + y)


def test_issue_2827_trigsimp_methods():
    measure1 = lambda expr: len(str(expr))
    measure2 = lambda expr: -count_ops(expr)
                                       # Return the most complicated result
    expr = (x + 1)/(x + sin(x)**2 + cos(x)**2)
    ans = Matrix([1])
    M = Matrix([expr])
    assert trigsimp(M, method='fu', measure=measure1) == ans
    assert trigsimp(M, method='fu', measure=measure2) != ans
    # all methods should work with Basic expressions even if they
    # aren't Expr
    M = Matrix.eye(1)
    assert all(trigsimp(M, method=m) == M for m in
        'fu matching groebner old'.split())
    # watch for E in exptrigsimp, not only exp()
    eq = 1/sqrt(E) + E
    assert exptrigsimp(eq) == eq

def test_issue_15129_trigsimp_methods():
    t1 = Matrix([sin(Rational(1, 50)), cos(Rational(1, 50)), 0])
    t2 = Matrix([sin(Rational(1, 25)), cos(Rational(1, 25)), 0])
    t3 = Matrix([cos(Rational(1, 25)), sin(Rational(1, 25)), 0])
    r1 = t1.dot(t2)
    r2 = t1.dot(t3)
    assert trigsimp(r1) == cos(Rational(1, 50))
    assert trigsimp(r2) == sin(Rational(3, 50))

def test_exptrigsimp():
    def valid(a, b):
        from sympy.core.random import verify_numerically as tn
        if not (tn(a, b) and a == b):
            return False
        return True

    assert exptrigsimp(exp(x) + exp(-x)) == 2*cosh(x)
    assert exptrigsimp(exp(x) - exp(-x)) == 2*sinh(x)
    assert exptrigsimp((2*exp(x)-2*exp(-x))/(exp(x)+exp(-x))) == 2*tanh(x)
    assert exptrigsimp((2*exp(2*x)-2)/(exp(2*x)+1)) == 2*tanh(x)
    e = [cos(x) + I*sin(x), cos(x) - I*sin(x),
         cosh(x) - sinh(x), cosh(x) + sinh(x)]
    ok = [exp(I*x), exp(-I*x), exp(-x), exp(x)]
    assert all(valid(i, j) for i, j in zip(
        [exptrigsimp(ei) for ei in e], ok))

    ue = [cos(x) + sin(x), cos(x) - sin(x),
          cosh(x) + I*sinh(x), cosh(x) - I*sinh(x)]
    assert [exptrigsimp(ei) == ei for ei in ue]

    res = []
    ok = [y*tanh(1), 1/(y*tanh(1)), I*y*tan(1), -I/(y*tan(1)),
        y*tanh(x), 1/(y*tanh(x)), I*y*tan(x), -I/(y*tan(x)),
        y*tanh(1 + I), 1/(y*tanh(1 + I))]
    for a in (1, I, x, I*x, 1 + I):
        w = exp(a)
        eq = y*(w - 1/w)/(w + 1/w)
        res.append(simplify(eq))
        res.append(simplify(1/eq))
    assert all(valid(i, j) for i, j in zip(res, ok))

    for a in range(1, 3):
        w = exp(a)
        e = w + 1/w
        s = simplify(e)
        assert s == exptrigsimp(e)
        assert valid(s, 2*cosh(a))
        e = w - 1/w
        s = simplify(e)
        assert s == exptrigsimp(e)
        assert valid(s, 2*sinh(a))

def test_exptrigsimp_noncommutative():
    a,b = symbols('a b', commutative=False)
    x = Symbol('x', commutative=True)
    assert exp(a + x) == exptrigsimp(exp(a)*exp(x))
    p = exp(a)*exp(b) - exp(b)*exp(a)
    assert p == exptrigsimp(p) != 0

def test_powsimp_on_numbers():
    assert 2**(Rational(1, 3) - 2) == 2**Rational(1, 3)/4


@XFAIL
def test_issue_6811_fail():
    # from doc/src/modules/physics/mechanics/examples.rst, the current `eq`
    # at Line 576 (in different variables) was formerly the equivalent and
    # shorter expression given below...it would be nice to get the short one
    # back again
    xp, y, x, z = symbols('xp, y, x, z')
    eq = 4*(-19*sin(x)*y + 5*sin(3*x)*y + 15*cos(2*x)*z - 21*z)*xp/(9*cos(x) - 5*cos(3*x))
    assert trigsimp(eq) == -2*(2*cos(x)*tan(x)*y + 3*z)*xp/cos(x)


def test_Piecewise():
    e1 = x*(x + y) - y*(x + y)
    e2 = sin(x)**2 + cos(x)**2
    e3 = expand((x + y)*y/x)
    # s1 = simplify(e1)
    s2 = simplify(e2)
    # s3 = simplify(e3)

    # trigsimp tries not to touch non-trig containing args
    assert trigsimp(Piecewise((e1, e3 < e2), (e3, True))) == \
        Piecewise((e1, e3 < s2), (e3, True))


def test_issue_21594():
    assert simplify(exp(Rational(1,2)) + exp(Rational(-1,2))) == cosh(S.Half)*2


def test_trigsimp_old():
    x, y = symbols('x,y')

    assert trigsimp(1 - sin(x)**2, old=True) == cos(x)**2
    assert trigsimp(1 - cos(x)**2, old=True) == sin(x)**2
    assert trigsimp(sin(x)**2 + cos(x)**2, old=True) == 1
    assert trigsimp(1 + tan(x)**2, old=True) == 1/cos(x)**2
    assert trigsimp(1/cos(x)**2 - 1, old=True) == tan(x)**2
    assert trigsimp(1/cos(x)**2 - tan(x)**2, old=True) == 1
    assert trigsimp(1 + cot(x)**2, old=True) == 1/sin(x)**2
    assert trigsimp(1/sin(x)**2 - cot(x)**2, old=True) == 1

    assert trigsimp(5*cos(x)**2 + 5*sin(x)**2, old=True) == 5

    assert trigsimp(sin(x)/cos(x), old=True) == tan(x)
    assert trigsimp(2*tan(x)*cos(x), old=True) == 2*sin(x)
    assert trigsimp(cot(x)**3*sin(x)**3, old=True) == cos(x)**3
    assert trigsimp(y*tan(x)**2/sin(x)**2, old=True) == y/cos(x)**2
    assert trigsimp(cot(x)/cos(x), old=True) == 1/sin(x)

    assert trigsimp(sin(x + y) + sin(x - y), old=True) == 2*sin(x)*cos(y)
    assert trigsimp(sin(x + y) - sin(x - y), old=True) == 2*sin(y)*cos(x)
    assert trigsimp(cos(x + y) + cos(x - y), old=True) == 2*cos(x)*cos(y)
    assert trigsimp(cos(x + y) - cos(x - y), old=True) == -2*sin(x)*sin(y)

    assert trigsimp(sinh(x + y) + sinh(x - y), old=True) == 2*sinh(x)*cosh(y)
    assert trigsimp(sinh(x + y) - sinh(x - y), old=True) == 2*sinh(y)*cosh(x)
    assert trigsimp(cosh(x + y) + cosh(x - y), old=True) == 2*cosh(x)*cosh(y)
    assert trigsimp(cosh(x + y) - cosh(x - y), old=True) == 2*sinh(x)*sinh(y)

    assert trigsimp(cos(0.12345)**2 + sin(0.12345)**2, old=True) == 1.0

    assert trigsimp(sin(x)/cos(x), old=True, method='combined') == tan(x)
    assert trigsimp(sin(x)/cos(x), old=True, method='groebner') == sin(x)/cos(x)
    assert trigsimp(sin(x)/cos(x), old=True, method='groebner', hints=[tan]) == tan(x)

    assert trigsimp(1-sin(sin(x)**2+cos(x)**2)**2, old=True, deep=True) == cos(1)**2


def test_trigsimp_inverse():
    alpha = symbols('alpha')
    s, c = sin(alpha), cos(alpha)

    for finv in [asin, acos, asec, acsc, atan, acot]:
        f = finv.inverse(None)
        assert alpha == trigsimp(finv(f(alpha)), inverse=True)

    # test atan2(cos, sin), atan2(sin, cos), etc...
    for a, b in [[c, s], [s, c]]:
        for i, j in product([-1, 1], repeat=2):
            angle = atan2(i*b, j*a)
            angle_inverted = trigsimp(angle, inverse=True)
            assert angle_inverted != angle  # assures simplification happened
            assert sin(angle_inverted) == trigsimp(sin(angle))
            assert cos(angle_inverted) == trigsimp(cos(angle))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\test_array.py ===
import datetime
import decimal
import re

import numpy as np
import pytest
import pytz

from pandas._config import using_string_dtype

import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import register_extension_dtype
from pandas.arrays import (
    BooleanArray,
    DatetimeArray,
    FloatingArray,
    IntegerArray,
    IntervalArray,
    SparseArray,
    TimedeltaArray,
)
from pandas.core.arrays import (
    NumpyExtensionArray,
    period_array,
)
from pandas.tests.extension.decimal import (
    DecimalArray,
    DecimalDtype,
    to_decimal,
)


@pytest.mark.parametrize("dtype_unit", ["M8[h]", "M8[m]", "m8[h]", "M8[m]"])
def test_dt64_array(dtype_unit):
    # PR 53817
    dtype_var = np.dtype(dtype_unit)
    msg = (
        r"datetime64 and timedelta64 dtype resolutions other than "
        r"'s', 'ms', 'us', and 'ns' are deprecated. "
        r"In future releases passing unsupported resolutions will "
        r"raise an exception."
    )
    with tm.assert_produces_warning(FutureWarning, match=re.escape(msg)):
        pd.array([], dtype=dtype_var)


@pytest.mark.parametrize(
    "data, dtype, expected",
    [
        # Basic NumPy defaults.
        ([], None, FloatingArray._from_sequence([], dtype="Float64")),
        ([1, 2], None, IntegerArray._from_sequence([1, 2], dtype="Int64")),
        ([1, 2], object, NumpyExtensionArray(np.array([1, 2], dtype=object))),
        (
            [1, 2],
            np.dtype("float32"),
            NumpyExtensionArray(np.array([1.0, 2.0], dtype=np.dtype("float32"))),
        ),
        (
            np.array([], dtype=object),
            None,
            NumpyExtensionArray(np.array([], dtype=object)),
        ),
        (
            np.array([1, 2], dtype="int64"),
            None,
            IntegerArray._from_sequence([1, 2], dtype="Int64"),
        ),
        (
            np.array([1.0, 2.0], dtype="float64"),
            None,
            FloatingArray._from_sequence([1.0, 2.0], dtype="Float64"),
        ),
        # String alias passes through to NumPy
        ([1, 2], "float32", NumpyExtensionArray(np.array([1, 2], dtype="float32"))),
        ([1, 2], "int64", NumpyExtensionArray(np.array([1, 2], dtype=np.int64))),
        # GH#44715 FloatingArray does not support float16, so fall
        #  back to NumpyExtensionArray
        (
            np.array([1, 2], dtype=np.float16),
            None,
            NumpyExtensionArray(np.array([1, 2], dtype=np.float16)),
        ),
        # idempotency with e.g. pd.array(pd.array([1, 2], dtype="int64"))
        (
            NumpyExtensionArray(np.array([1, 2], dtype=np.int32)),
            None,
            NumpyExtensionArray(np.array([1, 2], dtype=np.int32)),
        ),
        # Period alias
        (
            [pd.Period("2000", "D"), pd.Period("2001", "D")],
            "Period[D]",
            period_array(["2000", "2001"], freq="D"),
        ),
        # Period dtype
        (
            [pd.Period("2000", "D")],
            pd.PeriodDtype("D"),
            period_array(["2000"], freq="D"),
        ),
        # Datetime (naive)
        (
            [1, 2],
            np.dtype("datetime64[ns]"),
            DatetimeArray._from_sequence(
                np.array([1, 2], dtype="M8[ns]"), dtype="M8[ns]"
            ),
        ),
        (
            [1, 2],
            np.dtype("datetime64[s]"),
            DatetimeArray._from_sequence(
                np.array([1, 2], dtype="M8[s]"), dtype="M8[s]"
            ),
        ),
        (
            np.array([1, 2], dtype="datetime64[ns]"),
            None,
            DatetimeArray._from_sequence(
                np.array([1, 2], dtype="M8[ns]"), dtype="M8[ns]"
            ),
        ),
        (
            pd.DatetimeIndex(["2000", "2001"]),
            np.dtype("datetime64[ns]"),
            DatetimeArray._from_sequence(["2000", "2001"], dtype="M8[ns]"),
        ),
        (
            pd.DatetimeIndex(["2000", "2001"]),
            None,
            DatetimeArray._from_sequence(["2000", "2001"], dtype="M8[ns]"),
        ),
        (
            ["2000", "2001"],
            np.dtype("datetime64[ns]"),
            DatetimeArray._from_sequence(["2000", "2001"], dtype="M8[ns]"),
        ),
        # Datetime (tz-aware)
        (
            ["2000", "2001"],
            pd.DatetimeTZDtype(tz="CET"),
            DatetimeArray._from_sequence(
                ["2000", "2001"], dtype=pd.DatetimeTZDtype(tz="CET")
            ),
        ),
        # Timedelta
        (
            ["1h", "2h"],
            np.dtype("timedelta64[ns]"),
            TimedeltaArray._from_sequence(["1h", "2h"], dtype="m8[ns]"),
        ),
        (
            pd.TimedeltaIndex(["1h", "2h"]),
            np.dtype("timedelta64[ns]"),
            TimedeltaArray._from_sequence(["1h", "2h"], dtype="m8[ns]"),
        ),
        (
            np.array([1, 2], dtype="m8[s]"),
            np.dtype("timedelta64[s]"),
            TimedeltaArray._from_sequence(
                np.array([1, 2], dtype="m8[s]"), dtype="m8[s]"
            ),
        ),
        (
            pd.TimedeltaIndex(["1h", "2h"]),
            None,
            TimedeltaArray._from_sequence(["1h", "2h"], dtype="m8[ns]"),
        ),
        (
            # preserve non-nano, i.e. don't cast to NumpyExtensionArray
            TimedeltaArray._simple_new(
                np.arange(5, dtype=np.int64).view("m8[s]"), dtype=np.dtype("m8[s]")
            ),
            None,
            TimedeltaArray._simple_new(
                np.arange(5, dtype=np.int64).view("m8[s]"), dtype=np.dtype("m8[s]")
            ),
        ),
        (
            # preserve non-nano, i.e. don't cast to NumpyExtensionArray
            TimedeltaArray._simple_new(
                np.arange(5, dtype=np.int64).view("m8[s]"), dtype=np.dtype("m8[s]")
            ),
            np.dtype("m8[s]"),
            TimedeltaArray._simple_new(
                np.arange(5, dtype=np.int64).view("m8[s]"), dtype=np.dtype("m8[s]")
            ),
        ),
        # Category
        (["a", "b"], "category", pd.Categorical(["a", "b"])),
        (
            ["a", "b"],
            pd.CategoricalDtype(None, ordered=True),
            pd.Categorical(["a", "b"], ordered=True),
        ),
        # Interval
        (
            [pd.Interval(1, 2), pd.Interval(3, 4)],
            "interval",
            IntervalArray.from_tuples([(1, 2), (3, 4)]),
        ),
        # Sparse
        ([0, 1], "Sparse[int64]", SparseArray([0, 1], dtype="int64")),
        # IntegerNA
        ([1, None], "Int16", pd.array([1, None], dtype="Int16")),
        (
            pd.Series([1, 2]),
            None,
            NumpyExtensionArray(np.array([1, 2], dtype=np.int64)),
        ),
        # String
        (
            ["a", None],
            "string",
            pd.StringDtype()
            .construct_array_type()
            ._from_sequence(["a", None], dtype=pd.StringDtype()),
        ),
        (
            ["a", None],
            "str",
            pd.StringDtype(na_value=np.nan)
            .construct_array_type()
            ._from_sequence(["a", None], dtype=pd.StringDtype(na_value=np.nan))
            if using_string_dtype()
            else NumpyExtensionArray(np.array(["a", "None"])),
        ),
        (
            ["a", None],
            pd.StringDtype(),
            pd.StringDtype()
            .construct_array_type()
            ._from_sequence(["a", None], dtype=pd.StringDtype()),
        ),
        (
            ["a", None],
            pd.StringDtype(na_value=np.nan),
            pd.StringDtype(na_value=np.nan)
            .construct_array_type()
            ._from_sequence(["a", None], dtype=pd.StringDtype(na_value=np.nan)),
        ),
        (
            # numpy array with string dtype
            np.array(["a", "b"], dtype=str),
            pd.StringDtype(),
            pd.StringDtype()
            .construct_array_type()
            ._from_sequence(["a", "b"], dtype=pd.StringDtype()),
        ),
        (
            # numpy array with string dtype
            np.array(["a", "b"], dtype=str),
            pd.StringDtype(na_value=np.nan),
            pd.StringDtype(na_value=np.nan)
            .construct_array_type()
            ._from_sequence(["a", "b"], dtype=pd.StringDtype(na_value=np.nan)),
        ),
        # Boolean
        (
            [True, None],
            "boolean",
            BooleanArray._from_sequence([True, None], dtype="boolean"),
        ),
        (
            [True, None],
            pd.BooleanDtype(),
            BooleanArray._from_sequence([True, None], dtype="boolean"),
        ),
        # Index
        (pd.Index([1, 2]), None, NumpyExtensionArray(np.array([1, 2], dtype=np.int64))),
        # Series[EA] returns the EA
        (
            pd.Series(pd.Categorical(["a", "b"], categories=["a", "b", "c"])),
            None,
            pd.Categorical(["a", "b"], categories=["a", "b", "c"]),
        ),
        # "3rd party" EAs work
        ([decimal.Decimal(0), decimal.Decimal(1)], "decimal", to_decimal([0, 1])),
        # pass an ExtensionArray, but a different dtype
        (
            period_array(["2000", "2001"], freq="D"),
            "category",
            pd.Categorical([pd.Period("2000", "D"), pd.Period("2001", "D")]),
        ),
    ],
)
def test_array(data, dtype, expected):
    result = pd.array(data, dtype=dtype)
    tm.assert_equal(result, expected)


def test_array_copy():
    a = np.array([1, 2])
    # default is to copy
    b = pd.array(a, dtype=a.dtype)
    assert not tm.shares_memory(a, b)

    # copy=True
    b = pd.array(a, dtype=a.dtype, copy=True)
    assert not tm.shares_memory(a, b)

    # copy=False
    b = pd.array(a, dtype=a.dtype, copy=False)
    assert tm.shares_memory(a, b)


cet = pytz.timezone("CET")


@pytest.mark.parametrize(
    "data, expected",
    [
        # period
        (
            [pd.Period("2000", "D"), pd.Period("2001", "D")],
            period_array(["2000", "2001"], freq="D"),
        ),
        # interval
        ([pd.Interval(0, 1), pd.Interval(1, 2)], IntervalArray.from_breaks([0, 1, 2])),
        # datetime
        (
            [pd.Timestamp("2000"), pd.Timestamp("2001")],
            DatetimeArray._from_sequence(["2000", "2001"], dtype="M8[ns]"),
        ),
        (
            [datetime.datetime(2000, 1, 1), datetime.datetime(2001, 1, 1)],
            DatetimeArray._from_sequence(["2000", "2001"], dtype="M8[ns]"),
        ),
        (
            np.array([1, 2], dtype="M8[ns]"),
            DatetimeArray._from_sequence(np.array([1, 2], dtype="M8[ns]")),
        ),
        (
            np.array([1, 2], dtype="M8[us]"),
            DatetimeArray._simple_new(
                np.array([1, 2], dtype="M8[us]"), dtype=np.dtype("M8[us]")
            ),
        ),
        # datetimetz
        (
            [pd.Timestamp("2000", tz="CET"), pd.Timestamp("2001", tz="CET")],
            DatetimeArray._from_sequence(
                ["2000", "2001"], dtype=pd.DatetimeTZDtype(tz="CET", unit="ns")
            ),
        ),
        (
            [
                datetime.datetime(2000, 1, 1, tzinfo=cet),
                datetime.datetime(2001, 1, 1, tzinfo=cet),
            ],
            DatetimeArray._from_sequence(
                ["2000", "2001"], dtype=pd.DatetimeTZDtype(tz=cet, unit="ns")
            ),
        ),
        # timedelta
        (
            [pd.Timedelta("1h"), pd.Timedelta("2h")],
            TimedeltaArray._from_sequence(["1h", "2h"], dtype="m8[ns]"),
        ),
        (
            np.array([1, 2], dtype="m8[ns]"),
            TimedeltaArray._from_sequence(np.array([1, 2], dtype="m8[ns]")),
        ),
        (
            np.array([1, 2], dtype="m8[us]"),
            TimedeltaArray._from_sequence(np.array([1, 2], dtype="m8[us]")),
        ),
        # integer
        ([1, 2], IntegerArray._from_sequence([1, 2], dtype="Int64")),
        ([1, None], IntegerArray._from_sequence([1, None], dtype="Int64")),
        ([1, pd.NA], IntegerArray._from_sequence([1, pd.NA], dtype="Int64")),
        ([1, np.nan], IntegerArray._from_sequence([1, np.nan], dtype="Int64")),
        # float
        ([0.1, 0.2], FloatingArray._from_sequence([0.1, 0.2], dtype="Float64")),
        ([0.1, None], FloatingArray._from_sequence([0.1, pd.NA], dtype="Float64")),
        ([0.1, np.nan], FloatingArray._from_sequence([0.1, pd.NA], dtype="Float64")),
        ([0.1, pd.NA], FloatingArray._from_sequence([0.1, pd.NA], dtype="Float64")),
        # integer-like float
        ([1.0, 2.0], FloatingArray._from_sequence([1.0, 2.0], dtype="Float64")),
        ([1.0, None], FloatingArray._from_sequence([1.0, pd.NA], dtype="Float64")),
        ([1.0, np.nan], FloatingArray._from_sequence([1.0, pd.NA], dtype="Float64")),
        ([1.0, pd.NA], FloatingArray._from_sequence([1.0, pd.NA], dtype="Float64")),
        # mixed-integer-float
        ([1, 2.0], FloatingArray._from_sequence([1.0, 2.0], dtype="Float64")),
        (
            [1, np.nan, 2.0],
            FloatingArray._from_sequence([1.0, None, 2.0], dtype="Float64"),
        ),
        # string
        (
            ["a", "b"],
            pd.StringDtype()
            .construct_array_type()
            ._from_sequence(["a", "b"], dtype=pd.StringDtype()),
        ),
        (
            ["a", None],
            pd.StringDtype()
            .construct_array_type()
            ._from_sequence(["a", None], dtype=pd.StringDtype()),
        ),
        (
            # numpy array with string dtype
            np.array(["a", "b"], dtype=str),
            pd.StringDtype()
            .construct_array_type()
            ._from_sequence(["a", "b"], dtype=pd.StringDtype()),
        ),
        # Boolean
        ([True, False], BooleanArray._from_sequence([True, False], dtype="boolean")),
        ([True, None], BooleanArray._from_sequence([True, None], dtype="boolean")),
    ],
)
def test_array_inference(data, expected):
    result = pd.array(data)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        # mix of frequencies
        [pd.Period("2000", "D"), pd.Period("2001", "Y")],
        # mix of closed
        [pd.Interval(0, 1, closed="left"), pd.Interval(1, 2, closed="right")],
        # Mix of timezones
        [pd.Timestamp("2000", tz="CET"), pd.Timestamp("2000", tz="UTC")],
        # Mix of tz-aware and tz-naive
        [pd.Timestamp("2000", tz="CET"), pd.Timestamp("2000")],
        np.array([pd.Timestamp("2000"), pd.Timestamp("2000", tz="CET")]),
    ],
)
def test_array_inference_fails(data):
    result = pd.array(data)
    expected = NumpyExtensionArray(np.array(data, dtype=object))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("data", [np.array(0)])
def test_nd_raises(data):
    with pytest.raises(ValueError, match="NumpyExtensionArray must be 1-dimensional"):
        pd.array(data, dtype="int64")


def test_scalar_raises():
    with pytest.raises(ValueError, match="Cannot pass scalar '1'"):
        pd.array(1)


def test_dataframe_raises():
    # GH#51167 don't accidentally cast to StringArray by doing inference on columns
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
    msg = "Cannot pass DataFrame to 'pandas.array'"
    with pytest.raises(TypeError, match=msg):
        pd.array(df)


def test_bounds_check():
    # GH21796
    with pytest.raises(
        TypeError, match=r"cannot safely cast non-equivalent int(32|64) to uint16"
    ):
        pd.array([-1, 2, 3], dtype="UInt16")


# ---------------------------------------------------------------------------
# A couple dummy classes to ensure that Series and Indexes are unboxed before
# getting to the EA classes.


@register_extension_dtype
class DecimalDtype2(DecimalDtype):
    name = "decimal2"

    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return DecimalArray2


class DecimalArray2(DecimalArray):
    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        if isinstance(scalars, (pd.Series, pd.Index)):
            raise TypeError("scalars should not be of type pd.Series or pd.Index")

        return super()._from_sequence(scalars, dtype=dtype, copy=copy)


def test_array_unboxes(index_or_series):
    box = index_or_series

    data = box([decimal.Decimal("1"), decimal.Decimal("2")])
    dtype = DecimalDtype2()
    # make sure it works
    with pytest.raises(
        TypeError, match="scalars should not be of type pd.Series or pd.Index"
    ):
        DecimalArray2._from_sequence(data, dtype=dtype)

    result = pd.array(data, dtype="decimal2")
    expected = DecimalArray2._from_sequence(data.values, dtype=dtype)
    tm.assert_equal(result, expected)


def test_array_to_numpy_na():
    # GH#40638
    arr = pd.array([pd.NA, 1], dtype="string[python]")
    result = arr.to_numpy(na_value=True, dtype=bool)
    expected = np.array([True, True])
    tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\test_base_indexer.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    concat,
    date_range,
)
import pandas._testing as tm
from pandas.api.indexers import (
    BaseIndexer,
    FixedForwardWindowIndexer,
)
from pandas.core.indexers.objects import (
    ExpandingIndexer,
    FixedWindowIndexer,
    VariableOffsetWindowIndexer,
)

from pandas.tseries.offsets import BusinessDay


def test_bad_get_window_bounds_signature():
    class BadIndexer(BaseIndexer):
        def get_window_bounds(self):
            return None

    indexer = BadIndexer()
    with pytest.raises(ValueError, match="BadIndexer does not implement"):
        Series(range(5)).rolling(indexer)


def test_expanding_indexer():
    s = Series(range(10))
    indexer = ExpandingIndexer()
    result = s.rolling(indexer).mean()
    expected = s.expanding().mean()
    tm.assert_series_equal(result, expected)


def test_indexer_constructor_arg():
    # Example found in computation.rst
    use_expanding = [True, False, True, False, True]
    df = DataFrame({"values": range(5)})

    class CustomIndexer(BaseIndexer):
        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            start = np.empty(num_values, dtype=np.int64)
            end = np.empty(num_values, dtype=np.int64)
            for i in range(num_values):
                if self.use_expanding[i]:
                    start[i] = 0
                    end[i] = i + 1
                else:
                    start[i] = i
                    end[i] = i + self.window_size
            return start, end

    indexer = CustomIndexer(window_size=1, use_expanding=use_expanding)
    result = df.rolling(indexer).sum()
    expected = DataFrame({"values": [0.0, 1.0, 3.0, 3.0, 10.0]})
    tm.assert_frame_equal(result, expected)


def test_indexer_accepts_rolling_args():
    df = DataFrame({"values": range(5)})

    class CustomIndexer(BaseIndexer):
        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            start = np.empty(num_values, dtype=np.int64)
            end = np.empty(num_values, dtype=np.int64)
            for i in range(num_values):
                if (
                    center
                    and min_periods == 1
                    and closed == "both"
                    and step == 1
                    and i == 2
                ):
                    start[i] = 0
                    end[i] = num_values
                else:
                    start[i] = i
                    end[i] = i + self.window_size
            return start, end

    indexer = CustomIndexer(window_size=1)
    result = df.rolling(
        indexer, center=True, min_periods=1, closed="both", step=1
    ).sum()
    expected = DataFrame({"values": [0.0, 1.0, 10.0, 3.0, 4.0]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func,np_func,expected,np_kwargs",
    [
        ("count", len, [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, np.nan], {}),
        ("min", np.min, [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 6.0, 7.0, 8.0, np.nan], {}),
        (
            "max",
            np.max,
            [2.0, 3.0, 4.0, 100.0, 100.0, 100.0, 8.0, 9.0, 9.0, np.nan],
            {},
        ),
        (
            "std",
            np.std,
            [
                1.0,
                1.0,
                1.0,
                55.71654452,
                54.85739087,
                53.9845657,
                1.0,
                1.0,
                0.70710678,
                np.nan,
            ],
            {"ddof": 1},
        ),
        (
            "var",
            np.var,
            [
                1.0,
                1.0,
                1.0,
                3104.333333,
                3009.333333,
                2914.333333,
                1.0,
                1.0,
                0.500000,
                np.nan,
            ],
            {"ddof": 1},
        ),
        (
            "median",
            np.median,
            [1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 7.0, 8.0, 8.5, np.nan],
            {},
        ),
    ],
)
def test_rolling_forward_window(
    frame_or_series, func, np_func, expected, np_kwargs, step
):
    # GH 32865
    values = np.arange(10.0)
    values[5] = 100.0

    indexer = FixedForwardWindowIndexer(window_size=3)

    match = "Forward-looking windows can't have center=True"
    with pytest.raises(ValueError, match=match):
        rolling = frame_or_series(values).rolling(window=indexer, center=True)
        getattr(rolling, func)()

    match = "Forward-looking windows don't support setting the closed argument"
    with pytest.raises(ValueError, match=match):
        rolling = frame_or_series(values).rolling(window=indexer, closed="right")
        getattr(rolling, func)()

    rolling = frame_or_series(values).rolling(window=indexer, min_periods=2, step=step)
    result = getattr(rolling, func)()

    # Check that the function output matches the explicitly provided array
    expected = frame_or_series(expected)[::step]
    tm.assert_equal(result, expected)

    # Check that the rolling function output matches applying an alternative
    # function to the rolling window object
    expected2 = frame_or_series(rolling.apply(lambda x: np_func(x, **np_kwargs)))
    tm.assert_equal(result, expected2)

    # Check that the function output matches applying an alternative function
    # if min_periods isn't specified
    # GH 39604: After count-min_periods deprecation, apply(lambda x: len(x))
    # is equivalent to count after setting min_periods=0
    min_periods = 0 if func == "count" else None
    rolling3 = frame_or_series(values).rolling(window=indexer, min_periods=min_periods)
    result3 = getattr(rolling3, func)()
    expected3 = frame_or_series(rolling3.apply(lambda x: np_func(x, **np_kwargs)))
    tm.assert_equal(result3, expected3)


def test_rolling_forward_skewness(frame_or_series, step):
    values = np.arange(10.0)
    values[5] = 100.0

    indexer = FixedForwardWindowIndexer(window_size=5)
    rolling = frame_or_series(values).rolling(window=indexer, min_periods=3, step=step)
    result = rolling.skew()

    expected = frame_or_series(
        [
            0.0,
            2.232396,
            2.229508,
            2.228340,
            2.229091,
            2.231989,
            0.0,
            0.0,
            np.nan,
            np.nan,
        ]
    )[::step]
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "func,expected",
    [
        ("cov", [2.0, 2.0, 2.0, 97.0, 2.0, -93.0, 2.0, 2.0, np.nan, np.nan]),
        (
            "corr",
            [
                1.0,
                1.0,
                1.0,
                0.8704775290207161,
                0.018229084250926637,
                -0.861357304646493,
                1.0,
                1.0,
                np.nan,
                np.nan,
            ],
        ),
    ],
)
def test_rolling_forward_cov_corr(func, expected):
    values1 = np.arange(10).reshape(-1, 1)
    values2 = values1 * 2
    values1[5, 0] = 100
    values = np.concatenate([values1, values2], axis=1)

    indexer = FixedForwardWindowIndexer(window_size=3)
    rolling = DataFrame(values).rolling(window=indexer, min_periods=3)
    # We are interested in checking only pairwise covariance / correlation
    result = getattr(rolling, func)().loc[(slice(None), 1), 0]
    result = result.reset_index(drop=True)
    expected = Series(expected).reset_index(drop=True)
    expected.name = result.name
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "closed,expected_data",
    [
        ["right", [0.0, 1.0, 2.0, 3.0, 7.0, 12.0, 6.0, 7.0, 8.0, 9.0]],
        ["left", [0.0, 0.0, 1.0, 2.0, 5.0, 9.0, 5.0, 6.0, 7.0, 8.0]],
    ],
)
def test_non_fixed_variable_window_indexer(closed, expected_data):
    index = date_range("2020", periods=10)
    df = DataFrame(range(10), index=index)
    offset = BusinessDay(1)
    indexer = VariableOffsetWindowIndexer(index=index, offset=offset)
    result = df.rolling(indexer, closed=closed).sum()
    expected = DataFrame(expected_data, index=index)
    tm.assert_frame_equal(result, expected)


def test_variableoffsetwindowindexer_not_dti():
    # GH 54379
    with pytest.raises(ValueError, match="index must be a DatetimeIndex."):
        VariableOffsetWindowIndexer(index="foo", offset=BusinessDay(1))


def test_variableoffsetwindowindexer_not_offset():
    # GH 54379
    idx = date_range("2020", periods=10)
    with pytest.raises(ValueError, match="offset must be a DateOffset-like object."):
        VariableOffsetWindowIndexer(index=idx, offset="foo")


def test_fixed_forward_indexer_count(step):
    # GH: 35579
    df = DataFrame({"b": [None, None, None, 7]})
    indexer = FixedForwardWindowIndexer(window_size=2)
    result = df.rolling(window=indexer, min_periods=0, step=step).count()
    expected = DataFrame({"b": [0.0, 0.0, 1.0, 1.0]})[::step]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("end_value", "values"), [(1, [0.0, 1, 1, 3, 2]), (-1, [0.0, 1, 0, 3, 1])]
)
@pytest.mark.parametrize(("func", "args"), [("median", []), ("quantile", [0.5])])
def test_indexer_quantile_sum(end_value, values, func, args):
    # GH 37153
    class CustomIndexer(BaseIndexer):
        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            start = np.empty(num_values, dtype=np.int64)
            end = np.empty(num_values, dtype=np.int64)
            for i in range(num_values):
                if self.use_expanding[i]:
                    start[i] = 0
                    end[i] = max(i + end_value, 1)
                else:
                    start[i] = i
                    end[i] = i + self.window_size
            return start, end

    use_expanding = [True, False, True, False, True]
    df = DataFrame({"values": range(5)})

    indexer = CustomIndexer(window_size=1, use_expanding=use_expanding)
    result = getattr(df.rolling(indexer), func)(*args)
    expected = DataFrame({"values": values})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "indexer_class", [FixedWindowIndexer, FixedForwardWindowIndexer, ExpandingIndexer]
)
@pytest.mark.parametrize("window_size", [1, 2, 12])
@pytest.mark.parametrize(
    "df_data",
    [
        {"a": [1, 1], "b": [0, 1]},
        {"a": [1, 2], "b": [0, 1]},
        {"a": [1] * 16, "b": [np.nan, 1, 2, np.nan] + list(range(4, 16))},
    ],
)
def test_indexers_are_reusable_after_groupby_rolling(
    indexer_class, window_size, df_data
):
    # GH 43267
    df = DataFrame(df_data)
    num_trials = 3
    indexer = indexer_class(window_size=window_size)
    original_window_size = indexer.window_size
    for i in range(num_trials):
        df.groupby("a")["b"].rolling(window=indexer, min_periods=1).mean()
        assert indexer.window_size == original_window_size


@pytest.mark.parametrize(
    "window_size, num_values, expected_start, expected_end",
    [
        (1, 1, [0], [1]),
        (1, 2, [0, 1], [1, 2]),
        (2, 1, [0], [1]),
        (2, 2, [0, 1], [2, 2]),
        (5, 12, range(12), list(range(5, 12)) + [12] * 5),
        (12, 5, range(5), [5] * 5),
        (0, 0, np.array([]), np.array([])),
        (1, 0, np.array([]), np.array([])),
        (0, 1, [0], [0]),
    ],
)
def test_fixed_forward_indexer_bounds(
    window_size, num_values, expected_start, expected_end, step
):
    # GH 43267
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values, step=step)

    tm.assert_numpy_array_equal(
        start, np.array(expected_start[::step]), check_dtype=False
    )
    tm.assert_numpy_array_equal(end, np.array(expected_end[::step]), check_dtype=False)
    assert len(start) == len(end)


@pytest.mark.parametrize(
    "df, window_size, expected",
    [
        (
            DataFrame({"b": [0, 1, 2], "a": [1, 2, 2]}),
            2,
            Series(
                [0, 1.5, 2.0],
                index=MultiIndex.from_arrays([[1, 2, 2], range(3)], names=["a", None]),
                name="b",
                dtype=np.float64,
            ),
        ),
        (
            DataFrame(
                {
                    "b": [np.nan, 1, 2, np.nan] + list(range(4, 18)),
                    "a": [1] * 7 + [2] * 11,
                    "c": range(18),
                }
            ),
            12,
            Series(
                [
                    3.6,
                    3.6,
                    4.25,
                    5.0,
                    5.0,
                    5.5,
                    6.0,
                    12.0,
                    12.5,
                    13.0,
                    13.5,
                    14.0,
                    14.5,
                    15.0,
                    15.5,
                    16.0,
                    16.5,
                    17.0,
                ],
                index=MultiIndex.from_arrays(
                    [[1] * 7 + [2] * 11, range(18)], names=["a", None]
                ),
                name="b",
                dtype=np.float64,
            ),
        ),
    ],
)
def test_rolling_groupby_with_fixed_forward_specific(df, window_size, expected):
    # GH 43267
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    result = df.groupby("a")["b"].rolling(window=indexer, min_periods=1).mean()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "group_keys",
    [
        (1,),
        (1, 2),
        (2, 1),
        (1, 1, 2),
        (1, 2, 1),
        (1, 1, 2, 2),
        (1, 2, 3, 2, 3),
        (1, 1, 2) * 4,
        (1, 2, 3) * 5,
    ],
)
@pytest.mark.parametrize("window_size", [1, 2, 3, 4, 5, 8, 20])
def test_rolling_groupby_with_fixed_forward_many(group_keys, window_size):
    # GH 43267
    df = DataFrame(
        {
            "a": np.array(list(group_keys)),
            "b": np.arange(len(group_keys), dtype=np.float64) + 17,
            "c": np.arange(len(group_keys), dtype=np.int64),
        }
    )

    indexer = FixedForwardWindowIndexer(window_size=window_size)
    result = df.groupby("a")["b"].rolling(window=indexer, min_periods=1).sum()
    result.index.names = ["a", "c"]

    groups = df.groupby("a")[["a", "b", "c"]]
    manual = concat(
        [
            g.assign(
                b=[
                    g["b"].iloc[i : i + window_size].sum(min_count=1)
                    for i in range(len(g))
                ]
            )
            for _, g in groups
        ]
    )
    manual = manual.set_index(["a", "c"])["b"]

    tm.assert_series_equal(result, manual)


def test_unequal_start_end_bounds():
    class CustomIndexer(BaseIndexer):
        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            return np.array([1]), np.array([1, 2])

    indexer = CustomIndexer()
    roll = Series(1).rolling(indexer)
    match = "start"
    with pytest.raises(ValueError, match=match):
        roll.mean()

    with pytest.raises(ValueError, match=match):
        next(iter(roll))

    with pytest.raises(ValueError, match=match):
        roll.corr(pairwise=True)

    with pytest.raises(ValueError, match=match):
        roll.cov(pairwise=True)


def test_unequal_bounds_to_object():
    # GH 44470
    class CustomIndexer(BaseIndexer):
        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            return np.array([1]), np.array([2])

    indexer = CustomIndexer()
    roll = Series([1, 1]).rolling(indexer)
    match = "start and end"
    with pytest.raises(ValueError, match=match):
        roll.mean()

    with pytest.raises(ValueError, match=match):
        next(iter(roll))

    with pytest.raises(ValueError, match=match):
        roll.corr(pairwise=True)

    with pytest.raises(ValueError, match=match):
        roll.cov(pairwise=True)