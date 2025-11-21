
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\json\test_json_table_schema.py ===
"""Tests for Table Schema integration."""
from collections import OrderedDict
from io import StringIO
import json

import numpy as np
import pytest

from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    PeriodDtype,
)

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm

from pandas.io.json._table_schema import (
    as_json_table_type,
    build_table_schema,
    convert_json_field_to_pandas_type,
    convert_pandas_type_to_json_field,
    set_default_names,
)


@pytest.fixture
def df_schema():
    return DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": ["a", "b", "c", "c"],
            "C": pd.date_range("2016-01-01", freq="d", periods=4),
            "D": pd.timedelta_range("1h", periods=4, freq="min"),
        },
        index=pd.Index(range(4), name="idx"),
    )


@pytest.fixture
def df_table():
    return DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": ["a", "b", "c", "c"],
            "C": pd.date_range("2016-01-01", freq="d", periods=4),
            "D": pd.timedelta_range("1h", periods=4, freq="min"),
            "E": pd.Series(pd.Categorical(["a", "b", "c", "c"])),
            "F": pd.Series(pd.Categorical(["a", "b", "c", "c"], ordered=True)),
            "G": [1.0, 2.0, 3, 4.0],
            "H": pd.date_range("2016-01-01", freq="d", periods=4, tz="US/Central"),
        },
        index=pd.Index(range(4), name="idx"),
    )


class TestBuildSchema:
    def test_build_table_schema(self, df_schema, using_infer_string):
        result = build_table_schema(df_schema, version=False)
        expected = {
            "fields": [
                {"name": "idx", "type": "integer"},
                {"name": "A", "type": "integer"},
                {"name": "B", "type": "string"},
                {"name": "C", "type": "datetime"},
                {"name": "D", "type": "duration"},
            ],
            "primaryKey": ["idx"],
        }
        if using_infer_string:
            expected["fields"][2] = {"name": "B", "type": "any", "extDtype": "str"}
        assert result == expected
        result = build_table_schema(df_schema)
        assert "pandas_version" in result

    def test_series(self):
        s = pd.Series([1, 2, 3], name="foo")
        result = build_table_schema(s, version=False)
        expected = {
            "fields": [
                {"name": "index", "type": "integer"},
                {"name": "foo", "type": "integer"},
            ],
            "primaryKey": ["index"],
        }
        assert result == expected
        result = build_table_schema(s)
        assert "pandas_version" in result

    def test_series_unnamed(self):
        result = build_table_schema(pd.Series([1, 2, 3]), version=False)
        expected = {
            "fields": [
                {"name": "index", "type": "integer"},
                {"name": "values", "type": "integer"},
            ],
            "primaryKey": ["index"],
        }
        assert result == expected

    def test_multiindex(self, df_schema, using_infer_string):
        df = df_schema
        idx = pd.MultiIndex.from_product([("a", "b"), (1, 2)])
        df.index = idx

        result = build_table_schema(df, version=False)
        expected = {
            "fields": [
                {"name": "level_0", "type": "string"},
                {"name": "level_1", "type": "integer"},
                {"name": "A", "type": "integer"},
                {"name": "B", "type": "string"},
                {"name": "C", "type": "datetime"},
                {"name": "D", "type": "duration"},
            ],
            "primaryKey": ["level_0", "level_1"],
        }
        if using_infer_string:
            expected["fields"][0] = {
                "name": "level_0",
                "type": "any",
                "extDtype": "str",
            }
            expected["fields"][3] = {"name": "B", "type": "any", "extDtype": "str"}
        assert result == expected

        df.index.names = ["idx0", None]
        expected["fields"][0]["name"] = "idx0"
        expected["primaryKey"] = ["idx0", "level_1"]
        result = build_table_schema(df, version=False)
        assert result == expected


class TestTableSchemaType:
    @pytest.mark.parametrize("int_type", [int, np.int16, np.int32, np.int64])
    def test_as_json_table_type_int_data(self, int_type):
        int_data = [1, 2, 3]
        assert as_json_table_type(np.array(int_data, dtype=int_type).dtype) == "integer"

    @pytest.mark.parametrize("float_type", [float, np.float16, np.float32, np.float64])
    def test_as_json_table_type_float_data(self, float_type):
        float_data = [1.0, 2.0, 3.0]
        assert (
            as_json_table_type(np.array(float_data, dtype=float_type).dtype) == "number"
        )

    @pytest.mark.parametrize("bool_type", [bool, np.bool_])
    def test_as_json_table_type_bool_data(self, bool_type):
        bool_data = [True, False]
        assert (
            as_json_table_type(np.array(bool_data, dtype=bool_type).dtype) == "boolean"
        )

    @pytest.mark.parametrize(
        "date_data",
        [
            pd.to_datetime(["2016"]),
            pd.to_datetime(["2016"], utc=True),
            pd.Series(pd.to_datetime(["2016"])),
            pd.Series(pd.to_datetime(["2016"], utc=True)),
            pd.period_range("2016", freq="Y", periods=3),
        ],
    )
    def test_as_json_table_type_date_data(self, date_data):
        assert as_json_table_type(date_data.dtype) == "datetime"

    @pytest.mark.parametrize(
        "str_data",
        [pd.Series(["a", "b"], dtype=object), pd.Index(["a", "b"], dtype=object)],
    )
    def test_as_json_table_type_string_data(self, str_data):
        assert as_json_table_type(str_data.dtype) == "string"

    @pytest.mark.parametrize(
        "cat_data",
        [
            pd.Categorical(["a"]),
            pd.Categorical([1]),
            pd.Series(pd.Categorical([1])),
            pd.CategoricalIndex([1]),
            pd.Categorical([1]),
        ],
    )
    def test_as_json_table_type_categorical_data(self, cat_data):
        assert as_json_table_type(cat_data.dtype) == "any"

    # ------
    # dtypes
    # ------
    @pytest.mark.parametrize("int_dtype", [int, np.int16, np.int32, np.int64])
    def test_as_json_table_type_int_dtypes(self, int_dtype):
        assert as_json_table_type(int_dtype) == "integer"

    @pytest.mark.parametrize("float_dtype", [float, np.float16, np.float32, np.float64])
    def test_as_json_table_type_float_dtypes(self, float_dtype):
        assert as_json_table_type(float_dtype) == "number"

    @pytest.mark.parametrize("bool_dtype", [bool, np.bool_])
    def test_as_json_table_type_bool_dtypes(self, bool_dtype):
        assert as_json_table_type(bool_dtype) == "boolean"

    @pytest.mark.parametrize(
        "date_dtype",
        [
            np.dtype("<M8[ns]"),
            PeriodDtype("D"),
            DatetimeTZDtype("ns", "US/Central"),
        ],
    )
    def test_as_json_table_type_date_dtypes(self, date_dtype):
        # TODO: datedate.date? datetime.time?
        assert as_json_table_type(date_dtype) == "datetime"

    @pytest.mark.parametrize("td_dtype", [np.dtype("<m8[ns]")])
    def test_as_json_table_type_timedelta_dtypes(self, td_dtype):
        assert as_json_table_type(td_dtype) == "duration"

    @pytest.mark.parametrize("str_dtype", [object])  # TODO(GH#14904) flesh out dtypes?
    def test_as_json_table_type_string_dtypes(self, str_dtype):
        assert as_json_table_type(str_dtype) == "string"

    def test_as_json_table_type_categorical_dtypes(self):
        assert as_json_table_type(pd.Categorical(["a"]).dtype) == "any"
        assert as_json_table_type(CategoricalDtype()) == "any"


class TestTableOrient:
    def test_build_series(self):
        s = pd.Series([1, 2], name="a")
        s.index.name = "id"
        result = s.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)

        assert "pandas_version" in result["schema"]
        result["schema"].pop("pandas_version")

        fields = [{"name": "id", "type": "integer"}, {"name": "a", "type": "integer"}]

        schema = {"fields": fields, "primaryKey": ["id"]}

        expected = OrderedDict(
            [
                ("schema", schema),
                (
                    "data",
                    [
                        OrderedDict([("id", 0), ("a", 1)]),
                        OrderedDict([("id", 1), ("a", 2)]),
                    ],
                ),
            ]
        )

        assert result == expected

    def test_read_json_from_to_json_results(self):
        # GH32383
        df = DataFrame(
            {
                "_id": {"row_0": 0},
                "category": {"row_0": "Goods"},
                "recommender_id": {"row_0": 3},
                "recommender_name_jp": {"row_0": "浦田"},
                "recommender_name_en": {"row_0": "Urata"},
                "name_jp": {"row_0": "博多人形(松尾吉将まつお よしまさ)"},
                "name_en": {"row_0": "Hakata Dolls Matsuo"},
            }
        )

        result1 = pd.read_json(StringIO(df.to_json()))
        result2 = DataFrame.from_dict(json.loads(df.to_json()))
        tm.assert_frame_equal(result1, df)
        tm.assert_frame_equal(result2, df)

    def test_to_json(self, df_table, using_infer_string):
        df = df_table
        df.index.name = "idx"
        result = df.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)

        assert "pandas_version" in result["schema"]
        result["schema"].pop("pandas_version")

        fields = [
            {"name": "idx", "type": "integer"},
            {"name": "A", "type": "integer"},
            {"name": "B", "type": "string"},
            {"name": "C", "type": "datetime"},
            {"name": "D", "type": "duration"},
            {
                "constraints": {"enum": ["a", "b", "c"]},
                "name": "E",
                "ordered": False,
                "type": "any",
            },
            {
                "constraints": {"enum": ["a", "b", "c"]},
                "name": "F",
                "ordered": True,
                "type": "any",
            },
            {"name": "G", "type": "number"},
            {"name": "H", "type": "datetime", "tz": "US/Central"},
        ]

        if using_infer_string:
            fields[2] = {"name": "B", "type": "any", "extDtype": "str"}

        schema = {"fields": fields, "primaryKey": ["idx"]}
        data = [
            OrderedDict(
                [
                    ("idx", 0),
                    ("A", 1),
                    ("B", "a"),
                    ("C", "2016-01-01T00:00:00.000"),
                    ("D", "P0DT1H0M0S"),
                    ("E", "a"),
                    ("F", "a"),
                    ("G", 1.0),
                    ("H", "2016-01-01T06:00:00.000Z"),
                ]
            ),
            OrderedDict(
                [
                    ("idx", 1),
                    ("A", 2),
                    ("B", "b"),
                    ("C", "2016-01-02T00:00:00.000"),
                    ("D", "P0DT1H1M0S"),
                    ("E", "b"),
                    ("F", "b"),
                    ("G", 2.0),
                    ("H", "2016-01-02T06:00:00.000Z"),
                ]
            ),
            OrderedDict(
                [
                    ("idx", 2),
                    ("A", 3),
                    ("B", "c"),
                    ("C", "2016-01-03T00:00:00.000"),
                    ("D", "P0DT1H2M0S"),
                    ("E", "c"),
                    ("F", "c"),
                    ("G", 3.0),
                    ("H", "2016-01-03T06:00:00.000Z"),
                ]
            ),
            OrderedDict(
                [
                    ("idx", 3),
                    ("A", 4),
                    ("B", "c"),
                    ("C", "2016-01-04T00:00:00.000"),
                    ("D", "P0DT1H3M0S"),
                    ("E", "c"),
                    ("F", "c"),
                    ("G", 4.0),
                    ("H", "2016-01-04T06:00:00.000Z"),
                ]
            ),
        ]
        expected = OrderedDict([("schema", schema), ("data", data)])

        assert result == expected

    def test_to_json_float_index(self):
        data = pd.Series(1, index=[1.0, 2.0])
        result = data.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)
        result["schema"].pop("pandas_version")

        expected = OrderedDict(
            [
                (
                    "schema",
                    {
                        "fields": [
                            {"name": "index", "type": "number"},
                            {"name": "values", "type": "integer"},
                        ],
                        "primaryKey": ["index"],
                    },
                ),
                (
                    "data",
                    [
                        OrderedDict([("index", 1.0), ("values", 1)]),
                        OrderedDict([("index", 2.0), ("values", 1)]),
                    ],
                ),
            ]
        )

        assert result == expected

    def test_to_json_period_index(self):
        idx = pd.period_range("2016", freq="Q-JAN", periods=2)
        data = pd.Series(1, idx)
        result = data.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)
        result["schema"].pop("pandas_version")

        fields = [
            {"freq": "QE-JAN", "name": "index", "type": "datetime"},
            {"name": "values", "type": "integer"},
        ]

        schema = {"fields": fields, "primaryKey": ["index"]}
        data = [
            OrderedDict([("index", "2015-11-01T00:00:00.000"), ("values", 1)]),
            OrderedDict([("index", "2016-02-01T00:00:00.000"), ("values", 1)]),
        ]
        expected = OrderedDict([("schema", schema), ("data", data)])

        assert result == expected

    def test_to_json_categorical_index(self):
        data = pd.Series(1, pd.CategoricalIndex(["a", "b"]))
        result = data.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)
        result["schema"].pop("pandas_version")

        expected = OrderedDict(
            [
                (
                    "schema",
                    {
                        "fields": [
                            {
                                "name": "index",
                                "type": "any",
                                "constraints": {"enum": ["a", "b"]},
                                "ordered": False,
                            },
                            {"name": "values", "type": "integer"},
                        ],
                        "primaryKey": ["index"],
                    },
                ),
                (
                    "data",
                    [
                        OrderedDict([("index", "a"), ("values", 1)]),
                        OrderedDict([("index", "b"), ("values", 1)]),
                    ],
                ),
            ]
        )

        assert result == expected

    def test_date_format_raises(self, df_table):
        msg = (
            "Trying to write with `orient='table'` and `date_format='epoch'`. Table "
            "Schema requires dates to be formatted with `date_format='iso'`"
        )
        with pytest.raises(ValueError, match=msg):
            df_table.to_json(orient="table", date_format="epoch")

        # others work
        df_table.to_json(orient="table", date_format="iso")
        df_table.to_json(orient="table")

    def test_convert_pandas_type_to_json_field_int(self, index_or_series):
        kind = index_or_series
        data = [1, 2, 3]
        result = convert_pandas_type_to_json_field(kind(data, name="name"))
        expected = {"name": "name", "type": "integer"}
        assert result == expected

    def test_convert_pandas_type_to_json_field_float(self, index_or_series):
        kind = index_or_series
        data = [1.0, 2.0, 3.0]
        result = convert_pandas_type_to_json_field(kind(data, name="name"))
        expected = {"name": "name", "type": "number"}
        assert result == expected

    @pytest.mark.parametrize(
        "dt_args,extra_exp", [({}, {}), ({"utc": True}, {"tz": "UTC"})]
    )
    @pytest.mark.parametrize("wrapper", [None, pd.Series])
    def test_convert_pandas_type_to_json_field_datetime(
        self, dt_args, extra_exp, wrapper
    ):
        data = [1.0, 2.0, 3.0]
        data = pd.to_datetime(data, **dt_args)
        if wrapper is pd.Series:
            data = pd.Series(data, name="values")
        result = convert_pandas_type_to_json_field(data)
        expected = {"name": "values", "type": "datetime"}
        expected.update(extra_exp)
        assert result == expected

    def test_convert_pandas_type_to_json_period_range(self):
        arr = pd.period_range("2016", freq="Y-DEC", periods=4)
        result = convert_pandas_type_to_json_field(arr)
        expected = {"name": "values", "type": "datetime", "freq": "YE-DEC"}
        assert result == expected

    @pytest.mark.parametrize("kind", [pd.Categorical, pd.CategoricalIndex])
    @pytest.mark.parametrize("ordered", [True, False])
    def test_convert_pandas_type_to_json_field_categorical(self, kind, ordered):
        data = ["a", "b", "c"]
        if kind is pd.Categorical:
            arr = pd.Series(kind(data, ordered=ordered), name="cats")
        elif kind is pd.CategoricalIndex:
            arr = kind(data, ordered=ordered, name="cats")

        result = convert_pandas_type_to_json_field(arr)
        expected = {
            "name": "cats",
            "type": "any",
            "constraints": {"enum": data},
            "ordered": ordered,
        }
        assert result == expected

    @pytest.mark.parametrize(
        "inp,exp",
        [
            ({"type": "integer"}, "int64"),
            ({"type": "number"}, "float64"),
            ({"type": "boolean"}, "bool"),
            ({"type": "duration"}, "timedelta64"),
            ({"type": "datetime"}, "datetime64[ns]"),
            ({"type": "datetime", "tz": "US/Hawaii"}, "datetime64[ns, US/Hawaii]"),
            ({"type": "any"}, "object"),
            (
                {
                    "type": "any",
                    "constraints": {"enum": ["a", "b", "c"]},
                    "ordered": False,
                },
                CategoricalDtype(categories=["a", "b", "c"], ordered=False),
            ),
            (
                {
                    "type": "any",
                    "constraints": {"enum": ["a", "b", "c"]},
                    "ordered": True,
                },
                CategoricalDtype(categories=["a", "b", "c"], ordered=True),
            ),
            ({"type": "string"}, "object"),
        ],
    )
    def test_convert_json_field_to_pandas_type(self, inp, exp):
        field = {"name": "foo"}
        field.update(inp)
        assert convert_json_field_to_pandas_type(field) == exp

    @pytest.mark.parametrize("inp", ["geopoint", "geojson", "fake_type"])
    def test_convert_json_field_to_pandas_type_raises(self, inp):
        field = {"type": inp}
        with pytest.raises(
            ValueError, match=f"Unsupported or invalid field type: {inp}"
        ):
            convert_json_field_to_pandas_type(field)

    def test_categorical(self):
        s = pd.Series(pd.Categorical(["a", "b", "a"]))
        s.index.name = "idx"
        result = s.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)
        result["schema"].pop("pandas_version")

        fields = [
            {"name": "idx", "type": "integer"},
            {
                "constraints": {"enum": ["a", "b"]},
                "name": "values",
                "ordered": False,
                "type": "any",
            },
        ]

        expected = OrderedDict(
            [
                ("schema", {"fields": fields, "primaryKey": ["idx"]}),
                (
                    "data",
                    [
                        OrderedDict([("idx", 0), ("values", "a")]),
                        OrderedDict([("idx", 1), ("values", "b")]),
                        OrderedDict([("idx", 2), ("values", "a")]),
                    ],
                ),
            ]
        )

        assert result == expected

    @pytest.mark.parametrize(
        "idx,nm,prop",
        [
            (pd.Index([1]), "index", "name"),
            (pd.Index([1], name="myname"), "myname", "name"),
            (
                pd.MultiIndex.from_product([("a", "b"), ("c", "d")]),
                ["level_0", "level_1"],
                "names",
            ),
            (
                pd.MultiIndex.from_product(
                    [("a", "b"), ("c", "d")], names=["n1", "n2"]
                ),
                ["n1", "n2"],
                "names",
            ),
            (
                pd.MultiIndex.from_product(
                    [("a", "b"), ("c", "d")], names=["n1", None]
                ),
                ["n1", "level_1"],
                "names",
            ),
        ],
    )
    def test_set_names_unset(self, idx, nm, prop):
        data = pd.Series(1, idx)
        result = set_default_names(data)
        assert getattr(result.index, prop) == nm

    @pytest.mark.parametrize(
        "idx",
        [
            pd.Index([], name="index"),
            pd.MultiIndex.from_arrays([["foo"], ["bar"]], names=("level_0", "level_1")),
            pd.MultiIndex.from_arrays([["foo"], ["bar"]], names=("foo", "level_1")),
        ],
    )
    def test_warns_non_roundtrippable_names(self, idx):
        # GH 19130
        df = DataFrame(index=idx)
        df.index.name = "index"
        with tm.assert_produces_warning():
            set_default_names(df)

    def test_timestamp_in_columns(self):
        df = DataFrame(
            [[1, 2]], columns=[pd.Timestamp("2016"), pd.Timedelta(10, unit="s")]
        )
        result = df.to_json(orient="table")
        js = json.loads(result)
        assert js["schema"]["fields"][1]["name"] == "2016-01-01T00:00:00.000"
        assert js["schema"]["fields"][2]["name"] == "P0DT0H0M10S"

    @pytest.mark.parametrize(
        "case",
        [
            pd.Series([1], index=pd.Index([1], name="a"), name="a"),
            DataFrame({"A": [1]}, index=pd.Index([1], name="A")),
            DataFrame(
                {"A": [1]},
                index=pd.MultiIndex.from_arrays([["a"], [1]], names=["A", "a"]),
            ),
        ],
    )
    def test_overlapping_names(self, case):
        with pytest.raises(ValueError, match="Overlapping"):
            case.to_json(orient="table")

    def test_mi_falsey_name(self):
        # GH 16203
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=pd.MultiIndex.from_product([("A", "B"), ("a", "b")]),
        )
        result = [x["name"] for x in build_table_schema(df)["fields"]]
        assert result == ["level_0", "level_1", 0, 1, 2, 3]


class TestTableOrientReader:
    @pytest.mark.parametrize(
        "index_nm",
        [None, "idx", pytest.param("index", marks=pytest.mark.xfail), "level_0"],
    )
    @pytest.mark.parametrize(
        "vals",
        [
            {"ints": [1, 2, 3, 4]},
            {"objects": ["a", "b", "c", "d"]},
            {"objects": ["1", "2", "3", "4"]},
            {"date_ranges": pd.date_range("2016-01-01", freq="d", periods=4)},
            {"categoricals": pd.Series(pd.Categorical(["a", "b", "c", "c"]))},
            {
                "ordered_cats": pd.Series(
                    pd.Categorical(["a", "b", "c", "c"], ordered=True)
                )
            },
            {"floats": [1.0, 2.0, 3.0, 4.0]},
            {"floats": [1.1, 2.2, 3.3, 4.4]},
            {"bools": [True, False, False, True]},
            {
                "timezones": pd.date_range(
                    "2016-01-01", freq="d", periods=4, tz="US/Central"
                )  # added in # GH 35973
            },
        ],
    )
    def test_read_json_table_orient(self, index_nm, vals, recwarn):
        df = DataFrame(vals, index=pd.Index(range(4), name=index_nm))
        out = df.to_json(orient="table")
        result = pd.read_json(out, orient="table")
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize("index_nm", [None, "idx", "index"])
    @pytest.mark.parametrize(
        "vals",
        [{"timedeltas": pd.timedelta_range("1h", periods=4, freq="min")}],
    )
    def test_read_json_table_orient_raises(self, index_nm, vals, recwarn):
        df = DataFrame(vals, index=pd.Index(range(4), name=index_nm))
        out = df.to_json(orient="table")
        with pytest.raises(NotImplementedError, match="can not yet read "):
            pd.read_json(out, orient="table")

    @pytest.mark.parametrize(
        "index_nm",
        [None, "idx", pytest.param("index", marks=pytest.mark.xfail), "level_0"],
    )
    @pytest.mark.parametrize(
        "vals",
        [
            {"ints": [1, 2, 3, 4]},
            {"objects": ["a", "b", "c", "d"]},
            {"objects": ["1", "2", "3", "4"]},
            {"date_ranges": pd.date_range("2016-01-01", freq="d", periods=4)},
            {"categoricals": pd.Series(pd.Categorical(["a", "b", "c", "c"]))},
            {
                "ordered_cats": pd.Series(
                    pd.Categorical(["a", "b", "c", "c"], ordered=True)
                )
            },
            {"floats": [1.0, 2.0, 3.0, 4.0]},
            {"floats": [1.1, 2.2, 3.3, 4.4]},
            {"bools": [True, False, False, True]},
            {
                "timezones": pd.date_range(
                    "2016-01-01", freq="d", periods=4, tz="US/Central"
                )  # added in # GH 35973
            },
        ],
    )
    def test_read_json_table_period_orient(self, index_nm, vals, recwarn):
        df = DataFrame(
            vals,
            index=pd.Index(
                (pd.Period(f"2022Q{q}") for q in range(1, 5)), name=index_nm
            ),
        )
        out = df.to_json(orient="table")
        result = pd.read_json(out, orient="table")
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize(
        "idx",
        [
            pd.Index(range(4)),
            pd.date_range(
                "2020-08-30",
                freq="d",
                periods=4,
            )._with_freq(None),
            pd.date_range(
                "2020-08-30", freq="d", periods=4, tz="US/Central"
            )._with_freq(None),
            pd.MultiIndex.from_product(
                [
                    pd.date_range("2020-08-30", freq="d", periods=2, tz="US/Central"),
                    ["x", "y"],
                ],
            ),
        ],
    )
    @pytest.mark.parametrize(
        "vals",
        [
            {"floats": [1.1, 2.2, 3.3, 4.4]},
            {"dates": pd.date_range("2020-08-30", freq="d", periods=4)},
            {
                "timezones": pd.date_range(
                    "2020-08-30", freq="d", periods=4, tz="Europe/London"
                )
            },
        ],
    )
    def test_read_json_table_timezones_orient(self, idx, vals, recwarn):
        # GH 35973
        df = DataFrame(vals, index=idx)
        out = df.to_json(orient="table")
        result = pd.read_json(out, orient="table")
        tm.assert_frame_equal(df, result)

    def test_comprehensive(self):
        df = DataFrame(
            {
                "A": [1, 2, 3, 4],
                "B": ["a", "b", "c", "c"],
                "C": pd.date_range("2016-01-01", freq="d", periods=4),
                # 'D': pd.timedelta_range('1h', periods=4, freq='min'),
                "E": pd.Series(pd.Categorical(["a", "b", "c", "c"])),
                "F": pd.Series(pd.Categorical(["a", "b", "c", "c"], ordered=True)),
                "G": [1.1, 2.2, 3.3, 4.4],
                "H": pd.date_range("2016-01-01", freq="d", periods=4, tz="US/Central"),
                "I": [True, False, False, True],
            },
            index=pd.Index(range(4), name="idx"),
        )

        out = StringIO(df.to_json(orient="table"))
        result = pd.read_json(out, orient="table")
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize(
        "index_names",
        [[None, None], ["foo", "bar"], ["foo", None], [None, "foo"], ["index", "foo"]],
    )
    def test_multiindex(self, index_names):
        # GH 18912
        df = DataFrame(
            [["Arr", "alpha", [1, 2, 3, 4]], ["Bee", "Beta", [10, 20, 30, 40]]],
            index=[["A", "B"], ["Null", "Eins"]],
            columns=["Aussprache", "Griechisch", "Args"],
        )
        df.index.names = index_names
        out = StringIO(df.to_json(orient="table"))
        result = pd.read_json(out, orient="table")
        tm.assert_frame_equal(df, result)

    def test_empty_frame_roundtrip(self):
        # GH 21287
        df = DataFrame(columns=["a", "b", "c"])
        expected = df.copy()
        out = StringIO(df.to_json(orient="table"))
        result = pd.read_json(out, orient="table")
        tm.assert_frame_equal(expected, result)

    def test_read_json_orient_table_old_schema_version(self):
        df_json = """
        {
            "schema":{
                "fields":[
                    {"name":"index","type":"integer"},
                    {"name":"a","type":"string"}
                ],
                "primaryKey":["index"],
                "pandas_version":"0.20.0"
            },
            "data":[
                {"index":0,"a":1},
                {"index":1,"a":2.0},
                {"index":2,"a":"s"}
            ]
        }
        """
        expected = DataFrame({"a": [1, 2.0, "s"]})
        result = pd.read_json(StringIO(df_json), orient="table")
        tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize("freq", ["M", "2M", "Q", "2Q", "Y", "2Y"])
    def test_read_json_table_orient_period_depr_freq(self, freq, recwarn):
        # GH#9586
        df = DataFrame(
            {"ints": [1, 2]},
            index=pd.PeriodIndex(["2020-01", "2021-06"], freq=freq),
        )
        out = df.to_json(orient="table")
        result = pd.read_json(out, orient="table")
        tm.assert_frame_equal(df, result)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\tests\test_latex_lark.py ===
from sympy.testing.pytest import XFAIL
from sympy.parsing.latex.lark import parse_latex_lark
from sympy.external import import_module

from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import Derivative, Function
from sympy.core.numbers import E, oo, Rational
from sympy.core.power import Pow
from sympy.core.parameters import evaluate
from sympy.core.relational import GreaterThan, LessThan, StrictGreaterThan, StrictLessThan, Unequality
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions.elementary.complexes import Abs, conjugate
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import root, sqrt, Min, Max
from sympy.functions.elementary.trigonometric import asin, cos, csc, sec, sin, tan
from sympy.integrals.integrals import Integral
from sympy.series.limits import Limit
from sympy import Matrix, MatAdd, MatMul, Transpose, Trace
from sympy import I

from sympy.core.relational import Eq, Ne, Lt, Le, Gt, Ge
from sympy.physics.quantum import Bra, Ket, InnerProduct
from sympy.abc import x, y, z, a, b, c, d, t, k, n

from .test_latex import theta, f, _Add, _Mul, _Pow, _Sqrt, _Conjugate, _Abs, _factorial, _exp, _binomial

lark = import_module("lark")

# disable tests if lark is not present
disabled = lark is None

# shorthand definitions that are only needed for the Lark LaTeX parser
def _Min(*args):
    return Min(*args, evaluate=False)


def _Max(*args):
    return Max(*args, evaluate=False)


def _log(a, b=E):
    if b == E:
        return log(a, evaluate=False)
    else:
        return log(a, b, evaluate=False)


def _MatAdd(a, b):
    return MatAdd(a, b, evaluate=False)


def _MatMul(a, b):
    return MatMul(a, b, evaluate=False)


# These LaTeX strings should parse to the corresponding SymPy expression
SYMBOL_EXPRESSION_PAIRS = [
    (r"x_0", Symbol('x_{0}')),
    (r"x_{1}", Symbol('x_{1}')),
    (r"x_a", Symbol('x_{a}')),
    (r"x_{b}", Symbol('x_{b}')),
    (r"h_\theta", Symbol('h_{theta}')),
    (r"h_{\theta}", Symbol('h_{theta}')),
    (r"y''_1", Symbol("y''_{1}")),
    (r"y_1''", Symbol("y_{1}''")),
    (r"\mathit{x}", Symbol('x')),
    (r"\mathit{test}", Symbol('test')),
    (r"\mathit{TEST}", Symbol('TEST')),
    (r"\mathit{HELLO world}", Symbol('HELLO world')),
    (r"a'", Symbol("a'")),
    (r"a''", Symbol("a''")),
    (r"\alpha'", Symbol("alpha'")),
    (r"\alpha''", Symbol("alpha''")),
    (r"a_b", Symbol("a_{b}")),
    (r"a_b'", Symbol("a_{b}'")),
    (r"a'_b", Symbol("a'_{b}")),
    (r"a'_b'", Symbol("a'_{b}'")),
    (r"a_{b'}", Symbol("a_{b'}")),
    (r"a_{b'}'", Symbol("a_{b'}'")),
    (r"a'_{b'}", Symbol("a'_{b'}")),
    (r"a'_{b'}'", Symbol("a'_{b'}'")),
    (r"\mathit{foo}'", Symbol("foo'")),
    (r"\mathit{foo'}", Symbol("foo'")),
    (r"\mathit{foo'}'", Symbol("foo''")),
    (r"a_b''", Symbol("a_{b}''")),
    (r"a''_b", Symbol("a''_{b}")),
    (r"a''_b'''", Symbol("a''_{b}'''")),
    (r"a_{b''}", Symbol("a_{b''}")),
    (r"a_{b''}''", Symbol("a_{b''}''")),
    (r"a''_{b''}", Symbol("a''_{b''}")),
    (r"a''_{b''}'''", Symbol("a''_{b''}'''")),
    (r"\mathit{foo}''", Symbol("foo''")),
    (r"\mathit{foo''}", Symbol("foo''")),
    (r"\mathit{foo''}'''", Symbol("foo'''''")),
    (r"a_\alpha", Symbol("a_{alpha}")),
    (r"a_\alpha'", Symbol("a_{alpha}'")),
    (r"a'_\alpha", Symbol("a'_{alpha}")),
    (r"a'_\alpha'", Symbol("a'_{alpha}'")),
    (r"a_{\alpha'}", Symbol("a_{alpha'}")),
    (r"a_{\alpha'}'", Symbol("a_{alpha'}'")),
    (r"a'_{\alpha'}", Symbol("a'_{alpha'}")),
    (r"a'_{\alpha'}'", Symbol("a'_{alpha'}'")),
    (r"a_\alpha''", Symbol("a_{alpha}''")),
    (r"a''_\alpha", Symbol("a''_{alpha}")),
    (r"a''_\alpha'''", Symbol("a''_{alpha}'''")),
    (r"a_{\alpha''}", Symbol("a_{alpha''}")),
    (r"a_{\alpha''}''", Symbol("a_{alpha''}''")),
    (r"a''_{\alpha''}", Symbol("a''_{alpha''}")),
    (r"a''_{\alpha''}'''", Symbol("a''_{alpha''}'''")),
    (r"\alpha_b", Symbol("alpha_{b}")),
    (r"\alpha_b'", Symbol("alpha_{b}'")),
    (r"\alpha'_b", Symbol("alpha'_{b}")),
    (r"\alpha'_b'", Symbol("alpha'_{b}'")),
    (r"\alpha_{b'}", Symbol("alpha_{b'}")),
    (r"\alpha_{b'}'", Symbol("alpha_{b'}'")),
    (r"\alpha'_{b'}", Symbol("alpha'_{b'}")),
    (r"\alpha'_{b'}'", Symbol("alpha'_{b'}'")),
    (r"\alpha_b''", Symbol("alpha_{b}''")),
    (r"\alpha''_b", Symbol("alpha''_{b}")),
    (r"\alpha''_b'''", Symbol("alpha''_{b}'''")),
    (r"\alpha_{b''}", Symbol("alpha_{b''}")),
    (r"\alpha_{b''}''", Symbol("alpha_{b''}''")),
    (r"\alpha''_{b''}", Symbol("alpha''_{b''}")),
    (r"\alpha''_{b''}'''", Symbol("alpha''_{b''}'''")),
    (r"\alpha_\beta", Symbol("alpha_{beta}")),
    (r"\alpha_{\beta}", Symbol("alpha_{beta}")),
    (r"\alpha_{\beta'}", Symbol("alpha_{beta'}")),
    (r"\alpha_{\beta''}", Symbol("alpha_{beta''}")),
    (r"\alpha'_\beta", Symbol("alpha'_{beta}")),
    (r"\alpha'_{\beta}", Symbol("alpha'_{beta}")),
    (r"\alpha'_{\beta'}", Symbol("alpha'_{beta'}")),
    (r"\alpha'_{\beta''}", Symbol("alpha'_{beta''}")),
    (r"\alpha''_\beta", Symbol("alpha''_{beta}")),
    (r"\alpha''_{\beta}", Symbol("alpha''_{beta}")),
    (r"\alpha''_{\beta'}", Symbol("alpha''_{beta'}")),
    (r"\alpha''_{\beta''}", Symbol("alpha''_{beta''}")),
    (r"\alpha_\beta'", Symbol("alpha_{beta}'")),
    (r"\alpha_{\beta}'", Symbol("alpha_{beta}'")),
    (r"\alpha_{\beta'}'", Symbol("alpha_{beta'}'")),
    (r"\alpha_{\beta''}'", Symbol("alpha_{beta''}'")),
    (r"\alpha'_\beta'", Symbol("alpha'_{beta}'")),
    (r"\alpha'_{\beta}'", Symbol("alpha'_{beta}'")),
    (r"\alpha'_{\beta'}'", Symbol("alpha'_{beta'}'")),
    (r"\alpha'_{\beta''}'", Symbol("alpha'_{beta''}'")),
    (r"\alpha''_\beta'", Symbol("alpha''_{beta}'")),
    (r"\alpha''_{\beta}'", Symbol("alpha''_{beta}'")),
    (r"\alpha''_{\beta'}'", Symbol("alpha''_{beta'}'")),
    (r"\alpha''_{\beta''}'", Symbol("alpha''_{beta''}'")),
    (r"\alpha_\beta''", Symbol("alpha_{beta}''")),
    (r"\alpha_{\beta}''", Symbol("alpha_{beta}''")),
    (r"\alpha_{\beta'}''", Symbol("alpha_{beta'}''")),
    (r"\alpha_{\beta''}''", Symbol("alpha_{beta''}''")),
    (r"\alpha'_\beta''", Symbol("alpha'_{beta}''")),
    (r"\alpha'_{\beta}''", Symbol("alpha'_{beta}''")),
    (r"\alpha'_{\beta'}''", Symbol("alpha'_{beta'}''")),
    (r"\alpha'_{\beta''}''", Symbol("alpha'_{beta''}''")),
    (r"\alpha''_\beta''", Symbol("alpha''_{beta}''")),
    (r"\alpha''_{\beta}''", Symbol("alpha''_{beta}''")),
    (r"\alpha''_{\beta'}''", Symbol("alpha''_{beta'}''")),
    (r"\alpha''_{\beta''}''", Symbol("alpha''_{beta''}''"))

]

UNEVALUATED_SIMPLE_EXPRESSION_PAIRS = [
    (r"0", 0),
    (r"1", 1),
    (r"-3.14", -3.14),
    (r"(-7.13)(1.5)", _Mul(-7.13, 1.5)),
    (r"1+1", _Add(1, 1)),
    (r"0+1", _Add(0, 1)),
    (r"1*2", _Mul(1, 2)),
    (r"0*1", _Mul(0, 1)),
    (r"x", x),
    (r"2x", 2 * x),
    (r"3x - 1", _Add(_Mul(3, x), -1)),
    (r"-c", -c),
    (r"\infty", oo),
    (r"a \cdot b", a * b),
    (r"1 \times 2 ", _Mul(1, 2)),
    (r"a / b", a / b),
    (r"a \div b", a / b),
    (r"a + b", a + b),
    (r"a + b - a", _Add(a + b, -a)),
    (r"(x + y) z", _Mul(_Add(x, y), z)),
    (r"a'b+ab'", _Add(_Mul(Symbol("a'"), b), _Mul(a, Symbol("b'"))))
]

EVALUATED_SIMPLE_EXPRESSION_PAIRS = [
    (r"(-7.13)(1.5)", -10.695),
    (r"1+1", 2),
    (r"0+1", 1),
    (r"1*2", 2),
    (r"0*1", 0),
    (r"2x", 2 * x),
    (r"3x - 1", 3 * x - 1),
    (r"-c", -c),
    (r"a \cdot b", a * b),
    (r"1 \times 2 ", 2),
    (r"a / b", a / b),
    (r"a \div b", a / b),
    (r"a + b", a + b),
    (r"a + b - a", b),
    (r"(x + y) z", (x + y) * z),
]

UNEVALUATED_FRACTION_EXPRESSION_PAIRS = [
    (r"\frac{a}{b}", a / b),
    (r"\dfrac{a}{b}", a / b),
    (r"\tfrac{a}{b}", a / b),
    (r"\frac12", _Mul(1, _Pow(2, -1))),
    (r"\frac12y", _Mul(_Mul(1, _Pow(2, -1)), y)),
    (r"\frac1234", _Mul(_Mul(1, _Pow(2, -1)), 34)),
    (r"\frac2{3}", _Mul(2, _Pow(3, -1))),
    (r"\frac{a + b}{c}", _Mul(a + b, _Pow(c, -1))),
    (r"\frac{7}{3}", _Mul(7, _Pow(3, -1)))
]

EVALUATED_FRACTION_EXPRESSION_PAIRS = [
    (r"\frac{a}{b}", a / b),
    (r"\dfrac{a}{b}", a / b),
    (r"\tfrac{a}{b}", a / b),
    (r"\frac12", Rational(1, 2)),
    (r"\frac12y", y / 2),
    (r"\frac1234", 17),
    (r"\frac2{3}", Rational(2, 3)),
    (r"\frac{a + b}{c}", (a + b) / c),
    (r"\frac{7}{3}", Rational(7, 3))
]

RELATION_EXPRESSION_PAIRS = [
    (r"x = y", Eq(x, y)),
    (r"x \neq y", Ne(x, y)),
    (r"x < y", Lt(x, y)),
    (r"x > y", Gt(x, y)),
    (r"x \leq y", Le(x, y)),
    (r"x \geq y", Ge(x, y)),
    (r"x \le y", Le(x, y)),
    (r"x \ge y", Ge(x, y)),
    (r"x < y", StrictLessThan(x, y)),
    (r"x \leq y", LessThan(x, y)),
    (r"x > y", StrictGreaterThan(x, y)),
    (r"x \geq y", GreaterThan(x, y)),
    (r"x \neq y", Unequality(x, y)), # same as 2nd one in the list
    (r"a^2 + b^2 = c^2", Eq(a**2 + b**2, c**2))
]

UNEVALUATED_POWER_EXPRESSION_PAIRS = [
    (r"x^2", x ** 2),
    (r"x^\frac{1}{2}", _Pow(x, _Mul(1, _Pow(2, -1)))),
    (r"x^{3 + 1}", x ** _Add(3, 1)),
    (r"\pi^{|xy|}", Symbol('pi') ** _Abs(x * y)),
    (r"5^0 - 4^0", _Add(_Pow(5, 0), _Mul(-1, _Pow(4, 0))))
]

EVALUATED_POWER_EXPRESSION_PAIRS = [
    (r"x^2", x ** 2),
    (r"x^\frac{1}{2}", sqrt(x)),
    (r"x^{3 + 1}", x ** 4),
    (r"\pi^{|xy|}", Symbol('pi') ** _Abs(x * y)),
    (r"5^0 - 4^0", 0)
]

UNEVALUATED_INTEGRAL_EXPRESSION_PAIRS = [
    (r"\int x dx", Integral(_Mul(1, x), x)),
    (r"\int x \, dx", Integral(_Mul(1, x), x)),
    (r"\int x d\theta", Integral(_Mul(1, x), theta)),
    (r"\int (x^2 - y)dx", Integral(_Mul(1, x ** 2 - y), x)),
    (r"\int x + a dx", Integral(_Mul(1, _Add(x, a)), x)),
    (r"\int da", Integral(_Mul(1, 1), a)),
    (r"\int_0^7 dx", Integral(_Mul(1, 1), (x, 0, 7))),
    (r"\int\limits_{0}^{1} x dx", Integral(_Mul(1, x), (x, 0, 1))),
    (r"\int_a^b x dx", Integral(_Mul(1, x), (x, a, b))),
    (r"\int^b_a x dx", Integral(_Mul(1, x), (x, a, b))),
    (r"\int_{a}^b x dx", Integral(_Mul(1, x), (x, a, b))),
    (r"\int^{b}_a x dx", Integral(_Mul(1, x), (x, a, b))),
    (r"\int_{a}^{b} x dx", Integral(_Mul(1, x), (x, a, b))),
    (r"\int^{b}_{a} x dx", Integral(_Mul(1, x), (x, a, b))),
    (r"\int_{f(a)}^{f(b)} f(z) dz", Integral(f(z), (z, f(a), f(b)))),
    (r"\int a + b + c dx", Integral(_Mul(1, _Add(_Add(a, b), c)), x)),
    (r"\int \frac{dz}{z}", Integral(_Mul(1, _Mul(1, Pow(z, -1))), z)),
    (r"\int \frac{3 dz}{z}", Integral(_Mul(1, _Mul(3, _Pow(z, -1))), z)),
    (r"\int \frac{1}{x} dx", Integral(_Mul(1, _Mul(1, Pow(x, -1))), x)),
    (r"\int \frac{1}{a} + \frac{1}{b} dx",
     Integral(_Mul(1, _Add(_Mul(1, _Pow(a, -1)), _Mul(1, Pow(b, -1)))), x)),
    (r"\int \frac{1}{x} + 1 dx", Integral(_Mul(1, _Add(_Mul(1, _Pow(x, -1)), 1)), x))
]

EVALUATED_INTEGRAL_EXPRESSION_PAIRS = [
    (r"\int x dx", Integral(x, x)),
    (r"\int x \, dx", Integral(x, x)),
    (r"\int x d\theta", Integral(x, theta)),
    (r"\int (x^2 - y)dx", Integral(x ** 2 - y, x)),
    (r"\int x + a dx", Integral(x + a, x)),
    (r"\int da", Integral(1, a)),
    (r"\int_0^7 dx", Integral(1, (x, 0, 7))),
    (r"\int\limits_{0}^{1} x dx", Integral(x, (x, 0, 1))),
    (r"\int_a^b x dx", Integral(x, (x, a, b))),
    (r"\int^b_a x dx", Integral(x, (x, a, b))),
    (r"\int_{a}^b x dx", Integral(x, (x, a, b))),
    (r"\int^{b}_a x dx", Integral(x, (x, a, b))),
    (r"\int_{a}^{b} x dx", Integral(x, (x, a, b))),
    (r"\int^{b}_{a} x dx", Integral(x, (x, a, b))),
    (r"\int_{f(a)}^{f(b)} f(z) dz", Integral(f(z), (z, f(a), f(b)))),
    (r"\int a + b + c dx", Integral(a + b + c, x)),
    (r"\int \frac{dz}{z}", Integral(Pow(z, -1), z)),
    (r"\int \frac{3 dz}{z}", Integral(3 * Pow(z, -1), z)),
    (r"\int \frac{1}{x} dx", Integral(1 / x, x)),
    (r"\int \frac{1}{a} + \frac{1}{b} dx", Integral(1 / a + 1 / b, x)),
    (r"\int \frac{1}{a} - \frac{1}{b} dx", Integral(1 / a - 1 / b, x)),
    (r"\int \frac{1}{x} + 1 dx", Integral(1 / x + 1, x))
]

DERIVATIVE_EXPRESSION_PAIRS = [
    (r"\frac{d}{dx} x", Derivative(x, x)),
    (r"\frac{d}{dt} x", Derivative(x, t)),
    (r"\frac{d}{dx} ( \tan x )", Derivative(tan(x), x)),
    (r"\frac{d f(x)}{dx}", Derivative(f(x), x)),
    (r"\frac{d\theta(x)}{dx}", Derivative(Function('theta')(x), x))
]

TRIGONOMETRIC_EXPRESSION_PAIRS = [
    (r"\sin \theta", sin(theta)),
    (r"\sin(\theta)", sin(theta)),
    (r"\sin^{-1} a", asin(a)),
    (r"\sin a \cos b", _Mul(sin(a), cos(b))),
    (r"\sin \cos \theta", sin(cos(theta))),
    (r"\sin(\cos \theta)", sin(cos(theta))),
    (r"(\csc x)(\sec y)", csc(x) * sec(y)),
    (r"\frac{\sin{x}}2", _Mul(sin(x), _Pow(2, -1)))
]

UNEVALUATED_LIMIT_EXPRESSION_PAIRS = [
    (r"\lim_{x \to 3} a", Limit(a, x, 3, dir="+-")),
    (r"\lim_{x \rightarrow 3} a", Limit(a, x, 3, dir="+-")),
    (r"\lim_{x \Rightarrow 3} a", Limit(a, x, 3, dir="+-")),
    (r"\lim_{x \longrightarrow 3} a", Limit(a, x, 3, dir="+-")),
    (r"\lim_{x \Longrightarrow 3} a", Limit(a, x, 3, dir="+-")),
    (r"\lim_{x \to 3^{+}} a", Limit(a, x, 3, dir="+")),
    (r"\lim_{x \to 3^{-}} a", Limit(a, x, 3, dir="-")),
    (r"\lim_{x \to 3^+} a", Limit(a, x, 3, dir="+")),
    (r"\lim_{x \to 3^-} a", Limit(a, x, 3, dir="-")),
    (r"\lim_{x \to \infty} \frac{1}{x}", Limit(_Mul(1, _Pow(x, -1)), x, oo))
]

EVALUATED_LIMIT_EXPRESSION_PAIRS = [
    (r"\lim_{x \to \infty} \frac{1}{x}", Limit(1 / x, x, oo))
]

UNEVALUATED_SQRT_EXPRESSION_PAIRS = [
    (r"\sqrt{x}", sqrt(x)),
    (r"\sqrt{x + b}", sqrt(_Add(x, b))),
    (r"\sqrt[3]{\sin x}", _Pow(sin(x), _Pow(3, -1))),
    # the above test needed to be handled differently than the ones below because root
    # acts differently if its second argument is a number
    (r"\sqrt[y]{\sin x}", root(sin(x), y)),
    (r"\sqrt[\theta]{\sin x}", root(sin(x), theta)),
    (r"\sqrt{\frac{12}{6}}", _Sqrt(_Mul(12, _Pow(6, -1))))
]

EVALUATED_SQRT_EXPRESSION_PAIRS = [
    (r"\sqrt{x}", sqrt(x)),
    (r"\sqrt{x + b}", sqrt(x + b)),
    (r"\sqrt[3]{\sin x}", root(sin(x), 3)),
    (r"\sqrt[y]{\sin x}", root(sin(x), y)),
    (r"\sqrt[\theta]{\sin x}", root(sin(x), theta)),
    (r"\sqrt{\frac{12}{6}}", sqrt(2))
]

UNEVALUATED_FACTORIAL_EXPRESSION_PAIRS = [
    (r"x!", _factorial(x)),
    (r"100!", _factorial(100)),
    (r"\theta!", _factorial(theta)),
    (r"(x + 1)!", _factorial(_Add(x, 1))),
    (r"(x!)!", _factorial(_factorial(x))),
    (r"x!!!", _factorial(_factorial(_factorial(x)))),
    (r"5!7!", _Mul(_factorial(5), _factorial(7)))
]

EVALUATED_FACTORIAL_EXPRESSION_PAIRS = [
    (r"x!", factorial(x)),
    (r"100!", factorial(100)),
    (r"\theta!", factorial(theta)),
    (r"(x + 1)!", factorial(x + 1)),
    (r"(x!)!", factorial(factorial(x))),
    (r"x!!!", factorial(factorial(factorial(x)))),
    (r"5!7!", factorial(5) * factorial(7)),
    (r"24! \times 24!", factorial(24) * factorial(24))
]

UNEVALUATED_SUM_EXPRESSION_PAIRS = [
    (r"\sum_{k = 1}^{3} c", Sum(_Mul(1, c), (k, 1, 3))),
    (r"\sum_{k = 1}^3 c", Sum(_Mul(1, c), (k, 1, 3))),
    (r"\sum^{3}_{k = 1} c", Sum(_Mul(1, c), (k, 1, 3))),
    (r"\sum^3_{k = 1} c", Sum(_Mul(1, c), (k, 1, 3))),
    (r"\sum_{k = 1}^{10} k^2", Sum(_Mul(1, k ** 2), (k, 1, 10))),
    (r"\sum_{n = 0}^{\infty} \frac{1}{n!}",
     Sum(_Mul(1, _Mul(1, _Pow(_factorial(n), -1))), (n, 0, oo)))
]

EVALUATED_SUM_EXPRESSION_PAIRS = [
    (r"\sum_{k = 1}^{3} c", Sum(c, (k, 1, 3))),
    (r"\sum_{k = 1}^3 c", Sum(c, (k, 1, 3))),
    (r"\sum^{3}_{k = 1} c", Sum(c, (k, 1, 3))),
    (r"\sum^3_{k = 1} c", Sum(c, (k, 1, 3))),
    (r"\sum_{k = 1}^{10} k^2", Sum(k ** 2, (k, 1, 10))),
    (r"\sum_{n = 0}^{\infty} \frac{1}{n!}", Sum(1 / factorial(n), (n, 0, oo)))
]

UNEVALUATED_PRODUCT_EXPRESSION_PAIRS = [
    (r"\prod_{a = b}^{c} x", Product(x, (a, b, c))),
    (r"\prod_{a = b}^c x", Product(x, (a, b, c))),
    (r"\prod^{c}_{a = b} x", Product(x, (a, b, c))),
    (r"\prod^c_{a = b} x", Product(x, (a, b, c)))
]

APPLIED_FUNCTION_EXPRESSION_PAIRS = [
    (r"f(x)", f(x)),
    (r"f(x, y)", f(x, y)),
    (r"f(x, y, z)", f(x, y, z)),
    (r"f'_1(x)", Function("f_{1}'")(x)),
    (r"f_{1}''(x+y)", Function("f_{1}''")(x + y)),
    (r"h_{\theta}(x_0, x_1)",
     Function('h_{theta}')(Symbol('x_{0}'), Symbol('x_{1}')))
]

UNEVALUATED_COMMON_FUNCTION_EXPRESSION_PAIRS = [
    (r"|x|", _Abs(x)),
    (r"||x||", _Abs(Abs(x))),
    (r"|x||y|", _Abs(x) * _Abs(y)),
    (r"||x||y||", _Abs(_Abs(x) * _Abs(y))),
    (r"\lfloor x \rfloor", floor(x)),
    (r"\lceil x \rceil", ceiling(x)),
    (r"\exp x", _exp(x)),
    (r"\exp(x)", _exp(x)),
    (r"\lg x", _log(x, 10)),
    (r"\ln x", _log(x)),
    (r"\ln xy", _log(x * y)),
    (r"\log x", _log(x)),
    (r"\log xy", _log(x * y)),
    (r"\log_{2} x", _log(x, 2)),
    (r"\log_{a} x", _log(x, a)),
    (r"\log_{11} x", _log(x, 11)),
    (r"\log_{a^2} x", _log(x, _Pow(a, 2))),
    (r"\log_2 x", _log(x, 2)),
    (r"\log_a x", _log(x, a)),
    (r"\overline{z}", _Conjugate(z)),
    (r"\overline{\overline{z}}", _Conjugate(_Conjugate(z))),
    (r"\overline{x + y}", _Conjugate(_Add(x, y))),
    (r"\overline{x} + \overline{y}", _Conjugate(x) + _Conjugate(y)),
    (r"\min(a, b)", _Min(a, b)),
    (r"\min(a, b, c - d, xy)", _Min(a, b, c - d, x * y)),
    (r"\max(a, b)", _Max(a, b)),
    (r"\max(a, b, c - d, xy)", _Max(a, b, c - d, x * y)),
    # physics things don't have an `evaluate=False` variant
    (r"\langle x |", Bra('x')),
    (r"| x \rangle", Ket('x')),
    (r"\langle x | y \rangle", InnerProduct(Bra('x'), Ket('y'))),
]

EVALUATED_COMMON_FUNCTION_EXPRESSION_PAIRS = [
    (r"|x|", Abs(x)),
    (r"||x||", Abs(Abs(x))),
    (r"|x||y|", Abs(x) * Abs(y)),
    (r"||x||y||", Abs(Abs(x) * Abs(y))),
    (r"\lfloor x \rfloor", floor(x)),
    (r"\lceil x \rceil", ceiling(x)),
    (r"\exp x", exp(x)),
    (r"\exp(x)", exp(x)),
    (r"\lg x", log(x, 10)),
    (r"\ln x", log(x)),
    (r"\ln xy", log(x * y)),
    (r"\log x", log(x)),
    (r"\log xy", log(x * y)),
    (r"\log_{2} x", log(x, 2)),
    (r"\log_{a} x", log(x, a)),
    (r"\log_{11} x", log(x, 11)),
    (r"\log_{a^2} x", log(x, _Pow(a, 2))),
    (r"\log_2 x", log(x, 2)),
    (r"\log_a x", log(x, a)),
    (r"\overline{z}", conjugate(z)),
    (r"\overline{\overline{z}}", conjugate(conjugate(z))),
    (r"\overline{x + y}", conjugate(x + y)),
    (r"\overline{x} + \overline{y}", conjugate(x) + conjugate(y)),
    (r"\min(a, b)", Min(a, b)),
    (r"\min(a, b, c - d, xy)", Min(a, b, c - d, x * y)),
    (r"\max(a, b)", Max(a, b)),
    (r"\max(a, b, c - d, xy)", Max(a, b, c - d, x * y)),
    (r"\langle x |", Bra('x')),
    (r"| x \rangle", Ket('x')),
    (r"\langle x | y \rangle", InnerProduct(Bra('x'), Ket('y'))),
]

SPACING_RELATED_EXPRESSION_PAIRS = [
    (r"a \, b", _Mul(a, b)),
    (r"a \thinspace b", _Mul(a, b)),
    (r"a \: b", _Mul(a, b)),
    (r"a \medspace b", _Mul(a, b)),
    (r"a \; b", _Mul(a, b)),
    (r"a \thickspace b", _Mul(a, b)),
    (r"a \quad b", _Mul(a, b)),
    (r"a \qquad b", _Mul(a, b)),
    (r"a \! b", _Mul(a, b)),
    (r"a \negthinspace b", _Mul(a, b)),
    (r"a \negmedspace b", _Mul(a, b)),
    (r"a \negthickspace b", _Mul(a, b))
]

UNEVALUATED_BINOMIAL_EXPRESSION_PAIRS = [
    (r"\binom{n}{k}", _binomial(n, k)),
    (r"\tbinom{n}{k}", _binomial(n, k)),
    (r"\dbinom{n}{k}", _binomial(n, k)),
    (r"\binom{n}{0}", _binomial(n, 0)),
    (r"x^\binom{n}{k}", _Pow(x, _binomial(n, k)))
]

EVALUATED_BINOMIAL_EXPRESSION_PAIRS = [
    (r"\binom{n}{k}", binomial(n, k)),
    (r"\tbinom{n}{k}", binomial(n, k)),
    (r"\dbinom{n}{k}", binomial(n, k)),
    (r"\binom{n}{0}", binomial(n, 0)),
    (r"x^\binom{n}{k}", x ** binomial(n, k))
]

MISCELLANEOUS_EXPRESSION_PAIRS = [
    (r"\left(x + y\right) z", _Mul(_Add(x, y), z)),
    (r"\left( x + y\right ) z", _Mul(_Add(x, y), z)),
    (r"\left(  x + y\right ) z", _Mul(_Add(x, y), z)),
]

UNEVALUATED_LITERAL_COMPLEX_NUMBER_EXPRESSION_PAIRS = [
    (r"\imaginaryunit^2", _Pow(I, 2)),
    (r"|\imaginaryunit|", _Abs(I)),
    (r"\overline{\imaginaryunit}", _Conjugate(I)),
    (r"\imaginaryunit+\imaginaryunit", _Add(I, I)),
    (r"\imaginaryunit-\imaginaryunit", _Add(I, -I)),
    (r"\imaginaryunit*\imaginaryunit", _Mul(I, I)),
    (r"\imaginaryunit/\imaginaryunit", _Mul(I, _Pow(I, -1))),
    (r"(1+\imaginaryunit)/|1+\imaginaryunit|", _Mul(_Add(1, I), _Pow(_Abs(_Add(1, I)), -1)))
]

UNEVALUATED_MATRIX_EXPRESSION_PAIRS = [
    (r"\begin{pmatrix}a & b \\x & y\end{pmatrix}",
     Matrix([[a, b], [x, y]])),
    (r"\begin{pmatrix}a & b \\x & y\\\end{pmatrix}",
     Matrix([[a, b], [x, y]])),
    (r"\begin{bmatrix}a & b \\x & y\end{bmatrix}",
     Matrix([[a, b], [x, y]])),
    (r"\left(\begin{matrix}a & b \\x & y\end{matrix}\right)",
     Matrix([[a, b], [x, y]])),
    (r"\left[\begin{matrix}a & b \\x & y\end{matrix}\right]",
     Matrix([[a, b], [x, y]])),
    (r"\left[\begin{array}{cc}a & b \\x & y\end{array}\right]",
     Matrix([[a, b], [x, y]])),
    (r"\left(\begin{array}{cc}a & b \\x & y\end{array}\right)",
     Matrix([[a, b], [x, y]])),
    (r"\left( { \begin{array}{cc}a & b \\x & y\end{array} } \right)",
     Matrix([[a, b], [x, y]])),
    (r"+\begin{pmatrix}a & b \\x & y\end{pmatrix}",
     Matrix([[a, b], [x, y]])),
    ((r"\begin{pmatrix}x & y \\a & b\end{pmatrix}+"
      r"\begin{pmatrix}a & b \\x & y\end{pmatrix}"),
     _MatAdd(Matrix([[x, y], [a, b]]), Matrix([[a, b], [x, y]]))),
    (r"-\begin{pmatrix}a & b \\x & y\end{pmatrix}",
     _MatMul(-1, Matrix([[a, b], [x, y]]))),
    ((r"\begin{pmatrix}x & y \\a & b\end{pmatrix}-"
      r"\begin{pmatrix}a & b \\x & y\end{pmatrix}"),
     _MatAdd(Matrix([[x, y], [a, b]]), _MatMul(-1, Matrix([[a, b], [x, y]])))),
    ((r"\begin{pmatrix}a & b & c \\x & y & z \\a & b & c \end{pmatrix}*"
      r"\begin{pmatrix}x & y & z \\a & b & c \\a & b & c \end{pmatrix}*"
      r"\begin{pmatrix}a & b & c \\x & y & z \\x & y & z \end{pmatrix}"),
     _MatMul(_MatMul(Matrix([[a, b, c], [x, y, z], [a, b, c]]),
                     Matrix([[x, y, z], [a, b, c], [a, b, c]])),
             Matrix([[a, b, c], [x, y, z], [x, y, z]]))),
    (r"\begin{pmatrix}a & b \\x & y\end{pmatrix}/2",
     _MatMul(Matrix([[a, b], [x, y]]), _Pow(2, -1))),
    (r"\begin{pmatrix}a & b \\x & y\end{pmatrix}^2",
     _Pow(Matrix([[a, b], [x, y]]), 2)),
    (r"\begin{pmatrix}a & b \\x & y\end{pmatrix}^{-1}",
     _Pow(Matrix([[a, b], [x, y]]), -1)),
    (r"\begin{pmatrix}a & b \\x & y\end{pmatrix}^T",
     Transpose(Matrix([[a, b], [x, y]]))),
    (r"\begin{pmatrix}a & b \\x & y\end{pmatrix}^{T}",
     Transpose(Matrix([[a, b], [x, y]]))),
    (r"\begin{pmatrix}a & b \\x & y\end{pmatrix}^\mathit{T}",
     Transpose(Matrix([[a, b], [x, y]]))),
    (r"\begin{pmatrix}1 & 2 \\3 & 4\end{pmatrix}^T",
     Transpose(Matrix([[1, 2], [3, 4]]))),
    ((r"(\begin{pmatrix}1 & 2 \\3 & 4\end{pmatrix}+"
      r"\begin{pmatrix}1 & 2 \\3 & 4\end{pmatrix}^T)*"
      r"\begin{bmatrix}1\\0\end{bmatrix}"),
     _MatMul(_MatAdd(Matrix([[1, 2], [3, 4]]),
                     Transpose(Matrix([[1, 2], [3, 4]]))),
                Matrix([[1], [0]]))),
    ((r"(\begin{pmatrix}a & b \\x & y\end{pmatrix}+"
      r"\begin{pmatrix}x & y \\a & b\end{pmatrix})^2"),
    _Pow(_MatAdd(Matrix([[a, b], [x, y]]),
                 Matrix([[x, y], [a, b]])), 2)),
    ((r"(\begin{pmatrix}a & b \\x & y\end{pmatrix}+"
      r"\begin{pmatrix}x & y \\a & b\end{pmatrix})^T"),
    Transpose(_MatAdd(Matrix([[a, b], [x, y]]),
                      Matrix([[x, y], [a, b]])))),
    (r"\overline{\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}}",
     _Conjugate(_MatAdd(Matrix([[I, 2], [3, 4]]),
                        Matrix([[I, 2], [3, 4]]))))
]

EVALUATED_MATRIX_EXPRESSION_PAIRS = [
    (r"\det\left(\left[   { \begin{array}{cc}a&b\\x&y\end{array} } \right]\right)",
     Matrix([[a, b], [x, y]]).det()),
    (r"\det \begin{pmatrix}1&2\\3&4\end{pmatrix}", -2),
    (r"\det{\begin{pmatrix}1&2\\3&4\end{pmatrix}}", -2),
    (r"\det(\begin{pmatrix}1&2\\3&4\end{pmatrix})", -2),
    (r"\det\left(\begin{pmatrix}1&2\\3&4\end{pmatrix}\right)", -2),
    (r"\begin{pmatrix}a & b \\x & y\end{pmatrix}/\begin{vmatrix}a & b \\x & y\end{vmatrix}",
     _MatMul(Matrix([[a, b], [x, y]]), _Pow(Matrix([[a, b], [x, y]]).det(), -1))),
    (r"\begin{pmatrix}a & b \\x & y\end{pmatrix}/|\begin{matrix}a & b \\x & y\end{matrix}|",
     _MatMul(Matrix([[a, b], [x, y]]), _Pow(Matrix([[a, b], [x, y]]).det(), -1))),
    (r"\frac{\begin{pmatrix}a & b \\x & y\end{pmatrix}}{| { \begin{matrix}a & b \\x & y\end{matrix} } |}",
     _MatMul(Matrix([[a, b], [x, y]]), _Pow(Matrix([[a, b], [x, y]]).det(), -1))),
    (r"\overline{\begin{pmatrix}\imaginaryunit & 1+\imaginaryunit \\-\imaginaryunit & 4\end{pmatrix}}",
     Matrix([[-I, 1-I], [I, 4]])),
    (r"\begin{pmatrix}\imaginaryunit & 1+\imaginaryunit \\-\imaginaryunit & 4\end{pmatrix}^H",
     Matrix([[-I, I], [1-I, 4]])),
    (r"\trace(\begin{pmatrix}\imaginaryunit & 1+\imaginaryunit \\-\imaginaryunit & 4\end{pmatrix})",
     Trace(Matrix([[I, 1+I], [-I, 4]]))),
    (r"\adjugate(\begin{pmatrix}1 & 2 \\3 & 4\end{pmatrix})",
     Matrix([[4, -2], [-3, 1]])),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})^\ast",
     Matrix([[-2*I, 6], [4, 8]])),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})^{\ast}",
     Matrix([[-2*I, 6], [4, 8]])),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})^{\ast\ast}",
     Matrix([[2*I, 4], [6, 8]])),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})^{\ast\ast\ast}",
     Matrix([[-2*I, 6], [4, 8]])),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})^{*}",
     Matrix([[-2*I, 6], [4, 8]])),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})^{**}",
     Matrix([[2*I, 4], [6, 8]])),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})^{***}",
     Matrix([[-2*I, 6], [4, 8]])),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})^\prime",
     Transpose(_MatAdd(Matrix([[I, 2], [3, 4]]),
                       Matrix([[I, 2], [3, 4]])))),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})^{\prime}",
     Transpose(_MatAdd(Matrix([[I, 2], [3, 4]]),
                       Matrix([[I, 2], [3, 4]])))),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})^{\prime\prime}",
     _MatAdd(Matrix([[I, 2], [3, 4]]),
             Matrix([[I, 2], [3, 4]]))),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})^{\prime\prime\prime}",
     Transpose(_MatAdd(Matrix([[I, 2], [3, 4]]),
                       Matrix([[I, 2], [3, 4]])))),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})^{'}",
     Transpose(_MatAdd(Matrix([[I, 2], [3, 4]]),
                       Matrix([[I, 2], [3, 4]])))),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})^{''}",
     _MatAdd(Matrix([[I, 2], [3, 4]]),
             Matrix([[I, 2], [3, 4]]))),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})^{'''}",
     Transpose(_MatAdd(Matrix([[I, 2], [3, 4]]),
                       Matrix([[I, 2], [3, 4]])))),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})'",
     Transpose(_MatAdd(Matrix([[I, 2], [3, 4]]),
                       Matrix([[I, 2], [3, 4]])))),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})''",
     _MatAdd(Matrix([[I, 2], [3, 4]]),
             Matrix([[I, 2], [3, 4]]))),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})'''",
     Transpose(_MatAdd(Matrix([[I, 2], [3, 4]]),
                       Matrix([[I, 2], [3, 4]])))),
    (r"\det(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})",
     (_MatAdd(Matrix([[I, 2], [3, 4]]),
              Matrix([[I, 2], [3, 4]]))).det()),
    (r"\trace(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})",
     Trace(_MatAdd(Matrix([[I, 2], [3, 4]]),
                   Matrix([[I, 2], [3, 4]])))),
    (r"\adjugate(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})",
     (Matrix([[8, -4], [-6, 2*I]]))),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})^T",
     Transpose(_MatAdd(Matrix([[I, 2], [3, 4]]),
                       Matrix([[I, 2], [3, 4]])))),
    (r"(\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix}+\begin{pmatrix}\imaginaryunit&2\\3&4\end{pmatrix})^H",
     (Matrix([[-2*I, 6], [4, 8]])))
]


def test_symbol_expressions():
    expected_failures = {6, 7}
    for i, (latex_str, sympy_expr) in enumerate(SYMBOL_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_simple_expressions():
    expected_failures = {20}
    for i, (latex_str, sympy_expr) in enumerate(UNEVALUATED_SIMPLE_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for i, (latex_str, sympy_expr) in enumerate(EVALUATED_SIMPLE_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_fraction_expressions():
    for latex_str, sympy_expr in UNEVALUATED_FRACTION_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for latex_str, sympy_expr in EVALUATED_FRACTION_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_relation_expressions():
    for latex_str, sympy_expr in RELATION_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_power_expressions():
    expected_failures = {3}
    for i, (latex_str, sympy_expr) in enumerate(UNEVALUATED_POWER_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for i, (latex_str, sympy_expr) in enumerate(EVALUATED_POWER_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_integral_expressions():
    expected_failures = {14}
    for i, (latex_str, sympy_expr) in enumerate(UNEVALUATED_INTEGRAL_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, i

    for i, (latex_str, sympy_expr) in enumerate(EVALUATED_INTEGRAL_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_derivative_expressions():
    expected_failures = {3, 4}
    for i, (latex_str, sympy_expr) in enumerate(DERIVATIVE_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for i, (latex_str, sympy_expr) in enumerate(DERIVATIVE_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_trigonometric_expressions():
    expected_failures = {3}
    for i, (latex_str, sympy_expr) in enumerate(TRIGONOMETRIC_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_limit_expressions():
    for latex_str, sympy_expr in UNEVALUATED_LIMIT_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_square_root_expressions():
    for latex_str, sympy_expr in UNEVALUATED_SQRT_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for latex_str, sympy_expr in EVALUATED_SQRT_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_factorial_expressions():
    for latex_str, sympy_expr in UNEVALUATED_FACTORIAL_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for latex_str, sympy_expr in EVALUATED_FACTORIAL_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_sum_expressions():
    for latex_str, sympy_expr in UNEVALUATED_SUM_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for latex_str, sympy_expr in EVALUATED_SUM_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_product_expressions():
    for latex_str, sympy_expr in UNEVALUATED_PRODUCT_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

@XFAIL
def test_applied_function_expressions():
    expected_failures = {0, 3, 4}  # 0 is ambiguous, and the others require not-yet-added features
    # not sure why 1, and 2 are failing
    for i, (latex_str, sympy_expr) in enumerate(APPLIED_FUNCTION_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_common_function_expressions():
    for latex_str, sympy_expr in UNEVALUATED_COMMON_FUNCTION_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for latex_str, sympy_expr in EVALUATED_COMMON_FUNCTION_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


# unhandled bug causing these to fail
@XFAIL
def test_spacing():
    for latex_str, sympy_expr in SPACING_RELATED_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_binomial_expressions():
    for latex_str, sympy_expr in UNEVALUATED_BINOMIAL_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for latex_str, sympy_expr in EVALUATED_BINOMIAL_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_miscellaneous_expressions():
    for latex_str, sympy_expr in MISCELLANEOUS_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_literal_complex_number_expressions():
    for latex_str, sympy_expr in UNEVALUATED_LITERAL_COMPLEX_NUMBER_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str


def test_matrix_expressions():
    for latex_str, sympy_expr in UNEVALUATED_MATRIX_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

    for latex_str, sympy_expr in EVALUATED_MATRIX_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_interpolate.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Index,
    MultiIndex,
    Series,
    date_range,
    isna,
)
import pandas._testing as tm


@pytest.fixture(
    params=[
        "linear",
        "index",
        "values",
        "nearest",
        "slinear",
        "zero",
        "quadratic",
        "cubic",
        "barycentric",
        "krogh",
        "polynomial",
        "spline",
        "piecewise_polynomial",
        "from_derivatives",
        "pchip",
        "akima",
        "cubicspline",
    ]
)
def nontemporal_method(request):
    """Fixture that returns an (method name, required kwargs) pair.

    This fixture does not include method 'time' as a parameterization; that
    method requires a Series with a DatetimeIndex, and is generally tested
    separately from these non-temporal methods.
    """
    method = request.param
    kwargs = {"order": 1} if method in ("spline", "polynomial") else {}
    return method, kwargs


@pytest.fixture(
    params=[
        "linear",
        "slinear",
        "zero",
        "quadratic",
        "cubic",
        "barycentric",
        "krogh",
        "polynomial",
        "spline",
        "piecewise_polynomial",
        "from_derivatives",
        "pchip",
        "akima",
        "cubicspline",
    ]
)
def interp_methods_ind(request):
    """Fixture that returns a (method name, required kwargs) pair to
    be tested for various Index types.

    This fixture does not include methods - 'time', 'index', 'nearest',
    'values' as a parameterization
    """
    method = request.param
    kwargs = {"order": 1} if method in ("spline", "polynomial") else {}
    return method, kwargs


class TestSeriesInterpolateData:
    @pytest.mark.xfail(reason="EA.fillna does not handle 'linear' method")
    def test_interpolate_period_values(self):
        orig = Series(date_range("2012-01-01", periods=5))
        ser = orig.copy()
        ser[2] = pd.NaT

        # period cast
        ser_per = ser.dt.to_period("D")
        res_per = ser_per.interpolate()
        expected_per = orig.dt.to_period("D")
        tm.assert_series_equal(res_per, expected_per)

    def test_interpolate(self, datetime_series):
        ts = Series(np.arange(len(datetime_series), dtype=float), datetime_series.index)

        ts_copy = ts.copy()
        ts_copy[5:10] = np.nan

        linear_interp = ts_copy.interpolate(method="linear")
        tm.assert_series_equal(linear_interp, ts)

        ord_ts = Series(
            [d.toordinal() for d in datetime_series.index], index=datetime_series.index
        ).astype(float)

        ord_ts_copy = ord_ts.copy()
        ord_ts_copy[5:10] = np.nan

        time_interp = ord_ts_copy.interpolate(method="time")
        tm.assert_series_equal(time_interp, ord_ts)

    def test_interpolate_time_raises_for_non_timeseries(self):
        # When method='time' is used on a non-TimeSeries that contains a null
        # value, a ValueError should be raised.
        non_ts = Series([0, 1, 2, np.nan])
        msg = "time-weighted interpolation only works on Series.* with a DatetimeIndex"
        with pytest.raises(ValueError, match=msg):
            non_ts.interpolate(method="time")

    def test_interpolate_cubicspline(self):
        pytest.importorskip("scipy")
        ser = Series([10, 11, 12, 13])

        expected = Series(
            [11.00, 11.25, 11.50, 11.75, 12.00, 12.25, 12.50, 12.75, 13.00],
            index=Index([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]),
        )
        # interpolate at new_index
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75])).astype(
            float
        )
        result = ser.reindex(new_index).interpolate(method="cubicspline").loc[1:3]
        tm.assert_series_equal(result, expected)

    def test_interpolate_pchip(self):
        pytest.importorskip("scipy")
        ser = Series(np.sort(np.random.default_rng(2).uniform(size=100)))

        # interpolate at new_index
        new_index = ser.index.union(
            Index([49.25, 49.5, 49.75, 50.25, 50.5, 50.75])
        ).astype(float)
        interp_s = ser.reindex(new_index).interpolate(method="pchip")
        # does not blow up, GH5977
        interp_s.loc[49:51]

    def test_interpolate_akima(self):
        pytest.importorskip("scipy")
        ser = Series([10, 11, 12, 13])

        # interpolate at new_index where `der` is zero
        expected = Series(
            [11.00, 11.25, 11.50, 11.75, 12.00, 12.25, 12.50, 12.75, 13.00],
            index=Index([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]),
        )
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75])).astype(
            float
        )
        interp_s = ser.reindex(new_index).interpolate(method="akima")
        tm.assert_series_equal(interp_s.loc[1:3], expected)

        # interpolate at new_index where `der` is a non-zero int
        expected = Series(
            [11.0, 1.0, 1.0, 1.0, 12.0, 1.0, 1.0, 1.0, 13.0],
            index=Index([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]),
        )
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75])).astype(
            float
        )
        interp_s = ser.reindex(new_index).interpolate(method="akima", der=1)
        tm.assert_series_equal(interp_s.loc[1:3], expected)

    def test_interpolate_piecewise_polynomial(self):
        pytest.importorskip("scipy")
        ser = Series([10, 11, 12, 13])

        expected = Series(
            [11.00, 11.25, 11.50, 11.75, 12.00, 12.25, 12.50, 12.75, 13.00],
            index=Index([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]),
        )
        # interpolate at new_index
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75])).astype(
            float
        )
        interp_s = ser.reindex(new_index).interpolate(method="piecewise_polynomial")
        tm.assert_series_equal(interp_s.loc[1:3], expected)

    def test_interpolate_from_derivatives(self):
        pytest.importorskip("scipy")
        ser = Series([10, 11, 12, 13])

        expected = Series(
            [11.00, 11.25, 11.50, 11.75, 12.00, 12.25, 12.50, 12.75, 13.00],
            index=Index([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]),
        )
        # interpolate at new_index
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75])).astype(
            float
        )
        interp_s = ser.reindex(new_index).interpolate(method="from_derivatives")
        tm.assert_series_equal(interp_s.loc[1:3], expected)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            pytest.param(
                {"method": "polynomial", "order": 1}, marks=td.skip_if_no("scipy")
            ),
        ],
    )
    def test_interpolate_corners(self, kwargs):
        s = Series([np.nan, np.nan])
        tm.assert_series_equal(s.interpolate(**kwargs), s)

        s = Series([], dtype=object).interpolate()
        tm.assert_series_equal(s.interpolate(**kwargs), s)

    def test_interpolate_index_values(self):
        s = Series(np.nan, index=np.sort(np.random.default_rng(2).random(30)))
        s.loc[::3] = np.random.default_rng(2).standard_normal(10)

        vals = s.index.values.astype(float)

        result = s.interpolate(method="index")

        expected = s.copy()
        bad = isna(expected.values)
        good = ~bad
        expected = Series(
            np.interp(vals[bad], vals[good], s.values[good]), index=s.index[bad]
        )

        tm.assert_series_equal(result[bad], expected)

        # 'values' is synonymous with 'index' for the method kwarg
        other_result = s.interpolate(method="values")

        tm.assert_series_equal(other_result, result)
        tm.assert_series_equal(other_result[bad], expected)

    def test_interpolate_non_ts(self):
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])
        msg = (
            "time-weighted interpolation only works on Series or DataFrames "
            "with a DatetimeIndex"
        )
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method="time")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            pytest.param(
                {"method": "polynomial", "order": 1}, marks=td.skip_if_no("scipy")
            ),
        ],
    )
    def test_nan_interpolate(self, kwargs):
        s = Series([0, 1, np.nan, 3])
        result = s.interpolate(**kwargs)
        expected = Series([0.0, 1.0, 2.0, 3.0])
        tm.assert_series_equal(result, expected)

    def test_nan_irregular_index(self):
        s = Series([1, 2, np.nan, 4], index=[1, 3, 5, 9])
        result = s.interpolate()
        expected = Series([1.0, 2.0, 3.0, 4.0], index=[1, 3, 5, 9])
        tm.assert_series_equal(result, expected)

    def test_nan_str_index(self):
        s = Series([0, 1, 2, np.nan], index=list("abcd"))
        result = s.interpolate()
        expected = Series([0.0, 1.0, 2.0, 2.0], index=list("abcd"))
        tm.assert_series_equal(result, expected)

    def test_interp_quad(self):
        pytest.importorskip("scipy")
        sq = Series([1, 4, np.nan, 16], index=[1, 2, 3, 4])
        result = sq.interpolate(method="quadratic")
        expected = Series([1.0, 4.0, 9.0, 16.0], index=[1, 2, 3, 4])
        tm.assert_series_equal(result, expected)

    def test_interp_scipy_basic(self):
        pytest.importorskip("scipy")
        s = Series([1, 3, np.nan, 12, np.nan, 25])
        # slinear
        expected = Series([1.0, 3.0, 7.5, 12.0, 18.5, 25.0])
        result = s.interpolate(method="slinear")
        tm.assert_series_equal(result, expected)

        msg = "The 'downcast' keyword in Series.interpolate is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = s.interpolate(method="slinear", downcast="infer")
        tm.assert_series_equal(result, expected)
        # nearest
        expected = Series([1, 3, 3, 12, 12, 25])
        result = s.interpolate(method="nearest")
        tm.assert_series_equal(result, expected.astype("float"))

        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = s.interpolate(method="nearest", downcast="infer")
        tm.assert_series_equal(result, expected)
        # zero
        expected = Series([1, 3, 3, 12, 12, 25])
        result = s.interpolate(method="zero")
        tm.assert_series_equal(result, expected.astype("float"))

        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = s.interpolate(method="zero", downcast="infer")
        tm.assert_series_equal(result, expected)
        # quadratic
        # GH #15662.
        expected = Series([1, 3.0, 6.823529, 12.0, 18.058824, 25.0])
        result = s.interpolate(method="quadratic")
        tm.assert_series_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = s.interpolate(method="quadratic", downcast="infer")
        tm.assert_series_equal(result, expected)
        # cubic
        expected = Series([1.0, 3.0, 6.8, 12.0, 18.2, 25.0])
        result = s.interpolate(method="cubic")
        tm.assert_series_equal(result, expected)

    def test_interp_limit(self):
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])

        expected = Series([1.0, 3.0, 5.0, 7.0, np.nan, 11.0])
        result = s.interpolate(method="linear", limit=2)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("limit", [-1, 0])
    def test_interpolate_invalid_nonpositive_limit(self, nontemporal_method, limit):
        # GH 9217: make sure limit is greater than zero.
        s = Series([1, 2, np.nan, 4])
        method, kwargs = nontemporal_method
        with pytest.raises(ValueError, match="Limit must be greater than 0"):
            s.interpolate(limit=limit, method=method, **kwargs)

    def test_interpolate_invalid_float_limit(self, nontemporal_method):
        # GH 9217: make sure limit is an integer.
        s = Series([1, 2, np.nan, 4])
        method, kwargs = nontemporal_method
        limit = 2.0
        with pytest.raises(ValueError, match="Limit must be an integer"):
            s.interpolate(limit=limit, method=method, **kwargs)

    @pytest.mark.parametrize("invalid_method", [None, "nonexistent_method"])
    def test_interp_invalid_method(self, invalid_method):
        s = Series([1, 3, np.nan, 12, np.nan, 25])

        msg = f"method must be one of.* Got '{invalid_method}' instead"
        if invalid_method is None:
            msg = "'method' should be a string, not None"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method=invalid_method)

        # When an invalid method and invalid limit (such as -1) are
        # provided, the error message reflects the invalid method.
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method=invalid_method, limit=-1)

    def test_interp_invalid_method_and_value(self):
        # GH#36624
        ser = Series([1, 3, np.nan, 12, np.nan, 25])

        msg = "'fill_value' is not a valid keyword for Series.interpolate"
        msg2 = "Series.interpolate with method=pad"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=msg2):
                ser.interpolate(fill_value=3, method="pad")

    def test_interp_limit_forward(self):
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])

        # Provide 'forward' (the default) explicitly here.
        expected = Series([1.0, 3.0, 5.0, 7.0, np.nan, 11.0])

        result = s.interpolate(method="linear", limit=2, limit_direction="forward")
        tm.assert_series_equal(result, expected)

        result = s.interpolate(method="linear", limit=2, limit_direction="FORWARD")
        tm.assert_series_equal(result, expected)

    def test_interp_unlimited(self):
        # these test are for issue #16282 default Limit=None is unlimited
        s = Series([np.nan, 1.0, 3.0, np.nan, np.nan, np.nan, 11.0, np.nan])
        expected = Series([1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 11.0])
        result = s.interpolate(method="linear", limit_direction="both")
        tm.assert_series_equal(result, expected)

        expected = Series([np.nan, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 11.0])
        result = s.interpolate(method="linear", limit_direction="forward")
        tm.assert_series_equal(result, expected)

        expected = Series([1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, np.nan])
        result = s.interpolate(method="linear", limit_direction="backward")
        tm.assert_series_equal(result, expected)

    def test_interp_limit_bad_direction(self):
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])

        msg = (
            r"Invalid limit_direction: expecting one of \['forward', "
            r"'backward', 'both'\], got 'abc'"
        )
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method="linear", limit=2, limit_direction="abc")

        # raises an error even if no limit is specified.
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method="linear", limit_direction="abc")

    # limit_area introduced GH #16284
    def test_interp_limit_area(self):
        # These tests are for issue #9218 -- fill NaNs in both directions.
        s = Series([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan])

        expected = Series([np.nan, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, np.nan, np.nan])
        result = s.interpolate(method="linear", limit_area="inside")
        tm.assert_series_equal(result, expected)

        expected = Series(
            [np.nan, np.nan, 3.0, 4.0, np.nan, np.nan, 7.0, np.nan, np.nan]
        )
        result = s.interpolate(method="linear", limit_area="inside", limit=1)
        tm.assert_series_equal(result, expected)

        expected = Series([np.nan, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, np.nan, np.nan])
        result = s.interpolate(
            method="linear", limit_area="inside", limit_direction="both", limit=1
        )
        tm.assert_series_equal(result, expected)

        expected = Series([np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, 7.0])
        result = s.interpolate(method="linear", limit_area="outside")
        tm.assert_series_equal(result, expected)

        expected = Series(
            [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, np.nan]
        )
        result = s.interpolate(method="linear", limit_area="outside", limit=1)
        tm.assert_series_equal(result, expected)

        expected = Series([np.nan, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, np.nan])
        result = s.interpolate(
            method="linear", limit_area="outside", limit_direction="both", limit=1
        )
        tm.assert_series_equal(result, expected)

        expected = Series([3.0, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan])
        result = s.interpolate(
            method="linear", limit_area="outside", limit_direction="backward"
        )
        tm.assert_series_equal(result, expected)

        # raises an error even if limit type is wrong.
        msg = r"Invalid limit_area: expecting one of \['inside', 'outside'\], got abc"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method="linear", limit_area="abc")

    @pytest.mark.parametrize(
        "method, limit_direction, expected",
        [
            ("pad", "backward", "forward"),
            ("ffill", "backward", "forward"),
            ("backfill", "forward", "backward"),
            ("bfill", "forward", "backward"),
            ("pad", "both", "forward"),
            ("ffill", "both", "forward"),
            ("backfill", "both", "backward"),
            ("bfill", "both", "backward"),
        ],
    )
    def test_interp_limit_direction_raises(self, method, limit_direction, expected):
        # https://github.com/pandas-dev/pandas/pull/34746
        s = Series([1, 2, 3])

        msg = f"`limit_direction` must be '{expected}' for method `{method}`"
        msg2 = "Series.interpolate with method="
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=msg2):
                s.interpolate(method=method, limit_direction=limit_direction)

    @pytest.mark.parametrize(
        "data, expected_data, kwargs",
        (
            (
                [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
                [np.nan, np.nan, 3.0, 3.0, 3.0, 3.0, 7.0, np.nan, np.nan],
                {"method": "pad", "limit_area": "inside"},
            ),
            (
                [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
                [np.nan, np.nan, 3.0, 3.0, np.nan, np.nan, 7.0, np.nan, np.nan],
                {"method": "pad", "limit_area": "inside", "limit": 1},
            ),
            (
                [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
                [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, 7.0],
                {"method": "pad", "limit_area": "outside"},
            ),
            (
                [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
                [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, np.nan],
                {"method": "pad", "limit_area": "outside", "limit": 1},
            ),
            (
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                {"method": "pad", "limit_area": "outside", "limit": 1},
            ),
            (
                range(5),
                range(5),
                {"method": "pad", "limit_area": "outside", "limit": 1},
            ),
        ),
    )
    def test_interp_limit_area_with_pad(self, data, expected_data, kwargs):
        # GH26796

        s = Series(data)
        expected = Series(expected_data)
        msg = "Series.interpolate with method=pad"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = s.interpolate(**kwargs)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "data, expected_data, kwargs",
        (
            (
                [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
                [np.nan, np.nan, 3.0, 7.0, 7.0, 7.0, 7.0, np.nan, np.nan],
                {"method": "bfill", "limit_area": "inside"},
            ),
            (
                [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
                [np.nan, np.nan, 3.0, np.nan, np.nan, 7.0, 7.0, np.nan, np.nan],
                {"method": "bfill", "limit_area": "inside", "limit": 1},
            ),
            (
                [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
                [3.0, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan],
                {"method": "bfill", "limit_area": "outside"},
            ),
            (
                [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
                [np.nan, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan],
                {"method": "bfill", "limit_area": "outside", "limit": 1},
            ),
        ),
    )
    def test_interp_limit_area_with_backfill(self, data, expected_data, kwargs):
        # GH26796

        s = Series(data)
        expected = Series(expected_data)
        msg = "Series.interpolate with method=bfill"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = s.interpolate(**kwargs)
        tm.assert_series_equal(result, expected)

    def test_interp_limit_direction(self):
        # These tests are for issue #9218 -- fill NaNs in both directions.
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])

        expected = Series([1.0, 3.0, np.nan, 7.0, 9.0, 11.0])
        result = s.interpolate(method="linear", limit=2, limit_direction="backward")
        tm.assert_series_equal(result, expected)

        expected = Series([1.0, 3.0, 5.0, np.nan, 9.0, 11.0])
        result = s.interpolate(method="linear", limit=1, limit_direction="both")
        tm.assert_series_equal(result, expected)

        # Check that this works on a longer series of nans.
        s = Series([1, 3, np.nan, np.nan, np.nan, 7, 9, np.nan, np.nan, 12, np.nan])

        expected = Series([1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 12.0])
        result = s.interpolate(method="linear", limit=2, limit_direction="both")
        tm.assert_series_equal(result, expected)

        expected = Series(
            [1.0, 3.0, 4.0, np.nan, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 12.0]
        )
        result = s.interpolate(method="linear", limit=1, limit_direction="both")
        tm.assert_series_equal(result, expected)

    def test_interp_limit_to_ends(self):
        # These test are for issue #10420 -- flow back to beginning.
        s = Series([np.nan, np.nan, 5, 7, 9, np.nan])

        expected = Series([5.0, 5.0, 5.0, 7.0, 9.0, np.nan])
        result = s.interpolate(method="linear", limit=2, limit_direction="backward")
        tm.assert_series_equal(result, expected)

        expected = Series([5.0, 5.0, 5.0, 7.0, 9.0, 9.0])
        result = s.interpolate(method="linear", limit=2, limit_direction="both")
        tm.assert_series_equal(result, expected)

    def test_interp_limit_before_ends(self):
        # These test are for issue #11115 -- limit ends properly.
        s = Series([np.nan, np.nan, 5, 7, np.nan, np.nan])

        expected = Series([np.nan, np.nan, 5.0, 7.0, 7.0, np.nan])
        result = s.interpolate(method="linear", limit=1, limit_direction="forward")
        tm.assert_series_equal(result, expected)

        expected = Series([np.nan, 5.0, 5.0, 7.0, np.nan, np.nan])
        result = s.interpolate(method="linear", limit=1, limit_direction="backward")
        tm.assert_series_equal(result, expected)

        expected = Series([np.nan, 5.0, 5.0, 7.0, 7.0, np.nan])
        result = s.interpolate(method="linear", limit=1, limit_direction="both")
        tm.assert_series_equal(result, expected)

    def test_interp_all_good(self):
        pytest.importorskip("scipy")
        s = Series([1, 2, 3])
        result = s.interpolate(method="polynomial", order=1)
        tm.assert_series_equal(result, s)

        # non-scipy
        result = s.interpolate()
        tm.assert_series_equal(result, s)

    @pytest.mark.parametrize(
        "check_scipy", [False, pytest.param(True, marks=td.skip_if_no("scipy"))]
    )
    def test_interp_multiIndex(self, check_scipy):
        idx = MultiIndex.from_tuples([(0, "a"), (1, "b"), (2, "c")])
        s = Series([1, 2, np.nan], index=idx)

        expected = s.copy()
        expected.loc[2] = 2
        result = s.interpolate()
        tm.assert_series_equal(result, expected)

        msg = "Only `method=linear` interpolation is supported on MultiIndexes"
        if check_scipy:
            with pytest.raises(ValueError, match=msg):
                s.interpolate(method="polynomial", order=1)

    def test_interp_nonmono_raise(self):
        pytest.importorskip("scipy")
        s = Series([1, np.nan, 3], index=[0, 2, 1])
        msg = "krogh interpolation requires that the index be monotonic"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method="krogh")

    @pytest.mark.parametrize("method", ["nearest", "pad"])
    def test_interp_datetime64(self, method, tz_naive_fixture):
        pytest.importorskip("scipy")
        df = Series(
            [1, np.nan, 3], index=date_range("1/1/2000", periods=3, tz=tz_naive_fixture)
        )
        warn = None if method == "nearest" else FutureWarning
        msg = "Series.interpolate with method=pad is deprecated"
        with tm.assert_produces_warning(warn, match=msg):
            result = df.interpolate(method=method)
        if warn is not None:
            # check the "use ffill instead" is equivalent
            alt = df.ffill()
            tm.assert_series_equal(result, alt)

        expected = Series(
            [1.0, 1.0, 3.0],
            index=date_range("1/1/2000", periods=3, tz=tz_naive_fixture),
        )
        tm.assert_series_equal(result, expected)

    def test_interp_pad_datetime64tz_values(self):
        # GH#27628 missing.interpolate_2d should handle datetimetz values
        dti = date_range("2015-04-05", periods=3, tz="US/Central")
        ser = Series(dti)
        ser[1] = pd.NaT

        msg = "Series.interpolate with method=pad is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ser.interpolate(method="pad")
        # check the "use ffill instead" is equivalent
        alt = ser.ffill()
        tm.assert_series_equal(result, alt)

        expected = Series(dti)
        expected[1] = expected[0]
        tm.assert_series_equal(result, expected)

    def test_interp_limit_no_nans(self):
        # GH 7173
        s = Series([1.0, 2.0, 3.0])
        result = s.interpolate(limit=1)
        expected = s
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("method", ["polynomial", "spline"])
    def test_no_order(self, method):
        # see GH-10633, GH-24014
        pytest.importorskip("scipy")
        s = Series([0, 1, np.nan, 3])
        msg = "You must specify the order of the spline or polynomial"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method=method)

    @pytest.mark.parametrize("order", [-1, -1.0, 0, 0.0, np.nan])
    def test_interpolate_spline_invalid_order(self, order):
        pytest.importorskip("scipy")
        s = Series([0, 1, np.nan, 3])
        msg = "order needs to be specified and greater than 0"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method="spline", order=order)

    def test_spline(self):
        pytest.importorskip("scipy")
        s = Series([1, 2, np.nan, 4, 5, np.nan, 7])
        result = s.interpolate(method="spline", order=1)
        expected = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        tm.assert_series_equal(result, expected)

    def test_spline_extrapolate(self):
        pytest.importorskip("scipy")
        s = Series([1, 2, 3, 4, np.nan, 6, np.nan])
        result3 = s.interpolate(method="spline", order=1, ext=3)
        expected3 = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0])
        tm.assert_series_equal(result3, expected3)

        result1 = s.interpolate(method="spline", order=1, ext=0)
        expected1 = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        tm.assert_series_equal(result1, expected1)

    def test_spline_smooth(self):
        pytest.importorskip("scipy")
        s = Series([1, 2, np.nan, 4, 5.1, np.nan, 7])
        assert (
            s.interpolate(method="spline", order=3, s=0)[5]
            != s.interpolate(method="spline", order=3)[5]
        )

    def test_spline_interpolation(self):
        # Explicit cast to float to avoid implicit cast when setting np.nan
        pytest.importorskip("scipy")
        s = Series(np.arange(10) ** 2, dtype="float")
        s[np.random.default_rng(2).integers(0, 9, 3)] = np.nan
        result1 = s.interpolate(method="spline", order=1)
        expected1 = s.interpolate(method="spline", order=1)
        tm.assert_series_equal(result1, expected1)

    def test_interp_timedelta64(self):
        # GH 6424
        df = Series([1, np.nan, 3], index=pd.to_timedelta([1, 2, 3]))
        result = df.interpolate(method="time")
        expected = Series([1.0, 2.0, 3.0], index=pd.to_timedelta([1, 2, 3]))
        tm.assert_series_equal(result, expected)

        # test for non uniform spacing
        df = Series([1, np.nan, 3], index=pd.to_timedelta([1, 2, 4]))
        result = df.interpolate(method="time")
        expected = Series([1.0, 1.666667, 3.0], index=pd.to_timedelta([1, 2, 4]))
        tm.assert_series_equal(result, expected)

    def test_series_interpolate_method_values(self):
        # GH#1646
        rng = date_range("1/1/2000", "1/20/2000", freq="D")
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

        ts[::2] = np.nan

        result = ts.interpolate(method="values")
        exp = ts.interpolate()
        tm.assert_series_equal(result, exp)

    def test_series_interpolate_intraday(self):
        # #1698
        index = date_range("1/1/2012", periods=4, freq="12D")
        ts = Series([0, 12, 24, 36], index)
        new_index = index.append(index + pd.DateOffset(days=1)).sort_values()

        exp = ts.reindex(new_index).interpolate(method="time")

        index = date_range("1/1/2012", periods=4, freq="12h")
        ts = Series([0, 12, 24, 36], index)
        new_index = index.append(index + pd.DateOffset(hours=1)).sort_values()
        result = ts.reindex(new_index).interpolate(method="time")

        tm.assert_numpy_array_equal(result.values, exp.values)

    @pytest.mark.parametrize(
        "ind",
        [
            ["a", "b", "c", "d"],
            pd.period_range(start="2019-01-01", periods=4),
            pd.interval_range(start=0, end=4),
        ],
    )
    def test_interp_non_timedelta_index(self, interp_methods_ind, ind):
        # gh 21662
        df = pd.DataFrame([0, 1, np.nan, 3], index=ind)

        method, kwargs = interp_methods_ind
        if method == "pchip":
            pytest.importorskip("scipy")

        if method == "linear":
            result = df[0].interpolate(**kwargs)
            expected = Series([0.0, 1.0, 2.0, 3.0], name=0, index=ind)
            tm.assert_series_equal(result, expected)
        else:
            expected_error = (
                "Index column must be numeric or datetime type when "
                f"using {method} method other than linear. "
                "Try setting a numeric or datetime index column before "
                "interpolating."
            )
            with pytest.raises(ValueError, match=expected_error):
                df[0].interpolate(method=method, **kwargs)

    def test_interpolate_timedelta_index(self, request, interp_methods_ind):
        """
        Tests for non numerical index types  - object, period, timedelta
        Note that all methods except time, index, nearest and values
        are tested here.
        """
        # gh 21662
        pytest.importorskip("scipy")
        ind = pd.timedelta_range(start=1, periods=4)
        df = pd.DataFrame([0, 1, np.nan, 3], index=ind)

        method, kwargs = interp_methods_ind

        if method in {"cubic", "zero"}:
            request.applymarker(
                pytest.mark.xfail(
                    reason=f"{method} interpolation is not supported for TimedeltaIndex"
                )
            )
        result = df[0].interpolate(method=method, **kwargs)
        expected = Series([0.0, 1.0, 2.0, 3.0], name=0, index=ind)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ascending, expected_values",
        [(True, [1, 2, 3, 9, 10]), (False, [10, 9, 3, 2, 1])],
    )
    def test_interpolate_unsorted_index(self, ascending, expected_values):
        # GH 21037
        ts = Series(data=[10, 9, np.nan, 2, 1], index=[10, 9, 3, 2, 1])
        result = ts.sort_index(ascending=ascending).interpolate(method="index")
        expected = Series(data=expected_values, index=expected_values, dtype=float)
        tm.assert_series_equal(result, expected)

    def test_interpolate_asfreq_raises(self):
        ser = Series(["a", None, "b"], dtype=object)
        msg2 = "Series.interpolate with object dtype"
        msg = "Invalid fill method"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=msg2):
                ser.interpolate(method="asfreq")

    def test_interpolate_fill_value(self):
        # GH#54920
        pytest.importorskip("scipy")
        ser = Series([np.nan, 0, 1, np.nan, 3, np.nan])
        result = ser.interpolate(method="nearest", fill_value=0)
        expected = Series([np.nan, 0, 1, 1, 3, 0])
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\test_common.py ===
from __future__ import annotations

import numpy as np
import pytest

from pandas.compat import HAS_PYARROW
import pandas.util._test_decorators as td

from pandas.core.dtypes.astype import astype_array
import pandas.core.dtypes.common as com
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    CategoricalDtypeType,
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    PeriodDtype,
)
from pandas.core.dtypes.missing import isna

import pandas as pd
import pandas._testing as tm
from pandas.api.types import pandas_dtype
from pandas.arrays import SparseArray
from pandas.util.version import Version


# EA & Actual Dtypes
def to_ea_dtypes(dtypes):
    """convert list of string dtypes to EA dtype"""
    return [getattr(pd, dt + "Dtype") for dt in dtypes]


def to_numpy_dtypes(dtypes):
    """convert list of string dtypes to numpy dtype"""
    return [getattr(np, dt) for dt in dtypes if isinstance(dt, str)]


class TestNumpyEADtype:
    # Passing invalid dtype, both as a string or object, must raise TypeError
    # Per issue GH15520
    @pytest.mark.parametrize("box", [pd.Timestamp, "pd.Timestamp", list])
    def test_invalid_dtype_error(self, box):
        with pytest.raises(TypeError, match="not understood"):
            com.pandas_dtype(box)

    @pytest.mark.parametrize(
        "dtype",
        [
            object,
            "float64",
            np.object_,
            np.dtype("object"),
            "O",
            np.float64,
            float,
            np.dtype("float64"),
            "object_",
        ],
    )
    def test_pandas_dtype_valid(self, dtype):
        assert com.pandas_dtype(dtype) == dtype

    @pytest.mark.parametrize(
        "dtype", ["M8[ns]", "m8[ns]", "object", "float64", "int64"]
    )
    def test_numpy_dtype(self, dtype):
        assert com.pandas_dtype(dtype) == np.dtype(dtype)

    def test_numpy_string_dtype(self):
        # do not parse freq-like string as period dtype
        assert com.pandas_dtype("U") == np.dtype("U")
        assert com.pandas_dtype("S") == np.dtype("S")

    @pytest.mark.parametrize(
        "dtype",
        [
            "datetime64[ns, US/Eastern]",
            "datetime64[ns, Asia/Tokyo]",
            "datetime64[ns, UTC]",
            # GH#33885 check that the M8 alias is understood
            "M8[ns, US/Eastern]",
            "M8[ns, Asia/Tokyo]",
            "M8[ns, UTC]",
        ],
    )
    def test_datetimetz_dtype(self, dtype):
        assert com.pandas_dtype(dtype) == DatetimeTZDtype.construct_from_string(dtype)
        assert com.pandas_dtype(dtype) == dtype

    def test_categorical_dtype(self):
        assert com.pandas_dtype("category") == CategoricalDtype()

    @pytest.mark.parametrize(
        "dtype",
        [
            "period[D]",
            "period[3M]",
            "period[us]",
            "Period[D]",
            "Period[3M]",
            "Period[us]",
        ],
    )
    def test_period_dtype(self, dtype):
        assert com.pandas_dtype(dtype) is not PeriodDtype(dtype)
        assert com.pandas_dtype(dtype) == PeriodDtype(dtype)
        assert com.pandas_dtype(dtype) == dtype


dtypes = {
    "datetime_tz": com.pandas_dtype("datetime64[ns, US/Eastern]"),
    "datetime": com.pandas_dtype("datetime64[ns]"),
    "timedelta": com.pandas_dtype("timedelta64[ns]"),
    "period": PeriodDtype("D"),
    "integer": np.dtype(np.int64),
    "float": np.dtype(np.float64),
    "object": np.dtype(object),
    "category": com.pandas_dtype("category"),
    "string": pd.StringDtype(),
}


@pytest.mark.parametrize("name1,dtype1", list(dtypes.items()), ids=lambda x: str(x))
@pytest.mark.parametrize("name2,dtype2", list(dtypes.items()), ids=lambda x: str(x))
def test_dtype_equal(name1, dtype1, name2, dtype2):
    # match equal to self, but not equal to other
    assert com.is_dtype_equal(dtype1, dtype1)
    if name1 != name2:
        assert not com.is_dtype_equal(dtype1, dtype2)


@pytest.mark.parametrize("name,dtype", list(dtypes.items()), ids=lambda x: str(x))
def test_pyarrow_string_import_error(name, dtype):
    # GH-44276
    assert not com.is_dtype_equal(dtype, "string[pyarrow]")


@pytest.mark.parametrize(
    "dtype1,dtype2",
    [
        (np.int8, np.int64),
        (np.int16, np.int64),
        (np.int32, np.int64),
        (np.float32, np.float64),
        (PeriodDtype("D"), PeriodDtype("2D")),  # PeriodType
        (
            com.pandas_dtype("datetime64[ns, US/Eastern]"),
            com.pandas_dtype("datetime64[ns, CET]"),
        ),  # Datetime
        (None, None),  # gh-15941: no exception should be raised.
    ],
)
def test_dtype_equal_strict(dtype1, dtype2):
    assert not com.is_dtype_equal(dtype1, dtype2)


def get_is_dtype_funcs():
    """
    Get all functions in pandas.core.dtypes.common that
    begin with 'is_' and end with 'dtype'

    """
    fnames = [f for f in dir(com) if (f.startswith("is_") and f.endswith("dtype"))]
    fnames.remove("is_string_or_object_np_dtype")  # fastpath requires np.dtype obj
    return [getattr(com, fname) for fname in fnames]


@pytest.mark.filterwarnings(
    "ignore:is_categorical_dtype is deprecated:DeprecationWarning"
)
@pytest.mark.parametrize("func", get_is_dtype_funcs(), ids=lambda x: x.__name__)
def test_get_dtype_error_catch(func):
    # see gh-15941
    #
    # No exception should be raised.

    msg = f"{func.__name__} is deprecated"
    warn = None
    if (
        func is com.is_int64_dtype
        or func is com.is_interval_dtype
        or func is com.is_datetime64tz_dtype
        or func is com.is_categorical_dtype
        or func is com.is_period_dtype
    ):
        warn = DeprecationWarning

    with tm.assert_produces_warning(warn, match=msg):
        assert not func(None)


def test_is_object():
    assert com.is_object_dtype(object)
    assert com.is_object_dtype(np.array([], dtype=object))

    assert not com.is_object_dtype(int)
    assert not com.is_object_dtype(np.array([], dtype=int))
    assert not com.is_object_dtype([1, 2, 3])


@pytest.mark.parametrize(
    "check_scipy", [False, pytest.param(True, marks=td.skip_if_no("scipy"))]
)
def test_is_sparse(check_scipy):
    msg = "is_sparse is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert com.is_sparse(SparseArray([1, 2, 3]))

        assert not com.is_sparse(np.array([1, 2, 3]))

        if check_scipy:
            import scipy.sparse

            assert not com.is_sparse(scipy.sparse.bsr_matrix([1, 2, 3]))


def test_is_scipy_sparse():
    sp_sparse = pytest.importorskip("scipy.sparse")

    assert com.is_scipy_sparse(sp_sparse.bsr_matrix([1, 2, 3]))

    assert not com.is_scipy_sparse(SparseArray([1, 2, 3]))


def test_is_datetime64_dtype():
    assert not com.is_datetime64_dtype(object)
    assert not com.is_datetime64_dtype([1, 2, 3])
    assert not com.is_datetime64_dtype(np.array([], dtype=int))

    assert com.is_datetime64_dtype(np.datetime64)
    assert com.is_datetime64_dtype(np.array([], dtype=np.datetime64))


def test_is_datetime64tz_dtype():
    msg = "is_datetime64tz_dtype is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert not com.is_datetime64tz_dtype(object)
        assert not com.is_datetime64tz_dtype([1, 2, 3])
        assert not com.is_datetime64tz_dtype(pd.DatetimeIndex([1, 2, 3]))
        assert com.is_datetime64tz_dtype(pd.DatetimeIndex(["2000"], tz="US/Eastern"))


def test_custom_ea_kind_M_not_datetime64tz():
    # GH 34986
    class NotTZDtype(ExtensionDtype):
        @property
        def kind(self) -> str:
            return "M"

    not_tz_dtype = NotTZDtype()
    msg = "is_datetime64tz_dtype is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert not com.is_datetime64tz_dtype(not_tz_dtype)
        assert not com.needs_i8_conversion(not_tz_dtype)


def test_is_timedelta64_dtype():
    assert not com.is_timedelta64_dtype(object)
    assert not com.is_timedelta64_dtype(None)
    assert not com.is_timedelta64_dtype([1, 2, 3])
    assert not com.is_timedelta64_dtype(np.array([], dtype=np.datetime64))
    assert not com.is_timedelta64_dtype("0 days")
    assert not com.is_timedelta64_dtype("0 days 00:00:00")
    assert not com.is_timedelta64_dtype(["0 days 00:00:00"])
    assert not com.is_timedelta64_dtype("NO DATE")

    assert com.is_timedelta64_dtype(np.timedelta64)
    assert com.is_timedelta64_dtype(pd.Series([], dtype="timedelta64[ns]"))
    assert com.is_timedelta64_dtype(pd.to_timedelta(["0 days", "1 days"]))


def test_is_period_dtype():
    msg = "is_period_dtype is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert not com.is_period_dtype(object)
        assert not com.is_period_dtype([1, 2, 3])
        assert not com.is_period_dtype(pd.Period("2017-01-01"))

        assert com.is_period_dtype(PeriodDtype(freq="D"))
        assert com.is_period_dtype(pd.PeriodIndex([], freq="Y"))


def test_is_interval_dtype():
    msg = "is_interval_dtype is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert not com.is_interval_dtype(object)
        assert not com.is_interval_dtype([1, 2, 3])

        assert com.is_interval_dtype(IntervalDtype())

        interval = pd.Interval(1, 2, closed="right")
        assert not com.is_interval_dtype(interval)
        assert com.is_interval_dtype(pd.IntervalIndex([interval]))


def test_is_categorical_dtype():
    msg = "is_categorical_dtype is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert not com.is_categorical_dtype(object)
        assert not com.is_categorical_dtype([1, 2, 3])

        assert com.is_categorical_dtype(CategoricalDtype())
        assert com.is_categorical_dtype(pd.Categorical([1, 2, 3]))
        assert com.is_categorical_dtype(pd.CategoricalIndex([1, 2, 3]))


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (int, False),
        (pd.Series([1, 2]), False),
        (str, True),
        (object, True),
        (np.array(["a", "b"]), True),
        (pd.StringDtype(), True),
        (pd.Index([], dtype="O"), True),
    ],
)
def test_is_string_dtype(dtype, expected):
    # GH#54661

    result = com.is_string_dtype(dtype)
    assert result is expected


@pytest.mark.parametrize(
    "data",
    [[(0, 1), (1, 1)], pd.Categorical([1, 2, 3]), np.array([1, 2], dtype=object)],
)
def test_is_string_dtype_arraylike_with_object_elements_not_strings(data):
    # GH 15585
    assert not com.is_string_dtype(pd.Series(data))


def test_is_string_dtype_nullable(nullable_string_dtype):
    assert com.is_string_dtype(pd.array(["a", "b"], dtype=nullable_string_dtype))


integer_dtypes: list = []


@pytest.mark.parametrize(
    "dtype",
    integer_dtypes
    + [pd.Series([1, 2])]
    + tm.ALL_INT_NUMPY_DTYPES
    + to_numpy_dtypes(tm.ALL_INT_NUMPY_DTYPES)
    + tm.ALL_INT_EA_DTYPES
    + to_ea_dtypes(tm.ALL_INT_EA_DTYPES),
)
def test_is_integer_dtype(dtype):
    assert com.is_integer_dtype(dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        str,
        float,
        np.datetime64,
        np.timedelta64,
        pd.Index([1, 2.0]),
        np.array(["a", "b"]),
        np.array([], dtype=np.timedelta64),
    ],
)
def test_is_not_integer_dtype(dtype):
    assert not com.is_integer_dtype(dtype)


signed_integer_dtypes: list = []


@pytest.mark.parametrize(
    "dtype",
    signed_integer_dtypes
    + [pd.Series([1, 2])]
    + tm.SIGNED_INT_NUMPY_DTYPES
    + to_numpy_dtypes(tm.SIGNED_INT_NUMPY_DTYPES)
    + tm.SIGNED_INT_EA_DTYPES
    + to_ea_dtypes(tm.SIGNED_INT_EA_DTYPES),
)
def test_is_signed_integer_dtype(dtype):
    assert com.is_integer_dtype(dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        str,
        float,
        np.datetime64,
        np.timedelta64,
        pd.Index([1, 2.0]),
        np.array(["a", "b"]),
        np.array([], dtype=np.timedelta64),
    ]
    + tm.UNSIGNED_INT_NUMPY_DTYPES
    + to_numpy_dtypes(tm.UNSIGNED_INT_NUMPY_DTYPES)
    + tm.UNSIGNED_INT_EA_DTYPES
    + to_ea_dtypes(tm.UNSIGNED_INT_EA_DTYPES),
)
def test_is_not_signed_integer_dtype(dtype):
    assert not com.is_signed_integer_dtype(dtype)


unsigned_integer_dtypes: list = []


@pytest.mark.parametrize(
    "dtype",
    unsigned_integer_dtypes
    + [pd.Series([1, 2], dtype=np.uint32)]
    + tm.UNSIGNED_INT_NUMPY_DTYPES
    + to_numpy_dtypes(tm.UNSIGNED_INT_NUMPY_DTYPES)
    + tm.UNSIGNED_INT_EA_DTYPES
    + to_ea_dtypes(tm.UNSIGNED_INT_EA_DTYPES),
)
def test_is_unsigned_integer_dtype(dtype):
    assert com.is_unsigned_integer_dtype(dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        str,
        float,
        np.datetime64,
        np.timedelta64,
        pd.Index([1, 2.0]),
        np.array(["a", "b"]),
        np.array([], dtype=np.timedelta64),
    ]
    + tm.SIGNED_INT_NUMPY_DTYPES
    + to_numpy_dtypes(tm.SIGNED_INT_NUMPY_DTYPES)
    + tm.SIGNED_INT_EA_DTYPES
    + to_ea_dtypes(tm.SIGNED_INT_EA_DTYPES),
)
def test_is_not_unsigned_integer_dtype(dtype):
    assert not com.is_unsigned_integer_dtype(dtype)


@pytest.mark.parametrize(
    "dtype", [np.int64, np.array([1, 2], dtype=np.int64), "Int64", pd.Int64Dtype]
)
def test_is_int64_dtype(dtype):
    msg = "is_int64_dtype is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert com.is_int64_dtype(dtype)


def test_type_comparison_with_numeric_ea_dtype(any_numeric_ea_dtype):
    # GH#43038
    assert pandas_dtype(any_numeric_ea_dtype) == any_numeric_ea_dtype


def test_type_comparison_with_real_numpy_dtype(any_real_numpy_dtype):
    # GH#43038
    assert pandas_dtype(any_real_numpy_dtype) == any_real_numpy_dtype


def test_type_comparison_with_signed_int_ea_dtype_and_signed_int_numpy_dtype(
    any_signed_int_ea_dtype, any_signed_int_numpy_dtype
):
    # GH#43038
    assert not pandas_dtype(any_signed_int_ea_dtype) == any_signed_int_numpy_dtype


@pytest.mark.parametrize(
    "dtype",
    [
        str,
        float,
        np.int32,
        np.uint64,
        pd.Index([1, 2.0]),
        np.array(["a", "b"]),
        np.array([1, 2], dtype=np.uint32),
        "int8",
        "Int8",
        pd.Int8Dtype,
    ],
)
def test_is_not_int64_dtype(dtype):
    msg = "is_int64_dtype is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert not com.is_int64_dtype(dtype)


def test_is_datetime64_any_dtype():
    assert not com.is_datetime64_any_dtype(int)
    assert not com.is_datetime64_any_dtype(str)
    assert not com.is_datetime64_any_dtype(np.array([1, 2]))
    assert not com.is_datetime64_any_dtype(np.array(["a", "b"]))

    assert com.is_datetime64_any_dtype(np.datetime64)
    assert com.is_datetime64_any_dtype(np.array([], dtype=np.datetime64))
    assert com.is_datetime64_any_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    assert com.is_datetime64_any_dtype(
        pd.DatetimeIndex([1, 2, 3], dtype="datetime64[ns]")
    )


def test_is_datetime64_ns_dtype():
    assert not com.is_datetime64_ns_dtype(int)
    assert not com.is_datetime64_ns_dtype(str)
    assert not com.is_datetime64_ns_dtype(np.datetime64)
    assert not com.is_datetime64_ns_dtype(np.array([1, 2]))
    assert not com.is_datetime64_ns_dtype(np.array(["a", "b"]))
    assert not com.is_datetime64_ns_dtype(np.array([], dtype=np.datetime64))

    # This datetime array has the wrong unit (ps instead of ns)
    assert not com.is_datetime64_ns_dtype(np.array([], dtype="datetime64[ps]"))

    assert com.is_datetime64_ns_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    assert com.is_datetime64_ns_dtype(
        pd.DatetimeIndex([1, 2, 3], dtype=np.dtype("datetime64[ns]"))
    )

    # non-nano dt64tz
    assert not com.is_datetime64_ns_dtype(DatetimeTZDtype("us", "US/Eastern"))


def test_is_timedelta64_ns_dtype():
    assert not com.is_timedelta64_ns_dtype(np.dtype("m8[ps]"))
    assert not com.is_timedelta64_ns_dtype(np.array([1, 2], dtype=np.timedelta64))

    assert com.is_timedelta64_ns_dtype(np.dtype("m8[ns]"))
    assert com.is_timedelta64_ns_dtype(np.array([1, 2], dtype="m8[ns]"))


def test_is_numeric_v_string_like():
    assert not com.is_numeric_v_string_like(np.array([1]), 1)
    assert not com.is_numeric_v_string_like(np.array([1]), np.array([2]))
    assert not com.is_numeric_v_string_like(np.array(["foo"]), np.array(["foo"]))

    assert com.is_numeric_v_string_like(np.array([1]), "foo")
    assert com.is_numeric_v_string_like(np.array([1, 2]), np.array(["foo"]))
    assert com.is_numeric_v_string_like(np.array(["foo"]), np.array([1, 2]))


def test_needs_i8_conversion():
    assert not com.needs_i8_conversion(str)
    assert not com.needs_i8_conversion(np.int64)
    assert not com.needs_i8_conversion(pd.Series([1, 2]))
    assert not com.needs_i8_conversion(np.array(["a", "b"]))

    assert not com.needs_i8_conversion(np.datetime64)
    assert com.needs_i8_conversion(np.dtype(np.datetime64))
    assert not com.needs_i8_conversion(pd.Series([], dtype="timedelta64[ns]"))
    assert com.needs_i8_conversion(pd.Series([], dtype="timedelta64[ns]").dtype)
    assert not com.needs_i8_conversion(pd.DatetimeIndex(["2000"], tz="US/Eastern"))
    assert com.needs_i8_conversion(pd.DatetimeIndex(["2000"], tz="US/Eastern").dtype)


def test_is_numeric_dtype():
    assert not com.is_numeric_dtype(str)
    assert not com.is_numeric_dtype(np.datetime64)
    assert not com.is_numeric_dtype(np.timedelta64)
    assert not com.is_numeric_dtype(np.array(["a", "b"]))
    assert not com.is_numeric_dtype(np.array([], dtype=np.timedelta64))

    assert com.is_numeric_dtype(int)
    assert com.is_numeric_dtype(float)
    assert com.is_numeric_dtype(np.uint64)
    assert com.is_numeric_dtype(pd.Series([1, 2]))
    assert com.is_numeric_dtype(pd.Index([1, 2.0]))

    class MyNumericDType(ExtensionDtype):
        @property
        def type(self):
            return str

        @property
        def name(self):
            raise NotImplementedError

        @classmethod
        def construct_array_type(cls):
            raise NotImplementedError

        def _is_numeric(self) -> bool:
            return True

    assert com.is_numeric_dtype(MyNumericDType())


def test_is_any_real_numeric_dtype():
    assert not com.is_any_real_numeric_dtype(str)
    assert not com.is_any_real_numeric_dtype(bool)
    assert not com.is_any_real_numeric_dtype(complex)
    assert not com.is_any_real_numeric_dtype(object)
    assert not com.is_any_real_numeric_dtype(np.datetime64)
    assert not com.is_any_real_numeric_dtype(np.array(["a", "b", complex(1, 2)]))
    assert not com.is_any_real_numeric_dtype(pd.DataFrame([complex(1, 2), True]))

    assert com.is_any_real_numeric_dtype(int)
    assert com.is_any_real_numeric_dtype(float)
    assert com.is_any_real_numeric_dtype(np.array([1, 2.5]))


def test_is_float_dtype():
    assert not com.is_float_dtype(str)
    assert not com.is_float_dtype(int)
    assert not com.is_float_dtype(pd.Series([1, 2]))
    assert not com.is_float_dtype(np.array(["a", "b"]))

    assert com.is_float_dtype(float)
    assert com.is_float_dtype(pd.Index([1, 2.0]))


def test_is_bool_dtype():
    assert not com.is_bool_dtype(int)
    assert not com.is_bool_dtype(str)
    assert not com.is_bool_dtype(pd.Series([1, 2]))
    assert not com.is_bool_dtype(pd.Series(["a", "b"], dtype="category"))
    assert not com.is_bool_dtype(np.array(["a", "b"]))
    assert not com.is_bool_dtype(pd.Index(["a", "b"]))
    assert not com.is_bool_dtype("Int64")

    assert com.is_bool_dtype(bool)
    assert com.is_bool_dtype(np.bool_)
    assert com.is_bool_dtype(pd.Series([True, False], dtype="category"))
    assert com.is_bool_dtype(np.array([True, False]))
    assert com.is_bool_dtype(pd.Index([True, False]))

    assert com.is_bool_dtype(pd.BooleanDtype())
    assert com.is_bool_dtype(pd.array([True, False, None], dtype="boolean"))
    assert com.is_bool_dtype("boolean")


def test_is_bool_dtype_numpy_error():
    # GH39010
    assert not com.is_bool_dtype("0 - Name")


@pytest.mark.parametrize(
    "check_scipy", [False, pytest.param(True, marks=td.skip_if_no("scipy"))]
)
def test_is_extension_array_dtype(check_scipy):
    assert not com.is_extension_array_dtype([1, 2, 3])
    assert not com.is_extension_array_dtype(np.array([1, 2, 3]))
    assert not com.is_extension_array_dtype(pd.DatetimeIndex([1, 2, 3]))

    cat = pd.Categorical([1, 2, 3])
    assert com.is_extension_array_dtype(cat)
    assert com.is_extension_array_dtype(pd.Series(cat))
    assert com.is_extension_array_dtype(SparseArray([1, 2, 3]))
    assert com.is_extension_array_dtype(pd.DatetimeIndex(["2000"], tz="US/Eastern"))

    dtype = DatetimeTZDtype("ns", tz="US/Eastern")
    s = pd.Series([], dtype=dtype)
    assert com.is_extension_array_dtype(s)

    if check_scipy:
        import scipy.sparse

        assert not com.is_extension_array_dtype(scipy.sparse.bsr_matrix([1, 2, 3]))


def test_is_complex_dtype():
    assert not com.is_complex_dtype(int)
    assert not com.is_complex_dtype(str)
    assert not com.is_complex_dtype(pd.Series([1, 2]))
    assert not com.is_complex_dtype(np.array(["a", "b"]))

    assert com.is_complex_dtype(np.complex128)
    assert com.is_complex_dtype(complex)
    assert com.is_complex_dtype(np.array([1 + 1j, 5]))


@pytest.mark.parametrize(
    "input_param,result",
    [
        (int, np.dtype(int)),
        ("int32", np.dtype("int32")),
        (float, np.dtype(float)),
        ("float64", np.dtype("float64")),
        (np.dtype("float64"), np.dtype("float64")),
        (str, np.dtype(str)),
        (pd.Series([1, 2], dtype=np.dtype("int16")), np.dtype("int16")),
        (pd.Series(["a", "b"], dtype=object), np.dtype(object)),
        (pd.Index([1, 2]), np.dtype("int64")),
        (pd.Index(["a", "b"], dtype=object), np.dtype(object)),
        ("category", "category"),
        (pd.Categorical(["a", "b"]).dtype, CategoricalDtype(["a", "b"])),
        (pd.Categorical(["a", "b"]), CategoricalDtype(["a", "b"])),
        (pd.CategoricalIndex(["a", "b"]).dtype, CategoricalDtype(["a", "b"])),
        (pd.CategoricalIndex(["a", "b"]), CategoricalDtype(["a", "b"])),
        (CategoricalDtype(), CategoricalDtype()),
        (pd.DatetimeIndex([1, 2]), np.dtype("=M8[ns]")),
        (pd.DatetimeIndex([1, 2]).dtype, np.dtype("=M8[ns]")),
        ("<M8[ns]", np.dtype("<M8[ns]")),
        ("datetime64[ns, Europe/London]", DatetimeTZDtype("ns", "Europe/London")),
        (PeriodDtype(freq="D"), PeriodDtype(freq="D")),
        ("period[D]", PeriodDtype(freq="D")),
        (IntervalDtype(), IntervalDtype()),
    ],
)
def test_get_dtype(input_param, result):
    assert com._get_dtype(input_param) == result


@pytest.mark.parametrize(
    "input_param,expected_error_message",
    [
        (None, "Cannot deduce dtype from null object"),
        (1, "data type not understood"),
        (1.2, "data type not understood"),
        # numpy dev changed from double-quotes to single quotes
        ("random string", "data type [\"']random string[\"'] not understood"),
        (pd.DataFrame([1, 2]), "data type not understood"),
    ],
)
def test_get_dtype_fails(input_param, expected_error_message):
    # python objects
    # 2020-02-02 npdev changed error message
    expected_error_message += f"|Cannot interpret '{input_param}' as a data type"
    with pytest.raises(TypeError, match=expected_error_message):
        com._get_dtype(input_param)


@pytest.mark.parametrize(
    "input_param,result",
    [
        (int, np.dtype(int).type),
        ("int32", np.int32),
        (float, np.dtype(float).type),
        ("float64", np.float64),
        (np.dtype("float64"), np.float64),
        (str, np.dtype(str).type),
        (pd.Series([1, 2], dtype=np.dtype("int16")), np.int16),
        (pd.Series(["a", "b"], dtype=object), np.object_),
        (pd.Index([1, 2], dtype="int64"), np.int64),
        (pd.Index(["a", "b"], dtype=object), np.object_),
        ("category", CategoricalDtypeType),
        (pd.Categorical(["a", "b"]).dtype, CategoricalDtypeType),
        (pd.Categorical(["a", "b"]), CategoricalDtypeType),
        (pd.CategoricalIndex(["a", "b"]).dtype, CategoricalDtypeType),
        (pd.CategoricalIndex(["a", "b"]), CategoricalDtypeType),
        (pd.DatetimeIndex([1, 2]), np.datetime64),
        (pd.DatetimeIndex([1, 2]).dtype, np.datetime64),
        ("<M8[ns]", np.datetime64),
        (pd.DatetimeIndex(["2000"], tz="Europe/London"), pd.Timestamp),
        (pd.DatetimeIndex(["2000"], tz="Europe/London").dtype, pd.Timestamp),
        ("datetime64[ns, Europe/London]", pd.Timestamp),
        (PeriodDtype(freq="D"), pd.Period),
        ("period[D]", pd.Period),
        (IntervalDtype(), pd.Interval),
        (None, type(None)),
        (1, type(None)),
        (1.2, type(None)),
        (pd.DataFrame([1, 2]), type(None)),  # composite dtype
    ],
)
def test__is_dtype_type(input_param, result):
    assert com._is_dtype_type(input_param, lambda tipo: tipo == result)


def test_astype_nansafe_copy_false(any_int_numpy_dtype):
    # GH#34457 use astype, not view
    arr = np.array([1, 2, 3], dtype=any_int_numpy_dtype)

    dtype = np.dtype("float64")
    result = astype_array(arr, dtype, copy=False)

    expected = np.array([1.0, 2.0, 3.0], dtype=dtype)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("from_type", [np.datetime64, np.timedelta64])
def test_astype_object_preserves_datetime_na(from_type):
    arr = np.array([from_type("NaT", "ns")])
    result = astype_array(arr, dtype=np.dtype("object"))

    assert isna(result)[0]


def test_validate_allhashable():
    assert com.validate_all_hashable(1, "a") is None

    with pytest.raises(TypeError, match="All elements must be hashable"):
        com.validate_all_hashable([])

    with pytest.raises(TypeError, match="list must be a hashable type"):
        com.validate_all_hashable([], error_name="list")


def test_pandas_dtype_numpy_warning():
    # GH#51523
    if Version(np.__version__) < Version("2.3.0.dev0"):
        ctx = tm.assert_produces_warning(
            DeprecationWarning,
            check_stacklevel=False,
            match=(
                "Converting `np.integer` or `np.signedinteger` to a dtype is deprecated"
            ),
        )
    else:
        ctx = tm.external_error_raised(TypeError)

    with ctx:
        pandas_dtype(np.integer)


def test_pandas_dtype_ea_not_instance():
    # GH 31356 GH 54592
    with tm.assert_produces_warning(UserWarning):
        assert pandas_dtype(CategoricalDtype) == CategoricalDtype()


def test_pandas_dtype_string_dtypes(string_storage):
    with pd.option_context("future.infer_string", True):
        # with the default string_storage setting
        result = pandas_dtype("str")
    assert result == pd.StringDtype(
        "pyarrow" if HAS_PYARROW else "python", na_value=np.nan
    )

    with pd.option_context("future.infer_string", True):
        # with the default string_storage setting
        result = pandas_dtype(str)
    assert result == pd.StringDtype(
        "pyarrow" if HAS_PYARROW else "python", na_value=np.nan
    )

    with pd.option_context("future.infer_string", True):
        with pd.option_context("string_storage", string_storage):
            result = pandas_dtype("str")
    assert result == pd.StringDtype(string_storage, na_value=np.nan)

    with pd.option_context("future.infer_string", True):
        with pd.option_context("string_storage", string_storage):
            result = pandas_dtype(str)
    assert result == pd.StringDtype(string_storage, na_value=np.nan)

    with pd.option_context("future.infer_string", False):
        with pd.option_context("string_storage", string_storage):
            result = pandas_dtype("str")
    assert result == np.dtype("U")

    with pd.option_context("string_storage", string_storage):
        result = pandas_dtype("string")
    assert result == pd.StringDtype(string_storage, na_value=pd.NA)


def test_pandas_dtype_string_dtype_alias_with_storage():
    with pytest.raises(TypeError, match="not understood"):
        pandas_dtype("str[python]")

    with pytest.raises(TypeError, match="not understood"):
        pandas_dtype("str[pyarrow]")

    result = pandas_dtype("string[python]")
    assert result == pd.StringDtype("python", na_value=pd.NA)

    if HAS_PYARROW:
        result = pandas_dtype("string[pyarrow]")
        assert result == pd.StringDtype("pyarrow", na_value=pd.NA)
    else:
        with pytest.raises(
            ImportError, match="required for PyArrow backed StringArray"
        ):
            pandas_dtype("string[pyarrow]")

# === atelier-kyo-manager/remove_duplicates.py ===
import os
from PIL import Image
import imagehash

def remove_duplicate_images(folder):
    hashes = {}
    removed = 0
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.avif')):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path)
                    hash = imagehash.phash(img)
                    if hash in hashes:
                        print(f"重複画像を削除: {path}")
                        os.remove(path)
                        removed += 1
                    else:
                        hashes[hash] = path
                except Exception as e:
                    print(f"エラー: {path} ({e})")
    print(f"重複削除数: {removed}")

# 例：imagesフォルダ配下すべてで重複除去
remove_duplicate_images('./images')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask_sqlalchemy\__init__.py ===
from __future__ import annotations

import typing as t

from .extension import SQLAlchemy

__all__ = [
    "SQLAlchemy",
]


def __getattr__(name: str) -> t.Any:
    if name == "__version__":
        import importlib.metadata
        import warnings

        warnings.warn(
            "The '__version__' attribute is deprecated and will be removed in"
            " Flask-SQLAlchemy 3.2. Use feature detection or"
            " 'importlib.metadata.version(\"flask-sqlalchemy\")' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return importlib.metadata.version("flask-sqlalchemy")

    raise AttributeError(name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\operators\base_operator.py ===
class QuantOperatorBase:
    def __init__(self, onnx_quantizer, onnx_node):
        self.quantizer = onnx_quantizer
        self.node = onnx_node

    def should_quantize(self):
        if not self.quantizer.should_quantize_node(self.node):
            return False

        return self.quantizer.is_float_tensor(self.node.input[0])

    def quantize(self):
        """
        Given a node which does not support quantization, this method checks whether the input to
        this node is quantized and adds a DequantizeLinear node to dequantize this input back to FP32
            parameter node: Current node
            parameter new_nodes_list: List of new nodes created before processing current node
            return: List of new nodes created
        """
        for _, node_input in enumerate(self.node.input):
            dequantize_node = self.quantizer._dequantize_value(node_input)
            if dequantize_node is not None:
                self.quantizer.new_nodes.append(dequantize_node)

        # Append the original node
        self.quantizer.new_nodes.append(self.node)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\TensorDataType.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

class TensorDataType(object):
    UNDEFINED = 0
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15
    BFLOAT16 = 16
    FLOAT8E4M3FN = 17
    FLOAT8E4M3FNUZ = 18
    FLOAT8E5M2 = 19
    FLOAT8E5M2FNUZ = 20

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\utils\bound_dictionary.py ===
# Copyright (c) 2010-2024 openpyxl

from collections import defaultdict


class BoundDictionary(defaultdict):
    """
    A default dictionary where elements are tightly coupled.

    The factory method is responsible for binding the parent object to the child.

    If a reference attribute is assigned then child objects will have the key assigned to this.

    Otherwise it's just a defaultdict.
    """

    def __init__(self, reference=None, *args, **kw):
        self.reference = reference
        super().__init__(*args, **kw)


    def __getitem__(self, key):
        value = super().__getitem__(key)
        if self.reference is not None:
            setattr(value, self.reference, key)
        return value

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\utils\filetypes.py ===
"""Filetype information."""

from typing import Tuple

from pip._internal.utils.misc import splitext

WHEEL_EXTENSION = ".whl"
BZ2_EXTENSIONS: Tuple[str, ...] = (".tar.bz2", ".tbz")
XZ_EXTENSIONS: Tuple[str, ...] = (
    ".tar.xz",
    ".txz",
    ".tlz",
    ".tar.lz",
    ".tar.lzma",
)
ZIP_EXTENSIONS: Tuple[str, ...] = (".zip", WHEEL_EXTENSION)
TAR_EXTENSIONS: Tuple[str, ...] = (".tar.gz", ".tgz", ".tar")
ARCHIVE_EXTENSIONS = ZIP_EXTENSIONS + BZ2_EXTENSIONS + TAR_EXTENSIONS + XZ_EXTENSIONS


def is_archive_file(name: str) -> bool:
    """Return True if `name` is a considered as an archive file."""
    ext = splitext(name)[1].lower()
    if ext in ARCHIVE_EXTENSIONS:
        return True
    return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\logger\__init__.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

from .control import (
    start_file_log,
    start_socket_log,
    stop_file_log,
    stop_logging,
    stop_socket_log,
)
from .log import log

__all__ = [
    "start_file_log",
    "start_socket_log",
    "stop_file_log",
    "stop_logging",
    "stop_socket_log",
    "log",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\exceptions.py ===
"""
Exceptions raised by the matrix module.
"""


class MatrixError(Exception):
    pass


class ShapeError(ValueError, MatrixError):
    """Wrong matrix shape"""
    pass


class NonSquareMatrixError(ShapeError):
    pass


class NonInvertibleMatrixError(ValueError, MatrixError):
    """The matrix in not invertible (division by multidimensional zero error)."""
    pass


class NonPositiveDefiniteMatrixError(ValueError, MatrixError):
    """The matrix is not a positive-definite matrix."""
    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_concat_tb.py ===
from __future__ import annotations

from types import TracebackType


# this is used for collapsing single-exception ExceptionGroups when using
# `strict_exception_groups=False`. Once that is retired this function can
# be removed as well.
def concat_tb(
    head: TracebackType | None,
    tail: TracebackType | None,
) -> TracebackType | None:
    # We have to use an iterative algorithm here, because in the worst case
    # this might be a RecursionError stack that is by definition too deep to
    # process by recursion!
    head_tbs = []
    pointer = head
    while pointer is not None:
        head_tbs.append(pointer)
        pointer = pointer.tb_next
    current_head = tail
    for head_tb in reversed(head_tbs):
        current_head = TracebackType(
            current_head, head_tb.tb_frame, head_tb.tb_lasti, head_tb.tb_lineno
        )
    return current_head

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\__init__.py ===
"""
__init__.py
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
from ._abnf import *
from ._app import WebSocketApp as WebSocketApp, setReconnect as setReconnect
from ._core import *
from ._exceptions import *
from ._logging import *
from ._socket import *

__version__ = "1.8.0"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\tests\test_line.py ===
from sympy.core.numbers import (Float, Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.sets import EmptySet
from sympy.simplify.simplify import simplify
from sympy.functions.elementary.trigonometric import tan
from sympy.geometry import (Circle, GeometryError, Line, Point, Ray,
    Segment, Triangle, intersection, Point3D, Line3D, Ray3D, Segment3D,
    Point2D, Line2D, Plane)
from sympy.geometry.line import Undecidable
from sympy.geometry.polygon import _asa as asa
from sympy.utilities.iterables import cartes
from sympy.testing.pytest import raises, warns


x = Symbol('x', real=True)
y = Symbol('y', real=True)
z = Symbol('z', real=True)
k = Symbol('k', real=True)
x1 = Symbol('x1', real=True)
y1 = Symbol('y1', real=True)
t = Symbol('t', real=True)
a, b = symbols('a,b', real=True)
m = symbols('m', real=True)


def test_object_from_equation():
    from sympy.abc import x, y, a, b
    assert Line(3*x + y + 18) == Line2D(Point2D(0, -18), Point2D(1, -21))
    assert Line(3*x + 5 * y + 1) == Line2D(
        Point2D(0, Rational(-1, 5)), Point2D(1, Rational(-4, 5)))
    assert Line(3*a + b + 18, x="a", y="b") == Line2D(
        Point2D(0, -18), Point2D(1, -21))
    assert Line(3*x + y) == Line2D(Point2D(0, 0), Point2D(1, -3))
    assert Line(x + y) == Line2D(Point2D(0, 0), Point2D(1, -1))
    assert Line(Eq(3*a + b, -18), x="a", y=b) == Line2D(
        Point2D(0, -18), Point2D(1, -21))
    # issue 22361
    assert Line(x - 1) == Line2D(Point2D(1, 0), Point2D(1, 1))
    assert Line(2*x - 2, y=x) == Line2D(Point2D(0, 1), Point2D(1, 1))
    assert Line(y) == Line2D(Point2D(0, 0), Point2D(1, 0))
    assert Line(2*y, x=y) == Line2D(Point2D(0, 0), Point2D(0, 1))
    assert Line(y, x=y) == Line2D(Point2D(0, 0), Point2D(0, 1))
    raises(ValueError, lambda: Line(x / y))
    raises(ValueError, lambda: Line(a / b, x='a', y='b'))
    raises(ValueError, lambda: Line(y / x))
    raises(ValueError, lambda: Line(b / a, x='a', y='b'))
    raises(ValueError, lambda: Line((x + 1)**2 + y))


def feq(a, b):
    """Test if two floating point values are 'equal'."""
    t_float = Float("1.0E-10")
    return -t_float < a - b < t_float


def test_angle_between():
    a = Point(1, 2, 3, 4)
    b = a.orthogonal_direction
    o = a.origin
    assert feq(Line.angle_between(Line(Point(0, 0), Point(1, 1)),
                                  Line(Point(0, 0), Point(5, 0))).evalf(), pi.evalf() / 4)
    assert Line(a, o).angle_between(Line(b, o)) == pi / 2
    z = Point3D(0, 0, 0)
    assert Line3D.angle_between(Line3D(z, Point3D(1, 1, 1)),
                                Line3D(z, Point3D(5, 0, 0))) == acos(sqrt(3) / 3)
    # direction of points is used to determine angle
    assert Line3D.angle_between(Line3D(z, Point3D(1, 1, 1)),
                                Line3D(Point3D(5, 0, 0), z)) == acos(-sqrt(3) / 3)


def test_closing_angle():
    a = Ray((0, 0), angle=0)
    b = Ray((1, 2), angle=pi/2)
    assert a.closing_angle(b) == -pi/2
    assert b.closing_angle(a) == pi/2
    assert a.closing_angle(a) == 0


def test_smallest_angle():
    a = Line(Point(1, 1), Point(1, 2))
    b = Line(Point(1, 1),Point(2, 3))
    assert a.smallest_angle_between(b) == acos(2*sqrt(5)/5)


def test_svg():
    a = Line(Point(1, 1),Point(1, 2))
    assert a._svg() == '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" stroke-width="2.0" opacity="0.6" d="M 1.00000000000000,1.00000000000000 L 1.00000000000000,2.00000000000000" marker-start="url(#markerReverseArrow)" marker-end="url(#markerArrow)"/>'
    a = Segment(Point(1, 0),Point(1, 1))
    assert a._svg() == '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" stroke-width="2.0" opacity="0.6" d="M 1.00000000000000,0 L 1.00000000000000,1.00000000000000" />'
    a = Ray(Point(2, 3), Point(3, 5))
    assert a._svg() == '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" stroke-width="2.0" opacity="0.6" d="M 2.00000000000000,3.00000000000000 L 3.00000000000000,5.00000000000000" marker-start="url(#markerCircle)" marker-end="url(#markerArrow)"/>'


def test_arbitrary_point():
    l1 = Line3D(Point3D(0, 0, 0), Point3D(1, 1, 1))
    l2 = Line(Point(x1, x1), Point(y1, y1))
    assert l2.arbitrary_point() in l2
    assert Ray((1, 1), angle=pi / 4).arbitrary_point() == \
           Point(t + 1, t + 1)
    assert Segment((1, 1), (2, 3)).arbitrary_point() == Point(1 + t, 1 + 2 * t)
    assert l1.perpendicular_segment(l1.arbitrary_point()) == l1.arbitrary_point()
    assert Ray3D((1, 1, 1), direction_ratio=[1, 2, 3]).arbitrary_point() == \
           Point3D(t + 1, 2 * t + 1, 3 * t + 1)
    assert Segment3D(Point3D(0, 0, 0), Point3D(1, 1, 1)).midpoint == \
           Point3D(S.Half, S.Half, S.Half)
    assert Segment3D(Point3D(x1, x1, x1), Point3D(y1, y1, y1)).length == sqrt(3) * sqrt((x1 - y1) ** 2)
    assert Segment3D((1, 1, 1), (2, 3, 4)).arbitrary_point() == \
           Point3D(t + 1, 2 * t + 1, 3 * t + 1)
    raises(ValueError, (lambda: Line((x, 1), (2, 3)).arbitrary_point(x)))


def test_are_concurrent_2d():
    l1 = Line(Point(0, 0), Point(1, 1))
    l2 = Line(Point(x1, x1), Point(x1, 1 + x1))
    assert Line.are_concurrent(l1) is False
    assert Line.are_concurrent(l1, l2)
    assert Line.are_concurrent(l1, l1, l1, l2)
    assert Line.are_concurrent(l1, l2, Line(Point(5, x1), Point(Rational(-3, 5), x1)))
    assert Line.are_concurrent(l1, Line(Point(0, 0), Point(-x1, x1)), l2) is False


def test_are_concurrent_3d():
    p1 = Point3D(0, 0, 0)
    l1 = Line(p1, Point3D(1, 1, 1))
    parallel_1 = Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0))
    parallel_2 = Line3D(Point3D(0, 1, 0), Point3D(1, 1, 0))
    assert Line3D.are_concurrent(l1) is False
    assert Line3D.are_concurrent(l1, Line(Point3D(x1, x1, x1), Point3D(y1, y1, y1))) is False
    assert Line3D.are_concurrent(l1, Line3D(p1, Point3D(x1, x1, x1)),
                                 Line(Point3D(x1, x1, x1), Point3D(x1, 1 + x1, 1))) is True
    assert Line3D.are_concurrent(parallel_1, parallel_2) is False


def test_arguments():
    """Functions accepting `Point` objects in `geometry`
    should also accept tuples, lists, and generators and
    automatically convert them to points."""
    from sympy.utilities.iterables import subsets

    singles2d = ((1, 2), [1, 3], Point(1, 5))
    doubles2d = subsets(singles2d, 2)
    l2d = Line(Point2D(1, 2), Point2D(2, 3))
    singles3d = ((1, 2, 3), [1, 2, 4], Point(1, 2, 6))
    doubles3d = subsets(singles3d, 2)
    l3d = Line(Point3D(1, 2, 3), Point3D(1, 1, 2))
    singles4d = ((1, 2, 3, 4), [1, 2, 3, 5], Point(1, 2, 3, 7))
    doubles4d = subsets(singles4d, 2)
    l4d = Line(Point(1, 2, 3, 4), Point(2, 2, 2, 2))
    # test 2D
    test_single = ['contains', 'distance', 'equals', 'parallel_line', 'perpendicular_line', 'perpendicular_segment',
                   'projection', 'intersection']
    for p in doubles2d:
        Line2D(*p)
    for func in test_single:
        for p in singles2d:
            getattr(l2d, func)(p)
    # test 3D
    for p in doubles3d:
        Line3D(*p)
    for func in test_single:
        for p in singles3d:
            getattr(l3d, func)(p)
    # test 4D
    for p in doubles4d:
        Line(*p)
    for func in test_single:
        for p in singles4d:
            getattr(l4d, func)(p)


def test_basic_properties_2d():
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    p10 = Point(2000, 2000)
    p_r3 = Ray(p1, p2).random_point()
    p_r4 = Ray(p2, p1).random_point()

    l1 = Line(p1, p2)
    l3 = Line(Point(x1, x1), Point(x1, 1 + x1))
    l4 = Line(p1, Point(1, 0))

    r1 = Ray(p1, Point(0, 1))
    r2 = Ray(Point(0, 1), p1)

    s1 = Segment(p1, p10)
    p_s1 = s1.random_point()

    assert Line((1, 1), slope=1) == Line((1, 1), (2, 2))
    assert Line((1, 1), slope=oo) == Line((1, 1), (1, 2))
    assert Line((1, 1), slope=oo).bounds == (1, 1, 1, 2)
    assert Line((1, 1), slope=-oo) == Line((1, 1), (1, 2))
    assert Line(p1, p2).scale(2, 1) == Line(p1, Point(2, 1))
    assert Line(p1, p2) == Line(p1, p2)
    assert Line(p1, p2) != Line(p2, p1)
    assert l1 != Line(Point(x1, x1), Point(y1, y1))
    assert l1 != l3
    assert Line(p1, p10) != Line(p10, p1)
    assert Line(p1, p10) != p1
    assert p1 in l1  # is p1 on the line l1?
    assert p1 not in l3
    assert s1 in Line(p1, p10)
    assert Ray(Point(0, 0), Point(0, 1)) in Ray(Point(0, 0), Point(0, 2))
    assert Ray(Point(0, 0), Point(0, 2)) in Ray(Point(0, 0), Point(0, 1))
    assert Ray(Point(0, 0), Point(0, 2)).xdirection == S.Zero
    assert Ray(Point(0, 0), Point(1, 2)).xdirection == S.Infinity
    assert Ray(Point(0, 0), Point(-1, 2)).xdirection == S.NegativeInfinity
    assert Ray(Point(0, 0), Point(2, 0)).ydirection == S.Zero
    assert Ray(Point(0, 0), Point(2, 2)).ydirection == S.Infinity
    assert Ray(Point(0, 0), Point(2, -2)).ydirection == S.NegativeInfinity
    assert (r1 in s1) is False
    assert Segment(p1, p2) in s1
    assert Ray(Point(x1, x1), Point(x1, 1 + x1)) != Ray(p1, Point(-1, 5))
    assert Segment(p1, p2).midpoint == Point(S.Half, S.Half)
    assert Segment(p1, Point(-x1, x1)).length == sqrt(2 * (x1 ** 2))

    assert l1.slope == 1
    assert l3.slope is oo
    assert l4.slope == 0
    assert Line(p1, Point(0, 1)).slope is oo
    assert Line(r1.source, r1.random_point()).slope == r1.slope
    assert Line(r2.source, r2.random_point()).slope == r2.slope
    assert Segment(Point(0, -1), Segment(p1, Point(0, 1)).random_point()).slope == Segment(p1, Point(0, 1)).slope

    assert l4.coefficients == (0, 1, 0)
    assert Line((-x, x), (-x + 1, x - 1)).coefficients == (1, 1, 0)
    assert Line(p1, Point(0, 1)).coefficients == (1, 0, 0)
    # issue 7963
    r = Ray((0, 0), angle=x)
    assert r.subs(x, 3 * pi / 4) == Ray((0, 0), (-1, 1))
    assert r.subs(x, 5 * pi / 4) == Ray((0, 0), (-1, -1))
    assert r.subs(x, -pi / 4) == Ray((0, 0), (1, -1))
    assert r.subs(x, pi / 2) == Ray((0, 0), (0, 1))
    assert r.subs(x, -pi / 2) == Ray((0, 0), (0, -1))

    for ind in range(0, 5):
        assert l3.random_point() in l3

    assert p_r3.x >= p1.x and p_r3.y >= p1.y
    assert p_r4.x <= p2.x and p_r4.y <= p2.y
    assert p1.x <= p_s1.x <= p10.x and p1.y <= p_s1.y <= p10.y
    assert hash(s1) != hash(Segment(p10, p1))

    assert s1.plot_interval() == [t, 0, 1]
    assert Line(p1, p10).plot_interval() == [t, -5, 5]
    assert Ray((0, 0), angle=pi / 4).plot_interval() == [t, 0, 10]


def test_basic_properties_3d():
    p1 = Point3D(0, 0, 0)
    p2 = Point3D(1, 1, 1)
    p3 = Point3D(x1, x1, x1)
    p5 = Point3D(x1, 1 + x1, 1)

    l1 = Line3D(p1, p2)
    l3 = Line3D(p3, p5)

    r1 = Ray3D(p1, Point3D(-1, 5, 0))
    r3 = Ray3D(p1, p2)

    s1 = Segment3D(p1, p2)

    assert Line3D((1, 1, 1), direction_ratio=[2, 3, 4]) == Line3D(Point3D(1, 1, 1), Point3D(3, 4, 5))
    assert Line3D((1, 1, 1), direction_ratio=[1, 5, 7]) == Line3D(Point3D(1, 1, 1), Point3D(2, 6, 8))
    assert Line3D((1, 1, 1), direction_ratio=[1, 2, 3]) == Line3D(Point3D(1, 1, 1), Point3D(2, 3, 4))
    assert Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0)).direction_cosine == [1, 0, 0]
    assert Line3D(Line3D(p1, Point3D(0, 1, 0))) == Line3D(p1, Point3D(0, 1, 0))
    assert Ray3D(Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0))) == Ray3D(p1, Point3D(1, 0, 0))
    assert Line3D(p1, p2) != Line3D(p2, p1)
    assert l1 != l3
    assert l1 != Line3D(p3, Point3D(y1, y1, y1))
    assert r3 != r1
    assert Ray3D(Point3D(0, 0, 0), Point3D(1, 1, 1)) in Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 2))
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 2)) in Ray3D(Point3D(0, 0, 0), Point3D(1, 1, 1))
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 2)).xdirection == S.Infinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 2)).ydirection == S.Infinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 2)).zdirection == S.Infinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(-2, 2, 2)).xdirection == S.NegativeInfinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, -2, 2)).ydirection == S.NegativeInfinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, -2)).zdirection == S.NegativeInfinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(0, 2, 2)).xdirection == S.Zero
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 0, 2)).ydirection == S.Zero
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 0)).zdirection == S.Zero
    assert p1 in l1
    assert p1 not in l3

    assert l1.direction_ratio == [1, 1, 1]

    assert s1.midpoint == Point3D(S.Half, S.Half, S.Half)
    # Test zdirection
    assert Ray3D(p1, Point3D(0, 0, -1)).zdirection is S.NegativeInfinity


def test_contains():
    p1 = Point(0, 0)

    r = Ray(p1, Point(4, 4))
    r1 = Ray3D(p1, Point3D(0, 0, -1))
    r2 = Ray3D(p1, Point3D(0, 1, 0))
    r3 = Ray3D(p1, Point3D(0, 0, 1))

    l = Line(Point(0, 1), Point(3, 4))
    # Segment contains
    assert Point(0, (a + b) / 2) in Segment((0, a), (0, b))
    assert Point((a + b) / 2, 0) in Segment((a, 0), (b, 0))
    assert Point3D(0, 1, 0) in Segment3D((0, 1, 0), (0, 1, 0))
    assert Point3D(1, 0, 0) in Segment3D((1, 0, 0), (1, 0, 0))
    assert Segment3D(Point3D(0, 0, 0), Point3D(1, 0, 0)).contains([]) is True
    assert Segment3D(Point3D(0, 0, 0), Point3D(1, 0, 0)).contains(
        Segment3D(Point3D(2, 2, 2), Point3D(3, 2, 2))) is False
    # Line contains
    assert l.contains(Point(0, 1)) is True
    assert l.contains((0, 1)) is True
    assert l.contains((0, 0)) is False
    # Ray contains
    assert r.contains(p1) is True
    assert r.contains((1, 1)) is True
    assert r.contains((1, 3)) is False
    assert r.contains(Segment((1, 1), (2, 2))) is True
    assert r.contains(Segment((1, 2), (2, 5))) is False
    assert r.contains(Ray((2, 2), (3, 3))) is True
    assert r.contains(Ray((2, 2), (3, 5))) is False
    assert r1.contains(Segment3D(p1, Point3D(0, 0, -10))) is True
    assert r1.contains(Segment3D(Point3D(1, 1, 1), Point3D(2, 2, 2))) is False
    assert r2.contains(Point3D(0, 0, 0)) is True
    assert r3.contains(Point3D(0, 0, 0)) is True
    assert Ray3D(Point3D(1, 1, 1), Point3D(1, 0, 0)).contains([]) is False
    assert Line3D((0, 0, 0), (x, y, z)).contains((2 * x, 2 * y, 2 * z))
    with warns(UserWarning, test_stacklevel=False):
        assert Line3D(p1, Point3D(0, 1, 0)).contains(Point(1.0, 1.0)) is False

    with warns(UserWarning, test_stacklevel=False):
        assert r3.contains(Point(1.0, 1.0)) is False


def test_contains_nonreal_symbols():
    u, v, w, z = symbols('u, v, w, z')
    l = Segment(Point(u, w), Point(v, z))
    p = Point(u*Rational(2, 3) + v/3, w*Rational(2, 3) + z/3)
    assert l.contains(p)


def test_distance_2d():
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    half = S.Half

    s1 = Segment(Point(0, 0), Point(1, 1))
    s2 = Segment(Point(half, half), Point(1, 0))

    r = Ray(p1, p2)

    assert s1.distance(Point(0, 0)) == 0
    assert s1.distance((0, 0)) == 0
    assert s2.distance(Point(0, 0)) == 2 ** half / 2
    assert s2.distance(Point(Rational(3) / 2, Rational(3) / 2)) == 2 ** half
    assert Line(p1, p2).distance(Point(-1, 1)) == sqrt(2)
    assert Line(p1, p2).distance(Point(1, -1)) == sqrt(2)
    assert Line(p1, p2).distance(Point(2, 2)) == 0
    assert Line(p1, p2).distance((-1, 1)) == sqrt(2)
    assert Line((0, 0), (0, 1)).distance(p1) == 0
    assert Line((0, 0), (0, 1)).distance(p2) == 1
    assert Line((0, 0), (1, 0)).distance(p1) == 0
    assert Line((0, 0), (1, 0)).distance(p2) == 1
    assert r.distance(Point(-1, -1)) == sqrt(2)
    assert r.distance(Point(1, 1)) == 0
    assert r.distance(Point(-1, 1)) == sqrt(2)
    assert Ray((1, 1), (2, 2)).distance(Point(1.5, 3)) == 3 * sqrt(2) / 4
    assert r.distance((1, 1)) == 0


def test_dimension_normalization():
    with warns(UserWarning, test_stacklevel=False):
        assert Ray((1, 1), (2, 1, 2)) == Ray((1, 1, 0), (2, 1, 2))


def test_distance_3d():
    p1, p2 = Point3D(0, 0, 0), Point3D(1, 1, 1)
    p3 = Point3D(Rational(3) / 2, Rational(3) / 2, Rational(3) / 2)

    s1 = Segment3D(Point3D(0, 0, 0), Point3D(1, 1, 1))
    s2 = Segment3D(Point3D(S.Half, S.Half, S.Half), Point3D(1, 0, 1))

    r = Ray3D(p1, p2)

    assert s1.distance(p1) == 0
    assert s2.distance(p1) == sqrt(3) / 2
    assert s2.distance(p3) == 2 * sqrt(6) / 3
    assert s1.distance((0, 0, 0)) == 0
    assert s2.distance((0, 0, 0)) == sqrt(3) / 2
    assert s1.distance(p1) == 0
    assert s2.distance(p1) == sqrt(3) / 2
    assert s2.distance(p3) == 2 * sqrt(6) / 3
    assert s1.distance((0, 0, 0)) == 0
    assert s2.distance((0, 0, 0)) == sqrt(3) / 2
    # Line to point
    assert Line3D(p1, p2).distance(Point3D(-1, 1, 1)) == 2 * sqrt(6) / 3
    assert Line3D(p1, p2).distance(Point3D(1, -1, 1)) == 2 * sqrt(6) / 3
    assert Line3D(p1, p2).distance(Point3D(2, 2, 2)) == 0
    assert Line3D(p1, p2).distance((2, 2, 2)) == 0
    assert Line3D(p1, p2).distance((1, -1, 1)) == 2 * sqrt(6) / 3
    assert Line3D((0, 0, 0), (0, 1, 0)).distance(p1) == 0
    assert Line3D((0, 0, 0), (0, 1, 0)).distance(p2) == sqrt(2)
    assert Line3D((0, 0, 0), (1, 0, 0)).distance(p1) == 0
    assert Line3D((0, 0, 0), (1, 0, 0)).distance(p2) == sqrt(2)
    # Line to line
    assert Line3D((0, 0, 0), (1, 0, 0)).distance(Line3D((0, 0, 0), (0, 1, 2))) == 0
    assert Line3D((0, 0, 0), (1, 0, 0)).distance(Line3D((0, 0, 0), (1, 0, 0))) == 0
    assert Line3D((0, 0, 0), (1, 0, 0)).distance(Line3D((10, 0, 0), (10, 1, 2))) == 0
    assert Line3D((0, 0, 0), (1, 0, 0)).distance(Line3D((0, 1, 0), (0, 1, 1))) == 1
    # Line to plane
    assert Line3D((0, 0, 0), (1, 0, 0)).distance(Plane((2, 0, 0), (0, 0, 1))) == 0
    assert Line3D((0, 0, 0), (1, 0, 0)).distance(Plane((0, 1, 0), (0, 1, 0))) == 1
    assert Line3D((0, 0, 0), (1, 0, 0)).distance(Plane((1, 1, 3), (1, 0, 0))) == 0
    # Ray to point
    assert r.distance(Point3D(-1, -1, -1)) == sqrt(3)
    assert r.distance(Point3D(1, 1, 1)) == 0
    assert r.distance((-1, -1, -1)) == sqrt(3)
    assert r.distance((1, 1, 1)) == 0
    assert Ray3D((0, 0, 0), (1, 1, 2)).distance((-1, -1, 2)) == 4 * sqrt(3) / 3
    assert Ray3D((1, 1, 1), (2, 2, 2)).distance(Point3D(1.5, -3, -1)) == Rational(9) / 2
    assert Ray3D((1, 1, 1), (2, 2, 2)).distance(Point3D(1.5, 3, 1)) == sqrt(78) / 6


def test_equals():
    p1 = Point(0, 0)
    p2 = Point(1, 1)

    l1 = Line(p1, p2)
    l2 = Line((0, 5), slope=m)
    l3 = Line(Point(x1, x1), Point(x1, 1 + x1))

    assert l1.perpendicular_line(p1.args).equals(Line(Point(0, 0), Point(1, -1)))
    assert l1.perpendicular_line(p1).equals(Line(Point(0, 0), Point(1, -1)))
    assert Line(Point(x1, x1), Point(y1, y1)).parallel_line(Point(-x1, x1)). \
        equals(Line(Point(-x1, x1), Point(-y1, 2 * x1 - y1)))
    assert l3.parallel_line(p1.args).equals(Line(Point(0, 0), Point(0, -1)))
    assert l3.parallel_line(p1).equals(Line(Point(0, 0), Point(0, -1)))
    assert (l2.distance(Point(2, 3)) - 2 * abs(m + 1) / sqrt(m ** 2 + 1)).equals(0)
    assert Line3D(p1, Point3D(0, 1, 0)).equals(Point(1.0, 1.0)) is False
    assert Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0)).equals(Line3D(Point3D(-5, 0, 0), Point3D(-1, 0, 0))) is True
    assert Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0)).equals(Line3D(p1, Point3D(0, 1, 0))) is False
    assert Ray3D(p1, Point3D(0, 0, -1)).equals(Point(1.0, 1.0)) is False
    assert Ray3D(p1, Point3D(0, 0, -1)).equals(Ray3D(p1, Point3D(0, 0, -1))) is True
    assert Line3D((0, 0), (t, t)).perpendicular_line(Point(0, 1, 0)).equals(
        Line3D(Point3D(0, 1, 0), Point3D(S.Half, S.Half, 0)))
    assert Line3D((0, 0), (t, t)).perpendicular_segment(Point(0, 1, 0)).equals(Segment3D((0, 1), (S.Half, S.Half)))
    assert Line3D(p1, Point3D(0, 1, 0)).equals(Point(1.0, 1.0)) is False


def test_equation():
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    l1 = Line(p1, p2)
    l3 = Line(Point(x1, x1), Point(x1, 1 + x1))

    assert simplify(l1.equation()) in (x - y, y - x)
    assert simplify(l3.equation()) in (x - x1, x1 - x)
    assert simplify(l1.equation()) in (x - y, y - x)
    assert simplify(l3.equation()) in (x - x1, x1 - x)

    assert Line(p1, Point(1, 0)).equation(x=x, y=y) == y
    assert Line(p1, Point(0, 1)).equation() == x
    assert Line(Point(2, 0), Point(2, 1)).equation() == x - 2
    assert Line(p2, Point(2, 1)).equation() == y - 1

    assert Line3D(Point(x1, x1, x1), Point(y1, y1, y1)
        ).equation() == (-x + y, -x + z)
    assert Line3D(Point(1, 2, 3), Point(2, 3, 4)
        ).equation() == (-x + y - 1, -x + z - 2)
    assert Line3D(Point(1, 2, 3), Point(1, 3, 4)
        ).equation() == (x - 1, -y + z - 1)
    assert Line3D(Point(1, 2, 3), Point(2, 2, 4)
        ).equation() == (y - 2, -x + z - 2)
    assert Line3D(Point(1, 2, 3), Point(2, 3, 3)
        ).equation() == (-x + y - 1, z - 3)
    assert Line3D(Point(1, 2, 3), Point(1, 2, 4)
        ).equation() == (x - 1, y - 2)
    assert Line3D(Point(1, 2, 3), Point(1, 3, 3)
        ).equation() == (x - 1, z - 3)
    assert Line3D(Point(1, 2, 3), Point(2, 2, 3)
        ).equation() == (y - 2, z - 3)


def test_intersection_2d():
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    p3 = Point(x1, x1)
    p4 = Point(y1, y1)

    l1 = Line(p1, p2)
    l3 = Line(Point(0, 0), Point(3, 4))

    r1 = Ray(Point(1, 1), Point(2, 2))
    r2 = Ray(Point(0, 0), Point(3, 4))
    r4 = Ray(p1, p2)
    r6 = Ray(Point(0, 1), Point(1, 2))
    r7 = Ray(Point(0.5, 0.5), Point(1, 1))

    s1 = Segment(p1, p2)
    s2 = Segment(Point(0.25, 0.25), Point(0.5, 0.5))
    s3 = Segment(Point(0, 0), Point(3, 4))

    assert intersection(l1, p1) == [p1]
    assert intersection(l1, Point(x1, 1 + x1)) == []
    assert intersection(l1, Line(p3, p4)) in [[l1], [Line(p3, p4)]]
    assert intersection(l1, l1.parallel_line(Point(x1, 1 + x1))) == []
    assert intersection(l3, l3) == [l3]
    assert intersection(l3, r2) == [r2]
    assert intersection(l3, s3) == [s3]
    assert intersection(s3, l3) == [s3]
    assert intersection(Segment(Point(-10, 10), Point(10, 10)), Segment(Point(-5, -5), Point(-5, 5))) == []
    assert intersection(r2, l3) == [r2]
    assert intersection(r1, Ray(Point(2, 2), Point(0, 0))) == [Segment(Point(1, 1), Point(2, 2))]
    assert intersection(r1, Ray(Point(1, 1), Point(-1, -1))) == [Point(1, 1)]
    assert intersection(r1, Segment(Point(0, 0), Point(2, 2))) == [Segment(Point(1, 1), Point(2, 2))]

    assert r4.intersection(s2) == [s2]
    assert r4.intersection(Segment(Point(2, 3), Point(3, 4))) == []
    assert r4.intersection(Segment(Point(-1, -1), Point(0.5, 0.5))) == [Segment(p1, Point(0.5, 0.5))]
    assert r4.intersection(Ray(p2, p1)) == [s1]
    assert Ray(p2, p1).intersection(r6) == []
    assert r4.intersection(r7) == r7.intersection(r4) == [r7]
    assert Ray3D((0, 0), (3, 0)).intersection(Ray3D((1, 0), (3, 0))) == [Ray3D((1, 0), (3, 0))]
    assert Ray3D((1, 0), (3, 0)).intersection(Ray3D((0, 0), (3, 0))) == [Ray3D((1, 0), (3, 0))]
    assert Ray(Point(0, 0), Point(0, 4)).intersection(Ray(Point(0, 1), Point(0, -1))) == \
           [Segment(Point(0, 0), Point(0, 1))]

    assert Segment3D((0, 0), (3, 0)).intersection(
        Segment3D((1, 0), (2, 0))) == [Segment3D((1, 0), (2, 0))]
    assert Segment3D((1, 0), (2, 0)).intersection(
        Segment3D((0, 0), (3, 0))) == [Segment3D((1, 0), (2, 0))]
    assert Segment3D((0, 0), (3, 0)).intersection(
        Segment3D((3, 0), (4, 0))) == [Point3D((3, 0))]
    assert Segment3D((0, 0), (3, 0)).intersection(
        Segment3D((2, 0), (5, 0))) == [Segment3D((2, 0), (3, 0))]
    assert Segment3D((0, 0), (3, 0)).intersection(
        Segment3D((-2, 0), (1, 0))) == [Segment3D((0, 0), (1, 0))]
    assert Segment3D((0, 0), (3, 0)).intersection(
        Segment3D((-2, 0), (0, 0))) == [Point3D(0, 0)]
    assert s1.intersection(Segment(Point(1, 1), Point(2, 2))) == [Point(1, 1)]
    assert s1.intersection(Segment(Point(0.5, 0.5), Point(1.5, 1.5))) == [Segment(Point(0.5, 0.5), p2)]
    assert s1.intersection(Segment(Point(4, 4), Point(5, 5))) == []
    assert s1.intersection(Segment(Point(-1, -1), p1)) == [p1]
    assert s1.intersection(Segment(Point(-1, -1), Point(0.5, 0.5))) == [Segment(p1, Point(0.5, 0.5))]
    assert s1.intersection(Line(Point(1, 0), Point(2, 1))) == []
    assert s1.intersection(s2) == [s2]
    assert s2.intersection(s1) == [s2]

    assert asa(120, 8, 52) == \
           Triangle(
               Point(0, 0),
               Point(8, 0),
               Point(-4 * cos(19 * pi / 90) / sin(2 * pi / 45),
                     4 * sqrt(3) * cos(19 * pi / 90) / sin(2 * pi / 45)))
    assert Line((0, 0), (1, 1)).intersection(Ray((1, 0), (1, 2))) == [Point(1, 1)]
    assert Line((0, 0), (1, 1)).intersection(Segment((1, 0), (1, 2))) == [Point(1, 1)]
    assert Ray((0, 0), (1, 1)).intersection(Ray((1, 0), (1, 2))) == [Point(1, 1)]
    assert Ray((0, 0), (1, 1)).intersection(Segment((1, 0), (1, 2))) == [Point(1, 1)]
    assert Ray((0, 0), (10, 10)).contains(Segment((1, 1), (2, 2))) is True
    assert Segment((1, 1), (2, 2)) in Line((0, 0), (10, 10))
    assert s1.intersection(Ray((1, 1), (4, 4))) == [Point(1, 1)]

    # This test is disabled because it hangs after rref changes which simplify
    # intermediate results and return a different representation from when the
    # test was written.
    # # 16628 - this should be fast
    # p0 = Point2D(Rational(249, 5), Rational(497999, 10000))
    # p1 = Point2D((-58977084786*sqrt(405639795226) + 2030690077184193 +
    #     20112207807*sqrt(630547164901) + 99600*sqrt(255775022850776494562626))
    #     /(2000*sqrt(255775022850776494562626) + 1991998000*sqrt(405639795226)
    #     + 1991998000*sqrt(630547164901) + 1622561172902000),
    #     (-498000*sqrt(255775022850776494562626) - 995999*sqrt(630547164901) +
    #     90004251917891999 +
    #     496005510002*sqrt(405639795226))/(10000*sqrt(255775022850776494562626)
    #     + 9959990000*sqrt(405639795226) + 9959990000*sqrt(630547164901) +
    #     8112805864510000))
    # p2 = Point2D(Rational(497, 10), Rational(-497, 10))
    # p3 = Point2D(Rational(-497, 10), Rational(-497, 10))
    # l = Line(p0, p1)
    # s = Segment(p2, p3)
    # n = (-52673223862*sqrt(405639795226) - 15764156209307469 -
    #     9803028531*sqrt(630547164901) +
    #     33200*sqrt(255775022850776494562626))
    # d = sqrt(405639795226) + 315274080450 + 498000*sqrt(
    #     630547164901) + sqrt(255775022850776494562626)
    # assert intersection(l, s) == [
    #     Point2D(n/d*Rational(3, 2000), Rational(-497, 10))]


def test_line_intersection():
    # see also test_issue_11238 in test_matrices.py
    x0 = tan(pi*Rational(13, 45))
    x1 = sqrt(3)
    x2 = x0**2
    x, y = [8*x0/(x0 + x1), (24*x0 - 8*x1*x2)/(x2 - 3)]
    assert Line(Point(0, 0), Point(1, -sqrt(3))).contains(Point(x, y)) is True


def test_intersection_3d():
    p1 = Point3D(0, 0, 0)
    p2 = Point3D(1, 1, 1)

    l1 = Line3D(p1, p2)
    l2 = Line3D(Point3D(0, 0, 0), Point3D(3, 4, 0))

    r1 = Ray3D(Point3D(1, 1, 1), Point3D(2, 2, 2))
    r2 = Ray3D(Point3D(0, 0, 0), Point3D(3, 4, 0))

    s1 = Segment3D(Point3D(0, 0, 0), Point3D(3, 4, 0))

    assert intersection(l1, p1) == [p1]
    assert intersection(l1, Point3D(x1, 1 + x1, 1)) == []
    assert intersection(l1, l1.parallel_line(p1)) == [Line3D(Point3D(0, 0, 0), Point3D(1, 1, 1))]
    assert intersection(l2, r2) == [r2]
    assert intersection(l2, s1) == [s1]
    assert intersection(r2, l2) == [r2]
    assert intersection(r1, Ray3D(Point3D(1, 1, 1), Point3D(-1, -1, -1))) == [Point3D(1, 1, 1)]
    assert intersection(r1, Segment3D(Point3D(0, 0, 0), Point3D(2, 2, 2))) == [
        Segment3D(Point3D(1, 1, 1), Point3D(2, 2, 2))]
    assert intersection(Ray3D(Point3D(1, 0, 0), Point3D(-1, 0, 0)), Ray3D(Point3D(0, 1, 0), Point3D(0, -1, 0))) \
           == [Point3D(0, 0, 0)]
    assert intersection(r1, Ray3D(Point3D(2, 2, 2), Point3D(0, 0, 0))) == \
           [Segment3D(Point3D(1, 1, 1), Point3D(2, 2, 2))]
    assert intersection(s1, r2) == [s1]

    assert Line3D(Point3D(4, 0, 1), Point3D(0, 4, 1)).intersection(Line3D(Point3D(0, 0, 1), Point3D(4, 4, 1))) == \
           [Point3D(2, 2, 1)]
    assert Line3D((0, 1, 2), (0, 2, 3)).intersection(Line3D((0, 1, 2), (0, 1, 1))) == [Point3D(0, 1, 2)]
    assert Line3D((0, 0), (t, t)).intersection(Line3D((0, 1), (t, t))) == \
           [Point3D(t, t)]

    assert Ray3D(Point3D(0, 0, 0), Point3D(0, 4, 0)).intersection(Ray3D(Point3D(0, 1, 1), Point3D(0, -1, 1))) == []


def test_is_parallel():
    p1 = Point3D(0, 0, 0)
    p2 = Point3D(1, 1, 1)
    p3 = Point3D(x1, x1, x1)

    l2 = Line(Point(x1, x1), Point(y1, y1))
    l2_1 = Line(Point(x1, x1), Point(x1, 1 + x1))

    assert Line.is_parallel(Line(Point(0, 0), Point(1, 1)), l2)
    assert Line.is_parallel(l2, Line(Point(x1, x1), Point(x1, 1 + x1))) is False
    assert Line.is_parallel(l2, l2.parallel_line(Point(-x1, x1)))
    assert Line.is_parallel(l2_1, l2_1.parallel_line(Point(0, 0)))
    assert Line3D(p1, p2).is_parallel(Line3D(p1, p2))  # same as in 2D
    assert Line3D(Point3D(4, 0, 1), Point3D(0, 4, 1)).is_parallel(Line3D(Point3D(0, 0, 1), Point3D(4, 4, 1))) is False
    assert Line3D(p1, p2).parallel_line(p3) == Line3D(Point3D(x1, x1, x1),
                                                      Point3D(x1 + 1, x1 + 1, x1 + 1))
    assert Line3D(p1, p2).parallel_line(p3.args) == \
           Line3D(Point3D(x1, x1, x1), Point3D(x1 + 1, x1 + 1, x1 + 1))
    assert Line3D(Point3D(4, 0, 1), Point3D(0, 4, 1)).is_parallel(Line3D(Point3D(0, 0, 1), Point3D(4, 4, 1))) is False


def test_is_perpendicular():
    p1 = Point(0, 0)
    p2 = Point(1, 1)

    l1 = Line(p1, p2)
    l2 = Line(Point(x1, x1), Point(y1, y1))
    l1_1 = Line(p1, Point(-x1, x1))
    # 2D
    assert Line.is_perpendicular(l1, l1_1)
    assert Line.is_perpendicular(l1, l2) is False
    p = l1.random_point()
    assert l1.perpendicular_segment(p) == p
    # 3D
    assert Line3D.is_perpendicular(Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0)),
                                   Line3D(Point3D(0, 0, 0), Point3D(0, 1, 0))) is True
    assert Line3D.is_perpendicular(Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0)),
                                   Line3D(Point3D(0, 1, 0), Point3D(1, 1, 0))) is False
    assert Line3D.is_perpendicular(Line3D(Point3D(0, 0, 0), Point3D(1, 1, 1)),
                                   Line3D(Point3D(x1, x1, x1), Point3D(y1, y1, y1))) is False


def test_is_similar():
    p1 = Point(2000, 2000)
    p2 = p1.scale(2, 2)

    r1 = Ray3D(Point3D(1, 1, 1), Point3D(1, 0, 0))
    r2 = Ray(Point(0, 0), Point(0, 1))

    s1 = Segment(Point(0, 0), p1)

    assert s1.is_similar(Segment(p1, p2))
    assert s1.is_similar(r2) is False
    assert r1.is_similar(Line3D(Point3D(1, 1, 1), Point3D(1, 0, 0))) is True
    assert r1.is_similar(Line3D(Point3D(0, 0, 0), Point3D(0, 1, 0))) is False


def test_length():
    s2 = Segment3D(Point3D(x1, x1, x1), Point3D(y1, y1, y1))
    assert Line(Point(0, 0), Point(1, 1)).length is oo
    assert s2.length == sqrt(3) * sqrt((x1 - y1) ** 2)
    assert Line3D(Point3D(0, 0, 0), Point3D(1, 1, 1)).length is oo


def test_projection():
    p1 = Point(0, 0)
    p2 = Point3D(0, 0, 0)
    p3 = Point(-x1, x1)

    l1 = Line(p1, Point(1, 1))
    l2 = Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0))
    l3 = Line3D(p2, Point3D(1, 1, 1))

    r1 = Ray(Point(1, 1), Point(2, 2))

    s1 = Segment(Point2D(0, 0), Point2D(0, 1))
    s2 = Segment(Point2D(1, 0), Point2D(2, 1/2))

    assert Line(Point(x1, x1), Point(y1, y1)).projection(Point(y1, y1)) == Point(y1, y1)
    assert Line(Point(x1, x1), Point(x1, 1 + x1)).projection(Point(1, 1)) == Point(x1, 1)
    assert Segment(Point(-2, 2), Point(0, 4)).projection(r1) == Segment(Point(-1, 3), Point(0, 4))
    assert Segment(Point(0, 4), Point(-2, 2)).projection(r1) == Segment(Point(0, 4), Point(-1, 3))
    assert s2.projection(s1) == EmptySet
    assert l1.projection(p3) == p1
    assert l1.projection(Ray(p1, Point(-1, 5))) == Ray(Point(0, 0), Point(2, 2))
    assert l1.projection(Ray(p1, Point(-1, 1))) == p1
    assert r1.projection(Ray(Point(1, 1), Point(-1, -1))) == Point(1, 1)
    assert r1.projection(Ray(Point(0, 4), Point(-1, -5))) == Segment(Point(1, 1), Point(2, 2))
    assert r1.projection(Segment(Point(-1, 5), Point(-5, -10))) == Segment(Point(1, 1), Point(2, 2))
    assert r1.projection(Ray(Point(1, 1), Point(-1, -1))) == Point(1, 1)
    assert r1.projection(Ray(Point(0, 4), Point(-1, -5))) == Segment(Point(1, 1), Point(2, 2))
    assert r1.projection(Segment(Point(-1, 5), Point(-5, -10))) == Segment(Point(1, 1), Point(2, 2))

    assert l3.projection(Ray3D(p2, Point3D(-1, 5, 0))) == Ray3D(Point3D(0, 0, 0), Point3D(Rational(4, 3), Rational(4, 3), Rational(4, 3)))
    assert l3.projection(Ray3D(p2, Point3D(-1, 1, 1))) == Ray3D(Point3D(0, 0, 0), Point3D(Rational(1, 3), Rational(1, 3), Rational(1, 3)))
    assert l2.projection(Point3D(5, 5, 0)) == Point3D(5, 0)
    assert l2.projection(Line3D(Point3D(0, 1, 0), Point3D(1, 1, 0))).equals(l2)


def test_perpendicular_line():
    # 3d - requires a particular orthogonal to be selected
    p1, p2, p3 = Point(0, 0, 0), Point(2, 3, 4), Point(-2, 2, 0)
    l = Line(p1, p2)
    p = l.perpendicular_line(p3)
    assert p.p1 == p3
    assert p.p2 in l
    # 2d - does not require special selection
    p1, p2, p3 = Point(0, 0), Point(2, 3), Point(-2, 2)
    l = Line(p1, p2)
    p = l.perpendicular_line(p3)
    assert p.p1 == p3
    # p is directed from l to p3
    assert p.direction.unit == (p3 - l.projection(p3)).unit


def test_perpendicular_bisector():
    s1 = Segment(Point(0, 0), Point(1, 1))
    aline = Line(Point(S.Half, S.Half), Point(Rational(3, 2), Rational(-1, 2)))
    on_line = Segment(Point(S.Half, S.Half), Point(Rational(3, 2), Rational(-1, 2))).midpoint

    assert s1.perpendicular_bisector().equals(aline)
    assert s1.perpendicular_bisector(on_line).equals(Segment(s1.midpoint, on_line))
    assert s1.perpendicular_bisector(on_line + (1, 0)).equals(aline)


def test_raises():
    d, e = symbols('a,b', real=True)
    s = Segment((d, 0), (e, 0))

    raises(TypeError, lambda: Line((1, 1), 1))
    raises(ValueError, lambda: Line(Point(0, 0), Point(0, 0)))
    raises(Undecidable, lambda: Point(2 * d, 0) in s)
    raises(ValueError, lambda: Ray3D(Point(1.0, 1.0)))
    raises(ValueError, lambda: Line3D(Point3D(0, 0, 0), Point3D(0, 0, 0)))
    raises(TypeError, lambda: Line3D((1, 1), 1))
    raises(ValueError, lambda: Line3D(Point3D(0, 0, 0)))
    raises(TypeError, lambda: Ray((1, 1), 1))
    raises(GeometryError, lambda: Line(Point(0, 0), Point(1, 0))
           .projection(Circle(Point(0, 0), 1)))


def test_ray_generation():
    assert Ray((1, 1), angle=pi / 4) == Ray((1, 1), (2, 2))
    assert Ray((1, 1), angle=pi / 2) == Ray((1, 1), (1, 2))
    assert Ray((1, 1), angle=-pi / 2) == Ray((1, 1), (1, 0))
    assert Ray((1, 1), angle=-3 * pi / 2) == Ray((1, 1), (1, 2))
    assert Ray((1, 1), angle=5 * pi / 2) == Ray((1, 1), (1, 2))
    assert Ray((1, 1), angle=5.0 * pi / 2) == Ray((1, 1), (1, 2))
    assert Ray((1, 1), angle=pi) == Ray((1, 1), (0, 1))
    assert Ray((1, 1), angle=3.0 * pi) == Ray((1, 1), (0, 1))
    assert Ray((1, 1), angle=4.0 * pi) == Ray((1, 1), (2, 1))
    assert Ray((1, 1), angle=0) == Ray((1, 1), (2, 1))
    assert Ray((1, 1), angle=4.05 * pi) == Ray(Point(1, 1),
                                               Point(2, -sqrt(5) * sqrt(2 * sqrt(5) + 10) / 4 - sqrt(
                                                   2 * sqrt(5) + 10) / 4 + 2 + sqrt(5)))
    assert Ray((1, 1), angle=4.02 * pi) == Ray(Point(1, 1),
                                               Point(2, 1 + tan(4.02 * pi)))
    assert Ray((1, 1), angle=5) == Ray((1, 1), (2, 1 + tan(5)))

    assert Ray3D((1, 1, 1), direction_ratio=[4, 4, 4]) == Ray3D(Point3D(1, 1, 1), Point3D(5, 5, 5))
    assert Ray3D((1, 1, 1), direction_ratio=[1, 2, 3]) == Ray3D(Point3D(1, 1, 1), Point3D(2, 3, 4))
    assert Ray3D((1, 1, 1), direction_ratio=[1, 1, 1]) == Ray3D(Point3D(1, 1, 1), Point3D(2, 2, 2))


def test_issue_7814():
    circle = Circle(Point(x, 0), y)
    line = Line(Point(k, z), slope=0)
    _s = sqrt((y - z)*(y + z))
    assert line.intersection(circle) == [Point2D(x + _s, z), Point2D(x - _s, z)]


def test_issue_2941():
    def _check():
        for f, g in cartes(*[(Line, Ray, Segment)] * 2):
            l1 = f(a, b)
            l2 = g(c, d)
            assert l1.intersection(l2) == l2.intersection(l1)
    # intersect at end point
    c, d = (-2, -2), (-2, 0)
    a, b = (0, 0), (1, 1)
    _check()
    # midline intersection
    c, d = (-2, -3), (-2, 0)
    _check()


def test_parameter_value():
    t = Symbol('t')
    p1, p2 = Point(0, 1), Point(5, 6)
    l = Line(p1, p2)
    assert l.parameter_value((5, 6), t) == {t: 1}
    raises(ValueError, lambda: l.parameter_value((0, 0), t))


def test_bisectors():
    r1 = Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0))
    r2 = Line3D(Point3D(0, 0, 0), Point3D(0, 1, 0))
    bisections = r1.bisectors(r2)
    assert bisections == [Line3D(Point3D(0, 0, 0), Point3D(1, 1, 0)),
        Line3D(Point3D(0, 0, 0), Point3D(1, -1, 0))]
    ans = [Line3D(Point3D(0, 0, 0), Point3D(1, 0, 1)),
        Line3D(Point3D(0, 0, 0), Point3D(-1, 0, 1))]
    l1 = (0, 0, 0), (0, 0, 1)
    l2 = (0, 0), (1, 0)
    for a, b in cartes((Line, Segment, Ray), repeat=2):
        assert a(*l1).bisectors(b(*l2)) == ans


def test_issue_8615():
    a = Line3D(Point3D(6, 5, 0), Point3D(6, -6, 0))
    b = Line3D(Point3D(6, -1, 19/10), Point3D(6, -1, 0))
    assert a.intersection(b) == [Point3D(6, -1, 0)]


def test_issue_12598():
    r1 = Ray(Point(0, 1), Point(0.98, 0.79).n(2))
    r2 = Ray(Point(0, 0), Point(0.71, 0.71).n(2))
    assert str(r1.intersection(r2)[0]) == 'Point2D(0.82, 0.82)'
    l1 = Line((0, 0), (1, 1))
    l2 = Segment((-1, 1), (0, -1)).n(2)
    assert str(l1.intersection(l2)[0]) == 'Point2D(-0.33, -0.33)'
    l2 = Segment((-1, 1), (-1/2, 1/2)).n(2)
    assert not l1.intersection(l2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_constructors.py ===
from datetime import (
    date,
    datetime,
)
import itertools

import numpy as np
import pytest

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike

import pandas as pd
from pandas import (
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


def test_constructor_single_level():
    result = MultiIndex(
        levels=[["foo", "bar", "baz", "qux"]], codes=[[0, 1, 2, 3]], names=["first"]
    )
    assert isinstance(result, MultiIndex)
    expected = Index(["foo", "bar", "baz", "qux"], name="first")
    tm.assert_index_equal(result.levels[0], expected)
    assert result.names == ["first"]


def test_constructor_no_levels():
    msg = "non-zero number of levels/codes"
    with pytest.raises(ValueError, match=msg):
        MultiIndex(levels=[], codes=[])

    msg = "Must pass both levels and codes"
    with pytest.raises(TypeError, match=msg):
        MultiIndex(levels=[])
    with pytest.raises(TypeError, match=msg):
        MultiIndex(codes=[])


def test_constructor_nonhashable_names():
    # GH 20527
    levels = [[1, 2], ["one", "two"]]
    codes = [[0, 0, 1, 1], [0, 1, 0, 1]]
    names = (["foo"], ["bar"])
    msg = r"MultiIndex\.name must be a hashable type"
    with pytest.raises(TypeError, match=msg):
        MultiIndex(levels=levels, codes=codes, names=names)

    # With .rename()
    mi = MultiIndex(
        levels=[[1, 2], ["one", "two"]],
        codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
        names=("foo", "bar"),
    )
    renamed = [["fooo"], ["barr"]]
    with pytest.raises(TypeError, match=msg):
        mi.rename(names=renamed)

    # With .set_names()
    with pytest.raises(TypeError, match=msg):
        mi.set_names(names=renamed)


def test_constructor_mismatched_codes_levels(idx):
    codes = [np.array([1]), np.array([2]), np.array([3])]
    levels = ["a"]

    msg = "Length of levels and codes must be the same"
    with pytest.raises(ValueError, match=msg):
        MultiIndex(levels=levels, codes=codes)

    length_error = (
        r"On level 0, code max \(3\) >= length of level \(1\)\. "
        "NOTE: this index is in an inconsistent state"
    )
    label_error = r"Unequal code lengths: \[4, 2\]"
    code_value_error = r"On level 0, code value \(-2\) < -1"

    # important to check that it's looking at the right thing.
    with pytest.raises(ValueError, match=length_error):
        MultiIndex(levels=[["a"], ["b"]], codes=[[0, 1, 2, 3], [0, 3, 4, 1]])

    with pytest.raises(ValueError, match=label_error):
        MultiIndex(levels=[["a"], ["b"]], codes=[[0, 0, 0, 0], [0, 0]])

    # external API
    with pytest.raises(ValueError, match=length_error):
        idx.copy().set_levels([["a"], ["b"]])

    with pytest.raises(ValueError, match=label_error):
        idx.copy().set_codes([[0, 0, 0, 0], [0, 0]])

    # test set_codes with verify_integrity=False
    # the setting should not raise any value error
    idx.copy().set_codes(codes=[[0, 0, 0, 0], [0, 0]], verify_integrity=False)

    # code value smaller than -1
    with pytest.raises(ValueError, match=code_value_error):
        MultiIndex(levels=[["a"], ["b"]], codes=[[0, -2], [0, 0]])


def test_na_levels():
    # GH26408
    # test if codes are re-assigned value -1 for levels
    # with missing values (NaN, NaT, None)
    result = MultiIndex(
        levels=[[np.nan, None, pd.NaT, 128, 2]], codes=[[0, -1, 1, 2, 3, 4]]
    )
    expected = MultiIndex(
        levels=[[np.nan, None, pd.NaT, 128, 2]], codes=[[-1, -1, -1, -1, 3, 4]]
    )
    tm.assert_index_equal(result, expected)

    result = MultiIndex(
        levels=[[np.nan, "s", pd.NaT, 128, None]], codes=[[0, -1, 1, 2, 3, 4]]
    )
    expected = MultiIndex(
        levels=[[np.nan, "s", pd.NaT, 128, None]], codes=[[-1, -1, 1, -1, 3, -1]]
    )
    tm.assert_index_equal(result, expected)

    # verify set_levels and set_codes
    result = MultiIndex(
        levels=[[1, 2, 3, 4, 5]], codes=[[0, -1, 1, 2, 3, 4]]
    ).set_levels([[np.nan, "s", pd.NaT, 128, None]])
    tm.assert_index_equal(result, expected)

    result = MultiIndex(
        levels=[[np.nan, "s", pd.NaT, 128, None]], codes=[[1, 2, 2, 2, 2, 2]]
    ).set_codes([[0, -1, 1, 2, 3, 4]])
    tm.assert_index_equal(result, expected)


def test_copy_in_constructor():
    levels = np.array(["a", "b", "c"])
    codes = np.array([1, 1, 2, 0, 0, 1, 1])
    val = codes[0]
    mi = MultiIndex(levels=[levels, levels], codes=[codes, codes], copy=True)
    assert mi.codes[0][0] == val
    codes[0] = 15
    assert mi.codes[0][0] == val
    val = levels[0]
    levels[0] = "PANDA"
    assert mi.levels[0][0] == val


# ----------------------------------------------------------------------------
# from_arrays
# ----------------------------------------------------------------------------
def test_from_arrays(idx):
    arrays = [
        np.asarray(lev).take(level_codes)
        for lev, level_codes in zip(idx.levels, idx.codes)
    ]

    # list of arrays as input
    result = MultiIndex.from_arrays(arrays, names=idx.names)
    tm.assert_index_equal(result, idx)

    # infer correctly
    result = MultiIndex.from_arrays([[pd.NaT, Timestamp("20130101")], ["a", "b"]])
    assert result.levels[0].equals(Index([Timestamp("20130101")]))
    assert result.levels[1].equals(Index(["a", "b"]))


def test_from_arrays_iterator(idx):
    # GH 18434
    arrays = [
        np.asarray(lev).take(level_codes)
        for lev, level_codes in zip(idx.levels, idx.codes)
    ]

    # iterator as input
    result = MultiIndex.from_arrays(iter(arrays), names=idx.names)
    tm.assert_index_equal(result, idx)

    # invalid iterator input
    msg = "Input must be a list / sequence of array-likes."
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_arrays(0)


def test_from_arrays_tuples(idx):
    arrays = tuple(
        tuple(np.asarray(lev).take(level_codes))
        for lev, level_codes in zip(idx.levels, idx.codes)
    )

    # tuple of tuples as input
    result = MultiIndex.from_arrays(arrays, names=idx.names)
    tm.assert_index_equal(result, idx)


@pytest.mark.parametrize(
    ("idx1", "idx2"),
    [
        (
            pd.period_range("2011-01-01", freq="D", periods=3),
            pd.period_range("2015-01-01", freq="h", periods=3),
        ),
        (
            date_range("2015-01-01 10:00", freq="D", periods=3, tz="US/Eastern"),
            date_range("2015-01-01 10:00", freq="h", periods=3, tz="Asia/Tokyo"),
        ),
        (
            pd.timedelta_range("1 days", freq="D", periods=3),
            pd.timedelta_range("2 hours", freq="h", periods=3),
        ),
    ],
)
def test_from_arrays_index_series_period_datetimetz_and_timedelta(idx1, idx2):
    result = MultiIndex.from_arrays([idx1, idx2])
    tm.assert_index_equal(result.get_level_values(0), idx1)
    tm.assert_index_equal(result.get_level_values(1), idx2)

    result2 = MultiIndex.from_arrays([Series(idx1), Series(idx2)])
    tm.assert_index_equal(result2.get_level_values(0), idx1)
    tm.assert_index_equal(result2.get_level_values(1), idx2)

    tm.assert_index_equal(result, result2)


def test_from_arrays_index_datetimelike_mixed():
    idx1 = date_range("2015-01-01 10:00", freq="D", periods=3, tz="US/Eastern")
    idx2 = date_range("2015-01-01 10:00", freq="h", periods=3)
    idx3 = pd.timedelta_range("1 days", freq="D", periods=3)
    idx4 = pd.period_range("2011-01-01", freq="D", periods=3)

    result = MultiIndex.from_arrays([idx1, idx2, idx3, idx4])
    tm.assert_index_equal(result.get_level_values(0), idx1)
    tm.assert_index_equal(result.get_level_values(1), idx2)
    tm.assert_index_equal(result.get_level_values(2), idx3)
    tm.assert_index_equal(result.get_level_values(3), idx4)

    result2 = MultiIndex.from_arrays(
        [Series(idx1), Series(idx2), Series(idx3), Series(idx4)]
    )
    tm.assert_index_equal(result2.get_level_values(0), idx1)
    tm.assert_index_equal(result2.get_level_values(1), idx2)
    tm.assert_index_equal(result2.get_level_values(2), idx3)
    tm.assert_index_equal(result2.get_level_values(3), idx4)

    tm.assert_index_equal(result, result2)


def test_from_arrays_index_series_categorical():
    # GH13743
    idx1 = pd.CategoricalIndex(list("abcaab"), categories=list("bac"), ordered=False)
    idx2 = pd.CategoricalIndex(list("abcaab"), categories=list("bac"), ordered=True)

    result = MultiIndex.from_arrays([idx1, idx2])
    tm.assert_index_equal(result.get_level_values(0), idx1)
    tm.assert_index_equal(result.get_level_values(1), idx2)

    result2 = MultiIndex.from_arrays([Series(idx1), Series(idx2)])
    tm.assert_index_equal(result2.get_level_values(0), idx1)
    tm.assert_index_equal(result2.get_level_values(1), idx2)

    result3 = MultiIndex.from_arrays([idx1.values, idx2.values])
    tm.assert_index_equal(result3.get_level_values(0), idx1)
    tm.assert_index_equal(result3.get_level_values(1), idx2)


def test_from_arrays_empty():
    # 0 levels
    msg = "Must pass non-zero number of levels/codes"
    with pytest.raises(ValueError, match=msg):
        MultiIndex.from_arrays(arrays=[])

    # 1 level
    result = MultiIndex.from_arrays(arrays=[[]], names=["A"])
    assert isinstance(result, MultiIndex)
    expected = Index([], name="A")
    tm.assert_index_equal(result.levels[0], expected)
    assert result.names == ["A"]

    # N levels
    for N in [2, 3]:
        arrays = [[]] * N
        names = list("ABC")[:N]
        result = MultiIndex.from_arrays(arrays=arrays, names=names)
        expected = MultiIndex(levels=[[]] * N, codes=[[]] * N, names=names)
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "invalid_sequence_of_arrays",
    [
        1,
        [1],
        [1, 2],
        [[1], 2],
        [1, [2]],
        "a",
        ["a"],
        ["a", "b"],
        [["a"], "b"],
        (1,),
        (1, 2),
        ([1], 2),
        (1, [2]),
        "a",
        ("a",),
        ("a", "b"),
        (["a"], "b"),
        [(1,), 2],
        [1, (2,)],
        [("a",), "b"],
        ((1,), 2),
        (1, (2,)),
        (("a",), "b"),
    ],
)
def test_from_arrays_invalid_input(invalid_sequence_of_arrays):
    msg = "Input must be a list / sequence of array-likes"
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_arrays(arrays=invalid_sequence_of_arrays)


@pytest.mark.parametrize(
    "idx1, idx2", [([1, 2, 3], ["a", "b"]), ([], ["a", "b"]), ([1, 2, 3], [])]
)
def test_from_arrays_different_lengths(idx1, idx2):
    # see gh-13599
    msg = "^all arrays must be same length$"
    with pytest.raises(ValueError, match=msg):
        MultiIndex.from_arrays([idx1, idx2])


def test_from_arrays_respects_none_names():
    # GH27292
    a = Series([1, 2, 3], name="foo")
    b = Series(["a", "b", "c"], name="bar")

    result = MultiIndex.from_arrays([a, b], names=None)
    expected = MultiIndex(
        levels=[[1, 2, 3], ["a", "b", "c"]], codes=[[0, 1, 2], [0, 1, 2]], names=None
    )

    tm.assert_index_equal(result, expected)


# ----------------------------------------------------------------------------
# from_tuples
# ----------------------------------------------------------------------------
def test_from_tuples():
    msg = "Cannot infer number of levels from empty list"
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_tuples([])

    expected = MultiIndex(
        levels=[[1, 3], [2, 4]], codes=[[0, 1], [0, 1]], names=["a", "b"]
    )

    # input tuples
    result = MultiIndex.from_tuples(((1, 2), (3, 4)), names=["a", "b"])
    tm.assert_index_equal(result, expected)


def test_from_tuples_iterator():
    # GH 18434
    # input iterator for tuples
    expected = MultiIndex(
        levels=[[1, 3], [2, 4]], codes=[[0, 1], [0, 1]], names=["a", "b"]
    )

    result = MultiIndex.from_tuples(zip([1, 3], [2, 4]), names=["a", "b"])
    tm.assert_index_equal(result, expected)

    # input non-iterables
    msg = "Input must be a list / sequence of tuple-likes."
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_tuples(0)


def test_from_tuples_empty():
    # GH 16777
    result = MultiIndex.from_tuples([], names=["a", "b"])
    expected = MultiIndex.from_arrays(arrays=[[], []], names=["a", "b"])
    tm.assert_index_equal(result, expected)


def test_from_tuples_index_values(idx):
    result = MultiIndex.from_tuples(idx)
    assert (result.values == idx.values).all()


def test_tuples_with_name_string():
    # GH 15110 and GH 14848

    li = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    msg = "Names should be list-like for a MultiIndex"
    with pytest.raises(ValueError, match=msg):
        Index(li, name="abc")
    with pytest.raises(ValueError, match=msg):
        Index(li, name="a")


def test_from_tuples_with_tuple_label():
    # GH 15457
    expected = pd.DataFrame(
        [[2, 1, 2], [4, (1, 2), 3]], columns=["a", "b", "c"]
    ).set_index(["a", "b"])
    idx = MultiIndex.from_tuples([(2, 1), (4, (1, 2))], names=("a", "b"))
    result = pd.DataFrame([2, 3], columns=["c"], index=idx)
    tm.assert_frame_equal(expected, result)


# ----------------------------------------------------------------------------
# from_product
# ----------------------------------------------------------------------------
def test_from_product_empty_zero_levels():
    # 0 levels
    msg = "Must pass non-zero number of levels/codes"
    with pytest.raises(ValueError, match=msg):
        MultiIndex.from_product([])


def test_from_product_empty_one_level():
    result = MultiIndex.from_product([[]], names=["A"])
    expected = Index([], name="A")
    tm.assert_index_equal(result.levels[0], expected)
    assert result.names == ["A"]


@pytest.mark.parametrize(
    "first, second", [([], []), (["foo", "bar", "baz"], []), ([], ["a", "b", "c"])]
)
def test_from_product_empty_two_levels(first, second):
    names = ["A", "B"]
    result = MultiIndex.from_product([first, second], names=names)
    expected = MultiIndex(levels=[first, second], codes=[[], []], names=names)
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("N", list(range(4)))
def test_from_product_empty_three_levels(N):
    # GH12258
    names = ["A", "B", "C"]
    lvl2 = list(range(N))
    result = MultiIndex.from_product([[], lvl2, []], names=names)
    expected = MultiIndex(levels=[[], lvl2, []], codes=[[], [], []], names=names)
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "invalid_input", [1, [1], [1, 2], [[1], 2], "a", ["a"], ["a", "b"], [["a"], "b"]]
)
def test_from_product_invalid_input(invalid_input):
    msg = r"Input must be a list / sequence of iterables|Input must be list-like"
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_product(iterables=invalid_input)


def test_from_product_datetimeindex():
    dt_index = date_range("2000-01-01", periods=2)
    mi = MultiIndex.from_product([[1, 2], dt_index])
    etalon = construct_1d_object_array_from_listlike(
        [
            (1, Timestamp("2000-01-01")),
            (1, Timestamp("2000-01-02")),
            (2, Timestamp("2000-01-01")),
            (2, Timestamp("2000-01-02")),
        ]
    )
    tm.assert_numpy_array_equal(mi.values, etalon)


def test_from_product_rangeindex():
    # RangeIndex is preserved by factorize, so preserved in levels
    rng = Index(range(5))
    other = ["a", "b"]
    mi = MultiIndex.from_product([rng, other])
    tm.assert_index_equal(mi._levels[0], rng, exact=True)


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("f", [lambda x: x, lambda x: Series(x), lambda x: x.values])
def test_from_product_index_series_categorical(ordered, f):
    # GH13743
    first = ["foo", "bar"]

    idx = pd.CategoricalIndex(list("abcaab"), categories=list("bac"), ordered=ordered)
    expected = pd.CategoricalIndex(
        list("abcaab") + list("abcaab"), categories=list("bac"), ordered=ordered
    )

    result = MultiIndex.from_product([first, f(idx)])
    tm.assert_index_equal(result.get_level_values(1), expected)


def test_from_product():
    first = ["foo", "bar", "buz"]
    second = ["a", "b", "c"]
    names = ["first", "second"]
    result = MultiIndex.from_product([first, second], names=names)

    tuples = [
        ("foo", "a"),
        ("foo", "b"),
        ("foo", "c"),
        ("bar", "a"),
        ("bar", "b"),
        ("bar", "c"),
        ("buz", "a"),
        ("buz", "b"),
        ("buz", "c"),
    ]
    expected = MultiIndex.from_tuples(tuples, names=names)

    tm.assert_index_equal(result, expected)


def test_from_product_iterator():
    # GH 18434
    first = ["foo", "bar", "buz"]
    second = ["a", "b", "c"]
    names = ["first", "second"]
    tuples = [
        ("foo", "a"),
        ("foo", "b"),
        ("foo", "c"),
        ("bar", "a"),
        ("bar", "b"),
        ("bar", "c"),
        ("buz", "a"),
        ("buz", "b"),
        ("buz", "c"),
    ]
    expected = MultiIndex.from_tuples(tuples, names=names)

    # iterator as input
    result = MultiIndex.from_product(iter([first, second]), names=names)
    tm.assert_index_equal(result, expected)

    # Invalid non-iterable input
    msg = "Input must be a list / sequence of iterables."
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_product(0)


@pytest.mark.parametrize(
    "a, b, expected_names",
    [
        (
            Series([1, 2, 3], name="foo"),
            Series(["a", "b"], name="bar"),
            ["foo", "bar"],
        ),
        (Series([1, 2, 3], name="foo"), ["a", "b"], ["foo", None]),
        ([1, 2, 3], ["a", "b"], None),
    ],
)
def test_from_product_infer_names(a, b, expected_names):
    # GH27292
    result = MultiIndex.from_product([a, b])
    expected = MultiIndex(
        levels=[[1, 2, 3], ["a", "b"]],
        codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        names=expected_names,
    )
    tm.assert_index_equal(result, expected)


def test_from_product_respects_none_names():
    # GH27292
    a = Series([1, 2, 3], name="foo")
    b = Series(["a", "b"], name="bar")

    result = MultiIndex.from_product([a, b], names=None)
    expected = MultiIndex(
        levels=[[1, 2, 3], ["a", "b"]],
        codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        names=None,
    )
    tm.assert_index_equal(result, expected)


def test_from_product_readonly():
    # GH#15286 passing read-only array to from_product
    a = np.array(range(3))
    b = ["a", "b"]
    expected = MultiIndex.from_product([a, b])

    a.setflags(write=False)
    result = MultiIndex.from_product([a, b])
    tm.assert_index_equal(result, expected)


def test_create_index_existing_name(idx):
    # GH11193, when an existing index is passed, and a new name is not
    # specified, the new index should inherit the previous object name
    index = idx
    index.names = ["foo", "bar"]
    result = Index(index)
    expected = Index(
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
        )
    )
    tm.assert_index_equal(result, expected)

    result = Index(index, name="A")
    expected = Index(
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
        name="A",
    )
    tm.assert_index_equal(result, expected)


# ----------------------------------------------------------------------------
# from_frame
# ----------------------------------------------------------------------------
def test_from_frame():
    # GH 22420
    df = pd.DataFrame(
        [["a", "a"], ["a", "b"], ["b", "a"], ["b", "b"]], columns=["L1", "L2"]
    )
    expected = MultiIndex.from_tuples(
        [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")], names=["L1", "L2"]
    )
    result = MultiIndex.from_frame(df)
    tm.assert_index_equal(expected, result)


def test_from_frame_missing_values_multiIndex():
    # GH 39984
    pa = pytest.importorskip("pyarrow")

    df = pd.DataFrame(
        {
            "a": Series([1, 2, None], dtype="Int64"),
            "b": pd.Float64Dtype().__from_arrow__(pa.array([0.2, np.nan, None])),
        }
    )
    multi_indexed = MultiIndex.from_frame(df)
    expected = MultiIndex.from_arrays(
        [
            Series([1, 2, None]).astype("Int64"),
            pd.Float64Dtype().__from_arrow__(pa.array([0.2, np.nan, None])),
        ],
        names=["a", "b"],
    )
    tm.assert_index_equal(multi_indexed, expected)


@pytest.mark.parametrize(
    "non_frame",
    [
        Series([1, 2, 3, 4]),
        [1, 2, 3, 4],
        [[1, 2], [3, 4], [5, 6]],
        Index([1, 2, 3, 4]),
        np.array([[1, 2], [3, 4], [5, 6]]),
        27,
    ],
)
def test_from_frame_error(non_frame):
    # GH 22420
    with pytest.raises(TypeError, match="Input must be a DataFrame"):
        MultiIndex.from_frame(non_frame)


def test_from_frame_dtype_fidelity():
    # GH 22420
    df = pd.DataFrame(
        {
            "dates": date_range("19910905", periods=6, tz="US/Eastern"),
            "a": [1, 1, 1, 2, 2, 2],
            "b": pd.Categorical(["a", "a", "b", "b", "c", "c"], ordered=True),
            "c": ["x", "x", "y", "z", "x", "y"],
        }
    )
    original_dtypes = df.dtypes.to_dict()

    expected_mi = MultiIndex.from_arrays(
        [
            date_range("19910905", periods=6, tz="US/Eastern"),
            [1, 1, 1, 2, 2, 2],
            pd.Categorical(["a", "a", "b", "b", "c", "c"], ordered=True),
            ["x", "x", "y", "z", "x", "y"],
        ],
        names=["dates", "a", "b", "c"],
    )
    mi = MultiIndex.from_frame(df)
    mi_dtypes = {name: mi.levels[i].dtype for i, name in enumerate(mi.names)}

    tm.assert_index_equal(expected_mi, mi)
    assert original_dtypes == mi_dtypes


@pytest.mark.parametrize(
    "names_in,names_out", [(None, [("L1", "x"), ("L2", "y")]), (["x", "y"], ["x", "y"])]
)
def test_from_frame_valid_names(names_in, names_out):
    # GH 22420
    df = pd.DataFrame(
        [["a", "a"], ["a", "b"], ["b", "a"], ["b", "b"]],
        columns=MultiIndex.from_tuples([("L1", "x"), ("L2", "y")]),
    )
    mi = MultiIndex.from_frame(df, names=names_in)
    assert mi.names == names_out


@pytest.mark.parametrize(
    "names,expected_error_msg",
    [
        ("bad_input", "Names should be list-like for a MultiIndex"),
        (["a", "b", "c"], "Length of names must match number of levels in MultiIndex"),
    ],
)
def test_from_frame_invalid_names(names, expected_error_msg):
    # GH 22420
    df = pd.DataFrame(
        [["a", "a"], ["a", "b"], ["b", "a"], ["b", "b"]],
        columns=MultiIndex.from_tuples([("L1", "x"), ("L2", "y")]),
    )
    with pytest.raises(ValueError, match=expected_error_msg):
        MultiIndex.from_frame(df, names=names)


def test_index_equal_empty_iterable():
    # #16844
    a = MultiIndex(levels=[[], []], codes=[[], []], names=["a", "b"])
    b = MultiIndex.from_arrays(arrays=[[], []], names=["a", "b"])
    tm.assert_index_equal(a, b)


def test_raise_invalid_sortorder():
    # Test that the MultiIndex constructor raise when a incorrect sortorder is given
    # GH#28518

    levels = [[0, 1], [0, 1, 2]]

    # Correct sortorder
    MultiIndex(
        levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]], sortorder=2
    )

    with pytest.raises(ValueError, match=r".* sortorder 2 with lexsort_depth 1.*"):
        MultiIndex(
            levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 2, 1]], sortorder=2
        )

    with pytest.raises(ValueError, match=r".* sortorder 1 with lexsort_depth 0.*"):
        MultiIndex(
            levels=levels, codes=[[0, 0, 1, 0, 1, 1], [0, 1, 0, 2, 2, 1]], sortorder=1
        )


def test_datetimeindex():
    idx1 = pd.DatetimeIndex(
        ["2013-04-01 9:00", "2013-04-02 9:00", "2013-04-03 9:00"] * 2, tz="Asia/Tokyo"
    )
    idx2 = date_range("2010/01/01", periods=6, freq="ME", tz="US/Eastern")
    idx = MultiIndex.from_arrays([idx1, idx2])

    expected1 = pd.DatetimeIndex(
        ["2013-04-01 9:00", "2013-04-02 9:00", "2013-04-03 9:00"], tz="Asia/Tokyo"
    )

    tm.assert_index_equal(idx.levels[0], expected1)
    tm.assert_index_equal(idx.levels[1], idx2)

    # from datetime combos
    # GH 7888
    date1 = np.datetime64("today")
    date2 = datetime.today()
    date3 = Timestamp.today()

    for d1, d2 in itertools.product([date1, date2, date3], [date1, date2, date3]):
        index = MultiIndex.from_product([[d1], [d2]])
        assert isinstance(index.levels[0], pd.DatetimeIndex)
        assert isinstance(index.levels[1], pd.DatetimeIndex)

    # but NOT date objects, matching Index behavior
    date4 = date.today()
    index = MultiIndex.from_product([[date4], [date2]])
    assert not isinstance(index.levels[0], pd.DatetimeIndex)
    assert isinstance(index.levels[1], pd.DatetimeIndex)


def test_constructor_with_tz():
    index = pd.DatetimeIndex(
        ["2013/01/01 09:00", "2013/01/02 09:00"], name="dt1", tz="US/Pacific"
    )
    columns = pd.DatetimeIndex(
        ["2014/01/01 09:00", "2014/01/02 09:00"], name="dt2", tz="Asia/Tokyo"
    )

    result = MultiIndex.from_arrays([index, columns])

    assert result.names == ["dt1", "dt2"]
    tm.assert_index_equal(result.levels[0], index)
    tm.assert_index_equal(result.levels[1], columns)

    result = MultiIndex.from_arrays([Series(index), Series(columns)])

    assert result.names == ["dt1", "dt2"]
    tm.assert_index_equal(result.levels[0], index)
    tm.assert_index_equal(result.levels[1], columns)


def test_multiindex_inference_consistency():
    # check that inference behavior matches the base class

    v = date.today()

    arr = [v, v]

    idx = Index(arr)
    assert idx.dtype == object

    mi = MultiIndex.from_arrays([arr])
    lev = mi.levels[0]
    assert lev.dtype == object

    mi = MultiIndex.from_product([arr])
    lev = mi.levels[0]
    assert lev.dtype == object

    mi = MultiIndex.from_tuples([(x,) for x in arr])
    lev = mi.levels[0]
    assert lev.dtype == object


def test_dtype_representation(using_infer_string):
    # GH#46900
    pmidx = MultiIndex.from_arrays([[1], ["a"]], names=[("a", "b"), ("c", "d")])
    result = pmidx.dtypes
    exp = "object" if not using_infer_string else pd.StringDtype(na_value=np.nan)
    expected = Series(
        ["int64", exp],
        index=MultiIndex.from_tuples([("a", "b"), ("c", "d")]),
        dtype=object,
    )
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\tests\test_error_functions.py ===
from sympy.core.function import (diff, expand, expand_func)
from sympy.core import EulerGamma
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols, Dummy)
from sympy.functions.elementary.complexes import (conjugate, im, polar_lift, re)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.functions.special.error_functions import (Chi, Ci, E1, Ei, Li, Shi, Si, erf, erf2, erf2inv, erfc, erfcinv, erfi, erfinv, expint, fresnelc, fresnels, li)
from sympy.functions.special.gamma_functions import (gamma, uppergamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.integrals.integrals import (Integral, integrate)
from sympy.series.gruntz import gruntz
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.functions.special.error_functions import _erfs, _eis
from sympy.testing.pytest import raises

x, y, z = symbols('x,y,z')
w = Symbol("w", real=True)
n = Symbol("n", integer=True)
t = Dummy('t')


def test_erf():
    assert erf(nan) is nan

    assert erf(oo) == 1
    assert erf(-oo) == -1

    assert erf(0) is S.Zero

    assert erf(I*oo) == oo*I
    assert erf(-I*oo) == -oo*I

    assert erf(-2) == -erf(2)
    assert erf(-x*y) == -erf(x*y)
    assert erf(-x - y) == -erf(x + y)

    assert erf(erfinv(x)) == x
    assert erf(erfcinv(x)) == 1 - x
    assert erf(erf2inv(0, x)) == x
    assert erf(erf2inv(0, x, evaluate=False)) == x # To cover code in erf
    assert erf(erf2inv(0, erf(erfcinv(1 - erf(erfinv(x)))))) == x

    alpha = symbols('alpha', extended_real=True)
    assert erf(alpha).is_real is True
    assert erf(alpha).is_finite is True
    alpha = symbols('alpha', extended_real=False)
    assert erf(alpha).is_real is None
    assert erf(alpha).is_finite is None
    assert erf(alpha).is_zero is None
    assert erf(alpha).is_positive is None
    assert erf(alpha).is_negative is None
    alpha = symbols('alpha', extended_positive=True)
    assert erf(alpha).is_positive is True
    alpha = symbols('alpha', extended_negative=True)
    assert erf(alpha).is_negative is True
    assert erf(I).is_real is False
    assert erf(0, evaluate=False).is_real
    assert erf(0, evaluate=False).is_zero

    assert conjugate(erf(z)) == erf(conjugate(z))

    assert erf(x).as_leading_term(x) == 2*x/sqrt(pi)
    assert erf(x*y).as_leading_term(y) == 2*x*y/sqrt(pi)
    assert (erf(x*y)/erf(y)).as_leading_term(y) == x
    assert erf(1/x).as_leading_term(x) == S.One

    assert erf(z).rewrite('uppergamma') == sqrt(z**2)*(1 - erfc(sqrt(z**2)))/z
    assert erf(z).rewrite('erfc') == S.One - erfc(z)
    assert erf(z).rewrite('erfi') == -I*erfi(I*z)
    assert erf(z).rewrite('fresnels') == (1 + I)*(fresnelc(z*(1 - I)/sqrt(pi)) -
        I*fresnels(z*(1 - I)/sqrt(pi)))
    assert erf(z).rewrite('fresnelc') == (1 + I)*(fresnelc(z*(1 - I)/sqrt(pi)) -
        I*fresnels(z*(1 - I)/sqrt(pi)))
    assert erf(z).rewrite('hyper') == 2*z*hyper([S.Half], [3*S.Half], -z**2)/sqrt(pi)
    assert erf(z).rewrite('meijerg') == z*meijerg([S.Half], [], [0], [Rational(-1, 2)], z**2)/sqrt(pi)
    assert erf(z).rewrite('expint') == sqrt(z**2)/z - z*expint(S.Half, z**2)/sqrt(S.Pi)

    assert limit(exp(x)*exp(x**2)*(erf(x + 1/exp(x)) - erf(x)), x, oo) == \
        2/sqrt(pi)
    assert limit((1 - erf(z))*exp(z**2)*z, z, oo) == 1/sqrt(pi)
    assert limit((1 - erf(x))*exp(x**2)*sqrt(pi)*x, x, oo) == 1
    assert limit(((1 - erf(x))*exp(x**2)*sqrt(pi)*x - 1)*2*x**2, x, oo) == -1
    assert limit(erf(x)/x, x, 0) == 2/sqrt(pi)
    assert limit(x**(-4) - sqrt(pi)*erf(x**2) / (2*x**6), x, 0) == S(1)/3

    assert erf(x).as_real_imag() == \
        (erf(re(x) - I*im(x))/2 + erf(re(x) + I*im(x))/2,
         -I*(-erf(re(x) - I*im(x)) + erf(re(x) + I*im(x)))/2)

    assert erf(x).as_real_imag(deep=False) == \
        (erf(re(x) - I*im(x))/2 + erf(re(x) + I*im(x))/2,
         -I*(-erf(re(x) - I*im(x)) + erf(re(x) + I*im(x)))/2)

    assert erf(w).as_real_imag() == (erf(w), 0)
    assert erf(w).as_real_imag(deep=False) == (erf(w), 0)
    # issue 13575
    assert erf(I).as_real_imag() == (0, -I*erf(I))

    raises(ArgumentIndexError, lambda: erf(x).fdiff(2))

    assert erf(x).inverse() == erfinv


def test_erf_series():
    assert erf(x).series(x, 0, 7) == 2*x/sqrt(pi) - \
        2*x**3/3/sqrt(pi) + x**5/5/sqrt(pi) + O(x**7)

    assert erf(x).series(x, oo) == \
        -exp(-x**2)*(3/(4*x**5) - 1/(2*x**3) + 1/x + O(x**(-6), (x, oo)))/sqrt(pi) + 1
    assert erf(x**2).series(x, oo, n=8) == \
        (-1/(2*x**6) + x**(-2) + O(x**(-8), (x, oo)))*exp(-x**4)/sqrt(pi)*-1 + 1
    assert erf(sqrt(x)).series(x, oo, n=3) == (sqrt(1/x) - (1/x)**(S(3)/2)/2\
        + 3*(1/x)**(S(5)/2)/4 + O(x**(-3), (x, oo)))*exp(-x)/sqrt(pi)*-1 + 1


def test_erf_evalf():
    assert abs( erf(Float(2.0)) - 0.995322265 ) < 1E-8 # XXX


def test__erfs():
    assert _erfs(z).diff(z) == -2/sqrt(S.Pi) + 2*z*_erfs(z)

    assert _erfs(1/z).series(z) == \
        z/sqrt(pi) - z**3/(2*sqrt(pi)) + 3*z**5/(4*sqrt(pi)) + O(z**6)

    assert expand(erf(z).rewrite('tractable').diff(z).rewrite('intractable')) \
        == erf(z).diff(z)
    assert _erfs(z).rewrite("intractable") == (-erf(z) + 1)*exp(z**2)
    raises(ArgumentIndexError, lambda: _erfs(z).fdiff(2))


def test_erfc():
    assert erfc(nan) is nan

    assert erfc(oo) is S.Zero
    assert erfc(-oo) == 2

    assert erfc(0) == 1

    assert erfc(I*oo) == -oo*I
    assert erfc(-I*oo) == oo*I

    assert erfc(-x) == S(2) - erfc(x)
    assert erfc(erfcinv(x)) == x

    alpha = symbols('alpha', extended_real=True)
    assert erfc(alpha).is_real is True
    alpha = symbols('alpha', extended_real=False)
    assert erfc(alpha).is_real is None
    assert erfc(I).is_real is False
    assert erfc(0, evaluate=False).is_real
    assert erfc(0, evaluate=False).is_zero is False

    assert erfc(erfinv(x)) == 1 - x

    assert conjugate(erfc(z)) == erfc(conjugate(z))

    assert erfc(x).as_leading_term(x) is S.One
    assert erfc(1/x).as_leading_term(x) == S.Zero

    assert erfc(z).rewrite('erf') == 1 - erf(z)
    assert erfc(z).rewrite('erfi') == 1 + I*erfi(I*z)
    assert erfc(z).rewrite('fresnels') == 1 - (1 + I)*(fresnelc(z*(1 - I)/sqrt(pi)) -
        I*fresnels(z*(1 - I)/sqrt(pi)))
    assert erfc(z).rewrite('fresnelc') == 1 - (1 + I)*(fresnelc(z*(1 - I)/sqrt(pi)) -
        I*fresnels(z*(1 - I)/sqrt(pi)))
    assert erfc(z).rewrite('hyper') == 1 - 2*z*hyper([S.Half], [3*S.Half], -z**2)/sqrt(pi)
    assert erfc(z).rewrite('meijerg') == 1 - z*meijerg([S.Half], [], [0], [Rational(-1, 2)], z**2)/sqrt(pi)
    assert erfc(z).rewrite('uppergamma') == 1 - sqrt(z**2)*(1 - erfc(sqrt(z**2)))/z
    assert erfc(z).rewrite('expint') == S.One - sqrt(z**2)/z + z*expint(S.Half, z**2)/sqrt(S.Pi)
    assert erfc(z).rewrite('tractable') == _erfs(z)*exp(-z**2)
    assert expand_func(erf(x) + erfc(x)) is S.One

    assert erfc(x).as_real_imag() == \
        (erfc(re(x) - I*im(x))/2 + erfc(re(x) + I*im(x))/2,
         -I*(-erfc(re(x) - I*im(x)) + erfc(re(x) + I*im(x)))/2)

    assert erfc(x).as_real_imag(deep=False) == \
        (erfc(re(x) - I*im(x))/2 + erfc(re(x) + I*im(x))/2,
         -I*(-erfc(re(x) - I*im(x)) + erfc(re(x) + I*im(x)))/2)

    assert erfc(w).as_real_imag() == (erfc(w), 0)
    assert erfc(w).as_real_imag(deep=False) == (erfc(w), 0)
    raises(ArgumentIndexError, lambda: erfc(x).fdiff(2))

    assert erfc(x).inverse() == erfcinv


def test_erfc_series():
    assert erfc(x).series(x, 0, 7) == 1 - 2*x/sqrt(pi) + \
        2*x**3/3/sqrt(pi) - x**5/5/sqrt(pi) + O(x**7)

    assert erfc(x).series(x, oo) == \
            (3/(4*x**5) - 1/(2*x**3) + 1/x + O(x**(-6), (x, oo)))*exp(-x**2)/sqrt(pi)


def test_erfc_evalf():
    assert abs( erfc(Float(2.0)) - 0.00467773 ) < 1E-8 # XXX


def test_erfi():
    assert erfi(nan) is nan

    assert erfi(oo) is S.Infinity
    assert erfi(-oo) is S.NegativeInfinity

    assert erfi(0) is S.Zero

    assert erfi(I*oo) == I
    assert erfi(-I*oo) == -I

    assert erfi(-x) == -erfi(x)

    assert erfi(I*erfinv(x)) == I*x
    assert erfi(I*erfcinv(x)) == I*(1 - x)
    assert erfi(I*erf2inv(0, x)) == I*x
    assert erfi(I*erf2inv(0, x, evaluate=False)) == I*x # To cover code in erfi

    assert erfi(I).is_real is False
    assert erfi(0, evaluate=False).is_real
    assert erfi(0, evaluate=False).is_zero

    assert conjugate(erfi(z)) == erfi(conjugate(z))

    assert erfi(x).as_leading_term(x) == 2*x/sqrt(pi)
    assert erfi(x*y).as_leading_term(y) == 2*x*y/sqrt(pi)
    assert (erfi(x*y)/erfi(y)).as_leading_term(y) == x
    assert erfi(1/x).as_leading_term(x) == erfi(1/x)

    assert erfi(z).rewrite('erf') == -I*erf(I*z)
    assert erfi(z).rewrite('erfc') == I*erfc(I*z) - I
    assert erfi(z).rewrite('fresnels') == (1 - I)*(fresnelc(z*(1 + I)/sqrt(pi)) -
        I*fresnels(z*(1 + I)/sqrt(pi)))
    assert erfi(z).rewrite('fresnelc') == (1 - I)*(fresnelc(z*(1 + I)/sqrt(pi)) -
        I*fresnels(z*(1 + I)/sqrt(pi)))
    assert erfi(z).rewrite('hyper') == 2*z*hyper([S.Half], [3*S.Half], z**2)/sqrt(pi)
    assert erfi(z).rewrite('meijerg') == z*meijerg([S.Half], [], [0], [Rational(-1, 2)], -z**2)/sqrt(pi)
    assert erfi(z).rewrite('uppergamma') == (sqrt(-z**2)/z*(uppergamma(S.Half,
        -z**2)/sqrt(S.Pi) - S.One))
    assert erfi(z).rewrite('expint') == sqrt(-z**2)/z - z*expint(S.Half, -z**2)/sqrt(S.Pi)
    assert erfi(z).rewrite('tractable') == -I*(-_erfs(I*z)*exp(z**2) + 1)
    assert expand_func(erfi(I*z)) == I*erf(z)

    assert erfi(x).as_real_imag() == \
        (erfi(re(x) - I*im(x))/2 + erfi(re(x) + I*im(x))/2,
         -I*(-erfi(re(x) - I*im(x)) + erfi(re(x) + I*im(x)))/2)
    assert erfi(x).as_real_imag(deep=False) == \
        (erfi(re(x) - I*im(x))/2 + erfi(re(x) + I*im(x))/2,
         -I*(-erfi(re(x) - I*im(x)) + erfi(re(x) + I*im(x)))/2)

    assert erfi(w).as_real_imag() == (erfi(w), 0)
    assert erfi(w).as_real_imag(deep=False) == (erfi(w), 0)

    raises(ArgumentIndexError, lambda: erfi(x).fdiff(2))


def test_erfi_series():
    assert erfi(x).series(x, 0, 7) == 2*x/sqrt(pi) + \
        2*x**3/3/sqrt(pi) + x**5/5/sqrt(pi) + O(x**7)

    assert erfi(x).series(x, oo) == \
        (3/(4*x**5) + 1/(2*x**3) + 1/x + O(x**(-6), (x, oo)))*exp(x**2)/sqrt(pi) - I


def test_erfi_evalf():
    assert abs( erfi(Float(2.0)) - 18.5648024145756 ) < 1E-13  # XXX


def test_erf2():

    assert erf2(0, 0) is S.Zero
    assert erf2(x, x) is S.Zero
    assert erf2(nan, 0) is nan

    assert erf2(-oo,  y) ==  erf(y) + 1
    assert erf2( oo,  y) ==  erf(y) - 1
    assert erf2(  x, oo) ==  1 - erf(x)
    assert erf2(  x,-oo) == -1 - erf(x)
    assert erf2(x, erf2inv(x, y)) == y

    assert erf2(-x, -y) == -erf2(x,y)
    assert erf2(-x,  y) == erf(y) + erf(x)
    assert erf2( x, -y) == -erf(y) - erf(x)
    assert erf2(x, y).rewrite('fresnels') == erf(y).rewrite(fresnels)-erf(x).rewrite(fresnels)
    assert erf2(x, y).rewrite('fresnelc') == erf(y).rewrite(fresnelc)-erf(x).rewrite(fresnelc)
    assert erf2(x, y).rewrite('hyper') == erf(y).rewrite(hyper)-erf(x).rewrite(hyper)
    assert erf2(x, y).rewrite('meijerg') == erf(y).rewrite(meijerg)-erf(x).rewrite(meijerg)
    assert erf2(x, y).rewrite('uppergamma') == erf(y).rewrite(uppergamma) - erf(x).rewrite(uppergamma)
    assert erf2(x, y).rewrite('expint') == erf(y).rewrite(expint)-erf(x).rewrite(expint)

    assert erf2(I, 0).is_real is False
    assert erf2(0, 0, evaluate=False).is_real
    assert erf2(0, 0, evaluate=False).is_zero
    assert erf2(x, x, evaluate=False).is_zero
    assert erf2(x, y).is_zero is None

    assert expand_func(erf(x) + erf2(x, y)) == erf(y)

    assert conjugate(erf2(x, y)) == erf2(conjugate(x), conjugate(y))

    assert erf2(x, y).rewrite('erf')  == erf(y) - erf(x)
    assert erf2(x, y).rewrite('erfc') == erfc(x) - erfc(y)
    assert erf2(x, y).rewrite('erfi') == I*(erfi(I*x) - erfi(I*y))

    assert erf2(x, y).diff(x) == erf2(x, y).fdiff(1)
    assert erf2(x, y).diff(y) == erf2(x, y).fdiff(2)
    assert erf2(x, y).diff(x) == -2*exp(-x**2)/sqrt(pi)
    assert erf2(x, y).diff(y) == 2*exp(-y**2)/sqrt(pi)
    raises(ArgumentIndexError, lambda: erf2(x, y).fdiff(3))

    assert erf2(x, y).is_extended_real is None
    xr, yr = symbols('xr yr', extended_real=True)
    assert erf2(xr, yr).is_extended_real is True


def test_erfinv():
    assert erfinv(0) is S.Zero
    assert erfinv(1) is S.Infinity
    assert erfinv(nan) is S.NaN
    assert erfinv(-1) is S.NegativeInfinity

    assert erfinv(erf(w)) == w
    assert erfinv(erf(-w)) == -w

    assert erfinv(x).diff() == sqrt(pi)*exp(erfinv(x)**2)/2
    raises(ArgumentIndexError, lambda: erfinv(x).fdiff(2))

    assert erfinv(z).rewrite('erfcinv') == erfcinv(1-z)
    assert erfinv(z).inverse() == erf


def test_erfinv_evalf():
    assert abs( erfinv(Float(0.2)) - 0.179143454621292 ) < 1E-13


def test_erfcinv():
    assert erfcinv(1) is S.Zero
    assert erfcinv(0) is S.Infinity
    assert erfcinv(0, evaluate=False).is_infinite is True
    assert erfcinv(2, evaluate=False).is_infinite is True
    assert erfcinv(nan) is S.NaN

    assert erfcinv(x).diff() == -sqrt(pi)*exp(erfcinv(x)**2)/2
    raises(ArgumentIndexError, lambda: erfcinv(x).fdiff(2))

    assert erfcinv(z).rewrite('erfinv') == erfinv(1-z)
    assert erfcinv(z).inverse() == erfc


def test_erf2inv():
    assert erf2inv(0, 0) is S.Zero
    assert erf2inv(0, 1) is S.Infinity
    assert erf2inv(1, 0) is S.One
    assert erf2inv(0, y) == erfinv(y)
    assert erf2inv(oo, y) == erfcinv(-y)
    assert erf2inv(x, 0) == x
    assert erf2inv(x, oo) == erfinv(x)
    assert erf2inv(nan, 0) is nan
    assert erf2inv(0, nan) is nan

    assert erf2inv(x, y).diff(x) == exp(-x**2 + erf2inv(x, y)**2)
    assert erf2inv(x, y).diff(y) == sqrt(pi)*exp(erf2inv(x, y)**2)/2
    raises(ArgumentIndexError, lambda: erf2inv(x, y).fdiff(3))


# NOTE we multiply by exp_polar(I*pi) and need this to be on the principal
# branch, hence take x in the lower half plane (d=0).


def mytn(expr1, expr2, expr3, x, d=0):
    from sympy.core.random import verify_numerically, random_complex_number
    subs = {}
    for a in expr1.free_symbols:
        if a != x:
            subs[a] = random_complex_number()
    return expr2 == expr3 and verify_numerically(expr1.subs(subs),
                                               expr2.subs(subs), x, d=d)


def mytd(expr1, expr2, x):
    from sympy.core.random import test_derivative_numerically, \
        random_complex_number
    subs = {}
    for a in expr1.free_symbols:
        if a != x:
            subs[a] = random_complex_number()
    return expr1.diff(x) == expr2 and test_derivative_numerically(expr1.subs(subs), x)


def tn_branch(func, s=None):
    from sympy.core.random import uniform

    def fn(x):
        if s is None:
            return func(x)
        return func(s, x)
    c = uniform(1, 5)
    expr = fn(c*exp_polar(I*pi)) - fn(c*exp_polar(-I*pi))
    eps = 1e-15
    expr2 = fn(-c + eps*I) - fn(-c - eps*I)
    return abs(expr.n() - expr2.n()).n() < 1e-10


def test_ei():
    assert Ei(0) is S.NegativeInfinity
    assert Ei(oo) is S.Infinity
    assert Ei(-oo) is S.Zero

    assert tn_branch(Ei)
    assert mytd(Ei(x), exp(x)/x, x)
    assert mytn(Ei(x), Ei(x).rewrite(uppergamma),
                -uppergamma(0, x*polar_lift(-1)) - I*pi, x)
    assert mytn(Ei(x), Ei(x).rewrite(expint),
                -expint(1, x*polar_lift(-1)) - I*pi, x)
    assert Ei(x).rewrite(expint).rewrite(Ei) == Ei(x)
    assert Ei(x*exp_polar(2*I*pi)) == Ei(x) + 2*I*pi
    assert Ei(x*exp_polar(-2*I*pi)) == Ei(x) - 2*I*pi

    assert mytn(Ei(x), Ei(x).rewrite(Shi), Chi(x) + Shi(x), x)
    assert mytn(Ei(x*polar_lift(I)), Ei(x*polar_lift(I)).rewrite(Si),
                Ci(x) + I*Si(x) + I*pi/2, x)

    assert Ei(log(x)).rewrite(li) == li(x)
    assert Ei(2*log(x)).rewrite(li) == li(x**2)

    assert gruntz(Ei(x+exp(-x))*exp(-x)*x, x, oo) == 1

    assert Ei(x).series(x) == EulerGamma + log(x) + x + x**2/4 + \
        x**3/18 + x**4/96 + x**5/600 + O(x**6)
    assert Ei(x).series(x, 1, 3) == Ei(1) + E*(x - 1) + O((x - 1)**3, (x, 1))
    assert Ei(x).series(x, oo) == \
        (120/x**5 + 24/x**4 + 6/x**3 + 2/x**2 + 1/x + 1 + O(x**(-6), (x, oo)))*exp(x)/x
    assert Ei(x).series(x, -oo) == \
        (120/x**5 + 24/x**4 + 6/x**3 + 2/x**2 + 1/x + 1 + O(x**(-6), (x, -oo)))*exp(x)/x
    assert Ei(-x).series(x, oo) == \
        -((-120/x**5 + 24/x**4 - 6/x**3 + 2/x**2 - 1/x + 1 + O(x**(-6), (x, oo)))*exp(-x)/x)

    assert str(Ei(cos(2)).evalf(n=10)) == '-0.6760647401'
    raises(ArgumentIndexError, lambda: Ei(x).fdiff(2))


def test_expint():
    assert mytn(expint(x, y), expint(x, y).rewrite(uppergamma),
                y**(x - 1)*uppergamma(1 - x, y), x)
    assert mytd(
        expint(x, y), -y**(x - 1)*meijerg([], [1, 1], [0, 0, 1 - x], [], y), x)
    assert mytd(expint(x, y), -expint(x - 1, y), y)
    assert mytn(expint(1, x), expint(1, x).rewrite(Ei),
                -Ei(x*polar_lift(-1)) + I*pi, x)

    assert expint(-4, x) == exp(-x)/x + 4*exp(-x)/x**2 + 12*exp(-x)/x**3 \
        + 24*exp(-x)/x**4 + 24*exp(-x)/x**5
    assert expint(Rational(-3, 2), x) == \
        exp(-x)/x + 3*exp(-x)/(2*x**2) + 3*sqrt(pi)*erfc(sqrt(x))/(4*x**S('5/2'))

    assert tn_branch(expint, 1)
    assert tn_branch(expint, 2)
    assert tn_branch(expint, 3)
    assert tn_branch(expint, 1.7)
    assert tn_branch(expint, pi)

    assert expint(y, x*exp_polar(2*I*pi)) == \
        x**(y - 1)*(exp(2*I*pi*y) - 1)*gamma(-y + 1) + expint(y, x)
    assert expint(y, x*exp_polar(-2*I*pi)) == \
        x**(y - 1)*(exp(-2*I*pi*y) - 1)*gamma(-y + 1) + expint(y, x)
    assert expint(2, x*exp_polar(2*I*pi)) == 2*I*pi*x + expint(2, x)
    assert expint(2, x*exp_polar(-2*I*pi)) == -2*I*pi*x + expint(2, x)
    assert expint(1, x).rewrite(Ei).rewrite(expint) == expint(1, x)
    assert expint(x, y).rewrite(Ei) == expint(x, y)
    assert expint(x, y).rewrite(Ci) == expint(x, y)

    assert mytn(E1(x), E1(x).rewrite(Shi), Shi(x) - Chi(x), x)
    assert mytn(E1(polar_lift(I)*x), E1(polar_lift(I)*x).rewrite(Si),
                -Ci(x) + I*Si(x) - I*pi/2, x)

    assert mytn(expint(2, x), expint(2, x).rewrite(Ei).rewrite(expint),
                -x*E1(x) + exp(-x), x)
    assert mytn(expint(3, x), expint(3, x).rewrite(Ei).rewrite(expint),
                x**2*E1(x)/2 + (1 - x)*exp(-x)/2, x)

    assert expint(Rational(3, 2), z).nseries(z) == \
        2 + 2*z - z**2/3 + z**3/15 - z**4/84 + z**5/540 - \
        2*sqrt(pi)*sqrt(z) + O(z**6)

    assert E1(z).series(z) == -EulerGamma - log(z) + z - \
        z**2/4 + z**3/18 - z**4/96 + z**5/600 + O(z**6)

    assert expint(4, z).series(z) == Rational(1, 3) - z/2 + z**2/2 + \
        z**3*(log(z)/6 - Rational(11, 36) + EulerGamma/6 - I*pi/6) - z**4/24 + \
        z**5/240 + O(z**6)

    assert expint(n, x).series(x, oo, n=3) == \
        (n*(n + 1)/x**2 - n/x + 1 + O(x**(-3), (x, oo)))*exp(-x)/x

    assert expint(z, y).series(z, 0, 2) == exp(-y)/y - z*meijerg(((), (1, 1)),
                                  ((0, 0, 1), ()), y)/y + O(z**2)
    raises(ArgumentIndexError, lambda: expint(x, y).fdiff(3))

    neg = Symbol('neg', negative=True)
    assert Ei(neg).rewrite(Si) == Shi(neg) + Chi(neg) - I*pi


def test__eis():
    assert _eis(z).diff(z) == -_eis(z) + 1/z

    assert _eis(1/z).series(z) == \
        z + z**2 + 2*z**3 + 6*z**4 + 24*z**5 + O(z**6)

    assert Ei(z).rewrite('tractable') == exp(z)*_eis(z)
    assert li(z).rewrite('tractable') == z*_eis(log(z))

    assert _eis(z).rewrite('intractable') == exp(-z)*Ei(z)

    assert expand(li(z).rewrite('tractable').diff(z).rewrite('intractable')) \
        == li(z).diff(z)

    assert expand(Ei(z).rewrite('tractable').diff(z).rewrite('intractable')) \
        == Ei(z).diff(z)

    assert _eis(z).series(z, n=3) == EulerGamma + log(z) + z*(-log(z) - \
        EulerGamma + 1) + z**2*(log(z)/2 - Rational(3, 4) + EulerGamma/2)\
        + O(z**3*log(z))
    raises(ArgumentIndexError, lambda: _eis(z).fdiff(2))


def tn_arg(func):
    def test(arg, e1, e2):
        from sympy.core.random import uniform
        v = uniform(1, 5)
        v1 = func(arg*x).subs(x, v).n()
        v2 = func(e1*v + e2*1e-15).n()
        return abs(v1 - v2).n() < 1e-10
    return test(exp_polar(I*pi/2), I, 1) and \
        test(exp_polar(-I*pi/2), -I, 1) and \
        test(exp_polar(I*pi), -1, I) and \
        test(exp_polar(-I*pi), -1, -I)


def test_li():
    z = Symbol("z")
    zr = Symbol("z", real=True)
    zp = Symbol("z", positive=True)
    zn = Symbol("z", negative=True)

    assert li(0) is S.Zero
    assert li(1) is -oo
    assert li(oo) is oo

    assert isinstance(li(z), li)
    assert unchanged(li, -zp)
    assert unchanged(li, zn)

    assert diff(li(z), z) == 1/log(z)

    assert conjugate(li(z)) == li(conjugate(z))
    assert conjugate(li(-zr)) == li(-zr)
    assert unchanged(conjugate, li(-zp))
    assert unchanged(conjugate, li(zn))

    assert li(z).rewrite(Li) == Li(z) + li(2)
    assert li(z).rewrite(Ei) == Ei(log(z))
    assert li(z).rewrite(uppergamma) == (-log(1/log(z))/2 - log(-log(z)) +
                                         log(log(z))/2 - expint(1, -log(z)))
    assert li(z).rewrite(Si) == (-log(I*log(z)) - log(1/log(z))/2 +
                                 log(log(z))/2 + Ci(I*log(z)) + Shi(log(z)))
    assert li(z).rewrite(Ci) == (-log(I*log(z)) - log(1/log(z))/2 +
                                 log(log(z))/2 + Ci(I*log(z)) + Shi(log(z)))
    assert li(z).rewrite(Shi) == (-log(1/log(z))/2 + log(log(z))/2 +
                                  Chi(log(z)) - Shi(log(z)))
    assert li(z).rewrite(Chi) == (-log(1/log(z))/2 + log(log(z))/2 +
                                  Chi(log(z)) - Shi(log(z)))
    assert li(z).rewrite(hyper) ==(log(z)*hyper((1, 1), (2, 2), log(z)) -
                                   log(1/log(z))/2 + log(log(z))/2 + EulerGamma)
    assert li(z).rewrite(meijerg) == (-log(1/log(z))/2 - log(-log(z)) + log(log(z))/2 -
                                      meijerg(((), (1,)), ((0, 0), ()), -log(z)))

    assert gruntz(1/li(z), z, oo) is S.Zero
    assert li(z).series(z) == log(z)**5/600 + log(z)**4/96 + log(z)**3/18 + log(z)**2/4 + \
            log(z) + log(log(z)) + EulerGamma
    raises(ArgumentIndexError, lambda: li(z).fdiff(2))


def test_Li():
    assert Li(2) is S.Zero
    assert Li(oo) is oo

    assert isinstance(Li(z), Li)

    assert diff(Li(z), z) == 1/log(z)

    assert gruntz(1/Li(z), z, oo) is S.Zero
    assert Li(z).rewrite(li) == li(z) - li(2)
    assert Li(z).series(z) == \
        log(z)**5/600 + log(z)**4/96 + log(z)**3/18 + log(z)**2/4 + log(z) + log(log(z)) - li(2) + EulerGamma
    raises(ArgumentIndexError, lambda: Li(z).fdiff(2))


def test_si():
    assert Si(I*x) == I*Shi(x)
    assert Shi(I*x) == I*Si(x)
    assert Si(-I*x) == -I*Shi(x)
    assert Shi(-I*x) == -I*Si(x)
    assert Si(-x) == -Si(x)
    assert Shi(-x) == -Shi(x)
    assert Si(exp_polar(2*pi*I)*x) == Si(x)
    assert Si(exp_polar(-2*pi*I)*x) == Si(x)
    assert Shi(exp_polar(2*pi*I)*x) == Shi(x)
    assert Shi(exp_polar(-2*pi*I)*x) == Shi(x)

    assert Si(oo) == pi/2
    assert Si(-oo) == -pi/2
    assert Shi(oo) is oo
    assert Shi(-oo) is -oo

    assert mytd(Si(x), sin(x)/x, x)
    assert mytd(Shi(x), sinh(x)/x, x)

    assert mytn(Si(x), Si(x).rewrite(Ei),
                -I*(-Ei(x*exp_polar(-I*pi/2))/2
               + Ei(x*exp_polar(I*pi/2))/2 - I*pi) + pi/2, x)
    assert mytn(Si(x), Si(x).rewrite(expint),
                -I*(-expint(1, x*exp_polar(-I*pi/2))/2 +
                    expint(1, x*exp_polar(I*pi/2))/2) + pi/2, x)
    assert mytn(Shi(x), Shi(x).rewrite(Ei),
                Ei(x)/2 - Ei(x*exp_polar(I*pi))/2 + I*pi/2, x)
    assert mytn(Shi(x), Shi(x).rewrite(expint),
                expint(1, x)/2 - expint(1, x*exp_polar(I*pi))/2 - I*pi/2, x)

    assert tn_arg(Si)
    assert tn_arg(Shi)

    assert Si(x)._eval_as_leading_term(x, None, 1) == x
    assert Si(2*x)._eval_as_leading_term(x, None, 1) == 2*x
    assert Si(sin(x))._eval_as_leading_term(x, None, 1) == x
    assert Si(x + 1)._eval_as_leading_term(x, None, 1) == Si(1)
    assert Si(1/x)._eval_as_leading_term(x, None, 1) == \
        Si(1/x)._eval_as_leading_term(x, None, -1) == Si(1/x)

    assert Si(x).nseries(x, n=8) == \
        x - x**3/18 + x**5/600 - x**7/35280 + O(x**8)
    assert Shi(x).nseries(x, n=8) == \
        x + x**3/18 + x**5/600 + x**7/35280 + O(x**8)
    assert Si(sin(x)).nseries(x, n=5) == x - 2*x**3/9 + O(x**5)
    assert Si(x).nseries(x, 1, n=3) == \
        Si(1) + (x - 1)*sin(1) + (x - 1)**2*(-sin(1)/2 + cos(1)/2) + O((x - 1)**3, (x, 1))

    assert Si(x).series(x, oo) == -sin(x)*(-6/x**4 + x**(-2) + O(x**(-6), (x, oo))) - \
        cos(x)*(24/x**5 - 2/x**3 + 1/x + O(x**(-6), (x, oo))) + pi/2

    t = Symbol('t', Dummy=True)
    assert Si(x).rewrite(sinc).dummy_eq(Integral(sinc(t), (t, 0, x)))

    assert limit(Shi(x), x, S.Infinity) == S.Infinity
    assert limit(Shi(x), x, S.NegativeInfinity) == S.NegativeInfinity


def test_ci():
    m1 = exp_polar(I*pi)
    m1_ = exp_polar(-I*pi)
    pI = exp_polar(I*pi/2)
    mI = exp_polar(-I*pi/2)

    assert Ci(m1*x) == Ci(x) + I*pi
    assert Ci(m1_*x) == Ci(x) - I*pi
    assert Ci(pI*x) == Chi(x) + I*pi/2
    assert Ci(mI*x) == Chi(x) - I*pi/2
    assert Chi(m1*x) == Chi(x) + I*pi
    assert Chi(m1_*x) == Chi(x) - I*pi
    assert Chi(pI*x) == Ci(x) + I*pi/2
    assert Chi(mI*x) == Ci(x) - I*pi/2
    assert Ci(exp_polar(2*I*pi)*x) == Ci(x) + 2*I*pi
    assert Chi(exp_polar(-2*I*pi)*x) == Chi(x) - 2*I*pi
    assert Chi(exp_polar(2*I*pi)*x) == Chi(x) + 2*I*pi
    assert Ci(exp_polar(-2*I*pi)*x) == Ci(x) - 2*I*pi

    assert Ci(oo) is S.Zero
    assert Ci(-oo) == I*pi
    assert Chi(oo) is oo
    assert Chi(-oo) is oo

    assert mytd(Ci(x), cos(x)/x, x)
    assert mytd(Chi(x), cosh(x)/x, x)

    assert mytn(Ci(x), Ci(x).rewrite(Ei),
                Ei(x*exp_polar(-I*pi/2))/2 + Ei(x*exp_polar(I*pi/2))/2, x)
    assert mytn(Chi(x), Chi(x).rewrite(Ei),
                Ei(x)/2 + Ei(x*exp_polar(I*pi))/2 - I*pi/2, x)

    assert tn_arg(Ci)
    assert tn_arg(Chi)

    assert Ci(x).nseries(x, n=4) == \
        EulerGamma + log(x) - x**2/4 + O(x**4)
    assert Chi(x).nseries(x, n=4) == \
        EulerGamma + log(x) + x**2/4 + O(x**4)

    assert Ci(x).series(x, oo) == -cos(x)*(-6/x**4 + x**(-2) + O(x**(-6), (x, oo))) + \
        sin(x)*(24/x**5 - 2/x**3 + 1/x + O(x**(-6), (x, oo)))

    assert Ci(x).series(x, -oo) == -cos(x)*(-6/x**4 + x**(-2) + O(x**(-6), (x, -oo))) + \
        sin(x)*(24/x**5 - 2/x**3 + 1/x + O(x**(-6), (x, -oo))) + I*pi

    assert limit(log(x) - Ci(2*x), x, 0) == -log(2) - EulerGamma
    assert Ci(x).rewrite(uppergamma) == -expint(1, x*exp_polar(-I*pi/2))/2 -\
                                        expint(1, x*exp_polar(I*pi/2))/2
    assert Ci(x).rewrite(expint) == -expint(1, x*exp_polar(-I*pi/2))/2 -\
                                        expint(1, x*exp_polar(I*pi/2))/2
    raises(ArgumentIndexError, lambda: Ci(x).fdiff(2))


def test_fresnel():
    assert fresnels(0) is S.Zero
    assert fresnels(oo) is S.Half
    assert fresnels(-oo) == Rational(-1, 2)
    assert fresnels(I*oo) == -I*S.Half

    assert unchanged(fresnels, z)
    assert fresnels(-z) == -fresnels(z)
    assert fresnels(I*z) == -I*fresnels(z)
    assert fresnels(-I*z) == I*fresnels(z)

    assert conjugate(fresnels(z)) == fresnels(conjugate(z))

    assert fresnels(z).diff(z) == sin(pi*z**2/2)

    assert fresnels(z).rewrite(erf) == (S.One + I)/4 * (
        erf((S.One + I)/2*sqrt(pi)*z) - I*erf((S.One - I)/2*sqrt(pi)*z))

    assert fresnels(z).rewrite(hyper) == \
        pi*z**3/6 * hyper([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)], -pi**2*z**4/16)

    assert fresnels(z).series(z, n=15) == \
        pi*z**3/6 - pi**3*z**7/336 + pi**5*z**11/42240 + O(z**15)

    assert fresnels(w).is_extended_real is True
    assert fresnels(w).is_finite is True

    assert fresnels(z).is_extended_real is None
    assert fresnels(z).is_finite is None

    assert fresnels(z).as_real_imag() == (fresnels(re(z) - I*im(z))/2 +
                    fresnels(re(z) + I*im(z))/2,
                    -I*(-fresnels(re(z) - I*im(z)) + fresnels(re(z) + I*im(z)))/2)

    assert fresnels(z).as_real_imag(deep=False) == (fresnels(re(z) - I*im(z))/2 +
                    fresnels(re(z) + I*im(z))/2,
                    -I*(-fresnels(re(z) - I*im(z)) + fresnels(re(z) + I*im(z)))/2)

    assert fresnels(w).as_real_imag() == (fresnels(w), 0)
    assert fresnels(w).as_real_imag(deep=True) == (fresnels(w), 0)

    assert fresnels(2 + 3*I).as_real_imag() == (
            fresnels(2 + 3*I)/2 + fresnels(2 - 3*I)/2,
            -I*(fresnels(2 + 3*I) - fresnels(2 - 3*I))/2
            )

    assert expand_func(integrate(fresnels(z), z)) == \
        z*fresnels(z) + cos(pi*z**2/2)/pi

    assert fresnels(z).rewrite(meijerg) == sqrt(2)*pi*z**Rational(9, 4) * \
        meijerg(((), (1,)), ((Rational(3, 4),),
        (Rational(1, 4), 0)), -pi**2*z**4/16)/(2*(-z)**Rational(3, 4)*(z**2)**Rational(3, 4))

    assert fresnelc(0) is S.Zero
    assert fresnelc(oo) == S.Half
    assert fresnelc(-oo) == Rational(-1, 2)
    assert fresnelc(I*oo) == I*S.Half

    assert unchanged(fresnelc, z)
    assert fresnelc(-z) == -fresnelc(z)
    assert fresnelc(I*z) == I*fresnelc(z)
    assert fresnelc(-I*z) == -I*fresnelc(z)

    assert conjugate(fresnelc(z)) == fresnelc(conjugate(z))

    assert fresnelc(z).diff(z) == cos(pi*z**2/2)

    assert fresnelc(z).rewrite(erf) == (S.One - I)/4 * (
        erf((S.One + I)/2*sqrt(pi)*z) + I*erf((S.One - I)/2*sqrt(pi)*z))

    assert fresnelc(z).rewrite(hyper) == \
        z * hyper([Rational(1, 4)], [S.Half, Rational(5, 4)], -pi**2*z**4/16)

    assert fresnelc(w).is_extended_real is True

    assert fresnelc(z).as_real_imag() == \
        (fresnelc(re(z) - I*im(z))/2 + fresnelc(re(z) + I*im(z))/2,
         -I*(-fresnelc(re(z) - I*im(z)) + fresnelc(re(z) + I*im(z)))/2)

    assert fresnelc(z).as_real_imag(deep=False) == \
        (fresnelc(re(z) - I*im(z))/2 + fresnelc(re(z) + I*im(z))/2,
         -I*(-fresnelc(re(z) - I*im(z)) + fresnelc(re(z) + I*im(z)))/2)

    assert fresnelc(2 + 3*I).as_real_imag() == (
        fresnelc(2 - 3*I)/2 + fresnelc(2 + 3*I)/2,
         -I*(fresnelc(2 + 3*I) - fresnelc(2 - 3*I))/2
    )

    assert expand_func(integrate(fresnelc(z), z)) == \
        z*fresnelc(z) - sin(pi*z**2/2)/pi

    assert fresnelc(z).rewrite(meijerg) == sqrt(2)*pi*z**Rational(3, 4) * \
        meijerg(((), (1,)), ((Rational(1, 4),),
        (Rational(3, 4), 0)), -pi**2*z**4/16)/(2*(-z)**Rational(1, 4)*(z**2)**Rational(1, 4))

    from sympy.core.random import verify_numerically

    verify_numerically(re(fresnels(z)), fresnels(z).as_real_imag()[0], z)
    verify_numerically(im(fresnels(z)), fresnels(z).as_real_imag()[1], z)
    verify_numerically(fresnels(z), fresnels(z).rewrite(hyper), z)
    verify_numerically(fresnels(z), fresnels(z).rewrite(meijerg), z)

    verify_numerically(re(fresnelc(z)), fresnelc(z).as_real_imag()[0], z)
    verify_numerically(im(fresnelc(z)), fresnelc(z).as_real_imag()[1], z)
    verify_numerically(fresnelc(z), fresnelc(z).rewrite(hyper), z)
    verify_numerically(fresnelc(z), fresnelc(z).rewrite(meijerg), z)

    raises(ArgumentIndexError, lambda: fresnels(z).fdiff(2))
    raises(ArgumentIndexError, lambda: fresnelc(z).fdiff(2))

    assert fresnels(x).taylor_term(-1, x) is S.Zero
    assert fresnelc(x).taylor_term(-1, x) is S.Zero
    assert fresnelc(x).taylor_term(1, x) == -pi**2*x**5/40


def test_fresnel_series():
    assert fresnelc(z).series(z, n=15) == \
        z - pi**2*z**5/40 + pi**4*z**9/3456 - pi**6*z**13/599040 + O(z**15)

    # issues 6510, 10102
    fs = (S.Half - sin(pi*z**2/2)/(pi**2*z**3)
        + (-1/(pi*z) + 3/(pi**3*z**5))*cos(pi*z**2/2))
    fc = (S.Half - cos(pi*z**2/2)/(pi**2*z**3)
        + (1/(pi*z) - 3/(pi**3*z**5))*sin(pi*z**2/2))
    assert fresnels(z).series(z, oo) == fs + O(z**(-6), (z, oo))
    assert fresnelc(z).series(z, oo) == fc + O(z**(-6), (z, oo))
    assert (fresnels(z).series(z, -oo) + fs.subs(z, -z)).expand().is_Order
    assert (fresnelc(z).series(z, -oo) + fc.subs(z, -z)).expand().is_Order
    assert (fresnels(1/z).series(z) - fs.subs(z, 1/z)).expand().is_Order
    assert (fresnelc(1/z).series(z) - fc.subs(z, 1/z)).expand().is_Order
    assert ((2*fresnels(3*z)).series(z, oo) - 2*fs.subs(z, 3*z)).expand().is_Order
    assert ((3*fresnelc(2*z)).series(z, oo) - 3*fc.subs(z, 2*z)).expand().is_Order


def test_integral_rewrites(): #issues 26134, 26144, 26306
    assert expint(n, x).rewrite(Integral).dummy_eq(Integral(t**-n * exp(-t*x), (t, 1, oo)))
    assert Si(x).rewrite(Integral).dummy_eq(Integral(sinc(t), (t, 0, x)))
    assert Ci(x).rewrite(Integral).dummy_eq(log(x) - Integral((1 - cos(t))/t, (t, 0, x)) + EulerGamma)
    assert fresnels(x).rewrite(Integral).dummy_eq(Integral(sin(pi*t**2/2), (t, 0, x)))
    assert fresnelc(x).rewrite(Integral).dummy_eq(Integral(cos(pi*t**2/2), (t, 0, x)))
    assert Ei(x).rewrite(Integral).dummy_eq(Integral(exp(t)/t, (t, -oo, x)))
    assert fresnels(x).diff(x) == fresnels(x).rewrite(Integral).diff(x)
    assert fresnelc(x).diff(x) == fresnelc(x).rewrite(Integral).diff(x)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_fortran.py ===
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda, diff)
from sympy.core.mod import Mod
from sympy.core import (Catalan, EulerGamma, GoldenRatio)
from sympy.core.numbers import (E, Float, I, Integer, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (conjugate, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan2, cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.sets.fancysets import Range

from sympy.codegen import For, Assignment, aug_assign
from sympy.codegen.ast import Declaration, Variable, float32, float64, \
        value_const, real, bool_, While, FunctionPrototype, FunctionDefinition, \
        integer, Return, Element
from sympy.core.expr import UnevaluatedExpr
from sympy.core.relational import Relational
from sympy.logic.boolalg import And, Or, Not, Equivalent, Xor
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.fortran import fcode, FCodePrinter
from sympy.tensor import IndexedBase, Idx
from sympy.tensor.array.expressions import ArraySymbol, ArrayElement
from sympy.utilities.lambdify import implemented_function
from sympy.testing.pytest import raises


def test_UnevaluatedExpr():
    p, q, r = symbols("p q r", real=True)
    q_r = UnevaluatedExpr(q + r)
    expr = abs(exp(p+q_r))
    assert fcode(expr, source_format="free") == "exp(p + (q + r))"
    x, y, z = symbols("x y z")
    y_z = UnevaluatedExpr(y + z)
    expr2 = abs(exp(x+y_z))
    assert fcode(expr2, human=False)[2].lstrip() == "exp(re(x) + re(y + z))"
    assert fcode(expr2, user_functions={"re": "realpart"}).lstrip() == "exp(realpart(x) + realpart(y + z))"


def test_printmethod():
    x = symbols('x')

    class nint(Function):
        def _fcode(self, printer):
            return "nint(%s)" % printer._print(self.args[0])
    assert fcode(nint(x)) == "      nint(x)"


def test_fcode_sign():  #issue 12267
    x=symbols('x')
    y=symbols('y', integer=True)
    z=symbols('z', complex=True)
    assert fcode(sign(x), standard=95, source_format='free') == "merge(0d0, dsign(1d0, x), x == 0d0)"
    assert fcode(sign(y), standard=95, source_format='free') == "merge(0, isign(1, y), y == 0)"
    assert fcode(sign(z), standard=95, source_format='free') == "merge(cmplx(0d0, 0d0), z/abs(z), abs(z) == 0d0)"
    raises(NotImplementedError, lambda: fcode(sign(x)))


def test_fcode_Pow():
    x, y = symbols('x,y')
    n = symbols('n', integer=True)

    assert fcode(x**3) == "      x**3"
    assert fcode(x**(y**3)) == "      x**(y**3)"
    assert fcode(1/(sin(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "      (3.5d0*sin(x))**(-x + y**x)/(x**2 + y)"
    assert fcode(sqrt(x)) == '      sqrt(x)'
    assert fcode(sqrt(n)) == '      sqrt(dble(n))'
    assert fcode(x**0.5) == '      sqrt(x)'
    assert fcode(sqrt(x)) == '      sqrt(x)'
    assert fcode(sqrt(10)) == '      sqrt(10.0d0)'
    assert fcode(x**-1.0) == '      1d0/x'
    assert fcode(x**-2.0, 'y', source_format='free') == 'y = x**(-2.0d0)'  # 2823
    assert fcode(x**Rational(3, 7)) == '      x**(3.0d0/7.0d0)'


def test_fcode_Rational():
    x = symbols('x')
    assert fcode(Rational(3, 7)) == "      3.0d0/7.0d0"
    assert fcode(Rational(18, 9)) == "      2"
    assert fcode(Rational(3, -7)) == "      -3.0d0/7.0d0"
    assert fcode(Rational(-3, -7)) == "      3.0d0/7.0d0"
    assert fcode(x + Rational(3, 7)) == "      x + 3.0d0/7.0d0"
    assert fcode(Rational(3, 7)*x) == "      (3.0d0/7.0d0)*x"


def test_fcode_Integer():
    assert fcode(Integer(67)) == "      67"
    assert fcode(Integer(-1)) == "      -1"


def test_fcode_Float():
    assert fcode(Float(42.0)) == "      42.0000000000000d0"
    assert fcode(Float(-1e20)) == "      -1.00000000000000d+20"


def test_fcode_functions():
    x, y = symbols('x,y')
    assert fcode(sin(x) ** cos(y)) == "      sin(x)**cos(y)"
    raises(NotImplementedError, lambda: fcode(Mod(x, y), standard=66))
    raises(NotImplementedError, lambda: fcode(x % y, standard=66))
    raises(NotImplementedError, lambda: fcode(Mod(x, y), standard=77))
    raises(NotImplementedError, lambda: fcode(x % y, standard=77))
    for standard in [90, 95, 2003, 2008]:
        assert fcode(Mod(x, y), standard=standard) == "      modulo(x, y)"
        assert fcode(x % y, standard=standard) == "      modulo(x, y)"


def test_case():
    ob = FCodePrinter()
    x,x_,x__,y,X,X_,Y = symbols('x,x_,x__,y,X,X_,Y')
    assert fcode(exp(x_) + sin(x*y) + cos(X*Y)) == \
                        '      exp(x_) + sin(x*y) + cos(X__*Y_)'
    assert fcode(exp(x__) + 2*x*Y*X_**Rational(7, 2)) == \
                        '      2*X_**(7.0d0/2.0d0)*Y*x + exp(x__)'
    assert fcode(exp(x_) + sin(x*y) + cos(X*Y), name_mangling=False) == \
                        '      exp(x_) + sin(x*y) + cos(X*Y)'
    assert fcode(x - cos(X), name_mangling=False) == '      x - cos(X)'
    assert ob.doprint(X*sin(x) + x_, assign_to='me') == '      me = X*sin(x_) + x__'
    assert ob.doprint(X*sin(x), assign_to='mu') == '      mu = X*sin(x_)'
    assert ob.doprint(x_, assign_to='ad') == '      ad = x__'
    n, m = symbols('n,m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    I = Idx('I', n)
    assert fcode(A[i, I]*x[I], assign_to=y[i], source_format='free') == (
                                            "do i = 1, m\n"
                                            "   y(i) = 0\n"
                                            "end do\n"
                                            "do i = 1, m\n"
                                            "   do I_ = 1, n\n"
                                            "      y(i) = A(i, I_)*x(I_) + y(i)\n"
                                            "   end do\n"
                                            "end do" )


#issue 6814
def test_fcode_functions_with_integers():
    x= symbols('x')
    log10_17 = log(10).evalf(17)
    loglog10_17 = '0.8340324452479558d0'
    assert fcode(x * log(10)) == "      x*%sd0" % log10_17
    assert fcode(x * log(10)) == "      x*%sd0" % log10_17
    assert fcode(x * log(S(10))) == "      x*%sd0" % log10_17
    assert fcode(log(S(10))) == "      %sd0" % log10_17
    assert fcode(exp(10)) == "      %sd0" % exp(10).evalf(17)
    assert fcode(x * log(log(10))) == "      x*%s" % loglog10_17
    assert fcode(x * log(log(S(10)))) == "      x*%s" % loglog10_17


def test_fcode_NumberSymbol():
    prec = 17
    p = FCodePrinter()
    assert fcode(Catalan) == '      parameter (Catalan = %sd0)\n      Catalan' % Catalan.evalf(prec)
    assert fcode(EulerGamma) == '      parameter (EulerGamma = %sd0)\n      EulerGamma' % EulerGamma.evalf(prec)
    assert fcode(E) == '      parameter (E = %sd0)\n      E' % E.evalf(prec)
    assert fcode(GoldenRatio) == '      parameter (GoldenRatio = %sd0)\n      GoldenRatio' % GoldenRatio.evalf(prec)
    assert fcode(pi) == '      parameter (pi = %sd0)\n      pi' % pi.evalf(prec)
    assert fcode(
        pi, precision=5) == '      parameter (pi = %sd0)\n      pi' % pi.evalf(5)
    assert fcode(Catalan, human=False) == ({
        (Catalan, p._print(Catalan.evalf(prec)))}, set(), '      Catalan')
    assert fcode(EulerGamma, human=False) == ({(EulerGamma, p._print(
        EulerGamma.evalf(prec)))}, set(), '      EulerGamma')
    assert fcode(E, human=False) == (
        {(E, p._print(E.evalf(prec)))}, set(), '      E')
    assert fcode(GoldenRatio, human=False) == ({(GoldenRatio, p._print(
        GoldenRatio.evalf(prec)))}, set(), '      GoldenRatio')
    assert fcode(pi, human=False) == (
        {(pi, p._print(pi.evalf(prec)))}, set(), '      pi')
    assert fcode(pi, precision=5, human=False) == (
        {(pi, p._print(pi.evalf(5)))}, set(), '      pi')


def test_fcode_complex():
    assert fcode(I) == "      cmplx(0,1)"
    x = symbols('x')
    assert fcode(4*I) == "      cmplx(0,4)"
    assert fcode(3 + 4*I) == "      cmplx(3,4)"
    assert fcode(3 + 4*I + x) == "      cmplx(3,4) + x"
    assert fcode(I*x) == "      cmplx(0,1)*x"
    assert fcode(3 + 4*I - x) == "      cmplx(3,4) - x"
    x = symbols('x', imaginary=True)
    assert fcode(5*x) == "      5*x"
    assert fcode(I*x) == "      cmplx(0,1)*x"
    assert fcode(3 + x) == "      x + 3"


def test_implicit():
    x, y = symbols('x,y')
    assert fcode(sin(x)) == "      sin(x)"
    assert fcode(atan2(x, y)) == "      atan2(x, y)"
    assert fcode(conjugate(x)) == "      conjg(x)"


def test_not_fortran():
    x = symbols('x')
    g = Function('g')
    with raises(NotImplementedError):
        fcode(gamma(x))
    assert fcode(Integral(sin(x)), strict=False) == "C     Not supported in Fortran:\nC     Integral\n      Integral(sin(x), x)"
    with raises(NotImplementedError):
        fcode(g(x))


def test_user_functions():
    x = symbols('x')
    assert fcode(sin(x), user_functions={"sin": "zsin"}) == "      zsin(x)"
    x = symbols('x')
    assert fcode(
        gamma(x), user_functions={"gamma": "mygamma"}) == "      mygamma(x)"
    g = Function('g')
    assert fcode(g(x), user_functions={"g": "great"}) == "      great(x)"
    n = symbols('n', integer=True)
    assert fcode(
        factorial(n), user_functions={"factorial": "fct"}) == "      fct(n)"


def test_inline_function():
    x = symbols('x')
    g = implemented_function('g', Lambda(x, 2*x))
    assert fcode(g(x)) == "      2*x"
    g = implemented_function('g', Lambda(x, 2*pi/x))
    assert fcode(g(x)) == (
        "      parameter (pi = %sd0)\n"
        "      2*pi/x"
    ) % pi.evalf(17)
    A = IndexedBase('A')
    i = Idx('i', symbols('n', integer=True))
    g = implemented_function('g', Lambda(x, x*(1 + x)*(2 + x)))
    assert fcode(g(A[i]), assign_to=A[i]) == (
        "      do i = 1, n\n"
        "         A(i) = (A(i) + 1)*(A(i) + 2)*A(i)\n"
        "      end do"
    )


def test_assign_to():
    x = symbols('x')
    assert fcode(sin(x), assign_to="s") == "      s = sin(x)"


def test_line_wrapping():
    x, y = symbols('x,y')
    assert fcode(((x + y)**10).expand(), assign_to="var") == (
        "      var = x**10 + 10*x**9*y + 45*x**8*y**2 + 120*x**7*y**3 + 210*x**6*\n"
        "     @ y**4 + 252*x**5*y**5 + 210*x**4*y**6 + 120*x**3*y**7 + 45*x**2*y\n"
        "     @ **8 + 10*x*y**9 + y**10"
    )
    e = [x**i for i in range(11)]
    assert fcode(Add(*e)) == (
        "      x**10 + x**9 + x**8 + x**7 + x**6 + x**5 + x**4 + x**3 + x**2 + x\n"
        "     @ + 1"
    )


def test_fcode_precedence():
    x, y = symbols("x y")
    assert fcode(And(x < y, y < x + 1), source_format="free") == \
        "x < y .and. y < x + 1"
    assert fcode(Or(x < y, y < x + 1), source_format="free") == \
        "x < y .or. y < x + 1"
    assert fcode(Xor(x < y, y < x + 1, evaluate=False),
        source_format="free") == "x < y .neqv. y < x + 1"
    assert fcode(Equivalent(x < y, y < x + 1), source_format="free") == \
        "x < y .eqv. y < x + 1"


def test_fcode_Logical():
    x, y, z = symbols("x y z")
    # unary Not
    assert fcode(Not(x), source_format="free") == ".not. x"
    # binary And
    assert fcode(And(x, y), source_format="free") == "x .and. y"
    assert fcode(And(x, Not(y)), source_format="free") == "x .and. .not. y"
    assert fcode(And(Not(x), y), source_format="free") == "y .and. .not. x"
    assert fcode(And(Not(x), Not(y)), source_format="free") == \
        ".not. x .and. .not. y"
    assert fcode(Not(And(x, y), evaluate=False), source_format="free") == \
        ".not. (x .and. y)"
    # binary Or
    assert fcode(Or(x, y), source_format="free") == "x .or. y"
    assert fcode(Or(x, Not(y)), source_format="free") == "x .or. .not. y"
    assert fcode(Or(Not(x), y), source_format="free") == "y .or. .not. x"
    assert fcode(Or(Not(x), Not(y)), source_format="free") == \
        ".not. x .or. .not. y"
    assert fcode(Not(Or(x, y), evaluate=False), source_format="free") == \
        ".not. (x .or. y)"
    # mixed And/Or
    assert fcode(And(Or(y, z), x), source_format="free") == "x .and. (y .or. z)"
    assert fcode(And(Or(z, x), y), source_format="free") == "y .and. (x .or. z)"
    assert fcode(And(Or(x, y), z), source_format="free") == "z .and. (x .or. y)"
    assert fcode(Or(And(y, z), x), source_format="free") == "x .or. y .and. z"
    assert fcode(Or(And(z, x), y), source_format="free") == "y .or. x .and. z"
    assert fcode(Or(And(x, y), z), source_format="free") == "z .or. x .and. y"
    # trinary And
    assert fcode(And(x, y, z), source_format="free") == "x .and. y .and. z"
    assert fcode(And(x, y, Not(z)), source_format="free") == \
        "x .and. y .and. .not. z"
    assert fcode(And(x, Not(y), z), source_format="free") == \
        "x .and. z .and. .not. y"
    assert fcode(And(Not(x), y, z), source_format="free") == \
        "y .and. z .and. .not. x"
    assert fcode(Not(And(x, y, z), evaluate=False), source_format="free") == \
        ".not. (x .and. y .and. z)"
    # trinary Or
    assert fcode(Or(x, y, z), source_format="free") == "x .or. y .or. z"
    assert fcode(Or(x, y, Not(z)), source_format="free") == \
        "x .or. y .or. .not. z"
    assert fcode(Or(x, Not(y), z), source_format="free") == \
        "x .or. z .or. .not. y"
    assert fcode(Or(Not(x), y, z), source_format="free") == \
        "y .or. z .or. .not. x"
    assert fcode(Not(Or(x, y, z), evaluate=False), source_format="free") == \
        ".not. (x .or. y .or. z)"


def test_fcode_Xlogical():
    x, y, z = symbols("x y z")
    # binary Xor
    assert fcode(Xor(x, y, evaluate=False), source_format="free") == \
        "x .neqv. y"
    assert fcode(Xor(x, Not(y), evaluate=False), source_format="free") == \
        "x .neqv. .not. y"
    assert fcode(Xor(Not(x), y, evaluate=False), source_format="free") == \
        "y .neqv. .not. x"
    assert fcode(Xor(Not(x), Not(y), evaluate=False),
        source_format="free") == ".not. x .neqv. .not. y"
    assert fcode(Not(Xor(x, y, evaluate=False), evaluate=False),
        source_format="free") == ".not. (x .neqv. y)"
    # binary Equivalent
    assert fcode(Equivalent(x, y), source_format="free") == "x .eqv. y"
    assert fcode(Equivalent(x, Not(y)), source_format="free") == \
        "x .eqv. .not. y"
    assert fcode(Equivalent(Not(x), y), source_format="free") == \
        "y .eqv. .not. x"
    assert fcode(Equivalent(Not(x), Not(y)), source_format="free") == \
        ".not. x .eqv. .not. y"
    assert fcode(Not(Equivalent(x, y), evaluate=False),
        source_format="free") == ".not. (x .eqv. y)"
    # mixed And/Equivalent
    assert fcode(Equivalent(And(y, z), x), source_format="free") == \
        "x .eqv. y .and. z"
    assert fcode(Equivalent(And(z, x), y), source_format="free") == \
        "y .eqv. x .and. z"
    assert fcode(Equivalent(And(x, y), z), source_format="free") == \
        "z .eqv. x .and. y"
    assert fcode(And(Equivalent(y, z), x), source_format="free") == \
        "x .and. (y .eqv. z)"
    assert fcode(And(Equivalent(z, x), y), source_format="free") == \
        "y .and. (x .eqv. z)"
    assert fcode(And(Equivalent(x, y), z), source_format="free") == \
        "z .and. (x .eqv. y)"
    # mixed Or/Equivalent
    assert fcode(Equivalent(Or(y, z), x), source_format="free") == \
        "x .eqv. y .or. z"
    assert fcode(Equivalent(Or(z, x), y), source_format="free") == \
        "y .eqv. x .or. z"
    assert fcode(Equivalent(Or(x, y), z), source_format="free") == \
        "z .eqv. x .or. y"
    assert fcode(Or(Equivalent(y, z), x), source_format="free") == \
        "x .or. (y .eqv. z)"
    assert fcode(Or(Equivalent(z, x), y), source_format="free") == \
        "y .or. (x .eqv. z)"
    assert fcode(Or(Equivalent(x, y), z), source_format="free") == \
        "z .or. (x .eqv. y)"
    # mixed Xor/Equivalent
    assert fcode(Equivalent(Xor(y, z, evaluate=False), x),
        source_format="free") == "x .eqv. (y .neqv. z)"
    assert fcode(Equivalent(Xor(z, x, evaluate=False), y),
        source_format="free") == "y .eqv. (x .neqv. z)"
    assert fcode(Equivalent(Xor(x, y, evaluate=False), z),
        source_format="free") == "z .eqv. (x .neqv. y)"
    assert fcode(Xor(Equivalent(y, z), x, evaluate=False),
        source_format="free") == "x .neqv. (y .eqv. z)"
    assert fcode(Xor(Equivalent(z, x), y, evaluate=False),
        source_format="free") == "y .neqv. (x .eqv. z)"
    assert fcode(Xor(Equivalent(x, y), z, evaluate=False),
        source_format="free") == "z .neqv. (x .eqv. y)"
    # mixed And/Xor
    assert fcode(Xor(And(y, z), x, evaluate=False), source_format="free") == \
        "x .neqv. y .and. z"
    assert fcode(Xor(And(z, x), y, evaluate=False), source_format="free") == \
        "y .neqv. x .and. z"
    assert fcode(Xor(And(x, y), z, evaluate=False), source_format="free") == \
        "z .neqv. x .and. y"
    assert fcode(And(Xor(y, z, evaluate=False), x), source_format="free") == \
        "x .and. (y .neqv. z)"
    assert fcode(And(Xor(z, x, evaluate=False), y), source_format="free") == \
        "y .and. (x .neqv. z)"
    assert fcode(And(Xor(x, y, evaluate=False), z), source_format="free") == \
        "z .and. (x .neqv. y)"
    # mixed Or/Xor
    assert fcode(Xor(Or(y, z), x, evaluate=False), source_format="free") == \
        "x .neqv. y .or. z"
    assert fcode(Xor(Or(z, x), y, evaluate=False), source_format="free") == \
        "y .neqv. x .or. z"
    assert fcode(Xor(Or(x, y), z, evaluate=False), source_format="free") == \
        "z .neqv. x .or. y"
    assert fcode(Or(Xor(y, z, evaluate=False), x), source_format="free") == \
        "x .or. (y .neqv. z)"
    assert fcode(Or(Xor(z, x, evaluate=False), y), source_format="free") == \
        "y .or. (x .neqv. z)"
    assert fcode(Or(Xor(x, y, evaluate=False), z), source_format="free") == \
        "z .or. (x .neqv. y)"
    # trinary Xor
    assert fcode(Xor(x, y, z, evaluate=False), source_format="free") == \
        "x .neqv. y .neqv. z"
    assert fcode(Xor(x, y, Not(z), evaluate=False), source_format="free") == \
        "x .neqv. y .neqv. .not. z"
    assert fcode(Xor(x, Not(y), z, evaluate=False), source_format="free") == \
        "x .neqv. z .neqv. .not. y"
    assert fcode(Xor(Not(x), y, z, evaluate=False), source_format="free") == \
        "y .neqv. z .neqv. .not. x"


def test_fcode_Relational():
    x, y = symbols("x y")
    assert fcode(Relational(x, y, "=="), source_format="free") == "x == y"
    assert fcode(Relational(x, y, "!="), source_format="free") == "x /= y"
    assert fcode(Relational(x, y, ">="), source_format="free") == "x >= y"
    assert fcode(Relational(x, y, "<="), source_format="free") == "x <= y"
    assert fcode(Relational(x, y, ">"), source_format="free") == "x > y"
    assert fcode(Relational(x, y, "<"), source_format="free") == "x < y"


def test_fcode_Piecewise():
    x = symbols('x')
    expr = Piecewise((x, x < 1), (x**2, True))
    # Check that inline conditional (merge) fails if standard isn't 95+
    raises(NotImplementedError, lambda: fcode(expr))
    code = fcode(expr, standard=95)
    expected = "      merge(x, x**2, x < 1)"
    assert code == expected
    assert fcode(Piecewise((x, x < 1), (x**2, True)), assign_to="var") == (
        "      if (x < 1) then\n"
        "         var = x\n"
        "      else\n"
        "         var = x**2\n"
        "      end if"
    )
    a = cos(x)/x
    b = sin(x)/x
    for i in range(10):
        a = diff(a, x)
        b = diff(b, x)
    expected = (
        "      if (x < 0) then\n"
        "         weird_name = -cos(x)/x + 10*sin(x)/x**2 + 90*cos(x)/x**3 - 720*\n"
        "     @ sin(x)/x**4 - 5040*cos(x)/x**5 + 30240*sin(x)/x**6 + 151200*cos(x\n"
        "     @ )/x**7 - 604800*sin(x)/x**8 - 1814400*cos(x)/x**9 + 3628800*sin(x\n"
        "     @ )/x**10 + 3628800*cos(x)/x**11\n"
        "      else\n"
        "         weird_name = -sin(x)/x - 10*cos(x)/x**2 + 90*sin(x)/x**3 + 720*\n"
        "     @ cos(x)/x**4 - 5040*sin(x)/x**5 - 30240*cos(x)/x**6 + 151200*sin(x\n"
        "     @ )/x**7 + 604800*cos(x)/x**8 - 1814400*sin(x)/x**9 - 3628800*cos(x\n"
        "     @ )/x**10 + 3628800*sin(x)/x**11\n"
        "      end if"
    )
    code = fcode(Piecewise((a, x < 0), (b, True)), assign_to="weird_name")
    assert code == expected
    code = fcode(Piecewise((x, x < 1), (x**2, x > 1), (sin(x), True)), standard=95)
    expected = "      merge(x, merge(x**2, sin(x), x > 1), x < 1)"
    assert code == expected
    # Check that Piecewise without a True (default) condition error
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: fcode(expr))


def test_wrap_fortran():
    #   "########################################################################"
    printer = FCodePrinter()
    lines = [
        "C     This is a long comment on a single line that must be wrapped properly to produce nice output",
        "      this = is + a + long + and + nasty + fortran + statement + that * must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +  that * must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that * must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement + that*must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that*must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +    that*must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +     that*must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement + that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +  that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +    that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +     that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement(that)/must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran +     statement(that)/must + be + wrapped + properly",
    ]
    wrapped_lines = printer._wrap_fortran(lines)
    expected_lines = [
        "C     This is a long comment on a single line that must be wrapped",
        "C     properly to produce nice output",
        "      this = is + a + long + and + nasty + fortran + statement + that *",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +  that *",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that",
        "     @ * must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement + that*",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that*",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +    that",
        "     @ *must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +",
        "     @ that*must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement + that**",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +  that**",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that",
        "     @ **must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +    that",
        "     @ **must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +",
        "     @ that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement(that)/",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran +     statement(that)",
        "     @ /must + be + wrapped + properly",
    ]
    for line in wrapped_lines:
        assert len(line) <= 72
    for w, e in zip(wrapped_lines, expected_lines):
        assert w == e
    assert len(wrapped_lines) == len(expected_lines)


def test_wrap_fortran_keep_d0():
    printer = FCodePrinter()
    lines = [
        '      this_variable_is_very_long_because_we_try_to_test_line_break=1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break =1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break  = 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break   = 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break    = 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break = 10.0d0'
    ]
    expected = [
        '      this_variable_is_very_long_because_we_try_to_test_line_break=1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break =',
        '     @ 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break  =',
        '     @ 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break   =',
        '     @ 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break    =',
        '     @ 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break =',
        '     @ 10.0d0'
    ]
    assert printer._wrap_fortran(lines) == expected


def test_settings():
    raises(TypeError, lambda: fcode(S(4), method="garbage"))


def test_free_form_code_line():
    x, y = symbols('x,y')
    assert fcode(cos(x) + sin(y), source_format='free') == "sin(y) + cos(x)"


def test_free_form_continuation_line():
    x, y = symbols('x,y')
    result = fcode(((cos(x) + sin(y))**(7)).expand(), source_format='free')
    expected = (
        'sin(y)**7 + 7*sin(y)**6*cos(x) + 21*sin(y)**5*cos(x)**2 + 35*sin(y)**4* &\n'
        '      cos(x)**3 + 35*sin(y)**3*cos(x)**4 + 21*sin(y)**2*cos(x)**5 + 7* &\n'
        '      sin(y)*cos(x)**6 + cos(x)**7'
    )
    assert result == expected


def test_free_form_comment_line():
    printer = FCodePrinter({'source_format': 'free'})
    lines = [ "! This is a long comment on a single line that must be wrapped properly to produce nice output"]
    expected = [
        '! This is a long comment on a single line that must be wrapped properly',
        '! to produce nice output']
    assert printer._wrap_fortran(lines) == expected


def test_loops():
    n, m = symbols('n,m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)

    expected = (
        'do i = 1, m\n'
        '   y(i) = 0\n'
        'end do\n'
        'do i = 1, m\n'
        '   do j = 1, n\n'
        '      y(i) = %(rhs)s\n'
        '   end do\n'
        'end do'
    )

    code = fcode(A[i, j]*x[j], assign_to=y[i], source_format='free')
    assert (code == expected % {'rhs': 'y(i) + A(i, j)*x(j)'} or
            code == expected % {'rhs': 'y(i) + x(j)*A(i, j)'} or
            code == expected % {'rhs': 'x(j)*A(i, j) + y(i)'} or
            code == expected % {'rhs': 'A(i, j)*x(j) + y(i)'})


def test_dummy_loops():
    i, m = symbols('i m', integer=True, cls=Dummy)
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx(i, m)

    expected = (
        'do i_%(icount)i = 1, m_%(mcount)i\n'
        '   y(i_%(icount)i) = x(i_%(icount)i)\n'
        'end do'
    ) % {'icount': i.label.dummy_index, 'mcount': m.dummy_index}
    code = fcode(x[i], assign_to=y[i], source_format='free')
    assert code == expected


def test_fcode_Indexed_without_looking_for_contraction():
    len_y = 5
    y = IndexedBase('y', shape=(len_y,))
    x = IndexedBase('x', shape=(len_y,))
    Dy = IndexedBase('Dy', shape=(len_y-1,))
    i = Idx('i', len_y-1)
    e=Eq(Dy[i], (y[i+1]-y[i])/(x[i+1]-x[i]))
    code0 = fcode(e.rhs, assign_to=e.lhs, contract=False)
    assert code0.endswith('Dy(i) = (y(i + 1) - y(i))/(x(i + 1) - x(i))')


def test_element_like_objects():
    len_y = 5
    y = ArraySymbol('y', shape=(len_y,))
    x = ArraySymbol('x', shape=(len_y,))
    Dy = ArraySymbol('Dy', shape=(len_y-1,))
    i = Idx('i', len_y-1)
    e=Eq(Dy[i], (y[i+1]-y[i])/(x[i+1]-x[i]))
    code0 = fcode(Assignment(e.lhs, e.rhs))
    assert code0.endswith('Dy(i) = (y(i + 1) - y(i))/(x(i + 1) - x(i))')

    class ElementExpr(Element, Expr):
        pass

    e = e.subs((a, ElementExpr(a.name, a.indices)) for a in e.atoms(ArrayElement)  )
    e=Eq(Dy[i], (y[i+1]-y[i])/(x[i+1]-x[i]))
    code0 = fcode(Assignment(e.lhs, e.rhs))
    assert code0.endswith('Dy(i) = (y(i + 1) - y(i))/(x(i + 1) - x(i))')


def test_derived_classes():
    class MyFancyFCodePrinter(FCodePrinter):
        _default_settings = FCodePrinter._default_settings.copy()

    printer = MyFancyFCodePrinter()
    x = symbols('x')
    assert printer.doprint(sin(x), "bork") == "      bork = sin(x)"


def test_indent():
    codelines = (
        'subroutine test(a)\n'
        'integer :: a, i, j\n'
        '\n'
        'do\n'
        'do \n'
        'do j = 1, 5\n'
        'if (a>b) then\n'
        'if(b>0) then\n'
        'a = 3\n'
        'donot_indent_me = 2\n'
        'do_not_indent_me_either = 2\n'
        'ifIam_indented_something_went_wrong = 2\n'
        'if_I_am_indented_something_went_wrong = 2\n'
        'end should not be unindented here\n'
        'end if\n'
        'endif\n'
        'end do\n'
        'end do\n'
        'enddo\n'
        'end subroutine\n'
        '\n'
        'subroutine test2(a)\n'
        'integer :: a\n'
        'do\n'
        'a = a + 1\n'
        'end do \n'
        'end subroutine\n'
    )
    expected = (
        'subroutine test(a)\n'
        'integer :: a, i, j\n'
        '\n'
        'do\n'
        '   do \n'
        '      do j = 1, 5\n'
        '         if (a>b) then\n'
        '            if(b>0) then\n'
        '               a = 3\n'
        '               donot_indent_me = 2\n'
        '               do_not_indent_me_either = 2\n'
        '               ifIam_indented_something_went_wrong = 2\n'
        '               if_I_am_indented_something_went_wrong = 2\n'
        '               end should not be unindented here\n'
        '            end if\n'
        '         endif\n'
        '      end do\n'
        '   end do\n'
        'enddo\n'
        'end subroutine\n'
        '\n'
        'subroutine test2(a)\n'
        'integer :: a\n'
        'do\n'
        '   a = a + 1\n'
        'end do \n'
        'end subroutine\n'
    )
    p = FCodePrinter({'source_format': 'free'})
    result = p.indent_code(codelines)
    assert result == expected

def test_Matrix_printing():
    x, y, z = symbols('x,y,z')
    # Test returning a Matrix
    mat = Matrix([x*y, Piecewise((2 + x, y>0), (y, True)), sin(z)])
    A = MatrixSymbol('A', 3, 1)
    assert fcode(mat, A) == (
        "      A(1, 1) = x*y\n"
        "      if (y > 0) then\n"
        "         A(2, 1) = x + 2\n"
        "      else\n"
        "         A(2, 1) = y\n"
        "      end if\n"
        "      A(3, 1) = sin(z)")
    # Test using MatrixElements in expressions
    expr = Piecewise((2*A[2, 0], x > 0), (A[2, 0], True)) + sin(A[1, 0]) + A[0, 0]
    assert fcode(expr, standard=95) == (
        "      merge(2*A(3, 1), A(3, 1), x > 0) + sin(A(2, 1)) + A(1, 1)")
    # Test using MatrixElements in a Matrix
    q = MatrixSymbol('q', 5, 1)
    M = MatrixSymbol('M', 3, 3)
    m = Matrix([[sin(q[1,0]), 0, cos(q[2,0])],
        [q[1,0] + q[2,0], q[3, 0], 5],
        [2*q[4, 0]/q[1,0], sqrt(q[0,0]) + 4, 0]])
    assert fcode(m, M) == (
        "      M(1, 1) = sin(q(2, 1))\n"
        "      M(2, 1) = q(2, 1) + q(3, 1)\n"
        "      M(3, 1) = 2*q(5, 1)/q(2, 1)\n"
        "      M(1, 2) = 0\n"
        "      M(2, 2) = q(4, 1)\n"
        "      M(3, 2) = sqrt(q(1, 1)) + 4\n"
        "      M(1, 3) = cos(q(3, 1))\n"
        "      M(2, 3) = 5\n"
        "      M(3, 3) = 0")


def test_fcode_For():
    x, y = symbols('x y')

    f = For(x, Range(0, 10, 2), [Assignment(y, x * y)])
    sol = fcode(f)
    assert sol == ("      do x = 0, 9, 2\n"
                   "         y = x*y\n"
                   "      end do")


def test_fcode_Declaration():
    def check(expr, ref, **kwargs):
        assert fcode(expr, standard=95, source_format='free', **kwargs) == ref

    i = symbols('i', integer=True)
    var1 = Variable.deduced(i)
    dcl1 = Declaration(var1)
    check(dcl1, "integer*4 :: i")


    x, y = symbols('x y')
    var2 = Variable(x, float32, value=42, attrs={value_const})
    dcl2b = Declaration(var2)
    check(dcl2b, 'real*4, parameter :: x = 42')

    var3 = Variable(y, type=bool_)
    dcl3 = Declaration(var3)
    check(dcl3, 'logical :: y')

    check(float32, "real*4")
    check(float64, "real*8")
    check(real, "real*4", type_aliases={real: float32})
    check(real, "real*8", type_aliases={real: float64})


def test_MatrixElement_printing():
    # test cases for issue #11821
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    assert(fcode(A[0, 0]) == "      A(1, 1)")
    assert(fcode(3 * A[0, 0]) == "      3*A(1, 1)")

    F = C[0, 0].subs(C, A - B)
    assert(fcode(F) == "      (A - B)(1, 1)")


def test_aug_assign():
    x = symbols('x')
    assert fcode(aug_assign(x, '+', 1), source_format='free') == 'x = x + 1'


def test_While():
    x = symbols('x')
    assert fcode(While(abs(x) > 1, [aug_assign(x, '-', 1)]), source_format='free') == (
        'do while (abs(x) > 1)\n'
        '   x = x - 1\n'
        'end do'
    )


def test_FunctionPrototype_print():
    x = symbols('x')
    n = symbols('n', integer=True)
    vx = Variable(x, type=real)
    vn = Variable(n, type=integer)
    fp1 = FunctionPrototype(real, 'power', [vx, vn])
    # Should be changed to proper test once multi-line generation is working
    # see https://github.com/sympy/sympy/issues/15824
    raises(NotImplementedError, lambda: fcode(fp1))


def test_FunctionDefinition_print():
    x = symbols('x')
    n = symbols('n', integer=True)
    vx = Variable(x, type=real)
    vn = Variable(n, type=integer)
    body = [Assignment(x, x**n), Return(x)]
    fd1 = FunctionDefinition(real, 'power', [vx, vn], body)
    # Should be changed to proper test once multi-line generation is working
    # see https://github.com/sympy/sympy/issues/15824
    raises(NotImplementedError, lambda: fcode(fd1))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\holonomic\tests\test_holonomic.py ===
from sympy.holonomic import (DifferentialOperator, HolonomicFunction,
                             DifferentialOperators, from_hyper,
                             from_meijerg, expr_to_holonomic)
from sympy.holonomic.recurrence import RecurrenceOperators, HolonomicSequence
from sympy.core import EulerGamma
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, cosh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.bessel import besselj
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.error_functions import (Ci, Si, erf, erfc)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.printing.str import sstr
from sympy.series.order import O
from sympy.simplify.hyperexpand import hyperexpand
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.realfield import RR


def test_DifferentialOperator():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    assert Dx == R.derivative_operator
    assert Dx == DifferentialOperator([R.base.zero, R.base.one], R)
    assert x * Dx + x**2 * Dx**2 == DifferentialOperator([0, x, x**2], R)
    assert (x**2 + 1) + Dx + x * \
        Dx**5 == DifferentialOperator([x**2 + 1, 1, 0, 0, 0, x], R)
    assert (x * Dx + x**2 + 1 - Dx * (x**3 + x))**3 == (-48 * x**6) + \
        (-57 * x**7) * Dx + (-15 * x**8) * Dx**2 + (-x**9) * Dx**3
    p = (x * Dx**2 + (x**2 + 3) * Dx**5) * (Dx + x**2)
    q = (2 * x) + (4 * x**2) * Dx + (x**3) * Dx**2 + \
        (20 * x**2 + x + 60) * Dx**3 + (10 * x**3 + 30 * x) * Dx**4 + \
        (x**4 + 3 * x**2) * Dx**5 + (x**2 + 3) * Dx**6
    assert p == q


def test_HolonomicFunction_addition():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx**2 * x, x)
    q = HolonomicFunction((2) * Dx + (x) * Dx**2, x)
    assert p == q
    p = HolonomicFunction(x * Dx + 1, x)
    q = HolonomicFunction(Dx + 1, x)
    r = HolonomicFunction((x - 2) + (x**2 - 2) * Dx + (x**2 - x) * Dx**2, x)
    assert p + q == r
    p = HolonomicFunction(x * Dx + Dx**2 * (x**2 + 2), x)
    q = HolonomicFunction(Dx - 3, x)
    r = HolonomicFunction((-54 * x**2 - 126 * x - 150) + (-135 * x**3 - 252 * x**2 - 270 * x + 140) * Dx +\
                 (-27 * x**4 - 24 * x**2 + 14 * x - 150) * Dx**2 + \
                 (9 * x**4 + 15 * x**3 + 38 * x**2 + 30 * x +40) * Dx**3, x)
    assert p + q == r
    p = HolonomicFunction(Dx**5 - 1, x)
    q = HolonomicFunction(x**3 + Dx, x)
    r = HolonomicFunction((-x**18 + 45*x**14 - 525*x**10 + 1575*x**6 - x**3 - 630*x**2) + \
        (-x**15 + 30*x**11 - 195*x**7 + 210*x**3 - 1)*Dx + (x**18 - 45*x**14 + 525*x**10 - \
        1575*x**6 + x**3 + 630*x**2)*Dx**5 + (x**15 - 30*x**11 + 195*x**7 - 210*x**3 + \
        1)*Dx**6, x)
    assert p+q == r

    p = x**2 + 3*x + 8
    q = x**3 - 7*x + 5
    p = p*Dx - p.diff()
    q = q*Dx - q.diff()
    r = HolonomicFunction(p, x) + HolonomicFunction(q, x)
    s = HolonomicFunction((6*x**2 + 18*x + 14) + (-4*x**3 - 18*x**2 - 62*x + 10)*Dx +\
        (x**4 + 6*x**3 + 31*x**2 - 10*x - 71)*Dx**2, x)
    assert r == s


def test_HolonomicFunction_multiplication():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx+x+x*Dx**2, x)
    q = HolonomicFunction(x*Dx+Dx*x+Dx**2, x)
    r = HolonomicFunction((8*x**6 + 4*x**4 + 6*x**2 + 3) + (24*x**5 - 4*x**3 + 24*x)*Dx + \
        (8*x**6 + 20*x**4 + 12*x**2 + 2)*Dx**2 + (8*x**5 + 4*x**3 + 4*x)*Dx**3 + \
        (2*x**4 + x**2)*Dx**4, x)
    assert p*q == r
    p = HolonomicFunction(Dx**2+1, x)
    q = HolonomicFunction(Dx-1, x)
    r = HolonomicFunction((2) + (-2)*Dx + (1)*Dx**2, x)
    assert p*q == r
    p = HolonomicFunction(Dx**2+1+x+Dx, x)
    q = HolonomicFunction((Dx*x-1)**2, x)
    r = HolonomicFunction((4*x**7 + 11*x**6 + 16*x**5 + 4*x**4 - 6*x**3 - 7*x**2 - 8*x - 2) + \
        (8*x**6 + 26*x**5 + 24*x**4 - 3*x**3 - 11*x**2 - 6*x - 2)*Dx + \
        (8*x**6 + 18*x**5 + 15*x**4 - 3*x**3 - 6*x**2 - 6*x - 2)*Dx**2 + (8*x**5 + \
            10*x**4 + 6*x**3 - 2*x**2 - 4*x)*Dx**3 + (4*x**5 + 3*x**4 - x**2)*Dx**4, x)
    assert p*q == r
    p = HolonomicFunction(x*Dx**2-1, x)
    q = HolonomicFunction(Dx*x-x, x)
    r = HolonomicFunction((x - 3) + (-2*x + 2)*Dx + (x)*Dx**2, x)
    assert p*q == r


def test_HolonomicFunction_power():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx+x+x*Dx**2, x)
    a = HolonomicFunction(Dx, x)
    for n in range(10):
        assert a == p**n
        a *= p


def test_addition_initial_condition():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx-1, x, 0, [3])
    q = HolonomicFunction(Dx**2+1, x, 0, [1, 0])
    r = HolonomicFunction(-1 + Dx - Dx**2 + Dx**3, x, 0, [4, 3, 2])
    assert p + q == r
    p = HolonomicFunction(Dx - x + Dx**2, x, 0, [1, 2])
    q = HolonomicFunction(Dx**2 + x, x, 0, [1, 0])
    r = HolonomicFunction((-x**4 - x**3/4 - x**2 + Rational(1, 4)) + (x**3 + x**2/4 + x*Rational(3, 4) + 1)*Dx + \
        (x*Rational(-3, 2) + Rational(7, 4))*Dx**2 + (x**2 - x*Rational(7, 4) + Rational(1, 4))*Dx**3 + (x**2 + x/4 + S.Half)*Dx**4, x, 0, [2, 2, -2, 2])
    assert p + q == r
    p = HolonomicFunction(Dx**2 + 4*x*Dx + x**2, x, 0, [3, 4])
    q = HolonomicFunction(Dx**2 + 1, x, 0, [1, 1])
    r = HolonomicFunction((x**6 + 2*x**4 - 5*x**2 - 6) + (4*x**5 + 36*x**3 - 32*x)*Dx + \
         (x**6 + 3*x**4 + 5*x**2 - 9)*Dx**2 + (4*x**5 + 36*x**3 - 32*x)*Dx**3 + (x**4 + \
            10*x**2 - 3)*Dx**4, x, 0, [4, 5, -1, -17])
    assert p + q == r
    q = HolonomicFunction(Dx**3 + x, x, 2, [3, 0, 1])
    p = HolonomicFunction(Dx - 1, x, 2, [1])
    r = HolonomicFunction((-x**2 - x + 1) + (x**2 + x)*Dx + (-x - 2)*Dx**3 + \
        (x + 1)*Dx**4, x, 2, [4, 1, 2, -5 ])
    assert p + q == r
    p = expr_to_holonomic(sin(x))
    q = expr_to_holonomic(1/x, x0=1)
    r = HolonomicFunction((x**2 + 6) + (x**3 + 2*x)*Dx + (x**2 + 6)*Dx**2 + (x**3 + 2*x)*Dx**3, \
        x, 1, [sin(1) + 1, -1 + cos(1), -sin(1) + 2])
    assert p + q == r
    C_1 = symbols('C_1')
    p = expr_to_holonomic(sqrt(x))
    q = expr_to_holonomic(sqrt(x**2-x))
    r = (p + q).to_expr().subs(C_1, -I/2).expand()
    assert r == I*sqrt(x)*sqrt(-x + 1) + sqrt(x)


def test_multiplication_initial_condition():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx**2 + x*Dx - 1, x, 0, [3, 1])
    q = HolonomicFunction(Dx**2 + 1, x, 0, [1, 1])
    r = HolonomicFunction((x**4 + 14*x**2 + 60) + 4*x*Dx + (x**4 + 9*x**2 + 20)*Dx**2 + \
        (2*x**3 + 18*x)*Dx**3 + (x**2 + 10)*Dx**4, x, 0, [3, 4, 2, 3])
    assert p * q == r
    p = HolonomicFunction(Dx**2 + x, x, 0, [1, 0])
    q = HolonomicFunction(Dx**3 - x**2, x, 0, [3, 3, 3])
    r = HolonomicFunction((x**8 - 37*x**7/27 - 10*x**6/27 - 164*x**5/9 - 184*x**4/9 + \
        160*x**3/27 + 404*x**2/9 + 8*x + Rational(40, 3)) + (6*x**7 - 128*x**6/9 - 98*x**5/9 - 28*x**4/9 + \
        8*x**3/9 + 28*x**2 + x*Rational(40, 9) - 40)*Dx + (3*x**6 - 82*x**5/9 + 76*x**4/9 + 4*x**3/3 + \
        220*x**2/9 - x*Rational(80, 3))*Dx**2 + (-2*x**6 + 128*x**5/27 - 2*x**4/3 -80*x**2/9 + Rational(200, 9))*Dx**3 + \
        (3*x**5 - 64*x**4/9 - 28*x**3/9 + 6*x**2 - x*Rational(20, 9) - Rational(20, 3))*Dx**4 + (-4*x**3 + 64*x**2/9 + \
            x*Rational(8, 3))*Dx**5 + (x**4 - 64*x**3/27 - 4*x**2/3 + Rational(20, 9))*Dx**6, x, 0, [3, 3, 3, -3, -12, -24])
    assert p * q == r
    p = HolonomicFunction(Dx - 1, x, 0, [2])
    q = HolonomicFunction(Dx**2 + 1, x, 0, [0, 1])
    r = HolonomicFunction(2 -2*Dx + Dx**2, x, 0, [0, 2])
    assert p * q == r
    q = HolonomicFunction(x*Dx**2 + 1 + 2*Dx, x, 0,[0, 1])
    r = HolonomicFunction((x - 1) + (-2*x + 2)*Dx + x*Dx**2, x, 0, [0, 2])
    assert p * q == r
    p = HolonomicFunction(Dx**2 - 1, x, 0, [1, 3])
    q = HolonomicFunction(Dx**3 + 1, x, 0, [1, 2, 1])
    r = HolonomicFunction(6*Dx + 3*Dx**2 + 2*Dx**3 - 3*Dx**4 + Dx**6, x, 0, [1, 5, 14, 17, 17, 2])
    assert p * q == r
    p = expr_to_holonomic(sin(x))
    q = expr_to_holonomic(1/x, x0=1)
    r = HolonomicFunction(x + 2*Dx + x*Dx**2, x, 1, [sin(1), -sin(1) + cos(1)])
    assert p * q == r
    p = expr_to_holonomic(sqrt(x))
    q = expr_to_holonomic(sqrt(x**2-x))
    r = (p * q).to_expr()
    assert r == I*x*sqrt(-x + 1)


def test_HolonomicFunction_composition():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx-1, x).composition(x**2+x)
    r = HolonomicFunction((-2*x - 1) + Dx, x)
    assert p == r
    p = HolonomicFunction(Dx**2+1, x).composition(x**5+x**2+1)
    r = HolonomicFunction((125*x**12 + 150*x**9 + 60*x**6 + 8*x**3) + (-20*x**3 - 2)*Dx + \
        (5*x**4 + 2*x)*Dx**2, x)
    assert p == r
    p = HolonomicFunction(Dx**2*x+x, x).composition(2*x**3+x**2+1)
    r = HolonomicFunction((216*x**9 + 324*x**8 + 180*x**7 + 152*x**6 + 112*x**5 + \
        36*x**4 + 4*x**3) + (24*x**4 + 16*x**3 + 3*x**2 - 6*x - 1)*Dx + (6*x**5 + 5*x**4 + \
        x**3 + 3*x**2 + x)*Dx**2, x)
    assert p == r
    p = HolonomicFunction(Dx**2+1, x).composition(1-x**2)
    r = HolonomicFunction((4*x**3) - Dx + x*Dx**2, x)
    assert p == r
    p = HolonomicFunction(Dx**2+1, x).composition(x - 2/(x**2 + 1))
    r = HolonomicFunction((x**12 + 6*x**10 + 12*x**9 + 15*x**8 + 48*x**7 + 68*x**6 + \
        72*x**5 + 111*x**4 + 112*x**3 + 54*x**2 + 12*x + 1) + (12*x**8 + 32*x**6 + \
        24*x**4 - 4)*Dx + (x**12 + 6*x**10 + 4*x**9 + 15*x**8 + 16*x**7 + 20*x**6 + 24*x**5+ \
        15*x**4 + 16*x**3 + 6*x**2 + 4*x + 1)*Dx**2, x)
    assert p == r


def test_from_hyper():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = hyper([1, 1], [Rational(3, 2)], x**2/4)
    q = HolonomicFunction((4*x) + (5*x**2 - 8)*Dx + (x**3 - 4*x)*Dx**2, x, 1, [2*sqrt(3)*pi/9, -4*sqrt(3)*pi/27 + Rational(4, 3)])
    r = from_hyper(p)
    assert r == q
    p = from_hyper(hyper([1], [Rational(3, 2)], x**2/4))
    q = HolonomicFunction(-x + (-x**2/2 + 2)*Dx + x*Dx**2, x)
    # x0 = 1
    y0 = '[sqrt(pi)*exp(1/4)*erf(1/2), -sqrt(pi)*exp(1/4)*erf(1/2)/2 + 1]'
    assert sstr(p.y0) == y0
    assert q.annihilator == p.annihilator


def test_from_meijerg():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = from_meijerg(meijerg(([], [Rational(3, 2)]), ([S.Half], [S.Half, 1]), x))
    q = HolonomicFunction(x/2 - Rational(1, 4) + (-x**2 + x/4)*Dx + x**2*Dx**2 + x**3*Dx**3, x, 1, \
        [1/sqrt(pi), 1/(2*sqrt(pi)), -1/(4*sqrt(pi))])
    assert p == q
    p = from_meijerg(meijerg(([], []), ([0], []), x))
    q = HolonomicFunction(1 + Dx, x, 0, [1])
    assert p == q
    p = from_meijerg(meijerg(([1], []), ([S.Half], [0]), x))
    q = HolonomicFunction((x + S.Half)*Dx + x*Dx**2, x, 1, [sqrt(pi)*erf(1), exp(-1)])
    assert p == q
    p = from_meijerg(meijerg(([0], [1]), ([0], []), 2*x**2))
    q = HolonomicFunction((3*x**2 - 1)*Dx + x**3*Dx**2, x, 1, [-exp(Rational(-1, 2)) + 1, -exp(Rational(-1, 2))])
    assert p == q


def test_to_Sequence():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    n = symbols('n', integer=True)
    _, Sn = RecurrenceOperators(ZZ.old_poly_ring(n), 'Sn')
    p = HolonomicFunction(x**2*Dx**4 + x + Dx, x).to_sequence()
    q = [(HolonomicSequence(1 + (n + 2)*Sn**2 + (n**4 + 6*n**3 + 11*n**2 + 6*n)*Sn**3), 0, 1)]
    assert p == q
    p = HolonomicFunction(x**2*Dx**4 + x**3 + Dx**2, x).to_sequence()
    q = [(HolonomicSequence(1 + (n**4 + 14*n**3 + 72*n**2 + 163*n + 140)*Sn**5), 0, 0)]
    assert p == q
    p = HolonomicFunction(x**3*Dx**4 + 1 + Dx**2, x).to_sequence()
    q = [(HolonomicSequence(1 + (n**4 - 2*n**3 - n**2 + 2*n)*Sn + (n**2 + 3*n + 2)*Sn**2), 0, 0)]
    assert p == q
    p = HolonomicFunction(3*x**3*Dx**4 + 2*x*Dx + x*Dx**3, x).to_sequence()
    q = [(HolonomicSequence(2*n + (3*n**4 - 6*n**3 - 3*n**2 + 6*n)*Sn + (n**3 + 3*n**2 + 2*n)*Sn**2), 0, 1)]
    assert p == q


def test_to_Sequence_Initial_Coniditons():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    n = symbols('n', integer=True)
    _, Sn = RecurrenceOperators(QQ.old_poly_ring(n), 'Sn')
    p = HolonomicFunction(Dx - 1, x, 0, [1]).to_sequence()
    q = [(HolonomicSequence(-1 + (n + 1)*Sn, 1), 0)]
    assert p == q
    p = HolonomicFunction(Dx**2 + 1, x, 0, [0, 1]).to_sequence()
    q = [(HolonomicSequence(1 + (n**2 + 3*n + 2)*Sn**2, [0, 1]), 0)]
    assert p == q
    p = HolonomicFunction(Dx**2 + 1 + x**3*Dx, x, 0, [2, 3]).to_sequence()
    q = [(HolonomicSequence(n + Sn**2 + (n**2 + 7*n + 12)*Sn**4, [2, 3, -1, Rational(-1, 2), Rational(1, 12)]), 1)]
    assert p == q
    p = HolonomicFunction(x**3*Dx**5 + 1 + Dx, x).to_sequence()
    q = [(HolonomicSequence(1 + (n + 1)*Sn + (n**5 - 5*n**3 + 4*n)*Sn**2), 0, 3)]
    assert p == q
    C_0, C_1, C_2, C_3 = symbols('C_0, C_1, C_2, C_3')
    p = expr_to_holonomic(log(1+x**2))
    q = [(HolonomicSequence(n**2 + (n**2 + 2*n)*Sn**2, [0, 0, C_2]), 0, 1)]
    assert p.to_sequence() == q
    p = p.diff()
    q = [(HolonomicSequence((n + 2) + (n + 2)*Sn**2, [C_0, 0]), 1, 0)]
    assert p.to_sequence() == q
    p = expr_to_holonomic(erf(x) + x).to_sequence()
    q = [(HolonomicSequence((2*n**2 - 2*n) + (n**3 + 2*n**2 - n - 2)*Sn**2, [0, 1 + 2/sqrt(pi), 0, C_3]), 0, 2)]
    assert p == q

def test_series():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx**2 + 2*x*Dx, x, 0, [0, 1]).series(n=10)
    q = x - x**3/3 + x**5/10 - x**7/42 + x**9/216 + O(x**10)
    assert p == q
    p = HolonomicFunction(Dx - 1, x).composition(x**2, 0, [1])  # e^(x**2)
    q = HolonomicFunction(Dx**2 + 1, x, 0, [1, 0])  # cos(x)
    r = (p * q).series(n=10)  # expansion of cos(x) * exp(x**2)
    s = 1 + x**2/2 + x**4/24 - 31*x**6/720 - 179*x**8/8064 + O(x**10)
    assert r == s
    t = HolonomicFunction((1 + x)*Dx**2 + Dx, x, 0, [0, 1])  # log(1 + x)
    r = (p * t + q).series(n=10)
    s = 1 + x - x**2 + 4*x**3/3 - 17*x**4/24 + 31*x**5/30 - 481*x**6/720 +\
     71*x**7/105 - 20159*x**8/40320 + 379*x**9/840 + O(x**10)
    assert r == s
    p = HolonomicFunction((6+6*x-3*x**2) - (10*x-3*x**2-3*x**3)*Dx + \
        (4-6*x**3+2*x**4)*Dx**2, x, 0, [0, 1]).series(n=7)
    q = x + x**3/6 - 3*x**4/16 + x**5/20 - 23*x**6/960 + O(x**7)
    assert p == q
    p = HolonomicFunction((6+6*x-3*x**2) - (10*x-3*x**2-3*x**3)*Dx + \
        (4-6*x**3+2*x**4)*Dx**2, x, 0, [1, 0]).series(n=7)
    q = 1 - 3*x**2/4 - x**3/4 - 5*x**4/32 - 3*x**5/40 - 17*x**6/384 + O(x**7)
    assert p == q
    p = expr_to_holonomic(erf(x) + x).series(n=10)
    C_3 = symbols('C_3')
    q = (erf(x) + x).series(n=10)
    assert p.subs(C_3, -2/(3*sqrt(pi))) == q
    assert expr_to_holonomic(sqrt(x**3 + x)).series(n=10) == sqrt(x**3 + x).series(n=10)
    assert expr_to_holonomic((2*x - 3*x**2)**Rational(1, 3)).series() == ((2*x - 3*x**2)**Rational(1, 3)).series()
    assert  expr_to_holonomic(sqrt(x**2-x)).series() == (sqrt(x**2-x)).series()
    assert expr_to_holonomic(cos(x)**2/x**2, y0={-2: [1, 0, -1]}).series(n=10) == (cos(x)**2/x**2).series(n=10)
    assert expr_to_holonomic(cos(x)**2/x**2, x0=1).series(n=10).together() == (cos(x)**2/x**2).series(n=10, x0=1).together()
    assert expr_to_holonomic(cos(x-1)**2/(x-1)**2, x0=1, y0={-2: [1, 0, -1]}).series(n=10) \
        == (cos(x-1)**2/(x-1)**2).series(x0=1, n=10)

def test_evalf_euler():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')

    # log(1+x)
    p = HolonomicFunction((1 + x)*Dx**2 + Dx, x, 0, [0, 1])

    # path taken is a straight line from 0 to 1, on the real axis
    r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    s = '0.699525841805253'  # approx. equal to log(2) i.e. 0.693147180559945
    assert sstr(p.evalf(r, method='Euler')[-1]) == s

    # path taken is a triangle 0-->1+i-->2
    r = [0.1 + 0.1*I]
    for i in range(9):
        r.append(r[-1]+0.1+0.1*I)
    for i in range(10):
        r.append(r[-1]+0.1-0.1*I)

    # close to the exact solution 1.09861228866811
    # imaginary part also close to zero
    s = '1.07530466271334 - 0.0251200594793912*I'
    assert sstr(p.evalf(r, method='Euler')[-1]) == s

    # sin(x)
    p = HolonomicFunction(Dx**2 + 1, x, 0, [0, 1])
    s = '0.905546532085401 - 6.93889390390723e-18*I'
    assert sstr(p.evalf(r, method='Euler')[-1]) == s

    # computing sin(pi/2) using this method
    # using a linear path from 0 to pi/2
    r = [0.1]
    for i in range(14):
        r.append(r[-1] + 0.1)
    r.append(pi/2)
    s = '1.08016557252834' # close to 1.0 (exact solution)
    assert sstr(p.evalf(r, method='Euler')[-1]) == s

    # trying different path, a rectangle (0-->i-->pi/2 + i-->pi/2)
    # computing the same value sin(pi/2) using different path
    r = [0.1*I]
    for i in range(9):
        r.append(r[-1]+0.1*I)
    for i in range(15):
        r.append(r[-1]+0.1)
    r.append(pi/2+I)
    for i in range(10):
        r.append(r[-1]-0.1*I)

    # close to 1.0
    s = '0.976882381836257 - 1.65557671738537e-16*I'
    assert sstr(p.evalf(r, method='Euler')[-1]) == s

    # cos(x)
    p = HolonomicFunction(Dx**2 + 1, x, 0, [1, 0])
    # compute cos(pi) along 0-->pi
    r = [0.05]
    for i in range(61):
        r.append(r[-1]+0.05)
    r.append(pi)
    # close to -1 (exact answer)
    s = '-1.08140824719196'
    assert sstr(p.evalf(r, method='Euler')[-1]) == s

    # a rectangular path (0 -> i -> 2+i -> 2)
    r = [0.1*I]
    for i in range(9):
        r.append(r[-1]+0.1*I)
    for i in range(20):
        r.append(r[-1]+0.1)
    for i in range(10):
        r.append(r[-1]-0.1*I)

    p = HolonomicFunction(Dx**2 + 1, x, 0, [1,1]).evalf(r, method='Euler')
    s = '0.501421652861245 - 3.88578058618805e-16*I'
    assert sstr(p[-1]) == s

def test_evalf_rk4():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')

    # log(1+x)
    p = HolonomicFunction((1 + x)*Dx**2 + Dx, x, 0, [0, 1])

    # path taken is a straight line from 0 to 1, on the real axis
    r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    s = '0.693146363174626'  # approx. equal to log(2) i.e. 0.693147180559945
    assert sstr(p.evalf(r)[-1]) == s

    # path taken is a triangle 0-->1+i-->2
    r = [0.1 + 0.1*I]
    for i in range(9):
        r.append(r[-1]+0.1+0.1*I)
    for i in range(10):
        r.append(r[-1]+0.1-0.1*I)

    # close to the exact solution 1.09861228866811
    # imaginary part also close to zero
    s = '1.098616 + 1.36083e-7*I'
    assert sstr(p.evalf(r)[-1].n(7)) == s

    # sin(x)
    p = HolonomicFunction(Dx**2 + 1, x, 0, [0, 1])
    s = '0.90929463522785 + 1.52655665885959e-16*I'
    assert sstr(p.evalf(r)[-1]) == s

    # computing sin(pi/2) using this method
    # using a linear path from 0 to pi/2
    r = [0.1]
    for i in range(14):
        r.append(r[-1] + 0.1)
    r.append(pi/2)
    s = '0.999999895088917' # close to 1.0 (exact solution)
    assert sstr(p.evalf(r)[-1]) == s

    # trying different path, a rectangle (0-->i-->pi/2 + i-->pi/2)
    # computing the same value sin(pi/2) using different path
    r = [0.1*I]
    for i in range(9):
        r.append(r[-1]+0.1*I)
    for i in range(15):
        r.append(r[-1]+0.1)
    r.append(pi/2+I)
    for i in range(10):
        r.append(r[-1]-0.1*I)

    # close to 1.0
    s = '1.00000003415141 + 6.11940487991086e-16*I'
    assert sstr(p.evalf(r)[-1]) == s

    # cos(x)
    p = HolonomicFunction(Dx**2 + 1, x, 0, [1, 0])
    # compute cos(pi) along 0-->pi
    r = [0.05]
    for i in range(61):
        r.append(r[-1]+0.05)
    r.append(pi)
    # close to -1 (exact answer)
    s = '-0.999999993238714'
    assert sstr(p.evalf(r)[-1]) == s

    # a rectangular path (0 -> i -> 2+i -> 2)
    r = [0.1*I]
    for i in range(9):
        r.append(r[-1]+0.1*I)
    for i in range(20):
        r.append(r[-1]+0.1)
    for i in range(10):
        r.append(r[-1]-0.1*I)

    p = HolonomicFunction(Dx**2 + 1, x, 0, [1,1]).evalf(r)
    s = '0.493152791638442 - 1.41553435639707e-15*I'
    assert sstr(p[-1]) == s


def test_expr_to_holonomic():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = expr_to_holonomic((sin(x)/x)**2)
    q = HolonomicFunction(8*x + (4*x**2 + 6)*Dx + 6*x*Dx**2 + x**2*Dx**3, x, 0, \
        [1, 0, Rational(-2, 3)])
    assert p == q
    p = expr_to_holonomic(1/(1+x**2)**2)
    q = HolonomicFunction(4*x + (x**2 + 1)*Dx, x, 0, [1])
    assert p == q
    p = expr_to_holonomic(exp(x)*sin(x)+x*log(1+x))
    q = HolonomicFunction((2*x**3 + 10*x**2 + 20*x + 18) + (-2*x**4 - 10*x**3 - 20*x**2 \
        - 18*x)*Dx + (2*x**5 + 6*x**4 + 7*x**3 + 8*x**2 + 10*x - 4)*Dx**2 + \
        (-2*x**5 - 5*x**4 - 2*x**3 + 2*x**2 - x + 4)*Dx**3 + (x**5 + 2*x**4 - x**3 - \
        7*x**2/2 + x + Rational(5, 2))*Dx**4, x, 0, [0, 1, 4, -1])
    assert p == q
    p = expr_to_holonomic(x*exp(x)+cos(x)+1)
    q = HolonomicFunction((-x - 3)*Dx + (x + 2)*Dx**2 + (-x - 3)*Dx**3 + (x + 2)*Dx**4, x, \
        0, [2, 1, 1, 3])
    assert p == q
    assert (x*exp(x)+cos(x)+1).series(n=10) == p.series(n=10)
    p = expr_to_holonomic(log(1 + x)**2 + 1)
    q = HolonomicFunction(Dx + (3*x + 3)*Dx**2 + (x**2 + 2*x + 1)*Dx**3, x, 0, [1, 0, 2])
    assert p == q
    p = expr_to_holonomic(erf(x)**2 + x)
    q = HolonomicFunction((8*x**4 - 2*x**2 + 2)*Dx**2 + (6*x**3 - x/2)*Dx**3 + \
        (x**2+ Rational(1, 4))*Dx**4, x, 0, [0, 1, 8/pi, 0])
    assert p == q
    p = expr_to_holonomic(cosh(x)*x)
    q = HolonomicFunction((-x**2 + 2) -2*x*Dx + x**2*Dx**2, x, 0, [0, 1])
    assert p == q
    p = expr_to_holonomic(besselj(2, x))
    q = HolonomicFunction((x**2 - 4) + x*Dx + x**2*Dx**2, x, 0, [0, 0])
    assert p == q
    p = expr_to_holonomic(besselj(0, x) + exp(x))
    q = HolonomicFunction((-x**2 - x/2 + S.Half) + (x**2 - x/2 - Rational(3, 2))*Dx + (-x**2 + x/2 + 1)*Dx**2 +\
        (x**2 + x/2)*Dx**3, x, 0, [2, 1, S.Half])
    assert p == q
    p = expr_to_holonomic(sin(x)**2/x)
    q = HolonomicFunction(4 + 4*x*Dx + 3*Dx**2 + x*Dx**3, x, 0, [0, 1, 0])
    assert p == q
    p = expr_to_holonomic(sin(x)**2/x, x0=2)
    q = HolonomicFunction((4) + (4*x)*Dx + (3)*Dx**2 + (x)*Dx**3, x, 2, [sin(2)**2/2,
        sin(2)*cos(2) - sin(2)**2/4, -3*sin(2)**2/4 + cos(2)**2 - sin(2)*cos(2)])
    assert p == q
    p = expr_to_holonomic(log(x)/2 - Ci(2*x)/2 + Ci(2)/2)
    q = HolonomicFunction(4*Dx + 4*x*Dx**2 + 3*Dx**3 + x*Dx**4, x, 0, \
        [-log(2)/2 - EulerGamma/2 + Ci(2)/2, 0, 1, 0])
    assert p == q
    p = p.to_expr()
    q = log(x)/2 - Ci(2*x)/2 + Ci(2)/2
    assert p == q
    p = expr_to_holonomic(x**S.Half, x0=1)
    q = HolonomicFunction(x*Dx - S.Half, x, 1, [1])
    assert p == q
    p = expr_to_holonomic(sqrt(1 + x**2))
    q = HolonomicFunction((-x) + (x**2 + 1)*Dx, x, 0, [1])
    assert p == q
    assert (expr_to_holonomic(sqrt(x) + sqrt(2*x)).to_expr()-\
        (sqrt(x) + sqrt(2*x))).simplify() == 0
    assert expr_to_holonomic(3*x+2*sqrt(x)).to_expr() == 3*x+2*sqrt(x)
    p = expr_to_holonomic((x**4+x**3+5*x**2+3*x+2)/x**2, lenics=3)
    q = HolonomicFunction((-2*x**4 - x**3 + 3*x + 4) + (x**5 + x**4 + 5*x**3 + 3*x**2 + \
        2*x)*Dx, x, 0, {-2: [2, 3, 5]})
    assert p == q
    p = expr_to_holonomic(1/(x-1)**2, lenics=3, x0=1)
    q = HolonomicFunction((2) + (x - 1)*Dx, x, 1, {-2: [1, 0, 0]})
    assert p == q
    a = symbols("a")
    p = expr_to_holonomic(sqrt(a*x), x=x)
    assert p.to_expr() == sqrt(a)*sqrt(x)

def test_to_hyper():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx - 2, x, 0, [3]).to_hyper()
    q = 3 * hyper([], [], 2*x)
    assert p == q
    p = hyperexpand(HolonomicFunction((1 + x) * Dx - 3, x, 0, [2]).to_hyper()).expand()
    q = 2*x**3 + 6*x**2 + 6*x + 2
    assert p == q
    p = HolonomicFunction((1 + x)*Dx**2 + Dx, x, 0, [0, 1]).to_hyper()
    q = -x**2*hyper((2, 2, 1), (3, 2), -x)/2 + x
    assert p == q
    p = HolonomicFunction(2*x*Dx + Dx**2, x, 0, [0, 2/sqrt(pi)]).to_hyper()
    q = 2*x*hyper((S.Half,), (Rational(3, 2),), -x**2)/sqrt(pi)
    assert p == q
    p = hyperexpand(HolonomicFunction(2*x*Dx + Dx**2, x, 0, [1, -2/sqrt(pi)]).to_hyper())
    q = erfc(x)
    assert p.rewrite(erfc) == q
    p =  hyperexpand(HolonomicFunction((x**2 - 1) + x*Dx + x**2*Dx**2,
        x, 0, [0, S.Half]).to_hyper())
    q = besselj(1, x)
    assert p == q
    p = hyperexpand(HolonomicFunction(x*Dx**2 + Dx + x, x, 0, [1, 0]).to_hyper())
    q = besselj(0, x)
    assert p == q

def test_to_expr():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx - 1, x, 0, [1]).to_expr()
    q = exp(x)
    assert p == q
    p = HolonomicFunction(Dx**2 + 1, x, 0, [1, 0]).to_expr()
    q = cos(x)
    assert p == q
    p = HolonomicFunction(Dx**2 - 1, x, 0, [1, 0]).to_expr()
    q = cosh(x)
    assert p == q
    p = HolonomicFunction(2 + (4*x - 1)*Dx + \
        (x**2 - x)*Dx**2, x, 0, [1, 2]).to_expr().expand()
    q = 1/(x**2 - 2*x + 1)
    assert p == q
    p = expr_to_holonomic(sin(x)**2/x).integrate((x, 0, x)).to_expr()
    q = (sin(x)**2/x).integrate((x, 0, x))
    assert p == q
    C_0, C_1, C_2, C_3 = symbols('C_0, C_1, C_2, C_3')
    p = expr_to_holonomic(log(1+x**2)).to_expr()
    q = C_2*log(x**2 + 1)
    assert p == q
    p = expr_to_holonomic(log(1+x**2)).diff().to_expr()
    q = C_0*x/(x**2 + 1)
    assert p == q
    p = expr_to_holonomic(erf(x) + x).to_expr()
    q = 3*C_3*x - 3*sqrt(pi)*C_3*erf(x)/2 + x + 2*x/sqrt(pi)
    assert p == q
    p = expr_to_holonomic(sqrt(x), x0=1).to_expr()
    assert p == sqrt(x)
    assert expr_to_holonomic(sqrt(x)).to_expr() == sqrt(x)
    p = expr_to_holonomic(sqrt(1 + x**2)).to_expr()
    assert p == sqrt(1+x**2)
    p = expr_to_holonomic((2*x**2 + 1)**Rational(2, 3)).to_expr()
    assert p == (2*x**2 + 1)**Rational(2, 3)
    p = expr_to_holonomic(sqrt(-x**2+2*x)).to_expr()
    assert p == sqrt(x)*sqrt(-x + 2)
    p = expr_to_holonomic((-2*x**3+7*x)**Rational(2, 3)).to_expr()
    q = x**Rational(2, 3)*(-2*x**2 + 7)**Rational(2, 3)
    assert p == q
    p = from_hyper(hyper((-2, -3), (S.Half, ), x))
    s = hyperexpand(hyper((-2, -3), (S.Half, ), x))
    D_0 = Symbol('D_0')
    C_0 = Symbol('C_0')
    assert (p.to_expr().subs({C_0:1, D_0:0}) - s).simplify() == 0
    p.y0 = {0: [1], S.Half: [0]}
    assert p.to_expr() == s
    assert expr_to_holonomic(x**5).to_expr() == x**5
    assert expr_to_holonomic(2*x**3-3*x**2).to_expr().expand() == \
        2*x**3-3*x**2
    a = symbols("a")
    p = (expr_to_holonomic(1.4*x)*expr_to_holonomic(a*x, x)).to_expr()
    q = 1.4*a*x**2
    assert p == q
    p = (expr_to_holonomic(1.4*x)+expr_to_holonomic(a*x, x)).to_expr()
    q = x*(a + 1.4)
    assert p == q
    p = (expr_to_holonomic(1.4*x)+expr_to_holonomic(x)).to_expr()
    assert p == 2.4*x


def test_integrate():
    x = symbols('x')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = expr_to_holonomic(sin(x)**2/x, x0=1).integrate((x, 2, 3))
    q = '0.166270406994788'
    assert sstr(p) == q
    p = expr_to_holonomic(sin(x)).integrate((x, 0, x)).to_expr()
    q = 1 - cos(x)
    assert p == q
    p = expr_to_holonomic(sin(x)).integrate((x, 0, 3))
    q = 1 - cos(3)
    assert p == q
    p = expr_to_holonomic(sin(x)/x, x0=1).integrate((x, 1, 2))
    q = '0.659329913368450'
    assert sstr(p) == q
    p = expr_to_holonomic(sin(x)**2/x, x0=1).integrate((x, 1, 0))
    q = '-0.423690480850035'
    assert sstr(p) == q
    p = expr_to_holonomic(sin(x)/x)
    assert p.integrate(x).to_expr() == Si(x)
    assert p.integrate((x, 0, 2)) == Si(2)
    p = expr_to_holonomic(sin(x)**2/x)
    q = p.to_expr()
    assert p.integrate(x).to_expr() == q.integrate((x, 0, x))
    assert p.integrate((x, 0, 1)) == q.integrate((x, 0, 1))
    assert expr_to_holonomic(1/x, x0=1).integrate(x).to_expr() == log(x)
    p = expr_to_holonomic((x + 1)**3*exp(-x), x0=-1).integrate(x).to_expr()
    q = (-x**3 - 6*x**2 - 15*x + 6*exp(x + 1) - 16)*exp(-x)
    assert p == q
    p = expr_to_holonomic(cos(x)**2/x**2, y0={-2: [1, 0, -1]}).integrate(x).to_expr()
    q = -Si(2*x) - cos(x)**2/x
    assert p == q
    p = expr_to_holonomic(sqrt(x**2+x)).integrate(x).to_expr()
    q = (x**Rational(3, 2)*(2*x**2 + 3*x + 1) - x*sqrt(x + 1)*asinh(sqrt(x)))/(4*x*sqrt(x + 1))
    assert p == q
    p = expr_to_holonomic(sqrt(x**2+1)).integrate(x).to_expr()
    q = (sqrt(x**2+1)).integrate(x)
    assert (p-q).simplify() == 0
    p = expr_to_holonomic(1/x**2, y0={-2:[1, 0, 0]})
    r = expr_to_holonomic(1/x**2, lenics=3)
    assert p == r
    q = expr_to_holonomic(cos(x)**2)
    assert (r*q).integrate(x).to_expr() == -Si(2*x) - cos(x)**2/x


def test_diff():
    x, y = symbols('x, y')
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(x*Dx**2 + 1, x, 0, [0, 1])
    assert p.diff().to_expr() == p.to_expr().diff().simplify()
    p = HolonomicFunction(Dx**2 - 1, x, 0, [1, 0])
    assert p.diff(x, 2).to_expr() == p.to_expr()
    p = expr_to_holonomic(Si(x))
    assert p.diff().to_expr() == sin(x)/x
    assert p.diff(y) == 0
    C_0, C_1, C_2, C_3 = symbols('C_0, C_1, C_2, C_3')
    q = Si(x)
    assert p.diff(x).to_expr() == q.diff()
    assert p.diff(x, 2).to_expr().subs(C_0, Rational(-1, 3)).cancel() == q.diff(x, 2).cancel()
    assert p.diff(x, 3).series().subs({C_3: Rational(-1, 3), C_0: 0}) == q.diff(x, 3).series()


def test_extended_domain_in_expr_to_holonomic():
    x = symbols('x')
    p = expr_to_holonomic(1.2*cos(3.1*x))
    assert p.to_expr() == 1.2*cos(3.1*x)
    assert sstr(p.integrate(x).to_expr()) == '0.387096774193548*sin(3.1*x)'
    _, Dx = DifferentialOperators(RR.old_poly_ring(x), 'Dx')
    p = expr_to_holonomic(1.1329138213*x)
    q = HolonomicFunction((-1.1329138213) + (1.1329138213*x)*Dx, x, 0, {1: [1.1329138213]})
    assert p == q
    assert p.to_expr() == 1.1329138213*x
    assert sstr(p.integrate((x, 1, 2))) == sstr((1.1329138213*x).integrate((x, 1, 2)))
    y, z = symbols('y, z')
    p = expr_to_holonomic(sin(x*y*z), x=x)
    assert p.to_expr() == sin(x*y*z)
    assert p.integrate(x).to_expr() == (-cos(x*y*z) + 1)/(y*z)
    p = expr_to_holonomic(sin(x*y + z), x=x).integrate(x).to_expr()
    q = (cos(z) - cos(x*y + z))/y
    assert p == q
    a = symbols('a')
    p = expr_to_holonomic(a*x, x)
    assert p.to_expr() == a*x
    assert p.integrate(x).to_expr() == a*x**2/2
    D_2, C_1 = symbols("D_2, C_1")
    p = expr_to_holonomic(x) + expr_to_holonomic(1.2*cos(x))
    p = p.to_expr().subs(D_2, 0)
    assert p - x - 1.2*cos(1.0*x) == 0
    p = expr_to_holonomic(x) * expr_to_holonomic(1.2*cos(x))
    p = p.to_expr().subs(C_1, 0)
    assert p - 1.2*x*cos(1.0*x) == 0


def test_to_meijerg():
    x = symbols('x')
    assert hyperexpand(expr_to_holonomic(sin(x)).to_meijerg()) == sin(x)
    assert hyperexpand(expr_to_holonomic(cos(x)).to_meijerg()) == cos(x)
    assert hyperexpand(expr_to_holonomic(exp(x)).to_meijerg()) == exp(x)
    assert hyperexpand(expr_to_holonomic(log(x)).to_meijerg()).simplify() == log(x)
    assert expr_to_holonomic(4*x**2/3 + 7).to_meijerg() == 4*x**2/3 + 7
    assert hyperexpand(expr_to_holonomic(besselj(2, x), lenics=3).to_meijerg()) == besselj(2, x)
    p = hyper((Rational(-1, 2), -3), (), x)
    assert from_hyper(p).to_meijerg() == hyperexpand(p)
    p = hyper((S.One, S(3)), (S(2), ), x)
    assert (hyperexpand(from_hyper(p).to_meijerg()) - hyperexpand(p)).expand() == 0
    p = from_hyper(hyper((-2, -3), (S.Half, ), x))
    s = hyperexpand(hyper((-2, -3), (S.Half, ), x))
    C_0 = Symbol('C_0')
    C_1 = Symbol('C_1')
    D_0 = Symbol('D_0')
    assert (hyperexpand(p.to_meijerg()).subs({C_0:1, D_0:0}) - s).simplify() == 0
    p.y0 = {0: [1], S.Half: [0]}
    assert (hyperexpand(p.to_meijerg()) - s).simplify() == 0
    p = expr_to_holonomic(besselj(S.Half, x), initcond=False)
    assert (p.to_expr() - (D_0*sin(x) + C_0*cos(x) + C_1*sin(x))/sqrt(x)).simplify() == 0
    p = expr_to_holonomic(besselj(S.Half, x), y0={Rational(-1, 2): [sqrt(2)/sqrt(pi), sqrt(2)/sqrt(pi)]})
    assert (p.to_expr() - besselj(S.Half, x) - besselj(Rational(-1, 2), x)).simplify() == 0


def test_gaussian():
    mu, x = symbols("mu x")
    sd = symbols("sd", positive=True)
    Q = QQ[mu, sd].get_field()
    e = sqrt(2)*exp(-(-mu + x)**2/(2*sd**2))/(2*sqrt(pi)*sd)
    h1 = expr_to_holonomic(e, x, domain=Q)

    _, Dx = DifferentialOperators(Q.old_poly_ring(x), 'Dx')
    h2 = HolonomicFunction((-mu/sd**2 + x/sd**2) + (1)*Dx, x)

    assert h1 == h2


def test_beta():
    a, b, x = symbols("a b x", positive=True)
    e = x**(a - 1)*(-x + 1)**(b - 1)/beta(a, b)
    Q = QQ[a, b].get_field()
    h1 = expr_to_holonomic(e, x, domain=Q)

    _, Dx = DifferentialOperators(Q.old_poly_ring(x), 'Dx')
    h2 = HolonomicFunction((a + x*(-a - b + 2) - 1) + (x**2 - x)*Dx, x)

    assert h1 == h2


def test_gamma():
    a, b, x = symbols("a b x", positive=True)
    e = b**(-a)*x**(a - 1)*exp(-x/b)/gamma(a)
    Q = QQ[a, b].get_field()
    h1 = expr_to_holonomic(e, x, domain=Q)

    _, Dx = DifferentialOperators(Q.old_poly_ring(x), 'Dx')
    h2 = HolonomicFunction((-a + 1 + x/b) + (x)*Dx, x)

    assert h1 == h2


def test_symbolic_power():
    x, n = symbols("x n")
    Q = QQ[n].get_field()
    _, Dx = DifferentialOperators(Q.old_poly_ring(x), 'Dx')
    h1 = HolonomicFunction((-1) + (x)*Dx, x) ** -n
    h2 = HolonomicFunction((n) + (x)*Dx, x)

    assert h1 == h2


def test_negative_power():
    x = symbols("x")
    _, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    h1 = HolonomicFunction((-1) + (x)*Dx, x) ** -2
    h2 = HolonomicFunction((2) + (x)*Dx, x)

    assert h1 == h2


def test_expr_in_power():
    x, n = symbols("x n")
    Q = QQ[n].get_field()
    _, Dx = DifferentialOperators(Q.old_poly_ring(x), 'Dx')
    h1 = HolonomicFunction((-1) + (x)*Dx, x) ** (n - 3)
    h2 = HolonomicFunction((-n + 3) + (x)*Dx, x)

    assert h1 == h2


def test_DifferentialOperatorEqPoly():
    x = symbols('x', integer=True)
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    do = DifferentialOperator([x**2, R.base.zero, R.base.zero], R)
    do2 = DifferentialOperator([x**2, 1, x], R)
    assert not do == do2

    # polynomial comparison issue, see https://github.com/sympy/sympy/pull/15799
    # should work once that is solved
    # p = do.listofpoly[0]
    # assert do == p

    p2 = do2.listofpoly[0]
    assert not do2 == p2


def test_DifferentialOperatorPow():
    x = symbols('x', integer=True)
    R, _ = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    do = DifferentialOperator([x**2, R.base.zero, R.base.zero], R)
    a = DifferentialOperator([R.base.one], R)
    for n in range(10):
        assert a == do**n
        a *= do

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\accessors\test_dt_accessor.py ===
import calendar
from datetime import (
    date,
    datetime,
    time,
)
import locale
import unicodedata

import numpy as np
import pytest
import pytz

from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas.errors import SettingWithCopyError

from pandas.core.dtypes.common import (
    is_integer_dtype,
    is_list_like,
)

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Period,
    PeriodIndex,
    Series,
    StringDtype,
    TimedeltaIndex,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    PeriodArray,
    TimedeltaArray,
)

ok_for_period = PeriodArray._datetimelike_ops
ok_for_period_methods = ["strftime", "to_timestamp", "asfreq"]
ok_for_dt = DatetimeArray._datetimelike_ops
ok_for_dt_methods = [
    "to_period",
    "to_pydatetime",
    "tz_localize",
    "tz_convert",
    "normalize",
    "strftime",
    "round",
    "floor",
    "ceil",
    "day_name",
    "month_name",
    "isocalendar",
    "as_unit",
]
ok_for_td = TimedeltaArray._datetimelike_ops
ok_for_td_methods = [
    "components",
    "to_pytimedelta",
    "total_seconds",
    "round",
    "floor",
    "ceil",
    "as_unit",
]


def get_dir(ser):
    # check limited display api
    results = [r for r in ser.dt.__dir__() if not r.startswith("_")]
    return sorted(set(results))


class TestSeriesDatetimeValues:
    def _compare(self, ser, name):
        # GH 7207, 11128
        # test .dt namespace accessor

        def get_expected(ser, prop):
            result = getattr(Index(ser._values), prop)
            if isinstance(result, np.ndarray):
                if is_integer_dtype(result):
                    result = result.astype("int64")
            elif not is_list_like(result) or isinstance(result, DataFrame):
                return result
            return Series(result, index=ser.index, name=ser.name)

        left = getattr(ser.dt, name)
        right = get_expected(ser, name)
        if not (is_list_like(left) and is_list_like(right)):
            assert left == right
        elif isinstance(left, DataFrame):
            tm.assert_frame_equal(left, right)
        else:
            tm.assert_series_equal(left, right)

    @pytest.mark.parametrize("freq", ["D", "s", "ms"])
    def test_dt_namespace_accessor_datetime64(self, freq):
        # GH#7207, GH#11128
        # test .dt namespace accessor

        # datetimeindex
        dti = date_range("20130101", periods=5, freq=freq)
        ser = Series(dti, name="xxx")

        for prop in ok_for_dt:
            # we test freq below
            if prop != "freq":
                self._compare(ser, prop)

        for prop in ok_for_dt_methods:
            getattr(ser.dt, prop)

        msg = "The behavior of DatetimeProperties.to_pydatetime is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ser.dt.to_pydatetime()
        assert isinstance(result, np.ndarray)
        assert result.dtype == object

        result = ser.dt.tz_localize("US/Eastern")
        exp_values = DatetimeIndex(ser.values).tz_localize("US/Eastern")
        expected = Series(exp_values, index=ser.index, name="xxx")
        tm.assert_series_equal(result, expected)

        tz_result = result.dt.tz
        assert str(tz_result) == "US/Eastern"
        freq_result = ser.dt.freq
        assert freq_result == DatetimeIndex(ser.values, freq="infer").freq

        # let's localize, then convert
        result = ser.dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
        exp_values = (
            DatetimeIndex(ser.values).tz_localize("UTC").tz_convert("US/Eastern")
        )
        expected = Series(exp_values, index=ser.index, name="xxx")
        tm.assert_series_equal(result, expected)

    def test_dt_namespace_accessor_datetime64tz(self):
        # GH#7207, GH#11128
        # test .dt namespace accessor

        # datetimeindex with tz
        dti = date_range("20130101", periods=5, tz="US/Eastern")
        ser = Series(dti, name="xxx")
        for prop in ok_for_dt:
            # we test freq below
            if prop != "freq":
                self._compare(ser, prop)

        for prop in ok_for_dt_methods:
            getattr(ser.dt, prop)

        msg = "The behavior of DatetimeProperties.to_pydatetime is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ser.dt.to_pydatetime()
        assert isinstance(result, np.ndarray)
        assert result.dtype == object

        result = ser.dt.tz_convert("CET")
        expected = Series(ser._values.tz_convert("CET"), index=ser.index, name="xxx")
        tm.assert_series_equal(result, expected)

        tz_result = result.dt.tz
        assert str(tz_result) == "CET"
        freq_result = ser.dt.freq
        assert freq_result == DatetimeIndex(ser.values, freq="infer").freq

    def test_dt_namespace_accessor_timedelta(self):
        # GH#7207, GH#11128
        # test .dt namespace accessor

        # timedelta index
        cases = [
            Series(
                timedelta_range("1 day", periods=5), index=list("abcde"), name="xxx"
            ),
            Series(timedelta_range("1 day 01:23:45", periods=5, freq="s"), name="xxx"),
            Series(
                timedelta_range("2 days 01:23:45.012345", periods=5, freq="ms"),
                name="xxx",
            ),
        ]
        for ser in cases:
            for prop in ok_for_td:
                # we test freq below
                if prop != "freq":
                    self._compare(ser, prop)

            for prop in ok_for_td_methods:
                getattr(ser.dt, prop)

            result = ser.dt.components
            assert isinstance(result, DataFrame)
            tm.assert_index_equal(result.index, ser.index)

            result = ser.dt.to_pytimedelta()
            assert isinstance(result, np.ndarray)
            assert result.dtype == object

            result = ser.dt.total_seconds()
            assert isinstance(result, Series)
            assert result.dtype == "float64"

            freq_result = ser.dt.freq
            assert freq_result == TimedeltaIndex(ser.values, freq="infer").freq

    def test_dt_namespace_accessor_period(self):
        # GH#7207, GH#11128
        # test .dt namespace accessor

        # periodindex
        pi = period_range("20130101", periods=5, freq="D")
        ser = Series(pi, name="xxx")

        for prop in ok_for_period:
            # we test freq below
            if prop != "freq":
                self._compare(ser, prop)

        for prop in ok_for_period_methods:
            getattr(ser.dt, prop)

        freq_result = ser.dt.freq
        assert freq_result == PeriodIndex(ser.values).freq

    def test_dt_namespace_accessor_index_and_values(self):
        # both
        index = date_range("20130101", periods=3, freq="D")
        dti = date_range("20140204", periods=3, freq="s")
        ser = Series(dti, index=index, name="xxx")
        exp = Series(
            np.array([2014, 2014, 2014], dtype="int32"), index=index, name="xxx"
        )
        tm.assert_series_equal(ser.dt.year, exp)

        exp = Series(np.array([2, 2, 2], dtype="int32"), index=index, name="xxx")
        tm.assert_series_equal(ser.dt.month, exp)

        exp = Series(np.array([0, 1, 2], dtype="int32"), index=index, name="xxx")
        tm.assert_series_equal(ser.dt.second, exp)

        exp = Series([ser.iloc[0]] * 3, index=index, name="xxx")
        tm.assert_series_equal(ser.dt.normalize(), exp)

    def test_dt_accessor_limited_display_api(self):
        # tznaive
        ser = Series(date_range("20130101", periods=5, freq="D"), name="xxx")
        results = get_dir(ser)
        tm.assert_almost_equal(results, sorted(set(ok_for_dt + ok_for_dt_methods)))

        # tzaware
        ser = Series(date_range("2015-01-01", "2016-01-01", freq="min"), name="xxx")
        ser = ser.dt.tz_localize("UTC").dt.tz_convert("America/Chicago")
        results = get_dir(ser)
        tm.assert_almost_equal(results, sorted(set(ok_for_dt + ok_for_dt_methods)))

        # Period
        idx = period_range("20130101", periods=5, freq="D", name="xxx").astype(object)
        with tm.assert_produces_warning(FutureWarning, match="Dtype inference"):
            ser = Series(idx)
        results = get_dir(ser)
        tm.assert_almost_equal(
            results, sorted(set(ok_for_period + ok_for_period_methods))
        )

    def test_dt_accessor_ambiguous_freq_conversions(self):
        # GH#11295
        # ambiguous time error on the conversions
        ser = Series(date_range("2015-01-01", "2016-01-01", freq="min"), name="xxx")
        ser = ser.dt.tz_localize("UTC").dt.tz_convert("America/Chicago")

        exp_values = date_range(
            "2015-01-01", "2016-01-01", freq="min", tz="UTC"
        ).tz_convert("America/Chicago")
        # freq not preserved by tz_localize above
        exp_values = exp_values._with_freq(None)
        expected = Series(exp_values, name="xxx")
        tm.assert_series_equal(ser, expected)

    def test_dt_accessor_not_writeable(self, using_copy_on_write, warn_copy_on_write):
        # no setting allowed
        ser = Series(date_range("20130101", periods=5, freq="D"), name="xxx")
        with pytest.raises(ValueError, match="modifications"):
            ser.dt.hour = 5

        # trying to set a copy
        msg = "modifications to a property of a datetimelike.+not supported"
        with pd.option_context("chained_assignment", "raise"):
            if using_copy_on_write:
                with tm.raises_chained_assignment_error():
                    ser.dt.hour[0] = 5
            elif warn_copy_on_write:
                with tm.assert_produces_warning(
                    FutureWarning, match="ChainedAssignmentError"
                ):
                    ser.dt.hour[0] = 5
            else:
                with pytest.raises(SettingWithCopyError, match=msg):
                    ser.dt.hour[0] = 5

    @pytest.mark.parametrize(
        "method, dates",
        [
            ["round", ["2012-01-02", "2012-01-02", "2012-01-01"]],
            ["floor", ["2012-01-01", "2012-01-01", "2012-01-01"]],
            ["ceil", ["2012-01-02", "2012-01-02", "2012-01-02"]],
        ],
    )
    def test_dt_round(self, method, dates):
        # round
        ser = Series(
            pd.to_datetime(
                ["2012-01-01 13:00:00", "2012-01-01 12:01:00", "2012-01-01 08:00:00"]
            ),
            name="xxx",
        )
        result = getattr(ser.dt, method)("D")
        expected = Series(pd.to_datetime(dates), name="xxx")
        tm.assert_series_equal(result, expected)

    def test_dt_round_tz(self):
        ser = Series(
            pd.to_datetime(
                ["2012-01-01 13:00:00", "2012-01-01 12:01:00", "2012-01-01 08:00:00"]
            ),
            name="xxx",
        )
        result = ser.dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.round("D")

        exp_values = pd.to_datetime(
            ["2012-01-01", "2012-01-01", "2012-01-01"]
        ).tz_localize("US/Eastern")
        expected = Series(exp_values, name="xxx")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("method", ["ceil", "round", "floor"])
    def test_dt_round_tz_ambiguous(self, method):
        # GH 18946 round near "fall back" DST
        df1 = DataFrame(
            [
                pd.to_datetime("2017-10-29 02:00:00+02:00", utc=True),
                pd.to_datetime("2017-10-29 02:00:00+01:00", utc=True),
                pd.to_datetime("2017-10-29 03:00:00+01:00", utc=True),
            ],
            columns=["date"],
        )
        df1["date"] = df1["date"].dt.tz_convert("Europe/Madrid")
        # infer
        result = getattr(df1.date.dt, method)("h", ambiguous="infer")
        expected = df1["date"]
        tm.assert_series_equal(result, expected)

        # bool-array
        result = getattr(df1.date.dt, method)("h", ambiguous=[True, False, False])
        tm.assert_series_equal(result, expected)

        # NaT
        result = getattr(df1.date.dt, method)("h", ambiguous="NaT")
        expected = df1["date"].copy()
        expected.iloc[0:2] = pd.NaT
        tm.assert_series_equal(result, expected)

        # raise
        with tm.external_error_raised(pytz.AmbiguousTimeError):
            getattr(df1.date.dt, method)("h", ambiguous="raise")

    @pytest.mark.parametrize(
        "method, ts_str, freq",
        [
            ["ceil", "2018-03-11 01:59:00-0600", "5min"],
            ["round", "2018-03-11 01:59:00-0600", "5min"],
            ["floor", "2018-03-11 03:01:00-0500", "2h"],
        ],
    )
    def test_dt_round_tz_nonexistent(self, method, ts_str, freq):
        # GH 23324 round near "spring forward" DST
        ser = Series([pd.Timestamp(ts_str, tz="America/Chicago")])
        result = getattr(ser.dt, method)(freq, nonexistent="shift_forward")
        expected = Series([pd.Timestamp("2018-03-11 03:00:00", tz="America/Chicago")])
        tm.assert_series_equal(result, expected)

        result = getattr(ser.dt, method)(freq, nonexistent="NaT")
        expected = Series([pd.NaT]).dt.tz_localize(result.dt.tz)
        tm.assert_series_equal(result, expected)

        with pytest.raises(pytz.NonExistentTimeError, match="2018-03-11 02:00:00"):
            getattr(ser.dt, method)(freq, nonexistent="raise")

    @pytest.mark.parametrize("freq", ["ns", "us", "1000us"])
    def test_dt_round_nonnano_higher_resolution_no_op(self, freq):
        # GH 52761
        ser = Series(
            ["2020-05-31 08:00:00", "2000-12-31 04:00:05", "1800-03-14 07:30:20"],
            dtype="datetime64[ms]",
        )
        expected = ser.copy()
        result = ser.dt.round(freq)
        tm.assert_series_equal(result, expected)

        assert not np.shares_memory(ser.array._ndarray, result.array._ndarray)

    def test_dt_namespace_accessor_categorical(self):
        # GH 19468
        dti = DatetimeIndex(["20171111", "20181212"]).repeat(2)
        ser = Series(pd.Categorical(dti), name="foo")
        result = ser.dt.year
        expected = Series([2017, 2017, 2018, 2018], dtype="int32", name="foo")
        tm.assert_series_equal(result, expected)

    def test_dt_tz_localize_categorical(self, tz_aware_fixture):
        # GH 27952
        tz = tz_aware_fixture
        datetimes = Series(
            ["2019-01-01", "2019-01-01", "2019-01-02"], dtype="datetime64[ns]"
        )
        categorical = datetimes.astype("category")
        result = categorical.dt.tz_localize(tz)
        expected = datetimes.dt.tz_localize(tz)
        tm.assert_series_equal(result, expected)

    def test_dt_tz_convert_categorical(self, tz_aware_fixture):
        # GH 27952
        tz = tz_aware_fixture
        datetimes = Series(
            ["2019-01-01", "2019-01-01", "2019-01-02"], dtype="datetime64[ns, MET]"
        )
        categorical = datetimes.astype("category")
        result = categorical.dt.tz_convert(tz)
        expected = datetimes.dt.tz_convert(tz)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("accessor", ["year", "month", "day"])
    def test_dt_other_accessors_categorical(self, accessor):
        # GH 27952
        datetimes = Series(
            ["2018-01-01", "2018-01-01", "2019-01-02"], dtype="datetime64[ns]"
        )
        categorical = datetimes.astype("category")
        result = getattr(categorical.dt, accessor)
        expected = getattr(datetimes.dt, accessor)
        tm.assert_series_equal(result, expected)

    def test_dt_accessor_no_new_attributes(self):
        # https://github.com/pandas-dev/pandas/issues/10673
        ser = Series(date_range("20130101", periods=5, freq="D"))
        with pytest.raises(AttributeError, match="You cannot add any new attribute"):
            ser.dt.xlabel = "a"

    # error: Unsupported operand types for + ("List[None]" and "List[str]")
    @pytest.mark.parametrize(
        "time_locale", [None] + tm.get_locales()  # type: ignore[operator]
    )
    def test_dt_accessor_datetime_name_accessors(self, time_locale):
        # Test Monday -> Sunday and January -> December, in that sequence
        if time_locale is None:
            # If the time_locale is None, day-name and month_name should
            # return the english attributes
            expected_days = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            expected_months = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
        else:
            with tm.set_locale(time_locale, locale.LC_TIME):
                expected_days = calendar.day_name[:]
                expected_months = calendar.month_name[1:]

        ser = Series(date_range(freq="D", start=datetime(1998, 1, 1), periods=365))
        english_days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        for day, name, eng_name in zip(range(4, 11), expected_days, english_days):
            name = name.capitalize()
            assert ser.dt.day_name(locale=time_locale)[day] == name
            assert ser.dt.day_name(locale=None)[day] == eng_name
        ser = pd.concat([ser, Series([pd.NaT])])
        assert np.isnan(ser.dt.day_name(locale=time_locale).iloc[-1])

        ser = Series(date_range(freq="ME", start="2012", end="2013"))
        result = ser.dt.month_name(locale=time_locale)
        expected = Series([month.capitalize() for month in expected_months])

        # work around https://github.com/pandas-dev/pandas/issues/22342
        result = result.str.normalize("NFD")
        expected = expected.str.normalize("NFD")

        tm.assert_series_equal(result, expected)

        for s_date, expected in zip(ser, expected_months):
            result = s_date.month_name(locale=time_locale)
            expected = expected.capitalize()

            result = unicodedata.normalize("NFD", result)
            expected = unicodedata.normalize("NFD", expected)

            assert result == expected

        ser = pd.concat([ser, Series([pd.NaT])])
        assert np.isnan(ser.dt.month_name(locale=time_locale).iloc[-1])

    def test_strftime(self):
        # GH 10086
        ser = Series(date_range("20130101", periods=5))
        result = ser.dt.strftime("%Y/%m/%d")
        expected = Series(
            ["2013/01/01", "2013/01/02", "2013/01/03", "2013/01/04", "2013/01/05"]
        )
        tm.assert_series_equal(result, expected)

        ser = Series(date_range("2015-02-03 11:22:33.4567", periods=5))
        result = ser.dt.strftime("%Y/%m/%d %H-%M-%S")
        expected = Series(
            [
                "2015/02/03 11-22-33",
                "2015/02/04 11-22-33",
                "2015/02/05 11-22-33",
                "2015/02/06 11-22-33",
                "2015/02/07 11-22-33",
            ]
        )
        tm.assert_series_equal(result, expected)

        ser = Series(period_range("20130101", periods=5))
        result = ser.dt.strftime("%Y/%m/%d")
        expected = Series(
            ["2013/01/01", "2013/01/02", "2013/01/03", "2013/01/04", "2013/01/05"]
        )
        tm.assert_series_equal(result, expected)

        ser = Series(period_range("2015-02-03 11:22:33.4567", periods=5, freq="s"))
        result = ser.dt.strftime("%Y/%m/%d %H-%M-%S")
        expected = Series(
            [
                "2015/02/03 11-22-33",
                "2015/02/03 11-22-34",
                "2015/02/03 11-22-35",
                "2015/02/03 11-22-36",
                "2015/02/03 11-22-37",
            ]
        )
        tm.assert_series_equal(result, expected)

    def test_strftime_dt64_days(self):
        ser = Series(date_range("20130101", periods=5))
        ser.iloc[0] = pd.NaT
        result = ser.dt.strftime("%Y/%m/%d")
        expected = Series(
            [np.nan, "2013/01/02", "2013/01/03", "2013/01/04", "2013/01/05"]
        )
        tm.assert_series_equal(result, expected)

        datetime_index = date_range("20150301", periods=5)
        result = datetime_index.strftime("%Y/%m/%d")

        expected = Index(
            ["2015/03/01", "2015/03/02", "2015/03/03", "2015/03/04", "2015/03/05"],
        )
        # dtype may be S10 or U10 depending on python version
        tm.assert_index_equal(result, expected)

    def test_strftime_period_days(self, using_infer_string):
        period_index = period_range("20150301", periods=5)
        result = period_index.strftime("%Y/%m/%d")
        expected = Index(
            ["2015/03/01", "2015/03/02", "2015/03/03", "2015/03/04", "2015/03/05"],
            dtype="=U10",
        )
        if using_infer_string:
            expected = expected.astype(StringDtype(na_value=np.nan))
        tm.assert_index_equal(result, expected)

    def test_strftime_dt64_microsecond_resolution(self):
        ser = Series([datetime(2013, 1, 1, 2, 32, 59), datetime(2013, 1, 2, 14, 32, 1)])
        result = ser.dt.strftime("%Y-%m-%d %H:%M:%S")
        expected = Series(["2013-01-01 02:32:59", "2013-01-02 14:32:01"])
        tm.assert_series_equal(result, expected)

    def test_strftime_period_hours(self):
        ser = Series(period_range("20130101", periods=4, freq="h"))
        result = ser.dt.strftime("%Y/%m/%d %H:%M:%S")
        expected = Series(
            [
                "2013/01/01 00:00:00",
                "2013/01/01 01:00:00",
                "2013/01/01 02:00:00",
                "2013/01/01 03:00:00",
            ]
        )
        tm.assert_series_equal(result, expected)

    def test_strftime_period_minutes(self):
        ser = Series(period_range("20130101", periods=4, freq="ms"))
        result = ser.dt.strftime("%Y/%m/%d %H:%M:%S.%l")
        expected = Series(
            [
                "2013/01/01 00:00:00.000",
                "2013/01/01 00:00:00.001",
                "2013/01/01 00:00:00.002",
                "2013/01/01 00:00:00.003",
            ]
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "data",
        [
            DatetimeIndex(["2019-01-01", pd.NaT]),
            PeriodIndex(["2019-01-01", pd.NaT], dtype="period[D]"),
        ],
    )
    def test_strftime_nat(self, data):
        # GH 29578
        ser = Series(data)
        result = ser.dt.strftime("%Y-%m-%d")
        expected = Series(["2019-01-01", np.nan])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "data", [DatetimeIndex([pd.NaT]), PeriodIndex([pd.NaT], dtype="period[D]")]
    )
    def test_strftime_all_nat(self, data):
        # https://github.com/pandas-dev/pandas/issues/45858
        ser = Series(data)
        with tm.assert_produces_warning(None):
            result = ser.dt.strftime("%Y-%m-%d")
        expected = Series([np.nan], dtype="str")
        tm.assert_series_equal(result, expected)

    def test_valid_dt_with_missing_values(self):
        # GH 8689
        ser = Series(date_range("20130101", periods=5, freq="D"))
        ser.iloc[2] = pd.NaT

        for attr in ["microsecond", "nanosecond", "second", "minute", "hour", "day"]:
            expected = getattr(ser.dt, attr).copy()
            expected.iloc[2] = np.nan
            result = getattr(ser.dt, attr)
            tm.assert_series_equal(result, expected)

        result = ser.dt.date
        expected = Series(
            [
                date(2013, 1, 1),
                date(2013, 1, 2),
                pd.NaT,
                date(2013, 1, 4),
                date(2013, 1, 5),
            ],
            dtype="object",
        )
        tm.assert_series_equal(result, expected)

        result = ser.dt.time
        expected = Series([time(0), time(0), pd.NaT, time(0), time(0)], dtype="object")
        tm.assert_series_equal(result, expected)

    def test_dt_accessor_api(self):
        # GH 9322
        from pandas.core.indexes.accessors import (
            CombinedDatetimelikeProperties,
            DatetimeProperties,
        )

        assert Series.dt is CombinedDatetimelikeProperties

        ser = Series(date_range("2000-01-01", periods=3))
        assert isinstance(ser.dt, DatetimeProperties)

    @pytest.mark.parametrize(
        "ser",
        [
            Series(np.arange(5)),
            Series(list("abcde")),
            Series(np.random.default_rng(2).standard_normal(5)),
        ],
    )
    def test_dt_accessor_invalid(self, ser):
        # GH#9322 check that series with incorrect dtypes don't have attr
        with pytest.raises(AttributeError, match="only use .dt accessor"):
            ser.dt
        assert not hasattr(ser, "dt")

    def test_dt_accessor_updates_on_inplace(self):
        ser = Series(date_range("2018-01-01", periods=10))
        ser[2] = None
        return_value = ser.fillna(pd.Timestamp("2018-01-01"), inplace=True)
        assert return_value is None
        result = ser.dt.date
        assert result[0] == result[2]

    def test_date_tz(self):
        # GH11757
        rng = DatetimeIndex(
            ["2014-04-04 23:56", "2014-07-18 21:24", "2015-11-22 22:14"],
            tz="US/Eastern",
        )
        ser = Series(rng)
        expected = Series([date(2014, 4, 4), date(2014, 7, 18), date(2015, 11, 22)])
        tm.assert_series_equal(ser.dt.date, expected)
        tm.assert_series_equal(ser.apply(lambda x: x.date()), expected)

    def test_dt_timetz_accessor(self, tz_naive_fixture):
        # GH21358
        tz = maybe_get_tz(tz_naive_fixture)

        dtindex = DatetimeIndex(
            ["2014-04-04 23:56", "2014-07-18 21:24", "2015-11-22 22:14"], tz=tz
        )
        ser = Series(dtindex)
        expected = Series(
            [time(23, 56, tzinfo=tz), time(21, 24, tzinfo=tz), time(22, 14, tzinfo=tz)]
        )
        result = ser.dt.timetz
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "input_series, expected_output",
        [
            [["2020-01-01"], [[2020, 1, 3]]],
            [[pd.NaT], [[np.nan, np.nan, np.nan]]],
            [["2019-12-31", "2019-12-29"], [[2020, 1, 2], [2019, 52, 7]]],
            [["2010-01-01", pd.NaT], [[2009, 53, 5], [np.nan, np.nan, np.nan]]],
            # see GH#36032
            [["2016-01-08", "2016-01-04"], [[2016, 1, 5], [2016, 1, 1]]],
            [["2016-01-07", "2016-01-01"], [[2016, 1, 4], [2015, 53, 5]]],
        ],
    )
    def test_isocalendar(self, input_series, expected_output):
        result = pd.to_datetime(Series(input_series)).dt.isocalendar()
        expected_frame = DataFrame(
            expected_output, columns=["year", "week", "day"], dtype="UInt32"
        )
        tm.assert_frame_equal(result, expected_frame)

    def test_hour_index(self):
        dt_series = Series(
            date_range(start="2021-01-01", periods=5, freq="h"),
            index=[2, 6, 7, 8, 11],
            dtype="category",
        )
        result = dt_series.dt.hour
        expected = Series(
            [0, 1, 2, 3, 4],
            dtype="int32",
            index=[2, 6, 7, 8, 11],
        )
        tm.assert_series_equal(result, expected)


class TestSeriesPeriodValuesDtAccessor:
    @pytest.mark.parametrize(
        "input_vals",
        [
            [Period("2016-01", freq="M"), Period("2016-02", freq="M")],
            [Period("2016-01-01", freq="D"), Period("2016-01-02", freq="D")],
            [
                Period("2016-01-01 00:00:00", freq="h"),
                Period("2016-01-01 01:00:00", freq="h"),
            ],
            [
                Period("2016-01-01 00:00:00", freq="M"),
                Period("2016-01-01 00:01:00", freq="M"),
            ],
            [
                Period("2016-01-01 00:00:00", freq="s"),
                Period("2016-01-01 00:00:01", freq="s"),
            ],
        ],
    )
    def test_end_time_timevalues(self, input_vals):
        # GH#17157
        # Check that the time part of the Period is adjusted by end_time
        # when using the dt accessor on a Series
        input_vals = PeriodArray._from_sequence(np.asarray(input_vals))

        ser = Series(input_vals)
        result = ser.dt.end_time
        expected = ser.apply(lambda x: x.end_time)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("input_vals", [("2001"), ("NaT")])
    def test_to_period(self, input_vals):
        # GH#21205
        expected = Series([input_vals], dtype="Period[D]")
        result = Series([input_vals], dtype="datetime64[ns]").dt.to_period("D")
        tm.assert_series_equal(result, expected)


def test_normalize_pre_epoch_dates():
    # GH: 36294
    ser = pd.to_datetime(Series(["1969-01-01 09:00:00", "2016-01-01 09:00:00"]))
    result = ser.dt.normalize()
    expected = pd.to_datetime(Series(["1969-01-01", "2016-01-01"]))
    tm.assert_series_equal(result, expected)


def test_day_attribute_non_nano_beyond_int32():
    # GH 52386
    data = np.array(
        [
            136457654736252,
            134736784364431,
            245345345545332,
            223432411,
            2343241,
            3634548734,
            23234,
        ],
        dtype="timedelta64[s]",
    )
    ser = Series(data)
    result = ser.dt.days
    expected = Series([1579371003, 1559453522, 2839645203, 2586, 27, 42066, 0])
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\test_datetimes.py ===
"""
Tests for DatetimeArray
"""
from __future__ import annotations

from datetime import timedelta
import operator

try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Cannot assign to a type
    ZoneInfo = None  # type: ignore[misc, assignment]

import numpy as np
import pytest

from pandas._libs.tslibs import tz_compare

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    TimedeltaArray,
)


class TestNonNano:
    @pytest.fixture(params=["s", "ms", "us"])
    def unit(self, request):
        """Fixture returning parametrized time units"""
        return request.param

    @pytest.fixture
    def dtype(self, unit, tz_naive_fixture):
        tz = tz_naive_fixture
        if tz is None:
            return np.dtype(f"datetime64[{unit}]")
        else:
            return DatetimeTZDtype(unit=unit, tz=tz)

    @pytest.fixture
    def dta_dti(self, unit, dtype):
        tz = getattr(dtype, "tz", None)

        dti = pd.date_range("2016-01-01", periods=55, freq="D", tz=tz)
        if tz is None:
            arr = np.asarray(dti).astype(f"M8[{unit}]")
        else:
            arr = np.asarray(dti.tz_convert("UTC").tz_localize(None)).astype(
                f"M8[{unit}]"
            )

        dta = DatetimeArray._simple_new(arr, dtype=dtype)
        return dta, dti

    @pytest.fixture
    def dta(self, dta_dti):
        dta, dti = dta_dti
        return dta

    def test_non_nano(self, unit, dtype):
        arr = np.arange(5, dtype=np.int64).view(f"M8[{unit}]")
        dta = DatetimeArray._simple_new(arr, dtype=dtype)

        assert dta.dtype == dtype
        assert dta[0].unit == unit
        assert tz_compare(dta.tz, dta[0].tz)
        assert (dta[0] == dta[:1]).all()

    @pytest.mark.parametrize(
        "field", DatetimeArray._field_ops + DatetimeArray._bool_ops
    )
    def test_fields(self, unit, field, dtype, dta_dti):
        dta, dti = dta_dti

        assert (dti == dta).all()

        res = getattr(dta, field)
        expected = getattr(dti._data, field)
        tm.assert_numpy_array_equal(res, expected)

    def test_normalize(self, unit):
        dti = pd.date_range("2016-01-01 06:00:00", periods=55, freq="D")
        arr = np.asarray(dti).astype(f"M8[{unit}]")

        dta = DatetimeArray._simple_new(arr, dtype=arr.dtype)

        assert not dta.is_normalized

        # TODO: simplify once we can just .astype to other unit
        exp = np.asarray(dti.normalize()).astype(f"M8[{unit}]")
        expected = DatetimeArray._simple_new(exp, dtype=exp.dtype)

        res = dta.normalize()
        tm.assert_extension_array_equal(res, expected)

    def test_simple_new_requires_match(self, unit):
        arr = np.arange(5, dtype=np.int64).view(f"M8[{unit}]")
        dtype = DatetimeTZDtype(unit, "UTC")

        dta = DatetimeArray._simple_new(arr, dtype=dtype)
        assert dta.dtype == dtype

        wrong = DatetimeTZDtype("ns", "UTC")
        with pytest.raises(AssertionError, match=""):
            DatetimeArray._simple_new(arr, dtype=wrong)

    def test_std_non_nano(self, unit):
        dti = pd.date_range("2016-01-01", periods=55, freq="D")
        arr = np.asarray(dti).astype(f"M8[{unit}]")

        dta = DatetimeArray._simple_new(arr, dtype=arr.dtype)

        # we should match the nano-reso std, but floored to our reso.
        res = dta.std()
        assert res._creso == dta._creso
        assert res == dti.std().floor(unit)

    @pytest.mark.filterwarnings("ignore:Converting to PeriodArray.*:UserWarning")
    def test_to_period(self, dta_dti):
        dta, dti = dta_dti
        result = dta.to_period("D")
        expected = dti._data.to_period("D")

        tm.assert_extension_array_equal(result, expected)

    def test_iter(self, dta):
        res = next(iter(dta))
        expected = dta[0]

        assert type(res) is pd.Timestamp
        assert res._value == expected._value
        assert res._creso == expected._creso
        assert res == expected

    def test_astype_object(self, dta):
        result = dta.astype(object)
        assert all(x._creso == dta._creso for x in result)
        assert all(x == y for x, y in zip(result, dta))

    def test_to_pydatetime(self, dta_dti):
        dta, dti = dta_dti

        result = dta.to_pydatetime()
        expected = dti.to_pydatetime()
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("meth", ["time", "timetz", "date"])
    def test_time_date(self, dta_dti, meth):
        dta, dti = dta_dti

        result = getattr(dta, meth)
        expected = getattr(dti, meth)
        tm.assert_numpy_array_equal(result, expected)

    def test_format_native_types(self, unit, dtype, dta_dti):
        # In this case we should get the same formatted values with our nano
        #  version dti._data as we do with the non-nano dta
        dta, dti = dta_dti

        res = dta._format_native_types()
        exp = dti._data._format_native_types()
        tm.assert_numpy_array_equal(res, exp)

    def test_repr(self, dta_dti, unit):
        dta, dti = dta_dti

        assert repr(dta) == repr(dti._data).replace("[ns", f"[{unit}")

    # TODO: tests with td64
    def test_compare_mismatched_resolutions(self, comparison_op):
        # comparison that numpy gets wrong bc of silent overflows
        op = comparison_op

        iinfo = np.iinfo(np.int64)
        vals = np.array([iinfo.min, iinfo.min + 1, iinfo.max], dtype=np.int64)

        # Construct so that arr2[1] < arr[1] < arr[2] < arr2[2]
        arr = np.array(vals).view("M8[ns]")
        arr2 = arr.view("M8[s]")

        left = DatetimeArray._simple_new(arr, dtype=arr.dtype)
        right = DatetimeArray._simple_new(arr2, dtype=arr2.dtype)

        if comparison_op is operator.eq:
            expected = np.array([False, False, False])
        elif comparison_op is operator.ne:
            expected = np.array([True, True, True])
        elif comparison_op in [operator.lt, operator.le]:
            expected = np.array([False, False, True])
        else:
            expected = np.array([False, True, False])

        result = op(left, right)
        tm.assert_numpy_array_equal(result, expected)

        result = op(left[1], right)
        tm.assert_numpy_array_equal(result, expected)

        if op not in [operator.eq, operator.ne]:
            # check that numpy still gets this wrong; if it is fixed we may be
            #  able to remove compare_mismatched_resolutions
            np_res = op(left._ndarray, right._ndarray)
            tm.assert_numpy_array_equal(np_res[1:], ~expected[1:])

    def test_add_mismatched_reso_doesnt_downcast(self):
        # https://github.com/pandas-dev/pandas/pull/48748#issuecomment-1260181008
        td = pd.Timedelta(microseconds=1)
        dti = pd.date_range("2016-01-01", periods=3) - td
        dta = dti._data.as_unit("us")

        res = dta + td.as_unit("us")
        # even though the result is an even number of days
        #  (so we _could_ downcast to unit="s"), we do not.
        assert res.unit == "us"

    @pytest.mark.parametrize(
        "scalar",
        [
            timedelta(hours=2),
            pd.Timedelta(hours=2),
            np.timedelta64(2, "h"),
            np.timedelta64(2 * 3600 * 1000, "ms"),
            pd.offsets.Minute(120),
            pd.offsets.Hour(2),
        ],
    )
    def test_add_timedeltalike_scalar_mismatched_reso(self, dta_dti, scalar):
        dta, dti = dta_dti

        td = pd.Timedelta(scalar)
        exp_unit = tm.get_finest_unit(dta.unit, td.unit)

        expected = (dti + td)._data.as_unit(exp_unit)
        result = dta + scalar
        tm.assert_extension_array_equal(result, expected)

        result = scalar + dta
        tm.assert_extension_array_equal(result, expected)

        expected = (dti - td)._data.as_unit(exp_unit)
        result = dta - scalar
        tm.assert_extension_array_equal(result, expected)

    def test_sub_datetimelike_scalar_mismatch(self):
        dti = pd.date_range("2016-01-01", periods=3)
        dta = dti._data.as_unit("us")

        ts = dta[0].as_unit("s")

        result = dta - ts
        expected = (dti - dti[0])._data.as_unit("us")
        assert result.dtype == "m8[us]"
        tm.assert_extension_array_equal(result, expected)

    def test_sub_datetime64_reso_mismatch(self):
        dti = pd.date_range("2016-01-01", periods=3)
        left = dti._data.as_unit("s")
        right = left.as_unit("ms")

        result = left - right
        exp_values = np.array([0, 0, 0], dtype="m8[ms]")
        expected = TimedeltaArray._simple_new(
            exp_values,
            dtype=exp_values.dtype,
        )
        tm.assert_extension_array_equal(result, expected)
        result2 = right - left
        tm.assert_extension_array_equal(result2, expected)


class TestDatetimeArrayComparisons:
    # TODO: merge this into tests/arithmetic/test_datetime64 once it is
    #  sufficiently robust

    def test_cmp_dt64_arraylike_tznaive(self, comparison_op):
        # arbitrary tz-naive DatetimeIndex
        op = comparison_op

        dti = pd.date_range("2016-01-1", freq="MS", periods=9, tz=None)
        arr = dti._data
        assert arr.freq == dti.freq
        assert arr.tz == dti.tz

        right = dti

        expected = np.ones(len(arr), dtype=bool)
        if comparison_op.__name__ in ["ne", "gt", "lt"]:
            # for these the comparisons should be all-False
            expected = ~expected

        result = op(arr, arr)
        tm.assert_numpy_array_equal(result, expected)
        for other in [
            right,
            np.array(right),
            list(right),
            tuple(right),
            right.astype(object),
        ]:
            result = op(arr, other)
            tm.assert_numpy_array_equal(result, expected)

            result = op(other, arr)
            tm.assert_numpy_array_equal(result, expected)


class TestDatetimeArray:
    def test_astype_ns_to_ms_near_bounds(self):
        # GH#55979
        ts = pd.Timestamp("1677-09-21 00:12:43.145225")
        target = ts.as_unit("ms")

        dta = DatetimeArray._from_sequence([ts], dtype="M8[ns]")
        assert (dta.view("i8") == ts.as_unit("ns").value).all()

        result = dta.astype("M8[ms]")
        assert result[0] == target

        expected = DatetimeArray._from_sequence([ts], dtype="M8[ms]")
        assert (expected.view("i8") == target._value).all()

        tm.assert_datetime_array_equal(result, expected)

    def test_astype_non_nano_tznaive(self):
        dti = pd.date_range("2016-01-01", periods=3)

        res = dti.astype("M8[s]")
        assert res.dtype == "M8[s]"

        dta = dti._data
        res = dta.astype("M8[s]")
        assert res.dtype == "M8[s]"
        assert isinstance(res, pd.core.arrays.DatetimeArray)  # used to be ndarray

    def test_astype_non_nano_tzaware(self):
        dti = pd.date_range("2016-01-01", periods=3, tz="UTC")

        res = dti.astype("M8[s, US/Pacific]")
        assert res.dtype == "M8[s, US/Pacific]"

        dta = dti._data
        res = dta.astype("M8[s, US/Pacific]")
        assert res.dtype == "M8[s, US/Pacific]"

        # from non-nano to non-nano, preserving reso
        res2 = res.astype("M8[s, UTC]")
        assert res2.dtype == "M8[s, UTC]"
        assert not tm.shares_memory(res2, res)

        res3 = res.astype("M8[s, UTC]", copy=False)
        assert res2.dtype == "M8[s, UTC]"
        assert tm.shares_memory(res3, res)

    def test_astype_to_same(self):
        arr = DatetimeArray._from_sequence(
            ["2000"], dtype=DatetimeTZDtype(tz="US/Central")
        )
        result = arr.astype(DatetimeTZDtype(tz="US/Central"), copy=False)
        assert result is arr

    @pytest.mark.parametrize("dtype", ["datetime64[ns]", "datetime64[ns, UTC]"])
    @pytest.mark.parametrize(
        "other", ["datetime64[ns]", "datetime64[ns, UTC]", "datetime64[ns, CET]"]
    )
    def test_astype_copies(self, dtype, other):
        # https://github.com/pandas-dev/pandas/pull/32490
        ser = pd.Series([1, 2], dtype=dtype)
        orig = ser.copy()

        err = False
        if (dtype == "datetime64[ns]") ^ (other == "datetime64[ns]"):
            # deprecated in favor of tz_localize
            err = True

        if err:
            if dtype == "datetime64[ns]":
                msg = "Use obj.tz_localize instead or series.dt.tz_localize instead"
            else:
                msg = "from timezone-aware dtype to timezone-naive dtype"
            with pytest.raises(TypeError, match=msg):
                ser.astype(other)
        else:
            t = ser.astype(other)
            t[:] = pd.NaT
            tm.assert_series_equal(ser, orig)

    @pytest.mark.parametrize("dtype", [int, np.int32, np.int64, "uint32", "uint64"])
    def test_astype_int(self, dtype):
        arr = DatetimeArray._from_sequence(
            [pd.Timestamp("2000"), pd.Timestamp("2001")], dtype="M8[ns]"
        )

        if np.dtype(dtype) != np.int64:
            with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
                arr.astype(dtype)
            return

        result = arr.astype(dtype)
        expected = arr._ndarray.view("i8")
        tm.assert_numpy_array_equal(result, expected)

    def test_astype_to_sparse_dt64(self):
        # GH#50082
        dti = pd.date_range("2016-01-01", periods=4)
        dta = dti._data
        result = dta.astype("Sparse[datetime64[ns]]")

        assert result.dtype == "Sparse[datetime64[ns]]"
        assert (result == dta).all()

    def test_tz_setter_raises(self):
        arr = DatetimeArray._from_sequence(
            ["2000"], dtype=DatetimeTZDtype(tz="US/Central")
        )
        with pytest.raises(AttributeError, match="tz_localize"):
            arr.tz = "UTC"

    def test_setitem_str_impute_tz(self, tz_naive_fixture):
        # Like for getitem, if we are passed a naive-like string, we impute
        #  our own timezone.
        tz = tz_naive_fixture

        data = np.array([1, 2, 3], dtype="M8[ns]")
        dtype = data.dtype if tz is None else DatetimeTZDtype(tz=tz)
        arr = DatetimeArray._from_sequence(data, dtype=dtype)
        expected = arr.copy()

        ts = pd.Timestamp("2020-09-08 16:50").tz_localize(tz)
        setter = str(ts.tz_localize(None))

        # Setting a scalar tznaive string
        expected[0] = ts
        arr[0] = setter
        tm.assert_equal(arr, expected)

        # Setting a listlike of tznaive strings
        expected[1] = ts
        arr[:2] = [setter, setter]
        tm.assert_equal(arr, expected)

    def test_setitem_different_tz_raises(self):
        # pre-2.0 we required exact tz match, in 2.0 we require only
        #  tzawareness-match
        data = np.array([1, 2, 3], dtype="M8[ns]")
        arr = DatetimeArray._from_sequence(
            data, copy=False, dtype=DatetimeTZDtype(tz="US/Central")
        )
        with pytest.raises(TypeError, match="Cannot compare tz-naive and tz-aware"):
            arr[0] = pd.Timestamp("2000")

        ts = pd.Timestamp("2000", tz="US/Eastern")
        arr[0] = ts
        assert arr[0] == ts.tz_convert("US/Central")

    def test_setitem_clears_freq(self):
        a = pd.date_range("2000", periods=2, freq="D", tz="US/Central")._data
        a[0] = pd.Timestamp("2000", tz="US/Central")
        assert a.freq is None

    @pytest.mark.parametrize(
        "obj",
        [
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-01-01").to_datetime64(),
            pd.Timestamp("2021-01-01").to_pydatetime(),
        ],
    )
    def test_setitem_objects(self, obj):
        # make sure we accept datetime64 and datetime in addition to Timestamp
        dti = pd.date_range("2000", periods=2, freq="D")
        arr = dti._data

        arr[0] = obj
        assert arr[0] == obj

    def test_repeat_preserves_tz(self):
        dti = pd.date_range("2000", periods=2, freq="D", tz="US/Central")
        arr = dti._data

        repeated = arr.repeat([1, 1])

        # preserves tz and values, but not freq
        expected = DatetimeArray._from_sequence(arr.asi8, dtype=arr.dtype)
        tm.assert_equal(repeated, expected)

    def test_value_counts_preserves_tz(self):
        dti = pd.date_range("2000", periods=2, freq="D", tz="US/Central")
        arr = dti._data.repeat([4, 3])

        result = arr.value_counts()

        # Note: not tm.assert_index_equal, since `freq`s do not match
        assert result.index.equals(dti)

        arr[-2] = pd.NaT
        result = arr.value_counts(dropna=False)
        expected = pd.Series([4, 2, 1], index=[dti[0], dti[1], pd.NaT], name="count")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("method", ["pad", "backfill"])
    def test_fillna_preserves_tz(self, method):
        dti = pd.date_range("2000-01-01", periods=5, freq="D", tz="US/Central")
        arr = DatetimeArray._from_sequence(dti, copy=True)
        arr[2] = pd.NaT

        fill_val = dti[1] if method == "pad" else dti[3]
        expected = DatetimeArray._from_sequence(
            [dti[0], dti[1], fill_val, dti[3], dti[4]],
            dtype=DatetimeTZDtype(tz="US/Central"),
        )

        result = arr._pad_or_backfill(method=method)
        tm.assert_extension_array_equal(result, expected)

        # assert that arr and dti were not modified in-place
        assert arr[2] is pd.NaT
        assert dti[2] == pd.Timestamp("2000-01-03", tz="US/Central")

    def test_fillna_2d(self):
        dti = pd.date_range("2016-01-01", periods=6, tz="US/Pacific")
        dta = dti._data.reshape(3, 2).copy()
        dta[0, 1] = pd.NaT
        dta[1, 0] = pd.NaT

        res1 = dta._pad_or_backfill(method="pad")
        expected1 = dta.copy()
        expected1[1, 0] = dta[0, 0]
        tm.assert_extension_array_equal(res1, expected1)

        res2 = dta._pad_or_backfill(method="backfill")
        expected2 = dta.copy()
        expected2 = dta.copy()
        expected2[1, 0] = dta[2, 0]
        expected2[0, 1] = dta[1, 1]
        tm.assert_extension_array_equal(res2, expected2)

        # with different ordering for underlying ndarray; behavior should
        #  be unchanged
        dta2 = dta._from_backing_data(dta._ndarray.copy(order="F"))
        assert dta2._ndarray.flags["F_CONTIGUOUS"]
        assert not dta2._ndarray.flags["C_CONTIGUOUS"]
        tm.assert_extension_array_equal(dta, dta2)

        res3 = dta2._pad_or_backfill(method="pad")
        tm.assert_extension_array_equal(res3, expected1)

        res4 = dta2._pad_or_backfill(method="backfill")
        tm.assert_extension_array_equal(res4, expected2)

        # test the DataFrame method while we're here
        df = pd.DataFrame(dta)
        res = df.ffill()
        expected = pd.DataFrame(expected1)
        tm.assert_frame_equal(res, expected)

        res = df.bfill()
        expected = pd.DataFrame(expected2)
        tm.assert_frame_equal(res, expected)

    def test_array_interface_tz(self):
        tz = "US/Central"
        data = pd.date_range("2017", periods=2, tz=tz)._data
        result = np.asarray(data)

        expected = np.array(
            [
                pd.Timestamp("2017-01-01T00:00:00", tz=tz),
                pd.Timestamp("2017-01-02T00:00:00", tz=tz),
            ],
            dtype=object,
        )
        tm.assert_numpy_array_equal(result, expected)

        result = np.asarray(data, dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        result = np.asarray(data, dtype="M8[ns]")

        expected = np.array(
            ["2017-01-01T06:00:00", "2017-01-02T06:00:00"], dtype="M8[ns]"
        )
        tm.assert_numpy_array_equal(result, expected)

    def test_array_interface(self):
        data = pd.date_range("2017", periods=2)._data
        expected = np.array(
            ["2017-01-01T00:00:00", "2017-01-02T00:00:00"], dtype="datetime64[ns]"
        )

        result = np.asarray(data)
        tm.assert_numpy_array_equal(result, expected)

        result = np.asarray(data, dtype=object)
        expected = np.array(
            [pd.Timestamp("2017-01-01T00:00:00"), pd.Timestamp("2017-01-02T00:00:00")],
            dtype=object,
        )
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("index", [True, False])
    def test_searchsorted_different_tz(self, index):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        arr = pd.DatetimeIndex(data, freq="D")._data.tz_localize("Asia/Tokyo")
        if index:
            arr = pd.Index(arr)

        expected = arr.searchsorted(arr[2])
        result = arr.searchsorted(arr[2].tz_convert("UTC"))
        assert result == expected

        expected = arr.searchsorted(arr[2:6])
        result = arr.searchsorted(arr[2:6].tz_convert("UTC"))
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("index", [True, False])
    def test_searchsorted_tzawareness_compat(self, index):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        arr = pd.DatetimeIndex(data, freq="D")._data
        if index:
            arr = pd.Index(arr)

        mismatch = arr.tz_localize("Asia/Tokyo")

        msg = "Cannot compare tz-naive and tz-aware datetime-like objects"
        with pytest.raises(TypeError, match=msg):
            arr.searchsorted(mismatch[0])
        with pytest.raises(TypeError, match=msg):
            arr.searchsorted(mismatch)

        with pytest.raises(TypeError, match=msg):
            mismatch.searchsorted(arr[0])
        with pytest.raises(TypeError, match=msg):
            mismatch.searchsorted(arr)

    @pytest.mark.parametrize(
        "other",
        [
            1,
            np.int64(1),
            1.0,
            np.timedelta64("NaT"),
            pd.Timedelta(days=2),
            "invalid",
            np.arange(10, dtype="i8") * 24 * 3600 * 10**9,
            np.arange(10).view("timedelta64[ns]") * 24 * 3600 * 10**9,
            pd.Timestamp("2021-01-01").to_period("D"),
        ],
    )
    @pytest.mark.parametrize("index", [True, False])
    def test_searchsorted_invalid_types(self, other, index):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        arr = pd.DatetimeIndex(data, freq="D")._data
        if index:
            arr = pd.Index(arr)

        msg = "|".join(
            [
                "searchsorted requires compatible dtype or scalar",
                "value should be a 'Timestamp', 'NaT', or array of those. Got",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            arr.searchsorted(other)

    def test_shift_fill_value(self):
        dti = pd.date_range("2016-01-01", periods=3)

        dta = dti._data
        expected = DatetimeArray._from_sequence(np.roll(dta._ndarray, 1))

        fv = dta[-1]
        for fill_value in [fv, fv.to_pydatetime(), fv.to_datetime64()]:
            result = dta.shift(1, fill_value=fill_value)
            tm.assert_datetime_array_equal(result, expected)

        dta = dta.tz_localize("UTC")
        expected = expected.tz_localize("UTC")
        fv = dta[-1]
        for fill_value in [fv, fv.to_pydatetime()]:
            result = dta.shift(1, fill_value=fill_value)
            tm.assert_datetime_array_equal(result, expected)

    def test_shift_value_tzawareness_mismatch(self):
        dti = pd.date_range("2016-01-01", periods=3)

        dta = dti._data

        fv = dta[-1].tz_localize("UTC")
        for invalid in [fv, fv.to_pydatetime()]:
            with pytest.raises(TypeError, match="Cannot compare"):
                dta.shift(1, fill_value=invalid)

        dta = dta.tz_localize("UTC")
        fv = dta[-1].tz_localize(None)
        for invalid in [fv, fv.to_pydatetime(), fv.to_datetime64()]:
            with pytest.raises(TypeError, match="Cannot compare"):
                dta.shift(1, fill_value=invalid)

    def test_shift_requires_tzmatch(self):
        # pre-2.0 we required exact tz match, in 2.0 we require just
        #  matching tzawareness
        dti = pd.date_range("2016-01-01", periods=3, tz="UTC")
        dta = dti._data

        fill_value = pd.Timestamp("2020-10-18 18:44", tz="US/Pacific")

        result = dta.shift(1, fill_value=fill_value)
        expected = dta.shift(1, fill_value=fill_value.tz_convert("UTC"))
        tm.assert_equal(result, expected)

    def test_tz_localize_t2d(self):
        dti = pd.date_range("1994-05-12", periods=12, tz="US/Pacific")
        dta = dti._data.reshape(3, 4)
        result = dta.tz_localize(None)

        expected = dta.ravel().tz_localize(None).reshape(dta.shape)
        tm.assert_datetime_array_equal(result, expected)

        roundtrip = expected.tz_localize("US/Pacific")
        tm.assert_datetime_array_equal(roundtrip, dta)

    easts = ["US/Eastern", "dateutil/US/Eastern"]
    if ZoneInfo is not None:
        try:
            tz = ZoneInfo("US/Eastern")
        except KeyError:
            # no tzdata
            pass
        else:
            # Argument 1 to "append" of "list" has incompatible type "ZoneInfo";
            # expected "str"
            easts.append(tz)  # type: ignore[arg-type]

    @pytest.mark.parametrize("tz", easts)
    def test_iter_zoneinfo_fold(self, tz):
        # GH#49684
        utc_vals = np.array(
            [1320552000, 1320555600, 1320559200, 1320562800], dtype=np.int64
        )
        utc_vals *= 1_000_000_000

        dta = DatetimeArray._from_sequence(utc_vals).tz_localize("UTC").tz_convert(tz)

        left = dta[2]
        right = list(dta)[2]
        assert str(left) == str(right)
        # previously there was a bug where with non-pytz right would be
        #  Timestamp('2011-11-06 01:00:00-0400', tz='US/Eastern')
        # while left would be
        #  Timestamp('2011-11-06 01:00:00-0500', tz='US/Eastern')
        # The .value's would match (so they would compare as equal),
        #  but the folds would not
        assert left.utcoffset() == right.utcoffset()

        # The same bug in ints_to_pydatetime affected .astype, so we test
        #  that here.
        right2 = dta.astype(object)[2]
        assert str(left) == str(right2)
        assert left.utcoffset() == right2.utcoffset()

    @pytest.mark.parametrize(
        "freq, freq_depr",
        [
            ("2ME", "2M"),
            ("2SME", "2SM"),
            ("2SME", "2sm"),
            ("2QE", "2Q"),
            ("2QE-SEP", "2Q-SEP"),
            ("1YE", "1Y"),
            ("2YE-MAR", "2Y-MAR"),
            ("1YE", "1A"),
            ("2YE-MAR", "2A-MAR"),
            ("2ME", "2m"),
            ("2QE-SEP", "2q-sep"),
            ("2YE-MAR", "2a-mar"),
            ("2YE", "2y"),
        ],
    )
    def test_date_range_frequency_M_Q_Y_A_deprecated(self, freq, freq_depr):
        # GH#9586, GH#54275
        depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed "
        f"in a future version, please use '{freq[1:]}' instead."

        expected = pd.date_range("1/1/2000", periods=4, freq=freq)
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            result = pd.date_range("1/1/2000", periods=4, freq=freq_depr)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("freq_depr", ["2H", "2CBH", "2MIN", "2S", "2mS", "2Us"])
    def test_date_range_uppercase_frequency_deprecated(self, freq_depr):
        # GH#9586, GH#54939
        depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a "
        f"future version. Please use '{freq_depr.lower()[1:]}' instead."

        expected = pd.date_range("1/1/2000", periods=4, freq=freq_depr.lower())
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            result = pd.date_range("1/1/2000", periods=4, freq=freq_depr)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "freq_depr",
        [
            "2ye-mar",
            "2ys",
            "2qe",
            "2qs-feb",
            "2bqs",
            "2sms",
            "2bms",
            "2cbme",
            "2me",
            "2w",
        ],
    )
    def test_date_range_lowercase_frequency_deprecated(self, freq_depr):
        # GH#9586, GH#54939
        depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a "
        f"future version, please use '{freq_depr.upper()[1:]}' instead."

        expected = pd.date_range("1/1/2000", periods=4, freq=freq_depr.upper())
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            result = pd.date_range("1/1/2000", periods=4, freq=freq_depr)
        tm.assert_index_equal(result, expected)


def test_factorize_sort_without_freq():
    dta = DatetimeArray._from_sequence([0, 2, 1], dtype="M8[ns]")

    msg = r"call pd.factorize\(obj, sort=True\) instead"
    with pytest.raises(NotImplementedError, match=msg):
        dta.factorize(sort=True)

    # Do TimedeltaArray while we're here
    tda = dta - dta[0]
    with pytest.raises(NotImplementedError, match=msg):
        tda.factorize(sort=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\biomechanics\tests\test_musculotendon.py ===
"""Tests for the ``sympy.physics.biomechanics.musculotendon.py`` module."""

import abc

import pytest

from sympy.core.expr import UnevaluatedExpr
from sympy.core.numbers import Float, Integer, Rational
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import MutableDenseMatrix as Matrix, eye, zeros
from sympy.physics.biomechanics.activation import (
    FirstOrderActivationDeGroote2016
)
from sympy.physics.biomechanics.curve import (
    CharacteristicCurveCollection,
    FiberForceLengthActiveDeGroote2016,
    FiberForceLengthPassiveDeGroote2016,
    FiberForceLengthPassiveInverseDeGroote2016,
    FiberForceVelocityDeGroote2016,
    FiberForceVelocityInverseDeGroote2016,
    TendonForceLengthDeGroote2016,
    TendonForceLengthInverseDeGroote2016,
)
from sympy.physics.biomechanics.musculotendon import (
    MusculotendonBase,
    MusculotendonDeGroote2016,
    MusculotendonFormulation,
)
from sympy.physics.biomechanics._mixin import _NamedMixin
from sympy.physics.mechanics.actuator import ForceActuator
from sympy.physics.mechanics.pathway import LinearPathway
from sympy.physics.vector.frame import ReferenceFrame
from sympy.physics.vector.functions import dynamicsymbols
from sympy.physics.vector.point import Point
from sympy.simplify.simplify import simplify


class TestMusculotendonFormulation:
    @staticmethod
    def test_rigid_tendon_member():
        assert MusculotendonFormulation(0) == 0
        assert MusculotendonFormulation.RIGID_TENDON == 0

    @staticmethod
    def test_fiber_length_explicit_member():
        assert MusculotendonFormulation(1) == 1
        assert MusculotendonFormulation.FIBER_LENGTH_EXPLICIT == 1

    @staticmethod
    def test_tendon_force_explicit_member():
        assert MusculotendonFormulation(2) == 2
        assert MusculotendonFormulation.TENDON_FORCE_EXPLICIT == 2

    @staticmethod
    def test_fiber_length_implicit_member():
        assert MusculotendonFormulation(3) == 3
        assert MusculotendonFormulation.FIBER_LENGTH_IMPLICIT == 3

    @staticmethod
    def test_tendon_force_implicit_member():
        assert MusculotendonFormulation(4) == 4
        assert MusculotendonFormulation.TENDON_FORCE_IMPLICIT == 4


class TestMusculotendonBase:

    @staticmethod
    def test_is_abstract_base_class():
        assert issubclass(MusculotendonBase, abc.ABC)

    @staticmethod
    def test_class():
        assert issubclass(MusculotendonBase, ForceActuator)
        assert issubclass(MusculotendonBase, _NamedMixin)
        assert MusculotendonBase.__name__ == 'MusculotendonBase'

    @staticmethod
    def test_cannot_instantiate_directly():
        with pytest.raises(TypeError):
            _ = MusculotendonBase()


@pytest.mark.parametrize('musculotendon_concrete', [MusculotendonDeGroote2016])
class TestMusculotendonRigidTendon:

    @pytest.fixture(autouse=True)
    def _musculotendon_rigid_tendon_fixture(self, musculotendon_concrete):
        self.name = 'name'
        self.N = ReferenceFrame('N')
        self.q = dynamicsymbols('q')
        self.origin = Point('pO')
        self.insertion = Point('pI')
        self.insertion.set_pos(self.origin, self.q*self.N.x)
        self.pathway = LinearPathway(self.origin, self.insertion)
        self.activation = FirstOrderActivationDeGroote2016(self.name)
        self.e = self.activation.excitation
        self.a = self.activation.activation
        self.tau_a = self.activation.activation_time_constant
        self.tau_d = self.activation.deactivation_time_constant
        self.b = self.activation.smoothing_rate
        self.formulation = MusculotendonFormulation.RIGID_TENDON
        self.l_T_slack = Symbol('l_T_slack')
        self.F_M_max = Symbol('F_M_max')
        self.l_M_opt = Symbol('l_M_opt')
        self.v_M_max = Symbol('v_M_max')
        self.alpha_opt = Symbol('alpha_opt')
        self.beta = Symbol('beta')
        self.instance = musculotendon_concrete(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=self.formulation,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        self.da_expr = (
            (1/(self.tau_a*(Rational(1, 2) + Rational(3, 2)*self.a)))
            *(Rational(1, 2) + Rational(1, 2)*tanh(self.b*(self.e - self.a)))
            + ((Rational(1, 2) + Rational(3, 2)*self.a)/self.tau_d)
            *(Rational(1, 2) - Rational(1, 2)*tanh(self.b*(self.e - self.a)))
        )*(self.e - self.a)

    def test_state_vars(self):
        assert hasattr(self.instance, 'x')
        assert hasattr(self.instance, 'state_vars')
        assert self.instance.x == self.instance.state_vars
        x_expected = Matrix([self.a])
        assert self.instance.x == x_expected
        assert self.instance.state_vars == x_expected
        assert isinstance(self.instance.x, Matrix)
        assert isinstance(self.instance.state_vars, Matrix)
        assert self.instance.x.shape == (1, 1)
        assert self.instance.state_vars.shape == (1, 1)

    def test_input_vars(self):
        assert hasattr(self.instance, 'r')
        assert hasattr(self.instance, 'input_vars')
        assert self.instance.r == self.instance.input_vars
        r_expected = Matrix([self.e])
        assert self.instance.r == r_expected
        assert self.instance.input_vars == r_expected
        assert isinstance(self.instance.r, Matrix)
        assert isinstance(self.instance.input_vars, Matrix)
        assert self.instance.r.shape == (1, 1)
        assert self.instance.input_vars.shape == (1, 1)

    def test_constants(self):
        assert hasattr(self.instance, 'p')
        assert hasattr(self.instance, 'constants')
        assert self.instance.p == self.instance.constants
        p_expected = Matrix(
            [
                self.l_T_slack,
                self.F_M_max,
                self.l_M_opt,
                self.v_M_max,
                self.alpha_opt,
                self.beta,
                self.tau_a,
                self.tau_d,
                self.b,
                Symbol('c_0_fl_T_name'),
                Symbol('c_1_fl_T_name'),
                Symbol('c_2_fl_T_name'),
                Symbol('c_3_fl_T_name'),
                Symbol('c_0_fl_M_pas_name'),
                Symbol('c_1_fl_M_pas_name'),
                Symbol('c_0_fl_M_act_name'),
                Symbol('c_1_fl_M_act_name'),
                Symbol('c_2_fl_M_act_name'),
                Symbol('c_3_fl_M_act_name'),
                Symbol('c_4_fl_M_act_name'),
                Symbol('c_5_fl_M_act_name'),
                Symbol('c_6_fl_M_act_name'),
                Symbol('c_7_fl_M_act_name'),
                Symbol('c_8_fl_M_act_name'),
                Symbol('c_9_fl_M_act_name'),
                Symbol('c_10_fl_M_act_name'),
                Symbol('c_11_fl_M_act_name'),
                Symbol('c_0_fv_M_name'),
                Symbol('c_1_fv_M_name'),
                Symbol('c_2_fv_M_name'),
                Symbol('c_3_fv_M_name'),
            ]
        )
        assert self.instance.p == p_expected
        assert self.instance.constants == p_expected
        assert isinstance(self.instance.p, Matrix)
        assert isinstance(self.instance.constants, Matrix)
        assert self.instance.p.shape == (31, 1)
        assert self.instance.constants.shape == (31, 1)

    def test_M(self):
        assert hasattr(self.instance, 'M')
        M_expected = Matrix([1])
        assert self.instance.M == M_expected
        assert isinstance(self.instance.M, Matrix)
        assert self.instance.M.shape == (1, 1)

    def test_F(self):
        assert hasattr(self.instance, 'F')
        F_expected = Matrix([self.da_expr])
        assert self.instance.F == F_expected
        assert isinstance(self.instance.F, Matrix)
        assert self.instance.F.shape == (1, 1)

    def test_rhs(self):
        assert hasattr(self.instance, 'rhs')
        rhs_expected = Matrix([self.da_expr])
        rhs = self.instance.rhs()
        assert isinstance(rhs, Matrix)
        assert rhs.shape == (1, 1)
        assert simplify(rhs - rhs_expected) == zeros(1)


@pytest.mark.parametrize(
    'musculotendon_concrete, curve',
    [
        (
            MusculotendonDeGroote2016,
            CharacteristicCurveCollection(
                tendon_force_length=TendonForceLengthDeGroote2016,
                tendon_force_length_inverse=TendonForceLengthInverseDeGroote2016,
                fiber_force_length_passive=FiberForceLengthPassiveDeGroote2016,
                fiber_force_length_passive_inverse=FiberForceLengthPassiveInverseDeGroote2016,
                fiber_force_length_active=FiberForceLengthActiveDeGroote2016,
                fiber_force_velocity=FiberForceVelocityDeGroote2016,
                fiber_force_velocity_inverse=FiberForceVelocityInverseDeGroote2016,
            ),
        )
    ],
)
class TestFiberLengthExplicit:

    @pytest.fixture(autouse=True)
    def _musculotendon_fiber_length_explicit_fixture(
        self,
        musculotendon_concrete,
        curve,
    ):
        self.name = 'name'
        self.N = ReferenceFrame('N')
        self.q = dynamicsymbols('q')
        self.origin = Point('pO')
        self.insertion = Point('pI')
        self.insertion.set_pos(self.origin, self.q*self.N.x)
        self.pathway = LinearPathway(self.origin, self.insertion)
        self.activation = FirstOrderActivationDeGroote2016(self.name)
        self.e = self.activation.excitation
        self.a = self.activation.activation
        self.tau_a = self.activation.activation_time_constant
        self.tau_d = self.activation.deactivation_time_constant
        self.b = self.activation.smoothing_rate
        self.formulation = MusculotendonFormulation.FIBER_LENGTH_EXPLICIT
        self.l_T_slack = Symbol('l_T_slack')
        self.F_M_max = Symbol('F_M_max')
        self.l_M_opt = Symbol('l_M_opt')
        self.v_M_max = Symbol('v_M_max')
        self.alpha_opt = Symbol('alpha_opt')
        self.beta = Symbol('beta')
        self.instance = musculotendon_concrete(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=self.formulation,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
            with_defaults=True,
        )
        self.l_M_tilde = dynamicsymbols('l_M_tilde_name')
        l_MT = self.pathway.length
        l_M = self.l_M_tilde*self.l_M_opt
        l_T = l_MT - sqrt(l_M**2 - (self.l_M_opt*sin(self.alpha_opt))**2)
        fl_T = curve.tendon_force_length.with_defaults(l_T/self.l_T_slack)
        fl_M_pas = curve.fiber_force_length_passive.with_defaults(self.l_M_tilde)
        fl_M_act = curve.fiber_force_length_active.with_defaults(self.l_M_tilde)
        v_M_tilde = curve.fiber_force_velocity_inverse.with_defaults(
            ((((fl_T*self.F_M_max)/((l_MT - l_T)/l_M))/self.F_M_max) - fl_M_pas)
            /(self.a*fl_M_act)
        )
        self.dl_M_tilde_expr = (self.v_M_max/self.l_M_opt)*v_M_tilde
        self.da_expr = (
            (1/(self.tau_a*(Rational(1, 2) + Rational(3, 2)*self.a)))
            *(Rational(1, 2) + Rational(1, 2)*tanh(self.b*(self.e - self.a)))
            + ((Rational(1, 2) + Rational(3, 2)*self.a)/self.tau_d)
            *(Rational(1, 2) - Rational(1, 2)*tanh(self.b*(self.e - self.a)))
        )*(self.e - self.a)

    def test_state_vars(self):
        assert hasattr(self.instance, 'x')
        assert hasattr(self.instance, 'state_vars')
        assert self.instance.x == self.instance.state_vars
        x_expected = Matrix([self.l_M_tilde, self.a])
        assert self.instance.x == x_expected
        assert self.instance.state_vars == x_expected
        assert isinstance(self.instance.x, Matrix)
        assert isinstance(self.instance.state_vars, Matrix)
        assert self.instance.x.shape == (2, 1)
        assert self.instance.state_vars.shape == (2, 1)

    def test_input_vars(self):
        assert hasattr(self.instance, 'r')
        assert hasattr(self.instance, 'input_vars')
        assert self.instance.r == self.instance.input_vars
        r_expected = Matrix([self.e])
        assert self.instance.r == r_expected
        assert self.instance.input_vars == r_expected
        assert isinstance(self.instance.r, Matrix)
        assert isinstance(self.instance.input_vars, Matrix)
        assert self.instance.r.shape == (1, 1)
        assert self.instance.input_vars.shape == (1, 1)

    def test_constants(self):
        assert hasattr(self.instance, 'p')
        assert hasattr(self.instance, 'constants')
        assert self.instance.p == self.instance.constants
        p_expected = Matrix(
            [
                self.l_T_slack,
                self.F_M_max,
                self.l_M_opt,
                self.v_M_max,
                self.alpha_opt,
                self.beta,
                self.tau_a,
                self.tau_d,
                self.b,
            ]
        )
        assert self.instance.p == p_expected
        assert self.instance.constants == p_expected
        assert isinstance(self.instance.p, Matrix)
        assert isinstance(self.instance.constants, Matrix)
        assert self.instance.p.shape == (9, 1)
        assert self.instance.constants.shape == (9, 1)

    def test_M(self):
        assert hasattr(self.instance, 'M')
        M_expected = eye(2)
        assert self.instance.M == M_expected
        assert isinstance(self.instance.M, Matrix)
        assert self.instance.M.shape == (2, 2)

    def test_F(self):
        assert hasattr(self.instance, 'F')
        F_expected = Matrix([self.dl_M_tilde_expr, self.da_expr])
        assert self.instance.F == F_expected
        assert isinstance(self.instance.F, Matrix)
        assert self.instance.F.shape == (2, 1)

    def test_rhs(self):
        assert hasattr(self.instance, 'rhs')
        rhs_expected = Matrix([self.dl_M_tilde_expr, self.da_expr])
        rhs = self.instance.rhs()
        assert isinstance(rhs, Matrix)
        assert rhs.shape == (2, 1)
        assert simplify(rhs - rhs_expected) == zeros(2, 1)


@pytest.mark.parametrize(
    'musculotendon_concrete, curve',
    [
        (
            MusculotendonDeGroote2016,
            CharacteristicCurveCollection(
                tendon_force_length=TendonForceLengthDeGroote2016,
                tendon_force_length_inverse=TendonForceLengthInverseDeGroote2016,
                fiber_force_length_passive=FiberForceLengthPassiveDeGroote2016,
                fiber_force_length_passive_inverse=FiberForceLengthPassiveInverseDeGroote2016,
                fiber_force_length_active=FiberForceLengthActiveDeGroote2016,
                fiber_force_velocity=FiberForceVelocityDeGroote2016,
                fiber_force_velocity_inverse=FiberForceVelocityInverseDeGroote2016,
            ),
        )
    ],
)
class TestTendonForceExplicit:

    @pytest.fixture(autouse=True)
    def _musculotendon_tendon_force_explicit_fixture(
        self,
        musculotendon_concrete,
        curve,
    ):
        self.name = 'name'
        self.N = ReferenceFrame('N')
        self.q = dynamicsymbols('q')
        self.origin = Point('pO')
        self.insertion = Point('pI')
        self.insertion.set_pos(self.origin, self.q*self.N.x)
        self.pathway = LinearPathway(self.origin, self.insertion)
        self.activation = FirstOrderActivationDeGroote2016(self.name)
        self.e = self.activation.excitation
        self.a = self.activation.activation
        self.tau_a = self.activation.activation_time_constant
        self.tau_d = self.activation.deactivation_time_constant
        self.b = self.activation.smoothing_rate
        self.formulation = MusculotendonFormulation.TENDON_FORCE_EXPLICIT
        self.l_T_slack = Symbol('l_T_slack')
        self.F_M_max = Symbol('F_M_max')
        self.l_M_opt = Symbol('l_M_opt')
        self.v_M_max = Symbol('v_M_max')
        self.alpha_opt = Symbol('alpha_opt')
        self.beta = Symbol('beta')
        self.instance = musculotendon_concrete(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=self.formulation,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
            with_defaults=True,
        )
        self.F_T_tilde = dynamicsymbols('F_T_tilde_name')
        l_T_tilde = curve.tendon_force_length_inverse.with_defaults(self.F_T_tilde)
        l_MT = self.pathway.length
        v_MT = self.pathway.extension_velocity
        l_T = l_T_tilde*self.l_T_slack
        l_M = sqrt((l_MT - l_T)**2 + (self.l_M_opt*sin(self.alpha_opt))**2)
        l_M_tilde = l_M/self.l_M_opt
        cos_alpha = (l_MT - l_T)/l_M
        F_T = self.F_T_tilde*self.F_M_max
        F_M = F_T/cos_alpha
        F_M_tilde = F_M/self.F_M_max
        fl_M_pas = curve.fiber_force_length_passive.with_defaults(l_M_tilde)
        fl_M_act = curve.fiber_force_length_active.with_defaults(l_M_tilde)
        fv_M = (F_M_tilde - fl_M_pas)/(self.a*fl_M_act)
        v_M_tilde = curve.fiber_force_velocity_inverse.with_defaults(fv_M)
        v_M = v_M_tilde*self.v_M_max
        v_T = v_MT - v_M/cos_alpha
        v_T_tilde = v_T/self.l_T_slack
        self.dF_T_tilde_expr = (
            Float('0.2')*Float('33.93669377311689')*exp(
                Float('33.93669377311689')*UnevaluatedExpr(l_T_tilde - Float('0.995'))
            )*v_T_tilde
        )
        self.da_expr = (
            (1/(self.tau_a*(Rational(1, 2) + Rational(3, 2)*self.a)))
            *(Rational(1, 2) + Rational(1, 2)*tanh(self.b*(self.e - self.a)))
            + ((Rational(1, 2) + Rational(3, 2)*self.a)/self.tau_d)
            *(Rational(1, 2) - Rational(1, 2)*tanh(self.b*(self.e - self.a)))
        )*(self.e - self.a)

    def test_state_vars(self):
        assert hasattr(self.instance, 'x')
        assert hasattr(self.instance, 'state_vars')
        assert self.instance.x == self.instance.state_vars
        x_expected = Matrix([self.F_T_tilde, self.a])
        assert self.instance.x == x_expected
        assert self.instance.state_vars == x_expected
        assert isinstance(self.instance.x, Matrix)
        assert isinstance(self.instance.state_vars, Matrix)
        assert self.instance.x.shape == (2, 1)
        assert self.instance.state_vars.shape == (2, 1)

    def test_input_vars(self):
        assert hasattr(self.instance, 'r')
        assert hasattr(self.instance, 'input_vars')
        assert self.instance.r == self.instance.input_vars
        r_expected = Matrix([self.e])
        assert self.instance.r == r_expected
        assert self.instance.input_vars == r_expected
        assert isinstance(self.instance.r, Matrix)
        assert isinstance(self.instance.input_vars, Matrix)
        assert self.instance.r.shape == (1, 1)
        assert self.instance.input_vars.shape == (1, 1)

    def test_constants(self):
        assert hasattr(self.instance, 'p')
        assert hasattr(self.instance, 'constants')
        assert self.instance.p == self.instance.constants
        p_expected = Matrix(
            [
                self.l_T_slack,
                self.F_M_max,
                self.l_M_opt,
                self.v_M_max,
                self.alpha_opt,
                self.beta,
                self.tau_a,
                self.tau_d,
                self.b,
            ]
        )
        assert self.instance.p == p_expected
        assert self.instance.constants == p_expected
        assert isinstance(self.instance.p, Matrix)
        assert isinstance(self.instance.constants, Matrix)
        assert self.instance.p.shape == (9, 1)
        assert self.instance.constants.shape == (9, 1)

    def test_M(self):
        assert hasattr(self.instance, 'M')
        M_expected = eye(2)
        assert self.instance.M == M_expected
        assert isinstance(self.instance.M, Matrix)
        assert self.instance.M.shape == (2, 2)

    def test_F(self):
        assert hasattr(self.instance, 'F')
        F_expected = Matrix([self.dF_T_tilde_expr, self.da_expr])
        assert self.instance.F == F_expected
        assert isinstance(self.instance.F, Matrix)
        assert self.instance.F.shape == (2, 1)

    def test_rhs(self):
        assert hasattr(self.instance, 'rhs')
        rhs_expected = Matrix([self.dF_T_tilde_expr, self.da_expr])
        rhs = self.instance.rhs()
        assert isinstance(rhs, Matrix)
        assert rhs.shape == (2, 1)
        assert simplify(rhs - rhs_expected) == zeros(2, 1)


class TestMusculotendonDeGroote2016:

    @staticmethod
    def test_class():
        assert issubclass(MusculotendonDeGroote2016, ForceActuator)
        assert issubclass(MusculotendonDeGroote2016, _NamedMixin)
        assert MusculotendonDeGroote2016.__name__ == 'MusculotendonDeGroote2016'

    @staticmethod
    def test_instance():
        origin = Point('pO')
        insertion = Point('pI')
        insertion.set_pos(origin, dynamicsymbols('q')*ReferenceFrame('N').x)
        pathway = LinearPathway(origin, insertion)
        activation = FirstOrderActivationDeGroote2016('name')
        l_T_slack = Symbol('l_T_slack')
        F_M_max = Symbol('F_M_max')
        l_M_opt = Symbol('l_M_opt')
        v_M_max = Symbol('v_M_max')
        alpha_opt = Symbol('alpha_opt')
        beta = Symbol('beta')
        instance = MusculotendonDeGroote2016(
            'name',
            pathway,
            activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=l_T_slack,
            peak_isometric_force=F_M_max,
            optimal_fiber_length=l_M_opt,
            maximal_fiber_velocity=v_M_max,
            optimal_pennation_angle=alpha_opt,
            fiber_damping_coefficient=beta,
        )
        assert isinstance(instance, MusculotendonDeGroote2016)

    @pytest.fixture(autouse=True)
    def _musculotendon_fixture(self):
        self.name = 'name'
        self.N = ReferenceFrame('N')
        self.q = dynamicsymbols('q')
        self.origin = Point('pO')
        self.insertion = Point('pI')
        self.insertion.set_pos(self.origin, self.q*self.N.x)
        self.pathway = LinearPathway(self.origin, self.insertion)
        self.activation = FirstOrderActivationDeGroote2016(self.name)
        self.l_T_slack = Symbol('l_T_slack')
        self.F_M_max = Symbol('F_M_max')
        self.l_M_opt = Symbol('l_M_opt')
        self.v_M_max = Symbol('v_M_max')
        self.alpha_opt = Symbol('alpha_opt')
        self.beta = Symbol('beta')

    def test_with_defaults(self):
        origin = Point('pO')
        insertion = Point('pI')
        insertion.set_pos(origin, dynamicsymbols('q')*ReferenceFrame('N').x)
        pathway = LinearPathway(origin, insertion)
        activation = FirstOrderActivationDeGroote2016('name')
        l_T_slack = Symbol('l_T_slack')
        F_M_max = Symbol('F_M_max')
        l_M_opt = Symbol('l_M_opt')
        v_M_max = Float('10.0')
        alpha_opt = Float('0.0')
        beta = Float('0.1')
        instance = MusculotendonDeGroote2016.with_defaults(
            'name',
            pathway,
            activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=l_T_slack,
            peak_isometric_force=F_M_max,
            optimal_fiber_length=l_M_opt,
        )
        assert instance.tendon_slack_length == l_T_slack
        assert instance.peak_isometric_force == F_M_max
        assert instance.optimal_fiber_length == l_M_opt
        assert instance.maximal_fiber_velocity == v_M_max
        assert instance.optimal_pennation_angle == alpha_opt
        assert instance.fiber_damping_coefficient == beta

    @pytest.mark.parametrize(
        'l_T_slack, expected',
        [
            (None, Symbol('l_T_slack_name')),
            (Symbol('l_T_slack'), Symbol('l_T_slack')),
            (Rational(1, 2), Rational(1, 2)),
            (Float('0.5'), Float('0.5')),
        ],
    )
    def test_tendon_slack_length(self, l_T_slack, expected):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        assert instance.l_T_slack == expected
        assert instance.tendon_slack_length == expected

    @pytest.mark.parametrize(
        'F_M_max, expected',
        [
            (None, Symbol('F_M_max_name')),
            (Symbol('F_M_max'), Symbol('F_M_max')),
            (Integer(1000), Integer(1000)),
            (Float('1000.0'), Float('1000.0')),
        ],
    )
    def test_peak_isometric_force(self, F_M_max, expected):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        assert instance.F_M_max == expected
        assert instance.peak_isometric_force == expected

    @pytest.mark.parametrize(
        'l_M_opt, expected',
        [
            (None, Symbol('l_M_opt_name')),
            (Symbol('l_M_opt'), Symbol('l_M_opt')),
            (Rational(1, 2), Rational(1, 2)),
            (Float('0.5'), Float('0.5')),
        ],
    )
    def test_optimal_fiber_length(self, l_M_opt, expected):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        assert instance.l_M_opt == expected
        assert instance.optimal_fiber_length == expected

    @pytest.mark.parametrize(
        'v_M_max, expected',
        [
            (None, Symbol('v_M_max_name')),
            (Symbol('v_M_max'), Symbol('v_M_max')),
            (Integer(10), Integer(10)),
            (Float('10.0'), Float('10.0')),
        ],
    )
    def test_maximal_fiber_velocity(self, v_M_max, expected):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        assert instance.v_M_max == expected
        assert instance.maximal_fiber_velocity == expected

    @pytest.mark.parametrize(
        'alpha_opt, expected',
        [
            (None, Symbol('alpha_opt_name')),
            (Symbol('alpha_opt'), Symbol('alpha_opt')),
            (Integer(0), Integer(0)),
            (Float('0.1'), Float('0.1')),
        ],
    )
    def test_optimal_pennation_angle(self, alpha_opt, expected):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        assert instance.alpha_opt == expected
        assert instance.optimal_pennation_angle == expected

    @pytest.mark.parametrize(
        'beta, expected',
        [
            (None, Symbol('beta_name')),
            (Symbol('beta'), Symbol('beta')),
            (Integer(0), Integer(0)),
            (Rational(1, 10), Rational(1, 10)),
            (Float('0.1'), Float('0.1')),
        ],
    )
    def test_fiber_damping_coefficient(self, beta, expected):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=beta,
        )
        assert instance.beta == expected
        assert instance.fiber_damping_coefficient == expected

    def test_excitation(self):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
        )
        assert hasattr(instance, 'e')
        assert hasattr(instance, 'excitation')
        e_expected = dynamicsymbols('e_name')
        assert instance.e == e_expected
        assert instance.excitation == e_expected
        assert instance.e is instance.excitation

    def test_excitation_is_immutable(self):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
        )
        with pytest.raises(AttributeError):
            instance.e = None
        with pytest.raises(AttributeError):
            instance.excitation = None

    def test_activation(self):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
        )
        assert hasattr(instance, 'a')
        assert hasattr(instance, 'activation')
        a_expected = dynamicsymbols('a_name')
        assert instance.a == a_expected
        assert instance.activation == a_expected

    def test_activation_is_immutable(self):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
        )
        with pytest.raises(AttributeError):
            instance.a = None
        with pytest.raises(AttributeError):
            instance.activation = None

    def test_repr(self):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        expected = (
            'MusculotendonDeGroote2016(\'name\', '
            'pathway=LinearPathway(pO, pI), '
            'activation_dynamics=FirstOrderActivationDeGroote2016(\'name\', '
            'activation_time_constant=tau_a_name, '
            'deactivation_time_constant=tau_d_name, '
            'smoothing_rate=b_name), '
            'musculotendon_dynamics=0, '
            'tendon_slack_length=l_T_slack, '
            'peak_isometric_force=F_M_max, '
            'optimal_fiber_length=l_M_opt, '
            'maximal_fiber_velocity=v_M_max, '
            'optimal_pennation_angle=alpha_opt, '
            'fiber_damping_coefficient=beta)'
        )
        assert repr(instance) == expected