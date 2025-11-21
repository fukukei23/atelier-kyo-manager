
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\test_datetimelike.py ===
""" Test cases for time series specific (freq conversion, etc) """
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
import pickle

import numpy as np
import pytest

from pandas._libs.tslibs import (
    BaseOffset,
    to_offset,
)
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr

from pandas import (
    DataFrame,
    Index,
    NaT,
    Series,
    concat,
    isna,
    to_datetime,
)
import pandas._testing as tm
from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    bdate_range,
    date_range,
)
from pandas.core.indexes.period import (
    Period,
    PeriodIndex,
    period_range,
)
from pandas.core.indexes.timedeltas import timedelta_range
from pandas.tests.plotting.common import _check_ticks_props

from pandas.tseries.offsets import WeekOfMonth

mpl = pytest.importorskip("matplotlib")


class TestTSPlot:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_ts_plot_with_tz(self, tz_aware_fixture):
        # GH2877, GH17173, GH31205, GH31580
        tz = tz_aware_fixture
        index = date_range("1/1/2011", periods=2, freq="h", tz=tz)
        ts = Series([188.5, 328.25], index=index)
        _check_plot_works(ts.plot)
        ax = ts.plot()
        xdata = next(iter(ax.get_lines())).get_xdata()
        # Check first and last points' labels are correct
        assert (xdata[0].hour, xdata[0].minute) == (0, 0)
        assert (xdata[-1].hour, xdata[-1].minute) == (1, 0)

    def test_fontsize_set_correctly(self):
        # For issue #8765
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 9)), index=range(10)
        )
        _, ax = mpl.pyplot.subplots()
        df.plot(fontsize=2, ax=ax)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            assert label.get_fontsize() == 2

    def test_frame_inferred(self):
        # inferred freq
        idx = date_range("1/1/1987", freq="MS", periods=100)
        idx = DatetimeIndex(idx.values, freq=None)

        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx
        )
        _check_plot_works(df.plot)

        # axes freq
        idx = idx[0:40].union(idx[45:99])
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx
        )
        _check_plot_works(df2.plot)

    def test_frame_inferred_n_gt_1(self):
        # N > 1
        idx = date_range("2008-1-1 00:15:00", freq="15min", periods=10)
        idx = DatetimeIndex(idx.values, freq=None)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx
        )
        _check_plot_works(df.plot)

    def test_is_error_nozeroindex(self):
        # GH11858
        i = np.array([1, 2, 3])
        a = DataFrame(i, index=i)
        _check_plot_works(a.plot, xerr=a)
        _check_plot_works(a.plot, yerr=a)

    def test_nonnumeric_exclude(self):
        idx = date_range("1/1/1987", freq="YE", periods=3)
        df = DataFrame({"A": ["x", "y", "z"], "B": [1, 2, 3]}, idx)

        fig, ax = mpl.pyplot.subplots()
        df.plot(ax=ax)  # it works
        assert len(ax.get_lines()) == 1  # B was plotted
        mpl.pyplot.close(fig)

    def test_nonnumeric_exclude_error(self):
        idx = date_range("1/1/1987", freq="YE", periods=3)
        df = DataFrame({"A": ["x", "y", "z"], "B": [1, 2, 3]}, idx)
        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            df["A"].plot()

    @pytest.mark.parametrize("freq", ["s", "min", "h", "D", "W", "M", "Q", "Y"])
    def test_tsplot_period(self, freq):
        idx = period_range("12/31/1999", freq=freq, periods=100)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _, ax = mpl.pyplot.subplots()
        _check_plot_works(ser.plot, ax=ax)

    @pytest.mark.parametrize(
        "freq", ["s", "min", "h", "D", "W", "ME", "QE-DEC", "YE", "1B30Min"]
    )
    def test_tsplot_datetime(self, freq):
        idx = date_range("12/31/1999", freq=freq, periods=100)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _, ax = mpl.pyplot.subplots()
        _check_plot_works(ser.plot, ax=ax)

    def test_tsplot(self):
        ts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        _, ax = mpl.pyplot.subplots()
        ts.plot(style="k", ax=ax)
        color = (0.0, 0.0, 0.0, 1)
        assert color == ax.get_lines()[0].get_color()

    def test_both_style_and_color(self):
        ts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        msg = (
            "Cannot pass 'style' string with a color symbol and 'color' "
            "keyword argument. Please use one or the other or pass 'style' "
            "without a color symbol"
        )
        with pytest.raises(ValueError, match=msg):
            ts.plot(style="b-", color="#000099")

        s = ts.reset_index(drop=True)
        with pytest.raises(ValueError, match=msg):
            s.plot(style="b-", color="#000099")

    @pytest.mark.parametrize("freq", ["ms", "us"])
    def test_high_freq(self, freq):
        _, ax = mpl.pyplot.subplots()
        rng = date_range("1/1/2012", periods=100, freq=freq)
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _check_plot_works(ser.plot, ax=ax)

    def test_get_datevalue(self):
        from pandas.plotting._matplotlib.converter import get_datevalue

        assert get_datevalue(None, "D") is None
        assert get_datevalue(1987, "Y") == 1987
        assert get_datevalue(Period(1987, "Y"), "M") == Period("1987-12", "M").ordinal
        assert get_datevalue("1/1/1987", "D") == Period("1987-1-1", "D").ordinal

    def test_ts_plot_format_coord(self):
        def check_format_of_first_point(ax, expected_string):
            first_line = ax.get_lines()[0]
            first_x = first_line.get_xdata()[0].ordinal
            first_y = first_line.get_ydata()[0]
            assert expected_string == ax.format_coord(first_x, first_y)

        annual = Series(1, index=date_range("2014-01-01", periods=3, freq="YE-DEC"))
        _, ax = mpl.pyplot.subplots()
        annual.plot(ax=ax)
        check_format_of_first_point(ax, "t = 2014  y = 1.000000")

        # note this is added to the annual plot already in existence, and
        # changes its freq field
        daily = Series(1, index=date_range("2014-01-01", periods=3, freq="D"))
        daily.plot(ax=ax)
        check_format_of_first_point(ax, "t = 2014-01-01  y = 1.000000")

    @pytest.mark.parametrize("freq", ["s", "min", "h", "D", "W", "M", "Q", "Y"])
    def test_line_plot_period_series(self, freq):
        idx = period_range("12/31/1999", freq=freq, periods=100)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _check_plot_works(ser.plot, ser.index.freq)

    @pytest.mark.parametrize(
        "frqncy", ["1s", "3s", "5min", "7h", "4D", "8W", "11M", "3Y"]
    )
    def test_line_plot_period_mlt_series(self, frqncy):
        # test period index line plot for series with multiples (`mlt`) of the
        # frequency (`frqncy`) rule code. tests resolution of issue #14763
        idx = period_range("12/31/1999", freq=frqncy, periods=100)
        s = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _check_plot_works(s.plot, s.index.freq.rule_code)

    @pytest.mark.parametrize(
        "freq", ["s", "min", "h", "D", "W", "ME", "QE-DEC", "YE", "1B30Min"]
    )
    def test_line_plot_datetime_series(self, freq):
        idx = date_range("12/31/1999", freq=freq, periods=100)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _check_plot_works(ser.plot, ser.index.freq.rule_code)

    @pytest.mark.parametrize("freq", ["s", "min", "h", "D", "W", "ME", "QE", "YE"])
    def test_line_plot_period_frame(self, freq):
        idx = date_range("12/31/1999", freq=freq, periods=100)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)),
            index=idx,
            columns=["A", "B", "C"],
        )
        _check_plot_works(df.plot, df.index.freq)

    @pytest.mark.parametrize(
        "frqncy", ["1s", "3s", "5min", "7h", "4D", "8W", "11M", "3Y"]
    )
    def test_line_plot_period_mlt_frame(self, frqncy):
        # test period index line plot for DataFrames with multiples (`mlt`)
        # of the frequency (`frqncy`) rule code. tests resolution of issue
        # #14763
        idx = period_range("12/31/1999", freq=frqncy, periods=100)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)),
            index=idx,
            columns=["A", "B", "C"],
        )
        freq = freq_to_period_freqstr(1, df.index.freq.rule_code)
        freq = df.index.asfreq(freq).freq
        _check_plot_works(df.plot, freq)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    @pytest.mark.parametrize(
        "freq", ["s", "min", "h", "D", "W", "ME", "QE-DEC", "YE", "1B30Min"]
    )
    def test_line_plot_datetime_frame(self, freq):
        idx = date_range("12/31/1999", freq=freq, periods=100)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)),
            index=idx,
            columns=["A", "B", "C"],
        )
        freq = freq_to_period_freqstr(1, df.index.freq.rule_code)
        freq = df.index.to_period(freq).freq
        _check_plot_works(df.plot, freq)

    @pytest.mark.parametrize(
        "freq", ["s", "min", "h", "D", "W", "ME", "QE-DEC", "YE", "1B30Min"]
    )
    def test_line_plot_inferred_freq(self, freq):
        idx = date_range("12/31/1999", freq=freq, periods=100)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        ser = Series(ser.values, Index(np.asarray(ser.index)))
        _check_plot_works(ser.plot, ser.index.inferred_freq)

        ser = ser.iloc[[0, 3, 5, 6]]
        _check_plot_works(ser.plot)

    def test_fake_inferred_business(self):
        _, ax = mpl.pyplot.subplots()
        rng = date_range("2001-1-1", "2001-1-10")
        ts = Series(range(len(rng)), index=rng)
        ts = concat([ts[:3], ts[5:]])
        ts.plot(ax=ax)
        assert not hasattr(ax, "freq")

    def test_plot_offset_freq(self):
        ser = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        _check_plot_works(ser.plot)

    def test_plot_offset_freq_business(self):
        dr = date_range("2023-01-01", freq="BQS", periods=10)
        ser = Series(np.random.default_rng(2).standard_normal(len(dr)), index=dr)
        _check_plot_works(ser.plot)

    def test_plot_multiple_inferred_freq(self):
        dr = Index([datetime(2000, 1, 1), datetime(2000, 1, 6), datetime(2000, 1, 11)])
        ser = Series(np.random.default_rng(2).standard_normal(len(dr)), index=dr)
        _check_plot_works(ser.plot)

    @pytest.mark.xfail(reason="Api changed in 3.6.0")
    def test_uhf(self):
        import pandas.plotting._matplotlib.converter as conv

        idx = date_range("2012-6-22 21:59:51.960928", freq="ms", periods=500)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 2)), index=idx
        )

        _, ax = mpl.pyplot.subplots()
        df.plot(ax=ax)
        axis = ax.get_xaxis()

        tlocs = axis.get_ticklocs()
        tlabels = axis.get_ticklabels()
        for loc, label in zip(tlocs, tlabels):
            xp = conv._from_ordinal(loc).strftime("%H:%M:%S.%f")
            rs = str(label.get_text())
            if len(rs):
                assert xp == rs

    def test_irreg_hf(self):
        idx = date_range("2012-6-22 21:59:51", freq="s", periods=10)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 2)), index=idx
        )

        irreg = df.iloc[[0, 1, 3, 4]]
        _, ax = mpl.pyplot.subplots()
        irreg.plot(ax=ax)
        diffs = Series(ax.get_lines()[0].get_xydata()[:, 0]).diff()

        sec = 1.0 / 24 / 60 / 60
        assert (np.fabs(diffs[1:] - [sec, sec * 2, sec]) < 1e-8).all()

    def test_irreg_hf_object(self):
        idx = date_range("2012-6-22 21:59:51", freq="s", periods=10)
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 2)), index=idx
        )
        _, ax = mpl.pyplot.subplots()
        df2.index = df2.index.astype(object)
        df2.plot(ax=ax)
        diffs = Series(ax.get_lines()[0].get_xydata()[:, 0]).diff()
        sec = 1.0 / 24 / 60 / 60
        assert (np.fabs(diffs[1:] - sec) < 1e-8).all()

    def test_irregular_datetime64_repr_bug(self):
        ser = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        ser = ser.iloc[[0, 1, 2, 7]]

        _, ax = mpl.pyplot.subplots()

        ret = ser.plot(ax=ax)
        assert ret is not None

        for rs, xp in zip(ax.get_lines()[0].get_xdata(), ser.index):
            assert rs == xp

    def test_business_freq(self):
        bts = Series(range(5), period_range("2020-01-01", periods=5))
        msg = r"PeriodDtype\[B\] is deprecated"
        dt = bts.index[0].to_timestamp()
        with tm.assert_produces_warning(FutureWarning, match=msg):
            bts.index = period_range(start=dt, periods=len(bts), freq="B")
        _, ax = mpl.pyplot.subplots()
        bts.plot(ax=ax)
        assert ax.get_lines()[0].get_xydata()[0, 0] == bts.index[0].ordinal
        idx = ax.get_lines()[0].get_xdata()
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert PeriodIndex(data=idx).freqstr == "B"

    def test_business_freq_convert(self):
        bts = Series(
            np.arange(300, dtype=np.float64),
            index=date_range("2020-01-01", periods=300, freq="B"),
        ).asfreq("BME")
        ts = bts.to_period("M")
        _, ax = mpl.pyplot.subplots()
        bts.plot(ax=ax)
        assert ax.get_lines()[0].get_xydata()[0, 0] == ts.index[0].ordinal
        idx = ax.get_lines()[0].get_xdata()
        assert PeriodIndex(data=idx).freqstr == "M"

    def test_freq_with_no_period_alias(self):
        # GH34487
        freq = WeekOfMonth()
        bts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        ).asfreq(freq)
        _, ax = mpl.pyplot.subplots()
        bts.plot(ax=ax)

        idx = ax.get_lines()[0].get_xdata()
        msg = "freq not specified and cannot be inferred"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(data=idx)

    def test_nonzero_base(self):
        # GH2571
        idx = date_range("2012-12-20", periods=24, freq="h") + timedelta(minutes=30)
        df = DataFrame(np.arange(24), index=idx)
        _, ax = mpl.pyplot.subplots()
        df.plot(ax=ax)
        rs = ax.get_lines()[0].get_xdata()
        assert not Index(rs).is_normalized

    def test_dataframe(self):
        bts = DataFrame(
            {
                "a": Series(
                    np.arange(10, dtype=np.float64),
                    index=date_range("2020-01-01", periods=10),
                )
            }
        )
        _, ax = mpl.pyplot.subplots()
        bts.plot(ax=ax)
        idx = ax.get_lines()[0].get_xdata()
        tm.assert_index_equal(bts.index.to_period(), PeriodIndex(idx))

    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    @pytest.mark.parametrize(
        "obj",
        [
            Series(
                np.arange(10, dtype=np.float64),
                index=date_range("2020-01-01", periods=10),
            ),
            DataFrame(
                {
                    "a": Series(
                        np.arange(10, dtype=np.float64),
                        index=date_range("2020-01-01", periods=10),
                    ),
                    "b": Series(
                        np.arange(10, dtype=np.float64),
                        index=date_range("2020-01-01", periods=10),
                    )
                    + 1,
                }
            ),
        ],
    )
    def test_axis_limits(self, obj):
        _, ax = mpl.pyplot.subplots()
        obj.plot(ax=ax)
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0] - 5, xlim[1] + 10)
        result = ax.get_xlim()
        assert result[0] == xlim[0] - 5
        assert result[1] == xlim[1] + 10

        # string
        expected = (Period("1/1/2000", ax.freq), Period("4/1/2000", ax.freq))
        ax.set_xlim("1/1/2000", "4/1/2000")
        result = ax.get_xlim()
        assert int(result[0]) == expected[0].ordinal
        assert int(result[1]) == expected[1].ordinal

        # datetime
        expected = (Period("1/1/2000", ax.freq), Period("4/1/2000", ax.freq))
        ax.set_xlim(datetime(2000, 1, 1), datetime(2000, 4, 1))
        result = ax.get_xlim()
        assert int(result[0]) == expected[0].ordinal
        assert int(result[1]) == expected[1].ordinal
        fig = ax.get_figure()
        mpl.pyplot.close(fig)

    def test_get_finder(self):
        import pandas.plotting._matplotlib.converter as conv

        assert conv.get_finder(to_offset("B")) == conv._daily_finder
        assert conv.get_finder(to_offset("D")) == conv._daily_finder
        assert conv.get_finder(to_offset("ME")) == conv._monthly_finder
        assert conv.get_finder(to_offset("QE")) == conv._quarterly_finder
        assert conv.get_finder(to_offset("YE")) == conv._annual_finder
        assert conv.get_finder(to_offset("W")) == conv._daily_finder

    def test_finder_daily(self):
        day_lst = [10, 40, 252, 400, 950, 2750, 10000]

        msg = "Period with BDay freq is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            xpl1 = xpl2 = [Period("1999-1-1", freq="B").ordinal] * len(day_lst)
        rs1 = []
        rs2 = []
        for n in day_lst:
            rng = bdate_range("1999-1-1", periods=n)
            ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            _, ax = mpl.pyplot.subplots()
            ser.plot(ax=ax)
            xaxis = ax.get_xaxis()
            rs1.append(xaxis.get_majorticklocs()[0])

            vmin, vmax = ax.get_xlim()
            ax.set_xlim(vmin + 0.9, vmax)
            rs2.append(xaxis.get_majorticklocs()[0])
            mpl.pyplot.close(ax.get_figure())

        assert rs1 == xpl1
        assert rs2 == xpl2

    def test_finder_quarterly(self):
        yrs = [3.5, 11]

        xpl1 = xpl2 = [Period("1988Q1").ordinal] * len(yrs)
        rs1 = []
        rs2 = []
        for n in yrs:
            rng = period_range("1987Q2", periods=int(n * 4), freq="Q")
            ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            _, ax = mpl.pyplot.subplots()
            ser.plot(ax=ax)
            xaxis = ax.get_xaxis()
            rs1.append(xaxis.get_majorticklocs()[0])

            (vmin, vmax) = ax.get_xlim()
            ax.set_xlim(vmin + 0.9, vmax)
            rs2.append(xaxis.get_majorticklocs()[0])
            mpl.pyplot.close(ax.get_figure())

        assert rs1 == xpl1
        assert rs2 == xpl2

    def test_finder_monthly(self):
        yrs = [1.15, 2.5, 4, 11]

        xpl1 = xpl2 = [Period("Jan 1988").ordinal] * len(yrs)
        rs1 = []
        rs2 = []
        for n in yrs:
            rng = period_range("1987Q2", periods=int(n * 12), freq="M")
            ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            _, ax = mpl.pyplot.subplots()
            ser.plot(ax=ax)
            xaxis = ax.get_xaxis()
            rs1.append(xaxis.get_majorticklocs()[0])

            vmin, vmax = ax.get_xlim()
            ax.set_xlim(vmin + 0.9, vmax)
            rs2.append(xaxis.get_majorticklocs()[0])
            mpl.pyplot.close(ax.get_figure())

        assert rs1 == xpl1
        assert rs2 == xpl2

    def test_finder_monthly_long(self):
        rng = period_range("1988Q1", periods=24 * 12, freq="M")
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        xaxis = ax.get_xaxis()
        rs = xaxis.get_majorticklocs()[0]
        xp = Period("1989Q1", "M").ordinal
        assert rs == xp

    def test_finder_annual(self):
        xp = [1987, 1988, 1990, 1990, 1995, 2020, 2070, 2170]
        xp = [Period(x, freq="Y").ordinal for x in xp]
        rs = []
        for nyears in [5, 10, 19, 49, 99, 199, 599, 1001]:
            rng = period_range("1987", periods=nyears, freq="Y")
            ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            _, ax = mpl.pyplot.subplots()
            ser.plot(ax=ax)
            xaxis = ax.get_xaxis()
            rs.append(xaxis.get_majorticklocs()[0])
            mpl.pyplot.close(ax.get_figure())

        assert rs == xp

    @pytest.mark.slow
    def test_finder_minutely(self):
        nminutes = 50 * 24 * 60
        rng = date_range("1/1/1999", freq="Min", periods=nminutes)
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        xaxis = ax.get_xaxis()
        rs = xaxis.get_majorticklocs()[0]
        xp = Period("1/1/1999", freq="Min").ordinal

        assert rs == xp

    def test_finder_hourly(self):
        nhours = 23
        rng = date_range("1/1/1999", freq="h", periods=nhours)
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        xaxis = ax.get_xaxis()
        rs = xaxis.get_majorticklocs()[0]
        xp = Period("1/1/1999", freq="h").ordinal

        assert rs == xp

    def test_gaps(self):
        ts = Series(
            np.arange(30, dtype=np.float64), index=date_range("2020-01-01", periods=30)
        )
        ts.iloc[5:25] = np.nan
        _, ax = mpl.pyplot.subplots()
        ts.plot(ax=ax)
        lines = ax.get_lines()
        assert len(lines) == 1
        line = lines[0]
        data = line.get_xydata()

        data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)

        assert isinstance(data, np.ma.core.MaskedArray)
        mask = data.mask
        assert mask[5:25, 1].all()
        mpl.pyplot.close(ax.get_figure())

    def test_gaps_irregular(self):
        # irregular
        ts = Series(
            np.arange(30, dtype=np.float64), index=date_range("2020-01-01", periods=30)
        )
        ts = ts.iloc[[0, 1, 2, 5, 7, 9, 12, 15, 20]]
        ts.iloc[2:5] = np.nan
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot(ax=ax)
        lines = ax.get_lines()
        assert len(lines) == 1
        line = lines[0]
        data = line.get_xydata()

        data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)

        assert isinstance(data, np.ma.core.MaskedArray)
        mask = data.mask
        assert mask[2:5, 1].all()
        mpl.pyplot.close(ax.get_figure())

    def test_gaps_non_ts(self):
        # non-ts
        idx = [0, 1, 2, 5, 7, 9, 12, 15, 20]
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        ser.iloc[2:5] = np.nan
        _, ax = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        lines = ax.get_lines()
        assert len(lines) == 1
        line = lines[0]
        data = line.get_xydata()
        data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)

        assert isinstance(data, np.ma.core.MaskedArray)
        mask = data.mask
        assert mask[2:5, 1].all()

    def test_gap_upsample(self):
        low = Series(
            np.arange(30, dtype=np.float64), index=date_range("2020-01-01", periods=30)
        )
        low.iloc[5:25] = np.nan
        _, ax = mpl.pyplot.subplots()
        low.plot(ax=ax)

        idxh = date_range(low.index[0], low.index[-1], freq="12h")
        s = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        s.plot(secondary_y=True)
        lines = ax.get_lines()
        assert len(lines) == 1
        assert len(ax.right_ax.get_lines()) == 1

        line = lines[0]
        data = line.get_xydata()
        data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)

        assert isinstance(data, np.ma.core.MaskedArray)
        mask = data.mask
        assert mask[5:25, 1].all()

    def test_secondary_y(self):
        ser = Series(np.random.default_rng(2).standard_normal(10))
        fig, _ = mpl.pyplot.subplots()
        ax = ser.plot(secondary_y=True)
        assert hasattr(ax, "left_ax")
        assert not hasattr(ax, "right_ax")
        axes = fig.get_axes()
        line = ax.get_lines()[0]
        xp = Series(line.get_ydata(), line.get_xdata())
        tm.assert_series_equal(ser, xp)
        assert ax.get_yaxis().get_ticks_position() == "right"
        assert not axes[0].get_yaxis().get_visible()
        mpl.pyplot.close(fig)

    def test_secondary_y_yaxis(self):
        Series(np.random.default_rng(2).standard_normal(10))
        ser2 = Series(np.random.default_rng(2).standard_normal(10))
        _, ax2 = mpl.pyplot.subplots()
        ser2.plot(ax=ax2)
        assert ax2.get_yaxis().get_ticks_position() == "left"
        mpl.pyplot.close(ax2.get_figure())

    def test_secondary_both(self):
        ser = Series(np.random.default_rng(2).standard_normal(10))
        ser2 = Series(np.random.default_rng(2).standard_normal(10))
        ax = ser2.plot()
        ax2 = ser.plot(secondary_y=True)
        assert ax.get_yaxis().get_visible()
        assert not hasattr(ax, "left_ax")
        assert hasattr(ax, "right_ax")
        assert hasattr(ax2, "left_ax")
        assert not hasattr(ax2, "right_ax")

    def test_secondary_y_ts(self):
        idx = date_range("1/1/2000", periods=10)
        ser = Series(np.random.default_rng(2).standard_normal(10), idx)
        fig, _ = mpl.pyplot.subplots()
        ax = ser.plot(secondary_y=True)
        assert hasattr(ax, "left_ax")
        assert not hasattr(ax, "right_ax")
        axes = fig.get_axes()
        line = ax.get_lines()[0]
        xp = Series(line.get_ydata(), line.get_xdata()).to_timestamp()
        tm.assert_series_equal(ser, xp)
        assert ax.get_yaxis().get_ticks_position() == "right"
        assert not axes[0].get_yaxis().get_visible()
        mpl.pyplot.close(fig)

    def test_secondary_y_ts_yaxis(self):
        idx = date_range("1/1/2000", periods=10)
        ser2 = Series(np.random.default_rng(2).standard_normal(10), idx)
        _, ax2 = mpl.pyplot.subplots()
        ser2.plot(ax=ax2)
        assert ax2.get_yaxis().get_ticks_position() == "left"
        mpl.pyplot.close(ax2.get_figure())

    def test_secondary_y_ts_visible(self):
        idx = date_range("1/1/2000", periods=10)
        ser2 = Series(np.random.default_rng(2).standard_normal(10), idx)
        ax = ser2.plot()
        assert ax.get_yaxis().get_visible()

    def test_secondary_kde(self):
        pytest.importorskip("scipy")
        ser = Series(np.random.default_rng(2).standard_normal(10))
        fig, ax = mpl.pyplot.subplots()
        ax = ser.plot(secondary_y=True, kind="density", ax=ax)
        assert hasattr(ax, "left_ax")
        assert not hasattr(ax, "right_ax")
        axes = fig.get_axes()
        assert axes[1].get_yaxis().get_ticks_position() == "right"

    def test_secondary_bar(self):
        ser = Series(np.random.default_rng(2).standard_normal(10))
        fig, ax = mpl.pyplot.subplots()
        ser.plot(secondary_y=True, kind="bar", ax=ax)
        axes = fig.get_axes()
        assert axes[1].get_yaxis().get_ticks_position() == "right"

    def test_secondary_frame(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), columns=["a", "b", "c"]
        )
        axes = df.plot(secondary_y=["a", "c"], subplots=True)
        assert axes[0].get_yaxis().get_ticks_position() == "right"
        assert axes[1].get_yaxis().get_ticks_position() == "left"
        assert axes[2].get_yaxis().get_ticks_position() == "right"

    def test_secondary_bar_frame(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), columns=["a", "b", "c"]
        )
        axes = df.plot(kind="bar", secondary_y=["a", "c"], subplots=True)
        assert axes[0].get_yaxis().get_ticks_position() == "right"
        assert axes[1].get_yaxis().get_ticks_position() == "left"
        assert axes[2].get_yaxis().get_ticks_position() == "right"

    def test_mixed_freq_regular_first(self):
        # TODO
        s1 = Series(
            np.arange(20, dtype=np.float64),
            index=date_range("2020-01-01", periods=20, freq="B"),
        )
        s2 = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15]]

        # it works!
        _, ax = mpl.pyplot.subplots()
        s1.plot(ax=ax)

        ax2 = s2.plot(style="g", ax=ax)
        lines = ax2.get_lines()
        msg = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            idx1 = PeriodIndex(lines[0].get_xdata())
            idx2 = PeriodIndex(lines[1].get_xdata())

            tm.assert_index_equal(idx1, s1.index.to_period("B"))
            tm.assert_index_equal(idx2, s2.index.to_period("B"))

            left, right = ax2.get_xlim()
            pidx = s1.index.to_period()
        assert left <= pidx[0].ordinal
        assert right >= pidx[-1].ordinal

    def test_mixed_freq_irregular_first(self):
        s1 = Series(
            np.arange(20, dtype=np.float64), index=date_range("2020-01-01", periods=20)
        )
        s2 = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15]]
        _, ax = mpl.pyplot.subplots()
        s2.plot(style="g", ax=ax)
        s1.plot(ax=ax)
        assert not hasattr(ax, "freq")
        lines = ax.get_lines()
        x1 = lines[0].get_xdata()
        tm.assert_numpy_array_equal(x1, s2.index.astype(object).values)
        x2 = lines[1].get_xdata()
        tm.assert_numpy_array_equal(x2, s1.index.astype(object).values)

    def test_mixed_freq_regular_first_df(self):
        # GH 9852
        s1 = Series(
            np.arange(20, dtype=np.float64),
            index=date_range("2020-01-01", periods=20, freq="B"),
        ).to_frame()
        s2 = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15], :]
        _, ax = mpl.pyplot.subplots()
        s1.plot(ax=ax)
        ax2 = s2.plot(style="g", ax=ax)
        lines = ax2.get_lines()
        msg = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            idx1 = PeriodIndex(lines[0].get_xdata())
            idx2 = PeriodIndex(lines[1].get_xdata())
            assert idx1.equals(s1.index.to_period("B"))
            assert idx2.equals(s2.index.to_period("B"))
            left, right = ax2.get_xlim()
            pidx = s1.index.to_period()
        assert left <= pidx[0].ordinal
        assert right >= pidx[-1].ordinal

    def test_mixed_freq_irregular_first_df(self):
        # GH 9852
        s1 = Series(
            np.arange(20, dtype=np.float64), index=date_range("2020-01-01", periods=20)
        ).to_frame()
        s2 = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15], :]
        _, ax = mpl.pyplot.subplots()
        s2.plot(style="g", ax=ax)
        s1.plot(ax=ax)
        assert not hasattr(ax, "freq")
        lines = ax.get_lines()
        x1 = lines[0].get_xdata()
        tm.assert_numpy_array_equal(x1, s2.index.astype(object).values)
        x2 = lines[1].get_xdata()
        tm.assert_numpy_array_equal(x2, s1.index.astype(object).values)

    def test_mixed_freq_hf_first(self):
        idxh = date_range("1/1/1999", periods=365, freq="D")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax = mpl.pyplot.subplots()
        high.plot(ax=ax)
        low.plot(ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == "D"

    def test_mixed_freq_alignment(self):
        ts_ind = date_range("2012-01-01 13:00", "2012-01-02", freq="h")
        ts_data = np.random.default_rng(2).standard_normal(12)

        ts = Series(ts_data, index=ts_ind)
        ts2 = ts.asfreq("min").interpolate()

        _, ax = mpl.pyplot.subplots()
        ax = ts.plot(ax=ax)
        ts2.plot(style="r", ax=ax)

        assert ax.lines[0].get_xdata()[0] == ax.lines[1].get_xdata()[0]

    def test_mixed_freq_lf_first(self):
        idxh = date_range("1/1/1999", periods=365, freq="D")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax = mpl.pyplot.subplots()
        low.plot(legend=True, ax=ax)
        high.plot(legend=True, ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == "D"
        leg = ax.get_legend()
        assert len(leg.texts) == 2
        mpl.pyplot.close(ax.get_figure())

    def test_mixed_freq_lf_first_hourly(self):
        idxh = date_range("1/1/1999", periods=240, freq="min")
        idxl = date_range("1/1/1999", periods=4, freq="h")
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax = mpl.pyplot.subplots()
        low.plot(ax=ax)
        high.plot(ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == "min"

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_mixed_freq_irreg_period(self):
        ts = Series(
            np.arange(30, dtype=np.float64), index=date_range("2020-01-01", periods=30)
        )
        irreg = ts.iloc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 29]]
        msg = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rng = period_range("1/3/2000", periods=30, freq="B")
        ps = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax = mpl.pyplot.subplots()
        irreg.plot(ax=ax)
        ps.plot(ax=ax)

    def test_mixed_freq_shared_ax(self):
        # GH13341, using sharex=True
        idx1 = date_range("2015-01-01", periods=3, freq="ME")
        idx2 = idx1[:1].union(idx1[2:])
        s1 = Series(range(len(idx1)), idx1)
        s2 = Series(range(len(idx2)), idx2)

        _, (ax1, ax2) = mpl.pyplot.subplots(nrows=2, sharex=True)
        s1.plot(ax=ax1)
        s2.plot(ax=ax2)

        assert ax1.freq == "M"
        assert ax2.freq == "M"
        assert ax1.lines[0].get_xydata()[0, 0] == ax2.lines[0].get_xydata()[0, 0]

    def test_mixed_freq_shared_ax_twin_x(self):
        # GH13341, using sharex=True
        idx1 = date_range("2015-01-01", periods=3, freq="ME")
        idx2 = idx1[:1].union(idx1[2:])
        s1 = Series(range(len(idx1)), idx1)
        s2 = Series(range(len(idx2)), idx2)
        # using twinx
        _, ax1 = mpl.pyplot.subplots()
        ax2 = ax1.twinx()
        s1.plot(ax=ax1)
        s2.plot(ax=ax2)

        assert ax1.lines[0].get_xydata()[0, 0] == ax2.lines[0].get_xydata()[0, 0]

    @pytest.mark.xfail(reason="TODO (GH14330, GH14322)")
    def test_mixed_freq_shared_ax_twin_x_irregular_first(self):
        # GH13341, using sharex=True
        idx1 = date_range("2015-01-01", periods=3, freq="M")
        idx2 = idx1[:1].union(idx1[2:])
        s1 = Series(range(len(idx1)), idx1)
        s2 = Series(range(len(idx2)), idx2)
        _, ax1 = mpl.pyplot.subplots()
        ax2 = ax1.twinx()
        s2.plot(ax=ax1)
        s1.plot(ax=ax2)
        assert ax1.lines[0].get_xydata()[0, 0] == ax2.lines[0].get_xydata()[0, 0]

    def test_nat_handling(self):
        _, ax = mpl.pyplot.subplots()

        dti = DatetimeIndex(["2015-01-01", NaT, "2015-01-03"])
        s = Series(range(len(dti)), dti)
        s.plot(ax=ax)
        xdata = ax.get_lines()[0].get_xdata()
        # plot x data is bounded by index values
        assert s.index.min() <= Series(xdata).min()
        assert Series(xdata).max() <= s.index.max()

    def test_to_weekly_resampling_disallow_how_kwd(self):
        idxh = date_range("1/1/1999", periods=52, freq="W")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax = mpl.pyplot.subplots()
        high.plot(ax=ax)

        msg = (
            "'how' is not a valid keyword for plotting functions. If plotting "
            "multiple objects on shared axes, resample manually first."
        )
        with pytest.raises(ValueError, match=msg):
            low.plot(ax=ax, how="foo")

    def test_to_weekly_resampling(self):
        idxh = date_range("1/1/1999", periods=52, freq="W")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax = mpl.pyplot.subplots()
        high.plot(ax=ax)
        low.plot(ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq

    def test_from_weekly_resampling(self):
        idxh = date_range("1/1/1999", periods=52, freq="W")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax = mpl.pyplot.subplots()
        low.plot(ax=ax)
        high.plot(ax=ax)

        expected_h = idxh.to_period().asi8.astype(np.float64)
        expected_l = np.array(
            [1514, 1519, 1523, 1527, 1531, 1536, 1540, 1544, 1549, 1553, 1558, 1562],
            dtype=np.float64,
        )
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq
            xdata = line.get_xdata(orig=False)
            if len(xdata) == 12:  # idxl lines
                tm.assert_numpy_array_equal(xdata, expected_l)
            else:
                tm.assert_numpy_array_equal(xdata, expected_h)

    @pytest.mark.parametrize("kind1, kind2", [("line", "area"), ("area", "line")])
    def test_from_resampling_area_line_mixed(self, kind1, kind2):
        idxh = date_range("1/1/1999", periods=52, freq="W")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        high = DataFrame(
            np.random.default_rng(2).random((len(idxh), 3)),
            index=idxh,
            columns=[0, 1, 2],
        )
        low = DataFrame(
            np.random.default_rng(2).random((len(idxl), 3)),
            index=idxl,
            columns=[0, 1, 2],
        )

        _, ax = mpl.pyplot.subplots()
        low.plot(kind=kind1, stacked=True, ax=ax)
        high.plot(kind=kind2, stacked=True, ax=ax)

        # check low dataframe result
        expected_x = np.array(
            [
                1514,
                1519,
                1523,
                1527,
                1531,
                1536,
                1540,
                1544,
                1549,
                1553,
                1558,
                1562,
            ],
            dtype=np.float64,
        )
        expected_y = np.zeros(len(expected_x), dtype=np.float64)
        for i in range(3):
            line = ax.lines[i]
            assert PeriodIndex(line.get_xdata()).freq == idxh.freq
            tm.assert_numpy_array_equal(line.get_xdata(orig=False), expected_x)
            # check stacked values are correct
            expected_y += low[i].values
            tm.assert_numpy_array_equal(line.get_ydata(orig=False), expected_y)

        # check high dataframe result
        expected_x = idxh.to_period().asi8.astype(np.float64)
        expected_y = np.zeros(len(expected_x), dtype=np.float64)
        for i in range(3):
            line = ax.lines[3 + i]
            assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq
            tm.assert_numpy_array_equal(line.get_xdata(orig=False), expected_x)
            expected_y += high[i].values
            tm.assert_numpy_array_equal(line.get_ydata(orig=False), expected_y)

    @pytest.mark.parametrize("kind1, kind2", [("line", "area"), ("area", "line")])
    def test_from_resampling_area_line_mixed_high_to_low(self, kind1, kind2):
        idxh = date_range("1/1/1999", periods=52, freq="W")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        high = DataFrame(
            np.random.default_rng(2).random((len(idxh), 3)),
            index=idxh,
            columns=[0, 1, 2],
        )
        low = DataFrame(
            np.random.default_rng(2).random((len(idxl), 3)),
            index=idxl,
            columns=[0, 1, 2],
        )
        _, ax = mpl.pyplot.subplots()
        high.plot(kind=kind1, stacked=True, ax=ax)
        low.plot(kind=kind2, stacked=True, ax=ax)

        # check high dataframe result
        expected_x = idxh.to_period().asi8.astype(np.float64)
        expected_y = np.zeros(len(expected_x), dtype=np.float64)
        for i in range(3):
            line = ax.lines[i]
            assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq
            tm.assert_numpy_array_equal(line.get_xdata(orig=False), expected_x)
            expected_y += high[i].values
            tm.assert_numpy_array_equal(line.get_ydata(orig=False), expected_y)

        # check low dataframe result
        expected_x = np.array(
            [
                1514,
                1519,
                1523,
                1527,
                1531,
                1536,
                1540,
                1544,
                1549,
                1553,
                1558,
                1562,
            ],
            dtype=np.float64,
        )
        expected_y = np.zeros(len(expected_x), dtype=np.float64)
        for i in range(3):
            lines = ax.lines[3 + i]
            assert PeriodIndex(data=lines.get_xdata()).freq == idxh.freq
            tm.assert_numpy_array_equal(lines.get_xdata(orig=False), expected_x)
            expected_y += low[i].values
            tm.assert_numpy_array_equal(lines.get_ydata(orig=False), expected_y)

    def test_mixed_freq_second_millisecond(self):
        # GH 7772, GH 7760
        idxh = date_range("2014-07-01 09:00", freq="s", periods=50)
        idxl = date_range("2014-07-01 09:00", freq="100ms", periods=500)
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        # high to low
        _, ax = mpl.pyplot.subplots()
        high.plot(ax=ax)
        low.plot(ax=ax)
        assert len(ax.get_lines()) == 2
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == "ms"

    def test_mixed_freq_second_millisecond_low_to_high(self):
        # GH 7772, GH 7760
        idxh = date_range("2014-07-01 09:00", freq="s", periods=50)
        idxl = date_range("2014-07-01 09:00", freq="100ms", periods=500)
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        # low to high
        _, ax = mpl.pyplot.subplots()
        low.plot(ax=ax)
        high.plot(ax=ax)
        assert len(ax.get_lines()) == 2
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == "ms"

    def test_irreg_dtypes(self):
        # date
        idx = [date(2000, 1, 1), date(2000, 1, 5), date(2000, 1, 20)]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)),
            Index(idx, dtype=object),
        )
        _check_plot_works(df.plot)

    def test_irreg_dtypes_dt64(self):
        # np.datetime64
        idx = date_range("1/1/2000", periods=10)
        idx = idx[[0, 2, 5, 9]].astype(object)
        df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), idx)
        _, ax = mpl.pyplot.subplots()
        _check_plot_works(df.plot, ax=ax)

    def test_time(self):
        t = datetime(1, 1, 1, 3, 30, 0)
        deltas = np.random.default_rng(2).integers(1, 20, 3).cumsum()
        ts = np.array([(t + timedelta(minutes=int(x))).time() for x in deltas])
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(len(ts)),
                "b": np.random.default_rng(2).standard_normal(len(ts)),
            },
            index=ts,
        )
        _, ax = mpl.pyplot.subplots()
        df.plot(ax=ax)

        # verify tick labels
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        for _tick, _label in zip(ticks, labels):
            m, s = divmod(int(_tick), 60)
            h, m = divmod(m, 60)
            rs = _label.get_text()
            if len(rs) > 0:
                if s != 0:
                    xp = time(h, m, s).strftime("%H:%M:%S")
                else:
                    xp = time(h, m, s).strftime("%H:%M")
                assert xp == rs

    def test_time_change_xlim(self):
        t = datetime(1, 1, 1, 3, 30, 0)
        deltas = np.random.default_rng(2).integers(1, 20, 3).cumsum()
        ts = np.array([(t + timedelta(minutes=int(x))).time() for x in deltas])
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(len(ts)),
                "b": np.random.default_rng(2).standard_normal(len(ts)),
            },
            index=ts,
        )
        _, ax = mpl.pyplot.subplots()
        df.plot(ax=ax)

        # verify tick labels
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        for _tick, _label in zip(ticks, labels):
            m, s = divmod(int(_tick), 60)
            h, m = divmod(m, 60)
            rs = _label.get_text()
            if len(rs) > 0:
                if s != 0:
                    xp = time(h, m, s).strftime("%H:%M:%S")
                else:
                    xp = time(h, m, s).strftime("%H:%M")
                assert xp == rs

        # change xlim
        ax.set_xlim("1:30", "5:00")

        # check tick labels again
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        for _tick, _label in zip(ticks, labels):
            m, s = divmod(int(_tick), 60)
            h, m = divmod(m, 60)
            rs = _label.get_text()
            if len(rs) > 0:
                if s != 0:
                    xp = time(h, m, s).strftime("%H:%M:%S")
                else:
                    xp = time(h, m, s).strftime("%H:%M")
                assert xp == rs

    def test_time_musec(self):
        t = datetime(1, 1, 1, 3, 30, 0)
        deltas = np.random.default_rng(2).integers(1, 20, 3).cumsum()
        ts = np.array([(t + timedelta(microseconds=int(x))).time() for x in deltas])
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(len(ts)),
                "b": np.random.default_rng(2).standard_normal(len(ts)),
            },
            index=ts,
        )
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(ax=ax)

        # verify tick labels
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        for _tick, _label in zip(ticks, labels):
            m, s = divmod(int(_tick), 60)

            us = round((_tick - int(_tick)) * 1e6)

            h, m = divmod(m, 60)
            rs = _label.get_text()
            if len(rs) > 0:
                if (us % 1000) != 0:
                    xp = time(h, m, s, us).strftime("%H:%M:%S.%f")
                elif (us // 1000) != 0:
                    xp = time(h, m, s, us).strftime("%H:%M:%S.%f")[:-3]
                elif s != 0:
                    xp = time(h, m, s, us).strftime("%H:%M:%S")
                else:
                    xp = time(h, m, s, us).strftime("%H:%M")
                assert xp == rs

    def test_secondary_upsample(self):
        idxh = date_range("1/1/1999", periods=365, freq="D")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax = mpl.pyplot.subplots()
        low.plot(ax=ax)
        ax = high.plot(secondary_y=True, ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(line.get_xdata()).freq == "D"
        assert hasattr(ax, "left_ax")
        assert not hasattr(ax, "right_ax")
        for line in ax.left_ax.get_lines():
            assert PeriodIndex(line.get_xdata()).freq == "D"

    def test_secondary_legend(self):
        fig = mpl.pyplot.figure()
        ax = fig.add_subplot(211)

        # ts
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        df.plot(secondary_y=["A", "B"], ax=ax)
        leg = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert leg.get_texts()[0].get_text() == "A (right)"
        assert leg.get_texts()[1].get_text() == "B (right)"
        assert leg.get_texts()[2].get_text() == "C"
        assert leg.get_texts()[3].get_text() == "D"
        assert ax.right_ax.get_legend() is None
        colors = set()
        for line in leg.get_lines():
            colors.add(line.get_color())

        # TODO: color cycle problems
        assert len(colors) == 4
        mpl.pyplot.close(fig)

    def test_secondary_legend_right(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        fig = mpl.pyplot.figure()
        ax = fig.add_subplot(211)
        df.plot(secondary_y=["A", "C"], mark_right=False, ax=ax)
        leg = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert leg.get_texts()[0].get_text() == "A"
        assert leg.get_texts()[1].get_text() == "B"
        assert leg.get_texts()[2].get_text() == "C"
        assert leg.get_texts()[3].get_text() == "D"
        mpl.pyplot.close(fig)

    def test_secondary_legend_bar(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        fig, ax = mpl.pyplot.subplots()
        df.plot(kind="bar", secondary_y=["A"], ax=ax)
        leg = ax.get_legend()
        assert leg.get_texts()[0].get_text() == "A (right)"
        assert leg.get_texts()[1].get_text() == "B"
        mpl.pyplot.close(fig)

    def test_secondary_legend_bar_right(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        fig, ax = mpl.pyplot.subplots()
        df.plot(kind="bar", secondary_y=["A"], mark_right=False, ax=ax)
        leg = ax.get_legend()
        assert leg.get_texts()[0].get_text() == "A"
        assert leg.get_texts()[1].get_text() == "B"
        mpl.pyplot.close(fig)

    def test_secondary_legend_multi_col(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        fig = mpl.pyplot.figure()
        ax = fig.add_subplot(211)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        ax = df.plot(secondary_y=["C", "D"], ax=ax)
        leg = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert ax.right_ax.get_legend() is None
        colors = set()
        for line in leg.get_lines():
            colors.add(line.get_color())

        # TODO: color cycle problems
        assert len(colors) == 4
        mpl.pyplot.close(fig)

    def test_secondary_legend_nonts(self):
        # non-ts
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        fig = mpl.pyplot.figure()
        ax = fig.add_subplot(211)
        ax = df.plot(secondary_y=["A", "B"], ax=ax)
        leg = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert ax.right_ax.get_legend() is None
        colors = set()
        for line in leg.get_lines():
            colors.add(line.get_color())

        # TODO: color cycle problems
        assert len(colors) == 4
        mpl.pyplot.close()

    def test_secondary_legend_nonts_multi_col(self):
        # non-ts
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        fig = mpl.pyplot.figure()
        ax = fig.add_subplot(211)
        ax = df.plot(secondary_y=["C", "D"], ax=ax)
        leg = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert ax.right_ax.get_legend() is None
        colors = set()
        for line in leg.get_lines():
            colors.add(line.get_color())

        # TODO: color cycle problems
        assert len(colors) == 4

    @pytest.mark.xfail(reason="Api changed in 3.6.0")
    def test_format_date_axis(self):
        rng = date_range("1/1/2012", periods=12, freq="ME")
        df = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), rng)
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(ax=ax)
        xaxis = ax.get_xaxis()
        for line in xaxis.get_ticklabels():
            if len(line.get_text()) > 0:
                assert line.get_rotation() == 30

    def test_ax_plot(self):
        x = date_range(start="2012-01-02", periods=10, freq="D")
        y = list(range(len(x)))
        _, ax = mpl.pyplot.subplots()
        lines = ax.plot(x, y, label="Y")
        tm.assert_index_equal(DatetimeIndex(lines[0].get_xdata()), x)

    def test_mpl_nopandas(self):
        dates = [date(2008, 12, 31), date(2009, 1, 31)]
        values1 = np.arange(10.0, 11.0, 0.5)
        values2 = np.arange(11.0, 12.0, 0.5)

        _, ax = mpl.pyplot.subplots()
        (
            line1,
            line2,
        ) = ax.plot(
            [x.toordinal() for x in dates],
            values1,
            "-",
            [x.toordinal() for x in dates],
            values2,
            "-",
            linewidth=4,
        )

        exp = np.array([x.toordinal() for x in dates], dtype=np.float64)
        tm.assert_numpy_array_equal(line1.get_xydata()[:, 0], exp)
        exp = np.array([x.toordinal() for x in dates], dtype=np.float64)
        tm.assert_numpy_array_equal(line2.get_xydata()[:, 0], exp)

    def test_irregular_ts_shared_ax_xlim(self):
        # GH 2960
        from pandas.plotting._matplotlib.converter import DatetimeConverter

        ts = Series(
            np.arange(20, dtype=np.float64), index=date_range("2020-01-01", periods=20)
        )
        ts_irregular = ts.iloc[[1, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 17, 18]]

        # plot the left section of the irregular series, then the right section
        _, ax = mpl.pyplot.subplots()
        ts_irregular[:5].plot(ax=ax)
        ts_irregular[5:].plot(ax=ax)

        # check that axis limits are correct
        left, right = ax.get_xlim()
        assert left <= DatetimeConverter.convert(ts_irregular.index.min(), "", ax)
        assert right >= DatetimeConverter.convert(ts_irregular.index.max(), "", ax)

    def test_secondary_y_non_ts_xlim(self):
        # GH 3490 - non-timeseries with secondary y
        index_1 = [1, 2, 3, 4]
        index_2 = [5, 6, 7, 8]
        s1 = Series(1, index=index_1)
        s2 = Series(2, index=index_2)

        _, ax = mpl.pyplot.subplots()
        s1.plot(ax=ax)
        left_before, right_before = ax.get_xlim()
        s2.plot(secondary_y=True, ax=ax)
        left_after, right_after = ax.get_xlim()

        assert left_before >= left_after
        assert right_before < right_after

    def test_secondary_y_regular_ts_xlim(self):
        # GH 3490 - regular-timeseries with secondary y
        index_1 = date_range(start="2000-01-01", periods=4, freq="D")
        index_2 = date_range(start="2000-01-05", periods=4, freq="D")
        s1 = Series(1, index=index_1)
        s2 = Series(2, index=index_2)

        _, ax = mpl.pyplot.subplots()
        s1.plot(ax=ax)
        left_before, right_before = ax.get_xlim()
        s2.plot(secondary_y=True, ax=ax)
        left_after, right_after = ax.get_xlim()

        assert left_before >= left_after
        assert right_before < right_after

    def test_secondary_y_mixed_freq_ts_xlim(self):
        # GH 3490 - mixed frequency timeseries with secondary y
        rng = date_range("2000-01-01", periods=10000, freq="min")
        ts = Series(1, index=rng)

        _, ax = mpl.pyplot.subplots()
        ts.plot(ax=ax)
        left_before, right_before = ax.get_xlim()
        ts.resample("D").mean().plot(secondary_y=True, ax=ax)
        left_after, right_after = ax.get_xlim()

        # a downsample should not have changed either limit
        assert left_before == left_after
        assert right_before == right_after

    def test_secondary_y_irregular_ts_xlim(self):
        # GH 3490 - irregular-timeseries with secondary y
        from pandas.plotting._matplotlib.converter import DatetimeConverter

        ts = Series(
            np.arange(20, dtype=np.float64), index=date_range("2020-01-01", periods=20)
        )
        ts_irregular = ts.iloc[[1, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 17, 18]]

        _, ax = mpl.pyplot.subplots()
        ts_irregular[:5].plot(ax=ax)
        # plot higher-x values on secondary axis
        ts_irregular[5:].plot(secondary_y=True, ax=ax)
        # ensure secondary limits aren't overwritten by plot on primary
        ts_irregular[:5].plot(ax=ax)

        left, right = ax.get_xlim()
        assert left <= DatetimeConverter.convert(ts_irregular.index.min(), "", ax)
        assert right >= DatetimeConverter.convert(ts_irregular.index.max(), "", ax)

    def test_plot_outofbounds_datetime(self):
        # 2579 - checking this does not raise
        values = [date(1677, 1, 1), date(1677, 1, 2)]
        _, ax = mpl.pyplot.subplots()
        ax.plot(values)

        values = [datetime(1677, 1, 1, 12), datetime(1677, 1, 2, 12)]
        ax.plot(values)

    def test_format_timedelta_ticks_narrow(self):
        expected_labels = [f"00:00:00.0000000{i:0>2d}" for i in np.arange(10)]

        rng = timedelta_range("0", periods=10, freq="ns")
        df = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), rng)
        _, ax = mpl.pyplot.subplots()
        df.plot(fontsize=2, ax=ax)
        mpl.pyplot.draw()
        labels = ax.get_xticklabels()

        result_labels = [x.get_text() for x in labels]
        assert len(result_labels) == len(expected_labels)
        assert result_labels == expected_labels

    def test_format_timedelta_ticks_wide(self):
        expected_labels = [
            "00:00:00",
            "1 days 03:46:40",
            "2 days 07:33:20",
            "3 days 11:20:00",
            "4 days 15:06:40",
            "5 days 18:53:20",
            "6 days 22:40:00",
            "8 days 02:26:40",
            "9 days 06:13:20",
        ]

        rng = timedelta_range("0", periods=10, freq="1 d")
        df = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), rng)
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(fontsize=2, ax=ax)
        mpl.pyplot.draw()
        labels = ax.get_xticklabels()

        result_labels = [x.get_text() for x in labels]
        assert len(result_labels) == len(expected_labels)
        assert result_labels == expected_labels

    def test_timedelta_plot(self):
        # test issue #8711
        s = Series(range(5), timedelta_range("1day", periods=5))
        _, ax = mpl.pyplot.subplots()
        _check_plot_works(s.plot, ax=ax)

    def test_timedelta_long_period(self):
        # test long period
        index = timedelta_range("1 day 2 hr 30 min 10 s", periods=10, freq="1 d")
        s = Series(np.random.default_rng(2).standard_normal(len(index)), index)
        _, ax = mpl.pyplot.subplots()
        _check_plot_works(s.plot, ax=ax)

    def test_timedelta_short_period(self):
        # test short period
        index = timedelta_range("1 day 2 hr 30 min 10 s", periods=10, freq="1 ns")
        s = Series(np.random.default_rng(2).standard_normal(len(index)), index)
        _, ax = mpl.pyplot.subplots()
        _check_plot_works(s.plot, ax=ax)

    def test_hist(self):
        # https://github.com/matplotlib/matplotlib/issues/8459
        rng = date_range("1/1/2011", periods=10, freq="h")
        x = rng
        w1 = np.arange(0, 1, 0.1)
        w2 = np.arange(0, 1, 0.1)[::-1]
        _, ax = mpl.pyplot.subplots()
        ax.hist([x, x], weights=[w1, w2])

    def test_overlapping_datetime(self):
        # GB 6608
        s1 = Series(
            [1, 2, 3],
            index=[
                datetime(1995, 12, 31),
                datetime(2000, 12, 31),
                datetime(2005, 12, 31),
            ],
        )
        s2 = Series(
            [1, 2, 3],
            index=[
                datetime(1997, 12, 31),
                datetime(2003, 12, 31),
                datetime(2008, 12, 31),
            ],
        )

        # plot first series, then add the second series to those axes,
        # then try adding the first series again
        _, ax = mpl.pyplot.subplots()
        s1.plot(ax=ax)
        s2.plot(ax=ax)
        s1.plot(ax=ax)

    @pytest.mark.xfail(reason="GH9053 matplotlib does not use ax.xaxis.converter")
    def test_add_matplotlib_datetime64(self):
        # GH9053 - ensure that a plot with PeriodConverter still understands
        # datetime64 data. This still fails because matplotlib overrides the
        # ax.xaxis.converter with a DatetimeConverter
        s = Series(
            np.random.default_rng(2).standard_normal(10),
            index=date_range("1970-01-02", periods=10),
        )
        ax = s.plot()
        with tm.assert_produces_warning(DeprecationWarning):
            # multi-dimensional indexing
            ax.plot(s.index, s.values, color="g")
        l1, l2 = ax.lines
        tm.assert_numpy_array_equal(l1.get_xydata(), l2.get_xydata())

    def test_matplotlib_scatter_datetime64(self):
        # https://github.com/matplotlib/matplotlib/issues/11391
        df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=["x", "y"])
        df["time"] = date_range("2018-01-01", periods=10, freq="D")
        _, ax = mpl.pyplot.subplots()
        ax.scatter(x="time", y="y", data=df)
        mpl.pyplot.draw()
        label = ax.get_xticklabels()[0]
        expected = "2018-01-01"
        assert label.get_text() == expected

    def test_check_xticks_rot(self):
        # https://github.com/pandas-dev/pandas/issues/29460
        # regular time series
        x = to_datetime(["2020-05-01", "2020-05-02", "2020-05-03"])
        df = DataFrame({"x": x, "y": [1, 2, 3]})
        axes = df.plot(x="x", y="y")
        _check_ticks_props(axes, xrot=0)

    def test_check_xticks_rot_irregular(self):
        # irregular time series
        x = to_datetime(["2020-05-01", "2020-05-02", "2020-05-04"])
        df = DataFrame({"x": x, "y": [1, 2, 3]})
        axes = df.plot(x="x", y="y")
        _check_ticks_props(axes, xrot=30)

    def test_check_xticks_rot_use_idx(self):
        # irregular time series
        x = to_datetime(["2020-05-01", "2020-05-02", "2020-05-04"])
        df = DataFrame({"x": x, "y": [1, 2, 3]})
        # use timeseries index or not
        axes = df.set_index("x").plot(y="y", use_index=True)
        _check_ticks_props(axes, xrot=30)
        axes = df.set_index("x").plot(y="y", use_index=False)
        _check_ticks_props(axes, xrot=0)

    def test_check_xticks_rot_sharex(self):
        # irregular time series
        x = to_datetime(["2020-05-01", "2020-05-02", "2020-05-04"])
        df = DataFrame({"x": x, "y": [1, 2, 3]})
        # separate subplots
        axes = df.plot(x="x", y="y", subplots=True, sharex=True)
        _check_ticks_props(axes, xrot=30)
        axes = df.plot(x="x", y="y", subplots=True, sharex=False)
        _check_ticks_props(axes, xrot=0)


def _check_plot_works(f, freq=None, series=None, *args, **kwargs):
    import matplotlib.pyplot as plt

    fig = plt.gcf()

    try:
        plt.clf()
        ax = fig.add_subplot(211)
        orig_ax = kwargs.pop("ax", plt.gca())
        orig_axfreq = getattr(orig_ax, "freq", None)

        ret = f(*args, **kwargs)
        assert ret is not None  # do something more intelligent

        ax = kwargs.pop("ax", plt.gca())
        if series is not None:
            dfreq = series.index.freq
            if isinstance(dfreq, BaseOffset):
                dfreq = dfreq.rule_code
            if orig_axfreq is None:
                assert ax.freq == dfreq

        if freq is not None:
            ax_freq = to_offset(ax.freq, is_period=True)
        if freq is not None and orig_axfreq is None:
            assert ax_freq == freq

        ax = fig.add_subplot(212)
        kwargs["ax"] = ax
        ret = f(*args, **kwargs)
        assert ret is not None  # TODO: do something more intelligent

        # GH18439, GH#24088, statsmodels#4772
        with tm.ensure_clean(return_filelike=True) as path:
            pickle.dump(fig, path)
    finally:
        plt.close(fig)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\itsdangerous\encoding.py ===
from __future__ import annotations

import base64
import string
import struct
import typing as t

from .exc import BadData


def want_bytes(
    s: str | bytes, encoding: str = "utf-8", errors: str = "strict"
) -> bytes:
    if isinstance(s, str):
        s = s.encode(encoding, errors)

    return s


def base64_encode(string: str | bytes) -> bytes:
    """Base64 encode a string of bytes or text. The resulting bytes are
    safe to use in URLs.
    """
    string = want_bytes(string)
    return base64.urlsafe_b64encode(string).rstrip(b"=")


def base64_decode(string: str | bytes) -> bytes:
    """Base64 decode a URL-safe string of bytes or text. The result is
    bytes.
    """
    string = want_bytes(string, encoding="ascii", errors="ignore")
    string += b"=" * (-len(string) % 4)

    try:
        return base64.urlsafe_b64decode(string)
    except (TypeError, ValueError) as e:
        raise BadData("Invalid base64-encoded data") from e


# The alphabet used by base64.urlsafe_*
_base64_alphabet = f"{string.ascii_letters}{string.digits}-_=".encode("ascii")

_int64_struct = struct.Struct(">Q")
_int_to_bytes = _int64_struct.pack
_bytes_to_int = t.cast("t.Callable[[bytes], tuple[int]]", _int64_struct.unpack)


def int_to_bytes(num: int) -> bytes:
    return _int_to_bytes(num).lstrip(b"\x00")


def bytes_to_int(bytestr: bytes) -> int:
    return _bytes_to_int(bytestr.rjust(8, b"\x00"))[0]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\stock_chart.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Sequence,
    Alias,
)
from openpyxl.descriptors.excel import ExtensionList

from ._chart import ChartBase
from .axis import TextAxis, NumericAxis, ChartLines
from .updown_bars import UpDownBars
from .label import DataLabelList
from .series import Series


class StockChart(ChartBase):

    tagname = "stockChart"

    ser = Sequence(expected_type=Series) #min 3, max4
    dLbls = Typed(expected_type=DataLabelList, allow_none=True)
    dataLabels = Alias('dLbls')
    dropLines = Typed(expected_type=ChartLines, allow_none=True)
    hiLowLines = Typed(expected_type=ChartLines, allow_none=True)
    upDownBars = Typed(expected_type=UpDownBars, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    x_axis = Typed(expected_type=TextAxis)
    y_axis = Typed(expected_type=NumericAxis)

    _series_type = "line"

    __elements__ = ('ser', 'dLbls', 'dropLines', 'hiLowLines', 'upDownBars',
                    'axId')

    def __init__(self,
                 ser=(),
                 dLbls=None,
                 dropLines=None,
                 hiLowLines=None,
                 upDownBars=None,
                 extLst=None,
                 **kw
                ):
        self.ser = ser
        self.dLbls = dLbls
        self.dropLines = dropLines
        self.hiLowLines = hiLowLines
        self.upDownBars = upDownBars
        self.x_axis = TextAxis()
        self.y_axis = NumericAxis()
        super().__init__(**kw)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\compat\__init__.py ===
# Copyright (c) 2010-2024 openpyxl

from .numbers import NUMERIC_TYPES
from .strings import safe_string

import warnings
from functools import wraps
import inspect


class DummyCode:

    pass


# from https://github.com/tantale/deprecated/blob/master/deprecated/__init__.py
# with an enhancement to update docstrings of deprecated functions
string_types = (type(b''), type(u''))
def deprecated(reason):

    if isinstance(reason, string_types):

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @wraps(func1)
            def new_func1(*args, **kwargs):
                #warnings.simplefilter('default', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                return func1(*args, **kwargs)

            # Enhance docstring with a deprecation note
            deprecationNote = "\n\n.. note::\n    Deprecated: " + reason
            if new_func1.__doc__:
                new_func1.__doc__ += deprecationNote
            else:
                new_func1.__doc__ = deprecationNote
            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):
        raise TypeError("Reason for deprecation must be supplied")

    else:
        raise TypeError(repr(type(reason)))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\distro\__init__.py ===
from .distro import (
    NORMALIZED_DISTRO_ID,
    NORMALIZED_LSB_ID,
    NORMALIZED_OS_ID,
    LinuxDistribution,
    __version__,
    build_number,
    codename,
    distro_release_attr,
    distro_release_info,
    id,
    info,
    like,
    linux_distribution,
    lsb_release_attr,
    lsb_release_info,
    major_version,
    minor_version,
    name,
    os_release_attr,
    os_release_info,
    uname_attr,
    uname_info,
    version,
    version_parts,
)

__all__ = [
    "NORMALIZED_DISTRO_ID",
    "NORMALIZED_LSB_ID",
    "NORMALIZED_OS_ID",
    "LinuxDistribution",
    "build_number",
    "codename",
    "distro_release_attr",
    "distro_release_info",
    "id",
    "info",
    "like",
    "linux_distribution",
    "lsb_release_attr",
    "lsb_release_info",
    "major_version",
    "minor_version",
    "name",
    "os_release_attr",
    "os_release_info",
    "uname_attr",
    "uname_info",
    "version",
    "version_parts",
]

__version__ = __version__

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\styles\_mapping.py ===
# Automatically generated by scripts/gen_mapfiles.py.
# DO NOT EDIT BY HAND; run `tox -e mapfiles` instead.

STYLES = {
    'AbapStyle': ('pygments.styles.abap', 'abap', ()),
    'AlgolStyle': ('pygments.styles.algol', 'algol', ()),
    'Algol_NuStyle': ('pygments.styles.algol_nu', 'algol_nu', ()),
    'ArduinoStyle': ('pygments.styles.arduino', 'arduino', ()),
    'AutumnStyle': ('pygments.styles.autumn', 'autumn', ()),
    'BlackWhiteStyle': ('pygments.styles.bw', 'bw', ()),
    'BorlandStyle': ('pygments.styles.borland', 'borland', ()),
    'CoffeeStyle': ('pygments.styles.coffee', 'coffee', ()),
    'ColorfulStyle': ('pygments.styles.colorful', 'colorful', ()),
    'DefaultStyle': ('pygments.styles.default', 'default', ()),
    'DraculaStyle': ('pygments.styles.dracula', 'dracula', ()),
    'EmacsStyle': ('pygments.styles.emacs', 'emacs', ()),
    'FriendlyGrayscaleStyle': ('pygments.styles.friendly_grayscale', 'friendly_grayscale', ()),
    'FriendlyStyle': ('pygments.styles.friendly', 'friendly', ()),
    'FruityStyle': ('pygments.styles.fruity', 'fruity', ()),
    'GhDarkStyle': ('pygments.styles.gh_dark', 'github-dark', ()),
    'GruvboxDarkStyle': ('pygments.styles.gruvbox', 'gruvbox-dark', ()),
    'GruvboxLightStyle': ('pygments.styles.gruvbox', 'gruvbox-light', ()),
    'IgorStyle': ('pygments.styles.igor', 'igor', ()),
    'InkPotStyle': ('pygments.styles.inkpot', 'inkpot', ()),
    'LightbulbStyle': ('pygments.styles.lightbulb', 'lightbulb', ()),
    'LilyPondStyle': ('pygments.styles.lilypond', 'lilypond', ()),
    'LovelaceStyle': ('pygments.styles.lovelace', 'lovelace', ()),
    'ManniStyle': ('pygments.styles.manni', 'manni', ()),
    'MaterialStyle': ('pygments.styles.material', 'material', ()),
    'MonokaiStyle': ('pygments.styles.monokai', 'monokai', ()),
    'MurphyStyle': ('pygments.styles.murphy', 'murphy', ()),
    'NativeStyle': ('pygments.styles.native', 'native', ()),
    'NordDarkerStyle': ('pygments.styles.nord', 'nord-darker', ()),
    'NordStyle': ('pygments.styles.nord', 'nord', ()),
    'OneDarkStyle': ('pygments.styles.onedark', 'one-dark', ()),
    'ParaisoDarkStyle': ('pygments.styles.paraiso_dark', 'paraiso-dark', ()),
    'ParaisoLightStyle': ('pygments.styles.paraiso_light', 'paraiso-light', ()),
    'PastieStyle': ('pygments.styles.pastie', 'pastie', ()),
    'PerldocStyle': ('pygments.styles.perldoc', 'perldoc', ()),
    'RainbowDashStyle': ('pygments.styles.rainbow_dash', 'rainbow_dash', ()),
    'RrtStyle': ('pygments.styles.rrt', 'rrt', ()),
    'SasStyle': ('pygments.styles.sas', 'sas', ()),
    'SolarizedDarkStyle': ('pygments.styles.solarized', 'solarized-dark', ()),
    'SolarizedLightStyle': ('pygments.styles.solarized', 'solarized-light', ()),
    'StarofficeStyle': ('pygments.styles.staroffice', 'staroffice', ()),
    'StataDarkStyle': ('pygments.styles.stata_dark', 'stata-dark', ()),
    'StataLightStyle': ('pygments.styles.stata_light', 'stata-light', ()),
    'TangoStyle': ('pygments.styles.tango', 'tango', ()),
    'TracStyle': ('pygments.styles.trac', 'trac', ()),
    'VimStyle': ('pygments.styles.vim', 'vim', ()),
    'VisualStudioStyle': ('pygments.styles.vs', 'vs', ()),
    'XcodeStyle': ('pygments.styles.xcode', 'xcode', ()),
    'ZenburnStyle': ('pygments.styles.zenburn', 'zenburn', ()),
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\screen.py ===
from typing import Optional, TYPE_CHECKING

from .segment import Segment
from .style import StyleType
from ._loop import loop_last


if TYPE_CHECKING:
    from .console import (
        Console,
        ConsoleOptions,
        RenderResult,
        RenderableType,
        Group,
    )


class Screen:
    """A renderable that fills the terminal screen and crops excess.

    Args:
        renderable (RenderableType): Child renderable.
        style (StyleType, optional): Optional background style. Defaults to None.
    """

    renderable: "RenderableType"

    def __init__(
        self,
        *renderables: "RenderableType",
        style: Optional[StyleType] = None,
        application_mode: bool = False,
    ) -> None:
        from pip._vendor.rich.console import Group

        self.renderable = Group(*renderables)
        self.style = style
        self.application_mode = application_mode

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        width, height = options.size
        style = console.get_style(self.style) if self.style else None
        render_options = options.update(width=width, height=height)
        lines = console.render_lines(
            self.renderable or "", render_options, style=style, pad=True
        )
        lines = Segment.set_shape(lines, width, height, style=style)
        new_line = Segment("\n\r") if self.application_mode else Segment.line()
        for last, line in loop_last(lines):
            yield from line
            if not last:
                yield new_line

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\event_breakpoints.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: EventBreakpoints (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

def set_instrumentation_breakpoint(
        event_name: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on particular native event.

    :param event_name: Instrumentation name to stop on.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    cmd_dict: T_JSON_DICT = {
        'method': 'EventBreakpoints.setInstrumentationBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def remove_instrumentation_breakpoint(
        event_name: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes breakpoint on particular native event.

    :param event_name: Instrumentation name to stop on.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    cmd_dict: T_JSON_DICT = {
        'method': 'EventBreakpoints.removeInstrumentationBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes all breakpoints
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'EventBreakpoints.disable',
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\event_breakpoints.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: EventBreakpoints (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

def set_instrumentation_breakpoint(
        event_name: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on particular native event.

    :param event_name: Instrumentation name to stop on.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    cmd_dict: T_JSON_DICT = {
        'method': 'EventBreakpoints.setInstrumentationBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def remove_instrumentation_breakpoint(
        event_name: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes breakpoint on particular native event.

    :param event_name: Instrumentation name to stop on.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    cmd_dict: T_JSON_DICT = {
        'method': 'EventBreakpoints.removeInstrumentationBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes all breakpoints
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'EventBreakpoints.disable',
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\event_breakpoints.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: EventBreakpoints (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

def set_instrumentation_breakpoint(
        event_name: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on particular native event.

    :param event_name: Instrumentation name to stop on.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    cmd_dict: T_JSON_DICT = {
        'method': 'EventBreakpoints.setInstrumentationBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def remove_instrumentation_breakpoint(
        event_name: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes breakpoint on particular native event.

    :param event_name: Instrumentation name to stop on.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    cmd_dict: T_JSON_DICT = {
        'method': 'EventBreakpoints.removeInstrumentationBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes all breakpoints
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'EventBreakpoints.disable',
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\systems\mksa.py ===
"""
MKS unit system.

MKS stands for "meter, kilogram, second, ampere".
"""

from __future__ import annotations

from sympy.physics.units.definitions import Z0, ampere, coulomb, farad, henry, siemens, tesla, volt, weber, ohm
from sympy.physics.units.definitions.dimension_definitions import (
    capacitance, charge, conductance, current, impedance, inductance,
    magnetic_density, magnetic_flux, voltage)
from sympy.physics.units.prefixes import PREFIXES, prefix_unit
from sympy.physics.units.systems.mks import MKS, dimsys_length_weight_time
from sympy.physics.units.quantities import Quantity

dims = (voltage, impedance, conductance, current, capacitance, inductance, charge,
        magnetic_density, magnetic_flux)

units = [ampere, volt, ohm, siemens, farad, henry, coulomb, tesla, weber]

all_units: list[Quantity] = []
for u in units:
    all_units.extend(prefix_unit(u, PREFIXES))
all_units.extend(units)

all_units.append(Z0)

dimsys_MKSA = dimsys_length_weight_time.extend([
    # Dimensional dependencies for base dimensions (MKSA not in MKS)
    current,
], new_dim_deps={
    # Dimensional dependencies for derived dimensions
    "voltage": {"mass": 1, "length": 2, "current": -1, "time": -3},
    "impedance": {"mass": 1, "length": 2, "current": -2, "time": -3},
    "conductance": {"mass": -1, "length": -2, "current": 2, "time": 3},
    "capacitance": {"mass": -1, "length": -2, "current": 2, "time": 4},
    "inductance": {"mass": 1, "length": 2, "current": -2, "time": -2},
    "charge": {"current": 1, "time": 1},
    "magnetic_density": {"mass": 1, "current": -1, "time": -2},
    "magnetic_flux": {"length": 2, "mass": 1, "current": -1, "time": -2},
})

MKSA = MKS.extend(base=(ampere,), units=all_units, name='MKSA', dimension_system=dimsys_MKSA, derived_units={
    magnetic_flux: weber,
    impedance: ohm,
    current: ampere,
    voltage: volt,
    inductance: henry,
    conductance: siemens,
    magnetic_density: tesla,
    charge: coulomb,
    capacitance: farad,
})

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\numberfields\exceptions.py ===
"""Special exception classes for numberfields. """


class ClosureFailure(Exception):
    r"""
    Signals that a :py:class:`ModuleElement` which we tried to represent in a
    certain :py:class:`Module` cannot in fact be represented there.

    Examples
    ========

    >>> from sympy.polys import Poly, cyclotomic_poly, ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> from sympy.polys.numberfields.modules import PowerBasis, to_col
    >>> T = Poly(cyclotomic_poly(5))
    >>> A = PowerBasis(T)
    >>> B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))

    Because we are in a cyclotomic field, the power basis ``A`` is an integral
    basis, and the submodule ``B`` is just the ideal $(2)$. Therefore ``B`` can
    represent an element having all even coefficients over the power basis:

    >>> a1 = A(to_col([2, 4, 6, 8]))
    >>> print(B.represent(a1))
    DomainMatrix([[1], [2], [3], [4]], (4, 1), ZZ)

    but ``B`` cannot represent an element with an odd coefficient:

    >>> a2 = A(to_col([1, 2, 2, 2]))
    >>> B.represent(a2)
    Traceback (most recent call last):
    ...
    ClosureFailure: Element in QQ-span but not ZZ-span of this basis.

    """
    pass


class StructureError(Exception):
    r"""
    Represents cases in which an algebraic structure was expected to have a
    certain property, or be of a certain type, but was not.
    """
    pass


class MissingUnityError(StructureError):
    r"""Structure should contain a unity element but does not."""
    pass


__all__ = [
    'ClosureFailure', 'StructureError', 'MissingUnityError',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\pytest_plugin.py ===
from __future__ import annotations

import inspect
from typing import NoReturn

import pytest

from ..testing import MockClock, trio_test

RUN_SLOW = True
SKIP_OPTIONAL_IMPORTS = False


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", help="run slow tests")
    parser.addoption(
        "--skip-optional-imports",
        action="store_true",
        help="skip tests that rely on libraries not required by trio itself",
    )


def pytest_configure(config: pytest.Config) -> None:
    global RUN_SLOW
    RUN_SLOW = config.getoption("--run-slow", default=True)
    global SKIP_OPTIONAL_IMPORTS
    SKIP_OPTIONAL_IMPORTS = config.getoption("--skip-optional-imports", default=False)


@pytest.fixture
def mock_clock() -> MockClock:
    return MockClock()


@pytest.fixture
def autojump_clock() -> MockClock:
    return MockClock(autojump_threshold=0)


# FIXME: split off into a package (or just make part of Trio's public
# interface?), with config file to enable? and I guess a mark option too; I
# guess it's useful with the class- and file-level marking machinery (where
# the raw @trio_test decorator isn't enough).
@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> None:
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        pyfuncitem.obj = trio_test(pyfuncitem.obj)


def skip_if_optional_else_raise(error: ImportError) -> NoReturn:
    if SKIP_OPTIONAL_IMPORTS:
        pytest.skip(error.msg, allow_module_level=True)
    else:  # pragma: no cover
        raise error

# === atelier-kyo-manager/migrations\versions\d175a32efd8a_initial_migration.py ===
"""Initial migration

Revision ID: d175a32efd8a
Revises:
Create Date: 2025-05-27 06:58:46.907497

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd175a32efd8a'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('product',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=128), nullable=False),
    sa.Column('brand', sa.String(length=128), nullable=True),
    sa.Column('purchase_price', sa.Float(), nullable=False),
    sa.Column('selling_price', sa.Float(), nullable=False),
    sa.Column('supplier_url', sa.String(length=500), nullable=True),
    sa.Column('image_url', sa.String(length=500), nullable=True),
    sa.Column('stock_status', sa.Boolean(), nullable=True),
    sa.Column('profit', sa.Float(), nullable=True),
    sa.Column('transaction_fee', sa.Float(), nullable=True),
    sa.Column('shipping_cost', sa.Float(), nullable=True),
    sa.Column('customs_duty', sa.Float(), nullable=True),
    sa.Column('procurement_fee', sa.Float(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('listing',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('platform', sa.String(length=50), nullable=False),
    sa.Column('fee_rate', sa.Float(), nullable=True),
    sa.Column('product_id', sa.Integer(), nullable=True),
    sa.Column('listing_date', sa.DateTime(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.ForeignKeyConstraint(['product_id'], ['product.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('listing')
    op.drop_table('product')
    # ### end Alembic commands ###

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\random\tests\test_random.py ===
import sys
import warnings

import pytest

import numpy as np
from numpy import random
from numpy.testing import (
    IS_WASM,
    assert_,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_no_warnings,
    assert_raises,
    assert_warns,
    suppress_warnings,
)


class TestSeed:
    def test_scalar(self):
        s = np.random.RandomState(0)
        assert_equal(s.randint(1000), 684)
        s = np.random.RandomState(4294967295)
        assert_equal(s.randint(1000), 419)

    def test_array(self):
        s = np.random.RandomState(range(10))
        assert_equal(s.randint(1000), 468)
        s = np.random.RandomState(np.arange(10))
        assert_equal(s.randint(1000), 468)
        s = np.random.RandomState([0])
        assert_equal(s.randint(1000), 973)
        s = np.random.RandomState([4294967295])
        assert_equal(s.randint(1000), 265)

    def test_invalid_scalar(self):
        # seed must be an unsigned 32 bit integer
        assert_raises(TypeError, np.random.RandomState, -0.5)
        assert_raises(ValueError, np.random.RandomState, -1)

    def test_invalid_array(self):
        # seed must be an unsigned 32 bit integer
        assert_raises(TypeError, np.random.RandomState, [-0.5])
        assert_raises(ValueError, np.random.RandomState, [-1])
        assert_raises(ValueError, np.random.RandomState, [4294967296])
        assert_raises(ValueError, np.random.RandomState, [1, 2, 4294967296])
        assert_raises(ValueError, np.random.RandomState, [1, -2, 4294967296])

    def test_invalid_array_shape(self):
        # gh-9832
        assert_raises(ValueError, np.random.RandomState,
                      np.array([], dtype=np.int64))
        assert_raises(ValueError, np.random.RandomState, [[1, 2, 3]])
        assert_raises(ValueError, np.random.RandomState, [[1, 2, 3],
                                                          [4, 5, 6]])


class TestBinomial:
    def test_n_zero(self):
        # Tests the corner case of n == 0 for the binomial distribution.
        # binomial(0, p) should be zero for any p in [0, 1].
        # This test addresses issue #3480.
        zeros = np.zeros(2, dtype='int')
        for p in [0, .5, 1]:
            assert_(random.binomial(0, p) == 0)
            assert_array_equal(random.binomial(zeros, p), zeros)

    def test_p_is_nan(self):
        # Issue #4571.
        assert_raises(ValueError, random.binomial, 1, np.nan)


class TestMultinomial:
    def test_basic(self):
        random.multinomial(100, [0.2, 0.8])

    def test_zero_probability(self):
        random.multinomial(100, [0.2, 0.8, 0.0, 0.0, 0.0])

    def test_int_negative_interval(self):
        assert_(-5 <= random.randint(-5, -1) < -1)
        x = random.randint(-5, -1, 5)
        assert_(np.all(-5 <= x))
        assert_(np.all(x < -1))

    def test_size(self):
        # gh-3173
        p = [0.5, 0.5]
        assert_equal(np.random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(np.random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(np.random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(np.random.multinomial(1, p, [2, 2]).shape, (2, 2, 2))
        assert_equal(np.random.multinomial(1, p, (2, 2)).shape, (2, 2, 2))
        assert_equal(np.random.multinomial(1, p, np.array((2, 2))).shape,
                     (2, 2, 2))

        assert_raises(TypeError, np.random.multinomial, 1, p,
                      float(1))

    def test_multidimensional_pvals(self):
        assert_raises(ValueError, np.random.multinomial, 10, [[0, 1]])
        assert_raises(ValueError, np.random.multinomial, 10, [[0], [1]])
        assert_raises(ValueError, np.random.multinomial, 10, [[[0], [1]], [[1], [0]]])
        assert_raises(ValueError, np.random.multinomial, 10, np.array([[0, 1], [1, 0]]))


class TestSetState:
    def setup_method(self):
        self.seed = 1234567890
        self.prng = random.RandomState(self.seed)
        self.state = self.prng.get_state()

    def test_basic(self):
        old = self.prng.tomaxint(16)
        self.prng.set_state(self.state)
        new = self.prng.tomaxint(16)
        assert_(np.all(old == new))

    def test_gaussian_reset(self):
        # Make sure the cached every-other-Gaussian is reset.
        old = self.prng.standard_normal(size=3)
        self.prng.set_state(self.state)
        new = self.prng.standard_normal(size=3)
        assert_(np.all(old == new))

    def test_gaussian_reset_in_media_res(self):
        # When the state is saved with a cached Gaussian, make sure the
        # cached Gaussian is restored.

        self.prng.standard_normal()
        state = self.prng.get_state()
        old = self.prng.standard_normal(size=3)
        self.prng.set_state(state)
        new = self.prng.standard_normal(size=3)
        assert_(np.all(old == new))

    def test_backwards_compatibility(self):
        # Make sure we can accept old state tuples that do not have the
        # cached Gaussian value.
        old_state = self.state[:-2]
        x1 = self.prng.standard_normal(size=16)
        self.prng.set_state(old_state)
        x2 = self.prng.standard_normal(size=16)
        self.prng.set_state(self.state)
        x3 = self.prng.standard_normal(size=16)
        assert_(np.all(x1 == x2))
        assert_(np.all(x1 == x3))

    def test_negative_binomial(self):
        # Ensure that the negative binomial results take floating point
        # arguments without truncation.
        self.prng.negative_binomial(0.5, 0.5)

    def test_set_invalid_state(self):
        # gh-25402
        with pytest.raises(IndexError):
            self.prng.set_state(())


class TestRandint:

    rfunc = np.random.randint

    # valid integer/boolean types
    itype = [np.bool, np.int8, np.uint8, np.int16, np.uint16,
             np.int32, np.uint32, np.int64, np.uint64]

    def test_unsupported_type(self):
        assert_raises(TypeError, self.rfunc, 1, dtype=float)

    def test_bounds_checking(self):
        for dt in self.itype:
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1
            assert_raises(ValueError, self.rfunc, lbnd - 1, ubnd, dtype=dt)
            assert_raises(ValueError, self.rfunc, lbnd, ubnd + 1, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, lbnd, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, 0, dtype=dt)

    def test_rng_zero_and_extremes(self):
        for dt in self.itype:
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1

            tgt = ubnd - 1
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

            tgt = lbnd
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

            tgt = (lbnd + ubnd) // 2
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

    def test_full_range(self):
        # Test for ticket #1690

        for dt in self.itype:
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1

            try:
                self.rfunc(lbnd, ubnd, dtype=dt)
            except Exception as e:
                raise AssertionError("No error should have been raised, "
                                     "but one was with the following "
                                     "message:\n\n%s" % str(e))

    def test_in_bounds_fuzz(self):
        # Don't use fixed seed
        np.random.seed()

        for dt in self.itype[1:]:
            for ubnd in [4, 8, 16]:
                vals = self.rfunc(2, ubnd, size=2**16, dtype=dt)
                assert_(vals.max() < ubnd)
                assert_(vals.min() >= 2)

        vals = self.rfunc(0, 2, size=2**16, dtype=np.bool)

        assert_(vals.max() < 2)
        assert_(vals.min() >= 0)

    def test_repeatability(self):
        import hashlib
        # We use a sha256 hash of generated sequences of 1000 samples
        # in the range [0, 6) for all but bool, where the range
        # is [0, 2). Hashes are for little endian numbers.
        tgt = {'bool':   '509aea74d792fb931784c4b0135392c65aec64beee12b0cc167548a2c3d31e71',  # noqa: E501
               'int16':  '7b07f1a920e46f6d0fe02314155a2330bcfd7635e708da50e536c5ebb631a7d4',  # noqa: E501
               'int32':  'e577bfed6c935de944424667e3da285012e741892dcb7051a8f1ce68ab05c92f',  # noqa: E501
               'int64':  '0fbead0b06759df2cfb55e43148822d4a1ff953c7eb19a5b08445a63bb64fa9e',  # noqa: E501
               'int8':   '001aac3a5acb935a9b186cbe14a1ca064b8bb2dd0b045d48abeacf74d0203404',  # noqa: E501
               'uint16': '7b07f1a920e46f6d0fe02314155a2330bcfd7635e708da50e536c5ebb631a7d4',  # noqa: E501
               'uint32': 'e577bfed6c935de944424667e3da285012e741892dcb7051a8f1ce68ab05c92f',  # noqa: E501
               'uint64': '0fbead0b06759df2cfb55e43148822d4a1ff953c7eb19a5b08445a63bb64fa9e',  # noqa: E501
               'uint8':  '001aac3a5acb935a9b186cbe14a1ca064b8bb2dd0b045d48abeacf74d0203404'}  # noqa: E501

        for dt in self.itype[1:]:
            np.random.seed(1234)

            # view as little endian for hash
            if sys.byteorder == 'little':
                val = self.rfunc(0, 6, size=1000, dtype=dt)
            else:
                val = self.rfunc(0, 6, size=1000, dtype=dt).byteswap()

            res = hashlib.sha256(val.view(np.int8)).hexdigest()
            assert_(tgt[np.dtype(dt).name] == res)

        # bools do not depend on endianness
        np.random.seed(1234)
        val = self.rfunc(0, 2, size=1000, dtype=bool).view(np.int8)
        res = hashlib.sha256(val).hexdigest()
        assert_(tgt[np.dtype(bool).name] == res)

    def test_int64_uint64_corner_case(self):
        # When stored in Numpy arrays, `lbnd` is casted
        # as np.int64, and `ubnd` is casted as np.uint64.
        # Checking whether `lbnd` >= `ubnd` used to be
        # done solely via direct comparison, which is incorrect
        # because when Numpy tries to compare both numbers,
        # it casts both to np.float64 because there is
        # no integer superset of np.int64 and np.uint64. However,
        # `ubnd` is too large to be represented in np.float64,
        # causing it be round down to np.iinfo(np.int64).max,
        # leading to a ValueError because `lbnd` now equals
        # the new `ubnd`.

        dt = np.int64
        tgt = np.iinfo(np.int64).max
        lbnd = np.int64(np.iinfo(np.int64).max)
        ubnd = np.uint64(np.iinfo(np.int64).max + 1)

        # None of these function calls should
        # generate a ValueError now.
        actual = np.random.randint(lbnd, ubnd, dtype=dt)
        assert_equal(actual, tgt)

    def test_respect_dtype_singleton(self):
        # See gh-7203
        for dt in self.itype:
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1

            sample = self.rfunc(lbnd, ubnd, dtype=dt)
            assert_equal(sample.dtype, np.dtype(dt))

        for dt in (bool, int):
            # The legacy rng uses "long" as the default integer:
            lbnd = 0 if dt is bool else np.iinfo("long").min
            ubnd = 2 if dt is bool else np.iinfo("long").max + 1

            # gh-7284: Ensure that we get Python data types
            sample = self.rfunc(lbnd, ubnd, dtype=dt)
            assert_(not hasattr(sample, 'dtype'))
            assert_equal(type(sample), dt)


class TestRandomDist:
    # Make sure the random distribution returns the correct value for a
    # given seed

    def setup_method(self):
        self.seed = 1234567890

    def test_rand(self):
        np.random.seed(self.seed)
        actual = np.random.rand(3, 2)
        desired = np.array([[0.61879477158567997, 0.59162362775974664],
                            [0.88868358904449662, 0.89165480011560816],
                            [0.4575674820298663, 0.7781880808593471]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_randn(self):
        np.random.seed(self.seed)
        actual = np.random.randn(3, 2)
        desired = np.array([[1.34016345771863121, 1.73759122771936081],
                           [1.498988344300628, -0.2286433324536169],
                           [2.031033998682787, 2.17032494605655257]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_randint(self):
        np.random.seed(self.seed)
        actual = np.random.randint(-99, 99, size=(3, 2))
        desired = np.array([[31, 3],
                            [-52, 41],
                            [-48, -66]])
        assert_array_equal(actual, desired)

    def test_random_integers(self):
        np.random.seed(self.seed)
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            actual = np.random.random_integers(-99, 99, size=(3, 2))
            assert_(len(w) == 1)
        desired = np.array([[31, 3],
                            [-52, 41],
                            [-48, -66]])
        assert_array_equal(actual, desired)

    def test_random_integers_max_int(self):
        # Tests whether random_integers can generate the
        # maximum allowed Python int that can be converted
        # into a C long. Previous implementations of this
        # method have thrown an OverflowError when attempting
        # to generate this integer.
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            actual = np.random.random_integers(np.iinfo('l').max,
                                               np.iinfo('l').max)
            assert_(len(w) == 1)

        desired = np.iinfo('l').max
        assert_equal(actual, desired)

    def test_random_integers_deprecated(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            # DeprecationWarning raised with high == None
            assert_raises(DeprecationWarning,
                          np.random.random_integers,
                          np.iinfo('l').max)

            # DeprecationWarning raised with high != None
            assert_raises(DeprecationWarning,
                          np.random.random_integers,
                          np.iinfo('l').max, np.iinfo('l').max)

    def test_random(self):
        np.random.seed(self.seed)
        actual = np.random.random((3, 2))
        desired = np.array([[0.61879477158567997, 0.59162362775974664],
                            [0.88868358904449662, 0.89165480011560816],
                            [0.4575674820298663, 0.7781880808593471]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_choice_uniform_replace(self):
        np.random.seed(self.seed)
        actual = np.random.choice(4, 4)
        desired = np.array([2, 3, 2, 3])
        assert_array_equal(actual, desired)

    def test_choice_nonuniform_replace(self):
        np.random.seed(self.seed)
        actual = np.random.choice(4, 4, p=[0.4, 0.4, 0.1, 0.1])
        desired = np.array([1, 1, 2, 2])
        assert_array_equal(actual, desired)

    def test_choice_uniform_noreplace(self):
        np.random.seed(self.seed)
        actual = np.random.choice(4, 3, replace=False)
        desired = np.array([0, 1, 3])
        assert_array_equal(actual, desired)

    def test_choice_nonuniform_noreplace(self):
        np.random.seed(self.seed)
        actual = np.random.choice(4, 3, replace=False,
                                  p=[0.1, 0.3, 0.5, 0.1])
        desired = np.array([2, 3, 1])
        assert_array_equal(actual, desired)

    def test_choice_noninteger(self):
        np.random.seed(self.seed)
        actual = np.random.choice(['a', 'b', 'c', 'd'], 4)
        desired = np.array(['c', 'd', 'c', 'd'])
        assert_array_equal(actual, desired)

    def test_choice_exceptions(self):
        sample = np.random.choice
        assert_raises(ValueError, sample, -1, 3)
        assert_raises(ValueError, sample, 3., 3)
        assert_raises(ValueError, sample, [[1, 2], [3, 4]], 3)
        assert_raises(ValueError, sample, [], 3)
        assert_raises(ValueError, sample, [1, 2, 3, 4], 3,
                      p=[[0.25, 0.25], [0.25, 0.25]])
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4, 0.2])
        assert_raises(ValueError, sample, [1, 2], 3, p=[1.1, -0.1])
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4])
        assert_raises(ValueError, sample, [1, 2, 3], 4, replace=False)
        # gh-13087
        assert_raises(ValueError, sample, [1, 2, 3], -2, replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], (-1,), replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], (-1, 1), replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], 2,
                      replace=False, p=[1, 0, 0])

    def test_choice_return_shape(self):
        p = [0.1, 0.9]
        # Check scalar
        assert_(np.isscalar(np.random.choice(2, replace=True)))
        assert_(np.isscalar(np.random.choice(2, replace=False)))
        assert_(np.isscalar(np.random.choice(2, replace=True, p=p)))
        assert_(np.isscalar(np.random.choice(2, replace=False, p=p)))
        assert_(np.isscalar(np.random.choice([1, 2], replace=True)))
        assert_(np.random.choice([None], replace=True) is None)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(np.random.choice(arr, replace=True) is a)

        # Check 0-d array
        s = ()
        assert_(not np.isscalar(np.random.choice(2, s, replace=True)))
        assert_(not np.isscalar(np.random.choice(2, s, replace=False)))
        assert_(not np.isscalar(np.random.choice(2, s, replace=True, p=p)))
        assert_(not np.isscalar(np.random.choice(2, s, replace=False, p=p)))
        assert_(not np.isscalar(np.random.choice([1, 2], s, replace=True)))
        assert_(np.random.choice([None], s, replace=True).ndim == 0)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(np.random.choice(arr, s, replace=True).item() is a)

        # Check multi dimensional array
        s = (2, 3)
        p = [0.1, 0.1, 0.1, 0.1, 0.4, 0.2]
        assert_equal(np.random.choice(6, s, replace=True).shape, s)
        assert_equal(np.random.choice(6, s, replace=False).shape, s)
        assert_equal(np.random.choice(6, s, replace=True, p=p).shape, s)
        assert_equal(np.random.choice(6, s, replace=False, p=p).shape, s)
        assert_equal(np.random.choice(np.arange(6), s, replace=True).shape, s)

        # Check zero-size
        assert_equal(np.random.randint(0, 0, size=(3, 0, 4)).shape, (3, 0, 4))
        assert_equal(np.random.randint(0, -10, size=0).shape, (0,))
        assert_equal(np.random.randint(10, 10, size=0).shape, (0,))
        assert_equal(np.random.choice(0, size=0).shape, (0,))
        assert_equal(np.random.choice([], size=(0,)).shape, (0,))
        assert_equal(np.random.choice(['a', 'b'], size=(3, 0, 4)).shape,
                     (3, 0, 4))
        assert_raises(ValueError, np.random.choice, [], 10)

    def test_choice_nan_probabilities(self):
        a = np.array([42, 1, 2])
        p = [None, None, None]
        assert_raises(ValueError, np.random.choice, a, p=p)

    def test_bytes(self):
        np.random.seed(self.seed)
        actual = np.random.bytes(10)
        desired = b'\x82Ui\x9e\xff\x97+Wf\xa5'
        assert_equal(actual, desired)

    def test_shuffle(self):
        # Test lists, arrays (of various dtypes), and multidimensional versions
        # of both, c-contiguous or not:
        for conv in [lambda x: np.array([]),
                     lambda x: x,
                     lambda x: np.asarray(x).astype(np.int8),
                     lambda x: np.asarray(x).astype(np.float32),
                     lambda x: np.asarray(x).astype(np.complex64),
                     lambda x: np.asarray(x).astype(object),
                     lambda x: [(i, i) for i in x],
                     lambda x: np.asarray([[i, i] for i in x]),
                     lambda x: np.vstack([x, x]).T,
                     # gh-11442
                     lambda x: (np.asarray([(i, i) for i in x],
                                           [("a", int), ("b", int)])
                                .view(np.recarray)),
                     # gh-4270
                     lambda x: np.asarray([(i, i) for i in x],
                                          [("a", object), ("b", np.int32)])]:
            np.random.seed(self.seed)
            alist = conv([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
            np.random.shuffle(alist)
            actual = alist
            desired = conv([0, 1, 9, 6, 2, 4, 5, 8, 7, 3])
            assert_array_equal(actual, desired)

    def test_shuffle_masked(self):
        # gh-3263
        a = np.ma.masked_values(np.reshape(range(20), (5, 4)) % 3 - 1, -1)
        b = np.ma.masked_values(np.arange(20) % 3 - 1, -1)
        a_orig = a.copy()
        b_orig = b.copy()
        for i in range(50):
            np.random.shuffle(a)
            assert_equal(
                sorted(a.data[~a.mask]), sorted(a_orig.data[~a_orig.mask]))
            np.random.shuffle(b)
            assert_equal(
                sorted(b.data[~b.mask]), sorted(b_orig.data[~b_orig.mask]))

    @pytest.mark.parametrize("random",
            [np.random, np.random.RandomState(), np.random.default_rng()])
    def test_shuffle_untyped_warning(self, random):
        # Create a dict works like a sequence but isn't one
        values = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
        with pytest.warns(UserWarning,
                match="you are shuffling a 'dict' object") as rec:
            random.shuffle(values)
        assert "test_random" in rec[0].filename

    @pytest.mark.parametrize("random",
        [np.random, np.random.RandomState(), np.random.default_rng()])
    @pytest.mark.parametrize("use_array_like", [True, False])
    def test_shuffle_no_object_unpacking(self, random, use_array_like):
        class MyArr(np.ndarray):
            pass

        items = [
            None, np.array([3]), np.float64(3), np.array(10), np.float64(7)
        ]
        arr = np.array(items, dtype=object)
        item_ids = {id(i) for i in items}
        if use_array_like:
            arr = arr.view(MyArr)

        # The array was created fine, and did not modify any objects:
        assert all(id(i) in item_ids for i in arr)

        if use_array_like and not isinstance(random, np.random.Generator):
            # The old API gives incorrect results, but warns about it.
            with pytest.warns(UserWarning,
                    match="Shuffling a one dimensional array.*"):
                random.shuffle(arr)
        else:
            random.shuffle(arr)
            assert all(id(i) in item_ids for i in arr)

    def test_shuffle_memoryview(self):
        # gh-18273
        # allow graceful handling of memoryviews
        # (treat the same as arrays)
        np.random.seed(self.seed)
        a = np.arange(5).data
        np.random.shuffle(a)
        assert_equal(np.asarray(a), [0, 1, 4, 3, 2])
        rng = np.random.RandomState(self.seed)
        rng.shuffle(a)
        assert_equal(np.asarray(a), [0, 1, 2, 3, 4])
        rng = np.random.default_rng(self.seed)
        rng.shuffle(a)
        assert_equal(np.asarray(a), [4, 1, 0, 3, 2])

    def test_shuffle_not_writeable(self):
        a = np.zeros(3)
        a.flags.writeable = False
        with pytest.raises(ValueError, match='read-only'):
            np.random.shuffle(a)

    def test_beta(self):
        np.random.seed(self.seed)
        actual = np.random.beta(.1, .9, size=(3, 2))
        desired = np.array(
                [[1.45341850513746058e-02, 5.31297615662868145e-04],
                 [1.85366619058432324e-06, 4.19214516800110563e-03],
                 [1.58405155108498093e-04, 1.26252891949397652e-04]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_binomial(self):
        np.random.seed(self.seed)
        actual = np.random.binomial(100, .456, size=(3, 2))
        desired = np.array([[37, 43],
                            [42, 48],
                            [46, 45]])
        assert_array_equal(actual, desired)

    def test_chisquare(self):
        np.random.seed(self.seed)
        actual = np.random.chisquare(50, size=(3, 2))
        desired = np.array([[63.87858175501090585, 68.68407748911370447],
                            [65.77116116901505904, 47.09686762438974483],
                            [72.3828403199695174, 74.18408615260374006]])
        assert_array_almost_equal(actual, desired, decimal=13)

    def test_dirichlet(self):
        np.random.seed(self.seed)
        alpha = np.array([51.72840233779265162, 39.74494232180943953])
        actual = np.random.mtrand.dirichlet(alpha, size=(3, 2))
        desired = np.array([[[0.54539444573611562, 0.45460555426388438],
                             [0.62345816822039413, 0.37654183177960598]],
                            [[0.55206000085785778, 0.44793999914214233],
                             [0.58964023305154301, 0.41035976694845688]],
                            [[0.59266909280647828, 0.40733090719352177],
                             [0.56974431743975207, 0.43025568256024799]]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_dirichlet_size(self):
        # gh-3173
        p = np.array([51.72840233779265162, 39.74494232180943953])
        assert_equal(np.random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(np.random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(np.random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(np.random.dirichlet(p, [2, 2]).shape, (2, 2, 2))
        assert_equal(np.random.dirichlet(p, (2, 2)).shape, (2, 2, 2))
        assert_equal(np.random.dirichlet(p, np.array((2, 2))).shape, (2, 2, 2))

        assert_raises(TypeError, np.random.dirichlet, p, float(1))

    def test_dirichlet_bad_alpha(self):
        # gh-2089
        alpha = np.array([5.4e-01, -1.0e-16])
        assert_raises(ValueError, np.random.mtrand.dirichlet, alpha)

        # gh-15876
        assert_raises(ValueError, random.dirichlet, [[5, 1]])
        assert_raises(ValueError, random.dirichlet, [[5], [1]])
        assert_raises(ValueError, random.dirichlet, [[[5], [1]], [[1], [5]]])
        assert_raises(ValueError, random.dirichlet, np.array([[5, 1], [1, 5]]))

    def test_exponential(self):
        np.random.seed(self.seed)
        actual = np.random.exponential(1.1234, size=(3, 2))
        desired = np.array([[1.08342649775011624, 1.00607889924557314],
                            [2.46628830085216721, 2.49668106809923884],
                            [0.68717433461363442, 1.69175666993575979]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_exponential_0(self):
        assert_equal(np.random.exponential(scale=0), 0)
        assert_raises(ValueError, np.random.exponential, scale=-0.)

    def test_f(self):
        np.random.seed(self.seed)
        actual = np.random.f(12, 77, size=(3, 2))
        desired = np.array([[1.21975394418575878, 1.75135759791559775],
                            [1.44803115017146489, 1.22108959480396262],
                            [1.02176975757740629, 1.34431827623300415]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_gamma(self):
        np.random.seed(self.seed)
        actual = np.random.gamma(5, 3, size=(3, 2))
        desired = np.array([[24.60509188649287182, 28.54993563207210627],
                            [26.13476110204064184, 12.56988482927716078],
                            [31.71863275789960568, 33.30143302795922011]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_gamma_0(self):
        assert_equal(np.random.gamma(shape=0, scale=0), 0)
        assert_raises(ValueError, np.random.gamma, shape=-0., scale=-0.)

    def test_geometric(self):
        np.random.seed(self.seed)
        actual = np.random.geometric(.123456789, size=(3, 2))
        desired = np.array([[8, 7],
                            [17, 17],
                            [5, 12]])
        assert_array_equal(actual, desired)

    def test_gumbel(self):
        np.random.seed(self.seed)
        actual = np.random.gumbel(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[0.19591898743416816, 0.34405539668096674],
                            [-1.4492522252274278, -1.47374816298446865],
                            [1.10651090478803416, -0.69535848626236174]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_gumbel_0(self):
        assert_equal(np.random.gumbel(scale=0), 0)
        assert_raises(ValueError, np.random.gumbel, scale=-0.)

    def test_hypergeometric(self):
        np.random.seed(self.seed)
        actual = np.random.hypergeometric(10, 5, 14, size=(3, 2))
        desired = np.array([[10, 10],
                            [10, 10],
                            [9, 9]])
        assert_array_equal(actual, desired)

        # Test nbad = 0
        actual = np.random.hypergeometric(5, 0, 3, size=4)
        desired = np.array([3, 3, 3, 3])
        assert_array_equal(actual, desired)

        actual = np.random.hypergeometric(15, 0, 12, size=4)
        desired = np.array([12, 12, 12, 12])
        assert_array_equal(actual, desired)

        # Test ngood = 0
        actual = np.random.hypergeometric(0, 5, 3, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

        actual = np.random.hypergeometric(0, 15, 12, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

    def test_laplace(self):
        np.random.seed(self.seed)
        actual = np.random.laplace(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[0.66599721112760157, 0.52829452552221945],
                            [3.12791959514407125, 3.18202813572992005],
                            [-0.05391065675859356, 1.74901336242837324]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_laplace_0(self):
        assert_equal(np.random.laplace(scale=0), 0)
        assert_raises(ValueError, np.random.laplace, scale=-0.)

    def test_logistic(self):
        np.random.seed(self.seed)
        actual = np.random.logistic(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[1.09232835305011444, 0.8648196662399954],
                            [4.27818590694950185, 4.33897006346929714],
                            [-0.21682183359214885, 2.63373365386060332]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_lognormal(self):
        np.random.seed(self.seed)
        actual = np.random.lognormal(mean=.123456789, sigma=2.0, size=(3, 2))
        desired = np.array([[16.50698631688883822, 36.54846706092654784],
                            [22.67886599981281748, 0.71617561058995771],
                            [65.72798501792723869, 86.84341601437161273]])
        assert_array_almost_equal(actual, desired, decimal=13)

    def test_lognormal_0(self):
        assert_equal(np.random.lognormal(sigma=0), 1)
        assert_raises(ValueError, np.random.lognormal, sigma=-0.)

    def test_logseries(self):
        np.random.seed(self.seed)
        actual = np.random.logseries(p=.923456789, size=(3, 2))
        desired = np.array([[2, 2],
                            [6, 17],
                            [3, 6]])
        assert_array_equal(actual, desired)

    def test_multinomial(self):
        np.random.seed(self.seed)
        actual = np.random.multinomial(20, [1 / 6.] * 6, size=(3, 2))
        desired = np.array([[[4, 3, 5, 4, 2, 2],
                             [5, 2, 8, 2, 2, 1]],
                            [[3, 4, 3, 6, 0, 4],
                             [2, 1, 4, 3, 6, 4]],
                            [[4, 4, 2, 5, 2, 3],
                             [4, 3, 4, 2, 3, 4]]])
        assert_array_equal(actual, desired)

    def test_multivariate_normal(self):
        np.random.seed(self.seed)
        mean = (.123456789, 10)
        cov = [[1, 0], [0, 1]]
        size = (3, 2)
        actual = np.random.multivariate_normal(mean, cov, size)
        desired = np.array([[[1.463620246718631, 11.73759122771936],
                             [1.622445133300628, 9.771356667546383]],
                            [[2.154490787682787, 12.170324946056553],
                             [1.719909438201865, 9.230548443648306]],
                            [[0.689515026297799, 9.880729819607714],
                             [-0.023054015651998, 9.201096623542879]]])

        assert_array_almost_equal(actual, desired, decimal=15)

        # Check for default size, was raising deprecation warning
        actual = np.random.multivariate_normal(mean, cov)
        desired = np.array([0.895289569463708, 9.17180864067987])
        assert_array_almost_equal(actual, desired, decimal=15)

        # Check that non positive-semidefinite covariance warns with
        # RuntimeWarning
        mean = [0, 0]
        cov = [[1, 2], [2, 1]]
        assert_warns(RuntimeWarning, np.random.multivariate_normal, mean, cov)

        # and that it doesn't warn with RuntimeWarning check_valid='ignore'
        assert_no_warnings(np.random.multivariate_normal, mean, cov,
                           check_valid='ignore')

        # and that it raises with RuntimeWarning check_valid='raises'
        assert_raises(ValueError, np.random.multivariate_normal, mean, cov,
                      check_valid='raise')

        cov = np.array([[1, 0.1], [0.1, 1]], dtype=np.float32)
        with suppress_warnings() as sup:
            np.random.multivariate_normal(mean, cov)
            w = sup.record(RuntimeWarning)
            assert len(w) == 0

    def test_negative_binomial(self):
        np.random.seed(self.seed)
        actual = np.random.negative_binomial(n=100, p=.12345, size=(3, 2))
        desired = np.array([[848, 841],
                            [892, 611],
                            [779, 647]])
        assert_array_equal(actual, desired)

    def test_noncentral_chisquare(self):
        np.random.seed(self.seed)
        actual = np.random.noncentral_chisquare(df=5, nonc=5, size=(3, 2))
        desired = np.array([[23.91905354498517511, 13.35324692733826346],
                            [31.22452661329736401, 16.60047399466177254],
                            [5.03461598262724586, 17.94973089023519464]])
        assert_array_almost_equal(actual, desired, decimal=14)

        actual = np.random.noncentral_chisquare(df=.5, nonc=.2, size=(3, 2))
        desired = np.array([[1.47145377828516666,  0.15052899268012659],
                            [0.00943803056963588,  1.02647251615666169],
                            [0.332334982684171,  0.15451287602753125]])
        assert_array_almost_equal(actual, desired, decimal=14)

        np.random.seed(self.seed)
        actual = np.random.noncentral_chisquare(df=5, nonc=0, size=(3, 2))
        desired = np.array([[9.597154162763948, 11.725484450296079],
                            [10.413711048138335, 3.694475922923986],
                            [13.484222138963087, 14.377255424602957]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_noncentral_f(self):
        np.random.seed(self.seed)
        actual = np.random.noncentral_f(dfnum=5, dfden=2, nonc=1,
                                        size=(3, 2))
        desired = np.array([[1.40598099674926669, 0.34207973179285761],
                            [3.57715069265772545, 7.92632662577829805],
                            [0.43741599463544162, 1.1774208752428319]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_normal(self):
        np.random.seed(self.seed)
        actual = np.random.normal(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[2.80378370443726244, 3.59863924443872163],
                            [3.121433477601256, -0.33382987590723379],
                            [4.18552478636557357, 4.46410668111310471]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_normal_0(self):
        assert_equal(np.random.normal(scale=0), 0)
        assert_raises(ValueError, np.random.normal, scale=-0.)

    def test_pareto(self):
        np.random.seed(self.seed)
        actual = np.random.pareto(a=.123456789, size=(3, 2))
        desired = np.array(
                [[2.46852460439034849e+03, 1.41286880810518346e+03],
                 [5.28287797029485181e+07, 6.57720981047328785e+07],
                 [1.40840323350391515e+02, 1.98390255135251704e+05]])
        # For some reason on 32-bit x86 Ubuntu 12.10 the [1, 0] entry in this
        # matrix differs by 24 nulps. Discussion:
        #   https://mail.python.org/pipermail/numpy-discussion/2012-September/063801.html
        # Consensus is that this is probably some gcc quirk that affects
        # rounding but not in any important way, so we just use a looser
        # tolerance on this test:
        np.testing.assert_array_almost_equal_nulp(actual, desired, nulp=30)

    def test_poisson(self):
        np.random.seed(self.seed)
        actual = np.random.poisson(lam=.123456789, size=(3, 2))
        desired = np.array([[0, 0],
                            [1, 0],
                            [0, 0]])
        assert_array_equal(actual, desired)

    def test_poisson_exceptions(self):
        lambig = np.iinfo('l').max
        lamneg = -1
        assert_raises(ValueError, np.random.poisson, lamneg)
        assert_raises(ValueError, np.random.poisson, [lamneg] * 10)
        assert_raises(ValueError, np.random.poisson, lambig)
        assert_raises(ValueError, np.random.poisson, [lambig] * 10)

    def test_power(self):
        np.random.seed(self.seed)
        actual = np.random.power(a=.123456789, size=(3, 2))
        desired = np.array([[0.02048932883240791, 0.01424192241128213],
                            [0.38446073748535298, 0.39499689943484395],
                            [0.00177699707563439, 0.13115505880863756]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_rayleigh(self):
        np.random.seed(self.seed)
        actual = np.random.rayleigh(scale=10, size=(3, 2))
        desired = np.array([[13.8882496494248393, 13.383318339044731],
                            [20.95413364294492098, 21.08285015800712614],
                            [11.06066537006854311, 17.35468505778271009]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_rayleigh_0(self):
        assert_equal(np.random.rayleigh(scale=0), 0)
        assert_raises(ValueError, np.random.rayleigh, scale=-0.)

    def test_standard_cauchy(self):
        np.random.seed(self.seed)
        actual = np.random.standard_cauchy(size=(3, 2))
        desired = np.array([[0.77127660196445336, -6.55601161955910605],
                            [0.93582023391158309, -2.07479293013759447],
                            [-4.74601644297011926, 0.18338989290760804]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_exponential(self):
        np.random.seed(self.seed)
        actual = np.random.standard_exponential(size=(3, 2))
        desired = np.array([[0.96441739162374596, 0.89556604882105506],
                            [2.1953785836319808, 2.22243285392490542],
                            [0.6116915921431676, 1.50592546727413201]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_gamma(self):
        np.random.seed(self.seed)
        actual = np.random.standard_gamma(shape=3, size=(3, 2))
        desired = np.array([[5.50841531318455058, 6.62953470301903103],
                            [5.93988484943779227, 2.31044849402133989],
                            [7.54838614231317084, 8.012756093271868]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_standard_gamma_0(self):
        assert_equal(np.random.standard_gamma(shape=0), 0)
        assert_raises(ValueError, np.random.standard_gamma, shape=-0.)

    def test_standard_normal(self):
        np.random.seed(self.seed)
        actual = np.random.standard_normal(size=(3, 2))
        desired = np.array([[1.34016345771863121, 1.73759122771936081],
                            [1.498988344300628, -0.2286433324536169],
                            [2.031033998682787, 2.17032494605655257]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_t(self):
        np.random.seed(self.seed)
        actual = np.random.standard_t(df=10, size=(3, 2))
        desired = np.array([[0.97140611862659965, -0.08830486548450577],
                            [1.36311143689505321, -0.55317463909867071],
                            [-0.18473749069684214, 0.61181537341755321]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_triangular(self):
        np.random.seed(self.seed)
        actual = np.random.triangular(left=5.12, mode=10.23, right=20.34,
                                      size=(3, 2))
        desired = np.array([[12.68117178949215784, 12.4129206149193152],
                            [16.20131377335158263, 16.25692138747600524],
                            [11.20400690911820263, 14.4978144835829923]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_uniform(self):
        np.random.seed(self.seed)
        actual = np.random.uniform(low=1.23, high=10.54, size=(3, 2))
        desired = np.array([[6.99097932346268003, 6.73801597444323974],
                            [9.50364421400426274, 9.53130618907631089],
                            [5.48995325769805476, 8.47493103280052118]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_uniform_range_bounds(self):
        fmin = np.finfo('float').min
        fmax = np.finfo('float').max

        func = np.random.uniform
        assert_raises(OverflowError, func, -np.inf, 0)
        assert_raises(OverflowError, func,  0,      np.inf)
        assert_raises(OverflowError, func,  fmin,   fmax)
        assert_raises(OverflowError, func, [-np.inf], [0])
        assert_raises(OverflowError, func, [0], [np.inf])

        # (fmax / 1e17) - fmin is within range, so this should not throw
        # account for i386 extended precision DBL_MAX / 1e17 + DBL_MAX >
        # DBL_MAX by increasing fmin a bit
        np.random.uniform(low=np.nextafter(fmin, 1), high=fmax / 1e17)

    def test_scalar_exception_propagation(self):
        # Tests that exceptions are correctly propagated in distributions
        # when called with objects that throw exceptions when converted to
        # scalars.
        #
        # Regression test for gh: 8865

        class ThrowingFloat(np.ndarray):
            def __float__(self):
                raise TypeError

        throwing_float = np.array(1.0).view(ThrowingFloat)
        assert_raises(TypeError, np.random.uniform, throwing_float,
                      throwing_float)

        class ThrowingInteger(np.ndarray):
            def __int__(self):
                raise TypeError

            __index__ = __int__

        throwing_int = np.array(1).view(ThrowingInteger)
        assert_raises(TypeError, np.random.hypergeometric, throwing_int, 1, 1)

    def test_vonmises(self):
        np.random.seed(self.seed)
        actual = np.random.vonmises(mu=1.23, kappa=1.54, size=(3, 2))
        desired = np.array([[2.28567572673902042, 2.89163838442285037],
                            [0.38198375564286025, 2.57638023113890746],
                            [1.19153771588353052, 1.83509849681825354]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_vonmises_small(self):
        # check infinite loop, gh-4720
        np.random.seed(self.seed)
        r = np.random.vonmises(mu=0., kappa=1.1e-8, size=10**6)
        np.testing.assert_(np.isfinite(r).all())

    def test_wald(self):
        np.random.seed(self.seed)
        actual = np.random.wald(mean=1.23, scale=1.54, size=(3, 2))
        desired = np.array([[3.82935265715889983, 5.13125249184285526],
                            [0.35045403618358717, 1.50832396872003538],
                            [0.24124319895843183, 0.22031101461955038]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_weibull(self):
        np.random.seed(self.seed)
        actual = np.random.weibull(a=1.23, size=(3, 2))
        desired = np.array([[0.97097342648766727, 0.91422896443565516],
                            [1.89517770034962929, 1.91414357960479564],
                            [0.67057783752390987, 1.39494046635066793]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_weibull_0(self):
        np.random.seed(self.seed)
        assert_equal(np.random.weibull(a=0, size=12), np.zeros(12))
        assert_raises(ValueError, np.random.weibull, a=-0.)

    def test_zipf(self):
        np.random.seed(self.seed)
        actual = np.random.zipf(a=1.23, size=(3, 2))
        desired = np.array([[66, 29],
                            [1, 1],
                            [3, 13]])
        assert_array_equal(actual, desired)


class TestBroadcast:
    # tests that functions that broadcast behave
    # correctly when presented with non-scalar arguments
    def setup_method(self):
        self.seed = 123456789

    def setSeed(self):
        np.random.seed(self.seed)

    # TODO: Include test for randint once it can broadcast
    # Can steal the test written in PR #6938

    def test_uniform(self):
        low = [0]
        high = [1]
        uniform = np.random.uniform
        desired = np.array([0.53283302478975902,
                            0.53413660089041659,
                            0.50955303552646702])

        self.setSeed()
        actual = uniform(low * 3, high)
        assert_array_almost_equal(actual, desired, decimal=14)

        self.setSeed()
        actual = uniform(low, high * 3)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_normal(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        normal = np.random.normal
        desired = np.array([2.2129019979039612,
                            2.1283977976520019,
                            1.8417114045748335])

        self.setSeed()
        actual = normal(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, normal, loc * 3, bad_scale)

        self.setSeed()
        actual = normal(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, normal, loc, bad_scale * 3)

    def test_beta(self):
        a = [1]
        b = [2]
        bad_a = [-1]
        bad_b = [-2]
        beta = np.random.beta
        desired = np.array([0.19843558305989056,
                            0.075230336409423643,
                            0.24976865978980844])

        self.setSeed()
        actual = beta(a * 3, b)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, beta, bad_a * 3, b)
        assert_raises(ValueError, beta, a * 3, bad_b)

        self.setSeed()
        actual = beta(a, b * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, beta, bad_a, b * 3)
        assert_raises(ValueError, beta, a, bad_b * 3)

    def test_exponential(self):
        scale = [1]
        bad_scale = [-1]
        exponential = np.random.exponential
        desired = np.array([0.76106853658845242,
                            0.76386282278691653,
                            0.71243813125891797])

        self.setSeed()
        actual = exponential(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, exponential, bad_scale * 3)

    def test_standard_gamma(self):
        shape = [1]
        bad_shape = [-1]
        std_gamma = np.random.standard_gamma
        desired = np.array([0.76106853658845242,
                            0.76386282278691653,
                            0.71243813125891797])

        self.setSeed()
        actual = std_gamma(shape * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, std_gamma, bad_shape * 3)

    def test_gamma(self):
        shape = [1]
        scale = [2]
        bad_shape = [-1]
        bad_scale = [-2]
        gamma = np.random.gamma
        desired = np.array([1.5221370731769048,
                            1.5277256455738331,
                            1.4248762625178359])

        self.setSeed()
        actual = gamma(shape * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape * 3, scale)
        assert_raises(ValueError, gamma, shape * 3, bad_scale)

        self.setSeed()
        actual = gamma(shape, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape, scale * 3)
        assert_raises(ValueError, gamma, shape, bad_scale * 3)

    def test_f(self):
        dfnum = [1]
        dfden = [2]
        bad_dfnum = [-1]
        bad_dfden = [-2]
        f = np.random.f
        desired = np.array([0.80038951638264799,
                            0.86768719635363512,
                            2.7251095168386801])

        self.setSeed()
        actual = f(dfnum * 3, dfden)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, f, bad_dfnum * 3, dfden)
        assert_raises(ValueError, f, dfnum * 3, bad_dfden)

        self.setSeed()
        actual = f(dfnum, dfden * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, f, bad_dfnum, dfden * 3)
        assert_raises(ValueError, f, dfnum, bad_dfden * 3)

    def test_noncentral_f(self):
        dfnum = [2]
        dfden = [3]
        nonc = [4]
        bad_dfnum = [0]
        bad_dfden = [-1]
        bad_nonc = [-2]
        nonc_f = np.random.noncentral_f
        desired = np.array([9.1393943263705211,
                            13.025456344595602,
                            8.8018098359100545])

        self.setSeed()
        actual = nonc_f(dfnum * 3, dfden, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum * 3, dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, bad_dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, dfden, bad_nonc)

        self.setSeed()
        actual = nonc_f(dfnum, dfden * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, dfden * 3, bad_nonc)

        self.setSeed()
        actual = nonc_f(dfnum, dfden, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, dfden, bad_nonc * 3)

    def test_noncentral_f_small_df(self):
        self.setSeed()
        desired = np.array([6.869638627492048, 0.785880199263955])
        actual = np.random.noncentral_f(0.9, 0.9, 2, size=2)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_chisquare(self):
        df = [1]
        bad_df = [-1]
        chisquare = np.random.chisquare
        desired = np.array([0.57022801133088286,
                            0.51947702108840776,
                            0.1320969254923558])

        self.setSeed()
        actual = chisquare(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, chisquare, bad_df * 3)

    def test_noncentral_chisquare(self):
        df = [1]
        nonc = [2]
        bad_df = [-1]
        bad_nonc = [-2]
        nonc_chi = np.random.noncentral_chisquare
        desired = np.array([9.0015599467913763,
                            4.5804135049718742,
                            6.0872302432834564])

        self.setSeed()
        actual = nonc_chi(df * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df * 3, nonc)
        assert_raises(ValueError, nonc_chi, df * 3, bad_nonc)

        self.setSeed()
        actual = nonc_chi(df, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df, nonc * 3)
        assert_raises(ValueError, nonc_chi, df, bad_nonc * 3)

    def test_standard_t(self):
        df = [1]
        bad_df = [-1]
        t = np.random.standard_t
        desired = np.array([3.0702872575217643,
                            5.8560725167361607,
                            1.0274791436474273])

        self.setSeed()
        actual = t(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, t, bad_df * 3)

    def test_vonmises(self):
        mu = [2]
        kappa = [1]
        bad_kappa = [-1]
        vonmises = np.random.vonmises
        desired = np.array([2.9883443664201312,
                            -2.7064099483995943,
                            -1.8672476700665914])

        self.setSeed()
        actual = vonmises(mu * 3, kappa)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, vonmises, mu * 3, bad_kappa)

        self.setSeed()
        actual = vonmises(mu, kappa * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, vonmises, mu, bad_kappa * 3)

    def test_pareto(self):
        a = [1]
        bad_a = [-1]
        pareto = np.random.pareto
        desired = np.array([1.1405622680198362,
                            1.1465519762044529,
                            1.0389564467453547])

        self.setSeed()
        actual = pareto(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, pareto, bad_a * 3)

    def test_weibull(self):
        a = [1]
        bad_a = [-1]
        weibull = np.random.weibull
        desired = np.array([0.76106853658845242,
                            0.76386282278691653,
                            0.71243813125891797])

        self.setSeed()
        actual = weibull(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, weibull, bad_a * 3)

    def test_power(self):
        a = [1]
        bad_a = [-1]
        power = np.random.power
        desired = np.array([0.53283302478975902,
                            0.53413660089041659,
                            0.50955303552646702])

        self.setSeed()
        actual = power(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, power, bad_a * 3)

    def test_laplace(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        laplace = np.random.laplace
        desired = np.array([0.067921356028507157,
                            0.070715642226971326,
                            0.019290950698972624])

        self.setSeed()
        actual = laplace(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc * 3, bad_scale)

        self.setSeed()
        actual = laplace(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc, bad_scale * 3)

    def test_gumbel(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        gumbel = np.random.gumbel
        desired = np.array([0.2730318639556768,
                            0.26936705726291116,
                            0.33906220393037939])

        self.setSeed()
        actual = gumbel(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc * 3, bad_scale)

        self.setSeed()
        actual = gumbel(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc, bad_scale * 3)

    def test_logistic(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        logistic = np.random.logistic
        desired = np.array([0.13152135837586171,
                            0.13675915696285773,
                            0.038216792802833396])

        self.setSeed()
        actual = logistic(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, logistic, loc * 3, bad_scale)

        self.setSeed()
        actual = logistic(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, logistic, loc, bad_scale * 3)

    def test_lognormal(self):
        mean = [0]
        sigma = [1]
        bad_sigma = [-1]
        lognormal = np.random.lognormal
        desired = np.array([9.1422086044848427,
                            8.4013952870126261,
                            6.3073234116578671])

        self.setSeed()
        actual = lognormal(mean * 3, sigma)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, lognormal, mean * 3, bad_sigma)

        self.setSeed()
        actual = lognormal(mean, sigma * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, lognormal, mean, bad_sigma * 3)

    def test_rayleigh(self):
        scale = [1]
        bad_scale = [-1]
        rayleigh = np.random.rayleigh
        desired = np.array([1.2337491937897689,
                            1.2360119924878694,
                            1.1936818095781789])

        self.setSeed()
        actual = rayleigh(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, rayleigh, bad_scale * 3)

    def test_wald(self):
        mean = [0.5]
        scale = [1]
        bad_mean = [0]
        bad_scale = [-2]
        wald = np.random.wald
        desired = np.array([0.11873681120271318,
                            0.12450084820795027,
                            0.9096122728408238])

        self.setSeed()
        actual = wald(mean * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, wald, bad_mean * 3, scale)
        assert_raises(ValueError, wald, mean * 3, bad_scale)

        self.setSeed()
        actual = wald(mean, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, wald, bad_mean, scale * 3)
        assert_raises(ValueError, wald, mean, bad_scale * 3)
        assert_raises(ValueError, wald, 0.0, 1)
        assert_raises(ValueError, wald, 0.5, 0.0)

    def test_triangular(self):
        left = [1]
        right = [3]
        mode = [2]
        bad_left_one = [3]
        bad_mode_one = [4]
        bad_left_two, bad_mode_two = right * 2
        triangular = np.random.triangular
        desired = np.array([2.03339048710429,
                            2.0347400359389356,
                            2.0095991069536208])

        self.setSeed()
        actual = triangular(left * 3, mode, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one * 3, mode, right)
        assert_raises(ValueError, triangular, left * 3, bad_mode_one, right)
        assert_raises(ValueError, triangular, bad_left_two * 3, bad_mode_two,
                      right)

        self.setSeed()
        actual = triangular(left, mode * 3, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode * 3, right)
        assert_raises(ValueError, triangular, left, bad_mode_one * 3, right)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two * 3,
                      right)

        self.setSeed()
        actual = triangular(left, mode, right * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode, right * 3)
        assert_raises(ValueError, triangular, left, bad_mode_one, right * 3)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two,
                      right * 3)

    def test_binomial(self):
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        binom = np.random.binomial
        desired = np.array([1, 1, 1])

        self.setSeed()
        actual = binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, binom, bad_n * 3, p)
        assert_raises(ValueError, binom, n * 3, bad_p_one)
        assert_raises(ValueError, binom, n * 3, bad_p_two)

        self.setSeed()
        actual = binom(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, binom, bad_n, p * 3)
        assert_raises(ValueError, binom, n, bad_p_one * 3)
        assert_raises(ValueError, binom, n, bad_p_two * 3)

    def test_negative_binomial(self):
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        neg_binom = np.random.negative_binomial
        desired = np.array([1, 0, 1])

        self.setSeed()
        actual = neg_binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n * 3, p)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_one)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_two)

        self.setSeed()
        actual = neg_binom(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n, p * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_one * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_two * 3)

    def test_poisson(self):
        max_lam = np.random.RandomState()._poisson_lam_max

        lam = [1]
        bad_lam_one = [-1]
        bad_lam_two = [max_lam * 2]
        poisson = np.random.poisson
        desired = np.array([1, 1, 0])

        self.setSeed()
        actual = poisson(lam * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, poisson, bad_lam_one * 3)
        assert_raises(ValueError, poisson, bad_lam_two * 3)

    def test_zipf(self):
        a = [2]
        bad_a = [0]
        zipf = np.random.zipf
        desired = np.array([2, 2, 1])

        self.setSeed()
        actual = zipf(a * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, zipf, bad_a * 3)
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, zipf, np.nan)
            assert_raises(ValueError, zipf, [0, 0, np.nan])

    def test_geometric(self):
        p = [0.5]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        geom = np.random.geometric
        desired = np.array([2, 2, 2])

        self.setSeed()
        actual = geom(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, geom, bad_p_one * 3)
        assert_raises(ValueError, geom, bad_p_two * 3)

    def test_hypergeometric(self):
        ngood = [1]
        nbad = [2]
        nsample = [2]
        bad_ngood = [-1]
        bad_nbad = [-2]
        bad_nsample_one = [0]
        bad_nsample_two = [4]
        hypergeom = np.random.hypergeometric
        desired = np.array([1, 1, 1])

        self.setSeed()
        actual = hypergeom(ngood * 3, nbad, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood * 3, nbad, nsample)
        assert_raises(ValueError, hypergeom, ngood * 3, bad_nbad, nsample)
        assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_one)
        assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_two)

        self.setSeed()
        actual = hypergeom(ngood, nbad * 3, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood, nbad * 3, nsample)
        assert_raises(ValueError, hypergeom, ngood, bad_nbad * 3, nsample)
        assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_one)
        assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_two)

        self.setSeed()
        actual = hypergeom(ngood, nbad, nsample * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood, nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, bad_nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_one * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_two * 3)

    def test_logseries(self):
        p = [0.5]
        bad_p_one = [2]
        bad_p_two = [-1]
        logseries = np.random.logseries
        desired = np.array([1, 1, 1])

        self.setSeed()
        actual = logseries(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, logseries, bad_p_one * 3)
        assert_raises(ValueError, logseries, bad_p_two * 3)


@pytest.mark.skipif(IS_WASM, reason="can't start thread")
class TestThread:
    # make sure each state produces the same sequence even in threads
    def setup_method(self):
        self.seeds = range(4)

    def check_function(self, function, sz):
        from threading import Thread

        out1 = np.empty((len(self.seeds),) + sz)
        out2 = np.empty((len(self.seeds),) + sz)

        # threaded generation
        t = [Thread(target=function, args=(np.random.RandomState(s), o))
             for s, o in zip(self.seeds, out1)]
        [x.start() for x in t]
        [x.join() for x in t]

        # the same serial
        for s, o in zip(self.seeds, out2):
            function(np.random.RandomState(s), o)

        # these platforms change x87 fpu precision mode in threads
        if np.intp().dtype.itemsize == 4 and sys.platform == "win32":
            assert_array_almost_equal(out1, out2)
        else:
            assert_array_equal(out1, out2)

    def test_normal(self):
        def gen_random(state, out):
            out[...] = state.normal(size=10000)
        self.check_function(gen_random, sz=(10000,))

    def test_exp(self):
        def gen_random(state, out):
            out[...] = state.exponential(scale=np.ones((100, 1000)))
        self.check_function(gen_random, sz=(100, 1000))

    def test_multinomial(self):
        def gen_random(state, out):
            out[...] = state.multinomial(10, [1 / 6.] * 6, size=10000)
        self.check_function(gen_random, sz=(10000, 6))


# See Issue #4263
class TestSingleEltArrayInput:
    def setup_method(self):
        self.argOne = np.array([2])
        self.argTwo = np.array([3])
        self.argThree = np.array([4])
        self.tgtShape = (1,)

    def test_one_arg_funcs(self):
        funcs = (np.random.exponential, np.random.standard_gamma,
                 np.random.chisquare, np.random.standard_t,
                 np.random.pareto, np.random.weibull,
                 np.random.power, np.random.rayleigh,
                 np.random.poisson, np.random.zipf,
                 np.random.geometric, np.random.logseries)

        probfuncs = (np.random.geometric, np.random.logseries)

        for func in funcs:
            if func in probfuncs:  # p < 1.0
                out = func(np.array([0.5]))

            else:
                out = func(self.argOne)

            assert_equal(out.shape, self.tgtShape)

    def test_two_arg_funcs(self):
        funcs = (np.random.uniform, np.random.normal,
                 np.random.beta, np.random.gamma,
                 np.random.f, np.random.noncentral_chisquare,
                 np.random.vonmises, np.random.laplace,
                 np.random.gumbel, np.random.logistic,
                 np.random.lognormal, np.random.wald,
                 np.random.binomial, np.random.negative_binomial)

        probfuncs = (np.random.binomial, np.random.negative_binomial)

        for func in funcs:
            if func in probfuncs:  # p <= 1
                argTwo = np.array([0.5])

            else:
                argTwo = self.argTwo

            out = func(self.argOne, argTwo)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne[0], argTwo)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne, argTwo[0])
            assert_equal(out.shape, self.tgtShape)

    def test_randint(self):
        itype = [bool, np.int8, np.uint8, np.int16, np.uint16,
                 np.int32, np.uint32, np.int64, np.uint64]
        func = np.random.randint
        high = np.array([1])
        low = np.array([0])

        for dt in itype:
            out = func(low, high, dtype=dt)
            assert_equal(out.shape, self.tgtShape)

            out = func(low[0], high, dtype=dt)
            assert_equal(out.shape, self.tgtShape)

            out = func(low, high[0], dtype=dt)
            assert_equal(out.shape, self.tgtShape)

    def test_three_arg_funcs(self):
        funcs = [np.random.noncentral_f, np.random.triangular,
                 np.random.hypergeometric]

        for func in funcs:
            out = func(self.argOne, self.argTwo, self.argThree)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne[0], self.argTwo, self.argThree)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne, self.argTwo[0], self.argThree)
            assert_equal(out.shape, self.tgtShape)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\tests\test_sets.py ===
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.containers import TupleKind
from sympy.core.function import Lambda
from sympy.core.kind import NumberKind, UndefinedKind
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.logic.boolalg import (false, true)
from sympy.matrices.kind import MatrixKind
from sympy.matrices.dense import Matrix
from sympy.polys.rootoftools import rootof
from sympy.sets.contains import Contains
from sympy.sets.fancysets import (ImageSet, Range)
from sympy.sets.sets import (Complement, DisjointUnion, FiniteSet, Intersection, Interval, ProductSet, Set, SymmetricDifference, Union, imageset, SetKind)
from mpmath import mpi

from sympy.core.expr import unchanged
from sympy.core.relational import Eq, Ne, Le, Lt, LessThan
from sympy.logic import And, Or, Xor
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
from sympy.utilities.iterables import cartes

from sympy.abc import x, y, z, m, n

EmptySet = S.EmptySet

def test_imageset():
    ints = S.Integers
    assert imageset(x, x - 1, S.Naturals) is S.Naturals0
    assert imageset(x, x + 1, S.Naturals0) is S.Naturals
    assert imageset(x, abs(x), S.Naturals0) is S.Naturals0
    assert imageset(x, abs(x), S.Naturals) is S.Naturals
    assert imageset(x, abs(x), S.Integers) is S.Naturals0
    # issue 16878a
    r = symbols('r', real=True)
    assert imageset(x, (x, x), S.Reals)._contains((1, r)) == None
    assert imageset(x, (x, x), S.Reals)._contains((1, 2)) == False
    assert (r, r) in imageset(x, (x, x), S.Reals)
    assert 1 + I in imageset(x, x + I, S.Reals)
    assert {1} not in imageset(x, (x,), S.Reals)
    assert (1, 1) not in imageset(x, (x,), S.Reals)
    raises(TypeError, lambda: imageset(x, ints))
    raises(ValueError, lambda: imageset(x, y, z, ints))
    raises(ValueError, lambda: imageset(Lambda(x, cos(x)), y))
    assert (1, 2) in imageset(Lambda((x, y), (x, y)), ints, ints)
    raises(ValueError, lambda: imageset(Lambda(x, x), ints, ints))
    assert imageset(cos, ints) == ImageSet(Lambda(x, cos(x)), ints)
    def f(x):
        return cos(x)
    assert imageset(f, ints) == imageset(x, cos(x), ints)
    f = lambda x: cos(x)
    assert imageset(f, ints) == ImageSet(Lambda(x, cos(x)), ints)
    assert imageset(x, 1, ints) == FiniteSet(1)
    assert imageset(x, y, ints) == {y}
    assert imageset((x, y), (1, z), ints, S.Reals) == {(1, z)}
    clash = Symbol('x', integer=true)
    assert (str(imageset(lambda x: x + clash, Interval(-2, 1)).lamda.expr)
        in ('x0 + x', 'x + x0'))
    x1, x2 = symbols("x1, x2")
    assert imageset(lambda x, y:
        Add(x, y), Interval(1, 2), Interval(2, 3)).dummy_eq(
        ImageSet(Lambda((x1, x2), x1 + x2),
        Interval(1, 2), Interval(2, 3)))


def test_is_empty():
    for s in [S.Naturals, S.Naturals0, S.Integers, S.Rationals, S.Reals,
            S.UniversalSet]:
        assert s.is_empty is False

    assert S.EmptySet.is_empty is True


def test_is_finiteset():
    for s in [S.Naturals, S.Naturals0, S.Integers, S.Rationals, S.Reals,
            S.UniversalSet]:
        assert s.is_finite_set is False

    assert S.EmptySet.is_finite_set is True

    assert FiniteSet(1, 2).is_finite_set is True
    assert Interval(1, 2).is_finite_set is False
    assert Interval(x, y).is_finite_set is None
    assert ProductSet(FiniteSet(1), FiniteSet(2)).is_finite_set is True
    assert ProductSet(FiniteSet(1), Interval(1, 2)).is_finite_set is False
    assert ProductSet(FiniteSet(1), Interval(x, y)).is_finite_set is None
    assert Union(Interval(0, 1), Interval(2, 3)).is_finite_set is False
    assert Union(FiniteSet(1), Interval(2, 3)).is_finite_set is False
    assert Union(FiniteSet(1), FiniteSet(2)).is_finite_set is True
    assert Union(FiniteSet(1), Interval(x, y)).is_finite_set is None
    assert Intersection(Interval(x, y), FiniteSet(1)).is_finite_set is True
    assert Intersection(Interval(x, y), Interval(1, 2)).is_finite_set is None
    assert Intersection(FiniteSet(x), FiniteSet(y)).is_finite_set is True
    assert Complement(FiniteSet(1), Interval(x, y)).is_finite_set is True
    assert Complement(Interval(x, y), FiniteSet(1)).is_finite_set is None
    assert Complement(Interval(1, 2), FiniteSet(x)).is_finite_set is False
    assert DisjointUnion(Interval(-5, 3), FiniteSet(x, y)).is_finite_set is False
    assert DisjointUnion(S.EmptySet, FiniteSet(x, y), S.EmptySet).is_finite_set is True


def test_deprecated_is_EmptySet():
    with warns_deprecated_sympy():
        S.EmptySet.is_EmptySet

    with warns_deprecated_sympy():
        FiniteSet(1).is_EmptySet


def test_interval_arguments():
    assert Interval(0, oo) == Interval(0, oo, False, True)
    assert Interval(0, oo).right_open is true
    assert Interval(-oo, 0) == Interval(-oo, 0, True, False)
    assert Interval(-oo, 0).left_open is true
    assert Interval(oo, -oo) == S.EmptySet
    assert Interval(oo, oo) == S.EmptySet
    assert Interval(-oo, -oo) == S.EmptySet
    assert Interval(oo, x) == S.EmptySet
    assert Interval(oo, oo) == S.EmptySet
    assert Interval(x, -oo) == S.EmptySet
    assert Interval(x, x) == {x}

    assert isinstance(Interval(1, 1), FiniteSet)
    e = Sum(x, (x, 1, 3))
    assert isinstance(Interval(e, e), FiniteSet)

    assert Interval(1, 0) == S.EmptySet
    assert Interval(1, 1).measure == 0

    assert Interval(1, 1, False, True) == S.EmptySet
    assert Interval(1, 1, True, False) == S.EmptySet
    assert Interval(1, 1, True, True) == S.EmptySet


    assert isinstance(Interval(0, Symbol('a')), Interval)
    assert Interval(Symbol('a', positive=True), 0) == S.EmptySet
    raises(ValueError, lambda: Interval(0, S.ImaginaryUnit))
    raises(ValueError, lambda: Interval(0, Symbol('z', extended_real=False)))
    raises(ValueError, lambda: Interval(x, x + S.ImaginaryUnit))

    raises(NotImplementedError, lambda: Interval(0, 1, And(x, y)))
    raises(NotImplementedError, lambda: Interval(0, 1, False, And(x, y)))
    raises(NotImplementedError, lambda: Interval(0, 1, z, And(x, y)))


def test_interval_symbolic_end_points():
    a = Symbol('a', real=True)

    assert Union(Interval(0, a), Interval(0, 3)).sup == Max(a, 3)
    assert Union(Interval(a, 0), Interval(-3, 0)).inf == Min(-3, a)

    assert Interval(0, a).contains(1) == LessThan(1, a)


def test_interval_is_empty():
    x, y = symbols('x, y')
    r = Symbol('r', real=True)
    p = Symbol('p', positive=True)
    n = Symbol('n', negative=True)
    nn = Symbol('nn', nonnegative=True)
    assert Interval(1, 2).is_empty == False
    assert Interval(3, 3).is_empty == False  # FiniteSet
    assert Interval(r, r).is_empty == False  # FiniteSet
    assert Interval(r, r + nn).is_empty == False
    assert Interval(x, x).is_empty == False
    assert Interval(1, oo).is_empty == False
    assert Interval(-oo, oo).is_empty == False
    assert Interval(-oo, 1).is_empty == False
    assert Interval(x, y).is_empty == None
    assert Interval(r, oo).is_empty == False  # real implies finite
    assert Interval(n, 0).is_empty == False
    assert Interval(n, 0, left_open=True).is_empty == False
    assert Interval(p, 0).is_empty == True  # EmptySet
    assert Interval(nn, 0).is_empty == None
    assert Interval(n, p).is_empty == False
    assert Interval(0, p, left_open=True).is_empty == False
    assert Interval(0, p, right_open=True).is_empty == False
    assert Interval(0, nn, left_open=True).is_empty == None
    assert Interval(0, nn, right_open=True).is_empty == None


def test_union():
    assert Union(Interval(1, 2), Interval(2, 3)) == Interval(1, 3)
    assert Union(Interval(1, 2), Interval(2, 3, True)) == Interval(1, 3)
    assert Union(Interval(1, 3), Interval(2, 4)) == Interval(1, 4)
    assert Union(Interval(1, 2), Interval(1, 3)) == Interval(1, 3)
    assert Union(Interval(1, 3), Interval(1, 2)) == Interval(1, 3)
    assert Union(Interval(1, 3, False, True), Interval(1, 2)) == \
        Interval(1, 3, False, True)
    assert Union(Interval(1, 3), Interval(1, 2, False, True)) == Interval(1, 3)
    assert Union(Interval(1, 2, True), Interval(1, 3)) == Interval(1, 3)
    assert Union(Interval(1, 2, True), Interval(1, 3, True)) == \
        Interval(1, 3, True)
    assert Union(Interval(1, 2, True), Interval(1, 3, True, True)) == \
        Interval(1, 3, True, True)
    assert Union(Interval(1, 2, True, True), Interval(1, 3, True)) == \
        Interval(1, 3, True)
    assert Union(Interval(1, 3), Interval(2, 3)) == Interval(1, 3)
    assert Union(Interval(1, 3, False, True), Interval(2, 3)) == \
        Interval(1, 3)
    assert Union(Interval(1, 2, False, True), Interval(2, 3, True)) != \
        Interval(1, 3)
    assert Union(Interval(1, 2), S.EmptySet) == Interval(1, 2)
    assert Union(S.EmptySet) == S.EmptySet

    assert Union(Interval(0, 1), *[FiniteSet(1.0/n) for n in range(1, 10)]) == \
        Interval(0, 1)
    # issue #18241:
    x = Symbol('x')
    assert Union(Interval(0, 1), FiniteSet(1, x)) == Union(
        Interval(0, 1), FiniteSet(x))
    assert unchanged(Union, Interval(0, 1), FiniteSet(2, x))

    assert Interval(1, 2).union(Interval(2, 3)) == \
        Interval(1, 2) + Interval(2, 3)

    assert Interval(1, 2).union(Interval(2, 3)) == Interval(1, 3)

    assert Union(Set()) == Set()

    assert FiniteSet(1) + FiniteSet(2) + FiniteSet(3) == FiniteSet(1, 2, 3)
    assert FiniteSet('ham') + FiniteSet('eggs') == FiniteSet('ham', 'eggs')
    assert FiniteSet(1, 2, 3) + S.EmptySet == FiniteSet(1, 2, 3)

    assert FiniteSet(1, 2, 3) & FiniteSet(2, 3, 4) == FiniteSet(2, 3)
    assert FiniteSet(1, 2, 3) | FiniteSet(2, 3, 4) == FiniteSet(1, 2, 3, 4)

    assert FiniteSet(1, 2, 3) & S.EmptySet == S.EmptySet
    assert FiniteSet(1, 2, 3) | S.EmptySet == FiniteSet(1, 2, 3)

    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    assert S.EmptySet | FiniteSet(x, FiniteSet(y, z)) == \
        FiniteSet(x, FiniteSet(y, z))

    # Test that Intervals and FiniteSets play nicely
    assert Interval(1, 3) + FiniteSet(2) == Interval(1, 3)
    assert Interval(1, 3, True, True) + FiniteSet(3) == \
        Interval(1, 3, True, False)
    X = Interval(1, 3) + FiniteSet(5)
    Y = Interval(1, 2) + FiniteSet(3)
    XandY = X.intersect(Y)
    assert 2 in X and 3 in X and 3 in XandY
    assert XandY.is_subset(X) and XandY.is_subset(Y)

    raises(TypeError, lambda: Union(1, 2, 3))

    assert X.is_iterable is False

    # issue 7843
    assert Union(S.EmptySet, FiniteSet(-sqrt(-I), sqrt(-I))) == \
        FiniteSet(-sqrt(-I), sqrt(-I))

    assert Union(S.Reals, S.Integers) == S.Reals


def test_union_iter():
    # Use Range because it is ordered
    u = Union(Range(3), Range(5), Range(4), evaluate=False)

    # Round robin
    assert list(u) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4]


def test_union_is_empty():
    assert (Interval(x, y) + FiniteSet(1)).is_empty == False
    assert (Interval(x, y) + Interval(-x, y)).is_empty == None


def test_difference():
    assert Interval(1, 3) - Interval(1, 2) == Interval(2, 3, True)
    assert Interval(1, 3) - Interval(2, 3) == Interval(1, 2, False, True)
    assert Interval(1, 3, True) - Interval(2, 3) == Interval(1, 2, True, True)
    assert Interval(1, 3, True) - Interval(2, 3, True) == \
        Interval(1, 2, True, False)
    assert Interval(0, 2) - FiniteSet(1) == \
        Union(Interval(0, 1, False, True), Interval(1, 2, True, False))

    # issue #18119
    assert S.Reals - FiniteSet(I) == S.Reals
    assert S.Reals - FiniteSet(-I, I) == S.Reals
    assert Interval(0, 10) - FiniteSet(-I, I) == Interval(0, 10)
    assert Interval(0, 10) - FiniteSet(1, I) == Union(
        Interval.Ropen(0, 1), Interval.Lopen(1, 10))
    assert S.Reals - FiniteSet(1, 2 + I, x, y**2) == Complement(
        Union(Interval.open(-oo, 1), Interval.open(1, oo)), FiniteSet(x, y**2),
        evaluate=False)

    assert FiniteSet(1, 2, 3) - FiniteSet(2) == FiniteSet(1, 3)
    assert FiniteSet('ham', 'eggs') - FiniteSet('eggs') == FiniteSet('ham')
    assert FiniteSet(1, 2, 3, 4) - Interval(2, 10, True, False) == \
        FiniteSet(1, 2)
    assert FiniteSet(1, 2, 3, 4) - S.EmptySet == FiniteSet(1, 2, 3, 4)
    assert Union(Interval(0, 2), FiniteSet(2, 3, 4)) - Interval(1, 3) == \
        Union(Interval(0, 1, False, True), FiniteSet(4))

    assert -1 in S.Reals - S.Naturals


def test_Complement():
    A = FiniteSet(1, 3, 4)
    B = FiniteSet(3, 4)
    C = Interval(1, 3)
    D = Interval(1, 2)

    assert Complement(A, B, evaluate=False).is_iterable is True
    assert Complement(A, C, evaluate=False).is_iterable is True
    assert Complement(C, D, evaluate=False).is_iterable is None

    assert FiniteSet(*Complement(A, B, evaluate=False)) == FiniteSet(1)
    assert FiniteSet(*Complement(A, C, evaluate=False)) == FiniteSet(4)
    raises(TypeError, lambda: FiniteSet(*Complement(C, A, evaluate=False)))

    assert Complement(Interval(1, 3), Interval(1, 2)) == Interval(2, 3, True)
    assert Complement(FiniteSet(1, 3, 4), FiniteSet(3, 4)) == FiniteSet(1)
    assert Complement(Union(Interval(0, 2), FiniteSet(2, 3, 4)),
                      Interval(1, 3)) == \
        Union(Interval(0, 1, False, True), FiniteSet(4))

    assert 3 not in Complement(Interval(0, 5), Interval(1, 4), evaluate=False)
    assert -1 in Complement(S.Reals, S.Naturals, evaluate=False)
    assert 1 not in Complement(S.Reals, S.Naturals, evaluate=False)

    assert Complement(S.Integers, S.UniversalSet) == EmptySet
    assert S.UniversalSet.complement(S.Integers) == EmptySet

    assert (0 not in S.Reals.intersect(S.Integers - FiniteSet(0)))

    assert S.EmptySet - S.Integers == S.EmptySet

    assert (S.Integers - FiniteSet(0)) - FiniteSet(1) == S.Integers - FiniteSet(0, 1)

    assert S.Reals - Union(S.Naturals, FiniteSet(pi)) == \
            Intersection(S.Reals - S.Naturals, S.Reals - FiniteSet(pi))
    # issue 12712
    assert Complement(FiniteSet(x, y, 2), Interval(-10, 10)) == \
            Complement(FiniteSet(x, y), Interval(-10, 10))

    A = FiniteSet(*symbols('a:c'))
    B = FiniteSet(*symbols('d:f'))
    assert unchanged(Complement, ProductSet(A, A), B)

    A2 = ProductSet(A, A)
    B3 = ProductSet(B, B, B)
    assert A2 - B3 == A2
    assert B3 - A2 == B3


def test_set_operations_nonsets():
    '''Tests that e.g. FiniteSet(1) * 2 raises TypeError'''
    ops = [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
        lambda a, b: a // b,
        lambda a, b: a | b,
        lambda a, b: a & b,
        lambda a, b: a ^ b,
        # FiniteSet(1) ** 2 gives a ProductSet
        #lambda a, b: a ** b,
    ]
    Sx = FiniteSet(x)
    Sy = FiniteSet(y)
    sets = [
        {1},
        FiniteSet(1),
        Interval(1, 2),
        Union(Sx, Interval(1, 2)),
        Intersection(Sx, Sy),
        Complement(Sx, Sy),
        ProductSet(Sx, Sy),
        S.EmptySet,
    ]
    nums = [0, 1, 2, S(0), S(1), S(2)]

    for si in sets:
        for ni in nums:
            for op in ops:
                raises(TypeError, lambda : op(si, ni))
                raises(TypeError, lambda : op(ni, si))
        raises(TypeError, lambda: si ** object())
        raises(TypeError, lambda: si ** {1})


def test_complement():
    assert Complement({1, 2}, {1}) == {2}
    assert Interval(0, 1).complement(S.Reals) == \
        Union(Interval(-oo, 0, True, True), Interval(1, oo, True, True))
    assert Interval(0, 1, True, False).complement(S.Reals) == \
        Union(Interval(-oo, 0, True, False), Interval(1, oo, True, True))
    assert Interval(0, 1, False, True).complement(S.Reals) == \
        Union(Interval(-oo, 0, True, True), Interval(1, oo, False, True))
    assert Interval(0, 1, True, True).complement(S.Reals) == \
        Union(Interval(-oo, 0, True, False), Interval(1, oo, False, True))

    assert S.UniversalSet.complement(S.EmptySet) == S.EmptySet
    assert S.UniversalSet.complement(S.Reals) == S.EmptySet
    assert S.UniversalSet.complement(S.UniversalSet) == S.EmptySet

    assert S.EmptySet.complement(S.Reals) == S.Reals

    assert Union(Interval(0, 1), Interval(2, 3)).complement(S.Reals) == \
        Union(Interval(-oo, 0, True, True), Interval(1, 2, True, True),
              Interval(3, oo, True, True))

    assert FiniteSet(0).complement(S.Reals) ==  \
        Union(Interval(-oo, 0, True, True), Interval(0, oo, True, True))

    assert (FiniteSet(5) + Interval(S.NegativeInfinity,
                                    0)).complement(S.Reals) == \
        Interval(0, 5, True, True) + Interval(5, S.Infinity, True, True)

    assert FiniteSet(1, 2, 3).complement(S.Reals) == \
        Interval(S.NegativeInfinity, 1, True, True) + \
        Interval(1, 2, True, True) + Interval(2, 3, True, True) +\
        Interval(3, S.Infinity, True, True)

    assert FiniteSet(x).complement(S.Reals) == Complement(S.Reals, FiniteSet(x))

    assert FiniteSet(0, x).complement(S.Reals) == Complement(Interval(-oo, 0, True, True) +
                                                             Interval(0, oo, True, True)
                                                             , FiniteSet(x), evaluate=False)

    square = Interval(0, 1) * Interval(0, 1)
    notsquare = square.complement(S.Reals*S.Reals)

    assert all(pt in square for pt in [(0, 0), (.5, .5), (1, 0), (1, 1)])
    assert not any(
        pt in notsquare for pt in [(0, 0), (.5, .5), (1, 0), (1, 1)])
    assert not any(pt in square for pt in [(-1, 0), (1.5, .5), (10, 10)])
    assert all(pt in notsquare for pt in [(-1, 0), (1.5, .5), (10, 10)])


def test_intersect1():
    assert all(S.Integers.intersection(i) is i for i in
        (S.Naturals, S.Naturals0))
    assert all(i.intersection(S.Integers) is i for i in
        (S.Naturals, S.Naturals0))
    s =  S.Naturals0
    assert S.Naturals.intersection(s) is S.Naturals
    assert s.intersection(S.Naturals) is S.Naturals
    x = Symbol('x')
    assert Interval(0, 2).intersect(Interval(1, 2)) == Interval(1, 2)
    assert Interval(0, 2).intersect(Interval(1, 2, True)) == \
        Interval(1, 2, True)
    assert Interval(0, 2, True).intersect(Interval(1, 2)) == \
        Interval(1, 2, False, False)
    assert Interval(0, 2, True, True).intersect(Interval(1, 2)) == \
        Interval(1, 2, False, True)
    assert Interval(0, 2).intersect(Union(Interval(0, 1), Interval(2, 3))) == \
        Union(Interval(0, 1), Interval(2, 2))

    assert FiniteSet(1, 2).intersect(FiniteSet(1, 2, 3)) == FiniteSet(1, 2)
    assert FiniteSet(1, 2, x).intersect(FiniteSet(x)) == FiniteSet(x)
    assert FiniteSet('ham', 'eggs').intersect(FiniteSet('ham')) == \
        FiniteSet('ham')
    assert FiniteSet(1, 2, 3, 4, 5).intersect(S.EmptySet) == S.EmptySet

    assert Interval(0, 5).intersect(FiniteSet(1, 3)) == FiniteSet(1, 3)
    assert Interval(0, 1, True, True).intersect(FiniteSet(1)) == S.EmptySet

    assert Union(Interval(0, 1), Interval(2, 3)).intersect(Interval(1, 2)) == \
        Union(Interval(1, 1), Interval(2, 2))
    assert Union(Interval(0, 1), Interval(2, 3)).intersect(Interval(0, 2)) == \
        Union(Interval(0, 1), Interval(2, 2))
    assert Union(Interval(0, 1), Interval(2, 3)).intersect(Interval(1, 2, True, True)) == \
        S.EmptySet
    assert Union(Interval(0, 1), Interval(2, 3)).intersect(S.EmptySet) == \
        S.EmptySet
    assert Union(Interval(0, 5), FiniteSet('ham')).intersect(FiniteSet(2, 3, 4, 5, 6)) == \
        Intersection(FiniteSet(2, 3, 4, 5, 6), Union(FiniteSet('ham'), Interval(0, 5)))
    assert Intersection(FiniteSet(1, 2, 3), Interval(2, x), Interval(3, y)) == \
        Intersection(FiniteSet(3), Interval(2, x), Interval(3, y), evaluate=False)
    assert Intersection(FiniteSet(1, 2), Interval(0, 3), Interval(x, y)) == \
        Intersection({1, 2}, Interval(x, y), evaluate=False)
    assert Intersection(FiniteSet(1, 2, 4), Interval(0, 3), Interval(x, y)) == \
        Intersection({1, 2}, Interval(x, y), evaluate=False)
    # XXX: Is the real=True necessary here?
    # https://github.com/sympy/sympy/issues/17532
    m, n = symbols('m, n', real=True)
    assert Intersection(FiniteSet(m), FiniteSet(m, n), Interval(m, m+1)) == \
        FiniteSet(m)

    # issue 8217
    assert Intersection(FiniteSet(x), FiniteSet(y)) == \
        Intersection(FiniteSet(x), FiniteSet(y), evaluate=False)
    assert FiniteSet(x).intersect(S.Reals) == \
        Intersection(S.Reals, FiniteSet(x), evaluate=False)

    # tests for the intersection alias
    assert Interval(0, 5).intersection(FiniteSet(1, 3)) == FiniteSet(1, 3)
    assert Interval(0, 1, True, True).intersection(FiniteSet(1)) == S.EmptySet

    assert Union(Interval(0, 1), Interval(2, 3)).intersection(Interval(1, 2)) == \
        Union(Interval(1, 1), Interval(2, 2))

    # canonical boundary selected
    a = sqrt(2*sqrt(6) + 5)
    b = sqrt(2) + sqrt(3)
    assert Interval(a, 4).intersection(Interval(b, 5)) == Interval(b, 4)
    assert Interval(1, a).intersection(Interval(0, b)) == Interval(1, b)


def test_intersection_interval_float():
    # intersection of Intervals with mixed Rational/Float boundaries should
    # lead to Float boundaries in all cases regardless of which Interval is
    # open or closed.
    typs = [
        (Interval, Interval, Interval),
        (Interval, Interval.open, Interval.open),
        (Interval, Interval.Lopen, Interval.Lopen),
        (Interval, Interval.Ropen, Interval.Ropen),
        (Interval.open, Interval.open, Interval.open),
        (Interval.open, Interval.Lopen, Interval.open),
        (Interval.open, Interval.Ropen, Interval.open),
        (Interval.Lopen, Interval.Lopen, Interval.Lopen),
        (Interval.Lopen, Interval.Ropen, Interval.open),
        (Interval.Ropen, Interval.Ropen, Interval.Ropen),
    ]

    as_float = lambda a1, a2: a2 if isinstance(a2, float) else a1

    for t1, t2, t3 in typs:
        for t1i, t2i in [(t1, t2), (t2, t1)]:
            for a1, a2, b1, b2 in cartes([2, 2.0], [2, 2.0], [3, 3.0], [3, 3.0]):
                I1 = t1(a1, b1)
                I2 = t2(a2, b2)
                I3 = t3(as_float(a1, a2), as_float(b1, b2))
                assert I1.intersect(I2) == I3


def test_intersection():
    # iterable
    i = Intersection(FiniteSet(1, 2, 3), Interval(2, 5), evaluate=False)
    assert i.is_iterable
    assert set(i) == {S(2), S(3)}

    # challenging intervals
    x = Symbol('x', real=True)
    i = Intersection(Interval(0, 3), Interval(x, 6))
    assert (5 in i) is False
    raises(TypeError, lambda: 2 in i)

    # Singleton special cases
    assert Intersection(Interval(0, 1), S.EmptySet) == S.EmptySet
    assert Intersection(Interval(-oo, oo), Interval(-oo, x)) == Interval(-oo, x)

    # Products
    line = Interval(0, 5)
    i = Intersection(line**2, line**3, evaluate=False)
    assert (2, 2) not in i
    assert (2, 2, 2) not in i
    raises(TypeError, lambda: list(i))

    a = Intersection(Intersection(S.Integers, S.Naturals, evaluate=False), S.Reals, evaluate=False)
    assert a._argset == frozenset([Intersection(S.Naturals, S.Integers, evaluate=False), S.Reals])

    assert Intersection(S.Complexes, FiniteSet(S.ComplexInfinity)) == S.EmptySet

    # issue 12178
    assert Intersection() == S.UniversalSet

    # issue 16987
    assert Intersection({1}, {1}, {x}) == Intersection({1}, {x})


def test_issue_9623():
    n = Symbol('n')

    a = S.Reals
    b = Interval(0, oo)
    c = FiniteSet(n)

    assert Intersection(a, b, c) == Intersection(b, c)
    assert Intersection(Interval(1, 2), Interval(3, 4), FiniteSet(n)) == EmptySet


def test_is_disjoint():
    assert Interval(0, 2).is_disjoint(Interval(1, 2)) == False
    assert Interval(0, 2).is_disjoint(Interval(3, 4)) == True


def test_ProductSet__len__():
    A = FiniteSet(1, 2)
    B = FiniteSet(1, 2, 3)
    assert ProductSet(A).__len__() == 2
    assert ProductSet(A).__len__() is not S(2)
    assert ProductSet(A, B).__len__() == 6
    assert ProductSet(A, B).__len__() is not S(6)


def test_ProductSet():
    # ProductSet is always a set of Tuples
    assert ProductSet(S.Reals) == S.Reals ** 1
    assert ProductSet(S.Reals, S.Reals) == S.Reals ** 2
    assert ProductSet(S.Reals, S.Reals, S.Reals) == S.Reals ** 3

    assert ProductSet(S.Reals) != S.Reals
    assert ProductSet(S.Reals, S.Reals) == S.Reals * S.Reals
    assert ProductSet(S.Reals, S.Reals, S.Reals) != S.Reals * S.Reals * S.Reals
    assert ProductSet(S.Reals, S.Reals, S.Reals) == (S.Reals * S.Reals * S.Reals).flatten()

    assert 1 not in ProductSet(S.Reals)
    assert (1,) in ProductSet(S.Reals)

    assert 1 not in ProductSet(S.Reals, S.Reals)
    assert (1, 2) in ProductSet(S.Reals, S.Reals)
    assert (1, I) not in ProductSet(S.Reals, S.Reals)

    assert (1, 2, 3) in ProductSet(S.Reals, S.Reals, S.Reals)
    assert (1, 2, 3) in S.Reals ** 3
    assert (1, 2, 3) not in S.Reals * S.Reals * S.Reals
    assert ((1, 2), 3) in S.Reals * S.Reals * S.Reals
    assert (1, (2, 3)) not in S.Reals * S.Reals * S.Reals
    assert (1, (2, 3)) in S.Reals * (S.Reals * S.Reals)

    assert ProductSet() == FiniteSet(())
    assert ProductSet(S.Reals, S.EmptySet) == S.EmptySet

    # See GH-17458

    for ni in range(5):
        Rn = ProductSet(*(S.Reals,) * ni)
        assert (1,) * ni in Rn
        assert 1 not in Rn

    assert (S.Reals * S.Reals) * S.Reals != S.Reals * (S.Reals * S.Reals)

    S1 = S.Reals
    S2 = S.Integers
    x1 = pi
    x2 = 3
    assert x1 in S1
    assert x2 in S2
    assert (x1, x2) in S1 * S2
    S3 = S1 * S2
    x3 = (x1, x2)
    assert x3 in S3
    assert (x3, x3) in S3 * S3
    assert x3 + x3 not in S3 * S3

    raises(ValueError, lambda: S.Reals**-1)
    with warns_deprecated_sympy():
        ProductSet(FiniteSet(s) for s in range(2))
    raises(TypeError, lambda: ProductSet(None))

    S1 = FiniteSet(1, 2)
    S2 = FiniteSet(3, 4)
    S3 = ProductSet(S1, S2)
    assert (S3.as_relational(x, y)
            == And(S1.as_relational(x), S2.as_relational(y))
            == And(Or(Eq(x, 1), Eq(x, 2)), Or(Eq(y, 3), Eq(y, 4))))
    raises(ValueError, lambda: S3.as_relational(x))
    raises(ValueError, lambda: S3.as_relational(x, 1))
    raises(ValueError, lambda: ProductSet(Interval(0, 1)).as_relational(x, y))

    Z2 = ProductSet(S.Integers, S.Integers)
    assert Z2.contains((1, 2)) is S.true
    assert Z2.contains((1,)) is S.false
    assert Z2.contains(x) == Contains(x, Z2, evaluate=False)
    assert Z2.contains(x).subs(x, 1) is S.false
    assert Z2.contains((x, 1)).subs(x, 2) is S.true
    assert Z2.contains((x, y)) == Contains(x, S.Integers) & Contains(y, S.Integers)
    assert unchanged(Contains, (x, y), Z2)
    assert Contains((1, 2), Z2) is S.true


def test_ProductSet_of_single_arg_is_not_arg():
    assert unchanged(ProductSet, Interval(0, 1))
    assert unchanged(ProductSet, ProductSet(Interval(0, 1)))


def test_ProductSet_is_empty():
    assert ProductSet(S.Integers, S.Reals).is_empty == False
    assert ProductSet(Interval(x, 1), S.Reals).is_empty == None


def test_interval_subs():
    a = Symbol('a', real=True)

    assert Interval(0, a).subs(a, 2) == Interval(0, 2)
    assert Interval(a, 0).subs(a, 2) == S.EmptySet


def test_interval_to_mpi():
    assert Interval(0, 1).to_mpi() == mpi(0, 1)
    assert Interval(0, 1, True, False).to_mpi() == mpi(0, 1)
    assert type(Interval(0, 1).to_mpi()) == type(mpi(0, 1))


def test_set_evalf():
    assert Interval(S(11)/64, S.Half).evalf() == Interval(
        Float('0.171875'), Float('0.5'))
    assert Interval(x, S.Half, right_open=True).evalf() == Interval(
        x, Float('0.5'), right_open=True)
    assert Interval(-oo, S.Half).evalf() == Interval(-oo, Float('0.5'))
    assert FiniteSet(2, x).evalf() == FiniteSet(Float('2.0'), x)


def test_measure():
    a = Symbol('a', real=True)

    assert Interval(1, 3).measure == 2
    assert Interval(0, a).measure == a
    assert Interval(1, a).measure == a - 1

    assert Union(Interval(1, 2), Interval(3, 4)).measure == 2
    assert Union(Interval(1, 2), Interval(3, 4), FiniteSet(5, 6, 7)).measure \
        == 2

    assert FiniteSet(1, 2, oo, a, -oo, -5).measure == 0

    assert S.EmptySet.measure == 0

    square = Interval(0, 10) * Interval(0, 10)
    offsetsquare = Interval(5, 15) * Interval(5, 15)
    band = Interval(-oo, oo) * Interval(2, 4)

    assert square.measure == offsetsquare.measure == 100
    assert (square + offsetsquare).measure == 175  # there is some overlap
    assert (square - offsetsquare).measure == 75
    assert (square * FiniteSet(1, 2, 3)).measure == 0
    assert (square.intersect(band)).measure == 20
    assert (square + band).measure is oo
    assert (band * FiniteSet(1, 2, 3)).measure is nan


def test_is_subset():
    assert Interval(0, 1).is_subset(Interval(0, 2)) is True
    assert Interval(0, 3).is_subset(Interval(0, 2)) is False
    assert Interval(0, 1).is_subset(FiniteSet(0, 1)) is False

    assert FiniteSet(1, 2).is_subset(FiniteSet(1, 2, 3, 4))
    assert FiniteSet(4, 5).is_subset(FiniteSet(1, 2, 3, 4)) is False
    assert FiniteSet(1).is_subset(Interval(0, 2))
    assert FiniteSet(1, 2).is_subset(Interval(0, 2, True, True)) is False
    assert (Interval(1, 2) + FiniteSet(3)).is_subset(
        Interval(0, 2, False, True) + FiniteSet(2, 3))

    assert Interval(3, 4).is_subset(Union(Interval(0, 1), Interval(2, 5))) is True
    assert Interval(3, 6).is_subset(Union(Interval(0, 1), Interval(2, 5))) is False

    assert FiniteSet(1, 2, 3, 4).is_subset(Interval(0, 5)) is True
    assert S.EmptySet.is_subset(FiniteSet(1, 2, 3)) is True

    assert Interval(0, 1).is_subset(S.EmptySet) is False
    assert S.EmptySet.is_subset(S.EmptySet) is True

    raises(ValueError, lambda: S.EmptySet.is_subset(1))

    # tests for the issubset alias
    assert FiniteSet(1, 2, 3, 4).issubset(Interval(0, 5)) is True
    assert S.EmptySet.issubset(FiniteSet(1, 2, 3)) is True

    assert S.Naturals.is_subset(S.Integers)
    assert S.Naturals0.is_subset(S.Integers)

    assert FiniteSet(x).is_subset(FiniteSet(y)) is None
    assert FiniteSet(x).is_subset(FiniteSet(y).subs(y, x)) is True
    assert FiniteSet(x).is_subset(FiniteSet(y).subs(y, x+1)) is False

    assert Interval(0, 1).is_subset(Interval(0, 1, left_open=True)) is False
    assert Interval(-2, 3).is_subset(Union(Interval(-oo, -2), Interval(3, oo))) is False

    n = Symbol('n', integer=True)
    assert Range(-3, 4, 1).is_subset(FiniteSet(-10, 10)) is False
    assert Range(S(10)**100).is_subset(FiniteSet(0, 1, 2)) is False
    assert Range(6, 0, -2).is_subset(FiniteSet(2, 4, 6)) is True
    assert Range(1, oo).is_subset(FiniteSet(1, 2)) is False
    assert Range(-oo, 1).is_subset(FiniteSet(1)) is False
    assert Range(3).is_subset(FiniteSet(0, 1, n)) is None
    assert Range(n, n + 2).is_subset(FiniteSet(n, n + 1)) is True
    assert Range(5).is_subset(Interval(0, 4, right_open=True)) is False
    #issue 19513
    assert imageset(Lambda(n, 1/n), S.Integers).is_subset(S.Reals) is None

def test_is_proper_subset():
    assert Interval(0, 1).is_proper_subset(Interval(0, 2)) is True
    assert Interval(0, 3).is_proper_subset(Interval(0, 2)) is False
    assert S.EmptySet.is_proper_subset(FiniteSet(1, 2, 3)) is True

    raises(ValueError, lambda: Interval(0, 1).is_proper_subset(0))


def test_is_superset():
    assert Interval(0, 1).is_superset(Interval(0, 2)) == False
    assert Interval(0, 3).is_superset(Interval(0, 2))

    assert FiniteSet(1, 2).is_superset(FiniteSet(1, 2, 3, 4)) == False
    assert FiniteSet(4, 5).is_superset(FiniteSet(1, 2, 3, 4)) == False
    assert FiniteSet(1).is_superset(Interval(0, 2)) == False
    assert FiniteSet(1, 2).is_superset(Interval(0, 2, True, True)) == False
    assert (Interval(1, 2) + FiniteSet(3)).is_superset(
        Interval(0, 2, False, True) + FiniteSet(2, 3)) == False

    assert Interval(3, 4).is_superset(Union(Interval(0, 1), Interval(2, 5))) == False

    assert FiniteSet(1, 2, 3, 4).is_superset(Interval(0, 5)) == False
    assert S.EmptySet.is_superset(FiniteSet(1, 2, 3)) == False

    assert Interval(0, 1).is_superset(S.EmptySet) == True
    assert S.EmptySet.is_superset(S.EmptySet) == True

    raises(ValueError, lambda: S.EmptySet.is_superset(1))

    # tests for the issuperset alias
    assert Interval(0, 1).issuperset(S.EmptySet) == True
    assert S.EmptySet.issuperset(S.EmptySet) == True


def test_is_proper_superset():
    assert Interval(0, 1).is_proper_superset(Interval(0, 2)) is False
    assert Interval(0, 3).is_proper_superset(Interval(0, 2)) is True
    assert FiniteSet(1, 2, 3).is_proper_superset(S.EmptySet) is True

    raises(ValueError, lambda: Interval(0, 1).is_proper_superset(0))


def test_contains():
    assert Interval(0, 2).contains(1) is S.true
    assert Interval(0, 2).contains(3) is S.false
    assert Interval(0, 2, True, False).contains(0) is S.false
    assert Interval(0, 2, True, False).contains(2) is S.true
    assert Interval(0, 2, False, True).contains(0) is S.true
    assert Interval(0, 2, False, True).contains(2) is S.false
    assert Interval(0, 2, True, True).contains(0) is S.false
    assert Interval(0, 2, True, True).contains(2) is S.false

    assert (Interval(0, 2) in Interval(0, 2)) is False

    assert FiniteSet(1, 2, 3).contains(2) is S.true
    assert FiniteSet(1, 2, Symbol('x')).contains(Symbol('x')) is S.true

    assert FiniteSet(y)._contains(x) == Eq(y, x, evaluate=False)
    raises(TypeError, lambda: x in FiniteSet(y))
    assert FiniteSet({x, y})._contains({x}) == Eq({x, y}, {x}, evaluate=False)
    assert FiniteSet({x, y}).subs(y, x)._contains({x}) is S.true
    assert FiniteSet({x, y}).subs(y, x+1)._contains({x}) is S.false

    # issue 8197
    from sympy.abc import a, b
    assert FiniteSet(b).contains(-a) == Eq(b, -a)
    assert FiniteSet(b).contains(a) == Eq(b, a)
    assert FiniteSet(a).contains(1) == Eq(a, 1)
    raises(TypeError, lambda: 1 in FiniteSet(a))

    # issue 8209
    rad1 = Pow(Pow(2, Rational(1, 3)) - 1, Rational(1, 3))
    rad2 = Pow(Rational(1, 9), Rational(1, 3)) - Pow(Rational(2, 9), Rational(1, 3)) + Pow(Rational(4, 9), Rational(1, 3))
    s1 = FiniteSet(rad1)
    s2 = FiniteSet(rad2)
    assert s1 - s2 == S.EmptySet

    items = [1, 2, S.Infinity, S('ham'), -1.1]
    fset = FiniteSet(*items)
    assert all(item in fset for item in items)
    assert all(fset.contains(item) is S.true for item in items)

    assert Union(Interval(0, 1), Interval(2, 5)).contains(3) is S.true
    assert Union(Interval(0, 1), Interval(2, 5)).contains(6) is S.false
    assert Union(Interval(0, 1), FiniteSet(2, 5)).contains(3) is S.false

    assert S.EmptySet.contains(1) is S.false
    assert FiniteSet(rootof(x**3 + x - 1, 0)).contains(S.Infinity) is S.false

    assert rootof(x**5 + x**3 + 1, 0) in S.Reals
    assert not rootof(x**5 + x**3 + 1, 1) in S.Reals

    # non-bool results
    assert Union(Interval(1, 2), Interval(3, 4)).contains(x) == \
        Or(And(S.One <= x, x <= 2), And(S(3) <= x, x <= 4))
    assert Intersection(Interval(1, x), Interval(2, 3)).contains(y) == \
        And(y <= 3, y <= x, S.One <= y, S(2) <= y)

    assert (S.Complexes).contains(S.ComplexInfinity) == S.false


def test_interval_symbolic():
    x = Symbol('x')
    e = Interval(0, 1)
    assert e.contains(x) == And(S.Zero <= x, x <= 1)
    raises(TypeError, lambda: x in e)
    e = Interval(0, 1, True, True)
    assert e.contains(x) == And(S.Zero < x, x < 1)
    c = Symbol('c', real=False)
    assert Interval(x, x + 1).contains(c) == False
    e = Symbol('e', extended_real=True)
    assert Interval(-oo, oo).contains(e) == And(
        S.NegativeInfinity < e, e < S.Infinity)


def test_union_contains():
    x = Symbol('x')
    i1 = Interval(0, 1)
    i2 = Interval(2, 3)
    i3 = Union(i1, i2)
    assert i3.as_relational(x) == Or(And(S.Zero <= x, x <= 1), And(S(2) <= x, x <= 3))
    raises(TypeError, lambda: x in i3)
    e = i3.contains(x)
    assert e == i3.as_relational(x)
    assert e.subs(x, -0.5) is false
    assert e.subs(x, 0.5) is true
    assert e.subs(x, 1.5) is false
    assert e.subs(x, 2.5) is true
    assert e.subs(x, 3.5) is false

    U = Interval(0, 2, True, True) + Interval(10, oo) + FiniteSet(-1, 2, 5, 6)
    assert all(el not in U for el in [0, 4, -oo])
    assert all(el in U for el in [2, 5, 10])


def test_is_number():
    assert Interval(0, 1).is_number is False
    assert Set().is_number is False


def test_Interval_is_left_unbounded():
    assert Interval(3, 4).is_left_unbounded is False
    assert Interval(-oo, 3).is_left_unbounded is True
    assert Interval(Float("-inf"), 3).is_left_unbounded is True


def test_Interval_is_right_unbounded():
    assert Interval(3, 4).is_right_unbounded is False
    assert Interval(3, oo).is_right_unbounded is True
    assert Interval(3, Float("+inf")).is_right_unbounded is True


def test_Interval_as_relational():
    x = Symbol('x')

    assert Interval(-1, 2, False, False).as_relational(x) == \
        And(Le(-1, x), Le(x, 2))
    assert Interval(-1, 2, True, False).as_relational(x) == \
        And(Lt(-1, x), Le(x, 2))
    assert Interval(-1, 2, False, True).as_relational(x) == \
        And(Le(-1, x), Lt(x, 2))
    assert Interval(-1, 2, True, True).as_relational(x) == \
        And(Lt(-1, x), Lt(x, 2))

    assert Interval(-oo, 2, right_open=False).as_relational(x) == And(Lt(-oo, x), Le(x, 2))
    assert Interval(-oo, 2, right_open=True).as_relational(x) == And(Lt(-oo, x), Lt(x, 2))

    assert Interval(-2, oo, left_open=False).as_relational(x) == And(Le(-2, x), Lt(x, oo))
    assert Interval(-2, oo, left_open=True).as_relational(x) == And(Lt(-2, x), Lt(x, oo))

    assert Interval(-oo, oo).as_relational(x) == And(Lt(-oo, x), Lt(x, oo))
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    assert Interval(x, y).as_relational(x) == (x <= y)
    assert Interval(y, x).as_relational(x) == (y <= x)


def test_Finite_as_relational():
    x = Symbol('x')
    y = Symbol('y')

    assert FiniteSet(1, 2).as_relational(x) == Or(Eq(x, 1), Eq(x, 2))
    assert FiniteSet(y, -5).as_relational(x) == Or(Eq(x, y), Eq(x, -5))


def test_Union_as_relational():
    x = Symbol('x')
    assert (Interval(0, 1) + FiniteSet(2)).as_relational(x) == \
        Or(And(Le(0, x), Le(x, 1)), Eq(x, 2))
    assert (Interval(0, 1, True, True) + FiniteSet(1)).as_relational(x) == \
        And(Lt(0, x), Le(x, 1))
    assert Or(x < 0, x > 0).as_set().as_relational(x) == \
        And((x > -oo), (x < oo), Ne(x, 0))
    assert (Interval.Ropen(1, 3) + Interval.Lopen(3, 5)
        ).as_relational(x) == And(Ne(x,3),(x>=1),(x<=5))


def test_Intersection_as_relational():
    x = Symbol('x')
    assert (Intersection(Interval(0, 1), FiniteSet(2),
            evaluate=False).as_relational(x)
            == And(And(Le(0, x), Le(x, 1)), Eq(x, 2)))


def test_Complement_as_relational():
    x = Symbol('x')
    expr = Complement(Interval(0, 1), FiniteSet(2), evaluate=False)
    assert expr.as_relational(x) == \
        And(Le(0, x), Le(x, 1), Ne(x, 2))


@XFAIL
def test_Complement_as_relational_fail():
    x = Symbol('x')
    expr = Complement(Interval(0, 1), FiniteSet(2), evaluate=False)
    # XXX This example fails because 0 <= x changes to x >= 0
    # during the evaluation.
    assert expr.as_relational(x) == \
            (0 <= x) & (x <= 1) & Ne(x, 2)


def test_SymmetricDifference_as_relational():
    x = Symbol('x')
    expr = SymmetricDifference(Interval(0, 1), FiniteSet(2), evaluate=False)
    assert expr.as_relational(x) == Xor(Eq(x, 2), Le(0, x) & Le(x, 1))


def test_EmptySet():
    assert S.EmptySet.as_relational(Symbol('x')) is S.false
    assert S.EmptySet.intersect(S.UniversalSet) == S.EmptySet
    assert S.EmptySet.boundary == S.EmptySet


def test_finite_basic():
    x = Symbol('x')
    A = FiniteSet(1, 2, 3)
    B = FiniteSet(3, 4, 5)
    AorB = Union(A, B)
    AandB = A.intersect(B)
    assert A.is_subset(AorB) and B.is_subset(AorB)
    assert AandB.is_subset(A)
    assert AandB == FiniteSet(3)

    assert A.inf == 1 and A.sup == 3
    assert AorB.inf == 1 and AorB.sup == 5
    assert FiniteSet(x, 1, 5).sup == Max(x, 5)
    assert FiniteSet(x, 1, 5).inf == Min(x, 1)

    # issue 7335
    assert FiniteSet(S.EmptySet) != S.EmptySet
    assert FiniteSet(FiniteSet(1, 2, 3)) != FiniteSet(1, 2, 3)
    assert FiniteSet((1, 2, 3)) != FiniteSet(1, 2, 3)

    # Ensure a variety of types can exist in a FiniteSet
    assert FiniteSet((1, 2), A, -5, x, 'eggs', x**2)

    assert (A > B) is False
    assert (A >= B) is False
    assert (A < B) is False
    assert (A <= B) is False
    assert AorB > A and AorB > B
    assert AorB >= A and AorB >= B
    assert A >= A and A <= A
    assert A >= AandB and B >= AandB
    assert A > AandB and B > AandB


def test_product_basic():
    H, T = 'H', 'T'
    unit_line = Interval(0, 1)
    d6 = FiniteSet(1, 2, 3, 4, 5, 6)
    d4 = FiniteSet(1, 2, 3, 4)
    coin = FiniteSet(H, T)

    square = unit_line * unit_line

    assert (0, 0) in square
    assert 0 not in square
    assert (H, T) in coin ** 2
    assert (.5, .5, .5) in (square * unit_line).flatten()
    assert ((.5, .5), .5) in square * unit_line
    assert (H, 3, 3) in (coin * d6 * d6).flatten()
    assert ((H, 3), 3) in coin * d6 * d6
    HH, TT = sympify(H), sympify(T)
    assert set(coin**2) == {(HH, HH), (HH, TT), (TT, HH), (TT, TT)}

    assert (d4*d4).is_subset(d6*d6)

    assert square.complement(Interval(-oo, oo)*Interval(-oo, oo)) == Union(
        (Interval(-oo, 0, True, True) +
         Interval(1, oo, True, True))*Interval(-oo, oo),
         Interval(-oo, oo)*(Interval(-oo, 0, True, True) +
                  Interval(1, oo, True, True)))

    assert (Interval(-5, 5)**3).is_subset(Interval(-10, 10)**3)
    assert not (Interval(-10, 10)**3).is_subset(Interval(-5, 5)**3)
    assert not (Interval(-5, 5)**2).is_subset(Interval(-10, 10)**3)

    assert (Interval(.2, .5)*FiniteSet(.5)).is_subset(square)  # segment in square

    assert len(coin*coin*coin) == 8
    assert len(S.EmptySet*S.EmptySet) == 0
    assert len(S.EmptySet*coin) == 0
    raises(TypeError, lambda: len(coin*Interval(0, 2)))


def test_real():
    x = Symbol('x', real=True)

    I = Interval(0, 5)
    J = Interval(10, 20)
    A = FiniteSet(1, 2, 30, x, S.Pi)
    B = FiniteSet(-4, 0)
    C = FiniteSet(100)
    D = FiniteSet('Ham', 'Eggs')

    assert all(s.is_subset(S.Reals) for s in [I, J, A, B, C])
    assert not D.is_subset(S.Reals)
    assert all((a + b).is_subset(S.Reals) for a in [I, J, A, B, C] for b in [I, J, A, B, C])
    assert not any((a + D).is_subset(S.Reals) for a in [I, J, A, B, C, D])

    assert not (I + A + D).is_subset(S.Reals)


def test_supinf():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)

    assert (Interval(0, 1) + FiniteSet(2)).sup == 2
    assert (Interval(0, 1) + FiniteSet(2)).inf == 0
    assert (Interval(0, 1) + FiniteSet(x)).sup == Max(1, x)
    assert (Interval(0, 1) + FiniteSet(x)).inf == Min(0, x)
    assert FiniteSet(5, 1, x).sup == Max(5, x)
    assert FiniteSet(5, 1, x).inf == Min(1, x)
    assert FiniteSet(5, 1, x, y).sup == Max(5, x, y)
    assert FiniteSet(5, 1, x, y).inf == Min(1, x, y)
    assert FiniteSet(5, 1, x, y, S.Infinity, S.NegativeInfinity).sup == \
        S.Infinity
    assert FiniteSet(5, 1, x, y, S.Infinity, S.NegativeInfinity).inf == \
        S.NegativeInfinity
    assert FiniteSet('Ham', 'Eggs').sup == Max('Ham', 'Eggs')


def test_universalset():
    U = S.UniversalSet
    x = Symbol('x')
    assert U.as_relational(x) is S.true
    assert U.union(Interval(2, 4)) == U

    assert U.intersect(Interval(2, 4)) == Interval(2, 4)
    assert U.measure is S.Infinity
    assert U.boundary == S.EmptySet
    assert U.contains(0) is S.true


def test_Union_of_ProductSets_shares():
    line = Interval(0, 2)
    points = FiniteSet(0, 1, 2)
    assert Union(line * line, line * points) == line * line


def test_Interval_free_symbols():
    # issue 6211
    assert Interval(0, 1).free_symbols == set()
    x = Symbol('x', real=True)
    assert Interval(0, x).free_symbols == {x}


def test_image_interval():
    x = Symbol('x', real=True)
    a = Symbol('a', real=True)
    assert imageset(x, 2*x, Interval(-2, 1)) == Interval(-4, 2)
    assert imageset(x, 2*x, Interval(-2, 1, True, False)) == \
        Interval(-4, 2, True, False)
    assert imageset(x, x**2, Interval(-2, 1, True, False)) == \
        Interval(0, 4, False, True)
    assert imageset(x, x**2, Interval(-2, 1)) == Interval(0, 4)
    assert imageset(x, x**2, Interval(-2, 1, True, False)) == \
        Interval(0, 4, False, True)
    assert imageset(x, x**2, Interval(-2, 1, True, True)) == \
        Interval(0, 4, False, True)
    assert imageset(x, (x - 2)**2, Interval(1, 3)) == Interval(0, 1)
    assert imageset(x, 3*x**4 - 26*x**3 + 78*x**2 - 90*x, Interval(0, 4)) == \
        Interval(-35, 0)  # Multiple Maxima
    assert imageset(x, x + 1/x, Interval(-oo, oo)) == Interval(-oo, -2) \
        + Interval(2, oo)  # Single Infinite discontinuity
    assert imageset(x, 1/x + 1/(x-1)**2, Interval(0, 2, True, False)) == \
        Interval(Rational(3, 2), oo, False)  # Multiple Infinite discontinuities

    # Test for Python lambda
    assert imageset(lambda x: 2*x, Interval(-2, 1)) == Interval(-4, 2)

    assert imageset(Lambda(x, a*x), Interval(0, 1)) == \
            ImageSet(Lambda(x, a*x), Interval(0, 1))

    assert imageset(Lambda(x, sin(cos(x))), Interval(0, 1)) == \
            ImageSet(Lambda(x, sin(cos(x))), Interval(0, 1))


def test_image_piecewise():
    f = Piecewise((x, x <= -1), (1/x**2, x <= 5), (x**3, True))
    f1 = Piecewise((0, x <= 1), (1, x <= 2), (2, True))
    assert imageset(x, f, Interval(-5, 5)) == Union(Interval(-5, -1), Interval(Rational(1, 25), oo))
    assert imageset(x, f1, Interval(1, 2)) == FiniteSet(0, 1)


@XFAIL  # See: https://github.com/sympy/sympy/pull/2723#discussion_r8659826
def test_image_Intersection():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    assert imageset(x, x**2, Interval(-2, 0).intersect(Interval(x, y))) == \
           Interval(0, 4).intersect(Interval(Min(x**2, y**2), Max(x**2, y**2)))


def test_image_FiniteSet():
    x = Symbol('x', real=True)
    assert imageset(x, 2*x, FiniteSet(1, 2, 3)) == FiniteSet(2, 4, 6)


def test_image_Union():
    x = Symbol('x', real=True)
    assert imageset(x, x**2, Interval(-2, 0) + FiniteSet(1, 2, 3)) == \
            (Interval(0, 4) + FiniteSet(9))


def test_image_EmptySet():
    x = Symbol('x', real=True)
    assert imageset(x, 2*x, S.EmptySet) == S.EmptySet


def test_issue_5724_7680():
    assert I not in S.Reals  # issue 7680
    assert Interval(-oo, oo).contains(I) is S.false


def test_boundary():
    assert FiniteSet(1).boundary == FiniteSet(1)
    assert all(Interval(0, 1, left_open, right_open).boundary == FiniteSet(0, 1)
            for left_open in (true, false) for right_open in (true, false))


def test_boundary_Union():
    assert (Interval(0, 1) + Interval(2, 3)).boundary == FiniteSet(0, 1, 2, 3)
    assert ((Interval(0, 1, False, True)
           + Interval(1, 2, True, False)).boundary == FiniteSet(0, 1, 2))

    assert (Interval(0, 1) + FiniteSet(2)).boundary == FiniteSet(0, 1, 2)
    assert Union(Interval(0, 10), Interval(5, 15), evaluate=False).boundary \
            == FiniteSet(0, 15)

    assert Union(Interval(0, 10), Interval(0, 1), evaluate=False).boundary \
            == FiniteSet(0, 10)
    assert Union(Interval(0, 10, True, True),
                 Interval(10, 15, True, True), evaluate=False).boundary \
            == FiniteSet(0, 10, 15)


@XFAIL
def test_union_boundary_of_joining_sets():
    """ Testing the boundary of unions is a hard problem """
    assert Union(Interval(0, 10), Interval(10, 15), evaluate=False).boundary \
            == FiniteSet(0, 15)


def test_boundary_ProductSet():
    open_square = Interval(0, 1, True, True) ** 2
    assert open_square.boundary == (FiniteSet(0, 1) * Interval(0, 1)
                                  + Interval(0, 1) * FiniteSet(0, 1))

    second_square = Interval(1, 2, True, True) * Interval(0, 1, True, True)
    assert (open_square + second_square).boundary == (
                FiniteSet(0, 1) * Interval(0, 1)
              + FiniteSet(1, 2) * Interval(0, 1)
              + Interval(0, 1) * FiniteSet(0, 1)
              + Interval(1, 2) * FiniteSet(0, 1))


def test_boundary_ProductSet_line():
    line_in_r2 = Interval(0, 1) * FiniteSet(0)
    assert line_in_r2.boundary == line_in_r2


def test_is_open():
    assert Interval(0, 1, False, False).is_open is False
    assert Interval(0, 1, True, False).is_open is False
    assert Interval(0, 1, True, True).is_open is True
    assert FiniteSet(1, 2, 3).is_open is False


def test_is_closed():
    assert Interval(0, 1, False, False).is_closed is True
    assert Interval(0, 1, True, False).is_closed is False
    assert FiniteSet(1, 2, 3).is_closed is True


def test_closure():
    assert Interval(0, 1, False, True).closure == Interval(0, 1, False, False)


def test_interior():
    assert Interval(0, 1, False, True).interior == Interval(0, 1, True, True)


def test_issue_7841():
    raises(TypeError, lambda: x in S.Reals)


def test_Eq():
    assert Eq(Interval(0, 1), Interval(0, 1))
    assert Eq(Interval(0, 1), Interval(0, 2)) == False

    s1 = FiniteSet(0, 1)
    s2 = FiniteSet(1, 2)

    assert Eq(s1, s1)
    assert Eq(s1, s2) == False

    assert Eq(s1*s2, s1*s2)
    assert Eq(s1*s2, s2*s1) == False

    assert unchanged(Eq, FiniteSet({x, y}), FiniteSet({x}))
    assert Eq(FiniteSet({x, y}).subs(y, x), FiniteSet({x})) is S.true
    assert Eq(FiniteSet({x, y}), FiniteSet({x})).subs(y, x) is S.true
    assert Eq(FiniteSet({x, y}).subs(y, x+1), FiniteSet({x})) is S.false
    assert Eq(FiniteSet({x, y}), FiniteSet({x})).subs(y, x+1) is S.false

    assert Eq(ProductSet({1}, {2}), Interval(1, 2)) is S.false
    assert Eq(ProductSet({1}), ProductSet({1}, {2})) is S.false

    assert Eq(FiniteSet(()), FiniteSet(1)) is S.false
    assert Eq(ProductSet(), FiniteSet(1)) is S.false

    i1 = Interval(0, 1)
    i2 = Interval(x, y)
    assert unchanged(Eq, ProductSet(i1, i1), ProductSet(i2, i2))


def test_SymmetricDifference():
    A = FiniteSet(0, 1, 2, 3, 4, 5)
    B = FiniteSet(2, 4, 6, 8, 10)
    C = Interval(8, 10)

    assert SymmetricDifference(A, B, evaluate=False).is_iterable is True
    assert SymmetricDifference(A, C, evaluate=False).is_iterable is None
    assert FiniteSet(*SymmetricDifference(A, B, evaluate=False)) == \
        FiniteSet(0, 1, 3, 5, 6, 8, 10)
    raises(TypeError,
        lambda: FiniteSet(*SymmetricDifference(A, C, evaluate=False)))

    assert SymmetricDifference(FiniteSet(0, 1, 2, 3, 4, 5), \
            FiniteSet(2, 4, 6, 8, 10)) == FiniteSet(0, 1, 3, 5, 6, 8, 10)
    assert SymmetricDifference(FiniteSet(2, 3, 4), FiniteSet(2, 3, 4 ,5)) \
            == FiniteSet(5)
    assert FiniteSet(1, 2, 3, 4, 5) ^ FiniteSet(1, 2, 5, 6) == \
            FiniteSet(3, 4, 6)
    assert Set(S(1), S(2), S(3)) ^ Set(S(2), S(3), S(4)) == Union(Set(S(1), S(2), S(3)) - Set(S(2), S(3), S(4)), \
            Set(S(2), S(3), S(4)) - Set(S(1), S(2), S(3)))
    assert Interval(0, 4) ^ Interval(2, 5) == Union(Interval(0, 4) - \
            Interval(2, 5), Interval(2, 5) - Interval(0, 4))


def test_issue_9536():
    from sympy.functions.elementary.exponential import log
    a = Symbol('a', real=True)
    assert FiniteSet(log(a)).intersect(S.Reals) == Intersection(S.Reals, FiniteSet(log(a)))


def test_issue_9637():
    n = Symbol('n')
    a = FiniteSet(n)
    b = FiniteSet(2, n)
    assert Complement(S.Reals, a) == Complement(S.Reals, a, evaluate=False)
    assert Complement(Interval(1, 3), a) == Complement(Interval(1, 3), a, evaluate=False)
    assert Complement(Interval(1, 3), b) == \
        Complement(Union(Interval(1, 2, False, True), Interval(2, 3, True, False)), a)
    assert Complement(a, S.Reals) == Complement(a, S.Reals, evaluate=False)
    assert Complement(a, Interval(1, 3)) == Complement(a, Interval(1, 3), evaluate=False)


def test_issue_9808():
    # See https://github.com/sympy/sympy/issues/16342
    assert Complement(FiniteSet(y), FiniteSet(1)) == Complement(FiniteSet(y), FiniteSet(1), evaluate=False)
    assert Complement(FiniteSet(1, 2, x), FiniteSet(x, y, 2, 3)) == \
        Complement(FiniteSet(1), FiniteSet(y), evaluate=False)


def test_issue_9956():
    assert Union(Interval(-oo, oo), FiniteSet(1)) == Interval(-oo, oo)
    assert Interval(-oo, oo).contains(1) is S.true


def test_issue_Symbol_inter():
    i = Interval(0, oo)
    r = S.Reals
    mat = Matrix([0, 0, 0])
    assert Intersection(r, i, FiniteSet(m), FiniteSet(m, n)) == \
        Intersection(i, FiniteSet(m))
    assert Intersection(FiniteSet(1, m, n), FiniteSet(m, n, 2), i) == \
        Intersection(i, FiniteSet(m, n))
    assert Intersection(FiniteSet(m, n, x), FiniteSet(m, z), r) == \
        Intersection(Intersection({m, z}, {m, n, x}), r)
    assert Intersection(FiniteSet(m, n, 3), FiniteSet(m, n, x), r) == \
        Intersection(FiniteSet(3, m, n), FiniteSet(m, n, x), r, evaluate=False)
    assert Intersection(FiniteSet(m, n, 3), FiniteSet(m, n, 2, 3), r) == \
        Intersection(FiniteSet(3, m, n), r)
    assert Intersection(r, FiniteSet(mat, 2, n), FiniteSet(0, mat, n)) == \
        Intersection(r, FiniteSet(n))
    assert Intersection(FiniteSet(sin(x), cos(x)), FiniteSet(sin(x), cos(x), 1), r) == \
        Intersection(r, FiniteSet(sin(x), cos(x)))
    assert Intersection(FiniteSet(x**2, 1, sin(x)), FiniteSet(x**2, 2, sin(x)), r) == \
        Intersection(r, FiniteSet(x**2, sin(x)))


def test_issue_11827():
    assert S.Naturals0**4


def test_issue_10113():
    f = x**2/(x**2 - 4)
    assert imageset(x, f, S.Reals) == Union(Interval(-oo, 0), Interval(1, oo, True, True))
    assert imageset(x, f, Interval(-2, 2)) == Interval(-oo, 0)
    assert imageset(x, f, Interval(-2, 3)) == Union(Interval(-oo, 0), Interval(Rational(9, 5), oo))


def test_issue_10248():
    raises(
        TypeError, lambda: list(Intersection(S.Reals, FiniteSet(x)))
    )
    A = Symbol('A', real=True)
    assert list(Intersection(S.Reals, FiniteSet(A))) == [A]


def test_issue_9447():
    a = Interval(0, 1) + Interval(2, 3)
    assert Complement(S.UniversalSet, a) == Complement(
            S.UniversalSet, Union(Interval(0, 1), Interval(2, 3)), evaluate=False)
    assert Complement(S.Naturals, a) == Complement(
            S.Naturals, Union(Interval(0, 1), Interval(2, 3)), evaluate=False)


def test_issue_10337():
    assert (FiniteSet(2) == 3) is False
    assert (FiniteSet(2) != 3) is True
    raises(TypeError, lambda: FiniteSet(2) < 3)
    raises(TypeError, lambda: FiniteSet(2) <= 3)
    raises(TypeError, lambda: FiniteSet(2) > 3)
    raises(TypeError, lambda: FiniteSet(2) >= 3)


def test_issue_10326():
    bad = [
        EmptySet,
        FiniteSet(1),
        Interval(1, 2),
        S.ComplexInfinity,
        S.ImaginaryUnit,
        S.Infinity,
        S.NaN,
        S.NegativeInfinity,
        ]
    interval = Interval(0, 5)
    for i in bad:
        assert i not in interval

    x = Symbol('x', real=True)
    nr = Symbol('nr', extended_real=False)
    assert x + 1 in Interval(x, x + 4)
    assert nr not in Interval(x, x + 4)
    assert Interval(1, 2) in FiniteSet(Interval(0, 5), Interval(1, 2))
    assert Interval(-oo, oo).contains(oo) is S.false
    assert Interval(-oo, oo).contains(-oo) is S.false


def test_issue_2799():
    U = S.UniversalSet
    a = Symbol('a', real=True)
    inf_interval = Interval(a, oo)
    R = S.Reals

    assert U + inf_interval == inf_interval + U
    assert U + R == R + U
    assert R + inf_interval == inf_interval + R


def test_issue_9706():
    assert Interval(-oo, 0).closure == Interval(-oo, 0, True, False)
    assert Interval(0, oo).closure == Interval(0, oo, False, True)
    assert Interval(-oo, oo).closure == Interval(-oo, oo)


def test_issue_8257():
    reals_plus_infinity = Union(Interval(-oo, oo), FiniteSet(oo))
    reals_plus_negativeinfinity = Union(Interval(-oo, oo), FiniteSet(-oo))
    assert Interval(-oo, oo) + FiniteSet(oo) == reals_plus_infinity
    assert FiniteSet(oo) + Interval(-oo, oo) == reals_plus_infinity
    assert Interval(-oo, oo) + FiniteSet(-oo) == reals_plus_negativeinfinity
    assert FiniteSet(-oo) + Interval(-oo, oo) == reals_plus_negativeinfinity


def test_issue_10931():
    assert S.Integers - S.Integers == EmptySet
    assert S.Integers - S.Reals == EmptySet


def test_issue_11174():
    soln = Intersection(Interval(-oo, oo), FiniteSet(-x), evaluate=False)
    assert Intersection(FiniteSet(-x), S.Reals) == soln

    soln = Intersection(S.Reals, FiniteSet(x), evaluate=False)
    assert Intersection(FiniteSet(x), S.Reals) == soln


def test_issue_18505():
    assert ImageSet(Lambda(n, sqrt(pi*n/2 - 1 + pi/2)), S.Integers).contains(0) == \
            Contains(0, ImageSet(Lambda(n, sqrt(pi*n/2 - 1 + pi/2)), S.Integers))


def test_finite_set_intersection():
    # The following should not produce recursion errors
    # Note: some of these are not completely correct. See
    # https://github.com/sympy/sympy/issues/16342.
    assert Intersection(FiniteSet(-oo, x), FiniteSet(x)) == FiniteSet(x)
    assert Intersection._handle_finite_sets([FiniteSet(-oo, x), FiniteSet(0, x)]) == FiniteSet(x)

    assert Intersection._handle_finite_sets([FiniteSet(-oo, x), FiniteSet(x)]) == FiniteSet(x)
    assert Intersection._handle_finite_sets([FiniteSet(2, 3, x, y), FiniteSet(1, 2, x)]) == \
        Intersection._handle_finite_sets([FiniteSet(1, 2, x), FiniteSet(2, 3, x, y)]) == \
        Intersection(FiniteSet(1, 2, x), FiniteSet(2, 3, x, y)) == \
        Intersection(FiniteSet(1, 2, x), FiniteSet(2, x, y))

    assert FiniteSet(1+x-y) & FiniteSet(1) == \
        FiniteSet(1) & FiniteSet(1+x-y) == \
        Intersection(FiniteSet(1+x-y), FiniteSet(1), evaluate=False)

    assert FiniteSet(1) & FiniteSet(x) == FiniteSet(x) & FiniteSet(1) == \
        Intersection(FiniteSet(1), FiniteSet(x), evaluate=False)

    assert FiniteSet({x}) & FiniteSet({x, y}) == \
        Intersection(FiniteSet({x}), FiniteSet({x, y}), evaluate=False)


def test_union_intersection_constructor():
    # The actual exception does not matter here, so long as these fail
    sets = [FiniteSet(1), FiniteSet(2)]
    raises(Exception, lambda: Union(sets))
    raises(Exception, lambda: Intersection(sets))
    raises(Exception, lambda: Union(tuple(sets)))
    raises(Exception, lambda: Intersection(tuple(sets)))
    raises(Exception, lambda: Union(i for i in sets))
    raises(Exception, lambda: Intersection(i for i in sets))

    # Python sets are treated the same as FiniteSet
    # The union of a single set (of sets) is the set (of sets) itself
    assert Union(set(sets)) == FiniteSet(*sets)
    assert Intersection(set(sets)) == FiniteSet(*sets)

    assert Union({1}, {2}) == FiniteSet(1, 2)
    assert Intersection({1, 2}, {2, 3}) == FiniteSet(2)


def test_Union_contains():
    assert zoo not in Union(
        Interval.open(-oo, 0), Interval.open(0, oo))


@XFAIL
def test_issue_16878b():
    # in intersection_sets for (ImageSet, Set) there is no code
    # that handles the base_set of S.Reals like there is
    # for Integers
    assert imageset(x, (x, x), S.Reals).is_subset(S.Reals**2) is True

def test_DisjointUnion():
    assert DisjointUnion(FiniteSet(1, 2, 3), FiniteSet(1, 2, 3), FiniteSet(1, 2, 3)).rewrite(Union) == (FiniteSet(1, 2, 3) * FiniteSet(0, 1, 2))
    assert DisjointUnion(Interval(1, 3), Interval(2, 4)).rewrite(Union) == Union(Interval(1, 3) * FiniteSet(0), Interval(2, 4) * FiniteSet(1))
    assert DisjointUnion(Interval(0, 5), Interval(0, 5)).rewrite(Union) == Union(Interval(0, 5) * FiniteSet(0), Interval(0, 5) * FiniteSet(1))
    assert DisjointUnion(Interval(-1, 2), S.EmptySet, S.EmptySet).rewrite(Union) == Interval(-1, 2) * FiniteSet(0)
    assert DisjointUnion(Interval(-1, 2)).rewrite(Union) == Interval(-1, 2) * FiniteSet(0)
    assert DisjointUnion(S.EmptySet, Interval(-1, 2), S.EmptySet).rewrite(Union) == Interval(-1, 2) * FiniteSet(1)
    assert DisjointUnion(Interval(-oo, oo)).rewrite(Union) == Interval(-oo, oo) * FiniteSet(0)
    assert DisjointUnion(S.EmptySet).rewrite(Union) == S.EmptySet
    assert DisjointUnion().rewrite(Union) == S.EmptySet
    raises(TypeError, lambda: DisjointUnion(Symbol('n')))

    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    assert DisjointUnion(FiniteSet(x), FiniteSet(y, z)).rewrite(Union) == (FiniteSet(x) * FiniteSet(0)) + (FiniteSet(y, z) * FiniteSet(1))

def test_DisjointUnion_is_empty():
    assert DisjointUnion(S.EmptySet).is_empty is True
    assert DisjointUnion(S.EmptySet, S.EmptySet).is_empty is True
    assert DisjointUnion(S.EmptySet, FiniteSet(1, 2, 3)).is_empty is False

def test_DisjointUnion_is_iterable():
    assert DisjointUnion(S.Integers, S.Naturals, S.Rationals).is_iterable is True
    assert DisjointUnion(S.EmptySet, S.Reals).is_iterable is False
    assert DisjointUnion(FiniteSet(1, 2, 3), S.EmptySet, FiniteSet(x, y)).is_iterable is True
    assert DisjointUnion(S.EmptySet, S.EmptySet).is_iterable is False

def test_DisjointUnion_contains():
    assert (0, 0) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    assert (0, 1) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    assert (0, 2) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    assert (1, 0) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    assert (1, 1) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    assert (1, 2) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    assert (2, 0) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    assert (2, 1) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    assert (2, 2) in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    assert (0, 1, 2) not in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    assert (0, 0.5) not in DisjointUnion(FiniteSet(0.5))
    assert (0, 5) not in DisjointUnion(FiniteSet(0, 1, 2), FiniteSet(0, 1, 2), FiniteSet(0, 1, 2))
    assert (x, 0) in DisjointUnion(FiniteSet(x, y, z), S.EmptySet, FiniteSet(y))
    assert (y, 0) in DisjointUnion(FiniteSet(x, y, z), S.EmptySet, FiniteSet(y))
    assert (z, 0) in DisjointUnion(FiniteSet(x, y, z), S.EmptySet, FiniteSet(y))
    assert (y, 2) in DisjointUnion(FiniteSet(x, y, z), S.EmptySet, FiniteSet(y))
    assert (0.5, 0) in DisjointUnion(Interval(0, 1), Interval(0, 2))
    assert (0.5, 1) in DisjointUnion(Interval(0, 1), Interval(0, 2))
    assert (1.5, 0) not in DisjointUnion(Interval(0, 1), Interval(0, 2))
    assert (1.5, 1) in DisjointUnion(Interval(0, 1), Interval(0, 2))

def test_DisjointUnion_iter():
    D = DisjointUnion(FiniteSet(3, 5, 7, 9), FiniteSet(x, y, z))
    it = iter(D)
    L1 = [(x, 1), (y, 1), (z, 1)]
    L2 = [(3, 0), (5, 0), (7, 0), (9, 0)]
    nxt = next(it)
    assert nxt in L2
    L2.remove(nxt)
    nxt = next(it)
    assert nxt in L1
    L1.remove(nxt)
    nxt = next(it)
    assert nxt in L2
    L2.remove(nxt)
    nxt = next(it)
    assert nxt in L1
    L1.remove(nxt)
    nxt = next(it)
    assert nxt in L2
    L2.remove(nxt)
    nxt = next(it)
    assert nxt in L1
    L1.remove(nxt)
    nxt = next(it)
    assert nxt in L2
    L2.remove(nxt)
    raises(StopIteration, lambda: next(it))

    raises(ValueError, lambda: iter(DisjointUnion(Interval(0, 1), S.EmptySet)))

def test_DisjointUnion_len():
    assert len(DisjointUnion(FiniteSet(3, 5, 7, 9), FiniteSet(x, y, z))) == 7
    assert len(DisjointUnion(S.EmptySet, S.EmptySet, FiniteSet(x, y, z), S.EmptySet)) == 3
    raises(ValueError, lambda: len(DisjointUnion(Interval(0, 1), S.EmptySet)))

def test_SetKind_ProductSet():
    p = ProductSet(FiniteSet(Matrix([1, 2])), FiniteSet(Matrix([1, 2])))
    mk = MatrixKind(NumberKind)
    k = SetKind(TupleKind(mk, mk))
    assert p.kind is k
    assert ProductSet(Interval(1, 2), FiniteSet(Matrix([1, 2]))).kind is SetKind(TupleKind(NumberKind, mk))

def test_SetKind_Interval():
    assert Interval(1, 2).kind is SetKind(NumberKind)

def test_SetKind_EmptySet_UniversalSet():
    assert S.UniversalSet.kind is SetKind(UndefinedKind)
    assert EmptySet.kind is SetKind()

def test_SetKind_FiniteSet():
    assert FiniteSet(1, Matrix([1, 2])).kind is SetKind(UndefinedKind)
    assert FiniteSet(1, 2).kind is SetKind(NumberKind)

def test_SetKind_Unions():
    assert Union(FiniteSet(Matrix([1, 2])), Interval(1, 2)).kind is SetKind(UndefinedKind)
    assert Union(Interval(1, 2), Interval(1, 7)).kind is SetKind(NumberKind)

def test_SetKind_DisjointUnion():
    A = FiniteSet(1, 2, 3)
    B = Interval(0, 5)
    assert DisjointUnion(A, B).kind is SetKind(NumberKind)

def test_SetKind_evaluate_False():
    U = lambda *args: Union(*args, evaluate=False)
    assert U({1}, EmptySet).kind is SetKind(NumberKind)
    assert U(Interval(1, 2), EmptySet).kind is SetKind(NumberKind)
    assert U({1}, S.UniversalSet).kind is SetKind(UndefinedKind)
    assert U(Interval(1, 2), Interval(4, 5),
            FiniteSet(1)).kind is SetKind(NumberKind)
    I = lambda *args: Intersection(*args, evaluate=False)
    assert I({1}, S.UniversalSet).kind is SetKind(NumberKind)
    assert I({1}, EmptySet).kind is SetKind()
    C = lambda *args: Complement(*args, evaluate=False)
    assert C(S.UniversalSet, {1, 2, 4, 5}).kind is SetKind(UndefinedKind)
    assert C({1, 2, 3, 4, 5}, EmptySet).kind is SetKind(NumberKind)
    assert C(EmptySet, {1, 2, 3, 4, 5}).kind is SetKind()

def test_SetKind_ImageSet_Special():
    f = ImageSet(Lambda(n, n ** 2), Interval(1, 4))
    assert (f - FiniteSet(3)).kind is SetKind(NumberKind)
    assert (f + Interval(16, 17)).kind is SetKind(NumberKind)
    assert (f + FiniteSet(17)).kind is SetKind(NumberKind)

def test_issue_20089():
    B = FiniteSet(FiniteSet(1, 2), FiniteSet(1))
    assert 1 not in B
    assert 1.0 not in B
    assert not Eq(1, FiniteSet(1, 2))
    assert FiniteSet(1) in B
    A = FiniteSet(1, 2)
    assert A in B
    assert B.issubset(B)
    assert not A.issubset(B)
    assert 1 in A
    C = FiniteSet(FiniteSet(1, 2), FiniteSet(1), 1, 2)
    assert A.issubset(C)
    assert B.issubset(C)

def test_issue_19378():
    a = FiniteSet(1, 2)
    b = ProductSet(a, a)
    c = FiniteSet((1, 1), (1, 2), (2, 1), (2, 2))
    assert b.is_subset(c) is True
    d = FiniteSet(1)
    assert b.is_subset(d) is False
    assert Eq(c, b).simplify() is S.true
    assert Eq(a, c).simplify() is S.false
    assert Eq({1}, {x}).simplify() == Eq({1}, {x})

def test_intersection_symbolic():
    n = Symbol('n')
    # These should not throw an error
    assert isinstance(Intersection(Range(n), Range(100)), Intersection)
    assert isinstance(Intersection(Range(n), Interval(1, 100)), Intersection)
    assert isinstance(Intersection(Range(100), Interval(1, n)), Intersection)


@XFAIL
def test_intersection_symbolic_failing():
    n = Symbol('n', integer=True, positive=True)
    assert Intersection(Range(10, n), Range(4, 500, 5)) == Intersection(
        Range(14, n), Range(14, 500, 5))
    assert Intersection(Interval(10, n), Range(4, 500, 5)) == Intersection(
        Interval(14, n), Range(14, 500, 5))


def test_issue_20379():
    #https://github.com/sympy/sympy/issues/20379
    x = pi - 3.14159265358979
    assert FiniteSet(x).evalf(2) == FiniteSet(Float('3.23108914886517e-15', 2))

def test_finiteset_simplify():
    S = FiniteSet(1, cos(1)**2 + sin(1)**2)
    assert S.simplify() == {1}

def test_issue_14336():
    #https://github.com/sympy/sympy/issues/14336
    U = S.Complexes
    x = Symbol("x")
    U -= U.intersect(Ne(x, 1).as_set())
    U -= U.intersect(S.true.as_set())

def test_issue_9855():
    #https://github.com/sympy/sympy/issues/9855
    x, y, z = symbols('x, y, z', real=True)
    s1 = Interval(1, x) & Interval(y, 2)
    s2 = Interval(1, 2)
    assert s1.is_subset(s2) == None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\apply\test_frame_apply.py ===
from datetime import datetime
import warnings

import numpy as np
import pytest

from pandas.compat import is_platform_arm

from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
from pandas.util.version import Version


@pytest.fixture
def int_frame_const_col():
    """
    Fixture for DataFrame of ints which are constant per column

    Columns are ['A', 'B', 'C'], with values (per column): [1, 2, 3]
    """
    df = DataFrame(
        np.tile(np.arange(3, dtype="int64"), 6).reshape(6, -1) + 1,
        columns=["A", "B", "C"],
    )
    return df


@pytest.fixture(params=["python", pytest.param("numba", marks=pytest.mark.single_cpu)])
def engine(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def test_apply(float_frame, engine, request):
    if engine == "numba":
        mark = pytest.mark.xfail(reason="numba engine not supporting numpy ufunc yet")
        request.node.add_marker(mark)
    with np.errstate(all="ignore"):
        # ufunc
        result = np.sqrt(float_frame["A"])
        expected = float_frame.apply(np.sqrt, engine=engine)["A"]
        tm.assert_series_equal(result, expected)

        # aggregator
        result = float_frame.apply(np.mean, engine=engine)["A"]
        expected = np.mean(float_frame["A"])
        assert result == expected

        d = float_frame.index[0]
        result = float_frame.apply(np.mean, axis=1, engine=engine)
        expected = np.mean(float_frame.xs(d))
        assert result[d] == expected
        assert result.index is float_frame.index


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("raw", [True, False])
def test_apply_args(float_frame, axis, raw, engine, request):
    if engine == "numba":
        numba = pytest.importorskip("numba")
        if Version(numba.__version__) == Version("0.61") and is_platform_arm():
            pytest.skip(f"Segfaults on ARM platforms with numba {numba.__version__}")
        mark = pytest.mark.xfail(reason="numba engine doesn't support args")
        request.node.add_marker(mark)
    result = float_frame.apply(
        lambda x, y: x + y, axis, args=(1,), raw=raw, engine=engine
    )
    expected = float_frame + 1
    tm.assert_frame_equal(result, expected)


def test_apply_categorical_func():
    # GH 9573
    df = DataFrame({"c0": ["A", "A", "B", "B"], "c1": ["C", "C", "D", "D"]})
    result = df.apply(lambda ts: ts.astype("category"))

    assert result.shape == (4, 2)
    assert isinstance(result["c0"].dtype, CategoricalDtype)
    assert isinstance(result["c1"].dtype, CategoricalDtype)


def test_apply_axis1_with_ea():
    # GH#36785
    expected = DataFrame({"A": [Timestamp("2013-01-01", tz="UTC")]})
    result = expected.apply(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data, dtype",
    [(1, None), (1, CategoricalDtype([1])), (Timestamp("2013-01-01", tz="UTC"), None)],
)
def test_agg_axis1_duplicate_index(data, dtype):
    # GH 42380
    expected = DataFrame([[data], [data]], index=["a", "a"], dtype=dtype)
    result = expected.agg(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)


def test_apply_mixed_datetimelike():
    # mixed datetimelike
    # GH 7778
    expected = DataFrame(
        {
            "A": date_range("20130101", periods=3),
            "B": pd.to_timedelta(np.arange(3), unit="s"),
        }
    )
    result = expected.apply(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func", [np.sqrt, np.mean])
def test_apply_empty(func, engine):
    # empty
    empty_frame = DataFrame()

    result = empty_frame.apply(func, engine=engine)
    assert result.empty


def test_apply_float_frame(float_frame, engine):
    no_rows = float_frame[:0]
    result = no_rows.apply(lambda x: x.mean(), engine=engine)
    expected = Series(np.nan, index=float_frame.columns)
    tm.assert_series_equal(result, expected)

    no_cols = float_frame.loc[:, []]
    result = no_cols.apply(lambda x: x.mean(), axis=1, engine=engine)
    expected = Series(np.nan, index=float_frame.index)
    tm.assert_series_equal(result, expected)


def test_apply_empty_except_index(engine):
    # GH 2476
    expected = DataFrame(index=["a"])
    result = expected.apply(lambda x: x["a"], axis=1, engine=engine)
    tm.assert_frame_equal(result, expected)


def test_apply_with_reduce_empty():
    # reduce with an empty DataFrame
    empty_frame = DataFrame()

    x = []
    result = empty_frame.apply(x.append, axis=1, result_type="expand")
    tm.assert_frame_equal(result, empty_frame)
    result = empty_frame.apply(x.append, axis=1, result_type="reduce")
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)

    empty_with_cols = DataFrame(columns=["a", "b", "c"])
    result = empty_with_cols.apply(x.append, axis=1, result_type="expand")
    tm.assert_frame_equal(result, empty_with_cols)
    result = empty_with_cols.apply(x.append, axis=1, result_type="reduce")
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)

    # Ensure that x.append hasn't been called
    assert x == []


@pytest.mark.parametrize("func", ["sum", "prod", "any", "all"])
def test_apply_funcs_over_empty(func):
    # GH 28213
    df = DataFrame(columns=["a", "b", "c"])

    result = df.apply(getattr(np, func))
    expected = getattr(df, func)()
    if func in ("sum", "prod"):
        expected = expected.astype(float)
    tm.assert_series_equal(result, expected)


def test_nunique_empty():
    # GH 28213
    df = DataFrame(columns=["a", "b", "c"])

    result = df.nunique()
    expected = Series(0, index=df.columns)
    tm.assert_series_equal(result, expected)

    result = df.T.nunique()
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)


def test_apply_standard_nonunique():
    df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=["a", "a", "c"])

    result = df.apply(lambda s: s[0], axis=1)
    expected = Series([1, 4, 7], ["a", "a", "c"])
    tm.assert_series_equal(result, expected)

    result = df.T.apply(lambda s: s[0], axis=0)
    tm.assert_series_equal(result, expected)


def test_apply_broadcast_scalars(float_frame):
    # scalars
    result = float_frame.apply(np.mean, result_type="broadcast")
    expected = DataFrame([float_frame.mean()], index=float_frame.index)
    tm.assert_frame_equal(result, expected)


def test_apply_broadcast_scalars_axis1(float_frame):
    result = float_frame.apply(np.mean, axis=1, result_type="broadcast")
    m = float_frame.mean(axis=1)
    expected = DataFrame({c: m for c in float_frame.columns})
    tm.assert_frame_equal(result, expected)


def test_apply_broadcast_lists_columns(float_frame):
    # lists
    result = float_frame.apply(
        lambda x: list(range(len(float_frame.columns))),
        axis=1,
        result_type="broadcast",
    )
    m = list(range(len(float_frame.columns)))
    expected = DataFrame(
        [m] * len(float_frame.index),
        dtype="float64",
        index=float_frame.index,
        columns=float_frame.columns,
    )
    tm.assert_frame_equal(result, expected)


def test_apply_broadcast_lists_index(float_frame):
    result = float_frame.apply(
        lambda x: list(range(len(float_frame.index))), result_type="broadcast"
    )
    m = list(range(len(float_frame.index)))
    expected = DataFrame(
        {c: m for c in float_frame.columns},
        dtype="float64",
        index=float_frame.index,
    )
    tm.assert_frame_equal(result, expected)


def test_apply_broadcast_list_lambda_func(int_frame_const_col):
    # preserve columns
    df = int_frame_const_col
    result = df.apply(lambda x: [1, 2, 3], axis=1, result_type="broadcast")
    tm.assert_frame_equal(result, df)


def test_apply_broadcast_series_lambda_func(int_frame_const_col):
    df = int_frame_const_col
    result = df.apply(
        lambda x: Series([1, 2, 3], index=list("abc")),
        axis=1,
        result_type="broadcast",
    )
    expected = df.copy()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("axis", [0, 1])
def test_apply_raw_float_frame(float_frame, axis, engine):
    if engine == "numba":
        pytest.skip("numba can't handle when UDF returns None.")

    def _assert_raw(x):
        assert isinstance(x, np.ndarray)
        assert x.ndim == 1

    float_frame.apply(_assert_raw, axis=axis, engine=engine, raw=True)


@pytest.mark.parametrize("axis", [0, 1])
def test_apply_raw_float_frame_lambda(float_frame, axis, engine):
    result = float_frame.apply(np.mean, axis=axis, engine=engine, raw=True)
    expected = float_frame.apply(lambda x: x.values.mean(), axis=axis)
    tm.assert_series_equal(result, expected)


def test_apply_raw_float_frame_no_reduction(float_frame, engine):
    # no reduction
    result = float_frame.apply(lambda x: x * 2, engine=engine, raw=True)
    expected = float_frame * 2
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("axis", [0, 1])
def test_apply_raw_mixed_type_frame(axis, engine):
    if engine == "numba":
        pytest.skip("isinstance check doesn't work with numba")

    def _assert_raw(x):
        assert isinstance(x, np.ndarray)
        assert x.ndim == 1

    # Mixed dtype (GH-32423)
    df = DataFrame(
        {
            "a": 1.0,
            "b": 2,
            "c": "foo",
            "float32": np.array([1.0] * 10, dtype="float32"),
            "int32": np.array([1] * 10, dtype="int32"),
        },
        index=np.arange(10),
    )
    df.apply(_assert_raw, axis=axis, engine=engine, raw=True)


def test_apply_axis1(float_frame):
    d = float_frame.index[0]
    result = float_frame.apply(np.mean, axis=1)[d]
    expected = np.mean(float_frame.xs(d))
    assert result == expected


def test_apply_mixed_dtype_corner():
    df = DataFrame({"A": ["foo"], "B": [1.0]})
    result = df[:0].apply(np.mean, axis=1)
    # the result here is actually kind of ambiguous, should it be a Series
    # or a DataFrame?
    expected = Series(np.nan, index=pd.Index([], dtype="int64"))
    tm.assert_series_equal(result, expected)


def test_apply_mixed_dtype_corner_indexing():
    df = DataFrame({"A": ["foo"], "B": [1.0]})
    result = df.apply(lambda x: x["A"], axis=1)
    expected = Series(["foo"], index=[0])
    tm.assert_series_equal(result, expected)

    result = df.apply(lambda x: x["B"], axis=1)
    expected = Series([1.0], index=[0])
    tm.assert_series_equal(result, expected)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("ax", ["index", "columns"])
@pytest.mark.parametrize(
    "func", [lambda x: x, lambda x: x.mean()], ids=["identity", "mean"]
)
@pytest.mark.parametrize("raw", [True, False])
@pytest.mark.parametrize("axis", [0, 1])
def test_apply_empty_infer_type(ax, func, raw, axis, engine, request):
    df = DataFrame(**{ax: ["a", "b", "c"]})

    with np.errstate(all="ignore"):
        test_res = func(np.array([], dtype="f8"))
        is_reduction = not isinstance(test_res, np.ndarray)

        result = df.apply(func, axis=axis, engine=engine, raw=raw)
        if is_reduction:
            agg_axis = df._get_agg_axis(axis)
            assert isinstance(result, Series)
            assert result.index is agg_axis
        else:
            assert isinstance(result, DataFrame)


def test_apply_empty_infer_type_broadcast():
    no_cols = DataFrame(index=["a", "b", "c"])
    result = no_cols.apply(lambda x: x.mean(), result_type="broadcast")
    assert isinstance(result, DataFrame)


def test_apply_with_args_kwds_add_some(float_frame):
    def add_some(x, howmuch=0):
        return x + howmuch

    result = float_frame.apply(add_some, howmuch=2)
    expected = float_frame.apply(lambda x: x + 2)
    tm.assert_frame_equal(result, expected)


def test_apply_with_args_kwds_agg_and_add(float_frame):
    def agg_and_add(x, howmuch=0):
        return x.mean() + howmuch

    result = float_frame.apply(agg_and_add, howmuch=2)
    expected = float_frame.apply(lambda x: x.mean() + 2)
    tm.assert_series_equal(result, expected)


def test_apply_with_args_kwds_subtract_and_divide(float_frame):
    def subtract_and_divide(x, sub, divide=1):
        return (x - sub) / divide

    result = float_frame.apply(subtract_and_divide, args=(2,), divide=2)
    expected = float_frame.apply(lambda x: (x - 2.0) / 2.0)
    tm.assert_frame_equal(result, expected)


def test_apply_yield_list(float_frame):
    result = float_frame.apply(list)
    tm.assert_frame_equal(result, float_frame)


def test_apply_reduce_Series(float_frame):
    float_frame.iloc[::2, float_frame.columns.get_loc("A")] = np.nan
    expected = float_frame.mean(1)
    result = float_frame.apply(np.mean, axis=1)
    tm.assert_series_equal(result, expected)


def test_apply_reduce_to_dict():
    # GH 25196 37544
    data = DataFrame([[1, 2], [3, 4]], columns=["c0", "c1"], index=["i0", "i1"])

    result = data.apply(dict, axis=0)
    expected = Series([{"i0": 1, "i1": 3}, {"i0": 2, "i1": 4}], index=data.columns)
    tm.assert_series_equal(result, expected)

    result = data.apply(dict, axis=1)
    expected = Series([{"c0": 1, "c1": 2}, {"c0": 3, "c1": 4}], index=data.index)
    tm.assert_series_equal(result, expected)


def test_apply_differently_indexed():
    df = DataFrame(np.random.default_rng(2).standard_normal((20, 10)))

    result = df.apply(Series.describe, axis=0)
    expected = DataFrame({i: v.describe() for i, v in df.items()}, columns=df.columns)
    tm.assert_frame_equal(result, expected)

    result = df.apply(Series.describe, axis=1)
    expected = DataFrame({i: v.describe() for i, v in df.T.items()}, columns=df.index).T
    tm.assert_frame_equal(result, expected)


def test_apply_bug():
    # GH 6125
    positions = DataFrame(
        [
            [1, "ABC0", 50],
            [1, "YUM0", 20],
            [1, "DEF0", 20],
            [2, "ABC1", 50],
            [2, "YUM1", 20],
            [2, "DEF1", 20],
        ],
        columns=["a", "market", "position"],
    )

    def f(r):
        return r["market"]

    expected = positions.apply(f, axis=1)

    positions = DataFrame(
        [
            [datetime(2013, 1, 1), "ABC0", 50],
            [datetime(2013, 1, 2), "YUM0", 20],
            [datetime(2013, 1, 3), "DEF0", 20],
            [datetime(2013, 1, 4), "ABC1", 50],
            [datetime(2013, 1, 5), "YUM1", 20],
            [datetime(2013, 1, 6), "DEF1", 20],
        ],
        columns=["a", "market", "position"],
    )
    result = positions.apply(f, axis=1)
    tm.assert_series_equal(result, expected)


def test_apply_convert_objects():
    expected = DataFrame(
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

    result = expected.apply(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)


def test_apply_attach_name(float_frame):
    result = float_frame.apply(lambda x: x.name)
    expected = Series(float_frame.columns, index=float_frame.columns)
    tm.assert_series_equal(result, expected)


def test_apply_attach_name_axis1(float_frame):
    result = float_frame.apply(lambda x: x.name, axis=1)
    expected = Series(float_frame.index, index=float_frame.index)
    tm.assert_series_equal(result, expected)


def test_apply_attach_name_non_reduction(float_frame):
    # non-reductions
    result = float_frame.apply(lambda x: np.repeat(x.name, len(x)))
    expected = DataFrame(
        np.tile(float_frame.columns, (len(float_frame.index), 1)),
        index=float_frame.index,
        columns=float_frame.columns,
    )
    tm.assert_frame_equal(result, expected)


def test_apply_attach_name_non_reduction_axis1(float_frame):
    result = float_frame.apply(lambda x: np.repeat(x.name, len(x)), axis=1)
    expected = Series(
        np.repeat(t[0], len(float_frame.columns)) for t in float_frame.itertuples()
    )
    expected.index = float_frame.index
    tm.assert_series_equal(result, expected)


def test_apply_multi_index():
    index = MultiIndex.from_arrays([["a", "a", "b"], ["c", "d", "d"]])
    s = DataFrame([[1, 2], [3, 4], [5, 6]], index=index, columns=["col1", "col2"])
    result = s.apply(lambda x: Series({"min": min(x), "max": max(x)}), 1)
    expected = DataFrame([[1, 2], [3, 4], [5, 6]], index=index, columns=["min", "max"])
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    "df, dicts",
    [
        [
            DataFrame([["foo", "bar"], ["spam", "eggs"]]),
            Series([{0: "foo", 1: "spam"}, {0: "bar", 1: "eggs"}]),
        ],
        [DataFrame([[0, 1], [2, 3]]), Series([{0: 0, 1: 2}, {0: 1, 1: 3}])],
    ],
)
def test_apply_dict(df, dicts):
    # GH 8735
    fn = lambda x: x.to_dict()
    reduce_true = df.apply(fn, result_type="reduce")
    reduce_false = df.apply(fn, result_type="expand")
    reduce_none = df.apply(fn)

    tm.assert_series_equal(reduce_true, dicts)
    tm.assert_frame_equal(reduce_false, df)
    tm.assert_series_equal(reduce_none, dicts)


def test_apply_non_numpy_dtype():
    # GH 12244
    df = DataFrame({"dt": date_range("2015-01-01", periods=3, tz="Europe/Brussels")})
    result = df.apply(lambda x: x)
    tm.assert_frame_equal(result, df)

    result = df.apply(lambda x: x + pd.Timedelta("1day"))
    expected = DataFrame(
        {"dt": date_range("2015-01-02", periods=3, tz="Europe/Brussels")}
    )
    tm.assert_frame_equal(result, expected)


def test_apply_non_numpy_dtype_category():
    df = DataFrame({"dt": ["a", "b", "c", "a"]}, dtype="category")
    result = df.apply(lambda x: x)
    tm.assert_frame_equal(result, df)


def test_apply_dup_names_multi_agg():
    # GH 21063
    df = DataFrame([[0, 1], [2, 3]], columns=["a", "a"])
    expected = DataFrame([[0, 1]], columns=["a", "a"], index=["min"])
    result = df.agg(["min"])

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("op", ["apply", "agg"])
def test_apply_nested_result_axis_1(op):
    # GH 13820
    def apply_list(row):
        return [2 * row["A"], 2 * row["C"], 2 * row["B"]]

    df = DataFrame(np.zeros((4, 4)), columns=list("ABCD"))
    result = getattr(df, op)(apply_list, axis=1)
    expected = Series(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    tm.assert_series_equal(result, expected)


def test_apply_noreduction_tzaware_object():
    # https://github.com/pandas-dev/pandas/issues/31505
    expected = DataFrame(
        {"foo": [Timestamp("2020", tz="UTC")]}, dtype="datetime64[ns, UTC]"
    )
    result = expected.apply(lambda x: x)
    tm.assert_frame_equal(result, expected)
    result = expected.apply(lambda x: x.copy())
    tm.assert_frame_equal(result, expected)


def test_apply_function_runs_once():
    # https://github.com/pandas-dev/pandas/issues/30815

    df = DataFrame({"a": [1, 2, 3]})
    names = []  # Save row names function is applied to

    def reducing_function(row):
        names.append(row.name)

    def non_reducing_function(row):
        names.append(row.name)
        return row

    for func in [reducing_function, non_reducing_function]:
        del names[:]

        df.apply(func, axis=1)
        assert names == list(df.index)


def test_apply_raw_function_runs_once(engine):
    # https://github.com/pandas-dev/pandas/issues/34506
    if engine == "numba":
        pytest.skip("appending to list outside of numba func is not supported")

    df = DataFrame({"a": [1, 2, 3]})
    values = []  # Save row values function is applied to

    def reducing_function(row):
        values.extend(row)

    def non_reducing_function(row):
        values.extend(row)
        return row

    for func in [reducing_function, non_reducing_function]:
        del values[:]

        df.apply(func, engine=engine, raw=True, axis=1)
        assert values == list(df.a.to_list())


def test_apply_with_byte_string():
    # GH 34529
    df = DataFrame(np.array([b"abcd", b"efgh"]), columns=["col"])
    expected = DataFrame(np.array([b"abcd", b"efgh"]), columns=["col"], dtype=object)
    # After we make the apply we expect a dataframe just
    # like the original but with the object datatype
    result = df.apply(lambda x: x.astype("object"))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("val", ["asd", 12, None, np.nan])
def test_apply_category_equalness(val):
    # Check if categorical comparisons on apply, GH 21239
    df_values = ["asd", None, 12, "asd", "cde", np.nan]
    df = DataFrame({"a": df_values}, dtype="category")

    result = df.a.apply(lambda x: x == val)
    expected = Series(
        [np.nan if pd.isnull(x) else x == val for x in df_values], name="a"
    )
    tm.assert_series_equal(result, expected)


# the user has supplied an opaque UDF where
# they are transforming the input that requires
# us to infer the output


def test_infer_row_shape():
    # GH 17437
    # if row shape is changing, infer it
    df = DataFrame(np.random.default_rng(2).random((10, 2)))
    result = df.apply(np.fft.fft, axis=0).shape
    assert result == (10, 2)

    result = df.apply(np.fft.rfft, axis=0).shape
    assert result == (6, 2)


@pytest.mark.parametrize(
    "ops, by_row, expected",
    [
        ({"a": lambda x: x + 1}, "compat", DataFrame({"a": [2, 3]})),
        ({"a": lambda x: x + 1}, False, DataFrame({"a": [2, 3]})),
        ({"a": lambda x: x.sum()}, "compat", Series({"a": 3})),
        ({"a": lambda x: x.sum()}, False, Series({"a": 3})),
        (
            {"a": ["sum", np.sum, lambda x: x.sum()]},
            "compat",
            DataFrame({"a": [3, 3, 3]}, index=["sum", "sum", "<lambda>"]),
        ),
        (
            {"a": ["sum", np.sum, lambda x: x.sum()]},
            False,
            DataFrame({"a": [3, 3, 3]}, index=["sum", "sum", "<lambda>"]),
        ),
        ({"a": lambda x: 1}, "compat", DataFrame({"a": [1, 1]})),
        ({"a": lambda x: 1}, False, Series({"a": 1})),
    ],
)
def test_dictlike_lambda(ops, by_row, expected):
    # GH53601
    df = DataFrame({"a": [1, 2]})
    result = df.apply(ops, by_row=by_row)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "ops",
    [
        {"a": lambda x: x + 1},
        {"a": lambda x: x.sum()},
        {"a": ["sum", np.sum, lambda x: x.sum()]},
        {"a": lambda x: 1},
    ],
)
def test_dictlike_lambda_raises(ops):
    # GH53601
    df = DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="by_row=True not allowed"):
        df.apply(ops, by_row=True)


def test_with_dictlike_columns():
    # GH 17602
    df = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    result = df.apply(lambda x: {"s": x["a"] + x["b"]}, axis=1)
    expected = Series([{"s": 3} for t in df.itertuples()])
    tm.assert_series_equal(result, expected)

    df["tm"] = [
        Timestamp("2017-05-01 00:00:00"),
        Timestamp("2017-05-02 00:00:00"),
    ]
    result = df.apply(lambda x: {"s": x["a"] + x["b"]}, axis=1)
    tm.assert_series_equal(result, expected)

    # compose a series
    result = (df["a"] + df["b"]).apply(lambda x: {"s": x})
    expected = Series([{"s": 3}, {"s": 3}])
    tm.assert_series_equal(result, expected)


def test_with_dictlike_columns_with_datetime():
    # GH 18775
    df = DataFrame()
    df["author"] = ["X", "Y", "Z"]
    df["publisher"] = ["BBC", "NBC", "N24"]
    df["date"] = pd.to_datetime(
        ["17-10-2010 07:15:30", "13-05-2011 08:20:35", "15-01-2013 09:09:09"],
        dayfirst=True,
    )
    result = df.apply(lambda x: {}, axis=1)
    expected = Series([{}, {}, {}])
    tm.assert_series_equal(result, expected)


def test_with_dictlike_columns_with_infer():
    # GH 17602
    df = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    result = df.apply(lambda x: {"s": x["a"] + x["b"]}, axis=1, result_type="expand")
    expected = DataFrame({"s": [3, 3]})
    tm.assert_frame_equal(result, expected)

    df["tm"] = [
        Timestamp("2017-05-01 00:00:00"),
        Timestamp("2017-05-02 00:00:00"),
    ]
    result = df.apply(lambda x: {"s": x["a"] + x["b"]}, axis=1, result_type="expand")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "ops, by_row, expected",
    [
        ([lambda x: x + 1], "compat", DataFrame({("a", "<lambda>"): [2, 3]})),
        ([lambda x: x + 1], False, DataFrame({("a", "<lambda>"): [2, 3]})),
        ([lambda x: x.sum()], "compat", DataFrame({"a": [3]}, index=["<lambda>"])),
        ([lambda x: x.sum()], False, DataFrame({"a": [3]}, index=["<lambda>"])),
        (
            ["sum", np.sum, lambda x: x.sum()],
            "compat",
            DataFrame({"a": [3, 3, 3]}, index=["sum", "sum", "<lambda>"]),
        ),
        (
            ["sum", np.sum, lambda x: x.sum()],
            False,
            DataFrame({"a": [3, 3, 3]}, index=["sum", "sum", "<lambda>"]),
        ),
        (
            [lambda x: x + 1, lambda x: 3],
            "compat",
            DataFrame([[2, 3], [3, 3]], columns=[["a", "a"], ["<lambda>", "<lambda>"]]),
        ),
        (
            [lambda x: 2, lambda x: 3],
            False,
            DataFrame({"a": [2, 3]}, ["<lambda>", "<lambda>"]),
        ),
    ],
)
def test_listlike_lambda(ops, by_row, expected):
    # GH53601
    df = DataFrame({"a": [1, 2]})
    result = df.apply(ops, by_row=by_row)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "ops",
    [
        [lambda x: x + 1],
        [lambda x: x.sum()],
        ["sum", np.sum, lambda x: x.sum()],
        [lambda x: x + 1, lambda x: 3],
    ],
)
def test_listlike_lambda_raises(ops):
    # GH53601
    df = DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="by_row=True not allowed"):
        df.apply(ops, by_row=True)


def test_with_listlike_columns():
    # GH 17348
    df = DataFrame(
        {
            "a": Series(np.random.default_rng(2).standard_normal(4)),
            "b": ["a", "list", "of", "words"],
            "ts": date_range("2016-10-01", periods=4, freq="h"),
        }
    )

    result = df[["a", "b"]].apply(tuple, axis=1)
    expected = Series([t[1:] for t in df[["a", "b"]].itertuples()])
    tm.assert_series_equal(result, expected)

    result = df[["a", "ts"]].apply(tuple, axis=1)
    expected = Series([t[1:] for t in df[["a", "ts"]].itertuples()])
    tm.assert_series_equal(result, expected)


def test_with_listlike_columns_returning_list():
    # GH 18919
    df = DataFrame({"x": Series([["a", "b"], ["q"]]), "y": Series([["z"], ["q", "t"]])})
    df.index = MultiIndex.from_tuples([("i0", "j0"), ("i1", "j1")])

    result = df.apply(lambda row: [el for el in row["x"] if el in row["y"]], axis=1)
    expected = Series([[], ["q"]], index=df.index)
    tm.assert_series_equal(result, expected)


def test_infer_output_shape_columns():
    # GH 18573

    df = DataFrame(
        {
            "number": [1.0, 2.0],
            "string": ["foo", "bar"],
            "datetime": [
                Timestamp("2017-11-29 03:30:00"),
                Timestamp("2017-11-29 03:45:00"),
            ],
        }
    )
    result = df.apply(lambda row: (row.number, row.string), axis=1)
    expected = Series([(t.number, t.string) for t in df.itertuples()])
    tm.assert_series_equal(result, expected)


def test_infer_output_shape_listlike_columns():
    # GH 16353

    df = DataFrame(
        np.random.default_rng(2).standard_normal((6, 3)), columns=["A", "B", "C"]
    )

    result = df.apply(lambda x: [1, 2, 3], axis=1)
    expected = Series([[1, 2, 3] for t in df.itertuples()])
    tm.assert_series_equal(result, expected)

    result = df.apply(lambda x: [1, 2], axis=1)
    expected = Series([[1, 2] for t in df.itertuples()])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("val", [1, 2])
def test_infer_output_shape_listlike_columns_np_func(val):
    # GH 17970
    df = DataFrame({"a": [1, 2, 3]}, index=list("abc"))

    result = df.apply(lambda row: np.ones(val), axis=1)
    expected = Series([np.ones(val) for t in df.itertuples()], index=df.index)
    tm.assert_series_equal(result, expected)


def test_infer_output_shape_listlike_columns_with_timestamp():
    # GH 17892
    df = DataFrame(
        {
            "a": [
                Timestamp("2010-02-01"),
                Timestamp("2010-02-04"),
                Timestamp("2010-02-05"),
                Timestamp("2010-02-06"),
            ],
            "b": [9, 5, 4, 3],
            "c": [5, 3, 4, 2],
            "d": [1, 2, 3, 4],
        }
    )

    def fun(x):
        return (1, 2)

    result = df.apply(fun, axis=1)
    expected = Series([(1, 2) for t in df.itertuples()])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("lst", [[1, 2, 3], [1, 2]])
def test_consistent_coerce_for_shapes(lst):
    # we want column names to NOT be propagated
    # just because the shape matches the input shape
    df = DataFrame(
        np.random.default_rng(2).standard_normal((4, 3)), columns=["A", "B", "C"]
    )

    result = df.apply(lambda x: lst, axis=1)
    expected = Series([lst for t in df.itertuples()])
    tm.assert_series_equal(result, expected)


def test_consistent_names(int_frame_const_col):
    # if a Series is returned, we should use the resulting index names
    df = int_frame_const_col

    result = df.apply(
        lambda x: Series([1, 2, 3], index=["test", "other", "cols"]), axis=1
    )
    expected = int_frame_const_col.rename(
        columns={"A": "test", "B": "other", "C": "cols"}
    )
    tm.assert_frame_equal(result, expected)

    result = df.apply(lambda x: Series([1, 2], index=["test", "other"]), axis=1)
    expected = expected[["test", "other"]]
    tm.assert_frame_equal(result, expected)


def test_result_type(int_frame_const_col):
    # result_type should be consistent no matter which
    # path we take in the code
    df = int_frame_const_col

    result = df.apply(lambda x: [1, 2, 3], axis=1, result_type="expand")
    expected = df.copy()
    expected.columns = [0, 1, 2]
    tm.assert_frame_equal(result, expected)


def test_result_type_shorter_list(int_frame_const_col):
    # result_type should be consistent no matter which
    # path we take in the code
    df = int_frame_const_col
    result = df.apply(lambda x: [1, 2], axis=1, result_type="expand")
    expected = df[["A", "B"]].copy()
    expected.columns = [0, 1]
    tm.assert_frame_equal(result, expected)


def test_result_type_broadcast(int_frame_const_col, request, engine):
    # result_type should be consistent no matter which
    # path we take in the code
    if engine == "numba":
        mark = pytest.mark.xfail(reason="numba engine doesn't support list return")
        request.node.add_marker(mark)
    df = int_frame_const_col
    # broadcast result
    result = df.apply(
        lambda x: [1, 2, 3], axis=1, result_type="broadcast", engine=engine
    )
    expected = df.copy()
    tm.assert_frame_equal(result, expected)


def test_result_type_broadcast_series_func(int_frame_const_col, engine, request):
    # result_type should be consistent no matter which
    # path we take in the code
    if engine == "numba":
        mark = pytest.mark.xfail(
            reason="numba Series constructor only support ndarrays not list data"
        )
        request.node.add_marker(mark)
    df = int_frame_const_col
    columns = ["other", "col", "names"]
    result = df.apply(
        lambda x: Series([1, 2, 3], index=columns),
        axis=1,
        result_type="broadcast",
        engine=engine,
    )
    expected = df.copy()
    tm.assert_frame_equal(result, expected)


def test_result_type_series_result(int_frame_const_col, engine, request):
    # result_type should be consistent no matter which
    # path we take in the code
    if engine == "numba":
        mark = pytest.mark.xfail(
            reason="numba Series constructor only support ndarrays not list data"
        )
        request.node.add_marker(mark)
    df = int_frame_const_col
    # series result
    result = df.apply(lambda x: Series([1, 2, 3], index=x.index), axis=1, engine=engine)
    expected = df.copy()
    tm.assert_frame_equal(result, expected)


def test_result_type_series_result_other_index(int_frame_const_col, engine, request):
    # result_type should be consistent no matter which
    # path we take in the code

    if engine == "numba":
        mark = pytest.mark.xfail(
            reason="no support in numba Series constructor for list of columns"
        )
        request.node.add_marker(mark)
    df = int_frame_const_col
    # series result with other index
    columns = ["other", "col", "names"]
    result = df.apply(lambda x: Series([1, 2, 3], index=columns), axis=1, engine=engine)
    expected = df.copy()
    expected.columns = columns
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "box",
    [lambda x: list(x), lambda x: tuple(x), lambda x: np.array(x, dtype="int64")],
    ids=["list", "tuple", "array"],
)
def test_consistency_for_boxed(box, int_frame_const_col):
    # passing an array or list should not affect the output shape
    df = int_frame_const_col

    result = df.apply(lambda x: box([1, 2]), axis=1)
    expected = Series([box([1, 2]) for t in df.itertuples()])
    tm.assert_series_equal(result, expected)

    result = df.apply(lambda x: box([1, 2]), axis=1, result_type="expand")
    expected = int_frame_const_col[["A", "B"]].rename(columns={"A": 0, "B": 1})
    tm.assert_frame_equal(result, expected)


def test_agg_transform(axis, float_frame):
    other_axis = 1 if axis in {0, "index"} else 0

    with np.errstate(all="ignore"):
        f_abs = np.abs(float_frame)
        f_sqrt = np.sqrt(float_frame)

        # ufunc
        expected = f_sqrt.copy()
        result = float_frame.apply(np.sqrt, axis=axis)
        tm.assert_frame_equal(result, expected)

        # list-like
        result = float_frame.apply([np.sqrt], axis=axis)
        expected = f_sqrt.copy()
        if axis in {0, "index"}:
            expected.columns = MultiIndex.from_product([float_frame.columns, ["sqrt"]])
        else:
            expected.index = MultiIndex.from_product([float_frame.index, ["sqrt"]])
        tm.assert_frame_equal(result, expected)

        # multiple items in list
        # these are in the order as if we are applying both
        # functions per series and then concatting
        result = float_frame.apply([np.abs, np.sqrt], axis=axis)
        expected = zip_frames([f_abs, f_sqrt], axis=other_axis)
        if axis in {0, "index"}:
            expected.columns = MultiIndex.from_product(
                [float_frame.columns, ["absolute", "sqrt"]]
            )
        else:
            expected.index = MultiIndex.from_product(
                [float_frame.index, ["absolute", "sqrt"]]
            )
        tm.assert_frame_equal(result, expected)


def test_demo():
    # demonstration tests
    df = DataFrame({"A": range(5), "B": 5})

    result = df.agg(["min", "max"])
    expected = DataFrame(
        {"A": [0, 4], "B": [5, 5]}, columns=["A", "B"], index=["min", "max"]
    )
    tm.assert_frame_equal(result, expected)


def test_demo_dict_agg():
    # demonstration tests
    df = DataFrame({"A": range(5), "B": 5})
    result = df.agg({"A": ["min", "max"], "B": ["sum", "max"]})
    expected = DataFrame(
        {"A": [4.0, 0.0, np.nan], "B": [5.0, np.nan, 25.0]},
        columns=["A", "B"],
        index=["max", "min", "sum"],
    )
    tm.assert_frame_equal(result.reindex_like(expected), expected)


def test_agg_with_name_as_column_name():
    # GH 36212 - Column name is "name"
    data = {"name": ["foo", "bar"]}
    df = DataFrame(data)

    # result's name should be None
    result = df.agg({"name": "count"})
    expected = Series({"name": 2})
    tm.assert_series_equal(result, expected)

    # Check if name is still preserved when aggregating series instead
    result = df["name"].agg({"name": "count"})
    expected = Series({"name": 2}, name="name")
    tm.assert_series_equal(result, expected)


def test_agg_multiple_mixed():
    # GH 20909
    mdf = DataFrame(
        {
            "A": [1, 2, 3],
            "B": [1.0, 2.0, 3.0],
            "C": ["foo", "bar", "baz"],
        }
    )
    expected = DataFrame(
        {
            "A": [1, 6],
            "B": [1.0, 6.0],
            "C": ["bar", "foobarbaz"],
        },
        index=["min", "sum"],
    )
    # sorted index
    result = mdf.agg(["min", "sum"])
    tm.assert_frame_equal(result, expected)

    result = mdf[["C", "B", "A"]].agg(["sum", "min"])
    # GH40420: the result of .agg should have an index that is sorted
    # according to the arguments provided to agg.
    expected = expected[["C", "B", "A"]].reindex(["sum", "min"])
    tm.assert_frame_equal(result, expected)


def test_agg_multiple_mixed_raises():
    # GH 20909
    mdf = DataFrame(
        {
            "A": [1, 2, 3],
            "B": [1.0, 2.0, 3.0],
            "C": ["foo", "bar", "baz"],
            "D": date_range("20130101", periods=3),
        }
    )

    # sorted index
    msg = "does not support reduction"
    with pytest.raises(TypeError, match=msg):
        mdf.agg(["min", "sum"])

    with pytest.raises(TypeError, match=msg):
        mdf[["D", "C", "B", "A"]].agg(["sum", "min"])


def test_agg_reduce(axis, float_frame):
    other_axis = 1 if axis in {0, "index"} else 0
    name1, name2 = float_frame.axes[other_axis].unique()[:2].sort_values()

    # all reducers
    expected = pd.concat(
        [
            float_frame.mean(axis=axis),
            float_frame.max(axis=axis),
            float_frame.sum(axis=axis),
        ],
        axis=1,
    )
    expected.columns = ["mean", "max", "sum"]
    expected = expected.T if axis in {0, "index"} else expected

    result = float_frame.agg(["mean", "max", "sum"], axis=axis)
    tm.assert_frame_equal(result, expected)

    # dict input with scalars
    func = {name1: "mean", name2: "sum"}
    result = float_frame.agg(func, axis=axis)
    expected = Series(
        [
            float_frame.loc(other_axis)[name1].mean(),
            float_frame.loc(other_axis)[name2].sum(),
        ],
        index=[name1, name2],
    )
    tm.assert_series_equal(result, expected)

    # dict input with lists
    func = {name1: ["mean"], name2: ["sum"]}
    result = float_frame.agg(func, axis=axis)
    expected = DataFrame(
        {
            name1: Series([float_frame.loc(other_axis)[name1].mean()], index=["mean"]),
            name2: Series([float_frame.loc(other_axis)[name2].sum()], index=["sum"]),
        }
    )
    expected = expected.T if axis in {1, "columns"} else expected
    tm.assert_frame_equal(result, expected)

    # dict input with lists with multiple
    func = {name1: ["mean", "sum"], name2: ["sum", "max"]}
    result = float_frame.agg(func, axis=axis)
    expected = pd.concat(
        {
            name1: Series(
                [
                    float_frame.loc(other_axis)[name1].mean(),
                    float_frame.loc(other_axis)[name1].sum(),
                ],
                index=["mean", "sum"],
            ),
            name2: Series(
                [
                    float_frame.loc(other_axis)[name2].sum(),
                    float_frame.loc(other_axis)[name2].max(),
                ],
                index=["sum", "max"],
            ),
        },
        axis=1,
    )
    expected = expected.T if axis in {1, "columns"} else expected
    tm.assert_frame_equal(result, expected)


def test_nuiscance_columns():
    # GH 15015
    df = DataFrame(
        {
            "A": [1, 2, 3],
            "B": [1.0, 2.0, 3.0],
            "C": ["foo", "bar", "baz"],
            "D": date_range("20130101", periods=3),
        }
    )

    result = df.agg("min")
    expected = Series([1, 1.0, "bar", Timestamp("20130101")], index=df.columns)
    tm.assert_series_equal(result, expected)

    result = df.agg(["min"])
    expected = DataFrame(
        [[1, 1.0, "bar", Timestamp("20130101").as_unit("ns")]],
        index=["min"],
        columns=df.columns,
    )
    tm.assert_frame_equal(result, expected)

    msg = "does not support reduction"
    with pytest.raises(TypeError, match=msg):
        df.agg("sum")

    result = df[["A", "B", "C"]].agg("sum")
    expected = Series([6, 6.0, "foobarbaz"], index=["A", "B", "C"])
    tm.assert_series_equal(result, expected)

    msg = "does not support reduction"
    with pytest.raises(TypeError, match=msg):
        df.agg(["sum"])


@pytest.mark.parametrize("how", ["agg", "apply"])
def test_non_callable_aggregates(how):
    # GH 16405
    # 'size' is a property of frame/series
    # validate that this is working
    # GH 39116 - expand to apply
    df = DataFrame(
        {"A": [None, 2, 3], "B": [1.0, np.nan, 3.0], "C": ["foo", None, "bar"]}
    )

    # Function aggregate
    result = getattr(df, how)({"A": "count"})
    expected = Series({"A": 2})

    tm.assert_series_equal(result, expected)

    # Non-function aggregate
    result = getattr(df, how)({"A": "size"})
    expected = Series({"A": 3})

    tm.assert_series_equal(result, expected)

    # Mix function and non-function aggs
    result1 = getattr(df, how)(["count", "size"])
    result2 = getattr(df, how)(
        {"A": ["count", "size"], "B": ["count", "size"], "C": ["count", "size"]}
    )
    expected = DataFrame(
        {
            "A": {"count": 2, "size": 3},
            "B": {"count": 2, "size": 3},
            "C": {"count": 2, "size": 3},
        }
    )

    tm.assert_frame_equal(result1, result2, check_like=True)
    tm.assert_frame_equal(result2, expected, check_like=True)

    # Just functional string arg is same as calling df.arg()
    result = getattr(df, how)("count")
    expected = df.count()

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("how", ["agg", "apply"])
def test_size_as_str(how, axis):
    # GH 39934
    df = DataFrame(
        {"A": [None, 2, 3], "B": [1.0, np.nan, 3.0], "C": ["foo", None, "bar"]}
    )
    # Just a string attribute arg same as calling df.arg
    # on the columns
    result = getattr(df, how)("size", axis=axis)
    if axis in (0, "index"):
        expected = Series(df.shape[0], index=df.columns)
    else:
        expected = Series(df.shape[1], index=df.index)
    tm.assert_series_equal(result, expected)


def test_agg_listlike_result():
    # GH-29587 user defined function returning list-likes
    df = DataFrame({"A": [2, 2, 3], "B": [1.5, np.nan, 1.5], "C": ["foo", None, "bar"]})

    def func(group_col):
        return list(group_col.dropna().unique())

    result = df.agg(func)
    expected = Series([[2, 3], [1.5], ["foo", "bar"]], index=["A", "B", "C"])
    tm.assert_series_equal(result, expected)

    result = df.agg([func])
    expected = expected.to_frame("func").T
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "args, kwargs",
    [
        ((1, 2, 3), {}),
        ((8, 7, 15), {}),
        ((1, 2), {}),
        ((1,), {"b": 2}),
        ((), {"a": 1, "b": 2}),
        ((), {"a": 2, "b": 1}),
        ((), {"a": 1, "b": 2, "c": 3}),
    ],
)
def test_agg_args_kwargs(axis, args, kwargs):
    def f(x, a, b, c=3):
        return x.sum() + (a + b) / c

    df = DataFrame([[1, 2], [3, 4]])

    if axis == 0:
        expected = Series([5.0, 7.0])
    else:
        expected = Series([4.0, 8.0])

    result = df.agg(f, axis, *args, **kwargs)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("num_cols", [2, 3, 5])
def test_frequency_is_original(num_cols, engine, request):
    # GH 22150
    if engine == "numba":
        mark = pytest.mark.xfail(reason="numba engine only supports numeric indices")
        request.node.add_marker(mark)
    index = pd.DatetimeIndex(["1950-06-30", "1952-10-24", "1953-05-29"])
    original = index.copy()
    df = DataFrame(1, index=index, columns=range(num_cols))
    df.apply(lambda x: x, engine=engine)
    assert index.freq == original.freq


def test_apply_datetime_tz_issue(engine, request):
    # GH 29052

    if engine == "numba":
        mark = pytest.mark.xfail(
            reason="numba engine doesn't support non-numeric indexes"
        )
        request.node.add_marker(mark)

    timestamps = [
        Timestamp("2019-03-15 12:34:31.909000+0000", tz="UTC"),
        Timestamp("2019-03-15 12:34:34.359000+0000", tz="UTC"),
        Timestamp("2019-03-15 12:34:34.660000+0000", tz="UTC"),
    ]
    df = DataFrame(data=[0, 1, 2], index=timestamps)
    result = df.apply(lambda x: x.name, axis=1, engine=engine)
    expected = Series(index=timestamps, data=timestamps)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("df", [DataFrame({"A": ["a", None], "B": ["c", "d"]})])
@pytest.mark.parametrize("method", ["min", "max", "sum"])
def test_mixed_column_raises(df, method, using_infer_string):
    # GH 16832
    if method == "sum":
        msg = r'can only concatenate str \(not "int"\) to str|does not support'
    else:
        msg = "not supported between instances of 'str' and 'float'"
    if not using_infer_string:
        with pytest.raises(TypeError, match=msg):
            getattr(df, method)()
    else:
        getattr(df, method)()


@pytest.mark.parametrize("col", [1, 1.0, True, "a", np.nan])
def test_apply_dtype(col):
    # GH 31466
    df = DataFrame([[1.0, col]], columns=["a", "b"])
    result = df.apply(lambda x: x.dtype)
    expected = df.dtypes

    tm.assert_series_equal(result, expected)


def test_apply_mutating(using_array_manager, using_copy_on_write, warn_copy_on_write):
    # GH#35462 case where applied func pins a new BlockManager to a row
    df = DataFrame({"a": range(100), "b": range(100, 200)})
    df_orig = df.copy()

    def func(row):
        mgr = row._mgr
        row.loc["a"] += 1
        assert row._mgr is not mgr
        return row

    expected = df.copy()
    expected["a"] += 1

    with tm.assert_cow_warning(warn_copy_on_write):
        result = df.apply(func, axis=1)

    tm.assert_frame_equal(result, expected)
    if using_copy_on_write or using_array_manager:
        # INFO(CoW) With copy on write, mutating a viewing row doesn't mutate the parent
        # INFO(ArrayManager) With BlockManager, the row is a view and mutated in place,
        # with ArrayManager the row is not a view, and thus not mutated in place
        tm.assert_frame_equal(df, df_orig)
    else:
        tm.assert_frame_equal(df, result)


def test_apply_empty_list_reduce():
    # GH#35683 get columns correct
    df = DataFrame([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], columns=["a", "b"])

    result = df.apply(lambda x: [], result_type="reduce")
    expected = Series({"a": [], "b": []}, dtype=object)
    tm.assert_series_equal(result, expected)


def test_apply_no_suffix_index(engine, request):
    # GH36189
    if engine == "numba":
        mark = pytest.mark.xfail(
            reason="numba engine doesn't support list-likes/dict-like callables"
        )
        request.node.add_marker(mark)
    pdf = DataFrame([[4, 9]] * 3, columns=["A", "B"])
    result = pdf.apply(["sum", lambda x: x.sum(), lambda x: x.sum()], engine=engine)
    expected = DataFrame(
        {"A": [12, 12, 12], "B": [27, 27, 27]}, index=["sum", "<lambda>", "<lambda>"]
    )

    tm.assert_frame_equal(result, expected)


def test_apply_raw_returns_string(engine):
    # https://github.com/pandas-dev/pandas/issues/35940
    if engine == "numba":
        pytest.skip("No object dtype support in numba")
    df = DataFrame({"A": ["aa", "bbb"]})
    result = df.apply(lambda x: x[0], engine=engine, axis=1, raw=True)
    expected = Series(["aa", "bbb"])
    tm.assert_series_equal(result, expected)


def test_aggregation_func_column_order():
    # GH40420: the result of .agg should have an index that is sorted
    # according to the arguments provided to agg.
    df = DataFrame(
        [
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
            (4, 5, 4),
            (5, 6, 6),
            (6, 7, 7),
        ],
        columns=("att1", "att2", "att3"),
    )

    def sum_div2(s):
        return s.sum() / 2

    aggs = ["sum", sum_div2, "count", "min"]
    result = df.agg(aggs)
    expected = DataFrame(
        {
            "att1": [21.0, 10.5, 6.0, 1.0],
            "att2": [18.0, 9.0, 6.0, 0.0],
            "att3": [17.0, 8.5, 6.0, 0.0],
        },
        index=["sum", "sum_div2", "count", "min"],
    )
    tm.assert_frame_equal(result, expected)


def test_apply_getitem_axis_1(engine, request):
    # GH 13427
    if engine == "numba":
        mark = pytest.mark.xfail(
            reason="numba engine not supporting duplicate index values"
        )
        request.node.add_marker(mark)
    df = DataFrame({"a": [0, 1, 2], "b": [1, 2, 3]})
    result = df[["a", "a"]].apply(
        lambda x: x.iloc[0] + x.iloc[1], axis=1, engine=engine
    )
    expected = Series([0, 2, 4])
    tm.assert_series_equal(result, expected)


def test_nuisance_depr_passes_through_warnings():
    # GH 43740
    # DataFrame.agg with list-likes may emit warnings for both individual
    # args and for entire columns, but we only want to emit once. We
    # catch and suppress the warnings for individual args, but need to make
    # sure if some other warnings were raised, they get passed through to
    # the user.

    def expected_warning(x):
        warnings.warn("Hello, World!")
        return x.sum()

    df = DataFrame({"a": [1, 2, 3]})
    with tm.assert_produces_warning(UserWarning, match="Hello, World!"):
        df.agg([expected_warning])


def test_apply_type():
    # GH 46719
    df = DataFrame(
        {"col1": [3, "string", float], "col2": [0.25, datetime(2020, 1, 1), np.nan]},
        index=["a", "b", "c"],
    )

    # axis=0
    result = df.apply(type, axis=0)
    expected = Series({"col1": Series, "col2": Series})
    tm.assert_series_equal(result, expected)

    # axis=1
    result = df.apply(type, axis=1)
    expected = Series({"a": Series, "b": Series, "c": Series})
    tm.assert_series_equal(result, expected)


def test_apply_on_empty_dataframe(engine):
    # GH 39111
    df = DataFrame({"a": [1, 2], "b": [3, 0]})
    result = df.head(0).apply(lambda x: max(x["a"], x["b"]), axis=1, engine=engine)
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)


def test_apply_return_list():
    df = DataFrame({"a": [1, 2], "b": [2, 3]})
    result = df.apply(lambda x: [x.values])
    expected = DataFrame({"a": [[1, 2]], "b": [[2, 3]]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "test, constant",
    [
        ({"a": [1, 2, 3], "b": [1, 1, 1]}, {"a": [1, 2, 3], "b": [1]}),
        ({"a": [2, 2, 2], "b": [1, 1, 1]}, {"a": [2], "b": [1]}),
    ],
)
def test_unique_agg_type_is_series(test, constant):
    # GH#22558
    df1 = DataFrame(test)
    expected = Series(data=constant, index=["a", "b"], dtype="object")
    aggregation = {"a": "unique", "b": "unique"}

    result = df1.agg(aggregation)

    tm.assert_series_equal(result, expected)


def test_any_apply_keyword_non_zero_axis_regression():
    # https://github.com/pandas-dev/pandas/issues/48656
    df = DataFrame({"A": [1, 2, 0], "B": [0, 2, 0], "C": [0, 0, 0]})
    expected = Series([True, True, False])
    tm.assert_series_equal(df.any(axis=1), expected)

    result = df.apply("any", axis=1)
    tm.assert_series_equal(result, expected)

    result = df.apply("any", 1)
    tm.assert_series_equal(result, expected)


def test_agg_mapping_func_deprecated():
    # GH 53325
    df = DataFrame({"x": [1, 2, 3]})

    def foo1(x, a=1, c=0):
        return x + a + c

    def foo2(x, b=2, c=0):
        return x + b + c

    # single func already takes the vectorized path
    result = df.agg(foo1, 0, 3, c=4)
    expected = df + 7
    tm.assert_frame_equal(result, expected)

    msg = "using .+ in Series.agg cannot aggregate and"

    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.agg([foo1, foo2], 0, 3, c=4)
    expected = DataFrame(
        [[8, 8], [9, 9], [10, 10]], columns=[["x", "x"], ["foo1", "foo2"]]
    )
    tm.assert_frame_equal(result, expected)

    # TODO: the result below is wrong, should be fixed (GH53325)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.agg({"x": foo1}, 0, 3, c=4)
    expected = DataFrame([2, 3, 4], columns=["x"])
    tm.assert_frame_equal(result, expected)


def test_agg_std():
    df = DataFrame(np.arange(6).reshape(3, 2), columns=["A", "B"])

    with tm.assert_produces_warning(FutureWarning, match="using DataFrame.std"):
        result = df.agg(np.std)
    expected = Series({"A": 2.0, "B": 2.0}, dtype=float)
    tm.assert_series_equal(result, expected)

    with tm.assert_produces_warning(FutureWarning, match="using Series.std"):
        result = df.agg([np.std])
    expected = DataFrame({"A": 2.0, "B": 2.0}, index=["std"])
    tm.assert_frame_equal(result, expected)


def test_agg_dist_like_and_nonunique_columns():
    # GH#51099
    df = DataFrame(
        {"A": [None, 2, 3], "B": [1.0, np.nan, 3.0], "C": ["foo", None, "bar"]}
    )
    df.columns = ["A", "A", "C"]

    result = df.agg({"A": "count"})
    expected = df["A"].count()
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\excel\test_readers.py ===
from __future__ import annotations

from datetime import (
    datetime,
    time,
)
from functools import partial
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from urllib.error import URLError
from zipfile import BadZipFile

import numpy as np
import pytest

from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    read_csv,
)
import pandas._testing as tm

if is_platform_windows():
    pytestmark = pytest.mark.single_cpu

read_ext_params = [".xls", ".xlsx", ".xlsm", ".xlsb", ".ods"]
engine_params = [
    # Add any engines to test here
    # When defusedxml is installed it triggers deprecation warnings for
    # xlrd and openpyxl, so catch those here
    pytest.param(
        "xlrd",
        marks=[
            td.skip_if_no("xlrd"),
        ],
    ),
    pytest.param(
        "openpyxl",
        marks=[
            td.skip_if_no("openpyxl"),
        ],
    ),
    pytest.param(
        None,
        marks=[
            td.skip_if_no("xlrd"),
        ],
    ),
    pytest.param("pyxlsb", marks=td.skip_if_no("pyxlsb")),
    pytest.param("odf", marks=td.skip_if_no("odf")),
    pytest.param("calamine", marks=td.skip_if_no("python_calamine")),
]


def _is_valid_engine_ext_pair(engine, read_ext: str) -> bool:
    """
    Filter out invalid (engine, ext) pairs instead of skipping, as that
    produces 500+ pytest.skips.
    """
    engine = engine.values[0]
    if engine == "openpyxl" and read_ext == ".xls":
        return False
    if engine == "odf" and read_ext != ".ods":
        return False
    if read_ext == ".ods" and engine not in {"odf", "calamine"}:
        return False
    if engine == "pyxlsb" and read_ext != ".xlsb":
        return False
    if read_ext == ".xlsb" and engine not in {"pyxlsb", "calamine"}:
        return False
    if engine == "xlrd" and read_ext != ".xls":
        return False
    return True


def _transfer_marks(engine, read_ext):
    """
    engine gives us a pytest.param object with some marks, read_ext is just
    a string.  We need to generate a new pytest.param inheriting the marks.
    """
    values = engine.values + (read_ext,)
    new_param = pytest.param(values, marks=engine.marks)
    return new_param


@pytest.fixture(
    params=[
        _transfer_marks(eng, ext)
        for eng in engine_params
        for ext in read_ext_params
        if _is_valid_engine_ext_pair(eng, ext)
    ],
    ids=str,
)
def engine_and_read_ext(request):
    """
    Fixture for Excel reader engine and read_ext, only including valid pairs.
    """
    return request.param


@pytest.fixture
def engine(engine_and_read_ext):
    engine, read_ext = engine_and_read_ext
    return engine


@pytest.fixture
def read_ext(engine_and_read_ext):
    engine, read_ext = engine_and_read_ext
    return read_ext


@pytest.fixture
def df_ref(datapath):
    """
    Obtain the reference data from read_csv with the Python engine.
    """
    filepath = datapath("io", "data", "csv", "test1.csv")
    df_ref = read_csv(filepath, index_col=0, parse_dates=True, engine="python")
    return df_ref


def get_exp_unit(read_ext: str, engine: str | None) -> str:
    return "ns"


def adjust_expected(expected: DataFrame, read_ext: str, engine: str) -> None:
    expected.index.name = None
    unit = get_exp_unit(read_ext, engine)
    # error: "Index" has no attribute "as_unit"
    expected.index = expected.index.as_unit(unit)  # type: ignore[attr-defined]


def xfail_datetimes_with_pyxlsb(engine, request):
    if engine == "pyxlsb":
        request.applymarker(
            pytest.mark.xfail(
                reason="Sheets containing datetimes not supported by pyxlsb"
            )
        )


class TestReaders:
    @pytest.fixture(autouse=True)
    def cd_and_set_engine(self, engine, datapath, monkeypatch):
        """
        Change directory and set engine for read_excel calls.
        """
        func = partial(pd.read_excel, engine=engine)
        monkeypatch.chdir(datapath("io", "data", "excel"))
        monkeypatch.setattr(pd, "read_excel", func)

    def test_engine_used(self, read_ext, engine, monkeypatch):
        # GH 38884
        def parser(self, *args, **kwargs):
            return self.engine

        monkeypatch.setattr(pd.ExcelFile, "parse", parser)

        expected_defaults = {
            "xlsx": "openpyxl",
            "xlsm": "openpyxl",
            "xlsb": "pyxlsb",
            "xls": "xlrd",
            "ods": "odf",
        }

        with open("test1" + read_ext, "rb") as f:
            result = pd.read_excel(f)

        if engine is not None:
            expected = engine
        else:
            expected = expected_defaults[read_ext[1:]]
        assert result == expected

    def test_engine_kwargs(self, read_ext, engine):
        # GH#52214
        expected_defaults = {
            "xlsx": {"foo": "abcd"},
            "xlsm": {"foo": 123},
            "xlsb": {"foo": "True"},
            "xls": {"foo": True},
            "ods": {"foo": "abcd"},
        }

        if engine in {"xlrd", "pyxlsb"}:
            msg = re.escape(r"open_workbook() got an unexpected keyword argument 'foo'")
        elif engine == "odf":
            msg = re.escape(r"load() got an unexpected keyword argument 'foo'")
        else:
            msg = re.escape(r"load_workbook() got an unexpected keyword argument 'foo'")

        if engine is not None:
            with pytest.raises(TypeError, match=msg):
                pd.read_excel(
                    "test1" + read_ext,
                    sheet_name="Sheet1",
                    index_col=0,
                    engine_kwargs=expected_defaults[read_ext[1:]],
                )

    def test_usecols_int(self, read_ext):
        # usecols as int
        msg = "Passing an integer for `usecols`"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(
                "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols=3
            )

        # usecols as int
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(
                "test1" + read_ext,
                sheet_name="Sheet2",
                skiprows=[1],
                index_col=0,
                usecols=3,
            )

    def test_usecols_list(self, request, engine, read_ext, df_ref):
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = df_ref[["B", "C"]]
        adjust_expected(expected, read_ext, engine)

        df1 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols=[0, 2, 3]
        )
        df2 = pd.read_excel(
            "test1" + read_ext,
            sheet_name="Sheet2",
            skiprows=[1],
            index_col=0,
            usecols=[0, 2, 3],
        )

        # TODO add index to xls file)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

    def test_usecols_str(self, request, engine, read_ext, df_ref):
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = df_ref[["A", "B", "C"]]
        adjust_expected(expected, read_ext, engine)

        df2 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols="A:D"
        )
        df3 = pd.read_excel(
            "test1" + read_ext,
            sheet_name="Sheet2",
            skiprows=[1],
            index_col=0,
            usecols="A:D",
        )

        # TODO add index to xls, read xls ignores index name ?
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)

        expected = df_ref[["B", "C"]]
        adjust_expected(expected, read_ext, engine)

        df2 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols="A,C,D"
        )
        df3 = pd.read_excel(
            "test1" + read_ext,
            sheet_name="Sheet2",
            skiprows=[1],
            index_col=0,
            usecols="A,C,D",
        )
        # TODO add index to xls file
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)

        df2 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols="A,C:D"
        )
        df3 = pd.read_excel(
            "test1" + read_ext,
            sheet_name="Sheet2",
            skiprows=[1],
            index_col=0,
            usecols="A,C:D",
        )
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)

    @pytest.mark.parametrize(
        "usecols", [[0, 1, 3], [0, 3, 1], [1, 0, 3], [1, 3, 0], [3, 0, 1], [3, 1, 0]]
    )
    def test_usecols_diff_positional_int_columns_order(
        self, request, engine, read_ext, usecols, df_ref
    ):
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = df_ref[["A", "C"]]
        adjust_expected(expected, read_ext, engine)

        result = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols=usecols
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("usecols", [["B", "D"], ["D", "B"]])
    def test_usecols_diff_positional_str_columns_order(self, read_ext, usecols, df_ref):
        expected = df_ref[["B", "D"]]
        expected.index = range(len(expected))

        result = pd.read_excel("test1" + read_ext, sheet_name="Sheet1", usecols=usecols)
        tm.assert_frame_equal(result, expected)

    def test_read_excel_without_slicing(self, request, engine, read_ext, df_ref):
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = df_ref
        adjust_expected(expected, read_ext, engine)

        result = pd.read_excel("test1" + read_ext, sheet_name="Sheet1", index_col=0)
        tm.assert_frame_equal(result, expected)

    def test_usecols_excel_range_str(self, request, engine, read_ext, df_ref):
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = df_ref[["C", "D"]]
        adjust_expected(expected, read_ext, engine)

        result = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols="A,D:E"
        )
        tm.assert_frame_equal(result, expected)

    def test_usecols_excel_range_str_invalid(self, read_ext):
        msg = "Invalid column name: E1"

        with pytest.raises(ValueError, match=msg):
            pd.read_excel("test1" + read_ext, sheet_name="Sheet1", usecols="D:E1")

    def test_index_col_label_error(self, read_ext):
        msg = "list indices must be integers.*, not str"

        with pytest.raises(TypeError, match=msg):
            pd.read_excel(
                "test1" + read_ext,
                sheet_name="Sheet1",
                index_col=["A"],
                usecols=["A", "C"],
            )

    def test_index_col_str(self, read_ext):
        # see gh-52716
        result = pd.read_excel("test1" + read_ext, sheet_name="Sheet3", index_col="A")
        expected = DataFrame(
            columns=["B", "C", "D", "E", "F"], index=Index([], name="A")
        )
        tm.assert_frame_equal(result, expected)

    def test_index_col_empty(self, read_ext):
        # see gh-9208
        result = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet3", index_col=["A", "B", "C"]
        )
        expected = DataFrame(
            columns=["D", "E", "F"],
            index=MultiIndex(levels=[[]] * 3, codes=[[]] * 3, names=["A", "B", "C"]),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("index_col", [None, 2])
    def test_index_col_with_unnamed(self, read_ext, index_col):
        # see gh-18792
        result = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet4", index_col=index_col
        )
        expected = DataFrame(
            [["i1", "a", "x"], ["i2", "b", "y"]], columns=["Unnamed: 0", "col1", "col2"]
        )
        if index_col:
            expected = expected.set_index(expected.columns[index_col])

        tm.assert_frame_equal(result, expected)

    def test_usecols_pass_non_existent_column(self, read_ext):
        msg = (
            "Usecols do not match columns, "
            "columns expected but not found: "
            r"\['E'\]"
        )

        with pytest.raises(ValueError, match=msg):
            pd.read_excel("test1" + read_ext, usecols=["E"])

    def test_usecols_wrong_type(self, read_ext):
        msg = (
            "'usecols' must either be list-like of "
            "all strings, all unicode, all integers or a callable."
        )

        with pytest.raises(ValueError, match=msg):
            pd.read_excel("test1" + read_ext, usecols=["E1", 0])

    def test_excel_stop_iterator(self, read_ext):
        parsed = pd.read_excel("test2" + read_ext, sheet_name="Sheet1")
        expected = DataFrame([["aaaa", "bbbbb"]], columns=["Test", "Test1"])
        tm.assert_frame_equal(parsed, expected)

    def test_excel_cell_error_na(self, request, engine, read_ext):
        xfail_datetimes_with_pyxlsb(engine, request)

        # https://github.com/tafia/calamine/issues/355
        if engine == "calamine" and read_ext == ".ods":
            request.applymarker(
                pytest.mark.xfail(reason="Calamine can't extract error from ods files")
            )

        parsed = pd.read_excel("test3" + read_ext, sheet_name="Sheet1")
        expected = DataFrame([[np.nan]], columns=["Test"])
        tm.assert_frame_equal(parsed, expected)

    def test_excel_table(self, request, engine, read_ext, df_ref):
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = df_ref
        adjust_expected(expected, read_ext, engine)

        df1 = pd.read_excel("test1" + read_ext, sheet_name="Sheet1", index_col=0)
        df2 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet2", skiprows=[1], index_col=0
        )
        # TODO add index to file
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

        df3 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, skipfooter=1
        )
        tm.assert_frame_equal(df3, df1.iloc[:-1])

    def test_reader_special_dtypes(self, request, engine, read_ext):
        xfail_datetimes_with_pyxlsb(engine, request)

        unit = get_exp_unit(read_ext, engine)
        expected = DataFrame.from_dict(
            {
                "IntCol": [1, 2, -3, 4, 0],
                "FloatCol": [1.25, 2.25, 1.83, 1.92, 0.0000000005],
                "BoolCol": [True, False, True, True, False],
                "StrCol": [1, 2, 3, 4, 5],
                "Str2Col": ["a", 3, "c", "d", "e"],
                "DateCol": Index(
                    [
                        datetime(2013, 10, 30),
                        datetime(2013, 10, 31),
                        datetime(1905, 1, 1),
                        datetime(2013, 12, 14),
                        datetime(2015, 3, 14),
                    ],
                    dtype=f"M8[{unit}]",
                ),
            },
        )
        basename = "test_types"

        # should read in correctly and infer types
        actual = pd.read_excel(basename + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, expected)

        # if not coercing number, then int comes in as float
        float_expected = expected.copy()
        float_expected.loc[float_expected.index[1], "Str2Col"] = 3.0
        actual = pd.read_excel(basename + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, float_expected)

        # check setting Index (assuming xls and xlsx are the same here)
        for icol, name in enumerate(expected.columns):
            actual = pd.read_excel(
                basename + read_ext, sheet_name="Sheet1", index_col=icol
            )
            exp = expected.set_index(name)
            tm.assert_frame_equal(actual, exp)

        expected["StrCol"] = expected["StrCol"].apply(str)
        actual = pd.read_excel(
            basename + read_ext, sheet_name="Sheet1", converters={"StrCol": str}
        )
        tm.assert_frame_equal(actual, expected)

    # GH8212 - support for converters and missing values
    def test_reader_converters(self, read_ext):
        basename = "test_converters"

        expected = DataFrame.from_dict(
            {
                "IntCol": [1, 2, -3, -1000, 0],
                "FloatCol": [12.5, np.nan, 18.3, 19.2, 0.000000005],
                "BoolCol": ["Found", "Found", "Found", "Not found", "Found"],
                "StrCol": ["1", np.nan, "3", "4", "5"],
            }
        )

        converters = {
            "IntCol": lambda x: int(x) if x != "" else -1000,
            "FloatCol": lambda x: 10 * x if x else np.nan,
            2: lambda x: "Found" if x != "" else "Not found",
            3: lambda x: str(x) if x else "",
        }

        # should read in correctly and set types of single cells (not array
        # dtypes)
        actual = pd.read_excel(
            basename + read_ext, sheet_name="Sheet1", converters=converters
        )
        tm.assert_frame_equal(actual, expected)

    def test_reader_dtype(self, read_ext):
        # GH 8212
        basename = "testdtype"
        actual = pd.read_excel(basename + read_ext)

        expected = DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [2.5, 3.5, 4.5, 5.5],
                "c": [1, 2, 3, 4],
                "d": [1.0, 2.0, np.nan, 4.0],
            }
        )

        tm.assert_frame_equal(actual, expected)

        actual = pd.read_excel(
            basename + read_ext, dtype={"a": "float64", "b": "float32", "c": str}
        )

        expected["a"] = expected["a"].astype("float64")
        expected["b"] = expected["b"].astype("float32")
        expected["c"] = Series(["001", "002", "003", "004"], dtype="str")
        tm.assert_frame_equal(actual, expected)

        msg = "Unable to convert column d to type int64"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(basename + read_ext, dtype={"d": "int64"})

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (
                None,
                DataFrame(
                    {
                        "a": [1, 2, 3, 4],
                        "b": [2.5, 3.5, 4.5, 5.5],
                        "c": [1, 2, 3, 4],
                        "d": [1.0, 2.0, np.nan, 4.0],
                    }
                ),
            ),
            (
                {"a": "float64", "b": "float32", "c": str, "d": str},
                DataFrame(
                    {
                        "a": Series([1, 2, 3, 4], dtype="float64"),
                        "b": Series([2.5, 3.5, 4.5, 5.5], dtype="float32"),
                        "c": Series(["001", "002", "003", "004"], dtype="str"),
                        "d": Series(["1", "2", np.nan, "4"], dtype="str"),
                    },
                ),
            ),
        ],
    )
    def test_reader_dtype_str(self, read_ext, dtype, expected):
        # see gh-20377
        basename = "testdtype"

        actual = pd.read_excel(basename + read_ext, dtype=dtype)
        tm.assert_frame_equal(actual, expected)

    def test_dtype_backend(self, read_ext, dtype_backend, engine):
        # GH#36712
        if read_ext in (".xlsb", ".xls"):
            pytest.skip(f"No engine for filetype: '{read_ext}'")

        df = DataFrame(
            {
                "a": Series([1, 3], dtype="Int64"),
                "b": Series([2.5, 4.5], dtype="Float64"),
                "c": Series([True, False], dtype="boolean"),
                "d": Series(["a", "b"], dtype="string"),
                "e": Series([pd.NA, 6], dtype="Int64"),
                "f": Series([pd.NA, 7.5], dtype="Float64"),
                "g": Series([pd.NA, True], dtype="boolean"),
                "h": Series([pd.NA, "a"], dtype="string"),
                "i": Series([pd.Timestamp("2019-12-31")] * 2),
                "j": Series([pd.NA, pd.NA], dtype="Int64"),
            }
        )
        with tm.ensure_clean(read_ext) as file_path:
            df.to_excel(file_path, sheet_name="test", index=False)
            result = pd.read_excel(
                file_path, sheet_name="test", dtype_backend=dtype_backend
            )
        if dtype_backend == "pyarrow":
            import pyarrow as pa

            from pandas.arrays import ArrowExtensionArray

            expected = DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(df[col], from_pandas=True))
                    for col in df.columns
                }
            )
            # pyarrow by default infers timestamp resolution as us, not ns
            expected["i"] = ArrowExtensionArray(
                expected["i"].array._pa_array.cast(pa.timestamp(unit="us"))
            )
            # pyarrow supports a null type, so don't have to default to Int64
            expected["j"] = ArrowExtensionArray(pa.array([None, None]))
        else:
            expected = df
            unit = get_exp_unit(read_ext, engine)
            expected["i"] = expected["i"].astype(f"M8[{unit}]")

        tm.assert_frame_equal(result, expected)

    def test_dtype_backend_and_dtype(self, read_ext):
        # GH#36712
        if read_ext in (".xlsb", ".xls"):
            pytest.skip(f"No engine for filetype: '{read_ext}'")

        df = DataFrame({"a": [np.nan, 1.0], "b": [2.5, np.nan]})
        with tm.ensure_clean(read_ext) as file_path:
            df.to_excel(file_path, sheet_name="test", index=False)
            result = pd.read_excel(
                file_path,
                sheet_name="test",
                dtype_backend="numpy_nullable",
                dtype="float64",
            )
        tm.assert_frame_equal(result, df)

    def test_dtype_backend_string(self, read_ext, string_storage):
        # GH#36712
        if read_ext in (".xlsb", ".xls"):
            pytest.skip(f"No engine for filetype: '{read_ext}'")

        with pd.option_context("mode.string_storage", string_storage):
            df = DataFrame(
                {
                    "a": np.array(["a", "b"], dtype=np.object_),
                    "b": np.array(["x", pd.NA], dtype=np.object_),
                }
            )

            with tm.ensure_clean(read_ext) as file_path:
                df.to_excel(file_path, sheet_name="test", index=False)
                result = pd.read_excel(
                    file_path, sheet_name="test", dtype_backend="numpy_nullable"
                )

            expected = DataFrame(
                {
                    "a": Series(["a", "b"], dtype=pd.StringDtype(string_storage)),
                    "b": Series(["x", None], dtype=pd.StringDtype(string_storage)),
                }
            )
            # the storage of the str columns' Index is also affected by the
            # string_storage setting -> ignore that for checking the result
            tm.assert_frame_equal(result, expected, check_column_type=False)

    @pytest.mark.parametrize("dtypes, exp_value", [({}, 1), ({"a.1": "int64"}, 1)])
    def test_dtype_mangle_dup_cols(self, read_ext, dtypes, exp_value):
        # GH#35211
        basename = "df_mangle_dup_col_dtypes"
        dtype_dict = {"a": object, **dtypes}
        dtype_dict_copy = dtype_dict.copy()
        # GH#42462
        result = pd.read_excel(basename + read_ext, dtype=dtype_dict)
        expected = DataFrame(
            {
                "a": Series([1], dtype=object),
                "a.1": Series([exp_value], dtype=object if not dtypes else None),
            }
        )
        assert dtype_dict == dtype_dict_copy, "dtype dict changed"
        tm.assert_frame_equal(result, expected)

    def test_reader_spaces(self, read_ext):
        # see gh-32207
        basename = "test_spaces"

        actual = pd.read_excel(basename + read_ext)
        expected = DataFrame(
            {
                "testcol": [
                    "this is great",
                    "4    spaces",
                    "1 trailing ",
                    " 1 leading",
                    "2  spaces  multiple  times",
                ]
            }
        )
        tm.assert_frame_equal(actual, expected)

    # gh-36122, gh-35802
    @pytest.mark.parametrize(
        "basename,expected",
        [
            ("gh-35802", DataFrame({"COLUMN": ["Test (1)"]})),
            ("gh-36122", DataFrame(columns=["got 2nd sa"])),
        ],
    )
    def test_read_excel_ods_nested_xml(self, engine, read_ext, basename, expected):
        # see gh-35802
        if engine != "odf":
            pytest.skip(f"Skipped for engine: {engine}")

        actual = pd.read_excel(basename + read_ext)
        tm.assert_frame_equal(actual, expected)

    def test_reading_all_sheets(self, read_ext):
        # Test reading all sheet names by setting sheet_name to None,
        # Ensure a dict is returned.
        # See PR #9450
        basename = "test_multisheet"
        dfs = pd.read_excel(basename + read_ext, sheet_name=None)
        # ensure this is not alphabetical to test order preservation
        expected_keys = ["Charlie", "Alpha", "Beta"]
        tm.assert_contains_all(expected_keys, dfs.keys())
        # Issue 9930
        # Ensure sheet order is preserved
        assert expected_keys == list(dfs.keys())

    def test_reading_multiple_specific_sheets(self, read_ext):
        # Test reading specific sheet names by specifying a mixed list
        # of integers and strings, and confirm that duplicated sheet
        # references (positions/names) are removed properly.
        # Ensure a dict is returned
        # See PR #9450
        basename = "test_multisheet"
        # Explicitly request duplicates. Only the set should be returned.
        expected_keys = [2, "Charlie", "Charlie"]
        dfs = pd.read_excel(basename + read_ext, sheet_name=expected_keys)
        expected_keys = list(set(expected_keys))
        tm.assert_contains_all(expected_keys, dfs.keys())
        assert len(expected_keys) == len(dfs.keys())

    def test_reading_all_sheets_with_blank(self, read_ext):
        # Test reading all sheet names by setting sheet_name to None,
        # In the case where some sheets are blank.
        # Issue #11711
        basename = "blank_with_header"
        dfs = pd.read_excel(basename + read_ext, sheet_name=None)
        expected_keys = ["Sheet1", "Sheet2", "Sheet3"]
        tm.assert_contains_all(expected_keys, dfs.keys())

    # GH6403
    def test_read_excel_blank(self, read_ext):
        actual = pd.read_excel("blank" + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, DataFrame())

    def test_read_excel_blank_with_header(self, read_ext):
        expected = DataFrame(columns=["col_1", "col_2"])
        actual = pd.read_excel("blank_with_header" + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, expected)

    def test_exception_message_includes_sheet_name(self, read_ext):
        # GH 48706
        with pytest.raises(ValueError, match=r" \(sheet: Sheet1\)$"):
            pd.read_excel("blank_with_header" + read_ext, header=[1], sheet_name=None)
        with pytest.raises(ZeroDivisionError, match=r" \(sheet: Sheet1\)$"):
            pd.read_excel("test1" + read_ext, usecols=lambda x: 1 / 0, sheet_name=None)

    @pytest.mark.filterwarnings("ignore:Cell A4 is marked:UserWarning:openpyxl")
    def test_date_conversion_overflow(self, request, engine, read_ext):
        # GH 10001 : pandas.ExcelFile ignore parse_dates=False
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = DataFrame(
            [
                [pd.Timestamp("2016-03-12"), "Marc Johnson"],
                [pd.Timestamp("2016-03-16"), "Jack Black"],
                [1e20, "Timothy Brown"],
            ],
            columns=["DateColWithBigInt", "StringCol"],
        )

        if engine == "openpyxl":
            request.applymarker(
                pytest.mark.xfail(reason="Maybe not supported by openpyxl")
            )

        if engine is None and read_ext in (".xlsx", ".xlsm"):
            # GH 35029
            request.applymarker(
                pytest.mark.xfail(reason="Defaults to openpyxl, maybe not supported")
            )

        result = pd.read_excel("testdateoverflow" + read_ext)
        tm.assert_frame_equal(result, expected)

    def test_sheet_name(self, request, read_ext, engine, df_ref):
        xfail_datetimes_with_pyxlsb(engine, request)

        filename = "test1"
        sheet_name = "Sheet1"

        expected = df_ref
        adjust_expected(expected, read_ext, engine)

        df1 = pd.read_excel(
            filename + read_ext, sheet_name=sheet_name, index_col=0
        )  # doc
        df2 = pd.read_excel(filename + read_ext, index_col=0, sheet_name=sheet_name)

        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

    def test_excel_read_buffer(self, read_ext):
        pth = "test1" + read_ext
        expected = pd.read_excel(pth, sheet_name="Sheet1", index_col=0)
        with open(pth, "rb") as f:
            actual = pd.read_excel(f, sheet_name="Sheet1", index_col=0)
            tm.assert_frame_equal(expected, actual)

    def test_bad_engine_raises(self):
        bad_engine = "foo"
        with pytest.raises(ValueError, match="Unknown engine: foo"):
            pd.read_excel("", engine=bad_engine)

    @pytest.mark.parametrize(
        "sheet_name",
        [3, [0, 3], [3, 0], "Sheet4", ["Sheet1", "Sheet4"], ["Sheet4", "Sheet1"]],
    )
    def test_bad_sheetname_raises(self, read_ext, sheet_name):
        # GH 39250
        msg = "Worksheet index 3 is invalid|Worksheet named 'Sheet4' not found"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel("blank" + read_ext, sheet_name=sheet_name)

    def test_missing_file_raises(self, read_ext):
        bad_file = f"foo{read_ext}"
        # CI tests with other languages, translates to "No such file or directory"
        match = "|".join(
            [
                "(No such file or directory",
                "没有那个文件或目录",
                "File o directory non esistente)",
            ]
        )
        with pytest.raises(FileNotFoundError, match=match):
            pd.read_excel(bad_file)

    def test_corrupt_bytes_raises(self, engine):
        bad_stream = b"foo"
        if engine is None:
            error = ValueError
            msg = (
                "Excel file format cannot be determined, you must "
                "specify an engine manually."
            )
        elif engine == "xlrd":
            from xlrd import XLRDError

            error = XLRDError
            msg = (
                "Unsupported format, or corrupt file: Expected BOF "
                "record; found b'foo'"
            )
        elif engine == "calamine":
            from python_calamine import CalamineError

            error = CalamineError
            msg = "Cannot detect file format"
        else:
            error = BadZipFile
            msg = "File is not a zip file"
        with pytest.raises(error, match=msg):
            pd.read_excel(BytesIO(bad_stream))

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_read_from_http_url(self, httpserver, read_ext):
        with open("test1" + read_ext, "rb") as f:
            httpserver.serve_content(content=f.read())
        url_table = pd.read_excel(httpserver.url)
        local_table = pd.read_excel("test1" + read_ext)
        tm.assert_frame_equal(url_table, local_table)

    @td.skip_if_not_us_locale
    @pytest.mark.single_cpu
    def test_read_from_s3_url(self, read_ext, s3_public_bucket, s3so):
        # Bucket created in tests/io/conftest.py
        with open("test1" + read_ext, "rb") as f:
            s3_public_bucket.put_object(Key="test1" + read_ext, Body=f)

        url = f"s3://{s3_public_bucket.name}/test1" + read_ext

        url_table = pd.read_excel(url, storage_options=s3so)
        local_table = pd.read_excel("test1" + read_ext)
        tm.assert_frame_equal(url_table, local_table)

    @pytest.mark.single_cpu
    def test_read_from_s3_object(self, read_ext, s3_public_bucket, s3so):
        # GH 38788
        # Bucket created in tests/io/conftest.py
        with open("test1" + read_ext, "rb") as f:
            s3_public_bucket.put_object(Key="test1" + read_ext, Body=f)

        import s3fs

        s3 = s3fs.S3FileSystem(**s3so)

        with s3.open(f"s3://{s3_public_bucket.name}/test1" + read_ext) as f:
            url_table = pd.read_excel(f)

        local_table = pd.read_excel("test1" + read_ext)
        tm.assert_frame_equal(url_table, local_table)

    @pytest.mark.slow
    def test_read_from_file_url(self, read_ext, datapath):
        # FILE
        localtable = os.path.join(datapath("io", "data", "excel"), "test1" + read_ext)
        local_table = pd.read_excel(localtable)

        try:
            url_table = pd.read_excel("file://localhost/" + localtable)
        except URLError:
            # fails on some systems
            platform_info = " ".join(platform.uname()).strip()
            pytest.skip(f"failing on {platform_info}")

        tm.assert_frame_equal(url_table, local_table)

    def test_read_from_pathlib_path(self, read_ext):
        # GH12655
        str_path = "test1" + read_ext
        expected = pd.read_excel(str_path, sheet_name="Sheet1", index_col=0)

        path_obj = Path("test1" + read_ext)
        actual = pd.read_excel(path_obj, sheet_name="Sheet1", index_col=0)

        tm.assert_frame_equal(expected, actual)

    @td.skip_if_no("py.path")
    def test_read_from_py_localpath(self, read_ext):
        # GH12655
        from py.path import local as LocalPath

        str_path = os.path.join("test1" + read_ext)
        expected = pd.read_excel(str_path, sheet_name="Sheet1", index_col=0)

        path_obj = LocalPath().join("test1" + read_ext)
        actual = pd.read_excel(path_obj, sheet_name="Sheet1", index_col=0)

        tm.assert_frame_equal(expected, actual)

    def test_close_from_py_localpath(self, read_ext):
        # GH31467
        str_path = os.path.join("test1" + read_ext)
        with open(str_path, "rb") as f:
            x = pd.read_excel(f, sheet_name="Sheet1", index_col=0)
            del x
            # should not throw an exception because the passed file was closed
            f.read()

    def test_reader_seconds(self, request, engine, read_ext):
        xfail_datetimes_with_pyxlsb(engine, request)

        # GH 55045
        if engine == "calamine" and read_ext == ".ods":
            request.applymarker(
                pytest.mark.xfail(
                    reason="ODS file contains bad datetime (seconds as text)"
                )
            )

        # Test reading times with and without milliseconds. GH5945.
        expected = DataFrame.from_dict(
            {
                "Time": [
                    time(1, 2, 3),
                    time(2, 45, 56, 100000),
                    time(4, 29, 49, 200000),
                    time(6, 13, 42, 300000),
                    time(7, 57, 35, 400000),
                    time(9, 41, 28, 500000),
                    time(11, 25, 21, 600000),
                    time(13, 9, 14, 700000),
                    time(14, 53, 7, 800000),
                    time(16, 37, 0, 900000),
                    time(18, 20, 54),
                ]
            }
        )

        actual = pd.read_excel("times_1900" + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, expected)

        actual = pd.read_excel("times_1904" + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_multiindex(self, request, engine, read_ext):
        # see gh-4679
        xfail_datetimes_with_pyxlsb(engine, request)

        unit = get_exp_unit(read_ext, engine)

        mi = MultiIndex.from_product([["foo", "bar"], ["a", "b"]])
        mi_file = "testmultiindex" + read_ext

        # "mi_column" sheet
        expected = DataFrame(
            [
                [1, 2.5, pd.Timestamp("2015-01-01"), True],
                [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=mi,
        )
        expected[mi[2]] = expected[mi[2]].astype(f"M8[{unit}]")

        actual = pd.read_excel(
            mi_file, sheet_name="mi_column", header=[0, 1], index_col=0
        )
        tm.assert_frame_equal(actual, expected)

        # "mi_index" sheet
        expected.index = mi
        expected.columns = ["a", "b", "c", "d"]

        actual = pd.read_excel(mi_file, sheet_name="mi_index", index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)

        # "both" sheet
        expected.columns = mi

        actual = pd.read_excel(
            mi_file, sheet_name="both", index_col=[0, 1], header=[0, 1]
        )
        tm.assert_frame_equal(actual, expected)

        # "mi_index_name" sheet
        expected.columns = ["a", "b", "c", "d"]
        expected.index = mi.set_names(["ilvl1", "ilvl2"])

        actual = pd.read_excel(mi_file, sheet_name="mi_index_name", index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)

        # "mi_column_name" sheet
        expected.index = list(range(4))
        expected.columns = mi.set_names(["c1", "c2"])
        actual = pd.read_excel(
            mi_file, sheet_name="mi_column_name", header=[0, 1], index_col=0
        )
        tm.assert_frame_equal(actual, expected)

        # see gh-11317
        # "name_with_int" sheet
        expected.columns = mi.set_levels([1, 2], level=1).set_names(["c1", "c2"])

        actual = pd.read_excel(
            mi_file, sheet_name="name_with_int", index_col=0, header=[0, 1]
        )
        tm.assert_frame_equal(actual, expected)

        # "both_name" sheet
        expected.columns = mi.set_names(["c1", "c2"])
        expected.index = mi.set_names(["ilvl1", "ilvl2"])

        actual = pd.read_excel(
            mi_file, sheet_name="both_name", index_col=[0, 1], header=[0, 1]
        )
        tm.assert_frame_equal(actual, expected)

        # "both_skiprows" sheet
        actual = pd.read_excel(
            mi_file,
            sheet_name="both_name_skiprows",
            index_col=[0, 1],
            header=[0, 1],
            skiprows=2,
        )
        tm.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "sheet_name,idx_lvl2",
        [
            ("both_name_blank_after_mi_name", [np.nan, "b", "a", "b"]),
            ("both_name_multiple_blanks", [np.nan] * 4),
        ],
    )
    def test_read_excel_multiindex_blank_after_name(
        self, request, engine, read_ext, sheet_name, idx_lvl2
    ):
        # GH34673
        xfail_datetimes_with_pyxlsb(engine, request)

        mi_file = "testmultiindex" + read_ext
        mi = MultiIndex.from_product([["foo", "bar"], ["a", "b"]], names=["c1", "c2"])

        unit = get_exp_unit(read_ext, engine)

        expected = DataFrame(
            [
                [1, 2.5, pd.Timestamp("2015-01-01"), True],
                [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=mi,
            index=MultiIndex.from_arrays(
                (["foo", "foo", "bar", "bar"], idx_lvl2),
                names=["ilvl1", "ilvl2"],
            ),
        )
        expected[mi[2]] = expected[mi[2]].astype(f"M8[{unit}]")
        result = pd.read_excel(
            mi_file,
            sheet_name=sheet_name,
            index_col=[0, 1],
            header=[0, 1],
        )
        tm.assert_frame_equal(result, expected)

    def test_read_excel_multiindex_header_only(self, read_ext):
        # see gh-11733.
        #
        # Don't try to parse a header name if there isn't one.
        mi_file = "testmultiindex" + read_ext
        result = pd.read_excel(mi_file, sheet_name="index_col_none", header=[0, 1])

        exp_columns = MultiIndex.from_product([("A", "B"), ("key", "val")])
        expected = DataFrame([[1, 2, 3, 4]] * 2, columns=exp_columns)
        tm.assert_frame_equal(result, expected)

    def test_excel_old_index_format(self, read_ext):
        # see gh-4679
        filename = "test_index_name_pre17" + read_ext

        # We detect headers to determine if index names exist, so
        # that "index" name in the "names" version of the data will
        # now be interpreted as rows that include null data.
        data = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                ["R0C0", "R0C1", "R0C2", "R0C3", "R0C4"],
                ["R1C0", "R1C1", "R1C2", "R1C3", "R1C4"],
                ["R2C0", "R2C1", "R2C2", "R2C3", "R2C4"],
                ["R3C0", "R3C1", "R3C2", "R3C3", "R3C4"],
                ["R4C0", "R4C1", "R4C2", "R4C3", "R4C4"],
            ],
            dtype=object,
        )
        columns = ["C_l0_g0", "C_l0_g1", "C_l0_g2", "C_l0_g3", "C_l0_g4"]
        mi = MultiIndex(
            levels=[
                ["R0", "R_l0_g0", "R_l0_g1", "R_l0_g2", "R_l0_g3", "R_l0_g4"],
                ["R1", "R_l1_g0", "R_l1_g1", "R_l1_g2", "R_l1_g3", "R_l1_g4"],
            ],
            codes=[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
            names=[None, None],
        )
        si = Index(
            ["R0", "R_l0_g0", "R_l0_g1", "R_l0_g2", "R_l0_g3", "R_l0_g4"], name=None
        )

        expected = DataFrame(data, index=si, columns=columns)

        actual = pd.read_excel(filename, sheet_name="single_names", index_col=0)
        tm.assert_frame_equal(actual, expected)

        expected.index = mi

        actual = pd.read_excel(filename, sheet_name="multi_names", index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)

        # The analogous versions of the "names" version data
        # where there are explicitly no names for the indices.
        data = np.array(
            [
                ["R0C0", "R0C1", "R0C2", "R0C3", "R0C4"],
                ["R1C0", "R1C1", "R1C2", "R1C3", "R1C4"],
                ["R2C0", "R2C1", "R2C2", "R2C3", "R2C4"],
                ["R3C0", "R3C1", "R3C2", "R3C3", "R3C4"],
                ["R4C0", "R4C1", "R4C2", "R4C3", "R4C4"],
            ]
        )
        columns = ["C_l0_g0", "C_l0_g1", "C_l0_g2", "C_l0_g3", "C_l0_g4"]
        mi = MultiIndex(
            levels=[
                ["R_l0_g0", "R_l0_g1", "R_l0_g2", "R_l0_g3", "R_l0_g4"],
                ["R_l1_g0", "R_l1_g1", "R_l1_g2", "R_l1_g3", "R_l1_g4"],
            ],
            codes=[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
            names=[None, None],
        )
        si = Index(["R_l0_g0", "R_l0_g1", "R_l0_g2", "R_l0_g3", "R_l0_g4"], name=None)

        expected = DataFrame(data, index=si, columns=columns)

        actual = pd.read_excel(filename, sheet_name="single_no_names", index_col=0)
        tm.assert_frame_equal(actual, expected)

        expected.index = mi

        actual = pd.read_excel(filename, sheet_name="multi_no_names", index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_bool_header_arg(self, read_ext):
        # GH 6114
        msg = "Passing a bool to header is invalid"
        for arg in [True, False]:
            with pytest.raises(TypeError, match=msg):
                pd.read_excel("test1" + read_ext, header=arg)

    def test_read_excel_skiprows(self, request, engine, read_ext):
        # GH 4903
        xfail_datetimes_with_pyxlsb(engine, request)

        unit = get_exp_unit(read_ext, engine)

        actual = pd.read_excel(
            "testskiprows" + read_ext, sheet_name="skiprows_list", skiprows=[0, 2]
        )
        expected = DataFrame(
            [
                [1, 2.5, pd.Timestamp("2015-01-01"), True],
                [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=["a", "b", "c", "d"],
        )
        expected["c"] = expected["c"].astype(f"M8[{unit}]")
        tm.assert_frame_equal(actual, expected)

        actual = pd.read_excel(
            "testskiprows" + read_ext,
            sheet_name="skiprows_list",
            skiprows=np.array([0, 2]),
        )
        tm.assert_frame_equal(actual, expected)

        # GH36435
        actual = pd.read_excel(
            "testskiprows" + read_ext,
            sheet_name="skiprows_list",
            skiprows=lambda x: x in [0, 2],
        )
        tm.assert_frame_equal(actual, expected)

        actual = pd.read_excel(
            "testskiprows" + read_ext,
            sheet_name="skiprows_list",
            skiprows=3,
            names=["a", "b", "c", "d"],
        )
        expected = DataFrame(
            [
                # [1, 2.5, pd.Timestamp("2015-01-01"), True],
                [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=["a", "b", "c", "d"],
        )
        expected["c"] = expected["c"].astype(f"M8[{unit}]")
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_skiprows_callable_not_in(self, request, engine, read_ext):
        # GH 4903
        xfail_datetimes_with_pyxlsb(engine, request)
        unit = get_exp_unit(read_ext, engine)

        actual = pd.read_excel(
            "testskiprows" + read_ext,
            sheet_name="skiprows_list",
            skiprows=lambda x: x not in [1, 3, 5],
        )
        expected = DataFrame(
            [
                [1, 2.5, pd.Timestamp("2015-01-01"), True],
                # [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                # [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=["a", "b", "c", "d"],
        )
        expected["c"] = expected["c"].astype(f"M8[{unit}]")
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_nrows(self, read_ext):
        # GH 16645
        num_rows_to_pull = 5
        actual = pd.read_excel("test1" + read_ext, nrows=num_rows_to_pull)
        expected = pd.read_excel("test1" + read_ext)
        expected = expected[:num_rows_to_pull]
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_nrows_greater_than_nrows_in_file(self, read_ext):
        # GH 16645
        expected = pd.read_excel("test1" + read_ext)
        num_records_in_file = len(expected)
        num_rows_to_pull = num_records_in_file + 10
        actual = pd.read_excel("test1" + read_ext, nrows=num_rows_to_pull)
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_nrows_non_integer_parameter(self, read_ext):
        # GH 16645
        msg = "'nrows' must be an integer >=0"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel("test1" + read_ext, nrows="5")

    @pytest.mark.parametrize(
        "filename,sheet_name,header,index_col,skiprows",
        [
            ("testmultiindex", "mi_column", [0, 1], 0, None),
            ("testmultiindex", "mi_index", None, [0, 1], None),
            ("testmultiindex", "both", [0, 1], [0, 1], None),
            ("testmultiindex", "mi_column_name", [0, 1], 0, None),
            ("testskiprows", "skiprows_list", None, None, [0, 2]),
            ("testskiprows", "skiprows_list", None, None, lambda x: x in (0, 2)),
        ],
    )
    def test_read_excel_nrows_params(
        self, read_ext, filename, sheet_name, header, index_col, skiprows
    ):
        """
        For various parameters, we should get the same result whether we
        limit the rows during load (nrows=3) or after (df.iloc[:3]).
        """
        # GH 46894
        expected = pd.read_excel(
            filename + read_ext,
            sheet_name=sheet_name,
            header=header,
            index_col=index_col,
            skiprows=skiprows,
        ).iloc[:3]
        actual = pd.read_excel(
            filename + read_ext,
            sheet_name=sheet_name,
            header=header,
            index_col=index_col,
            skiprows=skiprows,
            nrows=3,
        )
        tm.assert_frame_equal(actual, expected)

    def test_deprecated_kwargs(self, read_ext):
        with pytest.raises(TypeError, match="but 3 positional arguments"):
            pd.read_excel("test1" + read_ext, "Sheet1", 0)

    def test_no_header_with_list_index_col(self, read_ext):
        # GH 31783
        file_name = "testmultiindex" + read_ext
        data = [("B", "B"), ("key", "val"), (3, 4), (3, 4)]
        idx = MultiIndex.from_tuples(
            [("A", "A"), ("key", "val"), (1, 2), (1, 2)], names=(0, 1)
        )
        expected = DataFrame(data, index=idx, columns=(2, 3))
        result = pd.read_excel(
            file_name, sheet_name="index_col_none", index_col=[0, 1], header=None
        )
        tm.assert_frame_equal(expected, result)

    def test_one_col_noskip_blank_line(self, read_ext):
        # GH 39808
        file_name = "one_col_blank_line" + read_ext
        data = [0.5, np.nan, 1, 2]
        expected = DataFrame(data, columns=["numbers"])
        result = pd.read_excel(file_name)
        tm.assert_frame_equal(result, expected)

    def test_multiheader_two_blank_lines(self, read_ext):
        # GH 40442
        file_name = "testmultiindex" + read_ext
        columns = MultiIndex.from_tuples([("a", "A"), ("b", "B")])
        data = [[np.nan, np.nan], [np.nan, np.nan], [1, 3], [2, 4]]
        expected = DataFrame(data, columns=columns)
        result = pd.read_excel(
            file_name, sheet_name="mi_column_empty_rows", header=[0, 1]
        )
        tm.assert_frame_equal(result, expected)

    def test_trailing_blanks(self, read_ext):
        """
        Sheets can contain blank cells with no data. Some of our readers
        were including those cells, creating many empty rows and columns
        """
        file_name = "trailing_blanks" + read_ext
        result = pd.read_excel(file_name)
        assert result.shape == (3, 3)

    def test_ignore_chartsheets_by_str(self, request, engine, read_ext):
        # GH 41448
        if read_ext == ".ods":
            pytest.skip("chartsheets do not exist in the ODF format")
        if engine == "pyxlsb":
            request.applymarker(
                pytest.mark.xfail(
                    reason="pyxlsb can't distinguish chartsheets from worksheets"
                )
            )
        with pytest.raises(ValueError, match="Worksheet named 'Chart1' not found"):
            pd.read_excel("chartsheet" + read_ext, sheet_name="Chart1")

    def test_ignore_chartsheets_by_int(self, request, engine, read_ext):
        # GH 41448
        if read_ext == ".ods":
            pytest.skip("chartsheets do not exist in the ODF format")
        if engine == "pyxlsb":
            request.applymarker(
                pytest.mark.xfail(
                    reason="pyxlsb can't distinguish chartsheets from worksheets"
                )
            )
        with pytest.raises(
            ValueError, match="Worksheet index 1 is invalid, 1 worksheets found"
        ):
            pd.read_excel("chartsheet" + read_ext, sheet_name=1)

    def test_euro_decimal_format(self, read_ext):
        # copied from read_csv
        result = pd.read_excel("test_decimal" + read_ext, decimal=",", skiprows=1)
        expected = DataFrame(
            [
                [1, 1521.1541, 187101.9543, "ABC", "poi", 4.738797819],
                [2, 121.12, 14897.76, "DEF", "uyt", 0.377320872],
                [3, 878.158, 108013.434, "GHI", "rez", 2.735694704],
            ],
            columns=["Id", "Number1", "Number2", "Text1", "Text2", "Number3"],
        )
        tm.assert_frame_equal(result, expected)


class TestExcelFileRead:
    def test_deprecate_bytes_input(self, engine, read_ext):
        # GH 53830
        msg = (
            "Passing bytes to 'read_excel' is deprecated and "
            "will be removed in a future version. To read from a "
            "byte string, wrap it in a `BytesIO` object."
        )

        with tm.assert_produces_warning(
            FutureWarning, match=msg, raise_on_extra_warnings=False
        ):
            with open("test1" + read_ext, "rb") as f:
                pd.read_excel(f.read(), engine=engine)

    @pytest.fixture(autouse=True)
    def cd_and_set_engine(self, engine, datapath, monkeypatch):
        """
        Change directory and set engine for ExcelFile objects.
        """
        func = partial(pd.ExcelFile, engine=engine)
        monkeypatch.chdir(datapath("io", "data", "excel"))
        monkeypatch.setattr(pd, "ExcelFile", func)

    def test_engine_used(self, read_ext, engine):
        expected_defaults = {
            "xlsx": "openpyxl",
            "xlsm": "openpyxl",
            "xlsb": "pyxlsb",
            "xls": "xlrd",
            "ods": "odf",
        }

        with pd.ExcelFile("test1" + read_ext) as excel:
            result = excel.engine

        if engine is not None:
            expected = engine
        else:
            expected = expected_defaults[read_ext[1:]]
        assert result == expected

    def test_excel_passes_na(self, read_ext):
        with pd.ExcelFile("test4" + read_ext) as excel:
            parsed = pd.read_excel(
                excel, sheet_name="Sheet1", keep_default_na=False, na_values=["apple"]
            )
        expected = DataFrame(
            [["NA"], [1], ["NA"], [np.nan], ["rabbit"]], columns=["Test"]
        )
        tm.assert_frame_equal(parsed, expected)

        with pd.ExcelFile("test4" + read_ext) as excel:
            parsed = pd.read_excel(
                excel, sheet_name="Sheet1", keep_default_na=True, na_values=["apple"]
            )
        expected = DataFrame(
            [[np.nan], [1], [np.nan], [np.nan], ["rabbit"]], columns=["Test"]
        )
        tm.assert_frame_equal(parsed, expected)

        # 13967
        with pd.ExcelFile("test5" + read_ext) as excel:
            parsed = pd.read_excel(
                excel, sheet_name="Sheet1", keep_default_na=False, na_values=["apple"]
            )
        expected = DataFrame(
            [["1.#QNAN"], [1], ["nan"], [np.nan], ["rabbit"]], columns=["Test"]
        )
        tm.assert_frame_equal(parsed, expected)

        with pd.ExcelFile("test5" + read_ext) as excel:
            parsed = pd.read_excel(
                excel, sheet_name="Sheet1", keep_default_na=True, na_values=["apple"]
            )
        expected = DataFrame(
            [[np.nan], [1], [np.nan], [np.nan], ["rabbit"]], columns=["Test"]
        )
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize("na_filter", [None, True, False])
    def test_excel_passes_na_filter(self, read_ext, na_filter):
        # gh-25453
        kwargs = {}

        if na_filter is not None:
            kwargs["na_filter"] = na_filter

        with pd.ExcelFile("test5" + read_ext) as excel:
            parsed = pd.read_excel(
                excel,
                sheet_name="Sheet1",
                keep_default_na=True,
                na_values=["apple"],
                **kwargs,
            )

        if na_filter is False:
            expected = [["1.#QNAN"], [1], ["nan"], ["apple"], ["rabbit"]]
        else:
            expected = [[np.nan], [1], [np.nan], [np.nan], ["rabbit"]]

        expected = DataFrame(expected, columns=["Test"])
        tm.assert_frame_equal(parsed, expected)

    def test_excel_table_sheet_by_index(self, request, engine, read_ext, df_ref):
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = df_ref
        adjust_expected(expected, read_ext, engine)

        with pd.ExcelFile("test1" + read_ext) as excel:
            df1 = pd.read_excel(excel, sheet_name=0, index_col=0)
            df2 = pd.read_excel(excel, sheet_name=1, skiprows=[1], index_col=0)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

        with pd.ExcelFile("test1" + read_ext) as excel:
            df1 = excel.parse(0, index_col=0)
            df2 = excel.parse(1, skiprows=[1], index_col=0)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

        with pd.ExcelFile("test1" + read_ext) as excel:
            df3 = pd.read_excel(excel, sheet_name=0, index_col=0, skipfooter=1)
        tm.assert_frame_equal(df3, df1.iloc[:-1])

        with pd.ExcelFile("test1" + read_ext) as excel:
            df3 = excel.parse(0, index_col=0, skipfooter=1)

        tm.assert_frame_equal(df3, df1.iloc[:-1])

    def test_sheet_name(self, request, engine, read_ext, df_ref):
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = df_ref
        adjust_expected(expected, read_ext, engine)

        filename = "test1"
        sheet_name = "Sheet1"

        with pd.ExcelFile(filename + read_ext) as excel:
            df1_parse = excel.parse(sheet_name=sheet_name, index_col=0)  # doc

        with pd.ExcelFile(filename + read_ext) as excel:
            df2_parse = excel.parse(index_col=0, sheet_name=sheet_name)

        tm.assert_frame_equal(df1_parse, expected)
        tm.assert_frame_equal(df2_parse, expected)

    @pytest.mark.parametrize(
        "sheet_name",
        [3, [0, 3], [3, 0], "Sheet4", ["Sheet1", "Sheet4"], ["Sheet4", "Sheet1"]],
    )
    def test_bad_sheetname_raises(self, read_ext, sheet_name):
        # GH 39250
        msg = "Worksheet index 3 is invalid|Worksheet named 'Sheet4' not found"
        with pytest.raises(ValueError, match=msg):
            with pd.ExcelFile("blank" + read_ext) as excel:
                excel.parse(sheet_name=sheet_name)

    def test_excel_read_buffer(self, engine, read_ext):
        pth = "test1" + read_ext
        expected = pd.read_excel(pth, sheet_name="Sheet1", index_col=0, engine=engine)

        with open(pth, "rb") as f:
            with pd.ExcelFile(f) as xls:
                actual = pd.read_excel(xls, sheet_name="Sheet1", index_col=0)

        tm.assert_frame_equal(expected, actual)

    def test_reader_closes_file(self, engine, read_ext):
        with open("test1" + read_ext, "rb") as f:
            with pd.ExcelFile(f) as xlsx:
                # parses okay
                pd.read_excel(xlsx, sheet_name="Sheet1", index_col=0, engine=engine)

        assert f.closed

    def test_conflicting_excel_engines(self, read_ext):
        # GH 26566
        msg = "Engine should not be specified when passing an ExcelFile"

        with pd.ExcelFile("test1" + read_ext) as xl:
            with pytest.raises(ValueError, match=msg):
                pd.read_excel(xl, engine="foo")

    def test_excel_read_binary(self, engine, read_ext):
        # GH 15914
        expected = pd.read_excel("test1" + read_ext, engine=engine)

        with open("test1" + read_ext, "rb") as f:
            data = f.read()

        actual = pd.read_excel(BytesIO(data), engine=engine)
        tm.assert_frame_equal(expected, actual)

    def test_excel_read_binary_via_read_excel(self, read_ext, engine):
        # GH 38424
        with open("test1" + read_ext, "rb") as f:
            result = pd.read_excel(f, engine=engine)
        expected = pd.read_excel("test1" + read_ext, engine=engine)
        tm.assert_frame_equal(result, expected)

    def test_read_excel_header_index_out_of_range(self, engine):
        # GH#43143
        with open("df_header_oob.xlsx", "rb") as f:
            with pytest.raises(ValueError, match="exceeds maximum"):
                pd.read_excel(f, header=[0, 1])

    @pytest.mark.parametrize("filename", ["df_empty.xlsx", "df_equals.xlsx"])
    def test_header_with_index_col(self, filename):
        # GH 33476
        idx = Index(["Z"], name="I2")
        cols = MultiIndex.from_tuples([("A", "B"), ("A", "B.1")], names=["I11", "I12"])
        expected = DataFrame([[1, 3]], index=idx, columns=cols, dtype="int64")
        result = pd.read_excel(
            filename, sheet_name="Sheet1", index_col=0, header=[0, 1]
        )
        tm.assert_frame_equal(expected, result)

    def test_read_datetime_multiindex(self, request, engine, read_ext):
        # GH 34748
        xfail_datetimes_with_pyxlsb(engine, request)

        f = "test_datetime_mi" + read_ext
        with pd.ExcelFile(f) as excel:
            actual = pd.read_excel(excel, header=[0, 1], index_col=0, engine=engine)

        unit = get_exp_unit(read_ext, engine)
        dti = pd.DatetimeIndex(["2020-02-29", "2020-03-01"], dtype=f"M8[{unit}]")
        expected_column_index = MultiIndex.from_arrays(
            [dti[:1], dti[1:]],
            names=[
                dti[0].to_pydatetime(),
                dti[1].to_pydatetime(),
            ],
        )
        expected = DataFrame([], index=[], columns=expected_column_index)

        tm.assert_frame_equal(expected, actual)

    def test_engine_invalid_option(self, read_ext):
        # read_ext includes the '.' hence the weird formatting
        with pytest.raises(ValueError, match="Value must be one of *"):
            with pd.option_context(f"io.excel{read_ext}.reader", "abc"):
                pass

    def test_ignore_chartsheets(self, request, engine, read_ext):
        # GH 41448
        if read_ext == ".ods":
            pytest.skip("chartsheets do not exist in the ODF format")
        if engine == "pyxlsb":
            request.applymarker(
                pytest.mark.xfail(
                    reason="pyxlsb can't distinguish chartsheets from worksheets"
                )
            )
        with pd.ExcelFile("chartsheet" + read_ext) as excel:
            assert excel.sheet_names == ["Sheet1"]

    def test_corrupt_files_closed(self, engine, read_ext):
        # GH41778
        errors = (BadZipFile,)
        if engine is None:
            pytest.skip(f"Invalid test for engine={engine}")
        elif engine == "xlrd":
            import xlrd

            errors = (BadZipFile, xlrd.biffh.XLRDError)
        elif engine == "calamine":
            from python_calamine import CalamineError

            errors = (CalamineError,)

        with tm.ensure_clean(f"corrupt{read_ext}") as file:
            Path(file).write_text("corrupt", encoding="utf-8")
            with tm.assert_produces_warning(False):
                try:
                    pd.ExcelFile(file, engine=engine)
                except errors:
                    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\test_base.py ===
from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re

import numpy as np
import pytest

from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td

from pandas.core.dtypes.common import (
    is_any_real_numeric_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

import pandas as pd
from pandas import (
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    IntervalIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    TimedeltaIndex,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    _get_combined_index,
    ensure_index,
    ensure_index_from_sequences,
)


class TestIndex:
    @pytest.fixture
    def simple_index(self) -> Index:
        return Index(list("abcde"))

    def test_can_hold_identifiers(self, simple_index):
        index = simple_index
        key = index[0]
        assert index._can_hold_identifiers_and_holds_name(key) is True

    @pytest.mark.parametrize("index", ["datetime"], indirect=True)
    def test_new_axis(self, index):
        # TODO: a bunch of scattered tests check this deprecation is enforced.
        #  de-duplicate/centralize them.
        with pytest.raises(ValueError, match="Multi-dimensional indexing"):
            # GH#30588 multi-dimensional indexing deprecated
            index[None, :]

    def test_constructor_regular(self, index):
        tm.assert_contains_all(index, index)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_constructor_casting(self, index):
        # casting
        arr = np.array(index)
        new_index = Index(arr)
        tm.assert_contains_all(arr, new_index)
        tm.assert_index_equal(index, new_index)

    def test_constructor_copy(self, using_infer_string):
        index = Index(list("abc"), name="name")
        arr = np.array(index)
        new_index = Index(arr, copy=True, name="name")
        assert isinstance(new_index, Index)
        assert new_index.name == "name"
        if using_infer_string:
            tm.assert_extension_array_equal(
                new_index.values, pd.array(arr, dtype="str")
            )
        else:
            tm.assert_numpy_array_equal(arr, new_index.values)
        arr[0] = "SOMEBIGLONGSTRING"
        assert new_index[0] != "SOMEBIGLONGSTRING"

    @pytest.mark.parametrize("cast_as_obj", [True, False])
    @pytest.mark.parametrize(
        "index",
        [
            date_range(
                "2015-01-01 10:00",
                freq="D",
                periods=3,
                tz="US/Eastern",
                name="Green Eggs & Ham",
            ),  # DTI with tz
            date_range("2015-01-01 10:00", freq="D", periods=3),  # DTI no tz
            timedelta_range("1 days", freq="D", periods=3),  # td
            period_range("2015-01-01", freq="D", periods=3),  # period
        ],
    )
    def test_constructor_from_index_dtlike(self, cast_as_obj, index):
        if cast_as_obj:
            with tm.assert_produces_warning(FutureWarning, match="Dtype inference"):
                result = Index(index.astype(object))
        else:
            result = Index(index)

        tm.assert_index_equal(result, index)

        if isinstance(index, DatetimeIndex):
            assert result.tz == index.tz
            if cast_as_obj:
                # GH#23524 check that Index(dti, dtype=object) does not
                #  incorrectly raise ValueError, and that nanoseconds are not
                #  dropped
                index += pd.Timedelta(nanoseconds=50)
                result = Index(index, dtype=object)
                assert result.dtype == np.object_
                assert list(result) == list(index)

    @pytest.mark.parametrize(
        "index,has_tz",
        [
            (
                date_range("2015-01-01 10:00", freq="D", periods=3, tz="US/Eastern"),
                True,
            ),  # datetimetz
            (timedelta_range("1 days", freq="D", periods=3), False),  # td
            (period_range("2015-01-01", freq="D", periods=3), False),  # period
        ],
    )
    def test_constructor_from_series_dtlike(self, index, has_tz):
        result = Index(Series(index))
        tm.assert_index_equal(result, index)

        if has_tz:
            assert result.tz == index.tz

    def test_constructor_from_series_freq(self):
        # GH 6273
        # create from a series, passing a freq
        dts = ["1-1-1990", "2-1-1990", "3-1-1990", "4-1-1990", "5-1-1990"]
        expected = DatetimeIndex(dts, freq="MS")

        s = Series(pd.to_datetime(dts))
        result = DatetimeIndex(s, freq="MS")

        tm.assert_index_equal(result, expected)

    def test_constructor_from_frame_series_freq(self, using_infer_string):
        # GH 6273
        # create from a series, passing a freq
        dts = ["1-1-1990", "2-1-1990", "3-1-1990", "4-1-1990", "5-1-1990"]
        expected = DatetimeIndex(dts, freq="MS")

        df = DataFrame(np.random.default_rng(2).random((5, 3)))
        df["date"] = dts
        result = DatetimeIndex(df["date"], freq="MS")
        dtype = object if not using_infer_string else "str"
        assert df["date"].dtype == dtype
        expected.name = "date"
        tm.assert_index_equal(result, expected)

        expected = Series(dts, name="date")
        tm.assert_series_equal(df["date"], expected)

        # GH 6274
        # infer freq of same
        if not using_infer_string:
            # Doesn't work with arrow strings
            freq = pd.infer_freq(df["date"])
            assert freq == "MS"

    def test_constructor_int_dtype_nan(self):
        # see gh-15187
        data = [np.nan]
        expected = Index(data, dtype=np.float64)
        result = Index(data, dtype="float")
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "klass,dtype,na_val",
        [
            (Index, np.float64, np.nan),
            (DatetimeIndex, "datetime64[ns]", pd.NaT),
        ],
    )
    def test_index_ctor_infer_nan_nat(self, klass, dtype, na_val):
        # GH 13467
        na_list = [na_val, na_val]
        expected = klass(na_list)
        assert expected.dtype == dtype

        result = Index(na_list)
        tm.assert_index_equal(result, expected)

        result = Index(np.array(na_list))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "vals,dtype",
        [
            ([1, 2, 3, 4, 5], "int"),
            ([1.1, np.nan, 2.2, 3.0], "float"),
            (["A", "B", "C", np.nan], "obj"),
        ],
    )
    def test_constructor_simple_new(self, vals, dtype):
        index = Index(vals, name=dtype)
        result = index._simple_new(index.values, dtype)
        tm.assert_index_equal(result, index)

    @pytest.mark.parametrize("attr", ["values", "asi8"])
    @pytest.mark.parametrize("klass", [Index, DatetimeIndex])
    def test_constructor_dtypes_datetime(self, tz_naive_fixture, attr, klass):
        # Test constructing with a datetimetz dtype
        # .values produces numpy datetimes, so these are considered naive
        # .asi8 produces integers, so these are considered epoch timestamps
        # ^the above will be true in a later version. Right now we `.view`
        # the i8 values as NS_DTYPE, effectively treating them as wall times.
        index = date_range("2011-01-01", periods=5)
        arg = getattr(index, attr)
        index = index.tz_localize(tz_naive_fixture)
        dtype = index.dtype

        # As of 2.0 astype raises on dt64.astype(dt64tz)
        err = tz_naive_fixture is not None
        msg = "Cannot use .astype to convert from timezone-naive dtype to"

        if attr == "asi8":
            result = DatetimeIndex(arg).tz_localize(tz_naive_fixture)
            tm.assert_index_equal(result, index)
        elif klass is Index:
            with pytest.raises(TypeError, match="unexpected keyword"):
                klass(arg, tz=tz_naive_fixture)
        else:
            result = klass(arg, tz=tz_naive_fixture)
            tm.assert_index_equal(result, index)

        if attr == "asi8":
            if err:
                with pytest.raises(TypeError, match=msg):
                    DatetimeIndex(arg).astype(dtype)
            else:
                result = DatetimeIndex(arg).astype(dtype)
                tm.assert_index_equal(result, index)
        else:
            result = klass(arg, dtype=dtype)
            tm.assert_index_equal(result, index)

        if attr == "asi8":
            result = DatetimeIndex(list(arg)).tz_localize(tz_naive_fixture)
            tm.assert_index_equal(result, index)
        elif klass is Index:
            with pytest.raises(TypeError, match="unexpected keyword"):
                klass(arg, tz=tz_naive_fixture)
        else:
            result = klass(list(arg), tz=tz_naive_fixture)
            tm.assert_index_equal(result, index)

        if attr == "asi8":
            if err:
                with pytest.raises(TypeError, match=msg):
                    DatetimeIndex(list(arg)).astype(dtype)
            else:
                result = DatetimeIndex(list(arg)).astype(dtype)
                tm.assert_index_equal(result, index)
        else:
            result = klass(list(arg), dtype=dtype)
            tm.assert_index_equal(result, index)

    @pytest.mark.parametrize("attr", ["values", "asi8"])
    @pytest.mark.parametrize("klass", [Index, TimedeltaIndex])
    def test_constructor_dtypes_timedelta(self, attr, klass):
        index = timedelta_range("1 days", periods=5)
        index = index._with_freq(None)  # won't be preserved by constructors
        dtype = index.dtype

        values = getattr(index, attr)

        result = klass(values, dtype=dtype)
        tm.assert_index_equal(result, index)

        result = klass(list(values), dtype=dtype)
        tm.assert_index_equal(result, index)

    @pytest.mark.parametrize("value", [[], iter([]), (_ for _ in [])])
    @pytest.mark.parametrize(
        "klass",
        [
            Index,
            CategoricalIndex,
            DatetimeIndex,
            TimedeltaIndex,
        ],
    )
    def test_constructor_empty(self, value, klass):
        empty = klass(value)
        assert isinstance(empty, klass)
        assert not len(empty)

    @pytest.mark.parametrize(
        "empty,klass",
        [
            (PeriodIndex([], freq="D"), PeriodIndex),
            (PeriodIndex(iter([]), freq="D"), PeriodIndex),
            (PeriodIndex((_ for _ in []), freq="D"), PeriodIndex),
            (RangeIndex(step=1), RangeIndex),
            (MultiIndex(levels=[[1, 2], ["blue", "red"]], codes=[[], []]), MultiIndex),
        ],
    )
    def test_constructor_empty_special(self, empty, klass):
        assert isinstance(empty, klass)
        assert not len(empty)

    @pytest.mark.parametrize(
        "index",
        [
            "datetime",
            "float64",
            "float32",
            "int64",
            "int32",
            "period",
            "range",
            "repeats",
            "timedelta",
            "tuples",
            "uint64",
            "uint32",
        ],
        indirect=True,
    )
    def test_view_with_args(self, index):
        index.view("i8")

    @pytest.mark.parametrize(
        "index",
        [
            "string",
            pytest.param("categorical", marks=pytest.mark.xfail(reason="gh-25464")),
            "bool-object",
            "bool-dtype",
            "empty",
        ],
        indirect=True,
    )
    def test_view_with_args_object_array_raises(self, index):
        if index.dtype == bool:
            msg = "When changing to a larger dtype"
            with pytest.raises(ValueError, match=msg):
                index.view("i8")
        else:
            msg = (
                r"Cannot change data-type for array of references\.|"
                r"Cannot change data-type for object array\.|"
                r"Cannot change data-type for array of strings\.|"
            )
            with pytest.raises(TypeError, match=msg):
                index.view("i8")

    @pytest.mark.parametrize(
        "index",
        ["int64", "int32", "range"],
        indirect=True,
    )
    def test_astype(self, index):
        casted = index.astype("i8")

        # it works!
        casted.get_loc(5)

        # pass on name
        index.name = "foobar"
        casted = index.astype("i8")
        assert casted.name == "foobar"

    def test_equals_object(self):
        # same
        assert Index(["a", "b", "c"]).equals(Index(["a", "b", "c"]))

    @pytest.mark.parametrize(
        "comp", [Index(["a", "b"]), Index(["a", "b", "d"]), ["a", "b", "c"]]
    )
    def test_not_equals_object(self, comp):
        assert not Index(["a", "b", "c"]).equals(comp)

    def test_identical(self):
        # index
        i1 = Index(["a", "b", "c"])
        i2 = Index(["a", "b", "c"])

        assert i1.identical(i2)

        i1 = i1.rename("foo")
        assert i1.equals(i2)
        assert not i1.identical(i2)

        i2 = i2.rename("foo")
        assert i1.identical(i2)

        i3 = Index([("a", "a"), ("a", "b"), ("b", "a")])
        i4 = Index([("a", "a"), ("a", "b"), ("b", "a")], tupleize_cols=False)
        assert not i3.identical(i4)

    def test_is_(self):
        ind = Index(range(10))
        assert ind.is_(ind)
        assert ind.is_(ind.view().view().view().view())
        assert not ind.is_(Index(range(10)))
        assert not ind.is_(ind.copy())
        assert not ind.is_(ind.copy(deep=False))
        assert not ind.is_(ind[:])
        assert not ind.is_(np.array(range(10)))

        # quasi-implementation dependent
        assert ind.is_(ind.view())
        ind2 = ind.view()
        ind2.name = "bob"
        assert ind.is_(ind2)
        assert ind2.is_(ind)
        # doesn't matter if Indices are *actually* views of underlying data,
        assert not ind.is_(Index(ind.values))
        arr = np.array(range(1, 11))
        ind1 = Index(arr, copy=False)
        ind2 = Index(arr, copy=False)
        assert not ind1.is_(ind2)

    def test_asof_numeric_vs_bool_raises(self):
        left = Index([1, 2, 3])
        right = Index([True, False], dtype=object)

        msg = "Cannot compare dtypes int64 and bool"
        with pytest.raises(TypeError, match=msg):
            left.asof(right[0])
        # TODO: should right.asof(left[0]) also raise?

        with pytest.raises(InvalidIndexError, match=re.escape(str(right))):
            left.asof(right)

        with pytest.raises(InvalidIndexError, match=re.escape(str(left))):
            right.asof(left)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_booleanindex(self, index):
        bool_index = np.ones(len(index), dtype=bool)
        bool_index[5:30:2] = False

        sub_index = index[bool_index]

        for i, val in enumerate(sub_index):
            assert sub_index.get_loc(val) == i

        sub_index = index[list(bool_index)]
        for i, val in enumerate(sub_index):
            assert sub_index.get_loc(val) == i

    def test_fancy(self, simple_index):
        index = simple_index
        sl = index[[1, 2, 3]]
        for i in sl:
            assert i == sl[sl.get_loc(i)]

    @pytest.mark.parametrize(
        "index",
        ["string", "int64", "int32", "uint64", "uint32", "float64", "float32"],
        indirect=True,
    )
    @pytest.mark.parametrize("dtype", [int, np.bool_])
    def test_empty_fancy(self, index, dtype, request, using_infer_string):
        if dtype is np.bool_ and using_infer_string and index.dtype == "string":
            request.applymarker(pytest.mark.xfail(reason="numpy behavior is buggy"))
        empty_arr = np.array([], dtype=dtype)
        empty_index = type(index)([], dtype=index.dtype)

        assert index[[]].identical(empty_index)
        if dtype == np.bool_:
            with tm.assert_produces_warning(FutureWarning, match="is deprecated"):
                assert index[empty_arr].identical(empty_index)
        else:
            assert index[empty_arr].identical(empty_index)

    @pytest.mark.parametrize(
        "index",
        ["string", "int64", "int32", "uint64", "uint32", "float64", "float32"],
        indirect=True,
    )
    def test_empty_fancy_raises(self, index):
        # DatetimeIndex is excluded, because it overrides getitem and should
        # be tested separately.
        empty_farr = np.array([], dtype=np.float64)
        empty_index = type(index)([], dtype=index.dtype)

        assert index[[]].identical(empty_index)
        # np.ndarray only accepts ndarray of int & bool dtypes, so should Index
        msg = r"arrays used as indices must be of integer"
        with pytest.raises(IndexError, match=msg):
            index[empty_farr]

    def test_union_dt_as_obj(self, simple_index):
        # TODO: Replace with fixturesult
        index = simple_index
        date_index = date_range("2019-01-01", periods=10)
        first_cat = index.union(date_index)
        second_cat = index.union(index)

        appended = Index(np.append(index, date_index.astype("O")))

        tm.assert_index_equal(first_cat, appended)
        tm.assert_index_equal(second_cat, index)
        tm.assert_contains_all(index, first_cat)
        tm.assert_contains_all(index, second_cat)
        tm.assert_contains_all(date_index, first_cat)

    def test_map_with_tuples(self):
        # GH 12766

        # Test that returning a single tuple from an Index
        #   returns an Index.
        index = Index(np.arange(3), dtype=np.int64)
        result = index.map(lambda x: (x,))
        expected = Index([(i,) for i in index])
        tm.assert_index_equal(result, expected)

        # Test that returning a tuple from a map of a single index
        #   returns a MultiIndex object.
        result = index.map(lambda x: (x, x == 1))
        expected = MultiIndex.from_tuples([(i, i == 1) for i in index])
        tm.assert_index_equal(result, expected)

    def test_map_with_tuples_mi(self):
        # Test that returning a single object from a MultiIndex
        #   returns an Index.
        first_level = ["foo", "bar", "baz"]
        multi_index = MultiIndex.from_tuples(zip(first_level, [1, 2, 3]))
        reduced_index = multi_index.map(lambda x: x[0])
        tm.assert_index_equal(reduced_index, Index(first_level))

    @pytest.mark.parametrize(
        "index",
        [
            date_range("2020-01-01", freq="D", periods=10),
            period_range("2020-01-01", freq="D", periods=10),
            timedelta_range("1 day", periods=10),
        ],
    )
    def test_map_tseries_indices_return_index(self, index):
        expected = Index([1] * 10)
        result = index.map(lambda x: 1)
        tm.assert_index_equal(expected, result)

    def test_map_tseries_indices_accsr_return_index(self):
        date_index = DatetimeIndex(
            date_range("2020-01-01", periods=24, freq="h"), name="hourly"
        )
        result = date_index.map(lambda x: x.hour)
        expected = Index(np.arange(24, dtype="int64"), name="hourly")
        tm.assert_index_equal(result, expected, exact=True)

    @pytest.mark.parametrize(
        "mapper",
        [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: Series(values, index),
        ],
    )
    def test_map_dictlike_simple(self, mapper):
        # GH 12756
        expected = Index(["foo", "bar", "baz"])
        index = Index(np.arange(3), dtype=np.int64)
        result = index.map(mapper(expected.values, index))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "mapper",
        [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: Series(values, index),
        ],
    )
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_map_dictlike(self, index, mapper, request):
        # GH 12756
        if isinstance(index, CategoricalIndex):
            pytest.skip("Tested in test_categorical")
        elif not index.is_unique:
            pytest.skip("Cannot map duplicated index")

        rng = np.arange(len(index), 0, -1, dtype=np.int64)

        if index.empty:
            # to match proper result coercion for uints
            expected = Index([])
        elif is_numeric_dtype(index.dtype):
            expected = index._constructor(rng, dtype=index.dtype)
        elif type(index) is Index and index.dtype != object:
            # i.e. EA-backed, for now just Nullable
            expected = Index(rng, dtype=index.dtype)
        else:
            expected = Index(rng)

        result = index.map(mapper(expected, index))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "mapper",
        [Series(["foo", 2.0, "baz"], index=[0, 2, -1]), {0: "foo", 2: 2.0, -1: "baz"}],
    )
    def test_map_with_non_function_missing_values(self, mapper):
        # GH 12756
        expected = Index([2.0, np.nan, "foo"])
        result = Index([2, 1, 0]).map(mapper)

        tm.assert_index_equal(expected, result)

    def test_map_na_exclusion(self):
        index = Index([1.5, np.nan, 3, np.nan, 5])

        result = index.map(lambda x: x * 2, na_action="ignore")
        expected = index * 2
        tm.assert_index_equal(result, expected)

    def test_map_defaultdict(self):
        index = Index([1, 2, 3])
        default_dict = defaultdict(lambda: "blank")
        default_dict[1] = "stuff"
        result = index.map(default_dict)
        expected = Index(["stuff", "blank", "blank"])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("name,expected", [("foo", "foo"), ("bar", None)])
    def test_append_empty_preserve_name(self, name, expected):
        left = Index([], name="foo")
        right = Index([1, 2, 3], name=name)

        msg = "The behavior of array concatenation with empty entries is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = left.append(right)
        assert result.name == expected

    @pytest.mark.parametrize(
        "index, expected",
        [
            ("string", False),
            ("bool-object", False),
            ("bool-dtype", False),
            ("categorical", False),
            ("int64", True),
            ("int32", True),
            ("uint64", True),
            ("uint32", True),
            ("datetime", False),
            ("float64", True),
            ("float32", True),
        ],
        indirect=["index"],
    )
    def test_is_numeric(self, index, expected):
        assert is_any_real_numeric_dtype(index) is expected

    @pytest.mark.parametrize(
        "index, expected",
        [
            ("string", True),
            ("bool-object", True),
            ("bool-dtype", False),
            ("categorical", False),
            ("int64", False),
            ("int32", False),
            ("uint64", False),
            ("uint32", False),
            ("datetime", False),
            ("float64", False),
            ("float32", False),
        ],
        indirect=["index"],
    )
    def test_is_object(self, index, expected, using_infer_string):
        if using_infer_string and index.dtype == "string" and expected:
            expected = False
        assert is_object_dtype(index) is expected

    def test_summary(self, index):
        index._summary()

    def test_format_bug(self):
        # GH 14626
        # windows has different precision on datetime.datetime.now (it doesn't
        # include us since the default for Timestamp shows these but Index
        # formatting does not we are skipping)
        now = datetime.now()
        msg = r"Index\.format is deprecated"

        if not str(now).endswith("000"):
            index = Index([now])
            with tm.assert_produces_warning(FutureWarning, match=msg):
                formatted = index.format()
            expected = [str(index[0])]
            assert formatted == expected

        with tm.assert_produces_warning(FutureWarning, match=msg):
            Index([]).format()

    @pytest.mark.parametrize("vals", [[1, 2.0 + 3.0j, 4.0], ["a", "b", "c"]])
    def test_format_missing(self, vals, nulls_fixture):
        # 2845
        vals = list(vals)  # Copy for each iteration
        vals.append(nulls_fixture)
        index = Index(vals, dtype=object)
        # TODO: case with complex dtype?

        msg = r"Index\.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = index.format()
        null_repr = "NaN" if isinstance(nulls_fixture, float) else str(nulls_fixture)
        expected = [str(index[0]), str(index[1]), str(index[2]), null_repr]

        assert formatted == expected
        assert index[3] is nulls_fixture

    @pytest.mark.parametrize("op", ["any", "all"])
    def test_logical_compat(self, op, simple_index):
        index = simple_index
        left = getattr(index, op)()
        assert left == getattr(index.values, op)()
        right = getattr(index.to_series(), op)()
        # left might not match right exactly in e.g. string cases where the
        # because we use np.any/all instead of .any/all
        assert bool(left) == bool(right)

    @pytest.mark.parametrize(
        "index", ["string", "int64", "int32", "float64", "float32"], indirect=True
    )
    def test_drop_by_str_label(self, index):
        n = len(index)
        drop = index[list(range(5, 10))]
        dropped = index.drop(drop)

        expected = index[list(range(5)) + list(range(10, n))]
        tm.assert_index_equal(dropped, expected)

        dropped = index.drop(index[0])
        expected = index[1:]
        tm.assert_index_equal(dropped, expected)

    @pytest.mark.parametrize(
        "index", ["string", "int64", "int32", "float64", "float32"], indirect=True
    )
    @pytest.mark.parametrize("keys", [["foo", "bar"], ["1", "bar"]])
    def test_drop_by_str_label_raises_missing_keys(self, index, keys):
        with pytest.raises(KeyError, match=""):
            index.drop(keys)

    @pytest.mark.parametrize(
        "index", ["string", "int64", "int32", "float64", "float32"], indirect=True
    )
    def test_drop_by_str_label_errors_ignore(self, index):
        n = len(index)
        drop = index[list(range(5, 10))]
        mixed = drop.tolist() + ["foo"]
        dropped = index.drop(mixed, errors="ignore")

        expected = index[list(range(5)) + list(range(10, n))]
        tm.assert_index_equal(dropped, expected)

        dropped = index.drop(["foo", "bar"], errors="ignore")
        expected = index[list(range(n))]
        tm.assert_index_equal(dropped, expected)

    def test_drop_by_numeric_label_loc(self):
        # TODO: Parametrize numeric and str tests after self.strIndex fixture
        index = Index([1, 2, 3])
        dropped = index.drop(1)
        expected = Index([2, 3])

        tm.assert_index_equal(dropped, expected)

    def test_drop_by_numeric_label_raises_missing_keys(self):
        index = Index([1, 2, 3])
        with pytest.raises(KeyError, match=""):
            index.drop([3, 4])

    @pytest.mark.parametrize(
        "key,expected", [(4, Index([1, 2, 3])), ([3, 4, 5], Index([1, 2]))]
    )
    def test_drop_by_numeric_label_errors_ignore(self, key, expected):
        index = Index([1, 2, 3])
        dropped = index.drop(key, errors="ignore")

        tm.assert_index_equal(dropped, expected)

    @pytest.mark.parametrize(
        "values",
        [["a", "b", ("c", "d")], ["a", ("c", "d"), "b"], [("c", "d"), "a", "b"]],
    )
    @pytest.mark.parametrize("to_drop", [[("c", "d"), "a"], ["a", ("c", "d")]])
    def test_drop_tuple(self, values, to_drop):
        # GH 18304
        index = Index(values)
        expected = Index(["b"], dtype=object)

        result = index.drop(to_drop)
        tm.assert_index_equal(result, expected)

        removed = index.drop(to_drop[0])
        for drop_me in to_drop[1], [to_drop[1]]:
            result = removed.drop(drop_me)
            tm.assert_index_equal(result, expected)

        removed = index.drop(to_drop[1])
        msg = rf"\"\[{re.escape(to_drop[1].__repr__())}\] not found in axis\""
        for drop_me in to_drop[1], [to_drop[1]]:
            with pytest.raises(KeyError, match=msg):
                removed.drop(drop_me)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_drop_with_duplicates_in_index(self, index):
        # GH38051
        if len(index) == 0 or isinstance(index, MultiIndex):
            pytest.skip("Test doesn't make sense for empty MultiIndex")
        if isinstance(index, IntervalIndex) and not IS64:
            pytest.skip("Cannot test IntervalIndex with int64 dtype on 32 bit platform")
        index = index.unique().repeat(2)
        expected = index[2:]
        result = index.drop(index[0])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "attr",
        [
            "is_monotonic_increasing",
            "is_monotonic_decreasing",
            "_is_strictly_monotonic_increasing",
            "_is_strictly_monotonic_decreasing",
        ],
    )
    def test_is_monotonic_incomparable(self, attr):
        index = Index([5, datetime.now(), 7])
        assert not getattr(index, attr)

    @pytest.mark.parametrize("values", [["foo", "bar", "quux"], {"foo", "bar", "quux"}])
    @pytest.mark.parametrize(
        "index,expected",
        [
            (Index(["qux", "baz", "foo", "bar"]), np.array([False, False, True, True])),
            (Index([]), np.array([], dtype=bool)),  # empty
        ],
    )
    def test_isin(self, values, index, expected):
        result = index.isin(values)
        tm.assert_numpy_array_equal(result, expected)

    def test_isin_nan_common_object(
        self, nulls_fixture, nulls_fixture2, using_infer_string
    ):
        # Test cartesian product of null fixtures and ensure that we don't
        # mangle the various types (save a corner case with PyPy)
        idx = Index(["a", nulls_fixture])

        # all nans are the same
        if (
            isinstance(nulls_fixture, float)
            and isinstance(nulls_fixture2, float)
            and math.isnan(nulls_fixture)
            and math.isnan(nulls_fixture2)
        ):
            tm.assert_numpy_array_equal(
                idx.isin([nulls_fixture2]),
                np.array([False, True]),
            )

        elif nulls_fixture is nulls_fixture2:  # should preserve NA type
            tm.assert_numpy_array_equal(
                idx.isin([nulls_fixture2]),
                np.array([False, True]),
            )

        elif using_infer_string and idx.dtype == "string":
            tm.assert_numpy_array_equal(
                idx.isin([nulls_fixture2]),
                np.array([False, True]),
            )

        else:
            tm.assert_numpy_array_equal(
                idx.isin([nulls_fixture2]),
                np.array([False, False]),
            )

    def test_isin_nan_common_float64(self, nulls_fixture, float_numpy_dtype):
        dtype = float_numpy_dtype

        if nulls_fixture is pd.NaT or nulls_fixture is pd.NA:
            # Check 1) that we cannot construct a float64 Index with this value
            #  and 2) that with an NaN we do not have .isin(nulls_fixture)
            msg = (
                r"float\(\) argument must be a string or a (real )?number, "
                f"not {repr(type(nulls_fixture).__name__)}"
            )
            with pytest.raises(TypeError, match=msg):
                Index([1.0, nulls_fixture], dtype=dtype)

            idx = Index([1.0, np.nan], dtype=dtype)
            assert not idx.isin([nulls_fixture]).any()
            return

        idx = Index([1.0, nulls_fixture], dtype=dtype)
        res = idx.isin([np.nan])
        tm.assert_numpy_array_equal(res, np.array([False, True]))

        # we cannot compare NaT with NaN
        res = idx.isin([pd.NaT])
        tm.assert_numpy_array_equal(res, np.array([False, False]))

    @pytest.mark.parametrize("level", [0, -1])
    @pytest.mark.parametrize(
        "index",
        [
            Index(["qux", "baz", "foo", "bar"]),
            Index([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        ],
    )
    def test_isin_level_kwarg(self, level, index):
        values = index.tolist()[-2:] + ["nonexisting"]

        expected = np.array([False, False, True, True])
        tm.assert_numpy_array_equal(expected, index.isin(values, level=level))

        index.name = "foobar"
        tm.assert_numpy_array_equal(expected, index.isin(values, level="foobar"))

    def test_isin_level_kwarg_bad_level_raises(self, index):
        for level in [10, index.nlevels, -(index.nlevels + 1)]:
            with pytest.raises(IndexError, match="Too many levels"):
                index.isin([], level=level)

    @pytest.mark.parametrize("label", [1.0, "foobar", "xyzzy", np.nan])
    def test_isin_level_kwarg_bad_label_raises(self, label, index):
        if isinstance(index, MultiIndex):
            index = index.rename(["foo", "bar"] + index.names[2:])
            msg = f"'Level {label} not found'"
        else:
            index = index.rename("foo")
            msg = rf"Requested level \({label}\) does not match index name \(foo\)"
        with pytest.raises(KeyError, match=msg):
            index.isin([], level=label)

    @pytest.mark.parametrize("empty", [[], Series(dtype=object), np.array([])])
    def test_isin_empty(self, empty):
        # see gh-16991
        index = Index(["a", "b"])
        expected = np.array([False, False])

        result = index.isin(empty)
        tm.assert_numpy_array_equal(expected, result)

    def test_isin_string_null(self, string_dtype_no_object):
        # GH#55821
        index = Index(["a", "b"], dtype=string_dtype_no_object)
        result = index.isin([None])
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "values",
        [
            [1, 2, 3, 4],
            [1.0, 2.0, 3.0, 4.0],
            [True, True, True, True],
            ["foo", "bar", "baz", "qux"],
            date_range("2018-01-01", freq="D", periods=4),
        ],
    )
    def test_boolean_cmp(self, values):
        index = Index(values)
        result = index == values
        expected = np.array([True, True, True, True], dtype=bool)

        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    @pytest.mark.parametrize("name,level", [(None, 0), ("a", "a")])
    def test_get_level_values(self, index, name, level):
        expected = index.copy()
        if name:
            expected.name = name

        result = expected.get_level_values(level)
        tm.assert_index_equal(result, expected)

    def test_slice_keep_name(self):
        index = Index(["a", "b"], name="asdf")
        assert index.name == index[1:].name

    @pytest.mark.parametrize(
        "index",
        [
            "string",
            "datetime",
            "int64",
            "int32",
            "uint64",
            "uint32",
            "float64",
            "float32",
        ],
        indirect=True,
    )
    def test_join_self(self, index, join_type):
        result = index.join(index, how=join_type)
        expected = index
        if join_type == "outer":
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("method", ["strip", "rstrip", "lstrip"])
    def test_str_attribute(self, method):
        # GH9068
        index = Index([" jack", "jill ", " jesse ", "frank"])
        expected = Index([getattr(str, method)(x) for x in index.values])

        result = getattr(index.str, method)()
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "index",
        [
            Index(range(5)),
            date_range("2020-01-01", periods=10),
            MultiIndex.from_tuples([("foo", "1"), ("bar", "3")]),
            period_range(start="2000", end="2010", freq="Y"),
        ],
    )
    def test_str_attribute_raises(self, index):
        with pytest.raises(AttributeError, match="only use .str accessor"):
            index.str.repeat(2)

    @pytest.mark.parametrize(
        "expand,expected",
        [
            (None, Index([["a", "b", "c"], ["d", "e"], ["f"]])),
            (False, Index([["a", "b", "c"], ["d", "e"], ["f"]])),
            (
                True,
                MultiIndex.from_tuples(
                    [("a", "b", "c"), ("d", "e", np.nan), ("f", np.nan, np.nan)]
                ),
            ),
        ],
    )
    def test_str_split(self, expand, expected):
        index = Index(["a b c", "d e", "f"])
        if expand is not None:
            result = index.str.split(expand=expand)
        else:
            result = index.str.split()

        tm.assert_index_equal(result, expected)

    def test_str_bool_return(self):
        # test boolean case, should return np.array instead of boolean Index
        index = Index(["a1", "a2", "b1", "b2"])
        result = index.str.startswith("a")
        expected = np.array([True, True, False, False])

        tm.assert_numpy_array_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_str_bool_series_indexing(self):
        index = Index(["a1", "a2", "b1", "b2"])
        s = Series(range(4), index=index)

        result = s[s.index.str.startswith("a")]
        expected = Series(range(2), index=["a1", "a2"])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "index,expected", [(Index(list("abcd")), True), (Index(range(4)), False)]
    )
    def test_tab_completion(self, index, expected):
        # GH 9910
        result = "str" in dir(index)
        assert result == expected

    def test_indexing_doesnt_change_class(self):
        index = Index([1, 2, 3, "a", "b", "c"])

        assert index[1:3].identical(Index([2, 3], dtype=np.object_))
        assert index[[0, 1]].identical(Index([1, 2], dtype=np.object_))

    def test_outer_join_sort(self):
        left_index = Index(np.random.default_rng(2).permutation(15))
        right_index = date_range("2020-01-01", periods=10)

        with tm.assert_produces_warning(RuntimeWarning):
            result = left_index.join(right_index, how="outer")

        with tm.assert_produces_warning(RuntimeWarning):
            expected = left_index.astype(object).union(right_index.astype(object))

        tm.assert_index_equal(result, expected)

    def test_take_fill_value(self):
        # GH 12631
        index = Index(list("ABC"), name="xxx")
        result = index.take(np.array([1, 0, -1]))
        expected = Index(list("BAC"), name="xxx")
        tm.assert_index_equal(result, expected)

        # fill_value
        result = index.take(np.array([1, 0, -1]), fill_value=True)
        expected = Index(["B", "A", np.nan], name="xxx")
        tm.assert_index_equal(result, expected)

        # allow_fill=False
        result = index.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = Index(["B", "A", "C"], name="xxx")
        tm.assert_index_equal(result, expected)

    def test_take_fill_value_none_raises(self):
        index = Index(list("ABC"), name="xxx")
        msg = (
            "When allow_fill=True and fill_value is not None, "
            "all indices must be >= -1"
        )

        with pytest.raises(ValueError, match=msg):
            index.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            index.take(np.array([1, 0, -5]), fill_value=True)

    def test_take_bad_bounds_raises(self):
        index = Index(list("ABC"), name="xxx")
        with pytest.raises(IndexError, match="out of bounds"):
            index.take(np.array([1, -5]))

    @pytest.mark.parametrize("name", [None, "foobar"])
    @pytest.mark.parametrize(
        "labels",
        [
            [],
            np.array([]),
            ["A", "B", "C"],
            ["C", "B", "A"],
            np.array(["A", "B", "C"]),
            np.array(["C", "B", "A"]),
            # Must preserve name even if dtype changes
            date_range("20130101", periods=3).values,
            date_range("20130101", periods=3).tolist(),
        ],
    )
    def test_reindex_preserves_name_if_target_is_list_or_ndarray(self, name, labels):
        # GH6552
        index = Index([0, 1, 2])
        index.name = name
        assert index.reindex(labels)[0].name == name

    @pytest.mark.parametrize("labels", [[], np.array([]), np.array([], dtype=np.int64)])
    def test_reindex_preserves_type_if_target_is_empty_list_or_array(self, labels):
        # GH7774
        index = Index(list("abc"))
        assert index.reindex(labels)[0].dtype.type == index.dtype.type

    @pytest.mark.parametrize(
        "labels,dtype",
        [
            (DatetimeIndex([]), np.datetime64),
        ],
    )
    def test_reindex_doesnt_preserve_type_if_target_is_empty_index(self, labels, dtype):
        # GH7774
        index = Index(list("abc"))
        assert index.reindex(labels)[0].dtype.type == dtype

    def test_reindex_doesnt_preserve_type_if_target_is_empty_index_numeric(
        self, any_real_numpy_dtype
    ):
        # GH7774
        dtype = any_real_numpy_dtype
        index = Index(list("abc"))
        labels = Index([], dtype=dtype)
        assert index.reindex(labels)[0].dtype == dtype

    def test_reindex_no_type_preserve_target_empty_mi(self):
        index = Index(list("abc"))
        result = index.reindex(
            MultiIndex([Index([], np.int64), Index([], np.float64)], [[], []])
        )[0]
        assert result.levels[0].dtype.type == np.int64
        assert result.levels[1].dtype.type == np.float64

    def test_reindex_ignoring_level(self):
        # GH#35132
        idx = Index([1, 2, 3], name="x")
        idx2 = Index([1, 2, 3, 4], name="x")
        expected = Index([1, 2, 3, 4], name="x")
        result, _ = idx.reindex(idx2, level="x")
        tm.assert_index_equal(result, expected)

    def test_groupby(self):
        index = Index(range(5))
        result = index.groupby(np.array([1, 1, 2, 2, 2]))
        expected = {1: Index([0, 1]), 2: Index([2, 3, 4])}

        tm.assert_dict_equal(result, expected)

    @pytest.mark.parametrize(
        "mi,expected",
        [
            (MultiIndex.from_tuples([(1, 2), (4, 5)]), np.array([True, True])),
            (MultiIndex.from_tuples([(1, 2), (4, 6)]), np.array([True, False])),
        ],
    )
    def test_equals_op_multiindex(self, mi, expected):
        # GH9785
        # test comparisons of multiindex
        df = DataFrame(
            [3, 6],
            columns=["c"],
            index=MultiIndex.from_arrays([[1, 4], [2, 5]], names=["a", "b"]),
        )

        result = df.index == mi
        tm.assert_numpy_array_equal(result, expected)

    def test_equals_op_multiindex_identify(self):
        df = DataFrame(
            [3, 6],
            columns=["c"],
            index=MultiIndex.from_arrays([[1, 4], [2, 5]], names=["a", "b"]),
        )

        result = df.index == df.index
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "index",
        [
            MultiIndex.from_tuples([(1, 2), (4, 5), (8, 9)]),
            Index(["foo", "bar", "baz"]),
        ],
    )
    def test_equals_op_mismatched_multiindex_raises(self, index):
        df = DataFrame(
            [3, 6],
            columns=["c"],
            index=MultiIndex.from_arrays([[1, 4], [2, 5]], names=["a", "b"]),
        )

        with pytest.raises(ValueError, match="Lengths must match"):
            df.index == index

    def test_equals_op_index_vs_mi_same_length(self, using_infer_string):
        mi = MultiIndex.from_tuples([(1, 2), (4, 5), (8, 9)])
        index = Index(["foo", "bar", "baz"])

        result = mi == index
        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "dt_conv, arg",
        [
            (pd.to_datetime, ["2000-01-01", "2000-01-02"]),
            (pd.to_timedelta, ["01:02:03", "01:02:04"]),
        ],
    )
    def test_dt_conversion_preserves_name(self, dt_conv, arg):
        # GH 10875
        index = Index(arg, name="label")
        assert index.name == dt_conv(index).name

    def test_cached_properties_not_settable(self):
        index = Index([1, 2, 3])
        with pytest.raises(AttributeError, match="Can't set attribute"):
            index.is_unique = False

    def test_tab_complete_warning(self, ip):
        # https://github.com/pandas-dev/pandas/issues/16409
        pytest.importorskip("IPython", minversion="6.0.0")
        from IPython.core.completer import provisionalcompleter

        code = "import pandas as pd; idx = pd.Index([1, 2])"
        ip.run_cell(code)

        # GH 31324 newer jedi version raises Deprecation warning;
        #  appears resolved 2021-02-02
        with tm.assert_produces_warning(None, raise_on_extra_warnings=False):
            with provisionalcompleter("ignore"):
                list(ip.Completer.completions("idx.", 4))

    def test_contains_method_removed(self, index):
        # GH#30103 method removed for all types except IntervalIndex
        if isinstance(index, IntervalIndex):
            index.contains(1)
        else:
            msg = f"'{type(index).__name__}' object has no attribute 'contains'"
            with pytest.raises(AttributeError, match=msg):
                index.contains(1)

    def test_sortlevel(self):
        index = Index([5, 4, 3, 2, 1])
        with pytest.raises(Exception, match="ascending must be a single bool value or"):
            index.sortlevel(ascending="True")

        with pytest.raises(
            Exception, match="ascending must be a list of bool values of length 1"
        ):
            index.sortlevel(ascending=[True, True])

        with pytest.raises(Exception, match="ascending must be a bool value"):
            index.sortlevel(ascending=["True"])

        expected = Index([1, 2, 3, 4, 5])
        result = index.sortlevel(ascending=[True])
        tm.assert_index_equal(result[0], expected)

        expected = Index([1, 2, 3, 4, 5])
        result = index.sortlevel(ascending=True)
        tm.assert_index_equal(result[0], expected)

        expected = Index([5, 4, 3, 2, 1])
        result = index.sortlevel(ascending=False)
        tm.assert_index_equal(result[0], expected)

    def test_sortlevel_na_position(self):
        # GH#51612
        idx = Index([1, np.nan])
        result = idx.sortlevel(na_position="first")[0]
        expected = Index([np.nan, 1])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "periods, expected_results",
        [
            (1, [np.nan, 10, 10, 10, 10]),
            (2, [np.nan, np.nan, 20, 20, 20]),
            (3, [np.nan, np.nan, np.nan, 30, 30]),
        ],
    )
    def test_index_diff(self, periods, expected_results):
        # GH#19708
        idx = Index([10, 20, 30, 40, 50])
        result = idx.diff(periods)
        expected = Index(expected_results)

        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "decimals, expected_results",
        [
            (0, [1.0, 2.0, 3.0]),
            (1, [1.2, 2.3, 3.5]),
            (2, [1.23, 2.35, 3.46]),
        ],
    )
    def test_index_round(self, decimals, expected_results):
        # GH#19708
        idx = Index([1.234, 2.345, 3.456])
        result = idx.round(decimals)
        expected = Index(expected_results)

        tm.assert_index_equal(result, expected)


class TestMixedIntIndex:
    # Mostly the tests from common.py for which the results differ
    # in py2 and py3 because ints and strings are uncomparable in py3
    # (GH 13514)
    @pytest.fixture
    def simple_index(self) -> Index:
        return Index([0, "a", 1, "b", 2, "c"])

    def test_argsort(self, simple_index):
        index = simple_index
        with pytest.raises(TypeError, match="'>|<' not supported"):
            index.argsort()

    def test_numpy_argsort(self, simple_index):
        index = simple_index
        with pytest.raises(TypeError, match="'>|<' not supported"):
            np.argsort(index)

    def test_copy_name(self, simple_index):
        # Check that "name" argument passed at initialization is honoured
        # GH12309
        index = simple_index

        first = type(index)(index, copy=True, name="mario")
        second = type(first)(first, copy=False)

        # Even though "copy=False", we want a new object.
        assert first is not second
        tm.assert_index_equal(first, second)

        assert first.name == "mario"
        assert second.name == "mario"

        s1 = Series(2, index=first)
        s2 = Series(3, index=second[:-1])

        s3 = s1 * s2

        assert s3.index.name == "mario"

    def test_copy_name2(self):
        # Check that adding a "name" parameter to the copy is honored
        # GH14302
        index = Index([1, 2], name="MyName")
        index1 = index.copy()

        tm.assert_index_equal(index, index1)

        index2 = index.copy(name="NewName")
        tm.assert_index_equal(index, index2, check_names=False)
        assert index.name == "MyName"
        assert index2.name == "NewName"

    def test_unique_na(self):
        idx = Index([2, np.nan, 2, 1], name="my_index")
        expected = Index([2, np.nan, 1], name="my_index")
        result = idx.unique()
        tm.assert_index_equal(result, expected)

    def test_logical_compat(self, simple_index):
        index = simple_index
        assert index.all() == index.values.all()
        assert index.any() == index.values.any()

    @pytest.mark.parametrize("how", ["any", "all"])
    @pytest.mark.parametrize("dtype", [None, object, "category"])
    @pytest.mark.parametrize(
        "vals,expected",
        [
            ([1, 2, 3], [1, 2, 3]),
            ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
            ([1.0, 2.0, np.nan, 3.0], [1.0, 2.0, 3.0]),
            (["A", "B", "C"], ["A", "B", "C"]),
            (["A", np.nan, "B", "C"], ["A", "B", "C"]),
        ],
    )
    def test_dropna(self, how, dtype, vals, expected):
        # GH 6194
        index = Index(vals, dtype=dtype)
        result = index.dropna(how=how)
        expected = Index(expected, dtype=dtype)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("how", ["any", "all"])
    @pytest.mark.parametrize(
        "index,expected",
        [
            (
                DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"]),
                DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"]),
            ),
            (
                DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03", pd.NaT]),
                DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"]),
            ),
            (
                TimedeltaIndex(["1 days", "2 days", "3 days"]),
                TimedeltaIndex(["1 days", "2 days", "3 days"]),
            ),
            (
                TimedeltaIndex([pd.NaT, "1 days", "2 days", "3 days", pd.NaT]),
                TimedeltaIndex(["1 days", "2 days", "3 days"]),
            ),
            (
                PeriodIndex(["2012-02", "2012-04", "2012-05"], freq="M"),
                PeriodIndex(["2012-02", "2012-04", "2012-05"], freq="M"),
            ),
            (
                PeriodIndex(["2012-02", "2012-04", "NaT", "2012-05"], freq="M"),
                PeriodIndex(["2012-02", "2012-04", "2012-05"], freq="M"),
            ),
        ],
    )
    def test_dropna_dt_like(self, how, index, expected):
        result = index.dropna(how=how)
        tm.assert_index_equal(result, expected)

    def test_dropna_invalid_how_raises(self):
        msg = "invalid how option: xxx"
        with pytest.raises(ValueError, match=msg):
            Index([1, 2, 3]).dropna(how="xxx")

    @pytest.mark.parametrize(
        "index",
        [
            Index([np.nan]),
            Index([np.nan, 1]),
            Index([1, 2, np.nan]),
            Index(["a", "b", np.nan]),
            pd.to_datetime(["NaT"]),
            pd.to_datetime(["NaT", "2000-01-01"]),
            pd.to_datetime(["2000-01-01", "NaT", "2000-01-02"]),
            pd.to_timedelta(["1 day", "NaT"]),
        ],
    )
    def test_is_monotonic_na(self, index):
        assert index.is_monotonic_increasing is False
        assert index.is_monotonic_decreasing is False
        assert index._is_strictly_monotonic_increasing is False
        assert index._is_strictly_monotonic_decreasing is False

    @pytest.mark.parametrize("dtype", ["f8", "m8[ns]", "M8[us]"])
    @pytest.mark.parametrize("unique_first", [True, False])
    def test_is_monotonic_unique_na(self, dtype, unique_first):
        # GH 55755
        index = Index([None, 1, 1], dtype=dtype)
        if unique_first:
            assert index.is_unique is False
            assert index.is_monotonic_increasing is False
            assert index.is_monotonic_decreasing is False
        else:
            assert index.is_monotonic_increasing is False
            assert index.is_monotonic_decreasing is False
            assert index.is_unique is False

    def test_int_name_format(self, frame_or_series):
        index = Index(["a", "b", "c"], name=0)
        result = frame_or_series(list(range(3)), index=index)
        assert "0" in repr(result)

    def test_str_to_bytes_raises(self):
        # GH 26447
        index = Index([str(x) for x in range(10)])
        msg = "^'str' object cannot be interpreted as an integer$"
        with pytest.raises(TypeError, match=msg):
            bytes(index)

    @pytest.mark.filterwarnings("ignore:elementwise comparison failed:FutureWarning")
    def test_index_with_tuple_bool(self):
        # GH34123
        # TODO: also this op right now produces FutureWarning from numpy
        #  https://github.com/numpy/numpy/issues/11521
        idx = Index([("a", "b"), ("b", "c"), ("c", "a")])
        result = idx == ("c", "a")
        expected = np.array([False, False, True])
        tm.assert_numpy_array_equal(result, expected)


class TestIndexUtils:
    @pytest.mark.parametrize(
        "data, names, expected",
        [
            ([[1, 2, 3]], None, Index([1, 2, 3])),
            ([[1, 2, 3]], ["name"], Index([1, 2, 3], name="name")),
            (
                [["a", "a"], ["c", "d"]],
                None,
                MultiIndex([["a"], ["c", "d"]], [[0, 0], [0, 1]]),
            ),
            (
                [["a", "a"], ["c", "d"]],
                ["L1", "L2"],
                MultiIndex([["a"], ["c", "d"]], [[0, 0], [0, 1]], names=["L1", "L2"]),
            ),
        ],
    )
    def test_ensure_index_from_sequences(self, data, names, expected):
        result = ensure_index_from_sequences(data, names)
        tm.assert_index_equal(result, expected)

    def test_ensure_index_mixed_closed_intervals(self):
        # GH27172
        intervals = [
            pd.Interval(0, 1, closed="left"),
            pd.Interval(1, 2, closed="right"),
            pd.Interval(2, 3, closed="neither"),
            pd.Interval(3, 4, closed="both"),
        ]
        result = ensure_index(intervals)
        expected = Index(intervals, dtype=object)
        tm.assert_index_equal(result, expected)

    def test_ensure_index_uint64(self):
        # with both 0 and a large-uint64, np.array will infer to float64
        #  https://github.com/numpy/numpy/issues/19146
        #  but a more accurate choice would be uint64
        values = [0, np.iinfo(np.uint64).max]

        result = ensure_index(values)
        assert list(result) == values

        expected = Index(values, dtype="uint64")
        tm.assert_index_equal(result, expected)

    def test_get_combined_index(self):
        result = _get_combined_index([])
        expected = Index([])
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "opname",
    [
        "eq",
        "ne",
        "le",
        "lt",
        "ge",
        "gt",
        "add",
        "radd",
        "sub",
        "rsub",
        "mul",
        "rmul",
        "truediv",
        "rtruediv",
        "floordiv",
        "rfloordiv",
        "pow",
        "rpow",
        "mod",
        "divmod",
    ],
)
def test_generated_op_names(opname, index):
    opname = f"__{opname}__"
    method = getattr(index, opname)
    assert method.__name__ == opname


@pytest.mark.parametrize(
    "klass",
    [
        partial(CategoricalIndex, data=[1]),
        partial(DatetimeIndex, data=["2020-01-01"]),
        partial(PeriodIndex, data=["2020-01-01"]),
        partial(TimedeltaIndex, data=["1 day"]),
        partial(RangeIndex, data=range(1)),
        partial(IntervalIndex, data=[pd.Interval(0, 1)]),
        partial(Index, data=["a"], dtype=object),
        partial(MultiIndex, levels=[1], codes=[0]),
    ],
)
def test_index_subclass_constructor_wrong_kwargs(klass):
    # GH #19348
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        klass(foo="bar")


def test_deprecated_fastpath():
    msg = "[Uu]nexpected keyword argument"
    with pytest.raises(TypeError, match=msg):
        Index(np.array(["a", "b"], dtype=object), name="test", fastpath=True)

    with pytest.raises(TypeError, match=msg):
        Index(np.array([1, 2, 3], dtype="int64"), name="test", fastpath=True)

    with pytest.raises(TypeError, match=msg):
        RangeIndex(0, 5, 2, name="test", fastpath=True)

    with pytest.raises(TypeError, match=msg):
        CategoricalIndex(["a", "b", "c"], name="test", fastpath=True)


def test_shape_of_invalid_index():
    # Pre-2.0, it was possible to create "invalid" index objects backed by
    # a multi-dimensional array (see https://github.com/pandas-dev/pandas/issues/27125
    # about this). However, as long as this is not solved in general,this test ensures
    # that the returned shape is consistent with this underlying array for
    # compat with matplotlib (see https://github.com/pandas-dev/pandas/issues/27775)
    idx = Index([0, 1, 2, 3])
    with pytest.raises(ValueError, match="Multi-dimensional indexing"):
        # GH#30588 multi-dimensional indexing deprecated
        idx[:, None]


@pytest.mark.parametrize("dtype", [None, np.int64, np.uint64, np.float64])
def test_validate_1d_input(dtype):
    # GH#27125 check that we do not have >1-dimensional input
    msg = "Index data must be 1-dimensional"

    arr = np.arange(8).reshape(2, 2, 2)
    with pytest.raises(ValueError, match=msg):
        Index(arr, dtype=dtype)

    df = DataFrame(arr.reshape(4, 2))
    with pytest.raises(ValueError, match=msg):
        Index(df, dtype=dtype)

    # GH#13601 trying to assign a multi-dimensional array to an index is not allowed
    ser = Series(0, range(4))
    with pytest.raises(ValueError, match=msg):
        ser.index = np.array([[2, 3]] * 4, dtype=dtype)


@pytest.mark.parametrize(
    "klass, extra_kwargs",
    [
        [Index, {}],
        *[[lambda x: Index(x, dtype=dtyp), {}] for dtyp in tm.ALL_REAL_NUMPY_DTYPES],
        [DatetimeIndex, {}],
        [TimedeltaIndex, {}],
        [PeriodIndex, {"freq": "Y"}],
    ],
)
def test_construct_from_memoryview(klass, extra_kwargs):
    # GH 13120
    result = klass(memoryview(np.arange(2000, 2005)), **extra_kwargs)
    expected = klass(list(range(2000, 2005)), **extra_kwargs)
    tm.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize("op", [operator.lt, operator.gt])
def test_nan_comparison_same_object(op):
    # GH#47105
    idx = Index([np.nan])
    expected = np.array([False])

    result = op(idx, idx)
    tm.assert_numpy_array_equal(result, expected)

    result = op(idx, idx.copy())
    tm.assert_numpy_array_equal(result, expected)


@td.skip_if_no("pyarrow")
def test_is_monotonic_pyarrow_list_type():
    # GH 57333
    import pyarrow as pa

    idx = Index([[1], [2, 3]], dtype=pd.ArrowDtype(pa.list_(pa.int64())))
    assert not idx.is_monotonic_increasing
    assert not idx.is_monotonic_decreasing