
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\modes\basemode.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2003-2006 Gary Bishop.
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


import glob
import math
import os
import re
import sys

import pyreadline3.clipboard as clipboard
import pyreadline3.lineeditor.history as history
import pyreadline3.lineeditor.lineobj as lineobj
from pyreadline3.error import ReadlineError
from pyreadline3.keysyms.common import make_KeyPress_from_keydescr
from pyreadline3.logger import log
from pyreadline3.py3k_compat import is_callable, is_ironpython
from pyreadline3.unicode_helper import ensure_str, ensure_unicode


class BaseMode(object):
    mode = "base"

    def __init__(self, rlobj):
        self.argument = 0
        self.rlobj = rlobj
        self.exit_dispatch = {}
        self.key_dispatch = {}
        self.argument = 1
        self.prevargument = None
        self.l_buffer = lineobj.ReadLineTextBuffer("")
        self._history = history.LineHistory()
        self.completer_delims = " \t\n\"\\'`@$><=;|&{("
        self.show_all_if_ambiguous = "on"
        self.mark_directories = "on"
        self.complete_filesystem = "off"
        self.completer = None
        self.begidx = 0
        self.endidx = 0
        self.tabstop = 4
        self.startup_hook = None
        self.pre_input_hook = None
        self.first_prompt = True
        self.cursor_size = 25

        self.prompt = ">>> "

        # Paste settings
        # assumes data on clipboard is path if shorter than 300 characters and doesn't contain \t or \n
        # and replace \ with / for easier use in ipython
        self.enable_ipython_paste_for_paths = True

        # automatically convert tabseparated data to list of lists or array
        # constructors
        self.enable_ipython_paste_list_of_lists = True
        self.enable_win32_clipboard = True

        self.paste_line_buffer = []

        self._sub_modes = []

    def __repr__(self):
        return "<BaseMode>"

    def _gs(x):
        def g(self):
            return getattr(self.rlobj, x)

        def s(self, q):
            setattr(self.rlobj, x, q)

        return g, s

    def _g(x):
        def g(self):
            return getattr(self.rlobj, x)

        return g

    def _argreset(self):
        val = self.argument
        self.argument = 0
        if val == 0:
            val = 1
        return val

    argument_reset = property(_argreset)

    # used in readline
    ctrl_c_tap_time_interval = property(*_gs("ctrl_c_tap_time_interval"))
    allow_ctrl_c = property(*_gs("allow_ctrl_c"))
    _print_prompt = property(_g("_print_prompt"))
    _update_line = property(_g("_update_line"))
    console = property(_g("console"))
    prompt_begin_pos = property(_g("prompt_begin_pos"))
    prompt_end_pos = property(_g("prompt_end_pos"))

    # used in completer _completions
    # completer_delims=property(*_gs("completer_delims"))
    _bell = property(_g("_bell"))
    bell_style = property(_g("bell_style"))

    # used in emacs
    _clear_after = property(_g("_clear_after"))
    _update_prompt_pos = property(_g("_update_prompt_pos"))

    # not used in basemode or emacs

    def process_keyevent(self, keyinfo):
        raise NotImplementedError

    def readline_setup(self, prompt=""):
        self.l_buffer.selection_mark = -1
        if self.first_prompt:
            self.first_prompt = False
            if self.startup_hook:
                try:
                    self.startup_hook()
                except BaseException:
                    print("startup hook failed")
                    traceback.print_exc()

        self.l_buffer.reset_line()
        self.prompt = prompt

        if self.pre_input_hook:
            try:
                self.pre_input_hook()
            except BaseException:
                print("pre_input_hook failed")
                traceback.print_exc()
                self.pre_input_hook = None

    # ###################################

    def finalize(self):
        """Every bindable command should call this function for cleanup.
        Except those that want to set argument to a non-zero value.
        """
        self.argument = 0

    def add_history(self, text):
        self._history.add_history(lineobj.ReadLineTextBuffer(text))

    # Create key bindings:

    def rl_settings_to_string(self):
        out = ["%-20s: %s" % ("show all if ambigous", self.show_all_if_ambiguous)]
        out.append("%-20s: %s" % ("mark_directories", self.mark_directories))
        out.append("%-20s: %s" % ("bell_style", self.bell_style))
        out.append("------------- key bindings ------------")
        tablepat = "%-7s %-7s %-7s %-15s %-15s "
        out.append(tablepat % ("Control", "Meta", "Shift", "Keycode/char", "Function"))
        bindings = sorted(
            [(k[0], k[1], k[2], k[3], v.__name__) for k, v in self.key_dispatch.items()]
        )
        for key in bindings:
            out.append(tablepat % (key))
        return out

    def _bind_key(self, key, func):
        """setup the mapping from key to call the function."""
        if not is_callable(func):
            print("Trying to bind non method to keystroke:%s,%s" % (key, func))
            raise ReadlineError(
                "Trying to bind non method to keystroke:%s,%s,%s,%s"
                % (key, func, type(func), type(self._bind_key))
            )
        keyinfo = make_KeyPress_from_keydescr(key.lower()).tuple()
        log(">>>%s -> %s<<<" % (keyinfo, func.__name__))
        self.key_dispatch[keyinfo] = func

    def _bind_exit_key(self, key):
        """setup the mapping from key to call the function."""
        keyinfo = make_KeyPress_from_keydescr(key.lower()).tuple()
        self.exit_dispatch[keyinfo] = None

    def init_editing_mode(self, e):  # (C-e)
        """When in vi command mode, this causes a switch to emacs editing
        mode."""

        raise NotImplementedError

    # completion commands

    def _get_completions(self):
        """Return a list of possible completions for the string ending at the point.
        Also set begidx and endidx in the process."""
        completions = []
        self.begidx = self.l_buffer.point
        self.endidx = self.l_buffer.point
        buf = self.l_buffer.line_buffer
        if self.completer:
            # get the string to complete
            while self.begidx > 0:
                self.begidx -= 1
                if buf[self.begidx] in self.completer_delims:
                    self.begidx += 1
                    break
            text = ensure_str("".join(buf[self.begidx : self.endidx]))
            log('complete text="%s"' % ensure_unicode(text))
            i = 0
            while True:
                try:
                    r = self.completer(ensure_unicode(text), i)
                except IndexError:
                    break
                i += 1
                if r is None:
                    break
                elif r and r not in completions:
                    completions.append(r)
                else:
                    pass
            log("text completions=<%s>" % list(map(ensure_unicode, completions)))
        if (self.complete_filesystem == "on") and not completions:
            # get the filename to complete
            while self.begidx > 0:
                self.begidx -= 1
                if buf[self.begidx] in " \t\n":
                    self.begidx += 1
                    break
            text = ensure_str("".join(buf[self.begidx : self.endidx]))
            log('file complete text="%s"' % ensure_unicode(text))
            completions = list(
                map(
                    ensure_unicode,
                    glob.glob(os.path.expanduser(text) + "*".encode("ascii")),
                )
            )
            if self.mark_directories == "on":
                mc = []
                for f in completions:
                    if os.path.isdir(f):
                        mc.append(f + os.sep)
                    else:
                        mc.append(f)
                completions = mc
            log("fnames=<%s>" % list(map(ensure_unicode, completions)))
        return completions

    def _display_completions(self, completions):
        if not completions:
            return
        self.console.write("\n")
        wmax = max(map(len, completions))
        w, h = self.console.size()
        cols = max(1, int((w - 1) / (wmax + 1)))
        rows = int(math.ceil(float(len(completions)) / cols))
        for row in range(rows):
            s = ""
            for col in range(cols):
                i = col * rows + row
                if i < len(completions):
                    self.console.write(completions[i].ljust(wmax + 1))
            self.console.write("\n")
        if is_ironpython:
            self.prompt = sys.ps1
        self._print_prompt()

    def complete(self, e):  # (TAB)
        """Attempt to perform completion on the text before point. The
        actual completion performed is application-specific. The default is
        filename completion."""
        completions = self._get_completions()
        if completions:
            cprefix = commonprefix(completions)
            if len(cprefix) > 0:
                rep = [c for c in cprefix]
                point = self.l_buffer.point
                self.l_buffer[self.begidx : self.endidx] = rep
                self.l_buffer.point = point + len(rep) - (self.endidx - self.begidx)
            if len(completions) > 1:
                if self.show_all_if_ambiguous == "on":
                    self._display_completions(completions)
                else:
                    self._bell()
        else:
            self._bell()
        self.finalize()

    def possible_completions(self, e):  # (M-?)
        """List the possible completions of the text before point."""
        completions = self._get_completions()
        self._display_completions(completions)
        self.finalize()

    def insert_completions(self, e):  # (M-*)
        """Insert all completions of the text before point that would have
        been generated by possible-completions."""
        completions = self._get_completions()
        b = self.begidx
        e = self.endidx
        for comp in completions:
            rep = [c for c in comp]
            rep.append(" ")
            self.l_buffer[b:e] = rep
            b += len(rep)
            e = b
        self.line_cursor = b
        self.finalize()

    def menu_complete(self, e):  # ()
        """Similar to complete, but replaces the word to be completed with a
        single match from the list of possible completions. Repeated
        execution of menu-complete steps through the list of possible
        completions, inserting each match in turn. At the end of the list of
        completions, the bell is rung (subject to the setting of bell-style)
        and the original text is restored. An argument of n moves n
        positions forward in the list of matches; a negative argument may be
        used to move backward through the list. This command is intended to
        be bound to TAB, but is unbound by default."""
        self.finalize()

    # Methods below here are bindable emacs functions

    def insert_text(self, string):
        """Insert text into the command line."""
        self.l_buffer.insert_text(string, self.argument_reset)
        self.finalize()

    def beginning_of_line(self, e):  # (C-a)
        """Move to the start of the current line."""
        self.l_buffer.beginning_of_line()
        self.finalize()

    def end_of_line(self, e):  # (C-e)
        """Move to the end of the line."""
        self.l_buffer.end_of_line()
        self.finalize()

    def forward_char(self, e):  # (C-f)
        """Move forward a character."""
        self.l_buffer.forward_char(self.argument_reset)
        self.finalize()

    def backward_char(self, e):  # (C-b)
        """Move back a character."""
        self.l_buffer.backward_char(self.argument_reset)
        self.finalize()

    def forward_word(self, e):  # (M-f)
        """Move forward to the end of the next word. Words are composed of
        letters and digits."""
        self.l_buffer.forward_word(self.argument_reset)
        self.finalize()

    def backward_word(self, e):  # (M-b)
        """Move back to the start of the current or previous word. Words are
        composed of letters and digits."""
        self.l_buffer.backward_word(self.argument_reset)
        self.finalize()

    def forward_word_end(self, e):  # ()
        """Move forward to the end of the next word. Words are composed of
        letters and digits."""
        self.l_buffer.forward_word_end(self.argument_reset)
        self.finalize()

    def backward_word_end(self, e):  # ()
        """Move forward to the end of the next word. Words are composed of
        letters and digits."""
        self.l_buffer.backward_word_end(self.argument_reset)
        self.finalize()

    # Movement with extend selection
    def beginning_of_line_extend_selection(self, e):
        """Move to the start of the current line."""
        self.l_buffer.beginning_of_line_extend_selection()
        self.finalize()

    def end_of_line_extend_selection(self, e):
        """Move to the end of the line."""
        self.l_buffer.end_of_line_extend_selection()
        self.finalize()

    def forward_char_extend_selection(self, e):
        """Move forward a character."""
        self.l_buffer.forward_char_extend_selection(self.argument_reset)
        self.finalize()

    def backward_char_extend_selection(self, e):
        """Move back a character."""
        self.l_buffer.backward_char_extend_selection(self.argument_reset)
        self.finalize()

    def forward_word_extend_selection(self, e):
        """Move forward to the end of the next word. Words are composed of
        letters and digits."""
        self.l_buffer.forward_word_extend_selection(self.argument_reset)
        self.finalize()

    def backward_word_extend_selection(self, e):
        """Move back to the start of the current or previous word. Words are
        composed of letters and digits."""
        self.l_buffer.backward_word_extend_selection(self.argument_reset)
        self.finalize()

    def forward_word_end_extend_selection(self, e):
        """Move forward to the end of the next word. Words are composed of
        letters and digits."""
        self.l_buffer.forward_word_end_extend_selection(self.argument_reset)
        self.finalize()

    def backward_word_end_extend_selection(self, e):
        """Move forward to the end of the next word. Words are composed of
        letters and digits."""
        self.l_buffer.forward_word_end_extend_selection(self.argument_reset)
        self.finalize()

    # Change case

    def upcase_word(self, e):  # (M-u)
        """Uppercase the current (or following) word. With a negative
        argument, uppercase the previous word, but do not move the cursor."""
        self.l_buffer.upcase_word()
        self.finalize()

    def downcase_word(self, e):  # (M-l)
        """Lowercase the current (or following) word. With a negative
        argument, lowercase the previous word, but do not move the cursor."""
        self.l_buffer.downcase_word()
        self.finalize()

    def capitalize_word(self, e):  # (M-c)
        """Capitalize the current (or following) word. With a negative
        argument, capitalize the previous word, but do not move the cursor."""
        self.l_buffer.capitalize_word()
        self.finalize()

    # #######

    def clear_screen(self, e):  # (C-l)
        """Clear the screen and redraw the current line, leaving the current
        line at the top of the screen."""
        self.console.page()
        self.finalize()

    def redraw_current_line(self, e):  # ()
        """Refresh the current line. By default, this is unbound."""
        self.finalize()

    def accept_line(self, e):  # (Newline or Return)
        """Accept the line regardless of where the cursor is. If this line
        is non-empty, it may be added to the history list for future recall
        with add_history(). If this line is a modified history line, the
        history line is restored to its original state."""
        self.finalize()
        return True

    def delete_char(self, e):  # (C-d)
        """Delete the character at point. If point is at the beginning of
        the line, there are no characters in the line, and the last
        character typed was not bound to delete-char, then return EOF."""
        self.l_buffer.delete_char(self.argument_reset)
        self.finalize()

    def backward_delete_char(self, e):  # (Rubout)
        """Delete the character behind the cursor. A numeric argument means
        to kill the characters instead of deleting them."""
        self.l_buffer.backward_delete_char(self.argument_reset)
        self.finalize()

    def backward_delete_word(self, e):  # (Control-Rubout)
        """Delete the character behind the cursor. A numeric argument means
        to kill the characters instead of deleting them."""
        self.l_buffer.backward_delete_word(self.argument_reset)
        self.finalize()

    def forward_delete_word(self, e):  # (Control-Delete)
        """Delete the character behind the cursor. A numeric argument means
        to kill the characters instead of deleting them."""
        self.l_buffer.forward_delete_word(self.argument_reset)
        self.finalize()

    def delete_horizontal_space(self, e):  # ()
        """Delete all spaces and tabs around point. By default, this is unbound."""
        self.l_buffer.delete_horizontal_space()
        self.finalize()

    def self_insert(self, e):  # (a, b, A, 1, !, ...)
        """Insert yourself."""
        if (
            e.char and ord(e.char) != 0
        ):  # don't insert null character in buffer, can happen with dead keys.
            self.insert_text(e.char)
        self.finalize()

    # Paste from clipboard

    def paste(self, e):
        """Paste windows clipboard.
        Assume single line strip other lines and end of line markers and trailing spaces
        """  # (Control-v)
        if self.enable_win32_clipboard:
            txt = clipboard.get_clipboard_text_and_convert(False)
            txt = txt.split("\n")[0].strip("\r").strip("\n")
            log("paste: >%s<" % list(map(ord, txt)))
            self.insert_text(txt)
        self.finalize()

    def paste_mulitline_code(self, e):
        """Paste windows clipboard as multiline code.
        Removes any empty lines in the code"""
        reg = re.compile("\r?\n")
        if self.enable_win32_clipboard:
            txt = clipboard.get_clipboard_text_and_convert(False)
            t = reg.split(txt)
            t = [row for row in t if row.strip() != ""]  # remove empty lines
            if t != [""]:
                self.insert_text(t[0])
                self.add_history(self.l_buffer.copy())
                self.paste_line_buffer = t[1:]
                log("multi: >%s<" % self.paste_line_buffer)
                return True
            else:
                return False
        self.finalize()

    def ipython_paste(self, e):
        """Paste windows clipboard. If enable_ipython_paste_list_of_lists is
        True then try to convert tabseparated data to repr of list of lists or
        repr of array.
        If enable_ipython_paste_for_paths==True then change \\ to / and spaces
        to \\space"""
        if self.enable_win32_clipboard:
            txt = clipboard.get_clipboard_text_and_convert(
                self.enable_ipython_paste_list_of_lists
            )
            if self.enable_ipython_paste_for_paths:
                if len(txt) < 300 and ("\t" not in txt) and ("\n" not in txt):
                    txt = txt.replace("\\", "/").replace(" ", r"\ ")
            self.insert_text(txt)
        self.finalize()

    def copy_region_to_clipboard(self, e):  # ()
        """Copy the text in the region to the windows clipboard."""
        self.l_buffer.copy_region_to_clipboard()
        self.finalize()

    def copy_selection_to_clipboard(self, e):  # ()
        """Copy the text in the region to the windows clipboard."""
        self.l_buffer.copy_selection_to_clipboard()
        self.finalize()

    def cut_selection_to_clipboard(self, e):  # ()
        """Copy the text in the region to the windows clipboard."""
        self.l_buffer.cut_selection_to_clipboard()
        self.finalize()

    def dump_functions(self, e):  # ()
        """Print all of the functions and their key bindings to the Readline
        output stream. If a numeric argument is supplied, the output is
        formatted in such a way that it can be made part of an inputrc
        file. This command is unbound by default."""
        print()
        txt = "\n".join(self.rl_settings_to_string())
        print(txt)
        self._print_prompt()
        self.finalize()


def commonprefix(m):
    "Given a list of pathnames, returns the longest common leading component"
    if not m:
        return ""
    prefix = m[0]
    for item in m:
        for i in range(len(prefix)):
            if prefix[: i + 1].lower() != item[: i + 1].lower():
                prefix = prefix[:i]
                if i == 0:
                    return ""
                break
    return prefix

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\plotting\_matplotlib\hist.py ===
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    final,
)

import numpy as np

from pandas.core.dtypes.common import (
    is_integer,
    is_list_like,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
)
from pandas.core.dtypes.missing import (
    isna,
    remove_na_arraylike,
)

from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
    LinePlot,
    MPLPlot,
)
from pandas.plotting._matplotlib.groupby import (
    create_iter_data_given_by,
    reformat_hist_y_given_by,
)
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.tools import (
    create_subplots,
    flatten_axes,
    maybe_adjust_figure,
    set_ticks_props,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from pandas._typing import PlottingOrientation

    from pandas import (
        DataFrame,
        Series,
    )


class HistPlot(LinePlot):
    @property
    def _kind(self) -> Literal["hist", "kde"]:
        return "hist"

    def __init__(
        self,
        data,
        bins: int | np.ndarray | list[np.ndarray] = 10,
        bottom: int | np.ndarray = 0,
        *,
        range=None,
        weights=None,
        **kwargs,
    ) -> None:
        if is_list_like(bottom):
            bottom = np.array(bottom)
        self.bottom = bottom

        self._bin_range = range
        self.weights = weights

        self.xlabel = kwargs.get("xlabel")
        self.ylabel = kwargs.get("ylabel")
        # Do not call LinePlot.__init__ which may fill nan
        MPLPlot.__init__(self, data, **kwargs)  # pylint: disable=non-parent-init-called

        self.bins = self._adjust_bins(bins)

    def _adjust_bins(self, bins: int | np.ndarray | list[np.ndarray]):
        if is_integer(bins):
            if self.by is not None:
                by_modified = unpack_single_str_list(self.by)
                grouped = self.data.groupby(by_modified)[self.columns]
                bins = [self._calculate_bins(group, bins) for key, group in grouped]
            else:
                bins = self._calculate_bins(self.data, bins)
        return bins

    def _calculate_bins(self, data: Series | DataFrame, bins) -> np.ndarray:
        """Calculate bins given data"""
        nd_values = data.infer_objects(copy=False)._get_numeric_data()
        values = np.ravel(nd_values)
        values = values[~isna(values)]

        hist, bins = np.histogram(values, bins=bins, range=self._bin_range)
        return bins

    # error: Signature of "_plot" incompatible with supertype "LinePlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        y: np.ndarray,
        style=None,
        bottom: int | np.ndarray = 0,
        column_num: int = 0,
        stacking_id=None,
        *,
        bins,
        **kwds,
    ):
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(bins) - 1)

        base = np.zeros(len(bins) - 1)
        bottom = bottom + cls._get_stacked_values(ax, stacking_id, base, kwds["label"])
        # ignore style
        n, bins, patches = ax.hist(y, bins=bins, bottom=bottom, **kwds)
        cls._update_stacker(ax, stacking_id, n)
        return patches

    def _make_plot(self, fig: Figure) -> None:
        colors = self._get_colors()
        stacking_id = self._get_stacking_id()

        # Re-create iterated data if `by` is assigned by users
        data = (
            create_iter_data_given_by(self.data, self._kind)
            if self.by is not None
            else self.data
        )

        # error: Argument "data" to "_iter_data" of "MPLPlot" has incompatible
        # type "object"; expected "DataFrame | dict[Hashable, Series | DataFrame]"
        for i, (label, y) in enumerate(self._iter_data(data=data)):  # type: ignore[arg-type]
            ax = self._get_ax(i)

            kwds = self.kwds.copy()
            if self.color is not None:
                kwds["color"] = self.color

            label = pprint_thing(label)
            label = self._mark_right_label(label, index=i)
            kwds["label"] = label

            style, kwds = self._apply_style_colors(colors, kwds, i, label)
            if style is not None:
                kwds["style"] = style

            self._make_plot_keywords(kwds, y)

            # the bins is multi-dimension array now and each plot need only 1-d and
            # when by is applied, label should be columns that are grouped
            if self.by is not None:
                kwds["bins"] = kwds["bins"][i]
                kwds["label"] = self.columns
                kwds.pop("color")

            if self.weights is not None:
                kwds["weights"] = type(self)._get_column_weights(self.weights, i, y)

            y = reformat_hist_y_given_by(y, self.by)

            artists = self._plot(ax, y, column_num=i, stacking_id=stacking_id, **kwds)

            # when by is applied, show title for subplots to know which group it is
            if self.by is not None:
                ax.set_title(pprint_thing(label))

            self._append_legend_handles_labels(artists[0], label)

    def _make_plot_keywords(self, kwds: dict[str, Any], y: np.ndarray) -> None:
        """merge BoxPlot/KdePlot properties to passed kwds"""
        # y is required for KdePlot
        kwds["bottom"] = self.bottom
        kwds["bins"] = self.bins

    @final
    @staticmethod
    def _get_column_weights(weights, i: int, y):
        # We allow weights to be a multi-dimensional array, e.g. a (10, 2) array,
        # and each sub-array (10,) will be called in each iteration. If users only
        # provide 1D array, we assume the same weights is used for all iterations
        if weights is not None:
            if np.ndim(weights) != 1 and np.shape(weights)[-1] != 1:
                try:
                    weights = weights[:, i]
                except IndexError as err:
                    raise ValueError(
                        "weights must have the same shape as data, "
                        "or be a single column"
                    ) from err
            weights = weights[~isna(y)]
        return weights

    def _post_plot_logic(self, ax: Axes, data) -> None:
        if self.orientation == "horizontal":
            # error: Argument 1 to "set_xlabel" of "_AxesBase" has incompatible
            # type "Hashable"; expected "str"
            ax.set_xlabel(
                "Frequency"
                if self.xlabel is None
                else self.xlabel  # type: ignore[arg-type]
            )
            ax.set_ylabel(self.ylabel)  # type: ignore[arg-type]
        else:
            ax.set_xlabel(self.xlabel)  # type: ignore[arg-type]
            ax.set_ylabel(
                "Frequency"
                if self.ylabel is None
                else self.ylabel  # type: ignore[arg-type]
            )

    @property
    def orientation(self) -> PlottingOrientation:
        if self.kwds.get("orientation", None) == "horizontal":
            return "horizontal"
        else:
            return "vertical"


class KdePlot(HistPlot):
    @property
    def _kind(self) -> Literal["kde"]:
        return "kde"

    @property
    def orientation(self) -> Literal["vertical"]:
        return "vertical"

    def __init__(
        self, data, bw_method=None, ind=None, *, weights=None, **kwargs
    ) -> None:
        # Do not call LinePlot.__init__ which may fill nan
        MPLPlot.__init__(self, data, **kwargs)  # pylint: disable=non-parent-init-called
        self.bw_method = bw_method
        self.ind = ind
        self.weights = weights

    @staticmethod
    def _get_ind(y: np.ndarray, ind):
        if ind is None:
            # np.nanmax() and np.nanmin() ignores the missing values
            sample_range = np.nanmax(y) - np.nanmin(y)
            ind = np.linspace(
                np.nanmin(y) - 0.5 * sample_range,
                np.nanmax(y) + 0.5 * sample_range,
                1000,
            )
        elif is_integer(ind):
            sample_range = np.nanmax(y) - np.nanmin(y)
            ind = np.linspace(
                np.nanmin(y) - 0.5 * sample_range,
                np.nanmax(y) + 0.5 * sample_range,
                ind,
            )
        return ind

    @classmethod
    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    def _plot(  #  type: ignore[override]
        cls,
        ax: Axes,
        y: np.ndarray,
        style=None,
        bw_method=None,
        ind=None,
        column_num=None,
        stacking_id: int | None = None,
        **kwds,
    ):
        from scipy.stats import gaussian_kde

        y = remove_na_arraylike(y)
        gkde = gaussian_kde(y, bw_method=bw_method)

        y = gkde.evaluate(ind)
        lines = MPLPlot._plot(ax, ind, y, style=style, **kwds)
        return lines

    def _make_plot_keywords(self, kwds: dict[str, Any], y: np.ndarray) -> None:
        kwds["bw_method"] = self.bw_method
        kwds["ind"] = type(self)._get_ind(y, ind=self.ind)

    def _post_plot_logic(self, ax: Axes, data) -> None:
        ax.set_ylabel("Density")


def _grouped_plot(
    plotf,
    data: Series | DataFrame,
    column=None,
    by=None,
    numeric_only: bool = True,
    figsize: tuple[float, float] | None = None,
    sharex: bool = True,
    sharey: bool = True,
    layout=None,
    rot: float = 0,
    ax=None,
    **kwargs,
):
    # error: Non-overlapping equality check (left operand type: "Optional[Tuple[float,
    # float]]", right operand type: "Literal['default']")
    if figsize == "default":  # type: ignore[comparison-overlap]
        # allowed to specify mpl default with 'default'
        raise ValueError(
            "figsize='default' is no longer supported. "
            "Specify figure size by tuple instead"
        )

    grouped = data.groupby(by)
    if column is not None:
        grouped = grouped[column]

    naxes = len(grouped)
    fig, axes = create_subplots(
        naxes=naxes, figsize=figsize, sharex=sharex, sharey=sharey, ax=ax, layout=layout
    )

    _axes = flatten_axes(axes)

    for i, (key, group) in enumerate(grouped):
        ax = _axes[i]
        if numeric_only and isinstance(group, ABCDataFrame):
            group = group._get_numeric_data()
        plotf(group, ax, **kwargs)
        ax.set_title(pprint_thing(key))

    return fig, axes


def _grouped_hist(
    data: Series | DataFrame,
    column=None,
    by=None,
    ax=None,
    bins: int = 50,
    figsize: tuple[float, float] | None = None,
    layout=None,
    sharex: bool = False,
    sharey: bool = False,
    rot: float = 90,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot=None,
    ylabelsize: int | None = None,
    yrot=None,
    legend: bool = False,
    **kwargs,
):
    """
    Grouped histogram

    Parameters
    ----------
    data : Series/DataFrame
    column : object, optional
    by : object, optional
    ax : axes, optional
    bins : int, default 50
    figsize : tuple, optional
    layout : optional
    sharex : bool, default False
    sharey : bool, default False
    rot : float, default 90
    grid : bool, default True
    legend: : bool, default False
    kwargs : dict, keyword arguments passed to matplotlib.Axes.hist

    Returns
    -------
    collection of Matplotlib Axes
    """
    if legend:
        assert "label" not in kwargs
        if data.ndim == 1:
            kwargs["label"] = data.name
        elif column is None:
            kwargs["label"] = data.columns
        else:
            kwargs["label"] = column

    def plot_group(group, ax) -> None:
        ax.hist(group.dropna().values, bins=bins, **kwargs)
        if legend:
            ax.legend()

    if xrot is None:
        xrot = rot

    fig, axes = _grouped_plot(
        plot_group,
        data,
        column=column,
        by=by,
        sharex=sharex,
        sharey=sharey,
        ax=ax,
        figsize=figsize,
        layout=layout,
        rot=rot,
    )

    set_ticks_props(
        axes, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot
    )

    maybe_adjust_figure(
        fig, bottom=0.15, top=0.9, left=0.1, right=0.9, hspace=0.5, wspace=0.3
    )
    return axes


def hist_series(
    self: Series,
    by=None,
    ax=None,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot=None,
    ylabelsize: int | None = None,
    yrot=None,
    figsize: tuple[float, float] | None = None,
    bins: int = 10,
    legend: bool = False,
    **kwds,
):
    import matplotlib.pyplot as plt

    if legend and "label" in kwds:
        raise ValueError("Cannot use both legend and label")

    if by is None:
        if kwds.get("layout", None) is not None:
            raise ValueError("The 'layout' keyword is not supported when 'by' is None")
        # hack until the plotting interface is a bit more unified
        fig = kwds.pop(
            "figure", plt.gcf() if plt.get_fignums() else plt.figure(figsize=figsize)
        )
        if figsize is not None and tuple(figsize) != tuple(fig.get_size_inches()):
            fig.set_size_inches(*figsize, forward=True)
        if ax is None:
            ax = fig.gca()
        elif ax.get_figure() != fig:
            raise AssertionError("passed axis not bound to passed figure")
        values = self.dropna().values
        if legend:
            kwds["label"] = self.name
        ax.hist(values, bins=bins, **kwds)
        if legend:
            ax.legend()
        ax.grid(grid)
        axes = np.array([ax])

        # error: Argument 1 to "set_ticks_props" has incompatible type "ndarray[Any,
        # dtype[Any]]"; expected "Axes | Sequence[Axes]"
        set_ticks_props(
            axes,  # type: ignore[arg-type]
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
        )

    else:
        if "figure" in kwds:
            raise ValueError(
                "Cannot pass 'figure' when using the "
                "'by' argument, since a new 'Figure' instance will be created"
            )
        axes = _grouped_hist(
            self,
            by=by,
            ax=ax,
            grid=grid,
            figsize=figsize,
            bins=bins,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            legend=legend,
            **kwds,
        )

    if hasattr(axes, "ndim"):
        if axes.ndim == 1 and len(axes) == 1:
            return axes[0]
    return axes


def hist_frame(
    data: DataFrame,
    column=None,
    by=None,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot=None,
    ylabelsize: int | None = None,
    yrot=None,
    ax=None,
    sharex: bool = False,
    sharey: bool = False,
    figsize: tuple[float, float] | None = None,
    layout=None,
    bins: int = 10,
    legend: bool = False,
    **kwds,
):
    if legend and "label" in kwds:
        raise ValueError("Cannot use both legend and label")
    if by is not None:
        axes = _grouped_hist(
            data,
            column=column,
            by=by,
            ax=ax,
            grid=grid,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            layout=layout,
            bins=bins,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            legend=legend,
            **kwds,
        )
        return axes

    if column is not None:
        if not isinstance(column, (list, np.ndarray, ABCIndex)):
            column = [column]
        data = data[column]
    # GH32590
    data = data.select_dtypes(
        include=(np.number, "datetime64", "datetimetz"), exclude="timedelta"
    )
    naxes = len(data.columns)

    if naxes == 0:
        raise ValueError(
            "hist method requires numerical or datetime columns, nothing to plot."
        )

    fig, axes = create_subplots(
        naxes=naxes,
        ax=ax,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        layout=layout,
    )
    _axes = flatten_axes(axes)

    can_set_label = "label" not in kwds

    for i, col in enumerate(data.columns):
        ax = _axes[i]
        if legend and can_set_label:
            kwds["label"] = col
        ax.hist(data[col].dropna().values, bins=bins, **kwds)
        ax.set_title(col)
        ax.grid(grid)
        if legend:
            ax.legend()

    set_ticks_props(
        axes, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot
    )
    maybe_adjust_figure(fig, wspace=0.3, hspace=0.3)

    return axes

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\pool\impl.py ===
# pool/impl.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php


"""Pool implementation classes.

"""
from __future__ import annotations

import threading
import traceback
import typing
from typing import Any
from typing import cast
from typing import List
from typing import Optional
from typing import Set
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import weakref

from .base import _AsyncConnDialect
from .base import _ConnectionFairy
from .base import _ConnectionRecord
from .base import _CreatorFnType
from .base import _CreatorWRecFnType
from .base import ConnectionPoolEntry
from .base import Pool
from .base import PoolProxiedConnection
from .. import exc
from .. import util
from ..util import chop_traceback
from ..util import queue as sqla_queue
from ..util.typing import Literal

if typing.TYPE_CHECKING:
    from ..engine.interfaces import DBAPIConnection


class QueuePool(Pool):
    """A :class:`_pool.Pool`
    that imposes a limit on the number of open connections.

    :class:`.QueuePool` is the default pooling implementation used for
    all :class:`_engine.Engine` objects other than SQLite with a ``:memory:``
    database.

    The :class:`.QueuePool` class **is not compatible** with asyncio and
    :func:`_asyncio.create_async_engine`.  The
    :class:`.AsyncAdaptedQueuePool` class is used automatically when
    using :func:`_asyncio.create_async_engine`, if no other kind of pool
    is specified.

    .. seealso::

        :class:`.AsyncAdaptedQueuePool`

    """

    _is_asyncio = False  # type: ignore[assignment]

    _queue_class: Type[sqla_queue.QueueCommon[ConnectionPoolEntry]] = (
        sqla_queue.Queue
    )

    _pool: sqla_queue.QueueCommon[ConnectionPoolEntry]

    def __init__(
        self,
        creator: Union[_CreatorFnType, _CreatorWRecFnType],
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 30.0,
        use_lifo: bool = False,
        **kw: Any,
    ):
        r"""
        Construct a QueuePool.

        :param creator: a callable function that returns a DB-API
          connection object, same as that of :paramref:`_pool.Pool.creator`.

        :param pool_size: The size of the pool to be maintained,
          defaults to 5. This is the largest number of connections that
          will be kept persistently in the pool. Note that the pool
          begins with no connections; once this number of connections
          is requested, that number of connections will remain.
          ``pool_size`` can be set to 0 to indicate no size limit; to
          disable pooling, use a :class:`~sqlalchemy.pool.NullPool`
          instead.

        :param max_overflow: The maximum overflow size of the
          pool. When the number of checked-out connections reaches the
          size set in pool_size, additional connections will be
          returned up to this limit. When those additional connections
          are returned to the pool, they are disconnected and
          discarded. It follows then that the total number of
          simultaneous connections the pool will allow is pool_size +
          `max_overflow`, and the total number of "sleeping"
          connections the pool will allow is pool_size. `max_overflow`
          can be set to -1 to indicate no overflow limit; no limit
          will be placed on the total number of concurrent
          connections. Defaults to 10.

        :param timeout: The number of seconds to wait before giving up
          on returning a connection. Defaults to 30.0. This can be a float
          but is subject to the limitations of Python time functions which
          may not be reliable in the tens of milliseconds.

        :param use_lifo: use LIFO (last-in-first-out) when retrieving
          connections instead of FIFO (first-in-first-out). Using LIFO, a
          server-side timeout scheme can reduce the number of connections used
          during non-peak periods of use.   When planning for server-side
          timeouts, ensure that a recycle or pre-ping strategy is in use to
          gracefully handle stale connections.

          .. versionadded:: 1.3

          .. seealso::

            :ref:`pool_use_lifo`

            :ref:`pool_disconnects`

        :param \**kw: Other keyword arguments including
          :paramref:`_pool.Pool.recycle`, :paramref:`_pool.Pool.echo`,
          :paramref:`_pool.Pool.reset_on_return` and others are passed to the
          :class:`_pool.Pool` constructor.

        """

        Pool.__init__(self, creator, **kw)
        self._pool = self._queue_class(pool_size, use_lifo=use_lifo)
        self._overflow = 0 - pool_size
        self._max_overflow = -1 if pool_size == 0 else max_overflow
        self._timeout = timeout
        self._overflow_lock = threading.Lock()

    def _do_return_conn(self, record: ConnectionPoolEntry) -> None:
        try:
            self._pool.put(record, False)
        except sqla_queue.Full:
            try:
                record.close()
            finally:
                self._dec_overflow()

    def _do_get(self) -> ConnectionPoolEntry:
        use_overflow = self._max_overflow > -1

        wait = use_overflow and self._overflow >= self._max_overflow
        try:
            return self._pool.get(wait, self._timeout)
        except sqla_queue.Empty:
            # don't do things inside of "except Empty", because when we say
            # we timed out or can't connect and raise, Python 3 tells
            # people the real error is queue.Empty which it isn't.
            pass
        if use_overflow and self._overflow >= self._max_overflow:
            if not wait:
                return self._do_get()
            else:
                raise exc.TimeoutError(
                    "QueuePool limit of size %d overflow %d reached, "
                    "connection timed out, timeout %0.2f"
                    % (self.size(), self.overflow(), self._timeout),
                    code="3o7r",
                )

        if self._inc_overflow():
            try:
                return self._create_connection()
            except:
                with util.safe_reraise():
                    self._dec_overflow()
                raise
        else:
            return self._do_get()

    def _inc_overflow(self) -> bool:
        if self._max_overflow == -1:
            self._overflow += 1
            return True
        with self._overflow_lock:
            if self._overflow < self._max_overflow:
                self._overflow += 1
                return True
            else:
                return False

    def _dec_overflow(self) -> Literal[True]:
        if self._max_overflow == -1:
            self._overflow -= 1
            return True
        with self._overflow_lock:
            self._overflow -= 1
            return True

    def recreate(self) -> QueuePool:
        self.logger.info("Pool recreating")
        return self.__class__(
            self._creator,
            pool_size=self._pool.maxsize,
            max_overflow=self._max_overflow,
            pre_ping=self._pre_ping,
            use_lifo=self._pool.use_lifo,
            timeout=self._timeout,
            recycle=self._recycle,
            echo=self.echo,
            logging_name=self._orig_logging_name,
            reset_on_return=self._reset_on_return,
            _dispatch=self.dispatch,
            dialect=self._dialect,
        )

    def dispose(self) -> None:
        while True:
            try:
                conn = self._pool.get(False)
                conn.close()
            except sqla_queue.Empty:
                break

        self._overflow = 0 - self.size()
        self.logger.info("Pool disposed. %s", self.status())

    def status(self) -> str:
        return (
            "Pool size: %d  Connections in pool: %d "
            "Current Overflow: %d Current Checked out "
            "connections: %d"
            % (
                self.size(),
                self.checkedin(),
                self.overflow(),
                self.checkedout(),
            )
        )

    def size(self) -> int:
        return self._pool.maxsize

    def timeout(self) -> float:
        return self._timeout

    def checkedin(self) -> int:
        return self._pool.qsize()

    def overflow(self) -> int:
        return self._overflow if self._pool.maxsize else 0

    def checkedout(self) -> int:
        return self._pool.maxsize - self._pool.qsize() + self._overflow


class AsyncAdaptedQueuePool(QueuePool):
    """An asyncio-compatible version of :class:`.QueuePool`.

    This pool is used by default when using :class:`.AsyncEngine` engines that
    were generated from :func:`_asyncio.create_async_engine`.   It uses an
    asyncio-compatible queue implementation that does not use
    ``threading.Lock``.

    The arguments and operation of :class:`.AsyncAdaptedQueuePool` are
    otherwise identical to that of :class:`.QueuePool`.

    """

    _is_asyncio = True  # type: ignore[assignment]
    _queue_class: Type[sqla_queue.QueueCommon[ConnectionPoolEntry]] = (
        sqla_queue.AsyncAdaptedQueue
    )

    _dialect = _AsyncConnDialect()


class FallbackAsyncAdaptedQueuePool(AsyncAdaptedQueuePool):
    _queue_class = sqla_queue.FallbackAsyncAdaptedQueue


class NullPool(Pool):
    """A Pool which does not pool connections.

    Instead it literally opens and closes the underlying DB-API connection
    per each connection open/close.

    Reconnect-related functions such as ``recycle`` and connection
    invalidation are not supported by this Pool implementation, since
    no connections are held persistently.

    The :class:`.NullPool` class **is compatible** with asyncio and
    :func:`_asyncio.create_async_engine`.

    """

    def status(self) -> str:
        return "NullPool"

    def _do_return_conn(self, record: ConnectionPoolEntry) -> None:
        record.close()

    def _do_get(self) -> ConnectionPoolEntry:
        return self._create_connection()

    def recreate(self) -> NullPool:
        self.logger.info("Pool recreating")

        return self.__class__(
            self._creator,
            recycle=self._recycle,
            echo=self.echo,
            logging_name=self._orig_logging_name,
            reset_on_return=self._reset_on_return,
            pre_ping=self._pre_ping,
            _dispatch=self.dispatch,
            dialect=self._dialect,
        )

    def dispose(self) -> None:
        pass


class SingletonThreadPool(Pool):
    """A Pool that maintains one connection per thread.

    Maintains one connection per each thread, never moving a connection to a
    thread other than the one which it was created in.

    .. warning::  the :class:`.SingletonThreadPool` will call ``.close()``
       on arbitrary connections that exist beyond the size setting of
       ``pool_size``, e.g. if more unique **thread identities**
       than what ``pool_size`` states are used.   This cleanup is
       non-deterministic and not sensitive to whether or not the connections
       linked to those thread identities are currently in use.

       :class:`.SingletonThreadPool` may be improved in a future release,
       however in its current status it is generally used only for test
       scenarios using a SQLite ``:memory:`` database and is not recommended
       for production use.

    The :class:`.SingletonThreadPool` class **is not compatible** with asyncio
    and :func:`_asyncio.create_async_engine`.


    Options are the same as those of :class:`_pool.Pool`, as well as:

    :param pool_size: The number of threads in which to maintain connections
        at once.  Defaults to five.

    :class:`.SingletonThreadPool` is used by the SQLite dialect
    automatically when a memory-based database is used.
    See :ref:`sqlite_toplevel`.

    """

    _is_asyncio = False  # type: ignore[assignment]

    def __init__(
        self,
        creator: Union[_CreatorFnType, _CreatorWRecFnType],
        pool_size: int = 5,
        **kw: Any,
    ):
        Pool.__init__(self, creator, **kw)
        self._conn = threading.local()
        self._fairy = threading.local()
        self._all_conns: Set[ConnectionPoolEntry] = set()
        self.size = pool_size

    def recreate(self) -> SingletonThreadPool:
        self.logger.info("Pool recreating")
        return self.__class__(
            self._creator,
            pool_size=self.size,
            recycle=self._recycle,
            echo=self.echo,
            pre_ping=self._pre_ping,
            logging_name=self._orig_logging_name,
            reset_on_return=self._reset_on_return,
            _dispatch=self.dispatch,
            dialect=self._dialect,
        )

    def dispose(self) -> None:
        """Dispose of this pool."""

        for conn in self._all_conns:
            try:
                conn.close()
            except Exception:
                # pysqlite won't even let you close a conn from a thread
                # that didn't create it
                pass

        self._all_conns.clear()

    def _cleanup(self) -> None:
        while len(self._all_conns) >= self.size:
            c = self._all_conns.pop()
            c.close()

    def status(self) -> str:
        return "SingletonThreadPool id:%d size: %d" % (
            id(self),
            len(self._all_conns),
        )

    def _do_return_conn(self, record: ConnectionPoolEntry) -> None:
        try:
            del self._fairy.current
        except AttributeError:
            pass

    def _do_get(self) -> ConnectionPoolEntry:
        try:
            if TYPE_CHECKING:
                c = cast(ConnectionPoolEntry, self._conn.current())
            else:
                c = self._conn.current()
            if c:
                return c
        except AttributeError:
            pass
        c = self._create_connection()
        self._conn.current = weakref.ref(c)
        if len(self._all_conns) >= self.size:
            self._cleanup()
        self._all_conns.add(c)
        return c

    def connect(self) -> PoolProxiedConnection:
        # vendored from Pool to include the now removed use_threadlocal
        # behavior
        try:
            rec = cast(_ConnectionFairy, self._fairy.current())
        except AttributeError:
            pass
        else:
            if rec is not None:
                return rec._checkout_existing()

        return _ConnectionFairy._checkout(self, self._fairy)


class StaticPool(Pool):
    """A Pool of exactly one connection, used for all requests.

    Reconnect-related functions such as ``recycle`` and connection
    invalidation (which is also used to support auto-reconnect) are only
    partially supported right now and may not yield good results.

    The :class:`.StaticPool` class **is compatible** with asyncio and
    :func:`_asyncio.create_async_engine`.

    """

    @util.memoized_property
    def connection(self) -> _ConnectionRecord:
        return _ConnectionRecord(self)

    def status(self) -> str:
        return "StaticPool"

    def dispose(self) -> None:
        if (
            "connection" in self.__dict__
            and self.connection.dbapi_connection is not None
        ):
            self.connection.close()
            del self.__dict__["connection"]

    def recreate(self) -> StaticPool:
        self.logger.info("Pool recreating")
        return self.__class__(
            creator=self._creator,
            recycle=self._recycle,
            reset_on_return=self._reset_on_return,
            pre_ping=self._pre_ping,
            echo=self.echo,
            logging_name=self._orig_logging_name,
            _dispatch=self.dispatch,
            dialect=self._dialect,
        )

    def _transfer_from(self, other_static_pool: StaticPool) -> None:
        # used by the test suite to make a new engine / pool without
        # losing the state of an existing SQLite :memory: connection
        def creator(rec: ConnectionPoolEntry) -> DBAPIConnection:
            conn = other_static_pool.connection.dbapi_connection
            assert conn is not None
            return conn

        self._invoke_creator = creator

    def _create_connection(self) -> ConnectionPoolEntry:
        raise NotImplementedError()

    def _do_return_conn(self, record: ConnectionPoolEntry) -> None:
        pass

    def _do_get(self) -> ConnectionPoolEntry:
        rec = self.connection
        if rec._is_hard_or_soft_invalidated():
            del self.__dict__["connection"]
            rec = self.connection

        return rec


class AssertionPool(Pool):
    """A :class:`_pool.Pool` that allows at most one checked out connection at
    any given time.

    This will raise an exception if more than one connection is checked out
    at a time.  Useful for debugging code that is using more connections
    than desired.

    The :class:`.AssertionPool` class **is compatible** with asyncio and
    :func:`_asyncio.create_async_engine`.

    """

    _conn: Optional[ConnectionPoolEntry]
    _checkout_traceback: Optional[List[str]]

    def __init__(self, *args: Any, **kw: Any):
        self._conn = None
        self._checked_out = False
        self._store_traceback = kw.pop("store_traceback", True)
        self._checkout_traceback = None
        Pool.__init__(self, *args, **kw)

    def status(self) -> str:
        return "AssertionPool"

    def _do_return_conn(self, record: ConnectionPoolEntry) -> None:
        if not self._checked_out:
            raise AssertionError("connection is not checked out")
        self._checked_out = False
        assert record is self._conn

    def dispose(self) -> None:
        self._checked_out = False
        if self._conn:
            self._conn.close()

    def recreate(self) -> AssertionPool:
        self.logger.info("Pool recreating")
        return self.__class__(
            self._creator,
            echo=self.echo,
            pre_ping=self._pre_ping,
            recycle=self._recycle,
            reset_on_return=self._reset_on_return,
            logging_name=self._orig_logging_name,
            _dispatch=self.dispatch,
            dialect=self._dialect,
        )

    def _do_get(self) -> ConnectionPoolEntry:
        if self._checked_out:
            if self._checkout_traceback:
                suffix = " at:\n%s" % "".join(
                    chop_traceback(self._checkout_traceback)
                )
            else:
                suffix = ""
            raise AssertionError("connection is already checked out" + suffix)

        if not self._conn:
            self._conn = self._create_connection()

        self._checked_out = True
        if self._store_traceback:
            self._checkout_traceback = traceback.format_stack()
        return self._conn

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pytz\tzinfo.py ===
'''Base classes and helpers for building zone specific tzinfo classes'''

from datetime import datetime, timedelta, tzinfo
from bisect import bisect_right
try:
    set
except NameError:
    from sets import Set as set

import pytz
from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError

__all__ = []

_timedelta_cache = {}


def memorized_timedelta(seconds):
    '''Create only one instance of each distinct timedelta'''
    try:
        return _timedelta_cache[seconds]
    except KeyError:
        delta = timedelta(seconds=seconds)
        _timedelta_cache[seconds] = delta
        return delta


_epoch = datetime(1970, 1, 1, 0, 0) # datetime.utcfromtimestamp(0)
_datetime_cache = {0: _epoch}


def memorized_datetime(seconds):
    '''Create only one instance of each distinct datetime'''
    try:
        return _datetime_cache[seconds]
    except KeyError:
        # NB. We can't just do datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)
        # as this fails with negative values under Windows (Bug #90096)
        dt = _epoch + timedelta(seconds=seconds)
        _datetime_cache[seconds] = dt
        return dt


_ttinfo_cache = {}


def memorized_ttinfo(*args):
    '''Create only one instance of each distinct tuple'''
    try:
        return _ttinfo_cache[args]
    except KeyError:
        ttinfo = (
            memorized_timedelta(args[0]),
            memorized_timedelta(args[1]),
            args[2]
        )
        _ttinfo_cache[args] = ttinfo
        return ttinfo


_notime = memorized_timedelta(0)


def _to_seconds(td):
    '''Convert a timedelta to seconds'''
    return td.seconds + td.days * 24 * 60 * 60


class BaseTzInfo(tzinfo):
    # Overridden in subclass
    _utcoffset = None
    _tzname = None
    zone = None

    def __str__(self):
        return self.zone


class StaticTzInfo(BaseTzInfo):
    '''A timezone that has a constant offset from UTC

    These timezones are rare, as most locations have changed their
    offset at some point in their history
    '''
    def fromutc(self, dt):
        '''See datetime.tzinfo.fromutc'''
        if dt.tzinfo is not None and dt.tzinfo is not self:
            raise ValueError('fromutc: dt.tzinfo is not self')
        return (dt + self._utcoffset).replace(tzinfo=self)

    def utcoffset(self, dt, is_dst=None):
        '''See datetime.tzinfo.utcoffset

        is_dst is ignored for StaticTzInfo, and exists only to
        retain compatibility with DstTzInfo.
        '''
        return self._utcoffset

    def dst(self, dt, is_dst=None):
        '''See datetime.tzinfo.dst

        is_dst is ignored for StaticTzInfo, and exists only to
        retain compatibility with DstTzInfo.
        '''
        return _notime

    def tzname(self, dt, is_dst=None):
        '''See datetime.tzinfo.tzname

        is_dst is ignored for StaticTzInfo, and exists only to
        retain compatibility with DstTzInfo.
        '''
        return self._tzname

    def localize(self, dt, is_dst=False):
        '''Convert naive time to local time'''
        if dt.tzinfo is not None:
            raise ValueError('Not naive datetime (tzinfo is already set)')
        return dt.replace(tzinfo=self)

    def normalize(self, dt, is_dst=False):
        '''Correct the timezone information on the given datetime.

        This is normally a no-op, as StaticTzInfo timezones never have
        ambiguous cases to correct:

        >>> from pytz import timezone
        >>> gmt = timezone('GMT')
        >>> isinstance(gmt, StaticTzInfo)
        True
        >>> dt = datetime(2011, 5, 8, 1, 2, 3, tzinfo=gmt)
        >>> gmt.normalize(dt) is dt
        True

        The supported method of converting between timezones is to use
        datetime.astimezone(). Currently normalize() also works:

        >>> la = timezone('America/Los_Angeles')
        >>> dt = la.localize(datetime(2011, 5, 7, 1, 2, 3))
        >>> fmt = '%Y-%m-%d %H:%M:%S %Z (%z)'
        >>> gmt.normalize(dt).strftime(fmt)
        '2011-05-07 08:02:03 GMT (+0000)'
        '''
        if dt.tzinfo is self:
            return dt
        if dt.tzinfo is None:
            raise ValueError('Naive time - no tzinfo set')
        return dt.astimezone(self)

    def __repr__(self):
        return '<StaticTzInfo %r>' % (self.zone,)

    def __reduce__(self):
        # Special pickle to zone remains a singleton and to cope with
        # database changes.
        return pytz._p, (self.zone,)


class DstTzInfo(BaseTzInfo):
    '''A timezone that has a variable offset from UTC

    The offset might change if daylight saving time comes into effect,
    or at a point in history when the region decides to change their
    timezone definition.
    '''
    # Overridden in subclass

    # Sorted list of DST transition times, UTC
    _utc_transition_times = None

    # [(utcoffset, dstoffset, tzname)] corresponding to
    # _utc_transition_times entries
    _transition_info = None

    zone = None

    # Set in __init__

    _tzinfos = None
    _dst = None  # DST offset

    def __init__(self, _inf=None, _tzinfos=None):
        if _inf:
            self._tzinfos = _tzinfos
            self._utcoffset, self._dst, self._tzname = _inf
        else:
            _tzinfos = {}
            self._tzinfos = _tzinfos
            self._utcoffset, self._dst, self._tzname = (
                self._transition_info[0])
            _tzinfos[self._transition_info[0]] = self
            for inf in self._transition_info[1:]:
                if inf not in _tzinfos:
                    _tzinfos[inf] = self.__class__(inf, _tzinfos)

    def fromutc(self, dt):
        '''See datetime.tzinfo.fromutc'''
        if (dt.tzinfo is not None and
                getattr(dt.tzinfo, '_tzinfos', None) is not self._tzinfos):
            raise ValueError('fromutc: dt.tzinfo is not self')
        dt = dt.replace(tzinfo=None)
        idx = max(0, bisect_right(self._utc_transition_times, dt) - 1)
        inf = self._transition_info[idx]
        return (dt + inf[0]).replace(tzinfo=self._tzinfos[inf])

    def normalize(self, dt):
        '''Correct the timezone information on the given datetime

        If date arithmetic crosses DST boundaries, the tzinfo
        is not magically adjusted. This method normalizes the
        tzinfo to the correct one.

        To test, first we need to do some setup

        >>> from pytz import timezone
        >>> utc = timezone('UTC')
        >>> eastern = timezone('US/Eastern')
        >>> fmt = '%Y-%m-%d %H:%M:%S %Z (%z)'

        We next create a datetime right on an end-of-DST transition point,
        the instant when the wallclocks are wound back one hour.

        >>> utc_dt = datetime(2002, 10, 27, 6, 0, 0, tzinfo=utc)
        >>> loc_dt = utc_dt.astimezone(eastern)
        >>> loc_dt.strftime(fmt)
        '2002-10-27 01:00:00 EST (-0500)'

        Now, if we subtract a few minutes from it, note that the timezone
        information has not changed.

        >>> before = loc_dt - timedelta(minutes=10)
        >>> before.strftime(fmt)
        '2002-10-27 00:50:00 EST (-0500)'

        But we can fix that by calling the normalize method

        >>> before = eastern.normalize(before)
        >>> before.strftime(fmt)
        '2002-10-27 01:50:00 EDT (-0400)'

        The supported method of converting between timezones is to use
        datetime.astimezone(). Currently, normalize() also works:

        >>> th = timezone('Asia/Bangkok')
        >>> am = timezone('Europe/Amsterdam')
        >>> dt = th.localize(datetime(2011, 5, 7, 1, 2, 3))
        >>> fmt = '%Y-%m-%d %H:%M:%S %Z (%z)'
        >>> am.normalize(dt).strftime(fmt)
        '2011-05-06 20:02:03 CEST (+0200)'
        '''
        if dt.tzinfo is None:
            raise ValueError('Naive time - no tzinfo set')

        # Convert dt in localtime to UTC
        offset = dt.tzinfo._utcoffset
        dt = dt.replace(tzinfo=None)
        dt = dt - offset
        # convert it back, and return it
        return self.fromutc(dt)

    def localize(self, dt, is_dst=False):
        '''Convert naive time to local time.

        This method should be used to construct localtimes, rather
        than passing a tzinfo argument to a datetime constructor.

        is_dst is used to determine the correct timezone in the ambigous
        period at the end of daylight saving time.

        >>> from pytz import timezone
        >>> fmt = '%Y-%m-%d %H:%M:%S %Z (%z)'
        >>> amdam = timezone('Europe/Amsterdam')
        >>> dt  = datetime(2004, 10, 31, 2, 0, 0)
        >>> loc_dt1 = amdam.localize(dt, is_dst=True)
        >>> loc_dt2 = amdam.localize(dt, is_dst=False)
        >>> loc_dt1.strftime(fmt)
        '2004-10-31 02:00:00 CEST (+0200)'
        >>> loc_dt2.strftime(fmt)
        '2004-10-31 02:00:00 CET (+0100)'
        >>> str(loc_dt2 - loc_dt1)
        '1:00:00'

        Use is_dst=None to raise an AmbiguousTimeError for ambiguous
        times at the end of daylight saving time

        >>> try:
        ...     loc_dt1 = amdam.localize(dt, is_dst=None)
        ... except AmbiguousTimeError:
        ...     print('Ambiguous')
        Ambiguous

        is_dst defaults to False

        >>> amdam.localize(dt) == amdam.localize(dt, False)
        True

        is_dst is also used to determine the correct timezone in the
        wallclock times jumped over at the start of daylight saving time.

        >>> pacific = timezone('US/Pacific')
        >>> dt = datetime(2008, 3, 9, 2, 0, 0)
        >>> ploc_dt1 = pacific.localize(dt, is_dst=True)
        >>> ploc_dt2 = pacific.localize(dt, is_dst=False)
        >>> ploc_dt1.strftime(fmt)
        '2008-03-09 02:00:00 PDT (-0700)'
        >>> ploc_dt2.strftime(fmt)
        '2008-03-09 02:00:00 PST (-0800)'
        >>> str(ploc_dt2 - ploc_dt1)
        '1:00:00'

        Use is_dst=None to raise a NonExistentTimeError for these skipped
        times.

        >>> try:
        ...     loc_dt1 = pacific.localize(dt, is_dst=None)
        ... except NonExistentTimeError:
        ...     print('Non-existent')
        Non-existent
        '''
        if dt.tzinfo is not None:
            raise ValueError('Not naive datetime (tzinfo is already set)')

        # Find the two best possibilities.
        possible_loc_dt = set()
        for delta in [timedelta(days=-1), timedelta(days=1)]:
            loc_dt = dt + delta
            idx = max(0, bisect_right(
                self._utc_transition_times, loc_dt) - 1)
            inf = self._transition_info[idx]
            tzinfo = self._tzinfos[inf]
            loc_dt = tzinfo.normalize(dt.replace(tzinfo=tzinfo))
            if loc_dt.replace(tzinfo=None) == dt:
                possible_loc_dt.add(loc_dt)

        if len(possible_loc_dt) == 1:
            return possible_loc_dt.pop()

        # If there are no possibly correct timezones, we are attempting
        # to convert a time that never happened - the time period jumped
        # during the start-of-DST transition period.
        if len(possible_loc_dt) == 0:
            # If we refuse to guess, raise an exception.
            if is_dst is None:
                raise NonExistentTimeError(dt)

            # If we are forcing the pre-DST side of the DST transition, we
            # obtain the correct timezone by winding the clock forward a few
            # hours.
            elif is_dst:
                return self.localize(
                    dt + timedelta(hours=6), is_dst=True) - timedelta(hours=6)

            # If we are forcing the post-DST side of the DST transition, we
            # obtain the correct timezone by winding the clock back.
            else:
                return self.localize(
                    dt - timedelta(hours=6),
                    is_dst=False) + timedelta(hours=6)

        # If we get this far, we have multiple possible timezones - this
        # is an ambiguous case occurring during the end-of-DST transition.

        # If told to be strict, raise an exception since we have an
        # ambiguous case
        if is_dst is None:
            raise AmbiguousTimeError(dt)

        # Filter out the possiblilities that don't match the requested
        # is_dst
        filtered_possible_loc_dt = [
            p for p in possible_loc_dt if bool(p.tzinfo._dst) == is_dst
        ]

        # Hopefully we only have one possibility left. Return it.
        if len(filtered_possible_loc_dt) == 1:
            return filtered_possible_loc_dt[0]

        if len(filtered_possible_loc_dt) == 0:
            filtered_possible_loc_dt = list(possible_loc_dt)

        # If we get this far, we have in a wierd timezone transition
        # where the clocks have been wound back but is_dst is the same
        # in both (eg. Europe/Warsaw 1915 when they switched to CET).
        # At this point, we just have to guess unless we allow more
        # hints to be passed in (such as the UTC offset or abbreviation),
        # but that is just getting silly.
        #
        # Choose the earliest (by UTC) applicable timezone if is_dst=True
        # Choose the latest (by UTC) applicable timezone if is_dst=False
        # i.e., behave like end-of-DST transition
        dates = {}  # utc -> local
        for local_dt in filtered_possible_loc_dt:
            utc_time = (
                local_dt.replace(tzinfo=None) - local_dt.tzinfo._utcoffset)
            assert utc_time not in dates
            dates[utc_time] = local_dt
        return dates[[min, max][not is_dst](dates)]

    def utcoffset(self, dt, is_dst=None):
        '''See datetime.tzinfo.utcoffset

        The is_dst parameter may be used to remove ambiguity during DST
        transitions.

        >>> from pytz import timezone
        >>> tz = timezone('America/St_Johns')
        >>> ambiguous = datetime(2009, 10, 31, 23, 30)

        >>> str(tz.utcoffset(ambiguous, is_dst=False))
        '-1 day, 20:30:00'

        >>> str(tz.utcoffset(ambiguous, is_dst=True))
        '-1 day, 21:30:00'

        >>> try:
        ...     tz.utcoffset(ambiguous)
        ... except AmbiguousTimeError:
        ...     print('Ambiguous')
        Ambiguous

        '''
        if dt is None:
            return None
        elif dt.tzinfo is not self:
            dt = self.localize(dt, is_dst)
            return dt.tzinfo._utcoffset
        else:
            return self._utcoffset

    def dst(self, dt, is_dst=None):
        '''See datetime.tzinfo.dst

        The is_dst parameter may be used to remove ambiguity during DST
        transitions.

        >>> from pytz import timezone
        >>> tz = timezone('America/St_Johns')

        >>> normal = datetime(2009, 9, 1)

        >>> str(tz.dst(normal))
        '1:00:00'
        >>> str(tz.dst(normal, is_dst=False))
        '1:00:00'
        >>> str(tz.dst(normal, is_dst=True))
        '1:00:00'

        >>> ambiguous = datetime(2009, 10, 31, 23, 30)

        >>> str(tz.dst(ambiguous, is_dst=False))
        '0:00:00'
        >>> str(tz.dst(ambiguous, is_dst=True))
        '1:00:00'
        >>> try:
        ...     tz.dst(ambiguous)
        ... except AmbiguousTimeError:
        ...     print('Ambiguous')
        Ambiguous

        '''
        if dt is None:
            return None
        elif dt.tzinfo is not self:
            dt = self.localize(dt, is_dst)
            return dt.tzinfo._dst
        else:
            return self._dst

    def tzname(self, dt, is_dst=None):
        '''See datetime.tzinfo.tzname

        The is_dst parameter may be used to remove ambiguity during DST
        transitions.

        >>> from pytz import timezone
        >>> tz = timezone('America/St_Johns')

        >>> normal = datetime(2009, 9, 1)

        >>> tz.tzname(normal)
        'NDT'
        >>> tz.tzname(normal, is_dst=False)
        'NDT'
        >>> tz.tzname(normal, is_dst=True)
        'NDT'

        >>> ambiguous = datetime(2009, 10, 31, 23, 30)

        >>> tz.tzname(ambiguous, is_dst=False)
        'NST'
        >>> tz.tzname(ambiguous, is_dst=True)
        'NDT'
        >>> try:
        ...     tz.tzname(ambiguous)
        ... except AmbiguousTimeError:
        ...     print('Ambiguous')
        Ambiguous
        '''
        if dt is None:
            return self.zone
        elif dt.tzinfo is not self:
            dt = self.localize(dt, is_dst)
            return dt.tzinfo._tzname
        else:
            return self._tzname

    def __repr__(self):
        if self._dst:
            dst = 'DST'
        else:
            dst = 'STD'
        if self._utcoffset > _notime:
            return '<DstTzInfo %r %s+%s %s>' % (
                self.zone, self._tzname, self._utcoffset, dst
            )
        else:
            return '<DstTzInfo %r %s%s %s>' % (
                self.zone, self._tzname, self._utcoffset, dst
            )

    def __reduce__(self):
        # Special pickle to zone remains a singleton and to cope with
        # database changes.
        return pytz._p, (
            self.zone,
            _to_seconds(self._utcoffset),
            _to_seconds(self._dst),
            self._tzname
        )


def unpickler(zone, utcoffset=None, dstoffset=None, tzname=None):
    """Factory function for unpickling pytz tzinfo instances.

    This is shared for both StaticTzInfo and DstTzInfo instances, because
    database changes could cause a zones implementation to switch between
    these two base classes and we can't break pickles on a pytz version
    upgrade.
    """
    # Raises a KeyError if zone no longer exists, which should never happen
    # and would be a bug.
    tz = pytz.timezone(zone)

    # A StaticTzInfo - just return it
    if utcoffset is None:
        return tz

    # This pickle was created from a DstTzInfo. We need to
    # determine which of the list of tzinfo instances for this zone
    # to use in order to restore the state of any datetime instances using
    # it correctly.
    utcoffset = memorized_timedelta(utcoffset)
    dstoffset = memorized_timedelta(dstoffset)
    try:
        return tz._tzinfos[(utcoffset, dstoffset, tzname)]
    except KeyError:
        # The particular state requested in this timezone no longer exists.
        # This indicates a corrupt pickle, or the timezone database has been
        # corrected violently enough to make this particular
        # (utcoffset,dstoffset) no longer exist in the zone, or the
        # abbreviation has been changed.
        pass

    # See if we can find an entry differing only by tzname. Abbreviations
    # get changed from the initial guess by the database maintainers to
    # match reality when this information is discovered.
    for localized_tz in tz._tzinfos.values():
        if (localized_tz._utcoffset == utcoffset and
                localized_tz._dst == dstoffset):
            return localized_tz

    # This (utcoffset, dstoffset) information has been removed from the
    # zone. Add it back. This might occur when the database maintainers have
    # corrected incorrect information. datetime instances using this
    # incorrect information will continue to do so, exactly as they were
    # before being pickled. This is purely an overly paranoid safety net - I
    # doubt this will ever been needed in real life.
    inf = (utcoffset, dstoffset, tzname)
    tz._tzinfos[inf] = tz.__class__(inf, tz._tzinfos)
    return tz._tzinfos[inf]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\util\version\__init__.py ===
# Vendored from https://github.com/pypa/packaging/blob/main/packaging/_structures.py
# and https://github.com/pypa/packaging/blob/main/packaging/_structures.py
# changeset ae891fd74d6dd4c6063bb04f2faeadaac6fc6313
# 04/30/2021

# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. Licence at LICENSES/PACKAGING_LICENSE
from __future__ import annotations

import collections
from collections.abc import Iterator
import itertools
import re
from typing import (
    Callable,
    SupportsInt,
    Tuple,
    Union,
)
import warnings

__all__ = ["parse", "Version", "LegacyVersion", "InvalidVersion", "VERSION_PATTERN"]


class InfinityType:
    def __repr__(self) -> str:
        return "Infinity"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        return False

    def __le__(self, other: object) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self))

    def __ne__(self, other: object) -> bool:
        return not isinstance(other, type(self))

    def __gt__(self, other: object) -> bool:
        return True

    def __ge__(self, other: object) -> bool:
        return True

    def __neg__(self: object) -> NegativeInfinityType:
        return NegativeInfinity


Infinity = InfinityType()


class NegativeInfinityType:
    def __repr__(self) -> str:
        return "-Infinity"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        return True

    def __le__(self, other: object) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self))

    def __ne__(self, other: object) -> bool:
        return not isinstance(other, type(self))

    def __gt__(self, other: object) -> bool:
        return False

    def __ge__(self, other: object) -> bool:
        return False

    def __neg__(self: object) -> InfinityType:
        return Infinity


NegativeInfinity = NegativeInfinityType()


InfiniteTypes = Union[InfinityType, NegativeInfinityType]
PrePostDevType = Union[InfiniteTypes, tuple[str, int]]
SubLocalType = Union[InfiniteTypes, int, str]
LocalType = Union[
    NegativeInfinityType,
    tuple[
        Union[
            SubLocalType,
            tuple[SubLocalType, str],
            tuple[NegativeInfinityType, SubLocalType],
        ],
        ...,
    ],
]
CmpKey = tuple[
    int, tuple[int, ...], PrePostDevType, PrePostDevType, PrePostDevType, LocalType
]
LegacyCmpKey = tuple[int, tuple[str, ...]]
VersionComparisonMethod = Callable[
    [Union[CmpKey, LegacyCmpKey], Union[CmpKey, LegacyCmpKey]], bool
]

_Version = collections.namedtuple(
    "_Version", ["epoch", "release", "dev", "pre", "post", "local"]
)


def parse(version: str) -> LegacyVersion | Version:
    """
    Parse the given version string and return either a :class:`Version` object
    or a :class:`LegacyVersion` object depending on if the given version is
    a valid PEP 440 version or a legacy version.
    """
    try:
        return Version(version)
    except InvalidVersion:
        return LegacyVersion(version)


class InvalidVersion(ValueError):
    """
    An invalid version was found, users should refer to PEP 440.

    Examples
    --------
    >>> pd.util.version.Version('1.')
    Traceback (most recent call last):
    InvalidVersion: Invalid version: '1.'
    """


class _BaseVersion:
    _key: CmpKey | LegacyCmpKey

    def __hash__(self) -> int:
        return hash(self._key)

    # Please keep the duplicated `isinstance` check
    # in the six comparisons hereunder
    # unless you find a way to avoid adding overhead function calls.
    def __lt__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key < other._key

    def __le__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key <= other._key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key == other._key

    def __ge__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key >= other._key

    def __gt__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key > other._key

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key != other._key


class LegacyVersion(_BaseVersion):
    def __init__(self, version: str) -> None:
        self._version = str(version)
        self._key = _legacy_cmpkey(self._version)

        warnings.warn(
            "Creating a LegacyVersion has been deprecated and will be "
            "removed in the next major release.",
            DeprecationWarning,
        )

    def __str__(self) -> str:
        return self._version

    def __repr__(self) -> str:
        return f"<LegacyVersion('{self}')>"

    @property
    def public(self) -> str:
        return self._version

    @property
    def base_version(self) -> str:
        return self._version

    @property
    def epoch(self) -> int:
        return -1

    @property
    def release(self) -> None:
        return None

    @property
    def pre(self) -> None:
        return None

    @property
    def post(self) -> None:
        return None

    @property
    def dev(self) -> None:
        return None

    @property
    def local(self) -> None:
        return None

    @property
    def is_prerelease(self) -> bool:
        return False

    @property
    def is_postrelease(self) -> bool:
        return False

    @property
    def is_devrelease(self) -> bool:
        return False


_legacy_version_component_re = re.compile(r"(\d+ | [a-z]+ | \.| -)", re.VERBOSE)

_legacy_version_replacement_map = {
    "pre": "c",
    "preview": "c",
    "-": "final-",
    "rc": "c",
    "dev": "@",
}


def _parse_version_parts(s: str) -> Iterator[str]:
    for part in _legacy_version_component_re.split(s):
        mapped_part = _legacy_version_replacement_map.get(part, part)

        if not mapped_part or mapped_part == ".":
            continue

        if mapped_part[:1] in "0123456789":
            # pad for numeric comparison
            yield mapped_part.zfill(8)
        else:
            yield "*" + mapped_part

    # ensure that alpha/beta/candidate are before final
    yield "*final"


def _legacy_cmpkey(version: str) -> LegacyCmpKey:
    # We hardcode an epoch of -1 here. A PEP 440 version can only have a epoch
    # greater than or equal to 0. This will effectively put the LegacyVersion,
    # which uses the defacto standard originally implemented by setuptools,
    # as before all PEP 440 versions.
    epoch = -1

    # This scheme is taken from pkg_resources.parse_version setuptools prior to
    # it's adoption of the packaging library.
    parts: list[str] = []
    for part in _parse_version_parts(version.lower()):
        if part.startswith("*"):
            # remove "-" before a prerelease tag
            if part < "*final":
                while parts and parts[-1] == "*final-":
                    parts.pop()

            # remove trailing zeros from each series of numeric parts
            while parts and parts[-1] == "00000000":
                parts.pop()

        parts.append(part)

    return epoch, tuple(parts)


# Deliberately not anchored to the start and end of the string, to make it
# easier for 3rd party code to reuse
VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""


class Version(_BaseVersion):
    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)

    def __init__(self, version: str) -> None:
        # Validate the version and parse it into pieces
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: '{version}'")

        # Store the parsed out pieces of the version
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,
            release=tuple(int(i) for i in match.group("release").split(".")),
            pre=_parse_letter_version(match.group("pre_l"), match.group("pre_n")),
            post=_parse_letter_version(
                match.group("post_l"), match.group("post_n1") or match.group("post_n2")
            ),
            dev=_parse_letter_version(match.group("dev_l"), match.group("dev_n")),
            local=_parse_local_version(match.group("local")),
        )

        # Generate a key which will be used for sorting
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local,
        )

    def __repr__(self) -> str:
        return f"<Version('{self}')>"

    def __str__(self) -> str:
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join([str(x) for x in self.release]))

        # Pre-release
        if self.pre is not None:
            parts.append("".join([str(x) for x in self.pre]))

        # Post-release
        if self.post is not None:
            parts.append(f".post{self.post}")

        # Development release
        if self.dev is not None:
            parts.append(f".dev{self.dev}")

        # Local version segment
        if self.local is not None:
            parts.append(f"+{self.local}")

        return "".join(parts)

    @property
    def epoch(self) -> int:
        _epoch: int = self._version.epoch
        return _epoch

    @property
    def release(self) -> tuple[int, ...]:
        _release: tuple[int, ...] = self._version.release
        return _release

    @property
    def pre(self) -> tuple[str, int] | None:
        _pre: tuple[str, int] | None = self._version.pre
        return _pre

    @property
    def post(self) -> int | None:
        return self._version.post[1] if self._version.post else None

    @property
    def dev(self) -> int | None:
        return self._version.dev[1] if self._version.dev else None

    @property
    def local(self) -> str | None:
        if self._version.local:
            return ".".join([str(x) for x in self._version.local])
        else:
            return None

    @property
    def public(self) -> str:
        return str(self).split("+", 1)[0]

    @property
    def base_version(self) -> str:
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join([str(x) for x in self.release]))

        return "".join(parts)

    @property
    def is_prerelease(self) -> bool:
        return self.dev is not None or self.pre is not None

    @property
    def is_postrelease(self) -> bool:
        return self.post is not None

    @property
    def is_devrelease(self) -> bool:
        return self.dev is not None

    @property
    def major(self) -> int:
        return self.release[0] if len(self.release) >= 1 else 0

    @property
    def minor(self) -> int:
        return self.release[1] if len(self.release) >= 2 else 0

    @property
    def micro(self) -> int:
        return self.release[2] if len(self.release) >= 3 else 0


def _parse_letter_version(
    letter: str, number: str | bytes | SupportsInt
) -> tuple[str, int] | None:
    if letter:
        # We consider there to be an implicit 0 in a pre-release if there is
        # not a numeral associated with it.
        if number is None:
            number = 0

        # We normalize any letters to their lower case form
        letter = letter.lower()

        # We consider some words to be alternate spellings of other words and
        # in those cases we want to normalize the spellings to our preferred
        # spelling.
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"

        return letter, int(number)
    if not letter and number:
        # We assume if we are given a number, but we are not given a letter
        # then this is using the implicit post release syntax (e.g. 1.0-1)
        letter = "post"

        return letter, int(number)

    return None


_local_version_separators = re.compile(r"[\._-]")


def _parse_local_version(local: str) -> LocalType | None:
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    if local is not None:
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_separators.split(local)
        )
    return None


def _cmpkey(
    epoch: int,
    release: tuple[int, ...],
    pre: tuple[str, int] | None,
    post: tuple[str, int] | None,
    dev: tuple[str, int] | None,
    local: tuple[SubLocalType] | None,
) -> CmpKey:
    # When we compare a release version, we want to compare it with all of the
    # trailing zeros removed. So we'll use a reverse the list, drop all the now
    # leading zeros until we come to something non zero, then take the rest
    # re-reverse it back into the correct order and make it a tuple and use
    # that for our sorting key.
    _release = tuple(
        reversed(list(itertools.dropwhile(lambda x: x == 0, reversed(release))))
    )

    # We need to "trick" the sorting algorithm to put 1.0.dev0 before 1.0a0.
    # We'll do this by abusing the pre segment, but we _only_ want to do this
    # if there is not a pre or a post segment. If we have one of those then
    # the normal sorting rules will handle this case correctly.
    if pre is None and post is None and dev is not None:
        _pre: PrePostDevType = NegativeInfinity
    # Versions without a pre-release (except as noted above) should sort after
    # those with one.
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre

    # Versions without a post segment should sort before those with one.
    if post is None:
        _post: PrePostDevType = NegativeInfinity

    else:
        _post = post

    # Versions without a development segment should sort after those with one.
    if dev is None:
        _dev: PrePostDevType = Infinity

    else:
        _dev = dev

    if local is None:
        # Versions without a local segment should sort before those with one.
        _local: LocalType = NegativeInfinity
    else:
        # Versions with a local segment need that segment parsed to implement
        # the sorting rules in PEP440.
        # - Alpha numeric segments sort before numeric segments
        # - Alpha numeric segments sort lexicographically
        # - Numeric segments sort numerically
        # - Shorter versions sort before longer versions when the prefixes
        #   match exactly
        _local = tuple(
            (i, "") if isinstance(i, int) else (NegativeInfinity, i) for i in local
        )

    return epoch, _release, _pre, _post, _dev, _local

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\resolution\resolvelib\candidates.py ===
import logging
import sys
from typing import TYPE_CHECKING, Any, FrozenSet, Iterable, Optional, Tuple, Union, cast

from pip._vendor.packaging.requirements import InvalidRequirement
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.packaging.version import Version

from pip._internal.exceptions import (
    HashError,
    InstallationSubprocessError,
    InvalidInstalledPackage,
    MetadataInconsistent,
    MetadataInvalid,
)
from pip._internal.metadata import BaseDistribution
from pip._internal.models.link import Link, links_equivalent
from pip._internal.models.wheel import Wheel
from pip._internal.req.constructors import (
    install_req_from_editable,
    install_req_from_line,
)
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.direct_url_helpers import direct_url_from_link
from pip._internal.utils.misc import normalize_version_info

from .base import Candidate, Requirement, format_name

if TYPE_CHECKING:
    from .factory import Factory

logger = logging.getLogger(__name__)

BaseCandidate = Union[
    "AlreadyInstalledCandidate",
    "EditableCandidate",
    "LinkCandidate",
]

# Avoid conflicting with the PyPI package "Python".
REQUIRES_PYTHON_IDENTIFIER = cast(NormalizedName, "<Python from Requires-Python>")


def as_base_candidate(candidate: Candidate) -> Optional[BaseCandidate]:
    """The runtime version of BaseCandidate."""
    base_candidate_classes = (
        AlreadyInstalledCandidate,
        EditableCandidate,
        LinkCandidate,
    )
    if isinstance(candidate, base_candidate_classes):
        return candidate
    return None


def make_install_req_from_link(
    link: Link, template: InstallRequirement
) -> InstallRequirement:
    assert not template.editable, "template is editable"
    if template.req:
        line = str(template.req)
    else:
        line = link.url
    ireq = install_req_from_line(
        line,
        user_supplied=template.user_supplied,
        comes_from=template.comes_from,
        use_pep517=template.use_pep517,
        isolated=template.isolated,
        constraint=template.constraint,
        global_options=template.global_options,
        hash_options=template.hash_options,
        config_settings=template.config_settings,
    )
    ireq.original_link = template.original_link
    ireq.link = link
    ireq.extras = template.extras
    return ireq


def make_install_req_from_editable(
    link: Link, template: InstallRequirement
) -> InstallRequirement:
    assert template.editable, "template not editable"
    ireq = install_req_from_editable(
        link.url,
        user_supplied=template.user_supplied,
        comes_from=template.comes_from,
        use_pep517=template.use_pep517,
        isolated=template.isolated,
        constraint=template.constraint,
        permit_editable_wheels=template.permit_editable_wheels,
        global_options=template.global_options,
        hash_options=template.hash_options,
        config_settings=template.config_settings,
    )
    ireq.extras = template.extras
    return ireq


def _make_install_req_from_dist(
    dist: BaseDistribution, template: InstallRequirement
) -> InstallRequirement:
    if template.req:
        line = str(template.req)
    elif template.link:
        line = f"{dist.canonical_name} @ {template.link.url}"
    else:
        line = f"{dist.canonical_name}=={dist.version}"
    ireq = install_req_from_line(
        line,
        user_supplied=template.user_supplied,
        comes_from=template.comes_from,
        use_pep517=template.use_pep517,
        isolated=template.isolated,
        constraint=template.constraint,
        global_options=template.global_options,
        hash_options=template.hash_options,
        config_settings=template.config_settings,
    )
    ireq.satisfied_by = dist
    return ireq


class _InstallRequirementBackedCandidate(Candidate):
    """A candidate backed by an ``InstallRequirement``.

    This represents a package request with the target not being already
    in the environment, and needs to be fetched and installed. The backing
    ``InstallRequirement`` is responsible for most of the leg work; this
    class exposes appropriate information to the resolver.

    :param link: The link passed to the ``InstallRequirement``. The backing
        ``InstallRequirement`` will use this link to fetch the distribution.
    :param source_link: The link this candidate "originates" from. This is
        different from ``link`` when the link is found in the wheel cache.
        ``link`` would point to the wheel cache, while this points to the
        found remote link (e.g. from pypi.org).
    """

    dist: BaseDistribution
    is_installed = False

    def __init__(
        self,
        link: Link,
        source_link: Link,
        ireq: InstallRequirement,
        factory: "Factory",
        name: Optional[NormalizedName] = None,
        version: Optional[Version] = None,
    ) -> None:
        self._link = link
        self._source_link = source_link
        self._factory = factory
        self._ireq = ireq
        self._name = name
        self._version = version
        self.dist = self._prepare()
        self._hash: Optional[int] = None

    def __str__(self) -> str:
        return f"{self.name} {self.version}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self._link)!r})"

    def __hash__(self) -> int:
        if self._hash is not None:
            return self._hash

        self._hash = hash((self.__class__, self._link))
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return links_equivalent(self._link, other._link)
        return False

    @property
    def source_link(self) -> Optional[Link]:
        return self._source_link

    @property
    def project_name(self) -> NormalizedName:
        """The normalised name of the project the candidate refers to"""
        if self._name is None:
            self._name = self.dist.canonical_name
        return self._name

    @property
    def name(self) -> str:
        return self.project_name

    @property
    def version(self) -> Version:
        if self._version is None:
            self._version = self.dist.version
        return self._version

    def format_for_error(self) -> str:
        return (
            f"{self.name} {self.version} "
            f"(from {self._link.file_path if self._link.is_file else self._link})"
        )

    def _prepare_distribution(self) -> BaseDistribution:
        raise NotImplementedError("Override in subclass")

    def _check_metadata_consistency(self, dist: BaseDistribution) -> None:
        """Check for consistency of project name and version of dist."""
        if self._name is not None and self._name != dist.canonical_name:
            raise MetadataInconsistent(
                self._ireq,
                "name",
                self._name,
                dist.canonical_name,
            )
        if self._version is not None and self._version != dist.version:
            raise MetadataInconsistent(
                self._ireq,
                "version",
                str(self._version),
                str(dist.version),
            )
        # check dependencies are valid
        # TODO performance: this means we iterate the dependencies at least twice,
        # we may want to cache parsed Requires-Dist
        try:
            list(dist.iter_dependencies(list(dist.iter_provided_extras())))
        except InvalidRequirement as e:
            raise MetadataInvalid(self._ireq, str(e))

    def _prepare(self) -> BaseDistribution:
        try:
            dist = self._prepare_distribution()
        except HashError as e:
            # Provide HashError the underlying ireq that caused it. This
            # provides context for the resulting error message to show the
            # offending line to the user.
            e.req = self._ireq
            raise
        except InstallationSubprocessError as exc:
            # The output has been presented already, so don't duplicate it.
            exc.context = "See above for output."
            raise

        self._check_metadata_consistency(dist)
        return dist

    def iter_dependencies(self, with_requires: bool) -> Iterable[Optional[Requirement]]:
        # Emit the Requires-Python requirement first to fail fast on
        # unsupported candidates and avoid pointless downloads/preparation.
        yield self._factory.make_requires_python_requirement(self.dist.requires_python)
        requires = self.dist.iter_dependencies() if with_requires else ()
        for r in requires:
            yield from self._factory.make_requirements_from_spec(str(r), self._ireq)

    def get_install_requirement(self) -> Optional[InstallRequirement]:
        return self._ireq


class LinkCandidate(_InstallRequirementBackedCandidate):
    is_editable = False

    def __init__(
        self,
        link: Link,
        template: InstallRequirement,
        factory: "Factory",
        name: Optional[NormalizedName] = None,
        version: Optional[Version] = None,
    ) -> None:
        source_link = link
        cache_entry = factory.get_wheel_cache_entry(source_link, name)
        if cache_entry is not None:
            logger.debug("Using cached wheel link: %s", cache_entry.link)
            link = cache_entry.link
        ireq = make_install_req_from_link(link, template)
        assert ireq.link == link
        if ireq.link.is_wheel and not ireq.link.is_file:
            wheel = Wheel(ireq.link.filename)
            wheel_name = canonicalize_name(wheel.name)
            assert name == wheel_name, f"{name!r} != {wheel_name!r} for wheel"
            # Version may not be present for PEP 508 direct URLs
            if version is not None:
                wheel_version = Version(wheel.version)
                assert (
                    version == wheel_version
                ), f"{version!r} != {wheel_version!r} for wheel {name}"

        if cache_entry is not None:
            assert ireq.link.is_wheel
            assert ireq.link.is_file
            if cache_entry.persistent and template.link is template.original_link:
                ireq.cached_wheel_source_link = source_link
            if cache_entry.origin is not None:
                ireq.download_info = cache_entry.origin
            else:
                # Legacy cache entry that does not have origin.json.
                # download_info may miss the archive_info.hashes field.
                ireq.download_info = direct_url_from_link(
                    source_link, link_is_in_wheel_cache=cache_entry.persistent
                )

        super().__init__(
            link=link,
            source_link=source_link,
            ireq=ireq,
            factory=factory,
            name=name,
            version=version,
        )

    def _prepare_distribution(self) -> BaseDistribution:
        preparer = self._factory.preparer
        return preparer.prepare_linked_requirement(self._ireq, parallel_builds=True)


class EditableCandidate(_InstallRequirementBackedCandidate):
    is_editable = True

    def __init__(
        self,
        link: Link,
        template: InstallRequirement,
        factory: "Factory",
        name: Optional[NormalizedName] = None,
        version: Optional[Version] = None,
    ) -> None:
        super().__init__(
            link=link,
            source_link=link,
            ireq=make_install_req_from_editable(link, template),
            factory=factory,
            name=name,
            version=version,
        )

    def _prepare_distribution(self) -> BaseDistribution:
        return self._factory.preparer.prepare_editable_requirement(self._ireq)


class AlreadyInstalledCandidate(Candidate):
    is_installed = True
    source_link = None

    def __init__(
        self,
        dist: BaseDistribution,
        template: InstallRequirement,
        factory: "Factory",
    ) -> None:
        self.dist = dist
        self._ireq = _make_install_req_from_dist(dist, template)
        self._factory = factory
        self._version = None

        # This is just logging some messages, so we can do it eagerly.
        # The returned dist would be exactly the same as self.dist because we
        # set satisfied_by in _make_install_req_from_dist.
        # TODO: Supply reason based on force_reinstall and upgrade_strategy.
        skip_reason = "already satisfied"
        factory.preparer.prepare_installed_requirement(self._ireq, skip_reason)

    def __str__(self) -> str:
        return str(self.dist)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.dist!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AlreadyInstalledCandidate):
            return NotImplemented
        return self.name == other.name and self.version == other.version

    def __hash__(self) -> int:
        return hash((self.name, self.version))

    @property
    def project_name(self) -> NormalizedName:
        return self.dist.canonical_name

    @property
    def name(self) -> str:
        return self.project_name

    @property
    def version(self) -> Version:
        if self._version is None:
            self._version = self.dist.version
        return self._version

    @property
    def is_editable(self) -> bool:
        return self.dist.editable

    def format_for_error(self) -> str:
        return f"{self.name} {self.version} (Installed)"

    def iter_dependencies(self, with_requires: bool) -> Iterable[Optional[Requirement]]:
        if not with_requires:
            return

        try:
            for r in self.dist.iter_dependencies():
                yield from self._factory.make_requirements_from_spec(str(r), self._ireq)
        except InvalidRequirement as exc:
            raise InvalidInstalledPackage(dist=self.dist, invalid_exc=exc) from None

    def get_install_requirement(self) -> Optional[InstallRequirement]:
        return None


class ExtrasCandidate(Candidate):
    """A candidate that has 'extras', indicating additional dependencies.

    Requirements can be for a project with dependencies, something like
    foo[extra].  The extras don't affect the project/version being installed
    directly, but indicate that we need additional dependencies. We model that
    by having an artificial ExtrasCandidate that wraps the "base" candidate.

    The ExtrasCandidate differs from the base in the following ways:

    1. It has a unique name, of the form foo[extra]. This causes the resolver
       to treat it as a separate node in the dependency graph.
    2. When we're getting the candidate's dependencies,
       a) We specify that we want the extra dependencies as well.
       b) We add a dependency on the base candidate.
          See below for why this is needed.
    3. We return None for the underlying InstallRequirement, as the base
       candidate will provide it, and we don't want to end up with duplicates.

    The dependency on the base candidate is needed so that the resolver can't
    decide that it should recommend foo[extra1] version 1.0 and foo[extra2]
    version 2.0. Having those candidates depend on foo=1.0 and foo=2.0
    respectively forces the resolver to recognise that this is a conflict.
    """

    def __init__(
        self,
        base: BaseCandidate,
        extras: FrozenSet[str],
        *,
        comes_from: Optional[InstallRequirement] = None,
    ) -> None:
        """
        :param comes_from: the InstallRequirement that led to this candidate if it
            differs from the base's InstallRequirement. This will often be the
            case in the sense that this candidate's requirement has the extras
            while the base's does not. Unlike the InstallRequirement backed
            candidates, this requirement is used solely for reporting purposes,
            it does not do any leg work.
        """
        self.base = base
        self.extras = frozenset(canonicalize_name(e) for e in extras)
        self._comes_from = comes_from if comes_from is not None else self.base._ireq

    def __str__(self) -> str:
        name, rest = str(self.base).split(" ", 1)
        return "{}[{}] {}".format(name, ",".join(self.extras), rest)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(base={self.base!r}, extras={self.extras!r})"

    def __hash__(self) -> int:
        return hash((self.base, self.extras))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self.base == other.base and self.extras == other.extras
        return False

    @property
    def project_name(self) -> NormalizedName:
        return self.base.project_name

    @property
    def name(self) -> str:
        """The normalised name of the project the candidate refers to"""
        return format_name(self.base.project_name, self.extras)

    @property
    def version(self) -> Version:
        return self.base.version

    def format_for_error(self) -> str:
        return "{} [{}]".format(
            self.base.format_for_error(), ", ".join(sorted(self.extras))
        )

    @property
    def is_installed(self) -> bool:
        return self.base.is_installed

    @property
    def is_editable(self) -> bool:
        return self.base.is_editable

    @property
    def source_link(self) -> Optional[Link]:
        return self.base.source_link

    def iter_dependencies(self, with_requires: bool) -> Iterable[Optional[Requirement]]:
        factory = self.base._factory

        # Add a dependency on the exact base
        # (See note 2b in the class docstring)
        yield factory.make_requirement_from_candidate(self.base)
        if not with_requires:
            return

        # The user may have specified extras that the candidate doesn't
        # support. We ignore any unsupported extras here.
        valid_extras = self.extras.intersection(self.base.dist.iter_provided_extras())
        invalid_extras = self.extras.difference(self.base.dist.iter_provided_extras())
        for extra in sorted(invalid_extras):
            logger.warning(
                "%s %s does not provide the extra '%s'",
                self.base.name,
                self.version,
                extra,
            )

        for r in self.base.dist.iter_dependencies(valid_extras):
            yield from factory.make_requirements_from_spec(
                str(r),
                self._comes_from,
                valid_extras,
            )

    def get_install_requirement(self) -> Optional[InstallRequirement]:
        # We don't return anything here, because we always
        # depend on the base candidate, and we'll get the
        # install requirement from that.
        return None


class RequiresPythonCandidate(Candidate):
    is_installed = False
    source_link = None

    def __init__(self, py_version_info: Optional[Tuple[int, ...]]) -> None:
        if py_version_info is not None:
            version_info = normalize_version_info(py_version_info)
        else:
            version_info = sys.version_info[:3]
        self._version = Version(".".join(str(c) for c in version_info))

    # We don't need to implement __eq__() and __ne__() since there is always
    # only one RequiresPythonCandidate in a resolution, i.e. the host Python.
    # The built-in object.__eq__() and object.__ne__() do exactly what we want.

    def __str__(self) -> str:
        return f"Python {self._version}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._version!r})"

    @property
    def project_name(self) -> NormalizedName:
        return REQUIRES_PYTHON_IDENTIFIER

    @property
    def name(self) -> str:
        return REQUIRES_PYTHON_IDENTIFIER

    @property
    def version(self) -> Version:
        return self._version

    def format_for_error(self) -> str:
        return f"Python {self.version}"

    def iter_dependencies(self, with_requires: bool) -> Iterable[Optional[Requirement]]:
        return ()

    def get_install_requirement(self) -> Optional[InstallRequirement]:
        return None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\web_authn.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: WebAuthn (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class AuthenticatorId(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> AuthenticatorId:
        return cls(json)

    def __repr__(self):
        return 'AuthenticatorId({})'.format(super().__repr__())


class AuthenticatorProtocol(enum.Enum):
    U2F = "u2f"
    CTAP2 = "ctap2"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class Ctap2Version(enum.Enum):
    CTAP2_0 = "ctap2_0"
    CTAP2_1 = "ctap2_1"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AuthenticatorTransport(enum.Enum):
    USB = "usb"
    NFC = "nfc"
    BLE = "ble"
    CABLE = "cable"
    INTERNAL = "internal"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class VirtualAuthenticatorOptions:
    protocol: AuthenticatorProtocol

    transport: AuthenticatorTransport

    #: Defaults to ctap2_0. Ignored if ``protocol`` == u2f.
    ctap2_version: typing.Optional[Ctap2Version] = None

    #: Defaults to false.
    has_resident_key: typing.Optional[bool] = None

    #: Defaults to false.
    has_user_verification: typing.Optional[bool] = None

    #: If set to true, the authenticator will support the largeBlob extension.
    #: https://w3c.github.io/webauthn#largeBlob
    #: Defaults to false.
    has_large_blob: typing.Optional[bool] = None

    #: If set to true, the authenticator will support the credBlob extension.
    #: https://fidoalliance.org/specs/fido-v2.1-rd-20201208/fido-client-to-authenticator-protocol-v2.1-rd-20201208.html#sctn-credBlob-extension
    #: Defaults to false.
    has_cred_blob: typing.Optional[bool] = None

    #: If set to true, the authenticator will support the minPinLength extension.
    #: https://fidoalliance.org/specs/fido-v2.1-ps-20210615/fido-client-to-authenticator-protocol-v2.1-ps-20210615.html#sctn-minpinlength-extension
    #: Defaults to false.
    has_min_pin_length: typing.Optional[bool] = None

    #: If set to true, the authenticator will support the prf extension.
    #: https://w3c.github.io/webauthn/#prf-extension
    #: Defaults to false.
    has_prf: typing.Optional[bool] = None

    #: If set to true, tests of user presence will succeed immediately.
    #: Otherwise, they will not be resolved. Defaults to true.
    automatic_presence_simulation: typing.Optional[bool] = None

    #: Sets whether User Verification succeeds or fails for an authenticator.
    #: Defaults to false.
    is_user_verified: typing.Optional[bool] = None

    #: Credentials created by this authenticator will have the backup
    #: eligibility (BE) flag set to this value. Defaults to false.
    #: https://w3c.github.io/webauthn/#sctn-credential-backup
    default_backup_eligibility: typing.Optional[bool] = None

    #: Credentials created by this authenticator will have the backup state
    #: (BS) flag set to this value. Defaults to false.
    #: https://w3c.github.io/webauthn/#sctn-credential-backup
    default_backup_state: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['protocol'] = self.protocol.to_json()
        json['transport'] = self.transport.to_json()
        if self.ctap2_version is not None:
            json['ctap2Version'] = self.ctap2_version.to_json()
        if self.has_resident_key is not None:
            json['hasResidentKey'] = self.has_resident_key
        if self.has_user_verification is not None:
            json['hasUserVerification'] = self.has_user_verification
        if self.has_large_blob is not None:
            json['hasLargeBlob'] = self.has_large_blob
        if self.has_cred_blob is not None:
            json['hasCredBlob'] = self.has_cred_blob
        if self.has_min_pin_length is not None:
            json['hasMinPinLength'] = self.has_min_pin_length
        if self.has_prf is not None:
            json['hasPrf'] = self.has_prf
        if self.automatic_presence_simulation is not None:
            json['automaticPresenceSimulation'] = self.automatic_presence_simulation
        if self.is_user_verified is not None:
            json['isUserVerified'] = self.is_user_verified
        if self.default_backup_eligibility is not None:
            json['defaultBackupEligibility'] = self.default_backup_eligibility
        if self.default_backup_state is not None:
            json['defaultBackupState'] = self.default_backup_state
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            protocol=AuthenticatorProtocol.from_json(json['protocol']),
            transport=AuthenticatorTransport.from_json(json['transport']),
            ctap2_version=Ctap2Version.from_json(json['ctap2Version']) if 'ctap2Version' in json else None,
            has_resident_key=bool(json['hasResidentKey']) if 'hasResidentKey' in json else None,
            has_user_verification=bool(json['hasUserVerification']) if 'hasUserVerification' in json else None,
            has_large_blob=bool(json['hasLargeBlob']) if 'hasLargeBlob' in json else None,
            has_cred_blob=bool(json['hasCredBlob']) if 'hasCredBlob' in json else None,
            has_min_pin_length=bool(json['hasMinPinLength']) if 'hasMinPinLength' in json else None,
            has_prf=bool(json['hasPrf']) if 'hasPrf' in json else None,
            automatic_presence_simulation=bool(json['automaticPresenceSimulation']) if 'automaticPresenceSimulation' in json else None,
            is_user_verified=bool(json['isUserVerified']) if 'isUserVerified' in json else None,
            default_backup_eligibility=bool(json['defaultBackupEligibility']) if 'defaultBackupEligibility' in json else None,
            default_backup_state=bool(json['defaultBackupState']) if 'defaultBackupState' in json else None,
        )


@dataclass
class Credential:
    credential_id: str

    is_resident_credential: bool

    #: The ECDSA P-256 private key in PKCS#8 format.
    private_key: str

    #: Signature counter. This is incremented by one for each successful
    #: assertion.
    #: See https://w3c.github.io/webauthn/#signature-counter
    sign_count: int

    #: Relying Party ID the credential is scoped to. Must be set when adding a
    #: credential.
    rp_id: typing.Optional[str] = None

    #: An opaque byte sequence with a maximum size of 64 bytes mapping the
    #: credential to a specific user.
    user_handle: typing.Optional[str] = None

    #: The large blob associated with the credential.
    #: See https://w3c.github.io/webauthn/#sctn-large-blob-extension
    large_blob: typing.Optional[str] = None

    #: Assertions returned by this credential will have the backup eligibility
    #: (BE) flag set to this value. Defaults to the authenticator's
    #: defaultBackupEligibility value.
    backup_eligibility: typing.Optional[bool] = None

    #: Assertions returned by this credential will have the backup state (BS)
    #: flag set to this value. Defaults to the authenticator's
    #: defaultBackupState value.
    backup_state: typing.Optional[bool] = None

    #: The credential's user.name property. Equivalent to empty if not set.
    #: https://w3c.github.io/webauthn/#dom-publickeycredentialentity-name
    user_name: typing.Optional[str] = None

    #: The credential's user.displayName property. Equivalent to empty if
    #: not set.
    #: https://w3c.github.io/webauthn/#dom-publickeycredentialuserentity-displayname
    user_display_name: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['credentialId'] = self.credential_id
        json['isResidentCredential'] = self.is_resident_credential
        json['privateKey'] = self.private_key
        json['signCount'] = self.sign_count
        if self.rp_id is not None:
            json['rpId'] = self.rp_id
        if self.user_handle is not None:
            json['userHandle'] = self.user_handle
        if self.large_blob is not None:
            json['largeBlob'] = self.large_blob
        if self.backup_eligibility is not None:
            json['backupEligibility'] = self.backup_eligibility
        if self.backup_state is not None:
            json['backupState'] = self.backup_state
        if self.user_name is not None:
            json['userName'] = self.user_name
        if self.user_display_name is not None:
            json['userDisplayName'] = self.user_display_name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            credential_id=str(json['credentialId']),
            is_resident_credential=bool(json['isResidentCredential']),
            private_key=str(json['privateKey']),
            sign_count=int(json['signCount']),
            rp_id=str(json['rpId']) if 'rpId' in json else None,
            user_handle=str(json['userHandle']) if 'userHandle' in json else None,
            large_blob=str(json['largeBlob']) if 'largeBlob' in json else None,
            backup_eligibility=bool(json['backupEligibility']) if 'backupEligibility' in json else None,
            backup_state=bool(json['backupState']) if 'backupState' in json else None,
            user_name=str(json['userName']) if 'userName' in json else None,
            user_display_name=str(json['userDisplayName']) if 'userDisplayName' in json else None,
        )


def enable(
        enable_ui: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enable the WebAuthn domain and start intercepting credential storage and
    retrieval with a virtual authenticator.

    :param enable_ui: *(Optional)* Whether to enable the WebAuthn user interface. Enabling the UI is recommended for debugging and demo purposes, as it is closer to the real experience. Disabling the UI is recommended for automated testing. Supported at the embedder's discretion if UI is available. Defaults to false.
    '''
    params: T_JSON_DICT = dict()
    if enable_ui is not None:
        params['enableUI'] = enable_ui
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.enable',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disable the WebAuthn domain.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.disable',
    }
    json = yield cmd_dict


def add_virtual_authenticator(
        options: VirtualAuthenticatorOptions
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,AuthenticatorId]:
    '''
    Creates and adds a virtual authenticator.

    :param options:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['options'] = options.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.addVirtualAuthenticator',
        'params': params,
    }
    json = yield cmd_dict
    return AuthenticatorId.from_json(json['authenticatorId'])


def set_response_override_bits(
        authenticator_id: AuthenticatorId,
        is_bogus_signature: typing.Optional[bool] = None,
        is_bad_uv: typing.Optional[bool] = None,
        is_bad_up: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Resets parameters isBogusSignature, isBadUV, isBadUP to false if they are not present.

    :param authenticator_id:
    :param is_bogus_signature: *(Optional)* If isBogusSignature is set, overrides the signature in the authenticator response to be zero. Defaults to false.
    :param is_bad_uv: *(Optional)* If isBadUV is set, overrides the UV bit in the flags in the authenticator response to be zero. Defaults to false.
    :param is_bad_up: *(Optional)* If isBadUP is set, overrides the UP bit in the flags in the authenticator response to be zero. Defaults to false.
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    if is_bogus_signature is not None:
        params['isBogusSignature'] = is_bogus_signature
    if is_bad_uv is not None:
        params['isBadUV'] = is_bad_uv
    if is_bad_up is not None:
        params['isBadUP'] = is_bad_up
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.setResponseOverrideBits',
        'params': params,
    }
    json = yield cmd_dict


def remove_virtual_authenticator(
        authenticator_id: AuthenticatorId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes the given authenticator.

    :param authenticator_id:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.removeVirtualAuthenticator',
        'params': params,
    }
    json = yield cmd_dict


def add_credential(
        authenticator_id: AuthenticatorId,
        credential: Credential
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Adds the credential to the specified authenticator.

    :param authenticator_id:
    :param credential:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['credential'] = credential.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.addCredential',
        'params': params,
    }
    json = yield cmd_dict


def get_credential(
        authenticator_id: AuthenticatorId,
        credential_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,Credential]:
    '''
    Returns a single credential stored in the given virtual authenticator that
    matches the credential ID.

    :param authenticator_id:
    :param credential_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['credentialId'] = credential_id
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.getCredential',
        'params': params,
    }
    json = yield cmd_dict
    return Credential.from_json(json['credential'])


def get_credentials(
        authenticator_id: AuthenticatorId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[Credential]]:
    '''
    Returns all the credentials stored in the given virtual authenticator.

    :param authenticator_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.getCredentials',
        'params': params,
    }
    json = yield cmd_dict
    return [Credential.from_json(i) for i in json['credentials']]


def remove_credential(
        authenticator_id: AuthenticatorId,
        credential_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes a credential from the authenticator.

    :param authenticator_id:
    :param credential_id:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['credentialId'] = credential_id
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.removeCredential',
        'params': params,
    }
    json = yield cmd_dict


def clear_credentials(
        authenticator_id: AuthenticatorId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears all the credentials from the specified device.

    :param authenticator_id:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.clearCredentials',
        'params': params,
    }
    json = yield cmd_dict


def set_user_verified(
        authenticator_id: AuthenticatorId,
        is_user_verified: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets whether User Verification succeeds or fails for an authenticator.
    The default is true.

    :param authenticator_id:
    :param is_user_verified:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['isUserVerified'] = is_user_verified
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.setUserVerified',
        'params': params,
    }
    json = yield cmd_dict


def set_automatic_presence_simulation(
        authenticator_id: AuthenticatorId,
        enabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets whether tests of user presence will succeed immediately (if true) or fail to resolve (if false) for an authenticator.
    The default is true.

    :param authenticator_id:
    :param enabled:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.setAutomaticPresenceSimulation',
        'params': params,
    }
    json = yield cmd_dict


def set_credential_properties(
        authenticator_id: AuthenticatorId,
        credential_id: str,
        backup_eligibility: typing.Optional[bool] = None,
        backup_state: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Allows setting credential properties.
    https://w3c.github.io/webauthn/#sctn-automation-set-credential-properties

    :param authenticator_id:
    :param credential_id:
    :param backup_eligibility: *(Optional)*
    :param backup_state: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['credentialId'] = credential_id
    if backup_eligibility is not None:
        params['backupEligibility'] = backup_eligibility
    if backup_state is not None:
        params['backupState'] = backup_state
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.setCredentialProperties',
        'params': params,
    }
    json = yield cmd_dict


@event_class('WebAuthn.credentialAdded')
@dataclass
class CredentialAdded:
    '''
    Triggered when a credential is added to an authenticator.
    '''
    authenticator_id: AuthenticatorId
    credential: Credential

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CredentialAdded:
        return cls(
            authenticator_id=AuthenticatorId.from_json(json['authenticatorId']),
            credential=Credential.from_json(json['credential'])
        )


@event_class('WebAuthn.credentialDeleted')
@dataclass
class CredentialDeleted:
    '''
    Triggered when a credential is deleted, e.g. through
    PublicKeyCredential.signalUnknownCredential().
    '''
    authenticator_id: AuthenticatorId
    credential_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CredentialDeleted:
        return cls(
            authenticator_id=AuthenticatorId.from_json(json['authenticatorId']),
            credential_id=str(json['credentialId'])
        )


@event_class('WebAuthn.credentialUpdated')
@dataclass
class CredentialUpdated:
    '''
    Triggered when a credential is updated, e.g. through
    PublicKeyCredential.signalCurrentUserDetails().
    '''
    authenticator_id: AuthenticatorId
    credential: Credential

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CredentialUpdated:
        return cls(
            authenticator_id=AuthenticatorId.from_json(json['authenticatorId']),
            credential=Credential.from_json(json['credential'])
        )


@event_class('WebAuthn.credentialAsserted')
@dataclass
class CredentialAsserted:
    '''
    Triggered when a credential is used in a webauthn assertion.
    '''
    authenticator_id: AuthenticatorId
    credential: Credential

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CredentialAsserted:
        return cls(
            authenticator_id=AuthenticatorId.from_json(json['authenticatorId']),
            credential=Credential.from_json(json['credential'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\web_authn.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: WebAuthn (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class AuthenticatorId(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> AuthenticatorId:
        return cls(json)

    def __repr__(self):
        return 'AuthenticatorId({})'.format(super().__repr__())


class AuthenticatorProtocol(enum.Enum):
    U2F = "u2f"
    CTAP2 = "ctap2"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class Ctap2Version(enum.Enum):
    CTAP2_0 = "ctap2_0"
    CTAP2_1 = "ctap2_1"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AuthenticatorTransport(enum.Enum):
    USB = "usb"
    NFC = "nfc"
    BLE = "ble"
    CABLE = "cable"
    INTERNAL = "internal"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class VirtualAuthenticatorOptions:
    protocol: AuthenticatorProtocol

    transport: AuthenticatorTransport

    #: Defaults to ctap2_0. Ignored if ``protocol`` == u2f.
    ctap2_version: typing.Optional[Ctap2Version] = None

    #: Defaults to false.
    has_resident_key: typing.Optional[bool] = None

    #: Defaults to false.
    has_user_verification: typing.Optional[bool] = None

    #: If set to true, the authenticator will support the largeBlob extension.
    #: https://w3c.github.io/webauthn#largeBlob
    #: Defaults to false.
    has_large_blob: typing.Optional[bool] = None

    #: If set to true, the authenticator will support the credBlob extension.
    #: https://fidoalliance.org/specs/fido-v2.1-rd-20201208/fido-client-to-authenticator-protocol-v2.1-rd-20201208.html#sctn-credBlob-extension
    #: Defaults to false.
    has_cred_blob: typing.Optional[bool] = None

    #: If set to true, the authenticator will support the minPinLength extension.
    #: https://fidoalliance.org/specs/fido-v2.1-ps-20210615/fido-client-to-authenticator-protocol-v2.1-ps-20210615.html#sctn-minpinlength-extension
    #: Defaults to false.
    has_min_pin_length: typing.Optional[bool] = None

    #: If set to true, the authenticator will support the prf extension.
    #: https://w3c.github.io/webauthn/#prf-extension
    #: Defaults to false.
    has_prf: typing.Optional[bool] = None

    #: If set to true, tests of user presence will succeed immediately.
    #: Otherwise, they will not be resolved. Defaults to true.
    automatic_presence_simulation: typing.Optional[bool] = None

    #: Sets whether User Verification succeeds or fails for an authenticator.
    #: Defaults to false.
    is_user_verified: typing.Optional[bool] = None

    #: Credentials created by this authenticator will have the backup
    #: eligibility (BE) flag set to this value. Defaults to false.
    #: https://w3c.github.io/webauthn/#sctn-credential-backup
    default_backup_eligibility: typing.Optional[bool] = None

    #: Credentials created by this authenticator will have the backup state
    #: (BS) flag set to this value. Defaults to false.
    #: https://w3c.github.io/webauthn/#sctn-credential-backup
    default_backup_state: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['protocol'] = self.protocol.to_json()
        json['transport'] = self.transport.to_json()
        if self.ctap2_version is not None:
            json['ctap2Version'] = self.ctap2_version.to_json()
        if self.has_resident_key is not None:
            json['hasResidentKey'] = self.has_resident_key
        if self.has_user_verification is not None:
            json['hasUserVerification'] = self.has_user_verification
        if self.has_large_blob is not None:
            json['hasLargeBlob'] = self.has_large_blob
        if self.has_cred_blob is not None:
            json['hasCredBlob'] = self.has_cred_blob
        if self.has_min_pin_length is not None:
            json['hasMinPinLength'] = self.has_min_pin_length
        if self.has_prf is not None:
            json['hasPrf'] = self.has_prf
        if self.automatic_presence_simulation is not None:
            json['automaticPresenceSimulation'] = self.automatic_presence_simulation
        if self.is_user_verified is not None:
            json['isUserVerified'] = self.is_user_verified
        if self.default_backup_eligibility is not None:
            json['defaultBackupEligibility'] = self.default_backup_eligibility
        if self.default_backup_state is not None:
            json['defaultBackupState'] = self.default_backup_state
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            protocol=AuthenticatorProtocol.from_json(json['protocol']),
            transport=AuthenticatorTransport.from_json(json['transport']),
            ctap2_version=Ctap2Version.from_json(json['ctap2Version']) if 'ctap2Version' in json else None,
            has_resident_key=bool(json['hasResidentKey']) if 'hasResidentKey' in json else None,
            has_user_verification=bool(json['hasUserVerification']) if 'hasUserVerification' in json else None,
            has_large_blob=bool(json['hasLargeBlob']) if 'hasLargeBlob' in json else None,
            has_cred_blob=bool(json['hasCredBlob']) if 'hasCredBlob' in json else None,
            has_min_pin_length=bool(json['hasMinPinLength']) if 'hasMinPinLength' in json else None,
            has_prf=bool(json['hasPrf']) if 'hasPrf' in json else None,
            automatic_presence_simulation=bool(json['automaticPresenceSimulation']) if 'automaticPresenceSimulation' in json else None,
            is_user_verified=bool(json['isUserVerified']) if 'isUserVerified' in json else None,
            default_backup_eligibility=bool(json['defaultBackupEligibility']) if 'defaultBackupEligibility' in json else None,
            default_backup_state=bool(json['defaultBackupState']) if 'defaultBackupState' in json else None,
        )


@dataclass
class Credential:
    credential_id: str

    is_resident_credential: bool

    #: The ECDSA P-256 private key in PKCS#8 format.
    private_key: str

    #: Signature counter. This is incremented by one for each successful
    #: assertion.
    #: See https://w3c.github.io/webauthn/#signature-counter
    sign_count: int

    #: Relying Party ID the credential is scoped to. Must be set when adding a
    #: credential.
    rp_id: typing.Optional[str] = None

    #: An opaque byte sequence with a maximum size of 64 bytes mapping the
    #: credential to a specific user.
    user_handle: typing.Optional[str] = None

    #: The large blob associated with the credential.
    #: See https://w3c.github.io/webauthn/#sctn-large-blob-extension
    large_blob: typing.Optional[str] = None

    #: Assertions returned by this credential will have the backup eligibility
    #: (BE) flag set to this value. Defaults to the authenticator's
    #: defaultBackupEligibility value.
    backup_eligibility: typing.Optional[bool] = None

    #: Assertions returned by this credential will have the backup state (BS)
    #: flag set to this value. Defaults to the authenticator's
    #: defaultBackupState value.
    backup_state: typing.Optional[bool] = None

    #: The credential's user.name property. Equivalent to empty if not set.
    #: https://w3c.github.io/webauthn/#dom-publickeycredentialentity-name
    user_name: typing.Optional[str] = None

    #: The credential's user.displayName property. Equivalent to empty if
    #: not set.
    #: https://w3c.github.io/webauthn/#dom-publickeycredentialuserentity-displayname
    user_display_name: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['credentialId'] = self.credential_id
        json['isResidentCredential'] = self.is_resident_credential
        json['privateKey'] = self.private_key
        json['signCount'] = self.sign_count
        if self.rp_id is not None:
            json['rpId'] = self.rp_id
        if self.user_handle is not None:
            json['userHandle'] = self.user_handle
        if self.large_blob is not None:
            json['largeBlob'] = self.large_blob
        if self.backup_eligibility is not None:
            json['backupEligibility'] = self.backup_eligibility
        if self.backup_state is not None:
            json['backupState'] = self.backup_state
        if self.user_name is not None:
            json['userName'] = self.user_name
        if self.user_display_name is not None:
            json['userDisplayName'] = self.user_display_name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            credential_id=str(json['credentialId']),
            is_resident_credential=bool(json['isResidentCredential']),
            private_key=str(json['privateKey']),
            sign_count=int(json['signCount']),
            rp_id=str(json['rpId']) if 'rpId' in json else None,
            user_handle=str(json['userHandle']) if 'userHandle' in json else None,
            large_blob=str(json['largeBlob']) if 'largeBlob' in json else None,
            backup_eligibility=bool(json['backupEligibility']) if 'backupEligibility' in json else None,
            backup_state=bool(json['backupState']) if 'backupState' in json else None,
            user_name=str(json['userName']) if 'userName' in json else None,
            user_display_name=str(json['userDisplayName']) if 'userDisplayName' in json else None,
        )


def enable(
        enable_ui: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enable the WebAuthn domain and start intercepting credential storage and
    retrieval with a virtual authenticator.

    :param enable_ui: *(Optional)* Whether to enable the WebAuthn user interface. Enabling the UI is recommended for debugging and demo purposes, as it is closer to the real experience. Disabling the UI is recommended for automated testing. Supported at the embedder's discretion if UI is available. Defaults to false.
    '''
    params: T_JSON_DICT = dict()
    if enable_ui is not None:
        params['enableUI'] = enable_ui
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.enable',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disable the WebAuthn domain.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.disable',
    }
    json = yield cmd_dict


def add_virtual_authenticator(
        options: VirtualAuthenticatorOptions
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,AuthenticatorId]:
    '''
    Creates and adds a virtual authenticator.

    :param options:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['options'] = options.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.addVirtualAuthenticator',
        'params': params,
    }
    json = yield cmd_dict
    return AuthenticatorId.from_json(json['authenticatorId'])


def set_response_override_bits(
        authenticator_id: AuthenticatorId,
        is_bogus_signature: typing.Optional[bool] = None,
        is_bad_uv: typing.Optional[bool] = None,
        is_bad_up: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Resets parameters isBogusSignature, isBadUV, isBadUP to false if they are not present.

    :param authenticator_id:
    :param is_bogus_signature: *(Optional)* If isBogusSignature is set, overrides the signature in the authenticator response to be zero. Defaults to false.
    :param is_bad_uv: *(Optional)* If isBadUV is set, overrides the UV bit in the flags in the authenticator response to be zero. Defaults to false.
    :param is_bad_up: *(Optional)* If isBadUP is set, overrides the UP bit in the flags in the authenticator response to be zero. Defaults to false.
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    if is_bogus_signature is not None:
        params['isBogusSignature'] = is_bogus_signature
    if is_bad_uv is not None:
        params['isBadUV'] = is_bad_uv
    if is_bad_up is not None:
        params['isBadUP'] = is_bad_up
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.setResponseOverrideBits',
        'params': params,
    }
    json = yield cmd_dict


def remove_virtual_authenticator(
        authenticator_id: AuthenticatorId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes the given authenticator.

    :param authenticator_id:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.removeVirtualAuthenticator',
        'params': params,
    }
    json = yield cmd_dict


def add_credential(
        authenticator_id: AuthenticatorId,
        credential: Credential
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Adds the credential to the specified authenticator.

    :param authenticator_id:
    :param credential:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['credential'] = credential.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.addCredential',
        'params': params,
    }
    json = yield cmd_dict


def get_credential(
        authenticator_id: AuthenticatorId,
        credential_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,Credential]:
    '''
    Returns a single credential stored in the given virtual authenticator that
    matches the credential ID.

    :param authenticator_id:
    :param credential_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['credentialId'] = credential_id
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.getCredential',
        'params': params,
    }
    json = yield cmd_dict
    return Credential.from_json(json['credential'])


def get_credentials(
        authenticator_id: AuthenticatorId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[Credential]]:
    '''
    Returns all the credentials stored in the given virtual authenticator.

    :param authenticator_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.getCredentials',
        'params': params,
    }
    json = yield cmd_dict
    return [Credential.from_json(i) for i in json['credentials']]


def remove_credential(
        authenticator_id: AuthenticatorId,
        credential_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes a credential from the authenticator.

    :param authenticator_id:
    :param credential_id:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['credentialId'] = credential_id
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.removeCredential',
        'params': params,
    }
    json = yield cmd_dict


def clear_credentials(
        authenticator_id: AuthenticatorId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears all the credentials from the specified device.

    :param authenticator_id:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.clearCredentials',
        'params': params,
    }
    json = yield cmd_dict


def set_user_verified(
        authenticator_id: AuthenticatorId,
        is_user_verified: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets whether User Verification succeeds or fails for an authenticator.
    The default is true.

    :param authenticator_id:
    :param is_user_verified:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['isUserVerified'] = is_user_verified
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.setUserVerified',
        'params': params,
    }
    json = yield cmd_dict


def set_automatic_presence_simulation(
        authenticator_id: AuthenticatorId,
        enabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets whether tests of user presence will succeed immediately (if true) or fail to resolve (if false) for an authenticator.
    The default is true.

    :param authenticator_id:
    :param enabled:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.setAutomaticPresenceSimulation',
        'params': params,
    }
    json = yield cmd_dict


def set_credential_properties(
        authenticator_id: AuthenticatorId,
        credential_id: str,
        backup_eligibility: typing.Optional[bool] = None,
        backup_state: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Allows setting credential properties.
    https://w3c.github.io/webauthn/#sctn-automation-set-credential-properties

    :param authenticator_id:
    :param credential_id:
    :param backup_eligibility: *(Optional)*
    :param backup_state: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['credentialId'] = credential_id
    if backup_eligibility is not None:
        params['backupEligibility'] = backup_eligibility
    if backup_state is not None:
        params['backupState'] = backup_state
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.setCredentialProperties',
        'params': params,
    }
    json = yield cmd_dict


@event_class('WebAuthn.credentialAdded')
@dataclass
class CredentialAdded:
    '''
    Triggered when a credential is added to an authenticator.
    '''
    authenticator_id: AuthenticatorId
    credential: Credential

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CredentialAdded:
        return cls(
            authenticator_id=AuthenticatorId.from_json(json['authenticatorId']),
            credential=Credential.from_json(json['credential'])
        )


@event_class('WebAuthn.credentialDeleted')
@dataclass
class CredentialDeleted:
    '''
    Triggered when a credential is deleted, e.g. through
    PublicKeyCredential.signalUnknownCredential().
    '''
    authenticator_id: AuthenticatorId
    credential_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CredentialDeleted:
        return cls(
            authenticator_id=AuthenticatorId.from_json(json['authenticatorId']),
            credential_id=str(json['credentialId'])
        )


@event_class('WebAuthn.credentialUpdated')
@dataclass
class CredentialUpdated:
    '''
    Triggered when a credential is updated, e.g. through
    PublicKeyCredential.signalCurrentUserDetails().
    '''
    authenticator_id: AuthenticatorId
    credential: Credential

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CredentialUpdated:
        return cls(
            authenticator_id=AuthenticatorId.from_json(json['authenticatorId']),
            credential=Credential.from_json(json['credential'])
        )


@event_class('WebAuthn.credentialAsserted')
@dataclass
class CredentialAsserted:
    '''
    Triggered when a credential is used in a webauthn assertion.
    '''
    authenticator_id: AuthenticatorId
    credential: Credential

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CredentialAsserted:
        return cls(
            authenticator_id=AuthenticatorId.from_json(json['authenticatorId']),
            credential=Credential.from_json(json['credential'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\web_authn.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: WebAuthn (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class AuthenticatorId(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> AuthenticatorId:
        return cls(json)

    def __repr__(self):
        return 'AuthenticatorId({})'.format(super().__repr__())


class AuthenticatorProtocol(enum.Enum):
    U2F = "u2f"
    CTAP2 = "ctap2"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class Ctap2Version(enum.Enum):
    CTAP2_0 = "ctap2_0"
    CTAP2_1 = "ctap2_1"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AuthenticatorTransport(enum.Enum):
    USB = "usb"
    NFC = "nfc"
    BLE = "ble"
    CABLE = "cable"
    INTERNAL = "internal"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class VirtualAuthenticatorOptions:
    protocol: AuthenticatorProtocol

    transport: AuthenticatorTransport

    #: Defaults to ctap2_0. Ignored if ``protocol`` == u2f.
    ctap2_version: typing.Optional[Ctap2Version] = None

    #: Defaults to false.
    has_resident_key: typing.Optional[bool] = None

    #: Defaults to false.
    has_user_verification: typing.Optional[bool] = None

    #: If set to true, the authenticator will support the largeBlob extension.
    #: https://w3c.github.io/webauthn#largeBlob
    #: Defaults to false.
    has_large_blob: typing.Optional[bool] = None

    #: If set to true, the authenticator will support the credBlob extension.
    #: https://fidoalliance.org/specs/fido-v2.1-rd-20201208/fido-client-to-authenticator-protocol-v2.1-rd-20201208.html#sctn-credBlob-extension
    #: Defaults to false.
    has_cred_blob: typing.Optional[bool] = None

    #: If set to true, the authenticator will support the minPinLength extension.
    #: https://fidoalliance.org/specs/fido-v2.1-ps-20210615/fido-client-to-authenticator-protocol-v2.1-ps-20210615.html#sctn-minpinlength-extension
    #: Defaults to false.
    has_min_pin_length: typing.Optional[bool] = None

    #: If set to true, the authenticator will support the prf extension.
    #: https://w3c.github.io/webauthn/#prf-extension
    #: Defaults to false.
    has_prf: typing.Optional[bool] = None

    #: If set to true, tests of user presence will succeed immediately.
    #: Otherwise, they will not be resolved. Defaults to true.
    automatic_presence_simulation: typing.Optional[bool] = None

    #: Sets whether User Verification succeeds or fails for an authenticator.
    #: Defaults to false.
    is_user_verified: typing.Optional[bool] = None

    #: Credentials created by this authenticator will have the backup
    #: eligibility (BE) flag set to this value. Defaults to false.
    #: https://w3c.github.io/webauthn/#sctn-credential-backup
    default_backup_eligibility: typing.Optional[bool] = None

    #: Credentials created by this authenticator will have the backup state
    #: (BS) flag set to this value. Defaults to false.
    #: https://w3c.github.io/webauthn/#sctn-credential-backup
    default_backup_state: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['protocol'] = self.protocol.to_json()
        json['transport'] = self.transport.to_json()
        if self.ctap2_version is not None:
            json['ctap2Version'] = self.ctap2_version.to_json()
        if self.has_resident_key is not None:
            json['hasResidentKey'] = self.has_resident_key
        if self.has_user_verification is not None:
            json['hasUserVerification'] = self.has_user_verification
        if self.has_large_blob is not None:
            json['hasLargeBlob'] = self.has_large_blob
        if self.has_cred_blob is not None:
            json['hasCredBlob'] = self.has_cred_blob
        if self.has_min_pin_length is not None:
            json['hasMinPinLength'] = self.has_min_pin_length
        if self.has_prf is not None:
            json['hasPrf'] = self.has_prf
        if self.automatic_presence_simulation is not None:
            json['automaticPresenceSimulation'] = self.automatic_presence_simulation
        if self.is_user_verified is not None:
            json['isUserVerified'] = self.is_user_verified
        if self.default_backup_eligibility is not None:
            json['defaultBackupEligibility'] = self.default_backup_eligibility
        if self.default_backup_state is not None:
            json['defaultBackupState'] = self.default_backup_state
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            protocol=AuthenticatorProtocol.from_json(json['protocol']),
            transport=AuthenticatorTransport.from_json(json['transport']),
            ctap2_version=Ctap2Version.from_json(json['ctap2Version']) if 'ctap2Version' in json else None,
            has_resident_key=bool(json['hasResidentKey']) if 'hasResidentKey' in json else None,
            has_user_verification=bool(json['hasUserVerification']) if 'hasUserVerification' in json else None,
            has_large_blob=bool(json['hasLargeBlob']) if 'hasLargeBlob' in json else None,
            has_cred_blob=bool(json['hasCredBlob']) if 'hasCredBlob' in json else None,
            has_min_pin_length=bool(json['hasMinPinLength']) if 'hasMinPinLength' in json else None,
            has_prf=bool(json['hasPrf']) if 'hasPrf' in json else None,
            automatic_presence_simulation=bool(json['automaticPresenceSimulation']) if 'automaticPresenceSimulation' in json else None,
            is_user_verified=bool(json['isUserVerified']) if 'isUserVerified' in json else None,
            default_backup_eligibility=bool(json['defaultBackupEligibility']) if 'defaultBackupEligibility' in json else None,
            default_backup_state=bool(json['defaultBackupState']) if 'defaultBackupState' in json else None,
        )


@dataclass
class Credential:
    credential_id: str

    is_resident_credential: bool

    #: The ECDSA P-256 private key in PKCS#8 format.
    private_key: str

    #: Signature counter. This is incremented by one for each successful
    #: assertion.
    #: See https://w3c.github.io/webauthn/#signature-counter
    sign_count: int

    #: Relying Party ID the credential is scoped to. Must be set when adding a
    #: credential.
    rp_id: typing.Optional[str] = None

    #: An opaque byte sequence with a maximum size of 64 bytes mapping the
    #: credential to a specific user.
    user_handle: typing.Optional[str] = None

    #: The large blob associated with the credential.
    #: See https://w3c.github.io/webauthn/#sctn-large-blob-extension
    large_blob: typing.Optional[str] = None

    #: Assertions returned by this credential will have the backup eligibility
    #: (BE) flag set to this value. Defaults to the authenticator's
    #: defaultBackupEligibility value.
    backup_eligibility: typing.Optional[bool] = None

    #: Assertions returned by this credential will have the backup state (BS)
    #: flag set to this value. Defaults to the authenticator's
    #: defaultBackupState value.
    backup_state: typing.Optional[bool] = None

    #: The credential's user.name property. Equivalent to empty if not set.
    #: https://w3c.github.io/webauthn/#dom-publickeycredentialentity-name
    user_name: typing.Optional[str] = None

    #: The credential's user.displayName property. Equivalent to empty if
    #: not set.
    #: https://w3c.github.io/webauthn/#dom-publickeycredentialuserentity-displayname
    user_display_name: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['credentialId'] = self.credential_id
        json['isResidentCredential'] = self.is_resident_credential
        json['privateKey'] = self.private_key
        json['signCount'] = self.sign_count
        if self.rp_id is not None:
            json['rpId'] = self.rp_id
        if self.user_handle is not None:
            json['userHandle'] = self.user_handle
        if self.large_blob is not None:
            json['largeBlob'] = self.large_blob
        if self.backup_eligibility is not None:
            json['backupEligibility'] = self.backup_eligibility
        if self.backup_state is not None:
            json['backupState'] = self.backup_state
        if self.user_name is not None:
            json['userName'] = self.user_name
        if self.user_display_name is not None:
            json['userDisplayName'] = self.user_display_name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            credential_id=str(json['credentialId']),
            is_resident_credential=bool(json['isResidentCredential']),
            private_key=str(json['privateKey']),
            sign_count=int(json['signCount']),
            rp_id=str(json['rpId']) if 'rpId' in json else None,
            user_handle=str(json['userHandle']) if 'userHandle' in json else None,
            large_blob=str(json['largeBlob']) if 'largeBlob' in json else None,
            backup_eligibility=bool(json['backupEligibility']) if 'backupEligibility' in json else None,
            backup_state=bool(json['backupState']) if 'backupState' in json else None,
            user_name=str(json['userName']) if 'userName' in json else None,
            user_display_name=str(json['userDisplayName']) if 'userDisplayName' in json else None,
        )


def enable(
        enable_ui: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enable the WebAuthn domain and start intercepting credential storage and
    retrieval with a virtual authenticator.

    :param enable_ui: *(Optional)* Whether to enable the WebAuthn user interface. Enabling the UI is recommended for debugging and demo purposes, as it is closer to the real experience. Disabling the UI is recommended for automated testing. Supported at the embedder's discretion if UI is available. Defaults to false.
    '''
    params: T_JSON_DICT = dict()
    if enable_ui is not None:
        params['enableUI'] = enable_ui
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.enable',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disable the WebAuthn domain.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.disable',
    }
    json = yield cmd_dict


def add_virtual_authenticator(
        options: VirtualAuthenticatorOptions
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,AuthenticatorId]:
    '''
    Creates and adds a virtual authenticator.

    :param options:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['options'] = options.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.addVirtualAuthenticator',
        'params': params,
    }
    json = yield cmd_dict
    return AuthenticatorId.from_json(json['authenticatorId'])


def set_response_override_bits(
        authenticator_id: AuthenticatorId,
        is_bogus_signature: typing.Optional[bool] = None,
        is_bad_uv: typing.Optional[bool] = None,
        is_bad_up: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Resets parameters isBogusSignature, isBadUV, isBadUP to false if they are not present.

    :param authenticator_id:
    :param is_bogus_signature: *(Optional)* If isBogusSignature is set, overrides the signature in the authenticator response to be zero. Defaults to false.
    :param is_bad_uv: *(Optional)* If isBadUV is set, overrides the UV bit in the flags in the authenticator response to be zero. Defaults to false.
    :param is_bad_up: *(Optional)* If isBadUP is set, overrides the UP bit in the flags in the authenticator response to be zero. Defaults to false.
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    if is_bogus_signature is not None:
        params['isBogusSignature'] = is_bogus_signature
    if is_bad_uv is not None:
        params['isBadUV'] = is_bad_uv
    if is_bad_up is not None:
        params['isBadUP'] = is_bad_up
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.setResponseOverrideBits',
        'params': params,
    }
    json = yield cmd_dict


def remove_virtual_authenticator(
        authenticator_id: AuthenticatorId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes the given authenticator.

    :param authenticator_id:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.removeVirtualAuthenticator',
        'params': params,
    }
    json = yield cmd_dict


def add_credential(
        authenticator_id: AuthenticatorId,
        credential: Credential
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Adds the credential to the specified authenticator.

    :param authenticator_id:
    :param credential:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['credential'] = credential.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.addCredential',
        'params': params,
    }
    json = yield cmd_dict


def get_credential(
        authenticator_id: AuthenticatorId,
        credential_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,Credential]:
    '''
    Returns a single credential stored in the given virtual authenticator that
    matches the credential ID.

    :param authenticator_id:
    :param credential_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['credentialId'] = credential_id
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.getCredential',
        'params': params,
    }
    json = yield cmd_dict
    return Credential.from_json(json['credential'])


def get_credentials(
        authenticator_id: AuthenticatorId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[Credential]]:
    '''
    Returns all the credentials stored in the given virtual authenticator.

    :param authenticator_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.getCredentials',
        'params': params,
    }
    json = yield cmd_dict
    return [Credential.from_json(i) for i in json['credentials']]


def remove_credential(
        authenticator_id: AuthenticatorId,
        credential_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes a credential from the authenticator.

    :param authenticator_id:
    :param credential_id:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['credentialId'] = credential_id
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.removeCredential',
        'params': params,
    }
    json = yield cmd_dict


def clear_credentials(
        authenticator_id: AuthenticatorId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears all the credentials from the specified device.

    :param authenticator_id:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.clearCredentials',
        'params': params,
    }
    json = yield cmd_dict


def set_user_verified(
        authenticator_id: AuthenticatorId,
        is_user_verified: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets whether User Verification succeeds or fails for an authenticator.
    The default is true.

    :param authenticator_id:
    :param is_user_verified:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['isUserVerified'] = is_user_verified
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.setUserVerified',
        'params': params,
    }
    json = yield cmd_dict


def set_automatic_presence_simulation(
        authenticator_id: AuthenticatorId,
        enabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets whether tests of user presence will succeed immediately (if true) or fail to resolve (if false) for an authenticator.
    The default is true.

    :param authenticator_id:
    :param enabled:
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.setAutomaticPresenceSimulation',
        'params': params,
    }
    json = yield cmd_dict


def set_credential_properties(
        authenticator_id: AuthenticatorId,
        credential_id: str,
        backup_eligibility: typing.Optional[bool] = None,
        backup_state: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Allows setting credential properties.
    https://w3c.github.io/webauthn/#sctn-automation-set-credential-properties

    :param authenticator_id:
    :param credential_id:
    :param backup_eligibility: *(Optional)*
    :param backup_state: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['credentialId'] = credential_id
    if backup_eligibility is not None:
        params['backupEligibility'] = backup_eligibility
    if backup_state is not None:
        params['backupState'] = backup_state
    cmd_dict: T_JSON_DICT = {
        'method': 'WebAuthn.setCredentialProperties',
        'params': params,
    }
    json = yield cmd_dict


@event_class('WebAuthn.credentialAdded')
@dataclass
class CredentialAdded:
    '''
    Triggered when a credential is added to an authenticator.
    '''
    authenticator_id: AuthenticatorId
    credential: Credential

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CredentialAdded:
        return cls(
            authenticator_id=AuthenticatorId.from_json(json['authenticatorId']),
            credential=Credential.from_json(json['credential'])
        )


@event_class('WebAuthn.credentialDeleted')
@dataclass
class CredentialDeleted:
    '''
    Triggered when a credential is deleted, e.g. through
    PublicKeyCredential.signalUnknownCredential().
    '''
    authenticator_id: AuthenticatorId
    credential_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CredentialDeleted:
        return cls(
            authenticator_id=AuthenticatorId.from_json(json['authenticatorId']),
            credential_id=str(json['credentialId'])
        )


@event_class('WebAuthn.credentialUpdated')
@dataclass
class CredentialUpdated:
    '''
    Triggered when a credential is updated, e.g. through
    PublicKeyCredential.signalCurrentUserDetails().
    '''
    authenticator_id: AuthenticatorId
    credential: Credential

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CredentialUpdated:
        return cls(
            authenticator_id=AuthenticatorId.from_json(json['authenticatorId']),
            credential=Credential.from_json(json['credential'])
        )


@event_class('WebAuthn.credentialAsserted')
@dataclass
class CredentialAsserted:
    '''
    Triggered when a credential is used in a webauthn assertion.
    '''
    authenticator_id: AuthenticatorId
    credential: Credential

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CredentialAsserted:
        return cls(
            authenticator_id=AuthenticatorId.from_json(json['authenticatorId']),
            credential=Credential.from_json(json['credential'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\plotting\_matplotlib\boxplot.py ===
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
)
import warnings

import matplotlib as mpl
from matplotlib.artist import setp
import numpy as np

from pandas._libs import lib
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import is_dict_like
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import remove_na_arraylike

import pandas as pd
import pandas.core.common as com
from pandas.util.version import Version

from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
    LinePlot,
    MPLPlot,
)
from pandas.plotting._matplotlib.groupby import create_iter_data_given_by
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.tools import (
    create_subplots,
    flatten_axes,
    maybe_adjust_figure,
)

if TYPE_CHECKING:
    from collections.abc import Collection

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from pandas._typing import MatplotlibColor


def _set_ticklabels(ax: Axes, labels: list[str], is_vertical: bool, **kwargs) -> None:
    """Set the tick labels of a given axis.

    Due to https://github.com/matplotlib/matplotlib/pull/17266, we need to handle the
    case of repeated ticks (due to `FixedLocator`) and thus we duplicate the number of
    labels.
    """
    ticks = ax.get_xticks() if is_vertical else ax.get_yticks()
    if len(ticks) != len(labels):
        i, remainder = divmod(len(ticks), len(labels))
        if Version(mpl.__version__) < Version("3.10"):
            assert remainder == 0, remainder
        labels *= i
    if is_vertical:
        ax.set_xticklabels(labels, **kwargs)
    else:
        ax.set_yticklabels(labels, **kwargs)


class BoxPlot(LinePlot):
    @property
    def _kind(self) -> Literal["box"]:
        return "box"

    _layout_type = "horizontal"

    _valid_return_types = (None, "axes", "dict", "both")

    class BP(NamedTuple):
        # namedtuple to hold results
        ax: Axes
        lines: dict[str, list[Line2D]]

    def __init__(self, data, return_type: str = "axes", **kwargs) -> None:
        if return_type not in self._valid_return_types:
            raise ValueError("return_type must be {None, 'axes', 'dict', 'both'}")

        self.return_type = return_type
        # Do not call LinePlot.__init__ which may fill nan
        MPLPlot.__init__(self, data, **kwargs)  # pylint: disable=non-parent-init-called

        if self.subplots:
            # Disable label ax sharing. Otherwise, all subplots shows last
            # column label
            if self.orientation == "vertical":
                self.sharex = False
            else:
                self.sharey = False

    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls, ax: Axes, y: np.ndarray, column_num=None, return_type: str = "axes", **kwds
    ):
        ys: np.ndarray | list[np.ndarray]
        if y.ndim == 2:
            ys = [remove_na_arraylike(v) for v in y]
            # Boxplot fails with empty arrays, so need to add a NaN
            #   if any cols are empty
            # GH 8181
            ys = [v if v.size > 0 else np.array([np.nan]) for v in ys]
        else:
            ys = remove_na_arraylike(y)
        bp = ax.boxplot(ys, **kwds)

        if return_type == "dict":
            return bp, bp
        elif return_type == "both":
            return cls.BP(ax=ax, lines=bp), bp
        else:
            return ax, bp

    def _validate_color_args(self, color, colormap):
        if color is lib.no_default:
            return None

        if colormap is not None:
            warnings.warn(
                "'color' and 'colormap' cannot be used "
                "simultaneously. Using 'color'",
                stacklevel=find_stack_level(),
            )

        if isinstance(color, dict):
            valid_keys = ["boxes", "whiskers", "medians", "caps"]
            for key in color:
                if key not in valid_keys:
                    raise ValueError(
                        f"color dict contains invalid key '{key}'. "
                        f"The key must be either {valid_keys}"
                    )
        return color

    @cache_readonly
    def _color_attrs(self):
        # get standard colors for default
        # use 2 colors by default, for box/whisker and median
        # flier colors isn't needed here
        # because it can be specified by ``sym`` kw
        return get_standard_colors(num_colors=3, colormap=self.colormap, color=None)

    @cache_readonly
    def _boxes_c(self):
        return self._color_attrs[0]

    @cache_readonly
    def _whiskers_c(self):
        return self._color_attrs[0]

    @cache_readonly
    def _medians_c(self):
        return self._color_attrs[2]

    @cache_readonly
    def _caps_c(self):
        return self._color_attrs[0]

    def _get_colors(
        self,
        num_colors=None,
        color_kwds: dict[str, MatplotlibColor]
        | MatplotlibColor
        | Collection[MatplotlibColor]
        | None = "color",
    ) -> None:
        pass

    def maybe_color_bp(self, bp) -> None:
        if isinstance(self.color, dict):
            boxes = self.color.get("boxes", self._boxes_c)
            whiskers = self.color.get("whiskers", self._whiskers_c)
            medians = self.color.get("medians", self._medians_c)
            caps = self.color.get("caps", self._caps_c)
        else:
            # Other types are forwarded to matplotlib
            # If None, use default colors
            boxes = self.color or self._boxes_c
            whiskers = self.color or self._whiskers_c
            medians = self.color or self._medians_c
            caps = self.color or self._caps_c

        color_tup = (boxes, whiskers, medians, caps)
        maybe_color_bp(bp, color_tup=color_tup, **self.kwds)

    def _make_plot(self, fig: Figure) -> None:
        if self.subplots:
            self._return_obj = pd.Series(dtype=object)

            # Re-create iterated data if `by` is assigned by users
            data = (
                create_iter_data_given_by(self.data, self._kind)
                if self.by is not None
                else self.data
            )

            # error: Argument "data" to "_iter_data" of "MPLPlot" has
            # incompatible type "object"; expected "DataFrame |
            # dict[Hashable, Series | DataFrame]"
            for i, (label, y) in enumerate(self._iter_data(data=data)):  # type: ignore[arg-type]
                ax = self._get_ax(i)
                kwds = self.kwds.copy()

                # When by is applied, show title for subplots to know which group it is
                # just like df.boxplot, and need to apply T on y to provide right input
                if self.by is not None:
                    y = y.T
                    ax.set_title(pprint_thing(label))

                    # When `by` is assigned, the ticklabels will become unique grouped
                    # values, instead of label which is used as subtitle in this case.
                    # error: "Index" has no attribute "levels"; maybe "nlevels"?
                    levels = self.data.columns.levels  # type: ignore[attr-defined]
                    ticklabels = [pprint_thing(col) for col in levels[0]]
                else:
                    ticklabels = [pprint_thing(label)]

                ret, bp = self._plot(
                    ax, y, column_num=i, return_type=self.return_type, **kwds
                )
                self.maybe_color_bp(bp)
                self._return_obj[label] = ret
                _set_ticklabels(
                    ax=ax, labels=ticklabels, is_vertical=self.orientation == "vertical"
                )
        else:
            y = self.data.values.T
            ax = self._get_ax(0)
            kwds = self.kwds.copy()

            ret, bp = self._plot(
                ax, y, column_num=0, return_type=self.return_type, **kwds
            )
            self.maybe_color_bp(bp)
            self._return_obj = ret

            labels = [pprint_thing(left) for left in self.data.columns]
            if not self.use_index:
                labels = [pprint_thing(key) for key in range(len(labels))]
            _set_ticklabels(
                ax=ax, labels=labels, is_vertical=self.orientation == "vertical"
            )

    def _make_legend(self) -> None:
        pass

    def _post_plot_logic(self, ax: Axes, data) -> None:
        # GH 45465: make sure that the boxplot doesn't ignore xlabel/ylabel
        if self.xlabel:
            ax.set_xlabel(pprint_thing(self.xlabel))
        if self.ylabel:
            ax.set_ylabel(pprint_thing(self.ylabel))

    @property
    def orientation(self) -> Literal["horizontal", "vertical"]:
        if self.kwds.get("vert", True):
            return "vertical"
        else:
            return "horizontal"

    @property
    def result(self):
        if self.return_type is None:
            return super().result
        else:
            return self._return_obj


def maybe_color_bp(bp, color_tup, **kwds) -> None:
    # GH#30346, when users specifying those arguments explicitly, our defaults
    # for these four kwargs should be overridden; if not, use Pandas settings
    if not kwds.get("boxprops"):
        setp(bp["boxes"], color=color_tup[0], alpha=1)
    if not kwds.get("whiskerprops"):
        setp(bp["whiskers"], color=color_tup[1], alpha=1)
    if not kwds.get("medianprops"):
        setp(bp["medians"], color=color_tup[2], alpha=1)
    if not kwds.get("capprops"):
        setp(bp["caps"], color=color_tup[3], alpha=1)


def _grouped_plot_by_column(
    plotf,
    data,
    columns=None,
    by=None,
    numeric_only: bool = True,
    grid: bool = False,
    figsize: tuple[float, float] | None = None,
    ax=None,
    layout=None,
    return_type=None,
    **kwargs,
):
    grouped = data.groupby(by, observed=False)
    if columns is None:
        if not isinstance(by, (list, tuple)):
            by = [by]
        columns = data._get_numeric_data().columns.difference(by)
    naxes = len(columns)
    fig, axes = create_subplots(
        naxes=naxes,
        sharex=kwargs.pop("sharex", True),
        sharey=kwargs.pop("sharey", True),
        figsize=figsize,
        ax=ax,
        layout=layout,
    )

    _axes = flatten_axes(axes)

    # GH 45465: move the "by" label based on "vert"
    xlabel, ylabel = kwargs.pop("xlabel", None), kwargs.pop("ylabel", None)
    if kwargs.get("vert", True):
        xlabel = xlabel or by
    else:
        ylabel = ylabel or by

    ax_values = []

    for i, col in enumerate(columns):
        ax = _axes[i]
        gp_col = grouped[col]
        keys, values = zip(*gp_col)
        re_plotf = plotf(keys, values, ax, xlabel=xlabel, ylabel=ylabel, **kwargs)
        ax.set_title(col)
        ax_values.append(re_plotf)
        ax.grid(grid)

    result = pd.Series(ax_values, index=columns, copy=False)

    # Return axes in multiplot case, maybe revisit later # 985
    if return_type is None:
        result = axes

    byline = by[0] if len(by) == 1 else by
    fig.suptitle(f"Boxplot grouped by {byline}")
    maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, wspace=0.2)

    return result


def boxplot(
    data,
    column=None,
    by=None,
    ax=None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout=None,
    return_type=None,
    **kwds,
):
    import matplotlib.pyplot as plt

    # validate return_type:
    if return_type not in BoxPlot._valid_return_types:
        raise ValueError("return_type must be {'axes', 'dict', 'both'}")

    if isinstance(data, ABCSeries):
        data = data.to_frame("x")
        column = "x"

    def _get_colors():
        #  num_colors=3 is required as method maybe_color_bp takes the colors
        #  in positions 0 and 2.
        #  if colors not provided, use same defaults as DataFrame.plot.box
        result = get_standard_colors(num_colors=3)
        result = np.take(result, [0, 0, 2])
        result = np.append(result, "k")

        colors = kwds.pop("color", None)
        if colors:
            if is_dict_like(colors):
                # replace colors in result array with user-specified colors
                # taken from the colors dict parameter
                # "boxes" value placed in position 0, "whiskers" in 1, etc.
                valid_keys = ["boxes", "whiskers", "medians", "caps"]
                key_to_index = dict(zip(valid_keys, range(4)))
                for key, value in colors.items():
                    if key in valid_keys:
                        result[key_to_index[key]] = value
                    else:
                        raise ValueError(
                            f"color dict contains invalid key '{key}'. "
                            f"The key must be either {valid_keys}"
                        )
            else:
                result.fill(colors)

        return result

    def plot_group(keys, values, ax: Axes, **kwds):
        # GH 45465: xlabel/ylabel need to be popped out before plotting happens
        xlabel, ylabel = kwds.pop("xlabel", None), kwds.pop("ylabel", None)
        if xlabel:
            ax.set_xlabel(pprint_thing(xlabel))
        if ylabel:
            ax.set_ylabel(pprint_thing(ylabel))

        keys = [pprint_thing(x) for x in keys]
        values = [np.asarray(remove_na_arraylike(v), dtype=object) for v in values]
        bp = ax.boxplot(values, **kwds)
        if fontsize is not None:
            ax.tick_params(axis="both", labelsize=fontsize)

        # GH 45465: x/y are flipped when "vert" changes
        _set_ticklabels(
            ax=ax, labels=keys, is_vertical=kwds.get("vert", True), rotation=rot
        )
        maybe_color_bp(bp, color_tup=colors, **kwds)

        # Return axes in multiplot case, maybe revisit later # 985
        if return_type == "dict":
            return bp
        elif return_type == "both":
            return BoxPlot.BP(ax=ax, lines=bp)
        else:
            return ax

    colors = _get_colors()
    if column is None:
        columns = None
    elif isinstance(column, (list, tuple)):
        columns = column
    else:
        columns = [column]

    if by is not None:
        # Prefer array return type for 2-D plots to match the subplot layout
        # https://github.com/pandas-dev/pandas/pull/12216#issuecomment-241175580
        result = _grouped_plot_by_column(
            plot_group,
            data,
            columns=columns,
            by=by,
            grid=grid,
            figsize=figsize,
            ax=ax,
            layout=layout,
            return_type=return_type,
            **kwds,
        )
    else:
        if return_type is None:
            return_type = "axes"
        if layout is not None:
            raise ValueError("The 'layout' keyword is not supported when 'by' is None")

        if ax is None:
            rc = {"figure.figsize": figsize} if figsize is not None else {}
            with plt.rc_context(rc):
                ax = plt.gca()
        data = data._get_numeric_data()
        naxes = len(data.columns)
        if naxes == 0:
            raise ValueError(
                "boxplot method requires numerical columns, nothing to plot."
            )
        if columns is None:
            columns = data.columns
        else:
            data = data[columns]

        result = plot_group(columns, data.values.T, ax, **kwds)
        ax.grid(grid)

    return result


def boxplot_frame(
    self,
    column=None,
    by=None,
    ax=None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout=None,
    return_type=None,
    **kwds,
):
    import matplotlib.pyplot as plt

    ax = boxplot(
        self,
        column=column,
        by=by,
        ax=ax,
        fontsize=fontsize,
        grid=grid,
        rot=rot,
        figsize=figsize,
        layout=layout,
        return_type=return_type,
        **kwds,
    )
    plt.draw_if_interactive()
    return ax


def boxplot_frame_groupby(
    grouped,
    subplots: bool = True,
    column=None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    ax=None,
    figsize: tuple[float, float] | None = None,
    layout=None,
    sharex: bool = False,
    sharey: bool = True,
    **kwds,
):
    if subplots is True:
        naxes = len(grouped)
        fig, axes = create_subplots(
            naxes=naxes,
            squeeze=False,
            ax=ax,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            layout=layout,
        )
        axes = flatten_axes(axes)

        ret = pd.Series(dtype=object)

        for (key, group), ax in zip(grouped, axes):
            d = group.boxplot(
                ax=ax, column=column, fontsize=fontsize, rot=rot, grid=grid, **kwds
            )
            ax.set_title(pprint_thing(key))
            ret.loc[key] = d
        maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, wspace=0.2)
    else:
        keys, frames = zip(*grouped)
        if grouped.axis == 0:
            df = pd.concat(frames, keys=keys, axis=1)
        elif len(frames) > 1:
            df = frames[0].join(frames[1::])
        else:
            df = frames[0]

        # GH 16748, DataFrameGroupby fails when subplots=False and `column` argument
        # is assigned, and in this case, since `df` here becomes MI after groupby,
        # so we need to couple the keys (grouped values) and column (original df
        # column) together to search for subset to plot
        if column is not None:
            column = com.convert_to_list_like(column)
            multi_key = pd.MultiIndex.from_product([keys, column])
            column = list(multi_key.values)
        ret = df.boxplot(
            column=column,
            fontsize=fontsize,
            rot=rot,
            grid=grid,
            ax=ax,
            figsize=figsize,
            layout=layout,
            **kwds,
        )
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\numpy_.py ===
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs import is_supported_dtype
from pandas.compat.numpy import function as nv

from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import isna

from pandas.core import (
    arraylike,
    missing,
    nanops,
    ops,
)
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.strings.object_array import ObjectStringArrayMixin

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import (
        AxisInt,
        Dtype,
        FillnaOptions,
        InterpolateOptions,
        NpDtype,
        Scalar,
        Self,
        npt,
    )

    from pandas import Index


# error: Definition of "_concat_same_type" in base class "NDArrayBacked" is
# incompatible with definition in base class "ExtensionArray"
class NumpyExtensionArray(  # type: ignore[misc]
    OpsMixin,
    NDArrayBackedExtensionArray,
    ObjectStringArrayMixin,
):
    """
    A pandas ExtensionArray for NumPy data.

    This is mostly for internal compatibility, and is not especially
    useful on its own.

    Parameters
    ----------
    values : ndarray
        The NumPy ndarray to wrap. Must be 1-dimensional.
    copy : bool, default False
        Whether to copy `values`.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.arrays.NumpyExtensionArray(np.array([0, 1, 2, 3]))
    <NumpyExtensionArray>
    [0, 1, 2, 3]
    Length: 4, dtype: int64
    """

    # If you're wondering why pd.Series(cls) doesn't put the array in an
    # ExtensionBlock, search for `ABCNumpyExtensionArray`. We check for
    # that _typ to ensure that users don't unnecessarily use EAs inside
    # pandas internals, which turns off things like block consolidation.
    _typ = "npy_extension"
    __array_priority__ = 1000
    _ndarray: np.ndarray
    _dtype: NumpyEADtype
    _internal_fill_value = np.nan

    # ------------------------------------------------------------------------
    # Constructors

    def __init__(
        self, values: np.ndarray | NumpyExtensionArray, copy: bool = False
    ) -> None:
        if isinstance(values, type(self)):
            values = values._ndarray
        if not isinstance(values, np.ndarray):
            raise ValueError(
                f"'values' must be a NumPy array, not {type(values).__name__}"
            )

        if values.ndim == 0:
            # Technically we support 2, but do not advertise that fact.
            raise ValueError("NumpyExtensionArray must be 1-dimensional.")

        if copy:
            values = values.copy()

        dtype = NumpyEADtype(values.dtype)
        super().__init__(values, dtype)

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> NumpyExtensionArray:
        if isinstance(dtype, NumpyEADtype):
            dtype = dtype._dtype

        # error: Argument "dtype" to "asarray" has incompatible type
        # "Union[ExtensionDtype, str, dtype[Any], dtype[floating[_64Bit]], Type[object],
        # None]"; expected "Union[dtype[Any], None, type, _SupportsDType, str,
        # Union[Tuple[Any, int], Tuple[Any, Union[int, Sequence[int]]], List[Any],
        # _DTypeDict, Tuple[Any, Any]]]"
        result = np.asarray(scalars, dtype=dtype)  # type: ignore[arg-type]
        if (
            result.ndim > 1
            and not hasattr(scalars, "dtype")
            and (dtype is None or dtype == object)
        ):
            # e.g. list-of-tuples
            result = construct_1d_object_array_from_listlike(scalars)

        if copy and result is scalars:
            result = result.copy()
        return cls(result)

    # ------------------------------------------------------------------------
    # Data

    @property
    def dtype(self) -> NumpyEADtype:
        return self._dtype

    # ------------------------------------------------------------------------
    # NumPy Array Interface

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        if copy is not None:
            # Note: branch avoids `copy=None` for NumPy 1.x support
            return np.array(self._ndarray, dtype=dtype, copy=copy)
        return np.asarray(self._ndarray, dtype=dtype)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # Lightly modified version of
        # https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html
        # The primary modification is not boxing scalar return values
        # in NumpyExtensionArray, since pandas' ExtensionArrays are 1-d.
        out = kwargs.get("out", ())

        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if "out" in kwargs:
            # e.g. test_ufunc_unary
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                # e.g. tests.series.test_ufunc.TestNumpyReductions
                return result

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(
            x._ndarray if isinstance(x, NumpyExtensionArray) else x for x in inputs
        )
        if out:
            kwargs["out"] = tuple(
                x._ndarray if isinstance(x, NumpyExtensionArray) else x for x in out
            )
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if ufunc.nout > 1:
            # multiple return values; re-box array-like results
            return tuple(type(self)(x) for x in result)
        elif method == "at":
            # no return value
            return None
        elif method == "reduce":
            if isinstance(result, np.ndarray):
                # e.g. test_np_reduce_2d
                return type(self)(result)

            # e.g. test_np_max_nested_tuples
            return result
        else:
            # one return value; re-box array-like results
            return type(self)(result)

    # ------------------------------------------------------------------------
    # Pandas ExtensionArray Interface

    def astype(self, dtype, copy: bool = True):
        dtype = pandas_dtype(dtype)

        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self

        result = astype_array(self._ndarray, dtype=dtype, copy=copy)
        return result

    def isna(self) -> np.ndarray:
        return isna(self._ndarray)

    def _validate_scalar(self, fill_value):
        if fill_value is None:
            # Primarily for subclasses
            fill_value = self.dtype.na_value
        return fill_value

    def _values_for_factorize(self) -> tuple[np.ndarray, float | None]:
        if self.dtype.kind in "iub":
            fv = None
        else:
            fv = np.nan
        return self._ndarray, fv

    # Base EA class (and all other EA classes) don't have limit_area keyword
    # This can be removed here as well when the interpolate ffill/bfill method
    # deprecation is enforced
    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        """
        ffill or bfill along axis=0.
        """
        if copy:
            out_data = self._ndarray.copy()
        else:
            out_data = self._ndarray

        meth = missing.clean_fill_method(method)
        missing.pad_or_backfill_inplace(
            out_data.T,
            method=meth,
            axis=0,
            limit=limit,
            limit_area=limit_area,
        )

        if not copy:
            return self
        return type(self)._simple_new(out_data, dtype=self.dtype)

    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        axis: int,
        index: Index,
        limit,
        limit_direction,
        limit_area,
        copy: bool,
        **kwargs,
    ) -> Self:
        """
        See NDFrame.interpolate.__doc__.
        """
        # NB: we return type(self) even if copy=False
        if not self.dtype._is_numeric:
            raise TypeError(f"Cannot interpolate with {self.dtype} dtype")

        if not copy:
            out_data = self._ndarray
        else:
            out_data = self._ndarray.copy()

        # TODO: assert we have floating dtype?
        missing.interpolate_2d_inplace(
            out_data,
            method=method,
            axis=axis,
            index=index,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            **kwargs,
        )
        if not copy:
            return self
        return type(self)._simple_new(out_data, dtype=self.dtype)

    # ------------------------------------------------------------------------
    # Reductions

    def any(
        self,
        *,
        axis: AxisInt | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_any((), {"out": out, "keepdims": keepdims})
        result = nanops.nanany(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def all(
        self,
        *,
        axis: AxisInt | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_all((), {"out": out, "keepdims": keepdims})
        result = nanops.nanall(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def min(
        self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs
    ) -> Scalar:
        nv.validate_min((), kwargs)
        result = nanops.nanmin(
            values=self._ndarray, axis=axis, mask=self.isna(), skipna=skipna
        )
        return self._wrap_reduction_result(axis, result)

    def max(
        self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs
    ) -> Scalar:
        nv.validate_max((), kwargs)
        result = nanops.nanmax(
            values=self._ndarray, axis=axis, mask=self.isna(), skipna=skipna
        )
        return self._wrap_reduction_result(axis, result)

    def sum(
        self,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs,
    ) -> Scalar:
        nv.validate_sum((), kwargs)
        result = nanops.nansum(
            self._ndarray, axis=axis, skipna=skipna, min_count=min_count
        )
        return self._wrap_reduction_result(axis, result)

    def prod(
        self,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs,
    ) -> Scalar:
        nv.validate_prod((), kwargs)
        result = nanops.nanprod(
            self._ndarray, axis=axis, skipna=skipna, min_count=min_count
        )
        return self._wrap_reduction_result(axis, result)

    def mean(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_mean((), {"dtype": dtype, "out": out, "keepdims": keepdims})
        result = nanops.nanmean(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def median(
        self,
        *,
        axis: AxisInt | None = None,
        out=None,
        overwrite_input: bool = False,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_median(
            (), {"out": out, "overwrite_input": overwrite_input, "keepdims": keepdims}
        )
        result = nanops.nanmedian(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def std(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="std"
        )
        result = nanops.nanstd(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def var(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="var"
        )
        result = nanops.nanvar(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def sem(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="sem"
        )
        result = nanops.nansem(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def kurt(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="kurt"
        )
        result = nanops.nankurt(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def skew(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="skew"
        )
        result = nanops.nanskew(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    # ------------------------------------------------------------------------
    # Additional Methods

    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        mask = self.isna()
        if na_value is not lib.no_default and mask.any():
            result = self._ndarray.copy()
            result[mask] = na_value
        else:
            result = self._ndarray

        result = np.asarray(result, dtype=dtype)

        if copy and result is self._ndarray:
            result = result.copy()

        return result

    # ------------------------------------------------------------------------
    # Ops

    def __invert__(self) -> NumpyExtensionArray:
        return type(self)(~self._ndarray)

    def __neg__(self) -> NumpyExtensionArray:
        return type(self)(-self._ndarray)

    def __pos__(self) -> NumpyExtensionArray:
        return type(self)(+self._ndarray)

    def __abs__(self) -> NumpyExtensionArray:
        return type(self)(abs(self._ndarray))

    def _cmp_method(self, other, op):
        if isinstance(other, NumpyExtensionArray):
            other = other._ndarray

        other = ops.maybe_prepare_scalar_for_op(other, (len(self),))
        pd_op = ops.get_array_op(op)
        other = ensure_wrapped_if_datetimelike(other)
        result = pd_op(self._ndarray, other)

        if op is divmod or op is ops.rdivmod:
            a, b = result
            if isinstance(a, np.ndarray):
                # for e.g. op vs TimedeltaArray, we may already
                #  have an ExtensionArray, in which case we do not wrap
                return self._wrap_ndarray_result(a), self._wrap_ndarray_result(b)
            return a, b

        if isinstance(result, np.ndarray):
            # for e.g. multiplication vs TimedeltaArray, we may already
            #  have an ExtensionArray, in which case we do not wrap
            return self._wrap_ndarray_result(result)
        return result

    _arith_method = _cmp_method

    def _wrap_ndarray_result(self, result: np.ndarray):
        # If we have timedelta64[ns] result, return a TimedeltaArray instead
        #  of a NumpyExtensionArray
        if result.dtype.kind == "m" and is_supported_dtype(result.dtype):
            from pandas.core.arrays import TimedeltaArray

            return TimedeltaArray._simple_new(result, dtype=result.dtype)
        return type(self)(result)

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]:
        # NEP 51: https://github.com/numpy/numpy/pull/22449
        if self.dtype.kind in "SU":
            return "'{}'".format
        elif self.dtype == "object":
            return repr
        else:
            return str

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\represent.py ===
"""Logic for representing operators in state in various bases.

TODO:

* Get represent working with continuous hilbert spaces.
* Document default basis functionality.
"""

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.integrals.integrals import integrate
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.matrixutils import flatten_scalar
from sympy.physics.quantum.state import KetBase, BraBase, StateBase
from sympy.physics.quantum.operator import Operator, OuterProduct
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.operatorset import operators_to_state, state_to_operators


__all__ = [
    'represent',
    'rep_innerproduct',
    'rep_expectation',
    'integrate_result',
    'get_basis',
    'enumerate_states'
]

#-----------------------------------------------------------------------------
# Represent
#-----------------------------------------------------------------------------


def _sympy_to_scalar(e):
    """Convert from a SymPy scalar to a Python scalar."""
    if isinstance(e, Expr):
        if e.is_Integer:
            return int(e)
        elif e.is_Float:
            return float(e)
        elif e.is_Rational:
            return float(e)
        elif e.is_Number or e.is_NumberSymbol or e == I:
            return complex(e)
    raise TypeError('Expected number, got: %r' % e)


def represent(expr, **options):
    """Represent the quantum expression in the given basis.

    In quantum mechanics abstract states and operators can be represented in
    various basis sets. Under this operation the follow transforms happen:

    * Ket -> column vector or function
    * Bra -> row vector of function
    * Operator -> matrix or differential operator

    This function is the top-level interface for this action.

    This function walks the SymPy expression tree looking for ``QExpr``
    instances that have a ``_represent`` method. This method is then called
    and the object is replaced by the representation returned by this method.
    By default, the ``_represent`` method will dispatch to other methods
    that handle the representation logic for a particular basis set. The
    naming convention for these methods is the following::

        def _represent_FooBasis(self, e, basis, **options)

    This function will have the logic for representing instances of its class
    in the basis set having a class named ``FooBasis``.

    Parameters
    ==========

    expr  : Expr
        The expression to represent.
    basis : Operator, basis set
        An object that contains the information about the basis set. If an
        operator is used, the basis is assumed to be the orthonormal
        eigenvectors of that operator. In general though, the basis argument
        can be any object that contains the basis set information.
    options : dict
        Key/value pairs of options that are passed to the underlying method
        that finds the representation. These options can be used to
        control how the representation is done. For example, this is where
        the size of the basis set would be set.

    Returns
    =======

    e : Expr
        The SymPy expression of the represented quantum expression.

    Examples
    ========

    Here we subclass ``Operator`` and ``Ket`` to create the z-spin operator
    and its spin 1/2 up eigenstate. By defining the ``_represent_SzOp``
    method, the ket can be represented in the z-spin basis.

    >>> from sympy.physics.quantum import Operator, represent, Ket
    >>> from sympy import Matrix

    >>> class SzUpKet(Ket):
    ...     def _represent_SzOp(self, basis, **options):
    ...         return Matrix([1,0])
    ...
    >>> class SzOp(Operator):
    ...     pass
    ...
    >>> sz = SzOp('Sz')
    >>> up = SzUpKet('up')
    >>> represent(up, basis=sz)
    Matrix([
    [1],
    [0]])

    Here we see an example of representations in a continuous
    basis. We see that the result of representing various combinations
    of cartesian position operators and kets give us continuous
    expressions involving DiracDelta functions.

    >>> from sympy.physics.quantum.cartesian import XOp, XKet, XBra
    >>> X = XOp()
    >>> x = XKet()
    >>> y = XBra('y')
    >>> represent(X*x)
    x*DiracDelta(x - x_2)
    """

    format = options.get('format', 'sympy')
    if format == 'numpy':
        import numpy as np
    if isinstance(expr, QExpr) and not isinstance(expr, OuterProduct):
        options['replace_none'] = False
        temp_basis = get_basis(expr, **options)
        if temp_basis is not None:
            options['basis'] = temp_basis
        try:
            return expr._represent(**options)
        except NotImplementedError as strerr:
            #If no _represent_FOO method exists, map to the
            #appropriate basis state and try
            #the other methods of representation
            options['replace_none'] = True

            if isinstance(expr, (KetBase, BraBase)):
                try:
                    return rep_innerproduct(expr, **options)
                except NotImplementedError:
                    raise NotImplementedError(strerr)
            elif isinstance(expr, Operator):
                try:
                    return rep_expectation(expr, **options)
                except NotImplementedError:
                    raise NotImplementedError(strerr)
            else:
                raise NotImplementedError(strerr)
    elif isinstance(expr, Add):
        result = represent(expr.args[0], **options)
        for args in expr.args[1:]:
            # scipy.sparse doesn't support += so we use plain = here.
            result = result + represent(args, **options)
        return result
    elif isinstance(expr, Pow):
        base, exp = expr.as_base_exp()
        if format in ('numpy', 'scipy.sparse'):
            exp = _sympy_to_scalar(exp)
        base = represent(base, **options)
        # scipy.sparse doesn't support negative exponents
        # and warns when inverting a matrix in csr format.
        if format == 'scipy.sparse' and exp < 0:
            from scipy.sparse.linalg import inv
            exp = - exp
            base = inv(base.tocsc()).tocsr()
        if format == 'numpy':
            return np.linalg.matrix_power(base, exp)
        return base ** exp
    elif isinstance(expr, TensorProduct):
        new_args = [represent(arg, **options) for arg in expr.args]
        return TensorProduct(*new_args)
    elif isinstance(expr, Dagger):
        return Dagger(represent(expr.args[0], **options))
    elif isinstance(expr, Commutator):
        A = expr.args[0]
        B = expr.args[1]
        return represent(Mul(A, B) - Mul(B, A), **options)
    elif isinstance(expr, AntiCommutator):
        A = expr.args[0]
        B = expr.args[1]
        return represent(Mul(A, B) + Mul(B, A), **options)
    elif not isinstance(expr, (Mul, OuterProduct, InnerProduct)):
        # We have removed special handling of inner products that used to be
        # required (before automatic transforms).
        # For numpy and scipy.sparse, we can only handle numerical prefactors.
        if format in ('numpy', 'scipy.sparse'):
            return _sympy_to_scalar(expr)
        return expr

    if not isinstance(expr, (Mul, OuterProduct, InnerProduct)):
        raise TypeError('Mul expected, got: %r' % expr)

    if "index" in options:
        options["index"] += 1
    else:
        options["index"] = 1

    if "unities" not in options:
        options["unities"] = []

    result = represent(expr.args[-1], **options)
    last_arg = expr.args[-1]

    for arg in reversed(expr.args[:-1]):
        if isinstance(last_arg, Operator):
            options["index"] += 1
            options["unities"].append(options["index"])
        elif isinstance(last_arg, BraBase) and isinstance(arg, KetBase):
            options["index"] += 1
        elif isinstance(last_arg, KetBase) and isinstance(arg, Operator):
            options["unities"].append(options["index"])
        elif isinstance(last_arg, KetBase) and isinstance(arg, BraBase):
            options["unities"].append(options["index"])

        next_arg = represent(arg, **options)
        if format == 'numpy' and isinstance(next_arg, np.ndarray):
            # Must use np.matmult to "matrix multiply" two np.ndarray
            result = np.matmul(next_arg, result)
        else:
            result = next_arg*result
        last_arg = arg

    # All three matrix formats create 1 by 1 matrices when inner products of
    # vectors are taken. In these cases, we simply return a scalar.
    result = flatten_scalar(result)

    result = integrate_result(expr, result, **options)

    return result


def rep_innerproduct(expr, **options):
    """
    Returns an innerproduct like representation (e.g. ``<x'|x>``) for the
    given state.

    Attempts to calculate inner product with a bra from the specified
    basis. Should only be passed an instance of KetBase or BraBase

    Parameters
    ==========

    expr : KetBase or BraBase
        The expression to be represented

    Examples
    ========

    >>> from sympy.physics.quantum.represent import rep_innerproduct
    >>> from sympy.physics.quantum.cartesian import XOp, XKet, PxOp, PxKet
    >>> rep_innerproduct(XKet())
    DiracDelta(x - x_1)
    >>> rep_innerproduct(XKet(), basis=PxOp())
    sqrt(2)*exp(-I*px_1*x/hbar)/(2*sqrt(hbar)*sqrt(pi))
    >>> rep_innerproduct(PxKet(), basis=XOp())
    sqrt(2)*exp(I*px*x_1/hbar)/(2*sqrt(hbar)*sqrt(pi))

    """

    if not isinstance(expr, (KetBase, BraBase)):
        raise TypeError("expr passed is not a Bra or Ket")

    basis = get_basis(expr, **options)

    if not isinstance(basis, StateBase):
        raise NotImplementedError("Can't form this representation!")

    if "index" not in options:
        options["index"] = 1

    basis_kets = enumerate_states(basis, options["index"], 2)

    if isinstance(expr, BraBase):
        bra = expr
        ket = (basis_kets[1] if basis_kets[0].dual == expr else basis_kets[0])
    else:
        bra = (basis_kets[1].dual if basis_kets[0]
               == expr else basis_kets[0].dual)
        ket = expr

    prod = InnerProduct(bra, ket)
    result = prod.doit()

    format = options.get('format', 'sympy')
    result = expr._format_represent(result, format)
    return result


def rep_expectation(expr, **options):
    """
    Returns an ``<x'|A|x>`` type representation for the given operator.

    Parameters
    ==========

    expr : Operator
        Operator to be represented in the specified basis

    Examples
    ========

    >>> from sympy.physics.quantum.cartesian import XOp, PxOp, PxKet
    >>> from sympy.physics.quantum.represent import rep_expectation
    >>> rep_expectation(XOp())
    x_1*DiracDelta(x_1 - x_2)
    >>> rep_expectation(XOp(), basis=PxOp())
    <px_2|*X*|px_1>
    >>> rep_expectation(XOp(), basis=PxKet())
    <px_2|*X*|px_1>

    """

    if "index" not in options:
        options["index"] = 1

    if not isinstance(expr, Operator):
        raise TypeError("The passed expression is not an operator")

    basis_state = get_basis(expr, **options)

    if basis_state is None or not isinstance(basis_state, StateBase):
        raise NotImplementedError("Could not get basis kets for this operator")

    basis_kets = enumerate_states(basis_state, options["index"], 2)

    bra = basis_kets[1].dual
    ket = basis_kets[0]

    result = qapply(bra*expr*ket)
    return result


def integrate_result(orig_expr, result, **options):
    """
    Returns the result of integrating over any unities ``(|x><x|)`` in
    the given expression. Intended for integrating over the result of
    representations in continuous bases.

    This function integrates over any unities that may have been
    inserted into the quantum expression and returns the result.
    It uses the interval of the Hilbert space of the basis state
    passed to it in order to figure out the limits of integration.
    The unities option must be
    specified for this to work.

    Note: This is mostly used internally by represent(). Examples are
    given merely to show the use cases.

    Parameters
    ==========

    orig_expr : quantum expression
        The original expression which was to be represented

    result: Expr
        The resulting representation that we wish to integrate over

    Examples
    ========

    >>> from sympy import symbols, DiracDelta
    >>> from sympy.physics.quantum.represent import integrate_result
    >>> from sympy.physics.quantum.cartesian import XOp, XKet
    >>> x_ket = XKet()
    >>> X_op = XOp()
    >>> x, x_1, x_2 = symbols('x, x_1, x_2')
    >>> integrate_result(X_op*x_ket, x*DiracDelta(x-x_1)*DiracDelta(x_1-x_2))
    x*DiracDelta(x - x_1)*DiracDelta(x_1 - x_2)
    >>> integrate_result(X_op*x_ket, x*DiracDelta(x-x_1)*DiracDelta(x_1-x_2),
    ...     unities=[1])
    x*DiracDelta(x - x_2)

    """
    if not isinstance(result, Expr):
        return result

    options['replace_none'] = True
    if "basis" not in options:
        arg = orig_expr.args[-1]
        options["basis"] = get_basis(arg, **options)
    elif not isinstance(options["basis"], StateBase):
        options["basis"] = get_basis(orig_expr, **options)

    basis = options.pop("basis", None)

    if basis is None:
        return result

    unities = options.pop("unities", [])

    if len(unities) == 0:
        return result

    kets = enumerate_states(basis, unities)
    coords = [k.label[0] for k in kets]

    for coord in coords:
        if coord in result.free_symbols:
            #TODO: Add support for sets of operators
            basis_op = state_to_operators(basis)
            start = basis_op.hilbert_space.interval.start
            end = basis_op.hilbert_space.interval.end
            result = integrate(result, (coord, start, end))

    return result


def get_basis(expr, *, basis=None, replace_none=True, **options):
    """
    Returns a basis state instance corresponding to the basis specified in
    options=s. If no basis is specified, the function tries to form a default
    basis state of the given expression.

    There are three behaviors:

    1. The basis specified in options is already an instance of StateBase. If
       this is the case, it is simply returned. If the class is specified but
       not an instance, a default instance is returned.

    2. The basis specified is an operator or set of operators. If this
       is the case, the operator_to_state mapping method is used.

    3. No basis is specified. If expr is a state, then a default instance of
       its class is returned.  If expr is an operator, then it is mapped to the
       corresponding state.  If it is neither, then we cannot obtain the basis
       state.

    If the basis cannot be mapped, then it is not changed.

    This will be called from within represent, and represent will
    only pass QExpr's.

    TODO (?): Support for Muls and other types of expressions?

    Parameters
    ==========

    expr : Operator or StateBase
        Expression whose basis is sought

    Examples
    ========

    >>> from sympy.physics.quantum.represent import get_basis
    >>> from sympy.physics.quantum.cartesian import XOp, XKet, PxOp, PxKet
    >>> x = XKet()
    >>> X = XOp()
    >>> get_basis(x)
    |x>
    >>> get_basis(X)
    |x>
    >>> get_basis(x, basis=PxOp())
    |px>
    >>> get_basis(x, basis=PxKet)
    |px>

    """

    if basis is None and not replace_none:
        return None

    if basis is None:
        if isinstance(expr, KetBase):
            return _make_default(expr.__class__)
        elif isinstance(expr, BraBase):
            return _make_default(expr.dual_class())
        elif isinstance(expr, Operator):
            state_inst = operators_to_state(expr)
            return (state_inst if state_inst is not None else None)
        else:
            return None
    elif (isinstance(basis, Operator) or
          (not isinstance(basis, StateBase) and issubclass(basis, Operator))):
        state = operators_to_state(basis)
        if state is None:
            return None
        elif isinstance(state, StateBase):
            return state
        else:
            return _make_default(state)
    elif isinstance(basis, StateBase):
        return basis
    elif issubclass(basis, StateBase):
        return _make_default(basis)
    else:
        return None


def _make_default(expr):
    # XXX: Catching TypeError like this is a bad way of distinguishing
    # instances from classes. The logic using this function should be
    # rewritten somehow.
    try:
        expr = expr()
    except TypeError:
        return expr

    return expr


def enumerate_states(*args, **options):
    """
    Returns instances of the given state with dummy indices appended

    Operates in two different modes:

    1. Two arguments are passed to it. The first is the base state which is to
       be indexed, and the second argument is a list of indices to append.

    2. Three arguments are passed. The first is again the base state to be
       indexed. The second is the start index for counting.  The final argument
       is the number of kets you wish to receive.

    Tries to call state._enumerate_state. If this fails, returns an empty list

    Parameters
    ==========

    args : list
        See list of operation modes above for explanation

    Examples
    ========

    >>> from sympy.physics.quantum.cartesian import XBra, XKet
    >>> from sympy.physics.quantum.represent import enumerate_states
    >>> test = XKet('foo')
    >>> enumerate_states(test, 1, 3)
    [|foo_1>, |foo_2>, |foo_3>]
    >>> test2 = XBra('bar')
    >>> enumerate_states(test2, [4, 5, 10])
    [<bar_4|, <bar_5|, <bar_10|]

    """

    state = args[0]

    if len(args) not in (2, 3):
        raise NotImplementedError("Wrong number of arguments!")

    if not isinstance(state, StateBase):
        raise TypeError("First argument is not a state!")

    if len(args) == 3:
        num_states = args[2]
        options['start_index'] = args[1]
    else:
        num_states = len(args[1])
        options['index_list'] = args[1]

    try:
        ret = state._enumerate_state(num_states, **options)
    except NotImplementedError:
        ret = []

    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\computation\ops.py ===
"""
Operator classes for eval.
"""

from __future__ import annotations

from datetime import datetime
from functools import partial
import operator
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
)

import numpy as np

from pandas._libs.tslibs import Timestamp

from pandas.core.dtypes.common import (
    is_list_like,
    is_scalar,
)

import pandas.core.common as com
from pandas.core.computation.common import (
    ensure_decoded,
    result_type_many,
)
from pandas.core.computation.scope import DEFAULT_GLOBALS

from pandas.io.formats.printing import (
    pprint_thing,
    pprint_thing_encoded,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Iterator,
    )

REDUCTIONS = ("sum", "prod", "min", "max")

_unary_math_ops = (
    "sin",
    "cos",
    "exp",
    "log",
    "expm1",
    "log1p",
    "sqrt",
    "sinh",
    "cosh",
    "tanh",
    "arcsin",
    "arccos",
    "arctan",
    "arccosh",
    "arcsinh",
    "arctanh",
    "abs",
    "log10",
    "floor",
    "ceil",
)
_binary_math_ops = ("arctan2",)

MATHOPS = _unary_math_ops + _binary_math_ops


LOCAL_TAG = "__pd_eval_local_"


class Term:
    def __new__(cls, name, env, side=None, encoding=None):
        klass = Constant if not isinstance(name, str) else cls
        # error: Argument 2 for "super" not an instance of argument 1
        supr_new = super(Term, klass).__new__  # type: ignore[misc]
        return supr_new(klass)

    is_local: bool

    def __init__(self, name, env, side=None, encoding=None) -> None:
        # name is a str for Term, but may be something else for subclasses
        self._name = name
        self.env = env
        self.side = side
        tname = str(name)
        self.is_local = tname.startswith(LOCAL_TAG) or tname in DEFAULT_GLOBALS
        self._value = self._resolve_name()
        self.encoding = encoding

    @property
    def local_name(self) -> str:
        return self.name.replace(LOCAL_TAG, "")

    def __repr__(self) -> str:
        return pprint_thing(self.name)

    def __call__(self, *args, **kwargs):
        return self.value

    def evaluate(self, *args, **kwargs) -> Term:
        return self

    def _resolve_name(self):
        local_name = str(self.local_name)
        is_local = self.is_local
        if local_name in self.env.scope and isinstance(
            self.env.scope[local_name], type
        ):
            is_local = False

        res = self.env.resolve(local_name, is_local=is_local)
        self.update(res)

        if hasattr(res, "ndim") and res.ndim > 2:
            raise NotImplementedError(
                "N-dimensional objects, where N > 2, are not supported with eval"
            )
        return res

    def update(self, value) -> None:
        """
        search order for local (i.e., @variable) variables:

        scope, key_variable
        [('locals', 'local_name'),
         ('globals', 'local_name'),
         ('locals', 'key'),
         ('globals', 'key')]
        """
        key = self.name

        # if it's a variable name (otherwise a constant)
        if isinstance(key, str):
            self.env.swapkey(self.local_name, key, new_value=value)

        self.value = value

    @property
    def is_scalar(self) -> bool:
        return is_scalar(self._value)

    @property
    def type(self):
        try:
            # potentially very slow for large, mixed dtype frames
            return self._value.values.dtype
        except AttributeError:
            try:
                # ndarray
                return self._value.dtype
            except AttributeError:
                # scalar
                return type(self._value)

    return_type = type

    @property
    def raw(self) -> str:
        return f"{type(self).__name__}(name={repr(self.name)}, type={self.type})"

    @property
    def is_datetime(self) -> bool:
        try:
            t = self.type.type
        except AttributeError:
            t = self.type

        return issubclass(t, (datetime, np.datetime64))

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value) -> None:
        self._value = new_value

    @property
    def name(self):
        return self._name

    @property
    def ndim(self) -> int:
        return self._value.ndim


class Constant(Term):
    def _resolve_name(self):
        return self._name

    @property
    def name(self):
        return self.value

    def __repr__(self) -> str:
        # in python 2 str() of float
        # can truncate shorter than repr()
        return repr(self.name)


_bool_op_map = {"not": "~", "and": "&", "or": "|"}


class Op:
    """
    Hold an operator of arbitrary arity.
    """

    op: str

    def __init__(self, op: str, operands: Iterable[Term | Op], encoding=None) -> None:
        self.op = _bool_op_map.get(op, op)
        self.operands = operands
        self.encoding = encoding

    def __iter__(self) -> Iterator:
        return iter(self.operands)

    def __repr__(self) -> str:
        """
        Print a generic n-ary operator and its operands using infix notation.
        """
        # recurse over the operands
        parened = (f"({pprint_thing(opr)})" for opr in self.operands)
        return pprint_thing(f" {self.op} ".join(parened))

    @property
    def return_type(self):
        # clobber types to bool if the op is a boolean operator
        if self.op in (CMP_OPS_SYMS + BOOL_OPS_SYMS):
            return np.bool_
        return result_type_many(*(term.type for term in com.flatten(self)))

    @property
    def has_invalid_return_type(self) -> bool:
        types = self.operand_types
        obj_dtype_set = frozenset([np.dtype("object")])
        return self.return_type == object and types - obj_dtype_set

    @property
    def operand_types(self):
        return frozenset(term.type for term in com.flatten(self))

    @property
    def is_scalar(self) -> bool:
        return all(operand.is_scalar for operand in self.operands)

    @property
    def is_datetime(self) -> bool:
        try:
            t = self.return_type.type
        except AttributeError:
            t = self.return_type

        return issubclass(t, (datetime, np.datetime64))


def _in(x, y):
    """
    Compute the vectorized membership of ``x in y`` if possible, otherwise
    use Python.
    """
    try:
        return x.isin(y)
    except AttributeError:
        if is_list_like(x):
            try:
                return y.isin(x)
            except AttributeError:
                pass
        return x in y


def _not_in(x, y):
    """
    Compute the vectorized membership of ``x not in y`` if possible,
    otherwise use Python.
    """
    try:
        return ~x.isin(y)
    except AttributeError:
        if is_list_like(x):
            try:
                return ~y.isin(x)
            except AttributeError:
                pass
        return x not in y


CMP_OPS_SYMS = (">", "<", ">=", "<=", "==", "!=", "in", "not in")
_cmp_ops_funcs = (
    operator.gt,
    operator.lt,
    operator.ge,
    operator.le,
    operator.eq,
    operator.ne,
    _in,
    _not_in,
)
_cmp_ops_dict = dict(zip(CMP_OPS_SYMS, _cmp_ops_funcs))

BOOL_OPS_SYMS = ("&", "|", "and", "or")
_bool_ops_funcs = (operator.and_, operator.or_, operator.and_, operator.or_)
_bool_ops_dict = dict(zip(BOOL_OPS_SYMS, _bool_ops_funcs))

ARITH_OPS_SYMS = ("+", "-", "*", "/", "**", "//", "%")
_arith_ops_funcs = (
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.pow,
    operator.floordiv,
    operator.mod,
)
_arith_ops_dict = dict(zip(ARITH_OPS_SYMS, _arith_ops_funcs))

SPECIAL_CASE_ARITH_OPS_SYMS = ("**", "//", "%")
_special_case_arith_ops_funcs = (operator.pow, operator.floordiv, operator.mod)
_special_case_arith_ops_dict = dict(
    zip(SPECIAL_CASE_ARITH_OPS_SYMS, _special_case_arith_ops_funcs)
)

_binary_ops_dict = {}

for d in (_cmp_ops_dict, _bool_ops_dict, _arith_ops_dict):
    _binary_ops_dict.update(d)


def is_term(obj) -> bool:
    return isinstance(obj, Term)


class BinOp(Op):
    """
    Hold a binary operator and its operands.

    Parameters
    ----------
    op : str
    lhs : Term or Op
    rhs : Term or Op
    """

    def __init__(self, op: str, lhs, rhs) -> None:
        super().__init__(op, (lhs, rhs))
        self.lhs = lhs
        self.rhs = rhs

        self._disallow_scalar_only_bool_ops()

        self.convert_values()

        try:
            self.func = _binary_ops_dict[op]
        except KeyError as err:
            # has to be made a list for python3
            keys = list(_binary_ops_dict.keys())
            raise ValueError(
                f"Invalid binary operator {repr(op)}, valid operators are {keys}"
            ) from err

    def __call__(self, env):
        """
        Recursively evaluate an expression in Python space.

        Parameters
        ----------
        env : Scope

        Returns
        -------
        object
            The result of an evaluated expression.
        """
        # recurse over the left/right nodes
        left = self.lhs(env)
        right = self.rhs(env)

        return self.func(left, right)

    def evaluate(self, env, engine: str, parser, term_type, eval_in_python):
        """
        Evaluate a binary operation *before* being passed to the engine.

        Parameters
        ----------
        env : Scope
        engine : str
        parser : str
        term_type : type
        eval_in_python : list

        Returns
        -------
        term_type
            The "pre-evaluated" expression as an instance of ``term_type``
        """
        if engine == "python":
            res = self(env)
        else:
            # recurse over the left/right nodes

            left = self.lhs.evaluate(
                env,
                engine=engine,
                parser=parser,
                term_type=term_type,
                eval_in_python=eval_in_python,
            )

            right = self.rhs.evaluate(
                env,
                engine=engine,
                parser=parser,
                term_type=term_type,
                eval_in_python=eval_in_python,
            )

            # base cases
            if self.op in eval_in_python:
                res = self.func(left.value, right.value)
            else:
                from pandas.core.computation.eval import eval

                res = eval(self, local_dict=env, engine=engine, parser=parser)

        name = env.add_tmp(res)
        return term_type(name, env=env)

    def convert_values(self) -> None:
        """
        Convert datetimes to a comparable value in an expression.
        """

        def stringify(value):
            encoder: Callable
            if self.encoding is not None:
                encoder = partial(pprint_thing_encoded, encoding=self.encoding)
            else:
                encoder = pprint_thing
            return encoder(value)

        lhs, rhs = self.lhs, self.rhs

        if is_term(lhs) and lhs.is_datetime and is_term(rhs) and rhs.is_scalar:
            v = rhs.value
            if isinstance(v, (int, float)):
                v = stringify(v)
            v = Timestamp(ensure_decoded(v))
            if v.tz is not None:
                v = v.tz_convert("UTC")
            self.rhs.update(v)

        if is_term(rhs) and rhs.is_datetime and is_term(lhs) and lhs.is_scalar:
            v = lhs.value
            if isinstance(v, (int, float)):
                v = stringify(v)
            v = Timestamp(ensure_decoded(v))
            if v.tz is not None:
                v = v.tz_convert("UTC")
            self.lhs.update(v)

    def _disallow_scalar_only_bool_ops(self):
        rhs = self.rhs
        lhs = self.lhs

        # GH#24883 unwrap dtype if necessary to ensure we have a type object
        rhs_rt = rhs.return_type
        rhs_rt = getattr(rhs_rt, "type", rhs_rt)
        lhs_rt = lhs.return_type
        lhs_rt = getattr(lhs_rt, "type", lhs_rt)
        if (
            (lhs.is_scalar or rhs.is_scalar)
            and self.op in _bool_ops_dict
            and (
                not (
                    issubclass(rhs_rt, (bool, np.bool_))
                    and issubclass(lhs_rt, (bool, np.bool_))
                )
            )
        ):
            raise NotImplementedError("cannot evaluate scalar only bool ops")


def isnumeric(dtype) -> bool:
    return issubclass(np.dtype(dtype).type, np.number)


UNARY_OPS_SYMS = ("+", "-", "~", "not")
_unary_ops_funcs = (operator.pos, operator.neg, operator.invert, operator.invert)
_unary_ops_dict = dict(zip(UNARY_OPS_SYMS, _unary_ops_funcs))


class UnaryOp(Op):
    """
    Hold a unary operator and its operands.

    Parameters
    ----------
    op : str
        The token used to represent the operator.
    operand : Term or Op
        The Term or Op operand to the operator.

    Raises
    ------
    ValueError
        * If no function associated with the passed operator token is found.
    """

    def __init__(self, op: Literal["+", "-", "~", "not"], operand) -> None:
        super().__init__(op, (operand,))
        self.operand = operand

        try:
            self.func = _unary_ops_dict[op]
        except KeyError as err:
            raise ValueError(
                f"Invalid unary operator {repr(op)}, "
                f"valid operators are {UNARY_OPS_SYMS}"
            ) from err

    def __call__(self, env) -> MathCall:
        operand = self.operand(env)
        # error: Cannot call function of unknown type
        return self.func(operand)  # type: ignore[operator]

    def __repr__(self) -> str:
        return pprint_thing(f"{self.op}({self.operand})")

    @property
    def return_type(self) -> np.dtype:
        operand = self.operand
        if operand.return_type == np.dtype("bool"):
            return np.dtype("bool")
        if isinstance(operand, Op) and (
            operand.op in _cmp_ops_dict or operand.op in _bool_ops_dict
        ):
            return np.dtype("bool")
        return np.dtype("int")


class MathCall(Op):
    def __init__(self, func, args) -> None:
        super().__init__(func.name, args)
        self.func = func

    def __call__(self, env):
        # error: "Op" not callable
        operands = [op(env) for op in self.operands]  # type: ignore[operator]
        return self.func.func(*operands)

    def __repr__(self) -> str:
        operands = map(str, self.operands)
        return pprint_thing(f"{self.op}({','.join(operands)})")


class FuncNode:
    def __init__(self, name: str) -> None:
        if name not in MATHOPS:
            raise ValueError(f'"{name}" is not a supported function')
        self.name = name
        self.func = getattr(np, name)

    def __call__(self, *args) -> MathCall:
        return MathCall(self, args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\formats\printing.py ===
"""
Printing tools.
"""
from __future__ import annotations

from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
)
import sys
from typing import (
    Any,
    Callable,
    TypeVar,
    Union,
)
from unicodedata import east_asian_width

from pandas._config import get_option

from pandas.core.dtypes.inference import is_sequence

from pandas.io.formats.console import get_console_size

EscapeChars = Union[Mapping[str, str], Iterable[str]]
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def adjoin(space: int, *lists: list[str], **kwargs) -> str:
    """
    Glues together two sets of strings using the amount of space requested.
    The idea is to prettify.

    ----------
    space : int
        number of spaces for padding
    lists : str
        list of str which being joined
    strlen : callable
        function used to calculate the length of each str. Needed for unicode
        handling.
    justfunc : callable
        function used to justify str. Needed for unicode handling.
    """
    strlen = kwargs.pop("strlen", len)
    justfunc = kwargs.pop("justfunc", _adj_justify)

    newLists = []
    lengths = [max(map(strlen, x)) + space for x in lists[:-1]]
    # not the last one
    lengths.append(max(map(len, lists[-1])))
    maxLen = max(map(len, lists))
    for i, lst in enumerate(lists):
        nl = justfunc(lst, lengths[i], mode="left")
        nl = ([" " * lengths[i]] * (maxLen - len(lst))) + nl
        newLists.append(nl)
    toJoin = zip(*newLists)
    return "\n".join("".join(lines) for lines in toJoin)


def _adj_justify(texts: Iterable[str], max_len: int, mode: str = "right") -> list[str]:
    """
    Perform ljust, center, rjust against string or list-like
    """
    if mode == "left":
        return [x.ljust(max_len) for x in texts]
    elif mode == "center":
        return [x.center(max_len) for x in texts]
    else:
        return [x.rjust(max_len) for x in texts]


# Unicode consolidation
# ---------------------
#
# pprinting utility functions for generating Unicode text or
# bytes(3.x)/str(2.x) representations of objects.
# Try to use these as much as possible rather than rolling your own.
#
# When to use
# -----------
#
# 1) If you're writing code internal to pandas (no I/O directly involved),
#    use pprint_thing().
#
#    It will always return unicode text which can handled by other
#    parts of the package without breakage.
#
# 2) if you need to write something out to file, use
#    pprint_thing_encoded(encoding).
#
#    If no encoding is specified, it defaults to utf-8. Since encoding pure
#    ascii with utf-8 is a no-op you can safely use the default utf-8 if you're
#    working with straight ascii.


def _pprint_seq(
    seq: Sequence, _nest_lvl: int = 0, max_seq_items: int | None = None, **kwds
) -> str:
    """
    internal. pprinter for iterables. you should probably use pprint_thing()
    rather than calling this directly.

    bounds length of printed sequence, depending on options
    """
    if isinstance(seq, set):
        fmt = "{{{body}}}"
    else:
        fmt = "[{body}]" if hasattr(seq, "__setitem__") else "({body})"

    if max_seq_items is False:
        nitems = len(seq)
    else:
        nitems = max_seq_items or get_option("max_seq_items") or len(seq)

    s = iter(seq)
    # handle sets, no slicing
    r = [
        pprint_thing(next(s), _nest_lvl + 1, max_seq_items=max_seq_items, **kwds)
        for i in range(min(nitems, len(seq)))
    ]
    body = ", ".join(r)

    if nitems < len(seq):
        body += ", ..."
    elif isinstance(seq, tuple) and len(seq) == 1:
        body += ","

    return fmt.format(body=body)


def _pprint_dict(
    seq: Mapping, _nest_lvl: int = 0, max_seq_items: int | None = None, **kwds
) -> str:
    """
    internal. pprinter for iterables. you should probably use pprint_thing()
    rather than calling this directly.
    """
    fmt = "{{{things}}}"
    pairs = []

    pfmt = "{key}: {val}"

    if max_seq_items is False:
        nitems = len(seq)
    else:
        nitems = max_seq_items or get_option("max_seq_items") or len(seq)

    for k, v in list(seq.items())[:nitems]:
        pairs.append(
            pfmt.format(
                key=pprint_thing(k, _nest_lvl + 1, max_seq_items=max_seq_items, **kwds),
                val=pprint_thing(v, _nest_lvl + 1, max_seq_items=max_seq_items, **kwds),
            )
        )

    if nitems < len(seq):
        return fmt.format(things=", ".join(pairs) + ", ...")
    else:
        return fmt.format(things=", ".join(pairs))


def pprint_thing(
    thing: Any,
    _nest_lvl: int = 0,
    escape_chars: EscapeChars | None = None,
    default_escapes: bool = False,
    quote_strings: bool = False,
    max_seq_items: int | None = None,
) -> str:
    """
    This function is the sanctioned way of converting objects
    to a string representation and properly handles nested sequences.

    Parameters
    ----------
    thing : anything to be formatted
    _nest_lvl : internal use only. pprint_thing() is mutually-recursive
        with pprint_sequence, this argument is used to keep track of the
        current nesting level, and limit it.
    escape_chars : list or dict, optional
        Characters to escape. If a dict is passed the values are the
        replacements
    default_escapes : bool, default False
        Whether the input escape characters replaces or adds to the defaults
    max_seq_items : int or None, default None
        Pass through to other pretty printers to limit sequence printing

    Returns
    -------
    str
    """

    def as_escaped_string(
        thing: Any, escape_chars: EscapeChars | None = escape_chars
    ) -> str:
        translate = {"\t": r"\t", "\n": r"\n", "\r": r"\r"}
        if isinstance(escape_chars, dict):
            if default_escapes:
                translate.update(escape_chars)
            else:
                translate = escape_chars
            escape_chars = list(escape_chars.keys())
        else:
            escape_chars = escape_chars or ()

        result = str(thing)
        for c in escape_chars:
            result = result.replace(c, translate[c])
        return result

    if hasattr(thing, "__next__"):
        return str(thing)
    elif isinstance(thing, dict) and _nest_lvl < get_option(
        "display.pprint_nest_depth"
    ):
        result = _pprint_dict(
            thing, _nest_lvl, quote_strings=True, max_seq_items=max_seq_items
        )
    elif is_sequence(thing) and _nest_lvl < get_option("display.pprint_nest_depth"):
        result = _pprint_seq(
            thing,
            _nest_lvl,
            escape_chars=escape_chars,
            quote_strings=quote_strings,
            max_seq_items=max_seq_items,
        )
    elif isinstance(thing, str) and quote_strings:
        result = f"'{as_escaped_string(thing)}'"
    else:
        result = as_escaped_string(thing)

    return result


def pprint_thing_encoded(
    object, encoding: str = "utf-8", errors: str = "replace"
) -> bytes:
    value = pprint_thing(object)  # get unicode representation of object
    return value.encode(encoding, errors)


def enable_data_resource_formatter(enable: bool) -> None:
    if "IPython" not in sys.modules:
        # definitely not in IPython
        return
    from IPython import get_ipython

    ip = get_ipython()
    if ip is None:
        # still not in IPython
        return

    formatters = ip.display_formatter.formatters
    mimetype = "application/vnd.dataresource+json"

    if enable:
        if mimetype not in formatters:
            # define tableschema formatter
            from IPython.core.formatters import BaseFormatter
            from traitlets import ObjectName

            class TableSchemaFormatter(BaseFormatter):
                print_method = ObjectName("_repr_data_resource_")
                _return_type = (dict,)

            # register it:
            formatters[mimetype] = TableSchemaFormatter()
        # enable it if it's been disabled:
        formatters[mimetype].enabled = True
    # unregister tableschema mime-type
    elif mimetype in formatters:
        formatters[mimetype].enabled = False


def default_pprint(thing: Any, max_seq_items: int | None = None) -> str:
    return pprint_thing(
        thing,
        escape_chars=("\t", "\r", "\n"),
        quote_strings=True,
        max_seq_items=max_seq_items,
    )


def format_object_summary(
    obj,
    formatter: Callable,
    is_justify: bool = True,
    name: str | None = None,
    indent_for_name: bool = True,
    line_break_each_value: bool = False,
) -> str:
    """
    Return the formatted obj as a unicode string

    Parameters
    ----------
    obj : object
        must be iterable and support __getitem__
    formatter : callable
        string formatter for an element
    is_justify : bool
        should justify the display
    name : name, optional
        defaults to the class name of the obj
    indent_for_name : bool, default True
        Whether subsequent lines should be indented to
        align with the name.
    line_break_each_value : bool, default False
        If True, inserts a line break for each value of ``obj``.
        If False, only break lines when the a line of values gets wider
        than the display width.

    Returns
    -------
    summary string
    """
    display_width, _ = get_console_size()
    if display_width is None:
        display_width = get_option("display.width") or 80
    if name is None:
        name = type(obj).__name__

    if indent_for_name:
        name_len = len(name)
        space1 = f'\n{(" " * (name_len + 1))}'
        space2 = f'\n{(" " * (name_len + 2))}'
    else:
        space1 = "\n"
        space2 = "\n "  # space for the opening '['

    n = len(obj)
    if line_break_each_value:
        # If we want to vertically align on each value of obj, we need to
        # separate values by a line break and indent the values
        sep = ",\n " + " " * len(name)
    else:
        sep = ","
    max_seq_items = get_option("display.max_seq_items") or n

    # are we a truncated display
    is_truncated = n > max_seq_items

    # adj can optionally handle unicode eastern asian width
    adj = get_adjustment()

    def _extend_line(
        s: str, line: str, value: str, display_width: int, next_line_prefix: str
    ) -> tuple[str, str]:
        if adj.len(line.rstrip()) + adj.len(value.rstrip()) >= display_width:
            s += line.rstrip()
            line = next_line_prefix
        line += value
        return s, line

    def best_len(values: list[str]) -> int:
        if values:
            return max(adj.len(x) for x in values)
        else:
            return 0

    close = ", "

    if n == 0:
        summary = f"[]{close}"
    elif n == 1 and not line_break_each_value:
        first = formatter(obj[0])
        summary = f"[{first}]{close}"
    elif n == 2 and not line_break_each_value:
        first = formatter(obj[0])
        last = formatter(obj[-1])
        summary = f"[{first}, {last}]{close}"
    else:
        if max_seq_items == 1:
            # If max_seq_items=1 show only last element
            head = []
            tail = [formatter(x) for x in obj[-1:]]
        elif n > max_seq_items:
            n = min(max_seq_items // 2, 10)
            head = [formatter(x) for x in obj[:n]]
            tail = [formatter(x) for x in obj[-n:]]
        else:
            head = []
            tail = [formatter(x) for x in obj]

        # adjust all values to max length if needed
        if is_justify:
            if line_break_each_value:
                # Justify each string in the values of head and tail, so the
                # strings will right align when head and tail are stacked
                # vertically.
                head, tail = _justify(head, tail)
            elif is_truncated or not (
                len(", ".join(head)) < display_width
                and len(", ".join(tail)) < display_width
            ):
                # Each string in head and tail should align with each other
                max_length = max(best_len(head), best_len(tail))
                head = [x.rjust(max_length) for x in head]
                tail = [x.rjust(max_length) for x in tail]
            # If we are not truncated and we are only a single
            # line, then don't justify

        if line_break_each_value:
            # Now head and tail are of type List[Tuple[str]]. Below we
            # convert them into List[str], so there will be one string per
            # value. Also truncate items horizontally if wider than
            # max_space
            max_space = display_width - len(space2)
            value = tail[0]
            max_items = 1
            for num_items in reversed(range(1, len(value) + 1)):
                pprinted_seq = _pprint_seq(value, max_seq_items=num_items)
                if len(pprinted_seq) < max_space:
                    max_items = num_items
                    break
            head = [_pprint_seq(x, max_seq_items=max_items) for x in head]
            tail = [_pprint_seq(x, max_seq_items=max_items) for x in tail]

        summary = ""
        line = space2

        for head_value in head:
            word = head_value + sep + " "
            summary, line = _extend_line(summary, line, word, display_width, space2)

        if is_truncated:
            # remove trailing space of last line
            summary += line.rstrip() + space2 + "..."
            line = space2

        for tail_item in tail[:-1]:
            word = tail_item + sep + " "
            summary, line = _extend_line(summary, line, word, display_width, space2)

        # last value: no sep added + 1 space of width used for trailing ','
        summary, line = _extend_line(summary, line, tail[-1], display_width - 2, space2)
        summary += line

        # right now close is either '' or ', '
        # Now we want to include the ']', but not the maybe space.
        close = "]" + close.rstrip(" ")
        summary += close

        if len(summary) > (display_width) or line_break_each_value:
            summary += space1
        else:  # one row
            summary += " "

        # remove initial space
        summary = "[" + summary[len(space2) :]

    return summary


def _justify(
    head: list[Sequence[str]], tail: list[Sequence[str]]
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """
    Justify items in head and tail, so they are right-aligned when stacked.

    Parameters
    ----------
    head : list-like of list-likes of strings
    tail : list-like of list-likes of strings

    Returns
    -------
    tuple of list of tuples of strings
        Same as head and tail, but items are right aligned when stacked
        vertically.

    Examples
    --------
    >>> _justify([['a', 'b']], [['abc', 'abcd']])
    ([('  a', '   b')], [('abc', 'abcd')])
    """
    combined = head + tail

    # For each position for the sequences in ``combined``,
    # find the length of the largest string.
    max_length = [0] * len(combined[0])
    for inner_seq in combined:
        length = [len(item) for item in inner_seq]
        max_length = [max(x, y) for x, y in zip(max_length, length)]

    # justify each item in each list-like in head and tail using max_length
    head_tuples = [
        tuple(x.rjust(max_len) for x, max_len in zip(seq, max_length)) for seq in head
    ]
    tail_tuples = [
        tuple(x.rjust(max_len) for x, max_len in zip(seq, max_length)) for seq in tail
    ]
    return head_tuples, tail_tuples


class PrettyDict(dict[_KT, _VT]):
    """Dict extension to support abbreviated __repr__"""

    def __repr__(self) -> str:
        return pprint_thing(self)


class _TextAdjustment:
    def __init__(self) -> None:
        self.encoding = get_option("display.encoding")

    def len(self, text: str) -> int:
        return len(text)

    def justify(self, texts: Any, max_len: int, mode: str = "right") -> list[str]:
        """
        Perform ljust, center, rjust against string or list-like
        """
        if mode == "left":
            return [x.ljust(max_len) for x in texts]
        elif mode == "center":
            return [x.center(max_len) for x in texts]
        else:
            return [x.rjust(max_len) for x in texts]

    def adjoin(self, space: int, *lists, **kwargs) -> str:
        return adjoin(space, *lists, strlen=self.len, justfunc=self.justify, **kwargs)


class _EastAsianTextAdjustment(_TextAdjustment):
    def __init__(self) -> None:
        super().__init__()
        if get_option("display.unicode.ambiguous_as_wide"):
            self.ambiguous_width = 2
        else:
            self.ambiguous_width = 1

        # Definition of East Asian Width
        # https://unicode.org/reports/tr11/
        # Ambiguous width can be changed by option
        self._EAW_MAP = {"Na": 1, "N": 1, "W": 2, "F": 2, "H": 1}

    def len(self, text: str) -> int:
        """
        Calculate display width considering unicode East Asian Width
        """
        if not isinstance(text, str):
            return len(text)

        return sum(
            self._EAW_MAP.get(east_asian_width(c), self.ambiguous_width) for c in text
        )

    def justify(
        self, texts: Iterable[str], max_len: int, mode: str = "right"
    ) -> list[str]:
        # re-calculate padding space per str considering East Asian Width
        def _get_pad(t):
            return max_len - self.len(t) + len(t)

        if mode == "left":
            return [x.ljust(_get_pad(x)) for x in texts]
        elif mode == "center":
            return [x.center(_get_pad(x)) for x in texts]
        else:
            return [x.rjust(_get_pad(x)) for x in texts]


def get_adjustment() -> _TextAdjustment:
    use_east_asian_width = get_option("display.unicode.east_asian_width")
    if use_east_asian_width:
        return _EastAsianTextAdjustment()
    else:
        return _TextAdjustment()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\connection.py ===
from __future__ import absolute_import

import datetime
import logging
import os
import re
import socket
import warnings
from socket import error as SocketError
from socket import timeout as SocketTimeout

from .packages import six
from .packages.six.moves.http_client import HTTPConnection as _HTTPConnection
from .packages.six.moves.http_client import HTTPException  # noqa: F401
from .util.proxy import create_proxy_ssl_context

try:  # Compiled with SSL?
    import ssl

    BaseSSLError = ssl.SSLError
except (ImportError, AttributeError):  # Platform-specific: No SSL.
    ssl = None

    class BaseSSLError(BaseException):
        pass


try:
    # Python 3: not a no-op, we're adding this to the namespace so it can be imported.
    ConnectionError = ConnectionError
except NameError:
    # Python 2
    class ConnectionError(Exception):
        pass


try:  # Python 3:
    # Not a no-op, we're adding this to the namespace so it can be imported.
    BrokenPipeError = BrokenPipeError
except NameError:  # Python 2:

    class BrokenPipeError(Exception):
        pass


from ._collections import HTTPHeaderDict  # noqa (historical, removed in v2)
from ._version import __version__
from .exceptions import (
    ConnectTimeoutError,
    NewConnectionError,
    SubjectAltNameWarning,
    SystemTimeWarning,
)
from .util import SKIP_HEADER, SKIPPABLE_HEADERS, connection
from .util.ssl_ import (
    assert_fingerprint,
    create_urllib3_context,
    is_ipaddress,
    resolve_cert_reqs,
    resolve_ssl_version,
    ssl_wrap_socket,
)
from .util.ssl_match_hostname import CertificateError, match_hostname

log = logging.getLogger(__name__)

port_by_scheme = {"http": 80, "https": 443}

# When it comes time to update this value as a part of regular maintenance
# (ie test_recent_date is failing) update it to ~6 months before the current date.
RECENT_DATE = datetime.date(2024, 1, 1)

_CONTAINS_CONTROL_CHAR_RE = re.compile(r"[^-!#$%&'*+.^_`|~0-9a-zA-Z]")


class HTTPConnection(_HTTPConnection, object):
    """
    Based on :class:`http.client.HTTPConnection` but provides an extra constructor
    backwards-compatibility layer between older and newer Pythons.

    Additional keyword parameters are used to configure attributes of the connection.
    Accepted parameters include:

    - ``strict``: See the documentation on :class:`urllib3.connectionpool.HTTPConnectionPool`
    - ``source_address``: Set the source address for the current connection.
    - ``socket_options``: Set specific options on the underlying socket. If not specified, then
      defaults are loaded from ``HTTPConnection.default_socket_options`` which includes disabling
      Nagle's algorithm (sets TCP_NODELAY to 1) unless the connection is behind a proxy.

      For example, if you wish to enable TCP Keep Alive in addition to the defaults,
      you might pass:

      .. code-block:: python

         HTTPConnection.default_socket_options + [
             (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
         ]

      Or you may want to disable the defaults by passing an empty list (e.g., ``[]``).
    """

    default_port = port_by_scheme["http"]

    #: Disable Nagle's algorithm by default.
    #: ``[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]``
    default_socket_options = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]

    #: Whether this connection verifies the host's certificate.
    is_verified = False

    #: Whether this proxy connection (if used) verifies the proxy host's
    #: certificate.
    proxy_is_verified = None

    def __init__(self, *args, **kw):
        if not six.PY2:
            kw.pop("strict", None)

        # Pre-set source_address.
        self.source_address = kw.get("source_address")

        #: The socket options provided by the user. If no options are
        #: provided, we use the default options.
        self.socket_options = kw.pop("socket_options", self.default_socket_options)

        # Proxy options provided by the user.
        self.proxy = kw.pop("proxy", None)
        self.proxy_config = kw.pop("proxy_config", None)

        _HTTPConnection.__init__(self, *args, **kw)

    @property
    def host(self):
        """
        Getter method to remove any trailing dots that indicate the hostname is an FQDN.

        In general, SSL certificates don't include the trailing dot indicating a
        fully-qualified domain name, and thus, they don't validate properly when
        checked against a domain name that includes the dot. In addition, some
        servers may not expect to receive the trailing dot when provided.

        However, the hostname with trailing dot is critical to DNS resolution; doing a
        lookup with the trailing dot will properly only resolve the appropriate FQDN,
        whereas a lookup without a trailing dot will search the system's search domain
        list. Thus, it's important to keep the original host around for use only in
        those cases where it's appropriate (i.e., when doing DNS lookup to establish the
        actual TCP connection across which we're going to send HTTP requests).
        """
        return self._dns_host.rstrip(".")

    @host.setter
    def host(self, value):
        """
        Setter for the `host` property.

        We assume that only urllib3 uses the _dns_host attribute; httplib itself
        only uses `host`, and it seems reasonable that other libraries follow suit.
        """
        self._dns_host = value

    def _new_conn(self):
        """Establish a socket connection and set nodelay settings on it.

        :return: New socket connection.
        """
        extra_kw = {}
        if self.source_address:
            extra_kw["source_address"] = self.source_address

        if self.socket_options:
            extra_kw["socket_options"] = self.socket_options

        try:
            conn = connection.create_connection(
                (self._dns_host, self.port), self.timeout, **extra_kw
            )

        except SocketTimeout:
            raise ConnectTimeoutError(
                self,
                "Connection to %s timed out. (connect timeout=%s)"
                % (self.host, self.timeout),
            )

        except SocketError as e:
            raise NewConnectionError(
                self, "Failed to establish a new connection: %s" % e
            )

        return conn

    def _is_using_tunnel(self):
        # Google App Engine's httplib does not define _tunnel_host
        return getattr(self, "_tunnel_host", None)

    def _prepare_conn(self, conn):
        self.sock = conn
        if self._is_using_tunnel():
            # TODO: Fix tunnel so it doesn't depend on self.sock state.
            self._tunnel()
            # Mark this connection as not reusable
            self.auto_open = 0

    def connect(self):
        conn = self._new_conn()
        self._prepare_conn(conn)

    def putrequest(self, method, url, *args, **kwargs):
        """ """
        # Empty docstring because the indentation of CPython's implementation
        # is broken but we don't want this method in our documentation.
        match = _CONTAINS_CONTROL_CHAR_RE.search(method)
        if match:
            raise ValueError(
                "Method cannot contain non-token characters %r (found at least %r)"
                % (method, match.group())
            )

        return _HTTPConnection.putrequest(self, method, url, *args, **kwargs)

    def putheader(self, header, *values):
        """ """
        if not any(isinstance(v, str) and v == SKIP_HEADER for v in values):
            _HTTPConnection.putheader(self, header, *values)
        elif six.ensure_str(header.lower()) not in SKIPPABLE_HEADERS:
            raise ValueError(
                "urllib3.util.SKIP_HEADER only supports '%s'"
                % ("', '".join(map(str.title, sorted(SKIPPABLE_HEADERS))),)
            )

    def request(self, method, url, body=None, headers=None):
        # Update the inner socket's timeout value to send the request.
        # This only triggers if the connection is re-used.
        if getattr(self, "sock", None) is not None:
            self.sock.settimeout(self.timeout)

        if headers is None:
            headers = {}
        else:
            # Avoid modifying the headers passed into .request()
            headers = headers.copy()
        if "user-agent" not in (six.ensure_str(k.lower()) for k in headers):
            headers["User-Agent"] = _get_default_user_agent()
        super(HTTPConnection, self).request(method, url, body=body, headers=headers)

    def request_chunked(self, method, url, body=None, headers=None):
        """
        Alternative to the common request method, which sends the
        body with chunked encoding and not as one block
        """
        headers = headers or {}
        header_keys = set([six.ensure_str(k.lower()) for k in headers])
        skip_accept_encoding = "accept-encoding" in header_keys
        skip_host = "host" in header_keys
        self.putrequest(
            method, url, skip_accept_encoding=skip_accept_encoding, skip_host=skip_host
        )
        if "user-agent" not in header_keys:
            self.putheader("User-Agent", _get_default_user_agent())
        for header, value in headers.items():
            self.putheader(header, value)
        if "transfer-encoding" not in header_keys:
            self.putheader("Transfer-Encoding", "chunked")
        self.endheaders()

        if body is not None:
            stringish_types = six.string_types + (bytes,)
            if isinstance(body, stringish_types):
                body = (body,)
            for chunk in body:
                if not chunk:
                    continue
                if not isinstance(chunk, bytes):
                    chunk = chunk.encode("utf8")
                len_str = hex(len(chunk))[2:]
                to_send = bytearray(len_str.encode())
                to_send += b"\r\n"
                to_send += chunk
                to_send += b"\r\n"
                self.send(to_send)

        # After the if clause, to always have a closed body
        self.send(b"0\r\n\r\n")


class HTTPSConnection(HTTPConnection):
    """
    Many of the parameters to this constructor are passed to the underlying SSL
    socket by means of :py:func:`urllib3.util.ssl_wrap_socket`.
    """

    default_port = port_by_scheme["https"]

    cert_reqs = None
    ca_certs = None
    ca_cert_dir = None
    ca_cert_data = None
    ssl_version = None
    assert_fingerprint = None
    tls_in_tls_required = False

    def __init__(
        self,
        host,
        port=None,
        key_file=None,
        cert_file=None,
        key_password=None,
        strict=None,
        timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
        ssl_context=None,
        server_hostname=None,
        **kw
    ):

        HTTPConnection.__init__(self, host, port, strict=strict, timeout=timeout, **kw)

        self.key_file = key_file
        self.cert_file = cert_file
        self.key_password = key_password
        self.ssl_context = ssl_context
        self.server_hostname = server_hostname

        # Required property for Google AppEngine 1.9.0 which otherwise causes
        # HTTPS requests to go out as HTTP. (See Issue #356)
        self._protocol = "https"

    def set_cert(
        self,
        key_file=None,
        cert_file=None,
        cert_reqs=None,
        key_password=None,
        ca_certs=None,
        assert_hostname=None,
        assert_fingerprint=None,
        ca_cert_dir=None,
        ca_cert_data=None,
    ):
        """
        This method should only be called once, before the connection is used.
        """
        # If cert_reqs is not provided we'll assume CERT_REQUIRED unless we also
        # have an SSLContext object in which case we'll use its verify_mode.
        if cert_reqs is None:
            if self.ssl_context is not None:
                cert_reqs = self.ssl_context.verify_mode
            else:
                cert_reqs = resolve_cert_reqs(None)

        self.key_file = key_file
        self.cert_file = cert_file
        self.cert_reqs = cert_reqs
        self.key_password = key_password
        self.assert_hostname = assert_hostname
        self.assert_fingerprint = assert_fingerprint
        self.ca_certs = ca_certs and os.path.expanduser(ca_certs)
        self.ca_cert_dir = ca_cert_dir and os.path.expanduser(ca_cert_dir)
        self.ca_cert_data = ca_cert_data

    def connect(self):
        # Add certificate verification
        self.sock = conn = self._new_conn()
        hostname = self.host
        tls_in_tls = False

        if self._is_using_tunnel():
            if self.tls_in_tls_required:
                self.sock = conn = self._connect_tls_proxy(hostname, conn)
                tls_in_tls = True

            # Calls self._set_hostport(), so self.host is
            # self._tunnel_host below.
            self._tunnel()
            # Mark this connection as not reusable
            self.auto_open = 0

            # Override the host with the one we're requesting data from.
            hostname = self._tunnel_host

        server_hostname = hostname
        if self.server_hostname is not None:
            server_hostname = self.server_hostname

        is_time_off = datetime.date.today() < RECENT_DATE
        if is_time_off:
            warnings.warn(
                (
                    "System time is way off (before {0}). This will probably "
                    "lead to SSL verification errors"
                ).format(RECENT_DATE),
                SystemTimeWarning,
            )

        # Wrap socket using verification with the root certs in
        # trusted_root_certs
        default_ssl_context = False
        if self.ssl_context is None:
            default_ssl_context = True
            self.ssl_context = create_urllib3_context(
                ssl_version=resolve_ssl_version(self.ssl_version),
                cert_reqs=resolve_cert_reqs(self.cert_reqs),
            )

        context = self.ssl_context
        context.verify_mode = resolve_cert_reqs(self.cert_reqs)

        # Try to load OS default certs if none are given.
        # Works well on Windows (requires Python3.4+)
        if (
            not self.ca_certs
            and not self.ca_cert_dir
            and not self.ca_cert_data
            and default_ssl_context
            and hasattr(context, "load_default_certs")
        ):
            context.load_default_certs()

        self.sock = ssl_wrap_socket(
            sock=conn,
            keyfile=self.key_file,
            certfile=self.cert_file,
            key_password=self.key_password,
            ca_certs=self.ca_certs,
            ca_cert_dir=self.ca_cert_dir,
            ca_cert_data=self.ca_cert_data,
            server_hostname=server_hostname,
            ssl_context=context,
            tls_in_tls=tls_in_tls,
        )

        # If we're using all defaults and the connection
        # is TLSv1 or TLSv1.1 we throw a DeprecationWarning
        # for the host.
        if (
            default_ssl_context
            and self.ssl_version is None
            and hasattr(self.sock, "version")
            and self.sock.version() in {"TLSv1", "TLSv1.1"}
        ):  # Defensive:
            warnings.warn(
                "Negotiating TLSv1/TLSv1.1 by default is deprecated "
                "and will be disabled in urllib3 v2.0.0. Connecting to "
                "'%s' with '%s' can be enabled by explicitly opting-in "
                "with 'ssl_version'" % (self.host, self.sock.version()),
                DeprecationWarning,
            )

        if self.assert_fingerprint:
            assert_fingerprint(
                self.sock.getpeercert(binary_form=True), self.assert_fingerprint
            )
        elif (
            context.verify_mode != ssl.CERT_NONE
            and not getattr(context, "check_hostname", False)
            and self.assert_hostname is not False
        ):
            # While urllib3 attempts to always turn off hostname matching from
            # the TLS library, this cannot always be done. So we check whether
            # the TLS Library still thinks it's matching hostnames.
            cert = self.sock.getpeercert()
            if not cert.get("subjectAltName", ()):
                warnings.warn(
                    (
                        "Certificate for {0} has no `subjectAltName`, falling back to check for a "
                        "`commonName` for now. This feature is being removed by major browsers and "
                        "deprecated by RFC 2818. (See https://github.com/urllib3/urllib3/issues/497 "
                        "for details.)".format(hostname)
                    ),
                    SubjectAltNameWarning,
                )
            _match_hostname(cert, self.assert_hostname or server_hostname)

        self.is_verified = (
            context.verify_mode == ssl.CERT_REQUIRED
            or self.assert_fingerprint is not None
        )

    def _connect_tls_proxy(self, hostname, conn):
        """
        Establish a TLS connection to the proxy using the provided SSL context.
        """
        proxy_config = self.proxy_config
        ssl_context = proxy_config.ssl_context
        if ssl_context:
            # If the user provided a proxy context, we assume CA and client
            # certificates have already been set
            return ssl_wrap_socket(
                sock=conn,
                server_hostname=hostname,
                ssl_context=ssl_context,
            )

        ssl_context = create_proxy_ssl_context(
            self.ssl_version,
            self.cert_reqs,
            self.ca_certs,
            self.ca_cert_dir,
            self.ca_cert_data,
        )

        # If no cert was provided, use only the default options for server
        # certificate validation
        socket = ssl_wrap_socket(
            sock=conn,
            ca_certs=self.ca_certs,
            ca_cert_dir=self.ca_cert_dir,
            ca_cert_data=self.ca_cert_data,
            server_hostname=hostname,
            ssl_context=ssl_context,
        )

        if ssl_context.verify_mode != ssl.CERT_NONE and not getattr(
            ssl_context, "check_hostname", False
        ):
            # While urllib3 attempts to always turn off hostname matching from
            # the TLS library, this cannot always be done. So we check whether
            # the TLS Library still thinks it's matching hostnames.
            cert = socket.getpeercert()
            if not cert.get("subjectAltName", ()):
                warnings.warn(
                    (
                        "Certificate for {0} has no `subjectAltName`, falling back to check for a "
                        "`commonName` for now. This feature is being removed by major browsers and "
                        "deprecated by RFC 2818. (See https://github.com/urllib3/urllib3/issues/497 "
                        "for details.)".format(hostname)
                    ),
                    SubjectAltNameWarning,
                )
            _match_hostname(cert, hostname)

        self.proxy_is_verified = ssl_context.verify_mode == ssl.CERT_REQUIRED
        return socket


def _match_hostname(cert, asserted_hostname):
    # Our upstream implementation of ssl.match_hostname()
    # only applies this normalization to IP addresses so it doesn't
    # match DNS SANs so we do the same thing!
    stripped_hostname = asserted_hostname.strip("u[]")
    if is_ipaddress(stripped_hostname):
        asserted_hostname = stripped_hostname

    try:
        match_hostname(cert, asserted_hostname)
    except CertificateError as e:
        log.warning(
            "Certificate did not match expected hostname: %s. Certificate: %s",
            asserted_hostname,
            cert,
        )
        # Add cert to exception and reraise so client code can inspect
        # the cert when catching the exception, if they want to
        e._peer_cert = cert
        raise


def _get_default_user_agent():
    return "python-urllib3/%s" % __version__


class DummyConnection(object):
    """Used to detect a failed ConnectionCls import."""

    pass


if not ssl:
    HTTPSConnection = DummyConnection  # noqa: F811


VerifiedHTTPSConnection = HTTPSConnection

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\reshape\encoding.py ===
from __future__ import annotations

from collections import defaultdict
from collections.abc import (
    Hashable,
    Iterable,
)
import itertools
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._libs import missing as libmissing
from pandas._libs.sparse import IntIndex

from pandas.core.dtypes.common import (
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
)

from pandas.core.arrays import SparseArray
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.arrays.string_ import StringDtype
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
    Index,
    default_index,
)
from pandas.core.series import Series

if TYPE_CHECKING:
    from pandas._typing import NpDtype


def get_dummies(
    data,
    prefix=None,
    prefix_sep: str | Iterable[str] | dict[str, str] = "_",
    dummy_na: bool = False,
    columns=None,
    sparse: bool = False,
    drop_first: bool = False,
    dtype: NpDtype | None = None,
) -> DataFrame:
    """
    Convert categorical variable into dummy/indicator variables.

    Each variable is converted in as many 0/1 variables as there are different
    values. Columns in the output are each named after a value; if the input is
    a DataFrame, the name of the original variable is prepended to the value.

    Parameters
    ----------
    data : array-like, Series, or DataFrame
        Data of which to get dummy indicators.
    prefix : str, list of str, or dict of str, default None
        String to append DataFrame column names.
        Pass a list with length equal to the number of columns
        when calling get_dummies on a DataFrame. Alternatively, `prefix`
        can be a dictionary mapping column names to prefixes.
    prefix_sep : str, default '_'
        If appending prefix, separator/delimiter to use. Or pass a
        list or dictionary as with `prefix`.
    dummy_na : bool, default False
        Add a column to indicate NaNs, if False NaNs are ignored.
    columns : list-like, default None
        Column names in the DataFrame to be encoded.
        If `columns` is None then all the columns with
        `object`, `string`, or `category` dtype will be converted.
    sparse : bool, default False
        Whether the dummy-encoded columns should be backed by
        a :class:`SparseArray` (True) or a regular NumPy array (False).
    drop_first : bool, default False
        Whether to get k-1 dummies out of k categorical levels by removing the
        first level.
    dtype : dtype, default bool
        Data type for new columns. Only a single dtype is allowed.

    Returns
    -------
    DataFrame
        Dummy-coded data. If `data` contains other columns than the
        dummy-coded one(s), these will be prepended, unaltered, to the result.

    See Also
    --------
    Series.str.get_dummies : Convert Series of strings to dummy codes.
    :func:`~pandas.from_dummies` : Convert dummy codes to categorical ``DataFrame``.

    Notes
    -----
    Reference :ref:`the user guide <reshaping.dummies>` for more examples.

    Examples
    --------
    >>> s = pd.Series(list('abca'))

    >>> pd.get_dummies(s)
           a      b      c
    0   True  False  False
    1  False   True  False
    2  False  False   True
    3   True  False  False

    >>> s1 = ['a', 'b', np.nan]

    >>> pd.get_dummies(s1)
           a      b
    0   True  False
    1  False   True
    2  False  False

    >>> pd.get_dummies(s1, dummy_na=True)
           a      b    NaN
    0   True  False  False
    1  False   True  False
    2  False  False   True

    >>> df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
    ...                    'C': [1, 2, 3]})

    >>> pd.get_dummies(df, prefix=['col1', 'col2'])
       C  col1_a  col1_b  col2_a  col2_b  col2_c
    0  1    True   False   False    True   False
    1  2   False    True    True   False   False
    2  3    True   False   False   False    True

    >>> pd.get_dummies(pd.Series(list('abcaa')))
           a      b      c
    0   True  False  False
    1  False   True  False
    2  False  False   True
    3   True  False  False
    4   True  False  False

    >>> pd.get_dummies(pd.Series(list('abcaa')), drop_first=True)
           b      c
    0  False  False
    1   True  False
    2  False   True
    3  False  False
    4  False  False

    >>> pd.get_dummies(pd.Series(list('abc')), dtype=float)
         a    b    c
    0  1.0  0.0  0.0
    1  0.0  1.0  0.0
    2  0.0  0.0  1.0
    """
    from pandas.core.reshape.concat import concat

    dtypes_to_encode = ["object", "string", "category"]

    if isinstance(data, DataFrame):
        # determine columns being encoded
        if columns is None:
            data_to_encode = data.select_dtypes(include=dtypes_to_encode)
        elif not is_list_like(columns):
            raise TypeError("Input must be a list-like for parameter `columns`")
        else:
            data_to_encode = data[columns]

        # validate prefixes and separator to avoid silently dropping cols
        def check_len(item, name: str):
            if is_list_like(item):
                if not len(item) == data_to_encode.shape[1]:
                    len_msg = (
                        f"Length of '{name}' ({len(item)}) did not match the "
                        "length of the columns being encoded "
                        f"({data_to_encode.shape[1]})."
                    )
                    raise ValueError(len_msg)

        check_len(prefix, "prefix")
        check_len(prefix_sep, "prefix_sep")

        if isinstance(prefix, str):
            prefix = itertools.cycle([prefix])
        if isinstance(prefix, dict):
            prefix = [prefix[col] for col in data_to_encode.columns]

        if prefix is None:
            prefix = data_to_encode.columns

        # validate separators
        if isinstance(prefix_sep, str):
            prefix_sep = itertools.cycle([prefix_sep])
        elif isinstance(prefix_sep, dict):
            prefix_sep = [prefix_sep[col] for col in data_to_encode.columns]

        with_dummies: list[DataFrame]
        if data_to_encode.shape == data.shape:
            # Encoding the entire df, do not prepend any dropped columns
            with_dummies = []
        elif columns is not None:
            # Encoding only cols specified in columns. Get all cols not in
            # columns to prepend to result.
            with_dummies = [data.drop(columns, axis=1)]
        else:
            # Encoding only object and category dtype columns. Get remaining
            # columns to prepend to result.
            with_dummies = [data.select_dtypes(exclude=dtypes_to_encode)]

        for col, pre, sep in zip(data_to_encode.items(), prefix, prefix_sep):
            # col is (column_name, column), use just column data here
            dummy = _get_dummies_1d(
                col[1],
                prefix=pre,
                prefix_sep=sep,
                dummy_na=dummy_na,
                sparse=sparse,
                drop_first=drop_first,
                dtype=dtype,
            )
            with_dummies.append(dummy)
        result = concat(with_dummies, axis=1)
    else:
        result = _get_dummies_1d(
            data,
            prefix,
            prefix_sep,
            dummy_na,
            sparse=sparse,
            drop_first=drop_first,
            dtype=dtype,
        )
    return result


def _get_dummies_1d(
    data,
    prefix,
    prefix_sep: str | Iterable[str] | dict[str, str] = "_",
    dummy_na: bool = False,
    sparse: bool = False,
    drop_first: bool = False,
    dtype: NpDtype | None = None,
) -> DataFrame:
    from pandas.core.reshape.concat import concat

    # Series avoids inconsistent NaN handling
    codes, levels = factorize_from_iterable(Series(data, copy=False))

    if dtype is None and hasattr(data, "dtype"):
        input_dtype = data.dtype
        if isinstance(input_dtype, CategoricalDtype):
            input_dtype = input_dtype.categories.dtype

        if isinstance(input_dtype, ArrowDtype):
            import pyarrow as pa

            dtype = ArrowDtype(pa.bool_())  # type: ignore[assignment]
        elif (
            isinstance(input_dtype, StringDtype)
            and input_dtype.na_value is libmissing.NA
        ):
            dtype = pandas_dtype("boolean")  # type: ignore[assignment]
        else:
            dtype = np.dtype(bool)
    elif dtype is None:
        dtype = np.dtype(bool)

    _dtype = pandas_dtype(dtype)

    if is_object_dtype(_dtype):
        raise ValueError("dtype=object is not a valid dtype for get_dummies")

    def get_empty_frame(data) -> DataFrame:
        index: Index | np.ndarray
        if isinstance(data, Series):
            index = data.index
        else:
            index = default_index(len(data))
        return DataFrame(index=index)

    # if all NaN
    if not dummy_na and len(levels) == 0:
        return get_empty_frame(data)

    codes = codes.copy()
    if dummy_na:
        codes[codes == -1] = len(levels)
        levels = levels.insert(len(levels), np.nan)

    # if dummy_na, we just fake a nan level. drop_first will drop it again
    if drop_first and len(levels) == 1:
        return get_empty_frame(data)

    number_of_cols = len(levels)

    if prefix is None:
        dummy_cols = levels
    else:
        dummy_cols = Index([f"{prefix}{prefix_sep}{level}" for level in levels])

    index: Index | None
    if isinstance(data, Series):
        index = data.index
    else:
        index = None

    if sparse:
        fill_value: bool | float
        if is_integer_dtype(dtype):
            fill_value = 0
        elif dtype == np.dtype(bool):
            fill_value = False
        else:
            fill_value = 0.0

        sparse_series = []
        N = len(data)
        sp_indices: list[list] = [[] for _ in range(len(dummy_cols))]
        mask = codes != -1
        codes = codes[mask]
        n_idx = np.arange(N)[mask]

        for ndx, code in zip(n_idx, codes):
            sp_indices[code].append(ndx)

        if drop_first:
            # remove first categorical level to avoid perfect collinearity
            # GH12042
            sp_indices = sp_indices[1:]
            dummy_cols = dummy_cols[1:]
        for col, ixs in zip(dummy_cols, sp_indices):
            sarr = SparseArray(
                np.ones(len(ixs), dtype=dtype),
                sparse_index=IntIndex(N, ixs),
                fill_value=fill_value,
                dtype=dtype,
            )
            sparse_series.append(Series(data=sarr, index=index, name=col, copy=False))

        return concat(sparse_series, axis=1, copy=False)

    else:
        # ensure ndarray layout is column-major
        shape = len(codes), number_of_cols
        dummy_dtype: NpDtype
        if isinstance(_dtype, np.dtype):
            dummy_dtype = _dtype
        else:
            dummy_dtype = np.bool_
        dummy_mat = np.zeros(shape=shape, dtype=dummy_dtype, order="F")
        dummy_mat[np.arange(len(codes)), codes] = 1

        if not dummy_na:
            # reset NaN GH4446
            dummy_mat[codes == -1] = 0

        if drop_first:
            # remove first GH12042
            dummy_mat = dummy_mat[:, 1:]
            dummy_cols = dummy_cols[1:]
        return DataFrame(dummy_mat, index=index, columns=dummy_cols, dtype=_dtype)


def from_dummies(
    data: DataFrame,
    sep: None | str = None,
    default_category: None | Hashable | dict[str, Hashable] = None,
) -> DataFrame:
    """
    Create a categorical ``DataFrame`` from a ``DataFrame`` of dummy variables.

    Inverts the operation performed by :func:`~pandas.get_dummies`.

    .. versionadded:: 1.5.0

    Parameters
    ----------
    data : DataFrame
        Data which contains dummy-coded variables in form of integer columns of
        1's and 0's.
    sep : str, default None
        Separator used in the column names of the dummy categories they are
        character indicating the separation of the categorical names from the prefixes.
        For example, if your column names are 'prefix_A' and 'prefix_B',
        you can strip the underscore by specifying sep='_'.
    default_category : None, Hashable or dict of Hashables, default None
        The default category is the implied category when a value has none of the
        listed categories specified with a one, i.e. if all dummies in a row are
        zero. Can be a single value for all variables or a dict directly mapping
        the default categories to a prefix of a variable.

    Returns
    -------
    DataFrame
        Categorical data decoded from the dummy input-data.

    Raises
    ------
    ValueError
        * When the input ``DataFrame`` ``data`` contains NA values.
        * When the input ``DataFrame`` ``data`` contains column names with separators
          that do not match the separator specified with ``sep``.
        * When a ``dict`` passed to ``default_category`` does not include an implied
          category for each prefix.
        * When a value in ``data`` has more than one category assigned to it.
        * When ``default_category=None`` and a value in ``data`` has no category
          assigned to it.
    TypeError
        * When the input ``data`` is not of type ``DataFrame``.
        * When the input ``DataFrame`` ``data`` contains non-dummy data.
        * When the passed ``sep`` is of a wrong data type.
        * When the passed ``default_category`` is of a wrong data type.

    See Also
    --------
    :func:`~pandas.get_dummies` : Convert ``Series`` or ``DataFrame`` to dummy codes.
    :class:`~pandas.Categorical` : Represent a categorical variable in classic.

    Notes
    -----
    The columns of the passed dummy data should only include 1's and 0's,
    or boolean values.

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 0, 0, 1], "b": [0, 1, 0, 0],
    ...                    "c": [0, 0, 1, 0]})

    >>> df
       a  b  c
    0  1  0  0
    1  0  1  0
    2  0  0  1
    3  1  0  0

    >>> pd.from_dummies(df)
    0     a
    1     b
    2     c
    3     a

    >>> df = pd.DataFrame({"col1_a": [1, 0, 1], "col1_b": [0, 1, 0],
    ...                    "col2_a": [0, 1, 0], "col2_b": [1, 0, 0],
    ...                    "col2_c": [0, 0, 1]})

    >>> df
          col1_a  col1_b  col2_a  col2_b  col2_c
    0       1       0       0       1       0
    1       0       1       1       0       0
    2       1       0       0       0       1

    >>> pd.from_dummies(df, sep="_")
        col1    col2
    0    a       b
    1    b       a
    2    a       c

    >>> df = pd.DataFrame({"col1_a": [1, 0, 0], "col1_b": [0, 1, 0],
    ...                    "col2_a": [0, 1, 0], "col2_b": [1, 0, 0],
    ...                    "col2_c": [0, 0, 0]})

    >>> df
          col1_a  col1_b  col2_a  col2_b  col2_c
    0       1       0       0       1       0
    1       0       1       1       0       0
    2       0       0       0       0       0

    >>> pd.from_dummies(df, sep="_", default_category={"col1": "d", "col2": "e"})
        col1    col2
    0    a       b
    1    b       a
    2    d       e
    """
    from pandas.core.reshape.concat import concat

    if not isinstance(data, DataFrame):
        raise TypeError(
            "Expected 'data' to be a 'DataFrame'; "
            f"Received 'data' of type: {type(data).__name__}"
        )

    col_isna_mask = cast(Series, data.isna().any())

    if col_isna_mask.any():
        raise ValueError(
            "Dummy DataFrame contains NA value in column: "
            f"'{col_isna_mask.idxmax()}'"
        )

    # index data with a list of all columns that are dummies
    try:
        data_to_decode = data.astype("boolean", copy=False)
    except TypeError:
        raise TypeError("Passed DataFrame contains non-dummy data")

    # collect prefixes and get lists to slice data for each prefix
    variables_slice = defaultdict(list)
    if sep is None:
        variables_slice[""] = list(data.columns)
    elif isinstance(sep, str):
        for col in data_to_decode.columns:
            prefix = col.split(sep)[0]
            if len(prefix) == len(col):
                raise ValueError(f"Separator not specified for column: {col}")
            variables_slice[prefix].append(col)
    else:
        raise TypeError(
            "Expected 'sep' to be of type 'str' or 'None'; "
            f"Received 'sep' of type: {type(sep).__name__}"
        )

    if default_category is not None:
        if isinstance(default_category, dict):
            if not len(default_category) == len(variables_slice):
                len_msg = (
                    f"Length of 'default_category' ({len(default_category)}) "
                    f"did not match the length of the columns being encoded "
                    f"({len(variables_slice)})"
                )
                raise ValueError(len_msg)
        elif isinstance(default_category, Hashable):
            default_category = dict(
                zip(variables_slice, [default_category] * len(variables_slice))
            )
        else:
            raise TypeError(
                "Expected 'default_category' to be of type "
                "'None', 'Hashable', or 'dict'; "
                "Received 'default_category' of type: "
                f"{type(default_category).__name__}"
            )

    cat_data = {}
    for prefix, prefix_slice in variables_slice.items():
        if sep is None:
            cats = prefix_slice.copy()
        else:
            cats = [col[len(prefix + sep) :] for col in prefix_slice]
        assigned = data_to_decode.loc[:, prefix_slice].sum(axis=1)
        if any(assigned > 1):
            raise ValueError(
                "Dummy DataFrame contains multi-assignment(s); "
                f"First instance in row: {assigned.idxmax()}"
            )
        if any(assigned == 0):
            if isinstance(default_category, dict):
                cats.append(default_category[prefix])
            else:
                raise ValueError(
                    "Dummy DataFrame contains unassigned value(s); "
                    f"First instance in row: {assigned.idxmin()}"
                )
            data_slice = concat(
                (data_to_decode.loc[:, prefix_slice], assigned == 0), axis=1
            )
        else:
            data_slice = data_to_decode.loc[:, prefix_slice]
        cats_array = data._constructor_sliced(cats, dtype=data.columns.dtype)
        # get indices of True entries along axis=1
        true_values = data_slice.idxmax(axis=1)
        indexer = data_slice.columns.get_indexer_for(true_values)
        cat_data[prefix] = cats_array.take(indexer).set_axis(data.index)

    result = DataFrame(cat_data)
    if sep is not None:
        result.columns = result.columns.astype(data.columns.dtype)
    return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\truststore\_macos.py ===
import contextlib
import ctypes
import platform
import ssl
import typing
from ctypes import (
    CDLL,
    POINTER,
    c_bool,
    c_char_p,
    c_int32,
    c_long,
    c_uint32,
    c_ulong,
    c_void_p,
)
from ctypes.util import find_library

from ._ssl_constants import _set_ssl_context_verify_mode

_mac_version = platform.mac_ver()[0]
_mac_version_info = tuple(map(int, _mac_version.split(".")))
if _mac_version_info < (10, 8):
    raise ImportError(
        f"Only OS X 10.8 and newer are supported, not {_mac_version_info[0]}.{_mac_version_info[1]}"
    )

_is_macos_version_10_14_or_later = _mac_version_info >= (10, 14)


def _load_cdll(name: str, macos10_16_path: str) -> CDLL:
    """Loads a CDLL by name, falling back to known path on 10.16+"""
    try:
        # Big Sur is technically 11 but we use 10.16 due to the Big Sur
        # beta being labeled as 10.16.
        path: str | None
        if _mac_version_info >= (10, 16):
            path = macos10_16_path
        else:
            path = find_library(name)
        if not path:
            raise OSError  # Caught and reraised as 'ImportError'
        return CDLL(path, use_errno=True)
    except OSError:
        raise ImportError(f"The library {name} failed to load") from None


Security = _load_cdll(
    "Security", "/System/Library/Frameworks/Security.framework/Security"
)
CoreFoundation = _load_cdll(
    "CoreFoundation",
    "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation",
)

Boolean = c_bool
CFIndex = c_long
CFStringEncoding = c_uint32
CFData = c_void_p
CFString = c_void_p
CFArray = c_void_p
CFMutableArray = c_void_p
CFError = c_void_p
CFType = c_void_p
CFTypeID = c_ulong
CFTypeRef = POINTER(CFType)
CFAllocatorRef = c_void_p

OSStatus = c_int32

CFErrorRef = POINTER(CFError)
CFDataRef = POINTER(CFData)
CFStringRef = POINTER(CFString)
CFArrayRef = POINTER(CFArray)
CFMutableArrayRef = POINTER(CFMutableArray)
CFArrayCallBacks = c_void_p
CFOptionFlags = c_uint32

SecCertificateRef = POINTER(c_void_p)
SecPolicyRef = POINTER(c_void_p)
SecTrustRef = POINTER(c_void_p)
SecTrustResultType = c_uint32
SecTrustOptionFlags = c_uint32

try:
    Security.SecCertificateCreateWithData.argtypes = [CFAllocatorRef, CFDataRef]
    Security.SecCertificateCreateWithData.restype = SecCertificateRef

    Security.SecCertificateCopyData.argtypes = [SecCertificateRef]
    Security.SecCertificateCopyData.restype = CFDataRef

    Security.SecCopyErrorMessageString.argtypes = [OSStatus, c_void_p]
    Security.SecCopyErrorMessageString.restype = CFStringRef

    Security.SecTrustSetAnchorCertificates.argtypes = [SecTrustRef, CFArrayRef]
    Security.SecTrustSetAnchorCertificates.restype = OSStatus

    Security.SecTrustSetAnchorCertificatesOnly.argtypes = [SecTrustRef, Boolean]
    Security.SecTrustSetAnchorCertificatesOnly.restype = OSStatus

    Security.SecPolicyCreateRevocation.argtypes = [CFOptionFlags]
    Security.SecPolicyCreateRevocation.restype = SecPolicyRef

    Security.SecPolicyCreateSSL.argtypes = [Boolean, CFStringRef]
    Security.SecPolicyCreateSSL.restype = SecPolicyRef

    Security.SecTrustCreateWithCertificates.argtypes = [
        CFTypeRef,
        CFTypeRef,
        POINTER(SecTrustRef),
    ]
    Security.SecTrustCreateWithCertificates.restype = OSStatus

    Security.SecTrustGetTrustResult.argtypes = [
        SecTrustRef,
        POINTER(SecTrustResultType),
    ]
    Security.SecTrustGetTrustResult.restype = OSStatus

    Security.SecTrustEvaluate.argtypes = [
        SecTrustRef,
        POINTER(SecTrustResultType),
    ]
    Security.SecTrustEvaluate.restype = OSStatus

    Security.SecTrustRef = SecTrustRef  # type: ignore[attr-defined]
    Security.SecTrustResultType = SecTrustResultType  # type: ignore[attr-defined]
    Security.OSStatus = OSStatus  # type: ignore[attr-defined]

    kSecRevocationUseAnyAvailableMethod = 3
    kSecRevocationRequirePositiveResponse = 8

    CoreFoundation.CFRelease.argtypes = [CFTypeRef]
    CoreFoundation.CFRelease.restype = None

    CoreFoundation.CFGetTypeID.argtypes = [CFTypeRef]
    CoreFoundation.CFGetTypeID.restype = CFTypeID

    CoreFoundation.CFStringCreateWithCString.argtypes = [
        CFAllocatorRef,
        c_char_p,
        CFStringEncoding,
    ]
    CoreFoundation.CFStringCreateWithCString.restype = CFStringRef

    CoreFoundation.CFStringGetCStringPtr.argtypes = [CFStringRef, CFStringEncoding]
    CoreFoundation.CFStringGetCStringPtr.restype = c_char_p

    CoreFoundation.CFStringGetCString.argtypes = [
        CFStringRef,
        c_char_p,
        CFIndex,
        CFStringEncoding,
    ]
    CoreFoundation.CFStringGetCString.restype = c_bool

    CoreFoundation.CFDataCreate.argtypes = [CFAllocatorRef, c_char_p, CFIndex]
    CoreFoundation.CFDataCreate.restype = CFDataRef

    CoreFoundation.CFDataGetLength.argtypes = [CFDataRef]
    CoreFoundation.CFDataGetLength.restype = CFIndex

    CoreFoundation.CFDataGetBytePtr.argtypes = [CFDataRef]
    CoreFoundation.CFDataGetBytePtr.restype = c_void_p

    CoreFoundation.CFArrayCreate.argtypes = [
        CFAllocatorRef,
        POINTER(CFTypeRef),
        CFIndex,
        CFArrayCallBacks,
    ]
    CoreFoundation.CFArrayCreate.restype = CFArrayRef

    CoreFoundation.CFArrayCreateMutable.argtypes = [
        CFAllocatorRef,
        CFIndex,
        CFArrayCallBacks,
    ]
    CoreFoundation.CFArrayCreateMutable.restype = CFMutableArrayRef

    CoreFoundation.CFArrayAppendValue.argtypes = [CFMutableArrayRef, c_void_p]
    CoreFoundation.CFArrayAppendValue.restype = None

    CoreFoundation.CFArrayGetCount.argtypes = [CFArrayRef]
    CoreFoundation.CFArrayGetCount.restype = CFIndex

    CoreFoundation.CFArrayGetValueAtIndex.argtypes = [CFArrayRef, CFIndex]
    CoreFoundation.CFArrayGetValueAtIndex.restype = c_void_p

    CoreFoundation.CFErrorGetCode.argtypes = [CFErrorRef]
    CoreFoundation.CFErrorGetCode.restype = CFIndex

    CoreFoundation.CFErrorCopyDescription.argtypes = [CFErrorRef]
    CoreFoundation.CFErrorCopyDescription.restype = CFStringRef

    CoreFoundation.kCFAllocatorDefault = CFAllocatorRef.in_dll(  # type: ignore[attr-defined]
        CoreFoundation, "kCFAllocatorDefault"
    )
    CoreFoundation.kCFTypeArrayCallBacks = c_void_p.in_dll(  # type: ignore[attr-defined]
        CoreFoundation, "kCFTypeArrayCallBacks"
    )

    CoreFoundation.CFTypeRef = CFTypeRef  # type: ignore[attr-defined]
    CoreFoundation.CFArrayRef = CFArrayRef  # type: ignore[attr-defined]
    CoreFoundation.CFStringRef = CFStringRef  # type: ignore[attr-defined]
    CoreFoundation.CFErrorRef = CFErrorRef  # type: ignore[attr-defined]

except AttributeError as e:
    raise ImportError(f"Error initializing ctypes: {e}") from None

# SecTrustEvaluateWithError is macOS 10.14+
if _is_macos_version_10_14_or_later:
    try:
        Security.SecTrustEvaluateWithError.argtypes = [
            SecTrustRef,
            POINTER(CFErrorRef),
        ]
        Security.SecTrustEvaluateWithError.restype = c_bool
    except AttributeError as e:
        raise ImportError(f"Error initializing ctypes: {e}") from None


def _handle_osstatus(result: OSStatus, _: typing.Any, args: typing.Any) -> typing.Any:
    """
    Raises an error if the OSStatus value is non-zero.
    """
    if int(result) == 0:
        return args

    # Returns a CFString which we need to transform
    # into a UTF-8 Python string.
    error_message_cfstring = None
    try:
        error_message_cfstring = Security.SecCopyErrorMessageString(result, None)

        # First step is convert the CFString into a C string pointer.
        # We try the fast no-copy way first.
        error_message_cfstring_c_void_p = ctypes.cast(
            error_message_cfstring, ctypes.POINTER(ctypes.c_void_p)
        )
        message = CoreFoundation.CFStringGetCStringPtr(
            error_message_cfstring_c_void_p, CFConst.kCFStringEncodingUTF8
        )

        # Quoting the Apple dev docs:
        #
        # "A pointer to a C string or NULL if the internal
        # storage of theString does not allow this to be
        # returned efficiently."
        #
        # So we need to get our hands dirty.
        if message is None:
            buffer = ctypes.create_string_buffer(1024)
            result = CoreFoundation.CFStringGetCString(
                error_message_cfstring_c_void_p,
                buffer,
                1024,
                CFConst.kCFStringEncodingUTF8,
            )
            if not result:
                raise OSError("Error copying C string from CFStringRef")
            message = buffer.value

    finally:
        if error_message_cfstring is not None:
            CoreFoundation.CFRelease(error_message_cfstring)

    # If no message can be found for this status we come
    # up with a generic one that forwards the status code.
    if message is None or message == "":
        message = f"SecureTransport operation returned a non-zero OSStatus: {result}"

    raise ssl.SSLError(message)


Security.SecTrustCreateWithCertificates.errcheck = _handle_osstatus  # type: ignore[assignment]
Security.SecTrustSetAnchorCertificates.errcheck = _handle_osstatus  # type: ignore[assignment]
Security.SecTrustSetAnchorCertificatesOnly.errcheck = _handle_osstatus  # type: ignore[assignment]
Security.SecTrustGetTrustResult.errcheck = _handle_osstatus  # type: ignore[assignment]
Security.SecTrustEvaluate.errcheck = _handle_osstatus  # type: ignore[assignment]


class CFConst:
    """CoreFoundation constants"""

    kCFStringEncodingUTF8 = CFStringEncoding(0x08000100)

    errSecIncompleteCertRevocationCheck = -67635
    errSecHostNameMismatch = -67602
    errSecCertificateExpired = -67818
    errSecNotTrusted = -67843


def _bytes_to_cf_data_ref(value: bytes) -> CFDataRef:  # type: ignore[valid-type]
    return CoreFoundation.CFDataCreate(  # type: ignore[no-any-return]
        CoreFoundation.kCFAllocatorDefault, value, len(value)
    )


def _bytes_to_cf_string(value: bytes) -> CFString:
    """
    Given a Python binary data, create a CFString.
    The string must be CFReleased by the caller.
    """
    c_str = ctypes.c_char_p(value)
    cf_str = CoreFoundation.CFStringCreateWithCString(
        CoreFoundation.kCFAllocatorDefault,
        c_str,
        CFConst.kCFStringEncodingUTF8,
    )
    return cf_str  # type: ignore[no-any-return]


def _cf_string_ref_to_str(cf_string_ref: CFStringRef) -> str | None:  # type: ignore[valid-type]
    """
    Creates a Unicode string from a CFString object. Used entirely for error
    reporting.
    Yes, it annoys me quite a lot that this function is this complex.
    """

    string = CoreFoundation.CFStringGetCStringPtr(
        cf_string_ref, CFConst.kCFStringEncodingUTF8
    )
    if string is None:
        buffer = ctypes.create_string_buffer(1024)
        result = CoreFoundation.CFStringGetCString(
            cf_string_ref, buffer, 1024, CFConst.kCFStringEncodingUTF8
        )
        if not result:
            raise OSError("Error copying C string from CFStringRef")
        string = buffer.value
    if string is not None:
        string = string.decode("utf-8")
    return string  # type: ignore[no-any-return]


def _der_certs_to_cf_cert_array(certs: list[bytes]) -> CFMutableArrayRef:  # type: ignore[valid-type]
    """Builds a CFArray of SecCertificateRefs from a list of DER-encoded certificates.
    Responsibility of the caller to call CoreFoundation.CFRelease on the CFArray.
    """
    cf_array = CoreFoundation.CFArrayCreateMutable(
        CoreFoundation.kCFAllocatorDefault,
        0,
        ctypes.byref(CoreFoundation.kCFTypeArrayCallBacks),
    )
    if not cf_array:
        raise MemoryError("Unable to allocate memory!")

    for cert_data in certs:
        cf_data = None
        sec_cert_ref = None
        try:
            cf_data = _bytes_to_cf_data_ref(cert_data)
            sec_cert_ref = Security.SecCertificateCreateWithData(
                CoreFoundation.kCFAllocatorDefault, cf_data
            )
            CoreFoundation.CFArrayAppendValue(cf_array, sec_cert_ref)
        finally:
            if cf_data:
                CoreFoundation.CFRelease(cf_data)
            if sec_cert_ref:
                CoreFoundation.CFRelease(sec_cert_ref)

    return cf_array  # type: ignore[no-any-return]


@contextlib.contextmanager
def _configure_context(ctx: ssl.SSLContext) -> typing.Iterator[None]:
    check_hostname = ctx.check_hostname
    verify_mode = ctx.verify_mode
    ctx.check_hostname = False
    _set_ssl_context_verify_mode(ctx, ssl.CERT_NONE)
    try:
        yield
    finally:
        ctx.check_hostname = check_hostname
        _set_ssl_context_verify_mode(ctx, verify_mode)


def _verify_peercerts_impl(
    ssl_context: ssl.SSLContext,
    cert_chain: list[bytes],
    server_hostname: str | None = None,
) -> None:
    certs = None
    policies = None
    trust = None
    try:
        # Only set a hostname on the policy if we're verifying the hostname
        # on the leaf certificate.
        if server_hostname is not None and ssl_context.check_hostname:
            cf_str_hostname = None
            try:
                cf_str_hostname = _bytes_to_cf_string(server_hostname.encode("ascii"))
                ssl_policy = Security.SecPolicyCreateSSL(True, cf_str_hostname)
            finally:
                if cf_str_hostname:
                    CoreFoundation.CFRelease(cf_str_hostname)
        else:
            ssl_policy = Security.SecPolicyCreateSSL(True, None)

        policies = ssl_policy
        if ssl_context.verify_flags & ssl.VERIFY_CRL_CHECK_CHAIN:
            # Add explicit policy requiring positive revocation checks
            policies = CoreFoundation.CFArrayCreateMutable(
                CoreFoundation.kCFAllocatorDefault,
                0,
                ctypes.byref(CoreFoundation.kCFTypeArrayCallBacks),
            )
            CoreFoundation.CFArrayAppendValue(policies, ssl_policy)
            CoreFoundation.CFRelease(ssl_policy)
            revocation_policy = Security.SecPolicyCreateRevocation(
                kSecRevocationUseAnyAvailableMethod
                | kSecRevocationRequirePositiveResponse
            )
            CoreFoundation.CFArrayAppendValue(policies, revocation_policy)
            CoreFoundation.CFRelease(revocation_policy)
        elif ssl_context.verify_flags & ssl.VERIFY_CRL_CHECK_LEAF:
            raise NotImplementedError("VERIFY_CRL_CHECK_LEAF not implemented for macOS")

        certs = None
        try:
            certs = _der_certs_to_cf_cert_array(cert_chain)

            # Now that we have certificates loaded and a SecPolicy
            # we can finally create a SecTrust object!
            trust = Security.SecTrustRef()
            Security.SecTrustCreateWithCertificates(
                certs, policies, ctypes.byref(trust)
            )

        finally:
            # The certs are now being held by SecTrust so we can
            # release our handles for the array.
            if certs:
                CoreFoundation.CFRelease(certs)

        # If there are additional trust anchors to load we need to transform
        # the list of DER-encoded certificates into a CFArray.
        ctx_ca_certs_der: list[bytes] | None = ssl_context.get_ca_certs(
            binary_form=True
        )
        if ctx_ca_certs_der:
            ctx_ca_certs = None
            try:
                ctx_ca_certs = _der_certs_to_cf_cert_array(ctx_ca_certs_der)
                Security.SecTrustSetAnchorCertificates(trust, ctx_ca_certs)
            finally:
                if ctx_ca_certs:
                    CoreFoundation.CFRelease(ctx_ca_certs)

        # We always want system certificates.
        Security.SecTrustSetAnchorCertificatesOnly(trust, False)

        # macOS 10.13 and earlier don't support SecTrustEvaluateWithError()
        # so we use SecTrustEvaluate() which means we need to construct error
        # messages ourselves.
        if _is_macos_version_10_14_or_later:
            _verify_peercerts_impl_macos_10_14(ssl_context, trust)
        else:
            _verify_peercerts_impl_macos_10_13(ssl_context, trust)
    finally:
        if policies:
            CoreFoundation.CFRelease(policies)
        if trust:
            CoreFoundation.CFRelease(trust)


def _verify_peercerts_impl_macos_10_13(
    ssl_context: ssl.SSLContext, sec_trust_ref: typing.Any
) -> None:
    """Verify using 'SecTrustEvaluate' API for macOS 10.13 and earlier.
    macOS 10.14 added the 'SecTrustEvaluateWithError' API.
    """
    sec_trust_result_type = Security.SecTrustResultType()
    Security.SecTrustEvaluate(sec_trust_ref, ctypes.byref(sec_trust_result_type))

    try:
        sec_trust_result_type_as_int = int(sec_trust_result_type.value)
    except (ValueError, TypeError):
        sec_trust_result_type_as_int = -1

    # Apple doesn't document these values in their own API docs.
    # See: https://github.com/xybp888/iOS-SDKs/blob/master/iPhoneOS13.0.sdk/System/Library/Frameworks/Security.framework/Headers/SecTrust.h#L84
    if (
        ssl_context.verify_mode == ssl.CERT_REQUIRED
        and sec_trust_result_type_as_int not in (1, 4)
    ):
        # Note that we're not able to ignore only hostname errors
        # for macOS 10.13 and earlier, so check_hostname=False will
        # still return an error.
        sec_trust_result_type_to_message = {
            0: "Invalid trust result type",
            # 1: "Trust evaluation succeeded",
            2: "User confirmation required",
            3: "User specified that certificate is not trusted",
            # 4: "Trust result is unspecified",
            5: "Recoverable trust failure occurred",
            6: "Fatal trust failure occurred",
            7: "Other error occurred, certificate may be revoked",
        }
        error_message = sec_trust_result_type_to_message.get(
            sec_trust_result_type_as_int,
            f"Unknown trust result: {sec_trust_result_type_as_int}",
        )

        err = ssl.SSLCertVerificationError(error_message)
        err.verify_message = error_message
        err.verify_code = sec_trust_result_type_as_int
        raise err


def _verify_peercerts_impl_macos_10_14(
    ssl_context: ssl.SSLContext, sec_trust_ref: typing.Any
) -> None:
    """Verify using 'SecTrustEvaluateWithError' API for macOS 10.14+."""
    cf_error = CoreFoundation.CFErrorRef()
    sec_trust_eval_result = Security.SecTrustEvaluateWithError(
        sec_trust_ref, ctypes.byref(cf_error)
    )
    # sec_trust_eval_result is a bool (0 or 1)
    # where 1 means that the certs are trusted.
    if sec_trust_eval_result == 1:
        is_trusted = True
    elif sec_trust_eval_result == 0:
        is_trusted = False
    else:
        raise ssl.SSLError(
            f"Unknown result from Security.SecTrustEvaluateWithError: {sec_trust_eval_result!r}"
        )

    cf_error_code = 0
    if not is_trusted:
        cf_error_code = CoreFoundation.CFErrorGetCode(cf_error)

        # If the error is a known failure that we're
        # explicitly okay with from SSLContext configuration
        # we can set is_trusted accordingly.
        if ssl_context.verify_mode != ssl.CERT_REQUIRED and (
            cf_error_code == CFConst.errSecNotTrusted
            or cf_error_code == CFConst.errSecCertificateExpired
        ):
            is_trusted = True

    # If we're still not trusted then we start to
    # construct and raise the SSLCertVerificationError.
    if not is_trusted:
        cf_error_string_ref = None
        try:
            cf_error_string_ref = CoreFoundation.CFErrorCopyDescription(cf_error)

            # Can this ever return 'None' if there's a CFError?
            cf_error_message = (
                _cf_string_ref_to_str(cf_error_string_ref)
                or "Certificate verification failed"
            )

            # TODO: Not sure if we need the SecTrustResultType for anything?
            # We only care whether or not it's a success or failure for now.
            sec_trust_result_type = Security.SecTrustResultType()
            Security.SecTrustGetTrustResult(
                sec_trust_ref, ctypes.byref(sec_trust_result_type)
            )

            err = ssl.SSLCertVerificationError(cf_error_message)
            err.verify_message = cf_error_message
            err.verify_code = cf_error_code
            raise err
        finally:
            if cf_error_string_ref:
                CoreFoundation.CFRelease(cf_error_string_ref)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\mysql\reserved_words.py ===
# dialects/mysql/reserved_words.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

# generated using:
# https://gist.github.com/kkirsche/4f31f2153ed7a3248be1ec44ca6ddbc9
#
# https://mariadb.com/kb/en/reserved-words/
# includes: Reserved Words, Oracle Mode (separate set unioned)
# excludes: Exceptions, Function Names
# mypy: ignore-errors

RESERVED_WORDS_MARIADB = {
    "accessible",
    "add",
    "all",
    "alter",
    "analyze",
    "and",
    "as",
    "asc",
    "asensitive",
    "before",
    "between",
    "bigint",
    "binary",
    "blob",
    "both",
    "by",
    "call",
    "cascade",
    "case",
    "change",
    "char",
    "character",
    "check",
    "collate",
    "column",
    "condition",
    "constraint",
    "continue",
    "convert",
    "create",
    "cross",
    "current_date",
    "current_role",
    "current_time",
    "current_timestamp",
    "current_user",
    "cursor",
    "database",
    "databases",
    "day_hour",
    "day_microsecond",
    "day_minute",
    "day_second",
    "dec",
    "decimal",
    "declare",
    "default",
    "delayed",
    "delete",
    "desc",
    "describe",
    "deterministic",
    "distinct",
    "distinctrow",
    "div",
    "do_domain_ids",
    "double",
    "drop",
    "dual",
    "each",
    "else",
    "elseif",
    "enclosed",
    "escaped",
    "except",
    "exists",
    "exit",
    "explain",
    "false",
    "fetch",
    "float",
    "float4",
    "float8",
    "for",
    "force",
    "foreign",
    "from",
    "fulltext",
    "general",
    "grant",
    "group",
    "having",
    "high_priority",
    "hour_microsecond",
    "hour_minute",
    "hour_second",
    "if",
    "ignore",
    "ignore_domain_ids",
    "ignore_server_ids",
    "in",
    "index",
    "infile",
    "inner",
    "inout",
    "insensitive",
    "insert",
    "int",
    "int1",
    "int2",
    "int3",
    "int4",
    "int8",
    "integer",
    "intersect",
    "interval",
    "into",
    "is",
    "iterate",
    "join",
    "key",
    "keys",
    "kill",
    "leading",
    "leave",
    "left",
    "like",
    "limit",
    "linear",
    "lines",
    "load",
    "localtime",
    "localtimestamp",
    "lock",
    "long",
    "longblob",
    "longtext",
    "loop",
    "low_priority",
    "master_heartbeat_period",
    "master_ssl_verify_server_cert",
    "match",
    "maxvalue",
    "mediumblob",
    "mediumint",
    "mediumtext",
    "middleint",
    "minute_microsecond",
    "minute_second",
    "mod",
    "modifies",
    "natural",
    "no_write_to_binlog",
    "not",
    "null",
    "numeric",
    "offset",
    "on",
    "optimize",
    "option",
    "optionally",
    "or",
    "order",
    "out",
    "outer",
    "outfile",
    "over",
    "page_checksum",
    "parse_vcol_expr",
    "partition",
    "position",
    "precision",
    "primary",
    "procedure",
    "purge",
    "range",
    "read",
    "read_write",
    "reads",
    "real",
    "recursive",
    "ref_system_id",
    "references",
    "regexp",
    "release",
    "rename",
    "repeat",
    "replace",
    "require",
    "resignal",
    "restrict",
    "return",
    "returning",
    "revoke",
    "right",
    "rlike",
    "rows",
    "row_number",
    "schema",
    "schemas",
    "second_microsecond",
    "select",
    "sensitive",
    "separator",
    "set",
    "show",
    "signal",
    "slow",
    "smallint",
    "spatial",
    "specific",
    "sql",
    "sql_big_result",
    "sql_calc_found_rows",
    "sql_small_result",
    "sqlexception",
    "sqlstate",
    "sqlwarning",
    "ssl",
    "starting",
    "stats_auto_recalc",
    "stats_persistent",
    "stats_sample_pages",
    "straight_join",
    "table",
    "terminated",
    "then",
    "tinyblob",
    "tinyint",
    "tinytext",
    "to",
    "trailing",
    "trigger",
    "true",
    "undo",
    "union",
    "unique",
    "unlock",
    "unsigned",
    "update",
    "usage",
    "use",
    "using",
    "utc_date",
    "utc_time",
    "utc_timestamp",
    "values",
    "varbinary",
    "varchar",
    "varcharacter",
    "varying",
    "when",
    "where",
    "while",
    "window",
    "with",
    "write",
    "xor",
    "year_month",
    "zerofill",
}.union(
    {
        "body",
        "elsif",
        "goto",
        "history",
        "others",
        "package",
        "period",
        "raise",
        "rowtype",
        "system",
        "system_time",
        "versioning",
        "without",
    }
)

# https://dev.mysql.com/doc/refman/8.3/en/keywords.html
# https://dev.mysql.com/doc/refman/8.0/en/keywords.html
# https://dev.mysql.com/doc/refman/5.7/en/keywords.html
# https://dev.mysql.com/doc/refman/5.6/en/keywords.html
# includes: MySQL x.0 Keywords and Reserved Words
# excludes: MySQL x.0 New Keywords and Reserved Words,
#       MySQL x.0 Removed Keywords and Reserved Words
RESERVED_WORDS_MYSQL = {
    "accessible",
    "add",
    "admin",
    "all",
    "alter",
    "analyze",
    "and",
    "array",
    "as",
    "asc",
    "asensitive",
    "before",
    "between",
    "bigint",
    "binary",
    "blob",
    "both",
    "by",
    "call",
    "cascade",
    "case",
    "change",
    "char",
    "character",
    "check",
    "collate",
    "column",
    "condition",
    "constraint",
    "continue",
    "convert",
    "create",
    "cross",
    "cube",
    "cume_dist",
    "current_date",
    "current_time",
    "current_timestamp",
    "current_user",
    "cursor",
    "database",
    "databases",
    "day_hour",
    "day_microsecond",
    "day_minute",
    "day_second",
    "dec",
    "decimal",
    "declare",
    "default",
    "delayed",
    "delete",
    "dense_rank",
    "desc",
    "describe",
    "deterministic",
    "distinct",
    "distinctrow",
    "div",
    "double",
    "drop",
    "dual",
    "each",
    "else",
    "elseif",
    "empty",
    "enclosed",
    "escaped",
    "except",
    "exists",
    "exit",
    "explain",
    "false",
    "fetch",
    "first_value",
    "float",
    "float4",
    "float8",
    "for",
    "force",
    "foreign",
    "from",
    "fulltext",
    "function",
    "general",
    "generated",
    "get",
    "get_master_public_key",
    "grant",
    "group",
    "grouping",
    "groups",
    "having",
    "high_priority",
    "hour_microsecond",
    "hour_minute",
    "hour_second",
    "if",
    "ignore",
    "ignore_server_ids",
    "in",
    "index",
    "infile",
    "inner",
    "inout",
    "insensitive",
    "insert",
    "int",
    "int1",
    "int2",
    "int3",
    "int4",
    "int8",
    "integer",
    "intersect",
    "interval",
    "into",
    "io_after_gtids",
    "io_before_gtids",
    "is",
    "iterate",
    "join",
    "json_table",
    "key",
    "keys",
    "kill",
    "lag",
    "last_value",
    "lateral",
    "lead",
    "leading",
    "leave",
    "left",
    "like",
    "limit",
    "linear",
    "lines",
    "load",
    "localtime",
    "localtimestamp",
    "lock",
    "long",
    "longblob",
    "longtext",
    "loop",
    "low_priority",
    "master_bind",
    "master_heartbeat_period",
    "master_ssl_verify_server_cert",
    "match",
    "maxvalue",
    "mediumblob",
    "mediumint",
    "mediumtext",
    "member",
    "middleint",
    "minute_microsecond",
    "minute_second",
    "mod",
    "modifies",
    "natural",
    "no_write_to_binlog",
    "not",
    "nth_value",
    "ntile",
    "null",
    "numeric",
    "of",
    "on",
    "optimize",
    "optimizer_costs",
    "option",
    "optionally",
    "or",
    "order",
    "out",
    "outer",
    "outfile",
    "over",
    "parse_gcol_expr",
    "parallel",
    "partition",
    "percent_rank",
    "persist",
    "persist_only",
    "precision",
    "primary",
    "procedure",
    "purge",
    "qualify",
    "range",
    "rank",
    "read",
    "read_write",
    "reads",
    "real",
    "recursive",
    "references",
    "regexp",
    "release",
    "rename",
    "repeat",
    "replace",
    "require",
    "resignal",
    "restrict",
    "return",
    "revoke",
    "right",
    "rlike",
    "role",
    "row",
    "row_number",
    "rows",
    "schema",
    "schemas",
    "second_microsecond",
    "select",
    "sensitive",
    "separator",
    "set",
    "show",
    "signal",
    "slow",
    "smallint",
    "spatial",
    "specific",
    "sql",
    "sql_after_gtids",
    "sql_before_gtids",
    "sql_big_result",
    "sql_calc_found_rows",
    "sql_small_result",
    "sqlexception",
    "sqlstate",
    "sqlwarning",
    "ssl",
    "starting",
    "stored",
    "straight_join",
    "system",
    "table",
    "terminated",
    "then",
    "tinyblob",
    "tinyint",
    "tinytext",
    "to",
    "trailing",
    "trigger",
    "true",
    "undo",
    "union",
    "unique",
    "unlock",
    "unsigned",
    "update",
    "usage",
    "use",
    "using",
    "utc_date",
    "utc_time",
    "utc_timestamp",
    "values",
    "varbinary",
    "varchar",
    "varcharacter",
    "varying",
    "virtual",
    "when",
    "where",
    "while",
    "window",
    "with",
    "write",
    "xor",
    "year_month",
    "zerofill",
}