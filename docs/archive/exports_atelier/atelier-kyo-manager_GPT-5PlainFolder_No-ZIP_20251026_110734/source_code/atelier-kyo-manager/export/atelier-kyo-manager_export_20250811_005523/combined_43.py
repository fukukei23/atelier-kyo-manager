
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\console.py ===
import inspect
import os
import sys
import threading
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from getpass import getpass
from html import escape
from inspect import isclass
from itertools import islice
from math import ceil
from time import monotonic
from types import FrameType, ModuleType, TracebackType
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    TextIO,
    Tuple,
    Type,
    Union,
    cast,
)

from pip._vendor.rich._null_file import NULL_FILE

if sys.version_info >= (3, 8):
    from typing import Literal, Protocol, runtime_checkable
else:
    from pip._vendor.typing_extensions import (
        Literal,
        Protocol,
        runtime_checkable,
    )  # pragma: no cover

from . import errors, themes
from ._emoji_replace import _emoji_replace
from ._export_format import CONSOLE_HTML_FORMAT, CONSOLE_SVG_FORMAT
from ._fileno import get_fileno
from ._log_render import FormatTimeCallable, LogRender
from .align import Align, AlignMethod
from .color import ColorSystem, blend_rgb
from .control import Control
from .emoji import EmojiVariant
from .highlighter import NullHighlighter, ReprHighlighter
from .markup import render as render_markup
from .measure import Measurement, measure_renderables
from .pager import Pager, SystemPager
from .pretty import Pretty, is_expandable
from .protocol import rich_cast
from .region import Region
from .scope import render_scope
from .screen import Screen
from .segment import Segment
from .style import Style, StyleType
from .styled import Styled
from .terminal_theme import DEFAULT_TERMINAL_THEME, SVG_EXPORT_THEME, TerminalTheme
from .text import Text, TextType
from .theme import Theme, ThemeStack

if TYPE_CHECKING:
    from ._windows import WindowsConsoleFeatures
    from .live import Live
    from .status import Status

JUPYTER_DEFAULT_COLUMNS = 115
JUPYTER_DEFAULT_LINES = 100
WINDOWS = sys.platform == "win32"

HighlighterType = Callable[[Union[str, "Text"]], "Text"]
JustifyMethod = Literal["default", "left", "center", "right", "full"]
OverflowMethod = Literal["fold", "crop", "ellipsis", "ignore"]


class NoChange:
    pass


NO_CHANGE = NoChange()

try:
    _STDIN_FILENO = sys.__stdin__.fileno()  # type: ignore[union-attr]
except Exception:
    _STDIN_FILENO = 0
try:
    _STDOUT_FILENO = sys.__stdout__.fileno()  # type: ignore[union-attr]
except Exception:
    _STDOUT_FILENO = 1
try:
    _STDERR_FILENO = sys.__stderr__.fileno()  # type: ignore[union-attr]
except Exception:
    _STDERR_FILENO = 2

_STD_STREAMS = (_STDIN_FILENO, _STDOUT_FILENO, _STDERR_FILENO)
_STD_STREAMS_OUTPUT = (_STDOUT_FILENO, _STDERR_FILENO)


_TERM_COLORS = {
    "kitty": ColorSystem.EIGHT_BIT,
    "256color": ColorSystem.EIGHT_BIT,
    "16color": ColorSystem.STANDARD,
}


class ConsoleDimensions(NamedTuple):
    """Size of the terminal."""

    width: int
    """The width of the console in 'cells'."""
    height: int
    """The height of the console in lines."""


@dataclass
class ConsoleOptions:
    """Options for __rich_console__ method."""

    size: ConsoleDimensions
    """Size of console."""
    legacy_windows: bool
    """legacy_windows: flag for legacy windows."""
    min_width: int
    """Minimum width of renderable."""
    max_width: int
    """Maximum width of renderable."""
    is_terminal: bool
    """True if the target is a terminal, otherwise False."""
    encoding: str
    """Encoding of terminal."""
    max_height: int
    """Height of container (starts as terminal)"""
    justify: Optional[JustifyMethod] = None
    """Justify value override for renderable."""
    overflow: Optional[OverflowMethod] = None
    """Overflow value override for renderable."""
    no_wrap: Optional[bool] = False
    """Disable wrapping for text."""
    highlight: Optional[bool] = None
    """Highlight override for render_str."""
    markup: Optional[bool] = None
    """Enable markup when rendering strings."""
    height: Optional[int] = None

    @property
    def ascii_only(self) -> bool:
        """Check if renderables should use ascii only."""
        return not self.encoding.startswith("utf")

    def copy(self) -> "ConsoleOptions":
        """Return a copy of the options.

        Returns:
            ConsoleOptions: a copy of self.
        """
        options: ConsoleOptions = ConsoleOptions.__new__(ConsoleOptions)
        options.__dict__ = self.__dict__.copy()
        return options

    def update(
        self,
        *,
        width: Union[int, NoChange] = NO_CHANGE,
        min_width: Union[int, NoChange] = NO_CHANGE,
        max_width: Union[int, NoChange] = NO_CHANGE,
        justify: Union[Optional[JustifyMethod], NoChange] = NO_CHANGE,
        overflow: Union[Optional[OverflowMethod], NoChange] = NO_CHANGE,
        no_wrap: Union[Optional[bool], NoChange] = NO_CHANGE,
        highlight: Union[Optional[bool], NoChange] = NO_CHANGE,
        markup: Union[Optional[bool], NoChange] = NO_CHANGE,
        height: Union[Optional[int], NoChange] = NO_CHANGE,
    ) -> "ConsoleOptions":
        """Update values, return a copy."""
        options = self.copy()
        if not isinstance(width, NoChange):
            options.min_width = options.max_width = max(0, width)
        if not isinstance(min_width, NoChange):
            options.min_width = min_width
        if not isinstance(max_width, NoChange):
            options.max_width = max_width
        if not isinstance(justify, NoChange):
            options.justify = justify
        if not isinstance(overflow, NoChange):
            options.overflow = overflow
        if not isinstance(no_wrap, NoChange):
            options.no_wrap = no_wrap
        if not isinstance(highlight, NoChange):
            options.highlight = highlight
        if not isinstance(markup, NoChange):
            options.markup = markup
        if not isinstance(height, NoChange):
            if height is not None:
                options.max_height = height
            options.height = None if height is None else max(0, height)
        return options

    def update_width(self, width: int) -> "ConsoleOptions":
        """Update just the width, return a copy.

        Args:
            width (int): New width (sets both min_width and max_width)

        Returns:
            ~ConsoleOptions: New console options instance.
        """
        options = self.copy()
        options.min_width = options.max_width = max(0, width)
        return options

    def update_height(self, height: int) -> "ConsoleOptions":
        """Update the height, and return a copy.

        Args:
            height (int): New height

        Returns:
            ~ConsoleOptions: New Console options instance.
        """
        options = self.copy()
        options.max_height = options.height = height
        return options

    def reset_height(self) -> "ConsoleOptions":
        """Return a copy of the options with height set to ``None``.

        Returns:
            ~ConsoleOptions: New console options instance.
        """
        options = self.copy()
        options.height = None
        return options

    def update_dimensions(self, width: int, height: int) -> "ConsoleOptions":
        """Update the width and height, and return a copy.

        Args:
            width (int): New width (sets both min_width and max_width).
            height (int): New height.

        Returns:
            ~ConsoleOptions: New console options instance.
        """
        options = self.copy()
        options.min_width = options.max_width = max(0, width)
        options.height = options.max_height = height
        return options


@runtime_checkable
class RichCast(Protocol):
    """An object that may be 'cast' to a console renderable."""

    def __rich__(
        self,
    ) -> Union["ConsoleRenderable", "RichCast", str]:  # pragma: no cover
        ...


@runtime_checkable
class ConsoleRenderable(Protocol):
    """An object that supports the console protocol."""

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":  # pragma: no cover
        ...


# A type that may be rendered by Console.
RenderableType = Union[ConsoleRenderable, RichCast, str]
"""A string or any object that may be rendered by Rich."""

# The result of calling a __rich_console__ method.
RenderResult = Iterable[Union[RenderableType, Segment]]

_null_highlighter = NullHighlighter()


class CaptureError(Exception):
    """An error in the Capture context manager."""


class NewLine:
    """A renderable to generate new line(s)"""

    def __init__(self, count: int = 1) -> None:
        self.count = count

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> Iterable[Segment]:
        yield Segment("\n" * self.count)


class ScreenUpdate:
    """Render a list of lines at a given offset."""

    def __init__(self, lines: List[List[Segment]], x: int, y: int) -> None:
        self._lines = lines
        self.x = x
        self.y = y

    def __rich_console__(
        self, console: "Console", options: ConsoleOptions
    ) -> RenderResult:
        x = self.x
        move_to = Control.move_to
        for offset, line in enumerate(self._lines, self.y):
            yield move_to(x, offset)
            yield from line


class Capture:
    """Context manager to capture the result of printing to the console.
    See :meth:`~rich.console.Console.capture` for how to use.

    Args:
        console (Console): A console instance to capture output.
    """

    def __init__(self, console: "Console") -> None:
        self._console = console
        self._result: Optional[str] = None

    def __enter__(self) -> "Capture":
        self._console.begin_capture()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._result = self._console.end_capture()

    def get(self) -> str:
        """Get the result of the capture."""
        if self._result is None:
            raise CaptureError(
                "Capture result is not available until context manager exits."
            )
        return self._result


class ThemeContext:
    """A context manager to use a temporary theme. See :meth:`~rich.console.Console.use_theme` for usage."""

    def __init__(self, console: "Console", theme: Theme, inherit: bool = True) -> None:
        self.console = console
        self.theme = theme
        self.inherit = inherit

    def __enter__(self) -> "ThemeContext":
        self.console.push_theme(self.theme)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.console.pop_theme()


class PagerContext:
    """A context manager that 'pages' content. See :meth:`~rich.console.Console.pager` for usage."""

    def __init__(
        self,
        console: "Console",
        pager: Optional[Pager] = None,
        styles: bool = False,
        links: bool = False,
    ) -> None:
        self._console = console
        self.pager = SystemPager() if pager is None else pager
        self.styles = styles
        self.links = links

    def __enter__(self) -> "PagerContext":
        self._console._enter_buffer()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if exc_type is None:
            with self._console._lock:
                buffer: List[Segment] = self._console._buffer[:]
                del self._console._buffer[:]
                segments: Iterable[Segment] = buffer
                if not self.styles:
                    segments = Segment.strip_styles(segments)
                elif not self.links:
                    segments = Segment.strip_links(segments)
                content = self._console._render_buffer(segments)
            self.pager.show(content)
        self._console._exit_buffer()


class ScreenContext:
    """A context manager that enables an alternative screen. See :meth:`~rich.console.Console.screen` for usage."""

    def __init__(
        self, console: "Console", hide_cursor: bool, style: StyleType = ""
    ) -> None:
        self.console = console
        self.hide_cursor = hide_cursor
        self.screen = Screen(style=style)
        self._changed = False

    def update(
        self, *renderables: RenderableType, style: Optional[StyleType] = None
    ) -> None:
        """Update the screen.

        Args:
            renderable (RenderableType, optional): Optional renderable to replace current renderable,
                or None for no change. Defaults to None.
            style: (Style, optional): Replacement style, or None for no change. Defaults to None.
        """
        if renderables:
            self.screen.renderable = (
                Group(*renderables) if len(renderables) > 1 else renderables[0]
            )
        if style is not None:
            self.screen.style = style
        self.console.print(self.screen, end="")

    def __enter__(self) -> "ScreenContext":
        self._changed = self.console.set_alt_screen(True)
        if self._changed and self.hide_cursor:
            self.console.show_cursor(False)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._changed:
            self.console.set_alt_screen(False)
            if self.hide_cursor:
                self.console.show_cursor(True)


class Group:
    """Takes a group of renderables and returns a renderable object that renders the group.

    Args:
        renderables (Iterable[RenderableType]): An iterable of renderable objects.
        fit (bool, optional): Fit dimension of group to contents, or fill available space. Defaults to True.
    """

    def __init__(self, *renderables: "RenderableType", fit: bool = True) -> None:
        self._renderables = renderables
        self.fit = fit
        self._render: Optional[List[RenderableType]] = None

    @property
    def renderables(self) -> List["RenderableType"]:
        if self._render is None:
            self._render = list(self._renderables)
        return self._render

    def __rich_measure__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "Measurement":
        if self.fit:
            return measure_renderables(console, options, self.renderables)
        else:
            return Measurement(options.max_width, options.max_width)

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> RenderResult:
        yield from self.renderables


def group(fit: bool = True) -> Callable[..., Callable[..., Group]]:
    """A decorator that turns an iterable of renderables in to a group.

    Args:
        fit (bool, optional): Fit dimension of group to contents, or fill available space. Defaults to True.
    """

    def decorator(
        method: Callable[..., Iterable[RenderableType]],
    ) -> Callable[..., Group]:
        """Convert a method that returns an iterable of renderables in to a Group."""

        @wraps(method)
        def _replace(*args: Any, **kwargs: Any) -> Group:
            renderables = method(*args, **kwargs)
            return Group(*renderables, fit=fit)

        return _replace

    return decorator


def _is_jupyter() -> bool:  # pragma: no cover
    """Check if we're running in a Jupyter notebook."""
    try:
        get_ipython  # type: ignore[name-defined]
    except NameError:
        return False
    ipython = get_ipython()  # type: ignore[name-defined]
    shell = ipython.__class__.__name__
    if (
        "google.colab" in str(ipython.__class__)
        or os.getenv("DATABRICKS_RUNTIME_VERSION")
        or shell == "ZMQInteractiveShell"
    ):
        return True  # Jupyter notebook or qtconsole
    elif shell == "TerminalInteractiveShell":
        return False  # Terminal running IPython
    else:
        return False  # Other type (?)


COLOR_SYSTEMS = {
    "standard": ColorSystem.STANDARD,
    "256": ColorSystem.EIGHT_BIT,
    "truecolor": ColorSystem.TRUECOLOR,
    "windows": ColorSystem.WINDOWS,
}

_COLOR_SYSTEMS_NAMES = {system: name for name, system in COLOR_SYSTEMS.items()}


@dataclass
class ConsoleThreadLocals(threading.local):
    """Thread local values for Console context."""

    theme_stack: ThemeStack
    buffer: List[Segment] = field(default_factory=list)
    buffer_index: int = 0


class RenderHook(ABC):
    """Provides hooks in to the render process."""

    @abstractmethod
    def process_renderables(
        self, renderables: List[ConsoleRenderable]
    ) -> List[ConsoleRenderable]:
        """Called with a list of objects to render.

        This method can return a new list of renderables, or modify and return the same list.

        Args:
            renderables (List[ConsoleRenderable]): A number of renderable objects.

        Returns:
            List[ConsoleRenderable]: A replacement list of renderables.
        """


_windows_console_features: Optional["WindowsConsoleFeatures"] = None


def get_windows_console_features() -> "WindowsConsoleFeatures":  # pragma: no cover
    global _windows_console_features
    if _windows_console_features is not None:
        return _windows_console_features
    from ._windows import get_windows_console_features

    _windows_console_features = get_windows_console_features()
    return _windows_console_features


def detect_legacy_windows() -> bool:
    """Detect legacy Windows."""
    return WINDOWS and not get_windows_console_features().vt


class Console:
    """A high level console interface.

    Args:
        color_system (str, optional): The color system supported by your terminal,
            either ``"standard"``, ``"256"`` or ``"truecolor"``. Leave as ``"auto"`` to autodetect.
        force_terminal (Optional[bool], optional): Enable/disable terminal control codes, or None to auto-detect terminal. Defaults to None.
        force_jupyter (Optional[bool], optional): Enable/disable Jupyter rendering, or None to auto-detect Jupyter. Defaults to None.
        force_interactive (Optional[bool], optional): Enable/disable interactive mode, or None to auto detect. Defaults to None.
        soft_wrap (Optional[bool], optional): Set soft wrap default on print method. Defaults to False.
        theme (Theme, optional): An optional style theme object, or ``None`` for default theme.
        stderr (bool, optional): Use stderr rather than stdout if ``file`` is not specified. Defaults to False.
        file (IO, optional): A file object where the console should write to. Defaults to stdout.
        quiet (bool, Optional): Boolean to suppress all output. Defaults to False.
        width (int, optional): The width of the terminal. Leave as default to auto-detect width.
        height (int, optional): The height of the terminal. Leave as default to auto-detect height.
        style (StyleType, optional): Style to apply to all output, or None for no style. Defaults to None.
        no_color (Optional[bool], optional): Enabled no color mode, or None to auto detect. Defaults to None.
        tab_size (int, optional): Number of spaces used to replace a tab character. Defaults to 8.
        record (bool, optional): Boolean to enable recording of terminal output,
            required to call :meth:`export_html`, :meth:`export_svg`, and :meth:`export_text`. Defaults to False.
        markup (bool, optional): Boolean to enable :ref:`console_markup`. Defaults to True.
        emoji (bool, optional): Enable emoji code. Defaults to True.
        emoji_variant (str, optional): Optional emoji variant, either "text" or "emoji". Defaults to None.
        highlight (bool, optional): Enable automatic highlighting. Defaults to True.
        log_time (bool, optional): Boolean to enable logging of time by :meth:`log` methods. Defaults to True.
        log_path (bool, optional): Boolean to enable the logging of the caller by :meth:`log`. Defaults to True.
        log_time_format (Union[str, TimeFormatterCallable], optional): If ``log_time`` is enabled, either string for strftime or callable that formats the time. Defaults to "[%X] ".
        highlighter (HighlighterType, optional): Default highlighter.
        legacy_windows (bool, optional): Enable legacy Windows mode, or ``None`` to auto detect. Defaults to ``None``.
        safe_box (bool, optional): Restrict box options that don't render on legacy Windows.
        get_datetime (Callable[[], datetime], optional): Callable that gets the current time as a datetime.datetime object (used by Console.log),
            or None for datetime.now.
        get_time (Callable[[], time], optional): Callable that gets the current time in seconds, default uses time.monotonic.
    """

    _environ: Mapping[str, str] = os.environ

    def __init__(
        self,
        *,
        color_system: Optional[
            Literal["auto", "standard", "256", "truecolor", "windows"]
        ] = "auto",
        force_terminal: Optional[bool] = None,
        force_jupyter: Optional[bool] = None,
        force_interactive: Optional[bool] = None,
        soft_wrap: bool = False,
        theme: Optional[Theme] = None,
        stderr: bool = False,
        file: Optional[IO[str]] = None,
        quiet: bool = False,
        width: Optional[int] = None,
        height: Optional[int] = None,
        style: Optional[StyleType] = None,
        no_color: Optional[bool] = None,
        tab_size: int = 8,
        record: bool = False,
        markup: bool = True,
        emoji: bool = True,
        emoji_variant: Optional[EmojiVariant] = None,
        highlight: bool = True,
        log_time: bool = True,
        log_path: bool = True,
        log_time_format: Union[str, FormatTimeCallable] = "[%X]",
        highlighter: Optional["HighlighterType"] = ReprHighlighter(),
        legacy_windows: Optional[bool] = None,
        safe_box: bool = True,
        get_datetime: Optional[Callable[[], datetime]] = None,
        get_time: Optional[Callable[[], float]] = None,
        _environ: Optional[Mapping[str, str]] = None,
    ):
        # Copy of os.environ allows us to replace it for testing
        if _environ is not None:
            self._environ = _environ

        self.is_jupyter = _is_jupyter() if force_jupyter is None else force_jupyter
        if self.is_jupyter:
            if width is None:
                jupyter_columns = self._environ.get("JUPYTER_COLUMNS")
                if jupyter_columns is not None and jupyter_columns.isdigit():
                    width = int(jupyter_columns)
                else:
                    width = JUPYTER_DEFAULT_COLUMNS
            if height is None:
                jupyter_lines = self._environ.get("JUPYTER_LINES")
                if jupyter_lines is not None and jupyter_lines.isdigit():
                    height = int(jupyter_lines)
                else:
                    height = JUPYTER_DEFAULT_LINES

        self.tab_size = tab_size
        self.record = record
        self._markup = markup
        self._emoji = emoji
        self._emoji_variant: Optional[EmojiVariant] = emoji_variant
        self._highlight = highlight
        self.legacy_windows: bool = (
            (detect_legacy_windows() and not self.is_jupyter)
            if legacy_windows is None
            else legacy_windows
        )

        if width is None:
            columns = self._environ.get("COLUMNS")
            if columns is not None and columns.isdigit():
                width = int(columns) - self.legacy_windows
        if height is None:
            lines = self._environ.get("LINES")
            if lines is not None and lines.isdigit():
                height = int(lines)

        self.soft_wrap = soft_wrap
        self._width = width
        self._height = height

        self._color_system: Optional[ColorSystem]

        self._force_terminal = None
        if force_terminal is not None:
            self._force_terminal = force_terminal

        self._file = file
        self.quiet = quiet
        self.stderr = stderr

        if color_system is None:
            self._color_system = None
        elif color_system == "auto":
            self._color_system = self._detect_color_system()
        else:
            self._color_system = COLOR_SYSTEMS[color_system]

        self._lock = threading.RLock()
        self._log_render = LogRender(
            show_time=log_time,
            show_path=log_path,
            time_format=log_time_format,
        )
        self.highlighter: HighlighterType = highlighter or _null_highlighter
        self.safe_box = safe_box
        self.get_datetime = get_datetime or datetime.now
        self.get_time = get_time or monotonic
        self.style = style
        self.no_color = (
            no_color
            if no_color is not None
            else self._environ.get("NO_COLOR", "") != ""
        )
        self.is_interactive = (
            (self.is_terminal and not self.is_dumb_terminal)
            if force_interactive is None
            else force_interactive
        )

        self._record_buffer_lock = threading.RLock()
        self._thread_locals = ConsoleThreadLocals(
            theme_stack=ThemeStack(themes.DEFAULT if theme is None else theme)
        )
        self._record_buffer: List[Segment] = []
        self._render_hooks: List[RenderHook] = []
        self._live: Optional["Live"] = None
        self._is_alt_screen = False

    def __repr__(self) -> str:
        return f"<console width={self.width} {self._color_system!s}>"

    @property
    def file(self) -> IO[str]:
        """Get the file object to write to."""
        file = self._file or (sys.stderr if self.stderr else sys.stdout)
        file = getattr(file, "rich_proxied_file", file)
        if file is None:
            file = NULL_FILE
        return file

    @file.setter
    def file(self, new_file: IO[str]) -> None:
        """Set a new file object."""
        self._file = new_file

    @property
    def _buffer(self) -> List[Segment]:
        """Get a thread local buffer."""
        return self._thread_locals.buffer

    @property
    def _buffer_index(self) -> int:
        """Get a thread local buffer."""
        return self._thread_locals.buffer_index

    @_buffer_index.setter
    def _buffer_index(self, value: int) -> None:
        self._thread_locals.buffer_index = value

    @property
    def _theme_stack(self) -> ThemeStack:
        """Get the thread local theme stack."""
        return self._thread_locals.theme_stack

    def _detect_color_system(self) -> Optional[ColorSystem]:
        """Detect color system from env vars."""
        if self.is_jupyter:
            return ColorSystem.TRUECOLOR
        if not self.is_terminal or self.is_dumb_terminal:
            return None
        if WINDOWS:  # pragma: no cover
            if self.legacy_windows:  # pragma: no cover
                return ColorSystem.WINDOWS
            windows_console_features = get_windows_console_features()
            return (
                ColorSystem.TRUECOLOR
                if windows_console_features.truecolor
                else ColorSystem.EIGHT_BIT
            )
        else:
            color_term = self._environ.get("COLORTERM", "").strip().lower()
            if color_term in ("truecolor", "24bit"):
                return ColorSystem.TRUECOLOR
            term = self._environ.get("TERM", "").strip().lower()
            _term_name, _hyphen, colors = term.rpartition("-")
            color_system = _TERM_COLORS.get(colors, ColorSystem.STANDARD)
            return color_system

    def _enter_buffer(self) -> None:
        """Enter in to a buffer context, and buffer all output."""
        self._buffer_index += 1

    def _exit_buffer(self) -> None:
        """Leave buffer context, and render content if required."""
        self._buffer_index -= 1
        self._check_buffer()

    def set_live(self, live: "Live") -> None:
        """Set Live instance. Used by Live context manager.

        Args:
            live (Live): Live instance using this Console.

        Raises:
            errors.LiveError: If this Console has a Live context currently active.
        """
        with self._lock:
            if self._live is not None:
                raise errors.LiveError("Only one live display may be active at once")
            self._live = live

    def clear_live(self) -> None:
        """Clear the Live instance."""
        with self._lock:
            self._live = None

    def push_render_hook(self, hook: RenderHook) -> None:
        """Add a new render hook to the stack.

        Args:
            hook (RenderHook): Render hook instance.
        """
        with self._lock:
            self._render_hooks.append(hook)

    def pop_render_hook(self) -> None:
        """Pop the last renderhook from the stack."""
        with self._lock:
            self._render_hooks.pop()

    def __enter__(self) -> "Console":
        """Own context manager to enter buffer context."""
        self._enter_buffer()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit buffer context."""
        self._exit_buffer()

    def begin_capture(self) -> None:
        """Begin capturing console output. Call :meth:`end_capture` to exit capture mode and return output."""
        self._enter_buffer()

    def end_capture(self) -> str:
        """End capture mode and return captured string.

        Returns:
            str: Console output.
        """
        render_result = self._render_buffer(self._buffer)
        del self._buffer[:]
        self._exit_buffer()
        return render_result

    def push_theme(self, theme: Theme, *, inherit: bool = True) -> None:
        """Push a new theme on to the top of the stack, replacing the styles from the previous theme.
        Generally speaking, you should call :meth:`~rich.console.Console.use_theme` to get a context manager, rather
        than calling this method directly.

        Args:
            theme (Theme): A theme instance.
            inherit (bool, optional): Inherit existing styles. Defaults to True.
        """
        self._theme_stack.push_theme(theme, inherit=inherit)

    def pop_theme(self) -> None:
        """Remove theme from top of stack, restoring previous theme."""
        self._theme_stack.pop_theme()

    def use_theme(self, theme: Theme, *, inherit: bool = True) -> ThemeContext:
        """Use a different theme for the duration of the context manager.

        Args:
            theme (Theme): Theme instance to user.
            inherit (bool, optional): Inherit existing console styles. Defaults to True.

        Returns:
            ThemeContext: [description]
        """
        return ThemeContext(self, theme, inherit)

    @property
    def color_system(self) -> Optional[str]:
        """Get color system string.

        Returns:
            Optional[str]: "standard", "256" or "truecolor".
        """

        if self._color_system is not None:
            return _COLOR_SYSTEMS_NAMES[self._color_system]
        else:
            return None

    @property
    def encoding(self) -> str:
        """Get the encoding of the console file, e.g. ``"utf-8"``.

        Returns:
            str: A standard encoding string.
        """
        return (getattr(self.file, "encoding", "utf-8") or "utf-8").lower()

    @property
    def is_terminal(self) -> bool:
        """Check if the console is writing to a terminal.

        Returns:
            bool: True if the console writing to a device capable of
                understanding escape sequences, otherwise False.
        """
        # If dev has explicitly set this value, return it
        if self._force_terminal is not None:
            return self._force_terminal

        # Fudge for Idle
        if hasattr(sys.stdin, "__module__") and sys.stdin.__module__.startswith(
            "idlelib"
        ):
            # Return False for Idle which claims to be a tty but can't handle ansi codes
            return False

        if self.is_jupyter:
            # return False for Jupyter, which may have FORCE_COLOR set
            return False

        environ = self._environ

        tty_compatible = environ.get("TTY_COMPATIBLE", "")
        # 0 indicates device is not tty compatible
        if tty_compatible == "0":
            return False
        # 1 indicates device is tty compatible
        if tty_compatible == "1":
            return True

        # https://force-color.org/
        force_color = environ.get("FORCE_COLOR")
        if force_color is not None:
            return force_color != ""

        # Any other value defaults to auto detect
        isatty: Optional[Callable[[], bool]] = getattr(self.file, "isatty", None)
        try:
            return False if isatty is None else isatty()
        except ValueError:
            # in some situation (at the end of a pytest run for example) isatty() can raise
            # ValueError: I/O operation on closed file
            # return False because we aren't in a terminal anymore
            return False

    @property
    def is_dumb_terminal(self) -> bool:
        """Detect dumb terminal.

        Returns:
            bool: True if writing to a dumb terminal, otherwise False.

        """
        _term = self._environ.get("TERM", "")
        is_dumb = _term.lower() in ("dumb", "unknown")
        return self.is_terminal and is_dumb

    @property
    def options(self) -> ConsoleOptions:
        """Get default console options."""
        return ConsoleOptions(
            max_height=self.size.height,
            size=self.size,
            legacy_windows=self.legacy_windows,
            min_width=1,
            max_width=self.width,
            encoding=self.encoding,
            is_terminal=self.is_terminal,
        )

    @property
    def size(self) -> ConsoleDimensions:
        """Get the size of the console.

        Returns:
            ConsoleDimensions: A named tuple containing the dimensions.
        """

        if self._width is not None and self._height is not None:
            return ConsoleDimensions(self._width - self.legacy_windows, self._height)

        if self.is_dumb_terminal:
            return ConsoleDimensions(80, 25)

        width: Optional[int] = None
        height: Optional[int] = None

        streams = _STD_STREAMS_OUTPUT if WINDOWS else _STD_STREAMS
        for file_descriptor in streams:
            try:
                width, height = os.get_terminal_size(file_descriptor)
            except (AttributeError, ValueError, OSError):  # Probably not a terminal
                pass
            else:
                break

        columns = self._environ.get("COLUMNS")
        if columns is not None and columns.isdigit():
            width = int(columns)
        lines = self._environ.get("LINES")
        if lines is not None and lines.isdigit():
            height = int(lines)

        # get_terminal_size can report 0, 0 if run from pseudo-terminal
        width = width or 80
        height = height or 25
        return ConsoleDimensions(
            width - self.legacy_windows if self._width is None else self._width,
            height if self._height is None else self._height,
        )

    @size.setter
    def size(self, new_size: Tuple[int, int]) -> None:
        """Set a new size for the terminal.

        Args:
            new_size (Tuple[int, int]): New width and height.
        """
        width, height = new_size
        self._width = width
        self._height = height

    @property
    def width(self) -> int:
        """Get the width of the console.

        Returns:
            int: The width (in characters) of the console.
        """
        return self.size.width

    @width.setter
    def width(self, width: int) -> None:
        """Set width.

        Args:
            width (int): New width.
        """
        self._width = width

    @property
    def height(self) -> int:
        """Get the height of the console.

        Returns:
            int: The height (in lines) of the console.
        """
        return self.size.height

    @height.setter
    def height(self, height: int) -> None:
        """Set height.

        Args:
            height (int): new height.
        """
        self._height = height

    def bell(self) -> None:
        """Play a 'bell' sound (if supported by the terminal)."""
        self.control(Control.bell())

    def capture(self) -> Capture:
        """A context manager to *capture* the result of print() or log() in a string,
        rather than writing it to the console.

        Example:
            >>> from rich.console import Console
            >>> console = Console()
            >>> with console.capture() as capture:
            ...     console.print("[bold magenta]Hello World[/]")
            >>> print(capture.get())

        Returns:
            Capture: Context manager with disables writing to the terminal.
        """
        capture = Capture(self)
        return capture

    def pager(
        self, pager: Optional[Pager] = None, styles: bool = False, links: bool = False
    ) -> PagerContext:
        """A context manager to display anything printed within a "pager". The pager application
        is defined by the system and will typically support at least pressing a key to scroll.

        Args:
            pager (Pager, optional): A pager object, or None to use :class:`~rich.pager.SystemPager`. Defaults to None.
            styles (bool, optional): Show styles in pager. Defaults to False.
            links (bool, optional): Show links in pager. Defaults to False.

        Example:
            >>> from rich.console import Console
            >>> from rich.__main__ import make_test_card
            >>> console = Console()
            >>> with console.pager():
                    console.print(make_test_card())

        Returns:
            PagerContext: A context manager.
        """
        return PagerContext(self, pager=pager, styles=styles, links=links)

    def line(self, count: int = 1) -> None:
        """Write new line(s).

        Args:
            count (int, optional): Number of new lines. Defaults to 1.
        """

        assert count >= 0, "count must be >= 0"
        self.print(NewLine(count))

    def clear(self, home: bool = True) -> None:
        """Clear the screen.

        Args:
            home (bool, optional): Also move the cursor to 'home' position. Defaults to True.
        """
        if home:
            self.control(Control.clear(), Control.home())
        else:
            self.control(Control.clear())

    def status(
        self,
        status: RenderableType,
        *,
        spinner: str = "dots",
        spinner_style: StyleType = "status.spinner",
        speed: float = 1.0,
        refresh_per_second: float = 12.5,
    ) -> "Status":
        """Display a status and spinner.

        Args:
            status (RenderableType): A status renderable (str or Text typically).
            spinner (str, optional): Name of spinner animation (see python -m rich.spinner). Defaults to "dots".
            spinner_style (StyleType, optional): Style of spinner. Defaults to "status.spinner".
            speed (float, optional): Speed factor for spinner animation. Defaults to 1.0.
            refresh_per_second (float, optional): Number of refreshes per second. Defaults to 12.5.

        Returns:
            Status: A Status object that may be used as a context manager.
        """
        from .status import Status

        status_renderable = Status(
            status,
            console=self,
            spinner=spinner,
            spinner_style=spinner_style,
            speed=speed,
            refresh_per_second=refresh_per_second,
        )
        return status_renderable

    def show_cursor(self, show: bool = True) -> bool:
        """Show or hide the cursor.

        Args:
            show (bool, optional): Set visibility of the cursor.
        """
        if self.is_terminal:
            self.control(Control.show_cursor(show))
            return True
        return False

    def set_alt_screen(self, enable: bool = True) -> bool:
        """Enables alternative screen mode.

        Note, if you enable this mode, you should ensure that is disabled before
        the application exits. See :meth:`~rich.Console.screen` for a context manager
        that handles this for you.

        Args:
            enable (bool, optional): Enable (True) or disable (False) alternate screen. Defaults to True.

        Returns:
            bool: True if the control codes were written.

        """
        changed = False
        if self.is_terminal and not self.legacy_windows:
            self.control(Control.alt_screen(enable))
            changed = True
            self._is_alt_screen = enable
        return changed

    @property
    def is_alt_screen(self) -> bool:
        """Check if the alt screen was enabled.

        Returns:
            bool: True if the alt screen was enabled, otherwise False.
        """
        return self._is_alt_screen

    def set_window_title(self, title: str) -> bool:
        """Set the title of the console terminal window.

        Warning: There is no means within Rich of "resetting" the window title to its
        previous value, meaning the title you set will persist even after your application
        exits.

        ``fish`` shell resets the window title before and after each command by default,
        negating this issue. Windows Terminal and command prompt will also reset the title for you.
        Most other shells and terminals, however, do not do this.

        Some terminals may require configuration changes before you can set the title.
        Some terminals may not support setting the title at all.

        Other software (including the terminal itself, the shell, custom prompts, plugins, etc.)
        may also set the terminal window title. This could result in whatever value you write
        using this method being overwritten.

        Args:
            title (str): The new title of the terminal window.

        Returns:
            bool: True if the control code to change the terminal title was
                written, otherwise False. Note that a return value of True
                does not guarantee that the window title has actually changed,
                since the feature may be unsupported/disabled in some terminals.
        """
        if self.is_terminal:
            self.control(Control.title(title))
            return True
        return False

    def screen(
        self, hide_cursor: bool = True, style: Optional[StyleType] = None
    ) -> "ScreenContext":
        """Context manager to enable and disable 'alternative screen' mode.

        Args:
            hide_cursor (bool, optional): Also hide the cursor. Defaults to False.
            style (Style, optional): Optional style for screen. Defaults to None.

        Returns:
            ~ScreenContext: Context which enables alternate screen on enter, and disables it on exit.
        """
        return ScreenContext(self, hide_cursor=hide_cursor, style=style or "")

    def measure(
        self, renderable: RenderableType, *, options: Optional[ConsoleOptions] = None
    ) -> Measurement:
        """Measure a renderable. Returns a :class:`~rich.measure.Measurement` object which contains
        information regarding the number of characters required to print the renderable.

        Args:
            renderable (RenderableType): Any renderable or string.
            options (Optional[ConsoleOptions], optional): Options to use when measuring, or None
                to use default options. Defaults to None.

        Returns:
            Measurement: A measurement of the renderable.
        """
        measurement = Measurement.get(self, options or self.options, renderable)
        return measurement

    def render(
        self, renderable: RenderableType, options: Optional[ConsoleOptions] = None
    ) -> Iterable[Segment]:
        """Render an object in to an iterable of `Segment` instances.

        This method contains the logic for rendering objects with the console protocol.
        You are unlikely to need to use it directly, unless you are extending the library.

        Args:
            renderable (RenderableType): An object supporting the console protocol, or
                an object that may be converted to a string.
            options (ConsoleOptions, optional): An options object, or None to use self.options. Defaults to None.

        Returns:
            Iterable[Segment]: An iterable of segments that may be rendered.
        """

        _options = options or self.options
        if _options.max_width < 1:
            # No space to render anything. This prevents potential recursion errors.
            return
        render_iterable: RenderResult

        renderable = rich_cast(renderable)
        if hasattr(renderable, "__rich_console__") and not isclass(renderable):
            render_iterable = renderable.__rich_console__(self, _options)
        elif isinstance(renderable, str):
            text_renderable = self.render_str(
                renderable, highlight=_options.highlight, markup=_options.markup
            )
            render_iterable = text_renderable.__rich_console__(self, _options)
        else:
            raise errors.NotRenderableError(
                f"Unable to render {renderable!r}; "
                "A str, Segment or object with __rich_console__ method is required"
            )

        try:
            iter_render = iter(render_iterable)
        except TypeError:
            raise errors.NotRenderableError(
                f"object {render_iterable!r} is not renderable"
            )
        _Segment = Segment
        _options = _options.reset_height()
        for render_output in iter_render:
            if isinstance(render_output, _Segment):
                yield render_output
            else:
                yield from self.render(render_output, _options)

    def render_lines(
        self,
        renderable: RenderableType,
        options: Optional[ConsoleOptions] = None,
        *,
        style: Optional[Style] = None,
        pad: bool = True,
        new_lines: bool = False,
    ) -> List[List[Segment]]:
        """Render objects in to a list of lines.

        The output of render_lines is useful when further formatting of rendered console text
        is required, such as the Panel class which draws a border around any renderable object.

        Args:
            renderable (RenderableType): Any object renderable in the console.
            options (Optional[ConsoleOptions], optional): Console options, or None to use self.options. Default to ``None``.
            style (Style, optional): Optional style to apply to renderables. Defaults to ``None``.
            pad (bool, optional): Pad lines shorter than render width. Defaults to ``True``.
            new_lines (bool, optional): Include "\n" characters at end of lines.

        Returns:
            List[List[Segment]]: A list of lines, where a line is a list of Segment objects.
        """
        with self._lock:
            render_options = options or self.options
            _rendered = self.render(renderable, render_options)
            if style:
                _rendered = Segment.apply_style(_rendered, style)

            render_height = render_options.height
            if render_height is not None:
                render_height = max(0, render_height)

            lines = list(
                islice(
                    Segment.split_and_crop_lines(
                        _rendered,
                        render_options.max_width,
                        include_new_lines=new_lines,
                        pad=pad,
                        style=style,
                    ),
                    None,
                    render_height,
                )
            )
            if render_options.height is not None:
                extra_lines = render_options.height - len(lines)
                if extra_lines > 0:
                    pad_line = [
                        (
                            [
                                Segment(" " * render_options.max_width, style),
                                Segment("\n"),
                            ]
                            if new_lines
                            else [Segment(" " * render_options.max_width, style)]
                        )
                    ]
                    lines.extend(pad_line * extra_lines)

            return lines

    def render_str(
        self,
        text: str,
        *,
        style: Union[str, Style] = "",
        justify: Optional[JustifyMethod] = None,
        overflow: Optional[OverflowMethod] = None,
        emoji: Optional[bool] = None,
        markup: Optional[bool] = None,
        highlight: Optional[bool] = None,
        highlighter: Optional[HighlighterType] = None,
    ) -> "Text":
        """Convert a string to a Text instance. This is called automatically if
        you print or log a string.

        Args:
            text (str): Text to render.
            style (Union[str, Style], optional): Style to apply to rendered text.
            justify (str, optional): Justify method: "default", "left", "center", "full", or "right". Defaults to ``None``.
            overflow (str, optional): Overflow method: "crop", "fold", or "ellipsis". Defaults to ``None``.
            emoji (Optional[bool], optional): Enable emoji, or ``None`` to use Console default.
            markup (Optional[bool], optional): Enable markup, or ``None`` to use Console default.
            highlight (Optional[bool], optional): Enable highlighting, or ``None`` to use Console default.
            highlighter (HighlighterType, optional): Optional highlighter to apply.
        Returns:
            ConsoleRenderable: Renderable object.

        """
        emoji_enabled = emoji or (emoji is None and self._emoji)
        markup_enabled = markup or (markup is None and self._markup)
        highlight_enabled = highlight or (highlight is None and self._highlight)

        if markup_enabled:
            rich_text = render_markup(
                text,
                style=style,
                emoji=emoji_enabled,
                emoji_variant=self._emoji_variant,
            )
            rich_text.justify = justify
            rich_text.overflow = overflow
        else:
            rich_text = Text(
                (
                    _emoji_replace(text, default_variant=self._emoji_variant)
                    if emoji_enabled
                    else text
                ),
                justify=justify,
                overflow=overflow,
                style=style,
            )

        _highlighter = (highlighter or self.highlighter) if highlight_enabled else None
        if _highlighter is not None:
            highlight_text = _highlighter(str(rich_text))
            highlight_text.copy_styles(rich_text)
            return highlight_text

        return rich_text

    def get_style(
        self, name: Union[str, Style], *, default: Optional[Union[Style, str]] = None
    ) -> Style:
        """Get a Style instance by its theme name or parse a definition.

        Args:
            name (str): The name of a style or a style definition.

        Returns:
            Style: A Style object.

        Raises:
            MissingStyle: If no style could be parsed from name.

        """
        if isinstance(name, Style):
            return name

        try:
            style = self._theme_stack.get(name)
            if style is None:
                style = Style.parse(name)
            return style.copy() if style.link else style
        except errors.StyleSyntaxError as error:
            if default is not None:
                return self.get_style(default)
            raise errors.MissingStyle(
                f"Failed to get style {name!r}; {error}"
            ) from None

    def _collect_renderables(
        self,
        objects: Iterable[Any],
        sep: str,
        end: str,
        *,
        justify: Optional[JustifyMethod] = None,
        emoji: Optional[bool] = None,
        markup: Optional[bool] = None,
        highlight: Optional[bool] = None,
    ) -> List[ConsoleRenderable]:
        """Combine a number of renderables and text into one renderable.

        Args:
            objects (Iterable[Any]): Anything that Rich can render.
            sep (str): String to write between print data.
            end (str): String to write at end of print data.
            justify (str, optional): One of "left", "right", "center", or "full". Defaults to ``None``.
            emoji (Optional[bool], optional): Enable emoji code, or ``None`` to use console default.
            markup (Optional[bool], optional): Enable markup, or ``None`` to use console default.
            highlight (Optional[bool], optional): Enable automatic highlighting, or ``None`` to use console default.

        Returns:
            List[ConsoleRenderable]: A list of things to render.
        """
        renderables: List[ConsoleRenderable] = []
        _append = renderables.append
        text: List[Text] = []
        append_text = text.append

        append = _append
        if justify in ("left", "center", "right"):

            def align_append(renderable: RenderableType) -> None:
                _append(Align(renderable, cast(AlignMethod, justify)))

            append = align_append

        _highlighter: HighlighterType = _null_highlighter
        if highlight or (highlight is None and self._highlight):
            _highlighter = self.highlighter

        def check_text() -> None:
            if text:
                sep_text = Text(sep, justify=justify, end=end)
                append(sep_text.join(text))
                text.clear()

        for renderable in objects:
            renderable = rich_cast(renderable)
            if isinstance(renderable, str):
                append_text(
                    self.render_str(
                        renderable,
                        emoji=emoji,
                        markup=markup,
                        highlight=highlight,
                        highlighter=_highlighter,
                    )
                )
            elif isinstance(renderable, Text):
                append_text(renderable)
            elif isinstance(renderable, ConsoleRenderable):
                check_text()
                append(renderable)
            elif is_expandable(renderable):
                check_text()
                append(Pretty(renderable, highlighter=_highlighter))
            else:
                append_text(_highlighter(str(renderable)))

        check_text()

        if self.style is not None:
            style = self.get_style(self.style)
            renderables = [Styled(renderable, style) for renderable in renderables]

        return renderables

    def rule(
        self,
        title: TextType = "",
        *,
        characters: str = "─",
        style: Union[str, Style] = "rule.line",
        align: AlignMethod = "center",
    ) -> None:
        """Draw a line with optional centered title.

        Args:
            title (str, optional): Text to render over the rule. Defaults to "".
            characters (str, optional): Character(s) to form the line. Defaults to "─".
            style (str, optional): Style of line. Defaults to "rule.line".
            align (str, optional): How to align the title, one of "left", "center", or "right". Defaults to "center".
        """
        from .rule import Rule

        rule = Rule(title=title, characters=characters, style=style, align=align)
        self.print(rule)

    def control(self, *control: Control) -> None:
        """Insert non-printing control codes.

        Args:
            control_codes (str): Control codes, such as those that may move the cursor.
        """
        if not self.is_dumb_terminal:
            with self:
                self._buffer.extend(_control.segment for _control in control)

    def out(
        self,
        *objects: Any,
        sep: str = " ",
        end: str = "\n",
        style: Optional[Union[str, Style]] = None,
        highlight: Optional[bool] = None,
    ) -> None:
        """Output to the terminal. This is a low-level way of writing to the terminal which unlike
        :meth:`~rich.console.Console.print` won't pretty print, wrap text, or apply markup, but will
        optionally apply highlighting and a basic style.

        Args:
            sep (str, optional): String to write between print data. Defaults to " ".
            end (str, optional): String to write at end of print data. Defaults to "\\\\n".
            style (Union[str, Style], optional): A style to apply to output. Defaults to None.
            highlight (Optional[bool], optional): Enable automatic highlighting, or ``None`` to use
                console default. Defaults to ``None``.
        """
        raw_output: str = sep.join(str(_object) for _object in objects)
        self.print(
            raw_output,
            style=style,
            highlight=highlight,
            emoji=False,
            markup=False,
            no_wrap=True,
            overflow="ignore",
            crop=False,
            end=end,
        )

    def print(
        self,
        *objects: Any,
        sep: str = " ",
        end: str = "\n",
        style: Optional[Union[str, Style]] = None,
        justify: Optional[JustifyMethod] = None,
        overflow: Optional[OverflowMethod] = None,
        no_wrap: Optional[bool] = None,
        emoji: Optional[bool] = None,
        markup: Optional[bool] = None,
        highlight: Optional[bool] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        crop: bool = True,
        soft_wrap: Optional[bool] = None,
        new_line_start: bool = False,
    ) -> None:
        """Print to the console.

        Args:
            objects (positional args): Objects to log to the terminal.
            sep (str, optional): String to write between print data. Defaults to " ".
            end (str, optional): String to write at end of print data. Defaults to "\\\\n".
            style (Union[str, Style], optional): A style to apply to output. Defaults to None.
            justify (str, optional): Justify method: "default", "left", "right", "center", or "full". Defaults to ``None``.
            overflow (str, optional): Overflow method: "ignore", "crop", "fold", or "ellipsis". Defaults to None.
            no_wrap (Optional[bool], optional): Disable word wrapping. Defaults to None.
            emoji (Optional[bool], optional): Enable emoji code, or ``None`` to use console default. Defaults to ``None``.
            markup (Optional[bool], optional): Enable markup, or ``None`` to use console default. Defaults to ``None``.
            highlight (Optional[bool], optional): Enable automatic highlighting, or ``None`` to use console default. Defaults to ``None``.
            width (Optional[int], optional): Width of output, or ``None`` to auto-detect. Defaults to ``None``.
            crop (Optional[bool], optional): Crop output to width of terminal. Defaults to True.
            soft_wrap (bool, optional): Enable soft wrap mode which disables word wrapping and cropping of text or ``None`` for
                Console default. Defaults to ``None``.
            new_line_start (bool, False): Insert a new line at the start if the output contains more than one line. Defaults to ``False``.
        """
        if not objects:
            objects = (NewLine(),)

        if soft_wrap is None:
            soft_wrap = self.soft_wrap
        if soft_wrap:
            if no_wrap is None:
                no_wrap = True
            if overflow is None:
                overflow = "ignore"
            crop = False
        render_hooks = self._render_hooks[:]
        with self:
            renderables = self._collect_renderables(
                objects,
                sep,
                end,
                justify=justify,
                emoji=emoji,
                markup=markup,
                highlight=highlight,
            )
            for hook in render_hooks:
                renderables = hook.process_renderables(renderables)
            render_options = self.options.update(
                justify=justify,
                overflow=overflow,
                width=min(width, self.width) if width is not None else NO_CHANGE,
                height=height,
                no_wrap=no_wrap,
                markup=markup,
                highlight=highlight,
            )

            new_segments: List[Segment] = []
            extend = new_segments.extend
            render = self.render
            if style is None:
                for renderable in renderables:
                    extend(render(renderable, render_options))
            else:
                for renderable in renderables:
                    extend(
                        Segment.apply_style(
                            render(renderable, render_options), self.get_style(style)
                        )
                    )
            if new_line_start:
                if (
                    len("".join(segment.text for segment in new_segments).splitlines())
                    > 1
                ):
                    new_segments.insert(0, Segment.line())
            if crop:
                buffer_extend = self._buffer.extend
                for line in Segment.split_and_crop_lines(
                    new_segments, self.width, pad=False
                ):
                    buffer_extend(line)
            else:
                self._buffer.extend(new_segments)

    def print_json(
        self,
        json: Optional[str] = None,
        *,
        data: Any = None,
        indent: Union[None, int, str] = 2,
        highlight: bool = True,
        skip_keys: bool = False,
        ensure_ascii: bool = False,
        check_circular: bool = True,
        allow_nan: bool = True,
        default: Optional[Callable[[Any], Any]] = None,
        sort_keys: bool = False,
    ) -> None:
        """Pretty prints JSON. Output will be valid JSON.

        Args:
            json (Optional[str]): A string containing JSON.
            data (Any): If json is not supplied, then encode this data.
            indent (Union[None, int, str], optional): Number of spaces to indent. Defaults to 2.
            highlight (bool, optional): Enable highlighting of output: Defaults to True.
            skip_keys (bool, optional): Skip keys not of a basic type. Defaults to False.
            ensure_ascii (bool, optional): Escape all non-ascii characters. Defaults to False.
            check_circular (bool, optional): Check for circular references. Defaults to True.
            allow_nan (bool, optional): Allow NaN and Infinity values. Defaults to True.
            default (Callable, optional): A callable that converts values that can not be encoded
                in to something that can be JSON encoded. Defaults to None.
            sort_keys (bool, optional): Sort dictionary keys. Defaults to False.
        """
        from pip._vendor.rich.json import JSON

        if json is None:
            json_renderable = JSON.from_data(
                data,
                indent=indent,
                highlight=highlight,
                skip_keys=skip_keys,
                ensure_ascii=ensure_ascii,
                check_circular=check_circular,
                allow_nan=allow_nan,
                default=default,
                sort_keys=sort_keys,
            )
        else:
            if not isinstance(json, str):
                raise TypeError(
                    f"json must be str. Did you mean print_json(data={json!r}) ?"
                )
            json_renderable = JSON(
                json,
                indent=indent,
                highlight=highlight,
                skip_keys=skip_keys,
                ensure_ascii=ensure_ascii,
                check_circular=check_circular,
                allow_nan=allow_nan,
                default=default,
                sort_keys=sort_keys,
            )
        self.print(json_renderable, soft_wrap=True)

    def update_screen(
        self,
        renderable: RenderableType,
        *,
        region: Optional[Region] = None,
        options: Optional[ConsoleOptions] = None,
    ) -> None:
        """Update the screen at a given offset.

        Args:
            renderable (RenderableType): A Rich renderable.
            region (Region, optional): Region of screen to update, or None for entire screen. Defaults to None.
            x (int, optional): x offset. Defaults to 0.
            y (int, optional): y offset. Defaults to 0.

        Raises:
            errors.NoAltScreen: If the Console isn't in alt screen mode.

        """
        if not self.is_alt_screen:
            raise errors.NoAltScreen("Alt screen must be enabled to call update_screen")
        render_options = options or self.options
        if region is None:
            x = y = 0
            render_options = render_options.update_dimensions(
                render_options.max_width, render_options.height or self.height
            )
        else:
            x, y, width, height = region
            render_options = render_options.update_dimensions(width, height)

        lines = self.render_lines(renderable, options=render_options)
        self.update_screen_lines(lines, x, y)

    def update_screen_lines(
        self, lines: List[List[Segment]], x: int = 0, y: int = 0
    ) -> None:
        """Update lines of the screen at a given offset.

        Args:
            lines (List[List[Segment]]): Rendered lines (as produced by :meth:`~rich.Console.render_lines`).
            x (int, optional): x offset (column no). Defaults to 0.
            y (int, optional): y offset (column no). Defaults to 0.

        Raises:
            errors.NoAltScreen: If the Console isn't in alt screen mode.
        """
        if not self.is_alt_screen:
            raise errors.NoAltScreen("Alt screen must be enabled to call update_screen")
        screen_update = ScreenUpdate(lines, x, y)
        segments = self.render(screen_update)
        self._buffer.extend(segments)
        self._check_buffer()

    def print_exception(
        self,
        *,
        width: Optional[int] = 100,
        extra_lines: int = 3,
        theme: Optional[str] = None,
        word_wrap: bool = False,
        show_locals: bool = False,
        suppress: Iterable[Union[str, ModuleType]] = (),
        max_frames: int = 100,
    ) -> None:
        """Prints a rich render of the last exception and traceback.

        Args:
            width (Optional[int], optional): Number of characters used to render code. Defaults to 100.
            extra_lines (int, optional): Additional lines of code to render. Defaults to 3.
            theme (str, optional): Override pygments theme used in traceback
            word_wrap (bool, optional): Enable word wrapping of long lines. Defaults to False.
            show_locals (bool, optional): Enable display of local variables. Defaults to False.
            suppress (Iterable[Union[str, ModuleType]]): Optional sequence of modules or paths to exclude from traceback.
            max_frames (int): Maximum number of frames to show in a traceback, 0 for no maximum. Defaults to 100.
        """
        from .traceback import Traceback

        traceback = Traceback(
            width=width,
            extra_lines=extra_lines,
            theme=theme,
            word_wrap=word_wrap,
            show_locals=show_locals,
            suppress=suppress,
            max_frames=max_frames,
        )
        self.print(traceback)

    @staticmethod
    def _caller_frame_info(
        offset: int,
        currentframe: Callable[[], Optional[FrameType]] = inspect.currentframe,
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Get caller frame information.

        Args:
            offset (int): the caller offset within the current frame stack.
            currentframe (Callable[[], Optional[FrameType]], optional): the callable to use to
                retrieve the current frame. Defaults to ``inspect.currentframe``.

        Returns:
            Tuple[str, int, Dict[str, Any]]: A tuple containing the filename, the line number and
                the dictionary of local variables associated with the caller frame.

        Raises:
            RuntimeError: If the stack offset is invalid.
        """
        # Ignore the frame of this local helper
        offset += 1

        frame = currentframe()
        if frame is not None:
            # Use the faster currentframe where implemented
            while offset and frame is not None:
                frame = frame.f_back
                offset -= 1
            assert frame is not None
            return frame.f_code.co_filename, frame.f_lineno, frame.f_locals
        else:
            # Fallback to the slower stack
            frame_info = inspect.stack()[offset]
            return frame_info.filename, frame_info.lineno, frame_info.frame.f_locals

    def log(
        self,
        *objects: Any,
        sep: str = " ",
        end: str = "\n",
        style: Optional[Union[str, Style]] = None,
        justify: Optional[JustifyMethod] = None,
        emoji: Optional[bool] = None,
        markup: Optional[bool] = None,
        highlight: Optional[bool] = None,
        log_locals: bool = False,
        _stack_offset: int = 1,
    ) -> None:
        """Log rich content to the terminal.

        Args:
            objects (positional args): Objects to log to the terminal.
            sep (str, optional): String to write between print data. Defaults to " ".
            end (str, optional): String to write at end of print data. Defaults to "\\\\n".
            style (Union[str, Style], optional): A style to apply to output. Defaults to None.
            justify (str, optional): One of "left", "right", "center", or "full". Defaults to ``None``.
            emoji (Optional[bool], optional): Enable emoji code, or ``None`` to use console default. Defaults to None.
            markup (Optional[bool], optional): Enable markup, or ``None`` to use console default. Defaults to None.
            highlight (Optional[bool], optional): Enable automatic highlighting, or ``None`` to use console default. Defaults to None.
            log_locals (bool, optional): Boolean to enable logging of locals where ``log()``
                was called. Defaults to False.
            _stack_offset (int, optional): Offset of caller from end of call stack. Defaults to 1.
        """
        if not objects:
            objects = (NewLine(),)

        render_hooks = self._render_hooks[:]

        with self:
            renderables = self._collect_renderables(
                objects,
                sep,
                end,
                justify=justify,
                emoji=emoji,
                markup=markup,
                highlight=highlight,
            )
            if style is not None:
                renderables = [Styled(renderable, style) for renderable in renderables]

            filename, line_no, locals = self._caller_frame_info(_stack_offset)
            link_path = None if filename.startswith("<") else os.path.abspath(filename)
            path = filename.rpartition(os.sep)[-1]
            if log_locals:
                locals_map = {
                    key: value
                    for key, value in locals.items()
                    if not key.startswith("__")
                }
                renderables.append(render_scope(locals_map, title="[i]locals"))

            renderables = [
                self._log_render(
                    self,
                    renderables,
                    log_time=self.get_datetime(),
                    path=path,
                    line_no=line_no,
                    link_path=link_path,
                )
            ]
            for hook in render_hooks:
                renderables = hook.process_renderables(renderables)
            new_segments: List[Segment] = []
            extend = new_segments.extend
            render = self.render
            render_options = self.options
            for renderable in renderables:
                extend(render(renderable, render_options))
            buffer_extend = self._buffer.extend
            for line in Segment.split_and_crop_lines(
                new_segments, self.width, pad=False
            ):
                buffer_extend(line)

    def on_broken_pipe(self) -> None:
        """This function is called when a `BrokenPipeError` is raised.

        This can occur when piping Textual output in Linux and macOS.
        The default implementation is to exit the app, but you could implement
        this method in a subclass to change the behavior.

        See https://docs.python.org/3/library/signal.html#note-on-sigpipe for details.
        """
        self.quiet = True
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        raise SystemExit(1)

    def _check_buffer(self) -> None:
        """Check if the buffer may be rendered. Render it if it can (e.g. Console.quiet is False)
        Rendering is supported on Windows, Unix and Jupyter environments. For
        legacy Windows consoles, the win32 API is called directly.
        This method will also record what it renders if recording is enabled via Console.record.
        """
        if self.quiet:
            del self._buffer[:]
            return

        try:
            self._write_buffer()
        except BrokenPipeError:
            self.on_broken_pipe()

    def _write_buffer(self) -> None:
        """Write the buffer to the output file."""

        with self._lock:
            if self.record and not self._buffer_index:
                with self._record_buffer_lock:
                    self._record_buffer.extend(self._buffer[:])

            if self._buffer_index == 0:
                if self.is_jupyter:  # pragma: no cover
                    from .jupyter import display

                    display(self._buffer, self._render_buffer(self._buffer[:]))
                    del self._buffer[:]
                else:
                    if WINDOWS:
                        use_legacy_windows_render = False
                        if self.legacy_windows:
                            fileno = get_fileno(self.file)
                            if fileno is not None:
                                use_legacy_windows_render = (
                                    fileno in _STD_STREAMS_OUTPUT
                                )

                        if use_legacy_windows_render:
                            from pip._vendor.rich._win32_console import LegacyWindowsTerm
                            from pip._vendor.rich._windows_renderer import legacy_windows_render

                            buffer = self._buffer[:]
                            if self.no_color and self._color_system:
                                buffer = list(Segment.remove_color(buffer))

                            legacy_windows_render(buffer, LegacyWindowsTerm(self.file))
                        else:
                            # Either a non-std stream on legacy Windows, or modern Windows.
                            text = self._render_buffer(self._buffer[:])
                            # https://bugs.python.org/issue37871
                            # https://github.com/python/cpython/issues/82052
                            # We need to avoid writing more than 32Kb in a single write, due to the above bug
                            write = self.file.write
                            # Worse case scenario, every character is 4 bytes of utf-8
                            MAX_WRITE = 32 * 1024 // 4
                            try:
                                if len(text) <= MAX_WRITE:
                                    write(text)
                                else:
                                    batch: List[str] = []
                                    batch_append = batch.append
                                    size = 0
                                    for line in text.splitlines(True):
                                        if size + len(line) > MAX_WRITE and batch:
                                            write("".join(batch))
                                            batch.clear()
                                            size = 0
                                        batch_append(line)
                                        size += len(line)
                                    if batch:
                                        write("".join(batch))
                                        batch.clear()
                            except UnicodeEncodeError as error:
                                error.reason = f"{error.reason}\n*** You may need to add PYTHONIOENCODING=utf-8 to your environment ***"
                                raise
                    else:
                        text = self._render_buffer(self._buffer[:])
                        try:
                            self.file.write(text)
                        except UnicodeEncodeError as error:
                            error.reason = f"{error.reason}\n*** You may need to add PYTHONIOENCODING=utf-8 to your environment ***"
                            raise

                    self.file.flush()
                    del self._buffer[:]

    def _render_buffer(self, buffer: Iterable[Segment]) -> str:
        """Render buffered output, and clear buffer."""
        output: List[str] = []
        append = output.append
        color_system = self._color_system
        legacy_windows = self.legacy_windows
        not_terminal = not self.is_terminal
        if self.no_color and color_system:
            buffer = Segment.remove_color(buffer)
        for text, style, control in buffer:
            if style:
                append(
                    style.render(
                        text,
                        color_system=color_system,
                        legacy_windows=legacy_windows,
                    )
                )
            elif not (not_terminal and control):
                append(text)

        rendered = "".join(output)
        return rendered

    def input(
        self,
        prompt: TextType = "",
        *,
        markup: bool = True,
        emoji: bool = True,
        password: bool = False,
        stream: Optional[TextIO] = None,
    ) -> str:
        """Displays a prompt and waits for input from the user. The prompt may contain color / style.

        It works in the same way as Python's builtin :func:`input` function and provides elaborate line editing and history features if Python's builtin :mod:`readline` module is previously loaded.

        Args:
            prompt (Union[str, Text]): Text to render in the prompt.
            markup (bool, optional): Enable console markup (requires a str prompt). Defaults to True.
            emoji (bool, optional): Enable emoji (requires a str prompt). Defaults to True.
            password: (bool, optional): Hide typed text. Defaults to False.
            stream: (TextIO, optional): Optional file to read input from (rather than stdin). Defaults to None.

        Returns:
            str: Text read from stdin.
        """
        if prompt:
            self.print(prompt, markup=markup, emoji=emoji, end="")
        if password:
            result = getpass("", stream=stream)
        else:
            if stream:
                result = stream.readline()
            else:
                result = input()
        return result

    def export_text(self, *, clear: bool = True, styles: bool = False) -> str:
        """Generate text from console contents (requires record=True argument in constructor).

        Args:
            clear (bool, optional): Clear record buffer after exporting. Defaults to ``True``.
            styles (bool, optional): If ``True``, ansi escape codes will be included. ``False`` for plain text.
                Defaults to ``False``.

        Returns:
            str: String containing console contents.

        """
        assert (
            self.record
        ), "To export console contents set record=True in the constructor or instance"

        with self._record_buffer_lock:
            if styles:
                text = "".join(
                    (style.render(text) if style else text)
                    for text, style, _ in self._record_buffer
                )
            else:
                text = "".join(
                    segment.text
                    for segment in self._record_buffer
                    if not segment.control
                )
            if clear:
                del self._record_buffer[:]
        return text

    def save_text(self, path: str, *, clear: bool = True, styles: bool = False) -> None:
        """Generate text from console and save to a given location (requires record=True argument in constructor).

        Args:
            path (str): Path to write text files.
            clear (bool, optional): Clear record buffer after exporting. Defaults to ``True``.
            styles (bool, optional): If ``True``, ansi style codes will be included. ``False`` for plain text.
                Defaults to ``False``.

        """
        text = self.export_text(clear=clear, styles=styles)
        with open(path, "w", encoding="utf-8") as write_file:
            write_file.write(text)

    def export_html(
        self,
        *,
        theme: Optional[TerminalTheme] = None,
        clear: bool = True,
        code_format: Optional[str] = None,
        inline_styles: bool = False,
    ) -> str:
        """Generate HTML from console contents (requires record=True argument in constructor).

        Args:
            theme (TerminalTheme, optional): TerminalTheme object containing console colors.
            clear (bool, optional): Clear record buffer after exporting. Defaults to ``True``.
            code_format (str, optional): Format string to render HTML. In addition to '{foreground}',
                '{background}', and '{code}', should contain '{stylesheet}' if inline_styles is ``False``.
            inline_styles (bool, optional): If ``True`` styles will be inlined in to spans, which makes files
                larger but easier to cut and paste markup. If ``False``, styles will be embedded in a style tag.
                Defaults to False.

        Returns:
            str: String containing console contents as HTML.
        """
        assert (
            self.record
        ), "To export console contents set record=True in the constructor or instance"
        fragments: List[str] = []
        append = fragments.append
        _theme = theme or DEFAULT_TERMINAL_THEME
        stylesheet = ""

        render_code_format = CONSOLE_HTML_FORMAT if code_format is None else code_format

        with self._record_buffer_lock:
            if inline_styles:
                for text, style, _ in Segment.filter_control(
                    Segment.simplify(self._record_buffer)
                ):
                    text = escape(text)
                    if style:
                        rule = style.get_html_style(_theme)
                        if style.link:
                            text = f'<a href="{style.link}">{text}</a>'
                        text = f'<span style="{rule}">{text}</span>' if rule else text
                    append(text)
            else:
                styles: Dict[str, int] = {}
                for text, style, _ in Segment.filter_control(
                    Segment.simplify(self._record_buffer)
                ):
                    text = escape(text)
                    if style:
                        rule = style.get_html_style(_theme)
                        style_number = styles.setdefault(rule, len(styles) + 1)
                        if style.link:
                            text = f'<a class="r{style_number}" href="{style.link}">{text}</a>'
                        else:
                            text = f'<span class="r{style_number}">{text}</span>'
                    append(text)
                stylesheet_rules: List[str] = []
                stylesheet_append = stylesheet_rules.append
                for style_rule, style_number in styles.items():
                    if style_rule:
                        stylesheet_append(f".r{style_number} {{{style_rule}}}")
                stylesheet = "\n".join(stylesheet_rules)

            rendered_code = render_code_format.format(
                code="".join(fragments),
                stylesheet=stylesheet,
                foreground=_theme.foreground_color.hex,
                background=_theme.background_color.hex,
            )
            if clear:
                del self._record_buffer[:]
        return rendered_code

    def save_html(
        self,
        path: str,
        *,
        theme: Optional[TerminalTheme] = None,
        clear: bool = True,
        code_format: str = CONSOLE_HTML_FORMAT,
        inline_styles: bool = False,
    ) -> None:
        """Generate HTML from console contents and write to a file (requires record=True argument in constructor).

        Args:
            path (str): Path to write html file.
            theme (TerminalTheme, optional): TerminalTheme object containing console colors.
            clear (bool, optional): Clear record buffer after exporting. Defaults to ``True``.
            code_format (str, optional): Format string to render HTML. In addition to '{foreground}',
                '{background}', and '{code}', should contain '{stylesheet}' if inline_styles is ``False``.
            inline_styles (bool, optional): If ``True`` styles will be inlined in to spans, which makes files
                larger but easier to cut and paste markup. If ``False``, styles will be embedded in a style tag.
                Defaults to False.

        """
        html = self.export_html(
            theme=theme,
            clear=clear,
            code_format=code_format,
            inline_styles=inline_styles,
        )
        with open(path, "w", encoding="utf-8") as write_file:
            write_file.write(html)

    def export_svg(
        self,
        *,
        title: str = "Rich",
        theme: Optional[TerminalTheme] = None,
        clear: bool = True,
        code_format: str = CONSOLE_SVG_FORMAT,
        font_aspect_ratio: float = 0.61,
        unique_id: Optional[str] = None,
    ) -> str:
        """
        Generate an SVG from the console contents (requires record=True in Console constructor).

        Args:
            title (str, optional): The title of the tab in the output image
            theme (TerminalTheme, optional): The ``TerminalTheme`` object to use to style the terminal
            clear (bool, optional): Clear record buffer after exporting. Defaults to ``True``
            code_format (str, optional): Format string used to generate the SVG. Rich will inject a number of variables
                into the string in order to form the final SVG output. The default template used and the variables
                injected by Rich can be found by inspecting the ``console.CONSOLE_SVG_FORMAT`` variable.
            font_aspect_ratio (float, optional): The width to height ratio of the font used in the ``code_format``
                string. Defaults to 0.61, which is the width to height ratio of Fira Code (the default font).
                If you aren't specifying a different font inside ``code_format``, you probably don't need this.
            unique_id (str, optional): unique id that is used as the prefix for various elements (CSS styles, node
                ids). If not set, this defaults to a computed value based on the recorded content.
        """

        from pip._vendor.rich.cells import cell_len

        style_cache: Dict[Style, str] = {}

        def get_svg_style(style: Style) -> str:
            """Convert a Style to CSS rules for SVG."""
            if style in style_cache:
                return style_cache[style]
            css_rules = []
            color = (
                _theme.foreground_color
                if (style.color is None or style.color.is_default)
                else style.color.get_truecolor(_theme)
            )
            bgcolor = (
                _theme.background_color
                if (style.bgcolor is None or style.bgcolor.is_default)
                else style.bgcolor.get_truecolor(_theme)
            )
            if style.reverse:
                color, bgcolor = bgcolor, color
            if style.dim:
                color = blend_rgb(color, bgcolor, 0.4)
            css_rules.append(f"fill: {color.hex}")
            if style.bold:
                css_rules.append("font-weight: bold")
            if style.italic:
                css_rules.append("font-style: italic;")
            if style.underline:
                css_rules.append("text-decoration: underline;")
            if style.strike:
                css_rules.append("text-decoration: line-through;")

            css = ";".join(css_rules)
            style_cache[style] = css
            return css

        _theme = theme or SVG_EXPORT_THEME

        width = self.width
        char_height = 20
        char_width = char_height * font_aspect_ratio
        line_height = char_height * 1.22

        margin_top = 1
        margin_right = 1
        margin_bottom = 1
        margin_left = 1

        padding_top = 40
        padding_right = 8
        padding_bottom = 8
        padding_left = 8

        padding_width = padding_left + padding_right
        padding_height = padding_top + padding_bottom
        margin_width = margin_left + margin_right
        margin_height = margin_top + margin_bottom

        text_backgrounds: List[str] = []
        text_group: List[str] = []
        classes: Dict[str, int] = {}
        style_no = 1

        def escape_text(text: str) -> str:
            """HTML escape text and replace spaces with nbsp."""
            return escape(text).replace(" ", "&#160;")

        def make_tag(
            name: str, content: Optional[str] = None, **attribs: object
        ) -> str:
            """Make a tag from name, content, and attributes."""

            def stringify(value: object) -> str:
                if isinstance(value, (float)):
                    return format(value, "g")
                return str(value)

            tag_attribs = " ".join(
                f'{k.lstrip("_").replace("_", "-")}="{stringify(v)}"'
                for k, v in attribs.items()
            )
            return (
                f"<{name} {tag_attribs}>{content}</{name}>"
                if content
                else f"<{name} {tag_attribs}/>"
            )

        with self._record_buffer_lock:
            segments = list(Segment.filter_control(self._record_buffer))
            if clear:
                self._record_buffer.clear()

        if unique_id is None:
            unique_id = "terminal-" + str(
                zlib.adler32(
                    ("".join(repr(segment) for segment in segments)).encode(
                        "utf-8",
                        "ignore",
                    )
                    + title.encode("utf-8", "ignore")
                )
            )
        y = 0
        for y, line in enumerate(Segment.split_and_crop_lines(segments, length=width)):
            x = 0
            for text, style, _control in line:
                style = style or Style()
                rules = get_svg_style(style)
                if rules not in classes:
                    classes[rules] = style_no
                    style_no += 1
                class_name = f"r{classes[rules]}"

                if style.reverse:
                    has_background = True
                    background = (
                        _theme.foreground_color.hex
                        if style.color is None
                        else style.color.get_truecolor(_theme).hex
                    )
                else:
                    bgcolor = style.bgcolor
                    has_background = bgcolor is not None and not bgcolor.is_default
                    background = (
                        _theme.background_color.hex
                        if style.bgcolor is None
                        else style.bgcolor.get_truecolor(_theme).hex
                    )

                text_length = cell_len(text)
                if has_background:
                    text_backgrounds.append(
                        make_tag(
                            "rect",
                            fill=background,
                            x=x * char_width,
                            y=y * line_height + 1.5,
                            width=char_width * text_length,
                            height=line_height + 0.25,
                            shape_rendering="crispEdges",
                        )
                    )

                if text != " " * len(text):
                    text_group.append(
                        make_tag(
                            "text",
                            escape_text(text),
                            _class=f"{unique_id}-{class_name}",
                            x=x * char_width,
                            y=y * line_height + char_height,
                            textLength=char_width * len(text),
                            clip_path=f"url(#{unique_id}-line-{y})",
                        )
                    )
                x += cell_len(text)

        line_offsets = [line_no * line_height + 1.5 for line_no in range(y)]
        lines = "\n".join(
            f"""<clipPath id="{unique_id}-line-{line_no}">
    {make_tag("rect", x=0, y=offset, width=char_width * width, height=line_height + 0.25)}
            </clipPath>"""
            for line_no, offset in enumerate(line_offsets)
        )

        styles = "\n".join(
            f".{unique_id}-r{rule_no} {{ {css} }}" for css, rule_no in classes.items()
        )
        backgrounds = "".join(text_backgrounds)
        matrix = "".join(text_group)

        terminal_width = ceil(width * char_width + padding_width)
        terminal_height = (y + 1) * line_height + padding_height
        chrome = make_tag(
            "rect",
            fill=_theme.background_color.hex,
            stroke="rgba(255,255,255,0.35)",
            stroke_width="1",
            x=margin_left,
            y=margin_top,
            width=terminal_width,
            height=terminal_height,
            rx=8,
        )

        title_color = _theme.foreground_color.hex
        if title:
            chrome += make_tag(
                "text",
                escape_text(title),
                _class=f"{unique_id}-title",
                fill=title_color,
                text_anchor="middle",
                x=terminal_width // 2,
                y=margin_top + char_height + 6,
            )
        chrome += f"""
            <g transform="translate(26,22)">
            <circle cx="0" cy="0" r="7" fill="#ff5f57"/>
            <circle cx="22" cy="0" r="7" fill="#febc2e"/>
            <circle cx="44" cy="0" r="7" fill="#28c840"/>
            </g>
        """

        svg = code_format.format(
            unique_id=unique_id,
            char_width=char_width,
            char_height=char_height,
            line_height=line_height,
            terminal_width=char_width * width - 1,
            terminal_height=(y + 1) * line_height - 1,
            width=terminal_width + margin_width,
            height=terminal_height + margin_height,
            terminal_x=margin_left + padding_left,
            terminal_y=margin_top + padding_top,
            styles=styles,
            chrome=chrome,
            backgrounds=backgrounds,
            matrix=matrix,
            lines=lines,
        )
        return svg

    def save_svg(
        self,
        path: str,
        *,
        title: str = "Rich",
        theme: Optional[TerminalTheme] = None,
        clear: bool = True,
        code_format: str = CONSOLE_SVG_FORMAT,
        font_aspect_ratio: float = 0.61,
        unique_id: Optional[str] = None,
    ) -> None:
        """Generate an SVG file from the console contents (requires record=True in Console constructor).

        Args:
            path (str): The path to write the SVG to.
            title (str, optional): The title of the tab in the output image
            theme (TerminalTheme, optional): The ``TerminalTheme`` object to use to style the terminal
            clear (bool, optional): Clear record buffer after exporting. Defaults to ``True``
            code_format (str, optional): Format string used to generate the SVG. Rich will inject a number of variables
                into the string in order to form the final SVG output. The default template used and the variables
                injected by Rich can be found by inspecting the ``console.CONSOLE_SVG_FORMAT`` variable.
            font_aspect_ratio (float, optional): The width to height ratio of the font used in the ``code_format``
                string. Defaults to 0.61, which is the width to height ratio of Fira Code (the default font).
                If you aren't specifying a different font inside ``code_format``, you probably don't need this.
            unique_id (str, optional): unique id that is used as the prefix for various elements (CSS styles, node
                ids). If not set, this defaults to a computed value based on the recorded content.
        """
        svg = self.export_svg(
            title=title,
            theme=theme,
            clear=clear,
            code_format=code_format,
            font_aspect_ratio=font_aspect_ratio,
            unique_id=unique_id,
        )
        with open(path, "w", encoding="utf-8") as write_file:
            write_file.write(svg)


def _svg_hash(svg_main_code: str) -> str:
    """Returns a unique hash for the given SVG main code.

    Args:
        svg_main_code (str): The content we're going to inject in the SVG envelope.

    Returns:
        str: a hash of the given content
    """
    return str(zlib.adler32(svg_main_code.encode()))


if __name__ == "__main__":  # pragma: no cover
    console = Console(record=True)

    console.log(
        "JSONRPC [i]request[/i]",
        5,
        1.3,
        True,
        False,
        None,
        {
            "jsonrpc": "2.0",
            "method": "subtract",
            "params": {"minuend": 42, "subtrahend": 23},
            "id": 3,
        },
    )

    console.log("Hello, World!", "{'a': 1}", repr(console))

    console.print(
        {
            "name": None,
            "empty": [],
            "quiz": {
                "sport": {
                    "answered": True,
                    "q1": {
                        "question": "Which one is correct team name in NBA?",
                        "options": [
                            "New York Bulls",
                            "Los Angeles Kings",
                            "Golden State Warriors",
                            "Huston Rocket",
                        ],
                        "answer": "Huston Rocket",
                    },
                },
                "maths": {
                    "answered": False,
                    "q1": {
                        "question": "5 + 7 = ?",
                        "options": [10, 11, 12, 13],
                        "answer": 12,
                    },
                    "q2": {
                        "question": "12 - 8 = ?",
                        "options": [1, 2, 3, 4],
                        "answer": 4,
                    },
                },
            },
        }
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sortedcontainers\sortedlist.py ===
"""Sorted List
==============

:doc:`Sorted Containers<index>` is an Apache2 licensed Python sorted
collections library, written in pure-Python, and fast as C-extensions. The
:doc:`introduction<introduction>` is the best way to get started.

Sorted list implementations:

.. currentmodule:: sortedcontainers

* :class:`SortedList`
* :class:`SortedKeyList`

"""
# pylint: disable=too-many-lines
from __future__ import print_function

import sys
import traceback

from bisect import bisect_left, bisect_right, insort
from itertools import chain, repeat, starmap
from math import log
from operator import add, eq, ne, gt, ge, lt, le, iadd
from textwrap import dedent

###############################################################################
# BEGIN Python 2/3 Shims
###############################################################################

try:
    from collections.abc import Sequence, MutableSequence
except ImportError:
    from collections import Sequence, MutableSequence

from functools import wraps
from sys import hexversion

if hexversion < 0x03000000:
    from itertools import imap as map  # pylint: disable=redefined-builtin
    from itertools import izip as zip  # pylint: disable=redefined-builtin
    try:
        from thread import get_ident
    except ImportError:
        from dummy_thread import get_ident
else:
    from functools import reduce
    try:
        from _thread import get_ident
    except ImportError:
        from _dummy_thread import get_ident


def recursive_repr(fillvalue='...'):
    "Decorator to make a repr function return fillvalue for a recursive call."
    # pylint: disable=missing-docstring
    # Copied from reprlib in Python 3
    # https://hg.python.org/cpython/file/3.6/Lib/reprlib.py

    def decorating_function(user_function):
        repr_running = set()

        @wraps(user_function)
        def wrapper(self):
            key = id(self), get_ident()
            if key in repr_running:
                return fillvalue
            repr_running.add(key)
            try:
                result = user_function(self)
            finally:
                repr_running.discard(key)
            return result

        return wrapper

    return decorating_function

###############################################################################
# END Python 2/3 Shims
###############################################################################


class SortedList(MutableSequence):
    """Sorted list is a sorted mutable sequence.

    Sorted list values are maintained in sorted order.

    Sorted list values must be comparable. The total ordering of values must
    not change while they are stored in the sorted list.

    Methods for adding values:

    * :func:`SortedList.add`
    * :func:`SortedList.update`
    * :func:`SortedList.__add__`
    * :func:`SortedList.__iadd__`
    * :func:`SortedList.__mul__`
    * :func:`SortedList.__imul__`

    Methods for removing values:

    * :func:`SortedList.clear`
    * :func:`SortedList.discard`
    * :func:`SortedList.remove`
    * :func:`SortedList.pop`
    * :func:`SortedList.__delitem__`

    Methods for looking up values:

    * :func:`SortedList.bisect_left`
    * :func:`SortedList.bisect_right`
    * :func:`SortedList.count`
    * :func:`SortedList.index`
    * :func:`SortedList.__contains__`
    * :func:`SortedList.__getitem__`

    Methods for iterating values:

    * :func:`SortedList.irange`
    * :func:`SortedList.islice`
    * :func:`SortedList.__iter__`
    * :func:`SortedList.__reversed__`

    Methods for miscellany:

    * :func:`SortedList.copy`
    * :func:`SortedList.__len__`
    * :func:`SortedList.__repr__`
    * :func:`SortedList._check`
    * :func:`SortedList._reset`

    Sorted lists use lexicographical ordering semantics when compared to other
    sequences.

    Some methods of mutable sequences are not supported and will raise
    not-implemented error.

    """
    DEFAULT_LOAD_FACTOR = 1000


    def __init__(self, iterable=None, key=None):
        """Initialize sorted list instance.

        Optional `iterable` argument provides an initial iterable of values to
        initialize the sorted list.

        Runtime complexity: `O(n*log(n))`

        >>> sl = SortedList()
        >>> sl
        SortedList([])
        >>> sl = SortedList([3, 1, 2, 5, 4])
        >>> sl
        SortedList([1, 2, 3, 4, 5])

        :param iterable: initial values (optional)

        """
        assert key is None
        self._len = 0
        self._load = self.DEFAULT_LOAD_FACTOR
        self._lists = []
        self._maxes = []
        self._index = []
        self._offset = 0

        if iterable is not None:
            self._update(iterable)


    def __new__(cls, iterable=None, key=None):
        """Create new sorted list or sorted-key list instance.

        Optional `key`-function argument will return an instance of subtype
        :class:`SortedKeyList`.

        >>> sl = SortedList()
        >>> isinstance(sl, SortedList)
        True
        >>> sl = SortedList(key=lambda x: -x)
        >>> isinstance(sl, SortedList)
        True
        >>> isinstance(sl, SortedKeyList)
        True

        :param iterable: initial values (optional)
        :param key: function used to extract comparison key (optional)
        :return: sorted list or sorted-key list instance

        """
        # pylint: disable=unused-argument
        if key is None:
            return object.__new__(cls)
        else:
            if cls is SortedList:
                return object.__new__(SortedKeyList)
            else:
                raise TypeError('inherit SortedKeyList for key argument')


    @property
    def key(self):  # pylint: disable=useless-return
        """Function used to extract comparison key from values.

        Sorted list compares values directly so the key function is none.

        """
        return None


    def _reset(self, load):
        """Reset sorted list load factor.

        The `load` specifies the load-factor of the list. The default load
        factor of 1000 works well for lists from tens to tens-of-millions of
        values. Good practice is to use a value that is the cube root of the
        list size. With billions of elements, the best load factor depends on
        your usage. It's best to leave the load factor at the default until you
        start benchmarking.

        See :doc:`implementation` and :doc:`performance-scale` for more
        information.

        Runtime complexity: `O(n)`

        :param int load: load-factor for sorted list sublists

        """
        values = reduce(iadd, self._lists, [])
        self._clear()
        self._load = load
        self._update(values)


    def clear(self):
        """Remove all values from sorted list.

        Runtime complexity: `O(n)`

        """
        self._len = 0
        del self._lists[:]
        del self._maxes[:]
        del self._index[:]
        self._offset = 0

    _clear = clear


    def add(self, value):
        """Add `value` to sorted list.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList()
        >>> sl.add(3)
        >>> sl.add(1)
        >>> sl.add(2)
        >>> sl
        SortedList([1, 2, 3])

        :param value: value to add to sorted list

        """
        _lists = self._lists
        _maxes = self._maxes

        if _maxes:
            pos = bisect_right(_maxes, value)

            if pos == len(_maxes):
                pos -= 1
                _lists[pos].append(value)
                _maxes[pos] = value
            else:
                insort(_lists[pos], value)

            self._expand(pos)
        else:
            _lists.append([value])
            _maxes.append(value)

        self._len += 1


    def _expand(self, pos):
        """Split sublists with length greater than double the load-factor.

        Updates the index when the sublist length is less than double the load
        level. This requires incrementing the nodes in a traversal from the
        leaf node to the root. For an example traversal see
        ``SortedList._loc``.

        """
        _load = self._load
        _lists = self._lists
        _index = self._index

        if len(_lists[pos]) > (_load << 1):
            _maxes = self._maxes

            _lists_pos = _lists[pos]
            half = _lists_pos[_load:]
            del _lists_pos[_load:]
            _maxes[pos] = _lists_pos[-1]

            _lists.insert(pos + 1, half)
            _maxes.insert(pos + 1, half[-1])

            del _index[:]
        else:
            if _index:
                child = self._offset + pos
                while child:
                    _index[child] += 1
                    child = (child - 1) >> 1
                _index[0] += 1


    def update(self, iterable):
        """Update sorted list by adding all values from `iterable`.

        Runtime complexity: `O(k*log(n))` -- approximate.

        >>> sl = SortedList()
        >>> sl.update([3, 1, 2])
        >>> sl
        SortedList([1, 2, 3])

        :param iterable: iterable of values to add

        """
        _lists = self._lists
        _maxes = self._maxes
        values = sorted(iterable)

        if _maxes:
            if len(values) * 4 >= self._len:
                _lists.append(values)
                values = reduce(iadd, _lists, [])
                values.sort()
                self._clear()
            else:
                _add = self.add
                for val in values:
                    _add(val)
                return

        _load = self._load
        _lists.extend(values[pos:(pos + _load)]
                      for pos in range(0, len(values), _load))
        _maxes.extend(sublist[-1] for sublist in _lists)
        self._len = len(values)
        del self._index[:]

    _update = update


    def __contains__(self, value):
        """Return true if `value` is an element of the sorted list.

        ``sl.__contains__(value)`` <==> ``value in sl``

        Runtime complexity: `O(log(n))`

        >>> sl = SortedList([1, 2, 3, 4, 5])
        >>> 3 in sl
        True

        :param value: search for value in sorted list
        :return: true if `value` in sorted list

        """
        _maxes = self._maxes

        if not _maxes:
            return False

        pos = bisect_left(_maxes, value)

        if pos == len(_maxes):
            return False

        _lists = self._lists
        idx = bisect_left(_lists[pos], value)

        return _lists[pos][idx] == value


    def discard(self, value):
        """Remove `value` from sorted list if it is a member.

        If `value` is not a member, do nothing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList([1, 2, 3, 4, 5])
        >>> sl.discard(5)
        >>> sl.discard(0)
        >>> sl == [1, 2, 3, 4]
        True

        :param value: `value` to discard from sorted list

        """
        _maxes = self._maxes

        if not _maxes:
            return

        pos = bisect_left(_maxes, value)

        if pos == len(_maxes):
            return

        _lists = self._lists
        idx = bisect_left(_lists[pos], value)

        if _lists[pos][idx] == value:
            self._delete(pos, idx)


    def remove(self, value):
        """Remove `value` from sorted list; `value` must be a member.

        If `value` is not a member, raise ValueError.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList([1, 2, 3, 4, 5])
        >>> sl.remove(5)
        >>> sl == [1, 2, 3, 4]
        True
        >>> sl.remove(0)
        Traceback (most recent call last):
          ...
        ValueError: 0 not in list

        :param value: `value` to remove from sorted list
        :raises ValueError: if `value` is not in sorted list

        """
        _maxes = self._maxes

        if not _maxes:
            raise ValueError('{0!r} not in list'.format(value))

        pos = bisect_left(_maxes, value)

        if pos == len(_maxes):
            raise ValueError('{0!r} not in list'.format(value))

        _lists = self._lists
        idx = bisect_left(_lists[pos], value)

        if _lists[pos][idx] == value:
            self._delete(pos, idx)
        else:
            raise ValueError('{0!r} not in list'.format(value))


    def _delete(self, pos, idx):
        """Delete value at the given `(pos, idx)`.

        Combines lists that are less than half the load level.

        Updates the index when the sublist length is more than half the load
        level. This requires decrementing the nodes in a traversal from the
        leaf node to the root. For an example traversal see
        ``SortedList._loc``.

        :param int pos: lists index
        :param int idx: sublist index

        """
        _lists = self._lists
        _maxes = self._maxes
        _index = self._index

        _lists_pos = _lists[pos]

        del _lists_pos[idx]
        self._len -= 1

        len_lists_pos = len(_lists_pos)

        if len_lists_pos > (self._load >> 1):
            _maxes[pos] = _lists_pos[-1]

            if _index:
                child = self._offset + pos
                while child > 0:
                    _index[child] -= 1
                    child = (child - 1) >> 1
                _index[0] -= 1
        elif len(_lists) > 1:
            if not pos:
                pos += 1

            prev = pos - 1
            _lists[prev].extend(_lists[pos])
            _maxes[prev] = _lists[prev][-1]

            del _lists[pos]
            del _maxes[pos]
            del _index[:]

            self._expand(prev)
        elif len_lists_pos:
            _maxes[pos] = _lists_pos[-1]
        else:
            del _lists[pos]
            del _maxes[pos]
            del _index[:]


    def _loc(self, pos, idx):
        """Convert an index pair (lists index, sublist index) into a single
        index number that corresponds to the position of the value in the
        sorted list.

        Many queries require the index be built. Details of the index are
        described in ``SortedList._build_index``.

        Indexing requires traversing the tree from a leaf node to the root. The
        parent of each node is easily computable at ``(pos - 1) // 2``.

        Left-child nodes are always at odd indices and right-child nodes are
        always at even indices.

        When traversing up from a right-child node, increment the total by the
        left-child node.

        The final index is the sum from traversal and the index in the sublist.

        For example, using the index from ``SortedList._build_index``::

            _index = 14 5 9 3 2 4 5
            _offset = 3

        Tree::

                 14
              5      9
            3   2  4   5

        Converting an index pair (2, 3) into a single index involves iterating
        like so:

        1. Starting at the leaf node: offset + alpha = 3 + 2 = 5. We identify
           the node as a left-child node. At such nodes, we simply traverse to
           the parent.

        2. At node 9, position 2, we recognize the node as a right-child node
           and accumulate the left-child in our total. Total is now 5 and we
           traverse to the parent at position 0.

        3. Iteration ends at the root.

        The index is then the sum of the total and sublist index: 5 + 3 = 8.

        :param int pos: lists index
        :param int idx: sublist index
        :return: index in sorted list

        """
        if not pos:
            return idx

        _index = self._index

        if not _index:
            self._build_index()

        total = 0

        # Increment pos to point in the index to len(self._lists[pos]).

        pos += self._offset

        # Iterate until reaching the root of the index tree at pos = 0.

        while pos:

            # Right-child nodes are at odd indices. At such indices
            # account the total below the left child node.

            if not pos & 1:
                total += _index[pos - 1]

            # Advance pos to the parent node.

            pos = (pos - 1) >> 1

        return total + idx


    def _pos(self, idx):
        """Convert an index into an index pair (lists index, sublist index)
        that can be used to access the corresponding lists position.

        Many queries require the index be built. Details of the index are
        described in ``SortedList._build_index``.

        Indexing requires traversing the tree to a leaf node. Each node has two
        children which are easily computable. Given an index, pos, the
        left-child is at ``pos * 2 + 1`` and the right-child is at ``pos * 2 +
        2``.

        When the index is less than the left-child, traversal moves to the
        left sub-tree. Otherwise, the index is decremented by the left-child
        and traversal moves to the right sub-tree.

        At a child node, the indexing pair is computed from the relative
        position of the child node as compared with the offset and the remaining
        index.

        For example, using the index from ``SortedList._build_index``::

            _index = 14 5 9 3 2 4 5
            _offset = 3

        Tree::

                 14
              5      9
            3   2  4   5

        Indexing position 8 involves iterating like so:

        1. Starting at the root, position 0, 8 is compared with the left-child
           node (5) which it is greater than. When greater the index is
           decremented and the position is updated to the right child node.

        2. At node 9 with index 3, we again compare the index to the left-child
           node with value 4. Because the index is the less than the left-child
           node, we simply traverse to the left.

        3. At node 4 with index 3, we recognize that we are at a leaf node and
           stop iterating.

        4. To compute the sublist index, we subtract the offset from the index
           of the leaf node: 5 - 3 = 2. To compute the index in the sublist, we
           simply use the index remaining from iteration. In this case, 3.

        The final index pair from our example is (2, 3) which corresponds to
        index 8 in the sorted list.

        :param int idx: index in sorted list
        :return: (lists index, sublist index) pair

        """
        if idx < 0:
            last_len = len(self._lists[-1])

            if (-idx) <= last_len:
                return len(self._lists) - 1, last_len + idx

            idx += self._len

            if idx < 0:
                raise IndexError('list index out of range')
        elif idx >= self._len:
            raise IndexError('list index out of range')

        if idx < len(self._lists[0]):
            return 0, idx

        _index = self._index

        if not _index:
            self._build_index()

        pos = 0
        child = 1
        len_index = len(_index)

        while child < len_index:
            index_child = _index[child]

            if idx < index_child:
                pos = child
            else:
                idx -= index_child
                pos = child + 1

            child = (pos << 1) + 1

        return (pos - self._offset, idx)


    def _build_index(self):
        """Build a positional index for indexing the sorted list.

        Indexes are represented as binary trees in a dense array notation
        similar to a binary heap.

        For example, given a lists representation storing integers::

            0: [1, 2, 3]
            1: [4, 5]
            2: [6, 7, 8, 9]
            3: [10, 11, 12, 13, 14]

        The first transformation maps the sub-lists by their length. The
        first row of the index is the length of the sub-lists::

            0: [3, 2, 4, 5]

        Each row after that is the sum of consecutive pairs of the previous
        row::

            1: [5, 9]
            2: [14]

        Finally, the index is built by concatenating these lists together::

            _index = [14, 5, 9, 3, 2, 4, 5]

        An offset storing the start of the first row is also stored::

            _offset = 3

        When built, the index can be used for efficient indexing into the list.
        See the comment and notes on ``SortedList._pos`` for details.

        """
        row0 = list(map(len, self._lists))

        if len(row0) == 1:
            self._index[:] = row0
            self._offset = 0
            return

        head = iter(row0)
        tail = iter(head)
        row1 = list(starmap(add, zip(head, tail)))

        if len(row0) & 1:
            row1.append(row0[-1])

        if len(row1) == 1:
            self._index[:] = row1 + row0
            self._offset = 1
            return

        size = 2 ** (int(log(len(row1) - 1, 2)) + 1)
        row1.extend(repeat(0, size - len(row1)))
        tree = [row0, row1]

        while len(tree[-1]) > 1:
            head = iter(tree[-1])
            tail = iter(head)
            row = list(starmap(add, zip(head, tail)))
            tree.append(row)

        reduce(iadd, reversed(tree), self._index)
        self._offset = size * 2 - 1


    def __delitem__(self, index):
        """Remove value at `index` from sorted list.

        ``sl.__delitem__(index)`` <==> ``del sl[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList('abcde')
        >>> del sl[2]
        >>> sl
        SortedList(['a', 'b', 'd', 'e'])
        >>> del sl[:2]
        >>> sl
        SortedList(['d', 'e'])

        :param index: integer or slice for indexing
        :raises IndexError: if index out of range

        """
        if isinstance(index, slice):
            start, stop, step = index.indices(self._len)

            if step == 1 and start < stop:
                if start == 0 and stop == self._len:
                    return self._clear()
                elif self._len <= 8 * (stop - start):
                    values = self._getitem(slice(None, start))
                    if stop < self._len:
                        values += self._getitem(slice(stop, None))
                    self._clear()
                    return self._update(values)

            indices = range(start, stop, step)

            # Delete items from greatest index to least so
            # that the indices remain valid throughout iteration.

            if step > 0:
                indices = reversed(indices)

            _pos, _delete = self._pos, self._delete

            for index in indices:
                pos, idx = _pos(index)
                _delete(pos, idx)
        else:
            pos, idx = self._pos(index)
            self._delete(pos, idx)


    def __getitem__(self, index):
        """Lookup value at `index` in sorted list.

        ``sl.__getitem__(index)`` <==> ``sl[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList('abcde')
        >>> sl[1]
        'b'
        >>> sl[-1]
        'e'
        >>> sl[2:5]
        ['c', 'd', 'e']

        :param index: integer or slice for indexing
        :return: value or list of values
        :raises IndexError: if index out of range

        """
        _lists = self._lists

        if isinstance(index, slice):
            start, stop, step = index.indices(self._len)

            if step == 1 and start < stop:
                # Whole slice optimization: start to stop slices the whole
                # sorted list.

                if start == 0 and stop == self._len:
                    return reduce(iadd, self._lists, [])

                start_pos, start_idx = self._pos(start)
                start_list = _lists[start_pos]
                stop_idx = start_idx + stop - start

                # Small slice optimization: start index and stop index are
                # within the start list.

                if len(start_list) >= stop_idx:
                    return start_list[start_idx:stop_idx]

                if stop == self._len:
                    stop_pos = len(_lists) - 1
                    stop_idx = len(_lists[stop_pos])
                else:
                    stop_pos, stop_idx = self._pos(stop)

                prefix = _lists[start_pos][start_idx:]
                middle = _lists[(start_pos + 1):stop_pos]
                result = reduce(iadd, middle, prefix)
                result += _lists[stop_pos][:stop_idx]

                return result

            if step == -1 and start > stop:
                result = self._getitem(slice(stop + 1, start + 1))
                result.reverse()
                return result

            # Return a list because a negative step could
            # reverse the order of the items and this could
            # be the desired behavior.

            indices = range(start, stop, step)
            return list(self._getitem(index) for index in indices)
        else:
            if self._len:
                if index == 0:
                    return _lists[0][0]
                elif index == -1:
                    return _lists[-1][-1]
            else:
                raise IndexError('list index out of range')

            if 0 <= index < len(_lists[0]):
                return _lists[0][index]

            len_last = len(_lists[-1])

            if -len_last < index < 0:
                return _lists[-1][len_last + index]

            pos, idx = self._pos(index)
            return _lists[pos][idx]

    _getitem = __getitem__


    def __setitem__(self, index, value):
        """Raise not-implemented error.

        ``sl.__setitem__(index, value)`` <==> ``sl[index] = value``

        :raises NotImplementedError: use ``del sl[index]`` and
            ``sl.add(value)`` instead

        """
        message = 'use ``del sl[index]`` and ``sl.add(value)`` instead'
        raise NotImplementedError(message)


    def __iter__(self):
        """Return an iterator over the sorted list.

        ``sl.__iter__()`` <==> ``iter(sl)``

        Iterating the sorted list while adding or deleting values may raise a
        :exc:`RuntimeError` or fail to iterate over all values.

        """
        return chain.from_iterable(self._lists)


    def __reversed__(self):
        """Return a reverse iterator over the sorted list.

        ``sl.__reversed__()`` <==> ``reversed(sl)``

        Iterating the sorted list while adding or deleting values may raise a
        :exc:`RuntimeError` or fail to iterate over all values.

        """
        return chain.from_iterable(map(reversed, reversed(self._lists)))


    def reverse(self):
        """Raise not-implemented error.

        Sorted list maintains values in ascending sort order. Values may not be
        reversed in-place.

        Use ``reversed(sl)`` for an iterator over values in descending sort
        order.

        Implemented to override `MutableSequence.reverse` which provides an
        erroneous default implementation.

        :raises NotImplementedError: use ``reversed(sl)`` instead

        """
        raise NotImplementedError('use ``reversed(sl)`` instead')


    def islice(self, start=None, stop=None, reverse=False):
        """Return an iterator that slices sorted list from `start` to `stop`.

        The `start` and `stop` index are treated inclusive and exclusive,
        respectively.

        Both `start` and `stop` default to `None` which is automatically
        inclusive of the beginning and end of the sorted list.

        When `reverse` is `True` the values are yielded from the iterator in
        reverse order; `reverse` defaults to `False`.

        >>> sl = SortedList('abcdefghij')
        >>> it = sl.islice(2, 6)
        >>> list(it)
        ['c', 'd', 'e', 'f']

        :param int start: start index (inclusive)
        :param int stop: stop index (exclusive)
        :param bool reverse: yield values in reverse order
        :return: iterator

        """
        _len = self._len

        if not _len:
            return iter(())

        start, stop, _ = slice(start, stop).indices(self._len)

        if start >= stop:
            return iter(())

        _pos = self._pos

        min_pos, min_idx = _pos(start)

        if stop == _len:
            max_pos = len(self._lists) - 1
            max_idx = len(self._lists[-1])
        else:
            max_pos, max_idx = _pos(stop)

        return self._islice(min_pos, min_idx, max_pos, max_idx, reverse)


    def _islice(self, min_pos, min_idx, max_pos, max_idx, reverse):
        """Return an iterator that slices sorted list using two index pairs.

        The index pairs are (min_pos, min_idx) and (max_pos, max_idx), the
        first inclusive and the latter exclusive. See `_pos` for details on how
        an index is converted to an index pair.

        When `reverse` is `True`, values are yielded from the iterator in
        reverse order.

        """
        _lists = self._lists

        if min_pos > max_pos:
            return iter(())

        if min_pos == max_pos:
            if reverse:
                indices = reversed(range(min_idx, max_idx))
                return map(_lists[min_pos].__getitem__, indices)

            indices = range(min_idx, max_idx)
            return map(_lists[min_pos].__getitem__, indices)

        next_pos = min_pos + 1

        if next_pos == max_pos:
            if reverse:
                min_indices = range(min_idx, len(_lists[min_pos]))
                max_indices = range(max_idx)
                return chain(
                    map(_lists[max_pos].__getitem__, reversed(max_indices)),
                    map(_lists[min_pos].__getitem__, reversed(min_indices)),
                )

            min_indices = range(min_idx, len(_lists[min_pos]))
            max_indices = range(max_idx)
            return chain(
                map(_lists[min_pos].__getitem__, min_indices),
                map(_lists[max_pos].__getitem__, max_indices),
            )

        if reverse:
            min_indices = range(min_idx, len(_lists[min_pos]))
            sublist_indices = range(next_pos, max_pos)
            sublists = map(_lists.__getitem__, reversed(sublist_indices))
            max_indices = range(max_idx)
            return chain(
                map(_lists[max_pos].__getitem__, reversed(max_indices)),
                chain.from_iterable(map(reversed, sublists)),
                map(_lists[min_pos].__getitem__, reversed(min_indices)),
            )

        min_indices = range(min_idx, len(_lists[min_pos]))
        sublist_indices = range(next_pos, max_pos)
        sublists = map(_lists.__getitem__, sublist_indices)
        max_indices = range(max_idx)
        return chain(
            map(_lists[min_pos].__getitem__, min_indices),
            chain.from_iterable(sublists),
            map(_lists[max_pos].__getitem__, max_indices),
        )


    def irange(self, minimum=None, maximum=None, inclusive=(True, True),
               reverse=False):
        """Create an iterator of values between `minimum` and `maximum`.

        Both `minimum` and `maximum` default to `None` which is automatically
        inclusive of the beginning and end of the sorted list.

        The argument `inclusive` is a pair of booleans that indicates whether
        the minimum and maximum ought to be included in the range,
        respectively. The default is ``(True, True)`` such that the range is
        inclusive of both minimum and maximum.

        When `reverse` is `True` the values are yielded from the iterator in
        reverse order; `reverse` defaults to `False`.

        >>> sl = SortedList('abcdefghij')
        >>> it = sl.irange('c', 'f')
        >>> list(it)
        ['c', 'd', 'e', 'f']

        :param minimum: minimum value to start iterating
        :param maximum: maximum value to stop iterating
        :param inclusive: pair of booleans
        :param bool reverse: yield values in reverse order
        :return: iterator

        """
        _maxes = self._maxes

        if not _maxes:
            return iter(())

        _lists = self._lists

        # Calculate the minimum (pos, idx) pair. By default this location
        # will be inclusive in our calculation.

        if minimum is None:
            min_pos = 0
            min_idx = 0
        else:
            if inclusive[0]:
                min_pos = bisect_left(_maxes, minimum)

                if min_pos == len(_maxes):
                    return iter(())

                min_idx = bisect_left(_lists[min_pos], minimum)
            else:
                min_pos = bisect_right(_maxes, minimum)

                if min_pos == len(_maxes):
                    return iter(())

                min_idx = bisect_right(_lists[min_pos], minimum)

        # Calculate the maximum (pos, idx) pair. By default this location
        # will be exclusive in our calculation.

        if maximum is None:
            max_pos = len(_maxes) - 1
            max_idx = len(_lists[max_pos])
        else:
            if inclusive[1]:
                max_pos = bisect_right(_maxes, maximum)

                if max_pos == len(_maxes):
                    max_pos -= 1
                    max_idx = len(_lists[max_pos])
                else:
                    max_idx = bisect_right(_lists[max_pos], maximum)
            else:
                max_pos = bisect_left(_maxes, maximum)

                if max_pos == len(_maxes):
                    max_pos -= 1
                    max_idx = len(_lists[max_pos])
                else:
                    max_idx = bisect_left(_lists[max_pos], maximum)

        return self._islice(min_pos, min_idx, max_pos, max_idx, reverse)


    def __len__(self):
        """Return the size of the sorted list.

        ``sl.__len__()`` <==> ``len(sl)``

        :return: size of sorted list

        """
        return self._len


    def bisect_left(self, value):
        """Return an index to insert `value` in the sorted list.

        If the `value` is already present, the insertion point will be before
        (to the left of) any existing values.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList([10, 11, 12, 13, 14])
        >>> sl.bisect_left(12)
        2

        :param value: insertion index of value in sorted list
        :return: index

        """
        _maxes = self._maxes

        if not _maxes:
            return 0

        pos = bisect_left(_maxes, value)

        if pos == len(_maxes):
            return self._len

        idx = bisect_left(self._lists[pos], value)
        return self._loc(pos, idx)


    def bisect_right(self, value):
        """Return an index to insert `value` in the sorted list.

        Similar to `bisect_left`, but if `value` is already present, the
        insertion point will be after (to the right of) any existing values.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList([10, 11, 12, 13, 14])
        >>> sl.bisect_right(12)
        3

        :param value: insertion index of value in sorted list
        :return: index

        """
        _maxes = self._maxes

        if not _maxes:
            return 0

        pos = bisect_right(_maxes, value)

        if pos == len(_maxes):
            return self._len

        idx = bisect_right(self._lists[pos], value)
        return self._loc(pos, idx)

    bisect = bisect_right
    _bisect_right = bisect_right


    def count(self, value):
        """Return number of occurrences of `value` in the sorted list.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        >>> sl.count(3)
        3

        :param value: value to count in sorted list
        :return: count

        """
        _maxes = self._maxes

        if not _maxes:
            return 0

        pos_left = bisect_left(_maxes, value)

        if pos_left == len(_maxes):
            return 0

        _lists = self._lists
        idx_left = bisect_left(_lists[pos_left], value)
        pos_right = bisect_right(_maxes, value)

        if pos_right == len(_maxes):
            return self._len - self._loc(pos_left, idx_left)

        idx_right = bisect_right(_lists[pos_right], value)

        if pos_left == pos_right:
            return idx_right - idx_left

        right = self._loc(pos_right, idx_right)
        left = self._loc(pos_left, idx_left)
        return right - left


    def copy(self):
        """Return a shallow copy of the sorted list.

        Runtime complexity: `O(n)`

        :return: new sorted list

        """
        return self.__class__(self)

    __copy__ = copy


    def append(self, value):
        """Raise not-implemented error.

        Implemented to override `MutableSequence.append` which provides an
        erroneous default implementation.

        :raises NotImplementedError: use ``sl.add(value)`` instead

        """
        raise NotImplementedError('use ``sl.add(value)`` instead')


    def extend(self, values):
        """Raise not-implemented error.

        Implemented to override `MutableSequence.extend` which provides an
        erroneous default implementation.

        :raises NotImplementedError: use ``sl.update(values)`` instead

        """
        raise NotImplementedError('use ``sl.update(values)`` instead')


    def insert(self, index, value):
        """Raise not-implemented error.

        :raises NotImplementedError: use ``sl.add(value)`` instead

        """
        raise NotImplementedError('use ``sl.add(value)`` instead')


    def pop(self, index=-1):
        """Remove and return value at `index` in sorted list.

        Raise :exc:`IndexError` if the sorted list is empty or index is out of
        range.

        Negative indices are supported.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList('abcde')
        >>> sl.pop()
        'e'
        >>> sl.pop(2)
        'c'
        >>> sl
        SortedList(['a', 'b', 'd'])

        :param int index: index of value (default -1)
        :return: value
        :raises IndexError: if index is out of range

        """
        if not self._len:
            raise IndexError('pop index out of range')

        _lists = self._lists

        if index == 0:
            val = _lists[0][0]
            self._delete(0, 0)
            return val

        if index == -1:
            pos = len(_lists) - 1
            loc = len(_lists[pos]) - 1
            val = _lists[pos][loc]
            self._delete(pos, loc)
            return val

        if 0 <= index < len(_lists[0]):
            val = _lists[0][index]
            self._delete(0, index)
            return val

        len_last = len(_lists[-1])

        if -len_last < index < 0:
            pos = len(_lists) - 1
            loc = len_last + index
            val = _lists[pos][loc]
            self._delete(pos, loc)
            return val

        pos, idx = self._pos(index)
        val = _lists[pos][idx]
        self._delete(pos, idx)
        return val


    def index(self, value, start=None, stop=None):
        """Return first index of value in sorted list.

        Raise ValueError if `value` is not present.

        Index must be between `start` and `stop` for the `value` to be
        considered present. The default value, None, for `start` and `stop`
        indicate the beginning and end of the sorted list.

        Negative indices are supported.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList('abcde')
        >>> sl.index('d')
        3
        >>> sl.index('z')
        Traceback (most recent call last):
          ...
        ValueError: 'z' is not in list

        :param value: value in sorted list
        :param int start: start index (default None, start of sorted list)
        :param int stop: stop index (default None, end of sorted list)
        :return: index of value
        :raises ValueError: if value is not present

        """
        _len = self._len

        if not _len:
            raise ValueError('{0!r} is not in list'.format(value))

        if start is None:
            start = 0
        if start < 0:
            start += _len
        if start < 0:
            start = 0

        if stop is None:
            stop = _len
        if stop < 0:
            stop += _len
        if stop > _len:
            stop = _len

        if stop <= start:
            raise ValueError('{0!r} is not in list'.format(value))

        _maxes = self._maxes
        pos_left = bisect_left(_maxes, value)

        if pos_left == len(_maxes):
            raise ValueError('{0!r} is not in list'.format(value))

        _lists = self._lists
        idx_left = bisect_left(_lists[pos_left], value)

        if _lists[pos_left][idx_left] != value:
            raise ValueError('{0!r} is not in list'.format(value))

        stop -= 1
        left = self._loc(pos_left, idx_left)

        if start <= left:
            if left <= stop:
                return left
        else:
            right = self._bisect_right(value) - 1

            if start <= right:
                return start

        raise ValueError('{0!r} is not in list'.format(value))


    def __add__(self, other):
        """Return new sorted list containing all values in both sequences.

        ``sl.__add__(other)`` <==> ``sl + other``

        Values in `other` do not need to be in sorted order.

        Runtime complexity: `O(n*log(n))`

        >>> sl1 = SortedList('bat')
        >>> sl2 = SortedList('cat')
        >>> sl1 + sl2
        SortedList(['a', 'a', 'b', 'c', 't', 't'])

        :param other: other iterable
        :return: new sorted list

        """
        values = reduce(iadd, self._lists, [])
        values.extend(other)
        return self.__class__(values)

    __radd__ = __add__


    def __iadd__(self, other):
        """Update sorted list with values from `other`.

        ``sl.__iadd__(other)`` <==> ``sl += other``

        Values in `other` do not need to be in sorted order.

        Runtime complexity: `O(k*log(n))` -- approximate.

        >>> sl = SortedList('bat')
        >>> sl += 'cat'
        >>> sl
        SortedList(['a', 'a', 'b', 'c', 't', 't'])

        :param other: other iterable
        :return: existing sorted list

        """
        self._update(other)
        return self


    def __mul__(self, num):
        """Return new sorted list with `num` shallow copies of values.

        ``sl.__mul__(num)`` <==> ``sl * num``

        Runtime complexity: `O(n*log(n))`

        >>> sl = SortedList('abc')
        >>> sl * 3
        SortedList(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'])

        :param int num: count of shallow copies
        :return: new sorted list

        """
        values = reduce(iadd, self._lists, []) * num
        return self.__class__(values)

    __rmul__ = __mul__


    def __imul__(self, num):
        """Update the sorted list with `num` shallow copies of values.

        ``sl.__imul__(num)`` <==> ``sl *= num``

        Runtime complexity: `O(n*log(n))`

        >>> sl = SortedList('abc')
        >>> sl *= 3
        >>> sl
        SortedList(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'])

        :param int num: count of shallow copies
        :return: existing sorted list

        """
        values = reduce(iadd, self._lists, []) * num
        self._clear()
        self._update(values)
        return self


    def __make_cmp(seq_op, symbol, doc):
        "Make comparator method."
        def comparer(self, other):
            "Compare method for sorted list and sequence."
            if not isinstance(other, Sequence):
                return NotImplemented

            self_len = self._len
            len_other = len(other)

            if self_len != len_other:
                if seq_op is eq:
                    return False
                if seq_op is ne:
                    return True

            for alpha, beta in zip(self, other):
                if alpha != beta:
                    return seq_op(alpha, beta)

            return seq_op(self_len, len_other)

        seq_op_name = seq_op.__name__
        comparer.__name__ = '__{0}__'.format(seq_op_name)
        doc_str = """Return true if and only if sorted list is {0} `other`.

        ``sl.__{1}__(other)`` <==> ``sl {2} other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        :param other: `other` sequence
        :return: true if sorted list is {0} `other`

        """
        comparer.__doc__ = dedent(doc_str.format(doc, seq_op_name, symbol))
        return comparer


    __eq__ = __make_cmp(eq, '==', 'equal to')
    __ne__ = __make_cmp(ne, '!=', 'not equal to')
    __lt__ = __make_cmp(lt, '<', 'less than')
    __gt__ = __make_cmp(gt, '>', 'greater than')
    __le__ = __make_cmp(le, '<=', 'less than or equal to')
    __ge__ = __make_cmp(ge, '>=', 'greater than or equal to')
    __make_cmp = staticmethod(__make_cmp)


    def __reduce__(self):
        values = reduce(iadd, self._lists, [])
        return (type(self), (values,))


    @recursive_repr()
    def __repr__(self):
        """Return string representation of sorted list.

        ``sl.__repr__()`` <==> ``repr(sl)``

        :return: string representation

        """
        return '{0}({1!r})'.format(type(self).__name__, list(self))


    def _check(self):
        """Check invariants of sorted list.

        Runtime complexity: `O(n)`

        """
        try:
            assert self._load >= 4
            assert len(self._maxes) == len(self._lists)
            assert self._len == sum(len(sublist) for sublist in self._lists)

            # Check all sublists are sorted.

            for sublist in self._lists:
                for pos in range(1, len(sublist)):
                    assert sublist[pos - 1] <= sublist[pos]

            # Check beginning/end of sublists are sorted.

            for pos in range(1, len(self._lists)):
                assert self._lists[pos - 1][-1] <= self._lists[pos][0]

            # Check _maxes index is the last value of each sublist.

            for pos in range(len(self._maxes)):
                assert self._maxes[pos] == self._lists[pos][-1]

            # Check sublist lengths are less than double load-factor.

            double = self._load << 1
            assert all(len(sublist) <= double for sublist in self._lists)

            # Check sublist lengths are greater than half load-factor for all
            # but the last sublist.

            half = self._load >> 1
            for pos in range(0, len(self._lists) - 1):
                assert len(self._lists[pos]) >= half

            if self._index:
                assert self._len == self._index[0]
                assert len(self._index) == self._offset + len(self._lists)

                # Check index leaf nodes equal length of sublists.

                for pos in range(len(self._lists)):
                    leaf = self._index[self._offset + pos]
                    assert leaf == len(self._lists[pos])

                # Check index branch nodes are the sum of their children.

                for pos in range(self._offset):
                    child = (pos << 1) + 1
                    if child >= len(self._index):
                        assert self._index[pos] == 0
                    elif child + 1 == len(self._index):
                        assert self._index[pos] == self._index[child]
                    else:
                        child_sum = self._index[child] + self._index[child + 1]
                        assert child_sum == self._index[pos]
        except:
            traceback.print_exc(file=sys.stdout)
            print('len', self._len)
            print('load', self._load)
            print('offset', self._offset)
            print('len_index', len(self._index))
            print('index', self._index)
            print('len_maxes', len(self._maxes))
            print('maxes', self._maxes)
            print('len_lists', len(self._lists))
            print('lists', self._lists)
            raise


def identity(value):
    "Identity function."
    return value


class SortedKeyList(SortedList):
    """Sorted-key list is a subtype of sorted list.

    The sorted-key list maintains values in comparison order based on the
    result of a key function applied to every value.

    All the same methods that are available in :class:`SortedList` are also
    available in :class:`SortedKeyList`.

    Additional methods provided:

    * :attr:`SortedKeyList.key`
    * :func:`SortedKeyList.bisect_key_left`
    * :func:`SortedKeyList.bisect_key_right`
    * :func:`SortedKeyList.irange_key`

    Some examples below use:

    >>> from operator import neg
    >>> neg
    <built-in function neg>
    >>> neg(1)
    -1

    """
    def __init__(self, iterable=None, key=identity):
        """Initialize sorted-key list instance.

        Optional `iterable` argument provides an initial iterable of values to
        initialize the sorted-key list.

        Optional `key` argument defines a callable that, like the `key`
        argument to Python's `sorted` function, extracts a comparison key from
        each value. The default is the identity function.

        Runtime complexity: `O(n*log(n))`

        >>> from operator import neg
        >>> skl = SortedKeyList(key=neg)
        >>> skl
        SortedKeyList([], key=<built-in function neg>)
        >>> skl = SortedKeyList([3, 1, 2], key=neg)
        >>> skl
        SortedKeyList([3, 2, 1], key=<built-in function neg>)

        :param iterable: initial values (optional)
        :param key: function used to extract comparison key (optional)

        """
        self._key = key
        self._len = 0
        self._load = self.DEFAULT_LOAD_FACTOR
        self._lists = []
        self._keys = []
        self._maxes = []
        self._index = []
        self._offset = 0

        if iterable is not None:
            self._update(iterable)


    def __new__(cls, iterable=None, key=identity):
        return object.__new__(cls)


    @property
    def key(self):
        "Function used to extract comparison key from values."
        return self._key


    def clear(self):
        """Remove all values from sorted-key list.

        Runtime complexity: `O(n)`

        """
        self._len = 0
        del self._lists[:]
        del self._keys[:]
        del self._maxes[:]
        del self._index[:]

    _clear = clear


    def add(self, value):
        """Add `value` to sorted-key list.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList(key=neg)
        >>> skl.add(3)
        >>> skl.add(1)
        >>> skl.add(2)
        >>> skl
        SortedKeyList([3, 2, 1], key=<built-in function neg>)

        :param value: value to add to sorted-key list

        """
        _lists = self._lists
        _keys = self._keys
        _maxes = self._maxes

        key = self._key(value)

        if _maxes:
            pos = bisect_right(_maxes, key)

            if pos == len(_maxes):
                pos -= 1
                _lists[pos].append(value)
                _keys[pos].append(key)
                _maxes[pos] = key
            else:
                idx = bisect_right(_keys[pos], key)
                _lists[pos].insert(idx, value)
                _keys[pos].insert(idx, key)

            self._expand(pos)
        else:
            _lists.append([value])
            _keys.append([key])
            _maxes.append(key)

        self._len += 1


    def _expand(self, pos):
        """Split sublists with length greater than double the load-factor.

        Updates the index when the sublist length is less than double the load
        level. This requires incrementing the nodes in a traversal from the
        leaf node to the root. For an example traversal see
        ``SortedList._loc``.

        """
        _lists = self._lists
        _keys = self._keys
        _index = self._index

        if len(_keys[pos]) > (self._load << 1):
            _maxes = self._maxes
            _load = self._load

            _lists_pos = _lists[pos]
            _keys_pos = _keys[pos]
            half = _lists_pos[_load:]
            half_keys = _keys_pos[_load:]
            del _lists_pos[_load:]
            del _keys_pos[_load:]
            _maxes[pos] = _keys_pos[-1]

            _lists.insert(pos + 1, half)
            _keys.insert(pos + 1, half_keys)
            _maxes.insert(pos + 1, half_keys[-1])

            del _index[:]
        else:
            if _index:
                child = self._offset + pos
                while child:
                    _index[child] += 1
                    child = (child - 1) >> 1
                _index[0] += 1


    def update(self, iterable):
        """Update sorted-key list by adding all values from `iterable`.

        Runtime complexity: `O(k*log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList(key=neg)
        >>> skl.update([3, 1, 2])
        >>> skl
        SortedKeyList([3, 2, 1], key=<built-in function neg>)

        :param iterable: iterable of values to add

        """
        _lists = self._lists
        _keys = self._keys
        _maxes = self._maxes
        values = sorted(iterable, key=self._key)

        if _maxes:
            if len(values) * 4 >= self._len:
                _lists.append(values)
                values = reduce(iadd, _lists, [])
                values.sort(key=self._key)
                self._clear()
            else:
                _add = self.add
                for val in values:
                    _add(val)
                return

        _load = self._load
        _lists.extend(values[pos:(pos + _load)]
                      for pos in range(0, len(values), _load))
        _keys.extend(list(map(self._key, _list)) for _list in _lists)
        _maxes.extend(sublist[-1] for sublist in _keys)
        self._len = len(values)
        del self._index[:]

    _update = update


    def __contains__(self, value):
        """Return true if `value` is an element of the sorted-key list.

        ``skl.__contains__(value)`` <==> ``value in skl``

        Runtime complexity: `O(log(n))`

        >>> from operator import neg
        >>> skl = SortedKeyList([1, 2, 3, 4, 5], key=neg)
        >>> 3 in skl
        True

        :param value: search for value in sorted-key list
        :return: true if `value` in sorted-key list

        """
        _maxes = self._maxes

        if not _maxes:
            return False

        key = self._key(value)
        pos = bisect_left(_maxes, key)

        if pos == len(_maxes):
            return False

        _lists = self._lists
        _keys = self._keys

        idx = bisect_left(_keys[pos], key)

        len_keys = len(_keys)
        len_sublist = len(_keys[pos])

        while True:
            if _keys[pos][idx] != key:
                return False
            if _lists[pos][idx] == value:
                return True
            idx += 1
            if idx == len_sublist:
                pos += 1
                if pos == len_keys:
                    return False
                len_sublist = len(_keys[pos])
                idx = 0


    def discard(self, value):
        """Remove `value` from sorted-key list if it is a member.

        If `value` is not a member, do nothing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        >>> skl.discard(1)
        >>> skl.discard(0)
        >>> skl == [5, 4, 3, 2]
        True

        :param value: `value` to discard from sorted-key list

        """
        _maxes = self._maxes

        if not _maxes:
            return

        key = self._key(value)
        pos = bisect_left(_maxes, key)

        if pos == len(_maxes):
            return

        _lists = self._lists
        _keys = self._keys
        idx = bisect_left(_keys[pos], key)
        len_keys = len(_keys)
        len_sublist = len(_keys[pos])

        while True:
            if _keys[pos][idx] != key:
                return
            if _lists[pos][idx] == value:
                self._delete(pos, idx)
                return
            idx += 1
            if idx == len_sublist:
                pos += 1
                if pos == len_keys:
                    return
                len_sublist = len(_keys[pos])
                idx = 0


    def remove(self, value):
        """Remove `value` from sorted-key list; `value` must be a member.

        If `value` is not a member, raise ValueError.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList([1, 2, 3, 4, 5], key=neg)
        >>> skl.remove(5)
        >>> skl == [4, 3, 2, 1]
        True
        >>> skl.remove(0)
        Traceback (most recent call last):
          ...
        ValueError: 0 not in list

        :param value: `value` to remove from sorted-key list
        :raises ValueError: if `value` is not in sorted-key list

        """
        _maxes = self._maxes

        if not _maxes:
            raise ValueError('{0!r} not in list'.format(value))

        key = self._key(value)
        pos = bisect_left(_maxes, key)

        if pos == len(_maxes):
            raise ValueError('{0!r} not in list'.format(value))

        _lists = self._lists
        _keys = self._keys
        idx = bisect_left(_keys[pos], key)
        len_keys = len(_keys)
        len_sublist = len(_keys[pos])

        while True:
            if _keys[pos][idx] != key:
                raise ValueError('{0!r} not in list'.format(value))
            if _lists[pos][idx] == value:
                self._delete(pos, idx)
                return
            idx += 1
            if idx == len_sublist:
                pos += 1
                if pos == len_keys:
                    raise ValueError('{0!r} not in list'.format(value))
                len_sublist = len(_keys[pos])
                idx = 0


    def _delete(self, pos, idx):
        """Delete value at the given `(pos, idx)`.

        Combines lists that are less than half the load level.

        Updates the index when the sublist length is more than half the load
        level. This requires decrementing the nodes in a traversal from the
        leaf node to the root. For an example traversal see
        ``SortedList._loc``.

        :param int pos: lists index
        :param int idx: sublist index

        """
        _lists = self._lists
        _keys = self._keys
        _maxes = self._maxes
        _index = self._index
        keys_pos = _keys[pos]
        lists_pos = _lists[pos]

        del keys_pos[idx]
        del lists_pos[idx]
        self._len -= 1

        len_keys_pos = len(keys_pos)

        if len_keys_pos > (self._load >> 1):
            _maxes[pos] = keys_pos[-1]

            if _index:
                child = self._offset + pos
                while child > 0:
                    _index[child] -= 1
                    child = (child - 1) >> 1
                _index[0] -= 1
        elif len(_keys) > 1:
            if not pos:
                pos += 1

            prev = pos - 1
            _keys[prev].extend(_keys[pos])
            _lists[prev].extend(_lists[pos])
            _maxes[prev] = _keys[prev][-1]

            del _lists[pos]
            del _keys[pos]
            del _maxes[pos]
            del _index[:]

            self._expand(prev)
        elif len_keys_pos:
            _maxes[pos] = keys_pos[-1]
        else:
            del _lists[pos]
            del _keys[pos]
            del _maxes[pos]
            del _index[:]


    def irange(self, minimum=None, maximum=None, inclusive=(True, True),
               reverse=False):
        """Create an iterator of values between `minimum` and `maximum`.

        Both `minimum` and `maximum` default to `None` which is automatically
        inclusive of the beginning and end of the sorted-key list.

        The argument `inclusive` is a pair of booleans that indicates whether
        the minimum and maximum ought to be included in the range,
        respectively. The default is ``(True, True)`` such that the range is
        inclusive of both minimum and maximum.

        When `reverse` is `True` the values are yielded from the iterator in
        reverse order; `reverse` defaults to `False`.

        >>> from operator import neg
        >>> skl = SortedKeyList([11, 12, 13, 14, 15], key=neg)
        >>> it = skl.irange(14.5, 11.5)
        >>> list(it)
        [14, 13, 12]

        :param minimum: minimum value to start iterating
        :param maximum: maximum value to stop iterating
        :param inclusive: pair of booleans
        :param bool reverse: yield values in reverse order
        :return: iterator

        """
        min_key = self._key(minimum) if minimum is not None else None
        max_key = self._key(maximum) if maximum is not None else None
        return self._irange_key(
            min_key=min_key, max_key=max_key,
            inclusive=inclusive, reverse=reverse,
        )


    def irange_key(self, min_key=None, max_key=None, inclusive=(True, True),
                   reverse=False):
        """Create an iterator of values between `min_key` and `max_key`.

        Both `min_key` and `max_key` default to `None` which is automatically
        inclusive of the beginning and end of the sorted-key list.

        The argument `inclusive` is a pair of booleans that indicates whether
        the minimum and maximum ought to be included in the range,
        respectively. The default is ``(True, True)`` such that the range is
        inclusive of both minimum and maximum.

        When `reverse` is `True` the values are yielded from the iterator in
        reverse order; `reverse` defaults to `False`.

        >>> from operator import neg
        >>> skl = SortedKeyList([11, 12, 13, 14, 15], key=neg)
        >>> it = skl.irange_key(-14, -12)
        >>> list(it)
        [14, 13, 12]

        :param min_key: minimum key to start iterating
        :param max_key: maximum key to stop iterating
        :param inclusive: pair of booleans
        :param bool reverse: yield values in reverse order
        :return: iterator

        """
        _maxes = self._maxes

        if not _maxes:
            return iter(())

        _keys = self._keys

        # Calculate the minimum (pos, idx) pair. By default this location
        # will be inclusive in our calculation.

        if min_key is None:
            min_pos = 0
            min_idx = 0
        else:
            if inclusive[0]:
                min_pos = bisect_left(_maxes, min_key)

                if min_pos == len(_maxes):
                    return iter(())

                min_idx = bisect_left(_keys[min_pos], min_key)
            else:
                min_pos = bisect_right(_maxes, min_key)

                if min_pos == len(_maxes):
                    return iter(())

                min_idx = bisect_right(_keys[min_pos], min_key)

        # Calculate the maximum (pos, idx) pair. By default this location
        # will be exclusive in our calculation.

        if max_key is None:
            max_pos = len(_maxes) - 1
            max_idx = len(_keys[max_pos])
        else:
            if inclusive[1]:
                max_pos = bisect_right(_maxes, max_key)

                if max_pos == len(_maxes):
                    max_pos -= 1
                    max_idx = len(_keys[max_pos])
                else:
                    max_idx = bisect_right(_keys[max_pos], max_key)
            else:
                max_pos = bisect_left(_maxes, max_key)

                if max_pos == len(_maxes):
                    max_pos -= 1
                    max_idx = len(_keys[max_pos])
                else:
                    max_idx = bisect_left(_keys[max_pos], max_key)

        return self._islice(min_pos, min_idx, max_pos, max_idx, reverse)

    _irange_key = irange_key


    def bisect_left(self, value):
        """Return an index to insert `value` in the sorted-key list.

        If the `value` is already present, the insertion point will be before
        (to the left of) any existing values.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        >>> skl.bisect_left(1)
        4

        :param value: insertion index of value in sorted-key list
        :return: index

        """
        return self._bisect_key_left(self._key(value))


    def bisect_right(self, value):
        """Return an index to insert `value` in the sorted-key list.

        Similar to `bisect_left`, but if `value` is already present, the
        insertion point will be after (to the right of) any existing values.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedList([5, 4, 3, 2, 1], key=neg)
        >>> skl.bisect_right(1)
        5

        :param value: insertion index of value in sorted-key list
        :return: index

        """
        return self._bisect_key_right(self._key(value))

    bisect = bisect_right


    def bisect_key_left(self, key):
        """Return an index to insert `key` in the sorted-key list.

        If the `key` is already present, the insertion point will be before (to
        the left of) any existing keys.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        >>> skl.bisect_key_left(-1)
        4

        :param key: insertion index of key in sorted-key list
        :return: index

        """
        _maxes = self._maxes

        if not _maxes:
            return 0

        pos = bisect_left(_maxes, key)

        if pos == len(_maxes):
            return self._len

        idx = bisect_left(self._keys[pos], key)

        return self._loc(pos, idx)

    _bisect_key_left = bisect_key_left


    def bisect_key_right(self, key):
        """Return an index to insert `key` in the sorted-key list.

        Similar to `bisect_key_left`, but if `key` is already present, the
        insertion point will be after (to the right of) any existing keys.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedList([5, 4, 3, 2, 1], key=neg)
        >>> skl.bisect_key_right(-1)
        5

        :param key: insertion index of key in sorted-key list
        :return: index

        """
        _maxes = self._maxes

        if not _maxes:
            return 0

        pos = bisect_right(_maxes, key)

        if pos == len(_maxes):
            return self._len

        idx = bisect_right(self._keys[pos], key)

        return self._loc(pos, idx)

    bisect_key = bisect_key_right
    _bisect_key_right = bisect_key_right


    def count(self, value):
        """Return number of occurrences of `value` in the sorted-key list.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList([4, 4, 4, 4, 3, 3, 3, 2, 2, 1], key=neg)
        >>> skl.count(2)
        2

        :param value: value to count in sorted-key list
        :return: count

        """
        _maxes = self._maxes

        if not _maxes:
            return 0

        key = self._key(value)
        pos = bisect_left(_maxes, key)

        if pos == len(_maxes):
            return 0

        _lists = self._lists
        _keys = self._keys
        idx = bisect_left(_keys[pos], key)
        total = 0
        len_keys = len(_keys)
        len_sublist = len(_keys[pos])

        while True:
            if _keys[pos][idx] != key:
                return total
            if _lists[pos][idx] == value:
                total += 1
            idx += 1
            if idx == len_sublist:
                pos += 1
                if pos == len_keys:
                    return total
                len_sublist = len(_keys[pos])
                idx = 0


    def copy(self):
        """Return a shallow copy of the sorted-key list.

        Runtime complexity: `O(n)`

        :return: new sorted-key list

        """
        return self.__class__(self, key=self._key)

    __copy__ = copy


    def index(self, value, start=None, stop=None):
        """Return first index of value in sorted-key list.

        Raise ValueError if `value` is not present.

        Index must be between `start` and `stop` for the `value` to be
        considered present. The default value, None, for `start` and `stop`
        indicate the beginning and end of the sorted-key list.

        Negative indices are supported.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        >>> skl.index(2)
        3
        >>> skl.index(0)
        Traceback (most recent call last):
          ...
        ValueError: 0 is not in list

        :param value: value in sorted-key list
        :param int start: start index (default None, start of sorted-key list)
        :param int stop: stop index (default None, end of sorted-key list)
        :return: index of value
        :raises ValueError: if value is not present

        """
        _len = self._len

        if not _len:
            raise ValueError('{0!r} is not in list'.format(value))

        if start is None:
            start = 0
        if start < 0:
            start += _len
        if start < 0:
            start = 0

        if stop is None:
            stop = _len
        if stop < 0:
            stop += _len
        if stop > _len:
            stop = _len

        if stop <= start:
            raise ValueError('{0!r} is not in list'.format(value))

        _maxes = self._maxes
        key = self._key(value)
        pos = bisect_left(_maxes, key)

        if pos == len(_maxes):
            raise ValueError('{0!r} is not in list'.format(value))

        stop -= 1
        _lists = self._lists
        _keys = self._keys
        idx = bisect_left(_keys[pos], key)
        len_keys = len(_keys)
        len_sublist = len(_keys[pos])

        while True:
            if _keys[pos][idx] != key:
                raise ValueError('{0!r} is not in list'.format(value))
            if _lists[pos][idx] == value:
                loc = self._loc(pos, idx)
                if start <= loc <= stop:
                    return loc
                elif loc > stop:
                    break
            idx += 1
            if idx == len_sublist:
                pos += 1
                if pos == len_keys:
                    raise ValueError('{0!r} is not in list'.format(value))
                len_sublist = len(_keys[pos])
                idx = 0

        raise ValueError('{0!r} is not in list'.format(value))


    def __add__(self, other):
        """Return new sorted-key list containing all values in both sequences.

        ``skl.__add__(other)`` <==> ``skl + other``

        Values in `other` do not need to be in sorted-key order.

        Runtime complexity: `O(n*log(n))`

        >>> from operator import neg
        >>> skl1 = SortedKeyList([5, 4, 3], key=neg)
        >>> skl2 = SortedKeyList([2, 1, 0], key=neg)
        >>> skl1 + skl2
        SortedKeyList([5, 4, 3, 2, 1, 0], key=<built-in function neg>)

        :param other: other iterable
        :return: new sorted-key list

        """
        values = reduce(iadd, self._lists, [])
        values.extend(other)
        return self.__class__(values, key=self._key)

    __radd__ = __add__


    def __mul__(self, num):
        """Return new sorted-key list with `num` shallow copies of values.

        ``skl.__mul__(num)`` <==> ``skl * num``

        Runtime complexity: `O(n*log(n))`

        >>> from operator import neg
        >>> skl = SortedKeyList([3, 2, 1], key=neg)
        >>> skl * 2
        SortedKeyList([3, 3, 2, 2, 1, 1], key=<built-in function neg>)

        :param int num: count of shallow copies
        :return: new sorted-key list

        """
        values = reduce(iadd, self._lists, []) * num
        return self.__class__(values, key=self._key)


    def __reduce__(self):
        values = reduce(iadd, self._lists, [])
        return (type(self), (values, self.key))


    @recursive_repr()
    def __repr__(self):
        """Return string representation of sorted-key list.

        ``skl.__repr__()`` <==> ``repr(skl)``

        :return: string representation

        """
        type_name = type(self).__name__
        return '{0}({1!r}, key={2!r})'.format(type_name, list(self), self._key)


    def _check(self):
        """Check invariants of sorted-key list.

        Runtime complexity: `O(n)`

        """
        try:
            assert self._load >= 4
            assert len(self._maxes) == len(self._lists) == len(self._keys)
            assert self._len == sum(len(sublist) for sublist in self._lists)

            # Check all sublists are sorted.

            for sublist in self._keys:
                for pos in range(1, len(sublist)):
                    assert sublist[pos - 1] <= sublist[pos]

            # Check beginning/end of sublists are sorted.

            for pos in range(1, len(self._keys)):
                assert self._keys[pos - 1][-1] <= self._keys[pos][0]

            # Check _keys matches _key mapped to _lists.

            for val_sublist, key_sublist in zip(self._lists, self._keys):
                assert len(val_sublist) == len(key_sublist)
                for val, key in zip(val_sublist, key_sublist):
                    assert self._key(val) == key

            # Check _maxes index is the last value of each sublist.

            for pos in range(len(self._maxes)):
                assert self._maxes[pos] == self._keys[pos][-1]

            # Check sublist lengths are less than double load-factor.

            double = self._load << 1
            assert all(len(sublist) <= double for sublist in self._lists)

            # Check sublist lengths are greater than half load-factor for all
            # but the last sublist.

            half = self._load >> 1
            for pos in range(0, len(self._lists) - 1):
                assert len(self._lists[pos]) >= half

            if self._index:
                assert self._len == self._index[0]
                assert len(self._index) == self._offset + len(self._lists)

                # Check index leaf nodes equal length of sublists.

                for pos in range(len(self._lists)):
                    leaf = self._index[self._offset + pos]
                    assert leaf == len(self._lists[pos])

                # Check index branch nodes are the sum of their children.

                for pos in range(self._offset):
                    child = (pos << 1) + 1
                    if child >= len(self._index):
                        assert self._index[pos] == 0
                    elif child + 1 == len(self._index):
                        assert self._index[pos] == self._index[child]
                    else:
                        child_sum = self._index[child] + self._index[child + 1]
                        assert child_sum == self._index[pos]
        except:
            traceback.print_exc(file=sys.stdout)
            print('len', self._len)
            print('load', self._load)
            print('offset', self._offset)
            print('len_index', len(self._index))
            print('index', self._index)
            print('len_maxes', len(self._maxes))
            print('maxes', self._maxes)
            print('len_keys', len(self._keys))
            print('keys', self._keys)
            print('len_lists', len(self._lists))
            print('lists', self._lists)
            raise


SortedListWithKey = SortedKeyList

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\operators.py ===
# sql/operators.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Defines operators used in SQL expressions."""

from __future__ import annotations

from enum import IntEnum
from operator import add as _uncast_add
from operator import and_ as _uncast_and_
from operator import contains as _uncast_contains
from operator import eq as _uncast_eq
from operator import floordiv as _uncast_floordiv
from operator import ge as _uncast_ge
from operator import getitem as _uncast_getitem
from operator import gt as _uncast_gt
from operator import inv as _uncast_inv
from operator import le as _uncast_le
from operator import lshift as _uncast_lshift
from operator import lt as _uncast_lt
from operator import mod as _uncast_mod
from operator import mul as _uncast_mul
from operator import ne as _uncast_ne
from operator import neg as _uncast_neg
from operator import or_ as _uncast_or_
from operator import rshift as _uncast_rshift
from operator import sub as _uncast_sub
from operator import truediv as _uncast_truediv
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from .. import exc
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol

if typing.TYPE_CHECKING:
    from ._typing import ColumnExpressionArgument
    from .cache_key import CacheConst
    from .elements import ColumnElement
    from .type_api import TypeEngine

_T = TypeVar("_T", bound=Any)
_FN = TypeVar("_FN", bound=Callable[..., Any])


class OperatorType(Protocol):
    """describe an op() function."""

    __slots__ = ()

    __name__: str

    @overload
    def __call__(
        self,
        left: ColumnExpressionArgument[Any],
        right: Optional[Any] = None,
        *other: Any,
        **kwargs: Any,
    ) -> ColumnElement[Any]: ...

    @overload
    def __call__(
        self,
        left: Operators,
        right: Optional[Any] = None,
        *other: Any,
        **kwargs: Any,
    ) -> Operators: ...

    def __call__(
        self,
        left: Any,
        right: Optional[Any] = None,
        *other: Any,
        **kwargs: Any,
    ) -> Operators: ...


add = cast(OperatorType, _uncast_add)
and_ = cast(OperatorType, _uncast_and_)
contains = cast(OperatorType, _uncast_contains)
eq = cast(OperatorType, _uncast_eq)
floordiv = cast(OperatorType, _uncast_floordiv)
ge = cast(OperatorType, _uncast_ge)
getitem = cast(OperatorType, _uncast_getitem)
gt = cast(OperatorType, _uncast_gt)
inv = cast(OperatorType, _uncast_inv)
le = cast(OperatorType, _uncast_le)
lshift = cast(OperatorType, _uncast_lshift)
lt = cast(OperatorType, _uncast_lt)
mod = cast(OperatorType, _uncast_mod)
mul = cast(OperatorType, _uncast_mul)
ne = cast(OperatorType, _uncast_ne)
neg = cast(OperatorType, _uncast_neg)
or_ = cast(OperatorType, _uncast_or_)
rshift = cast(OperatorType, _uncast_rshift)
sub = cast(OperatorType, _uncast_sub)
truediv = cast(OperatorType, _uncast_truediv)


class Operators:
    """Base of comparison and logical operators.

    Implements base methods
    :meth:`~sqlalchemy.sql.operators.Operators.operate` and
    :meth:`~sqlalchemy.sql.operators.Operators.reverse_operate`, as well as
    :meth:`~sqlalchemy.sql.operators.Operators.__and__`,
    :meth:`~sqlalchemy.sql.operators.Operators.__or__`,
    :meth:`~sqlalchemy.sql.operators.Operators.__invert__`.

    Usually is used via its most common subclass
    :class:`.ColumnOperators`.

    """

    __slots__ = ()

    def __and__(self, other: Any) -> Operators:
        """Implement the ``&`` operator.

        When used with SQL expressions, results in an
        AND operation, equivalent to
        :func:`_expression.and_`, that is::

            a & b

        is equivalent to::

            from sqlalchemy import and_

            and_(a, b)

        Care should be taken when using ``&`` regarding
        operator precedence; the ``&`` operator has the highest precedence.
        The operands should be enclosed in parenthesis if they contain
        further sub expressions::

            (a == 2) & (b == 4)

        """
        return self.operate(and_, other)

    def __or__(self, other: Any) -> Operators:
        """Implement the ``|`` operator.

        When used with SQL expressions, results in an
        OR operation, equivalent to
        :func:`_expression.or_`, that is::

            a | b

        is equivalent to::

            from sqlalchemy import or_

            or_(a, b)

        Care should be taken when using ``|`` regarding
        operator precedence; the ``|`` operator has the highest precedence.
        The operands should be enclosed in parenthesis if they contain
        further sub expressions::

            (a == 2) | (b == 4)

        """
        return self.operate(or_, other)

    def __invert__(self) -> Operators:
        """Implement the ``~`` operator.

        When used with SQL expressions, results in a
        NOT operation, equivalent to
        :func:`_expression.not_`, that is::

            ~a

        is equivalent to::

            from sqlalchemy import not_

            not_(a)

        """
        return self.operate(inv)

    def op(
        self,
        opstring: str,
        precedence: int = 0,
        is_comparison: bool = False,
        return_type: Optional[
            Union[Type[TypeEngine[Any]], TypeEngine[Any]]
        ] = None,
        python_impl: Optional[Callable[..., Any]] = None,
    ) -> Callable[[Any], Operators]:
        """Produce a generic operator function.

        e.g.::

          somecolumn.op("*")(5)

        produces::

          somecolumn * 5

        This function can also be used to make bitwise operators explicit. For
        example::

          somecolumn.op("&")(0xFF)

        is a bitwise AND of the value in ``somecolumn``.

        :param opstring: a string which will be output as the infix operator
          between this element and the expression passed to the
          generated function.

        :param precedence: precedence which the database is expected to apply
         to the operator in SQL expressions. This integer value acts as a hint
         for the SQL compiler to know when explicit parenthesis should be
         rendered around a particular operation. A lower number will cause the
         expression to be parenthesized when applied against another operator
         with higher precedence. The default value of ``0`` is lower than all
         operators except for the comma (``,``) and ``AS`` operators. A value
         of 100 will be higher or equal to all operators, and -100 will be
         lower than or equal to all operators.

         .. seealso::

            :ref:`faq_sql_expression_op_parenthesis` - detailed description
            of how the SQLAlchemy SQL compiler renders parenthesis

        :param is_comparison: legacy; if True, the operator will be considered
         as a "comparison" operator, that is which evaluates to a boolean
         true/false value, like ``==``, ``>``, etc.  This flag is provided
         so that ORM relationships can establish that the operator is a
         comparison operator when used in a custom join condition.

         Using the ``is_comparison`` parameter is superseded by using the
         :meth:`.Operators.bool_op` method instead;  this more succinct
         operator sets this parameter automatically, but also provides
         correct :pep:`484` typing support as the returned object will
         express a "boolean" datatype, i.e. ``BinaryExpression[bool]``.

        :param return_type: a :class:`.TypeEngine` class or object that will
          force the return type of an expression produced by this operator
          to be of that type.   By default, operators that specify
          :paramref:`.Operators.op.is_comparison` will resolve to
          :class:`.Boolean`, and those that do not will be of the same
          type as the left-hand operand.

        :param python_impl: an optional Python function that can evaluate
         two Python values in the same way as this operator works when
         run on the database server.  Useful for in-Python SQL expression
         evaluation functions, such as for ORM hybrid attributes, and the
         ORM "evaluator" used to match objects in a session after a multi-row
         update or delete.

         e.g.::

            >>> expr = column("x").op("+", python_impl=lambda a, b: a + b)("y")

         The operator for the above expression will also work for non-SQL
         left and right objects::

            >>> expr.operator(5, 10)
            15

         .. versionadded:: 2.0


        .. seealso::

            :meth:`.Operators.bool_op`

            :ref:`types_operators`

            :ref:`relationship_custom_operator`

        """
        operator = custom_op(
            opstring,
            precedence,
            is_comparison,
            return_type,
            python_impl=python_impl,
        )

        def against(other: Any) -> Operators:
            return operator(self, other)

        return against

    def bool_op(
        self,
        opstring: str,
        precedence: int = 0,
        python_impl: Optional[Callable[..., Any]] = None,
    ) -> Callable[[Any], Operators]:
        """Return a custom boolean operator.

        This method is shorthand for calling
        :meth:`.Operators.op` and passing the
        :paramref:`.Operators.op.is_comparison`
        flag with True.    A key advantage to using :meth:`.Operators.bool_op`
        is that when using column constructs, the "boolean" nature of the
        returned expression will be present for :pep:`484` purposes.

        .. seealso::

            :meth:`.Operators.op`

        """
        return self.op(
            opstring,
            precedence=precedence,
            is_comparison=True,
            python_impl=python_impl,
        )

    def operate(
        self, op: OperatorType, *other: Any, **kwargs: Any
    ) -> Operators:
        r"""Operate on an argument.

        This is the lowest level of operation, raises
        :class:`NotImplementedError` by default.

        Overriding this on a subclass can allow common
        behavior to be applied to all operations.
        For example, overriding :class:`.ColumnOperators`
        to apply ``func.lower()`` to the left and right
        side::

            class MyComparator(ColumnOperators):
                def operate(self, op, other, **kwargs):
                    return op(func.lower(self), func.lower(other), **kwargs)

        :param op:  Operator callable.
        :param \*other: the 'other' side of the operation. Will
         be a single scalar for most operations.
        :param \**kwargs: modifiers.  These may be passed by special
         operators such as :meth:`ColumnOperators.contains`.


        """
        raise NotImplementedError(str(op))

    __sa_operate__ = operate

    def reverse_operate(
        self, op: OperatorType, other: Any, **kwargs: Any
    ) -> Operators:
        """Reverse operate on an argument.

        Usage is the same as :meth:`operate`.

        """
        raise NotImplementedError(str(op))


class custom_op(OperatorType, Generic[_T]):
    """Represent a 'custom' operator.

    :class:`.custom_op` is normally instantiated when the
    :meth:`.Operators.op` or :meth:`.Operators.bool_op` methods
    are used to create a custom operator callable.  The class can also be
    used directly when programmatically constructing expressions.   E.g.
    to represent the "factorial" operation::

        from sqlalchemy.sql import UnaryExpression
        from sqlalchemy.sql import operators
        from sqlalchemy import Numeric

        unary = UnaryExpression(
            table.c.somecolumn, modifier=operators.custom_op("!"), type_=Numeric
        )

    .. seealso::

        :meth:`.Operators.op`

        :meth:`.Operators.bool_op`

    """  # noqa: E501

    __name__ = "custom_op"

    __slots__ = (
        "opstring",
        "precedence",
        "is_comparison",
        "natural_self_precedent",
        "eager_grouping",
        "return_type",
        "python_impl",
    )

    def __init__(
        self,
        opstring: str,
        precedence: int = 0,
        is_comparison: bool = False,
        return_type: Optional[
            Union[Type[TypeEngine[_T]], TypeEngine[_T]]
        ] = None,
        natural_self_precedent: bool = False,
        eager_grouping: bool = False,
        python_impl: Optional[Callable[..., Any]] = None,
    ):
        self.opstring = opstring
        self.precedence = precedence
        self.is_comparison = is_comparison
        self.natural_self_precedent = natural_self_precedent
        self.eager_grouping = eager_grouping
        self.return_type = (
            return_type._to_instance(return_type) if return_type else None
        )
        self.python_impl = python_impl

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, custom_op)
            and other._hash_key() == self._hash_key()
        )

    def __hash__(self) -> int:
        return hash(self._hash_key())

    def _hash_key(self) -> Union[CacheConst, Tuple[Any, ...]]:
        return (
            self.__class__,
            self.opstring,
            self.precedence,
            self.is_comparison,
            self.natural_self_precedent,
            self.eager_grouping,
            self.return_type._static_cache_key if self.return_type else None,
        )

    @overload
    def __call__(
        self,
        left: ColumnExpressionArgument[Any],
        right: Optional[Any] = None,
        *other: Any,
        **kwargs: Any,
    ) -> ColumnElement[Any]: ...

    @overload
    def __call__(
        self,
        left: Operators,
        right: Optional[Any] = None,
        *other: Any,
        **kwargs: Any,
    ) -> Operators: ...

    def __call__(
        self,
        left: Any,
        right: Optional[Any] = None,
        *other: Any,
        **kwargs: Any,
    ) -> Operators:
        if hasattr(left, "__sa_operate__"):
            return left.operate(self, right, *other, **kwargs)  # type: ignore
        elif self.python_impl:
            return self.python_impl(left, right, *other, **kwargs)  # type: ignore  # noqa: E501
        else:
            raise exc.InvalidRequestError(
                f"Custom operator {self.opstring!r} can't be used with "
                "plain Python objects unless it includes the "
                "'python_impl' parameter."
            )


class ColumnOperators(Operators):
    """Defines boolean, comparison, and other operators for
    :class:`_expression.ColumnElement` expressions.

    By default, all methods call down to
    :meth:`.operate` or :meth:`.reverse_operate`,
    passing in the appropriate operator function from the
    Python builtin ``operator`` module or
    a SQLAlchemy-specific operator function from
    :mod:`sqlalchemy.expression.operators`.   For example
    the ``__eq__`` function::

        def __eq__(self, other):
            return self.operate(operators.eq, other)

    Where ``operators.eq`` is essentially::

        def eq(a, b):
            return a == b

    The core column expression unit :class:`_expression.ColumnElement`
    overrides :meth:`.Operators.operate` and others
    to return further :class:`_expression.ColumnElement` constructs,
    so that the ``==`` operation above is replaced by a clause
    construct.

    .. seealso::

        :ref:`types_operators`

        :attr:`.TypeEngine.comparator_factory`

        :class:`.ColumnOperators`

        :class:`.PropComparator`

    """

    __slots__ = ()

    timetuple: Literal[None] = None
    """Hack, allows datetime objects to be compared on the LHS."""

    if typing.TYPE_CHECKING:

        def operate(
            self, op: OperatorType, *other: Any, **kwargs: Any
        ) -> ColumnOperators: ...

        def reverse_operate(
            self, op: OperatorType, other: Any, **kwargs: Any
        ) -> ColumnOperators: ...

    def __lt__(self, other: Any) -> ColumnOperators:
        """Implement the ``<`` operator.

        In a column context, produces the clause ``a < b``.

        """
        return self.operate(lt, other)

    def __le__(self, other: Any) -> ColumnOperators:
        """Implement the ``<=`` operator.

        In a column context, produces the clause ``a <= b``.

        """
        return self.operate(le, other)

    # ColumnOperators defines an __eq__ so it must explicitly declare also
    # an hash or it's set to None by python:
    # https://docs.python.org/3/reference/datamodel.html#object.__hash__
    if TYPE_CHECKING:

        def __hash__(self) -> int: ...

    else:
        __hash__ = Operators.__hash__

    def __eq__(self, other: Any) -> ColumnOperators:  # type: ignore[override]
        """Implement the ``==`` operator.

        In a column context, produces the clause ``a = b``.
        If the target is ``None``, produces ``a IS NULL``.

        """
        return self.operate(eq, other)

    def __ne__(self, other: Any) -> ColumnOperators:  # type: ignore[override]
        """Implement the ``!=`` operator.

        In a column context, produces the clause ``a != b``.
        If the target is ``None``, produces ``a IS NOT NULL``.

        """
        return self.operate(ne, other)

    def is_distinct_from(self, other: Any) -> ColumnOperators:
        """Implement the ``IS DISTINCT FROM`` operator.

        Renders "a IS DISTINCT FROM b" on most platforms;
        on some such as SQLite may render "a IS NOT b".

        """
        return self.operate(is_distinct_from, other)

    def is_not_distinct_from(self, other: Any) -> ColumnOperators:
        """Implement the ``IS NOT DISTINCT FROM`` operator.

        Renders "a IS NOT DISTINCT FROM b" on most platforms;
        on some such as SQLite may render "a IS b".

        .. versionchanged:: 1.4 The ``is_not_distinct_from()`` operator is
           renamed from ``isnot_distinct_from()`` in previous releases.
           The previous name remains available for backwards compatibility.

        """
        return self.operate(is_not_distinct_from, other)

    # deprecated 1.4; see #5435
    if TYPE_CHECKING:

        def isnot_distinct_from(self, other: Any) -> ColumnOperators: ...

    else:
        isnot_distinct_from = is_not_distinct_from

    def __gt__(self, other: Any) -> ColumnOperators:
        """Implement the ``>`` operator.

        In a column context, produces the clause ``a > b``.

        """
        return self.operate(gt, other)

    def __ge__(self, other: Any) -> ColumnOperators:
        """Implement the ``>=`` operator.

        In a column context, produces the clause ``a >= b``.

        """
        return self.operate(ge, other)

    def __neg__(self) -> ColumnOperators:
        """Implement the ``-`` operator.

        In a column context, produces the clause ``-a``.

        """
        return self.operate(neg)

    def __contains__(self, other: Any) -> ColumnOperators:
        return self.operate(contains, other)

    def __getitem__(self, index: Any) -> ColumnOperators:
        """Implement the [] operator.

        This can be used by some database-specific types
        such as PostgreSQL ARRAY and HSTORE.

        """
        return self.operate(getitem, index)

    def __lshift__(self, other: Any) -> ColumnOperators:
        """implement the << operator.

        Not used by SQLAlchemy core, this is provided
        for custom operator systems which want to use
        << as an extension point.
        """
        return self.operate(lshift, other)

    def __rshift__(self, other: Any) -> ColumnOperators:
        """implement the >> operator.

        Not used by SQLAlchemy core, this is provided
        for custom operator systems which want to use
        >> as an extension point.
        """
        return self.operate(rshift, other)

    def concat(self, other: Any) -> ColumnOperators:
        """Implement the 'concat' operator.

        In a column context, produces the clause ``a || b``,
        or uses the ``concat()`` operator on MySQL.

        """
        return self.operate(concat_op, other)

    def _rconcat(self, other: Any) -> ColumnOperators:
        """Implement an 'rconcat' operator.

        this is for internal use at the moment

        .. versionadded:: 1.4.40

        """
        return self.reverse_operate(concat_op, other)

    def like(
        self, other: Any, escape: Optional[str] = None
    ) -> ColumnOperators:
        r"""Implement the ``like`` operator.

        In a column context, produces the expression:

        .. sourcecode:: sql

            a LIKE other

        E.g.::

            stmt = select(sometable).where(sometable.c.column.like("%foobar%"))

        :param other: expression to be compared
        :param escape: optional escape character, renders the ``ESCAPE``
          keyword, e.g.::

            somecolumn.like("foo/%bar", escape="/")

        .. seealso::

            :meth:`.ColumnOperators.ilike`

        """
        return self.operate(like_op, other, escape=escape)

    def ilike(
        self, other: Any, escape: Optional[str] = None
    ) -> ColumnOperators:
        r"""Implement the ``ilike`` operator, e.g. case insensitive LIKE.

        In a column context, produces an expression either of the form:

        .. sourcecode:: sql

            lower(a) LIKE lower(other)

        Or on backends that support the ILIKE operator:

        .. sourcecode:: sql

            a ILIKE other

        E.g.::

            stmt = select(sometable).where(sometable.c.column.ilike("%foobar%"))

        :param other: expression to be compared
        :param escape: optional escape character, renders the ``ESCAPE``
          keyword, e.g.::

            somecolumn.ilike("foo/%bar", escape="/")

        .. seealso::

            :meth:`.ColumnOperators.like`

        """  # noqa: E501
        return self.operate(ilike_op, other, escape=escape)

    def bitwise_xor(self, other: Any) -> ColumnOperators:
        """Produce a bitwise XOR operation, typically via the ``^``
        operator, or ``#`` for PostgreSQL.

        .. versionadded:: 2.0.2

        .. seealso::

            :ref:`operators_bitwise`

        """

        return self.operate(bitwise_xor_op, other)

    def bitwise_or(self, other: Any) -> ColumnOperators:
        """Produce a bitwise OR operation, typically via the ``|``
        operator.

        .. versionadded:: 2.0.2

        .. seealso::

            :ref:`operators_bitwise`

        """

        return self.operate(bitwise_or_op, other)

    def bitwise_and(self, other: Any) -> ColumnOperators:
        """Produce a bitwise AND operation, typically via the ``&``
        operator.

        .. versionadded:: 2.0.2

        .. seealso::

            :ref:`operators_bitwise`

        """

        return self.operate(bitwise_and_op, other)

    def bitwise_not(self) -> ColumnOperators:
        """Produce a bitwise NOT operation, typically via the ``~``
        operator.

        .. versionadded:: 2.0.2

        .. seealso::

            :ref:`operators_bitwise`

        """

        return self.operate(bitwise_not_op)

    def bitwise_lshift(self, other: Any) -> ColumnOperators:
        """Produce a bitwise LSHIFT operation, typically via the ``<<``
        operator.

        .. versionadded:: 2.0.2

        .. seealso::

            :ref:`operators_bitwise`

        """

        return self.operate(bitwise_lshift_op, other)

    def bitwise_rshift(self, other: Any) -> ColumnOperators:
        """Produce a bitwise RSHIFT operation, typically via the ``>>``
        operator.

        .. versionadded:: 2.0.2

        .. seealso::

            :ref:`operators_bitwise`

        """

        return self.operate(bitwise_rshift_op, other)

    def in_(self, other: Any) -> ColumnOperators:
        """Implement the ``in`` operator.

        In a column context, produces the clause ``column IN <other>``.

        The given parameter ``other`` may be:

        * A list of literal values,
          e.g.::

            stmt.where(column.in_([1, 2, 3]))

          In this calling form, the list of items is converted to a set of
          bound parameters the same length as the list given:

          .. sourcecode:: sql

            WHERE COL IN (?, ?, ?)

        * A list of tuples may be provided if the comparison is against a
          :func:`.tuple_` containing multiple expressions::

            from sqlalchemy import tuple_

            stmt.where(tuple_(col1, col2).in_([(1, 10), (2, 20), (3, 30)]))

        * An empty list,
          e.g.::

            stmt.where(column.in_([]))

          In this calling form, the expression renders an "empty set"
          expression.  These expressions are tailored to individual backends
          and are generally trying to get an empty SELECT statement as a
          subquery.  Such as on SQLite, the expression is:

          .. sourcecode:: sql

            WHERE col IN (SELECT 1 FROM (SELECT 1) WHERE 1!=1)

          .. versionchanged:: 1.4  empty IN expressions now use an
             execution-time generated SELECT subquery in all cases.

        * A bound parameter, e.g. :func:`.bindparam`, may be used if it
          includes the :paramref:`.bindparam.expanding` flag::

            stmt.where(column.in_(bindparam("value", expanding=True)))

          In this calling form, the expression renders a special non-SQL
          placeholder expression that looks like:

          .. sourcecode:: sql

            WHERE COL IN ([EXPANDING_value])

          This placeholder expression is intercepted at statement execution
          time to be converted into the variable number of bound parameter
          form illustrated earlier.   If the statement were executed as::

            connection.execute(stmt, {"value": [1, 2, 3]})

          The database would be passed a bound parameter for each value:

          .. sourcecode:: sql

            WHERE COL IN (?, ?, ?)

          .. versionadded:: 1.2 added "expanding" bound parameters

          If an empty list is passed, a special "empty list" expression,
          which is specific to the database in use, is rendered.  On
          SQLite this would be:

          .. sourcecode:: sql

            WHERE COL IN (SELECT 1 FROM (SELECT 1) WHERE 1!=1)

          .. versionadded:: 1.3 "expanding" bound parameters now support
             empty lists

        * a :func:`_expression.select` construct, which is usually a
          correlated scalar select::

            stmt.where(
                column.in_(select(othertable.c.y).where(table.c.x == othertable.c.x))
            )

          In this calling form, :meth:`.ColumnOperators.in_` renders as given:

          .. sourcecode:: sql

            WHERE COL IN (SELECT othertable.y
            FROM othertable WHERE othertable.x = table.x)

        :param other: a list of literals, a :func:`_expression.select`
         construct, or a :func:`.bindparam` construct that includes the
         :paramref:`.bindparam.expanding` flag set to True.

        """  # noqa: E501
        return self.operate(in_op, other)

    def not_in(self, other: Any) -> ColumnOperators:
        """implement the ``NOT IN`` operator.

        This is equivalent to using negation with
        :meth:`.ColumnOperators.in_`, i.e. ``~x.in_(y)``.

        In the case that ``other`` is an empty sequence, the compiler
        produces an "empty not in" expression.   This defaults to the
        expression "1 = 1" to produce true in all cases.  The
        :paramref:`_sa.create_engine.empty_in_strategy` may be used to
        alter this behavior.

        .. versionchanged:: 1.4 The ``not_in()`` operator is renamed from
           ``notin_()`` in previous releases.  The previous name remains
           available for backwards compatibility.

        .. versionchanged:: 1.2  The :meth:`.ColumnOperators.in_` and
           :meth:`.ColumnOperators.not_in` operators
           now produce a "static" expression for an empty IN sequence
           by default.

        .. seealso::

            :meth:`.ColumnOperators.in_`

        """
        return self.operate(not_in_op, other)

    # deprecated 1.4; see #5429
    if TYPE_CHECKING:

        def notin_(self, other: Any) -> ColumnOperators: ...

    else:
        notin_ = not_in

    def not_like(
        self, other: Any, escape: Optional[str] = None
    ) -> ColumnOperators:
        """implement the ``NOT LIKE`` operator.

        This is equivalent to using negation with
        :meth:`.ColumnOperators.like`, i.e. ``~x.like(y)``.

        .. versionchanged:: 1.4 The ``not_like()`` operator is renamed from
           ``notlike()`` in previous releases.  The previous name remains
           available for backwards compatibility.

        .. seealso::

            :meth:`.ColumnOperators.like`

        """
        return self.operate(not_like_op, other, escape=escape)

    # deprecated 1.4; see #5435
    if TYPE_CHECKING:

        def notlike(
            self, other: Any, escape: Optional[str] = None
        ) -> ColumnOperators: ...

    else:
        notlike = not_like

    def not_ilike(
        self, other: Any, escape: Optional[str] = None
    ) -> ColumnOperators:
        """implement the ``NOT ILIKE`` operator.

        This is equivalent to using negation with
        :meth:`.ColumnOperators.ilike`, i.e. ``~x.ilike(y)``.

        .. versionchanged:: 1.4 The ``not_ilike()`` operator is renamed from
           ``notilike()`` in previous releases.  The previous name remains
           available for backwards compatibility.

        .. seealso::

            :meth:`.ColumnOperators.ilike`

        """
        return self.operate(not_ilike_op, other, escape=escape)

    # deprecated 1.4; see #5435
    if TYPE_CHECKING:

        def notilike(
            self, other: Any, escape: Optional[str] = None
        ) -> ColumnOperators: ...

    else:
        notilike = not_ilike

    def is_(self, other: Any) -> ColumnOperators:
        """Implement the ``IS`` operator.

        Normally, ``IS`` is generated automatically when comparing to a
        value of ``None``, which resolves to ``NULL``.  However, explicit
        usage of ``IS`` may be desirable if comparing to boolean values
        on certain platforms.

        .. seealso:: :meth:`.ColumnOperators.is_not`

        """
        return self.operate(is_, other)

    def is_not(self, other: Any) -> ColumnOperators:
        """Implement the ``IS NOT`` operator.

        Normally, ``IS NOT`` is generated automatically when comparing to a
        value of ``None``, which resolves to ``NULL``.  However, explicit
        usage of ``IS NOT`` may be desirable if comparing to boolean values
        on certain platforms.

        .. versionchanged:: 1.4 The ``is_not()`` operator is renamed from
           ``isnot()`` in previous releases.  The previous name remains
           available for backwards compatibility.

        .. seealso:: :meth:`.ColumnOperators.is_`

        """
        return self.operate(is_not, other)

    # deprecated 1.4; see #5429
    if TYPE_CHECKING:

        def isnot(self, other: Any) -> ColumnOperators: ...

    else:
        isnot = is_not

    def startswith(
        self,
        other: Any,
        escape: Optional[str] = None,
        autoescape: bool = False,
    ) -> ColumnOperators:
        r"""Implement the ``startswith`` operator.

        Produces a LIKE expression that tests against a match for the start
        of a string value:

        .. sourcecode:: sql

            column LIKE <other> || '%'

        E.g.::

            stmt = select(sometable).where(sometable.c.column.startswith("foobar"))

        Since the operator uses ``LIKE``, wildcard characters
        ``"%"`` and ``"_"`` that are present inside the <other> expression
        will behave like wildcards as well.   For literal string
        values, the :paramref:`.ColumnOperators.startswith.autoescape` flag
        may be set to ``True`` to apply escaping to occurrences of these
        characters within the string value so that they match as themselves
        and not as wildcard characters.  Alternatively, the
        :paramref:`.ColumnOperators.startswith.escape` parameter will establish
        a given character as an escape character which can be of use when
        the target expression is not a literal string.

        :param other: expression to be compared.   This is usually a plain
          string value, but can also be an arbitrary SQL expression.  LIKE
          wildcard characters ``%`` and ``_`` are not escaped by default unless
          the :paramref:`.ColumnOperators.startswith.autoescape` flag is
          set to True.

        :param autoescape: boolean; when True, establishes an escape character
          within the LIKE expression, then applies it to all occurrences of
          ``"%"``, ``"_"`` and the escape character itself within the
          comparison value, which is assumed to be a literal string and not a
          SQL expression.

          An expression such as::

            somecolumn.startswith("foo%bar", autoescape=True)

          Will render as:

          .. sourcecode:: sql

            somecolumn LIKE :param || '%' ESCAPE '/'

          With the value of ``:param`` as ``"foo/%bar"``.

        :param escape: a character which when given will render with the
          ``ESCAPE`` keyword to establish that character as the escape
          character.  This character can then be placed preceding occurrences
          of ``%`` and ``_`` to allow them to act as themselves and not
          wildcard characters.

          An expression such as::

            somecolumn.startswith("foo/%bar", escape="^")

          Will render as:

          .. sourcecode:: sql

            somecolumn LIKE :param || '%' ESCAPE '^'

          The parameter may also be combined with
          :paramref:`.ColumnOperators.startswith.autoescape`::

            somecolumn.startswith("foo%bar^bat", escape="^", autoescape=True)

          Where above, the given literal parameter will be converted to
          ``"foo^%bar^^bat"`` before being passed to the database.

        .. seealso::

            :meth:`.ColumnOperators.endswith`

            :meth:`.ColumnOperators.contains`

            :meth:`.ColumnOperators.like`

        """  # noqa: E501
        return self.operate(
            startswith_op, other, escape=escape, autoescape=autoescape
        )

    def istartswith(
        self,
        other: Any,
        escape: Optional[str] = None,
        autoescape: bool = False,
    ) -> ColumnOperators:
        r"""Implement the ``istartswith`` operator, e.g. case insensitive
        version of :meth:`.ColumnOperators.startswith`.

        Produces a LIKE expression that tests against an insensitive
        match for the start of a string value:

        .. sourcecode:: sql

            lower(column) LIKE lower(<other>) || '%'

        E.g.::

            stmt = select(sometable).where(sometable.c.column.istartswith("foobar"))

        Since the operator uses ``LIKE``, wildcard characters
        ``"%"`` and ``"_"`` that are present inside the <other> expression
        will behave like wildcards as well.   For literal string
        values, the :paramref:`.ColumnOperators.istartswith.autoescape` flag
        may be set to ``True`` to apply escaping to occurrences of these
        characters within the string value so that they match as themselves
        and not as wildcard characters.  Alternatively, the
        :paramref:`.ColumnOperators.istartswith.escape` parameter will
        establish a given character as an escape character which can be of
        use when the target expression is not a literal string.

        :param other: expression to be compared.   This is usually a plain
          string value, but can also be an arbitrary SQL expression.  LIKE
          wildcard characters ``%`` and ``_`` are not escaped by default unless
          the :paramref:`.ColumnOperators.istartswith.autoescape` flag is
          set to True.

        :param autoescape: boolean; when True, establishes an escape character
          within the LIKE expression, then applies it to all occurrences of
          ``"%"``, ``"_"`` and the escape character itself within the
          comparison value, which is assumed to be a literal string and not a
          SQL expression.

          An expression such as::

            somecolumn.istartswith("foo%bar", autoescape=True)

          Will render as:

          .. sourcecode:: sql

            lower(somecolumn) LIKE lower(:param) || '%' ESCAPE '/'

          With the value of ``:param`` as ``"foo/%bar"``.

        :param escape: a character which when given will render with the
          ``ESCAPE`` keyword to establish that character as the escape
          character.  This character can then be placed preceding occurrences
          of ``%`` and ``_`` to allow them to act as themselves and not
          wildcard characters.

          An expression such as::

            somecolumn.istartswith("foo/%bar", escape="^")

          Will render as:

          .. sourcecode:: sql

            lower(somecolumn) LIKE lower(:param) || '%' ESCAPE '^'

          The parameter may also be combined with
          :paramref:`.ColumnOperators.istartswith.autoescape`::

            somecolumn.istartswith("foo%bar^bat", escape="^", autoescape=True)

          Where above, the given literal parameter will be converted to
          ``"foo^%bar^^bat"`` before being passed to the database.

        .. seealso::

            :meth:`.ColumnOperators.startswith`
        """  # noqa: E501
        return self.operate(
            istartswith_op, other, escape=escape, autoescape=autoescape
        )

    def endswith(
        self,
        other: Any,
        escape: Optional[str] = None,
        autoescape: bool = False,
    ) -> ColumnOperators:
        r"""Implement the 'endswith' operator.

        Produces a LIKE expression that tests against a match for the end
        of a string value:

        .. sourcecode:: sql

            column LIKE '%' || <other>

        E.g.::

            stmt = select(sometable).where(sometable.c.column.endswith("foobar"))

        Since the operator uses ``LIKE``, wildcard characters
        ``"%"`` and ``"_"`` that are present inside the <other> expression
        will behave like wildcards as well.   For literal string
        values, the :paramref:`.ColumnOperators.endswith.autoescape` flag
        may be set to ``True`` to apply escaping to occurrences of these
        characters within the string value so that they match as themselves
        and not as wildcard characters.  Alternatively, the
        :paramref:`.ColumnOperators.endswith.escape` parameter will establish
        a given character as an escape character which can be of use when
        the target expression is not a literal string.

        :param other: expression to be compared.   This is usually a plain
          string value, but can also be an arbitrary SQL expression.  LIKE
          wildcard characters ``%`` and ``_`` are not escaped by default unless
          the :paramref:`.ColumnOperators.endswith.autoescape` flag is
          set to True.

        :param autoescape: boolean; when True, establishes an escape character
          within the LIKE expression, then applies it to all occurrences of
          ``"%"``, ``"_"`` and the escape character itself within the
          comparison value, which is assumed to be a literal string and not a
          SQL expression.

          An expression such as::

            somecolumn.endswith("foo%bar", autoescape=True)

          Will render as:

          .. sourcecode:: sql

            somecolumn LIKE '%' || :param ESCAPE '/'

          With the value of ``:param`` as ``"foo/%bar"``.

        :param escape: a character which when given will render with the
          ``ESCAPE`` keyword to establish that character as the escape
          character.  This character can then be placed preceding occurrences
          of ``%`` and ``_`` to allow them to act as themselves and not
          wildcard characters.

          An expression such as::

            somecolumn.endswith("foo/%bar", escape="^")

          Will render as:

          .. sourcecode:: sql

            somecolumn LIKE '%' || :param ESCAPE '^'

          The parameter may also be combined with
          :paramref:`.ColumnOperators.endswith.autoescape`::

            somecolumn.endswith("foo%bar^bat", escape="^", autoescape=True)

          Where above, the given literal parameter will be converted to
          ``"foo^%bar^^bat"`` before being passed to the database.

        .. seealso::

            :meth:`.ColumnOperators.startswith`

            :meth:`.ColumnOperators.contains`

            :meth:`.ColumnOperators.like`

        """  # noqa: E501
        return self.operate(
            endswith_op, other, escape=escape, autoescape=autoescape
        )

    def iendswith(
        self,
        other: Any,
        escape: Optional[str] = None,
        autoescape: bool = False,
    ) -> ColumnOperators:
        r"""Implement the ``iendswith`` operator, e.g. case insensitive
        version of :meth:`.ColumnOperators.endswith`.

        Produces a LIKE expression that tests against an insensitive match
        for the end of a string value:

        .. sourcecode:: sql

            lower(column) LIKE '%' || lower(<other>)

        E.g.::

            stmt = select(sometable).where(sometable.c.column.iendswith("foobar"))

        Since the operator uses ``LIKE``, wildcard characters
        ``"%"`` and ``"_"`` that are present inside the <other> expression
        will behave like wildcards as well.   For literal string
        values, the :paramref:`.ColumnOperators.iendswith.autoescape` flag
        may be set to ``True`` to apply escaping to occurrences of these
        characters within the string value so that they match as themselves
        and not as wildcard characters.  Alternatively, the
        :paramref:`.ColumnOperators.iendswith.escape` parameter will establish
        a given character as an escape character which can be of use when
        the target expression is not a literal string.

        :param other: expression to be compared.   This is usually a plain
          string value, but can also be an arbitrary SQL expression.  LIKE
          wildcard characters ``%`` and ``_`` are not escaped by default unless
          the :paramref:`.ColumnOperators.iendswith.autoescape` flag is
          set to True.

        :param autoescape: boolean; when True, establishes an escape character
          within the LIKE expression, then applies it to all occurrences of
          ``"%"``, ``"_"`` and the escape character itself within the
          comparison value, which is assumed to be a literal string and not a
          SQL expression.

          An expression such as::

            somecolumn.iendswith("foo%bar", autoescape=True)

          Will render as:

          .. sourcecode:: sql

            lower(somecolumn) LIKE '%' || lower(:param) ESCAPE '/'

          With the value of ``:param`` as ``"foo/%bar"``.

        :param escape: a character which when given will render with the
          ``ESCAPE`` keyword to establish that character as the escape
          character.  This character can then be placed preceding occurrences
          of ``%`` and ``_`` to allow them to act as themselves and not
          wildcard characters.

          An expression such as::

            somecolumn.iendswith("foo/%bar", escape="^")

          Will render as:

          .. sourcecode:: sql

            lower(somecolumn) LIKE '%' || lower(:param) ESCAPE '^'

          The parameter may also be combined with
          :paramref:`.ColumnOperators.iendswith.autoescape`::

            somecolumn.endswith("foo%bar^bat", escape="^", autoescape=True)

          Where above, the given literal parameter will be converted to
          ``"foo^%bar^^bat"`` before being passed to the database.

        .. seealso::

            :meth:`.ColumnOperators.endswith`
        """  # noqa: E501
        return self.operate(
            iendswith_op, other, escape=escape, autoescape=autoescape
        )

    def contains(self, other: Any, **kw: Any) -> ColumnOperators:
        r"""Implement the 'contains' operator.

        Produces a LIKE expression that tests against a match for the middle
        of a string value:

        .. sourcecode:: sql

            column LIKE '%' || <other> || '%'

        E.g.::

            stmt = select(sometable).where(sometable.c.column.contains("foobar"))

        Since the operator uses ``LIKE``, wildcard characters
        ``"%"`` and ``"_"`` that are present inside the <other> expression
        will behave like wildcards as well.   For literal string
        values, the :paramref:`.ColumnOperators.contains.autoescape` flag
        may be set to ``True`` to apply escaping to occurrences of these
        characters within the string value so that they match as themselves
        and not as wildcard characters.  Alternatively, the
        :paramref:`.ColumnOperators.contains.escape` parameter will establish
        a given character as an escape character which can be of use when
        the target expression is not a literal string.

        :param other: expression to be compared.   This is usually a plain
          string value, but can also be an arbitrary SQL expression.  LIKE
          wildcard characters ``%`` and ``_`` are not escaped by default unless
          the :paramref:`.ColumnOperators.contains.autoescape` flag is
          set to True.

        :param autoescape: boolean; when True, establishes an escape character
          within the LIKE expression, then applies it to all occurrences of
          ``"%"``, ``"_"`` and the escape character itself within the
          comparison value, which is assumed to be a literal string and not a
          SQL expression.

          An expression such as::

            somecolumn.contains("foo%bar", autoescape=True)

          Will render as:

          .. sourcecode:: sql

            somecolumn LIKE '%' || :param || '%' ESCAPE '/'

          With the value of ``:param`` as ``"foo/%bar"``.

        :param escape: a character which when given will render with the
          ``ESCAPE`` keyword to establish that character as the escape
          character.  This character can then be placed preceding occurrences
          of ``%`` and ``_`` to allow them to act as themselves and not
          wildcard characters.

          An expression such as::

            somecolumn.contains("foo/%bar", escape="^")

          Will render as:

          .. sourcecode:: sql

            somecolumn LIKE '%' || :param || '%' ESCAPE '^'

          The parameter may also be combined with
          :paramref:`.ColumnOperators.contains.autoescape`::

            somecolumn.contains("foo%bar^bat", escape="^", autoescape=True)

          Where above, the given literal parameter will be converted to
          ``"foo^%bar^^bat"`` before being passed to the database.

        .. seealso::

            :meth:`.ColumnOperators.startswith`

            :meth:`.ColumnOperators.endswith`

            :meth:`.ColumnOperators.like`


        """  # noqa: E501
        return self.operate(contains_op, other, **kw)

    def icontains(self, other: Any, **kw: Any) -> ColumnOperators:
        r"""Implement the ``icontains`` operator, e.g. case insensitive
        version of :meth:`.ColumnOperators.contains`.

        Produces a LIKE expression that tests against an insensitive match
        for the middle of a string value:

        .. sourcecode:: sql

            lower(column) LIKE '%' || lower(<other>) || '%'

        E.g.::

            stmt = select(sometable).where(sometable.c.column.icontains("foobar"))

        Since the operator uses ``LIKE``, wildcard characters
        ``"%"`` and ``"_"`` that are present inside the <other> expression
        will behave like wildcards as well.   For literal string
        values, the :paramref:`.ColumnOperators.icontains.autoescape` flag
        may be set to ``True`` to apply escaping to occurrences of these
        characters within the string value so that they match as themselves
        and not as wildcard characters.  Alternatively, the
        :paramref:`.ColumnOperators.icontains.escape` parameter will establish
        a given character as an escape character which can be of use when
        the target expression is not a literal string.

        :param other: expression to be compared.   This is usually a plain
          string value, but can also be an arbitrary SQL expression.  LIKE
          wildcard characters ``%`` and ``_`` are not escaped by default unless
          the :paramref:`.ColumnOperators.icontains.autoescape` flag is
          set to True.

        :param autoescape: boolean; when True, establishes an escape character
          within the LIKE expression, then applies it to all occurrences of
          ``"%"``, ``"_"`` and the escape character itself within the
          comparison value, which is assumed to be a literal string and not a
          SQL expression.

          An expression such as::

            somecolumn.icontains("foo%bar", autoescape=True)

          Will render as:

          .. sourcecode:: sql

            lower(somecolumn) LIKE '%' || lower(:param) || '%' ESCAPE '/'

          With the value of ``:param`` as ``"foo/%bar"``.

        :param escape: a character which when given will render with the
          ``ESCAPE`` keyword to establish that character as the escape
          character.  This character can then be placed preceding occurrences
          of ``%`` and ``_`` to allow them to act as themselves and not
          wildcard characters.

          An expression such as::

            somecolumn.icontains("foo/%bar", escape="^")

          Will render as:

          .. sourcecode:: sql

            lower(somecolumn) LIKE '%' || lower(:param) || '%' ESCAPE '^'

          The parameter may also be combined with
          :paramref:`.ColumnOperators.contains.autoescape`::

            somecolumn.icontains("foo%bar^bat", escape="^", autoescape=True)

          Where above, the given literal parameter will be converted to
          ``"foo^%bar^^bat"`` before being passed to the database.

        .. seealso::

            :meth:`.ColumnOperators.contains`

        """  # noqa: E501
        return self.operate(icontains_op, other, **kw)

    def match(self, other: Any, **kwargs: Any) -> ColumnOperators:
        """Implements a database-specific 'match' operator.

        :meth:`_sql.ColumnOperators.match` attempts to resolve to
        a MATCH-like function or operator provided by the backend.
        Examples include:

        * PostgreSQL - renders ``x @@ plainto_tsquery(y)``

            .. versionchanged:: 2.0  ``plainto_tsquery()`` is used instead
               of ``to_tsquery()`` for PostgreSQL now; for compatibility with
               other forms, see :ref:`postgresql_match`.


        * MySQL - renders ``MATCH (x) AGAINST (y IN BOOLEAN MODE)``

          .. seealso::

                :class:`_mysql.match` - MySQL specific construct with
                additional features.

        * Oracle Database - renders ``CONTAINS(x, y)``
        * other backends may provide special implementations.
        * Backends without any special implementation will emit
          the operator as "MATCH".  This is compatible with SQLite, for
          example.

        """
        return self.operate(match_op, other, **kwargs)

    def regexp_match(
        self, pattern: Any, flags: Optional[str] = None
    ) -> ColumnOperators:
        """Implements a database-specific 'regexp match' operator.

        E.g.::

            stmt = select(table.c.some_column).where(
                table.c.some_column.regexp_match("^(b|c)")
            )

        :meth:`_sql.ColumnOperators.regexp_match` attempts to resolve to
        a REGEXP-like function or operator provided by the backend, however
        the specific regular expression syntax and flags available are
        **not backend agnostic**.

        Examples include:

        * PostgreSQL - renders ``x ~ y`` or ``x !~ y`` when negated.
        * Oracle Database - renders ``REGEXP_LIKE(x, y)``
        * SQLite - uses SQLite's ``REGEXP`` placeholder operator and calls into
          the Python ``re.match()`` builtin.
        * other backends may provide special implementations.
        * Backends without any special implementation will emit
          the operator as "REGEXP" or "NOT REGEXP".  This is compatible with
          SQLite and MySQL, for example.

        Regular expression support is currently implemented for Oracle
        Database, PostgreSQL, MySQL and MariaDB.  Partial support is available
        for SQLite.  Support among third-party dialects may vary.

        :param pattern: The regular expression pattern string or column
          clause.
        :param flags: Any regular expression string flags to apply, passed as
          plain Python string only.  These flags are backend specific.
          Some backends, like PostgreSQL and MariaDB, may alternatively
          specify the flags as part of the pattern.
          When using the ignore case flag 'i' in PostgreSQL, the ignore case
          regexp match operator ``~*`` or ``!~*`` will be used.

        .. versionadded:: 1.4

        .. versionchanged:: 1.4.48, 2.0.18  Note that due to an implementation
           error, the "flags" parameter previously accepted SQL expression
           objects such as column expressions in addition to plain Python
           strings.   This implementation did not work correctly with caching
           and was removed; strings only should be passed for the "flags"
           parameter, as these flags are rendered as literal inline values
           within SQL expressions.

        .. seealso::

            :meth:`_sql.ColumnOperators.regexp_replace`


        """
        return self.operate(regexp_match_op, pattern, flags=flags)

    def regexp_replace(
        self, pattern: Any, replacement: Any, flags: Optional[str] = None
    ) -> ColumnOperators:
        """Implements a database-specific 'regexp replace' operator.

        E.g.::

            stmt = select(
                table.c.some_column.regexp_replace("b(..)", "X\1Y", flags="g")
            )

        :meth:`_sql.ColumnOperators.regexp_replace` attempts to resolve to
        a REGEXP_REPLACE-like function provided by the backend, that
        usually emit the function ``REGEXP_REPLACE()``.  However,
        the specific regular expression syntax and flags available are
        **not backend agnostic**.

        Regular expression replacement support is currently implemented for
        Oracle Database, PostgreSQL, MySQL 8 or greater and MariaDB.  Support
        among third-party dialects may vary.

        :param pattern: The regular expression pattern string or column
          clause.
        :param pattern: The replacement string or column clause.
        :param flags: Any regular expression string flags to apply, passed as
          plain Python string only.  These flags are backend specific.
          Some backends, like PostgreSQL and MariaDB, may alternatively
          specify the flags as part of the pattern.

        .. versionadded:: 1.4

        .. versionchanged:: 1.4.48, 2.0.18  Note that due to an implementation
           error, the "flags" parameter previously accepted SQL expression
           objects such as column expressions in addition to plain Python
           strings.   This implementation did not work correctly with caching
           and was removed; strings only should be passed for the "flags"
           parameter, as these flags are rendered as literal inline values
           within SQL expressions.


        .. seealso::

            :meth:`_sql.ColumnOperators.regexp_match`

        """
        return self.operate(
            regexp_replace_op,
            pattern,
            replacement=replacement,
            flags=flags,
        )

    def desc(self) -> ColumnOperators:
        """Produce a :func:`_expression.desc` clause against the
        parent object."""
        return self.operate(desc_op)

    def asc(self) -> ColumnOperators:
        """Produce a :func:`_expression.asc` clause against the
        parent object."""
        return self.operate(asc_op)

    def nulls_first(self) -> ColumnOperators:
        """Produce a :func:`_expression.nulls_first` clause against the
        parent object.

        .. versionchanged:: 1.4 The ``nulls_first()`` operator is
           renamed from ``nullsfirst()`` in previous releases.
           The previous name remains available for backwards compatibility.
        """
        return self.operate(nulls_first_op)

    # deprecated 1.4; see #5435
    if TYPE_CHECKING:

        def nullsfirst(self) -> ColumnOperators: ...

    else:
        nullsfirst = nulls_first

    def nulls_last(self) -> ColumnOperators:
        """Produce a :func:`_expression.nulls_last` clause against the
        parent object.

        .. versionchanged:: 1.4 The ``nulls_last()`` operator is
           renamed from ``nullslast()`` in previous releases.
           The previous name remains available for backwards compatibility.
        """
        return self.operate(nulls_last_op)

    # deprecated 1.4; see #5429
    if TYPE_CHECKING:

        def nullslast(self) -> ColumnOperators: ...

    else:
        nullslast = nulls_last

    def collate(self, collation: str) -> ColumnOperators:
        """Produce a :func:`_expression.collate` clause against
        the parent object, given the collation string.

        .. seealso::

            :func:`_expression.collate`

        """
        return self.operate(collate, collation)

    def __radd__(self, other: Any) -> ColumnOperators:
        """Implement the ``+`` operator in reverse.

        See :meth:`.ColumnOperators.__add__`.

        """
        return self.reverse_operate(add, other)

    def __rsub__(self, other: Any) -> ColumnOperators:
        """Implement the ``-`` operator in reverse.

        See :meth:`.ColumnOperators.__sub__`.

        """
        return self.reverse_operate(sub, other)

    def __rmul__(self, other: Any) -> ColumnOperators:
        """Implement the ``*`` operator in reverse.

        See :meth:`.ColumnOperators.__mul__`.

        """
        return self.reverse_operate(mul, other)

    def __rmod__(self, other: Any) -> ColumnOperators:
        """Implement the ``%`` operator in reverse.

        See :meth:`.ColumnOperators.__mod__`.

        """
        return self.reverse_operate(mod, other)

    def between(
        self, cleft: Any, cright: Any, symmetric: bool = False
    ) -> ColumnOperators:
        """Produce a :func:`_expression.between` clause against
        the parent object, given the lower and upper range.

        """
        return self.operate(between_op, cleft, cright, symmetric=symmetric)

    def distinct(self) -> ColumnOperators:
        """Produce a :func:`_expression.distinct` clause against the
        parent object.

        """
        return self.operate(distinct_op)

    def any_(self) -> ColumnOperators:
        """Produce an :func:`_expression.any_` clause against the
        parent object.

        See the documentation for :func:`_sql.any_` for examples.

        .. note:: be sure to not confuse the newer
            :meth:`_sql.ColumnOperators.any_` method with the **legacy**
            version of this method, the :meth:`_types.ARRAY.Comparator.any`
            method that's specific to :class:`_types.ARRAY`, which uses a
            different calling style.

        """
        return self.operate(any_op)

    def all_(self) -> ColumnOperators:
        """Produce an :func:`_expression.all_` clause against the
        parent object.

        See the documentation for :func:`_sql.all_` for examples.

        .. note:: be sure to not confuse the newer
            :meth:`_sql.ColumnOperators.all_` method with the **legacy**
            version of this method, the :meth:`_types.ARRAY.Comparator.all`
            method that's specific to :class:`_types.ARRAY`, which uses a
            different calling style.

        """
        return self.operate(all_op)

    def __add__(self, other: Any) -> ColumnOperators:
        """Implement the ``+`` operator.

        In a column context, produces the clause ``a + b``
        if the parent object has non-string affinity.
        If the parent object has a string affinity,
        produces the concatenation operator, ``a || b`` -
        see :meth:`.ColumnOperators.concat`.

        """
        return self.operate(add, other)

    def __sub__(self, other: Any) -> ColumnOperators:
        """Implement the ``-`` operator.

        In a column context, produces the clause ``a - b``.

        """
        return self.operate(sub, other)

    def __mul__(self, other: Any) -> ColumnOperators:
        """Implement the ``*`` operator.

        In a column context, produces the clause ``a * b``.

        """
        return self.operate(mul, other)

    def __mod__(self, other: Any) -> ColumnOperators:
        """Implement the ``%`` operator.

        In a column context, produces the clause ``a % b``.

        """
        return self.operate(mod, other)

    def __truediv__(self, other: Any) -> ColumnOperators:
        """Implement the ``/`` operator.

        In a column context, produces the clause ``a / b``, and
        considers the result type to be numeric.

        .. versionchanged:: 2.0  The truediv operator against two integers
           is now considered to return a numeric value.    Behavior on specific
           backends may vary.

        """
        return self.operate(truediv, other)

    def __rtruediv__(self, other: Any) -> ColumnOperators:
        """Implement the ``/`` operator in reverse.

        See :meth:`.ColumnOperators.__truediv__`.

        """
        return self.reverse_operate(truediv, other)

    def __floordiv__(self, other: Any) -> ColumnOperators:
        """Implement the ``//`` operator.

        In a column context, produces the clause ``a / b``,
        which is the same as "truediv", but considers the result
        type to be integer.

        .. versionadded:: 2.0

        """
        return self.operate(floordiv, other)

    def __rfloordiv__(self, other: Any) -> ColumnOperators:
        """Implement the ``//`` operator in reverse.

        See :meth:`.ColumnOperators.__floordiv__`.

        """
        return self.reverse_operate(floordiv, other)


_commutative: Set[Any] = {eq, ne, add, mul}
_comparison: Set[Any] = {eq, ne, lt, gt, ge, le}


def _operator_fn(fn: Callable[..., Any]) -> OperatorType:
    return cast(OperatorType, fn)


def commutative_op(fn: _FN) -> _FN:
    _commutative.add(fn)
    return fn


def comparison_op(fn: _FN) -> _FN:
    _comparison.add(fn)
    return fn


@_operator_fn
def from_() -> Any:
    raise NotImplementedError()


@_operator_fn
@comparison_op
def function_as_comparison_op() -> Any:
    raise NotImplementedError()


@_operator_fn
def as_() -> Any:
    raise NotImplementedError()


@_operator_fn
def exists() -> Any:
    raise NotImplementedError()


@_operator_fn
def is_true(a: Any) -> Any:
    raise NotImplementedError()


# 1.4 deprecated; see #5435
if TYPE_CHECKING:

    @_operator_fn
    def istrue(a: Any) -> Any: ...

else:
    istrue = is_true


@_operator_fn
def is_false(a: Any) -> Any:
    raise NotImplementedError()


# 1.4 deprecated; see #5435
if TYPE_CHECKING:

    @_operator_fn
    def isfalse(a: Any) -> Any: ...

else:
    isfalse = is_false


@comparison_op
@_operator_fn
def is_distinct_from(a: Any, b: Any) -> Any:
    return a.is_distinct_from(b)


@comparison_op
@_operator_fn
def is_not_distinct_from(a: Any, b: Any) -> Any:
    return a.is_not_distinct_from(b)


# deprecated 1.4; see #5435
if TYPE_CHECKING:

    @_operator_fn
    def isnot_distinct_from(a: Any, b: Any) -> Any: ...

else:
    isnot_distinct_from = is_not_distinct_from


@comparison_op
@_operator_fn
def is_(a: Any, b: Any) -> Any:
    return a.is_(b)


@comparison_op
@_operator_fn
def is_not(a: Any, b: Any) -> Any:
    return a.is_not(b)


# 1.4 deprecated; see #5429
if TYPE_CHECKING:

    @_operator_fn
    def isnot(a: Any, b: Any) -> Any: ...

else:
    isnot = is_not


@_operator_fn
def collate(a: Any, b: Any) -> Any:
    return a.collate(b)


@_operator_fn
def op(a: Any, opstring: str, b: Any) -> Any:
    return a.op(opstring)(b)


@comparison_op
@_operator_fn
def like_op(a: Any, b: Any, escape: Optional[str] = None) -> Any:
    return a.like(b, escape=escape)


@comparison_op
@_operator_fn
def not_like_op(a: Any, b: Any, escape: Optional[str] = None) -> Any:
    return a.notlike(b, escape=escape)


# 1.4 deprecated; see #5435
if TYPE_CHECKING:

    @_operator_fn
    def notlike_op(a: Any, b: Any, escape: Optional[str] = None) -> Any: ...

else:
    notlike_op = not_like_op


@comparison_op
@_operator_fn
def ilike_op(a: Any, b: Any, escape: Optional[str] = None) -> Any:
    return a.ilike(b, escape=escape)


@comparison_op
@_operator_fn
def not_ilike_op(a: Any, b: Any, escape: Optional[str] = None) -> Any:
    return a.not_ilike(b, escape=escape)


# 1.4 deprecated; see #5435
if TYPE_CHECKING:

    @_operator_fn
    def notilike_op(a: Any, b: Any, escape: Optional[str] = None) -> Any: ...

else:
    notilike_op = not_ilike_op


@comparison_op
@_operator_fn
def between_op(a: Any, b: Any, c: Any, symmetric: bool = False) -> Any:
    return a.between(b, c, symmetric=symmetric)


@comparison_op
@_operator_fn
def not_between_op(a: Any, b: Any, c: Any, symmetric: bool = False) -> Any:
    return ~a.between(b, c, symmetric=symmetric)


# 1.4 deprecated; see #5435
if TYPE_CHECKING:

    @_operator_fn
    def notbetween_op(
        a: Any, b: Any, c: Any, symmetric: bool = False
    ) -> Any: ...

else:
    notbetween_op = not_between_op


@comparison_op
@_operator_fn
def in_op(a: Any, b: Any) -> Any:
    return a.in_(b)


@comparison_op
@_operator_fn
def not_in_op(a: Any, b: Any) -> Any:
    return a.not_in(b)


# 1.4 deprecated; see #5429
if TYPE_CHECKING:

    @_operator_fn
    def notin_op(a: Any, b: Any) -> Any: ...

else:
    notin_op = not_in_op


@_operator_fn
def distinct_op(a: Any) -> Any:
    return a.distinct()


@_operator_fn
def any_op(a: Any) -> Any:
    return a.any_()


@_operator_fn
def all_op(a: Any) -> Any:
    return a.all_()


def _escaped_like_impl(
    fn: Callable[..., Any], other: Any, escape: Optional[str], autoescape: bool
) -> Any:
    if autoescape:
        if autoescape is not True:
            util.warn(
                "The autoescape parameter is now a simple boolean True/False"
            )
        if escape is None:
            escape = "/"

        if not isinstance(other, str):
            raise TypeError("String value expected when autoescape=True")

        if escape not in ("%", "_"):
            other = other.replace(escape, escape + escape)

        other = other.replace("%", escape + "%").replace("_", escape + "_")

    return fn(other, escape=escape)


@comparison_op
@_operator_fn
def startswith_op(
    a: Any, b: Any, escape: Optional[str] = None, autoescape: bool = False
) -> Any:
    return _escaped_like_impl(a.startswith, b, escape, autoescape)


@comparison_op
@_operator_fn
def not_startswith_op(
    a: Any, b: Any, escape: Optional[str] = None, autoescape: bool = False
) -> Any:
    return ~_escaped_like_impl(a.startswith, b, escape, autoescape)


# 1.4 deprecated; see #5435
if TYPE_CHECKING:

    @_operator_fn
    def notstartswith_op(
        a: Any, b: Any, escape: Optional[str] = None, autoescape: bool = False
    ) -> Any: ...

else:
    notstartswith_op = not_startswith_op


@comparison_op
@_operator_fn
def istartswith_op(
    a: Any, b: Any, escape: Optional[str] = None, autoescape: bool = False
) -> Any:
    return _escaped_like_impl(a.istartswith, b, escape, autoescape)


@comparison_op
@_operator_fn
def not_istartswith_op(
    a: Any, b: Any, escape: Optional[str] = None, autoescape: bool = False
) -> Any:
    return ~_escaped_like_impl(a.istartswith, b, escape, autoescape)


@comparison_op
@_operator_fn
def endswith_op(
    a: Any, b: Any, escape: Optional[str] = None, autoescape: bool = False
) -> Any:
    return _escaped_like_impl(a.endswith, b, escape, autoescape)


@comparison_op
@_operator_fn
def not_endswith_op(
    a: Any, b: Any, escape: Optional[str] = None, autoescape: bool = False
) -> Any:
    return ~_escaped_like_impl(a.endswith, b, escape, autoescape)


# 1.4 deprecated; see #5435
if TYPE_CHECKING:

    @_operator_fn
    def notendswith_op(
        a: Any, b: Any, escape: Optional[str] = None, autoescape: bool = False
    ) -> Any: ...

else:
    notendswith_op = not_endswith_op


@comparison_op
@_operator_fn
def iendswith_op(
    a: Any, b: Any, escape: Optional[str] = None, autoescape: bool = False
) -> Any:
    return _escaped_like_impl(a.iendswith, b, escape, autoescape)


@comparison_op
@_operator_fn
def not_iendswith_op(
    a: Any, b: Any, escape: Optional[str] = None, autoescape: bool = False
) -> Any:
    return ~_escaped_like_impl(a.iendswith, b, escape, autoescape)


@comparison_op
@_operator_fn
def contains_op(
    a: Any, b: Any, escape: Optional[str] = None, autoescape: bool = False
) -> Any:
    return _escaped_like_impl(a.contains, b, escape, autoescape)


@comparison_op
@_operator_fn
def not_contains_op(
    a: Any, b: Any, escape: Optional[str] = None, autoescape: bool = False
) -> Any:
    return ~_escaped_like_impl(a.contains, b, escape, autoescape)


# 1.4 deprecated; see #5435
if TYPE_CHECKING:

    @_operator_fn
    def notcontains_op(
        a: Any, b: Any, escape: Optional[str] = None, autoescape: bool = False
    ) -> Any: ...

else:
    notcontains_op = not_contains_op


@comparison_op
@_operator_fn
def icontains_op(
    a: Any, b: Any, escape: Optional[str] = None, autoescape: bool = False
) -> Any:
    return _escaped_like_impl(a.icontains, b, escape, autoescape)


@comparison_op
@_operator_fn
def not_icontains_op(
    a: Any, b: Any, escape: Optional[str] = None, autoescape: bool = False
) -> Any:
    return ~_escaped_like_impl(a.icontains, b, escape, autoescape)


@comparison_op
@_operator_fn
def match_op(a: Any, b: Any, **kw: Any) -> Any:
    return a.match(b, **kw)


@comparison_op
@_operator_fn
def regexp_match_op(a: Any, b: Any, flags: Optional[str] = None) -> Any:
    return a.regexp_match(b, flags=flags)


@comparison_op
@_operator_fn
def not_regexp_match_op(a: Any, b: Any, flags: Optional[str] = None) -> Any:
    return ~a.regexp_match(b, flags=flags)


@_operator_fn
def regexp_replace_op(
    a: Any, b: Any, replacement: Any, flags: Optional[str] = None
) -> Any:
    return a.regexp_replace(b, replacement=replacement, flags=flags)


@comparison_op
@_operator_fn
def not_match_op(a: Any, b: Any, **kw: Any) -> Any:
    return ~a.match(b, **kw)


# 1.4 deprecated; see #5429
if TYPE_CHECKING:

    @_operator_fn
    def notmatch_op(a: Any, b: Any, **kw: Any) -> Any: ...

else:
    notmatch_op = not_match_op


@_operator_fn
def comma_op(a: Any, b: Any) -> Any:
    raise NotImplementedError()


@_operator_fn
def filter_op(a: Any, b: Any) -> Any:
    raise NotImplementedError()


@_operator_fn
def concat_op(a: Any, b: Any) -> Any:
    try:
        concat = a.concat
    except AttributeError:
        return b._rconcat(a)
    else:
        return concat(b)


@_operator_fn
def desc_op(a: Any) -> Any:
    return a.desc()


@_operator_fn
def asc_op(a: Any) -> Any:
    return a.asc()


@_operator_fn
def nulls_first_op(a: Any) -> Any:
    return a.nulls_first()


# 1.4 deprecated; see #5435
if TYPE_CHECKING:

    @_operator_fn
    def nullsfirst_op(a: Any) -> Any: ...

else:
    nullsfirst_op = nulls_first_op


@_operator_fn
def nulls_last_op(a: Any) -> Any:
    return a.nulls_last()


# 1.4 deprecated; see #5435
if TYPE_CHECKING:

    @_operator_fn
    def nullslast_op(a: Any) -> Any: ...

else:
    nullslast_op = nulls_last_op


@_operator_fn
def json_getitem_op(a: Any, b: Any) -> Any:
    raise NotImplementedError()


@_operator_fn
def json_path_getitem_op(a: Any, b: Any) -> Any:
    raise NotImplementedError()


@_operator_fn
def bitwise_xor_op(a: Any, b: Any) -> Any:
    return a.bitwise_xor(b)


@_operator_fn
def bitwise_or_op(a: Any, b: Any) -> Any:
    return a.bitwise_or(b)


@_operator_fn
def bitwise_and_op(a: Any, b: Any) -> Any:
    return a.bitwise_and(b)


@_operator_fn
def bitwise_not_op(a: Any) -> Any:
    return a.bitwise_not()


@_operator_fn
def bitwise_lshift_op(a: Any, b: Any) -> Any:
    return a.bitwise_lshift(b)


@_operator_fn
def bitwise_rshift_op(a: Any, b: Any) -> Any:
    return a.bitwise_rshift(b)


def is_comparison(op: OperatorType) -> bool:
    return op in _comparison or isinstance(op, custom_op) and op.is_comparison


def is_commutative(op: OperatorType) -> bool:
    return op in _commutative


def is_ordering_modifier(op: OperatorType) -> bool:
    return op in (asc_op, desc_op, nulls_first_op, nulls_last_op)


def is_natural_self_precedent(op: OperatorType) -> bool:
    return (
        op in _natural_self_precedent
        or isinstance(op, custom_op)
        and op.natural_self_precedent
    )


_booleans = (inv, is_true, is_false, and_, or_)


def is_boolean(op: OperatorType) -> bool:
    return is_comparison(op) or op in _booleans


_mirror = {gt: lt, ge: le, lt: gt, le: ge}


def mirror(op: OperatorType) -> OperatorType:
    """rotate a comparison operator 180 degrees.

    Note this is not the same as negation.

    """
    return _mirror.get(op, op)


_associative = _commutative.union([concat_op, and_, or_]).difference([eq, ne])


def is_associative(op: OperatorType) -> bool:
    return op in _associative


def is_order_by_modifier(op: Optional[OperatorType]) -> bool:
    return op in _order_by_modifier


_order_by_modifier = {desc_op, asc_op, nulls_first_op, nulls_last_op}

_natural_self_precedent = _associative.union(
    [getitem, json_getitem_op, json_path_getitem_op]
)
"""Operators where if we have (a op b) op c, we don't want to
parenthesize (a op b).

"""


@_operator_fn
def _asbool(a: Any) -> Any:
    raise NotImplementedError()


class _OpLimit(IntEnum):
    _smallest = -100
    _largest = 100


_PRECEDENCE: Dict[OperatorType, int] = {
    from_: 15,
    function_as_comparison_op: 15,
    any_op: 15,
    all_op: 15,
    getitem: 15,
    json_getitem_op: 15,
    json_path_getitem_op: 15,
    mul: 8,
    truediv: 8,
    floordiv: 8,
    mod: 8,
    neg: 8,
    bitwise_not_op: 8,
    add: 7,
    sub: 7,
    bitwise_xor_op: 7,
    bitwise_or_op: 7,
    bitwise_and_op: 7,
    bitwise_lshift_op: 7,
    bitwise_rshift_op: 7,
    filter_op: 6,
    concat_op: 5,
    match_op: 5,
    not_match_op: 5,
    regexp_match_op: 5,
    not_regexp_match_op: 5,
    regexp_replace_op: 5,
    ilike_op: 5,
    not_ilike_op: 5,
    like_op: 5,
    not_like_op: 5,
    in_op: 5,
    not_in_op: 5,
    is_: 5,
    is_not: 5,
    eq: 5,
    ne: 5,
    is_distinct_from: 5,
    is_not_distinct_from: 5,
    gt: 5,
    lt: 5,
    ge: 5,
    le: 5,
    between_op: 5,
    not_between_op: 5,
    distinct_op: 5,
    inv: 5,
    is_true: 5,
    is_false: 5,
    and_: 3,
    or_: 2,
    comma_op: -1,
    desc_op: 3,
    asc_op: 3,
    collate: 4,
    as_: -1,
    exists: 0,
    _asbool: -10,
}


def is_precedent(
    operator: OperatorType, against: Optional[OperatorType]
) -> bool:
    if operator is against and is_natural_self_precedent(operator):
        return False
    elif against is None:
        return True
    else:
        return bool(
            _PRECEDENCE.get(
                operator, getattr(operator, "precedence", _OpLimit._smallest)
            )
            <= _PRECEDENCE.get(
                against, getattr(against, "precedence", _OpLimit._largest)
            )
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\base.py ===
"""
An interface for extending pandas with custom arrays.

.. warning::

   This is an experimental API and subject to breaking changes
   without warning.
"""
from __future__ import annotations

import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Literal,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    algos as libalgos,
    lib,
)
from pandas.compat import set_function_name
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
    Appender,
    Substitution,
    cache_readonly,
)
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
    validate_bool_kwarg,
    validate_fillna_kwargs,
    validate_insert_loc,
)

from pandas.core.dtypes.cast import maybe_cast_pointwise_result
from pandas.core.dtypes.common import (
    is_list_like,
    is_scalar,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna

from pandas.core import (
    arraylike,
    missing,
    roperator,
)
from pandas.core.algorithms import (
    duplicated,
    factorize_array,
    isin,
    map_array,
    mode,
    rank,
    unique,
)
from pandas.core.array_algos.quantile import quantile_with_mask
from pandas.core.missing import _fill_limit_area_1d
from pandas.core.sorting import (
    nargminmax,
    nargsort,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        AstypeArg,
        AxisInt,
        Dtype,
        DtypeObj,
        FillnaOptions,
        InterpolateOptions,
        NumpySorter,
        NumpyValueArrayLike,
        PositionalIndexer,
        ScalarIndexer,
        Self,
        SequenceIndexer,
        Shape,
        SortKind,
        TakeIndexer,
        npt,
    )

    from pandas import Index

_extension_array_shared_docs: dict[str, str] = {}


class ExtensionArray:
    """
    Abstract base class for custom 1-D array types.

    pandas will recognize instances of this class as proper arrays
    with a custom type and will not attempt to coerce them to objects. They
    may be stored directly inside a :class:`DataFrame` or :class:`Series`.

    Attributes
    ----------
    dtype
    nbytes
    ndim
    shape

    Methods
    -------
    argsort
    astype
    copy
    dropna
    duplicated
    factorize
    fillna
    equals
    insert
    interpolate
    isin
    isna
    ravel
    repeat
    searchsorted
    shift
    take
    tolist
    unique
    view
    _accumulate
    _concat_same_type
    _explode
    _formatter
    _from_factorized
    _from_sequence
    _from_sequence_of_strings
    _hash_pandas_object
    _pad_or_backfill
    _reduce
    _values_for_argsort
    _values_for_factorize

    Notes
    -----
    The interface includes the following abstract methods that must be
    implemented by subclasses:

    * _from_sequence
    * _from_factorized
    * __getitem__
    * __len__
    * __eq__
    * dtype
    * nbytes
    * isna
    * take
    * copy
    * _concat_same_type
    * interpolate

    A default repr displaying the type, (truncated) data, length,
    and dtype is provided. It can be customized or replaced by
    by overriding:

    * __repr__ : A default repr for the ExtensionArray.
    * _formatter : Print scalars inside a Series or DataFrame.

    Some methods require casting the ExtensionArray to an ndarray of Python
    objects with ``self.astype(object)``, which may be expensive. When
    performance is a concern, we highly recommend overriding the following
    methods:

    * fillna
    * _pad_or_backfill
    * dropna
    * unique
    * factorize / _values_for_factorize
    * argsort, argmax, argmin / _values_for_argsort
    * searchsorted
    * map

    The remaining methods implemented on this class should be performant,
    as they only compose abstract methods. Still, a more efficient
    implementation may be available, and these methods can be overridden.

    One can implement methods to handle array accumulations or reductions.

    * _accumulate
    * _reduce

    One can implement methods to handle parsing from strings that will be used
    in methods such as ``pandas.io.parsers.read_csv``.

    * _from_sequence_of_strings

    This class does not inherit from 'abc.ABCMeta' for performance reasons.
    Methods and properties required by the interface raise
    ``pandas.errors.AbstractMethodError`` and no ``register`` method is
    provided for registering virtual subclasses.

    ExtensionArrays are limited to 1 dimension.

    They may be backed by none, one, or many NumPy arrays. For example,
    ``pandas.Categorical`` is an extension array backed by two arrays,
    one for codes and one for categories. An array of IPv6 address may
    be backed by a NumPy structured array with two fields, one for the
    lower 64 bits and one for the upper 64 bits. Or they may be backed
    by some other storage type, like Python lists. Pandas makes no
    assumptions on how the data are stored, just that it can be converted
    to a NumPy array.
    The ExtensionArray interface does not impose any rules on how this data
    is stored. However, currently, the backing data cannot be stored in
    attributes called ``.values`` or ``._values`` to ensure full compatibility
    with pandas internals. But other names as ``.data``, ``._data``,
    ``._items``, ... can be freely used.

    If implementing NumPy's ``__array_ufunc__`` interface, pandas expects
    that

    1. You defer by returning ``NotImplemented`` when any Series are present
       in `inputs`. Pandas will extract the arrays and call the ufunc again.
    2. You define a ``_HANDLED_TYPES`` tuple as an attribute on the class.
       Pandas inspect this to determine whether the ufunc is valid for the
       types present.

    See :ref:`extending.extension.ufunc` for more.

    By default, ExtensionArrays are not hashable.  Immutable subclasses may
    override this behavior.

    Examples
    --------
    Please see the following:

    https://github.com/pandas-dev/pandas/blob/main/pandas/tests/extension/list/array.py
    """

    # '_typ' is for pandas.core.dtypes.generic.ABCExtensionArray.
    # Don't override this.
    _typ = "extension"

    # similar to __array_priority__, positions ExtensionArray after Index,
    #  Series, and DataFrame.  EA subclasses may override to choose which EA
    #  subclass takes priority. If overriding, the value should always be
    #  strictly less than 2000 to be below Index.__pandas_priority__.
    __pandas_priority__ = 1000

    # ------------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------------

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy: bool = False):
        """
        Construct a new ExtensionArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type`` or be converted into this type in this method.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray

        Examples
        --------
        >>> pd.arrays.IntegerArray._from_sequence([4, 5])
        <IntegerArray>
        [4, 5]
        Length: 2, dtype: Int64
        """
        raise AbstractMethodError(cls)

    @classmethod
    def _from_scalars(cls, scalars, *, dtype: DtypeObj) -> Self:
        """
        Strict analogue to _from_sequence, allowing only sequences of scalars
        that should be specifically inferred to the given dtype.

        Parameters
        ----------
        scalars : sequence
        dtype : ExtensionDtype

        Raises
        ------
        TypeError or ValueError

        Notes
        -----
        This is called in a try/except block when casting the result of a
        pointwise operation.
        """
        try:
            return cls._from_sequence(scalars, dtype=dtype, copy=False)
        except (ValueError, TypeError):
            raise
        except Exception:
            warnings.warn(
                "_from_scalars should only raise ValueError or TypeError. "
                "Consider overriding _from_scalars where appropriate.",
                stacklevel=find_stack_level(),
            )
            raise

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: Dtype | None = None, copy: bool = False
    ):
        """
        Construct a new ExtensionArray from a sequence of strings.

        Parameters
        ----------
        strings : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray

        Examples
        --------
        >>> pd.arrays.IntegerArray._from_sequence_of_strings(["1", "2", "3"])
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        raise AbstractMethodError(cls)

    @classmethod
    def _from_factorized(cls, values, original):
        """
        Reconstruct an ExtensionArray after factorization.

        Parameters
        ----------
        values : ndarray
            An integer ndarray with the factorized values.
        original : ExtensionArray
            The original ExtensionArray that factorize was called on.

        See Also
        --------
        factorize : Top-level factorize method that dispatches here.
        ExtensionArray.factorize : Encode the extension array as an enumerated type.

        Examples
        --------
        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1),
        ...                                      pd.Interval(1, 5), pd.Interval(1, 5)])
        >>> codes, uniques = pd.factorize(interv_arr)
        >>> pd.arrays.IntervalArray._from_factorized(uniques, interv_arr)
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        """
        raise AbstractMethodError(cls)

    # ------------------------------------------------------------------------
    # Must be a Sequence
    # ------------------------------------------------------------------------
    @overload
    def __getitem__(self, item: ScalarIndexer) -> Any:
        ...

    @overload
    def __getitem__(self, item: SequenceIndexer) -> Self:
        ...

    def __getitem__(self, item: PositionalIndexer) -> Self | Any:
        """
        Select a subset of self.

        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.

            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None

            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'

            * list[int]:  A list of int

        Returns
        -------
        item : scalar or ExtensionArray

        Notes
        -----
        For scalar ``item``, return a scalar value suitable for the array's
        type. This should be an instance of ``self.dtype.type``.

        For slice ``key``, return an instance of ``ExtensionArray``, even
        if the slice is length 0 or 1.

        For a boolean mask, return an instance of ``ExtensionArray``, filtered
        to the values where ``item`` is True.
        """
        raise AbstractMethodError(self)

    def __setitem__(self, key, value) -> None:
        """
        Set one or more values inplace.

        This method is not required to satisfy the pandas extension array
        interface.

        Parameters
        ----------
        key : int, ndarray, or slice
            When called from, e.g. ``Series.__setitem__``, ``key`` will be
            one of

            * scalar int
            * ndarray of integers.
            * boolean ndarray
            * slice object

        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object
            value or values to be set of ``key``.

        Returns
        -------
        None
        """
        # Some notes to the ExtensionArray implementer who may have ended up
        # here. While this method is not required for the interface, if you
        # *do* choose to implement __setitem__, then some semantics should be
        # observed:
        #
        # * Setting multiple values : ExtensionArrays should support setting
        #   multiple values at once, 'key' will be a sequence of integers and
        #  'value' will be a same-length sequence.
        #
        # * Broadcasting : For a sequence 'key' and a scalar 'value',
        #   each position in 'key' should be set to 'value'.
        #
        # * Coercion : Most users will expect basic coercion to work. For
        #   example, a string like '2018-01-01' is coerced to a datetime
        #   when setting on a datetime64ns array. In general, if the
        #   __init__ method coerces that value, then so should __setitem__
        # Note, also, that Series/DataFrame.where internally use __setitem__
        # on a copy of the data.
        raise NotImplementedError(f"{type(self)} does not implement __setitem__.")

    def __len__(self) -> int:
        """
        Length of this array

        Returns
        -------
        length : int
        """
        raise AbstractMethodError(self)

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over elements of the array.
        """
        # This needs to be implemented so that pandas recognizes extension
        # arrays as list-like. The default implementation makes successive
        # calls to ``__getitem__``, which may be slower than necessary.
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, item: object) -> bool | np.bool_:
        """
        Return for `item in self`.
        """
        # GH37867
        # comparisons of any item to pd.NA always return pd.NA, so e.g. "a" in [pd.NA]
        # would raise a TypeError. The implementation below works around that.
        if is_scalar(item) and isna(item):
            if not self._can_hold_na:
                return False
            elif item is self.dtype.na_value or isinstance(item, self.dtype.type):
                return self._hasna
            else:
                return False
        else:
            # error: Item "ExtensionArray" of "Union[ExtensionArray, ndarray]" has no
            # attribute "any"
            return (item == self).any()  # type: ignore[union-attr]

    # error: Signature of "__eq__" incompatible with supertype "object"
    def __eq__(self, other: object) -> ArrayLike:  # type: ignore[override]
        """
        Return for `self == other` (element-wise equality).
        """
        # Implementer note: this should return a boolean numpy ndarray or
        # a boolean ExtensionArray.
        # When `other` is one of Series, Index, or DataFrame, this method should
        # return NotImplemented (to ensure that those objects are responsible for
        # first unpacking the arrays, and then dispatch the operation to the
        # underlying arrays)
        raise AbstractMethodError(self)

    # error: Signature of "__ne__" incompatible with supertype "object"
    def __ne__(self, other: object) -> ArrayLike:  # type: ignore[override]
        """
        Return for `self != other` (element-wise in-equality).
        """
        # error: Unsupported operand type for ~ ("ExtensionArray")
        return ~(self == other)  # type: ignore[operator]

    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        """
        Convert to a NumPy ndarray.

        This is similar to :meth:`numpy.asarray`, but may provide additional control
        over how the conversion is done.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.
        na_value : Any, optional
            The value to use for missing values. The default value depends
            on `dtype` and the type of the array.

        Returns
        -------
        numpy.ndarray
        """
        result = np.asarray(self, dtype=dtype)
        if copy or na_value is not lib.no_default:
            result = result.copy()
        if na_value is not lib.no_default:
            result[self.isna()] = na_value
        return result

    # ------------------------------------------------------------------------
    # Required attributes
    # ------------------------------------------------------------------------

    @property
    def dtype(self) -> ExtensionDtype:
        """
        An instance of ExtensionDtype.

        Examples
        --------
        >>> pd.array([1, 2, 3]).dtype
        Int64Dtype()
        """
        raise AbstractMethodError(self)

    @property
    def shape(self) -> Shape:
        """
        Return a tuple of the array dimensions.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.shape
        (3,)
        """
        return (len(self),)

    @property
    def size(self) -> int:
        """
        The number of elements in the array.
        """
        # error: Incompatible return value type (got "signedinteger[_64Bit]",
        # expected "int")  [return-value]
        return np.prod(self.shape)  # type: ignore[return-value]

    @property
    def ndim(self) -> int:
        """
        Extension Arrays are only allowed to be 1-dimensional.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.ndim
        1
        """
        return 1

    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.

        Examples
        --------
        >>> pd.array([1, 2, 3]).nbytes
        27
        """
        # If this is expensive to compute, return an approximate lower bound
        # on the number of bytes needed.
        raise AbstractMethodError(self)

    # ------------------------------------------------------------------------
    # Additional Methods
    # ------------------------------------------------------------------------

    @overload
    def astype(self, dtype: npt.DTypeLike, copy: bool = ...) -> np.ndarray:
        ...

    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = ...) -> ExtensionArray:
        ...

    @overload
    def astype(self, dtype: AstypeArg, copy: bool = ...) -> ArrayLike:
        ...

    def astype(self, dtype: AstypeArg, copy: bool = True) -> ArrayLike:
        """
        Cast to a NumPy array or ExtensionArray with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        np.ndarray or pandas.api.extensions.ExtensionArray
            An ``ExtensionArray`` if ``dtype`` is ``ExtensionDtype``,
            otherwise a Numpy ndarray with ``dtype`` for its dtype.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64

        Casting to another ``ExtensionDtype`` returns an ``ExtensionArray``:

        >>> arr1 = arr.astype('Float64')
        >>> arr1
        <FloatingArray>
        [1.0, 2.0, 3.0]
        Length: 3, dtype: Float64
        >>> arr1.dtype
        Float64Dtype()

        Otherwise, we will get a Numpy ndarray:

        >>> arr2 = arr.astype('float64')
        >>> arr2
        array([1., 2., 3.])
        >>> arr2.dtype
        dtype('float64')
        """
        dtype = pandas_dtype(dtype)
        if dtype == self.dtype:
            if not copy:
                return self
            else:
                return self.copy()

        if isinstance(dtype, ExtensionDtype):
            cls = dtype.construct_array_type()
            return cls._from_sequence(self, dtype=dtype, copy=copy)

        elif lib.is_np_dtype(dtype, "M"):
            from pandas.core.arrays import DatetimeArray

            return DatetimeArray._from_sequence(self, dtype=dtype, copy=copy)

        elif lib.is_np_dtype(dtype, "m"):
            from pandas.core.arrays import TimedeltaArray

            return TimedeltaArray._from_sequence(self, dtype=dtype, copy=copy)

        if not copy:
            return np.asarray(self, dtype=dtype)
        else:
            return np.array(self, dtype=dtype, copy=copy)

    def isna(self) -> np.ndarray | ExtensionArraySupportsAnyAll:
        """
        A 1-D array indicating if each value is missing.

        Returns
        -------
        numpy.ndarray or pandas.api.extensions.ExtensionArray
            In most cases, this should return a NumPy ndarray. For
            exceptional cases like ``SparseArray``, where returning
            an ndarray would be expensive, an ExtensionArray may be
            returned.

        Notes
        -----
        If returning an ExtensionArray, then

        * ``na_values._is_boolean`` should be True
        * `na_values` should implement :func:`ExtensionArray._reduce`
        * ``na_values.any`` and ``na_values.all`` should be implemented

        Examples
        --------
        >>> arr = pd.array([1, 2, np.nan, np.nan])
        >>> arr.isna()
        array([False, False,  True,  True])
        """
        raise AbstractMethodError(self)

    @property
    def _hasna(self) -> bool:
        # GH#22680
        """
        Equivalent to `self.isna().any()`.

        Some ExtensionArray subclasses may be able to optimize this check.
        """
        return bool(self.isna().any())

    def _values_for_argsort(self) -> np.ndarray:
        """
        Return values for sorting.

        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.

        See Also
        --------
        ExtensionArray.argsort : Return the indices that would sort this array.

        Notes
        -----
        The caller is responsible for *not* modifying these values in-place, so
        it is safe for implementers to give views on ``self``.

        Functions that use this (e.g. ``ExtensionArray.argsort``) should ignore
        entries with missing values in the original array (according to
        ``self.isna()``). This means that the corresponding entries in the returned
        array don't need to be modified to sort correctly.

        Examples
        --------
        In most cases, this is the underlying Numpy array of the ``ExtensionArray``:

        >>> arr = pd.array([1, 2, 3])
        >>> arr._values_for_argsort()
        array([1, 2, 3])
        """
        # Note: this is used in `ExtensionArray.argsort/argmin/argmax`.
        return np.array(self)

    def argsort(
        self,
        *,
        ascending: bool = True,
        kind: SortKind = "quicksort",
        na_position: str = "last",
        **kwargs,
    ) -> np.ndarray:
        """
        Return the indices that would sort this array.

        Parameters
        ----------
        ascending : bool, default True
            Whether the indices should result in an ascending
            or descending sort.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            Sorting algorithm.
        na_position : {'first', 'last'}, default 'last'
            If ``'first'``, put ``NaN`` values at the beginning.
            If ``'last'``, put ``NaN`` values at the end.
        *args, **kwargs:
            Passed through to :func:`numpy.argsort`.

        Returns
        -------
        np.ndarray[np.intp]
            Array of indices that sort ``self``. If NaN values are contained,
            NaN values are placed at the end.

        See Also
        --------
        numpy.argsort : Sorting implementation used internally.

        Examples
        --------
        >>> arr = pd.array([3, 1, 2, 5, 4])
        >>> arr.argsort()
        array([1, 2, 0, 4, 3])
        """
        # Implementer note: You have two places to override the behavior of
        # argsort.
        # 1. _values_for_argsort : construct the values passed to np.argsort
        # 2. argsort : total control over sorting. In case of overriding this,
        #    it is recommended to also override argmax/argmin
        ascending = nv.validate_argsort_with_ascending(ascending, (), kwargs)

        values = self._values_for_argsort()
        return nargsort(
            values,
            kind=kind,
            ascending=ascending,
            na_position=na_position,
            mask=np.asarray(self.isna()),
        )

    def argmin(self, skipna: bool = True) -> int:
        """
        Return the index of minimum value.

        In case of multiple occurrences of the minimum value, the index
        corresponding to the first occurrence is returned.

        Parameters
        ----------
        skipna : bool, default True

        Returns
        -------
        int

        See Also
        --------
        ExtensionArray.argmax : Return the index of the maximum value.

        Examples
        --------
        >>> arr = pd.array([3, 1, 2, 5, 4])
        >>> arr.argmin()
        1
        """
        # Implementer note: You have two places to override the behavior of
        # argmin.
        # 1. _values_for_argsort : construct the values used in nargminmax
        # 2. argmin itself : total control over sorting.
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise NotImplementedError
        return nargminmax(self, "argmin")

    def argmax(self, skipna: bool = True) -> int:
        """
        Return the index of maximum value.

        In case of multiple occurrences of the maximum value, the index
        corresponding to the first occurrence is returned.

        Parameters
        ----------
        skipna : bool, default True

        Returns
        -------
        int

        See Also
        --------
        ExtensionArray.argmin : Return the index of the minimum value.

        Examples
        --------
        >>> arr = pd.array([3, 1, 2, 5, 4])
        >>> arr.argmax()
        3
        """
        # Implementer note: You have two places to override the behavior of
        # argmax.
        # 1. _values_for_argsort : construct the values used in nargminmax
        # 2. argmax itself : total control over sorting.
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise NotImplementedError
        return nargminmax(self, "argmax")

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
        See DataFrame.interpolate.__doc__.

        Examples
        --------
        >>> arr = pd.arrays.NumpyExtensionArray(np.array([0, 1, np.nan, 3]))
        >>> arr.interpolate(method="linear",
        ...                 limit=3,
        ...                 limit_direction="forward",
        ...                 index=pd.Index([1, 2, 3, 4]),
        ...                 fill_value=1,
        ...                 copy=False,
        ...                 axis=0,
        ...                 limit_area="inside"
        ...                 )
        <NumpyExtensionArray>
        [0.0, 1.0, 2.0, 3.0]
        Length: 4, dtype: float64
        """
        # NB: we return type(self) even if copy=False
        raise NotImplementedError(
            f"{type(self).__name__} does not implement interpolate"
        )

    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        """
        Pad or backfill values, used by Series/DataFrame ffill and bfill.

        Parameters
        ----------
        method : {'backfill', 'bfill', 'pad', 'ffill'}
            Method to use for filling holes in reindexed Series:

            * pad / ffill: propagate last valid observation forward to next valid.
            * backfill / bfill: use NEXT valid observation to fill gap.

        limit : int, default None
            This is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.

        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author's discretion whether to ignore "copy=False" or to raise.
            The base class implementation ignores the keyword if any NAs are
            present.

        Returns
        -------
        Same type as self

        Examples
        --------
        >>> arr = pd.array([np.nan, np.nan, 2, 3, np.nan, np.nan])
        >>> arr._pad_or_backfill(method="backfill", limit=1)
        <IntegerArray>
        [<NA>, 2, 2, 3, <NA>, <NA>]
        Length: 6, dtype: Int64
        """

        # If a 3rd-party EA has implemented this functionality in fillna,
        #  we warn that they need to implement _pad_or_backfill instead.
        if (
            type(self).fillna is not ExtensionArray.fillna
            and type(self)._pad_or_backfill is ExtensionArray._pad_or_backfill
        ):
            # Check for _pad_or_backfill here allows us to call
            #  super()._pad_or_backfill without getting this warning
            warnings.warn(
                "ExtensionArray.fillna 'method' keyword is deprecated. "
                "In a future version. arr._pad_or_backfill will be called "
                "instead. 3rd-party ExtensionArray authors need to implement "
                "_pad_or_backfill.",
                DeprecationWarning,
                stacklevel=find_stack_level(),
            )
            if limit_area is not None:
                raise NotImplementedError(
                    f"{type(self).__name__} does not implement limit_area "
                    "(added in pandas 2.2). 3rd-party ExtnsionArray authors "
                    "need to add this argument to _pad_or_backfill."
                )
            return self.fillna(method=method, limit=limit)

        mask = self.isna()

        if mask.any():
            # NB: the base class does not respect the "copy" keyword
            meth = missing.clean_fill_method(method)

            npmask = np.asarray(mask)
            if limit_area is not None and not npmask.all():
                _fill_limit_area_1d(npmask, limit_area)
            if meth == "pad":
                indexer = libalgos.get_fill_indexer(npmask, limit=limit)
                return self.take(indexer, allow_fill=True)
            else:
                # i.e. meth == "backfill"
                indexer = libalgos.get_fill_indexer(npmask[::-1], limit=limit)[::-1]
                return self[::-1].take(indexer, allow_fill=True)

        else:
            if not copy:
                return self
            new_values = self.copy()
        return new_values

    def fillna(
        self,
        value: object | ArrayLike | None = None,
        method: FillnaOptions | None = None,
        limit: int | None = None,
        copy: bool = True,
    ) -> Self:
        """
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like "value" can be given. It's expected
            that the array-like have the same length as 'self'.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series:

            * pad / ffill: propagate last valid observation forward to next valid.
            * backfill / bfill: use NEXT valid observation to fill gap.

            .. deprecated:: 2.1.0

        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.

            .. deprecated:: 2.1.0

        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author's discretion whether to ignore "copy=False" or to raise.
            The base class implementation ignores the keyword in pad/backfill
            cases.

        Returns
        -------
        ExtensionArray
            With NA/NaN filled.

        Examples
        --------
        >>> arr = pd.array([np.nan, np.nan, 2, 3, np.nan, np.nan])
        >>> arr.fillna(0)
        <IntegerArray>
        [0, 0, 2, 3, 0, 0]
        Length: 6, dtype: Int64
        """
        if method is not None:
            warnings.warn(
                f"The 'method' keyword in {type(self).__name__}.fillna is "
                "deprecated and will be removed in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        value, method = validate_fillna_kwargs(value, method)

        mask = self.isna()
        # error: Argument 2 to "check_value_size" has incompatible type
        # "ExtensionArray"; expected "ndarray"
        value = missing.check_value_size(
            value, mask, len(self)  # type: ignore[arg-type]
        )

        if mask.any():
            if method is not None:
                meth = missing.clean_fill_method(method)

                npmask = np.asarray(mask)
                if meth == "pad":
                    indexer = libalgos.get_fill_indexer(npmask, limit=limit)
                    return self.take(indexer, allow_fill=True)
                else:
                    # i.e. meth == "backfill"
                    indexer = libalgos.get_fill_indexer(npmask[::-1], limit=limit)[::-1]
                    return self[::-1].take(indexer, allow_fill=True)
            else:
                # fill with value
                if not copy:
                    new_values = self[:]
                else:
                    new_values = self.copy()
                new_values[mask] = value
        else:
            if not copy:
                new_values = self[:]
            else:
                new_values = self.copy()
        return new_values

    def dropna(self) -> Self:
        """
        Return ExtensionArray without NA values.

        Returns
        -------

        Examples
        --------
        >>> pd.array([1, 2, np.nan]).dropna()
        <IntegerArray>
        [1, 2]
        Length: 2, dtype: Int64
        """
        # error: Unsupported operand type for ~ ("ExtensionArray")
        return self[~self.isna()]  # type: ignore[operator]

    def duplicated(
        self, keep: Literal["first", "last", False] = "first"
    ) -> npt.NDArray[np.bool_]:
        """
        Return boolean ndarray denoting duplicate values.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            - ``first`` : Mark duplicates as ``True`` except for the first occurrence.
            - ``last`` : Mark duplicates as ``True`` except for the last occurrence.
            - False : Mark all duplicates as ``True``.

        Returns
        -------
        ndarray[bool]

        Examples
        --------
        >>> pd.array([1, 1, 2, 3, 3], dtype="Int64").duplicated()
        array([False,  True, False, False,  True])
        """
        mask = self.isna().astype(np.bool_, copy=False)
        return duplicated(values=self, keep=keep, mask=mask)

    def shift(self, periods: int = 1, fill_value: object = None) -> ExtensionArray:
        """
        Shift values by desired number.

        Newly introduced missing values are filled with
        ``self.dtype.na_value``.

        Parameters
        ----------
        periods : int, default 1
            The number of periods to shift. Negative values are allowed
            for shifting backwards.

        fill_value : object, optional
            The scalar value to use for newly introduced missing values.
            The default is ``self.dtype.na_value``.

        Returns
        -------
        ExtensionArray
            Shifted.

        Notes
        -----
        If ``self`` is empty or ``periods`` is 0, a copy of ``self`` is
        returned.

        If ``periods > len(self)``, then an array of size
        len(self) is returned, with all values filled with
        ``self.dtype.na_value``.

        For 2-dimensional ExtensionArrays, we are always shifting along axis=0.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.shift(2)
        <IntegerArray>
        [<NA>, <NA>, 1]
        Length: 3, dtype: Int64
        """
        # Note: this implementation assumes that `self.dtype.na_value` can be
        # stored in an instance of your ExtensionArray with `self.dtype`.
        if not len(self) or periods == 0:
            return self.copy()

        if isna(fill_value):
            fill_value = self.dtype.na_value

        empty = self._from_sequence(
            [fill_value] * min(abs(periods), len(self)), dtype=self.dtype
        )
        if periods > 0:
            a = empty
            b = self[:-periods]
        else:
            a = self[abs(periods) :]
            b = empty
        return self._concat_same_type([a, b])

    def unique(self) -> Self:
        """
        Compute the ExtensionArray of unique values.

        Returns
        -------
        pandas.api.extensions.ExtensionArray

        Examples
        --------
        >>> arr = pd.array([1, 2, 3, 1, 2, 3])
        >>> arr.unique()
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        uniques = unique(self.astype(object))
        return self._from_sequence(uniques, dtype=self.dtype)

    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted array `self` (a) such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.

        Assuming that `self` is sorted:

        ======  ================================
        `side`  returned index `i` satisfies
        ======  ================================
        left    ``self[i-1] < value <= self[i]``
        right   ``self[i-1] <= value < self[i]``
        ======  ================================

        Parameters
        ----------
        value : array-like, list or scalar
            Value(s) to insert into `self`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array-like, optional
            Optional array of integer indices that sort array a into ascending
            order. They are typically the result of argsort.

        Returns
        -------
        array of ints or int
            If value is array-like, array of insertion points.
            If value is scalar, a single integer.

        See Also
        --------
        numpy.searchsorted : Similar method from NumPy.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3, 5])
        >>> arr.searchsorted([4])
        array([3])
        """
        # Note: the base tests provided by pandas only test the basics.
        # We do not test
        # 1. Values outside the range of the `data_for_sorting` fixture
        # 2. Values between the values in the `data_for_sorting` fixture
        # 3. Missing values.
        arr = self.astype(object)
        if isinstance(value, ExtensionArray):
            value = value.astype(object)
        return arr.searchsorted(value, side=side, sorter=sorter)

    def equals(self, other: object) -> bool:
        """
        Return if another array is equivalent to this array.

        Equivalent means that both arrays have the same shape and dtype, and
        all values compare equal. Missing values in the same location are
        considered equal (in contrast with normal equality).

        Parameters
        ----------
        other : ExtensionArray
            Array to compare to this Array.

        Returns
        -------
        boolean
            Whether the arrays are equivalent.

        Examples
        --------
        >>> arr1 = pd.array([1, 2, np.nan])
        >>> arr2 = pd.array([1, 2, np.nan])
        >>> arr1.equals(arr2)
        True
        """
        if type(self) != type(other):
            return False
        other = cast(ExtensionArray, other)
        if self.dtype != other.dtype:
            return False
        elif len(self) != len(other):
            return False
        else:
            equal_values = self == other
            if isinstance(equal_values, ExtensionArray):
                # boolean array with NA -> fill with False
                equal_values = equal_values.fillna(False)
            # error: Unsupported left operand type for & ("ExtensionArray")
            equal_na = self.isna() & other.isna()  # type: ignore[operator]
            return bool((equal_values | equal_na).all())

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        """
        Pointwise comparison for set containment in the given values.

        Roughly equivalent to `np.array([x in values for x in self])`

        Parameters
        ----------
        values : np.ndarray or ExtensionArray

        Returns
        -------
        np.ndarray[bool]

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.isin([1])
        <BooleanArray>
        [True, False, False]
        Length: 3, dtype: boolean
        """
        return isin(np.asarray(self), values)

    def _values_for_factorize(self) -> tuple[np.ndarray, Any]:
        """
        Return an array and missing value suitable for factorization.

        Returns
        -------
        values : ndarray
            An array suitable for factorization. This should maintain order
            and be a supported dtype (Float64, Int64, UInt64, String, Object).
            By default, the extension array is cast to object dtype.
        na_value : object
            The value in `values` to consider missing. This will be treated
            as NA in the factorization routines, so it will be coded as
            `-1` and not included in `uniques`. By default,
            ``np.nan`` is used.

        Notes
        -----
        The values returned by this method are also used in
        :func:`pandas.util.hash_pandas_object`. If needed, this can be
        overridden in the ``self._hash_pandas_object()`` method.

        Examples
        --------
        >>> pd.array([1, 2, 3])._values_for_factorize()
        (array([1, 2, 3], dtype=object), nan)
        """
        return self.astype(object), np.nan

    def factorize(
        self,
        use_na_sentinel: bool = True,
    ) -> tuple[np.ndarray, ExtensionArray]:
        """
        Encode the extension array as an enumerated type.

        Parameters
        ----------
        use_na_sentinel : bool, default True
            If True, the sentinel -1 will be used for NaN values. If False,
            NaN values will be encoded as non-negative integers and will not drop the
            NaN from the uniques of the values.

            .. versionadded:: 1.5.0

        Returns
        -------
        codes : ndarray
            An integer NumPy array that's an indexer into the original
            ExtensionArray.
        uniques : ExtensionArray
            An ExtensionArray containing the unique values of `self`.

            .. note::

               uniques will *not* contain an entry for the NA value of
               the ExtensionArray if there are any missing values present
               in `self`.

        See Also
        --------
        factorize : Top-level factorize method that dispatches here.

        Notes
        -----
        :meth:`pandas.factorize` offers a `sort` keyword as well.

        Examples
        --------
        >>> idx1 = pd.PeriodIndex(["2014-01", "2014-01", "2014-02", "2014-02",
        ...                       "2014-03", "2014-03"], freq="M")
        >>> arr, idx = idx1.factorize()
        >>> arr
        array([0, 0, 1, 1, 2, 2])
        >>> idx
        PeriodIndex(['2014-01', '2014-02', '2014-03'], dtype='period[M]')
        """
        # Implementer note: There are two ways to override the behavior of
        # pandas.factorize
        # 1. _values_for_factorize and _from_factorize.
        #    Specify the values passed to pandas' internal factorization
        #    routines, and how to convert from those values back to the
        #    original ExtensionArray.
        # 2. ExtensionArray.factorize.
        #    Complete control over factorization.
        arr, na_value = self._values_for_factorize()

        codes, uniques = factorize_array(
            arr, use_na_sentinel=use_na_sentinel, na_value=na_value
        )

        uniques_ea = self._from_factorized(uniques, self)
        return codes, uniques_ea

    _extension_array_shared_docs[
        "repeat"
    ] = """
        Repeat elements of a %(klass)s.

        Returns a new %(klass)s where each element of the current %(klass)s
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            %(klass)s.
        axis : None
            Must be ``None``. Has no effect but is accepted for compatibility
            with numpy.

        Returns
        -------
        %(klass)s
            Newly created %(klass)s with repeated elements.

        See Also
        --------
        Series.repeat : Equivalent function for Series.
        Index.repeat : Equivalent function for Index.
        numpy.repeat : Similar method for :class:`numpy.ndarray`.
        ExtensionArray.take : Take arbitrary positions.

        Examples
        --------
        >>> cat = pd.Categorical(['a', 'b', 'c'])
        >>> cat
        ['a', 'b', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> cat.repeat(2)
        ['a', 'a', 'b', 'b', 'c', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> cat.repeat([1, 2, 3])
        ['a', 'b', 'b', 'c', 'c', 'c']
        Categories (3, object): ['a', 'b', 'c']
        """

    @Substitution(klass="ExtensionArray")
    @Appender(_extension_array_shared_docs["repeat"])
    def repeat(self, repeats: int | Sequence[int], axis: AxisInt | None = None) -> Self:
        nv.validate_repeat((), {"axis": axis})
        ind = np.arange(len(self)).repeat(repeats)
        return self.take(ind)

    # ------------------------------------------------------------------------
    # Indexing methods
    # ------------------------------------------------------------------------

    def take(
        self,
        indices: TakeIndexer,
        *,
        allow_fill: bool = False,
        fill_value: Any = None,
    ) -> Self:
        """
        Take elements from an array.

        Parameters
        ----------
        indices : sequence of int or one-dimensional np.ndarray of int
            Indices to be taken.
        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.

        fill_value : any, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if necessary.

        Returns
        -------
        ExtensionArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.

        See Also
        --------
        numpy.take : Take elements from an array along an axis.
        api.extensions.take : Take elements from an array.

        Notes
        -----
        ExtensionArray.take is called by ``Series.__getitem__``, ``.loc``,
        ``iloc``, when `indices` is a sequence of values. Additionally,
        it's called by :meth:`Series.reindex`, or any other method
        that causes realignment, with a `fill_value`.

        Examples
        --------
        Here's an example implementation, which relies on casting the
        extension array to object dtype. This uses the helper method
        :func:`pandas.api.extensions.take`.

        .. code-block:: python

           def take(self, indices, allow_fill=False, fill_value=None):
               from pandas.core.algorithms import take

               # If the ExtensionArray is backed by an ndarray, then
               # just pass that here instead of coercing to object.
               data = self.astype(object)

               if allow_fill and fill_value is None:
                   fill_value = self.dtype.na_value

               # fill value should always be translated from the scalar
               # type for the array, to the physical storage type for
               # the data, before passing to take.

               result = take(data, indices, fill_value=fill_value,
                             allow_fill=allow_fill)
               return self._from_sequence(result, dtype=self.dtype)
        """
        # Implementer note: The `fill_value` parameter should be a user-facing
        # value, an instance of self.dtype.type. When passed `fill_value=None`,
        # the default of `self.dtype.na_value` should be used.
        # This may differ from the physical storage type your ExtensionArray
        # uses. In this case, your implementation is responsible for casting
        # the user-facing type to the storage type, before using
        # pandas.api.extensions.take
        raise AbstractMethodError(self)

    def copy(self) -> Self:
        """
        Return a copy of the array.

        Returns
        -------
        ExtensionArray

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr2 = arr.copy()
        >>> arr[0] = 2
        >>> arr2
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        raise AbstractMethodError(self)

    def view(self, dtype: Dtype | None = None) -> ArrayLike:
        """
        Return a view on the array.

        Parameters
        ----------
        dtype : str, np.dtype, or ExtensionDtype, optional
            Default None.

        Returns
        -------
        ExtensionArray or np.ndarray
            A view on the :class:`ExtensionArray`'s data.

        Examples
        --------
        This gives view on the underlying data of an ``ExtensionArray`` and is not a
        copy. Modifications on either the view or the original ``ExtensionArray``
        will be reflectd on the underlying data:

        >>> arr = pd.array([1, 2, 3])
        >>> arr2 = arr.view()
        >>> arr[0] = 2
        >>> arr2
        <IntegerArray>
        [2, 2, 3]
        Length: 3, dtype: Int64
        """
        # NB:
        # - This must return a *new* object referencing the same data, not self.
        # - The only case that *must* be implemented is with dtype=None,
        #   giving a view with the same dtype as self.
        if dtype is not None:
            raise NotImplementedError(dtype)
        return self[:]

    # ------------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.ndim > 1:
            return self._repr_2d()

        from pandas.io.formats.printing import format_object_summary

        # the short repr has no trailing newline, while the truncated
        # repr does. So we include a newline in our template, and strip
        # any trailing newlines from format_object_summary
        data = format_object_summary(
            self, self._formatter(), indent_for_name=False
        ).rstrip(", \n")
        class_name = f"<{type(self).__name__}>\n"
        footer = self._get_repr_footer()
        return f"{class_name}{data}\n{footer}"

    def _get_repr_footer(self) -> str:
        # GH#24278
        if self.ndim > 1:
            return f"Shape: {self.shape}, dtype: {self.dtype}"
        return f"Length: {len(self)}, dtype: {self.dtype}"

    def _repr_2d(self) -> str:
        from pandas.io.formats.printing import format_object_summary

        # the short repr has no trailing newline, while the truncated
        # repr does. So we include a newline in our template, and strip
        # any trailing newlines from format_object_summary
        lines = [
            format_object_summary(x, self._formatter(), indent_for_name=False).rstrip(
                ", \n"
            )
            for x in self
        ]
        data = ",\n".join(lines)
        class_name = f"<{type(self).__name__}>"
        footer = self._get_repr_footer()
        return f"{class_name}\n[\n{data}\n]\n{footer}"

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]:
        """
        Formatting function for scalar values.

        This is used in the default '__repr__'. The returned formatting
        function receives instances of your scalar type.

        Parameters
        ----------
        boxed : bool, default False
            An indicated for whether or not your array is being printed
            within a Series, DataFrame, or Index (True), or just by
            itself (False). This may be useful if you want scalar values
            to appear differently within a Series versus on its own (e.g.
            quoted or not).

        Returns
        -------
        Callable[[Any], str]
            A callable that gets instances of the scalar type and
            returns a string. By default, :func:`repr` is used
            when ``boxed=False`` and :func:`str` is used when
            ``boxed=True``.

        Examples
        --------
        >>> class MyExtensionArray(pd.arrays.NumpyExtensionArray):
        ...     def _formatter(self, boxed=False):
        ...         return lambda x: '*' + str(x) + '*' if boxed else repr(x) + '*'
        >>> MyExtensionArray(np.array([1, 2, 3, 4]))
        <MyExtensionArray>
        [1*, 2*, 3*, 4*]
        Length: 4, dtype: int64
        """
        if boxed:
            return str
        return repr

    # ------------------------------------------------------------------------
    # Reshaping
    # ------------------------------------------------------------------------

    def transpose(self, *axes: int) -> ExtensionArray:
        """
        Return a transposed view on this array.

        Because ExtensionArrays are always 1D, this is a no-op.  It is included
        for compatibility with np.ndarray.

        Returns
        -------
        ExtensionArray

        Examples
        --------
        >>> pd.array([1, 2, 3]).transpose()
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        return self[:]

    @property
    def T(self) -> ExtensionArray:
        return self.transpose()

    def ravel(self, order: Literal["C", "F", "A", "K"] | None = "C") -> ExtensionArray:
        """
        Return a flattened view on this array.

        Parameters
        ----------
        order : {None, 'C', 'F', 'A', 'K'}, default 'C'

        Returns
        -------
        ExtensionArray

        Notes
        -----
        - Because ExtensionArrays are 1D-only, this is a no-op.
        - The "order" argument is ignored, is for compatibility with NumPy.

        Examples
        --------
        >>> pd.array([1, 2, 3]).ravel()
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        return self

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self:
        """
        Concatenate multiple array of this dtype.

        Parameters
        ----------
        to_concat : sequence of this type

        Returns
        -------
        ExtensionArray

        Examples
        --------
        >>> arr1 = pd.array([1, 2, 3])
        >>> arr2 = pd.array([4, 5, 6])
        >>> pd.arrays.IntegerArray._concat_same_type([arr1, arr2])
        <IntegerArray>
        [1, 2, 3, 4, 5, 6]
        Length: 6, dtype: Int64
        """
        # Implementer note: this method will only be called with a sequence of
        # ExtensionArrays of this class and with the same dtype as self. This
        # should allow "easy" concatenation (no upcasting needed), and result
        # in a new ExtensionArray of the same dtype.
        # Note: this strict behaviour is only guaranteed starting with pandas 1.1
        raise AbstractMethodError(cls)

    # The _can_hold_na attribute is set to True so that pandas internals
    # will use the ExtensionDtype.na_value as the NA value in operations
    # such as take(), reindex(), shift(), etc.  In addition, those results
    # will then be of the ExtensionArray subclass rather than an array
    # of objects
    @cache_readonly
    def _can_hold_na(self) -> bool:
        return self.dtype._can_hold_na

    def _accumulate(
        self, name: str, *, skipna: bool = True, **kwargs
    ) -> ExtensionArray:
        """
        Return an ExtensionArray performing an accumulation operation.

        The underlying data type might change.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            - cummin
            - cummax
            - cumsum
            - cumprod
        skipna : bool, default True
            If True, skip NA values.
        **kwargs
            Additional keyword arguments passed to the accumulation function.
            Currently, there is no supported kwarg.

        Returns
        -------
        array

        Raises
        ------
        NotImplementedError : subclass does not define accumulations

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr._accumulate(name='cumsum')
        <IntegerArray>
        [1, 3, 6]
        Length: 3, dtype: Int64
        """
        raise NotImplementedError(f"cannot perform {name} with type {self.dtype}")

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        """
        Return a scalar result of performing the reduction operation.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values.
        keepdims : bool, default False
            If False, a scalar is returned.
            If True, the result has dimension with size one along the reduced axis.

            .. versionadded:: 2.1

               This parameter is not required in the _reduce signature to keep backward
               compatibility, but will become required in the future. If the parameter
               is not found in the method signature, a FutureWarning will be emitted.
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.

        Returns
        -------
        scalar

        Raises
        ------
        TypeError : subclass does not define reductions

        Examples
        --------
        >>> pd.array([1, 2, 3])._reduce("min")
        1
        """
        meth = getattr(self, name, None)
        if meth is None:
            raise TypeError(
                f"'{type(self).__name__}' with dtype {self.dtype} "
                f"does not support reduction '{name}'"
            )
        result = meth(skipna=skipna, **kwargs)
        if keepdims:
            result = np.array([result])

        return result

    # https://github.com/python/typeshed/issues/2148#issuecomment-520783318
    # Incompatible types in assignment (expression has type "None", base class
    # "object" defined the type as "Callable[[object], int]")
    __hash__: ClassVar[None]  # type: ignore[assignment]

    # ------------------------------------------------------------------------
    # Non-Optimized Default Methods; in the case of the private methods here,
    #  these are not guaranteed to be stable across pandas versions.

    def _values_for_json(self) -> np.ndarray:
        """
        Specify how to render our entries in to_json.

        Notes
        -----
        The dtype on the returned ndarray is not restricted, but for non-native
        types that are not specifically handled in objToJSON.c, to_json is
        liable to raise. In these cases, it may be safer to return an ndarray
        of strings.
        """
        return np.asarray(self)

    def _hash_pandas_object(
        self, *, encoding: str, hash_key: str, categorize: bool
    ) -> npt.NDArray[np.uint64]:
        """
        Hook for hash_pandas_object.

        Default is to use the values returned by _values_for_factorize.

        Parameters
        ----------
        encoding : str
            Encoding for data & key when strings.
        hash_key : str
            Hash_key for string key to encode.
        categorize : bool
            Whether to first categorize object arrays before hashing. This is more
            efficient when the array contains duplicate values.

        Returns
        -------
        np.ndarray[uint64]

        Examples
        --------
        >>> pd.array([1, 2])._hash_pandas_object(encoding='utf-8',
        ...                                      hash_key="1000000000000000",
        ...                                      categorize=False
        ...                                      )
        array([ 6238072747940578789, 15839785061582574730], dtype=uint64)
        """
        from pandas.core.util.hashing import hash_array

        values, _ = self._values_for_factorize()
        return hash_array(
            values, encoding=encoding, hash_key=hash_key, categorize=categorize
        )

    def _explode(self) -> tuple[Self, npt.NDArray[np.uint64]]:
        """
        Transform each element of list-like to a row.

        For arrays that do not contain list-like elements the default
        implementation of this method just returns a copy and an array
        of ones (unchanged index).

        Returns
        -------
        ExtensionArray
            Array with the exploded values.
        np.ndarray[uint64]
            The original lengths of each list-like for determining the
            resulting index.

        See Also
        --------
        Series.explode : The method on the ``Series`` object that this
            extension array method is meant to support.

        Examples
        --------
        >>> import pyarrow as pa
        >>> a = pd.array([[1, 2, 3], [4], [5, 6]],
        ...              dtype=pd.ArrowDtype(pa.list_(pa.int64())))
        >>> a._explode()
        (<ArrowExtensionArray>
        [1, 2, 3, 4, 5, 6]
        Length: 6, dtype: int64[pyarrow], array([3, 1, 2], dtype=int32))
        """
        values = self.copy()
        counts = np.ones(shape=(len(self),), dtype=np.uint64)
        return values, counts

    def tolist(self) -> list:
        """
        Return a list of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        list

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.tolist()
        [1, 2, 3]
        """
        if self.ndim > 1:
            return [x.tolist() for x in self]
        return list(self)

    def delete(self, loc: PositionalIndexer) -> Self:
        indexer = np.delete(np.arange(len(self)), loc)
        return self.take(indexer)

    def insert(self, loc: int, item) -> Self:
        """
        Insert an item at the given position.

        Parameters
        ----------
        loc : int
        item : scalar-like

        Returns
        -------
        same type as self

        Notes
        -----
        This method should be both type and dtype-preserving.  If the item
        cannot be held in an array of this type/dtype, either ValueError or
        TypeError should be raised.

        The default implementation relies on _from_sequence to raise on invalid
        items.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.insert(2, -1)
        <IntegerArray>
        [1, 2, -1, 3]
        Length: 4, dtype: Int64
        """
        loc = validate_insert_loc(loc, len(self))

        item_arr = type(self)._from_sequence([item], dtype=self.dtype)

        return type(self)._concat_same_type([self[:loc], item_arr, self[loc:]])

    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        """
        Analogue to np.putmask(self, mask, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
        value : scalar or listlike
            If listlike, must be arraylike with same length as self.

        Returns
        -------
        None

        Notes
        -----
        Unlike np.putmask, we do not repeat listlike values with mismatched length.
        'value' should either be a scalar or an arraylike with the same length
        as self.
        """
        if is_list_like(value):
            val = value[mask]
        else:
            val = value

        self[mask] = val

    def _where(self, mask: npt.NDArray[np.bool_], value) -> Self:
        """
        Analogue to np.where(mask, self, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
        value : scalar or listlike

        Returns
        -------
        same type as self
        """
        result = self.copy()

        if is_list_like(value):
            val = value[~mask]
        else:
            val = value

        result[~mask] = val
        return result

    # TODO(3.0): this can be removed once GH#33302 deprecation is enforced
    def _fill_mask_inplace(
        self, method: str, limit: int | None, mask: npt.NDArray[np.bool_]
    ) -> None:
        """
        Replace values in locations specified by 'mask' using pad or backfill.

        See also
        --------
        ExtensionArray.fillna
        """
        func = missing.get_fill_func(method)
        npvalues = self.astype(object)
        # NB: if we don't copy mask here, it may be altered inplace, which
        #  would mess up the `self[mask] = ...` below.
        func(npvalues, limit=limit, mask=mask.copy())
        new_values = self._from_sequence(npvalues, dtype=self.dtype)
        self[mask] = new_values[mask]

    def _rank(
        self,
        *,
        axis: AxisInt = 0,
        method: str = "average",
        na_option: str = "keep",
        ascending: bool = True,
        pct: bool = False,
    ):
        """
        See Series.rank.__doc__.
        """
        if axis != 0:
            raise NotImplementedError

        return rank(
            self._values_for_argsort(),
            axis=axis,
            method=method,
            na_option=na_option,
            ascending=ascending,
            pct=pct,
        )

    @classmethod
    def _empty(cls, shape: Shape, dtype: ExtensionDtype):
        """
        Create an ExtensionArray with the given shape and dtype.

        See also
        --------
        ExtensionDtype.empty
            ExtensionDtype.empty is the 'official' public version of this API.
        """
        # Implementer note: while ExtensionDtype.empty is the public way to
        # call this method, it is still required to implement this `_empty`
        # method as well (it is called internally in pandas)
        obj = cls._from_sequence([], dtype=dtype)

        taker = np.broadcast_to(np.intp(-1), shape)
        result = obj.take(taker, allow_fill=True)
        if not isinstance(result, cls) or dtype != result.dtype:
            raise NotImplementedError(
                f"Default 'empty' implementation is invalid for dtype='{dtype}'"
            )
        return result

    def _quantile(self, qs: npt.NDArray[np.float64], interpolation: str) -> Self:
        """
        Compute the quantiles of self for each quantile in `qs`.

        Parameters
        ----------
        qs : np.ndarray[float64]
        interpolation: str

        Returns
        -------
        same type as self
        """
        mask = np.asarray(self.isna())
        arr = np.asarray(self)
        fill_value = np.nan

        res_values = quantile_with_mask(arr, mask, fill_value, qs, interpolation)
        return type(self)._from_sequence(res_values)

    def _mode(self, dropna: bool = True) -> Self:
        """
        Returns the mode(s) of the ExtensionArray.

        Always returns `ExtensionArray` even if only one value.

        Parameters
        ----------
        dropna : bool, default True
            Don't consider counts of NA values.

        Returns
        -------
        same type as self
            Sorted, if possible.
        """
        # error: Incompatible return value type (got "Union[ExtensionArray,
        # ndarray[Any, Any]]", expected "Self")
        return mode(self, dropna=dropna)  # type: ignore[return-value]

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        if any(
            isinstance(other, (ABCSeries, ABCIndex, ABCDataFrame)) for other in inputs
        ):
            return NotImplemented

        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if "out" in kwargs:
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                return result

        return arraylike.default_array_ufunc(self, ufunc, method, *inputs, **kwargs)

    def map(self, mapper, na_action=None):
        """
        Map values using an input mapping or function.

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NA values, without passing them to the
            mapping correspondence. If 'ignore' is not supported, a
            ``NotImplementedError`` should be raised.

        Returns
        -------
        Union[ndarray, Index, ExtensionArray]
            The output of the mapping function applied to the array.
            If the function returns a tuple with more than one element
            a MultiIndex will be returned.
        """
        return map_array(self, mapper, na_action=na_action)

    # ------------------------------------------------------------------------
    # GroupBy Methods

    def _groupby_op(
        self,
        *,
        how: str,
        has_dropped_na: bool,
        min_count: int,
        ngroups: int,
        ids: npt.NDArray[np.intp],
        **kwargs,
    ) -> ArrayLike:
        """
        Dispatch GroupBy reduction or transformation operation.

        This is an *experimental* API to allow ExtensionArray authors to implement
        reductions and transformations. The API is subject to change.

        Parameters
        ----------
        how : {'any', 'all', 'sum', 'prod', 'min', 'max', 'mean', 'median',
               'median', 'var', 'std', 'sem', 'nth', 'last', 'ohlc',
               'cumprod', 'cumsum', 'cummin', 'cummax', 'rank'}
        has_dropped_na : bool
        min_count : int
        ngroups : int
        ids : np.ndarray[np.intp]
            ids[i] gives the integer label for the group that self[i] belongs to.
        **kwargs : operation-specific
            'any', 'all' -> ['skipna']
            'var', 'std', 'sem' -> ['ddof']
            'cumprod', 'cumsum', 'cummin', 'cummax' -> ['skipna']
            'rank' -> ['ties_method', 'ascending', 'na_option', 'pct']

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        from pandas.core.arrays.string_ import StringDtype
        from pandas.core.groupby.ops import WrappedCythonOp

        kind = WrappedCythonOp.get_kind_from_how(how)
        op = WrappedCythonOp(how=how, kind=kind, has_dropped_na=has_dropped_na)

        # GH#43682
        if isinstance(self.dtype, StringDtype):
            # StringArray
            if op.how in [
                "prod",
                "mean",
                "median",
                "cumsum",
                "cumprod",
                "std",
                "sem",
                "var",
                "skew",
            ]:
                raise TypeError(
                    f"dtype '{self.dtype}' does not support operation '{how}'"
                )
            if op.how not in ["any", "all"]:
                # Fail early to avoid conversion to object
                op._get_cython_function(op.kind, op.how, np.dtype(object), False)
            npvalues = self.to_numpy(object, na_value=np.nan)
        else:
            raise NotImplementedError(
                f"function is not implemented for this dtype: {self.dtype}"
            )

        res_values = op._cython_op_ndim_compat(
            npvalues,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=ids,
            mask=None,
            **kwargs,
        )

        if op.how in op.cast_blocklist:
            # i.e. how in ["rank"], since other cast_blocklist methods don't go
            #  through cython_operation
            return res_values

        if isinstance(self.dtype, StringDtype):
            dtype = self.dtype
            string_array_cls = dtype.construct_array_type()
            return string_array_cls._from_sequence(res_values, dtype=dtype)

        else:
            raise NotImplementedError


class ExtensionArraySupportsAnyAll(ExtensionArray):
    def any(self, *, skipna: bool = True) -> bool:
        raise AbstractMethodError(self)

    def all(self, *, skipna: bool = True) -> bool:
        raise AbstractMethodError(self)


class ExtensionOpsMixin:
    """
    A base class for linking the operators to their dunder names.

    .. note::

       You may want to set ``__array_priority__`` if you want your
       implementation to be called when involved in binary operations
       with NumPy arrays.
    """

    @classmethod
    def _create_arithmetic_method(cls, op):
        raise AbstractMethodError(cls)

    @classmethod
    def _add_arithmetic_ops(cls) -> None:
        setattr(cls, "__add__", cls._create_arithmetic_method(operator.add))
        setattr(cls, "__radd__", cls._create_arithmetic_method(roperator.radd))
        setattr(cls, "__sub__", cls._create_arithmetic_method(operator.sub))
        setattr(cls, "__rsub__", cls._create_arithmetic_method(roperator.rsub))
        setattr(cls, "__mul__", cls._create_arithmetic_method(operator.mul))
        setattr(cls, "__rmul__", cls._create_arithmetic_method(roperator.rmul))
        setattr(cls, "__pow__", cls._create_arithmetic_method(operator.pow))
        setattr(cls, "__rpow__", cls._create_arithmetic_method(roperator.rpow))
        setattr(cls, "__mod__", cls._create_arithmetic_method(operator.mod))
        setattr(cls, "__rmod__", cls._create_arithmetic_method(roperator.rmod))
        setattr(cls, "__floordiv__", cls._create_arithmetic_method(operator.floordiv))
        setattr(
            cls, "__rfloordiv__", cls._create_arithmetic_method(roperator.rfloordiv)
        )
        setattr(cls, "__truediv__", cls._create_arithmetic_method(operator.truediv))
        setattr(cls, "__rtruediv__", cls._create_arithmetic_method(roperator.rtruediv))
        setattr(cls, "__divmod__", cls._create_arithmetic_method(divmod))
        setattr(cls, "__rdivmod__", cls._create_arithmetic_method(roperator.rdivmod))

    @classmethod
    def _create_comparison_method(cls, op):
        raise AbstractMethodError(cls)

    @classmethod
    def _add_comparison_ops(cls) -> None:
        setattr(cls, "__eq__", cls._create_comparison_method(operator.eq))
        setattr(cls, "__ne__", cls._create_comparison_method(operator.ne))
        setattr(cls, "__lt__", cls._create_comparison_method(operator.lt))
        setattr(cls, "__gt__", cls._create_comparison_method(operator.gt))
        setattr(cls, "__le__", cls._create_comparison_method(operator.le))
        setattr(cls, "__ge__", cls._create_comparison_method(operator.ge))

    @classmethod
    def _create_logical_method(cls, op):
        raise AbstractMethodError(cls)

    @classmethod
    def _add_logical_ops(cls) -> None:
        setattr(cls, "__and__", cls._create_logical_method(operator.and_))
        setattr(cls, "__rand__", cls._create_logical_method(roperator.rand_))
        setattr(cls, "__or__", cls._create_logical_method(operator.or_))
        setattr(cls, "__ror__", cls._create_logical_method(roperator.ror_))
        setattr(cls, "__xor__", cls._create_logical_method(operator.xor))
        setattr(cls, "__rxor__", cls._create_logical_method(roperator.rxor))


class ExtensionScalarOpsMixin(ExtensionOpsMixin):
    """
    A mixin for defining ops on an ExtensionArray.

    It is assumed that the underlying scalar objects have the operators
    already defined.

    Notes
    -----
    If you have defined a subclass MyExtensionArray(ExtensionArray), then
    use MyExtensionArray(ExtensionArray, ExtensionScalarOpsMixin) to
    get the arithmetic operators.  After the definition of MyExtensionArray,
    insert the lines

    MyExtensionArray._add_arithmetic_ops()
    MyExtensionArray._add_comparison_ops()

    to link the operators to your class.

    .. note::

       You may want to set ``__array_priority__`` if you want your
       implementation to be called when involved in binary operations
       with NumPy arrays.
    """

    @classmethod
    def _create_method(cls, op, coerce_to_dtype: bool = True, result_dtype=None):
        """
        A class method that returns a method that will correspond to an
        operator for an ExtensionArray subclass, by dispatching to the
        relevant operator defined on the individual elements of the
        ExtensionArray.

        Parameters
        ----------
        op : function
            An operator that takes arguments op(a, b)
        coerce_to_dtype : bool, default True
            boolean indicating whether to attempt to convert
            the result to the underlying ExtensionArray dtype.
            If it's not possible to create a new ExtensionArray with the
            values, an ndarray is returned instead.

        Returns
        -------
        Callable[[Any, Any], Union[ndarray, ExtensionArray]]
            A method that can be bound to a class. When used, the method
            receives the two arguments, one of which is the instance of
            this class, and should return an ExtensionArray or an ndarray.

            Returning an ndarray may be necessary when the result of the
            `op` cannot be stored in the ExtensionArray. The dtype of the
            ndarray uses NumPy's normal inference rules.

        Examples
        --------
        Given an ExtensionArray subclass called MyExtensionArray, use

            __add__ = cls._create_method(operator.add)

        in the class definition of MyExtensionArray to create the operator
        for addition, that will be based on the operator implementation
        of the underlying elements of the ExtensionArray
        """

        def _binop(self, other):
            def convert_values(param):
                if isinstance(param, ExtensionArray) or is_list_like(param):
                    ovalues = param
                else:  # Assume its an object
                    ovalues = [param] * len(self)
                return ovalues

            if isinstance(other, (ABCSeries, ABCIndex, ABCDataFrame)):
                # rely on pandas to unbox and dispatch to us
                return NotImplemented

            lvalues = self
            rvalues = convert_values(other)

            # If the operator is not defined for the underlying objects,
            # a TypeError should be raised
            res = [op(a, b) for (a, b) in zip(lvalues, rvalues)]

            def _maybe_convert(arr):
                if coerce_to_dtype:
                    # https://github.com/pandas-dev/pandas/issues/22850
                    # We catch all regular exceptions here, and fall back
                    # to an ndarray.
                    res = maybe_cast_pointwise_result(arr, self.dtype, same_dtype=False)
                    if not isinstance(res, type(self)):
                        # exception raised in _from_sequence; ensure we have ndarray
                        res = np.asarray(arr)
                else:
                    res = np.asarray(arr, dtype=result_dtype)
                return res

            if op.__name__ in {"divmod", "rdivmod"}:
                a, b = zip(*res)
                return _maybe_convert(a), _maybe_convert(b)

            return _maybe_convert(res)

        op_name = f"__{op.__name__}__"
        return set_function_name(_binop, op_name, cls)

    @classmethod
    def _create_arithmetic_method(cls, op):
        return cls._create_method(op)

    @classmethod
    def _create_comparison_method(cls, op):
        return cls._create_method(op, coerce_to_dtype=False, result_dtype=bool)