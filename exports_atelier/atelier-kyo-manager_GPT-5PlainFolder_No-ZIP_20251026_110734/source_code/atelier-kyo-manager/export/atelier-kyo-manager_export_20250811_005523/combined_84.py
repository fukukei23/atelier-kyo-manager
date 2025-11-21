
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_subprocess.py ===
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import warnings
from contextlib import ExitStack
from functools import partial
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    TypedDict,
    Union,
    overload,
)

import trio

from ._core import ClosedResourceError, TaskStatus
from ._highlevel_generic import StapledStream
from ._subprocess_platform import (
    create_pipe_from_child_output,
    create_pipe_to_child_stdin,
    wait_child_exiting,
)
from ._sync import Lock
from ._util import NoPublicConstructor, final

if TYPE_CHECKING:
    import signal
    from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
    from io import TextIOWrapper

    from typing_extensions import TypeAlias, Unpack

    from ._abc import ReceiveStream, SendStream


# Sphinx cannot parse the stringified version
StrOrBytesPath: TypeAlias = Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]


# Linux-specific, but has complex lifetime management stuff so we hard-code it
# here instead of hiding it behind the _subprocess_platform abstraction
can_try_pidfd_open: bool
if TYPE_CHECKING:

    def pidfd_open(fd: int, flags: int) -> int: ...

    from ._subprocess_platform import ClosableReceiveStream, ClosableSendStream

else:
    can_try_pidfd_open = True
    try:
        from os import pidfd_open
    except ImportError:
        if sys.platform == "linux":
            # this workaround is needed on:
            #  - CPython <= 3.8
            #  - non-CPython (maybe?)
            #  - Anaconda's interpreter (as it is built to assume an older
            #    than current linux kernel)
            #
            # The last point implies that other custom builds might not work;
            # therefore, no assertion should be here.
            import ctypes

            _cdll_for_pidfd_open = ctypes.CDLL(None, use_errno=True)
            _cdll_for_pidfd_open.syscall.restype = ctypes.c_long
            # pid and flags are actually int-sized, but the syscall() function
            # always takes longs. (Except on x32 where long is 32-bits and syscall
            # takes 64-bit arguments. But in the unlikely case that anyone is
            # using x32, this will still work, b/c we only need to pass in 32 bits
            # of data, and the C ABI doesn't distinguish between passing 32-bit vs
            # 64-bit integers; our 32-bit values will get loaded into 64-bit
            # registers where syscall() will find them.)
            _cdll_for_pidfd_open.syscall.argtypes = [
                ctypes.c_long,  # syscall number
                ctypes.c_long,  # pid
                ctypes.c_long,  # flags
            ]
            __NR_pidfd_open = 434

            def pidfd_open(fd: int, flags: int) -> int:
                result = _cdll_for_pidfd_open.syscall(__NR_pidfd_open, fd, flags)
                if result < 0:  # pragma: no cover
                    err = ctypes.get_errno()
                    raise OSError(err, os.strerror(err))
                return result

        else:
            can_try_pidfd_open = False


class HasFileno(Protocol):
    """Represents any file-like object that has a file descriptor."""

    def fileno(self) -> int: ...


@final
class Process(metaclass=NoPublicConstructor):
    r"""A child process. Like :class:`subprocess.Popen`, but async.

    This class has no public constructor. The most common way to get a
    `Process` object is to combine `Nursery.start` with `run_process`::

       process_object = await nursery.start(run_process, ...)

    This way, `run_process` supervises the process and makes sure that it is
    cleaned up properly, while optionally checking the return value, feeding
    it input, and so on.

    If you need more control – for example, because you want to spawn a child
    process that outlives your program – then another option is to use
    `trio.lowlevel.open_process`::

       process_object = await trio.lowlevel.open_process(...)

    Attributes:
      args (str or list): The ``command`` passed at construction time,
          specifying the process to execute and its arguments.
      pid (int): The process ID of the child process managed by this object.
      stdin (trio.abc.SendStream or None): A stream connected to the child's
          standard input stream: when you write bytes here, they become available
          for the child to read. Only available if the :class:`Process`
          was constructed using ``stdin=PIPE``; otherwise this will be None.
      stdout (trio.abc.ReceiveStream or None): A stream connected to
          the child's standard output stream: when the child writes to
          standard output, the written bytes become available for you
          to read here. Only available if the :class:`Process` was
          constructed using ``stdout=PIPE``; otherwise this will be None.
      stderr (trio.abc.ReceiveStream or None): A stream connected to
          the child's standard error stream: when the child writes to
          standard error, the written bytes become available for you
          to read here. Only available if the :class:`Process` was
          constructed using ``stderr=PIPE``; otherwise this will be None.
      stdio (trio.StapledStream or None): A stream that sends data to
          the child's standard input and receives from the child's standard
          output. Only available if both :attr:`stdin` and :attr:`stdout` are
          available; otherwise this will be None.

    """

    # We're always in binary mode.
    universal_newlines: Final = False
    encoding: Final = None
    errors: Final = None

    # Available for the per-platform wait_child_exiting() implementations
    # to stash some state; waitid platforms use this to avoid spawning
    # arbitrarily many threads if wait() keeps getting cancelled.
    _wait_for_exit_data: object = None

    def __init__(
        self,
        popen: subprocess.Popen[bytes],
        stdin: SendStream | None,
        stdout: ReceiveStream | None,
        stderr: ReceiveStream | None,
    ) -> None:
        self._proc = popen
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr

        self.stdio: StapledStream[SendStream, ReceiveStream] | None = None
        if self.stdin is not None and self.stdout is not None:
            self.stdio = StapledStream(self.stdin, self.stdout)

        self._wait_lock: Lock = Lock()

        self._pidfd: TextIOWrapper | None = None
        if can_try_pidfd_open:
            try:
                fd: int = pidfd_open(self._proc.pid, 0)
            except OSError:  # pragma: no cover
                # Well, we tried, but it didn't work (probably because we're
                # running on an older kernel, or in an older sandbox, that
                # hasn't been updated to support pidfd_open). We'll fall back
                # on waitid instead.
                pass
            else:
                # It worked! Wrap the raw fd up in a Python file object to
                # make sure it'll get closed.
                # SIM115: open-file-with-context-handler
                self._pidfd = open(fd)  # noqa: SIM115

        self.args: StrOrBytesPath | Sequence[StrOrBytesPath] = self._proc.args
        self.pid: int = self._proc.pid

    def __repr__(self) -> str:
        returncode = self.returncode
        if returncode is None:
            status = f"running with PID {self.pid}"
        else:
            if returncode < 0:
                status = f"exited with signal {-returncode}"
            else:
                status = f"exited with status {returncode}"
        return f"<trio.Process {self.args!r}: {status}>"

    @property
    def returncode(self) -> int | None:
        """The exit status of the process (an integer), or ``None`` if it's
        still running.

        By convention, a return code of zero indicates success.  On
        UNIX, negative values indicate termination due to a signal,
        e.g., -11 if terminated by signal 11 (``SIGSEGV``).  On
        Windows, a process that exits due to a call to
        :meth:`Process.terminate` will have an exit status of 1.

        Unlike the standard library `subprocess.Popen.returncode`, you don't
        have to call `poll` or `wait` to update this attribute; it's
        automatically updated as needed, and will always give you the latest
        information.

        """
        result = self._proc.poll()
        if result is not None:
            self._close_pidfd()
        return result

    def _close_pidfd(self) -> None:
        if self._pidfd is not None:
            trio.lowlevel.notify_closing(self._pidfd.fileno())
            self._pidfd.close()
            self._pidfd = None

    async def wait(self) -> int:
        """Block until the process exits.

        Returns:
          The exit status of the process; see :attr:`returncode`.
        """
        async with self._wait_lock:
            if self.poll() is None:
                if self._pidfd is not None:
                    with contextlib.suppress(
                        ClosedResourceError,
                    ):  # something else (probably a call to poll) already closed the pidfd
                        await trio.lowlevel.wait_readable(self._pidfd.fileno())
                else:
                    await wait_child_exiting(self)
                # We have to use .wait() here, not .poll(), because on macOS
                # (and maybe other systems, who knows), there's a race
                # condition inside the kernel that creates a tiny window where
                # kqueue reports that the process has exited, but
                # waitpid(WNOHANG) can't yet reap it. So this .wait() may
                # actually block for a tiny fraction of a second.
                self._proc.wait()
                self._close_pidfd()
        assert self._proc.returncode is not None
        return self._proc.returncode

    def poll(self) -> int | None:
        """Returns the exit status of the process (an integer), or ``None`` if
        it's still running.

        Note that on Trio (unlike the standard library `subprocess.Popen`),
        ``process.poll()`` and ``process.returncode`` always give the same
        result. See `returncode` for more details. This method is only
        included to make it easier to port code from `subprocess`.

        """
        return self.returncode

    def send_signal(self, sig: signal.Signals | int) -> None:
        """Send signal ``sig`` to the process.

        On UNIX, ``sig`` may be any signal defined in the
        :mod:`signal` module, such as ``signal.SIGINT`` or
        ``signal.SIGTERM``. On Windows, it may be anything accepted by
        the standard library :meth:`subprocess.Popen.send_signal`.
        """
        self._proc.send_signal(sig)

    def terminate(self) -> None:
        """Terminate the process, politely if possible.

        On UNIX, this is equivalent to
        ``send_signal(signal.SIGTERM)``; by convention this requests
        graceful termination, but a misbehaving or buggy process might
        ignore it. On Windows, :meth:`terminate` forcibly terminates the
        process in the same manner as :meth:`kill`.
        """
        self._proc.terminate()

    def kill(self) -> None:
        """Immediately terminate the process.

        On UNIX, this is equivalent to
        ``send_signal(signal.SIGKILL)``.  On Windows, it calls
        ``TerminateProcess``. In both cases, the process cannot
        prevent itself from being killed, but the termination will be
        delivered asynchronously; use :meth:`wait` if you want to
        ensure the process is actually dead before proceeding.
        """
        self._proc.kill()


async def _open_process(
    command: StrOrBytesPath | Sequence[StrOrBytesPath],
    *,
    stdin: int | HasFileno | None = None,
    stdout: int | HasFileno | None = None,
    stderr: int | HasFileno | None = None,
    **options: object,
) -> Process:
    r"""Execute a child program in a new process.

    After construction, you can interact with the child process by writing data to its
    `~trio.Process.stdin` stream (a `~trio.abc.SendStream`), reading data from its
    `~trio.Process.stdout` and/or `~trio.Process.stderr` streams (both
    `~trio.abc.ReceiveStream`\s), sending it signals using `~trio.Process.terminate`,
    `~trio.Process.kill`, or `~trio.Process.send_signal`, and waiting for it to exit
    using `~trio.Process.wait`. See `trio.Process` for details.

    Each standard stream is only available if you specify that a pipe should be created
    for it. For example, if you pass ``stdin=subprocess.PIPE``, you can write to the
    `~trio.Process.stdin` stream, else `~trio.Process.stdin` will be ``None``.

    Unlike `trio.run_process`, this function doesn't do any kind of automatic
    management of the child process. It's up to you to implement whatever semantics you
    want.

    Args:
      command: The command to run. Typically this is a sequence of strings or
          bytes such as ``['ls', '-l', 'directory with spaces']``, where the
          first element names the executable to invoke and the other elements
          specify its arguments. With ``shell=True`` in the ``**options``, or on
          Windows, ``command`` can be a string or bytes, which will be parsed
          following platform-dependent :ref:`quoting rules
          <subprocess-quoting>`. In all cases ``command`` can be a path or a
          sequence of paths.
      stdin: Specifies what the child process's standard input
          stream should connect to: output written by the parent
          (``subprocess.PIPE``), nothing (``subprocess.DEVNULL``),
          or an open file (pass a file descriptor or something whose
          ``fileno`` method returns one). If ``stdin`` is unspecified,
          the child process will have the same standard input stream
          as its parent.
      stdout: Like ``stdin``, but for the child process's standard output
          stream.
      stderr: Like ``stdin``, but for the child process's standard error
          stream. An additional value ``subprocess.STDOUT`` is supported,
          which causes the child's standard output and standard error
          messages to be intermixed on a single standard output stream,
          attached to whatever the ``stdout`` option says to attach it to.
      **options: Other :ref:`general subprocess options <subprocess-options>`
          are also accepted.

    Returns:
      A new `trio.Process` object.

    Raises:
      OSError: if the process spawning fails, for example because the
         specified command could not be found.

    """
    for key in ("universal_newlines", "text", "encoding", "errors", "bufsize"):
        if options.get(key):
            raise TypeError(
                "trio.Process only supports communicating over "
                f"unbuffered byte streams; the '{key}' option is not supported",
            )

    if os.name == "posix":
        # TODO: how do paths and sequences thereof play with `shell=True`?
        if isinstance(command, (str, bytes)) and not options.get("shell"):
            raise TypeError(
                "command must be a sequence (not a string or bytes) if "
                "shell=False on UNIX systems",
            )
        if not isinstance(command, (str, bytes)) and options.get("shell"):
            raise TypeError(
                "command must be a string or bytes (not a sequence) if "
                "shell=True on UNIX systems",
            )

    trio_stdin: ClosableSendStream | None = None
    trio_stdout: ClosableReceiveStream | None = None
    trio_stderr: ClosableReceiveStream | None = None
    # Close the parent's handle for each child side of a pipe; we want the child to
    # have the only copy, so that when it exits we can read EOF on our side. The
    # trio ends of pipes will be transferred to the Process object, which will be
    # responsible for their lifetime. If process spawning fails, though, we still
    # want to close them before letting the failure bubble out
    with ExitStack() as always_cleanup, ExitStack() as cleanup_on_fail:
        if stdin == subprocess.PIPE:
            trio_stdin, stdin = create_pipe_to_child_stdin()
            always_cleanup.callback(os.close, stdin)
            cleanup_on_fail.callback(trio_stdin.close)
        if stdout == subprocess.PIPE:
            trio_stdout, stdout = create_pipe_from_child_output()
            always_cleanup.callback(os.close, stdout)
            cleanup_on_fail.callback(trio_stdout.close)
        if stderr == subprocess.STDOUT:
            # If we created a pipe for stdout, pass the same pipe for
            # stderr.  If stdout was some non-pipe thing (DEVNULL or a
            # given FD), pass the same thing. If stdout was passed as
            # None, keep stderr as STDOUT to allow subprocess to dup
            # our stdout. Regardless of which of these is applicable,
            # don't create a new Trio stream for stderr -- if stdout
            # is piped, stderr will be intermixed on the stdout stream.
            if stdout is not None:
                stderr = stdout
        elif stderr == subprocess.PIPE:
            trio_stderr, stderr = create_pipe_from_child_output()
            always_cleanup.callback(os.close, stderr)
            cleanup_on_fail.callback(trio_stderr.close)

        popen = await trio.to_thread.run_sync(
            partial(
                subprocess.Popen,
                command,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                **options,
            ),
        )
        # We did not fail, so dismiss the stack for the trio ends
        cleanup_on_fail.pop_all()

    return Process._create(popen, trio_stdin, trio_stdout, trio_stderr)


# async function missing await
async def _windows_deliver_cancel(p: Process) -> None:  # noqa: RUF029
    try:
        p.terminate()
    except OSError as exc:
        warnings.warn(
            RuntimeWarning(f"TerminateProcess on {p!r} failed with: {exc!r}"),
            stacklevel=1,
        )


async def _posix_deliver_cancel(p: Process) -> None:
    try:
        p.terminate()
        await trio.sleep(5)
        warnings.warn(
            RuntimeWarning(
                f"process {p!r} ignored SIGTERM for 5 seconds. "
                "(Maybe you should pass a custom deliver_cancel?) "
                "Trying SIGKILL.",
            ),
            stacklevel=1,
        )
        p.kill()
    except OSError as exc:
        warnings.warn(
            RuntimeWarning(f"tried to kill process {p!r}, but failed with: {exc!r}"),
            stacklevel=1,
        )


# Use a private name, so we can declare platform-specific stubs below.
# This is also the signature read by Sphinx
async def _run_process(
    command: StrOrBytesPath | Sequence[StrOrBytesPath],
    *,
    stdin: bytes | bytearray | memoryview | int | HasFileno | None = b"",
    capture_stdout: bool = False,
    capture_stderr: bool = False,
    check: bool = True,
    deliver_cancel: Callable[[Process], Awaitable[object]] | None = None,
    task_status: TaskStatus[Process] = trio.TASK_STATUS_IGNORED,
    **options: object,
) -> subprocess.CompletedProcess[bytes]:
    """Run ``command`` in a subprocess and wait for it to complete.

    This function can be called in two different ways.

    One option is a direct call, like::

        completed_process_info = await trio.run_process(...)

    In this case, it returns a :class:`subprocess.CompletedProcess` instance
    describing the results. Use this if you want to treat a process like a
    function call.

    The other option is to run it as a task using `Nursery.start` – the enhanced version
    of `~Nursery.start_soon` that lets a task pass back a value during startup::

        process = await nursery.start(trio.run_process, ...)

    In this case, `~Nursery.start` returns a `Process` object that you can use
    to interact with the process while it's running. Use this if you want to
    treat a process like a background task.

    Either way, `run_process` makes sure that the process has exited before
    returning, handles cancellation, optionally checks for errors, and
    provides some convenient shorthands for dealing with the child's
    input/output.

    **Input:** `run_process` supports all the same ``stdin=`` arguments as
    `subprocess.Popen`. In addition, if you simply want to pass in some fixed
    data, you can pass a plain `bytes` object, and `run_process` will take
    care of setting up a pipe, feeding in the data you gave, and then sending
    end-of-file. The default is ``b""``, which means that the child will receive
    an empty stdin. If you want the child to instead read from the parent's
    stdin, use ``stdin=None``.

    **Output:** By default, any output produced by the subprocess is
    passed through to the standard output and error streams of the
    parent Trio process.

    When calling `run_process` directly, you can capture the subprocess's output by
    passing ``capture_stdout=True`` to capture the subprocess's standard output, and/or
    ``capture_stderr=True`` to capture its standard error. Captured data is collected up
    by Trio into an in-memory buffer, and then provided as the
    :attr:`~subprocess.CompletedProcess.stdout` and/or
    :attr:`~subprocess.CompletedProcess.stderr` attributes of the returned
    :class:`~subprocess.CompletedProcess` object. The value for any stream that was not
    captured will be ``None``.

    If you want to capture both stdout and stderr while keeping them
    separate, pass ``capture_stdout=True, capture_stderr=True``.

    If you want to capture both stdout and stderr but mixed together
    in the order they were printed, use: ``capture_stdout=True, stderr=subprocess.STDOUT``.
    This directs the child's stderr into its stdout, so the combined
    output will be available in the `~subprocess.CompletedProcess.stdout`
    attribute.

    If you're using ``await nursery.start(trio.run_process, ...)`` and want to capture
    the subprocess's output for further processing, then use ``stdout=subprocess.PIPE``
    and then make sure to read the data out of the `Process.stdout` stream. If you want
    to capture stderr separately, use ``stderr=subprocess.PIPE``. If you want to capture
    both, but mixed together in the correct order, use ``stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT``.

    **Error checking:** If the subprocess exits with a nonzero status
    code, indicating failure, :func:`run_process` raises a
    :exc:`subprocess.CalledProcessError` exception rather than
    returning normally. The captured outputs are still available as
    the :attr:`~subprocess.CalledProcessError.stdout` and
    :attr:`~subprocess.CalledProcessError.stderr` attributes of that
    exception.  To disable this behavior, so that :func:`run_process`
    returns normally even if the subprocess exits abnormally, pass ``check=False``.

    Note that this can make the ``capture_stdout`` and ``capture_stderr``
    arguments useful even when starting `run_process` as a task: if you only
    care about the output if the process fails, then you can enable capturing
    and then read the output off of the `~subprocess.CalledProcessError`.

    **Cancellation:** If cancelled, `run_process` sends a termination
    request to the subprocess, then waits for it to fully exit. The
    ``deliver_cancel`` argument lets you control how the process is terminated.

    .. note:: `run_process` is intentionally similar to the standard library
       `subprocess.run`, but some of the defaults are different. Specifically, we
       default to:

       - ``check=True``, because `"errors should never pass silently / unless
         explicitly silenced" <https://www.python.org/dev/peps/pep-0020/>`__.

       - ``stdin=b""``, because it produces less-confusing results if a subprocess
         unexpectedly tries to read from stdin.

       To get the `subprocess.run` semantics, use ``check=False, stdin=None``.

    Args:
      command (list or str): The command to run. Typically this is a
          sequence of strings such as ``['ls', '-l', 'directory with spaces']``,
          where the first element names the executable to invoke and the other
          elements specify its arguments. With ``shell=True`` in the
          ``**options``, or on Windows, ``command`` may alternatively
          be a string, which will be parsed following platform-dependent
          :ref:`quoting rules <subprocess-quoting>`.

      stdin (:obj:`bytes`, subprocess.PIPE, file descriptor, or None): The
          bytes to provide to the subprocess on its standard input stream, or
          ``None`` if the subprocess's standard input should come from the
          same place as the parent Trio process's standard input. As is the
          case with the :mod:`subprocess` module, you can also pass a file
          descriptor or an object with a ``fileno()`` method, in which case
          the subprocess's standard input will come from that file.

          When starting `run_process` as a background task, you can also use
          ``stdin=subprocess.PIPE``, in which case `Process.stdin` will be a
          `~trio.abc.SendStream` that you can use to send data to the child.

      capture_stdout (bool): If true, capture the bytes that the subprocess
          writes to its standard output stream and return them in the
          `~subprocess.CompletedProcess.stdout` attribute of the returned
          `subprocess.CompletedProcess` or `subprocess.CalledProcessError`.

      capture_stderr (bool): If true, capture the bytes that the subprocess
          writes to its standard error stream and return them in the
          `~subprocess.CompletedProcess.stderr` attribute of the returned
          `~subprocess.CompletedProcess` or `subprocess.CalledProcessError`.

      check (bool): If false, don't validate that the subprocess exits
          successfully. You should be sure to check the
          ``returncode`` attribute of the returned object if you pass
          ``check=False``, so that errors don't pass silently.

      deliver_cancel (async function or None): If `run_process` is cancelled,
          then it needs to kill the child process. There are multiple ways to
          do this, so we let you customize it.

          If you pass None (the default), then the behavior depends on the
          platform:

          - On Windows, Trio calls ``TerminateProcess``, which should kill the
            process immediately.

          - On Unix-likes, the default behavior is to send a ``SIGTERM``, wait
            5 seconds, and send a ``SIGKILL``.

          Alternatively, you can customize this behavior by passing in an
          arbitrary async function, which will be called with the `Process`
          object as an argument. For example, the default Unix behavior could
          be implemented like this::

             async def my_deliver_cancel(process):
                 process.send_signal(signal.SIGTERM)
                 await trio.sleep(5)
                 process.send_signal(signal.SIGKILL)

          When the process actually exits, the ``deliver_cancel`` function
          will automatically be cancelled – so if the process exits after
          ``SIGTERM``, then we'll never reach the ``SIGKILL``.

          In any case, `run_process` will always wait for the child process to
          exit before raising `Cancelled`.

      **options: :func:`run_process` also accepts any :ref:`general subprocess
          options <subprocess-options>` and passes them on to the
          :class:`~trio.Process` constructor. This includes the
          ``stdout`` and ``stderr`` options, which provide additional
          redirection possibilities such as ``stderr=subprocess.STDOUT``,
          ``stdout=subprocess.DEVNULL``, or file descriptors.

    Returns:

      When called normally – a `subprocess.CompletedProcess` instance
      describing the return code and outputs.

      When called via `Nursery.start` – a `trio.Process` instance.

    Raises:
      UnicodeError: if ``stdin`` is specified as a Unicode string, rather
          than bytes
      ValueError: if multiple redirections are specified for the same
          stream, e.g., both ``capture_stdout=True`` and
          ``stdout=subprocess.DEVNULL``
      subprocess.CalledProcessError: if ``check=False`` is not passed
          and the process exits with a nonzero exit status
      OSError: if an error is encountered starting or communicating with
          the process
      ExceptionGroup: if exceptions occur in ``deliver_cancel``,
          or when exceptions occur when communicating with the subprocess.
          If strict_exception_groups is set to false in the global context,
          which is deprecated, then single exceptions will be collapsed.

    .. note:: The child process runs in the same process group as the parent
       Trio process, so a Ctrl+C will be delivered simultaneously to both
       parent and child. If you don't want this behavior, consult your
       platform's documentation for starting child processes in a different
       process group.

    """

    if isinstance(stdin, str):
        raise UnicodeError("process stdin must be bytes, not str")
    if task_status is trio.TASK_STATUS_IGNORED:
        if stdin is subprocess.PIPE:
            raise ValueError(
                "stdout=subprocess.PIPE is only valid with nursery.start, "
                "since that's the only way to access the pipe; use nursery.start "
                "or pass the data you want to write directly",
            )
        if options.get("stdout") is subprocess.PIPE:
            raise ValueError(
                "stdout=subprocess.PIPE is only valid with nursery.start, "
                "since that's the only way to access the pipe",
            )
        if options.get("stderr") is subprocess.PIPE:
            raise ValueError(
                "stderr=subprocess.PIPE is only valid with nursery.start, "
                "since that's the only way to access the pipe",
            )
    if isinstance(stdin, (bytes, bytearray, memoryview)):
        input_ = stdin
        options["stdin"] = subprocess.PIPE
    else:
        # stdin should be something acceptable to Process
        # (None, DEVNULL, a file descriptor, etc) and Process
        # will raise if it's not
        input_ = None
        options["stdin"] = stdin

    if capture_stdout:
        if "stdout" in options:
            raise ValueError("can't specify both stdout and capture_stdout")
        options["stdout"] = subprocess.PIPE
    if capture_stderr:
        if "stderr" in options:
            raise ValueError("can't specify both stderr and capture_stderr")
        options["stderr"] = subprocess.PIPE

    if deliver_cancel is None:
        if os.name == "nt":
            deliver_cancel = _windows_deliver_cancel
        else:
            assert os.name == "posix"
            deliver_cancel = _posix_deliver_cancel

    stdout_chunks: list[bytes | bytearray] = []
    stderr_chunks: list[bytes | bytearray] = []

    async def feed_input(stream: SendStream) -> None:
        async with stream:
            try:
                assert input_ is not None
                await stream.send_all(input_)
            except trio.BrokenResourceError:
                pass

    async def read_output(
        stream: ReceiveStream,
        chunks: list[bytes | bytearray],
    ) -> None:
        async with stream:
            async for chunk in stream:
                chunks.append(chunk)  # noqa: PERF401

    # Opening the process does not need to be inside the nursery, so we put it outside
    # so any exceptions get directly seen by users.
    # options needs a complex TypedDict. The overload error only occurs on Unix.
    proc = await open_process(command, **options)  # type: ignore[arg-type, call-overload, unused-ignore]
    async with trio.open_nursery() as nursery:
        try:
            if input_ is not None:
                assert proc.stdin is not None
                nursery.start_soon(feed_input, proc.stdin)
                proc.stdin = None
                proc.stdio = None
            if capture_stdout:
                assert proc.stdout is not None
                nursery.start_soon(read_output, proc.stdout, stdout_chunks)
                proc.stdout = None
                proc.stdio = None
            if capture_stderr:
                assert proc.stderr is not None
                nursery.start_soon(read_output, proc.stderr, stderr_chunks)
                proc.stderr = None
            task_status.started(proc)
            await proc.wait()
        except BaseException:
            with trio.CancelScope(shield=True):
                killer_cscope = trio.CancelScope(shield=True)

                async def killer() -> None:
                    with killer_cscope:
                        await deliver_cancel(proc)

                nursery.start_soon(killer)
                await proc.wait()
                killer_cscope.cancel()
                raise

    stdout = b"".join(stdout_chunks) if capture_stdout else None
    stderr = b"".join(stderr_chunks) if capture_stderr else None

    if proc.returncode and check:
        raise subprocess.CalledProcessError(
            proc.returncode,
            proc.args,
            output=stdout,
            stderr=stderr,
        )
    else:
        assert proc.returncode is not None
        return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)


# There's a lot of duplication here because type checkers don't
# have a good way to represent overloads that differ only
# slightly. A cheat sheet:
#
# - on Windows, command is Union[str, Sequence[str]];
#   on Unix, command is str if shell=True and Sequence[str] otherwise
#
# - on Windows, there are startupinfo and creationflags options;
#   on Unix, there are preexec_fn, restore_signals, start_new_session,
#            pass_fds, group (3.9+), extra_groups (3.9+), user (3.9+),
#            umask (3.9+), pipesize (3.10+), process_group (3.11+)
#
# - run_process() has the signature of open_process() plus arguments
#   capture_stdout, capture_stderr, check, deliver_cancel, the ability
#   to pass bytes as stdin, and the ability to run in `nursery.start`


class GeneralProcessArgs(TypedDict, total=False):
    """Arguments shared between all runs."""

    stdout: int | HasFileno | None
    stderr: int | HasFileno | None
    close_fds: bool
    cwd: StrOrBytesPath | None
    env: Mapping[str, str] | None
    executable: StrOrBytesPath | None


if TYPE_CHECKING:
    if sys.platform == "win32":

        class WindowsProcessArgs(GeneralProcessArgs, total=False):
            """Arguments shared between all Windows runs."""

            shell: bool
            startupinfo: subprocess.STARTUPINFO | None
            creationflags: int

        async def open_process(
            command: StrOrBytesPath | Sequence[StrOrBytesPath],
            *,
            stdin: int | HasFileno | None = None,
            **kwargs: Unpack[WindowsProcessArgs],
        ) -> trio.Process:
            r"""Execute a child program in a new process.

            After construction, you can interact with the child process by writing data to its
            `~trio.Process.stdin` stream (a `~trio.abc.SendStream`), reading data from its
            `~trio.Process.stdout` and/or `~trio.Process.stderr` streams (both
            `~trio.abc.ReceiveStream`\s), sending it signals using `~trio.Process.terminate`,
            `~trio.Process.kill`, or `~trio.Process.send_signal`, and waiting for it to exit
            using `~trio.Process.wait`. See `trio.Process` for details.

            Each standard stream is only available if you specify that a pipe should be created
            for it. For example, if you pass ``stdin=subprocess.PIPE``, you can write to the
            `~trio.Process.stdin` stream, else `~trio.Process.stdin` will be ``None``.

            Unlike `trio.run_process`, this function doesn't do any kind of automatic
            management of the child process. It's up to you to implement whatever semantics you
            want.

            Args:
              command (list or str): The command to run. Typically this is a
                  sequence of strings such as ``['ls', '-l', 'directory with spaces']``,
                  where the first element names the executable to invoke and the other
                  elements specify its arguments. With ``shell=True`` in the
                  ``**options``, or on Windows, ``command`` may alternatively
                  be a string, which will be parsed following platform-dependent
                  :ref:`quoting rules <subprocess-quoting>`.
              stdin: Specifies what the child process's standard input
                  stream should connect to: output written by the parent
                  (``subprocess.PIPE``), nothing (``subprocess.DEVNULL``),
                  or an open file (pass a file descriptor or something whose
                  ``fileno`` method returns one). If ``stdin`` is unspecified,
                  the child process will have the same standard input stream
                  as its parent.
              stdout: Like ``stdin``, but for the child process's standard output
                  stream.
              stderr: Like ``stdin``, but for the child process's standard error
                  stream. An additional value ``subprocess.STDOUT`` is supported,
                  which causes the child's standard output and standard error
                  messages to be intermixed on a single standard output stream,
                  attached to whatever the ``stdout`` option says to attach it to.
              **options: Other :ref:`general subprocess options <subprocess-options>`
                  are also accepted.

            Returns:
              A new `trio.Process` object.

            Raises:
              OSError: if the process spawning fails, for example because the
                 specified command could not be found.

            """
            ...

        async def run_process(
            command: StrOrBytesPath | Sequence[StrOrBytesPath],
            *,
            task_status: TaskStatus[Process] = trio.TASK_STATUS_IGNORED,
            stdin: bytes | bytearray | memoryview | int | HasFileno | None = None,
            capture_stdout: bool = False,
            capture_stderr: bool = False,
            check: bool = True,
            deliver_cancel: Callable[[Process], Awaitable[object]] | None = None,
            **kwargs: Unpack[WindowsProcessArgs],
        ) -> subprocess.CompletedProcess[bytes]:
            """Run ``command`` in a subprocess and wait for it to complete.

            This function can be called in two different ways.

            One option is a direct call, like::

                completed_process_info = await trio.run_process(...)

            In this case, it returns a :class:`subprocess.CompletedProcess` instance
            describing the results. Use this if you want to treat a process like a
            function call.

            The other option is to run it as a task using `Nursery.start` – the enhanced version
            of `~Nursery.start_soon` that lets a task pass back a value during startup::

                process = await nursery.start(trio.run_process, ...)

            In this case, `~Nursery.start` returns a `Process` object that you can use
            to interact with the process while it's running. Use this if you want to
            treat a process like a background task.

            Either way, `run_process` makes sure that the process has exited before
            returning, handles cancellation, optionally checks for errors, and
            provides some convenient shorthands for dealing with the child's
            input/output.

            **Input:** `run_process` supports all the same ``stdin=`` arguments as
            `subprocess.Popen`. In addition, if you simply want to pass in some fixed
            data, you can pass a plain `bytes` object, and `run_process` will take
            care of setting up a pipe, feeding in the data you gave, and then sending
            end-of-file. The default is ``b""``, which means that the child will receive
            an empty stdin. If you want the child to instead read from the parent's
            stdin, use ``stdin=None``.

            **Output:** By default, any output produced by the subprocess is
            passed through to the standard output and error streams of the
            parent Trio process.

            When calling `run_process` directly, you can capture the subprocess's output by
            passing ``capture_stdout=True`` to capture the subprocess's standard output, and/or
            ``capture_stderr=True`` to capture its standard error. Captured data is collected up
            by Trio into an in-memory buffer, and then provided as the
            :attr:`~subprocess.CompletedProcess.stdout` and/or
            :attr:`~subprocess.CompletedProcess.stderr` attributes of the returned
            :class:`~subprocess.CompletedProcess` object. The value for any stream that was not
            captured will be ``None``.

            If you want to capture both stdout and stderr while keeping them
            separate, pass ``capture_stdout=True, capture_stderr=True``.

            If you want to capture both stdout and stderr but mixed together
            in the order they were printed, use: ``capture_stdout=True, stderr=subprocess.STDOUT``.
            This directs the child's stderr into its stdout, so the combined
            output will be available in the `~subprocess.CompletedProcess.stdout`
            attribute.

            If you're using ``await nursery.start(trio.run_process, ...)`` and want to capture
            the subprocess's output for further processing, then use ``stdout=subprocess.PIPE``
            and then make sure to read the data out of the `Process.stdout` stream. If you want
            to capture stderr separately, use ``stderr=subprocess.PIPE``. If you want to capture
            both, but mixed together in the correct order, use ``stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT``.

            **Error checking:** If the subprocess exits with a nonzero status
            code, indicating failure, :func:`run_process` raises a
            :exc:`subprocess.CalledProcessError` exception rather than
            returning normally. The captured outputs are still available as
            the :attr:`~subprocess.CalledProcessError.stdout` and
            :attr:`~subprocess.CalledProcessError.stderr` attributes of that
            exception.  To disable this behavior, so that :func:`run_process`
            returns normally even if the subprocess exits abnormally, pass ``check=False``.

            Note that this can make the ``capture_stdout`` and ``capture_stderr``
            arguments useful even when starting `run_process` as a task: if you only
            care about the output if the process fails, then you can enable capturing
            and then read the output off of the `~subprocess.CalledProcessError`.

            **Cancellation:** If cancelled, `run_process` sends a termination
            request to the subprocess, then waits for it to fully exit. The
            ``deliver_cancel`` argument lets you control how the process is terminated.

            .. note:: `run_process` is intentionally similar to the standard library
               `subprocess.run`, but some of the defaults are different. Specifically, we
               default to:

               - ``check=True``, because `"errors should never pass silently / unless
                 explicitly silenced" <https://www.python.org/dev/peps/pep-0020/>`__.

               - ``stdin=b""``, because it produces less-confusing results if a subprocess
                 unexpectedly tries to read from stdin.

               To get the `subprocess.run` semantics, use ``check=False, stdin=None``.

            Args:
              command (list or str): The command to run. Typically this is a
                  sequence of strings such as ``['ls', '-l', 'directory with spaces']``,
                  where the first element names the executable to invoke and the other
                  elements specify its arguments. With ``shell=True`` in the
                  ``**options``, or on Windows, ``command`` may alternatively
                  be a string, which will be parsed following platform-dependent
                  :ref:`quoting rules <subprocess-quoting>`.

              stdin (:obj:`bytes`, subprocess.PIPE, file descriptor, or None): The
                  bytes to provide to the subprocess on its standard input stream, or
                  ``None`` if the subprocess's standard input should come from the
                  same place as the parent Trio process's standard input. As is the
                  case with the :mod:`subprocess` module, you can also pass a file
                  descriptor or an object with a ``fileno()`` method, in which case
                  the subprocess's standard input will come from that file.

                  When starting `run_process` as a background task, you can also use
                  ``stdin=subprocess.PIPE``, in which case `Process.stdin` will be a
                  `~trio.abc.SendStream` that you can use to send data to the child.

              capture_stdout (bool): If true, capture the bytes that the subprocess
                  writes to its standard output stream and return them in the
                  `~subprocess.CompletedProcess.stdout` attribute of the returned
                  `subprocess.CompletedProcess` or `subprocess.CalledProcessError`.

              capture_stderr (bool): If true, capture the bytes that the subprocess
                  writes to its standard error stream and return them in the
                  `~subprocess.CompletedProcess.stderr` attribute of the returned
                  `~subprocess.CompletedProcess` or `subprocess.CalledProcessError`.

              check (bool): If false, don't validate that the subprocess exits
                  successfully. You should be sure to check the
                  ``returncode`` attribute of the returned object if you pass
                  ``check=False``, so that errors don't pass silently.

              deliver_cancel (async function or None): If `run_process` is cancelled,
                  then it needs to kill the child process. There are multiple ways to
                  do this, so we let you customize it.

                  If you pass None (the default), then the behavior depends on the
                  platform:

                  - On Windows, Trio calls ``TerminateProcess``, which should kill the
                    process immediately.

                  - On Unix-likes, the default behavior is to send a ``SIGTERM``, wait
                    5 seconds, and send a ``SIGKILL``.

                  Alternatively, you can customize this behavior by passing in an
                  arbitrary async function, which will be called with the `Process`
                  object as an argument. For example, the default Unix behavior could
                  be implemented like this::

                     async def my_deliver_cancel(process):
                         process.send_signal(signal.SIGTERM)
                         await trio.sleep(5)
                         process.send_signal(signal.SIGKILL)

                  When the process actually exits, the ``deliver_cancel`` function
                  will automatically be cancelled – so if the process exits after
                  ``SIGTERM``, then we'll never reach the ``SIGKILL``.

                  In any case, `run_process` will always wait for the child process to
                  exit before raising `Cancelled`.

              **options: :func:`run_process` also accepts any :ref:`general subprocess
                  options <subprocess-options>` and passes them on to the
                  :class:`~trio.Process` constructor. This includes the
                  ``stdout`` and ``stderr`` options, which provide additional
                  redirection possibilities such as ``stderr=subprocess.STDOUT``,
                  ``stdout=subprocess.DEVNULL``, or file descriptors.

            Returns:

              When called normally – a `subprocess.CompletedProcess` instance
              describing the return code and outputs.

              When called via `Nursery.start` – a `trio.Process` instance.

            Raises:
              UnicodeError: if ``stdin`` is specified as a Unicode string, rather
                  than bytes
              ValueError: if multiple redirections are specified for the same
                  stream, e.g., both ``capture_stdout=True`` and
                  ``stdout=subprocess.DEVNULL``
              subprocess.CalledProcessError: if ``check=False`` is not passed
                  and the process exits with a nonzero exit status
              OSError: if an error is encountered starting or communicating with
                  the process

            .. note:: The child process runs in the same process group as the parent
               Trio process, so a Ctrl+C will be delivered simultaneously to both
               parent and child. If you don't want this behavior, consult your
               platform's documentation for starting child processes in a different
               process group.

            """
            ...

    else:  # Unix
        # pyright doesn't give any error about overloads missing docstrings as they're
        # overloads. But might still be a problem for other static analyzers / docstring
        # readers (?)

        class UnixProcessArgs3_9(GeneralProcessArgs, total=False):
            """Arguments shared between all Unix runs."""

            preexec_fn: Callable[[], object] | None
            restore_signals: bool
            start_new_session: bool
            pass_fds: Sequence[int]

            # 3.9+
            group: str | int | None
            extra_groups: Iterable[str | int] | None
            user: str | int | None
            umask: int

        class UnixProcessArgs3_10(UnixProcessArgs3_9, total=False):
            """Arguments shared between all Unix runs on 3.10+."""

            pipesize: int

        class UnixProcessArgs3_11(UnixProcessArgs3_10, total=False):
            """Arguments shared between all Unix runs on 3.11+."""

            process_group: int | None

        class UnixRunProcessMixin(TypedDict, total=False):
            """Arguments unique to run_process on Unix."""

            task_status: TaskStatus[Process]
            capture_stdout: bool
            capture_stderr: bool
            check: bool
            deliver_cancel: Callable[[Process], Awaitable[None]] | None

        # TODO: once https://github.com/python/mypy/issues/18692 is
        #       fixed, move the `UnixRunProcessArgs` definition down.
        if sys.version_info >= (3, 11):
            UnixProcessArgs = UnixProcessArgs3_11

            class UnixRunProcessArgs(UnixProcessArgs3_11, UnixRunProcessMixin):
                """Arguments for run_process on Unix with 3.11+"""

        elif sys.version_info >= (3, 10):
            UnixProcessArgs = UnixProcessArgs3_10

            class UnixRunProcessArgs(UnixProcessArgs3_10, UnixRunProcessMixin):
                """Arguments for run_process on Unix with 3.10+"""

        else:
            UnixProcessArgs = UnixProcessArgs3_9

            class UnixRunProcessArgs(UnixProcessArgs3_9, UnixRunProcessMixin):
                """Arguments for run_process on Unix with 3.9+"""

        @overload  # type: ignore[no-overload-impl]
        async def open_process(
            command: StrOrBytesPath,
            *,
            stdin: int | HasFileno | None = None,
            shell: Literal[True],
            **kwargs: Unpack[UnixProcessArgs],
        ) -> trio.Process: ...

        @overload
        async def open_process(
            command: Sequence[StrOrBytesPath],
            *,
            stdin: int | HasFileno | None = None,
            shell: bool = False,
            **kwargs: Unpack[UnixProcessArgs],
        ) -> trio.Process: ...

        @overload  # type: ignore[no-overload-impl]
        async def run_process(
            command: StrOrBytesPath,
            *,
            stdin: bytes | bytearray | memoryview | int | HasFileno | None = None,
            shell: Literal[True],
            **kwargs: Unpack[UnixRunProcessArgs],
        ) -> subprocess.CompletedProcess[bytes]: ...

        @overload
        async def run_process(
            command: Sequence[StrOrBytesPath],
            *,
            stdin: bytes | bytearray | memoryview | int | HasFileno | None = None,
            shell: bool = False,
            **kwargs: Unpack[UnixRunProcessArgs],
        ) -> subprocess.CompletedProcess[bytes]: ...

else:
    # At runtime, use the actual implementations.
    open_process = _open_process
    open_process.__name__ = open_process.__qualname__ = "open_process"

    run_process = _run_process
    run_process.__name__ = run_process.__qualname__ = "run_process"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\timedeltas.py ===
from __future__ import annotations

from datetime import timedelta
import operator
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._libs import (
    lib,
    tslibs,
)
from pandas._libs.tslibs import (
    NaT,
    NaTType,
    Tick,
    Timedelta,
    astype_overflowsafe,
    get_supported_dtype,
    iNaT,
    is_supported_dtype,
    periods_per_second,
)
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas._libs.tslibs.fields import (
    get_timedelta_days,
    get_timedelta_field,
)
from pandas._libs.tslibs.timedeltas import (
    array_to_timedelta64,
    floordiv_object_array,
    ints_to_pytimedelta,
    parse_timedelta_unit,
    truediv_object_array,
)
from pandas.compat.numpy import function as nv
from pandas.util._validators import validate_endpoints

from pandas.core.dtypes.common import (
    TD64NS_DTYPE,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import isna

from pandas.core import (
    nanops,
    roperator,
)
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.core.ops.common import unpack_zerodim_and_defer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas._typing import (
        AxisInt,
        DateTimeErrorChoices,
        DtypeObj,
        NpDtype,
        Self,
        npt,
    )

    from pandas import DataFrame

import textwrap


def _field_accessor(name: str, alias: str, docstring: str):
    def f(self) -> np.ndarray:
        values = self.asi8
        if alias == "days":
            result = get_timedelta_days(values, reso=self._creso)
        else:
            # error: Incompatible types in assignment (
            # expression has type "ndarray[Any, dtype[signedinteger[_32Bit]]]",
            # variable has type "ndarray[Any, dtype[signedinteger[_64Bit]]]
            result = get_timedelta_field(values, alias, reso=self._creso)  # type: ignore[assignment]
        if self._hasna:
            result = self._maybe_mask_results(
                result, fill_value=None, convert="float64"
            )

        return result

    f.__name__ = name
    f.__doc__ = f"\n{docstring}\n"
    return property(f)


class TimedeltaArray(dtl.TimelikeOps):
    """
    Pandas ExtensionArray for timedelta data.

    .. warning::

       TimedeltaArray is currently experimental, and its API may change
       without warning. In particular, :attr:`TimedeltaArray.dtype` is
       expected to change to be an instance of an ``ExtensionDtype``
       subclass.

    Parameters
    ----------
    values : array-like
        The timedelta data.

    dtype : numpy.dtype
        Currently, only ``numpy.dtype("timedelta64[ns]")`` is accepted.
    freq : Offset, optional
    copy : bool, default False
        Whether to copy the underlying array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.arrays.TimedeltaArray._from_sequence(pd.TimedeltaIndex(['1h', '2h']))
    <TimedeltaArray>
    ['0 days 01:00:00', '0 days 02:00:00']
    Length: 2, dtype: timedelta64[ns]
    """

    _typ = "timedeltaarray"
    _internal_fill_value = np.timedelta64("NaT", "ns")
    _recognized_scalars = (timedelta, np.timedelta64, Tick)
    _is_recognized_dtype = lambda x: lib.is_np_dtype(x, "m")
    _infer_matches = ("timedelta", "timedelta64")

    @property
    def _scalar_type(self) -> type[Timedelta]:
        return Timedelta

    __array_priority__ = 1000
    # define my properties & methods for delegation
    _other_ops: list[str] = []
    _bool_ops: list[str] = []
    _object_ops: list[str] = ["freq"]
    _field_ops: list[str] = ["days", "seconds", "microseconds", "nanoseconds"]
    _datetimelike_ops: list[str] = _field_ops + _object_ops + _bool_ops + ["unit"]
    _datetimelike_methods: list[str] = [
        "to_pytimedelta",
        "total_seconds",
        "round",
        "floor",
        "ceil",
        "as_unit",
    ]

    # Note: ndim must be defined to ensure NaT.__richcmp__(TimedeltaArray)
    #  operates pointwise.

    def _box_func(self, x: np.timedelta64) -> Timedelta | NaTType:
        y = x.view("i8")
        if y == NaT._value:
            return NaT
        return Timedelta._from_value_and_reso(y, reso=self._creso)

    @property
    # error: Return type "dtype" of "dtype" incompatible with return type
    # "ExtensionDtype" in supertype "ExtensionArray"
    def dtype(self) -> np.dtype[np.timedelta64]:  # type: ignore[override]
        """
        The dtype for the TimedeltaArray.

        .. warning::

           A future version of pandas will change dtype to be an instance
           of a :class:`pandas.api.extensions.ExtensionDtype` subclass,
           not a ``numpy.dtype``.

        Returns
        -------
        numpy.dtype
        """
        return self._ndarray.dtype

    # ----------------------------------------------------------------
    # Constructors

    _freq = None
    _default_dtype = TD64NS_DTYPE  # used in TimeLikeOps.__init__

    @classmethod
    def _validate_dtype(cls, values, dtype):
        # used in TimeLikeOps.__init__
        dtype = _validate_td64_dtype(dtype)
        _validate_td64_dtype(values.dtype)
        if dtype != values.dtype:
            raise ValueError("Values resolution does not match dtype.")
        return dtype

    # error: Signature of "_simple_new" incompatible with supertype "NDArrayBacked"
    @classmethod
    def _simple_new(  # type: ignore[override]
        cls,
        values: npt.NDArray[np.timedelta64],
        freq: Tick | None = None,
        dtype: np.dtype[np.timedelta64] = TD64NS_DTYPE,
    ) -> Self:
        # Require td64 dtype, not unit-less, matching values.dtype
        assert lib.is_np_dtype(dtype, "m")
        assert not tslibs.is_unitless(dtype)
        assert isinstance(values, np.ndarray), type(values)
        assert dtype == values.dtype
        assert freq is None or isinstance(freq, Tick)

        result = super()._simple_new(values=values, dtype=dtype)
        result._freq = freq
        return result

    @classmethod
    def _from_sequence(cls, data, *, dtype=None, copy: bool = False) -> Self:
        if dtype:
            dtype = _validate_td64_dtype(dtype)

        data, freq = sequence_to_td64ns(data, copy=copy, unit=None)

        if dtype is not None:
            data = astype_overflowsafe(data, dtype=dtype, copy=False)

        return cls._simple_new(data, dtype=data.dtype, freq=freq)

    @classmethod
    def _from_sequence_not_strict(
        cls,
        data,
        *,
        dtype=None,
        copy: bool = False,
        freq=lib.no_default,
        unit=None,
    ) -> Self:
        """
        _from_sequence_not_strict but without responsibility for finding the
        result's `freq`.
        """
        if dtype:
            dtype = _validate_td64_dtype(dtype)

        assert unit not in ["Y", "y", "M"]  # caller is responsible for checking

        data, inferred_freq = sequence_to_td64ns(data, copy=copy, unit=unit)

        if dtype is not None:
            data = astype_overflowsafe(data, dtype=dtype, copy=False)

        result = cls._simple_new(data, dtype=data.dtype, freq=inferred_freq)

        result._maybe_pin_freq(freq, {})
        return result

    @classmethod
    def _generate_range(
        cls, start, end, periods, freq, closed=None, *, unit: str | None = None
    ) -> Self:
        periods = dtl.validate_periods(periods)
        if freq is None and any(x is None for x in [periods, start, end]):
            raise ValueError("Must provide freq argument if no data is supplied")

        if com.count_not_none(start, end, periods, freq) != 3:
            raise ValueError(
                "Of the four parameters: start, end, periods, "
                "and freq, exactly three must be specified"
            )

        if start is not None:
            start = Timedelta(start).as_unit("ns")

        if end is not None:
            end = Timedelta(end).as_unit("ns")

        if unit is not None:
            if unit not in ["s", "ms", "us", "ns"]:
                raise ValueError("'unit' must be one of 's', 'ms', 'us', 'ns'")
        else:
            unit = "ns"

        if start is not None and unit is not None:
            start = start.as_unit(unit, round_ok=False)
        if end is not None and unit is not None:
            end = end.as_unit(unit, round_ok=False)

        left_closed, right_closed = validate_endpoints(closed)

        if freq is not None:
            index = generate_regular_range(start, end, periods, freq, unit=unit)
        else:
            index = np.linspace(start._value, end._value, periods).astype("i8")

        if not left_closed:
            index = index[1:]
        if not right_closed:
            index = index[:-1]

        td64values = index.view(f"m8[{unit}]")
        return cls._simple_new(td64values, dtype=td64values.dtype, freq=freq)

    # ----------------------------------------------------------------
    # DatetimeLike Interface

    def _unbox_scalar(self, value) -> np.timedelta64:
        if not isinstance(value, self._scalar_type) and value is not NaT:
            raise ValueError("'value' should be a Timedelta.")
        self._check_compatible_with(value)
        if value is NaT:
            return np.timedelta64(value._value, self.unit)
        else:
            return value.as_unit(self.unit).asm8

    def _scalar_from_string(self, value) -> Timedelta | NaTType:
        return Timedelta(value)

    def _check_compatible_with(self, other) -> None:
        # we don't have anything to validate.
        pass

    # ----------------------------------------------------------------
    # Array-Like / EA-Interface Methods

    def astype(self, dtype, copy: bool = True):
        # We handle
        #   --> timedelta64[ns]
        #   --> timedelta64
        # DatetimeLikeArrayMixin super call handles other cases
        dtype = pandas_dtype(dtype)

        if lib.is_np_dtype(dtype, "m"):
            if dtype == self.dtype:
                if copy:
                    return self.copy()
                return self

            if is_supported_dtype(dtype):
                # unit conversion e.g. timedelta64[s]
                res_values = astype_overflowsafe(self._ndarray, dtype, copy=False)
                return type(self)._simple_new(
                    res_values, dtype=res_values.dtype, freq=self.freq
                )
            else:
                raise ValueError(
                    f"Cannot convert from {self.dtype} to {dtype}. "
                    "Supported resolutions are 's', 'ms', 'us', 'ns'"
                )

        return dtl.DatetimeLikeArrayMixin.astype(self, dtype, copy=copy)

    def __iter__(self) -> Iterator:
        if self.ndim > 1:
            for i in range(len(self)):
                yield self[i]
        else:
            # convert in chunks of 10k for efficiency
            data = self._ndarray
            length = len(self)
            chunksize = 10000
            chunks = (length // chunksize) + 1
            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, length)
                converted = ints_to_pytimedelta(data[start_i:end_i], box=True)
                yield from converted

    # ----------------------------------------------------------------
    # Reductions

    def sum(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        keepdims: bool = False,
        initial=None,
        skipna: bool = True,
        min_count: int = 0,
    ):
        nv.validate_sum(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims, "initial": initial}
        )

        result = nanops.nansum(
            self._ndarray, axis=axis, skipna=skipna, min_count=min_count
        )
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
        if axis is None or self.ndim == 1:
            return self._box_func(result)
        return self._from_backing_data(result)

    # ----------------------------------------------------------------
    # Accumulations

    def _accumulate(self, name: str, *, skipna: bool = True, **kwargs):
        if name == "cumsum":
            op = getattr(datetimelike_accumulations, name)
            result = op(self._ndarray.copy(), skipna=skipna, **kwargs)

            return type(self)._simple_new(result, freq=None, dtype=self.dtype)
        elif name == "cumprod":
            raise TypeError("cumprod not supported for Timedelta.")

        else:
            return super()._accumulate(name, skipna=skipna, **kwargs)

    # ----------------------------------------------------------------
    # Rendering Methods

    def _formatter(self, boxed: bool = False):
        from pandas.io.formats.format import get_format_timedelta64

        return get_format_timedelta64(self, box=True)

    def _format_native_types(
        self, *, na_rep: str | float = "NaT", date_format=None, **kwargs
    ) -> npt.NDArray[np.object_]:
        from pandas.io.formats.format import get_format_timedelta64

        # Relies on TimeDelta._repr_base
        formatter = get_format_timedelta64(self, na_rep)
        # equiv: np.array([formatter(x) for x in self._ndarray])
        #  but independent of dimension
        return np.frompyfunc(formatter, 1, 1)(self._ndarray)

    # ----------------------------------------------------------------
    # Arithmetic Methods

    def _add_offset(self, other):
        assert not isinstance(other, Tick)
        raise TypeError(
            f"cannot add the type {type(other).__name__} to a {type(self).__name__}"
        )

    @unpack_zerodim_and_defer("__mul__")
    def __mul__(self, other) -> Self:
        if is_scalar(other):
            # numpy will accept float and int, raise TypeError for others
            result = self._ndarray * other
            if result.dtype.kind != "m":
                # numpy >= 2.1 may not raise a TypeError
                # and seems to dispatch to others.__rmul__?
                raise TypeError(f"Cannot multiply with {type(other).__name__}")
            freq = None
            if self.freq is not None and not isna(other):
                freq = self.freq * other
                if freq.n == 0:
                    # GH#51575 Better to have no freq than an incorrect one
                    freq = None
            return type(self)._simple_new(result, dtype=result.dtype, freq=freq)

        if not hasattr(other, "dtype"):
            # list, tuple
            other = np.array(other)
        if len(other) != len(self) and not lib.is_np_dtype(other.dtype, "m"):
            # Exclude timedelta64 here so we correctly raise TypeError
            #  for that instead of ValueError
            raise ValueError("Cannot multiply with unequal lengths")

        if is_object_dtype(other.dtype):
            # this multiplication will succeed only if all elements of other
            #  are int or float scalars, so we will end up with
            #  timedelta64[ns]-dtyped result
            arr = self._ndarray
            result = [arr[n] * other[n] for n in range(len(self))]
            result = np.array(result)
            return type(self)._simple_new(result, dtype=result.dtype)

        # numpy will accept float or int dtype, raise TypeError for others
        result = self._ndarray * other
        if result.dtype.kind != "m":
            # numpy >= 2.1 may not raise a TypeError
            # and seems to dispatch to others.__rmul__?
            raise TypeError(f"Cannot multiply with {type(other).__name__}")
        return type(self)._simple_new(result, dtype=result.dtype)

    __rmul__ = __mul__

    def _scalar_divlike_op(self, other, op):
        """
        Shared logic for __truediv__, __rtruediv__, __floordiv__, __rfloordiv__
        with scalar 'other'.
        """
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
            # mypy assumes that __new__ returns an instance of the class
            # github.com/python/mypy/issues/1020
            if cast("Timedelta | NaTType", other) is NaT:
                # specifically timedelta64-NaT
                res = np.empty(self.shape, dtype=np.float64)
                res.fill(np.nan)
                return res

            # otherwise, dispatch to Timedelta implementation
            return op(self._ndarray, other)

        else:
            # caller is responsible for checking lib.is_scalar(other)
            # assume other is numeric, otherwise numpy will raise

            if op in [roperator.rtruediv, roperator.rfloordiv]:
                raise TypeError(
                    f"Cannot divide {type(other).__name__} by {type(self).__name__}"
                )

            result = op(self._ndarray, other)
            freq = None

            if self.freq is not None:
                # Note: freq gets division, not floor-division, even if op
                #  is floordiv.
                freq = self.freq / other
                if freq.nanos == 0 and self.freq.nanos != 0:
                    # e.g. if self.freq is Nano(1) then dividing by 2
                    #  rounds down to zero
                    freq = None

            return type(self)._simple_new(result, dtype=result.dtype, freq=freq)

    def _cast_divlike_op(self, other):
        if not hasattr(other, "dtype"):
            # e.g. list, tuple
            other = np.array(other)

        if len(other) != len(self):
            raise ValueError("Cannot divide vectors with unequal lengths")
        return other

    def _vector_divlike_op(self, other, op) -> np.ndarray | Self:
        """
        Shared logic for __truediv__, __floordiv__, and their reversed versions
        with timedelta64-dtype ndarray other.
        """
        # Let numpy handle it
        result = op(self._ndarray, np.asarray(other))

        if (is_integer_dtype(other.dtype) or is_float_dtype(other.dtype)) and op in [
            operator.truediv,
            operator.floordiv,
        ]:
            return type(self)._simple_new(result, dtype=result.dtype)

        if op in [operator.floordiv, roperator.rfloordiv]:
            mask = self.isna() | isna(other)
            if mask.any():
                result = result.astype(np.float64)
                np.putmask(result, mask, np.nan)

        return result

    @unpack_zerodim_and_defer("__truediv__")
    def __truediv__(self, other):
        # timedelta / X is well-defined for timedelta-like or numeric X
        op = operator.truediv
        if is_scalar(other):
            return self._scalar_divlike_op(other, op)

        other = self._cast_divlike_op(other)
        if (
            lib.is_np_dtype(other.dtype, "m")
            or is_integer_dtype(other.dtype)
            or is_float_dtype(other.dtype)
        ):
            return self._vector_divlike_op(other, op)

        if is_object_dtype(other.dtype):
            other = np.asarray(other)
            if self.ndim > 1:
                res_cols = [left / right for left, right in zip(self, other)]
                res_cols2 = [x.reshape(1, -1) for x in res_cols]
                result = np.concatenate(res_cols2, axis=0)
            else:
                result = truediv_object_array(self._ndarray, other)

            return result

        else:
            return NotImplemented

    @unpack_zerodim_and_defer("__rtruediv__")
    def __rtruediv__(self, other):
        # X / timedelta is defined only for timedelta-like X
        op = roperator.rtruediv
        if is_scalar(other):
            return self._scalar_divlike_op(other, op)

        other = self._cast_divlike_op(other)
        if lib.is_np_dtype(other.dtype, "m"):
            return self._vector_divlike_op(other, op)

        elif is_object_dtype(other.dtype):
            # Note: unlike in __truediv__, we do not _need_ to do type
            #  inference on the result.  It does not raise, a numeric array
            #  is returned.  GH#23829
            result_list = [other[n] / self[n] for n in range(len(self))]
            return np.array(result_list)

        else:
            return NotImplemented

    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other):
        op = operator.floordiv
        if is_scalar(other):
            return self._scalar_divlike_op(other, op)

        other = self._cast_divlike_op(other)
        if (
            lib.is_np_dtype(other.dtype, "m")
            or is_integer_dtype(other.dtype)
            or is_float_dtype(other.dtype)
        ):
            return self._vector_divlike_op(other, op)

        elif is_object_dtype(other.dtype):
            other = np.asarray(other)
            if self.ndim > 1:
                res_cols = [left // right for left, right in zip(self, other)]
                res_cols2 = [x.reshape(1, -1) for x in res_cols]
                result = np.concatenate(res_cols2, axis=0)
            else:
                result = floordiv_object_array(self._ndarray, other)

            assert result.dtype == object
            return result

        else:
            return NotImplemented

    @unpack_zerodim_and_defer("__rfloordiv__")
    def __rfloordiv__(self, other):
        op = roperator.rfloordiv
        if is_scalar(other):
            return self._scalar_divlike_op(other, op)

        other = self._cast_divlike_op(other)
        if lib.is_np_dtype(other.dtype, "m"):
            return self._vector_divlike_op(other, op)

        elif is_object_dtype(other.dtype):
            result_list = [other[n] // self[n] for n in range(len(self))]
            result = np.array(result_list)
            return result

        else:
            return NotImplemented

    @unpack_zerodim_and_defer("__mod__")
    def __mod__(self, other):
        # Note: This is a naive implementation, can likely be optimized
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
        return self - (self // other) * other

    @unpack_zerodim_and_defer("__rmod__")
    def __rmod__(self, other):
        # Note: This is a naive implementation, can likely be optimized
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
        return other - (other // self) * self

    @unpack_zerodim_and_defer("__divmod__")
    def __divmod__(self, other):
        # Note: This is a naive implementation, can likely be optimized
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)

        res1 = self // other
        res2 = self - res1 * other
        return res1, res2

    @unpack_zerodim_and_defer("__rdivmod__")
    def __rdivmod__(self, other):
        # Note: This is a naive implementation, can likely be optimized
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)

        res1 = other // self
        res2 = other - res1 * self
        return res1, res2

    def __neg__(self) -> TimedeltaArray:
        freq = None
        if self.freq is not None:
            freq = -self.freq
        return type(self)._simple_new(-self._ndarray, dtype=self.dtype, freq=freq)

    def __pos__(self) -> TimedeltaArray:
        return type(self)._simple_new(
            self._ndarray.copy(), dtype=self.dtype, freq=self.freq
        )

    def __abs__(self) -> TimedeltaArray:
        # Note: freq is not preserved
        return type(self)._simple_new(np.abs(self._ndarray), dtype=self.dtype)

    # ----------------------------------------------------------------
    # Conversion Methods - Vectorized analogues of Timedelta methods

    def total_seconds(self) -> npt.NDArray[np.float64]:
        """
        Return total duration of each element expressed in seconds.

        This method is available directly on TimedeltaArray, TimedeltaIndex
        and on Series containing timedelta values under the ``.dt`` namespace.

        Returns
        -------
        ndarray, Index or Series
            When the calling object is a TimedeltaArray, the return type
            is ndarray.  When the calling object is a TimedeltaIndex,
            the return type is an Index with a float64 dtype. When the calling object
            is a Series, the return type is Series of type `float64` whose
            index is the same as the original.

        See Also
        --------
        datetime.timedelta.total_seconds : Standard library version
            of this method.
        TimedeltaIndex.components : Return a DataFrame with components of
            each Timedelta.

        Examples
        --------
        **Series**

        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit='d'))
        >>> s
        0   0 days
        1   1 days
        2   2 days
        3   3 days
        4   4 days
        dtype: timedelta64[ns]

        >>> s.dt.total_seconds()
        0         0.0
        1     86400.0
        2    172800.0
        3    259200.0
        4    345600.0
        dtype: float64

        **TimedeltaIndex**

        >>> idx = pd.to_timedelta(np.arange(5), unit='d')
        >>> idx
        TimedeltaIndex(['0 days', '1 days', '2 days', '3 days', '4 days'],
                       dtype='timedelta64[ns]', freq=None)

        >>> idx.total_seconds()
        Index([0.0, 86400.0, 172800.0, 259200.0, 345600.0], dtype='float64')
        """
        pps = periods_per_second(self._creso)
        return self._maybe_mask_results(self.asi8 / pps, fill_value=None)

    def to_pytimedelta(self) -> npt.NDArray[np.object_]:
        """
        Return an ndarray of datetime.timedelta objects.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='D')
        >>> tdelta_idx
        TimedeltaIndex(['1 days', '2 days', '3 days'],
                        dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.to_pytimedelta()
        array([datetime.timedelta(days=1), datetime.timedelta(days=2),
               datetime.timedelta(days=3)], dtype=object)
        """
        return ints_to_pytimedelta(self._ndarray)

    days_docstring = textwrap.dedent(
        """Number of days for each element.

    Examples
    --------
    For Series:

    >>> ser = pd.Series(pd.to_timedelta([1, 2, 3], unit='d'))
    >>> ser
    0   1 days
    1   2 days
    2   3 days
    dtype: timedelta64[ns]
    >>> ser.dt.days
    0    1
    1    2
    2    3
    dtype: int64

    For TimedeltaIndex:

    >>> tdelta_idx = pd.to_timedelta(["0 days", "10 days", "20 days"])
    >>> tdelta_idx
    TimedeltaIndex(['0 days', '10 days', '20 days'],
                    dtype='timedelta64[ns]', freq=None)
    >>> tdelta_idx.days
    Index([0, 10, 20], dtype='int64')"""
    )
    days = _field_accessor("days", "days", days_docstring)

    seconds_docstring = textwrap.dedent(
        """Number of seconds (>= 0 and less than 1 day) for each element.

    Examples
    --------
    For Series:

    >>> ser = pd.Series(pd.to_timedelta([1, 2, 3], unit='s'))
    >>> ser
    0   0 days 00:00:01
    1   0 days 00:00:02
    2   0 days 00:00:03
    dtype: timedelta64[ns]
    >>> ser.dt.seconds
    0    1
    1    2
    2    3
    dtype: int32

    For TimedeltaIndex:

    >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='s')
    >>> tdelta_idx
    TimedeltaIndex(['0 days 00:00:01', '0 days 00:00:02', '0 days 00:00:03'],
                   dtype='timedelta64[ns]', freq=None)
    >>> tdelta_idx.seconds
    Index([1, 2, 3], dtype='int32')"""
    )
    seconds = _field_accessor(
        "seconds",
        "seconds",
        seconds_docstring,
    )

    microseconds_docstring = textwrap.dedent(
        """Number of microseconds (>= 0 and less than 1 second) for each element.

    Examples
    --------
    For Series:

    >>> ser = pd.Series(pd.to_timedelta([1, 2, 3], unit='us'))
    >>> ser
    0   0 days 00:00:00.000001
    1   0 days 00:00:00.000002
    2   0 days 00:00:00.000003
    dtype: timedelta64[ns]
    >>> ser.dt.microseconds
    0    1
    1    2
    2    3
    dtype: int32

    For TimedeltaIndex:

    >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='us')
    >>> tdelta_idx
    TimedeltaIndex(['0 days 00:00:00.000001', '0 days 00:00:00.000002',
                    '0 days 00:00:00.000003'],
                   dtype='timedelta64[ns]', freq=None)
    >>> tdelta_idx.microseconds
    Index([1, 2, 3], dtype='int32')"""
    )
    microseconds = _field_accessor(
        "microseconds",
        "microseconds",
        microseconds_docstring,
    )

    nanoseconds_docstring = textwrap.dedent(
        """Number of nanoseconds (>= 0 and less than 1 microsecond) for each element.

    Examples
    --------
    For Series:

    >>> ser = pd.Series(pd.to_timedelta([1, 2, 3], unit='ns'))
    >>> ser
    0   0 days 00:00:00.000000001
    1   0 days 00:00:00.000000002
    2   0 days 00:00:00.000000003
    dtype: timedelta64[ns]
    >>> ser.dt.nanoseconds
    0    1
    1    2
    2    3
    dtype: int32

    For TimedeltaIndex:

    >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='ns')
    >>> tdelta_idx
    TimedeltaIndex(['0 days 00:00:00.000000001', '0 days 00:00:00.000000002',
                    '0 days 00:00:00.000000003'],
                   dtype='timedelta64[ns]', freq=None)
    >>> tdelta_idx.nanoseconds
    Index([1, 2, 3], dtype='int32')"""
    )
    nanoseconds = _field_accessor(
        "nanoseconds",
        "nanoseconds",
        nanoseconds_docstring,
    )

    @property
    def components(self) -> DataFrame:
        """
        Return a DataFrame of the individual resolution components of the Timedeltas.

        The components (days, hours, minutes seconds, milliseconds, microseconds,
        nanoseconds) are returned as columns in a DataFrame.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> tdelta_idx = pd.to_timedelta(['1 day 3 min 2 us 42 ns'])
        >>> tdelta_idx
        TimedeltaIndex(['1 days 00:03:00.000002042'],
                       dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.components
           days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds
        0     1      0        3        0             0             2           42
        """
        from pandas import DataFrame

        columns = [
            "days",
            "hours",
            "minutes",
            "seconds",
            "milliseconds",
            "microseconds",
            "nanoseconds",
        ]
        hasnans = self._hasna
        if hasnans:

            def f(x):
                if isna(x):
                    return [np.nan] * len(columns)
                return x.components

        else:

            def f(x):
                return x.components

        result = DataFrame([f(x) for x in self], columns=columns)
        if not hasnans:
            result = result.astype("int64")
        return result


# ---------------------------------------------------------------------
# Constructor Helpers


def sequence_to_td64ns(
    data,
    copy: bool = False,
    unit=None,
    errors: DateTimeErrorChoices = "raise",
) -> tuple[np.ndarray, Tick | None]:
    """
    Parameters
    ----------
    data : list-like
    copy : bool, default False
    unit : str, optional
        The timedelta unit to treat integers as multiples of. For numeric
        data this defaults to ``'ns'``.
        Must be un-specified if the data contains a str and ``errors=="raise"``.
    errors : {"raise", "coerce", "ignore"}, default "raise"
        How to handle elements that cannot be converted to timedelta64[ns].
        See ``pandas.to_timedelta`` for details.

    Returns
    -------
    converted : numpy.ndarray
        The sequence converted to a numpy array with dtype ``timedelta64[ns]``.
    inferred_freq : Tick or None
        The inferred frequency of the sequence.

    Raises
    ------
    ValueError : Data cannot be converted to timedelta64[ns].

    Notes
    -----
    Unlike `pandas.to_timedelta`, if setting ``errors=ignore`` will not cause
    errors to be ignored; they are caught and subsequently ignored at a
    higher level.
    """
    assert unit not in ["Y", "y", "M"]  # caller is responsible for checking

    inferred_freq = None
    if unit is not None:
        unit = parse_timedelta_unit(unit)

    data, copy = dtl.ensure_arraylike_for_datetimelike(
        data, copy, cls_name="TimedeltaArray"
    )

    if isinstance(data, TimedeltaArray):
        inferred_freq = data.freq

    # Convert whatever we have into timedelta64[ns] dtype
    if data.dtype == object or is_string_dtype(data.dtype):
        # no need to make a copy, need to convert if string-dtyped
        data = _objects_to_td64ns(data, unit=unit, errors=errors)
        copy = False

    elif is_integer_dtype(data.dtype):
        # treat as multiples of the given unit
        data, copy_made = _ints_to_td64ns(data, unit=unit)
        copy = copy and not copy_made

    elif is_float_dtype(data.dtype):
        # cast the unit, multiply base/frac separately
        # to avoid precision issues from float -> int
        if isinstance(data.dtype, ExtensionDtype):
            mask = data._mask
            data = data._data
        else:
            mask = np.isnan(data)

        data = cast_from_unit_vectorized(data, unit or "ns")
        data[mask] = iNaT
        data = data.view("m8[ns]")
        copy = False

    elif lib.is_np_dtype(data.dtype, "m"):
        if not is_supported_dtype(data.dtype):
            # cast to closest supported unit, i.e. s or ns
            new_dtype = get_supported_dtype(data.dtype)
            data = astype_overflowsafe(data, dtype=new_dtype, copy=False)
            copy = False

    else:
        # This includes datetime64-dtype, see GH#23539, GH#29794
        raise TypeError(f"dtype {data.dtype} cannot be converted to timedelta64[ns]")

    if not copy:
        data = np.asarray(data)
    else:
        data = np.array(data, copy=copy)

    assert data.dtype.kind == "m"
    assert data.dtype != "m8"  # i.e. not unit-less

    return data, inferred_freq


def _ints_to_td64ns(data, unit: str = "ns"):
    """
    Convert an ndarray with integer-dtype to timedelta64[ns] dtype, treating
    the integers as multiples of the given timedelta unit.

    Parameters
    ----------
    data : numpy.ndarray with integer-dtype
    unit : str, default "ns"
        The timedelta unit to treat integers as multiples of.

    Returns
    -------
    numpy.ndarray : timedelta64[ns] array converted from data
    bool : whether a copy was made
    """
    copy_made = False
    unit = unit if unit is not None else "ns"

    if data.dtype != np.int64:
        # converting to int64 makes a copy, so we can avoid
        # re-copying later
        data = data.astype(np.int64)
        copy_made = True

    if unit != "ns":
        dtype_str = f"timedelta64[{unit}]"
        data = data.view(dtype_str)

        data = astype_overflowsafe(data, dtype=TD64NS_DTYPE)

        # the astype conversion makes a copy, so we can avoid re-copying later
        copy_made = True

    else:
        data = data.view("timedelta64[ns]")

    return data, copy_made


def _objects_to_td64ns(data, unit=None, errors: DateTimeErrorChoices = "raise"):
    """
    Convert a object-dtyped or string-dtyped array into an
    timedelta64[ns]-dtyped array.

    Parameters
    ----------
    data : ndarray or Index
    unit : str, default "ns"
        The timedelta unit to treat integers as multiples of.
        Must not be specified if the data contains a str.
    errors : {"raise", "coerce", "ignore"}, default "raise"
        How to handle elements that cannot be converted to timedelta64[ns].
        See ``pandas.to_timedelta`` for details.

    Returns
    -------
    numpy.ndarray : timedelta64[ns] array converted from data

    Raises
    ------
    ValueError : Data cannot be converted to timedelta64[ns].

    Notes
    -----
    Unlike `pandas.to_timedelta`, if setting `errors=ignore` will not cause
    errors to be ignored; they are caught and subsequently ignored at a
    higher level.
    """
    # coerce Index to np.ndarray, converting string-dtype if necessary
    values = np.asarray(data, dtype=np.object_)

    result = array_to_timedelta64(values, unit=unit, errors=errors)
    return result.view("timedelta64[ns]")


def _validate_td64_dtype(dtype) -> DtypeObj:
    dtype = pandas_dtype(dtype)
    if dtype == np.dtype("m8"):
        # no precision disallowed GH#24806
        msg = (
            "Passing in 'timedelta' dtype with no precision is not allowed. "
            "Please pass in 'timedelta64[ns]' instead."
        )
        raise ValueError(msg)

    if not lib.is_np_dtype(dtype, "m"):
        raise ValueError(f"dtype '{dtype}' is invalid, should be np.timedelta64 dtype")
    elif not is_supported_dtype(dtype):
        raise ValueError("Supported timedelta64 resolutions are 's', 'ms', 'us', 'ns'")

    return dtype

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\hyper.py ===
"""Hypergeometric and Meijer G-functions"""
from collections import Counter

from sympy.core import S, Mod
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import DefinedFunction, Derivative, ArgumentIndexError

from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.numbers import I, pi, oo, zoo
from sympy.core.parameters import global_parameters
from sympy.core.relational import Ne
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy

from sympy.external.gmpy import lcm
from sympy.functions import (sqrt, exp, log, sin, cos, asin, atan,
        sinh, cosh, asinh, acosh, atanh, acoth)
from sympy.functions import factorial, RisingFactorial
from sympy.functions.elementary.complexes import Abs, re, unpolarify
from sympy.functions.elementary.exponential import exp_polar
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import (And, Or)
from sympy import ordered


class TupleArg(Tuple):

    # This method is only needed because hyper._eval_as_leading_term falls back
    # (via super()) on using Function._eval_as_leading_term, which in turn
    # calls as_leading_term on the args of the hyper. Ideally hyper should just
    # have an _eval_as_leading_term method that handles all cases and this
    # method should be removed because leading terms of tuples don't make
    # sense.
    def as_leading_term(self, *x, logx=None, cdir=0):
        return TupleArg(*[f.as_leading_term(*x, logx=logx, cdir=cdir) for f in self.args])

    def limit(self, x, xlim, dir='+'):
        """ Compute limit x->xlim.
        """
        from sympy.series.limits import limit
        return TupleArg(*[limit(f, x, xlim, dir) for f in self.args])


# TODO should __new__ accept **options?
# TODO should constructors should check if parameters are sensible?


def _prep_tuple(v):
    """
    Turn an iterable argument *v* into a tuple and unpolarify, since both
    hypergeometric and meijer g-functions are unbranched in their parameters.

    Examples
    ========

    >>> from sympy.functions.special.hyper import _prep_tuple
    >>> _prep_tuple([1, 2, 3])
    (1, 2, 3)
    >>> _prep_tuple((4, 5))
    (4, 5)
    >>> _prep_tuple((7, 8, 9))
    (7, 8, 9)

    """
    return TupleArg(*[unpolarify(x) for x in v])


class TupleParametersBase(DefinedFunction):
    """ Base class that takes care of differentiation, when some of
        the arguments are actually tuples. """
    # This is not deduced automatically since there are Tuples as arguments.
    is_commutative = True

    def _eval_derivative(self, s):
        try:
            res = 0
            if self.args[0].has(s) or self.args[1].has(s):
                for i, p in enumerate(self._diffargs):
                    m = self._diffargs[i].diff(s)
                    if m != 0:
                        res += self.fdiff((1, i))*m
            return res + self.fdiff(3)*self.args[2].diff(s)
        except (ArgumentIndexError, NotImplementedError):
            return Derivative(self, s)


class hyper(TupleParametersBase):
    r"""
    The generalized hypergeometric function is defined by a series where
    the ratios of successive terms are a rational function of the summation
    index. When convergent, it is continued analytically to the largest
    possible domain.

    Explanation
    ===========

    The hypergeometric function depends on two vectors of parameters, called
    the numerator parameters $a_p$, and the denominator parameters
    $b_q$. It also has an argument $z$. The series definition is

    .. math ::
        {}_pF_q\left(\begin{matrix} a_1, \cdots, a_p \\ b_1, \cdots, b_q \end{matrix}
                     \middle| z \right)
        = \sum_{n=0}^\infty \frac{(a_1)_n \cdots (a_p)_n}{(b_1)_n \cdots (b_q)_n}
                            \frac{z^n}{n!},

    where $(a)_n = (a)(a+1)\cdots(a+n-1)$ denotes the rising factorial.

    If one of the $b_q$ is a non-positive integer then the series is
    undefined unless one of the $a_p$ is a larger (i.e., smaller in
    magnitude) non-positive integer. If none of the $b_q$ is a
    non-positive integer and one of the $a_p$ is a non-positive
    integer, then the series reduces to a polynomial. To simplify the
    following discussion, we assume that none of the $a_p$ or
    $b_q$ is a non-positive integer. For more details, see the
    references.

    The series converges for all $z$ if $p \le q$, and thus
    defines an entire single-valued function in this case. If $p =
    q+1$ the series converges for $|z| < 1$, and can be continued
    analytically into a half-plane. If $p > q+1$ the series is
    divergent for all $z$.

    Please note the hypergeometric function constructor currently does *not*
    check if the parameters actually yield a well-defined function.

    Examples
    ========

    The parameters $a_p$ and $b_q$ can be passed as arbitrary
    iterables, for example:

    >>> from sympy import hyper
    >>> from sympy.abc import x, n, a
    >>> h = hyper((1, 2, 3), [3, 4], x); h
    hyper((1, 2), (4,), x)
    >>> hyper((3, 1, 2), [3, 4], x, evaluate=False)  # don't remove duplicates
    hyper((1, 2, 3), (3, 4), x)

    There is also pretty printing (it looks better using Unicode):

    >>> from sympy import pprint
    >>> pprint(h, use_unicode=False)
      _
     |_  /1, 2 |  \
     |   |     | x|
    2  1 \  4  |  /

    The parameters must always be iterables, even if they are vectors of
    length one or zero:

    >>> hyper((1, ), [], x)
    hyper((1,), (), x)

    But of course they may be variables (but if they depend on $x$ then you
    should not expect much implemented functionality):

    >>> hyper((n, a), (n**2,), x)
    hyper((a, n), (n**2,), x)

    The hypergeometric function generalizes many named special functions.
    The function ``hyperexpand()`` tries to express a hypergeometric function
    using named special functions. For example:

    >>> from sympy import hyperexpand
    >>> hyperexpand(hyper([], [], x))
    exp(x)

    You can also use ``expand_func()``:

    >>> from sympy import expand_func
    >>> expand_func(x*hyper([1, 1], [2], -x))
    log(x + 1)

    More examples:

    >>> from sympy import S
    >>> hyperexpand(hyper([], [S(1)/2], -x**2/4))
    cos(x)
    >>> hyperexpand(x*hyper([S(1)/2, S(1)/2], [S(3)/2], x**2))
    asin(x)

    We can also sometimes ``hyperexpand()`` parametric functions:

    >>> from sympy.abc import a
    >>> hyperexpand(hyper([-a], [], x))
    (1 - x)**a

    See Also
    ========

    sympy.simplify.hyperexpand
    gamma
    meijerg

    References
    ==========

    .. [1] Luke, Y. L. (1969), The Special Functions and Their Approximations,
           Volume 1
    .. [2] https://en.wikipedia.org/wiki/Generalized_hypergeometric_function

    """


    def __new__(cls, ap, bq, z, **kwargs):
        # TODO should we check convergence conditions?
        if kwargs.pop('evaluate', global_parameters.evaluate):
            ca = Counter(Tuple(*ap))
            cb = Counter(Tuple(*bq))
            common = ca & cb
            arg = ap, bq = [], []
            for i, c in enumerate((ca, cb)):
                c -= common
                for k in ordered(c):
                    arg[i].extend([k]*c[k])
        else:
            ap = list(ordered(ap))
            bq = list(ordered(bq))
        return super().__new__(cls, _prep_tuple(ap), _prep_tuple(bq), z, **kwargs)

    @classmethod
    def eval(cls, ap, bq, z):
        if len(ap) <= len(bq) or (len(ap) == len(bq) + 1 and (Abs(z) <= 1) == True):
            nz = unpolarify(z)
            if z != nz:
                return hyper(ap, bq, nz)

    def fdiff(self, argindex=3):
        if argindex != 3:
            raise ArgumentIndexError(self, argindex)
        nap = Tuple(*[a + 1 for a in self.ap])
        nbq = Tuple(*[b + 1 for b in self.bq])
        fac = Mul(*self.ap)/Mul(*self.bq)
        return fac*hyper(nap, nbq, self.argument)

    def _eval_expand_func(self, **hints):
        from sympy.functions.special.gamma_functions import gamma
        from sympy.simplify.hyperexpand import hyperexpand
        if len(self.ap) == 2 and len(self.bq) == 1 and self.argument == 1:
            a, b = self.ap
            c = self.bq[0]
            return gamma(c)*gamma(c - a - b)/gamma(c - a)/gamma(c - b)
        return hyperexpand(self)

    def _eval_rewrite_as_Sum(self, ap, bq, z, **kwargs):
        from sympy.concrete.summations import Sum
        n = Dummy("n", integer=True)
        rfap = [RisingFactorial(a, n) for a in ap]
        rfbq = [RisingFactorial(b, n) for b in bq]
        coeff = Mul(*rfap) / Mul(*rfbq)
        return Piecewise((Sum(coeff * z**n / factorial(n), (n, 0, oo)),
                         self.convergence_statement), (self, True))

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[2]
        x0 = arg.subs(x, 0)
        if x0 is S.NaN:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')

        if x0 is S.Zero:
            return S.One
        return super()._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):

        from sympy.series.order import Order

        arg = self.args[2]
        x0 = arg.limit(x, 0)
        ap = self.args[0]
        bq = self.args[1]

        if not (arg == x and x0 == 0):
            # It would be better to do something with arg.nseries here, rather
            # than falling back on Function._eval_nseries. The code below
            # though is not sufficient if arg is something like x/(x+1).
            from sympy.simplify.hyperexpand import hyperexpand
            return hyperexpand(super()._eval_nseries(x, n, logx))

        terms = []

        for i in range(n):
            num = Mul(*[RisingFactorial(a, i) for a in ap])
            den = Mul(*[RisingFactorial(b, i) for b in bq])
            terms.append(((num/den) * (arg**i)) / factorial(i))

        return (Add(*terms) + Order(x**n,x))

    @property
    def argument(self):
        """ Argument of the hypergeometric function. """
        return self.args[2]

    @property
    def ap(self):
        """ Numerator parameters of the hypergeometric function. """
        return Tuple(*self.args[0])

    @property
    def bq(self):
        """ Denominator parameters of the hypergeometric function. """
        return Tuple(*self.args[1])

    @property
    def _diffargs(self):
        return self.ap + self.bq

    @property
    def eta(self):
        """ A quantity related to the convergence of the series. """
        return sum(self.ap) - sum(self.bq)

    @property
    def radius_of_convergence(self):
        """
        Compute the radius of convergence of the defining series.

        Explanation
        ===========

        Note that even if this is not ``oo``, the function may still be
        evaluated outside of the radius of convergence by analytic
        continuation. But if this is zero, then the function is not actually
        defined anywhere else.

        Examples
        ========

        >>> from sympy import hyper
        >>> from sympy.abc import z
        >>> hyper((1, 2), [3], z).radius_of_convergence
        1
        >>> hyper((1, 2, 3), [4], z).radius_of_convergence
        0
        >>> hyper((1, 2), (3, 4), z).radius_of_convergence
        oo

        """
        if any(a.is_integer and (a <= 0) == True for a in self.ap + self.bq):
            aints = [a for a in self.ap if a.is_Integer and (a <= 0) == True]
            bints = [a for a in self.bq if a.is_Integer and (a <= 0) == True]
            if len(aints) < len(bints):
                return S.Zero
            popped = False
            for b in bints:
                cancelled = False
                while aints:
                    a = aints.pop()
                    if a >= b:
                        cancelled = True
                        break
                    popped = True
                if not cancelled:
                    return S.Zero
            if aints or popped:
                # There are still non-positive numerator parameters.
                # This is a polynomial.
                return oo
        if len(self.ap) == len(self.bq) + 1:
            return S.One
        elif len(self.ap) <= len(self.bq):
            return oo
        else:
            return S.Zero

    @property
    def convergence_statement(self):
        """ Return a condition on z under which the series converges. """
        R = self.radius_of_convergence
        if R == 0:
            return False
        if R == oo:
            return True
        # The special functions and their approximations, page 44
        e = self.eta
        z = self.argument
        c1 = And(re(e) < 0, abs(z) <= 1)
        c2 = And(0 <= re(e), re(e) < 1, abs(z) <= 1, Ne(z, 1))
        c3 = And(re(e) >= 1, abs(z) < 1)
        return Or(c1, c2, c3)

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.hyperexpand import hyperexpand
        return hyperexpand(self)


class meijerg(TupleParametersBase):
    r"""
    The Meijer G-function is defined by a Mellin-Barnes type integral that
    resembles an inverse Mellin transform. It generalizes the hypergeometric
    functions.

    Explanation
    ===========

    The Meijer G-function depends on four sets of parameters. There are
    "*numerator parameters*"
    $a_1, \ldots, a_n$ and $a_{n+1}, \ldots, a_p$, and there are
    "*denominator parameters*"
    $b_1, \ldots, b_m$ and $b_{m+1}, \ldots, b_q$.
    Confusingly, it is traditionally denoted as follows (note the position
    of $m$, $n$, $p$, $q$, and how they relate to the lengths of the four
    parameter vectors):

    .. math ::
        G_{p,q}^{m,n} \left(\begin{matrix}a_1, \cdots, a_n & a_{n+1}, \cdots, a_p \\
                                        b_1, \cdots, b_m & b_{m+1}, \cdots, b_q
                          \end{matrix} \middle| z \right).

    However, in SymPy the four parameter vectors are always available
    separately (see examples), so that there is no need to keep track of the
    decorating sub- and super-scripts on the G symbol.

    The G function is defined as the following integral:

    .. math ::
         \frac{1}{2 \pi i} \int_L \frac{\prod_{j=1}^m \Gamma(b_j - s)
         \prod_{j=1}^n \Gamma(1 - a_j + s)}{\prod_{j=m+1}^q \Gamma(1- b_j +s)
         \prod_{j=n+1}^p \Gamma(a_j - s)} z^s \mathrm{d}s,

    where $\Gamma(z)$ is the gamma function. There are three possible
    contours which we will not describe in detail here (see the references).
    If the integral converges along more than one of them, the definitions
    agree. The contours all separate the poles of $\Gamma(1-a_j+s)$
    from the poles of $\Gamma(b_k-s)$, so in particular the G function
    is undefined if $a_j - b_k \in \mathbb{Z}_{>0}$ for some
    $j \le n$ and $k \le m$.

    The conditions under which one of the contours yields a convergent integral
    are complicated and we do not state them here, see the references.

    Please note currently the Meijer G-function constructor does *not* check any
    convergence conditions.

    Examples
    ========

    You can pass the parameters either as four separate vectors:

    >>> from sympy import meijerg, Tuple, pprint
    >>> from sympy.abc import x, a
    >>> pprint(meijerg((1, 2), (a, 4), (5,), [], x), use_unicode=False)
     __1, 2 /1, 2  4, a |  \
    /__     |           | x|
    \_|4, 1 \ 5         |  /

    Or as two nested vectors:

    >>> pprint(meijerg([(1, 2), (3, 4)], ([5], Tuple()), x), use_unicode=False)
     __1, 2 /1, 2  3, 4 |  \
    /__     |           | x|
    \_|4, 1 \ 5         |  /

    As with the hypergeometric function, the parameters may be passed as
    arbitrary iterables. Vectors of length zero and one also have to be
    passed as iterables. The parameters need not be constants, but if they
    depend on the argument then not much implemented functionality should be
    expected.

    All the subvectors of parameters are available:

    >>> from sympy import pprint
    >>> g = meijerg([1], [2], [3], [4], x)
    >>> pprint(g, use_unicode=False)
     __1, 1 /1  2 |  \
    /__     |     | x|
    \_|2, 2 \3  4 |  /
    >>> g.an
    (1,)
    >>> g.ap
    (1, 2)
    >>> g.aother
    (2,)
    >>> g.bm
    (3,)
    >>> g.bq
    (3, 4)
    >>> g.bother
    (4,)

    The Meijer G-function generalizes the hypergeometric functions.
    In some cases it can be expressed in terms of hypergeometric functions,
    using Slater's theorem. For example:

    >>> from sympy import hyperexpand
    >>> from sympy.abc import a, b, c
    >>> hyperexpand(meijerg([a], [], [c], [b], x), allow_hyper=True)
    x**c*gamma(-a + c + 1)*hyper((-a + c + 1,),
                                 (-b + c + 1,), -x)/gamma(-b + c + 1)

    Thus the Meijer G-function also subsumes many named functions as special
    cases. You can use ``expand_func()`` or ``hyperexpand()`` to (try to)
    rewrite a Meijer G-function in terms of named special functions. For
    example:

    >>> from sympy import expand_func, S
    >>> expand_func(meijerg([[],[]], [[0],[]], -x))
    exp(x)
    >>> hyperexpand(meijerg([[],[]], [[S(1)/2],[0]], (x/2)**2))
    sin(x)/sqrt(pi)

    See Also
    ========

    hyper
    sympy.simplify.hyperexpand

    References
    ==========

    .. [1] Luke, Y. L. (1969), The Special Functions and Their Approximations,
           Volume 1
    .. [2] https://en.wikipedia.org/wiki/Meijer_G-function

    """


    def __new__(cls, *args, **kwargs):
        if len(args) == 5:
            args = [(args[0], args[1]), (args[2], args[3]), args[4]]
        if len(args) != 3:
            raise TypeError("args must be either as, as', bs, bs', z or "
                            "as, bs, z")

        def tr(p):
            if len(p) != 2:
                raise TypeError("wrong argument")
            p = [list(ordered(i)) for i in p]
            return TupleArg(_prep_tuple(p[0]), _prep_tuple(p[1]))

        arg0, arg1 = tr(args[0]), tr(args[1])
        if Tuple(arg0, arg1).has(oo, zoo, -oo):
            raise ValueError("G-function parameters must be finite")
        if any((a - b).is_Integer and a - b > 0
               for a in arg0[0] for b in arg1[0]):
            raise ValueError("no parameter a1, ..., an may differ from "
                         "any b1, ..., bm by a positive integer")

        # TODO should we check convergence conditions?
        return super().__new__(cls, arg0, arg1, args[2], **kwargs)

    def fdiff(self, argindex=3):
        if argindex != 3:
            return self._diff_wrt_parameter(argindex[1])
        if len(self.an) >= 1:
            a = list(self.an)
            a[0] -= 1
            G = meijerg(a, self.aother, self.bm, self.bother, self.argument)
            return 1/self.argument * ((self.an[0] - 1)*self + G)
        elif len(self.bm) >= 1:
            b = list(self.bm)
            b[0] += 1
            G = meijerg(self.an, self.aother, b, self.bother, self.argument)
            return 1/self.argument * (self.bm[0]*self - G)
        else:
            return S.Zero

    def _diff_wrt_parameter(self, idx):
        # Differentiation wrt a parameter can only be done in very special
        # cases. In particular, if we want to differentiate with respect to
        # `a`, all other gamma factors have to reduce to rational functions.
        #
        # Let MT denote mellin transform. Suppose T(-s) is the gamma factor
        # appearing in the definition of G. Then
        #
        #   MT(log(z)G(z)) = d/ds T(s) = d/da T(s) + ...
        #
        # Thus d/da G(z) = log(z)G(z) - ...
        # The ... can be evaluated as a G function under the above conditions,
        # the formula being most easily derived by using
        #
        # d  Gamma(s + n)    Gamma(s + n) / 1    1                1     \
        # -- ------------ =  ------------ | - + ----  + ... + --------- |
        # ds Gamma(s)        Gamma(s)     \ s   s + 1         s + n - 1 /
        #
        # which follows from the difference equation of the digamma function.
        # (There is a similar equation for -n instead of +n).

        # We first figure out how to pair the parameters.
        an = list(self.an)
        ap = list(self.aother)
        bm = list(self.bm)
        bq = list(self.bother)
        if idx < len(an):
            an.pop(idx)
        else:
            idx -= len(an)
            if idx < len(ap):
                ap.pop(idx)
            else:
                idx -= len(ap)
                if idx < len(bm):
                    bm.pop(idx)
                else:
                    bq.pop(idx - len(bm))
        pairs1 = []
        pairs2 = []
        for l1, l2, pairs in [(an, bq, pairs1), (ap, bm, pairs2)]:
            while l1:
                x = l1.pop()
                found = None
                for i, y in enumerate(l2):
                    if not Mod((x - y).simplify(), 1):
                        found = i
                        break
                if found is None:
                    raise NotImplementedError('Derivative not expressible '
                                              'as G-function?')
                y = l2[i]
                l2.pop(i)
                pairs.append((x, y))

        # Now build the result.
        res = log(self.argument)*self

        for a, b in pairs1:
            sign = 1
            n = a - b
            base = b
            if n < 0:
                sign = -1
                n = b - a
                base = a
            for k in range(n):
                res -= sign*meijerg(self.an + (base + k + 1,), self.aother,
                                    self.bm, self.bother + (base + k + 0,),
                                    self.argument)

        for a, b in pairs2:
            sign = 1
            n = b - a
            base = a
            if n < 0:
                sign = -1
                n = a - b
                base = b
            for k in range(n):
                res -= sign*meijerg(self.an, self.aother + (base + k + 1,),
                                    self.bm + (base + k + 0,), self.bother,
                                    self.argument)

        return res

    def get_period(self):
        """
        Return a number $P$ such that $G(x*exp(I*P)) == G(x)$.

        Examples
        ========

        >>> from sympy import meijerg, pi, S
        >>> from sympy.abc import z

        >>> meijerg([1], [], [], [], z).get_period()
        2*pi
        >>> meijerg([pi], [], [], [], z).get_period()
        oo
        >>> meijerg([1, 2], [], [], [], z).get_period()
        oo
        >>> meijerg([1,1], [2], [1, S(1)/2, S(1)/3], [1], z).get_period()
        12*pi

        """
        # This follows from slater's theorem.
        def compute(l):
            # first check that no two differ by an integer
            for i, b in enumerate(l):
                if not b.is_Rational:
                    return oo
                for j in range(i + 1, len(l)):
                    if not Mod((b - l[j]).simplify(), 1):
                        return oo
            return lcm(*(x.q for x in l))
        beta = compute(self.bm)
        alpha = compute(self.an)
        p, q = len(self.ap), len(self.bq)
        if p == q:
            if oo in (alpha, beta):
                return oo
            return 2*pi*lcm(alpha, beta)
        elif p < q:
            return 2*pi*beta
        else:
            return 2*pi*alpha

    def _eval_expand_func(self, **hints):
        from sympy.simplify.hyperexpand import hyperexpand
        return hyperexpand(self)

    def _eval_evalf(self, prec):
        # The default code is insufficient for polar arguments.
        # mpmath provides an optional argument "r", which evaluates
        # G(z**(1/r)). I am not sure what its intended use is, but we hijack it
        # here in the following way: to evaluate at a number z of |argument|
        # less than (say) n*pi, we put r=1/n, compute z' = root(z, n)
        # (carefully so as not to loose the branch information), and evaluate
        # G(z'**(1/r)) = G(z'**n) = G(z).
        import mpmath
        znum = self.argument._eval_evalf(prec)
        if znum.has(exp_polar):
            znum, branch = znum.as_coeff_mul(exp_polar)
            if len(branch) != 1:
                return
            branch = branch[0].args[0]/I
        else:
            branch = S.Zero
        n = ceiling(abs(branch/pi)) + 1
        znum = znum**(S.One/n)*exp(I*branch / n)

        # Convert all args to mpf or mpc
        try:
            [z, r, ap, bq] = [arg._to_mpmath(prec)
                    for arg in [znum, 1/n, self.args[0], self.args[1]]]
        except ValueError:
            return

        with mpmath.workprec(prec):
            v = mpmath.meijerg(ap, bq, z, r)

        return Expr._from_mpmath(v, prec)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.simplify.hyperexpand import hyperexpand
        return hyperexpand(self).as_leading_term(x, logx=logx, cdir=cdir)

    def integrand(self, s):
        """ Get the defining integrand D(s). """
        from sympy.functions.special.gamma_functions import gamma
        return self.argument**s \
            * Mul(*(gamma(b - s) for b in self.bm)) \
            * Mul(*(gamma(1 - a + s) for a in self.an)) \
            / Mul(*(gamma(1 - b + s) for b in self.bother)) \
            / Mul(*(gamma(a - s) for a in self.aother))

    @property
    def argument(self):
        """ Argument of the Meijer G-function. """
        return self.args[2]

    @property
    def an(self):
        """ First set of numerator parameters. """
        return Tuple(*self.args[0][0])

    @property
    def ap(self):
        """ Combined numerator parameters. """
        return Tuple(*(self.args[0][0] + self.args[0][1]))

    @property
    def aother(self):
        """ Second set of numerator parameters. """
        return Tuple(*self.args[0][1])

    @property
    def bm(self):
        """ First set of denominator parameters. """
        return Tuple(*self.args[1][0])

    @property
    def bq(self):
        """ Combined denominator parameters. """
        return Tuple(*(self.args[1][0] + self.args[1][1]))

    @property
    def bother(self):
        """ Second set of denominator parameters. """
        return Tuple(*self.args[1][1])

    @property
    def _diffargs(self):
        return self.ap + self.bq

    @property
    def nu(self):
        """ A quantity related to the convergence region of the integral,
            c.f. references. """
        return sum(self.bq) - sum(self.ap)

    @property
    def delta(self):
        """ A quantity related to the convergence region of the integral,
            c.f. references. """
        return len(self.bm) + len(self.an) - S(len(self.ap) + len(self.bq))/2

    @property
    def is_number(self):
        """ Returns true if expression has numeric data only. """
        return not self.free_symbols


class HyperRep(DefinedFunction):
    """
    A base class for "hyper representation functions".

    This is used exclusively in ``hyperexpand()``, but fits more logically here.

    pFq is branched at 1 if p == q+1. For use with slater-expansion, we want
    define an "analytic continuation" to all polar numbers, which is
    continuous on circles and on the ray t*exp_polar(I*pi). Moreover, we want
    a "nice" expression for the various cases.

    This base class contains the core logic, concrete derived classes only
    supply the actual functions.

    """


    @classmethod
    def eval(cls, *args):
        newargs = tuple(map(unpolarify, args[:-1])) + args[-1:]
        if args != newargs:
            return cls(*newargs)

    @classmethod
    def _expr_small(cls, x):
        """ An expression for F(x) which holds for |x| < 1. """
        raise NotImplementedError

    @classmethod
    def _expr_small_minus(cls, x):
        """ An expression for F(-x) which holds for |x| < 1. """
        raise NotImplementedError

    @classmethod
    def _expr_big(cls, x, n):
        """ An expression for F(exp_polar(2*I*pi*n)*x), |x| > 1. """
        raise NotImplementedError

    @classmethod
    def _expr_big_minus(cls, x, n):
        """ An expression for F(exp_polar(2*I*pi*n + pi*I)*x), |x| > 1. """
        raise NotImplementedError

    def _eval_rewrite_as_nonrep(self, *args, **kwargs):
        x, n = self.args[-1].extract_branch_factor(allow_half=True)
        minus = False
        newargs = self.args[:-1] + (x,)
        if not n.is_Integer:
            minus = True
            n -= S.Half
        newerargs = newargs + (n,)
        if minus:
            small = self._expr_small_minus(*newargs)
            big = self._expr_big_minus(*newerargs)
        else:
            small = self._expr_small(*newargs)
            big = self._expr_big(*newerargs)

        if big == small:
            return small
        return Piecewise((big, abs(x) > 1), (small, True))

    def _eval_rewrite_as_nonrepsmall(self, *args, **kwargs):
        x, n = self.args[-1].extract_branch_factor(allow_half=True)
        args = self.args[:-1] + (x,)
        if not n.is_Integer:
            return self._expr_small_minus(*args)
        return self._expr_small(*args)


class HyperRep_power1(HyperRep):
    """ Return a representative for hyper([-a], [], z) == (1 - z)**a. """

    @classmethod
    def _expr_small(cls, a, x):
        return (1 - x)**a

    @classmethod
    def _expr_small_minus(cls, a, x):
        return (1 + x)**a

    @classmethod
    def _expr_big(cls, a, x, n):
        if a.is_integer:
            return cls._expr_small(a, x)
        return (x - 1)**a*exp((2*n - 1)*pi*I*a)

    @classmethod
    def _expr_big_minus(cls, a, x, n):
        if a.is_integer:
            return cls._expr_small_minus(a, x)
        return (1 + x)**a*exp(2*n*pi*I*a)


class HyperRep_power2(HyperRep):
    """ Return a representative for hyper([a, a - 1/2], [2*a], z). """

    @classmethod
    def _expr_small(cls, a, x):
        return 2**(2*a - 1)*(1 + sqrt(1 - x))**(1 - 2*a)

    @classmethod
    def _expr_small_minus(cls, a, x):
        return 2**(2*a - 1)*(1 + sqrt(1 + x))**(1 - 2*a)

    @classmethod
    def _expr_big(cls, a, x, n):
        sgn = -1
        if n.is_odd:
            sgn = 1
            n -= 1
        return 2**(2*a - 1)*(1 + sgn*I*sqrt(x - 1))**(1 - 2*a) \
            *exp(-2*n*pi*I*a)

    @classmethod
    def _expr_big_minus(cls, a, x, n):
        sgn = 1
        if n.is_odd:
            sgn = -1
        return sgn*2**(2*a - 1)*(sqrt(1 + x) + sgn)**(1 - 2*a)*exp(-2*pi*I*a*n)


class HyperRep_log1(HyperRep):
    """ Represent -z*hyper([1, 1], [2], z) == log(1 - z). """
    @classmethod
    def _expr_small(cls, x):
        return log(1 - x)

    @classmethod
    def _expr_small_minus(cls, x):
        return log(1 + x)

    @classmethod
    def _expr_big(cls, x, n):
        return log(x - 1) + (2*n - 1)*pi*I

    @classmethod
    def _expr_big_minus(cls, x, n):
        return log(1 + x) + 2*n*pi*I


class HyperRep_atanh(HyperRep):
    """ Represent hyper([1/2, 1], [3/2], z) == atanh(sqrt(z))/sqrt(z). """
    @classmethod
    def _expr_small(cls, x):
        return atanh(sqrt(x))/sqrt(x)

    def _expr_small_minus(cls, x):
        return atan(sqrt(x))/sqrt(x)

    def _expr_big(cls, x, n):
        if n.is_even:
            return (acoth(sqrt(x)) + I*pi/2)/sqrt(x)
        else:
            return (acoth(sqrt(x)) - I*pi/2)/sqrt(x)

    def _expr_big_minus(cls, x, n):
        if n.is_even:
            return atan(sqrt(x))/sqrt(x)
        else:
            return (atan(sqrt(x)) - pi)/sqrt(x)


class HyperRep_asin1(HyperRep):
    """ Represent hyper([1/2, 1/2], [3/2], z) == asin(sqrt(z))/sqrt(z). """
    @classmethod
    def _expr_small(cls, z):
        return asin(sqrt(z))/sqrt(z)

    @classmethod
    def _expr_small_minus(cls, z):
        return asinh(sqrt(z))/sqrt(z)

    @classmethod
    def _expr_big(cls, z, n):
        return S.NegativeOne**n*((S.Half - n)*pi/sqrt(z) + I*acosh(sqrt(z))/sqrt(z))

    @classmethod
    def _expr_big_minus(cls, z, n):
        return S.NegativeOne**n*(asinh(sqrt(z))/sqrt(z) + n*pi*I/sqrt(z))


class HyperRep_asin2(HyperRep):
    """ Represent hyper([1, 1], [3/2], z) == asin(sqrt(z))/sqrt(z)/sqrt(1-z). """
    # TODO this can be nicer
    @classmethod
    def _expr_small(cls, z):
        return HyperRep_asin1._expr_small(z) \
            /HyperRep_power1._expr_small(S.Half, z)

    @classmethod
    def _expr_small_minus(cls, z):
        return HyperRep_asin1._expr_small_minus(z) \
            /HyperRep_power1._expr_small_minus(S.Half, z)

    @classmethod
    def _expr_big(cls, z, n):
        return HyperRep_asin1._expr_big(z, n) \
            /HyperRep_power1._expr_big(S.Half, z, n)

    @classmethod
    def _expr_big_minus(cls, z, n):
        return HyperRep_asin1._expr_big_minus(z, n) \
            /HyperRep_power1._expr_big_minus(S.Half, z, n)


class HyperRep_sqrts1(HyperRep):
    """ Return a representative for hyper([-a, 1/2 - a], [1/2], z). """

    @classmethod
    def _expr_small(cls, a, z):
        return ((1 - sqrt(z))**(2*a) + (1 + sqrt(z))**(2*a))/2

    @classmethod
    def _expr_small_minus(cls, a, z):
        return (1 + z)**a*cos(2*a*atan(sqrt(z)))

    @classmethod
    def _expr_big(cls, a, z, n):
        if n.is_even:
            return ((sqrt(z) + 1)**(2*a)*exp(2*pi*I*n*a) +
                    (sqrt(z) - 1)**(2*a)*exp(2*pi*I*(n - 1)*a))/2
        else:
            n -= 1
            return ((sqrt(z) - 1)**(2*a)*exp(2*pi*I*a*(n + 1)) +
                    (sqrt(z) + 1)**(2*a)*exp(2*pi*I*a*n))/2

    @classmethod
    def _expr_big_minus(cls, a, z, n):
        if n.is_even:
            return (1 + z)**a*exp(2*pi*I*n*a)*cos(2*a*atan(sqrt(z)))
        else:
            return (1 + z)**a*exp(2*pi*I*n*a)*cos(2*a*atan(sqrt(z)) - 2*pi*a)


class HyperRep_sqrts2(HyperRep):
    """ Return a representative for
          sqrt(z)/2*[(1-sqrt(z))**2a - (1 + sqrt(z))**2a]
          == -2*z/(2*a+1) d/dz hyper([-a - 1/2, -a], [1/2], z)"""

    @classmethod
    def _expr_small(cls, a, z):
        return sqrt(z)*((1 - sqrt(z))**(2*a) - (1 + sqrt(z))**(2*a))/2

    @classmethod
    def _expr_small_minus(cls, a, z):
        return sqrt(z)*(1 + z)**a*sin(2*a*atan(sqrt(z)))

    @classmethod
    def _expr_big(cls, a, z, n):
        if n.is_even:
            return sqrt(z)/2*((sqrt(z) - 1)**(2*a)*exp(2*pi*I*a*(n - 1)) -
                              (sqrt(z) + 1)**(2*a)*exp(2*pi*I*a*n))
        else:
            n -= 1
            return sqrt(z)/2*((sqrt(z) - 1)**(2*a)*exp(2*pi*I*a*(n + 1)) -
                              (sqrt(z) + 1)**(2*a)*exp(2*pi*I*a*n))

    def _expr_big_minus(cls, a, z, n):
        if n.is_even:
            return (1 + z)**a*exp(2*pi*I*n*a)*sqrt(z)*sin(2*a*atan(sqrt(z)))
        else:
            return (1 + z)**a*exp(2*pi*I*n*a)*sqrt(z) \
                *sin(2*a*atan(sqrt(z)) - 2*pi*a)


class HyperRep_log2(HyperRep):
    """ Represent log(1/2 + sqrt(1 - z)/2) == -z/4*hyper([3/2, 1, 1], [2, 2], z) """

    @classmethod
    def _expr_small(cls, z):
        return log(S.Half + sqrt(1 - z)/2)

    @classmethod
    def _expr_small_minus(cls, z):
        return log(S.Half + sqrt(1 + z)/2)

    @classmethod
    def _expr_big(cls, z, n):
        if n.is_even:
            return (n - S.Half)*pi*I + log(sqrt(z)/2) + I*asin(1/sqrt(z))
        else:
            return (n - S.Half)*pi*I + log(sqrt(z)/2) - I*asin(1/sqrt(z))

    def _expr_big_minus(cls, z, n):
        if n.is_even:
            return pi*I*n + log(S.Half + sqrt(1 + z)/2)
        else:
            return pi*I*n + log(sqrt(1 + z)/2 - S.Half)


class HyperRep_cosasin(HyperRep):
    """ Represent hyper([a, -a], [1/2], z) == cos(2*a*asin(sqrt(z))). """
    # Note there are many alternative expressions, e.g. as powers of a sum of
    # square roots.

    @classmethod
    def _expr_small(cls, a, z):
        return cos(2*a*asin(sqrt(z)))

    @classmethod
    def _expr_small_minus(cls, a, z):
        return cosh(2*a*asinh(sqrt(z)))

    @classmethod
    def _expr_big(cls, a, z, n):
        return cosh(2*a*acosh(sqrt(z)) + a*pi*I*(2*n - 1))

    @classmethod
    def _expr_big_minus(cls, a, z, n):
        return cosh(2*a*asinh(sqrt(z)) + 2*a*pi*I*n)


class HyperRep_sinasin(HyperRep):
    """ Represent 2*a*z*hyper([1 - a, 1 + a], [3/2], z)
        == sqrt(z)/sqrt(1-z)*sin(2*a*asin(sqrt(z))) """

    @classmethod
    def _expr_small(cls, a, z):
        return sqrt(z)/sqrt(1 - z)*sin(2*a*asin(sqrt(z)))

    @classmethod
    def _expr_small_minus(cls, a, z):
        return -sqrt(z)/sqrt(1 + z)*sinh(2*a*asinh(sqrt(z)))

    @classmethod
    def _expr_big(cls, a, z, n):
        return -1/sqrt(1 - 1/z)*sinh(2*a*acosh(sqrt(z)) + a*pi*I*(2*n - 1))

    @classmethod
    def _expr_big_minus(cls, a, z, n):
        return -1/sqrt(1 + 1/z)*sinh(2*a*asinh(sqrt(z)) + 2*a*pi*I*n)

class appellf1(DefinedFunction):
    r"""
    This is the Appell hypergeometric function of two variables as:

    .. math ::
        F_1(a,b_1,b_2,c,x,y) = \sum_{m=0}^{\infty} \sum_{n=0}^{\infty}
        \frac{(a)_{m+n} (b_1)_m (b_2)_n}{(c)_{m+n}}
        \frac{x^m y^n}{m! n!}.

    Examples
    ========

    >>> from sympy import appellf1, symbols
    >>> x, y, a, b1, b2, c = symbols('x y a b1 b2 c')
    >>> appellf1(2., 1., 6., 4., 5., 6.)
    0.0063339426292673
    >>> appellf1(12., 12., 6., 4., 0.5, 0.12)
    172870711.659936
    >>> appellf1(40, 2, 6, 4, 15, 60)
    appellf1(40, 2, 6, 4, 15, 60)
    >>> appellf1(20., 12., 10., 3., 0.5, 0.12)
    15605338197184.4
    >>> appellf1(40, 2, 6, 4, x, y)
    appellf1(40, 2, 6, 4, x, y)
    >>> appellf1(a, b1, b2, c, x, y)
    appellf1(a, b1, b2, c, x, y)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Appell_series
    .. [2] https://functions.wolfram.com/HypergeometricFunctions/AppellF1/

    """

    @classmethod
    def eval(cls, a, b1, b2, c, x, y):
        if default_sort_key(b1) > default_sort_key(b2):
            b1, b2 = b2, b1
            x, y = y, x
            return cls(a, b1, b2, c, x, y)
        elif b1 == b2 and default_sort_key(x) > default_sort_key(y):
            x, y = y, x
            return cls(a, b1, b2, c, x, y)
        if x == 0 and y == 0:
            return S.One

    def fdiff(self, argindex=5):
        a, b1, b2, c, x, y = self.args
        if argindex == 5:
            return (a*b1/c)*appellf1(a + 1, b1 + 1, b2, c + 1, x, y)
        elif argindex == 6:
            return (a*b2/c)*appellf1(a + 1, b1, b2 + 1, c + 1, x, y)
        elif argindex in (1, 2, 3, 4):
            return Derivative(self, self.args[argindex-1])
        else:
            raise ArgumentIndexError(self, argindex)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\stable_diffusion\diffusion_schedulers.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Modified from utilities.py of TensorRT demo diffusion, which has the following license:
#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------


import numpy as np
import torch


class DDIMScheduler:
    def __init__(
        self,
        device="cuda",
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        clip_sample: bool = False,
        set_alpha_to_one: bool = False,
        steps_offset: int = 1,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "leading",
    ):
        # this schedule is very specific to the latent diffusion model.
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2

        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))
        self.steps_offset = steps_offset
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        self.device = device
        self.timestep_spacing = timestep_spacing

    def configure(self):
        variance = np.zeros(self.num_inference_steps, dtype=np.float32)
        for idx, timestep in enumerate(self.timesteps):
            prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
            variance[idx] = self._get_variance(timestep, prev_timestep)
        self.variance = torch.from_numpy(variance).to(self.device)

        timesteps = self.timesteps.long().cpu()
        self.filtered_alphas_cumprod = self.alphas_cumprod[timesteps].to(self.device)
        self.final_alpha_cumprod = self.final_alpha_cumprod.to(self.device)

    def scale_model_input(self, sample: torch.FloatTensor, idx, *args, **kwargs) -> torch.FloatTensor:
        return sample

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        if self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(np.arange(self.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        self.timesteps = torch.from_numpy(timesteps).to(self.device)

    def step(
        self,
        model_output,
        sample,
        idx,
        timestep,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: torch.FloatTensor = None,
    ):
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        prev_idx = idx + 1
        alpha_prod_t = self.filtered_alphas_cumprod[idx]
        alpha_prod_t_prev = (
            self.filtered_alphas_cumprod[prev_idx] if prev_idx < self.num_inference_steps else self.final_alpha_cumprod
        )

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            # predict V
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`"
            )

        # 4. Clip "predicted x_0"
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # o_t = sqrt((1 - a_t-1)/(1 - a_t)) * sqrt(1 - a_t/a_t-1)
        variance = self.variance[idx]
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the model_output is always re-derived from the clipped x_0 in Glide
            model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            # randn_like does not support generator https://github.com/pytorch/pytorch/issues/27072
            device = model_output.device
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = torch.randn(
                    model_output.shape, generator=generator, device=device, dtype=model_output.dtype
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        return prev_sample

    def add_noise(self, init_latents, noise, idx, latent_timestep):
        sqrt_alpha_prod = self.filtered_alphas_cumprod[idx] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.filtered_alphas_cumprod[idx]) ** 0.5
        noisy_latents = sqrt_alpha_prod * init_latents + sqrt_one_minus_alpha_prod * noise

        return noisy_latents


class EulerAncestralDiscreteScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device="cuda",
        steps_offset: int = 1,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "trailing",  # set default to trailing for SDXL Turbo
    ):
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = self.sigmas.max()

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.is_scale_input_called = False

        self._step_index = None

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.steps_offset = steps_offset
        self.prediction_type = prediction_type
        self.timestep_spacing = timestep_spacing

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index
    def _init_step_index(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        index_candidates = (self.timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        if len(index_candidates) > 1:
            step_index = index_candidates[1]
        else:
            step_index = index_candidates[0]

        self._step_index = step_index.item()

    def scale_model_input(self, sample: torch.FloatTensor, idx, timestep, *args, **kwargs) -> torch.FloatTensor:
        if self._step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self._step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)
        self.is_scale_input_called = True
        return sample

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps

        if self.timestep_spacing == "linspace":
            timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps, dtype=np.float32)[::-1].copy()
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
            timesteps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(self.num_train_timesteps, 0, -step_ratio)).round().copy().astype(np.float32)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas).to(device=self.device)
        self.timesteps = torch.from_numpy(timesteps).to(device=self.device)

        self._step_index = None

    def configure(self):
        dts = np.zeros(self.num_inference_steps, dtype=np.float32)
        sigmas_up = np.zeros(self.num_inference_steps, dtype=np.float32)
        for idx, timestep in enumerate(self.timesteps):
            step_index = (self.timesteps == timestep).nonzero().item()
            sigma = self.sigmas[step_index]

            sigma_from = self.sigmas[step_index]
            sigma_to = self.sigmas[step_index + 1]
            sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
            sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
            dt = sigma_down - sigma
            dts[idx] = dt
            sigmas_up[idx] = sigma_up

        self.dts = torch.from_numpy(dts).to(self.device)
        self.sigmas_up = torch.from_numpy(sigmas_up).to(self.device)

    def step(
        self,
        model_output,
        sample,
        idx,
        timestep,
        generator=None,
    ):
        if self._step_index is None:
            self._init_step_index(timestep)
        sigma = self.sigmas[self._step_index]

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        sigma_from = self.sigmas[self._step_index]
        sigma_to = self.sigmas[self._step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma

        dt = sigma_down - sigma

        prev_sample = sample + derivative * dt

        device = model_output.device
        noise = torch.randn(model_output.shape, dtype=model_output.dtype, device=device, generator=generator).to(device)

        prev_sample = prev_sample + noise * sigma_up

        # upon completion increase step index by one
        self._step_index += 1

        return prev_sample

    def add_noise(self, original_samples, noise, idx, timestep=None):
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        schedule_timesteps = self.timesteps.to(original_samples.device)
        timesteps = timestep.to(original_samples.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples


class UniPCMultistepScheduler:
    def __init__(
        self,
        device="cuda",
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        solver_order: int = 2,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: list[int] | None = None,
        use_karras_sigmas: bool | None = False,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        sigma_min=None,
        sigma_max=None,
    ):
        self.device = device
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # Currently we only support VP-type noise schedule
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        self.predict_x0 = predict_x0
        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.disable_corrector = disable_corrector if disable_corrector else []
        self.last_sample = None

        self._step_index = None

        self.num_train_timesteps = num_train_timesteps
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.use_karras_sigmas = use_karras_sigmas
        self.timestep_spacing = timestep_spacing
        self.steps_offset = steps_offset
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    def set_timesteps(self, num_inference_steps: int):
        if self.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.num_train_timesteps - 1, num_inference_steps + 1)
                .round()[::-1][:-1]
                .copy()
                .astype(np.int64)
            )
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps // (num_inference_steps + 1)
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].copy().astype(np.int64)
            timesteps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.arange(self.num_train_timesteps, 0, -step_ratio).round().copy().astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        if self.use_karras_sigmas:
            log_sigmas = np.log(sigmas)
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
            sigmas = np.concatenate([sigmas, sigmas[-1:]]).astype(np.float32)
        else:
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=self.device, dtype=torch.int64)

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [
            None,
        ] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t
    def _sigma_to_t(self, sigma, log_sigmas):
        # get log sigma
        log_sigma = np.log(np.maximum(sigma, 1e-10))

        # get distribution
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # get sigmas range
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        return t

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler._sigma_to_alpha_sigma_t
    def _sigma_to_alpha_sigma_t(self, sigma):
        alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
        sigma_t = sigma * alpha_t

        return alpha_t, sigma_t

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras
    def _convert_to_karras(self, in_sigmas: torch.FloatTensor, num_inference_steps) -> torch.FloatTensor:
        """Constructs the noise schedule of Karras et al. (2022)."""

        sigma_min = self.sigma_min
        sigma_max = self.sigma_max

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        rho = 7.0  # 7.0 is the value used in the paper
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def convert_model_output(
        self,
        model_output: torch.FloatTensor,
        *args,
        sample: torch.FloatTensor = None,
        **kwargs,
    ) -> torch.FloatTensor:
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyword argument")
        if timestep is not None:
            print(
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        sigma = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        if self.predict_x0:
            if self.prediction_type == "epsilon":
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.prediction_type == "sample":
                x0_pred = model_output
            elif self.prediction_type == "v_prediction":
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the UniPCMultistepScheduler."
                )

            if self.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred
        else:
            if self.prediction_type == "epsilon":
                return model_output
            elif self.prediction_type == "sample":
                epsilon = (sample - alpha_t * model_output) / sigma_t
                return epsilon
            elif self.prediction_type == "v_prediction":
                epsilon = alpha_t * model_output + sigma_t * sample
                return epsilon
            else:
                raise ValueError(
                    f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the UniPCMultistepScheduler."
                )

    def multistep_uni_p_bh_update(
        self,
        model_output: torch.FloatTensor,
        *args,
        sample: torch.FloatTensor = None,
        order: int | None = None,
        **kwargs,
    ) -> torch.FloatTensor:
        prev_timestep = args[0] if len(args) > 0 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError(" missing `sample` as a required keyword argument")
        if order is None:
            if len(args) > 2:
                order = args[2]
            else:
                raise ValueError(" missing `order` as a required keyword argument")
        if prev_timestep is not None:
            print(
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )
        model_output_list = self.model_outputs

        # s0 = self.timestep_list[-1]
        m0 = model_output_list[-1]
        x = sample

        sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = sample.device

        rks = []
        d1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        r = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.solver_type == "bh1":
            b_h = hh
        elif self.solver_type == "bh2":
            b_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            r.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        r = torch.stack(r)
        b = torch.tensor(b, device=device)

        if len(d1s) > 0:
            d1s = torch.stack(d1s, dim=1)  # (B, K)
            # for order 2, we use a simplified version
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(r[:-1, :-1], b[:-1])
        else:
            d1s = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if d1s is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1s)
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * b_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if d1s is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1s)
            else:
                pred_res = 0
            x_t = x_t_ - sigma_t * b_h * pred_res

        x_t = x_t.to(x.dtype)
        return x_t

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.FloatTensor,
        *args,
        last_sample: torch.FloatTensor = None,
        this_sample: torch.FloatTensor = None,
        order: int | None = None,
        **kwargs,
    ) -> torch.FloatTensor:
        this_timestep = args[0] if len(args) > 0 else kwargs.pop("this_timestep", None)
        if last_sample is None:
            if len(args) > 1:
                last_sample = args[1]
            else:
                raise ValueError(" missing`last_sample` as a required keyword argument")
        if this_sample is None:
            if len(args) > 2:
                this_sample = args[2]
            else:
                raise ValueError(" missing`this_sample` as a required keyword argument")
        if order is None:
            if len(args) > 3:
                order = args[3]
            else:
                raise ValueError(" missing`order` as a required keyword argument")
        if this_timestep is not None:
            print(
                "Passing `this_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        model_output_list = self.model_outputs

        m0 = model_output_list[-1]
        x = last_sample
        # x_t = this_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = this_sample.device

        rks = []
        d1s = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        r = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.solver_type == "bh1":
            b_h = hh
        elif self.solver_type == "bh2":
            b_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            r.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        r = torch.stack(r)
        b = torch.tensor(b, device=device)

        if len(d1s) > 0:
            d1s = torch.stack(d1s, dim=1)
        else:
            d1s = None

        # for order 1, we use a simplified version
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(r, b)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if d1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s)
            else:
                corr_res = 0
            d1_t = model_t - m0
            x_t = x_t_ - alpha_t * b_h * (corr_res + rhos_c[-1] * d1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if d1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s)
            else:
                corr_res = 0
            d1_t = model_t - m0
            x_t = x_t_ - sigma_t * b_h * (corr_res + rhos_c[-1] * d1_t)
        x_t = x_t.to(x.dtype)
        return x_t

    def _init_step_index(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        index_candidates = (self.timesteps == timestep).nonzero()

        if len(index_candidates) == 0:
            step_index = len(self.timesteps) - 1
        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            step_index = index_candidates[0].item()

        self._step_index = step_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ):
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        use_corrector = (
            self.step_index > 0 and self.step_index - 1 not in self.disable_corrector and self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(model_output, sample=sample)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.lower_order_final:
            this_order = min(self.solver_order, len(self.timesteps) - self.step_index)
        else:
            this_order = self.solver_order

        self.this_order = min(this_order, self.lower_order_nums + 1)  # warmup for multistep
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,  # pass the original non-converted model output, in case solver-p is used
            sample=sample,
            order=self.this_order,
        )

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return prev_sample

    def scale_model_input(self, sample: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        return sample

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        idx,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        schedule_timesteps = self.timesteps.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

    def configure(self):
        pass

    def __len__(self):
        return self.num_train_timesteps


# Modified from diffusers.schedulers.LCMScheduler
class LCMScheduler:
    def __init__(
        self,
        device="cuda",
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        original_inference_steps: int = 50,
        clip_sample: bool = False,
        clip_sample_range: float = 1.0,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        timestep_scaling: float = 10.0,
    ):
        self.device = device
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = self.alphas_cumprod[0]
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0
        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.steps_offset = steps_offset
        self.prediction_type = prediction_type
        self.thresholding = thresholding
        self.timestep_spacing = timestep_spacing
        self.timestep_scaling = timestep_scaling
        self.original_inference_steps = original_inference_steps
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value

        self._step_index = None

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index
    def _init_step_index(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        index_candidates = (self.timesteps == timestep).nonzero()

        if len(index_candidates) > 1:
            step_index = index_candidates[1]
        else:
            step_index = index_candidates[0]

        self._step_index = step_index.item()

    @property
    def step_index(self):
        return self._step_index

    def scale_model_input(self, sample: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        return sample

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    def set_timesteps(
        self,
        num_inference_steps: int,
        strength: int = 1.0,
    ):
        assert num_inference_steps <= self.num_train_timesteps

        self.num_inference_steps = num_inference_steps
        original_steps = self.original_inference_steps

        assert original_steps <= self.num_train_timesteps
        assert num_inference_steps <= original_steps

        # LCM Timesteps Setting
        # Currently, only linear spacing is supported.
        c = self.num_train_timesteps // original_steps
        # LCM Training Steps Schedule
        lcm_origin_timesteps = np.asarray(list(range(1, int(original_steps * strength) + 1))) * c - 1
        skipping_step = len(lcm_origin_timesteps) // num_inference_steps
        # LCM Inference Steps Schedule
        timesteps = lcm_origin_timesteps[::-skipping_step][:num_inference_steps]

        self.timesteps = torch.from_numpy(timesteps.copy()).to(device=self.device, dtype=torch.long)

        self._step_index = None

    def get_scalings_for_boundary_condition_discrete(self, timestep):
        self.sigma_data = 0.5  # Default: 0.5
        scaled_timestep = timestep * self.timestep_scaling

        c_skip = self.sigma_data**2 / (scaled_timestep**2 + self.sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator: torch.Generator | None = None,
    ):
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # 1. get previous step value
        prev_step_index = self.step_index + 1
        if prev_step_index < len(self.timesteps):
            prev_timestep = self.timesteps[prev_step_index]
        else:
            prev_timestep = timestep

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 3. Get scalings for boundary conditions
        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        # 4. Compute the predicted original sample x_0 based on the model parameterization
        if self.prediction_type == "epsilon":  # noise-prediction
            predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif self.prediction_type == "sample":  # x-prediction
            predicted_original_sample = model_output
        elif self.prediction_type == "v_prediction":  # v-prediction
            predicted_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction` for `LCMScheduler`."
            )

        # 5. Clip or threshold "predicted x_0"
        if self.thresholding:
            predicted_original_sample = self._threshold_sample(predicted_original_sample)
        elif self.clip_sample:
            predicted_original_sample = predicted_original_sample.clamp(-self.clip_sample_range, self.clip_sample_range)

        # 6. Denoise model output using boundary conditions
        denoised = c_out * predicted_original_sample + c_skip * sample

        # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
        # Noise is not used on the final timestep of the timestep schedule.
        # This also means that noise is not used for one-step sampling.
        if self.step_index != self.num_inference_steps - 1:
            noise = torch.randn(
                model_output.shape, device=model_output.device, dtype=denoised.dtype, generator=generator
            )
            prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
        else:
            prev_sample = denoised

        # upon completion increase step index by one
        self._step_index += 1

        return (prev_sample,)

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def configure(self):
        pass

    def __len__(self):
        return self.num_train_timesteps

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\autowrap.py ===
"""Module for compiling codegen output, and wrap the binary for use in
python.

.. note:: To use the autowrap module it must first be imported

   >>> from sympy.utilities.autowrap import autowrap

This module provides a common interface for different external backends, such
as f2py, fwrap, Cython, SWIG(?) etc. (Currently only f2py and Cython are
implemented) The goal is to provide access to compiled binaries of acceptable
performance with a one-button user interface, e.g.,

    >>> from sympy.abc import x,y
    >>> expr = (x - y)**25
    >>> flat = expr.expand()
    >>> binary_callable = autowrap(flat)
    >>> binary_callable(2, 3)
    -1.0

Although a SymPy user might primarily be interested in working with
mathematical expressions and not in the details of wrapping tools
needed to evaluate such expressions efficiently in numerical form,
the user cannot do so without some understanding of the
limits in the target language. For example, the expanded expression
contains large coefficients which result in loss of precision when
computing the expression:

    >>> binary_callable(3, 2)
    0.0
    >>> binary_callable(4, 5), binary_callable(5, 4)
    (-22925376.0, 25165824.0)

Wrapping the unexpanded expression gives the expected behavior:

    >>> e = autowrap(expr)
    >>> e(4, 5), e(5, 4)
    (-1.0, 1.0)

The callable returned from autowrap() is a binary Python function, not a
SymPy object.  If it is desired to use the compiled function in symbolic
expressions, it is better to use binary_function() which returns a SymPy
Function object.  The binary callable is attached as the _imp_ attribute and
invoked when a numerical evaluation is requested with evalf(), or with
lambdify().

    >>> from sympy.utilities.autowrap import binary_function
    >>> f = binary_function('f', expr)
    >>> 2*f(x, y) + y
    y + 2*f(x, y)
    >>> (2*f(x, y) + y).evalf(2, subs={x: 1, y:2})
    0.e-110

When is this useful?

    1) For computations on large arrays, Python iterations may be too slow,
       and depending on the mathematical expression, it may be difficult to
       exploit the advanced index operations provided by NumPy.

    2) For *really* long expressions that will be called repeatedly, the
       compiled binary should be significantly faster than SymPy's .evalf()

    3) If you are generating code with the codegen utility in order to use
       it in another project, the automatic Python wrappers let you test the
       binaries immediately from within SymPy.

    4) To create customized ufuncs for use with numpy arrays.
       See *ufuncify*.

When is this module NOT the best approach?

    1) If you are really concerned about speed or memory optimizations,
       you will probably get better results by working directly with the
       wrapper tools and the low level code.  However, the files generated
       by this utility may provide a useful starting point and reference
       code. Temporary files will be left intact if you supply the keyword
       tempdir="path/to/files/".

    2) If the array computation can be handled easily by numpy, and you
       do not need the binaries for another project.

"""

import sys
import os
import shutil
import tempfile
from pathlib import Path
from subprocess import STDOUT, CalledProcessError, check_output
from string import Template
from warnings import warn

from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy, Symbol
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.utilities.codegen import (make_routine, get_code_generator,
                                     OutputArgument, InOutArgument,
                                     InputArgument, CodeGenArgumentListError,
                                     Result, ResultBase, C99CodeGen)
from sympy.utilities.iterables import iterable
from sympy.utilities.lambdify import implemented_function
from sympy.utilities.decorator import doctest_depends_on

_doctest_depends_on = {'exe': ('f2py', 'gfortran', 'gcc'),
                       'modules': ('numpy',)}


class CodeWrapError(Exception):
    pass


class CodeWrapper:
    """Base Class for code wrappers"""
    _filename = "wrapped_code"
    _module_basename = "wrapper_module"
    _module_counter = 0

    @property
    def filename(self):
        return "%s_%s" % (self._filename, CodeWrapper._module_counter)

    @property
    def module_name(self):
        return "%s_%s" % (self._module_basename, CodeWrapper._module_counter)

    def __init__(self, generator, filepath=None, flags=[], verbose=False):
        """
        generator -- the code generator to use
        """
        self.generator = generator
        self.filepath = filepath
        self.flags = flags
        self.quiet = not verbose

    @property
    def include_header(self):
        return bool(self.filepath)

    @property
    def include_empty(self):
        return bool(self.filepath)

    def _generate_code(self, main_routine, routines):
        routines.append(main_routine)
        self.generator.write(
            routines, self.filename, True, self.include_header,
            self.include_empty)

    def wrap_code(self, routine, helpers=None):
        helpers = helpers or []
        if self.filepath:
            workdir = os.path.abspath(self.filepath)
        else:
            workdir = tempfile.mkdtemp("_sympy_compile")
        if not os.access(workdir, os.F_OK):
            os.mkdir(workdir)
        oldwork = os.getcwd()
        os.chdir(workdir)
        try:
            sys.path.append(workdir)
            self._generate_code(routine, helpers)
            self._prepare_files(routine)
            self._process_files(routine)
            mod = __import__(self.module_name)
        finally:
            sys.path.remove(workdir)
            CodeWrapper._module_counter += 1
            os.chdir(oldwork)
            if not self.filepath:
                try:
                    shutil.rmtree(workdir)
                except OSError:
                    # Could be some issues on Windows
                    pass

        return self._get_wrapped_function(mod, routine.name)

    def _process_files(self, routine):
        command = self.command
        command.extend(self.flags)
        try:
            retoutput = check_output(command, stderr=STDOUT)
        except CalledProcessError as e:
            raise CodeWrapError(
                "Error while executing command: %s. Command output is:\n%s" % (
                    " ".join(command), e.output.decode('utf-8')))
        if not self.quiet:
            print(retoutput)


class DummyWrapper(CodeWrapper):
    """Class used for testing independent of backends """

    template = """# dummy module for testing of SymPy
def %(name)s():
    return "%(expr)s"
%(name)s.args = "%(args)s"
%(name)s.returns = "%(retvals)s"
"""

    def _prepare_files(self, routine):
        return

    def _generate_code(self, routine, helpers):
        with open('%s.py' % self.module_name, 'w') as f:
            printed = ", ".join(
                [str(res.expr) for res in routine.result_variables])
            # convert OutputArguments to return value like f2py
            args = filter(lambda x: not isinstance(
                x, OutputArgument), routine.arguments)
            retvals = []
            for val in routine.result_variables:
                if isinstance(val, Result):
                    retvals.append('nameless')
                else:
                    retvals.append(val.result_var)

            print(DummyWrapper.template % {
                'name': routine.name,
                'expr': printed,
                'args': ", ".join([str(a.name) for a in args]),
                'retvals': ", ".join([str(val) for val in retvals])
            }, end="", file=f)

    def _process_files(self, routine):
        return

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name)


class CythonCodeWrapper(CodeWrapper):
    """Wrapper that uses Cython"""

    setup_template = """\
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
cy_opts = {cythonize_options}
{np_import}
ext_mods = [Extension(
    {ext_args},
    include_dirs={include_dirs},
    library_dirs={library_dirs},
    libraries={libraries},
    extra_compile_args={extra_compile_args},
    extra_link_args={extra_link_args}
)]
setup(ext_modules=cythonize(ext_mods, **cy_opts))
"""

    _cythonize_options = {'compiler_directives':{'language_level' : "3"}}

    pyx_imports = (
        "import numpy as np\n"
        "cimport numpy as np\n\n")

    pyx_header = (
        "cdef extern from '{header_file}.h':\n"
        "    {prototype}\n\n")

    pyx_func = (
        "def {name}_c({arg_string}):\n"
        "\n"
        "{declarations}"
        "{body}")

    std_compile_flag = '-std=c99'

    def __init__(self, *args, **kwargs):
        """Instantiates a Cython code wrapper.

        The following optional parameters get passed to ``setuptools.Extension``
        for building the Python extension module. Read its documentation to
        learn more.

        Parameters
        ==========
        include_dirs : [list of strings]
            A list of directories to search for C/C++ header files (in Unix
            form for portability).
        library_dirs : [list of strings]
            A list of directories to search for C/C++ libraries at link time.
        libraries : [list of strings]
            A list of library names (not filenames or paths) to link against.
        extra_compile_args : [list of strings]
            Any extra platform- and compiler-specific information to use when
            compiling the source files in 'sources'.  For platforms and
            compilers where "command line" makes sense, this is typically a
            list of command-line arguments, but for other platforms it could be
            anything. Note that the attribute ``std_compile_flag`` will be
            appended to this list.
        extra_link_args : [list of strings]
            Any extra platform- and compiler-specific information to use when
            linking object files together to create the extension (or to create
            a new static Python interpreter). Similar interpretation as for
            'extra_compile_args'.
        cythonize_options : [dictionary]
            Keyword arguments passed on to cythonize.

        """

        self._include_dirs = kwargs.pop('include_dirs', [])
        self._library_dirs = kwargs.pop('library_dirs', [])
        self._libraries = kwargs.pop('libraries', [])
        self._extra_compile_args = kwargs.pop('extra_compile_args', [])
        self._extra_compile_args.append(self.std_compile_flag)
        self._extra_link_args = kwargs.pop('extra_link_args', [])
        self._cythonize_options = kwargs.pop('cythonize_options', self._cythonize_options)

        self._need_numpy = False

        super().__init__(*args, **kwargs)

    @property
    def command(self):
        command = [sys.executable, "setup.py", "build_ext", "--inplace"]
        return command

    def _prepare_files(self, routine, build_dir=os.curdir):
        # NOTE : build_dir is used for testing purposes.
        pyxfilename = self.module_name + '.pyx'
        codefilename = "%s.%s" % (self.filename, self.generator.code_extension)

        # pyx
        with open(os.path.join(build_dir, pyxfilename), 'w') as f:
            self.dump_pyx([routine], f, self.filename)

        # setup.py
        ext_args = [repr(self.module_name), repr([pyxfilename, codefilename])]
        if self._need_numpy:
            np_import = 'import numpy as np\n'
            self._include_dirs.append('np.get_include()')
        else:
            np_import = ''

        includes = str(self._include_dirs).replace("'np.get_include()'",
                                                    'np.get_include()')
        code = self.setup_template.format(
                ext_args=", ".join(ext_args),
                np_import=np_import,
                include_dirs=includes,
                library_dirs=self._library_dirs,
                libraries=self._libraries,
                extra_compile_args=self._extra_compile_args,
                extra_link_args=self._extra_link_args,
                cythonize_options=self._cythonize_options)
        Path(os.path.join(build_dir, 'setup.py')).write_text(code)

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name + '_c')

    def dump_pyx(self, routines, f, prefix):
        """Write a Cython file with Python wrappers

        This file contains all the definitions of the routines in c code and
        refers to the header file.

        Arguments
        ---------
        routines
            List of Routine instances
        f
            File-like object to write the file to
        prefix
            The filename prefix, used to refer to the proper header file.
            Only the basename of the prefix is used.
        """
        headers = []
        functions = []
        for routine in routines:
            prototype = self.generator.get_prototype(routine)

            # C Function Header Import
            headers.append(self.pyx_header.format(header_file=prefix,
                                                  prototype=prototype))

            # Partition the C function arguments into categories
            py_rets, py_args, py_loc, py_inf = self._partition_args(routine.arguments)

            # Function prototype
            name = routine.name
            arg_string = ", ".join(self._prototype_arg(arg) for arg in py_args)

            # Local Declarations
            local_decs = []
            for arg, val in py_inf.items():
                proto = self._prototype_arg(arg)
                mat, ind = [self._string_var(v) for v in val]
                local_decs.append("    cdef {} = {}.shape[{}]".format(proto, mat, ind))
            local_decs.extend(["    cdef {}".format(self._declare_arg(a)) for a in py_loc])
            declarations = "\n".join(local_decs)
            if declarations:
                declarations = declarations + "\n"

            # Function Body
            args_c = ", ".join([self._call_arg(a) for a in routine.arguments])
            rets = ", ".join([self._string_var(r.name) for r in py_rets])
            if routine.results:
                body = '    return %s(%s)' % (routine.name, args_c)
                if rets:
                    body = body + ', ' + rets
            else:
                body = '    %s(%s)\n' % (routine.name, args_c)
                body = body + '    return ' + rets

            functions.append(self.pyx_func.format(name=name, arg_string=arg_string,
                    declarations=declarations, body=body))

        # Write text to file
        if self._need_numpy:
            # Only import numpy if required
            f.write(self.pyx_imports)
        f.write('\n'.join(headers))
        f.write('\n'.join(functions))

    def _partition_args(self, args):
        """Group function arguments into categories."""
        py_args = []
        py_returns = []
        py_locals = []
        py_inferred = {}
        for arg in args:
            if isinstance(arg, OutputArgument):
                py_returns.append(arg)
                py_locals.append(arg)
            elif isinstance(arg, InOutArgument):
                py_returns.append(arg)
                py_args.append(arg)
            else:
                py_args.append(arg)
        # Find arguments that are array dimensions. These can be inferred
        # locally in the Cython code.
            if isinstance(arg, (InputArgument, InOutArgument)) and arg.dimensions:
                dims = [d[1] + 1 for d in arg.dimensions]
                sym_dims = [(i, d) for (i, d) in enumerate(dims) if
                            isinstance(d, Symbol)]
                for (i, d) in sym_dims:
                    py_inferred[d] = (arg.name, i)
        for arg in args:
            if arg.name in py_inferred:
                py_inferred[arg] = py_inferred.pop(arg.name)
        # Filter inferred arguments from py_args
        py_args = [a for a in py_args if a not in py_inferred]
        return py_returns, py_args, py_locals, py_inferred

    def _prototype_arg(self, arg):
        mat_dec = "np.ndarray[{mtype}, ndim={ndim}] {name}"
        np_types = {'double': 'np.double_t',
                    'int': 'np.int_t'}
        t = arg.get_datatype('c')
        if arg.dimensions:
            self._need_numpy = True
            ndim = len(arg.dimensions)
            mtype = np_types[t]
            return mat_dec.format(mtype=mtype, ndim=ndim, name=self._string_var(arg.name))
        else:
            return "%s %s" % (t, self._string_var(arg.name))

    def _declare_arg(self, arg):
        proto = self._prototype_arg(arg)
        if arg.dimensions:
            shape = '(' + ','.join(self._string_var(i[1] + 1) for i in arg.dimensions) + ')'
            return proto + " = np.empty({shape})".format(shape=shape)
        else:
            return proto + " = 0"

    def _call_arg(self, arg):
        if arg.dimensions:
            t = arg.get_datatype('c')
            return "<{}*> {}.data".format(t, self._string_var(arg.name))
        elif isinstance(arg, ResultBase):
            return "&{}".format(self._string_var(arg.name))
        else:
            return self._string_var(arg.name)

    def _string_var(self, var):
        printer = self.generator.printer.doprint
        return printer(var)


class F2PyCodeWrapper(CodeWrapper):
    """Wrapper that uses f2py"""

    def __init__(self, *args, **kwargs):

        ext_keys = ['include_dirs', 'library_dirs', 'libraries',
                    'extra_compile_args', 'extra_link_args']
        msg = ('The compilation option kwarg {} is not supported with the f2py '
               'backend.')

        for k in ext_keys:
            if k in kwargs.keys():
                warn(msg.format(k))
            kwargs.pop(k, None)

        super().__init__(*args, **kwargs)

    @property
    def command(self):
        filename = self.filename + '.' + self.generator.code_extension
        args = ['-c', '-m', self.module_name, filename]
        command = [sys.executable, "-c", "import numpy.f2py as f2py2e;f2py2e.main()"]+args
        return command

    def _prepare_files(self, routine):
        pass

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name)


# Here we define a lookup of backends -> tuples of languages. For now, each
# tuple is of length 1, but if a backend supports more than one language,
# the most preferable language is listed first.
_lang_lookup = {'CYTHON': ('C99', 'C89', 'C'),
                'F2PY': ('F95',),
                'NUMPY': ('C99', 'C89', 'C'),
                'DUMMY': ('F95',)}     # Dummy here just for testing


def _infer_language(backend):
    """For a given backend, return the top choice of language"""
    langs = _lang_lookup.get(backend.upper(), False)
    if not langs:
        raise ValueError("Unrecognized backend: " + backend)
    return langs[0]


def _validate_backend_language(backend, language):
    """Throws error if backend and language are incompatible"""
    langs = _lang_lookup.get(backend.upper(), False)
    if not langs:
        raise ValueError("Unrecognized backend: " + backend)
    if language.upper() not in langs:
        raise ValueError(("Backend {} and language {} are "
                          "incompatible").format(backend, language))


@cacheit
@doctest_depends_on(exe=('f2py', 'gfortran'), modules=('numpy',))
def autowrap(expr, language=None, backend='f2py', tempdir=None, args=None,
             flags=None, verbose=False, helpers=None, code_gen=None, **kwargs):
    """Generates Python callable binaries based on the math expression.

    Parameters
    ==========

    expr
        The SymPy expression that should be wrapped as a binary routine.
    language : string, optional
        If supplied, (options: 'C' or 'F95'), specifies the language of the
        generated code. If ``None`` [default], the language is inferred based
        upon the specified backend.
    backend : string, optional
        Backend used to wrap the generated code. Either 'f2py' [default],
        or 'cython'.
    tempdir : string, optional
        Path to directory for temporary files. If this argument is supplied,
        the generated code and the wrapper input files are left intact in the
        specified path.
    args : iterable, optional
        An ordered iterable of symbols. Specifies the argument sequence for the
        function.
    flags : iterable, optional
        Additional option flags that will be passed to the backend.
    verbose : bool, optional
        If True, autowrap will not mute the command line backends. This can be
        helpful for debugging.
    helpers : 3-tuple or iterable of 3-tuples, optional
        Used to define auxiliary functions needed for the main expression.
        Each tuple should be of the form (name, expr, args) where:

        - name : str, the function name
        - expr : sympy expression, the function
        - args : iterable, the function arguments (can be any iterable of symbols)

    code_gen : CodeGen instance
        An instance of a CodeGen subclass. Overrides ``language``.
    include_dirs : [string]
        A list of directories to search for C/C++ header files (in Unix form
        for portability).
    library_dirs : [string]
        A list of directories to search for C/C++ libraries at link time.
    libraries : [string]
        A list of library names (not filenames or paths) to link against.
    extra_compile_args : [string]
        Any extra platform- and compiler-specific information to use when
        compiling the source files in 'sources'.  For platforms and compilers
        where "command line" makes sense, this is typically a list of
        command-line arguments, but for other platforms it could be anything.
    extra_link_args : [string]
        Any extra platform- and compiler-specific information to use when
        linking object files together to create the extension (or to create a
        new static Python interpreter).  Similar interpretation as for
        'extra_compile_args'.

    Examples
    ========

    Basic usage:

    >>> from sympy.abc import x, y, z
    >>> from sympy.utilities.autowrap import autowrap
    >>> expr = ((x - y + z)**(13)).expand()
    >>> binary_func = autowrap(expr)
    >>> binary_func(1, 4, 2)
    -1.0

    Using helper functions:

    >>> from sympy.abc import x, t
    >>> from sympy import Function
    >>> helper_func = Function('helper_func')  # Define symbolic function
    >>> expr = 3*x + helper_func(t)  # Main expression using helper function
    >>> # Define helper_func(x) = 4*x using f2py backend
    >>> binary_func = autowrap(expr, args=[x, t],
    ...                       helpers=('helper_func', 4*x, [x]))
    >>> binary_func(2, 5)  # 3*2 + helper_func(5) = 6 + 20
    26.0
    >>> # Same example using cython backend
    >>> binary_func = autowrap(expr, args=[x, t], backend='cython',
    ...                       helpers=[('helper_func', 4*x, [x])])
    >>> binary_func(2, 5)  # 3*2 + helper_func(5) = 6 + 20
    26.0

    Type handling example:

    >>> import numpy as np
    >>> expr = x + y
    >>> f_cython = autowrap(expr, backend='cython')
    >>> f_cython(1, 2)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    TypeError: Argument '_x' has incorrect type (expected numpy.ndarray, got int)
    >>> f_cython(np.array([1.0]), np.array([2.0]))
    array([ 3.])

    """
    if language:
        if not isinstance(language, type):
            _validate_backend_language(backend, language)
    else:
        language = _infer_language(backend)

    # two cases 1) helpers is an iterable of 3-tuples and 2) helpers is a
    # 3-tuple
    if iterable(helpers) and len(helpers) != 0 and iterable(helpers[0]):
        helpers = helpers if helpers else ()
    else:
        helpers = [helpers] if helpers else ()
    args = list(args) if iterable(args, exclude=set) else args

    if code_gen is None:
        code_gen = get_code_generator(language, "autowrap")

    CodeWrapperClass = {
        'F2PY': F2PyCodeWrapper,
        'CYTHON': CythonCodeWrapper,
        'DUMMY': DummyWrapper
    }[backend.upper()]
    code_wrapper = CodeWrapperClass(code_gen, tempdir, flags if flags else (),
                                    verbose, **kwargs)

    helps = []
    for name_h, expr_h, args_h in helpers:
        helps.append(code_gen.routine(name_h, expr_h, args_h))

    for name_h, expr_h, args_h in helpers:
        if expr.has(expr_h):
            name_h = binary_function(name_h, expr_h, backend='dummy')
            expr = expr.subs(expr_h, name_h(*args_h))
    try:
        routine = code_gen.routine('autofunc', expr, args)
    except CodeGenArgumentListError as e:
        # if all missing arguments are for pure output, we simply attach them
        # at the end and try again, because the wrappers will silently convert
        # them to return values anyway.
        new_args = []
        for missing in e.missing_args:
            if not isinstance(missing, OutputArgument):
                raise
            new_args.append(missing.name)
        routine = code_gen.routine('autofunc', expr, args + new_args)

    return code_wrapper.wrap_code(routine, helpers=helps)


@doctest_depends_on(exe=('f2py', 'gfortran'), modules=('numpy',))
def binary_function(symfunc, expr, **kwargs):
    """Returns a SymPy function with expr as binary implementation

    This is a convenience function that automates the steps needed to
    autowrap the SymPy expression and attaching it to a Function object
    with implemented_function().

    Parameters
    ==========

    symfunc : SymPy Function
        The function to bind the callable to.
    expr : SymPy Expression
        The expression used to generate the function.
    kwargs : dict
        Any kwargs accepted by autowrap.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.utilities.autowrap import binary_function
    >>> expr = ((x - y)**(25)).expand()
    >>> f = binary_function('f', expr)
    >>> type(f)
    <class 'sympy.core.function.UndefinedFunction'>
    >>> 2*f(x, y)
    2*f(x, y)
    >>> f(x, y).evalf(2, subs={x: 1, y: 2})
    -1.0

    """
    binary = autowrap(expr, **kwargs)
    return implemented_function(symfunc, binary)

#################################################################
#                           UFUNCIFY                            #
#################################################################

_ufunc_top = Template("""\
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include ${include_file}

static PyMethodDef ${module}Methods[] = {
        {NULL, NULL, 0, NULL}
};""")

_ufunc_outcalls = Template("*((double *)out${outnum}) = ${funcname}(${call_args});")

_ufunc_body = Template("""\
#ifdef NPY_1_19_API_VERSION
static void ${funcname}_ufunc(char **args, const npy_intp *dimensions, const npy_intp* steps, void* data)
#else
static void ${funcname}_ufunc(char **args, npy_intp *dimensions, npy_intp* steps, void* data)
#endif
{
    npy_intp i;
    npy_intp n = dimensions[0];
    ${declare_args}
    ${declare_steps}
    for (i = 0; i < n; i++) {
        ${outcalls}
        ${step_increments}
    }
}
PyUFuncGenericFunction ${funcname}_funcs[1] = {&${funcname}_ufunc};
static char ${funcname}_types[${n_types}] = ${types}
static void *${funcname}_data[1] = {NULL};""")

_ufunc_bottom = Template("""\
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "${module}",
    NULL,
    -1,
    ${module}Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_${module}(void)
{
    PyObject *m, *d;
    ${function_creation}
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ${ufunc_init}
    return m;
}
#else
PyMODINIT_FUNC init${module}(void)
{
    PyObject *m, *d;
    ${function_creation}
    m = Py_InitModule("${module}", ${module}Methods);
    if (m == NULL) {
        return;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ${ufunc_init}
}
#endif\
""")

_ufunc_init_form = Template("""\
ufunc${ind} = PyUFunc_FromFuncAndData(${funcname}_funcs, ${funcname}_data, ${funcname}_types, 1, ${n_in}, ${n_out},
            PyUFunc_None, "${module}", ${docstring}, 0);
    PyDict_SetItemString(d, "${funcname}", ufunc${ind});
    Py_DECREF(ufunc${ind});""")

_ufunc_setup = Template("""\
from setuptools.extension import Extension
from setuptools import setup

from numpy import get_include

if __name__ == "__main__":
    setup(ext_modules=[
        Extension('${module}',
                  sources=['${module}.c', '${filename}.c'],
                  include_dirs=[get_include()])])
""")


class UfuncifyCodeWrapper(CodeWrapper):
    """Wrapper for Ufuncify"""

    def __init__(self, *args, **kwargs):

        ext_keys = ['include_dirs', 'library_dirs', 'libraries',
                    'extra_compile_args', 'extra_link_args']
        msg = ('The compilation option kwarg {} is not supported with the numpy'
               ' backend.')

        for k in ext_keys:
            if k in kwargs.keys():
                warn(msg.format(k))
            kwargs.pop(k, None)

        super().__init__(*args, **kwargs)

    @property
    def command(self):
        command = [sys.executable, "setup.py", "build_ext", "--inplace"]
        return command

    def wrap_code(self, routines, helpers=None):
        # This routine overrides CodeWrapper because we can't assume funcname == routines[0].name
        # Therefore we have to break the CodeWrapper private API.
        # There isn't an obvious way to extend multi-expr support to
        # the other autowrap backends, so we limit this change to ufuncify.
        helpers = helpers if helpers is not None else []
        # We just need a consistent name
        funcname = 'wrapped_' + str(id(routines) + id(helpers))

        workdir = self.filepath or tempfile.mkdtemp("_sympy_compile")
        if not os.access(workdir, os.F_OK):
            os.mkdir(workdir)
        oldwork = os.getcwd()
        os.chdir(workdir)
        try:
            sys.path.append(workdir)
            self._generate_code(routines, helpers)
            self._prepare_files(routines, funcname)
            self._process_files(routines)
            mod = __import__(self.module_name)
        finally:
            sys.path.remove(workdir)
            CodeWrapper._module_counter += 1
            os.chdir(oldwork)
            if not self.filepath:
                try:
                    shutil.rmtree(workdir)
                except OSError:
                    # Could be some issues on Windows
                    pass

        return self._get_wrapped_function(mod, funcname)

    def _generate_code(self, main_routines, helper_routines):
        all_routines = main_routines + helper_routines
        self.generator.write(
            all_routines, self.filename, True, self.include_header,
            self.include_empty)

    def _prepare_files(self, routines, funcname):

        # C
        codefilename = self.module_name + '.c'
        with open(codefilename, 'w') as f:
            self.dump_c(routines, f, self.filename, funcname=funcname)

        # setup.py
        with open('setup.py', 'w') as f:
            self.dump_setup(f)

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name)

    def dump_setup(self, f):
        setup = _ufunc_setup.substitute(module=self.module_name,
                                        filename=self.filename)
        f.write(setup)

    def dump_c(self, routines, f, prefix, funcname=None):
        """Write a C file with Python wrappers

        This file contains all the definitions of the routines in c code.

        Arguments
        ---------
        routines
            List of Routine instances
        f
            File-like object to write the file to
        prefix
            The filename prefix, used to name the imported module.
        funcname
            Name of the main function to be returned.
        """
        if funcname is None:
            if len(routines) == 1:
                funcname = routines[0].name
            else:
                msg = 'funcname must be specified for multiple output routines'
                raise ValueError(msg)
        functions = []
        function_creation = []
        ufunc_init = []
        module = self.module_name
        include_file = "\"{}.h\"".format(prefix)
        top = _ufunc_top.substitute(include_file=include_file, module=module)

        name = funcname

        # Partition the C function arguments into categories
        # Here we assume all routines accept the same arguments
        r_index = 0
        py_in, _ = self._partition_args(routines[0].arguments)
        n_in = len(py_in)
        n_out = len(routines)

        # Declare Args
        form = "char *{0}{1} = args[{2}];"
        arg_decs = [form.format('in', i, i) for i in range(n_in)]
        arg_decs.extend([form.format('out', i, i+n_in) for i in range(n_out)])
        declare_args = '\n    '.join(arg_decs)

        # Declare Steps
        form = "npy_intp {0}{1}_step = steps[{2}];"
        step_decs = [form.format('in', i, i) for i in range(n_in)]
        step_decs.extend([form.format('out', i, i+n_in) for i in range(n_out)])
        declare_steps = '\n    '.join(step_decs)

        # Call Args
        form = "*(double *)in{0}"
        call_args = ', '.join([form.format(a) for a in range(n_in)])

        # Step Increments
        form = "{0}{1} += {0}{1}_step;"
        step_incs = [form.format('in', i) for i in range(n_in)]
        step_incs.extend([form.format('out', i, i) for i in range(n_out)])
        step_increments = '\n        '.join(step_incs)

        # Types
        n_types = n_in + n_out
        types = "{" + ', '.join(["NPY_DOUBLE"]*n_types) + "};"

        # Docstring
        docstring = '"Created in SymPy with Ufuncify"'

        # Function Creation
        function_creation.append("PyObject *ufunc{};".format(r_index))

        # Ufunc initialization
        init_form = _ufunc_init_form.substitute(module=module,
                                                funcname=name,
                                                docstring=docstring,
                                                n_in=n_in, n_out=n_out,
                                                ind=r_index)
        ufunc_init.append(init_form)

        outcalls = [_ufunc_outcalls.substitute(
            outnum=i, call_args=call_args, funcname=routines[i].name) for i in
            range(n_out)]

        body = _ufunc_body.substitute(module=module, funcname=name,
                                      declare_args=declare_args,
                                      declare_steps=declare_steps,
                                      call_args=call_args,
                                      step_increments=step_increments,
                                      n_types=n_types, types=types,
                                      outcalls='\n        '.join(outcalls))
        functions.append(body)

        body = '\n\n'.join(functions)
        ufunc_init = '\n    '.join(ufunc_init)
        function_creation = '\n    '.join(function_creation)
        bottom = _ufunc_bottom.substitute(module=module,
                                          ufunc_init=ufunc_init,
                                          function_creation=function_creation)
        text = [top, body, bottom]
        f.write('\n\n'.join(text))

    def _partition_args(self, args):
        """Group function arguments into categories."""
        py_in = []
        py_out = []
        for arg in args:
            if isinstance(arg, OutputArgument):
                py_out.append(arg)
            elif isinstance(arg, InOutArgument):
                raise ValueError("Ufuncify doesn't support InOutArguments")
            else:
                py_in.append(arg)
        return py_in, py_out


@cacheit
@doctest_depends_on(exe=('f2py', 'gfortran', 'gcc'), modules=('numpy',))
def ufuncify(args, expr, language=None, backend='numpy', tempdir=None,
             flags=None, verbose=False, helpers=None, **kwargs):
    """Generates a binary function that supports broadcasting on numpy arrays.

    Parameters
    ==========

    args : iterable
        Either a Symbol or an iterable of symbols. Specifies the argument
        sequence for the function.
    expr
        A SymPy expression that defines the element wise operation.
    language : string, optional
        If supplied, (options: 'C' or 'F95'), specifies the language of the
        generated code. If ``None`` [default], the language is inferred based
        upon the specified backend.
    backend : string, optional
        Backend used to wrap the generated code. Either 'numpy' [default],
        'cython', or 'f2py'.
    tempdir : string, optional
        Path to directory for temporary files. If this argument is supplied,
        the generated code and the wrapper input files are left intact in
        the specified path.
    flags : iterable, optional
        Additional option flags that will be passed to the backend.
    verbose : bool, optional
        If True, autowrap will not mute the command line backends. This can
        be helpful for debugging.
    helpers : 3-tuple or iterable of 3-tuples, optional
        Used to define auxiliary functions needed for the main expression.
        Each tuple should be of the form (name, expr, args) where:

        - name : str, the function name
        - expr : sympy expression, the function
        - args : iterable, the function arguments (can be any iterable of symbols)

    kwargs : dict
        These kwargs will be passed to autowrap if the `f2py` or `cython`
        backend is used and ignored if the `numpy` backend is used.

    Notes
    =====

    The default backend ('numpy') will create actual instances of
    ``numpy.ufunc``. These support ndimensional broadcasting, and implicit type
    conversion. Use of the other backends will result in a "ufunc-like"
    function, which requires equal length 1-dimensional arrays for all
    arguments, and will not perform any type conversions.

    References
    ==========

    .. [1] https://numpy.org/doc/stable/reference/ufuncs.html

    Examples
    ========

    Basic usage:

    >>> from sympy.utilities.autowrap import ufuncify
    >>> from sympy.abc import x, y
    >>> import numpy as np
    >>> f = ufuncify((x, y), y + x**2)
    >>> type(f)
    <class 'numpy.ufunc'>
    >>> f([1, 2, 3], 2)
    array([  3.,   6.,  11.])
    >>> f(np.arange(5), 3)
    array([  3.,   4.,   7.,  12.,  19.])

    Using helper functions:

    >>> from sympy import Function
    >>> helper_func = Function('helper_func')  # Define symbolic function
    >>> expr = x**2 + y*helper_func(x)  # Main expression using helper function
    >>> # Define helper_func(x) = x**3
    >>> f = ufuncify((x, y), expr, helpers=[('helper_func', x**3, [x])])
    >>> f([1, 2], [3, 4])
    array([  4.,  36.])

    Type handling with different backends:

    For the 'f2py' and 'cython' backends, inputs are required to be equal length
    1-dimensional arrays. The 'f2py' backend will perform type conversion, but
    the Cython backend will error if the inputs are not of the expected type.

    >>> f_fortran = ufuncify((x, y), y + x**2, backend='f2py')
    >>> f_fortran(1, 2)
    array([ 3.])
    >>> f_fortran(np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]))
    array([  2.,   6.,  12.])
    >>> f_cython = ufuncify((x, y), y + x**2, backend='Cython')
    >>> f_cython(1, 2)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    TypeError: Argument '_x' has incorrect type (expected numpy.ndarray, got int)
    >>> f_cython(np.array([1.0]), np.array([2.0]))
    array([ 3.])

    """

    if isinstance(args, Symbol):
        args = (args,)
    else:
        args = tuple(args)

    if language:
        _validate_backend_language(backend, language)
    else:
        language = _infer_language(backend)

    helpers = helpers if helpers else ()
    flags = flags if flags else ()

    if backend.upper() == 'NUMPY':
        # maxargs is set by numpy compile-time constant NPY_MAXARGS
        # If a future version of numpy modifies or removes this restriction
        # this variable should be changed or removed
        maxargs = 32
        helps = []
        for name, expr, args in helpers:
            helps.append(make_routine(name, expr, args))
        code_wrapper = UfuncifyCodeWrapper(C99CodeGen("ufuncify"), tempdir,
                                           flags, verbose)
        if not isinstance(expr, (list, tuple)):
            expr = [expr]
        if len(expr) == 0:
            raise ValueError('Expression iterable has zero length')
        if len(expr) + len(args) > maxargs:
            msg = ('Cannot create ufunc with more than {0} total arguments: '
                   'got {1} in, {2} out')
            raise ValueError(msg.format(maxargs, len(args), len(expr)))
        routines = [make_routine('autofunc{}'.format(idx), exprx, args) for
                    idx, exprx in enumerate(expr)]
        return code_wrapper.wrap_code(routines, helpers=helps)
    else:
        # Dummies are used for all added expressions to prevent name clashes
        # within the original expression.
        y = IndexedBase(Dummy('y'))
        m = Dummy('m', integer=True)
        i = Idx(Dummy('i', integer=True), m)
        f_dummy = Dummy('f')
        f = implemented_function('%s_%d' % (f_dummy.name, f_dummy.dummy_index), Lambda(args, expr))
        # For each of the args create an indexed version.
        indexed_args = [IndexedBase(Dummy(str(a))) for a in args]
        # Order the arguments (out, args, dim)
        args = [y] + indexed_args + [m]
        args_with_indices = [a[i] for a in indexed_args]
        return autowrap(Eq(y[i], f(*args_with_indices)), language, backend,
                        tempdir, args, flags, verbose, helpers, **kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\connectionpool.py ===
from __future__ import annotations

import errno
import logging
import queue
import sys
import typing
import warnings
import weakref
from socket import timeout as SocketTimeout
from types import TracebackType

from ._base_connection import _TYPE_BODY
from ._collections import HTTPHeaderDict
from ._request_methods import RequestMethods
from .connection import (
    BaseSSLError,
    BrokenPipeError,
    DummyConnection,
    HTTPConnection,
    HTTPException,
    HTTPSConnection,
    ProxyConfig,
    _wrap_proxy_error,
)
from .connection import port_by_scheme as port_by_scheme
from .exceptions import (
    ClosedPoolError,
    EmptyPoolError,
    FullPoolError,
    HostChangedError,
    InsecureRequestWarning,
    LocationValueError,
    MaxRetryError,
    NewConnectionError,
    ProtocolError,
    ProxyError,
    ReadTimeoutError,
    SSLError,
    TimeoutError,
)
from .response import BaseHTTPResponse
from .util.connection import is_connection_dropped
from .util.proxy import connection_requires_http_tunnel
from .util.request import _TYPE_BODY_POSITION, set_file_position
from .util.retry import Retry
from .util.ssl_match_hostname import CertificateError
from .util.timeout import _DEFAULT_TIMEOUT, _TYPE_DEFAULT, Timeout
from .util.url import Url, _encode_target
from .util.url import _normalize_host as normalize_host
from .util.url import parse_url
from .util.util import to_str

if typing.TYPE_CHECKING:
    import ssl

    from typing_extensions import Self

    from ._base_connection import BaseHTTPConnection, BaseHTTPSConnection

log = logging.getLogger(__name__)

_TYPE_TIMEOUT = typing.Union[Timeout, float, _TYPE_DEFAULT, None]


# Pool objects
class ConnectionPool:
    """
    Base class for all connection pools, such as
    :class:`.HTTPConnectionPool` and :class:`.HTTPSConnectionPool`.

    .. note::
       ConnectionPool.urlopen() does not normalize or percent-encode target URIs
       which is useful if your target server doesn't support percent-encoded
       target URIs.
    """

    scheme: str | None = None
    QueueCls = queue.LifoQueue

    def __init__(self, host: str, port: int | None = None) -> None:
        if not host:
            raise LocationValueError("No host specified.")

        self.host = _normalize_host(host, scheme=self.scheme)
        self.port = port

        # This property uses 'normalize_host()' (not '_normalize_host()')
        # to avoid removing square braces around IPv6 addresses.
        # This value is sent to `HTTPConnection.set_tunnel()` if called
        # because square braces are required for HTTP CONNECT tunneling.
        self._tunnel_host = normalize_host(host, scheme=self.scheme).lower()

    def __str__(self) -> str:
        return f"{type(self).__name__}(host={self.host!r}, port={self.port!r})"

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> typing.Literal[False]:
        self.close()
        # Return False to re-raise any potential exceptions
        return False

    def close(self) -> None:
        """
        Close all pooled connections and disable the pool.
        """


# This is taken from http://hg.python.org/cpython/file/7aaba721ebc0/Lib/socket.py#l252
_blocking_errnos = {errno.EAGAIN, errno.EWOULDBLOCK}


class HTTPConnectionPool(ConnectionPool, RequestMethods):
    """
    Thread-safe connection pool for one host.

    :param host:
        Host used for this HTTP Connection (e.g. "localhost"), passed into
        :class:`http.client.HTTPConnection`.

    :param port:
        Port used for this HTTP Connection (None is equivalent to 80), passed
        into :class:`http.client.HTTPConnection`.

    :param timeout:
        Socket timeout in seconds for each individual connection. This can
        be a float or integer, which sets the timeout for the HTTP request,
        or an instance of :class:`urllib3.util.Timeout` which gives you more
        fine-grained control over request timeouts. After the constructor has
        been parsed, this is always a `urllib3.util.Timeout` object.

    :param maxsize:
        Number of connections to save that can be reused. More than 1 is useful
        in multithreaded situations. If ``block`` is set to False, more
        connections will be created but they will not be saved once they've
        been used.

    :param block:
        If set to True, no more than ``maxsize`` connections will be used at
        a time. When no free connections are available, the call will block
        until a connection has been released. This is a useful side effect for
        particular multithreaded situations where one does not want to use more
        than maxsize connections per host to prevent flooding.

    :param headers:
        Headers to include with all requests, unless other headers are given
        explicitly.

    :param retries:
        Retry configuration to use by default with requests in this pool.

    :param _proxy:
        Parsed proxy URL, should not be used directly, instead, see
        :class:`urllib3.ProxyManager`

    :param _proxy_headers:
        A dictionary with proxy headers, should not be used directly,
        instead, see :class:`urllib3.ProxyManager`

    :param \\**conn_kw:
        Additional parameters are used to create fresh :class:`urllib3.connection.HTTPConnection`,
        :class:`urllib3.connection.HTTPSConnection` instances.
    """

    scheme = "http"
    ConnectionCls: type[BaseHTTPConnection] | type[BaseHTTPSConnection] = HTTPConnection

    def __init__(
        self,
        host: str,
        port: int | None = None,
        timeout: _TYPE_TIMEOUT | None = _DEFAULT_TIMEOUT,
        maxsize: int = 1,
        block: bool = False,
        headers: typing.Mapping[str, str] | None = None,
        retries: Retry | bool | int | None = None,
        _proxy: Url | None = None,
        _proxy_headers: typing.Mapping[str, str] | None = None,
        _proxy_config: ProxyConfig | None = None,
        **conn_kw: typing.Any,
    ):
        ConnectionPool.__init__(self, host, port)
        RequestMethods.__init__(self, headers)

        if not isinstance(timeout, Timeout):
            timeout = Timeout.from_float(timeout)

        if retries is None:
            retries = Retry.DEFAULT

        self.timeout = timeout
        self.retries = retries

        self.pool: queue.LifoQueue[typing.Any] | None = self.QueueCls(maxsize)
        self.block = block

        self.proxy = _proxy
        self.proxy_headers = _proxy_headers or {}
        self.proxy_config = _proxy_config

        # Fill the queue up so that doing get() on it will block properly
        for _ in range(maxsize):
            self.pool.put(None)

        # These are mostly for testing and debugging purposes.
        self.num_connections = 0
        self.num_requests = 0
        self.conn_kw = conn_kw

        if self.proxy:
            # Enable Nagle's algorithm for proxies, to avoid packet fragmentation.
            # We cannot know if the user has added default socket options, so we cannot replace the
            # list.
            self.conn_kw.setdefault("socket_options", [])

            self.conn_kw["proxy"] = self.proxy
            self.conn_kw["proxy_config"] = self.proxy_config

        # Do not pass 'self' as callback to 'finalize'.
        # Then the 'finalize' would keep an endless living (leak) to self.
        # By just passing a reference to the pool allows the garbage collector
        # to free self if nobody else has a reference to it.
        pool = self.pool

        # Close all the HTTPConnections in the pool before the
        # HTTPConnectionPool object is garbage collected.
        weakref.finalize(self, _close_pool_connections, pool)

    def _new_conn(self) -> BaseHTTPConnection:
        """
        Return a fresh :class:`HTTPConnection`.
        """
        self.num_connections += 1
        log.debug(
            "Starting new HTTP connection (%d): %s:%s",
            self.num_connections,
            self.host,
            self.port or "80",
        )

        conn = self.ConnectionCls(
            host=self.host,
            port=self.port,
            timeout=self.timeout.connect_timeout,
            **self.conn_kw,
        )
        return conn

    def _get_conn(self, timeout: float | None = None) -> BaseHTTPConnection:
        """
        Get a connection. Will return a pooled connection if one is available.

        If no connections are available and :prop:`.block` is ``False``, then a
        fresh connection is returned.

        :param timeout:
            Seconds to wait before giving up and raising
            :class:`urllib3.exceptions.EmptyPoolError` if the pool is empty and
            :prop:`.block` is ``True``.
        """
        conn = None

        if self.pool is None:
            raise ClosedPoolError(self, "Pool is closed.")

        try:
            conn = self.pool.get(block=self.block, timeout=timeout)

        except AttributeError:  # self.pool is None
            raise ClosedPoolError(self, "Pool is closed.") from None  # Defensive:

        except queue.Empty:
            if self.block:
                raise EmptyPoolError(
                    self,
                    "Pool is empty and a new connection can't be opened due to blocking mode.",
                ) from None
            pass  # Oh well, we'll create a new connection then

        # If this is a persistent connection, check if it got disconnected
        if conn and is_connection_dropped(conn):
            log.debug("Resetting dropped connection: %s", self.host)
            conn.close()

        return conn or self._new_conn()

    def _put_conn(self, conn: BaseHTTPConnection | None) -> None:
        """
        Put a connection back into the pool.

        :param conn:
            Connection object for the current host and port as returned by
            :meth:`._new_conn` or :meth:`._get_conn`.

        If the pool is already full, the connection is closed and discarded
        because we exceeded maxsize. If connections are discarded frequently,
        then maxsize should be increased.

        If the pool is closed, then the connection will be closed and discarded.
        """
        if self.pool is not None:
            try:
                self.pool.put(conn, block=False)
                return  # Everything is dandy, done.
            except AttributeError:
                # self.pool is None.
                pass
            except queue.Full:
                # Connection never got put back into the pool, close it.
                if conn:
                    conn.close()

                if self.block:
                    # This should never happen if you got the conn from self._get_conn
                    raise FullPoolError(
                        self,
                        "Pool reached maximum size and no more connections are allowed.",
                    ) from None

                log.warning(
                    "Connection pool is full, discarding connection: %s. Connection pool size: %s",
                    self.host,
                    self.pool.qsize(),
                )

        # Connection never got put back into the pool, close it.
        if conn:
            conn.close()

    def _validate_conn(self, conn: BaseHTTPConnection) -> None:
        """
        Called right before a request is made, after the socket is created.
        """

    def _prepare_proxy(self, conn: BaseHTTPConnection) -> None:
        # Nothing to do for HTTP connections.
        pass

    def _get_timeout(self, timeout: _TYPE_TIMEOUT) -> Timeout:
        """Helper that always returns a :class:`urllib3.util.Timeout`"""
        if timeout is _DEFAULT_TIMEOUT:
            return self.timeout.clone()

        if isinstance(timeout, Timeout):
            return timeout.clone()
        else:
            # User passed us an int/float. This is for backwards compatibility,
            # can be removed later
            return Timeout.from_float(timeout)

    def _raise_timeout(
        self,
        err: BaseSSLError | OSError | SocketTimeout,
        url: str,
        timeout_value: _TYPE_TIMEOUT | None,
    ) -> None:
        """Is the error actually a timeout? Will raise a ReadTimeout or pass"""

        if isinstance(err, SocketTimeout):
            raise ReadTimeoutError(
                self, url, f"Read timed out. (read timeout={timeout_value})"
            ) from err

        # See the above comment about EAGAIN in Python 3.
        if hasattr(err, "errno") and err.errno in _blocking_errnos:
            raise ReadTimeoutError(
                self, url, f"Read timed out. (read timeout={timeout_value})"
            ) from err

    def _make_request(
        self,
        conn: BaseHTTPConnection,
        method: str,
        url: str,
        body: _TYPE_BODY | None = None,
        headers: typing.Mapping[str, str] | None = None,
        retries: Retry | None = None,
        timeout: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
        chunked: bool = False,
        response_conn: BaseHTTPConnection | None = None,
        preload_content: bool = True,
        decode_content: bool = True,
        enforce_content_length: bool = True,
    ) -> BaseHTTPResponse:
        """
        Perform a request on a given urllib connection object taken from our
        pool.

        :param conn:
            a connection from one of our connection pools

        :param method:
            HTTP request method (such as GET, POST, PUT, etc.)

        :param url:
            The URL to perform the request on.

        :param body:
            Data to send in the request body, either :class:`str`, :class:`bytes`,
            an iterable of :class:`str`/:class:`bytes`, or a file-like object.

        :param headers:
            Dictionary of custom headers to send, such as User-Agent,
            If-None-Match, etc. If None, pool headers are used. If provided,
            these headers completely replace any pool-specific headers.

        :param retries:
            Configure the number of retries to allow before raising a
            :class:`~urllib3.exceptions.MaxRetryError` exception.

            Pass ``None`` to retry until you receive a response. Pass a
            :class:`~urllib3.util.retry.Retry` object for fine-grained control
            over different types of retries.
            Pass an integer number to retry connection errors that many times,
            but no other types of errors. Pass zero to never retry.

            If ``False``, then retries are disabled and any exception is raised
            immediately. Also, instead of raising a MaxRetryError on redirects,
            the redirect response will be returned.

        :type retries: :class:`~urllib3.util.retry.Retry`, False, or an int.

        :param timeout:
            If specified, overrides the default timeout for this one
            request. It may be a float (in seconds) or an instance of
            :class:`urllib3.util.Timeout`.

        :param chunked:
            If True, urllib3 will send the body using chunked transfer
            encoding. Otherwise, urllib3 will send the body using the standard
            content-length form. Defaults to False.

        :param response_conn:
            Set this to ``None`` if you will handle releasing the connection or
            set the connection to have the response release it.

        :param preload_content:
          If True, the response's body will be preloaded during construction.

        :param decode_content:
            If True, will attempt to decode the body based on the
            'content-encoding' header.

        :param enforce_content_length:
            Enforce content length checking. Body returned by server must match
            value of Content-Length header, if present. Otherwise, raise error.
        """
        self.num_requests += 1

        timeout_obj = self._get_timeout(timeout)
        timeout_obj.start_connect()
        conn.timeout = Timeout.resolve_default_timeout(timeout_obj.connect_timeout)

        try:
            # Trigger any extra validation we need to do.
            try:
                self._validate_conn(conn)
            except (SocketTimeout, BaseSSLError) as e:
                self._raise_timeout(err=e, url=url, timeout_value=conn.timeout)
                raise

        # _validate_conn() starts the connection to an HTTPS proxy
        # so we need to wrap errors with 'ProxyError' here too.
        except (
            OSError,
            NewConnectionError,
            TimeoutError,
            BaseSSLError,
            CertificateError,
            SSLError,
        ) as e:
            new_e: Exception = e
            if isinstance(e, (BaseSSLError, CertificateError)):
                new_e = SSLError(e)
            # If the connection didn't successfully connect to it's proxy
            # then there
            if isinstance(
                new_e, (OSError, NewConnectionError, TimeoutError, SSLError)
            ) and (conn and conn.proxy and not conn.has_connected_to_proxy):
                new_e = _wrap_proxy_error(new_e, conn.proxy.scheme)
            raise new_e

        # conn.request() calls http.client.*.request, not the method in
        # urllib3.request. It also calls makefile (recv) on the socket.
        try:
            conn.request(
                method,
                url,
                body=body,
                headers=headers,
                chunked=chunked,
                preload_content=preload_content,
                decode_content=decode_content,
                enforce_content_length=enforce_content_length,
            )

        # We are swallowing BrokenPipeError (errno.EPIPE) since the server is
        # legitimately able to close the connection after sending a valid response.
        # With this behaviour, the received response is still readable.
        except BrokenPipeError:
            pass
        except OSError as e:
            # MacOS/Linux
            # EPROTOTYPE and ECONNRESET are needed on macOS
            # https://erickt.github.io/blog/2014/11/19/adventures-in-debugging-a-potential-osx-kernel-bug/
            # Condition changed later to emit ECONNRESET instead of only EPROTOTYPE.
            if e.errno != errno.EPROTOTYPE and e.errno != errno.ECONNRESET:
                raise

        # Reset the timeout for the recv() on the socket
        read_timeout = timeout_obj.read_timeout

        if not conn.is_closed:
            # In Python 3 socket.py will catch EAGAIN and return None when you
            # try and read into the file pointer created by http.client, which
            # instead raises a BadStatusLine exception. Instead of catching
            # the exception and assuming all BadStatusLine exceptions are read
            # timeouts, check for a zero timeout before making the request.
            if read_timeout == 0:
                raise ReadTimeoutError(
                    self, url, f"Read timed out. (read timeout={read_timeout})"
                )
            conn.timeout = read_timeout

        # Receive the response from the server
        try:
            response = conn.getresponse()
        except (BaseSSLError, OSError) as e:
            self._raise_timeout(err=e, url=url, timeout_value=read_timeout)
            raise

        # Set properties that are used by the pooling layer.
        response.retries = retries
        response._connection = response_conn  # type: ignore[attr-defined]
        response._pool = self  # type: ignore[attr-defined]

        log.debug(
            '%s://%s:%s "%s %s %s" %s %s',
            self.scheme,
            self.host,
            self.port,
            method,
            url,
            response.version_string,
            response.status,
            response.length_remaining,
        )

        return response

    def close(self) -> None:
        """
        Close all pooled connections and disable the pool.
        """
        if self.pool is None:
            return
        # Disable access to the pool
        old_pool, self.pool = self.pool, None

        # Close all the HTTPConnections in the pool.
        _close_pool_connections(old_pool)

    def is_same_host(self, url: str) -> bool:
        """
        Check if the given ``url`` is a member of the same host as this
        connection pool.
        """
        if url.startswith("/"):
            return True

        # TODO: Add optional support for socket.gethostbyname checking.
        scheme, _, host, port, *_ = parse_url(url)
        scheme = scheme or "http"
        if host is not None:
            host = _normalize_host(host, scheme=scheme)

        # Use explicit default port for comparison when none is given
        if self.port and not port:
            port = port_by_scheme.get(scheme)
        elif not self.port and port == port_by_scheme.get(scheme):
            port = None

        return (scheme, host, port) == (self.scheme, self.host, self.port)

    def urlopen(  # type: ignore[override]
        self,
        method: str,
        url: str,
        body: _TYPE_BODY | None = None,
        headers: typing.Mapping[str, str] | None = None,
        retries: Retry | bool | int | None = None,
        redirect: bool = True,
        assert_same_host: bool = True,
        timeout: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
        pool_timeout: int | None = None,
        release_conn: bool | None = None,
        chunked: bool = False,
        body_pos: _TYPE_BODY_POSITION | None = None,
        preload_content: bool = True,
        decode_content: bool = True,
        **response_kw: typing.Any,
    ) -> BaseHTTPResponse:
        """
        Get a connection from the pool and perform an HTTP request. This is the
        lowest level call for making a request, so you'll need to specify all
        the raw details.

        .. note::

           More commonly, it's appropriate to use a convenience method
           such as :meth:`request`.

        .. note::

           `release_conn` will only behave as expected if
           `preload_content=False` because we want to make
           `preload_content=False` the default behaviour someday soon without
           breaking backwards compatibility.

        :param method:
            HTTP request method (such as GET, POST, PUT, etc.)

        :param url:
            The URL to perform the request on.

        :param body:
            Data to send in the request body, either :class:`str`, :class:`bytes`,
            an iterable of :class:`str`/:class:`bytes`, or a file-like object.

        :param headers:
            Dictionary of custom headers to send, such as User-Agent,
            If-None-Match, etc. If None, pool headers are used. If provided,
            these headers completely replace any pool-specific headers.

        :param retries:
            Configure the number of retries to allow before raising a
            :class:`~urllib3.exceptions.MaxRetryError` exception.

            If ``None`` (default) will retry 3 times, see ``Retry.DEFAULT``. Pass a
            :class:`~urllib3.util.retry.Retry` object for fine-grained control
            over different types of retries.
            Pass an integer number to retry connection errors that many times,
            but no other types of errors. Pass zero to never retry.

            If ``False``, then retries are disabled and any exception is raised
            immediately. Also, instead of raising a MaxRetryError on redirects,
            the redirect response will be returned.

        :type retries: :class:`~urllib3.util.retry.Retry`, False, or an int.

        :param redirect:
            If True, automatically handle redirects (status codes 301, 302,
            303, 307, 308). Each redirect counts as a retry. Disabling retries
            will disable redirect, too.

        :param assert_same_host:
            If ``True``, will make sure that the host of the pool requests is
            consistent else will raise HostChangedError. When ``False``, you can
            use the pool on an HTTP proxy and request foreign hosts.

        :param timeout:
            If specified, overrides the default timeout for this one
            request. It may be a float (in seconds) or an instance of
            :class:`urllib3.util.Timeout`.

        :param pool_timeout:
            If set and the pool is set to block=True, then this method will
            block for ``pool_timeout`` seconds and raise EmptyPoolError if no
            connection is available within the time period.

        :param bool preload_content:
            If True, the response's body will be preloaded into memory.

        :param bool decode_content:
            If True, will attempt to decode the body based on the
            'content-encoding' header.

        :param release_conn:
            If False, then the urlopen call will not release the connection
            back into the pool once a response is received (but will release if
            you read the entire contents of the response such as when
            `preload_content=True`). This is useful if you're not preloading
            the response's content immediately. You will need to call
            ``r.release_conn()`` on the response ``r`` to return the connection
            back into the pool. If None, it takes the value of ``preload_content``
            which defaults to ``True``.

        :param bool chunked:
            If True, urllib3 will send the body using chunked transfer
            encoding. Otherwise, urllib3 will send the body using the standard
            content-length form. Defaults to False.

        :param int body_pos:
            Position to seek to in file-like body in the event of a retry or
            redirect. Typically this won't need to be set because urllib3 will
            auto-populate the value when needed.
        """
        parsed_url = parse_url(url)
        destination_scheme = parsed_url.scheme

        if headers is None:
            headers = self.headers

        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, redirect=redirect, default=self.retries)

        if release_conn is None:
            release_conn = preload_content

        # Check host
        if assert_same_host and not self.is_same_host(url):
            raise HostChangedError(self, url, retries)

        # Ensure that the URL we're connecting to is properly encoded
        if url.startswith("/"):
            url = to_str(_encode_target(url))
        else:
            url = to_str(parsed_url.url)

        conn = None

        # Track whether `conn` needs to be released before
        # returning/raising/recursing. Update this variable if necessary, and
        # leave `release_conn` constant throughout the function. That way, if
        # the function recurses, the original value of `release_conn` will be
        # passed down into the recursive call, and its value will be respected.
        #
        # See issue #651 [1] for details.
        #
        # [1] <https://github.com/urllib3/urllib3/issues/651>
        release_this_conn = release_conn

        http_tunnel_required = connection_requires_http_tunnel(
            self.proxy, self.proxy_config, destination_scheme
        )

        # Merge the proxy headers. Only done when not using HTTP CONNECT. We
        # have to copy the headers dict so we can safely change it without those
        # changes being reflected in anyone else's copy.
        if not http_tunnel_required:
            headers = headers.copy()  # type: ignore[attr-defined]
            headers.update(self.proxy_headers)  # type: ignore[union-attr]

        # Must keep the exception bound to a separate variable or else Python 3
        # complains about UnboundLocalError.
        err = None

        # Keep track of whether we cleanly exited the except block. This
        # ensures we do proper cleanup in finally.
        clean_exit = False

        # Rewind body position, if needed. Record current position
        # for future rewinds in the event of a redirect/retry.
        body_pos = set_file_position(body, body_pos)

        try:
            # Request a connection from the queue.
            timeout_obj = self._get_timeout(timeout)
            conn = self._get_conn(timeout=pool_timeout)

            conn.timeout = timeout_obj.connect_timeout  # type: ignore[assignment]

            # Is this a closed/new connection that requires CONNECT tunnelling?
            if self.proxy is not None and http_tunnel_required and conn.is_closed:
                try:
                    self._prepare_proxy(conn)
                except (BaseSSLError, OSError, SocketTimeout) as e:
                    self._raise_timeout(
                        err=e, url=self.proxy.url, timeout_value=conn.timeout
                    )
                    raise

            # If we're going to release the connection in ``finally:``, then
            # the response doesn't need to know about the connection. Otherwise
            # it will also try to release it and we'll have a double-release
            # mess.
            response_conn = conn if not release_conn else None

            # Make the request on the HTTPConnection object
            response = self._make_request(
                conn,
                method,
                url,
                timeout=timeout_obj,
                body=body,
                headers=headers,
                chunked=chunked,
                retries=retries,
                response_conn=response_conn,
                preload_content=preload_content,
                decode_content=decode_content,
                **response_kw,
            )

            # Everything went great!
            clean_exit = True

        except EmptyPoolError:
            # Didn't get a connection from the pool, no need to clean up
            clean_exit = True
            release_this_conn = False
            raise

        except (
            TimeoutError,
            HTTPException,
            OSError,
            ProtocolError,
            BaseSSLError,
            SSLError,
            CertificateError,
            ProxyError,
        ) as e:
            # Discard the connection for these exceptions. It will be
            # replaced during the next _get_conn() call.
            clean_exit = False
            new_e: Exception = e
            if isinstance(e, (BaseSSLError, CertificateError)):
                new_e = SSLError(e)
            if isinstance(
                new_e,
                (
                    OSError,
                    NewConnectionError,
                    TimeoutError,
                    SSLError,
                    HTTPException,
                ),
            ) and (conn and conn.proxy and not conn.has_connected_to_proxy):
                new_e = _wrap_proxy_error(new_e, conn.proxy.scheme)
            elif isinstance(new_e, (OSError, HTTPException)):
                new_e = ProtocolError("Connection aborted.", new_e)

            retries = retries.increment(
                method, url, error=new_e, _pool=self, _stacktrace=sys.exc_info()[2]
            )
            retries.sleep()

            # Keep track of the error for the retry warning.
            err = e

        finally:
            if not clean_exit:
                # We hit some kind of exception, handled or otherwise. We need
                # to throw the connection away unless explicitly told not to.
                # Close the connection, set the variable to None, and make sure
                # we put the None back in the pool to avoid leaking it.
                if conn:
                    conn.close()
                    conn = None
                release_this_conn = True

            if release_this_conn:
                # Put the connection back to be reused. If the connection is
                # expired then it will be None, which will get replaced with a
                # fresh connection during _get_conn.
                self._put_conn(conn)

        if not conn:
            # Try again
            log.warning(
                "Retrying (%r) after connection broken by '%r': %s", retries, err, url
            )
            return self.urlopen(
                method,
                url,
                body,
                headers,
                retries,
                redirect,
                assert_same_host,
                timeout=timeout,
                pool_timeout=pool_timeout,
                release_conn=release_conn,
                chunked=chunked,
                body_pos=body_pos,
                preload_content=preload_content,
                decode_content=decode_content,
                **response_kw,
            )

        # Handle redirect?
        redirect_location = redirect and response.get_redirect_location()
        if redirect_location:
            if response.status == 303:
                # Change the method according to RFC 9110, Section 15.4.4.
                method = "GET"
                # And lose the body not to transfer anything sensitive.
                body = None
                headers = HTTPHeaderDict(headers)._prepare_for_method_change()

            try:
                retries = retries.increment(method, url, response=response, _pool=self)
            except MaxRetryError:
                if retries.raise_on_redirect:
                    response.drain_conn()
                    raise
                return response

            response.drain_conn()
            retries.sleep_for_retry(response)
            log.debug("Redirecting %s -> %s", url, redirect_location)
            return self.urlopen(
                method,
                redirect_location,
                body,
                headers,
                retries=retries,
                redirect=redirect,
                assert_same_host=assert_same_host,
                timeout=timeout,
                pool_timeout=pool_timeout,
                release_conn=release_conn,
                chunked=chunked,
                body_pos=body_pos,
                preload_content=preload_content,
                decode_content=decode_content,
                **response_kw,
            )

        # Check if we should retry the HTTP response.
        has_retry_after = bool(response.headers.get("Retry-After"))
        if retries.is_retry(method, response.status, has_retry_after):
            try:
                retries = retries.increment(method, url, response=response, _pool=self)
            except MaxRetryError:
                if retries.raise_on_status:
                    response.drain_conn()
                    raise
                return response

            response.drain_conn()
            retries.sleep(response)
            log.debug("Retry: %s", url)
            return self.urlopen(
                method,
                url,
                body,
                headers,
                retries=retries,
                redirect=redirect,
                assert_same_host=assert_same_host,
                timeout=timeout,
                pool_timeout=pool_timeout,
                release_conn=release_conn,
                chunked=chunked,
                body_pos=body_pos,
                preload_content=preload_content,
                decode_content=decode_content,
                **response_kw,
            )

        return response


class HTTPSConnectionPool(HTTPConnectionPool):
    """
    Same as :class:`.HTTPConnectionPool`, but HTTPS.

    :class:`.HTTPSConnection` uses one of ``assert_fingerprint``,
    ``assert_hostname`` and ``host`` in this order to verify connections.
    If ``assert_hostname`` is False, no verification is done.

    The ``key_file``, ``cert_file``, ``cert_reqs``, ``ca_certs``,
    ``ca_cert_dir``, ``ssl_version``, ``key_password`` are only used if :mod:`ssl`
    is available and are fed into :meth:`urllib3.util.ssl_wrap_socket` to upgrade
    the connection socket into an SSL socket.
    """

    scheme = "https"
    ConnectionCls: type[BaseHTTPSConnection] = HTTPSConnection

    def __init__(
        self,
        host: str,
        port: int | None = None,
        timeout: _TYPE_TIMEOUT | None = _DEFAULT_TIMEOUT,
        maxsize: int = 1,
        block: bool = False,
        headers: typing.Mapping[str, str] | None = None,
        retries: Retry | bool | int | None = None,
        _proxy: Url | None = None,
        _proxy_headers: typing.Mapping[str, str] | None = None,
        key_file: str | None = None,
        cert_file: str | None = None,
        cert_reqs: int | str | None = None,
        key_password: str | None = None,
        ca_certs: str | None = None,
        ssl_version: int | str | None = None,
        ssl_minimum_version: ssl.TLSVersion | None = None,
        ssl_maximum_version: ssl.TLSVersion | None = None,
        assert_hostname: str | typing.Literal[False] | None = None,
        assert_fingerprint: str | None = None,
        ca_cert_dir: str | None = None,
        **conn_kw: typing.Any,
    ) -> None:
        super().__init__(
            host,
            port,
            timeout,
            maxsize,
            block,
            headers,
            retries,
            _proxy,
            _proxy_headers,
            **conn_kw,
        )

        self.key_file = key_file
        self.cert_file = cert_file
        self.cert_reqs = cert_reqs
        self.key_password = key_password
        self.ca_certs = ca_certs
        self.ca_cert_dir = ca_cert_dir
        self.ssl_version = ssl_version
        self.ssl_minimum_version = ssl_minimum_version
        self.ssl_maximum_version = ssl_maximum_version
        self.assert_hostname = assert_hostname
        self.assert_fingerprint = assert_fingerprint

    def _prepare_proxy(self, conn: HTTPSConnection) -> None:  # type: ignore[override]
        """Establishes a tunnel connection through HTTP CONNECT."""
        if self.proxy and self.proxy.scheme == "https":
            tunnel_scheme = "https"
        else:
            tunnel_scheme = "http"

        conn.set_tunnel(
            scheme=tunnel_scheme,
            host=self._tunnel_host,
            port=self.port,
            headers=self.proxy_headers,
        )
        conn.connect()

    def _new_conn(self) -> BaseHTTPSConnection:
        """
        Return a fresh :class:`urllib3.connection.HTTPConnection`.
        """
        self.num_connections += 1
        log.debug(
            "Starting new HTTPS connection (%d): %s:%s",
            self.num_connections,
            self.host,
            self.port or "443",
        )

        if not self.ConnectionCls or self.ConnectionCls is DummyConnection:  # type: ignore[comparison-overlap]
            raise ImportError(
                "Can't connect to HTTPS URL because the SSL module is not available."
            )

        actual_host: str = self.host
        actual_port = self.port
        if self.proxy is not None and self.proxy.host is not None:
            actual_host = self.proxy.host
            actual_port = self.proxy.port

        return self.ConnectionCls(
            host=actual_host,
            port=actual_port,
            timeout=self.timeout.connect_timeout,
            cert_file=self.cert_file,
            key_file=self.key_file,
            key_password=self.key_password,
            cert_reqs=self.cert_reqs,
            ca_certs=self.ca_certs,
            ca_cert_dir=self.ca_cert_dir,
            assert_hostname=self.assert_hostname,
            assert_fingerprint=self.assert_fingerprint,
            ssl_version=self.ssl_version,
            ssl_minimum_version=self.ssl_minimum_version,
            ssl_maximum_version=self.ssl_maximum_version,
            **self.conn_kw,
        )

    def _validate_conn(self, conn: BaseHTTPConnection) -> None:
        """
        Called right before a request is made, after the socket is created.
        """
        super()._validate_conn(conn)

        # Force connect early to allow us to validate the connection.
        if conn.is_closed:
            conn.connect()

        # TODO revise this, see https://github.com/urllib3/urllib3/issues/2791
        if not conn.is_verified and not conn.proxy_is_verified:
            warnings.warn(
                (
                    f"Unverified HTTPS request is being made to host '{conn.host}'. "
                    "Adding certificate verification is strongly advised. See: "
                    "https://urllib3.readthedocs.io/en/latest/advanced-usage.html"
                    "#tls-warnings"
                ),
                InsecureRequestWarning,
            )


def connection_from_url(url: str, **kw: typing.Any) -> HTTPConnectionPool:
    """
    Given a url, return an :class:`.ConnectionPool` instance of its host.

    This is a shortcut for not having to parse out the scheme, host, and port
    of the url before creating an :class:`.ConnectionPool` instance.

    :param url:
        Absolute URL string that must include the scheme. Port is optional.

    :param \\**kw:
        Passes additional parameters to the constructor of the appropriate
        :class:`.ConnectionPool`. Useful for specifying things like
        timeout, maxsize, headers, etc.

    Example::

        >>> conn = connection_from_url('http://google.com/')
        >>> r = conn.request('GET', '/')
    """
    scheme, _, host, port, *_ = parse_url(url)
    scheme = scheme or "http"
    port = port or port_by_scheme.get(scheme, 80)
    if scheme == "https":
        return HTTPSConnectionPool(host, port=port, **kw)  # type: ignore[arg-type]
    else:
        return HTTPConnectionPool(host, port=port, **kw)  # type: ignore[arg-type]


@typing.overload
def _normalize_host(host: None, scheme: str | None) -> None: ...


@typing.overload
def _normalize_host(host: str, scheme: str | None) -> str: ...


def _normalize_host(host: str | None, scheme: str | None) -> str | None:
    """
    Normalize hosts for comparisons and use with sockets.
    """

    host = normalize_host(host, scheme)

    # httplib doesn't like it when we include brackets in IPv6 addresses
    # Specifically, if we include brackets but also pass the port then
    # httplib crazily doubles up the square brackets on the Host header.
    # Instead, we need to make sure we never pass ``None`` as the port.
    # However, for backward compatibility reasons we can't actually
    # *assert* that.  See http://bugs.python.org/issue28539
    if host and host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host


def _url_from_pool(
    pool: HTTPConnectionPool | HTTPSConnectionPool, path: str | None = None
) -> str:
    """Returns the URL from a given connection pool. This is mainly used for testing and logging."""
    return Url(scheme=pool.scheme, host=pool.host, port=pool.port, path=path).url


def _close_pool_connections(pool: queue.LifoQueue[typing.Any]) -> None:
    """Drains a queue of connections and closes each one."""
    try:
        while True:
            conn = pool.get(block=False)
            if conn:
                conn.close()
    except queue.Empty:
        pass  # Done.

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\xml.py ===
"""
:mod:``pandas.io.xml`` is a module for reading XML.
"""

from __future__ import annotations

import io
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)
import warnings

from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
    AbstractMethodError,
    ParserError,
)
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import is_list_like

from pandas.core.shared_docs import _shared_docs

from pandas.io.common import (
    file_exists,
    get_handle,
    infer_compression,
    is_file_like,
    is_fsspec_url,
    is_url,
    stringify_path,
)
from pandas.io.parsers import TextParser

if TYPE_CHECKING:
    from collections.abc import Sequence
    from xml.etree.ElementTree import Element

    from lxml import etree

    from pandas._typing import (
        CompressionOptions,
        ConvertersArg,
        DtypeArg,
        DtypeBackend,
        FilePath,
        ParseDatesArg,
        ReadBuffer,
        StorageOptions,
        XMLParsers,
    )

    from pandas import DataFrame


@doc(
    storage_options=_shared_docs["storage_options"],
    decompression_options=_shared_docs["decompression_options"] % "path_or_buffer",
)
class _XMLFrameParser:
    """
    Internal subclass to parse XML into DataFrames.

    Parameters
    ----------
    path_or_buffer : a valid JSON ``str``, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file.

    xpath : str or regex
        The ``XPath`` expression to parse required set of nodes for
        migration to :class:`~pandas.DataFrame`. ``etree`` supports limited ``XPath``.

    namespaces : dict
        The namespaces defined in XML document (``xmlns:namespace='URI'``)
        as dicts with key being namespace and value the URI.

    elems_only : bool
        Parse only the child elements at the specified ``xpath``.

    attrs_only : bool
        Parse only the attributes at the specified ``xpath``.

    names : list
        Column names for :class:`~pandas.DataFrame` of parsed XML data.

    dtype : dict
        Data type for data or columns. E.g. {{'a': np.float64,
        'b': np.int32, 'c': 'Int64'}}

        .. versionadded:: 1.5.0

    converters : dict, optional
        Dict of functions for converting values in certain columns. Keys can
        either be integers or column labels.

        .. versionadded:: 1.5.0

    parse_dates : bool or list of int or names or list of lists or dict
        Converts either index or select columns to datetimes

        .. versionadded:: 1.5.0

    encoding : str
        Encoding of xml object or document.

    stylesheet : str or file-like
        URL, file, file-like object, or a raw string containing XSLT,
        ``etree`` does not support XSLT but retained for consistency.

    iterparse : dict, optional
        Dict with row element as key and list of descendant elements
        and/or attributes as value to be retrieved in iterparsing of
        XML document.

        .. versionadded:: 1.5.0

    {decompression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    {storage_options}

    See also
    --------
    pandas.io.xml._EtreeFrameParser
    pandas.io.xml._LxmlFrameParser

    Notes
    -----
    To subclass this class effectively you must override the following methods:`
        * :func:`parse_data`
        * :func:`_parse_nodes`
        * :func:`_iterparse_nodes`
        * :func:`_parse_doc`
        * :func:`_validate_names`
        * :func:`_validate_path`


    See each method's respective documentation for details on their
    functionality.
    """

    def __init__(
        self,
        path_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
        xpath: str,
        namespaces: dict[str, str] | None,
        elems_only: bool,
        attrs_only: bool,
        names: Sequence[str] | None,
        dtype: DtypeArg | None,
        converters: ConvertersArg | None,
        parse_dates: ParseDatesArg | None,
        encoding: str | None,
        stylesheet: FilePath | ReadBuffer[bytes] | ReadBuffer[str] | None,
        iterparse: dict[str, list[str]] | None,
        compression: CompressionOptions,
        storage_options: StorageOptions,
    ) -> None:
        self.path_or_buffer = path_or_buffer
        self.xpath = xpath
        self.namespaces = namespaces
        self.elems_only = elems_only
        self.attrs_only = attrs_only
        self.names = names
        self.dtype = dtype
        self.converters = converters
        self.parse_dates = parse_dates
        self.encoding = encoding
        self.stylesheet = stylesheet
        self.iterparse = iterparse
        self.is_style = None
        self.compression: CompressionOptions = compression
        self.storage_options = storage_options

    def parse_data(self) -> list[dict[str, str | None]]:
        """
        Parse xml data.

        This method will call the other internal methods to
        validate ``xpath``, names, parse and return specific nodes.
        """

        raise AbstractMethodError(self)

    def _parse_nodes(self, elems: list[Any]) -> list[dict[str, str | None]]:
        """
        Parse xml nodes.

        This method will parse the children and attributes of elements
        in ``xpath``, conditionally for only elements, only attributes
        or both while optionally renaming node names.

        Raises
        ------
        ValueError
            * If only elements and only attributes are specified.

        Notes
        -----
        Namespace URIs will be removed from return node values. Also,
        elements with missing children or attributes compared to siblings
        will have optional keys filled with None values.
        """

        dicts: list[dict[str, str | None]]

        if self.elems_only and self.attrs_only:
            raise ValueError("Either element or attributes can be parsed not both.")
        if self.elems_only:
            if self.names:
                dicts = [
                    {
                        **(
                            {el.tag: el.text}
                            if el.text and not el.text.isspace()
                            else {}
                        ),
                        **{
                            nm: ch.text if ch.text else None
                            for nm, ch in zip(self.names, el.findall("*"))
                        },
                    }
                    for el in elems
                ]
            else:
                dicts = [
                    {ch.tag: ch.text if ch.text else None for ch in el.findall("*")}
                    for el in elems
                ]

        elif self.attrs_only:
            dicts = [
                {k: v if v else None for k, v in el.attrib.items()} for el in elems
            ]

        elif self.names:
            dicts = [
                {
                    **el.attrib,
                    **({el.tag: el.text} if el.text and not el.text.isspace() else {}),
                    **{
                        nm: ch.text if ch.text else None
                        for nm, ch in zip(self.names, el.findall("*"))
                    },
                }
                for el in elems
            ]

        else:
            dicts = [
                {
                    **el.attrib,
                    **({el.tag: el.text} if el.text and not el.text.isspace() else {}),
                    **{ch.tag: ch.text if ch.text else None for ch in el.findall("*")},
                }
                for el in elems
            ]

        dicts = [
            {k.split("}")[1] if "}" in k else k: v for k, v in d.items()} for d in dicts
        ]

        keys = list(dict.fromkeys([k for d in dicts for k in d.keys()]))
        dicts = [{k: d[k] if k in d.keys() else None for k in keys} for d in dicts]

        if self.names:
            dicts = [dict(zip(self.names, d.values())) for d in dicts]

        return dicts

    def _iterparse_nodes(self, iterparse: Callable) -> list[dict[str, str | None]]:
        """
        Iterparse xml nodes.

        This method will read in local disk, decompressed XML files for elements
        and underlying descendants using iterparse, a method to iterate through
        an XML tree without holding entire XML tree in memory.

        Raises
        ------
        TypeError
            * If ``iterparse`` is not a dict or its dict value is not list-like.
        ParserError
            * If ``path_or_buffer`` is not a physical file on disk or file-like object.
            * If no data is returned from selected items in ``iterparse``.

        Notes
        -----
        Namespace URIs will be removed from return node values. Also,
        elements with missing children or attributes in submitted list
        will have optional keys filled with None values.
        """

        dicts: list[dict[str, str | None]] = []
        row: dict[str, str | None] | None = None

        if not isinstance(self.iterparse, dict):
            raise TypeError(
                f"{type(self.iterparse).__name__} is not a valid type for iterparse"
            )

        row_node = next(iter(self.iterparse.keys())) if self.iterparse else ""
        if not is_list_like(self.iterparse[row_node]):
            raise TypeError(
                f"{type(self.iterparse[row_node])} is not a valid type "
                "for value in iterparse"
            )

        if (not hasattr(self.path_or_buffer, "read")) and (
            not isinstance(self.path_or_buffer, (str, PathLike))
            or is_url(self.path_or_buffer)
            or is_fsspec_url(self.path_or_buffer)
            or (
                isinstance(self.path_or_buffer, str)
                and self.path_or_buffer.startswith(("<?xml", "<"))
            )
            or infer_compression(self.path_or_buffer, "infer") is not None
        ):
            raise ParserError(
                "iterparse is designed for large XML files that are fully extracted on "
                "local disk and not as compressed files or online sources."
            )

        iterparse_repeats = len(self.iterparse[row_node]) != len(
            set(self.iterparse[row_node])
        )

        for event, elem in iterparse(self.path_or_buffer, events=("start", "end")):
            curr_elem = elem.tag.split("}")[1] if "}" in elem.tag else elem.tag

            if event == "start":
                if curr_elem == row_node:
                    row = {}

            if row is not None:
                if self.names and iterparse_repeats:
                    for col, nm in zip(self.iterparse[row_node], self.names):
                        if curr_elem == col:
                            elem_val = elem.text if elem.text else None
                            if elem_val not in row.values() and nm not in row:
                                row[nm] = elem_val

                        if col in elem.attrib:
                            if elem.attrib[col] not in row.values() and nm not in row:
                                row[nm] = elem.attrib[col]
                else:
                    for col in self.iterparse[row_node]:
                        if curr_elem == col:
                            row[col] = elem.text if elem.text else None
                        if col in elem.attrib:
                            row[col] = elem.attrib[col]

            if event == "end":
                if curr_elem == row_node and row is not None:
                    dicts.append(row)
                    row = None

                elem.clear()
                if hasattr(elem, "getprevious"):
                    while (
                        elem.getprevious() is not None and elem.getparent() is not None
                    ):
                        del elem.getparent()[0]

        if dicts == []:
            raise ParserError("No result from selected items in iterparse.")

        keys = list(dict.fromkeys([k for d in dicts for k in d.keys()]))
        dicts = [{k: d[k] if k in d.keys() else None for k in keys} for d in dicts]

        if self.names:
            dicts = [dict(zip(self.names, d.values())) for d in dicts]

        return dicts

    def _validate_path(self) -> list[Any]:
        """
        Validate ``xpath``.

        This method checks for syntax, evaluation, or empty nodes return.

        Raises
        ------
        SyntaxError
            * If xpah is not supported or issues with namespaces.

        ValueError
            * If xpah does not return any nodes.
        """

        raise AbstractMethodError(self)

    def _validate_names(self) -> None:
        """
        Validate names.

        This method will check if names is a list-like and aligns
        with length of parse nodes.

        Raises
        ------
        ValueError
            * If value is not a list and less then length of nodes.
        """
        raise AbstractMethodError(self)

    def _parse_doc(
        self, raw_doc: FilePath | ReadBuffer[bytes] | ReadBuffer[str]
    ) -> Element | etree._Element:
        """
        Build tree from path_or_buffer.

        This method will parse XML object into tree
        either from string/bytes or file location.
        """
        raise AbstractMethodError(self)


class _EtreeFrameParser(_XMLFrameParser):
    """
    Internal class to parse XML into DataFrames with the Python
    standard library XML module: `xml.etree.ElementTree`.
    """

    def parse_data(self) -> list[dict[str, str | None]]:
        from xml.etree.ElementTree import iterparse

        if self.stylesheet is not None:
            raise ValueError(
                "To use stylesheet, you need lxml installed and selected as parser."
            )

        if self.iterparse is None:
            self.xml_doc = self._parse_doc(self.path_or_buffer)
            elems = self._validate_path()

        self._validate_names()

        xml_dicts: list[dict[str, str | None]] = (
            self._parse_nodes(elems)
            if self.iterparse is None
            else self._iterparse_nodes(iterparse)
        )

        return xml_dicts

    def _validate_path(self) -> list[Any]:
        """
        Notes
        -----
        ``etree`` supports limited ``XPath``. If user attempts a more complex
        expression syntax error will raise.
        """

        msg = (
            "xpath does not return any nodes or attributes. "
            "Be sure to specify in `xpath` the parent nodes of "
            "children and attributes to parse. "
            "If document uses namespaces denoted with "
            "xmlns, be sure to define namespaces and "
            "use them in xpath."
        )
        try:
            elems = self.xml_doc.findall(self.xpath, namespaces=self.namespaces)
            children = [ch for el in elems for ch in el.findall("*")]
            attrs = {k: v for el in elems for k, v in el.attrib.items()}

            if elems is None:
                raise ValueError(msg)

            if elems is not None:
                if self.elems_only and children == []:
                    raise ValueError(msg)
                if self.attrs_only and attrs == {}:
                    raise ValueError(msg)
                if children == [] and attrs == {}:
                    raise ValueError(msg)

        except (KeyError, SyntaxError):
            raise SyntaxError(
                "You have used an incorrect or unsupported XPath "
                "expression for etree library or you used an "
                "undeclared namespace prefix."
            )

        return elems

    def _validate_names(self) -> None:
        children: list[Any]

        if self.names:
            if self.iterparse:
                children = self.iterparse[next(iter(self.iterparse))]
            else:
                parent = self.xml_doc.find(self.xpath, namespaces=self.namespaces)
                children = parent.findall("*") if parent is not None else []

            if is_list_like(self.names):
                if len(self.names) < len(children):
                    raise ValueError(
                        "names does not match length of child elements in xpath."
                    )
            else:
                raise TypeError(
                    f"{type(self.names).__name__} is not a valid type for names"
                )

    def _parse_doc(
        self, raw_doc: FilePath | ReadBuffer[bytes] | ReadBuffer[str]
    ) -> Element:
        from xml.etree.ElementTree import (
            XMLParser,
            parse,
        )

        handle_data = get_data_from_filepath(
            filepath_or_buffer=raw_doc,
            encoding=self.encoding,
            compression=self.compression,
            storage_options=self.storage_options,
        )

        with preprocess_data(handle_data) as xml_data:
            curr_parser = XMLParser(encoding=self.encoding)
            document = parse(xml_data, parser=curr_parser)

        return document.getroot()


class _LxmlFrameParser(_XMLFrameParser):
    """
    Internal class to parse XML into :class:`~pandas.DataFrame` with third-party
    full-featured XML library, ``lxml``, that supports
    ``XPath`` 1.0 and XSLT 1.0.
    """

    def parse_data(self) -> list[dict[str, str | None]]:
        """
        Parse xml data.

        This method will call the other internal methods to
        validate ``xpath``, names, optionally parse and run XSLT,
        and parse original or transformed XML and return specific nodes.
        """
        from lxml.etree import iterparse

        if self.iterparse is None:
            self.xml_doc = self._parse_doc(self.path_or_buffer)

            if self.stylesheet:
                self.xsl_doc = self._parse_doc(self.stylesheet)
                self.xml_doc = self._transform_doc()

            elems = self._validate_path()

        self._validate_names()

        xml_dicts: list[dict[str, str | None]] = (
            self._parse_nodes(elems)
            if self.iterparse is None
            else self._iterparse_nodes(iterparse)
        )

        return xml_dicts

    def _validate_path(self) -> list[Any]:
        msg = (
            "xpath does not return any nodes or attributes. "
            "Be sure to specify in `xpath` the parent nodes of "
            "children and attributes to parse. "
            "If document uses namespaces denoted with "
            "xmlns, be sure to define namespaces and "
            "use them in xpath."
        )

        elems = self.xml_doc.xpath(self.xpath, namespaces=self.namespaces)
        children = [ch for el in elems for ch in el.xpath("*")]
        attrs = {k: v for el in elems for k, v in el.attrib.items()}

        if elems == []:
            raise ValueError(msg)

        if elems != []:
            if self.elems_only and children == []:
                raise ValueError(msg)
            if self.attrs_only and attrs == {}:
                raise ValueError(msg)
            if children == [] and attrs == {}:
                raise ValueError(msg)

        return elems

    def _validate_names(self) -> None:
        children: list[Any]

        if self.names:
            if self.iterparse:
                children = self.iterparse[next(iter(self.iterparse))]
            else:
                children = self.xml_doc.xpath(
                    self.xpath + "[1]/*", namespaces=self.namespaces
                )

            if is_list_like(self.names):
                if len(self.names) < len(children):
                    raise ValueError(
                        "names does not match length of child elements in xpath."
                    )
            else:
                raise TypeError(
                    f"{type(self.names).__name__} is not a valid type for names"
                )

    def _parse_doc(
        self, raw_doc: FilePath | ReadBuffer[bytes] | ReadBuffer[str]
    ) -> etree._Element:
        from lxml.etree import (
            XMLParser,
            fromstring,
            parse,
        )

        handle_data = get_data_from_filepath(
            filepath_or_buffer=raw_doc,
            encoding=self.encoding,
            compression=self.compression,
            storage_options=self.storage_options,
        )

        with preprocess_data(handle_data) as xml_data:
            curr_parser = XMLParser(encoding=self.encoding)

            if isinstance(xml_data, io.StringIO):
                if self.encoding is None:
                    raise TypeError(
                        "Can not pass encoding None when input is StringIO."
                    )

                document = fromstring(
                    xml_data.getvalue().encode(self.encoding), parser=curr_parser
                )
            else:
                document = parse(xml_data, parser=curr_parser)

        return document

    def _transform_doc(self) -> etree._XSLTResultTree:
        """
        Transform original tree using stylesheet.

        This method will transform original xml using XSLT script into
        am ideally flatter xml document for easier parsing and migration
        to Data Frame.
        """
        from lxml.etree import XSLT

        transformer = XSLT(self.xsl_doc)
        new_doc = transformer(self.xml_doc)

        return new_doc


def get_data_from_filepath(
    filepath_or_buffer: FilePath | bytes | ReadBuffer[bytes] | ReadBuffer[str],
    encoding: str | None,
    compression: CompressionOptions,
    storage_options: StorageOptions,
) -> str | bytes | ReadBuffer[bytes] | ReadBuffer[str]:
    """
    Extract raw XML data.

    The method accepts three input types:
        1. filepath (string-like)
        2. file-like object (e.g. open file object, StringIO)
        3. XML string or bytes

    This method turns (1) into (2) to simplify the rest of the processing.
    It returns input types (2) and (3) unchanged.
    """
    if not isinstance(filepath_or_buffer, bytes):
        filepath_or_buffer = stringify_path(filepath_or_buffer)

    if (
        isinstance(filepath_or_buffer, str)
        and not filepath_or_buffer.startswith(("<?xml", "<"))
    ) and (
        not isinstance(filepath_or_buffer, str)
        or is_url(filepath_or_buffer)
        or is_fsspec_url(filepath_or_buffer)
        or file_exists(filepath_or_buffer)
    ):
        with get_handle(
            filepath_or_buffer,
            "r",
            encoding=encoding,
            compression=compression,
            storage_options=storage_options,
        ) as handle_obj:
            filepath_or_buffer = (
                handle_obj.handle.read()
                if hasattr(handle_obj.handle, "read")
                else handle_obj.handle
            )

    return filepath_or_buffer


def preprocess_data(data) -> io.StringIO | io.BytesIO:
    """
    Convert extracted raw data.

    This method will return underlying data of extracted XML content.
    The data either has a `read` attribute (e.g. a file object or a
    StringIO/BytesIO) or is a string or bytes that is an XML document.
    """

    if isinstance(data, str):
        data = io.StringIO(data)

    elif isinstance(data, bytes):
        data = io.BytesIO(data)

    return data


def _data_to_frame(data, **kwargs) -> DataFrame:
    """
    Convert parsed data to Data Frame.

    This method will bind xml dictionary data of keys and values
    into named columns of Data Frame using the built-in TextParser
    class that build Data Frame and infers specific dtypes.
    """

    tags = next(iter(data))
    nodes = [list(d.values()) for d in data]

    try:
        with TextParser(nodes, names=tags, **kwargs) as tp:
            return tp.read()
    except ParserError:
        raise ParserError(
            "XML document may be too complex for import. "
            "Try to flatten document and use distinct "
            "element and attribute names."
        )


def _parse(
    path_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    xpath: str,
    namespaces: dict[str, str] | None,
    elems_only: bool,
    attrs_only: bool,
    names: Sequence[str] | None,
    dtype: DtypeArg | None,
    converters: ConvertersArg | None,
    parse_dates: ParseDatesArg | None,
    encoding: str | None,
    parser: XMLParsers,
    stylesheet: FilePath | ReadBuffer[bytes] | ReadBuffer[str] | None,
    iterparse: dict[str, list[str]] | None,
    compression: CompressionOptions,
    storage_options: StorageOptions,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    **kwargs,
) -> DataFrame:
    """
    Call internal parsers.

    This method will conditionally call internal parsers:
    LxmlFrameParser and/or EtreeParser.

    Raises
    ------
    ImportError
        * If lxml is not installed if selected as parser.

    ValueError
        * If parser is not lxml or etree.
    """

    p: _EtreeFrameParser | _LxmlFrameParser

    if isinstance(path_or_buffer, str) and not any(
        [
            is_file_like(path_or_buffer),
            file_exists(path_or_buffer),
            is_url(path_or_buffer),
            is_fsspec_url(path_or_buffer),
        ]
    ):
        warnings.warn(
            "Passing literal xml to 'read_xml' is deprecated and "
            "will be removed in a future version. To read from a "
            "literal string, wrap it in a 'StringIO' object.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    if parser == "lxml":
        lxml = import_optional_dependency("lxml.etree", errors="ignore")

        if lxml is not None:
            p = _LxmlFrameParser(
                path_or_buffer,
                xpath,
                namespaces,
                elems_only,
                attrs_only,
                names,
                dtype,
                converters,
                parse_dates,
                encoding,
                stylesheet,
                iterparse,
                compression,
                storage_options,
            )
        else:
            raise ImportError("lxml not found, please install or use the etree parser.")

    elif parser == "etree":
        p = _EtreeFrameParser(
            path_or_buffer,
            xpath,
            namespaces,
            elems_only,
            attrs_only,
            names,
            dtype,
            converters,
            parse_dates,
            encoding,
            stylesheet,
            iterparse,
            compression,
            storage_options,
        )
    else:
        raise ValueError("Values for parser can only be lxml or etree.")

    data_dicts = p.parse_data()

    return _data_to_frame(
        data=data_dicts,
        dtype=dtype,
        converters=converters,
        parse_dates=parse_dates,
        dtype_backend=dtype_backend,
        **kwargs,
    )


@doc(
    storage_options=_shared_docs["storage_options"],
    decompression_options=_shared_docs["decompression_options"] % "path_or_buffer",
)
def read_xml(
    path_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    *,
    xpath: str = "./*",
    namespaces: dict[str, str] | None = None,
    elems_only: bool = False,
    attrs_only: bool = False,
    names: Sequence[str] | None = None,
    dtype: DtypeArg | None = None,
    converters: ConvertersArg | None = None,
    parse_dates: ParseDatesArg | None = None,
    # encoding can not be None for lxml and StringIO input
    encoding: str | None = "utf-8",
    parser: XMLParsers = "lxml",
    stylesheet: FilePath | ReadBuffer[bytes] | ReadBuffer[str] | None = None,
    iterparse: dict[str, list[str]] | None = None,
    compression: CompressionOptions = "infer",
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame:
    r"""
    Read XML document into a :class:`~pandas.DataFrame` object.

    .. versionadded:: 1.3.0

    Parameters
    ----------
    path_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a ``read()`` function. The string can be any valid XML
        string or a path. The string can further be a URL. Valid URL schemes
        include http, ftp, s3, and file.

        .. deprecated:: 2.1.0
            Passing xml literal strings is deprecated.
            Wrap literal xml input in ``io.StringIO`` or ``io.BytesIO`` instead.

    xpath : str, optional, default './\*'
        The ``XPath`` to parse required set of nodes for migration to
        :class:`~pandas.DataFrame`.``XPath`` should return a collection of elements
        and not a single element. Note: The ``etree`` parser supports limited ``XPath``
        expressions. For more complex ``XPath``, use ``lxml`` which requires
        installation.

    namespaces : dict, optional
        The namespaces defined in XML document as dicts with key being
        namespace prefix and value the URI. There is no need to include all
        namespaces in XML, only the ones used in ``xpath`` expression.
        Note: if XML document uses default namespace denoted as
        `xmlns='<URI>'` without a prefix, you must assign any temporary
        namespace prefix such as 'doc' to the URI in order to parse
        underlying nodes and/or attributes. For example, ::

            namespaces = {{"doc": "https://example.com"}}

    elems_only : bool, optional, default False
        Parse only the child elements at the specified ``xpath``. By default,
        all child elements and non-empty text nodes are returned.

    attrs_only :  bool, optional, default False
        Parse only the attributes at the specified ``xpath``.
        By default, all attributes are returned.

    names :  list-like, optional
        Column names for DataFrame of parsed XML data. Use this parameter to
        rename original element names and distinguish same named elements and
        attributes.

    dtype : Type name or dict of column -> type, optional
        Data type for data or columns. E.g. {{'a': np.float64, 'b': np.int32,
        'c': 'Int64'}}
        Use `str` or `object` together with suitable `na_values` settings
        to preserve and not interpret dtype.
        If converters are specified, they will be applied INSTEAD
        of dtype conversion.

        .. versionadded:: 1.5.0

    converters : dict, optional
        Dict of functions for converting values in certain columns. Keys can either
        be integers or column labels.

        .. versionadded:: 1.5.0

    parse_dates : bool or list of int or names or list of lists or dict, default False
        Identifiers to parse index or columns to datetime. The behavior is as follows:

        * boolean. If True -> try parsing the index.
        * list of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3
          each as a separate date column.
        * list of lists. e.g.  If [[1, 3]] -> combine columns 1 and 3 and parse as
          a single date column.
        * dict, e.g. {{'foo' : [1, 3]}} -> parse columns 1, 3 as date and call
          result 'foo'

        .. versionadded:: 1.5.0

    encoding : str, optional, default 'utf-8'
        Encoding of XML document.

    parser : {{'lxml','etree'}}, default 'lxml'
        Parser module to use for retrieval of data. Only 'lxml' and
        'etree' are supported. With 'lxml' more complex ``XPath`` searches
        and ability to use XSLT stylesheet are supported.

    stylesheet : str, path object or file-like object
        A URL, file-like object, or a raw string containing an XSLT script.
        This stylesheet should flatten complex, deeply nested XML documents
        for easier parsing. To use this feature you must have ``lxml`` module
        installed and specify 'lxml' as ``parser``. The ``xpath`` must
        reference nodes of transformed XML document generated after XSLT
        transformation and not the original XML document. Only XSLT 1.0
        scripts and not later versions is currently supported.

    iterparse : dict, optional
        The nodes or attributes to retrieve in iterparsing of XML document
        as a dict with key being the name of repeating element and value being
        list of elements or attribute names that are descendants of the repeated
        element. Note: If this option is used, it will replace ``xpath`` parsing
        and unlike ``xpath``, descendants do not need to relate to each other but can
        exist any where in document under the repeating element. This memory-
        efficient method should be used for very large XML files (500MB, 1GB, or 5GB+).
        For example, ::

            iterparse = {{"row_element": ["child_elem", "attr", "grandchild_elem"]}}

        .. versionadded:: 1.5.0

    {decompression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    {storage_options}

    dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    Returns
    -------
    df
        A DataFrame.

    See Also
    --------
    read_json : Convert a JSON string to pandas object.
    read_html : Read HTML tables into a list of DataFrame objects.

    Notes
    -----
    This method is best designed to import shallow XML documents in
    following format which is the ideal fit for the two-dimensions of a
    ``DataFrame`` (row by column). ::

            <root>
                <row>
                  <column1>data</column1>
                  <column2>data</column2>
                  <column3>data</column3>
                  ...
               </row>
               <row>
                  ...
               </row>
               ...
            </root>

    As a file format, XML documents can be designed any way including
    layout of elements and attributes as long as it conforms to W3C
    specifications. Therefore, this method is a convenience handler for
    a specific flatter design and not all possible XML structures.

    However, for more complex XML documents, ``stylesheet`` allows you to
    temporarily redesign original document with XSLT (a special purpose
    language) for a flatter version for migration to a DataFrame.

    This function will *always* return a single :class:`DataFrame` or raise
    exceptions due to issues with XML document, ``xpath``, or other
    parameters.

    See the :ref:`read_xml documentation in the IO section of the docs
    <io.read_xml>` for more information in using this method to parse XML
    files to DataFrames.

    Examples
    --------
    >>> from io import StringIO
    >>> xml = '''<?xml version='1.0' encoding='utf-8'?>
    ... <data xmlns="http://example.com">
    ...  <row>
    ...    <shape>square</shape>
    ...    <degrees>360</degrees>
    ...    <sides>4.0</sides>
    ...  </row>
    ...  <row>
    ...    <shape>circle</shape>
    ...    <degrees>360</degrees>
    ...    <sides/>
    ...  </row>
    ...  <row>
    ...    <shape>triangle</shape>
    ...    <degrees>180</degrees>
    ...    <sides>3.0</sides>
    ...  </row>
    ... </data>'''

    >>> df = pd.read_xml(StringIO(xml))
    >>> df
          shape  degrees  sides
    0    square      360    4.0
    1    circle      360    NaN
    2  triangle      180    3.0

    >>> xml = '''<?xml version='1.0' encoding='utf-8'?>
    ... <data>
    ...   <row shape="square" degrees="360" sides="4.0"/>
    ...   <row shape="circle" degrees="360"/>
    ...   <row shape="triangle" degrees="180" sides="3.0"/>
    ... </data>'''

    >>> df = pd.read_xml(StringIO(xml), xpath=".//row")
    >>> df
          shape  degrees  sides
    0    square      360    4.0
    1    circle      360    NaN
    2  triangle      180    3.0

    >>> xml = '''<?xml version='1.0' encoding='utf-8'?>
    ... <doc:data xmlns:doc="https://example.com">
    ...   <doc:row>
    ...     <doc:shape>square</doc:shape>
    ...     <doc:degrees>360</doc:degrees>
    ...     <doc:sides>4.0</doc:sides>
    ...   </doc:row>
    ...   <doc:row>
    ...     <doc:shape>circle</doc:shape>
    ...     <doc:degrees>360</doc:degrees>
    ...     <doc:sides/>
    ...   </doc:row>
    ...   <doc:row>
    ...     <doc:shape>triangle</doc:shape>
    ...     <doc:degrees>180</doc:degrees>
    ...     <doc:sides>3.0</doc:sides>
    ...   </doc:row>
    ... </doc:data>'''

    >>> df = pd.read_xml(StringIO(xml),
    ...                  xpath="//doc:row",
    ...                  namespaces={{"doc": "https://example.com"}})
    >>> df
          shape  degrees  sides
    0    square      360    4.0
    1    circle      360    NaN
    2  triangle      180    3.0

    >>> xml_data = '''
    ...         <data>
    ...            <row>
    ...               <index>0</index>
    ...               <a>1</a>
    ...               <b>2.5</b>
    ...               <c>True</c>
    ...               <d>a</d>
    ...               <e>2019-12-31 00:00:00</e>
    ...            </row>
    ...            <row>
    ...               <index>1</index>
    ...               <b>4.5</b>
    ...               <c>False</c>
    ...               <d>b</d>
    ...               <e>2019-12-31 00:00:00</e>
    ...            </row>
    ...         </data>
    ...         '''

    >>> df = pd.read_xml(StringIO(xml_data),
    ...                  dtype_backend="numpy_nullable",
    ...                  parse_dates=["e"])
    >>> df
       index     a    b      c  d          e
    0      0     1  2.5   True  a 2019-12-31
    1      1  <NA>  4.5  False  b 2019-12-31
    """
    check_dtype_backend(dtype_backend)

    return _parse(
        path_or_buffer=path_or_buffer,
        xpath=xpath,
        namespaces=namespaces,
        elems_only=elems_only,
        attrs_only=attrs_only,
        names=names,
        dtype=dtype,
        converters=converters,
        parse_dates=parse_dates,
        encoding=encoding,
        parser=parser,
        stylesheet=stylesheet,
        iterparse=iterparse,
        compression=compression,
        storage_options=storage_options,
        dtype_backend=dtype_backend,
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\ddm.py ===
"""

Module for the DDM class.

The DDM class is an internal representation used by DomainMatrix. The letters
DDM stand for Dense Domain Matrix. A DDM instance represents a matrix using
elements from a polynomial Domain (e.g. ZZ, QQ, ...) in a dense-matrix
representation.

Basic usage:

    >>> from sympy import ZZ, QQ
    >>> from sympy.polys.matrices.ddm import DDM
    >>> A = DDM([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
    >>> A.shape
    (2, 2)
    >>> A
    [[0, 1], [-1, 0]]
    >>> type(A)
    <class 'sympy.polys.matrices.ddm.DDM'>
    >>> A @ A
    [[-1, 0], [0, -1]]

The ddm_* functions are designed to operate on DDM as well as on an ordinary
list of lists:

    >>> from sympy.polys.matrices.dense import ddm_idet
    >>> ddm_idet(A, QQ)
    1
    >>> ddm_idet([[0, 1], [-1, 0]], QQ)
    1
    >>> A
    [[-1, 0], [0, -1]]

Note that ddm_idet modifies the input matrix in-place. It is recommended to
use the DDM.det method as a friendlier interface to this instead which takes
care of copying the matrix:

    >>> B = DDM([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
    >>> B.det()
    1

Normally DDM would not be used directly and is just part of the internal
representation of DomainMatrix which adds further functionality including e.g.
unifying domains.

The dense format used by DDM is a list of lists of elements e.g. the 2x2
identity matrix is like [[1, 0], [0, 1]]. The DDM class itself is a subclass
of list and its list items are plain lists. Elements are accessed as e.g.
ddm[i][j] where ddm[i] gives the ith row and ddm[i][j] gets the element in the
jth column of that row. Subclassing list makes e.g. iteration and indexing
very efficient. We do not override __getitem__ because it would lose that
benefit.

The core routines are implemented by the ddm_* functions defined in dense.py.
Those functions are intended to be able to operate on a raw list-of-lists
representation of matrices with most functions operating in-place. The DDM
class takes care of copying etc and also stores a Domain object associated
with its elements. This makes it possible to implement things like A + B with
domain checking and also shape checking so that the list of lists
representation is friendlier.

"""
from itertools import chain

from sympy.external.gmpy import GROUND_TYPES
from sympy.utilities.decorator import doctest_depends_on

from .exceptions import (
    DMBadInputError,
    DMDomainError,
    DMNonSquareMatrixError,
    DMShapeError,
)

from sympy.polys.domains import QQ

from .dense import (
        ddm_transpose,
        ddm_iadd,
        ddm_isub,
        ddm_ineg,
        ddm_imul,
        ddm_irmul,
        ddm_imatmul,
        ddm_irref,
        ddm_irref_den,
        ddm_idet,
        ddm_iinv,
        ddm_ilu_split,
        ddm_ilu_solve,
        ddm_berk,
        )

from .lll import ddm_lll, ddm_lll_transform


if GROUND_TYPES != 'flint':
    __doctest_skip__ = ['DDM.to_dfm', 'DDM.to_dfm_or_ddm']


class DDM(list):
    """Dense matrix based on polys domain elements

    This is a list subclass and is a wrapper for a list of lists that supports
    basic matrix arithmetic +, -, *, **.
    """

    fmt = 'dense'
    is_DFM = False
    is_DDM = True

    def __init__(self, rowslist, shape, domain):
        if not (isinstance(rowslist, list) and all(type(row) is list for row in rowslist)):
            raise DMBadInputError("rowslist must be a list of lists")
        m, n = shape
        if len(rowslist) != m or any(len(row) != n for row in rowslist):
            raise DMBadInputError("Inconsistent row-list/shape")

        super().__init__([i.copy() for i in rowslist])
        self.shape = (m, n)
        self.rows = m
        self.cols = n
        self.domain = domain

    def getitem(self, i, j):
        return self[i][j]

    def setitem(self, i, j, value):
        self[i][j] = value

    def extract_slice(self, slice1, slice2):
        ddm = [row[slice2] for row in self[slice1]]
        rows = len(ddm)
        cols = len(ddm[0]) if ddm else len(range(self.shape[1])[slice2])
        return DDM(ddm, (rows, cols), self.domain)

    def extract(self, rows, cols):
        ddm = []
        for i in rows:
            rowi = self[i]
            ddm.append([rowi[j] for j in cols])
        return DDM(ddm, (len(rows), len(cols)), self.domain)

    @classmethod
    def from_list(cls, rowslist, shape, domain):
        """
        Create a :class:`DDM` from a list of lists.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM.from_list([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
        >>> A
        [[0, 1], [-1, 0]]
        >>> A == DDM([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
        True

        See Also
        ========

        from_list_flat
        """
        return cls(rowslist, shape, domain)

    @classmethod
    def from_ddm(cls, other):
        return other.copy()

    def to_list(self):
        """
        Convert to a list of lists.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_list()
        [[1, 2], [3, 4]]

        See Also
        ========

        to_list_flat
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_list
        """
        return [row[:] for row in self]

    def to_list_flat(self):
        """
        Convert to a flat list of elements.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_list_flat()
        [1, 2, 3, 4]
        >>> A == DDM.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.to_list_flat
        """
        flat = []
        for row in self:
            flat.extend(row)
        return flat

    @classmethod
    def from_list_flat(cls, flat, shape, domain):
        """
        Create a :class:`DDM` from a flat list of elements.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM.from_list_flat([1, 2, 3, 4], (2, 2), QQ)
        >>> A
        [[1, 2], [3, 4]]
        >>> A == DDM.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        to_list_flat
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_list_flat
        """
        assert type(flat) is list
        rows, cols = shape
        if not (len(flat) == rows*cols):
            raise DMBadInputError("Inconsistent flat-list shape")
        lol = [flat[i*cols:(i+1)*cols] for i in range(rows)]
        return cls(lol, shape, domain)

    def flatiter(self):
        return chain.from_iterable(self)

    def flat(self):
        items = []
        for row in self:
            items.extend(row)
        return items

    def to_flat_nz(self):
        """
        Convert to a flat list of nonzero elements and data.

        Explanation
        ===========

        This is used to operate on a list of the elements of a matrix and then
        reconstruct a matrix using :meth:`from_flat_nz`. Zero elements are
        included in the list but that may change in the future.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [1, 2, 3, 4]
        >>> A == DDM.from_flat_nz(elements, data, A.domain)
        True

        See Also
        ========

        from_flat_nz
        sympy.polys.matrices.sdm.SDM.to_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_flat_nz
        """
        return self.to_sdm().to_flat_nz()

    @classmethod
    def from_flat_nz(cls, elements, data, domain):
        """
        Reconstruct a :class:`DDM` after calling :meth:`to_flat_nz`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [1, 2, 3, 4]
        >>> A == DDM.from_flat_nz(elements, data, A.domain)
        True

        See Also
        ========

        to_flat_nz
        sympy.polys.matrices.sdm.SDM.from_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_flat_nz
        """
        return SDM.from_flat_nz(elements, data, domain).to_ddm()

    def to_dod(self):
        """
        Convert to a dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dod()
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}

        See Also
        ========

        from_dod
        sympy.polys.matrices.sdm.SDM.to_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dod
        """
        dod = {}
        for i, row in enumerate(self):
            row = {j:e for j, e in enumerate(row) if e}
            if row:
                dod[i] = row
        return dod

    @classmethod
    def from_dod(cls, dod, shape, domain):
        """
        Create a :class:`DDM` from a dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> dod = {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
        >>> A = DDM.from_dod(dod, (2, 2), QQ)
        >>> A
        [[1, 2], [3, 4]]

        See Also
        ========

        to_dod
        sympy.polys.matrices.sdm.SDM.from_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_dod
        """
        rows, cols = shape
        lol = [[domain.zero] * cols for _ in range(rows)]
        for i, row in dod.items():
            for j, element in row.items():
                lol[i][j] = element
        return DDM(lol, shape, domain)

    def to_dok(self):
        """
        Convert :class:`DDM` to dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dok()
        {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}

        See Also
        ========

        from_dok
        sympy.polys.matrices.sdm.SDM.to_dok
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dok
        """
        dok = {}
        for i, row in enumerate(self):
            for j, element in enumerate(row):
                if element:
                    dok[i, j] = element
        return dok

    @classmethod
    def from_dok(cls, dok, shape, domain):
        """
        Create a :class:`DDM` from a dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> dok = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
        >>> A = DDM.from_dok(dok, (2, 2), QQ)
        >>> A
        [[1, 2], [3, 4]]

        See Also
        ========

        to_dok
        sympy.polys.matrices.sdm.SDM.from_dok
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_dok
        """
        rows, cols = shape
        lol = [[domain.zero] * cols for _ in range(rows)]
        for (i, j), element in dok.items():
            lol[i][j] = element
        return DDM(lol, shape, domain)

    def iter_values(self):
        """
        Iterate over the non-zero values of the matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[QQ(1), QQ(0)], [QQ(3), QQ(4)]], (2, 2), QQ)
        >>> list(A.iter_values())
        [1, 3, 4]

        See Also
        ========

        iter_items
        to_list_flat
        sympy.polys.matrices.domainmatrix.DomainMatrix.iter_values
        """
        for row in self:
            yield from filter(None, row)

    def iter_items(self):
        """
        Iterate over indices and values of nonzero elements of the matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[QQ(1), QQ(0)], [QQ(3), QQ(4)]], (2, 2), QQ)
        >>> list(A.iter_items())
        [((0, 0), 1), ((1, 0), 3), ((1, 1), 4)]

        See Also
        ========

        iter_values
        to_dok
        sympy.polys.matrices.domainmatrix.DomainMatrix.iter_items
        """
        for i, row in enumerate(self):
            for j, element in enumerate(row):
                if element:
                    yield (i, j), element

    def to_ddm(self):
        """
        Convert to a :class:`DDM`.

        This just returns ``self`` but exists to parallel the corresponding
        method in other matrix types like :class:`~.SDM`.

        See Also
        ========

        to_sdm
        to_dfm
        to_dfm_or_ddm
        sympy.polys.matrices.sdm.SDM.to_ddm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_ddm
        """
        return self

    def to_sdm(self):
        """
        Convert to a :class:`~.SDM`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_sdm()
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
        >>> type(A.to_sdm())
        <class 'sympy.polys.matrices.sdm.SDM'>

        See Also
        ========

        SDM
        sympy.polys.matrices.sdm.SDM.to_ddm
        """
        return SDM.from_list(self, self.shape, self.domain)

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm(self):
        """
        Convert to :class:`~.DDM` to :class:`~.DFM`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dfm()
        [[1, 2], [3, 4]]
        >>> type(A.to_dfm())
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        DFM
        sympy.polys.matrices._dfm.DFM.to_ddm
        """
        return DFM(list(self), self.shape, self.domain)

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm_or_ddm(self):
        """
        Convert to :class:`~.DFM` if possible or otherwise return self.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dfm_or_ddm()
        [[1, 2], [3, 4]]
        >>> type(A.to_dfm_or_ddm())
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        to_dfm
        to_ddm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm_or_ddm
        """
        if DFM._supports_domain(self.domain):
            return self.to_dfm()
        return self

    def convert_to(self, K):
        Kold = self.domain
        if K == Kold:
            return self.copy()
        rows = [[K.convert_from(e, Kold) for e in row] for row in self]
        return DDM(rows, self.shape, K)

    def __str__(self):
        rowsstr = ['[%s]' % ', '.join(map(str, row)) for row in self]
        return '[%s]' % ', '.join(rowsstr)

    def __repr__(self):
        cls = type(self).__name__
        rows = list.__repr__(self)
        return '%s(%s, %s, %s)' % (cls, rows, self.shape, self.domain)

    def __eq__(self, other):
        if not isinstance(other, DDM):
            return False
        return (super().__eq__(other) and self.domain == other.domain)

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def zeros(cls, shape, domain):
        z = domain.zero
        m, n = shape
        rowslist = [[z] * n for _ in range(m)]
        return DDM(rowslist, shape, domain)

    @classmethod
    def ones(cls, shape, domain):
        one = domain.one
        m, n = shape
        rowlist = [[one] * n for _ in range(m)]
        return DDM(rowlist, shape, domain)

    @classmethod
    def eye(cls, size, domain):
        if isinstance(size, tuple):
            m, n = size
        elif isinstance(size, int):
            m = n = size
        one = domain.one
        ddm = cls.zeros((m, n), domain)
        for i in range(min(m, n)):
            ddm[i][i] = one
        return ddm

    def copy(self):
        copyrows = [row[:] for row in self]
        return DDM(copyrows, self.shape, self.domain)

    def transpose(self):
        rows, cols = self.shape
        if rows:
            ddmT = ddm_transpose(self)
        else:
            ddmT = [[]] * cols
        return DDM(ddmT, (cols, rows), self.domain)

    def __add__(a, b):
        if not isinstance(b, DDM):
            return NotImplemented
        return a.add(b)

    def __sub__(a, b):
        if not isinstance(b, DDM):
            return NotImplemented
        return a.sub(b)

    def __neg__(a):
        return a.neg()

    def __mul__(a, b):
        if b in a.domain:
            return a.mul(b)
        else:
            return NotImplemented

    def __rmul__(a, b):
        if b in a.domain:
            return a.mul(b)
        else:
            return NotImplemented

    def __matmul__(a, b):
        if isinstance(b, DDM):
            return a.matmul(b)
        else:
            return NotImplemented

    @classmethod
    def _check(cls, a, op, b, ashape, bshape):
        if a.domain != b.domain:
            msg = "Domain mismatch: %s %s %s" % (a.domain, op, b.domain)
            raise DMDomainError(msg)
        if ashape != bshape:
            msg = "Shape mismatch: %s %s %s" % (a.shape, op, b.shape)
            raise DMShapeError(msg)

    def add(a, b):
        """a + b"""
        a._check(a, '+', b, a.shape, b.shape)
        c = a.copy()
        ddm_iadd(c, b)
        return c

    def sub(a, b):
        """a - b"""
        a._check(a, '-', b, a.shape, b.shape)
        c = a.copy()
        ddm_isub(c, b)
        return c

    def neg(a):
        """-a"""
        b = a.copy()
        ddm_ineg(b)
        return b

    def mul(a, b):
        c = a.copy()
        ddm_imul(c, b)
        return c

    def rmul(a, b):
        c = a.copy()
        ddm_irmul(c, b)
        return c

    def matmul(a, b):
        """a @ b (matrix product)"""
        m, o = a.shape
        o2, n = b.shape
        a._check(a, '*', b, o, o2)
        c = a.zeros((m, n), a.domain)
        ddm_imatmul(c, a, b)
        return c

    def mul_elementwise(a, b):
        assert a.shape == b.shape
        assert a.domain == b.domain
        c = [[aij * bij for aij, bij in zip(ai, bi)] for ai, bi in zip(a, b)]
        return DDM(c, a.shape, a.domain)

    def hstack(A, *B):
        """Horizontally stacks :py:class:`~.DDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM

        >>> A = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DDM([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
        >>> A.hstack(B)
        [[1, 2, 5, 6], [3, 4, 7, 8]]

        >>> C = DDM([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
        >>> A.hstack(B, C)
        [[1, 2, 5, 6, 9, 10], [3, 4, 7, 8, 11, 12]]
        """
        Anew = list(A.copy())
        rows, cols = A.shape
        domain = A.domain

        for Bk in B:
            Bkrows, Bkcols = Bk.shape
            assert Bkrows == rows
            assert Bk.domain == domain

            cols += Bkcols

            for i, Bki in enumerate(Bk):
                Anew[i].extend(Bki)

        return DDM(Anew, (rows, cols), A.domain)

    def vstack(A, *B):
        """Vertically stacks :py:class:`~.DDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM

        >>> A = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DDM([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
        >>> A.vstack(B)
        [[1, 2], [3, 4], [5, 6], [7, 8]]

        >>> C = DDM([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
        >>> A.vstack(B, C)
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
        """
        Anew = list(A.copy())
        rows, cols = A.shape
        domain = A.domain

        for Bk in B:
            Bkrows, Bkcols = Bk.shape
            assert Bkcols == cols
            assert Bk.domain == domain

            rows += Bkrows

            Anew.extend(Bk.copy())

        return DDM(Anew, (rows, cols), A.domain)

    def applyfunc(self, func, domain):
        elements = [list(map(func, row)) for row in self]
        return DDM(elements, self.shape, domain)

    def nnz(a):
        """Number of non-zero entries in :py:class:`~.DDM` matrix.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nnz
        """
        return sum(sum(map(bool, row)) for row in a)

    def scc(a):
        """Strongly connected components of a square matrix *a*.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM
        >>> A = DDM([[ZZ(1), ZZ(0)], [ZZ(0), ZZ(1)]], (2, 2), ZZ)
        >>> A.scc()
        [[0], [1]]

        See also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.scc

        """
        return a.to_sdm().scc()

    @classmethod
    def diag(cls, values, domain):
        """Returns a square diagonal matrix with *values* on the diagonal.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM
        >>> DDM.diag([ZZ(1), ZZ(2), ZZ(3)], ZZ)
        [[1, 0, 0], [0, 2, 0], [0, 0, 3]]

        See also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.diag
        """
        return SDM.diag(values, domain).to_ddm()

    def rref(a):
        """Reduced-row echelon form of a and list of pivots.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.rref
            Higher level interface to this function.
        sympy.polys.matrices.dense.ddm_irref
            The underlying algorithm.
        """
        b = a.copy()
        K = a.domain
        partial_pivot = K.is_RealField or K.is_ComplexField
        pivots = ddm_irref(b, _partial_pivot=partial_pivot)
        return b, pivots

    def rref_den(a):
        """Reduced-row echelon form of a with denominator and list of pivots

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den
            Higher level interface to this function.
        sympy.polys.matrices.dense.ddm_irref_den
            The underlying algorithm.
        """
        b = a.copy()
        K = a.domain
        denom, pivots = ddm_irref_den(b, K)
        return b, denom, pivots

    def nullspace(a):
        """Returns a basis for the nullspace of a.

        The domain of the matrix must be a field.

        See Also
        ========

        rref
        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
        """
        rref, pivots = a.rref()
        return rref.nullspace_from_rref(pivots)

    def nullspace_from_rref(a, pivots=None):
        """Compute the nullspace of a matrix from its rref.

        The domain of the matrix can be any domain.

        Returns a tuple (basis, nonpivots).

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
            The higher level interface to this function.
        """
        m, n = a.shape
        K = a.domain

        if pivots is None:
            pivots = []
            last_pivot = -1
            for i in range(m):
                ai = a[i]
                for j in range(last_pivot+1, n):
                    if ai[j]:
                        last_pivot = j
                        pivots.append(j)
                        break

        if not pivots:
            return (a.eye(n, K), list(range(n)))

        # After rref the pivots are all one but after rref_den they may not be.
        pivot_val = a[0][pivots[0]]

        basis = []
        nonpivots = []
        for i in range(n):
            if i in pivots:
                continue
            nonpivots.append(i)
            vec = [pivot_val if i == j else K.zero for j in range(n)]
            for ii, jj in enumerate(pivots):
                vec[jj] -= a[ii][i]
            basis.append(vec)

        basis_ddm = DDM(basis, (len(basis), n), K)

        return (basis_ddm, nonpivots)

    def particular(a):
        return a.to_sdm().particular().to_ddm()

    def det(a):
        """Determinant of a"""
        m, n = a.shape
        if m != n:
            raise DMNonSquareMatrixError("Determinant of non-square matrix")
        b = a.copy()
        K = b.domain
        deta = ddm_idet(b, K)
        return deta

    def inv(a):
        """Inverse of a"""
        m, n = a.shape
        if m != n:
            raise DMNonSquareMatrixError("Determinant of non-square matrix")
        ainv = a.copy()
        K = a.domain
        ddm_iinv(ainv, a, K)
        return ainv

    def lu(a):
        """L, U decomposition of a"""
        m, n = a.shape
        K = a.domain

        U = a.copy()
        L = a.eye(m, K)
        swaps = ddm_ilu_split(L, U, K)

        return L, U, swaps

    def _fflu(self):
        """
        Private method for Phase 1 of fraction-free LU decomposition.
        Performs row operations and elimination to compute U and permutation indices.

        Returns:
            LU : decomposition as a single matrix.
            perm (list): Permutation indices for row swaps.
        """
        rows, cols = self.shape
        K = self.domain

        LU = self.copy()
        perm = list(range(rows))
        rank = 0

        for j in range(min(rows, cols)):
            # Skip columns where all entries are zero
            if all(LU[i][j] == K.zero for i in range(rows)):
                continue

            # Find the first non-zero pivot in the current column
            pivot_row = -1
            for i in range(rank, rows):
                if LU[i][j] != K.zero:
                    pivot_row = i
                    break

            # If no pivot is found, skip column
            if pivot_row == -1:
                continue

            # Swap rows to bring the pivot to the current rank
            if pivot_row != rank:
                LU[rank], LU[pivot_row] = LU[pivot_row], LU[rank]
                perm[rank], perm[pivot_row] = perm[pivot_row], perm[rank]

            # Found pivot - (Gauss-Bareiss elimination)
            pivot = LU[rank][j]
            for i in range(rank + 1, rows):
                multiplier = LU[i][j]
                # Denominator is previous pivot or 1
                denominator = LU[rank - 1][rank - 1] if rank > 0 else K.one
                for k in range(j + 1, cols):
                    LU[i][k] = K.exquo(pivot * LU[i][k] - LU[rank][k] * multiplier, denominator)
                # Keep the multiplier for L matrix
                LU[i][j] = multiplier
            rank += 1

        return LU, perm

    def fflu(self):
        """
        Fraction-free LU decomposition of DDM.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.fflu
            The higher-level interface to this function.
        """
        rows, cols = self.shape
        K = self.domain

        # Phase 1: Perform row operations and get permutation
        U, perm = self._fflu()

        # Phase 2: Construct P, L, D matrices
        # Create P from permutation
        P = self.zeros((rows, rows), K)
        for i, pi in enumerate(perm):
            P[i][pi] = K.one

        # Create L matrix
        L = self.zeros((rows, rows), K)
        i = j = 0
        while i < rows and j < cols:
            if U[i][j] != K.zero:
                # Found non-zero pivot
                # Diagonal entry is the pivot
                L[i][i] = U[i][j]
                for l in range(i + 1, rows):
                    # Off-diagonal entries are the multipliers
                    L[l][i] = U[l][j]
                    # zero out the entries in U
                    U[l][j] = K.zero
                i += 1
            j += 1

        # Fill remaining diagonal of L with ones
        for i in range(i, rows):
            L[i][i] = K.one

        # Create D matrix - using FLINT's approach with accumulator
        D = self.zeros((rows, rows), K)
        if rows >= 1:
            D[0][0] = L[0][0]
        di = K.one
        for i in range(1, rows):
            # Accumulate product of pivots
            di = L[i - 1][i - 1] * L[i][i]
            D[i][i] = di

        return P, L, D, U

    def qr(self):
        """
        QR decomposition for DDM.

        Returns:
            - Q: Orthogonal matrix as a DDM.
            - R: Upper triangular matrix as a DDM.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.qr
            The higher-level interface to this function.
        """
        rows, cols = self.shape
        K = self.domain
        Q = self.copy()
        R = self.zeros((min(rows, cols), cols), K)

        # Check that the domain is a field
        if not K.is_Field:
            raise DMDomainError("QR decomposition requires a field (e.g. QQ).")

        dot_cols = lambda i, j: K.sum(Q[k][i] * Q[k][j] for k in range(rows))

        for j in range(cols):
            for i in range(min(j, rows)):
                dot_ii = dot_cols(i, i)
                if dot_ii != K.zero:
                    R[i][j] = dot_cols(i, j) / dot_ii
                    for k in range(rows):
                        Q[k][j] -= R[i][j] * Q[k][i]

            if j < rows:
                dot_jj = dot_cols(j, j)
                if dot_jj != K.zero:
                    R[j][j] = K.one

        Q = Q.extract(range(rows), range(min(rows, cols)))

        return Q, R

    def lu_solve(a, b):
        """x where a*x = b"""
        m, n = a.shape
        m2, o = b.shape
        a._check(a, 'lu_solve', b, m, m2)
        if not a.domain.is_Field:
            raise DMDomainError("lu_solve requires a field")

        L, U, swaps = a.lu()
        x = a.zeros((n, o), a.domain)
        ddm_ilu_solve(x, L, U, swaps, b)
        return x

    def charpoly(a):
        """Coefficients of characteristic polynomial of a"""
        K = a.domain
        m, n = a.shape
        if m != n:
            raise DMNonSquareMatrixError("Charpoly of non-square matrix")
        vec = ddm_berk(a, K)
        coeffs = [vec[i][0] for i in range(n+1)]
        return coeffs

    def is_zero_matrix(self):
        """
        Says whether this matrix has all zero entries.
        """
        zero = self.domain.zero
        return all(Mij == zero for Mij in self.flatiter())

    def is_upper(self):
        """
        Says whether this matrix is upper-triangular. True can be returned
        even if the matrix is not square.
        """
        zero = self.domain.zero
        return all(Mij == zero for i, Mi in enumerate(self) for Mij in Mi[:i])

    def is_lower(self):
        """
        Says whether this matrix is lower-triangular. True can be returned
        even if the matrix is not square.
        """
        zero = self.domain.zero
        return all(Mij == zero for i, Mi in enumerate(self) for Mij in Mi[i+1:])

    def is_diagonal(self):
        """
        Says whether this matrix is diagonal. True can be returned even if
        the matrix is not square.
        """
        return self.is_upper() and self.is_lower()

    def diagonal(self):
        """
        Returns a list of the elements from the diagonal of the matrix.
        """
        m, n = self.shape
        return [self[i][i] for i in range(min(m, n))]

    def lll(A, delta=QQ(3, 4)):
        return ddm_lll(A, delta=delta)

    def lll_transform(A, delta=QQ(3, 4)):
        return ddm_lll_transform(A, delta=delta)


from .sdm import SDM
from .dfm import DFM

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\__init__.py ===
"""Beautiful Soup Elixir and Tonic - "The Screen-Scraper's Friend".

http://www.crummy.com/software/BeautifulSoup/

Beautiful Soup uses a pluggable XML or HTML parser to parse a
(possibly invalid) document into a tree representation. Beautiful Soup
provides methods and Pythonic idioms that make it easy to navigate,
search, and modify the parse tree.

Beautiful Soup works with Python 3.7 and up. It works better if lxml
and/or html5lib is installed, but they are not required.

For more than you ever wanted to know about Beautiful Soup, see the
documentation: http://www.crummy.com/software/BeautifulSoup/bs4/doc/
"""

__author__ = "Leonard Richardson (leonardr@segfault.org)"
__version__ = "4.13.4"
__copyright__ = "Copyright (c) 2004-2025 Leonard Richardson"
# Use of this source code is governed by the MIT license.
__license__ = "MIT"

__all__ = [
    "AttributeResemblesVariableWarning",
    "BeautifulSoup",
    "Comment",
    "Declaration",
    "ProcessingInstruction",
    "ResultSet",
    "CSS",
    "Script",
    "Stylesheet",
    "Tag",
    "TemplateString",
    "ElementFilter",
    "UnicodeDammit",
    "CData",
    "Doctype",

    # Exceptions
    "FeatureNotFound",
    "ParserRejectedMarkup",
    "StopParsing",

    # Warnings
    "AttributeResemblesVariableWarning",
    "GuessedAtParserWarning",
    "MarkupResemblesLocatorWarning",
    "UnusualUsageWarning",
    "XMLParsedAsHTMLWarning",
]

from collections import Counter
import sys
import warnings

# The very first thing we do is give a useful error if someone is
# running this code under Python 2.
if sys.version_info.major < 3:
    raise ImportError(
        "You are trying to use a Python 3-specific version of Beautiful Soup under Python 2. This will not work. The final version of Beautiful Soup to support Python 2 was 4.9.3."
    )

from .builder import (
    builder_registry,
    TreeBuilder,
)
from .builder._htmlparser import HTMLParserTreeBuilder
from .dammit import UnicodeDammit
from .css import CSS
from ._deprecation import (
    _deprecated,
)
from .element import (
    CData,
    Comment,
    DEFAULT_OUTPUT_ENCODING,
    Declaration,
    Doctype,
    NavigableString,
    PageElement,
    ProcessingInstruction,
    PYTHON_SPECIFIC_ENCODINGS,
    ResultSet,
    Script,
    Stylesheet,
    Tag,
    TemplateString,
)
from .formatter import Formatter
from .filter import (
    ElementFilter,
    SoupStrainer,
)
from typing import (
    Any,
    cast,
    Counter as CounterType,
    Dict,
    Iterator,
    List,
    Sequence,
    Optional,
    Type,
    Union,
)

from bs4._typing import (
    _Encoding,
    _Encodings,
    _IncomingMarkup,
    _InsertableElement,
    _RawAttributeValue,
    _RawAttributeValues,
    _RawMarkup,
)

# Import all warnings and exceptions into the main package.
from bs4.exceptions import (
    FeatureNotFound,
    ParserRejectedMarkup,
    StopParsing,
)
from bs4._warnings import (
    AttributeResemblesVariableWarning,
    GuessedAtParserWarning,
    MarkupResemblesLocatorWarning,
    UnusualUsageWarning,
    XMLParsedAsHTMLWarning,
)


class BeautifulSoup(Tag):
    """A data structure representing a parsed HTML or XML document.

    Most of the methods you'll call on a BeautifulSoup object are inherited from
    PageElement or Tag.

    Internally, this class defines the basic interface called by the
    tree builders when converting an HTML/XML document into a data
    structure. The interface abstracts away the differences between
    parsers. To write a new tree builder, you'll need to understand
    these methods as a whole.

    These methods will be called by the BeautifulSoup constructor:
      * reset()
      * feed(markup)

    The tree builder may call these methods from its feed() implementation:
      * handle_starttag(name, attrs) # See note about return value
      * handle_endtag(name)
      * handle_data(data) # Appends to the current data node
      * endData(containerClass) # Ends the current data node

    No matter how complicated the underlying parser is, you should be
    able to build a tree using 'start tag' events, 'end tag' events,
    'data' events, and "done with data" events.

    If you encounter an empty-element tag (aka a self-closing tag,
    like HTML's <br> tag), call handle_starttag and then
    handle_endtag.
    """

    #: Since `BeautifulSoup` subclasses `Tag`, it's possible to treat it as
    #: a `Tag` with a `Tag.name`. Hoever, this name makes it clear the
    #: `BeautifulSoup` object isn't a real markup tag.
    ROOT_TAG_NAME: str = "[document]"

    #: If the end-user gives no indication which tree builder they
    #: want, look for one with these features.
    DEFAULT_BUILDER_FEATURES: Sequence[str] = ["html", "fast"]

    #: A string containing all ASCII whitespace characters, used in
    #: during parsing to detect data chunks that seem 'empty'.
    ASCII_SPACES: str = "\x20\x0a\x09\x0c\x0d"

    # FUTURE PYTHON:
    element_classes: Dict[Type[PageElement], Type[PageElement]]  #: :meta private:
    builder: TreeBuilder  #: :meta private:
    is_xml: bool
    known_xml: Optional[bool]
    parse_only: Optional[SoupStrainer]  #: :meta private:

    # These members are only used while parsing markup.
    markup: Optional[_RawMarkup]  #: :meta private:
    current_data: List[str]  #: :meta private:
    currentTag: Optional[Tag]  #: :meta private:
    tagStack: List[Tag]  #: :meta private:
    open_tag_counter: CounterType[str]  #: :meta private:
    preserve_whitespace_tag_stack: List[Tag]  #: :meta private:
    string_container_stack: List[Tag]  #: :meta private:
    _most_recent_element: Optional[PageElement]  #: :meta private:

    #: Beautiful Soup's best guess as to the character encoding of the
    #: original document.
    original_encoding: Optional[_Encoding]

    #: The character encoding, if any, that was explicitly defined
    #: in the original document. This may or may not match
    #: `BeautifulSoup.original_encoding`.
    declared_html_encoding: Optional[_Encoding]

    #: This is True if the markup that was parsed contains
    #: U+FFFD REPLACEMENT_CHARACTER characters which were not present
    #: in the original markup. These mark character sequences that
    #: could not be represented in Unicode.
    contains_replacement_characters: bool

    def __init__(
        self,
        markup: _IncomingMarkup = "",
        features: Optional[Union[str, Sequence[str]]] = None,
        builder: Optional[Union[TreeBuilder, Type[TreeBuilder]]] = None,
        parse_only: Optional[SoupStrainer] = None,
        from_encoding: Optional[_Encoding] = None,
        exclude_encodings: Optional[_Encodings] = None,
        element_classes: Optional[Dict[Type[PageElement], Type[PageElement]]] = None,
        **kwargs: Any,
    ):
        """Constructor.

        :param markup: A string or a file-like object representing
         markup to be parsed.

        :param features: Desirable features of the parser to be
         used. This may be the name of a specific parser ("lxml",
         "lxml-xml", "html.parser", or "html5lib") or it may be the
         type of markup to be used ("html", "html5", "xml"). It's
         recommended that you name a specific parser, so that
         Beautiful Soup gives you the same results across platforms
         and virtual environments.

        :param builder: A TreeBuilder subclass to instantiate (or
         instance to use) instead of looking one up based on
         `features`. You only need to use this if you've implemented a
         custom TreeBuilder.

        :param parse_only: A SoupStrainer. Only parts of the document
         matching the SoupStrainer will be considered. This is useful
         when parsing part of a document that would otherwise be too
         large to fit into memory.

        :param from_encoding: A string indicating the encoding of the
         document to be parsed. Pass this in if Beautiful Soup is
         guessing wrongly about the document's encoding.

        :param exclude_encodings: A list of strings indicating
         encodings known to be wrong. Pass this in if you don't know
         the document's encoding but you know Beautiful Soup's guess is
         wrong.

        :param element_classes: A dictionary mapping BeautifulSoup
         classes like Tag and NavigableString, to other classes you'd
         like to be instantiated instead as the parse tree is
         built. This is useful for subclassing Tag or NavigableString
         to modify default behavior.

        :param kwargs: For backwards compatibility purposes, the
         constructor accepts certain keyword arguments used in
         Beautiful Soup 3. None of these arguments do anything in
         Beautiful Soup 4; they will result in a warning and then be
         ignored.

         Apart from this, any keyword arguments passed into the
         BeautifulSoup constructor are propagated to the TreeBuilder
         constructor. This makes it possible to configure a
         TreeBuilder by passing in arguments, not just by saying which
         one to use.
        """
        if "convertEntities" in kwargs:
            del kwargs["convertEntities"]
            warnings.warn(
                "BS4 does not respect the convertEntities argument to the "
                "BeautifulSoup constructor. Entities are always converted "
                "to Unicode characters."
            )

        if "markupMassage" in kwargs:
            del kwargs["markupMassage"]
            warnings.warn(
                "BS4 does not respect the markupMassage argument to the "
                "BeautifulSoup constructor. The tree builder is responsible "
                "for any necessary markup massage."
            )

        if "smartQuotesTo" in kwargs:
            del kwargs["smartQuotesTo"]
            warnings.warn(
                "BS4 does not respect the smartQuotesTo argument to the "
                "BeautifulSoup constructor. Smart quotes are always converted "
                "to Unicode characters."
            )

        if "selfClosingTags" in kwargs:
            del kwargs["selfClosingTags"]
            warnings.warn(
                "Beautiful Soup 4 does not respect the selfClosingTags argument to the "
                "BeautifulSoup constructor. The tree builder is responsible "
                "for understanding self-closing tags."
            )

        if "isHTML" in kwargs:
            del kwargs["isHTML"]
            warnings.warn(
                "Beautiful Soup 4 does not respect the isHTML argument to the "
                "BeautifulSoup constructor. Suggest you use "
                "features='lxml' for HTML and features='lxml-xml' for "
                "XML."
            )

        def deprecated_argument(old_name: str, new_name: str) -> Optional[Any]:
            if old_name in kwargs:
                warnings.warn(
                    'The "%s" argument to the BeautifulSoup constructor '
                    'was renamed to "%s" in Beautiful Soup 4.0.0'
                    % (old_name, new_name),
                    DeprecationWarning,
                    stacklevel=3,
                )
                return kwargs.pop(old_name)
            return None

        parse_only = parse_only or deprecated_argument("parseOnlyThese", "parse_only")
        if parse_only is not None:
            # Issue a warning if we can tell in advance that
            # parse_only will exclude the entire tree.
            if parse_only.excludes_everything:
                warnings.warn(
                    f"The given value for parse_only will exclude everything: {parse_only}",
                    UserWarning,
                    stacklevel=3,
                )

        from_encoding = from_encoding or deprecated_argument(
            "fromEncoding", "from_encoding"
        )

        if from_encoding and isinstance(markup, str):
            warnings.warn(
                "You provided Unicode markup but also provided a value for from_encoding. Your from_encoding will be ignored."
            )
            from_encoding = None

        self.element_classes = element_classes or dict()

        # We need this information to track whether or not the builder
        # was specified well enough that we can omit the 'you need to
        # specify a parser' warning.
        original_builder = builder
        original_features = features

        builder_class: Type[TreeBuilder]
        if isinstance(builder, type):
            # A builder class was passed in; it needs to be instantiated.
            builder_class = builder
            builder = None
        elif builder is None:
            if isinstance(features, str):
                features = [features]
            if features is None or len(features) == 0:
                features = self.DEFAULT_BUILDER_FEATURES
            possible_builder_class = builder_registry.lookup(*features)
            if possible_builder_class is None:
                raise FeatureNotFound(
                    "Couldn't find a tree builder with the features you "
                    "requested: %s. Do you need to install a parser library?"
                    % ",".join(features)
                )
            builder_class = possible_builder_class

        # At this point either we have a TreeBuilder instance in
        # builder, or we have a builder_class that we can instantiate
        # with the remaining **kwargs.
        if builder is None:
            builder = builder_class(**kwargs)
            if (
                not original_builder
                and not (
                    original_features == builder.NAME
                    or (
                        isinstance(original_features, str)
                        and original_features in builder.ALTERNATE_NAMES
                    )
                )
                and markup
            ):
                # The user did not tell us which TreeBuilder to use,
                # and we had to guess. Issue a warning.
                if builder.is_xml:
                    markup_type = "XML"
                else:
                    markup_type = "HTML"

                # This code adapted from warnings.py so that we get the same line
                # of code as our warnings.warn() call gets, even if the answer is wrong
                # (as it may be in a multithreading situation).
                caller = None
                try:
                    caller = sys._getframe(1)
                except ValueError:
                    pass
                if caller:
                    globals = caller.f_globals
                    line_number = caller.f_lineno
                else:
                    globals = sys.__dict__
                    line_number = 1
                filename = globals.get("__file__")
                if filename:
                    fnl = filename.lower()
                    if fnl.endswith((".pyc", ".pyo")):
                        filename = filename[:-1]
                if filename:
                    # If there is no filename at all, the user is most likely in a REPL,
                    # and the warning is not necessary.
                    values = dict(
                        filename=filename,
                        line_number=line_number,
                        parser=builder.NAME,
                        markup_type=markup_type,
                    )
                    warnings.warn(
                        GuessedAtParserWarning.MESSAGE % values,
                        GuessedAtParserWarning,
                        stacklevel=2,
                    )
        else:
            if kwargs:
                warnings.warn(
                    "Keyword arguments to the BeautifulSoup constructor will be ignored. These would normally be passed into the TreeBuilder constructor, but a TreeBuilder instance was passed in as `builder`."
                )

        self.builder = builder
        self.is_xml = builder.is_xml
        self.known_xml = self.is_xml
        self._namespaces = dict()
        self.parse_only = parse_only

        if hasattr(markup, "read"):  # It's a file-type object.
            markup = markup.read()
        elif not isinstance(markup, (bytes, str)) and not hasattr(markup, "__len__"):
            raise TypeError(
                f"Incoming markup is of an invalid type: {markup!r}. Markup must be a string, a bytestring, or an open filehandle."
            )
        elif len(markup) <= 256 and (
            (isinstance(markup, bytes) and b"<" not in markup and b"\n" not in markup)
            or (isinstance(markup, str) and "<" not in markup and "\n" not in markup)
        ):
            # Issue warnings for a couple beginner problems
            # involving passing non-markup to Beautiful Soup.
            # Beautiful Soup will still parse the input as markup,
            # since that is sometimes the intended behavior.
            if not self._markup_is_url(markup):
                self._markup_resembles_filename(markup)

        # At this point we know markup is a string or bytestring.  If
        # it was a file-type object, we've read from it.
        markup = cast(_RawMarkup, markup)

        rejections = []
        success = False
        for (
            self.markup,
            self.original_encoding,
            self.declared_html_encoding,
            self.contains_replacement_characters,
        ) in self.builder.prepare_markup(
            markup, from_encoding, exclude_encodings=exclude_encodings
        ):
            self.reset()
            self.builder.initialize_soup(self)
            try:
                self._feed()
                success = True
                break
            except ParserRejectedMarkup as e:
                rejections.append(e)
                pass

        if not success:
            other_exceptions = [str(e) for e in rejections]
            raise ParserRejectedMarkup(
                "The markup you provided was rejected by the parser. Trying a different parser or a different encoding may help.\n\nOriginal exception(s) from parser:\n "
                + "\n ".join(other_exceptions)
            )

        # Clear out the markup and remove the builder's circular
        # reference to this object.
        self.markup = None
        self.builder.soup = None

    def copy_self(self) -> "BeautifulSoup":
        """Create a new BeautifulSoup object with the same TreeBuilder,
        but not associated with any markup.

        This is the first step of the deepcopy process.
        """
        clone = type(self)("", None, self.builder)

        # Keep track of the encoding of the original document,
        # since we won't be parsing it again.
        clone.original_encoding = self.original_encoding
        return clone

    def __getstate__(self) -> Dict[str, Any]:
        # Frequently a tree builder can't be pickled.
        d = dict(self.__dict__)
        if "builder" in d and d["builder"] is not None and not self.builder.picklable:
            d["builder"] = type(self.builder)
        # Store the contents as a Unicode string.
        d["contents"] = []
        d["markup"] = self.decode()

        # If _most_recent_element is present, it's a Tag object left
        # over from initial parse. It might not be picklable and we
        # don't need it.
        if "_most_recent_element" in d:
            del d["_most_recent_element"]
        return d

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # If necessary, restore the TreeBuilder by looking it up.
        self.__dict__ = state
        if isinstance(self.builder, type):
            self.builder = self.builder()
        elif not self.builder:
            # We don't know which builder was used to build this
            # parse tree, so use a default we know is always available.
            self.builder = HTMLParserTreeBuilder()
        self.builder.soup = self
        self.reset()
        self._feed()

    @classmethod
    @_deprecated(
        replaced_by="nothing (private method, will be removed)", version="4.13.0"
    )
    def _decode_markup(cls, markup: _RawMarkup) -> str:
        """Ensure `markup` is Unicode so it's safe to send into warnings.warn.

        warnings.warn had this problem back in 2010 but fortunately
        not anymore. This has not been used for a long time; I just
        noticed that fact while working on 4.13.0.
        """
        if isinstance(markup, bytes):
            decoded = markup.decode("utf-8", "replace")
        else:
            decoded = markup
        return decoded

    @classmethod
    def _markup_is_url(cls, markup: _RawMarkup) -> bool:
        """Error-handling method to raise a warning if incoming markup looks
        like a URL.

        :param markup: A string of markup.
        :return: Whether or not the markup resembled a URL
            closely enough to justify issuing a warning.
        """
        problem: bool = False
        if isinstance(markup, bytes):
            problem = (
                any(markup.startswith(prefix) for prefix in (b"http:", b"https:"))
                and b" " not in markup
            )
        elif isinstance(markup, str):
            problem = (
                any(markup.startswith(prefix) for prefix in ("http:", "https:"))
                and " " not in markup
            )
        else:
            return False

        if not problem:
            return False
        warnings.warn(
            MarkupResemblesLocatorWarning.URL_MESSAGE % dict(what="URL"),
            MarkupResemblesLocatorWarning,
            stacklevel=3,
        )
        return True

    @classmethod
    def _markup_resembles_filename(cls, markup: _RawMarkup) -> bool:
        """Error-handling method to issue a warning if incoming markup
        resembles a filename.

        :param markup: A string of markup.
        :return: Whether or not the markup resembled a filename
            closely enough to justify issuing a warning.
        """
        markup_b: bytes

        # We're only checking ASCII characters, so rather than write
        # the same tests twice, convert Unicode to a bytestring and
        # operate on the bytestring.
        if isinstance(markup, str):
            markup_b = markup.encode("utf8")
        else:
            markup_b = markup

        # Step 1: does it end with a common textual file extension?
        filelike = False
        lower = markup_b.lower()
        extensions = [b".html", b".htm", b".xml", b".xhtml", b".txt"]
        if any(lower.endswith(ext) for ext in extensions):
            filelike = True
        if not filelike:
            return False

        # Step 2: it _might_ be a file, but there are a few things
        # we can look for that aren't very common in filenames.

        # Characters that have special meaning to Unix shells. (< was
        # excluded before this method was called.)
        #
        # Many of these are also reserved characters that cannot
        # appear in Windows filenames.
        for byte in markup_b:
            if byte in b"?*#&;>$|":
                return False

        # Two consecutive forward slashes (as seen in a URL) or two
        # consecutive spaces (as seen in fixed-width data).
        #
        # (Paths to Windows network shares contain consecutive
        #  backslashes, so checking that doesn't seem as helpful.)
        if b"//" in markup_b:
            return False
        if b"  " in markup_b:
            return False

        # A colon in any position other than position 1 (e.g. after a
        # Windows drive letter).
        if markup_b.startswith(b":"):
            return False
        colon_i = markup_b.rfind(b":")
        if colon_i not in (-1, 1):
            return False

        # Step 3: If it survived all of those checks, it's similar
        # enough to a file to justify issuing a warning.
        warnings.warn(
            MarkupResemblesLocatorWarning.FILENAME_MESSAGE % dict(what="filename"),
            MarkupResemblesLocatorWarning,
            stacklevel=3,
        )
        return True

    def _feed(self) -> None:
        """Internal method that parses previously set markup, creating a large
        number of Tag and NavigableString objects.
        """
        # Convert the document to Unicode.
        self.builder.reset()

        if self.markup is not None:
            self.builder.feed(self.markup)
        # Close out any unfinished strings and close all the open tags.
        self.endData()
        while (
            self.currentTag is not None and self.currentTag.name != self.ROOT_TAG_NAME
        ):
            self.popTag()

    def reset(self) -> None:
        """Reset this object to a state as though it had never parsed any
        markup.
        """
        Tag.__init__(self, self, self.builder, self.ROOT_TAG_NAME)
        self.hidden = True
        self.builder.reset()
        self.current_data = []
        self.currentTag = None
        self.tagStack = []
        self.open_tag_counter = Counter()
        self.preserve_whitespace_tag_stack = []
        self.string_container_stack = []
        self._most_recent_element = None
        self.pushTag(self)

    def new_tag(
        self,
        name: str,
        namespace: Optional[str] = None,
        nsprefix: Optional[str] = None,
        attrs: Optional[_RawAttributeValues] = None,
        sourceline: Optional[int] = None,
        sourcepos: Optional[int] = None,
        string: Optional[str] = None,
        **kwattrs: _RawAttributeValue,
    ) -> Tag:
        """Create a new Tag associated with this BeautifulSoup object.

        :param name: The name of the new Tag.
        :param namespace: The URI of the new Tag's XML namespace, if any.
        :param prefix: The prefix for the new Tag's XML namespace, if any.
        :param attrs: A dictionary of this Tag's attribute values; can
            be used instead of ``kwattrs`` for attributes like 'class'
            that are reserved words in Python.
        :param sourceline: The line number where this tag was
            (purportedly) found in its source document.
        :param sourcepos: The character position within ``sourceline`` where this
            tag was (purportedly) found.
        :param string: String content for the new Tag, if any.
        :param kwattrs: Keyword arguments for the new Tag's attribute values.

        """
        attr_container = self.builder.attribute_dict_class(**kwattrs)
        if attrs is not None:
            attr_container.update(attrs)
        tag_class = self.element_classes.get(Tag, Tag)

        # Assume that this is either Tag or a subclass of Tag. If not,
        # the user brought type-unsafety upon themselves.
        tag_class = cast(Type[Tag], tag_class)
        tag = tag_class(
            None,
            self.builder,
            name,
            namespace,
            nsprefix,
            attr_container,
            sourceline=sourceline,
            sourcepos=sourcepos,
        )

        if string is not None:
            tag.string = string
        return tag

    def string_container(
        self, base_class: Optional[Type[NavigableString]] = None
    ) -> Type[NavigableString]:
        """Find the class that should be instantiated to hold a given kind of
        string.

        This may be a built-in Beautiful Soup class or a custom class passed
        in to the BeautifulSoup constructor.
        """
        container = base_class or NavigableString

        # The user may want us to use some other class (hopefully a
        # custom subclass) instead of the one we'd use normally.
        container = cast(
            Type[NavigableString], self.element_classes.get(container, container)
        )

        # On top of that, we may be inside a tag that needs a special
        # container class.
        if self.string_container_stack and container is NavigableString:
            container = self.builder.string_containers.get(
                self.string_container_stack[-1].name, container
            )
        return container

    def new_string(
        self, s: str, subclass: Optional[Type[NavigableString]] = None
    ) -> NavigableString:
        """Create a new `NavigableString` associated with this `BeautifulSoup`
        object.

        :param s: The string content of the `NavigableString`
        :param subclass: The subclass of `NavigableString`, if any, to
               use. If a document is being processed, an appropriate
               subclass for the current location in the document will
               be determined automatically.
        """
        container = self.string_container(subclass)
        return container(s)

    def insert_before(self, *args: _InsertableElement) -> List[PageElement]:
        """This method is part of the PageElement API, but `BeautifulSoup` doesn't implement
        it because there is nothing before or after it in the parse tree.
        """
        raise NotImplementedError(
            "BeautifulSoup objects don't support insert_before()."
        )

    def insert_after(self, *args: _InsertableElement) -> List[PageElement]:
        """This method is part of the PageElement API, but `BeautifulSoup` doesn't implement
        it because there is nothing before or after it in the parse tree.
        """
        raise NotImplementedError("BeautifulSoup objects don't support insert_after().")

    def popTag(self) -> Optional[Tag]:
        """Internal method called by _popToTag when a tag is closed.

        :meta private:
        """
        if not self.tagStack:
            # Nothing to pop. This shouldn't happen.
            return None
        tag = self.tagStack.pop()
        if tag.name in self.open_tag_counter:
            self.open_tag_counter[tag.name] -= 1
        if (
            self.preserve_whitespace_tag_stack
            and tag == self.preserve_whitespace_tag_stack[-1]
        ):
            self.preserve_whitespace_tag_stack.pop()
        if self.string_container_stack and tag == self.string_container_stack[-1]:
            self.string_container_stack.pop()
        # print("Pop", tag.name)
        if self.tagStack:
            self.currentTag = self.tagStack[-1]
        return self.currentTag

    def pushTag(self, tag: Tag) -> None:
        """Internal method called by handle_starttag when a tag is opened.

        :meta private:
        """
        # print("Push", tag.name)
        if self.currentTag is not None:
            self.currentTag.contents.append(tag)
        self.tagStack.append(tag)
        self.currentTag = self.tagStack[-1]
        if tag.name != self.ROOT_TAG_NAME:
            self.open_tag_counter[tag.name] += 1
        if tag.name in self.builder.preserve_whitespace_tags:
            self.preserve_whitespace_tag_stack.append(tag)
        if tag.name in self.builder.string_containers:
            self.string_container_stack.append(tag)

    def endData(self, containerClass: Optional[Type[NavigableString]] = None) -> None:
        """Method called by the TreeBuilder when the end of a data segment
        occurs.

        :param containerClass: The class to use when incorporating the
        data segment into the parse tree.

        :meta private:
        """
        if self.current_data:
            current_data = "".join(self.current_data)
            # If whitespace is not preserved, and this string contains
            # nothing but ASCII spaces, replace it with a single space
            # or newline.
            if not self.preserve_whitespace_tag_stack:
                strippable = True
                for i in current_data:
                    if i not in self.ASCII_SPACES:
                        strippable = False
                        break
                if strippable:
                    if "\n" in current_data:
                        current_data = "\n"
                    else:
                        current_data = " "

            # Reset the data collector.
            self.current_data = []

            # Should we add this string to the tree at all?
            if (
                self.parse_only
                and len(self.tagStack) <= 1
                and (not self.parse_only.allow_string_creation(current_data))
            ):
                return

            containerClass = self.string_container(containerClass)
            o = containerClass(current_data)
            self.object_was_parsed(o)

    def object_was_parsed(
        self,
        o: PageElement,
        parent: Optional[Tag] = None,
        most_recent_element: Optional[PageElement] = None,
    ) -> None:
        """Method called by the TreeBuilder to integrate an object into the
        parse tree.

        :meta private:
        """
        if parent is None:
            parent = self.currentTag
        assert parent is not None
        previous_element: Optional[PageElement]
        if most_recent_element is not None:
            previous_element = most_recent_element
        else:
            previous_element = self._most_recent_element

        next_element = previous_sibling = next_sibling = None
        if isinstance(o, Tag):
            next_element = o.next_element
            next_sibling = o.next_sibling
            previous_sibling = o.previous_sibling
            if previous_element is None:
                previous_element = o.previous_element

        fix = parent.next_element is not None

        o.setup(parent, previous_element, next_element, previous_sibling, next_sibling)

        self._most_recent_element = o
        parent.contents.append(o)

        # Check if we are inserting into an already parsed node.
        if fix:
            self._linkage_fixer(parent)

    def _linkage_fixer(self, el: Tag) -> None:
        """Make sure linkage of this fragment is sound."""

        first = el.contents[0]
        child = el.contents[-1]
        descendant: PageElement = child

        if child is first and el.parent is not None:
            # Parent should be linked to first child
            el.next_element = child
            # We are no longer linked to whatever this element is
            prev_el = child.previous_element
            if prev_el is not None and prev_el is not el:
                prev_el.next_element = None
            # First child should be linked to the parent, and no previous siblings.
            child.previous_element = el
            child.previous_sibling = None

        # We have no sibling as we've been appended as the last.
        child.next_sibling = None

        # This index is a tag, dig deeper for a "last descendant"
        if isinstance(child, Tag) and child.contents:
            # _last_decendant is typed as returning Optional[PageElement],
            # but the value can't be None here, because el is a Tag
            # which we know has contents.
            descendant = cast(PageElement, child._last_descendant(False))

        # As the final step, link last descendant. It should be linked
        # to the parent's next sibling (if found), else walk up the chain
        # and find a parent with a sibling. It should have no next sibling.
        descendant.next_element = None
        descendant.next_sibling = None

        target: Optional[Tag] = el
        while True:
            if target is None:
                break
            elif target.next_sibling is not None:
                descendant.next_element = target.next_sibling
                target.next_sibling.previous_element = child
                break
            target = target.parent

    def _popToTag(
        self, name: str, nsprefix: Optional[str] = None, inclusivePop: bool = True
    ) -> Optional[Tag]:
        """Pops the tag stack up to and including the most recent
        instance of the given tag.

        If there are no open tags with the given name, nothing will be
        popped.

        :param name: Pop up to the most recent tag with this name.
        :param nsprefix: The namespace prefix that goes with `name`.
        :param inclusivePop: It this is false, pops the tag stack up
          to but *not* including the most recent instqance of the
          given tag.

        :meta private:
        """
        # print("Popping to %s" % name)
        if name == self.ROOT_TAG_NAME:
            # The BeautifulSoup object itself can never be popped.
            return None

        most_recently_popped = None

        stack_size = len(self.tagStack)
        for i in range(stack_size - 1, 0, -1):
            if not self.open_tag_counter.get(name):
                break
            t = self.tagStack[i]
            if name == t.name and nsprefix == t.prefix:
                if inclusivePop:
                    most_recently_popped = self.popTag()
                break
            most_recently_popped = self.popTag()

        return most_recently_popped

    def handle_starttag(
        self,
        name: str,
        namespace: Optional[str],
        nsprefix: Optional[str],
        attrs: _RawAttributeValues,
        sourceline: Optional[int] = None,
        sourcepos: Optional[int] = None,
        namespaces: Optional[Dict[str, str]] = None,
    ) -> Optional[Tag]:
        """Called by the tree builder when a new tag is encountered.

        :param name: Name of the tag.
        :param nsprefix: Namespace prefix for the tag.
        :param attrs: A dictionary of attribute values. Note that
           attribute values are expected to be simple strings; processing
           of multi-valued attributes such as "class" comes later.
        :param sourceline: The line number where this tag was found in its
            source document.
        :param sourcepos: The character position within `sourceline` where this
            tag was found.
        :param namespaces: A dictionary of all namespace prefix mappings
            currently in scope in the document.

        If this method returns None, the tag was rejected by an active
        `ElementFilter`. You should proceed as if the tag had not occurred
        in the document. For instance, if this was a self-closing tag,
        don't call handle_endtag.

        :meta private:
        """
        # print("Start tag %s: %s" % (name, attrs))
        self.endData()

        if (
            self.parse_only
            and len(self.tagStack) <= 1
            and not self.parse_only.allow_tag_creation(nsprefix, name, attrs)
        ):
            return None

        tag_class = self.element_classes.get(Tag, Tag)
        # Assume that this is either Tag or a subclass of Tag. If not,
        # the user brought type-unsafety upon themselves.
        tag_class = cast(Type[Tag], tag_class)
        tag = tag_class(
            self,
            self.builder,
            name,
            namespace,
            nsprefix,
            attrs,
            self.currentTag,
            self._most_recent_element,
            sourceline=sourceline,
            sourcepos=sourcepos,
            namespaces=namespaces,
        )
        if tag is None:
            return tag
        if self._most_recent_element is not None:
            self._most_recent_element.next_element = tag
        self._most_recent_element = tag
        self.pushTag(tag)
        return tag

    def handle_endtag(self, name: str, nsprefix: Optional[str] = None) -> None:
        """Called by the tree builder when an ending tag is encountered.

        :param name: Name of the tag.
        :param nsprefix: Namespace prefix for the tag.

        :meta private:
        """
        # print("End tag: " + name)
        self.endData()
        self._popToTag(name, nsprefix)

    def handle_data(self, data: str) -> None:
        """Called by the tree builder when a chunk of textual data is
        encountered.

        :meta private:
        """
        self.current_data.append(data)

    def decode(
        self,
        indent_level: Optional[int] = None,
        eventual_encoding: _Encoding = DEFAULT_OUTPUT_ENCODING,
        formatter: Union[Formatter, str] = "minimal",
        iterator: Optional[Iterator[PageElement]] = None,
        **kwargs: Any,
    ) -> str:
        """Returns a string representation of the parse tree
            as a full HTML or XML document.

        :param indent_level: Each line of the rendering will be
           indented this many levels. (The ``formatter`` decides what a
           'level' means, in terms of spaces or other characters
           output.) This is used internally in recursive calls while
           pretty-printing.
        :param eventual_encoding: The encoding of the final document.
            If this is None, the document will be a Unicode string.
        :param formatter: Either a `Formatter` object, or a string naming one of
            the standard formatters.
        :param iterator: The iterator to use when navigating over the
            parse tree. This is only used by `Tag.decode_contents` and
            you probably won't need to use it.
        """
        if self.is_xml:
            # Print the XML declaration
            encoding_part = ""
            declared_encoding: Optional[str] = eventual_encoding
            if eventual_encoding in PYTHON_SPECIFIC_ENCODINGS:
                # This is a special Python encoding; it can't actually
                # go into an XML document because it means nothing
                # outside of Python.
                declared_encoding = None
            if declared_encoding is not None:
                encoding_part = ' encoding="%s"' % declared_encoding
            prefix = '<?xml version="1.0"%s?>\n' % encoding_part
        else:
            prefix = ""

        # Prior to 4.13.0, the first argument to this method was a
        # bool called pretty_print, which gave the method a different
        # signature from its superclass implementation, Tag.decode.
        #
        # The signatures of the two methods now match, but just in
        # case someone is still passing a boolean in as the first
        # argument to this method (or a keyword argument with the old
        # name), we can handle it and put out a DeprecationWarning.
        warning: Optional[str] = None
        if isinstance(indent_level, bool):
            if indent_level is True:
                indent_level = 0
            elif indent_level is False:
                indent_level = None
            warning = f"As of 4.13.0, the first argument to BeautifulSoup.decode has been changed from bool to int, to match Tag.decode. Pass in a value of {indent_level} instead."
        else:
            pretty_print = kwargs.pop("pretty_print", None)
            assert not kwargs
            if pretty_print is not None:
                if pretty_print is True:
                    indent_level = 0
                elif pretty_print is False:
                    indent_level = None
                warning = f"As of 4.13.0, the pretty_print argument to BeautifulSoup.decode has been removed, to match Tag.decode. Pass in a value of indent_level={indent_level} instead."

        if warning:
            warnings.warn(warning, DeprecationWarning, stacklevel=2)
        elif indent_level is False or pretty_print is False:
            indent_level = None
        return prefix + super(BeautifulSoup, self).decode(
            indent_level, eventual_encoding, formatter, iterator
        )


# Aliases to make it easier to get started quickly, e.g. 'from bs4 import _soup'
_s = BeautifulSoup
_soup = BeautifulSoup


class BeautifulStoneSoup(BeautifulSoup):
    """Deprecated interface to an XML parser."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["features"] = "xml"
        warnings.warn(
            "The BeautifulStoneSoup class was deprecated in version 4.0.0. Instead of using "
            'it, pass features="xml" into the BeautifulSoup constructor.',
            DeprecationWarning,
            stacklevel=2,
        )
        super(BeautifulStoneSoup, self).__init__(*args, **kwargs)


# If this file is run as a script, act as an HTML pretty-printer.
if __name__ == "__main__":
    import sys

    soup = BeautifulSoup(sys.stdin)
    print((soup.prettify()))