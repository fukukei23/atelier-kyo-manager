
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_thread_cache.py ===
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import traceback
from functools import partial
from itertools import count
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import outcome

if TYPE_CHECKING:
    from collections.abc import Callable

RetT = TypeVar("RetT")


def _to_os_thread_name(name: str) -> bytes:
    # ctypes handles the trailing \00
    return name.encode("ascii", errors="replace")[:15]


# used to construct the method used to set os thread name, or None, depending on platform.
# called once on import
def get_os_thread_name_func() -> Callable[[int | None, str], None] | None:
    def namefunc(
        setname: Callable[[int, bytes], int],
        ident: int | None,
        name: str,
    ) -> None:
        # Thread.ident is None "if it has not been started". Unclear if that can happen
        # with current usage.
        if ident is not None:  # pragma: no cover
            setname(ident, _to_os_thread_name(name))

    # namefunc on Mac also takes an ident, even if pthread_setname_np doesn't/can't use it
    # so the caller don't need to care about platform.
    def darwin_namefunc(
        setname: Callable[[bytes], int],
        ident: int | None,
        name: str,
    ) -> None:
        # I don't know if Mac can rename threads that hasn't been started, but default
        # to no to be on the safe side.
        if ident is not None:  # pragma: no cover
            setname(_to_os_thread_name(name))

    # find the pthread library
    # this will fail on windows and musl
    libpthread_path = ctypes.util.find_library("pthread")
    if not libpthread_path:
        # musl includes pthread functions directly in libc.so
        # (but note that find_library("c") does not work on musl,
        #  see: https://github.com/python/cpython/issues/65821)
        # so try that library instead
        # if it doesn't exist, CDLL() will fail below
        libpthread_path = "libc.so"

    # Sometimes windows can find the path, but gives a permission error when
    # accessing it. Catching a wider exception in case of more esoteric errors.
    # https://github.com/python-trio/trio/issues/2688
    try:
        libpthread = ctypes.CDLL(libpthread_path)
    except Exception:  # pragma: no cover
        return None

    # get the setname method from it
    # afaik this should never fail
    pthread_setname_np = getattr(libpthread, "pthread_setname_np", None)
    if pthread_setname_np is None:  # pragma: no cover
        return None

    # specify function prototype
    pthread_setname_np.restype = ctypes.c_int

    # on mac OSX pthread_setname_np does not take a thread id,
    # it only lets threads name themselves, which is not a problem for us.
    # Just need to make sure to call it correctly
    if sys.platform == "darwin":
        pthread_setname_np.argtypes = [ctypes.c_char_p]
        return partial(darwin_namefunc, pthread_setname_np)

    # otherwise assume linux parameter conventions. Should also work on *BSD
    pthread_setname_np.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    return partial(namefunc, pthread_setname_np)


# construct os thread name method
set_os_thread_name = get_os_thread_name_func()

# The "thread cache" is a simple unbounded thread pool, i.e., it automatically
# spawns as many threads as needed to handle all the requests its given. Its
# only purpose is to cache worker threads so that they don't have to be
# started from scratch every time we want to delegate some work to a thread.
# It's expected that some higher-level code will track how many threads are in
# use to avoid overwhelming the system (e.g. the limiter= argument to
# trio.to_thread.run_sync).
#
# To maximize sharing, there's only one thread cache per process, even if you
# have multiple calls to trio.run.
#
# Guarantees:
#
# It's safe to call start_thread_soon simultaneously from
# multiple threads.
#
# Idle threads are chosen in LIFO order, i.e. we *don't* spread work evenly
# over all threads. Instead we try to let some threads do most of the work
# while others sit idle as much as possible. Compared to FIFO, this has better
# memory cache behavior, and it makes it easier to detect when we have too
# many threads, so idle ones can exit.
#
# This code assumes that 'dict' has the following properties:
#
# - __setitem__, __delitem__, and popitem are all thread-safe and atomic with
#   respect to each other. This is guaranteed by the GIL.
#
# - popitem returns the most-recently-added item (i.e., __setitem__ + popitem
#   give you a LIFO queue). This relies on dicts being insertion-ordered, like
#   they are in py36+.

# How long a thread will idle waiting for new work before gives up and exits.
# This value is pretty arbitrary; I don't think it matters too much.
IDLE_TIMEOUT = 10  # seconds

name_counter = count()


class WorkerThread(Generic[RetT]):
    __slots__ = ("_default_name", "_job", "_thread", "_thread_cache", "_worker_lock")

    def __init__(self, thread_cache: ThreadCache) -> None:
        self._job: (
            tuple[
                Callable[[], RetT],
                Callable[[outcome.Outcome[RetT]], object],
                str | None,
            ]
            | None
        ) = None
        self._thread_cache = thread_cache
        # This Lock is used in an unconventional way.
        #
        # "Unlocked" means we have a pending job that's been assigned to us;
        # "locked" means that we don't.
        #
        # Initially we have no job, so it starts out in locked state.
        self._worker_lock = Lock()
        self._worker_lock.acquire()
        self._default_name = f"Trio thread {next(name_counter)}"

        self._thread = Thread(target=self._work, name=self._default_name, daemon=True)

        if set_os_thread_name:
            set_os_thread_name(self._thread.ident, self._default_name)
        self._thread.start()

    def _handle_job(self) -> None:
        # Handle job in a separate method to ensure user-created
        # objects are cleaned up in a consistent manner.
        assert self._job is not None
        fn, deliver, name = self._job
        self._job = None

        # set name
        if name is not None:
            self._thread.name = name
            if set_os_thread_name:
                set_os_thread_name(self._thread.ident, name)
        result = outcome.capture(fn)

        # reset name if it was changed
        if name is not None:
            self._thread.name = self._default_name
            if set_os_thread_name:
                set_os_thread_name(self._thread.ident, self._default_name)

        # Tell the cache that we're available to be assigned a new
        # job. We do this *before* calling 'deliver', so that if
        # 'deliver' triggers a new job, it can be assigned to us
        # instead of spawning a new thread.
        self._thread_cache._idle_workers[self] = None
        try:
            deliver(result)
        except BaseException as e:
            print("Exception while delivering result of thread", file=sys.stderr)
            traceback.print_exception(type(e), e, e.__traceback__)

    def _work(self) -> None:
        while True:
            if self._worker_lock.acquire(timeout=IDLE_TIMEOUT):
                # We got a job
                self._handle_job()
            else:
                # Timeout acquiring lock, so we can probably exit. But,
                # there's a race condition: we might be assigned a job *just*
                # as we're about to exit. So we have to check.
                try:
                    del self._thread_cache._idle_workers[self]
                except KeyError:
                    # Someone else removed us from the idle worker queue, so
                    # they must be in the process of assigning us a job - loop
                    # around and wait for it.
                    continue
                else:
                    # We successfully removed ourselves from the idle
                    # worker queue, so no more jobs are incoming; it's safe to
                    # exit.
                    return


class ThreadCache:
    __slots__ = ("_idle_workers",)

    def __init__(self) -> None:
        self._idle_workers: dict[WorkerThread[Any], None] = {}  # type: ignore[explicit-any]

    def start_thread_soon(
        self,
        fn: Callable[[], RetT],
        deliver: Callable[[outcome.Outcome[RetT]], object],
        name: str | None = None,
    ) -> None:
        worker: WorkerThread[RetT]
        try:
            worker, _ = self._idle_workers.popitem()
        except KeyError:
            worker = WorkerThread(self)
        worker._job = (fn, deliver, name)
        worker._worker_lock.release()


THREAD_CACHE = ThreadCache()


def start_thread_soon(
    fn: Callable[[], RetT],
    deliver: Callable[[outcome.Outcome[RetT]], object],
    name: str | None = None,
) -> None:
    """Runs ``deliver(outcome.capture(fn))`` in a worker thread.

    Generally ``fn`` does some blocking work, and ``deliver`` delivers the
    result back to whoever is interested.

    This is a low-level, no-frills interface, very similar to using
    `threading.Thread` to spawn a thread directly. The main difference is
    that this function tries to reuse threads when possible, so it can be
    a bit faster than `threading.Thread`.

    Worker threads have the `~threading.Thread.daemon` flag set, which means
    that if your main thread exits, worker threads will automatically be
    killed. If you want to make sure that your ``fn`` runs to completion, then
    you should make sure that the main thread remains alive until ``deliver``
    is called.

    It is safe to call this function simultaneously from multiple threads.

    Args:

        fn (sync function): Performs arbitrary blocking work.

        deliver (sync function): Takes the `outcome.Outcome` of ``fn``, and
          delivers it. *Must not block.*

    Because worker threads are cached and reused for multiple calls, neither
    function should mutate thread-level state, like `threading.local` objects
    – or if they do, they should be careful to revert their changes before
    returning.

    Note:

        The split between ``fn`` and ``deliver`` serves two purposes. First,
        it's convenient, since most callers need something like this anyway.

        Second, it avoids a small race condition that could cause too many
        threads to be spawned. Consider a program that wants to run several
        jobs sequentially on a thread, so the main thread submits a job, waits
        for it to finish, submits another job, etc. In theory, this program
        should only need one worker thread. But what could happen is:

        1. Worker thread: First job finishes, and calls ``deliver``.

        2. Main thread: receives notification that the job finished, and calls
           ``start_thread_soon``.

        3. Main thread: sees that no worker threads are marked idle, so spawns
           a second worker thread.

        4. Original worker thread: marks itself as idle.

        To avoid this, threads mark themselves as idle *before* calling
        ``deliver``.

        Is this potential extra thread a major problem? Maybe not, but it's
        easy enough to avoid, and we figure that if the user is trying to
        limit how many threads they're using then it's polite to respect that.

    """
    THREAD_CACHE.start_thread_soon(fn, deliver, name)


def clear_worker_threads() -> None:
    # This is OK because the child process does not actually have any
    # worker threads. Additionally, while WorkerThread keeps a strong
    # reference and so would get affected, the only place those are
    # stored is here.
    THREAD_CACHE._idle_workers.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=clear_worker_threads)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_tests\test_instrumentation.py ===
from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import attrs
import pytest

from ... import _abc, _core
from .tutil import check_sequence_matches

if TYPE_CHECKING:
    from collections.abc import Container, Iterable

    from ...lowlevel import Task


@attrs.define(eq=False, slots=False)
class TaskRecorder(_abc.Instrument):
    record: list[tuple[str, Task | None]] = attrs.Factory(list)

    def before_run(self) -> None:
        self.record.append(("before_run", None))

    def task_scheduled(self, task: Task) -> None:
        self.record.append(("schedule", task))

    def before_task_step(self, task: Task) -> None:
        assert task is _core.current_task()
        self.record.append(("before", task))

    def after_task_step(self, task: Task) -> None:
        assert task is _core.current_task()
        self.record.append(("after", task))

    def after_run(self) -> None:
        self.record.append(("after_run", None))

    def filter_tasks(self, tasks: Container[Task]) -> Iterable[tuple[str, Task | None]]:
        for item in self.record:
            if item[0] in ("schedule", "before", "after") and item[1] in tasks:
                yield item
            if item[0] in ("before_run", "after_run"):
                yield item


def test_instruments(recwarn: object) -> None:
    r1 = TaskRecorder()
    r2 = TaskRecorder()
    r3 = TaskRecorder()

    task = None

    # We use a child task for this, because the main task does some extra
    # bookkeeping stuff that can leak into the instrument results, and we
    # don't want to deal with it.
    async def task_fn() -> None:
        nonlocal task
        task = _core.current_task()

        for _ in range(4):
            await _core.checkpoint()
        # replace r2 with r3, to test that we can manipulate them as we go
        _core.remove_instrument(r2)
        with pytest.raises(KeyError):
            _core.remove_instrument(r2)
        # add is idempotent
        _core.add_instrument(r3)
        _core.add_instrument(r3)
        for _ in range(1):
            await _core.checkpoint()

    async def main() -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(task_fn)

    _core.run(main, instruments=[r1, r2])

    # It sleeps 5 times, so it runs 6 times.  Note that checkpoint()
    # reschedules the task immediately upon yielding, before the
    # after_task_step event fires.
    expected = (
        [("before_run", None), ("schedule", task)]
        + [("before", task), ("schedule", task), ("after", task)] * 5
        + [("before", task), ("after", task), ("after_run", None)]
    )
    assert r1.record == r2.record + r3.record
    assert task is not None
    assert list(r1.filter_tasks([task])) == expected


def test_instruments_interleave() -> None:
    tasks = {}

    async def two_step1() -> None:
        tasks["t1"] = _core.current_task()
        await _core.checkpoint()

    async def two_step2() -> None:
        tasks["t2"] = _core.current_task()
        await _core.checkpoint()

    async def main() -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(two_step1)
            nursery.start_soon(two_step2)

    r = TaskRecorder()
    _core.run(main, instruments=[r])

    expected = [
        ("before_run", None),
        ("schedule", tasks["t1"]),
        ("schedule", tasks["t2"]),
        {
            ("before", tasks["t1"]),
            ("schedule", tasks["t1"]),
            ("after", tasks["t1"]),
            ("before", tasks["t2"]),
            ("schedule", tasks["t2"]),
            ("after", tasks["t2"]),
        },
        {
            ("before", tasks["t1"]),
            ("after", tasks["t1"]),
            ("before", tasks["t2"]),
            ("after", tasks["t2"]),
        },
        ("after_run", None),
    ]
    print(list(r.filter_tasks(tasks.values())))
    check_sequence_matches(list(r.filter_tasks(tasks.values())), expected)


def test_null_instrument() -> None:
    # undefined instrument methods are skipped
    class NullInstrument(_abc.Instrument):
        def something_unrelated(self) -> None:
            pass  # pragma: no cover

    async def main() -> None:
        await _core.checkpoint()

    _core.run(main, instruments=[NullInstrument()])


def test_instrument_before_after_run() -> None:
    record = []

    class BeforeAfterRun(_abc.Instrument):
        def before_run(self) -> None:
            record.append("before_run")

        def after_run(self) -> None:
            record.append("after_run")

    async def main() -> None:
        pass

    _core.run(main, instruments=[BeforeAfterRun()])
    assert record == ["before_run", "after_run"]


def test_instrument_task_spawn_exit() -> None:
    record = []

    class SpawnExitRecorder(_abc.Instrument):
        def task_spawned(self, task: Task) -> None:
            record.append(("spawned", task))

        def task_exited(self, task: Task) -> None:
            record.append(("exited", task))

    async def main() -> Task:
        return _core.current_task()

    main_task = _core.run(main, instruments=[SpawnExitRecorder()])
    assert ("spawned", main_task) in record
    assert ("exited", main_task) in record


# This test also tests having a crash before the initial task is even spawned,
# which is very difficult to handle.
def test_instruments_crash(caplog: pytest.LogCaptureFixture) -> None:
    record = []

    class BrokenInstrument(_abc.Instrument):
        def task_scheduled(self, task: Task) -> NoReturn:
            record.append("scheduled")
            raise ValueError("oops")

        def close(self) -> None:
            # Shouldn't be called -- tests that the instrument disabling logic
            # works right.
            record.append("closed")  # pragma: no cover

    async def main() -> Task:
        record.append("main ran")
        return _core.current_task()

    r = TaskRecorder()
    main_task = _core.run(main, instruments=[r, BrokenInstrument()])
    assert record == ["scheduled", "main ran"]
    # the TaskRecorder kept going throughout, even though the BrokenInstrument
    # was disabled
    assert ("after", main_task) in r.record
    assert ("after_run", None) in r.record
    # And we got a log message
    assert caplog.records[0].exc_info is not None
    exc_type, exc_value, _exc_traceback = caplog.records[0].exc_info
    assert exc_type is ValueError
    assert str(exc_value) == "oops"
    assert "Instrument has been disabled" in caplog.records[0].message


def test_instruments_monkeypatch() -> None:
    class NullInstrument(_abc.Instrument):
        pass

    instrument = NullInstrument()

    async def main() -> None:
        record: list[Task] = []

        # Changing the set of hooks implemented by an instrument after
        # it's installed doesn't make them start being called right away
        instrument.before_task_step = (  # type: ignore[method-assign]
            record.append  # type: ignore[assignment] # append is pos-only
        )

        await _core.checkpoint()
        await _core.checkpoint()
        assert len(record) == 0

        # But if we remove and re-add the instrument, the new hooks are
        # picked up
        _core.remove_instrument(instrument)
        _core.add_instrument(instrument)
        await _core.checkpoint()
        await _core.checkpoint()
        assert record.count(_core.current_task()) == 2

        _core.remove_instrument(instrument)
        await _core.checkpoint()
        await _core.checkpoint()
        assert record.count(_core.current_task()) == 2

    _core.run(main, instruments=[instrument])


def test_instrument_that_raises_on_getattr() -> None:
    class EvilInstrument(_abc.Instrument):
        def task_exited(self, task: Task) -> NoReturn:
            raise AssertionError("this should never happen")  # pragma: no cover

        @property
        def after_run(self) -> NoReturn:
            raise ValueError("oops")

    async def main() -> None:
        with pytest.raises(ValueError, match=r"^oops$"):
            _core.add_instrument(EvilInstrument())

        # Make sure the instrument is fully removed from the per-method lists
        runner = _core.current_task()._runner
        assert "after_run" not in runner.instruments
        assert "task_exited" not in runner.instruments

    _core.run(main)


def test_instrument_call_trio_context() -> None:
    called = set()

    class Instrument(_abc.Instrument):
        pass

    hooks = {
        # not run in task context
        "after_io_wait": (True, False),
        "before_io_wait": (True, False),
        "before_run": (True, False),
        "after_run": (True, False),
        # run in task context
        "before_task_step": (True, True),
        "after_task_step": (True, True),
        "task_exited": (True, True),
        # depends
        "task_scheduled": (True, None),
        "task_spawned": (True, None),
    }
    for hook, val in hooks.items():

        def h(
            self: Instrument,
            *args: object,
            hook: str = hook,
            val: tuple[bool, bool | None] = val,
        ) -> None:
            fail_str = f"failed in {hook}"

            assert _core.in_trio_run() == val[0], fail_str
            if val[1] is not None:
                assert _core.in_trio_task() == val[1], fail_str
            called.add(hook)

        setattr(Instrument, hook, h)

    async def main() -> None:
        await _core.checkpoint()

        async with _core.open_nursery() as nursery:
            nursery.start_soon(_core.checkpoint)

    _core.run(main, instruments=[Instrument()])
    assert called == set(hooks)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wsproto\extensions.py ===
"""
wsproto/extensions
~~~~~~~~~~~~~~~~~~

WebSocket extensions.
"""

import zlib
from typing import Optional, Tuple, Union

from .frame_protocol import CloseReason, FrameDecoder, FrameProtocol, Opcode, RsvBits


class Extension:
    name: str

    def enabled(self) -> bool:
        return False

    def offer(self) -> Union[bool, str]:
        pass

    def accept(self, offer: str) -> Optional[Union[bool, str]]:
        pass

    def finalize(self, offer: str) -> None:
        pass

    def frame_inbound_header(
        self,
        proto: Union[FrameDecoder, FrameProtocol],
        opcode: Opcode,
        rsv: RsvBits,
        payload_length: int,
    ) -> Union[CloseReason, RsvBits]:
        return RsvBits(False, False, False)

    def frame_inbound_payload_data(
        self, proto: Union[FrameDecoder, FrameProtocol], data: bytes
    ) -> Union[bytes, CloseReason]:
        return data

    def frame_inbound_complete(
        self, proto: Union[FrameDecoder, FrameProtocol], fin: bool
    ) -> Union[bytes, CloseReason, None]:
        pass

    def frame_outbound(
        self,
        proto: Union[FrameDecoder, FrameProtocol],
        opcode: Opcode,
        rsv: RsvBits,
        data: bytes,
        fin: bool,
    ) -> Tuple[RsvBits, bytes]:
        return (rsv, data)


class PerMessageDeflate(Extension):
    name = "permessage-deflate"

    DEFAULT_CLIENT_MAX_WINDOW_BITS = 15
    DEFAULT_SERVER_MAX_WINDOW_BITS = 15

    def __init__(
        self,
        client_no_context_takeover: bool = False,
        client_max_window_bits: Optional[int] = None,
        server_no_context_takeover: bool = False,
        server_max_window_bits: Optional[int] = None,
    ) -> None:
        self.client_no_context_takeover = client_no_context_takeover
        self.server_no_context_takeover = server_no_context_takeover
        self._client_max_window_bits = self.DEFAULT_CLIENT_MAX_WINDOW_BITS
        self._server_max_window_bits = self.DEFAULT_SERVER_MAX_WINDOW_BITS
        if client_max_window_bits is not None:
            self.client_max_window_bits = client_max_window_bits
        if server_max_window_bits is not None:
            self.server_max_window_bits = server_max_window_bits

        self._compressor: Optional[zlib._Compress] = None  # noqa
        self._decompressor: Optional[zlib._Decompress] = None  # noqa
        # This refers to the current frame
        self._inbound_is_compressible: Optional[bool] = None
        # This refers to the ongoing message (which might span multiple
        # frames). Only the first frame in a fragmented message is flagged for
        # compression, so this carries that bit forward.
        self._inbound_compressed: Optional[bool] = None

        self._enabled = False

    @property
    def client_max_window_bits(self) -> int:
        return self._client_max_window_bits

    @client_max_window_bits.setter
    def client_max_window_bits(self, value: int) -> None:
        if value < 9 or value > 15:
            raise ValueError("Window size must be between 9 and 15 inclusive")
        self._client_max_window_bits = value

    @property
    def server_max_window_bits(self) -> int:
        return self._server_max_window_bits

    @server_max_window_bits.setter
    def server_max_window_bits(self, value: int) -> None:
        if value < 9 or value > 15:
            raise ValueError("Window size must be between 9 and 15 inclusive")
        self._server_max_window_bits = value

    def _compressible_opcode(self, opcode: Opcode) -> bool:
        return opcode in (Opcode.TEXT, Opcode.BINARY, Opcode.CONTINUATION)

    def enabled(self) -> bool:
        return self._enabled

    def offer(self) -> Union[bool, str]:
        parameters = [
            "client_max_window_bits=%d" % self.client_max_window_bits,
            "server_max_window_bits=%d" % self.server_max_window_bits,
        ]

        if self.client_no_context_takeover:
            parameters.append("client_no_context_takeover")
        if self.server_no_context_takeover:
            parameters.append("server_no_context_takeover")

        return "; ".join(parameters)

    def finalize(self, offer: str) -> None:
        bits = [b.strip() for b in offer.split(";")]
        for bit in bits[1:]:
            if bit.startswith("client_no_context_takeover"):
                self.client_no_context_takeover = True
            elif bit.startswith("server_no_context_takeover"):
                self.server_no_context_takeover = True
            elif bit.startswith("client_max_window_bits"):
                self.client_max_window_bits = int(bit.split("=", 1)[1].strip())
            elif bit.startswith("server_max_window_bits"):
                self.server_max_window_bits = int(bit.split("=", 1)[1].strip())

        self._enabled = True

    def _parse_params(self, params: str) -> Tuple[Optional[int], Optional[int]]:
        client_max_window_bits = None
        server_max_window_bits = None

        bits = [b.strip() for b in params.split(";")]
        for bit in bits[1:]:
            if bit.startswith("client_no_context_takeover"):
                self.client_no_context_takeover = True
            elif bit.startswith("server_no_context_takeover"):
                self.server_no_context_takeover = True
            elif bit.startswith("client_max_window_bits"):
                if "=" in bit:
                    client_max_window_bits = int(bit.split("=", 1)[1].strip())
                else:
                    client_max_window_bits = self.client_max_window_bits
            elif bit.startswith("server_max_window_bits"):
                if "=" in bit:
                    server_max_window_bits = int(bit.split("=", 1)[1].strip())
                else:
                    server_max_window_bits = self.server_max_window_bits

        return client_max_window_bits, server_max_window_bits

    def accept(self, offer: str) -> Union[bool, None, str]:
        client_max_window_bits, server_max_window_bits = self._parse_params(offer)

        parameters = []

        if self.client_no_context_takeover:
            parameters.append("client_no_context_takeover")
        if self.server_no_context_takeover:
            parameters.append("server_no_context_takeover")
        try:
            if client_max_window_bits is not None:
                parameters.append("client_max_window_bits=%d" % client_max_window_bits)
                self.client_max_window_bits = client_max_window_bits
            if server_max_window_bits is not None:
                parameters.append("server_max_window_bits=%d" % server_max_window_bits)
                self.server_max_window_bits = server_max_window_bits
        except ValueError:
            return None
        else:
            self._enabled = True
            return "; ".join(parameters)

    def frame_inbound_header(
        self,
        proto: Union[FrameDecoder, FrameProtocol],
        opcode: Opcode,
        rsv: RsvBits,
        payload_length: int,
    ) -> Union[CloseReason, RsvBits]:
        if rsv.rsv1 and opcode.iscontrol():
            return CloseReason.PROTOCOL_ERROR
        if rsv.rsv1 and opcode is Opcode.CONTINUATION:
            return CloseReason.PROTOCOL_ERROR

        self._inbound_is_compressible = self._compressible_opcode(opcode)

        if self._inbound_compressed is None:
            self._inbound_compressed = rsv.rsv1
            if self._inbound_compressed:
                assert self._inbound_is_compressible
                if proto.client:
                    bits = self.server_max_window_bits
                else:
                    bits = self.client_max_window_bits
                if self._decompressor is None:
                    self._decompressor = zlib.decompressobj(-int(bits))

        return RsvBits(True, False, False)

    def frame_inbound_payload_data(
        self, proto: Union[FrameDecoder, FrameProtocol], data: bytes
    ) -> Union[bytes, CloseReason]:
        if not self._inbound_compressed or not self._inbound_is_compressible:
            return data
        assert self._decompressor is not None

        try:
            return self._decompressor.decompress(bytes(data))
        except zlib.error:
            return CloseReason.INVALID_FRAME_PAYLOAD_DATA

    def frame_inbound_complete(
        self, proto: Union[FrameDecoder, FrameProtocol], fin: bool
    ) -> Union[bytes, CloseReason, None]:
        if not fin:
            return None
        if not self._inbound_is_compressible:
            self._inbound_compressed = None
            return None
        if not self._inbound_compressed:
            self._inbound_compressed = None
            return None
        assert self._decompressor is not None

        try:
            data = self._decompressor.decompress(b"\x00\x00\xff\xff")
            data += self._decompressor.flush()
        except zlib.error:
            return CloseReason.INVALID_FRAME_PAYLOAD_DATA

        if proto.client:
            no_context_takeover = self.server_no_context_takeover
        else:
            no_context_takeover = self.client_no_context_takeover

        if no_context_takeover:
            self._decompressor = None

        self._inbound_compressed = None

        return data

    def frame_outbound(
        self,
        proto: Union[FrameDecoder, FrameProtocol],
        opcode: Opcode,
        rsv: RsvBits,
        data: bytes,
        fin: bool,
    ) -> Tuple[RsvBits, bytes]:
        if not self._compressible_opcode(opcode):
            return (rsv, data)

        if opcode is not Opcode.CONTINUATION:
            rsv = RsvBits(True, *rsv[1:])

        if self._compressor is None:
            assert opcode is not Opcode.CONTINUATION
            if proto.client:
                bits = self.client_max_window_bits
            else:
                bits = self.server_max_window_bits
            self._compressor = zlib.compressobj(
                zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -int(bits)
            )

        data = self._compressor.compress(bytes(data))

        if fin:
            data += self._compressor.flush(zlib.Z_SYNC_FLUSH)
            data = data[:-4]

            if proto.client:
                no_context_takeover = self.client_no_context_takeover
            else:
                no_context_takeover = self.server_no_context_takeover

            if no_context_takeover:
                self._compressor = None

        return (rsv, data)

    def __repr__(self) -> str:
        descr = ["client_max_window_bits=%d" % self.client_max_window_bits]
        if self.client_no_context_takeover:
            descr.append("client_no_context_takeover")
        descr.append("server_max_window_bits=%d" % self.server_max_window_bits)
        if self.server_no_context_takeover:
            descr.append("server_no_context_takeover")

        return "<{} {}>".format(self.__class__.__name__, "; ".join(descr))


#: SUPPORTED_EXTENSIONS maps all supported extension names to their class.
#: This can be used to iterate all supported extensions of wsproto, instantiate
#: new extensions based on their name, or check if a given extension is
#: supported or not.
SUPPORTED_EXTENSIONS = {PerMessageDeflate.name: PerMessageDeflate}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\network\download.py ===
"""Download files with progress indicators."""

import email.message
import logging
import mimetypes
import os
from http import HTTPStatus
from typing import BinaryIO, Iterable, Optional, Tuple

from pip._vendor.requests.models import Response
from pip._vendor.urllib3.exceptions import ReadTimeoutError

from pip._internal.cli.progress_bars import get_download_progress_renderer
from pip._internal.exceptions import IncompleteDownloadError, NetworkConnectionError
from pip._internal.models.index import PyPI
from pip._internal.models.link import Link
from pip._internal.network.cache import is_from_cache
from pip._internal.network.session import PipSession
from pip._internal.network.utils import HEADERS, raise_for_status, response_chunks
from pip._internal.utils.misc import format_size, redact_auth_from_url, splitext

logger = logging.getLogger(__name__)


def _get_http_response_size(resp: Response) -> Optional[int]:
    try:
        return int(resp.headers["content-length"])
    except (ValueError, KeyError, TypeError):
        return None


def _get_http_response_etag_or_last_modified(resp: Response) -> Optional[str]:
    """
    Return either the ETag or Last-Modified header (or None if neither exists).
    The return value can be used in an If-Range header.
    """
    return resp.headers.get("etag", resp.headers.get("last-modified"))


def _prepare_download(
    resp: Response,
    link: Link,
    progress_bar: str,
    total_length: Optional[int],
    range_start: Optional[int] = 0,
) -> Iterable[bytes]:
    if link.netloc == PyPI.file_storage_domain:
        url = link.show_url
    else:
        url = link.url_without_fragment

    logged_url = redact_auth_from_url(url)

    if total_length:
        if range_start:
            logged_url = (
                f"{logged_url} ({format_size(range_start)}/{format_size(total_length)})"
            )
        else:
            logged_url = f"{logged_url} ({format_size(total_length)})"

    if is_from_cache(resp):
        logger.info("Using cached %s", logged_url)
    elif range_start:
        logger.info("Resuming download %s", logged_url)
    else:
        logger.info("Downloading %s", logged_url)

    if logger.getEffectiveLevel() > logging.INFO:
        show_progress = False
    elif is_from_cache(resp):
        show_progress = False
    elif not total_length:
        show_progress = True
    elif total_length > (512 * 1024):
        show_progress = True
    else:
        show_progress = False

    chunks = response_chunks(resp)

    if not show_progress:
        return chunks

    renderer = get_download_progress_renderer(
        bar_type=progress_bar, size=total_length, initial_progress=range_start
    )
    return renderer(chunks)


def sanitize_content_filename(filename: str) -> str:
    """
    Sanitize the "filename" value from a Content-Disposition header.
    """
    return os.path.basename(filename)


def parse_content_disposition(content_disposition: str, default_filename: str) -> str:
    """
    Parse the "filename" value from a Content-Disposition header, and
    return the default filename if the result is empty.
    """
    m = email.message.Message()
    m["content-type"] = content_disposition
    filename = m.get_param("filename")
    if filename:
        # We need to sanitize the filename to prevent directory traversal
        # in case the filename contains ".." path parts.
        filename = sanitize_content_filename(str(filename))
    return filename or default_filename


def _get_http_response_filename(resp: Response, link: Link) -> str:
    """Get an ideal filename from the given HTTP response, falling back to
    the link filename if not provided.
    """
    filename = link.filename  # fallback
    # Have a look at the Content-Disposition header for a better guess
    content_disposition = resp.headers.get("content-disposition")
    if content_disposition:
        filename = parse_content_disposition(content_disposition, filename)
    ext: Optional[str] = splitext(filename)[1]
    if not ext:
        ext = mimetypes.guess_extension(resp.headers.get("content-type", ""))
        if ext:
            filename += ext
    if not ext and link.url != resp.url:
        ext = os.path.splitext(resp.url)[1]
        if ext:
            filename += ext
    return filename


def _http_get_download(
    session: PipSession,
    link: Link,
    range_start: Optional[int] = 0,
    if_range: Optional[str] = None,
) -> Response:
    target_url = link.url.split("#", 1)[0]
    headers = HEADERS.copy()
    # request a partial download
    if range_start:
        headers["Range"] = f"bytes={range_start}-"
    # make sure the file hasn't changed
    if if_range:
        headers["If-Range"] = if_range
    try:
        resp = session.get(target_url, headers=headers, stream=True)
        raise_for_status(resp)
    except NetworkConnectionError as e:
        assert e.response is not None
        logger.critical("HTTP error %s while getting %s", e.response.status_code, link)
        raise
    return resp


class Downloader:
    def __init__(
        self,
        session: PipSession,
        progress_bar: str,
        resume_retries: int,
    ) -> None:
        assert (
            resume_retries >= 0
        ), "Number of max resume retries must be bigger or equal to zero"
        self._session = session
        self._progress_bar = progress_bar
        self._resume_retries = resume_retries

    def __call__(self, link: Link, location: str) -> Tuple[str, str]:
        """Download the file given by link into location."""
        resp = _http_get_download(self._session, link)
        # NOTE: The original download size needs to be passed down everywhere
        # so if the download is resumed (with a HTTP Range request) the progress
        # bar will report the right size.
        total_length = _get_http_response_size(resp)
        content_type = resp.headers.get("Content-Type", "")

        filename = _get_http_response_filename(resp, link)
        filepath = os.path.join(location, filename)

        with open(filepath, "wb") as content_file:
            bytes_received = self._process_response(
                resp, link, content_file, 0, total_length
            )
            # If possible, check for an incomplete download and attempt resuming.
            if total_length and bytes_received < total_length:
                self._attempt_resume(
                    resp, link, content_file, total_length, bytes_received
                )

        return filepath, content_type

    def _process_response(
        self,
        resp: Response,
        link: Link,
        content_file: BinaryIO,
        bytes_received: int,
        total_length: Optional[int],
    ) -> int:
        """Process the response and write the chunks to the file."""
        chunks = _prepare_download(
            resp, link, self._progress_bar, total_length, range_start=bytes_received
        )
        return self._write_chunks_to_file(
            chunks, content_file, allow_partial=bool(total_length)
        )

    def _write_chunks_to_file(
        self, chunks: Iterable[bytes], content_file: BinaryIO, *, allow_partial: bool
    ) -> int:
        """Write the chunks to the file and return the number of bytes received."""
        bytes_received = 0
        try:
            for chunk in chunks:
                bytes_received += len(chunk)
                content_file.write(chunk)
        except ReadTimeoutError as e:
            # If partial downloads are OK (the download will be retried), don't bail.
            if not allow_partial:
                raise e

            # Ensuring bytes_received is returned to attempt resume
            logger.warning("Connection timed out while downloading.")

        return bytes_received

    def _attempt_resume(
        self,
        resp: Response,
        link: Link,
        content_file: BinaryIO,
        total_length: Optional[int],
        bytes_received: int,
    ) -> None:
        """Attempt to resume the download if connection was dropped."""
        etag_or_last_modified = _get_http_response_etag_or_last_modified(resp)

        attempts_left = self._resume_retries
        while total_length and attempts_left and bytes_received < total_length:
            attempts_left -= 1

            logger.warning(
                "Attempting to resume incomplete download (%s/%s, attempt %d)",
                format_size(bytes_received),
                format_size(total_length),
                (self._resume_retries - attempts_left),
            )

            try:
                # Try to resume the download using a HTTP range request.
                resume_resp = _http_get_download(
                    self._session,
                    link,
                    range_start=bytes_received,
                    if_range=etag_or_last_modified,
                )

                # Fallback: if the server responded with 200 (i.e., the file has
                # since been modified or range requests are unsupported) or any
                # other unexpected status, restart the download from the beginning.
                must_restart = resume_resp.status_code != HTTPStatus.PARTIAL_CONTENT
                if must_restart:
                    bytes_received, total_length, etag_or_last_modified = (
                        self._reset_download_state(resume_resp, content_file)
                    )

                bytes_received += self._process_response(
                    resume_resp, link, content_file, bytes_received, total_length
                )
            except (ConnectionError, ReadTimeoutError, OSError):
                continue

        # No more resume attempts. Raise an error if the download is still incomplete.
        if total_length and bytes_received < total_length:
            os.remove(content_file.name)
            raise IncompleteDownloadError(
                link, bytes_received, total_length, retries=self._resume_retries
            )

    def _reset_download_state(
        self,
        resp: Response,
        content_file: BinaryIO,
    ) -> Tuple[int, Optional[int], Optional[str]]:
        """Reset the download state to restart downloading from the beginning."""
        content_file.seek(0)
        content_file.truncate()
        bytes_received = 0
        total_length = _get_http_response_size(resp)
        etag_or_last_modified = _get_http_response_etag_or_last_modified(resp)

        return bytes_received, total_length, etag_or_last_modified


class BatchDownloader:
    def __init__(
        self,
        session: PipSession,
        progress_bar: str,
        resume_retries: int,
    ) -> None:
        self._downloader = Downloader(session, progress_bar, resume_retries)

    def __call__(
        self, links: Iterable[Link], location: str
    ) -> Iterable[Tuple[Link, Tuple[str, str]]]:
        """Download the files given by links into location."""
        for link in links:
            filepath, content_type = self._downloader(link, location)
            yield link, (filepath, content_type)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\requests\auth.py ===
"""
requests.auth
~~~~~~~~~~~~~

This module contains the authentication handlers for Requests.
"""

import hashlib
import os
import re
import threading
import time
import warnings
from base64 import b64encode

from ._internal_utils import to_native_string
from .compat import basestring, str, urlparse
from .cookies import extract_cookies_to_jar
from .utils import parse_dict_header

CONTENT_TYPE_FORM_URLENCODED = "application/x-www-form-urlencoded"
CONTENT_TYPE_MULTI_PART = "multipart/form-data"


def _basic_auth_str(username, password):
    """Returns a Basic Auth string."""

    # "I want us to put a big-ol' comment on top of it that
    # says that this behaviour is dumb but we need to preserve
    # it because people are relying on it."
    #    - Lukasa
    #
    # These are here solely to maintain backwards compatibility
    # for things like ints. This will be removed in 3.0.0.
    if not isinstance(username, basestring):
        warnings.warn(
            "Non-string usernames will no longer be supported in Requests "
            "3.0.0. Please convert the object you've passed in ({!r}) to "
            "a string or bytes object in the near future to avoid "
            "problems.".format(username),
            category=DeprecationWarning,
        )
        username = str(username)

    if not isinstance(password, basestring):
        warnings.warn(
            "Non-string passwords will no longer be supported in Requests "
            "3.0.0. Please convert the object you've passed in ({!r}) to "
            "a string or bytes object in the near future to avoid "
            "problems.".format(type(password)),
            category=DeprecationWarning,
        )
        password = str(password)
    # -- End Removal --

    if isinstance(username, str):
        username = username.encode("latin1")

    if isinstance(password, str):
        password = password.encode("latin1")

    authstr = "Basic " + to_native_string(
        b64encode(b":".join((username, password))).strip()
    )

    return authstr


class AuthBase:
    """Base class that all auth implementations derive from"""

    def __call__(self, r):
        raise NotImplementedError("Auth hooks must be callable.")


class HTTPBasicAuth(AuthBase):
    """Attaches HTTP Basic Authentication to the given Request object."""

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def __eq__(self, other):
        return all(
            [
                self.username == getattr(other, "username", None),
                self.password == getattr(other, "password", None),
            ]
        )

    def __ne__(self, other):
        return not self == other

    def __call__(self, r):
        r.headers["Authorization"] = _basic_auth_str(self.username, self.password)
        return r


class HTTPProxyAuth(HTTPBasicAuth):
    """Attaches HTTP Proxy Authentication to a given Request object."""

    def __call__(self, r):
        r.headers["Proxy-Authorization"] = _basic_auth_str(self.username, self.password)
        return r


class HTTPDigestAuth(AuthBase):
    """Attaches HTTP Digest Authentication to the given Request object."""

    def __init__(self, username, password):
        self.username = username
        self.password = password
        # Keep state in per-thread local storage
        self._thread_local = threading.local()

    def init_per_thread_state(self):
        # Ensure state is initialized just once per-thread
        if not hasattr(self._thread_local, "init"):
            self._thread_local.init = True
            self._thread_local.last_nonce = ""
            self._thread_local.nonce_count = 0
            self._thread_local.chal = {}
            self._thread_local.pos = None
            self._thread_local.num_401_calls = None

    def build_digest_header(self, method, url):
        """
        :rtype: str
        """

        realm = self._thread_local.chal["realm"]
        nonce = self._thread_local.chal["nonce"]
        qop = self._thread_local.chal.get("qop")
        algorithm = self._thread_local.chal.get("algorithm")
        opaque = self._thread_local.chal.get("opaque")
        hash_utf8 = None

        if algorithm is None:
            _algorithm = "MD5"
        else:
            _algorithm = algorithm.upper()
        # lambdas assume digest modules are imported at the top level
        if _algorithm == "MD5" or _algorithm == "MD5-SESS":

            def md5_utf8(x):
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.md5(x).hexdigest()

            hash_utf8 = md5_utf8
        elif _algorithm == "SHA":

            def sha_utf8(x):
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha1(x).hexdigest()

            hash_utf8 = sha_utf8
        elif _algorithm == "SHA-256":

            def sha256_utf8(x):
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha256(x).hexdigest()

            hash_utf8 = sha256_utf8
        elif _algorithm == "SHA-512":

            def sha512_utf8(x):
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha512(x).hexdigest()

            hash_utf8 = sha512_utf8

        KD = lambda s, d: hash_utf8(f"{s}:{d}")  # noqa:E731

        if hash_utf8 is None:
            return None

        # XXX not implemented yet
        entdig = None
        p_parsed = urlparse(url)
        #: path is request-uri defined in RFC 2616 which should not be empty
        path = p_parsed.path or "/"
        if p_parsed.query:
            path += f"?{p_parsed.query}"

        A1 = f"{self.username}:{realm}:{self.password}"
        A2 = f"{method}:{path}"

        HA1 = hash_utf8(A1)
        HA2 = hash_utf8(A2)

        if nonce == self._thread_local.last_nonce:
            self._thread_local.nonce_count += 1
        else:
            self._thread_local.nonce_count = 1
        ncvalue = f"{self._thread_local.nonce_count:08x}"
        s = str(self._thread_local.nonce_count).encode("utf-8")
        s += nonce.encode("utf-8")
        s += time.ctime().encode("utf-8")
        s += os.urandom(8)

        cnonce = hashlib.sha1(s).hexdigest()[:16]
        if _algorithm == "MD5-SESS":
            HA1 = hash_utf8(f"{HA1}:{nonce}:{cnonce}")

        if not qop:
            respdig = KD(HA1, f"{nonce}:{HA2}")
        elif qop == "auth" or "auth" in qop.split(","):
            noncebit = f"{nonce}:{ncvalue}:{cnonce}:auth:{HA2}"
            respdig = KD(HA1, noncebit)
        else:
            # XXX handle auth-int.
            return None

        self._thread_local.last_nonce = nonce

        # XXX should the partial digests be encoded too?
        base = (
            f'username="{self.username}", realm="{realm}", nonce="{nonce}", '
            f'uri="{path}", response="{respdig}"'
        )
        if opaque:
            base += f', opaque="{opaque}"'
        if algorithm:
            base += f', algorithm="{algorithm}"'
        if entdig:
            base += f', digest="{entdig}"'
        if qop:
            base += f', qop="auth", nc={ncvalue}, cnonce="{cnonce}"'

        return f"Digest {base}"

    def handle_redirect(self, r, **kwargs):
        """Reset num_401_calls counter on redirects."""
        if r.is_redirect:
            self._thread_local.num_401_calls = 1

    def handle_401(self, r, **kwargs):
        """
        Takes the given response and tries digest-auth, if needed.

        :rtype: requests.Response
        """

        # If response is not 4xx, do not auth
        # See https://github.com/psf/requests/issues/3772
        if not 400 <= r.status_code < 500:
            self._thread_local.num_401_calls = 1
            return r

        if self._thread_local.pos is not None:
            # Rewind the file position indicator of the body to where
            # it was to resend the request.
            r.request.body.seek(self._thread_local.pos)
        s_auth = r.headers.get("www-authenticate", "")

        if "digest" in s_auth.lower() and self._thread_local.num_401_calls < 2:
            self._thread_local.num_401_calls += 1
            pat = re.compile(r"digest ", flags=re.IGNORECASE)
            self._thread_local.chal = parse_dict_header(pat.sub("", s_auth, count=1))

            # Consume content and release the original connection
            # to allow our new request to reuse the same one.
            r.content
            r.close()
            prep = r.request.copy()
            extract_cookies_to_jar(prep._cookies, r.request, r.raw)
            prep.prepare_cookies(prep._cookies)

            prep.headers["Authorization"] = self.build_digest_header(
                prep.method, prep.url
            )
            _r = r.connection.send(prep, **kwargs)
            _r.history.append(r)
            _r.request = prep

            return _r

        self._thread_local.num_401_calls = 1
        return r

    def __call__(self, r):
        # Initialize per-thread state, if needed
        self.init_per_thread_state()
        # If we have a saved nonce, skip the 401
        if self._thread_local.last_nonce:
            r.headers["Authorization"] = self.build_digest_header(r.method, r.url)
        try:
            self._thread_local.pos = r.body.tell()
        except AttributeError:
            # In the case of HTTPDigestAuth being reused and the body of
            # the previous request was a file-like object, pos has the
            # file position of the previous body. Ensure it's set to
            # None.
            self._thread_local.pos = None
        r.register_hook("response", self.handle_401)
        r.register_hook("response", self.handle_redirect)
        self._thread_local.num_401_calls = 1

        return r

    def __eq__(self, other):
        return all(
            [
                self.username == getattr(other, "username", None),
                self.password == getattr(other, "password", None),
            ]
        )

    def __ne__(self, other):
        return not self == other

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\contrib\appengine.py ===
"""
This module provides a pool manager that uses Google App Engine's
`URLFetch Service <https://cloud.google.com/appengine/docs/python/urlfetch>`_.

Example usage::

    from pip._vendor.urllib3 import PoolManager
    from pip._vendor.urllib3.contrib.appengine import AppEngineManager, is_appengine_sandbox

    if is_appengine_sandbox():
        # AppEngineManager uses AppEngine's URLFetch API behind the scenes
        http = AppEngineManager()
    else:
        # PoolManager uses a socket-level API behind the scenes
        http = PoolManager()

    r = http.request('GET', 'https://google.com/')

There are `limitations <https://cloud.google.com/appengine/docs/python/\
urlfetch/#Python_Quotas_and_limits>`_ to the URLFetch service and it may not be
the best choice for your application. There are three options for using
urllib3 on Google App Engine:

1. You can use :class:`AppEngineManager` with URLFetch. URLFetch is
   cost-effective in many circumstances as long as your usage is within the
   limitations.
2. You can use a normal :class:`~urllib3.PoolManager` by enabling sockets.
   Sockets also have `limitations and restrictions
   <https://cloud.google.com/appengine/docs/python/sockets/\
   #limitations-and-restrictions>`_ and have a lower free quota than URLFetch.
   To use sockets, be sure to specify the following in your ``app.yaml``::

        env_variables:
            GAE_USE_SOCKETS_HTTPLIB : 'true'

3. If you are using `App Engine Flexible
<https://cloud.google.com/appengine/docs/flexible/>`_, you can use the standard
:class:`PoolManager` without any configuration or special environment variables.
"""

from __future__ import absolute_import

import io
import logging
import warnings

from ..exceptions import (
    HTTPError,
    HTTPWarning,
    MaxRetryError,
    ProtocolError,
    SSLError,
    TimeoutError,
)
from ..packages.six.moves.urllib.parse import urljoin
from ..request import RequestMethods
from ..response import HTTPResponse
from ..util.retry import Retry
from ..util.timeout import Timeout
from . import _appengine_environ

try:
    from google.appengine.api import urlfetch
except ImportError:
    urlfetch = None


log = logging.getLogger(__name__)


class AppEnginePlatformWarning(HTTPWarning):
    pass


class AppEnginePlatformError(HTTPError):
    pass


class AppEngineManager(RequestMethods):
    """
    Connection manager for Google App Engine sandbox applications.

    This manager uses the URLFetch service directly instead of using the
    emulated httplib, and is subject to URLFetch limitations as described in
    the App Engine documentation `here
    <https://cloud.google.com/appengine/docs/python/urlfetch>`_.

    Notably it will raise an :class:`AppEnginePlatformError` if:
        * URLFetch is not available.
        * If you attempt to use this on App Engine Flexible, as full socket
          support is available.
        * If a request size is more than 10 megabytes.
        * If a response size is more than 32 megabytes.
        * If you use an unsupported request method such as OPTIONS.

    Beyond those cases, it will raise normal urllib3 errors.
    """

    def __init__(
        self,
        headers=None,
        retries=None,
        validate_certificate=True,
        urlfetch_retries=True,
    ):
        if not urlfetch:
            raise AppEnginePlatformError(
                "URLFetch is not available in this environment."
            )

        warnings.warn(
            "urllib3 is using URLFetch on Google App Engine sandbox instead "
            "of sockets. To use sockets directly instead of URLFetch see "
            "https://urllib3.readthedocs.io/en/1.26.x/reference/urllib3.contrib.html.",
            AppEnginePlatformWarning,
        )

        RequestMethods.__init__(self, headers)
        self.validate_certificate = validate_certificate
        self.urlfetch_retries = urlfetch_retries

        self.retries = retries or Retry.DEFAULT

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Return False to re-raise any potential exceptions
        return False

    def urlopen(
        self,
        method,
        url,
        body=None,
        headers=None,
        retries=None,
        redirect=True,
        timeout=Timeout.DEFAULT_TIMEOUT,
        **response_kw
    ):

        retries = self._get_retries(retries, redirect)

        try:
            follow_redirects = redirect and retries.redirect != 0 and retries.total
            response = urlfetch.fetch(
                url,
                payload=body,
                method=method,
                headers=headers or {},
                allow_truncated=False,
                follow_redirects=self.urlfetch_retries and follow_redirects,
                deadline=self._get_absolute_timeout(timeout),
                validate_certificate=self.validate_certificate,
            )
        except urlfetch.DeadlineExceededError as e:
            raise TimeoutError(self, e)

        except urlfetch.InvalidURLError as e:
            if "too large" in str(e):
                raise AppEnginePlatformError(
                    "URLFetch request too large, URLFetch only "
                    "supports requests up to 10mb in size.",
                    e,
                )
            raise ProtocolError(e)

        except urlfetch.DownloadError as e:
            if "Too many redirects" in str(e):
                raise MaxRetryError(self, url, reason=e)
            raise ProtocolError(e)

        except urlfetch.ResponseTooLargeError as e:
            raise AppEnginePlatformError(
                "URLFetch response too large, URLFetch only supports"
                "responses up to 32mb in size.",
                e,
            )

        except urlfetch.SSLCertificateError as e:
            raise SSLError(e)

        except urlfetch.InvalidMethodError as e:
            raise AppEnginePlatformError(
                "URLFetch does not support method: %s" % method, e
            )

        http_response = self._urlfetch_response_to_http_response(
            response, retries=retries, **response_kw
        )

        # Handle redirect?
        redirect_location = redirect and http_response.get_redirect_location()
        if redirect_location:
            # Check for redirect response
            if self.urlfetch_retries and retries.raise_on_redirect:
                raise MaxRetryError(self, url, "too many redirects")
            else:
                if http_response.status == 303:
                    method = "GET"

                try:
                    retries = retries.increment(
                        method, url, response=http_response, _pool=self
                    )
                except MaxRetryError:
                    if retries.raise_on_redirect:
                        raise MaxRetryError(self, url, "too many redirects")
                    return http_response

                retries.sleep_for_retry(http_response)
                log.debug("Redirecting %s -> %s", url, redirect_location)
                redirect_url = urljoin(url, redirect_location)
                return self.urlopen(
                    method,
                    redirect_url,
                    body,
                    headers,
                    retries=retries,
                    redirect=redirect,
                    timeout=timeout,
                    **response_kw
                )

        # Check if we should retry the HTTP response.
        has_retry_after = bool(http_response.headers.get("Retry-After"))
        if retries.is_retry(method, http_response.status, has_retry_after):
            retries = retries.increment(method, url, response=http_response, _pool=self)
            log.debug("Retry: %s", url)
            retries.sleep(http_response)
            return self.urlopen(
                method,
                url,
                body=body,
                headers=headers,
                retries=retries,
                redirect=redirect,
                timeout=timeout,
                **response_kw
            )

        return http_response

    def _urlfetch_response_to_http_response(self, urlfetch_resp, **response_kw):

        if is_prod_appengine():
            # Production GAE handles deflate encoding automatically, but does
            # not remove the encoding header.
            content_encoding = urlfetch_resp.headers.get("content-encoding")

            if content_encoding == "deflate":
                del urlfetch_resp.headers["content-encoding"]

        transfer_encoding = urlfetch_resp.headers.get("transfer-encoding")
        # We have a full response's content,
        # so let's make sure we don't report ourselves as chunked data.
        if transfer_encoding == "chunked":
            encodings = transfer_encoding.split(",")
            encodings.remove("chunked")
            urlfetch_resp.headers["transfer-encoding"] = ",".join(encodings)

        original_response = HTTPResponse(
            # In order for decoding to work, we must present the content as
            # a file-like object.
            body=io.BytesIO(urlfetch_resp.content),
            msg=urlfetch_resp.header_msg,
            headers=urlfetch_resp.headers,
            status=urlfetch_resp.status_code,
            **response_kw
        )

        return HTTPResponse(
            body=io.BytesIO(urlfetch_resp.content),
            headers=urlfetch_resp.headers,
            status=urlfetch_resp.status_code,
            original_response=original_response,
            **response_kw
        )

    def _get_absolute_timeout(self, timeout):
        if timeout is Timeout.DEFAULT_TIMEOUT:
            return None  # Defer to URLFetch's default.
        if isinstance(timeout, Timeout):
            if timeout._read is not None or timeout._connect is not None:
                warnings.warn(
                    "URLFetch does not support granular timeout settings, "
                    "reverting to total or default URLFetch timeout.",
                    AppEnginePlatformWarning,
                )
            return timeout.total
        return timeout

    def _get_retries(self, retries, redirect):
        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, redirect=redirect, default=self.retries)

        if retries.connect or retries.read or retries.redirect:
            warnings.warn(
                "URLFetch only supports total retries and does not "
                "recognize connect, read, or redirect retry parameters.",
                AppEnginePlatformWarning,
            )

        return retries


# Alias methods from _appengine_environ to maintain public API interface.

is_appengine = _appengine_environ.is_appengine
is_appengine_sandbox = _appengine_environ.is_appengine_sandbox
is_local_appengine = _appengine_environ.is_local_appengine
is_prod_appengine = _appengine_environ.is_prod_appengine
is_prod_appengine_mvms = _appengine_environ.is_prod_appengine_mvms

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\requests\auth.py ===
"""
requests.auth
~~~~~~~~~~~~~

This module contains the authentication handlers for Requests.
"""

import hashlib
import os
import re
import threading
import time
import warnings
from base64 import b64encode

from ._internal_utils import to_native_string
from .compat import basestring, str, urlparse
from .cookies import extract_cookies_to_jar
from .utils import parse_dict_header

CONTENT_TYPE_FORM_URLENCODED = "application/x-www-form-urlencoded"
CONTENT_TYPE_MULTI_PART = "multipart/form-data"


def _basic_auth_str(username, password):
    """Returns a Basic Auth string."""

    # "I want us to put a big-ol' comment on top of it that
    # says that this behaviour is dumb but we need to preserve
    # it because people are relying on it."
    #    - Lukasa
    #
    # These are here solely to maintain backwards compatibility
    # for things like ints. This will be removed in 3.0.0.
    if not isinstance(username, basestring):
        warnings.warn(
            "Non-string usernames will no longer be supported in Requests "
            "3.0.0. Please convert the object you've passed in ({!r}) to "
            "a string or bytes object in the near future to avoid "
            "problems.".format(username),
            category=DeprecationWarning,
        )
        username = str(username)

    if not isinstance(password, basestring):
        warnings.warn(
            "Non-string passwords will no longer be supported in Requests "
            "3.0.0. Please convert the object you've passed in ({!r}) to "
            "a string or bytes object in the near future to avoid "
            "problems.".format(type(password)),
            category=DeprecationWarning,
        )
        password = str(password)
    # -- End Removal --

    if isinstance(username, str):
        username = username.encode("latin1")

    if isinstance(password, str):
        password = password.encode("latin1")

    authstr = "Basic " + to_native_string(
        b64encode(b":".join((username, password))).strip()
    )

    return authstr


class AuthBase:
    """Base class that all auth implementations derive from"""

    def __call__(self, r):
        raise NotImplementedError("Auth hooks must be callable.")


class HTTPBasicAuth(AuthBase):
    """Attaches HTTP Basic Authentication to the given Request object."""

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def __eq__(self, other):
        return all(
            [
                self.username == getattr(other, "username", None),
                self.password == getattr(other, "password", None),
            ]
        )

    def __ne__(self, other):
        return not self == other

    def __call__(self, r):
        r.headers["Authorization"] = _basic_auth_str(self.username, self.password)
        return r


class HTTPProxyAuth(HTTPBasicAuth):
    """Attaches HTTP Proxy Authentication to a given Request object."""

    def __call__(self, r):
        r.headers["Proxy-Authorization"] = _basic_auth_str(self.username, self.password)
        return r


class HTTPDigestAuth(AuthBase):
    """Attaches HTTP Digest Authentication to the given Request object."""

    def __init__(self, username, password):
        self.username = username
        self.password = password
        # Keep state in per-thread local storage
        self._thread_local = threading.local()

    def init_per_thread_state(self):
        # Ensure state is initialized just once per-thread
        if not hasattr(self._thread_local, "init"):
            self._thread_local.init = True
            self._thread_local.last_nonce = ""
            self._thread_local.nonce_count = 0
            self._thread_local.chal = {}
            self._thread_local.pos = None
            self._thread_local.num_401_calls = None

    def build_digest_header(self, method, url):
        """
        :rtype: str
        """

        realm = self._thread_local.chal["realm"]
        nonce = self._thread_local.chal["nonce"]
        qop = self._thread_local.chal.get("qop")
        algorithm = self._thread_local.chal.get("algorithm")
        opaque = self._thread_local.chal.get("opaque")
        hash_utf8 = None

        if algorithm is None:
            _algorithm = "MD5"
        else:
            _algorithm = algorithm.upper()
        # lambdas assume digest modules are imported at the top level
        if _algorithm == "MD5" or _algorithm == "MD5-SESS":

            def md5_utf8(x):
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.md5(x).hexdigest()

            hash_utf8 = md5_utf8
        elif _algorithm == "SHA":

            def sha_utf8(x):
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha1(x).hexdigest()

            hash_utf8 = sha_utf8
        elif _algorithm == "SHA-256":

            def sha256_utf8(x):
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha256(x).hexdigest()

            hash_utf8 = sha256_utf8
        elif _algorithm == "SHA-512":

            def sha512_utf8(x):
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha512(x).hexdigest()

            hash_utf8 = sha512_utf8

        KD = lambda s, d: hash_utf8(f"{s}:{d}")  # noqa:E731

        if hash_utf8 is None:
            return None

        # XXX not implemented yet
        entdig = None
        p_parsed = urlparse(url)
        #: path is request-uri defined in RFC 2616 which should not be empty
        path = p_parsed.path or "/"
        if p_parsed.query:
            path += f"?{p_parsed.query}"

        A1 = f"{self.username}:{realm}:{self.password}"
        A2 = f"{method}:{path}"

        HA1 = hash_utf8(A1)
        HA2 = hash_utf8(A2)

        if nonce == self._thread_local.last_nonce:
            self._thread_local.nonce_count += 1
        else:
            self._thread_local.nonce_count = 1
        ncvalue = f"{self._thread_local.nonce_count:08x}"
        s = str(self._thread_local.nonce_count).encode("utf-8")
        s += nonce.encode("utf-8")
        s += time.ctime().encode("utf-8")
        s += os.urandom(8)

        cnonce = hashlib.sha1(s).hexdigest()[:16]
        if _algorithm == "MD5-SESS":
            HA1 = hash_utf8(f"{HA1}:{nonce}:{cnonce}")

        if not qop:
            respdig = KD(HA1, f"{nonce}:{HA2}")
        elif qop == "auth" or "auth" in qop.split(","):
            noncebit = f"{nonce}:{ncvalue}:{cnonce}:auth:{HA2}"
            respdig = KD(HA1, noncebit)
        else:
            # XXX handle auth-int.
            return None

        self._thread_local.last_nonce = nonce

        # XXX should the partial digests be encoded too?
        base = (
            f'username="{self.username}", realm="{realm}", nonce="{nonce}", '
            f'uri="{path}", response="{respdig}"'
        )
        if opaque:
            base += f', opaque="{opaque}"'
        if algorithm:
            base += f', algorithm="{algorithm}"'
        if entdig:
            base += f', digest="{entdig}"'
        if qop:
            base += f', qop="auth", nc={ncvalue}, cnonce="{cnonce}"'

        return f"Digest {base}"

    def handle_redirect(self, r, **kwargs):
        """Reset num_401_calls counter on redirects."""
        if r.is_redirect:
            self._thread_local.num_401_calls = 1

    def handle_401(self, r, **kwargs):
        """
        Takes the given response and tries digest-auth, if needed.

        :rtype: requests.Response
        """

        # If response is not 4xx, do not auth
        # See https://github.com/psf/requests/issues/3772
        if not 400 <= r.status_code < 500:
            self._thread_local.num_401_calls = 1
            return r

        if self._thread_local.pos is not None:
            # Rewind the file position indicator of the body to where
            # it was to resend the request.
            r.request.body.seek(self._thread_local.pos)
        s_auth = r.headers.get("www-authenticate", "")

        if "digest" in s_auth.lower() and self._thread_local.num_401_calls < 2:
            self._thread_local.num_401_calls += 1
            pat = re.compile(r"digest ", flags=re.IGNORECASE)
            self._thread_local.chal = parse_dict_header(pat.sub("", s_auth, count=1))

            # Consume content and release the original connection
            # to allow our new request to reuse the same one.
            r.content
            r.close()
            prep = r.request.copy()
            extract_cookies_to_jar(prep._cookies, r.request, r.raw)
            prep.prepare_cookies(prep._cookies)

            prep.headers["Authorization"] = self.build_digest_header(
                prep.method, prep.url
            )
            _r = r.connection.send(prep, **kwargs)
            _r.history.append(r)
            _r.request = prep

            return _r

        self._thread_local.num_401_calls = 1
        return r

    def __call__(self, r):
        # Initialize per-thread state, if needed
        self.init_per_thread_state()
        # If we have a saved nonce, skip the 401
        if self._thread_local.last_nonce:
            r.headers["Authorization"] = self.build_digest_header(r.method, r.url)
        try:
            self._thread_local.pos = r.body.tell()
        except AttributeError:
            # In the case of HTTPDigestAuth being reused and the body of
            # the previous request was a file-like object, pos has the
            # file position of the previous body. Ensure it's set to
            # None.
            self._thread_local.pos = None
        r.register_hook("response", self.handle_401)
        r.register_hook("response", self.handle_redirect)
        self._thread_local.num_401_calls = 1

        return r

    def __eq__(self, other):
        return all(
            [
                self.username == getattr(other, "username", None),
                self.password == getattr(other, "password", None),
            ]
        )

    def __ne__(self, other):
        return not self == other

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\rigidbody.py ===
from sympy import Symbol, S
from sympy.physics.vector import ReferenceFrame, Dyadic, Point, dot
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.mechanics.inertia import inertia_of_point_mass, Inertia
from sympy.utilities.exceptions import sympy_deprecation_warning

__all__ = ['RigidBody']


class RigidBody(BodyBase):
    """An idealized rigid body.

    Explanation
    ===========

    This is essentially a container which holds the various components which
    describe a rigid body: a name, mass, center of mass, reference frame, and
    inertia.

    All of these need to be supplied on creation, but can be changed
    afterwards.

    Attributes
    ==========

    name : string
        The body's name.
    masscenter : Point
        The point which represents the center of mass of the rigid body.
    frame : ReferenceFrame
        The ReferenceFrame which the rigid body is fixed in.
    mass : Sympifyable
        The body's mass.
    inertia : (Dyadic, Point)
        The body's inertia about a point; stored in a tuple as shown above.
    potential_energy : Sympifyable
        The potential energy of the RigidBody.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.physics.mechanics import ReferenceFrame, Point, RigidBody
    >>> from sympy.physics.mechanics import outer
    >>> m = Symbol('m')
    >>> A = ReferenceFrame('A')
    >>> P = Point('P')
    >>> I = outer (A.x, A.x)
    >>> inertia_tuple = (I, P)
    >>> B = RigidBody('B', P, A, m, inertia_tuple)
    >>> # Or you could change them afterwards
    >>> m2 = Symbol('m2')
    >>> B.mass = m2

    """

    def __init__(self, name, masscenter=None, frame=None, mass=None,
                 inertia=None):
        super().__init__(name, masscenter, mass)
        if frame is None:
            frame = ReferenceFrame(f'{name}_frame')
        self.frame = frame
        if inertia is None:
            ixx = Symbol(f'{name}_ixx')
            iyy = Symbol(f'{name}_iyy')
            izz = Symbol(f'{name}_izz')
            izx = Symbol(f'{name}_izx')
            ixy = Symbol(f'{name}_ixy')
            iyz = Symbol(f'{name}_iyz')
            inertia = Inertia.from_inertia_scalars(self.masscenter, self.frame,
                                                   ixx, iyy, izz, ixy, iyz, izx)
        self.inertia = inertia

    def __repr__(self):
        return (f'{self.__class__.__name__}({repr(self.name)}, masscenter='
                f'{repr(self.masscenter)}, frame={repr(self.frame)}, mass='
                f'{repr(self.mass)}, inertia={repr(self.inertia)})')

    @property
    def frame(self):
        """The ReferenceFrame fixed to the body."""
        return self._frame

    @frame.setter
    def frame(self, F):
        if not isinstance(F, ReferenceFrame):
            raise TypeError("RigidBody frame must be a ReferenceFrame object.")
        self._frame = F

    @property
    def x(self):
        """The basis Vector for the body, in the x direction. """
        return self.frame.x

    @property
    def y(self):
        """The basis Vector for the body, in the y direction. """
        return self.frame.y

    @property
    def z(self):
        """The basis Vector for the body, in the z direction. """
        return self.frame.z

    @property
    def inertia(self):
        """The body's inertia about a point; stored as (Dyadic, Point)."""
        return self._inertia

    @inertia.setter
    def inertia(self, I):
        # check if I is of the form (Dyadic, Point)
        if len(I) != 2 or not isinstance(I[0], Dyadic) or not isinstance(I[1], Point):
            raise TypeError("RigidBody inertia must be a tuple of the form (Dyadic, Point).")

        self._inertia = Inertia(I[0], I[1])
        # have I S/O, want I S/S*
        # I S/O = I S/S* + I S*/O; I S/S* = I S/O - I S*/O
        # I_S/S* = I_S/O - I_S*/O
        I_Ss_O = inertia_of_point_mass(self.mass,
                                       self.masscenter.pos_from(I[1]),
                                       self.frame)
        self._central_inertia = I[0] - I_Ss_O

    @property
    def central_inertia(self):
        """The body's central inertia dyadic."""
        return self._central_inertia

    @central_inertia.setter
    def central_inertia(self, I):
        if not isinstance(I, Dyadic):
            raise TypeError("RigidBody inertia must be a Dyadic object.")
        self.inertia = Inertia(I, self.masscenter)

    def linear_momentum(self, frame):
        """ Linear momentum of the rigid body.

        Explanation
        ===========

        The linear momentum L, of a rigid body B, with respect to frame N is
        given by:

        ``L = m * v``

        where m is the mass of the rigid body, and v is the velocity of the mass
        center of B in the frame N.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which linear momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer
        >>> from sympy.physics.mechanics import RigidBody, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> m, v = dynamicsymbols('m v')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> P.set_vel(N, v * N.x)
        >>> I = outer (N.x, N.x)
        >>> Inertia_tuple = (I, P)
        >>> B = RigidBody('B', P, N, m, Inertia_tuple)
        >>> B.linear_momentum(N)
        m*v*N.x

        """

        return self.mass * self.masscenter.vel(frame)

    def angular_momentum(self, point, frame):
        """Returns the angular momentum of the rigid body about a point in the
        given frame.

        Explanation
        ===========

        The angular momentum H of a rigid body B about some point O in a frame N
        is given by:

        ``H = dot(I, w) + cross(r, m * v)``

        where I and m are the central inertia dyadic and mass of rigid body B, w
        is the angular velocity of body B in the frame N, r is the position
        vector from point O to the mass center of B, and v is the velocity of
        the mass center in the frame N.

        Parameters
        ==========

        point : Point
            The point about which angular momentum is desired.
        frame : ReferenceFrame
            The frame in which angular momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer
        >>> from sympy.physics.mechanics import RigidBody, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> m, v, r, omega = dynamicsymbols('m v r omega')
        >>> N = ReferenceFrame('N')
        >>> b = ReferenceFrame('b')
        >>> b.set_ang_vel(N, omega * b.x)
        >>> P = Point('P')
        >>> P.set_vel(N, 1 * N.x)
        >>> I = outer(b.x, b.x)
        >>> B = RigidBody('B', P, b, m, (I, P))
        >>> B.angular_momentum(P, N)
        omega*b.x

        """
        I = self.central_inertia
        w = self.frame.ang_vel_in(frame)
        m = self.mass
        r = self.masscenter.pos_from(point)
        v = self.masscenter.vel(frame)

        return I.dot(w) + r.cross(m * v)

    def kinetic_energy(self, frame):
        """Kinetic energy of the rigid body.

        Explanation
        ===========

        The kinetic energy, T, of a rigid body, B, is given by:

        ``T = 1/2 * (dot(dot(I, w), w) + dot(m * v, v))``

        where I and m are the central inertia dyadic and mass of rigid body B
        respectively, w is the body's angular velocity, and v is the velocity of
        the body's mass center in the supplied ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The RigidBody's angular velocity and the velocity of it's mass
            center are typically defined with respect to an inertial frame but
            any relevant frame in which the velocities are known can be
            supplied.

        Examples
        ========

        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer
        >>> from sympy.physics.mechanics import RigidBody
        >>> from sympy import symbols
        >>> m, v, r, omega = symbols('m v r omega')
        >>> N = ReferenceFrame('N')
        >>> b = ReferenceFrame('b')
        >>> b.set_ang_vel(N, omega * b.x)
        >>> P = Point('P')
        >>> P.set_vel(N, v * N.x)
        >>> I = outer (b.x, b.x)
        >>> inertia_tuple = (I, P)
        >>> B = RigidBody('B', P, b, m, inertia_tuple)
        >>> B.kinetic_energy(N)
        m*v**2/2 + omega**2/2

        """

        rotational_KE = S.Half * dot(
            self.frame.ang_vel_in(frame),
            dot(self.central_inertia, self.frame.ang_vel_in(frame)))
        translational_KE = S.Half * self.mass * dot(self.masscenter.vel(frame),
                                                    self.masscenter.vel(frame))
        return rotational_KE + translational_KE

    def set_potential_energy(self, scalar):
        sympy_deprecation_warning(
            """
The sympy.physics.mechanics.RigidBody.set_potential_energy()
method is deprecated. Instead use

    B.potential_energy = scalar
            """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-set-potential-energy",
        )
        self.potential_energy = scalar

    def parallel_axis(self, point, frame=None):
        """Returns the inertia dyadic of the body with respect to another point.

        Parameters
        ==========

        point : sympy.physics.vector.Point
            The point to express the inertia dyadic about.
        frame : sympy.physics.vector.ReferenceFrame
            The reference frame used to construct the dyadic.

        Returns
        =======

        inertia : sympy.physics.vector.Dyadic
            The inertia dyadic of the rigid body expressed about the provided
            point.

        """
        if frame is None:
            frame = self.frame
        return self.central_inertia + inertia_of_point_mass(
            self.mass, self.masscenter.pos_from(point), frame)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\visualization.py ===
"""
Plotting (requires matplotlib)
"""

from colorsys import hsv_to_rgb, hls_to_rgb
from .libmp import NoConvergence
from .libmp.backend import xrange

class VisualizationMethods(object):
    plot_ignore = (ValueError, ArithmeticError, ZeroDivisionError, NoConvergence)

def plot(ctx, f, xlim=[-5,5], ylim=None, points=200, file=None, dpi=None,
    singularities=[], axes=None):
    r"""
    Shows a simple 2D plot of a function `f(x)` or list of functions
    `[f_0(x), f_1(x), \ldots, f_n(x)]` over a given interval
    specified by *xlim*. Some examples::

        plot(lambda x: exp(x)*li(x), [1, 4])
        plot([cos, sin], [-4, 4])
        plot([fresnels, fresnelc], [-4, 4])
        plot([sqrt, cbrt], [-4, 4])
        plot(lambda t: zeta(0.5+t*j), [-20, 20])
        plot([floor, ceil, abs, sign], [-5, 5])

    Points where the function raises a numerical exception or
    returns an infinite value are removed from the graph.
    Singularities can also be excluded explicitly
    as follows (useful for removing erroneous vertical lines)::

        plot(cot, ylim=[-5, 5])   # bad
        plot(cot, ylim=[-5, 5], singularities=[-pi, 0, pi])  # good

    For parts where the function assumes complex values, the
    real part is plotted with dashes and the imaginary part
    is plotted with dots.

    .. note :: This function requires matplotlib (pylab).
    """
    if file:
        axes = None
    fig = None
    if not axes:
        import pylab
        fig = pylab.figure()
        axes = fig.add_subplot(111)
    if not isinstance(f, (tuple, list)):
        f = [f]
    a, b = xlim
    colors = ['b', 'r', 'g', 'm', 'k']
    for n, func in enumerate(f):
        x = ctx.arange(a, b, (b-a)/float(points))
        segments = []
        segment = []
        in_complex = False
        for i in xrange(len(x)):
            try:
                if i != 0:
                    for sing in singularities:
                        if x[i-1] <= sing and x[i] >= sing:
                            raise ValueError
                v = func(x[i])
                if ctx.isnan(v) or abs(v) > 1e300:
                    raise ValueError
                if hasattr(v, "imag") and v.imag:
                    re = float(v.real)
                    im = float(v.imag)
                    if not in_complex:
                        in_complex = True
                        segments.append(segment)
                        segment = []
                    segment.append((float(x[i]), re, im))
                else:
                    if in_complex:
                        in_complex = False
                        segments.append(segment)
                        segment = []
                    if hasattr(v, "real"):
                        v = v.real
                    segment.append((float(x[i]), v))
            except ctx.plot_ignore:
                if segment:
                    segments.append(segment)
                segment = []
        if segment:
            segments.append(segment)
        for segment in segments:
            x = [s[0] for s in segment]
            y = [s[1] for s in segment]
            if not x:
                continue
            c = colors[n % len(colors)]
            if len(segment[0]) == 3:
                z = [s[2] for s in segment]
                axes.plot(x, y, '--'+c, linewidth=3)
                axes.plot(x, z, ':'+c, linewidth=3)
            else:
                axes.plot(x, y, c, linewidth=3)
    axes.set_xlim([float(_) for _ in xlim])
    if ylim:
        axes.set_ylim([float(_) for _ in ylim])
    axes.set_xlabel('x')
    axes.set_ylabel('f(x)')
    axes.grid(True)
    if fig:
        if file:
            pylab.savefig(file, dpi=dpi)
        else:
            pylab.show()

def default_color_function(ctx, z):
    if ctx.isinf(z):
        return (1.0, 1.0, 1.0)
    if ctx.isnan(z):
        return (0.5, 0.5, 0.5)
    pi = 3.1415926535898
    a = (float(ctx.arg(z)) + ctx.pi) / (2*ctx.pi)
    a = (a + 0.5) % 1.0
    b = 1.0 - float(1/(1.0+abs(z)**0.3))
    return hls_to_rgb(a, b, 0.8)

blue_orange_colors = [
  (-1.0,  (0.0, 0.0, 0.0)),
  (-0.95, (0.1, 0.2, 0.5)),   # dark blue
  (-0.5,  (0.0, 0.5, 1.0)),   # blueish
  (-0.05, (0.4, 0.8, 0.8)),   # cyanish
  ( 0.0,  (1.0, 1.0, 1.0)),
  ( 0.05, (1.0, 0.9, 0.3)),   # yellowish
  ( 0.5,  (0.9, 0.5, 0.0)),   # orangeish
  ( 0.95, (0.7, 0.1, 0.0)),   # redish
  ( 1.0,  (0.0, 0.0, 0.0)),
  ( 2.0,  (0.0, 0.0, 0.0)),
]

def phase_color_function(ctx, z):
    if ctx.isinf(z):
        return (1.0, 1.0, 1.0)
    if ctx.isnan(z):
        return (0.5, 0.5, 0.5)
    pi = 3.1415926535898
    w = float(ctx.arg(z)) / pi
    w = max(min(w, 1.0), -1.0)
    for i in range(1,len(blue_orange_colors)):
        if blue_orange_colors[i][0] > w:
            a, (ra, ga, ba) = blue_orange_colors[i-1]
            b, (rb, gb, bb) = blue_orange_colors[i]
            s = (w-a) / (b-a)
            return ra+(rb-ra)*s, ga+(gb-ga)*s, ba+(bb-ba)*s

def cplot(ctx, f, re=[-5,5], im=[-5,5], points=2000, color=None,
    verbose=False, file=None, dpi=None, axes=None):
    """
    Plots the given complex-valued function *f* over a rectangular part
    of the complex plane specified by the pairs of intervals *re* and *im*.
    For example::

        cplot(lambda z: z, [-2, 2], [-10, 10])
        cplot(exp)
        cplot(zeta, [0, 1], [0, 50])

    By default, the complex argument (phase) is shown as color (hue) and
    the magnitude is show as brightness. You can also supply a
    custom color function (*color*). This function should take a
    complex number as input and return an RGB 3-tuple containing
    floats in the range 0.0-1.0.

    Alternatively, you can select a builtin color function by passing
    a string as *color*:

      * "default" - default color scheme
      * "phase" - a color scheme that only renders the phase of the function,
         with white for positive reals, black for negative reals, gold in the
         upper half plane, and blue in the lower half plane.

    To obtain a sharp image, the number of points may need to be
    increased to 100,000 or thereabout. Since evaluating the
    function that many times is likely to be slow, the 'verbose'
    option is useful to display progress.

    .. note :: This function requires matplotlib (pylab).
    """
    if color is None or color == "default":
        color = ctx.default_color_function
    if color == "phase":
        color = ctx.phase_color_function
    import pylab
    if file:
        axes = None
    fig = None
    if not axes:
        fig = pylab.figure()
        axes = fig.add_subplot(111)
    rea, reb = re
    ima, imb = im
    dre = reb - rea
    dim = imb - ima
    M = int(ctx.sqrt(points*dre/dim)+1)
    N = int(ctx.sqrt(points*dim/dre)+1)
    x = pylab.linspace(rea, reb, M)
    y = pylab.linspace(ima, imb, N)
    # Note: we have to be careful to get the right rotation.
    # Test with these plots:
    #   cplot(lambda z: z if z.real < 0 else 0)
    #   cplot(lambda z: z if z.imag < 0 else 0)
    w = pylab.zeros((N, M, 3))
    for n in xrange(N):
        for m in xrange(M):
            z = ctx.mpc(x[m], y[n])
            try:
                v = color(f(z))
            except ctx.plot_ignore:
                v = (0.5, 0.5, 0.5)
            w[n,m] = v
        if verbose:
            print(str(n) + ' of ' + str(N))
    rea, reb, ima, imb = [float(_) for _ in [rea, reb, ima, imb]]
    axes.imshow(w, extent=(rea, reb, ima, imb), origin='lower')
    axes.set_xlabel('Re(z)')
    axes.set_ylabel('Im(z)')
    if fig:
        if file:
            pylab.savefig(file, dpi=dpi)
        else:
            pylab.show()

def splot(ctx, f, u=[-5,5], v=[-5,5], points=100, keep_aspect=True, \
          wireframe=False, file=None, dpi=None, axes=None):
    """
    Plots the surface defined by `f`.

    If `f` returns a single component, then this plots the surface
    defined by `z = f(x,y)` over the rectangular domain with
    `x = u` and `y = v`.

    If `f` returns three components, then this plots the parametric
    surface `x, y, z = f(u,v)` over the pairs of intervals `u` and `v`.

    For example, to plot a simple function::

        >>> from mpmath import *
        >>> f = lambda x, y: sin(x+y)*cos(y)
        >>> splot(f, [-pi,pi], [-pi,pi])    # doctest: +SKIP

    Plotting a donut::

        >>> r, R = 1, 2.5
        >>> f = lambda u, v: [r*cos(u), (R+r*sin(u))*cos(v), (R+r*sin(u))*sin(v)]
        >>> splot(f, [0, 2*pi], [0, 2*pi])    # doctest: +SKIP

    .. note :: This function requires matplotlib (pylab) 0.98.5.3 or higher.
    """
    import pylab
    import mpl_toolkits.mplot3d as mplot3d
    if file:
        axes = None
    fig = None
    if not axes:
        fig = pylab.figure()
        axes = mplot3d.axes3d.Axes3D(fig)
    ua, ub = u
    va, vb = v
    du = ub - ua
    dv = vb - va
    if not isinstance(points, (list, tuple)):
        points = [points, points]
    M, N = points
    u = pylab.linspace(ua, ub, M)
    v = pylab.linspace(va, vb, N)
    x, y, z = [pylab.zeros((M, N)) for i in xrange(3)]
    xab, yab, zab = [[0, 0] for i in xrange(3)]
    for n in xrange(N):
        for m in xrange(M):
            fdata = f(ctx.convert(u[m]), ctx.convert(v[n]))
            try:
                x[m,n], y[m,n], z[m,n] = fdata
            except TypeError:
                x[m,n], y[m,n], z[m,n] = u[m], v[n], fdata
            for c, cab in [(x[m,n], xab), (y[m,n], yab), (z[m,n], zab)]:
                if c < cab[0]:
                    cab[0] = c
                if c > cab[1]:
                    cab[1] = c
    if wireframe:
        axes.plot_wireframe(x, y, z, rstride=4, cstride=4)
    else:
        axes.plot_surface(x, y, z, rstride=4, cstride=4)
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    if keep_aspect:
        dx, dy, dz = [cab[1] - cab[0] for cab in [xab, yab, zab]]
        maxd = max(dx, dy, dz)
        if dx < maxd:
            delta = maxd - dx
            axes.set_xlim3d(xab[0] - delta / 2.0, xab[1] + delta / 2.0)
        if dy < maxd:
            delta = maxd - dy
            axes.set_ylim3d(yab[0] - delta / 2.0, yab[1] + delta / 2.0)
        if dz < maxd:
            delta = maxd - dz
            axes.set_zlim3d(zab[0] - delta / 2.0, zab[1] + delta / 2.0)
    if fig:
        if file:
            pylab.savefig(file, dpi=dpi)
        else:
            pylab.show()


VisualizationMethods.plot = plot
VisualizationMethods.default_color_function = default_color_function
VisualizationMethods.phase_color_function = phase_color_function
VisualizationMethods.cplot = cplot
VisualizationMethods.splot = splot

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\postgresql\types.py ===
# dialects/postgresql/types.py
# Copyright (C) 2013-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
from __future__ import annotations

import datetime as dt
from typing import Any
from typing import Optional
from typing import overload
from typing import Type
from typing import TYPE_CHECKING
from uuid import UUID as _python_UUID

from ...sql import sqltypes
from ...sql import type_api
from ...util.typing import Literal

if TYPE_CHECKING:
    from ...engine.interfaces import Dialect
    from ...sql.operators import OperatorType
    from ...sql.type_api import _LiteralProcessorType
    from ...sql.type_api import TypeEngine

_DECIMAL_TYPES = (1231, 1700)
_FLOAT_TYPES = (700, 701, 1021, 1022)
_INT_TYPES = (20, 21, 23, 26, 1005, 1007, 1016)


class PGUuid(sqltypes.UUID[sqltypes._UUID_RETURN]):
    render_bind_cast = True
    render_literal_cast = True

    if TYPE_CHECKING:

        @overload
        def __init__(
            self: PGUuid[_python_UUID], as_uuid: Literal[True] = ...
        ) -> None: ...

        @overload
        def __init__(
            self: PGUuid[str], as_uuid: Literal[False] = ...
        ) -> None: ...

        def __init__(self, as_uuid: bool = True) -> None: ...


class BYTEA(sqltypes.LargeBinary):
    __visit_name__ = "BYTEA"


class _NetworkAddressTypeMixin:

    def coerce_compared_value(
        self, op: Optional[OperatorType], value: Any
    ) -> TypeEngine[Any]:
        if TYPE_CHECKING:
            assert isinstance(self, TypeEngine)
        return self


class INET(_NetworkAddressTypeMixin, sqltypes.TypeEngine[str]):
    __visit_name__ = "INET"


PGInet = INET


class CIDR(_NetworkAddressTypeMixin, sqltypes.TypeEngine[str]):
    __visit_name__ = "CIDR"


PGCidr = CIDR


class MACADDR(_NetworkAddressTypeMixin, sqltypes.TypeEngine[str]):
    __visit_name__ = "MACADDR"


PGMacAddr = MACADDR


class MACADDR8(_NetworkAddressTypeMixin, sqltypes.TypeEngine[str]):
    __visit_name__ = "MACADDR8"


PGMacAddr8 = MACADDR8


class MONEY(sqltypes.TypeEngine[str]):
    r"""Provide the PostgreSQL MONEY type.

    Depending on driver, result rows using this type may return a
    string value which includes currency symbols.

    For this reason, it may be preferable to provide conversion to a
    numerically-based currency datatype using :class:`_types.TypeDecorator`::

        import re
        import decimal
        from sqlalchemy import Dialect
        from sqlalchemy import TypeDecorator


        class NumericMoney(TypeDecorator):
            impl = MONEY

            def process_result_value(self, value: Any, dialect: Dialect) -> None:
                if value is not None:
                    # adjust this for the currency and numeric
                    m = re.match(r"\$([\d.]+)", value)
                    if m:
                        value = decimal.Decimal(m.group(1))
                return value

    Alternatively, the conversion may be applied as a CAST using
    the :meth:`_types.TypeDecorator.column_expression` method as follows::

        import decimal
        from sqlalchemy import cast
        from sqlalchemy import TypeDecorator


        class NumericMoney(TypeDecorator):
            impl = MONEY

            def column_expression(self, column: Any):
                return cast(column, Numeric())

    .. versionadded:: 1.2

    """  # noqa: E501

    __visit_name__ = "MONEY"


class OID(sqltypes.TypeEngine[int]):
    """Provide the PostgreSQL OID type."""

    __visit_name__ = "OID"


class REGCONFIG(sqltypes.TypeEngine[str]):
    """Provide the PostgreSQL REGCONFIG type.

    .. versionadded:: 2.0.0rc1

    """

    __visit_name__ = "REGCONFIG"


class TSQUERY(sqltypes.TypeEngine[str]):
    """Provide the PostgreSQL TSQUERY type.

    .. versionadded:: 2.0.0rc1

    """

    __visit_name__ = "TSQUERY"


class REGCLASS(sqltypes.TypeEngine[str]):
    """Provide the PostgreSQL REGCLASS type.

    .. versionadded:: 1.2.7

    """

    __visit_name__ = "REGCLASS"


class TIMESTAMP(sqltypes.TIMESTAMP):
    """Provide the PostgreSQL TIMESTAMP type."""

    __visit_name__ = "TIMESTAMP"

    def __init__(
        self, timezone: bool = False, precision: Optional[int] = None
    ) -> None:
        """Construct a TIMESTAMP.

        :param timezone: boolean value if timezone present, default False
        :param precision: optional integer precision value

         .. versionadded:: 1.4

        """
        super().__init__(timezone=timezone)
        self.precision = precision


class TIME(sqltypes.TIME):
    """PostgreSQL TIME type."""

    __visit_name__ = "TIME"

    def __init__(
        self, timezone: bool = False, precision: Optional[int] = None
    ) -> None:
        """Construct a TIME.

        :param timezone: boolean value if timezone present, default False
        :param precision: optional integer precision value

         .. versionadded:: 1.4

        """
        super().__init__(timezone=timezone)
        self.precision = precision


class INTERVAL(type_api.NativeForEmulated, sqltypes._AbstractInterval):
    """PostgreSQL INTERVAL type."""

    __visit_name__ = "INTERVAL"
    native = True

    def __init__(
        self, precision: Optional[int] = None, fields: Optional[str] = None
    ) -> None:
        """Construct an INTERVAL.

        :param precision: optional integer precision value
        :param fields: string fields specifier.  allows storage of fields
         to be limited, such as ``"YEAR"``, ``"MONTH"``, ``"DAY TO HOUR"``,
         etc.

         .. versionadded:: 1.2

        """
        self.precision = precision
        self.fields = fields

    @classmethod
    def adapt_emulated_to_native(
        cls, interval: sqltypes.Interval, **kw: Any  # type: ignore[override]
    ) -> INTERVAL:
        return INTERVAL(precision=interval.second_precision)

    @property
    def _type_affinity(self) -> Type[sqltypes.Interval]:
        return sqltypes.Interval

    def as_generic(self, allow_nulltype: bool = False) -> sqltypes.Interval:
        return sqltypes.Interval(native=True, second_precision=self.precision)

    @property
    def python_type(self) -> Type[dt.timedelta]:
        return dt.timedelta

    def literal_processor(
        self, dialect: Dialect
    ) -> Optional[_LiteralProcessorType[dt.timedelta]]:
        def process(value: dt.timedelta) -> str:
            return f"make_interval(secs=>{value.total_seconds()})"

        return process


PGInterval = INTERVAL


class BIT(sqltypes.TypeEngine[int]):
    __visit_name__ = "BIT"

    def __init__(
        self, length: Optional[int] = None, varying: bool = False
    ) -> None:
        if varying:
            # BIT VARYING can be unlimited-length, so no default
            self.length = length
        else:
            # BIT without VARYING defaults to length 1
            self.length = length or 1
        self.varying = varying


PGBit = BIT


class TSVECTOR(sqltypes.TypeEngine[str]):
    """The :class:`_postgresql.TSVECTOR` type implements the PostgreSQL
    text search type TSVECTOR.

    It can be used to do full text queries on natural language
    documents.

    .. seealso::

        :ref:`postgresql_match`

    """

    __visit_name__ = "TSVECTOR"


class CITEXT(sqltypes.TEXT):
    """Provide the PostgreSQL CITEXT type.

    .. versionadded:: 2.0.7

    """

    __visit_name__ = "CITEXT"

    def coerce_compared_value(
        self, op: Optional[OperatorType], value: Any
    ) -> TypeEngine[Any]:
        return self

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\vector\fieldfunctions.py ===
from sympy.core.function import diff
from sympy.core.singleton import S
from sympy.integrals.integrals import integrate
from sympy.physics.vector import Vector, express
from sympy.physics.vector.frame import _check_frame
from sympy.physics.vector.vector import _check_vector


__all__ = ['curl', 'divergence', 'gradient', 'is_conservative',
           'is_solenoidal', 'scalar_potential',
           'scalar_potential_difference']


def curl(vect, frame):
    """
    Returns the curl of a vector field computed wrt the coordinate
    symbols of the given frame.

    Parameters
    ==========

    vect : Vector
        The vector operand

    frame : ReferenceFrame
        The reference frame to calculate the curl in

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame
    >>> from sympy.physics.vector import curl
    >>> R = ReferenceFrame('R')
    >>> v1 = R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z
    >>> curl(v1, R)
    0
    >>> v2 = R[0]*R[1]*R[2]*R.x
    >>> curl(v2, R)
    R_x*R_y*R.y - R_x*R_z*R.z

    """

    _check_vector(vect)
    if vect == 0:
        return Vector(0)
    vect = express(vect, frame, variables=True)
    # A mechanical approach to avoid looping overheads
    vectx = vect.dot(frame.x)
    vecty = vect.dot(frame.y)
    vectz = vect.dot(frame.z)
    outvec = Vector(0)
    outvec += (diff(vectz, frame[1]) - diff(vecty, frame[2])) * frame.x
    outvec += (diff(vectx, frame[2]) - diff(vectz, frame[0])) * frame.y
    outvec += (diff(vecty, frame[0]) - diff(vectx, frame[1])) * frame.z
    return outvec


def divergence(vect, frame):
    """
    Returns the divergence of a vector field computed wrt the coordinate
    symbols of the given frame.

    Parameters
    ==========

    vect : Vector
        The vector operand

    frame : ReferenceFrame
        The reference frame to calculate the divergence in

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame
    >>> from sympy.physics.vector import divergence
    >>> R = ReferenceFrame('R')
    >>> v1 = R[0]*R[1]*R[2] * (R.x+R.y+R.z)
    >>> divergence(v1, R)
    R_x*R_y + R_x*R_z + R_y*R_z
    >>> v2 = 2*R[1]*R[2]*R.y
    >>> divergence(v2, R)
    2*R_z

    """

    _check_vector(vect)
    if vect == 0:
        return S.Zero
    vect = express(vect, frame, variables=True)
    vectx = vect.dot(frame.x)
    vecty = vect.dot(frame.y)
    vectz = vect.dot(frame.z)
    out = S.Zero
    out += diff(vectx, frame[0])
    out += diff(vecty, frame[1])
    out += diff(vectz, frame[2])
    return out


def gradient(scalar, frame):
    """
    Returns the vector gradient of a scalar field computed wrt the
    coordinate symbols of the given frame.

    Parameters
    ==========

    scalar : sympifiable
        The scalar field to take the gradient of

    frame : ReferenceFrame
        The frame to calculate the gradient in

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame
    >>> from sympy.physics.vector import gradient
    >>> R = ReferenceFrame('R')
    >>> s1 = R[0]*R[1]*R[2]
    >>> gradient(s1, R)
    R_y*R_z*R.x + R_x*R_z*R.y + R_x*R_y*R.z
    >>> s2 = 5*R[0]**2*R[2]
    >>> gradient(s2, R)
    10*R_x*R_z*R.x + 5*R_x**2*R.z

    """

    _check_frame(frame)
    outvec = Vector(0)
    scalar = express(scalar, frame, variables=True)
    for i, x in enumerate(frame):
        outvec += diff(scalar, frame[i]) * x  # noqa: PLR1736
    return outvec


def is_conservative(field):
    """
    Checks if a field is conservative.

    Parameters
    ==========

    field : Vector
        The field to check for conservative property

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame
    >>> from sympy.physics.vector import is_conservative
    >>> R = ReferenceFrame('R')
    >>> is_conservative(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z)
    True
    >>> is_conservative(R[2] * R.y)
    False

    """

    # Field is conservative irrespective of frame
    # Take the first frame in the result of the separate method of Vector
    if field == Vector(0):
        return True
    frame = list(field.separate())[0]
    return curl(field, frame).simplify() == Vector(0)


def is_solenoidal(field):
    """
    Checks if a field is solenoidal.

    Parameters
    ==========

    field : Vector
        The field to check for solenoidal property

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame
    >>> from sympy.physics.vector import is_solenoidal
    >>> R = ReferenceFrame('R')
    >>> is_solenoidal(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z)
    True
    >>> is_solenoidal(R[1] * R.y)
    False

    """

    # Field is solenoidal irrespective of frame
    # Take the first frame in the result of the separate method in Vector
    if field == Vector(0):
        return True
    frame = list(field.separate())[0]
    return divergence(field, frame).simplify() is S.Zero


def scalar_potential(field, frame):
    """
    Returns the scalar potential function of a field in a given frame
    (without the added integration constant).

    Parameters
    ==========

    field : Vector
        The vector field whose scalar potential function is to be
        calculated

    frame : ReferenceFrame
        The frame to do the calculation in

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame
    >>> from sympy.physics.vector import scalar_potential, gradient
    >>> R = ReferenceFrame('R')
    >>> scalar_potential(R.z, R) == R[2]
    True
    >>> scalar_field = 2*R[0]**2*R[1]*R[2]
    >>> grad_field = gradient(scalar_field, R)
    >>> scalar_potential(grad_field, R)
    2*R_x**2*R_y*R_z

    """

    # Check whether field is conservative
    if not is_conservative(field):
        raise ValueError("Field is not conservative")
    if field == Vector(0):
        return S.Zero
    # Express the field exntirely in frame
    # Substitute coordinate variables also
    _check_frame(frame)
    field = express(field, frame, variables=True)
    # Make a list of dimensions of the frame
    dimensions = list(frame)
    # Calculate scalar potential function
    temp_function = integrate(field.dot(dimensions[0]), frame[0])
    for i, dim in enumerate(dimensions[1:]):
        partial_diff = diff(temp_function, frame[i + 1])
        partial_diff = field.dot(dim) - partial_diff
        temp_function += integrate(partial_diff, frame[i + 1])
    return temp_function


def scalar_potential_difference(field, frame, point1, point2, origin):
    """
    Returns the scalar potential difference between two points in a
    certain frame, wrt a given field.

    If a scalar field is provided, its values at the two points are
    considered. If a conservative vector field is provided, the values
    of its scalar potential function at the two points are used.

    Returns (potential at position 2) - (potential at position 1)

    Parameters
    ==========

    field : Vector/sympyfiable
        The field to calculate wrt

    frame : ReferenceFrame
        The frame to do the calculations in

    point1 : Point
        The initial Point in given frame

    position2 : Point
        The second Point in the given frame

    origin : Point
        The Point to use as reference point for position vector
        calculation

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, Point
    >>> from sympy.physics.vector import scalar_potential_difference
    >>> R = ReferenceFrame('R')
    >>> O = Point('O')
    >>> P = O.locatenew('P', R[0]*R.x + R[1]*R.y + R[2]*R.z)
    >>> vectfield = 4*R[0]*R[1]*R.x + 2*R[0]**2*R.y
    >>> scalar_potential_difference(vectfield, R, O, P, O)
    2*R_x**2*R_y
    >>> Q = O.locatenew('O', 3*R.x + R.y + 2*R.z)
    >>> scalar_potential_difference(vectfield, R, P, Q, O)
    -2*R_x**2*R_y + 18

    """

    _check_frame(frame)
    if isinstance(field, Vector):
        # Get the scalar potential function
        scalar_fn = scalar_potential(field, frame)
    else:
        # Field is a scalar
        scalar_fn = field
    # Express positions in required frame
    position1 = express(point1.pos_from(origin), frame, variables=True)
    position2 = express(point2.pos_from(origin), frame, variables=True)
    # Get the two positions as substitution dicts for coordinate variables
    subs_dict1 = {}
    subs_dict2 = {}
    for i, x in enumerate(frame):
        subs_dict1[frame[i]] = x.dot(position1)
        subs_dict2[frame[i]] = x.dot(position2)
    return scalar_fn.subs(subs_dict2) - scalar_fn.subs(subs_dict1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\internal\field_mask.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Contains FieldMask class."""

from google.protobuf.descriptor import FieldDescriptor


class FieldMask(object):
  """Class for FieldMask message type."""

  __slots__ = ()

  def ToJsonString(self):
    """Converts FieldMask to string according to proto3 JSON spec."""
    camelcase_paths = []
    for path in self.paths:
      camelcase_paths.append(_SnakeCaseToCamelCase(path))
    return ','.join(camelcase_paths)

  def FromJsonString(self, value):
    """Converts string to FieldMask according to proto3 JSON spec."""
    if not isinstance(value, str):
      raise ValueError('FieldMask JSON value not a string: {!r}'.format(value))
    self.Clear()
    if value:
      for path in value.split(','):
        self.paths.append(_CamelCaseToSnakeCase(path))

  def IsValidForDescriptor(self, message_descriptor):
    """Checks whether the FieldMask is valid for Message Descriptor."""
    for path in self.paths:
      if not _IsValidPath(message_descriptor, path):
        return False
    return True

  def AllFieldsFromDescriptor(self, message_descriptor):
    """Gets all direct fields of Message Descriptor to FieldMask."""
    self.Clear()
    for field in message_descriptor.fields:
      self.paths.append(field.name)

  def CanonicalFormFromMask(self, mask):
    """Converts a FieldMask to the canonical form.

    Removes paths that are covered by another path. For example,
    "foo.bar" is covered by "foo" and will be removed if "foo"
    is also in the FieldMask. Then sorts all paths in alphabetical order.

    Args:
      mask: The original FieldMask to be converted.
    """
    tree = _FieldMaskTree(mask)
    tree.ToFieldMask(self)

  def Union(self, mask1, mask2):
    """Merges mask1 and mask2 into this FieldMask."""
    _CheckFieldMaskMessage(mask1)
    _CheckFieldMaskMessage(mask2)
    tree = _FieldMaskTree(mask1)
    tree.MergeFromFieldMask(mask2)
    tree.ToFieldMask(self)

  def Intersect(self, mask1, mask2):
    """Intersects mask1 and mask2 into this FieldMask."""
    _CheckFieldMaskMessage(mask1)
    _CheckFieldMaskMessage(mask2)
    tree = _FieldMaskTree(mask1)
    intersection = _FieldMaskTree()
    for path in mask2.paths:
      tree.IntersectPath(path, intersection)
    intersection.ToFieldMask(self)

  def MergeMessage(
      self, source, destination,
      replace_message_field=False, replace_repeated_field=False):
    """Merges fields specified in FieldMask from source to destination.

    Args:
      source: Source message.
      destination: The destination message to be merged into.
      replace_message_field: Replace message field if True. Merge message
          field if False.
      replace_repeated_field: Replace repeated field if True. Append
          elements of repeated field if False.
    """
    tree = _FieldMaskTree(self)
    tree.MergeMessage(
        source, destination, replace_message_field, replace_repeated_field)


def _IsValidPath(message_descriptor, path):
  """Checks whether the path is valid for Message Descriptor."""
  parts = path.split('.')
  last = parts.pop()
  for name in parts:
    field = message_descriptor.fields_by_name.get(name)
    if (field is None or
        field.label == FieldDescriptor.LABEL_REPEATED or
        field.type != FieldDescriptor.TYPE_MESSAGE):
      return False
    message_descriptor = field.message_type
  return last in message_descriptor.fields_by_name


def _CheckFieldMaskMessage(message):
  """Raises ValueError if message is not a FieldMask."""
  message_descriptor = message.DESCRIPTOR
  if (message_descriptor.name != 'FieldMask' or
      message_descriptor.file.name != 'google/protobuf/field_mask.proto'):
    raise ValueError('Message {0} is not a FieldMask.'.format(
        message_descriptor.full_name))


def _SnakeCaseToCamelCase(path_name):
  """Converts a path name from snake_case to camelCase."""
  result = []
  after_underscore = False
  for c in path_name:
    if c.isupper():
      raise ValueError(
          'Fail to print FieldMask to Json string: Path name '
          '{0} must not contain uppercase letters.'.format(path_name))
    if after_underscore:
      if c.islower():
        result.append(c.upper())
        after_underscore = False
      else:
        raise ValueError(
            'Fail to print FieldMask to Json string: The '
            'character after a "_" must be a lowercase letter '
            'in path name {0}.'.format(path_name))
    elif c == '_':
      after_underscore = True
    else:
      result += c

  if after_underscore:
    raise ValueError('Fail to print FieldMask to Json string: Trailing "_" '
                     'in path name {0}.'.format(path_name))
  return ''.join(result)


def _CamelCaseToSnakeCase(path_name):
  """Converts a field name from camelCase to snake_case."""
  result = []
  for c in path_name:
    if c == '_':
      raise ValueError('Fail to parse FieldMask: Path name '
                       '{0} must not contain "_"s.'.format(path_name))
    if c.isupper():
      result += '_'
      result += c.lower()
    else:
      result += c
  return ''.join(result)


class _FieldMaskTree(object):
  """Represents a FieldMask in a tree structure.

  For example, given a FieldMask "foo.bar,foo.baz,bar.baz",
  the FieldMaskTree will be:
      [_root] -+- foo -+- bar
            |       |
            |       +- baz
            |
            +- bar --- baz
  In the tree, each leaf node represents a field path.
  """

  __slots__ = ('_root',)

  def __init__(self, field_mask=None):
    """Initializes the tree by FieldMask."""
    self._root = {}
    if field_mask:
      self.MergeFromFieldMask(field_mask)

  def MergeFromFieldMask(self, field_mask):
    """Merges a FieldMask to the tree."""
    for path in field_mask.paths:
      self.AddPath(path)

  def AddPath(self, path):
    """Adds a field path into the tree.

    If the field path to add is a sub-path of an existing field path
    in the tree (i.e., a leaf node), it means the tree already matches
    the given path so nothing will be added to the tree. If the path
    matches an existing non-leaf node in the tree, that non-leaf node
    will be turned into a leaf node with all its children removed because
    the path matches all the node's children. Otherwise, a new path will
    be added.

    Args:
      path: The field path to add.
    """
    node = self._root
    for name in path.split('.'):
      if name not in node:
        node[name] = {}
      elif not node[name]:
        # Pre-existing empty node implies we already have this entire tree.
        return
      node = node[name]
    # Remove any sub-trees we might have had.
    node.clear()

  def ToFieldMask(self, field_mask):
    """Converts the tree to a FieldMask."""
    field_mask.Clear()
    _AddFieldPaths(self._root, '', field_mask)

  def IntersectPath(self, path, intersection):
    """Calculates the intersection part of a field path with this tree.

    Args:
      path: The field path to calculates.
      intersection: The out tree to record the intersection part.
    """
    node = self._root
    for name in path.split('.'):
      if name not in node:
        return
      elif not node[name]:
        intersection.AddPath(path)
        return
      node = node[name]
    intersection.AddLeafNodes(path, node)

  def AddLeafNodes(self, prefix, node):
    """Adds leaf nodes begin with prefix to this tree."""
    if not node:
      self.AddPath(prefix)
    for name in node:
      child_path = prefix + '.' + name
      self.AddLeafNodes(child_path, node[name])

  def MergeMessage(
      self, source, destination,
      replace_message, replace_repeated):
    """Merge all fields specified by this tree from source to destination."""
    _MergeMessage(
        self._root, source, destination, replace_message, replace_repeated)


def _StrConvert(value):
  """Converts value to str if it is not."""
  # This file is imported by c extension and some methods like ClearField
  # requires string for the field name. py2/py3 has different text
  # type and may use unicode.
  if not isinstance(value, str):
    return value.encode('utf-8')
  return value


def _MergeMessage(
    node, source, destination, replace_message, replace_repeated):
  """Merge all fields specified by a sub-tree from source to destination."""
  source_descriptor = source.DESCRIPTOR
  for name in node:
    child = node[name]
    field = source_descriptor.fields_by_name[name]
    if field is None:
      raise ValueError('Error: Can\'t find field {0} in message {1}.'.format(
          name, source_descriptor.full_name))
    if child:
      # Sub-paths are only allowed for singular message fields.
      if (field.label == FieldDescriptor.LABEL_REPEATED or
          field.cpp_type != FieldDescriptor.CPPTYPE_MESSAGE):
        raise ValueError('Error: Field {0} in message {1} is not a singular '
                         'message field and cannot have sub-fields.'.format(
                             name, source_descriptor.full_name))
      if source.HasField(name):
        _MergeMessage(
            child, getattr(source, name), getattr(destination, name),
            replace_message, replace_repeated)
      continue
    if field.label == FieldDescriptor.LABEL_REPEATED:
      if replace_repeated:
        destination.ClearField(_StrConvert(name))
      repeated_source = getattr(source, name)
      repeated_destination = getattr(destination, name)
      repeated_destination.MergeFrom(repeated_source)
    else:
      if field.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE:
        if replace_message:
          destination.ClearField(_StrConvert(name))
        if source.HasField(name):
          getattr(destination, name).MergeFrom(getattr(source, name))
      elif not field.has_presence or source.HasField(name):
        setattr(destination, name, getattr(source, name))
      else:
        destination.ClearField(_StrConvert(name))


def _AddFieldPaths(node, prefix, field_mask):
  """Adds the field paths descended from node to field_mask."""
  if not node and prefix:
    field_mask.paths.append(prefix)
    return
  for name in sorted(node):
    if prefix:
      child_path = prefix + '.' + name
    else:
      child_path = name
    _AddFieldPaths(node[name], child_path, field_mask)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\align.py ===
import sys
from itertools import chain
from typing import TYPE_CHECKING, Iterable, Optional

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from pip._vendor.typing_extensions import Literal  # pragma: no cover

from .constrain import Constrain
from .jupyter import JupyterMixin
from .measure import Measurement
from .segment import Segment
from .style import StyleType

if TYPE_CHECKING:
    from .console import Console, ConsoleOptions, RenderableType, RenderResult

AlignMethod = Literal["left", "center", "right"]
VerticalAlignMethod = Literal["top", "middle", "bottom"]


class Align(JupyterMixin):
    """Align a renderable by adding spaces if necessary.

    Args:
        renderable (RenderableType): A console renderable.
        align (AlignMethod): One of "left", "center", or "right""
        style (StyleType, optional): An optional style to apply to the background.
        vertical (Optional[VerticalAlignMethod], optional): Optional vertical align, one of "top", "middle", or "bottom". Defaults to None.
        pad (bool, optional): Pad the right with spaces. Defaults to True.
        width (int, optional): Restrict contents to given width, or None to use default width. Defaults to None.
        height (int, optional): Set height of align renderable, or None to fit to contents. Defaults to None.

    Raises:
        ValueError: if ``align`` is not one of the expected values.
    """

    def __init__(
        self,
        renderable: "RenderableType",
        align: AlignMethod = "left",
        style: Optional[StyleType] = None,
        *,
        vertical: Optional[VerticalAlignMethod] = None,
        pad: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        if align not in ("left", "center", "right"):
            raise ValueError(
                f'invalid value for align, expected "left", "center", or "right" (not {align!r})'
            )
        if vertical is not None and vertical not in ("top", "middle", "bottom"):
            raise ValueError(
                f'invalid value for vertical, expected "top", "middle", or "bottom" (not {vertical!r})'
            )
        self.renderable = renderable
        self.align = align
        self.style = style
        self.vertical = vertical
        self.pad = pad
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return f"Align({self.renderable!r}, {self.align!r})"

    @classmethod
    def left(
        cls,
        renderable: "RenderableType",
        style: Optional[StyleType] = None,
        *,
        vertical: Optional[VerticalAlignMethod] = None,
        pad: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> "Align":
        """Align a renderable to the left."""
        return cls(
            renderable,
            "left",
            style=style,
            vertical=vertical,
            pad=pad,
            width=width,
            height=height,
        )

    @classmethod
    def center(
        cls,
        renderable: "RenderableType",
        style: Optional[StyleType] = None,
        *,
        vertical: Optional[VerticalAlignMethod] = None,
        pad: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> "Align":
        """Align a renderable to the center."""
        return cls(
            renderable,
            "center",
            style=style,
            vertical=vertical,
            pad=pad,
            width=width,
            height=height,
        )

    @classmethod
    def right(
        cls,
        renderable: "RenderableType",
        style: Optional[StyleType] = None,
        *,
        vertical: Optional[VerticalAlignMethod] = None,
        pad: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> "Align":
        """Align a renderable to the right."""
        return cls(
            renderable,
            "right",
            style=style,
            vertical=vertical,
            pad=pad,
            width=width,
            height=height,
        )

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        align = self.align
        width = console.measure(self.renderable, options=options).maximum
        rendered = console.render(
            Constrain(
                self.renderable, width if self.width is None else min(width, self.width)
            ),
            options.update(height=None),
        )
        lines = list(Segment.split_lines(rendered))
        width, height = Segment.get_shape(lines)
        lines = Segment.set_shape(lines, width, height)
        new_line = Segment.line()
        excess_space = options.max_width - width
        style = console.get_style(self.style) if self.style is not None else None

        def generate_segments() -> Iterable[Segment]:
            if excess_space <= 0:
                # Exact fit
                for line in lines:
                    yield from line
                    yield new_line

            elif align == "left":
                # Pad on the right
                pad = Segment(" " * excess_space, style) if self.pad else None
                for line in lines:
                    yield from line
                    if pad:
                        yield pad
                    yield new_line

            elif align == "center":
                # Pad left and right
                left = excess_space // 2
                pad = Segment(" " * left, style)
                pad_right = (
                    Segment(" " * (excess_space - left), style) if self.pad else None
                )
                for line in lines:
                    if left:
                        yield pad
                    yield from line
                    if pad_right:
                        yield pad_right
                    yield new_line

            elif align == "right":
                # Padding on left
                pad = Segment(" " * excess_space, style)
                for line in lines:
                    yield pad
                    yield from line
                    yield new_line

        blank_line = (
            Segment(f"{' ' * (self.width or options.max_width)}\n", style)
            if self.pad
            else Segment("\n")
        )

        def blank_lines(count: int) -> Iterable[Segment]:
            if count > 0:
                for _ in range(count):
                    yield blank_line

        vertical_height = self.height or options.height
        iter_segments: Iterable[Segment]
        if self.vertical and vertical_height is not None:
            if self.vertical == "top":
                bottom_space = vertical_height - height
                iter_segments = chain(generate_segments(), blank_lines(bottom_space))
            elif self.vertical == "middle":
                top_space = (vertical_height - height) // 2
                bottom_space = vertical_height - top_space - height
                iter_segments = chain(
                    blank_lines(top_space),
                    generate_segments(),
                    blank_lines(bottom_space),
                )
            else:  #  self.vertical == "bottom":
                top_space = vertical_height - height
                iter_segments = chain(blank_lines(top_space), generate_segments())
        else:
            iter_segments = generate_segments()
        if self.style:
            style = console.get_style(self.style)
            iter_segments = Segment.apply_style(iter_segments, style)
        yield from iter_segments

    def __rich_measure__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> Measurement:
        measurement = Measurement.get(console, options, self.renderable)
        return measurement


class VerticalCenter(JupyterMixin):
    """Vertically aligns a renderable.

    Warn:
        This class is deprecated and may be removed in a future version. Use Align class with
        `vertical="middle"`.

    Args:
        renderable (RenderableType): A renderable object.
        style (StyleType, optional): An optional style to apply to the background. Defaults to None.
    """

    def __init__(
        self,
        renderable: "RenderableType",
        style: Optional[StyleType] = None,
    ) -> None:
        self.renderable = renderable
        self.style = style

    def __repr__(self) -> str:
        return f"VerticalCenter({self.renderable!r})"

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        style = console.get_style(self.style) if self.style is not None else None
        lines = console.render_lines(
            self.renderable, options.update(height=None), pad=False
        )
        width, _height = Segment.get_shape(lines)
        new_line = Segment.line()
        height = options.height or options.size.height
        top_space = (height - len(lines)) // 2
        bottom_space = height - top_space - len(lines)
        blank_line = Segment(f"{' ' * width}", style)

        def blank_lines(count: int) -> Iterable[Segment]:
            for _ in range(count):
                yield blank_line
                yield new_line

        if top_space > 0:
            yield from blank_lines(top_space)
        for line in lines:
            yield from line
            yield new_line
        if bottom_space > 0:
            yield from blank_lines(bottom_space)

    def __rich_measure__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> Measurement:
        measurement = Measurement.get(console, options, self.renderable)
        return measurement


if __name__ == "__main__":  # pragma: no cover
    from pip._vendor.rich.console import Console, Group
    from pip._vendor.rich.highlighter import ReprHighlighter
    from pip._vendor.rich.panel import Panel

    highlighter = ReprHighlighter()
    console = Console()

    panel = Panel(
        Group(
            Align.left(highlighter("align='left'")),
            Align.center(highlighter("align='center'")),
            Align.right(highlighter("align='right'")),
        ),
        width=60,
        style="on dark_blue",
        title="Align",
    )

    console.print(
        Align.center(panel, vertical="middle", style="on red", height=console.height)
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\dom_debugger.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: DOMDebugger
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime


class DOMBreakpointType(enum.Enum):
    '''
    DOM breakpoint type.
    '''
    SUBTREE_MODIFIED = "subtree-modified"
    ATTRIBUTE_MODIFIED = "attribute-modified"
    NODE_REMOVED = "node-removed"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CSPViolationType(enum.Enum):
    '''
    CSP Violation type.
    '''
    TRUSTEDTYPE_SINK_VIOLATION = "trustedtype-sink-violation"
    TRUSTEDTYPE_POLICY_VIOLATION = "trustedtype-policy-violation"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class EventListener:
    '''
    Object event listener.
    '''
    #: ``EventListener``'s type.
    type_: str

    #: ``EventListener``'s useCapture.
    use_capture: bool

    #: ``EventListener``'s passive flag.
    passive: bool

    #: ``EventListener``'s once flag.
    once: bool

    #: Script id of the handler code.
    script_id: runtime.ScriptId

    #: Line number in the script (0-based).
    line_number: int

    #: Column number in the script (0-based).
    column_number: int

    #: Event handler function value.
    handler: typing.Optional[runtime.RemoteObject] = None

    #: Event original handler function value.
    original_handler: typing.Optional[runtime.RemoteObject] = None

    #: Node the listener is added to (if any).
    backend_node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        json['useCapture'] = self.use_capture
        json['passive'] = self.passive
        json['once'] = self.once
        json['scriptId'] = self.script_id.to_json()
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        if self.handler is not None:
            json['handler'] = self.handler.to_json()
        if self.original_handler is not None:
            json['originalHandler'] = self.original_handler.to_json()
        if self.backend_node_id is not None:
            json['backendNodeId'] = self.backend_node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            use_capture=bool(json['useCapture']),
            passive=bool(json['passive']),
            once=bool(json['once']),
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']),
            handler=runtime.RemoteObject.from_json(json['handler']) if 'handler' in json else None,
            original_handler=runtime.RemoteObject.from_json(json['originalHandler']) if 'originalHandler' in json else None,
            backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId']) if 'backendNodeId' in json else None,
        )


def get_event_listeners(
        object_id: runtime.RemoteObjectId,
        depth: typing.Optional[int] = None,
        pierce: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[EventListener]]:
    '''
    Returns event listeners of the given object.

    :param object_id: Identifier of the object to return listeners for.
    :param depth: *(Optional)* The maximum depth at which Node children should be retrieved, defaults to 1. Use -1 for the entire subtree or provide an integer larger than 0.
    :param pierce: *(Optional)* Whether or not iframes and shadow roots should be traversed when returning the subtree (default is false). Reports listeners for all contexts if pierce is enabled.
    :returns: Array of relevant listeners.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    if depth is not None:
        params['depth'] = depth
    if pierce is not None:
        params['pierce'] = pierce
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.getEventListeners',
        'params': params,
    }
    json = yield cmd_dict
    return [EventListener.from_json(i) for i in json['listeners']]


def remove_dom_breakpoint(
        node_id: dom.NodeId,
        type_: DOMBreakpointType
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes DOM breakpoint that was set using ``setDOMBreakpoint``.

    :param node_id: Identifier of the node to remove breakpoint from.
    :param type_: Type of the breakpoint to remove.
    '''
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    params['type'] = type_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.removeDOMBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def remove_event_listener_breakpoint(
        event_name: str,
        target_name: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes breakpoint on particular DOM event.

    :param event_name: Event name.
    :param target_name: **(EXPERIMENTAL)** *(Optional)* EventTarget interface name.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    if target_name is not None:
        params['targetName'] = target_name
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.removeEventListenerBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def remove_instrumentation_breakpoint(
        event_name: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes breakpoint on particular native event.

    **EXPERIMENTAL**

    :param event_name: Instrumentation name to stop on.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.removeInstrumentationBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def remove_xhr_breakpoint(
        url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes breakpoint from XMLHttpRequest.

    :param url: Resource URL substring.
    '''
    params: T_JSON_DICT = dict()
    params['url'] = url
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.removeXHRBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def set_break_on_csp_violation(
        violation_types: typing.List[CSPViolationType]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on particular CSP violations.

    **EXPERIMENTAL**

    :param violation_types: CSP Violations to stop upon.
    '''
    params: T_JSON_DICT = dict()
    params['violationTypes'] = [i.to_json() for i in violation_types]
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.setBreakOnCSPViolation',
        'params': params,
    }
    json = yield cmd_dict


def set_dom_breakpoint(
        node_id: dom.NodeId,
        type_: DOMBreakpointType
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on particular operation with DOM.

    :param node_id: Identifier of the node to set breakpoint on.
    :param type_: Type of the operation to stop upon.
    '''
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    params['type'] = type_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.setDOMBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def set_event_listener_breakpoint(
        event_name: str,
        target_name: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on particular DOM event.

    :param event_name: DOM Event name to stop on (any DOM event will do).
    :param target_name: **(EXPERIMENTAL)** *(Optional)* EventTarget interface name to stop on. If equal to ```"*"``` or not provided, will stop on any EventTarget.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    if target_name is not None:
        params['targetName'] = target_name
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.setEventListenerBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def set_instrumentation_breakpoint(
        event_name: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on particular native event.

    **EXPERIMENTAL**

    :param event_name: Instrumentation name to stop on.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.setInstrumentationBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def set_xhr_breakpoint(
        url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on XMLHttpRequest.

    :param url: Resource URL substring. All XHRs having this substring in the URL will get stopped upon.
    '''
    params: T_JSON_DICT = dict()
    params['url'] = url
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.setXHRBreakpoint',
        'params': params,
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\dom_debugger.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: DOMDebugger
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime


class DOMBreakpointType(enum.Enum):
    '''
    DOM breakpoint type.
    '''
    SUBTREE_MODIFIED = "subtree-modified"
    ATTRIBUTE_MODIFIED = "attribute-modified"
    NODE_REMOVED = "node-removed"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CSPViolationType(enum.Enum):
    '''
    CSP Violation type.
    '''
    TRUSTEDTYPE_SINK_VIOLATION = "trustedtype-sink-violation"
    TRUSTEDTYPE_POLICY_VIOLATION = "trustedtype-policy-violation"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class EventListener:
    '''
    Object event listener.
    '''
    #: ``EventListener``'s type.
    type_: str

    #: ``EventListener``'s useCapture.
    use_capture: bool

    #: ``EventListener``'s passive flag.
    passive: bool

    #: ``EventListener``'s once flag.
    once: bool

    #: Script id of the handler code.
    script_id: runtime.ScriptId

    #: Line number in the script (0-based).
    line_number: int

    #: Column number in the script (0-based).
    column_number: int

    #: Event handler function value.
    handler: typing.Optional[runtime.RemoteObject] = None

    #: Event original handler function value.
    original_handler: typing.Optional[runtime.RemoteObject] = None

    #: Node the listener is added to (if any).
    backend_node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        json['useCapture'] = self.use_capture
        json['passive'] = self.passive
        json['once'] = self.once
        json['scriptId'] = self.script_id.to_json()
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        if self.handler is not None:
            json['handler'] = self.handler.to_json()
        if self.original_handler is not None:
            json['originalHandler'] = self.original_handler.to_json()
        if self.backend_node_id is not None:
            json['backendNodeId'] = self.backend_node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            use_capture=bool(json['useCapture']),
            passive=bool(json['passive']),
            once=bool(json['once']),
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']),
            handler=runtime.RemoteObject.from_json(json['handler']) if 'handler' in json else None,
            original_handler=runtime.RemoteObject.from_json(json['originalHandler']) if 'originalHandler' in json else None,
            backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId']) if 'backendNodeId' in json else None,
        )


def get_event_listeners(
        object_id: runtime.RemoteObjectId,
        depth: typing.Optional[int] = None,
        pierce: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[EventListener]]:
    '''
    Returns event listeners of the given object.

    :param object_id: Identifier of the object to return listeners for.
    :param depth: *(Optional)* The maximum depth at which Node children should be retrieved, defaults to 1. Use -1 for the entire subtree or provide an integer larger than 0.
    :param pierce: *(Optional)* Whether or not iframes and shadow roots should be traversed when returning the subtree (default is false). Reports listeners for all contexts if pierce is enabled.
    :returns: Array of relevant listeners.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    if depth is not None:
        params['depth'] = depth
    if pierce is not None:
        params['pierce'] = pierce
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.getEventListeners',
        'params': params,
    }
    json = yield cmd_dict
    return [EventListener.from_json(i) for i in json['listeners']]


def remove_dom_breakpoint(
        node_id: dom.NodeId,
        type_: DOMBreakpointType
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes DOM breakpoint that was set using ``setDOMBreakpoint``.

    :param node_id: Identifier of the node to remove breakpoint from.
    :param type_: Type of the breakpoint to remove.
    '''
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    params['type'] = type_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.removeDOMBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def remove_event_listener_breakpoint(
        event_name: str,
        target_name: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes breakpoint on particular DOM event.

    :param event_name: Event name.
    :param target_name: **(EXPERIMENTAL)** *(Optional)* EventTarget interface name.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    if target_name is not None:
        params['targetName'] = target_name
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.removeEventListenerBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def remove_instrumentation_breakpoint(
        event_name: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes breakpoint on particular native event.

    **EXPERIMENTAL**

    :param event_name: Instrumentation name to stop on.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.removeInstrumentationBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def remove_xhr_breakpoint(
        url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes breakpoint from XMLHttpRequest.

    :param url: Resource URL substring.
    '''
    params: T_JSON_DICT = dict()
    params['url'] = url
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.removeXHRBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def set_break_on_csp_violation(
        violation_types: typing.List[CSPViolationType]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on particular CSP violations.

    **EXPERIMENTAL**

    :param violation_types: CSP Violations to stop upon.
    '''
    params: T_JSON_DICT = dict()
    params['violationTypes'] = [i.to_json() for i in violation_types]
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.setBreakOnCSPViolation',
        'params': params,
    }
    json = yield cmd_dict


def set_dom_breakpoint(
        node_id: dom.NodeId,
        type_: DOMBreakpointType
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on particular operation with DOM.

    :param node_id: Identifier of the node to set breakpoint on.
    :param type_: Type of the operation to stop upon.
    '''
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    params['type'] = type_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.setDOMBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def set_event_listener_breakpoint(
        event_name: str,
        target_name: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on particular DOM event.

    :param event_name: DOM Event name to stop on (any DOM event will do).
    :param target_name: **(EXPERIMENTAL)** *(Optional)* EventTarget interface name to stop on. If equal to ```"*"``` or not provided, will stop on any EventTarget.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    if target_name is not None:
        params['targetName'] = target_name
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.setEventListenerBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def set_instrumentation_breakpoint(
        event_name: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on particular native event.

    **EXPERIMENTAL**

    :param event_name: Instrumentation name to stop on.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.setInstrumentationBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def set_xhr_breakpoint(
        url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on XMLHttpRequest.

    :param url: Resource URL substring. All XHRs having this substring in the URL will get stopped upon.
    '''
    params: T_JSON_DICT = dict()
    params['url'] = url
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.setXHRBreakpoint',
        'params': params,
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\dom_debugger.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: DOMDebugger
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime


class DOMBreakpointType(enum.Enum):
    '''
    DOM breakpoint type.
    '''
    SUBTREE_MODIFIED = "subtree-modified"
    ATTRIBUTE_MODIFIED = "attribute-modified"
    NODE_REMOVED = "node-removed"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CSPViolationType(enum.Enum):
    '''
    CSP Violation type.
    '''
    TRUSTEDTYPE_SINK_VIOLATION = "trustedtype-sink-violation"
    TRUSTEDTYPE_POLICY_VIOLATION = "trustedtype-policy-violation"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class EventListener:
    '''
    Object event listener.
    '''
    #: ``EventListener``'s type.
    type_: str

    #: ``EventListener``'s useCapture.
    use_capture: bool

    #: ``EventListener``'s passive flag.
    passive: bool

    #: ``EventListener``'s once flag.
    once: bool

    #: Script id of the handler code.
    script_id: runtime.ScriptId

    #: Line number in the script (0-based).
    line_number: int

    #: Column number in the script (0-based).
    column_number: int

    #: Event handler function value.
    handler: typing.Optional[runtime.RemoteObject] = None

    #: Event original handler function value.
    original_handler: typing.Optional[runtime.RemoteObject] = None

    #: Node the listener is added to (if any).
    backend_node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        json['useCapture'] = self.use_capture
        json['passive'] = self.passive
        json['once'] = self.once
        json['scriptId'] = self.script_id.to_json()
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        if self.handler is not None:
            json['handler'] = self.handler.to_json()
        if self.original_handler is not None:
            json['originalHandler'] = self.original_handler.to_json()
        if self.backend_node_id is not None:
            json['backendNodeId'] = self.backend_node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            use_capture=bool(json['useCapture']),
            passive=bool(json['passive']),
            once=bool(json['once']),
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']),
            handler=runtime.RemoteObject.from_json(json['handler']) if 'handler' in json else None,
            original_handler=runtime.RemoteObject.from_json(json['originalHandler']) if 'originalHandler' in json else None,
            backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId']) if 'backendNodeId' in json else None,
        )


def get_event_listeners(
        object_id: runtime.RemoteObjectId,
        depth: typing.Optional[int] = None,
        pierce: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[EventListener]]:
    '''
    Returns event listeners of the given object.

    :param object_id: Identifier of the object to return listeners for.
    :param depth: *(Optional)* The maximum depth at which Node children should be retrieved, defaults to 1. Use -1 for the entire subtree or provide an integer larger than 0.
    :param pierce: *(Optional)* Whether or not iframes and shadow roots should be traversed when returning the subtree (default is false). Reports listeners for all contexts if pierce is enabled.
    :returns: Array of relevant listeners.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    if depth is not None:
        params['depth'] = depth
    if pierce is not None:
        params['pierce'] = pierce
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.getEventListeners',
        'params': params,
    }
    json = yield cmd_dict
    return [EventListener.from_json(i) for i in json['listeners']]


def remove_dom_breakpoint(
        node_id: dom.NodeId,
        type_: DOMBreakpointType
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes DOM breakpoint that was set using ``setDOMBreakpoint``.

    :param node_id: Identifier of the node to remove breakpoint from.
    :param type_: Type of the breakpoint to remove.
    '''
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    params['type'] = type_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.removeDOMBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def remove_event_listener_breakpoint(
        event_name: str,
        target_name: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes breakpoint on particular DOM event.

    :param event_name: Event name.
    :param target_name: **(EXPERIMENTAL)** *(Optional)* EventTarget interface name.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    if target_name is not None:
        params['targetName'] = target_name
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.removeEventListenerBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def remove_instrumentation_breakpoint(
        event_name: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes breakpoint on particular native event.

    **EXPERIMENTAL**

    :param event_name: Instrumentation name to stop on.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.removeInstrumentationBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def remove_xhr_breakpoint(
        url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes breakpoint from XMLHttpRequest.

    :param url: Resource URL substring.
    '''
    params: T_JSON_DICT = dict()
    params['url'] = url
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.removeXHRBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def set_break_on_csp_violation(
        violation_types: typing.List[CSPViolationType]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on particular CSP violations.

    **EXPERIMENTAL**

    :param violation_types: CSP Violations to stop upon.
    '''
    params: T_JSON_DICT = dict()
    params['violationTypes'] = [i.to_json() for i in violation_types]
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.setBreakOnCSPViolation',
        'params': params,
    }
    json = yield cmd_dict


def set_dom_breakpoint(
        node_id: dom.NodeId,
        type_: DOMBreakpointType
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on particular operation with DOM.

    :param node_id: Identifier of the node to set breakpoint on.
    :param type_: Type of the operation to stop upon.
    '''
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    params['type'] = type_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.setDOMBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def set_event_listener_breakpoint(
        event_name: str,
        target_name: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on particular DOM event.

    :param event_name: DOM Event name to stop on (any DOM event will do).
    :param target_name: **(EXPERIMENTAL)** *(Optional)* EventTarget interface name to stop on. If equal to ```"*"``` or not provided, will stop on any EventTarget.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    if target_name is not None:
        params['targetName'] = target_name
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.setEventListenerBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def set_instrumentation_breakpoint(
        event_name: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on particular native event.

    **EXPERIMENTAL**

    :param event_name: Instrumentation name to stop on.
    '''
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.setInstrumentationBreakpoint',
        'params': params,
    }
    json = yield cmd_dict


def set_xhr_breakpoint(
        url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets breakpoint on XMLHttpRequest.

    :param url: Resource URL substring. All XHRs having this substring in the URL will get stopped upon.
    '''
    params: T_JSON_DICT = dict()
    params['url'] = url
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMDebugger.setXHRBreakpoint',
        'params': params,
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\postgresql\pg_catalog.py ===
# dialects/postgresql/pg_catalog.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Sequence
from typing import TYPE_CHECKING

from .array import ARRAY
from .types import OID
from .types import REGCLASS
from ... import Column
from ... import func
from ... import MetaData
from ... import Table
from ...types import BigInteger
from ...types import Boolean
from ...types import CHAR
from ...types import Float
from ...types import Integer
from ...types import SmallInteger
from ...types import String
from ...types import Text
from ...types import TypeDecorator

if TYPE_CHECKING:
    from ...engine.interfaces import Dialect
    from ...sql.type_api import _ResultProcessorType


# types
class NAME(TypeDecorator[str]):
    impl = String(64, collation="C")
    cache_ok = True


class PG_NODE_TREE(TypeDecorator[str]):
    impl = Text(collation="C")
    cache_ok = True


class INT2VECTOR(TypeDecorator[Sequence[int]]):
    impl = ARRAY(SmallInteger)
    cache_ok = True


class OIDVECTOR(TypeDecorator[Sequence[int]]):
    impl = ARRAY(OID)
    cache_ok = True


class _SpaceVector:
    def result_processor(
        self, dialect: Dialect, coltype: object
    ) -> _ResultProcessorType[list[int]]:
        def process(value: Any) -> Optional[list[int]]:
            if value is None:
                return value
            return [int(p) for p in value.split(" ")]

        return process


REGPROC = REGCLASS  # seems an alias

# functions
_pg_cat = func.pg_catalog
quote_ident = _pg_cat.quote_ident
pg_table_is_visible = _pg_cat.pg_table_is_visible
pg_type_is_visible = _pg_cat.pg_type_is_visible
pg_get_viewdef = _pg_cat.pg_get_viewdef
pg_get_serial_sequence = _pg_cat.pg_get_serial_sequence
format_type = _pg_cat.format_type
pg_get_expr = _pg_cat.pg_get_expr
pg_get_constraintdef = _pg_cat.pg_get_constraintdef
pg_get_indexdef = _pg_cat.pg_get_indexdef

# constants
RELKINDS_TABLE_NO_FOREIGN = ("r", "p")
RELKINDS_TABLE = RELKINDS_TABLE_NO_FOREIGN + ("f",)
RELKINDS_VIEW = ("v",)
RELKINDS_MAT_VIEW = ("m",)
RELKINDS_ALL_TABLE_LIKE = RELKINDS_TABLE + RELKINDS_VIEW + RELKINDS_MAT_VIEW

# tables
pg_catalog_meta = MetaData(schema="pg_catalog")

pg_namespace = Table(
    "pg_namespace",
    pg_catalog_meta,
    Column("oid", OID),
    Column("nspname", NAME),
    Column("nspowner", OID),
)

pg_class = Table(
    "pg_class",
    pg_catalog_meta,
    Column("oid", OID, info={"server_version": (9, 3)}),
    Column("relname", NAME),
    Column("relnamespace", OID),
    Column("reltype", OID),
    Column("reloftype", OID),
    Column("relowner", OID),
    Column("relam", OID),
    Column("relfilenode", OID),
    Column("reltablespace", OID),
    Column("relpages", Integer),
    Column("reltuples", Float),
    Column("relallvisible", Integer, info={"server_version": (9, 2)}),
    Column("reltoastrelid", OID),
    Column("relhasindex", Boolean),
    Column("relisshared", Boolean),
    Column("relpersistence", CHAR, info={"server_version": (9, 1)}),
    Column("relkind", CHAR),
    Column("relnatts", SmallInteger),
    Column("relchecks", SmallInteger),
    Column("relhasrules", Boolean),
    Column("relhastriggers", Boolean),
    Column("relhassubclass", Boolean),
    Column("relrowsecurity", Boolean),
    Column("relforcerowsecurity", Boolean, info={"server_version": (9, 5)}),
    Column("relispopulated", Boolean, info={"server_version": (9, 3)}),
    Column("relreplident", CHAR, info={"server_version": (9, 4)}),
    Column("relispartition", Boolean, info={"server_version": (10,)}),
    Column("relrewrite", OID, info={"server_version": (11,)}),
    Column("reloptions", ARRAY(Text)),
)

pg_type = Table(
    "pg_type",
    pg_catalog_meta,
    Column("oid", OID, info={"server_version": (9, 3)}),
    Column("typname", NAME),
    Column("typnamespace", OID),
    Column("typowner", OID),
    Column("typlen", SmallInteger),
    Column("typbyval", Boolean),
    Column("typtype", CHAR),
    Column("typcategory", CHAR),
    Column("typispreferred", Boolean),
    Column("typisdefined", Boolean),
    Column("typdelim", CHAR),
    Column("typrelid", OID),
    Column("typelem", OID),
    Column("typarray", OID),
    Column("typinput", REGPROC),
    Column("typoutput", REGPROC),
    Column("typreceive", REGPROC),
    Column("typsend", REGPROC),
    Column("typmodin", REGPROC),
    Column("typmodout", REGPROC),
    Column("typanalyze", REGPROC),
    Column("typalign", CHAR),
    Column("typstorage", CHAR),
    Column("typnotnull", Boolean),
    Column("typbasetype", OID),
    Column("typtypmod", Integer),
    Column("typndims", Integer),
    Column("typcollation", OID, info={"server_version": (9, 1)}),
    Column("typdefault", Text),
)

pg_index = Table(
    "pg_index",
    pg_catalog_meta,
    Column("indexrelid", OID),
    Column("indrelid", OID),
    Column("indnatts", SmallInteger),
    Column("indnkeyatts", SmallInteger, info={"server_version": (11,)}),
    Column("indisunique", Boolean),
    Column("indnullsnotdistinct", Boolean, info={"server_version": (15,)}),
    Column("indisprimary", Boolean),
    Column("indisexclusion", Boolean, info={"server_version": (9, 1)}),
    Column("indimmediate", Boolean),
    Column("indisclustered", Boolean),
    Column("indisvalid", Boolean),
    Column("indcheckxmin", Boolean),
    Column("indisready", Boolean),
    Column("indislive", Boolean, info={"server_version": (9, 3)}),  # 9.3
    Column("indisreplident", Boolean),
    Column("indkey", INT2VECTOR),
    Column("indcollation", OIDVECTOR, info={"server_version": (9, 1)}),  # 9.1
    Column("indclass", OIDVECTOR),
    Column("indoption", INT2VECTOR),
    Column("indexprs", PG_NODE_TREE),
    Column("indpred", PG_NODE_TREE),
)

pg_attribute = Table(
    "pg_attribute",
    pg_catalog_meta,
    Column("attrelid", OID),
    Column("attname", NAME),
    Column("atttypid", OID),
    Column("attstattarget", Integer),
    Column("attlen", SmallInteger),
    Column("attnum", SmallInteger),
    Column("attndims", Integer),
    Column("attcacheoff", Integer),
    Column("atttypmod", Integer),
    Column("attbyval", Boolean),
    Column("attstorage", CHAR),
    Column("attalign", CHAR),
    Column("attnotnull", Boolean),
    Column("atthasdef", Boolean),
    Column("atthasmissing", Boolean, info={"server_version": (11,)}),
    Column("attidentity", CHAR, info={"server_version": (10,)}),
    Column("attgenerated", CHAR, info={"server_version": (12,)}),
    Column("attisdropped", Boolean),
    Column("attislocal", Boolean),
    Column("attinhcount", Integer),
    Column("attcollation", OID, info={"server_version": (9, 1)}),
)

pg_constraint = Table(
    "pg_constraint",
    pg_catalog_meta,
    Column("oid", OID),  # 9.3
    Column("conname", NAME),
    Column("connamespace", OID),
    Column("contype", CHAR),
    Column("condeferrable", Boolean),
    Column("condeferred", Boolean),
    Column("convalidated", Boolean, info={"server_version": (9, 1)}),
    Column("conrelid", OID),
    Column("contypid", OID),
    Column("conindid", OID),
    Column("conparentid", OID, info={"server_version": (11,)}),
    Column("confrelid", OID),
    Column("confupdtype", CHAR),
    Column("confdeltype", CHAR),
    Column("confmatchtype", CHAR),
    Column("conislocal", Boolean),
    Column("coninhcount", Integer),
    Column("connoinherit", Boolean, info={"server_version": (9, 2)}),
    Column("conkey", ARRAY(SmallInteger)),
    Column("confkey", ARRAY(SmallInteger)),
)

pg_sequence = Table(
    "pg_sequence",
    pg_catalog_meta,
    Column("seqrelid", OID),
    Column("seqtypid", OID),
    Column("seqstart", BigInteger),
    Column("seqincrement", BigInteger),
    Column("seqmax", BigInteger),
    Column("seqmin", BigInteger),
    Column("seqcache", BigInteger),
    Column("seqcycle", Boolean),
    info={"server_version": (10,)},
)

pg_attrdef = Table(
    "pg_attrdef",
    pg_catalog_meta,
    Column("oid", OID, info={"server_version": (9, 3)}),
    Column("adrelid", OID),
    Column("adnum", SmallInteger),
    Column("adbin", PG_NODE_TREE),
)

pg_description = Table(
    "pg_description",
    pg_catalog_meta,
    Column("objoid", OID),
    Column("classoid", OID),
    Column("objsubid", Integer),
    Column("description", Text(collation="C")),
)

pg_enum = Table(
    "pg_enum",
    pg_catalog_meta,
    Column("oid", OID, info={"server_version": (9, 3)}),
    Column("enumtypid", OID),
    Column("enumsortorder", Float(), info={"server_version": (9, 1)}),
    Column("enumlabel", NAME),
)

pg_am = Table(
    "pg_am",
    pg_catalog_meta,
    Column("oid", OID, info={"server_version": (9, 3)}),
    Column("amname", NAME),
    Column("amhandler", REGPROC, info={"server_version": (9, 6)}),
    Column("amtype", CHAR, info={"server_version": (9, 6)}),
)

pg_collation = Table(
    "pg_collation",
    pg_catalog_meta,
    Column("oid", OID, info={"server_version": (9, 3)}),
    Column("collname", NAME),
    Column("collnamespace", OID),
    Column("collowner", OID),
    Column("collprovider", CHAR, info={"server_version": (10,)}),
    Column("collisdeterministic", Boolean, info={"server_version": (12,)}),
    Column("collencoding", Integer),
    Column("collcollate", Text),
    Column("collctype", Text),
    Column("colliculocale", Text),
    Column("collicurules", Text, info={"server_version": (16,)}),
    Column("collversion", Text, info={"server_version": (10,)}),
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\sorting.py ===
from collections import defaultdict

from .sympify import sympify, SympifyError
from sympy.utilities.iterables import iterable, uniq


__all__ = ['default_sort_key', 'ordered']


def default_sort_key(item, order=None):
    """Return a key that can be used for sorting.

    The key has the structure:

    (class_key, (len(args), args), exponent.sort_key(), coefficient)

    This key is supplied by the sort_key routine of Basic objects when
    ``item`` is a Basic object or an object (other than a string) that
    sympifies to a Basic object. Otherwise, this function produces the
    key.

    The ``order`` argument is passed along to the sort_key routine and is
    used to determine how the terms *within* an expression are ordered.
    (See examples below) ``order`` options are: 'lex', 'grlex', 'grevlex',
    and reversed values of the same (e.g. 'rev-lex'). The default order
    value is None (which translates to 'lex').

    Examples
    ========

    >>> from sympy import S, I, default_sort_key, sin, cos, sqrt
    >>> from sympy.core.function import UndefinedFunction
    >>> from sympy.abc import x

    The following are equivalent ways of getting the key for an object:

    >>> x.sort_key() == default_sort_key(x)
    True

    Here are some examples of the key that is produced:

    >>> default_sort_key(UndefinedFunction('f'))
    ((0, 0, 'UndefinedFunction'), (1, ('f',)), ((1, 0, 'Number'),
        (0, ()), (), 1), 1)
    >>> default_sort_key('1')
    ((0, 0, 'str'), (1, ('1',)), ((1, 0, 'Number'), (0, ()), (), 1), 1)
    >>> default_sort_key(S.One)
    ((1, 0, 'Number'), (0, ()), (), 1)
    >>> default_sort_key(2)
    ((1, 0, 'Number'), (0, ()), (), 2)

    While sort_key is a method only defined for SymPy objects,
    default_sort_key will accept anything as an argument so it is
    more robust as a sorting key. For the following, using key=
    lambda i: i.sort_key() would fail because 2 does not have a sort_key
    method; that's why default_sort_key is used. Note, that it also
    handles sympification of non-string items likes ints:

    >>> a = [2, I, -I]
    >>> sorted(a, key=default_sort_key)
    [2, -I, I]

    The returned key can be used anywhere that a key can be specified for
    a function, e.g. sort, min, max, etc...:

    >>> a.sort(key=default_sort_key); a[0]
    2
    >>> min(a, key=default_sort_key)
    2

    Notes
    =====

    The key returned is useful for getting items into a canonical order
    that will be the same across platforms. It is not directly useful for
    sorting lists of expressions:

    >>> a, b = x, 1/x

    Since ``a`` has only 1 term, its value of sort_key is unaffected by
    ``order``:

    >>> a.sort_key() == a.sort_key('rev-lex')
    True

    If ``a`` and ``b`` are combined then the key will differ because there
    are terms that can be ordered:

    >>> eq = a + b
    >>> eq.sort_key() == eq.sort_key('rev-lex')
    False
    >>> eq.as_ordered_terms()
    [x, 1/x]
    >>> eq.as_ordered_terms('rev-lex')
    [1/x, x]

    But since the keys for each of these terms are independent of ``order``'s
    value, they do not sort differently when they appear separately in a list:

    >>> sorted(eq.args, key=default_sort_key)
    [1/x, x]
    >>> sorted(eq.args, key=lambda i: default_sort_key(i, order='rev-lex'))
    [1/x, x]

    The order of terms obtained when using these keys is the order that would
    be obtained if those terms were *factors* in a product.

    Although it is useful for quickly putting expressions in canonical order,
    it does not sort expressions based on their complexity defined by the
    number of operations, power of variables and others:

    >>> sorted([sin(x)*cos(x), sin(x)], key=default_sort_key)
    [sin(x)*cos(x), sin(x)]
    >>> sorted([x, x**2, sqrt(x), x**3], key=default_sort_key)
    [sqrt(x), x, x**2, x**3]

    See Also
    ========

    ordered, sympy.core.expr.Expr.as_ordered_factors, sympy.core.expr.Expr.as_ordered_terms

    """
    from .basic import Basic
    from .singleton import S

    if isinstance(item, Basic):
        return item.sort_key(order=order)

    if iterable(item, exclude=str):
        if isinstance(item, dict):
            args = item.items()
            unordered = True
        elif isinstance(item, set):
            args = item
            unordered = True
        else:
            # e.g. tuple, list
            args = list(item)
            unordered = False

        args = [default_sort_key(arg, order=order) for arg in args]

        if unordered:
            # e.g. dict, set
            args = sorted(args)

        cls_index, args = 10, (len(args), tuple(args))
    else:
        if not isinstance(item, str):
            try:
                item = sympify(item, strict=True)
            except SympifyError:
                # e.g. lambda x: x
                pass
            else:
                if isinstance(item, Basic):
                    # e.g int -> Integer
                    return default_sort_key(item)
                # e.g. UndefinedFunction

        # e.g. str
        cls_index, args = 0, (1, (str(item),))

    return (cls_index, 0, item.__class__.__name__
            ), args, S.One.sort_key(), S.One


def _node_count(e):
    # this not only counts nodes, it affirms that the
    # args are Basic (i.e. have an args property). If
    # some object has a non-Basic arg, it needs to be
    # fixed since it is intended that all Basic args
    # are of Basic type (though this is not easy to enforce).
    if e.is_Float:
        return 0.5
    return 1 + sum(map(_node_count, e.args))


def _nodes(e):
    """
    A helper for ordered() which returns the node count of ``e`` which
    for Basic objects is the number of Basic nodes in the expression tree
    but for other objects is 1 (unless the object is an iterable or dict
    for which the sum of nodes is returned).
    """
    from .basic import Basic
    from .function import Derivative

    if isinstance(e, Basic):
        if isinstance(e, Derivative):
            return _nodes(e.expr) + sum(i[1] if i[1].is_Number else
                _nodes(i[1]) for i in e.variable_count)
        return _node_count(e)
    elif iterable(e):
        return 1 + sum(_nodes(ei) for ei in e)
    elif isinstance(e, dict):
        return 1 + sum(_nodes(k) + _nodes(v) for k, v in e.items())
    else:
        return 1


def ordered(seq, keys=None, default=True, warn=False):
    """Return an iterator of the seq where keys are used to break ties
    in a conservative fashion: if, after applying a key, there are no
    ties then no other keys will be computed.

    Two default keys will be applied if 1) keys are not provided or
    2) the given keys do not resolve all ties (but only if ``default``
    is True). The two keys are ``_nodes`` (which places smaller
    expressions before large) and ``default_sort_key`` which (if the
    ``sort_key`` for an object is defined properly) should resolve
    any ties. This strategy is similar to sorting done by
    ``Basic.compare``, but differs in that ``ordered`` never makes a
    decision based on an objects name.

    If ``warn`` is True then an error will be raised if there were no
    keys remaining to break ties. This can be used if it was expected that
    there should be no ties between items that are not identical.

    Examples
    ========

    >>> from sympy import ordered, count_ops
    >>> from sympy.abc import x, y

    The count_ops is not sufficient to break ties in this list and the first
    two items appear in their original order (i.e. the sorting is stable):

    >>> list(ordered([y + 2, x + 2, x**2 + y + 3],
    ...    count_ops, default=False, warn=False))
    ...
    [y + 2, x + 2, x**2 + y + 3]

    The default_sort_key allows the tie to be broken:

    >>> list(ordered([y + 2, x + 2, x**2 + y + 3]))
    ...
    [x + 2, y + 2, x**2 + y + 3]

    Here, sequences are sorted by length, then sum:

    >>> seq, keys = [[[1, 2, 1], [0, 3, 1], [1, 1, 3], [2], [1]], [
    ...    lambda x: len(x),
    ...    lambda x: sum(x)]]
    ...
    >>> list(ordered(seq, keys, default=False, warn=False))
    [[1], [2], [1, 2, 1], [0, 3, 1], [1, 1, 3]]

    If ``warn`` is True, an error will be raised if there were not
    enough keys to break ties:

    >>> list(ordered(seq, keys, default=False, warn=True))
    Traceback (most recent call last):
    ...
    ValueError: not enough keys to break ties


    Notes
    =====

    The decorated sort is one of the fastest ways to sort a sequence for
    which special item comparison is desired: the sequence is decorated,
    sorted on the basis of the decoration (e.g. making all letters lower
    case) and then undecorated. If one wants to break ties for items that
    have the same decorated value, a second key can be used. But if the
    second key is expensive to compute then it is inefficient to decorate
    all items with both keys: only those items having identical first key
    values need to be decorated. This function applies keys successively
    only when needed to break ties. By yielding an iterator, use of the
    tie-breaker is delayed as long as possible.

    This function is best used in cases when use of the first key is
    expected to be a good hashing function; if there are no unique hashes
    from application of a key, then that key should not have been used. The
    exception, however, is that even if there are many collisions, if the
    first group is small and one does not need to process all items in the
    list then time will not be wasted sorting what one was not interested
    in. For example, if one were looking for the minimum in a list and
    there were several criteria used to define the sort order, then this
    function would be good at returning that quickly if the first group
    of candidates is small relative to the number of items being processed.

    """

    d = defaultdict(list)
    if keys:
        if isinstance(keys, (list, tuple)):
            keys = list(keys)
            f = keys.pop(0)
        else:
            f = keys
            keys = []
        for a in seq:
            d[f(a)].append(a)
    else:
        if not default:
            raise ValueError('if default=False then keys must be provided')
        d[None].extend(seq)

    for k, value in sorted(d.items()):
        if len(value) > 1:
            if keys:
                value = ordered(value, keys, default, warn)
            elif default:
                value = ordered(value, (_nodes, default_sort_key,),
                               default=False, warn=warn)
            elif warn:
                u = list(uniq(value))
                if len(u) > 1:
                    raise ValueError(
                        'not enough keys to break ties: %s' % u)
        yield from value

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\_compilation\util.py ===
from collections import namedtuple
from hashlib import sha256
import os
import shutil
import sys
import fnmatch

from sympy.testing.pytest import XFAIL


def may_xfail(func):
    if sys.platform.lower() == 'darwin' or os.name == 'nt':
        # sympy.utilities._compilation needs more testing on Windows and macOS
        # once those two platforms are reliably supported this xfail decorator
        # may be removed.
        return XFAIL(func)
    else:
        return func


class CompilerNotFoundError(FileNotFoundError):
    pass


class CompileError (Exception):
    """Failure to compile one or more C/C++ source files."""


def get_abspath(path, cwd='.'):
    """ Returns the absolute path.

    Parameters
    ==========

    path : str
        (relative) path.
    cwd : str
        Path to root of relative path.
    """
    if os.path.isabs(path):
        return path
    else:
        if not os.path.isabs(cwd):
            cwd = os.path.abspath(cwd)
        return os.path.abspath(
            os.path.join(cwd, path)
        )


def make_dirs(path):
    """ Create directories (equivalent of ``mkdir -p``). """
    if path[-1] == '/':
        parent = os.path.dirname(path[:-1])
    else:
        parent = os.path.dirname(path)

    if len(parent) > 0:
        if not os.path.exists(parent):
            make_dirs(parent)

    if not os.path.exists(path):
        os.mkdir(path, 0o777)
    else:
        assert os.path.isdir(path)

def missing_or_other_newer(path, other_path, cwd=None):
    """
    Investigate if path is non-existent or older than provided reference
    path.

    Parameters
    ==========
    path: string
        path to path which might be missing or too old
    other_path: string
        reference path
    cwd: string
        working directory (root of relative paths)

    Returns
    =======
    True if path is older or missing.
    """
    cwd = cwd or '.'
    path = get_abspath(path, cwd=cwd)
    other_path = get_abspath(other_path, cwd=cwd)
    if not os.path.exists(path):
        return True
    if os.path.getmtime(other_path) - 1e-6 >= os.path.getmtime(path):
        # 1e-6 is needed because http://stackoverflow.com/questions/17086426/
        return True
    return False

def copy(src, dst, only_update=False, copystat=True, cwd=None,
         dest_is_dir=False, create_dest_dirs=False):
    """ Variation of ``shutil.copy`` with extra options.

    Parameters
    ==========

    src : str
        Path to source file.
    dst : str
        Path to destination.
    only_update : bool
        Only copy if source is newer than destination
        (returns None if it was newer), default: ``False``.
    copystat : bool
        See ``shutil.copystat``. default: ``True``.
    cwd : str
        Path to working directory (root of relative paths).
    dest_is_dir : bool
        Ensures that dst is treated as a directory. default: ``False``
    create_dest_dirs : bool
        Creates directories if needed.

    Returns
    =======

    Path to the copied file.

    """
    if cwd:  # Handle working directory
        if not os.path.isabs(src):
            src = os.path.join(cwd, src)
        if not os.path.isabs(dst):
            dst = os.path.join(cwd, dst)

    if not os.path.exists(src):  # Make sure source file exists
        raise FileNotFoundError("Source: `{}` does not exist".format(src))

    # We accept both (re)naming destination file _or_
    # passing a (possible non-existent) destination directory
    if dest_is_dir:
        if not dst[-1] == '/':
            dst = dst+'/'
    else:
        if os.path.exists(dst) and os.path.isdir(dst):
            dest_is_dir = True

    if dest_is_dir:
        dest_dir = dst
        dest_fname = os.path.basename(src)
        dst = os.path.join(dest_dir, dest_fname)
    else:
        dest_dir = os.path.dirname(dst)

    if not os.path.exists(dest_dir):
        if create_dest_dirs:
            make_dirs(dest_dir)
        else:
            raise FileNotFoundError("You must create directory first.")

    if only_update:
        if not missing_or_other_newer(dst, src):
            return

    if os.path.islink(dst):
        dst = os.path.abspath(os.path.realpath(dst), cwd=cwd)

    shutil.copy(src, dst)
    if copystat:
        shutil.copystat(src, dst)

    return dst

Glob = namedtuple('Glob', 'pathname')
ArbitraryDepthGlob = namedtuple('ArbitraryDepthGlob', 'filename')

def glob_at_depth(filename_glob, cwd=None):
    if cwd is not None:
        cwd = '.'
    globbed = []
    for root, dirs, filenames in os.walk(cwd):
        for fn in filenames:
            # This is not tested:
            if fnmatch.fnmatch(fn, filename_glob):
                globbed.append(os.path.join(root, fn))
    return globbed

def sha256_of_file(path, nblocks=128):
    """ Computes the SHA256 hash of a file.

    Parameters
    ==========

    path : string
        Path to file to compute hash of.
    nblocks : int
        Number of blocks to read per iteration.

    Returns
    =======

    hashlib sha256 hash object. Use ``.digest()`` or ``.hexdigest()``
    on returned object to get binary or hex encoded string.
    """
    sh = sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(nblocks*sh.block_size), b''):
            sh.update(chunk)
    return sh


def sha256_of_string(string):
    """ Computes the SHA256 hash of a string. """
    sh = sha256()
    sh.update(string)
    return sh


def pyx_is_cplus(path):
    """
    Inspect a Cython source file (.pyx) and look for comment line like:

    # distutils: language = c++

    Returns True if such a file is present in the file, else False.
    """
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') and '=' in line:
                splitted = line.split('=')
                if len(splitted) != 2:
                    continue
                lhs, rhs = splitted
                if lhs.strip().split()[-1].lower() == 'language' and \
                       rhs.strip().split()[0].lower() == 'c++':
                            return True
    return False

def import_module_from_file(filename, only_if_newer_than=None):
    """ Imports Python extension (from shared object file)

    Provide a list of paths in `only_if_newer_than` to check
    timestamps of dependencies. import_ raises an ImportError
    if any is newer.

    Word of warning: The OS may cache shared objects which makes
    reimporting same path of an shared object file very problematic.

    It will not detect the new time stamp, nor new checksum, but will
    instead silently use old module. Use unique names for this reason.

    Parameters
    ==========

    filename : str
        Path to shared object.
    only_if_newer_than : iterable of strings
        Paths to dependencies of the shared object.

    Raises
    ======

    ``ImportError`` if any of the files specified in ``only_if_newer_than`` are newer
    than the file given by filename.
    """
    path, name = os.path.split(filename)
    name, ext = os.path.splitext(name)
    name = name.split('.')[0]
    if sys.version_info[0] == 2:
        from imp import find_module, load_module
        fobj, filename, data = find_module(name, [path])
        if only_if_newer_than:
            for dep in only_if_newer_than:
                if os.path.getmtime(filename) < os.path.getmtime(dep):
                    raise ImportError("{} is newer than {}".format(dep, filename))
        mod = load_module(name, fobj, filename, data)
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, filename)
        if spec is None:
            raise ImportError("Failed to import: '%s'" % filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def find_binary_of_command(candidates):
    """ Finds binary first matching name among candidates.

    Calls ``which`` from shutils for provided candidates and returns
    first hit.

    Parameters
    ==========

    candidates : iterable of str
        Names of candidate commands

    Raises
    ======

    CompilerNotFoundError if no candidates match.
    """
    from shutil import which
    for c in candidates:
        binary_path = which(c)
        if c and binary_path:
            return c, binary_path

    raise CompilerNotFoundError('No binary located for candidates: {}'.format(candidates))


def unique_list(l):
    """ Uniquify a list (skip duplicate items). """
    result = []
    for x in l:
        if x not in result:
            result.append(x)
    return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\fusions\fusion.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

from collections import deque

import onnx

from ..onnx_model import ONNXModel


class Fusion:
    """
    Base class for fusions.
    """

    def __init__(self, model: ONNXModel, fused_op_type: str, search_op_type: str):
        self.search_op_type: str = search_op_type
        self.fused_op_type: str = fused_op_type
        self.model: ONNXModel = model
        self.nodes_to_remove: list = []
        self.nodes_to_add: list = []

        self._new_node_name_prefix = self.fused_op_type + "_fused_" + self.search_op_type + "_"
        self._new_node_name_suffix = None  # int|None used to create unique node names for the fused ops.

    def fuse(
        self,
        node: onnx.NodeProto,
        input_name_to_nodes: dict[str, list[onnx.NodeProto]],
        output_name_to_node: dict[str, onnx.NodeProto],
    ):
        """
        Interface function for derived fusion classes. Tries to fuse a node sequence containing
        the specified node.
        """
        raise NotImplementedError

    def apply(self) -> bool:
        """
        Apply graph fusion on the entire model graph.
        """
        input_name_to_nodes = self.model.input_name_to_nodes()
        output_name_to_node = self.model.output_name_to_node()

        for node in self.model.nodes():
            if node.op_type == self.search_op_type:
                self.fuse(node, input_name_to_nodes, output_name_to_node)

        self.model.remove_nodes(self.nodes_to_remove)
        self.model.add_nodes(self.nodes_to_add)

        graph_updated = bool(self.nodes_to_remove or self.nodes_to_add)

        if graph_updated:
            self.model.remove_unused_constant()

        return graph_updated

    def create_unique_node_name(self):
        prefix = self._new_node_name_prefix

        if self._new_node_name_suffix is None:
            largest_suffix: int = self.model.get_largest_node_name_suffix(prefix)
            self._new_node_name_suffix = largest_suffix + 1

        new_name = f"{prefix}{self._new_node_name_suffix!s}"
        self._new_node_name_suffix += 1

        return new_name

    @staticmethod
    def is_safe_to_fuse_nodes(
        nodes_to_remove: list[onnx.NodeProto],
        keep_outputs: list[str],
        input_name_to_nodes: dict[str, list[onnx.NodeProto]],
        output_name_to_node: dict[str, onnx.NodeProto],
    ) -> bool:
        for node_to_remove in nodes_to_remove:
            for output_to_remove in node_to_remove.output:
                if output_to_remove in keep_outputs:
                    continue

                if output_to_remove in input_name_to_nodes:
                    for impacted_node in input_name_to_nodes[output_to_remove]:
                        if impacted_node not in nodes_to_remove:
                            # Not safe to remove nodes since output is used by impacted_node
                            return False
        return True

    @staticmethod
    def get_node_attribute(node: onnx.NodeProto, attribute_name: str):
        for attr in node.attribute:
            if attr.name == attribute_name:
                value = onnx.helper.get_attribute_value(attr)
                return value
        return None

    @staticmethod
    def input_index(node_output: str, child_node: onnx.NodeProto) -> int:
        for index, input_name in enumerate(child_node.input):
            if input_name == node_output:
                return index
        return -1

    @staticmethod
    def tensor_shape_to_list(tensor_type) -> list[int]:
        shape_list = []
        for d in tensor_type.shape.dim:
            if d.HasField("dim_value"):
                shape_list.append(d.dim_value)  # known dimension
            elif d.HasField("dim_param"):
                shape_list.append(d.dim_param)  # unknown dimension with symbolic name
            else:
                shape_list.append("?")  # shall not happen
        return shape_list

    def get_constant_input(self, node: onnx.NodeProto):
        for i, inp in enumerate(node.input):
            value = self.model.get_constant_value(inp)
            if value is not None:
                return i, value

        return None, None

    def find_constant_input(self, node: onnx.NodeProto, expected_value: float, delta: float = 0.000001) -> int:
        i, value = self.get_constant_input(node)
        if value is not None and value.size == 1 and abs(value - expected_value) < delta:
            return i

        return -1

    def has_constant_input(self, node: onnx.NodeProto, expected_value: float, delta: float = 0.000001) -> bool:
        return self.find_constant_input(node, expected_value, delta) >= 0

    def is_constant_with_specified_rank(self, output_name: str, rank: int) -> bool:
        value = self.model.get_constant_value(output_name)
        if value is None:
            return False  # Not an initializer

        if len(value.shape) != rank:
            return False  # Wrong dimensions

        return True

    def match_first_parent(
        self,
        node: onnx.NodeProto,
        parent_op_type: str,
        output_name_to_node: dict[str, onnx.NodeProto] | None = None,
        exclude: list[onnx.NodeProto] = [],  # noqa: B006
    ) -> tuple[onnx.NodeProto | None, int | None]:
        """
        Find parent node based on constraints on op_type.

        Args:
            node: current node.
            parent_op_type (str): constraint of parent node op_type.
            output_name_to_node (dict): dictionary with output name as key, and node as value.
            exclude (list): list of nodes that are excluded (not allowed to match as parent).

        Returns:
            parent: The matched parent node. None if not found.
            index: The input index of matched parent node. None if not found.
        """
        if output_name_to_node is None:
            output_name_to_node = self.model.output_name_to_node()

        for i, inp in enumerate(node.input):
            if inp in output_name_to_node:
                parent = output_name_to_node[inp]
                if parent.op_type == parent_op_type and parent not in exclude:
                    return parent, i

        return None, None

    def match_parent(
        self,
        node: onnx.NodeProto,
        parent_op_type: str,
        input_index: int | None = None,
        output_name_to_node: dict[str, onnx.NodeProto] | None = None,
        exclude: list[onnx.NodeProto] = [],  # noqa: B006
        return_indice: list[int] | None = None,
    ) -> onnx.NodeProto | None:
        """
        Find parent node based on constraints on op_type and index.
        When input_index is None, we will find the first parent node based on constraints,
        and return_indice will be appended the corresponding input index.

        Args:
            node (str): current node name.
            parent_op_type (str): constraint of parent node op_type.
            input_index (int or None): only check the parent given input index of current node.
            output_name_to_node (dict): dictionary with output name as key, and node as value.
            exclude (list): list of nodes that are excluded (not allowed to match as parent).
            return_indice (list): a list to append the input index when input_index is None.

        Returns:
            parent: The matched parent node.
        """
        assert node is not None
        assert input_index is None or input_index >= 0

        if output_name_to_node is None:
            output_name_to_node = self.model.output_name_to_node()

        if input_index is None:
            parent, index = self.match_first_parent(node, parent_op_type, output_name_to_node, exclude)
            if return_indice is not None:
                return_indice.append(index)
            return parent

        if input_index >= len(node.input):
            # Input index out of bounds.
            return None

        parent = self.model.get_parent(node, input_index, output_name_to_node)
        if parent is not None and parent.op_type == parent_op_type and parent not in exclude:
            return parent

        return None

    def match_parent_path(
        self,
        node: onnx.NodeProto,
        parent_op_types: list[str],
        parent_input_index: list[int] | None = None,
        output_name_to_node: dict[str, onnx.NodeProto] | None = None,
        return_indice: list[int] | None = None,
    ) -> list[onnx.NodeProto] | None:
        """
        Find a sequence of input edges based on constraints on parent op_type and index.
        When input_index is None, we will find the first parent node based on constraints,
        and return_indice will be appended the corresponding input index.

        Args:
            node (str): current node name.
            parent_op_types (str): constraint of parent node op_type of each input edge.
            parent_input_index (list): constraint of input index of each input edge. None means no constraint.
            output_name_to_node (dict): dictionary with output name as key, and node as value.
            return_indice (list): a list to append the input index
                                  When there is no constraint on input index of an edge.

        Returns:
            parents: a list of matched parent node.
        """
        if parent_input_index is not None:
            assert len(parent_input_index) == len(parent_op_types)

        if output_name_to_node is None:
            output_name_to_node = self.model.output_name_to_node()

        current_node = node
        matched_parents = []
        for i, op_type in enumerate(parent_op_types):
            matched_parent = self.match_parent(
                current_node,
                op_type,
                parent_input_index[i] if parent_input_index is not None else None,
                output_name_to_node,
                exclude=[],
                return_indice=return_indice,
            )
            if matched_parent is None:
                return None

            matched_parents.append(matched_parent)
            current_node = matched_parent

        return matched_parents

    def match_parent_paths(
        self,
        node: onnx.NodeProto,
        paths: list[tuple[list[str], list[int]]],
        output_name_to_node: dict[str, onnx.NodeProto],
    ) -> tuple[int, list[onnx.NodeProto] | None, list[int] | None]:
        """
        Find a matching parent path to the given node.
        """
        for i, path in enumerate(paths):
            return_indice = []
            matched = self.match_parent_path(node, path[0], path[1], output_name_to_node, return_indice)
            if matched:
                return i, matched, return_indice
        return -1, None, None

    def find_first_child_by_type(
        self,
        node: onnx.NodeProto,
        child_type: str,
        input_name_to_nodes: dict[str, list[onnx.NodeProto]] | None = None,
        recursive: bool = True,
    ) -> onnx.NodeProto | None:
        children = self.model.get_children(node, input_name_to_nodes)
        dq = deque(children)
        while len(dq) > 0:
            current_node = dq.pop()
            if current_node.op_type == child_type:
                return current_node

            if recursive:
                children = self.model.get_children(current_node, input_name_to_nodes)
                for child in children:
                    dq.appendleft(child)

        return None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\maple.py ===
"""
Maple code printer

The MapleCodePrinter converts single SymPy expressions into single
Maple expressions, using the functions defined in the Maple objects where possible.


FIXME: This module is still under actively developed. Some functions may be not completed.
"""

from sympy.core import S
from sympy.core.numbers import Integer, IntegerConstant, equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE

import sympy

_known_func_same_name = (
    'sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'sinh', 'cosh', 'tanh', 'sech',
    'csch', 'coth', 'exp', 'floor', 'factorial', 'bernoulli',  'euler',
    'fibonacci', 'gcd', 'lcm', 'conjugate', 'Ci', 'Chi', 'Ei', 'Li', 'Si', 'Shi',
    'erf', 'erfc', 'harmonic', 'LambertW',
    'sqrt', # For automatic rewrites
)

known_functions = {
    # SymPy -> Maple
    'Abs': 'abs',
    'log': 'ln',
    'asin': 'arcsin',
    'acos': 'arccos',
    'atan': 'arctan',
    'asec': 'arcsec',
    'acsc': 'arccsc',
    'acot': 'arccot',
    'asinh': 'arcsinh',
    'acosh': 'arccosh',
    'atanh': 'arctanh',
    'asech': 'arcsech',
    'acsch': 'arccsch',
    'acoth': 'arccoth',
    'ceiling': 'ceil',
    'Max' : 'max',
    'Min' : 'min',

    'factorial2': 'doublefactorial',
    'RisingFactorial': 'pochhammer',
    'besseli': 'BesselI',
    'besselj': 'BesselJ',
    'besselk': 'BesselK',
    'bessely': 'BesselY',
    'hankelh1': 'HankelH1',
    'hankelh2': 'HankelH2',
    'airyai': 'AiryAi',
    'airybi': 'AiryBi',
    'appellf1': 'AppellF1',
    'fresnelc': 'FresnelC',
    'fresnels': 'FresnelS',
    'lerchphi' : 'LerchPhi',
}

for _func in _known_func_same_name:
    known_functions[_func] = _func

number_symbols = {
    # SymPy -> Maple
    S.Pi: 'Pi',
    S.Exp1: 'exp(1)',
    S.Catalan: 'Catalan',
    S.EulerGamma: 'gamma',
    S.GoldenRatio: '(1/2 + (1/2)*sqrt(5))'
}

spec_relational_ops = {
    # SymPy -> Maple
    '==': '=',
    '!=': '<>'
}

not_supported_symbol = [
    S.ComplexInfinity
]

class MapleCodePrinter(CodePrinter):
    """
    Printer which converts a SymPy expression into a maple code.
    """
    printmethod = "_maple"
    language = "maple"

    _operators = {
        'and': 'and',
        'or': 'or',
        'not': 'not ',
    }

    _default_settings = dict(CodePrinter._default_settings, **{
        'inline': True,
        'allow_unknown_functions': True,
    })

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        super().__init__(settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)

    def _get_statement(self, codestring):
        return "%s;" % codestring

    def _get_comment(self, text):
        return "# {}".format(text)

    def _declare_number_const(self, name, value):
        return "{} := {};".format(name,
                                    value.evalf(self._settings['precision']))

    def _format_code(self, lines):
        return lines

    def _print_tuple(self, expr):
        return self._print(list(expr))

    def _print_Tuple(self, expr):
        return self._print(list(expr))

    def _print_Assignment(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return "{lhs} := {rhs}".format(lhs=lhs, rhs=rhs)

    def _print_Pow(self, expr, **kwargs):
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '1/%s' % (self.parenthesize(expr.base, PREC))
        elif equal_valued(expr.exp, 0.5):
            return 'sqrt(%s)' % self._print(expr.base)
        elif equal_valued(expr.exp, -0.5):
            return '1/sqrt(%s)' % self._print(expr.base)
        else:
            return '{base}^{exp}'.format(
                base=self.parenthesize(expr.base, PREC),
                exp=self.parenthesize(expr.exp, PREC))

    def _print_Piecewise(self, expr):
        if (expr.args[-1].cond is not True) and (expr.args[-1].cond != S.BooleanTrue):
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        _coup_list = [
            ("{c}, {e}".format(c=self._print(c),
                               e=self._print(e)) if c is not True and c is not S.BooleanTrue else "{e}".format(
                e=self._print(e)))
            for e, c in expr.args]
        _inbrace = ', '.join(_coup_list)
        return 'piecewise({_inbrace})'.format(_inbrace=_inbrace)

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return "{p}/{q}".format(p=str(p), q=str(q))

    def _print_Relational(self, expr):
        PREC=precedence(expr)
        lhs_code = self.parenthesize(expr.lhs, PREC)
        rhs_code = self.parenthesize(expr.rhs, PREC)
        op = expr.rel_op
        if op in spec_relational_ops:
            op = spec_relational_ops[op]
        return "{lhs} {rel_op} {rhs}".format(lhs=lhs_code, rel_op=op, rhs=rhs_code)

    def _print_NumberSymbol(self, expr):
        return number_symbols[expr]

    def _print_NegativeInfinity(self, expr):
        return '-infinity'

    def _print_Infinity(self, expr):
        return 'infinity'

    def _print_BooleanTrue(self, expr):
        return "true"

    def _print_BooleanFalse(self, expr):
        return "false"

    def _print_bool(self, expr):
        return 'true' if expr else 'false'

    def _print_NaN(self, expr):
        return 'undefined'

    def _get_matrix(self, expr, sparse=False):
        if S.Zero in expr.shape:
            _strM = 'Matrix([], storage = {storage})'.format(
                storage='sparse' if sparse else 'rectangular')
        else:
            _strM = 'Matrix({list}, storage = {storage})'.format(
                list=self._print(expr.tolist()),
                storage='sparse' if sparse else 'rectangular')
        return _strM

    def _print_MatrixElement(self, expr):
        return "{parent}[{i_maple}, {j_maple}]".format(
            parent=self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True),
            i_maple=self._print(expr.i + 1),
            j_maple=self._print(expr.j + 1))

    def _print_MatrixBase(self, expr):
        return self._get_matrix(expr, sparse=False)

    def _print_SparseRepMatrix(self, expr):
        return self._get_matrix(expr, sparse=True)

    def _print_Identity(self, expr):
        if isinstance(expr.rows, (Integer, IntegerConstant)):
            return self._print(sympy.SparseMatrix(expr))
        else:
            return "Matrix({var_size}, shape = identity)".format(var_size=self._print(expr.rows))

    def _print_MatMul(self, expr):
        PREC=precedence(expr)
        _fact_list = list(expr.args)
        _const = None
        if not isinstance(_fact_list[0], (sympy.MatrixBase, sympy.MatrixExpr,
                                          sympy.MatrixSlice, sympy.MatrixSymbol)):
            _const, _fact_list = _fact_list[0], _fact_list[1:]

        if _const is None or _const == 1:
            return '.'.join(self.parenthesize(_m, PREC) for _m in _fact_list)
        else:
            return '{c}*{m}'.format(c=_const, m='.'.join(self.parenthesize(_m, PREC) for _m in _fact_list))

    def _print_MatPow(self, expr):
        # This function requires LinearAlgebra Function in Maple
        return 'MatrixPower({A}, {n})'.format(A=self._print(expr.base), n=self._print(expr.exp))

    def _print_HadamardProduct(self, expr):
        PREC = precedence(expr)
        _fact_list = list(expr.args)
        return '*'.join(self.parenthesize(_m, PREC) for _m in _fact_list)

    def _print_Derivative(self, expr):
        _f, (_var, _order) = expr.args

        if _order != 1:
            _second_arg = '{var}${order}'.format(var=self._print(_var),
                                                 order=self._print(_order))
        else:
            _second_arg = '{var}'.format(var=self._print(_var))
        return 'diff({func_expr}, {sec_arg})'.format(func_expr=self._print(_f), sec_arg=_second_arg)


def maple_code(expr, assign_to=None, **settings):
    r"""Converts ``expr`` to a string of Maple code.

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned.  Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type.  This can be helpful for
        expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi  [default=16].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations.  Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)].  See
        below for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols.  If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text).  [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].
    inline: bool, optional
        If True, we try to create single-statement code instead of multiple
        statements.  [default=True].

    """
    return MapleCodePrinter(settings).doprint(expr, assign_to)


def print_maple_code(expr, **settings):
    """Prints the Maple representation of the given expression.

    See :func:`maple_code` for the meaning of the optional arguments.

    Examples
    ========

    >>> from sympy import print_maple_code, symbols
    >>> x, y = symbols('x y')
    >>> print_maple_code(x, assign_to=y)
    y := x
    """
    print(maple_code(expr, **settings))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\coloredlogs\converter\colors.py ===
# Mapping of ANSI color codes to HTML/CSS colors.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: January 14, 2018
# URL: https://coloredlogs.readthedocs.io

"""Mapping of ANSI color codes to HTML/CSS colors."""

EIGHT_COLOR_PALETTE = (
    '#010101',  # black
    '#DE382B',  # red
    '#39B54A',  # green
    '#FFC706',  # yellow
    '#006FB8',  # blue
    '#762671',  # magenta
    '#2CB5E9',  # cyan
    '#CCC',     # white
)
"""
A tuple of strings mapping basic color codes to CSS colors.

The items in this tuple correspond to the eight basic color codes for black,
red, green, yellow, blue, magenta, cyan and white as defined in the original
standard for ANSI escape sequences. The CSS colors are based on the `Ubuntu
color scheme`_ described on Wikipedia and they are encoded as hexadecimal
values to get the shortest strings, which reduces the size (in bytes) of
conversion output.

.. _Ubuntu color scheme: https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
"""

BRIGHT_COLOR_PALETTE = (
    '#808080',  # black
    '#F00',     # red
    '#0F0',     # green
    '#FF0',     # yellow
    '#00F',     # blue
    '#F0F',     # magenta
    '#0FF',     # cyan
    '#FFF',     # white
)
"""
A tuple of strings mapping bright color codes to CSS colors.

This tuple maps the bright color variants of :data:`EIGHT_COLOR_PALETTE`.
"""

EXTENDED_COLOR_PALETTE = (
    '#000000',
    '#800000',
    '#008000',
    '#808000',
    '#000080',
    '#800080',
    '#008080',
    '#C0C0C0',
    '#808080',
    '#FF0000',
    '#00FF00',
    '#FFFF00',
    '#0000FF',
    '#FF00FF',
    '#00FFFF',
    '#FFFFFF',
    '#000000',
    '#00005F',
    '#000087',
    '#0000AF',
    '#0000D7',
    '#0000FF',
    '#005F00',
    '#005F5F',
    '#005F87',
    '#005FAF',
    '#005FD7',
    '#005FFF',
    '#008700',
    '#00875F',
    '#008787',
    '#0087AF',
    '#0087D7',
    '#0087FF',
    '#00AF00',
    '#00AF5F',
    '#00AF87',
    '#00AFAF',
    '#00AFD7',
    '#00AFFF',
    '#00D700',
    '#00D75F',
    '#00D787',
    '#00D7AF',
    '#00D7D7',
    '#00D7FF',
    '#00FF00',
    '#00FF5F',
    '#00FF87',
    '#00FFAF',
    '#00FFD7',
    '#00FFFF',
    '#5F0000',
    '#5F005F',
    '#5F0087',
    '#5F00AF',
    '#5F00D7',
    '#5F00FF',
    '#5F5F00',
    '#5F5F5F',
    '#5F5F87',
    '#5F5FAF',
    '#5F5FD7',
    '#5F5FFF',
    '#5F8700',
    '#5F875F',
    '#5F8787',
    '#5F87AF',
    '#5F87D7',
    '#5F87FF',
    '#5FAF00',
    '#5FAF5F',
    '#5FAF87',
    '#5FAFAF',
    '#5FAFD7',
    '#5FAFFF',
    '#5FD700',
    '#5FD75F',
    '#5FD787',
    '#5FD7AF',
    '#5FD7D7',
    '#5FD7FF',
    '#5FFF00',
    '#5FFF5F',
    '#5FFF87',
    '#5FFFAF',
    '#5FFFD7',
    '#5FFFFF',
    '#870000',
    '#87005F',
    '#870087',
    '#8700AF',
    '#8700D7',
    '#8700FF',
    '#875F00',
    '#875F5F',
    '#875F87',
    '#875FAF',
    '#875FD7',
    '#875FFF',
    '#878700',
    '#87875F',
    '#878787',
    '#8787AF',
    '#8787D7',
    '#8787FF',
    '#87AF00',
    '#87AF5F',
    '#87AF87',
    '#87AFAF',
    '#87AFD7',
    '#87AFFF',
    '#87D700',
    '#87D75F',
    '#87D787',
    '#87D7AF',
    '#87D7D7',
    '#87D7FF',
    '#87FF00',
    '#87FF5F',
    '#87FF87',
    '#87FFAF',
    '#87FFD7',
    '#87FFFF',
    '#AF0000',
    '#AF005F',
    '#AF0087',
    '#AF00AF',
    '#AF00D7',
    '#AF00FF',
    '#AF5F00',
    '#AF5F5F',
    '#AF5F87',
    '#AF5FAF',
    '#AF5FD7',
    '#AF5FFF',
    '#AF8700',
    '#AF875F',
    '#AF8787',
    '#AF87AF',
    '#AF87D7',
    '#AF87FF',
    '#AFAF00',
    '#AFAF5F',
    '#AFAF87',
    '#AFAFAF',
    '#AFAFD7',
    '#AFAFFF',
    '#AFD700',
    '#AFD75F',
    '#AFD787',
    '#AFD7AF',
    '#AFD7D7',
    '#AFD7FF',
    '#AFFF00',
    '#AFFF5F',
    '#AFFF87',
    '#AFFFAF',
    '#AFFFD7',
    '#AFFFFF',
    '#D70000',
    '#D7005F',
    '#D70087',
    '#D700AF',
    '#D700D7',
    '#D700FF',
    '#D75F00',
    '#D75F5F',
    '#D75F87',
    '#D75FAF',
    '#D75FD7',
    '#D75FFF',
    '#D78700',
    '#D7875F',
    '#D78787',
    '#D787AF',
    '#D787D7',
    '#D787FF',
    '#D7AF00',
    '#D7AF5F',
    '#D7AF87',
    '#D7AFAF',
    '#D7AFD7',
    '#D7AFFF',
    '#D7D700',
    '#D7D75F',
    '#D7D787',
    '#D7D7AF',
    '#D7D7D7',
    '#D7D7FF',
    '#D7FF00',
    '#D7FF5F',
    '#D7FF87',
    '#D7FFAF',
    '#D7FFD7',
    '#D7FFFF',
    '#FF0000',
    '#FF005F',
    '#FF0087',
    '#FF00AF',
    '#FF00D7',
    '#FF00FF',
    '#FF5F00',
    '#FF5F5F',
    '#FF5F87',
    '#FF5FAF',
    '#FF5FD7',
    '#FF5FFF',
    '#FF8700',
    '#FF875F',
    '#FF8787',
    '#FF87AF',
    '#FF87D7',
    '#FF87FF',
    '#FFAF00',
    '#FFAF5F',
    '#FFAF87',
    '#FFAFAF',
    '#FFAFD7',
    '#FFAFFF',
    '#FFD700',
    '#FFD75F',
    '#FFD787',
    '#FFD7AF',
    '#FFD7D7',
    '#FFD7FF',
    '#FFFF00',
    '#FFFF5F',
    '#FFFF87',
    '#FFFFAF',
    '#FFFFD7',
    '#FFFFFF',
    '#080808',
    '#121212',
    '#1C1C1C',
    '#262626',
    '#303030',
    '#3A3A3A',
    '#444444',
    '#4E4E4E',
    '#585858',
    '#626262',
    '#6C6C6C',
    '#767676',
    '#808080',
    '#8A8A8A',
    '#949494',
    '#9E9E9E',
    '#A8A8A8',
    '#B2B2B2',
    '#BCBCBC',
    '#C6C6C6',
    '#D0D0D0',
    '#DADADA',
    '#E4E4E4',
    '#EEEEEE',
)
"""
A tuple of strings mapping 256 color mode color codes to CSS colors.

The items in this tuple correspond to the color codes in the 256 color mode palette.
"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\humanfriendly\terminal\spinners.py ===
# Human friendly input/output in Python.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: March 1, 2020
# URL: https://humanfriendly.readthedocs.io

"""
Support for spinners that represent progress on interactive terminals.

The :class:`Spinner` class shows a "spinner" on the terminal to let the user
know that something is happening during long running operations that would
otherwise be silent (leaving the user to wonder what they're waiting for).
Below are some visual examples that should illustrate the point.

**Simple spinners:**

 Here's a screen capture that shows the simplest form of spinner:

  .. image:: images/spinner-basic.gif
     :alt: Animated screen capture of a simple spinner.

 The following code was used to create the spinner above:

 .. code-block:: python

    import itertools
    import time
    from humanfriendly import Spinner

    with Spinner(label="Downloading") as spinner:
        for i in itertools.count():
            # Do something useful here.
            time.sleep(0.1)
            # Advance the spinner.
            spinner.step()

**Spinners that show elapsed time:**

 Here's a spinner that shows the elapsed time since it started:

  .. image:: images/spinner-with-timer.gif
     :alt: Animated screen capture of a spinner showing elapsed time.

 The following code was used to create the spinner above:

 .. code-block:: python

    import itertools
    import time
    from humanfriendly import Spinner, Timer

    with Spinner(label="Downloading", timer=Timer()) as spinner:
        for i in itertools.count():
            # Do something useful here.
            time.sleep(0.1)
            # Advance the spinner.
            spinner.step()

**Spinners that show progress:**

 Here's a spinner that shows a progress percentage:

  .. image:: images/spinner-with-progress.gif
     :alt: Animated screen capture of spinner showing progress.

 The following code was used to create the spinner above:

 .. code-block:: python

    import itertools
    import random
    import time
    from humanfriendly import Spinner, Timer

    with Spinner(label="Downloading", total=100) as spinner:
        progress = 0
        while progress < 100:
            # Do something useful here.
            time.sleep(0.1)
            # Advance the spinner.
            spinner.step(progress)
            # Determine the new progress value.
            progress += random.random() * 5

If you want to provide user feedback during a long running operation but it's
not practical to periodically call the :func:`~Spinner.step()` method consider
using :class:`AutomaticSpinner` instead.

As you may already have noticed in the examples above, :class:`Spinner` objects
can be used as context managers to automatically call :func:`Spinner.clear()`
when the spinner ends.
"""

# Standard library modules.
import multiprocessing
import sys
import time

# Modules included in our package.
from humanfriendly import Timer
from humanfriendly.deprecation import deprecated_args
from humanfriendly.terminal import ANSI_ERASE_LINE

# Public identifiers that require documentation.
__all__ = ("AutomaticSpinner", "GLYPHS", "MINIMUM_INTERVAL", "Spinner")

GLYPHS = ["-", "\\", "|", "/"]
"""A list of strings with characters that together form a crude animation :-)."""

MINIMUM_INTERVAL = 0.2
"""Spinners are redrawn with a frequency no higher than this number (a floating point number of seconds)."""


class Spinner(object):

    """Show a spinner on the terminal as a simple means of feedback to the user."""

    @deprecated_args('label', 'total', 'stream', 'interactive', 'timer')
    def __init__(self, **options):
        """
        Initialize a :class:`Spinner` object.

        :param label:

          The label for the spinner (a string or :data:`None`, defaults to
          :data:`None`).

        :param total:

          The expected number of steps (an integer or :data:`None`). If this is
          provided the spinner will show a progress percentage.

        :param stream:

          The output stream to show the spinner on (a file-like object,
          defaults to :data:`sys.stderr`).

        :param interactive:

          :data:`True` to enable rendering of the spinner, :data:`False` to
          disable (defaults to the result of ``stream.isatty()``).

        :param timer:

          A :class:`.Timer` object (optional). If this is given the spinner
          will show the elapsed time according to the timer.

        :param interval:

          The spinner will be updated at most once every this many seconds
          (a floating point number, defaults to :data:`MINIMUM_INTERVAL`).

        :param glyphs:

          A list of strings with single characters that are drawn in the same
          place in succession to implement a simple animated effect (defaults
          to :data:`GLYPHS`).
        """
        # Store initializer arguments.
        self.interactive = options.get('interactive')
        self.interval = options.get('interval', MINIMUM_INTERVAL)
        self.label = options.get('label')
        self.states = options.get('glyphs', GLYPHS)
        self.stream = options.get('stream', sys.stderr)
        self.timer = options.get('timer')
        self.total = options.get('total')
        # Define instance variables.
        self.counter = 0
        self.last_update = 0
        # Try to automatically discover whether the stream is connected to
        # a terminal, but don't fail if no isatty() method is available.
        if self.interactive is None:
            try:
                self.interactive = self.stream.isatty()
            except Exception:
                self.interactive = False

    def step(self, progress=0, label=None):
        """
        Advance the spinner by one step and redraw it.

        :param progress: The number of the current step, relative to the total
                         given to the :class:`Spinner` constructor (an integer,
                         optional). If not provided the spinner will not show
                         progress.
        :param label: The label to use while redrawing (a string, optional). If
                      not provided the label given to the :class:`Spinner`
                      constructor is used instead.

        This method advances the spinner by one step without starting a new
        line, causing an animated effect which is very simple but much nicer
        than waiting for a prompt which is completely silent for a long time.

        .. note:: This method uses time based rate limiting to avoid redrawing
                  the spinner too frequently. If you know you're dealing with
                  code that will call :func:`step()` at a high frequency,
                  consider using :func:`sleep()` to avoid creating the
                  equivalent of a busy loop that's rate limiting the spinner
                  99% of the time.
        """
        if self.interactive:
            time_now = time.time()
            if time_now - self.last_update >= self.interval:
                self.last_update = time_now
                state = self.states[self.counter % len(self.states)]
                label = label or self.label
                if not label:
                    raise Exception("No label set for spinner!")
                elif self.total and progress:
                    label = "%s: %.2f%%" % (label, progress / (self.total / 100.0))
                elif self.timer and self.timer.elapsed_time > 2:
                    label = "%s (%s)" % (label, self.timer.rounded)
                self.stream.write("%s %s %s ..\r" % (ANSI_ERASE_LINE, state, label))
                self.counter += 1

    def sleep(self):
        """
        Sleep for a short period before redrawing the spinner.

        This method is useful when you know you're dealing with code that will
        call :func:`step()` at a high frequency. It will sleep for the interval
        with which the spinner is redrawn (less than a second). This avoids
        creating the equivalent of a busy loop that's rate limiting the
        spinner 99% of the time.

        This method doesn't redraw the spinner, you still have to call
        :func:`step()` in order to do that.
        """
        time.sleep(MINIMUM_INTERVAL)

    def clear(self):
        """
        Clear the spinner.

        The next line which is shown on the standard output or error stream
        after calling this method will overwrite the line that used to show the
        spinner.
        """
        if self.interactive:
            self.stream.write(ANSI_ERASE_LINE)

    def __enter__(self):
        """
        Enable the use of spinners as context managers.

        :returns: The :class:`Spinner` object.
        """
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """Clear the spinner when leaving the context."""
        self.clear()


class AutomaticSpinner(object):

    """
    Show a spinner on the terminal that automatically starts animating.

    This class shows a spinner on the terminal (just like :class:`Spinner`
    does) that automatically starts animating. This class should be used as a
    context manager using the :keyword:`with` statement. The animation
    continues for as long as the context is active.

    :class:`AutomaticSpinner` provides an alternative to :class:`Spinner`
    for situations where it is not practical for the caller to periodically
    call :func:`~Spinner.step()` to advance the animation, e.g. because
    you're performing a blocking call and don't fancy implementing threading or
    subprocess handling just to provide some user feedback.

    This works using the :mod:`multiprocessing` module by spawning a
    subprocess to render the spinner while the main process is busy doing
    something more useful. By using the :keyword:`with` statement you're
    guaranteed that the subprocess is properly terminated at the appropriate
    time.
    """

    def __init__(self, label, show_time=True):
        """
        Initialize an automatic spinner.

        :param label: The label for the spinner (a string).
        :param show_time: If this is :data:`True` (the default) then the spinner
                          shows elapsed time.
        """
        self.label = label
        self.show_time = show_time
        self.shutdown_event = multiprocessing.Event()
        self.subprocess = multiprocessing.Process(target=self._target)

    def __enter__(self):
        """Enable the use of automatic spinners as context managers."""
        self.subprocess.start()

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """Enable the use of automatic spinners as context managers."""
        self.shutdown_event.set()
        self.subprocess.join()

    def _target(self):
        try:
            timer = Timer() if self.show_time else None
            with Spinner(label=self.label, timer=timer) as spinner:
                while not self.shutdown_event.is_set():
                    spinner.step()
                    spinner.sleep()
        except KeyboardInterrupt:
            # Swallow Control-C signals without producing a nasty traceback that
            # won't make any sense to the average user.
            pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\sas\sas_constants.py ===
from __future__ import annotations

from typing import Final

magic: Final = (
    b"\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\xc2\xea\x81\x60"
    b"\xb3\x14\x11\xcf\xbd\x92\x08\x00"
    b"\x09\xc7\x31\x8c\x18\x1f\x10\x11"
)

align_1_checker_value: Final = b"3"
align_1_offset: Final = 32
align_1_length: Final = 1
align_1_value: Final = 4
u64_byte_checker_value: Final = b"3"
align_2_offset: Final = 35
align_2_length: Final = 1
align_2_value: Final = 4
endianness_offset: Final = 37
endianness_length: Final = 1
platform_offset: Final = 39
platform_length: Final = 1
encoding_offset: Final = 70
encoding_length: Final = 1
dataset_offset: Final = 92
dataset_length: Final = 64
file_type_offset: Final = 156
file_type_length: Final = 8
date_created_offset: Final = 164
date_created_length: Final = 8
date_modified_offset: Final = 172
date_modified_length: Final = 8
header_size_offset: Final = 196
header_size_length: Final = 4
page_size_offset: Final = 200
page_size_length: Final = 4
page_count_offset: Final = 204
page_count_length: Final = 4
sas_release_offset: Final = 216
sas_release_length: Final = 8
sas_server_type_offset: Final = 224
sas_server_type_length: Final = 16
os_version_number_offset: Final = 240
os_version_number_length: Final = 16
os_maker_offset: Final = 256
os_maker_length: Final = 16
os_name_offset: Final = 272
os_name_length: Final = 16
page_bit_offset_x86: Final = 16
page_bit_offset_x64: Final = 32
subheader_pointer_length_x86: Final = 12
subheader_pointer_length_x64: Final = 24
page_type_offset: Final = 0
page_type_length: Final = 2
block_count_offset: Final = 2
block_count_length: Final = 2
subheader_count_offset: Final = 4
subheader_count_length: Final = 2
page_type_mask: Final = 0x0F00
# Keep "page_comp_type" bits
page_type_mask2: Final = 0xF000 | page_type_mask
page_meta_type: Final = 0x0000
page_data_type: Final = 0x0100
page_mix_type: Final = 0x0200
page_amd_type: Final = 0x0400
page_meta2_type: Final = 0x4000
page_comp_type: Final = 0x9000
page_meta_types: Final = [page_meta_type, page_meta2_type]
subheader_pointers_offset: Final = 8
truncated_subheader_id: Final = 1
compressed_subheader_id: Final = 4
compressed_subheader_type: Final = 1
text_block_size_length: Final = 2
row_length_offset_multiplier: Final = 5
row_count_offset_multiplier: Final = 6
col_count_p1_multiplier: Final = 9
col_count_p2_multiplier: Final = 10
row_count_on_mix_page_offset_multiplier: Final = 15
column_name_pointer_length: Final = 8
column_name_text_subheader_offset: Final = 0
column_name_text_subheader_length: Final = 2
column_name_offset_offset: Final = 2
column_name_offset_length: Final = 2
column_name_length_offset: Final = 4
column_name_length_length: Final = 2
column_data_offset_offset: Final = 8
column_data_length_offset: Final = 8
column_data_length_length: Final = 4
column_type_offset: Final = 14
column_type_length: Final = 1
column_format_text_subheader_index_offset: Final = 22
column_format_text_subheader_index_length: Final = 2
column_format_offset_offset: Final = 24
column_format_offset_length: Final = 2
column_format_length_offset: Final = 26
column_format_length_length: Final = 2
column_label_text_subheader_index_offset: Final = 28
column_label_text_subheader_index_length: Final = 2
column_label_offset_offset: Final = 30
column_label_offset_length: Final = 2
column_label_length_offset: Final = 32
column_label_length_length: Final = 2
rle_compression: Final = b"SASYZCRL"
rdc_compression: Final = b"SASYZCR2"

compression_literals: Final = [rle_compression, rdc_compression]

# Incomplete list of encodings, using SAS nomenclature:
# https://support.sas.com/documentation/onlinedoc/dfdmstudio/2.6/dmpdmsug/Content/dfU_Encodings_SAS.html
# corresponding to the Python documentation of standard encodings
# https://docs.python.org/3/library/codecs.html#standard-encodings
encoding_names: Final = {
    20: "utf-8",
    29: "latin1",
    30: "latin2",
    31: "latin3",
    32: "latin4",
    33: "cyrillic",
    34: "arabic",
    35: "greek",
    36: "hebrew",
    37: "latin5",
    38: "latin6",
    39: "cp874",
    40: "latin9",
    41: "cp437",
    42: "cp850",
    43: "cp852",
    44: "cp857",
    45: "cp858",
    46: "cp862",
    47: "cp864",
    48: "cp865",
    49: "cp866",
    50: "cp869",
    51: "cp874",
    # 52: "",  # not found
    # 53: "",  # not found
    # 54: "",  # not found
    55: "cp720",
    56: "cp737",
    57: "cp775",
    58: "cp860",
    59: "cp863",
    60: "cp1250",
    61: "cp1251",
    62: "cp1252",
    63: "cp1253",
    64: "cp1254",
    65: "cp1255",
    66: "cp1256",
    67: "cp1257",
    68: "cp1258",
    118: "cp950",
    # 119: "",  # not found
    123: "big5",
    125: "gb2312",
    126: "cp936",
    134: "euc_jp",
    136: "cp932",
    138: "shift_jis",
    140: "euc-kr",
    141: "cp949",
    227: "latin8",
    # 228: "", # not found
    # 229: ""  # not found
}


class SASIndex:
    row_size_index: Final = 0
    column_size_index: Final = 1
    subheader_counts_index: Final = 2
    column_text_index: Final = 3
    column_name_index: Final = 4
    column_attributes_index: Final = 5
    format_and_label_index: Final = 6
    column_list_index: Final = 7
    data_subheader_index: Final = 8


subheader_signature_to_index: Final = {
    b"\xF7\xF7\xF7\xF7": SASIndex.row_size_index,
    b"\x00\x00\x00\x00\xF7\xF7\xF7\xF7": SASIndex.row_size_index,
    b"\xF7\xF7\xF7\xF7\x00\x00\x00\x00": SASIndex.row_size_index,
    b"\xF7\xF7\xF7\xF7\xFF\xFF\xFB\xFE": SASIndex.row_size_index,
    b"\xF6\xF6\xF6\xF6": SASIndex.column_size_index,
    b"\x00\x00\x00\x00\xF6\xF6\xF6\xF6": SASIndex.column_size_index,
    b"\xF6\xF6\xF6\xF6\x00\x00\x00\x00": SASIndex.column_size_index,
    b"\xF6\xF6\xF6\xF6\xFF\xFF\xFB\xFE": SASIndex.column_size_index,
    b"\x00\xFC\xFF\xFF": SASIndex.subheader_counts_index,
    b"\xFF\xFF\xFC\x00": SASIndex.subheader_counts_index,
    b"\x00\xFC\xFF\xFF\xFF\xFF\xFF\xFF": SASIndex.subheader_counts_index,
    b"\xFF\xFF\xFF\xFF\xFF\xFF\xFC\x00": SASIndex.subheader_counts_index,
    b"\xFD\xFF\xFF\xFF": SASIndex.column_text_index,
    b"\xFF\xFF\xFF\xFD": SASIndex.column_text_index,
    b"\xFD\xFF\xFF\xFF\xFF\xFF\xFF\xFF": SASIndex.column_text_index,
    b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFD": SASIndex.column_text_index,
    b"\xFF\xFF\xFF\xFF": SASIndex.column_name_index,
    b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF": SASIndex.column_name_index,
    b"\xFC\xFF\xFF\xFF": SASIndex.column_attributes_index,
    b"\xFF\xFF\xFF\xFC": SASIndex.column_attributes_index,
    b"\xFC\xFF\xFF\xFF\xFF\xFF\xFF\xFF": SASIndex.column_attributes_index,
    b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFC": SASIndex.column_attributes_index,
    b"\xFE\xFB\xFF\xFF": SASIndex.format_and_label_index,
    b"\xFF\xFF\xFB\xFE": SASIndex.format_and_label_index,
    b"\xFE\xFB\xFF\xFF\xFF\xFF\xFF\xFF": SASIndex.format_and_label_index,
    b"\xFF\xFF\xFF\xFF\xFF\xFF\xFB\xFE": SASIndex.format_and_label_index,
    b"\xFE\xFF\xFF\xFF": SASIndex.column_list_index,
    b"\xFF\xFF\xFF\xFE": SASIndex.column_list_index,
    b"\xFE\xFF\xFF\xFF\xFF\xFF\xFF\xFF": SASIndex.column_list_index,
    b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFE": SASIndex.column_list_index,
}


# List of frequently used SAS date and datetime formats
# http://support.sas.com/documentation/cdl/en/etsug/60372/HTML/default/viewer.htm#etsug_intervals_sect009.htm
# https://github.com/epam/parso/blob/master/src/main/java/com/epam/parso/impl/SasFileConstants.java
sas_date_formats: Final = (
    "DATE",
    "DAY",
    "DDMMYY",
    "DOWNAME",
    "JULDAY",
    "JULIAN",
    "MMDDYY",
    "MMYY",
    "MMYYC",
    "MMYYD",
    "MMYYP",
    "MMYYS",
    "MMYYN",
    "MONNAME",
    "MONTH",
    "MONYY",
    "QTR",
    "QTRR",
    "NENGO",
    "WEEKDATE",
    "WEEKDATX",
    "WEEKDAY",
    "WEEKV",
    "WORDDATE",
    "WORDDATX",
    "YEAR",
    "YYMM",
    "YYMMC",
    "YYMMD",
    "YYMMP",
    "YYMMS",
    "YYMMN",
    "YYMON",
    "YYMMDD",
    "YYQ",
    "YYQC",
    "YYQD",
    "YYQP",
    "YYQS",
    "YYQN",
    "YYQR",
    "YYQRC",
    "YYQRD",
    "YYQRP",
    "YYQRS",
    "YYQRN",
    "YYMMDDP",
    "YYMMDDC",
    "E8601DA",
    "YYMMDDN",
    "MMDDYYC",
    "MMDDYYS",
    "MMDDYYD",
    "YYMMDDS",
    "B8601DA",
    "DDMMYYN",
    "YYMMDDD",
    "DDMMYYB",
    "DDMMYYP",
    "MMDDYYP",
    "YYMMDDB",
    "MMDDYYN",
    "DDMMYYC",
    "DDMMYYD",
    "DDMMYYS",
    "MINGUO",
)

sas_datetime_formats: Final = (
    "DATETIME",
    "DTWKDATX",
    "B8601DN",
    "B8601DT",
    "B8601DX",
    "B8601DZ",
    "B8601LX",
    "E8601DN",
    "E8601DT",
    "E8601DX",
    "E8601DZ",
    "E8601LX",
    "DATEAMPM",
    "DTDATE",
    "DTMONYY",
    "DTMONYY",
    "DTWKDATX",
    "DTYEAR",
    "TOD",
    "MDYAMPM",
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_traps.py ===
"""These are the only functions that ever yield back to the task runner."""

from __future__ import annotations

import enum
import types

# Jedi gets mad in test_static_tool_sees_class_members if we use collections Callable
from typing import TYPE_CHECKING, Any, Callable, NoReturn, Union, cast

import attrs
import outcome

from . import _run

if TYPE_CHECKING:
    from collections.abc import Awaitable, Generator

    from typing_extensions import TypeAlias

    from ._run import Task

RaiseCancelT: TypeAlias = Callable[[], NoReturn]


# This class object is used as a singleton.
# Not exported in the trio._core namespace, but imported directly by _run.
class CancelShieldedCheckpoint:
    __slots__ = ()


# Not exported in the trio._core namespace, but imported directly by _run.
@attrs.frozen(slots=False)
class WaitTaskRescheduled:
    abort_func: Callable[[RaiseCancelT], Abort]


# Not exported in the trio._core namespace, but imported directly by _run.
@attrs.frozen(slots=False)
class PermanentlyDetachCoroutineObject:
    final_outcome: outcome.Outcome[object]


MessageType: TypeAlias = Union[
    type[CancelShieldedCheckpoint],
    WaitTaskRescheduled,
    PermanentlyDetachCoroutineObject,
    object,
]


# Helper for the bottommost 'yield'. You can't use 'yield' inside an async
# function, but you can inside a generator, and if you decorate your generator
# with @types.coroutine, then it's even awaitable. However, it's still not a
# real async function: in particular, it isn't recognized by
# inspect.iscoroutinefunction, and it doesn't trigger the unawaited coroutine
# tracking machinery. Since our traps are public APIs, we make them real async
# functions, and then this helper takes care of the actual yield:
@types.coroutine
def _real_async_yield(
    obj: MessageType,
) -> Generator[MessageType, None, None]:
    return (yield obj)


# Real yield value is from trio's main loop, but type checkers can't
# understand that, so we cast it to make type checkers understand.
_async_yield = cast(
    "Callable[[MessageType], Awaitable[outcome.Outcome[object]]]",
    _real_async_yield,
)


async def cancel_shielded_checkpoint() -> None:
    """Introduce a schedule point, but not a cancel point.

    This is *not* a :ref:`checkpoint <checkpoints>`, but it is half of a
    checkpoint, and when combined with :func:`checkpoint_if_cancelled` it can
    make a full checkpoint.

    Equivalent to (but potentially more efficient than)::

        with trio.CancelScope(shield=True):
            await trio.lowlevel.checkpoint()

    """
    (await _async_yield(CancelShieldedCheckpoint)).unwrap()


# Return values for abort functions
class Abort(enum.Enum):
    """:class:`enum.Enum` used as the return value from abort functions.

    See :func:`wait_task_rescheduled` for details.

    .. data:: SUCCEEDED
              FAILED

    """

    SUCCEEDED = 1
    FAILED = 2


# Should always return the type a Task "expects", unless you willfully reschedule it
# with a bad value.
async def wait_task_rescheduled(  # type: ignore[explicit-any]
    abort_func: Callable[[RaiseCancelT], Abort],
) -> Any:
    """Put the current task to sleep, with cancellation support.

    This is the lowest-level API for blocking in Trio. Every time a
    :class:`~trio.lowlevel.Task` blocks, it does so by calling this function
    (usually indirectly via some higher-level API).

    This is a tricky interface with no guard rails. If you can use
    :class:`ParkingLot` or the built-in I/O wait functions instead, then you
    should.

    Generally the way it works is that before calling this function, you make
    arrangements for "someone" to call :func:`reschedule` on the current task
    at some later point.

    Then you call :func:`wait_task_rescheduled`, passing in ``abort_func``, an
    "abort callback".

    (Terminology: in Trio, "aborting" is the process of attempting to
    interrupt a blocked task to deliver a cancellation.)

    There are two possibilities for what happens next:

    1. "Someone" calls :func:`reschedule` on the current task, and
       :func:`wait_task_rescheduled` returns or raises whatever value or error
       was passed to :func:`reschedule`.

    2. The call's context transitions to a cancelled state (e.g. due to a
       timeout expiring). When this happens, the ``abort_func`` is called. Its
       interface looks like::

           def abort_func(raise_cancel):
               ...
               return trio.lowlevel.Abort.SUCCEEDED  # or FAILED

       It should attempt to clean up any state associated with this call, and
       in particular, arrange that :func:`reschedule` will *not* be called
       later. If (and only if!) it is successful, then it should return
       :data:`Abort.SUCCEEDED`, in which case the task will automatically be
       rescheduled with an appropriate :exc:`~trio.Cancelled` error.

       Otherwise, it should return :data:`Abort.FAILED`. This means that the
       task can't be cancelled at this time, and still has to make sure that
       "someone" eventually calls :func:`reschedule`.

       At that point there are again two possibilities. You can simply ignore
       the cancellation altogether: wait for the operation to complete and
       then reschedule and continue as normal. (For example, this is what
       :func:`trio.to_thread.run_sync` does if cancellation is disabled.)
       The other possibility is that the ``abort_func`` does succeed in
       cancelling the operation, but for some reason isn't able to report that
       right away. (Example: on Windows, it's possible to request that an
       async ("overlapped") I/O operation be cancelled, but this request is
       *also* asynchronous – you don't find out until later whether the
       operation was actually cancelled or not.)  To report a delayed
       cancellation, then you should reschedule the task yourself, and call
       the ``raise_cancel`` callback passed to ``abort_func`` to raise a
       :exc:`~trio.Cancelled` (or possibly :exc:`KeyboardInterrupt`) exception
       into this task. Either of the approaches sketched below can work::

          # Option 1:
          # Catch the exception from raise_cancel and inject it into the task.
          # (This is what Trio does automatically for you if you return
          # Abort.SUCCEEDED.)
          trio.lowlevel.reschedule(task, outcome.capture(raise_cancel))

          # Option 2:
          # wait to be woken by "someone", and then decide whether to raise
          # the error from inside the task.
          outer_raise_cancel = None
          def abort(inner_raise_cancel):
              nonlocal outer_raise_cancel
              outer_raise_cancel = inner_raise_cancel
              TRY_TO_CANCEL_OPERATION()
              return trio.lowlevel.Abort.FAILED
          await wait_task_rescheduled(abort)
          if OPERATION_WAS_SUCCESSFULLY_CANCELLED:
              # raises the error
              outer_raise_cancel()

       In any case it's guaranteed that we only call the ``abort_func`` at most
       once per call to :func:`wait_task_rescheduled`.

    Sometimes, it's useful to be able to share some mutable sleep-related data
    between the sleeping task, the abort function, and the waking task. You
    can use the sleeping task's :data:`~Task.custom_sleep_data` attribute to
    store this data, and Trio won't touch it, except to make sure that it gets
    cleared when the task is rescheduled.

    .. warning::

       If your ``abort_func`` raises an error, or returns any value other than
       :data:`Abort.SUCCEEDED` or :data:`Abort.FAILED`, then Trio will crash
       violently. Be careful! Similarly, it is entirely possible to deadlock a
       Trio program by failing to reschedule a blocked task, or cause havoc by
       calling :func:`reschedule` too many times. Remember what we said up
       above about how you should use a higher-level API if at all possible?

    """
    return (await _async_yield(WaitTaskRescheduled(abort_func))).unwrap()


async def permanently_detach_coroutine_object(
    final_outcome: outcome.Outcome[object],
) -> object:
    """Permanently detach the current task from the Trio scheduler.

    Normally, a Trio task doesn't exit until its coroutine object exits. When
    you call this function, Trio acts like the coroutine object just exited
    and the task terminates with the given outcome. This is useful if you want
    to permanently switch the coroutine object over to a different coroutine
    runner.

    When the calling coroutine enters this function it's running under Trio,
    and when the function returns it's running under the foreign coroutine
    runner.

    You should make sure that the coroutine object has released any
    Trio-specific resources it has acquired (e.g. nurseries).

    Args:
      final_outcome (outcome.Outcome): Trio acts as if the current task exited
          with the given return value or exception.

    Returns or raises whatever value or exception the new coroutine runner
    uses to resume the coroutine.

    """
    if _run.current_task().child_nurseries:
        raise RuntimeError(
            "can't permanently detach a coroutine object with open nurseries",
        )
    return await _async_yield(PermanentlyDetachCoroutineObject(final_outcome))


async def temporarily_detach_coroutine_object(
    abort_func: Callable[[RaiseCancelT], Abort],
) -> object:
    """Temporarily detach the current coroutine object from the Trio
    scheduler.

    When the calling coroutine enters this function it's running under Trio,
    and when the function returns it's running under the foreign coroutine
    runner.

    The Trio :class:`Task` will continue to exist, but will be suspended until
    you use :func:`reattach_detached_coroutine_object` to resume it. In the
    mean time, you can use another coroutine runner to schedule the coroutine
    object. In fact, you have to – the function doesn't return until the
    coroutine is advanced from outside.

    Note that you'll need to save the current :class:`Task` object to later
    resume; you can retrieve it with :func:`current_task`. You can also use
    this :class:`Task` object to retrieve the coroutine object – see
    :data:`Task.coro`.

    Args:
      abort_func: Same as for :func:`wait_task_rescheduled`, except that it
          must return :data:`Abort.FAILED`. (If it returned
          :data:`Abort.SUCCEEDED`, then Trio would attempt to reschedule the
          detached task directly without going through
          :func:`reattach_detached_coroutine_object`, which would be bad.)
          Your ``abort_func`` should still arrange for whatever the coroutine
          object is doing to be cancelled, and then reattach to Trio and call
          the ``raise_cancel`` callback, if possible.

    Returns or raises whatever value or exception the new coroutine runner
    uses to resume the coroutine.

    """
    return await _async_yield(WaitTaskRescheduled(abort_func))


async def reattach_detached_coroutine_object(task: Task, yield_value: object) -> None:
    """Reattach a coroutine object that was detached using
    :func:`temporarily_detach_coroutine_object`.

    When the calling coroutine enters this function it's running under the
    foreign coroutine runner, and when the function returns it's running under
    Trio.

    This must be called from inside the coroutine being resumed, and yields
    whatever value you pass in. (Presumably you'll pass a value that will
    cause the current coroutine runner to stop scheduling this task.) Then the
    coroutine is resumed by the Trio scheduler at the next opportunity.

    Args:
      task (Task): The Trio task object that the current coroutine was
          detached from.
      yield_value (object): The object to yield to the current coroutine
          runner.

    """
    # This is a kind of crude check – in particular, it can fail if the
    # passed-in task is where the coroutine *runner* is running. But this is
    # an experts-only interface, and there's no easy way to do a more accurate
    # check, so I guess that's OK.
    if not task.coro.cr_running:
        raise RuntimeError("given task does not match calling coroutine")
    _run.reschedule(task, outcome.Value("reattaching"))
    value = await _async_yield(yield_value)
    assert value == outcome.Value("reattaching")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\_palettes.py ===
from .palette import Palette


# Taken from https://en.wikipedia.org/wiki/ANSI_escape_code (Windows 10 column)
WINDOWS_PALETTE = Palette(
    [
        (12, 12, 12),
        (197, 15, 31),
        (19, 161, 14),
        (193, 156, 0),
        (0, 55, 218),
        (136, 23, 152),
        (58, 150, 221),
        (204, 204, 204),
        (118, 118, 118),
        (231, 72, 86),
        (22, 198, 12),
        (249, 241, 165),
        (59, 120, 255),
        (180, 0, 158),
        (97, 214, 214),
        (242, 242, 242),
    ]
)

# # The standard ansi colors (including bright variants)
STANDARD_PALETTE = Palette(
    [
        (0, 0, 0),
        (170, 0, 0),
        (0, 170, 0),
        (170, 85, 0),
        (0, 0, 170),
        (170, 0, 170),
        (0, 170, 170),
        (170, 170, 170),
        (85, 85, 85),
        (255, 85, 85),
        (85, 255, 85),
        (255, 255, 85),
        (85, 85, 255),
        (255, 85, 255),
        (85, 255, 255),
        (255, 255, 255),
    ]
)


# The 256 color palette
EIGHT_BIT_PALETTE = Palette(
    [
        (0, 0, 0),
        (128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 128),
        (0, 128, 128),
        (192, 192, 192),
        (128, 128, 128),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (0, 0, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
        (0, 0, 0),
        (0, 0, 95),
        (0, 0, 135),
        (0, 0, 175),
        (0, 0, 215),
        (0, 0, 255),
        (0, 95, 0),
        (0, 95, 95),
        (0, 95, 135),
        (0, 95, 175),
        (0, 95, 215),
        (0, 95, 255),
        (0, 135, 0),
        (0, 135, 95),
        (0, 135, 135),
        (0, 135, 175),
        (0, 135, 215),
        (0, 135, 255),
        (0, 175, 0),
        (0, 175, 95),
        (0, 175, 135),
        (0, 175, 175),
        (0, 175, 215),
        (0, 175, 255),
        (0, 215, 0),
        (0, 215, 95),
        (0, 215, 135),
        (0, 215, 175),
        (0, 215, 215),
        (0, 215, 255),
        (0, 255, 0),
        (0, 255, 95),
        (0, 255, 135),
        (0, 255, 175),
        (0, 255, 215),
        (0, 255, 255),
        (95, 0, 0),
        (95, 0, 95),
        (95, 0, 135),
        (95, 0, 175),
        (95, 0, 215),
        (95, 0, 255),
        (95, 95, 0),
        (95, 95, 95),
        (95, 95, 135),
        (95, 95, 175),
        (95, 95, 215),
        (95, 95, 255),
        (95, 135, 0),
        (95, 135, 95),
        (95, 135, 135),
        (95, 135, 175),
        (95, 135, 215),
        (95, 135, 255),
        (95, 175, 0),
        (95, 175, 95),
        (95, 175, 135),
        (95, 175, 175),
        (95, 175, 215),
        (95, 175, 255),
        (95, 215, 0),
        (95, 215, 95),
        (95, 215, 135),
        (95, 215, 175),
        (95, 215, 215),
        (95, 215, 255),
        (95, 255, 0),
        (95, 255, 95),
        (95, 255, 135),
        (95, 255, 175),
        (95, 255, 215),
        (95, 255, 255),
        (135, 0, 0),
        (135, 0, 95),
        (135, 0, 135),
        (135, 0, 175),
        (135, 0, 215),
        (135, 0, 255),
        (135, 95, 0),
        (135, 95, 95),
        (135, 95, 135),
        (135, 95, 175),
        (135, 95, 215),
        (135, 95, 255),
        (135, 135, 0),
        (135, 135, 95),
        (135, 135, 135),
        (135, 135, 175),
        (135, 135, 215),
        (135, 135, 255),
        (135, 175, 0),
        (135, 175, 95),
        (135, 175, 135),
        (135, 175, 175),
        (135, 175, 215),
        (135, 175, 255),
        (135, 215, 0),
        (135, 215, 95),
        (135, 215, 135),
        (135, 215, 175),
        (135, 215, 215),
        (135, 215, 255),
        (135, 255, 0),
        (135, 255, 95),
        (135, 255, 135),
        (135, 255, 175),
        (135, 255, 215),
        (135, 255, 255),
        (175, 0, 0),
        (175, 0, 95),
        (175, 0, 135),
        (175, 0, 175),
        (175, 0, 215),
        (175, 0, 255),
        (175, 95, 0),
        (175, 95, 95),
        (175, 95, 135),
        (175, 95, 175),
        (175, 95, 215),
        (175, 95, 255),
        (175, 135, 0),
        (175, 135, 95),
        (175, 135, 135),
        (175, 135, 175),
        (175, 135, 215),
        (175, 135, 255),
        (175, 175, 0),
        (175, 175, 95),
        (175, 175, 135),
        (175, 175, 175),
        (175, 175, 215),
        (175, 175, 255),
        (175, 215, 0),
        (175, 215, 95),
        (175, 215, 135),
        (175, 215, 175),
        (175, 215, 215),
        (175, 215, 255),
        (175, 255, 0),
        (175, 255, 95),
        (175, 255, 135),
        (175, 255, 175),
        (175, 255, 215),
        (175, 255, 255),
        (215, 0, 0),
        (215, 0, 95),
        (215, 0, 135),
        (215, 0, 175),
        (215, 0, 215),
        (215, 0, 255),
        (215, 95, 0),
        (215, 95, 95),
        (215, 95, 135),
        (215, 95, 175),
        (215, 95, 215),
        (215, 95, 255),
        (215, 135, 0),
        (215, 135, 95),
        (215, 135, 135),
        (215, 135, 175),
        (215, 135, 215),
        (215, 135, 255),
        (215, 175, 0),
        (215, 175, 95),
        (215, 175, 135),
        (215, 175, 175),
        (215, 175, 215),
        (215, 175, 255),
        (215, 215, 0),
        (215, 215, 95),
        (215, 215, 135),
        (215, 215, 175),
        (215, 215, 215),
        (215, 215, 255),
        (215, 255, 0),
        (215, 255, 95),
        (215, 255, 135),
        (215, 255, 175),
        (215, 255, 215),
        (215, 255, 255),
        (255, 0, 0),
        (255, 0, 95),
        (255, 0, 135),
        (255, 0, 175),
        (255, 0, 215),
        (255, 0, 255),
        (255, 95, 0),
        (255, 95, 95),
        (255, 95, 135),
        (255, 95, 175),
        (255, 95, 215),
        (255, 95, 255),
        (255, 135, 0),
        (255, 135, 95),
        (255, 135, 135),
        (255, 135, 175),
        (255, 135, 215),
        (255, 135, 255),
        (255, 175, 0),
        (255, 175, 95),
        (255, 175, 135),
        (255, 175, 175),
        (255, 175, 215),
        (255, 175, 255),
        (255, 215, 0),
        (255, 215, 95),
        (255, 215, 135),
        (255, 215, 175),
        (255, 215, 215),
        (255, 215, 255),
        (255, 255, 0),
        (255, 255, 95),
        (255, 255, 135),
        (255, 255, 175),
        (255, 255, 215),
        (255, 255, 255),
        (8, 8, 8),
        (18, 18, 18),
        (28, 28, 28),
        (38, 38, 38),
        (48, 48, 48),
        (58, 58, 58),
        (68, 68, 68),
        (78, 78, 78),
        (88, 88, 88),
        (98, 98, 98),
        (108, 108, 108),
        (118, 118, 118),
        (128, 128, 128),
        (138, 138, 138),
        (148, 148, 148),
        (158, 158, 158),
        (168, 168, 168),
        (178, 178, 178),
        (188, 188, 188),
        (198, 198, 198),
        (208, 208, 208),
        (218, 218, 218),
        (228, 228, 228),
        (238, 238, 238),
    ]
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\cache_storage.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: CacheStorage (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import storage


class CacheId(str):
    '''
    Unique identifier of the Cache object.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> CacheId:
        return cls(json)

    def __repr__(self):
        return 'CacheId({})'.format(super().__repr__())


class CachedResponseType(enum.Enum):
    '''
    type of HTTP response cached
    '''
    BASIC = "basic"
    CORS = "cors"
    DEFAULT = "default"
    ERROR = "error"
    OPAQUE_RESPONSE = "opaqueResponse"
    OPAQUE_REDIRECT = "opaqueRedirect"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class DataEntry:
    '''
    Data entry.
    '''
    #: Request URL.
    request_url: str

    #: Request method.
    request_method: str

    #: Request headers
    request_headers: typing.List[Header]

    #: Number of seconds since epoch.
    response_time: float

    #: HTTP response status code.
    response_status: int

    #: HTTP response status text.
    response_status_text: str

    #: HTTP response type
    response_type: CachedResponseType

    #: Response headers
    response_headers: typing.List[Header]

    def to_json(self):
        json = dict()
        json['requestURL'] = self.request_url
        json['requestMethod'] = self.request_method
        json['requestHeaders'] = [i.to_json() for i in self.request_headers]
        json['responseTime'] = self.response_time
        json['responseStatus'] = self.response_status
        json['responseStatusText'] = self.response_status_text
        json['responseType'] = self.response_type.to_json()
        json['responseHeaders'] = [i.to_json() for i in self.response_headers]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            request_url=str(json['requestURL']),
            request_method=str(json['requestMethod']),
            request_headers=[Header.from_json(i) for i in json['requestHeaders']],
            response_time=float(json['responseTime']),
            response_status=int(json['responseStatus']),
            response_status_text=str(json['responseStatusText']),
            response_type=CachedResponseType.from_json(json['responseType']),
            response_headers=[Header.from_json(i) for i in json['responseHeaders']],
        )


@dataclass
class Cache:
    '''
    Cache identifier.
    '''
    #: An opaque unique id of the cache.
    cache_id: CacheId

    #: Security origin of the cache.
    security_origin: str

    #: Storage key of the cache.
    storage_key: str

    #: The name of the cache.
    cache_name: str

    #: Storage bucket of the cache.
    storage_bucket: typing.Optional[storage.StorageBucket] = None

    def to_json(self):
        json = dict()
        json['cacheId'] = self.cache_id.to_json()
        json['securityOrigin'] = self.security_origin
        json['storageKey'] = self.storage_key
        json['cacheName'] = self.cache_name
        if self.storage_bucket is not None:
            json['storageBucket'] = self.storage_bucket.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            cache_id=CacheId.from_json(json['cacheId']),
            security_origin=str(json['securityOrigin']),
            storage_key=str(json['storageKey']),
            cache_name=str(json['cacheName']),
            storage_bucket=storage.StorageBucket.from_json(json['storageBucket']) if 'storageBucket' in json else None,
        )


@dataclass
class Header:
    name: str

    value: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=str(json['value']),
        )


@dataclass
class CachedResponse:
    '''
    Cached response
    '''
    #: Entry content, base64-encoded.
    body: str

    def to_json(self):
        json = dict()
        json['body'] = self.body
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            body=str(json['body']),
        )


def delete_cache(
        cache_id: CacheId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes a cache.

    :param cache_id: Id of cache for deletion.
    '''
    params: T_JSON_DICT = dict()
    params['cacheId'] = cache_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'CacheStorage.deleteCache',
        'params': params,
    }
    json = yield cmd_dict


def delete_entry(
        cache_id: CacheId,
        request: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes a cache entry.

    :param cache_id: Id of cache where the entry will be deleted.
    :param request: URL spec of the request.
    '''
    params: T_JSON_DICT = dict()
    params['cacheId'] = cache_id.to_json()
    params['request'] = request
    cmd_dict: T_JSON_DICT = {
        'method': 'CacheStorage.deleteEntry',
        'params': params,
    }
    json = yield cmd_dict


def request_cache_names(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[Cache]]:
    '''
    Requests cache names.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :returns: Caches for the security origin.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'CacheStorage.requestCacheNames',
        'params': params,
    }
    json = yield cmd_dict
    return [Cache.from_json(i) for i in json['caches']]


def request_cached_response(
        cache_id: CacheId,
        request_url: str,
        request_headers: typing.List[Header]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,CachedResponse]:
    '''
    Fetches cache entry.

    :param cache_id: Id of cache that contains the entry.
    :param request_url: URL spec of the request.
    :param request_headers: headers of the request.
    :returns: Response read from the cache.
    '''
    params: T_JSON_DICT = dict()
    params['cacheId'] = cache_id.to_json()
    params['requestURL'] = request_url
    params['requestHeaders'] = [i.to_json() for i in request_headers]
    cmd_dict: T_JSON_DICT = {
        'method': 'CacheStorage.requestCachedResponse',
        'params': params,
    }
    json = yield cmd_dict
    return CachedResponse.from_json(json['response'])


def request_entries(
        cache_id: CacheId,
        skip_count: typing.Optional[int] = None,
        page_size: typing.Optional[int] = None,
        path_filter: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[DataEntry], float]]:
    '''
    Requests data from cache.

    :param cache_id: ID of cache to get entries from.
    :param skip_count: *(Optional)* Number of records to skip.
    :param page_size: *(Optional)* Number of records to fetch.
    :param path_filter: *(Optional)* If present, only return the entries containing this substring in the path
    :returns: A tuple with the following items:

        0. **cacheDataEntries** - Array of object store data entries.
        1. **returnCount** - Count of returned entries from this storage. If pathFilter is empty, it is the count of all entries from this storage.
    '''
    params: T_JSON_DICT = dict()
    params['cacheId'] = cache_id.to_json()
    if skip_count is not None:
        params['skipCount'] = skip_count
    if page_size is not None:
        params['pageSize'] = page_size
    if path_filter is not None:
        params['pathFilter'] = path_filter
    cmd_dict: T_JSON_DICT = {
        'method': 'CacheStorage.requestEntries',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [DataEntry.from_json(i) for i in json['cacheDataEntries']],
        float(json['returnCount'])
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\cache_storage.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: CacheStorage (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import storage


class CacheId(str):
    '''
    Unique identifier of the Cache object.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> CacheId:
        return cls(json)

    def __repr__(self):
        return 'CacheId({})'.format(super().__repr__())


class CachedResponseType(enum.Enum):
    '''
    type of HTTP response cached
    '''
    BASIC = "basic"
    CORS = "cors"
    DEFAULT = "default"
    ERROR = "error"
    OPAQUE_RESPONSE = "opaqueResponse"
    OPAQUE_REDIRECT = "opaqueRedirect"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class DataEntry:
    '''
    Data entry.
    '''
    #: Request URL.
    request_url: str

    #: Request method.
    request_method: str

    #: Request headers
    request_headers: typing.List[Header]

    #: Number of seconds since epoch.
    response_time: float

    #: HTTP response status code.
    response_status: int

    #: HTTP response status text.
    response_status_text: str

    #: HTTP response type
    response_type: CachedResponseType

    #: Response headers
    response_headers: typing.List[Header]

    def to_json(self):
        json = dict()
        json['requestURL'] = self.request_url
        json['requestMethod'] = self.request_method
        json['requestHeaders'] = [i.to_json() for i in self.request_headers]
        json['responseTime'] = self.response_time
        json['responseStatus'] = self.response_status
        json['responseStatusText'] = self.response_status_text
        json['responseType'] = self.response_type.to_json()
        json['responseHeaders'] = [i.to_json() for i in self.response_headers]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            request_url=str(json['requestURL']),
            request_method=str(json['requestMethod']),
            request_headers=[Header.from_json(i) for i in json['requestHeaders']],
            response_time=float(json['responseTime']),
            response_status=int(json['responseStatus']),
            response_status_text=str(json['responseStatusText']),
            response_type=CachedResponseType.from_json(json['responseType']),
            response_headers=[Header.from_json(i) for i in json['responseHeaders']],
        )


@dataclass
class Cache:
    '''
    Cache identifier.
    '''
    #: An opaque unique id of the cache.
    cache_id: CacheId

    #: Security origin of the cache.
    security_origin: str

    #: Storage key of the cache.
    storage_key: str

    #: The name of the cache.
    cache_name: str

    #: Storage bucket of the cache.
    storage_bucket: typing.Optional[storage.StorageBucket] = None

    def to_json(self):
        json = dict()
        json['cacheId'] = self.cache_id.to_json()
        json['securityOrigin'] = self.security_origin
        json['storageKey'] = self.storage_key
        json['cacheName'] = self.cache_name
        if self.storage_bucket is not None:
            json['storageBucket'] = self.storage_bucket.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            cache_id=CacheId.from_json(json['cacheId']),
            security_origin=str(json['securityOrigin']),
            storage_key=str(json['storageKey']),
            cache_name=str(json['cacheName']),
            storage_bucket=storage.StorageBucket.from_json(json['storageBucket']) if 'storageBucket' in json else None,
        )


@dataclass
class Header:
    name: str

    value: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=str(json['value']),
        )


@dataclass
class CachedResponse:
    '''
    Cached response
    '''
    #: Entry content, base64-encoded.
    body: str

    def to_json(self):
        json = dict()
        json['body'] = self.body
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            body=str(json['body']),
        )


def delete_cache(
        cache_id: CacheId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes a cache.

    :param cache_id: Id of cache for deletion.
    '''
    params: T_JSON_DICT = dict()
    params['cacheId'] = cache_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'CacheStorage.deleteCache',
        'params': params,
    }
    json = yield cmd_dict


def delete_entry(
        cache_id: CacheId,
        request: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes a cache entry.

    :param cache_id: Id of cache where the entry will be deleted.
    :param request: URL spec of the request.
    '''
    params: T_JSON_DICT = dict()
    params['cacheId'] = cache_id.to_json()
    params['request'] = request
    cmd_dict: T_JSON_DICT = {
        'method': 'CacheStorage.deleteEntry',
        'params': params,
    }
    json = yield cmd_dict


def request_cache_names(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[Cache]]:
    '''
    Requests cache names.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :returns: Caches for the security origin.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'CacheStorage.requestCacheNames',
        'params': params,
    }
    json = yield cmd_dict
    return [Cache.from_json(i) for i in json['caches']]


def request_cached_response(
        cache_id: CacheId,
        request_url: str,
        request_headers: typing.List[Header]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,CachedResponse]:
    '''
    Fetches cache entry.

    :param cache_id: Id of cache that contains the entry.
    :param request_url: URL spec of the request.
    :param request_headers: headers of the request.
    :returns: Response read from the cache.
    '''
    params: T_JSON_DICT = dict()
    params['cacheId'] = cache_id.to_json()
    params['requestURL'] = request_url
    params['requestHeaders'] = [i.to_json() for i in request_headers]
    cmd_dict: T_JSON_DICT = {
        'method': 'CacheStorage.requestCachedResponse',
        'params': params,
    }
    json = yield cmd_dict
    return CachedResponse.from_json(json['response'])


def request_entries(
        cache_id: CacheId,
        skip_count: typing.Optional[int] = None,
        page_size: typing.Optional[int] = None,
        path_filter: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[DataEntry], float]]:
    '''
    Requests data from cache.

    :param cache_id: ID of cache to get entries from.
    :param skip_count: *(Optional)* Number of records to skip.
    :param page_size: *(Optional)* Number of records to fetch.
    :param path_filter: *(Optional)* If present, only return the entries containing this substring in the path
    :returns: A tuple with the following items:

        0. **cacheDataEntries** - Array of object store data entries.
        1. **returnCount** - Count of returned entries from this storage. If pathFilter is empty, it is the count of all entries from this storage.
    '''
    params: T_JSON_DICT = dict()
    params['cacheId'] = cache_id.to_json()
    if skip_count is not None:
        params['skipCount'] = skip_count
    if page_size is not None:
        params['pageSize'] = page_size
    if path_filter is not None:
        params['pathFilter'] = path_filter
    cmd_dict: T_JSON_DICT = {
        'method': 'CacheStorage.requestEntries',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [DataEntry.from_json(i) for i in json['cacheDataEntries']],
        float(json['returnCount'])
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\cache_storage.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: CacheStorage (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import storage


class CacheId(str):
    '''
    Unique identifier of the Cache object.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> CacheId:
        return cls(json)

    def __repr__(self):
        return 'CacheId({})'.format(super().__repr__())


class CachedResponseType(enum.Enum):
    '''
    type of HTTP response cached
    '''
    BASIC = "basic"
    CORS = "cors"
    DEFAULT = "default"
    ERROR = "error"
    OPAQUE_RESPONSE = "opaqueResponse"
    OPAQUE_REDIRECT = "opaqueRedirect"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class DataEntry:
    '''
    Data entry.
    '''
    #: Request URL.
    request_url: str

    #: Request method.
    request_method: str

    #: Request headers
    request_headers: typing.List[Header]

    #: Number of seconds since epoch.
    response_time: float

    #: HTTP response status code.
    response_status: int

    #: HTTP response status text.
    response_status_text: str

    #: HTTP response type
    response_type: CachedResponseType

    #: Response headers
    response_headers: typing.List[Header]

    def to_json(self):
        json = dict()
        json['requestURL'] = self.request_url
        json['requestMethod'] = self.request_method
        json['requestHeaders'] = [i.to_json() for i in self.request_headers]
        json['responseTime'] = self.response_time
        json['responseStatus'] = self.response_status
        json['responseStatusText'] = self.response_status_text
        json['responseType'] = self.response_type.to_json()
        json['responseHeaders'] = [i.to_json() for i in self.response_headers]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            request_url=str(json['requestURL']),
            request_method=str(json['requestMethod']),
            request_headers=[Header.from_json(i) for i in json['requestHeaders']],
            response_time=float(json['responseTime']),
            response_status=int(json['responseStatus']),
            response_status_text=str(json['responseStatusText']),
            response_type=CachedResponseType.from_json(json['responseType']),
            response_headers=[Header.from_json(i) for i in json['responseHeaders']],
        )


@dataclass
class Cache:
    '''
    Cache identifier.
    '''
    #: An opaque unique id of the cache.
    cache_id: CacheId

    #: Security origin of the cache.
    security_origin: str

    #: Storage key of the cache.
    storage_key: str

    #: The name of the cache.
    cache_name: str

    #: Storage bucket of the cache.
    storage_bucket: typing.Optional[storage.StorageBucket] = None

    def to_json(self):
        json = dict()
        json['cacheId'] = self.cache_id.to_json()
        json['securityOrigin'] = self.security_origin
        json['storageKey'] = self.storage_key
        json['cacheName'] = self.cache_name
        if self.storage_bucket is not None:
            json['storageBucket'] = self.storage_bucket.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            cache_id=CacheId.from_json(json['cacheId']),
            security_origin=str(json['securityOrigin']),
            storage_key=str(json['storageKey']),
            cache_name=str(json['cacheName']),
            storage_bucket=storage.StorageBucket.from_json(json['storageBucket']) if 'storageBucket' in json else None,
        )


@dataclass
class Header:
    name: str

    value: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=str(json['value']),
        )


@dataclass
class CachedResponse:
    '''
    Cached response
    '''
    #: Entry content, base64-encoded.
    body: str

    def to_json(self):
        json = dict()
        json['body'] = self.body
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            body=str(json['body']),
        )


def delete_cache(
        cache_id: CacheId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes a cache.

    :param cache_id: Id of cache for deletion.
    '''
    params: T_JSON_DICT = dict()
    params['cacheId'] = cache_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'CacheStorage.deleteCache',
        'params': params,
    }
    json = yield cmd_dict


def delete_entry(
        cache_id: CacheId,
        request: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes a cache entry.

    :param cache_id: Id of cache where the entry will be deleted.
    :param request: URL spec of the request.
    '''
    params: T_JSON_DICT = dict()
    params['cacheId'] = cache_id.to_json()
    params['request'] = request
    cmd_dict: T_JSON_DICT = {
        'method': 'CacheStorage.deleteEntry',
        'params': params,
    }
    json = yield cmd_dict


def request_cache_names(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[Cache]]:
    '''
    Requests cache names.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :returns: Caches for the security origin.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'CacheStorage.requestCacheNames',
        'params': params,
    }
    json = yield cmd_dict
    return [Cache.from_json(i) for i in json['caches']]


def request_cached_response(
        cache_id: CacheId,
        request_url: str,
        request_headers: typing.List[Header]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,CachedResponse]:
    '''
    Fetches cache entry.

    :param cache_id: Id of cache that contains the entry.
    :param request_url: URL spec of the request.
    :param request_headers: headers of the request.
    :returns: Response read from the cache.
    '''
    params: T_JSON_DICT = dict()
    params['cacheId'] = cache_id.to_json()
    params['requestURL'] = request_url
    params['requestHeaders'] = [i.to_json() for i in request_headers]
    cmd_dict: T_JSON_DICT = {
        'method': 'CacheStorage.requestCachedResponse',
        'params': params,
    }
    json = yield cmd_dict
    return CachedResponse.from_json(json['response'])


def request_entries(
        cache_id: CacheId,
        skip_count: typing.Optional[int] = None,
        page_size: typing.Optional[int] = None,
        path_filter: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[DataEntry], float]]:
    '''
    Requests data from cache.

    :param cache_id: ID of cache to get entries from.
    :param skip_count: *(Optional)* Number of records to skip.
    :param page_size: *(Optional)* Number of records to fetch.
    :param path_filter: *(Optional)* If present, only return the entries containing this substring in the path
    :returns: A tuple with the following items:

        0. **cacheDataEntries** - Array of object store data entries.
        1. **returnCount** - Count of returned entries from this storage. If pathFilter is empty, it is the count of all entries from this storage.
    '''
    params: T_JSON_DICT = dict()
    params['cacheId'] = cache_id.to_json()
    if skip_count is not None:
        params['skipCount'] = skip_count
    if page_size is not None:
        params['pageSize'] = page_size
    if path_filter is not None:
        params['pathFilter'] = path_filter
    cmd_dict: T_JSON_DICT = {
        'method': 'CacheStorage.requestEntries',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [DataEntry.from_json(i) for i in json['cacheDataEntries']],
        float(json['returnCount'])
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\click\exceptions.py ===
from __future__ import annotations

import collections.abc as cabc
import typing as t
from gettext import gettext as _
from gettext import ngettext

from ._compat import get_text_stderr
from .globals import resolve_color_default
from .utils import echo
from .utils import format_filename

if t.TYPE_CHECKING:
    from .core import Command
    from .core import Context
    from .core import Parameter


def _join_param_hints(param_hint: cabc.Sequence[str] | str | None) -> str | None:
    if param_hint is not None and not isinstance(param_hint, str):
        return " / ".join(repr(x) for x in param_hint)

    return param_hint


class ClickException(Exception):
    """An exception that Click can handle and show to the user."""

    #: The exit code for this exception.
    exit_code = 1

    def __init__(self, message: str) -> None:
        super().__init__(message)
        # The context will be removed by the time we print the message, so cache
        # the color settings here to be used later on (in `show`)
        self.show_color: bool | None = resolve_color_default()
        self.message = message

    def format_message(self) -> str:
        return self.message

    def __str__(self) -> str:
        return self.message

    def show(self, file: t.IO[t.Any] | None = None) -> None:
        if file is None:
            file = get_text_stderr()

        echo(
            _("Error: {message}").format(message=self.format_message()),
            file=file,
            color=self.show_color,
        )


class UsageError(ClickException):
    """An internal exception that signals a usage error.  This typically
    aborts any further handling.

    :param message: the error message to display.
    :param ctx: optionally the context that caused this error.  Click will
                fill in the context automatically in some situations.
    """

    exit_code = 2

    def __init__(self, message: str, ctx: Context | None = None) -> None:
        super().__init__(message)
        self.ctx = ctx
        self.cmd: Command | None = self.ctx.command if self.ctx else None

    def show(self, file: t.IO[t.Any] | None = None) -> None:
        if file is None:
            file = get_text_stderr()
        color = None
        hint = ""
        if (
            self.ctx is not None
            and self.ctx.command.get_help_option(self.ctx) is not None
        ):
            hint = _("Try '{command} {option}' for help.").format(
                command=self.ctx.command_path, option=self.ctx.help_option_names[0]
            )
            hint = f"{hint}\n"
        if self.ctx is not None:
            color = self.ctx.color
            echo(f"{self.ctx.get_usage()}\n{hint}", file=file, color=color)
        echo(
            _("Error: {message}").format(message=self.format_message()),
            file=file,
            color=color,
        )


class BadParameter(UsageError):
    """An exception that formats out a standardized error message for a
    bad parameter.  This is useful when thrown from a callback or type as
    Click will attach contextual information to it (for instance, which
    parameter it is).

    .. versionadded:: 2.0

    :param param: the parameter object that caused this error.  This can
                  be left out, and Click will attach this info itself
                  if possible.
    :param param_hint: a string that shows up as parameter name.  This
                       can be used as alternative to `param` in cases
                       where custom validation should happen.  If it is
                       a string it's used as such, if it's a list then
                       each item is quoted and separated.
    """

    def __init__(
        self,
        message: str,
        ctx: Context | None = None,
        param: Parameter | None = None,
        param_hint: str | None = None,
    ) -> None:
        super().__init__(message, ctx)
        self.param = param
        self.param_hint = param_hint

    def format_message(self) -> str:
        if self.param_hint is not None:
            param_hint = self.param_hint
        elif self.param is not None:
            param_hint = self.param.get_error_hint(self.ctx)  # type: ignore
        else:
            return _("Invalid value: {message}").format(message=self.message)

        return _("Invalid value for {param_hint}: {message}").format(
            param_hint=_join_param_hints(param_hint), message=self.message
        )


class MissingParameter(BadParameter):
    """Raised if click required an option or argument but it was not
    provided when invoking the script.

    .. versionadded:: 4.0

    :param param_type: a string that indicates the type of the parameter.
                       The default is to inherit the parameter type from
                       the given `param`.  Valid values are ``'parameter'``,
                       ``'option'`` or ``'argument'``.
    """

    def __init__(
        self,
        message: str | None = None,
        ctx: Context | None = None,
        param: Parameter | None = None,
        param_hint: str | None = None,
        param_type: str | None = None,
    ) -> None:
        super().__init__(message or "", ctx, param, param_hint)
        self.param_type = param_type

    def format_message(self) -> str:
        if self.param_hint is not None:
            param_hint: str | None = self.param_hint
        elif self.param is not None:
            param_hint = self.param.get_error_hint(self.ctx)  # type: ignore
        else:
            param_hint = None

        param_hint = _join_param_hints(param_hint)
        param_hint = f" {param_hint}" if param_hint else ""

        param_type = self.param_type
        if param_type is None and self.param is not None:
            param_type = self.param.param_type_name

        msg = self.message
        if self.param is not None:
            msg_extra = self.param.type.get_missing_message(
                param=self.param, ctx=self.ctx
            )
            if msg_extra:
                if msg:
                    msg += f". {msg_extra}"
                else:
                    msg = msg_extra

        msg = f" {msg}" if msg else ""

        # Translate param_type for known types.
        if param_type == "argument":
            missing = _("Missing argument")
        elif param_type == "option":
            missing = _("Missing option")
        elif param_type == "parameter":
            missing = _("Missing parameter")
        else:
            missing = _("Missing {param_type}").format(param_type=param_type)

        return f"{missing}{param_hint}.{msg}"

    def __str__(self) -> str:
        if not self.message:
            param_name = self.param.name if self.param else None
            return _("Missing parameter: {param_name}").format(param_name=param_name)
        else:
            return self.message


class NoSuchOption(UsageError):
    """Raised if click attempted to handle an option that does not
    exist.

    .. versionadded:: 4.0
    """

    def __init__(
        self,
        option_name: str,
        message: str | None = None,
        possibilities: cabc.Sequence[str] | None = None,
        ctx: Context | None = None,
    ) -> None:
        if message is None:
            message = _("No such option: {name}").format(name=option_name)

        super().__init__(message, ctx)
        self.option_name = option_name
        self.possibilities = possibilities

    def format_message(self) -> str:
        if not self.possibilities:
            return self.message

        possibility_str = ", ".join(sorted(self.possibilities))
        suggest = ngettext(
            "Did you mean {possibility}?",
            "(Possible options: {possibilities})",
            len(self.possibilities),
        ).format(possibility=possibility_str, possibilities=possibility_str)
        return f"{self.message} {suggest}"


class BadOptionUsage(UsageError):
    """Raised if an option is generally supplied but the use of the option
    was incorrect.  This is for instance raised if the number of arguments
    for an option is not correct.

    .. versionadded:: 4.0

    :param option_name: the name of the option being used incorrectly.
    """

    def __init__(
        self, option_name: str, message: str, ctx: Context | None = None
    ) -> None:
        super().__init__(message, ctx)
        self.option_name = option_name


class BadArgumentUsage(UsageError):
    """Raised if an argument is generally supplied but the use of the argument
    was incorrect.  This is for instance raised if the number of values
    for an argument is not correct.

    .. versionadded:: 6.0
    """


class NoArgsIsHelpError(UsageError):
    def __init__(self, ctx: Context) -> None:
        self.ctx: Context
        super().__init__(ctx.get_help(), ctx=ctx)

    def show(self, file: t.IO[t.Any] | None = None) -> None:
        echo(self.format_message(), file=file, err=True, color=self.ctx.color)


class FileError(ClickException):
    """Raised if a file cannot be opened."""

    def __init__(self, filename: str, hint: str | None = None) -> None:
        if hint is None:
            hint = _("unknown error")

        super().__init__(hint)
        self.ui_filename: str = format_filename(filename)
        self.filename = filename

    def format_message(self) -> str:
        return _("Could not open file {filename!r}: {message}").format(
            filename=self.ui_filename, message=self.message
        )


class Abort(RuntimeError):
    """An internal signalling exception that signals Click to abort."""


class Exit(RuntimeError):
    """An exception that indicates that the application should exit with some
    status code.

    :param code: the status code to exit with.
    """

    __slots__ = ("exit_code",)

    def __init__(self, code: int = 0) -> None:
        self.exit_code: int = code

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\logic\algorithms\dpll.py ===
"""Implementation of DPLL algorithm

Further improvements: eliminate calls to pl_true, implement branching rules,
efficient unit propagation.

References:
  - https://en.wikipedia.org/wiki/DPLL_algorithm
  - https://www.researchgate.net/publication/242384772_Implementations_of_the_DPLL_Algorithm
"""

from sympy.core.sorting import default_sort_key
from sympy.logic.boolalg import Or, Not, conjuncts, disjuncts, to_cnf, \
    to_int_repr, _find_predicates
from sympy.assumptions.cnf import CNF
from sympy.logic.inference import pl_true, literal_symbol


def dpll_satisfiable(expr):
    """
    Check satisfiability of a propositional sentence.
    It returns a model rather than True when it succeeds

    >>> from sympy.abc import A, B
    >>> from sympy.logic.algorithms.dpll import dpll_satisfiable
    >>> dpll_satisfiable(A & ~B)
    {A: True, B: False}
    >>> dpll_satisfiable(A & ~A)
    False

    """
    if not isinstance(expr, CNF):
        clauses = conjuncts(to_cnf(expr))
    else:
        clauses = expr.clauses
    if False in clauses:
        return False
    symbols = sorted(_find_predicates(expr), key=default_sort_key)
    symbols_int_repr = set(range(1, len(symbols) + 1))
    clauses_int_repr = to_int_repr(clauses, symbols)
    result = dpll_int_repr(clauses_int_repr, symbols_int_repr, {})
    if not result:
        return result
    output = {}
    for key in result:
        output.update({symbols[key - 1]: result[key]})
    return output


def dpll(clauses, symbols, model):
    """
    Compute satisfiability in a partial model.
    Clauses is an array of conjuncts.

    >>> from sympy.abc import A, B, D
    >>> from sympy.logic.algorithms.dpll import dpll
    >>> dpll([A, B, D], [A, B], {D: False})
    False

    """
    # compute DP kernel
    P, value = find_unit_clause(clauses, model)
    while P:
        model.update({P: value})
        symbols.remove(P)
        if not value:
            P = ~P
        clauses = unit_propagate(clauses, P)
        P, value = find_unit_clause(clauses, model)
    P, value = find_pure_symbol(symbols, clauses)
    while P:
        model.update({P: value})
        symbols.remove(P)
        if not value:
            P = ~P
        clauses = unit_propagate(clauses, P)
        P, value = find_pure_symbol(symbols, clauses)
    # end DP kernel
    unknown_clauses = []
    for c in clauses:
        val = pl_true(c, model)
        if val is False:
            return False
        if val is not True:
            unknown_clauses.append(c)
    if not unknown_clauses:
        return model
    if not clauses:
        return model
    P = symbols.pop()
    model_copy = model.copy()
    model.update({P: True})
    model_copy.update({P: False})
    symbols_copy = symbols[:]
    return (dpll(unit_propagate(unknown_clauses, P), symbols, model) or
            dpll(unit_propagate(unknown_clauses, Not(P)), symbols_copy, model_copy))


def dpll_int_repr(clauses, symbols, model):
    """
    Compute satisfiability in a partial model.
    Arguments are expected to be in integer representation

    >>> from sympy.logic.algorithms.dpll import dpll_int_repr
    >>> dpll_int_repr([{1}, {2}, {3}], {1, 2}, {3: False})
    False

    """
    # compute DP kernel
    P, value = find_unit_clause_int_repr(clauses, model)
    while P:
        model.update({P: value})
        symbols.remove(P)
        if not value:
            P = -P
        clauses = unit_propagate_int_repr(clauses, P)
        P, value = find_unit_clause_int_repr(clauses, model)
    P, value = find_pure_symbol_int_repr(symbols, clauses)
    while P:
        model.update({P: value})
        symbols.remove(P)
        if not value:
            P = -P
        clauses = unit_propagate_int_repr(clauses, P)
        P, value = find_pure_symbol_int_repr(symbols, clauses)
    # end DP kernel
    unknown_clauses = []
    for c in clauses:
        val = pl_true_int_repr(c, model)
        if val is False:
            return False
        if val is not True:
            unknown_clauses.append(c)
    if not unknown_clauses:
        return model
    P = symbols.pop()
    model_copy = model.copy()
    model.update({P: True})
    model_copy.update({P: False})
    symbols_copy = symbols.copy()
    return (dpll_int_repr(unit_propagate_int_repr(unknown_clauses, P), symbols, model) or
            dpll_int_repr(unit_propagate_int_repr(unknown_clauses, -P), symbols_copy, model_copy))

### helper methods for DPLL


def pl_true_int_repr(clause, model={}):
    """
    Lightweight version of pl_true.
    Argument clause represents the set of args of an Or clause. This is used
    inside dpll_int_repr, it is not meant to be used directly.

    >>> from sympy.logic.algorithms.dpll import pl_true_int_repr
    >>> pl_true_int_repr({1, 2}, {1: False})
    >>> pl_true_int_repr({1, 2}, {1: False, 2: False})
    False

    """
    result = False
    for lit in clause:
        if lit < 0:
            p = model.get(-lit)
            if p is not None:
                p = not p
        else:
            p = model.get(lit)
        if p is True:
            return True
        elif p is None:
            result = None
    return result


def unit_propagate(clauses, symbol):
    """
    Returns an equivalent set of clauses
    If a set of clauses contains the unit clause l, the other clauses are
    simplified by the application of the two following rules:

      1. every clause containing l is removed
      2. in every clause that contains ~l this literal is deleted

    Arguments are expected to be in CNF.

    >>> from sympy.abc import A, B, D
    >>> from sympy.logic.algorithms.dpll import unit_propagate
    >>> unit_propagate([A | B, D | ~B, B], B)
    [D, B]

    """
    output = []
    for c in clauses:
        if c.func != Or:
            output.append(c)
            continue
        for arg in c.args:
            if arg == ~symbol:
                output.append(Or(*[x for x in c.args if x != ~symbol]))
                break
            if arg == symbol:
                break
        else:
            output.append(c)
    return output


def unit_propagate_int_repr(clauses, s):
    """
    Same as unit_propagate, but arguments are expected to be in integer
    representation

    >>> from sympy.logic.algorithms.dpll import unit_propagate_int_repr
    >>> unit_propagate_int_repr([{1, 2}, {3, -2}, {2}], 2)
    [{3}]

    """
    negated = {-s}
    return [clause - negated for clause in clauses if s not in clause]


def find_pure_symbol(symbols, unknown_clauses):
    """
    Find a symbol and its value if it appears only as a positive literal
    (or only as a negative) in clauses.

    >>> from sympy.abc import A, B, D
    >>> from sympy.logic.algorithms.dpll import find_pure_symbol
    >>> find_pure_symbol([A, B, D], [A|~B,~B|~D,D|A])
    (A, True)

    """
    for sym in symbols:
        found_pos, found_neg = False, False
        for c in unknown_clauses:
            if not found_pos and sym in disjuncts(c):
                found_pos = True
            if not found_neg and Not(sym) in disjuncts(c):
                found_neg = True
        if found_pos != found_neg:
            return sym, found_pos
    return None, None


def find_pure_symbol_int_repr(symbols, unknown_clauses):
    """
    Same as find_pure_symbol, but arguments are expected
    to be in integer representation

    >>> from sympy.logic.algorithms.dpll import find_pure_symbol_int_repr
    >>> find_pure_symbol_int_repr({1,2,3},
    ...     [{1, -2}, {-2, -3}, {3, 1}])
    (1, True)

    """
    all_symbols = set().union(*unknown_clauses)
    found_pos = all_symbols.intersection(symbols)
    found_neg = all_symbols.intersection([-s for s in symbols])
    for p in found_pos:
        if -p not in found_neg:
            return p, True
    for p in found_neg:
        if -p not in found_pos:
            return -p, False
    return None, None


def find_unit_clause(clauses, model):
    """
    A unit clause has only 1 variable that is not bound in the model.

    >>> from sympy.abc import A, B, D
    >>> from sympy.logic.algorithms.dpll import find_unit_clause
    >>> find_unit_clause([A | B | D, B | ~D, A | ~B], {A:True})
    (B, False)

    """
    for clause in clauses:
        num_not_in_model = 0
        for literal in disjuncts(clause):
            sym = literal_symbol(literal)
            if sym not in model:
                num_not_in_model += 1
                P, value = sym, not isinstance(literal, Not)
        if num_not_in_model == 1:
            return P, value
    return None, None


def find_unit_clause_int_repr(clauses, model):
    """
    Same as find_unit_clause, but arguments are expected to be in
    integer representation.

    >>> from sympy.logic.algorithms.dpll import find_unit_clause_int_repr
    >>> find_unit_clause_int_repr([{1, 2, 3},
    ...     {2, -3}, {1, -2}], {1: True})
    (2, False)

    """
    bound = set(model) | {-sym for sym in model}
    for clause in clauses:
        unbound = clause - bound
        if len(unbound) == 1:
            p = unbound.pop()
            if p < 0:
                return -p, False
            else:
                return p, True
    return None, None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\symbolic_multivariate_probability.py ===
import itertools

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand as _expand
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.matrices.exceptions import ShapeError
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.stats.rv import RandomSymbol, is_random
from sympy.core.sympify import _sympify
from sympy.stats.symbolic_probability import Variance, Covariance, Expectation


class ExpectationMatrix(Expectation, MatrixExpr):
    """
    Expectation of a random matrix expression.

    Examples
    ========

    >>> from sympy.stats import ExpectationMatrix, Normal
    >>> from sympy.stats.rv import RandomMatrixSymbol
    >>> from sympy import symbols, MatrixSymbol, Matrix
    >>> k = symbols("k")
    >>> A, B = MatrixSymbol("A", k, k), MatrixSymbol("B", k, k)
    >>> X, Y = RandomMatrixSymbol("X", k, 1), RandomMatrixSymbol("Y", k, 1)
    >>> ExpectationMatrix(X)
    ExpectationMatrix(X)
    >>> ExpectationMatrix(A*X).shape
    (k, 1)

    To expand the expectation in its expression, use ``expand()``:

    >>> ExpectationMatrix(A*X + B*Y).expand()
    A*ExpectationMatrix(X) + B*ExpectationMatrix(Y)
    >>> ExpectationMatrix((X + Y)*(X - Y).T).expand()
    ExpectationMatrix(X*X.T) - ExpectationMatrix(X*Y.T) + ExpectationMatrix(Y*X.T) - ExpectationMatrix(Y*Y.T)

    To evaluate the ``ExpectationMatrix``, use ``doit()``:

    >>> N11, N12 = Normal('N11', 11, 1), Normal('N12', 12, 1)
    >>> N21, N22 = Normal('N21', 21, 1), Normal('N22', 22, 1)
    >>> M11, M12 = Normal('M11', 1, 1), Normal('M12', 2, 1)
    >>> M21, M22 = Normal('M21', 3, 1), Normal('M22', 4, 1)
    >>> x1 = Matrix([[N11, N12], [N21, N22]])
    >>> x2 = Matrix([[M11, M12], [M21, M22]])
    >>> ExpectationMatrix(x1 + x2).doit()
    Matrix([
    [12, 14],
    [24, 26]])

    """
    def __new__(cls, expr, condition=None):
        expr = _sympify(expr)
        if condition is None:
            if not is_random(expr):
                return expr
            obj = Expr.__new__(cls, expr)
        else:
            condition = _sympify(condition)
            obj = Expr.__new__(cls, expr, condition)

        obj._shape = expr.shape
        obj._condition = condition
        return obj

    @property
    def shape(self):
        return self._shape

    def expand(self, **hints):
        expr = self.args[0]
        condition = self._condition
        if not is_random(expr):
            return expr

        if isinstance(expr, Add):
            return Add.fromiter(Expectation(a, condition=condition).expand()
                    for a in expr.args)

        expand_expr = _expand(expr)
        if isinstance(expand_expr, Add):
            return Add.fromiter(Expectation(a, condition=condition).expand()
                    for a in expand_expr.args)

        elif isinstance(expr, (Mul, MatMul)):
            rv = []
            nonrv = []
            postnon = []

            for a in expr.args:
                if is_random(a):
                    if rv:
                        rv.extend(postnon)
                    else:
                        nonrv.extend(postnon)
                    postnon = []
                    rv.append(a)
                elif a.is_Matrix:
                    postnon.append(a)
                else:
                    nonrv.append(a)

            # In order to avoid infinite-looping (MatMul may call .doit() again),
            # do not rebuild
            if len(nonrv) == 0:
                return self
            return Mul.fromiter(nonrv)*Expectation(Mul.fromiter(rv),
                    condition=condition)*Mul.fromiter(postnon)

        return self

class VarianceMatrix(Variance, MatrixExpr):
    """
    Variance of a random matrix probability expression. Also known as
    Covariance matrix, auto-covariance matrix, dispersion matrix,
    or variance-covariance matrix.

    Examples
    ========

    >>> from sympy.stats import VarianceMatrix
    >>> from sympy.stats.rv import RandomMatrixSymbol
    >>> from sympy import symbols, MatrixSymbol
    >>> k = symbols("k")
    >>> A, B = MatrixSymbol("A", k, k), MatrixSymbol("B", k, k)
    >>> X, Y = RandomMatrixSymbol("X", k, 1), RandomMatrixSymbol("Y", k, 1)
    >>> VarianceMatrix(X)
    VarianceMatrix(X)
    >>> VarianceMatrix(X).shape
    (k, k)

    To expand the variance in its expression, use ``expand()``:

    >>> VarianceMatrix(A*X).expand()
    A*VarianceMatrix(X)*A.T
    >>> VarianceMatrix(A*X + B*Y).expand()
    2*A*CrossCovarianceMatrix(X, Y)*B.T + A*VarianceMatrix(X)*A.T + B*VarianceMatrix(Y)*B.T
    """
    def __new__(cls, arg, condition=None):
        arg = _sympify(arg)

        if 1 not in arg.shape:
            raise ShapeError("Expression is not a vector")

        shape = (arg.shape[0], arg.shape[0]) if arg.shape[1] == 1 else (arg.shape[1], arg.shape[1])

        if condition:
            obj = Expr.__new__(cls, arg, condition)
        else:
            obj = Expr.__new__(cls, arg)

        obj._shape = shape
        obj._condition = condition
        return obj

    @property
    def shape(self):
        return self._shape

    def expand(self, **hints):
        arg = self.args[0]
        condition = self._condition

        if not is_random(arg):
            return ZeroMatrix(*self.shape)

        if isinstance(arg, RandomSymbol):
            return self
        elif isinstance(arg, Add):
            rv = []
            for a in arg.args:
                if is_random(a):
                    rv.append(a)
            variances = Add(*(Variance(xv, condition).expand() for xv in rv))
            map_to_covar = lambda x: 2*Covariance(*x, condition=condition).expand()
            covariances = Add(*map(map_to_covar, itertools.combinations(rv, 2)))
            return variances + covariances
        elif isinstance(arg, (Mul, MatMul)):
            nonrv = []
            rv = []
            for a in arg.args:
                if is_random(a):
                    rv.append(a)
                else:
                    nonrv.append(a)
            if len(rv) == 0:
                return ZeroMatrix(*self.shape)
            # Avoid possible infinite loops with MatMul:
            if len(nonrv) == 0:
                return self
            # Variance of many multiple matrix products is not implemented:
            if len(rv) > 1:
                return self
            return Mul.fromiter(nonrv)*Variance(Mul.fromiter(rv),
                            condition)*(Mul.fromiter(nonrv)).transpose()

        # this expression contains a RandomSymbol somehow:
        return self

class CrossCovarianceMatrix(Covariance, MatrixExpr):
    """
    Covariance of a random matrix probability expression.

    Examples
    ========

    >>> from sympy.stats import CrossCovarianceMatrix
    >>> from sympy.stats.rv import RandomMatrixSymbol
    >>> from sympy import symbols, MatrixSymbol
    >>> k = symbols("k")
    >>> A, B = MatrixSymbol("A", k, k), MatrixSymbol("B", k, k)
    >>> C, D = MatrixSymbol("C", k, k), MatrixSymbol("D", k, k)
    >>> X, Y = RandomMatrixSymbol("X", k, 1), RandomMatrixSymbol("Y", k, 1)
    >>> Z, W = RandomMatrixSymbol("Z", k, 1), RandomMatrixSymbol("W", k, 1)
    >>> CrossCovarianceMatrix(X, Y)
    CrossCovarianceMatrix(X, Y)
    >>> CrossCovarianceMatrix(X, Y).shape
    (k, k)

    To expand the covariance in its expression, use ``expand()``:

    >>> CrossCovarianceMatrix(X + Y, Z).expand()
    CrossCovarianceMatrix(X, Z) + CrossCovarianceMatrix(Y, Z)
    >>> CrossCovarianceMatrix(A*X, Y).expand()
    A*CrossCovarianceMatrix(X, Y)
    >>> CrossCovarianceMatrix(A*X, B.T*Y).expand()
    A*CrossCovarianceMatrix(X, Y)*B
    >>> CrossCovarianceMatrix(A*X + B*Y, C.T*Z + D.T*W).expand()
    A*CrossCovarianceMatrix(X, W)*D + A*CrossCovarianceMatrix(X, Z)*C + B*CrossCovarianceMatrix(Y, W)*D + B*CrossCovarianceMatrix(Y, Z)*C

    """
    def __new__(cls, arg1, arg2, condition=None):
        arg1 = _sympify(arg1)
        arg2 = _sympify(arg2)

        if (1 not in arg1.shape) or (1 not in arg2.shape) or (arg1.shape[1] != arg2.shape[1]):
            raise ShapeError("Expression is not a vector")

        shape = (arg1.shape[0], arg2.shape[0]) if arg1.shape[1] == 1 and arg2.shape[1] == 1 \
                    else (1, 1)

        if condition:
            obj = Expr.__new__(cls, arg1, arg2, condition)
        else:
            obj = Expr.__new__(cls, arg1, arg2)

        obj._shape = shape
        obj._condition = condition
        return obj

    @property
    def shape(self):
        return self._shape

    def expand(self, **hints):
        arg1 = self.args[0]
        arg2 = self.args[1]
        condition = self._condition

        if arg1 == arg2:
            return VarianceMatrix(arg1, condition).expand()

        if not is_random(arg1) or not is_random(arg2):
            return ZeroMatrix(*self.shape)

        if isinstance(arg1, RandomSymbol) and isinstance(arg2, RandomSymbol):
            return CrossCovarianceMatrix(arg1, arg2, condition)

        coeff_rv_list1 = self._expand_single_argument(arg1.expand())
        coeff_rv_list2 = self._expand_single_argument(arg2.expand())

        addends = [a*CrossCovarianceMatrix(r1, r2, condition=condition)*b.transpose()
                   for (a, r1) in coeff_rv_list1 for (b, r2) in coeff_rv_list2]
        return Add.fromiter(addends)

    @classmethod
    def _expand_single_argument(cls, expr):
        # return (coefficient, random_symbol) pairs:
        if isinstance(expr, RandomSymbol):
            return [(S.One, expr)]
        elif isinstance(expr, Add):
            outval = []
            for a in expr.args:
                if isinstance(a, (Mul, MatMul)):
                    outval.append(cls._get_mul_nonrv_rv_tuple(a))
                elif is_random(a):
                    outval.append((S.One, a))

            return outval
        elif isinstance(expr, (Mul, MatMul)):
            return [cls._get_mul_nonrv_rv_tuple(expr)]
        elif is_random(expr):
            return [(S.One, expr)]

    @classmethod
    def _get_mul_nonrv_rv_tuple(cls, m):
        rv = []
        nonrv = []
        for a in m.args:
            if is_random(a):
                rv.append(a)
            else:
                nonrv.append(a)
        return (Mul.fromiter(nonrv), Mul.fromiter(rv))