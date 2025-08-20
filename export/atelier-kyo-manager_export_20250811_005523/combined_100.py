
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_sync.py ===
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

import attrs

import trio

from . import _core
from ._core import (
    Abort,
    ParkingLot,
    RaiseCancelT,
    add_parking_lot_breaker,
    enable_ki_protection,
    remove_parking_lot_breaker,
)
from ._util import final

if TYPE_CHECKING:
    from types import TracebackType

    from ._core import Task
    from ._core._parking_lot import ParkingLotStatistics


@attrs.frozen
class EventStatistics:
    """An object containing debugging information.

    Currently the following fields are defined:

    * ``tasks_waiting``: The number of tasks blocked on this event's
      :meth:`trio.Event.wait` method.

    """

    tasks_waiting: int


@final
@attrs.define(repr=False, eq=False)
class Event:
    """A waitable boolean value useful for inter-task synchronization,
    inspired by :class:`threading.Event`.

    An event object has an internal boolean flag, representing whether
    the event has happened yet. The flag is initially False, and the
    :meth:`wait` method waits until the flag is True. If the flag is
    already True, then :meth:`wait` returns immediately. (If the event has
    already happened, there's nothing to wait for.) The :meth:`set` method
    sets the flag to True, and wakes up any waiters.

    This behavior is useful because it helps avoid race conditions and
    lost wakeups: it doesn't matter whether :meth:`set` gets called just
    before or after :meth:`wait`. If you want a lower-level wakeup
    primitive that doesn't have this protection, consider :class:`Condition`
    or :class:`trio.lowlevel.ParkingLot`.

    .. note:: Unlike `threading.Event`, `trio.Event` has no
       `~threading.Event.clear` method. In Trio, once an `Event` has happened,
       it cannot un-happen. If you need to represent a series of events,
       consider creating a new `Event` object for each one (they're cheap!),
       or other synchronization methods like :ref:`channels <channels>` or
       `trio.lowlevel.ParkingLot`.

    """

    _tasks: set[Task] = attrs.field(factory=set, init=False)
    _flag: bool = attrs.field(default=False, init=False)

    def is_set(self) -> bool:
        """Return the current value of the internal flag."""
        return self._flag

    @enable_ki_protection
    def set(self) -> None:
        """Set the internal flag value to True, and wake any waiting tasks."""
        if not self._flag:
            self._flag = True
            for task in self._tasks:
                _core.reschedule(task)
            self._tasks.clear()

    async def wait(self) -> None:
        """Block until the internal flag value becomes True.

        If it's already True, then this method returns immediately.

        """
        if self._flag:
            await trio.lowlevel.checkpoint()
        else:
            task = _core.current_task()
            self._tasks.add(task)

            def abort_fn(_: RaiseCancelT) -> Abort:
                self._tasks.remove(task)
                return _core.Abort.SUCCEEDED

            await _core.wait_task_rescheduled(abort_fn)

    def statistics(self) -> EventStatistics:
        """Return an object containing debugging information.

        Currently the following fields are defined:

        * ``tasks_waiting``: The number of tasks blocked on this event's
          :meth:`wait` method.

        """
        return EventStatistics(tasks_waiting=len(self._tasks))


class _HasAcquireRelease(Protocol):
    """Only classes with acquire() and release() can use the mixin's implementations."""

    async def acquire(self) -> object: ...

    def release(self) -> object: ...


class AsyncContextManagerMixin:
    @enable_ki_protection
    async def __aenter__(self: _HasAcquireRelease) -> None:
        await self.acquire()

    @enable_ki_protection
    async def __aexit__(
        self: _HasAcquireRelease,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


@attrs.frozen
class CapacityLimiterStatistics:
    """An object containing debugging information.

    Currently the following fields are defined:

    * ``borrowed_tokens``: The number of tokens currently borrowed from
      the sack.
    * ``total_tokens``: The total number of tokens in the sack. Usually
      this will be larger than ``borrowed_tokens``, but it's possibly for
      it to be smaller if :attr:`trio.CapacityLimiter.total_tokens` was recently decreased.
    * ``borrowers``: A list of all tasks or other entities that currently
      hold a token.
    * ``tasks_waiting``: The number of tasks blocked on this
      :class:`CapacityLimiter`\'s :meth:`trio.CapacityLimiter.acquire` or
      :meth:`trio.CapacityLimiter.acquire_on_behalf_of` methods.

    """

    borrowed_tokens: int
    total_tokens: int | float
    borrowers: list[Task | object]
    tasks_waiting: int


# Can be a generic type with a default of Task if/when PEP 696 is released
# and implemented in type checkers. Making it fully generic would currently
# introduce a lot of unnecessary hassle.
@final
class CapacityLimiter(AsyncContextManagerMixin):
    """An object for controlling access to a resource with limited capacity.

    Sometimes you need to put a limit on how many tasks can do something at
    the same time. For example, you might want to use some threads to run
    multiple blocking I/O operations in parallel... but if you use too many
    threads at once, then your system can become overloaded and it'll actually
    make things slower. One popular solution is to impose a policy like "run
    up to 40 threads at the same time, but no more". But how do you implement
    a policy like this?

    That's what :class:`CapacityLimiter` is for. You can think of a
    :class:`CapacityLimiter` object as a sack that starts out holding some fixed
    number of tokens::

       limit = trio.CapacityLimiter(40)

    Then tasks can come along and borrow a token out of the sack::

       # Borrow a token:
       async with limit:
           # We are holding a token!
           await perform_expensive_operation()
       # Exiting the 'async with' block puts the token back into the sack

    And crucially, if you try to borrow a token but the sack is empty, then
    you have to wait for another task to finish what it's doing and put its
    token back first before you can take it and continue.

    Another way to think of it: a :class:`CapacityLimiter` is like a sofa with a
    fixed number of seats, and if they're all taken then you have to wait for
    someone to get up before you can sit down.

    By default, :func:`trio.to_thread.run_sync` uses a
    :class:`CapacityLimiter` to limit the number of threads running at once;
    see `trio.to_thread.current_default_thread_limiter` for details.

    If you're familiar with semaphores, then you can think of this as a
    restricted semaphore that's specialized for one common use case, with
    additional error checking. For a more traditional semaphore, see
    :class:`Semaphore`.

    .. note::

       Don't confuse this with the `"leaky bucket"
       <https://en.wikipedia.org/wiki/Leaky_bucket>`__ or `"token bucket"
       <https://en.wikipedia.org/wiki/Token_bucket>`__ algorithms used to
       limit bandwidth usage on networks. The basic idea of using tokens to
       track a resource limit is similar, but this is a very simple sack where
       tokens aren't automatically created or destroyed over time; they're
       just borrowed and then put back.

    """

    # total_tokens would ideally be int|Literal[math.inf] - but that's not valid typing
    def __init__(self, total_tokens: int | float) -> None:  # noqa: PYI041
        self._lot = ParkingLot()
        self._borrowers: set[Task | object] = set()
        # Maps tasks attempting to acquire -> borrower, to handle on-behalf-of
        self._pending_borrowers: dict[Task, Task | object] = {}
        # invoke the property setter for validation
        self.total_tokens: int | float = total_tokens
        assert self._total_tokens == total_tokens

    def __repr__(self) -> str:
        return f"<trio.CapacityLimiter at {id(self):#x}, {len(self._borrowers)}/{self._total_tokens} with {len(self._lot)} waiting>"

    @property
    def total_tokens(self) -> int | float:
        """The total capacity available.

        You can change :attr:`total_tokens` by assigning to this attribute. If
        you make it larger, then the appropriate number of waiting tasks will
        be woken immediately to take the new tokens. If you decrease
        total_tokens below the number of tasks that are currently using the
        resource, then all current tasks will be allowed to finish as normal,
        but no new tasks will be allowed in until the total number of tasks
        drops below the new total_tokens.

        """
        return self._total_tokens

    @total_tokens.setter
    def total_tokens(self, new_total_tokens: int | float) -> None:  # noqa: PYI041
        if not isinstance(new_total_tokens, int) and new_total_tokens != math.inf:
            raise TypeError("total_tokens must be an int or math.inf")
        if new_total_tokens < 1:
            raise ValueError("total_tokens must be >= 1")
        self._total_tokens = new_total_tokens
        self._wake_waiters()

    def _wake_waiters(self) -> None:
        available = self._total_tokens - len(self._borrowers)
        for woken in self._lot.unpark(count=available):
            self._borrowers.add(self._pending_borrowers.pop(woken))

    @property
    def borrowed_tokens(self) -> int:
        """The amount of capacity that's currently in use."""
        return len(self._borrowers)

    @property
    def available_tokens(self) -> int | float:
        """The amount of capacity that's available to use."""
        return self.total_tokens - self.borrowed_tokens

    @enable_ki_protection
    def acquire_nowait(self) -> None:
        """Borrow a token from the sack, without blocking.

        Raises:
          WouldBlock: if no tokens are available.
          RuntimeError: if the current task already holds one of this sack's
              tokens.

        """
        self.acquire_on_behalf_of_nowait(trio.lowlevel.current_task())

    @enable_ki_protection
    def acquire_on_behalf_of_nowait(self, borrower: Task | object) -> None:
        """Borrow a token from the sack on behalf of ``borrower``, without
        blocking.

        Args:
          borrower: A :class:`trio.lowlevel.Task` or arbitrary opaque object
             used to record who is borrowing this token. This is used by
             :func:`trio.to_thread.run_sync` to allow threads to "hold
             tokens", with the intention in the future of using it to `allow
             deadlock detection and other useful things
             <https://github.com/python-trio/trio/issues/182>`__

        Raises:
          WouldBlock: if no tokens are available.
          RuntimeError: if ``borrower`` already holds one of this sack's
              tokens.

        """
        if borrower in self._borrowers:
            raise RuntimeError(
                "this borrower is already holding one of this CapacityLimiter's tokens",
            )
        if len(self._borrowers) < self._total_tokens and not self._lot:
            self._borrowers.add(borrower)
        else:
            raise trio.WouldBlock

    @enable_ki_protection
    async def acquire(self) -> None:
        """Borrow a token from the sack, blocking if necessary.

        Raises:
          RuntimeError: if the current task already holds one of this sack's
              tokens.

        """
        await self.acquire_on_behalf_of(trio.lowlevel.current_task())

    @enable_ki_protection
    async def acquire_on_behalf_of(self, borrower: Task | object) -> None:
        """Borrow a token from the sack on behalf of ``borrower``, blocking if
        necessary.

        Args:
          borrower: A :class:`trio.lowlevel.Task` or arbitrary opaque object
             used to record who is borrowing this token; see
             :meth:`acquire_on_behalf_of_nowait` for details.

        Raises:
          RuntimeError: if ``borrower`` task already holds one of this sack's
             tokens.

        """
        await trio.lowlevel.checkpoint_if_cancelled()
        try:
            self.acquire_on_behalf_of_nowait(borrower)
        except trio.WouldBlock:
            task = trio.lowlevel.current_task()
            self._pending_borrowers[task] = borrower
            try:
                await self._lot.park()
            except trio.Cancelled:
                self._pending_borrowers.pop(task)
                raise
        else:
            await trio.lowlevel.cancel_shielded_checkpoint()

    @enable_ki_protection
    def release(self) -> None:
        """Put a token back into the sack.

        Raises:
          RuntimeError: if the current task has not acquired one of this
              sack's tokens.

        """
        self.release_on_behalf_of(trio.lowlevel.current_task())

    @enable_ki_protection
    def release_on_behalf_of(self, borrower: Task | object) -> None:
        """Put a token back into the sack on behalf of ``borrower``.

        Raises:
          RuntimeError: if the given borrower has not acquired one of this
              sack's tokens.

        """
        if borrower not in self._borrowers:
            raise RuntimeError(
                "this borrower isn't holding any of this CapacityLimiter's tokens",
            )
        self._borrowers.remove(borrower)
        self._wake_waiters()

    def statistics(self) -> CapacityLimiterStatistics:
        """Return an object containing debugging information.

        Currently the following fields are defined:

        * ``borrowed_tokens``: The number of tokens currently borrowed from
          the sack.
        * ``total_tokens``: The total number of tokens in the sack. Usually
          this will be larger than ``borrowed_tokens``, but it's possibly for
          it to be smaller if :attr:`total_tokens` was recently decreased.
        * ``borrowers``: A list of all tasks or other entities that currently
          hold a token.
        * ``tasks_waiting``: The number of tasks blocked on this
          :class:`CapacityLimiter`\'s :meth:`acquire` or
          :meth:`acquire_on_behalf_of` methods.

        """
        return CapacityLimiterStatistics(
            borrowed_tokens=len(self._borrowers),
            total_tokens=self._total_tokens,
            # Use a list instead of a frozenset just in case we start to allow
            # one borrower to hold multiple tokens in the future
            borrowers=list(self._borrowers),
            tasks_waiting=len(self._lot),
        )


@final
class Semaphore(AsyncContextManagerMixin):
    """A `semaphore <https://en.wikipedia.org/wiki/Semaphore_(programming)>`__.

    A semaphore holds an integer value, which can be incremented by
    calling :meth:`release` and decremented by calling :meth:`acquire` – but
    the value is never allowed to drop below zero. If the value is zero, then
    :meth:`acquire` will block until someone calls :meth:`release`.

    If you're looking for a :class:`Semaphore` to limit the number of tasks
    that can access some resource simultaneously, then consider using a
    :class:`CapacityLimiter` instead.

    This object's interface is similar to, but different from, that of
    :class:`threading.Semaphore`.

    A :class:`Semaphore` object can be used as an async context manager; it
    blocks on entry but not on exit.

    Args:
      initial_value (int): A non-negative integer giving semaphore's initial
        value.
      max_value (int or None): If given, makes this a "bounded" semaphore that
        raises an error if the value is about to exceed the given
        ``max_value``.

    """

    def __init__(self, initial_value: int, *, max_value: int | None = None) -> None:
        if not isinstance(initial_value, int):
            raise TypeError("initial_value must be an int")
        if initial_value < 0:
            raise ValueError("initial value must be >= 0")
        if max_value is not None:
            if not isinstance(max_value, int):
                raise TypeError("max_value must be None or an int")
            if max_value < initial_value:
                raise ValueError("max_values must be >= initial_value")

        # Invariants:
        # bool(self._lot) implies self._value == 0
        # (or equivalently: self._value > 0 implies not self._lot)
        self._lot = trio.lowlevel.ParkingLot()
        self._value = initial_value
        self._max_value = max_value

    def __repr__(self) -> str:
        if self._max_value is None:
            max_value_str = ""
        else:
            max_value_str = f", max_value={self._max_value}"
        return f"<trio.Semaphore({self._value}{max_value_str}) at {id(self):#x}>"

    @property
    def value(self) -> int:
        """The current value of the semaphore."""
        return self._value

    @property
    def max_value(self) -> int | None:
        """The maximum allowed value. May be None to indicate no limit."""
        return self._max_value

    @enable_ki_protection
    def acquire_nowait(self) -> None:
        """Attempt to decrement the semaphore value, without blocking.

        Raises:
          WouldBlock: if the value is zero.

        """
        if self._value > 0:
            assert not self._lot
            self._value -= 1
        else:
            raise trio.WouldBlock

    @enable_ki_protection
    async def acquire(self) -> None:
        """Decrement the semaphore value, blocking if necessary to avoid
        letting it drop below zero.

        """
        await trio.lowlevel.checkpoint_if_cancelled()
        try:
            self.acquire_nowait()
        except trio.WouldBlock:
            await self._lot.park()
        else:
            await trio.lowlevel.cancel_shielded_checkpoint()

    @enable_ki_protection
    def release(self) -> None:
        """Increment the semaphore value, possibly waking a task blocked in
        :meth:`acquire`.

        Raises:
          ValueError: if incrementing the value would cause it to exceed
              :attr:`max_value`.

        """
        if self._lot:
            assert self._value == 0
            self._lot.unpark(count=1)
        else:
            if self._max_value is not None and self._value == self._max_value:
                raise ValueError("semaphore released too many times")
            self._value += 1

    def statistics(self) -> ParkingLotStatistics:
        """Return an object containing debugging information.

        Currently the following fields are defined:

        * ``tasks_waiting``: The number of tasks blocked on this semaphore's
          :meth:`acquire` method.

        """
        return self._lot.statistics()


@attrs.frozen
class LockStatistics:
    """An object containing debugging information for a Lock.

    Currently the following fields are defined:

    * ``locked`` (boolean): indicating whether the lock is held.
    * ``owner``: the :class:`trio.lowlevel.Task` currently holding the lock,
      or None if the lock is not held.
    * ``tasks_waiting`` (int): The number of tasks blocked on this lock's
      :meth:`trio.Lock.acquire` method.

    """

    locked: bool
    owner: Task | None
    tasks_waiting: int


@attrs.define(eq=False, repr=False, slots=False)
class _LockImpl(AsyncContextManagerMixin):
    _lot: ParkingLot = attrs.field(factory=ParkingLot, init=False)
    _owner: Task | None = attrs.field(default=None, init=False)

    def __repr__(self) -> str:
        if self.locked():
            s1 = "locked"
            s2 = f" with {len(self._lot)} waiters"
        else:
            s1 = "unlocked"
            s2 = ""
        return f"<{s1} {self.__class__.__name__} object at {id(self):#x}{s2}>"

    def locked(self) -> bool:
        """Check whether the lock is currently held.

        Returns:
          bool: True if the lock is held, False otherwise.

        """
        return self._owner is not None

    @enable_ki_protection
    def acquire_nowait(self) -> None:
        """Attempt to acquire the lock, without blocking.

        Raises:
          WouldBlock: if the lock is held.

        """

        task = trio.lowlevel.current_task()
        if self._owner is task:
            raise RuntimeError("attempt to re-acquire an already held Lock")
        elif self._owner is None and not self._lot:
            # No-one owns it
            self._owner = task
            add_parking_lot_breaker(task, self._lot)
        else:
            raise trio.WouldBlock

    @enable_ki_protection
    async def acquire(self) -> None:
        """Acquire the lock, blocking if necessary.

        Raises:
          BrokenResourceError: if the owner of the lock exits without releasing.
        """
        await trio.lowlevel.checkpoint_if_cancelled()
        try:
            self.acquire_nowait()
        except trio.WouldBlock:
            try:
                # NOTE: it's important that the contended acquire path is just
                # "_lot.park()", because that's how Condition.wait() acquires the
                # lock as well.
                await self._lot.park()
            except trio.BrokenResourceError:
                raise trio.BrokenResourceError(
                    f"Owner of this lock exited without releasing: {self._owner}",
                ) from None
        else:
            await trio.lowlevel.cancel_shielded_checkpoint()

    @enable_ki_protection
    def release(self) -> None:
        """Release the lock.

        Raises:
          RuntimeError: if the calling task does not hold the lock.

        """
        task = trio.lowlevel.current_task()
        if task is not self._owner:
            raise RuntimeError("can't release a Lock you don't own")
        remove_parking_lot_breaker(self._owner, self._lot)
        if self._lot:
            (self._owner,) = self._lot.unpark(count=1)
            add_parking_lot_breaker(self._owner, self._lot)
        else:
            self._owner = None

    def statistics(self) -> LockStatistics:
        """Return an object containing debugging information.

        Currently the following fields are defined:

        * ``locked``: boolean indicating whether the lock is held.
        * ``owner``: the :class:`trio.lowlevel.Task` currently holding the lock,
          or None if the lock is not held.
        * ``tasks_waiting``: The number of tasks blocked on this lock's
          :meth:`acquire` method.

        """
        return LockStatistics(
            locked=self.locked(),
            owner=self._owner,
            tasks_waiting=len(self._lot),
        )


@final
class Lock(_LockImpl):
    """A classic `mutex
    <https://en.wikipedia.org/wiki/Lock_(computer_science)>`__.

    This is a non-reentrant, single-owner lock. Unlike
    :class:`threading.Lock`, only the owner of the lock is allowed to release
    it.

    A :class:`Lock` object can be used as an async context manager; it
    blocks on entry but not on exit.

    """


@final
class StrictFIFOLock(_LockImpl):
    r"""A variant of :class:`Lock` where tasks are guaranteed to acquire the
    lock in strict first-come-first-served order.

    An example of when this is useful is if you're implementing something like
    :class:`trio.SSLStream` or an HTTP/2 server using `h2
    <https://hyper-h2.readthedocs.io/>`__, where you have multiple concurrent
    tasks that are interacting with a shared state machine, and at
    unpredictable moments the state machine requests that a chunk of data be
    sent over the network. (For example, when using h2 simply reading incoming
    data can occasionally `create outgoing data to send
    <https://http2.github.io/http2-spec/#PING>`__.) The challenge is to make
    sure that these chunks are sent in the correct order, without being
    garbled.

    One option would be to use a regular :class:`Lock`, and wrap it around
    every interaction with the state machine::

        # This approach is sometimes workable but often sub-optimal; see below
        async with lock:
            state_machine.do_something()
            if state_machine.has_data_to_send():
                await conn.sendall(state_machine.get_data_to_send())

    But this can be problematic. If you're using h2 then *usually* reading
    incoming data doesn't create the need to send any data, so we don't want
    to force every task that tries to read from the network to sit and wait
    a potentially long time for ``sendall`` to finish. And in some situations
    this could even potentially cause a deadlock, if the remote peer is
    waiting for you to read some data before it accepts the data you're
    sending.

    :class:`StrictFIFOLock` provides an alternative. We can rewrite our
    example like::

        # Note: no awaits between when we start using the state machine and
        # when we block to take the lock!
        state_machine.do_something()
        if state_machine.has_data_to_send():
            # Notice that we fetch the data to send out of the state machine
            # *before* sleeping, so that other tasks won't see it.
            chunk = state_machine.get_data_to_send()
            async with strict_fifo_lock:
                await conn.sendall(chunk)

    First we do all our interaction with the state machine in a single
    scheduling quantum (notice there are no ``await``\s in there), so it's
    automatically atomic with respect to other tasks. And then if and only if
    we have data to send, we get in line to send it – and
    :class:`StrictFIFOLock` guarantees that each task will send its data in
    the same order that the state machine generated it.

    Currently, :class:`StrictFIFOLock` is identical to :class:`Lock`,
    but (a) this may not always be true in the future, especially if Trio ever
    implements `more sophisticated scheduling policies
    <https://github.com/python-trio/trio/issues/32>`__, and (b) the above code
    is relying on a pretty subtle property of its lock. Using a
    :class:`StrictFIFOLock` acts as an executable reminder that you're relying
    on this property.

    """


@attrs.frozen
class ConditionStatistics:
    r"""An object containing debugging information for a Condition.

    Currently the following fields are defined:

    * ``tasks_waiting`` (int): The number of tasks blocked on this condition's
      :meth:`trio.Condition.wait` method.
    * ``lock_statistics``: The result of calling the underlying
      :class:`Lock`\s  :meth:`~Lock.statistics` method.

    """

    tasks_waiting: int
    lock_statistics: LockStatistics


@final
class Condition(AsyncContextManagerMixin):
    """A classic `condition variable
    <https://en.wikipedia.org/wiki/Monitor_(synchronization)>`__, similar to
    :class:`threading.Condition`.

    A :class:`Condition` object can be used as an async context manager to
    acquire the underlying lock; it blocks on entry but not on exit.

    Args:
      lock (Lock): the lock object to use. If given, must be a
          :class:`trio.Lock`. If None, a new :class:`Lock` will be allocated
          and used.

    """

    def __init__(self, lock: Lock | None = None) -> None:
        if lock is None:
            lock = Lock()
        if type(lock) is not Lock:
            raise TypeError("lock must be a trio.Lock")
        self._lock = lock
        self._lot = trio.lowlevel.ParkingLot()

    def locked(self) -> bool:
        """Check whether the underlying lock is currently held.

        Returns:
          bool: True if the lock is held, False otherwise.

        """
        return self._lock.locked()

    def acquire_nowait(self) -> None:
        """Attempt to acquire the underlying lock, without blocking.

        Raises:
          WouldBlock: if the lock is currently held.

        """
        return self._lock.acquire_nowait()

    async def acquire(self) -> None:
        """Acquire the underlying lock, blocking if necessary.

        Raises:
          BrokenResourceError: if the owner of the underlying lock exits without releasing.
        """
        await self._lock.acquire()

    def release(self) -> None:
        """Release the underlying lock."""
        self._lock.release()

    @enable_ki_protection
    async def wait(self) -> None:
        """Wait for another task to call :meth:`notify` or
        :meth:`notify_all`.

        When calling this method, you must hold the lock. It releases the lock
        while waiting, and then re-acquires it before waking up.

        There is a subtlety with how this method interacts with cancellation:
        when cancelled it will block to re-acquire the lock before raising
        :exc:`Cancelled`. This may cause cancellation to be less prompt than
        expected. The advantage is that it makes code like this work::

           async with condition:
               await condition.wait()

        If we didn't re-acquire the lock before waking up, and :meth:`wait`
        were cancelled here, then we'd crash in ``condition.__aexit__`` when
        we tried to release the lock we no longer held.

        Raises:
          RuntimeError: if the calling task does not hold the lock.
          BrokenResourceError: if the owner of the lock exits without releasing, when attempting to re-acquire.

        """
        if trio.lowlevel.current_task() is not self._lock._owner:
            raise RuntimeError("must hold the lock to wait")
        self.release()
        # NOTE: we go to sleep on self._lot, but we'll wake up on
        # self._lock._lot. That's all that's required to acquire a Lock.
        try:
            await self._lot.park()
        except:
            with trio.CancelScope(shield=True):
                await self.acquire()
            raise

    def notify(self, n: int = 1) -> None:
        """Wake one or more tasks that are blocked in :meth:`wait`.

        Args:
          n (int): The number of tasks to wake.

        Raises:
          RuntimeError: if the calling task does not hold the lock.

        """
        if trio.lowlevel.current_task() is not self._lock._owner:
            raise RuntimeError("must hold the lock to notify")
        self._lot.repark(self._lock._lot, count=n)

    def notify_all(self) -> None:
        """Wake all tasks that are currently blocked in :meth:`wait`.

        Raises:
          RuntimeError: if the calling task does not hold the lock.

        """
        if trio.lowlevel.current_task() is not self._lock._owner:
            raise RuntimeError("must hold the lock to notify")
        self._lot.repark_all(self._lock._lot)

    def statistics(self) -> ConditionStatistics:
        r"""Return an object containing debugging information.

        Currently the following fields are defined:

        * ``tasks_waiting``: The number of tasks blocked on this condition's
          :meth:`wait` method.
        * ``lock_statistics``: The result of calling the underlying
          :class:`Lock`\s  :meth:`~Lock.statistics` method.

        """
        return ConditionStatistics(
            tasks_waiting=len(self._lot),
            lock_statistics=self._lock.statistics(),
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\frv_types.py ===
"""
Finite Discrete Random Variables - Prebuilt variable types

Contains
========
FiniteRV
DiscreteUniform
Die
Bernoulli
Coin
Binomial
BetaBinomial
Hypergeometric
Rademacher
IdealSoliton
RobustSoliton
"""


from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.numbers import (Integer, Rational)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import Or
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Intersection, Interval)
from sympy.functions.special.beta_functions import beta as beta_fn
from sympy.stats.frv import (SingleFiniteDistribution,
                             SingleFinitePSpace)
from sympy.stats.rv import _value_check, Density, is_random
from sympy.utilities.iterables import multiset
from sympy.utilities.misc import filldedent


__all__ = ['FiniteRV',
'DiscreteUniform',
'Die',
'Bernoulli',
'Coin',
'Binomial',
'BetaBinomial',
'Hypergeometric',
'Rademacher',
'IdealSoliton',
'RobustSoliton',
]

def rv(name, cls, *args, **kwargs):
    args = list(map(sympify, args))
    dist = cls(*args)
    if kwargs.pop('check', True):
        dist.check(*args)
    pspace = SingleFinitePSpace(name, dist)
    if any(is_random(arg) for arg in args):
        from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
        pspace = CompoundPSpace(name, CompoundDistribution(dist))
    return pspace.value

class FiniteDistributionHandmade(SingleFiniteDistribution):

    @property
    def dict(self):
        return self.args[0]

    def pmf(self, x):
        x = Symbol('x')
        return Lambda(x, Piecewise(*(
            [(v, Eq(k, x)) for k, v in self.dict.items()] + [(S.Zero, True)])))

    @property
    def set(self):
        return set(self.dict.keys())

    @staticmethod
    def check(density):
        for p in density.values():
            _value_check((p >= 0, p <= 1),
                        "Probability at a point must be between 0 and 1.")
        val = sum(density.values())
        _value_check(Eq(val, 1) != S.false, "Total Probability must be 1.")

def FiniteRV(name, density, **kwargs):
    r"""
    Create a Finite Random Variable given a dict representing the density.

    Parameters
    ==========

    name : Symbol
        Represents name of the random variable.
    density : dict
        Dictionary containing the pdf of finite distribution
    check : bool
        If True, it will check whether the given density
        integrates to 1 over the given set. If False, it
        will not perform this check. Default is False.

    Examples
    ========

    >>> from sympy.stats import FiniteRV, P, E

    >>> density = {0: .1, 1: .2, 2: .3, 3: .4}
    >>> X = FiniteRV('X', density)

    >>> E(X)
    2.00000000000000
    >>> P(X >= 2)
    0.700000000000000

    Returns
    =======

    RandomSymbol

    """
    # have a default of False while `rv` should have a default of True
    kwargs['check'] = kwargs.pop('check', False)
    return rv(name, FiniteDistributionHandmade, density, **kwargs)

class DiscreteUniformDistribution(SingleFiniteDistribution):

    @staticmethod
    def check(*args):
        # not using _value_check since there is a
        # suggestion for the user
        if len(set(args)) != len(args):
            weights = multiset(args)
            n = Integer(len(args))
            for k in weights:
                weights[k] /= n
            raise ValueError(filldedent("""
                Repeated args detected but set expected. For a
                distribution having different weights for each
                item use the following:""") + (
                '\nS("FiniteRV(%s, %s)")' % ("'X'", weights)))

    @property
    def p(self):
        return Rational(1, len(self.args))

    @property  # type: ignore
    @cacheit
    def dict(self):
        return dict.fromkeys(self.set, self.p)

    @property
    def set(self):
        return set(self.args)

    def pmf(self, x):
        if x in self.args:
            return self.p
        else:
            return S.Zero


def DiscreteUniform(name, items):
    r"""
    Create a Finite Random Variable representing a uniform distribution over
    the input set.

    Parameters
    ==========

    items : list/tuple
        Items over which Uniform distribution is to be made

    Examples
    ========

    >>> from sympy.stats import DiscreteUniform, density
    >>> from sympy import symbols

    >>> X = DiscreteUniform('X', symbols('a b c')) # equally likely over a, b, c
    >>> density(X).dict
    {a: 1/3, b: 1/3, c: 1/3}

    >>> Y = DiscreteUniform('Y', list(range(5))) # distribution over a range
    >>> density(Y).dict
    {0: 1/5, 1: 1/5, 2: 1/5, 3: 1/5, 4: 1/5}

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Discrete_uniform_distribution
    .. [2] https://mathworld.wolfram.com/DiscreteUniformDistribution.html

    """
    return rv(name, DiscreteUniformDistribution, *items)


class DieDistribution(SingleFiniteDistribution):
    _argnames = ('sides',)

    @staticmethod
    def check(sides):
        _value_check((sides.is_positive, sides.is_integer),
                    "number of sides must be a positive integer.")

    @property
    def is_symbolic(self):
        return not self.sides.is_number

    @property
    def high(self):
        return self.sides

    @property
    def low(self):
        return S.One

    @property
    def set(self):
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(0, self.sides))
        return set(map(Integer, range(1, self.sides + 1)))

    def pmf(self, x):
        x = sympify(x)
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or "
                        "'RandomSymbol' not %s" % (type(x)))
        cond = Ge(x, 1) & Le(x, self.sides) & Contains(x, S.Integers)
        return Piecewise((S.One/self.sides, cond), (S.Zero, True))

def Die(name, sides=6):
    r"""
    Create a Finite Random Variable representing a fair die.

    Parameters
    ==========

    sides : Integer
        Represents the number of sides of the Die, by default is 6

    Examples
    ========

    >>> from sympy.stats import Die, density
    >>> from sympy import Symbol

    >>> D6 = Die('D6', 6) # Six sided Die
    >>> density(D6).dict
    {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}

    >>> D4 = Die('D4', 4) # Four sided Die
    >>> density(D4).dict
    {1: 1/4, 2: 1/4, 3: 1/4, 4: 1/4}

    >>> n = Symbol('n', positive=True, integer=True)
    >>> Dn = Die('Dn', n) # n sided Die
    >>> density(Dn).dict
    Density(DieDistribution(n))
    >>> density(Dn).dict.subs(n, 4).doit()
    {1: 1/4, 2: 1/4, 3: 1/4, 4: 1/4}

    Returns
    =======

    RandomSymbol
    """

    return rv(name, DieDistribution, sides)


class BernoulliDistribution(SingleFiniteDistribution):
    _argnames = ('p', 'succ', 'fail')

    @staticmethod
    def check(p, succ, fail):
        _value_check((p >= 0, p <= 1),
                    "p should be in range [0, 1].")

    @property
    def set(self):
        return {self.succ, self.fail}

    def pmf(self, x):
        if isinstance(self.succ, Symbol) and isinstance(self.fail, Symbol):
            return Piecewise((self.p, x == self.succ),
                             (1 - self.p, x == self.fail),
                             (S.Zero, True))
        return Piecewise((self.p, Eq(x, self.succ)),
                         (1 - self.p, Eq(x, self.fail)),
                         (S.Zero, True))


def Bernoulli(name, p, succ=1, fail=0):
    r"""
    Create a Finite Random Variable representing a Bernoulli process.

    Parameters
    ==========

    p : Rational number between 0 and 1
       Represents probability of success
    succ : Integer/symbol/string
       Represents event of success
    fail : Integer/symbol/string
       Represents event of failure

    Examples
    ========

    >>> from sympy.stats import Bernoulli, density
    >>> from sympy import S

    >>> X = Bernoulli('X', S(3)/4) # 1-0 Bernoulli variable, probability = 3/4
    >>> density(X).dict
    {0: 1/4, 1: 3/4}

    >>> X = Bernoulli('X', S.Half, 'Heads', 'Tails') # A fair coin toss
    >>> density(X).dict
    {Heads: 1/2, Tails: 1/2}

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bernoulli_distribution
    .. [2] https://mathworld.wolfram.com/BernoulliDistribution.html

    """

    return rv(name, BernoulliDistribution, p, succ, fail)


def Coin(name, p=S.Half):
    r"""
    Create a Finite Random Variable representing a Coin toss.

    This is an equivalent of a Bernoulli random variable with
    "H" and "T" as success and failure events respectively.

    Parameters
    ==========

    p : Rational Number between 0 and 1
      Represents probability of getting "Heads", by default is Half

    Examples
    ========

    >>> from sympy.stats import Coin, density
    >>> from sympy import Rational

    >>> C = Coin('C') # A fair coin toss
    >>> density(C).dict
    {H: 1/2, T: 1/2}

    >>> C2 = Coin('C2', Rational(3, 5)) # An unfair coin
    >>> density(C2).dict
    {H: 3/5, T: 2/5}

    Returns
    =======

    RandomSymbol

    See Also
    ========

    sympy.stats.Binomial

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Coin_flipping

    """
    return rv(name, BernoulliDistribution, p, 'H', 'T')


class BinomialDistribution(SingleFiniteDistribution):
    _argnames = ('n', 'p', 'succ', 'fail')

    @staticmethod
    def check(n, p, succ, fail):
        _value_check((n.is_integer, n.is_nonnegative),
                    "'n' must be nonnegative integer.")
        _value_check((p <= 1, p >= 0),
                    "p should be in range [0, 1].")

    @property
    def high(self):
        return self.n

    @property
    def low(self):
        return S.Zero

    @property
    def is_symbolic(self):
        return not self.n.is_number

    @property
    def set(self):
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(0, self.n))
        return set(self.dict.keys())

    def pmf(self, x):
        n, p = self.n, self.p
        x = sympify(x)
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or "
                        "'RandomSymbol' not %s" % (type(x)))
        cond = Ge(x, 0) & Le(x, n) & Contains(x, S.Integers)
        return Piecewise((binomial(n, x) * p**x * (1 - p)**(n - x), cond), (S.Zero, True))

    @property  # type: ignore
    @cacheit
    def dict(self):
        if self.is_symbolic:
            return Density(self)
        return {k*self.succ + (self.n-k)*self.fail: self.pmf(k)
                    for k in range(0, self.n + 1)}


def Binomial(name, n, p, succ=1, fail=0):
    r"""
    Create a Finite Random Variable representing a binomial distribution.

    Parameters
    ==========

    n : Positive Integer
      Represents number of trials
    p : Rational Number between 0 and 1
      Represents probability of success
    succ : Integer/symbol/string
      Represents event of success, by default is 1
    fail : Integer/symbol/string
      Represents event of failure, by default is 0

    Examples
    ========

    >>> from sympy.stats import Binomial, density
    >>> from sympy import S, Symbol

    >>> X = Binomial('X', 4, S.Half) # Four "coin flips"
    >>> density(X).dict
    {0: 1/16, 1: 1/4, 2: 3/8, 3: 1/4, 4: 1/16}

    >>> n = Symbol('n', positive=True, integer=True)
    >>> p = Symbol('p', positive=True)
    >>> X = Binomial('X', n, S.Half) # n "coin flips"
    >>> density(X).dict
    Density(BinomialDistribution(n, 1/2, 1, 0))
    >>> density(X).dict.subs(n, 4).doit()
    {0: 1/16, 1: 1/4, 2: 3/8, 3: 1/4, 4: 1/16}

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Binomial_distribution
    .. [2] https://mathworld.wolfram.com/BinomialDistribution.html

    """

    return rv(name, BinomialDistribution, n, p, succ, fail)

#-------------------------------------------------------------------------------
# Beta-binomial distribution ----------------------------------------------------------

class BetaBinomialDistribution(SingleFiniteDistribution):
    _argnames = ('n', 'alpha', 'beta')

    @staticmethod
    def check(n, alpha, beta):
        _value_check((n.is_integer, n.is_nonnegative),
        "'n' must be nonnegative integer. n = %s." % str(n))
        _value_check((alpha > 0),
        "'alpha' must be: alpha > 0 . alpha = %s" % str(alpha))
        _value_check((beta > 0),
        "'beta' must be: beta > 0 . beta = %s" % str(beta))

    @property
    def high(self):
        return self.n

    @property
    def low(self):
        return S.Zero

    @property
    def is_symbolic(self):
        return not self.n.is_number

    @property
    def set(self):
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(0, self.n))
        return set(map(Integer, range(self.n + 1)))

    def pmf(self, k):
        n, a, b = self.n, self.alpha, self.beta
        return binomial(n, k) * beta_fn(k + a, n - k + b) / beta_fn(a, b)


def BetaBinomial(name, n, alpha, beta):
    r"""
    Create a Finite Random Variable representing a Beta-binomial distribution.

    Parameters
    ==========

    n : Positive Integer
      Represents number of trials
    alpha : Real positive number
    beta : Real positive number

    Examples
    ========

    >>> from sympy.stats import BetaBinomial, density

    >>> X = BetaBinomial('X', 2, 1, 1)
    >>> density(X).dict
    {0: 1/3, 1: 2*beta(2, 2), 2: 1/3}

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta-binomial_distribution
    .. [2] https://mathworld.wolfram.com/BetaBinomialDistribution.html

    """

    return rv(name, BetaBinomialDistribution, n, alpha, beta)


class HypergeometricDistribution(SingleFiniteDistribution):
    _argnames = ('N', 'm', 'n')

    @staticmethod
    def check(n, N, m):
        _value_check((N.is_integer, N.is_nonnegative),
                     "'N' must be nonnegative integer. N = %s." % str(N))
        _value_check((n.is_integer, n.is_nonnegative),
                     "'n' must be nonnegative integer. n = %s." % str(n))
        _value_check((m.is_integer, m.is_nonnegative),
                     "'m' must be nonnegative integer. m = %s." % str(m))

    @property
    def is_symbolic(self):
        return not all(x.is_number for x in (self.N, self.m, self.n))

    @property
    def high(self):
        return Piecewise((self.n, Lt(self.n, self.m) != False), (self.m, True))

    @property
    def low(self):
        return Piecewise((0, Gt(0, self.n + self.m - self.N) != False), (self.n + self.m - self.N, True))

    @property
    def set(self):
        N, m, n = self.N, self.m, self.n
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(self.low, self.high))
        return set(range(max(0, n + m - N), min(n, m) + 1))

    def pmf(self, k):
        N, m, n = self.N, self.m, self.n
        return S(binomial(m, k) * binomial(N - m, n - k))/binomial(N, n)


def Hypergeometric(name, N, m, n):
    r"""
    Create a Finite Random Variable representing a hypergeometric distribution.

    Parameters
    ==========

    N : Positive Integer
      Represents finite population of size N.
    m : Positive Integer
      Represents number of trials with required feature.
    n : Positive Integer
      Represents numbers of draws.


    Examples
    ========

    >>> from sympy.stats import Hypergeometric, density

    >>> X = Hypergeometric('X', 10, 5, 3) # 10 marbles, 5 white (success), 3 draws
    >>> density(X).dict
    {0: 1/12, 1: 5/12, 2: 5/12, 3: 1/12}

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hypergeometric_distribution
    .. [2] https://mathworld.wolfram.com/HypergeometricDistribution.html

    """
    return rv(name, HypergeometricDistribution, N, m, n)


class RademacherDistribution(SingleFiniteDistribution):

    @property
    def set(self):
        return {-1, 1}

    @property
    def pmf(self):
        k = Dummy('k')
        return Lambda(k, Piecewise((S.Half, Or(Eq(k, -1), Eq(k, 1))), (S.Zero, True)))

def Rademacher(name):
    r"""
    Create a Finite Random Variable representing a Rademacher distribution.

    Examples
    ========

    >>> from sympy.stats import Rademacher, density

    >>> X = Rademacher('X')
    >>> density(X).dict
    {-1: 1/2, 1: 1/2}

    Returns
    =======

    RandomSymbol

    See Also
    ========

    sympy.stats.Bernoulli

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Rademacher_distribution

    """
    return rv(name, RademacherDistribution)

class IdealSolitonDistribution(SingleFiniteDistribution):
    _argnames = ('k',)

    @staticmethod
    def check(k):
        _value_check(k.is_integer and k.is_positive,
                    "'k' must be a positive integer.")

    @property
    def low(self):
        return S.One

    @property
    def high(self):
        return self.k

    @property
    def set(self):
        return set(map(Integer, range(1, self.k + 1)))

    @property # type: ignore
    @cacheit
    def dict(self):
        if self.k.is_Symbol:
            return Density(self)
        d = {1: Rational(1, self.k)}
        d.update({i: Rational(1, i*(i - 1)) for i in range(2, self.k + 1)})
        return d

    def pmf(self, x):
        x = sympify(x)
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or "
                        "'RandomSymbol' not %s" % (type(x)))
        cond1 = Eq(x, 1) & x.is_integer
        cond2 = Ge(x, 1) & Le(x, self.k) & x.is_integer
        return Piecewise((1/self.k, cond1), (1/(x*(x - 1)), cond2), (S.Zero, True))

def IdealSoliton(name, k):
    r"""
    Create a Finite Random Variable of Ideal Soliton Distribution

    Parameters
    ==========

    k : Positive Integer
        Represents the number of input symbols in an LT (Luby Transform) code.

    Examples
    ========

    >>> from sympy.stats import IdealSoliton, density, P, E
    >>> sol = IdealSoliton('sol', 5)
    >>> density(sol).dict
    {1: 1/5, 2: 1/2, 3: 1/6, 4: 1/12, 5: 1/20}
    >>> density(sol).set
    {1, 2, 3, 4, 5}

    >>> from sympy import Symbol
    >>> k = Symbol('k', positive=True, integer=True)
    >>> sol = IdealSoliton('sol', k)
    >>> density(sol).dict
    Density(IdealSolitonDistribution(k))
    >>> density(sol).dict.subs(k, 10).doit()
    {1: 1/10, 2: 1/2, 3: 1/6, 4: 1/12, 5: 1/20, 6: 1/30, 7: 1/42, 8: 1/56, 9: 1/72, 10: 1/90}

    >>> E(sol.subs(k, 10))
    7381/2520

    >>> P(sol.subs(k, 4) > 2)
    1/4

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Soliton_distribution#Ideal_distribution
    .. [2] https://pages.cs.wisc.edu/~suman/courses/740/papers/luby02lt.pdf

    """
    return rv(name, IdealSolitonDistribution, k)

class RobustSolitonDistribution(SingleFiniteDistribution):
    _argnames= ('k', 'delta', 'c')

    @staticmethod
    def check(k, delta, c):
        _value_check(k.is_integer and k.is_positive,
                    "'k' must be a positive integer")
        _value_check(Gt(delta, 0) and Le(delta, 1),
                    "'delta' must be a real number in the interval (0,1)")
        _value_check(c.is_positive,
                    "'c' must be a positive real number.")

    @property
    def R(self):
        return self.c * log(self.k/self.delta) * self.k**0.5

    @property
    def Z(self):
        z = 0
        for i in Range(1, round(self.k/self.R)):
            z += (1/i)
        z += log(self.R/self.delta)
        return 1 + z * self.R/self.k

    @property
    def low(self):
        return S.One

    @property
    def high(self):
        return self.k

    @property
    def set(self):
        return set(map(Integer, range(1, self.k + 1)))

    @property
    def is_symbolic(self):
        return not (self.k.is_number and self.c.is_number and self.delta.is_number)

    def pmf(self, x):
        x = sympify(x)
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or "
                        "'RandomSymbol' not %s" % (type(x)))

        cond1 = Eq(x, 1) & x.is_integer
        cond2 = Ge(x, 1) & Le(x, self.k) & x.is_integer
        rho = Piecewise((Rational(1, self.k), cond1), (Rational(1, x*(x-1)), cond2), (S.Zero, True))

        cond1 = Ge(x, 1) & Le(x, round(self.k/self.R)-1)
        cond2 = Eq(x, round(self.k/self.R))
        tau = Piecewise((self.R/(self.k * x), cond1), (self.R * log(self.R/self.delta)/self.k, cond2), (S.Zero, True))

        return (rho + tau)/self.Z

def RobustSoliton(name, k, delta, c):
    r'''
    Create a Finite Random Variable of Robust Soliton Distribution

    Parameters
    ==========

    k : Positive Integer
        Represents the number of input symbols in an LT (Luby Transform) code.
    delta : Positive Rational Number
            Represents the failure probability. Must be in the interval (0,1).
    c : Positive Rational Number
        Constant of proportionality. Values close to 1 are recommended

    Examples
    ========

    >>> from sympy.stats import RobustSoliton, density, P, E
    >>> robSol = RobustSoliton('robSol', 5, 0.5, 0.01)
    >>> density(robSol).dict
    {1: 0.204253668152708, 2: 0.490631107897393, 3: 0.165210624506162, 4: 0.0834387731899302, 5: 0.0505633404760675}
    >>> density(robSol).set
    {1, 2, 3, 4, 5}

    >>> from sympy import Symbol
    >>> k = Symbol('k', positive=True, integer=True)
    >>> c = Symbol('c', positive=True)
    >>> robSol = RobustSoliton('robSol', k, 0.5, c)
    >>> density(robSol).dict
    Density(RobustSolitonDistribution(k, 0.5, c))
    >>> density(robSol).dict.subs(k, 10).subs(c, 0.03).doit()
    {1: 0.116641095387194, 2: 0.467045731687165, 3: 0.159984123349381, 4: 0.0821431680681869, 5: 0.0505765646770100,
    6: 0.0345781523420719, 7: 0.0253132820710503, 8: 0.0194459129233227, 9: 0.0154831166726115, 10: 0.0126733075238887}

    >>> E(robSol.subs(k, 10).subs(c, 0.05))
    2.91358846104106

    >>> P(robSol.subs(k, 4).subs(c, 0.1) > 2)
    0.243650614389834

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Soliton_distribution#Robust_distribution
    .. [2] https://www.inference.org.uk/mackay/itprnn/ps/588.596.pdf
    .. [3] https://pages.cs.wisc.edu/~suman/courses/740/papers/luby02lt.pdf

    '''
    return rv(name, RobustSolitonDistribution, k, delta, c)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\polysys.py ===
"""Solvers of systems of polynomial equations. """

from __future__ import annotations

from typing import Any
from collections.abc import Sequence, Iterable

import itertools

from sympy import Dummy
from sympy.core import S
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.sorting import default_sort_key
from sympy.logic.boolalg import Boolean
from sympy.polys import Poly, groebner, roots
from sympy.polys.domains import ZZ
from sympy.polys.polyoptions import build_options
from sympy.polys.polytools import parallel_poly_from_expr, sqf_part
from sympy.polys.polyerrors import (
    ComputationFailed,
    PolificationFailed,
    CoercionFailed,
    GeneratorsNeeded,
    DomainError
)
from sympy.simplify import rcollect
from sympy.utilities import postfixes
from sympy.utilities.iterables import cartes
from sympy.utilities.misc import filldedent
from sympy.logic.boolalg import Or, And
from sympy.core.relational import Eq


class SolveFailed(Exception):
    """Raised when solver's conditions were not met. """


def solve_poly_system(seq, *gens, strict=False, **args):
    """
    Return a list of solutions for the system of polynomial equations
    or else None.

    Parameters
    ==========

    seq: a list/tuple/set
        Listing all the equations that are needed to be solved
    gens: generators
        generators of the equations in seq for which we want the
        solutions
    strict: a boolean (default is False)
        if strict is True, NotImplementedError will be raised if
        the solution is known to be incomplete (which can occur if
        not all solutions are expressible in radicals)
    args: Keyword arguments
        Special options for solving the equations.


    Returns
    =======

    List[Tuple]
        a list of tuples with elements being solutions for the
        symbols in the order they were passed as gens
    None
        None is returned when the computed basis contains only the ground.

    Examples
    ========

    >>> from sympy import solve_poly_system
    >>> from sympy.abc import x, y

    >>> solve_poly_system([x*y - 2*y, 2*y**2 - x**2], x, y)
    [(0, 0), (2, -sqrt(2)), (2, sqrt(2))]

    >>> solve_poly_system([x**5 - x + y**3, y**2 - 1], x, y, strict=True)
    Traceback (most recent call last):
    ...
    UnsolvableFactorError

    """
    try:
        polys, opt = parallel_poly_from_expr(seq, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('solve_poly_system', len(seq), exc)

    if len(polys) == len(opt.gens) == 2:
        f, g = polys

        if all(i <= 2 for i in f.degree_list() + g.degree_list()):
            try:
                return solve_biquadratic(f, g, opt)
            except SolveFailed:
                pass

    return solve_generic(polys, opt, strict=strict)


def solve_biquadratic(f, g, opt):
    """Solve a system of two bivariate quadratic polynomial equations.

    Parameters
    ==========

    f: a single Expr or Poly
        First equation
    g: a single Expr or Poly
        Second Equation
    opt: an Options object
        For specifying keyword arguments and generators

    Returns
    =======

    List[Tuple]
        a list of tuples with elements being solutions for the
        symbols in the order they were passed as gens
    None
        None is returned when the computed basis contains only the ground.

    Examples
    ========

    >>> from sympy import Options, Poly
    >>> from sympy.abc import x, y
    >>> from sympy.solvers.polysys import solve_biquadratic
    >>> NewOption = Options((x, y), {'domain': 'ZZ'})

    >>> a = Poly(y**2 - 4 + x, y, x, domain='ZZ')
    >>> b = Poly(y*2 + 3*x - 7, y, x, domain='ZZ')
    >>> solve_biquadratic(a, b, NewOption)
    [(1/3, 3), (41/27, 11/9)]

    >>> a = Poly(y + x**2 - 3, y, x, domain='ZZ')
    >>> b = Poly(-y + x - 4, y, x, domain='ZZ')
    >>> solve_biquadratic(a, b, NewOption)
    [(7/2 - sqrt(29)/2, -sqrt(29)/2 - 1/2), (sqrt(29)/2 + 7/2, -1/2 + \
      sqrt(29)/2)]
    """
    G = groebner([f, g])

    if len(G) == 1 and G[0].is_ground:
        return None

    if len(G) != 2:
        raise SolveFailed

    x, y = opt.gens
    p, q = G
    if not p.gcd(q).is_ground:
        # not 0-dimensional
        raise SolveFailed

    p = Poly(p, x, expand=False)
    p_roots = [rcollect(expr, y) for expr in roots(p).keys()]

    q = q.ltrim(-1)
    q_roots = list(roots(q).keys())

    solutions = [(p_root.subs(y, q_root), q_root) for q_root, p_root in
                 itertools.product(q_roots, p_roots)]

    return sorted(solutions, key=default_sort_key)


def solve_generic(polys, opt, strict=False):
    """
    Solve a generic system of polynomial equations.

    Returns all possible solutions over C[x_1, x_2, ..., x_m] of a
    set F = { f_1, f_2, ..., f_n } of polynomial equations, using
    Groebner basis approach. For now only zero-dimensional systems
    are supported, which means F can have at most a finite number
    of solutions. If the basis contains only the ground, None is
    returned.

    The algorithm works by the fact that, supposing G is the basis
    of F with respect to an elimination order (here lexicographic
    order is used), G and F generate the same ideal, they have the
    same set of solutions. By the elimination property, if G is a
    reduced, zero-dimensional Groebner basis, then there exists an
    univariate polynomial in G (in its last variable). This can be
    solved by computing its roots. Substituting all computed roots
    for the last (eliminated) variable in other elements of G, new
    polynomial system is generated. Applying the above procedure
    recursively, a finite number of solutions can be found.

    The ability of finding all solutions by this procedure depends
    on the root finding algorithms. If no solutions were found, it
    means only that roots() failed, but the system is solvable. To
    overcome this difficulty use numerical algorithms instead.

    Parameters
    ==========

    polys: a list/tuple/set
        Listing all the polynomial equations that are needed to be solved
    opt: an Options object
        For specifying keyword arguments and generators
    strict: a boolean
        If strict is True, NotImplementedError will be raised if the solution
        is known to be incomplete

    Returns
    =======

    List[Tuple]
        a list of tuples with elements being solutions for the
        symbols in the order they were passed as gens
    None
        None is returned when the computed basis contains only the ground.

    References
    ==========

    .. [Buchberger01] B. Buchberger, Groebner Bases: A Short
    Introduction for Systems Theorists, In: R. Moreno-Diaz,
    B. Buchberger, J.L. Freire, Proceedings of EUROCAST'01,
    February, 2001

    .. [Cox97] D. Cox, J. Little, D. O'Shea, Ideals, Varieties
    and Algorithms, Springer, Second Edition, 1997, pp. 112

    Raises
    ========

    NotImplementedError
        If the system is not zero-dimensional (does not have a finite
        number of solutions)

    UnsolvableFactorError
        If ``strict`` is True and not all solution components are
        expressible in radicals

    Examples
    ========

    >>> from sympy import Poly, Options
    >>> from sympy.solvers.polysys import solve_generic
    >>> from sympy.abc import x, y
    >>> NewOption = Options((x, y), {'domain': 'ZZ'})

    >>> a = Poly(x - y + 5, x, y, domain='ZZ')
    >>> b = Poly(x + y - 3, x, y, domain='ZZ')
    >>> solve_generic([a, b], NewOption)
    [(-1, 4)]

    >>> a = Poly(x - 2*y + 5, x, y, domain='ZZ')
    >>> b = Poly(2*x - y - 3, x, y, domain='ZZ')
    >>> solve_generic([a, b], NewOption)
    [(11/3, 13/3)]

    >>> a = Poly(x**2 + y, x, y, domain='ZZ')
    >>> b = Poly(x + y*4, x, y, domain='ZZ')
    >>> solve_generic([a, b], NewOption)
    [(0, 0), (1/4, -1/16)]

    >>> a = Poly(x**5 - x + y**3, x, y, domain='ZZ')
    >>> b = Poly(y**2 - 1, x, y, domain='ZZ')
    >>> solve_generic([a, b], NewOption, strict=True)
    Traceback (most recent call last):
    ...
    UnsolvableFactorError

    """
    def _is_univariate(f):
        """Returns True if 'f' is univariate in its last variable. """
        for monom in f.monoms():
            if any(monom[:-1]):
                return False

        return True

    def _subs_root(f, gen, zero):
        """Replace generator with a root so that the result is nice. """
        p = f.as_expr({gen: zero})

        if f.degree(gen) >= 2:
            p = p.expand(deep=False)

        return p

    def _solve_reduced_system(system, gens, entry=False):
        """Recursively solves reduced polynomial systems. """
        if len(system) == len(gens) == 1:
            # the below line will produce UnsolvableFactorError if
            # strict=True and the solution from `roots` is incomplete
            zeros = list(roots(system[0], gens[-1], strict=strict).keys())
            return [(zero,) for zero in zeros]

        basis = groebner(system, gens, polys=True)

        if len(basis) == 1 and basis[0].is_ground:
            if not entry:
                return []
            else:
                return None

        univariate = list(filter(_is_univariate, basis))

        if len(basis) < len(gens):
            raise NotImplementedError(filldedent('''
                only zero-dimensional systems supported
                (finite number of solutions)
                '''))

        if len(univariate) == 1:
            f = univariate.pop()
        else:
            raise NotImplementedError(filldedent('''
                only zero-dimensional systems supported
                (finite number of solutions)
                '''))

        gens = f.gens
        gen = gens[-1]

        # the below line will produce UnsolvableFactorError if
        # strict=True and the solution from `roots` is incomplete
        zeros = list(roots(f.ltrim(gen), strict=strict).keys())

        if not zeros:
            return []

        if len(basis) == 1:
            return [(zero,) for zero in zeros]

        solutions = []

        for zero in zeros:
            new_system = []
            new_gens = gens[:-1]

            for b in basis[:-1]:
                eq = _subs_root(b, gen, zero)

                if eq is not S.Zero:
                    new_system.append(eq)

            for solution in _solve_reduced_system(new_system, new_gens):
                solutions.append(solution + (zero,))

        if solutions and len(solutions[0]) != len(gens):
            raise NotImplementedError(filldedent('''
                only zero-dimensional systems supported
                (finite number of solutions)
                '''))
        return solutions

    try:
        result = _solve_reduced_system(polys, opt.gens, entry=True)
    except CoercionFailed:
        raise NotImplementedError

    if result is not None:
        return sorted(result, key=default_sort_key)


def solve_triangulated(polys, *gens, **args):
    """
    Solve a polynomial system using Gianni-Kalkbrenner algorithm.

    The algorithm proceeds by computing one Groebner basis in the ground
    domain and then by iteratively computing polynomial factorizations in
    appropriately constructed algebraic extensions of the ground domain.

    Parameters
    ==========

    polys: a list/tuple/set
        Listing all the equations that are needed to be solved
    gens: generators
        generators of the equations in polys for which we want the
        solutions
    args: Keyword arguments
        Special options for solving the equations

    Returns
    =======

    List[Tuple]
        A List of tuples. Solutions for symbols that satisfy the
        equations listed in polys

    Examples
    ========

    >>> from sympy import solve_triangulated
    >>> from sympy.abc import x, y, z

    >>> F = [x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1]

    >>> solve_triangulated(F, x, y, z)
    [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

    Using extension for algebraic solutions.

    >>> solve_triangulated(F, x, y, z, extension=True) #doctest: +NORMALIZE_WHITESPACE
    [(0, 0, 1), (0, 1, 0), (1, 0, 0),
     (CRootOf(x**2 + 2*x - 1, 0), CRootOf(x**2 + 2*x - 1, 0), CRootOf(x**2 + 2*x - 1, 0)),
     (CRootOf(x**2 + 2*x - 1, 1), CRootOf(x**2 + 2*x - 1, 1), CRootOf(x**2 + 2*x - 1, 1))]

    References
    ==========

    1. Patrizia Gianni, Teo Mora, Algebraic Solution of System of
    Polynomial Equations using Groebner Bases, AAECC-5 on Applied Algebra,
    Algebraic Algorithms and Error-Correcting Codes, LNCS 356 247--257, 1989

    """
    opt = build_options(gens, args)

    G = groebner(polys, gens, polys=True)
    G = list(reversed(G))

    extension = opt.get('extension', False)
    if extension:
        def _solve_univariate(f):
            return [r for r, _ in f.all_roots(multiple=False, radicals=False)]
    else:
        domain = opt.get('domain')

        if domain is not None:
            for i, g in enumerate(G):
                G[i] = g.set_domain(domain)

        def _solve_univariate(f):
            return list(f.ground_roots().keys())

    f, G = G[0].ltrim(-1), G[1:]
    dom = f.get_domain()

    zeros = _solve_univariate(f)

    if extension:
        solutions = {((zero,), dom.algebraic_field(zero)) for zero in zeros}
    else:
        solutions = {((zero,), dom) for zero in zeros}

    var_seq = reversed(gens[:-1])
    vars_seq = postfixes(gens[1:])

    for var, vars in zip(var_seq, vars_seq):
        _solutions = set()

        for values, dom in solutions:
            H, mapping = [], list(zip(vars, values))

            for g in G:
                _vars = (var,) + vars

                if g.has_only_gens(*_vars) and g.degree(var) != 0:
                    if extension:
                        g = g.set_domain(g.domain.unify(dom))
                    h = g.ltrim(var).eval(dict(mapping))

                    if g.degree(var) == h.degree():
                        H.append(h)

            p = min(H, key=lambda h: h.degree())
            zeros = _solve_univariate(p)

            for zero in zeros:
                if not (zero in dom):
                    dom_zero = dom.algebraic_field(zero)
                else:
                    dom_zero = dom

                _solutions.add(((zero,) + values, dom_zero))

        solutions = _solutions
    return sorted((s for s, _ in solutions), key=default_sort_key)


def factor_system(eqs: Sequence[Expr | complex], gens: Sequence[Expr] = (), **kwargs: Any) -> list[list[Expr]]:
    """
    Factorizes a system of polynomial equations into
    irreducible subsystems.

    Parameters
    ==========

    eqs : list
        List of expressions to be factored.
        Each expression is assumed to be equal to zero.

    gens : list, optional
        Generator(s) of the polynomial ring.
        If not provided, all free symbols will be used.

    **kwargs : dict, optional
        Same optional arguments taken by ``factor``

    Returns
    =======

    list[list[Expr]]
        A list of lists of expressions, where each sublist represents
        an irreducible subsystem. When solved, each subsystem gives
        one component of the solution. Only generic solutions are
        returned (cases not requiring parameters to be zero).

    Examples
    ========

    >>> from sympy.solvers.polysys import factor_system, factor_system_cond
    >>> from sympy.abc import x, y, a, b, c

    A simple system with multiple solutions:

    >>> factor_system([x**2 - 1, y - 1])
    [[x + 1, y - 1], [x - 1, y - 1]]

    A system with no solution:

    >>> factor_system([x, 1])
    []

    A system where any value of the symbol(s) is a solution:

    >>> factor_system([x - x, (x + 1)**2 - (x**2 + 2*x + 1)])
    [[]]

    A system with no generic solution:

    >>> factor_system([a*x*(x-1), b*y, c], [x, y])
    []

    If c is added to the unknowns then the system has a generic solution:

    >>> factor_system([a*x*(x-1), b*y, c], [x, y, c])
    [[x - 1, y, c], [x, y, c]]

    Alternatively :func:`factor_system_cond` can be used to get degenerate
    cases as well:

    >>> factor_system_cond([a*x*(x-1), b*y, c], [x, y])
    [[x - 1, y, c], [x, y, c], [x - 1, b, c], [x, b, c], [y, a, c], [a, b, c]]

    Each of the above cases is only satisfiable in the degenerate case `c = 0`.

    The solution set of the original system represented
    by eqs is the union of the solution sets of the
    factorized systems.

    An empty list [] means no generic solution exists.
    A list containing an empty list [[]] means any value of
    the symbol(s) is a solution.

    See Also
    ========

    factor_system_cond : Returns both generic and degenerate solutions
    factor_system_bool : Returns a Boolean combination representing all solutions
    sympy.polys.polytools.factor : Factors a polynomial into irreducible factors
                                   over the rational numbers
    """

    systems = _factor_system_poly_from_expr(eqs, gens, **kwargs)
    systems_generic = [sys for sys in systems if not _is_degenerate(sys)]
    systems_expr = [[p.as_expr() for p in system] for system in systems_generic]
    return systems_expr


def _is_degenerate(system: list[Poly]) -> bool:
    """Helper function to check if a system is degenerate"""
    return any(p.is_ground for p in system)


def factor_system_bool(eqs: Sequence[Expr | complex], gens: Sequence[Expr] = (), **kwargs: Any) -> Boolean:
    """
    Factorizes a system of polynomial equations into irreducible DNF.

    The system of expressions(eqs) is taken and a Boolean combination
    of equations is returned that represents the same solution set.
    The result is in disjunctive normal form (OR of ANDs).

    Parameters
    ==========

    eqs : list
       List of expressions to be factored.
       Each expression is assumed to be equal to zero.

    gens : list, optional
       Generator(s) of the polynomial ring.
       If not provided, all free symbols will be used.

    **kwargs : dict, optional
       Optional keyword arguments


    Returns
    =======

    Boolean:
       A Boolean combination of equations. The result is typically in
       the form of a conjunction (AND) of a disjunctive normal form
       with additional conditions.

    Examples
    ========

    >>> from sympy.solvers.polysys import factor_system_bool
    >>> from sympy.abc import x, y, a, b, c
    >>> factor_system_bool([x**2 - 1])
    Eq(x - 1, 0) | Eq(x + 1, 0)

    >>> factor_system_bool([x**2 - 1, y - 1])
    (Eq(x - 1, 0) & Eq(y - 1, 0)) | (Eq(x + 1, 0) & Eq(y - 1, 0))

    >>> eqs = [a * (x - 1), b]
    >>> factor_system_bool([a*(x - 1), b])
    (Eq(a, 0) & Eq(b, 0)) | (Eq(b, 0) & Eq(x - 1, 0))

    >>> factor_system_bool([a*x**2 - a, b*(x + 1), c], [x])
    (Eq(c, 0) & Eq(x + 1, 0)) | (Eq(a, 0) & Eq(b, 0) & Eq(c, 0)) | (Eq(b, 0) & Eq(c, 0) & Eq(x - 1, 0))

    >>> factor_system_bool([x**2 + 2*x + 1 - (x + 1)**2])
    True

    The result is logically equivalent to the system of equations
    i.e. eqs. The function returns ``True`` when all values of
    the symbol(s) is a solution and ``False`` when the system
    cannot be solved.

    See Also
    ========

    factor_system : Returns factors and solvability condition separately
    factor_system_cond : Returns both factors and conditions

    """

    systems = factor_system_cond(eqs, gens, **kwargs)
    return Or(*[And(*[Eq(eq, 0) for eq in sys]) for sys in systems])


def factor_system_cond(eqs: Sequence[Expr | complex], gens: Sequence[Expr] = (), **kwargs: Any) -> list[list[Expr]]:
    """
    Factorizes a polynomial system into irreducible components and returns
    both generic and degenerate solutions.

    Parameters
    ==========

    eqs : list
        List of expressions to be factored.
        Each expression is assumed to be equal to zero.

    gens : list, optional
        Generator(s) of the polynomial ring.
        If not provided, all free symbols will be used.

    **kwargs : dict, optional
        Optional keyword arguments.

    Returns
    =======

    list[list[Expr]]
        A list of lists of expressions, where each sublist represents
        an irreducible subsystem. Includes both generic solutions and
        degenerate cases requiring equality conditions on parameters.

    Examples
    ========

    >>> from sympy.solvers.polysys import factor_system_cond
    >>> from sympy.abc import x, y, a, b, c

    >>> factor_system_cond([x**2 - 4, a*y, b], [x, y])
    [[x + 2, y, b], [x - 2, y, b], [x + 2, a, b], [x - 2, a, b]]

    >>> factor_system_cond([a*x*(x-1), b*y, c], [x, y])
    [[x - 1, y, c], [x, y, c], [x - 1, b, c], [x, b, c], [y, a, c], [a, b, c]]

    An empty list [] means no solution exists.
    A list containing an empty list [[]] means any value of
    the symbol(s) is a solution.

    See Also
    ========

    factor_system : Returns only generic solutions
    factor_system_bool : Returns a Boolean combination representing all solutions
    sympy.polys.polytools.factor : Factors a polynomial into irreducible factors
                                   over the rational numbers
    """
    systems_poly = _factor_system_poly_from_expr(eqs, gens, **kwargs)
    systems = [[p.as_expr() for p in system] for system in systems_poly]
    return systems


def _factor_system_poly_from_expr(
        eqs: Sequence[Expr | complex], gens: Sequence[Expr], **kwargs: Any
) -> list[list[Poly]]:
    """
    Convert expressions to polynomials and factor the system.

    Takes a sequence of expressions, converts them to
    polynomials, and factors the resulting system. Handles both regular
    polynomial systems and purely numerical cases.
    """
    try:
        polys, opts = parallel_poly_from_expr(eqs, *gens, **kwargs)
        only_numbers = False
    except (GeneratorsNeeded, PolificationFailed):
        _u = Dummy('u')
        polys, opts = parallel_poly_from_expr(eqs, [_u], **kwargs)
        assert opts['domain'].is_Numerical
        only_numbers = True

    if only_numbers:
        return [[]] if all(p == 0 for p in polys) else []

    return factor_system_poly(polys)


def factor_system_poly(polys: list[Poly]) -> list[list[Poly]]:
    """
    Factors a system of polynomial equations into irreducible subsystems

    Core implementation that works directly with Poly instances.

    Parameters
    ==========

    polys : list[Poly]
        A list of Poly instances to be factored.

    Returns
    =======

    list[list[Poly]]
        A list of lists of polynomials, where each sublist represents
        an irreducible component of the solution. Includes both
        generic and degenerate cases.

    Examples
    ========

    >>> from sympy import symbols, Poly, ZZ
    >>> from sympy.solvers.polysys import factor_system_poly
    >>> a, b, c, x = symbols('a b c x')
    >>> p1 = Poly((a - 1)*(x - 2), x, domain=ZZ[a,b,c])
    >>> p2 = Poly((b - 3)*(x - 2), x, domain=ZZ[a,b,c])
    >>> p3 = Poly(c, x, domain=ZZ[a,b,c])

    The equation to be solved for x is ``x - 2 = 0`` provided either
    of the two conditions on the parameters ``a`` and ``b`` is nonzero
    and the constant parameter ``c`` should be zero.

    >>> sys1, sys2 = factor_system_poly([p1, p2, p3])
    >>> sys1
    [Poly(x - 2, x, domain='ZZ[a,b,c]'),
     Poly(c, x, domain='ZZ[a,b,c]')]
    >>> sys2
    [Poly(a - 1, x, domain='ZZ[a,b,c]'),
     Poly(b - 3, x, domain='ZZ[a,b,c]'),
     Poly(c, x, domain='ZZ[a,b,c]')]

     An empty list [] when returned means no solution exists.
     Whereas a list containing an empty list [[]] means any value is a solution.

    See Also
    ========

    factor_system : Returns only generic solutions
    factor_system_bool : Returns a Boolean combination representing the solutions
    factor_system_cond : Returns both generic and degenerate solutions
    sympy.polys.polytools.factor : Factors a polynomial into irreducible factors
                                   over the rational numbers
    """
    if not all(isinstance(poly, Poly) for poly in polys):
        raise TypeError("polys should be a list of Poly instances")
    if not polys:
        return [[]]

    base_domain = polys[0].domain
    base_gens = polys[0].gens
    if not all(poly.domain == base_domain and poly.gens == base_gens for poly in polys[1:]):
        raise DomainError("All polynomials must have the same domain and generators")

    factor_sets = []
    for poly in polys:
        constant, factors_mult = poly.factor_list()

        if constant.is_zero is True:
            continue
        elif constant.is_zero is False:
            if not factors_mult:
                return []
            factor_sets.append([f for f, _ in factors_mult])
        else:
            constant = sqf_part(factor_terms(constant).as_coeff_Mul()[1])
            constp = Poly(constant, base_gens, domain=base_domain)
            factors = [f for f, _ in factors_mult]
            factors.append(constp)
            factor_sets.append(factors)

    if not factor_sets:
        return [[]]

    result = _factor_sets(factor_sets)
    return _sort_systems(result)


def _factor_sets_slow(eqs: list[list]) -> set[frozenset]:
    """
    Helper to find the minimal set of factorised subsystems that is
    equivalent to the original system.

    The result is in DNF.
    """
    if not eqs:
        return {frozenset()}
    systems_set = {frozenset(sys) for sys in cartes(*eqs)}
    return {s1 for s1 in systems_set if not any(s1 > s2 for s2 in systems_set)}


def _factor_sets(eqs: list[list]) -> set[frozenset]:
    """
    Helper that builds factor combinations.
    """
    if not eqs:
        return {frozenset()}

    current_set = min(eqs, key=len)
    other_sets = [s for s in eqs if s is not current_set]

    stack = [(factor, [s for s in other_sets if factor not in s], {factor})
             for factor in current_set]

    result = set()

    while stack:
        factor, remaining_sets, current_solution = stack.pop()

        if not remaining_sets:
            result.add(frozenset(current_solution))
            continue

        next_set = min(remaining_sets, key=len)
        next_remaining = [s for s in remaining_sets if s is not next_set]

        for next_factor in next_set:
            valid_remaining = [s for s in next_remaining if next_factor not in s]
            new_solution = current_solution | {next_factor}
            stack.append((next_factor, valid_remaining, new_solution))

    return {s1 for s1 in result if not any(s1 > s2 for s2 in result)}


def _sort_systems(systems: Iterable[Iterable[Poly]]) -> list[list[Poly]]:
    """Sorts a list of lists of polynomials"""
    systems_list = [sorted(s, key=_poly_sort_key, reverse=True) for s in systems]
    return sorted(systems_list, key=_sys_sort_key, reverse=True)


def _poly_sort_key(poly):
    """Sort key for polynomials"""
    if poly.domain.is_FF:
        poly = poly.set_domain(ZZ)
    return poly.degree_list(), poly.rep.to_list()


def _sys_sort_key(sys):
    """Sort key for lists of polynomials"""
    return list(zip(*map(_poly_sort_key, sys)))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\ext.py ===
"""Extension API for adding custom tags and behavior."""

import pprint
import re
import typing as t

from markupsafe import Markup

from . import defaults
from . import nodes
from .environment import Environment
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .runtime import concat  # type: ignore
from .runtime import Context
from .runtime import Undefined
from .utils import import_string
from .utils import pass_context

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .lexer import Token
    from .lexer import TokenStream
    from .parser import Parser

    class _TranslationsBasic(te.Protocol):
        def gettext(self, message: str) -> str: ...

        def ngettext(self, singular: str, plural: str, n: int) -> str:
            pass

    class _TranslationsContext(_TranslationsBasic):
        def pgettext(self, context: str, message: str) -> str: ...

        def npgettext(
            self, context: str, singular: str, plural: str, n: int
        ) -> str: ...

    _SupportedTranslations = t.Union[_TranslationsBasic, _TranslationsContext]


# I18N functions available in Jinja templates. If the I18N library
# provides ugettext, it will be assigned to gettext.
GETTEXT_FUNCTIONS: t.Tuple[str, ...] = (
    "_",
    "gettext",
    "ngettext",
    "pgettext",
    "npgettext",
)
_ws_re = re.compile(r"\s*\n\s*")


class Extension:
    """Extensions can be used to add extra functionality to the Jinja template
    system at the parser level.  Custom extensions are bound to an environment
    but may not store environment specific data on `self`.  The reason for
    this is that an extension can be bound to another environment (for
    overlays) by creating a copy and reassigning the `environment` attribute.

    As extensions are created by the environment they cannot accept any
    arguments for configuration.  One may want to work around that by using
    a factory function, but that is not possible as extensions are identified
    by their import name.  The correct way to configure the extension is
    storing the configuration values on the environment.  Because this way the
    environment ends up acting as central configuration storage the
    attributes may clash which is why extensions have to ensure that the names
    they choose for configuration are not too generic.  ``prefix`` for example
    is a terrible name, ``fragment_cache_prefix`` on the other hand is a good
    name as includes the name of the extension (fragment cache).
    """

    identifier: t.ClassVar[str]

    def __init_subclass__(cls) -> None:
        cls.identifier = f"{cls.__module__}.{cls.__name__}"

    #: if this extension parses this is the list of tags it's listening to.
    tags: t.Set[str] = set()

    #: the priority of that extension.  This is especially useful for
    #: extensions that preprocess values.  A lower value means higher
    #: priority.
    #:
    #: .. versionadded:: 2.4
    priority = 100

    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    def bind(self, environment: Environment) -> "te.Self":
        """Create a copy of this extension bound to another environment."""
        rv = object.__new__(self.__class__)
        rv.__dict__.update(self.__dict__)
        rv.environment = environment
        return rv

    def preprocess(
        self, source: str, name: t.Optional[str], filename: t.Optional[str] = None
    ) -> str:
        """This method is called before the actual lexing and can be used to
        preprocess the source.  The `filename` is optional.  The return value
        must be the preprocessed source.
        """
        return source

    def filter_stream(
        self, stream: "TokenStream"
    ) -> t.Union["TokenStream", t.Iterable["Token"]]:
        """It's passed a :class:`~jinja2.lexer.TokenStream` that can be used
        to filter tokens returned.  This method has to return an iterable of
        :class:`~jinja2.lexer.Token`\\s, but it doesn't have to return a
        :class:`~jinja2.lexer.TokenStream`.
        """
        return stream

    def parse(self, parser: "Parser") -> t.Union[nodes.Node, t.List[nodes.Node]]:
        """If any of the :attr:`tags` matched this method is called with the
        parser as first argument.  The token the parser stream is pointing at
        is the name token that matched.  This method has to return one or a
        list of multiple nodes.
        """
        raise NotImplementedError()

    def attr(
        self, name: str, lineno: t.Optional[int] = None
    ) -> nodes.ExtensionAttribute:
        """Return an attribute node for the current extension.  This is useful
        to pass constants on extensions to generated template code.

        ::

            self.attr('_my_attribute', lineno=lineno)
        """
        return nodes.ExtensionAttribute(self.identifier, name, lineno=lineno)

    def call_method(
        self,
        name: str,
        args: t.Optional[t.List[nodes.Expr]] = None,
        kwargs: t.Optional[t.List[nodes.Keyword]] = None,
        dyn_args: t.Optional[nodes.Expr] = None,
        dyn_kwargs: t.Optional[nodes.Expr] = None,
        lineno: t.Optional[int] = None,
    ) -> nodes.Call:
        """Call a method of the extension.  This is a shortcut for
        :meth:`attr` + :class:`jinja2.nodes.Call`.
        """
        if args is None:
            args = []
        if kwargs is None:
            kwargs = []
        return nodes.Call(
            self.attr(name, lineno=lineno),
            args,
            kwargs,
            dyn_args,
            dyn_kwargs,
            lineno=lineno,
        )


@pass_context
def _gettext_alias(
    __context: Context, *args: t.Any, **kwargs: t.Any
) -> t.Union[t.Any, Undefined]:
    return __context.call(__context.resolve("gettext"), *args, **kwargs)


def _make_new_gettext(func: t.Callable[[str], str]) -> t.Callable[..., str]:
    @pass_context
    def gettext(__context: Context, __string: str, **variables: t.Any) -> str:
        rv = __context.call(func, __string)
        if __context.eval_ctx.autoescape:
            rv = Markup(rv)
        # Always treat as a format string, even if there are no
        # variables. This makes translation strings more consistent
        # and predictable. This requires escaping
        return rv % variables  # type: ignore

    return gettext


def _make_new_ngettext(func: t.Callable[[str, str, int], str]) -> t.Callable[..., str]:
    @pass_context
    def ngettext(
        __context: Context,
        __singular: str,
        __plural: str,
        __num: int,
        **variables: t.Any,
    ) -> str:
        variables.setdefault("num", __num)
        rv = __context.call(func, __singular, __plural, __num)
        if __context.eval_ctx.autoescape:
            rv = Markup(rv)
        # Always treat as a format string, see gettext comment above.
        return rv % variables  # type: ignore

    return ngettext


def _make_new_pgettext(func: t.Callable[[str, str], str]) -> t.Callable[..., str]:
    @pass_context
    def pgettext(
        __context: Context, __string_ctx: str, __string: str, **variables: t.Any
    ) -> str:
        variables.setdefault("context", __string_ctx)
        rv = __context.call(func, __string_ctx, __string)

        if __context.eval_ctx.autoescape:
            rv = Markup(rv)

        # Always treat as a format string, see gettext comment above.
        return rv % variables  # type: ignore

    return pgettext


def _make_new_npgettext(
    func: t.Callable[[str, str, str, int], str],
) -> t.Callable[..., str]:
    @pass_context
    def npgettext(
        __context: Context,
        __string_ctx: str,
        __singular: str,
        __plural: str,
        __num: int,
        **variables: t.Any,
    ) -> str:
        variables.setdefault("context", __string_ctx)
        variables.setdefault("num", __num)
        rv = __context.call(func, __string_ctx, __singular, __plural, __num)

        if __context.eval_ctx.autoescape:
            rv = Markup(rv)

        # Always treat as a format string, see gettext comment above.
        return rv % variables  # type: ignore

    return npgettext


class InternationalizationExtension(Extension):
    """This extension adds gettext support to Jinja."""

    tags = {"trans"}

    # TODO: the i18n extension is currently reevaluating values in a few
    # situations.  Take this example:
    #   {% trans count=something() %}{{ count }} foo{% pluralize
    #     %}{{ count }} fooss{% endtrans %}
    # something is called twice here.  One time for the gettext value and
    # the other time for the n-parameter of the ngettext function.

    def __init__(self, environment: Environment) -> None:
        super().__init__(environment)
        environment.globals["_"] = _gettext_alias
        environment.extend(
            install_gettext_translations=self._install,
            install_null_translations=self._install_null,
            install_gettext_callables=self._install_callables,
            uninstall_gettext_translations=self._uninstall,
            extract_translations=self._extract,
            newstyle_gettext=False,
        )

    def _install(
        self, translations: "_SupportedTranslations", newstyle: t.Optional[bool] = None
    ) -> None:
        # ugettext and ungettext are preferred in case the I18N library
        # is providing compatibility with older Python versions.
        gettext = getattr(translations, "ugettext", None)
        if gettext is None:
            gettext = translations.gettext
        ngettext = getattr(translations, "ungettext", None)
        if ngettext is None:
            ngettext = translations.ngettext

        pgettext = getattr(translations, "pgettext", None)
        npgettext = getattr(translations, "npgettext", None)
        self._install_callables(
            gettext, ngettext, newstyle=newstyle, pgettext=pgettext, npgettext=npgettext
        )

    def _install_null(self, newstyle: t.Optional[bool] = None) -> None:
        import gettext

        translations = gettext.NullTranslations()

        if hasattr(translations, "pgettext"):
            # Python < 3.8
            pgettext = translations.pgettext
        else:

            def pgettext(c: str, s: str) -> str:  # type: ignore[misc]
                return s

        if hasattr(translations, "npgettext"):
            npgettext = translations.npgettext
        else:

            def npgettext(c: str, s: str, p: str, n: int) -> str:  # type: ignore[misc]
                return s if n == 1 else p

        self._install_callables(
            gettext=translations.gettext,
            ngettext=translations.ngettext,
            newstyle=newstyle,
            pgettext=pgettext,
            npgettext=npgettext,
        )

    def _install_callables(
        self,
        gettext: t.Callable[[str], str],
        ngettext: t.Callable[[str, str, int], str],
        newstyle: t.Optional[bool] = None,
        pgettext: t.Optional[t.Callable[[str, str], str]] = None,
        npgettext: t.Optional[t.Callable[[str, str, str, int], str]] = None,
    ) -> None:
        if newstyle is not None:
            self.environment.newstyle_gettext = newstyle  # type: ignore
        if self.environment.newstyle_gettext:  # type: ignore
            gettext = _make_new_gettext(gettext)
            ngettext = _make_new_ngettext(ngettext)

            if pgettext is not None:
                pgettext = _make_new_pgettext(pgettext)

            if npgettext is not None:
                npgettext = _make_new_npgettext(npgettext)

        self.environment.globals.update(
            gettext=gettext, ngettext=ngettext, pgettext=pgettext, npgettext=npgettext
        )

    def _uninstall(self, translations: "_SupportedTranslations") -> None:
        for key in ("gettext", "ngettext", "pgettext", "npgettext"):
            self.environment.globals.pop(key, None)

    def _extract(
        self,
        source: t.Union[str, nodes.Template],
        gettext_functions: t.Sequence[str] = GETTEXT_FUNCTIONS,
    ) -> t.Iterator[
        t.Tuple[int, str, t.Union[t.Optional[str], t.Tuple[t.Optional[str], ...]]]
    ]:
        if isinstance(source, str):
            source = self.environment.parse(source)
        return extract_from_ast(source, gettext_functions)

    def parse(self, parser: "Parser") -> t.Union[nodes.Node, t.List[nodes.Node]]:
        """Parse a translatable tag."""
        lineno = next(parser.stream).lineno

        context = None
        context_token = parser.stream.next_if("string")

        if context_token is not None:
            context = context_token.value

        # find all the variables referenced.  Additionally a variable can be
        # defined in the body of the trans block too, but this is checked at
        # a later state.
        plural_expr: t.Optional[nodes.Expr] = None
        plural_expr_assignment: t.Optional[nodes.Assign] = None
        num_called_num = False
        variables: t.Dict[str, nodes.Expr] = {}
        trimmed = None
        while parser.stream.current.type != "block_end":
            if variables:
                parser.stream.expect("comma")

            # skip colon for python compatibility
            if parser.stream.skip_if("colon"):
                break

            token = parser.stream.expect("name")
            if token.value in variables:
                parser.fail(
                    f"translatable variable {token.value!r} defined twice.",
                    token.lineno,
                    exc=TemplateAssertionError,
                )

            # expressions
            if parser.stream.current.type == "assign":
                next(parser.stream)
                variables[token.value] = var = parser.parse_expression()
            elif trimmed is None and token.value in ("trimmed", "notrimmed"):
                trimmed = token.value == "trimmed"
                continue
            else:
                variables[token.value] = var = nodes.Name(token.value, "load")

            if plural_expr is None:
                if isinstance(var, nodes.Call):
                    plural_expr = nodes.Name("_trans", "load")
                    variables[token.value] = plural_expr
                    plural_expr_assignment = nodes.Assign(
                        nodes.Name("_trans", "store"), var
                    )
                else:
                    plural_expr = var
                num_called_num = token.value == "num"

        parser.stream.expect("block_end")

        plural = None
        have_plural = False
        referenced = set()

        # now parse until endtrans or pluralize
        singular_names, singular = self._parse_block(parser, True)
        if singular_names:
            referenced.update(singular_names)
            if plural_expr is None:
                plural_expr = nodes.Name(singular_names[0], "load")
                num_called_num = singular_names[0] == "num"

        # if we have a pluralize block, we parse that too
        if parser.stream.current.test("name:pluralize"):
            have_plural = True
            next(parser.stream)
            if parser.stream.current.type != "block_end":
                token = parser.stream.expect("name")
                if token.value not in variables:
                    parser.fail(
                        f"unknown variable {token.value!r} for pluralization",
                        token.lineno,
                        exc=TemplateAssertionError,
                    )
                plural_expr = variables[token.value]
                num_called_num = token.value == "num"
            parser.stream.expect("block_end")
            plural_names, plural = self._parse_block(parser, False)
            next(parser.stream)
            referenced.update(plural_names)
        else:
            next(parser.stream)

        # register free names as simple name expressions
        for name in referenced:
            if name not in variables:
                variables[name] = nodes.Name(name, "load")

        if not have_plural:
            plural_expr = None
        elif plural_expr is None:
            parser.fail("pluralize without variables", lineno)

        if trimmed is None:
            trimmed = self.environment.policies["ext.i18n.trimmed"]
        if trimmed:
            singular = self._trim_whitespace(singular)
            if plural:
                plural = self._trim_whitespace(plural)

        node = self._make_node(
            singular,
            plural,
            context,
            variables,
            plural_expr,
            bool(referenced),
            num_called_num and have_plural,
        )
        node.set_lineno(lineno)
        if plural_expr_assignment is not None:
            return [plural_expr_assignment, node]
        else:
            return node

    def _trim_whitespace(self, string: str, _ws_re: t.Pattern[str] = _ws_re) -> str:
        return _ws_re.sub(" ", string.strip())

    def _parse_block(
        self, parser: "Parser", allow_pluralize: bool
    ) -> t.Tuple[t.List[str], str]:
        """Parse until the next block tag with a given name."""
        referenced = []
        buf = []

        while True:
            if parser.stream.current.type == "data":
                buf.append(parser.stream.current.value.replace("%", "%%"))
                next(parser.stream)
            elif parser.stream.current.type == "variable_begin":
                next(parser.stream)
                name = parser.stream.expect("name").value
                referenced.append(name)
                buf.append(f"%({name})s")
                parser.stream.expect("variable_end")
            elif parser.stream.current.type == "block_begin":
                next(parser.stream)
                block_name = (
                    parser.stream.current.value
                    if parser.stream.current.type == "name"
                    else None
                )
                if block_name == "endtrans":
                    break
                elif block_name == "pluralize":
                    if allow_pluralize:
                        break
                    parser.fail(
                        "a translatable section can have only one pluralize section"
                    )
                elif block_name == "trans":
                    parser.fail(
                        "trans blocks can't be nested; did you mean `endtrans`?"
                    )
                parser.fail(
                    f"control structures in translatable sections are not allowed; "
                    f"saw `{block_name}`"
                )
            elif parser.stream.eos:
                parser.fail("unclosed translation block")
            else:
                raise RuntimeError("internal parser error")

        return referenced, concat(buf)

    def _make_node(
        self,
        singular: str,
        plural: t.Optional[str],
        context: t.Optional[str],
        variables: t.Dict[str, nodes.Expr],
        plural_expr: t.Optional[nodes.Expr],
        vars_referenced: bool,
        num_called_num: bool,
    ) -> nodes.Output:
        """Generates a useful node from the data provided."""
        newstyle = self.environment.newstyle_gettext  # type: ignore
        node: nodes.Expr

        # no variables referenced?  no need to escape for old style
        # gettext invocations only if there are vars.
        if not vars_referenced and not newstyle:
            singular = singular.replace("%%", "%")
            if plural:
                plural = plural.replace("%%", "%")

        func_name = "gettext"
        func_args: t.List[nodes.Expr] = [nodes.Const(singular)]

        if context is not None:
            func_args.insert(0, nodes.Const(context))
            func_name = f"p{func_name}"

        if plural_expr is not None:
            func_name = f"n{func_name}"
            func_args.extend((nodes.Const(plural), plural_expr))

        node = nodes.Call(nodes.Name(func_name, "load"), func_args, [], None, None)

        # in case newstyle gettext is used, the method is powerful
        # enough to handle the variable expansion and autoescape
        # handling itself
        if newstyle:
            for key, value in variables.items():
                # the function adds that later anyways in case num was
                # called num, so just skip it.
                if num_called_num and key == "num":
                    continue
                node.kwargs.append(nodes.Keyword(key, value))

        # otherwise do that here
        else:
            # mark the return value as safe if we are in an
            # environment with autoescaping turned on
            node = nodes.MarkSafeIfAutoescape(node)
            if variables:
                node = nodes.Mod(
                    node,
                    nodes.Dict(
                        [
                            nodes.Pair(nodes.Const(key), value)
                            for key, value in variables.items()
                        ]
                    ),
                )
        return nodes.Output([node])


class ExprStmtExtension(Extension):
    """Adds a `do` tag to Jinja that works like the print statement just
    that it doesn't print the return value.
    """

    tags = {"do"}

    def parse(self, parser: "Parser") -> nodes.ExprStmt:
        node = nodes.ExprStmt(lineno=next(parser.stream).lineno)
        node.node = parser.parse_tuple()
        return node


class LoopControlExtension(Extension):
    """Adds break and continue to the template engine."""

    tags = {"break", "continue"}

    def parse(self, parser: "Parser") -> t.Union[nodes.Break, nodes.Continue]:
        token = next(parser.stream)
        if token.value == "break":
            return nodes.Break(lineno=token.lineno)
        return nodes.Continue(lineno=token.lineno)


class DebugExtension(Extension):
    """A ``{% debug %}`` tag that dumps the available variables,
    filters, and tests.

    .. code-block:: html+jinja

        <pre>{% debug %}</pre>

    .. code-block:: text

        {'context': {'cycler': <class 'jinja2.utils.Cycler'>,
                     ...,
                     'namespace': <class 'jinja2.utils.Namespace'>},
         'filters': ['abs', 'attr', 'batch', 'capitalize', 'center', 'count', 'd',
                     ..., 'urlencode', 'urlize', 'wordcount', 'wordwrap', 'xmlattr'],
         'tests': ['!=', '<', '<=', '==', '>', '>=', 'callable', 'defined',
                   ..., 'odd', 'sameas', 'sequence', 'string', 'undefined', 'upper']}

    .. versionadded:: 2.11.0
    """

    tags = {"debug"}

    def parse(self, parser: "Parser") -> nodes.Output:
        lineno = parser.stream.expect("name:debug").lineno
        context = nodes.ContextReference()
        result = self.call_method("_render", [context], lineno=lineno)
        return nodes.Output([result], lineno=lineno)

    def _render(self, context: Context) -> str:
        result = {
            "context": context.get_all(),
            "filters": sorted(self.environment.filters.keys()),
            "tests": sorted(self.environment.tests.keys()),
        }

        # Set the depth since the intent is to show the top few names.
        return pprint.pformat(result, depth=3, compact=True)


def extract_from_ast(
    ast: nodes.Template,
    gettext_functions: t.Sequence[str] = GETTEXT_FUNCTIONS,
    babel_style: bool = True,
) -> t.Iterator[
    t.Tuple[int, str, t.Union[t.Optional[str], t.Tuple[t.Optional[str], ...]]]
]:
    """Extract localizable strings from the given template node.  Per
    default this function returns matches in babel style that means non string
    parameters as well as keyword arguments are returned as `None`.  This
    allows Babel to figure out what you really meant if you are using
    gettext functions that allow keyword arguments for placeholder expansion.
    If you don't want that behavior set the `babel_style` parameter to `False`
    which causes only strings to be returned and parameters are always stored
    in tuples.  As a consequence invalid gettext calls (calls without a single
    string parameter or string parameters after non-string parameters) are
    skipped.

    This example explains the behavior:

    >>> from jinja2 import Environment
    >>> env = Environment()
    >>> node = env.parse('{{ (_("foo"), _(), ngettext("foo", "bar", 42)) }}')
    >>> list(extract_from_ast(node))
    [(1, '_', 'foo'), (1, '_', ()), (1, 'ngettext', ('foo', 'bar', None))]
    >>> list(extract_from_ast(node, babel_style=False))
    [(1, '_', ('foo',)), (1, 'ngettext', ('foo', 'bar'))]

    For every string found this function yields a ``(lineno, function,
    message)`` tuple, where:

    * ``lineno`` is the number of the line on which the string was found,
    * ``function`` is the name of the ``gettext`` function used (if the
      string was extracted from embedded Python code), and
    *   ``message`` is the string, or a tuple of strings for functions
         with multiple string arguments.

    This extraction function operates on the AST and is because of that unable
    to extract any comments.  For comment support you have to use the babel
    extraction interface or extract comments yourself.
    """
    out: t.Union[t.Optional[str], t.Tuple[t.Optional[str], ...]]

    for node in ast.find_all(nodes.Call):
        if (
            not isinstance(node.node, nodes.Name)
            or node.node.name not in gettext_functions
        ):
            continue

        strings: t.List[t.Optional[str]] = []

        for arg in node.args:
            if isinstance(arg, nodes.Const) and isinstance(arg.value, str):
                strings.append(arg.value)
            else:
                strings.append(None)

        for _ in node.kwargs:
            strings.append(None)
        if node.dyn_args is not None:
            strings.append(None)
        if node.dyn_kwargs is not None:
            strings.append(None)

        if not babel_style:
            out = tuple(x for x in strings if x is not None)

            if not out:
                continue
        else:
            if len(strings) == 1:
                out = strings[0]
            else:
                out = tuple(strings)

        yield node.lineno, node.node.name, out


class _CommentFinder:
    """Helper class to find comments in a token stream.  Can only
    find comments for gettext calls forwards.  Once the comment
    from line 4 is found, a comment for line 1 will not return a
    usable value.
    """

    def __init__(
        self, tokens: t.Sequence[t.Tuple[int, str, str]], comment_tags: t.Sequence[str]
    ) -> None:
        self.tokens = tokens
        self.comment_tags = comment_tags
        self.offset = 0
        self.last_lineno = 0

    def find_backwards(self, offset: int) -> t.List[str]:
        try:
            for _, token_type, token_value in reversed(
                self.tokens[self.offset : offset]
            ):
                if token_type in ("comment", "linecomment"):
                    try:
                        prefix, comment = token_value.split(None, 1)
                    except ValueError:
                        continue
                    if prefix in self.comment_tags:
                        return [comment.rstrip()]
            return []
        finally:
            self.offset = offset

    def find_comments(self, lineno: int) -> t.List[str]:
        if not self.comment_tags or self.last_lineno > lineno:
            return []
        for idx, (token_lineno, _, _) in enumerate(self.tokens[self.offset :]):
            if token_lineno > lineno:
                return self.find_backwards(self.offset + idx)
        return self.find_backwards(len(self.tokens))


def babel_extract(
    fileobj: t.BinaryIO,
    keywords: t.Sequence[str],
    comment_tags: t.Sequence[str],
    options: t.Dict[str, t.Any],
) -> t.Iterator[
    t.Tuple[
        int, str, t.Union[t.Optional[str], t.Tuple[t.Optional[str], ...]], t.List[str]
    ]
]:
    """Babel extraction method for Jinja templates.

    .. versionchanged:: 2.3
       Basic support for translation comments was added.  If `comment_tags`
       is now set to a list of keywords for extraction, the extractor will
       try to find the best preceding comment that begins with one of the
       keywords.  For best results, make sure to not have more than one
       gettext call in one line of code and the matching comment in the
       same line or the line before.

    .. versionchanged:: 2.5.1
       The `newstyle_gettext` flag can be set to `True` to enable newstyle
       gettext calls.

    .. versionchanged:: 2.7
       A `silent` option can now be provided.  If set to `False` template
       syntax errors are propagated instead of being ignored.

    :param fileobj: the file-like object the messages should be extracted from
    :param keywords: a list of keywords (i.e. function names) that should be
                     recognized as translation functions
    :param comment_tags: a list of translator tags to search for and include
                         in the results.
    :param options: a dictionary of additional options (optional)
    :return: an iterator over ``(lineno, funcname, message, comments)`` tuples.
             (comments will be empty currently)
    """
    extensions: t.Dict[t.Type[Extension], None] = {}

    for extension_name in options.get("extensions", "").split(","):
        extension_name = extension_name.strip()

        if not extension_name:
            continue

        extensions[import_string(extension_name)] = None

    if InternationalizationExtension not in extensions:
        extensions[InternationalizationExtension] = None

    def getbool(options: t.Mapping[str, str], key: str, default: bool = False) -> bool:
        return options.get(key, str(default)).lower() in {"1", "on", "yes", "true"}

    silent = getbool(options, "silent", True)
    environment = Environment(
        options.get("block_start_string", defaults.BLOCK_START_STRING),
        options.get("block_end_string", defaults.BLOCK_END_STRING),
        options.get("variable_start_string", defaults.VARIABLE_START_STRING),
        options.get("variable_end_string", defaults.VARIABLE_END_STRING),
        options.get("comment_start_string", defaults.COMMENT_START_STRING),
        options.get("comment_end_string", defaults.COMMENT_END_STRING),
        options.get("line_statement_prefix") or defaults.LINE_STATEMENT_PREFIX,
        options.get("line_comment_prefix") or defaults.LINE_COMMENT_PREFIX,
        getbool(options, "trim_blocks", defaults.TRIM_BLOCKS),
        getbool(options, "lstrip_blocks", defaults.LSTRIP_BLOCKS),
        defaults.NEWLINE_SEQUENCE,
        getbool(options, "keep_trailing_newline", defaults.KEEP_TRAILING_NEWLINE),
        tuple(extensions),
        cache_size=0,
        auto_reload=False,
    )

    if getbool(options, "trimmed"):
        environment.policies["ext.i18n.trimmed"] = True
    if getbool(options, "newstyle_gettext"):
        environment.newstyle_gettext = True  # type: ignore

    source = fileobj.read().decode(options.get("encoding", "utf-8"))
    try:
        node = environment.parse(source)
        tokens = list(environment.lex(environment.preprocess(source)))
    except TemplateSyntaxError:
        if not silent:
            raise
        # skip templates with syntax errors
        return

    finder = _CommentFinder(tokens, comment_tags)
    for lineno, func, message in extract_from_ast(node, keywords):
        yield lineno, func, message, finder.find_comments(lineno)


#: nicer import names
i18n = InternationalizationExtension
do = ExprStmtExtension
loopcontrols = LoopControlExtension
debug = DebugExtension

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\dom_snapshot.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: DOMSnapshot (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import dom_debugger
from . import page


@dataclass
class DOMNode:
    '''
    A Node in the DOM tree.
    '''
    #: ``Node``'s nodeType.
    node_type: int

    #: ``Node``'s nodeName.
    node_name: str

    #: ``Node``'s nodeValue.
    node_value: str

    #: ``Node``'s id, corresponds to DOM.Node.backendNodeId.
    backend_node_id: dom.BackendNodeId

    #: Only set for textarea elements, contains the text value.
    text_value: typing.Optional[str] = None

    #: Only set for input elements, contains the input's associated text value.
    input_value: typing.Optional[str] = None

    #: Only set for radio and checkbox input elements, indicates if the element has been checked
    input_checked: typing.Optional[bool] = None

    #: Only set for option elements, indicates if the element has been selected
    option_selected: typing.Optional[bool] = None

    #: The indexes of the node's child nodes in the ``domNodes`` array returned by ``getSnapshot``, if
    #: any.
    child_node_indexes: typing.Optional[typing.List[int]] = None

    #: Attributes of an ``Element`` node.
    attributes: typing.Optional[typing.List[NameValue]] = None

    #: Indexes of pseudo elements associated with this node in the ``domNodes`` array returned by
    #: ``getSnapshot``, if any.
    pseudo_element_indexes: typing.Optional[typing.List[int]] = None

    #: The index of the node's related layout tree node in the ``layoutTreeNodes`` array returned by
    #: ``getSnapshot``, if any.
    layout_node_index: typing.Optional[int] = None

    #: Document URL that ``Document`` or ``FrameOwner`` node points to.
    document_url: typing.Optional[str] = None

    #: Base URL that ``Document`` or ``FrameOwner`` node uses for URL completion.
    base_url: typing.Optional[str] = None

    #: Only set for documents, contains the document's content language.
    content_language: typing.Optional[str] = None

    #: Only set for documents, contains the document's character set encoding.
    document_encoding: typing.Optional[str] = None

    #: ``DocumentType`` node's publicId.
    public_id: typing.Optional[str] = None

    #: ``DocumentType`` node's systemId.
    system_id: typing.Optional[str] = None

    #: Frame ID for frame owner elements and also for the document node.
    frame_id: typing.Optional[page.FrameId] = None

    #: The index of a frame owner element's content document in the ``domNodes`` array returned by
    #: ``getSnapshot``, if any.
    content_document_index: typing.Optional[int] = None

    #: Type of a pseudo element node.
    pseudo_type: typing.Optional[dom.PseudoType] = None

    #: Shadow root type.
    shadow_root_type: typing.Optional[dom.ShadowRootType] = None

    #: Whether this DOM node responds to mouse clicks. This includes nodes that have had click
    #: event listeners attached via JavaScript as well as anchor tags that naturally navigate when
    #: clicked.
    is_clickable: typing.Optional[bool] = None

    #: Details of the node's event listeners, if any.
    event_listeners: typing.Optional[typing.List[dom_debugger.EventListener]] = None

    #: The selected url for nodes with a srcset attribute.
    current_source_url: typing.Optional[str] = None

    #: The url of the script (if any) that generates this node.
    origin_url: typing.Optional[str] = None

    #: Scroll offsets, set when this node is a Document.
    scroll_offset_x: typing.Optional[float] = None

    scroll_offset_y: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['nodeType'] = self.node_type
        json['nodeName'] = self.node_name
        json['nodeValue'] = self.node_value
        json['backendNodeId'] = self.backend_node_id.to_json()
        if self.text_value is not None:
            json['textValue'] = self.text_value
        if self.input_value is not None:
            json['inputValue'] = self.input_value
        if self.input_checked is not None:
            json['inputChecked'] = self.input_checked
        if self.option_selected is not None:
            json['optionSelected'] = self.option_selected
        if self.child_node_indexes is not None:
            json['childNodeIndexes'] = [i for i in self.child_node_indexes]
        if self.attributes is not None:
            json['attributes'] = [i.to_json() for i in self.attributes]
        if self.pseudo_element_indexes is not None:
            json['pseudoElementIndexes'] = [i for i in self.pseudo_element_indexes]
        if self.layout_node_index is not None:
            json['layoutNodeIndex'] = self.layout_node_index
        if self.document_url is not None:
            json['documentURL'] = self.document_url
        if self.base_url is not None:
            json['baseURL'] = self.base_url
        if self.content_language is not None:
            json['contentLanguage'] = self.content_language
        if self.document_encoding is not None:
            json['documentEncoding'] = self.document_encoding
        if self.public_id is not None:
            json['publicId'] = self.public_id
        if self.system_id is not None:
            json['systemId'] = self.system_id
        if self.frame_id is not None:
            json['frameId'] = self.frame_id.to_json()
        if self.content_document_index is not None:
            json['contentDocumentIndex'] = self.content_document_index
        if self.pseudo_type is not None:
            json['pseudoType'] = self.pseudo_type.to_json()
        if self.shadow_root_type is not None:
            json['shadowRootType'] = self.shadow_root_type.to_json()
        if self.is_clickable is not None:
            json['isClickable'] = self.is_clickable
        if self.event_listeners is not None:
            json['eventListeners'] = [i.to_json() for i in self.event_listeners]
        if self.current_source_url is not None:
            json['currentSourceURL'] = self.current_source_url
        if self.origin_url is not None:
            json['originURL'] = self.origin_url
        if self.scroll_offset_x is not None:
            json['scrollOffsetX'] = self.scroll_offset_x
        if self.scroll_offset_y is not None:
            json['scrollOffsetY'] = self.scroll_offset_y
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            node_type=int(json['nodeType']),
            node_name=str(json['nodeName']),
            node_value=str(json['nodeValue']),
            backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId']),
            text_value=str(json['textValue']) if 'textValue' in json else None,
            input_value=str(json['inputValue']) if 'inputValue' in json else None,
            input_checked=bool(json['inputChecked']) if 'inputChecked' in json else None,
            option_selected=bool(json['optionSelected']) if 'optionSelected' in json else None,
            child_node_indexes=[int(i) for i in json['childNodeIndexes']] if 'childNodeIndexes' in json else None,
            attributes=[NameValue.from_json(i) for i in json['attributes']] if 'attributes' in json else None,
            pseudo_element_indexes=[int(i) for i in json['pseudoElementIndexes']] if 'pseudoElementIndexes' in json else None,
            layout_node_index=int(json['layoutNodeIndex']) if 'layoutNodeIndex' in json else None,
            document_url=str(json['documentURL']) if 'documentURL' in json else None,
            base_url=str(json['baseURL']) if 'baseURL' in json else None,
            content_language=str(json['contentLanguage']) if 'contentLanguage' in json else None,
            document_encoding=str(json['documentEncoding']) if 'documentEncoding' in json else None,
            public_id=str(json['publicId']) if 'publicId' in json else None,
            system_id=str(json['systemId']) if 'systemId' in json else None,
            frame_id=page.FrameId.from_json(json['frameId']) if 'frameId' in json else None,
            content_document_index=int(json['contentDocumentIndex']) if 'contentDocumentIndex' in json else None,
            pseudo_type=dom.PseudoType.from_json(json['pseudoType']) if 'pseudoType' in json else None,
            shadow_root_type=dom.ShadowRootType.from_json(json['shadowRootType']) if 'shadowRootType' in json else None,
            is_clickable=bool(json['isClickable']) if 'isClickable' in json else None,
            event_listeners=[dom_debugger.EventListener.from_json(i) for i in json['eventListeners']] if 'eventListeners' in json else None,
            current_source_url=str(json['currentSourceURL']) if 'currentSourceURL' in json else None,
            origin_url=str(json['originURL']) if 'originURL' in json else None,
            scroll_offset_x=float(json['scrollOffsetX']) if 'scrollOffsetX' in json else None,
            scroll_offset_y=float(json['scrollOffsetY']) if 'scrollOffsetY' in json else None,
        )


@dataclass
class InlineTextBox:
    '''
    Details of post layout rendered text positions. The exact layout should not be regarded as
    stable and may change between versions.
    '''
    #: The bounding box in document coordinates. Note that scroll offset of the document is ignored.
    bounding_box: dom.Rect

    #: The starting index in characters, for this post layout textbox substring. Characters that
    #: would be represented as a surrogate pair in UTF-16 have length 2.
    start_character_index: int

    #: The number of characters in this post layout textbox substring. Characters that would be
    #: represented as a surrogate pair in UTF-16 have length 2.
    num_characters: int

    def to_json(self):
        json = dict()
        json['boundingBox'] = self.bounding_box.to_json()
        json['startCharacterIndex'] = self.start_character_index
        json['numCharacters'] = self.num_characters
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            bounding_box=dom.Rect.from_json(json['boundingBox']),
            start_character_index=int(json['startCharacterIndex']),
            num_characters=int(json['numCharacters']),
        )


@dataclass
class LayoutTreeNode:
    '''
    Details of an element in the DOM tree with a LayoutObject.
    '''
    #: The index of the related DOM node in the ``domNodes`` array returned by ``getSnapshot``.
    dom_node_index: int

    #: The bounding box in document coordinates. Note that scroll offset of the document is ignored.
    bounding_box: dom.Rect

    #: Contents of the LayoutText, if any.
    layout_text: typing.Optional[str] = None

    #: The post-layout inline text nodes, if any.
    inline_text_nodes: typing.Optional[typing.List[InlineTextBox]] = None

    #: Index into the ``computedStyles`` array returned by ``getSnapshot``.
    style_index: typing.Optional[int] = None

    #: Global paint order index, which is determined by the stacking order of the nodes. Nodes
    #: that are painted together will have the same index. Only provided if includePaintOrder in
    #: getSnapshot was true.
    paint_order: typing.Optional[int] = None

    #: Set to true to indicate the element begins a new stacking context.
    is_stacking_context: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['domNodeIndex'] = self.dom_node_index
        json['boundingBox'] = self.bounding_box.to_json()
        if self.layout_text is not None:
            json['layoutText'] = self.layout_text
        if self.inline_text_nodes is not None:
            json['inlineTextNodes'] = [i.to_json() for i in self.inline_text_nodes]
        if self.style_index is not None:
            json['styleIndex'] = self.style_index
        if self.paint_order is not None:
            json['paintOrder'] = self.paint_order
        if self.is_stacking_context is not None:
            json['isStackingContext'] = self.is_stacking_context
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            dom_node_index=int(json['domNodeIndex']),
            bounding_box=dom.Rect.from_json(json['boundingBox']),
            layout_text=str(json['layoutText']) if 'layoutText' in json else None,
            inline_text_nodes=[InlineTextBox.from_json(i) for i in json['inlineTextNodes']] if 'inlineTextNodes' in json else None,
            style_index=int(json['styleIndex']) if 'styleIndex' in json else None,
            paint_order=int(json['paintOrder']) if 'paintOrder' in json else None,
            is_stacking_context=bool(json['isStackingContext']) if 'isStackingContext' in json else None,
        )


@dataclass
class ComputedStyle:
    '''
    A subset of the full ComputedStyle as defined by the request whitelist.
    '''
    #: Name/value pairs of computed style properties.
    properties: typing.List[NameValue]

    def to_json(self):
        json = dict()
        json['properties'] = [i.to_json() for i in self.properties]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            properties=[NameValue.from_json(i) for i in json['properties']],
        )


@dataclass
class NameValue:
    '''
    A name/value pair.
    '''
    #: Attribute/property name.
    name: str

    #: Attribute/property value.
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


class StringIndex(int):
    '''
    Index of the string in the strings table.
    '''
    def to_json(self) -> int:
        return self

    @classmethod
    def from_json(cls, json: int) -> StringIndex:
        return cls(json)

    def __repr__(self):
        return 'StringIndex({})'.format(super().__repr__())


class ArrayOfStrings(list):
    '''
    Index of the string in the strings table.
    '''
    def to_json(self) -> typing.List[StringIndex]:
        return self

    @classmethod
    def from_json(cls, json: typing.List[StringIndex]) -> ArrayOfStrings:
        return cls(json)

    def __repr__(self):
        return 'ArrayOfStrings({})'.format(super().__repr__())


@dataclass
class RareStringData:
    '''
    Data that is only present on rare nodes.
    '''
    index: typing.List[int]

    value: typing.List[StringIndex]

    def to_json(self):
        json = dict()
        json['index'] = [i for i in self.index]
        json['value'] = [i.to_json() for i in self.value]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            index=[int(i) for i in json['index']],
            value=[StringIndex.from_json(i) for i in json['value']],
        )


@dataclass
class RareBooleanData:
    index: typing.List[int]

    def to_json(self):
        json = dict()
        json['index'] = [i for i in self.index]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            index=[int(i) for i in json['index']],
        )


@dataclass
class RareIntegerData:
    index: typing.List[int]

    value: typing.List[int]

    def to_json(self):
        json = dict()
        json['index'] = [i for i in self.index]
        json['value'] = [i for i in self.value]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            index=[int(i) for i in json['index']],
            value=[int(i) for i in json['value']],
        )


class Rectangle(list):
    def to_json(self) -> typing.List[float]:
        return self

    @classmethod
    def from_json(cls, json: typing.List[float]) -> Rectangle:
        return cls(json)

    def __repr__(self):
        return 'Rectangle({})'.format(super().__repr__())


@dataclass
class DocumentSnapshot:
    '''
    Document snapshot.
    '''
    #: Document URL that ``Document`` or ``FrameOwner`` node points to.
    document_url: StringIndex

    #: Document title.
    title: StringIndex

    #: Base URL that ``Document`` or ``FrameOwner`` node uses for URL completion.
    base_url: StringIndex

    #: Contains the document's content language.
    content_language: StringIndex

    #: Contains the document's character set encoding.
    encoding_name: StringIndex

    #: ``DocumentType`` node's publicId.
    public_id: StringIndex

    #: ``DocumentType`` node's systemId.
    system_id: StringIndex

    #: Frame ID for frame owner elements and also for the document node.
    frame_id: StringIndex

    #: A table with dom nodes.
    nodes: NodeTreeSnapshot

    #: The nodes in the layout tree.
    layout: LayoutTreeSnapshot

    #: The post-layout inline text nodes.
    text_boxes: TextBoxSnapshot

    #: Horizontal scroll offset.
    scroll_offset_x: typing.Optional[float] = None

    #: Vertical scroll offset.
    scroll_offset_y: typing.Optional[float] = None

    #: Document content width.
    content_width: typing.Optional[float] = None

    #: Document content height.
    content_height: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['documentURL'] = self.document_url.to_json()
        json['title'] = self.title.to_json()
        json['baseURL'] = self.base_url.to_json()
        json['contentLanguage'] = self.content_language.to_json()
        json['encodingName'] = self.encoding_name.to_json()
        json['publicId'] = self.public_id.to_json()
        json['systemId'] = self.system_id.to_json()
        json['frameId'] = self.frame_id.to_json()
        json['nodes'] = self.nodes.to_json()
        json['layout'] = self.layout.to_json()
        json['textBoxes'] = self.text_boxes.to_json()
        if self.scroll_offset_x is not None:
            json['scrollOffsetX'] = self.scroll_offset_x
        if self.scroll_offset_y is not None:
            json['scrollOffsetY'] = self.scroll_offset_y
        if self.content_width is not None:
            json['contentWidth'] = self.content_width
        if self.content_height is not None:
            json['contentHeight'] = self.content_height
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            document_url=StringIndex.from_json(json['documentURL']),
            title=StringIndex.from_json(json['title']),
            base_url=StringIndex.from_json(json['baseURL']),
            content_language=StringIndex.from_json(json['contentLanguage']),
            encoding_name=StringIndex.from_json(json['encodingName']),
            public_id=StringIndex.from_json(json['publicId']),
            system_id=StringIndex.from_json(json['systemId']),
            frame_id=StringIndex.from_json(json['frameId']),
            nodes=NodeTreeSnapshot.from_json(json['nodes']),
            layout=LayoutTreeSnapshot.from_json(json['layout']),
            text_boxes=TextBoxSnapshot.from_json(json['textBoxes']),
            scroll_offset_x=float(json['scrollOffsetX']) if 'scrollOffsetX' in json else None,
            scroll_offset_y=float(json['scrollOffsetY']) if 'scrollOffsetY' in json else None,
            content_width=float(json['contentWidth']) if 'contentWidth' in json else None,
            content_height=float(json['contentHeight']) if 'contentHeight' in json else None,
        )


@dataclass
class NodeTreeSnapshot:
    '''
    Table containing nodes.
    '''
    #: Parent node index.
    parent_index: typing.Optional[typing.List[int]] = None

    #: ``Node``'s nodeType.
    node_type: typing.Optional[typing.List[int]] = None

    #: Type of the shadow root the ``Node`` is in. String values are equal to the ``ShadowRootType`` enum.
    shadow_root_type: typing.Optional[RareStringData] = None

    #: ``Node``'s nodeName.
    node_name: typing.Optional[typing.List[StringIndex]] = None

    #: ``Node``'s nodeValue.
    node_value: typing.Optional[typing.List[StringIndex]] = None

    #: ``Node``'s id, corresponds to DOM.Node.backendNodeId.
    backend_node_id: typing.Optional[typing.List[dom.BackendNodeId]] = None

    #: Attributes of an ``Element`` node. Flatten name, value pairs.
    attributes: typing.Optional[typing.List[ArrayOfStrings]] = None

    #: Only set for textarea elements, contains the text value.
    text_value: typing.Optional[RareStringData] = None

    #: Only set for input elements, contains the input's associated text value.
    input_value: typing.Optional[RareStringData] = None

    #: Only set for radio and checkbox input elements, indicates if the element has been checked
    input_checked: typing.Optional[RareBooleanData] = None

    #: Only set for option elements, indicates if the element has been selected
    option_selected: typing.Optional[RareBooleanData] = None

    #: The index of the document in the list of the snapshot documents.
    content_document_index: typing.Optional[RareIntegerData] = None

    #: Type of a pseudo element node.
    pseudo_type: typing.Optional[RareStringData] = None

    #: Pseudo element identifier for this node. Only present if there is a
    #: valid pseudoType.
    pseudo_identifier: typing.Optional[RareStringData] = None

    #: Whether this DOM node responds to mouse clicks. This includes nodes that have had click
    #: event listeners attached via JavaScript as well as anchor tags that naturally navigate when
    #: clicked.
    is_clickable: typing.Optional[RareBooleanData] = None

    #: The selected url for nodes with a srcset attribute.
    current_source_url: typing.Optional[RareStringData] = None

    #: The url of the script (if any) that generates this node.
    origin_url: typing.Optional[RareStringData] = None

    def to_json(self):
        json = dict()
        if self.parent_index is not None:
            json['parentIndex'] = [i for i in self.parent_index]
        if self.node_type is not None:
            json['nodeType'] = [i for i in self.node_type]
        if self.shadow_root_type is not None:
            json['shadowRootType'] = self.shadow_root_type.to_json()
        if self.node_name is not None:
            json['nodeName'] = [i.to_json() for i in self.node_name]
        if self.node_value is not None:
            json['nodeValue'] = [i.to_json() for i in self.node_value]
        if self.backend_node_id is not None:
            json['backendNodeId'] = [i.to_json() for i in self.backend_node_id]
        if self.attributes is not None:
            json['attributes'] = [i.to_json() for i in self.attributes]
        if self.text_value is not None:
            json['textValue'] = self.text_value.to_json()
        if self.input_value is not None:
            json['inputValue'] = self.input_value.to_json()
        if self.input_checked is not None:
            json['inputChecked'] = self.input_checked.to_json()
        if self.option_selected is not None:
            json['optionSelected'] = self.option_selected.to_json()
        if self.content_document_index is not None:
            json['contentDocumentIndex'] = self.content_document_index.to_json()
        if self.pseudo_type is not None:
            json['pseudoType'] = self.pseudo_type.to_json()
        if self.pseudo_identifier is not None:
            json['pseudoIdentifier'] = self.pseudo_identifier.to_json()
        if self.is_clickable is not None:
            json['isClickable'] = self.is_clickable.to_json()
        if self.current_source_url is not None:
            json['currentSourceURL'] = self.current_source_url.to_json()
        if self.origin_url is not None:
            json['originURL'] = self.origin_url.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            parent_index=[int(i) for i in json['parentIndex']] if 'parentIndex' in json else None,
            node_type=[int(i) for i in json['nodeType']] if 'nodeType' in json else None,
            shadow_root_type=RareStringData.from_json(json['shadowRootType']) if 'shadowRootType' in json else None,
            node_name=[StringIndex.from_json(i) for i in json['nodeName']] if 'nodeName' in json else None,
            node_value=[StringIndex.from_json(i) for i in json['nodeValue']] if 'nodeValue' in json else None,
            backend_node_id=[dom.BackendNodeId.from_json(i) for i in json['backendNodeId']] if 'backendNodeId' in json else None,
            attributes=[ArrayOfStrings.from_json(i) for i in json['attributes']] if 'attributes' in json else None,
            text_value=RareStringData.from_json(json['textValue']) if 'textValue' in json else None,
            input_value=RareStringData.from_json(json['inputValue']) if 'inputValue' in json else None,
            input_checked=RareBooleanData.from_json(json['inputChecked']) if 'inputChecked' in json else None,
            option_selected=RareBooleanData.from_json(json['optionSelected']) if 'optionSelected' in json else None,
            content_document_index=RareIntegerData.from_json(json['contentDocumentIndex']) if 'contentDocumentIndex' in json else None,
            pseudo_type=RareStringData.from_json(json['pseudoType']) if 'pseudoType' in json else None,
            pseudo_identifier=RareStringData.from_json(json['pseudoIdentifier']) if 'pseudoIdentifier' in json else None,
            is_clickable=RareBooleanData.from_json(json['isClickable']) if 'isClickable' in json else None,
            current_source_url=RareStringData.from_json(json['currentSourceURL']) if 'currentSourceURL' in json else None,
            origin_url=RareStringData.from_json(json['originURL']) if 'originURL' in json else None,
        )


@dataclass
class LayoutTreeSnapshot:
    '''
    Table of details of an element in the DOM tree with a LayoutObject.
    '''
    #: Index of the corresponding node in the ``NodeTreeSnapshot`` array returned by ``captureSnapshot``.
    node_index: typing.List[int]

    #: Array of indexes specifying computed style strings, filtered according to the ``computedStyles`` parameter passed to ``captureSnapshot``.
    styles: typing.List[ArrayOfStrings]

    #: The absolute position bounding box.
    bounds: typing.List[Rectangle]

    #: Contents of the LayoutText, if any.
    text: typing.List[StringIndex]

    #: Stacking context information.
    stacking_contexts: RareBooleanData

    #: Global paint order index, which is determined by the stacking order of the nodes. Nodes
    #: that are painted together will have the same index. Only provided if includePaintOrder in
    #: captureSnapshot was true.
    paint_orders: typing.Optional[typing.List[int]] = None

    #: The offset rect of nodes. Only available when includeDOMRects is set to true
    offset_rects: typing.Optional[typing.List[Rectangle]] = None

    #: The scroll rect of nodes. Only available when includeDOMRects is set to true
    scroll_rects: typing.Optional[typing.List[Rectangle]] = None

    #: The client rect of nodes. Only available when includeDOMRects is set to true
    client_rects: typing.Optional[typing.List[Rectangle]] = None

    #: The list of background colors that are blended with colors of overlapping elements.
    blended_background_colors: typing.Optional[typing.List[StringIndex]] = None

    #: The list of computed text opacities.
    text_color_opacities: typing.Optional[typing.List[float]] = None

    def to_json(self):
        json = dict()
        json['nodeIndex'] = [i for i in self.node_index]
        json['styles'] = [i.to_json() for i in self.styles]
        json['bounds'] = [i.to_json() for i in self.bounds]
        json['text'] = [i.to_json() for i in self.text]
        json['stackingContexts'] = self.stacking_contexts.to_json()
        if self.paint_orders is not None:
            json['paintOrders'] = [i for i in self.paint_orders]
        if self.offset_rects is not None:
            json['offsetRects'] = [i.to_json() for i in self.offset_rects]
        if self.scroll_rects is not None:
            json['scrollRects'] = [i.to_json() for i in self.scroll_rects]
        if self.client_rects is not None:
            json['clientRects'] = [i.to_json() for i in self.client_rects]
        if self.blended_background_colors is not None:
            json['blendedBackgroundColors'] = [i.to_json() for i in self.blended_background_colors]
        if self.text_color_opacities is not None:
            json['textColorOpacities'] = [i for i in self.text_color_opacities]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            node_index=[int(i) for i in json['nodeIndex']],
            styles=[ArrayOfStrings.from_json(i) for i in json['styles']],
            bounds=[Rectangle.from_json(i) for i in json['bounds']],
            text=[StringIndex.from_json(i) for i in json['text']],
            stacking_contexts=RareBooleanData.from_json(json['stackingContexts']),
            paint_orders=[int(i) for i in json['paintOrders']] if 'paintOrders' in json else None,
            offset_rects=[Rectangle.from_json(i) for i in json['offsetRects']] if 'offsetRects' in json else None,
            scroll_rects=[Rectangle.from_json(i) for i in json['scrollRects']] if 'scrollRects' in json else None,
            client_rects=[Rectangle.from_json(i) for i in json['clientRects']] if 'clientRects' in json else None,
            blended_background_colors=[StringIndex.from_json(i) for i in json['blendedBackgroundColors']] if 'blendedBackgroundColors' in json else None,
            text_color_opacities=[float(i) for i in json['textColorOpacities']] if 'textColorOpacities' in json else None,
        )


@dataclass
class TextBoxSnapshot:
    '''
    Table of details of the post layout rendered text positions. The exact layout should not be regarded as
    stable and may change between versions.
    '''
    #: Index of the layout tree node that owns this box collection.
    layout_index: typing.List[int]

    #: The absolute position bounding box.
    bounds: typing.List[Rectangle]

    #: The starting index in characters, for this post layout textbox substring. Characters that
    #: would be represented as a surrogate pair in UTF-16 have length 2.
    start: typing.List[int]

    #: The number of characters in this post layout textbox substring. Characters that would be
    #: represented as a surrogate pair in UTF-16 have length 2.
    length: typing.List[int]

    def to_json(self):
        json = dict()
        json['layoutIndex'] = [i for i in self.layout_index]
        json['bounds'] = [i.to_json() for i in self.bounds]
        json['start'] = [i for i in self.start]
        json['length'] = [i for i in self.length]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            layout_index=[int(i) for i in json['layoutIndex']],
            bounds=[Rectangle.from_json(i) for i in json['bounds']],
            start=[int(i) for i in json['start']],
            length=[int(i) for i in json['length']],
        )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables DOM snapshot agent for the given page.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMSnapshot.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables DOM snapshot agent for the given page.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMSnapshot.enable',
    }
    json = yield cmd_dict


def get_snapshot(
        computed_style_whitelist: typing.List[str],
        include_event_listeners: typing.Optional[bool] = None,
        include_paint_order: typing.Optional[bool] = None,
        include_user_agent_shadow_tree: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[DOMNode], typing.List[LayoutTreeNode], typing.List[ComputedStyle]]]:
    '''
    Returns a document snapshot, including the full DOM tree of the root node (including iframes,
    template contents, and imported documents) in a flattened array, as well as layout and
    white-listed computed style information for the nodes. Shadow DOM in the returned DOM tree is
    flattened.

    :param computed_style_whitelist: Whitelist of computed styles to return.
    :param include_event_listeners: *(Optional)* Whether or not to retrieve details of DOM listeners (default false).
    :param include_paint_order: *(Optional)* Whether to determine and include the paint order index of LayoutTreeNodes (default false).
    :param include_user_agent_shadow_tree: *(Optional)* Whether to include UA shadow tree in the snapshot (default false).
    :returns: A tuple with the following items:

        0. **domNodes** - The nodes in the DOM tree. The DOMNode at index 0 corresponds to the root document.
        1. **layoutTreeNodes** - The nodes in the layout tree.
        2. **computedStyles** - Whitelisted ComputedStyle properties for each node in the layout tree.
    '''
    params: T_JSON_DICT = dict()
    params['computedStyleWhitelist'] = [i for i in computed_style_whitelist]
    if include_event_listeners is not None:
        params['includeEventListeners'] = include_event_listeners
    if include_paint_order is not None:
        params['includePaintOrder'] = include_paint_order
    if include_user_agent_shadow_tree is not None:
        params['includeUserAgentShadowTree'] = include_user_agent_shadow_tree
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMSnapshot.getSnapshot',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [DOMNode.from_json(i) for i in json['domNodes']],
        [LayoutTreeNode.from_json(i) for i in json['layoutTreeNodes']],
        [ComputedStyle.from_json(i) for i in json['computedStyles']]
    )


def capture_snapshot(
        computed_styles: typing.List[str],
        include_paint_order: typing.Optional[bool] = None,
        include_dom_rects: typing.Optional[bool] = None,
        include_blended_background_colors: typing.Optional[bool] = None,
        include_text_color_opacities: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[DocumentSnapshot], typing.List[str]]]:
    '''
    Returns a document snapshot, including the full DOM tree of the root node (including iframes,
    template contents, and imported documents) in a flattened array, as well as layout and
    white-listed computed style information for the nodes. Shadow DOM in the returned DOM tree is
    flattened.

    :param computed_styles: Whitelist of computed styles to return.
    :param include_paint_order: *(Optional)* Whether to include layout object paint orders into the snapshot.
    :param include_dom_rects: *(Optional)* Whether to include DOM rectangles (offsetRects, clientRects, scrollRects) into the snapshot
    :param include_blended_background_colors: **(EXPERIMENTAL)** *(Optional)* Whether to include blended background colors in the snapshot (default: false). Blended background color is achieved by blending background colors of all elements that overlap with the current element.
    :param include_text_color_opacities: **(EXPERIMENTAL)** *(Optional)* Whether to include text color opacity in the snapshot (default: false). An element might have the opacity property set that affects the text color of the element. The final text color opacity is computed based on the opacity of all overlapping elements.
    :returns: A tuple with the following items:

        0. **documents** - The nodes in the DOM tree. The DOMNode at index 0 corresponds to the root document.
        1. **strings** - Shared string table that all string properties refer to with indexes.
    '''
    params: T_JSON_DICT = dict()
    params['computedStyles'] = [i for i in computed_styles]
    if include_paint_order is not None:
        params['includePaintOrder'] = include_paint_order
    if include_dom_rects is not None:
        params['includeDOMRects'] = include_dom_rects
    if include_blended_background_colors is not None:
        params['includeBlendedBackgroundColors'] = include_blended_background_colors
    if include_text_color_opacities is not None:
        params['includeTextColorOpacities'] = include_text_color_opacities
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMSnapshot.captureSnapshot',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [DocumentSnapshot.from_json(i) for i in json['documents']],
        [str(i) for i in json['strings']]
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\dom_snapshot.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: DOMSnapshot (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import dom_debugger
from . import page


@dataclass
class DOMNode:
    '''
    A Node in the DOM tree.
    '''
    #: ``Node``'s nodeType.
    node_type: int

    #: ``Node``'s nodeName.
    node_name: str

    #: ``Node``'s nodeValue.
    node_value: str

    #: ``Node``'s id, corresponds to DOM.Node.backendNodeId.
    backend_node_id: dom.BackendNodeId

    #: Only set for textarea elements, contains the text value.
    text_value: typing.Optional[str] = None

    #: Only set for input elements, contains the input's associated text value.
    input_value: typing.Optional[str] = None

    #: Only set for radio and checkbox input elements, indicates if the element has been checked
    input_checked: typing.Optional[bool] = None

    #: Only set for option elements, indicates if the element has been selected
    option_selected: typing.Optional[bool] = None

    #: The indexes of the node's child nodes in the ``domNodes`` array returned by ``getSnapshot``, if
    #: any.
    child_node_indexes: typing.Optional[typing.List[int]] = None

    #: Attributes of an ``Element`` node.
    attributes: typing.Optional[typing.List[NameValue]] = None

    #: Indexes of pseudo elements associated with this node in the ``domNodes`` array returned by
    #: ``getSnapshot``, if any.
    pseudo_element_indexes: typing.Optional[typing.List[int]] = None

    #: The index of the node's related layout tree node in the ``layoutTreeNodes`` array returned by
    #: ``getSnapshot``, if any.
    layout_node_index: typing.Optional[int] = None

    #: Document URL that ``Document`` or ``FrameOwner`` node points to.
    document_url: typing.Optional[str] = None

    #: Base URL that ``Document`` or ``FrameOwner`` node uses for URL completion.
    base_url: typing.Optional[str] = None

    #: Only set for documents, contains the document's content language.
    content_language: typing.Optional[str] = None

    #: Only set for documents, contains the document's character set encoding.
    document_encoding: typing.Optional[str] = None

    #: ``DocumentType`` node's publicId.
    public_id: typing.Optional[str] = None

    #: ``DocumentType`` node's systemId.
    system_id: typing.Optional[str] = None

    #: Frame ID for frame owner elements and also for the document node.
    frame_id: typing.Optional[page.FrameId] = None

    #: The index of a frame owner element's content document in the ``domNodes`` array returned by
    #: ``getSnapshot``, if any.
    content_document_index: typing.Optional[int] = None

    #: Type of a pseudo element node.
    pseudo_type: typing.Optional[dom.PseudoType] = None

    #: Shadow root type.
    shadow_root_type: typing.Optional[dom.ShadowRootType] = None

    #: Whether this DOM node responds to mouse clicks. This includes nodes that have had click
    #: event listeners attached via JavaScript as well as anchor tags that naturally navigate when
    #: clicked.
    is_clickable: typing.Optional[bool] = None

    #: Details of the node's event listeners, if any.
    event_listeners: typing.Optional[typing.List[dom_debugger.EventListener]] = None

    #: The selected url for nodes with a srcset attribute.
    current_source_url: typing.Optional[str] = None

    #: The url of the script (if any) that generates this node.
    origin_url: typing.Optional[str] = None

    #: Scroll offsets, set when this node is a Document.
    scroll_offset_x: typing.Optional[float] = None

    scroll_offset_y: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['nodeType'] = self.node_type
        json['nodeName'] = self.node_name
        json['nodeValue'] = self.node_value
        json['backendNodeId'] = self.backend_node_id.to_json()
        if self.text_value is not None:
            json['textValue'] = self.text_value
        if self.input_value is not None:
            json['inputValue'] = self.input_value
        if self.input_checked is not None:
            json['inputChecked'] = self.input_checked
        if self.option_selected is not None:
            json['optionSelected'] = self.option_selected
        if self.child_node_indexes is not None:
            json['childNodeIndexes'] = [i for i in self.child_node_indexes]
        if self.attributes is not None:
            json['attributes'] = [i.to_json() for i in self.attributes]
        if self.pseudo_element_indexes is not None:
            json['pseudoElementIndexes'] = [i for i in self.pseudo_element_indexes]
        if self.layout_node_index is not None:
            json['layoutNodeIndex'] = self.layout_node_index
        if self.document_url is not None:
            json['documentURL'] = self.document_url
        if self.base_url is not None:
            json['baseURL'] = self.base_url
        if self.content_language is not None:
            json['contentLanguage'] = self.content_language
        if self.document_encoding is not None:
            json['documentEncoding'] = self.document_encoding
        if self.public_id is not None:
            json['publicId'] = self.public_id
        if self.system_id is not None:
            json['systemId'] = self.system_id
        if self.frame_id is not None:
            json['frameId'] = self.frame_id.to_json()
        if self.content_document_index is not None:
            json['contentDocumentIndex'] = self.content_document_index
        if self.pseudo_type is not None:
            json['pseudoType'] = self.pseudo_type.to_json()
        if self.shadow_root_type is not None:
            json['shadowRootType'] = self.shadow_root_type.to_json()
        if self.is_clickable is not None:
            json['isClickable'] = self.is_clickable
        if self.event_listeners is not None:
            json['eventListeners'] = [i.to_json() for i in self.event_listeners]
        if self.current_source_url is not None:
            json['currentSourceURL'] = self.current_source_url
        if self.origin_url is not None:
            json['originURL'] = self.origin_url
        if self.scroll_offset_x is not None:
            json['scrollOffsetX'] = self.scroll_offset_x
        if self.scroll_offset_y is not None:
            json['scrollOffsetY'] = self.scroll_offset_y
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            node_type=int(json['nodeType']),
            node_name=str(json['nodeName']),
            node_value=str(json['nodeValue']),
            backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId']),
            text_value=str(json['textValue']) if 'textValue' in json else None,
            input_value=str(json['inputValue']) if 'inputValue' in json else None,
            input_checked=bool(json['inputChecked']) if 'inputChecked' in json else None,
            option_selected=bool(json['optionSelected']) if 'optionSelected' in json else None,
            child_node_indexes=[int(i) for i in json['childNodeIndexes']] if 'childNodeIndexes' in json else None,
            attributes=[NameValue.from_json(i) for i in json['attributes']] if 'attributes' in json else None,
            pseudo_element_indexes=[int(i) for i in json['pseudoElementIndexes']] if 'pseudoElementIndexes' in json else None,
            layout_node_index=int(json['layoutNodeIndex']) if 'layoutNodeIndex' in json else None,
            document_url=str(json['documentURL']) if 'documentURL' in json else None,
            base_url=str(json['baseURL']) if 'baseURL' in json else None,
            content_language=str(json['contentLanguage']) if 'contentLanguage' in json else None,
            document_encoding=str(json['documentEncoding']) if 'documentEncoding' in json else None,
            public_id=str(json['publicId']) if 'publicId' in json else None,
            system_id=str(json['systemId']) if 'systemId' in json else None,
            frame_id=page.FrameId.from_json(json['frameId']) if 'frameId' in json else None,
            content_document_index=int(json['contentDocumentIndex']) if 'contentDocumentIndex' in json else None,
            pseudo_type=dom.PseudoType.from_json(json['pseudoType']) if 'pseudoType' in json else None,
            shadow_root_type=dom.ShadowRootType.from_json(json['shadowRootType']) if 'shadowRootType' in json else None,
            is_clickable=bool(json['isClickable']) if 'isClickable' in json else None,
            event_listeners=[dom_debugger.EventListener.from_json(i) for i in json['eventListeners']] if 'eventListeners' in json else None,
            current_source_url=str(json['currentSourceURL']) if 'currentSourceURL' in json else None,
            origin_url=str(json['originURL']) if 'originURL' in json else None,
            scroll_offset_x=float(json['scrollOffsetX']) if 'scrollOffsetX' in json else None,
            scroll_offset_y=float(json['scrollOffsetY']) if 'scrollOffsetY' in json else None,
        )


@dataclass
class InlineTextBox:
    '''
    Details of post layout rendered text positions. The exact layout should not be regarded as
    stable and may change between versions.
    '''
    #: The bounding box in document coordinates. Note that scroll offset of the document is ignored.
    bounding_box: dom.Rect

    #: The starting index in characters, for this post layout textbox substring. Characters that
    #: would be represented as a surrogate pair in UTF-16 have length 2.
    start_character_index: int

    #: The number of characters in this post layout textbox substring. Characters that would be
    #: represented as a surrogate pair in UTF-16 have length 2.
    num_characters: int

    def to_json(self):
        json = dict()
        json['boundingBox'] = self.bounding_box.to_json()
        json['startCharacterIndex'] = self.start_character_index
        json['numCharacters'] = self.num_characters
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            bounding_box=dom.Rect.from_json(json['boundingBox']),
            start_character_index=int(json['startCharacterIndex']),
            num_characters=int(json['numCharacters']),
        )


@dataclass
class LayoutTreeNode:
    '''
    Details of an element in the DOM tree with a LayoutObject.
    '''
    #: The index of the related DOM node in the ``domNodes`` array returned by ``getSnapshot``.
    dom_node_index: int

    #: The bounding box in document coordinates. Note that scroll offset of the document is ignored.
    bounding_box: dom.Rect

    #: Contents of the LayoutText, if any.
    layout_text: typing.Optional[str] = None

    #: The post-layout inline text nodes, if any.
    inline_text_nodes: typing.Optional[typing.List[InlineTextBox]] = None

    #: Index into the ``computedStyles`` array returned by ``getSnapshot``.
    style_index: typing.Optional[int] = None

    #: Global paint order index, which is determined by the stacking order of the nodes. Nodes
    #: that are painted together will have the same index. Only provided if includePaintOrder in
    #: getSnapshot was true.
    paint_order: typing.Optional[int] = None

    #: Set to true to indicate the element begins a new stacking context.
    is_stacking_context: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['domNodeIndex'] = self.dom_node_index
        json['boundingBox'] = self.bounding_box.to_json()
        if self.layout_text is not None:
            json['layoutText'] = self.layout_text
        if self.inline_text_nodes is not None:
            json['inlineTextNodes'] = [i.to_json() for i in self.inline_text_nodes]
        if self.style_index is not None:
            json['styleIndex'] = self.style_index
        if self.paint_order is not None:
            json['paintOrder'] = self.paint_order
        if self.is_stacking_context is not None:
            json['isStackingContext'] = self.is_stacking_context
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            dom_node_index=int(json['domNodeIndex']),
            bounding_box=dom.Rect.from_json(json['boundingBox']),
            layout_text=str(json['layoutText']) if 'layoutText' in json else None,
            inline_text_nodes=[InlineTextBox.from_json(i) for i in json['inlineTextNodes']] if 'inlineTextNodes' in json else None,
            style_index=int(json['styleIndex']) if 'styleIndex' in json else None,
            paint_order=int(json['paintOrder']) if 'paintOrder' in json else None,
            is_stacking_context=bool(json['isStackingContext']) if 'isStackingContext' in json else None,
        )


@dataclass
class ComputedStyle:
    '''
    A subset of the full ComputedStyle as defined by the request whitelist.
    '''
    #: Name/value pairs of computed style properties.
    properties: typing.List[NameValue]

    def to_json(self):
        json = dict()
        json['properties'] = [i.to_json() for i in self.properties]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            properties=[NameValue.from_json(i) for i in json['properties']],
        )


@dataclass
class NameValue:
    '''
    A name/value pair.
    '''
    #: Attribute/property name.
    name: str

    #: Attribute/property value.
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


class StringIndex(int):
    '''
    Index of the string in the strings table.
    '''
    def to_json(self) -> int:
        return self

    @classmethod
    def from_json(cls, json: int) -> StringIndex:
        return cls(json)

    def __repr__(self):
        return 'StringIndex({})'.format(super().__repr__())


class ArrayOfStrings(list):
    '''
    Index of the string in the strings table.
    '''
    def to_json(self) -> typing.List[StringIndex]:
        return self

    @classmethod
    def from_json(cls, json: typing.List[StringIndex]) -> ArrayOfStrings:
        return cls(json)

    def __repr__(self):
        return 'ArrayOfStrings({})'.format(super().__repr__())


@dataclass
class RareStringData:
    '''
    Data that is only present on rare nodes.
    '''
    index: typing.List[int]

    value: typing.List[StringIndex]

    def to_json(self):
        json = dict()
        json['index'] = [i for i in self.index]
        json['value'] = [i.to_json() for i in self.value]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            index=[int(i) for i in json['index']],
            value=[StringIndex.from_json(i) for i in json['value']],
        )


@dataclass
class RareBooleanData:
    index: typing.List[int]

    def to_json(self):
        json = dict()
        json['index'] = [i for i in self.index]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            index=[int(i) for i in json['index']],
        )


@dataclass
class RareIntegerData:
    index: typing.List[int]

    value: typing.List[int]

    def to_json(self):
        json = dict()
        json['index'] = [i for i in self.index]
        json['value'] = [i for i in self.value]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            index=[int(i) for i in json['index']],
            value=[int(i) for i in json['value']],
        )


class Rectangle(list):
    def to_json(self) -> typing.List[float]:
        return self

    @classmethod
    def from_json(cls, json: typing.List[float]) -> Rectangle:
        return cls(json)

    def __repr__(self):
        return 'Rectangle({})'.format(super().__repr__())


@dataclass
class DocumentSnapshot:
    '''
    Document snapshot.
    '''
    #: Document URL that ``Document`` or ``FrameOwner`` node points to.
    document_url: StringIndex

    #: Document title.
    title: StringIndex

    #: Base URL that ``Document`` or ``FrameOwner`` node uses for URL completion.
    base_url: StringIndex

    #: Contains the document's content language.
    content_language: StringIndex

    #: Contains the document's character set encoding.
    encoding_name: StringIndex

    #: ``DocumentType`` node's publicId.
    public_id: StringIndex

    #: ``DocumentType`` node's systemId.
    system_id: StringIndex

    #: Frame ID for frame owner elements and also for the document node.
    frame_id: StringIndex

    #: A table with dom nodes.
    nodes: NodeTreeSnapshot

    #: The nodes in the layout tree.
    layout: LayoutTreeSnapshot

    #: The post-layout inline text nodes.
    text_boxes: TextBoxSnapshot

    #: Horizontal scroll offset.
    scroll_offset_x: typing.Optional[float] = None

    #: Vertical scroll offset.
    scroll_offset_y: typing.Optional[float] = None

    #: Document content width.
    content_width: typing.Optional[float] = None

    #: Document content height.
    content_height: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['documentURL'] = self.document_url.to_json()
        json['title'] = self.title.to_json()
        json['baseURL'] = self.base_url.to_json()
        json['contentLanguage'] = self.content_language.to_json()
        json['encodingName'] = self.encoding_name.to_json()
        json['publicId'] = self.public_id.to_json()
        json['systemId'] = self.system_id.to_json()
        json['frameId'] = self.frame_id.to_json()
        json['nodes'] = self.nodes.to_json()
        json['layout'] = self.layout.to_json()
        json['textBoxes'] = self.text_boxes.to_json()
        if self.scroll_offset_x is not None:
            json['scrollOffsetX'] = self.scroll_offset_x
        if self.scroll_offset_y is not None:
            json['scrollOffsetY'] = self.scroll_offset_y
        if self.content_width is not None:
            json['contentWidth'] = self.content_width
        if self.content_height is not None:
            json['contentHeight'] = self.content_height
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            document_url=StringIndex.from_json(json['documentURL']),
            title=StringIndex.from_json(json['title']),
            base_url=StringIndex.from_json(json['baseURL']),
            content_language=StringIndex.from_json(json['contentLanguage']),
            encoding_name=StringIndex.from_json(json['encodingName']),
            public_id=StringIndex.from_json(json['publicId']),
            system_id=StringIndex.from_json(json['systemId']),
            frame_id=StringIndex.from_json(json['frameId']),
            nodes=NodeTreeSnapshot.from_json(json['nodes']),
            layout=LayoutTreeSnapshot.from_json(json['layout']),
            text_boxes=TextBoxSnapshot.from_json(json['textBoxes']),
            scroll_offset_x=float(json['scrollOffsetX']) if 'scrollOffsetX' in json else None,
            scroll_offset_y=float(json['scrollOffsetY']) if 'scrollOffsetY' in json else None,
            content_width=float(json['contentWidth']) if 'contentWidth' in json else None,
            content_height=float(json['contentHeight']) if 'contentHeight' in json else None,
        )


@dataclass
class NodeTreeSnapshot:
    '''
    Table containing nodes.
    '''
    #: Parent node index.
    parent_index: typing.Optional[typing.List[int]] = None

    #: ``Node``'s nodeType.
    node_type: typing.Optional[typing.List[int]] = None

    #: Type of the shadow root the ``Node`` is in. String values are equal to the ``ShadowRootType`` enum.
    shadow_root_type: typing.Optional[RareStringData] = None

    #: ``Node``'s nodeName.
    node_name: typing.Optional[typing.List[StringIndex]] = None

    #: ``Node``'s nodeValue.
    node_value: typing.Optional[typing.List[StringIndex]] = None

    #: ``Node``'s id, corresponds to DOM.Node.backendNodeId.
    backend_node_id: typing.Optional[typing.List[dom.BackendNodeId]] = None

    #: Attributes of an ``Element`` node. Flatten name, value pairs.
    attributes: typing.Optional[typing.List[ArrayOfStrings]] = None

    #: Only set for textarea elements, contains the text value.
    text_value: typing.Optional[RareStringData] = None

    #: Only set for input elements, contains the input's associated text value.
    input_value: typing.Optional[RareStringData] = None

    #: Only set for radio and checkbox input elements, indicates if the element has been checked
    input_checked: typing.Optional[RareBooleanData] = None

    #: Only set for option elements, indicates if the element has been selected
    option_selected: typing.Optional[RareBooleanData] = None

    #: The index of the document in the list of the snapshot documents.
    content_document_index: typing.Optional[RareIntegerData] = None

    #: Type of a pseudo element node.
    pseudo_type: typing.Optional[RareStringData] = None

    #: Pseudo element identifier for this node. Only present if there is a
    #: valid pseudoType.
    pseudo_identifier: typing.Optional[RareStringData] = None

    #: Whether this DOM node responds to mouse clicks. This includes nodes that have had click
    #: event listeners attached via JavaScript as well as anchor tags that naturally navigate when
    #: clicked.
    is_clickable: typing.Optional[RareBooleanData] = None

    #: The selected url for nodes with a srcset attribute.
    current_source_url: typing.Optional[RareStringData] = None

    #: The url of the script (if any) that generates this node.
    origin_url: typing.Optional[RareStringData] = None

    def to_json(self):
        json = dict()
        if self.parent_index is not None:
            json['parentIndex'] = [i for i in self.parent_index]
        if self.node_type is not None:
            json['nodeType'] = [i for i in self.node_type]
        if self.shadow_root_type is not None:
            json['shadowRootType'] = self.shadow_root_type.to_json()
        if self.node_name is not None:
            json['nodeName'] = [i.to_json() for i in self.node_name]
        if self.node_value is not None:
            json['nodeValue'] = [i.to_json() for i in self.node_value]
        if self.backend_node_id is not None:
            json['backendNodeId'] = [i.to_json() for i in self.backend_node_id]
        if self.attributes is not None:
            json['attributes'] = [i.to_json() for i in self.attributes]
        if self.text_value is not None:
            json['textValue'] = self.text_value.to_json()
        if self.input_value is not None:
            json['inputValue'] = self.input_value.to_json()
        if self.input_checked is not None:
            json['inputChecked'] = self.input_checked.to_json()
        if self.option_selected is not None:
            json['optionSelected'] = self.option_selected.to_json()
        if self.content_document_index is not None:
            json['contentDocumentIndex'] = self.content_document_index.to_json()
        if self.pseudo_type is not None:
            json['pseudoType'] = self.pseudo_type.to_json()
        if self.pseudo_identifier is not None:
            json['pseudoIdentifier'] = self.pseudo_identifier.to_json()
        if self.is_clickable is not None:
            json['isClickable'] = self.is_clickable.to_json()
        if self.current_source_url is not None:
            json['currentSourceURL'] = self.current_source_url.to_json()
        if self.origin_url is not None:
            json['originURL'] = self.origin_url.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            parent_index=[int(i) for i in json['parentIndex']] if 'parentIndex' in json else None,
            node_type=[int(i) for i in json['nodeType']] if 'nodeType' in json else None,
            shadow_root_type=RareStringData.from_json(json['shadowRootType']) if 'shadowRootType' in json else None,
            node_name=[StringIndex.from_json(i) for i in json['nodeName']] if 'nodeName' in json else None,
            node_value=[StringIndex.from_json(i) for i in json['nodeValue']] if 'nodeValue' in json else None,
            backend_node_id=[dom.BackendNodeId.from_json(i) for i in json['backendNodeId']] if 'backendNodeId' in json else None,
            attributes=[ArrayOfStrings.from_json(i) for i in json['attributes']] if 'attributes' in json else None,
            text_value=RareStringData.from_json(json['textValue']) if 'textValue' in json else None,
            input_value=RareStringData.from_json(json['inputValue']) if 'inputValue' in json else None,
            input_checked=RareBooleanData.from_json(json['inputChecked']) if 'inputChecked' in json else None,
            option_selected=RareBooleanData.from_json(json['optionSelected']) if 'optionSelected' in json else None,
            content_document_index=RareIntegerData.from_json(json['contentDocumentIndex']) if 'contentDocumentIndex' in json else None,
            pseudo_type=RareStringData.from_json(json['pseudoType']) if 'pseudoType' in json else None,
            pseudo_identifier=RareStringData.from_json(json['pseudoIdentifier']) if 'pseudoIdentifier' in json else None,
            is_clickable=RareBooleanData.from_json(json['isClickable']) if 'isClickable' in json else None,
            current_source_url=RareStringData.from_json(json['currentSourceURL']) if 'currentSourceURL' in json else None,
            origin_url=RareStringData.from_json(json['originURL']) if 'originURL' in json else None,
        )


@dataclass
class LayoutTreeSnapshot:
    '''
    Table of details of an element in the DOM tree with a LayoutObject.
    '''
    #: Index of the corresponding node in the ``NodeTreeSnapshot`` array returned by ``captureSnapshot``.
    node_index: typing.List[int]

    #: Array of indexes specifying computed style strings, filtered according to the ``computedStyles`` parameter passed to ``captureSnapshot``.
    styles: typing.List[ArrayOfStrings]

    #: The absolute position bounding box.
    bounds: typing.List[Rectangle]

    #: Contents of the LayoutText, if any.
    text: typing.List[StringIndex]

    #: Stacking context information.
    stacking_contexts: RareBooleanData

    #: Global paint order index, which is determined by the stacking order of the nodes. Nodes
    #: that are painted together will have the same index. Only provided if includePaintOrder in
    #: captureSnapshot was true.
    paint_orders: typing.Optional[typing.List[int]] = None

    #: The offset rect of nodes. Only available when includeDOMRects is set to true
    offset_rects: typing.Optional[typing.List[Rectangle]] = None

    #: The scroll rect of nodes. Only available when includeDOMRects is set to true
    scroll_rects: typing.Optional[typing.List[Rectangle]] = None

    #: The client rect of nodes. Only available when includeDOMRects is set to true
    client_rects: typing.Optional[typing.List[Rectangle]] = None

    #: The list of background colors that are blended with colors of overlapping elements.
    blended_background_colors: typing.Optional[typing.List[StringIndex]] = None

    #: The list of computed text opacities.
    text_color_opacities: typing.Optional[typing.List[float]] = None

    def to_json(self):
        json = dict()
        json['nodeIndex'] = [i for i in self.node_index]
        json['styles'] = [i.to_json() for i in self.styles]
        json['bounds'] = [i.to_json() for i in self.bounds]
        json['text'] = [i.to_json() for i in self.text]
        json['stackingContexts'] = self.stacking_contexts.to_json()
        if self.paint_orders is not None:
            json['paintOrders'] = [i for i in self.paint_orders]
        if self.offset_rects is not None:
            json['offsetRects'] = [i.to_json() for i in self.offset_rects]
        if self.scroll_rects is not None:
            json['scrollRects'] = [i.to_json() for i in self.scroll_rects]
        if self.client_rects is not None:
            json['clientRects'] = [i.to_json() for i in self.client_rects]
        if self.blended_background_colors is not None:
            json['blendedBackgroundColors'] = [i.to_json() for i in self.blended_background_colors]
        if self.text_color_opacities is not None:
            json['textColorOpacities'] = [i for i in self.text_color_opacities]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            node_index=[int(i) for i in json['nodeIndex']],
            styles=[ArrayOfStrings.from_json(i) for i in json['styles']],
            bounds=[Rectangle.from_json(i) for i in json['bounds']],
            text=[StringIndex.from_json(i) for i in json['text']],
            stacking_contexts=RareBooleanData.from_json(json['stackingContexts']),
            paint_orders=[int(i) for i in json['paintOrders']] if 'paintOrders' in json else None,
            offset_rects=[Rectangle.from_json(i) for i in json['offsetRects']] if 'offsetRects' in json else None,
            scroll_rects=[Rectangle.from_json(i) for i in json['scrollRects']] if 'scrollRects' in json else None,
            client_rects=[Rectangle.from_json(i) for i in json['clientRects']] if 'clientRects' in json else None,
            blended_background_colors=[StringIndex.from_json(i) for i in json['blendedBackgroundColors']] if 'blendedBackgroundColors' in json else None,
            text_color_opacities=[float(i) for i in json['textColorOpacities']] if 'textColorOpacities' in json else None,
        )


@dataclass
class TextBoxSnapshot:
    '''
    Table of details of the post layout rendered text positions. The exact layout should not be regarded as
    stable and may change between versions.
    '''
    #: Index of the layout tree node that owns this box collection.
    layout_index: typing.List[int]

    #: The absolute position bounding box.
    bounds: typing.List[Rectangle]

    #: The starting index in characters, for this post layout textbox substring. Characters that
    #: would be represented as a surrogate pair in UTF-16 have length 2.
    start: typing.List[int]

    #: The number of characters in this post layout textbox substring. Characters that would be
    #: represented as a surrogate pair in UTF-16 have length 2.
    length: typing.List[int]

    def to_json(self):
        json = dict()
        json['layoutIndex'] = [i for i in self.layout_index]
        json['bounds'] = [i.to_json() for i in self.bounds]
        json['start'] = [i for i in self.start]
        json['length'] = [i for i in self.length]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            layout_index=[int(i) for i in json['layoutIndex']],
            bounds=[Rectangle.from_json(i) for i in json['bounds']],
            start=[int(i) for i in json['start']],
            length=[int(i) for i in json['length']],
        )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables DOM snapshot agent for the given page.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMSnapshot.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables DOM snapshot agent for the given page.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMSnapshot.enable',
    }
    json = yield cmd_dict


def get_snapshot(
        computed_style_whitelist: typing.List[str],
        include_event_listeners: typing.Optional[bool] = None,
        include_paint_order: typing.Optional[bool] = None,
        include_user_agent_shadow_tree: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[DOMNode], typing.List[LayoutTreeNode], typing.List[ComputedStyle]]]:
    '''
    Returns a document snapshot, including the full DOM tree of the root node (including iframes,
    template contents, and imported documents) in a flattened array, as well as layout and
    white-listed computed style information for the nodes. Shadow DOM in the returned DOM tree is
    flattened.

    :param computed_style_whitelist: Whitelist of computed styles to return.
    :param include_event_listeners: *(Optional)* Whether or not to retrieve details of DOM listeners (default false).
    :param include_paint_order: *(Optional)* Whether to determine and include the paint order index of LayoutTreeNodes (default false).
    :param include_user_agent_shadow_tree: *(Optional)* Whether to include UA shadow tree in the snapshot (default false).
    :returns: A tuple with the following items:

        0. **domNodes** - The nodes in the DOM tree. The DOMNode at index 0 corresponds to the root document.
        1. **layoutTreeNodes** - The nodes in the layout tree.
        2. **computedStyles** - Whitelisted ComputedStyle properties for each node in the layout tree.
    '''
    params: T_JSON_DICT = dict()
    params['computedStyleWhitelist'] = [i for i in computed_style_whitelist]
    if include_event_listeners is not None:
        params['includeEventListeners'] = include_event_listeners
    if include_paint_order is not None:
        params['includePaintOrder'] = include_paint_order
    if include_user_agent_shadow_tree is not None:
        params['includeUserAgentShadowTree'] = include_user_agent_shadow_tree
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMSnapshot.getSnapshot',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [DOMNode.from_json(i) for i in json['domNodes']],
        [LayoutTreeNode.from_json(i) for i in json['layoutTreeNodes']],
        [ComputedStyle.from_json(i) for i in json['computedStyles']]
    )


def capture_snapshot(
        computed_styles: typing.List[str],
        include_paint_order: typing.Optional[bool] = None,
        include_dom_rects: typing.Optional[bool] = None,
        include_blended_background_colors: typing.Optional[bool] = None,
        include_text_color_opacities: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[DocumentSnapshot], typing.List[str]]]:
    '''
    Returns a document snapshot, including the full DOM tree of the root node (including iframes,
    template contents, and imported documents) in a flattened array, as well as layout and
    white-listed computed style information for the nodes. Shadow DOM in the returned DOM tree is
    flattened.

    :param computed_styles: Whitelist of computed styles to return.
    :param include_paint_order: *(Optional)* Whether to include layout object paint orders into the snapshot.
    :param include_dom_rects: *(Optional)* Whether to include DOM rectangles (offsetRects, clientRects, scrollRects) into the snapshot
    :param include_blended_background_colors: **(EXPERIMENTAL)** *(Optional)* Whether to include blended background colors in the snapshot (default: false). Blended background color is achieved by blending background colors of all elements that overlap with the current element.
    :param include_text_color_opacities: **(EXPERIMENTAL)** *(Optional)* Whether to include text color opacity in the snapshot (default: false). An element might have the opacity property set that affects the text color of the element. The final text color opacity is computed based on the opacity of all overlapping elements.
    :returns: A tuple with the following items:

        0. **documents** - The nodes in the DOM tree. The DOMNode at index 0 corresponds to the root document.
        1. **strings** - Shared string table that all string properties refer to with indexes.
    '''
    params: T_JSON_DICT = dict()
    params['computedStyles'] = [i for i in computed_styles]
    if include_paint_order is not None:
        params['includePaintOrder'] = include_paint_order
    if include_dom_rects is not None:
        params['includeDOMRects'] = include_dom_rects
    if include_blended_background_colors is not None:
        params['includeBlendedBackgroundColors'] = include_blended_background_colors
    if include_text_color_opacities is not None:
        params['includeTextColorOpacities'] = include_text_color_opacities
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMSnapshot.captureSnapshot',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [DocumentSnapshot.from_json(i) for i in json['documents']],
        [str(i) for i in json['strings']]
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\dom_snapshot.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: DOMSnapshot (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import dom_debugger
from . import page


@dataclass
class DOMNode:
    '''
    A Node in the DOM tree.
    '''
    #: ``Node``'s nodeType.
    node_type: int

    #: ``Node``'s nodeName.
    node_name: str

    #: ``Node``'s nodeValue.
    node_value: str

    #: ``Node``'s id, corresponds to DOM.Node.backendNodeId.
    backend_node_id: dom.BackendNodeId

    #: Only set for textarea elements, contains the text value.
    text_value: typing.Optional[str] = None

    #: Only set for input elements, contains the input's associated text value.
    input_value: typing.Optional[str] = None

    #: Only set for radio and checkbox input elements, indicates if the element has been checked
    input_checked: typing.Optional[bool] = None

    #: Only set for option elements, indicates if the element has been selected
    option_selected: typing.Optional[bool] = None

    #: The indexes of the node's child nodes in the ``domNodes`` array returned by ``getSnapshot``, if
    #: any.
    child_node_indexes: typing.Optional[typing.List[int]] = None

    #: Attributes of an ``Element`` node.
    attributes: typing.Optional[typing.List[NameValue]] = None

    #: Indexes of pseudo elements associated with this node in the ``domNodes`` array returned by
    #: ``getSnapshot``, if any.
    pseudo_element_indexes: typing.Optional[typing.List[int]] = None

    #: The index of the node's related layout tree node in the ``layoutTreeNodes`` array returned by
    #: ``getSnapshot``, if any.
    layout_node_index: typing.Optional[int] = None

    #: Document URL that ``Document`` or ``FrameOwner`` node points to.
    document_url: typing.Optional[str] = None

    #: Base URL that ``Document`` or ``FrameOwner`` node uses for URL completion.
    base_url: typing.Optional[str] = None

    #: Only set for documents, contains the document's content language.
    content_language: typing.Optional[str] = None

    #: Only set for documents, contains the document's character set encoding.
    document_encoding: typing.Optional[str] = None

    #: ``DocumentType`` node's publicId.
    public_id: typing.Optional[str] = None

    #: ``DocumentType`` node's systemId.
    system_id: typing.Optional[str] = None

    #: Frame ID for frame owner elements and also for the document node.
    frame_id: typing.Optional[page.FrameId] = None

    #: The index of a frame owner element's content document in the ``domNodes`` array returned by
    #: ``getSnapshot``, if any.
    content_document_index: typing.Optional[int] = None

    #: Type of a pseudo element node.
    pseudo_type: typing.Optional[dom.PseudoType] = None

    #: Shadow root type.
    shadow_root_type: typing.Optional[dom.ShadowRootType] = None

    #: Whether this DOM node responds to mouse clicks. This includes nodes that have had click
    #: event listeners attached via JavaScript as well as anchor tags that naturally navigate when
    #: clicked.
    is_clickable: typing.Optional[bool] = None

    #: Details of the node's event listeners, if any.
    event_listeners: typing.Optional[typing.List[dom_debugger.EventListener]] = None

    #: The selected url for nodes with a srcset attribute.
    current_source_url: typing.Optional[str] = None

    #: The url of the script (if any) that generates this node.
    origin_url: typing.Optional[str] = None

    #: Scroll offsets, set when this node is a Document.
    scroll_offset_x: typing.Optional[float] = None

    scroll_offset_y: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['nodeType'] = self.node_type
        json['nodeName'] = self.node_name
        json['nodeValue'] = self.node_value
        json['backendNodeId'] = self.backend_node_id.to_json()
        if self.text_value is not None:
            json['textValue'] = self.text_value
        if self.input_value is not None:
            json['inputValue'] = self.input_value
        if self.input_checked is not None:
            json['inputChecked'] = self.input_checked
        if self.option_selected is not None:
            json['optionSelected'] = self.option_selected
        if self.child_node_indexes is not None:
            json['childNodeIndexes'] = [i for i in self.child_node_indexes]
        if self.attributes is not None:
            json['attributes'] = [i.to_json() for i in self.attributes]
        if self.pseudo_element_indexes is not None:
            json['pseudoElementIndexes'] = [i for i in self.pseudo_element_indexes]
        if self.layout_node_index is not None:
            json['layoutNodeIndex'] = self.layout_node_index
        if self.document_url is not None:
            json['documentURL'] = self.document_url
        if self.base_url is not None:
            json['baseURL'] = self.base_url
        if self.content_language is not None:
            json['contentLanguage'] = self.content_language
        if self.document_encoding is not None:
            json['documentEncoding'] = self.document_encoding
        if self.public_id is not None:
            json['publicId'] = self.public_id
        if self.system_id is not None:
            json['systemId'] = self.system_id
        if self.frame_id is not None:
            json['frameId'] = self.frame_id.to_json()
        if self.content_document_index is not None:
            json['contentDocumentIndex'] = self.content_document_index
        if self.pseudo_type is not None:
            json['pseudoType'] = self.pseudo_type.to_json()
        if self.shadow_root_type is not None:
            json['shadowRootType'] = self.shadow_root_type.to_json()
        if self.is_clickable is not None:
            json['isClickable'] = self.is_clickable
        if self.event_listeners is not None:
            json['eventListeners'] = [i.to_json() for i in self.event_listeners]
        if self.current_source_url is not None:
            json['currentSourceURL'] = self.current_source_url
        if self.origin_url is not None:
            json['originURL'] = self.origin_url
        if self.scroll_offset_x is not None:
            json['scrollOffsetX'] = self.scroll_offset_x
        if self.scroll_offset_y is not None:
            json['scrollOffsetY'] = self.scroll_offset_y
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            node_type=int(json['nodeType']),
            node_name=str(json['nodeName']),
            node_value=str(json['nodeValue']),
            backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId']),
            text_value=str(json['textValue']) if 'textValue' in json else None,
            input_value=str(json['inputValue']) if 'inputValue' in json else None,
            input_checked=bool(json['inputChecked']) if 'inputChecked' in json else None,
            option_selected=bool(json['optionSelected']) if 'optionSelected' in json else None,
            child_node_indexes=[int(i) for i in json['childNodeIndexes']] if 'childNodeIndexes' in json else None,
            attributes=[NameValue.from_json(i) for i in json['attributes']] if 'attributes' in json else None,
            pseudo_element_indexes=[int(i) for i in json['pseudoElementIndexes']] if 'pseudoElementIndexes' in json else None,
            layout_node_index=int(json['layoutNodeIndex']) if 'layoutNodeIndex' in json else None,
            document_url=str(json['documentURL']) if 'documentURL' in json else None,
            base_url=str(json['baseURL']) if 'baseURL' in json else None,
            content_language=str(json['contentLanguage']) if 'contentLanguage' in json else None,
            document_encoding=str(json['documentEncoding']) if 'documentEncoding' in json else None,
            public_id=str(json['publicId']) if 'publicId' in json else None,
            system_id=str(json['systemId']) if 'systemId' in json else None,
            frame_id=page.FrameId.from_json(json['frameId']) if 'frameId' in json else None,
            content_document_index=int(json['contentDocumentIndex']) if 'contentDocumentIndex' in json else None,
            pseudo_type=dom.PseudoType.from_json(json['pseudoType']) if 'pseudoType' in json else None,
            shadow_root_type=dom.ShadowRootType.from_json(json['shadowRootType']) if 'shadowRootType' in json else None,
            is_clickable=bool(json['isClickable']) if 'isClickable' in json else None,
            event_listeners=[dom_debugger.EventListener.from_json(i) for i in json['eventListeners']] if 'eventListeners' in json else None,
            current_source_url=str(json['currentSourceURL']) if 'currentSourceURL' in json else None,
            origin_url=str(json['originURL']) if 'originURL' in json else None,
            scroll_offset_x=float(json['scrollOffsetX']) if 'scrollOffsetX' in json else None,
            scroll_offset_y=float(json['scrollOffsetY']) if 'scrollOffsetY' in json else None,
        )


@dataclass
class InlineTextBox:
    '''
    Details of post layout rendered text positions. The exact layout should not be regarded as
    stable and may change between versions.
    '''
    #: The bounding box in document coordinates. Note that scroll offset of the document is ignored.
    bounding_box: dom.Rect

    #: The starting index in characters, for this post layout textbox substring. Characters that
    #: would be represented as a surrogate pair in UTF-16 have length 2.
    start_character_index: int

    #: The number of characters in this post layout textbox substring. Characters that would be
    #: represented as a surrogate pair in UTF-16 have length 2.
    num_characters: int

    def to_json(self):
        json = dict()
        json['boundingBox'] = self.bounding_box.to_json()
        json['startCharacterIndex'] = self.start_character_index
        json['numCharacters'] = self.num_characters
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            bounding_box=dom.Rect.from_json(json['boundingBox']),
            start_character_index=int(json['startCharacterIndex']),
            num_characters=int(json['numCharacters']),
        )


@dataclass
class LayoutTreeNode:
    '''
    Details of an element in the DOM tree with a LayoutObject.
    '''
    #: The index of the related DOM node in the ``domNodes`` array returned by ``getSnapshot``.
    dom_node_index: int

    #: The bounding box in document coordinates. Note that scroll offset of the document is ignored.
    bounding_box: dom.Rect

    #: Contents of the LayoutText, if any.
    layout_text: typing.Optional[str] = None

    #: The post-layout inline text nodes, if any.
    inline_text_nodes: typing.Optional[typing.List[InlineTextBox]] = None

    #: Index into the ``computedStyles`` array returned by ``getSnapshot``.
    style_index: typing.Optional[int] = None

    #: Global paint order index, which is determined by the stacking order of the nodes. Nodes
    #: that are painted together will have the same index. Only provided if includePaintOrder in
    #: getSnapshot was true.
    paint_order: typing.Optional[int] = None

    #: Set to true to indicate the element begins a new stacking context.
    is_stacking_context: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['domNodeIndex'] = self.dom_node_index
        json['boundingBox'] = self.bounding_box.to_json()
        if self.layout_text is not None:
            json['layoutText'] = self.layout_text
        if self.inline_text_nodes is not None:
            json['inlineTextNodes'] = [i.to_json() for i in self.inline_text_nodes]
        if self.style_index is not None:
            json['styleIndex'] = self.style_index
        if self.paint_order is not None:
            json['paintOrder'] = self.paint_order
        if self.is_stacking_context is not None:
            json['isStackingContext'] = self.is_stacking_context
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            dom_node_index=int(json['domNodeIndex']),
            bounding_box=dom.Rect.from_json(json['boundingBox']),
            layout_text=str(json['layoutText']) if 'layoutText' in json else None,
            inline_text_nodes=[InlineTextBox.from_json(i) for i in json['inlineTextNodes']] if 'inlineTextNodes' in json else None,
            style_index=int(json['styleIndex']) if 'styleIndex' in json else None,
            paint_order=int(json['paintOrder']) if 'paintOrder' in json else None,
            is_stacking_context=bool(json['isStackingContext']) if 'isStackingContext' in json else None,
        )


@dataclass
class ComputedStyle:
    '''
    A subset of the full ComputedStyle as defined by the request whitelist.
    '''
    #: Name/value pairs of computed style properties.
    properties: typing.List[NameValue]

    def to_json(self):
        json = dict()
        json['properties'] = [i.to_json() for i in self.properties]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            properties=[NameValue.from_json(i) for i in json['properties']],
        )


@dataclass
class NameValue:
    '''
    A name/value pair.
    '''
    #: Attribute/property name.
    name: str

    #: Attribute/property value.
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


class StringIndex(int):
    '''
    Index of the string in the strings table.
    '''
    def to_json(self) -> int:
        return self

    @classmethod
    def from_json(cls, json: int) -> StringIndex:
        return cls(json)

    def __repr__(self):
        return 'StringIndex({})'.format(super().__repr__())


class ArrayOfStrings(list):
    '''
    Index of the string in the strings table.
    '''
    def to_json(self) -> typing.List[StringIndex]:
        return self

    @classmethod
    def from_json(cls, json: typing.List[StringIndex]) -> ArrayOfStrings:
        return cls(json)

    def __repr__(self):
        return 'ArrayOfStrings({})'.format(super().__repr__())


@dataclass
class RareStringData:
    '''
    Data that is only present on rare nodes.
    '''
    index: typing.List[int]

    value: typing.List[StringIndex]

    def to_json(self):
        json = dict()
        json['index'] = [i for i in self.index]
        json['value'] = [i.to_json() for i in self.value]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            index=[int(i) for i in json['index']],
            value=[StringIndex.from_json(i) for i in json['value']],
        )


@dataclass
class RareBooleanData:
    index: typing.List[int]

    def to_json(self):
        json = dict()
        json['index'] = [i for i in self.index]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            index=[int(i) for i in json['index']],
        )


@dataclass
class RareIntegerData:
    index: typing.List[int]

    value: typing.List[int]

    def to_json(self):
        json = dict()
        json['index'] = [i for i in self.index]
        json['value'] = [i for i in self.value]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            index=[int(i) for i in json['index']],
            value=[int(i) for i in json['value']],
        )


class Rectangle(list):
    def to_json(self) -> typing.List[float]:
        return self

    @classmethod
    def from_json(cls, json: typing.List[float]) -> Rectangle:
        return cls(json)

    def __repr__(self):
        return 'Rectangle({})'.format(super().__repr__())


@dataclass
class DocumentSnapshot:
    '''
    Document snapshot.
    '''
    #: Document URL that ``Document`` or ``FrameOwner`` node points to.
    document_url: StringIndex

    #: Document title.
    title: StringIndex

    #: Base URL that ``Document`` or ``FrameOwner`` node uses for URL completion.
    base_url: StringIndex

    #: Contains the document's content language.
    content_language: StringIndex

    #: Contains the document's character set encoding.
    encoding_name: StringIndex

    #: ``DocumentType`` node's publicId.
    public_id: StringIndex

    #: ``DocumentType`` node's systemId.
    system_id: StringIndex

    #: Frame ID for frame owner elements and also for the document node.
    frame_id: StringIndex

    #: A table with dom nodes.
    nodes: NodeTreeSnapshot

    #: The nodes in the layout tree.
    layout: LayoutTreeSnapshot

    #: The post-layout inline text nodes.
    text_boxes: TextBoxSnapshot

    #: Horizontal scroll offset.
    scroll_offset_x: typing.Optional[float] = None

    #: Vertical scroll offset.
    scroll_offset_y: typing.Optional[float] = None

    #: Document content width.
    content_width: typing.Optional[float] = None

    #: Document content height.
    content_height: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['documentURL'] = self.document_url.to_json()
        json['title'] = self.title.to_json()
        json['baseURL'] = self.base_url.to_json()
        json['contentLanguage'] = self.content_language.to_json()
        json['encodingName'] = self.encoding_name.to_json()
        json['publicId'] = self.public_id.to_json()
        json['systemId'] = self.system_id.to_json()
        json['frameId'] = self.frame_id.to_json()
        json['nodes'] = self.nodes.to_json()
        json['layout'] = self.layout.to_json()
        json['textBoxes'] = self.text_boxes.to_json()
        if self.scroll_offset_x is not None:
            json['scrollOffsetX'] = self.scroll_offset_x
        if self.scroll_offset_y is not None:
            json['scrollOffsetY'] = self.scroll_offset_y
        if self.content_width is not None:
            json['contentWidth'] = self.content_width
        if self.content_height is not None:
            json['contentHeight'] = self.content_height
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            document_url=StringIndex.from_json(json['documentURL']),
            title=StringIndex.from_json(json['title']),
            base_url=StringIndex.from_json(json['baseURL']),
            content_language=StringIndex.from_json(json['contentLanguage']),
            encoding_name=StringIndex.from_json(json['encodingName']),
            public_id=StringIndex.from_json(json['publicId']),
            system_id=StringIndex.from_json(json['systemId']),
            frame_id=StringIndex.from_json(json['frameId']),
            nodes=NodeTreeSnapshot.from_json(json['nodes']),
            layout=LayoutTreeSnapshot.from_json(json['layout']),
            text_boxes=TextBoxSnapshot.from_json(json['textBoxes']),
            scroll_offset_x=float(json['scrollOffsetX']) if 'scrollOffsetX' in json else None,
            scroll_offset_y=float(json['scrollOffsetY']) if 'scrollOffsetY' in json else None,
            content_width=float(json['contentWidth']) if 'contentWidth' in json else None,
            content_height=float(json['contentHeight']) if 'contentHeight' in json else None,
        )


@dataclass
class NodeTreeSnapshot:
    '''
    Table containing nodes.
    '''
    #: Parent node index.
    parent_index: typing.Optional[typing.List[int]] = None

    #: ``Node``'s nodeType.
    node_type: typing.Optional[typing.List[int]] = None

    #: Type of the shadow root the ``Node`` is in. String values are equal to the ``ShadowRootType`` enum.
    shadow_root_type: typing.Optional[RareStringData] = None

    #: ``Node``'s nodeName.
    node_name: typing.Optional[typing.List[StringIndex]] = None

    #: ``Node``'s nodeValue.
    node_value: typing.Optional[typing.List[StringIndex]] = None

    #: ``Node``'s id, corresponds to DOM.Node.backendNodeId.
    backend_node_id: typing.Optional[typing.List[dom.BackendNodeId]] = None

    #: Attributes of an ``Element`` node. Flatten name, value pairs.
    attributes: typing.Optional[typing.List[ArrayOfStrings]] = None

    #: Only set for textarea elements, contains the text value.
    text_value: typing.Optional[RareStringData] = None

    #: Only set for input elements, contains the input's associated text value.
    input_value: typing.Optional[RareStringData] = None

    #: Only set for radio and checkbox input elements, indicates if the element has been checked
    input_checked: typing.Optional[RareBooleanData] = None

    #: Only set for option elements, indicates if the element has been selected
    option_selected: typing.Optional[RareBooleanData] = None

    #: The index of the document in the list of the snapshot documents.
    content_document_index: typing.Optional[RareIntegerData] = None

    #: Type of a pseudo element node.
    pseudo_type: typing.Optional[RareStringData] = None

    #: Pseudo element identifier for this node. Only present if there is a
    #: valid pseudoType.
    pseudo_identifier: typing.Optional[RareStringData] = None

    #: Whether this DOM node responds to mouse clicks. This includes nodes that have had click
    #: event listeners attached via JavaScript as well as anchor tags that naturally navigate when
    #: clicked.
    is_clickable: typing.Optional[RareBooleanData] = None

    #: The selected url for nodes with a srcset attribute.
    current_source_url: typing.Optional[RareStringData] = None

    #: The url of the script (if any) that generates this node.
    origin_url: typing.Optional[RareStringData] = None

    def to_json(self):
        json = dict()
        if self.parent_index is not None:
            json['parentIndex'] = [i for i in self.parent_index]
        if self.node_type is not None:
            json['nodeType'] = [i for i in self.node_type]
        if self.shadow_root_type is not None:
            json['shadowRootType'] = self.shadow_root_type.to_json()
        if self.node_name is not None:
            json['nodeName'] = [i.to_json() for i in self.node_name]
        if self.node_value is not None:
            json['nodeValue'] = [i.to_json() for i in self.node_value]
        if self.backend_node_id is not None:
            json['backendNodeId'] = [i.to_json() for i in self.backend_node_id]
        if self.attributes is not None:
            json['attributes'] = [i.to_json() for i in self.attributes]
        if self.text_value is not None:
            json['textValue'] = self.text_value.to_json()
        if self.input_value is not None:
            json['inputValue'] = self.input_value.to_json()
        if self.input_checked is not None:
            json['inputChecked'] = self.input_checked.to_json()
        if self.option_selected is not None:
            json['optionSelected'] = self.option_selected.to_json()
        if self.content_document_index is not None:
            json['contentDocumentIndex'] = self.content_document_index.to_json()
        if self.pseudo_type is not None:
            json['pseudoType'] = self.pseudo_type.to_json()
        if self.pseudo_identifier is not None:
            json['pseudoIdentifier'] = self.pseudo_identifier.to_json()
        if self.is_clickable is not None:
            json['isClickable'] = self.is_clickable.to_json()
        if self.current_source_url is not None:
            json['currentSourceURL'] = self.current_source_url.to_json()
        if self.origin_url is not None:
            json['originURL'] = self.origin_url.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            parent_index=[int(i) for i in json['parentIndex']] if 'parentIndex' in json else None,
            node_type=[int(i) for i in json['nodeType']] if 'nodeType' in json else None,
            shadow_root_type=RareStringData.from_json(json['shadowRootType']) if 'shadowRootType' in json else None,
            node_name=[StringIndex.from_json(i) for i in json['nodeName']] if 'nodeName' in json else None,
            node_value=[StringIndex.from_json(i) for i in json['nodeValue']] if 'nodeValue' in json else None,
            backend_node_id=[dom.BackendNodeId.from_json(i) for i in json['backendNodeId']] if 'backendNodeId' in json else None,
            attributes=[ArrayOfStrings.from_json(i) for i in json['attributes']] if 'attributes' in json else None,
            text_value=RareStringData.from_json(json['textValue']) if 'textValue' in json else None,
            input_value=RareStringData.from_json(json['inputValue']) if 'inputValue' in json else None,
            input_checked=RareBooleanData.from_json(json['inputChecked']) if 'inputChecked' in json else None,
            option_selected=RareBooleanData.from_json(json['optionSelected']) if 'optionSelected' in json else None,
            content_document_index=RareIntegerData.from_json(json['contentDocumentIndex']) if 'contentDocumentIndex' in json else None,
            pseudo_type=RareStringData.from_json(json['pseudoType']) if 'pseudoType' in json else None,
            pseudo_identifier=RareStringData.from_json(json['pseudoIdentifier']) if 'pseudoIdentifier' in json else None,
            is_clickable=RareBooleanData.from_json(json['isClickable']) if 'isClickable' in json else None,
            current_source_url=RareStringData.from_json(json['currentSourceURL']) if 'currentSourceURL' in json else None,
            origin_url=RareStringData.from_json(json['originURL']) if 'originURL' in json else None,
        )


@dataclass
class LayoutTreeSnapshot:
    '''
    Table of details of an element in the DOM tree with a LayoutObject.
    '''
    #: Index of the corresponding node in the ``NodeTreeSnapshot`` array returned by ``captureSnapshot``.
    node_index: typing.List[int]

    #: Array of indexes specifying computed style strings, filtered according to the ``computedStyles`` parameter passed to ``captureSnapshot``.
    styles: typing.List[ArrayOfStrings]

    #: The absolute position bounding box.
    bounds: typing.List[Rectangle]

    #: Contents of the LayoutText, if any.
    text: typing.List[StringIndex]

    #: Stacking context information.
    stacking_contexts: RareBooleanData

    #: Global paint order index, which is determined by the stacking order of the nodes. Nodes
    #: that are painted together will have the same index. Only provided if includePaintOrder in
    #: captureSnapshot was true.
    paint_orders: typing.Optional[typing.List[int]] = None

    #: The offset rect of nodes. Only available when includeDOMRects is set to true
    offset_rects: typing.Optional[typing.List[Rectangle]] = None

    #: The scroll rect of nodes. Only available when includeDOMRects is set to true
    scroll_rects: typing.Optional[typing.List[Rectangle]] = None

    #: The client rect of nodes. Only available when includeDOMRects is set to true
    client_rects: typing.Optional[typing.List[Rectangle]] = None

    #: The list of background colors that are blended with colors of overlapping elements.
    blended_background_colors: typing.Optional[typing.List[StringIndex]] = None

    #: The list of computed text opacities.
    text_color_opacities: typing.Optional[typing.List[float]] = None

    def to_json(self):
        json = dict()
        json['nodeIndex'] = [i for i in self.node_index]
        json['styles'] = [i.to_json() for i in self.styles]
        json['bounds'] = [i.to_json() for i in self.bounds]
        json['text'] = [i.to_json() for i in self.text]
        json['stackingContexts'] = self.stacking_contexts.to_json()
        if self.paint_orders is not None:
            json['paintOrders'] = [i for i in self.paint_orders]
        if self.offset_rects is not None:
            json['offsetRects'] = [i.to_json() for i in self.offset_rects]
        if self.scroll_rects is not None:
            json['scrollRects'] = [i.to_json() for i in self.scroll_rects]
        if self.client_rects is not None:
            json['clientRects'] = [i.to_json() for i in self.client_rects]
        if self.blended_background_colors is not None:
            json['blendedBackgroundColors'] = [i.to_json() for i in self.blended_background_colors]
        if self.text_color_opacities is not None:
            json['textColorOpacities'] = [i for i in self.text_color_opacities]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            node_index=[int(i) for i in json['nodeIndex']],
            styles=[ArrayOfStrings.from_json(i) for i in json['styles']],
            bounds=[Rectangle.from_json(i) for i in json['bounds']],
            text=[StringIndex.from_json(i) for i in json['text']],
            stacking_contexts=RareBooleanData.from_json(json['stackingContexts']),
            paint_orders=[int(i) for i in json['paintOrders']] if 'paintOrders' in json else None,
            offset_rects=[Rectangle.from_json(i) for i in json['offsetRects']] if 'offsetRects' in json else None,
            scroll_rects=[Rectangle.from_json(i) for i in json['scrollRects']] if 'scrollRects' in json else None,
            client_rects=[Rectangle.from_json(i) for i in json['clientRects']] if 'clientRects' in json else None,
            blended_background_colors=[StringIndex.from_json(i) for i in json['blendedBackgroundColors']] if 'blendedBackgroundColors' in json else None,
            text_color_opacities=[float(i) for i in json['textColorOpacities']] if 'textColorOpacities' in json else None,
        )


@dataclass
class TextBoxSnapshot:
    '''
    Table of details of the post layout rendered text positions. The exact layout should not be regarded as
    stable and may change between versions.
    '''
    #: Index of the layout tree node that owns this box collection.
    layout_index: typing.List[int]

    #: The absolute position bounding box.
    bounds: typing.List[Rectangle]

    #: The starting index in characters, for this post layout textbox substring. Characters that
    #: would be represented as a surrogate pair in UTF-16 have length 2.
    start: typing.List[int]

    #: The number of characters in this post layout textbox substring. Characters that would be
    #: represented as a surrogate pair in UTF-16 have length 2.
    length: typing.List[int]

    def to_json(self):
        json = dict()
        json['layoutIndex'] = [i for i in self.layout_index]
        json['bounds'] = [i.to_json() for i in self.bounds]
        json['start'] = [i for i in self.start]
        json['length'] = [i for i in self.length]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            layout_index=[int(i) for i in json['layoutIndex']],
            bounds=[Rectangle.from_json(i) for i in json['bounds']],
            start=[int(i) for i in json['start']],
            length=[int(i) for i in json['length']],
        )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables DOM snapshot agent for the given page.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMSnapshot.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables DOM snapshot agent for the given page.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMSnapshot.enable',
    }
    json = yield cmd_dict


def get_snapshot(
        computed_style_whitelist: typing.List[str],
        include_event_listeners: typing.Optional[bool] = None,
        include_paint_order: typing.Optional[bool] = None,
        include_user_agent_shadow_tree: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[DOMNode], typing.List[LayoutTreeNode], typing.List[ComputedStyle]]]:
    '''
    Returns a document snapshot, including the full DOM tree of the root node (including iframes,
    template contents, and imported documents) in a flattened array, as well as layout and
    white-listed computed style information for the nodes. Shadow DOM in the returned DOM tree is
    flattened.

    :param computed_style_whitelist: Whitelist of computed styles to return.
    :param include_event_listeners: *(Optional)* Whether or not to retrieve details of DOM listeners (default false).
    :param include_paint_order: *(Optional)* Whether to determine and include the paint order index of LayoutTreeNodes (default false).
    :param include_user_agent_shadow_tree: *(Optional)* Whether to include UA shadow tree in the snapshot (default false).
    :returns: A tuple with the following items:

        0. **domNodes** - The nodes in the DOM tree. The DOMNode at index 0 corresponds to the root document.
        1. **layoutTreeNodes** - The nodes in the layout tree.
        2. **computedStyles** - Whitelisted ComputedStyle properties for each node in the layout tree.
    '''
    params: T_JSON_DICT = dict()
    params['computedStyleWhitelist'] = [i for i in computed_style_whitelist]
    if include_event_listeners is not None:
        params['includeEventListeners'] = include_event_listeners
    if include_paint_order is not None:
        params['includePaintOrder'] = include_paint_order
    if include_user_agent_shadow_tree is not None:
        params['includeUserAgentShadowTree'] = include_user_agent_shadow_tree
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMSnapshot.getSnapshot',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [DOMNode.from_json(i) for i in json['domNodes']],
        [LayoutTreeNode.from_json(i) for i in json['layoutTreeNodes']],
        [ComputedStyle.from_json(i) for i in json['computedStyles']]
    )


def capture_snapshot(
        computed_styles: typing.List[str],
        include_paint_order: typing.Optional[bool] = None,
        include_dom_rects: typing.Optional[bool] = None,
        include_blended_background_colors: typing.Optional[bool] = None,
        include_text_color_opacities: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[DocumentSnapshot], typing.List[str]]]:
    '''
    Returns a document snapshot, including the full DOM tree of the root node (including iframes,
    template contents, and imported documents) in a flattened array, as well as layout and
    white-listed computed style information for the nodes. Shadow DOM in the returned DOM tree is
    flattened.

    :param computed_styles: Whitelist of computed styles to return.
    :param include_paint_order: *(Optional)* Whether to include layout object paint orders into the snapshot.
    :param include_dom_rects: *(Optional)* Whether to include DOM rectangles (offsetRects, clientRects, scrollRects) into the snapshot
    :param include_blended_background_colors: **(EXPERIMENTAL)** *(Optional)* Whether to include blended background colors in the snapshot (default: false). Blended background color is achieved by blending background colors of all elements that overlap with the current element.
    :param include_text_color_opacities: **(EXPERIMENTAL)** *(Optional)* Whether to include text color opacity in the snapshot (default: false). An element might have the opacity property set that affects the text color of the element. The final text color opacity is computed based on the opacity of all overlapping elements.
    :returns: A tuple with the following items:

        0. **documents** - The nodes in the DOM tree. The DOMNode at index 0 corresponds to the root document.
        1. **strings** - Shared string table that all string properties refer to with indexes.
    '''
    params: T_JSON_DICT = dict()
    params['computedStyles'] = [i for i in computed_styles]
    if include_paint_order is not None:
        params['includePaintOrder'] = include_paint_order
    if include_dom_rects is not None:
        params['includeDOMRects'] = include_dom_rects
    if include_blended_background_colors is not None:
        params['includeBlendedBackgroundColors'] = include_blended_background_colors
    if include_text_color_opacities is not None:
        params['includeTextColorOpacities'] = include_text_color_opacities
    cmd_dict: T_JSON_DICT = {
        'method': 'DOMSnapshot.captureSnapshot',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [DocumentSnapshot.from_json(i) for i in json['documents']],
        [str(i) for i in json['strings']]
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\biomechanics\activation.py ===
r"""Activation dynamics for musclotendon models.

Musculotendon models are able to produce active force when they are activated,
which is when a chemical process has taken place within the muscle fibers
causing them to voluntarily contract. Biologically this chemical process (the
diffusion of :math:`\textrm{Ca}^{2+}` ions) is not the input in the system,
electrical signals from the nervous system are. These are termed excitations.
Activation dynamics, which relates the normalized excitation level to the
normalized activation level, can be modeled by the models present in this
module.

"""

from abc import ABC, abstractmethod
from functools import cached_property

from sympy.core.symbol import Symbol
from sympy.core.numbers import Float, Integer, Rational
from sympy.functions.elementary.hyperbolic import tanh
from sympy.matrices.dense import MutableDenseMatrix as Matrix, zeros
from sympy.physics.biomechanics._mixin import _NamedMixin
from sympy.physics.mechanics import dynamicsymbols


__all__ = [
    'ActivationBase',
    'FirstOrderActivationDeGroote2016',
    'ZerothOrderActivation',
]


class ActivationBase(ABC, _NamedMixin):
    """Abstract base class for all activation dynamics classes to inherit from.

    Notes
    =====

    Instances of this class cannot be directly instantiated by users. However,
    it can be used to created custom activation dynamics types through
    subclassing.

    """

    def __init__(self, name):
        """Initializer for ``ActivationBase``."""
        self.name = str(name)

        # Symbols
        self._e = dynamicsymbols(f"e_{name}")
        self._a = dynamicsymbols(f"a_{name}")

    @classmethod
    @abstractmethod
    def with_defaults(cls, name):
        """Alternate constructor that provides recommended defaults for
        constants."""
        pass

    @property
    def excitation(self):
        """Dynamic symbol representing excitation.

        Explanation
        ===========

        The alias ``e`` can also be used to access the same attribute.

        """
        return self._e

    @property
    def e(self):
        """Dynamic symbol representing excitation.

        Explanation
        ===========

        The alias ``excitation`` can also be used to access the same attribute.

        """
        return self._e

    @property
    def activation(self):
        """Dynamic symbol representing activation.

        Explanation
        ===========

        The alias ``a`` can also be used to access the same attribute.

        """
        return self._a

    @property
    def a(self):
        """Dynamic symbol representing activation.

        Explanation
        ===========

        The alias ``activation`` can also be used to access the same attribute.

        """
        return self._a

    @property
    @abstractmethod
    def order(self):
        """Order of the (differential) equation governing activation."""
        pass

    @property
    @abstractmethod
    def state_vars(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``x`` can also be used to access the same attribute.

        """
        pass

    @property
    @abstractmethod
    def x(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``state_vars`` can also be used to access the same attribute.

        """
        pass

    @property
    @abstractmethod
    def input_vars(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``r`` can also be used to access the same attribute.

        """
        pass

    @property
    @abstractmethod
    def r(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``input_vars`` can also be used to access the same attribute.

        """
        pass

    @property
    @abstractmethod
    def constants(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        The alias ``p`` can also be used to access the same attribute.

        """
        pass

    @property
    @abstractmethod
    def p(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        The alias ``constants`` can also be used to access the same attribute.

        """
        pass

    @property
    @abstractmethod
    def M(self):
        """Ordered square matrix of coefficients on the LHS of ``M x' = F``.

        Explanation
        ===========

        The square matrix that forms part of the LHS of the linear system of
        ordinary differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
        pass

    @property
    @abstractmethod
    def F(self):
        """Ordered column matrix of equations on the RHS of ``M x' = F``.

        Explanation
        ===========

        The column matrix that forms the RHS of the linear system of ordinary
        differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
        pass

    @abstractmethod
    def rhs(self):
        """

        Explanation
        ===========

        The solution to the linear system of ordinary differential equations
        governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
        pass

    def __eq__(self, other):
        """Equality check for activation dynamics."""
        if type(self) != type(other):
            return False
        if self.name != other.name:
            return False
        return True

    def __repr__(self):
        """Default representation of activation dynamics."""
        return f'{self.__class__.__name__}({self.name!r})'


class ZerothOrderActivation(ActivationBase):
    """Simple zeroth-order activation dynamics mapping excitation to
    activation.

    Explanation
    ===========

    Zeroth-order activation dynamics are useful in instances where you want to
    reduce the complexity of your musculotendon dynamics as they simple map
    exictation to activation. As a result, no additional state equations are
    introduced to your system. They also remove a potential source of delay
    between the input and dynamics of your system as no (ordinary) differential
    equations are involved.

    """

    def __init__(self, name):
        """Initializer for ``ZerothOrderActivation``.

        Parameters
        ==========

        name : str
            The name identifier associated with the instance. Must be a string
            of length at least 1.

        """
        super().__init__(name)

        # Zeroth-order activation dynamics has activation equal excitation so
        # overwrite the symbol for activation with the excitation symbol.
        self._a = self._e

    @classmethod
    def with_defaults(cls, name):
        """Alternate constructor that provides recommended defaults for
        constants.

        Explanation
        ===========

        As this concrete class doesn't implement any constants associated with
        its dynamics, this ``classmethod`` simply creates a standard instance
        of ``ZerothOrderActivation``. An implementation is provided to ensure
        a consistent interface between all ``ActivationBase`` concrete classes.

        """
        return cls(name)

    @property
    def order(self):
        """Order of the (differential) equation governing activation."""
        return 0

    @property
    def state_vars(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        As zeroth-order activation dynamics simply maps excitation to
        activation, this class has no associated state variables and so this
        property return an empty column ``Matrix`` with shape (0, 1).

        The alias ``x`` can also be used to access the same attribute.

        """
        return zeros(0, 1)

    @property
    def x(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        As zeroth-order activation dynamics simply maps excitation to
        activation, this class has no associated state variables and so this
        property return an empty column ``Matrix`` with shape (0, 1).

        The alias ``state_vars`` can also be used to access the same attribute.

        """
        return zeros(0, 1)

    @property
    def input_vars(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        Excitation is the only input in zeroth-order activation dynamics and so
        this property returns a column ``Matrix`` with one entry, ``e``, and
        shape (1, 1).

        The alias ``r`` can also be used to access the same attribute.

        """
        return Matrix([self._e])

    @property
    def r(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        Excitation is the only input in zeroth-order activation dynamics and so
        this property returns a column ``Matrix`` with one entry, ``e``, and
        shape (1, 1).

        The alias ``input_vars`` can also be used to access the same attribute.

        """
        return Matrix([self._e])

    @property
    def constants(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        As zeroth-order activation dynamics simply maps excitation to
        activation, this class has no associated constants and so this property
        return an empty column ``Matrix`` with shape (0, 1).

        The alias ``p`` can also be used to access the same attribute.

        """
        return zeros(0, 1)

    @property
    def p(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        As zeroth-order activation dynamics simply maps excitation to
        activation, this class has no associated constants and so this property
        return an empty column ``Matrix`` with shape (0, 1).

        The alias ``constants`` can also be used to access the same attribute.

        """
        return zeros(0, 1)

    @property
    def M(self):
        """Ordered square matrix of coefficients on the LHS of ``M x' = F``.

        Explanation
        ===========

        The square matrix that forms part of the LHS of the linear system of
        ordinary differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear system has dimension 0 and therefore ``M`` is an empty square
        ``Matrix`` with shape (0, 0).

        """
        return Matrix([])

    @property
    def F(self):
        """Ordered column matrix of equations on the RHS of ``M x' = F``.

        Explanation
        ===========

        The column matrix that forms the RHS of the linear system of ordinary
        differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear system has dimension 0 and therefore ``F`` is an empty column
        ``Matrix`` with shape (0, 1).

        """
        return zeros(0, 1)

    def rhs(self):
        """Ordered column matrix of equations for the solution of ``M x' = F``.

        Explanation
        ===========

        The solution to the linear system of ordinary differential equations
        governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear has dimension 0 and therefore this method returns an empty
        column ``Matrix`` with shape (0, 1).

        """
        return zeros(0, 1)


class FirstOrderActivationDeGroote2016(ActivationBase):
    r"""First-order activation dynamics based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the first-order activation dynamics equation for the rate of change
    of activation with respect to time as a function of excitation and
    activation.

    The function is defined by the equation:

    .. math::

        \frac{da}{dt} = \left(\frac{\frac{1}{2} + a0}{\tau_a \left(\frac{1}{2}
            + \frac{3a}{2}\right)} + \frac{\left(\frac{1}{2}
            + \frac{3a}{2}\right) \left(\frac{1}{2} - a0\right)}{\tau_d}\right)
            \left(e - a\right)

    where

    .. math::

        a0 = \frac{\tanh{\left(b \left(e - a\right) \right)}}{2}

    with constant values of :math:`tau_a = 0.015`, :math:`tau_d = 0.060`, and
    :math:`b = 10`.

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    def __init__(self,
        name,
        activation_time_constant=None,
        deactivation_time_constant=None,
        smoothing_rate=None,
    ):
        """Initializer for ``FirstOrderActivationDeGroote2016``.

        Parameters
        ==========
        activation time constant : Symbol | Number | None
            The value of the activation time constant governing the delay
            between excitation and activation when excitation exceeds
            activation.
        deactivation time constant : Symbol | Number | None
            The value of the deactivation time constant governing the delay
            between excitation and activation when activation exceeds
            excitation.
        smoothing_rate : Symbol | Number | None
            The slope of the hyperbolic tangent function used to smooth between
            the switching of the equations where excitation exceed activation
            and where activation exceeds excitation. The recommended value to
            use is ``10``, but values between ``0.1`` and ``100`` can be used.

        """
        super().__init__(name)

        # Symbols
        self.activation_time_constant = activation_time_constant
        self.deactivation_time_constant = deactivation_time_constant
        self.smoothing_rate = smoothing_rate

    @classmethod
    def with_defaults(cls, name):
        r"""Alternate constructor that will use the published constants.

        Explanation
        ===========

        Returns an instance of ``FirstOrderActivationDeGroote2016`` using the
        three constant values specified in the original publication.

        These have the values:

        :math:`tau_a = 0.015`
        :math:`tau_d = 0.060`
        :math:`b = 10`

        """
        tau_a = Float('0.015')
        tau_d = Float('0.060')
        b = Float('10.0')
        return cls(name, tau_a, tau_d, b)

    @property
    def activation_time_constant(self):
        """Delay constant for activation.

        Explanation
        ===========

        The alias ```tau_a`` can also be used to access the same attribute.

        """
        return self._tau_a

    @activation_time_constant.setter
    def activation_time_constant(self, tau_a):
        if hasattr(self, '_tau_a'):
            msg = (
                f'Can\'t set attribute `activation_time_constant` to '
                f'{repr(tau_a)} as it is immutable and already has value '
                f'{self._tau_a}.'
            )
            raise AttributeError(msg)
        self._tau_a = Symbol(f'tau_a_{self.name}') if tau_a is None else tau_a

    @property
    def tau_a(self):
        """Delay constant for activation.

        Explanation
        ===========

        The alias ``activation_time_constant`` can also be used to access the
        same attribute.

        """
        return self._tau_a

    @property
    def deactivation_time_constant(self):
        """Delay constant for deactivation.

        Explanation
        ===========

        The alias ``tau_d`` can also be used to access the same attribute.

        """
        return self._tau_d

    @deactivation_time_constant.setter
    def deactivation_time_constant(self, tau_d):
        if hasattr(self, '_tau_d'):
            msg = (
                f'Can\'t set attribute `deactivation_time_constant` to '
                f'{repr(tau_d)} as it is immutable and already has value '
                f'{self._tau_d}.'
            )
            raise AttributeError(msg)
        self._tau_d = Symbol(f'tau_d_{self.name}') if tau_d is None else tau_d

    @property
    def tau_d(self):
        """Delay constant for deactivation.

        Explanation
        ===========

        The alias ``deactivation_time_constant`` can also be used to access the
        same attribute.

        """
        return self._tau_d

    @property
    def smoothing_rate(self):
        """Smoothing constant for the hyperbolic tangent term.

        Explanation
        ===========

        The alias ``b`` can also be used to access the same attribute.

        """
        return self._b

    @smoothing_rate.setter
    def smoothing_rate(self, b):
        if hasattr(self, '_b'):
            msg = (
                f'Can\'t set attribute `smoothing_rate` to {b!r} as it is '
                f'immutable and already has value {self._b!r}.'
            )
            raise AttributeError(msg)
        self._b = Symbol(f'b_{self.name}') if b is None else b

    @property
    def b(self):
        """Smoothing constant for the hyperbolic tangent term.

        Explanation
        ===========

        The alias ``smoothing_rate`` can also be used to access the same
        attribute.

        """
        return self._b

    @property
    def order(self):
        """Order of the (differential) equation governing activation."""
        return 1

    @property
    def state_vars(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``x`` can also be used to access the same attribute.

        """
        return Matrix([self._a])

    @property
    def x(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``state_vars`` can also be used to access the same attribute.

        """
        return Matrix([self._a])

    @property
    def input_vars(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``r`` can also be used to access the same attribute.

        """
        return Matrix([self._e])

    @property
    def r(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``input_vars`` can also be used to access the same attribute.

        """
        return Matrix([self._e])

    @property
    def constants(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        The alias ``p`` can also be used to access the same attribute.

        """
        constants = [self._tau_a, self._tau_d, self._b]
        symbolic_constants = [c for c in constants if not c.is_number]
        return Matrix(symbolic_constants) if symbolic_constants else zeros(0, 1)

    @property
    def p(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Explanation
        ===========

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        The alias ``constants`` can also be used to access the same attribute.

        """
        constants = [self._tau_a, self._tau_d, self._b]
        symbolic_constants = [c for c in constants if not c.is_number]
        return Matrix(symbolic_constants) if symbolic_constants else zeros(0, 1)

    @property
    def M(self):
        """Ordered square matrix of coefficients on the LHS of ``M x' = F``.

        Explanation
        ===========

        The square matrix that forms part of the LHS of the linear system of
        ordinary differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
        return Matrix([Integer(1)])

    @property
    def F(self):
        """Ordered column matrix of equations on the RHS of ``M x' = F``.

        Explanation
        ===========

        The column matrix that forms the RHS of the linear system of ordinary
        differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
        return Matrix([self._da_eqn])

    def rhs(self):
        """Ordered column matrix of equations for the solution of ``M x' = F``.

        Explanation
        ===========

        The solution to the linear system of ordinary differential equations
        governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
        return Matrix([self._da_eqn])

    @cached_property
    def _da_eqn(self):
        HALF = Rational(1, 2)
        a0 = HALF * tanh(self._b * (self._e - self._a))
        a1 = (HALF + Rational(3, 2) * self._a)
        a2 = (HALF + a0) / (self._tau_a * a1)
        a3 = a1 * (HALF - a0) / self._tau_d
        activation_dynamics_equation = (a2 + a3) * (self._e - self._a)
        return activation_dynamics_equation

    def __eq__(self, other):
        """Equality check for ``FirstOrderActivationDeGroote2016``."""
        if type(self) != type(other):
            return False
        self_attrs = (self.name, self.tau_a, self.tau_d, self.b)
        other_attrs = (other.name, other.tau_a, other.tau_d, other.b)
        if self_attrs == other_attrs:
            return True
        return False

    def __repr__(self):
        """Representation of ``FirstOrderActivationDeGroote2016``."""
        return (
            f'{self.__class__.__name__}({self.name!r}, '
            f'activation_time_constant={self.tau_a!r}, '
            f'deactivation_time_constant={self.tau_d!r}, '
            f'smoothing_rate={self.b!r})'
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\lexer.py ===
"""Implements a Jinja / Python combination lexer. The ``Lexer`` class
is used to do some preprocessing. It filters out invalid operators like
the bitshift operators we don't allow in templates. It separates
template code and python code in expressions.
"""

import re
import typing as t
from ast import literal_eval
from collections import deque
from sys import intern

from ._identifier import pattern as name_re
from .exceptions import TemplateSyntaxError
from .utils import LRUCache

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .environment import Environment

# cache for the lexers. Exists in order to be able to have multiple
# environments with the same lexer
_lexer_cache: t.MutableMapping[t.Tuple, "Lexer"] = LRUCache(50)  # type: ignore

# static regular expressions
whitespace_re = re.compile(r"\s+")
newline_re = re.compile(r"(\r\n|\r|\n)")
string_re = re.compile(
    r"('([^'\\]*(?:\\.[^'\\]*)*)'" r'|"([^"\\]*(?:\\.[^"\\]*)*)")', re.S
)
integer_re = re.compile(
    r"""
    (
        0b(_?[0-1])+ # binary
    |
        0o(_?[0-7])+ # octal
    |
        0x(_?[\da-f])+ # hex
    |
        [1-9](_?\d)* # decimal
    |
        0(_?0)* # decimal zero
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
float_re = re.compile(
    r"""
    (?<!\.)  # doesn't start with a .
    (\d+_)*\d+  # digits, possibly _ separated
    (
        (\.(\d+_)*\d+)?  # optional fractional part
        e[+\-]?(\d+_)*\d+  # exponent part
    |
        \.(\d+_)*\d+  # required fractional part
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# internal the tokens and keep references to them
TOKEN_ADD = intern("add")
TOKEN_ASSIGN = intern("assign")
TOKEN_COLON = intern("colon")
TOKEN_COMMA = intern("comma")
TOKEN_DIV = intern("div")
TOKEN_DOT = intern("dot")
TOKEN_EQ = intern("eq")
TOKEN_FLOORDIV = intern("floordiv")
TOKEN_GT = intern("gt")
TOKEN_GTEQ = intern("gteq")
TOKEN_LBRACE = intern("lbrace")
TOKEN_LBRACKET = intern("lbracket")
TOKEN_LPAREN = intern("lparen")
TOKEN_LT = intern("lt")
TOKEN_LTEQ = intern("lteq")
TOKEN_MOD = intern("mod")
TOKEN_MUL = intern("mul")
TOKEN_NE = intern("ne")
TOKEN_PIPE = intern("pipe")
TOKEN_POW = intern("pow")
TOKEN_RBRACE = intern("rbrace")
TOKEN_RBRACKET = intern("rbracket")
TOKEN_RPAREN = intern("rparen")
TOKEN_SEMICOLON = intern("semicolon")
TOKEN_SUB = intern("sub")
TOKEN_TILDE = intern("tilde")
TOKEN_WHITESPACE = intern("whitespace")
TOKEN_FLOAT = intern("float")
TOKEN_INTEGER = intern("integer")
TOKEN_NAME = intern("name")
TOKEN_STRING = intern("string")
TOKEN_OPERATOR = intern("operator")
TOKEN_BLOCK_BEGIN = intern("block_begin")
TOKEN_BLOCK_END = intern("block_end")
TOKEN_VARIABLE_BEGIN = intern("variable_begin")
TOKEN_VARIABLE_END = intern("variable_end")
TOKEN_RAW_BEGIN = intern("raw_begin")
TOKEN_RAW_END = intern("raw_end")
TOKEN_COMMENT_BEGIN = intern("comment_begin")
TOKEN_COMMENT_END = intern("comment_end")
TOKEN_COMMENT = intern("comment")
TOKEN_LINESTATEMENT_BEGIN = intern("linestatement_begin")
TOKEN_LINESTATEMENT_END = intern("linestatement_end")
TOKEN_LINECOMMENT_BEGIN = intern("linecomment_begin")
TOKEN_LINECOMMENT_END = intern("linecomment_end")
TOKEN_LINECOMMENT = intern("linecomment")
TOKEN_DATA = intern("data")
TOKEN_INITIAL = intern("initial")
TOKEN_EOF = intern("eof")

# bind operators to token types
operators = {
    "+": TOKEN_ADD,
    "-": TOKEN_SUB,
    "/": TOKEN_DIV,
    "//": TOKEN_FLOORDIV,
    "*": TOKEN_MUL,
    "%": TOKEN_MOD,
    "**": TOKEN_POW,
    "~": TOKEN_TILDE,
    "[": TOKEN_LBRACKET,
    "]": TOKEN_RBRACKET,
    "(": TOKEN_LPAREN,
    ")": TOKEN_RPAREN,
    "{": TOKEN_LBRACE,
    "}": TOKEN_RBRACE,
    "==": TOKEN_EQ,
    "!=": TOKEN_NE,
    ">": TOKEN_GT,
    ">=": TOKEN_GTEQ,
    "<": TOKEN_LT,
    "<=": TOKEN_LTEQ,
    "=": TOKEN_ASSIGN,
    ".": TOKEN_DOT,
    ":": TOKEN_COLON,
    "|": TOKEN_PIPE,
    ",": TOKEN_COMMA,
    ";": TOKEN_SEMICOLON,
}

reverse_operators = {v: k for k, v in operators.items()}
assert len(operators) == len(reverse_operators), "operators dropped"
operator_re = re.compile(
    f"({'|'.join(re.escape(x) for x in sorted(operators, key=lambda x: -len(x)))})"
)

ignored_tokens = frozenset(
    [
        TOKEN_COMMENT_BEGIN,
        TOKEN_COMMENT,
        TOKEN_COMMENT_END,
        TOKEN_WHITESPACE,
        TOKEN_LINECOMMENT_BEGIN,
        TOKEN_LINECOMMENT_END,
        TOKEN_LINECOMMENT,
    ]
)
ignore_if_empty = frozenset(
    [TOKEN_WHITESPACE, TOKEN_DATA, TOKEN_COMMENT, TOKEN_LINECOMMENT]
)


def _describe_token_type(token_type: str) -> str:
    if token_type in reverse_operators:
        return reverse_operators[token_type]

    return {
        TOKEN_COMMENT_BEGIN: "begin of comment",
        TOKEN_COMMENT_END: "end of comment",
        TOKEN_COMMENT: "comment",
        TOKEN_LINECOMMENT: "comment",
        TOKEN_BLOCK_BEGIN: "begin of statement block",
        TOKEN_BLOCK_END: "end of statement block",
        TOKEN_VARIABLE_BEGIN: "begin of print statement",
        TOKEN_VARIABLE_END: "end of print statement",
        TOKEN_LINESTATEMENT_BEGIN: "begin of line statement",
        TOKEN_LINESTATEMENT_END: "end of line statement",
        TOKEN_DATA: "template data / text",
        TOKEN_EOF: "end of template",
    }.get(token_type, token_type)


def describe_token(token: "Token") -> str:
    """Returns a description of the token."""
    if token.type == TOKEN_NAME:
        return token.value

    return _describe_token_type(token.type)


def describe_token_expr(expr: str) -> str:
    """Like `describe_token` but for token expressions."""
    if ":" in expr:
        type, value = expr.split(":", 1)

        if type == TOKEN_NAME:
            return value
    else:
        type = expr

    return _describe_token_type(type)


def count_newlines(value: str) -> int:
    """Count the number of newline characters in the string.  This is
    useful for extensions that filter a stream.
    """
    return len(newline_re.findall(value))


def compile_rules(environment: "Environment") -> t.List[t.Tuple[str, str]]:
    """Compiles all the rules from the environment into a list of rules."""
    e = re.escape
    rules = [
        (
            len(environment.comment_start_string),
            TOKEN_COMMENT_BEGIN,
            e(environment.comment_start_string),
        ),
        (
            len(environment.block_start_string),
            TOKEN_BLOCK_BEGIN,
            e(environment.block_start_string),
        ),
        (
            len(environment.variable_start_string),
            TOKEN_VARIABLE_BEGIN,
            e(environment.variable_start_string),
        ),
    ]

    if environment.line_statement_prefix is not None:
        rules.append(
            (
                len(environment.line_statement_prefix),
                TOKEN_LINESTATEMENT_BEGIN,
                r"^[ \t\v]*" + e(environment.line_statement_prefix),
            )
        )
    if environment.line_comment_prefix is not None:
        rules.append(
            (
                len(environment.line_comment_prefix),
                TOKEN_LINECOMMENT_BEGIN,
                r"(?:^|(?<=\S))[^\S\r\n]*" + e(environment.line_comment_prefix),
            )
        )

    return [x[1:] for x in sorted(rules, reverse=True)]


class Failure:
    """Class that raises a `TemplateSyntaxError` if called.
    Used by the `Lexer` to specify known errors.
    """

    def __init__(
        self, message: str, cls: t.Type[TemplateSyntaxError] = TemplateSyntaxError
    ) -> None:
        self.message = message
        self.error_class = cls

    def __call__(self, lineno: int, filename: t.Optional[str]) -> "te.NoReturn":
        raise self.error_class(self.message, lineno, filename)


class Token(t.NamedTuple):
    lineno: int
    type: str
    value: str

    def __str__(self) -> str:
        return describe_token(self)

    def test(self, expr: str) -> bool:
        """Test a token against a token expression.  This can either be a
        token type or ``'token_type:token_value'``.  This can only test
        against string values and types.
        """
        # here we do a regular string equality check as test_any is usually
        # passed an iterable of not interned strings.
        if self.type == expr:
            return True

        if ":" in expr:
            return expr.split(":", 1) == [self.type, self.value]

        return False

    def test_any(self, *iterable: str) -> bool:
        """Test against multiple token expressions."""
        return any(self.test(expr) for expr in iterable)


class TokenStreamIterator:
    """The iterator for tokenstreams.  Iterate over the stream
    until the eof token is reached.
    """

    def __init__(self, stream: "TokenStream") -> None:
        self.stream = stream

    def __iter__(self) -> "TokenStreamIterator":
        return self

    def __next__(self) -> Token:
        token = self.stream.current

        if token.type is TOKEN_EOF:
            self.stream.close()
            raise StopIteration

        next(self.stream)
        return token


class TokenStream:
    """A token stream is an iterable that yields :class:`Token`\\s.  The
    parser however does not iterate over it but calls :meth:`next` to go
    one token ahead.  The current active token is stored as :attr:`current`.
    """

    def __init__(
        self,
        generator: t.Iterable[Token],
        name: t.Optional[str],
        filename: t.Optional[str],
    ):
        self._iter = iter(generator)
        self._pushed: te.Deque[Token] = deque()
        self.name = name
        self.filename = filename
        self.closed = False
        self.current = Token(1, TOKEN_INITIAL, "")
        next(self)

    def __iter__(self) -> TokenStreamIterator:
        return TokenStreamIterator(self)

    def __bool__(self) -> bool:
        return bool(self._pushed) or self.current.type is not TOKEN_EOF

    @property
    def eos(self) -> bool:
        """Are we at the end of the stream?"""
        return not self

    def push(self, token: Token) -> None:
        """Push a token back to the stream."""
        self._pushed.append(token)

    def look(self) -> Token:
        """Look at the next token."""
        old_token = next(self)
        result = self.current
        self.push(result)
        self.current = old_token
        return result

    def skip(self, n: int = 1) -> None:
        """Got n tokens ahead."""
        for _ in range(n):
            next(self)

    def next_if(self, expr: str) -> t.Optional[Token]:
        """Perform the token test and return the token if it matched.
        Otherwise the return value is `None`.
        """
        if self.current.test(expr):
            return next(self)

        return None

    def skip_if(self, expr: str) -> bool:
        """Like :meth:`next_if` but only returns `True` or `False`."""
        return self.next_if(expr) is not None

    def __next__(self) -> Token:
        """Go one token ahead and return the old one.

        Use the built-in :func:`next` instead of calling this directly.
        """
        rv = self.current

        if self._pushed:
            self.current = self._pushed.popleft()
        elif self.current.type is not TOKEN_EOF:
            try:
                self.current = next(self._iter)
            except StopIteration:
                self.close()

        return rv

    def close(self) -> None:
        """Close the stream."""
        self.current = Token(self.current.lineno, TOKEN_EOF, "")
        self._iter = iter(())
        self.closed = True

    def expect(self, expr: str) -> Token:
        """Expect a given token type and return it.  This accepts the same
        argument as :meth:`jinja2.lexer.Token.test`.
        """
        if not self.current.test(expr):
            expr = describe_token_expr(expr)

            if self.current.type is TOKEN_EOF:
                raise TemplateSyntaxError(
                    f"unexpected end of template, expected {expr!r}.",
                    self.current.lineno,
                    self.name,
                    self.filename,
                )

            raise TemplateSyntaxError(
                f"expected token {expr!r}, got {describe_token(self.current)!r}",
                self.current.lineno,
                self.name,
                self.filename,
            )

        return next(self)


def get_lexer(environment: "Environment") -> "Lexer":
    """Return a lexer which is probably cached."""
    key = (
        environment.block_start_string,
        environment.block_end_string,
        environment.variable_start_string,
        environment.variable_end_string,
        environment.comment_start_string,
        environment.comment_end_string,
        environment.line_statement_prefix,
        environment.line_comment_prefix,
        environment.trim_blocks,
        environment.lstrip_blocks,
        environment.newline_sequence,
        environment.keep_trailing_newline,
    )
    lexer = _lexer_cache.get(key)

    if lexer is None:
        _lexer_cache[key] = lexer = Lexer(environment)

    return lexer


class OptionalLStrip(tuple):  # type: ignore[type-arg]
    """A special tuple for marking a point in the state that can have
    lstrip applied.
    """

    __slots__ = ()

    # Even though it looks like a no-op, creating instances fails
    # without this.
    def __new__(cls, *members, **kwargs):  # type: ignore
        return super().__new__(cls, members)


class _Rule(t.NamedTuple):
    pattern: t.Pattern[str]
    tokens: t.Union[str, t.Tuple[str, ...], t.Tuple[Failure]]
    command: t.Optional[str]


class Lexer:
    """Class that implements a lexer for a given environment. Automatically
    created by the environment class, usually you don't have to do that.

    Note that the lexer is not automatically bound to an environment.
    Multiple environments can share the same lexer.
    """

    def __init__(self, environment: "Environment") -> None:
        # shortcuts
        e = re.escape

        def c(x: str) -> t.Pattern[str]:
            return re.compile(x, re.M | re.S)

        # lexing rules for tags
        tag_rules: t.List[_Rule] = [
            _Rule(whitespace_re, TOKEN_WHITESPACE, None),
            _Rule(float_re, TOKEN_FLOAT, None),
            _Rule(integer_re, TOKEN_INTEGER, None),
            _Rule(name_re, TOKEN_NAME, None),
            _Rule(string_re, TOKEN_STRING, None),
            _Rule(operator_re, TOKEN_OPERATOR, None),
        ]

        # assemble the root lexing rule. because "|" is ungreedy
        # we have to sort by length so that the lexer continues working
        # as expected when we have parsing rules like <% for block and
        # <%= for variables. (if someone wants asp like syntax)
        # variables are just part of the rules if variable processing
        # is required.
        root_tag_rules = compile_rules(environment)

        block_start_re = e(environment.block_start_string)
        block_end_re = e(environment.block_end_string)
        comment_end_re = e(environment.comment_end_string)
        variable_end_re = e(environment.variable_end_string)

        # block suffix if trimming is enabled
        block_suffix_re = "\\n?" if environment.trim_blocks else ""

        self.lstrip_blocks = environment.lstrip_blocks

        self.newline_sequence = environment.newline_sequence
        self.keep_trailing_newline = environment.keep_trailing_newline

        root_raw_re = (
            rf"(?P<raw_begin>{block_start_re}(\-|\+|)\s*raw\s*"
            rf"(?:\-{block_end_re}\s*|{block_end_re}))"
        )
        root_parts_re = "|".join(
            [root_raw_re] + [rf"(?P<{n}>{r}(\-|\+|))" for n, r in root_tag_rules]
        )

        # global lexing rules
        self.rules: t.Dict[str, t.List[_Rule]] = {
            "root": [
                # directives
                _Rule(
                    c(rf"(.*?)(?:{root_parts_re})"),
                    OptionalLStrip(TOKEN_DATA, "#bygroup"),  # type: ignore
                    "#bygroup",
                ),
                # data
                _Rule(c(".+"), TOKEN_DATA, None),
            ],
            # comments
            TOKEN_COMMENT_BEGIN: [
                _Rule(
                    c(
                        rf"(.*?)((?:\+{comment_end_re}|\-{comment_end_re}\s*"
                        rf"|{comment_end_re}{block_suffix_re}))"
                    ),
                    (TOKEN_COMMENT, TOKEN_COMMENT_END),
                    "#pop",
                ),
                _Rule(c(r"(.)"), (Failure("Missing end of comment tag"),), None),
            ],
            # blocks
            TOKEN_BLOCK_BEGIN: [
                _Rule(
                    c(
                        rf"(?:\+{block_end_re}|\-{block_end_re}\s*"
                        rf"|{block_end_re}{block_suffix_re})"
                    ),
                    TOKEN_BLOCK_END,
                    "#pop",
                ),
            ]
            + tag_rules,
            # variables
            TOKEN_VARIABLE_BEGIN: [
                _Rule(
                    c(rf"\-{variable_end_re}\s*|{variable_end_re}"),
                    TOKEN_VARIABLE_END,
                    "#pop",
                )
            ]
            + tag_rules,
            # raw block
            TOKEN_RAW_BEGIN: [
                _Rule(
                    c(
                        rf"(.*?)((?:{block_start_re}(\-|\+|))\s*endraw\s*"
                        rf"(?:\+{block_end_re}|\-{block_end_re}\s*"
                        rf"|{block_end_re}{block_suffix_re}))"
                    ),
                    OptionalLStrip(TOKEN_DATA, TOKEN_RAW_END),  # type: ignore
                    "#pop",
                ),
                _Rule(c(r"(.)"), (Failure("Missing end of raw directive"),), None),
            ],
            # line statements
            TOKEN_LINESTATEMENT_BEGIN: [
                _Rule(c(r"\s*(\n|$)"), TOKEN_LINESTATEMENT_END, "#pop")
            ]
            + tag_rules,
            # line comments
            TOKEN_LINECOMMENT_BEGIN: [
                _Rule(
                    c(r"(.*?)()(?=\n|$)"),
                    (TOKEN_LINECOMMENT, TOKEN_LINECOMMENT_END),
                    "#pop",
                )
            ],
        }

    def _normalize_newlines(self, value: str) -> str:
        """Replace all newlines with the configured sequence in strings
        and template data.
        """
        return newline_re.sub(self.newline_sequence, value)

    def tokenize(
        self,
        source: str,
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
        state: t.Optional[str] = None,
    ) -> TokenStream:
        """Calls tokeniter + tokenize and wraps it in a token stream."""
        stream = self.tokeniter(source, name, filename, state)
        return TokenStream(self.wrap(stream, name, filename), name, filename)

    def wrap(
        self,
        stream: t.Iterable[t.Tuple[int, str, str]],
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
    ) -> t.Iterator[Token]:
        """This is called with the stream as returned by `tokenize` and wraps
        every token in a :class:`Token` and converts the value.
        """
        for lineno, token, value_str in stream:
            if token in ignored_tokens:
                continue

            value: t.Any = value_str

            if token == TOKEN_LINESTATEMENT_BEGIN:
                token = TOKEN_BLOCK_BEGIN
            elif token == TOKEN_LINESTATEMENT_END:
                token = TOKEN_BLOCK_END
            # we are not interested in those tokens in the parser
            elif token in (TOKEN_RAW_BEGIN, TOKEN_RAW_END):
                continue
            elif token == TOKEN_DATA:
                value = self._normalize_newlines(value_str)
            elif token == "keyword":
                token = value_str
            elif token == TOKEN_NAME:
                value = value_str

                if not value.isidentifier():
                    raise TemplateSyntaxError(
                        "Invalid character in identifier", lineno, name, filename
                    )
            elif token == TOKEN_STRING:
                # try to unescape string
                try:
                    value = (
                        self._normalize_newlines(value_str[1:-1])
                        .encode("ascii", "backslashreplace")
                        .decode("unicode-escape")
                    )
                except Exception as e:
                    msg = str(e).split(":")[-1].strip()
                    raise TemplateSyntaxError(msg, lineno, name, filename) from e
            elif token == TOKEN_INTEGER:
                value = int(value_str.replace("_", ""), 0)
            elif token == TOKEN_FLOAT:
                # remove all "_" first to support more Python versions
                value = literal_eval(value_str.replace("_", ""))
            elif token == TOKEN_OPERATOR:
                token = operators[value_str]

            yield Token(lineno, token, value)

    def tokeniter(
        self,
        source: str,
        name: t.Optional[str],
        filename: t.Optional[str] = None,
        state: t.Optional[str] = None,
    ) -> t.Iterator[t.Tuple[int, str, str]]:
        """This method tokenizes the text and returns the tokens in a
        generator. Use this method if you just want to tokenize a template.

        .. versionchanged:: 3.0
            Only ``\\n``, ``\\r\\n`` and ``\\r`` are treated as line
            breaks.
        """
        lines = newline_re.split(source)[::2]

        if not self.keep_trailing_newline and lines[-1] == "":
            del lines[-1]

        source = "\n".join(lines)
        pos = 0
        lineno = 1
        stack = ["root"]

        if state is not None and state != "root":
            assert state in ("variable", "block"), "invalid state"
            stack.append(state + "_begin")

        statetokens = self.rules[stack[-1]]
        source_length = len(source)
        balancing_stack: t.List[str] = []
        newlines_stripped = 0
        line_starting = True

        while True:
            # tokenizer loop
            for regex, tokens, new_state in statetokens:
                m = regex.match(source, pos)

                # if no match we try again with the next rule
                if m is None:
                    continue

                # we only match blocks and variables if braces / parentheses
                # are balanced. continue parsing with the lower rule which
                # is the operator rule. do this only if the end tags look
                # like operators
                if balancing_stack and tokens in (
                    TOKEN_VARIABLE_END,
                    TOKEN_BLOCK_END,
                    TOKEN_LINESTATEMENT_END,
                ):
                    continue

                # tuples support more options
                if isinstance(tokens, tuple):
                    groups: t.Sequence[str] = m.groups()

                    if isinstance(tokens, OptionalLStrip):
                        # Rule supports lstrip. Match will look like
                        # text, block type, whitespace control, type, control, ...
                        text = groups[0]
                        # Skipping the text and first type, every other group is the
                        # whitespace control for each type. One of the groups will be
                        # -, +, or empty string instead of None.
                        strip_sign = next(g for g in groups[2::2] if g is not None)

                        if strip_sign == "-":
                            # Strip all whitespace between the text and the tag.
                            stripped = text.rstrip()
                            newlines_stripped = text[len(stripped) :].count("\n")
                            groups = [stripped, *groups[1:]]
                        elif (
                            # Not marked for preserving whitespace.
                            strip_sign != "+"
                            # lstrip is enabled.
                            and self.lstrip_blocks
                            # Not a variable expression.
                            and not m.groupdict().get(TOKEN_VARIABLE_BEGIN)
                        ):
                            # The start of text between the last newline and the tag.
                            l_pos = text.rfind("\n") + 1

                            if l_pos > 0 or line_starting:
                                # If there's only whitespace between the newline and the
                                # tag, strip it.
                                if whitespace_re.fullmatch(text, l_pos):
                                    groups = [text[:l_pos], *groups[1:]]

                    for idx, token in enumerate(tokens):
                        # failure group
                        if isinstance(token, Failure):
                            raise token(lineno, filename)
                        # bygroup is a bit more complex, in that case we
                        # yield for the current token the first named
                        # group that matched
                        elif token == "#bygroup":
                            for key, value in m.groupdict().items():
                                if value is not None:
                                    yield lineno, key, value
                                    lineno += value.count("\n")
                                    break
                            else:
                                raise RuntimeError(
                                    f"{regex!r} wanted to resolve the token dynamically"
                                    " but no group matched"
                                )
                        # normal group
                        else:
                            data = groups[idx]

                            if data or token not in ignore_if_empty:
                                yield lineno, token, data  # type: ignore[misc]

                            lineno += data.count("\n") + newlines_stripped
                            newlines_stripped = 0

                # strings as token just are yielded as it.
                else:
                    data = m.group()

                    # update brace/parentheses balance
                    if tokens == TOKEN_OPERATOR:
                        if data == "{":
                            balancing_stack.append("}")
                        elif data == "(":
                            balancing_stack.append(")")
                        elif data == "[":
                            balancing_stack.append("]")
                        elif data in ("}", ")", "]"):
                            if not balancing_stack:
                                raise TemplateSyntaxError(
                                    f"unexpected '{data}'", lineno, name, filename
                                )

                            expected_op = balancing_stack.pop()

                            if expected_op != data:
                                raise TemplateSyntaxError(
                                    f"unexpected '{data}', expected '{expected_op}'",
                                    lineno,
                                    name,
                                    filename,
                                )

                    # yield items
                    if data or tokens not in ignore_if_empty:
                        yield lineno, tokens, data

                    lineno += data.count("\n")

                line_starting = m.group()[-1:] == "\n"
                # fetch new position into new variable so that we can check
                # if there is a internal parsing error which would result
                # in an infinite loop
                pos2 = m.end()

                # handle state changes
                if new_state is not None:
                    # remove the uppermost state
                    if new_state == "#pop":
                        stack.pop()
                    # resolve the new state by group checking
                    elif new_state == "#bygroup":
                        for key, value in m.groupdict().items():
                            if value is not None:
                                stack.append(key)
                                break
                        else:
                            raise RuntimeError(
                                f"{regex!r} wanted to resolve the new state dynamically"
                                f" but no group matched"
                            )
                    # direct state name given
                    else:
                        stack.append(new_state)

                    statetokens = self.rules[stack[-1]]
                # we are still at the same position and no stack change.
                # this means a loop without break condition, avoid that and
                # raise error
                elif pos2 == pos:
                    raise RuntimeError(
                        f"{regex!r} yielded empty string without stack change"
                    )

                # publish new function and start again
                pos = pos2
                break
            # if loop terminated without break we haven't found a single match
            # either we are at the end of the file or we have a problem
            else:
                # end of text
                if pos >= source_length:
                    return

                # something went wrong
                raise TemplateSyntaxError(
                    f"unexpected char {source[pos]!r} at {pos}", lineno, name, filename
                )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\plugin\pytestplugin.py ===
# testing/plugin/pytestplugin.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

from __future__ import annotations

import argparse
import collections
from functools import update_wrapper
import inspect
import itertools
import operator
import os
import re
import sys
from typing import TYPE_CHECKING
import uuid

import pytest

try:
    # installed by bootstrap.py
    if not TYPE_CHECKING:
        import sqla_plugin_base as plugin_base
except ImportError:
    # assume we're a package, use traditional import
    from . import plugin_base


def pytest_addoption(parser):
    group = parser.getgroup("sqlalchemy")

    def make_option(name, **kw):
        callback_ = kw.pop("callback", None)
        if callback_:

            class CallableAction(argparse.Action):
                def __call__(
                    self, parser, namespace, values, option_string=None
                ):
                    callback_(option_string, values, parser)

            kw["action"] = CallableAction

        zeroarg_callback = kw.pop("zeroarg_callback", None)
        if zeroarg_callback:

            class CallableAction(argparse.Action):
                def __init__(
                    self,
                    option_strings,
                    dest,
                    default=False,
                    required=False,
                    help=None,  # noqa
                ):
                    super().__init__(
                        option_strings=option_strings,
                        dest=dest,
                        nargs=0,
                        const=True,
                        default=default,
                        required=required,
                        help=help,
                    )

                def __call__(
                    self, parser, namespace, values, option_string=None
                ):
                    zeroarg_callback(option_string, values, parser)

            kw["action"] = CallableAction

        group.addoption(name, **kw)

    plugin_base.setup_options(make_option)


def pytest_configure(config: pytest.Config):
    plugin_base.read_config(config.rootpath)
    if plugin_base.exclude_tags or plugin_base.include_tags:
        new_expr = " and ".join(
            list(plugin_base.include_tags)
            + [f"not {tag}" for tag in plugin_base.exclude_tags]
        )

        if config.option.markexpr:
            config.option.markexpr += f" and {new_expr}"
        else:
            config.option.markexpr = new_expr

    if config.pluginmanager.hasplugin("xdist"):
        config.pluginmanager.register(XDistHooks())

    if hasattr(config, "workerinput"):
        plugin_base.restore_important_follower_config(config.workerinput)
        plugin_base.configure_follower(config.workerinput["follower_ident"])
    else:
        if config.option.write_idents and os.path.exists(
            config.option.write_idents
        ):
            os.remove(config.option.write_idents)

    plugin_base.pre_begin(config.option)

    plugin_base.set_coverage_flag(
        bool(getattr(config.option, "cov_source", False))
    )

    plugin_base.set_fixture_functions(PytestFixtureFunctions)

    if config.option.dump_pyannotate:
        global DUMP_PYANNOTATE
        DUMP_PYANNOTATE = True


DUMP_PYANNOTATE = False


@pytest.fixture(autouse=True)
def collect_types_fixture():
    if DUMP_PYANNOTATE:
        from pyannotate_runtime import collect_types

        collect_types.start()
    yield
    if DUMP_PYANNOTATE:
        collect_types.stop()


def _log_sqlalchemy_info(session):
    import sqlalchemy
    from sqlalchemy import __version__
    from sqlalchemy.util import has_compiled_ext
    from sqlalchemy.util._has_cy import _CYEXTENSION_MSG

    greet = "sqlalchemy installation"
    site = "no user site" if sys.flags.no_user_site else "user site loaded"
    msgs = [
        f"SQLAlchemy {__version__} ({site})",
        f"Path: {sqlalchemy.__file__}",
    ]

    if has_compiled_ext():
        from sqlalchemy.cyextension import util

        msgs.append(f"compiled extension enabled, e.g. {util.__file__} ")
    else:
        msgs.append(f"compiled extension not enabled; {_CYEXTENSION_MSG}")

    pm = session.config.pluginmanager.get_plugin("terminalreporter")
    if pm:
        pm.write_sep("=", greet)
        for m in msgs:
            pm.write_line(m)
    else:
        # fancy pants reporter not found, fallback to plain print
        print("=" * 25, greet, "=" * 25)
        for m in msgs:
            print(m)


def pytest_sessionstart(session):
    from sqlalchemy.testing import asyncio

    _log_sqlalchemy_info(session)
    asyncio._assume_async(plugin_base.post_begin)


def pytest_sessionfinish(session):
    from sqlalchemy.testing import asyncio

    asyncio._maybe_async_provisioning(plugin_base.final_process_cleanup)

    if session.config.option.dump_pyannotate:
        from pyannotate_runtime import collect_types

        collect_types.dump_stats(session.config.option.dump_pyannotate)


def pytest_unconfigure(config):
    from sqlalchemy.testing import asyncio

    asyncio._shutdown()


def pytest_collection_finish(session):
    if session.config.option.dump_pyannotate:
        from pyannotate_runtime import collect_types

        lib_sqlalchemy = os.path.abspath("lib/sqlalchemy")

        def _filter(filename):
            filename = os.path.normpath(os.path.abspath(filename))
            if "lib/sqlalchemy" not in os.path.commonpath(
                [filename, lib_sqlalchemy]
            ):
                return None
            if "testing" in filename:
                return None

            return filename

        collect_types.init_types_collection(filter_filename=_filter)


class XDistHooks:
    def pytest_configure_node(self, node):
        from sqlalchemy.testing import provision
        from sqlalchemy.testing import asyncio

        # the master for each node fills workerinput dictionary
        # which pytest-xdist will transfer to the subprocess

        plugin_base.memoize_important_follower_config(node.workerinput)

        node.workerinput["follower_ident"] = "test_%s" % uuid.uuid4().hex[0:12]

        asyncio._maybe_async_provisioning(
            provision.create_follower_db, node.workerinput["follower_ident"]
        )

    def pytest_testnodedown(self, node, error):
        from sqlalchemy.testing import provision
        from sqlalchemy.testing import asyncio

        asyncio._maybe_async_provisioning(
            provision.drop_follower_db, node.workerinput["follower_ident"]
        )


def pytest_collection_modifyitems(session, config, items):
    # look for all those classes that specify __backend__ and
    # expand them out into per-database test cases.

    # this is much easier to do within pytest_pycollect_makeitem, however
    # pytest is iterating through cls.__dict__ as makeitem is
    # called which causes a "dictionary changed size" error on py3k.
    # I'd submit a pullreq for them to turn it into a list first, but
    # it's to suit the rather odd use case here which is that we are adding
    # new classes to a module on the fly.

    from sqlalchemy.testing import asyncio

    rebuilt_items = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    items[:] = [
        item
        for item in items
        if item.getparent(pytest.Class) is not None
        and not item.getparent(pytest.Class).name.startswith("_")
    ]

    test_classes = {item.getparent(pytest.Class) for item in items}

    def collect(element):
        for inst_or_fn in element.collect():
            if isinstance(inst_or_fn, pytest.Collector):
                yield from collect(inst_or_fn)
            else:
                yield inst_or_fn

    def setup_test_classes():
        for test_class in test_classes:
            # transfer legacy __backend__ and __sparse_backend__ symbols
            # to be markers
            if getattr(test_class.cls, "__backend__", False) or getattr(
                test_class.cls, "__only_on__", False
            ):
                add_markers = {"backend"}
            elif getattr(test_class.cls, "__sparse_backend__", False):
                add_markers = {"sparse_backend"}
            else:
                add_markers = frozenset()

            existing_markers = {
                mark.name for mark in test_class.iter_markers()
            }
            add_markers = add_markers - existing_markers
            all_markers = existing_markers.union(add_markers)

            for marker in add_markers:
                test_class.add_marker(marker)

            for sub_cls in plugin_base.generate_sub_tests(
                test_class.cls, test_class.module, all_markers
            ):
                if sub_cls is not test_class.cls:
                    per_cls_dict = rebuilt_items[test_class.cls]

                    module = test_class.getparent(pytest.Module)

                    new_cls = pytest.Class.from_parent(
                        name=sub_cls.__name__, parent=module
                    )
                    for marker in add_markers:
                        new_cls.add_marker(marker)

                    for fn in collect(new_cls):
                        per_cls_dict[fn.name].append(fn)

    # class requirements will sometimes need to access the DB to check
    # capabilities, so need to do this for async
    asyncio._maybe_async_provisioning(setup_test_classes)

    newitems = []
    for item in items:
        cls_ = item.cls
        if cls_ in rebuilt_items:
            newitems.extend(rebuilt_items[cls_][item.name])
        else:
            newitems.append(item)

    # seems like the functions attached to a test class aren't sorted already?
    # is that true and why's that? (when using unittest, they're sorted)
    items[:] = sorted(
        newitems,
        key=lambda item: (
            item.getparent(pytest.Module).name,
            item.getparent(pytest.Class).name,
            item.name,
        ),
    )


def pytest_pycollect_makeitem(collector, name, obj):
    if inspect.isclass(obj) and plugin_base.want_class(name, obj):
        from sqlalchemy.testing import config

        if config.any_async:
            obj = _apply_maybe_async(obj)

        return [
            pytest.Class.from_parent(
                name=parametrize_cls.__name__, parent=collector
            )
            for parametrize_cls in _parametrize_cls(collector.module, obj)
        ]
    elif (
        inspect.isfunction(obj)
        and collector.cls is not None
        and plugin_base.want_method(collector.cls, obj)
    ):
        # None means, fall back to default logic, which includes
        # method-level parametrize
        return None
    else:
        # empty list means skip this item
        return []


def _is_wrapped_coroutine_function(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__

    return inspect.iscoroutinefunction(fn)


def _apply_maybe_async(obj, recurse=True):
    from sqlalchemy.testing import asyncio

    for name, value in vars(obj).items():
        if (
            (callable(value) or isinstance(value, classmethod))
            and not getattr(value, "_maybe_async_applied", False)
            and (name.startswith("test_"))
            and not _is_wrapped_coroutine_function(value)
        ):
            is_classmethod = False
            if isinstance(value, classmethod):
                value = value.__func__
                is_classmethod = True

            @_pytest_fn_decorator
            def make_async(fn, *args, **kwargs):
                return asyncio._maybe_async(fn, *args, **kwargs)

            do_async = make_async(value)
            if is_classmethod:
                do_async = classmethod(do_async)
            do_async._maybe_async_applied = True

            setattr(obj, name, do_async)
    if recurse:
        for cls in obj.mro()[1:]:
            if cls != object:
                _apply_maybe_async(cls, False)
    return obj


def _parametrize_cls(module, cls):
    """implement a class-based version of pytest parametrize."""

    if "_sa_parametrize" not in cls.__dict__:
        return [cls]

    _sa_parametrize = cls._sa_parametrize
    classes = []
    for full_param_set in itertools.product(
        *[params for argname, params in _sa_parametrize]
    ):
        cls_variables = {}

        for argname, param in zip(
            [_sa_param[0] for _sa_param in _sa_parametrize], full_param_set
        ):
            if not argname:
                raise TypeError("need argnames for class-based combinations")
            argname_split = re.split(r",\s*", argname)
            for arg, val in zip(argname_split, param.values):
                cls_variables[arg] = val
        parametrized_name = "_".join(
            re.sub(r"\W", "", token)
            for param in full_param_set
            for token in param.id.split("-")
        )
        name = "%s_%s" % (cls.__name__, parametrized_name)
        newcls = type.__new__(type, name, (cls,), cls_variables)
        setattr(module, name, newcls)
        classes.append(newcls)
    return classes


_current_class = None


def pytest_runtest_setup(item):
    from sqlalchemy.testing import asyncio

    # pytest_runtest_setup runs *before* pytest fixtures with scope="class".
    # plugin_base.start_test_class_outside_fixtures may opt to raise SkipTest
    # for the whole class and has to run things that are across all current
    # databases, so we run this outside of the pytest fixture system altogether
    # and ensure asyncio greenlet if any engines are async

    global _current_class

    if isinstance(item, pytest.Function) and _current_class is None:
        asyncio._maybe_async_provisioning(
            plugin_base.start_test_class_outside_fixtures,
            item.cls,
        )
        _current_class = item.getparent(pytest.Class)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    # runs inside of pytest function fixture scope
    # after test function runs

    from sqlalchemy.testing import asyncio

    asyncio._maybe_async(plugin_base.after_test, item)

    yield
    # this is now after all the fixture teardown have run, the class can be
    # finalized. Since pytest v7 this finalizer can no longer be added in
    # pytest_runtest_setup since the class has not yet been setup at that
    # time.
    # See https://github.com/pytest-dev/pytest/issues/9343
    global _current_class, _current_report

    if _current_class is not None and (
        # last test or a new class
        nextitem is None
        or nextitem.getparent(pytest.Class) is not _current_class
    ):
        _current_class = None

        try:
            asyncio._maybe_async_provisioning(
                plugin_base.stop_test_class_outside_fixtures, item.cls
            )
        except Exception as e:
            # in case of an exception during teardown attach the original
            # error to the exception message, otherwise it will get lost
            if _current_report.failed:
                if not e.args:
                    e.args = (
                        "__Original test failure__:\n"
                        + _current_report.longreprtext,
                    )
                elif e.args[-1] and isinstance(e.args[-1], str):
                    args = list(e.args)
                    args[-1] += (
                        "\n__Original test failure__:\n"
                        + _current_report.longreprtext
                    )
                    e.args = tuple(args)
                else:
                    e.args += (
                        "__Original test failure__",
                        _current_report.longreprtext,
                    )
            raise
        finally:
            _current_report = None


def pytest_runtest_call(item):
    # runs inside of pytest function fixture scope
    # before test function runs

    from sqlalchemy.testing import asyncio

    asyncio._maybe_async(
        plugin_base.before_test,
        item,
        item.module.__name__,
        item.cls,
        item.name,
    )


_current_report = None


def pytest_runtest_logreport(report):
    global _current_report
    if report.when == "call":
        _current_report = report


@pytest.fixture(scope="class")
def setup_class_methods(request):
    from sqlalchemy.testing import asyncio

    cls = request.cls

    if hasattr(cls, "setup_test_class"):
        asyncio._maybe_async(cls.setup_test_class)

    yield

    if hasattr(cls, "teardown_test_class"):
        asyncio._maybe_async(cls.teardown_test_class)

    asyncio._maybe_async(plugin_base.stop_test_class, cls)


@pytest.fixture(scope="function")
def setup_test_methods(request):
    from sqlalchemy.testing import asyncio

    # called for each test

    self = request.instance

    # before this fixture runs:

    # 1. function level "autouse" fixtures under py3k (examples: TablesTest
    #    define tables / data, MappedTest define tables / mappers / data)

    # 2. was for p2k. no longer applies

    # 3. run outer xdist-style setup
    if hasattr(self, "setup_test"):
        asyncio._maybe_async(self.setup_test)

    # alembic test suite is using setUp and tearDown
    # xdist methods; support these in the test suite
    # for the near term
    if hasattr(self, "setUp"):
        asyncio._maybe_async(self.setUp)

    # inside the yield:
    # 4. function level fixtures defined on test functions themselves,
    #    e.g. "connection", "metadata" run next

    # 5. pytest hook pytest_runtest_call then runs

    # 6. test itself runs

    yield

    # yield finishes:

    # 7. function level fixtures defined on test functions
    #    themselves, e.g. "connection" rolls back the transaction, "metadata"
    #    emits drop all

    # 8. pytest hook pytest_runtest_teardown hook runs, this is associated
    #    with fixtures close all sessions, provisioning.stop_test_class(),
    #    engines.testing_reaper -> ensure all connection pool connections
    #    are returned, engines created by testing_engine that aren't the
    #    config engine are disposed

    asyncio._maybe_async(plugin_base.after_test_fixtures, self)

    # 10. run xdist-style teardown
    if hasattr(self, "tearDown"):
        asyncio._maybe_async(self.tearDown)

    if hasattr(self, "teardown_test"):
        asyncio._maybe_async(self.teardown_test)

    # 11. was for p2k. no longer applies

    # 12. function level "autouse" fixtures under py3k (examples: TablesTest /
    #    MappedTest delete table data, possibly drop tables and clear mappers
    #    depending on the flags defined by the test class)


def _pytest_fn_decorator(target):
    """Port of langhelpers.decorator with pytest-specific tricks."""

    from sqlalchemy.util.langhelpers import format_argspec_plus
    from sqlalchemy.util.compat import inspect_getfullargspec

    def _exec_code_in_env(code, env, fn_name):
        # note this is affected by "from __future__ import annotations" at
        # the top; exec'ed code will use non-evaluated annotations
        # which allows us to be more flexible with code rendering
        # in format_argpsec_plus()
        exec(code, env)
        return env[fn_name]

    def decorate(fn, add_positional_parameters=()):
        spec = inspect_getfullargspec(fn)
        if add_positional_parameters:
            spec.args.extend(add_positional_parameters)

        metadata = dict(
            __target_fn="__target_fn", __orig_fn="__orig_fn", name=fn.__name__
        )
        metadata.update(format_argspec_plus(spec, grouped=False))
        code = (
            """\
def %(name)s%(grouped_args)s:
    return %(__target_fn)s(%(__orig_fn)s, %(apply_kw)s)
"""
            % metadata
        )
        decorated = _exec_code_in_env(
            code, {"__target_fn": target, "__orig_fn": fn}, fn.__name__
        )
        if not add_positional_parameters:
            decorated.__defaults__ = getattr(fn, "__func__", fn).__defaults__
            decorated.__wrapped__ = fn
            return update_wrapper(decorated, fn)
        else:
            # this is the pytest hacky part.  don't do a full update wrapper
            # because pytest is really being sneaky about finding the args
            # for the wrapped function
            decorated.__module__ = fn.__module__
            decorated.__name__ = fn.__name__
            if hasattr(fn, "pytestmark"):
                decorated.pytestmark = fn.pytestmark
            return decorated

    return decorate


class PytestFixtureFunctions(plugin_base.FixtureFunctions):
    def skip_test_exception(self, *arg, **kw):
        return pytest.skip.Exception(*arg, **kw)

    @property
    def add_to_marker(self):
        return pytest.mark

    def mark_base_test_class(self):
        return pytest.mark.usefixtures(
            "setup_class_methods", "setup_test_methods"
        )

    _combination_id_fns = {
        "i": lambda obj: obj,
        "r": repr,
        "s": str,
        "n": lambda obj: (
            obj.__name__ if hasattr(obj, "__name__") else type(obj).__name__
        ),
    }

    def combinations(self, *arg_sets, **kw):
        """Facade for pytest.mark.parametrize.

        Automatically derives argument names from the callable which in our
        case is always a method on a class with positional arguments.

        ids for parameter sets are derived using an optional template.

        """
        from sqlalchemy.testing import exclusions

        if len(arg_sets) == 1 and hasattr(arg_sets[0], "__next__"):
            arg_sets = list(arg_sets[0])

        argnames = kw.pop("argnames", None)

        def _filter_exclusions(args):
            result = []
            gathered_exclusions = []
            for a in args:
                if isinstance(a, exclusions.compound):
                    gathered_exclusions.append(a)
                else:
                    result.append(a)

            return result, gathered_exclusions

        id_ = kw.pop("id_", None)

        tobuild_pytest_params = []
        has_exclusions = False
        if id_:
            _combination_id_fns = self._combination_id_fns

            # because itemgetter is not consistent for one argument vs.
            # multiple, make it multiple in all cases and use a slice
            # to omit the first argument
            _arg_getter = operator.itemgetter(
                0,
                *[
                    idx
                    for idx, char in enumerate(id_)
                    if char in ("n", "r", "s", "a")
                ],
            )
            fns = [
                (operator.itemgetter(idx), _combination_id_fns[char])
                for idx, char in enumerate(id_)
                if char in _combination_id_fns
            ]

            for arg in arg_sets:
                if not isinstance(arg, tuple):
                    arg = (arg,)

                fn_params, param_exclusions = _filter_exclusions(arg)

                parameters = _arg_getter(fn_params)[1:]

                if param_exclusions:
                    has_exclusions = True

                tobuild_pytest_params.append(
                    (
                        parameters,
                        param_exclusions,
                        "-".join(
                            comb_fn(getter(arg)) for getter, comb_fn in fns
                        ),
                    )
                )

        else:
            for arg in arg_sets:
                if not isinstance(arg, tuple):
                    arg = (arg,)

                fn_params, param_exclusions = _filter_exclusions(arg)

                if param_exclusions:
                    has_exclusions = True

                tobuild_pytest_params.append(
                    (fn_params, param_exclusions, None)
                )

        pytest_params = []
        for parameters, param_exclusions, id_ in tobuild_pytest_params:
            if has_exclusions:
                parameters += (param_exclusions,)

            param = pytest.param(*parameters, id=id_)
            pytest_params.append(param)

        def decorate(fn):
            if inspect.isclass(fn):
                if has_exclusions:
                    raise NotImplementedError(
                        "exclusions not supported for class level combinations"
                    )
                if "_sa_parametrize" not in fn.__dict__:
                    fn._sa_parametrize = []
                fn._sa_parametrize.append((argnames, pytest_params))
                return fn
            else:
                _fn_argnames = inspect.getfullargspec(fn).args[1:]
                if argnames is None:
                    _argnames = _fn_argnames
                else:
                    _argnames = re.split(r", *", argnames)

                if has_exclusions:
                    existing_exl = sum(
                        1 for n in _fn_argnames if n.startswith("_exclusions")
                    )
                    current_exclusion_name = f"_exclusions_{existing_exl}"
                    _argnames += [current_exclusion_name]

                    @_pytest_fn_decorator
                    def check_exclusions(fn, *args, **kw):
                        _exclusions = args[-1]
                        if _exclusions:
                            exlu = exclusions.compound().add(*_exclusions)
                            fn = exlu(fn)
                        return fn(*args[:-1], **kw)

                    fn = check_exclusions(
                        fn, add_positional_parameters=(current_exclusion_name,)
                    )

                return pytest.mark.parametrize(_argnames, pytest_params)(fn)

        return decorate

    def param_ident(self, *parameters):
        ident = parameters[0]
        return pytest.param(*parameters[1:], id=ident)

    def fixture(self, *arg, **kw):
        from sqlalchemy.testing import config
        from sqlalchemy.testing import asyncio

        # wrapping pytest.fixture function.  determine if
        # decorator was called as @fixture or @fixture().
        if len(arg) > 0 and callable(arg[0]):
            # was called as @fixture(), we have the function to wrap.
            fn = arg[0]
            arg = arg[1:]
        else:
            # was called as @fixture, don't have the function yet.
            fn = None

        # create a pytest.fixture marker.  because the fn is not being
        # passed, this is always a pytest.FixtureFunctionMarker()
        # object (or whatever pytest is calling it when you read this)
        # that is waiting for a function.
        fixture = pytest.fixture(*arg, **kw)

        # now apply wrappers to the function, including fixture itself

        def wrap(fn):
            if config.any_async:
                fn = asyncio._maybe_async_wrapper(fn)
            # other wrappers may be added here

            # now apply FixtureFunctionMarker
            fn = fixture(fn)

            return fn

        if fn:
            return wrap(fn)
        else:
            return wrap

    def get_current_test_name(self):
        return os.environ.get("PYTEST_CURRENT_TEST")

    def async_test(self, fn):
        from sqlalchemy.testing import asyncio

        @_pytest_fn_decorator
        def decorate(fn, *args, **kwargs):
            asyncio._run_coroutine_function(fn, *args, **kwargs)

        return decorate(fn)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\lineeditor\lineobj.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


import pyreadline3.clipboard as clipboard
from pyreadline3.unicode_helper import biter, ensure_unicode

from . import wordmatcher

# set to true to copy every addition to kill ring to clipboard
kill_ring_to_clipboard = False


class NotAWordError(IndexError):
    pass


def quote_char(c):
    if ord(c) > 0:
        return c


# ############# Line positioner ########################


class LinePositioner(object):
    def __call__(self, line):
        NotImplementedError("Base class !!!")


class NextChar(LinePositioner):
    def __call__(self, line):
        if line.point < len(line.line_buffer):
            return line.point + 1
        else:
            return line.point


NextChar = NextChar()


class PrevChar(LinePositioner):
    def __call__(self, line):
        if line.point > 0:
            return line.point - 1
        else:
            return line.point


PrevChar = PrevChar()


class NextWordStart(LinePositioner):
    def __call__(self, line):
        return line.next_start_segment(line.line_buffer, line.is_word_token)[line.point]


NextWordStart = NextWordStart()


class NextWordEnd(LinePositioner):
    def __call__(self, line):
        return line.next_end_segment(line.line_buffer, line.is_word_token)[line.point]


NextWordEnd = NextWordEnd()


class PrevWordStart(LinePositioner):
    def __call__(self, line):
        return line.prev_start_segment(line.line_buffer, line.is_word_token)[line.point]


PrevWordStart = PrevWordStart()


class WordStart(LinePositioner):
    def __call__(self, line):
        if line.is_word_token(line.get_line_text()[Point(line) : Point(line) + 1]):
            if Point(line) > 0 and line.is_word_token(
                line.get_line_text()[Point(line) - 1 : Point(line)]
            ):
                return PrevWordStart(line)
            else:
                return line.point
        else:
            raise NotAWordError("Point is not in a word")


WordStart = WordStart()


class WordEnd(LinePositioner):
    def __call__(self, line):
        if line.is_word_token(line.get_line_text()[Point(line) : Point(line) + 1]):
            if line.is_word_token(
                line.get_line_text()[Point(line) + 1 : Point(line) + 2]
            ):
                return NextWordEnd(line)
            else:
                return line.point
        else:
            raise NotAWordError("Point is not in a word")


WordEnd = WordEnd()


class PrevWordEnd(LinePositioner):
    def __call__(self, line):
        return line.prev_end_segment(line.line_buffer, line.is_word_token)[line.point]


PrevWordEnd = PrevWordEnd()


class PrevSpace(LinePositioner):
    def __call__(self, line):
        point = line.point
        if line[point - 1 : point].get_line_text() == " ":
            while point > 0 and line[point - 1 : point].get_line_text() == " ":
                point -= 1
        while point > 0 and line[point - 1 : point].get_line_text() != " ":
            point -= 1
        return point


PrevSpace = PrevSpace()


class StartOfLine(LinePositioner):
    def __call__(self, line):
        return 0


StartOfLine = StartOfLine()


class EndOfLine(LinePositioner):
    def __call__(self, line):
        return len(line.line_buffer)


EndOfLine = EndOfLine()


class Point(LinePositioner):
    def __call__(self, line):
        return line.point


Point = Point()


class Mark(LinePositioner):
    def __call__(self, line):
        return line.mark


k = Mark()

all_positioners = sorted(
    [
        (value.__class__.__name__, value)
        for key, value in globals().items()
        if isinstance(value, LinePositioner)
    ]
)

# ############## LineSlice #################


class LineSlice(object):
    def __call__(self, line):
        NotImplementedError("Base class !!!")


class CurrentWord(LineSlice):
    def __call__(self, line):
        return slice(WordStart(line), WordEnd(line), None)


CurrentWord = CurrentWord()


class NextWord(LineSlice):
    def __call__(self, line):
        work = TextLine(line)
        work.point = NextWordStart
        start = work.point
        stop = NextWordEnd(work)
        return slice(start, stop)


NextWord = NextWord()


class PrevWord(LineSlice):
    def __call__(self, line):
        work = TextLine(line)
        work.point = PrevWordEnd
        stop = work.point
        start = PrevWordStart(work)
        return slice(start, stop)


PrevWord = PrevWord()


class PointSlice(LineSlice):
    def __call__(self, line):
        return slice(Point(line), Point(line) + 1, None)


PointSlice = PointSlice()


# ##############  TextLine  ######################


class TextLine(object):
    def __init__(self, txtstr, point=None, mark=None):
        self.line_buffer = []
        self._point = 0
        self.mark = -1
        self.undo_stack = []
        self.overwrite = False
        if isinstance(txtstr, TextLine):  # copy
            self.line_buffer = txtstr.line_buffer[:]
            if point is None:
                self.point = txtstr.point
            else:
                self.point = point
            if mark is None:
                self.mark = txtstr.mark
            else:
                self.mark = mark
        else:
            self._insert_text(txtstr)
            if point is None:
                self.point = 0
            else:
                self.point = point
            if mark is None:
                self.mark = -1
            else:
                self.mark = mark

        self.is_word_token = wordmatcher.is_word_token
        self.next_start_segment = wordmatcher.next_start_segment
        self.next_end_segment = wordmatcher.next_end_segment
        self.prev_start_segment = wordmatcher.prev_start_segment
        self.prev_end_segment = wordmatcher.prev_end_segment

    def push_undo(self):
        l_text = self.get_line_text()
        if self.undo_stack and l_text == self.undo_stack[-1].get_line_text():
            self.undo_stack[-1].point = self.point
        else:
            self.undo_stack.append(self.copy())

    def pop_undo(self):
        if len(self.undo_stack) >= 2:
            self.undo_stack.pop()
            self.set_top_undo()
            self.undo_stack.pop()
        else:
            self.reset_line()
            self.undo_stack = []

    def set_top_undo(self):
        if self.undo_stack:
            undo = self.undo_stack[-1]
            self.line_buffer = undo.line_buffer
            self.point = undo.point
            self.mark = undo.mark
        else:
            pass

    def __repr__(self):
        return 'TextLine("%s",point=%s,mark=%s)' % (
            self.line_buffer,
            self.point,
            self.mark,
        )

    def copy(self):
        return self.__class__(self)

    def set_point(self, value):
        if isinstance(value, LinePositioner):
            value = value(self)
        assert value <= len(self.line_buffer)
        if value > len(self.line_buffer):
            value = len(self.line_buffer)
        self._point = value

    def get_point(self):
        return self._point

    point = property(get_point, set_point)

    def visible_line_width(self, position=Point):
        """Return the visible width of the text up to position."""
        extra_char_width = len(
            [None for c in self[:position].line_buffer if 0x2013 <= ord(c) <= 0xFFFD]
        )
        return (
            len(self[:position].quoted_text())
            + self[:position].line_buffer.count("\t") * 7
            + extra_char_width
        )

    def quoted_text(self):
        quoted = [quote_char(c) for c in self.line_buffer]
        return "".join(map(ensure_unicode, quoted))

    def get_line_text(self):
        buf = self.line_buffer
        buf = list(map(ensure_unicode, buf))
        return "".join(buf)

    def set_line(self, text, cursor=None):
        self.line_buffer = [c for c in str(text)]
        if cursor is None:
            self.point = len(self.line_buffer)
        else:
            self.point = cursor

    def reset_line(self):
        self.line_buffer = []
        self.point = 0

    def end_of_line(self):
        self.point = len(self.line_buffer)

    def _insert_text(self, text, argument=1):
        text = text * argument
        if self.overwrite:
            for c in biter(text):
                # if self.point:
                self.line_buffer[self.point] = c
                self.point += 1
        else:
            for c in biter(text):
                self.line_buffer.insert(self.point, c)
                self.point += 1

    def __getitem__(self, key):
        # Check if key is LineSlice, convert to regular slice
        # and continue processing
        if isinstance(key, LineSlice):
            key = key(self)
        if isinstance(key, slice):
            if key.step is None:
                pass
            else:
                raise RuntimeError('step is not "None"')
            if key.start is None:
                start = StartOfLine(self)
            elif isinstance(key.start, LinePositioner):
                start = key.start(self)
            else:
                start = key.start
            if key.stop is None:
                stop = EndOfLine(self)
            elif isinstance(key.stop, LinePositioner):
                stop = key.stop(self)
            else:
                stop = key.stop
            return self.__class__(self.line_buffer[start:stop], point=0)
        elif isinstance(key, LinePositioner):
            return self.line_buffer[key(self)]
        elif isinstance(key, tuple):
            # Multiple slice not allowed
            raise IndexError("Cannot use step in line buffer indexing")
        else:
            # return TextLine(self.line_buffer[key])
            return self.line_buffer[key]

    def __delitem__(self, key):
        point = self.point
        if isinstance(key, LineSlice):
            key = key(self)
        if isinstance(key, slice):
            start = key.start
            stop = key.stop
            if isinstance(start, LinePositioner):
                start = start(self)
            elif start is None:
                start = 0
            if isinstance(stop, LinePositioner):
                stop = stop(self)
            elif stop is None:
                stop = EndOfLine(self)
        elif isinstance(key, LinePositioner):
            start = key(self)
            stop = start + 1
        else:
            start = key
            stop = key + 1
        prev = self.line_buffer[:start]
        rest = self.line_buffer[stop:]
        self.line_buffer = prev + rest
        if point > stop:
            self.point = point - (stop - start)
        elif point >= start and point <= stop:
            self.point = start

    def __setitem__(self, key, value):
        if isinstance(key, LineSlice):
            key = key(self)
        if isinstance(key, slice):
            start = key.start
            stop = key.stop
        elif isinstance(key, LinePositioner):
            start = key(self)
            stop = start + 1
        else:
            start = key
            stop = key + 1
        prev = self.line_buffer[:start]
        value = self.__class__(value).line_buffer
        rest = self.line_buffer[stop:]
        out = prev + value + rest
        if len(out) >= len(self):
            self.point = len(self)
        self.line_buffer = out

    def __len__(self):
        return len(self.line_buffer)

    def upper(self):
        self.line_buffer = [x.upper() for x in self.line_buffer]
        return self

    def lower(self):
        self.line_buffer = [x.lower() for x in self.line_buffer]
        return self

    def capitalize(self):
        self.set_line(self.get_line_text().capitalize(), self.point)
        return self

    def startswith(self, txt):
        return self.get_line_text().startswith(txt)

    def endswith(self, txt):
        return self.get_line_text().endswith(txt)

    def __contains__(self, txt):
        return txt in self.get_line_text()


class ReadLineTextBuffer(TextLine):
    def __init__(self, txtstr, point=None, mark=None):
        super().__init__(txtstr, point, mark)

        self.enable_win32_clipboard = True
        self.selection_mark = -1
        self.enable_selection = True
        self.kill_ring = []

    def __repr__(self):
        return "ReadLineTextBuffer" '("%s",point=%s,mark=%s,selection_mark=%s)' % (
            self.line_buffer,
            self.point,
            self.mark,
            self.selection_mark,
        )

    def insert_text(self, char, argument=1):
        self.delete_selection()
        self.selection_mark = -1
        self._insert_text(char, argument)

    def to_clipboard(self):
        if self.enable_win32_clipboard:
            clipboard.set_clipboard_text(self.get_line_text())

    # Movement

    def beginning_of_line(self):
        self.selection_mark = -1
        self.point = StartOfLine

    def end_of_line(self):
        self.selection_mark = -1
        self.point = EndOfLine

    def forward_char(self, argument=1):
        if argument < 0:
            self.backward_char(-argument)
        self.selection_mark = -1
        for _ in range(argument):
            self.point = NextChar

    def backward_char(self, argument=1):
        if argument < 0:
            self.forward_char(-argument)
        self.selection_mark = -1
        for _ in range(argument):
            self.point = PrevChar

    def forward_word(self, argument=1):
        if argument < 0:
            self.backward_word(-argument)
        self.selection_mark = -1
        for _ in range(argument):
            self.point = NextWordStart

    def backward_word(self, argument=1):
        if argument < 0:
            self.forward_word(-argument)
        self.selection_mark = -1
        for _ in range(argument):
            self.point = PrevWordStart

    def forward_word_end(self, argument=1):
        if argument < 0:
            self.backward_word_end(-argument)
        self.selection_mark = -1
        for _ in range(argument):
            self.point = NextWordEnd

    def backward_word_end(self, argument=1):
        if argument < 0:
            self.forward_word_end(-argument)
        self.selection_mark = -1
        for _ in range(argument):
            self.point = NextWordEnd

    # Movement select
    def beginning_of_line_extend_selection(self):
        if self.enable_selection and self.selection_mark < 0:
            self.selection_mark = self.point
        self.point = StartOfLine

    def end_of_line_extend_selection(self):
        if self.enable_selection and self.selection_mark < 0:
            self.selection_mark = self.point
        self.point = EndOfLine

    def forward_char_extend_selection(self, argument=1):
        if argument < 0:
            self.backward_char_extend_selection(-argument)
        if self.enable_selection and self.selection_mark < 0:
            self.selection_mark = self.point
        for _ in range(argument):
            self.point = NextChar

    def backward_char_extend_selection(self, argument=1):
        if argument < 0:
            self.forward_char_extend_selection(-argument)
        if self.enable_selection and self.selection_mark < 0:
            self.selection_mark = self.point
        for _ in range(argument):
            self.point = PrevChar

    def forward_word_extend_selection(self, argument=1):
        if argument < 0:
            self.backward_word_extend_selection(-argument)
        if self.enable_selection and self.selection_mark < 0:
            self.selection_mark = self.point
        for _ in range(argument):
            self.point = NextWordStart

    def backward_word_extend_selection(self, argument=1):
        if argument < 0:
            self.forward_word_extend_selection(-argument)
        if self.enable_selection and self.selection_mark < 0:
            self.selection_mark = self.point
        for _ in range(argument):
            self.point = PrevWordStart

    def forward_word_end_extend_selection(self, argument=1):
        if argument < 0:
            self.backward_word_end_extend_selection(-argument)
        if self.enable_selection and self.selection_mark < 0:
            self.selection_mark = self.point
        for _ in range(argument):
            self.point = NextWordEnd

    def backward_word_end_extend_selection(self, argument=1):
        if argument < 0:
            self.forward_word_end_extend_selection(-argument)
        if self.enable_selection and self.selection_mark < 0:
            self.selection_mark = self.point
        for _ in range(argument):
            self.point = PrevWordEnd

    # delete

    def delete_selection(self):
        if self.enable_selection and self.selection_mark >= 0:
            if self.selection_mark < self.point:
                del self[self.selection_mark : self.point]
                self.selection_mark = -1
            else:
                del self[self.point : self.selection_mark]
                self.selection_mark = -1
            return True
        else:
            self.selection_mark = -1
            return False

    def delete_char(self, argument=1):
        if argument < 0:
            self.backward_delete_char(-argument)
        if self.delete_selection():
            argument -= 1
        for _ in range(argument):
            del self[Point]

    def backward_delete_char(self, argument=1):
        if argument < 0:
            self.delete_char(-argument)
        if self.delete_selection():
            argument -= 1
        for _ in range(argument):
            if self.point > 0:
                self.backward_char()
                self.delete_char()

    def forward_delete_word(self, argument=1):
        if argument < 0:
            self.backward_delete_word(-argument)
        if self.delete_selection():
            argument -= 1
        for _ in range(argument):
            del self[Point:NextWordStart]

    def backward_delete_word(self, argument=1):
        if argument < 0:
            self.forward_delete_word(-argument)
        if self.delete_selection():
            argument -= 1
        for _ in range(argument):
            del self[PrevWordStart:Point]

    def delete_current_word(self):
        if not self.delete_selection():
            del self[CurrentWord]
        self.selection_mark = -1

    def delete_horizontal_space(self):
        if self[Point] in " \t":
            del self[PrevWordEnd:NextWordStart]
        self.selection_mark = -1

    # Case

    def upcase_word(self):
        p = self.point
        try:
            self[CurrentWord] = self[CurrentWord].upper()
            self.point = p
        except NotAWordError:
            pass

    def downcase_word(self):
        p = self.point
        try:
            self[CurrentWord] = self[CurrentWord].lower()
            self.point = p
        except NotAWordError:
            pass

    def capitalize_word(self):
        p = self.point
        try:
            self[CurrentWord] = self[CurrentWord].capitalize()
            self.point = p
        except NotAWordError:
            pass

    # Transpose

    def transpose_chars(self):
        p2 = Point(self)
        if p2 == 0:
            return
        elif p2 == len(self):
            p2 = p2 - 1
        p1 = p2 - 1
        self[p2], self[p1] = self[p1], self[p2]
        self.point = p2 + 1

    def transpose_words(self):
        word1 = TextLine(self)
        word2 = TextLine(self)
        if self.point == len(self):
            word2.point = PrevWordStart
            word1.point = PrevWordStart(word2)
        else:
            word1.point = PrevWordStart
            word2.point = NextWordStart
        stop1 = NextWordEnd(word1)
        stop2 = NextWordEnd(word2)
        start1 = word1.point
        start2 = word2.point
        self[start2:stop2] = word1[Point:NextWordEnd]
        self[start1:stop1] = word2[Point:NextWordEnd]
        self.point = stop2

    # Kill

    def kill_line(self):
        self.add_to_kill_ring(self[self.point :])
        del self.line_buffer[self.point :]

    def kill_whole_line(self):
        self.add_to_kill_ring(self[:])
        del self[:]

    def backward_kill_line(self):
        del self[StartOfLine:Point]

    def unix_line_discard(self):
        del self[StartOfLine:Point]

    def kill_word(self):
        """Kills to next word ending"""
        del self[Point:NextWordEnd]

    def backward_kill_word(self):
        """Kills to next word ending"""
        if not self.delete_selection():
            del self[PrevWordStart:Point]
        self.selection_mark = -1

    def forward_kill_word(self):
        """Kills to next word ending"""
        if not self.delete_selection():
            del self[Point:NextWordEnd]
        self.selection_mark = -1

    def unix_word_rubout(self):
        if not self.delete_selection():
            del self[PrevSpace:Point]
        self.selection_mark = -1

    def kill_region(self):
        pass

    def copy_region_as_kill(self):
        pass

    def copy_backward_word(self):
        pass

    def copy_forward_word(self):
        pass

    def yank(self):
        self.paste_from_kill_ring()

    def yank_pop(self):
        pass

    # Mark

    def set_mark(self):
        self.mark = self.point

    def exchange_point_and_mark(self):
        pass

    def copy_region_to_clipboard(self):  # ()
        """Copy the text in the region to the windows clipboard."""
        if self.enable_win32_clipboard:
            mark = min(self.mark, len(self.line_buffer))
            cursor = min(self.point, len(self.line_buffer))
            if self.mark == -1:
                return
            begin = min(cursor, mark)
            end = max(cursor, mark)
            toclipboard = "".join(self.line_buffer[begin:end])
            clipboard.set_clipboard_text(toclipboard)

    def copy_selection_to_clipboard(self):  # ()
        """Copy the text in the region to the windows clipboard."""
        if (
            self.enable_win32_clipboard
            and self.enable_selection
            and self.selection_mark >= 0
        ):
            selection_mark = min(self.selection_mark, len(self.line_buffer))
            cursor = min(self.point, len(self.line_buffer))
            if self.selection_mark == -1:
                return
            begin = min(cursor, selection_mark)
            end = max(cursor, selection_mark)
            toclipboard = "".join(self.line_buffer[begin:end])
            clipboard.set_clipboard_text(toclipboard)

    def cut_selection_to_clipboard(self):  # ()
        self.copy_selection_to_clipboard()
        self.delete_selection()

    # Paste

    # Kill ring

    def add_to_kill_ring(self, txt):
        self.kill_ring = [txt]
        if kill_ring_to_clipboard:
            clipboard.set_clipboard_text(txt.get_line_text())

    def paste_from_kill_ring(self):
        if self.kill_ring:
            self.insert_text(self.kill_ring[0])


##################################################################
q = ReadLineTextBuffer("asff asFArw  ewrWErhg", point=8)
q = TextLine("asff asFArw  ewrWErhg", point=8)


def show_pos(buff, pos, chr="."):
    l_n = len(buff.line_buffer)

    def choice(is_bool):
        if is_bool:
            return chr
        else:
            return " "

    return "".join([choice(pos == idx) for idx in range(l_n + 1)])


def test_positioner(buff, points, positioner):
    print((" %s " % positioner.__class__.__name__).center(40, "-"))
    buffstr = buff.line_buffer

    print('"%s"' % (buffstr))
    for point in points:
        b = TextLine(buff, point=point)
        out = [" "] * (len(buffstr) + 1)
        pos = positioner(b)
        if pos == point:
            out[pos] = "&"
        else:
            out[point] = "."
            out[pos] = "^"
        print('"%s"' % ("".join(out)))


if __name__ == "__main__":

    print('%-15s "%s"' % ("Position", q.get_line_text()))
    print('%-15s "%s"' % ("Point", show_pos(q, q.point)))

    for name, positioner_q in all_positioners:
        pos_q = positioner_q(q)

        print('%-15s "%s"' % (name, show_pos(q, pos_q, "^")))

    l_t = ReadLineTextBuffer("kjjk asads   asad")
    l_t.point = EndOfLine

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\packaging\metadata.py ===
from __future__ import annotations

import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import pathlib
import sys
import typing
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    TypedDict,
    cast,
)

from . import licenses, requirements, specifiers, utils
from . import version as version_module
from .licenses import NormalizedLicenseExpression

T = typing.TypeVar("T")


if sys.version_info >= (3, 11):  # pragma: no cover
    ExceptionGroup = ExceptionGroup
else:  # pragma: no cover

    class ExceptionGroup(Exception):
        """A minimal implementation of :external:exc:`ExceptionGroup` from Python 3.11.

        If :external:exc:`ExceptionGroup` is already defined by Python itself,
        that version is used instead.
        """

        message: str
        exceptions: list[Exception]

        def __init__(self, message: str, exceptions: list[Exception]) -> None:
            self.message = message
            self.exceptions = exceptions

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}({self.message!r}, {self.exceptions!r})"


class InvalidMetadata(ValueError):
    """A metadata field contains invalid data."""

    field: str
    """The name of the field that contains invalid data."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(message)


# The RawMetadata class attempts to make as few assumptions about the underlying
# serialization formats as possible. The idea is that as long as a serialization
# formats offer some very basic primitives in *some* way then we can support
# serializing to and from that format.
class RawMetadata(TypedDict, total=False):
    """A dictionary of raw core metadata.

    Each field in core metadata maps to a key of this dictionary (when data is
    provided). The key is lower-case and underscores are used instead of dashes
    compared to the equivalent core metadata field. Any core metadata field that
    can be specified multiple times or can hold multiple values in a single
    field have a key with a plural name. See :class:`Metadata` whose attributes
    match the keys of this dictionary.

    Core metadata fields that can be specified multiple times are stored as a
    list or dict depending on which is appropriate for the field. Any fields
    which hold multiple values in a single field are stored as a list.

    """

    # Metadata 1.0 - PEP 241
    metadata_version: str
    name: str
    version: str
    platforms: list[str]
    summary: str
    description: str
    keywords: list[str]
    home_page: str
    author: str
    author_email: str
    license: str

    # Metadata 1.1 - PEP 314
    supported_platforms: list[str]
    download_url: str
    classifiers: list[str]
    requires: list[str]
    provides: list[str]
    obsoletes: list[str]

    # Metadata 1.2 - PEP 345
    maintainer: str
    maintainer_email: str
    requires_dist: list[str]
    provides_dist: list[str]
    obsoletes_dist: list[str]
    requires_python: str
    requires_external: list[str]
    project_urls: dict[str, str]

    # Metadata 2.0
    # PEP 426 attempted to completely revamp the metadata format
    # but got stuck without ever being able to build consensus on
    # it and ultimately ended up withdrawn.
    #
    # However, a number of tools had started emitting METADATA with
    # `2.0` Metadata-Version, so for historical reasons, this version
    # was skipped.

    # Metadata 2.1 - PEP 566
    description_content_type: str
    provides_extra: list[str]

    # Metadata 2.2 - PEP 643
    dynamic: list[str]

    # Metadata 2.3 - PEP 685
    # No new fields were added in PEP 685, just some edge case were
    # tightened up to provide better interoptability.

    # Metadata 2.4 - PEP 639
    license_expression: str
    license_files: list[str]


_STRING_FIELDS = {
    "author",
    "author_email",
    "description",
    "description_content_type",
    "download_url",
    "home_page",
    "license",
    "license_expression",
    "maintainer",
    "maintainer_email",
    "metadata_version",
    "name",
    "requires_python",
    "summary",
    "version",
}

_LIST_FIELDS = {
    "classifiers",
    "dynamic",
    "license_files",
    "obsoletes",
    "obsoletes_dist",
    "platforms",
    "provides",
    "provides_dist",
    "provides_extra",
    "requires",
    "requires_dist",
    "requires_external",
    "supported_platforms",
}

_DICT_FIELDS = {
    "project_urls",
}


def _parse_keywords(data: str) -> list[str]:
    """Split a string of comma-separated keywords into a list of keywords."""
    return [k.strip() for k in data.split(",")]


def _parse_project_urls(data: list[str]) -> dict[str, str]:
    """Parse a list of label/URL string pairings separated by a comma."""
    urls = {}
    for pair in data:
        # Our logic is slightly tricky here as we want to try and do
        # *something* reasonable with malformed data.
        #
        # The main thing that we have to worry about, is data that does
        # not have a ',' at all to split the label from the Value. There
        # isn't a singular right answer here, and we will fail validation
        # later on (if the caller is validating) so it doesn't *really*
        # matter, but since the missing value has to be an empty str
        # and our return value is dict[str, str], if we let the key
        # be the missing value, then they'd have multiple '' values that
        # overwrite each other in a accumulating dict.
        #
        # The other potentional issue is that it's possible to have the
        # same label multiple times in the metadata, with no solid "right"
        # answer with what to do in that case. As such, we'll do the only
        # thing we can, which is treat the field as unparseable and add it
        # to our list of unparsed fields.
        parts = [p.strip() for p in pair.split(",", 1)]
        parts.extend([""] * (max(0, 2 - len(parts))))  # Ensure 2 items

        # TODO: The spec doesn't say anything about if the keys should be
        #       considered case sensitive or not... logically they should
        #       be case-preserving and case-insensitive, but doing that
        #       would open up more cases where we might have duplicate
        #       entries.
        label, url = parts
        if label in urls:
            # The label already exists in our set of urls, so this field
            # is unparseable, and we can just add the whole thing to our
            # unparseable data and stop processing it.
            raise KeyError("duplicate labels in project urls")
        urls[label] = url

    return urls


def _get_payload(msg: email.message.Message, source: bytes | str) -> str:
    """Get the body of the message."""
    # If our source is a str, then our caller has managed encodings for us,
    # and we don't need to deal with it.
    if isinstance(source, str):
        payload = msg.get_payload()
        assert isinstance(payload, str)
        return payload
    # If our source is a bytes, then we're managing the encoding and we need
    # to deal with it.
    else:
        bpayload = msg.get_payload(decode=True)
        assert isinstance(bpayload, bytes)
        try:
            return bpayload.decode("utf8", "strict")
        except UnicodeDecodeError as exc:
            raise ValueError("payload in an invalid encoding") from exc


# The various parse_FORMAT functions here are intended to be as lenient as
# possible in their parsing, while still returning a correctly typed
# RawMetadata.
#
# To aid in this, we also generally want to do as little touching of the
# data as possible, except where there are possibly some historic holdovers
# that make valid data awkward to work with.
#
# While this is a lower level, intermediate format than our ``Metadata``
# class, some light touch ups can make a massive difference in usability.

# Map METADATA fields to RawMetadata.
_EMAIL_TO_RAW_MAPPING = {
    "author": "author",
    "author-email": "author_email",
    "classifier": "classifiers",
    "description": "description",
    "description-content-type": "description_content_type",
    "download-url": "download_url",
    "dynamic": "dynamic",
    "home-page": "home_page",
    "keywords": "keywords",
    "license": "license",
    "license-expression": "license_expression",
    "license-file": "license_files",
    "maintainer": "maintainer",
    "maintainer-email": "maintainer_email",
    "metadata-version": "metadata_version",
    "name": "name",
    "obsoletes": "obsoletes",
    "obsoletes-dist": "obsoletes_dist",
    "platform": "platforms",
    "project-url": "project_urls",
    "provides": "provides",
    "provides-dist": "provides_dist",
    "provides-extra": "provides_extra",
    "requires": "requires",
    "requires-dist": "requires_dist",
    "requires-external": "requires_external",
    "requires-python": "requires_python",
    "summary": "summary",
    "supported-platform": "supported_platforms",
    "version": "version",
}
_RAW_TO_EMAIL_MAPPING = {raw: email for email, raw in _EMAIL_TO_RAW_MAPPING.items()}


def parse_email(data: bytes | str) -> tuple[RawMetadata, dict[str, list[str]]]:
    """Parse a distribution's metadata stored as email headers (e.g. from ``METADATA``).

    This function returns a two-item tuple of dicts. The first dict is of
    recognized fields from the core metadata specification. Fields that can be
    parsed and translated into Python's built-in types are converted
    appropriately. All other fields are left as-is. Fields that are allowed to
    appear multiple times are stored as lists.

    The second dict contains all other fields from the metadata. This includes
    any unrecognized fields. It also includes any fields which are expected to
    be parsed into a built-in type but were not formatted appropriately. Finally,
    any fields that are expected to appear only once but are repeated are
    included in this dict.

    """
    raw: dict[str, str | list[str] | dict[str, str]] = {}
    unparsed: dict[str, list[str]] = {}

    if isinstance(data, str):
        parsed = email.parser.Parser(policy=email.policy.compat32).parsestr(data)
    else:
        parsed = email.parser.BytesParser(policy=email.policy.compat32).parsebytes(data)

    # We have to wrap parsed.keys() in a set, because in the case of multiple
    # values for a key (a list), the key will appear multiple times in the
    # list of keys, but we're avoiding that by using get_all().
    for name in frozenset(parsed.keys()):
        # Header names in RFC are case insensitive, so we'll normalize to all
        # lower case to make comparisons easier.
        name = name.lower()

        # We use get_all() here, even for fields that aren't multiple use,
        # because otherwise someone could have e.g. two Name fields, and we
        # would just silently ignore it rather than doing something about it.
        headers = parsed.get_all(name) or []

        # The way the email module works when parsing bytes is that it
        # unconditionally decodes the bytes as ascii using the surrogateescape
        # handler. When you pull that data back out (such as with get_all() ),
        # it looks to see if the str has any surrogate escapes, and if it does
        # it wraps it in a Header object instead of returning the string.
        #
        # As such, we'll look for those Header objects, and fix up the encoding.
        value = []
        # Flag if we have run into any issues processing the headers, thus
        # signalling that the data belongs in 'unparsed'.
        valid_encoding = True
        for h in headers:
            # It's unclear if this can return more types than just a Header or
            # a str, so we'll just assert here to make sure.
            assert isinstance(h, (email.header.Header, str))

            # If it's a header object, we need to do our little dance to get
            # the real data out of it. In cases where there is invalid data
            # we're going to end up with mojibake, but there's no obvious, good
            # way around that without reimplementing parts of the Header object
            # ourselves.
            #
            # That should be fine since, if mojibacked happens, this key is
            # going into the unparsed dict anyways.
            if isinstance(h, email.header.Header):
                # The Header object stores it's data as chunks, and each chunk
                # can be independently encoded, so we'll need to check each
                # of them.
                chunks: list[tuple[bytes, str | None]] = []
                for bin, encoding in email.header.decode_header(h):
                    try:
                        bin.decode("utf8", "strict")
                    except UnicodeDecodeError:
                        # Enable mojibake.
                        encoding = "latin1"
                        valid_encoding = False
                    else:
                        encoding = "utf8"
                    chunks.append((bin, encoding))

                # Turn our chunks back into a Header object, then let that
                # Header object do the right thing to turn them into a
                # string for us.
                value.append(str(email.header.make_header(chunks)))
            # This is already a string, so just add it.
            else:
                value.append(h)

        # We've processed all of our values to get them into a list of str,
        # but we may have mojibake data, in which case this is an unparsed
        # field.
        if not valid_encoding:
            unparsed[name] = value
            continue

        raw_name = _EMAIL_TO_RAW_MAPPING.get(name)
        if raw_name is None:
            # This is a bit of a weird situation, we've encountered a key that
            # we don't know what it means, so we don't know whether it's meant
            # to be a list or not.
            #
            # Since we can't really tell one way or another, we'll just leave it
            # as a list, even though it may be a single item list, because that's
            # what makes the most sense for email headers.
            unparsed[name] = value
            continue

        # If this is one of our string fields, then we'll check to see if our
        # value is a list of a single item. If it is then we'll assume that
        # it was emitted as a single string, and unwrap the str from inside
        # the list.
        #
        # If it's any other kind of data, then we haven't the faintest clue
        # what we should parse it as, and we have to just add it to our list
        # of unparsed stuff.
        if raw_name in _STRING_FIELDS and len(value) == 1:
            raw[raw_name] = value[0]
        # If this is one of our list of string fields, then we can just assign
        # the value, since email *only* has strings, and our get_all() call
        # above ensures that this is a list.
        elif raw_name in _LIST_FIELDS:
            raw[raw_name] = value
        # Special Case: Keywords
        # The keywords field is implemented in the metadata spec as a str,
        # but it conceptually is a list of strings, and is serialized using
        # ", ".join(keywords), so we'll do some light data massaging to turn
        # this into what it logically is.
        elif raw_name == "keywords" and len(value) == 1:
            raw[raw_name] = _parse_keywords(value[0])
        # Special Case: Project-URL
        # The project urls is implemented in the metadata spec as a list of
        # specially-formatted strings that represent a key and a value, which
        # is fundamentally a mapping, however the email format doesn't support
        # mappings in a sane way, so it was crammed into a list of strings
        # instead.
        #
        # We will do a little light data massaging to turn this into a map as
        # it logically should be.
        elif raw_name == "project_urls":
            try:
                raw[raw_name] = _parse_project_urls(value)
            except KeyError:
                unparsed[name] = value
        # Nothing that we've done has managed to parse this, so it'll just
        # throw it in our unparseable data and move on.
        else:
            unparsed[name] = value

    # We need to support getting the Description from the message payload in
    # addition to getting it from the the headers. This does mean, though, there
    # is the possibility of it being set both ways, in which case we put both
    # in 'unparsed' since we don't know which is right.
    try:
        payload = _get_payload(parsed, data)
    except ValueError:
        unparsed.setdefault("description", []).append(
            parsed.get_payload(decode=isinstance(data, bytes))  # type: ignore[call-overload]
        )
    else:
        if payload:
            # Check to see if we've already got a description, if so then both
            # it, and this body move to unparseable.
            if "description" in raw:
                description_header = cast(str, raw.pop("description"))
                unparsed.setdefault("description", []).extend(
                    [description_header, payload]
                )
            elif "description" in unparsed:
                unparsed["description"].append(payload)
            else:
                raw["description"] = payload

    # We need to cast our `raw` to a metadata, because a TypedDict only support
    # literal key names, but we're computing our key names on purpose, but the
    # way this function is implemented, our `TypedDict` can only have valid key
    # names.
    return cast(RawMetadata, raw), unparsed


_NOT_FOUND = object()


# Keep the two values in sync.
_VALID_METADATA_VERSIONS = ["1.0", "1.1", "1.2", "2.1", "2.2", "2.3", "2.4"]
_MetadataVersion = Literal["1.0", "1.1", "1.2", "2.1", "2.2", "2.3", "2.4"]

_REQUIRED_ATTRS = frozenset(["metadata_version", "name", "version"])


class _Validator(Generic[T]):
    """Validate a metadata field.

    All _process_*() methods correspond to a core metadata field. The method is
    called with the field's raw value. If the raw value is valid it is returned
    in its "enriched" form (e.g. ``version.Version`` for the ``Version`` field).
    If the raw value is invalid, :exc:`InvalidMetadata` is raised (with a cause
    as appropriate).
    """

    name: str
    raw_name: str
    added: _MetadataVersion

    def __init__(
        self,
        *,
        added: _MetadataVersion = "1.0",
    ) -> None:
        self.added = added

    def __set_name__(self, _owner: Metadata, name: str) -> None:
        self.name = name
        self.raw_name = _RAW_TO_EMAIL_MAPPING[name]

    def __get__(self, instance: Metadata, _owner: type[Metadata]) -> T:
        # With Python 3.8, the caching can be replaced with functools.cached_property().
        # No need to check the cache as attribute lookup will resolve into the
        # instance's __dict__ before __get__ is called.
        cache = instance.__dict__
        value = instance._raw.get(self.name)

        # To make the _process_* methods easier, we'll check if the value is None
        # and if this field is NOT a required attribute, and if both of those
        # things are true, we'll skip the the converter. This will mean that the
        # converters never have to deal with the None union.
        if self.name in _REQUIRED_ATTRS or value is not None:
            try:
                converter: Callable[[Any], T] = getattr(self, f"_process_{self.name}")
            except AttributeError:
                pass
            else:
                value = converter(value)

        cache[self.name] = value
        try:
            del instance._raw[self.name]  # type: ignore[misc]
        except KeyError:
            pass

        return cast(T, value)

    def _invalid_metadata(
        self, msg: str, cause: Exception | None = None
    ) -> InvalidMetadata:
        exc = InvalidMetadata(
            self.raw_name, msg.format_map({"field": repr(self.raw_name)})
        )
        exc.__cause__ = cause
        return exc

    def _process_metadata_version(self, value: str) -> _MetadataVersion:
        # Implicitly makes Metadata-Version required.
        if value not in _VALID_METADATA_VERSIONS:
            raise self._invalid_metadata(f"{value!r} is not a valid metadata version")
        return cast(_MetadataVersion, value)

    def _process_name(self, value: str) -> str:
        if not value:
            raise self._invalid_metadata("{field} is a required field")
        # Validate the name as a side-effect.
        try:
            utils.canonicalize_name(value, validate=True)
        except utils.InvalidName as exc:
            raise self._invalid_metadata(
                f"{value!r} is invalid for {{field}}", cause=exc
            ) from exc
        else:
            return value

    def _process_version(self, value: str) -> version_module.Version:
        if not value:
            raise self._invalid_metadata("{field} is a required field")
        try:
            return version_module.parse(value)
        except version_module.InvalidVersion as exc:
            raise self._invalid_metadata(
                f"{value!r} is invalid for {{field}}", cause=exc
            ) from exc

    def _process_summary(self, value: str) -> str:
        """Check the field contains no newlines."""
        if "\n" in value:
            raise self._invalid_metadata("{field} must be a single line")
        return value

    def _process_description_content_type(self, value: str) -> str:
        content_types = {"text/plain", "text/x-rst", "text/markdown"}
        message = email.message.EmailMessage()
        message["content-type"] = value

        content_type, parameters = (
            # Defaults to `text/plain` if parsing failed.
            message.get_content_type().lower(),
            message["content-type"].params,
        )
        # Check if content-type is valid or defaulted to `text/plain` and thus was
        # not parseable.
        if content_type not in content_types or content_type not in value.lower():
            raise self._invalid_metadata(
                f"{{field}} must be one of {list(content_types)}, not {value!r}"
            )

        charset = parameters.get("charset", "UTF-8")
        if charset != "UTF-8":
            raise self._invalid_metadata(
                f"{{field}} can only specify the UTF-8 charset, not {list(charset)}"
            )

        markdown_variants = {"GFM", "CommonMark"}
        variant = parameters.get("variant", "GFM")  # Use an acceptable default.
        if content_type == "text/markdown" and variant not in markdown_variants:
            raise self._invalid_metadata(
                f"valid Markdown variants for {{field}} are {list(markdown_variants)}, "
                f"not {variant!r}",
            )
        return value

    def _process_dynamic(self, value: list[str]) -> list[str]:
        for dynamic_field in map(str.lower, value):
            if dynamic_field in {"name", "version", "metadata-version"}:
                raise self._invalid_metadata(
                    f"{dynamic_field!r} is not allowed as a dynamic field"
                )
            elif dynamic_field not in _EMAIL_TO_RAW_MAPPING:
                raise self._invalid_metadata(
                    f"{dynamic_field!r} is not a valid dynamic field"
                )
        return list(map(str.lower, value))

    def _process_provides_extra(
        self,
        value: list[str],
    ) -> list[utils.NormalizedName]:
        normalized_names = []
        try:
            for name in value:
                normalized_names.append(utils.canonicalize_name(name, validate=True))
        except utils.InvalidName as exc:
            raise self._invalid_metadata(
                f"{name!r} is invalid for {{field}}", cause=exc
            ) from exc
        else:
            return normalized_names

    def _process_requires_python(self, value: str) -> specifiers.SpecifierSet:
        try:
            return specifiers.SpecifierSet(value)
        except specifiers.InvalidSpecifier as exc:
            raise self._invalid_metadata(
                f"{value!r} is invalid for {{field}}", cause=exc
            ) from exc

    def _process_requires_dist(
        self,
        value: list[str],
    ) -> list[requirements.Requirement]:
        reqs = []
        try:
            for req in value:
                reqs.append(requirements.Requirement(req))
        except requirements.InvalidRequirement as exc:
            raise self._invalid_metadata(
                f"{req!r} is invalid for {{field}}", cause=exc
            ) from exc
        else:
            return reqs

    def _process_license_expression(
        self, value: str
    ) -> NormalizedLicenseExpression | None:
        try:
            return licenses.canonicalize_license_expression(value)
        except ValueError as exc:
            raise self._invalid_metadata(
                f"{value!r} is invalid for {{field}}", cause=exc
            ) from exc

    def _process_license_files(self, value: list[str]) -> list[str]:
        paths = []
        for path in value:
            if ".." in path:
                raise self._invalid_metadata(
                    f"{path!r} is invalid for {{field}}, "
                    "parent directory indicators are not allowed"
                )
            if "*" in path:
                raise self._invalid_metadata(
                    f"{path!r} is invalid for {{field}}, paths must be resolved"
                )
            if (
                pathlib.PurePosixPath(path).is_absolute()
                or pathlib.PureWindowsPath(path).is_absolute()
            ):
                raise self._invalid_metadata(
                    f"{path!r} is invalid for {{field}}, paths must be relative"
                )
            if pathlib.PureWindowsPath(path).as_posix() != path:
                raise self._invalid_metadata(
                    f"{path!r} is invalid for {{field}}, paths must use '/' delimiter"
                )
            paths.append(path)
        return paths


class Metadata:
    """Representation of distribution metadata.

    Compared to :class:`RawMetadata`, this class provides objects representing
    metadata fields instead of only using built-in types. Any invalid metadata
    will cause :exc:`InvalidMetadata` to be raised (with a
    :py:attr:`~BaseException.__cause__` attribute as appropriate).
    """

    _raw: RawMetadata

    @classmethod
    def from_raw(cls, data: RawMetadata, *, validate: bool = True) -> Metadata:
        """Create an instance from :class:`RawMetadata`.

        If *validate* is true, all metadata will be validated. All exceptions
        related to validation will be gathered and raised as an :class:`ExceptionGroup`.
        """
        ins = cls()
        ins._raw = data.copy()  # Mutations occur due to caching enriched values.

        if validate:
            exceptions: list[Exception] = []
            try:
                metadata_version = ins.metadata_version
                metadata_age = _VALID_METADATA_VERSIONS.index(metadata_version)
            except InvalidMetadata as metadata_version_exc:
                exceptions.append(metadata_version_exc)
                metadata_version = None

            # Make sure to check for the fields that are present, the required
            # fields (so their absence can be reported).
            fields_to_check = frozenset(ins._raw) | _REQUIRED_ATTRS
            # Remove fields that have already been checked.
            fields_to_check -= {"metadata_version"}

            for key in fields_to_check:
                try:
                    if metadata_version:
                        # Can't use getattr() as that triggers descriptor protocol which
                        # will fail due to no value for the instance argument.
                        try:
                            field_metadata_version = cls.__dict__[key].added
                        except KeyError:
                            exc = InvalidMetadata(key, f"unrecognized field: {key!r}")
                            exceptions.append(exc)
                            continue
                        field_age = _VALID_METADATA_VERSIONS.index(
                            field_metadata_version
                        )
                        if field_age > metadata_age:
                            field = _RAW_TO_EMAIL_MAPPING[key]
                            exc = InvalidMetadata(
                                field,
                                f"{field} introduced in metadata version "
                                f"{field_metadata_version}, not {metadata_version}",
                            )
                            exceptions.append(exc)
                            continue
                    getattr(ins, key)
                except InvalidMetadata as exc:
                    exceptions.append(exc)

            if exceptions:
                raise ExceptionGroup("invalid metadata", exceptions)

        return ins

    @classmethod
    def from_email(cls, data: bytes | str, *, validate: bool = True) -> Metadata:
        """Parse metadata from email headers.

        If *validate* is true, the metadata will be validated. All exceptions
        related to validation will be gathered and raised as an :class:`ExceptionGroup`.
        """
        raw, unparsed = parse_email(data)

        if validate:
            exceptions: list[Exception] = []
            for unparsed_key in unparsed:
                if unparsed_key in _EMAIL_TO_RAW_MAPPING:
                    message = f"{unparsed_key!r} has invalid data"
                else:
                    message = f"unrecognized field: {unparsed_key!r}"
                exceptions.append(InvalidMetadata(unparsed_key, message))

            if exceptions:
                raise ExceptionGroup("unparsed", exceptions)

        try:
            return cls.from_raw(raw, validate=validate)
        except ExceptionGroup as exc_group:
            raise ExceptionGroup(
                "invalid or unparsed metadata", exc_group.exceptions
            ) from None

    metadata_version: _Validator[_MetadataVersion] = _Validator()
    """:external:ref:`core-metadata-metadata-version`
    (required; validated to be a valid metadata version)"""
    # `name` is not normalized/typed to NormalizedName so as to provide access to
    # the original/raw name.
    name: _Validator[str] = _Validator()
    """:external:ref:`core-metadata-name`
    (required; validated using :func:`~packaging.utils.canonicalize_name` and its
    *validate* parameter)"""
    version: _Validator[version_module.Version] = _Validator()
    """:external:ref:`core-metadata-version` (required)"""
    dynamic: _Validator[list[str] | None] = _Validator(
        added="2.2",
    )
    """:external:ref:`core-metadata-dynamic`
    (validated against core metadata field names and lowercased)"""
    platforms: _Validator[list[str] | None] = _Validator()
    """:external:ref:`core-metadata-platform`"""
    supported_platforms: _Validator[list[str] | None] = _Validator(added="1.1")
    """:external:ref:`core-metadata-supported-platform`"""
    summary: _Validator[str | None] = _Validator()
    """:external:ref:`core-metadata-summary` (validated to contain no newlines)"""
    description: _Validator[str | None] = _Validator()  # TODO 2.1: can be in body
    """:external:ref:`core-metadata-description`"""
    description_content_type: _Validator[str | None] = _Validator(added="2.1")
    """:external:ref:`core-metadata-description-content-type` (validated)"""
    keywords: _Validator[list[str] | None] = _Validator()
    """:external:ref:`core-metadata-keywords`"""
    home_page: _Validator[str | None] = _Validator()
    """:external:ref:`core-metadata-home-page`"""
    download_url: _Validator[str | None] = _Validator(added="1.1")
    """:external:ref:`core-metadata-download-url`"""
    author: _Validator[str | None] = _Validator()
    """:external:ref:`core-metadata-author`"""
    author_email: _Validator[str | None] = _Validator()
    """:external:ref:`core-metadata-author-email`"""
    maintainer: _Validator[str | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-maintainer`"""
    maintainer_email: _Validator[str | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-maintainer-email`"""
    license: _Validator[str | None] = _Validator()
    """:external:ref:`core-metadata-license`"""
    license_expression: _Validator[NormalizedLicenseExpression | None] = _Validator(
        added="2.4"
    )
    """:external:ref:`core-metadata-license-expression`"""
    license_files: _Validator[list[str] | None] = _Validator(added="2.4")
    """:external:ref:`core-metadata-license-file`"""
    classifiers: _Validator[list[str] | None] = _Validator(added="1.1")
    """:external:ref:`core-metadata-classifier`"""
    requires_dist: _Validator[list[requirements.Requirement] | None] = _Validator(
        added="1.2"
    )
    """:external:ref:`core-metadata-requires-dist`"""
    requires_python: _Validator[specifiers.SpecifierSet | None] = _Validator(
        added="1.2"
    )
    """:external:ref:`core-metadata-requires-python`"""
    # Because `Requires-External` allows for non-PEP 440 version specifiers, we
    # don't do any processing on the values.
    requires_external: _Validator[list[str] | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-requires-external`"""
    project_urls: _Validator[dict[str, str] | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-project-url`"""
    # PEP 685 lets us raise an error if an extra doesn't pass `Name` validation
    # regardless of metadata version.
    provides_extra: _Validator[list[utils.NormalizedName] | None] = _Validator(
        added="2.1",
    )
    """:external:ref:`core-metadata-provides-extra`"""
    provides_dist: _Validator[list[str] | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-provides-dist`"""
    obsoletes_dist: _Validator[list[str] | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-obsoletes-dist`"""
    requires: _Validator[list[str] | None] = _Validator(added="1.1")
    """``Requires`` (deprecated)"""
    provides: _Validator[list[str] | None] = _Validator(added="1.1")
    """``Provides`` (deprecated)"""
    obsoletes: _Validator[list[str] | None] = _Validator(added="1.1")
    """``Obsoletes`` (deprecated)"""