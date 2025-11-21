
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_socket.py ===
from __future__ import annotations

import errno
import inspect
import os
import socket as stdlib_socket
import sys
import tempfile
from pathlib import Path
from socket import AddressFamily, SocketKind
from typing import TYPE_CHECKING, Union, cast

import attrs
import pytest

from .. import _core, socket as tsocket
from .._core._tests.tutil import binds_ipv6, can_create_ipv6, creates_ipv6, slow
from .._socket import _NUMERIC_ONLY, AddressFormat, SocketType, _SocketType, _try_sync
from ..testing import assert_checkpoints, wait_all_tasks_blocked

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import TypeAlias

    from .._highlevel_socket import SocketStream

    GaiTuple: TypeAlias = tuple[
        AddressFamily,
        SocketKind,
        int,
        str,
        Union[tuple[str, int], tuple[str, int, int, int], tuple[int, bytes]],
    ]
    GetAddrInfoResponse: TypeAlias = list[GaiTuple]
    GetAddrInfoArgs: TypeAlias = tuple[
        Union[str, bytes, None],
        Union[str, bytes, int, None],
        int,
        int,
        int,
        int,
    ]
else:
    GaiTuple: object
    GetAddrInfoResponse = object
    GetAddrInfoArgs = object

################################################################
# utils
################################################################


class MonkeypatchedGAI:
    __slots__ = ("_orig_getaddrinfo", "_responses", "record")

    def __init__(
        self,
        orig_getaddrinfo: Callable[
            [str | bytes | None, str | bytes | int | None, int, int, int, int],
            GetAddrInfoResponse,
        ],
    ) -> None:
        self._orig_getaddrinfo = orig_getaddrinfo
        self._responses: dict[
            GetAddrInfoArgs,
            GetAddrInfoResponse | str,
        ] = {}
        self.record: list[GetAddrInfoArgs] = []

    # get a normalized getaddrinfo argument tuple
    def _frozenbind(
        self,
        host: str | bytes | None,
        port: str | bytes | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> GetAddrInfoArgs:
        sig = inspect.signature(self._orig_getaddrinfo)
        bound = sig.bind(host, port, family=family, type=type, proto=proto, flags=flags)
        bound.apply_defaults()
        frozenbound = bound.args
        assert not bound.kwargs
        return frozenbound

    def set(
        self,
        response: GetAddrInfoResponse | str,
        host: str | bytes | None,
        port: str | bytes | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> None:
        self._responses[
            self._frozenbind(
                host,
                port,
                family=family,
                type=type,
                proto=proto,
                flags=flags,
            )
        ] = response

    def getaddrinfo(
        self,
        host: str | bytes | None,
        port: str | bytes | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> GetAddrInfoResponse | str:
        bound = self._frozenbind(host, port, family, type, proto, flags)
        self.record.append(bound)
        if bound in self._responses:
            return self._responses[bound]
        elif flags & stdlib_socket.AI_NUMERICHOST:
            return self._orig_getaddrinfo(host, port, family, type, proto, flags)
        else:
            raise RuntimeError(f"gai called with unexpected arguments {bound}")


@pytest.fixture
def monkeygai(monkeypatch: pytest.MonkeyPatch) -> MonkeypatchedGAI:
    controller = MonkeypatchedGAI(stdlib_socket.getaddrinfo)
    monkeypatch.setattr(stdlib_socket, "getaddrinfo", controller.getaddrinfo)
    return controller


async def test__try_sync() -> None:
    with assert_checkpoints():
        async with _try_sync():
            pass

    with assert_checkpoints():
        with pytest.raises(KeyError):
            async with _try_sync():
                raise KeyError

    async with _try_sync():
        raise BlockingIOError

    def _is_ValueError(exc: BaseException) -> bool:
        return isinstance(exc, ValueError)

    async with _try_sync(_is_ValueError):
        raise ValueError

    with assert_checkpoints():
        with pytest.raises(BlockingIOError):
            async with _try_sync(_is_ValueError):
                raise BlockingIOError


################################################################
# basic re-exports
################################################################


def test_socket_has_some_reexports() -> None:
    assert tsocket.SOL_SOCKET == stdlib_socket.SOL_SOCKET
    assert tsocket.TCP_NODELAY == stdlib_socket.TCP_NODELAY
    assert tsocket.gaierror == stdlib_socket.gaierror
    assert tsocket.ntohs == stdlib_socket.ntohs


################################################################
# name resolution
################################################################


async def test_getaddrinfo(monkeygai: MonkeypatchedGAI) -> None:
    def check(got: GetAddrInfoResponse, expected: GetAddrInfoResponse) -> None:
        # win32 returns 0 for the proto field
        # musl and glibc have inconsistent handling of the canonical name
        # field (https://github.com/python-trio/trio/issues/1499)
        # Neither field gets used much and there isn't much opportunity for us
        # to mess them up, so we don't bother checking them here
        def interesting_fields(
            gai_tup: GaiTuple,
        ) -> tuple[
            AddressFamily,
            SocketKind,
            tuple[str, int] | tuple[str, int, int, int] | tuple[int, bytes],
        ]:
            # (family, type, proto, canonname, sockaddr)
            family, type_, _proto, _canonname, sockaddr = gai_tup
            return (family, type_, sockaddr)

        def filtered(
            gai_list: GetAddrInfoResponse,
        ) -> list[
            tuple[
                AddressFamily,
                SocketKind,
                tuple[str, int] | tuple[str, int, int, int] | tuple[int, bytes],
            ]
        ]:
            return [interesting_fields(gai_tup) for gai_tup in gai_list]

        assert filtered(got) == filtered(expected)

    # Simple non-blocking non-error cases, ipv4 and ipv6:
    with assert_checkpoints():
        res = await tsocket.getaddrinfo("127.0.0.1", "12345", type=tsocket.SOCK_STREAM)

    check(
        res,
        [
            (
                tsocket.AF_INET,  # 127.0.0.1 is ipv4
                tsocket.SOCK_STREAM,
                tsocket.IPPROTO_TCP,
                "",
                ("127.0.0.1", 12345),
            ),
        ],
    )

    with assert_checkpoints():
        res = await tsocket.getaddrinfo("::1", "12345", type=tsocket.SOCK_DGRAM)
    check(
        res,
        [
            (
                tsocket.AF_INET6,
                tsocket.SOCK_DGRAM,
                tsocket.IPPROTO_UDP,
                "",
                ("::1", 12345, 0, 0),
            ),
        ],
    )

    monkeygai.set("x", b"host", "port", family=0, type=0, proto=0, flags=0)
    with assert_checkpoints():
        res = await tsocket.getaddrinfo("host", "port")
    assert res == "x"
    assert monkeygai.record[-1] == (b"host", "port", 0, 0, 0, 0)

    # check raising an error from a non-blocking getaddrinfo
    with assert_checkpoints():
        with pytest.raises(tsocket.gaierror) as excinfo:
            await tsocket.getaddrinfo("::1", "12345", type=-1)
    # Linux + glibc, Windows
    expected_errnos = {tsocket.EAI_SOCKTYPE}
    # Linux + musl
    expected_errnos.add(tsocket.EAI_SERVICE)
    # macOS
    if hasattr(tsocket, "EAI_BADHINTS"):
        expected_errnos.add(tsocket.EAI_BADHINTS)
    assert excinfo.value.errno in expected_errnos

    # check raising an error from a blocking getaddrinfo (exploits the fact
    # that monkeygai raises if it gets a non-numeric request it hasn't been
    # given an answer for)
    with assert_checkpoints():
        with pytest.raises(RuntimeError):
            await tsocket.getaddrinfo("asdf", "12345")


async def test_getnameinfo() -> None:
    # Trivial test:
    ni_numeric = stdlib_socket.NI_NUMERICHOST | stdlib_socket.NI_NUMERICSERV
    with assert_checkpoints():
        got = await tsocket.getnameinfo(("127.0.0.1", 1234), ni_numeric)
    assert got == ("127.0.0.1", "1234")

    # getnameinfo requires a numeric address as input:
    with assert_checkpoints():
        with pytest.raises(tsocket.gaierror):
            await tsocket.getnameinfo(("google.com", 80), 0)

    with assert_checkpoints():
        with pytest.raises(tsocket.gaierror):
            await tsocket.getnameinfo(("localhost", 80), 0)

    # Blocking call to get expected values:
    host, service = stdlib_socket.getnameinfo(("127.0.0.1", 80), 0)

    # Some working calls:
    got = await tsocket.getnameinfo(("127.0.0.1", 80), 0)
    assert got == (host, service)

    got = await tsocket.getnameinfo(("127.0.0.1", 80), tsocket.NI_NUMERICHOST)
    assert got == ("127.0.0.1", service)

    got = await tsocket.getnameinfo(("127.0.0.1", 80), tsocket.NI_NUMERICSERV)
    assert got == (host, "80")


################################################################
# constructors
################################################################


async def test_from_stdlib_socket() -> None:
    sa, sb = stdlib_socket.socketpair()
    assert not isinstance(sa, tsocket.SocketType)
    with sa, sb:
        ta = tsocket.from_stdlib_socket(sa)
        assert isinstance(ta, tsocket.SocketType)
        assert sa.fileno() == ta.fileno()
        await ta.send(b"x")
        assert sb.recv(1) == b"x"

    # rejects other types
    with pytest.raises(TypeError):
        tsocket.from_stdlib_socket(1)  # type: ignore[arg-type]

    class MySocket(stdlib_socket.socket):
        pass

    with MySocket() as mysock:
        with pytest.raises(TypeError):
            tsocket.from_stdlib_socket(mysock)


async def test_from_fd() -> None:
    sa, sb = stdlib_socket.socketpair()
    ta = tsocket.fromfd(sa.fileno(), sa.family, sa.type, sa.proto)
    with sa, sb, ta:
        assert ta.fileno() != sa.fileno()
        await ta.send(b"x")
        assert sb.recv(3) == b"x"


async def test_socketpair_simple() -> None:
    async def child(sock: SocketType) -> None:
        print("sending hello")
        await sock.send(b"h")
        assert await sock.recv(1) == b"h"

    a, b = tsocket.socketpair()
    with a, b:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(child, a)
            nursery.start_soon(child, b)


@pytest.mark.skipif(not hasattr(tsocket, "fromshare"), reason="windows only")
async def test_fromshare() -> None:
    if TYPE_CHECKING and sys.platform != "win32":  # pragma: no cover
        return
    a, b = tsocket.socketpair()
    with a, b:
        # share with ourselves
        shared = a.share(os.getpid())
        a2 = tsocket.fromshare(shared)
        with a2:
            assert a.fileno() != a2.fileno()
            await a2.send(b"x")
            assert await b.recv(1) == b"x"


async def test_socket() -> None:
    with tsocket.socket() as s:
        assert isinstance(s, tsocket.SocketType)
        assert s.family == tsocket.AF_INET


@creates_ipv6
async def test_socket_v6() -> None:
    with tsocket.socket(tsocket.AF_INET6, tsocket.SOCK_DGRAM) as s:
        assert isinstance(s, tsocket.SocketType)
        assert s.family == tsocket.AF_INET6


@pytest.mark.skipif(sys.platform != "linux", reason="linux only")
async def test_sniff_sockopts() -> None:
    from socket import AF_INET, AF_INET6, SOCK_DGRAM, SOCK_STREAM

    # generate the combinations of families/types we're testing:
    families = (AF_INET, AF_INET6) if can_create_ipv6 else (AF_INET,)
    sockets = [
        stdlib_socket.socket(family, type_)
        for family in families
        for type_ in [SOCK_DGRAM, SOCK_STREAM]
    ]
    for socket in sockets:
        # regular Trio socket constructor
        tsocket_socket = tsocket.socket(fileno=socket.fileno())
        # check family / type for correctness:
        assert tsocket_socket.family == socket.family
        assert tsocket_socket.type == socket.type
        tsocket_socket.detach()

        # fromfd constructor
        tsocket_from_fd = tsocket.fromfd(socket.fileno(), AF_INET, SOCK_STREAM)
        # check family / type for correctness:
        assert tsocket_from_fd.family == socket.family
        assert tsocket_from_fd.type == socket.type
        tsocket_from_fd.close()

        socket.close()


################################################################
# _SocketType
################################################################


async def test_SocketType_basics() -> None:
    sock = tsocket.socket()
    with sock as cm_enter_value:
        assert cm_enter_value is sock
        assert isinstance(sock.fileno(), int)
        assert not sock.get_inheritable()
        sock.set_inheritable(True)
        assert sock.get_inheritable()

        sock.setsockopt(tsocket.IPPROTO_TCP, tsocket.TCP_NODELAY, False)
        assert not sock.getsockopt(tsocket.IPPROTO_TCP, tsocket.TCP_NODELAY)
        sock.setsockopt(tsocket.IPPROTO_TCP, tsocket.TCP_NODELAY, True)
        assert sock.getsockopt(tsocket.IPPROTO_TCP, tsocket.TCP_NODELAY)
    # closed sockets have fileno() == -1
    assert sock.fileno() == -1

    # smoke test
    repr(sock)

    # detach
    with tsocket.socket() as sock:
        fd = sock.fileno()
        assert sock.detach() == fd
        assert sock.fileno() == -1

    # close
    sock = tsocket.socket()
    assert sock.fileno() >= 0
    sock.close()
    assert sock.fileno() == -1

    # share was tested above together with fromshare

    # check __dir__
    assert "family" in dir(sock)
    assert "recv" in dir(sock)
    assert "setsockopt" in dir(sock)

    # our __getattr__ handles unknown names
    with pytest.raises(AttributeError):
        sock.asdf  # type: ignore[attr-defined]  # noqa: B018

    # type family proto
    stdlib_sock = stdlib_socket.socket()
    sock = tsocket.from_stdlib_socket(stdlib_sock)
    assert sock.type == stdlib_sock.type
    assert sock.family == stdlib_sock.family
    assert sock.proto == stdlib_sock.proto
    sock.close()


async def test_SocketType_setsockopt() -> None:
    sock = tsocket.socket()
    with sock as _:
        setsockopt_tests(sock)


def setsockopt_tests(sock: SocketType | SocketStream) -> None:
    """Extract these out, to be reused for SocketStream also."""
    # specifying optlen. Not supported on pypy, and I couldn't find
    # valid calls on darwin or win32.
    if hasattr(tsocket, "SO_BINDTODEVICE"):
        try:
            sock.setsockopt(tsocket.SOL_SOCKET, tsocket.SO_BINDTODEVICE, None, 0)
        except OSError as e:
            assert e.errno in [  # noqa: PT017
                # some versions of Python have the attribute yet can run on
                # platforms that do not support it. For instance, MacOS 15
                # gained support for SO_BINDTODEVICE and CPython 3.13.1 was
                # built on it (presumably), but our CI runners ran MacOS 14 and
                # so failed.
                42,
                # Older Linux kernels (prior to patch
                # https://lore.kernel.org/netdev/m37drhs1jn.fsf@bernat.ch/t/)
                # do not support SO_BINDTODEVICE as an unprivileged user.
                errno.EPERM,
            ]

    # specifying value
    sock.setsockopt(tsocket.IPPROTO_TCP, tsocket.TCP_NODELAY, False)

    # specifying both
    with pytest.raises(TypeError, match="invalid value for argument 'value'"):
        sock.setsockopt(tsocket.IPPROTO_TCP, tsocket.TCP_NODELAY, False, 5)  # type: ignore[call-overload]

    # specifying neither
    with pytest.raises(TypeError, match="invalid value for argument 'value'"):
        sock.setsockopt(tsocket.IPPROTO_TCP, tsocket.TCP_NODELAY, None)  # type: ignore[call-overload]


async def test_SocketType_dup() -> None:
    a, b = tsocket.socketpair()
    with a, b:
        a2 = a.dup()
        with a2:
            assert isinstance(a2, tsocket.SocketType)
            assert a2.fileno() != a.fileno()
            a.close()
            await a2.send(b"x")
            assert await b.recv(1) == b"x"


async def test_SocketType_shutdown() -> None:
    a, b = tsocket.socketpair()
    with a, b:
        await a.send(b"x")
        assert await b.recv(1) == b"x"
        assert not a.did_shutdown_SHUT_WR
        assert not b.did_shutdown_SHUT_WR
        a.shutdown(tsocket.SHUT_WR)
        assert a.did_shutdown_SHUT_WR
        assert not b.did_shutdown_SHUT_WR
        assert await b.recv(1) == b""
        await b.send(b"y")
        assert await a.recv(1) == b"y"

    a, b = tsocket.socketpair()
    with a, b:
        assert not a.did_shutdown_SHUT_WR
        a.shutdown(tsocket.SHUT_RD)
        assert not a.did_shutdown_SHUT_WR

    a, b = tsocket.socketpair()
    with a, b:
        assert not a.did_shutdown_SHUT_WR
        a.shutdown(tsocket.SHUT_RDWR)
        assert a.did_shutdown_SHUT_WR


@pytest.mark.parametrize(
    ("address", "socket_type"),
    [
        ("127.0.0.1", tsocket.AF_INET),
        pytest.param("::1", tsocket.AF_INET6, marks=binds_ipv6),
    ],
)
async def test_SocketType_simple_server(
    address: str,
    socket_type: AddressFamily,
) -> None:
    # listen, bind, accept, connect, getpeername, getsockname
    listener = tsocket.socket(socket_type)
    client = tsocket.socket(socket_type)
    with listener, client:
        await listener.bind((address, 0))
        listener.listen(20)
        addr = listener.getsockname()[:2]
        async with _core.open_nursery() as nursery:
            nursery.start_soon(client.connect, addr)
            server, client_addr = await listener.accept()
        with server:
            assert client_addr == server.getpeername() == client.getsockname()
            await server.send(b"x")
            assert await client.recv(1) == b"x"


async def test_SocketType_is_readable() -> None:
    a, b = tsocket.socketpair()
    with a, b:
        assert not a.is_readable()
        await b.send(b"x")
        await _core.wait_readable(a)
        assert a.is_readable()
        assert await a.recv(1) == b"x"
        assert not a.is_readable()


# On some macOS systems, getaddrinfo likes to return V4-mapped addresses even
# when we *don't* pass AI_V4MAPPED.
# https://github.com/python-trio/trio/issues/580
def gai_without_v4mapped_is_buggy() -> bool:  # pragma: no cover
    try:
        stdlib_socket.getaddrinfo("1.2.3.4", 0, family=stdlib_socket.AF_INET6)
    except stdlib_socket.gaierror:
        return False
    else:
        return True


@attrs.define(slots=False)
class Addresses:
    bind_all: str
    localhost: str
    arbitrary: str
    broadcast: str


# Direct thorough tests of the implicit resolver helpers
@pytest.mark.parametrize(
    ("socket_type", "addrs"),
    [
        (
            tsocket.AF_INET,
            Addresses(
                bind_all="0.0.0.0",
                localhost="127.0.0.1",
                arbitrary="1.2.3.4",
                broadcast="255.255.255.255",
            ),
        ),
        pytest.param(
            tsocket.AF_INET6,
            Addresses(
                bind_all="::",
                localhost="::1",
                arbitrary="1::2",
                broadcast="::ffff:255.255.255.255",
            ),
            marks=creates_ipv6,
        ),
    ],
)
async def test_SocketType_resolve(socket_type: AddressFamily, addrs: Addresses) -> None:
    v6 = socket_type == tsocket.AF_INET6

    def pad(addr: tuple[str | int, ...]) -> tuple[str | int, ...]:
        if v6:
            while len(addr) < 4:
                addr += (0,)
        return addr

    def assert_eq(
        actual: tuple[str | int, ...],
        expected: tuple[str | int, ...],
    ) -> None:
        assert pad(expected) == pad(actual)

    with tsocket.socket(family=socket_type) as sock:
        # testing internal functionality, so we check it against the internal type
        assert isinstance(sock, _SocketType)

        # For some reason the stdlib special-cases "" to pass NULL to
        # getaddrinfo. They also error out on None, but whatever, None is much
        # more consistent, so we accept it too.
        # TODO: this implies that we can send host=None, but what does that imply for the return value, and other stuff?
        for null in [None, ""]:
            got = await sock._resolve_address_nocp((null, 80), local=True)
            assert not isinstance(got, (str, bytes))
            assert_eq(got, (addrs.bind_all, 80))
            got = await sock._resolve_address_nocp((null, 80), local=False)
            assert not isinstance(got, (str, bytes))
            assert_eq(got, (addrs.localhost, 80))

        # AI_PASSIVE only affects the wildcard address, so for everything else
        # local=True/local=False should work the same:
        for local in [False, True]:

            async def res(
                args: (
                    tuple[str, int]
                    | tuple[str, int, int]
                    | tuple[str, int, int, int]
                    | tuple[str, str]
                    | tuple[str, str, int]
                    | tuple[str, str, int, int]
                ),
            ) -> tuple[str | int, ...]:
                value = await sock._resolve_address_nocp(
                    args,
                    local=local,  # noqa: B023  # local is not bound in function definition
                )
                assert isinstance(value, tuple)
                return cast("tuple[Union[str, int], ...]", value)

            assert_eq(await res((addrs.arbitrary, "http")), (addrs.arbitrary, 80))
            if v6:
                # Check handling of different length ipv6 address tuples
                assert_eq(await res(("1::2", 80)), ("1::2", 80, 0, 0))
                assert_eq(await res(("1::2", 80, 0)), ("1::2", 80, 0, 0))
                assert_eq(await res(("1::2", 80, 0, 0)), ("1::2", 80, 0, 0))
                # Non-zero flowinfo/scopeid get passed through
                assert_eq(await res(("1::2", 80, 1)), ("1::2", 80, 1, 0))
                assert_eq(await res(("1::2", 80, 1, 2)), ("1::2", 80, 1, 2))

                # And again with a string port, as a trick to avoid the
                # already-resolved address fastpath and make sure we call
                # getaddrinfo
                assert_eq(await res(("1::2", "80")), ("1::2", 80, 0, 0))
                assert_eq(await res(("1::2", "80", 0)), ("1::2", 80, 0, 0))
                assert_eq(await res(("1::2", "80", 0, 0)), ("1::2", 80, 0, 0))
                assert_eq(await res(("1::2", "80", 1)), ("1::2", 80, 1, 0))
                assert_eq(await res(("1::2", "80", 1, 2)), ("1::2", 80, 1, 2))

                # V4 mapped addresses resolved if V6ONLY is False
                sock.setsockopt(tsocket.IPPROTO_IPV6, tsocket.IPV6_V6ONLY, False)
                assert_eq(await res(("1.2.3.4", "http")), ("::ffff:1.2.3.4", 80))

            # Check the <broadcast> special case, because why not
            assert_eq(await res(("<broadcast>", 123)), (addrs.broadcast, 123))

            # But not if it's true (at least on systems where getaddrinfo works
            # correctly)
            if v6 and not gai_without_v4mapped_is_buggy():
                sock.setsockopt(tsocket.IPPROTO_IPV6, tsocket.IPV6_V6ONLY, True)
                with pytest.raises(tsocket.gaierror) as excinfo:
                    await res(("1.2.3.4", 80))
                # Windows, macOS, musl/Linux
                expected_errnos = {tsocket.EAI_NONAME, tsocket.EAI_NODATA}
                # Linux
                if hasattr(tsocket, "EAI_ADDRFAMILY"):
                    expected_errnos.add(tsocket.EAI_ADDRFAMILY)
                assert excinfo.value.errno in expected_errnos

            # A family where we know nothing about the addresses, so should just
            # pass them through. This should work on Linux, which is enough to
            # smoke test the basic functionality...
            try:
                netlink_sock = tsocket.socket(
                    family=tsocket.AF_NETLINK,
                    type=tsocket.SOCK_DGRAM,
                )
            except (AttributeError, OSError):
                pass
            else:
                assert isinstance(netlink_sock, _SocketType)
                assert (
                    await netlink_sock._resolve_address_nocp("asdf", local=local)
                    == "asdf"
                )
                netlink_sock.close()

            address = r"^address should be a \(host, port(, \[flowinfo, \[scopeid\]\])*\) tuple$"
            with pytest.raises(ValueError, match=address):
                await res("1.2.3.4")  # type: ignore[arg-type]
            with pytest.raises(ValueError, match=address):
                await res(("1.2.3.4",))  # type: ignore[arg-type]
            with pytest.raises(
                ValueError,
                match=address,
            ):
                if v6:
                    await res(("1.2.3.4", 80, 0, 0, 0))  # type: ignore[arg-type]
                else:
                    # I guess in theory there could be enough overloads that this could error?
                    await res(("1.2.3.4", 80, 0, 0))


async def test_SocketType_unresolved_names() -> None:
    with tsocket.socket() as sock:
        await sock.bind(("localhost", 0))
        assert sock.getsockname()[0] == "127.0.0.1"
        sock.listen(10)

        with tsocket.socket() as sock2:
            await sock2.connect(("localhost", sock.getsockname()[1]))
            assert sock2.getpeername() == sock.getsockname()

    # check gaierror propagates out
    with tsocket.socket() as sock:
        with pytest.raises(tsocket.gaierror):
            # definitely not a valid request
            await sock.bind(("1.2:3", -1))


# This tests all the complicated paths through _nonblocking_helper, using recv
# as a stand-in for all the methods that use _nonblocking_helper.
async def test_SocketType_non_blocking_paths() -> None:
    a, b = stdlib_socket.socketpair()
    with a, b:
        ta = tsocket.from_stdlib_socket(a)
        b.setblocking(False)

        # cancel before even calling
        b.send(b"1")
        with _core.CancelScope() as cscope:
            cscope.cancel()
            with assert_checkpoints():
                with pytest.raises(_core.Cancelled):
                    await ta.recv(10)
        # immediate success (also checks that the previous attempt didn't
        # actually read anything)
        with assert_checkpoints():
            assert await ta.recv(10) == b"1"
        # immediate failure
        with assert_checkpoints():
            with pytest.raises(TypeError):
                await ta.recv("haha")  # type: ignore[arg-type]
        # block then succeed

        async def do_successful_blocking_recv() -> None:
            with assert_checkpoints():
                assert await ta.recv(10) == b"2"

        async with _core.open_nursery() as nursery:
            nursery.start_soon(do_successful_blocking_recv)
            await wait_all_tasks_blocked()
            b.send(b"2")
        # block then cancelled

        async def do_cancelled_blocking_recv() -> None:
            with assert_checkpoints():
                with pytest.raises(_core.Cancelled):
                    await ta.recv(10)

        async with _core.open_nursery() as nursery:
            nursery.start_soon(do_cancelled_blocking_recv)
            await wait_all_tasks_blocked()
            nursery.cancel_scope.cancel()
        # Okay, here's the trickiest one: we want to exercise the path where
        # the task is signaled to wake, goes to recv, but then the recv fails,
        # so it has to go back to sleep and try again. Strategy: have two
        # tasks waiting on two sockets (to work around the rule against having
        # two tasks waiting on the same socket), wake them both up at the same
        # time, and whichever one runs first "steals" the data from the
        # other:
        tb = tsocket.from_stdlib_socket(b)

        async def t1() -> None:
            with assert_checkpoints():
                assert await ta.recv(1) == b"a"
            with assert_checkpoints():
                assert await tb.recv(1) == b"b"

        async def t2() -> None:
            with assert_checkpoints():
                assert await tb.recv(1) == b"b"
            with assert_checkpoints():
                assert await ta.recv(1) == b"a"

        async with _core.open_nursery() as nursery:
            nursery.start_soon(t1)
            nursery.start_soon(t2)
            await wait_all_tasks_blocked()
            a.send(b"b")
            b.send(b"a")
            await wait_all_tasks_blocked()
            a.send(b"b")
            b.send(b"a")


# This tests the complicated paths through connect
@slow
async def test_SocketType_connect_paths() -> None:
    with tsocket.socket() as sock:
        with pytest.raises(
            ValueError,
            match=r"^address should be a \(host, port(, \[flowinfo, \[scopeid\]\])*\) tuple$",
        ):
            # Should be a tuple
            await sock.connect("localhost")

    # cancelled before we start
    with tsocket.socket() as sock:
        with _core.CancelScope() as cancel_scope:
            cancel_scope.cancel()
            with pytest.raises(_core.Cancelled):
                await sock.connect(("127.0.0.1", 80))

    # Cancelled in between the connect() call and the connect completing
    with _core.CancelScope() as cancel_scope:
        with tsocket.socket() as sock, tsocket.socket() as listener:
            await listener.bind(("127.0.0.1", 0))
            listener.listen()

            # Swap in our weird subclass under the trio.socket._SocketType's
            # nose -- and then swap it back out again before we hit
            # wait_socket_writable, which insists on a real socket.
            class CancelSocket(stdlib_socket.socket):
                def connect(
                    self,
                    address: AddressFormat,
                ) -> None:
                    # accessing private method only available in _SocketType
                    assert isinstance(sock, _SocketType)

                    cancel_scope.cancel()
                    sock._sock = stdlib_socket.fromfd(
                        self.detach(),
                        self.family,
                        self.type,
                    )
                    sock._sock.connect(address)
                    # If connect *doesn't* raise, then pretend it did
                    raise BlockingIOError  # pragma: no cover

            # accessing private method only available in _SocketType
            assert isinstance(sock, _SocketType)
            sock._sock.close()
            sock._sock = CancelSocket()

            with assert_checkpoints():
                with pytest.raises(_core.Cancelled):
                    await sock.connect(listener.getsockname())
            assert sock.fileno() == -1

    # Failed connect (hopefully after raising BlockingIOError)
    with tsocket.socket() as sock:
        with pytest.raises(
            OSError,
            match=r"^\[\w+ \d+\] Error connecting to \('127\.0\.0\.\d', \d+\): (Connection refused|Unknown error)$",
        ):
            # TCP port 2 is not assigned. Pretty sure nothing will be
            # listening there. (We used to bind a port and then *not* call
            # listen() to ensure nothing was listening there, but it turns
            # out on macOS if you do this it takes 30 seconds for the
            # connect to fail. Really. Also if you use a non-routable
            # address. This way fails instantly though. As long as nothing
            # is listening on port 2.)

            # Windows retries failed connections so this takes seconds
            # (and that's why this is marked @slow)
            await sock.connect(("127.0.0.1", 2))


# Fix issue #1810
@slow
async def test_address_in_socket_error() -> None:
    address = "127.0.0.1"
    with tsocket.socket() as sock:
        with pytest.raises(
            OSError,
            match=rf"^\[\w+ \d+\] Error connecting to \({address!r}, 2\): (Connection refused|Unknown error)$",
        ):
            # Windows retries failed connections so this takes seconds
            # (and that's why this is marked @slow)
            await sock.connect((address, 2))


async def test_resolve_address_exception_in_connect_closes_socket() -> None:
    # Here we are testing issue 247, any cancellation will leave the socket closed
    with _core.CancelScope() as cancel_scope:
        with tsocket.socket() as sock:

            async def _resolve_address_nocp(
                address: AddressFormat,
                *,
                local: bool,
            ) -> None:
                assert address == ""
                assert not local
                cancel_scope.cancel()
                await _core.checkpoint()

            assert isinstance(sock, _SocketType)
            sock._resolve_address_nocp = _resolve_address_nocp  # type: ignore[method-assign]
            with assert_checkpoints():
                with pytest.raises(_core.Cancelled):
                    await sock.connect("")
            assert sock.fileno() == -1


async def test_send_recv_variants() -> None:
    a, b = tsocket.socketpair()
    with a, b:
        # recv, including with flags
        assert await a.send(b"x") == 1
        assert await b.recv(10, tsocket.MSG_PEEK) == b"x"
        assert await b.recv(10) == b"x"

        # recv_into
        await a.send(b"x")
        buf = bytearray(10)
        await b.recv_into(buf)
        assert buf == b"x" + b"\x00" * 9

        if hasattr(a, "sendmsg"):
            assert await a.sendmsg([b"xxx"], []) == 3
            assert await b.recv(10) == b"xxx"

    a = tsocket.socket(type=tsocket.SOCK_DGRAM)
    b = tsocket.socket(type=tsocket.SOCK_DGRAM)
    with a, b:
        await a.bind(("127.0.0.1", 0))
        await b.bind(("127.0.0.1", 0))

        targets = [b.getsockname(), ("localhost", b.getsockname()[1])]

        # recvfrom + sendto, with and without names
        for target in targets:
            assert await a.sendto(b"xxx", target) == 3
            (data, addr) = await b.recvfrom(10)
            assert data == b"xxx"
            assert addr == a.getsockname()

        # sendto + flags
        #
        # I can't find any flags that send() accepts... on Linux at least
        # passing MSG_MORE to send_some on a connected UDP socket seems to
        # just be ignored.
        #
        # But there's no MSG_MORE on Windows or macOS. I guess send_some flags
        # are really not very useful, but at least this tests them a bit.
        if hasattr(tsocket, "MSG_MORE"):
            await a.sendto(b"xxx", tsocket.MSG_MORE, b.getsockname())
            await a.sendto(b"yyy", tsocket.MSG_MORE, b.getsockname())
            await a.sendto(b"zzz", b.getsockname())
            (data, addr) = await b.recvfrom(10)
            assert data == b"xxxyyyzzz"
            assert addr == a.getsockname()

        # recvfrom_into
        assert await a.sendto(b"xxx", b.getsockname()) == 3
        buf = bytearray(10)
        (nbytes, addr) = await b.recvfrom_into(buf)
        assert nbytes == 3
        assert buf == b"xxx" + b"\x00" * 7
        assert addr == a.getsockname()

        if hasattr(b, "recvmsg"):
            assert await a.sendto(b"xxx", b.getsockname()) == 3
            (data, ancdata, msg_flags, addr) = await b.recvmsg(10)
            assert data == b"xxx"
            assert ancdata == []
            assert msg_flags == 0
            assert addr == a.getsockname()

        if hasattr(b, "recvmsg_into"):
            assert await a.sendto(b"xyzw", b.getsockname()) == 4
            buf1 = bytearray(2)
            buf2 = bytearray(3)
            ret = await b.recvmsg_into([buf1, buf2])
            (nbytes, ancdata, msg_flags, addr) = ret
            assert nbytes == 4
            assert buf1 == b"xy"
            assert buf2 == b"zw" + b"\x00"
            assert ancdata == []
            assert msg_flags == 0
            assert addr == a.getsockname()

        if hasattr(a, "sendmsg"):
            for target in targets:
                assert await a.sendmsg([b"x", b"yz"], [], 0, target) == 3
                assert await b.recvfrom(10) == (b"xyz", a.getsockname())

    a = tsocket.socket(type=tsocket.SOCK_DGRAM)
    b = tsocket.socket(type=tsocket.SOCK_DGRAM)
    with a, b:
        await b.bind(("127.0.0.1", 0))
        await a.connect(b.getsockname())
        # send on a connected udp socket; each call creates a separate
        # datagram
        await a.send(b"xxx")
        await a.send(b"yyy")
        assert await b.recv(10) == b"xxx"
        assert await b.recv(10) == b"yyy"


async def test_idna(monkeygai: MonkeypatchedGAI) -> None:
    # This is the encoding for "faß.de", which uses one of the characters that
    # IDNA 2003 handles incorrectly:
    monkeygai.set("ok faß.de", b"xn--fa-hia.de", 80)
    monkeygai.set("ok ::1", "::1", 80, flags=_NUMERIC_ONLY)
    monkeygai.set("ok ::1", b"::1", 80, flags=_NUMERIC_ONLY)
    # Some things that should not reach the underlying socket.getaddrinfo:
    monkeygai.set("bad", "fass.de", 80)
    # We always call socket.getaddrinfo with bytes objects:
    monkeygai.set("bad", "xn--fa-hia.de", 80)

    assert await tsocket.getaddrinfo("::1", 80) == "ok ::1"
    assert await tsocket.getaddrinfo(b"::1", 80) == "ok ::1"
    assert await tsocket.getaddrinfo("faß.de", 80) == "ok faß.de"
    assert await tsocket.getaddrinfo("xn--fa-hia.de", 80) == "ok faß.de"
    assert await tsocket.getaddrinfo(b"xn--fa-hia.de", 80) == "ok faß.de"


async def test_getprotobyname() -> None:
    # These are the constants used in IP header fields, so the numeric values
    # had *better* be stable across systems...
    assert await tsocket.getprotobyname("udp") == 17
    assert await tsocket.getprotobyname("tcp") == 6


async def test_custom_hostname_resolver(monkeygai: MonkeypatchedGAI) -> None:
    # This intentionally breaks the signatures used in HostnameResolver
    class CustomResolver:
        async def getaddrinfo(
            self,
            host: str,
            port: str,
            family: int,
            type: int,
            proto: int,
            flags: int,
        ) -> tuple[str, str, str, int, int, int, int]:
            return ("custom_gai", host, port, family, type, proto, flags)

        async def getnameinfo(
            self,
            sockaddr: tuple[str, int] | tuple[str, int, int, int],
            flags: int,
        ) -> tuple[str, tuple[str, int] | tuple[str, int, int, int], int]:
            return ("custom_gni", sockaddr, flags)

    cr = CustomResolver()

    assert tsocket.set_custom_hostname_resolver(cr) is None  # type: ignore[arg-type]

    # Check that the arguments are all getting passed through.
    # We have to use valid calls to avoid making the underlying system
    # getaddrinfo cranky when it's used for NUMERIC checks.
    for vals in [
        (tsocket.AF_INET, 0, 0, 0),
        (0, tsocket.SOCK_STREAM, 0, 0),
        (0, 0, tsocket.IPPROTO_TCP, 0),
        (0, 0, 0, tsocket.AI_CANONNAME),
    ]:
        assert await tsocket.getaddrinfo("localhost", "foo", *vals) == (
            "custom_gai",
            b"localhost",
            "foo",
            *vals,
        )

    # IDNA encoding is handled before calling the special object
    got = await tsocket.getaddrinfo("föö", "foo")
    expected = ("custom_gai", b"xn--f-1gaa", "foo", 0, 0, 0, 0)
    assert got == expected

    assert await tsocket.getnameinfo("a", 0) == (  # type: ignore[arg-type]
        "custom_gni",
        "a",
        0,
    )

    # We can set it back to None
    assert tsocket.set_custom_hostname_resolver(None) is cr

    # And now Trio switches back to calling socket.getaddrinfo (specifically
    # our monkeypatched version of socket.getaddrinfo)
    monkeygai.set("x", b"host", "port", family=0, type=0, proto=0, flags=0)
    assert await tsocket.getaddrinfo("host", "port") == "x"


async def test_custom_socket_factory() -> None:
    class CustomSocketFactory:
        def socket(
            self,
            family: AddressFamily,
            type: SocketKind,
            proto: int,
        ) -> tuple[str, AddressFamily, SocketKind, int]:
            return ("hi", family, type, proto)

    csf = CustomSocketFactory()

    assert tsocket.set_custom_socket_factory(csf) is None  # type: ignore[arg-type]

    assert tsocket.socket() == ("hi", tsocket.AF_INET, tsocket.SOCK_STREAM, 0)
    assert tsocket.socket(1, 2, 3) == ("hi", 1, 2, 3)

    # socket with fileno= doesn't call our custom method
    fd = stdlib_socket.socket().detach()
    wrapped = tsocket.socket(fileno=fd)
    assert hasattr(wrapped, "bind")
    wrapped.close()

    # Likewise for socketpair
    a, b = tsocket.socketpair()
    with a, b:
        assert hasattr(a, "bind")
        assert hasattr(b, "bind")

    assert tsocket.set_custom_socket_factory(None) is csf


def test_SocketType_is_abstract() -> None:
    with pytest.raises(TypeError):
        tsocket.SocketType()


@pytest.mark.skipif(not hasattr(tsocket, "AF_UNIX"), reason="no unix domain sockets")
async def test_unix_domain_socket() -> None:
    # Bind has a special branch to use a thread, since it has to do filesystem
    # traversal. Maybe connect should too? Not sure.

    async def check_AF_UNIX(path: str | bytes | os.PathLike[str]) -> None:
        with tsocket.socket(family=tsocket.AF_UNIX) as lsock:
            await lsock.bind(path)
            lsock.listen(10)
            with tsocket.socket(family=tsocket.AF_UNIX) as csock:
                await csock.connect(path)
                ssock, _ = await lsock.accept()
                with ssock:
                    await csock.send(b"x")
                    assert await ssock.recv(1) == b"x"

    # Can't use tmpdir fixture, because we can exceed the maximum AF_UNIX path
    # length on macOS.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test passing various supported types as path
        # Must use different filenames to prevent "address already in use"
        await check_AF_UNIX(f"{tmpdir}/sock")
        await check_AF_UNIX(Path(f"{tmpdir}/sock1"))
        await check_AF_UNIX(os.fsencode(f"{tmpdir}/sock2"))

    try:
        cookie = os.urandom(20).hex().encode("ascii")
        await check_AF_UNIX(b"\x00trio-test-" + cookie)
    except FileNotFoundError:
        # macOS doesn't support abstract filenames with the leading NUL byte
        pass


async def test_interrupted_by_close() -> None:
    a_stdlib, b_stdlib = stdlib_socket.socketpair()
    with a_stdlib, b_stdlib:
        a_stdlib.setblocking(False)

        data = b"x" * 99999

        try:
            while True:
                a_stdlib.send(data)
        except BlockingIOError:
            pass

        a = tsocket.from_stdlib_socket(a_stdlib)

        async def sender() -> None:
            with pytest.raises(_core.ClosedResourceError):
                await a.send(data)

        async def receiver() -> None:
            with pytest.raises(_core.ClosedResourceError):
                await a.recv(1)

        async with _core.open_nursery() as nursery:
            nursery.start_soon(sender)
            nursery.start_soon(receiver)
            await wait_all_tasks_blocked()
            a.close()


async def test_many_sockets() -> None:
    total = 1000  # Must be more than MAX_AFD_GROUP_SIZE
    sockets = []
    # Open at most <total> socket pairs
    for opened in range(0, total, 2):
        try:
            a, b = stdlib_socket.socketpair()
        except OSError as exc:  # pragma: no cover
            # Semi-expecting following errors (sockets are files):
            # EMFILE: "Too many open files" (reached kernel cap)
            # ENFILE: "File table overflow" (beyond kernel cap)
            assert exc.errno in (errno.EMFILE, errno.ENFILE)  # noqa: PT017
            print(f"Unable to open more than {opened} sockets.")
            # Stop opening any more sockets if too many are open
            break
        sockets += [a, b]
    async with _core.open_nursery() as nursery:
        for socket in sockets:
            nursery.start_soon(_core.wait_readable, socket)
        await _core.wait_all_tasks_blocked()
        nursery.cancel_scope.cancel()
    for socket in sockets:
        socket.close()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\calibrate.py ===
#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import abc
import copy
import itertools
import os
import uuid
from collections.abc import Sequence
from enum import Enum
from pathlib import Path

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, helper, numpy_helper

import onnxruntime

from .quant_utils import apply_plot, load_model_with_shape_infer, smooth_distribution


def rel_entr(pk: np.ndarray, qk: np.ndarray) -> np.ndarray:
    """
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html#scipy.special.rel_entr.
    Python implementation.
    """
    res = np.empty(pk.shape, dtype=pk.dtype)
    res[:] = pk[:] * np.log(pk[:] / qk[:])
    c2 = (pk == 0) & (qk >= 0)
    res[c2] = 0
    c1 = (pk > 0) & (qk > 0)
    res[~c1] = np.inf
    return res


def entropy(
    pk: np.ndarray,
    qk: np.ndarray,
    base: float | None = None,
    axis: int = 0,
) -> np.ndarray:
    """
    Simplifeied version of entropy.
    Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html.
    This avoids taking a dependency on scipy just for this function.
    """
    assert base is None or base > 0, "base={base} must be a positive number or `None`."
    assert qk is not None, "qk is None"

    pk = np.asarray(pk).astype(np.float32)
    pk = 1.0 * pk / np.sum(pk, axis=axis, keepdims=True)

    qk = np.asarray(qk).astype(np.float32)
    pk, qk = np.broadcast_arrays(pk, qk)
    qk = 1.0 * qk / np.sum(qk, axis=axis, keepdims=True)
    vec = rel_entr(pk, qk)

    s = np.sum(vec, axis=axis)
    if base is not None:
        s /= np.log(base)
    return s.astype(pk.dtype)


class TensorData:
    _allowed = frozenset(["avg", "std", "lowest", "highest", "hist", "hist_edges", "bins"])
    _floats = frozenset(["avg", "std", "lowest", "highest", "hist_edges"])

    def __init__(self, **kwargs):
        self._attrs = list(kwargs.keys())
        for k, v in kwargs.items():
            if k not in TensorData._allowed:
                raise ValueError(f"Unexpected value {k!r} not in {TensorData._allowed}.")
            if k in TensorData._floats:
                if not hasattr(v, "dtype"):
                    raise ValueError(f"Unexpected type {type(v)} for k={k!r}")
                if v.dtype not in (np.float16, np.float32):
                    raise ValueError(f"Unexpected dtype {v.dtype} for k={k!r}")
            setattr(self, k, v)

    @property
    def range_value(self):
        if not hasattr(self, "lowest") or not hasattr(self, "highest"):
            raise AttributeError(f"Attributes 'lowest' and/or 'highest' missing in {dir(self)}.")
        return (self.lowest, self.highest)

    @property
    def avg_std(self):
        if not hasattr(self, "avg") or not hasattr(self, "std"):
            raise AttributeError(f"Attributes 'avg' and/or 'std' missing in {dir(self)}.")
        return (self.avg, self.std)

    def to_dict(self):
        # This is needed to serialize the data into JSON.
        data = {k: getattr(self, k) for k in self._attrs}
        data["CLS"] = self.__class__.__name__
        return data


class TensorsData:
    def __init__(self, calibration_method, data: dict[str, TensorData | tuple]):
        self.calibration_method = calibration_method
        self.data = {}
        for k, v in data.items():
            if not isinstance(k, str):
                raise TypeError(f"Keys must be strings not {type(k)}.")
            if isinstance(v, tuple):
                if calibration_method == CalibrationMethod.MinMax and len(v) == 2:
                    self.data[k] = TensorData(lowest=v[0], highest=v[1])
                    continue
                if len(v) == 4:
                    self.data[k] = TensorData(lowest=v[0], highest=v[1], hist=v[2], bins=v[3])
                    continue
                raise TypeError(f"Unexpected tuple for {k:r}, it has {len(v)} elements: {v}.")
            if not isinstance(v, TensorData):
                raise TypeError(f"Values must be TensorData not {type(v)}.")
            self.data[k] = v

    def __iter__(self):
        yield from self.data

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        if key not in self.data:
            raise RuntimeError(f"Only an existing tensor can be modified, {key!r} is not.")
        self.data[key] = value

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def to_dict(self):
        # This is needed to serialize the data into JSON.
        data = {
            "CLS": self.__class__.__name__,
            "data": self.data,
            "calibration_method": self.calibration_method,
        }
        return data


class CalibrationMethod(Enum):
    MinMax = 0
    Entropy = 1
    Percentile = 2
    Distribution = 3


class CalibrationDataReader(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, "get_next") and callable(subclass.get_next)) or NotImplemented

    @abc.abstractmethod
    def get_next(self) -> dict:
        """generate the input data dict for ONNXinferenceSession run"""
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        result = self.get_next()
        if result is None:
            raise StopIteration
        return result

    def __len__(self):
        raise NotImplementedError

    def set_range(self, start_index: int, end_index: int):
        raise NotImplementedError


class CalibraterBase:
    def __init__(
        self,
        model_path: str | Path,
        op_types_to_calibrate: Sequence[str] | None = None,
        augmented_model_path="augmented_model.onnx",
        symmetric=False,
        use_external_data_format=False,
        per_channel=False,
    ):
        """
        :param model_path: ONNX model to calibrate. It should be a model file path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        :param symmetric: make range of tensor symmetric (central point is 0).
        :param use_external_data_format: use external data format to store model which size is >= 2Gb.
        :param per_channel: whether to compute ranges per each channel.
        """
        if isinstance(model_path, str):
            self.model = load_model_with_shape_infer(Path(model_path))
        elif isinstance(model_path, Path):
            self.model = load_model_with_shape_infer(model_path)
        else:
            raise ValueError("model_path should be model path.")

        self.op_types_to_calibrate = op_types_to_calibrate
        self.augmented_model_path = augmented_model_path
        self.symmetric = symmetric
        self.use_external_data_format = use_external_data_format
        self.per_channel = per_channel

        self.augment_model = None
        self.infer_session = None
        self.execution_providers = ["CPUExecutionProvider"]

    def set_execution_providers(self, execution_providers=["CPUExecutionProvider"]):  # noqa: B006
        """
        reset the execution providers to execute the collect_data. It triggers to re-creating inference session.
        """
        self.execution_providers = execution_providers
        self.create_inference_session()

    def create_inference_session(self):
        """
        create an OnnxRuntime InferenceSession.
        """
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.infer_session = onnxruntime.InferenceSession(
            self.augmented_model_path,
            sess_options=sess_options,
            providers=self.execution_providers,
        )

    def select_tensors_to_calibrate(self, model: ModelProto):
        """
        select input/output tensors of candidate nodes to calibrate.
        returns:
            tensors (set): set of tensor name.
            value_infos (dict): tensor name to value info.
        """
        value_infos = {vi.name: vi for vi in model.graph.value_info}
        value_infos.update({ot.name: ot for ot in model.graph.output})
        value_infos.update({it.name: it for it in model.graph.input})
        initializer = {init.name for init in model.graph.initializer}

        tensors_to_calibrate = set()
        tensor_type_to_calibrate = {TensorProto.FLOAT, TensorProto.FLOAT16}

        for node in model.graph.node:
            if not self.op_types_to_calibrate or node.op_type in self.op_types_to_calibrate:
                for tensor_name in itertools.chain(node.input, node.output):
                    if tensor_name in value_infos:
                        vi = value_infos[tensor_name]
                        if (
                            vi.type.HasField("tensor_type")
                            and (vi.type.tensor_type.elem_type in tensor_type_to_calibrate)
                            and (tensor_name not in initializer)
                        ):
                            tensors_to_calibrate.add(tensor_name)

        return tensors_to_calibrate, value_infos

    def get_augment_model(self):
        """
        return: augmented onnx model. Call after calling augment_graph
        """
        return self.model

    def augment_graph(self):
        """
        abstract method: augment the input model to prepare for collecting data. It will:
            1. augment the model to be able to collect desired statistics data
            2. save augmented model to augmented_model_paths
        """
        raise NotImplementedError

    def collect_data(self, data_reader: CalibrationDataReader):
        """
        abstract method: collect the tensors that will be used for range computation. It can be called multiple times.
        """
        raise NotImplementedError

    def compute_data(self) -> TensorsData:
        """
        abstract method: compute data based on the calibration method stored in TensorsData
        """
        raise NotImplementedError


class MinMaxCalibrater(CalibraterBase):
    def __init__(
        self,
        model_path: str | Path,
        op_types_to_calibrate: Sequence[str] | None = None,
        augmented_model_path="augmented_model.onnx",
        symmetric=False,
        use_external_data_format=False,
        moving_average=False,
        averaging_constant=0.01,
        max_intermediate_outputs=None,
        per_channel=False,
    ):
        """
        :param model_path: ONNX model to calibrate. It is a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        :param symmetric: make range of tensor symmetric (central point is 0).
        :param use_external_data_format: use external data format to store model which size is >= 2Gb
        :param moving_average: compute the moving average of the minimum and maximum values instead of the global minimum and maximum.
        :param averaging_constant: constant smoothing factor to use when computing the moving average.
        :param max_intermediate_outputs: maximum number of intermediate outputs before an intermediate range is computed.
        :param per_channel: whether to compute ranges per each channel.
        """
        super().__init__(
            model_path,
            op_types_to_calibrate=op_types_to_calibrate,
            augmented_model_path=augmented_model_path,
            symmetric=symmetric,
            use_external_data_format=use_external_data_format,
            per_channel=per_channel,
        )
        self.intermediate_outputs = []
        self.calibrate_tensors_range = None
        self.num_model_outputs = len(self.model.graph.output)
        self.model_original_outputs = {output.name for output in self.model.graph.output}
        self.moving_average = moving_average
        if moving_average and (averaging_constant < 0 or averaging_constant > 1):
            raise ValueError("Invalid averaging constant, which should not be < 0 or > 1.")
        self.averaging_constant = averaging_constant
        self.max_intermediate_outputs = max_intermediate_outputs

    def augment_graph(self):
        """
        Adds ReduceMin and ReduceMax nodes to all quantization_candidates op type nodes in
        model and ensures their outputs are stored as part of the graph output
        :return: augmented ONNX model
        """
        tensors, _ = self.select_tensors_to_calibrate(self.model)
        reshape_shape_name = str(uuid.uuid4())
        reshape_shape = numpy_helper.from_array(np.array([-1], dtype=np.int64), reshape_shape_name)
        self.model.graph.initializer.append(reshape_shape)

        def get_op_version(op_type, model):
            for opset_import in model.opset_import:
                if onnx.defs.has(op_type, opset_import.domain):
                    return opset_import.version
            raise RuntimeError(f"Model does not contain a version for '{op_type}'.")

        def add_reduce_min_max(tensor_name, reduce_op_name):
            # When doing ReduceMax/ReduceMin, ORT can't reduce on dim with value of 0 if 'keepdims' is false.
            # To make the code simple, we always let keepdims to be 1.
            keepdims = 1

            # Adding ReduceMin/ReduceMax nodes: ReduceMin/ReduceMax -> Reshape-> (output)
            reduce_output = tensor_name + "_" + reduce_op_name
            intermediate_output = reduce_output + "_Reshape"
            reduce_node = onnx.helper.make_node(
                reduce_op_name, [tensor_name], [intermediate_output], keepdims=keepdims, name=reduce_output
            )

            reshape_node = onnx.helper.make_node(
                "Reshape",
                inputs=[intermediate_output, reshape_shape_name],
                outputs=[reduce_output],
                name=intermediate_output,
            )

            value_infos = {vi.name: vi for vi in self.model.graph.value_info}
            value_infos.update({o.name: o for o in self.model.graph.output})
            value_infos.update({i.name: i for i in self.model.graph.input})
            if tensor_name in value_infos:
                onnx_type = value_infos[tensor_name].type.tensor_type.elem_type
            else:
                raise ValueError(
                    f"Unable to guess tensor type for tensor {tensor_name!r}, "
                    "running shape inference before quantization may resolve this issue."
                )

            # Include axes in reduce_op when per_channel, always keeping axis=1
            if self.per_channel:
                tensor_rank = len(value_infos[tensor_name].type.tensor_type.shape.dim)
                reduced_axes = [0, *range(2, tensor_rank)]
                # Depending on opset version, axes in ReduceMin/ReduceMax are in attribute or inputs
                if get_op_version(reduce_op_name, self.model) < 18:
                    reduce_node.attribute.append(helper.make_attribute("axes", reduced_axes))
                else:
                    reduce_axes_name = str(uuid.uuid4())
                    reduce_axes = numpy_helper.from_array(np.array(reduced_axes, dtype=np.int64), reduce_axes_name)
                    reduce_node.input.append(reduce_axes_name)
                    self.model.graph.initializer.append(reduce_axes)

            self.model.graph.node.extend([reduce_node, reshape_node])
            self.model.graph.output.append(helper.make_tensor_value_info(reduce_output, onnx_type, [None]))

        for tensor in tensors:
            add_reduce_min_max(tensor, "ReduceMin")
            add_reduce_min_max(tensor, "ReduceMax")

        onnx.save(
            self.model,
            self.augmented_model_path,
            save_as_external_data=self.use_external_data_format,
        )

    def clear_collected_data(self):
        self.intermediate_outputs = []

    def collect_data(self, data_reader: CalibrationDataReader):
        while True:
            inputs = data_reader.get_next()
            if not inputs:
                break
            self.intermediate_outputs.append(self.infer_session.run(None, inputs))
            if (
                self.max_intermediate_outputs is not None
                and len(self.intermediate_outputs) == self.max_intermediate_outputs
            ):
                self.clear_collected_data()

        if len(self.intermediate_outputs) == 0 and self.calibrate_tensors_range is None:
            raise ValueError("No data is collected.")

        t = self.compute_data()
        if not isinstance(t, TensorsData):
            raise TypeError(f"compute_data must return a TensorsData not {type(t)}.")
        self.clear_collected_data()

    def merge_range(self, old_range, new_range):
        if not old_range:
            return new_range

        for key, value in old_range.items():
            # Handling for structured data types with TensorData
            if isinstance(value, TensorData):
                old_min = value.range_value[0]
                old_max = value.range_value[1]
            else:
                old_min, old_max = value

            if isinstance(new_range[key], TensorData):
                new_min = new_range[key].range_value[0]
                new_max = new_range[key].range_value[1]
            else:
                new_min, new_max = new_range[key]

            if self.moving_average:
                min_value = old_min + self.averaging_constant * (new_min - old_min)
                max_value = old_max + self.averaging_constant * (new_max - old_max)
            else:
                min_value = min(old_min, new_min)
                max_value = max(old_max, new_max)

            # If structured as TensorData, wrap the result accordingly
            if isinstance(value, TensorData) or isinstance(new_range[key], TensorData):
                new_range[key] = TensorData(lowest=min_value, highest=max_value)
            else:
                new_range[key] = (min_value, max_value)

        return new_range

    def compute_data(self) -> TensorsData:
        """
        Compute the min-max range of tensor
        :return: dictionary mapping: {added node names: (ReduceMin, ReduceMax) pairs }
        """

        if len(self.intermediate_outputs) == 0:
            return self.calibrate_tensors_range

        output_names = [self.infer_session.get_outputs()[i].name for i in range(len(self.intermediate_outputs[0]))]
        output_dicts_list = [
            dict(zip(output_names, intermediate_output, strict=False))
            for intermediate_output in self.intermediate_outputs
        ]

        merged_output_dict = {}
        for d in output_dicts_list:
            for k, v in d.items():
                merged_output_dict.setdefault(k, []).append(v)
        added_output_names = output_names[self.num_model_outputs :]
        calibrate_tensor_names = [
            added_output_names[i].rpartition("_")[0] for i in range(0, len(added_output_names), 2)
        ]  # output names

        merged_added_output_dict = {
            i: merged_output_dict[i] for i in merged_output_dict if i not in self.model_original_outputs
        }

        pairs = []
        for i in range(0, len(added_output_names), 2):
            if self.moving_average:
                min_value_array = np.nanmean(merged_added_output_dict[added_output_names[i]], axis=0)
                max_value_array = np.nanmean(merged_added_output_dict[added_output_names[i + 1]], axis=0)
            else:
                min_value_array = np.nanmin(merged_added_output_dict[added_output_names[i]], axis=0)
                max_value_array = np.nanmax(merged_added_output_dict[added_output_names[i + 1]], axis=0)

            if self.symmetric:
                max_absolute_value = np.nanmax([np.abs(min_value_array), np.abs(max_value_array)], axis=0)
                pairs.append((-max_absolute_value, max_absolute_value))
            else:
                pairs.append((min_value_array, max_value_array))

        new_calibrate_tensors_range = TensorsData(
            CalibrationMethod.MinMax, dict(zip(calibrate_tensor_names, pairs, strict=False))
        )
        if self.calibrate_tensors_range:
            self.calibrate_tensors_range = self.merge_range(self.calibrate_tensors_range, new_calibrate_tensors_range)
        else:
            self.calibrate_tensors_range = new_calibrate_tensors_range

        return self.calibrate_tensors_range


class HistogramCalibrater(CalibraterBase):
    def __init__(
        self,
        model_path: str | Path,
        op_types_to_calibrate: Sequence[str] | None = None,
        augmented_model_path="augmented_model.onnx",
        use_external_data_format=False,
        method="percentile",
        symmetric=False,
        num_bins=128,
        num_quantized_bins=2048,
        percentile=99.999,
        scenario="same",
    ):
        """
        :param model_path: ONNX model to calibrate. It is a model path.
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        :param use_external_data_format: use external data format to store model which size is >= 2Gb
        :param method: A string. One of ['entropy', 'percentile'].
        :param symmetric: make range of tensor symmetric (central point is 0).
        :param num_bins: number of bins to create a new histogram for collecting tensor values.
        :param num_quantized_bins: number of quantized bins. Default 128.
        :param percentile: A float number between [0, 100]. Default 99.99.
        :param scenario: see :class:`DistributionCalibrater`
        """
        super().__init__(
            model_path,
            op_types_to_calibrate=op_types_to_calibrate,
            augmented_model_path=augmented_model_path,
            symmetric=symmetric,
            use_external_data_format=use_external_data_format,
        )
        self.intermediate_outputs = []
        self.calibrate_tensors_range = None
        self.num_model_outputs = len(self.model.graph.output)
        self.model_original_outputs = {output.name for output in self.model.graph.output}
        self.collector = None
        self.method = method
        self.num_bins = num_bins
        self.num_quantized_bins = num_quantized_bins
        self.percentile = percentile
        self.tensors_to_calibrate = None
        self.scenario = scenario

    def augment_graph(self):
        """
        make all quantization_candidates op type nodes as part of the graph output.
        :return: augmented ONNX model
        """
        self.tensors_to_calibrate, value_infos = self.select_tensors_to_calibrate(self.model)
        for tensor in self.tensors_to_calibrate:
            if tensor not in self.model_original_outputs:
                self.model.graph.output.append(value_infos[tensor])

        onnx.save(
            self.model,
            self.augmented_model_path,
            save_as_external_data=self.use_external_data_format,
        )

    def clear_collected_data(self):
        self.intermediate_outputs = []

    def collect_data(self, data_reader: CalibrationDataReader):
        """
        Entropy Calibrator collects operators' tensors as well as generates tensor histogram for each operator.
        """
        input_names_set = {node_arg.name for node_arg in self.infer_session.get_inputs()}
        output_names = [node_arg.name for node_arg in self.infer_session.get_outputs()]

        while True:
            inputs = data_reader.get_next()
            if not inputs:
                break
            outputs = self.infer_session.run(None, inputs)

            # Copy np.ndarray only for graph outputs that are also graph inputs to workaround bug:
            # https://github.com/microsoft/onnxruntime/issues/21922
            fixed_outputs = []
            for output_index, output in enumerate(outputs):
                if output_names[output_index] in input_names_set:
                    fixed_outputs.append(copy.copy(output))
                else:
                    fixed_outputs.append(output)

            self.intermediate_outputs.append(fixed_outputs)

        if len(self.intermediate_outputs) == 0:
            raise ValueError("No data is collected.")

        output_dicts_list = [
            dict(zip(output_names, intermediate_output, strict=False))
            for intermediate_output in self.intermediate_outputs
        ]

        merged_dict = {}
        for d in output_dicts_list:
            for k, v in d.items():
                merged_dict.setdefault(k, []).append(v)

        clean_merged_dict = {i: merged_dict[i] for i in merged_dict if i in self.tensors_to_calibrate}

        if not self.collector:
            self.collector = HistogramCollector(
                method=self.method,
                symmetric=self.symmetric,
                num_bins=self.num_bins,
                num_quantized_bins=self.num_quantized_bins,
                percentile=self.percentile,
                scenario=self.scenario,
            )
        self.collector.collect(clean_merged_dict)

        self.clear_collected_data()

    def compute_data(self) -> TensorsData:
        """
        Compute the min-max range of tensor
        :return: dictionary mapping: {tensor name: (min value, max value)}
        """
        if not self.collector:
            raise ValueError("No collector created and can't generate calibration data.")

        if isinstance(self, EntropyCalibrater):
            cal = CalibrationMethod.Entropy
        elif isinstance(self, PercentileCalibrater):
            cal = CalibrationMethod.Percentile
        elif isinstance(self, DistributionCalibrater):
            cal = CalibrationMethod.Distribution
        else:
            raise TypeError(f"Unknown calibrater {type(self)}. This method must be overwritten.")
        return TensorsData(cal, self.collector.compute_collection_result())


class EntropyCalibrater(HistogramCalibrater):
    def __init__(
        self,
        model_path: str | Path,
        op_types_to_calibrate: Sequence[str] | None = None,
        augmented_model_path="augmented_model.onnx",
        use_external_data_format=False,
        method="entropy",
        symmetric=False,
        num_bins=128,
        num_quantized_bins=128,
    ):
        """
        :param model_path: ONNX model to calibrate. It is a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        :param use_external_data_format: use external data format to store model which size is >= 2Gb
        :param method: A string. One of ['entropy', 'percentile', 'distribution'].
        :param symmetric: make range of tensor symmetric (central point is 0).
        :param num_bins: number of bins to create a new histogram for collecting tensor values.
        :param num_quantized_bins: number of quantized bins. Default 128.
        """
        super().__init__(
            model_path,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format,
            method=method,
            symmetric=symmetric,
            num_bins=num_bins,
            num_quantized_bins=num_quantized_bins,
        )


class PercentileCalibrater(HistogramCalibrater):
    def __init__(
        self,
        model_path: str | Path,
        op_types_to_calibrate: Sequence[str] | None = None,
        augmented_model_path="augmented_model.onnx",
        use_external_data_format=False,
        method="percentile",
        symmetric=False,
        num_bins=2048,
        percentile=99.999,
    ):
        """
        :param model_path: ONNX model to calibrate. It is a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        :param use_external_data_format: use external data format to store model which size is >= 2Gb
        :param method: A string. One of ['entropy', 'percentile', 'distribution'].
        :param symmetric: make range of tensor symmetric (central point is 0).
        :param num_quantized_bins: number of quantized bins. Default 128.
        :param percentile: A float number between [0, 100]. Default 99.99.
        """
        super().__init__(
            model_path,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format,
            method=method,
            symmetric=symmetric,
            num_bins=num_bins,
            percentile=percentile,
        )


class DistributionCalibrater(HistogramCalibrater):
    def __init__(
        self,
        model_path: str | Path,
        op_types_to_calibrate: Sequence[str] | None = None,
        augmented_model_path="augmented_model.onnx",
        use_external_data_format=False,
        method="distribution",
        num_bins=128,
        scenario="same",
    ):
        """
        :param model_path: ONNX model to calibrate. It is a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        :param use_external_data_format: use external data format to store model which size is >= 2Gb
        :param method: A string. One of ['entropy', 'percentile', 'distribution'].
        :param symmetric: make range of tensor symmetric (central point is 0).
        :param num_bins: number of bins to create a new histogram for collecting tensor values.
        :param scenario: for float 8 only, if `scenario="same"`,
            the algorithm weights and float 8 follow the same distribution,
            if `scenario="p3"`, it assumes the weights follow
            a gaussian law and float 8 ~ X^3 where X is a gaussian law
        """
        super().__init__(
            model_path,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format,
            method=method,
            num_bins=num_bins,
            scenario=scenario,
        )


class CalibrationDataCollector(metaclass=abc.ABCMeta):
    """
    Base class for collecting data for calibration-based quantization.
    """

    @abc.abstractmethod
    def collect(self, name_to_arr):
        """
        Generate informative data based on given data.
            name_to_arr : dict
                tensor name to NDArray data
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_collection_result(self):
        """
        Get the optimal result among collection data.
        """
        raise NotImplementedError


class HistogramCollector(CalibrationDataCollector):
    """
    Collecting histogram for each tensor. Percentile and Entropy method are supported.

    ref: https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    ref: https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/_modules/
                 pytorch_quantization/calib/histogram.html
    """

    def __init__(self, method, symmetric, num_bins, num_quantized_bins, percentile, scenario):
        self.histogram_dict = {}
        self.method = method
        self.symmetric = symmetric
        self.num_bins = num_bins
        self.num_quantized_bins = num_quantized_bins
        self.percentile = percentile
        self.scenario = scenario

    def get_histogram_dict(self):
        return self.histogram_dict

    def collect(self, name_to_arr):
        print("Collecting tensor data and making histogram ...")

        # TODO: Currently we have different collect() for entropy and percentile method respectively.
        #       Need unified collect in the future.
        if self.method in {"distribution", "entropy"}:
            return self.collect_value(name_to_arr)
        elif self.method == "percentile":
            if self.symmetric:
                return self.collect_absolute_value(name_to_arr)
            else:
                return self.collect_value(name_to_arr)
        else:
            raise ValueError("Only 'entropy', 'percentile' or 'distribution' methods are supported")

    def collect_absolute_value(self, name_to_arr):
        """
        Collect histogram on absolute value
        """
        for tensor, data_arr in name_to_arr.items():
            if isinstance(data_arr, list):
                for arr in data_arr:
                    assert isinstance(arr, np.ndarray), f"Unexpected type {type(arr)} for tensor={tensor!r}"
                dtypes = {a.dtype for a in data_arr}
                assert len(dtypes) == 1, (
                    f"The calibration expects only one element type but got {dtypes} for tensor={tensor!r}"
                )
                data_arr_np = np.asarray(data_arr)
            elif not isinstance(data_arr, np.ndarray):
                raise ValueError(f"Unexpected type {type(data_arr)} for tensor={tensor!r}")
            else:
                data_arr_np = data_arr
            data_arr_np = data_arr_np.flatten()
            if data_arr_np.size > 0:
                min_value = np.nanmin(data_arr_np)
                max_value = np.nanmax(data_arr_np)
            else:
                min_value = np.array(0, dtype=data_arr_np.dtype)
                max_value = np.array(0, dtype=data_arr_np.dtype)

            data_arr_np = np.absolute(data_arr_np)  # only consider absolute value

            if tensor not in self.histogram_dict:
                # first time it uses num_bins to compute histogram.
                hist, hist_edges = np.histogram(data_arr_np, bins=self.num_bins)
                hist_edges = hist_edges.astype(data_arr_np.dtype)
                assert data_arr_np.dtype != np.float64, (
                    "only float32 or float16 is supported, every constant must be explicitly typed"
                )
                self.histogram_dict[tensor] = (hist, hist_edges, min_value, max_value)
            else:
                old_histogram = self.histogram_dict[tensor]
                old_min = old_histogram[2]
                old_max = old_histogram[3]
                assert hasattr(old_min, "dtype"), f"old_min should be a numpy array but is {type(old_min)}"
                assert hasattr(old_max, "dtype"), f"old_min should be a numpy array but is {type(old_max)}"
                old_hist = old_histogram[0]
                old_hist_edges = old_histogram[1]
                temp_amax = np.nanmax(data_arr_np)
                if temp_amax > old_hist_edges[-1]:
                    # increase the number of bins
                    width = old_hist_edges[1] - old_hist_edges[0]
                    # NOTE: np.arange may create an extra bin after the one containing temp_amax
                    new_bin_edges = np.arange(old_hist_edges[-1] + width, temp_amax + width, width)
                    old_hist_edges = np.hstack((old_hist_edges, new_bin_edges))
                hist, hist_edges = np.histogram(data_arr_np, bins=old_hist_edges)
                hist_edges = hist_edges.astype(data_arr_np.dtype)
                hist[: len(old_hist)] += old_hist
                assert data_arr_np.dtype != np.float64, (
                    "only float32 or float16 is supported, every constant must be explicitly typed"
                )
                self.histogram_dict[tensor] = (hist, hist_edges, min(old_min, min_value), max(old_max, max_value))

    def collect_value(self, name_to_arr):
        """
        Collect histogram on real value
        """
        for tensor, data_arr in name_to_arr.items():
            data_arr = np.asarray(data_arr)  # noqa: PLW2901
            data_arr = data_arr.flatten()  # noqa: PLW2901

            if data_arr.size > 0:
                min_value = np.nanmin(data_arr)
                max_value = np.nanmax(data_arr)
            else:
                min_value = np.array(0, dtype=data_arr.dtype)
                max_value = np.array(0, dtype=data_arr.dtype)

            threshold = np.array(max(abs(min_value), abs(max_value)), dtype=data_arr.dtype)

            if tensor in self.histogram_dict:
                old_histogram = self.histogram_dict[tensor]
                self.histogram_dict[tensor] = self.merge_histogram(
                    old_histogram, data_arr, min_value, max_value, threshold
                )
            else:
                hist, hist_edges = np.histogram(data_arr, self.num_bins, range=(-threshold, threshold))
                self.histogram_dict[tensor] = (
                    hist,
                    hist_edges,
                    min_value,
                    max_value,
                    threshold,
                )

    def merge_histogram(self, old_histogram, data_arr, new_min, new_max, new_threshold):
        (old_hist, old_hist_edges, old_min, old_max, old_threshold) = old_histogram

        if new_threshold <= old_threshold:
            new_hist, _ = np.histogram(data_arr, len(old_hist), range=(-old_threshold, old_threshold))
            return (
                new_hist + old_hist,
                old_hist_edges,
                min(old_min, new_min),
                max(old_max, new_max),
                old_threshold,
            )
        else:
            if old_threshold == 0:
                hist, hist_edges = np.histogram(data_arr, len(old_hist), range=(-new_threshold, new_threshold))
                hist += old_hist
            else:
                old_num_bins = len(old_hist)
                old_stride = 2 * old_threshold / old_num_bins
                half_increased_bins = int((new_threshold - old_threshold) // old_stride + 1)
                new_num_bins = old_num_bins + 2 * half_increased_bins
                new_threshold = half_increased_bins * old_stride + old_threshold
                hist, hist_edges = np.histogram(data_arr, new_num_bins, range=(-new_threshold, new_threshold))
                hist[half_increased_bins : new_num_bins - half_increased_bins] += old_hist
            return (
                hist,
                hist_edges,
                min(old_min, new_min),
                max(old_max, new_max),
                new_threshold,
            )

    def compute_collection_result(self):
        if not self.histogram_dict or len(self.histogram_dict) == 0:
            raise ValueError("Histogram has not been collected. Please run collect() first.")
        print(f"Finding optimal threshold for each tensor using {self.method!r} algorithm ...")

        if self.method == "entropy":
            return self.compute_entropy()
        elif self.method == "percentile":
            return self.compute_percentile()
        elif self.method == "distribution":
            return self.compute_distribution()
        else:
            raise ValueError("Only 'entropy', 'percentile' or 'distribution' methods are supported")

    def compute_percentile(self):
        if self.percentile < 0 or self.percentile > 100:
            raise ValueError("Invalid percentile. Must be in range 0 <= percentile <= 100.")

        histogram_dict = self.histogram_dict
        percentile = self.percentile

        thresholds_dict = {}  # per tensor thresholds

        print(f"Number of tensors : {len(histogram_dict)}")
        print(f"Number of histogram bins : {self.num_bins}")
        print(f"Percentile : ({100.0 - percentile},{percentile})")

        for tensor, histogram in histogram_dict.items():
            hist = histogram[0]
            hist_edges = histogram[1]
            total = hist.sum()
            cdf = np.cumsum(hist / total)
            if self.symmetric:
                idx_right = np.searchsorted(cdf, percentile / 100.0)

                thresholds_dict[tensor] = (
                    -np.array(hist_edges[idx_right], dtype=hist_edges.dtype),
                    np.array(hist_edges[idx_right], dtype=hist_edges.dtype),
                )
            else:
                percent_to_cut_one_side = (100.0 - percentile) / 200.0
                idx_right = np.searchsorted(cdf, 1.0 - percent_to_cut_one_side)
                idx_left = np.searchsorted(cdf, percent_to_cut_one_side)
                thresholds_dict[tensor] = (
                    np.array(hist_edges[idx_left], dtype=hist_edges.dtype),
                    np.array(hist_edges[idx_right], dtype=hist_edges.dtype),
                )
            min_value = histogram[2]
            max_value = histogram[3]
            if thresholds_dict[tensor][0] < min_value:
                thresholds_dict[tensor] = (min_value, thresholds_dict[tensor][1])
            if thresholds_dict[tensor][1] > max_value:
                thresholds_dict[tensor] = (thresholds_dict[tensor][0], max_value)
            thresholds_dict[tensor] = (*thresholds_dict[tensor], *hist[:2])
            # Plot histogram for debug only
            if os.environ.get("QUANTIZATION_DEBUG", 0) in (1, "1"):
                apply_plot(hist, hist_edges)

        return thresholds_dict

    def compute_entropy(self):
        histogram_dict = self.histogram_dict
        num_quantized_bins = self.num_quantized_bins

        thresholds_dict = {}  # per tensor thresholds

        print(f"Number of tensors : {len(histogram_dict)}")
        print(f"Number of histogram bins : {self.num_bins} (The number may increase depends on the data it collects)")
        print(f"Number of quantized bins : {self.num_quantized_bins}")

        for tensor, histogram in histogram_dict.items():
            optimal_threshold = self.get_entropy_threshold(histogram, num_quantized_bins)
            thresholds_dict[tensor] = optimal_threshold
            thresholds_dict[tensor] = (*optimal_threshold, *histogram[:2])

            # Plot histogram for debug only
            if os.environ.get("QUANTIZATION_DEBUG", 0) in (1, "1"):
                apply_plot(histogram[0], histogram[1])

        return thresholds_dict

    @staticmethod
    def _avg_std(hist, hist_edges, power=1):
        if power <= 0:
            raise ValueError(f"power={power} <= 0 is invalid.")
        values = (hist_edges[:-1] + hist_edges[1:]) * 0.5
        if power == 1:
            avg = (hist * values).sum() / hist.sum()
            std = ((hist * values**2).sum() / hist.sum() - avg**2) ** 0.5
            return np.array(avg, dtype=hist_edges.dtype), np.array(std, dtype=hist_edges.dtype)
        if int(power) == power and int(power) % 2 == 1:
            avg = (hist * values**power).sum() / hist.sum()
            std = ((hist * (values**power - avg) ** 2).sum() / hist.sum()) ** 0.5
            return np.array(avg, dtype=hist_edges.dtype), np.array(std, dtype=hist_edges.dtype)

        fact = np.abs(values) / values
        fact[np.isnan(fact)] = 1
        fact[np.isinf(fact)] = 1
        values = np.abs(values) ** power * fact
        avg = (hist * values).sum() / hist.sum()
        std = ((hist * values**2).sum() / hist.sum() - avg**2) ** 0.5
        return np.array(avg, dtype=hist_edges.dtype), np.array(std, dtype=hist_edges.dtype)

    def compute_distribution(self):
        if self.num_bins < 512:
            raise ValueError("Invalid num_bins. Must be in range 512 <= num_bins.")

        histogram_dict = self.histogram_dict
        thresholds_dict = {}  # per tensor thresholds

        print(f"Number of tensors : {len(histogram_dict)}")
        print(f"Number of histogram bins : {self.num_bins}")
        print(f"Scenario : {self.scenario!r})")

        for tensor, histogram in histogram_dict.items():
            hist = histogram[0]
            hist_edges = histogram[1]

            assert hist_edges.dtype != np.float64
            if self.scenario == "same":
                avg_coef, std_coef = self._avg_std(hist, hist_edges, power=1)
            elif self.scenario == "p3":
                avg_coef, std_coef = self._avg_std(hist, hist_edges, power=1.0 / 3.0)
            else:
                raise ValueError("Invalid scenario. Must be in {'same', 'p3'}.")
            assert avg_coef.dtype != np.float64
            assert std_coef.dtype != np.float64
            assert hist_edges.dtype != np.float64
            thresholds_dict[tensor] = TensorData(
                avg=avg_coef,
                std=std_coef,
                hist=hist,
                hist_edges=hist_edges,
                lowest=hist_edges.min(),
                highest=hist_edges.max(),
            )

            # Plot histogram for debug only
            if os.environ.get("QUANTIZATION_DEBUG", 0) in (1, "1"):
                apply_plot(hist, hist_edges)

        return thresholds_dict

    def get_entropy_threshold(self, histogram, num_quantized_bins):
        """Given a dataset, find the optimal threshold for quantizing it.
        The reference distribution is `q`, and the candidate distribution is `p`.
        `q` is a truncated version of the original distribution.
        Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
        """
        hist = histogram[0]
        hist_edges = histogram[1]
        num_bins = hist.size
        zero_bin_index = num_bins // 2
        num_half_quantized_bin = num_quantized_bins // 2

        dtype = histogram[1].dtype
        kl_divergence = np.zeros(zero_bin_index - num_half_quantized_bin + 1)
        thresholds = [(np.array(0, dtype=dtype), np.array(0, dtype=dtype)) for i in range(kl_divergence.size)]

        # <------------ num bins ---------------->
        #        <--- quantized bins ---->
        # |======|===========|===========|=======|
        #              zero bin index
        #        ^                       ^
        #        |                       |
        #   start index               end index          (start of iteration)
        #     ^                             ^
        #     |                             |
        #  start index                  end index               ...
        # ^                                      ^
        # |                                      |
        # start index                    end index       (end of iteration)

        for i in range(num_half_quantized_bin, zero_bin_index + 1, 1):
            start_index = zero_bin_index - i
            end_index = min(zero_bin_index + i + 1, num_bins)

            thresholds[i - num_half_quantized_bin] = (hist_edges[start_index], hist_edges[end_index])

            sliced_distribution = copy.deepcopy(hist[start_index:end_index])

            # reference distribution p
            p = sliced_distribution.copy()  # a copy of np array
            left_outliers_count = sum(hist[:start_index])
            right_outliers_count = sum(hist[end_index:])
            p[0] += left_outliers_count
            p[-1] += right_outliers_count

            # nonzeros[i] incidates whether p[i] is non-zero
            nonzeros = (p != 0).astype(np.int64)

            # quantize p.size bins into quantized bins (default 128 bins)
            quantized_bins = np.zeros(num_quantized_bins, dtype=np.int64)
            num_merged_bins = sliced_distribution.size // num_quantized_bins

            # merge bins into quantized bins
            for index in range(num_quantized_bins):
                start = index * num_merged_bins
                end = start + num_merged_bins
                quantized_bins[index] = sum(sliced_distribution[start:end])
            quantized_bins[-1] += sum(sliced_distribution[num_quantized_bins * num_merged_bins :])

            # in order to compare p and q, we need to make length of q equals to length of p
            # expand quantized bins into p.size bins
            q = np.zeros(p.size, dtype=np.int64)
            for index in range(num_quantized_bins):
                start = index * num_merged_bins
                end = start + num_merged_bins

                norm = sum(nonzeros[start:end])
                if norm != 0:
                    q[start:end] = quantized_bins[index] / norm

            p = smooth_distribution(p)
            q = smooth_distribution(q)
            if p is None or q is None:
                div = np.array(np.inf, dtype=dtype)
            else:
                div = np.array(entropy(p, q), dtype=dtype)
            kl_divergence[i - num_half_quantized_bin] = div

        min_kl_divergence_idx = np.argmin(kl_divergence)
        optimal_threshold = thresholds[min_kl_divergence_idx]
        min_value = histogram[2]
        max_value = histogram[3]
        if optimal_threshold[0] < min_value:
            optimal_threshold = (min_value, optimal_threshold[1])
        if optimal_threshold[1] > max_value:
            optimal_threshold = (optimal_threshold[0], max_value)
        assert hasattr(optimal_threshold[0], "dtype")
        assert hasattr(optimal_threshold[1], "dtype")
        return optimal_threshold


def create_calibrator(
    model: str | Path,
    op_types_to_calibrate: Sequence[str] | None = None,
    augmented_model_path="augmented_model.onnx",
    calibrate_method=CalibrationMethod.MinMax,
    use_external_data_format=False,
    providers=None,
    extra_options={},  # noqa: B006
):
    calibrator = None
    if calibrate_method == CalibrationMethod.MinMax:
        # default settings for min-max algorithm
        symmetric = extra_options.get("symmetric", False)
        moving_average = extra_options.get("moving_average", False)
        averaging_constant = extra_options.get("averaging_constant", 0.01)
        max_intermediate_outputs = extra_options.get("max_intermediate_outputs", None)
        per_channel = extra_options.get("per_channel", False)
        calibrator = MinMaxCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            moving_average=moving_average,
            averaging_constant=averaging_constant,
            max_intermediate_outputs=max_intermediate_outputs,
            per_channel=per_channel,
        )
    elif calibrate_method == CalibrationMethod.Entropy:
        # default settings for entropy algorithm
        num_bins = extra_options.get("num_bins", 128)
        num_quantized_bins = extra_options.get("num_quantized_bins", 128)
        symmetric = extra_options.get("symmetric", False)
        calibrator = EntropyCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            num_bins=num_bins,
            num_quantized_bins=num_quantized_bins,
        )
    elif calibrate_method == CalibrationMethod.Percentile:
        # default settings for percentile algorithm
        num_bins = extra_options.get("num_bins", 2048)
        percentile = extra_options.get("percentile", 99.999)
        symmetric = extra_options.get("symmetric", True)
        calibrator = PercentileCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            num_bins=num_bins,
            percentile=percentile,
        )

    elif calibrate_method == CalibrationMethod.Distribution:
        # default settings for percentile algorithm
        num_bins = extra_options.get("num_bins", 2048)
        scenario = extra_options.get("scenario", "same")

        calibrator = DistributionCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            num_bins=num_bins,
            scenario=scenario,
        )

    if calibrator:
        calibrator.augment_graph()
        if providers:
            calibrator.execution_providers = providers
        calibrator.create_inference_session()
        return calibrator

    raise ValueError(f"Unsupported calibration method {calibrate_method}")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\trigsimp.py ===
from collections import defaultdict
from functools import reduce

from sympy.core import (sympify, Basic, S, Expr, factor_terms,
                        Mul, Add, bottom_up)
from sympy.core.cache import cacheit
from sympy.core.function import (count_ops, _mexpand, FunctionClass, expand,
                                 expand_mul, _coeff_isneg, Derivative)
from sympy.core.numbers import I, Integer
from sympy.core.intfunc import igcd
from sympy.core.sorting import _nodes
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import sin, cos, exp, cosh, tanh, sinh, tan, cot, coth
from sympy.functions import atan2
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.polys import Poly, factor, cancel, parallel_poly_from_expr
from sympy.polys.domains import ZZ
from sympy.polys.polyerrors import PolificationFailed
from sympy.polys.polytools import groebner
from sympy.simplify.cse_main import cse
from sympy.strategies.core import identity
from sympy.strategies.tree import greedy
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug

def trigsimp_groebner(expr, hints=[], quick=False, order="grlex",
                      polynomial=False):
    """
    Simplify trigonometric expressions using a groebner basis algorithm.

    Explanation
    ===========

    This routine takes a fraction involving trigonometric or hyperbolic
    expressions, and tries to simplify it. The primary metric is the
    total degree. Some attempts are made to choose the simplest possible
    expression of the minimal degree, but this is non-rigorous, and also
    very slow (see the ``quick=True`` option).

    If ``polynomial`` is set to True, instead of simplifying numerator and
    denominator together, this function just brings numerator and denominator
    into a canonical form. This is much faster, but has potentially worse
    results. However, if the input is a polynomial, then the result is
    guaranteed to be an equivalent polynomial of minimal degree.

    The most important option is hints. Its entries can be any of the
    following:

    - a natural number
    - a function
    - an iterable of the form (func, var1, var2, ...)
    - anything else, interpreted as a generator

    A number is used to indicate that the search space should be increased.
    A function is used to indicate that said function is likely to occur in a
    simplified expression.
    An iterable is used indicate that func(var1 + var2 + ...) is likely to
    occur in a simplified .
    An additional generator also indicates that it is likely to occur.
    (See examples below).

    This routine carries out various computationally intensive algorithms.
    The option ``quick=True`` can be used to suppress one particularly slow
    step (at the expense of potentially more complicated results, but never at
    the expense of increased total degree).

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import sin, tan, cos, sinh, cosh, tanh
    >>> from sympy.simplify.trigsimp import trigsimp_groebner

    Suppose you want to simplify ``sin(x)*cos(x)``. Naively, nothing happens:

    >>> ex = sin(x)*cos(x)
    >>> trigsimp_groebner(ex)
    sin(x)*cos(x)

    This is because ``trigsimp_groebner`` only looks for a simplification
    involving just ``sin(x)`` and ``cos(x)``. You can tell it to also try
    ``2*x`` by passing ``hints=[2]``:

    >>> trigsimp_groebner(ex, hints=[2])
    sin(2*x)/2
    >>> trigsimp_groebner(sin(x)**2 - cos(x)**2, hints=[2])
    -cos(2*x)

    Increasing the search space this way can quickly become expensive. A much
    faster way is to give a specific expression that is likely to occur:

    >>> trigsimp_groebner(ex, hints=[sin(2*x)])
    sin(2*x)/2

    Hyperbolic expressions are similarly supported:

    >>> trigsimp_groebner(sinh(2*x)/sinh(x))
    2*cosh(x)

    Note how no hints had to be passed, since the expression already involved
    ``2*x``.

    The tangent function is also supported. You can either pass ``tan`` in the
    hints, to indicate that tan should be tried whenever cosine or sine are,
    or you can pass a specific generator:

    >>> trigsimp_groebner(sin(x)/cos(x), hints=[tan])
    tan(x)
    >>> trigsimp_groebner(sinh(x)/cosh(x), hints=[tanh(x)])
    tanh(x)

    Finally, you can use the iterable form to suggest that angle sum formulae
    should be tried:

    >>> ex = (tan(x) + tan(y))/(1 - tan(x)*tan(y))
    >>> trigsimp_groebner(ex, hints=[(tan, x, y)])
    tan(x + y)
    """
    # TODO
    #  - preprocess by replacing everything by funcs we can handle
    # - optionally use cot instead of tan
    # - more intelligent hinting.
    #     For example, if the ideal is small, and we have sin(x), sin(y),
    #     add sin(x + y) automatically... ?
    # - algebraic numbers ...
    # - expressions of lowest degree are not distinguished properly
    #   e.g. 1 - sin(x)**2
    # - we could try to order the generators intelligently, so as to influence
    #   which monomials appear in the quotient basis

    # THEORY
    # ------
    # Ratsimpmodprime above can be used to "simplify" a rational function
    # modulo a prime ideal. "Simplify" mainly means finding an equivalent
    # expression of lower total degree.
    #
    # We intend to use this to simplify trigonometric functions. To do that,
    # we need to decide (a) which ring to use, and (b) modulo which ideal to
    # simplify. In practice, (a) means settling on a list of "generators"
    # a, b, c, ..., such that the fraction we want to simplify is a rational
    # function in a, b, c, ..., with coefficients in ZZ (integers).
    # (2) means that we have to decide what relations to impose on the
    # generators. There are two practical problems:
    #   (1) The ideal has to be *prime* (a technical term).
    #   (2) The relations have to be polynomials in the generators.
    #
    # We typically have two kinds of generators:
    # - trigonometric expressions, like sin(x), cos(5*x), etc
    # - "everything else", like gamma(x), pi, etc.
    #
    # Since this function is trigsimp, we will concentrate on what to do with
    # trigonometric expressions. We can also simplify hyperbolic expressions,
    # but the extensions should be clear.
    #
    # One crucial point is that all *other* generators really should behave
    # like indeterminates. In particular if (say) "I" is one of them, then
    # in fact I**2 + 1 = 0 and we may and will compute non-sensical
    # expressions. However, we can work with a dummy and add the relation
    # I**2 + 1 = 0 to our ideal, then substitute back in the end.
    #
    # Now regarding trigonometric generators. We split them into groups,
    # according to the argument of the trigonometric functions. We want to
    # organise this in such a way that most trigonometric identities apply in
    # the same group. For example, given sin(x), cos(2*x) and cos(y), we would
    # group as [sin(x), cos(2*x)] and [cos(y)].
    #
    # Our prime ideal will be built in three steps:
    # (1) For each group, compute a "geometrically prime" ideal of relations.
    #     Geometrically prime means that it generates a prime ideal in
    #     CC[gens], not just ZZ[gens].
    # (2) Take the union of all the generators of the ideals for all groups.
    #     By the geometric primality condition, this is still prime.
    # (3) Add further inter-group relations which preserve primality.
    #
    # Step (1) works as follows. We will isolate common factors in the
    # argument, so that all our generators are of the form sin(n*x), cos(n*x)
    # or tan(n*x), with n an integer. Suppose first there are no tan terms.
    # The ideal [sin(x)**2 + cos(x)**2 - 1] is geometrically prime, since
    # X**2 + Y**2 - 1 is irreducible over CC.
    # Now, if we have a generator sin(n*x), than we can, using trig identities,
    # express sin(n*x) as a polynomial in sin(x) and cos(x). We can add this
    # relation to the ideal, preserving geometric primality, since the quotient
    # ring is unchanged.
    # Thus we have treated all sin and cos terms.
    # For tan(n*x), we add a relation tan(n*x)*cos(n*x) - sin(n*x) = 0.
    # (This requires of course that we already have relations for cos(n*x) and
    # sin(n*x).) It is not obvious, but it seems that this preserves geometric
    # primality.
    # XXX A real proof would be nice. HELP!
    #     Sketch that <S**2 + C**2 - 1, C*T - S> is a prime ideal of
    #     CC[S, C, T]:
    #     - it suffices to show that the projective closure in CP**3 is
    #       irreducible
    #     - using the half-angle substitutions, we can express sin(x), tan(x),
    #       cos(x) as rational functions in tan(x/2)
    #     - from this, we get a rational map from CP**1 to our curve
    #     - this is a morphism, hence the curve is prime
    #
    # Step (2) is trivial.
    #
    # Step (3) works by adding selected relations of the form
    # sin(x + y) - sin(x)*cos(y) - sin(y)*cos(x), etc. Geometric primality is
    # preserved by the same argument as before.

    def parse_hints(hints):
        """Split hints into (n, funcs, iterables, gens)."""
        n = 1
        funcs, iterables, gens = [], [], []
        for e in hints:
            if isinstance(e, (SYMPY_INTS, Integer)):
                n = e
            elif isinstance(e, FunctionClass):
                funcs.append(e)
            elif iterable(e):
                iterables.append((e[0], e[1:]))
                # XXX sin(x+2y)?
                # Note: we go through polys so e.g.
                # sin(-x) -> -sin(x) -> sin(x)
                gens.extend(parallel_poly_from_expr(
                    [e[0](x) for x in e[1:]] + [e[0](Add(*e[1:]))])[1].gens)
            else:
                gens.append(e)
        return n, funcs, iterables, gens

    def build_ideal(x, terms):
        """
        Build generators for our ideal. ``Terms`` is an iterable with elements of
        the form (fn, coeff), indicating that we have a generator fn(coeff*x).

        If any of the terms is trigonometric, sin(x) and cos(x) are guaranteed
        to appear in terms. Similarly for hyperbolic functions. For tan(n*x),
        sin(n*x) and cos(n*x) are guaranteed.
        """
        I = []
        y = Dummy('y')
        for fn, coeff in terms:
            for c, s, t, rel in (
                    [cos, sin, tan, cos(x)**2 + sin(x)**2 - 1],
                    [cosh, sinh, tanh, cosh(x)**2 - sinh(x)**2 - 1]):
                if coeff == 1 and fn in [c, s]:
                    I.append(rel)
                elif fn == t:
                    I.append(t(coeff*x)*c(coeff*x) - s(coeff*x))
                elif fn in [c, s]:
                    cn = fn(coeff*y).expand(trig=True).subs(y, x)
                    I.append(fn(coeff*x) - cn)
        return list(set(I))

    def analyse_gens(gens, hints):
        """
        Analyse the generators ``gens``, using the hints ``hints``.

        The meaning of ``hints`` is described in the main docstring.
        Return a new list of generators, and also the ideal we should
        work with.
        """
        # First parse the hints
        n, funcs, iterables, extragens = parse_hints(hints)
        debug('n=%s   funcs: %s   iterables: %s    extragens: %s',
              (funcs, iterables, extragens))

        # We just add the extragens to gens and analyse them as before
        gens = list(gens)
        gens.extend(extragens)

        # remove duplicates
        funcs = list(set(funcs))
        iterables = list(set(iterables))
        gens = list(set(gens))

        # all the functions we can do anything with
        allfuncs = {sin, cos, tan, sinh, cosh, tanh}
        # sin(3*x) -> ((3, x), sin)
        trigterms = [(g.args[0].as_coeff_mul(), g.func) for g in gens
                     if g.func in allfuncs]
        # Our list of new generators - start with anything that we cannot
        # work with (i.e. is not a trigonometric term)
        freegens = [g for g in gens if g.func not in allfuncs]
        newgens = []
        trigdict = {}
        for (coeff, var), fn in trigterms:
            trigdict.setdefault(var, []).append((coeff, fn))
        res = [] # the ideal

        for key, val in trigdict.items():
            # We have now assembeled a dictionary. Its keys are common
            # arguments in trigonometric expressions, and values are lists of
            # pairs (fn, coeff). x0, (fn, coeff) in trigdict means that we
            # need to deal with fn(coeff*x0). We take the rational gcd of the
            # coeffs, call it ``gcd``. We then use x = x0/gcd as "base symbol",
            # all other arguments are integral multiples thereof.
            # We will build an ideal which works with sin(x), cos(x).
            # If hint tan is provided, also work with tan(x). Moreover, if
            # n > 1, also work with sin(k*x) for k <= n, and similarly for cos
            # (and tan if the hint is provided). Finally, any generators which
            # the ideal does not work with but we need to accommodate (either
            # because it was in expr or because it was provided as a hint)
            # we also build into the ideal.
            # This selection process is expressed in the list ``terms``.
            # build_ideal then generates the actual relations in our ideal,
            # from this list.
            fns = [x[1] for x in val]
            val = [x[0] for x in val]
            gcd = reduce(igcd, val)
            terms = [(fn, v/gcd) for (fn, v) in zip(fns, val)]
            fs = set(funcs + fns)
            for c, s, t in ([cos, sin, tan], [cosh, sinh, tanh]):
                if any(x in fs for x in (c, s, t)):
                    fs.add(c)
                    fs.add(s)
            for fn in fs:
                terms.extend((fn, k) for k in range(1, n + 1))
            extra = []
            for fn, v in terms:
                if fn == tan:
                    extra.append((sin, v))
                    extra.append((cos, v))
                if fn in [sin, cos] and tan in fs:
                    extra.append((tan, v))
                if fn == tanh:
                    extra.append((sinh, v))
                    extra.append((cosh, v))
                if fn in [sinh, cosh] and tanh in fs:
                    extra.append((tanh, v))
            terms.extend(extra)
            x = gcd*Mul(*key)
            r = build_ideal(x, terms)
            res.extend(r)
            newgens.extend({fn(v*x) for fn, v in terms})

        # Add generators for compound expressions from iterables
        for fn, args in iterables:
            if fn == tan:
                # Tan expressions are recovered from sin and cos.
                iterables.extend([(sin, args), (cos, args)])
            elif fn == tanh:
                # Tanh expressions are recovered from sihn and cosh.
                iterables.extend([(sinh, args), (cosh, args)])
            else:
                dummys = symbols('d:%i' % len(args), cls=Dummy)
                expr = fn( Add(*dummys)).expand(trig=True).subs(list(zip(dummys, args)))
                res.append(fn(Add(*args)) - expr)

        if myI in gens:
            res.append(myI**2 + 1)
            freegens.remove(myI)
            newgens.append(myI)

        return res, freegens, newgens

    myI = Dummy('I')
    expr = expr.subs(S.ImaginaryUnit, myI)
    subs = [(myI, S.ImaginaryUnit)]

    num, denom = cancel(expr).as_numer_denom()
    try:
        (pnum, pdenom), opt = parallel_poly_from_expr([num, denom])
    except PolificationFailed:
        return expr
    debug('initial gens:', opt.gens)
    ideal, freegens, gens = analyse_gens(opt.gens, hints)
    debug('ideal:', ideal)
    debug('new gens:', gens, " -- len", len(gens))
    debug('free gens:', freegens, " -- len", len(gens))
    # NOTE we force the domain to be ZZ to stop polys from injecting generators
    #      (which is usually a sign of a bug in the way we build the ideal)
    if not gens:
        return expr
    G = groebner(ideal, order=order, gens=gens, domain=ZZ)
    debug('groebner basis:', list(G), " -- len", len(G))

    # If our fraction is a polynomial in the free generators, simplify all
    # coefficients separately:

    from sympy.simplify.ratsimp import ratsimpmodprime

    if freegens and pdenom.has_only_gens(*set(gens).intersection(pdenom.gens)):
        num = Poly(num, gens=gens+freegens).eject(*gens)
        res = []
        for monom, coeff in num.terms():
            ourgens = set(parallel_poly_from_expr([coeff, denom])[1].gens)
            # We compute the transitive closure of all generators that can
            # be reached from our generators through relations in the ideal.
            changed = True
            while changed:
                changed = False
                for p in ideal:
                    p = Poly(p)
                    if not ourgens.issuperset(p.gens) and \
                       not p.has_only_gens(*set(p.gens).difference(ourgens)):
                        changed = True
                        ourgens.update(p.exclude().gens)
            # NOTE preserve order!
            realgens = [x for x in gens if x in ourgens]
            # The generators of the ideal have now been (implicitly) split
            # into two groups: those involving ourgens and those that don't.
            # Since we took the transitive closure above, these two groups
            # live in subgrings generated by a *disjoint* set of variables.
            # Any sensible groebner basis algorithm will preserve this disjoint
            # structure (i.e. the elements of the groebner basis can be split
            # similarly), and and the two subsets of the groebner basis then
            # form groebner bases by themselves. (For the smaller generating
            # sets, of course.)
            ourG = [g.as_expr() for g in G.polys if
                    g.has_only_gens(*ourgens.intersection(g.gens))]
            res.append(Mul(*[a**b for a, b in zip(freegens, monom)]) * \
                       ratsimpmodprime(coeff/denom, ourG, order=order,
                                       gens=realgens, quick=quick, domain=ZZ,
                                       polynomial=polynomial).subs(subs))
        return Add(*res)
        # NOTE The following is simpler and has less assumptions on the
        #      groebner basis algorithm. If the above turns out to be broken,
        #      use this.
        return Add(*[Mul(*[a**b for a, b in zip(freegens, monom)]) * \
                     ratsimpmodprime(coeff/denom, list(G), order=order,
                                     gens=gens, quick=quick, domain=ZZ)
                     for monom, coeff in num.terms()])
    else:
        return ratsimpmodprime(
            expr, list(G), order=order, gens=freegens+gens,
            quick=quick, domain=ZZ, polynomial=polynomial).subs(subs)


_trigs = (TrigonometricFunction, HyperbolicFunction)


def _trigsimp_inverse(rv):

    def check_args(x, y):
        try:
            return x.args[0] == y.args[0]
        except IndexError:
            return False

    def f(rv):
        # for simple functions
        g = getattr(rv, 'inverse', None)
        if (g is not None and isinstance(rv.args[0], g()) and
                isinstance(g()(1), TrigonometricFunction)):
            return rv.args[0].args[0]

        # for atan2 simplifications, harder because atan2 has 2 args
        if isinstance(rv, atan2):
            y, x = rv.args
            if _coeff_isneg(y):
                return -f(atan2(-y, x))
            elif _coeff_isneg(x):
                return S.Pi - f(atan2(y, -x))

            if check_args(x, y):
                if isinstance(y, sin) and isinstance(x, cos):
                    return x.args[0]
                if isinstance(y, cos) and isinstance(x, sin):
                    return S.Pi / 2 - x.args[0]

        return rv

    return bottom_up(rv, f)


def trigsimp(expr, inverse=False, **opts):
    """Returns a reduced expression by using known trig identities.

    Parameters
    ==========

    inverse : bool, optional
        If ``inverse=True``, it will be assumed that a composition of inverse
        functions, such as sin and asin, can be cancelled in any order.
        For example, ``asin(sin(x))`` will yield ``x`` without checking whether
        x belongs to the set where this relation is true. The default is False.
        Default : True

    method : string, optional
        Specifies the method to use. Valid choices are:

        - ``'matching'``, default
        - ``'groebner'``
        - ``'combined'``
        - ``'fu'``
        - ``'old'``

        If ``'matching'``, simplify the expression recursively by targeting
        common patterns. If ``'groebner'``, apply an experimental groebner
        basis algorithm. In this case further options are forwarded to
        ``trigsimp_groebner``, please refer to
        its docstring. If ``'combined'``, it first runs the groebner basis
        algorithm with small default parameters, then runs the ``'matching'``
        algorithm. If ``'fu'``, run the collection of trigonometric
        transformations described by Fu, et al. (see the
        :py:func:`~sympy.simplify.fu.fu` docstring). If ``'old'``, the original
        SymPy trig simplification function is run.
    opts :
        Optional keyword arguments passed to the method. See each method's
        function docstring for details.

    Examples
    ========

    >>> from sympy import trigsimp, sin, cos, log
    >>> from sympy.abc import x
    >>> e = 2*sin(x)**2 + 2*cos(x)**2
    >>> trigsimp(e)
    2

    Simplification occurs wherever trigonometric functions are located.

    >>> trigsimp(log(e))
    log(2)

    Using ``method='groebner'`` (or ``method='combined'``) might lead to
    greater simplification.

    The old trigsimp routine can be accessed as with method ``method='old'``.

    >>> from sympy import coth, tanh
    >>> t = 3*tanh(x)**7 - 2/coth(x)**7
    >>> trigsimp(t, method='old') == t
    True
    >>> trigsimp(t)
    tanh(x)**7

    """
    from sympy.simplify.fu import fu

    expr = sympify(expr)

    _eval_trigsimp = getattr(expr, '_eval_trigsimp', None)
    if _eval_trigsimp is not None:
        return _eval_trigsimp(**opts)

    old = opts.pop('old', False)
    if not old:
        opts.pop('deep', None)
        opts.pop('recursive', None)
        method = opts.pop('method', 'matching')
    else:
        method = 'old'

    def groebnersimp(ex, **opts):
        def traverse(e):
            if e.is_Atom:
                return e
            args = [traverse(x) for x in e.args]
            if e.is_Function or e.is_Pow:
                args = [trigsimp_groebner(x, **opts) for x in args]
            return e.func(*args)
        new = traverse(ex)
        if not isinstance(new, Expr):
            return new
        return trigsimp_groebner(new, **opts)

    trigsimpfunc = {
        'fu': (lambda x: fu(x, **opts)),
        'matching': (lambda x: futrig(x)),
        'groebner': (lambda x: groebnersimp(x, **opts)),
        'combined': (lambda x: futrig(groebnersimp(x,
                               polynomial=True, hints=[2, tan]))),
        'old': lambda x: trigsimp_old(x, **opts),
                   }[method]

    expr_simplified = trigsimpfunc(expr)
    if inverse:
        expr_simplified = _trigsimp_inverse(expr_simplified)

    return expr_simplified


def exptrigsimp(expr):
    """
    Simplifies exponential / trigonometric / hyperbolic functions.

    Examples
    ========

    >>> from sympy import exptrigsimp, exp, cosh, sinh
    >>> from sympy.abc import z

    >>> exptrigsimp(exp(z) + exp(-z))
    2*cosh(z)
    >>> exptrigsimp(cosh(z) - sinh(z))
    exp(-z)
    """
    from sympy.simplify.fu import hyper_as_trig, TR2i

    def exp_trig(e):
        # select the better of e, and e rewritten in terms of exp or trig
        # functions
        choices = [e]
        if e.has(*_trigs):
            choices.append(e.rewrite(exp))
        choices.append(e.rewrite(cos))
        return min(*choices, key=count_ops)
    newexpr = bottom_up(expr, exp_trig)

    def f(rv):
        if not rv.is_Mul:
            return rv
        commutative_part, noncommutative_part = rv.args_cnc()
        # Since as_powers_dict loses order information,
        # if there is more than one noncommutative factor,
        # it should only be used to simplify the commutative part.
        if (len(noncommutative_part) > 1):
            return f(Mul(*commutative_part))*Mul(*noncommutative_part)
        rvd = rv.as_powers_dict()
        newd = rvd.copy()

        def signlog(expr, sign=S.One):
            if expr is S.Exp1:
                return sign, S.One
            elif isinstance(expr, exp) or (expr.is_Pow and expr.base == S.Exp1):
                return sign, expr.exp
            elif sign is S.One:
                return signlog(-expr, sign=-S.One)
            else:
                return None, None

        ee = rvd[S.Exp1]
        for k in rvd:
            if k.is_Add and len(k.args) == 2:
                # k == c*(1 + sign*E**x)
                c = k.args[0]
                sign, x = signlog(k.args[1]/c)
                if not x:
                    continue
                m = rvd[k]
                newd[k] -= m
                if ee == -x*m/2:
                    # sinh and cosh
                    newd[S.Exp1] -= ee
                    ee = 0
                    if sign == 1:
                        newd[2*c*cosh(x/2)] += m
                    else:
                        newd[-2*c*sinh(x/2)] += m
                elif newd[1 - sign*S.Exp1**x] == -m:
                    # tanh
                    del newd[1 - sign*S.Exp1**x]
                    if sign == 1:
                        newd[-c/tanh(x/2)] += m
                    else:
                        newd[-c*tanh(x/2)] += m
                else:
                    newd[1 + sign*S.Exp1**x] += m
                    newd[c] += m

        return Mul(*[k**newd[k] for k in newd])
    newexpr = bottom_up(newexpr, f)

    # sin/cos and sinh/cosh ratios to tan and tanh, respectively
    if newexpr.has(HyperbolicFunction):
        e, f = hyper_as_trig(newexpr)
        newexpr = f(TR2i(e))
    if newexpr.has(TrigonometricFunction):
        newexpr = TR2i(newexpr)

    # can we ever generate an I where there was none previously?
    if not (newexpr.has(I) and not expr.has(I)):
        expr = newexpr
    return expr

#-------------------- the old trigsimp routines ---------------------

def trigsimp_old(expr, *, first=True, **opts):
    """
    Reduces expression by using known trig identities.

    Notes
    =====

    deep:
    - Apply trigsimp inside all objects with arguments

    recursive:
    - Use common subexpression elimination (cse()) and apply
    trigsimp recursively (this is quite expensive if the
    expression is large)

    method:
    - Determine the method to use. Valid choices are 'matching' (default),
    'groebner', 'combined', 'fu' and 'futrig'. If 'matching', simplify the
    expression recursively by pattern matching. If 'groebner', apply an
    experimental groebner basis algorithm. In this case further options
    are forwarded to ``trigsimp_groebner``, please refer to its docstring.
    If 'combined', first run the groebner basis algorithm with small
    default parameters, then run the 'matching' algorithm. 'fu' runs the
    collection of trigonometric transformations described by Fu, et al.
    (see the `fu` docstring) while `futrig` runs a subset of Fu-transforms
    that mimic the behavior of `trigsimp`.

    compare:
    - show input and output from `trigsimp` and `futrig` when different,
    but returns the `trigsimp` value.

    Examples
    ========

    >>> from sympy import trigsimp, sin, cos, log, cot
    >>> from sympy.abc import x
    >>> e = 2*sin(x)**2 + 2*cos(x)**2
    >>> trigsimp(e, old=True)
    2
    >>> trigsimp(log(e), old=True)
    log(2*sin(x)**2 + 2*cos(x)**2)
    >>> trigsimp(log(e), deep=True, old=True)
    log(2)

    Using `method="groebner"` (or `"combined"`) can sometimes lead to a lot
    more simplification:

    >>> e = (-sin(x) + 1)/cos(x) + cos(x)/(-sin(x) + 1)
    >>> trigsimp(e, old=True)
    (1 - sin(x))/cos(x) + cos(x)/(1 - sin(x))
    >>> trigsimp(e, method="groebner", old=True)
    2/cos(x)

    >>> trigsimp(1/cot(x)**2, compare=True, old=True)
          futrig: tan(x)**2
    cot(x)**(-2)

    """
    old = expr
    if first:
        if not expr.has(*_trigs):
            return expr

        trigsyms = set().union(*[t.free_symbols for t in expr.atoms(*_trigs)])
        if len(trigsyms) > 1:
            from sympy.simplify.simplify import separatevars

            d = separatevars(expr)
            if d.is_Mul:
                d = separatevars(d, dict=True) or d
            if isinstance(d, dict):
                expr = 1
                for v in d.values():
                    # remove hollow factoring
                    was = v
                    v = expand_mul(v)
                    opts['first'] = False
                    vnew = trigsimp(v, **opts)
                    if vnew == v:
                        vnew = was
                    expr *= vnew
                old = expr
            else:
                if d.is_Add:
                    for s in trigsyms:
                        r, e = expr.as_independent(s)
                        if r:
                            opts['first'] = False
                            expr = r + trigsimp(e, **opts)
                            if not expr.is_Add:
                                break
                    old = expr

    recursive = opts.pop('recursive', False)
    deep = opts.pop('deep', False)
    method = opts.pop('method', 'matching')

    def groebnersimp(ex, deep, **opts):
        def traverse(e):
            if e.is_Atom:
                return e
            args = [traverse(x) for x in e.args]
            if e.is_Function or e.is_Pow:
                args = [trigsimp_groebner(x, **opts) for x in args]
            return e.func(*args)
        if deep:
            ex = traverse(ex)
        return trigsimp_groebner(ex, **opts)

    trigsimpfunc = {
        'matching': (lambda x, d: _trigsimp(x, d)),
        'groebner': (lambda x, d: groebnersimp(x, d, **opts)),
        'combined': (lambda x, d: _trigsimp(groebnersimp(x,
                                       d, polynomial=True, hints=[2, tan]),
                                   d))
                   }[method]

    if recursive:
        w, g = cse(expr)
        g = trigsimpfunc(g[0], deep)

        for sub in reversed(w):
            g = g.subs(sub[0], sub[1])
            g = trigsimpfunc(g, deep)
        result = g
    else:
        result = trigsimpfunc(expr, deep)

    if opts.get('compare', False):
        f = futrig(old)
        if f != result:
            print('\tfutrig:', f)

    return result


def _dotrig(a, b):
    """Helper to tell whether ``a`` and ``b`` have the same sorts
    of symbols in them -- no need to test hyperbolic patterns against
    expressions that have no hyperbolics in them."""
    return a.func == b.func and (
        a.has(TrigonometricFunction) and b.has(TrigonometricFunction) or
        a.has(HyperbolicFunction) and b.has(HyperbolicFunction))


_trigpat = None
def _trigpats():
    global _trigpat
    a, b, c = symbols('a b c', cls=Wild)
    d = Wild('d', commutative=False)

    # for the simplifications like sinh/cosh -> tanh:
    # DO NOT REORDER THE FIRST 14 since these are assumed to be in this
    # order in _match_div_rewrite.
    matchers_division = (
        (a*sin(b)**c/cos(b)**c, a*tan(b)**c, sin(b), cos(b)),
        (a*tan(b)**c*cos(b)**c, a*sin(b)**c, sin(b), cos(b)),
        (a*cot(b)**c*sin(b)**c, a*cos(b)**c, sin(b), cos(b)),
        (a*tan(b)**c/sin(b)**c, a/cos(b)**c, sin(b), cos(b)),
        (a*cot(b)**c/cos(b)**c, a/sin(b)**c, sin(b), cos(b)),
        (a*cot(b)**c*tan(b)**c, a, sin(b), cos(b)),
        (a*(cos(b) + 1)**c*(cos(b) - 1)**c,
            a*(-sin(b)**2)**c, cos(b) + 1, cos(b) - 1),
        (a*(sin(b) + 1)**c*(sin(b) - 1)**c,
            a*(-cos(b)**2)**c, sin(b) + 1, sin(b) - 1),

        (a*sinh(b)**c/cosh(b)**c, a*tanh(b)**c, S.One, S.One),
        (a*tanh(b)**c*cosh(b)**c, a*sinh(b)**c, S.One, S.One),
        (a*coth(b)**c*sinh(b)**c, a*cosh(b)**c, S.One, S.One),
        (a*tanh(b)**c/sinh(b)**c, a/cosh(b)**c, S.One, S.One),
        (a*coth(b)**c/cosh(b)**c, a/sinh(b)**c, S.One, S.One),
        (a*coth(b)**c*tanh(b)**c, a, S.One, S.One),

        (c*(tanh(a) + tanh(b))/(1 + tanh(a)*tanh(b)),
            tanh(a + b)*c, S.One, S.One),
    )

    matchers_add = (
        (c*sin(a)*cos(b) + c*cos(a)*sin(b) + d, sin(a + b)*c + d),
        (c*cos(a)*cos(b) - c*sin(a)*sin(b) + d, cos(a + b)*c + d),
        (c*sin(a)*cos(b) - c*cos(a)*sin(b) + d, sin(a - b)*c + d),
        (c*cos(a)*cos(b) + c*sin(a)*sin(b) + d, cos(a - b)*c + d),
        (c*sinh(a)*cosh(b) + c*sinh(b)*cosh(a) + d, sinh(a + b)*c + d),
        (c*cosh(a)*cosh(b) + c*sinh(a)*sinh(b) + d, cosh(a + b)*c + d),
    )

    # for cos(x)**2 + sin(x)**2 -> 1
    matchers_identity = (
        (a*sin(b)**2, a - a*cos(b)**2),
        (a*tan(b)**2, a*(1/cos(b))**2 - a),
        (a*cot(b)**2, a*(1/sin(b))**2 - a),
        (a*sin(b + c), a*(sin(b)*cos(c) + sin(c)*cos(b))),
        (a*cos(b + c), a*(cos(b)*cos(c) - sin(b)*sin(c))),
        (a*tan(b + c), a*((tan(b) + tan(c))/(1 - tan(b)*tan(c)))),

        (a*sinh(b)**2, a*cosh(b)**2 - a),
        (a*tanh(b)**2, a - a*(1/cosh(b))**2),
        (a*coth(b)**2, a + a*(1/sinh(b))**2),
        (a*sinh(b + c), a*(sinh(b)*cosh(c) + sinh(c)*cosh(b))),
        (a*cosh(b + c), a*(cosh(b)*cosh(c) + sinh(b)*sinh(c))),
        (a*tanh(b + c), a*((tanh(b) + tanh(c))/(1 + tanh(b)*tanh(c)))),

    )

    # Reduce any lingering artifacts, such as sin(x)**2 changing
    # to 1-cos(x)**2 when sin(x)**2 was "simpler"
    artifacts = (
        (a - a*cos(b)**2 + c, a*sin(b)**2 + c, cos),
        (a - a*(1/cos(b))**2 + c, -a*tan(b)**2 + c, cos),
        (a - a*(1/sin(b))**2 + c, -a*cot(b)**2 + c, sin),

        (a - a*cosh(b)**2 + c, -a*sinh(b)**2 + c, cosh),
        (a - a*(1/cosh(b))**2 + c, a*tanh(b)**2 + c, cosh),
        (a + a*(1/sinh(b))**2 + c, a*coth(b)**2 + c, sinh),

        # same as above but with noncommutative prefactor
        (a*d - a*d*cos(b)**2 + c, a*d*sin(b)**2 + c, cos),
        (a*d - a*d*(1/cos(b))**2 + c, -a*d*tan(b)**2 + c, cos),
        (a*d - a*d*(1/sin(b))**2 + c, -a*d*cot(b)**2 + c, sin),

        (a*d - a*d*cosh(b)**2 + c, -a*d*sinh(b)**2 + c, cosh),
        (a*d - a*d*(1/cosh(b))**2 + c, a*d*tanh(b)**2 + c, cosh),
        (a*d + a*d*(1/sinh(b))**2 + c, a*d*coth(b)**2 + c, sinh),
    )

    _trigpat = (a, b, c, d, matchers_division, matchers_add,
        matchers_identity, artifacts)
    return _trigpat


def _replace_mul_fpowxgpow(expr, f, g, rexp, h, rexph):
    """Helper for _match_div_rewrite.

    Replace f(b_)**c_*g(b_)**(rexp(c_)) with h(b)**rexph(c) if f(b_)
    and g(b_) are both positive or if c_ is an integer.
    """
    # assert expr.is_Mul and expr.is_commutative and f != g
    fargs = defaultdict(int)
    gargs = defaultdict(int)
    args = []
    for x in expr.args:
        if x.is_Pow or x.func in (f, g):
            b, e = x.as_base_exp()
            if b.is_positive or e.is_integer:
                if b.func == f:
                    fargs[b.args[0]] += e
                    continue
                elif b.func == g:
                    gargs[b.args[0]] += e
                    continue
        args.append(x)
    common = set(fargs) & set(gargs)
    hit = False
    while common:
        key = common.pop()
        fe = fargs.pop(key)
        ge = gargs.pop(key)
        if fe == rexp(ge):
            args.append(h(key)**rexph(fe))
            hit = True
        else:
            fargs[key] = fe
            gargs[key] = ge
    if not hit:
        return expr
    while fargs:
        key, e = fargs.popitem()
        args.append(f(key)**e)
    while gargs:
        key, e = gargs.popitem()
        args.append(g(key)**e)
    return Mul(*args)


_idn = lambda x: x
_midn = lambda x: -x
_one = lambda x: S.One

def _match_div_rewrite(expr, i):
    """helper for __trigsimp"""
    if i == 0:
        expr = _replace_mul_fpowxgpow(expr, sin, cos,
            _midn, tan, _idn)
    elif i == 1:
        expr = _replace_mul_fpowxgpow(expr, tan, cos,
            _idn, sin, _idn)
    elif i == 2:
        expr = _replace_mul_fpowxgpow(expr, cot, sin,
            _idn, cos, _idn)
    elif i == 3:
        expr = _replace_mul_fpowxgpow(expr, tan, sin,
            _midn, cos, _midn)
    elif i == 4:
        expr = _replace_mul_fpowxgpow(expr, cot, cos,
            _midn, sin, _midn)
    elif i == 5:
        expr = _replace_mul_fpowxgpow(expr, cot, tan,
            _idn, _one, _idn)
    # i in (6, 7) is skipped
    elif i == 8:
        expr = _replace_mul_fpowxgpow(expr, sinh, cosh,
            _midn, tanh, _idn)
    elif i == 9:
        expr = _replace_mul_fpowxgpow(expr, tanh, cosh,
            _idn, sinh, _idn)
    elif i == 10:
        expr = _replace_mul_fpowxgpow(expr, coth, sinh,
            _idn, cosh, _idn)
    elif i == 11:
        expr = _replace_mul_fpowxgpow(expr, tanh, sinh,
            _midn, cosh, _midn)
    elif i == 12:
        expr = _replace_mul_fpowxgpow(expr, coth, cosh,
            _midn, sinh, _midn)
    elif i == 13:
        expr = _replace_mul_fpowxgpow(expr, coth, tanh,
            _idn, _one, _idn)
    else:
        return None
    return expr


def _trigsimp(expr, deep=False):
    # protect the cache from non-trig patterns; we only allow
    # trig patterns to enter the cache
    if expr.has(*_trigs):
        return __trigsimp(expr, deep)
    return expr


@cacheit
def __trigsimp(expr, deep=False):
    """recursive helper for trigsimp"""
    from sympy.simplify.fu import TR10i

    if _trigpat is None:
        _trigpats()
    a, b, c, d, matchers_division, matchers_add, \
    matchers_identity, artifacts = _trigpat

    if expr.is_Mul:
        # do some simplifications like sin/cos -> tan:
        if not expr.is_commutative:
            com, nc = expr.args_cnc()
            expr = _trigsimp(Mul._from_args(com), deep)*Mul._from_args(nc)
        else:
            for i, (pattern, simp, ok1, ok2) in enumerate(matchers_division):
                if not _dotrig(expr, pattern):
                    continue

                newexpr = _match_div_rewrite(expr, i)
                if newexpr is not None:
                    if newexpr != expr:
                        expr = newexpr
                        break
                    else:
                        continue

                # use SymPy matching instead
                res = expr.match(pattern)
                if res and res.get(c, 0):
                    if not res[c].is_integer:
                        ok = ok1.subs(res)
                        if not ok.is_positive:
                            continue
                        ok = ok2.subs(res)
                        if not ok.is_positive:
                            continue
                    # if "a" contains any of trig or hyperbolic funcs with
                    # argument "b" then skip the simplification
                    if any(w.args[0] == res[b] for w in res[a].atoms(
                            TrigonometricFunction, HyperbolicFunction)):
                        continue
                    # simplify and finish:
                    expr = simp.subs(res)
                    break  # process below

    if expr.is_Add:
        args = []
        for term in expr.args:
            if not term.is_commutative:
                com, nc = term.args_cnc()
                nc = Mul._from_args(nc)
                term = Mul._from_args(com)
            else:
                nc = S.One
            term = _trigsimp(term, deep)
            for pattern, result in matchers_identity:
                res = term.match(pattern)
                if res is not None:
                    term = result.subs(res)
                    break
            args.append(term*nc)
        if args != expr.args:
            expr = Add(*args)
            expr = min(expr, expand(expr), key=count_ops)
        if expr.is_Add:
            for pattern, result in matchers_add:
                if not _dotrig(expr, pattern):
                    continue
                expr = TR10i(expr)
                if expr.has(HyperbolicFunction):
                    res = expr.match(pattern)
                    # if "d" contains any trig or hyperbolic funcs with
                    # argument "a" or "b" then skip the simplification;
                    # this isn't perfect -- see tests
                    if res is None or not (a in res and b in res) or any(
                        w.args[0] in (res[a], res[b]) for w in res[d].atoms(
                            TrigonometricFunction, HyperbolicFunction)):
                        continue
                    expr = result.subs(res)
                    break

        # Reduce any lingering artifacts, such as sin(x)**2 changing
        # to 1 - cos(x)**2 when sin(x)**2 was "simpler"
        for pattern, result, ex in artifacts:
            if not _dotrig(expr, pattern):
                continue
            # Substitute a new wild that excludes some function(s)
            # to help influence a better match. This is because
            # sometimes, for example, 'a' would match sec(x)**2
            a_t = Wild('a', exclude=[ex])
            pattern = pattern.subs(a, a_t)
            result = result.subs(a, a_t)

            m = expr.match(pattern)
            was = None
            while m and was != expr:
                was = expr
                if m[a_t] == 0 or \
                        -m[a_t] in m[c].args or m[a_t] + m[c] == 0:
                    break
                if d in m and m[a_t]*m[d] + m[c] == 0:
                    break
                expr = result.subs(m)
                m = expr.match(pattern)
                m.setdefault(c, S.Zero)

    elif expr.is_Mul or expr.is_Pow or deep and expr.args:
        expr = expr.func(*[_trigsimp(a, deep) for a in expr.args])

    try:
        if not expr.has(*_trigs):
            raise TypeError
        e = expr.atoms(exp)
        new = expr.rewrite(exp, deep=deep)
        if new == e:
            raise TypeError
        fnew = factor(new)
        if fnew != new:
            new = min([new, factor(new)], key=count_ops)
        # if all exp that were introduced disappeared then accept it
        if not (new.atoms(exp) - e):
            expr = new
    except TypeError:
        pass

    return expr
#------------------- end of old trigsimp routines --------------------


def futrig(e, *, hyper=True, **kwargs):
    """Return simplified ``e`` using Fu-like transformations.
    This is not the "Fu" algorithm. This is called by default
    from ``trigsimp``. By default, hyperbolics subexpressions
    will be simplified, but this can be disabled by setting
    ``hyper=False``.

    Examples
    ========

    >>> from sympy import trigsimp, tan, sinh, tanh
    >>> from sympy.simplify.trigsimp import futrig
    >>> from sympy.abc import x
    >>> trigsimp(1/tan(x)**2)
    tan(x)**(-2)

    >>> futrig(sinh(x)/tanh(x))
    cosh(x)

    """
    from sympy.simplify.fu import hyper_as_trig

    e = sympify(e)

    if not isinstance(e, Basic):
        return e

    if not e.args:
        return e

    old = e
    e = bottom_up(e, _futrig)

    if hyper and e.has(HyperbolicFunction):
        e, f = hyper_as_trig(e)
        e = f(bottom_up(e, _futrig))

    if e != old and e.is_Mul and e.args[0].is_Rational:
        # redistribute leading coeff on 2-arg Add
        e = Mul(*e.as_coeff_Mul())
    return e


def _futrig(e):
    """Helper for futrig."""
    from sympy.simplify.fu import (
        TR1, TR2, TR3, TR2i, TR10, L, TR10i,
        TR8, TR6, TR15, TR16, TR111, TR5, TRmorrie, TR11, _TR11, TR14, TR22,
        TR12)

    if not e.has(TrigonometricFunction):
        return e

    if e.is_Mul:
        coeff, e = e.as_independent(TrigonometricFunction)
    else:
        coeff = None

    Lops = lambda x: (L(x), x.count_ops(), _nodes(x), len(x.args), x.is_Add)
    trigs = lambda x: x.has(TrigonometricFunction)

    tree = [identity,
        (
        TR3,  # canonical angles
        TR1,  # sec-csc -> cos-sin
        TR12,  # expand tan of sum
        lambda x: _eapply(factor, x, trigs),
        TR2,  # tan-cot -> sin-cos
        [identity, lambda x: _eapply(_mexpand, x, trigs)],
        TR2i,  # sin-cos ratio -> tan
        lambda x: _eapply(lambda i: factor(i.normal()), x, trigs),
        TR14,  # factored identities
        TR5,  # sin-pow -> cos_pow
        TR10,  # sin-cos of sums -> sin-cos prod
        TR11, _TR11, TR6, # reduce double angles and rewrite cos pows
        lambda x: _eapply(factor, x, trigs),
        TR14,  # factored powers of identities
        [identity, lambda x: _eapply(_mexpand, x, trigs)],
        TR10i,  # sin-cos products > sin-cos of sums
        TRmorrie,
        [identity, TR8],  # sin-cos products -> sin-cos of sums
        [identity, lambda x: TR2i(TR2(x))],  # tan -> sin-cos -> tan
        [
            lambda x: _eapply(expand_mul, TR5(x), trigs),
            lambda x: _eapply(
                expand_mul, TR15(x), trigs)], # pos/neg powers of sin
        [
            lambda x:  _eapply(expand_mul, TR6(x), trigs),
            lambda x:  _eapply(
                expand_mul, TR16(x), trigs)], # pos/neg powers of cos
        TR111,  # tan, sin, cos to neg power -> cot, csc, sec
        [identity, TR2i],  # sin-cos ratio to tan
        [identity, lambda x: _eapply(
            expand_mul, TR22(x), trigs)],  # tan-cot to sec-csc
        TR1, TR2, TR2i,
        [identity, lambda x: _eapply(
            factor_terms, TR12(x), trigs)],  # expand tan of sum
        )]
    e = greedy(tree, objective=Lops)(e)

    if coeff is not None:
        e = coeff * e

    return e


def _is_Expr(e):
    """_eapply helper to tell whether ``e`` and all its args
    are Exprs."""
    if isinstance(e, Derivative):
        return _is_Expr(e.expr)
    if not isinstance(e, Expr):
        return False
    return all(_is_Expr(i) for i in e.args)


def _eapply(func, e, cond=None):
    """Apply ``func`` to ``e`` if all args are Exprs else only
    apply it to those args that *are* Exprs."""
    if not isinstance(e, Expr):
        return e
    if _is_Expr(e) or not e.args:
        return func(e)
    return e.func(*[
        _eapply(func, ei) if (cond is None or cond(ei)) else ei
        for ei in e.args])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\tools\datetimes.py ===
from __future__ import annotations

from collections import abc
from datetime import date
from functools import partial
from itertools import islice
from typing import (
    TYPE_CHECKING,
    Callable,
    TypedDict,
    Union,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._config import using_string_dtype

from pandas._libs import (
    lib,
    tslib,
)
from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    Timedelta,
    Timestamp,
    astype_overflowsafe,
    is_supported_dtype,
    timezones as libtimezones,
)
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas._libs.tslibs.parsing import (
    DateParseError,
    guess_datetime_format,
)
from pandas._libs.tslibs.strptime import array_strptime
from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    DateTimeErrorChoices,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    ensure_object,
    is_float,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_numeric_dtype,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    DatetimeTZDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

from pandas.arrays import (
    DatetimeArray,
    IntegerArray,
    NumpyExtensionArray,
)
from pandas.core.algorithms import unique
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.datetimes import (
    maybe_convert_dtype,
    objects_to_datetime64,
    tz_to_dtype,
)
from pandas.core.construction import extract_array
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._libs.tslibs.nattype import NaTType
    from pandas._libs.tslibs.timedeltas import UnitChoices

    from pandas import (
        DataFrame,
        Series,
    )

# ---------------------------------------------------------------------
# types used in annotations

ArrayConvertible = Union[list, tuple, AnyArrayLike]
Scalar = Union[float, str]
DatetimeScalar = Union[Scalar, date, np.datetime64]

DatetimeScalarOrArrayConvertible = Union[DatetimeScalar, ArrayConvertible]

DatetimeDictArg = Union[list[Scalar], tuple[Scalar, ...], AnyArrayLike]


class YearMonthDayDict(TypedDict, total=True):
    year: DatetimeDictArg
    month: DatetimeDictArg
    day: DatetimeDictArg


class FulldatetimeDict(YearMonthDayDict, total=False):
    hour: DatetimeDictArg
    hours: DatetimeDictArg
    minute: DatetimeDictArg
    minutes: DatetimeDictArg
    second: DatetimeDictArg
    seconds: DatetimeDictArg
    ms: DatetimeDictArg
    us: DatetimeDictArg
    ns: DatetimeDictArg


DictConvertible = Union[FulldatetimeDict, "DataFrame"]
start_caching_at = 50


# ---------------------------------------------------------------------


def _guess_datetime_format_for_array(arr, dayfirst: bool | None = False) -> str | None:
    # Try to guess the format based on the first non-NaN element, return None if can't
    if (first_non_null := tslib.first_non_null(arr)) != -1:
        if type(first_non_nan_element := arr[first_non_null]) is str:  # noqa: E721
            # GH#32264 np.str_ object
            guessed_format = guess_datetime_format(
                first_non_nan_element, dayfirst=dayfirst
            )
            if guessed_format is not None:
                return guessed_format
            # If there are multiple non-null elements, warn about
            # how parsing might not be consistent
            if tslib.first_non_null(arr[first_non_null + 1 :]) != -1:
                warnings.warn(
                    "Could not infer format, so each element will be parsed "
                    "individually, falling back to `dateutil`. To ensure parsing is "
                    "consistent and as-expected, please specify a format.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
    return None


def should_cache(
    arg: ArrayConvertible, unique_share: float = 0.7, check_count: int | None = None
) -> bool:
    """
    Decides whether to do caching.

    If the percent of unique elements among `check_count` elements less
    than `unique_share * 100` then we can do caching.

    Parameters
    ----------
    arg: listlike, tuple, 1-d array, Series
    unique_share: float, default=0.7, optional
        0 < unique_share < 1
    check_count: int, optional
        0 <= check_count <= len(arg)

    Returns
    -------
    do_caching: bool

    Notes
    -----
    By default for a sequence of less than 50 items in size, we don't do
    caching; for the number of elements less than 5000, we take ten percent of
    all elements to check for a uniqueness share; if the sequence size is more
    than 5000, then we check only the first 500 elements.
    All constants were chosen empirically by.
    """
    do_caching = True

    # default realization
    if check_count is None:
        # in this case, the gain from caching is negligible
        if len(arg) <= start_caching_at:
            return False

        if len(arg) <= 5000:
            check_count = len(arg) // 10
        else:
            check_count = 500
    else:
        assert (
            0 <= check_count <= len(arg)
        ), "check_count must be in next bounds: [0; len(arg)]"
        if check_count == 0:
            return False

    assert 0 < unique_share < 1, "unique_share must be in next bounds: (0; 1)"

    try:
        # We can't cache if the items are not hashable.
        unique_elements = set(islice(arg, check_count))
    except TypeError:
        return False
    if len(unique_elements) > check_count * unique_share:
        do_caching = False
    return do_caching


def _maybe_cache(
    arg: ArrayConvertible,
    format: str | None,
    cache: bool,
    convert_listlike: Callable,
) -> Series:
    """
    Create a cache of unique dates from an array of dates

    Parameters
    ----------
    arg : listlike, tuple, 1-d array, Series
    format : string
        Strftime format to parse time
    cache : bool
        True attempts to create a cache of converted values
    convert_listlike : function
        Conversion function to apply on dates

    Returns
    -------
    cache_array : Series
        Cache of converted, unique dates. Can be empty
    """
    from pandas import Series

    cache_array = Series(dtype=object)

    if cache:
        # Perform a quicker unique check
        if not should_cache(arg):
            return cache_array

        if not isinstance(arg, (np.ndarray, ExtensionArray, Index, ABCSeries)):
            arg = np.array(arg)

        unique_dates = unique(arg)
        if len(unique_dates) < len(arg):
            cache_dates = convert_listlike(unique_dates, format)
            # GH#45319
            try:
                cache_array = Series(cache_dates, index=unique_dates, copy=False)
            except OutOfBoundsDatetime:
                return cache_array
            # GH#39882 and GH#35888 in case of None and NaT we get duplicates
            if not cache_array.index.is_unique:
                cache_array = cache_array[~cache_array.index.duplicated()]
    return cache_array


def _box_as_indexlike(
    dt_array: ArrayLike, utc: bool = False, name: Hashable | None = None
) -> Index:
    """
    Properly boxes the ndarray of datetimes to DatetimeIndex
    if it is possible or to generic Index instead

    Parameters
    ----------
    dt_array: 1-d array
        Array of datetimes to be wrapped in an Index.
    utc : bool
        Whether to convert/localize timestamps to UTC.
    name : string, default None
        Name for a resulting index

    Returns
    -------
    result : datetime of converted dates
        - DatetimeIndex if convertible to sole datetime64 type
        - general Index otherwise
    """

    if lib.is_np_dtype(dt_array.dtype, "M"):
        tz = "utc" if utc else None
        return DatetimeIndex(dt_array, tz=tz, name=name)
    return Index(dt_array, name=name, dtype=dt_array.dtype)


def _convert_and_box_cache(
    arg: DatetimeScalarOrArrayConvertible,
    cache_array: Series,
    name: Hashable | None = None,
) -> Index:
    """
    Convert array of dates with a cache and wrap the result in an Index.

    Parameters
    ----------
    arg : integer, float, string, datetime, list, tuple, 1-d array, Series
    cache_array : Series
        Cache of converted, unique dates
    name : string, default None
        Name for a DatetimeIndex

    Returns
    -------
    result : Index-like of converted dates
    """
    from pandas import Series

    result = Series(arg, dtype=cache_array.index.dtype).map(cache_array)
    return _box_as_indexlike(result._values, utc=False, name=name)


def _convert_listlike_datetimes(
    arg,
    format: str | None,
    name: Hashable | None = None,
    utc: bool = False,
    unit: str | None = None,
    errors: DateTimeErrorChoices = "raise",
    dayfirst: bool | None = None,
    yearfirst: bool | None = None,
    exact: bool = True,
):
    """
    Helper function for to_datetime. Performs the conversions of 1D listlike
    of dates

    Parameters
    ----------
    arg : list, tuple, ndarray, Series, Index
        date to be parsed
    name : object
        None or string for the Index name
    utc : bool
        Whether to convert/localize timestamps to UTC.
    unit : str
        None or string of the frequency of the passed data
    errors : str
        error handing behaviors from to_datetime, 'raise', 'coerce', 'ignore'
    dayfirst : bool
        dayfirst parsing behavior from to_datetime
    yearfirst : bool
        yearfirst parsing behavior from to_datetime
    exact : bool, default True
        exact format matching behavior from to_datetime

    Returns
    -------
    Index-like of parsed dates
    """
    if isinstance(arg, (list, tuple)):
        arg = np.array(arg, dtype="O")
    elif isinstance(arg, NumpyExtensionArray):
        arg = np.array(arg)

    arg_dtype = getattr(arg, "dtype", None)
    # these are shortcutable
    tz = "utc" if utc else None
    if isinstance(arg_dtype, DatetimeTZDtype):
        if not isinstance(arg, (DatetimeArray, DatetimeIndex)):
            return DatetimeIndex(arg, tz=tz, name=name)
        if utc:
            arg = arg.tz_convert(None).tz_localize("utc")
        return arg

    elif isinstance(arg_dtype, ArrowDtype) and arg_dtype.type is Timestamp:
        # TODO: Combine with above if DTI/DTA supports Arrow timestamps
        if utc:
            # pyarrow uses UTC, not lowercase utc
            if isinstance(arg, Index):
                arg_array = cast(ArrowExtensionArray, arg.array)
                if arg_dtype.pyarrow_dtype.tz is not None:
                    arg_array = arg_array._dt_tz_convert("UTC")
                else:
                    arg_array = arg_array._dt_tz_localize("UTC")
                arg = Index(arg_array)
            else:
                # ArrowExtensionArray
                if arg_dtype.pyarrow_dtype.tz is not None:
                    arg = arg._dt_tz_convert("UTC")
                else:
                    arg = arg._dt_tz_localize("UTC")
        return arg

    elif lib.is_np_dtype(arg_dtype, "M"):
        if not is_supported_dtype(arg_dtype):
            # We go to closest supported reso, i.e. "s"
            arg = astype_overflowsafe(
                # TODO: looks like we incorrectly raise with errors=="ignore"
                np.asarray(arg),
                np.dtype("M8[s]"),
                is_coerce=errors == "coerce",
            )

        if not isinstance(arg, (DatetimeArray, DatetimeIndex)):
            return DatetimeIndex(arg, tz=tz, name=name)
        elif utc:
            # DatetimeArray, DatetimeIndex
            return arg.tz_localize("utc")

        return arg

    elif unit is not None:
        if format is not None:
            raise ValueError("cannot specify both format and unit")
        return _to_datetime_with_unit(arg, unit, name, utc, errors)
    elif getattr(arg, "ndim", 1) > 1:
        raise TypeError(
            "arg must be a string, datetime, list, tuple, 1-d array, or Series"
        )

    # warn if passing timedelta64, raise for PeriodDtype
    # NB: this must come after unit transformation
    try:
        arg, _ = maybe_convert_dtype(arg, copy=False, tz=libtimezones.maybe_get_tz(tz))
    except TypeError:
        if errors == "coerce":
            npvalues = np.array(["NaT"], dtype="datetime64[ns]").repeat(len(arg))
            return DatetimeIndex(npvalues, name=name)
        elif errors == "ignore":
            idx = Index(arg, name=name)
            return idx
        raise

    arg = ensure_object(arg)

    if format is None:
        format = _guess_datetime_format_for_array(arg, dayfirst=dayfirst)

    # `format` could be inferred, or user didn't ask for mixed-format parsing.
    if format is not None and format != "mixed":
        return _array_strptime_with_fallback(arg, name, utc, format, exact, errors)

    result, tz_parsed = objects_to_datetime64(
        arg,
        dayfirst=dayfirst,
        yearfirst=yearfirst,
        utc=utc,
        errors=errors,
        allow_object=True,
    )

    if tz_parsed is not None:
        # We can take a shortcut since the datetime64 numpy array
        # is in UTC
        out_unit = np.datetime_data(result.dtype)[0]
        dtype = cast(DatetimeTZDtype, tz_to_dtype(tz_parsed, out_unit))
        dt64_values = result.view(f"M8[{dtype.unit}]")
        dta = DatetimeArray._simple_new(dt64_values, dtype=dtype)
        return DatetimeIndex._simple_new(dta, name=name)

    return _box_as_indexlike(result, utc=utc, name=name)


def _array_strptime_with_fallback(
    arg,
    name,
    utc: bool,
    fmt: str,
    exact: bool,
    errors: str,
) -> Index:
    """
    Call array_strptime, with fallback behavior depending on 'errors'.
    """
    result, tz_out = array_strptime(arg, fmt, exact=exact, errors=errors, utc=utc)
    if tz_out is not None:
        unit = np.datetime_data(result.dtype)[0]
        dtype = DatetimeTZDtype(tz=tz_out, unit=unit)
        dta = DatetimeArray._simple_new(result, dtype=dtype)
        if utc:
            dta = dta.tz_convert("UTC")
        return Index(dta, name=name)
    elif result.dtype != object and utc:
        unit = np.datetime_data(result.dtype)[0]
        res = Index(result, dtype=f"M8[{unit}, UTC]", name=name)
        return res
    elif using_string_dtype() and result.dtype == object:
        if lib.is_string_array(result):
            return Index(result, dtype="str", name=name)
    return Index(result, dtype=result.dtype, name=name)


def _to_datetime_with_unit(arg, unit, name, utc: bool, errors: str) -> Index:
    """
    to_datetime specalized to the case where a 'unit' is passed.
    """
    arg = extract_array(arg, extract_numpy=True)

    # GH#30050 pass an ndarray to tslib.array_with_unit_to_datetime
    # because it expects an ndarray argument
    if isinstance(arg, IntegerArray):
        arr = arg.astype(f"datetime64[{unit}]")
        tz_parsed = None
    else:
        arg = np.asarray(arg)

        if arg.dtype.kind in "iu":
            # Note we can't do "f" here because that could induce unwanted
            #  rounding GH#14156, GH#20445
            arr = arg.astype(f"datetime64[{unit}]", copy=False)
            try:
                arr = astype_overflowsafe(arr, np.dtype("M8[ns]"), copy=False)
            except OutOfBoundsDatetime:
                if errors == "raise":
                    raise
                arg = arg.astype(object)
                return _to_datetime_with_unit(arg, unit, name, utc, errors)
            tz_parsed = None

        elif arg.dtype.kind == "f":
            with np.errstate(over="raise"):
                try:
                    arr = cast_from_unit_vectorized(arg, unit=unit)
                except OutOfBoundsDatetime:
                    if errors != "raise":
                        return _to_datetime_with_unit(
                            arg.astype(object), unit, name, utc, errors
                        )
                    raise OutOfBoundsDatetime(
                        f"cannot convert input with unit '{unit}'"
                    )

            arr = arr.view("M8[ns]")
            tz_parsed = None
        else:
            arg = arg.astype(object, copy=False)
            arr, tz_parsed = tslib.array_with_unit_to_datetime(arg, unit, errors=errors)

    if errors == "ignore":
        # Index constructor _may_ infer to DatetimeIndex
        result = Index._with_infer(arr, name=name)
    else:
        result = DatetimeIndex(arr, name=name)

    if not isinstance(result, DatetimeIndex):
        return result

    # GH#23758: We may still need to localize the result with tz
    # GH#25546: Apply tz_parsed first (from arg), then tz (from caller)
    # result will be naive but in UTC
    result = result.tz_localize("UTC").tz_convert(tz_parsed)

    if utc:
        if result.tz is None:
            result = result.tz_localize("utc")
        else:
            result = result.tz_convert("utc")
    return result


def _adjust_to_origin(arg, origin, unit):
    """
    Helper function for to_datetime.
    Adjust input argument to the specified origin

    Parameters
    ----------
    arg : list, tuple, ndarray, Series, Index
        date to be adjusted
    origin : 'julian' or Timestamp
        origin offset for the arg
    unit : str
        passed unit from to_datetime, must be 'D'

    Returns
    -------
    ndarray or scalar of adjusted date(s)
    """
    if origin == "julian":
        original = arg
        j0 = Timestamp(0).to_julian_date()
        if unit != "D":
            raise ValueError("unit must be 'D' for origin='julian'")
        try:
            arg = arg - j0
        except TypeError as err:
            raise ValueError(
                "incompatible 'arg' type for given 'origin'='julian'"
            ) from err

        # preemptively check this for a nice range
        j_max = Timestamp.max.to_julian_date() - j0
        j_min = Timestamp.min.to_julian_date() - j0
        if np.any(arg > j_max) or np.any(arg < j_min):
            raise OutOfBoundsDatetime(
                f"{original} is Out of Bounds for origin='julian'"
            )
    else:
        # arg must be numeric
        if not (
            (is_integer(arg) or is_float(arg)) or is_numeric_dtype(np.asarray(arg))
        ):
            raise ValueError(
                f"'{arg}' is not compatible with origin='{origin}'; "
                "it must be numeric with a unit specified"
            )

        # we are going to offset back to unix / epoch time
        try:
            offset = Timestamp(origin, unit=unit)
        except OutOfBoundsDatetime as err:
            raise OutOfBoundsDatetime(f"origin {origin} is Out of Bounds") from err
        except ValueError as err:
            raise ValueError(
                f"origin {origin} cannot be converted to a Timestamp"
            ) from err

        if offset.tz is not None:
            raise ValueError(f"origin offset {offset} must be tz-naive")
        td_offset = offset - Timestamp(0)

        # convert the offset to the unit of the arg
        # this should be lossless in terms of precision
        ioffset = td_offset // Timedelta(1, unit=unit)

        # scalars & ndarray-like can handle the addition
        if is_list_like(arg) and not isinstance(arg, (ABCSeries, Index, np.ndarray)):
            arg = np.asarray(arg)
        arg = arg + ioffset
    return arg


@overload
def to_datetime(
    arg: DatetimeScalar,
    errors: DateTimeErrorChoices = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    infer_datetime_format: bool = ...,
    origin=...,
    cache: bool = ...,
) -> Timestamp:
    ...


@overload
def to_datetime(
    arg: Series | DictConvertible,
    errors: DateTimeErrorChoices = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    infer_datetime_format: bool = ...,
    origin=...,
    cache: bool = ...,
) -> Series:
    ...


@overload
def to_datetime(
    arg: list | tuple | Index | ArrayLike,
    errors: DateTimeErrorChoices = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    infer_datetime_format: bool = ...,
    origin=...,
    cache: bool = ...,
) -> DatetimeIndex:
    ...


def to_datetime(
    arg: DatetimeScalarOrArrayConvertible | DictConvertible,
    errors: DateTimeErrorChoices = "raise",
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool = False,
    format: str | None = None,
    exact: bool | lib.NoDefault = lib.no_default,
    unit: str | None = None,
    infer_datetime_format: lib.NoDefault | bool = lib.no_default,
    origin: str = "unix",
    cache: bool = True,
) -> DatetimeIndex | Series | DatetimeScalar | NaTType | None:
    """
    Convert argument to datetime.

    This function converts a scalar, array-like, :class:`Series` or
    :class:`DataFrame`/dict-like to a pandas datetime object.

    Parameters
    ----------
    arg : int, float, str, datetime, list, tuple, 1-d array, Series, DataFrame/dict-like
        The object to convert to a datetime. If a :class:`DataFrame` is provided, the
        method expects minimally the following columns: :const:`"year"`,
        :const:`"month"`, :const:`"day"`. The column "year"
        must be specified in 4-digit format.
    errors : {'ignore', 'raise', 'coerce'}, default 'raise'
        - If :const:`'raise'`, then invalid parsing will raise an exception.
        - If :const:`'coerce'`, then invalid parsing will be set as :const:`NaT`.
        - If :const:`'ignore'`, then invalid parsing will return the input.
    dayfirst : bool, default False
        Specify a date parse order if `arg` is str or is list-like.
        If :const:`True`, parses dates with the day first, e.g. :const:`"10/11/12"`
        is parsed as :const:`2012-11-10`.

        .. warning::

            ``dayfirst=True`` is not strict, but will prefer to parse
            with day first.

    yearfirst : bool, default False
        Specify a date parse order if `arg` is str or is list-like.

        - If :const:`True` parses dates with the year first, e.g.
          :const:`"10/11/12"` is parsed as :const:`2010-11-12`.
        - If both `dayfirst` and `yearfirst` are :const:`True`, `yearfirst` is
          preceded (same as :mod:`dateutil`).

        .. warning::

            ``yearfirst=True`` is not strict, but will prefer to parse
            with year first.

    utc : bool, default False
        Control timezone-related parsing, localization and conversion.

        - If :const:`True`, the function *always* returns a timezone-aware
          UTC-localized :class:`Timestamp`, :class:`Series` or
          :class:`DatetimeIndex`. To do this, timezone-naive inputs are
          *localized* as UTC, while timezone-aware inputs are *converted* to UTC.

        - If :const:`False` (default), inputs will not be coerced to UTC.
          Timezone-naive inputs will remain naive, while timezone-aware ones
          will keep their time offsets. Limitations exist for mixed
          offsets (typically, daylight savings), see :ref:`Examples
          <to_datetime_tz_examples>` section for details.

        .. warning::

            In a future version of pandas, parsing datetimes with mixed time
            zones will raise an error unless `utc=True`.
            Please specify `utc=True` to opt in to the new behaviour
            and silence this warning. To create a `Series` with mixed offsets and
            `object` dtype, please use `apply` and `datetime.datetime.strptime`.

        See also: pandas general documentation about `timezone conversion and
        localization
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        #time-zone-handling>`_.

    format : str, default None
        The strftime to parse time, e.g. :const:`"%d/%m/%Y"`. See
        `strftime documentation
        <https://docs.python.org/3/library/datetime.html
        #strftime-and-strptime-behavior>`_ for more information on choices, though
        note that :const:`"%f"` will parse all the way up to nanoseconds.
        You can also pass:

        - "ISO8601", to parse any `ISO8601 <https://en.wikipedia.org/wiki/ISO_8601>`_
          time string (not necessarily in exactly the same format);
        - "mixed", to infer the format for each element individually. This is risky,
          and you should probably use it along with `dayfirst`.

        .. note::

            If a :class:`DataFrame` is passed, then `format` has no effect.

    exact : bool, default True
        Control how `format` is used:

        - If :const:`True`, require an exact `format` match.
        - If :const:`False`, allow the `format` to match anywhere in the target
          string.

        Cannot be used alongside ``format='ISO8601'`` or ``format='mixed'``.
    unit : str, default 'ns'
        The unit of the arg (D,s,ms,us,ns) denote the unit, which is an
        integer or float number. This will be based off the origin.
        Example, with ``unit='ms'`` and ``origin='unix'``, this would calculate
        the number of milliseconds to the unix epoch start.
    infer_datetime_format : bool, default False
        If :const:`True` and no `format` is given, attempt to infer the format
        of the datetime strings based on the first non-NaN element,
        and if it can be inferred, switch to a faster method of parsing them.
        In some cases this can increase the parsing speed by ~5-10x.

        .. deprecated:: 2.0.0
            A strict version of this argument is now the default, passing it has
            no effect.

    origin : scalar, default 'unix'
        Define the reference date. The numeric values would be parsed as number
        of units (defined by `unit`) since this reference date.

        - If :const:`'unix'` (or POSIX) time; origin is set to 1970-01-01.
        - If :const:`'julian'`, unit must be :const:`'D'`, and origin is set to
          beginning of Julian Calendar. Julian day number :const:`0` is assigned
          to the day starting at noon on January 1, 4713 BC.
        - If Timestamp convertible (Timestamp, dt.datetime, np.datetimt64 or date
          string), origin is set to Timestamp identified by origin.
        - If a float or integer, origin is the difference
          (in units determined by the ``unit`` argument) relative to 1970-01-01.
    cache : bool, default True
        If :const:`True`, use a cache of unique, converted dates to apply the
        datetime conversion. May produce significant speed-up when parsing
        duplicate date strings, especially ones with timezone offsets. The cache
        is only used when there are at least 50 values. The presence of
        out-of-bounds values will render the cache unusable and may slow down
        parsing.

    Returns
    -------
    datetime
        If parsing succeeded.
        Return type depends on input (types in parenthesis correspond to
        fallback in case of unsuccessful timezone or out-of-range timestamp
        parsing):

        - scalar: :class:`Timestamp` (or :class:`datetime.datetime`)
        - array-like: :class:`DatetimeIndex` (or :class:`Series` with
          :class:`object` dtype containing :class:`datetime.datetime`)
        - Series: :class:`Series` of :class:`datetime64` dtype (or
          :class:`Series` of :class:`object` dtype containing
          :class:`datetime.datetime`)
        - DataFrame: :class:`Series` of :class:`datetime64` dtype (or
          :class:`Series` of :class:`object` dtype containing
          :class:`datetime.datetime`)

    Raises
    ------
    ParserError
        When parsing a date from string fails.
    ValueError
        When another datetime conversion error happens. For example when one
        of 'year', 'month', day' columns is missing in a :class:`DataFrame`, or
        when a Timezone-aware :class:`datetime.datetime` is found in an array-like
        of mixed time offsets, and ``utc=False``.

    See Also
    --------
    DataFrame.astype : Cast argument to a specified dtype.
    to_timedelta : Convert argument to timedelta.
    convert_dtypes : Convert dtypes.

    Notes
    -----

    Many input types are supported, and lead to different output types:

    - **scalars** can be int, float, str, datetime object (from stdlib :mod:`datetime`
      module or :mod:`numpy`). They are converted to :class:`Timestamp` when
      possible, otherwise they are converted to :class:`datetime.datetime`.
      None/NaN/null scalars are converted to :const:`NaT`.

    - **array-like** can contain int, float, str, datetime objects. They are
      converted to :class:`DatetimeIndex` when possible, otherwise they are
      converted to :class:`Index` with :class:`object` dtype, containing
      :class:`datetime.datetime`. None/NaN/null entries are converted to
      :const:`NaT` in both cases.

    - **Series** are converted to :class:`Series` with :class:`datetime64`
      dtype when possible, otherwise they are converted to :class:`Series` with
      :class:`object` dtype, containing :class:`datetime.datetime`. None/NaN/null
      entries are converted to :const:`NaT` in both cases.

    - **DataFrame/dict-like** are converted to :class:`Series` with
      :class:`datetime64` dtype. For each row a datetime is created from assembling
      the various dataframe columns. Column keys can be common abbreviations
      like ['year', 'month', 'day', 'minute', 'second', 'ms', 'us', 'ns']) or
      plurals of the same.

    The following causes are responsible for :class:`datetime.datetime` objects
    being returned (possibly inside an :class:`Index` or a :class:`Series` with
    :class:`object` dtype) instead of a proper pandas designated type
    (:class:`Timestamp`, :class:`DatetimeIndex` or :class:`Series`
    with :class:`datetime64` dtype):

    - when any input element is before :const:`Timestamp.min` or after
      :const:`Timestamp.max`, see `timestamp limitations
      <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
      #timeseries-timestamp-limits>`_.

    - when ``utc=False`` (default) and the input is an array-like or
      :class:`Series` containing mixed naive/aware datetime, or aware with mixed
      time offsets. Note that this happens in the (quite frequent) situation when
      the timezone has a daylight savings policy. In that case you may wish to
      use ``utc=True``.

    Examples
    --------

    **Handling various input formats**

    Assembling a datetime from multiple columns of a :class:`DataFrame`. The keys
    can be common abbreviations like ['year', 'month', 'day', 'minute', 'second',
    'ms', 'us', 'ns']) or plurals of the same

    >>> df = pd.DataFrame({'year': [2015, 2016],
    ...                    'month': [2, 3],
    ...                    'day': [4, 5]})
    >>> pd.to_datetime(df)
    0   2015-02-04
    1   2016-03-05
    dtype: datetime64[ns]

    Using a unix epoch time

    >>> pd.to_datetime(1490195805, unit='s')
    Timestamp('2017-03-22 15:16:45')
    >>> pd.to_datetime(1490195805433502912, unit='ns')
    Timestamp('2017-03-22 15:16:45.433502912')

    .. warning:: For float arg, precision rounding might happen. To prevent
        unexpected behavior use a fixed-width exact type.

    Using a non-unix epoch origin

    >>> pd.to_datetime([1, 2, 3], unit='D',
    ...                origin=pd.Timestamp('1960-01-01'))
    DatetimeIndex(['1960-01-02', '1960-01-03', '1960-01-04'],
                  dtype='datetime64[ns]', freq=None)

    **Differences with strptime behavior**

    :const:`"%f"` will parse all the way up to nanoseconds.

    >>> pd.to_datetime('2018-10-26 12:00:00.0000000011',
    ...                format='%Y-%m-%d %H:%M:%S.%f')
    Timestamp('2018-10-26 12:00:00.000000001')

    **Non-convertible date/times**

    Passing ``errors='coerce'`` will force an out-of-bounds date to :const:`NaT`,
    in addition to forcing non-dates (or non-parseable dates) to :const:`NaT`.

    >>> pd.to_datetime('13000101', format='%Y%m%d', errors='coerce')
    NaT

    .. _to_datetime_tz_examples:

    **Timezones and time offsets**

    The default behaviour (``utc=False``) is as follows:

    - Timezone-naive inputs are converted to timezone-naive :class:`DatetimeIndex`:

    >>> pd.to_datetime(['2018-10-26 12:00:00', '2018-10-26 13:00:15'])
    DatetimeIndex(['2018-10-26 12:00:00', '2018-10-26 13:00:15'],
                  dtype='datetime64[ns]', freq=None)

    - Timezone-aware inputs *with constant time offset* are converted to
      timezone-aware :class:`DatetimeIndex`:

    >>> pd.to_datetime(['2018-10-26 12:00 -0500', '2018-10-26 13:00 -0500'])
    DatetimeIndex(['2018-10-26 12:00:00-05:00', '2018-10-26 13:00:00-05:00'],
                  dtype='datetime64[ns, UTC-05:00]', freq=None)

    - However, timezone-aware inputs *with mixed time offsets* (for example
      issued from a timezone with daylight savings, such as Europe/Paris)
      are **not successfully converted** to a :class:`DatetimeIndex`.
      Parsing datetimes with mixed time zones will show a warning unless
      `utc=True`. If you specify `utc=False` the warning below will be shown
      and a simple :class:`Index` containing :class:`datetime.datetime`
      objects will be returned:

    >>> pd.to_datetime(['2020-10-25 02:00 +0200',
    ...                 '2020-10-25 04:00 +0100'])  # doctest: +SKIP
    FutureWarning: In a future version of pandas, parsing datetimes with mixed
    time zones will raise an error unless `utc=True`. Please specify `utc=True`
    to opt in to the new behaviour and silence this warning. To create a `Series`
    with mixed offsets and `object` dtype, please use `apply` and
    `datetime.datetime.strptime`.
    Index([2020-10-25 02:00:00+02:00, 2020-10-25 04:00:00+01:00],
          dtype='object')

    - A mix of timezone-aware and timezone-naive inputs is also converted to
      a simple :class:`Index` containing :class:`datetime.datetime` objects:

    >>> from datetime import datetime
    >>> pd.to_datetime(["2020-01-01 01:00:00-01:00",
    ...                 datetime(2020, 1, 1, 3, 0)])  # doctest: +SKIP
    FutureWarning: In a future version of pandas, parsing datetimes with mixed
    time zones will raise an error unless `utc=True`. Please specify `utc=True`
    to opt in to the new behaviour and silence this warning. To create a `Series`
    with mixed offsets and `object` dtype, please use `apply` and
    `datetime.datetime.strptime`.
    Index([2020-01-01 01:00:00-01:00, 2020-01-01 03:00:00], dtype='object')

    |

    Setting ``utc=True`` solves most of the above issues:

    - Timezone-naive inputs are *localized* as UTC

    >>> pd.to_datetime(['2018-10-26 12:00', '2018-10-26 13:00'], utc=True)
    DatetimeIndex(['2018-10-26 12:00:00+00:00', '2018-10-26 13:00:00+00:00'],
                  dtype='datetime64[ns, UTC]', freq=None)

    - Timezone-aware inputs are *converted* to UTC (the output represents the
      exact same datetime, but viewed from the UTC time offset `+00:00`).

    >>> pd.to_datetime(['2018-10-26 12:00 -0530', '2018-10-26 12:00 -0500'],
    ...                utc=True)
    DatetimeIndex(['2018-10-26 17:30:00+00:00', '2018-10-26 17:00:00+00:00'],
                  dtype='datetime64[ns, UTC]', freq=None)

    - Inputs can contain both string or datetime, the above
      rules still apply

    >>> pd.to_datetime(['2018-10-26 12:00', datetime(2020, 1, 1, 18)], utc=True)
    DatetimeIndex(['2018-10-26 12:00:00+00:00', '2020-01-01 18:00:00+00:00'],
                  dtype='datetime64[ns, UTC]', freq=None)
    """
    if exact is not lib.no_default and format in {"mixed", "ISO8601"}:
        raise ValueError("Cannot use 'exact' when 'format' is 'mixed' or 'ISO8601'")
    if infer_datetime_format is not lib.no_default:
        warnings.warn(
            "The argument 'infer_datetime_format' is deprecated and will "
            "be removed in a future version. "
            "A strict version of it is now the default, see "
            "https://pandas.pydata.org/pdeps/0004-consistent-to-datetime-parsing.html. "
            "You can safely remove this argument.",
            stacklevel=find_stack_level(),
        )
    if errors == "ignore":
        # GH#54467
        warnings.warn(
            "errors='ignore' is deprecated and will raise in a future version. "
            "Use to_datetime without passing `errors` and catch exceptions "
            "explicitly instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    if arg is None:
        return None

    if origin != "unix":
        arg = _adjust_to_origin(arg, origin, unit)

    convert_listlike = partial(
        _convert_listlike_datetimes,
        utc=utc,
        unit=unit,
        dayfirst=dayfirst,
        yearfirst=yearfirst,
        errors=errors,
        exact=exact,
    )
    # pylint: disable-next=used-before-assignment
    result: Timestamp | NaTType | Series | Index

    if isinstance(arg, Timestamp):
        result = arg
        if utc:
            if arg.tz is not None:
                result = arg.tz_convert("utc")
            else:
                result = arg.tz_localize("utc")
    elif isinstance(arg, ABCSeries):
        cache_array = _maybe_cache(arg, format, cache, convert_listlike)
        if not cache_array.empty:
            result = arg.map(cache_array)
        else:
            values = convert_listlike(arg._values, format)
            result = arg._constructor(values, index=arg.index, name=arg.name)
    elif isinstance(arg, (ABCDataFrame, abc.MutableMapping)):
        result = _assemble_from_unit_mappings(arg, errors, utc)
    elif isinstance(arg, Index):
        cache_array = _maybe_cache(arg, format, cache, convert_listlike)
        if not cache_array.empty:
            result = _convert_and_box_cache(arg, cache_array, name=arg.name)
        else:
            result = convert_listlike(arg, format, name=arg.name)
    elif is_list_like(arg):
        try:
            # error: Argument 1 to "_maybe_cache" has incompatible type
            # "Union[float, str, datetime, List[Any], Tuple[Any, ...], ExtensionArray,
            # ndarray[Any, Any], Series]"; expected "Union[List[Any], Tuple[Any, ...],
            # Union[Union[ExtensionArray, ndarray[Any, Any]], Index, Series], Series]"
            argc = cast(
                Union[list, tuple, ExtensionArray, np.ndarray, "Series", Index], arg
            )
            cache_array = _maybe_cache(argc, format, cache, convert_listlike)
        except OutOfBoundsDatetime:
            # caching attempts to create a DatetimeIndex, which may raise
            # an OOB. If that's the desired behavior, then just reraise...
            if errors == "raise":
                raise
            # ... otherwise, continue without the cache.
            from pandas import Series

            cache_array = Series([], dtype=object)  # just an empty array
        if not cache_array.empty:
            result = _convert_and_box_cache(argc, cache_array)
        else:
            result = convert_listlike(argc, format)
    else:
        result = convert_listlike(np.array([arg]), format)[0]
        if isinstance(arg, bool) and isinstance(result, np.bool_):
            result = bool(result)  # TODO: avoid this kludge.

    #  error: Incompatible return value type (got "Union[Timestamp, NaTType,
    # Series, Index]", expected "Union[DatetimeIndex, Series, float, str,
    # NaTType, None]")
    return result  # type: ignore[return-value]


# mappings for assembling units
_unit_map = {
    "year": "year",
    "years": "year",
    "month": "month",
    "months": "month",
    "day": "day",
    "days": "day",
    "hour": "h",
    "hours": "h",
    "minute": "m",
    "minutes": "m",
    "second": "s",
    "seconds": "s",
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "us": "us",
    "microsecond": "us",
    "microseconds": "us",
    "ns": "ns",
    "nanosecond": "ns",
    "nanoseconds": "ns",
}


def _assemble_from_unit_mappings(arg, errors: DateTimeErrorChoices, utc: bool):
    """
    assemble the unit specified fields from the arg (DataFrame)
    Return a Series for actual parsing

    Parameters
    ----------
    arg : DataFrame
    errors : {'ignore', 'raise', 'coerce'}, default 'raise'

        - If :const:`'raise'`, then invalid parsing will raise an exception
        - If :const:`'coerce'`, then invalid parsing will be set as :const:`NaT`
        - If :const:`'ignore'`, then invalid parsing will return the input
    utc : bool
        Whether to convert/localize timestamps to UTC.

    Returns
    -------
    Series
    """
    from pandas import (
        DataFrame,
        to_numeric,
        to_timedelta,
    )

    arg = DataFrame(arg)
    if not arg.columns.is_unique:
        raise ValueError("cannot assemble with duplicate keys")

    # replace passed unit with _unit_map
    def f(value):
        if value in _unit_map:
            return _unit_map[value]

        # m is case significant
        if value.lower() in _unit_map:
            return _unit_map[value.lower()]

        return value

    unit = {k: f(k) for k in arg.keys()}
    unit_rev = {v: k for k, v in unit.items()}

    # we require at least Ymd
    required = ["year", "month", "day"]
    req = sorted(set(required) - set(unit_rev.keys()))
    if len(req):
        _required = ",".join(req)
        raise ValueError(
            "to assemble mappings requires at least that "
            f"[year, month, day] be specified: [{_required}] is missing"
        )

    # keys we don't recognize
    excess = sorted(set(unit_rev.keys()) - set(_unit_map.values()))
    if len(excess):
        _excess = ",".join(excess)
        raise ValueError(
            f"extra keys have been passed to the datetime assemblage: [{_excess}]"
        )

    def coerce(values):
        # we allow coercion to if errors allows
        values = to_numeric(values, errors=errors)

        # prevent overflow in case of int8 or int16
        if is_integer_dtype(values.dtype):
            values = values.astype("int64", copy=False)
        return values

    values = (
        coerce(arg[unit_rev["year"]]) * 10000
        + coerce(arg[unit_rev["month"]]) * 100
        + coerce(arg[unit_rev["day"]])
    )
    try:
        values = to_datetime(values, format="%Y%m%d", errors=errors, utc=utc)
    except (TypeError, ValueError) as err:
        raise ValueError(f"cannot assemble the datetimes: {err}") from err

    units: list[UnitChoices] = ["h", "m", "s", "ms", "us", "ns"]
    for u in units:
        value = unit_rev.get(u)
        if value is not None and value in arg:
            try:
                values += to_timedelta(coerce(arg[value]), unit=u, errors=errors)
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"cannot assemble the datetimes [{value}]: {err}"
                ) from err
    return values


__all__ = [
    "DateParseError",
    "should_cache",
    "to_datetime",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\sequences.py ===
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import call_highest_priority
from sympy.core.parameters import global_parameters
from sympy.core.function import AppliedUndef, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S, Singleton
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, Wild
from sympy.core.sympify import sympify
from sympy.matrices import Matrix
from sympy.polys import lcm, factor
from sympy.sets.sets import Interval, Intersection
from sympy.tensor.indexed import Idx
from sympy.utilities.iterables import flatten, is_sequence, iterable


###############################################################################
#                            SEQUENCES                                        #
###############################################################################


class SeqBase(Basic):
    """Base class for sequences"""

    is_commutative = True
    _op_priority = 15

    @staticmethod
    def _start_key(expr):
        """Return start (if possible) else S.Infinity.

        adapted from Set._infimum_key
        """
        try:
            start = expr.start
        except NotImplementedError:
            start = S.Infinity
        return start

    def _intersect_interval(self, other):
        """Returns start and stop.

        Takes intersection over the two intervals.
        """
        interval = Intersection(self.interval, other.interval)
        return interval.inf, interval.sup

    @property
    def gen(self):
        """Returns the generator for the sequence"""
        raise NotImplementedError("(%s).gen" % self)

    @property
    def interval(self):
        """The interval on which the sequence is defined"""
        raise NotImplementedError("(%s).interval" % self)

    @property
    def start(self):
        """The starting point of the sequence. This point is included"""
        raise NotImplementedError("(%s).start" % self)

    @property
    def stop(self):
        """The ending point of the sequence. This point is included"""
        raise NotImplementedError("(%s).stop" % self)

    @property
    def length(self):
        """Length of the sequence"""
        raise NotImplementedError("(%s).length" % self)

    @property
    def variables(self):
        """Returns a tuple of variables that are bounded"""
        return ()

    @property
    def free_symbols(self):
        """
        This method returns the symbols in the object, excluding those
        that take on a specific value (i.e. the dummy symbols).

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n, m
        >>> SeqFormula(m*n**2, (n, 0, 5)).free_symbols
        {m}
        """
        return ({j for i in self.args for j in i.free_symbols
                   .difference(self.variables)})

    @cacheit
    def coeff(self, pt):
        """Returns the coefficient at point pt"""
        if pt < self.start or pt > self.stop:
            raise IndexError("Index %s out of bounds %s" % (pt, self.interval))
        return self._eval_coeff(pt)

    def _eval_coeff(self, pt):
        raise NotImplementedError("The _eval_coeff method should be added to"
                                  "%s to return coefficient so it is available"
                                  "when coeff calls it."
                                  % self.func)

    def _ith_point(self, i):
        """Returns the i'th point of a sequence.

        Explanation
        ===========

        If start point is negative infinity, point is returned from the end.
        Assumes the first point to be indexed zero.

        Examples
        =========

        >>> from sympy import oo
        >>> from sympy.series.sequences import SeqPer

        bounded

        >>> SeqPer((1, 2, 3), (-10, 10))._ith_point(0)
        -10
        >>> SeqPer((1, 2, 3), (-10, 10))._ith_point(5)
        -5

        End is at infinity

        >>> SeqPer((1, 2, 3), (0, oo))._ith_point(5)
        5

        Starts at negative infinity

        >>> SeqPer((1, 2, 3), (-oo, 0))._ith_point(5)
        -5
        """
        if self.start is S.NegativeInfinity:
            initial = self.stop
        else:
            initial = self.start

        if self.start is S.NegativeInfinity:
            step = -1
        else:
            step = 1

        return initial + i*step

    def _add(self, other):
        """
        Should only be used internally.

        Explanation
        ===========

        self._add(other) returns a new, term-wise added sequence if self
        knows how to add with other, otherwise it returns ``None``.

        ``other`` should only be a sequence object.

        Used within :class:`SeqAdd` class.
        """
        return None

    def _mul(self, other):
        """
        Should only be used internally.

        Explanation
        ===========

        self._mul(other) returns a new, term-wise multiplied sequence if self
        knows how to multiply with other, otherwise it returns ``None``.

        ``other`` should only be a sequence object.

        Used within :class:`SeqMul` class.
        """
        return None

    def coeff_mul(self, other):
        """
        Should be used when ``other`` is not a sequence. Should be
        defined to define custom behaviour.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2).coeff_mul(2)
        SeqFormula(2*n**2, (n, 0, oo))

        Notes
        =====

        '*' defines multiplication of sequences with sequences only.
        """
        return Mul(self, other)

    def __add__(self, other):
        """Returns the term-wise addition of 'self' and 'other'.

        ``other`` should be a sequence.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2) + SeqFormula(n**3)
        SeqFormula(n**3 + n**2, (n, 0, oo))
        """
        if not isinstance(other, SeqBase):
            raise TypeError('cannot add sequence and %s' % type(other))
        return SeqAdd(self, other)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        """Returns the term-wise subtraction of ``self`` and ``other``.

        ``other`` should be a sequence.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2) - (SeqFormula(n))
        SeqFormula(n**2 - n, (n, 0, oo))
        """
        if not isinstance(other, SeqBase):
            raise TypeError('cannot subtract sequence and %s' % type(other))
        return SeqAdd(self, -other)

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        """Negates the sequence.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> -SeqFormula(n**2)
        SeqFormula(-n**2, (n, 0, oo))
        """
        return self.coeff_mul(-1)

    def __mul__(self, other):
        """Returns the term-wise multiplication of 'self' and 'other'.

        ``other`` should be a sequence. For ``other`` not being a
        sequence see :func:`coeff_mul` method.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2) * (SeqFormula(n))
        SeqFormula(n**3, (n, 0, oo))
        """
        if not isinstance(other, SeqBase):
            raise TypeError('cannot multiply sequence and %s' % type(other))
        return SeqMul(self, other)

    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return self * other

    def __iter__(self):
        for i in range(self.length):
            pt = self._ith_point(i)
            yield self.coeff(pt)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self._ith_point(index)
            return self.coeff(index)
        elif isinstance(index, slice):
            start, stop = index.start, index.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.length
            return [self.coeff(self._ith_point(i)) for i in
                    range(start, stop, index.step or 1)]

    def find_linear_recurrence(self,n,d=None,gfvar=None):
        r"""
        Finds the shortest linear recurrence that satisfies the first n
        terms of sequence of order `\leq` ``n/2`` if possible.
        If ``d`` is specified, find shortest linear recurrence of order
        `\leq` min(d, n/2) if possible.
        Returns list of coefficients ``[b(1), b(2), ...]`` corresponding to the
        recurrence relation ``x(n) = b(1)*x(n-1) + b(2)*x(n-2) + ...``
        Returns ``[]`` if no recurrence is found.
        If gfvar is specified, also returns ordinary generating function as a
        function of gfvar.

        Examples
        ========

        >>> from sympy import sequence, sqrt, oo, lucas
        >>> from sympy.abc import n, x, y
        >>> sequence(n**2).find_linear_recurrence(10, 2)
        []
        >>> sequence(n**2).find_linear_recurrence(10)
        [3, -3, 1]
        >>> sequence(2**n).find_linear_recurrence(10)
        [2]
        >>> sequence(23*n**4+91*n**2).find_linear_recurrence(10)
        [5, -10, 10, -5, 1]
        >>> sequence(sqrt(5)*(((1 + sqrt(5))/2)**n - (-(1 + sqrt(5))/2)**(-n))/5).find_linear_recurrence(10)
        [1, 1]
        >>> sequence(x+y*(-2)**(-n), (n, 0, oo)).find_linear_recurrence(30)
        [1/2, 1/2]
        >>> sequence(3*5**n + 12).find_linear_recurrence(20,gfvar=x)
        ([6, -5], 3*(5 - 21*x)/((x - 1)*(5*x - 1)))
        >>> sequence(lucas(n)).find_linear_recurrence(15,gfvar=x)
        ([1, 1], (x - 2)/(x**2 + x - 1))
        """
        from sympy.simplify import simplify
        x = [simplify(expand(t)) for t in self[:n]]
        lx = len(x)
        if d is None:
            r = lx//2
        else:
            r = min(d,lx//2)
        coeffs = []
        for l in range(1, r+1):
            l2 = 2*l
            mlist = []
            for k in range(l):
                mlist.append(x[k:k+l])
            m = Matrix(mlist)
            if m.det() != 0:
                y = simplify(m.LUsolve(Matrix(x[l:l2])))
                if lx == l2:
                    coeffs = flatten(y[::-1])
                    break
                mlist = []
                for k in range(l,lx-l):
                    mlist.append(x[k:k+l])
                m = Matrix(mlist)
                if m*y == Matrix(x[l2:]):
                    coeffs = flatten(y[::-1])
                    break
        if gfvar is None:
            return coeffs
        else:
            l = len(coeffs)
            if l == 0:
                return [], None
            else:
                n, d = x[l-1]*gfvar**(l-1), 1 - coeffs[l-1]*gfvar**l
                for i in range(l-1):
                    n += x[i]*gfvar**i
                    for j in range(l-i-1):
                        n -= coeffs[i]*x[j]*gfvar**(i+j+1)
                    d -= coeffs[i]*gfvar**(i+1)
                return coeffs, simplify(factor(n)/factor(d))

class EmptySequence(SeqBase, metaclass=Singleton):
    """Represents an empty sequence.

    The empty sequence is also available as a singleton as
    ``S.EmptySequence``.

    Examples
    ========

    >>> from sympy import EmptySequence, SeqPer
    >>> from sympy.abc import x
    >>> EmptySequence
    EmptySequence
    >>> SeqPer((1, 2), (x, 0, 10)) + EmptySequence
    SeqPer((1, 2), (x, 0, 10))
    >>> SeqPer((1, 2)) * EmptySequence
    EmptySequence
    >>> EmptySequence.coeff_mul(-1)
    EmptySequence
    """

    @property
    def interval(self):
        return S.EmptySet

    @property
    def length(self):
        return S.Zero

    def coeff_mul(self, coeff):
        """See docstring of SeqBase.coeff_mul"""
        return self

    def __iter__(self):
        return iter([])


class SeqExpr(SeqBase):
    """Sequence expression class.

    Various sequences should inherit from this class.

    Examples
    ========

    >>> from sympy.series.sequences import SeqExpr
    >>> from sympy.abc import x
    >>> from sympy import Tuple
    >>> s = SeqExpr(Tuple(1, 2, 3), Tuple(x, 0, 10))
    >>> s.gen
    (1, 2, 3)
    >>> s.interval
    Interval(0, 10)
    >>> s.length
    11

    See Also
    ========

    sympy.series.sequences.SeqPer
    sympy.series.sequences.SeqFormula
    """

    @property
    def gen(self):
        return self.args[0]

    @property
    def interval(self):
        return Interval(self.args[1][1], self.args[1][2])

    @property
    def start(self):
        return self.interval.inf

    @property
    def stop(self):
        return self.interval.sup

    @property
    def length(self):
        return self.stop - self.start + 1

    @property
    def variables(self):
        return (self.args[1][0],)


class SeqPer(SeqExpr):
    """
    Represents a periodic sequence.

    The elements are repeated after a given period.

    Examples
    ========

    >>> from sympy import SeqPer, oo
    >>> from sympy.abc import k

    >>> s = SeqPer((1, 2, 3), (0, 5))
    >>> s.periodical
    (1, 2, 3)
    >>> s.period
    3

    For value at a particular point

    >>> s.coeff(3)
    1

    supports slicing

    >>> s[:]
    [1, 2, 3, 1, 2, 3]

    iterable

    >>> list(s)
    [1, 2, 3, 1, 2, 3]

    sequence starts from negative infinity

    >>> SeqPer((1, 2, 3), (-oo, 0))[0:6]
    [1, 2, 3, 1, 2, 3]

    Periodic formulas

    >>> SeqPer((k, k**2, k**3), (k, 0, oo))[0:6]
    [0, 1, 8, 3, 16, 125]

    See Also
    ========

    sympy.series.sequences.SeqFormula
    """

    def __new__(cls, periodical, limits=None):
        periodical = sympify(periodical)

        def _find_x(periodical):
            free = periodical.free_symbols
            if len(periodical.free_symbols) == 1:
                return free.pop()
            else:
                return Dummy('k')

        x, start, stop = None, None, None
        if limits is None:
            x, start, stop = _find_x(periodical), 0, S.Infinity
        if is_sequence(limits, Tuple):
            if len(limits) == 3:
                x, start, stop = limits
            elif len(limits) == 2:
                x = _find_x(periodical)
                start, stop = limits

        if not isinstance(x, (Symbol, Idx)) or start is None or stop is None:
            raise ValueError('Invalid limits given: %s' % str(limits))

        if start is S.NegativeInfinity and stop is S.Infinity:
                raise ValueError("Both the start and end value"
                                 "cannot be unbounded")

        limits = sympify((x, start, stop))

        if is_sequence(periodical, Tuple):
            periodical = sympify(tuple(flatten(periodical)))
        else:
            raise ValueError("invalid period %s should be something "
                             "like e.g (1, 2) " % periodical)

        if Interval(limits[1], limits[2]) is S.EmptySet:
            return S.EmptySequence

        return Basic.__new__(cls, periodical, limits)

    @property
    def period(self):
        return len(self.gen)

    @property
    def periodical(self):
        return self.gen

    def _eval_coeff(self, pt):
        if self.start is S.NegativeInfinity:
            idx = (self.stop - pt) % self.period
        else:
            idx = (pt - self.start) % self.period
        return self.periodical[idx].subs(self.variables[0], pt)

    def _add(self, other):
        """See docstring of SeqBase._add"""
        if isinstance(other, SeqPer):
            per1, lper1 = self.periodical, self.period
            per2, lper2 = other.periodical, other.period

            per_length = lcm(lper1, lper2)

            new_per = []
            for x in range(per_length):
                ele1 = per1[x % lper1]
                ele2 = per2[x % lper2]
                new_per.append(ele1 + ele2)

            start, stop = self._intersect_interval(other)
            return SeqPer(new_per, (self.variables[0], start, stop))

    def _mul(self, other):
        """See docstring of SeqBase._mul"""
        if isinstance(other, SeqPer):
            per1, lper1 = self.periodical, self.period
            per2, lper2 = other.periodical, other.period

            per_length = lcm(lper1, lper2)

            new_per = []
            for x in range(per_length):
                ele1 = per1[x % lper1]
                ele2 = per2[x % lper2]
                new_per.append(ele1 * ele2)

            start, stop = self._intersect_interval(other)
            return SeqPer(new_per, (self.variables[0], start, stop))

    def coeff_mul(self, coeff):
        """See docstring of SeqBase.coeff_mul"""
        coeff = sympify(coeff)
        per = [x * coeff for x in self.periodical]
        return SeqPer(per, self.args[1])


class SeqFormula(SeqExpr):
    """
    Represents sequence based on a formula.

    Elements are generated using a formula.

    Examples
    ========

    >>> from sympy import SeqFormula, oo, Symbol
    >>> n = Symbol('n')
    >>> s = SeqFormula(n**2, (n, 0, 5))
    >>> s.formula
    n**2

    For value at a particular point

    >>> s.coeff(3)
    9

    supports slicing

    >>> s[:]
    [0, 1, 4, 9, 16, 25]

    iterable

    >>> list(s)
    [0, 1, 4, 9, 16, 25]

    sequence starts from negative infinity

    >>> SeqFormula(n**2, (-oo, 0))[0:6]
    [0, 1, 4, 9, 16, 25]

    See Also
    ========

    sympy.series.sequences.SeqPer
    """

    def __new__(cls, formula, limits=None):
        formula = sympify(formula)

        def _find_x(formula):
            free = formula.free_symbols
            if len(free) == 1:
                return free.pop()
            elif not free:
                return Dummy('k')
            else:
                raise ValueError(
                    " specify dummy variables for %s. If the formula contains"
                    " more than one free symbol, a dummy variable should be"
                    " supplied explicitly e.g., SeqFormula(m*n**2, (n, 0, 5))"
                    % formula)

        x, start, stop = None, None, None
        if limits is None:
            x, start, stop = _find_x(formula), 0, S.Infinity
        if is_sequence(limits, Tuple):
            if len(limits) == 3:
                x, start, stop = limits
            elif len(limits) == 2:
                x = _find_x(formula)
                start, stop = limits

        if not isinstance(x, (Symbol, Idx)) or start is None or stop is None:
            raise ValueError('Invalid limits given: %s' % str(limits))

        if start is S.NegativeInfinity and stop is S.Infinity:
                raise ValueError("Both the start and end value "
                                 "cannot be unbounded")
        limits = sympify((x, start, stop))

        if Interval(limits[1], limits[2]) is S.EmptySet:
            return S.EmptySequence

        return Basic.__new__(cls, formula, limits)

    @property
    def formula(self):
        return self.gen

    def _eval_coeff(self, pt):
        d = self.variables[0]
        return self.formula.subs(d, pt)

    def _add(self, other):
        """See docstring of SeqBase._add"""
        if isinstance(other, SeqFormula):
            form1, v1 = self.formula, self.variables[0]
            form2, v2 = other.formula, other.variables[0]
            formula = form1 + form2.subs(v2, v1)
            start, stop = self._intersect_interval(other)
            return SeqFormula(formula, (v1, start, stop))

    def _mul(self, other):
        """See docstring of SeqBase._mul"""
        if isinstance(other, SeqFormula):
            form1, v1 = self.formula, self.variables[0]
            form2, v2 = other.formula, other.variables[0]
            formula = form1 * form2.subs(v2, v1)
            start, stop = self._intersect_interval(other)
            return SeqFormula(formula, (v1, start, stop))

    def coeff_mul(self, coeff):
        """See docstring of SeqBase.coeff_mul"""
        coeff = sympify(coeff)
        formula = self.formula * coeff
        return SeqFormula(formula, self.args[1])

    def expand(self, *args, **kwargs):
        return SeqFormula(expand(self.formula, *args, **kwargs), self.args[1])

class RecursiveSeq(SeqBase):
    """
    A finite degree recursive sequence.

    Explanation
    ===========

    That is, a sequence a(n) that depends on a fixed, finite number of its
    previous values. The general form is

        a(n) = f(a(n - 1), a(n - 2), ..., a(n - d))

    for some fixed, positive integer d, where f is some function defined by a
    SymPy expression.

    Parameters
    ==========

    recurrence : SymPy expression defining recurrence
        This is *not* an equality, only the expression that the nth term is
        equal to. For example, if :code:`a(n) = f(a(n - 1), ..., a(n - d))`,
        then the expression should be :code:`f(a(n - 1), ..., a(n - d))`.

    yn : applied undefined function
        Represents the nth term of the sequence as e.g. :code:`y(n)` where
        :code:`y` is an undefined function and `n` is the sequence index.

    n : symbolic argument
        The name of the variable that the recurrence is in, e.g., :code:`n` if
        the recurrence function is :code:`y(n)`.

    initial : iterable with length equal to the degree of the recurrence
        The initial values of the recurrence.

    start : start value of sequence (inclusive)

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy.series.sequences import RecursiveSeq
    >>> y = Function("y")
    >>> n = symbols("n")
    >>> fib = RecursiveSeq(y(n - 1) + y(n - 2), y(n), n, [0, 1])

    >>> fib.coeff(3) # Value at a particular point
    2

    >>> fib[:6] # supports slicing
    [0, 1, 1, 2, 3, 5]

    >>> fib.recurrence # inspect recurrence
    Eq(y(n), y(n - 2) + y(n - 1))

    >>> fib.degree # automatically determine degree
    2

    >>> for x in zip(range(10), fib): # supports iteration
    ...     print(x)
    (0, 0)
    (1, 1)
    (2, 1)
    (3, 2)
    (4, 3)
    (5, 5)
    (6, 8)
    (7, 13)
    (8, 21)
    (9, 34)

    See Also
    ========

    sympy.series.sequences.SeqFormula

    """

    def __new__(cls, recurrence, yn, n, initial=None, start=0):
        if not isinstance(yn, AppliedUndef):
            raise TypeError("recurrence sequence must be an applied undefined function"
                            ", found `{}`".format(yn))

        if not isinstance(n, Basic) or not n.is_symbol:
            raise TypeError("recurrence variable must be a symbol"
                            ", found `{}`".format(n))

        if yn.args != (n,):
            raise TypeError("recurrence sequence does not match symbol")

        y = yn.func

        k = Wild("k", exclude=(n,))
        degree = 0

        # Find all applications of y in the recurrence and check that:
        #   1. The function y is only being used with a single argument; and
        #   2. All arguments are n + k for constant negative integers k.

        prev_ys = recurrence.find(y)
        for prev_y in prev_ys:
            if len(prev_y.args) != 1:
                raise TypeError("Recurrence should be in a single variable")

            shift = prev_y.args[0].match(n + k)[k]
            if not (shift.is_constant() and shift.is_integer and shift < 0):
                raise TypeError("Recurrence should have constant,"
                                " negative, integer shifts"
                                " (found {})".format(prev_y))

            if -shift > degree:
                degree = -shift

        if not initial:
            initial = [Dummy("c_{}".format(k)) for k in range(degree)]

        if len(initial) != degree:
            raise ValueError("Number of initial terms must equal degree")

        degree = Integer(degree)
        start = sympify(start)

        initial = Tuple(*(sympify(x) for x in initial))

        seq = Basic.__new__(cls, recurrence, yn, n, initial, start)

        seq.cache = {y(start + k): init for k, init in enumerate(initial)}
        seq.degree = degree

        return seq

    @property
    def _recurrence(self):
        """Equation defining recurrence."""
        return self.args[0]

    @property
    def recurrence(self):
        """Equation defining recurrence."""
        return Eq(self.yn, self.args[0])

    @property
    def yn(self):
        """Applied function representing the nth term"""
        return self.args[1]

    @property
    def y(self):
        """Undefined function for the nth term of the sequence"""
        return self.yn.func

    @property
    def n(self):
        """Sequence index symbol"""
        return self.args[2]

    @property
    def initial(self):
        """The initial values of the sequence"""
        return self.args[3]

    @property
    def start(self):
        """The starting point of the sequence. This point is included"""
        return self.args[4]

    @property
    def stop(self):
        """The ending point of the sequence. (oo)"""
        return S.Infinity

    @property
    def interval(self):
        """Interval on which sequence is defined."""
        return (self.start, S.Infinity)

    def _eval_coeff(self, index):
        if index - self.start < len(self.cache):
            return self.cache[self.y(index)]

        for current in range(len(self.cache), index + 1):
            # Use xreplace over subs for performance.
            # See issue #10697.
            seq_index = self.start + current
            current_recurrence = self._recurrence.xreplace({self.n: seq_index})
            new_term = current_recurrence.xreplace(self.cache)

            self.cache[self.y(seq_index)] = new_term

        return self.cache[self.y(self.start + current)]

    def __iter__(self):
        index = self.start
        while True:
            yield self._eval_coeff(index)
            index += 1


def sequence(seq, limits=None):
    """
    Returns appropriate sequence object.

    Explanation
    ===========

    If ``seq`` is a SymPy sequence, returns :class:`SeqPer` object
    otherwise returns :class:`SeqFormula` object.

    Examples
    ========

    >>> from sympy import sequence
    >>> from sympy.abc import n
    >>> sequence(n**2, (n, 0, 5))
    SeqFormula(n**2, (n, 0, 5))
    >>> sequence((1, 2, 3), (n, 0, 5))
    SeqPer((1, 2, 3), (n, 0, 5))

    See Also
    ========

    sympy.series.sequences.SeqPer
    sympy.series.sequences.SeqFormula
    """
    seq = sympify(seq)

    if is_sequence(seq, Tuple):
        return SeqPer(seq, limits)
    else:
        return SeqFormula(seq, limits)


###############################################################################
#                            OPERATIONS                                       #
###############################################################################


class SeqExprOp(SeqBase):
    """
    Base class for operations on sequences.

    Examples
    ========

    >>> from sympy.series.sequences import SeqExprOp, sequence
    >>> from sympy.abc import n
    >>> s1 = sequence(n**2, (n, 0, 10))
    >>> s2 = sequence((1, 2, 3), (n, 5, 10))
    >>> s = SeqExprOp(s1, s2)
    >>> s.gen
    (n**2, (1, 2, 3))
    >>> s.interval
    Interval(5, 10)
    >>> s.length
    6

    See Also
    ========

    sympy.series.sequences.SeqAdd
    sympy.series.sequences.SeqMul
    """
    @property
    def gen(self):
        """Generator for the sequence.

        returns a tuple of generators of all the argument sequences.
        """
        return tuple(a.gen for a in self.args)

    @property
    def interval(self):
        """Sequence is defined on the intersection
        of all the intervals of respective sequences
        """
        return Intersection(*(a.interval for a in self.args))

    @property
    def start(self):
        return self.interval.inf

    @property
    def stop(self):
        return self.interval.sup

    @property
    def variables(self):
        """Cumulative of all the bound variables"""
        return tuple(flatten([a.variables for a in self.args]))

    @property
    def length(self):
        return self.stop - self.start + 1


class SeqAdd(SeqExprOp):
    """Represents term-wise addition of sequences.

    Rules:
        * The interval on which sequence is defined is the intersection
          of respective intervals of sequences.
        * Anything + :class:`EmptySequence` remains unchanged.
        * Other rules are defined in ``_add`` methods of sequence classes.

    Examples
    ========

    >>> from sympy import EmptySequence, oo, SeqAdd, SeqPer, SeqFormula
    >>> from sympy.abc import n
    >>> SeqAdd(SeqPer((1, 2), (n, 0, oo)), EmptySequence)
    SeqPer((1, 2), (n, 0, oo))
    >>> SeqAdd(SeqPer((1, 2), (n, 0, 5)), SeqPer((1, 2), (n, 6, 10)))
    EmptySequence
    >>> SeqAdd(SeqPer((1, 2), (n, 0, oo)), SeqFormula(n**2, (n, 0, oo)))
    SeqAdd(SeqFormula(n**2, (n, 0, oo)), SeqPer((1, 2), (n, 0, oo)))
    >>> SeqAdd(SeqFormula(n**3), SeqFormula(n**2))
    SeqFormula(n**3 + n**2, (n, 0, oo))

    See Also
    ========

    sympy.series.sequences.SeqMul
    """

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)

        # flatten inputs
        args = list(args)

        # adapted from sympy.sets.sets.Union
        def _flatten(arg):
            if isinstance(arg, SeqBase):
                if isinstance(arg, SeqAdd):
                    return sum(map(_flatten, arg.args), [])
                else:
                    return [arg]
            if iterable(arg):
                return sum(map(_flatten, arg), [])
            raise TypeError("Input must be Sequences or "
                            " iterables of Sequences")
        args = _flatten(args)

        args = [a for a in args if a is not S.EmptySequence]

        # Addition of no sequences is EmptySequence
        if not args:
            return S.EmptySequence

        if Intersection(*(a.interval for a in args)) is S.EmptySet:
            return S.EmptySequence

        # reduce using known rules
        if evaluate:
            return SeqAdd.reduce(args)

        args = list(ordered(args, SeqBase._start_key))

        return Basic.__new__(cls, *args)

    @staticmethod
    def reduce(args):
        """Simplify :class:`SeqAdd` using known rules.

        Iterates through all pairs and ask the constituent
        sequences if they can simplify themselves with any other constituent.

        Notes
        =====

        adapted from ``Union.reduce``

        """
        new_args = True
        while new_args:
            for id1, s in enumerate(args):
                new_args = False
                for id2, t in enumerate(args):
                    if id1 == id2:
                        continue
                    new_seq = s._add(t)
                    # This returns None if s does not know how to add
                    # with t. Returns the newly added sequence otherwise
                    if new_seq is not None:
                        new_args = [a for a in args if a not in (s, t)]
                        new_args.append(new_seq)
                        break
                if new_args:
                    args = new_args
                    break

        if len(args) == 1:
            return args.pop()
        else:
            return SeqAdd(args, evaluate=False)

    def _eval_coeff(self, pt):
        """adds up the coefficients of all the sequences at point pt"""
        return sum(a.coeff(pt) for a in self.args)


class SeqMul(SeqExprOp):
    r"""Represents term-wise multiplication of sequences.

    Explanation
    ===========

    Handles multiplication of sequences only. For multiplication
    with other objects see :func:`SeqBase.coeff_mul`.

    Rules:
        * The interval on which sequence is defined is the intersection
          of respective intervals of sequences.
        * Anything \* :class:`EmptySequence` returns :class:`EmptySequence`.
        * Other rules are defined in ``_mul`` methods of sequence classes.

    Examples
    ========

    >>> from sympy import EmptySequence, oo, SeqMul, SeqPer, SeqFormula
    >>> from sympy.abc import n
    >>> SeqMul(SeqPer((1, 2), (n, 0, oo)), EmptySequence)
    EmptySequence
    >>> SeqMul(SeqPer((1, 2), (n, 0, 5)), SeqPer((1, 2), (n, 6, 10)))
    EmptySequence
    >>> SeqMul(SeqPer((1, 2), (n, 0, oo)), SeqFormula(n**2))
    SeqMul(SeqFormula(n**2, (n, 0, oo)), SeqPer((1, 2), (n, 0, oo)))
    >>> SeqMul(SeqFormula(n**3), SeqFormula(n**2))
    SeqFormula(n**5, (n, 0, oo))

    See Also
    ========

    sympy.series.sequences.SeqAdd
    """

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)

        # flatten inputs
        args = list(args)

        # adapted from sympy.sets.sets.Union
        def _flatten(arg):
            if isinstance(arg, SeqBase):
                if isinstance(arg, SeqMul):
                    return sum(map(_flatten, arg.args), [])
                else:
                    return [arg]
            elif iterable(arg):
                return sum(map(_flatten, arg), [])
            raise TypeError("Input must be Sequences or "
                            " iterables of Sequences")
        args = _flatten(args)

        # Multiplication of no sequences is EmptySequence
        if not args:
            return S.EmptySequence

        if Intersection(*(a.interval for a in args)) is S.EmptySet:
            return S.EmptySequence

        # reduce using known rules
        if evaluate:
            return SeqMul.reduce(args)

        args = list(ordered(args, SeqBase._start_key))

        return Basic.__new__(cls, *args)

    @staticmethod
    def reduce(args):
        """Simplify a :class:`SeqMul` using known rules.

        Explanation
        ===========

        Iterates through all pairs and ask the constituent
        sequences if they can simplify themselves with any other constituent.

        Notes
        =====

        adapted from ``Union.reduce``

        """
        new_args = True
        while new_args:
            for id1, s in enumerate(args):
                new_args = False
                for id2, t in enumerate(args):
                    if id1 == id2:
                        continue
                    new_seq = s._mul(t)
                    # This returns None if s does not know how to multiply
                    # with t. Returns the newly multiplied sequence otherwise
                    if new_seq is not None:
                        new_args = [a for a in args if a not in (s, t)]
                        new_args.append(new_seq)
                        break
                if new_args:
                    args = new_args
                    break

        if len(args) == 1:
            return args.pop()
        else:
            return SeqMul(args, evaluate=False)

    def _eval_coeff(self, pt):
        """multiplies the coefficients of all the sequences at point pt"""
        val = 1
        for a in self.args:
            val *= a.coeff(pt)
        return val

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\datastructures\structures.py ===
from __future__ import annotations

import collections.abc as cabc
import typing as t
from copy import deepcopy

from .. import exceptions
from .._internal import _missing
from .mixins import ImmutableDictMixin
from .mixins import ImmutableListMixin
from .mixins import ImmutableMultiDictMixin
from .mixins import UpdateDictMixin

if t.TYPE_CHECKING:
    import typing_extensions as te

K = t.TypeVar("K")
V = t.TypeVar("V")
T = t.TypeVar("T")


def iter_multi_items(
    mapping: (
        MultiDict[K, V]
        | cabc.Mapping[K, V | list[V] | tuple[V, ...] | set[V]]
        | cabc.Iterable[tuple[K, V]]
    ),
) -> cabc.Iterator[tuple[K, V]]:
    """Iterates over the items of a mapping yielding keys and values
    without dropping any from more complex structures.
    """
    if isinstance(mapping, MultiDict):
        yield from mapping.items(multi=True)
    elif isinstance(mapping, cabc.Mapping):
        for key, value in mapping.items():
            if isinstance(value, (list, tuple, set)):
                for v in value:
                    yield key, v
            else:
                yield key, value
    else:
        yield from mapping


class ImmutableList(ImmutableListMixin, list[V]):  # type: ignore[misc]
    """An immutable :class:`list`.

    .. versionadded:: 0.5

    :private:
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list.__repr__(self)})"


class TypeConversionDict(dict[K, V]):
    """Works like a regular dict but the :meth:`get` method can perform
    type conversions.  :class:`MultiDict` and :class:`CombinedMultiDict`
    are subclasses of this class and provide the same feature.

    .. versionadded:: 0.5
    """

    @t.overload  # type: ignore[override]
    def get(self, key: K) -> V | None: ...
    @t.overload
    def get(self, key: K, default: V) -> V: ...
    @t.overload
    def get(self, key: K, default: T) -> V | T: ...
    @t.overload
    def get(self, key: str, type: cabc.Callable[[V], T]) -> T | None: ...
    @t.overload
    def get(self, key: str, default: T, type: cabc.Callable[[V], T]) -> T: ...
    def get(  # type: ignore[misc]
        self,
        key: K,
        default: V | T | None = None,
        type: cabc.Callable[[V], T] | None = None,
    ) -> V | T | None:
        """Return the default value if the requested data doesn't exist.
        If `type` is provided and is a callable it should convert the value,
        return it or raise a :exc:`ValueError` if that is not possible.  In
        this case the function will return the default as if the value was not
        found:

        >>> d = TypeConversionDict(foo='42', bar='blub')
        >>> d.get('foo', type=int)
        42
        >>> d.get('bar', -1, type=int)
        -1

        :param key: The key to be looked up.
        :param default: The default value to be returned if the key can't
                        be looked up.  If not further specified `None` is
                        returned.
        :param type: A callable that is used to cast the value in the
                     :class:`MultiDict`.  If a :exc:`ValueError` or a
                     :exc:`TypeError` is raised by this callable the default
                     value is returned.

        .. versionchanged:: 3.0.2
           Returns the default value on :exc:`TypeError`, too.
        """
        try:
            rv = self[key]
        except KeyError:
            return default

        if type is None:
            return rv

        try:
            return type(rv)
        except (ValueError, TypeError):
            return default


class ImmutableTypeConversionDict(ImmutableDictMixin[K, V], TypeConversionDict[K, V]):  # type: ignore[misc]
    """Works like a :class:`TypeConversionDict` but does not support
    modifications.

    .. versionadded:: 0.5
    """

    def copy(self) -> TypeConversionDict[K, V]:
        """Return a shallow mutable copy of this object.  Keep in mind that
        the standard library's :func:`copy` function is a no-op for this class
        like for any other python immutable type (eg: :class:`tuple`).
        """
        return TypeConversionDict(self)

    def __copy__(self) -> te.Self:
        return self


class MultiDict(TypeConversionDict[K, V]):
    """A :class:`MultiDict` is a dictionary subclass customized to deal with
    multiple values for the same key which is for example used by the parsing
    functions in the wrappers.  This is necessary because some HTML form
    elements pass multiple values for the same key.

    :class:`MultiDict` implements all standard dictionary methods.
    Internally, it saves all values for a key as a list, but the standard dict
    access methods will only return the first value for a key. If you want to
    gain access to the other values, too, you have to use the `list` methods as
    explained below.

    Basic Usage:

    >>> d = MultiDict([('a', 'b'), ('a', 'c')])
    >>> d
    MultiDict([('a', 'b'), ('a', 'c')])
    >>> d['a']
    'b'
    >>> d.getlist('a')
    ['b', 'c']
    >>> 'a' in d
    True

    It behaves like a normal dict thus all dict functions will only return the
    first value when multiple values for one key are found.

    From Werkzeug 0.3 onwards, the `KeyError` raised by this class is also a
    subclass of the :exc:`~exceptions.BadRequest` HTTP exception and will
    render a page for a ``400 BAD REQUEST`` if caught in a catch-all for HTTP
    exceptions.

    A :class:`MultiDict` can be constructed from an iterable of
    ``(key, value)`` tuples, a dict, a :class:`MultiDict` or from Werkzeug 0.2
    onwards some keyword parameters.

    :param mapping: the initial value for the :class:`MultiDict`.  Either a
                    regular dict, an iterable of ``(key, value)`` tuples
                    or `None`.

    .. versionchanged:: 3.1
        Implement ``|`` and ``|=`` operators.
    """

    def __init__(
        self,
        mapping: (
            MultiDict[K, V]
            | cabc.Mapping[K, V | list[V] | tuple[V, ...] | set[V]]
            | cabc.Iterable[tuple[K, V]]
            | None
        ) = None,
    ) -> None:
        if mapping is None:
            super().__init__()
        elif isinstance(mapping, MultiDict):
            super().__init__((k, vs[:]) for k, vs in mapping.lists())
        elif isinstance(mapping, cabc.Mapping):
            tmp = {}
            for key, value in mapping.items():
                if isinstance(value, (list, tuple, set)):
                    value = list(value)

                    if not value:
                        continue
                else:
                    value = [value]
                tmp[key] = value
            super().__init__(tmp)  # type: ignore[arg-type]
        else:
            tmp = {}
            for key, value in mapping:
                tmp.setdefault(key, []).append(value)
            super().__init__(tmp)  # type: ignore[arg-type]

    def __getstate__(self) -> t.Any:
        return dict(self.lists())

    def __setstate__(self, value: t.Any) -> None:
        super().clear()
        super().update(value)

    def __iter__(self) -> cabc.Iterator[K]:
        # https://github.com/python/cpython/issues/87412
        # If __iter__ is not overridden, Python uses a fast path for dict(md),
        # taking the data directly and getting lists of values, rather than
        # calling __getitem__ and getting only the first value.
        return super().__iter__()

    def __getitem__(self, key: K) -> V:
        """Return the first data value for this key;
        raises KeyError if not found.

        :param key: The key to be looked up.
        :raise KeyError: if the key does not exist.
        """

        if key in self:
            lst = super().__getitem__(key)
            if len(lst) > 0:  # type: ignore[arg-type]
                return lst[0]  # type: ignore[index,no-any-return]
        raise exceptions.BadRequestKeyError(key)

    def __setitem__(self, key: K, value: V) -> None:
        """Like :meth:`add` but removes an existing key first.

        :param key: the key for the value.
        :param value: the value to set.
        """
        super().__setitem__(key, [value])  # type: ignore[assignment]

    def add(self, key: K, value: V) -> None:
        """Adds a new value for the key.

        .. versionadded:: 0.6

        :param key: the key for the value.
        :param value: the value to add.
        """
        super().setdefault(key, []).append(value)  # type: ignore[arg-type,attr-defined]

    @t.overload
    def getlist(self, key: K) -> list[V]: ...
    @t.overload
    def getlist(self, key: K, type: cabc.Callable[[V], T]) -> list[T]: ...
    def getlist(
        self, key: K, type: cabc.Callable[[V], T] | None = None
    ) -> list[V] | list[T]:
        """Return the list of items for a given key. If that key is not in the
        `MultiDict`, the return value will be an empty list.  Just like `get`,
        `getlist` accepts a `type` parameter.  All items will be converted
        with the callable defined there.

        :param key: The key to be looked up.
        :param type: Callable to convert each value. If a ``ValueError`` or
            ``TypeError`` is raised, the value is omitted.
        :return: a :class:`list` of all the values for the key.

        .. versionchanged:: 3.1
            Catches ``TypeError`` in addition to ``ValueError``.
        """
        try:
            rv: list[V] = super().__getitem__(key)  # type: ignore[assignment]
        except KeyError:
            return []
        if type is None:
            return list(rv)
        result = []
        for item in rv:
            try:
                result.append(type(item))
            except (ValueError, TypeError):
                pass
        return result

    def setlist(self, key: K, new_list: cabc.Iterable[V]) -> None:
        """Remove the old values for a key and add new ones.  Note that the list
        you pass the values in will be shallow-copied before it is inserted in
        the dictionary.

        >>> d = MultiDict()
        >>> d.setlist('foo', ['1', '2'])
        >>> d['foo']
        '1'
        >>> d.getlist('foo')
        ['1', '2']

        :param key: The key for which the values are set.
        :param new_list: An iterable with the new values for the key.  Old values
                         are removed first.
        """
        super().__setitem__(key, list(new_list))  # type: ignore[assignment]

    @t.overload
    def setdefault(self, key: K) -> None: ...
    @t.overload
    def setdefault(self, key: K, default: V) -> V: ...
    def setdefault(self, key: K, default: V | None = None) -> V | None:
        """Returns the value for the key if it is in the dict, otherwise it
        returns `default` and sets that value for `key`.

        :param key: The key to be looked up.
        :param default: The default value to be returned if the key is not
                        in the dict.  If not further specified it's `None`.
        """
        if key not in self:
            self[key] = default  # type: ignore[assignment]

        return self[key]

    def setlistdefault(
        self, key: K, default_list: cabc.Iterable[V] | None = None
    ) -> list[V]:
        """Like `setdefault` but sets multiple values.  The list returned
        is not a copy, but the list that is actually used internally.  This
        means that you can put new values into the dict by appending items
        to the list:

        >>> d = MultiDict({"foo": 1})
        >>> d.setlistdefault("foo").extend([2, 3])
        >>> d.getlist("foo")
        [1, 2, 3]

        :param key: The key to be looked up.
        :param default_list: An iterable of default values.  It is either copied
                             (in case it was a list) or converted into a list
                             before returned.
        :return: a :class:`list`
        """
        if key not in self:
            super().__setitem__(key, list(default_list or ()))  # type: ignore[assignment]

        return super().__getitem__(key)  # type: ignore[return-value]

    def items(self, multi: bool = False) -> cabc.Iterable[tuple[K, V]]:  # type: ignore[override]
        """Return an iterator of ``(key, value)`` pairs.

        :param multi: If set to `True` the iterator returned will have a pair
                      for each value of each key.  Otherwise it will only
                      contain pairs for the first value of each key.
        """
        values: list[V]

        for key, values in super().items():  # type: ignore[assignment]
            if multi:
                for value in values:
                    yield key, value
            else:
                yield key, values[0]

    def lists(self) -> cabc.Iterable[tuple[K, list[V]]]:
        """Return a iterator of ``(key, values)`` pairs, where values is the list
        of all values associated with the key."""
        values: list[V]

        for key, values in super().items():  # type: ignore[assignment]
            yield key, list(values)

    def values(self) -> cabc.Iterable[V]:  # type: ignore[override]
        """Returns an iterator of the first value on every key's value list."""
        values: list[V]

        for values in super().values():  # type: ignore[assignment]
            yield values[0]

    def listvalues(self) -> cabc.Iterable[list[V]]:
        """Return an iterator of all values associated with a key.  Zipping
        :meth:`keys` and this is the same as calling :meth:`lists`:

        >>> d = MultiDict({"foo": [1, 2, 3]})
        >>> zip(d.keys(), d.listvalues()) == d.lists()
        True
        """
        return super().values()  # type: ignore[return-value]

    def copy(self) -> te.Self:
        """Return a shallow copy of this object."""
        return self.__class__(self)

    def deepcopy(self, memo: t.Any = None) -> te.Self:
        """Return a deep copy of this object."""
        return self.__class__(deepcopy(self.to_dict(flat=False), memo))

    @t.overload
    def to_dict(self) -> dict[K, V]: ...
    @t.overload
    def to_dict(self, flat: t.Literal[False]) -> dict[K, list[V]]: ...
    def to_dict(self, flat: bool = True) -> dict[K, V] | dict[K, list[V]]:
        """Return the contents as regular dict.  If `flat` is `True` the
        returned dict will only have the first item present, if `flat` is
        `False` all values will be returned as lists.

        :param flat: If set to `False` the dict returned will have lists
                     with all the values in it.  Otherwise it will only
                     contain the first value for each key.
        :return: a :class:`dict`
        """
        if flat:
            return dict(self.items())
        return dict(self.lists())

    def update(  # type: ignore[override]
        self,
        mapping: (
            MultiDict[K, V]
            | cabc.Mapping[K, V | list[V] | tuple[V, ...] | set[V]]
            | cabc.Iterable[tuple[K, V]]
        ),
    ) -> None:
        """update() extends rather than replaces existing key lists:

        >>> a = MultiDict({'x': 1})
        >>> b = MultiDict({'x': 2, 'y': 3})
        >>> a.update(b)
        >>> a
        MultiDict([('y', 3), ('x', 1), ('x', 2)])

        If the value list for a key in ``other_dict`` is empty, no new values
        will be added to the dict and the key will not be created:

        >>> x = {'empty_list': []}
        >>> y = MultiDict()
        >>> y.update(x)
        >>> y
        MultiDict([])
        """
        for key, value in iter_multi_items(mapping):
            self.add(key, value)

    def __or__(  # type: ignore[override]
        self, other: cabc.Mapping[K, V | list[V] | tuple[V, ...] | set[V]]
    ) -> MultiDict[K, V]:
        if not isinstance(other, cabc.Mapping):
            return NotImplemented

        rv = self.copy()
        rv.update(other)
        return rv

    def __ior__(  # type: ignore[override]
        self,
        other: (
            cabc.Mapping[K, V | list[V] | tuple[V, ...] | set[V]]
            | cabc.Iterable[tuple[K, V]]
        ),
    ) -> te.Self:
        if not isinstance(other, (cabc.Mapping, cabc.Iterable)):
            return NotImplemented

        self.update(other)
        return self

    @t.overload
    def pop(self, key: K) -> V: ...
    @t.overload
    def pop(self, key: K, default: V) -> V: ...
    @t.overload
    def pop(self, key: K, default: T) -> V | T: ...
    def pop(
        self,
        key: K,
        default: V | T = _missing,  # type: ignore[assignment]
    ) -> V | T:
        """Pop the first item for a list on the dict.  Afterwards the
        key is removed from the dict, so additional values are discarded:

        >>> d = MultiDict({"foo": [1, 2, 3]})
        >>> d.pop("foo")
        1
        >>> "foo" in d
        False

        :param key: the key to pop.
        :param default: if provided the value to return if the key was
                        not in the dictionary.
        """
        lst: list[V]

        try:
            lst = super().pop(key)  # type: ignore[assignment]

            if len(lst) == 0:
                raise exceptions.BadRequestKeyError(key)

            return lst[0]
        except KeyError:
            if default is not _missing:
                return default

            raise exceptions.BadRequestKeyError(key) from None

    def popitem(self) -> tuple[K, V]:
        """Pop an item from the dict."""
        item: tuple[K, list[V]]

        try:
            item = super().popitem()  # type: ignore[assignment]

            if len(item[1]) == 0:
                raise exceptions.BadRequestKeyError(item[0])

            return item[0], item[1][0]
        except KeyError as e:
            raise exceptions.BadRequestKeyError(e.args[0]) from None

    def poplist(self, key: K) -> list[V]:
        """Pop the list for a key from the dict.  If the key is not in the dict
        an empty list is returned.

        .. versionchanged:: 0.5
           If the key does no longer exist a list is returned instead of
           raising an error.
        """
        return super().pop(key, [])  # type: ignore[return-value]

    def popitemlist(self) -> tuple[K, list[V]]:
        """Pop a ``(key, list)`` tuple from the dict."""
        try:
            return super().popitem()  # type: ignore[return-value]
        except KeyError as e:
            raise exceptions.BadRequestKeyError(e.args[0]) from None

    def __copy__(self) -> te.Self:
        return self.copy()

    def __deepcopy__(self, memo: t.Any) -> te.Self:
        return self.deepcopy(memo=memo)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self.items(multi=True))!r})"


class _omd_bucket(t.Generic[K, V]):
    """Wraps values in the :class:`OrderedMultiDict`.  This makes it
    possible to keep an order over multiple different keys.  It requires
    a lot of extra memory and slows down access a lot, but makes it
    possible to access elements in O(1) and iterate in O(n).
    """

    __slots__ = ("prev", "key", "value", "next")

    def __init__(self, omd: _OrderedMultiDict[K, V], key: K, value: V) -> None:
        self.prev: _omd_bucket[K, V] | None = omd._last_bucket
        self.key: K = key
        self.value: V = value
        self.next: _omd_bucket[K, V] | None = None

        if omd._first_bucket is None:
            omd._first_bucket = self
        if omd._last_bucket is not None:
            omd._last_bucket.next = self
        omd._last_bucket = self

    def unlink(self, omd: _OrderedMultiDict[K, V]) -> None:
        if self.prev:
            self.prev.next = self.next
        if self.next:
            self.next.prev = self.prev
        if omd._first_bucket is self:
            omd._first_bucket = self.next
        if omd._last_bucket is self:
            omd._last_bucket = self.prev


class _OrderedMultiDict(MultiDict[K, V]):
    """Works like a regular :class:`MultiDict` but preserves the
    order of the fields.  To convert the ordered multi dict into a
    list you can use the :meth:`items` method and pass it ``multi=True``.

    In general an :class:`OrderedMultiDict` is an order of magnitude
    slower than a :class:`MultiDict`.

    .. admonition:: note

       Due to a limitation in Python you cannot convert an ordered
       multi dict into a regular dict by using ``dict(multidict)``.
       Instead you have to use the :meth:`to_dict` method, otherwise
       the internal bucket objects are exposed.

    .. deprecated:: 3.1
        Will be removed in Werkzeug 3.2. Use ``MultiDict`` instead.
    """

    def __init__(
        self,
        mapping: (
            MultiDict[K, V]
            | cabc.Mapping[K, V | list[V] | tuple[V, ...] | set[V]]
            | cabc.Iterable[tuple[K, V]]
            | None
        ) = None,
    ) -> None:
        import warnings

        warnings.warn(
            "'OrderedMultiDict' is deprecated and will be removed in Werkzeug"
            " 3.2. Use 'MultiDict' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        self._first_bucket: _omd_bucket[K, V] | None = None
        self._last_bucket: _omd_bucket[K, V] | None = None
        if mapping is not None:
            self.update(mapping)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiDict):
            return NotImplemented
        if isinstance(other, _OrderedMultiDict):
            iter1 = iter(self.items(multi=True))
            iter2 = iter(other.items(multi=True))
            try:
                for k1, v1 in iter1:
                    k2, v2 = next(iter2)
                    if k1 != k2 or v1 != v2:
                        return False
            except StopIteration:
                return False
            try:
                next(iter2)
            except StopIteration:
                return True
            return False
        if len(self) != len(other):
            return False
        for key, values in self.lists():
            if other.getlist(key) != values:
                return False
        return True

    __hash__ = None  # type: ignore[assignment]

    def __reduce_ex__(self, protocol: t.SupportsIndex) -> t.Any:
        return type(self), (list(self.items(multi=True)),)

    def __getstate__(self) -> t.Any:
        return list(self.items(multi=True))

    def __setstate__(self, values: t.Any) -> None:
        self.clear()

        for key, value in values:
            self.add(key, value)

    def __getitem__(self, key: K) -> V:
        if key in self:
            return dict.__getitem__(self, key)[0].value  # type: ignore[index,no-any-return]
        raise exceptions.BadRequestKeyError(key)

    def __setitem__(self, key: K, value: V) -> None:
        self.poplist(key)
        self.add(key, value)

    def __delitem__(self, key: K) -> None:
        self.pop(key)

    def keys(self) -> cabc.Iterable[K]:  # type: ignore[override]
        return (key for key, _ in self.items())

    def __iter__(self) -> cabc.Iterator[K]:
        return iter(self.keys())

    def values(self) -> cabc.Iterable[V]:  # type: ignore[override]
        return (value for key, value in self.items())

    def items(self, multi: bool = False) -> cabc.Iterable[tuple[K, V]]:  # type: ignore[override]
        ptr = self._first_bucket
        if multi:
            while ptr is not None:
                yield ptr.key, ptr.value
                ptr = ptr.next
        else:
            returned_keys = set()
            while ptr is not None:
                if ptr.key not in returned_keys:
                    returned_keys.add(ptr.key)
                    yield ptr.key, ptr.value
                ptr = ptr.next

    def lists(self) -> cabc.Iterable[tuple[K, list[V]]]:
        returned_keys = set()
        ptr = self._first_bucket
        while ptr is not None:
            if ptr.key not in returned_keys:
                yield ptr.key, self.getlist(ptr.key)
                returned_keys.add(ptr.key)
            ptr = ptr.next

    def listvalues(self) -> cabc.Iterable[list[V]]:
        for _key, values in self.lists():
            yield values

    def add(self, key: K, value: V) -> None:
        dict.setdefault(self, key, []).append(_omd_bucket(self, key, value))  # type: ignore[arg-type,attr-defined]

    @t.overload
    def getlist(self, key: K) -> list[V]: ...
    @t.overload
    def getlist(self, key: K, type: cabc.Callable[[V], T]) -> list[T]: ...
    def getlist(
        self, key: K, type: cabc.Callable[[V], T] | None = None
    ) -> list[V] | list[T]:
        rv: list[_omd_bucket[K, V]]

        try:
            rv = dict.__getitem__(self, key)  # type: ignore[index]
        except KeyError:
            return []
        if type is None:
            return [x.value for x in rv]
        result = []
        for item in rv:
            try:
                result.append(type(item.value))
            except (ValueError, TypeError):
                pass
        return result

    def setlist(self, key: K, new_list: cabc.Iterable[V]) -> None:
        self.poplist(key)
        for value in new_list:
            self.add(key, value)

    def setlistdefault(self, key: t.Any, default_list: t.Any = None) -> t.NoReturn:
        raise TypeError("setlistdefault is unsupported for ordered multi dicts")

    def update(  # type: ignore[override]
        self,
        mapping: (
            MultiDict[K, V]
            | cabc.Mapping[K, V | list[V] | tuple[V, ...] | set[V]]
            | cabc.Iterable[tuple[K, V]]
        ),
    ) -> None:
        for key, value in iter_multi_items(mapping):
            self.add(key, value)

    def poplist(self, key: K) -> list[V]:
        buckets: cabc.Iterable[_omd_bucket[K, V]] = dict.pop(self, key, ())  # type: ignore[arg-type]
        for bucket in buckets:
            bucket.unlink(self)
        return [x.value for x in buckets]

    @t.overload
    def pop(self, key: K) -> V: ...
    @t.overload
    def pop(self, key: K, default: V) -> V: ...
    @t.overload
    def pop(self, key: K, default: T) -> V | T: ...
    def pop(
        self,
        key: K,
        default: V | T = _missing,  # type: ignore[assignment]
    ) -> V | T:
        buckets: list[_omd_bucket[K, V]]

        try:
            buckets = dict.pop(self, key)  # type: ignore[arg-type]
        except KeyError:
            if default is not _missing:
                return default

            raise exceptions.BadRequestKeyError(key) from None

        for bucket in buckets:
            bucket.unlink(self)

        return buckets[0].value

    def popitem(self) -> tuple[K, V]:
        key: K
        buckets: list[_omd_bucket[K, V]]

        try:
            key, buckets = dict.popitem(self)  # type: ignore[arg-type,assignment]
        except KeyError as e:
            raise exceptions.BadRequestKeyError(e.args[0]) from None

        for bucket in buckets:
            bucket.unlink(self)

        return key, buckets[0].value

    def popitemlist(self) -> tuple[K, list[V]]:
        key: K
        buckets: list[_omd_bucket[K, V]]

        try:
            key, buckets = dict.popitem(self)  # type: ignore[arg-type,assignment]
        except KeyError as e:
            raise exceptions.BadRequestKeyError(e.args[0]) from None

        for bucket in buckets:
            bucket.unlink(self)

        return key, [x.value for x in buckets]


class CombinedMultiDict(ImmutableMultiDictMixin[K, V], MultiDict[K, V]):  # type: ignore[misc]
    """A read only :class:`MultiDict` that you can pass multiple :class:`MultiDict`
    instances as sequence and it will combine the return values of all wrapped
    dicts:

    >>> from werkzeug.datastructures import CombinedMultiDict, MultiDict
    >>> post = MultiDict([('foo', 'bar')])
    >>> get = MultiDict([('blub', 'blah')])
    >>> combined = CombinedMultiDict([get, post])
    >>> combined['foo']
    'bar'
    >>> combined['blub']
    'blah'

    This works for all read operations and will raise a `TypeError` for
    methods that usually change data which isn't possible.

    From Werkzeug 0.3 onwards, the `KeyError` raised by this class is also a
    subclass of the :exc:`~exceptions.BadRequest` HTTP exception and will
    render a page for a ``400 BAD REQUEST`` if caught in a catch-all for HTTP
    exceptions.
    """

    def __reduce_ex__(self, protocol: t.SupportsIndex) -> t.Any:
        return type(self), (self.dicts,)

    def __init__(self, dicts: cabc.Iterable[MultiDict[K, V]] | None = None) -> None:
        super().__init__()
        self.dicts: list[MultiDict[K, V]] = list(dicts or ())

    @classmethod
    def fromkeys(cls, keys: t.Any, value: t.Any = None) -> t.NoReturn:
        raise TypeError(f"cannot create {cls.__name__!r} instances by fromkeys")

    def __getitem__(self, key: K) -> V:
        for d in self.dicts:
            if key in d:
                return d[key]
        raise exceptions.BadRequestKeyError(key)

    @t.overload  # type: ignore[override]
    def get(self, key: K) -> V | None: ...
    @t.overload
    def get(self, key: K, default: V) -> V: ...
    @t.overload
    def get(self, key: K, default: T) -> V | T: ...
    @t.overload
    def get(self, key: str, type: cabc.Callable[[V], T]) -> T | None: ...
    @t.overload
    def get(self, key: str, default: T, type: cabc.Callable[[V], T]) -> T: ...
    def get(  # type: ignore[misc]
        self,
        key: K,
        default: V | T | None = None,
        type: cabc.Callable[[V], T] | None = None,
    ) -> V | T | None:
        for d in self.dicts:
            if key in d:
                if type is not None:
                    try:
                        return type(d[key])
                    except (ValueError, TypeError):
                        continue
                return d[key]
        return default

    @t.overload
    def getlist(self, key: K) -> list[V]: ...
    @t.overload
    def getlist(self, key: K, type: cabc.Callable[[V], T]) -> list[T]: ...
    def getlist(
        self, key: K, type: cabc.Callable[[V], T] | None = None
    ) -> list[V] | list[T]:
        rv = []
        for d in self.dicts:
            rv.extend(d.getlist(key, type))  # type: ignore[arg-type]
        return rv

    def _keys_impl(self) -> set[K]:
        """This function exists so __len__ can be implemented more efficiently,
        saving one list creation from an iterator.
        """
        return set(k for d in self.dicts for k in d)

    def keys(self) -> cabc.Iterable[K]:  # type: ignore[override]
        return self._keys_impl()

    def __iter__(self) -> cabc.Iterator[K]:
        return iter(self._keys_impl())

    @t.overload  # type: ignore[override]
    def items(self) -> cabc.Iterable[tuple[K, V]]: ...
    @t.overload
    def items(self, multi: t.Literal[True]) -> cabc.Iterable[tuple[K, list[V]]]: ...
    def items(
        self, multi: bool = False
    ) -> cabc.Iterable[tuple[K, V]] | cabc.Iterable[tuple[K, list[V]]]:
        found = set()
        for d in self.dicts:
            for key, value in d.items(multi):
                if multi:
                    yield key, value
                elif key not in found:
                    found.add(key)
                    yield key, value

    def values(self) -> cabc.Iterable[V]:  # type: ignore[override]
        for _, value in self.items():
            yield value

    def lists(self) -> cabc.Iterable[tuple[K, list[V]]]:
        rv: dict[K, list[V]] = {}
        for d in self.dicts:
            for key, values in d.lists():
                rv.setdefault(key, []).extend(values)
        return rv.items()

    def listvalues(self) -> cabc.Iterable[list[V]]:
        return (x[1] for x in self.lists())

    def copy(self) -> MultiDict[K, V]:  # type: ignore[override]
        """Return a shallow mutable copy of this object.

        This returns a :class:`MultiDict` representing the data at the
        time of copying. The copy will no longer reflect changes to the
        wrapped dicts.

        .. versionchanged:: 0.15
            Return a mutable :class:`MultiDict`.
        """
        return MultiDict(self)

    def __len__(self) -> int:
        return len(self._keys_impl())

    def __contains__(self, key: K) -> bool:  # type: ignore[override]
        for d in self.dicts:
            if key in d:
                return True
        return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.dicts!r})"


class ImmutableDict(ImmutableDictMixin[K, V], dict[K, V]):  # type: ignore[misc]
    """An immutable :class:`dict`.

    .. versionadded:: 0.5
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict.__repr__(self)})"

    def copy(self) -> dict[K, V]:
        """Return a shallow mutable copy of this object.  Keep in mind that
        the standard library's :func:`copy` function is a no-op for this class
        like for any other python immutable type (eg: :class:`tuple`).
        """
        return dict(self)

    def __copy__(self) -> te.Self:
        return self


class ImmutableMultiDict(ImmutableMultiDictMixin[K, V], MultiDict[K, V]):  # type: ignore[misc]
    """An immutable :class:`MultiDict`.

    .. versionadded:: 0.5
    """

    def copy(self) -> MultiDict[K, V]:  # type: ignore[override]
        """Return a shallow mutable copy of this object.  Keep in mind that
        the standard library's :func:`copy` function is a no-op for this class
        like for any other python immutable type (eg: :class:`tuple`).
        """
        return MultiDict(self)

    def __copy__(self) -> te.Self:
        return self


class _ImmutableOrderedMultiDict(  # type: ignore[misc]
    ImmutableMultiDictMixin[K, V], _OrderedMultiDict[K, V]
):
    """An immutable :class:`OrderedMultiDict`.

    .. deprecated:: 3.1
        Will be removed in Werkzeug 3.2. Use ``ImmutableMultiDict`` instead.

    .. versionadded:: 0.6
    """

    def __init__(
        self,
        mapping: (
            MultiDict[K, V]
            | cabc.Mapping[K, V | list[V] | tuple[V, ...] | set[V]]
            | cabc.Iterable[tuple[K, V]]
            | None
        ) = None,
    ) -> None:
        super().__init__()

        if mapping is not None:
            for k, v in iter_multi_items(mapping):
                _OrderedMultiDict.add(self, k, v)

    def _iter_hashitems(self) -> cabc.Iterable[t.Any]:
        return enumerate(self.items(multi=True))

    def copy(self) -> _OrderedMultiDict[K, V]:  # type: ignore[override]
        """Return a shallow mutable copy of this object.  Keep in mind that
        the standard library's :func:`copy` function is a no-op for this class
        like for any other python immutable type (eg: :class:`tuple`).
        """
        return _OrderedMultiDict(self)

    def __copy__(self) -> te.Self:
        return self


class CallbackDict(UpdateDictMixin[K, V], dict[K, V]):
    """A dict that calls a function passed every time something is changed.
    The function is passed the dict instance.
    """

    def __init__(
        self,
        initial: cabc.Mapping[K, V] | cabc.Iterable[tuple[K, V]] | None = None,
        on_update: cabc.Callable[[te.Self], None] | None = None,
    ) -> None:
        if initial is None:
            super().__init__()
        else:
            super().__init__(initial)

        self.on_update = on_update

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {super().__repr__()}>"


class HeaderSet(cabc.MutableSet[str]):
    """Similar to the :class:`ETags` class this implements a set-like structure.
    Unlike :class:`ETags` this is case insensitive and used for vary, allow, and
    content-language headers.

    If not constructed using the :func:`parse_set_header` function the
    instantiation works like this:

    >>> hs = HeaderSet(['foo', 'bar', 'baz'])
    >>> hs
    HeaderSet(['foo', 'bar', 'baz'])
    """

    def __init__(
        self,
        headers: cabc.Iterable[str] | None = None,
        on_update: cabc.Callable[[te.Self], None] | None = None,
    ) -> None:
        self._headers = list(headers or ())
        self._set = {x.lower() for x in self._headers}
        self.on_update = on_update

    def add(self, header: str) -> None:
        """Add a new header to the set."""
        self.update((header,))

    def remove(self: te.Self, header: str) -> None:
        """Remove a header from the set.  This raises an :exc:`KeyError` if the
        header is not in the set.

        .. versionchanged:: 0.5
            In older versions a :exc:`IndexError` was raised instead of a
            :exc:`KeyError` if the object was missing.

        :param header: the header to be removed.
        """
        key = header.lower()
        if key not in self._set:
            raise KeyError(header)
        self._set.remove(key)
        for idx, key in enumerate(self._headers):
            if key.lower() == header:
                del self._headers[idx]
                break
        if self.on_update is not None:
            self.on_update(self)

    def update(self: te.Self, iterable: cabc.Iterable[str]) -> None:
        """Add all the headers from the iterable to the set.

        :param iterable: updates the set with the items from the iterable.
        """
        inserted_any = False
        for header in iterable:
            key = header.lower()
            if key not in self._set:
                self._headers.append(header)
                self._set.add(key)
                inserted_any = True
        if inserted_any and self.on_update is not None:
            self.on_update(self)

    def discard(self, header: str) -> None:
        """Like :meth:`remove` but ignores errors.

        :param header: the header to be discarded.
        """
        try:
            self.remove(header)
        except KeyError:
            pass

    def find(self, header: str) -> int:
        """Return the index of the header in the set or return -1 if not found.

        :param header: the header to be looked up.
        """
        header = header.lower()
        for idx, item in enumerate(self._headers):
            if item.lower() == header:
                return idx
        return -1

    def index(self, header: str) -> int:
        """Return the index of the header in the set or raise an
        :exc:`IndexError`.

        :param header: the header to be looked up.
        """
        rv = self.find(header)
        if rv < 0:
            raise IndexError(header)
        return rv

    def clear(self: te.Self) -> None:
        """Clear the set."""
        self._set.clear()
        self._headers.clear()

        if self.on_update is not None:
            self.on_update(self)

    def as_set(self, preserve_casing: bool = False) -> set[str]:
        """Return the set as real python set type.  When calling this, all
        the items are converted to lowercase and the ordering is lost.

        :param preserve_casing: if set to `True` the items in the set returned
                                will have the original case like in the
                                :class:`HeaderSet`, otherwise they will
                                be lowercase.
        """
        if preserve_casing:
            return set(self._headers)
        return set(self._set)

    def to_header(self) -> str:
        """Convert the header set into an HTTP header string."""
        return ", ".join(map(http.quote_header_value, self._headers))

    def __getitem__(self, idx: t.SupportsIndex) -> str:
        return self._headers[idx]

    def __delitem__(self: te.Self, idx: t.SupportsIndex) -> None:
        rv = self._headers.pop(idx)
        self._set.remove(rv.lower())
        if self.on_update is not None:
            self.on_update(self)

    def __setitem__(self: te.Self, idx: t.SupportsIndex, value: str) -> None:
        old = self._headers[idx]
        self._set.remove(old.lower())
        self._headers[idx] = value
        self._set.add(value.lower())
        if self.on_update is not None:
            self.on_update(self)

    def __contains__(self, header: str) -> bool:  # type: ignore[override]
        return header.lower() in self._set

    def __len__(self) -> int:
        return len(self._set)

    def __iter__(self) -> cabc.Iterator[str]:
        return iter(self._headers)

    def __bool__(self) -> bool:
        return bool(self._set)

    def __str__(self) -> str:
        return self.to_header()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._headers!r})"


# circular dependencies
from .. import http


def __getattr__(name: str) -> t.Any:
    import warnings

    if name == "OrderedMultiDict":
        warnings.warn(
            "'OrderedMultiDict' is deprecated and will be removed in Werkzeug"
            " 3.2. Use 'MultiDict' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _OrderedMultiDict

    if name == "ImmutableOrderedMultiDict":
        warnings.warn(
            "'ImmutableOrderedMultiDict' is deprecated and will be removed in"
            " Werkzeug 3.2. Use 'ImmutableMultiDict' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _ImmutableOrderedMultiDict

    raise AttributeError(name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\plot.py ===
"""Plotting module for SymPy.

A plot is represented by the ``Plot`` class that contains a reference to the
backend and a list of the data series to be plotted. The data series are
instances of classes meant to simplify getting points and meshes from SymPy
expressions. ``plot_backends`` is a dictionary with all the backends.

This module gives only the essential. For all the fancy stuff use directly
the backend. You can get the backend wrapper for every plot from the
``_backend`` attribute. Moreover the data series classes have various useful
methods like ``get_points``, ``get_meshes``, etc, that may
be useful if you wish to use another plotting library.

Especially if you need publication ready graphs and this module is not enough
for you - just get the ``_backend`` attribute and add whatever you want
directly to it. In the case of matplotlib (the common way to graph data in
python) just copy ``_backend.fig`` which is the figure and ``_backend.ax``
which is the axis and work on them as you would on any other matplotlib object.

Simplicity of code takes much greater importance than performance. Do not use it
if you care at all about performance. A new backend instance is initialized
every time you call ``show()`` and the old one is left to the garbage collector.
"""

from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function, AppliedUndef
from sympy.core.symbol import (Dummy, Symbol, Wild)
from sympy.external import import_module
from sympy.functions import sign
from sympy.plotting.backends.base_backend import Plot
from sympy.plotting.backends.matplotlibbackend import MatplotlibBackend
from sympy.plotting.backends.textbackend import TextBackend
from sympy.plotting.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries,
    ParametricSurfaceSeries, SurfaceOver2DRangeSeries, ContourSeries)
from sympy.plotting.utils import _check_arguments, _plot_sympify
from sympy.tensor.indexed import Indexed
# to maintain back-compatibility
from sympy.plotting.plotgrid import PlotGrid # noqa: F401
from sympy.plotting.series import BaseSeries # noqa: F401
from sympy.plotting.series import Line2DBaseSeries # noqa: F401
from sympy.plotting.series import Line3DBaseSeries # noqa: F401
from sympy.plotting.series import SurfaceBaseSeries # noqa: F401
from sympy.plotting.series import List2DSeries # noqa: F401
from sympy.plotting.series import GenericDataSeries # noqa: F401
from sympy.plotting.series import centers_of_faces # noqa: F401
from sympy.plotting.series import centers_of_segments # noqa: F401
from sympy.plotting.series import flat # noqa: F401
from sympy.plotting.backends.base_backend import unset_show # noqa: F401
from sympy.plotting.backends.matplotlibbackend import _matplotlib_list # noqa: F401
from sympy.plotting.textplot import textplot # noqa: F401


__doctest_requires__ = {
    ('plot3d',
     'plot3d_parametric_line',
     'plot3d_parametric_surface',
     'plot_parametric'): ['matplotlib'],
    # XXX: The plot doctest possibly should not require matplotlib. It fails at
    # plot(x**2, (x, -5, 5)) which should be fine for text backend.
    ('plot',): ['matplotlib'],
}


def _process_summations(sum_bound, *args):
    """Substitute oo (infinity) in the lower/upper bounds of a summation with
    some integer number.

    Parameters
    ==========

    sum_bound : int
        oo will be substituted with this integer number.
    *args : list/tuple
        pre-processed arguments of the form (expr, range, ...)

    Notes
    =====
    Let's consider the following summation: ``Sum(1 / x**2, (x, 1, oo))``.
    The current implementation of lambdify (SymPy 1.12 at the time of
    writing this) will create something of this form:
    ``sum(1 / x**2 for x in range(1, INF))``
    The problem is that ``type(INF)`` is float, while ``range`` requires
    integers: the evaluation fails.
    Instead of modifying ``lambdify`` (which requires a deep knowledge), just
    replace it with some integer number.
    """
    def new_bound(t, bound):
        if (not t.is_number) or t.is_finite:
            return t
        if sign(t) >= 0:
            return bound
        return -bound

    args = list(args)
    expr = args[0]

    # select summations whose lower/upper bound is infinity
    w = Wild("w", properties=[
        lambda t: isinstance(t, Sum),
        lambda t: any((not a[1].is_finite) or (not a[2].is_finite) for i, a in enumerate(t.args) if i > 0)
    ])

    for t in list(expr.find(w)):
        sums_args = list(t.args)
        for i, a in enumerate(sums_args):
            if i > 0:
                sums_args[i] = (a[0], new_bound(a[1], sum_bound),
                    new_bound(a[2], sum_bound))
        s = Sum(*sums_args)
        expr = expr.subs(t, s)
    args[0] = expr
    return args


def _build_line_series(*args, **kwargs):
    """Loop over the provided arguments and create the necessary line series.
    """
    series = []
    sum_bound = int(kwargs.get("sum_bound", 1000))
    for arg in args:
        expr, r, label, rendering_kw = arg
        kw = kwargs.copy()
        if rendering_kw is not None:
            kw["rendering_kw"] = rendering_kw
        # TODO: _process_piecewise check goes here
        if not callable(expr):
            arg = _process_summations(sum_bound, *arg)
        series.append(LineOver1DRangeSeries(*arg[:-1], **kw))
    return series


def _create_series(series_type, plot_expr, **kwargs):
    """Extract the rendering_kw dictionary from the provided arguments and
    create an appropriate data series.
    """
    series = []
    for args in plot_expr:
        kw = kwargs.copy()
        if args[-1] is not None:
            kw["rendering_kw"] = args[-1]
        series.append(series_type(*args[:-1], **kw))
    return series


def _set_labels(series, labels, rendering_kw):
    """Apply the `label` and `rendering_kw` keyword arguments to the series.
    """
    if not isinstance(labels, (list, tuple)):
        labels = [labels]
    if len(labels) > 0:
        if len(labels) == 1 and len(series) > 1:
            # if one label is provided and multiple series are being plotted,
            # set the same label to all data series. It maintains
            # back-compatibility
            labels *= len(series)
        if len(series) != len(labels):
            raise ValueError("The number of labels must be equal to the "
                "number of expressions being plotted.\nReceived "
                f"{len(series)} expressions and {len(labels)} labels")

        for s, l in zip(series, labels):
            s.label = l

    if rendering_kw:
        if isinstance(rendering_kw, dict):
            rendering_kw = [rendering_kw]
        if len(rendering_kw) == 1:
            rendering_kw *= len(series)
        elif len(series) != len(rendering_kw):
            raise ValueError("The number of rendering dictionaries must be "
                "equal to the number of expressions being plotted.\nReceived "
                f"{len(series)} expressions and {len(labels)} labels")
        for s, r in zip(series, rendering_kw):
            s.rendering_kw = r


def plot_factory(*args, **kwargs):
    backend = kwargs.pop("backend", "default")
    if isinstance(backend, str):
        if backend == "default":
            matplotlib = import_module('matplotlib',
                min_module_version='1.1.0', catch=(RuntimeError,))
            if matplotlib:
                return MatplotlibBackend(*args, **kwargs)
            return TextBackend(*args, **kwargs)
        return plot_backends[backend](*args, **kwargs)
    elif (type(backend) == type) and issubclass(backend, Plot):
        return backend(*args, **kwargs)
    else:
        raise TypeError("backend must be either a string or a subclass of ``Plot``.")


plot_backends = {
    'matplotlib': MatplotlibBackend,
    'text': TextBackend,
}


####New API for plotting module ####

# TODO: Add color arrays for plots.
# TODO: Add more plotting options for 3d plots.
# TODO: Adaptive sampling for 3D plots.

def plot(*args, show=True, **kwargs):
    """Plots a function of a single variable as a curve.

    Parameters
    ==========

    args :
        The first argument is the expression representing the function
        of single variable to be plotted.

        The last argument is a 3-tuple denoting the range of the free
        variable. e.g. ``(x, 0, 5)``

        Typical usage examples are in the following:

        - Plotting a single expression with a single range.
            ``plot(expr, range, **kwargs)``
        - Plotting a single expression with the default range (-10, 10).
            ``plot(expr, **kwargs)``
        - Plotting multiple expressions with a single range.
            ``plot(expr1, expr2, ..., range, **kwargs)``
        - Plotting multiple expressions with multiple ranges.
            ``plot((expr1, range1), (expr2, range2), ..., **kwargs)``

        It is best practice to specify range explicitly because default
        range may change in the future if a more advanced default range
        detection algorithm is implemented.

    show : bool, optional
        The default value is set to ``True``. Set show to ``False`` and
        the function will not display the plot. The returned instance of
        the ``Plot`` class can then be used to save or display the plot
        by calling the ``save()`` and ``show()`` methods respectively.

    line_color : string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    label : str, optional
        The label of the expression in the plot. It will be used when
        called with ``legend``. Default is the name of the expression.
        e.g. ``sin(x)``

    xlabel : str or expression, optional
        Label for the x-axis.

    ylabel : str or expression, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}

    xlim : (float, float), optional
        Denotes the x-axis limits, ``(min, max)```.

    ylim : (float, float), optional
        Denotes the y-axis limits, ``(min, max)```.

    annotations : list, optional
        A list of dictionaries specifying the type of annotation
        required. The keys in the dictionary should be equivalent
        to the arguments of the :external:mod:`matplotlib`'s
        :external:meth:`~matplotlib.axes.Axes.annotate` method.

    markers : list, optional
        A list of dictionaries specifying the type the markers required.
        The keys in the dictionary should be equivalent to the arguments
        of the :external:mod:`matplotlib`'s :external:func:`~matplotlib.pyplot.plot()` function
        along with the marker related keyworded arguments.

    rectangles : list, optional
        A list of dictionaries specifying the dimensions of the
        rectangles to be plotted. The keys in the dictionary should be
        equivalent to the arguments of the :external:mod:`matplotlib`'s
        :external:class:`~matplotlib.patches.Rectangle` class.

    fill : dict, optional
        A dictionary specifying the type of color filling required in
        the plot. The keys in the dictionary should be equivalent to the
        arguments of the :external:mod:`matplotlib`'s
        :external:meth:`~matplotlib.axes.Axes.fill_between` method.

    adaptive : bool, optional
        The default value for the ``adaptive`` parameter is now ``False``.
        To enable adaptive sampling, set ``adaptive=True`` and specify ``n`` if uniform sampling is required.

        The plotting uses an adaptive algorithm which samples
        recursively to accurately plot. The adaptive algorithm uses a
        random point near the midpoint of two points that has to be
        further sampled. Hence the same plots can appear slightly
        different.

    depth : int, optional
        Recursion depth of the adaptive algorithm. A depth of value
        `n` samples a maximum of `2^{n}` points.

        If the ``adaptive`` flag is set to ``False``, this will be
        ignored.

    n : int, optional
        Used when the ``adaptive`` is set to ``False``. The function
        is uniformly sampled at ``n`` number of points. If the ``adaptive``
        flag is set to ``True``, this will be ignored.
        This keyword argument replaces ``nb_of_points``, which should be
        considered deprecated.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    Examples
    ========

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> from sympy import symbols
       >>> from sympy.plotting import plot
       >>> x = symbols('x')

    Single Plot

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(x**2, (x, -5, 5))
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-5.0, 5.0)

    Multiple plots with single range.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(x, x**2, x**3, (x, -5, 5))
       Plot object containing:
       [0]: cartesian line: x for x over (-5.0, 5.0)
       [1]: cartesian line: x**2 for x over (-5.0, 5.0)
       [2]: cartesian line: x**3 for x over (-5.0, 5.0)

    Multiple plots with different ranges.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot((x**2, (x, -6, 6)), (x, (x, -5, 5)))
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-6.0, 6.0)
       [1]: cartesian line: x for x over (-5.0, 5.0)

    No adaptive sampling by default. If adaptive sampling is required, set ``adaptive=True``.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(x**2, adaptive=True, n=400)
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-10.0, 10.0)

    See Also
    ========

    Plot, LineOver1DRangeSeries

    """
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 1, 1, **kwargs)
    params = kwargs.get("params", None)
    free = set()
    for p in plot_expr:
        if not isinstance(p[1][0], str):
            free |= {p[1][0]}
        else:
            free |= {Symbol(p[1][0])}
    if params:
        free = free.difference(params.keys())
    x = free.pop() if free else Symbol("x")
    kwargs.setdefault('xlabel', x)
    kwargs.setdefault('ylabel', Function('f')(x))

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _build_line_series(*plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

    plots = plot_factory(*series, **kwargs)
    if show:
        plots.show()
    return plots


def plot_parametric(*args, show=True, **kwargs):
    """
    Plots a 2D parametric curve.

    Parameters
    ==========

    args
        Common specifications are:

        - Plotting a single parametric curve with a range
            ``plot_parametric((expr_x, expr_y), range)``
        - Plotting multiple parametric curves with the same range
            ``plot_parametric((expr_x, expr_y), ..., range)``
        - Plotting multiple parametric curves with different ranges
            ``plot_parametric((expr_x, expr_y, range), ...)``

        ``expr_x`` is the expression representing $x$ component of the
        parametric function.

        ``expr_y`` is the expression representing $y$ component of the
        parametric function.

        ``range`` is a 3-tuple denoting the parameter symbol, start and
        stop. For example, ``(u, 0, 5)``.

        If the range is not specified, then a default range of (-10, 10)
        is used.

        However, if the arguments are specified as
        ``(expr_x, expr_y, range), ...``, you must specify the ranges
        for each expressions manually.

        Default range may change in the future if a more advanced
        algorithm is implemented.

    adaptive : bool, optional
        Specifies whether to use the adaptive sampling or not.

        The default value is set to ``True``. Set adaptive to ``False``
        and specify ``n`` if uniform sampling is required.

    depth :  int, optional
        The recursion depth of the adaptive algorithm. A depth of
        value $n$ samples a maximum of $2^n$ points.

    n : int, optional
        Used when the ``adaptive`` flag is set to ``False``. Specifies the
        number of the points used for the uniform sampling.
        This keyword argument replaces ``nb_of_points``, which should be
        considered deprecated.

    line_color : string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

    label : str, optional
        The label of the expression in the plot. It will be used when
        called with ``legend``. Default is the name of the expression.
        e.g. ``sin(x)``

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}

    xlim : (float, float), optional
        Denotes the x-axis limits, ``(min, max)```.

    ylim : (float, float), optional
        Denotes the y-axis limits, ``(min, max)```.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import plot_parametric, symbols, cos, sin
       >>> u = symbols('u')

    A parametric plot with a single expression:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((cos(u), sin(u)), (u, -5, 5))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-5.0, 5.0)

    A parametric plot with multiple expressions with the same range:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((cos(u), sin(u)), (u, cos(u)), (u, -10, 10))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-10.0, 10.0)
       [1]: parametric cartesian line: (u, cos(u)) for u over (-10.0, 10.0)

    A parametric plot with multiple expressions with different ranges
    for each curve:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((cos(u), sin(u), (u, -5, 5)),
       ...     (cos(u), u, (u, -5, 5)))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-5.0, 5.0)
       [1]: parametric cartesian line: (cos(u), u) for u over (-5.0, 5.0)

    Notes
    =====

    The plotting uses an adaptive algorithm which samples recursively to
    accurately plot the curve. The adaptive algorithm uses a random point
    near the midpoint of two points that has to be further sampled.
    Hence, repeating the same plot command can give slightly different
    results because of the random sampling.

    If there are multiple plots, then the same optional arguments are
    applied to all the plots drawn in the same canvas. If you want to
    set these options separately, you can index the returned ``Plot``
    object and set it.

    For example, when you specify ``line_color`` once, it would be
    applied simultaneously to both series.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> from sympy import pi
        >>> expr1 = (u, cos(2*pi*u)/2 + 1/2)
        >>> expr2 = (u, sin(2*pi*u)/2 + 1/2)
        >>> p = plot_parametric(expr1, expr2, (u, 0, 1), line_color='blue')

    If you want to specify the line color for the specific series, you
    should index each item and apply the property manually.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> p[0].line_color = 'red'
        >>> p.show()

    See Also
    ========

    Plot, Parametric2DLineSeries
    """
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 2, 1, **kwargs)

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(Parametric2DLineSeries, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

    plots = plot_factory(*series, **kwargs)
    if show:
        plots.show()
    return plots


def plot3d_parametric_line(*args, show=True, **kwargs):
    """
    Plots a 3D parametric line plot.

    Usage
    =====

    Single plot:

    ``plot3d_parametric_line(expr_x, expr_y, expr_z, range, **kwargs)``

    If the range is not specified, then a default range of (-10, 10) is used.

    Multiple plots.

    ``plot3d_parametric_line((expr_x, expr_y, expr_z, range), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    expr_x : Expression representing the function along x.

    expr_y : Expression representing the function along y.

    expr_z : Expression representing the function along z.

    range : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the parameter variable, e.g., (u, 0, 5).

    Keyword Arguments
    =================

    Arguments for ``Parametric3DLineSeries`` class.

    n : int
        The range is uniformly sampled at ``n`` number of points.
        This keyword argument replaces ``nb_of_points``, which should be
        considered deprecated.

    Aesthetics:

    line_color : string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

    label : str
        The label to the plot. It will be used when called with ``legend=True``
        to denote the function with the given label in the plot.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class.

    title : str
        Title of the plot.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin
       >>> from sympy.plotting import plot3d_parametric_line
       >>> u = symbols('u')

    Single plot.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_line(cos(u), sin(u), u, (u, -5, 5))
       Plot object containing:
       [0]: 3D parametric cartesian line: (cos(u), sin(u), u) for u over (-5.0, 5.0)


    Multiple plots.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_line((cos(u), sin(u), u, (u, -5, 5)),
       ...     (sin(u), u**2, u, (u, -5, 5)))
       Plot object containing:
       [0]: 3D parametric cartesian line: (cos(u), sin(u), u) for u over (-5.0, 5.0)
       [1]: 3D parametric cartesian line: (sin(u), u**2, u) for u over (-5.0, 5.0)


    See Also
    ========

    Plot, Parametric3DLineSeries

    """
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 3, 1, **kwargs)
    kwargs.setdefault("xlabel", "x")
    kwargs.setdefault("ylabel", "y")
    kwargs.setdefault("zlabel", "z")

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(Parametric3DLineSeries, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

    plots = plot_factory(*series, **kwargs)
    if show:
        plots.show()
    return plots


def _plot3d_plot_contour_helper(Series, *args, **kwargs):
    """plot3d and plot_contour are structurally identical. Let's reduce
    code repetition.
    """
    # NOTE: if this import would be at the top-module level, it would trigger
    # SymPy's optional-dependencies tests to fail.
    from sympy.vector import BaseScalar

    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 1, 2, **kwargs)

    free_x = set()
    free_y = set()
    _types = (Symbol, BaseScalar, Indexed, AppliedUndef)
    for p in plot_expr:
        free_x |= {p[1][0]} if isinstance(p[1][0], _types) else {Symbol(p[1][0])}
        free_y |= {p[2][0]} if isinstance(p[2][0], _types) else {Symbol(p[2][0])}
    x = free_x.pop() if free_x else Symbol("x")
    y = free_y.pop() if free_y else Symbol("y")
    kwargs.setdefault("xlabel", x)
    kwargs.setdefault("ylabel", y)
    kwargs.setdefault("zlabel", Function('f')(x, y))

    # if a polar discretization is requested and automatic labelling has ben
    # applied, hide the labels on the x-y axis.
    if kwargs.get("is_polar", False):
        if callable(kwargs["xlabel"]):
            kwargs["xlabel"] = ""
        if callable(kwargs["ylabel"]):
            kwargs["ylabel"] = ""

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(Series, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)
    plots = plot_factory(*series, **kwargs)
    if kwargs.get("show", True):
        plots.show()
    return plots


def plot3d(*args, show=True, **kwargs):
    """
    Plots a 3D surface plot.

    Usage
    =====

    Single plot

    ``plot3d(expr, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plot with the same range.

    ``plot3d(expr1, expr2, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plots with different ranges.

    ``plot3d((expr1, range_x, range_y), (expr2, range_x, range_y), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    expr : Expression representing the function along x.

    range_x : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the x variable, e.g. (x, 0, 5).

    range_y : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the y variable, e.g. (y, 0, 5).

    Keyword Arguments
    =================

    Arguments for ``SurfaceOver2DRangeSeries`` class:

    n1 : int
        The x range is sampled uniformly at ``n1`` of points.
        This keyword argument replaces ``nb_of_points_x``, which should be
        considered deprecated.

    n2 : int
        The y range is sampled uniformly at ``n2`` of points.
        This keyword argument replaces ``nb_of_points_y``, which should be
        considered deprecated.

    Aesthetics:

    surface_color : Function which returns a float
        Specifies the color for the surface of the plot.
        See :class:`~.Plot` for more details.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class:

    title : str
        Title of the plot.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of the
        overall figure. The default value is set to ``None``, meaning the size will
        be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols
       >>> from sympy.plotting import plot3d
       >>> x, y = symbols('x y')

    Single plot

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d(x*y, (x, -5, 5), (y, -5, 5))
       Plot object containing:
       [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)


    Multiple plots with same range

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d(x*y, -x*y, (x, -5, 5), (y, -5, 5))
       Plot object containing:
       [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)
       [1]: cartesian surface: -x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)


    Multiple plots with different ranges.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d((x**2 + y**2, (x, -5, 5), (y, -5, 5)),
       ...     (x*y, (x, -3, 3), (y, -3, 3)))
       Plot object containing:
       [0]: cartesian surface: x**2 + y**2 for x over (-5.0, 5.0) and y over (-5.0, 5.0)
       [1]: cartesian surface: x*y for x over (-3.0, 3.0) and y over (-3.0, 3.0)


    See Also
    ========

    Plot, SurfaceOver2DRangeSeries

    """
    kwargs.setdefault("show", show)
    return _plot3d_plot_contour_helper(
        SurfaceOver2DRangeSeries, *args, **kwargs)


def plot3d_parametric_surface(*args, show=True, **kwargs):
    """
    Plots a 3D parametric surface plot.

    Explanation
    ===========

    Single plot.

    ``plot3d_parametric_surface(expr_x, expr_y, expr_z, range_u, range_v, **kwargs)``

    If the ranges is not specified, then a default range of (-10, 10) is used.

    Multiple plots.

    ``plot3d_parametric_surface((expr_x, expr_y, expr_z, range_u, range_v), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    expr_x : Expression representing the function along ``x``.

    expr_y : Expression representing the function along ``y``.

    expr_z : Expression representing the function along ``z``.

    range_u : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the u variable, e.g. (u, 0, 5).

    range_v : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the v variable, e.g. (v, 0, 5).

    Keyword Arguments
    =================

    Arguments for ``ParametricSurfaceSeries`` class:

    n1 : int
        The ``u`` range is sampled uniformly at ``n1`` of points.
        This keyword argument replaces ``nb_of_points_u``, which should be
        considered deprecated.

    n2 : int
        The ``v`` range is sampled uniformly at ``n2`` of points.
        This keyword argument replaces ``nb_of_points_v``, which should be
        considered deprecated.

    Aesthetics:

    surface_color : Function which returns a float
        Specifies the color for the surface of the plot. See
        :class:`~Plot` for more details.

    If there are multiple plots, then the same series arguments are applied for
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.


    Arguments for ``Plot`` class:

    title : str
        Title of the plot.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of the
        overall figure. The default value is set to ``None``, meaning the size will
        be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin
       >>> from sympy.plotting import plot3d_parametric_surface
       >>> u, v = symbols('u v')

    Single plot.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_surface(cos(u + v), sin(u - v), u - v,
       ...     (u, -5, 5), (v, -5, 5))
       Plot object containing:
       [0]: parametric cartesian surface: (cos(u + v), sin(u - v), u - v) for u over (-5.0, 5.0) and v over (-5.0, 5.0)


    See Also
    ========

    Plot, ParametricSurfaceSeries

    """

    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 3, 2, **kwargs)
    kwargs.setdefault("xlabel", "x")
    kwargs.setdefault("ylabel", "y")
    kwargs.setdefault("zlabel", "z")

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(ParametricSurfaceSeries, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

    plots = plot_factory(*series, **kwargs)
    if show:
        plots.show()
    return plots

def plot_contour(*args, show=True, **kwargs):
    """
    Draws contour plot of a function

    Usage
    =====

    Single plot

    ``plot_contour(expr, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plot with the same range.

    ``plot_contour(expr1, expr2, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plots with different ranges.

    ``plot_contour((expr1, range_x, range_y), (expr2, range_x, range_y), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    expr : Expression representing the function along x.

    range_x : (:class:`Symbol`, float, float)
        A 3-tuple denoting the range of the x variable, e.g. (x, 0, 5).

    range_y : (:class:`Symbol`, float, float)
        A 3-tuple denoting the range of the y variable, e.g. (y, 0, 5).

    Keyword Arguments
    =================

    Arguments for ``ContourSeries`` class:

    n1 : int
        The x range is sampled uniformly at ``n1`` of points.
        This keyword argument replaces ``nb_of_points_x``, which should be
        considered deprecated.

    n2 : int
        The y range is sampled uniformly at ``n2`` of points.
        This keyword argument replaces ``nb_of_points_y``, which should be
        considered deprecated.

    Aesthetics:

    surface_color : Function which returns a float
        Specifies the color for the surface of the plot. See
        :class:`sympy.plotting.Plot` for more details.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class:

    title : str
        Title of the plot.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    See Also
    ========

    Plot, ContourSeries

    """
    kwargs.setdefault("show", show)
    return _plot3d_plot_contour_helper(ContourSeries, *args, **kwargs)


def check_arguments(args, expr_len, nb_of_free_symbols):
    """
    Checks the arguments and converts into tuples of the
    form (exprs, ranges).

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import cos, sin, symbols
       >>> from sympy.plotting.plot import check_arguments
       >>> x = symbols('x')
       >>> check_arguments([cos(x), sin(x)], 2, 1)
           [(cos(x), sin(x), (x, -10, 10))]

       >>> check_arguments([x, x**2], 1, 1)
           [(x, (x, -10, 10)), (x**2, (x, -10, 10))]
    """
    if not args:
        return []
    if expr_len > 1 and isinstance(args[0], Expr):
        # Multiple expressions same range.
        # The arguments are tuples when the expression length is
        # greater than 1.
        if len(args) < expr_len:
            raise ValueError("len(args) should not be less than expr_len")
        for i in range(len(args)):
            if isinstance(args[i], Tuple):
                break
        else:
            i = len(args) + 1

        exprs = Tuple(*args[:i])
        free_symbols = list(set().union(*[e.free_symbols for e in exprs]))
        if len(args) == expr_len + nb_of_free_symbols:
            #Ranges given
            plots = [exprs + Tuple(*args[expr_len:])]
        else:
            default_range = Tuple(-10, 10)
            ranges = []
            for symbol in free_symbols:
                ranges.append(Tuple(symbol) + default_range)

            for i in range(len(free_symbols) - nb_of_free_symbols):
                ranges.append(Tuple(Dummy()) + default_range)
            plots = [exprs + Tuple(*ranges)]
        return plots

    if isinstance(args[0], Expr) or (isinstance(args[0], Tuple) and
                                     len(args[0]) == expr_len and
                                     expr_len != 3):
        # Cannot handle expressions with number of expression = 3. It is
        # not possible to differentiate between expressions and ranges.
        #Series of plots with same range
        for i in range(len(args)):
            if isinstance(args[i], Tuple) and len(args[i]) != expr_len:
                break
            if not isinstance(args[i], Tuple):
                args[i] = Tuple(args[i])
        else:
            i = len(args) + 1

        exprs = args[:i]
        assert all(isinstance(e, Expr) for expr in exprs for e in expr)
        free_symbols = list(set().union(*[e.free_symbols for expr in exprs
                                        for e in expr]))

        if len(free_symbols) > nb_of_free_symbols:
            raise ValueError("The number of free_symbols in the expression "
                             "is greater than %d" % nb_of_free_symbols)
        if len(args) == i + nb_of_free_symbols and isinstance(args[i], Tuple):
            ranges = Tuple(*list(args[
                           i:i + nb_of_free_symbols]))
            plots = [expr + ranges for expr in exprs]
            return plots
        else:
            # Use default ranges.
            default_range = Tuple(-10, 10)
            ranges = []
            for symbol in free_symbols:
                ranges.append(Tuple(symbol) + default_range)

            for i in range(nb_of_free_symbols - len(free_symbols)):
                ranges.append(Tuple(Dummy()) + default_range)
            ranges = Tuple(*ranges)
            plots = [expr + ranges for expr in exprs]
            return plots

    elif isinstance(args[0], Tuple) and len(args[0]) == expr_len + nb_of_free_symbols:
        # Multiple plots with different ranges.
        for arg in args:
            for i in range(expr_len):
                if not isinstance(arg[i], Expr):
                    raise ValueError("Expected an expression, given %s" %
                                     str(arg[i]))
            for i in range(nb_of_free_symbols):
                if not len(arg[i + expr_len]) == 3:
                    raise ValueError("The ranges should be a tuple of "
                                     "length 3, got %s" % str(arg[i + expr_len]))
        return args

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\radsimp.py ===
from collections import defaultdict

from sympy.core import sympify, S, Mul, Derivative, Pow
from sympy.core.add import _unevaluated_Add, Add
from sympy.core.assumptions import assumptions
from sympy.core.exprtools import Factors, gcd_terms
from sympy.core.function import _mexpand, expand_mul, expand_power_base
from sympy.core.mul import _keep_coeff, _unevaluated_Mul, _mulsort
from sympy.core.numbers import Rational, zoo, nan
from sympy.core.parameters import global_parameters
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.symbol import Dummy, Wild, symbols
from sympy.functions import exp, sqrt, log
from sympy.functions.elementary.complexes import Abs
from sympy.polys import gcd
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.utilities.iterables import iterable, sift




def collect(expr, syms, func=None, evaluate=None, exact=False, distribute_order_term=True):
    """
    Collect additive terms of an expression.

    Explanation
    ===========

    This function collects additive terms of an expression with respect
    to a list of expression up to powers with rational exponents. By the
    term symbol here are meant arbitrary expressions, which can contain
    powers, products, sums etc. In other words symbol is a pattern which
    will be searched for in the expression's terms.

    The input expression is not expanded by :func:`collect`, so user is
    expected to provide an expression in an appropriate form. This makes
    :func:`collect` more predictable as there is no magic happening behind the
    scenes. However, it is important to note, that powers of products are
    converted to products of powers using the :func:`~.expand_power_base`
    function.

    There are two possible types of output. First, if ``evaluate`` flag is
    set, this function will return an expression with collected terms or
    else it will return a dictionary with expressions up to rational powers
    as keys and collected coefficients as values.

    Examples
    ========

    >>> from sympy import S, collect, expand, factor, Wild
    >>> from sympy.abc import a, b, c, x, y

    This function can collect symbolic coefficients in polynomials or
    rational expressions. It will manage to find all integer or rational
    powers of collection variable::

        >>> collect(a*x**2 + b*x**2 + a*x - b*x + c, x)
        c + x**2*(a + b) + x*(a - b)

    The same result can be achieved in dictionary form::

        >>> d = collect(a*x**2 + b*x**2 + a*x - b*x + c, x, evaluate=False)
        >>> d[x**2]
        a + b
        >>> d[x]
        a - b
        >>> d[S.One]
        c

    You can also work with multivariate polynomials. However, remember that
    this function is greedy so it will care only about a single symbol at time,
    in specification order::

        >>> collect(x**2 + y*x**2 + x*y + y + a*y, [x, y])
        x**2*(y + 1) + x*y + y*(a + 1)

    Also more complicated expressions can be used as patterns::

        >>> from sympy import sin, log
        >>> collect(a*sin(2*x) + b*sin(2*x), sin(2*x))
        (a + b)*sin(2*x)

        >>> collect(a*x*log(x) + b*(x*log(x)), x*log(x))
        x*(a + b)*log(x)

    You can use wildcards in the pattern::

        >>> w = Wild('w1')
        >>> collect(a*x**y - b*x**y, w**y)
        x**y*(a - b)

    It is also possible to work with symbolic powers, although it has more
    complicated behavior, because in this case power's base and symbolic part
    of the exponent are treated as a single symbol::

        >>> collect(a*x**c + b*x**c, x)
        a*x**c + b*x**c
        >>> collect(a*x**c + b*x**c, x**c)
        x**c*(a + b)

    However if you incorporate rationals to the exponents, then you will get
    well known behavior::

        >>> collect(a*x**(2*c) + b*x**(2*c), x**c)
        x**(2*c)*(a + b)

    Note also that all previously stated facts about :func:`collect` function
    apply to the exponential function, so you can get::

        >>> from sympy import exp
        >>> collect(a*exp(2*x) + b*exp(2*x), exp(x))
        (a + b)*exp(2*x)

    If you are interested only in collecting specific powers of some symbols
    then set ``exact`` flag to True::

        >>> collect(a*x**7 + b*x**7, x, exact=True)
        a*x**7 + b*x**7
        >>> collect(a*x**7 + b*x**7, x**7, exact=True)
        x**7*(a + b)

    If you want to collect on any object containing symbols, set
    ``exact`` to None:

        >>> collect(x*exp(x) + sin(x)*y + sin(x)*2 + 3*x, x, exact=None)
        x*exp(x) + 3*x + (y + 2)*sin(x)
        >>> collect(a*x*y + x*y + b*x + x, [x, y], exact=None)
        x*y*(a + 1) + x*(b + 1)

    You can also apply this function to differential equations, where
    derivatives of arbitrary order can be collected. Note that if you
    collect with respect to a function or a derivative of a function, all
    derivatives of that function will also be collected. Use
    ``exact=True`` to prevent this from happening::

        >>> from sympy import Derivative as D, collect, Function
        >>> f = Function('f') (x)

        >>> collect(a*D(f,x) + b*D(f,x), D(f,x))
        (a + b)*Derivative(f(x), x)

        >>> collect(a*D(D(f,x),x) + b*D(D(f,x),x), f)
        (a + b)*Derivative(f(x), (x, 2))

        >>> collect(a*D(D(f,x),x) + b*D(D(f,x),x), D(f,x), exact=True)
        a*Derivative(f(x), (x, 2)) + b*Derivative(f(x), (x, 2))

        >>> collect(a*D(f,x) + b*D(f,x) + a*f + b*f, f)
        (a + b)*f(x) + (a + b)*Derivative(f(x), x)

    Or you can even match both derivative order and exponent at the same time::

        >>> collect(a*D(D(f,x),x)**2 + b*D(D(f,x),x)**2, D(f,x))
        (a + b)*Derivative(f(x), (x, 2))**2

    Finally, you can apply a function to each of the collected coefficients.
    For example you can factorize symbolic coefficients of polynomial::

        >>> f = expand((x + a + 1)**3)

        >>> collect(f, x, factor)
        x**3 + 3*x**2*(a + 1) + 3*x*(a + 1)**2 + (a + 1)**3

    .. note:: Arguments are expected to be in expanded form, so you might have
              to call :func:`~.expand` prior to calling this function.

    See Also
    ========

    collect_const, collect_sqrt, rcollect
    """
    expr = sympify(expr)
    syms = [sympify(i) for i in (syms if iterable(syms) else [syms])]

    # replace syms[i] if it is not x, -x or has Wild symbols
    cond = lambda x: x.is_Symbol or (-x).is_Symbol or bool(
        x.atoms(Wild))
    _, nonsyms = sift(syms, cond, binary=True)
    if nonsyms:
        reps = dict(zip(nonsyms, [Dummy(**assumptions(i)) for i in nonsyms]))
        syms = [reps.get(s, s) for s in syms]
        rv = collect(expr.subs(reps), syms,
            func=func, evaluate=evaluate, exact=exact,
            distribute_order_term=distribute_order_term)
        urep = {v: k for k, v in reps.items()}
        if not isinstance(rv, dict):
            return rv.xreplace(urep)
        else:
            return {urep.get(k, k).xreplace(urep): v.xreplace(urep)
                    for k, v in rv.items()}

    # see if other expressions should be considered
    if exact is None:
        _syms = set()
        for i in Add.make_args(expr):
            if not i.has_free(*syms) or i in syms:
                continue
            if not i.is_Mul and i not in syms:
                _syms.add(i)
            else:
                # identify compound generators
                g = i._new_rawargs(*i.as_coeff_mul(*syms)[1])
                if g not in syms:
                    _syms.add(g)
        simple = all(i.is_Pow and i.base in syms for i in _syms)
        syms = syms + list(ordered(_syms))
        if not simple:
            return collect(expr, syms,
            func=func, evaluate=evaluate, exact=False,
            distribute_order_term=distribute_order_term)

    if evaluate is None:
        evaluate = global_parameters.evaluate

    def make_expression(terms):
        product = []

        for term, rat, sym, deriv in terms:
            if deriv is not None:
                var, order = deriv
                for _ in range(order):
                    term = Derivative(term, var)

            if sym is None:
                if rat is S.One:
                    product.append(term)
                else:
                    product.append(Pow(term, rat))
            else:
                product.append(Pow(term, rat*sym))

        return Mul(*product)

    def parse_derivative(deriv):
        # scan derivatives tower in the input expression and return
        # underlying function and maximal differentiation order
        expr, sym, order = deriv.expr, deriv.variables[0], 1

        for s in deriv.variables[1:]:
            if s == sym:
                order += 1
            else:
                raise NotImplementedError(
                    'Improve MV Derivative support in collect')

        while isinstance(expr, Derivative):
            s0 = expr.variables[0]

            if any(s != s0 for s in expr.variables):
                raise NotImplementedError(
                    'Improve MV Derivative support in collect')

            if s0 == sym:
                expr, order = expr.expr, order + len(expr.variables)
            else:
                break

        return expr, (sym, Rational(order))

    def parse_term(expr):
        """Parses expression expr and outputs tuple (sexpr, rat_expo,
        sym_expo, deriv)
        where:
         - sexpr is the base expression
         - rat_expo is the rational exponent that sexpr is raised to
         - sym_expo is the symbolic exponent that sexpr is raised to
         - deriv contains the derivatives of the expression

         For example, the output of x would be (x, 1, None, None)
         the output of 2**x would be (2, 1, x, None).
        """
        rat_expo, sym_expo = S.One, None
        sexpr, deriv = expr, None

        if expr.is_Pow:
            if isinstance(expr.base, Derivative):
                sexpr, deriv = parse_derivative(expr.base)
            else:
                sexpr = expr.base

            if expr.base == S.Exp1:
                arg = expr.exp
                if arg.is_Rational:
                    sexpr, rat_expo = S.Exp1, arg
                elif arg.is_Mul:
                    coeff, tail = arg.as_coeff_Mul(rational=True)
                    sexpr, rat_expo = exp(tail), coeff

            elif expr.exp.is_Number:
                rat_expo = expr.exp
            else:
                coeff, tail = expr.exp.as_coeff_Mul()

                if coeff.is_Number:
                    rat_expo, sym_expo = coeff, tail
                else:
                    sym_expo = expr.exp
        elif isinstance(expr, exp):
            arg = expr.exp
            if arg.is_Rational:
                sexpr, rat_expo = S.Exp1, arg
            elif arg.is_Mul:
                coeff, tail = arg.as_coeff_Mul(rational=True)
                sexpr, rat_expo = exp(tail), coeff
        elif isinstance(expr, Derivative):
            sexpr, deriv = parse_derivative(expr)

        return sexpr, rat_expo, sym_expo, deriv

    def parse_expression(terms, pattern):
        """Parse terms searching for a pattern.
        Terms is a list of tuples as returned by parse_terms;
        Pattern is an expression treated as a product of factors.
        """
        pattern = Mul.make_args(pattern)

        if len(terms) < len(pattern):
            # pattern is longer than matched product
            # so no chance for positive parsing result
            return None
        else:
            pattern = [parse_term(elem) for elem in pattern]

            terms = terms[:]  # need a copy
            elems, common_expo, has_deriv = [], None, False

            for elem, e_rat, e_sym, e_ord in pattern:

                if elem.is_Number and e_rat == 1 and e_sym is None:
                    # a constant is a match for everything
                    continue

                for j in range(len(terms)):
                    if terms[j] is None:
                        continue

                    term, t_rat, t_sym, t_ord = terms[j]

                    # keeping track of whether one of the terms had
                    # a derivative or not as this will require rebuilding
                    # the expression later
                    if t_ord is not None:
                        has_deriv = True

                    if (term.match(elem) is not None and
                            (t_sym == e_sym or t_sym is not None and
                            e_sym is not None and
                            t_sym.match(e_sym) is not None)):
                        if exact is False:
                            # we don't have to be exact so find common exponent
                            # for both expression's term and pattern's element
                            expo = t_rat / e_rat

                            if common_expo is None:
                                # first time
                                common_expo = expo
                            else:
                                # common exponent was negotiated before so
                                # there is no chance for a pattern match unless
                                # common and current exponents are equal
                                if common_expo != expo:
                                    common_expo = 1
                        else:
                            # we ought to be exact so all fields of
                            # interest must match in every details
                            if e_rat != t_rat or e_ord != t_ord:
                                continue

                        # found common term so remove it from the expression
                        # and try to match next element in the pattern
                        elems.append(terms[j])
                        terms[j] = None

                        break

                else:
                    # pattern element not found
                    return None

            return [_f for _f in terms if _f], elems, common_expo, has_deriv

    if evaluate:
        if expr.is_Add:
            o = expr.getO() or 0
            expr = expr.func(*[
                    collect(a, syms, func, True, exact, distribute_order_term)
                    for a in expr.args if a != o]) + o
        elif expr.is_Mul:
            return expr.func(*[
                collect(term, syms, func, True, exact, distribute_order_term)
                for term in expr.args])
        elif expr.is_Pow:
            b = collect(
                expr.base, syms, func, True, exact, distribute_order_term)
            return Pow(b, expr.exp)

    syms = [expand_power_base(i, deep=False) for i in syms]

    order_term = None

    if distribute_order_term:
        order_term = expr.getO()

        if order_term is not None:
            if order_term.has(*syms):
                order_term = None
            else:
                expr = expr.removeO()

    summa = [expand_power_base(i, deep=False) for i in Add.make_args(expr)]

    collected, disliked = defaultdict(list), S.Zero
    for product in summa:
        c, nc = product.args_cnc(split_1=False)
        args = list(ordered(c)) + nc
        terms = [parse_term(i) for i in args]
        small_first = True

        for symbol in syms:
            if isinstance(symbol, Derivative) and small_first:
                terms = list(reversed(terms))
                small_first = not small_first
            result = parse_expression(terms, symbol)

            if result is not None:
                if not symbol.is_commutative:
                    raise AttributeError("Can not collect noncommutative symbol")

                terms, elems, common_expo, has_deriv = result

                # when there was derivative in current pattern we
                # will need to rebuild its expression from scratch
                if not has_deriv:
                    margs = []
                    for elem in elems:
                        if elem[2] is None:
                            e = elem[1]
                        else:
                            e = elem[1]*elem[2]
                        margs.append(Pow(elem[0], e))
                    index = Mul(*margs)
                else:
                    index = make_expression(elems)
                terms = expand_power_base(make_expression(terms), deep=False)
                index = expand_power_base(index, deep=False)
                collected[index].append(terms)
                break
        else:
            # none of the patterns matched
            disliked += product
    # add terms now for each key
    collected = {k: Add(*v) for k, v in collected.items()}

    if disliked is not S.Zero:
        collected[S.One] = disliked

    if order_term is not None:
        for key, val in collected.items():
            collected[key] = val + order_term

    if func is not None:
        collected = {
            key: func(val) for key, val in collected.items()}

    if evaluate:
        return Add(*[key*val for key, val in collected.items()])
    else:
        return collected


def rcollect(expr, *vars):
    """
    Recursively collect sums in an expression.

    Examples
    ========

    >>> from sympy.simplify import rcollect
    >>> from sympy.abc import x, y

    >>> expr = (x**2*y + x*y + x + y)/(x + y)

    >>> rcollect(expr, y)
    (x + y*(x**2 + x + 1))/(x + y)

    See Also
    ========

    collect, collect_const, collect_sqrt
    """
    if expr.is_Atom or not expr.has(*vars):
        return expr
    else:
        expr = expr.__class__(*[rcollect(arg, *vars) for arg in expr.args])

        if expr.is_Add:
            return collect(expr, vars)
        else:
            return expr


def collect_sqrt(expr, evaluate=None):
    """Return expr with terms having common square roots collected together.
    If ``evaluate`` is False a count indicating the number of sqrt-containing
    terms will be returned and, if non-zero, the terms of the Add will be
    returned, else the expression itself will be returned as a single term.
    If ``evaluate`` is True, the expression with any collected terms will be
    returned.

    Note: since I = sqrt(-1), it is collected, too.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.radsimp import collect_sqrt
    >>> from sympy.abc import a, b

    >>> r2, r3, r5 = [sqrt(i) for i in [2, 3, 5]]
    >>> collect_sqrt(a*r2 + b*r2)
    sqrt(2)*(a + b)
    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r3)
    sqrt(2)*(a + b) + sqrt(3)*(a + b)
    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r5)
    sqrt(3)*a + sqrt(5)*b + sqrt(2)*(a + b)

    If evaluate is False then the arguments will be sorted and
    returned as a list and a count of the number of sqrt-containing
    terms will be returned:

    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r5, evaluate=False)
    ((sqrt(3)*a, sqrt(5)*b, sqrt(2)*(a + b)), 3)
    >>> collect_sqrt(a*sqrt(2) + b, evaluate=False)
    ((b, sqrt(2)*a), 1)
    >>> collect_sqrt(a + b, evaluate=False)
    ((a + b,), 0)

    See Also
    ========

    collect, collect_const, rcollect
    """
    if evaluate is None:
        evaluate = global_parameters.evaluate
    # this step will help to standardize any complex arguments
    # of sqrts
    coeff, expr = expr.as_content_primitive()
    vars = set()
    for a in Add.make_args(expr):
        for m in a.args_cnc()[0]:
            if m.is_number and (
                    m.is_Pow and m.exp.is_Rational and m.exp.q == 2 or
                    m is S.ImaginaryUnit):
                vars.add(m)

    # we only want radicals, so exclude Number handling; in this case
    # d will be evaluated
    d = collect_const(expr, *vars, Numbers=False)
    hit = expr != d

    if not evaluate:
        nrad = 0
        # make the evaluated args canonical
        args = list(ordered(Add.make_args(d)))
        for i, m in enumerate(args):
            c, nc = m.args_cnc()
            for ci in c:
                # XXX should this be restricted to ci.is_number as above?
                if ci.is_Pow and ci.exp.is_Rational and ci.exp.q == 2 or \
                        ci is S.ImaginaryUnit:
                    nrad += 1
                    break
            args[i] *= coeff
        if not (hit or nrad):
            args = [Add(*args)]
        return tuple(args), nrad

    return coeff*d


def collect_abs(expr):
    """Return ``expr`` with arguments of multiple Abs in a term collected
    under a single instance.

    Examples
    ========

    >>> from sympy.simplify.radsimp import collect_abs
    >>> from sympy.abc import x
    >>> collect_abs(abs(x + 1)/abs(x**2 - 1))
    Abs((x + 1)/(x**2 - 1))
    >>> collect_abs(abs(1/x))
    Abs(1/x)
    """
    def _abs(mul):
        c, nc = mul.args_cnc()
        a = []
        o = []
        for i in c:
            if isinstance(i, Abs):
                a.append(i.args[0])
            elif isinstance(i, Pow) and isinstance(i.base, Abs) and i.exp.is_real:
                a.append(i.base.args[0]**i.exp)
            else:
                o.append(i)
        if len(a) < 2 and not any(i.exp.is_negative for i in a if isinstance(i, Pow)):
            return mul
        absarg = Mul(*a)
        A = Abs(absarg)
        args = [A]
        args.extend(o)
        if not A.has(Abs):
            args.extend(nc)
            return Mul(*args)
        if not isinstance(A, Abs):
            # reevaluate and make it unevaluated
            A = Abs(absarg, evaluate=False)
        args[0] = A
        _mulsort(args)
        args.extend(nc)  # nc always go last
        return Mul._from_args(args, is_commutative=not nc)

    return expr.replace(
        lambda x: isinstance(x, Mul),
        lambda x: _abs(x)).replace(
            lambda x: isinstance(x, Pow),
            lambda x: _abs(x))


def collect_const(expr, *vars, Numbers=True):
    """A non-greedy collection of terms with similar number coefficients in
    an Add expr. If ``vars`` is given then only those constants will be
    targeted. Although any Number can also be targeted, if this is not
    desired set ``Numbers=False`` and no Float or Rational will be collected.

    Parameters
    ==========

    expr : SymPy expression
        This parameter defines the expression the expression from which
        terms with similar coefficients are to be collected. A non-Add
        expression is returned as it is.

    vars : variable length collection of Numbers, optional
        Specifies the constants to target for collection. Can be multiple in
        number.

    Numbers : bool
        Specifies to target all instance of
        :class:`sympy.core.numbers.Number` class. If ``Numbers=False``, then
        no Float or Rational will be collected.

    Returns
    =======

    expr : Expr
        Returns an expression with similar coefficient terms collected.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.abc import s, x, y, z
    >>> from sympy.simplify.radsimp import collect_const
    >>> collect_const(sqrt(3) + sqrt(3)*(1 + sqrt(2)))
    sqrt(3)*(sqrt(2) + 2)
    >>> collect_const(sqrt(3)*s + sqrt(7)*s + sqrt(3) + sqrt(7))
    (sqrt(3) + sqrt(7))*(s + 1)
    >>> s = sqrt(2) + 2
    >>> collect_const(sqrt(3)*s + sqrt(3) + sqrt(7)*s + sqrt(7))
    (sqrt(2) + 3)*(sqrt(3) + sqrt(7))
    >>> collect_const(sqrt(3)*s + sqrt(3) + sqrt(7)*s + sqrt(7), sqrt(3))
    sqrt(7) + sqrt(3)*(sqrt(2) + 3) + sqrt(7)*(sqrt(2) + 2)

    The collection is sign-sensitive, giving higher precedence to the
    unsigned values:

    >>> collect_const(x - y - z)
    x - (y + z)
    >>> collect_const(-y - z)
    -(y + z)
    >>> collect_const(2*x - 2*y - 2*z, 2)
    2*(x - y - z)
    >>> collect_const(2*x - 2*y - 2*z, -2)
    2*x - 2*(y + z)

    See Also
    ========

    collect, collect_sqrt, rcollect
    """
    if not expr.is_Add:
        return expr

    recurse = False

    if not vars:
        recurse = True
        vars = set()
        for a in expr.args:
            for m in Mul.make_args(a):
                if m.is_number:
                    vars.add(m)
    else:
        vars = sympify(vars)
    if not Numbers:
        vars = [v for v in vars if not v.is_Number]

    vars = list(ordered(vars))
    for v in vars:
        terms = defaultdict(list)
        Fv = Factors(v)
        for m in Add.make_args(expr):
            f = Factors(m)
            q, r = f.div(Fv)
            if r.is_one:
                # only accept this as a true factor if
                # it didn't change an exponent from an Integer
                # to a non-Integer, e.g. 2/sqrt(2) -> sqrt(2)
                # -- we aren't looking for this sort of change
                fwas = f.factors.copy()
                fnow = q.factors
                if not any(k in fwas and fwas[k].is_Integer and not
                        fnow[k].is_Integer for k in fnow):
                    terms[v].append(q.as_expr())
                    continue
            terms[S.One].append(m)

        args = []
        hit = False
        uneval = False
        for k in ordered(terms):
            v = terms[k]
            if k is S.One:
                args.extend(v)
                continue

            if len(v) > 1:
                v = Add(*v)
                hit = True
                if recurse and v != expr:
                    vars.append(v)
            else:
                v = v[0]

            # be careful not to let uneval become True unless
            # it must be because it's going to be more expensive
            # to rebuild the expression as an unevaluated one
            if Numbers and k.is_Number and v.is_Add:
                args.append(_keep_coeff(k, v, sign=True))
                uneval = True
            else:
                args.append(k*v)

        if hit:
            if uneval:
                expr = _unevaluated_Add(*args)
            else:
                expr = Add(*args)
            if not expr.is_Add:
                break

    return expr


def radsimp(expr, symbolic=True, max_terms=4):
    r"""
    Rationalize the denominator by removing square roots.

    Explanation
    ===========

    The expression returned from radsimp must be used with caution
    since if the denominator contains symbols, it will be possible to make
    substitutions that violate the assumptions of the simplification process:
    that for a denominator matching a + b*sqrt(c), a != +/-b*sqrt(c). (If
    there are no symbols, this assumptions is made valid by collecting terms
    of sqrt(c) so the match variable ``a`` does not contain ``sqrt(c)``.) If
    you do not want the simplification to occur for symbolic denominators, set
    ``symbolic`` to False.

    If there are more than ``max_terms`` radical terms then the expression is
    returned unchanged.

    Examples
    ========

    >>> from sympy import radsimp, sqrt, Symbol, pprint
    >>> from sympy import factor_terms, fraction, signsimp
    >>> from sympy.simplify.radsimp import collect_sqrt
    >>> from sympy.abc import a, b, c

    >>> radsimp(1/(2 + sqrt(2)))
    (2 - sqrt(2))/2
    >>> x,y = map(Symbol, 'xy')
    >>> e = ((2 + 2*sqrt(2))*x + (2 + sqrt(8))*y)/(2 + sqrt(2))
    >>> radsimp(e)
    sqrt(2)*(x + y)

    No simplification beyond removal of the gcd is done. One might
    want to polish the result a little, however, by collecting
    square root terms:

    >>> r2 = sqrt(2)
    >>> r5 = sqrt(5)
    >>> ans = radsimp(1/(y*r2 + x*r2 + a*r5 + b*r5)); pprint(ans)
        ___       ___       ___       ___
      \/ 5 *a + \/ 5 *b - \/ 2 *x - \/ 2 *y
    ------------------------------------------
       2               2      2              2
    5*a  + 10*a*b + 5*b  - 2*x  - 4*x*y - 2*y

    >>> n, d = fraction(ans)
    >>> pprint(factor_terms(signsimp(collect_sqrt(n))/d, radical=True))
            ___             ___
          \/ 5 *(a + b) - \/ 2 *(x + y)
    ------------------------------------------
       2               2      2              2
    5*a  + 10*a*b + 5*b  - 2*x  - 4*x*y - 2*y

    If radicals in the denominator cannot be removed or there is no denominator,
    the original expression will be returned.

    >>> radsimp(sqrt(2)*x + sqrt(2))
    sqrt(2)*x + sqrt(2)

    Results with symbols will not always be valid for all substitutions:

    >>> eq = 1/(a + b*sqrt(c))
    >>> eq.subs(a, b*sqrt(c))
    1/(2*b*sqrt(c))
    >>> radsimp(eq).subs(a, b*sqrt(c))
    nan

    If ``symbolic=False``, symbolic denominators will not be transformed (but
    numeric denominators will still be processed):

    >>> radsimp(eq, symbolic=False)
    1/(a + b*sqrt(c))

    """
    from sympy.core.expr import Expr
    from sympy.simplify.simplify import signsimp

    syms = symbols("a:d A:D")
    def _num(rterms):
        # return the multiplier that will simplify the expression described
        # by rterms [(sqrt arg, coeff), ... ]
        a, b, c, d, A, B, C, D = syms
        if len(rterms) == 2:
            reps = dict(list(zip([A, a, B, b], [j for i in rterms for j in i])))
            return (
            sqrt(A)*a - sqrt(B)*b).xreplace(reps)
        if len(rterms) == 3:
            reps = dict(list(zip([A, a, B, b, C, c], [j for i in rterms for j in i])))
            return (
            (sqrt(A)*a + sqrt(B)*b - sqrt(C)*c)*(2*sqrt(A)*sqrt(B)*a*b - A*a**2 -
            B*b**2 + C*c**2)).xreplace(reps)
        elif len(rterms) == 4:
            reps = dict(list(zip([A, a, B, b, C, c, D, d], [j for i in rterms for j in i])))
            return ((sqrt(A)*a + sqrt(B)*b - sqrt(C)*c - sqrt(D)*d)*(2*sqrt(A)*sqrt(B)*a*b
                - A*a**2 - B*b**2 - 2*sqrt(C)*sqrt(D)*c*d + C*c**2 +
                D*d**2)*(-8*sqrt(A)*sqrt(B)*sqrt(C)*sqrt(D)*a*b*c*d + A**2*a**4 -
                2*A*B*a**2*b**2 - 2*A*C*a**2*c**2 - 2*A*D*a**2*d**2 + B**2*b**4 -
                2*B*C*b**2*c**2 - 2*B*D*b**2*d**2 + C**2*c**4 - 2*C*D*c**2*d**2 +
                D**2*d**4)).xreplace(reps)
        elif len(rterms) == 1:
            return sqrt(rterms[0][0])
        else:
            raise NotImplementedError

    def ispow2(d, log2=False):
        if not d.is_Pow:
            return False
        e = d.exp
        if e.is_Rational and e.q == 2 or symbolic and denom(e) == 2:
            return True
        if log2:
            q = 1
            if e.is_Rational:
                q = e.q
            elif symbolic:
                d = denom(e)
                if d.is_Integer:
                    q = d
            if q != 1 and log(q, 2).is_Integer:
                return True
        return False

    def handle(expr):
        # Handle first reduces to the case
        # expr = 1/d, where d is an add, or d is base**p/2.
        # We do this by recursively calling handle on each piece.
        from sympy.simplify.simplify import nsimplify

        if expr.is_Atom:
            return expr
        elif not isinstance(expr, Expr):
            return expr.func(*[handle(a) for a in expr.args])

        n, d = fraction(expr)

        if d.is_Atom and n.is_Atom:
            return expr
        elif not n.is_Atom:
            n = n.func(*[handle(a) for a in n.args])
            return _unevaluated_Mul(n, handle(1/d))
        elif n is not S.One:
            return _unevaluated_Mul(n, handle(1/d))
        elif d.is_Mul:
            return _unevaluated_Mul(*[handle(1/d) for d in d.args])

        # By this step, expr is 1/d, and d is not a mul.
        if not symbolic and d.free_symbols:
            return expr

        if ispow2(d):
            d2 = sqrtdenest(sqrt(d.base))**numer(d.exp)
            if d2 != d:
                return handle(1/d2)
        elif d.is_Pow and (d.exp.is_integer or d.base.is_positive):
            # (1/d**i) = (1/d)**i
            return handle(1/d.base)**d.exp

        if not (d.is_Add or ispow2(d)):
            return 1/d.func(*[handle(a) for a in d.args])

        # handle 1/d treating d as an Add (though it may not be)

        keep = True  # keep changes that are made

        # flatten it and collect radicals after checking for special
        # conditions
        d = _mexpand(d)

        # did it change?
        if d.is_Atom:
            return 1/d

        # is it a number that might be handled easily?
        if d.is_number:
            _d = nsimplify(d)
            if _d.is_Number and _d.equals(d):
                return 1/_d

        while True:
            # collect similar terms
            collected = defaultdict(list)
            for m in Add.make_args(d):  # d might have become non-Add
                p2 = []
                other = []
                for i in Mul.make_args(m):
                    if ispow2(i, log2=True):
                        p2.append(i.base if i.exp is S.Half else i.base**(2*i.exp))
                    elif i is S.ImaginaryUnit:
                        p2.append(S.NegativeOne)
                    else:
                        other.append(i)
                collected[tuple(ordered(p2))].append(Mul(*other))
            rterms = list(ordered(list(collected.items())))
            rterms = [(Mul(*i), Add(*j)) for i, j in rterms]
            nrad = len(rterms) - (1 if rterms[0][0] is S.One else 0)
            if nrad < 1:
                break
            elif nrad > max_terms:
                # there may have been invalid operations leading to this point
                # so don't keep changes, e.g. this expression is troublesome
                # in collecting terms so as not to raise the issue of 2834:
                # r = sqrt(sqrt(5) + 5)
                # eq = 1/(sqrt(5)*r + 2*sqrt(5)*sqrt(-sqrt(5) + 5) + 5*r)
                keep = False
                break
            if len(rterms) > 4:
                # in general, only 4 terms can be removed with repeated squaring
                # but other considerations can guide selection of radical terms
                # so that radicals are removed
                if all(x.is_Integer and (y**2).is_Rational for x, y in rterms):
                    nd, d = rad_rationalize(S.One, Add._from_args(
                        [sqrt(x)*y for x, y in rterms]))
                    n *= nd
                else:
                    # is there anything else that might be attempted?
                    keep = False
                break
            from sympy.simplify.powsimp import powsimp, powdenest

            num = powsimp(_num(rterms))
            n *= num
            d *= num
            d = powdenest(_mexpand(d), force=symbolic)
            if d.has(S.Zero, nan, zoo):
                return expr
            if d.is_Atom:
                break

        if not keep:
            return expr
        return _unevaluated_Mul(n, 1/d)

    if not isinstance(expr, Expr):
        return expr.func(*[radsimp(a, symbolic=symbolic, max_terms=max_terms) for a in expr.args])

    coeff, expr = expr.as_coeff_Add()
    expr = expr.normal()
    old = fraction(expr)
    n, d = fraction(handle(expr))
    if old != (n, d):
        if not d.is_Atom:
            was = (n, d)
            n = signsimp(n, evaluate=False)
            d = signsimp(d, evaluate=False)
            u = Factors(_unevaluated_Mul(n, 1/d))
            u = _unevaluated_Mul(*[k**v for k, v in u.factors.items()])
            n, d = fraction(u)
            if old == (n, d):
                n, d = was
        n = expand_mul(n)
        if d.is_Number or d.is_Add:
            n2, d2 = fraction(gcd_terms(_unevaluated_Mul(n, 1/d)))
            if d2.is_Number or (d2.count_ops() <= d.count_ops()):
                n, d = [signsimp(i) for i in (n2, d2)]
                if n.is_Mul and n.args[0].is_Number:
                    n = n.func(*n.args)

    return coeff + _unevaluated_Mul(n, 1/d)


def rad_rationalize(num, den):
    """
    Rationalize ``num/den`` by removing square roots in the denominator;
    num and den are sum of terms whose squares are positive rationals.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.radsimp import rad_rationalize
    >>> rad_rationalize(sqrt(3), 1 + sqrt(2)/3)
    (-sqrt(3) + sqrt(6)/3, -7/9)
    """
    if not den.is_Add:
        return num, den
    g, a, b = split_surds(den)
    a = a*sqrt(g)
    num = _mexpand((a - b)*num)
    den = _mexpand(a**2 - b**2)
    return rad_rationalize(num, den)


def fraction(expr, exact=False):
    """Returns a pair with expression's numerator and denominator.
       If the given expression is not a fraction then this function
       will return the tuple (expr, 1).

       This function will not make any attempt to simplify nested
       fractions or to do any term rewriting at all.

       If only one of the numerator/denominator pair is needed then
       use numer(expr) or denom(expr) functions respectively.

       >>> from sympy import fraction, Rational, Symbol
       >>> from sympy.abc import x, y

       >>> fraction(x/y)
       (x, y)
       >>> fraction(x)
       (x, 1)

       >>> fraction(1/y**2)
       (1, y**2)

       >>> fraction(x*y/2)
       (x*y, 2)
       >>> fraction(Rational(1, 2))
       (1, 2)

       This function will also work fine with assumptions:

       >>> k = Symbol('k', negative=True)
       >>> fraction(x * y**k)
       (x, y**(-k))

       If we know nothing about sign of some exponent and ``exact``
       flag is unset, then the exponent's structure will
       be analyzed and pretty fraction will be returned:

       >>> from sympy import exp, Mul
       >>> fraction(2*x**(-y))
       (2, x**y)

       >>> fraction(exp(-x))
       (1, exp(x))

       >>> fraction(exp(-x), exact=True)
       (exp(-x), 1)

       The ``exact`` flag will also keep any unevaluated Muls from
       being evaluated:

       >>> u = Mul(2, x + 1, evaluate=False)
       >>> fraction(u)
       (2*x + 2, 1)
       >>> fraction(u, exact=True)
       (2*(x  + 1), 1)
    """
    expr = sympify(expr)

    numer, denom = [], []

    for term in Mul.make_args(expr):
        if term.is_commutative and (term.is_Pow or isinstance(term, exp)):
            b, ex = term.as_base_exp()
            if ex.is_negative:
                if ex is S.NegativeOne:
                    denom.append(b)
                elif exact:
                    if ex.is_constant():
                        denom.append(Pow(b, -ex))
                    else:
                        numer.append(term)
                else:
                    denom.append(Pow(b, -ex))
            elif ex.is_positive:
                numer.append(term)
            elif not exact and ex.is_Mul:
                n, d = term.as_numer_denom()  # this will cause evaluation
                if n != 1:
                    numer.append(n)
                denom.append(d)
            else:
                numer.append(term)
        elif term.is_Rational and not term.is_Integer:
            if term.p != 1:
                numer.append(term.p)
            denom.append(term.q)
        else:
            numer.append(term)
    return Mul(*numer, evaluate=not exact), Mul(*denom, evaluate=not exact)


def numer(expr, exact=False):  # default matches fraction's default
    return fraction(expr, exact=exact)[0]


def denom(expr, exact=False):  # default matches fraction's default
    return fraction(expr, exact=exact)[1]


def fraction_expand(expr, **hints):
    return expr.expand(frac=True, **hints)


def numer_expand(expr, **hints):
    # default matches fraction's default
    a, b = fraction(expr, exact=hints.get('exact', False))
    return a.expand(numer=True, **hints) / b


def denom_expand(expr, **hints):
    # default matches fraction's default
    a, b = fraction(expr, exact=hints.get('exact', False))
    return a / b.expand(denom=True, **hints)


expand_numer = numer_expand
expand_denom = denom_expand
expand_fraction = fraction_expand


def split_surds(expr):
    """
    Split an expression with terms whose squares are positive rationals
    into a sum of terms whose surds squared have gcd equal to g
    and a sum of terms with surds squared prime with g.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.radsimp import split_surds
    >>> split_surds(3*sqrt(3) + sqrt(5)/7 + sqrt(6) + sqrt(10) + sqrt(15))
    (3, sqrt(2) + sqrt(5) + 3, sqrt(5)/7 + sqrt(10))
    """
    args = sorted(expr.args, key=default_sort_key)
    coeff_muls = [x.as_coeff_Mul() for x in args]
    surds = [x[1]**2 for x in coeff_muls if x[1].is_Pow]
    surds.sort(key=default_sort_key)
    g, b1, b2 = _split_gcd(*surds)
    g2 = g
    if not b2 and len(b1) >= 2:
        b1n = [x/g for x in b1]
        b1n = [x for x in b1n if x != 1]
        # only a common factor has been factored; split again
        g1, b1n, b2 = _split_gcd(*b1n)
        g2 = g*g1
    a1v, a2v = [], []
    for c, s in coeff_muls:
        if s.is_Pow and s.exp == S.Half:
            s1 = s.base
            if s1 in b1:
                a1v.append(c*sqrt(s1/g2))
            else:
                a2v.append(c*s)
        else:
            a2v.append(c*s)
    a = Add(*a1v)
    b = Add(*a2v)
    return g2, a, b


def _split_gcd(*a):
    """
    Split the list of integers ``a`` into a list of integers, ``a1`` having
    ``g = gcd(a1)``, and a list ``a2`` whose elements are not divisible by
    ``g``.  Returns ``g, a1, a2``.

    Examples
    ========

    >>> from sympy.simplify.radsimp import _split_gcd
    >>> _split_gcd(55, 35, 22, 14, 77, 10)
    (5, [55, 35, 10], [22, 14, 77])
    """
    g = a[0]
    b1 = [g]
    b2 = []
    for x in a[1:]:
        g1 = gcd(g, x)
        if g1 == 1:
            b2.append(x)
        else:
            g = g1
            b1.append(x)
    return g, b1, b2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\polyroots.py ===
"""Algorithms for computing symbolic roots of polynomials. """


import math
from functools import reduce

from sympy.core import S, I, pi
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.logic import fuzzy_not
from sympy.core.mul import expand_2arg, Mul
from sympy.core.intfunc import igcd
from sympy.core.numbers import Rational, comp
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, symbols
from sympy.core.sympify import sympify
from sympy.functions import exp, im, cos, acos, Piecewise
from sympy.functions.elementary.miscellaneous import root, sqrt
from sympy.ntheory import divisors, isprime, nextprime
from sympy.polys.domains import EX
from sympy.polys.polyerrors import (PolynomialError, GeneratorsNeeded,
    DomainError, UnsolvableFactorError)
from sympy.polys.polyquinticconst import PolyQuintic
from sympy.polys.polytools import Poly, cancel, factor, gcd_list, discriminant
from sympy.polys.rationaltools import together
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.utilities import public
from sympy.utilities.misc import filldedent



z = Symbol('z')  # importing from abc cause O to be lost as clashing symbol


def roots_linear(f):
    """Returns a list of roots of a linear polynomial."""
    r = -f.nth(0)/f.nth(1)
    dom = f.get_domain()

    if not dom.is_Numerical:
        if dom.is_Composite:
            r = factor(r)
        else:
            from sympy.simplify.simplify import simplify
            r = simplify(r)

    return [r]


def roots_quadratic(f):
    """Returns a list of roots of a quadratic polynomial. If the domain is ZZ
    then the roots will be sorted with negatives coming before positives.
    The ordering will be the same for any numerical coefficients as long as
    the assumptions tested are correct, otherwise the ordering will not be
    sorted (but will be canonical).
    """

    a, b, c = f.all_coeffs()
    dom = f.get_domain()

    def _sqrt(d):
        # remove squares from square root since both will be represented
        # in the results; a similar thing is happening in roots() but
        # must be duplicated here because not all quadratics are binomials
        co = []
        other = []
        for di in Mul.make_args(d):
            if di.is_Pow and di.exp.is_Integer and di.exp % 2 == 0:
                co.append(Pow(di.base, di.exp//2))
            else:
                other.append(di)
        if co:
            d = Mul(*other)
            co = Mul(*co)
            return co*sqrt(d)
        return sqrt(d)

    def _simplify(expr):
        if dom.is_Composite:
            return factor(expr)
        else:
            from sympy.simplify.simplify import simplify
            return simplify(expr)

    if c is S.Zero:
        r0, r1 = S.Zero, -b/a

        if not dom.is_Numerical:
            r1 = _simplify(r1)
        elif r1.is_negative:
            r0, r1 = r1, r0
    elif b is S.Zero:
        r = -c/a
        if not dom.is_Numerical:
            r = _simplify(r)

        R = _sqrt(r)
        r0 = -R
        r1 = R
    else:
        d = b**2 - 4*a*c
        A = 2*a
        B = -b/A

        if not dom.is_Numerical:
            d = _simplify(d)
            B = _simplify(B)

        D = factor_terms(_sqrt(d)/A)
        r0 = B - D
        r1 = B + D
        if a.is_negative:
            r0, r1 = r1, r0
        elif not dom.is_Numerical:
            r0, r1 = [expand_2arg(i) for i in (r0, r1)]

    return [r0, r1]


def roots_cubic(f, trig=False):
    """Returns a list of roots of a cubic polynomial.

    References
    ==========
    [1] https://en.wikipedia.org/wiki/Cubic_function, General formula for roots,
    (accessed November 17, 2014).
    """
    if trig:
        a, b, c, d = f.all_coeffs()
        p = (3*a*c - b**2)/(3*a**2)
        q = (2*b**3 - 9*a*b*c + 27*a**2*d)/(27*a**3)
        D = 18*a*b*c*d - 4*b**3*d + b**2*c**2 - 4*a*c**3 - 27*a**2*d**2
        if (D > 0) == True:
            rv = []
            for k in range(3):
                rv.append(2*sqrt(-p/3)*cos(acos(q/p*sqrt(-3/p)*Rational(3, 2))/3 - k*pi*Rational(2, 3)))
            return [i - b/3/a for i in rv]

    # a*x**3 + b*x**2 + c*x + d -> x**3 + a*x**2 + b*x + c
    _, a, b, c = f.monic().all_coeffs()

    if c is S.Zero:
        x1, x2 = roots([1, a, b], multiple=True)
        return [x1, S.Zero, x2]

    # x**3 + a*x**2 + b*x + c -> u**3 + p*u + q
    p = b - a**2/3
    q = c - a*b/3 + 2*a**3/27

    pon3 = p/3
    aon3 = a/3

    u1 = None
    if p is S.Zero:
        if q is S.Zero:
            return [-aon3]*3
        u1 = -root(q, 3) if q.is_positive else root(-q, 3)
    elif q is S.Zero:
        y1, y2 = roots([1, 0, p], multiple=True)
        return [tmp - aon3 for tmp in [y1, S.Zero, y2]]
    elif q.is_real and q.is_negative:
        u1 = -root(-q/2 + sqrt(q**2/4 + pon3**3), 3)

    coeff = I*sqrt(3)/2
    if u1 is None:
        u1 = S.One
        u2 = Rational(-1, 2) + coeff
        u3 = Rational(-1, 2) - coeff
        b, c, d = a, b, c  # a, b, c, d = S.One, a, b, c
        D0 = b**2 - 3*c  # b**2 - 3*a*c
        D1 = 2*b**3 - 9*b*c + 27*d  # 2*b**3 - 9*a*b*c + 27*a**2*d
        C = root((D1 + sqrt(D1**2 - 4*D0**3))/2, 3)
        return [-(b + uk*C + D0/C/uk)/3 for uk in [u1, u2, u3]]  # -(b + uk*C + D0/C/uk)/3/a

    u2 = u1*(Rational(-1, 2) + coeff)
    u3 = u1*(Rational(-1, 2) - coeff)

    if p is S.Zero:
        return [u1 - aon3, u2 - aon3, u3 - aon3]

    soln = [
        -u1 + pon3/u1 - aon3,
        -u2 + pon3/u2 - aon3,
        -u3 + pon3/u3 - aon3
    ]

    return soln

def _roots_quartic_euler(p, q, r, a):
    """
    Descartes-Euler solution of the quartic equation

    Parameters
    ==========

    p, q, r: coefficients of ``x**4 + p*x**2 + q*x + r``
    a: shift of the roots

    Notes
    =====

    This is a helper function for ``roots_quartic``.

    Look for solutions of the form ::

      ``x1 = sqrt(R) - sqrt(A + B*sqrt(R))``
      ``x2 = -sqrt(R) - sqrt(A - B*sqrt(R))``
      ``x3 = -sqrt(R) + sqrt(A - B*sqrt(R))``
      ``x4 = sqrt(R) + sqrt(A + B*sqrt(R))``

    To satisfy the quartic equation one must have
    ``p = -2*(R + A); q = -4*B*R; r = (R - A)**2 - B**2*R``
    so that ``R`` must satisfy the Descartes-Euler resolvent equation
    ``64*R**3 + 32*p*R**2 + (4*p**2 - 16*r)*R - q**2 = 0``

    If the resolvent does not have a rational solution, return None;
    in that case it is likely that the Ferrari method gives a simpler
    solution.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.polys.polyroots import _roots_quartic_euler
    >>> p, q, r = -S(64)/5, -S(512)/125, -S(1024)/3125
    >>> _roots_quartic_euler(p, q, r, S(0))[0]
    -sqrt(32*sqrt(5)/125 + 16/5) + 4*sqrt(5)/5
    """
    # solve the resolvent equation
    x = Dummy('x')
    eq = 64*x**3 + 32*p*x**2 + (4*p**2 - 16*r)*x - q**2
    xsols = list(roots(Poly(eq, x), cubics=False).keys())
    xsols = [sol for sol in xsols if sol.is_rational and sol.is_nonzero]
    if not xsols:
        return None
    R = max(xsols)
    c1 = sqrt(R)
    B = -q*c1/(4*R)
    A = -R - p/2
    c2 = sqrt(A + B)
    c3 = sqrt(A - B)
    return [c1 - c2 - a, -c1 - c3 - a, -c1 + c3 - a, c1 + c2 - a]


def roots_quartic(f):
    r"""
    Returns a list of roots of a quartic polynomial.

    There are many references for solving quartic expressions available [1-5].
    This reviewer has found that many of them require one to select from among
    2 or more possible sets of solutions and that some solutions work when one
    is searching for real roots but do not work when searching for complex roots
    (though this is not always stated clearly). The following routine has been
    tested and found to be correct for 0, 2 or 4 complex roots.

    The quasisymmetric case solution [6] looks for quartics that have the form
    `x**4 + A*x**3 + B*x**2 + C*x + D = 0` where `(C/A)**2 = D`.

    Although no general solution that is always applicable for all
    coefficients is known to this reviewer, certain conditions are tested
    to determine the simplest 4 expressions that can be returned:

      1) `f = c + a*(a**2/8 - b/2) == 0`
      2) `g = d - a*(a*(3*a**2/256 - b/16) + c/4) = 0`
      3) if `f != 0` and `g != 0` and `p = -d + a*c/4 - b**2/12` then
        a) `p == 0`
        b) `p != 0`

    Examples
    ========

        >>> from sympy import Poly
        >>> from sympy.polys.polyroots import roots_quartic

        >>> r = roots_quartic(Poly('x**4-6*x**3+17*x**2-26*x+20'))

        >>> # 4 complex roots: 1+-I*sqrt(3), 2+-I
        >>> sorted(str(tmp.evalf(n=2)) for tmp in r)
        ['1.0 + 1.7*I', '1.0 - 1.7*I', '2.0 + 1.0*I', '2.0 - 1.0*I']

    References
    ==========

    1. http://mathforum.org/dr.math/faq/faq.cubic.equations.html
    2. https://en.wikipedia.org/wiki/Quartic_function#Summary_of_Ferrari.27s_method
    3. https://planetmath.org/encyclopedia/GaloisTheoreticDerivationOfTheQuarticFormula.html
    4. https://people.bath.ac.uk/masjhd/JHD-CA.pdf
    5. http://www.albmath.org/files/Math_5713.pdf
    6. https://web.archive.org/web/20171002081448/http://www.statemaster.com/encyclopedia/Quartic-equation
    7. https://eqworld.ipmnet.ru/en/solutions/ae/ae0108.pdf
    """
    _, a, b, c, d = f.monic().all_coeffs()

    if not d:
        return [S.Zero] + roots([1, a, b, c], multiple=True)
    elif (c/a)**2 == d:
        x, m = f.gen, c/a

        g = Poly(x**2 + a*x + b - 2*m, x)

        z1, z2 = roots_quadratic(g)

        h1 = Poly(x**2 - z1*x + m, x)
        h2 = Poly(x**2 - z2*x + m, x)

        r1 = roots_quadratic(h1)
        r2 = roots_quadratic(h2)

        return r1 + r2
    else:
        a2 = a**2
        e = b - 3*a2/8
        f = _mexpand(c + a*(a2/8 - b/2))
        aon4 = a/4
        g = _mexpand(d - aon4*(a*(3*a2/64 - b/4) + c))

        if f.is_zero:
            y1, y2 = [sqrt(tmp) for tmp in
                      roots([1, e, g], multiple=True)]
            return [tmp - aon4 for tmp in [-y1, -y2, y1, y2]]
        if g.is_zero:
            y = [S.Zero] + roots([1, 0, e, f], multiple=True)
            return [tmp - aon4 for tmp in y]
        else:
            # Descartes-Euler method, see [7]
            sols = _roots_quartic_euler(e, f, g, aon4)
            if sols:
                return sols
            # Ferrari method, see [1, 2]
            p = -e**2/12 - g
            q = -e**3/108 + e*g/3 - f**2/8
            TH = Rational(1, 3)

            def _ans(y):
                w = sqrt(e + 2*y)
                arg1 = 3*e + 2*y
                arg2 = 2*f/w
                ans = []
                for s in [-1, 1]:
                    root = sqrt(-(arg1 + s*arg2))
                    for t in [-1, 1]:
                        ans.append((s*w - t*root)/2 - aon4)
                return ans

            # whether a Piecewise is returned or not
            # depends on knowing p, so try to put
            # in a simple form
            p = _mexpand(p)


            # p == 0 case
            y1 = e*Rational(-5, 6) - q**TH
            if p.is_zero:
                return _ans(y1)

            # if p != 0 then u below is not 0
            root = sqrt(q**2/4 + p**3/27)
            r = -q/2 + root  # or -q/2 - root
            u = r**TH  # primary root of solve(x**3 - r, x)
            y2 = e*Rational(-5, 6) + u - p/u/3
            if fuzzy_not(p.is_zero):
                return _ans(y2)

            # sort it out once they know the values of the coefficients
            return [Piecewise((a1, Eq(p, 0)), (a2, True))
                    for a1, a2 in zip(_ans(y1), _ans(y2))]


def roots_binomial(f):
    """Returns a list of roots of a binomial polynomial. If the domain is ZZ
    then the roots will be sorted with negatives coming before positives.
    The ordering will be the same for any numerical coefficients as long as
    the assumptions tested are correct, otherwise the ordering will not be
    sorted (but will be canonical).
    """
    n = f.degree()

    a, b = f.nth(n), f.nth(0)
    base = -cancel(b/a)
    alpha = root(base, n)

    if alpha.is_number:
        alpha = alpha.expand(complex=True)

    # define some parameters that will allow us to order the roots.
    # If the domain is ZZ this is guaranteed to return roots sorted
    # with reals before non-real roots and non-real sorted according
    # to real part and imaginary part, e.g. -1, 1, -1 + I, 2 - I
    neg = base.is_negative
    even = n % 2 == 0
    if neg:
        if even == True and (base + 1).is_positive:
            big = True
        else:
            big = False

    # get the indices in the right order so the computed
    # roots will be sorted when the domain is ZZ
    ks = []
    imax = n//2
    if even:
        ks.append(imax)
        imax -= 1
    if not neg:
        ks.append(0)
    for i in range(imax, 0, -1):
        if neg:
            ks.extend([i, -i])
        else:
            ks.extend([-i, i])
    if neg:
        ks.append(0)
        if big:
            for i in range(0, len(ks), 2):
                pair = ks[i: i + 2]
                pair = list(reversed(pair))

    # compute the roots
    roots, d = [], 2*I*pi/n
    for k in ks:
        zeta = exp(k*d).expand(complex=True)
        roots.append((alpha*zeta).expand(power_base=False))

    return roots


def _inv_totient_estimate(m):
    """
    Find ``(L, U)`` such that ``L <= phi^-1(m) <= U``.

    Examples
    ========

    >>> from sympy.polys.polyroots import _inv_totient_estimate

    >>> _inv_totient_estimate(192)
    (192, 840)
    >>> _inv_totient_estimate(400)
    (400, 1750)

    """
    primes = [ d + 1 for d in divisors(m) if isprime(d + 1) ]

    a, b = 1, 1

    for p in primes:
        a *= p
        b *= p - 1

    L = m
    U = int(math.ceil(m*(float(a)/b)))

    P = p = 2
    primes = []

    while P <= U:
        p = nextprime(p)
        primes.append(p)
        P *= p

    P //= p
    b = 1

    for p in primes[:-1]:
        b *= p - 1

    U = int(math.ceil(m*(float(P)/b)))

    return L, U


def roots_cyclotomic(f, factor=False):
    """Compute roots of cyclotomic polynomials. """
    L, U = _inv_totient_estimate(f.degree())

    for n in range(L, U + 1):
        g = cyclotomic_poly(n, f.gen, polys=True)

        if f.expr == g.expr:
            break
    else:  # pragma: no cover
        raise RuntimeError("failed to find index of a cyclotomic polynomial")

    roots = []

    if not factor:
        # get the indices in the right order so the computed
        # roots will be sorted
        h = n//2
        ks = [i for i in range(1, n + 1) if igcd(i, n) == 1]
        ks.sort(key=lambda x: (x, -1) if x <= h else (abs(x - n), 1))
        d = 2*I*pi/n
        for k in reversed(ks):
            roots.append(exp(k*d).expand(complex=True))
    else:
        g = Poly(f, extension=root(-1, n))

        for h, _ in ordered(g.factor_list()[1]):
            roots.append(-h.TC())

    return roots


def roots_quintic(f):
    """
    Calculate exact roots of a solvable irreducible quintic with rational coefficients.
    Return an empty list if the quintic is reducible or not solvable.
    """
    result = []

    coeff_5, coeff_4, p_, q_, r_, s_ = f.all_coeffs()

    if not all(coeff.is_Rational for coeff in (coeff_5, coeff_4, p_, q_, r_, s_)):
        return result

    if coeff_5 != 1:
        f = Poly(f / coeff_5)
        _, coeff_4, p_, q_, r_, s_ = f.all_coeffs()

    # Cancel coeff_4 to form x^5 + px^3 + qx^2 + rx + s
    if coeff_4:
        p = p_ - 2*coeff_4*coeff_4/5
        q = q_ - 3*coeff_4*p_/5 + 4*coeff_4**3/25
        r = r_ - 2*coeff_4*q_/5 + 3*coeff_4**2*p_/25 - 3*coeff_4**4/125
        s = s_ - coeff_4*r_/5 + coeff_4**2*q_/25 - coeff_4**3*p_/125 + 4*coeff_4**5/3125
        x = f.gen
        f = Poly(x**5 + p*x**3 + q*x**2 + r*x + s)
    else:
        p, q, r, s = p_, q_, r_, s_

    quintic = PolyQuintic(f)

    # Eqn standardized. Algo for solving starts here
    if not f.is_irreducible:
        return result
    f20 = quintic.f20
    # Check if f20 has linear factors over domain Z
    if f20.is_irreducible:
        return result
    # Now, we know that f is solvable
    for _factor in f20.factor_list()[1]:
        if _factor[0].is_linear:
            theta = _factor[0].root(0)
            break
    d = discriminant(f)
    delta = sqrt(d)
    # zeta = a fifth root of unity
    zeta1, zeta2, zeta3, zeta4 = quintic.zeta
    T = quintic.T(theta, d)
    tol = S(1e-10)
    alpha = T[1] + T[2]*delta
    alpha_bar = T[1] - T[2]*delta
    beta = T[3] + T[4]*delta
    beta_bar = T[3] - T[4]*delta

    disc = alpha**2 - 4*beta
    disc_bar = alpha_bar**2 - 4*beta_bar

    l0 = quintic.l0(theta)
    Stwo = S(2)
    l1 = _quintic_simplify((-alpha + sqrt(disc)) / Stwo)
    l4 = _quintic_simplify((-alpha - sqrt(disc)) / Stwo)

    l2 = _quintic_simplify((-alpha_bar + sqrt(disc_bar)) / Stwo)
    l3 = _quintic_simplify((-alpha_bar - sqrt(disc_bar)) / Stwo)

    order = quintic.order(theta, d)
    test = (order*delta.n()) - ( (l1.n() - l4.n())*(l2.n() - l3.n()) )
    # Comparing floats
    if not comp(test, 0, tol):
        l2, l3 = l3, l2

    # Now we have correct order of l's
    R1 = l0 + l1*zeta1 + l2*zeta2 + l3*zeta3 + l4*zeta4
    R2 = l0 + l3*zeta1 + l1*zeta2 + l4*zeta3 + l2*zeta4
    R3 = l0 + l2*zeta1 + l4*zeta2 + l1*zeta3 + l3*zeta4
    R4 = l0 + l4*zeta1 + l3*zeta2 + l2*zeta3 + l1*zeta4

    Res = [None, [None]*5, [None]*5, [None]*5, [None]*5]
    Res_n = [None, [None]*5, [None]*5, [None]*5, [None]*5]

    # Simplifying improves performance a lot for exact expressions
    R1 = _quintic_simplify(R1)
    R2 = _quintic_simplify(R2)
    R3 = _quintic_simplify(R3)
    R4 = _quintic_simplify(R4)

    # hard-coded results for [factor(i) for i in _vsolve(x**5 - a - I*b, x)]
    x0 = z**(S(1)/5)
    x1 = sqrt(2)
    x2 = sqrt(5)
    x3 = sqrt(5 - x2)
    x4 = I*x2
    x5 = x4 + I
    x6 = I*x0/4
    x7 = x1*sqrt(x2 + 5)
    sol = [x0, -x6*(x1*x3 - x5), x6*(x1*x3 + x5), -x6*(x4 + x7 - I), x6*(-x4 + x7 + I)]

    R1 = R1.as_real_imag()
    R2 = R2.as_real_imag()
    R3 = R3.as_real_imag()
    R4 = R4.as_real_imag()

    for i, s in enumerate(sol):
        Res[1][i] = _quintic_simplify(s.xreplace({z: R1[0] + I*R1[1]}))
        Res[2][i] = _quintic_simplify(s.xreplace({z: R2[0] + I*R2[1]}))
        Res[3][i] = _quintic_simplify(s.xreplace({z: R3[0] + I*R3[1]}))
        Res[4][i] = _quintic_simplify(s.xreplace({z: R4[0] + I*R4[1]}))

    for i in range(1, 5):
        for j in range(5):
            Res_n[i][j] = Res[i][j].n()
            Res[i][j] = _quintic_simplify(Res[i][j])
    r1 = Res[1][0]
    r1_n = Res_n[1][0]

    for i in range(5):
        if comp(im(r1_n*Res_n[4][i]), 0, tol):
            r4 = Res[4][i]
            break

    # Now we have various Res values. Each will be a list of five
    # values. We have to pick one r value from those five for each Res
    u, v = quintic.uv(theta, d)
    testplus = (u + v*delta*sqrt(5)).n()
    testminus = (u - v*delta*sqrt(5)).n()

    # Evaluated numbers suffixed with _n
    # We will use evaluated numbers for calculation. Much faster.
    r4_n = r4.n()
    r2 = r3 = None

    for i in range(5):
        r2temp_n = Res_n[2][i]
        for j in range(5):
            # Again storing away the exact number and using
            # evaluated numbers in computations
            r3temp_n = Res_n[3][j]
            if (comp((r1_n*r2temp_n**2 + r4_n*r3temp_n**2 - testplus).n(), 0, tol) and
                comp((r3temp_n*r1_n**2 + r2temp_n*r4_n**2 - testminus).n(), 0, tol)):
                r2 = Res[2][i]
                r3 = Res[3][j]
                break
        if r2 is not None:
            break
    else:
        return []  # fall back to normal solve

    # Now, we have r's so we can get roots
    x1 = (r1 + r2 + r3 + r4)/5
    x2 = (r1*zeta4 + r2*zeta3 + r3*zeta2 + r4*zeta1)/5
    x3 = (r1*zeta3 + r2*zeta1 + r3*zeta4 + r4*zeta2)/5
    x4 = (r1*zeta2 + r2*zeta4 + r3*zeta1 + r4*zeta3)/5
    x5 = (r1*zeta1 + r2*zeta2 + r3*zeta3 + r4*zeta4)/5
    result = [x1, x2, x3, x4, x5]

    # Now check if solutions are distinct

    saw = set()
    for r in result:
        r = r.n(2)
        if r in saw:
            # Roots were identical. Abort, return []
            # and fall back to usual solve
            return []
        saw.add(r)

    # Restore to original equation where coeff_4 is nonzero
    if coeff_4:
        result = [x - coeff_4 / 5 for x in result]
    return result


def _quintic_simplify(expr):
    from sympy.simplify.simplify import powsimp
    expr = powsimp(expr)
    expr = cancel(expr)
    return together(expr)


def _integer_basis(poly):
    """Compute coefficient basis for a polynomial over integers.

    Returns the integer ``div`` such that substituting ``x = div*y``
    ``p(x) = m*q(y)`` where the coefficients of ``q`` are smaller
    than those of ``p``.

    For example ``x**5 + 512*x + 1024 = 0``
    with ``div = 4`` becomes ``y**5 + 2*y + 1 = 0``

    Returns the integer ``div`` or ``None`` if there is no possible scaling.

    Examples
    ========

    >>> from sympy.polys import Poly
    >>> from sympy.abc import x
    >>> from sympy.polys.polyroots import _integer_basis
    >>> p = Poly(x**5 + 512*x + 1024, x, domain='ZZ')
    >>> _integer_basis(p)
    4
    """
    monoms, coeffs = list(zip(*poly.terms()))

    monoms, = list(zip(*monoms))
    coeffs = list(map(abs, coeffs))

    if coeffs[0] < coeffs[-1]:
        coeffs = list(reversed(coeffs))
        n = monoms[0]
        monoms = [n - i for i in reversed(monoms)]
    else:
        return None

    monoms = monoms[:-1]
    coeffs = coeffs[:-1]

    # Special case for two-term polynominals
    if len(monoms) == 1:
        r = Pow(coeffs[0], S.One/monoms[0])
        if r.is_Integer:
            return int(r)
        else:
            return None

    divs = reversed(divisors(gcd_list(coeffs))[1:])

    try:
        div = next(divs)
    except StopIteration:
        return None

    while True:
        for monom, coeff in zip(monoms, coeffs):
            if coeff % div**monom != 0:
                try:
                    div = next(divs)
                except StopIteration:
                    return None
                else:
                    break
        else:
            return div


def preprocess_roots(poly):
    """Try to get rid of symbolic coefficients from ``poly``. """
    coeff = S.One

    poly_func = poly.func
    try:
        _, poly = poly.clear_denoms(convert=True)
    except DomainError:
        return coeff, poly

    poly = poly.primitive()[1]
    poly = poly.retract()

    # TODO: This is fragile. Figure out how to make this independent of construct_domain().
    if poly.get_domain().is_Poly and all(c.is_term for c in poly.rep.coeffs()):
        poly = poly.inject()

        strips = list(zip(*poly.monoms()))
        gens = list(poly.gens[1:])

        base, strips = strips[0], strips[1:]

        for gen, strip in zip(list(gens), strips):
            reverse = False

            if strip[0] < strip[-1]:
                strip = reversed(strip)
                reverse = True

            ratio = None

            for a, b in zip(base, strip):
                if not a and not b:
                    continue
                elif not a or not b:
                    break
                elif b % a != 0:
                    break
                else:
                    _ratio = b // a

                    if ratio is None:
                        ratio = _ratio
                    elif ratio != _ratio:
                        break
            else:
                if reverse:
                    ratio = -ratio

                poly = poly.eval(gen, 1)
                coeff *= gen**(-ratio)
                gens.remove(gen)

        if gens:
            poly = poly.eject(*gens)

    if poly.is_univariate and poly.get_domain().is_ZZ:
        basis = _integer_basis(poly)

        if basis is not None:
            n = poly.degree()

            def func(k, coeff):
                return coeff//basis**(n - k[0])

            poly = poly.termwise(func)
            coeff *= basis

    if not isinstance(poly, poly_func):
        poly = poly_func(poly)
    return coeff, poly


@public
def roots(f, *gens,
        auto=True,
        cubics=True,
        trig=False,
        quartics=True,
        quintics=False,
        multiple=False,
        filter=None,
        predicate=None,
        strict=False,
        **flags):
    """
    Computes symbolic roots of a univariate polynomial.

    Given a univariate polynomial f with symbolic coefficients (or
    a list of the polynomial's coefficients), returns a dictionary
    with its roots and their multiplicities.

    Only roots expressible via radicals will be returned.  To get
    a complete set of roots use RootOf class or numerical methods
    instead. By default cubic and quartic formulas are used in
    the algorithm. To disable them because of unreadable output
    set ``cubics=False`` or ``quartics=False`` respectively. If cubic
    roots are real but are expressed in terms of complex numbers
    (casus irreducibilis [1]) the ``trig`` flag can be set to True to
    have the solutions returned in terms of cosine and inverse cosine
    functions.

    To get roots from a specific domain set the ``filter`` flag with
    one of the following specifiers: Z, Q, R, I, C. By default all
    roots are returned (this is equivalent to setting ``filter='C'``).

    By default a dictionary is returned giving a compact result in
    case of multiple roots.  However to get a list containing all
    those roots set the ``multiple`` flag to True; the list will
    have identical roots appearing next to each other in the result.
    (For a given Poly, the all_roots method will give the roots in
    sorted numerical order.)

    If the ``strict`` flag is True, ``UnsolvableFactorError`` will be
    raised if the roots found are known to be incomplete (because
    some roots are not expressible in radicals).

    Examples
    ========

    >>> from sympy import Poly, roots, degree
    >>> from sympy.abc import x, y

    >>> roots(x**2 - 1, x)
    {-1: 1, 1: 1}

    >>> p = Poly(x**2-1, x)
    >>> roots(p)
    {-1: 1, 1: 1}

    >>> p = Poly(x**2-y, x, y)

    >>> roots(Poly(p, x))
    {-sqrt(y): 1, sqrt(y): 1}

    >>> roots(x**2 - y, x)
    {-sqrt(y): 1, sqrt(y): 1}

    >>> roots([1, 0, -1])
    {-1: 1, 1: 1}

    ``roots`` will only return roots expressible in radicals. If
    the given polynomial has some or all of its roots inexpressible in
    radicals, the result of ``roots`` will be incomplete or empty
    respectively.

    Example where result is incomplete:

    >>> roots((x-1)*(x**5-x+1), x)
    {1: 1}

    In this case, the polynomial has an unsolvable quintic factor
    whose roots cannot be expressed by radicals. The polynomial has a
    rational root (due to the factor `(x-1)`), which is returned since
    ``roots`` always finds all rational roots.

    Example where result is empty:

    >>> roots(x**7-3*x**2+1, x)
    {}

    Here, the polynomial has no roots expressible in radicals, so
    ``roots`` returns an empty dictionary.

    The result produced by ``roots`` is complete if and only if the
    sum of the multiplicity of each root is equal to the degree of
    the polynomial. If strict=True, UnsolvableFactorError will be
    raised if the result is incomplete.

    The result can be be checked for completeness as follows:

    >>> f = x**3-2*x**2+1
    >>> sum(roots(f, x).values()) == degree(f, x)
    True
    >>> f = (x-1)*(x**5-x+1)
    >>> sum(roots(f, x).values()) == degree(f, x)
    False


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cubic_equation#Trigonometric_and_hyperbolic_solutions

    """
    from sympy.polys.polytools import to_rational_coeffs
    flags = dict(flags)

    if isinstance(f, list):
        if gens:
            raise ValueError('redundant generators given')

        x = Dummy('x')

        poly, i = {}, len(f) - 1

        for coeff in f:
            poly[i], i = sympify(coeff), i - 1

        f = Poly(poly, x, field=True)
    else:
        try:
            F = Poly(f, *gens, **flags)
            if not isinstance(f, Poly) and not F.gen.is_Symbol:
                raise PolynomialError("generator must be a Symbol")
            f = F
        except GeneratorsNeeded:
            if multiple:
                return []
            else:
                return {}
        else:
            n = f.degree()
            if f.length() == 2 and n > 2:
                # check for foo**n in constant if dep is c*gen**m
                con, dep = f.as_expr().as_independent(*f.gens)
                fcon = -(-con).factor()
                if fcon != con:
                    con = fcon
                    bases = []
                    for i in Mul.make_args(con):
                        if i.is_Pow:
                            b, e = i.as_base_exp()
                            if e.is_Integer and b.is_Add:
                                bases.append((b, Dummy(positive=True)))
                    if bases:
                        rv = roots(Poly((dep + con).xreplace(dict(bases)),
                            *f.gens), *F.gens,
                            auto=auto,
                            cubics=cubics,
                            trig=trig,
                            quartics=quartics,
                            quintics=quintics,
                            multiple=multiple,
                            filter=filter,
                            predicate=predicate,
                            **flags)
                        return {factor_terms(k.xreplace(
                            {v: k for k, v in bases})
                        ): v for k, v in rv.items()}

        if f.is_multivariate:
            raise PolynomialError('multivariate polynomials are not supported')

    def _update_dict(result, zeros, currentroot, k):
        if currentroot == S.Zero:
            if S.Zero in zeros:
                zeros[S.Zero] += k
            else:
                zeros[S.Zero] = k
        if currentroot in result:
            result[currentroot] += k
        else:
            result[currentroot] = k

    def _try_decompose(f):
        """Find roots using functional decomposition. """
        factors, roots = f.decompose(), []

        for currentroot in _try_heuristics(factors[0]):
            roots.append(currentroot)

        for currentfactor in factors[1:]:
            previous, roots = list(roots), []

            for currentroot in previous:
                g = currentfactor - Poly(currentroot, f.gen)

                for currentroot in _try_heuristics(g):
                    roots.append(currentroot)

        return roots

    def _try_heuristics(f):
        """Find roots using formulas and some tricks. """
        if f.is_ground:
            return []
        if f.is_monomial:
            return [S.Zero]*f.degree()

        if f.length() == 2:
            if f.degree() == 1:
                return list(map(cancel, roots_linear(f)))
            else:
                return roots_binomial(f)

        result = []

        for i in [-1, 1]:
            if not f.eval(i):
                f = f.quo(Poly(f.gen - i, f.gen))
                result.append(i)
                break

        n = f.degree()

        if n == 1:
            result += list(map(cancel, roots_linear(f)))
        elif n == 2:
            result += list(map(cancel, roots_quadratic(f)))
        elif f.is_cyclotomic:
            result += roots_cyclotomic(f)
        elif n == 3 and cubics:
            result += roots_cubic(f, trig=trig)
        elif n == 4 and quartics:
            result += roots_quartic(f)
        elif n == 5 and quintics:
            result += roots_quintic(f)

        return result

    # Convert the generators to symbols
    dumgens = symbols('x:%d' % len(f.gens), cls=Dummy)
    f = f.per(f.rep, dumgens)

    (k,), f = f.terms_gcd()

    if not k:
        zeros = {}
    else:
        zeros = {S.Zero: k}

    coeff, f = preprocess_roots(f)

    if auto and f.get_domain().is_Ring:
        f = f.to_field()

    # Use EX instead of ZZ_I or QQ_I
    if f.get_domain().is_QQ_I:
        f = f.per(f.rep.convert(EX))

    rescale_x = None
    translate_x = None

    result = {}

    if not f.is_ground:
        dom = f.get_domain()
        if not dom.is_Exact and dom.is_Numerical:
            for r in f.nroots():
                _update_dict(result, zeros, r, 1)
        elif f.degree() == 1:
            _update_dict(result, zeros, roots_linear(f)[0], 1)
        elif f.length() == 2:
            roots_fun = roots_quadratic if f.degree() == 2 else roots_binomial
            for r in roots_fun(f):
                _update_dict(result, zeros, r, 1)
        else:
            _, factors = Poly(f.as_expr()).factor_list()
            if len(factors) == 1 and f.degree() == 2:
                for r in roots_quadratic(f):
                    _update_dict(result, zeros, r, 1)
            else:
                if len(factors) == 1 and factors[0][1] == 1:
                    if f.get_domain().is_EX:
                        res = to_rational_coeffs(f)
                        if res:
                            if res[0] is None:
                                translate_x, f = res[2:]
                            else:
                                rescale_x, f = res[1], res[-1]
                            result = roots(f)
                            if not result:
                                for currentroot in _try_decompose(f):
                                    _update_dict(result, zeros, currentroot, 1)
                        else:
                            for r in _try_heuristics(f):
                                _update_dict(result, zeros, r, 1)
                    else:
                        for currentroot in _try_decompose(f):
                            _update_dict(result, zeros, currentroot, 1)
                else:
                    for currentfactor, k in factors:
                        for r in _try_heuristics(Poly(currentfactor, f.gen, field=True)):
                            _update_dict(result, zeros, r, k)

    if coeff is not S.One:
        _result, result, = result, {}

        for currentroot, k in _result.items():
            result[coeff*currentroot] = k

    if filter not in [None, 'C']:
        handlers = {
            'Z': lambda r: r.is_Integer,
            'Q': lambda r: r.is_Rational,
            'R': lambda r: all(a.is_real for a in r.as_numer_denom()),
            'I': lambda r: r.is_imaginary,
        }

        try:
            query = handlers[filter]
        except KeyError:
            raise ValueError("Invalid filter: %s" % filter)

        for zero in dict(result).keys():
            if not query(zero):
                del result[zero]

    if predicate is not None:
        for zero in dict(result).keys():
            if not predicate(zero):
                del result[zero]
    if rescale_x:
        result1 = {}
        for k, v in result.items():
            result1[k*rescale_x] = v
        result = result1
    if translate_x:
        result1 = {}
        for k, v in result.items():
            result1[k + translate_x] = v
        result = result1

    # adding zero roots after non-trivial roots have been translated
    result.update(zeros)

    if strict and sum(result.values()) < f.degree():
        raise UnsolvableFactorError(filldedent('''
            Strict mode: some factors cannot be solved in radicals, so
            a complete list of solutions cannot be returned. Call
            roots with strict=False to get solutions expressible in
            radicals (if there are any).
            '''))

    if not multiple:
        return result
    else:
        zeros = []

        for zero in ordered(result):
            zeros.extend([zero]*result[zero])

        return zeros


def root_factors(f, *gens, filter=None, **args):
    """
    Returns all factors of a univariate polynomial.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.polys.polyroots import root_factors

    >>> root_factors(x**2 - y, x)
    [x - sqrt(y), x + sqrt(y)]

    """
    args = dict(args)

    F = Poly(f, *gens, **args)

    if not F.is_Poly:
        return [f]

    if F.is_multivariate:
        raise ValueError('multivariate polynomials are not supported')

    x = F.gens[0]

    zeros = roots(F, filter=filter)

    if not zeros:
        factors = [F]
    else:
        factors, N = [], 0

        for r, n in ordered(zeros.items()):
            factors, N = factors + [Poly(x - r, x)]*n, N + n

        if N < F.degree():
            G = reduce(lambda p, q: p*q, factors)
            factors.append(F.quo(G))

    if not isinstance(f, Poly):
        factors = [ f.as_expr() for f in factors ]

    return factors