"""
Tests for the explicit atomic publish mode of `.SFTPClient`
(`put_atomic` / `putfo_atomic`).

Everything runs against in-process, loopback-only stub SFTP servers; no
real SSH accounts, user configuration files or network hosts are ever
consulted.
"""

import threading
from io import BytesIO

from paramiko import RSAKey, SFTP, SFTPServer, Transport
from paramiko.sftp import SFTPError, SFTP_FAILURE
from paramiko.sftp_client import POSIX_RENAME_EXTENSION, SFTPPublishError
from paramiko.sftp_file import SFTPFile

import pytest

from ._loop import LoopSocket
from ._stub_sftp import StubSFTPServer, StubServer
from ._util import _support


class PosixRenameStubSFTPServer(StubSFTPServer):
    """
    Stub server advertising (and recording use of) posix-rename.
    """

    server_extensions = [
        "check-file",
        "md5,sha1",
        POSIX_RENAME_EXTENSION,
        "1",
    ]
    posix_rename_calls = []
    rename_calls = []

    def posix_rename(self, oldpath, newpath):
        self.posix_rename_calls.append((oldpath, newpath))
        return super().posix_rename(oldpath, newpath)

    def rename(self, oldpath, newpath):
        self.rename_calls.append((oldpath, newpath))
        return super().rename(oldpath, newpath)


class PlainStubSFTPServer(StubSFTPServer):
    """
    Stub server that does NOT advertise the posix-rename extension.
    """

    server_extensions = ["check-file", "md5,sha1"]


class FailingPosixRenameStubSFTPServer(PosixRenameStubSFTPServer):
    """
    Advertises posix-rename but always fails the actual request.
    """

    def posix_rename(self, oldpath, newpath):
        self.posix_rename_calls.append((oldpath, newpath))
        return SFTP_FAILURE


@pytest.fixture
def stub_sftp(tmp_path):
    """
    Factory creating SFTP clients bound to per-test loopback stub servers.

    Each fake server is rooted in the test's own temporary directory, so no
    real user or system locations are touched.
    """
    clients = []
    transports = []

    def factory(server_interface):
        root = tmp_path / "remote"
        root.mkdir(exist_ok=True)
        # per-test subclass so ROOT and the call logs never leak between
        # tests
        interface = type(
            server_interface.__name__,
            (server_interface,),
            {
                "ROOT": str(root),
                "posix_rename_calls": [],
                "rename_calls": [],
            },
        )
        socks = LoopSocket()
        sockc = LoopSocket()
        sockc.link(socks)
        tc = Transport(sockc)
        ts = Transport(socks)
        host_key = RSAKey.from_private_key_file(_support("rsa.key"))
        ts.add_server_key(host_key)
        event = threading.Event()
        ts.set_subsystem_handler("sftp", SFTPServer, interface)
        ts.start_server(event, StubServer())
        event.wait(1.0)
        tc.connect(username="fake-user", password="fake-password")
        client = SFTP.from_transport(tc)
        clients.append(client)
        transports.extend([tc, ts])
        return client, interface

    yield factory

    for client in clients:
        try:
            client.close()
        except Exception:
            pass
    for transport in transports:
        try:
            transport.close()
        except Exception:
            pass


def _read(client, path):
    with client.open(path, "r") as f:
        return f.read()


def _write(client, path, data):
    with client.open(path, "w") as f:
        f.write(data)


class ExplodingReader:
    """File-like object that dies partway through being read."""

    def __init__(self):
        self.reads = 0

    def read(self, size=-1):
        self.reads += 1
        if self.reads > 1:
            raise IOError("simulated read failure")
        return b"partial data"


def test_publish_with_extension_uses_write_close_posixrename(stub_sftp):
    client, interface = stub_sftp(PosixRenameStubSFTPServer)
    # capability was confirmed from the handshake
    assert client.server_extensions[POSIX_RENAME_EXTENSION] == "1"
    _write(client, "/target", b"old contents")

    attrs = client.putfo_atomic(
        BytesIO(b"new contents"), "/target", file_size=len(b"new contents")
    )

    # new contents were published over the old target
    assert _read(client, "/target") == b"new contents"
    assert attrs.st_size == len(b"new contents")
    # publish went through the posix-rename extension, never plain rename
    assert interface.rename_calls == []
    assert len(interface.posix_rename_calls) == 1
    temp_path, dest = interface.posix_rename_calls[0]
    assert dest == "/target"
    assert temp_path.startswith("/target.paramiko-tmp-")
    # no staging file left behind
    assert client.listdir("/") == ["target"]


def test_put_atomic_roundtrip_from_local_file(stub_sftp, tmp_path):
    client, interface = stub_sftp(PosixRenameStubSFTPServer)
    local = tmp_path / "local.bin"
    payload = b"x" * 100000
    local.write_bytes(payload)

    client.put_atomic(str(local), "/published")

    assert _read(client, "/published") == payload
    assert client.listdir("/") == ["published"]
    assert len(interface.posix_rename_calls) == 1


def test_default_server_advertisement_is_unchanged(stub_sftp):
    # an interface that does not declare server_extensions still gets the
    # library default advertisement, and the client records it
    client, interface = stub_sftp(StubSFTPServer)
    assert client.server_extensions == {"check-file": "md5,sha1"}
    # and atomic publish is refused, before writing anything
    with pytest.raises(SFTPError, match="posix-rename"):
        client.putfo_atomic(BytesIO(b"nope"), "/target")
    assert client.listdir("/") == []


def test_publish_without_extension_is_refused_before_writing(stub_sftp):
    client, interface = stub_sftp(PlainStubSFTPServer)
    # the extension was not advertised, and nothing else was silently
    # assumed
    assert client.server_extensions == {"check-file": "md5,sha1"}
    _write(client, "/target", b"old contents")

    with pytest.raises(SFTPError, match="posix-rename"):
        client.putfo_atomic(BytesIO(b"new contents"), "/target")

    # refused before any write: target intact, no temp file ever created
    assert _read(client, "/target") == b"old contents"
    assert client.listdir("/") == ["target"]
    assert interface.posix_rename_calls == []


def test_close_failure_does_not_overwrite_old_target(stub_sftp, monkeypatch):
    client, interface = stub_sftp(PosixRenameStubSFTPServer)
    _write(client, "/target", b"old contents")

    real_close = SFTPFile.close

    def boom(self):
        # simulate a close whose server-side handle shut down but which
        # still surfaces an error to the caller (e.g. a failed final
        # flush); closing for real first also keeps the handle from
        # lingering on the fake server.
        real_close(self)
        raise IOError("simulated close failure")

    with monkeypatch.context() as m:
        m.setattr(SFTPFile, "close", boom)
        with pytest.raises(SFTPPublishError) as excinfo:
            client.putfo_atomic(BytesIO(b"new contents"), "/target")

    assert excinfo.value.phase == "close"
    assert "close" in str(excinfo.value)
    # old target untouched; publish never reached the rename phase
    assert _read(client, "/target") == b"old contents"
    assert interface.posix_rename_calls == []
    # the staged temp file was cleaned up
    assert client.listdir("/") == ["target"]


def test_write_failure_cleans_up_only_its_own_temp_file(stub_sftp):
    client, interface = stub_sftp(PosixRenameStubSFTPServer)
    _write(client, "/target", b"old contents")
    # a stale temp-looking file NOT created by this publish; cleanup must
    # be limited to paths owned by the failing call itself
    _write(client, "/target.paramiko-tmp-stale", b"stale leftover")

    with pytest.raises(SFTPPublishError) as excinfo:
        client.putfo_atomic(ExplodingReader(), "/target")

    assert excinfo.value.phase == "write"
    assert _read(client, "/target") == b"old contents"
    # this call's temp file is gone, but the stale look-alike survives
    assert sorted(client.listdir("/")) == [
        "target",
        "target.paramiko-tmp-stale",
    ]
    assert interface.posix_rename_calls == []


def test_plain_rename_entry_point_is_unchanged(stub_sftp):
    # the standard rename entry point still works, is not gated on the
    # posix-rename capability, and does not overwrite existing targets
    client, interface = stub_sftp(PlainStubSFTPServer)
    _write(client, "/a", b"aaa")
    _write(client, "/b", b"bbb")

    client.rename("/a", "/c")
    assert _read(client, "/c") == b"aaa"
    # standard rename must still refuse to clobber an existing target
    with pytest.raises(IOError):
        client.rename("/c", "/b")
    assert _read(client, "/b") == b"bbb"


def test_rename_failure_preserves_old_target(stub_sftp):
    client, interface = stub_sftp(FailingPosixRenameStubSFTPServer)
    _write(client, "/target", b"old contents")

    with pytest.raises(SFTPPublishError) as excinfo:
        client.putfo_atomic(BytesIO(b"new contents"), "/target")

    assert excinfo.value.phase == "rename"
    # the extension was attempted exactly once, against a temp path
    assert len(interface.posix_rename_calls) == 1
    temp_path, dest = interface.posix_rename_calls[0]
    assert dest == "/target"
    assert temp_path.startswith("/target.paramiko-tmp-")
    # old target untouched and the staged temp file was cleaned up
    assert _read(client, "/target") == b"old contents"
    assert client.listdir("/") == ["target"]
