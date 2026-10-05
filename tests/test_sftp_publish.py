# Copyright (C) 2026  Robey Pointer <robeypointer@gmail.com>
#
# This file is part of Paramiko.
#
# Paramiko is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 2.1 of the License, or (at your option)
# any later version.
#
# Paramiko is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Paramiko; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.

"""
Tests for the explicit atomic "publish" mode of `.SFTPClient`.

A publish writes a temporary file, closes it, and then uses the OpenSSH
``posix-rename@openssh.com`` extension to rename the temporary file over the
destination.  These tests use an in-process fake SFTP service (a loopback
transport backed by a local temporary folder); no real SSH account
configuration is read.
"""

import io
import os
import shutil
import struct
import threading

import pytest

from paramiko import (
    POSIX_RENAME_EXTENSION,
    RSAKey,
    SFTP,
    SFTPError,
    SFTPublishError,
    SFTPServer,
    Transport,
)
from paramiko.message import Message
from paramiko.sftp import (
    CMD_INIT,
    CMD_VERSION,
    SFTP_FAILURE,
    SFTP_OK,
    _VERSION,
)

from ._loop import LoopSocket
from ._stub_sftp import StubServer, StubSFTPHandle, StubSFTPServer
from ._util import _support


class FakeHandle(StubSFTPHandle):
    """
    A file handle that can be told to fail its close request, while still
    closing the underlying OS file (so the temporary file can be cleaned up
    afterwards).
    """

    fail_close = False
    fail_write = False

    def close(self):
        # Always close the underlying OS file first; the simulated failure
        # only affects the SFTP close response, not local cleanup.
        super().close()
        if self.fail_close and not getattr(self, "_close_reported", False):
            self._close_reported = True
            raise IOError("simulated close failure")

    def write(self, offset, data):
        if self.fail_write:
            return SFTP_FAILURE
        return super().write(offset, data)


class FakeSFTPServerInterface(StubSFTPServer):
    """
    Fake SFTP interface backed by a local temporary folder.

    Records every operation and can be configured (via class attributes,
    toggled per test) to advertise the posix-rename extension and/or fail
    close, rename, or write requests.
    """

    # Configuration toggles (set per test before the client connects).
    advertise_posix_rename = True
    fail_close = False
    fail_rename = False
    fail_write = False
    # Recorded operations: list of (op, detail) tuples.
    operations = []
    # Roots created for fake servers, for cleanup.
    roots = []

    def __init__(self, server, *args, **kwargs):
        super().__init__(server, *args, **kwargs)
        self.root = os.path.join(
            os.getcwd(), ".publish-test-{:08x}".format(id(self) & 0xFFFFFFFF)
        )
        os.mkdir(self.root)
        FakeSFTPServerInterface.roots.append(self.root)

    def _realpath(self, path):
        return self.root + self.canonicalize(path)

    def _record(self, op, detail):
        FakeSFTPServerInterface.operations.append((op, detail))

    def open(self, path, flags, attr):
        self._record("open", path)
        path = self._realpath(path)
        try:
            binary_flag = getattr(os, "O_BINARY", 0)
            flags |= binary_flag
            mode = getattr(attr, "st_mode", None)
            if mode is not None:
                fd = os.open(path, flags, mode)
            else:
                fd = os.open(path, flags, 0o666)
        except OSError as e:
            return SFTPServer.convert_errno(e.errno)
        if (flags & os.O_CREAT) and (attr is not None):
            attr._flags &= ~attr.FLAG_PERMISSIONS
            SFTPServer.set_file_attr(path, attr)
        if flags & os.O_WRONLY:
            if flags & os.O_APPEND:
                fstr = "ab"
            else:
                fstr = "wb"
        elif flags & os.O_RDWR:
            if flags & os.O_APPEND:
                fstr = "a+b"
            else:
                fstr = "r+b"
        else:
            fstr = "rb"
        try:
            f = os.fdopen(fd, fstr)
        except OSError as e:
            return SFTPServer.convert_errno(e.errno)
        handle = FakeHandle(flags)
        handle.filename = path
        handle.readfile = f
        handle.writefile = f
        handle.fail_close = self.fail_close
        handle.fail_write = self.fail_write
        return handle

    def remove(self, path):
        self._record("remove", path)
        path = self._realpath(path)
        try:
            os.remove(path)
        except OSError as e:
            return SFTPServer.convert_errno(e.errno)
        return SFTP_OK

    def rename(self, oldpath, newpath):
        self._record("rename", "{} -> {}".format(oldpath, newpath))
        oldpath = self._realpath(oldpath)
        newpath = self._realpath(newpath)
        if os.path.exists(newpath):
            return SFTP_FAILURE
        try:
            os.rename(oldpath, newpath)
        except OSError as e:
            return SFTPServer.convert_errno(e.errno)
        return SFTP_OK

    def posix_rename(self, oldpath, newpath):
        self._record("posix_rename", "{} -> {}".format(oldpath, newpath))
        if self.fail_rename:
            return SFTP_FAILURE
        oldpath = self._realpath(oldpath)
        newpath = self._realpath(newpath)
        try:
            # posix-rename overwrites an existing destination atomically;
            # os.replace does the same on Windows.
            os.replace(oldpath, newpath)
        except OSError as e:
            return SFTPServer.convert_errno(e.errno)
        return SFTP_OK


class FakeSFTPServer(SFTPServer):
    """SFTPServer that advertises posix-rename when the interface allows it."""

    def _send_server_version(self):
        t, data = self._read_packet()
        if t != CMD_INIT:
            raise SFTPError("Incompatible sftp protocol")
        version = struct.unpack(">I", data[:4])[0]
        msg = Message()
        msg.add_int(_VERSION)
        msg.add_string("check-file")
        msg.add_string("md5,sha1")
        if getattr(self.server, "advertise_posix_rename", False):
            msg.add_string(POSIX_RENAME_EXTENSION)
            msg.add_string("1")
        self._send_packet(CMD_VERSION, msg)
        return version


@pytest.fixture
def fake_sftp_factory():
    """
    Yield a factory that creates fresh fake SFTP services with configurable
    behaviour.  Everything runs in-process against a local temp folder; no
    SSH account configuration is read.
    """
    created = []

    def _make(
        advertise_posix_rename=True,
        fail_close=False,
        fail_rename=False,
        fail_write=False,
    ):
        FakeSFTPServerInterface.operations = []
        FakeSFTPServerInterface.advertise_posix_rename = advertise_posix_rename
        FakeSFTPServerInterface.fail_close = fail_close
        FakeSFTPServerInterface.fail_rename = fail_rename
        FakeSFTPServerInterface.fail_write = fail_write

        socks = LoopSocket()
        sockc = LoopSocket()
        sockc.link(socks)
        tc = Transport(sockc)
        ts = Transport(socks)
        host_key = RSAKey.from_private_key_file(_support("rsa.key"))
        ts.add_server_key(host_key)
        event = threading.Event()
        ts.set_subsystem_handler(
            "sftp", FakeSFTPServer, FakeSFTPServerInterface
        )
        ts.start_server(event, StubServer())
        event.wait(1.0)
        tc.connect(username="slowdive", password="pygmalion")
        client = SFTP.from_transport(tc)
        root = FakeSFTPServerInterface.roots[-1]
        created.append((client, tc, ts, root))
        return client, root

    yield _make

    for client, tc, ts, root in created:
        client.close()
        tc.close()
        ts.close()
        shutil.rmtree(root, ignore_errors=True)


def _leftover_temps(root):
    return [n for n in os.listdir(root) if ".publish." in n]


def _ops_of_type(op_type):
    return [
        detail
        for op, detail in FakeSFTPServerInterface.operations
        if op == op_type
    ]


def test_publish_with_posix_rename(fake_sftp_factory):
    """With the extension, publish writes a temp file, closes it, then
    posix-renames over the destination; the old target is replaced."""
    client, root = fake_sftp_factory(advertise_posix_rename=True)
    target = "target.txt"
    with client.open(target, "w") as f:
        f.write(b"old content")

    FakeSFTPServerInterface.operations = []
    client.publishfo(io.BytesIO(b"new content"), target)

    assert client.open(target, "rb").read() == b"new content"
    # posix-rename was used, not delete-then-rename.
    assert _ops_of_type("posix_rename")
    assert not _ops_of_type("remove")
    # No temporary file left behind.
    assert _leftover_temps(root) == []


def test_publish_new_target(fake_sftp_factory):
    """Publishing to a not-yet-existing destination works."""
    client, root = fake_sftp_factory()
    client.publishfo(io.BytesIO(b"brand new"), "new.txt")
    assert client.open("new.txt", "rb").read() == b"brand new"
    assert _leftover_temps(root) == []


def test_publish_without_extension_rejects_before_write(fake_sftp_factory):
    """Without the extension, publish is rejected before anything is
    written; the existing target is untouched and no temp file is made."""
    client, root = fake_sftp_factory(advertise_posix_rename=False)
    target = "target.txt"
    with client.open(target, "w") as f:
        f.write(b"old content")

    FakeSFTPServerInterface.operations = []
    with pytest.raises(SFTPublishError) as exc_info:
        client.publishfo(io.BytesIO(b"new content"), target)

    assert exc_info.value.phase == "capability"
    assert not _ops_of_type("open")
    assert not _ops_of_type("posix_rename")
    assert not _ops_of_type("remove")
    assert client.open(target, "rb").read() == b"old content"
    assert _leftover_temps(root) == []


def test_close_failure_preserves_target(fake_sftp_factory):
    """A close failure is reported as the close phase and must not overwrite
    the existing target; the temp file is cleaned up and no rename happens."""
    client, root = fake_sftp_factory(fail_close=False)
    target = "target.txt"
    with client.open(target, "w") as f:
        f.write(b"old content")

    # Fail closes only for the publish attempt, not the setup above.
    FakeSFTPServerInterface.fail_close = True
    FakeSFTPServerInterface.operations = []
    with pytest.raises(SFTPublishError) as exc_info:
        client.publishfo(io.BytesIO(b"new content"), target)

    assert exc_info.value.phase == "close"
    assert client.open(target, "rb").read() == b"old content"
    assert not _ops_of_type("posix_rename")
    assert _leftover_temps(root) == []


def test_rename_failure_preserves_target(fake_sftp_factory):
    """A rename failure is reported as the rename phase and must not
    overwrite the existing target; the temp file is cleaned up."""
    client, root = fake_sftp_factory(fail_rename=False)
    target = "target.txt"
    with client.open(target, "w") as f:
        f.write(b"old content")

    # Fail the rename only for the publish attempt.
    FakeSFTPServerInterface.fail_rename = True
    FakeSFTPServerInterface.operations = []
    with pytest.raises(SFTPublishError) as exc_info:
        client.publishfo(io.BytesIO(b"new content"), target)

    assert exc_info.value.phase == "rename"
    assert client.open(target, "rb").read() == b"old content"
    assert _ops_of_type("posix_rename")
    assert _leftover_temps(root) == []


def test_write_failure_preserves_target(fake_sftp_factory):
    """A write failure is reported as the write phase and must not
    overwrite the existing target; the temp file is cleaned up."""
    client, root = fake_sftp_factory(fail_write=False)
    target = "target.txt"
    with client.open(target, "w") as f:
        f.write(b"old content")

    # Fail writes only for the publish attempt, not the setup above.
    FakeSFTPServerInterface.fail_write = True
    FakeSFTPServerInterface.operations = []
    with pytest.raises(SFTPublishError) as exc_info:
        client.publishfo(io.BytesIO(b"new content"), target)

    assert exc_info.value.phase == "write"
    assert client.open(target, "rb").read() == b"old content"
    assert not _ops_of_type("posix_rename")
    assert _leftover_temps(root) == []


def test_failed_publish_only_cleans_its_own_temp(fake_sftp_factory):
    """A failed publish must not remove unrelated pre-existing files."""
    client, root = fake_sftp_factory(fail_rename=True)
    target = "target.txt"
    with client.open(target, "w") as f:
        f.write(b"old content")
    with client.open("keep.txt", "w") as f:
        f.write(b"keep me")

    with pytest.raises(SFTPublishError):
        client.publishfo(io.BytesIO(b"new content"), target)

    assert client.open("keep.txt", "rb").read() == b"keep me"
    assert client.open(target, "rb").read() == b"old content"
    assert _leftover_temps(root) == []


def test_normal_rename_entry_is_unchanged(fake_sftp_factory):
    """The plain rename() entry still uses standard RENAME semantics and
    refuses to overwrite an existing destination."""
    client, root = fake_sftp_factory()
    with client.open("other.txt", "w") as f:
        f.write(b"other")
    with client.open("target.txt", "w") as f:
        f.write(b"target")

    FakeSFTPServerInterface.operations = []
    with pytest.raises(IOError):
        client.rename("other.txt", "target.txt")

    # Standard rename was used (it failed because the target exists), and
    # posix-rename was not used.
    assert _ops_of_type("rename")
    assert not _ops_of_type("posix_rename")
    assert client.open("target.txt", "rb").read() == b"target"


def test_publish_local_file(fake_sftp_factory, tmp_path):
    """The publish() convenience method publishes from a local file."""
    client, root = fake_sftp_factory()
    local = tmp_path / "local.txt"
    local.write_bytes(b"from a local file")
    client.publish(str(local), "published.txt")
    assert client.open("published.txt", "rb").read() == b"from a local file"
    assert _leftover_temps(root) == []
