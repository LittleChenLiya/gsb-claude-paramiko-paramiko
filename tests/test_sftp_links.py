# Copyright (C) 2026  Paramiko Contributors
#
# This file is part of Paramiko.
#
# Paramiko is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
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
Tests for root-constrained, component-wise remote symlink resolution
(``SFTPClient.resolve_links`` and the ``resolve_links`` option of
``SFTPClient.open``).

These tests run entirely against an in-memory "fake SFTP" filesystem: the
resolver is only allowed to use lstat/readlink, and the fixture records
every call so zero file opens can be asserted. No real server, no OS
symlink privileges and no temp directories on disk are required.
"""

import errno
import stat

import pytest

from paramiko import SFTPAttributes
from paramiko.message import Message
from paramiko.sftp import (
    CMD_HANDLE,
    CMD_OPEN,
    CMD_READLINK,
)
from paramiko.sftp_client import SFTPClient, SFTPLinkError
from paramiko.util import b

DIR_MODE = stat.S_IFDIR | 0o755
FILE_MODE = stat.S_IFREG | 0o644
LINK_MODE = stat.S_IFLNK | 0o777


def _attrs(mode):
    attr = SFTPAttributes()
    attr.st_mode = mode
    return attr


class FakeSFTP:
    """
    In-memory stand-in for the parts of ``SFTPClient`` the resolver needs.

    ``tree`` maps absolute POSIX paths to either an integer st_mode
    (directories/regular files) or a string symbolic link target. Only
    ``lstat``/``readlink`` are served; opening a target is impossible here,
    which keeps the "zero open calls" guarantee enforceable.
    """

    def __init__(self, tree, cwd=None):
        self.tree = dict(tree)
        self._cwd = b(cwd) if cwd is not None else None
        self.lstat_calls = []
        self.readlink_calls = []
        self.open_calls = []

        class _Logger:
            def debug(self, *a, **k):
                pass

        self.logger = _Logger()
        self.ultra_debug = False

    # reuse the real client cwd adjustment/resolver without constructing one
    _adjust_cwd = SFTPClient._adjust_cwd
    resolve_links = SFTPClient.resolve_links

    def _log(self, level, msg, *args):
        pass

    def lstat(self, path):
        path = path.decode("utf-8") if isinstance(path, bytes) else path
        self.lstat_calls.append(path)
        if path not in self.tree:
            raise IOError(errno.ENOENT, "No such file: {!r}".format(path))
        entry = self.tree[path]
        mode = LINK_MODE if isinstance(entry, str) else entry
        return _attrs(mode)

    def readlink(self, path):
        path = path.decode("utf-8") if isinstance(path, bytes) else path
        self.readlink_calls.append(path)
        entry = self.tree.get(path)
        if not isinstance(entry, str):
            raise IOError(errno.EINVAL, "Not a symlink: {!r}".format(path))
        return entry


@pytest.fixture
def fake_tree():
    """
    A fake remote tree rooted at ``/sandbox`` containing:

    - a plain relative link (``rel -> dir/real.txt``)
    - a chain of relative links (``chain1 -> chain2 -> dir/real.txt``)
    - a two-node cycle (``loop1 <-> loop2``)
    - an out-of-root link using ``..`` (``escape -> ../secret.txt``)
    - an out-of-root absolute link (``abs_escape -> /etc/passwd``)
    - a safe absolute link back inside the root
      (``abs_safe -> /sandbox/dir/real.txt``)
    - a regular file and a directory
    """
    return {
        "/sandbox": DIR_MODE,
        "/sandbox/dir": DIR_MODE,
        "/sandbox/dir/real.txt": FILE_MODE,
        "/sandbox/rel": "dir/real.txt",
        "/sandbox/chain1": "chain2",
        "/sandbox/chain2": "dir/real.txt",
        "/sandbox/loop1": "loop2",
        "/sandbox/loop2": "loop1",
        "/sandbox/escape": "../secret.txt",
        "/sandbox/deep_escape": "../../../../etc/passwd",
        "/sandbox/abs_escape": "/etc/passwd",
        "/sandbox/abs_safe": "/sandbox/dir/real.txt",
        "/secret.txt": FILE_MODE,
        "/etc": DIR_MODE,
        "/etc/passwd": FILE_MODE,
    }


@pytest.fixture
def fake(fake_tree):
    return FakeSFTP(fake_tree)


class RecordingClient(SFTPClient):
    """
    SFTPClient whose wire traffic is served straight from the in-memory tree.

    Construction skips network negotiation; ``_request`` answers lstat,
    readlink and open from the fake, recording opens instead of performing
    them. SFTPFile is stubbed by the tests so ``open()`` never touches a
    file.
    """

    def __init__(self, tree, cwd=None):
        # Intentionally bypass SFTPClient.__init__ (no socket/handshake).
        self.tree = dict(tree)
        self._cwd = b(cwd) if cwd is not None else None
        self.requests = []

        class _Logger:
            def debug(self, *a, **k):
                pass

        self.logger = _Logger()
        self.ultra_debug = False

    def _log(self, level, msg, *args):
        pass

    def _request(self, t, *args):
        # Only CMD_OPEN reaches the wire here: lstat/readlink are served
        # directly below so the resolver's exact request mix is observable.
        self.requests.append((t, tuple(args)))
        if t == CMD_OPEN:
            # Pretend the server handed back a handle; no file is opened.
            msg = Message()
            msg.add_string(b("fake-handle"))
            return CMD_HANDLE, msg
        raise AssertionError("Unexpected request type: {!r}".format(t))

    def readlink(self, path):
        # Serve the link target directly (the real client readlink parses a
        # NAME packet; bypassing the wire encoding keeps the fixture small).
        path = path.decode("utf-8") if isinstance(path, bytes) else path
        entry = self.tree[path]
        if not isinstance(entry, str):
            raise IOError("Not a symlink")
        return entry

    def lstat(self, path):
        # Serve attributes directly instead of hand-building a wire packet,
        # which would otherwise need the request-id prefix the client reads.
        path = path.decode("utf-8") if isinstance(path, bytes) else path
        if path not in self.tree:
            raise IOError(errno.ENOENT, "No such file")
        entry = self.tree[path]
        mode = LINK_MODE if isinstance(entry, str) else entry
        return _attrs(mode)

    @property
    def open_paths(self):
        return [
            args[0].decode("utf-8") if isinstance(args[0], bytes) else args[0]
            for t, args in self.requests
            if t == CMD_OPEN
        ]


class TestResolveLinks:
    def test_relative_link_resolves_from_link_directory(self, fake):
        result = fake.resolve_links("/sandbox/rel", root="/sandbox")
        assert result == "/sandbox/dir/real.txt"
        assert fake.readlink_calls == ["/sandbox/rel"]
        assert fake.open_calls == []
        assert "/sandbox/rel" in fake.lstat_calls

    def test_relative_chain(self, fake):
        result = fake.resolve_links("/sandbox/chain1", root="/sandbox")
        assert result == "/sandbox/dir/real.txt"
        assert fake.readlink_calls == [
            "/sandbox/chain1",
            "/sandbox/chain2",
        ]
        assert fake.open_calls == []

    def test_plain_path_needs_no_readlink(self, fake):
        result = fake.resolve_links(
            "/sandbox/dir/real.txt", root="/sandbox"
        )
        assert result == "/sandbox/dir/real.txt"
        assert fake.readlink_calls == []
        assert fake.open_calls == []

    def test_two_node_cycle_reports_chain(self, fake):
        with pytest.raises(SFTPLinkError) as excinfo:
            fake.resolve_links("/sandbox/loop1", root="/sandbox")
        err = excinfo.value
        assert err.errno == errno.ELOOP
        assert err.chain == [
            "/sandbox/loop1",
            "/sandbox/loop2",
            "/sandbox/loop1",
        ]
        assert err.link_chain() == err.chain
        assert "/sandbox/loop1 -> /sandbox/loop2 -> /sandbox/loop1" in str(err)
        assert fake.open_calls == []

    def test_relative_dotdot_escape_rejected(self, fake):
        with pytest.raises(SFTPLinkError) as excinfo:
            fake.resolve_links("/sandbox/escape", root="/sandbox")
        err = excinfo.value
        assert err.errno == errno.EACCES
        assert err.target == "../secret.txt"
        assert err.chain == ["/sandbox/escape"]
        assert "escapes remote root" in str(err)
        assert fake.open_calls == []

    def test_absolute_escape_rejected(self, fake):
        with pytest.raises(SFTPLinkError) as excinfo:
            fake.resolve_links("/sandbox/abs_escape", root="/sandbox")
        err = excinfo.value
        assert err.errno == errno.EACCES
        assert err.target == "/etc/passwd"
        assert fake.open_calls == []

    def test_relative_escape_past_server_root_rejected(self, fake):
        with pytest.raises(SFTPLinkError) as excinfo:
            fake.resolve_links("/sandbox/deep_escape", root="/sandbox")
        err = excinfo.value
        assert err.errno == errno.EACCES
        assert err.target == "../../../../etc/passwd"
        assert fake.open_calls == []

    def test_absolute_link_inside_root_allowed(self, fake):
        result = fake.resolve_links("/sandbox/abs_safe", root="/sandbox")
        assert result == "/sandbox/dir/real.txt"
        assert fake.open_calls == []

    def test_relative_input_uses_session_cwd(self, fake_tree):
        fake = FakeSFTP(fake_tree, cwd="/sandbox/dir")
        result = fake.resolve_links("../rel", root="/sandbox")
        assert result == "/sandbox/dir/real.txt"
        assert fake.open_calls == []

    def test_missing_component_propagates_ioerror(self, fake):
        with pytest.raises(IOError) as excinfo:
            fake.resolve_links("/sandbox/nope/real.txt", root="/sandbox")
        assert excinfo.value.errno == errno.ENOENT
        assert fake.open_calls == []

    def test_bytes_path_supported(self, fake):
        result = fake.resolve_links(b"/sandbox/rel", root="/sandbox")
        assert result == "/sandbox/dir/real.txt"

    def test_too_many_links(self, fake_tree):
        tree = dict(fake_tree)
        tree["/sandbox/l0"] = "l1"
        for i in range(1, 9):
            tree["/sandbox/l{}".format(i)] = "l{}".format(i + 1)
        tree["/sandbox/l9"] = "real"
        tree["/sandbox/real"] = FILE_MODE
        fake = FakeSFTP(tree)
        with pytest.raises(SFTPLinkError) as excinfo:
            fake.resolve_links("/sandbox/l0", root="/sandbox", max_links=5)
        assert excinfo.value.errno == errno.ELOOP
        assert fake.open_calls == []

    def test_root_must_be_absolute(self, fake):
        with pytest.raises(SFTPLinkError) as excinfo:
            fake.resolve_links("rel", root="sandbox")
        assert excinfo.value.errno == errno.EINVAL


class TestOpenIntegration:
    @pytest.fixture
    def no_sftp_file(self, monkeypatch):
        # prevent open() from constructing a real SFTPFile
        monkeypatch.setattr(
            "paramiko.sftp_client.SFTPFile", lambda *a, **k: None
        )

    def test_disabled_option_passes_path_through(
        self, fake_tree, no_sftp_file
    ):
        client = RecordingClient(fake_tree)
        client.open("/sandbox/rel", "r")
        # default behavior: the unresolved link goes straight to the server
        assert client.open_paths == ["/sandbox/rel"]

    def test_enabled_option_opens_resolved_target(
        self, fake_tree, no_sftp_file
    ):
        client = RecordingClient(fake_tree)
        client.open(
            "/sandbox/rel", "r", resolve_links=True, root="/sandbox"
        )
        assert client.open_paths == ["/sandbox/dir/real.txt"]
        assert len(client.open_paths) == 1

    def test_enabled_option_blocks_cycle_before_open(
        self, fake_tree, no_sftp_file
    ):
        client = RecordingClient(fake_tree)
        with pytest.raises(SFTPLinkError) as excinfo:
            client.open(
                "/sandbox/loop1", "r", resolve_links=True, root="/sandbox"
            )
        assert excinfo.value.errno == errno.ELOOP
        assert client.open_paths == []

    def test_enabled_option_blocks_escape_before_open(
        self, fake_tree, no_sftp_file
    ):
        client = RecordingClient(fake_tree)
        with pytest.raises(SFTPLinkError) as excinfo:
            client.open(
                "/sandbox/escape", "r", resolve_links=True, root="/sandbox"
            )
        assert excinfo.value.errno == errno.EACCES
        assert client.open_paths == []
