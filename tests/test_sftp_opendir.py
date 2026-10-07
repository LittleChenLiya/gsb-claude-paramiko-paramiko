# Copyright (C) 2026 Paramiko Authors
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
# You should have received a copy of the GNU Lesser General Public License
# along with Paramiko; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.

"""
Deterministic, in-memory tests for the SFTP directory iterator handle
lifecycle (``SFTPClient.opendir`` / ``SFTPClient.listdir_iter``).

These tests use a tiny scripted fake ``SFTPClient`` transport instead of a
real (or stubbed loopback) server: every outgoing packet is recorded and a
pre-queued reply is returned, so the exact number and target handle of
``SSH_FXP_CLOSE`` requests can be asserted with no sockets, threads, or
external fixtures.

A short loopback (stub-server) regression test for the historical
``listdir_iter`` entry point lives at the bottom of the module and is marked
``slow`` like the rest of ``test_sftp.py``.
"""

from collections import deque
import threading

import pytest

from paramiko.message import Message
from paramiko.sftp import (
    CMD_CLOSE,
    CMD_HANDLE,
    CMD_NAME,
    CMD_OPEN,
    CMD_OPENDIR,
    CMD_READDIR,
    CMD_STATUS,
    SFTP_EOF,
    SFTP_OK,
    SFTPError,
)
from paramiko.sftp_attr import SFTPAttributes
from paramiko.sftp_client import SFTPClient
from paramiko.sftp_file import SFTPFile
from paramiko import util
from paramiko.util import b


# a two-page ("two SFTP readdir response") directory listing:
# page 1 -> alpha, beta ; page 2 -> gamma ; then SSH_FX_EOF
FIRST_PAGE = ["alpha", "beta"]
SECOND_PAGE = ["gamma"]
DIR_HANDLE = b"dir-handle-1"
# an unrelated *file* handle that must never be touched by directory cleanup
FILE_HANDLE = b"file-handle-9"
LISTING = FIRST_PAGE + SECOND_PAGE


def _name_reply(names):
    """Build an SSH_FXP_NAME response body containing ``names``."""
    msg = Message()
    msg.add_int(len(names))
    for name in names:
        # filename, longname, empty attribute block
        msg.add_string(b(name))
        msg.add_string(b("-" + name))
        SFTPAttributes()._pack(msg)
    return msg.asbytes()


def _status_reply(code=SFTP_OK, text=""):
    msg = Message()
    msg.add_int(code)
    msg.add_string(b(text))
    msg.add_string(b(""))  # language tag
    return msg.asbytes()


def _handle_reply(handle):
    msg = Message()
    msg.add_string(handle)
    return msg.asbytes()


class ScriptedSFTPClient(SFTPClient):
    """
    SFTPClient without a socket: outgoing packets are recorded and answered
    from per-command canned responses.

    Recorded attributes:

    * ``requests``: list of ``(command, args)`` for every packet sent, in
      order;
    * ``closes``: list of the handle bytes passed to every SSH_FXP_CLOSE.
    """

    def __init__(self, readdir_responses, dir_handle=DIR_HANDLE):
        # Bypass SFTPClient.__init__ entirely: no version negotiation,
        # channel, or socket is needed for these code paths.
        self.sock = None
        self.ultra_debug = False
        self.request_number = 1
        self._lock = threading.Lock()
        self._cwd = None
        # plain dict on purpose: type(None) is a valid key
        self._expecting = {}
        self.logger = util.get_logger("paramiko.sftp.scripted")
        self.requests = []
        self.closes = []
        self._queue = deque()
        self._dir_handle = dir_handle
        self._readdir_responses = deque(readdir_responses)
        self._replies = {
            CMD_OPENDIR: lambda m: (
                CMD_HANDLE,
                _handle_reply(self._dir_handle),
            ),
            CMD_CLOSE: lambda m: (CMD_STATUS, _status_reply(SFTP_OK)),
        }

    def _log(self, level, msg, *args):
        # SFTPClient._log requires a real channel; use the base logger
        from paramiko.sftp import BaseSFTP

        BaseSFTP._log(self, level, msg, *args)

    def _send_packet(self, t, packet):
        body = packet.asbytes()
        msg = Message(body)
        num = msg.get_int()
        if t in (CMD_OPENDIR, CMD_READDIR, CMD_CLOSE, CMD_OPEN):
            args = (msg.get_string(),)
        else:
            args = (body,)
        self.requests.append((t, args))
        if t == CMD_CLOSE:
            self.closes.append(args[0])
        self._enqueue_reply(t, num)

    def _enqueue_reply(self, t, num):
        handler = self._replies.get(t)
        if handler is not None:
            response_t, payload = handler(None)
        elif t == CMD_READDIR:
            # like a real server: serve the scripted pages, then EOF for
            # every further pipelined readdir on the same handle
            if self._readdir_responses:
                response_t, payload = self._readdir_responses.popleft()
            else:
                response_t, payload = (
                    CMD_STATUS,
                    _status_reply(SFTP_EOF, "eof"),
                )
        else:
            raise AssertionError(
                "unexpected canned request: {!r}".format(t)
            )
        # prefix the response body with the echoed request id
        out = Message()
        out.add_int(num)
        out.packet.write(Message(payload).packet.getvalue())
        self._queue.append((response_t, out.asbytes()))

    # wire SFTPClient's packet reader/writer to the in-memory queue
    def _read_packet(self):
        return self._queue.popleft()

    def _write_all(self, out):
        # packets are fully handled in _send_packet; nothing to do here
        pass


def _scripted_client(*readdir_responses, **kwargs):
    return ScriptedSFTPClient(list(readdir_responses), **kwargs)


def _two_page_responses():
    """readdir response script: name page, name page, then EOF."""
    return [
        (CMD_NAME, _name_reply(FIRST_PAGE)),
        (CMD_NAME, _name_reply(SECOND_PAGE)),
        (CMD_STATUS, _status_reply(SFTP_EOF, "eof")),
    ]


def _bad_name_reply():
    """an SSH_FXP_NAME entry whose filename is not valid utf-8."""
    msg = Message()
    msg.add_int(1)
    msg.add_string(b"\xff\xfe-bad-utf8")  # filename (undecodable)
    msg.add_string(b"longname")
    SFTPAttributes()._pack(msg)
    return msg.asbytes()


# ---------------------------------------------------------------------------
# SFTPClient.opendir: SFTPDirectoryIterator
# ---------------------------------------------------------------------------


def test_opendir_early_exit_after_one_item_closes_handle_once():
    """read one entry then abandon: exactly one CLOSE for the dir handle."""
    sftp = _scripted_client(*_two_page_responses())
    it = sftp.opendir(".")
    first = next(it)
    assert first.filename == "alpha"
    assert sftp.closes == []  # still open mid-iteration
    it.close()
    assert sftp.closes == [DIR_HANDLE]
    # idempotent: a second close sends nothing more
    it.close()
    assert sftp.closes == [DIR_HANDLE]
    assert it.closed is True
    # iteration after close ends cleanly
    assert list(it) == []


def test_opendir_full_consumption_closes_handle_once():
    sftp = _scripted_client(*_two_page_responses())
    with sftp.opendir(".") as it:
        names = [attr.filename for attr in it]
    assert names == LISTING
    assert sftp.closes == [DIR_HANDLE]
    assert it.closed is True
    # closing again is a no-op
    it.close()
    assert sftp.closes == [DIR_HANDLE]


def test_opendir_context_manager_closes_on_early_break():
    sftp = _scripted_client(*_two_page_responses())
    with sftp.opendir(".") as it:
        for attr in it:
            if attr.filename == "beta":
                break
    assert sftp.closes == [DIR_HANDLE]


def test_opendir_bad_filename_still_closes_handle_once():
    sftp = _scripted_client(
        (CMD_NAME, _bad_name_reply()),
        (CMD_STATUS, _status_reply(SFTP_EOF, "eof")),
    )
    it = sftp.opendir(".")
    with pytest.raises(UnicodeDecodeError):
        next(it)
    # the decode failure must not leak the handle...
    assert sftp.closes == [DIR_HANDLE]
    # ...and the iterator reports itself closed
    assert it.closed is True
    it.close()  # idempotent, no second packet
    assert sftp.closes == [DIR_HANDLE]


def test_directory_close_does_not_touch_other_handles():
    """an open file handle must never be closed by directory cleanup."""
    sftp = _scripted_client(*_two_page_responses())
    sftp._replies[CMD_OPEN] = lambda m: (
        CMD_HANDLE,
        _handle_reply(FILE_HANDLE),
    )
    opened = sftp.open("some-file.txt", "r")
    assert isinstance(opened, SFTPFile)
    assert sftp.closes == []

    it = sftp.opendir(".")
    next(it)
    it.close()
    # only the directory was closed; the file handle stays open
    assert sftp.closes == [DIR_HANDLE]
    assert FILE_HANDLE not in sftp.closes
    # the file closes exactly its own handle when asked
    opened.close()
    assert sftp.closes == [DIR_HANDLE, FILE_HANDLE]


def test_opendir_abandoned_iterator_is_closed_on_gc():
    import gc

    sftp = _scripted_client(*_two_page_responses())
    it = sftp.opendir(".")
    next(it)
    del it
    gc.collect()
    # __del__ best-effort cleanup closes the handle exactly once too
    assert sftp.closes == [DIR_HANDLE]


def test_opendir_eager_open_failure_sends_no_close():
    from paramiko.sftp import CMD_ATTRS

    sftp = _scripted_client(*_two_page_responses())
    # reply to OPENDIR with a non-HANDLE packet; opendir must error out
    # before any handle exists, so no CLOSE request is possible/allowed
    sftp._replies[CMD_OPENDIR] = lambda m: (CMD_ATTRS, b"")
    with pytest.raises(SFTPError):
        sftp.opendir(".")
    assert sftp.closes == []


# ---------------------------------------------------------------------------
# Regression for the historical SFTPClient.listdir_iter generator entry
# ---------------------------------------------------------------------------


def test_listdir_iter_regression_full_partial_and_bad_name():
    # full consumption via the legacy entry point still yields everything...
    sftp = _scripted_client(*_two_page_responses())
    assert [a.filename for a in sftp.listdir_iter(".")] == LISTING
    assert sftp.closes == [DIR_HANDLE]

    # ...and an early break now closes the (previously leaked) handle too
    sftp = _scripted_client(*_two_page_responses())
    gen = sftp.listdir_iter(".")
    assert next(gen).filename == "alpha"
    gen.close()
    assert sftp.closes == [DIR_HANDLE]

    # a decode error in the legacy generator also closes exactly once
    sftp = _scripted_client(
        (CMD_NAME, _bad_name_reply()),
        (CMD_STATUS, _status_reply(SFTP_EOF, "eof")),
    )
    gen = sftp.listdir_iter(".")
    with pytest.raises(UnicodeDecodeError):
        next(gen)
    assert sftp.closes == [DIR_HANDLE]


# ---------------------------------------------------------------------------
# Short loopback regression against the real stub SFTP server (slow, like
# the rest of tests/test_sftp.py)
# ---------------------------------------------------------------------------

try:
    from ._util import slow

    @slow
    class TestOpendirLoopback:
        def test_listdir_iter_matches_listdir_attr(self, sftp):
            """legacy listdir_iter agrees with listdir_attr; handle closes."""
            expected = sorted(
                x.filename for x in sftp.listdir_attr(sftp.FOLDER)
            )
            got = sorted(x.filename for x in sftp.listdir_iter(sftp.FOLDER))
            assert got == expected

        def test_opendir_iteration_and_early_exit(self, sftp):
            # populate the freshly created stub folder with two local files
            # (the stub server serves the local working directory)
            import os

            for name in ("one.txt", "two.txt"):
                with open(os.path.join(sftp.FOLDER, name), "w") as fh:
                    fh.write(name)
            expected = sorted(
                x.filename for x in sftp.listdir_attr(sftp.FOLDER)
            )
            with sftp.opendir(sftp.FOLDER) as it:
                names = sorted(x.filename for x in it)
            assert names == expected
            # early exit on a non-empty folder leaves no dangling handle;
            # the server must still accept a fresh opendir afterwards
            it2 = sftp.opendir(sftp.FOLDER)
            next(it2)
            it2.close()
            it2.close()  # double close must be a client-side no-op
            assert sorted(
                x.filename for x in sftp.listdir_iter(sftp.FOLDER)
            ) == expected

except ImportError:
    # standalone execution: the loopback regression is simply not defined
    pass
