"""
Acceptance tests for the closable directory-iteration entry point
(`.SFTPClient.listdir_iterator` / `.SFTPDirIterator`), plus a short
regression check that the original `.SFTPClient.listdir_iter` interface
behaves exactly as before.

Everything runs against an in-memory fake: no sockets, no threads, no
temp dirs.  The fake serves a two-page directory listing and tracks every
``SSH_FXP_CLOSE`` it receives, so tests can assert both the *count* of
close requests and the *handle* they target.
"""

import pytest

from paramiko.message import Message
from paramiko.sftp import (
    CMD_HANDLE,
    CMD_NAME,
    CMD_OPENDIR,
    CMD_READDIR,
    CMD_STATUS,
    CMD_CLOSE,
    SFTP_EOF,
    SFTP_OK,
)
from paramiko.sftp_client import SFTPClient, SFTPDirIterator

DIR_HANDLE = b"dir-handle-1"
# An unrelated, still-open *file* handle: the directory iterator must
# never send a close for it.
FILE_HANDLE = b"file-handle-9"


def name_packet(num, filenames):
    """Build a CMD_NAME response packet (one page of a listing)."""
    msg = Message()
    msg.add_int(num)
    msg.add_int(len(filenames))
    for fn in filenames:
        msg.add_string(fn)  # bytes, so invalid utf-8 can be injected
        msg.add_string("longname-of-{!r}".format(fn))
        msg.add_int(0)  # SFTPAttributes flags: no fields present
    return CMD_NAME, bytes(msg)


def status_packet(num, code=SFTP_EOF, text="end of directory"):
    msg = Message()
    msg.add_int(num)
    msg.add_int(code)
    msg.add_string(text)
    msg.add_string("en")
    return CMD_STATUS, bytes(msg)


class FakeSFTPClient(SFTPClient):
    """
    In-memory SFTPClient stub serving a scripted, two-page directory.

    Implements only the surface used by the listdir machinery
    (``_adjust_cwd``/``_log``/``_request``/``_async_request``/
    ``_read_packet``); no transport is created.
    """

    def __init__(self, pages):
        self._pages = list(pages)
        self._readdir_calls = 0
        self._next_num = 0
        self._pending = []
        self.closed_handles = []  # every handle passed to CMD_CLOSE
        self.open_handles = {DIR_HANDLE, FILE_HANDLE}

    def _adjust_cwd(self, path):
        return path

    def _log(self, *args, **kwargs):
        pass

    def _request(self, t, *args):
        if t == CMD_OPENDIR:
            msg = Message()
            msg.add_string(DIR_HANDLE)
            msg.rewind()
            return CMD_HANDLE, msg
        if t == CMD_CLOSE:
            self.closed_handles.append(args[0])
            self.open_handles.discard(args[0])
            msg = Message()
            msg.add_int(SFTP_OK)
            msg.add_string("ok")
            msg.add_string("en")
            return CMD_STATUS, msg
        raise AssertionError("unexpected request type {}".format(t))

    def _async_request(self, fileobj, t, *args):
        assert t == CMD_READDIR
        num = self._next_num
        self._next_num += 1
        page_idx = self._readdir_calls
        self._readdir_calls += 1
        if page_idx < len(self._pages):
            self._pending.append(name_packet(num, self._pages[page_idx]))
        else:
            self._pending.append(status_packet(num))
        return num

    def _read_packet(self):
        return self._pending.pop(0)


def make_client(bad=False):
    # Two pages: 2 entries, then 1 entry, then EOF.
    pages = [
        [b"alpha.txt", b"beta.txt"],
        [b"\xff\xfe-bad-name" if bad else b"gamma.txt"],
    ]
    return FakeSFTPClient(pages)


def assert_only_dir_handle_closed(sftp):
    assert sftp.closed_handles == [DIR_HANDLE]
    assert FILE_HANDLE in sftp.open_handles  # untouched


def test_explicit_close_on_early_exit():
    """Read one entry of two pages, then close(): exactly one close."""
    sftp = make_client()
    it = sftp.listdir_iterator("/tmp", read_aheads=2)
    assert isinstance(it, SFTPDirIterator)
    assert next(it).filename == "alpha.txt"
    it.close()
    it.close()  # idempotent: still exactly one close request
    assert_only_dir_handle_closed(sftp)


def test_context_manager_on_early_exit():
    """Same, but exiting a ``with`` block performs the close."""
    sftp = make_client()
    with sftp.listdir_iterator("/tmp", read_aheads=2) as it:
        assert next(it).filename == "alpha.txt"
    assert_only_dir_handle_closed(sftp)


def test_close_after_full_consumption():
    """Consuming both pages to EOF closes the handle exactly once."""
    sftp = make_client()
    with sftp.listdir_iterator("/tmp", read_aheads=2) as it:
        names = [attr.filename for attr in it]
    assert names == ["alpha.txt", "beta.txt", "gamma.txt"]
    assert_only_dir_handle_closed(sftp)


def test_close_on_decode_error():
    """A filename that fails utf-8 decoding still triggers one close."""
    sftp = make_client(bad=True)
    it = sftp.listdir_iterator("/tmp", read_aheads=2)
    assert next(it).filename == "alpha.txt"
    assert next(it).filename == "beta.txt"
    with pytest.raises(UnicodeDecodeError):
        next(it)  # second page contains the undecodable filename
    assert_only_dir_handle_closed(sftp)
    it.close()  # idempotent after an error
    assert_only_dir_handle_closed(sftp)


def test_listdir_iter_regression():
    """Original generator interface: same items, one close at EOF."""
    sftp = make_client()
    names = [
        attr.filename for attr in sftp.listdir_iter("/tmp", read_aheads=2)
    ]
    assert names == ["alpha.txt", "beta.txt", "gamma.txt"]
    assert_only_dir_handle_closed(sftp)
