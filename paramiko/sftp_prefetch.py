# Copyright (C) 2026 ...
#
# This file is part of paramiko.
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
Explicit SFTP prefetch reader.

The legacy prefetch buffer stitched DATA blocks into a single ``offset ->
data`` dict and spun forever when a duplicate response arrived.  This reader
is stricter:

* blocks are reassembled by the offset each request asked for, never by
  response arrival order;
* a short DATA block or an EOF STATUS closes the requested extent, so the
  real gap is recorded and missing bytes are reported as EOF instead of
  being silently stitched or re-requested;
* duplicate / late responses for an already-finished or cancelled request
  are ignored idempotently;
* the number of in-flight requests is bounded by a prefetch window;
* request ids are isolated per file handle, and cancel reclaims only this
  handle's in-flight requests from the shared session map, leaving other
  handles untouched.

The reader is driven explicitly: start issues the initial window, and read
pumps responses (via the SFTP packet interface) until it has the requested
bytes or reaches a real gap.  No background thread is used, which makes the
behaviour deterministic and testable with a fake packet interface.
"""

import threading

from paramiko.sftp import (
    CMD_DATA,
    CMD_READ,
    CMD_STATUS,
    SFTP_EOF,
    SFTP_OK,
    SFTPError,
    int64,
)


# Some SFTP servers choke on read requests larger than this.
MAX_REQUEST_SIZE = 32768


class PrefetchReader:
    """
    An explicit, bounded prefetch reader for one SFTP file handle.

    :param sftp:
        the SFTP session; must expose the same packet interface used by the
        rest of paramiko (``_async_request`` / ``_read_response`` /
        ``_cancel_requests``).
    :param handle:
        the remote file handle (an opaque byte string).
    :param int file_size:
        the total file size, if known.  When ``None`` the reader requests
        blocks until the server reports EOF.
    :param int window:
        the maximum number of read requests allowed in flight at once.
    :param int request_size:
        the size of each read request (defaults to ``MAX_REQUEST_SIZE``).
    """

    def __init__(
        self,
        sftp,
        handle,
        file_size=None,
        window=64,
        request_size=MAX_REQUEST_SIZE,
    ):
        self.sftp = sftp
        self.handle = handle
        self.file_size = file_size
        self.window = window
        self.request_size = request_size

        self._lock = threading.RLock()
        # offset -> bytes received for that request (full or partial).  Keyed
        # by the offset the request asked for, never by arrival order.
        self._blocks = {}
        # num -> (offset, length) for requests currently in flight.
        self._inflight = {}
        # offsets whose requested extent was closed by a short read / EOF:
        # offset -> bytes actually covered (== requested => complete,
        # < requested => a real gap).
        self._closed = {}
        # offset at which the first real gap begins (None == no gap yet).
        self._gap_start = None
        # next offset to request (kept contiguous from 0).
        self._next_offset = 0
        # current read position.
        self._pos = 0
        self._started = False
        self._eof = False
        self._cancelled = False
        self._exc = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def start(self):
        """
        Issue the initial prefetch window of requests.

        Safe to call more than once; subsequent calls are no-ops.
        """
        with self._lock:
            if self._started:
                return
            self._started = True
        self._replenish()

    def read(self, size=-1):
        """
        Read up to ``size`` bytes, reassembling blocks by request offset.

        Returns fewer bytes than requested when a real gap (short read /
        EOF) is reached; check :attr:`eof` / :attr:`complete` to tell a
        short read from a full one.  When ``size`` is negative, reads until
        EOF.

        :raises SFTPError: if the reader was cancelled or a fatal status was
            received.
        """
        if size is None:
            size = -1
        if size == 0:
            return b""
        with self._lock:
            if self._cancelled:
                raise SFTPError("PrefetchReader is cancelled")
            if self._exc is not None:
                raise self._exc

        out = b""
        while True:
            with self._lock:
                if self._cancelled:
                    raise SFTPError("PrefetchReader is cancelled")
                if self._exc is not None:
                    raise self._exc
                want = size - len(out) if size > 0 else self.request_size
                chunk = self._serve_locked(want)
                if chunk:
                    out += chunk
                    if size > 0 and len(out) >= size:
                        break
                    continue
                # no contiguous data at _pos.  only stop once _pos has
                # actually reached a real gap; a gap at a higher offset must
                # not starve lower-offset blocks still in flight.
                if self._eof and self._pos >= (self._gap_start or 0):
                    break
            # need a response: pump one packet (blocks until it arrives)
            try:
                self._pump()
            except EOFError:
                # no more packets and no contiguous data -> real gap
                break
        return out

    def cancel(self):
        """
        Cancel the reader and reclaim this handle's in-flight requests.

        Only requests belonging to this reader's handle are removed from the
        shared session map; other file handles are untouched.  Late responses
        arriving after cancellation are dropped.
        """
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            nums = list(self._inflight)
            self._inflight.clear()
            self._eof = True
        self.sftp._cancel_requests(self.handle, nums)

    @property
    def eof(self):
        """``True`` once a real gap (short read / EOF) has been reached."""
        with self._lock:
            return self._eof

    @property
    def cancelled(self):
        with self._lock:
            return self._cancelled

    @property
    def complete(self):
        """
        ``True`` if the requested range was read in full, without hitting a
        real gap.
        """
        with self._lock:
            if self._cancelled or self._exc is not None:
                return False
            if self.file_size is not None:
                return self._pos >= self.file_size
            return self._eof

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _serve_locked(self, want):
        """
        Serve up to ``want`` contiguous bytes starting at ``self._pos``.

        Must be called with ``self._lock`` held.  Returns ``b""`` when no
        contiguous byte is available at ``self._pos`` (either a gap or data
        not yet arrived).
        """
        # find the block covering _pos (largest offset <= _pos)
        candidates = [o for o in self._blocks.keys() if o <= self._pos]
        if not candidates:
            return b""
        off = max(candidates)
        data = self._blocks[off]
        end = off + len(data)
        if end <= self._pos:
            # this block ends at/before _pos; nothing contiguous here
            return b""
        take = min(end - self._pos, want)
        chunk = data[self._pos - off : self._pos - off + take]
        self._pos += take
        if self._pos >= end:
            # fully consumed this block; drop it so it is never re-delivered
            del self._blocks[off]
        return chunk

    def _pump(self):
        """
        Read one response packet and dispatch it to _async_response, then
        replenish the prefetch window.
        """
        self.sftp._read_response()
        self._replenish()

    def _replenish(self):
        """
        Issue new requests until the in-flight window is full or the file
        size / EOF is reached.
        """
        with self._lock:
            if self._cancelled or self._eof:
                return
            while len(self._inflight) < self.window:
                offset = self._next_offset
                if self.file_size is not None and offset >= self.file_size:
                    break
                length = self.request_size
                if self.file_size is not None:
                    length = min(length, self.file_size - offset)
                num = self.sftp._async_request(
                    self,
                    CMD_READ,
                    self.handle,
                    int64(offset),
                    int(length),
                )
                self._inflight[num] = (offset, length)
                self._next_offset = offset + length

    def _async_response(self, t, msg, num):
        """
        Dispatch a response for one of this reader's requests.

        Called by the SFTP packet interface.  Duplicate / late responses for
        an unknown or already-finished request are ignored idempotently.
        """
        with self._lock:
            if self._cancelled or num not in self._inflight:
                # late / duplicate / cancelled: never deliver twice
                return
            offset, length = self._inflight.pop(num)

            if t == CMD_STATUS:
                code = msg.get_int()
                if code == SFTP_EOF:
                    # the file ends at this request's offset
                    self._close_gap(offset, 0)
                elif code == SFTP_OK:
                    # success no-op (matches SFTPClient._convert_status);
                    # the request yielded no data but is not an error/EOF
                    pass
                else:
                    self._exc = SFTPError(
                        "SFTP read failed with status code {}".format(code)
                    )
                    self._eof = True
                return

            if t != CMD_DATA:
                self._exc = SFTPError(
                    "Expected DATA response, got command {}".format(t)
                )
                self._eof = True
                return

            data = msg.get_string()
            got = len(data)
            if got >= length:
                # full block: store exactly the requested bytes
                self._blocks[offset] = data[:length]
            else:
                # short read: the extent [offset, offset+length) is only
                # covered up to offset+got; the rest is a real EOF gap
                self._blocks[offset] = data
                self._close_gap(offset, got)

    def _close_gap(self, offset, got):
        """
        Record that the extent starting at ``offset`` is only covered up to
        ``offset + got``; the remainder is a real gap (EOF).
        """
        self._closed[offset] = got
        gap = offset + got
        if self._gap_start is None or gap < self._gap_start:
            self._gap_start = gap
        self._eof = True
