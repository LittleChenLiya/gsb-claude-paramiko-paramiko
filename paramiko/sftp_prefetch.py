"""
Explicit-offset SFTP prefetch reader.

Unlike a naive "append whatever arrives" prefetcher, this reader
reassembles response blocks by the *requested* offset of each read
request, never by packet arrival order:

- Blocks are keyed by requested offset, so out-of-order responses still
  reassemble into the correct byte stream.
- Short reads and EOF responses locate the *real* gap: the missing byte
  range is recorded explicitly instead of being silently skipped, and a
  read spanning a real hole raises :class:`PrefetchIncomplete`.
- Duplicate responses for an already-completed request are dropped and
  never delivered twice.
- The prefetch window is bounded: at most ``window`` read requests are
  outstanding at any time, and a new request is only issued when one
  completes.
- Request IDs are namespaced per file handle (see
  :class:`HandleRequestIDs`), so :meth:`PrefetchReader.cancel` reclaims
  exactly the requests of this reader and cannot touch other handles.
- After cancellation, late-arriving responses for reclaimed IDs are
  dropped, not published.

The reader talks to a minimal "fake SFTP packet" interface only; any
object with ``send_read(request_id, handle, offset, length)`` and
``cancel(request_id)`` works.  Responses are pushed back in via
:meth:`PrefetchReader.feed_response`.

Three-packet example (chunk_size=4, window=3, file = b"ABCDEFGHIJKL")::

    fake = FakeSFTPPackets({b"h": b"ABCDEFGHIJKL"})   # see tests
    reader = PrefetchReader(fake, b"h", file_size=12, chunk_size=4, window=3)
    # three READ packets are now in flight: offsets 0, 4, 8
    fake.serve(reader, offset=8)   # packet 3 arrives first:  b"IJKL"
    fake.serve(reader, offset=4)   # packet 2 arrives second: b"EFGH"
    fake.serve(reader, offset=0)   # packet 1 arrives last:   b"ABCD"
    assert reader.read(12) == b"ABCDEFGHIJKL"

Run the acceptance tests with::

    python -m pytest tests/test_sftp_prefetch.py -v
"""

from __future__ import annotations

import threading


class PrefetchIncomplete(Exception):
    """
    A read hit a real gap (short read / missing data) and is incomplete.

    Carries the ``offset`` and ``length`` of the real hole, a ``reason``
    (``"gap"``, ``"pending"`` or ``"missing"``), and any ``partial``
    bytes that were delivered before the hole was hit (those bytes have
    been consumed from the reader exactly once).
    """

    def __init__(self, offset, length, reason="gap", partial=b""):
        self.offset = offset
        self.length = length
        self.reason = reason
        self.partial = bytes(partial)
        super().__init__(
            "prefetch read incomplete: {} byte(s) missing at offset {} "
            "({})".format(length, offset, reason)
        )


class ReaderCancelled(Exception):
    """Raised when reading from a cancelled :class:`PrefetchReader`."""


class HandleRequestIDs:
    """
    Allocates request IDs namespaced per file handle.

    IDs are ``(handle, sequence)`` tuples, so two handles can never be
    issued the same ID, and reclaiming the IDs of one handle can never
    touch the requests of another handle.
    """

    def __init__(self):
        self._next = {}
        self._lock = threading.Lock()

    def allocate(self, handle):
        with self._lock:
            seq = self._next.get(handle, 0)
            self._next[handle] = seq + 1
        return (handle, seq)

    @staticmethod
    def belongs_to(request_id, handle):
        return (
            isinstance(request_id, tuple)
            and len(request_id) == 2
            and request_id[0] == handle
        )


class _Request:
    __slots__ = ("request_id", "offset", "length")

    def __init__(self, request_id, offset, length):
        self.request_id = request_id
        self.offset = offset
        self.length = length


class PrefetchReader:
    """
    Explicit-offset prefetching reader over a fake SFTP packet interface.

    :param packets: object with ``send_read(request_id, handle, offset,
        length)`` and ``cancel(request_id)``.
    :param handle: the SFTP file handle this reader reads from.
    :param file_size: total readable size in bytes (reads are clamped).
    :param chunk_size: per-request read length.
    :param window: maximum number of concurrently outstanding requests.
    :param id_allocator: shared :class:`HandleRequestIDs` (a private one
        is used if not given).
    """

    def __init__(
        self,
        packets,
        handle,
        file_size,
        chunk_size=32768,
        window=4,
        id_allocator=None,
    ):
        if window < 1:
            raise ValueError("window must be >= 1")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.packets = packets
        self.handle = handle
        self.file_size = file_size
        self.chunk_size = chunk_size
        self.window = window
        self._ids = (
            id_allocator if id_allocator is not None else HandleRequestIDs()
        )
        self._pos = 0
        self._next_request_offset = 0
        self._pending = {}  # request_id -> _Request (in flight)
        self._blocks = {}  # requested offset -> bytes (arrival order ignored)
        self._gaps = []  # [(offset, length)] real holes from short reads
        self._eof_at = None  # clean end-of-file offset, if reported
        self._cancelled = False
        # diagnostics / acceptance counters
        self.duplicates_dropped = 0
        self.late_dropped = 0
        self._issue_requests()

    # -- request side -------------------------------------------------

    def _issue_requests(self):
        """Top up the bounded prefetch window."""
        while not self._cancelled and len(self._pending) < self.window:
            offset = self._next_request_offset
            limit = self.file_size
            if self._eof_at is not None:
                limit = min(limit, self._eof_at)
            if offset >= limit:
                break
            length = min(self.chunk_size, limit - offset)
            request_id = self._ids.allocate(self.handle)
            self._pending[request_id] = _Request(request_id, offset, length)
            self._next_request_offset += length
            self.packets.send_read(request_id, self.handle, offset, length)

    @property
    def outstanding(self):
        """Number of in-flight requests (never exceeds ``window``)."""
        return len(self._pending)

    # -- response side ------------------------------------------------

    def feed_response(self, request_id, data=b"", eof=False):
        """
        Consume one response packet.  Returns True if it was accepted.

        The block is stored under the *requested* offset of
        ``request_id`` regardless of arrival order.  Duplicates of
        completed requests and late packets for reclaimed (cancelled)
        IDs are dropped and never delivered.
        """
        req = self._pending.pop(request_id, None)
        if req is None:
            if self._cancelled and self._ids.belongs_to(
                request_id, self.handle
            ):
                self.late_dropped += 1
            else:
                self.duplicates_dropped += 1
            return False

        data = bytes(data)
        if data:
            self._blocks[req.offset] = data
        shortfall = req.length - len(data)
        if shortfall > 0:
            hole = req.offset + len(data)
            if eof:
                # EOF pins the real end of file; reclaim requests past it.
                self._eof_at = (
                    hole if self._eof_at is None else min(self._eof_at, hole)
                )
                for rid, pending in list(self._pending.items()):
                    if pending.offset >= self._eof_at:
                        self.packets.cancel(rid)
                        del self._pending[rid]
            else:
                # short read without EOF: a real gap, record it exactly.
                self._gaps.append((hole, shortfall))
        self._issue_requests()
        return True

    # -- read side ----------------------------------------------------

    def _gap_at(self, pos):
        for offset, length in self._gaps:
            if offset <= pos < offset + length:
                return offset, length
        return None

    def _block_covering(self, pos):
        for offset, block in self._blocks.items():
            if offset <= pos < offset + len(block):
                return offset
        return None

    def _pending_covering(self, pos):
        for req in self._pending.values():
            if req.offset <= pos < req.offset + req.length:
                return True
        return False

    def read(self, n):
        """
        Read up to ``n`` bytes, reassembled strictly by offset.

        Returns fewer than ``n`` bytes only at a clean EOF.  Raises
        :class:`PrefetchIncomplete` when a real gap (short read, or data
        still pending/missing) is hit; the already-reassembled prefix is
        attached as ``exc.partial`` and consumed exactly once.
        """
        if self._cancelled:
            raise ReaderCancelled("reader has been cancelled")
        out = bytearray()
        pos = self._pos
        end = min(self._pos + n, self.file_size)
        while pos < end:
            if self._eof_at is not None and pos >= self._eof_at:
                break  # clean EOF: short return is legitimate
            gap = self._gap_at(pos)
            if gap is not None:
                self._pos = pos
                raise PrefetchIncomplete(
                    pos, gap[1] - (pos - gap[0]), "gap", partial=out
                )
            block_off = self._block_covering(pos)
            if block_off is None:
                reason = (
                    "pending" if self._pending_covering(pos) else "missing"
                )
                self._pos = pos
                raise PrefetchIncomplete(pos, end - pos, reason, partial=out)
            block = self._blocks[block_off]
            start = pos - block_off
            take = min(len(block) - start, end - pos)
            out += block[start : start + take]
            pos += take
        self._pos = pos
        return bytes(out)

    # -- teardown -----------------------------------------------------

    def cancel(self):
        """
        Reclaim every outstanding request of this reader.

        Only IDs allocated for the handle of this reader are cancelled,
        so other file handles are unaffected.  Late responses arriving
        afterwards are dropped by :meth:`feed_response`.
        """
        if self._cancelled:
            return
        self._cancelled = True
        for request_id in list(self._pending):
            # namespaced IDs guarantee we only ever reclaim requests
            # belonging to our own handle here.
            assert self._ids.belongs_to(request_id, self.handle)
            self.packets.cancel(request_id)
        self._pending.clear()
