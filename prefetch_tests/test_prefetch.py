"""
Packet-level tests for the explicit SFTP prefetch reader.

These tests drive ``PrefetchReader`` entirely through a *fake* SFTP packet
interface (``FakeSFTP``): no SSH transport, no loopback server, no real
socket.  Each test injects raw SFTP packets (``CMD_DATA`` / ``CMD_STATUS``)
and asserts on the reassembled output.

Acceptance scenarios covered:

* out-of-order responses (offset 4 before offset 0) still reassemble into
  the correct 8 bytes;
* a short second block is reported as clearly incomplete (real gap);
* after cancellation, late blocks are no longer published and other file
  handles are unaffected;
* request ids are isolated per file handle;
* the prefetch window is bounded;
* duplicate responses are delivered at most once.

Run with::

    pytest prefetch_tests/test_prefetch.py -v
"""

import queue
import threading

import pytest

from paramiko.message import Message
from paramiko.sftp import (
    CMD_DATA,
    CMD_STATUS,
    SFTP_EOF,
    SFTP_FAILURE,
    SFTP_OK,
    SFTPError,
)
from paramiko.sftp_prefetch import PrefetchReader


# ----------------------------------------------------------------------
# fake SFTP packet interface
# ----------------------------------------------------------------------


class FakeSFTP:
    """
    A minimal fake SFTP client that speaks the same packet interface the
    prefetch machinery uses (``_async_request`` / ``_read_response`` /
    ``_cancel_requests``), but with in-memory queues instead of a socket.

    Request ids are allocated *per file handle*, so two handles can each
    have an in-flight request #1 without colliding.  Incoming packets are
    injected by the test as ``(t, handle, payload)`` tuples where
    ``payload`` is the raw SFTP packet body (id + body), exactly as a real
    server would send.
    """

    def __init__(self, timeout=5.0):
        self._lock = threading.Lock()
        # (handle, num) -> fileobj
        self._expecting = {}
        # handle -> next request id (per-handle isolation)
        self._counters = {}
        # (num, handle, offset, length) of every request sent
        self.sent = []
        self._inbox = queue.Queue()
        self._timeout = timeout

    def _async_request(self, fileobj, t, *args):
        # args for CMD_READ: (handle, int64(offset), int(length))
        handle = args[0]
        with self._lock:
            num = self._counters.get(handle, 1)
            self._counters[handle] = num + 1
            key = (handle, num)
            if key in self._expecting:
                raise AssertionError("duplicate request id for handle")
            self._expecting[key] = fileobj
        self.sent.append((num, handle, int(args[1]), int(args[2])))
        return num

    def _read_response(self, waitfor=None):
        try:
            t, handle, payload = self._inbox.get(timeout=self._timeout)
        except queue.Empty:
            raise EOFError("no more packets")
        msg = Message(payload)
        num = msg.get_int()
        key = (handle, num)
        with self._lock:
            if key not in self._expecting:
                # late / cancelled / duplicate: drop it on the floor
                return None, None
            fileobj = self._expecting.pop(key)
        if fileobj is not None:
            fileobj._async_response(t, msg, num)
        return t, msg

    def _cancel_requests(self, handle, nums):
        with self._lock:
            for num in nums:
                self._expecting.pop((handle, num), None)

    # -- test helpers --------------------------------------------------

    def inject(self, t, handle, num, body=b""):
        """Inject a raw SFTP packet (id + body) for ``handle``."""
        payload = Message().add_int(num).asbytes() + body
        self._inbox.put((t, handle, payload))

    def inflight_for(self, handle):
        with self._lock:
            return [num for (h, num) in self._expecting if h == handle]


def _data_packet(num, data):
    return Message().add_string(data).asbytes()


def _status_packet(num, code):
    return Message().add_int(code).asbytes()


# ----------------------------------------------------------------------
# packet-level acceptance tests
# ----------------------------------------------------------------------


def test_out_of_order_packets_reassemble_by_offset():
    """
    The acceptance scenario: offset 0 and offset 4 responses arrive in
    reverse order, yet the reader reassembles the correct 8 bytes.

    Three packets are injected: DATA(offset=4), DATA(offset=0), then an
    EOF STATUS.  Reassembly must follow the request offsets, not the
    arrival order.
    """
    fake = FakeSFTP()
    handle = b"handle-oo"
    reader = PrefetchReader(fake, handle, file_size=8, window=2, request_size=4)
    reader.start()
    nums = [s[0] for s in fake.sent]
    offsets = [s[2] for s in fake.sent]
    assert offsets == [0, 4]

    # arrival order is deliberately reversed: offset 4 first, offset 0 second
    fake.inject(CMD_DATA, handle, nums[1], _data_packet(nums[1], b"efgh"))
    fake.inject(CMD_DATA, handle, nums[0], _data_packet(nums[0], b"abcd"))
    fake.inject(CMD_STATUS, handle, nums[0], _status_packet(nums[0], SFTP_EOF))

    out = reader.read(8)
    assert out == b"abcdefgh"
    assert reader.complete


def test_short_second_block_is_clearly_incomplete():
    """
    When the second block is short-read (2 bytes instead of 4), the reader
    must report the read as clearly incomplete: it returns the 6 bytes it
    could reassemble, sets ``eof``, and does not claim ``complete``.
    """
    fake = FakeSFTP()
    handle = b"handle-short"
    reader = PrefetchReader(fake, handle, file_size=8, window=2, request_size=4)
    reader.start()
    nums = [s[0] for s in fake.sent]

    fake.inject(CMD_DATA, handle, nums[1], _data_packet(nums[1], b"ef"))  # short
    fake.inject(CMD_DATA, handle, nums[0], _data_packet(nums[0], b"abcd"))
    fake.inject(CMD_STATUS, handle, nums[0], _status_packet(nums[0], SFTP_EOF))

    out = reader.read(8)
    assert out == b"abcdef"
    assert len(out) == 6
    assert reader.eof
    assert not reader.complete


def test_cancel_drops_late_packets_and_isolates_other_handles():
    """
    After cancelling handle A, a late DATA packet for A must not be
    published, while a separate handle B continues to work normally.
    """
    fake = FakeSFTP()
    ha = b"handle-A"
    hb = b"handle-B"
    ra = PrefetchReader(fake, ha, file_size=8, window=2, request_size=4)
    rb = PrefetchReader(fake, hb, file_size=8, window=2, request_size=4)
    ra.start()
    rb.start()
    nums_a = [s[0] for s in fake.sent if s[1] == ha]
    nums_b = [s[0] for s in fake.sent if s[1] == hb]

    # per-handle ids: both handles can have an in-flight request #1
    assert nums_a[0] == 1
    assert nums_b[0] == 1

    ra.cancel()
    assert ra.cancelled
    # A's in-flight requests were reclaimed from the shared map
    assert fake.inflight_for(ha) == []

    # late packet for A arriving after cancel -> must be dropped
    fake.inject(CMD_DATA, ha, nums_a[0], _data_packet(nums_a[0], b"ZZZZ"))

    # B is unaffected and still reassembles correctly
    fake.inject(CMD_DATA, hb, nums_b[0], _data_packet(nums_b[0], b"abcd"))
    fake.inject(CMD_DATA, hb, nums_b[1], _data_packet(nums_b[1], b"efgh"))
    fake.inject(CMD_STATUS, hb, nums_b[0], _status_packet(nums_b[0], SFTP_EOF))
    out_b = rb.read(8)
    assert out_b == b"abcdefgh"
    assert rb.complete

    # the late packet never made it into A's buffer
    assert ra._blocks == {}
    # reading a cancelled reader is explicit
    with pytest.raises(Exception):
        ra.read(8)


def test_request_ids_are_isolated_per_handle():
    """
    Request ids are allocated per file handle: two handles each get their
    own #1, and a response for one handle never matches the other.
    """
    fake = FakeSFTP()
    h1 = b"h1"
    h2 = b"h2"
    r1 = PrefetchReader(fake, h1, file_size=4, window=1, request_size=4)
    r2 = PrefetchReader(fake, h2, file_size=4, window=1, request_size=4)
    r1.start()
    r2.start()
    n1 = [s[0] for s in fake.sent if s[1] == h1][0]
    n2 = [s[0] for s in fake.sent if s[1] == h2][0]
    assert n1 == 1
    assert n2 == 1

    # response for h1 must not be delivered to h2
    fake.inject(CMD_DATA, h1, n1, _data_packet(n1, b"1111"))
    fake.inject(CMD_DATA, h2, n2, _data_packet(n2, b"2222"))
    assert r1.read(4) == b"1111"
    assert r2.read(4) == b"2222"


def test_prefetch_window_is_bounded():
    """
    The number of in-flight requests never exceeds the configured window,
    even as responses arrive and the reader replenishes.
    """
    fake = FakeSFTP()
    handle = b"handle-win"
    reader = PrefetchReader(fake, handle, file_size=12, window=2, request_size=4)
    reader.start()
    nums = [s[0] for s in fake.sent]
    assert len(nums) == 2  # initial window is exactly 2
    assert len(fake.inflight_for(handle)) == 2

    for n in nums:
        fake.inject(CMD_DATA, handle, n, _data_packet(n, b"xxxx"))
    out = reader.read(8)
    assert out == b"xxxxxxxx"
    # replenishment kept the window bounded
    assert len(fake.inflight_for(handle)) <= 2

    # drain the rest of the file
    for s in fake.sent:
        if s[0] not in nums:
            fake.inject(CMD_DATA, handle, s[0], _data_packet(s[0], b"yyyy"))
    out2 = reader.read(4)
    assert out2 == b"yyyy"
    assert reader.complete


def test_duplicate_response_is_delivered_once():
    """
    A duplicate DATA packet for an already-finished request must not be
    delivered twice (no double bytes, no spin).
    """
    fake = FakeSFTP()
    handle = b"handle-dup"
    reader = PrefetchReader(fake, handle, file_size=8, window=2, request_size=4)
    reader.start()
    nums = [s[0] for s in fake.sent]

    fake.inject(CMD_DATA, handle, nums[0], _data_packet(nums[0], b"abcd"))
    # duplicate of the same request id, with different payload
    fake.inject(CMD_DATA, handle, nums[0], _data_packet(nums[0], b"ZZZZ"))
    fake.inject(CMD_DATA, handle, nums[1], _data_packet(nums[1], b"efgh"))
    fake.inject(CMD_STATUS, handle, nums[0], _status_packet(nums[0], SFTP_EOF))

    out = reader.read(8)
    assert out == b"abcdefgh"
    assert reader.complete


def test_eof_status_without_data_returns_empty():
    """
    An EOF STATUS with no preceding DATA means the file is empty at that
    offset; read returns b"" and reports EOF.
    """
    fake = FakeSFTP()
    handle = b"handle-eof"
    reader = PrefetchReader(fake, handle, file_size=8, window=2, request_size=4)
    reader.start()
    nums = [s[0] for s in fake.sent]

    fake.inject(CMD_STATUS, handle, nums[0], _status_packet(nums[0], SFTP_EOF))
    out = reader.read(8)
    assert out == b""
    assert reader.eof
    assert not reader.complete


def test_failure_status_sets_exception():
    """
    A non-EOF error STATUS (e.g. SFTP_FAILURE) is recorded as an exception
    and raised on the next read, rather than being silently treated as EOF.
    """
    fake = FakeSFTP()
    handle = b"handle-fail"
    reader = PrefetchReader(fake, handle, file_size=4, window=1, request_size=4)
    reader.start()
    nums = [s[0] for s in fake.sent]

    fake.inject(CMD_STATUS, handle, nums[0], _status_packet(nums[0], SFTP_FAILURE))
    with pytest.raises(SFTPError):
        reader.read(4)


def test_sftp_ok_status_is_not_treated_as_eof():
    """
    A STATUS with code SFTP_OK is a success no-op (matching the real
    client's _convert_status): it is neither an error nor EOF.
    """
    fake = FakeSFTP()
    handle = b"handle-ok"
    reader = PrefetchReader(fake, handle, file_size=4, window=1, request_size=4)
    reader.start()
    nums = [s[0] for s in fake.sent]

    fake.inject(CMD_STATUS, handle, nums[0], _status_packet(nums[0], SFTP_OK))
    # OK is a no-op: no exception, not EOF
    assert not reader.eof
    assert reader._exc is None
