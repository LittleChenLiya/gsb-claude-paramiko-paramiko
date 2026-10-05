"""
Acceptance tests for paramiko.sftp_prefetch.PrefetchReader.

Run with:

    python -m pytest tests/test_sftp_prefetch.py -v

The reader is driven exclusively through FakeSFTPPackets, a fake SFTP
packet interface: the reader emits READ request packets via
``send_read`` / reclaims them via ``cancel``, and the tests push
response packets back with ``feed_response`` (via ``FakeSFTPPackets.serve``).
"""

from paramiko.sftp_prefetch import (
    HandleRequestIDs,
    PrefetchIncomplete,
    PrefetchReader,
    ReaderCancelled,
)

import pytest


class FakeSFTPPackets:
    """
    Fake SFTP packet transport.

    Logs every READ request packet the reader publishes, records
    cancellations, and lets tests deliver response packets (in any
    order, with short reads or EOF) back into a reader.
    """

    def __init__(self, files):
        self.files = files  # {handle: bytes}
        self.reads = []  # [(request_id, handle, offset, length)]
        self.cancelled = []  # [request_id]

    # -- packet interface used by PrefetchReader ----------------------

    def send_read(self, request_id, handle, offset, length):
        self.reads.append((request_id, handle, offset, length))

    def cancel(self, request_id):
        self.cancelled.append(request_id)

    # -- test-side helpers --------------------------------------------

    def request_for(self, handle=None, offset=None):
        for request_id, h, off, length in self.reads:
            if handle is not None and h != handle:
                continue
            if offset is not None and off != offset:
                continue
            return request_id, h, off, length
        raise KeyError(
            "no such request: handle=%r offset=%r" % (handle, offset)
        )

    def serve(self, reader, handle=None, offset=None, short=None, eof=False):
        """Deliver the response packet for the matching READ request."""
        request_id, h, off, length = self.request_for(handle, offset)
        data = self.files[h][off : off + length]
        if short is not None:
            data = data[:short]
        return reader.feed_response(request_id, data, eof=eof)


def test_reversed_responses_reassemble_by_requested_offset():
    # acceptance: responses for offsets 0 and 4 arrive reversed, the
    # reader must still output the correct eight bytes in order.
    fake = FakeSFTPPackets({b"h": b"ABCDEFGH"})
    reader = PrefetchReader(fake, b"h", file_size=8, chunk_size=4, window=2)
    assert [off for (_, _, off, _) in fake.reads] == [0, 4]
    fake.serve(reader, offset=4)  # second block arrives first: b"EFGH"
    fake.serve(reader, offset=0)  # first block arrives last:  b"ABCD"
    assert reader.read(8) == b"ABCDEFGH"


def test_three_packet_example():
    # three-packet example: blocks at offsets 0/4/8 arrive in reverse
    # packet order and are reassembled strictly by requested offset.
    fake = FakeSFTPPackets({b"h": b"ABCDEFGHIJKL"})
    reader = PrefetchReader(fake, b"h", file_size=12, chunk_size=4, window=3)
    assert [off for (_, _, off, _) in fake.reads] == [0, 4, 8]
    fake.serve(reader, offset=8)  # packet 3 arrives first:  b"IJKL"
    fake.serve(reader, offset=4)  # packet 2 arrives second: b"EFGH"
    fake.serve(reader, offset=0)  # packet 1 arrives last:   b"ABCD"
    assert reader.read(12) == b"ABCDEFGHIJKL"


def test_short_read_locates_real_gap_and_reports_incomplete():
    # acceptance: second block is a short read (2 of 4 bytes, no EOF);
    # the reader must pinpoint the real gap instead of silently
    # concatenating, and the read must fail as explicitly incomplete.
    fake = FakeSFTPPackets({b"h": b"ABCDEFGH"})
    reader = PrefetchReader(fake, b"h", file_size=8, chunk_size=4, window=2)
    fake.serve(reader, offset=0)  # full block: b"ABCD"
    fake.serve(reader, offset=4, short=2)  # short block: b"EF", hole at 6..8
    with pytest.raises(PrefetchIncomplete) as excinfo:
        reader.read(8)
    exc = excinfo.value
    assert exc.reason == "gap"
    assert exc.offset == 6  # the real gap, not the block start
    assert exc.length == 2
    assert exc.partial == b"ABCDEF"  # prefix delivered exactly once


def test_eof_marks_clean_end_and_reclaims_requests_beyond_it():
    fake = FakeSFTPPackets({b"h": b"ABCDEF"})
    reader = PrefetchReader(fake, b"h", file_size=16, chunk_size=4, window=4)
    assert reader.outstanding == 4
    fake.serve(reader, offset=0)
    fake.serve(reader, offset=4, short=2, eof=True)  # EOF pins end at 6
    # clean EOF: short return is legitimate, not an error
    assert reader.read(16) == b"ABCDEF"
    # requests past the real end (offsets 8 and 12) were reclaimed
    cancelled_offsets = sorted(
        off for rid, _, off, _ in fake.reads if rid in fake.cancelled
    )
    assert cancelled_offsets == [8, 12]


def test_duplicate_response_is_not_delivered_twice():
    fake = FakeSFTPPackets({b"h": b"ABCDEFGH"})
    reader = PrefetchReader(fake, b"h", file_size=8, chunk_size=4, window=2)
    rid0, _, _, _ = fake.request_for(offset=0)
    assert reader.feed_response(rid0, b"ABCD") is True
    # duplicate packet for the same request: dropped, first copy wins
    assert reader.feed_response(rid0, b"XXXX") is False
    assert reader.duplicates_dropped == 1
    fake.serve(reader, offset=4)
    assert reader.read(8) == b"ABCDEFGH"  # no duplicated/corrupted bytes


def test_prefetch_window_is_bounded():
    fake = FakeSFTPPackets({b"h": b"A" * 20})
    reader = PrefetchReader(fake, b"h", file_size=20, chunk_size=4, window=2)
    assert len(fake.reads) == 2  # only `window` requests in flight
    assert reader.outstanding == 2
    fake.serve(reader, offset=0)
    assert len(fake.reads) == 3  # one new request per completed one
    assert reader.outstanding == 2
    fake.serve(reader, offset=4)
    assert len(fake.reads) == 4
    assert reader.outstanding == 2


def test_read_before_data_arrives_reports_pending():
    fake = FakeSFTPPackets({b"h": b"ABCDEFGH"})
    reader = PrefetchReader(fake, b"h", file_size=8, chunk_size=4, window=2)
    with pytest.raises(PrefetchIncomplete) as excinfo:
        reader.read(8)
    assert excinfo.value.reason == "pending"
    assert excinfo.value.partial == b""


def test_cancel_reclaims_requests_and_drops_late_blocks():
    # acceptance: after cancel, this reader reclaims its requests and
    # late-arriving blocks are never published.
    fake = FakeSFTPPackets({b"h": b"ABCDEFGH"})
    reader = PrefetchReader(fake, b"h", file_size=8, chunk_size=4, window=2)
    rid0, _, _, _ = fake.request_for(offset=0)
    rid4, _, _, _ = fake.request_for(offset=4)
    reader.cancel()
    assert sorted(fake.cancelled) == sorted([rid0, rid4])
    assert reader.outstanding == 0
    # late packet for a reclaimed request: dropped, not published
    assert reader.feed_response(rid0, b"ABCD") is False
    assert reader.late_dropped == 1
    assert reader._blocks == {}
    with pytest.raises(ReaderCancelled):
        reader.read(8)


def test_request_ids_are_isolated_per_handle():
    # acceptance: request IDs are namespaced per handle, so cancelling
    # one reader never reclaims requests of another file handle.
    ids = HandleRequestIDs()
    fake = FakeSFTPPackets({b"A": b"aaaaaaaa", b"B": b"BBBBBBBB"})
    reader_a = PrefetchReader(
        fake, b"A", file_size=8, chunk_size=4, window=2, id_allocator=ids
    )
    reader_b = PrefetchReader(
        fake, b"B", file_size=8, chunk_size=4, window=2, id_allocator=ids
    )
    ids_a = [rid for rid, h, _, _ in fake.reads if h == b"A"]
    ids_b = [rid for rid, h, _, _ in fake.reads if h == b"B"]
    assert ids_a and ids_b
    assert all(rid[0] == b"A" for rid in ids_a)
    assert all(rid[0] == b"B" for rid in ids_b)
    assert not set(ids_a) & set(ids_b)  # disjoint ID namespaces

    reader_a.cancel()
    # every reclaimed ID belongs to handle A; handle B is untouched
    assert sorted(fake.cancelled) == sorted(ids_a)
    assert all(rid[0] == b"A" for rid in fake.cancelled)
    assert reader_b.outstanding == 2

    # handle B still completes normally, out of order
    fake.serve(reader_b, handle=b"B", offset=4)
    fake.serve(reader_b, handle=b"B", offset=0)
    assert reader_b.read(8) == b"BBBBBBBB"

    # and a late packet for the cancelled reader A is dropped
    assert reader_a.feed_response(ids_a[0], b"aaaa") is False
    assert reader_a.late_dropped == 1
