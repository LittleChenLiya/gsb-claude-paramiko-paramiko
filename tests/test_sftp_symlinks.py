"""
Acceptance tests for root-constrained, component-by-component remote
symbolic-link resolution (`.SFTPClient.resolve_symlinks` and the
``resolve_root`` option of `.SFTPClient.open`).

The "remote server" here is a tiny in-memory fake SFTP tree that only
implements ``lstat`` / ``readlink``; it records every call and deliberately
has no ``open`` implementation, so the resolver cannot open a target file
even by accident. No Docker, no sockets and no local filesystem path
resolution are involved.
"""

import errno
import stat

import pytest

from paramiko import SFTPClient, SFTPFile
from paramiko.sftp import (
    CMD_HANDLE,
    CMD_OPEN,
    SFTPLinkLoopError,
    SFTPRootEscapeError,
)
from paramiko.sftp_attr import SFTPAttributes


def _attrs(mode):
    attr = SFTPAttributes()
    attr.st_mode = mode
    return attr


REG = stat.S_IFREG | 0o644
DIR = stat.S_IFDIR | 0o755
LNK = stat.S_IFLNK | 0o777


class RecordingFakeSFTP:
    """
    In-memory stand-in for the remote end of an SFTP session.

    ``tree`` maps absolute POSIX paths to either an integer mode (regular
    file / directory) or a ``("link", target)`` tuple. Only the two methods
    the resolver is allowed to use exist; there is deliberately no
    ``open``/``file`` method, so any attempt to open a target through this
    fake raises ``AttributeError`` instead of silently passing.
    """

    def __init__(self, tree):
        self.tree = tree
        self.lstat_calls = []
        self.readlink_calls = []

    def lstat(self, path):
        path = path.decode() if isinstance(path, bytes) else path
        self.lstat_calls.append(path)
        if path not in self.tree:
            raise IOError(errno.ENOENT, "no such file")
        entry = self.tree[path]
        if isinstance(entry, tuple) and entry[0] == "link":
            return _attrs(LNK)
        return _attrs(entry)

    def readlink(self, path):
        path = path.decode() if isinstance(path, bytes) else path
        self.readlink_calls.append(path)
        return self.tree[path][1]


def build_tree():
    """
    Acceptance fixture: a chroot-like tree with plain relative links, a
    nested relative chain, a two-node loop, a self loop, and links that
    escape the root both relatively and absolutely.
    """
    root = "/srv"
    return root, RecordingFakeSFTP(
        {
            "/srv": DIR,
            "/srv/data": DIR,
            "/srv/data/secret.txt": REG,
            "/srv/www": DIR,
            # relative link: interpreted relative to the link directory
            "/srv/www/alias.txt": ("link", "../data/secret.txt"),
            # nested relative chain: a -> b -> ../../data/secret.txt
            "/srv/www/sub": DIR,
            "/srv/www/sub/a": ("link", "b"),
            "/srv/www/sub/b": ("link", "../../data/secret.txt"),
            # two-node loop
            "/srv/loop": DIR,
            "/srv/loop/x": ("link", "y"),
            "/srv/loop/y": ("link", "x"),
            # self loop
            "/srv/loop/self": ("link", "self"),
            # relative escape via ".."
            "/srv/evil_rel.txt": ("link", "../../etc/passwd"),
            # absolute escape outside the root
            "/srv/evil_abs.txt": ("link", "/etc/passwd"),
            # link pointing into the two-node loop
            "/srv/into_loop": ("link", "loop/x"),
            # outside the root: must never be reachable
            "/etc": DIR,
            "/etc/passwd": REG,
        }
    )


def test_relative_link_resolves_from_link_directory():
    from paramiko.sftp_client import _resolve_link_chain

    root, sftp = build_tree()
    resolved, chain = _resolve_link_chain(sftp, "/srv/www/alias.txt", root, 40)
    assert resolved == "/srv/data/secret.txt"
    assert chain == [("/srv/www/alias.txt", "../data/secret.txt")]
    assert sftp.readlink_calls == ["/srv/www/alias.txt"]
    assert "/srv/data/secret.txt" in sftp.lstat_calls
    # zero file opens: the fake has no open method at all
    assert not hasattr(sftp, "open")


def test_nested_relative_chain():
    from paramiko.sftp_client import _resolve_link_chain

    root, sftp = build_tree()
    resolved, chain = _resolve_link_chain(sftp, "/srv/www/sub/a", root, 40)
    assert resolved == "/srv/data/secret.txt"
    assert chain == [
        ("/srv/www/sub/a", "b"),
        ("/srv/www/sub/b", "../../data/secret.txt"),
    ]
    assert not hasattr(sftp, "open")


def test_two_node_loop_raises_link_loop_error():
    from paramiko.sftp_client import _resolve_link_chain

    root, sftp = build_tree()
    with pytest.raises(SFTPLinkLoopError) as excinfo:
        _resolve_link_chain(sftp, "/srv/loop/x", root, 40)
    err = excinfo.value
    assert err.root == root
    paths = [link for link, _ in err.chain]
    assert paths == ["/srv/loop/x", "/srv/loop/y"]
    assert err.link == "/srv/loop/x"
    assert "loop" in str(err)
    # outside content is never inspected
    assert "/etc/passwd" not in sftp.lstat_calls


def test_self_loop_raises_link_loop_error():
    from paramiko.sftp_client import _resolve_link_chain

    root, sftp = build_tree()
    with pytest.raises(SFTPLinkLoopError):
        _resolve_link_chain(sftp, "/srv/loop/self", root, 40)


def test_relative_escape_raises_root_escape_error():
    from paramiko.sftp_client import _resolve_link_chain

    root, sftp = build_tree()
    with pytest.raises(SFTPRootEscapeError) as excinfo:
        _resolve_link_chain(sftp, "/srv/evil_rel.txt", root, 40)
    err = excinfo.value
    assert not err.chain[0][1].startswith("/")  # relative target
    assert err.link == "/srv/evil_rel.txt"
    assert "root" in str(err)
    # containment is enforced before lstat-ing the outside target
    assert "/etc/passwd" not in sftp.lstat_calls


def test_absolute_escape_raises_root_escape_error():
    from paramiko.sftp_client import _resolve_link_chain

    root, sftp = build_tree()
    with pytest.raises(SFTPRootEscapeError) as excinfo:
        _resolve_link_chain(sftp, "/srv/evil_abs.txt", root, 40)
    assert excinfo.value.root == root
    assert "/etc/passwd" not in sftp.lstat_calls


def test_dotdot_step_without_link_is_contained():
    from paramiko.sftp_client import _resolve_link_chain

    root, sftp = build_tree()
    resolved, chain = _resolve_link_chain(
        sftp, "/srv/www/../data/secret.txt", root, 40
    )
    assert resolved == "/srv/data/secret.txt"
    assert chain == []


def test_plain_dotdot_escape_rejected_before_any_lstat():
    from paramiko.sftp_client import _resolve_link_chain

    root, sftp = build_tree()
    with pytest.raises(SFTPRootEscapeError):
        _resolve_link_chain(sftp, "/srv/../etc/passwd", root, 40)
    assert sftp.lstat_calls == []
    assert sftp.readlink_calls == []


def test_link_into_loop_reports_full_chain():
    from paramiko.sftp_client import _resolve_link_chain

    root, sftp = build_tree()
    with pytest.raises(SFTPLinkLoopError) as excinfo:
        _resolve_link_chain(sftp, "/srv/into_loop", root, 40)
    paths = [link for link, _ in excinfo.value.chain]
    assert paths == ["/srv/into_loop", "/srv/loop/x", "/srv/loop/y"]


def test_max_links_limit():
    from paramiko.sftp_client import _resolve_link_chain

    root, sftp = build_tree()
    with pytest.raises(SFTPLinkLoopError) as excinfo:
        _resolve_link_chain(sftp, "/srv/into_loop", root, max_links=2)
    assert excinfo.value.limit == 2
    assert len(excinfo.value.chain) == 3


def test_missing_trailing_component_resolves_non_strict():
    from paramiko.sftp_client import _resolve_link_chain

    root, sftp = build_tree()
    resolved, chain = _resolve_link_chain(
        sftp, "/srv/www/alias.txt.new", root, 40
    )
    assert resolved == "/srv/www/alias.txt.new"
    assert chain == []


# ---------------------------------------------------------------------------
# Public SFTPClient surface, including the "option disabled" backwards
# compatibility contract of open().
# ---------------------------------------------------------------------------


class StubSFTPClient(SFTPClient):
    """SFTPClient bypassing socket/version negotiation."""

    def __init__(self):
        self._cwd = None
        self._fake = build_tree()[1]

    def lstat(self, path):
        return self._fake.lstat(path)

    def readlink(self, path):
        return self._fake.readlink(path)

    def _log(self, level, msg, *args):
        pass


def test_resolve_symlinks_public_method_relative_input():
    client = StubSFTPClient()
    resolved, chain = client.resolve_symlinks("www/alias.txt", "/srv")
    assert resolved == "/srv/data/secret.txt"
    assert chain == [("/srv/www/alias.txt", "../data/secret.txt")]


def test_resolve_symlinks_rejects_non_absolute_root():
    client = StubSFTPClient()
    with pytest.raises(Exception):
        client.resolve_symlinks("www/alias.txt", "srv")


def test_resolve_symlinks_rejects_input_outside_root():
    client = StubSFTPClient()
    with pytest.raises(SFTPRootEscapeError):
        client.resolve_symlinks("/etc/passwd", "/srv")


class _FakeMsg:
    def __init__(self, handle):
        self._handle = handle

    def get_binary(self):
        return self._handle


def _install_dummy_sftp_file():
    import paramiko.sftp_client as sc

    class _DummyFile:
        def __init__(self, sftp, handle, mode, bufsize):
            self.handle = handle

    sc.SFTPFile = _DummyFile
    return sc


def test_open_without_option_keeps_original_path_and_no_resolution():
    client = StubSFTPClient()
    calls = []

    def fake_request(t, *args):
        calls.append((t, args))
        return CMD_HANDLE, _FakeMsg(b"h")

    client._request = fake_request

    def boom(*a, **k):
        raise AssertionError("resolver must not run when resolve_root unset")

    client.resolve_symlinks = boom
    sc = _install_dummy_sftp_file()
    try:
        f = client.open("www/alias.txt", "rb")
        assert f.handle == b"h"
    finally:
        sc.SFTPFile = SFTPFile

    open_calls = [a for t, a in calls if t == CMD_OPEN]
    assert len(open_calls) == 1
    # path passed through verbatim; no lstat/readlink traffic either
    assert open_calls[0][0] == b"www/alias.txt"
    assert client._fake.lstat_calls == []
    assert client._fake.readlink_calls == []


def test_open_with_resolve_root_uses_resolved_path():
    client = StubSFTPClient()
    seen = []

    def fake_request(t, *args):
        seen.append((t, args))
        return CMD_HANDLE, _FakeMsg(b"h")

    client._request = fake_request
    sc = _install_dummy_sftp_file()
    try:
        client.open("/srv/www/alias.txt", "rb", resolve_root="/srv")
    finally:
        sc.SFTPFile = SFTPFile

    open_args = [a for t, a in seen if t == CMD_OPEN]
    assert len(open_args) == 1
    assert open_args[0][0] == b"/srv/data/secret.txt"
    assert client._fake.readlink_calls == ["/srv/www/alias.txt"]


def test_open_with_resolve_root_blocks_loop():
    client = StubSFTPClient()

    def boom(*a, **k):
        raise AssertionError("no open request must reach the fake server")

    client._request = boom
    with pytest.raises(SFTPLinkLoopError):
        client.open("/srv/loop/x", "rb", resolve_root="/srv")
    assert client._fake.readlink_calls == [
        "/srv/loop/x",
        "/srv/loop/y",
    ]
