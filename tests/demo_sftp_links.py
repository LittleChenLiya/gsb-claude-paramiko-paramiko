#!/usr/bin/env python
"""
Tiny, dependency free demonstration of the root-constrained SFTP symlink
resolver.

Run it directly:

    python tests/demo_sftp_links.py

It builds an in-memory fake SFTP tree (no socket, no real server, no disk
files and no OS symlink privilege required) and shows, for a relative link,
a two-node link cycle and an out-of-root link:

* the final resolved path (or the raised link error)
* the error's link chain
* that zero file-open calls happened during resolution
"""

import errno
import os
import stat
import sys

# allow running straight from a checkout: ``python tests/demo_sftp_links.py``
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paramiko.sftp_client import SFTPClient, SFTPLinkError  # noqa: E402
from paramiko import SFTPAttributes  # noqa: E402

DIR_MODE = stat.S_IFDIR | 0o755
FILE_MODE = stat.S_IFREG | 0o644
LINK_MODE = stat.S_IFLNK | 0o777


class FakeSFTP:
    """Minimal in-memory SFTP client exposing only lstat/readlink."""

    def __init__(self, tree):
        self.tree = tree
        self.lstat_count = 0
        self.readlink_count = 0
        self.opens = 0
        self._cwd = None

    _adjust_cwd = SFTPClient._adjust_cwd
    resolve_links = SFTPClient.resolve_links

    def _log(self, *args):
        pass

    def lstat(self, path):
        self.lstat_count += 1
        entry = self.tree[path]
        attr = SFTPAttributes()
        attr.st_mode = LINK_MODE if isinstance(entry, str) else entry
        return attr

    def readlink(self, path):
        self.readlink_count += 1
        return self.tree[path]

    def open(self, *args, **kwargs):
        # If this ever fires, the "never opens the target" contract broke.
        self.opens += 1
        raise AssertionError("resolve_links must not open files")


TREE = {
    "/sandbox": DIR_MODE,
    "/sandbox/dir": DIR_MODE,
    "/sandbox/dir/real.txt": FILE_MODE,
    # relative link, resolved from the link's own directory
    "/sandbox/rel": "dir/real.txt",
    # two-node cycle
    "/sandbox/loop1": "loop2",
    "/sandbox/loop2": "loop1",
    # relative link escaping the root via ..
    "/sandbox/escape": "../secret.txt",
    "/secret.txt": FILE_MODE,
}

ROOT = "/sandbox"


def show(title, path):
    print("== {} ==".format(title))
    fake = FakeSFTP(TREE)
    try:
        resolved = fake.resolve_links(path, root=ROOT)
        print("  input    : {}".format(path))
        print("  resolved : {}".format(resolved))
    except SFTPLinkError as err:
        print("  input    : {}".format(path))
        print("  error    : [errno {}] {}".format(err.errno, err.strerror))
        print("  chain    : {}".format(" -> ".join(err.link_chain())))
        if err.target is not None:
            print("  target   : {!r}".format(err.target))
    else:
        assert resolved == "/sandbox/dir/real.txt"
    print(
        "  calls    : lstat={}, readlink={}, open={}".format(
            fake.lstat_count, fake.readlink_count, fake.opens
        )
    )
    assert fake.opens == 0, "resolution opened a file!"
    print()


def main():
    show("relative symlink", "/sandbox/rel")
    show("two-node symlink cycle", "/sandbox/loop1")
    show("symlink escaping the root", "/sandbox/escape")
    print("OK: all scenarios resolved without a single file open.")


if __name__ == "__main__":
    main()
