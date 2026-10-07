#!/usr/bin/env python
"""
Tiny, self-contained demonstration of root-constrained SFTP symbolic-link
resolution (paramiko 5.0).

Run with:

    python demos/sftp_controlled_symlinks.py

The "remote SFTP server" is an in-memory fake tree (a dict) that answers
only lstat()/readlink() and has *no* open() method, so the resolver is
structurally incapable of opening a target file. No Docker, sockets, temp
files or local path resolution (os.path.realpath, ...) are used.
"""

import errno
import os
import stat
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paramiko.sftp import SFTPLinkLoopError, SFTPRootEscapeError
from paramiko.sftp_client import _resolve_link_chain
from paramiko.sftp_attr import SFTPAttributes

REG = stat.S_IFREG | 0o644
DIR = stat.S_IFDIR | 0o755
LNK = stat.S_IFLNK | 0o777

ROOT = "/srv"

# Fake remote tree: relative link, nested relative chain, two-node loop,
# self loop, relative escape and absolute escape.
TREE = {
    "/srv": DIR,
    "/srv/data": DIR,
    "/srv/data/secret.txt": REG,
    "/srv/www": DIR,
    "/srv/www/alias.txt": ("link", "../data/secret.txt"),
    "/srv/www/sub": DIR,
    "/srv/www/sub/a": ("link", "b"),
    "/srv/www/sub/b": ("link", "../../data/secret.txt"),
    "/srv/loop": DIR,
    "/srv/loop/x": ("link", "y"),
    "/srv/loop/y": ("link", "x"),
    "/srv/loop/self": ("link", "self"),
    "/srv/evil_rel.txt": ("link", "../../etc/passwd"),
    "/srv/evil_abs.txt": ("link", "/etc/passwd"),
    "/etc": DIR,
    "/etc/passwd": REG,
}


class FakeSFTP:
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
        attr = SFTPAttributes()
        attr.st_mode = LNK if isinstance(entry, tuple) else entry
        return attr

    def readlink(self, path):
        path = path.decode() if isinstance(path, bytes) else path
        self.readlink_calls.append(path)
        return self.tree[path][1]

    # NOTE: no open()/file() -- resolution must never open the target.


def show(title, query):
    print("-" * 72)
    print("resolve: {}  (root={})".format(query, ROOT))
    sftp = FakeSFTP(TREE)
    try:
        resolved, chain = _resolve_link_chain(sftp, query, ROOT, 40)
    except SFTPLinkLoopError as err:
        print("  -> SFTPLinkLoopError: {}".format(err))
    except SFTPRootEscapeError as err:
        print("  -> SFTPRootEscapeError: {}".format(err))
    else:
        print("  -> final path : {}".format(resolved))
        print("  -> link chain : {}".format(chain or "(no links)"))
    print("  remote lstat calls   : {}".format(sftp.lstat_calls))
    print("  remote readlink calls: {}".format(sftp.readlink_calls))
    print("  file OPEN calls      : 0 (fake has no open())")
    print("  [{}]".format(title))


def main():
    print("Controlled SFTP symlink resolution demo")
    show("relative link, resolved from the link directory",
         "/srv/www/alias.txt")
    show("nested relative chain", "/srv/www/sub/a")
    show("two-node loop", "/srv/loop/x")
    show("self loop", "/srv/loop/self")
    show("relative escape over the root", "/srv/evil_rel.txt")
    show("absolute escape over the root", "/srv/evil_abs.txt")
    show("plain '..' escape without any link", "/srv/../etc/passwd")


if __name__ == "__main__":
    main()
