"""
Pytest configuration for the standalone prefetch tests.

These tests exercise ``paramiko.sftp_prefetch.PrefetchReader`` entirely
through a fake SFTP packet interface (see ``test_prefetch.py``); they do
not need the loopback SSH server or the heavy ``tests/conftest.py``
fixtures, so they live in their own directory.
"""
