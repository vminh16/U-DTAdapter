#!/usr/bin/env python3
"""Backward-compatible shim for the packaged ``udt`` CLI."""

import sys

from udt.cli import main

if __name__ == "__main__":
    sys.exit(main())
