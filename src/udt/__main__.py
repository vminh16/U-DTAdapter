"""Module execution entrypoint for ``python -m udt``."""

import sys

from udt.cli import main

if __name__ == "__main__":
    sys.exit(main())
