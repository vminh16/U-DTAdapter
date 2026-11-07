"""UDT module execution entrypoint forwarding to run.py."""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import run

if __name__ == "__main__":
    sys.exit(run.main())
