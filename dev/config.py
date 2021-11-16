import os
from pathlib import Path

try:
    from local_config import DATA_REPO
except ImportError:
    DATA_REPO = Path("../Safe")
    print("Could not find local_config")
