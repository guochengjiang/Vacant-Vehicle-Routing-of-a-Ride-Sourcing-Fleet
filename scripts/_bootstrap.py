"""Allow direct PyCharm execution without working-directory assumptions."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
