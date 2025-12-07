"""
Runner script for data cleaning:
- Text cleaning (lowercase, remove punctuation except apostrophes, drop digits)
- Audio resampling to 16 kHz mono
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src import preprocess
from src.utils import set_all_seeds


if __name__ == "__main__":
    set_all_seeds(42)
    preprocess.main()
