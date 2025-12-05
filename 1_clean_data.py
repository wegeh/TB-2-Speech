"""
Runner script for data cleaning:
- Text cleaning (lowercase, remove punctuation except apostrophes, drop digits)
- Audio resampling to 16 kHz mono
"""

from src import preprocess


if __name__ == "__main__":
    preprocess.main()
