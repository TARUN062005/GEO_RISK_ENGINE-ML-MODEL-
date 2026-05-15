# conftest.py
# -----------
# Adds the project root to sys.path so pytest can resolve
# local packages (core, ml, storage, ingestion, etc.)
# without needing pip install -e .

import sys
import pathlib

# Insert project root at position 0 so it takes priority
sys.path.insert(0, str(pathlib.Path(__file__).parent))
