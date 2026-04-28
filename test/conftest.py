"""This file is used to set up the testing environment for pytest. 
It can be used to define fixtures, mock objects, or any other setup code that is needed for the tests."""

import sys
from pathlib import Path

#Adds both the root and src directories to the system path, so that the test files can import modules from those directories without needing to modify the import statements. 
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

