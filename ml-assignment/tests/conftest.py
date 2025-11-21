# tests/conftest.py
import os
import sys

# Path to the ml-assignment directory (one level up from tests/)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

# Add ml-assignment to sys.path so `import src...` works
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
