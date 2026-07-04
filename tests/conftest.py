import pathlib
import sys

# Make the repository root importable so `from src.x import y` works
# regardless of the directory pytest is invoked from.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
