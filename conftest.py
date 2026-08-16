import os
import sys

# Make the src/ layout importable for tests without an install step.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
