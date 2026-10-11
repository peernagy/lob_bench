import os
import sys

# the library is a flat set of modules at the repo root (import data_loading, import eval, ...)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
