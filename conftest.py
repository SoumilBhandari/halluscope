"""Put the repo root on sys.path so tests can `import model`, `import probe`, etc."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
