import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
EXT2 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXT2)
for p in (EXT2, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/ext2_numba')
warnings.filterwarnings('ignore', category=DeprecationWarning)
