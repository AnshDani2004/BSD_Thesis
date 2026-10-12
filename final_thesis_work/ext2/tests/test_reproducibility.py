import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXT2 = os.path.dirname(HERE)

SCRIPT = r'''
import sys, hashlib, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, %r); sys.path.insert(0, %r)
from ext2lib import dgp, detectors as D, cppi as CP, thesis_agents as TA
h = hashlib.sha256()
seeds = np.arange(0, 60)
Z, V = dgp.step_draws(seeds)
X = dgp.shock_log_returns(Z, V, dgp.THESIS_BEAR, 125)
for k, v in sorted(D.all_scores(X, 0.95).items()):
    h.update(v.tobytes())
Y = dgp.fine_log_increments(seeds[:20], within="jump")
for f, v in sorted(CP.cppi_paths(Y, n_reb=5).items()):
    h.update(np.ascontiguousarray(v).tobytes())
for name in ("Thompson Sampling", "Risk-Constrained CPPI"):
    r = TA.run_agent((name, "Drift", "bounded", seeds[:10]))
    h.update(r["terminal"].tobytes())
print(h.hexdigest())
''' % (EXT2, os.path.dirname(EXT2))


def run_once():
    env = dict(os.environ, NUMBA_CACHE_DIR='/tmp/ext2_numba', PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run([sys.executable, '-c', SCRIPT], capture_output=True, text=True, env=env,
                          check=True).stdout.strip().splitlines()[-1]


def test_two_executions_bit_identical():
    a = run_once()
    b = run_once()
    assert len(a) == 64 and a == b
