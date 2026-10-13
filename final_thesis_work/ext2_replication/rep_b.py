"""Independent replication of Extension 2 outcome (b) (CPPI gap risk), from PREREG.md 2, 3, 6b and
CPPI_SPEC.md 1 to 11 only."""
import json
import sys
import time

import numpy as np

T, K, TAU = 250, 50, 125
NS = [1, 2, 5, 10, 50]
MS = np.array([1.0, 2.0, 3.0, 5.0])


def streams(seeds):
    P = len(seeds)
    e = np.empty((P, T, K))
    V = np.empty((P, T))
    J = np.empty((P, T), np.int64)
    for i, s in enumerate(seeds):
        r2 = np.random.default_rng(np.random.SeedSequence([int(s), 2]))
        e[i] = r2.standard_normal((T, K))
        V[i] = r2.chisquare(3, T)
        r3 = np.random.default_rng(np.random.SeedSequence([int(s), 3]))
        J[i] = r3.integers(0, 50, size=T)
    return e, V, J


def fine_returns(e, V, J, dgp):
    y = 0.08 / K + 0.15 * e / np.sqrt(K)
    if dgp == "placebo":
        return y
    c = np.sqrt(3.0 / V[:, TAU:])  # P, 125
    if dgp == "diffusive":
        y[:, TAU:, :] = -0.10 / K + 0.30 * c[:, :, None] * e[:, TAU:, :] / np.sqrt(K)
    elif dgp == "jump":
        Z = e[:, TAU:, :].sum(axis=2) / np.sqrt(K)
        yb = np.full(e[:, TAU:, :].shape, -0.10 / K)
        P = e.shape[0]
        ii, tt = np.meshgrid(np.arange(P), np.arange(T - TAU), indexing="ij")
        yb[ii, tt, J[:, TAU:]] += 0.30 * c * Z
        y[:, TAU:, :] = yb
    return y


def simulate(y, lmin, n, peak_mode, unbounded=False):
    """Returns dict of (4, P) arrays: breach_step, breach_fine, terminal, maxdd_step, ruined."""
    P = y.shape[0]
    M = len(MS)
    m = MS[:, None]
    a = np.ones(P)
    b = np.ones(P)
    logW = np.zeros((M, P))
    Pk = np.ones((M, P))
    ruined = np.zeros((M, P), bool)
    pstar = np.ones((M, P))
    maxdd = np.zeros((M, P))
    fpeak = np.ones((M, P))
    fmaxdd = np.zeros((M, P))
    sub = K // n
    for t in range(T):
        p_hat = a / (a + b)
        fb = np.clip(2 * p_hat - 1, 0, 1)[None, :]
        for i in range(n):
            W = np.where(ruined, 0.0, np.exp(logW))
            F = 0.8 * Pk
            C = W - F
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = np.where(C <= 0, lmin, np.clip(m * C / np.where(W > 0, W, 1.0), lmin, 0.5))
            f = lam * fb
            if unbounded:
                xstep = y[:, t, :].sum(axis=1)[None, :]
                Mt = 1 + f * xstep
                newr = (Mt <= 0) & ~ruined
                ruined = ruined | newr
                logW = np.where(ruined, logW, logW + np.log(np.where(Mt > 0, Mt, 1.0)))
                Wf = np.where(ruined, 0.0, np.exp(logW))[:, :, None]
            else:
                Yc = np.cumsum(y[:, t, i * sub:(i + 1) * sub], axis=1)[None, :, :]
                Mf = 1 + f[:, :, None] * (np.exp(Yc) - 1)
                lf = logW[:, :, None] + np.log(Mf)
                logW = lf[:, :, -1]
                Wf = np.exp(lf)
            # fine-grid peak / drawdown (every substep)
            run = np.maximum.accumulate(np.concatenate([fpeak[:, :, None], Wf], axis=2), axis=2)[:, :, 1:]
            fmaxdd = np.maximum(fmaxdd, ((run - Wf) / run).max(axis=2))
            fpeak = run[:, :, -1]
            if peak_mode == "peak_every" and i < n - 1:
                Pk = np.maximum(Pk, np.where(ruined, 0.0, np.exp(logW)))
        W = np.where(ruined, 0.0, np.exp(logW))
        Pk = np.maximum(Pk, W)
        pstar = np.maximum(pstar, W)
        maxdd = np.maximum(maxdd, (pstar - W) / pstar)
        xs = y[:, t, :].sum(axis=1)
        pos = xs > 0
        a = a + pos
        b = b + ~pos
    W = np.where(ruined, 0.0, np.exp(logW))
    return {"breach_step": maxdd > 0.20, "breach_fine": fmaxdd > 0.20, "terminal": W, "maxdd_step": maxdd,
            "ruined": ruined}


def logit_fit(y, N, X, ridge=1e-8, it=100):
    beta = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (y - N * p) - ridge * beta
        H = (X * (N * p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            break
    return beta


def regression(B):
    """B: (P, 20) bool breach matrix, cells ordered (n, m) as in CELLS."""
    P = B.shape[0]
    X = np.array([[1.0, np.log(n), m] for n in NS for m in MS])
    N = np.full(20, float(P))
    beta = logit_fit(B.sum(axis=0).astype(float), N, X)
    rng = np.random.default_rng(20261012)
    boots = np.empty((2000, 3))
    Bf = B.astype(float)
    for r in range(2000):
        idx = rng.integers(0, P, size=P)
        w = np.bincount(idx, minlength=P).astype(float)
        boots[r] = logit_fit(w @ Bf, N, X)
    return {"b0": float(beta[0]), "b1": float(beta[1]), "b2": float(beta[2]),
            "ci_b1": [float(np.percentile(boots[:, 1], 2.5)), float(np.percentile(boots[:, 1], 97.5))],
            "ci_b2": [float(np.percentile(boots[:, 2], 2.5)), float(np.percentile(boots[:, 2], 97.5))],
            "p_b1_lt0": float((1 + np.sum(boots[:, 1] >= 0)) / 2001),
            "p_b2_gt0": float((1 + np.sum(boots[:, 2] <= 0)) / 2001)}


def main():
    t0 = time.time()
    seeds = np.arange(100000, 102000)
    e, V, J = streams(seeds)
    out = {"rates": {}, "regression": {}, "unbounded_n1": {}}
    variants = {"V1_code": 0.01, "V2_pure": 0.0}
    for dgp in ("diffusive", "jump", "placebo"):
        y = fine_returns(e, V, J, dgp)
        assert np.allclose(y.sum(axis=2)[:, TAU:], fine_returns(e, V, J, "diffusive").sum(axis=2)[:, TAU:]) or dgp == "placebo"
        for vn, lmin in variants.items():
            for pm in ("peak_step", "peak_every"):
                cols = []
                key = f"{dgp}|{vn}|{pm}"
                out["rates"][key] = {}
                for n in NS:
                    if pm == "peak_every" and n == 1:
                        r = cache[(dgp, vn, n)]
                    else:
                        r = simulate(y, lmin, n, pm)
                    if pm == "peak_step" and n == 1:
                        cache[(dgp, vn, n)] = r
                    for j, m in enumerate(MS):
                        cols.append(r["breach_step"][j])
                        out["rates"][key][f"n={n},m={int(m)}"] = {
                            "breach_step": float(r["breach_step"][j].mean()),
                            "breach_fine": float(r["breach_fine"][j].mean()),
                            "median_terminal": float(np.median(r["terminal"][j])),
                            "mean_maxdd_step": float(r["maxdd_step"][j].mean())}
                B = np.stack(cols, axis=1)
                if dgp != "placebo":
                    out["regression"][key] = regression(B)
                print(key, time.time() - t0, flush=True)
        if dgp == "diffusive":
            r = simulate(y, 0.01, 1, "peak_step", unbounded=True)
            for j, m in enumerate(MS):
                out["unbounded_n1"][f"V1_code,m={int(m)}"] = {"breach_step": float(r["breach_step"][j].mean()),
                                                             "ruin": float(r["ruined"][j].mean()),
                                                             "median_terminal": float(np.median(r["terminal"][j]))}
    out["runtime_s"] = time.time() - t0
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "b_out.json", "w"), indent=1)
    print("done", time.time() - t0)


cache = {}
if __name__ == "__main__":
    main()
