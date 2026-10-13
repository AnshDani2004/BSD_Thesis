"""Independent replication of Extension 2 outcome (c) for the one thesis agent that the documents specify
(the CPPI agent, CPPI_SPEC.md sections 1 to 8). The other five thesis agents are not specified in
PREREG.md or CPPI_SPEC.md, so they are not replicated."""
import json
import sys

import numpy as np

from rep_a import stream1, MU_B, S_B, MU_R, S_R, T, TAU


def scen(Z, V, name):
    if name == "Stationary":
        return MU_B + S_B * Z
    if name == "Adversarial":
        x = MU_B + S_B * Z
        x[:, TAU:] = MU_R + S_R * Z[:, TAU:] * np.sqrt(3.0 / V[:, TAU:])
        return x
    if name == "Drift":
        return np.linspace(0.08, -0.05, T)[None, :] + 0.15 * Z


def cppi(r):
    P = r.shape[0]
    a = np.ones(P); b = np.ones(P)
    logW = np.zeros(P); Pk = np.ones(P); ruined = np.zeros(P, bool)
    Wpath = np.ones((P, T + 1))
    for t in range(T):
        W = np.where(ruined, 0.0, np.exp(logW))
        ph = a / (a + b)
        fb = np.clip(ph - (1 - ph), 0, 1)
        C = W - 0.8 * Pk
        lam = np.where(C <= 0, 0.01, np.clip(3.0 * C / np.where(W > 0, W, 1.0), 0.01, 0.5))
        f = lam * fb
        M = 1 + f * r[:, t]
        newr = (M <= 0) & ~ruined
        ruined |= newr
        logW = np.where(ruined, logW, logW + np.log(np.where(M > 0, M, 1.0)))
        Wn = np.where(ruined, 0.0, np.exp(logW))
        pos = r[:, t] > 0
        a += pos; b += ~pos
        Pk = np.maximum(Pk, Wn)
        Wpath[:, t + 1] = Wn
    peak = np.maximum.accumulate(Wpath, axis=1)
    dd = ((peak - Wpath) / peak).max(axis=1)
    return ruined, Wpath, dd


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def main():
    Z, V = stream1(np.arange(100000, 102000))
    out = {}
    rng = np.random.default_rng(20261012)
    for sc in ("Stationary", "Adversarial", "Drift"):
        x = scen(Z, V, sc)
        for model in ("bounded", "unbounded"):
            r = np.exp(x) - 1 if model == "bounded" else x
            ruined, Wp, dd = cppi(r)
            term = Wp[:, -1]
            idx = rng.integers(0, term.size, size=(2000, term.size))
            meds = np.median(term[idx], axis=1)
            out[f"{sc}|{model}|CPPI"] = {
                "thesis_ruin": float(ruined.mean()), "thesis_ruin_wilson": wilson(ruined.sum(), ruined.size),
                "practical_ruin": float((Wp <= 0.10).any(axis=1).mean()),
                "median_terminal": float(np.median(term)),
                "median_terminal_ci": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))],
                "floor_breach": float((dd > 0.20).mean()), "floor_breach_wilson": wilson((dd > 0.2).sum(), dd.size)}
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "c_out.json", "w"), indent=1)
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
