"""Independent replication of Extension 2 outcome (a), written from PREREG.md section 2, 3, 5, 6a only.

Primary cell: A_ii = 0.95, thesis shock (-0.10, 0.30), D = 125, H = 50. Thresholds tuned on
exploratory seeds 0..999 (pre-break steps 10..124 pooled), applied to confirmatory seeds 100000..101999.
"""
import json
import sys
import time

import numpy as np
from scipy.special import gammaln, logsumexp

T = 250
TAU = 125
BURN = 10
MU_B, S_B = 0.08, 0.15
MU_R, S_R = -0.10, 0.30


def stream1(seeds):
    Z = np.empty((len(seeds), T))
    V = np.empty((len(seeds), T))
    for i, s in enumerate(seeds):
        rng = np.random.default_rng(np.random.SeedSequence([int(s), 1]))
        Z[i] = rng.standard_normal(T)
        V[i] = rng.chisquare(3, T)
    return Z, V


def scenario_a(Z, V, mu, s, D):
    x = MU_B + S_B * Z
    sl = slice(TAU, min(T, TAU + D))
    x[:, sl] = mu + s * Z[:, sl] * np.sqrt(3.0 / V[:, sl])
    return x


def norm_logpdf(x, m, s):
    return -0.5 * np.log(2 * np.pi) - np.log(s) - 0.5 * ((x - m) / s) ** 2


def gamma_logpdf(v, k, th):
    return (k - 1) * np.log(v) - v / th - gammaln(k) - k * np.log(th)


def vol5(x, plus=1e-6, first=0.05):
    P, n = x.shape
    v = np.empty_like(x)
    for t in range(n):
        lo = max(0, t - 4)
        if t + 1 < 2:
            v[:, t] = first
        else:
            v[:, t] = x[:, lo:t + 1].std(axis=1) + plus
    return v


def hmm_filter(x, a, vol=False, cusum=False, cusum_lag=0):
    """Two-state forward filter in log space. Returns s_t = log P(bear) - log P(bull)."""
    P, n = x.shape
    logA = np.log(np.array([[a, 1 - a], [1 - a, a]]))
    e_bull = norm_logpdf(x, MU_B, S_B)
    e_bear = norm_logpdf(x, MU_R, S_R)
    if vol:
        v = vol5(x)
        e_bull = e_bull + gamma_logpdf(v, 2.0, 0.075)
        e_bear = e_bear + gamma_logpdf(v, 4.0, 0.15)
    if cusum:
        S = np.zeros(P)
        latched = np.zeros(P, bool)
        bonus = np.zeros((P, n))
        for t in range(n):
            S = np.maximum(0.0, S - (x[:, t] - MU_B) / S_B - 0.5)
            latched = latched | (S > 3.0)
            tt = t + cusum_lag
            if tt < n:
                bonus[:, tt] = np.where(latched, 2.0, 0.0)
        e_bear = e_bear + bonus
    l = np.log(np.full((P, 2), 0.5))
    out = np.empty((P, n))
    for t in range(n):
        pred = logsumexp(l[:, :, None] + logA[None, :, :], axis=1)  # over i -> j
        l = pred + np.stack([e_bull[:, t], e_bear[:, t]], axis=1)
        l = l - logsumexp(l, axis=1, keepdims=True)
        out[:, t] = l[:, 1] - l[:, 0]
    return out


def student_logpdf(x, mu, kappa, alpha, beta):
    nu = 2 * alpha
    s2 = beta * (kappa + 1) / (alpha * kappa)
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * s2)
            - (nu + 1) / 2 * np.log1p((x - mu) ** 2 / (nu * s2)))


def bocpd(x, a):
    """Adams-MacKay BOCPD, no truncation. Returns (s_rl, s_mean)."""
    H = 1 - a
    P, n = x.shape
    mu0, k0, a0, b0 = 0.0, 1.0, 2.0, 0.04
    # index l-1 holds run length l (1..t+1)
    logR = np.full((P, n), -np.inf)
    MU = np.zeros((P, n))
    BE = np.zeros((P, n))
    s_rl = np.empty((P, n))
    s_mean = np.empty((P, n))
    for t in range(n):
        xt = x[:, t:t + 1]
        lp_prior = student_logpdf(x[:, t], mu0, k0, a0, b0)
        new = np.full((P, n), -np.inf)
        if t == 0:
            new[:, 0] = lp_prior  # H*pred + (1-H)*pred from l=0
        else:
            L = np.arange(1, t + 1)  # existing run lengths
            kap = k0 + L
            alp = a0 + L / 2
            lp = student_logpdf(xt, MU[:, :t], kap[None, :], alp[None, :], BE[:, :t])
            new[:, 0] = np.log(H) + lp_prior  # previous total mass = 1
            new[:, 1:t + 1] = logR[:, :t] + np.log(1 - H) + lp
        new = new - logsumexp(new[:, :t + 1], axis=1, keepdims=True)
        # update params: run length l+1 from l; run length 1 from prior
        nMU = np.zeros((P, n))
        nBE = np.zeros((P, n))
        nMU[:, 0] = (k0 * mu0 + x[:, t]) / (k0 + 1)
        nBE[:, 0] = b0 + k0 * (x[:, t] - mu0) ** 2 / (2 * (k0 + 1))
        if t > 0:
            L = np.arange(1, t + 1)
            kap = (k0 + L)[None, :]
            nMU[:, 1:t + 1] = (kap * MU[:, :t] + xt) / (kap + 1)
            nBE[:, 1:t + 1] = BE[:, :t] + kap * (xt - MU[:, :t]) ** 2 / (2 * (kap + 1))
        logR, MU, BE = new, nMU, nBE
        m = t + 1
        if t <= 9:
            s_rl[:, t] = 1e9
        else:
            lo = logsumexp(logR[:, :10], axis=1)
            hi = logsumexp(logR[:, 10:m], axis=1)
            s_rl[:, t] = np.where(np.isneginf(hi), 1e9, lo - hi)
        s_mean[:, t] = -np.sum(np.exp(logR[:, :m]) * MU[:, :m], axis=1)
    return s_rl, s_mean


def jm_scores(x, a, vol_reading="plain"):
    """Online statistical jump model score s_t = V_end(bull) - V_end(bear)."""
    P, n = x.shape
    lam = 2 * np.log(a / (1 - a))
    if vol_reading == "plain":
        v = vol5(x, plus=0.0, first=0.0)  # std of one obs = 0
    else:
        v = vol5(x)  # volhmm's v (1e-6, 0.05 when < 2 obs)
    out = np.zeros((P, n))
    for t in range(n):
        w = min(60, t + 1)
        if w < 3:
            continue
        F = np.stack([x[:, t - w + 1:t + 1], v[:, t - w + 1:t + 1]], axis=2)  # P,w,2
        mean = F.mean(axis=1, keepdims=True)
        sd = F.std(axis=1, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        X = (F - mean) / sd
        med = np.median(X[:, :, 0], axis=1, keepdims=True)
        st = (X[:, :, 0] > med).astype(np.int64)  # at or below median -> 0
        C = np.zeros((P, 2, 2))
        C = update_centroids(X, st, C)
        Vend = np.zeros((P, 2))
        done = np.zeros(P, bool)
        prev_st = st.copy()
        for p in range(1, 11):
            act = np.where(~done)[0]
            if act.size == 0:
                break
            Xa, Ca = X[act], C[act]
            V, st_new = dp(Xa, Ca, lam)
            if p >= 2:
                same = np.all(st_new == prev_st[act], axis=1)
            else:
                same = np.zeros(act.size, bool)
            # stopped paths: record Vend
            idx_s = act[same]
            Vend[idx_s] = V[same]
            done[idx_s] = True
            idx_c = act[~same]
            prev_st[idx_c] = st_new[~same]
            C[idx_c] = update_centroids(X[idx_c], st_new[~same], C[idx_c])
            if p == 10 and idx_c.size:
                V2, _ = dp(X[idx_c], C[idx_c], lam)
                Vend[idx_c] = V2
                done[idx_c] = True
        bear = np.where(C[:, 1, 0] < C[:, 0, 0], 1, 0)  # state 0 on ties
        bull = 1 - bear
        r = np.arange(P)
        out[:, t] = Vend[r, bull] - Vend[r, bear]
    return out


def update_centroids(X, st, C):
    C = C.copy()
    for k in (0, 1):
        m = (st == k)
        cnt = m.sum(axis=1)
        s = np.einsum('pw,pwf->pf', m.astype(float), X)
        ok = cnt > 0
        C[ok, k, :] = s[ok] / cnt[ok, None]
    return C


def dp(X, C, lam):
    P, w, _ = X.shape
    L = ((X[:, :, None, :] - C[:, None, :, :]) ** 2).sum(axis=3)  # P,w,2
    V = L[:, 0, :].copy()
    prev = np.zeros((P, w, 2), np.int64)
    for j in range(1, w):
        stay0, sw0 = V[:, 0], V[:, 1] + lam
        stay1, sw1 = V[:, 1], V[:, 0] + lam
        k0 = stay0 <= sw0
        k1 = stay1 <= sw1
        prev[:, j, 0] = np.where(k0, 0, 1)
        prev[:, j, 1] = np.where(k1, 1, 0)
        V = np.stack([L[:, j, 0] + np.minimum(stay0, sw0), L[:, j, 1] + np.minimum(stay1, sw1)], axis=1)
    st = np.empty((P, w), np.int64)
    st[:, -1] = np.where(V[:, 1] < V[:, 0], 1, 0)
    r = np.arange(P)
    for j in range(w - 1, 0, -1):
        st[:, j - 1] = prev[r, j, st[:, j]]
    return V, st


def all_scores(x, a, jm_reading="plain"):
    sc = {}
    sc["hmm_ret"] = hmm_filter(x, a)
    sc["volhmm"] = hmm_filter(x, a, vol=True)
    sc["volhmm_cusum"] = hmm_filter(x, a, vol=True, cusum=True, cusum_lag=0)
    sc["volhmm_cusum_lag1"] = hmm_filter(x, a, vol=True, cusum=True, cusum_lag=1)
    rl, mn = bocpd(x, a)
    sc["bocpd_rl"] = rl
    sc["bocpd_mean"] = mn
    sc["jm"] = jm_scores(x, a, jm_reading)
    if jm_reading == "plain":
        sc["jm_volhmm_v"] = jm_scores(x, a, "volhmm_v")
    return sc


def tune(scores, target):
    pre = scores[:, BURN:TAU].ravel()
    k = int(np.floor(target / 1000 * pre.size))
    srt = np.sort(pre)[::-1]
    return float(srt[k]), k


def signflip(d, nrep=10000, seed=20261012):
    rng = np.random.default_rng(seed)
    obs = d.mean()
    sg = rng.integers(0, 2, size=(nrep, d.size)) * 2 - 1
    null = (sg * d[None, :]).mean(axis=1)
    cnt = int(np.sum(np.abs(null) >= abs(obs) - 1e-12))
    return float(obs), (1 + cnt) / (nrep + 1), cnt / nrep


def boot_ci(d, nrep=2000, seed=20261012):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(nrep, d.size))
    m = d[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    t0 = time.time()
    a = 0.95
    expl = np.arange(0, 1000)
    conf = np.arange(100000, 102000)
    Ze, Ve = stream1(expl)
    Zc, Vc = stream1(conf)
    xe = scenario_a(Ze, Ve, MU_R, S_R, 125)
    xc = scenario_a(Zc, Vc, MU_R, S_R, 125)
    se = all_scores(xe, a)
    print("expl scores", time.time() - t0, flush=True)
    sc = all_scores(xc, a)
    print("conf scores", time.time() - t0, flush=True)
    # check pre-break independence of shock cell (small shock D=5 on exploratory paths)
    xe2 = scenario_a(Ze, Ve, 0.0, 0.20, 5)
    pre_same = bool(np.array_equal(xe2[:, :TAU], xe[:, :TAU]))
    res = {"A_ii": a, "cell": "thesis shock (-0.10,0.30), D=125, H=50", "prebreak_x_identical_across_cells": pre_same,
           "thresholds": {}, "frontier": {}}
    Hh = 50
    for det in se:
        res["thresholds"][det] = {}
        res["frontier"][det] = {}
        for target in (1, 5, 10, "native0"):
            if target == "native0":
                h, k = 0.0, None
            else:
                h, k = tune(se[det], target)
            s = sc[det]
            alarm = s > h
            alarm[:, :BURN] = False
            fa = alarm[:, BURN:TAU].mean() * 1000
            prev = np.concatenate([np.zeros((s.shape[0], 1), bool), alarm[:, :-1]], axis=1)
            onset = (alarm & ~prev)[:, BURN:TAU].mean() * 1000
            fa_expl = ((se[det] > h)[:, BURN:TAU]).mean() * 1000
            win = alarm[:, TAU:TAU + Hh]
            hit = win.any(axis=1)
            first = np.where(hit, win.argmax(axis=1), Hh).astype(float)
            res["thresholds"][det][str(target)] = {"h": h, "k": k, "fa_per_1000_exploratory": float(fa_expl)}
            res["frontier"][det][str(target)] = {
                "mean_delay": float(first.mean()), "median_delay": float(np.median(first)),
                "p90_delay": float(np.percentile(first, 90)), "share_never": float(1 - hit.mean()),
                "fa_per_1000_confirmatory": float(fa), "onsets_per_1000_confirmatory": float(onset)}
            if target == 5:
                res["frontier"][det]["_delays5"] = first.tolist()
    # contrasts at target 5
    D5 = {d: np.array(res["frontier"][d].pop("_delays5")) for d in se}
    for d in se:
        for tg in ("1", "10", "native0"):
            pass
    contr = {}
    pairs = {"A1": ("volhmm", "hmm_ret"), "A2": ("volhmm_cusum", "volhmm"), "A3": ("bocpd_rl", "volhmm_cusum"),
             "A4": ("jm", "volhmm_cusum"), "A5": ("bocpd_mean", "volhmm_cusum"),
             "A2_lag1": ("volhmm_cusum_lag1", "volhmm"), "A3_lag1": ("bocpd_rl", "volhmm_cusum_lag1"),
             "A4_lag1": ("jm", "volhmm_cusum_lag1"), "A4_jm_volhmm_v": ("jm_volhmm_v", "volhmm_cusum")}
    for name, (p, q) in pairs.items():
        d = D5[p] - D5[q]
        obs, pv, pv_raw = signflip(d)
        contr[name] = {"contrast": f"{p} - {q}", "mean_diff": obs, "p_signflip_plus1": pv, "p_signflip_raw": pv_raw,
                       "ci95_boot": boot_ci(d), "n_paths_nonzero_diff": int(np.sum(d != 0))}
    res["contrasts_target5"] = contr
    res["runtime_s"] = time.time() - t0
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "a_out.json", "w"), indent=1)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
