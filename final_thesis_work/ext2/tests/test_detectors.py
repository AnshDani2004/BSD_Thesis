import numpy as np
import pytest
from ext2lib import dgp, detectors as D, stats as ST

EXPL_A = np.arange(0, 500)      # tuning half of the exploratory seeds
EXPL_B = np.arange(500, 1000)   # held-out half
TEST = np.arange(900000, 900300)


@pytest.fixture(scope='module')
def tuned():
    Z, V = dgp.step_draws(EXPL_A)
    X = dgp.shock_log_returns(Z, V, dgp.THESIS_BEAR, 0)      # no break: pre-break only matters
    S = D.all_scores(X, 0.95)
    return {d: {tg: ST.tune_threshold(S[d], tg) for tg in (1, 5, 10)} for d in D.DETECTORS}


def test_threshold_tuning_holds_out_of_sample(tuned):
    Z, V = dgp.step_draws(EXPL_B)
    X = dgp.shock_log_returns(Z, V, dgp.THESIS_BEAR, 0)
    S = D.all_scores(X, 0.95)
    tol = {1: (0.25, 3.0), 5: (0.5, 2.0), 10: (0.5, 2.0)}
    for d in D.DETECTORS:
        for tg, (lo, hi) in tol.items():
            rate = ST.prebreak_rate(S[d], tuned[d][tg])
            assert lo * tg <= rate <= hi * tg, (d, tg, rate)


def test_tuning_hits_target_in_sample(tuned):
    Z, V = dgp.step_draws(EXPL_A)
    X = dgp.shock_log_returns(Z, V, dgp.THESIS_BEAR, 0)
    S = D.all_scores(X, 0.95)
    for d in D.DETECTORS:
        for tg in (1, 5, 10):
            rate = ST.prebreak_rate(S[d], tuned[d][tg])
            assert 0.8 * tg <= rate <= tg + 1e-9, (d, tg, rate)


def test_huge_break_detected_within_two_steps(tuned):
    Z, V = dgp.step_draws(TEST)
    X = dgp.shock_log_returns(Z, V, dgp.THESIS_BEAR, 125)
    X[:, 125:] = -2.0 + 0.01 * Z[:, 125:]
    S = D.all_scores(X, 0.95)
    for d in D.DETECTORS:
        dl, det = ST.detection_delays(S[d], tuned[d][5], 50)
        assert (dl <= 2).mean() >= 0.99, (d, np.bincount(dl.astype(int))[:5])


def test_constant_regime_few_alarms(tuned):
    Z, V = dgp.step_draws(TEST)
    X = dgp.stationary_log_returns(Z)
    S = D.all_scores(X, 0.95)
    for d in D.DETECTORS:
        a = S[d][:, ST.BURN_IN:] > tuned[d][5]
        assert 1000 * a.mean() <= 15.0, (d, 1000 * a.mean())


def test_no_look_ahead():
    Z, V = dgp.step_draws(TEST[:40])
    X = dgp.shock_log_returns(Z, V, dgp.THESIS_BEAR, 125)
    S = D.all_scores(X, 0.97)
    for t0 in (30, 124, 200):
        X2 = X.copy()
        X2[:, t0 + 1:] = np.random.default_rng(1).normal(-1, 2, size=X2[:, t0 + 1:].shape)
        S2 = D.all_scores(X2, 0.97)
        for d in D.DETECTORS:
            assert np.array_equal(S[d][:, :t0 + 1], S2[d][:, :t0 + 1]), (d, t0)


def test_bocpd_run_length_collapses_after_break():
    g = np.random.default_rng(900001)
    x = np.concatenate([g.normal(0.0, 0.1, 100), g.normal(3.0, 0.1, 10)])
    lp = D.bocpd_runlength_posterior(x[:100], 0.05)
    assert int(np.argmax(lp)) == 100            # whole history is one segment before the break
    for k in (101, 102, 103):
        lp = D.bocpd_runlength_posterior(x[:k], 0.05)
        p = np.exp(lp)
    assert p[1:6].sum() > 0.9                    # within 3 steps, mass on run lengths <= 5


def test_jump_model_recovers_known_segments():
    g = np.random.default_rng(900002)
    Z = np.vstack([g.normal(0, 0.5, (50, 2)), g.normal(3, 0.5, (50, 2)), g.normal(0, 0.5, (40, 2))])
    truth = np.r_[np.zeros(50), np.ones(50), np.zeros(40)]
    st, cent, _ = D.jm_offline(Z, D.jump_penalty(0.95))
    acc = max((st == truth).mean(), (st != truth).mean())
    assert acc >= 0.95, acc


def test_hmms_match_thesis_class():
    from simulation.hmm_refined import VolAugmentedHMM
    Z, V = dgp.step_draws(TEST[:4])
    X = dgp.shock_log_returns(Z, V, dgp.THESIS_BEAR, 125)
    for vol, cus, name in [(False, False, 'hmm_ret'), (True, False, 'volhmm'), (True, True, 'volhmm_cusum')]:
        sc = D.hmm_scores(X, 0.95, vol, cus)
        for i in range(X.shape[0]):
            bull = {'mu': 0.08, 'sigma': 0.15, 'vol_shape': 2.0, 'vol_scale': 0.075}
            bear = {'mu': -0.10, 'sigma': 0.30, 'vol_shape': 4.0, 'vol_scale': 0.15}
            if not vol:
                bear = dict(bear, vol_shape=2.0, vol_scale=0.075)   # Gamma terms cancel: return-only HMM
            h = VolAugmentedHMM({'bull': bull, 'bear': bear}, max_persistence=0.95,
                                cusum_threshold=3.0 if cus else np.inf)
            ref = []
            for t in range(X.shape[1]):
                h.update(float(X[i, t]))
                ref.append(h.log_beliefs[1] - h.log_beliefs[0])
            np.testing.assert_allclose(sc[i], ref, rtol=0, atol=1e-9, err_msg=name)
