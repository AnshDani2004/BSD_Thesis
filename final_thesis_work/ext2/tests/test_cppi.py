import numpy as np
from ext2lib import dgp, cppi as CP

TEST = np.arange(900000, 900100)


def test_matches_thesis_class_on_step_returns():
    from simulation.risk_constrained import RiskConstrainedKelly
    from simulation.wealth_tracker import WealthTracker
    Z, V = dgp.step_draws(TEST[:20])
    R = np.expm1(dgp.shock_log_returns(Z, V, dgp.THESIS_BEAR, 125))
    out = CP.cppi_paths(np.log1p(R))
    out2 = CP.cppi_simple_returns(R)
    for i in range(R.shape[0]):
        a = RiskConstrainedKelly(1, max_drawdown=0.20, cppi_multiplier=3.0)
        tr = WealthTracker(1.0)
        peak, br = 1.0, False
        for t in range(R.shape[1]):
            f = a.act()[0]
            tr.update(f, R[i, t])
            a.update(np.array([R[i, t]]), wealth=tr.wealth)
            peak = max(peak, tr.wealth)
            br |= tr.wealth < 0.8 * peak
        assert abs(out['terminal'][i] / tr.wealth - 1) < 1e-12
        assert abs(out2['terminal'][i] / tr.wealth - 1) < 1e-12
        assert out['breach_step'][i] == br == out2['breach_step'][i]


def test_continuous_style_rebalancing_gaussian_almost_never_breaches():
    seeds = TEST[:100]
    for mu, sig in ((0.08, 0.15), (-0.10, 0.30)):
        Y = dgp.fine_gaussian_increments(seeds, 100, 1000, mu, sig)
        for lmin in (0.0, 0.01):
            if mu < 0 and lmin > 0:
                continue   # V1's 0.01 minimum keeps a position below the floor: not a CPPI guarantee
            out = CP.cppi_paths(Y, n_reb=1000, m=3.0, l_min=lmin)
            assert out['breach_step'].mean() <= 0.01, (mu, lmin, out['breach_step'].mean())


def test_m_zero_never_breaches():
    Y = dgp.fine_log_increments(TEST[:50])
    for n in (1, 5, 50):
        out = CP.cppi_paths(Y, n_reb=n, m=0.0, l_min=0.0)
        assert not out['breach_step'].any() and not out['breach_fine'].any()
        assert np.all(out['terminal'] == 1.0)


def test_pure_cppi_m_one_never_breaches_bounded():
    for within in ('diffusive', 'jump'):
        Y = dgp.fine_log_increments(TEST[:100], within=within)
        for n in (1, 2, 5, 10, 50):
            out = CP.cppi_paths(Y, n_reb=n, m=1.0, l_min=0.0)
            assert not out['breach_step'].any(), (within, n)


def test_bounded_model_no_ruin():
    Y = dgp.fine_log_increments(TEST[:100])
    for n in (1, 50):
        for m in (1.0, 5.0):
            assert not CP.cppi_paths(Y, n_reb=n, m=m)['ruined'].any()


def test_peak_modes_agree_at_one_rebalance_and_theorem_holds_for_both():
    Y = dgp.fine_log_increments(TEST[:100])
    for m in (1.0, 3.0):
        a = CP.cppi_paths(Y, n_reb=1, m=m, peak_every_rebalance=True)
        b = CP.cppi_paths(Y, n_reb=1, m=m, peak_every_rebalance=False)
        for k in ('terminal', 'breach_step', 'mdd_step'):
            assert np.array_equal(a[k], b[k])
    for n in (2, 10, 50):
        assert not CP.cppi_paths(Y, n_reb=n, m=1.0, l_min=0.0, peak_every_rebalance=False)['breach_step'].any()


def test_simple_return_version_matches_with_external_base_fraction():
    Z, V = dgp.step_draws(TEST[:50])
    X = dgp.shock_log_returns(Z, V, dgp.THESIS_BEAR, 125)
    fdet = (np.random.default_rng(3).random(X.shape) > 0.3).astype(float)
    a = CP.cppi_paths(X, fdet=fdet)
    b = CP.cppi_simple_returns(np.expm1(X), fdet=fdet)
    np.testing.assert_allclose(a['terminal'], b['terminal'], rtol=1e-12)
    assert np.array_equal(a['breach_step'], b['breach_step'])
    a0 = CP.cppi_paths(X, fdet=fdet, use_cppi=False)
    b0 = CP.cppi_simple_returns(np.expm1(X), fdet=fdet, use_cppi=False)
    np.testing.assert_allclose(a0['terminal'], b0['terminal'], rtol=1e-12)


def test_breach_channel_flag():
    Y = dgp.fine_log_increments(TEST[:100])
    out = CP.cppi_paths(Y, n_reb=1, m=3.0, l_min=0.01)
    fm = out['first_breach_lambda_at_min']
    assert np.array_equal(fm >= 0, out['breach_step'])
