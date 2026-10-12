import numpy as np
from ext2lib import dgp

TEST_SEEDS = np.arange(900000, 900200)


def test_bounded_simple_returns_above_minus_one():
    Z, V = dgp.step_draws(TEST_SEEDS)
    for shock in dgp.SHOCKS.values():
        x = dgp.shock_log_returns(Z, V, shock, 125)
        r = dgp.to_simple(x, 'bounded')
        assert (r > -1).all()
    # the unbounded comparison model does produce returns below -100% on these paths
    x = dgp.shock_log_returns(Z, V, dgp.SHOCKS['large'], 125)
    assert (dgp.to_simple(x, 'unbounded') <= -1).any()


def test_fine_increments_sum_to_step_distribution():
    seeds = TEST_SEEDS[:50]
    for within in ('diffusive', 'jump'):
        y = dgp.fine_log_increments(seeds, within=within)
        g = dgp.path_rng(seeds[0], 2)
        e = g.standard_normal((250, 50))
        V = g.chisquare(3.0, 250)
        Zt = e.sum(axis=1) / np.sqrt(50)
        x = np.where(np.arange(250) < 125, 0.08 + 0.15 * Zt, -0.10 + 0.30 * Zt * np.sqrt(3.0 / V))
        np.testing.assert_allclose(y[0].sum(axis=1), x, rtol=0, atol=1e-12)
    yd = dgp.fine_log_increments(seeds, within='diffusive')
    yj = dgp.fine_log_increments(seeds, within='jump')
    np.testing.assert_allclose(yd.sum(axis=2), yj.sum(axis=2), atol=1e-12)
    np.testing.assert_array_equal(yd[:, :125], yj[:, :125])


def test_fine_step_distribution_is_t3_in_bear():
    # standardised bear step returns should match Student-t(3) quantiles (large sample)
    from scipy import stats
    y = dgp.fine_log_increments(np.arange(900000, 900400), within='diffusive')
    xs = (y[:, 125:].sum(axis=2).ravel() + 0.10) / 0.30
    q = np.quantile(xs, [0.05, 0.25, 0.5, 0.75, 0.95])
    np.testing.assert_allclose(q, stats.t(3).ppf([0.05, 0.25, 0.5, 0.75, 0.95]), atol=0.06)


def test_streams_are_deterministic_and_distinct():
    a = dgp.path_rng(5, 1).standard_normal(10)
    b = dgp.path_rng(5, 1).standard_normal(10)
    c = dgp.path_rng(5, 2).standard_normal(10)
    d = dgp.path_rng(6, 1).standard_normal(10)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c) and not np.array_equal(a, d)


def test_seed_ranges_disjoint():
    assert not set(dgp.EXPLORATORY_SEEDS) & set(dgp.CONFIRMATORY_SEEDS)
