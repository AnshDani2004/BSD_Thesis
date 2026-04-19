import React, { useState, useEffect } from 'react';

const SCENARIOS = [
  {
    key:   'Stationary',
    label: 'Stationary',
    desc:  'Evaluates pure MAB exploration-exploitation trade-offs where arm distributions remain fixed globally. Validates asymptotic Bayesian and Upper Confidence Bound properties against the Kelly Oracle.',
  },
  {
    key:   'Adversarial',
    label: 'Adversarial',
    desc:  'Subject to a catastrophic deterministic variance shock at T=125 (Bull Market crashes into a Student-t₃ fat-tailed Bear Market). Validates ruin-reduction properties of Vol-HMM and CPPI floor.',
  },
  {
    key:   'Drift',
    label: 'Drift',
    desc:  'Undergoes a slow, continuous probabilistic regime shift (Gaussian random walk on μ). Tests an algorithm\'s capability to untangle stickiness and unlearn its prior distribution in real time.',
  },
];

function Experiments() {
  const [activeExp, setActiveExp] = useState('Stationary');
  const [statsData, setStatsData] = useState(null);

  useEffect(() => {
    setStatsData(null);
    fetch(`/results/tables/${activeExp}_stats.json`)
      .then(res => res.json())
      .then(data => setStatsData(data))
      .catch(err => console.error('Could not load stats data', err));
  }, [activeExp]);

  const scenario = SCENARIOS.find(s => s.key === activeExp);

  const ruinClass = (ruin) => {
    if (ruin === 0)   return 'ruin-zero';
    if (ruin < 0.05)  return 'ruin-low';
    return 'ruin-high';
  };

  return (
    <div style={{ display: 'flex', height: '100%', width: '100%' }}>

      {/* ── Sidebar ── */}
      <aside className="sidebar" style={{ width: '220px' }}>
        <div className="sidebar-section-title">Experiment Logs</div>
        {SCENARIOS.map(s => (
          <button
            key={s.key}
            className={`exp-sidebar-btn ${activeExp === s.key ? 'active' : ''}`}
            onClick={() => setActiveExp(s.key)}
          >
            <span className="btn-dot" />
            {s.label} Scenario
          </button>
        ))}

        <div style={{ marginTop: 'auto', paddingTop: '16px', borderTop: '1px solid var(--border)' }}>
          <div style={{ fontSize: '10px', color: 'var(--text-dim)', lineHeight: 1.6 }}>
            N = 100 simulations<br />
            T = 250 steps<br />
            CRN enforced
          </div>
        </div>
      </aside>

      {/* ── Main content ── */}
      <section
        className="content-pane"
        style={{ padding: '24px', overflowY: 'auto' }}
      >
        {/* Page header */}
        <div style={{ marginBottom: '20px' }}>
          <h2 style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '18px',
            fontWeight: 700,
            color: 'var(--asu-maroon)',
            marginBottom: '8px',
          }}>
            Offline Experiment Batch: {activeExp.toUpperCase()} MAB
          </h2>
          <hr className="gold-rule" style={{ marginBottom: '12px' }} />
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.7, maxWidth: '800px' }}>
            {scenario?.desc}
          </p>
        </div>

        {/* Chart grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '20px',
        }}>

          {/* Fan chart */}
          <div className="chart-panel">
            <h3>Monte Carlo Asymptotic Trajectory Fan</h3>
            <img
              src={`/results/figures/${activeExp}_wealth_fan.png`}
              alt={`${activeExp} Fan Chart`}
            />
            <p className="chart-caption">
              Solid line: median expected wealth. Shaded bands: 10th–90th percentile
              across all Monte Carlo paths.
            </p>
          </div>

          {/* CDF chart */}
          <div className="chart-panel">
            <h3>Empirical CDF of Final Wealth</h3>
            <img
              src={`/results/figures/${activeExp}_cdf.png`}
              alt={`${activeExp} CDF`}
            />
            <p className="chart-caption">
              Non-parametric ECDF computing P(W_T ≤ w). A right-shifted CDF indicates
              stochastically greater wealth. Left mass at zero denotes ruin.
            </p>
          </div>

          {/* Regret dynamics — full width */}
          <div className="chart-panel" style={{ gridColumn: '1 / -1' }}>
            <h3>Cumulative Regret Dynamics (Oracle Deviation)</h3>
            <img
              src={`/results/figures/${activeExp}_regret_dynamics.png`}
              alt={`${activeExp} Regret`}
              style={{ maxHeight: '400px', objectFit: 'contain' }}
            />
            <p className="chart-caption">
              Cumulative deviation of algorithmic log-wealth relative to the omniscient
              Kelly Oracle. Lower paths indicate superior structural convergence toward
              theoretical maximal growth. Oracle bets f* = clip(μ/σ², 0, 1).
            </p>
          </div>

          {/* Stats table — full width */}
          <div className="chart-panel" style={{ gridColumn: '1 / -1' }}>
            <h3>Terminal Distribution Statistics</h3>
            {statsData ? (
              <table className="stats-table">
                <thead>
                  <tr>
                    <th>Agent Algorithm</th>
                    <th>Median Terminal Wealth</th>
                    <th>Max Observed Peak</th>
                    <th className="col-ruin">Ruin Probability</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(statsData.agents).map(([agentName, stats]) => (
                    <tr key={agentName}>
                      <td className="agent-name">{agentName}</td>
                      <td>{stats.median_wealth.toFixed(2)}×</td>
                      <td>{stats.max_wealth.toFixed(2)}×</td>
                      <td className={ruinClass(stats.ruin_risk)}>
                        {(stats.ruin_risk * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="empty-state" style={{ height: '80px' }}>
                <span style={{ color: 'var(--text-dim)', fontSize: '12px' }}>
                  Loading statistics…
                </span>
              </div>
            )}
          </div>

        </div>
      </section>
    </div>
  );
}

export default Experiments;
