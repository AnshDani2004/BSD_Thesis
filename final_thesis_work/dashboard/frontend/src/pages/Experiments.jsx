import React, { useState, useEffect } from 'react';

function Experiments() {
  const [activeExp, setActiveExp] = useState('Stationary');
  const [statsData, setStatsData] = useState(null);
  
  const scenarios = ['Stationary', 'Adversarial', 'Drift'];

  useEffect(() => {
    fetch(`/results/tables/${activeExp}_stats.json`)
      .then(res => res.json())
      .then(data => setStatsData(data))
      .catch(err => console.error("Could not load stats data", err));
  }, [activeExp]);

  return (
    <div style={{ display: 'flex', height: '100%', width: '100%' }}>
      <aside className="sidebar" style={{ width: '250px' }}>
        <h3 style={{ marginBottom: '20px' }}>EXPERIMENT LOGS</h3>
        {scenarios.map(s => (
          <button 
            key={s}
            className={`term-btn-exec ${activeExp === s ? 'active' : ''}`}
            onClick={() => setActiveExp(s)}
            style={{ 
              marginBottom: '10px', 
              background: activeExp === s ? 'var(--term-cyan)' : 'transparent',
              color: activeExp === s ? '#000' : 'var(--term-amber)',
              border: `1px solid ${activeExp === s ? 'transparent' : 'var(--term-dim)'}`
            }}
          >
            {s.toUpperCase()} SCENARIO
          </button>
        ))}
      </aside>

      <section className="content-pane" style={{ padding: '30px', overflowY: 'auto' }}>
        <h2 style={{ fontSize: '24px', borderBottom: '2px solid var(--term-border)', paddingBottom: '10px', marginBottom: '30px' }}>
          OFFLINE EXPERIMENT BATCH: {activeExp.toUpperCase()} MAB
        </h2>
        
        <div style={{ marginBottom: '30px' }}>
          <p style={{ color: '#CCC', fontSize: '15px' }}>
            {activeExp === 'Stationary' && "Evaluates pure MAB exploration-exploitation trade-offs where arm distributions remain fixed globally. Validates pure asymptotic Bayesian and Upper Confidence Bound properties."}
            {activeExp === 'Adversarial' && "Subject to a catastrophic deterministic variance shock at T=125 (e.g., Bull Market crashes directly into a Student-T Fat-Tailed Bear Market). Validates EXP3 robust adversarial bounds."}
            {activeExp === 'Drift' && "Undergoes a slow, continuous probabilistic regime shift (Gaussian random walk on the expected return $\mu$). Tests an algorithm's capability to untangle stickiness and unlearn its prior distribution."}
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
          {/* Fan Chart */}
          <div style={{ background: '#050505', border: '1px solid var(--term-border)', padding: '15px' }}>
            <h3 style={{ marginBottom: '15px' }}>MONTE CARLO ASYMPTOTIC TRAJECTORY FAN</h3>
            <img 
              src={`/results/figures/${activeExp}_wealth_fan.png`} 
              alt={`${activeExp} Fan Chart`} 
              style={{ width: '100%', borderRadius: '4px' }} 
            />
            <p style={{ marginTop: '10px', fontSize: '11px', color: 'var(--term-dim)' }}>
              Solid line plots Median Expected Wealth. Shaded bands capture 10th-90th empirical percentiles globally across all Monte Carlo paths.
            </p>
          </div>

          {/* CDF Chart */}
          <div style={{ background: '#050505', border: '1px solid var(--term-border)', padding: '15px' }}>
            <h3 style={{ marginBottom: '15px' }}>EMPIRICAL CDF (PROBABILITY OF RUIN / DECAY)</h3>
            <img 
              src={`/results/figures/${activeExp}_cdf.png`} 
              alt={`${activeExp} CDF`} 
              style={{ width: '100%', borderRadius: '4px' }} 
            />
            <p style={{ marginTop: '10px', fontSize: '11px', color: 'var(--term-dim)' }}>
              Non-parametric Empirical Cumulative Distribution computing $\Pr(W_T \leq w)$. Intersection at extreme left denotes absolute ruin absorbing barriers.
            </p>
          </div>

          {/* Regret Dynamics Chart */}
          <div style={{ background: '#050505', border: '1px solid var(--term-border)', padding: '15px', gridColumn: '1 / -1' }}>
            <h3 style={{ marginBottom: '15px' }}>CUMULATIVE REGRET DYNAMICS (ORACLE DEVIATION)</h3>
            <img 
              src={`/results/figures/${activeExp}_regret_dynamics.png`} 
              alt={`${activeExp} Regret`} 
              style={{ width: '100%', borderRadius: '4px', maxHeight: '450px', objectFit: 'contain' }} 
            />
            <p style={{ marginTop: '10px', fontSize: '11px', color: 'var(--term-dim)' }}>
              Cumulative deviation of algorithmic log-wealth relative to an omniscient Kelly Oracle. Lower paths indicate superior structural convergence towards theoretical maximal growth.
            </p>
          </div>

          {/* Statistical Output Table */}
          <div style={{ background: '#050505', border: '1px solid var(--term-border)', padding: '15px', gridColumn: '1 / -1' }}>
            <h3 style={{ marginBottom: '15px', color: 'var(--term-amber)' }}>TERMINAL DISTRIBUTION STATISTICS (DATA ARRAYS)</h3>
            {statsData ? (
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--term-border)' }}>
                    <th style={{ padding: '8px', color: 'var(--term-dim)' }}>AGENT ALGORITHM</th>
                    <th style={{ padding: '8px', color: 'var(--term-dim)' }}>MEDIAN TERMINAL WEALTH</th>
                    <th style={{ padding: '8px', color: 'var(--term-dim)' }}>MAX THEORETICAL PEAK</th>
                    <th style={{ padding: '8px', color: 'var(--term-red)' }}>PROBABILITY OF RUIN</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(statsData.agents).map(([agentName, stats]) => (
                    <tr key={agentName} style={{ borderBottom: '1px dotted #333' }}>
                      <td style={{ padding: '8px', color: 'var(--term-cyan)' }}>{agentName}</td>
                      <td style={{ padding: '8px' }}>{stats.median_wealth.toFixed(2)}x</td>
                      <td style={{ padding: '8px' }}>{stats.max_wealth.toFixed(2)}x</td>
                      <td style={{ padding: '8px', color: stats.ruin_risk > 0 ? 'var(--term-red)' : 'var(--term-green)' }}>
                        {(stats.ruin_risk * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p style={{ color: 'var(--term-dim)' }}>Loading statistics array...</p>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

export default Experiments;
