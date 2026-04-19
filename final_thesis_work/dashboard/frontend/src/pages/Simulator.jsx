import React, { useEffect, useRef } from 'react';
import { createChart } from 'lightweight-charts';
import { useSimulation } from '../context/SimulationContext';

/* Agent colour map — matches presentation palette */
const AGENT_COLORS = {
  'Vol-Augmented HMM (Proposed)': '#22783A',  /* forestgreen */
  'Risk-Constrained CPPI':         '#7B2D8B',  /* purple */
  'Thompson Sampling':             '#C17F24',  /* amber */
  'EXP3':                          '#1A6FA8',  /* info blue */
  'Naive Bayes':                   '#B93232',  /* softred */
  'UCB':                           '#5A6470',  /* slate */
  'Softmax':                       '#D97706',  /* orange */
};

const HISTORICAL_KEYS = [
  'real_2000_dotcom', 'real_2008_crisis',
  'real_2020_covid',  'real_2013_bull',
];

function Simulator() {
  const {
    tSteps, setTSteps,
    nSims, setNSims,
    scenario, setScenario,
    friction, setFriction,
    cppiDrawdown, setCppiDrawdown,
    cppiMultiplier, setCppiMultiplier,
    hmmBullMu, setHmmBullMu,
    hmmBearMu, setHmmBearMu,
    results: data, setResults: setData,
    loading, setLoading,
    error, setError,
    activeTab, setActiveTab,
    showAdvanced, setShowAdvanced,
  } = useSimulation();

  const isHistorical = HISTORICAL_KEYS.includes(scenario);
  const chartContainerRef = useRef();
  const chartRef          = useRef(null);
  const seriesMap         = useRef({});

  /* ── Execute simulation ──────────────────────────────────── */
  const handleSimulate = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/simulate/compare', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          t_steps:  tSteps,
          n_sims:   nSims,
          scenario: scenario,
          friction: friction,
          advanced_params: {
            cppi_drawdown:   cppiDrawdown,
            cppi_multiplier: cppiMultiplier,
            hmm_bull_mu:     hmmBullMu,
            hmm_bear_mu:     hmmBearMu,
          },
        }),
      });
      const resData = await res.json();
      if (resData.status === 'success') {
        setData(resData);
        updateChart(resData.strategies);
        if (activeTab !== 'CHART' && activeTab !== 'STATS') setActiveTab('CHART');
      } else {
        setError('API Error: Simulation failed.');
      }
    } catch (e) {
      setError('Connection refused. Is the Python backend running on port 5001?');
    }
    setLoading(false);
  };

  /* ── Chart update ────────────────────────────────────────── */
  const updateChart = (strategies) => {
    if (!chartRef.current) return;
    Object.values(seriesMap.current).forEach(s => chartRef.current.removeSeries(s));
    seriesMap.current = {};

    Object.entries(strategies).forEach(([name, strat]) => {
      const series = chartRef.current.addLineSeries({
        color:     AGENT_COLORS[name] || '#8C1D40',
        lineWidth: 2,
        title:     name,
      });
      series.setData(strat.chart_data);
      seriesMap.current[name] = series;
    });
    chartRef.current.timeScale().fitContent();
  };

  /* ── Chart init ──────────────────────────────────────────── */
  useEffect(() => {
    if (chartContainerRef.current && !chartRef.current) {
      chartRef.current = createChart(chartContainerRef.current, {
        layout: {
          background:  { type: 'solid', color: '#FFFFFF' },
          textColor:   '#5A6470',
          fontFamily:  "'JetBrains Mono', 'Fira Code', monospace",
        },
        grid: {
          vertLines: { color: '#E8EAF0' },
          horzLines: { color: '#E8EAF0' },
        },
        crosshair:       { mode: 0 },
        rightPriceScale: { borderColor: '#D8DAE0' },
        timeScale:       { borderColor: '#D8DAE0' },
      });

      const handleResize = () => {
        if (!chartContainerRef.current || !chartRef.current) return;
        chartRef.current.applyOptions({
          width:  chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      };
      window.addEventListener('resize', handleResize);

      if (data?.strategies) updateChart(data.strategies);
      return () => window.removeEventListener('resize', handleResize);
    }
  }, []);

  /* ── Render ──────────────────────────────────────────────── */
  return (
    <div style={{ display: 'flex', height: '100%', width: '100%' }}>

      {/* ── Sidebar ── */}
      <aside className="sidebar">

        <div className="sidebar-section-title">Configuration</div>

        {/* Scenario */}
        <div className="control-block">
          <label>Scenario Type</label>
          <select className="term-input" value={scenario} onChange={e => setScenario(e.target.value)}>
            <optgroup label="── Synthetic Environments ──">
              <option value="stationary">Stationary (Flat MAB)</option>
              <option value="adversarial">Adversarial Crash</option>
              <option value="drift">Slow Gaussian Drift</option>
            </optgroup>
            <optgroup label="── Historical S&P 500 ──">
              <option value="real_2000_dotcom">2000 Dot-Com Collapse</option>
              <option value="real_2008_crisis">2008 Financial Crisis</option>
              <option value="real_2020_covid">2020 COVID Crash</option>
              <option value="real_2013_bull">2013–2019 Bull Market</option>
            </optgroup>
          </select>
          {isHistorical && (
            <div className="scenario-badge">
              ▶ Empirical mode — single historical path. Monte Carlo ignored.
            </div>
          )}
        </div>

        {/* Horizon */}
        <div className="control-block">
          <label>Horizon [T steps]</label>
          <input
            type="number"
            className="term-input"
            value={tSteps}
            onChange={e => setTSteps(e.target.value)}
          />
        </div>

        {/* Monte Carlo */}
        <div className="control-block">
          <label>Monte Carlo [N sims]</label>
          <input
            type="number"
            className="term-input"
            value={nSims}
            onChange={e => setNSims(e.target.value)}
          />
        </div>

        {/* Friction */}
        <div className="control-block">
          <label>Transaction Costs [bps]</label>
          <input
            type="number"
            step="0.1"
            className="term-input"
            value={friction}
            onChange={e => setFriction(e.target.value)}
          />
          <div style={{ fontSize: '10px', color: 'var(--text-dim)', marginTop: '2px' }}>
            1 bps = 0.01% fee on trade volume
          </div>
        </div>

        {/* Advanced accordion */}
        <div>
          <button className="accordion-header" onClick={() => setShowAdvanced(!showAdvanced)}>
            <span>{showAdvanced ? '▼' : '▶'} Advanced Calibration</span>
          </button>
          {showAdvanced && (
            <div className="accordion-body">
              <div className="control-block">
                <label className="label-cppi">CPPI Max Drawdown</label>
                <input type="number" step="0.05" className="term-input"
                  value={cppiDrawdown} onChange={e => setCppiDrawdown(e.target.value)} />
              </div>
              <div className="control-block">
                <label className="label-cppi">CPPI Multiplier</label>
                <input type="number" step="0.5" className="term-input"
                  value={cppiMultiplier} onChange={e => setCppiMultiplier(e.target.value)} />
              </div>
              <div className="control-block">
                <label className="label-hmm-bull">HMM Bull Prior E[r]</label>
                <input type="number" step="0.01" className="term-input"
                  value={hmmBullMu} onChange={e => setHmmBullMu(e.target.value)} />
              </div>
              <div className="control-block">
                <label className="label-hmm-bear">HMM Bear Prior E[r]</label>
                <input type="number" step="0.01" className="term-input"
                  value={hmmBearMu} onChange={e => setHmmBearMu(e.target.value)} />
              </div>
            </div>
          )}
        </div>

        {/* Execute */}
        <button
          className={`term-btn-exec ${loading ? 'loading' : ''}`}
          onClick={handleSimulate}
          disabled={loading}
        >
          {loading ? '⟳  Calculating…' : '▶  Execute [F9]'}
        </button>

        {error && <div className="status-error">⚠ {error}</div>}

        {/* Status log */}
        <div className="status-log">
          <span className="status-log-label">Status</span>
          {data
            ? <span className="status-ready">
                ✓ {Object.keys(data.strategies).length} agents calculated
              </span>
            : <span className="status-waiting">Awaiting parameters…</span>
          }
        </div>
      </aside>

      {/* ── Content pane ── */}
      <section className="content-pane">

        {/* Tab bar */}
        <div className="tab-bar">
          <button
            className={`tab-btn ${activeTab === 'CHART' ? 'active' : ''}`}
            onClick={() => setActiveTab('CHART')}
          >
            Live Chart
          </button>
          <button
            className={`tab-btn ${activeTab === 'STATS' ? 'active' : ''}`}
            onClick={() => setActiveTab('STATS')}
          >
            Math Report
          </button>
        </div>

        {/* CHART TAB */}
        <div style={{
          display:  activeTab === 'CHART' ? 'block' : 'none',
          flex:     1,
          position: 'relative',
        }}>
          <div
            ref={chartContainerRef}
            style={{ position: 'absolute', top: 0, bottom: 0, left: 0, right: 0 }}
          />
        </div>

        {/* STATS TAB */}
        <div style={{
          display:    activeTab === 'STATS' ? 'flex' : 'none',
          flex:       1,
          padding:    '20px',
          overflowY:  'auto',
          flexDirection: 'column',
          gap:        '20px',
        }}>
          {data ? (
            <>
              {/* Agent stat cards */}
              <div className="stat-grid">
                {Object.entries(data.strategies).map(([name, stat]) => (
                  <div
                    className="stat-box"
                    key={name}
                    style={{ borderLeftColor: AGENT_COLORS[name] || 'var(--asu-maroon)' }}
                  >
                    <h3>{name}</h3>
                    <div className="stat-row">
                      <div>
                        <div className="stat-item-label">Median Wealth</div>
                        <div className="stat-val">{stat.final_median_wealth.toFixed(2)}×</div>
                      </div>
                      <div>
                        <div className="stat-item-label">Ruin Risk</div>
                        <div className={`stat-val ${stat.ruin_risk > 0.1 ? 'bad' : 'good'}`}>
                          {(stat.ruin_risk * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Console output */}
              <span className="console-label">Realtime Analytics Synthesis</span>
              <div className="console-output">
                {data.statistics.raw_output}
              </div>
            </>
          ) : (
            <div className="empty-state">
              <span className="empty-state-icon">◎</span>
              <span>No data. Fire simulation first.</span>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

export default Simulator;
