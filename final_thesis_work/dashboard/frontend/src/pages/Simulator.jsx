import React, { useEffect, useRef } from 'react';
import { createChart } from 'lightweight-charts';
import { useSimulation } from '../context/SimulationContext';

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
    showAdvanced, setShowAdvanced
  } = useSimulation();

  const HISTORICAL_KEYS = ['real_2000_dotcom', 'real_2008_crisis', 'real_2020_covid', 'real_2013_bull'];
  const isHistorical = HISTORICAL_KEYS.includes(scenario);

  const chartContainerRef = useRef();
  const chartRef = useRef(null);
  const seriesMap = useRef({});

  const COLORS = {
    'Vol-Augmented HMM (Proposed)': '#00FF00', // Neon Green
    'Risk-Constrained CPPI': '#FF33FF',        // Neon Pink
    'Thompson Sampling': '#FFB000',
    'EXP3': '#00E5FF',
    'Naive Bayes': '#FF0033',
    'UCB': '#A64D79',
    'Softmax': '#FF9900'
  };

  const handleSimulate = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://127.0.0.1:5001/api/simulate/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          t_steps: tSteps, 
          n_sims: nSims, 
          scenario: scenario,
          friction: friction,
          advanced_params: {
            cppi_drawdown: cppiDrawdown,
            cppi_multiplier: cppiMultiplier,
            hmm_bull_mu: hmmBullMu,
            hmm_bear_mu: hmmBearMu
          }
        })
      });
      const resData = await res.json();
      if(resData.status === 'success') {
        setData(resData);
        updateChart(resData.strategies);
        if(activeTab !== 'CHART' && activeTab !== 'STATS') {
           setActiveTab('CHART');
        }
      } else {
        setError('API Error: Simulation failed.');
      }
    } catch (e) {
      console.error(e);
      setError('Connection refused. Is the Python backend running on 5001?');
    }
    setLoading(false);
  };

  const updateChart = (strategies) => {
    if (!chartRef.current) return;
    Object.values(seriesMap.current).forEach(series => chartRef.current.removeSeries(series));
    seriesMap.current = {};

    Object.entries(strategies).forEach(([name, strat]) => {
      const series = chartRef.current.addLineSeries({
        color: COLORS[name] || '#FFF',
        lineWidth: 2,
        title: name,
      });
      series.setData(strat.chart_data);
      seriesMap.current[name] = series;
    });
    chartRef.current.timeScale().fitContent();
  };

  useEffect(() => {
    if (chartContainerRef.current && !chartRef.current) {
      chartRef.current = createChart(chartContainerRef.current, {
        layout: {
          background: { type: 'solid', color: '#000000' },
          textColor: '#FFB000',
          fontFamily: "'Fira Code', 'Courier New', Courier, monospace"
        },
        grid: {
          vertLines: { color: '#333333' },
          horzLines: { color: '#333333' },
        },
        crosshair: { mode: 0 },
        rightPriceScale: { borderColor: '#333333' }
      });
      
      const handleResize = () => {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      };
      window.addEventListener('resize', handleResize);
      
      // If data already exists (from Context), hydrate chart
      if (data && data.strategies) {
        updateChart(data.strategies);
      }

      return () => window.removeEventListener('resize', handleResize);
    }
  }, []);

  return (
    <div style={{ display: 'flex', height: '100%', width: '100%' }}>
      <aside className="sidebar">
        <div className="control-block">
          <label>SCENARIO TYPE</label>
          <select 
            className="term-input" 
            value={scenario}
            onChange={e => setScenario(e.target.value)}
          >
            <optgroup label="── Synthetic Environments ──">
              <option value="stationary">Stationary (Flat MAB)</option>
              <option value="adversarial">Adversarial Crash</option>
              <option value="drift">Slow Gaussian Drift</option>
            </optgroup>
            <optgroup label="── Real-World Historical ──">
              <option value="real_2000_dotcom">S&amp;P 500 · 2000 Dot-Com Collapse</option>
              <option value="real_2008_crisis">S&amp;P 500 · 2008 Financial Crisis</option>
              <option value="real_2020_covid">S&amp;P 500 · 2020 COVID Crash</option>
              <option value="real_2013_bull">S&amp;P 500 · 2013–2019 Bull Market</option>
            </optgroup>
          </select>
          {isHistorical && (
            <div style={{ marginTop: '8px', padding: '4px 8px', background: '#001a00', border: '1px solid var(--term-green)', fontSize: '10px', color: 'var(--term-green)' }}>
              ▶ EMPIRICAL MODE: Single historical path. Monte Carlo parameters ignored.
            </div>
          )}
        </div>
        <div className="control-block">
          <label>HORIZON [STEPS_T]</label>
          <input 
            type="number" 
            className="term-input" 
            value={tSteps} 
            onChange={e => setTSteps(e.target.value)} 
          />
        </div>
        <div className="control-block">
          <label>MONTE CARLO [N_SIMS]</label>
          <input 
            type="number" 
            className="term-input" 
            value={nSims} 
            onChange={e => setNSims(e.target.value)} 
          />
        </div>
        <div className="control-block">
          <label>TRANSACTION COSTS [BPS]</label>
          <input 
            type="number" 
            step="0.1"
            className="term-input" 
            value={friction} 
            onChange={e => setFriction(e.target.value)} 
          />
          <div style={{fontSize: '9px', color: 'var(--term-dim)', marginTop: '4px'}}>
            1 bps = 0.01% fee on trade volume.
          </div>
        </div>

        {/* ADVANCED CALIBRATION ACCORDION */}
        <div style={{ marginTop: '20px', borderTop: '1px solid #333', paddingTop: '15px' }}>
          <button 
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="term-input"
            style={{ width: '100%', textAlign: 'left', background: '#111', cursor: 'pointer', display: 'flex', justifyContent: 'space-between' }}
          >
            <span>{showAdvanced ? '▼' : '▶'} ADVANCED CALIBRATION</span>
          </button>
          
          {showAdvanced && (
            <div style={{ marginTop: '10px', paddingLeft: '5px' }}>
              <div className="control-block">
                <label style={{ fontSize: '10px', color: 'var(--term-cyan)' }}>CPPI MAX DRAWDOWN</label>
                <input type="number" step="0.05" className="term-input" value={cppiDrawdown} onChange={e => setCppiDrawdown(e.target.value)} />
              </div>
              <div className="control-block">
                <label style={{ fontSize: '10px', color: 'var(--term-cyan)' }}>CPPI MULTIPLIER</label>
                <input type="number" step="0.5" className="term-input" value={cppiMultiplier} onChange={e => setCppiMultiplier(e.target.value)} />
              </div>
              <div className="control-block">
                <label style={{ fontSize: '10px', color: 'var(--term-green)' }}>HMM BULL prior E[r]</label>
                <input type="number" step="0.01" className="term-input" value={hmmBullMu} onChange={e => setHmmBullMu(e.target.value)} />
              </div>
              <div className="control-block">
                <label style={{ fontSize: '10px', color: 'var(--term-red)' }}>HMM BEAR prior E[r]</label>
                <input type="number" step="0.01" className="term-input" value={hmmBearMu} onChange={e => setHmmBearMu(e.target.value)} />
              </div>
            </div>
          )}
        </div>

        <button className="term-btn-exec" onClick={handleSimulate}>
          {loading ? 'CALCULATING...' : 'EXECUTE [F9]'}
        </button>

        {error && <div style={{color: 'var(--term-red)', fontSize: '11px', mt: 2}}>ERR: {error}</div>}

        <div style={{marginTop: 'auto', borderTop: '1px solid var(--term-border)', paddingTop: '15px'}}>
          <label style={{color: 'var(--term-dim)', fontSize: '10px'}}>STATUS LOG</label><br/>
          <span style={{color: data ? 'var(--term-green)' : 'var(--term-amber)', fontSize: '11px'}}>
            {data ? '> SIM RUN. ' + Object.keys(data.strategies).length + ' AGENTS CALCULATED.' : '> AWAITING PARAMS.'}
          </span>
        </div>
      </aside>

      <section className="content-pane" style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', borderBottom: '1px solid var(--term-border)' }}>
            <button className="term-input" style={{flex: 1, border: 'none', background: activeTab === 'CHART' ? '#111' : 'transparent'}} onClick={() => setActiveTab('CHART')}>LIVE CHART</button>
            <button className="term-input" style={{flex: 1, border: 'none', background: activeTab === 'STATS' ? '#111' : 'transparent'}} onClick={() => setActiveTab('STATS')}>MATH REPORT</button>
        </div>

        {/* CHART TAB */}
        <div style={{ display: activeTab === 'CHART' ? 'block' : 'none', flex: 1, position: 'relative' }}>
          <div ref={chartContainerRef} style={{ position: 'absolute', top: 0, bottom: 0, left: 0, right: 0 }} />
        </div>

        {/* STATS TAB */}
        <div style={{ display: activeTab === 'STATS' ? 'block' : 'none', flex: 1, padding: '20px', overflowY: 'auto' }}>
          {data ? (
            <>
              <div className="stat-grid" style={{ marginBottom: '20px' }}>
                {Object.entries(data.strategies).map(([name, stat]) => (
                  <div className="stat-box" key={name} style={{ borderLeft: `4px solid ${COLORS[name] || '#FFF'}`}}>
                    <h3 style={{ color: '#FFF' }}>{name}</h3>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <div>
                            <span style={{fontSize: '10px', color: 'var(--term-dim)'}}>MEDIAN WEALTH</span><br/>
                            <span className="stat-val">{stat.final_median_wealth.toFixed(2)}x</span>
                        </div>
                        <div>
                            <span style={{fontSize: '10px', color: 'var(--term-dim)'}}>RUIN RISK</span><br/>
                            <span className={`stat-val ${stat.ruin_risk > 0.1 ? 'bad' : 'good'}`}>{(stat.ruin_risk * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                  </div>
                ))}
              </div>
              
              <label style={{color: 'var(--term-cyan)', marginBottom: '10px'}}>REALTIME ANALYTICS SYNTHESIS</label>
              <div className="console-output">
                {data.statistics.raw_output}
              </div>
            </>
          ) : (
            <div style={{textAlign: 'center', marginTop: '100px', color: 'var(--term-dim)'}}>
              [ NO DATA. FIRE SIMULATION FIRST. ]
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

export default Simulator;
