import React from 'react';
import { Link } from 'react-router-dom';

function Home() {
  return (
    <div style={{ padding: '60px 20px', maxWidth: '1200px', margin: '0 auto', color: '#FFF', fontFamily: "'Fira Code', monospace" }}>
      {/* HERO SECTION */}
      <section style={{ marginBottom: '80px', textAlign: 'center' }}>
        <h1 style={{ fontSize: 'clamp(32px, 5vw, 48px)', fontWeight: '800', marginBottom: '20px', color: 'var(--term-cyan)', letterSpacing: '-1.5px' }}>
          BAYESIAN SEQUENTIAL DECISIONS
        </h1>
        <h2 style={{ fontSize: 'clamp(14px, 2vw, 18px)', color: 'var(--term-dim)', fontWeight: '400', letterSpacing: '3px', marginBottom: '50px', textTransform: 'uppercase' }}>
          Adaptive Risk Management in Non-Stationary Environments
        </h2>
        <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'center', alignItems: 'center', gap: '30px', flexWrap: 'wrap', width: '100%' }}>
          <Link to="/simulator" style={{ textDecoration: 'none', display: 'flex' }}>
            <button className="term-btn-exec" style={{ 
              minWidth: '240px', 
              height: '50px',
              padding: '0 24px', 
              margin: '0', 
              fontSize: '13px', 
              fontWeight: '700',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              letterSpacing: '1px',
              cursor: 'pointer'
            }}>
              LAUNCH SIMULATOR [F9]
            </button>
          </Link>
          <Link to="/experiments" style={{ textDecoration: 'none', display: 'flex' }}>
            <button className="term-input" style={{ 
              minWidth: '240px', 
              height: '50px',
              padding: '0 24px', 
              margin: '0', 
              fontSize: '13px', 
              background: 'transparent', 
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid #444',
              color: 'var(--term-amber)',
              letterSpacing: '1px'
            }}>
              ACADEMIC LOGS [F2]
            </button>
          </Link>
        </div>
      </section>

      {/* RATIONALE GRID */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '50px', marginBottom: '80px' }}>
        <div>
          <h2 style={{ color: 'var(--term-cyan)', borderBottom: '1px solid #333', paddingBottom: '15px', marginBottom: '25px', fontSize: '20px' }}>
             01 // THE CORE PROBLEM
          </h2>
          <p style={{ lineHeight: '1.8', color: 'var(--term-dim)', fontSize: '15px' }}>
            Standard probability theory assumes the future looks like the past. In financial crashes, this assumption breaks. 
            We investigate the <strong>"Sticky Prior Paradox"</strong>: why Bayesian computers "freeze" during crisis because their historical confidence outweighs new evidence of danger.
          </p>
          <p style={{ lineHeight: '1.8', color: 'var(--term-dim)', marginTop: '20px', fontSize: '15px' }}>
            Our thesis proves that by adding a <strong>Volatility Dimension</strong> to the Bayesian brain, we can detect regime changes $87\%$ faster than standard models.
          </p>
        </div>
        <div style={{ background: 'rgba(255,176,0,0.03)', padding: '35px', border: '1px solid rgba(255,176,0,0.1)', borderRadius: '2px' }}>
          <h3 style={{ color: 'var(--term-amber)', marginBottom: '20px', fontSize: '16px' }}>RESEARCH OBJECTIVES</h3>
          <ul style={{ listStyle: 'none', padding: 0, fontSize: '13px' }}>
            <li style={{ marginBottom: '15px', display: 'flex', alignItems: 'start' }}>
               <span style={{ color: 'var(--term-amber)', marginRight: '10px' }}>▶</span>
               <span>Quantify Bayesian lag during $T=125$ structural crashes.</span>
            </li>
            <li style={{ marginBottom: '15px', display: 'flex', alignItems: 'start' }}>
               <span style={{ color: 'var(--term-amber)', marginRight: '10px' }}>▶</span>
               <span>Implement HMM-based likelihood ratios for regime inference.</span>
            </li>
            <li style={{ marginBottom: '15px', display: 'flex', alignItems: 'start' }}>
               <span style={{ color: 'var(--term-amber)', marginRight: '10px' }}>▶</span>
               <span>Evaluate survival cost (Friction) in S&P 500 Historical Replay.</span>
            </li>
          </ul>
        </div>
      </div>

      {/* STRATEGY GLOSSARY (UNDERGRAD MATH) */}
      <h2 style={{ textAlign: 'center', marginBottom: '40px', color: 'var(--term-green)', fontSize: '20px' }}>
        02 // ALGORITHM INTUITION (UNDERGRAD LEVEL)
      </h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '25px', marginBottom: '80px' }}>
        <div className="stat-box" style={{ padding: '25px' }}>
          <h4 style={{ color: 'var(--term-cyan)', marginBottom: '10px' }}>KELLY CRITERION</h4>
          <p style={{ fontSize: '12px', color: 'var(--term-dim)', lineHeight: '1.6' }}>
            <strong>Math:</strong> {'$arg max E[ln(W)]$'}<br/>
            <strong>Intuition:</strong> Finds the perfect bet size that maximizes compound growth over time while mathematically ensuring you never hit zero wealth.
          </p>
        </div>
        <div className="stat-box" style={{ padding: '25px' }}>
          <h4 style={{ color: 'var(--term-amber)' }}>THOMPSON SAMPLING</h4>
          <p style={{ fontSize: '12px', color: 'var(--term-dim)', lineHeight: '1.6' }}>
            <strong>Math:</strong> {'$f \\sim P(\\theta | D)$'}<br/>
            <strong>Intuition:</strong> A "brave" Bayesian approach. It gambles on the best option based on a random sample of its current belief, balancing learning vs. earning.
          </p>
        </div>
        <div className="stat-box" style={{ padding: '25px' }}>
          <h4 style={{ color: 'var(--term-pink)' }}>CPPI SAFETY FLOOR</h4>
          <p style={{ fontSize: '12px', color: 'var(--term-dim)', lineHeight: '1.6' }}>
            <strong>Math:</strong> {'$Exposure = m(W - Floor)$'}<br/>
            <strong>Intuition:</strong> Keeps a "Safety Floor" below your money. It only lets you bet with the "Cushion" above that floor. If the cushion hits zero, you stop betting.
          </p>
        </div>
        <div className="stat-box" style={{ padding: '25px' }}>
          <h4 style={{ color: 'var(--term-red)' }}>EXP3 (ROBUST)</h4>
          <p style={{ fontSize: '12px', color: 'var(--term-dim)', lineHeight: '1.6' }}>
            <strong>Math:</strong> {'$w_i(t+1) = w_i(t) \\exp(\\eta \\hat{r}_i)$'}<br/>
            <strong>Intuition:</strong> A "forgetful" expert. It heavily weights recent performance so it doesn't get trapped by old ideas that worked yesterday but fail today.
          </p>
        </div>
        <div className="stat-box" style={{ padding: '25px' }}>
          <h4 style={{ color: '#A64D79' }}>UCB (OPTIMISTIC)</h4>
          <p style={{ fontSize: '12px', color: 'var(--term-dim)', lineHeight: '1.6' }}>
            <strong>Math:</strong> {'$\\bar{\\mu}_i + \\sqrt{\\frac{2 \\ln T}{n_i}}$'}<br/>
            <strong>Intuition:</strong> Adds a "Bonus" to options it hasn't seen in a while. Forces the computer to explore "mysterious" paths just in case they are better.
          </p>
        </div>
        <div className="stat-box" style={{ padding: '25px' }}>
          <h4 style={{ color: 'var(--term-green)' }}>VOL-HMM (PROPOSED)</h4>
          <p style={{ fontSize: '12px', color: 'var(--term-dim)', lineHeight: '1.6' }}>
            <strong>Math:</strong> {'$P(S_t | r_t, \\sigma_{roll})$'}<br/>
            <strong>Intuition:</strong> Our core thesis. It looks at the "Chaos" (Volatility) to guess if the market has shifted regime, allowing for instant defensive maneuvers.
          </p>
        </div>
      </div>

      {/* TECH STACK */}
      <div style={{ border: '1px solid #333', padding: '30px', background: '#080808', marginBottom: '80px', position: 'relative' }}>
        <div style={{ position: 'absolute', top: '-10px', left: '20px', background: '#000', padding: '0 10px', fontSize: '10px', color: 'var(--term-cyan)' }}>
           // CRITICAL_INFRASTRUCTURE
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '30px' }}>
          <div>
            <label style={{ fontSize: '10px', color: 'var(--term-dim)', textTransform: 'uppercase' }}>Computing Engine</label><br />
            <span style={{ fontSize: '14px', color: '#FFF' }}>Python 3.11 / Flask</span>
          </div>
          <div>
            <label style={{ fontSize: '10px', color: 'var(--term-dim)', textTransform: 'uppercase' }}>Visual Matrix</label><br />
            <span style={{ fontSize: '14px', color: '#FFF' }}>React / Vite / TV-Charts</span>
          </div>
          <div>
            <label style={{ fontSize: '10px', color: 'var(--term-dim)', textTransform: 'uppercase' }}>Numerical Core</label><br />
            <span style={{ fontSize: '14px', color: '#FFF' }}>Numba / NumPy / SciPy</span>
          </div>
          <div>
            <label style={{ fontSize: '10px', color: 'var(--term-dim)', textTransform: 'uppercase' }}>Log Stability</label><br />
            <span style={{ fontSize: '14px', color: '#FFF' }}>Log-Wealth Arithmetic</span>
          </div>
        </div>
      </div>

      {/* TIMELINE */}
      <h2 style={{ borderBottom: '1px solid var(--term-border)', paddingBottom: '15px', marginBottom: '40px', fontSize: '20px' }}>
        03 // PROJECT EXPERIMENTS & TIMELINE
      </h2>
      <div style={{ display: 'grid', gap: '20px' }}>
        {[
          { color: 'var(--term-cyan)', title: 'EXPERIMENT 1: STATIONARY LEARNING', desc: 'Can an agent learn a constant probability? Proves that Thompson Sampling matches the Kelly Oracle over 1000 steps.' },
          { color: 'var(--term-amber)', title: 'EXPERIMENT 2: TARGET HORIZONS', desc: 'Evaluating "The Gambler\'s Ruin." Does betting size change when you have a 250-day deadline to double your money?' },
          { color: 'var(--term-red)', title: 'EXPERIMENT 4: ADVERSARIAL SHOCKS', desc: 'The "Crash Test." We suddenly break the world at T=125 to see which Bayesian brains survive and which go extinct.' },
          { color: 'var(--term-green)', title: 'PHASE IV: PROPOSED VOL-HMM', desc: 'Implementing the 2D regime detector. Proves survival in 2008 and 2020 Real-World S&P 500 crash data.' },
          { color: 'var(--term-pink)', title: 'PHASE VI: MARKET REALISM', desc: 'Adding Transaction Fees (BPS) and slippage. Calculating the real "Cost of Survival" in modern markets.' }
        ].map((phase, idx) => (
          <div key={idx} style={{ borderLeft: `2px solid ${phase.color}`, paddingLeft: '25px', marginBottom: '20px' }}>
            <h4 style={{ color: phase.color, marginBottom: '8px', fontSize: '14px' }}>{phase.title}</h4>
            <p style={{ fontSize: '13px', color: 'var(--term-dim)', lineHeight: '1.6' }}>{phase.desc}</p>
          </div>
        ))}
      </div>
      
      <footer style={{ marginTop: '100px', textAlign: 'center', borderTop: '1px solid #333', paddingTop: '40px', color: 'var(--term-dim)', fontSize: '11px' }}>
        [ END OF LANDING PAGE ] — PREPARED FOR THESIS DEFENSE 2026 — VERSION 2.1.4
      </footer>
    </div>
  );
}

export default Home;
