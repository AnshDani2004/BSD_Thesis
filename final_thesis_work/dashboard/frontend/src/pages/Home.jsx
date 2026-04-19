import React from 'react';
import { Link } from 'react-router-dom';

const ALGORITHMS = [
  {
    cls: 'card-kelly',
    h4cls: 'kelly',
    title: 'Kelly Criterion',
    math: 'f* = argmax E[ln(1 + fr)]',
    body: 'Finds the optimal bet fraction that maximises compound log-growth while ensuring asymptotic dominance over any fixed alternative strategy (Kelly–Breiman theorem).',
  },
  {
    cls: 'card-thompson',
    h4cls: 'thompson',
    title: 'Thompson Sampling',
    math: 'f̃ ~ Beta(α, β)',
    body: 'Calibrated Bayesian exploration — samples a belief about the best arm and bets accordingly. Asymptotically optimal in stationary IID; suffers O(T₀) detection lag after a regime switch.',
  },
  {
    cls: 'card-cppi',
    h4cls: 'cppi',
    title: 'CPPI Safety Floor',
    math: 'λt = min(1, m·Ct/Wt)',
    body: 'Keeps a mechanical drawdown floor below wealth. Bets only with the cushion above that floor. As the cushion shrinks, leverage drops to zero — regime-agnostic capital protection.',
  },
  {
    cls: 'card-exp3',
    h4cls: 'exp3',
    title: 'EXP3 (Adversarial)',
    math: 'w_i ← w_i · exp(η·r̂_i)',
    body: 'Exponential-weight algorithm with provable regret bounds against adversarial sequences. In the single-asset setting degenerates to fixed f = 0.5 — cannot de-lever in a bear regime.',
  },
  {
    cls: 'card-ucb',
    h4cls: 'ucb',
    title: 'UCB (Optimistic)',
    math: 'μ̄_i + √(2 ln T / n_i)',
    body: 'Adds an exploration bonus to under-sampled options. Forces the agent to investigate "uncertain" paths, trading short-term exploitation for information gain.',
  },
  {
    cls: 'card-volhmm',
    h4cls: 'volhmm',
    title: 'Vol-HMM (Proposed)',
    math: 'ln b_j(r_t, v_t) = ln N + ln Γ',
    body: 'Core thesis contribution. Augments the observation with rolling volatility v_t that spikes immediately on a regime break — bypassing the sticky prior. CUSUM trigger provides an additional hard threshold.',
  },
];

const TIMELINE = [
  {
    title: 'Experiment 1: Stationary Learning',
    desc:  'Can an agent learn a constant probability? Validates that Thompson Sampling and EXP3 converge toward Kelly Oracle growth rates over T = 250 steps.',
  },
  {
    title: 'Experiment 2: Horizon Tradeoffs',
    desc:  'Evaluating the Gambler\'s Ruin problem. How does optimal bet size change when you have a finite deadline to reach a target wealth?',
  },
  {
    title: 'Experiment 4: Adversarial Shock',
    desc:  'The crash test. A structural break at T = 125 flips the market from Gaussian bull to Student-t₃ bear. Which agents survive and which collapse?',
  },
  {
    title: 'Proposed: Vol-Augmented HMM',
    desc:  'Implementing the 2D regime detector. Reduces detection lag from 14.3 to 1.9 steps (≈87% reduction). Validated on 2008 GFC and 2020 COVID S&P 500 data.',
  },
  {
    title: 'Market Realism: Transaction Costs',
    desc:  'Adding BPS friction and computing the true "cost of survival." Vol-HMM achieves 0.15%/yr fee drag vs 0.6%/yr for Bayesian baselines — net advantage exceeds gross.',
  },
];

function Home() {
  return (
    <div className="home-wrapper">

      {/* ── Hero ── */}
      <section style={{ marginBottom: '64px' }}>
        <h1 className="home-hero-title">Bayesian Sequential Decision-Making</h1>
        <p className="home-hero-subtitle">
          Adaptive Risk Management in Non-Stationary, Heavy-Tailed Environments
        </p>
        <div className="home-cta-row">
          <Link to="/simulator" style={{ textDecoration: 'none' }}>
            <button className="btn-primary">▶  Launch Simulator [F9]</button>
          </Link>
          <Link to="/experiments" style={{ textDecoration: 'none' }}>
            <button className="btn-secondary">Academic Logs [F2]</button>
          </Link>
        </div>
      </section>

      {/* ── Problem + Objectives ── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
        gap: '40px',
        marginBottom: '64px',
      }}>
        <div>
          <h2 className="section-heading">01 // The Core Problem</h2>
          <p style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.8, marginBottom: '16px' }}>
            Standard probability theory assumes the future looks like the past.
            In financial crashes, this assumption breaks catastrophically.
            We investigate the <strong style={{ color: 'var(--asu-maroon)' }}>Sticky Prior Paradox</strong>:
            why Bayesian agents "freeze" during crises because their accumulated confidence
            outweighs new evidence of danger.
          </p>
          <p style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
            Adding a <strong style={{ color: 'var(--good)' }}>volatility dimension</strong> to the
            Bayesian observation space reduces regime detection lag by ≈87% relative to
            standard posterior-update agents (Proposition 3.3).
          </p>
        </div>

        <div className="info-card" style={{ borderTop: '3px solid var(--asu-gold)' }}>
          <h4 style={{ color: 'var(--asu-maroon)', marginBottom: '16px' }}>Research Objectives</h4>
          <ul style={{ listStyle: 'none', padding: 0, display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {[
              'Quantify Bayesian detection lag during T=125 structural crashes (Prop 3.3).',
              'Implement Vol-HMM + CUSUM for sub-2-step regime inference.',
              'Provide a CPPI floor with provable continuous-time drawdown bound.',
              'Evaluate net survival cost under realistic BPS transaction friction.',
            ].map((item, i) => (
              <li key={i} style={{ display: 'flex', gap: '10px', alignItems: 'flex-start' }}>
                <span style={{
                  color: 'var(--asu-gold)',
                  background: 'var(--asu-maroon)',
                  borderRadius: '2px',
                  padding: '1px 6px',
                  fontSize: '10px',
                  fontWeight: 700,
                  fontFamily: 'var(--font-mono)',
                  marginTop: '2px',
                  flexShrink: 0,
                }}>
                  {String(i + 1).padStart(2, '0')}
                </span>
                <span style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* ── Algorithm Intuition ── */}
      <h2 className="section-heading" style={{ textAlign: 'center', marginBottom: '28px' }}>
        02 // Algorithm Intuition
      </h2>
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: '20px',
        marginBottom: '64px',
      }}>
        {ALGORITHMS.map(alg => (
          <div key={alg.title} className={`info-card ${alg.cls}`}>
            <h4 className={alg.h4cls}>{alg.title}</h4>
            <code style={{
              display: 'block',
              fontSize: '11px',
              fontFamily: 'var(--font-mono)',
              background: 'var(--surface-2)',
              padding: '6px 10px',
              borderRadius: 'var(--radius)',
              marginBottom: '10px',
              color: 'var(--text-primary)',
            }}>
              {alg.math}
            </code>
            <p>{alg.body}</p>
          </div>
        ))}
      </div>

      {/* ── Infrastructure ── */}
      <h2 className="section-heading" style={{ marginBottom: '16px' }}>
        // Infrastructure
      </h2>
      <div className="infra-grid" style={{ marginBottom: '64px' }}>
        {[
          { label: 'Computing Engine',  val: 'Python 3.11 / Flask' },
          { label: 'Frontend',          val: 'React / Vite / TV-Charts' },
          { label: 'Numerical Core',    val: 'NumPy / SciPy / Numba' },
          { label: 'Stability',         val: 'Log-Wealth Arithmetic' },
          { label: 'Reproducibility',   val: 'Seeded RNG + CRN' },
          { label: 'Statistical Tests', val: 'Fisher Exact / Mann-Whitney' },
        ].map(item => (
          <div key={item.label} className="infra-item">
            <label>{item.label}</label>
            <span>{item.val}</span>
          </div>
        ))}
      </div>

      {/* ── Timeline ── */}
      <h2 className="section-heading" style={{ marginBottom: '24px' }}>
        03 // Experiments &amp; Timeline
      </h2>
      <div style={{ marginBottom: '64px' }}>
        {TIMELINE.map((item, i) => (
          <div key={i} className="timeline-item">
            <h4>{item.title}</h4>
            <p>{item.desc}</p>
          </div>
        ))}
      </div>

      {/* ── Footer ── */}
      <footer style={{
        borderTop: '1px solid var(--border)',
        paddingTop: '24px',
        textAlign: 'center',
        color: 'var(--text-dim)',
        fontSize: '11px',
        fontFamily: 'var(--font-mono)',
        marginBottom: '32px',
      }}>
        Barrett Honors College — Arizona State University — Thesis Defense 2026 — v2.1
      </footer>
    </div>
  );
}

export default Home;
