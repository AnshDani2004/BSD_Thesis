import React, { createContext, useContext, useState } from 'react';

const SimulationContext = createContext();

export const SimulationProvider = ({ children }) => {
  // Simulator Settings
  const [tSteps, setTSteps] = useState(250);
  const [nSims, setNSims] = useState(20);
  const [scenario, setScenario] = useState('adversarial');
  const [friction, setFriction] = useState(0.0);
  
  // Advanced Calibration
  const [cppiDrawdown, setCppiDrawdown] = useState(0.20);
  const [cppiMultiplier, setCppiMultiplier] = useState(3.0);
  const [hmmBullMu, setHmmBullMu] = useState(0.08);
  const [hmmBearMu, setHmmBearMu] = useState(-0.10);

  // Simulation Results
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // UI States
  const [activeTab, setActiveTab] = useState('CHART');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const value = {
    tSteps, setTSteps,
    nSims, setNSims,
    scenario, setScenario,
    friction, setFriction,
    cppiDrawdown, setCppiDrawdown,
    cppiMultiplier, setCppiMultiplier,
    hmmBullMu, setHmmBullMu,
    hmmBearMu, setHmmBearMu,
    results, setResults,
    loading, setLoading,
    error, setError,
    activeTab, setActiveTab,
    showAdvanced, setShowAdvanced
  };

  return (
    <SimulationContext.Provider value={value}>
      {children}
    </SimulationContext.Provider>
  );
};

export const useSimulation = () => {
  const context = useContext(SimulationContext);
  if (!context) {
    throw new Error('useSimulation must be used within a SimulationProvider');
  }
  return context;
};
