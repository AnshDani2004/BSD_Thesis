import { Routes, Route, Link, useLocation } from 'react-router-dom';
import Home from './pages/Home';
import Simulator from './pages/Simulator';
import Experiments from './pages/Experiments';

function App() {
  const location = useLocation();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <header className="terminal-header">
        <h1>[BSD_THESIS_ENGINE]</h1>
        <div className="toolbar">
          <Link to="/">
            <button className={location.pathname === '/' ? 'active' : ''}>HOME</button>
          </Link>
          <Link to="/simulator">
            <button className={location.pathname === '/simulator' ? 'active' : ''}>SIMULATOR</button>
          </Link>
          <Link to="/experiments">
            <button className={location.pathname === '/experiments' ? 'active' : ''}>ACADEMIC LOGS</button>
          </Link>
        </div>
      </header>
      
      <div className="main-layout" style={{ flex: 1, overflow: 'auto' }}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/simulator" element={<Simulator />} />
          <Route path="/experiments" element={<Experiments />} />
        </Routes>
      </div>
    </div>
  );
}

export default App;
