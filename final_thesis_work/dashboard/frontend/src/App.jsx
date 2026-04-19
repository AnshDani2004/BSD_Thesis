import { Routes, Route, Link, useLocation } from 'react-router-dom';
import Home from './pages/Home';
import Simulator from './pages/Simulator';
import Experiments from './pages/Experiments';

function App() {
  const location = useLocation();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <header className="terminal-header">
        <h1>BSD_THESIS_ENGINE</h1>
        <nav className="toolbar">
          <Link to="/">
            <button className={location.pathname === '/' ? 'active' : ''}>
              Home
            </button>
          </Link>
          <Link to="/simulator">
            <button className={location.pathname === '/simulator' ? 'active' : ''}>
              Simulator
            </button>
          </Link>
          <Link to="/experiments">
            <button className={location.pathname === '/experiments' ? 'active' : ''}>
              Academic Logs
            </button>
          </Link>
        </nav>
      </header>

      <div className="main-layout" style={{ flex: 1, overflow: 'auto' }}>
        <Routes>
          <Route path="/"            element={<Home />} />
          <Route path="/simulator"   element={<Simulator />} />
          <Route path="/experiments" element={<Experiments />} />
        </Routes>
      </div>
    </div>
  );
}

export default App;
