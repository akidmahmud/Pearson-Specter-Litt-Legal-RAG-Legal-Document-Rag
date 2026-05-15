import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Upload from './pages/Upload';
import Search from './pages/Search';
import Documents from './pages/Documents';
import Chatbot from './components/Chatbot';
import './App.css';

function App() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    checkApiStatus();
  }, []);

  const checkApiStatus = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/health');
      if (response.ok) {
        const data = await response.json();
        setStatus(data);
        setError(null);
      } else {
        setError('API server is not responding');
      }
    } catch (err) {
      setError('Cannot connect to API server. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Router>
      <div className="App">
        <Header status={status} />
        {loading && <div className="loading-message">Checking API connection...</div>}
        {error && <div className="error-banner">{error}</div>}
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Chatbot />} />
            <Route path="/search" element={<Search />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/documents" element={<Documents />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
