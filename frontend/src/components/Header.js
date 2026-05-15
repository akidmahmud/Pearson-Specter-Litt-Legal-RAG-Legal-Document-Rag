import React from 'react';
import { Link } from 'react-router-dom';
import { Scale, Upload, FileText, Search, MessageCircle } from 'lucide-react';
import './Header.css';

function Header({ status }) {
  return (
    <header className="header">
      <div className="header-container">
        <div className="header-left">
          <Link to="/" className="logo">
            <Scale size={24} />
            <h1>Pearson Specter Litt Legal RAG</h1>
          </Link>
          <p className="subtitle">Pearson Specter Litt — Legal Document Intelligence</p>
        </div>

        <nav className="nav">
          <Link to="/" className="nav-link">
            <MessageCircle size={18} />
            <span>Chat</span>
          </Link>
          <Link to="/search" className="nav-link">
            <Search size={18} />
            <span>Search</span>
          </Link>
          <Link to="/upload" className="nav-link">
            <Upload size={18} />
            <span>Upload</span>
          </Link>
          <Link to="/documents" className="nav-link">
            <FileText size={18} />
            <span>Documents</span>
          </Link>
        </nav>

        <div className="status-indicator">
          {status ? (
            <>
              <div className="status-dot active"></div>
              <span className="status-text">Connected</span>
            </>
          ) : (
            <>
              <div className="status-dot inactive"></div>
              <span className="status-text">Connecting...</span>
            </>
          )}
        </div>
      </div>
    </header>
  );
}

export default Header;
