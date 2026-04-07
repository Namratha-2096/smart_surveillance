import React from 'react';
import { Activity, ShieldCheck } from 'lucide-react';

function Header() {
  return (
    <header className="glass-panel header">
      <h1 className="header-title">
        <ShieldCheck size={28} />
        Edge AI Surveillance
      </h1>
      <div className="header-stats">
        <div className="stat-item">
          <span className="stat-label">System Status</span>
          <span className="stat-value">
            <div className="status-dot"></div>
            Online
          </span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Feed Stream</span>
          <span className="stat-value" style={{ color: "var(--accent-blue)" }}>
            <Activity size={18} /> Active
          </span>
        </div>
      </div>
    </header>
  );
}

export default Header;
