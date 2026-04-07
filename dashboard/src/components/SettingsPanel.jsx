import React, { useState, useEffect } from 'react';
import { Settings } from 'lucide-react';

function SettingsPanel({ config, fetchConfig }) {
  const [localConfig, setLocalConfig] = useState(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (config && !localConfig) {
      setLocalConfig({ ...config });
    }
  }, [config]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setLocalConfig(prev => ({ ...prev, [name]: parseFloat(value) }));
  };

  const handleSave = async () => {
    if (!localConfig) return;
    setSaving(true);
    try {
      await fetch('http://localhost:8000/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(localConfig)
      });
      await fetchConfig();
    } catch (e) {
      console.error(e);
    }
    setSaving(false);
  };

  if (!localConfig) {
    return <div className="glass-panel">Loading settings...</div>;
  }

  return (
    <div className="glass-panel settings-panel">
      <h2 className="settings-title"><Settings size={20} /> Tuning</h2>
      
      <div className="setting-item">
        <label>Loitering Alert Delay (seconds)</label>
        <div className="setting-value">
          <span>{localConfig.loiter_time_sec}s</span>
        </div>
        <input 
          type="range" 
          name="loiter_time_sec" 
          min="1" max="60" step="1" 
          value={localConfig.loiter_time_sec} 
          onChange={handleChange} 
        />
      </div>

      <div className="setting-item">
        <label>Face Verification Strictness</label>
        <div className="setting-value">
          <span>{localConfig.face_threshold}</span>
        </div>
        <input 
          type="range" 
          name="face_threshold" 
          min="0.1" max="1.0" step="0.01" 
          value={localConfig.face_threshold} 
          onChange={handleChange} 
        />
      </div>

      <div className="setting-item">
        <label>Fall Aspect Ratio Trigger</label>
        <div className="setting-value">
          <span>{localConfig.fall_aspect_ratio}</span>
        </div>
        <input 
          type="range" 
          name="fall_aspect_ratio" 
          min="0.5" max="2.0" step="0.1" 
          value={localConfig.fall_aspect_ratio} 
          onChange={handleChange} 
        />
      </div>

      <button className="btn-save" onClick={handleSave} disabled={saving}>
        {saving ? 'Saving...' : 'Apply Filters'}
      </button>
    </div>
  );
}

export default SettingsPanel;
