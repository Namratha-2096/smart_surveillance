import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import LiveView from './components/LiveView';
import EventSidebar from './components/EventSidebar';
import SettingsPanel from './components/SettingsPanel';

function App() {
  const [events, setEvents] = useState([]);
  const [config, setConfig] = useState(null);

  useEffect(() => {
    if (Notification && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, []);

  const fetchEvents = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/events');
      if (!res.ok) return;
      const data = await res.json();
      setEvents(prev => {
        const knownIds = new Set(prev.map(e => e.id));
        const newEvents = data.events.filter(e => !knownIds.has(e.id));
        
        newEvents.forEach(e => {
          if (e.type === 'fall' && Notification.permission === 'granted') {
            new Notification('CRITICAL: Fall Detected!', {
              body: `Details: ${e.detail} at ${e.timestamp}`,
            });
          }
        });
        
        return data.events;
      });
    } catch (e) {
      console.error("Failed to fetch events", e);
    }
  };

  const fetchConfig = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/config');
      if (!res.ok) return;
      const data = await res.json();
      setConfig(data);
    } catch (e) {
      console.error("Failed to fetch config", e);
    }
  };

  useEffect(() => {
    fetchConfig();
    fetchEvents();
    const interval = setInterval(() => {
      fetchEvents();
      fetchConfig();
    }, 2000); // reduced frequency to 2000ms to save CPU
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="dashboard-layout">
      <Header />
      <div className="main-content">
        <LiveView />
        <div className="sidebar">
          <SettingsPanel config={config} fetchConfig={fetchConfig} />
          <EventSidebar events={events} />
        </div>
      </div>
    </div>
  );
}

export default App;
