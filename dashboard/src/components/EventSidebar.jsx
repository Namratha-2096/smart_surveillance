import React from 'react';
import { AlertTriangle, UserX, Eye } from 'lucide-react';

function EventSidebar({ events }) {
  const getIcon = (type) => {
    switch (type) {
      case 'fall': return <AlertTriangle size={18} />;
      case 'loitering': return <Eye size={18} />;
      case 'face_unknown': return <UserX size={18} />;
      default: return null;
    }
  };

  const getLabel = (type) => {
    switch (type) {
      case 'fall': return 'Fall Detected';
      case 'loitering': return 'Loitering';
      case 'face_unknown': return 'Unknown Face';
      default: return type;
    }
  };

  return (
    <div className="glass-panel event-list" style={{ minHeight: '300px' }}>
      <h2 style={{ fontSize: '1.2rem', marginBottom: '15px', marginTop: 0 }}>Recent Alerts</h2>
      {events.length === 0 ? (
        <div style={{ color: 'var(--text-secondary)', textAlign: 'center', marginTop: '20px' }}>
          No recent events logged.
        </div>
      ) : (
        events.map((ev) => (
          <div key={ev.id} className={`event-item type-${ev.type}`}>
            <div className="event-header">
              <span className="event-type">
                {getIcon(ev.type)} {getLabel(ev.type)}
              </span>
              <span className="event-time">
                {ev.timestamp.split(' ')[1]}
              </span>
            </div>
            <div className="event-detail">{ev.detail}</div>
          </div>
        ))
      )}
    </div>
  );
}

export default EventSidebar;
