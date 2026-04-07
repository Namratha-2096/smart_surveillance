import React, { useState } from 'react';
import { VideoOff } from 'lucide-react';

function LiveView() {
  const [key, setKey] = useState(0);
  const [errorAct, setErrorAct] = useState(false);

  return (
    <div className="glass-panel video-container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px' }}>
      {errorAct && (
        <div style={{ position: 'absolute', zIndex: 10, display: 'flex', flexDirection: 'column', alignItems: 'center', color: 'var(--text-secondary)' }}>
          <VideoOff size={48} style={{ marginBottom: '10px' }} />
          <span>Waiting for video feed...</span>
        </div>
      )}
      <img 
        key={key}
        src={`http://localhost:8000/video_feed?retry=${key}`} 
        alt="Live Camera Feed"
        className="video-feed"
        style={{ opacity: errorAct ? 0.3 : 1, transition: 'opacity 0.3s' }}
        onLoad={() => setErrorAct(false)}
        onError={(e) => { 
          setErrorAct(true);
          setTimeout(() => setKey(k => k + 1), 2000); 
        }} 
      />
    </div>
  );
}

export default LiveView;
