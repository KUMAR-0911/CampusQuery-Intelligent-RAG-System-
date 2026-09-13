import { useAuth } from '../context/AuthContext';
import { FileText, MessageSquare, UploadCloud, Zap } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function Dashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();

  const firstName = user?.email?.split('@')[0] || 'User';

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Welcome back, {firstName}</h1>
        <p className="page-subtitle">Here's an overview of your CampusQuery workspace</p>
      </div>

      <div className="stats-grid">
        <div className="stat-card" onClick={() => navigate('/documents')} style={{ cursor: 'pointer' }}>
          <div className="stat-icon" style={{ background: 'var(--color-info-bg)' }}>
            <FileText size={20} color="var(--color-info)" />
          </div>
          <div className="stat-label">Documents</div>
          <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Upload &amp; manage your files
          </div>
        </div>

        <div className="stat-card" onClick={() => navigate('/chat')} style={{ cursor: 'pointer' }}>
          <div className="stat-icon" style={{ background: 'rgba(139,92,246,0.12)' }}>
            <MessageSquare size={20} color="var(--accent-3)" />
          </div>
          <div className="stat-label">RAG Chat</div>
          <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Ask questions about your docs
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'var(--color-success-bg)' }}>
            <Zap size={20} color="var(--color-success)" />
          </div>
          <div className="stat-label">System Status</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', marginTop: '0.25rem' }}>
            <span style={{
              width: '8px', height: '8px', borderRadius: '50%',
              background: 'var(--color-success)', display: 'inline-block',
              boxShadow: '0 0 8px rgba(63,185,80,0.4)'
            }} />
            <span style={{ fontSize: '0.8125rem', color: 'var(--color-success)' }}>All systems operational</span>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: '0.5rem' }}>
        <div className="card-header">
          <div>
            <h3 className="card-title">Quick Actions</h3>
            <p className="card-subtitle">Get started with common tasks</p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          <button className="btn btn-secondary" onClick={() => navigate('/documents')}>
            <UploadCloud size={16} /> Upload Documents
          </button>
          <button className="btn btn-primary" onClick={() => navigate('/chat')} style={{ width: 'auto' }}>
            <MessageSquare size={16} /> Start Chat
          </button>
        </div>
      </div>
    </div>
  );
}
