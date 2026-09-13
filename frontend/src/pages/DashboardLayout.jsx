import { useState } from 'react';
import { Outlet, NavLink, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
  LogOut, User, Shield, Zap, Menu, X, Plus, ArrowLeft, MessageSquare
} from 'lucide-react';

export default function DashboardLayout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  const closeSidebar = () => setSidebarOpen(false);

  const displayName = user?.name || user?.email?.split('@')[0] || 'User';
  const userInitial = displayName[0]?.toUpperCase() || 'U';

  const isChatPage = location.pathname === '/' || location.pathname === '/chat';

  return (
    <div className="app-layout">
      {/* Mobile toggle */}
      <button className="mobile-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
        {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      {/* Mobile overlay */}
      <div
        className={`mobile-overlay ${sidebarOpen ? 'visible' : ''}`}
        onClick={closeSidebar}
      />

      {/* Sidebar - ChatGPT Style navigation */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="sidebar-logo-icon">
              <Zap size={18} color="white" />
            </div>
            <span className="sidebar-logo-text">Resume Analyzer</span>
          </div>
        </div>

        <nav className="sidebar-nav" style={{ padding: '1rem 0.5rem' }}>
          {/* Direct link to return to active Chat Session */}
          <NavLink
            to="/chat"
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            onClick={closeSidebar}
            style={{ marginBottom: '0.75rem', fontWeight: 600 }}
          >
            <MessageSquare size={18} className="text-accent" /> Active Chat Session
          </NavLink>

          {user?.role === 'ADMIN' && (
            <>
              <div className="nav-section-label">Administration</div>
              <NavLink
                to="/admin"
                className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                onClick={closeSidebar}
              >
                <Shield size={18} /> Admin Dashboard
              </NavLink>
            </>
          )}

          <div className="nav-section-label">Workspace</div>
          
          {/* Prominent New Chat Button */}
          <button
            onClick={() => {
              closeSidebar();
              navigate('/chat', { state: { resetSession: Date.now() } });
            }}
            className="btn btn-primary"
            style={{
              width: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem',
              fontSize: '0.9rem',
              fontWeight: 600,
              padding: '0.75rem 1rem',
              borderRadius: 'var(--radius-md)',
              boxShadow: 'var(--shadow-glow)'
            }}
          >
            <Plus size={18} /> New Chat Session
          </button>
        </nav>

        {/* Sidebar Footer Down Side with Profile right above Sign Out (Role Removed) */}
        <div className="sidebar-footer">
          <NavLink
            to="/profile"
            onClick={closeSidebar}
            className={({ isActive }) => `sidebar-user-card ${isActive ? 'active' : ''}`}
            title="View Profile Settings"
          >
            <div className="user-avatar">{userInitial}</div>
            <div className="user-info">
              <div className="user-name" style={{ fontWeight: 600, fontSize: '0.875rem' }}>{displayName}</div>
              <div className="user-email" style={{ fontSize: '0.75rem', opacity: 0.7 }}>{user?.email}</div>
            </div>
            <User size={16} className="user-profile-icon" />
          </NavLink>

          <button
            onClick={handleLogout}
            className="btn btn-secondary"
            style={{ width: '100%', fontSize: '0.8125rem', marginTop: '0.625rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
          >
            <LogOut size={16} /> Sign out
          </button>
        </div>
      </aside>

      {/* Main content - 100% Full Page container for ChatGPT style */}
      <main className={`main-content ${isChatPage ? 'full-page-chat-main' : ''}`}>
        
        {/* Top Header Bar with prominent "Back to Chat" button when on Profile / Admin pages */}
        {!isChatPage && (
          <header style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            paddingBottom: '0.75rem',
            borderBottom: '1px solid var(--border-default)',
            marginBottom: '1.25rem'
          }}>
            {/* Prominent Back to Chat button right beside sidebar */}
            <NavLink
              to="/chat"
              className="btn btn-primary"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.5rem',
                fontSize: '0.875rem',
                fontWeight: 600,
                padding: '0.5rem 1rem',
                borderRadius: 'var(--radius-full)',
                width: 'auto'
              }}
            >
              <ArrowLeft size={16} /> Back to Chat Session
            </NavLink>

            <NavLink
              to="/profile"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.625rem',
                textDecoration: 'none',
                color: 'var(--text-primary)',
                background: 'var(--bg-elevated)',
                border: '1px solid var(--border-default)',
                padding: '0.375rem 0.875rem',
                borderRadius: 'var(--radius-full)',
                transition: 'all 0.2s ease',
                boxShadow: 'var(--shadow-sm)'
              }}
              className="top-profile-badge"
            >
              <div className="user-avatar" style={{ width: 28, height: 28, fontSize: '0.75rem' }}>
                {userInitial}
              </div>
              <span style={{ fontSize: '0.875rem', fontWeight: 600 }}>{displayName}</span>
              <User size={15} style={{ opacity: 0.7 }} />
            </NavLink>
          </header>
        )}

        <div style={{ flex: 1, height: '100%', display: 'flex', flexDirection: 'column' }}>
          <Outlet />
        </div>
      </main>
    </div>
  );
}
