import { useState, useEffect, useMemo } from 'react';
import {
  Users,
  UserCheck,
  Lock,
  Clock,
  Search,
  Trash2,
  Activity,
  Gauge,
  Zap,
  RefreshCw,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  CheckCircle,
  Shield,
} from 'lucide-react';
import api, { getErrorMessage } from '../api/api';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../context/ToastContext';

export default function AdminDashboard() {
  const { user: currentAdmin } = useAuth();
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');

  // Metrics state
  const [timeWindow, setTimeWindow] = useState(24);
  const [metricsData, setMetricsData] = useState(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState('');
  const [showEndpointBreakdown, setShowEndpointBreakdown] = useState(false);

  // Delete modal state
  const [userToDelete, setUserToDelete] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const toast = useToast();

  useEffect(() => {
    fetchUsers();
    fetchMetrics(timeWindow);
  }, [timeWindow]);

  const fetchUsers = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await api.get('/users');
      setUsers(res.data);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const fetchMetrics = async (hours = timeWindow) => {
    setMetricsLoading(true);
    setMetricsError('');
    try {
      const res = await api.get(`/admin/metrics/latency?hours=${hours}`);
      setMetricsData(res.data);
    } catch (err) {
      setMetricsError(getErrorMessage(err));
    } finally {
      setMetricsLoading(false);
    }
  };

  const handleRefreshAll = () => {
    fetchUsers();
    fetchMetrics(timeWindow);
  };

  const updateStatus = async (userId, newStatus) => {
    const previousUsers = [...users];
    // Optimistic update in UI
    setUsers((prev) => prev.map((u) => (u.id === userId ? { ...u, status: newStatus } : u)));
    try {
      await api.put(`/users/${userId}/status?status=${newStatus}`);
      toast.success(`User status updated to ${newStatus}`);
    } catch (err) {
      setUsers(previousUsers);
      toast.error(getErrorMessage(err));
    }
  };

  const confirmDeleteUser = async () => {
    if (!userToDelete) return;
    const target = userToDelete;
    const previousUsers = [...users];

    // Optimistic instant UI removal: 0ms perceived latency
    setUsers((prev) => prev.filter((u) => u.id !== target.id));
    setUserToDelete(null);
    setIsDeleting(false);

    try {
      const res = await api.delete(`/users/${target.id}`);
      toast.success(res.data?.message || `User ${target.email} has been permanently deleted.`);
      fetchMetrics(timeWindow);
    } catch (err) {
      // Revert state if backend delete fails
      setUsers(previousUsers);
      toast.error(getErrorMessage(err));
    }
  };

  const filteredUsers = useMemo(() => {
    if (!search.trim()) return users;
    const q = search.toLowerCase();
    return users.filter(
      (u) =>
        u.email.toLowerCase().includes(q) ||
        u.role.toLowerCase().includes(q) ||
        u.status.toLowerCase().includes(q)
    );
  }, [users, search]);

  const stats = useMemo(
    () => ({
      total: users.length,
      active: users.filter((u) => u.status === 'ACTIVE').length,
      locked: users.filter((u) => u.status === 'LOCKED').length,
      pending: users.filter((u) => u.status === 'PENDING_VERIFICATION').length,
    }),
    [users]
  );

  const kpis = metricsData?.kpi_metrics || {};

  return (
    <div>
      {/* Header */}
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h1 className="page-title">Admin Dashboard & Observability</h1>
          <p className="page-subtitle">Real-time API performance metrics, percentile distributions, and user governance</p>
        </div>
        <button
          type="button"
          onClick={handleRefreshAll}
          disabled={loading || metricsLoading}
          className="admin-btn-cancel"
          style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}
          title="Refresh metrics and user records"
        >
          <RefreshCw size={15} className={loading || metricsLoading ? 'spinner' : ''} />
          <span>Refresh Data</span>
        </button>
      </div>

      {/* Observability & Latency Percentiles Section */}
      <div className="admin-metrics-section">
        <div className="metrics-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
            <Activity size={20} color="var(--accent-1)" />
            <h2 style={{ fontSize: '1.125rem', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
              API Latency & Percentiles
            </h2>
          </div>

          {/* Time Window Selector */}
          <div className="time-window-selector">
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', padding: '0 0.5rem' }}>
              WINDOW:
            </span>
            {[
              { label: '1h', hours: 1 },
              { label: '6h', hours: 6 },
              { label: '24h', hours: 24 },
              { label: '7d', hours: 168 },
            ].map((tw) => (
              <button
                key={tw.hours}
                type="button"
                className={`time-btn ${timeWindow === tw.hours ? 'active' : ''}`}
                onClick={() => setTimeWindow(tw.hours)}
              >
                {tw.label}
              </button>
            ))}
          </div>
        </div>

        {metricsError && <div className="alert alert-error">{metricsError}</div>}

        {/* Latency Percentile Cards (P50, P95, P99) */}
        <div className="stats-grid">
          {/* P50 Latency (Median) */}
          <div className="stat-card" style={{ borderLeft: '4px solid #059669' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
              <div className="stat-icon" style={{ background: 'rgba(5, 150, 105, 0.1)', marginBottom: 0 }}>
                <Gauge size={20} color="#059669" />
              </div>
              <span className="percentile-badge badge-p50">P50 (Median)</span>
            </div>
            <div className="stat-value" style={{ color: '#059669' }}>
              {kpis.p50_latency_ms !== undefined ? `${kpis.p50_latency_ms} ms` : '—'}
            </div>
            <div className="stat-label">P50 Latency</div>
            <p style={{ margin: '0.35rem 0 0 0', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              50% of requests served faster than this
            </p>
          </div>

          {/* P95 Latency */}
          <div className="stat-card" style={{ borderLeft: '4px solid #d97706' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
              <div className="stat-icon" style={{ background: 'rgba(217, 119, 6, 0.1)', marginBottom: 0 }}>
                <Activity size={20} color="#d97706" />
              </div>
              <span className="percentile-badge badge-p95">P95 (95th %)</span>
            </div>
            <div className="stat-value" style={{ color: '#d97706' }}>
              {kpis.p95_latency_ms !== undefined ? `${kpis.p95_latency_ms} ms` : '—'}
            </div>
            <div className="stat-label">P95 Latency</div>
            <p style={{ margin: '0.35rem 0 0 0', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              95% of requests served faster than this
            </p>
          </div>

          {/* P99 Latency */}
          <div className="stat-card" style={{ borderLeft: '4px solid #dc2626' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
              <div className="stat-icon" style={{ background: 'rgba(220, 38, 38, 0.1)', marginBottom: 0 }}>
                <Zap size={20} color="#dc2626" />
              </div>
              <span className="percentile-badge badge-p99">P99 (Tail)</span>
            </div>
            <div className="stat-value" style={{ color: '#dc2626' }}>
              {kpis.p99_latency_ms !== undefined ? `${kpis.p99_latency_ms} ms` : '—'}
            </div>
            <div className="stat-label">P99 Latency</div>
            <p style={{ margin: '0.35rem 0 0 0', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Worst 1% tail latency experienced by users
            </p>
          </div>

          {/* System Traffic */}
          <div className="stat-card" style={{ borderLeft: '4px solid var(--accent-1)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
              <div className="stat-icon" style={{ background: 'rgba(67, 56, 202, 0.1)', marginBottom: 0 }}>
                <Clock size={20} color="var(--accent-1)" />
              </div>
              <span className="percentile-badge badge-throughput">Throughput</span>
            </div>
            <div className="stat-value" style={{ color: 'var(--accent-1)' }}>
              {kpis.requests_per_min !== undefined ? `${kpis.requests_per_min} /min` : '—'}
            </div>
            <div className="stat-label">Request Rate</div>
            <p style={{ margin: '0.35rem 0 0 0', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Total: {kpis.total_requests ?? 0} requests recorded
            </p>
          </div>
        </div>

        {/* System Health Summary Bar */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: '1rem',
            marginBottom: '1rem',
          }}
        >
          <div className="card" style={{ padding: '0.875rem 1.25rem', display: 'flex', alignItems: 'center', gap: '1rem', margin: 0 }}>
            <div style={{ width: 34, height: 34, borderRadius: '50%', background: 'rgba(5, 150, 105, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <CheckCircle size={18} color="#059669" />
            </div>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 500 }}>SUCCESS RATE</div>
              <div style={{ fontSize: '1.125rem', fontWeight: 700, color: '#059669' }}>
                {kpis.success_rate ?? '100%'}
              </div>
            </div>
          </div>

          <div className="card" style={{ padding: '0.875rem 1.25rem', display: 'flex', alignItems: 'center', gap: '1rem', margin: 0 }}>
            <div style={{ width: 34, height: 34, borderRadius: '50%', background: 'rgba(220, 38, 38, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <AlertTriangle size={18} color="#dc2626" />
            </div>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 500 }}>ERROR RATE</div>
              <div style={{ fontSize: '1.125rem', fontWeight: 700, color: '#dc2626' }}>
                {kpis.error_rate ?? '0%'}
              </div>
            </div>
          </div>

          <div className="card" style={{ padding: '0.875rem 1.25rem', display: 'flex', alignItems: 'center', gap: '1rem', margin: 0 }}>
            <div style={{ width: 34, height: 34, borderRadius: '50%', background: 'rgba(67, 56, 202, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Users size={18} color="var(--accent-1)" />
            </div>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 500 }}>ACTIVE USERS ({timeWindow}h)</div>
              <div style={{ fontSize: '1.125rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                {kpis.active_users ?? 0} unique users
              </div>
            </div>
          </div>
        </div>

        {/* Collapsible Endpoint Latency Breakdown Table */}
        {metricsData?.by_endpoint && metricsData.by_endpoint.length > 0 && (
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <div
              className="card-header"
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
              onClick={() => setShowEndpointBreakdown(!showEndpointBreakdown)}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Activity size={18} color="var(--accent-1)" />
                <h3 className="card-title" style={{ margin: 0 }}>
                  Endpoint Latency Percentiles Breakdown ({metricsData.by_endpoint.length} Endpoints)
                </h3>
              </div>
              <button
                type="button"
                className="time-btn"
                style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', background: 'var(--bg-surface)' }}
              >
                <span>{showEndpointBreakdown ? 'Hide Breakdown' : 'View Breakdown'}</span>
                {showEndpointBreakdown ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
              </button>
            </div>

            {showEndpointBreakdown && (
              <div style={{ overflowX: 'auto', padding: '0 1.25rem 1.25rem 1.25rem' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Endpoint Route</th>
                      <th>Method</th>
                      <th>Requests</th>
                      <th>Avg Latency</th>
                      <th>P50 (Median)</th>
                      <th>P95</th>
                      <th>P99 (Tail)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metricsData.by_endpoint.map((ep, idx) => (
                      <tr key={`${ep.endpoint}-${ep.method}-${idx}`}>
                        <td style={{ fontFamily: 'monospace', fontWeight: 600, color: 'var(--text-primary)' }}>
                          {ep.endpoint}
                        </td>
                        <td>
                          <span
                            className="badge"
                            style={{
                              background:
                                ep.method === 'GET'
                                  ? 'rgba(5, 150, 105, 0.1)'
                                  : ep.method === 'POST'
                                  ? 'rgba(67, 56, 202, 0.1)'
                                  : ep.method === 'DELETE'
                                  ? 'rgba(220, 38, 38, 0.1)'
                                  : 'rgba(217, 119, 6, 0.1)',
                              color:
                                ep.method === 'GET'
                                  ? '#059669'
                                  : ep.method === 'POST'
                                  ? '#4338ca'
                                  : ep.method === 'DELETE'
                                  ? '#dc2626'
                                  : '#d97706',
                              fontWeight: 700,
                            }}
                          >
                            {ep.method}
                          </span>
                        </td>
                        <td style={{ fontWeight: 500 }}>{ep.request_count}</td>
                        <td style={{ color: 'var(--text-secondary)' }}>{ep.avg_ms} ms</td>
                        <td style={{ color: '#059669', fontWeight: 600 }}>{ep.p50_ms} ms</td>
                        <td style={{ color: '#d97706', fontWeight: 600 }}>{ep.p95_ms} ms</td>
                        <td style={{ color: '#dc2626', fontWeight: 600 }}>{ep.p99_ms} ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>

      {/* User Governance & Account Statistics */}
      <div className="metrics-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
          <Users size={20} color="var(--accent-1)" />
          <h2 style={{ fontSize: '1.125rem', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
            User Account Management
          </h2>
        </div>
      </div>

      {/* User Stats Grid */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'var(--color-info-bg)' }}>
            <Users size={20} color="var(--color-info)" />
          </div>
          <div className="stat-value">{stats.total}</div>
          <div className="stat-label">Total Users</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'var(--color-success-bg)' }}>
            <UserCheck size={20} color="var(--color-success)" />
          </div>
          <div className="stat-value">{stats.active}</div>
          <div className="stat-label">Active</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'var(--color-warning-bg)' }}>
            <Lock size={20} color="var(--color-warning)" />
          </div>
          <div className="stat-value">{stats.locked}</div>
          <div className="stat-label">Locked</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'rgba(139,92,246,0.12)' }}>
            <Clock size={20} color="var(--accent-3)" />
          </div>
          <div className="stat-value">{stats.pending}</div>
          <div className="stat-label">Pending</div>
        </div>
      </div>

      {/* Users table */}
      <div className="card">
        <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 className="card-title">All Registered Users ({filteredUsers.length})</h3>
        </div>

        {error && <div className="alert alert-error">{error}</div>}

        <div className="search-input-wrapper">
          <Search size={16} className="search-icon" />
          <input
            type="text"
            className="form-input"
            placeholder="Search by email, role, or status..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        {loading ? (
          <div className="loading-spinner" />
        ) : filteredUsers.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">
              <Users size={28} />
            </div>
            <h3>No users found</h3>
            <p>Try a different search term</p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Email</th>
                  <th>Role</th>
                  <th>Status</th>
                  <th>Joined</th>
                  <th>Status Control</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredUsers.map((user) => {
                  const isCurrentAdmin = currentAdmin && (currentAdmin.id === user.id || currentAdmin.email === user.email);
                  return (
                    <tr key={user.id}>
                      <td style={{ color: 'var(--text-muted)' }}>#{user.id}</td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <span style={{ fontWeight: 500 }}>{user.email}</span>
                          {isCurrentAdmin && (
                            <span
                              className="badge"
                              style={{
                                background: 'rgba(67, 56, 202, 0.1)',
                                color: 'var(--accent-1)',
                                fontSize: '0.7rem',
                                padding: '0.1rem 0.4rem',
                              }}
                            >
                              You
                            </span>
                          )}
                        </div>
                      </td>
                      <td>
                        <span className={`badge badge-${user.role.toLowerCase()}`}>
                          {user.role}
                        </span>
                      </td>
                      <td>
                        <span className={`badge badge-${user.status.toLowerCase()}`}>
                          {user.status}
                        </span>
                      </td>
                      <td style={{ color: 'var(--text-muted)' }}>
                        {user.created_at ? new Date(user.created_at).toLocaleDateString() : '—'}
                      </td>
                      <td>
                        <select
                          value={user.status}
                          onChange={(e) => updateStatus(user.id, e.target.value)}
                          className="form-input"
                          style={{ padding: '0.25rem 0.5rem', width: 'auto', fontSize: '0.8125rem' }}
                        >
                          <option value="ACTIVE">ACTIVE</option>
                          <option value="INACTIVE">INACTIVE</option>
                          <option value="LOCKED">LOCKED</option>
                          <option value="PENDING_VERIFICATION">PENDING</option>
                        </select>
                      </td>
                      <td>
                        <button
                          type="button"
                          className="btn-delete"
                          disabled={isCurrentAdmin}
                          onClick={() => setUserToDelete(user)}
                          title={isCurrentAdmin ? 'You cannot delete your own admin account' : `Delete user ${user.email} permanently`}
                        >
                          <Trash2 size={14} />
                          <span>Delete</span>
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Permanent Delete Confirmation Modal */}
      {userToDelete && (
        <div className="admin-modal-backdrop" onClick={() => !isDeleting && setUserToDelete(null)}>
          <div className="admin-modal-box" onClick={(e) => e.stopPropagation()}>
            <div className="admin-modal-header">
              <div className="admin-modal-icon-danger">
                <AlertTriangle size={20} />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '1.0625rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Permanently Delete User?
                </h3>
                <span style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                  This action is permanent and irreversible
                </span>
              </div>
            </div>

            <div className="admin-modal-body">
              <p style={{ marginTop: 0 }}>
                Are you sure you want to permanently delete the following user from the database?
              </p>

              <div
                style={{
                  background: 'var(--bg-surface)',
                  padding: '0.75rem 1rem',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-default)',
                  marginBottom: '1rem',
                  fontSize: '0.875rem',
                }}
              >
                <div><strong>Email:</strong> {userToDelete.email}</div>
                <div><strong>User ID:</strong> #{userToDelete.id}</div>
                <div><strong>Role:</strong> {userToDelete.role}</div>
                <div><strong>Current Status:</strong> {userToDelete.status}</div>
              </div>

              <div
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '0.5rem',
                  padding: '0.625rem 0.75rem',
                  background: 'rgba(220, 38, 38, 0.08)',
                  borderRadius: 'var(--radius-md)',
                  color: '#dc2626',
                  fontSize: '0.8125rem',
                }}
              >
                <Shield size={16} style={{ flexShrink: 0, marginTop: '2px' }} />
                <span>
                  <strong>Database Cleanup Notice:</strong> Deleting this account will immediately purge all stored resume vector chunks, chat history, semantic memories, and API metric references.
                </span>
              </div>
            </div>

            <div className="admin-modal-actions">
              <button
                type="button"
                className="admin-btn-cancel"
                onClick={() => setUserToDelete(null)}
                disabled={isDeleting}
              >
                Cancel
              </button>
              <button
                type="button"
                className="admin-btn-confirm-delete"
                onClick={confirmDeleteUser}
                disabled={isDeleting}
              >
                {isDeleting ? (
                  <>
                    <RefreshCw size={15} className="spinner" />
                    <span>Deleting User...</span>
                  </>
                ) : (
                  <>
                    <Trash2 size={15} />
                    <span>Delete User Permanently</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

