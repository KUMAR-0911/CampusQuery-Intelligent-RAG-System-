import { useState, useEffect, useMemo } from 'react';
import { Users, UserCheck, Lock, Clock, Search } from 'lucide-react';
import api, { getErrorMessage } from '../api/api';
import { useToast } from '../context/ToastContext';

export default function AdminDashboard() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const toast = useToast();

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    try {
      const res = await api.get('/users');
      setUsers(res.data);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const updateStatus = async (userId, newStatus) => {
    try {
      await api.put(`/users/${userId}/status?status=${newStatus}`);
      toast.success(`User status updated to ${newStatus}`);
      fetchUsers();
    } catch (err) {
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

  const stats = useMemo(() => ({
    total: users.length,
    active: users.filter((u) => u.status === 'ACTIVE').length,
    locked: users.filter((u) => u.status === 'LOCKED').length,
    pending: users.filter((u) => u.status === 'PENDING_VERIFICATION').length,
  }), [users]);

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">User Management</h1>
        <p className="page-subtitle">Manage user accounts and permissions</p>
      </div>

      {/* Stats */}
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
        <div className="card-header">
          <h3 className="card-title">All Users</h3>
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
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredUsers.map((user) => (
                  <tr key={user.id}>
                    <td style={{ color: 'var(--text-muted)' }}>#{user.id}</td>
                    <td>{user.email}</td>
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
                      {new Date(user.created_at).toLocaleDateString()}
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
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
