import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { Mail, User, Globe, Edit2, Check, Lock, Eye, EyeOff, Save, Trash2, ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';
import api, { getErrorMessage } from '../api/api';
import { useToast } from '../context/ToastContext';

const WORLD_COUNTRIES = [
  "United States", "United Kingdom", "Canada", "Australia", "India", "Germany",
  "France", "Japan", "China", "Brazil", "Mexico", "Spain", "Italy", "Singapore",
  "South Korea", "United Arab Emirates", "Saudi Arabia", "Netherlands", "Sweden",
  "Switzerland", "Norway", "Denmark", "Finland", "Belgium", "Austria", "Ireland",
  "New Zealand", "South Africa", "Nigeria", "Kenya", "Egypt", "Argentina", "Chile",
  "Colombia", "Peru", "Pakistan", "Bangladesh", "Sri Lanka", "Indonesia", "Malaysia",
  "Thailand", "Vietnam", "Philippines", "Turkey", "Poland", "Czech Republic", "Greece",
  "Portugal", "Romania", "Ukraine", "Israel", "Qatar", "Kuwait", "Oman", "Bahrain"
];

export default function Profile() {
  const { user, checkAuth } = useAuth();
  const toast = useToast();

  const [name, setName] = useState(user?.name || '');
  const [nationality, setNationality] = useState(user?.nationality || 'Not specified');
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [savingProfile, setSavingProfile] = useState(false);

  // Password fields state with visibility toggles
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [updatingPassword, setUpdatingPassword] = useState(false);

  const displayName = user?.name || user?.email?.split('@')[0] || 'User';
  const userInitial = displayName[0]?.toUpperCase() || 'U';

  const handleSaveProfile = async () => {
    if (!name.trim()) {
      toast.error('Name cannot be empty.');
      return;
    }
    setSavingProfile(true);
    try {
      const res = await api.post('/update-profile', {
        name: name.trim(),
        nationality: nationality,
      });
      toast.success(res.data.message || 'Profile updated successfully!');
      await checkAuth();
      setIsEditingProfile(false);
    } catch (err) {
      toast.error(`Could not update profile: ${getErrorMessage(err)}`);
    } finally {
      setSavingProfile(false);
    }
  };

  const handleChangePassword = async (e) => {
    e.preventDefault();
    if (!currentPassword) {
      toast.error('Please enter your current password.');
      return;
    }
    if (!newPassword) {
      toast.error('Please enter a new password.');
      return;
    }
    if (newPassword.length < 6) {
      toast.error('New password must be at least 6 characters long.');
      return;
    }
    if (newPassword !== confirmPassword) {
      toast.error('New password and confirm password do not match.');
      return;
    }
    setUpdatingPassword(true);
    try {
      const res = await api.post('/change-password', {
        current_password: currentPassword,
        new_password: newPassword,
      });
      toast.success(res.data.message || 'Password updated successfully!');
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (err) {
      toast.error(`Failed to update password: ${getErrorMessage(err)}`);
    } finally {
      setUpdatingPassword(false);
    }
  };

  const handleClearPasswordInputs = () => {
    setCurrentPassword('');
    setNewPassword('');
    setConfirmPassword('');
    toast.info('Password inputs cleared.');
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.5rem', paddingBottom: '2rem' }}>
      
      {/* Top Header Navigation to easily return to Chat */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Link
          to="/chat"
          className="btn btn-secondary"
          style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}
        >
          <ArrowLeft size={16} /> Back to Chat Session
        </Link>
      </div>

      <div className="page-header" style={{ marginBottom: '0.5rem' }}>
        <h1 className="page-title">User Profile & Security</h1>
        <p className="page-subtitle">Manage your personal account details, country, and security settings</p>
      </div>

      {/* User Header Section - ROLE REMOVED */}
      <div className="profile-header">
        <div className="profile-avatar">{userInitial}</div>
        <div className="profile-info">
          <h2>{displayName}</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginTop: '0.125rem' }}>{user?.email}</p>
          <p style={{ color: 'var(--text-accent)', fontSize: '0.8125rem', marginTop: '0.375rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Globe size={14} /> {user?.nationality || 'Country not specified'}
          </p>
        </div>
      </div>

      {/* Account Details Card with Name & Country Update */}
      <div className="card">
        <div className="card-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h3 className="card-title">Account & Personal Details</h3>
          {!isEditingProfile && (
            <button
              className="btn btn-ghost"
              onClick={() => {
                setName(user?.name || '');
                setNationality(user?.nationality || 'United States');
                setIsEditingProfile(true);
              }}
              style={{ fontSize: '0.8125rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}
            >
              <Edit2 size={15} /> Edit Details
            </button>
          )}
        </div>

        {isEditingProfile ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginTop: '0.5rem' }}>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Full Name</label>
              <div className="form-input-wrapper">
                <User size={16} className="form-input-icon" />
                <input
                  type="text"
                  className="form-input"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Enter your full name"
                  autoFocus
                />
              </div>
            </div>

            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Country / Nationality</label>
              <div className="form-input-wrapper">
                <Globe size={16} className="form-input-icon" />
                <select
                  className="form-input"
                  value={nationality}
                  onChange={(e) => setNationality(e.target.value)}
                  style={{ background: 'var(--bg-primary)', cursor: 'pointer' }}
                >
                  <option value="Not specified">-- Select Country --</option>
                  {WORLD_COUNTRIES.map((country, idx) => (
                    <option key={idx} value={country}>{country}</option>
                  ))}
                </select>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
              <button
                className="btn btn-primary"
                onClick={handleSaveProfile}
                disabled={savingProfile}
                style={{ width: 'auto', display: 'flex', alignItems: 'center', gap: '0.35rem' }}
              >
                <Check size={16} /> Save Changes
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => {
                  setIsEditingProfile(false);
                  setName(user?.name || '');
                  setNationality(user?.nationality || 'Not specified');
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <div className="profile-grid">
            <div className="profile-field">
              <div className="profile-field-label">
                <User size={14} style={{ display: 'inline', marginRight: '0.375rem', verticalAlign: 'middle' }} />
                Full Name
              </div>
              <div className="profile-field-value">{user?.name || 'Not provided'}</div>
            </div>

            <div className="profile-field">
              <div className="profile-field-label">
                <Globe size={14} style={{ display: 'inline', marginRight: '0.375rem', verticalAlign: 'middle' }} />
                Country / Nationality
              </div>
              <div className="profile-field-value">{user?.nationality || 'Not specified'}</div>
            </div>

            <div className="profile-field" style={{ gridColumn: 'span 2' }}>
              <div className="profile-field-label">
                <Mail size={14} style={{ display: 'inline', marginRight: '0.375rem', verticalAlign: 'middle' }} />
                Email Address
              </div>
              <div className="profile-field-value">{user?.email}</div>
            </div>
          </div>
        )}
      </div>

      {/* Password Update Card */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Lock size={18} className="text-accent" /> Password & Security
          </h3>
        </div>

        <form onSubmit={handleChangePassword} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          
          <div className="form-group">
            <label className="form-label">Current Password</label>
            <div className="form-input-wrapper" style={{ position: 'relative' }}>
              <Lock size={16} className="form-input-icon" />
              <input
                type={showCurrentPassword ? 'text' : 'password'}
                className="form-input"
                placeholder="Enter current password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                style={{ paddingRight: '2.5rem' }}
              />
              <button
                type="button"
                className="password-toggle-btn"
                onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                title={showCurrentPassword ? 'Hide password' : 'Show password'}
              >
                {showCurrentPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">New Password</label>
            <div className="form-input-wrapper" style={{ position: 'relative' }}>
              <Lock size={16} className="form-input-icon" />
              <input
                type={showNewPassword ? 'text' : 'password'}
                className="form-input"
                placeholder="Enter new password (min 6 chars)"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                style={{ paddingRight: '2.5rem' }}
              />
              <button
                type="button"
                className="password-toggle-btn"
                onClick={() => setShowNewPassword(!showNewPassword)}
                title={showNewPassword ? 'Hide password' : 'Show password'}
              >
                {showNewPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Confirm New Password</label>
            <div className="form-input-wrapper" style={{ position: 'relative' }}>
              <Lock size={16} className="form-input-icon" />
              <input
                type={showConfirmPassword ? 'text' : 'password'}
                className="form-input"
                placeholder="Confirm new password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                style={{ paddingRight: '2.5rem' }}
              />
              <button
                type="button"
                className="password-toggle-btn"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                title={showConfirmPassword ? 'Hide password' : 'Show password'}
              >
                {showConfirmPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', marginTop: '0.5rem' }}>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={updatingPassword || !currentPassword || !newPassword}
              style={{ width: 'auto', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <Save size={16} /> Update Password
            </button>
            
            {(currentPassword || newPassword || confirmPassword) && (
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleClearPasswordInputs}
                style={{ width: 'auto', display: 'flex', alignItems: 'center', gap: '0.35rem' }}
              >
                <Trash2 size={15} /> Clear Fields
              </button>
            )}
          </div>
        </form>
      </div>

    </div>
  );
}
