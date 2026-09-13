import { useState, useEffect } from 'react';
import { useNavigate, useLocation, Link } from 'react-router-dom';
import { Lock, ShieldCheck, Zap, Eye, EyeOff } from 'lucide-react';
import api, { getErrorMessage } from '../api/api';
import { useToast } from '../context/ToastContext';

export default function ResetPassword() {
  const location = useLocation();
  const navigate = useNavigate();
  const email = location.state?.email;
  const toast = useToast();

  const [otp, setOtp] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!email) {
      navigate('/login');
    }
  }, [email, navigate]);

  if (!email) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (newPassword !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }
    if (newPassword.length < 6) {
      setError('Password must be at least 6 characters.');
      return;
    }
    setLoading(true);
    try {
      await api.post('/reset-password', { email, otp, new_password: newPassword });
      toast.success('Password reset successfully!');
      navigate('/login');
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-brand">
          <div className="auth-brand-icon">
            <Zap size={24} color="white" />
          </div>
          <h1>CampusQuery</h1>
        </div>

        <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
          <div style={{
            width: '48px', height: '48px', borderRadius: 'var(--radius-md)',
            background: 'var(--color-success-bg)', display: 'inline-flex',
            alignItems: 'center', justifyContent: 'center', marginBottom: '0.75rem'
          }}>
            <ShieldCheck size={24} color="var(--color-success)" />
          </div>
          <h2 className="auth-title">Create new password</h2>
          <p className="auth-subtitle">Enter the OTP code and your new password</p>
        </div>

        {error && <div className="alert alert-error">{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label" htmlFor="reset-otp">OTP Code</label>
            <div className="form-input-wrapper">
              <ShieldCheck size={16} className="form-input-icon" />
              <input
                id="reset-otp"
                type="text"
                className="form-input"
                placeholder="Enter 6-digit code"
                value={otp}
                onChange={(e) => setOtp(e.target.value)}
                required
                autoComplete="one-time-code"
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="reset-password">New Password</label>
            <div className="form-input-wrapper" style={{ position: 'relative' }}>
              <Lock size={16} className="form-input-icon" />
              <input
                id="reset-password"
                type={showPassword ? 'text' : 'password'}
                className="form-input"
                placeholder="Min 6 characters"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
                minLength="6"
                autoComplete="new-password"
                style={{ paddingRight: '2.5rem' }}
              />
              <button
                type="button"
                className="password-toggle-btn"
                onClick={() => setShowPassword(!showPassword)}
                title={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="reset-confirm">Confirm Password</label>
            <div className="form-input-wrapper" style={{ position: 'relative' }}>
              <Lock size={16} className="form-input-icon" />
              <input
                id="reset-confirm"
                type={showConfirmPassword ? 'text' : 'password'}
                className="form-input"
                placeholder="Re-enter new password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                minLength="6"
                autoComplete="new-password"
                style={{ paddingRight: '2.5rem' }}
              />
              <button
                type="button"
                className="password-toggle-btn"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                title={showConfirmPassword ? "Hide password" : "Show password"}
              >
                {showConfirmPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? <><span className="spinner" /> Resetting...</> : 'Reset Password'}
          </button>
        </form>

        <div className="auth-links">
          <Link to="/login" className="auth-link">
            ← Back to <span>Sign in</span>
          </Link>
        </div>
      </div>
    </div>
  );
}
