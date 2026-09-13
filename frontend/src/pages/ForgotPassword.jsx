import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Mail, Zap } from 'lucide-react';
import api, { getErrorMessage } from '../api/api';
import { useToast } from '../context/ToastContext';

export default function ForgotPassword() {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const toast = useToast();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await api.post('/forgot-password', { email });
      toast.info('If the email is registered, an OTP was sent.');
      navigate('/reset-password', { state: { email } });
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

        <h2 className="auth-title">Forgot password?</h2>
        <p className="auth-subtitle">Enter your email and we'll send you a reset code</p>

        {error && <div className="alert alert-error">{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label" htmlFor="forgot-email">Email</label>
            <div className="form-input-wrapper">
              <Mail size={16} className="form-input-icon" />
              <input
                id="forgot-email"
                type="email"
                className="form-input"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoComplete="email"
              />
            </div>
          </div>
          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? <><span className="spinner" /> Sending...</> : 'Send Reset Code'}
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
