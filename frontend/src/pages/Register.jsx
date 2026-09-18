import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Mail, Lock, Zap, User, Globe, Eye, EyeOff } from 'lucide-react';
import api, { getErrorMessage } from '../api/api';
import { useToast } from '../context/ToastContext';

export default function Register() {
  const [name, setName] = useState('');
  const [nationality, setNationality] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const toast = useToast();

  const getPasswordStrength = (pw) => {
    if (!pw) return { label: '', width: '0%', color: 'transparent' };
    let score = 0;
    if (pw.length >= 6) score++;
    if (pw.length >= 10) score++;
    if (/[A-Z]/.test(pw)) score++;
    if (/[0-9]/.test(pw)) score++;
    if (/[^A-Za-z0-9]/.test(pw)) score++;
    if (score <= 1) return { label: 'Weak', width: '20%', color: 'var(--color-danger)' };
    if (score <= 2) return { label: 'Fair', width: '40%', color: 'var(--color-warning)' };
    if (score <= 3) return { label: 'Good', width: '60%', color: 'var(--color-info)' };
    if (score <= 4) return { label: 'Strong', width: '80%', color: 'var(--color-success)' };
    return { label: 'Very strong', width: '100%', color: 'var(--color-success)' };
  };

  const strength = getPasswordStrength(password);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters.');
      return;
    }
    setLoading(true);
    try {
      const res = await api.post('/register', {
        name: name.trim() || undefined,
        nationality: nationality.trim() || undefined,
        email,
        password
      });
      toast.success('Account created! Please verify your email.');
      navigate('/verify-otp', { state: { email, otp_debug: res.data?.otp_debug } });
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
          <h1>Resume & Career Analyzer Toolkit</h1>
          <p>Student & Job Seeker AI Assistant</p>
        </div>

        <h2 className="auth-title">Create Account</h2>
        <p className="auth-subtitle">Sign up to analyze resumes and documents</p>

        {error && <div className="alert alert-error">{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label" htmlFor="register-name">Full Name <span style={{ opacity: 0.6, fontWeight: 400 }}>(Optional)</span></label>
            <div className="form-input-wrapper">
              <User size={16} className="form-input-icon" />
              <input
                id="register-name"
                type="text"
                className="form-input"
                placeholder="John Doe"
                value={name}
                onChange={(e) => setName(e.target.value)}
                autoComplete="name"
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="register-nationality">Country / Nationality <span style={{ opacity: 0.6, fontWeight: 400 }}>(Optional)</span></label>
            <div className="form-input-wrapper">
              <Globe size={16} className="form-input-icon" />
              <input
                id="register-nationality"
                type="text"
                className="form-input"
                placeholder="e.g. United States, India, Canada"
                value={nationality}
                onChange={(e) => setNationality(e.target.value)}
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="register-email">Email Address</label>
            <div className="form-input-wrapper">
              <Mail size={16} className="form-input-icon" />
              <input
                id="register-email"
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

          <div className="form-group">
            <label className="form-label" htmlFor="register-password">Password</label>
            <div className="form-input-wrapper" style={{ position: 'relative' }}>
              <Lock size={16} className="form-input-icon" />
              <input
                id="register-password"
                type={showPassword ? 'text' : 'password'}
                className="form-input"
                placeholder="Min 6 characters"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
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
            {password && (
              <div style={{ marginTop: '0.5rem' }}>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: strength.width, background: strength.color }} />
                </div>
                <div style={{ fontSize: '0.75rem', color: strength.color, marginTop: '0.25rem' }}>
                  {strength.label}
                </div>
              </div>
            )}
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="register-confirm">Confirm Password</label>
            <div className="form-input-wrapper" style={{ position: 'relative' }}>
              <Lock size={16} className="form-input-icon" />
              <input
                id="register-confirm"
                type={showConfirmPassword ? 'text' : 'password'}
                className="form-input"
                placeholder="Re-enter your password"
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
            {loading ? <><span className="spinner" /> Creating account...</> : 'Create Account'}
          </button>
        </form>

        <div className="auth-links">
          <Link to="/login" className="auth-link">
            Already have an account? <span>Sign in</span>
          </Link>
        </div>
      </div>
    </div>
  );
}
