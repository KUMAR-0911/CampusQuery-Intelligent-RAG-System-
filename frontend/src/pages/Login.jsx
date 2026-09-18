import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Mail, Lock, Zap, Eye, EyeOff } from 'lucide-react';
import api, { getErrorMessage, setTokens } from '../api/api';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../context/ToastContext';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { login } = useAuth();
  const toast = useToast();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('username', email);
      formData.append('password', password);

      const res = await api.post('/login', formData);
      if (res.data?.access_token) {
        setTokens(res.data);
      }
      
      const userRes = await api.get('/me');

      login(userRes.data);
      toast.success('Welcome back!');
      navigate('/');
    } catch (err) {
      const detail = err.response?.data?.detail || '';
      if (err.response?.status === 403) {
        if (detail.includes('PENDING_VERIFICATION')) {
          toast.info('Please verify your email first.');
          navigate('/verify-otp', { state: { email } });
          return;
        }
        if (detail.includes('LOCKED')) {
          setError('Your account has been temporarily locked due to too many failed attempts. Please contact support or try again later.');
          return;
        }
        if (detail.includes('INACTIVE')) {
          setError('Your account is inactive. Please contact support.');
          return;
        }
      }
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
          <p>Student & Placement RAG System</p>
        </div>

        <h2 className="auth-title">Welcome back</h2>
        <p className="auth-subtitle">Sign in to your account</p>

        {error && <div className="alert alert-error">{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label" htmlFor="login-email">Email Address</label>
            <div className="form-input-wrapper">
              <Mail size={16} className="form-input-icon" />
              <input
                id="login-email"
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
            <label className="form-label" htmlFor="login-password">Password</label>
            <div className="form-input-wrapper" style={{ position: 'relative' }}>
              <Lock size={16} className="form-input-icon" />
              <input
                id="login-password"
                type={showPassword ? 'text' : 'password'}
                className="form-input"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="current-password"
                style={{ paddingRight: '2.5rem' }}
              />
              <button
                type="button"
                className="password-toggle-btn"
                onClick={() => setShowPassword(!showPassword)}
                title={showPassword ? "Hide password" : "Show password"}
                aria-label="Toggle password visibility"
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? <><span className="spinner" /> Signing in...</> : 'Sign In'}
          </button>
        </form>

        <div className="auth-links">
          <Link to="/register" className="auth-link">
            Don't have an account? <span>Sign up</span>
          </Link>
          <Link to="/forgot-password" className="auth-link">
            Forgot your password?
          </Link>
        </div>
      </div>
    </div>
  );
}
