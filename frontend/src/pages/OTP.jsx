import { useState, useRef, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { ShieldCheck, Zap } from 'lucide-react';
import api, { getErrorMessage } from '../api/api';
import { useToast } from '../context/ToastContext';

export default function OTP() {
  const location = useLocation();
  const navigate = useNavigate();
  const email = location.state?.email;
  const toast = useToast();

  const [otp, setOtp] = useState(['', '', '', '', '', '']);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const inputRefs = useRef([]);

  useEffect(() => {
    if (!email) {
      navigate('/login');
    }
  }, [email, navigate]);

  if (!email) return null;

  const handleChange = (index, value) => {
    if (!/^\d*$/.test(value)) return;
    const newOtp = [...otp];
    newOtp[index] = value.slice(-1);
    setOtp(newOtp);
    if (value && index < 5) {
      inputRefs.current[index + 1]?.focus();
    }
  };

  const handleKeyDown = (index, e) => {
    if (e.key === 'Backspace' && !otp[index] && index > 0) {
      inputRefs.current[index - 1]?.focus();
    }
  };

  const handlePaste = (e) => {
    e.preventDefault();
    const pasted = e.clipboardData.getData('text').replace(/\D/g, '').slice(0, 6);
    if (pasted.length > 0) {
      const newOtp = [...otp];
      for (let i = 0; i < 6; i++) {
        newOtp[i] = pasted[i] || '';
      }
      setOtp(newOtp);
      const lastFilledIndex = Math.min(pasted.length, 5);
      inputRefs.current[lastFilledIndex]?.focus();
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const otpValue = otp.join('');
    if (otpValue.length !== 6) {
      setError('Please enter the full 6-digit code.');
      return;
    }
    setError('');
    setLoading(true);
    try {
      await api.post('/verify-otp', { email, otp: otpValue });
      toast.success('Email verified successfully!');
      navigate('/login');
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const maskedEmail = email.replace(/(.{2})(.*)(@.*)/, '$1***$3');

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
            background: 'var(--color-info-bg)', display: 'inline-flex',
            alignItems: 'center', justifyContent: 'center', marginBottom: '0.75rem'
          }}>
            <ShieldCheck size={24} color="var(--color-info)" />
          </div>
          <h2 className="auth-title">Verify your email</h2>
          <p className="auth-subtitle">
            Enter the 6-digit code sent to<br />
            <strong style={{ color: 'var(--text-primary)' }}>{maskedEmail}</strong>
          </p>
        </div>

        {error && <div className="alert alert-error">{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className="otp-inputs" onPaste={handlePaste}>
            {otp.map((digit, i) => (
              <input
                key={i}
                ref={(el) => (inputRefs.current[i] = el)}
                type="text"
                inputMode="numeric"
                maxLength="1"
                className={`otp-input ${digit ? 'filled' : ''}`}
                value={digit}
                onChange={(e) => handleChange(i, e.target.value)}
                onKeyDown={(e) => handleKeyDown(i, e)}
                autoFocus={i === 0}
                aria-label={`Digit ${i + 1}`}
              />
            ))}
          </div>
          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? <><span className="spinner" /> Verifying...</> : 'Verify Email'}
          </button>
        </form>
      </div>
    </div>
  );
}
