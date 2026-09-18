import { createContext, useState, useEffect, useContext } from 'react';
import api, { clearTokens, getAccessToken, getRefreshToken } from '../api/api';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  // Pre-load cached user profile for instant 0ms rendering
  const [user, setUser] = useState(() => {
    try {
      const saved = localStorage.getItem('user_profile');
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });

  // Only show loading if there are tokens to validate and no cached user
  const [loading, setLoading] = useState(() => {
    return Boolean((getAccessToken() || getRefreshToken()) && !localStorage.getItem('user_profile'));
  });

  useEffect(() => {
    // If no tokens exist, user is definitely logged out; skip network request entirely
    if (!getAccessToken() && !getRefreshToken()) {
      setUser(null);
      localStorage.removeItem('user_profile');
      setLoading(false);
      return;
    }

    // Validate and refresh authenticated user in background
    let isMounted = true;
    const fetchUser = async () => {
      try {
        const res = await api.get('/me');
        if (isMounted && res.data) {
          setUser(res.data);
          localStorage.setItem('user_profile', JSON.stringify(res.data));
        }
      } catch {
        if (isMounted) {
          setUser(null);
          localStorage.removeItem('user_profile');
        }
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    fetchUser();
    return () => { isMounted = false; };
  }, []);

  const login = (userData) => {
    setUser(userData);
    if (userData) {
      localStorage.setItem('user_profile', JSON.stringify(userData));
    }
  };

  const logout = async () => {
    try {
      await api.post('/logout');
    } catch (err) {
      console.warn('Logout notice:', err);
    } finally {
      clearTokens();
      localStorage.removeItem('user_profile');
      setUser(null);
    }
  };

  const checkAuth = async () => {
    if (!getAccessToken() && !getRefreshToken()) {
      setUser(null);
      localStorage.removeItem('user_profile');
      return;
    }
    try {
      const res = await api.get('/me');
      setUser(res.data);
      if (res.data) {
        localStorage.setItem('user_profile', JSON.stringify(res.data));
      }
    } catch {
      setUser(null);
      localStorage.removeItem('user_profile');
    }
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, logout, checkAuth, setUser }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
