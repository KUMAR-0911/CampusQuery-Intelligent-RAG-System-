/**
 * Native Fetch API client with credentials, Bearer token authorization, automatic token refresh, and standardized response.
 */
export const API_URL = import.meta.env.VITE_API_URL || 'https://resumetool.onrender.com';

export const getAccessToken = () => localStorage.getItem('access_token');
export const getRefreshToken = () => localStorage.getItem('refresh_token');

export const setTokens = (tokens) => {
  if (tokens?.access_token) {
    localStorage.setItem('access_token', tokens.access_token);
  }
  if (tokens?.refresh_token) {
    localStorage.setItem('refresh_token', tokens.refresh_token);
  }
};

export const clearTokens = () => {
  localStorage.removeItem('access_token');
  localStorage.removeItem('refresh_token');
};

let isRefreshing = false;
let failedQueue = [];

const processQueue = (error, token = null) => {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token);
    }
  });
  failedQueue = [];
};

async function executeFetch(endpoint, options = {}) {
  const base = (API_URL || '').replace(/\/+$/, '');
  const cleanPath = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  const url = endpoint.startsWith('http') ? endpoint : `${base}${cleanPath}`;

  const headers = { ...options.headers };

  // Attach Bearer token if present
  const accessToken = getAccessToken();
  if (accessToken && !headers['Authorization']) {
    headers['Authorization'] = `Bearer ${accessToken}`;
  }

  // If sending JSON body, set Content-Type unless it's FormData
  if (options.body && !(options.body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
    if (typeof options.body === 'object') {
      options.body = JSON.stringify(options.body);
    }
  }

  const fetchOptions = {
    ...options,
    headers,
    credentials: options.credentials || 'include',
  };

  const response = await fetch(url, fetchOptions);

  let data = null;
  const contentType = response.headers.get('content-type');
  if (contentType && contentType.includes('application/json')) {
    try {
      data = await response.json();
    } catch {
      data = null;
    }
  } else {
    try {
      data = await response.text();
    } catch {
      data = null;
    }
  }

  if (!response.ok) {
    const error = new Error(`Request failed with status ${response.status}`);
    error.response = {
      status: response.status,
      statusText: response.statusText,
      data,
    };
    error.config = { endpoint, options };

    // 401 Unauthorized handling for automatic token refresh (skip if calling /login or /refresh itself)
    const isAuthEndpoint = endpoint.includes('/login') || endpoint.includes('/refresh') || endpoint.includes('/register');
    if (response.status === 401 && !options._retry && !isAuthEndpoint) {
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        }).then(() => executeFetch(endpoint, { ...options, _retry: true }));
      }

      options._retry = true;
      isRefreshing = true;

      try {
        const refreshTok = getRefreshToken();
        const refreshHeaders = { 'Content-Type': 'application/json' };
        if (refreshTok) {
          refreshHeaders['Authorization'] = `Bearer ${refreshTok}`;
        }

        const refreshRes = await fetch(`${base}/refresh`, {
          method: 'POST',
          headers: refreshHeaders,
          body: JSON.stringify({ refresh_token: refreshTok }),
          credentials: 'include',
        });

        if (!refreshRes.ok) {
          throw new Error('Refresh token invalid');
        }

        const refreshData = await refreshRes.json();
        if (refreshData?.access_token) {
          setTokens(refreshData);
        }

        isRefreshing = false;
        processQueue(null, refreshData?.access_token);
        return executeFetch(endpoint, options);
      } catch (refreshErr) {
        isRefreshing = false;
        processQueue(refreshErr, null);
        clearTokens();
        if (
          !window.location.pathname.startsWith('/login') &&
          !window.location.pathname.startsWith('/register') &&
          !window.location.pathname.startsWith('/verify-otp') &&
          !window.location.pathname.startsWith('/forgot-password') &&
          !window.location.pathname.startsWith('/reset-password')
        ) {
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    }

    return Promise.reject(error);
  }

  return {
    data,
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  };
}

export const api = {
  get(endpoint, options = {}) {
    return executeFetch(endpoint, { ...options, method: 'GET' });
  },
  post(endpoint, body, options = {}) {
    return executeFetch(endpoint, { ...options, method: 'POST', body });
  },
  put(endpoint, body, options = {}) {
    return executeFetch(endpoint, { ...options, method: 'PUT', body });
  },
  delete(endpoint, options = {}) {
    return executeFetch(endpoint, { ...options, method: 'DELETE' });
  },
};

/**
 * Extract a user-friendly error message from a fetch error.
 */
export function getErrorMessage(error) {
  if (error?.response?.data?.detail) {
    return typeof error.response.data.detail === 'string'
      ? error.response.data.detail
      : JSON.stringify(error.response.data.detail);
  }
  if (error?.message) return error.message;
  return 'Something went wrong';
}

export default api;
