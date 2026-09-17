import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'https://campusquery-intelligent-rag-system-1.onrender.com';

const api = axios.create({
  baseURL: API_URL,
  withCredentials: true,
});

// Flag to prevent multiple refresh calls simultaneously
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

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        return new Promise(function(resolve, reject) {
          failedQueue.push({ resolve, reject });
        }).then(() => {
          return api(originalRequest);
        }).catch((err) => {
          return Promise.reject(err);
        });
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        await axios.post(`${API_URL}/refresh`, {}, { withCredentials: true });
        isRefreshing = false;
        processQueue(null);
        return api(originalRequest);
      } catch (err) {
        isRefreshing = false;
        processQueue(err, null);
        // Do not redirect if we are already on an auth page
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
);

/**
 * Extract a user-friendly error message from an Axios error.
 */
export function getErrorMessage(error) {
  if (error.response?.data?.detail) {
    return typeof error.response.data.detail === 'string'
      ? error.response.data.detail
      : JSON.stringify(error.response.data.detail);
  }
  if (error.message) return error.message;
  return 'Something went wrong';
}

export default api;
