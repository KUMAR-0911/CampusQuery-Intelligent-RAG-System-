import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ToastProvider } from './context/ToastContext';

import Login from './pages/Login';
import Register from './pages/Register';
import OTP from './pages/OTP';
import ForgotPassword from './pages/ForgotPassword';
import ResetPassword from './pages/ResetPassword';

// Lazy-load dashboard and RAG components so Login loads zero unrelated functionality
const DashboardLayout = lazy(() => import('./pages/DashboardLayout'));
const Profile = lazy(() => import('./pages/Profile'));
const RagChat = lazy(() => import('./pages/RagChat'));
const AdminDashboard = lazy(() => import('./pages/AdminDashboard'));

const LoadingFallback = () => (
  <div className="loading-page">
    <div className="loading-spinner" />
    <span>Loading...</span>
  </div>
);

const ProtectedRoute = ({ children, requireAdmin = false }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return <LoadingFallback />;
  }
  if (!user) return <Navigate to="/login" replace />;
  if (requireAdmin && user.role !== 'ADMIN') return <Navigate to="/" replace />;

  return children;
};

export default function App() {
  return (
    <ToastProvider>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            {/* Auth pages */}
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/verify-otp" element={<OTP />} />
            <Route path="/forgot-password" element={<ForgotPassword />} />
            <Route path="/reset-password" element={<ResetPassword />} />

            {/* Protected dashboard / layout (lazy-loaded) */}
            <Route
              path="/"
              element={
                <ProtectedRoute>
                  <Suspense fallback={<LoadingFallback />}>
                    <DashboardLayout />
                  </Suspense>
                </ProtectedRoute>
              }
            >
              <Route index element={<Navigate to="/chat" replace />} />
              <Route
                path="chat"
                element={
                  <Suspense fallback={<LoadingFallback />}>
                    <RagChat />
                  </Suspense>
                }
              />
              <Route
                path="profile"
                element={
                  <Suspense fallback={<LoadingFallback />}>
                    <Profile />
                  </Suspense>
                }
              />
              <Route
                path="admin"
                element={
                  <ProtectedRoute requireAdmin={true}>
                    <Suspense fallback={<LoadingFallback />}>
                      <AdminDashboard />
                    </Suspense>
                  </ProtectedRoute>
                }
              />
            </Route>

          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </ToastProvider>
  );
}
