export interface User {
  id: number;
  email: string;
  name?: string;
  nationality?: string;
  role: 'USER' | 'ADMIN';
  status: 'ACTIVE' | 'INACTIVE' | 'LOCKED' | 'PENDING_VERIFICATION';
}

export interface AuthState {
  user: User | null;
  loading: boolean;
}

export interface ChatMessage {
  id?: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
  citations?: string[];
  retrieved_chunks?: Array<{
    page_content: string;
    metadata: Record<string, any>;
    score?: number;
  }>;
}

export interface AdminKPIMetrics {
  total_requests: number;
  requests_per_min: number;
  success_rate: string;
  error_rate: string;
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  active_users: number;
}

export interface AdminMetricsResponse {
  time_window_hours: number;
  kpi_metrics: AdminKPIMetrics;
  global_summary: Record<string, any>;
  by_endpoint: Array<{
    endpoint: string;
    method: string;
    request_count: number;
    avg_ms: number;
    p50_ms: number;
    p95_ms: number;
    p99_ms: number;
  }>;
  by_user: Array<{
    user_email: string;
    request_count: number;
    avg_ms: number;
    p50_ms: number;
    p95_ms: number;
    max_ms: number;
  }>;
  slowest_recent_requests: Array<{
    user_email: string;
    endpoint: string;
    method: string;
    status_code: number;
    duration_ms: number;
    created_at: string;
  }>;
}
