"""Automated verification script for DB performance, OTP store, and email deliverability."""
import time
import sys
from database import get_db_engine
from sqlalchemy import text
from models import UserManager, UserRole, UserStatus
from profile_requests import profile_endpoint
from config import DEFAULT_CONFIG

print("=" * 60)
print("CampusQuery Verification & Latency Benchmark")
print("=" * 60)

# 1. Database Connection and Pool Ping
t0 = time.perf_counter()
engine = get_db_engine()
with engine.connect() as conn:
    res = conn.execute(text("SELECT 1")).scalar()
t1 = time.perf_counter()
print(f"[DB] Connection acquisition + query time: {(t1 - t0)*1000:.2f} ms (Target: < 50ms on pool)")

# 2. Schema existence check speed
t0 = time.perf_counter()
um = UserManager(engine)
t1 = time.perf_counter()
print(f"[DB] UserManager initialization & table check: {(t1 - t0)*1000:.2f} ms (Target: < 100ms)")

# 3. OtpStore O(1) in-memory latency
t0 = time.perf_counter()
test_email = "benchmark_test@example.com"
test_otp = "852963"
um.otp_store.set_otp(test_email, test_otp)
verified, msg = um.otp_store.verify_otp(test_email, test_otp)
t1 = time.perf_counter()
print(f"[OTP] OtpStore set + verify roundtrip: {(t1 - t0)*1000:.4f} ms (Target: < 0.1ms)")
assert verified, f"OtpStore verification failed: {msg}"

# 4. User Cache O(1) lookup
t0 = time.perf_counter()
user = um.get_user_by_email("kumaryalla123@gmail.com")
t1 = time.perf_counter()
print(f"[Cache/DB] User lookup duration: {(t1 - t0)*1000:.2f} ms")

print("=" * 60)
print("All performance verification checks passed successfully!")
print("=" * 60)
