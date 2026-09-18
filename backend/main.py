"""FastAPI backend for user-scoped RAG chat with background memory compaction and auth."""

from __future__ import annotations

import os
import tempfile
import re
import warnings
import logging
from pathlib import Path
from functools import lru_cache
from typing import Any, List, Optional
from datetime import timedelta
import random
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Suppress noisy library warnings and logs
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

from fastapi import BackgroundTasks, FastAPI, HTTPException, Depends, status, UploadFile, File, Form, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr

from contextlib import asynccontextmanager

from config import DEFAULT_CONFIG

try:
    import logfire
    _token = DEFAULT_CONFIG.logfire_token or os.getenv("LOGFIRE_TOKEN")
    _security_patterns = [
        "password", "new_password", "current_password", "hashed_password",
        "otp", "otp_code", "token", "auth", "secret", "key", "access_token", "refresh_token"
    ]
    logfire.configure(
        token=_token if _token else None,
        service_name=DEFAULT_CONFIG.logfire_service_name,
        send_to_logfire="if-token-present",
        scrubbing=logfire.ScrubbingOptions(extra_patterns=_security_patterns) if DEFAULT_CONFIG.logfire_scrubbing else False,
    )
except Exception as _logfire_err:
    logfire = None

from postgres_memory import PostgresMemory
from retrieval_agent import RetrievalAgent
from memory_summarizer import MemorySummarizer
from guardrails import CampusGuardrails
from RETRIVAL.pipeline import answer_question, ingest_to_pgvector, get_affinda_extractor, get_hybrid_chunker
from RETRIVAL.pgvectorstore import PgVectorStore
from database import get_db_engine

import auth
from models import UserManager, UserRole, UserStatus


def clean_markdown_text(text: str) -> str:
    """Thoroughly strip all *, #, --, unnecessary divider symbols, and eliminate blank lines between paragraphs."""
    if not text:
        return ""
    # 0. Strip citation markers and references ([1], [Source: ...], [Document Chunk 1])
    cleaned = re.sub(r'\[\d+\]', '', text)
    cleaned = re.sub(r'\[Document Chunk\s*\d*.*?\]', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\[Source:?.*?\]', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\[\s*filename.*?\s*\]', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(?i)(^|\n)(citations?|sources?):\s*[\s\S]*$', '', cleaned)
    # 1. Strip markdown headings (# Title -> Title)
    cleaned = re.sub(r'(?m)^[ \t]*#{1,6}[ \t]*', '', cleaned)
    # 2. Strip all # characters anywhere in the response
    cleaned = cleaned.replace('#', '')
    # 3. Strip bold / italic asterisks (**text** or *text* -> text)
    cleaned = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', cleaned)
    # 4. Strip bullet asterisks at line starts (* item -> item) and any remaining *
    cleaned = re.sub(r'(?m)^[ \t]*\*[ \t]+', '', cleaned)
    cleaned = cleaned.replace('*', '')
    # 5. Strip horizontal rules / dividers (---, ---, etc.)
    cleaned = re.sub(r'(?m)^[ \t]*-{2,}[ \t]*$', '', cleaned)
    # 6. Strip double hyphens (-- or --- or more) anywhere in the text
    cleaned = re.sub(r'-{2,}', '', cleaned)
    # 7. Replace unicode dashes (em-dash, en-dash) with a space
    cleaned = cleaned.replace('—', ' ').replace('–', ' ')
    # 8. Strip leading bullet dashes (- item -> item) if followed by space
    cleaned = re.sub(r'(?m)^[ \t]*-[ \t]+', '', cleaned)
    # 9. Clean horizontal spacing on each line and remove empty blank lines
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in cleaned.splitlines()]
    non_empty_lines = [line for line in lines if line]
    cleaned = '\n'.join(non_empty_lines)
    return cleaned.strip()


@lru_cache(maxsize=1)
def get_user_manager() -> UserManager:
    """Return singleton UserManager without loading vector store, rerankers, or summarizers."""
    if not DEFAULT_CONFIG.postgres_url:
        raise RuntimeError("Set POSTGRES_URL before starting the FastAPI chat service.")
    return UserManager(get_db_engine())


@lru_cache(maxsize=1)
def get_resources() -> tuple[PgVectorStore, RetrievalAgent, PostgresMemory, MemorySummarizer, UserManager, CampusGuardrails]:
    """Create shared model, agent, memory store, background summarizer, user manager, and guardrails."""
    if not DEFAULT_CONFIG.postgres_url:
        raise RuntimeError("Set POSTGRES_URL before starting the FastAPI chat service.")
    engine = get_db_engine()
    memory = PostgresMemory(
        engine,
        recent_messages=DEFAULT_CONFIG.memory_recent_messages,
        summary_max_chars=DEFAULT_CONFIG.memory_summary_max_chars,
    )
    summarizer = MemorySummarizer(DEFAULT_CONFIG)
    user_manager = get_user_manager()
    guardrails = CampusGuardrails(DEFAULT_CONFIG)
    
    return (
        PgVectorStore(DEFAULT_CONFIG),
        RetrievalAgent(DEFAULT_CONFIG),
        memory,
        summarizer,
        user_manager,
        guardrails
    )


def _warmup_cloud_components():
    """Warm up remote cloud inference clients and extractors asynchronously in the background."""
    print("[startup] Preloading cloud components in background...")
    try:
        store, agent, memory, summarizer, user_manager, guardrails = get_resources()
        
        # 1. Preload PgVectorStore, remote Embedder (HF Inference) & Reranker (HF Inference)
        if hasattr(store, "preload"):
            store.preload()
        elif hasattr(store.embedder, "preload"):
            store.embedder.preload()

        # 2. Preload Remote Safety Guardrails (HF Inference)
        guardrails.preload()

        # 3. Preload Background Memory Summarizer (HF Inference / Groq)
        summarizer.preload()

        # 4. Preload Retrieval Agent & Query Rewriter (HF Inference / Groq)
        if hasattr(agent, "preload"):
            agent.preload()

        # 5. Preload Document Extractor & Chunker
        get_affinda_extractor()
        get_hybrid_chunker()

        print("[startup] All cloud components (Embedding, Reranker, Guardrails, Memory, Extractor) preloaded successfully in background!")
    except Exception as exc:
        print(f"[startup] Non-critical cloud component background init notice: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fast non-blocking startup: bootstrap essentials instantly and warm up cloud models in background."""
    print("[startup] Initializing CampusQuery backend...")
    try:
        get_user_manager()
    except Exception as exc:
        print(f"[startup] Database connection warning: {exc}")

    # Kick off remote cloud preloading asynchronously in background thread so app serves immediately
    asyncio.create_task(asyncio.to_thread(_warmup_cloud_components))
    print("[startup] CampusQuery backend is ready to accept requests!")
    yield
    # Shutdown logic: flush buffered metrics
    try:
        um = get_user_manager()
        um.metrics_buffer.flush_all()
    except Exception:
        pass


app = FastAPI(title="Resume & Career Analyzer Toolkit API", version="1.0.0", lifespan=lifespan)

if logfire:
    try:
        logfire.instrument_fastapi(app)
        print("[logfire] Pydantic Logfire instrumented successfully (send_to_logfire='if-token-present').")
    except Exception as _inst_err:
        print(f"[logfire] Fastapi instrumentation notice: {_inst_err}")


origins = [origin.strip() for origin in DEFAULT_CONFIG.cors_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://.*\.vercel\.app$|^https://.*\.onrender\.com$|^https://.*\.netlify\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def normalize_request_path(request: Request, call_next):
    """Normalize double slashes like //login to /login to avoid 404 errors."""
    path = request.scope.get("path", "")
    if "//" in path:
        request.scope["path"] = re.sub(r"/+", "/", path)
    return await call_next(request)


@app.middleware("http")
async def track_latency_and_metrics(request: Request, call_next):
    """Accurately measures request latency and stores metrics in PostgreSQL for P50/P95/P99 analytics."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    path = request.url.path
    # Ignore high-frequency static/doc endpoints
    if path in ["/", "/docs", "/redoc", "/openapi.json", "/favicon.ico"]:
        return response

    # Non-blocking user identification for per-user analytics
    user_email = None
    try:
        auth_header = request.headers.get("Authorization")
        token = None
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
        elif "access_token" in request.cookies:
            token = request.cookies.get("access_token")
        if token:
            payload = auth.decode_access_token(token)
            if payload:
                user_email = payload.get("sub")
    except Exception:
        pass

    # Record metric to PostgreSQL in background (zero response delay)
    try:
        um = get_user_manager()
        asyncio.create_task(
            asyncio.to_thread(
                um.log_api_metric,
                endpoint=path,
                method=request.method,
                status_code=response.status_code,
                duration_ms=duration_ms,
                user_email=user_email,
            )
        )
    except Exception:
        pass

    return response


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {
        "status": "online",
        "service": "Resume & Career Analyzer Toolkit API",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# Pydantic models for Auth
class UserRegister(BaseModel):
    name: Optional[str] = None
    nationality: Optional[str] = None
    email: EmailStr
    password: str

class OTPVerify(BaseModel):
    email: EmailStr
    otp: str

class ForgotPassword(BaseModel):
    email: EmailStr

class ResetPassword(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

class ProfileUpdate(BaseModel):
    name: str = Field(min_length=1)
    nationality: Optional[str] = None


class ChangePassword(BaseModel):
    current_password: str
    new_password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: Optional[str] = None


class ChatRequest(BaseModel):
    """Input required for one user-scoped chat request."""
    question: str = Field(min_length=1)
    # user_id is now retrieved from token


class ChatResponse(BaseModel):
    """Answer and retrieval metadata returned to the client."""
    answer: str
    citations: list[str]
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)



async def get_current_user(request: Request, um: UserManager = Depends(get_user_manager)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    token = None
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    if not token:
        token = request.cookies.get("access_token")
    if not token:
        token = request.query_params.get("token")
    if not token:
        raise credentials_exception
    payload = auth.decode_access_token(token, expected_type="access")
    if payload is None:
        raise credentials_exception
    email: str = payload.get("sub")
    if not email:
        raise credentials_exception
    clean_email = email.lower().strip()
    user = um.get_user_by_email(clean_email)
    if user is None:
        raise credentials_exception
    if user["status"] != UserStatus.ACTIVE.value:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Account is {user['status']}")
    return user

async def get_current_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != UserRole.ADMIN.value:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough privileges")
    return user

import smtplib
import requests
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid

# Dedicated thread pool for non-blocking OTP email dispatching
email_executor = ThreadPoolExecutor(max_workers=max(5, DEFAULT_CONFIG.smtp_pool_size), thread_name_prefix="otp-mailer")


def generate_otp() -> str:
    return str(random.randint(100000, 999999))


class ResendOTPRequest(BaseModel):
    email: EmailStr


def _build_otp_message(to_email: str, otp: str, from_header: str, from_addr: str, sender_domain: str) -> MIMEText:
    """Build a plain-text OTP email optimized for inbox delivery (anti-spam)."""
    body = (
        f"Hi,\n\n"
        f"Your verification code for CampusQuery is:\n\n"
        f"    {otp}\n\n"
        f"This code expires in 10 minutes.\n\n"
        f"If you did not request this code, you can safely ignore this email.\n\n"
        f"- CampusQuery Team"
    )

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = f"Verify your CampusQuery account"
    msg["From"] = from_header
    msg["To"] = to_email
    msg["Reply-To"] = from_addr
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=sender_domain)
    return msg


def _send_via_resend(to_email: str, otp: str) -> bool:
    """Send OTP email via Resend HTTPS REST API (Port 443 - works on Render Free Tier)."""
    api_key = (DEFAULT_CONFIG.resend_api_key or "").strip()
    if not api_key:
        return False
    try:
        t0 = time.perf_counter()
        from_email = (DEFAULT_CONFIG.smtp_from_email or "onboarding@resend.dev").strip()
        if not ("<" in from_email or "@" in from_email):
            from_email = "CampusQuery <onboarding@resend.dev>"
        elif "<" not in from_email:
            from_email = f"CampusQuery <{from_email}>"

        resp = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": from_email,
                "to": [to_email],
                "subject": "Verify your CampusQuery account",
                "text": (
                    f"Hi,\n\n"
                    f"Your verification code for CampusQuery is:\n\n"
                    f"    {otp}\n\n"
                    f"This code expires in 10 minutes.\n\n"
                    f"If you did not request this code, you can safely ignore this email.\n\n"
                    f"- CampusQuery Team"
                ),
            },
            timeout=5.0,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        if resp.status_code in (200, 201):
            print(f"[email:RESEND:OK] OTP delivered to {to_email} via Resend HTTPS API in {elapsed:.0f}ms")
            return True
        else:
            print(f"[email:RESEND:WARN] Resend API returned status {resp.status_code}: {resp.text}")
            return False
    except Exception as exc:
        print(f"[email:RESEND:ERROR] Resend API exception: {exc}")
        return False


def _send_via_brevo(to_email: str, otp: str) -> bool:
    """Send OTP email via Brevo HTTPS REST API (Port 443 - works on Render Free Tier)."""
    api_key = (DEFAULT_CONFIG.brevo_api_key or "").strip()
    if not api_key:
        return False
    try:
        t0 = time.perf_counter()
        from_email = (DEFAULT_CONFIG.smtp_from_email or "noreply@campusquery.com").strip()
        resp = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={
                "api-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "sender": {"name": "CampusQuery", "email": from_email},
                "to": [{"email": to_email}],
                "subject": "Verify your CampusQuery account",
                "textContent": (
                    f"Hi,\n\n"
                    f"Your verification code for CampusQuery is:\n\n"
                    f"    {otp}\n\n"
                    f"This code expires in 10 minutes.\n\n"
                    f"- CampusQuery Team"
                ),
            },
            timeout=5.0,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        if resp.status_code in (200, 201):
            print(f"[email:BREVO:OK] OTP delivered to {to_email} via Brevo HTTPS API in {elapsed:.0f}ms")
            return True
        else:
            print(f"[email:BREVO:WARN] Brevo API returned status {resp.status_code}: {resp.text}")
            return False
    except Exception as exc:
        print(f"[email:BREVO:ERROR] Brevo API exception: {exc}")
        return False


def _send_via_smtp(to_email: str, otp: str) -> bool:
    """Send OTP email via SMTP with direct SSL (port 465) / STARTTLS (port 587) fallback."""
    raw_username = DEFAULT_CONFIG.smtp_username
    raw_password = DEFAULT_CONFIG.smtp_password
    if not (raw_username and raw_password):
        return False

    username = raw_username.strip()
    password = raw_password.replace(" ", "").strip()
    from_addr = (DEFAULT_CONFIG.smtp_from_email or username).strip()
    from_header = f"CampusQuery <{from_addr}>" if "<" not in from_addr else from_addr
    server_host = (DEFAULT_CONFIG.smtp_server or "smtp.gmail.com").strip()
    configured_port = int(DEFAULT_CONFIG.smtp_port or 465)
    sender_domain = from_addr.split("@")[-1].rstrip(">").strip() if "@" in from_addr else "gmail.com"
    timeout = min(DEFAULT_CONFIG.smtp_timeout, 4.0)

    msg = _build_otp_message(to_email, otp, from_header, from_addr, sender_domain)

    alt_port = 465 if configured_port == 587 else 587
    ports_to_try = [configured_port, alt_port]

    last_error = None
    for port in ports_to_try:
        try:
            t0 = time.perf_counter()
            if port == 465:
                with smtplib.SMTP_SSL(server_host, port, timeout=timeout) as server:
                    server.login(username, password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(server_host, port, timeout=timeout) as server:
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                    server.login(username, password)
                    server.send_message(msg)
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[email:SMTP:OK] OTP delivered to {to_email} via port {port} in {elapsed:.0f}ms")
            return True
        except Exception as exc:
            last_error = exc
            print(f"[email:SMTP:RETRY] Port {port} failed for {to_email}: {exc}")

    print(f"[email:SMTP:FAIL] All SMTP ports failed for {to_email}: {last_error}")
    return False


def send_otp_email(to_email: str, otp: str) -> bool:
    """Send OTP email using HTTP API (Resend / Brevo) first, falling back to SMTP."""
    # 1. Try Resend HTTP API (Port 443 - works everywhere, immune to Render SMTP block)
    if DEFAULT_CONFIG.resend_api_key:
        if _send_via_resend(to_email, otp):
            return True

    # 2. Try Brevo HTTP API (Port 443)
    if DEFAULT_CONFIG.brevo_api_key:
        if _send_via_brevo(to_email, otp):
            return True

    # 3. Direct SMTP (Ports 465 / 587 - works locally and on paid VPS)
    if DEFAULT_CONFIG.smtp_username and DEFAULT_CONFIG.smtp_password:
        if _send_via_smtp(to_email, otp):
            return True

    # 4. Fallback logging
    print(f"[email:FAIL] Could not send OTP to {to_email}. OTP is: {otp}")
    if os.getenv("RENDER"):
        print("[email:NOTICE] Running on Render Free Tier: SMTP ports 25, 465, and 587 are blocked by Render.")
        print("[email:NOTICE] Add RESEND_API_KEY to your Render Environment Variables for instant HTTP delivery (resend.com).")
    return False


def _on_email_done(future):
    """Log result of background OTP email dispatch."""
    try:
        res = future.result()
        if not res:
            print("[email:WARN] Background email dispatch returned False (check logs above).")
    except Exception as exc:
        print(f"[email:ERROR] Background email thread crashed: {exc}")


def dispatch_otp_email(to_email: str, otp: str) -> None:
    """Non-blocking background dispatch of OTP emails to eliminate HTTP response latency."""
    fut = email_executor.submit(send_otp_email, to_email, otp)
    fut.add_done_callback(_on_email_done)


@app.post("/register")
def register(data: UserRegister, background_tasks: BackgroundTasks, um: UserManager = Depends(get_user_manager)):
    email = data.email.lower().strip()
    name = (data.name or "").strip() or email.split("@")[0]
    nationality = (data.nationality or "").strip() or "Not specified"
    existing_user = um.get_user_by_email(email)

    if existing_user:
        if existing_user["status"] == UserStatus.ACTIVE.value:
            raise HTTPException(status_code=400, detail="Email is already registered and verified. Please log in.")
        
        # User exists but is PENDING_VERIFICATION: update password and generate a fresh OTP
        hashed_password = auth.get_password_hash(data.password)
        otp = generate_otp()
        um.otp_store.set_otp(email, otp)
        um.update_password(email, hashed_password)
        background_tasks.add_task(um.update_otp, email, otp)
        # Non-blocking async dispatch (< 150ms response!)
        dispatch_otp_email(email, otp)
        resp = {"message": "Verification pending. A new OTP has been sent to your email.", "email": email}
        if DEFAULT_CONFIG.debug_otp:
            resp["otp_debug"] = otp
        return resp
    
    hashed_password = auth.get_password_hash(data.password)
    otp = generate_otp()
    um.otp_store.set_otp(email, otp)
    user = um.create_user(email=email, hashed_password=hashed_password, name=name, nationality=nationality, otp_code=otp)
    # Non-blocking async dispatch (< 150ms response!)
    dispatch_otp_email(email, otp)
    
    resp = {"message": "User registered. Please check email for OTP.", "email": user["email"]}
    if DEFAULT_CONFIG.debug_otp:
        resp["otp_debug"] = otp
    return resp


@app.post("/resend-otp")
def resend_otp(data: ResendOTPRequest, background_tasks: BackgroundTasks, um: UserManager = Depends(get_user_manager)):
    email = data.email.lower().strip()
    user = um.get_user_by_email(email)
    if not user:
        return {"message": "If the account exists, a new verification code has been dispatched."}
    
    if user["status"] == UserStatus.ACTIVE.value:
        return {"message": "Account is already verified. Please log in."}
        
    otp = generate_otp()
    um.otp_store.set_otp(email, otp)
    background_tasks.add_task(um.update_otp, email, otp)
    dispatch_otp_email(email, otp)
    resp = {"message": "A new verification code has been dispatched to your email."}
    if DEFAULT_CONFIG.debug_otp:
        resp["otp_debug"] = otp
    return resp


@app.post("/verify-otp")
def verify_otp(data: OTPVerify, um: UserManager = Depends(get_user_manager)):
    email = data.email.lower().strip()
    input_otp = str(data.otp).strip()
    
    # 1. Fast path: O(1) in-memory OtpStore verification (< 0.05ms)
    verified, reason = um.otp_store.verify_otp(email, input_otp)
    user = um.get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if user["status"] == UserStatus.ACTIVE.value:
        return {"message": "Email is already verified. Please log in."}
        
    if not verified:
        # Fallback to database otp_code if server restarted
        if not user.get("otp_code") or str(user["otp_code"]).strip() != input_otp:
            raise HTTPException(status_code=400, detail=reason or "Invalid OTP code. Please check your email.")
    
    um.update_user_status(email, UserStatus.ACTIVE.value)
    um.update_otp(email, None)
    return {"message": "Email verified successfully."}


@app.post("/login")
def login(response: Response, form_data: OAuth2PasswordRequestForm = Depends(), um: UserManager = Depends(get_user_manager)):
    email = form_data.username.lower().strip()
    user = um.get_user_by_email(email)
    if not user or not auth.verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    
    if user["status"] != UserStatus.ACTIVE.value:
        raise HTTPException(status_code=403, detail=f"Account status is {user['status']}")
        
    access_token = auth.create_access_token(data={"sub": user["email"]})
    refresh_token = auth.create_refresh_token(data={"sub": user["email"]})
    
    # Store refresh token and access token in secure HTTP cookies
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        max_age=auth.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
        samesite="none",
        secure=True,
        path="/"
    )
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=auth.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="none",
        secure=True,
        path="/"
    )
    
    return {
        "access_token": access_token, 
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "name": user.get("name") or user["email"].split("@")[0],
            "nationality": user.get("nationality") or "Not specified",
            "email": user["email"],
            "role": user["role"],
            "status": user["status"]
        }
    }


@app.post("/forgot-password")
def forgot_password(data: ForgotPassword, background_tasks: BackgroundTasks, um: UserManager = Depends(get_user_manager)):
    email = data.email.lower().strip()
    user = um.get_user_by_email(email)
    if not user:
        return {"message": "If the email is registered, an OTP was sent."}
    
    otp = generate_otp()
    um.otp_store.set_otp(email, otp)
    background_tasks.add_task(um.update_otp, email, otp)
    dispatch_otp_email(email, otp)
    resp = {"message": "If the email is registered, an OTP was sent."}
    if DEFAULT_CONFIG.debug_otp:
        resp["otp_debug"] = otp
    return resp


@app.post("/reset-password")
def reset_password(data: ResetPassword, um: UserManager = Depends(get_user_manager)):
    email = data.email.lower().strip()
    input_otp = str(data.otp).strip()
    verified, reason = um.otp_store.verify_otp(email, input_otp)
    user = um.get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if not verified:
        if not user.get("otp_code") or str(user["otp_code"]).strip() != input_otp:
            raise HTTPException(status_code=400, detail=reason or "Invalid OTP")
        
    hashed_password = auth.get_password_hash(data.new_password)
    um.update_password(email, hashed_password)
    um.update_otp(email, None)
    return {"message": "Password reset successfully."}


@app.post("/change-password")
def change_password(data: ChangePassword, current_user: dict = Depends(get_current_user), um: UserManager = Depends(get_user_manager)):
    if not auth.verify_password(data.current_password, current_user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    if len(data.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters long")
    hashed_password = auth.get_password_hash(data.new_password)
    um.update_password(current_user["email"], hashed_password)
    return {"message": "Password updated successfully."}


@app.post("/refresh")
async def refresh_token(request: Request, response: Response, data: Optional[RefreshTokenRequest] = None, um: UserManager = Depends(get_user_manager)):
    refresh_token_val = None
    if data and data.refresh_token:
        refresh_token_val = data.refresh_token
    if not refresh_token_val:
        refresh_token_val = request.cookies.get("refresh_token")
    if not refresh_token_val:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            refresh_token_val = auth_header.split(" ", 1)[1].strip()
    if not refresh_token_val:
        raise HTTPException(status_code=401, detail="Refresh token missing")
        
    payload = auth.decode_access_token(refresh_token_val, expected_type="refresh")
    if not payload or not payload.get("sub"):
        raise HTTPException(status_code=401, detail="Invalid refresh token")
        
    email = payload.get("sub")
    user = um.get_user_by_email(email)
    if not user or user["status"] != UserStatus.ACTIVE.value:
        raise HTTPException(status_code=401, detail="Invalid user or inactive account")
        
    access_token = auth.create_access_token(data={"sub": user["email"]})
    new_refresh_token = auth.create_refresh_token(data={"sub": user["email"]})
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite="none",
        secure=True,
        path="/"
    )
    response.set_cookie(
        key="refresh_token",
        value=new_refresh_token,
        httponly=True,
        samesite="none",
        secure=True,
        path="/"
    )
    return {
        "message": "Token refreshed",
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
    }

@app.post("/logout")
def logout(response: Response):
    response.delete_cookie("access_token", path="/", samesite="none", secure=True)
    response.delete_cookie("refresh_token", path="/", samesite="none", secure=True)
    return {"message": "Logged out successfully"}


@app.get("/me")
def get_me(user: dict = Depends(get_current_user)):
    return {
        "id": user["id"],
        "name": user.get("name") or user["email"].split("@")[0],
        "nationality": user.get("nationality") or "Not specified",
        "email": user["email"],
        "role": user["role"],
        "status": user["status"]
    }


@app.post("/update-profile")
def update_profile(data: ProfileUpdate, user: dict = Depends(get_current_user), um: UserManager = Depends(get_user_manager)):
    new_name = data.name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    new_nationality = (data.nationality or "").strip() or "Not specified"
    um.update_user_profile(user["email"], new_name, new_nationality)
    return {"message": "Profile updated successfully!", "name": new_name, "nationality": new_nationality}




@app.get("/history")
def get_history(user: dict = Depends(get_current_user)):
    user_id = str(user["id"])
    _, _, memory, _, _, _ = get_resources()
    messages = memory.get_all_messages(user_id)
    return {"messages": messages}


@app.post("/clear-chat")
def clear_chat_history(user: dict = Depends(get_current_user)):
    user_id = str(user["id"])
    print(f"\n[api:clear-chat] New Chat requested by user '{user_id}'. Clearing all data...")
    store, _, memory, _, _, _ = get_resources()
    memory.clear_chat(user_id)
    print(f"[api:clear-chat] Chat history cleared for user '{user_id}'.")
    store.delete_user_chunks(user_id)
    print(f"[api:clear-chat] All document chunks deleted. User '{user_id}' is ready for fresh document upload.\n")
    return {"message": "Chat history and document chunks cleared successfully."}



@app.get("/users")
def get_users(admin_user: dict = Depends(get_current_admin), um: UserManager = Depends(get_user_manager)):
    return um.get_all_users()


@app.put("/users/{user_id}/status")
def update_user_status_admin(user_id: int, status: str, admin_user: dict = Depends(get_current_admin), um: UserManager = Depends(get_user_manager)):
    user = um.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    um.update_user_status(user["email"], status)
    return {"message": f"User status updated to {status}"}


@app.delete("/users/{user_id}")
def delete_user_admin(
    user_id: int,
    admin_user: dict = Depends(get_current_admin),
    um: UserManager = Depends(get_user_manager),
):
    """Admin endpoint to permanently delete a user and clear their indexed resume chunks & chat memory in sub-second time."""
    user = um.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user["id"] == admin_user["id"]:
        raise HTTPException(status_code=400, detail="You cannot delete your own admin account.")

    deleted = um.delete_user(user_id, table_name=DEFAULT_CONFIG.pgvector_table)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete user from database")

    return {"message": f"User {user['email']} has been permanently deleted."}




@app.get("/admin/metrics/latency")
def get_latency_metrics_admin(
    hours: int = 24,
    admin_user: dict = Depends(get_current_admin),
    um: UserManager = Depends(get_user_manager),
):
    """Admin endpoint returning P50, P95, P99 API latency, per-user and per-endpoint breakdowns."""
    return um.get_latency_metrics(hours=hours)


@app.post("/upload")
def upload_documents(files: List[UploadFile] = File(...), user: dict = Depends(get_current_user)):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="Please select a resume file to upload.")
        if len(files) > 1:
            raise HTTPException(
                status_code=400,
                detail="Only one resume document can be uploaded and analyzed at a time. Multi-document upload is not supported."
            )

        user_id = str(user["id"])
        store, *_ = get_resources()

        # Clear previous document chunks so only one active resume is retained per user
        store.delete_user_chunks(user_id)
        print(f"[api:upload] Cleared prior document chunks for user '{user_id}'. Indexing single resume...")

        with tempfile.TemporaryDirectory(prefix="campusquery_") as temporary_directory:
            paths = []
            for idx, uploaded_file in enumerate(files):
                # Preserve original extension
                ext = Path(uploaded_file.filename).suffix
                path = Path(temporary_directory) / f"{idx}_{uploaded_file.filename}"
                path.write_bytes(uploaded_file.file.read())
                paths.append(path)

            # Ingestion logic using the preloaded store
            ingest_to_pgvector(paths, DEFAULT_CONFIG, user_id=user_id, store=store)

        return {"message": "Successfully indexed your resume document. Ready for analysis."}
    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        print(f"[api] Upload error: {exc!r}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")


async def compact_user_memory_async(
    memory: PostgresMemory, summarizer: MemorySummarizer, user_id: str
) -> None:
    """Asynchronous, non-blocking background task that extracts durable memory via thread pool."""
    print(f"\n[background-memory] >>> Triggered ASYNCHRONOUS background compaction for user: {user_id}")
    print(f"[background-memory] Using background model: {summarizer.model} (provider={getattr(summarizer.config, 'summarizer_inference_provider', summarizer.provider)})")
    try:
        await asyncio.to_thread(memory.compact_history, user_id, summarizer)
        print(f"[background-memory] <<< Asynchronous background memory compaction completed successfully for user {user_id}!\n")
    except Exception as exc:
        print(f"[background-memory] !!! Background memory compaction failed for user {user_id}: {exc!r}\n")


@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> dict[str, str]:
    """Return a lightweight service health response."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)) -> ChatResponse:
    """Answer a question, persist the turn, and schedule memory compaction."""
    user_id = str(user["id"])
    question = request.question.strip()
    print(f"\n[api:chat] >>> Incoming HTTP /chat request from user ID: {user_id}")
    print(f"[api:chat] Question: \"{question}\"")
    try:
        store, agent, memory, summarizer, _, guardrails = get_resources()
        if guardrails.token:
            print(f"[api:chat] Evaluating safety & relevance via Guardrail model: {guardrails.model} (provider={guardrails.provider})")
            input_guard = guardrails.check_input(question)
            print(f"[api:chat] Guardrail verdict: allowed={input_guard.allowed}, category={input_guard.category}")
            if not input_guard.allowed:
                print(f"[api:chat] Request blocked by guardrails: {input_guard.message}")
                return ChatResponse(
                    answer=input_guard.message,
                    citations=[],
                    retrieved_chunks=[],
                )

        context = memory.context_for(user_id)
        user_name = user.get("name") or "User"
        user_nat = user.get("nationality") or "Not specified"
        profile_prefix = f"User Profile Context:\nName: {user_name}\nNationality: {user_nat}\n\n"
        if isinstance(context, dict):
            context["summary"] = profile_prefix + (context.get("summary") or "")

        print(f"[api:chat] Invoking direct RAG answering pipeline...")
        result = answer_question(
            question,
            store,
            DEFAULT_CONFIG,
            agent=agent,
            metadata_filters={"user_id": [user_id]},
            memory=memory,
            user_id=user_id,
            conversation_context=context,
        )
        clean_ans = clean_markdown_text(result.get("answer", ""))
        memory.add_message(user_id, "user", question)
        memory.add_message(user_id, "assistant", clean_ans)
        msg_count = memory.message_count(user_id)
        print(f"[api:chat] Message saved. Current count for user {user_id}: {msg_count} (threshold: {DEFAULT_CONFIG.memory_compaction_threshold})")
        if msg_count > DEFAULT_CONFIG.memory_compaction_threshold:
            print(f"[api:chat] Message count exceeded threshold ({DEFAULT_CONFIG.memory_compaction_threshold}). Scheduling async background compaction.")
            background_tasks.add_task(compact_user_memory_async, memory, summarizer, user_id)

        print(f"[api:chat] <<< Response ready ({len(clean_ans)} chars). Returning ChatResponse.\n")
        return ChatResponse(
            answer=clean_ans,
            citations=result.get("citations", []),
            retrieved_chunks=result.get("retrieved_chunks", []),
        )
    except Exception as exc:
        print(f"[api] Chat request failed: {exc!r}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc



@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            question = str(data.get("question", "")).strip()
            token = (
                data.get("token")
                or websocket.query_params.get("token")
                or websocket.cookies.get("access_token")
                or (websocket.headers.get("authorization") or "").replace("Bearer ", "").strip()
            )

            if not question:
                continue

            # Validate user credentials
            um = get_user_manager()
            user = None
            if token:
                payload = auth.decode_access_token(token, expected_type="access")
                if payload and payload.get("sub"):
                    user = um.get_user_by_email(payload.get("sub"))

            if not user or user.get("status") != UserStatus.ACTIVE.value:
                await websocket.send_json({"type": "error", "message": "Authentication required. Please log in again."})
                await websocket.close(code=1008)
                break

            user_id = str(user["id"])
            await websocket.send_json({"type": "start"})
            await websocket.send_json({"type": "status", "message": "🔍 Detecting intent & rewriting query before retrieval..."})

            store, agent, memory, summarizer, _, guardrails = get_resources()

            if guardrails.token:
                input_guard = guardrails.check_input(question)
                if not input_guard.allowed:
                    await websocket.send_json({"type": "chunk", "text": input_guard.message})
                    await websocket.send_json({"type": "end", "answer": input_guard.message})
                    continue

            context = memory.context_for(user_id)
            user_name = user.get("name") or "User"
            user_nat = user.get("nationality") or "Not specified"
            profile_prefix = f"User Profile Context:\nName: {user_name}\nNationality: {user_nat}\n\n"
            if isinstance(context, dict):
                context["summary"] = profile_prefix + (context.get("summary") or "")

            await websocket.send_json({"type": "status", "message": "🧠 Thinking & analyzing your resume details..."})

            result = answer_question(
                question,
                store,
                DEFAULT_CONFIG,
                agent=agent,
                metadata_filters={"user_id": [user_id]},
                memory=memory,
                user_id=user_id,
                conversation_context=context,
            )

            full_answer = clean_markdown_text(result.get("answer", ""))
            await websocket.send_json({"type": "status", "message": "✍️ Formulating final response..."})

            # Stream response in token/word chunks for real-time WebSocket feedback
            words = full_answer.split(" ")
            for i in range(0, len(words), 4):
                chunk = " ".join(words[i:i+4]) + (" " if i + 4 < len(words) else "")
                await websocket.send_json({"type": "chunk", "text": chunk})
                await asyncio.sleep(0.015)

            memory.add_message(user_id, "user", question)
            memory.add_message(user_id, "assistant", full_answer)
            msg_count = memory.message_count(user_id)
            print(f"[ws:chat] Turn saved. Current message count for user {user_id}: {msg_count} (threshold: {DEFAULT_CONFIG.memory_compaction_threshold})")
            if msg_count > DEFAULT_CONFIG.memory_compaction_threshold:
                print(f"[ws:chat] Message count exceeded threshold ({DEFAULT_CONFIG.memory_compaction_threshold}). Spawning async background compaction.")
                asyncio.create_task(compact_user_memory_async(memory, summarizer, user_id))

            await websocket.send_json({
                "type": "end",
                "answer": full_answer,
                "citations": [],
                "retrieved_chunks": result.get("retrieved_chunks", []),
            })
            print(f"[ws:chat] <<< Stream completed for user {user_id}.\n")

    except WebSocketDisconnect:
        print("[ws] Client disconnected from chat WebSocket.")
    except Exception as exc:
        print(f"[ws] Error in WebSocket chat: {exc!r}")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass

