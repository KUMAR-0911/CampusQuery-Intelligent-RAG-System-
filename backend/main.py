"""FastAPI backend for user-scoped RAG chat with background memory compaction and auth."""

from __future__ import annotations

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources and eagerly load all cloud components on application startup."""
    print("[startup] Initializing backend resources and preloading all cloud components...")
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

        print("[startup] All cloud components (Embedding, Reranker, Guardrails, Memory, Extractor) preloaded successfully! Ready for requests.")
    except Exception as exc:
        print(f"[startup] Non-critical cloud component init notice: {exc}")
    yield


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
    if email is None:
        raise credentials_exception
    user = um.get_user_by_email(email)
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
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def generate_otp() -> str:
    return str(random.randint(100000, 999999))


class ResendOTPRequest(BaseModel):
    email: EmailStr


def send_otp_email(to_email: str, otp: str) -> None:
    """Send OTP email using SMTP if configured with fast connection timeouts; fallback to printing in console."""
    username = DEFAULT_CONFIG.smtp_username
    password = DEFAULT_CONFIG.smtp_password
    if username and password:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"CampusQuery - Your Verification Code: {otp}"
            msg["From"] = DEFAULT_CONFIG.smtp_from_email or username
            msg["To"] = to_email

            text_content = f"Your CampusQuery OTP verification code is: {otp}\nValid for 10 minutes."
            html_content = f"""
            <html>
              <body style="font-family: Arial, sans-serif; padding: 20px; color: #333;">
                <h2>CampusQuery Verification Code</h2>
                <p>Your OTP verification code is:</p>
                <h1 style="color: #4F46E5; letter-spacing: 4px;">{otp}</h1>
                <p>Please enter this code in the application to complete your verification.</p>
              </body>
            </html>
            """
            msg.attach(MIMEText(text_content, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            port = int(DEFAULT_CONFIG.smtp_port)
            server_host = DEFAULT_CONFIG.smtp_server

            if port == 465:
                with smtplib.SMTP_SSL(server_host, port, timeout=7.0) as server:
                    server.login(username, password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(server_host, port, timeout=7.0) as server:
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                    server.login(username, password)
                    server.send_message(msg)
            print(f"[email] OTP email successfully sent to {to_email}")
        except Exception as exc:
            print(f"[email] Failed to send email to {to_email}: {exc}. Fallback OTP in logs: {otp}")
    else:
        print(f"--- OTP for {to_email} is {otp} (Configure SMTP_USERNAME & SMTP_PASSWORD in .env to enable real email sending) ---")


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
        um.update_password(email, hashed_password)
        um.update_otp(email, otp)
        background_tasks.add_task(send_otp_email, email, otp)
        return {"message": "Verification pending. A new OTP has been sent to your email.", "email": email}
    
    hashed_password = auth.get_password_hash(data.password)
    otp = generate_otp()
    user = um.create_user(email=email, hashed_password=hashed_password, name=name, nationality=nationality, otp_code=otp)
    
    background_tasks.add_task(send_otp_email, email, otp)
    
    return {"message": "User registered. Please check email for OTP.", "email": user["email"]}


@app.post("/resend-otp")
def resend_otp(data: ResendOTPRequest, background_tasks: BackgroundTasks, um: UserManager = Depends(get_user_manager)):
    email = data.email.lower().strip()
    user = um.get_user_by_email(email)
    if not user:
        return {"message": "If the account exists, a new verification code has been dispatched."}
    
    if user["status"] == UserStatus.ACTIVE.value:
        return {"message": "Account is already verified. Please log in."}
        
    otp = generate_otp()
    um.update_otp(email, otp)
    background_tasks.add_task(send_otp_email, email, otp)
    return {"message": "A new verification code has been dispatched to your email."}


@app.post("/verify-otp")
def verify_otp(data: OTPVerify, um: UserManager = Depends(get_user_manager)):
    email = data.email.lower().strip()
    user = um.get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if user["status"] == UserStatus.ACTIVE.value:
        return {"message": "Email is already verified. Please log in."}
        
    if not user["otp_code"] or str(user["otp_code"]).strip() != str(data.otp).strip():
        raise HTTPException(status_code=400, detail="Invalid OTP code. Please check the backend console or your email.")
    
    um.update_user_status(email, UserStatus.ACTIVE.value)
    um.update_otp(email, None)
    return {"message": "Email verified successfully."}


@app.post("/login")
def login(response: Response, form_data: OAuth2PasswordRequestForm = Depends(), um: UserManager = Depends(get_user_manager)):
    email = form_data.username.lower()
    user = um.get_user_by_email(email)
    if not user or not auth.verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    
    if user["status"] != UserStatus.ACTIVE.value:
        raise HTTPException(status_code=403, detail=f"Account status is {user['status']}")
        
    access_token = auth.create_access_token(data={"sub": user["email"]})
    refresh_token = auth.create_refresh_token(data={"sub": user["email"]})
    
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite="none",
        secure=True,  # Set to True in production with HTTPS
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        samesite="none",
        secure=True,
    )
    
    return {
        "message": "Login successful",
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
    email = data.email.lower()
    user = um.get_user_by_email(email)
    if not user:
        return {"message": "If the email is registered, an OTP was sent."}
    
    otp = generate_otp()
    um.update_otp(email, otp)
    background_tasks.add_task(send_otp_email, email, otp)
    return {"message": "If the email is registered, an OTP was sent."}



@app.post("/reset-password")
def reset_password(data: ResetPassword, um: UserManager = Depends(get_user_manager)):
    email = data.email.lower()
    user = um.get_user_by_email(email)
    if not user or user["otp_code"] != data.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
        
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
    )
    response.set_cookie(
        key="refresh_token",
        value=new_refresh_token,
        httponly=True,
        samesite="none",
        secure=True,
    )
    return {
        "message": "Token refreshed",
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
    }

@app.post("/logout")
def logout(response: Response):
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")
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
    """Admin endpoint to permanently delete a user and clear their indexed resume chunks & chat memory."""
    user = um.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user["id"] == admin_user["id"]:
        raise HTTPException(status_code=400, detail="You cannot delete your own admin account.")

    try:
        store, _, memory, _, _, _ = get_resources()
        store.delete_user_chunks(str(user_id))
        memory.clear_chat(str(user_id))
    except Exception as exc:
        print(f"[admin:delete_user] Notice during resource cleanup: {exc}")

    deleted = um.delete_user(user_id)
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


import asyncio

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
            for i in range(0, len(words), 3):
                chunk = " ".join(words[i:i+3]) + (" " if i + 3 < len(words) else "")
                await websocket.send_json({"type": "chunk", "text": chunk})
                await asyncio.sleep(0.04)

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

