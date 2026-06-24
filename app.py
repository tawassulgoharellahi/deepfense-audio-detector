import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import tempfile
import shutil
import numpy as np
import uvicorn
import secrets
import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from starlette.middleware.sessions import SessionMiddleware
import gradio as gr

from utils import preprocess_audio, get_audio_duration
from detector import AudioDeepfakeDetector
from quota_manager import QuotaManager

# 1. Initialize Quota Manager
quota_manager = QuotaManager()

# 2. Initialize FastAPI Application
app = FastAPI(
    title="DeepFense AI Audio Detector API",
    description="JSON API endpoint for next-generation deepfake and AI voice detection.",
    version="1.0.0"
)

session_secret = os.getenv("SESSION_SECRET", secrets.token_hex(32))

# 3. Initialize Model Detector
print("Loading detector pipeline at startup...")
detector = AudioDeepfakeDetector()
print("Detector pipeline loaded successfully.")

# 4. Authentication Middleware
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    
    # Define endpoints exempt from user authentication
    exempt_paths = [
        "/login",
        "/login/google",
        "/login/google/callback",
        "/api/detect",
        "/api/auth-status",
        "/docs",
        "/openapi.json"
    ]
    
    # Check if request path is exempt or targets static client-side resources
    is_exempt = any(path == p or path.startswith(p + "/") for p in exempt_paths) or \
                path.startswith("/assets/") or \
                path.endswith((".js", ".css", ".png", ".jpg", ".ico", ".svg", ".map"))
                
    if not is_exempt:
        user = request.session.get("user")
        if not user:
            # For page loads, redirect to login page. For backend calls, raise 401.
            accept_header = request.headers.get("accept", "")
            if "text/html" in accept_header:
                return RedirectResponse(url="/login")
            else:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Unauthorized. Please log in first."}
                )
                
    response = await call_next(request)
    return response

# Register middlewares in correct LIFO order: SessionMiddleware must execute first,
# so it is registered last (outermost layer).
from starlette.middleware.base import BaseHTTPMiddleware
app.add_middleware(BaseHTTPMiddleware, dispatch=auth_middleware)
app.add_middleware(
    SessionMiddleware,
    secret_key=session_secret,
    max_age=86400,      # 1 day session validity
    same_site="none",   # Required: allow cookie in cross-site iframe (HF embeds in huggingface.co)
    https_only=True     # Required when SameSite=None; HF Spaces serves over HTTPS
)

# 5. OAuth & Session Authentication Endpoints
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - DeepFense AI Audio Detector</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        html, body {{
            background-color: #080d1a;
            color: #e2e8f0;
            font-family: 'JetBrains Mono', monospace;
            margin: 0;
            padding: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            overflow: hidden;
        }}
        .container {{
            background-color: #0b132b;
            border: 1px solid rgba(0, 240, 255, 0.25);
            box-shadow: 0 0 25px rgba(0, 240, 255, 0.08);
            border-radius: 8px;
            padding: 40px;
            width: 100%;
            max-width: 420px;
            text-align: center;
        }}
        h1 {{
            color: #00f0ff;
            font-size: 20px;
            font-weight: 800;
            letter-spacing: 1px;
            margin-bottom: 30px;
            text-transform: uppercase;
        }}
        .btn {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #00f0ff, #0077ff);
            border: none;
            border-radius: 4px;
            color: #080d1a;
            font-weight: 700;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 0 10px rgba(0, 240, 255, 0.15);
            text-decoration: none;
            box-sizing: border-box;
        }}
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 0 20px rgba(0, 240, 255, 0.3);
        }}
        .footer {{
            margin-top: 20px;
            font-size: 10px;
            color: #64748b;
        }}
    </style>
    <script>
        // Adjust body size if inside an iframe (e.g. Hugging Face Spaces) to allow auto-resizing
        if (window.self !== window.top) {{
            document.documentElement.style.height = 'auto';
            document.body.style.height = 'auto';
            document.body.style.minHeight = 'auto';
            document.documentElement.style.overflow = 'hidden';
            document.body.style.overflow = 'hidden';
        }}
    </script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js" async></script>
</head>
<body>
    <div class="container">
        <h1>DEEPFENSE ACCESS</h1>
        
        {google_button}
        
        <div class="footer">
            DeepFense AI Audio Detector &copy; 2026
        </div>
    </div>
</body>
</html>
"""

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.session.get("user"):
        return RedirectResponse(url="/")
        
    google_button = """
    <button onclick="login()" class="btn">Sign In with Google</button>
    <script>
        function login() {
            const width = 600;
            const height = 700;
            const left = (window.screen.width / 2) - (width / 2);
            const top = (window.screen.height / 2) - (height / 2);
            window.open(
                '/login/google',
                'Google Login',
                'width=' + width + ',height=' + height + ',top=' + top + ',left=' + left + ',resizable=yes,scrollbars=yes,status=yes'
            );

            // Start polling server for auth status after popup opens
            // This is the most reliable method — uses the same session cookie
            // mechanism as manual reload, bypassing storage partitioning issues
            const pollId = setInterval(async () => {
                try {
                    const resp = await fetch('/api/auth-status', { credentials: 'include' });
                    const data = await resp.json();
                    if (data.authenticated) {
                        console.log('[AUTH] Server confirmed authentication, redirecting...');
                        clearInterval(pollId);
                        window.location.href = '/';
                    }
                } catch (e) {
                    console.error('[AUTH] Poll failed:', e);
                }
            }, 2000);

            // Stop polling after 10 minutes
            setTimeout(() => clearInterval(pollId), 600000);
        }
    </script>
    """
    return LOGIN_TEMPLATE.format(google_button=google_button)

@app.get("/login/google")
async def login_google(request: Request):
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not client_id:
        raise HTTPException(status_code=500, detail="Google client ID not configured in .env")
        
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    if not redirect_uri:
        host = request.headers.get("x-forwarded-host")
        if not host:
            host = request.headers.get("host", "localhost:7860")
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        redirect_uri = f"{proto}://{host}/login/google/callback"
        
    auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?response_type=code"
        f"&client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&scope=openid%20email%20profile"
        f"&prompt=select_account"
    )
    return RedirectResponse(url=auth_url)

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login")

@app.get("/api/auth-status")
async def auth_status(request: Request):
    """Lightweight endpoint polled by login page to detect successful authentication."""
    user = request.session.get("user")
    return JSONResponse(content={"authenticated": bool(user)})

@app.get("/login/google/callback")
async def auth_callback(request: Request, code: str = None, error: str = None):
    if error:
        return HTMLResponse(content=f"<h3>Authentication failed: {error}</h3>", status_code=400)
    if not code:
        return RedirectResponse(url="/login")
        
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    if not redirect_uri:
        host = request.headers.get("x-forwarded-host")
        if not host:
            host = request.headers.get("host", "localhost:7860")
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        redirect_uri = f"{proto}://{host}/login/google/callback"
    
    if not client_id or not client_secret:
        return HTMLResponse(content="<h3>Google OAuth credentials missing on server.</h3>", status_code=500)
        
    import httpx
    async with httpx.AsyncClient() as client:
        # Exchange authorization code for token
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code"
        }
        token_resp = await client.post(token_url, data=token_data)
        if token_resp.status_code != 200:
            return HTMLResponse(content=f"<h3>Token exchange failed: {token_resp.text}</h3>", status_code=400)
            
        tokens = token_resp.json()
        access_token = tokens.get("access_token")
        
        # Retrieve user info
        userinfo_url = "https://www.googleapis.com/oauth2/v3/userinfo"
        userinfo_resp = await client.get(userinfo_url, headers={"Authorization": f"Bearer {access_token}"})
        if userinfo_resp.status_code != 200:
            return HTMLResponse(content=f"<h3>User profile fetch failed: {userinfo_resp.text}</h3>", status_code=400)
            
        userinfo = userinfo_resp.json()
        email = userinfo.get("email")
        name = userinfo.get("name", email)
        
        request.session["user"] = {
            "email": email,
            "name": name
        }
        
    success_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Access Granted</title>
        <style>
            body {
                background-color: #080d1a;
                color: #e2e8f0;
                font-family: 'Inter', sans-serif;
                text-align: center;
                padding-top: 80px;
            }
            .spinner {
                width: 40px; height: 40px;
                border: 3px solid rgba(99,102,241,0.2);
                border-top-color: #6366f1;
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
                margin: 0 auto 20px;
            }
            @keyframes spin { to { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="spinner"></div>
        <h2>Access Granted Successfully!</h2>
        <p style="color: #64748b; font-size: 0.85rem;">Redirecting back to portal...</p>
        <script>
            // 1. Send signal via BroadcastChannel
            try {
                const bc = new BroadcastChannel("auth_channel");
                bc.postMessage("login_success");
                bc.close();
            } catch (e) {
                console.error("BroadcastChannel postMessage failed:", e);
            }

            // 2. Send signal via LocalStorage storage event fallback
            try {
                localStorage.setItem("login_success", Date.now().toString());
            } catch (e) {
                console.error("LocalStorage write failed:", e);
            }

            // 3. Fallback to postMessage (if window.opener is still alive)
            try {
                if (window.opener) {
                    window.opener.postMessage("login_success", "*");
                }
            } catch (e) {
                console.error("Opener postMessage failed:", e);
            }

            // 4. Close the popup
            setTimeout(() => {
                window.close();
            }, 800);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=success_html, status_code=200)

# 6. Define FastAPI JSON Endpoint (Protected by Session/IP Quotas)
@app.post("/api/detect")
async def detect_api(request: Request, file: UploadFile = File(...)):
    """
    Direct JSON API endpoint to detect AI spoofed/deepfake audio.
    Accepts any standard audio file, validates its length (<= 12 seconds),
    resamples to 16kHz mono, trims silence, and returns segment-level prediction report.
    Enforces user quota limits (10 scans per day).
    """
    if not file:
        raise HTTPException(status_code=400, detail="No audio file uploaded.")
        
    # Map identity: email if logged in, otherwise client IP
    user = request.session.get("user")
    identifier = user["email"] if user else request.client.host
    
    # Check/consume quota (limit 10 scans per day)
    allowed, count, remaining = quota_manager.consume_quota(identifier, limit=10)
    if not allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Quota Exceeded: You have used your daily limit of 10 scans."
        )
        
    # Write uploaded stream to a secure temporary file
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
        
    try:
        # Validate duration limit first
        duration = get_audio_duration(temp_path)
        if duration > 12.0:
            raise HTTPException(
                status_code=400,
                detail=f"Audio duration ({duration:.2f}s) exceeds the maximum limit of 12.0 seconds."
            )
            
        # Preprocess and predict
        audio_data = preprocess_audio(temp_path)
        result = detector.predict(audio_data)
        
        # Include duration and remaining scans count in API response metadata
        result["duration_seconds"] = duration
        result["quota_remaining"] = remaining
        
        return JSONResponse(content=result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference processing error: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# 7. Define Gradio Interface & Business Logic
def run_gradio_inference(audio_path, request: gr.Request):
    if not request:
        return (
            "System: Request context missing.\n[Ready]",
            "Error",
            "<div>Not Logged In</div>"
        )
        
    user = request.request.session.get("user")
    if not user:
        return (
            "System: Session expired.\n[Ready]",
            "Error",
            "<div>Session Expired. Please login again.</div>"
        )
        
    email = user["email"]
    
    # Helper to generate status bar HTML
    def make_status_html(email, remaining):
        warning_style = "color: #ef4444;" if remaining <= 2 else "color: #00ff66;"
        return f"""
        <div class="custom-status-bar">
            <div class="status-user">
                <span style="color: #94a3b8;">User:</span> <span style="color: #00f0ff; font-weight: 600;">{email}</span>
            </div>
            <div class="status-quota">
                <div>
                    <span style="color: #94a3b8;">Daily Quota:</span> <span style="{warning_style} font-weight: 600;">{remaining} / 10 scans remaining</span>
                </div>
                <a href="/logout" class="logout-link">Logout</a>
            </div>
        </div>
        """
        
    # Get current status (no decrement yet)
    current, remaining = quota_manager.get_quota_status(email, limit=10)
    
    if not audio_path:
        return (
            "System: No audio input provided.\n[Ready]",
            """
            <div class="terminal-card" style="text-align: center; padding: 40px 20px; color: #94a3b8;">
                Please upload an audio file or record voice to begin analysis.
            </div>
            """,
            make_status_html(email, remaining)
        )
        
    # Atomically check and consume quota
    allowed, count, remaining = quota_manager.consume_quota(email, limit=10)
    status_bar_html = make_status_html(email, remaining)
    
    if not allowed:
        log_output = "[QUOTA EXCEEDED]\n>>> You have exceeded your limit of 10 scans per day.\n>>> Reset occurs at midnight local server time.\n[SYSTEM STANDBY]"
        error_html = f"""
        <div class="verdict-fake" style="border-color: #ef4444; background: rgba(239, 68, 68, 0.1);">
            <h2 style="color: #ef4444; margin: 0; font-size: 18px;">Scan Blocked</h2>
            <p style="color: #e2e8f0; font-size: 14px; margin-top: 8px;">Quota Exceeded: You have used your daily limit of 10 scans.</p>
        </div>
        """
        return log_output, error_html, status_bar_html
        
    log_output = []
    log_output.append("[SYSTEM INITIALIZATION]")
    log_output.append(">>> Loading audio source file...")
    
    try:
        # Step 1: Duration check
        duration = get_audio_duration(audio_path)
        log_output.append(f">>> File duration: {duration:.2f} seconds")
        if duration > 12.0:
            raise ValueError(f"File duration ({duration:.2f}s) exceeds the maximum limit of 12.0 seconds.")
            
        # Step 2: Resampling & silence trimming
        log_output.append(">>> Running preprocessing pipeline (resampling to 16kHz mono, trimming silence)...")
        preprocessed = preprocess_audio(audio_path)
        log_output.append(f">>> Preprocessing complete. Target length: {len(preprocessed)} samples.")
        
        num_chunks = int(np.ceil(len(preprocessed) / 64000))
        log_output.append(f">>> Splitting sequence into {num_chunks} segment(s) of 4.0s.")
        log_output.append(">>> Running model inference (WavLM-Large frontend + Nes2Net classifier)...")
        
        # Step 3: Run inference
        result = detector.predict(preprocessed)
        log_output.append(">>> Model inference complete.")
        log_output.append(f">>> Final Verdict: {result['overall_label']}")
        log_output.append(f">>> Real Confidence: {result['real_confidence']*100:.1f}% | Spoof Confidence: {result['spoof_confidence']*100:.1f}%")
        log_output.append("[DIAGNOSTIC WORKFLOW COMPLETE]")
        
        # Step 4: Render HTML reports
        is_spoof = result["is_spoof"]
        verdict_class = "verdict-fake" if is_spoof else "verdict-real"
        verdict_text = "FAKE / AI SPOOF DETECTED" if is_spoof else "VERIFIED REAL AUDIO"
        badge_color = "#ef4444" if is_spoof else "#10b981"
        
        html_output = f"""
        <div style="font-family: 'JetBrains Mono', 'Inter', sans-serif;">
            <!-- Verdict Banner -->
            <div class="{verdict_class}" style="margin-bottom: 20px;">
                <h1 style="font-size: 24px; font-weight: 700; margin: 0; display: flex; align-items: center; justify-content: center; gap: 10px; color: {badge_color};">
                    {verdict_text}
                </h1>
                <p style="margin: 10px 0 0 0; color: #94a3b8; font-size: 14px;">
                    Overall Spoof Confidence: {result['spoof_confidence']*100:.2f}% | Real Confidence: {result['real_confidence']*100:.2f}%
                </p>
            </div>
            
            <!-- Details Grid -->
            <div style="display: grid; grid-template-columns: 1fr; gap: 15px;">
                <!-- Confidence Meter -->
                <div class="terminal-card">
                    <h3 style="margin-top: 0; color: #3b82f6; font-size: 16px; font-weight: 600; text-transform: uppercase;">Deepfake Probability Meter</h3>
                    <div style="display: flex; justify-content: space-between; font-size: 11px; color: #64748b; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 4px;">
                        <span>Genuine</span>
                        <span>Spoofed</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" style="width: {result['spoof_confidence']*100:.1f}%; background-color: {badge_color};"></div>
                    </div>
                    <div style="text-align: right; margin-top: 5px; font-size: 14px; font-weight: 600; color: {badge_color};">
                        {result['spoof_confidence']*100:.1f}% Spoof Probability
                    </div>
                </div>
                
                <!-- Segment Timeline -->
                <div class="terminal-card">
                    <h3 style="margin-top: 0; color: #3b82f6; font-size: 16px; font-weight: 600; text-transform: uppercase;">Segment Timeline Analysis</h3>
                    <p style="font-size: 12px; color: #94a3b8; margin-top: 0; margin-bottom: 15px;">
                        The audio is analyzed in 4-second intervals. If any single segment is spoofed, the entire clip is marked as fake.
                      </p>
                      <div>
          """
          
        for seg in result["segments"]:
            seg_is_fake = seg["label"] == "Fake/AI"
            seg_class = "fake" if seg_is_fake else "real"
            seg_badge_color = "#ef4444" if seg_is_fake else "#10b981"
            seg_badge = "SPOOF" if seg_is_fake else "REAL"
            
            html_output += f"""
                        <div class="segment-row {seg_class}">
                            <div>
                                <span style="font-weight: 600; color: #f1f5f9;">{seg['time_range']}</span>
                                <span style="margin-inline-start: 10px; font-size: 11px; padding: 2px 6px; border-radius: 4px; background: rgba({(239 if seg_is_fake else 16)}, {(68 if seg_is_fake else 185)}, {(68 if seg_is_fake else 129)}, 0.2); color: {seg_badge_color}; font-weight: 600; border: 1px solid {seg_badge_color};">{seg_badge}</span>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 13px; font-weight: 600; color: #f1f5f9;">
                                    {seg['spoof_probability']*100:.1f}% Spoof
                                </div>
                                <div style="font-size: 11px; color: #94a3b8;">
                                    Real: {seg['real_probability']*100:.1f}%
                                </div>
                            </div>
                        </div>
            """
            
        html_output += """
                    </div>
                </div>
            </div>
        </div>
        """
        return "\n".join(log_output), html_output, status_bar_html
        
    except Exception as e:
        log_output.append(f"\n[ERROR] Diagnostic analysis aborted:")
        log_output.append(f">>> {str(e)}")
        log_output.append("[SYSTEM SHUTDOWN]")
        
        error_html = f"""
        <div class="verdict-fake" style="border-color: #ef4444; background: rgba(239, 68, 68, 0.1);">
            <h2 style="color: #ef4444; margin: 0; font-size: 18px;">Analysis Failed</h2>
            <p style="color: #e2e8f0; font-size: 14px; margin-top: 8px;">{str(e)}</p>
        </div>
        """
        return "\n".join(log_output), error_html, status_bar_html

def on_page_load(request: gr.Request):
    if not request:
        return """
        <div style="padding: 10px; color: #ef4444; font-family: monospace;">
            Error: Session context unavailable.
        </div>
        """
    user = request.request.session.get("user")
    if not user:
        return """
        <div style="padding: 10px; color: #ef4444; font-family: monospace;">
            Error: Session missing. Please log in first.
        </div>
        """
    email = user["email"]
    current, remaining = quota_manager.get_quota_status(email, limit=10)
    
    warning_style = "color: #ef4444;" if remaining <= 2 else "color: #00ff66;"
    
    return f"""
    <div class="custom-status-bar">
        <div class="status-user">
            <span style="color: #94a3b8;">User:</span> <span style="color: #00f0ff; font-weight: 600;">{email}</span>
        </div>
        <div class="status-quota">
            <div>
                <span style="color: #94a3b8;">Daily Quota:</span> <span style="{warning_style} font-weight: 600;">{remaining} / 10 scans remaining</span>
            </div>
            <a href="/logout" class="logout-link">Logout</a>
        </div>
    </div>
    """

# Define visual style custom CSS
custom_css = """
body {
    background-color: #080d1a !important;
}
.gradio-container {
    background-color: #080d1a !important;
    font-family: 'JetBrains Mono', 'Inter', -apple-system, sans-serif !important;
    color: #e2e8f0 !important;
}
.terminal-card {
    background-color: #0b132b !important;
    border: 1px solid rgba(0, 240, 255, 0.2) !important;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.05) !important;
    border-radius: 8px !important;
    color: #00ff66 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.verdict-real {
    background: rgba(16, 185, 129, 0.08) !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.1) !important;
    border-radius: 8px !important;
    padding: 18px !important;
}
.verdict-fake {
    background: rgba(239, 68, 68, 0.08) !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    box-shadow: 0 0 15px rgba(239, 68, 68, 0.1) !important;
    border-radius: 8px !important;
    padding: 18px !important;
}
.progress-bar-container {
    background: #1e293b;
    border-radius: 4px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    margin-top: 8px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.progress-bar-fill {
    height: 100%;
    transition: width 0.5s ease-out;
}
.segment-row {
    margin-bottom: 8px;
    padding: 12px 16px;
    border-radius: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.segment-row.fake {
    border-left: 4px solid #ef4444;
    background: rgba(239, 68, 68, 0.03);
}
.segment-row.real {
    border-left: 4px solid #10b981;
    background: rgba(16, 185, 129, 0.03);
}
button.primary {
    background: linear-gradient(135deg, #00f0ff, #0077ff) !important;
    border: none !important;
    color: #080d1a !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.2) !important;
}
button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 25px rgba(0, 240, 255, 0.4) !important;
}
.custom-status-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 15px;
    background: rgba(0, 240, 255, 0.05);
    border: 1px solid rgba(0, 240, 255, 0.15);
    border-radius: 6px;
    margin-bottom: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}
.status-user {
    text-overflow: ellipsis;
    overflow: hidden;
    white-space: nowrap;
    max-width: 250px;
}
.status-quota {
    display: flex;
    align-items: center;
    gap: 15px;
}
.logout-link {
    color: #ef4444 !important;
    text-decoration: none;
    border-bottom: 1px dashed #ef4444;
    transition: all 0.2s ease;
    font-weight: 600;
}
.logout-link:hover {
    color: #f87171 !important;
    border-bottom-style: solid;
}

@media (max-width: 600px) {
    .custom-status-bar {
        flex-direction: column;
        align-items: flex-start;
        gap: 10px;
        padding: 12px;
    }
    .status-user {
        max-width: 100%;
        width: 100%;
    }
    .status-quota {
        width: 100%;
        justify-content: space-between;
        gap: 10px;
    }
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Default(primary_hue="cyan", neutral_hue="slate")) as demo:
    user_status_bar = gr.HTML(value="Checking session status...")
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 25px;">
        <h1 style="font-size: 28px; font-weight: 800; color: #00f0ff; margin: 0; text-transform: uppercase; letter-spacing: 2px;">
            DEEPFENSE AI AUDIO DETECTOR
        </h1>
        <p style="color: #94a3b8; font-size: 14px; margin: 8px 0 0 0;">
            Next-Generation Deepfake & AI Spoofing Diagnostics Engine | Powered by WavLM-Large + Nes2Net
        </p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Diagnostic Control Panel")
            audio_input = gr.Audio(
                label="Upload Audio or Record Voice (Max 12 seconds)",
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            analyze_btn = gr.Button("Run Diagnostics", variant="primary")
            
            gr.Markdown("### Real-time Diagnostic Terminal")
            log_terminal = gr.Textbox(
                label="",
                value="[Ready] Standby for audio upload...",
                interactive=False,
                max_lines=10,
                lines=8,
                elem_classes=["terminal-card"]
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### Inference & Analysis Reports")
            report_output = gr.HTML(
                value="""
                <div class="terminal-card" style="text-align: center; padding: 40px 20px; color: #94a3b8; height: 100%;">
                    Waiting for analysis to start... Upload an audio file and click "Run Diagnostics".
                </div>
                """
            )
            
    analyze_btn.click(
        fn=run_gradio_inference,
        inputs=[audio_input],
        outputs=[log_terminal, report_output, user_status_bar]
    )
    
    demo.load(
        fn=on_page_load,
        inputs=None,
        outputs=[user_status_bar]
    )

    # Register client-side listeners + polling to detect login_success from popup
    demo.load(
        fn=None,
        inputs=[],
        outputs=[],
        js="""() => {
            // 1. BroadcastChannel method
            try {
                const bc = new BroadcastChannel("auth_channel");
                bc.onmessage = (event) => {
                    if (event.data === "login_success") {
                        console.log("[AUTH] BroadcastChannel signal received");
                        window.location.reload();
                    }
                };
            } catch (e) {
                console.error("BroadcastChannel listener registration failed:", e);
            }

            // 2. LocalStorage storage event fallback
            window.addEventListener("storage", (event) => {
                if (event.key === "login_success") {
                    console.log("[AUTH] LocalStorage signal received");
                    window.location.reload();
                }
            });

            // 3. postMessage event listener fallback
            window.addEventListener("message", (event) => {
                if (event.data === "login_success") {
                    console.log("[AUTH] postMessage signal received");
                    window.location.reload();
                }
            });

            // 4. LocalStorage polling fallback — works even in sandboxed iframes
            const pollId = setInterval(() => {
                let hasFlag = false;
                try { hasFlag = localStorage.getItem('login_success') !== null; } catch(e) {}
                if (hasFlag) {
                    console.log("[AUTH] login_success flag detected via polling, reloading...");
                    clearInterval(pollId);
                    try { localStorage.removeItem('login_success'); } catch(e) {}
                    window.location.reload();
                }
            }, 1500);

            // Stop polling after 10 minutes
            setTimeout(() => clearInterval(pollId), 600000);
        }"""
    )

# 8. Mount Gradio interface inside FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
