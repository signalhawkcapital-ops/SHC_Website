# app.py — Signal Hawk Capital (cleaned & rebuilt)
import os, logging, time, sqlite3, smtplib
from datetime import datetime, date, timezone, timedelta
from typing import List
from contextlib import closing
from email.message import EmailMessage
from functools import wraps

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import stripe
import yfinance as yf
from dotenv import load_dotenv
from flask import (Flask, render_template, request, redirect, session,
                   jsonify, url_for, flash)
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from werkzeug.security import generate_password_hash, check_password_hash

load_dotenv()

# ── App & config ──────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("signal-hawk")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
_is_production = os.environ.get("FLASK_ENV") == "production"
_base_url = os.environ.get("APP_BASE_URL", "http://127.0.0.1:5000").rstrip("/")
_use_secure = _is_production and _base_url.startswith("https")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=_use_secure,
    SESSION_COOKIE_PATH="/",
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
)
if not _is_production:
    log.info("Dev mode: SESSION_COOKIE_SECURE=%s", _use_secure)

APP_BASE_URL = _base_url
serializer   = URLSafeTimedSerializer(app.secret_key)

# ── Stripe ────────────────────────────────────
STRIPE_SECRET_KEY     = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_INDIVIDUAL_MONTHLY  = os.environ.get("STRIPE_PRICE_INDIVIDUAL_MONTHLY", "")
STRIPE_PRICE_INDIVIDUAL_YEARLY   = os.environ.get("STRIPE_PRICE_INDIVIDUAL_YEARLY", "")
STRIPE_PRICE_INDIVIDUAL_LIFETIME = os.environ.get("STRIPE_PRICE_INDIVIDUAL_LIFETIME", "")
STRIPE_PRICE_PRO_MONTHLY  = os.environ.get("STRIPE_PRICE_PRO_MONTHLY", "")
STRIPE_PRICE_PRO_YEARLY   = os.environ.get("STRIPE_PRICE_PRO_YEARLY", "")
STRIPE_PRICE_PRO_LIFETIME = os.environ.get("STRIPE_PRICE_PRO_LIFETIME", "")
STRIPE_PRICE_MONTHLY  = os.environ.get("STRIPE_PRICE_MONTHLY", "")
STRIPE_PRICE_YEARLY   = os.environ.get("STRIPE_PRICE_YEARLY", "")
STRIPE_PRICE_LIFETIME = os.environ.get("STRIPE_PRICE_LIFETIME", "")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
else:
    log.warning("STRIPE_SECRET_KEY not set — Stripe Checkout will fail.")

# ── OAuth (optional) ──────────────────────────
OAUTH_AVAILABLE = False
try:
    from authlib.integrations.flask_client import OAuth
    import jwt as pyjwt
    oauth = OAuth(app)
    if os.environ.get("GOOGLE_CLIENT_ID"):
        oauth.register(name="google", client_id=os.environ.get("GOOGLE_CLIENT_ID"),
            client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"})
    def _build_apple_secret():
        t, c, k, p = (os.environ.get(x) for x in
                       ["APPLE_TEAM_ID","APPLE_CLIENT_ID","APPLE_KEY_ID","APPLE_PRIVATE_KEY_PEM"])
        if not all([t, c, k, p]): return None
        now = int(time.time())
        return pyjwt.encode({"iss":t,"iat":now,"exp":now+3600,
            "aud":"https://appleid.apple.com","sub":c}, p,
            algorithm="ES256", headers={"kid": k})
    if os.environ.get("APPLE_CLIENT_ID"):
        oauth.register(name="apple", client_id=os.environ.get("APPLE_CLIENT_ID"),
            client_secret=_build_apple_secret,
            authorize_url="https://appleid.apple.com/auth/authorize",
            access_token_url="https://appleid.apple.com/auth/token",
            client_kwargs={"scope": "name email"})
    OAUTH_AVAILABLE = True
except ImportError:
    log.info("authlib not installed — OAuth disabled.")

# ── Database ──────────────────────────────────
DB_PATH = os.environ.get("DB_PATH", os.environ.get("DATABASE_PATH",
    "/data/signal_hawk.db" if os.path.isdir("/data") else "signal_hawk.db"))

def db_conn():
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row; return conn

def init_db():
    with closing(db_conn()) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,
            phone TEXT, plan TEXT, stripe_customer_id TEXT, subscription_status TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL)''')
        conn.commit()
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        extras = {"password_hash":"TEXT","oauth_provider":"TEXT","oauth_sub":"TEXT",
            "email_alerts":"INTEGER DEFAULT 1","sms_alerts":"INTEGER DEFAULT 1",
            "marketing_emails":"INTEGER DEFAULT 0","alert_frequency":"TEXT DEFAULT 'all'",
            "quiet_hours_start":"TEXT","quiet_hours_end":"TEXT",
            "timezone":"TEXT DEFAULT 'America/New_York'","is_admin":"INTEGER DEFAULT 0"}
        for col, ct in extras.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE users ADD COLUMN {col} {ct}")
        conn.execute("UPDATE users SET email_alerts=1 WHERE email_alerts IS NULL")
        conn.execute("UPDATE users SET sms_alerts=1 WHERE sms_alerts IS NULL")
        conn.execute("UPDATE users SET marketing_emails=0 WHERE marketing_emails IS NULL")
        conn.execute("UPDATE users SET alert_frequency='all' WHERE alert_frequency IS NULL")
        conn.execute("UPDATE users SET timezone='America/New_York' WHERE timezone IS NULL")
        conn.commit()

init_db()

def get_user(email):
    with closing(db_conn()) as conn:
        return conn.execute("SELECT * FROM users WHERE email=?", ((email or "").strip().lower(),)).fetchone()

def upsert_user(email, phone=None, plan=None, stripe_customer_id=None, status=None):
    now = datetime.now(timezone.utc).isoformat()
    with closing(db_conn()) as conn:
        if conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone():
            conn.execute("""UPDATE users SET phone=COALESCE(?,phone), plan=COALESCE(?,plan),
                stripe_customer_id=COALESCE(?,stripe_customer_id),
                subscription_status=COALESCE(?,subscription_status), updated_at=?
                WHERE email=?""", (phone, plan, stripe_customer_id, status, now, email))
        else:
            conn.execute("""INSERT INTO users (email,phone,plan,stripe_customer_id,
                subscription_status,created_at,updated_at) VALUES (?,?,?,?,?,?,?)""",
                (email, phone, plan, stripe_customer_id, status, now, now))
        conn.commit()

def update_user(email, **fields):
    if not fields: return
    email = (email or "").strip().lower()
    keys = list(fields.keys()); vals = [fields[k] for k in keys]
    now = datetime.now(timezone.utc).isoformat()
    sets = ", ".join(f"{k}=?" for k in keys) + ", updated_at=?"
    with closing(db_conn()) as conn:
        conn.execute(f"UPDATE users SET {sets} WHERE email=?", (*vals, now, email))
        conn.commit()

def set_user_password(email, raw_password):
    update_user(email, password_hash=generate_password_hash(raw_password))

# ── Email ─────────────────────────────────────
def send_email(to_email, subject, html, text=None):
    host, port = os.environ.get("SMTP_HOST"), int(os.environ.get("SMTP_PORT", "587"))
    username, password = os.environ.get("SMTP_USERNAME"), os.environ.get("SMTP_PASSWORD")
    from_addr = os.environ.get("EMAIL_FROM", username)
    if not all([host, username, password]):
        raise RuntimeError("SMTP env vars missing.")
    msg = EmailMessage(); msg["Subject"]=subject; msg["From"]=from_addr; msg["To"]=to_email
    msg.set_content(text or "Please view this email in an HTML-capable client.")
    msg.add_alternative(html, subtype="html")
    with smtplib.SMTP(host, port) as s:
        s.starttls(); s.login(username, password); s.send_message(msg)

# ── Tokens ────────────────────────────────────
def make_setup_token(email):
    return serializer.dumps({"email": email}, salt="account-setup")
def read_setup_token(token, max_age=86400):
    return (serializer.loads(token, salt="account-setup", max_age=max_age).get("email") or "").strip().lower()
def make_reset_token(email):
    return serializer.dumps({"email": email}, salt="password-reset")
def read_reset_token(token, max_age=3600):
    return (serializer.loads(token, salt="password-reset", max_age=max_age).get("email") or "").strip().lower()

# ── Auth decorator ────────────────────────────
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("user_email"):
            return redirect(url_for("login", msg="Please log in to continue."))
        return f(*args, **kwargs)
    return wrapper

# ═══════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════

# ── Public pages ──────────────────────────────
@app.before_request
def inject_user():
    """Make current user available to all templates for conditional nav."""
    from flask import g
    email = session.get("user_email")
    g.logged_in = bool(email)
    g.user_email = email

# Temporary debug endpoint — remove before production deploy
if not _is_production:
    @app.route("/debug/session")
    def debug_session():
        return jsonify({
            "session": dict(session),
            "cookie_secure": app.config.get("SESSION_COOKIE_SECURE"),
            "cookie_samesite": app.config.get("SESSION_COOKIE_SAMESITE"),
            "cookie_path": app.config.get("SESSION_COOKIE_PATH"),
            "flask_env": os.environ.get("FLASK_ENV"),
            "is_production": _is_production,
            "secret_key_set": bool(app.secret_key and app.secret_key != "dev-secret-change-me"),
        })

@app.route("/")
def index():
    return render_template("home.html", page_title="Signal Hawk Capital")

@app.route("/about")
def about():
    return render_template("about.html", page_title="About")

@app.route("/resources")
def resources():
    return render_template("resources.html", page_title="Resources")

@app.route("/signup")
def signup():
    return render_template("signup.html", page_title="Subscribe")

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/disclosures")
def disclosures():
    return render_template("disclosures.html", page_title="Disclosures")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html", page_title="Privacy Policy")

@app.route("/set-password", methods=["GET", "POST"])
def set_password():
    """Alias for account setup — handles the set-password link from emails."""
    return account_setup()

# ── Auth ──────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html", msg=request.args.get("msg", ""))
    email    = (request.form.get("email") or "").strip().lower()
    password = (request.form.get("password") or "")
    if not email or not password:
        return render_template("login.html", msg="Enter email and password.")
    u = get_user(email)
    if not u:
        return render_template("login.html", msg="No account found. Please set up your account first.")
    if not u["password_hash"]:
        return render_template("login.html", msg="No password set. Use the setup link or reset your password.")
    if not check_password_hash(u["password_hash"], password):
        return render_template("login.html", msg="Incorrect password.")
    session.clear()
    session.permanent = True
    session["user_email"] = email
    log.info("Login successful for %s — session: %s", email, dict(session))
    return redirect(url_for("account"))

@app.route("/logout")
def logout():
    session.clear(); return redirect("/")

# ── Forgot / Reset password ──────────────────
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "GET":
        return render_template("forgot_password.html")
    email = (request.form.get("email") or "").strip().lower()
    if email and get_user(email):
        token = make_reset_token(email)
        link  = f"{APP_BASE_URL}/reset-password?token={token}"
        try:
            send_email(email, "Reset Your Signal Hawk Password",
                f'''<div style="font-family:'Helvetica Neue',Arial,sans-serif;max-width:560px;margin:0 auto;color:#e2e8f0;background:#0f172a;border-radius:12px;overflow:hidden;">
                <div style="background:linear-gradient(135deg,#1e3a5f,#0f172a);padding:32px 28px 20px;text-align:center;">
                  <h1 style="margin:0;font-size:22px;color:#fff;font-weight:700;">Signal Hawk Capital</h1>
                </div>
                <div style="padding:28px;">
                  <p style="margin:0 0 16px;font-size:15px;">Hi{" " + email.split("@")[0].title() if email else ""},</p>
                  <p style="margin:0 0 20px;font-size:15px;">We received a request to reset your password. Click below to choose a new one:</p>
                  <div style="text-align:center;margin:24px 0;">
                    <a href="{link}" style="display:inline-block;padding:14px 28px;border-radius:10px;background:#3b82f6;color:#fff;text-decoration:none;font-weight:600;font-size:15px;">Reset Password</a>
                  </div>
                  <p style="margin:0 0 16px;font-size:13px;color:#94a3b8;">This link expires in 1&nbsp;hour. If you didn't request this, you can safely ignore this email.</p>
                  <hr style="border:none;border-top:1px solid #1e293b;margin:20px 0;"/>
                  <p style="margin:0;font-size:11px;color:#64748b;line-height:1.6;">Signal Hawk Capital, LLC &mdash; Educational information only.</p>
                </div>
                </div>''')
        except Exception: log.exception("Reset email failed")
    return render_template("forgot_password.html",
        msg="If an account exists for that email, a reset link has been sent.")

@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    if request.method == "GET":
        token = (request.args.get("token") or "").strip()
        if not token: return ("Missing reset token.", 400)
        try: email = read_reset_token(token)
        except SignatureExpired: return ("Reset link expired.", 400)
        except BadSignature: return ("Invalid reset link.", 400)
        return render_template("reset_password_confirm.html", email=email, token=token)
    token = (request.form.get("token") or request.args.get("token") or "").strip()
    pw  = request.form.get("password") or ""
    pw2 = request.form.get("password_confirm") or request.form.get("password2") or ""
    if not token or not pw: return ("Missing token or password.", 400)
    if pw != pw2:
        return render_template("reset_password_confirm.html", token=token, email="", error="Passwords do not match.")
    if len(pw) < 10:
        return render_template("reset_password_confirm.html", token=token, email="", error="Password must be at least 10 characters.")
    try: email = read_reset_token(token)
    except Exception: return ("Invalid or expired token.", 400)
    set_user_password(email, pw)
    session.clear()
    session.permanent = True
    session["user_email"] = email
    return redirect(url_for("account"))

# ── Account setup ─────────────────────────────
@app.route("/account/request-setup", methods=["GET", "POST"])
def request_setup():
    if request.method == "GET":
        return render_template("account_request_setup.html")
    email = (request.form.get("email") or "").strip().lower()
    if not email:
        flash("Email is required.", "error"); return redirect(url_for("request_setup"))
    if not get_user(email):
        upsert_user(email=email, status="pending")
    token = make_setup_token(email)
    link  = url_for("account_setup", token=token, _external=True)
    try:
        send_email(email, "Complete Your Signal Hawk Capital Account Setup",
            f'''<div style="font-family:'Helvetica Neue',Arial,sans-serif;max-width:560px;margin:0 auto;color:#e2e8f0;background:#0f172a;border-radius:12px;overflow:hidden;">
            <div style="background:linear-gradient(135deg,#1e3a5f,#0f172a);padding:32px 28px 20px;text-align:center;">
              <h1 style="margin:0;font-size:22px;color:#fff;font-weight:700;">Signal Hawk Capital</h1>
            </div>
            <div style="padding:28px;">
              <p style="margin:0 0 16px;font-size:15px;">Hi{" " + email.split("@")[0].title() if email else ""},</p>
              <p style="margin:0 0 20px;font-size:15px;">You requested a link to set up your account. Click below to create your password and get started:</p>
              <div style="text-align:center;margin:24px 0;">
                <a href="{link}" style="display:inline-block;padding:14px 28px;border-radius:10px;background:#3b82f6;color:#fff;text-decoration:none;font-weight:600;font-size:15px;">Set Up Your Account</a>
              </div>
              <p style="margin:0 0 16px;font-size:13px;color:#94a3b8;">This link expires in 24&nbsp;hours. If you didn't request this, you can safely ignore this email.</p>
              <hr style="border:none;border-top:1px solid #1e293b;margin:20px 0;"/>
              <p style="margin:0;font-size:11px;color:#64748b;line-height:1.6;">Signal Hawk Capital, LLC &mdash; Educational information only. Investing involves risk.</p>
            </div>
            </div>''')
        flash("Setup link sent — check your email.", "success")
    except Exception:
        log.exception("Setup email failed"); flash("Could not send email.", "error")
    return redirect(url_for("login"))

@app.route("/account/setup", methods=["GET", "POST"])
def account_setup():
    if request.method == "GET":
        token = (request.args.get("token") or "").strip()
        if not token: return ("Missing setup token.", 400)
        try: email = read_setup_token(token)
        except SignatureExpired: return ("Setup link expired.", 400)
        except BadSignature: return ("Invalid setup link.", 400)
        return render_template("account_setup.html", token=token, email=email)
    token = (request.form.get("token") or "").strip()
    phone = (request.form.get("phone") or "").strip()
    pw    = request.form.get("password") or ""
    pw2   = request.form.get("confirm_password") or ""
    if not token:
        flash("Missing setup token.", "error"); return redirect(url_for("login"))
    try: email = read_setup_token(token)
    except Exception:
        flash("Invalid or expired setup link.", "error"); return redirect(url_for("request_setup"))
    if pw != pw2:
        flash("Passwords do not match.", "error")
        return render_template("account_setup.html", token=token, email=email)
    if len(pw) < 10:
        flash("Password must be at least 10 characters.", "error")
        return render_template("account_setup.html", token=token, email=email)
    with closing(db_conn()) as conn:
        conn.execute("UPDATE users SET phone=?, password_hash=? WHERE email=?",
                     (phone, generate_password_hash(pw), email))
        conn.commit()
    session.clear()
    session.permanent = True
    session["user_email"] = email
    flash("Account setup complete!", "success")
    return redirect(url_for("account"))

# ── Account dashboard ─────────────────────────
@app.route("/account")
@login_required
def account():
    u = get_user(session["user_email"])
    if not u:
        session.clear(); return redirect(url_for("login", msg="Account not found."))
    return render_template("account.html", user=u)

@app.route("/account/profile", methods=["POST"])
@login_required
def account_profile_update():
    email = session["user_email"]
    new_email = (request.form.get("email") or "").strip().lower()
    new_phone = (request.form.get("phone") or "").strip()
    if not new_email or "@" not in new_email:
        flash("Please enter a valid email.", "error"); return redirect(url_for("account"))
    try:
        with closing(db_conn()) as conn:
            conn.execute("UPDATE users SET email=?, phone=?, updated_at=? WHERE email=?",
                         (new_email, new_phone, datetime.now(timezone.utc).isoformat(), email))
            conn.commit()
        session["user_email"] = new_email; flash("Profile updated.", "success")
    except Exception: flash("That email may already be in use.", "error")
    return redirect(url_for("account"))

@app.route("/account/notifications", methods=["POST"])
@login_required
def account_notifications_update():
    update_user(session["user_email"],
        email_alerts=1 if request.form.get("email_alerts")=="on" else 0,
        sms_alerts=1 if request.form.get("sms_alerts")=="on" else 0,
        marketing_emails=1 if request.form.get("marketing_emails")=="on" else 0)
    flash("Notification preferences saved.", "success")
    return redirect(url_for("account"))

@app.route("/account/password", methods=["POST"])
@login_required
def account_password_update():
    pw1, pw2 = request.form.get("password",""), request.form.get("password2","")
    if len(pw1) < 10:
        flash("Password must be at least 10 characters.", "error"); return redirect(url_for("account"))
    if pw1 != pw2:
        flash("Passwords do not match.", "error"); return redirect(url_for("account"))
    set_user_password(session["user_email"], pw1)
    flash("Password updated.", "success")
    return redirect(url_for("account"))

@app.route("/account/billing-portal", methods=["POST"])
@login_required
def account_billing_portal():
    with closing(db_conn()) as conn:
        u = conn.execute("SELECT stripe_customer_id FROM users WHERE email=?",
                         (session["user_email"],)).fetchone()
    if not u or not u["stripe_customer_id"]:
        return redirect("/signup?msg=Please+subscribe+to+manage+billing")
    portal = stripe.billing_portal.Session.create(
        customer=u["stripe_customer_id"],
        return_url=url_for("account", _external=True))
    return redirect(portal.url)

# ── OAuth ─────────────────────────────────────
if OAUTH_AVAILABLE:
    def _oauth_upsert(email, provider, sub):
        with closing(db_conn()) as conn:
            if conn.execute("SELECT email FROM users WHERE email=?", (email,)).fetchone():
                conn.execute("UPDATE users SET oauth_provider=?, oauth_sub=? WHERE email=?",
                             (provider, sub, email))
            else:
                now = datetime.now(timezone.utc).isoformat()
                conn.execute("""INSERT INTO users (email,oauth_provider,oauth_sub,
                    subscription_status,created_at,updated_at) VALUES (?,?,?,?,?,?)""",
                    (email, provider, sub, "inactive", now, now))
            conn.commit()

    @app.route("/auth/google")
    def auth_google():
        return oauth.google.authorize_redirect(url_for("auth_google_callback", _external=True))

    @app.route("/auth/google/callback")
    def auth_google_callback():
        token = oauth.google.authorize_access_token()
        info  = token.get("userinfo") or oauth.google.parse_id_token(token)
        email = (info.get("email") or "").strip().lower()
        if not email: return "Google login failed: no email.", 400
        _oauth_upsert(email, "google", info.get("sub"))
        session["user_email"] = email
        return redirect(url_for("account"))

    @app.route("/auth/apple")
    def auth_apple():
        return oauth.apple.authorize_redirect(url_for("auth_apple_callback", _external=True))

    @app.route("/auth/apple/callback")
    def auth_apple_callback():
        token = oauth.apple.authorize_access_token()
        id_token = token.get("id_token")
        if not id_token: return "Apple login failed.", 400
        claims = pyjwt.decode(id_token, options={"verify_signature": False})
        email, sub = (claims.get("email") or "").strip().lower(), claims.get("sub")
        if not email:
            with closing(db_conn()) as conn:
                row = conn.execute("SELECT email FROM users WHERE oauth_provider=? AND oauth_sub=?",
                                   ("apple", sub)).fetchone()
                if not row: return "Apple login failed: no email on file.", 400
                email = row["email"]
        _oauth_upsert(email, "apple", sub)
        session["user_email"] = email
        return redirect(url_for("account"))

# ── Stripe Checkout ───────────────────────────
def _get_price_id(investor_type, plan):
    inv = "professional" if (investor_type or "").lower().strip() in ("pro","professional") else "individual"
    plan = (plan or "").lower().strip()
    if plan not in ("monthly","yearly","lifetime"): return ""
    m = ({"monthly":STRIPE_PRICE_PRO_MONTHLY,"yearly":STRIPE_PRICE_PRO_YEARLY,
          "lifetime":STRIPE_PRICE_PRO_LIFETIME} if inv=="professional" else
         {"monthly":STRIPE_PRICE_INDIVIDUAL_MONTHLY,"yearly":STRIPE_PRICE_INDIVIDUAL_YEARLY,
          "lifetime":STRIPE_PRICE_INDIVIDUAL_LIFETIME})
    return m.get(plan) or {"monthly":STRIPE_PRICE_MONTHLY,"yearly":STRIPE_PRICE_YEARLY,
                           "lifetime":STRIPE_PRICE_LIFETIME}.get(plan,"")

@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    if not STRIPE_SECRET_KEY: return ("Stripe not configured", 500)
    data = request.get_json(silent=True) or {}
    if not data.get("accepted_terms"):
        return ("You must accept the Terms/Disclosures/Privacy Policy.", 400)
    email = (data.get("email") or "").strip().lower()
    phone = (data.get("phone") or "").strip()
    plan  = (data.get("plan") or "").strip().lower()
    inv   = data.get("investorType") or data.get("investor_type") or "individual"
    if not email or not phone or not plan:
        return ("Missing email, phone, or plan", 400)
    price_id = _get_price_id(inv, plan)
    if not price_id: return ("Stripe price not configured.", 500)
    try: upsert_user(email=email, phone=phone, plan=plan, status="initiated")
    except Exception as e: log.warning("upsert during checkout: %s", e)
    try:
        checkout = stripe.checkout.Session.create(
            mode="subscription" if plan in ("monthly","yearly") else "payment",
            line_items=[{"price": price_id, "quantity": 1}],
            customer_email=email, phone_number_collection={"enabled": True},
            allow_promotion_codes=True,
            metadata={"alerts_email":email,"alerts_phone":phone,"plan":plan,
                "terms_accepted":"true","terms_accepted_at":datetime.now(timezone.utc).isoformat(),
                "terms_accepted_ip":request.headers.get("X-Forwarded-For",request.remote_addr) or ""},
            success_url=f"{APP_BASE_URL}/?checkout=success",
            cancel_url=f"{APP_BASE_URL}/signup?checkout=canceled")
        return jsonify({"url": checkout.url})
    except stripe.error.StripeError as e:
        log.exception("Stripe error"); return (getattr(e,"user_message",None) or str(e), 500)

@app.post("/stripe-webhook")
def stripe_webhook():
    try:
        event = stripe.Webhook.construct_event(
            request.data, request.headers.get("Stripe-Signature",""), STRIPE_WEBHOOK_SECRET)
    except Exception as e: return str(e), 400
    if event["type"] == "checkout.session.completed":
        obj = event["data"]["object"]
        email = (obj.get("customer_details") or {}).get("email")
        cid   = obj.get("customer")
        if email:
            email = email.strip().lower()
            upsert_user(email=email, plan="unknown", stripe_customer_id=cid, status="active")
            token = make_setup_token(email)
            link  = f"{APP_BASE_URL}/account/setup?token={token}"
            try:
                send_email(email, "Welcome to Signal Hawk Capital — Complete Your Account Setup",
                    f'''<div style="font-family:'Helvetica Neue',Arial,sans-serif;max-width:560px;margin:0 auto;color:#e2e8f0;background:#0f172a;border-radius:12px;overflow:hidden;">
                    <div style="background:linear-gradient(135deg,#1e3a5f,#0f172a);padding:32px 28px 20px;text-align:center;">
                      <h1 style="margin:0;font-size:22px;color:#fff;font-weight:700;">Welcome to Signal Hawk Capital</h1>
                    </div>
                    <div style="padding:28px;">
                      <p style="margin:0 0 16px;font-size:15px;">Hi{" " + email.split("@")[0].title() if email else ""},</p>
                      <p style="margin:0 0 16px;font-size:15px;">Congratulations on subscribing! Your subscription is now <b style="color:#22c55e;">active</b>, and you're one step away from accessing real-time buy &amp; sell alerts on our S&amp;P 500 and SPXL strategies.</p>
                      <p style="margin:0 0 20px;font-size:15px;">Click below to set your password and finish setting up your account:</p>
                      <div style="text-align:center;margin:24px 0;">
                        <a href="{link}" style="display:inline-block;padding:14px 28px;border-radius:10px;background:#3b82f6;color:#fff;text-decoration:none;font-weight:600;font-size:15px;">Set Up Your Account</a>
                      </div>
                      <p style="margin:0 0 20px;font-size:13px;color:#94a3b8;">This link expires in 24&nbsp;hours. If it expires, you can request a new one from the <a href="{APP_BASE_URL}/login" style="color:#60a5fa;">login page</a>.</p>
                      <hr style="border:none;border-top:1px solid #1e293b;margin:20px 0;"/>
                      <p style="margin:0 0 8px;font-size:13px;color:#94a3b8;"><b style="color:#cbd5e1;">What to expect next:</b></p>
                      <p style="margin:0 0 4px;font-size:13px;color:#94a3b8;">&bull; Set your password and configure alert preferences</p>
                      <p style="margin:0 0 4px;font-size:13px;color:#94a3b8;">&bull; Receive email and SMS buy/sell signal alerts</p>
                      <p style="margin:0 0 16px;font-size:13px;color:#94a3b8;">&bull; Access the Investment Growth Analyzer</p>
                      <hr style="border:none;border-top:1px solid #1e293b;margin:20px 0;"/>
                      <p style="margin:0;font-size:11px;color:#64748b;line-height:1.6;">Signal Hawk Capital, LLC provides educational information only and is not a registered investment adviser. Past performance is not indicative of future results. Investing involves risk, including loss of principal.</p>
                    </div>
                    </div>''')
            except Exception: log.exception("Onboarding email failed")
    return "", 200

@app.route("/resend-setup", methods=["POST"])
def resend_setup():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    if not email: return jsonify({"error": "Email required"}), 400
    if not get_user(email): return jsonify({"error": "No account found"}), 404
    token = make_setup_token(email)
    link  = url_for("account_setup", token=token, _external=True)
    try:
        send_email(email, "Complete Your Signal Hawk Capital Account Setup",
            f'''<div style="font-family:'Helvetica Neue',Arial,sans-serif;max-width:560px;margin:0 auto;color:#e2e8f0;background:#0f172a;border-radius:12px;overflow:hidden;">
            <div style="background:linear-gradient(135deg,#1e3a5f,#0f172a);padding:32px 28px 20px;text-align:center;">
              <h1 style="margin:0;font-size:22px;color:#fff;font-weight:700;">Signal Hawk Capital</h1>
            </div>
            <div style="padding:28px;">
              <p style="margin:0 0 16px;font-size:15px;">Hi{" " + email.split("@")[0].title() if email else ""},</p>
              <p style="margin:0 0 20px;font-size:15px;">Here's your account setup link. Click below to create your password:</p>
              <div style="text-align:center;margin:24px 0;">
                <a href="{link}" style="display:inline-block;padding:14px 28px;border-radius:10px;background:#3b82f6;color:#fff;text-decoration:none;font-weight:600;font-size:15px;">Set Up Your Account</a>
              </div>
              <p style="margin:0;font-size:13px;color:#94a3b8;">This link expires in 24&nbsp;hours.</p>
            </div>
            </div>''')
    except Exception:
        log.exception("Resend failed"); return jsonify({"error": "Email delivery failed"}), 500
    return jsonify({"ok": True})

# ═══════════════════════════════════════════════
#  INVESTMENT ANALYZER
# ═══════════════════════════════════════════════
TICKER_META = {
    "SPXL":{"label":"SPXL Buy & Hold","er":0.0095},
    "SPY":{"label":"S&P 500 (SPY) Buy & Hold","er":0.0009},
    "^GSPC":{"label":"S&P 500 Index Buy & Hold","er":0.0},
    "NVDA":{"label":"NVDA Buy & Hold","er":0.0},
    "AMZN":{"label":"AMZN Buy & Hold","er":0.0},
    "AAPL":{"label":"AAPL Buy & Hold","er":0.0},
    "TSLA":{"label":"TSLA Buy & Hold","er":0.0},
}

def _download_close(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex): df = df["Close"]
    if isinstance(df, pd.Series): df = df.to_frame()
    return df.sort_index().ffill()

def _expense_drag(s, r):
    if r <= 0 or s.empty: return s
    d = (1-r)**(1/252); ret = s.pct_change().fillna(0.0)
    out = ((1+ret)*d).cumprod(); return out*(s.iloc[0]/out.iloc[0])

def _equity(price, init, inc_er, er, inc_tax, tax=0.2):
    px = price.copy().dropna()
    if px.empty: return px
    if inc_er and er > 0: px = _expense_drag(px, er)
    ret = px.pct_change().fillna(0.0)
    if inc_tax: ret = pd.Series(np.where(ret>0, ret*(1-tax), ret), index=px.index)
    return (1+ret).cumprod() * (init or 10000.0)

def _sh_equity(spx, spxl, init, inc_er, inc_tax):
    sl = spxl.copy().dropna()
    if sl.empty: return sl
    if inc_er: sl = _expense_drag(sl, 0.0095)
    sx = spx.reindex_like(sl).ffill()
    sig = (sx > sx.rolling(80).mean()).astype(int).shift(1).fillna(0)
    ret = sl.pct_change().fillna(0.0) * sig
    if inc_tax: ret = pd.Series(np.where(ret>0, ret*0.8, ret), index=sl.index)
    eq = (1+ret).cumprod() * (init or 10000.0); eq.name = "Signal Hawk Strategy"; return eq

@app.route("/analyzer", methods=["GET","POST"])
def analyzer():
    today = date.today()
    start = request.values.get("start", "2010-01-01")
    try: start_date = datetime.strptime(start, "%Y-%m-%d").date()
    except ValueError: start_date = date(2010, 1, 1)
    initial = float(request.values.get("initial", 10000))
    inc_er  = request.values.get("include_er", "on") == "on"
    inc_tax = request.values.get("include_taxes", "off") == "on"
    defaults = {"series_SH":"on","series_HG_SIM":"on","series_SPXL":"on",
        "series_SP500":"on","series_NVDA":"off","series_AMZN":"off",
        "series_AAPL":"off","series_TSLA":"off"}
    sf = {k: request.values.get(k, defaults[k])=="on" for k in defaults}

    # ── Fetch market data (may fail due to rate limits) ──
    try:
        need = ["SPXL","SPY","^GSPC"]
        for sym,flag in [("NVDA","series_NVDA"),("AMZN","series_AMZN"),
                         ("AAPL","series_AAPL"),("TSLA","series_TSLA")]:
            if sf[flag]: need.append(sym)
        df = _download_close(sorted(set(need)), start_date, today)
        if df.empty:
            raise ValueError("No market data returned.")
    except Exception as e:
        log.warning("Analyzer data fetch failed: %s", e)
        return render_template("growth_analyzer.html", page_title="Investment Growth Analyzer",
            price_graph="", message="⚠ Market data temporarily unavailable — please try again in a moment.",
            start=start_date.strftime("%Y-%m-%d"), initial=initial,
            include_er=inc_er, include_taxes=inc_tax, series_flags=sf, table_rows=[])

    def c(n):
        s = df.get(n); return s.iloc[:,0] if isinstance(s, pd.DataFrame) else s
    spx  = c("^GSPC") if "^GSPC" in df.columns else c("SPY")
    spxl, spy = c("SPXL"), c("SPY")
    curves = {}
    if sf["series_SH"] and spx is not None and spxl is not None:
        curves["Signal Hawk Strategy"] = _sh_equity(spx, spxl, initial, inc_er, inc_tax)
    if sf["series_HG_SIM"] and spx is not None and spy is not None:
        sig = (spx > spx.rolling(80).mean()).astype(int).shift(1).fillna(0)
        r = spy.pct_change().fillna(0.0)*2.5
        if inc_er: r = pd.Series(np.where(r>0, r*(1-0.0095/0.4), r), index=spy.index)
        if inc_tax: r = pd.Series(np.where(r>0, r*0.8, r), index=spy.index)
        curves["HG_SIM Strategy"] = (1+r*sig).cumprod()*initial
    if sf["series_SPXL"] and spxl is not None:
        curves["SPXL Buy & Hold"] = _equity(spxl, initial, inc_er, 0.0095, inc_tax)
    src = "^GSPC" if "^GSPC" in df.columns else "SPY"
    if sf["series_SP500"] and src in df.columns:
        lbl = "S&P 500 (SPY) Buy & Hold" if src=="SPY" else "S&P 500 Index Buy & Hold"
        curves[lbl] = _equity(c(src), initial, inc_er, 0.0009 if src=="SPY" else 0, inc_tax)
    for sym,flag in [("NVDA","series_NVDA"),("AMZN","series_AMZN"),
                     ("AAPL","series_AAPL"),("TSLA","series_TSLA")]:
        if sf[flag] and sym in df.columns:
            curves[f"{sym} Buy & Hold"] = _equity(c(sym), initial, inc_er, 0.0, inc_tax)
    colors = {"Signal Hawk Strategy":"#3b82f6","HG_SIM Strategy":"#8b5cf6",
        "SPXL Buy & Hold":"#f97316","S&P 500 (SPY) Buy & Hold":"#22c55e",
        "S&P 500 Index Buy & Hold":"#22c55e","NVDA Buy & Hold":"#06b6d4",
        "AMZN Buy & Hold":"#a855f7","AAPL Buy & Hold":"#64748b","TSLA Buy & Hold":"#ec4899"}
    # Remove any empty series that slipped through
    curves = {k: v for k, v in curves.items() if v is not None and not v.empty}
    fig = go.Figure()
    for n,s in curves.items():
        fig.add_trace(go.Scatter(x=s.index,y=s,mode="lines",name=n,
            line=dict(width=2,color=colors.get(n))))
    fig.update_layout(title="Investment Growth Over Time",xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",height=600,template="plotly_dark",
        margin=dict(l=30,r=20,t=40,b=30),legend=dict(bgcolor="rgba(0,0,0,0)"))
    pg = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    rows = []
    for n,s in curves.items():
        y = (s.index[-1]-s.index[0]).days/365.25
        cagr = (s.iloc[-1]/s.iloc[0])**(1/y)-1 if y>0 else float('nan')
        mdd = (s/s.cummax()-1).min()
        rows.append({"name":n,"cagr":f"{cagr*100:.2f}%","mdd":f"{mdd*100:.2f}%","final":f"${s.iloc[-1]:,.0f}"})
    msg = f"If you invested ${initial:,.2f} on {start_date.strftime('%m/%d/%Y')} and used the Signal Hawk buy & sell alerts your account value would be "
    msg += f"${curves['Signal Hawk Strategy'].iloc[-1]:,.2f} by {today.strftime('%m/%d/%Y')}.*" if "Signal Hawk Strategy" in curves else "— (insufficient data).*"
    return render_template("growth_analyzer.html", page_title="Investment Growth Analyzer",
        price_graph=pg, message=msg, start=start_date.strftime("%Y-%m-%d"), initial=initial,
        include_er=inc_er, include_taxes=inc_tax, series_flags=sf, table_rows=rows)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG","0")=="1")
