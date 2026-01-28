# app.py
import os, logging
import stripe
from datetime import datetime, date
from typing import Dict, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from werkzeug.security import generate_password_hash, check_password_hash

from flask import session, flash
import smtplib
from email.message import EmailMessage
from datetime import datetime
from contextlib import closing

import stripe
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, session, jsonify
from flask import Flask, render_template, request, jsonify, url_for
from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("signal-hawk")

# Stripe (configured via environment variables / .env)

import sqlite3

DB_PATH = os.environ.get("DB_PATH", "signal_hawk.db")

def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_users_columns():
    """
    Ensures the users table has the columns needed for auth.
    Safe to run on startup (it only adds missing columns).
    """
    cols = {
        "password_hash": "TEXT",
        "oauth_provider": "TEXT",
        "oauth_sub": "TEXT",
    }
    with db_conn() as conn:
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        for c, ctype in cols.items():
            if c not in existing:
                conn.execute(f"ALTER TABLE users ADD COLUMN {c} {ctype}")
        conn.commit()

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get('SESSION_SECRET', 'dev-secret-change-me')
ensure_users_columns()
app = Flask(__name__)
import os
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_change_me")

APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://127.0.0.1:5000").rstrip("/")

serializer = URLSafeTimedSerializer(app.secret_key)

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")



# ---------- Stripe configuration ----------
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://127.0.0.1:5000").rstrip("/")

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

# New pricing scheme (preferred)
STRIPE_PRICE_INDIVIDUAL_MONTHLY  = os.environ.get("STRIPE_PRICE_INDIVIDUAL_MONTHLY", "")
STRIPE_PRICE_INDIVIDUAL_YEARLY   = os.environ.get("STRIPE_PRICE_INDIVIDUAL_YEARLY", "")
STRIPE_PRICE_INDIVIDUAL_LIFETIME = os.environ.get("STRIPE_PRICE_INDIVIDUAL_LIFETIME", "")
STRIPE_PRICE_PRO_MONTHLY         = os.environ.get("STRIPE_PRICE_PRO_MONTHLY", "")
STRIPE_PRICE_PRO_YEARLY          = os.environ.get("STRIPE_PRICE_PRO_YEARLY", "")
STRIPE_PRICE_PRO_LIFETIME        = os.environ.get("STRIPE_PRICE_PRO_LIFETIME", "")

# Legacy fallback (older env naming)
STRIPE_PRICE_MONTHLY  = os.environ.get("STRIPE_PRICE_MONTHLY", "")
STRIPE_PRICE_YEARLY   = os.environ.get("STRIPE_PRICE_YEARLY", "")
STRIPE_PRICE_LIFETIME = os.environ.get("STRIPE_PRICE_LIFETIME", "")

from authlib.integrations.flask_client import OAuth

oauth = OAuth(app)

oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
else:
    log.warning("WARNING: STRIPE_SECRET_KEY not set – Stripe Checkout will fail.")

def init_db():
    with closing(db_connect()) as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            plan TEXT,
            stripe_customer_id TEXT,
            subscription_status TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')
        conn.commit()

        # Add password_hash column if it doesn't exist yet
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
        if "password_hash" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
            conn.commit()

def upsert_user(email: str, phone: str | None, plan: str | None, stripe_customer_id: str | None, status: str | None):
    now = datetime.utcnow().isoformat()

    with closing(db_connect()) as conn:
        existing = conn.execute("SELECT id, created_at FROM users WHERE email=?", (email,)).fetchone()

        if existing:
            conn.execute("""
                UPDATE users
                SET phone = COALESCE(?, phone),
                    plan = COALESCE(?, plan),
                    stripe_customer_id = COALESCE(?, stripe_customer_id),
                    subscription_status = COALESCE(?, subscription_status),
                    updated_at = ?
                WHERE email = ?
            """, (phone, plan, stripe_customer_id, status, now, email))
        else:
            conn.execute("""
                INSERT INTO users (email, phone, plan, stripe_customer_id, subscription_status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (email, phone, plan, stripe_customer_id, status, now, now))

        conn.commit()

def get_user_by_email(email: str):
    with closing(db_connect()) as conn:
        return conn.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        ).fetchone()


def set_user_password(email: str, raw_password: str):
    now = datetime.utcnow().isoformat()
    pw_hash = generate_password_hash(raw_password)

    with closing(db_connect()) as conn:
        conn.execute(
            "UPDATE users SET password_hash=?, updated_at=? WHERE email=?",
            (pw_hash, now, email)
        )
        conn.commit()

def send_email(to_email: str, subject: str, html: str, text: str = None):
    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "587"))
    username = os.environ.get("SMTP_USERNAME")
    password = os.environ.get("SMTP_PASSWORD")
    from_addr = os.environ.get("EMAIL_FROM", username)

    if not all([host, username, password]):
        raise RuntimeError("SMTP env vars missing (SMTP_HOST/SMTP_USERNAME/SMTP_PASSWORD).")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_email

    if text:
        msg.set_content(text)
    else:
        msg.set_content("Please view this email in an HTML-capable email client.")

    msg.add_alternative(html, subtype="html")

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(msg)

def make_setup_token(email: str) -> str:
    return serializer.dumps({"email": email}, salt="account-setup")

def read_setup_token(token: str, max_age_seconds: int = 60 * 60 * 24) -> str:
    data = serializer.loads(token, salt="account-setup", max_age=max_age_seconds)
    return (data.get("email") or "").strip().lower()

def db_connect():
    conn = sqlite3.connect("signal_hawk.db")
    conn.row_factory = sqlite3.Row
    return conn

def get_user_by_email(email: str):
    with closing(db_connect()) as conn:
        return conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()

def set_user_password(email: str, raw_password: str):
    pw_hash = generate_password_hash(raw_password)
    now = datetime.utcnow().isoformat()
    with closing(db_connect()) as conn:
        conn.execute("UPDATE users SET password_hash=?, updated_at=? WHERE email=?",
                     (pw_hash, now, email))
        conn.commit()

@app.route("/test-email")
def test_email():
    send_email(
        "gabeevans2012@gmail.com",
        "Signal Hawk Test Email",
        "<p>If you received this, Gmail SMTP is working 🎉</p>"
    )
    return "Email sent"


@app.route("/account/setup", methods=["GET", "POST"], strict_slashes=False)
def account_setup():
    if request.method == "GET":
        token = request.args.get("token", "").strip()
        if not token:
            return ("Missing setup token.", 400)

        try:
            email = read_setup_token(token)
        except SignatureExpired:
            return ("Setup link expired. Please request a new one.", 400)
        except BadSignature:
            return ("Invalid setup link.", 400)

        return render_template("account_setup.html", email=email, token=token)

    # POST
    token = request.form.get("token", "").strip()
    phone = (request.form.get("phone") or "").strip()
    password = (request.form.get("password") or "").strip()

    if not token or not password:
        return ("Missing token or password.", 400)

    try:
        email = read_setup_token(token)
    except Exception:
        return ("Invalid or expired token.", 400)

    # Ensure user exists (Stripe webhook should create it, but this is safe)
    u = get_user_by_email(email)
    if not u:
        upsert_user(email=email, phone=phone or None, plan=None, stripe_customer_id=None, status="active")
    else:
        # update phone if provided
        if phone:
            upsert_user(email=email, phone=phone, plan=u["plan"], stripe_customer_id=u["stripe_customer_id"], status=u["subscription_status"])

    set_user_password(email, password)

    session["user"] = {"email": email}
    return redirect("/account?status=active")
@app.post("/stripe-webhook")
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        return str(e), 400

    # ✅ Fires after successful checkout/subscription creation
    if event["type"] == "checkout.session.completed":
        session_obj = event["data"]["object"]

        email = (session_obj.get("customer_details") or {}).get("email")
        customer_id = session_obj.get("customer")

        if email:
            email = email.strip().lower()

            # Create / update user as active
            upsert_user(
                email=email,
                phone=None,
                plan="unknown",
                stripe_customer_id=customer_id,
                status="active"
            )

            token = make_setup_token(email)
            setup_link = f"{APP_BASE_URL}/account/setup?token={token}"

            subject = "Welcome to Signal Hawk Capital — Complete Your Account Setup"
            html = f"""
            <div style="font-family: Arial, sans-serif; line-height:1.5;">
              <p>Hi,</p>
              <p>Welcome to <b>Signal Hawk Capital</b> — your subscription is now active.</p>
              <p>To access your account, alerts, and analytics, please complete your account setup:</p>
              <p><a href="{setup_link}" style="display:inline-block;padding:12px 16px;border-radius:10px;background:#4f46e5;color:#fff;text-decoration:none;">
                Set up your account
              </a></p>
              <p style="color:#666;">This link expires in 24 hours.</p>
              <hr/>
              <p style="color:#666;font-size:12px;">
                Educational use only. Investing involves risk, including loss of principal.
              </p>
            </div>
            """

            # Send email
            try:
                send_email(email, subject, html)
            except Exception as e:
                # don’t fail webhook if email fails
                print("Email send failed:", e)

    return "", 200


def _get_price_id(investor_type: str, plan: str) -> str:
    """Return the Stripe *price* id for the selected investor type + plan.

    investor_type: 'individual' | 'professional' (also accepts 'pro')
    plan: 'monthly' | 'yearly' | 'lifetime'
    """
    inv = (investor_type or "individual").lower().strip()
    if inv in ("pro", "professional"):
        inv = "professional"

    plan = (plan or "").lower().strip()
    if plan not in ("monthly", "yearly", "lifetime"):
        return ""

    if inv == "professional":
        lookup = {
            "monthly": STRIPE_PRICE_PRO_MONTHLY,
            "yearly": STRIPE_PRICE_PRO_YEARLY,
            "lifetime": STRIPE_PRICE_PRO_LIFETIME,
        }
    else:
        lookup = {
            "monthly": STRIPE_PRICE_INDIVIDUAL_MONTHLY,
            "yearly": STRIPE_PRICE_INDIVIDUAL_YEARLY,
            "lifetime": STRIPE_PRICE_INDIVIDUAL_LIFETIME,
        }

    # If new scheme isn't set, fall back to legacy vars
    price_id = lookup.get(plan) or ""
    if not price_id:
        legacy_lookup = {
            "monthly": STRIPE_PRICE_MONTHLY,
            "yearly": STRIPE_PRICE_YEARLY,
            "lifetime": STRIPE_PRICE_LIFETIME,
        }
        price_id = legacy_lookup.get(plan) or ""

    return price_id
# ---------- Stripe Checkout: create session ----------
@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    """Create a Stripe Checkout session and return a redirect URL.

    Expects JSON:
      - email (required)
      - phone (required)
      - plan: monthly | yearly | lifetime (required)
      - investorType: individual | professional (optional; defaults individual)
    """
    if not STRIPE_SECRET_KEY:
        return ("Stripe not configured – set STRIPE_SECRET_KEY", 500)

    data = request.get_json(silent=True) or {}
    accepted_terms = bool(data.get("accepted_terms"))
    if not accepted_terms:
        return ("You must accept the Terms/Disclosures/Privacy Policy to continue.", 400)
    email = (data.get("email") or "").strip()
    phone = (data.get("phone") or "").strip()
    plan = (data.get("plan") or "").strip().lower()
    investor_type = (data.get("investorType") or data.get("investor_type") or "individual").strip().lower()

    if not email or not phone or not plan:
        return ("Missing email, phone, or plan", 400)

    price_id = _get_price_id(investor_type, plan)
    if not price_id:
        return (
            "Stripe price not configured for this selection. "
            "Set STRIPE_PRICE_INDIVIDUAL_* / STRIPE_PRICE_PRO_* (or legacy STRIPE_PRICE_*).",
            500,
        )

    mode = "subscription" if plan in ("monthly", "yearly") else "payment"

    # Optional: record subscription intent (don't block checkout if DB fails)
    try:
        if "upsert_user" in globals():
            upsert_user(
                email=email,
                phone=phone,
                plan=plan,
                stripe_customer_id=None,
                status="initiated",
            )
    except Exception as exc:
        log.warning("upsert_user failed (ignored): %s", exc)

    success_url = f"{APP_BASE_URL}/?checkout=success"
    cancel_url = f"{APP_BASE_URL}/signup?checkout=canceled"

    try:
        session = stripe.checkout.Session.create(
            mode=mode,
            line_items=[{"price": price_id, "quantity": 1}],
            customer_email=email,
            phone_number_collection={"enabled": True},
            allow_promotion_codes=True,
            metadata={
                "alerts_email": email,
                "alerts_phone": phone,
                "plan": plan,
                "terms_accepted": "true",
                "terms_accepted_at": datetime.utcnow().isoformat(),
                "terms_accepted_ip": request.headers.get("X-Forwarded-For", request.remote_addr) or ""
            },
            success_url=success_url,
            cancel_url=cancel_url,
        )
        return jsonify({"url": session.url})
    except stripe.error.StripeError as e:
        msg = getattr(e, "user_message", None) or str(e)
        log.exception("Stripe error creating checkout session")
        return (msg, 500)
    except Exception as e:
        log.exception("Unexpected error creating checkout session")
        return (str(e), 500)

@app.route("/")
def index():
    return render_template("home.html", page_title="Signal Hawk Capital")

from flask import session, redirect

@app.route("/auth/google")
def auth_google():
    redirect_uri = url_for("auth_google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route("/auth/google/callback")
def auth_google_callback():
    token = oauth.google.authorize_access_token()
    userinfo = token.get("userinfo")
    if not userinfo:
        userinfo = oauth.google.parse_id_token(token)

    email = (userinfo.get("email") or "").strip().lower()
    sub = userinfo.get("sub")

    if not email:
        return "Google login failed: no email returned.", 400

    # Upsert user
    with db_conn() as conn:
        existing = conn.execute("SELECT email FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE users SET oauth_provider=?, oauth_sub=? WHERE email=?",
                ("google", sub, email),
            )
        else:
            conn.execute(
                "INSERT INTO users (email, oauth_provider, oauth_sub, subscription_status) VALUES (?, ?, ?, ?)",
                (email, "google", sub, "inactive"),
            )
        conn.commit()

    session["user_email"] = email
    return redirect(url_for("account"))

import jwt
import time

def build_apple_client_secret():
    team_id = os.environ.get("APPLE_TEAM_ID")
    client_id = os.environ.get("APPLE_CLIENT_ID")
    key_id = os.environ.get("APPLE_KEY_ID")
    private_key = os.environ.get("APPLE_PRIVATE_KEY_PEM")

    if not all([team_id, client_id, key_id, private_key]):
        return None

    now = int(time.time())
    payload = {
        "iss": team_id,
        "iat": now,
        "exp": now + 3600,
        "aud": "https://appleid.apple.com",
        "sub": client_id,
    }
    headers = {"kid": key_id}

    return jwt.encode(payload, private_key, algorithm="ES256", headers=headers)
import jwt
import time

oauth.register(
    name="apple",
    client_id=os.environ.get("APPLE_CLIENT_ID"),
    client_secret=build_apple_client_secret,
    authorize_url="https://appleid.apple.com/auth/authorize",
    access_token_url="https://appleid.apple.com/auth/token",
    client_kwargs={"scope": "name email"},
)

@app.route("/auth/apple")
def auth_apple():
    redirect_uri = url_for("auth_apple_callback", _external=True)
    return oauth.apple.authorize_redirect(redirect_uri)

@app.route("/auth/apple/callback")
def auth_apple_callback():
    token = oauth.apple.authorize_access_token()

    # Apple returns an id_token (JWT)
    id_token = token.get("id_token")
    if not id_token:
        return "Apple login failed: missing id_token.", 400

    # We can decode without verification for basic fields (email/sub).
    # For full verification, fetch Apple's JWKS and verify signature.
    claims = jwt.decode(id_token, options={"verify_signature": False})

    email = (claims.get("email") or "").strip().lower()
    sub = claims.get("sub")

    if not email:
        # Apple may not return email after the first auth; you should store email the first time.
        # If missing, you can fall back to matching oauth_sub only.
        with db_conn() as conn:
            row = conn.execute("SELECT email FROM users WHERE oauth_provider=? AND oauth_sub=?", ("apple", sub)).fetchone()
            if not row:
                return "Apple login succeeded but no email on file. Please contact support.", 400
            email = row["email"]

    with db_conn() as conn:
        existing = conn.execute("SELECT email FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE users SET oauth_provider=?, oauth_sub=? WHERE email=?",
                ("apple", sub, email),
            )
        else:
            conn.execute(
                "INSERT INTO users (email, oauth_provider, oauth_sub, subscription_status) VALUES (?, ?, ?, ?)",
                (email, "apple", sub, "inactive"),
            )
        conn.commit()

    session["user_email"] = email
    return redirect(url_for("account"))

from flask import session, redirect, render_template, request, url_for

@app.route("/account")
def account():
    # Must be logged in
    user_sess = session.get("user") or {}
    email = (user_sess.get("email") or "").strip().lower()

    if not email:
        return redirect("/login?msg=Please log in to access your account.")

    u = get_user_by_email(email)

    # ✅ Convert sqlite Row to dict for Jinja safety
    user = dict(u) if u else None

    return render_template("account.html", user=user)

@app.route("/account/send-setup-link")
def send_setup_link():
    user_sess = session.get("user") or {}
    email = (user_sess.get("email") or "").strip().lower()

    if not email:
        return redirect("/login?msg=Please log in first.")

    token = make_setup_token(email)
    setup_link = f"{APP_BASE_URL}/account/setup?token={token}"

    subject = "Finish setting up your Signal Hawk Capital account"
    html = f"""
      <p>Click below to finish setting up your account:</p>
      <p><a href="{setup_link}">{setup_link}</a></p>
      <p>This link expires in 24 hours.</p>
    """

    send_email(email, subject, html)
    return redirect("/account?status=Setup link sent to your email.")

@app.route("/account/request-setup", methods=["GET", "POST"])
def request_setup():
    if request.method == "GET":
        # simple form to ask for email
        return """
        <div style="max-width:520px;margin:40px auto;font-family:Arial;">
          <h2>Send Setup Link</h2>
          <form method="POST">
            <label>Email</label><br/>
            <input name="email" type="email" required style="width:100%;padding:10px;margin:8px 0;border-radius:8px;border:1px solid #ccc"/>
            <button type="submit" style="padding:10px 14px;border-radius:8px;border:none;background:#4f46e5;color:#fff;font-weight:700;cursor:pointer">
              Email me a setup link
            </button>
          </form>
        </div>
        """

    email = (request.form.get("email") or "").strip().lower()
    if not email:
        return ("Email required.", 400)

    # Ensure user exists (create a stub row if needed)
    u = get_user_by_email(email)
    if not u:
        upsert_user(email=email, phone=None, plan=None, stripe_customer_id=None, status="active")

    token = make_setup_token(email)
    setup_link = f"{APP_BASE_URL}/account/setup?token={token}"

    subject = "Finish setting up your Signal Hawk Capital account"
    html = f"""
      <p>Click below to finish setting up your account:</p>
      <p><a href="{setup_link}">{setup_link}</a></p>
      <p>This link expires in 24 hours.</p>
    """

    send_email(email, subject, html)
    return redirect("/login?msg=Setup link sent. Check your email.")


import sqlite3
from contextlib import closing

DB_PATH = os.environ.get("DATABASE_PATH", "signal_hawk.db")

def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_current_user():
    email = session.get("user_email")
    if not email:
        return None
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return dict(row) if row else None

import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.environ.get("DB_PATH", "signal_hawk.db")

def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_users_columns():
    cols = {
        "password_hash": "TEXT",
        "oauth_provider": "TEXT",
        "oauth_sub": "TEXT"
    }
    with db_conn() as conn:
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        for c, ctype in cols.items():
            if c not in existing:
                conn.execute(f"ALTER TABLE users ADD COLUMN {c} {ctype}")
        conn.commit()


from flask import render_template, request, redirect, session



@app.route("/billing-portal", methods=["POST"])
def billing_portal():
    user = get_current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401
    if not user.get("stripe_customer_id"):
        return jsonify({"error": "No Stripe customer on file"}), 400

    if not STRIPE_SECRET_KEY:
        return jsonify({"error": "Stripe not configured"}), 500

    stripe.api_key = STRIPE_SECRET_KEY

    portal = stripe.billing_portal.Session.create(
        customer=user["stripe_customer_id"],
        return_url=url_for("account", _external=True)
    )
    return jsonify({"url": portal.url})

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/resend-setup", methods=["POST"])
def resend_setup():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "Email required"}), 400

    # TODO: lookup user, verify active subscription, create token, send email
    # For now:
    return jsonify({"ok": True})

from flask import session, redirect


@app.route("/login-password", methods=["POST"])
def login_password():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    if not email or not password:
        return render_template("login.html", error="Email and password are required.")

    with db_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not row or not row["password_hash"]:
            return render_template("login.html", error="No password set for this email. Use the setup email or reset password.")
        if not check_password_hash(row["password_hash"], password):
            return render_template("login.html", error="Incorrect email or password.")

    session["user_email"] = email
    return redirect(url_for("account"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        msg = request.args.get("msg", "")
        return render_template("login.html", msg=msg)

    email = (request.form.get("email") or "").strip().lower()
    password = (request.form.get("password") or "").strip()

    if not email or not password:
        return render_template("login.html", msg="Enter email and password.")

    u = get_user_by_email(email)
    if not u:
        return render_template("login.html", msg="No account found for this email. Please set up your account.")

    pw_hash = u["password_hash"]
    if not pw_hash:
        return render_template("login.html", msg="No password set for this email. Use the setup email or reset password.")

    if not check_password_hash(pw_hash, password):
        return render_template("login.html", msg="Incorrect password.")

    session["user"] = {"email": email}
    return redirect("/account?status=active")

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/disclosures")
def disclosures():
    return render_template("disclosures.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/resources")
def resources():
    return render_template("resources_limina.html", page_title="Resources")

@app.route("/about")
def about():
    return render_template("about.html", page_title="About")

@app.route("/signup")
def signup():
    return render_template("signup_limina.html", page_title="Subscribe")

TICKER_META = {
    "SPXL": {"label": "SPXL Buy & Hold", "expense_ratio": 0.0095},
    "SPY": {"label": "S&P 500 (SPY) Buy & Hold", "expense_ratio": 0.0009},
    "^GSPC": {"label": "S&P 500 Index (^GSPC) Buy & Hold", "expense_ratio": 0.0},
    "NVDA": {"label": "NVDA Buy & Hold", "expense_ratio": 0.0},
    "AMZN": {"label": "AMZN Buy & Hold", "expense_ratio": 0.0},
    "AAPL": {"label": "AAPL Buy & Hold", "expense_ratio": 0.0},
    "TSLA": {"label": "TSLA Buy & Hold", "expense_ratio": 0.0},
}

def _download_close(tickers: List[str], start: date, end: date) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.sort_index().ffill()

def _apply_expense_drag(series: pd.Series, ratio_annual: float) -> pd.Series:
    if ratio_annual <= 0: 
        return series
    daily_drag = (1 - ratio_annual) ** (1/252)
    ret = series.pct_change().fillna(0.0)
    adj_growth = (1 + ret) * daily_drag
    out = adj_growth.cumprod()
    return out * (series.iloc[0] / out.iloc[0])

def _equity_curve_from_price(price: pd.Series, initial: float, include_er: bool, expense_ratio: float,
                             include_taxes: bool, tax_rate: float = 0.2) -> pd.Series:
    px = price.copy().dropna()
    if include_er and expense_ratio > 0:
        px = _apply_expense_drag(px, expense_ratio)
    ret = px.pct_change().fillna(0.0)
    if include_taxes and tax_rate > 0:
        ret = np.where(ret > 0, ret * (1 - tax_rate), ret)
        ret = pd.Series(ret, index=px.index)
    equity = (1 + ret).cumprod() * (initial if initial else 10000.0)
    return equity

def _signal_hawk_equity(spx_price: pd.Series, spxl_price: pd.Series, initial: float,
                        include_er: bool, include_taxes: bool) -> pd.Series:
    spxl = spxl_price.copy().dropna()
    if include_er:
        spxl = _apply_expense_drag(spxl, TICKER_META["SPXL"]["expense_ratio"])
    spx = spx_price.reindex_like(spxl).ffill()
    sma = spx.rolling(80).mean()
    signal = (spx > sma).astype(int).shift(1).fillna(0)
    ret_spxl = spxl.pct_change().fillna(0.0)
    strat_ret = ret_spxl * signal
    if include_taxes:
        strat_ret = np.where(strat_ret > 0, strat_ret * (1 - 0.2), strat_ret)
        strat_ret = pd.Series(strat_ret, index=spxl.index)
    equity = (1 + strat_ret).cumprod() * (initial if initial else 10000.0)
    equity.name = "Signal Hawk Strategy"
    return equity

@app.route("/analyzer", methods=["GET", "POST"])
def analyzer():
    today = date.today()
    start = request.values.get("start", "2010-01-01")
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
    except ValueError:
        start_date = date(2010, 1, 1)

    initial = float(request.values.get("initial", 10000))
    include_er = request.values.get("include_er", "on") == "on"
    include_taxes = request.values.get("include_taxes", "off") == "on"

    defaults = {
        "series_SH": "on",
        "series_HG_SIM": "on",
        "series_SPXL": "on",
        "series_SP500": "on",
        "series_NVDA": "off",
        "series_AMZN": "off",
        "series_AAPL": "off",
        "series_TSLA": "off",
    }
    series_flags = {k: request.values.get(k, defaults[k]) == "on" for k in defaults}

    need = ["SPXL", "SPY", "^GSPC"]
    for sym, flag in [("NVDA","series_NVDA"),("AMZN","series_AMZN"),("AAPL","series_AAPL"),("TSLA","series_TSLA")]:
        if series_flags[flag]:
            need.append(sym)
    df_close = _download_close(sorted(set(need)), start_date, today)

    curves = {}
    spx = df_close.get("^GSPC") if "^GSPC" in df_close.columns else df_close.get("SPY")
    if isinstance(spx, pd.DataFrame): spx = spx.iloc[:,0]
    spxl = df_close.get("SPXL", None)
    if isinstance(spxl, pd.DataFrame): spxl = spxl.iloc[:,0]
    spy = df_close.get("SPY", None)
    if isinstance(spy, pd.DataFrame): spy = spy.iloc[:,0]

    if series_flags["series_SH"] and spx is not None and spxl is not None:
        curves["Signal Hawk Strategy"] = _signal_hawk_equity(spx, spxl, initial, include_er, include_taxes)

    if series_flags["series_HG_SIM"] and spx is not None and spy is not None:
        sma = spx.rolling(80).mean()
        signal = (spx > sma).astype(int).shift(1).fillna(0)
        ret_spy = spy.pct_change().fillna(0.0) * 2.5
        if include_er:
            ret_spy = pd.Series(np.where(ret_spy>0, ret_spy*(1 - 0.0095/0.4), ret_spy), index=spy.index)
        if include_taxes:
            ret_spy = pd.Series(np.where(ret_spy>0, ret_spy*(1-0.2), ret_spy), index=spy.index)
        curves["HG_SIM Strategy"] = (1 + ret_spy*signal).cumprod() * initial

    if series_flags["series_SPXL"] and spxl is not None:
        curves["SPXL Buy & Hold"] = _equity_curve_from_price(spxl, initial, include_er, 0.0095, include_taxes)

    spx_source = "^GSPC" if "^GSPC" in df_close.columns else "SPY"
    if series_flags["series_SP500"] and spx_source in df_close.columns:
        series = df_close[spx_source]
        if isinstance(series, pd.DataFrame): series = series.iloc[:,0]
        curves[("S&P 500 (SPY) Buy & Hold" if spx_source=="SPY" else "S&P 500 Index (^GSPC) Buy & Hold")] = _equity_curve_from_price(series, initial, include_er, 0.0009 if spx_source=="SPY" else 0.0, include_taxes)

    for sym, flag in [("NVDA","series_NVDA"),("AMZN","series_AMZN"),("AAPL","series_AAPL"),("TSLA","series_TSLA")]:
        if series_flags[flag] and sym in df_close.columns:
            series = df_close[sym]
            if isinstance(series, pd.DataFrame): series = series.iloc[:,0]
            curves[f"{sym} Buy & Hold"] = _equity_curve_from_price(series, initial, include_er, 0.0, include_taxes)

    color_map = {
        "Signal Hawk Strategy": "#3b5bdb",
        "HG_SIM Strategy": "#6f42c1",
        "SPXL Buy & Hold": "#fd7e14",
        "S&P 500 (SPY) Buy & Hold": "#2fbf71",
        "S&P 500 Index (^GSPC) Buy & Hold": "#2fbf71",
        "NVDA Buy & Hold": "#00bcd4",
        "AMZN Buy & Hold": "#9c27b0",
        "AAPL Buy & Hold": "#607d8b",
        "TSLA Buy & Hold": "#e91e63",
    }

    fig = go.Figure()
    for name, series in curves.items():
        fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name=name,
                                 line=dict(width=2, color=color_map.get(name))))
    fig.update_layout(
        title="Investment Growth Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=600,
        template="plotly_dark",
        margin=dict(l=30,r=20,t=30,b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    price_graph = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

    table_rows = []
    for name, series in curves.items():
        years = (series.index[-1] - series.index[0]).days / 365.25
        cagr = (series.iloc[-1] / series.iloc[0]) ** (1/years) - 1 if years > 0 else float('nan')
        mdd = (series / series.cummax() - 1).min()
        table_rows.append({"name": name, "cagr": f"{cagr*100:.2f}%", "mdd": f"{mdd*100:.2f}%", "final": f"${series.iloc[-1]:,.0f}"})

    msg = f"If you invested ${initial:,.2f} on {start_date.strftime('%m/%d/%Y')} and used the Signal Hawk buy & sell alerts your account value would be "
    if "Signal Hawk Strategy" in curves:
        msg += f"${curves['Signal Hawk Strategy'].iloc[-1]:,.2f} by {date.today().strftime('%m/%d/%Y')}.*"
    else:
        msg += "— (insufficient data).*"

    return render_template("growth_analyzer.html",
                           page_title="Investment Growth Analyzer",
                           price_graph=price_graph,
                           message=msg,
                           start=start_date.strftime("%Y-%m-%d"),
                           initial=initial,
                           include_er=include_er,
                           include_taxes=include_taxes,
                           series_flags=series_flags,
                           table_rows=table_rows)

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=debug)
