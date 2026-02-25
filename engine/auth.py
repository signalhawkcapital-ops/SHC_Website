"""
Authentication module — SQLite-backed user accounts.
Uses werkzeug for password hashing (no bcrypt dependency).
"""

import os
import sqlite3
import secrets
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from flask import session, redirect, url_for, request, flash

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "instance", "users.db")


def _get_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    """Create tables if they don't exist, and migrate if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = _get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT DEFAULT '',
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            subscription_status TEXT DEFAULT 'none',
            plan TEXT DEFAULT 'free',
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    # Migrate: add is_admin column if table already existed without it
    try:
        db.execute("SELECT is_admin FROM users LIMIT 1")
    except Exception:
        db.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
    db.commit()
    db.close()


# ── Admin seed ──

ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@wingspan.com")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "batman2026!")


def seed_admin():
    """Create or update the admin account with full access.
    
    Admin bypasses subscription checks.  Credentials can be overridden
    with ADMIN_EMAIL / ADMIN_PASSWORD env vars.
    """
    db = _get_db()
    existing = db.execute("SELECT id FROM users WHERE email = ?", (ADMIN_EMAIL,)).fetchone()
    if existing:
        db.execute(
            """UPDATE users SET password_hash = ?, subscription_status = 'active',
               plan = 'admin', is_admin = 1 WHERE email = ?""",
            (generate_password_hash(ADMIN_PASSWORD), ADMIN_EMAIL),
        )
    else:
        db.execute(
            """INSERT INTO users (email, password_hash, name, subscription_status, plan, is_admin)
               VALUES (?, ?, 'Admin', 'active', 'admin', 1)""",
            (ADMIN_EMAIL, generate_password_hash(ADMIN_PASSWORD)),
        )
    db.commit()
    db.close()


# ── User CRUD ──

def create_user(email: str, password: str, name: str = "") -> dict:
    db = _get_db()
    try:
        db.execute(
            "INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)",
            (email.lower().strip(), generate_password_hash(password), name.strip()),
        )
        db.commit()
        user = db.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
        return dict(user)
    except sqlite3.IntegrityError:
        return None
    finally:
        db.close()


def authenticate(email: str, password: str) -> dict:
    db = _get_db()
    user = db.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
    db.close()
    if user and check_password_hash(user["password_hash"], password):
        return dict(user)
    return None


def get_user_by_id(user_id: int) -> dict:
    db = _get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    db.close()
    return dict(user) if user else None


def get_user_by_email(email: str) -> dict:
    db = _get_db()
    user = db.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
    db.close()
    return dict(user) if user else None


def get_user_by_stripe_customer(customer_id: str) -> dict:
    db = _get_db()
    user = db.execute("SELECT * FROM users WHERE stripe_customer_id = ?", (customer_id,)).fetchone()
    db.close()
    return dict(user) if user else None


def update_user(user_id: int, **kwargs):
    db = _get_db()
    for key, val in kwargs.items():
        if key in ("stripe_customer_id", "stripe_subscription_id", "subscription_status", "plan", "last_login", "name"):
            db.execute(f"UPDATE users SET {key} = ? WHERE id = ?", (val, user_id))
    db.commit()
    db.close()


# ── Session management ──

def login_user(user: dict):
    """Set session for authenticated user."""
    session["user_id"] = user["id"]
    session["user_email"] = user["email"]
    session["user_name"] = user.get("name", "")
    session["plan"] = user.get("plan", "free")
    session["sub_status"] = user.get("subscription_status", "none")
    update_user(user["id"], last_login=datetime.utcnow().isoformat())


def logout_user():
    session.clear()


def get_current_user() -> dict:
    """Return current logged-in user or None."""
    uid = session.get("user_id")
    if not uid:
        return None
    return get_user_by_id(uid)


def is_subscribed() -> bool:
    """Check if current session has an active subscription."""
    return session.get("sub_status") in ("active", "trialing")


# ── Decorators ──

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login_page", next=request.path))
        return f(*args, **kwargs)
    return decorated


def subscription_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login_page", next=request.path))
        if not is_subscribed():
            flash("An active subscription is required to access this tool.", "info")
            return redirect(url_for("account_page"))
        return f(*args, **kwargs)
    return decorated
