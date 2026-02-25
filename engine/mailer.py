"""
Email module — SMTP-based onboarding and notification emails.

Required env vars:
  SMTP_HOST      — e.g. smtp.gmail.com
  SMTP_PORT      — e.g. 587
  SMTP_USER      — your email
  SMTP_PASS      — app password
  MAIL_FROM      — From address (defaults to SMTP_USER)
  APP_URL        — https://yourdomain.com
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
MAIL_FROM = os.environ.get("MAIL_FROM", "") or SMTP_USER
APP_URL = os.environ.get("APP_URL", "http://localhost:5000")


def is_mail_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASS)


def _send(to: str, subject: str, html: str) -> bool:
    """Send an HTML email. Returns True on success."""
    if not is_mail_configured():
        print(f"[MAIL] Not configured — would send to {to}: {subject}")
        return False

    msg = MIMEMultipart("alternative")
    msg["From"] = f"Batman Engine <{MAIL_FROM}>"
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print(f"[MAIL] Sent to {to}: {subject}")
        return True
    except Exception as e:
        print(f"[MAIL] Failed to send to {to}: {e}")
        return False


# ── Email Templates ──

def send_welcome(to: str, name: str = ""):
    greeting = f"Hi {name}," if name else "Welcome!"
    html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:600px;margin:0 auto;background:#0a0b0f;color:#e8e9f0;padding:40px;border-radius:12px">
      <div style="text-align:center;margin-bottom:32px">
        <div style="font-size:28px;font-weight:700;color:#4f6ef7">Batman Engine</div>
        <div style="font-size:13px;color:#8a8b9a;margin-top:4px">0DTE SPX Strategy Platform</div>
      </div>
      <h2 style="color:#e8e9f0;font-size:22px">{greeting}</h2>
      <p style="color:#8a8b9a;line-height:1.7;font-size:15px">
        Your account is ready. Here's what you can do with Batman Engine:
      </p>
      <div style="background:#12131a;border:1px solid #2a2b3a;border-radius:10px;padding:20px;margin:20px 0">
        <div style="margin-bottom:16px">
          <span style="color:#22c55e;font-weight:600">Batman Strategy</span>
          <span style="color:#8a8b9a"> — Butterfly position builder with live data, trap timing, and P&L visualization</span>
        </div>
        <div style="margin-bottom:16px">
          <span style="color:#06b6d4;font-weight:600">Directional Signals</span>
          <span style="color:#8a8b9a"> — 4 backtested strategies for intraday SPX direction calls</span>
        </div>
        <div>
          <span style="color:#a855f7;font-weight:600">Backtester</span>
          <span style="color:#8a8b9a"> — Test strategies against real historical data with full diagnostics</span>
        </div>
      </div>
      <div style="text-align:center;margin:32px 0">
        <a href="{APP_URL}/account" style="background:#4f6ef7;color:#fff;padding:14px 40px;border-radius:10px;text-decoration:none;font-weight:600;font-size:15px;display:inline-block">
          Get Started →
        </a>
      </div>
      <p style="color:#5a5b6a;font-size:12px;text-align:center;margin-top:32px;border-top:1px solid #2a2b3a;padding-top:20px">
        Batman Engine — Statistical models for educational purposes. Not financial advice.
      </p>
    </div>
    """
    return _send(to, "Welcome to Batman Engine", html)


def send_subscription_active(to: str, name: str = ""):
    greeting = f"Hi {name}," if name else "Great news!"
    html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:600px;margin:0 auto;background:#0a0b0f;color:#e8e9f0;padding:40px;border-radius:12px">
      <div style="text-align:center;margin-bottom:32px">
        <div style="font-size:28px;font-weight:700;color:#4f6ef7">Batman Engine</div>
      </div>
      <h2 style="color:#22c55e;font-size:22px">✓ Subscription Active</h2>
      <p style="color:#8a8b9a;line-height:1.7;font-size:15px">
        {greeting} your Pro subscription is now active. You have full access to all Batman Engine tools.
      </p>
      <div style="background:#12131a;border:1px solid #22c55e40;border-radius:10px;padding:20px;margin:20px 0">
        <div style="color:#22c55e;font-weight:600;margin-bottom:8px">Your Pro access includes:</div>
        <ul style="color:#8a8b9a;line-height:2;padding-left:20px;font-size:14px">
          <li>Batman Strategy builder with live SPX/VIX data</li>
          <li>4 directional signal strategies with real-time analysis</li>
          <li>Full historical backtester with parameter optimization</li>
          <li>VIX calibration engine with regime analysis</li>
          <li>All future features and updates</li>
        </ul>
      </div>
      <div style="text-align:center;margin:32px 0">
        <a href="{APP_URL}/batman" style="background:#22c55e;color:#fff;padding:14px 40px;border-radius:10px;text-decoration:none;font-weight:600;font-size:15px;display:inline-block">
          Launch Batman Strategy →
        </a>
      </div>
      <p style="color:#5a5b6a;font-size:12px;text-align:center;margin-top:32px">
        Manage your subscription anytime from your <a href="{APP_URL}/account" style="color:#4f6ef7">account page</a>.
      </p>
    </div>
    """
    return _send(to, "Your Batman Engine Pro subscription is active", html)


def send_subscription_canceled(to: str, name: str = ""):
    html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:600px;margin:0 auto;background:#0a0b0f;color:#e8e9f0;padding:40px;border-radius:12px">
      <div style="text-align:center;margin-bottom:32px">
        <div style="font-size:28px;font-weight:700;color:#4f6ef7">Batman Engine</div>
      </div>
      <h2 style="color:#f59e0b;font-size:22px">Subscription Update</h2>
      <p style="color:#8a8b9a;line-height:1.7;font-size:15px">
        Your Pro subscription has been canceled. You'll retain access until the end of your current billing period.
      </p>
      <p style="color:#8a8b9a;line-height:1.7;font-size:15px">
        If this was a mistake, you can resubscribe anytime from your account page.
      </p>
      <div style="text-align:center;margin:32px 0">
        <a href="{APP_URL}/account" style="background:#4f6ef7;color:#fff;padding:14px 40px;border-radius:10px;text-decoration:none;font-weight:600;font-size:15px;display:inline-block">
          Manage Account
        </a>
      </div>
    </div>
    """
    return _send(to, "Batman Engine — Subscription Update", html)
