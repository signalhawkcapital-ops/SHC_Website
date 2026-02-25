"""
Batman Strategy Engine — Flask Web Application
================================================
Uses REAL SPX/VIX data from Yahoo Finance.
Auto-downloads on first run, caches to data/spx_vix_daily.csv.
"""

import os
import numpy as np
import json
from datetime import date, time, datetime
from flask import Flask, render_template, render_template_string, request, jsonify, redirect, url_for, flash, session

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional

from engine.strategy import generate_strategy, Regime, RiskProfile, compute_trap_score, round_to_strike
from engine.backtester import (
    run_backtest, optimize_parameters,
    generate_synthetic_data, load_csv_data, DailyBar,
)
from engine.data_fetcher import fetch_and_cache, load_cached_data, get_data_info
from engine.calibration import calibrate_from_bars, load_calibration, get_calibrated_em
from engine.directional import (
    generate_directional_signals, backtest_all_directional, Direction,
)
from engine.auth import (
    init_db, seed_admin, create_user, authenticate, get_current_user,
    login_user, logout_user, is_subscribed, update_user,
    login_required, subscription_required, get_user_by_stripe_customer,
    ADMIN_EMAIL,
)
from engine.billing import (
    is_stripe_configured, create_checkout_session,
    create_portal_session, handle_webhook, STRIPE_PUBLISHABLE_KEY,
)
from engine.mailer import send_welcome, send_subscription_active, send_subscription_canceled

app = Flask(__name__)
_secret_key = os.environ.get("SECRET_KEY")
if not _secret_key:
    if os.environ.get("FLASK_ENV") == "production":
        raise RuntimeError("SECRET_KEY environment variable must be set in production")
    _secret_key = "dev-only-insecure-key-change-in-prod"
    print("WARNING: Using insecure default SECRET_KEY. Set SECRET_KEY env var for production.")
app.config["SECRET_KEY"] = _secret_key

# ── Rate limiter with proxy support ──
from collections import defaultdict
import time as _time
_rate_limits = defaultdict(list)  # ip -> [timestamps]
_rate_limit_last_cleanup = _time.time()

def _get_client_ip():
    """Get real client IP, respecting X-Forwarded-For behind proxy."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"

def _check_rate_limit(limit=30, window=60):
    """Allow `limit` requests per `window` seconds per IP."""
    global _rate_limit_last_cleanup
    ip = _get_client_ip()
    now = _time.time()
    _rate_limits[ip] = [t for t in _rate_limits[ip] if now - t < window]
    if len(_rate_limits[ip]) >= limit:
        return False
    _rate_limits[ip].append(now)
    # Periodic cleanup of stale IPs (every 5 minutes)
    if now - _rate_limit_last_cleanup > 300:
        stale = [k for k, v in _rate_limits.items() if not v or now - v[-1] > 300]
        for k in stale:
            del _rate_limits[k]
        _rate_limit_last_cleanup = now
    return True

@app.before_request
def _rate_limit_check():
    if request.path.startswith('/api/'):
        if not _check_rate_limit(limit=60, window=60):
            return jsonify({"status": "error", "message": "Rate limit exceeded. Try again shortly."}), 429


def api_auth_required(f):
    """Auth decorator for API endpoints — returns JSON 401 instead of redirect."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"status": "error", "message": "Authentication required"}), 401
        if not is_subscribed():
            return jsonify({"status": "error", "message": "Active subscription required"}), 403
        return f(*args, **kwargs)
    return decorated


# ── Security headers ──
_is_production = os.environ.get("FLASK_ENV") == "production"

if _is_production:
    app.config["SESSION_COOKIE_SECURE"] = True
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

@app.after_request
def _add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if _is_production:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Initialize user database
init_db()
# Seed admin only if env vars are set (no hardcoded credentials)
_admin_email = os.environ.get("ADMIN_EMAIL")
_admin_password = os.environ.get("ADMIN_PASSWORD")
if _admin_email and _admin_password:
    try:
        seed_admin(email=_admin_email, password=_admin_password)
    except TypeError:
        # Fallback if seed_admin doesn't accept kwargs yet
        seed_admin()
elif os.environ.get("FLASK_ENV") != "production":
    # In dev, use default seed_admin (which has its own defaults)
    seed_admin()
else:
    print("WARNING: ADMIN_EMAIL and ADMIN_PASSWORD not set. No admin account created.")

# ── Auto-fetch real data on startup ──
_real_data_cache = None
_data_source = "unknown"  # "real" or "synthetic"


def _get_real_data() -> list:
    """Load real SPX/VIX data, downloading if needed."""
    global _real_data_cache, _data_source
    if _real_data_cache is not None:
        return _real_data_cache

    csv_path = load_cached_data()
    if csv_path:
        bars = load_csv_data(csv_path)
        if bars:
            _real_data_cache = bars
            _data_source = "real"
            return bars

    # Try to download
    try:
        csv_path = fetch_and_cache(start="2020-01-02")
        bars = load_csv_data(csv_path)
        if bars:
            _real_data_cache = bars
            _data_source = "real"
            return bars
    except Exception as e:
        print(f"Warning: Could not fetch real data: {e}")

    # Fallback to synthetic
    print("Using synthetic data as fallback")
    _data_source = "synthetic"
    _real_data_cache = generate_synthetic_data(
        date(2020, 1, 2), date(2025, 12, 31), 3250, 14, seed=42)
    return _real_data_cache


def _filter_data(bars, start_date=None, end_date=None):
    """Filter bars by date range."""
    filtered = bars
    if start_date:
        sd = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
        filtered = [b for b in filtered if b.trade_date >= sd]
    if end_date:
        ed = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date
        filtered = [b for b in filtered if b.trade_date <= ed]
    return filtered


# Warm up on import
_calibration_cache = None
try:
    _startup_data = _get_real_data()
    print(f"Data loaded: {len(_startup_data)} trading days")
    # Auto-calibrate from real data
    _calibration_cache = calibrate_from_bars(_startup_data)
    if "error" not in _calibration_cache:
        print(f"Calibration: VIX overstatement={_calibration_cache['vix_overstatement']:.2f}x, "
              f"EM hit rate={_calibration_cache['em_hit_rate_1sigma']:.0f}%, "
              f"{_calibration_cache['count']} days analyzed")
    else:
        print(f"Calibration skipped: {_calibration_cache.get('error')}")
        _calibration_cache = None
except Exception as e:
    print(f"Startup deferred: {e}")


# ────────────────────────────────────────────────────────────
# Routes — Public Pages
# ────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        user = authenticate(request.form.get("email", ""), request.form.get("password", ""))
        if user:
            login_user(user)
            flash("Welcome back!", "success")
            next_url = request.form.get("next", "/batman")
            return redirect(next_url)
        flash("Invalid email or password.", "error")
    return render_template("auth/login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup_page():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        password2 = request.form.get("password2", "")
        name = request.form.get("name", "").strip()
        agreed_tos = request.form.get("agree_tos")

        if not email or not password:
            flash("Email and password are required.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
        elif password != password2:
            flash("Passwords don't match.", "error")
        elif not agreed_tos:
            flash("You must agree to the Terms of Service to create an account.", "error")
        else:
            user = create_user(email, password, name)
            if user:
                # Record TOS acceptance
                update_user(user["id"], tos_accepted_at=datetime.utcnow().isoformat())
                login_user(user)
                send_welcome(email, name)
                flash("Account created! Welcome to Wingspan.", "success")
                return redirect("/account")
            else:
                flash("An account with that email already exists.", "error")
    return render_template("auth/signup.html")


@app.route("/logout")
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect("/")


@app.route("/account")
@login_required
def account_page():
    user = get_current_user()
    sub_active = is_subscribed()
    return render_template("auth/account.html",
                           user=user, sub_active=sub_active,
                           stripe_configured=is_stripe_configured())


# ────────────────────────────────────────────────────────────
# Routes — Stripe Billing
# ────────────────────────────────────────────────────────────

@app.route("/billing/checkout", methods=["POST"])
@login_required
def billing_checkout():
    # Require TOS acceptance before payment
    agreed_tos = request.form.get("agree_tos")
    if not agreed_tos:
        flash("You must agree to the Terms of Service before subscribing.", "error")
        return redirect("/account")
    user = get_current_user()
    # Record TOS acceptance timestamp
    update_user(user["id"], tos_accepted_at=datetime.utcnow().isoformat())
    result = create_checkout_session(
        user["email"], user["id"], user.get("stripe_customer_id"))
    if "error" in result:
        flash(f"Checkout error: {result['error']}", "error")
        return redirect("/account")
    return redirect(result["url"])


@app.route("/billing/portal", methods=["POST"])
@login_required
def billing_portal():
    user = get_current_user()
    if not user.get("stripe_customer_id"):
        flash("No billing account found.", "error")
        return redirect("/account")
    result = create_portal_session(user["stripe_customer_id"])
    if "error" in result:
        flash(f"Portal error: {result['error']}", "error")
        return redirect("/account")
    return redirect(result["url"])


@app.route("/stripe/webhook", methods=["POST"])
def stripe_webhook():
    """Handle Stripe webhook events."""
    payload = request.get_data()
    sig = request.headers.get("Stripe-Signature", "")
    result = handle_webhook(payload, sig)

    if "error" in result:
        return jsonify({"error": result["error"]}), 400

    event_type = result.get("event_type", "")
    customer_id = result.get("customer_id")
    status = result.get("status")
    user_id = result.get("user_id")

    if event_type == "checkout.session.completed":
        # Link customer to user and activate subscription
        if user_id:
            from engine.auth import get_user_by_id
            user = get_user_by_id(int(user_id))
            if user:
                update_user(user["id"],
                            stripe_customer_id=customer_id,
                            stripe_subscription_id=result.get("subscription_id"),
                            subscription_status="active",
                            plan="pro")
                send_subscription_active(user["email"], user.get("name", ""))

    elif event_type in ("customer.subscription.updated", "customer.subscription.created"):
        if customer_id and status:
            user = get_user_by_stripe_customer(customer_id)
            if user:
                plan = "pro" if status in ("active", "trialing") else "free"
                update_user(user["id"], subscription_status=status, plan=plan)

    elif event_type == "customer.subscription.deleted":
        if customer_id:
            user = get_user_by_stripe_customer(customer_id)
            if user:
                update_user(user["id"], subscription_status="canceled", plan="free")
                send_subscription_canceled(user["email"], user.get("name", ""))

    elif event_type == "invoice.payment_failed":
        if customer_id:
            user = get_user_by_stripe_customer(customer_id)
            if user:
                update_user(user["id"], subscription_status="past_due")

    return jsonify({"received": True}), 200


# ────────────────────────────────────────────────────────────
# Routes — Protected Tool Pages
# ────────────────────────────────────────────────────────────

@app.route("/batman")
@subscription_required
def batman_page():
    return render_template("index.html")


@app.route("/backtest")
@subscription_required
def backtest_page():
    info = get_data_info()
    return render_template("backtest.html", data_info=info)


@app.route("/directional")
@subscription_required
def directional_page():
    return render_template("directional.html")


@app.route("/methodology")
@subscription_required
def methodology_page():
    return render_template("methodology.html", data_source=_data_source)


@app.route("/resources")
def resources_page():
    return render_template("resources.html")


@app.route("/methodology/pdf")
@subscription_required
def methodology_pdf():
    """Serve the methodology PDF for download."""
    pdf_path = os.path.join(os.path.dirname(__file__), "static", "batman_engine_methodology.pdf")
    if os.path.exists(pdf_path):
        from flask import send_file
        return send_file(pdf_path, as_attachment=True, download_name="Batman_Engine_Methodology.pdf")
    flash("PDF not yet generated. Please try again later.", "warning")
    return redirect(url_for("methodology_page"))


@app.route("/terms")
def terms_page():
    return render_template_string(TERMS_TEMPLATE)


TERMS_TEMPLATE = r"""
{% extends "base.html" %}
{% block title %}Terms of Service — Wingspan{% endblock %}
{% block content %}
<div style="max-width:760px;margin:0 auto;padding:24px 0">
<h1 style="font-size:24px;font-weight:700;margin-bottom:4px">Terms of Service</h1>
<p style="font-size:12px;color:var(--text-muted);margin-bottom:32px">Last updated: February 23, 2026 · Effective upon acceptance</p>

<div style="padding:14px 16px;background:var(--surface);border:1px solid var(--amber);border-radius:8px;margin-bottom:28px;font-size:12px;color:var(--text-dim);line-height:1.7">
<strong style="color:var(--amber)">IMPORTANT:</strong> By creating an account, subscribing, or using Wingspan in any capacity, you acknowledge that you have read, understood, and agree to be bound by these Terms of Service in their entirety. If you do not agree, do not use the Service.
</div>

<style>
.tos h2{font-size:15px;font-weight:700;margin:28px 0 8px;color:var(--text)}
.tos h3{font-size:13px;font-weight:600;margin:16px 0 6px;color:var(--text-dim)}
.tos p,.tos li{font-size:12px;color:var(--text-dim);line-height:1.8;margin-bottom:6px}
.tos ol,.tos ul{padding-left:20px;margin:8px 0}
.tos .warn{padding:10px 14px;background:var(--surface);border-left:3px solid var(--red);border-radius:4px;margin:12px 0;font-size:11px}
</style>
<div class="tos">

<h2>1. Definitions</h2>
<p>"Service" means the Wingspan web application, APIs, data, analysis tools, and all associated content. "Company," "we," or "us" refers to Wingspan LLC. "User," "you," or "subscriber" refers to any individual or entity accessing or using the Service. "Content" means all outputs, calculations, charts, strike recommendations, order strings, backtest results, scores, and any other data generated by the Service.</p>

<h2>2. Nature of the Service</h2>
<p>Wingspan is a <strong>quantitative research and analysis tool</strong>. The Service provides mathematical models, historical data analysis, and hypothetical calculations related to S&P 500 Index (SPX) options strategies.</p>

<div class="warn">
<strong style="color:var(--red)">THE SERVICE IS NOT:</strong> investment advice, a recommendation to buy or sell any security, a solicitation of any transaction, a registered investment advisory service, a broker-dealer, a trading signal service, or a substitute for professional financial guidance. Wingspan LLC is not registered as an investment adviser, broker-dealer, or in any other capacity with the SEC, FINRA, CFTC, NFA, or any state securities regulator.
</div>

<h2>3. No Investment Advice — Complete Disclaimer</h2>
<p>All Content provided by the Service constitutes general information and hypothetical analysis only. Nothing in the Service constitutes a recommendation, solicitation, or offer to buy or sell any securities, derivatives, or other financial instruments.</p>
<p>You acknowledge and agree that:</p>
<ol>
<li>All strike calculations, order strings, butterfly parameters, trap timing, scoring, and every other output are <strong>mathematical calculations only</strong> and do not constitute trading recommendations or advice.</li>
<li>All backtest results, win rates, profit factors, Sharpe ratios, equity curves, and performance statistics are <strong>entirely hypothetical</strong> and based on simulated or modeled data. They do not represent actual trading results.</li>
<li>Hypothetical performance results have many inherent limitations including but not limited to: the benefit of hindsight, the inability to account for the impact of financial risk in actual trading, and the fact that no actual money was risked.</li>
<li>You are solely responsible for all trading decisions you make. The Service does not make, recommend, or execute any trades on your behalf.</li>
<li>You will independently verify all outputs before using them in any capacity.</li>
<li>You will consult a qualified, licensed financial professional before making any trading or investment decisions.</li>
</ol>

<h2>4. Risk Acknowledgment</h2>
<p>By using the Service, you expressly acknowledge and accept the following risks:</p>
<ol>
<li><strong>Options trading involves substantial risk of loss.</strong> SPX 0DTE (zero days to expiration) options are among the highest-risk instruments available and can result in the complete loss of your invested capital within hours or minutes.</li>
<li><strong>Butterfly spreads can lose 100% of the debit paid.</strong> While risk is theoretically capped at the debit, you can lose the entire amount on every trade.</li>
<li><strong>Past hypothetical performance does not predict future results.</strong> Market conditions change. A strategy that backtested profitably may produce losses in live trading.</li>
<li><strong>Model risk.</strong> The Service relies on mathematical models (Black-Scholes, VIX calibration, historical statistics) that simplify reality. Models can be wrong, inputs can be stale or inaccurate, and outputs may not reflect actual market conditions.</li>
<li><strong>Execution risk.</strong> The Service does not account for bid-ask spreads, slippage, partial fills, market impact, broker outages, connectivity issues, or any other execution factors that affect real trading.</li>
<li><strong>Data risk.</strong> Market data (SPX, VIX) is sourced from third parties and may be delayed, inaccurate, or unavailable. The Service does not guarantee the accuracy, completeness, or timeliness of any data.</li>
<li><strong>Tax and regulatory risk.</strong> SPX options have specific tax treatment (Section 1256 contracts). You are solely responsible for understanding and complying with all applicable tax laws and regulations.</li>
</ol>

<h2>5. Limitation of Liability</h2>
<p><strong>TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW:</strong></p>
<ol>
<li>THE SERVICE IS PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTIES OF ANY KIND, WHETHER EXPRESS, IMPLIED, STATUTORY, OR OTHERWISE, INCLUDING WITHOUT LIMITATION WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, ACCURACY, COMPLETENESS, OR NON-INFRINGEMENT.</li>
<li>IN NO EVENT SHALL WINGSPAN LLC, ITS OFFICERS, DIRECTORS, EMPLOYEES, AGENTS, AFFILIATES, SUCCESSORS, OR ASSIGNS BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, EXEMPLARY, OR PUNITIVE DAMAGES, INCLUDING BUT NOT LIMITED TO LOSS OF PROFITS, TRADING LOSSES, LOSS OF DATA, LOSS OF GOODWILL, OR ANY OTHER INTANGIBLE LOSSES, REGARDLESS OF THE CAUSE OF ACTION OR THE THEORY OF LIABILITY, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.</li>
<li>OUR TOTAL AGGREGATE LIABILITY TO YOU FOR ALL CLAIMS ARISING OUT OF OR RELATING TO THE SERVICE SHALL NOT EXCEED THE AMOUNT YOU PAID TO US IN SUBSCRIPTION FEES DURING THE THREE (3) MONTHS IMMEDIATELY PRECEDING THE EVENT GIVING RISE TO THE CLAIM.</li>
<li>YOU EXPRESSLY AGREE THAT YOUR USE OF THE SERVICE IS AT YOUR SOLE RISK. YOU ASSUME FULL RESPONSIBILITY AND RISK OF LOSS RESULTING FROM YOUR USE OF THE SERVICE, INCLUDING ANY TRADING DECISIONS YOU MAKE BASED ON OR INFLUENCED BY THE SERVICE'S OUTPUTS.</li>
<li>WE ARE NOT LIABLE FOR ANY TRADING LOSSES, MISSED OPPORTUNITIES, OR FINANCIAL HARM OF ANY KIND ARISING FROM YOUR USE OF THE SERVICE, REGARDLESS OF WHETHER SUCH LOSSES RESULT FROM ERRORS IN THE SERVICE'S CALCULATIONS, DATA INACCURACIES, SERVICE OUTAGES, OR ANY OTHER CAUSE.</li>
</ol>

<h2>6. Indemnification</h2>
<p>You agree to indemnify, defend, and hold harmless Wingspan LLC, its officers, directors, employees, agents, and affiliates from and against any and all claims, liabilities, damages, losses, costs, and expenses (including reasonable attorneys' fees) arising out of or in any way connected with:</p>
<ol>
<li>Your use of the Service or any Content generated by the Service;</li>
<li>Any trading activity you undertake, whether or not informed by the Service;</li>
<li>Your violation of these Terms;</li>
<li>Your violation of any applicable law, rule, or regulation;</li>
<li>Any claim by a third party arising from your use of the Service.</li>
</ol>

<h2>7. Subscription and Billing</h2>
<p>The Service is offered on a monthly subscription basis at the rate displayed at the time of purchase. Subscriptions are processed through Stripe, Inc. By subscribing, you authorize recurring monthly charges to your payment method.</p>
<ol>
<li><strong>Free trial:</strong> New subscribers may receive a free trial period. You will not be charged during the trial. You may cancel at any time before the trial ends to avoid charges.</li>
<li><strong>Cancellation:</strong> You may cancel your subscription at any time through your account page or Stripe billing portal. Cancellation takes effect at the end of the current billing period. No partial refunds are provided for unused portions of a billing period.</li>
<li><strong>Refund policy:</strong> All subscription fees are non-refundable except as required by applicable law. Given the nature of the Service (instant access to proprietary analytical tools), refunds are not available after the subscription period has begun.</li>
<li><strong>Price changes:</strong> We reserve the right to modify subscription pricing with 30 days' advance notice to existing subscribers.</li>
</ol>

<h2>8. Acceptable Use</h2>
<p>You agree not to:</p>
<ol>
<li>Share, redistribute, resell, sublicense, or make the Service available to any third party;</li>
<li>Scrape, crawl, or use automated tools to extract data from the Service;</li>
<li>Reverse-engineer, decompile, or attempt to derive the source code of the Service's proprietary algorithms;</li>
<li>Use the Service in any way that violates applicable laws or regulations;</li>
<li>Misrepresent the Service's hypothetical results as actual trading performance;</li>
<li>Use the Service to provide investment advice to others without proper registration and licensing.</li>
</ol>

<h2>9. Intellectual Property</h2>
<p>All proprietary algorithms, models, software, design, trade names, trademarks, and Content are the exclusive property of Wingspan LLC. Your subscription grants you a limited, non-exclusive, non-transferable, revocable license to access and use the Service for your personal, non-commercial trading research only.</p>

<h2>10. Dispute Resolution and Arbitration</h2>
<p><strong>PLEASE READ THIS SECTION CAREFULLY — IT AFFECTS YOUR LEGAL RIGHTS.</strong></p>
<ol>
<li>Any dispute, claim, or controversy arising out of or relating to these Terms or the Service shall be resolved through binding individual arbitration administered by the American Arbitration Association (AAA) under its Commercial Arbitration Rules.</li>
<li><strong>CLASS ACTION WAIVER:</strong> You agree that any arbitration or proceeding shall be conducted only on an individual basis and not in a class, consolidated, or representative action. You expressly waive any right to participate in a class action lawsuit or class-wide arbitration.</li>
<li>The arbitration shall be conducted in the State of Delaware. The arbitrator's decision shall be final and binding.</li>
<li>Notwithstanding the above, either party may seek injunctive or equitable relief in a court of competent jurisdiction.</li>
</ol>

<h2>11. Governing Law</h2>
<p>These Terms shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of law principles.</p>

<h2>12. Termination</h2>
<p>We reserve the right to suspend or terminate your access to the Service at any time, with or without cause, with or without notice. Upon termination, your right to use the Service ceases immediately. Sections 3–6 (disclaimers, limitation of liability, and indemnification) survive termination.</p>

<h2>13. Modifications</h2>
<p>We may modify these Terms at any time by posting the revised version on this page. Continued use of the Service after any modification constitutes acceptance of the updated Terms. Material changes will be communicated via email to active subscribers.</p>

<h2>14. Severability</h2>
<p>If any provision of these Terms is held to be unenforceable, the remaining provisions shall continue in full force and effect. The unenforceable provision shall be modified to the minimum extent necessary to make it enforceable.</p>

<h2>15. Entire Agreement</h2>
<p>These Terms, together with the <a href="/privacy" style="color:var(--accent)">Privacy Policy</a> and <a href="/disclosures" style="color:var(--accent)">Risk Disclosures</a>, constitute the entire agreement between you and Wingspan LLC regarding the Service and supersede all prior agreements and understandings.</p>

<h2>16. Contact</h2>
<p>For questions about these Terms, contact us at: <strong>legal@wingspan.app</strong></p>

</div>

<div style="margin-top:36px;padding:16px;background:var(--surface);border:1px solid var(--border);border-radius:10px;text-align:center">
<p style="font-size:12px;color:var(--text-dim);margin-bottom:12px">By creating an account or subscribing to Wingspan, you agree to these Terms of Service.</p>
<a href="/signup" class="btn btn-primary" style="text-decoration:none">Create Account</a>
</div>
</div>
{% endblock %}
"""


@app.route("/privacy")
def privacy_page():
    return render_template("legal/privacy.html")


@app.route("/disclosures")
def disclosures_page():
    return render_template("legal/disclosures.html")


# ────────────────────────────────────────────────────────────
# API — Directional Signals
# ────────────────────────────────────────────────────────────

@app.route("/api/directional", methods=["POST"])
@api_auth_required
def api_directional():
    """Generate directional signals for today."""
    data = request.get_json() or {}
    try:
        spx_open = float(data.get("spx_open", 6000))
        vix = float(data.get("vix", 16))
        trade_date_str = data.get("trade_date", date.today().isoformat())
        trade_date_val = date.fromisoformat(trade_date_str)

        # Find this date (or closest) in historical data for context
        all_bars = _get_real_data()
        # Find bars up to the requested date for context
        context_bars = [b for b in all_bars if b.trade_date <= trade_date_val]

        if len(context_bars) < 5:
            return jsonify({"status": "error", "message": "Insufficient historical data"}), 400

        # Use the last bar as "today" — override with user inputs
        today_bar = context_bars[-1]
        # Create a synthetic bar with user's inputs but real close/high/low from data
        from engine.backtester import DailyBar
        sim_bar = DailyBar(
            trade_date=trade_date_val,
            spx_open=spx_open,
            spx_high=max(spx_open, today_bar.spx_high * spx_open / today_bar.spx_open),
            spx_low=min(spx_open, today_bar.spx_low * spx_open / today_bar.spx_open),
            spx_close=today_bar.spx_close * spx_open / today_bar.spx_open,
            vix_open=vix,
            vix_close=vix,
        )
        prev_bars = list(reversed(context_bars[-6:-1]))

        summary = generate_directional_signals(sim_bar, prev_bars, vix)
        return jsonify({"status": "ok", "data": summary.to_dict()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/directional/backtest", methods=["POST"])
@api_auth_required
def api_directional_backtest():
    """Backtest all directional strategies against historical data."""
    data = request.get_json() or {}
    try:
        all_bars = _get_real_data()
        bars = _filter_data(
            all_bars,
            data.get("start_date"),
            data.get("end_date"),
        )
        if len(bars) < 30:
            return jsonify({"status": "error", "message": "Need at least 30 days"}), 400

        results = backtest_all_directional(bars)
        return jsonify({"status": "ok", "data": [r.to_dict() for r in results]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ────────────────────────────────────────────────────────────
# API — Data Info
# ────────────────────────────────────────────────────────────

@app.route("/api/data-info")
@api_auth_required
def api_data_info():
    """Return info about available market data + calibration."""
    info = get_data_info()
    all_bars = _get_real_data()
    if all_bars:
        info["total_days"] = len(all_bars)
        info["first_date"] = all_bars[0].trade_date.isoformat()
        info["last_date"] = all_bars[-1].trade_date.isoformat()
        info["is_synthetic"] = not info.get("available", False)
    if _calibration_cache and "error" not in _calibration_cache:
        info["calibration"] = {
            "vix_overstatement": _calibration_cache["vix_overstatement"],
            "em_hit_rate": _calibration_cache["em_hit_rate_1sigma"],
            "avg_vix": _calibration_cache["avg_vix"],
            "annualized_vol": _calibration_cache["annualized_realized_vol"],
        }
    return jsonify(info)


@app.route("/api/calibration")
@api_auth_required
def api_calibration():
    """Return full calibration statistics from real data."""
    global _calibration_cache
    if _calibration_cache and "error" not in _calibration_cache:
        return jsonify({"status": "ok", "data": _calibration_cache})
    bars = _get_real_data()
    _calibration_cache = calibrate_from_bars(bars)
    if "error" not in _calibration_cache:
        return jsonify({"status": "ok", "data": _calibration_cache})
    return jsonify({"status": "error", "message": "Insufficient data for calibration"}), 400


@app.route("/api/refresh-data", methods=["POST"])
@api_auth_required
def api_refresh_data():
    """Force re-download and recalibrate."""
    global _real_data_cache, _calibration_cache
    try:
        csv_path = fetch_and_cache(start="2020-01-02", force=True)
        bars = load_csv_data(csv_path)
        _real_data_cache = bars
        _calibration_cache = calibrate_from_bars(bars)
        return jsonify({
            "status": "ok",
            "message": f"Downloaded {len(bars)} days, calibration updated",
            "info": get_data_info(),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/live-quote")
@api_auth_required
def api_live_quote():
    """Fetch current SPX and VIX prices from Yahoo Finance."""
    try:
        import yfinance as yf

        def _get_quote(ticker_str):
            """Get open and last price, preferring .history() over fast_info."""
            tk = yf.Ticker(ticker_str)
            # Primary: .history() is more reliable than fast_info
            try:
                hist = tk.history(period="1d")
                if not hist.empty:
                    return (
                        round(float(hist['Open'].iloc[-1]), 2),
                        round(float(hist['Close'].iloc[-1]), 2),
                    )
            except Exception:
                pass
            # Fallback: fast_info (can return stale data)
            try:
                fi = tk.fast_info
                opn = getattr(fi, 'open', None) or getattr(fi, 'last_price', None)
                lst = getattr(fi, 'last_price', None) or opn
                if opn:
                    return round(float(opn), 2), round(float(lst), 2)
            except Exception:
                pass
            return None, None

        spx_open, spx_last = _get_quote("^GSPC")
        vix_open, vix_last = _get_quote("^VIX")

        return jsonify({
            "status": "ok",
            "spx_open": spx_open,
            "spx_last": spx_last,
            "vix_open": vix_open,
            "vix_last": vix_last,
        })
    except ImportError:
        return jsonify({"status": "error", "message": "yfinance not installed"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ────────────────────────────────────────────────────────────
# API — Strategy
# ────────────────────────────────────────────────────────────

@app.route("/api/strategy", methods=["POST"])
@api_auth_required
def api_strategy():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Request body required"}), 400
    try:
        spx_open = float(data.get("spx_open", 6000))
        vix = float(data.get("vix", 15.5))
        if not (100 < spx_open < 50000):
            return jsonify({"status": "error", "message": "SPX price out of valid range (100-50000)"}), 400
        if not (0.5 < vix < 150):
            return jsonify({"status": "error", "message": "VIX out of valid range (0.5-150)"}), 400
        trade_date = date.fromisoformat(data.get("trade_date", date.today().isoformat()))
        entry_time_str = data.get("entry_time", "0935")
        entry_time = time(int(entry_time_str[:2]), int(entry_time_str[2:]))

        em_override = data.get("expected_move")
        if em_override:
            em_override = float(em_override)

        risk_profile = RiskProfile(data.get("risk_profile", "moderate"))
        regime_str = data.get("regime", "auto")
        regime_override = Regime(regime_str) if regime_str != "auto" else None

        rec = generate_strategy(
            spx_open=spx_open, vix=vix, trade_date=trade_date,
            entry_time=entry_time, expected_move_override=em_override,
            risk_profile=risk_profile, regime_override=regime_override,
            calibration=_calibration_cache,
        )
        return jsonify({"status": "ok", "data": rec.to_dict()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ────────────────────────────────────────────────────────────
# API — Backtest (uses REAL data by default)
# ────────────────────────────────────────────────────────────

@app.route("/api/backtest", methods=["POST"])
@api_auth_required
def api_backtest():
    data = request.get_json() or {}
    try:
        risk_profile = RiskProfile(data.get("risk_profile", "moderate"))
        min_vix = float(data.get("min_vix", 0))
        max_vix = float(data.get("max_vix", 100))
        debit = data.get("debit_per_side")
        if debit:
            debit = float(debit)

        source = data.get("data_source", "real")

        if source == "real":
            all_bars = _get_real_data()
            bars = _filter_data(
                all_bars,
                data.get("start_date"),
                data.get("end_date"),
            )
        elif source == "csv" and data.get("csv_path"):
            bars = load_csv_data(data["csv_path"])
        else:
            # Synthetic fallback
            start = date.fromisoformat(data.get("start_date", "2023-01-02"))
            end = date.fromisoformat(data.get("end_date", "2025-12-31"))
            init_spx = float(data.get("initial_spx", 4700))
            init_vix = float(data.get("initial_vix", 15))
            seed = int(data.get("seed", 42))
            bars = generate_synthetic_data(start, end, init_spx, init_vix, seed)

        if not bars:
            return jsonify({"status": "error", "message": "No data for selected range"}), 400

        summary = run_backtest(bars, risk_profile, debit, min_vix, max_vix)
        result = summary.to_dict()
        # Add metadata for frontend display
        result["_data_source"] = source
        result["_date_range"] = [bars[0].trade_date.isoformat(), bars[-1].trade_date.isoformat()]
        result["_spx_start"] = bars[0].spx_open
        result["_spx_end"] = bars[-1].spx_close
        return jsonify({"status": "ok", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/optimize", methods=["POST"])
@api_auth_required
def api_optimize():
    data = request.get_json() or {}
    try:
        risk_profile = RiskProfile(data.get("risk_profile", "moderate"))

        source = data.get("data_source", "real")
        if source == "real":
            all_bars = _get_real_data()
            bars = _filter_data(
                all_bars,
                data.get("start_date"),
                data.get("end_date"),
            )
        else:
            start = date.fromisoformat(data.get("start_date", "2023-01-02"))
            end = date.fromisoformat(data.get("end_date", "2025-12-31"))
            init_spx = float(data.get("initial_spx", 4700))
            init_vix = float(data.get("initial_vix", 15))
            seed = int(data.get("seed", 42))
            bars = generate_synthetic_data(start, end, init_spx, init_vix, seed)

        result = optimize_parameters(bars, risk_profile=risk_profile)
        return jsonify({"status": "ok", "data": result.to_dict()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ────────────────────────────────────────────────────────────
# API — Traps
# ────────────────────────────────────────────────────────────

@app.route("/api/traps", methods=["POST"])
@api_auth_required
def api_traps():
    """
    Generate trap chains.
    
    The trap butterfly centers at the open price, filling the gap 
    between the put fly and call fly profit zones. This creates a 
    continuous profit tent across the full range.
    
    Example: Open=6975, Call fly 6980/7040/7100, Put fly 6970/6910/6850
      Trap call fly: 6915C / 6975C / 7035C  (centered at open, same width)
      Trap put fly:  7035P / 6975P / 6915P  (centered at open, same width)
      Outside call:  sell the 7100C  (cap upper risk)
      Outside put:   sell the 6850P  (cap lower risk)
    """
    data = request.get_json()
    try:
        trade_date = data.get("trade_date", date.today().isoformat())
        spx = float(data["spx_open"])
        call_center = float(data["call_center"])
        put_center = float(data["put_center"])
        bfly_width = float(data["butterfly_width"])
        trap_width = float(data.get("trap_width", bfly_width))
        trap_width = max(25, round(trap_width / 5) * 5)

        date_str = trade_date.replace("-", "")[2:]

        # Trap center = open price (rounded to 5)
        trap_center = round(spx / 5) * 5

        # Trap call butterfly: centered at open, bridges the gap
        # Lower wing at center - width, upper wing at center + width
        trap_call_lower = int(trap_center - trap_width)
        trap_call_upper = int(trap_center + trap_width)
        trap_center_int = int(trap_center)

        # Trap put butterfly: same center, same width (gives extra coverage)
        trap_put_upper = int(trap_center + trap_width)
        trap_put_lower = int(trap_center - trap_width)

        # Outside singles: sell the outer wings of the main butterflies
        outside_call_strike = int(call_center + bfly_width)
        outside_put_strike = int(put_center - (spx - start_offset - put_center)) if False else int(put_center - bfly_width)

        chains = {
            "trap_call_fly": (
                f".SPXW{date_str}C{trap_call_lower}-2*"
                f".SPXW{date_str}C{trap_center_int}+"
                f".SPXW{date_str}C{trap_call_upper}"
            ),
            "trap_put_fly": (
                f".SPXW{date_str}P{trap_put_upper}-2*"
                f".SPXW{date_str}P{trap_center_int}+"
                f".SPXW{date_str}P{trap_put_lower}"
            ),
            "outside_call": f".SPXW{date_str}C{outside_call_strike}",
            "outside_put": f".SPXW{date_str}P{int(put_center - bfly_width)}",
            # Include metadata for the UI
            "trap_center": trap_center_int,
            "trap_width": trap_width,
        }
        return jsonify({"status": "ok", "data": chains})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/trap-score", methods=["POST"])
@api_auth_required
def api_trap_score():
    """
    Real-time trap confidence score.
    
    Called at trap decision time (1:30-2:30 PM) with current market data.
    Returns a score (0-100), grade (A-D), and specific trap recommendation.
    
    Required inputs:
      spx_open: today's opening price
      spx_current: current SPX price
      spx_high: today's high so far
      spx_low: today's low so far
      vix: current VIX
      expected_move: today's EM (from strategy page)
    """
    data = request.get_json() or {}
    try:
        spx_open = float(data["spx_open"])
        spx_high = float(data["spx_high"])
        spx_low = float(data["spx_low"])
        vix = float(data["vix"])
        em = float(data["expected_move"])
        bfly_width = float(data.get("butterfly_width", 45))
        
        # Compute signals
        max_move = max(spx_high - spx_open, spx_open - spx_low)
        max_move_vs_em = max_move / em if em > 0 else 0
        
        # Use recent data for 5-day range average
        all_bars = _get_real_data()
        if all_bars and len(all_bars) >= 5:
            recent = all_bars[-5:]
            avg_range_5d = np.mean([b.spx_high - b.spx_low for b in recent])
            avg_vix_5d = np.mean([b.vix_open for b in recent])
        else:
            avg_range_5d = em * 1.2  # rough fallback
            avg_vix_5d = vix
        
        today_range = spx_high - spx_low
        range_expansion = today_range / avg_range_5d if avg_range_5d > 0 else 1.0
        vix_change = vix - avg_vix_5d
        
        result = compute_trap_score(max_move_vs_em, range_expansion, vix_change)
        
        # Add context
        result["inputs"] = {
            "max_move": round(max_move, 1),
            "max_move_vs_em": round(max_move_vs_em, 2),
            "today_range": round(today_range, 1),
            "avg_range_5d": round(avg_range_5d, 1),
            "range_expansion": round(range_expansion, 2),
            "vix_change_5d": round(vix_change, 1),
        }
        
        # Add trap strikes if action is full or half
        if result["action"] in ("full", "half"):
            trap_w_mult = result["trap_width_mult"]
            trap_w = round(max(10, bfly_width * trap_w_mult) / 5) * 5
            
            # Determine direction
            up = spx_high - spx_open
            down = spx_open - spx_low
            call_center = round_to_strike(spx_open + bfly_width)
            put_center = round_to_strike(spx_open - bfly_width)
            
            if up > down:
                tc = round_to_strike((spx_open + call_center) / 2)
                result["trap_direction"] = "CALL"
            else:
                tc = round_to_strike((spx_open + put_center) / 2)
                result["trap_direction"] = "PUT"
            
            result["trap_strikes"] = {
                "lower": tc - trap_w,
                "center": tc,
                "upper": tc + trap_w,
                "width": trap_w,
            }
        
        return jsonify({"status": "ok", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ────────────────────────────────────────────────────────────
# Run
# ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({"status": "error", "message": "Endpoint not found"}), 404
    return render_template_string("""
    {% extends "base.html" %}
    {% block content %}
    <div style="text-align:center;padding:80px 20px">
      <div style="font-size:48px;font-weight:700;color:var(--text-muted);margin-bottom:8px">404</div>
      <p style="color:var(--text-dim);margin-bottom:24px">Page not found</p>
      <a href="/" class="btn btn-primary">Go Home</a>
    </div>
    {% endblock %}
    """), 404


@app.route("/health")
def health_check():
    """Health check for monitoring and load balancers."""
    checks = {"status": "ok"}
    try:
        data = _get_real_data()
        checks["data"] = f"{len(data)} bars" if data else "no data"
    except Exception as e:
        checks["data"] = f"error: {e}"
        checks["status"] = "degraded"
    checks["calibrated"] = bool(_calibration_cache and "error" not in _calibration_cache)
    return jsonify(checks), 200 if checks["status"] == "ok" else 503

@app.errorhandler(500)
def server_error(e):
    if request.path.startswith('/api/'):
        return jsonify({"status": "error", "message": "Internal server error"}), 500
    return render_template_string("""
    {% extends "base.html" %}
    {% block content %}
    <div style="text-align:center;padding:80px 20px">
      <div style="font-size:48px;font-weight:700;color:var(--red);margin-bottom:8px">500</div>
      <p style="color:var(--text-dim);margin-bottom:24px">Something went wrong</p>
      <a href="/" class="btn btn-primary">Go Home</a>
    </div>
    {% endblock %}
    """), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
