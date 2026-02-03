# Signal Hawk Capital

A data-driven investment signal platform built with Flask. Delivers systematic buy/sell alerts on S&P 500 and SPXL strategies via email and SMS, with a built-in investment growth analyzer for backtesting.

---

## Quick Start (PyCharm / Local)

### 1. Clone & Open

```bash
git clone <your-repo-url>
cd shc-website
```

Open the `shc-website` folder in PyCharm as a project.

### 2. Create a Virtual Environment

**PyCharm:** File → Settings → Project → Python Interpreter → Add Interpreter → Virtualenv

**Terminal:**
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Open `.env` and fill in your values. At minimum for local dev:

| Variable | Required For |
|---|---|
| `FLASK_SECRET_KEY` | Sessions (set any random string) |
| `STRIPE_SECRET_KEY` | Checkout flow |
| `STRIPE_WEBHOOK_SECRET` | Webhook verification |
| `STRIPE_PRICE_*` | Plan pricing (6 price IDs from Stripe dashboard) |
| `SMTP_HOST`, `SMTP_USERNAME`, `SMTP_PASSWORD` | Email delivery |

The app will start without Stripe/SMTP configured — those features will log warnings but won't crash.

### 5. Run

**PyCharm:** Right-click `app.py` → Run, or create a Flask run configuration pointing to `app.py`.

**Terminal:**
```bash
python app.py
```

Visit [http://localhost:5000](http://localhost:5000).

For debug mode:
```bash
FLASK_DEBUG=1 python app.py
```

---

## Project Structure

```
shc-website/
├── app.py                  # Flask application (all routes, models, logic)
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version for Render
├── Procfile                # Gunicorn start command
├── render.yaml             # Render deployment blueprint
├── .env.example            # Environment variable template
├── .gitignore
├── static/
│   ├── style.css           # Full stylesheet (dark theme)
│   ├── SHC_Logo.png        # Nav logo
│   ├── Signal_Hawk_Captial_Logo.png
│   ├── SHCpreview.png      # Hero preview image (add your own)
│   └── favicon.ico         # Browser tab icon (add your own)
├── templates/
│   ├── layout.html         # Base template (nav, footer, flash messages)
│   ├── home.html           # Landing page with hero + features grid
│   ├── signup.html         # Subscription page (Stripe Checkout)
│   ├── login.html          # Email/password login
│   ├── account.html        # User dashboard (profile, billing, notifications)
│   ├── growth_analyzer.html# Investment backtesting tool
│   ├── resources.html      # Educational content (strategy, fundamentals, brokerages)
│   ├── about.html          # Company info
│   ├── account_setup.html  # Post-checkout password creation
│   ├── account_request_setup.html  # Request a new setup link
│   ├── forgot_password.html        # Request password reset
│   ├── reset_password_confirm.html # Set new password via reset link
│   ├── terms.html          # Terms and conditions
│   ├── privacy.html        # Privacy policy
│   └── disclosures.html    # Investment disclosures
└── signal_hawk.db          # SQLite database (auto-created on first run)
```

---

## Features

### Public Pages
- **Home** — Hero section, 6-card features grid, checkout success banner
- **Resources** — Strategy overview, investment fundamentals (compounding, DCA, diversification, volatility, taxes), account types, brokerage onboarding guides
- **About** — Company info, methodology overview, contact
- **Growth Analyzer** — Interactive backtesting tool comparing Signal Hawk Strategy, HG_SIM, SPXL, S&P 500, NVDA, AMZN, AAPL, TSLA with configurable start date, investment amount, expense ratios, and taxes

### Subscription & Payments
- **Stripe Checkout** integration with Individual ($120/mo, $1,000/yr, $7,000/10yr) and Professional (10×) pricing tiers
- **Webhook handler** auto-creates user records and sends onboarding emails on successful checkout
- **Billing portal** redirect for subscription management (cancel, update payment method)

### Authentication
- Email/password login with secure hashed passwords (Werkzeug)
- Token-based account setup flow (emailed after Stripe checkout)
- Password reset via timed email links
- Optional Google and Apple OAuth (requires `authlib` + `PyJWT`)
- `@login_required` decorator for protected routes

### Account Dashboard
- Profile management (email, phone)
- Notification preferences (email alerts, SMS alerts, marketing)
- Password change
- Stripe billing portal access
- Subscription status display

### UI/UX
- Dark theme with glassmorphic design (DM Sans + Space Grotesk typography)
- Fully responsive with mobile hamburger nav
- Conditional nav (logged-in users see Account/Log out; logged-out see Log in)
- Global flash messages (success/error)

---

## Deploy to Render

### Option A: Blueprint (Recommended)

1. Push your repo to GitHub/GitLab.
2. In Render, click **New** → **Blueprint** → connect your repo.
3. Render reads `render.yaml` and auto-configures the service.
4. Add your environment variables in Render's dashboard (all the `sync: false` vars).
5. Set `APP_BASE_URL` to your Render URL (e.g. `https://signal-hawk-capital.onrender.com`).

### Option B: Manual

1. **New Web Service** → connect your repo.
2. **Runtime:** Python
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
5. **Add a Disk** → mount path: `/data`, size: 1 GB (for SQLite persistence).
6. Add all environment variables from `.env.example`.

### Stripe Webhook

After deploying, add a webhook endpoint in Stripe:

- **URL:** `https://your-app.onrender.com/stripe-webhook`
- **Events:** `checkout.session.completed`
- Copy the signing secret to `STRIPE_WEBHOOK_SECRET`.

---

## Static Assets You Need to Provide

The following files are referenced but not included in the repo (add your own):

| File | Used In | Purpose |
|---|---|---|
| `static/SHCpreview.png` | home.html | Hero section strategy preview image |
| `static/favicon.ico` | layout.html | Browser tab icon |
| `static/Signal_Hawk_Capital_Disclosures.pdf` | terms.html, disclosures.html | Downloadable disclosures PDF |
| `static/Signal_Hawk_Capital_Privacy_Policy.pdf` | privacy.html | Downloadable privacy policy PDF |

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `FLASK_SECRET_KEY` | Yes | Session encryption key |
| `FLASK_DEBUG` | No | Set to `1` for debug mode |
| `FLASK_ENV` | No | `production` or `development` |
| `APP_BASE_URL` | Yes | Full URL of deployed app |
| `DB_PATH` | No | SQLite file path (auto-detects Render disk) |
| `STRIPE_SECRET_KEY` | Yes | Stripe API secret key |
| `STRIPE_WEBHOOK_SECRET` | Yes | Stripe webhook signing secret |
| `STRIPE_PRICE_INDIVIDUAL_MONTHLY` | Yes | Stripe price ID |
| `STRIPE_PRICE_INDIVIDUAL_YEARLY` | Yes | Stripe price ID |
| `STRIPE_PRICE_INDIVIDUAL_LIFETIME` | Yes | Stripe price ID |
| `STRIPE_PRICE_PRO_MONTHLY` | Yes | Stripe price ID |
| `STRIPE_PRICE_PRO_YEARLY` | Yes | Stripe price ID |
| `STRIPE_PRICE_PRO_LIFETIME` | Yes | Stripe price ID |
| `STRIPE_PRICE_MONTHLY` | No | Fallback price IDs |
| `STRIPE_PRICE_YEARLY` | No | Fallback price IDs |
| `STRIPE_PRICE_LIFETIME` | No | Fallback price IDs |
| `SMTP_HOST` | Yes | SMTP server hostname |
| `SMTP_PORT` | No | SMTP port (default: 587) |
| `SMTP_USERNAME` | Yes | SMTP login |
| `SMTP_PASSWORD` | Yes | SMTP password / app password |
| `EMAIL_FROM` | No | From address (defaults to SMTP_USERNAME) |
| `GOOGLE_CLIENT_ID` | No | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | No | Google OAuth secret |
| `APPLE_CLIENT_ID` | No | Apple Sign In client ID |
| `APPLE_TEAM_ID` | No | Apple Developer team ID |
| `APPLE_KEY_ID` | No | Apple Sign In key ID |
| `APPLE_PRIVATE_KEY_PEM` | No | Apple private key (PEM format) |

---

## License

Proprietary — Signal Hawk Capital, LLC. All rights reserved.
