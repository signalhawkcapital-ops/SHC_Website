
# Payments + Users + SMS (Twilio) additions

Environment variables to set before running the app:

STRIPE_SECRET_KEY=sk_...
STRIPE_PRICE_MONTHLY=price_...
STRIPE_PRICE_YEARLY=price_...
STRIPE_PRICE_LIFETIME=price_...
STRIPE_WEBHOOK_SECRET=whsec_...

TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_FROM_NUMBER=+15551234567

# Install deps
pip install stripe twilio

# Webhook (run with your public URL tunnel or prod domain):
stripe listen --forward-to http://localhost:5000/stripe-webhook
