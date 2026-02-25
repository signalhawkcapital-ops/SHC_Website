"""
Stripe billing module — subscription checkout, webhooks, customer portal.

Required env vars:
  STRIPE_SECRET_KEY       — sk_test_... or sk_live_...
  STRIPE_PUBLISHABLE_KEY  — pk_test_... or pk_live_...
  STRIPE_PRICE_ID         — price_... (monthly subscription price)
  STRIPE_WEBHOOK_SECRET   — whsec_... (webhook signing secret)
  APP_URL                 — https://yourdomain.com (for redirect URLs)
"""

import os
import json

# Stripe is optional — app works without it for local dev
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False
    stripe = None

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
APP_URL = os.environ.get("APP_URL", "http://localhost:5000")

if STRIPE_AVAILABLE and STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


def is_stripe_configured() -> bool:
    return STRIPE_AVAILABLE and bool(STRIPE_SECRET_KEY) and bool(STRIPE_PRICE_ID)


def create_checkout_session(user_email: str, user_id: int, stripe_customer_id: str = None) -> dict:
    """Create a Stripe Checkout session for subscription signup."""
    if not is_stripe_configured():
        return {"error": "Stripe not configured. Set STRIPE_SECRET_KEY and STRIPE_PRICE_ID."}

    params = {
        "mode": "subscription",
        "line_items": [{"price": STRIPE_PRICE_ID, "quantity": 1}],
        "success_url": f"{APP_URL}/account?session_id={{CHECKOUT_SESSION_ID}}",
        "cancel_url": f"{APP_URL}/account?cancelled=1",
        "client_reference_id": str(user_id),
        "metadata": {"user_id": str(user_id)},
    }

    if stripe_customer_id:
        params["customer"] = stripe_customer_id
    else:
        params["customer_email"] = user_email

    try:
        session = stripe.checkout.Session.create(**params)
        return {"url": session.url, "session_id": session.id}
    except stripe.error.StripeError as e:
        return {"error": str(e)}


def create_portal_session(stripe_customer_id: str) -> dict:
    """Create a Stripe Customer Portal session for managing subscription."""
    if not is_stripe_configured():
        return {"error": "Stripe not configured."}

    try:
        session = stripe.billing_portal.Session.create(
            customer=stripe_customer_id,
            return_url=f"{APP_URL}/account",
        )
        return {"url": session.url}
    except stripe.error.StripeError as e:
        return {"error": str(e)}


def handle_webhook(payload: bytes, sig_header: str) -> dict:
    """
    Process Stripe webhook events.
    Returns {event_type, customer_id, subscription_id, status} or {error}.
    """
    if not STRIPE_AVAILABLE:
        return {"error": "Stripe not available"}

    try:
        if STRIPE_WEBHOOK_SECRET:
            event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        else:
            event = json.loads(payload)
    except (stripe.error.SignatureVerificationError, ValueError) as e:
        return {"error": f"Webhook signature verification failed: {e}"}

    event_type = event.get("type", "")
    data = event.get("data", {}).get("object", {})

    result = {
        "event_type": event_type,
        "customer_id": data.get("customer"),
        "subscription_id": data.get("id") if "subscription" in event_type else data.get("subscription"),
    }

    if event_type == "checkout.session.completed":
        result["customer_id"] = data.get("customer")
        result["subscription_id"] = data.get("subscription")
        result["user_id"] = data.get("client_reference_id") or data.get("metadata", {}).get("user_id")
        result["status"] = "active"

    elif event_type in (
        "customer.subscription.created",
        "customer.subscription.updated",
    ):
        result["status"] = data.get("status")  # active, past_due, canceled, trialing

    elif event_type == "customer.subscription.deleted":
        result["status"] = "canceled"

    elif event_type == "invoice.payment_failed":
        result["status"] = "past_due"

    else:
        result["status"] = None  # unhandled event type

    return result


def get_subscription_info(stripe_customer_id: str) -> dict:
    """Get current subscription status from Stripe."""
    if not is_stripe_configured() or not stripe_customer_id:
        return {"status": "unknown"}

    try:
        subs = stripe.Subscription.list(customer=stripe_customer_id, limit=1)
        if subs.data:
            sub = subs.data[0]
            return {
                "status": sub.status,
                "plan": sub.items.data[0].price.id if sub.items.data else "",
                "current_period_end": sub.current_period_end,
                "cancel_at_period_end": sub.cancel_at_period_end,
            }
        return {"status": "none"}
    except stripe.error.StripeError:
        return {"status": "unknown"}
