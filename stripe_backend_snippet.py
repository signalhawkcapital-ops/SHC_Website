
# --- Add to your Flask app.py ---
# pip install stripe
import os, stripe
from flask import request, jsonify

# Set your Stripe keys via env vars
# export STRIPE_SECRET_KEY='sk_live_...'
# export STRIPE_PRICE_MONTHLY='price_...'
# export STRIPE_PRICE_YEARLY='price_...'
# export STRIPE_PRICE_LIFETIME='price_...'

stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    data = request.get_json(force=True)
    email = data.get('email')
    phone = data.get('phone')
    plan  = data.get('plan')

    # Map the plan to Stripe Price IDs
    price_lookup = {
        'monthly': os.environ.get('STRIPE_PRICE_MONTHLY'),
        'yearly': os.environ.get('STRIPE_PRICE_YEARLY'),
        'lifetime': os.environ.get('STRIPE_PRICE_LIFETIME'),
    }
    price_id = price_lookup.get(plan)
    if not price_id:
        return ('Unknown plan', 400)

    # For subscriptions (monthly/yearly) use mode='subscription'
    mode = 'subscription' if plan in ('monthly', 'yearly') else 'payment'

    try:
        session = stripe.checkout.Session.create(
            mode=mode,
            line_items=[{'price': price_id, 'quantity': 1}],
            customer_email=email,
            phone_number_collection={'enabled': True},
            metadata={'alerts_email': email, 'alerts_phone': phone, 'plan': plan},
            success_url=url_for('index', _external=True) + '?checkout=success',
            cancel_url=url_for('signup', _external=True) + '?checkout=canceled',
            automatic_tax={'enabled': True},
            allow_promotion_codes=True
        )
        return jsonify({'url': session.url})
    except Exception as e:
        return (str(e), 500)

# Optional: Stripe webhook to grant access after payment
# Set endpoint secret in STRIPE_WEBHOOK_SECRET
from flask import abort
@app.route('/stripe-webhook', methods=['POST'])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    endpoint_secret = os.environ.get('STRIPE_WEBHOOK_SECRET')
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except Exception as e:
        return (str(e), 400)

    # Handle completed checkout
    if event['type'] in ('checkout.session.completed', 'customer.subscription.created'):
        session = event['data']['object']
        # TODO: mark user active in your DB using session.get('customer_email') or session['metadata']
        # You can also send a welcome email/SMS here.
    return ('', 200)
