"""Long-running listener that turns your Slack clicks / SMS replies into orders.

- Slack: Socket Mode (outbound websocket) - no public URL or open port needed.
- SMS:   Twilio posts replies to /sms on this Flask app. Needs a public HTTPS
         URL (e.g. Caddy/nginx on your Oracle VM). Signatures are verified and
         only ALERT_TO_NUMBER may approve.
"""

from __future__ import annotations

import logging
import os
import re
import threading

from .agent import execute, reject
from .approvals import ApprovalStore
from .config import Config

log = logging.getLogger(__name__)

SMS_COMMAND = re.compile(r"^\s*(YES|Y|APPROVE|NO|N|REJECT)\s+([A-Z0-9]{6})\s*$", re.I)


def _in_background(fn, *args, **kwargs) -> None:
    threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()


def start_slack(cfg: Config) -> None:
    bot, app_token = os.environ.get("SLACK_BOT_TOKEN"), os.environ.get("SLACK_APP_TOKEN")
    approver = os.environ.get("SLACK_APPROVER_USER_ID")
    if not (bot and app_token):
        log.info("Slack Socket Mode not configured")
        return
    if not approver:
        raise RuntimeError("Set SLACK_APPROVER_USER_ID so only you can approve trades")

    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    app = App(token=bot)

    def handle(ack, body, respond, approve: bool):
        ack()
        user = body["user"]["id"]
        code = body["actions"][0]["value"]
        if user != approver:
            respond(text=f"<@{user}> is not authorized to act on trades.", replace_original=False)
            return
        respond(text=f"{'Executing' if approve else 'Rejecting'} proposal `{code}`...", replace_original=False)
        if approve:
            _in_background(execute, cfg, code, approver=f"slack:{user}")
        else:
            _in_background(reject, cfg, code, approver=f"slack:{user}")

    app.action("approve_trades")(lambda ack, body, respond: handle(ack, body, respond, True))
    app.action("reject_trades")(lambda ack, body, respond: handle(ack, body, respond, False))

    handler = SocketModeHandler(app, app_token)
    threading.Thread(target=handler.start, daemon=True, name="slack-socket").start()
    log.info("Slack Socket Mode listener started")


def create_sms_app(cfg: Config):
    from flask import Flask, abort, request
    from twilio.request_validator import RequestValidator
    from twilio.twiml.messaging_response import MessagingResponse

    app = Flask(__name__)
    validator = RequestValidator(os.environ.get("TWILIO_AUTH_TOKEN", ""))
    allowed_from = os.environ.get("ALERT_TO_NUMBER")
    public_url = os.environ.get("TWILIO_WEBHOOK_URL")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/sms")
    def sms():
        sig = request.headers.get("X-Twilio-Signature", "")
        if not validator.validate(public_url or request.url, request.form, sig):
            abort(403)
        if request.form.get("From") != allowed_from:
            abort(403)

        body = request.form.get("Body", "")
        resp = MessagingResponse()
        m = SMS_COMMAND.match(body)
        if body.strip().upper() == "STATUS":
            store = ApprovalStore(cfg.state_dir)
            resp.message(f"Agent running. Mode={cfg.mode}. Traded today: ${store.executed_notional_today():,.0f}")
        elif not m:
            resp.message("Reply 'YES <code>' to execute, 'NO <code>' to reject, or 'STATUS'.")
        else:
            verb, code = m.group(1).upper(), m.group(2).upper()
            if verb in ("YES", "Y", "APPROVE"):
                _in_background(execute, cfg, code, approver="sms")
                resp.message(f"Executing {code}...")
            else:
                _in_background(reject, cfg, code, approver="sms")
                resp.message(f"Rejecting {code}.")
        return str(resp), 200, {"Content-Type": "application/xml"}

    return app


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8080) -> None:
    start_slack(cfg)
    app = create_sms_app(cfg)
    # Bind to localhost; put a TLS reverse proxy (Caddy) in front for Twilio.
    app.run(host=host, port=port)
