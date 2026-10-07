"""Outbound notifications: Slack (with Approve/Reject buttons) and Twilio SMS."""

from __future__ import annotations

import logging
import os

from .config import Config

log = logging.getLogger(__name__)

SMS_LIMIT = 1500  # Twilio concatenates up to 1600 chars


def send_slack(text: str, proposal_code: str | None = None) -> None:
    token, channel = os.environ.get("SLACK_BOT_TOKEN"), os.environ.get("SLACK_CHANNEL_ID")
    if not token or not channel:
        log.info("Slack not configured; skipping")
        return
    from slack_sdk import WebClient

    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": text[:2900]}}]
    if proposal_code:
        blocks.append(
            {
                "type": "actions",
                "block_id": f"proposal_{proposal_code}",
                "elements": [
                    {"type": "button", "style": "primary", "action_id": "approve_trades",
                     "text": {"type": "plain_text", "text": "Approve & execute"}, "value": proposal_code,
                     "confirm": {"title": {"type": "plain_text", "text": "Place these orders?"},
                                 "text": {"type": "plain_text", "text": "Orders go to Robinhood immediately."},
                                 "confirm": {"type": "plain_text", "text": "Execute"},
                                 "deny": {"type": "plain_text", "text": "Cancel"}}},
                    {"type": "button", "style": "danger", "action_id": "reject_trades",
                     "text": {"type": "plain_text", "text": "Reject"}, "value": proposal_code},
                ],
            }
        )
    WebClient(token=token).chat_postMessage(channel=channel, text=text[:3000], blocks=blocks)


def send_sms(text: str) -> None:
    sid, tok = os.environ.get("TWILIO_ACCOUNT_SID"), os.environ.get("TWILIO_AUTH_TOKEN")
    from_, to = os.environ.get("TWILIO_FROM_NUMBER"), os.environ.get("ALERT_TO_NUMBER")
    if not all([sid, tok, from_, to]):
        log.info("Twilio not configured; skipping SMS")
        return
    from twilio.rest import Client

    if len(text) > SMS_LIMIT:
        text = text[: SMS_LIMIT - 20] + "\n...(see Slack)"
    Client(sid, tok).messages.create(body=text, from_=from_, to=to)


def notify(cfg: Config, text: str, sms_text: str | None = None, proposal_code: str | None = None) -> None:
    for enabled, fn, args in (
        (cfg.notify.slack, send_slack, (text, proposal_code)),
        (cfg.notify.sms, send_sms, (sms_text or text,)),
    ):
        if not enabled:
            continue
        try:
            fn(*args)
        except Exception:  # noqa: BLE001 - a failed alert must not crash the run
            log.exception("Notification via %s failed", fn.__name__)
