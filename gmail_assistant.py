"""
title: Gmail Assistant
author: local
description: Fetches, categorises, and organises Gmail inbox emails. Flags unwanted emails
             for manual deletion using Gmail labels. Remembers processed emails via a
             Processed label so emails are never re-scanned on subsequent runs.
version: 0.2
requirements: google-auth-oauthlib,google-auth-httplib2,google-api-python-client
"""

import json
import os
import base64
import re
from typing import Callable, Any

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

# Secrets folder mounted from the host machine into the container.
# On your Mac: ~/.openwebui-secrets/
# Mounted into the container at: /secrets
SECRETS_DIR = "/llm_local_gmail_creds"

CREDENTIALS_FILE = os.path.join(SECRETS_DIR, "cred_llm_local.json")
TOKEN_FILE = os.path.join(SECRETS_DIR, "cred_llm_token.json")

# Gmail labels used as state
LABEL_PROCESSED = "AI-Processed"
LABEL_TO_DELETE = "AI-ToDelete"

# Category labels — nested under AI/ to keep them grouped
CATEGORY_LABELS = {
    "important": "AI/Important",
    "work": "AI/Work",
    "newsletter": "AI/Newsletter",
    "promotion": "AI/Promotion",
    "spam": "AI/Spam",
    "receipt": "AI/Receipt",
    "other": "AI/Other",
}

# How many emails to return per batch to avoid context overflow on 4B models
BATCH_SIZE = 5

# Max snippet length passed to the model per email
SNIPPET_LENGTH = 200

# Sender patterns that are always automated — flag for deletion without asking the model
AUTOMATED_SENDER_PATTERNS = [
    "noreply",
    "no-reply",
    "donotreply",
    "do-not-reply",
    "notifications@",
    "updates@",
    "newsletter@",
    "news@",
    "mailer@",
    "info@",
    "alerts@",
    "digest@",
    "subscriptions@",
    "obserwowane@",
    "invitations@linkedin",
    "messages-noreply@linkedin",
]


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _get_service():
    """Authenticate and return a Gmail API service object."""
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"credentials.json not found at {CREDENTIALS_FILE}. "
                    "Download it from Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def _get_or_create_label(service, name: str) -> str:
    """Return the label ID for `name`, creating it if it doesn't exist."""
    result = service.users().labels().list(userId="me").execute()
    for label in result.get("labels", []):
        if label["name"] == name:
            return label["id"]

    created = (
        service.users()
        .labels()
        .create(
            userId="me",
            body={
                "name": name,
                "labelListVisibility": "labelShow",
                "messageListVisibility": "show",
            },
        )
        .execute()
    )
    return created["id"]


def _ensure_labels(service) -> dict:
    """Ensure all required labels exist and return a name→id map."""
    ids = {}
    ids[LABEL_PROCESSED] = _get_or_create_label(service, LABEL_PROCESSED)
    ids[LABEL_TO_DELETE] = _get_or_create_label(service, LABEL_TO_DELETE)
    for key, label_name in CATEGORY_LABELS.items():
        ids[key] = _get_or_create_label(service, label_name)
    return ids


# ---------------------------------------------------------------------------
# Email helpers
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    """Strip zero-width and invisible unicode characters from email text."""
    import re

    text = re.sub(r"[͏​‌‍⁠﻿­]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_snippet(service, msg_id: str) -> str:
    """Return subject, sender and a short body snippet for an email."""
    msg = (
        service.users()
        .messages()
        .get(
            userId="me",
            id=msg_id,
            format="metadata",
            metadataHeaders=["Subject", "From", "Date"],
        )
        .execute()
    )

    headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
    subject = _clean(headers.get("Subject", "(no subject)"))
    sender = _clean(headers.get("From", "(unknown sender)"))
    date = headers.get("Date", "")
    snippet = _clean(msg.get("snippet", ""))[:SNIPPET_LENGTH]

    return subject, sender, date, snippet


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

MAX_FETCH_CALLS = 10


class Tools:
    def __init__(self):
        self._fetch_count = 0

    def fetch_unprocessed_emails(self) -> str:
        """
        Fetch a batch of unprocessed emails from the Gmail inbox (last 24 hours).
        Returns up to 8 emails with their ID, sender, subject, date and a short snippet.
        Call this first to get emails to categorise. If there are more emails, call it
        again after processing the current batch — it will continue from where it left off.
        """
        self._fetch_count += 1
        if self._fetch_count > MAX_FETCH_CALLS:
            return json.dumps(
                {
                    "status": "limit_reached",
                    "message": f"Fetch limit of {MAX_FETCH_CALLS} batches reached. Stop fetching and present your final summary now.",
                    "more_remaining": False,
                    "emails": [],
                }
            )

        try:
            service = _get_service()
            label_ids = _ensure_labels(service)

            # Fetch emails since midnight yesterday that have NOT been processed yet
            from datetime import date, timedelta

            since = (date.today() - timedelta(days=1)).strftime("%Y/%m/%d")
            query = f"in:inbox after:{since} -label:{LABEL_PROCESSED}"
            result = (
                service.users()
                .messages()
                .list(userId="me", q=query, maxResults=BATCH_SIZE)
                .execute()
            )

            messages = result.get("messages", [])
            if not messages:
                return json.dumps(
                    {
                        "status": "done",
                        "message": "All emails have been processed. Present your final summary now.",
                        "more_remaining": False,
                        "emails": [],
                    }
                )

            emails = []
            for msg in messages:
                subject, sender, date, snippet = _extract_snippet(service, msg["id"])
                sender_lower = sender.lower()
                is_automated = any(p in sender_lower for p in AUTOMATED_SENDER_PATTERNS)
                emails.append(
                    {
                        "id": msg["id"],
                        "from": sender,
                        "subject": subject,
                        "date": date,
                        "snippet": snippet,
                        "auto_to_delete": is_automated,
                    }
                )

            # Do a second count query to get accurate remaining count
            count_result = (
                service.users()
                .messages()
                .list(userId="me", q=query, maxResults=BATCH_SIZE + 1)
                .execute()
            )
            total = len(count_result.get("messages", []))
            more_remaining = total > BATCH_SIZE

            return json.dumps(
                {
                    "status": "ok",
                    "batch_size": len(emails),
                    "more_remaining": more_remaining,
                    "next_action": (
                        "You MUST call categorise_email() for EACH email — no exceptions, no summaries until all calls are done. "
                        "Use the EXACT id value. For emails where auto_to_delete is true, set to_delete=true automatically. "
                        "Only call fetch_unprocessed_emails() again after ALL categorise_email() calls are complete and more_remaining is true."
                    ),
                    "emails": emails,
                }
            )

        except FileNotFoundError as e:
            return json.dumps({"status": "error", "message": str(e)})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Gmail error: {str(e)}"})

    def categorise_email(
        self,
        email_id: str,
        category: str,
        to_delete: bool,
        reason: str,
    ) -> str:
        """
        Categorise a single email and apply the appropriate Gmail labels.
        Marks the email as processed so it won't be fetched again.

        :param email_id: The email ID returned by fetch_unprocessed_emails.
        :param category: One of: important, work, newsletter, promotion, spam, receipt, other.
        :param to_delete: Set true if this email should be moved to the ToDelete folder
                          for manual review and deletion.
        :param reason: A short explanation of why you assigned this category and/or flagged
                       it for deletion. Used to show the user a summary.
        """
        try:
            service = _get_service()
            label_ids = _ensure_labels(service)

            category = category.lower().strip()
            if category not in CATEGORY_LABELS:
                category = "other"

            labels_to_add = [
                label_ids[LABEL_PROCESSED],
                label_ids[category],
            ]
            labels_to_remove = []

            if to_delete:
                labels_to_add.append(label_ids[LABEL_TO_DELETE])
                # Move out of inbox so it's clearly staged for deletion
                labels_to_remove.append("INBOX")

            service.users().messages().modify(
                userId="me",
                id=email_id,
                body={
                    "addLabelIds": labels_to_add,
                    "removeLabelIds": labels_to_remove,
                },
            ).execute()

            action = (
                "moved to ToDelete"
                if to_delete
                else f"labelled as {CATEGORY_LABELS[category]}"
            )
            return json.dumps(
                {
                    "status": "ok",
                    "email_id": email_id,
                    "action": action,
                    "reason": reason,
                }
            )

        except Exception as e:
            return json.dumps(
                {"status": "error", "email_id": email_id, "message": str(e)}
            )

    def get_todelete_summary(self) -> str:
        """
        Returns a summary of all emails currently in the ToDelete folder.
        Use this to show the user what has been flagged for deletion so they can
        review before manually emptying the folder.
        """
        try:
            service = _get_service()
            _ensure_labels(service)

            result = (
                service.users()
                .messages()
                .list(
                    userId="me",
                    q=f"label:{LABEL_TO_DELETE}",
                    maxResults=50,
                )
                .execute()
            )

            messages = result.get("messages", [])
            if not messages:
                return json.dumps(
                    {
                        "status": "ok",
                        "message": "The ToDelete folder is empty.",
                        "emails": [],
                    }
                )

            emails = []
            for msg in messages:
                subject, sender, date, _ = _extract_snippet(service, msg["id"])
                emails.append(
                    {
                        "id": msg["id"],
                        "from": sender,
                        "subject": subject,
                        "date": date,
                    }
                )

            return json.dumps(
                {
                    "status": "ok",
                    "count": len(emails),
                    "message": (
                        f"{len(emails)} email(s) are staged for deletion. "
                        "Review the list below and delete them manually in Gmail "
                        "by emptying the AI-ToDelete label."
                    ),
                    "emails": emails,
                }
            )

        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

