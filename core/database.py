"""SQLite storage for encrypted biometric templates and audit events.

Only encrypted SFace embeddings are stored.  Original enrolment photographs are
never written to the database.  Each database operation uses a short-lived
connection, which keeps the class safe to call from the Tkinter main thread and
from camera/evaluation worker threads.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from core.security import (
    SecurityError,
    TemplateCipher,
    set_private_permissions,
    validate_display_name,
    validate_user_id,
)


class DatabaseError(RuntimeError):
    """Base error for database operations."""


class DuplicateUserError(DatabaseError):
    """Raised when an enrolment attempts to reuse an existing user ID."""


class UserNotFoundError(DatabaseError):
    """Raised when a requested user does not exist."""


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    display_name: str
    embedding: np.ndarray
    embedding_dim: int
    model: str
    threshold: float
    created_at: str
    template_size: int


@dataclass(frozen=True)
class UserSummary:
    user_id: str
    display_name: str
    embedding_dim: int
    model: str
    threshold: float
    created_at: str
    template_size: int


@dataclass(frozen=True)
class AuditEvent:
    id: int
    created_at: str
    event_type: str
    user_id: str | None
    detail: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    template      BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    model         TEXT NOT NULL,
    threshold     REAL NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    user_id     TEXT,
    detail      TEXT NOT NULL DEFAULT ''
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FaceDatabase:
    """Encrypted template store backed by SQLite."""

    def __init__(self, db_path: Path | str, cipher: TemplateCipher) -> None:
        self.db_path = Path(db_path)
        self.cipher = cipher
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        set_private_permissions(self.db_path.parent, 0o700)
        self._initialise()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialise(self) -> None:
        try:
            with self._connect() as connection:
                connection.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot initialise database: {exc}") from exc
        set_private_permissions(self.db_path, 0o600)

    def user_exists(self, user_id: str) -> bool:
        cleaned = validate_user_id(user_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM users WHERE user_id = ?", (cleaned,)
            ).fetchone()
        return row is not None

    def add_user(
        self,
        user_id: str,
        display_name: str,
        embedding: np.ndarray,
        model: str,
        threshold: float,
    ) -> UserSummary:
        """Encrypt and store an embedding.  Duplicate IDs are rejected."""
        cleaned_id = validate_user_id(user_id)
        cleaned_name = validate_display_name(display_name)
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if vector.size == 0:
            raise DatabaseError("Embedding is empty.")
        if not np.isfinite(vector).all():
            raise DatabaseError("Embedding contains non-finite values.")

        plaintext = vector.tobytes(order="C")
        token = self.cipher.encrypt(plaintext)
        created_at = _utc_now()

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO users
                        (user_id, display_name, template, embedding_dim, model,
                         threshold, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cleaned_id,
                        cleaned_name,
                        sqlite3.Binary(token),
                        int(vector.shape[1]),
                        model,
                        float(threshold),
                        created_at,
                    ),
                )
                self._log_event_in_connection(
                    connection, "enrollment", cleaned_id, "Encrypted template created"
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateUserError(
                f"User ID '{cleaned_id}' is already enrolled."
            ) from exc
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot save user: {exc}") from exc

        return UserSummary(
            user_id=cleaned_id,
            display_name=cleaned_name,
            embedding_dim=int(vector.shape[1]),
            model=model,
            threshold=float(threshold),
            created_at=created_at,
            template_size=len(token),
        )

    def get_user(self, user_id: str) -> UserRecord:
        cleaned = validate_user_id(user_id)
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM users WHERE user_id = ?", (cleaned,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot read user: {exc}") from exc

        if row is None:
            raise UserNotFoundError(f"User '{cleaned}' is not enrolled.")

        token = bytes(row["template"])
        try:
            plaintext = self.cipher.decrypt(token)
        except SecurityError:
            raise
        vector = np.frombuffer(plaintext, dtype=np.float32).copy()
        expected_dim = int(row["embedding_dim"])
        if vector.size != expected_dim:
            raise DatabaseError(
                "Stored template has an unexpected size; it may be corrupted."
            )
        return UserRecord(
            user_id=row["user_id"],
            display_name=row["display_name"],
            embedding=vector.reshape(1, -1),
            embedding_dim=expected_dim,
            model=row["model"],
            threshold=float(row["threshold"]),
            created_at=row["created_at"],
            template_size=len(token),
        )

    def list_users(self) -> list[UserSummary]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT user_id, display_name, embedding_dim, model,
                           threshold, created_at, length(template) AS template_size
                    FROM users
                    ORDER BY created_at DESC, user_id ASC
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot list users: {exc}") from exc

        return [
            UserSummary(
                user_id=row["user_id"],
                display_name=row["display_name"],
                embedding_dim=row["embedding_dim"],
                model=row["model"],
                threshold=float(row["threshold"]),
                created_at=row["created_at"],
                template_size=int(row["template_size"]),
            )
            for row in rows
        ]

    def count_users(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        return int(row["n"])

    def delete_user(self, user_id: str) -> bool:
        """Delete a user and their encrypted template."""
        cleaned = validate_user_id(user_id)
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "DELETE FROM users WHERE user_id = ?", (cleaned,)
                )
                deleted = cursor.rowcount > 0
                if deleted:
                    self._log_event_in_connection(
                        connection, "deletion", cleaned, "Template deleted"
                    )
                else:
                    self._log_event_in_connection(
                        connection, "deletion_failed", cleaned, "User not found"
                    )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot delete user: {exc}") from exc
        return deleted

    def log_event(
        self, event_type: str, user_id: str | None = None, detail: str = ""
    ) -> None:
        """Record a non-biometric audit event."""
        safe_detail = detail[:500]
        try:
            with self._connect() as connection:
                self._log_event_in_connection(
                    connection, event_type, user_id, safe_detail
                )
        except sqlite3.Error:
            # Audit logging must never crash a biometric workflow.
            pass

    @staticmethod
    def _log_event_in_connection(
        connection: sqlite3.Connection,
        event_type: str,
        user_id: str | None,
        detail: str,
    ) -> None:
        connection.execute(
            "INSERT INTO audit_log (created_at, event_type, user_id, detail) "
            "VALUES (?, ?, ?, ?)",
            (_utc_now(), event_type, user_id, detail[:500]),
        )

    def recent_events(self, limit: int = 20) -> list[AuditEvent]:
        safe_limit = max(1, min(int(limit), 200))
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
                    (safe_limit,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot read audit log: {exc}") from exc
        return [
            AuditEvent(
                id=int(row["id"]),
                created_at=row["created_at"],
                event_type=row["event_type"],
                user_id=row["user_id"],
                detail=row["detail"],
            )
            for row in rows
        ]
