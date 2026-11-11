"""Security helpers: input validation, private files and template encryption.

Biometric templates are encrypted with Fernet (AES-128-CBC + HMAC-SHA256).
The key is stored in a separate file from the SQLite database.  This protects
against casual inspection of a copied database, but it does **not** protect
against an attacker who can read the whole project directory or who controls
the running process.  See ``docs/security_privacy.md`` for the full threat
model.
"""

from __future__ import annotations

import os
import re
import secrets
from pathlib import Path

try:  # cryptography is a declared dependency, but the error is clearer if absent.
    from cryptography.fernet import Fernet, InvalidToken
except ImportError as exc:  # pragma: no cover - exercised only on broken installs
    raise RuntimeError(
        "The 'cryptography' package is required. Install it with: "
        "venv/bin/pip install cryptography"
    ) from exc


class SecurityError(RuntimeError):
    """Raised when a cryptographic operation or file permission fails."""


class ValidationError(ValueError):
    """Raised when user-supplied input does not pass validation."""


_USER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


def validate_user_id(user_id: str) -> str:
    """Validate and normalise a user identifier.

    Identifiers are intentionally restricted to a safe ASCII subset so they can
    never be interpreted as paths, SQL fragments or shell input.
    """
    cleaned = (user_id or "").strip()
    if not cleaned:
        raise ValidationError("User ID is required.")
    if not _USER_ID_RE.fullmatch(cleaned):
        raise ValidationError(
            "User ID must be 1-64 characters using letters, numbers, dot, dash "
            "or underscore, and must start with a letter or number."
        )
    return cleaned


def validate_display_name(name: str) -> str:
    """Validate a human-readable display name."""
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValidationError("Display name is required.")
    if len(cleaned) > 80:
        raise ValidationError("Display name must be at most 80 characters.")
    if _CONTROL_CHARS_RE.search(cleaned):
        raise ValidationError("Display name contains control characters.")
    return cleaned


def set_private_permissions(path: Path, mode: int = 0o600) -> None:
    """Best-effort restriction of a file or directory to the current user."""
    try:
        os.chmod(path, mode)
    except OSError:
        # On some filesystems (e.g. FAT) chmod is unsupported.  Do not crash.
        pass


def safe_unlink(path: Path | str) -> bool:
    """Delete a temporary file if it exists, ignoring races and permissions."""
    try:
        Path(path).unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


class TemplateCipher:
    """Encrypt/decrypt biometric templates using a locally stored Fernet key."""

    def __init__(self, key_path: Path | str) -> None:
        self.key_path = Path(key_path)
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        set_private_permissions(self.key_path.parent, 0o700)
        key = self._load_or_create_key()
        try:
            self._fernet = Fernet(key)
        except (ValueError, TypeError) as exc:
            raise SecurityError(
                f"Template key at {self.key_path} is not a valid Fernet key."
            ) from exc

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            try:
                key = self.key_path.read_bytes().strip()
            except OSError as exc:
                raise SecurityError(f"Cannot read template key: {exc}") from exc
            if not key:
                raise SecurityError("Template key file is empty.")
            return key

        key = Fernet.generate_key()
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            fd = os.open(self.key_path, flags, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(key)
        except FileExistsError:
            # Another process won the race; use its key.
            return self.key_path.read_bytes().strip()
        except OSError as exc:
            raise SecurityError(f"Cannot create template key: {exc}") from exc
        set_private_permissions(self.key_path, 0o600)
        return key

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt arbitrary bytes and return a Fernet token."""
        try:
            return self._fernet.encrypt(plaintext)
        except Exception as exc:  # pragma: no cover - defensive
            raise SecurityError("Failed to encrypt biometric template.") from exc

    def decrypt(self, token: bytes) -> bytes:
        """Decrypt a Fernet token, raising SecurityError on failure."""
        try:
            return self._fernet.decrypt(token)
        except InvalidToken as exc:
            raise SecurityError(
                "Stored template cannot be decrypted. The key may have changed "
                "or the database may be corrupted."
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise SecurityError("Failed to decrypt biometric template.") from exc


def generate_audit_id() -> str:
    """Return a short random identifier for non-sensitive audit records."""
    return secrets.token_hex(8)
