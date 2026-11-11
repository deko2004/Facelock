from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from core.security import (
    SecurityError,
    TemplateCipher,
    ValidationError,
    safe_unlink,
    validate_display_name,
    validate_user_id,
)


def test_validate_user_id_valid() -> None:
    assert validate_user_id("  alice_01  ") == "alice_01"
    assert validate_user_id("A-1.b") == "A-1.b"


@pytest.mark.parametrize(
    "value",
    ["", " ", "../etc/passwd", "a b", "user/name", "-bad", "x" * 65],
)
def test_validate_user_id_invalid(value: str) -> None:
    with pytest.raises(ValidationError):
        validate_user_id(value)


def test_validate_display_name() -> None:
    assert validate_display_name(" Alice Example ") == "Alice Example"
    with pytest.raises(ValidationError):
        validate_display_name("")
    with pytest.raises(ValidationError):
        validate_display_name("x" * 81)


def test_cipher_roundtrip(tmp_path: Path) -> None:
    cipher = TemplateCipher(tmp_path / "key.bin")
    token = cipher.encrypt(b"secret biometric bytes")
    assert token != b"secret biometric bytes"
    assert cipher.decrypt(token) == b"secret biometric bytes"


def test_cipher_wrong_key_fails(tmp_path: Path) -> None:
    cipher_a = TemplateCipher(tmp_path / "a.key")
    cipher_b = TemplateCipher(tmp_path / "b.key")
    token = cipher_a.encrypt(b"template")
    with pytest.raises(SecurityError):
        cipher_b.decrypt(token)


def test_key_permissions(tmp_path: Path) -> None:
    key_path = tmp_path / "key.bin"
    TemplateCipher(key_path)
    mode = stat.S_IMODE(os.stat(key_path).st_mode)
    assert mode == 0o600


def test_safe_unlink(tmp_path: Path) -> None:
    path = tmp_path / "temp.jpg"
    path.write_bytes(b"x")
    assert safe_unlink(path)
    assert not path.exists()
    assert safe_unlink(path) is False
