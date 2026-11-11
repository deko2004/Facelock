from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path

import numpy as np
import pytest

from core.database import (
    DatabaseError,
    DuplicateUserError,
    FaceDatabase,
    UserNotFoundError,
)
from core.security import SecurityError, ValidationError


def _embedding(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=(1, 128)).astype(np.float32)
    return vector / np.linalg.norm(vector)


def test_add_get_list_and_count(database: FaceDatabase) -> None:
    vector = _embedding()
    summary = database.add_user("alice", "Alice Example", vector, "SFace", 0.363)
    assert summary.user_id == "alice"
    assert summary.display_name == "Alice Example"
    assert summary.embedding_dim == 128
    assert summary.template_size > 128 * 4
    assert database.count_users() == 1
    assert database.user_exists("alice")

    record = database.get_user("alice")
    assert record.embedding.shape == (1, 128)
    assert np.allclose(record.embedding, vector)
    assert record.model == "SFace"
    assert record.threshold == pytest.approx(0.363)

    users = database.list_users()
    assert len(users) == 1
    assert users[0].user_id == "alice"


def test_duplicate_user_rejected(database: FaceDatabase) -> None:
    vector = _embedding()
    database.add_user("alice", "Alice", vector, "SFace", 0.363)
    with pytest.raises(DuplicateUserError):
        database.add_user("alice", "Alice Again", vector, "SFace", 0.363)


def test_get_missing_user(database: FaceDatabase) -> None:
    with pytest.raises(UserNotFoundError):
        database.get_user("missing")


def test_delete_user(database: FaceDatabase) -> None:
    database.add_user("alice", "Alice", _embedding(), "SFace", 0.363)
    assert database.delete_user("alice")
    assert not database.user_exists("alice")
    assert database.delete_user("alice") is False


def test_invalid_identifier_rejected(database: FaceDatabase) -> None:
    with pytest.raises(ValidationError):
        database.add_user("../alice", "Alice", _embedding(), "SFace", 0.363)


def test_non_finite_embedding_rejected(database: FaceDatabase) -> None:
    vector = np.full((1, 128), np.nan, dtype=np.float32)
    with pytest.raises(DatabaseError):
        database.add_user("alice", "Alice", vector, "SFace", 0.363)


def test_template_is_encrypted_at_rest(
    database: FaceDatabase, cipher
) -> None:
    vector = _embedding()
    database.add_user("alice", "Alice", vector, "SFace", 0.363)
    raw = database.db_path.read_bytes()
    assert vector.tobytes() not in raw
    assert cipher.key_path.read_bytes() not in raw


def test_database_permissions(database: FaceDatabase) -> None:
    mode = stat.S_IMODE(os.stat(database.db_path).st_mode)
    assert mode == 0o600


def test_corrupted_template_raises(database: FaceDatabase) -> None:
    database.add_user("alice", "Alice", _embedding(), "SFace", 0.363)
    with sqlite3.connect(database.db_path) as connection:
        connection.execute("UPDATE users SET template = ? WHERE user_id = 'alice'", (b"bad",))
    with pytest.raises(SecurityError):
        database.get_user("alice")


def test_audit_log(database: FaceDatabase) -> None:
    database.add_user("alice", "Alice", _embedding(), "SFace", 0.363)
    database.log_event("test_event", "alice", "unit test")
    events = database.recent_events(limit=10)
    assert events
    assert events[0].event_type == "test_event"
    assert any(event.event_type == "enrollment" for event in events)
