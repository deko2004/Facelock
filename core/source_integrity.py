"""Camera source classification, session binding and fail-closed policy.

This is an application-level integrity check.  It can distinguish a local video
file from a camera-like source and it can notice that the selected source changed
during a session.  It cannot prove that frames are authentic: an attacker with
control of the operating system, camera driver, virtual-camera stack or process
memory can bypass every check in this module.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from core.frame_source import (
    SOURCE_LABELS,
    SOURCE_PHYSICAL,
    SOURCE_RECORDED,
    SOURCE_UNKNOWN,
    SOURCE_VIRTUAL,
    FrameSourceInfo,
)

SOURCE_NOT_RUN = "NOT_RUN"
SOURCE_TRUSTED = "TRUSTED"
SOURCE_REJECTED = "REJECTED"
SOURCE_UNCERTAIN = "UNCERTAIN"

SOURCE_STATUS_LABELS = {
    SOURCE_NOT_RUN: "Not Run",
    SOURCE_TRUSTED: "Trusted",
    SOURCE_REJECTED: "Rejected",
    SOURCE_UNCERTAIN: "Uncertain",
}

_VIRTUAL_MARKERS = (
    "virtual",
    "v4l2loopback",
    "v4l2 loopback",
    "loopback",
    "dummy",
    "obs virtual",
    "obs-camera",
    "manycam",
    "webcamoid",
    "akvcam",
    "chromium",
    "screen capture",
)


@dataclass(frozen=True)
class SourceIntegrityPolicy:
    """Policy applied when source integrity protection is enabled."""

    allow_physical: bool = True
    allow_recorded: bool = False
    allow_virtual: bool = False
    allow_unknown: bool = False
    require_session_binding: bool = True

    def summary(self) -> str:
        return (
            f"physical={self.allow_physical}, recorded={self.allow_recorded}, "
            f"virtual={self.allow_virtual}, unknown={self.allow_unknown}, "
            f"session_binding={self.require_session_binding}"
        )


@dataclass
class SourceSession:
    """A source identity bound to one authentication session."""

    session_id: str
    source_identity: str
    kind: str
    started_at: float
    last_seen_at: float
    invalidated: bool = False
    invalidation_reason: str = ""
    last_status: str = ""
    last_reason: str = ""


@dataclass(frozen=True)
class SourceAssessment:
    status: str
    trusted: bool
    kind: str
    kind_label: str
    reason: str
    identity: str
    session_id: str | None = None
    policy_applied: bool = True
    source_changed: bool = False
    details: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def rejected(self) -> bool:
        return self.status in {SOURCE_REJECTED, SOURCE_UNCERTAIN}

    @property
    def status_label(self) -> str:
        return SOURCE_STATUS_LABELS.get(self.status, self.status)


def _looks_virtual(info: FrameSourceInfo) -> bool:
    combined = " ".join(
        str(part)
        for part in (
            info.device_name,
            info.device_path,
            info.sysfs_path,
            info.sysfs_device_path,
            info.display_name,
        )
        if part
    ).lower()
    return any(marker in combined for marker in _VIRTUAL_MARKERS)


def classify_source(info: FrameSourceInfo) -> SourceAssessment:
    """Classify a source without applying session binding.

    This is suitable for preview/UI labelling only.  Authentication must call
    :meth:`CameraSourceIntegrityService.evaluate`, which also enforces policy and
    session binding.
    """
    label = info.kind_label or SOURCE_LABELS.get(SOURCE_UNKNOWN, "Unknown")
    if info.kind == SOURCE_RECORDED:
        return SourceAssessment(
            SOURCE_REJECTED,
            False,
            info.kind,
            label,
            "Recorded video files are not a trusted physical-camera source.",
            info.identifier,
            policy_applied=False,
        )
    if info.kind == SOURCE_VIRTUAL or _looks_virtual(info):
        return SourceAssessment(
            SOURCE_REJECTED,
            False,
            SOURCE_VIRTUAL if info.kind == SOURCE_PHYSICAL else info.kind,
            SOURCE_LABELS[SOURCE_VIRTUAL],
            "The source has virtual-camera indicators.",
            info.identifier,
            policy_applied=False,
        )
    if info.kind == SOURCE_PHYSICAL:
        if not info.device_path or not info.sysfs_device_path:
            return SourceAssessment(
                SOURCE_UNCERTAIN,
                False,
                info.kind,
                label,
                "Physical-camera metadata is incomplete; source trust cannot be established.",
                info.identifier,
                policy_applied=False,
            )
        if not any(
            marker in info.sysfs_device_path.lower()
            for marker in ("/usb", "/pci", "/platform")
        ):
            return SourceAssessment(
                SOURCE_UNCERTAIN,
                False,
                info.kind,
                label,
                "The device topology is not a supported physical bus.",
                info.identifier,
                policy_applied=False,
            )
        return SourceAssessment(
            SOURCE_TRUSTED,
            True,
            info.kind,
            label,
            "Source matches the configured physical-camera policy.",
            info.identifier,
            policy_applied=False,
        )
    return SourceAssessment(
        SOURCE_UNCERTAIN,
        False,
        SOURCE_UNKNOWN,
        SOURCE_LABELS[SOURCE_UNKNOWN],
        "The source type could not be verified.",
        info.identifier,
        policy_applied=False,
    )


class CameraSourceIntegrityService:
    """Classify sources, bind sessions and reject untrusted frame origins."""

    def __init__(
        self,
        policy: SourceIntegrityPolicy | None = None,
        audit: Callable[[str, str | None, str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy or SourceIntegrityPolicy()
        self._audit_callback = audit
        self._clock = clock
        self._lock = threading.RLock()

    def start_session(
        self, info: FrameSourceInfo, now: float | None = None
    ) -> SourceSession:
        timestamp = self._clock() if now is None else float(now)
        session = SourceSession(
            session_id=secrets.token_hex(8),
            source_identity=info.identifier,
            kind=info.kind,
            started_at=timestamp,
            last_seen_at=timestamp,
        )
        self._audit(
            "source_session_started",
            None,
            f"session={session.session_id}; kind={info.kind}; "
            f"source={info.display_name}; identity={info.identifier}",
        )
        return session

    def end_session(
        self, session: SourceSession | None, reason: str = "session ended"
    ) -> None:
        if session is None:
            return
        with self._lock:
            if session.invalidated:
                return
            session.invalidated = True
            session.invalidation_reason = reason
        self._audit(
            "source_session_ended",
            None,
            f"session={session.session_id}; reason={reason}",
        )

    def evaluate(
        self,
        info: FrameSourceInfo,
        session: SourceSession | None = None,
        enabled: bool = True,
        now: float | None = None,
    ) -> SourceAssessment:
        """Apply classification, session binding and fail-closed policy."""
        if not enabled:
            classified = classify_source(info)
            return SourceAssessment(
                SOURCE_NOT_RUN,
                False,
                info.kind,
                info.kind_label,
                "Source integrity protection is off. This source is not trusted.",
                info.identifier,
                session_id=session.session_id if session else None,
                policy_applied=False,
                details=info.details,
            )

        timestamp = self._clock() if now is None else float(now)
        classified = classify_source(info)
        if session is None and self.policy.require_session_binding:
            return self._reject(
                classified,
                "No source session is bound to this authentication attempt.",
                session,
                info,
                source_changed=False,
            )
        if session is not None:
            with self._lock:
                if session.invalidated:
                    return self._reject(
                        classified,
                        session.invalidation_reason
                        or "The source session was invalidated.",
                        session,
                        info,
                        source_changed=False,
                    )
                if session.source_identity != info.identifier:
                    session.invalidated = True
                    session.invalidation_reason = (
                        "The input source changed during the authentication session."
                    )
                    changed = True
                else:
                    session.last_seen_at = timestamp
                    changed = False
            if changed:
                self._audit(
                    "source_session_invalidated",
                    None,
                    f"session={session.session_id}; expected={session.source_identity}; "
                    f"observed={info.identifier}",
                )
                return self._reject(
                    classified,
                    session.invalidation_reason,
                    session,
                    info,
                    source_changed=True,
                )

        if classified.kind == SOURCE_RECORDED and not self.policy.allow_recorded:
            return self._reject(
                classified,
                "Recorded video injection is rejected by the source policy.",
                session,
                info,
            )
        if (
            classified.kind == SOURCE_VIRTUAL or _looks_virtual(info)
        ) and not self.policy.allow_virtual:
            return self._reject(
                classified,
                "Known virtual-camera sources are rejected by the source policy.",
                session,
                info,
            )
        if classified.kind == SOURCE_UNKNOWN and not self.policy.allow_unknown:
            return self._reject(
                classified,
                "Unknown or unsupported sources are rejected by the source policy.",
                session,
                info,
            )
        if classified.kind == SOURCE_PHYSICAL and not self.policy.allow_physical:
            return self._reject(
                classified,
                "Physical cameras are disabled by the source policy.",
                session,
                info,
            )

        if classified.status != SOURCE_TRUSTED:
            if classified.status == SOURCE_UNCERTAIN:
                return self._reject(
                    classified,
                    classified.reason,
                    session,
                    info,
                )
            return self._reject(
                classified,
                classified.reason,
                session,
                info,
            )

        self._audit_transition(
            session,
            SOURCE_TRUSTED,
            "Source integrity verified.",
            "source_integrity_verified",
            f"session={session.session_id if session else 'none'}; "
            f"source={info.display_name}; identity={info.identifier}",
        )
        return SourceAssessment(
            SOURCE_TRUSTED,
            True,
            info.kind,
            info.kind_label,
            "Source integrity verified by the configured policy.",
            info.identifier,
            session_id=session.session_id if session else None,
            policy_applied=True,
            details=info.details,
        )

    def _reject(
        self,
        classified: SourceAssessment,
        reason: str,
        session: SourceSession | None,
        info: FrameSourceInfo,
        source_changed: bool = False,
    ) -> SourceAssessment:
        status = classified.status
        if status == SOURCE_TRUSTED:
            status = SOURCE_REJECTED
        self._audit_transition(
            session,
            status,
            reason,
            "source_rejected",
            f"session={session.session_id if session else 'none'}; "
            f"kind={info.kind}; source={info.display_name}; reason={reason}",
        )
        return SourceAssessment(
            status,
            False,
            classified.kind,
            classified.kind_label,
            reason,
            info.identifier,
            session_id=session.session_id if session else None,
            policy_applied=True,
            source_changed=source_changed,
            details=info.details,
        )

    def _audit_transition(
        self,
        session: SourceSession | None,
        status: str,
        reason: str,
        event: str,
        detail: str,
    ) -> None:
        if session is None:
            return
        with self._lock:
            if session.last_status == status and session.last_reason == reason:
                return
            session.last_status = status
            session.last_reason = reason
        self._audit(event, None, detail)

    def _audit(
        self, event_type: str, user_id: str | None, detail: str
    ) -> None:
        if self._audit_callback is None:
            return
        try:
            self._audit_callback(event_type, user_id, detail)
        except Exception:
            # Audit logging must not break authentication.
            pass
