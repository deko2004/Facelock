from __future__ import annotations

from core.frame_source import (
    SOURCE_PHYSICAL,
    SOURCE_RECORDED,
    SOURCE_UNKNOWN,
    SOURCE_VIRTUAL,
    FrameSourceInfo,
)
from core.source_integrity import (
    SOURCE_NOT_RUN,
    SOURCE_REJECTED,
    SOURCE_TRUSTED,
    SOURCE_UNCERTAIN,
    CameraSourceIntegrityService,
    SourceIntegrityPolicy,
    classify_source,
)


def _physical_info(identifier: str = "v4l2:/dev/video0:usb1") -> FrameSourceInfo:
    return FrameSourceInfo(
        kind=SOURCE_PHYSICAL,
        identifier=identifier,
        display_name="Integrated Camera",
        device_path="/dev/video0",
        device_name="Integrated Camera",
        sysfs_path="/sys/class/video4linux/video0",
        sysfs_device_path="/sys/devices/pci0000:00/0000:00:14.0/usb1/1-1",
    )


def _recorded_info() -> FrameSourceInfo:
    return FrameSourceInfo(
        kind=SOURCE_RECORDED,
        identifier="file:/tmp/attack.avi",
        display_name="attack.avi",
        file_path="/tmp/attack.avi",
    )


def _virtual_info() -> FrameSourceInfo:
    return FrameSourceInfo(
        kind=SOURCE_VIRTUAL,
        identifier="v4l2:/dev/video10:v4l2loopback",
        display_name="OBS Virtual Camera",
        device_path="/dev/video10",
        device_name="OBS Virtual Camera",
        sysfs_path="/sys/class/video4linux/video10",
        sysfs_device_path="/sys/devices/virtual/video4linux/video10",
    )


def test_classify_source_distinguishes_recorded_physical_and_virtual() -> None:
    assert classify_source(_recorded_info()).status == SOURCE_REJECTED
    assert classify_source(_physical_info()).status == SOURCE_TRUSTED
    assert classify_source(_virtual_info()).status == SOURCE_REJECTED
    unknown = FrameSourceInfo(
        kind=SOURCE_UNKNOWN,
        identifier="unknown",
        display_name="Unknown",
    )
    assert classify_source(unknown).status == SOURCE_UNCERTAIN


def test_recorded_source_is_rejected_by_default_even_with_session() -> None:
    service = CameraSourceIntegrityService()
    info = _recorded_info()
    session = service.start_session(info, now=1.0)
    assessment = service.evaluate(info, session, enabled=True, now=2.0)
    assert assessment.status == SOURCE_REJECTED
    assert not assessment.trusted
    assert "Recorded" in assessment.reason


def test_physical_source_requires_session_binding() -> None:
    service = CameraSourceIntegrityService()
    info = _physical_info()
    unbound = service.evaluate(info, enabled=True)
    assert unbound.status == SOURCE_REJECTED
    assert "session" in unbound.reason.lower()

    session = service.start_session(info, now=1.0)
    trusted = service.evaluate(info, session, enabled=True, now=2.0)
    assert trusted.status == SOURCE_TRUSTED
    assert trusted.trusted
    assert trusted.session_id == session.session_id


def test_session_binding_detects_source_change() -> None:
    service = CameraSourceIntegrityService()
    first = _physical_info("v4l2:/dev/video0:usb1")
    second = _physical_info("v4l2:/dev/video1:usb2")
    session = service.start_session(first, now=1.0)
    assessment = service.evaluate(second, session, enabled=True, now=2.0)
    assert assessment.status == SOURCE_REJECTED
    assert assessment.source_changed
    assert session.invalidated


def test_disabled_source_integrity_is_not_run_and_not_trusted() -> None:
    service = CameraSourceIntegrityService()
    info = _physical_info()
    session = service.start_session(info, now=1.0)
    assessment = service.evaluate(info, session, enabled=False, now=2.0)
    assert assessment.status == SOURCE_NOT_RUN
    assert not assessment.trusted
    assert "off" in assessment.reason.lower()


def test_audit_callback_receives_session_and_rejection_events() -> None:
    events: list[tuple[str, str | None, str]] = []
    service = CameraSourceIntegrityService(
        audit=lambda event, user, detail: events.append((event, user, detail))
    )
    info = _recorded_info()
    session = service.start_session(info, now=1.0)
    service.evaluate(info, session, enabled=True, now=2.0)
    service.end_session(session, "test")
    event_types = [event[0] for event in events]
    assert "source_session_started" in event_types
    assert "source_rejected" in event_types
    assert "source_session_ended" in event_types


def test_policy_can_disable_session_binding_for_labelling_only() -> None:
    service = CameraSourceIntegrityService(
        policy=SourceIntegrityPolicy(require_session_binding=False)
    )
    assessment = service.evaluate(_physical_info(), enabled=True, now=1.0)
    assert assessment.status == SOURCE_TRUSTED
