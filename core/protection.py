"""Configuration shared by the Video Injection Lab protection pipeline."""

from __future__ import annotations

from dataclasses import dataclass

FINAL_NOT_RUN = "NOT RUN"
FINAL_ACCEPTED = "ACCEPTED"
FINAL_REJECTED = "REJECTED"

LAYER_SOURCE_INTEGRITY = "Source Integrity"
LAYER_ACTIVE_LIVENESS = "Active Liveness"
LAYER_PASSIVE_PAD = "Passive PAD"
LAYER_IDENTITY = "Face Recognition"
LAYER_NONE = "None"


@dataclass(frozen=True)
class ProtectionConfig:
    """Which additional layers are active for one experiment run."""

    source_integrity_enabled: bool = False
    active_liveness_enabled: bool = False
    label: str = "Original / Vulnerable"
    simulation: bool = False

    @property
    def any_enabled(self) -> bool:
        return self.source_integrity_enabled or self.active_liveness_enabled

    def summary(self) -> str:
        return (
            f"source_integrity={self.source_integrity_enabled}, "
            f"active_liveness={self.active_liveness_enabled}, "
            f"label={self.label}, simulation={self.simulation}"
        )
