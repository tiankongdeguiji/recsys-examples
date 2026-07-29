# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 operator-fusion policy."""

from __future__ import annotations

import os
from dataclasses import dataclass

FUSION_ENV = "GR_INFERENCE_QWEN3_DISABLE_FUSIONS"
FUSION_FAMILIES = frozenset({"add_rmsnorm", "mlp", "qk_norm", "rope"})


@dataclass(frozen=True)
class Qwen3FusionPolicy:
    """Controls which Qwen3 operator-fusion families may run."""

    disabled: frozenset[str] = frozenset()
    requested: str = ""

    @classmethod
    def from_env(cls) -> "Qwen3FusionPolicy":
        """Build and validate a policy from the process environment."""

        requested = os.environ.get(FUSION_ENV, "").strip()
        if not requested:
            return cls()
        values = frozenset(
            value.strip().lower() for value in requested.split(",") if value.strip()
        )
        if "all" in values:
            if values != {"all"}:
                raise ValueError(f"{FUSION_ENV}=all cannot be combined with other values")
            return cls(disabled=FUSION_FAMILIES, requested=requested)
        unknown = values - FUSION_FAMILIES
        if unknown:
            supported = ", ".join(sorted(FUSION_FAMILIES | {"all"}))
            invalid = ", ".join(sorted(unknown))
            raise ValueError(
                f"unsupported {FUSION_ENV} value(s): {invalid}; supported: {supported}"
            )
        return cls(disabled=values, requested=requested)

    def enabled(self, family: str) -> bool:
        """Return whether one known fusion family is enabled."""

        if family not in FUSION_FAMILIES:
            raise ValueError(f"unknown Qwen3 fusion family: {family}")
        return family not in self.disabled

    def summary(self) -> dict[str, object]:
        """Return a stable JSON-serializable policy description."""

        return {
            "env": FUSION_ENV,
            "requested": self.requested,
            "disabled": sorted(self.disabled),
            "enabled": sorted(FUSION_FAMILIES - self.disabled),
        }


_DEFAULT_POLICY: Qwen3FusionPolicy | None = None


def default_qwen3_fusion_policy() -> Qwen3FusionPolicy:
    """Return the process-global validated Qwen3 fusion policy."""

    global _DEFAULT_POLICY
    if _DEFAULT_POLICY is None:
        _DEFAULT_POLICY = Qwen3FusionPolicy.from_env()
    return _DEFAULT_POLICY


def qwen3_fusion_enabled(family: str) -> bool:
    """Return whether one Qwen3 fusion family may be dispatched."""

    return default_qwen3_fusion_policy().enabled(family)


def reset_default_qwen3_fusion_policy() -> None:
    """Clear the cached policy for tests and explicit environment reconfiguration."""

    global _DEFAULT_POLICY
    _DEFAULT_POLICY = None
