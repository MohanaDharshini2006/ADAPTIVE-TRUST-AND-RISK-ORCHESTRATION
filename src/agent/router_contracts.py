"""Explicit contract boundary between the shared router and the Technical Support classifier.

This module does not add a router, retriever, response generator, or ATRO decision engine.
It only formalizes the allowed labels and validation rules so the two boundaries remain
separate and the frozen six-category Technical Support baseline is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ROUTER_AGENTS: tuple[str, ...] = (
    "billing",
    "order",
    "return",
    "technical_support",
)

TECHNICAL_SUPPORT_BASELINE_LABELS: tuple[str, ...] = (
    "color_fade",
    "fabric_material",
    "fastener_hardware",
    "general_quality_complaint",
    "sizing_fit",
    "stitching_durability",
)


def _ensure_allowed(value: str, allowed_values: tuple[str, ...], field_name: str) -> None:
    if value not in allowed_values:
        allowed = ", ".join(allowed_values)
        raise ValueError(f"{field_name} must be one of: {allowed}. Received: {value!r}")


@dataclass(frozen=True)
class RouterInput:
    """Input contract for the shared four-agent router."""

    conversation_id: str
    user_message: str
    session_context: dict[str, Any] | None = None
    prior_turns: list[dict[str, Any]] | None = None
    customer_profile: dict[str, Any] | None = None
    request_time_utc: str | None = None

    def __post_init__(self) -> None:
        if not self.conversation_id or not str(self.conversation_id).strip():
            raise ValueError("conversation_id is required.")
        if not self.user_message or not str(self.user_message).strip():
            raise ValueError("user_message is required.")


@dataclass(frozen=True)
class RouterOutput:
    """Output contract for the shared four-agent router.

    This router is not allowed to emit Technical Support subcategory labels.
    """

    conversation_id: str
    selected_agent: str
    confidence: float | None = None
    reasons: list[str] = field(default_factory=list)
    secondary_agents: list[str] = field(default_factory=list)
    requires_human_review: bool = False
    route_notes: str | None = None

    def __post_init__(self) -> None:
        _ensure_allowed(self.selected_agent, ROUTER_AGENTS, "selected_agent")

        for agent in self.secondary_agents:
            _ensure_allowed(agent, ROUTER_AGENTS, "secondary_agents[]")

        if self.confidence is not None:
            if not 0.0 <= float(self.confidence) <= 1.0:
                raise ValueError("confidence must be between 0.0 and 1.0.")

        if self.selected_agent in self.secondary_agents:
            raise ValueError("selected_agent and secondary_agents must not contain the same agent.")


@dataclass(frozen=True)
class TechnicalSupportClassifierInput:
    """Input contract for the Technical Support baseline classifier.

    This classifier is only for the Technical Support domain. It never replaces the router.
    """

    user_message: str
    route_context: str = "technical_support"
    taxonomy_version: str = "baseline_v1"
    include_future_technical_labels: bool = False

    def __post_init__(self) -> None:
        if not self.user_message or not str(self.user_message).strip():
            raise ValueError("user_message is required.")
        if self.route_context != "technical_support":
            raise ValueError(
                "route_context for the Technical Support classifier must be 'technical_support'. "
                f"Received: {self.route_context!r}"
            )


@dataclass(frozen=True)
class TechnicalSupportClassifierOutput:
    """Output contract for the Technical Support six-category baseline classifier.

    The model may optionally surface future technical labels separately in
    future_label_candidates, but those labels are never merged into predicted_category.
    """

    route_context: str
    predicted_category: str
    category_confidence: float | None = None
    taxonomy_version: str = "baseline_v1"
    future_label_candidates: list[str] | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.route_context != "technical_support":
            raise ValueError(
                "route_context for the Technical Support classifier must be 'technical_support'. "
                f"Received: {self.route_context!r}"
            )

        _ensure_allowed(
            self.predicted_category,
            TECHNICAL_SUPPORT_BASELINE_LABELS,
            "predicted_category",
        )

        if self.category_confidence is not None:
            if not 0.0 <= float(self.category_confidence) <= 1.0:
                raise ValueError("category_confidence must be between 0.0 and 1.0.")

        if self.future_label_candidates is not None:
            for future_label in self.future_label_candidates:
                if future_label in ROUTER_AGENTS:
                    raise ValueError(
                        "future_label_candidates must not contain router labels. "
                        f"Received: {future_label!r}"
                    )
                if future_label in TECHNICAL_SUPPORT_BASELINE_LABELS:
                    raise ValueError(
                        "future_label_candidates must not duplicate the frozen baseline labels. "
                        f"Received: {future_label!r}"
                    )

        if self.predicted_category in ROUTER_AGENTS:
            raise ValueError(
                "predicted_category must remain on the Technical Support baseline taxonomy. "
                "It cannot be a shared-router label."
            )
