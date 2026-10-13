import pytest

from src.agent.router_contracts import (
    ROUTER_AGENTS,
    TECHNICAL_SUPPORT_BASELINE_LABELS,
    RouterInput,
    RouterOutput,
    TechnicalSupportClassifierInput,
    TechnicalSupportClassifierOutput,
)


def test_router_accepts_only_four_router_labels():
    output = RouterOutput(
        conversation_id="c-123",
        selected_agent="technical_support",
        confidence=0.92,
        reasons=["technical issue"],
        secondary_agents=["billing"],
    )

    assert output.selected_agent == "technical_support"
    assert set(ROUTER_AGENTS) == {"billing", "order", "return", "technical_support"}
    assert output.secondary_agents == ["billing"]


def test_router_rejects_non_router_labels():
    with pytest.raises(ValueError, match="selected_agent"):
        RouterOutput(
            conversation_id="c-123",
            selected_agent="color_fade",
            confidence=0.8,
        )


def test_classifier_accepts_only_baseline_labels():
    output = TechnicalSupportClassifierOutput(
        route_context="technical_support",
        predicted_category="stitching_durability",
        category_confidence=0.81,
        taxonomy_version="baseline_v1",
    )

    assert output.predicted_category == "stitching_durability"
    assert "stitching_durability" in TECHNICAL_SUPPORT_BASELINE_LABELS


def test_classifier_rejects_router_label_in_predicted_category():
    with pytest.raises(ValueError, match="predicted_category"):
        TechnicalSupportClassifierOutput(
            route_context="technical_support",
            predicted_category="technical_support",
            category_confidence=0.7,
        )


def test_classifier_rejects_future_baseline_label_duplication():
    with pytest.raises(ValueError, match="future_label_candidates"):
        TechnicalSupportClassifierOutput(
            route_context="technical_support",
            predicted_category="color_fade",
            future_label_candidates=["color_fade"],
        )


def test_classifier_rejects_invalid_route_context():
    with pytest.raises(ValueError, match="route_context"):
        TechnicalSupportClassifierInput(user_message="error while loading", route_context="billing")


def test_router_and_classifier_domains_stay_distinct():
    router_output = RouterOutput(
        conversation_id="c-123",
        selected_agent="technical_support",
        confidence=0.9,
    )
    classifier_output = TechnicalSupportClassifierOutput(
        route_context="technical_support",
        predicted_category="sizing_fit",
        category_confidence=0.9,
        future_label_candidates=["login_failure"],
    )

    assert router_output.selected_agent in ROUTER_AGENTS
    assert classifier_output.predicted_category in TECHNICAL_SUPPORT_BASELINE_LABELS
    assert classifier_output.future_label_candidates == ["login_failure"]
    assert router_output.selected_agent not in TECHNICAL_SUPPORT_BASELINE_LABELS


def test_router_rejects_invalid_secondary_agent_labels():
    with pytest.raises(ValueError, match="secondary_agents"):
        RouterOutput(
            conversation_id="c-123",
            selected_agent="billing",
            secondary_agents=["unknown_agent"],
        )


def test_router_rejects_selected_agent_in_secondary_agents():
    with pytest.raises(ValueError, match="selected_agent and secondary_agents"):
        RouterOutput(
            conversation_id="c-123",
            selected_agent="billing",
            secondary_agents=["billing"],
        )


def test_router_input_requires_non_blank_conversation_id():
    with pytest.raises(ValueError, match="conversation_id"):
        RouterInput(conversation_id="   ", user_message="customer issue")


def test_classifier_requires_non_blank_user_message():
    with pytest.raises(ValueError, match="user_message"):
        TechnicalSupportClassifierInput(user_message="   ", route_context="technical_support")


def test_router_confidence_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="confidence"):
        RouterOutput(
            conversation_id="c-123",
            selected_agent="billing",
            confidence=-0.01,
        )

    with pytest.raises(ValueError, match="confidence"):
        RouterOutput(
            conversation_id="c-123",
            selected_agent="billing",
            confidence=1.01,
        )


def test_classifier_confidence_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="category_confidence"):
        TechnicalSupportClassifierOutput(
            route_context="technical_support",
            predicted_category="sizing_fit",
            category_confidence=-0.01,
        )

    with pytest.raises(ValueError, match="category_confidence"):
        TechnicalSupportClassifierOutput(
            route_context="technical_support",
            predicted_category="sizing_fit",
            category_confidence=1.01,
        )


def test_classifier_rejects_empty_or_invalid_route_context():
    with pytest.raises(ValueError, match="route_context"):
        TechnicalSupportClassifierInput(user_message="loading issue", route_context="")

    with pytest.raises(ValueError, match="route_context"):
        TechnicalSupportClassifierOutput(
            route_context="billing",
            predicted_category="color_fade",
        )


def test_future_technical_labels_reject_router_labels_and_duplicate_baseline_labels():
    with pytest.raises(ValueError, match="future_label_candidates"):
        TechnicalSupportClassifierOutput(
            route_context="technical_support",
            predicted_category="color_fade",
            future_label_candidates=["billing"],
        )

    with pytest.raises(ValueError, match="future_label_candidates"):
        TechnicalSupportClassifierOutput(
            route_context="technical_support",
            predicted_category="color_fade",
            future_label_candidates=["color_fade"],
        )

    with pytest.raises(ValueError, match="future_label_candidates"):
        TechnicalSupportClassifierOutput(
            route_context="technical_support",
            predicted_category="color_fade",
            future_label_candidates=["order"],
        )
