"""Grok subscription routing through StrixProvider.get_model."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from typing import TYPE_CHECKING, Any
from unittest import mock

import httpx
import pytest
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

from strix.config import grok, subscription
from strix.config.models import StrixProvider, _TurnGuardModel, routes_through_litellm
from strix.interface import scan_setup, utils
from strix.llm.request_log import RequestLoggingModel
from strix.report import state as state_mod


if TYPE_CHECKING:
    from pathlib import Path


def test_grok_prefix_routes_to_chat_completions(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client = mock.MagicMock()
    monkeypatch.setattr(grok, "get_subscription_client", lambda: client)

    model = StrixProvider().get_model("grok/grok-4")

    assert isinstance(model, _TurnGuardModel)
    assert isinstance(model._inner, RequestLoggingModel)
    assert isinstance(model._inner._inner, OpenAIChatCompletionsModel)
    # The provider strips the grok/ prefix and passes xAI's bare model slug.
    assert model._inner._inner.model == "grok-4"


def test_non_subscription_model_is_not_hijacked_by_grok(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def _boom() -> object:
        msg = "grok client must not be built for a non-grok model"
        raise AssertionError(msg)

    monkeypatch.setattr(grok, "get_subscription_client", _boom)

    # A metered xai/* key model must fall through to the normal provider path,
    # not the subscription route.
    model = StrixProvider().get_model("xai/grok-4")
    assert isinstance(model, _TurnGuardModel)
    assert not isinstance(model._inner, OpenAIChatCompletionsModel)


def test_provider_label_names_the_subscription() -> None:
    assert subscription.provider_label("grok/grok-4") == "Grok"
    assert subscription.provider_label("chatgpt/gpt-5.4") == "ChatGPT"
    # Metered API-key models are not subscriptions.
    assert subscription.provider_label("xai/grok-4") is None
    assert subscription.provider_label("openai/gpt-5.4") is None


def test_litellm_model_name_maps_subscription_prefixes() -> None:
    # Model metadata (context window, output cap) is keyed "xai/…" for Grok and
    # bare for ChatGPT; the routing prefixes themselves are never LiteLLM keys.
    assert subscription.litellm_model_name("grok/grok-4") == "xai/grok-4"
    assert subscription.litellm_model_name("chatgpt/gpt-5.4") == "gpt-5.4"
    # Non-subscription models pass through untouched.
    assert subscription.litellm_model_name("xai/grok-4") == "xai/grok-4"
    assert subscription.litellm_model_name("openai/gpt-5.4") == "openai/gpt-5.4"
    assert subscription.litellm_model_name(None) is None


def test_run_record_reports_grok_provider(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    settings = mock.MagicMock()
    settings.llm.model = "grok/grok-4"
    monkeypatch.setattr(state_mod, "load_settings", lambda: settings)

    record = state_mod.ReportState(run_name="run-test").run_record
    assert record["auth_mode"] == "subscription"
    assert record["subscription_provider"] == "Grok"


def test_subscription_label_prefers_persisted_provider(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    settings = mock.MagicMock()
    settings.llm.model = "chatgpt/gpt-5.4"  # current settings point at ChatGPT
    monkeypatch.setattr(utils, "load_settings", lambda: settings)

    # A resumed Grok run keeps its persisted provider even though settings changed.
    resumed = mock.MagicMock(
        run_record={"auth_mode": "subscription", "subscription_provider": "Grok"}
    )
    assert utils.subscription_label(resumed) == "Grok subscription"

    # With no persisted provider, it derives the label from settings (not a
    # hardcoded default).
    fresh = mock.MagicMock(run_record={})
    assert utils.subscription_label(fresh) == "ChatGPT subscription"


def test_persisted_run_record_carries_provider(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    settings = mock.MagicMock()
    settings.llm.model = "grok/grok-4"
    monkeypatch.setattr(scan_setup, "load_settings", lambda: settings)
    monkeypatch.setattr(scan_setup, "run_dir_for", lambda _name: tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "strix.report.writer.write_run_record", lambda _dir, rec: captured.update(rec)
    )

    args = argparse.Namespace(
        run_name="run-test",
        targets_info=[],
        scan_mode="scan",
        instruction=None,
        non_interactive=True,
        local_sources=[],
        diff_scope={"active": False},
        scope_mode="mode",
        diff_base=None,
    )
    scan_setup._persist_run_record(args)

    # The resume/viewer record must carry the provider so resumed runs stay labeled.
    assert captured["auth_mode"] == "subscription"
    assert captured["subscription_provider"] == "Grok"


def test_grok_route_is_sdk_native() -> None:
    assert routes_through_litellm("grok/grok-build") is False
    assert routes_through_litellm("xai/grok-4") is True


def test_subscription_startup_errors_are_actionable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    main = importlib.import_module("strix.interface.main")

    settings = mock.MagicMock()
    settings.llm.model = "grok/grok-build"
    monkeypatch.setattr(main, "load_settings", lambda: settings)
    assert "strix auth login grok" in (main._subscription_error_hint(Exception("HTTP 401")) or "")
    assert "quota" in (main._subscription_error_hint(Exception("Error code: 429")) or "").lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_grok_streams_even_for_preflight_and_compaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
) -> None:
    monkeypatch.setattr(grok, "AUTH_PATH", tmp_path / "subscription-auth.json")
    grok.save_record(
        {
            "type": "oauth",
            "access": "access",
            "refresh": "refresh",
            "expires_at": time.time() + 3600,
        }
    )
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload.get("stream_options") == {"include_usage": True}
        assert request.headers["x-grok-model-override"] == "grok-build"
        if payload.get("stream") is not True:
            return httpx.Response(400, json={"error": {"message": "stream required"}})
        chunks = [
            {
                "id": "chat1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "grok-build",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "OK"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chat1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "grok-build",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
            },
        ]
        body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
        return httpx.Response(
            200, text=body + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"}
        )

    client = grok.build_openai_client()
    client._client._transport = httpx.MockTransport(respond)
    monkeypatch.setattr(grok, "get_subscription_client", lambda: client)
    model = StrixProvider().get_model("grok/grok-build")
    kwargs = {
        "system_instructions": "Reply OK",
        "input": "hello",
        "model_settings": ModelSettings(),
        "tools": [],
        "output_schema": None,
        "handoffs": [],
        "tracing": ModelTracing.DISABLED,
        "previous_response_id": None,
        "conversation_id": None,
        "prompt": None,
    }
    try:
        if streaming:
            events = [event async for event in model.stream_response(**kwargs)]
            response = next(
                event.response for event in events if event.type == "response.completed"
            )
        else:
            response = await model.get_response(**kwargs)
        assert response.output[0].content[0].text == "OK"
        assert response.usage.total_tokens == 14
        assert len(requests) == 1
        assert requests[0]["stream"] is True
    finally:
        await client.close()


def test_legacy_resume_derives_provider_from_recorded_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = mock.MagicMock()
    settings.llm.model = "grok/grok-build"
    monkeypatch.setattr(state_mod, "load_settings", lambda: settings)
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "auth_mode": "subscription",
                "llm_usage": {"agents": [{"agent_id": "root", "model": "chatgpt/gpt-5.4"}]},
            }
        )
    )
    state = state_mod.ReportState(run_name="old-run")
    state._run_dir = tmp_path
    state.hydrate_from_run_dir()
    assert state.run_record["subscription_provider"] == "ChatGPT"
    assert utils.subscription_label(state) == "ChatGPT subscription"
