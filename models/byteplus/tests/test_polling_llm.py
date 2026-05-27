import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dify_plugin.config.config import DifyPluginEnv
from dify_plugin.core.plugin_registration import PluginRegistration
from dify_plugin.entities.model import ModelType
from dify_plugin.entities.model.llm import LLMPollingStatus, LLMUsage
from dify_plugin.entities.model.message import (
    AudioPromptMessageContent,
    ImagePromptMessageContent,
    TextPromptMessageContent,
    UserPromptMessage,
    VideoPromptMessageContent,
)
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeError

from models.llm import llm as llm_module
from models.llm.llm import BytePlusArkLargeLanguageModel, _ArkHTTPError


def _model() -> BytePlusArkLargeLanguageModel:
    return BytePlusArkLargeLanguageModel(model_schemas=[])


def _credentials() -> dict[str, str]:
    return {
        "ark_api_key": "test-key",
        "api_endpoint_host": "https://ark.ap-southeast.bytepluses.com/api/v3",
    }


def test_get_num_tokens_counts_text_from_multimodal_prompt() -> None:
    model = _model()
    prompt_text = "make a calm ocean video with detailed camera movement"

    token_count = model.get_num_tokens(
        model="seedance-2-0-260128",
        credentials=_credentials(),
        prompt_messages=[
            UserPromptMessage(
                content=[
                    TextPromptMessageContent(data=prompt_text),
                    ImagePromptMessageContent(
                        format="png",
                        mime_type="image/png",
                        url="https://example.com/frame.png",
                    ),
                ],
            )
        ],
    )

    assert token_count == max(1, len(prompt_text) // 4)


class _FakeResponse:
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"ok": true}'


def test_request_json_uses_configured_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    requests = []

    def fake_urlopen(request: Any, timeout: int) -> _FakeResponse:
        requests.append((request, timeout))
        return _FakeResponse()

    monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)

    response = model._request_json(
        credentials=_credentials(),
        method="GET",
        path="images/generations",
    )

    assert response == {"ok": True}
    assert (
        requests[0][0].full_url
        == "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
    )
    assert requests[0][0].get_header("Authorization") == "Bearer test-key"
    assert requests[0][1] == 60


def test_seedance_validate_credentials_uses_non_generation_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    requests: list[dict[str, Any]] = []

    def fake_request_json(**kwargs: Any) -> dict[str, Any]:
        requests.append(kwargs)
        return {"data": []}

    monkeypatch.setattr(model, "_request_json", fake_request_json)

    model.validate_credentials("seedance-2-0-260128", _credentials())

    assert requests == [
        {
            "credentials": _credentials(),
            "method": "GET",
            "path": "contents/generations/tasks?page_num=1&page_size=1",
        }
    ]


def test_seedream_validate_credentials_uses_image_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    requests: list[dict[str, Any]] = []

    def fake_request_json(**kwargs: Any) -> dict[str, Any]:
        requests.append(kwargs)
        raise _ArkHTTPError(405, "method not allowed")

    monkeypatch.setattr(model, "_request_json", fake_request_json)

    model.validate_credentials("seedream-5-0-260128", _credentials())

    assert requests == [
        {
            "credentials": _credentials(),
            "method": "GET",
            "path": "images/generations",
        }
    ]


def test_polling_validate_credentials_rejects_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()

    def fake_request_json(**_: Any) -> dict[str, Any]:
        raise _ArkHTTPError(401, "unauthorized")

    monkeypatch.setattr(model, "_request_json", fake_request_json)

    with pytest.raises(CredentialsValidateFailedError):
        model.validate_credentials("seedream-5-0-260128", _credentials())


def test_live_credentials_validation() -> None:
    api_key = os.getenv("BYTEPLUS_API_KEY")
    if not api_key:
        pytest.skip("BYTEPLUS_API_KEY is not set")

    model = _model()
    model.validate_credentials(
        "seedance-2-0-260128",
        {
            "ark_api_key": api_key,
            "api_endpoint_host": os.getenv(
                "BYTEPLUS_API_ENDPOINT",
                "https://ark.ap-southeast.bytepluses.com/api/v3",
            ),
        },
    )


def test_seedance_start_polling_creates_task_without_web_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    requests: list[dict[str, Any]] = []

    def fake_request_json(**kwargs: Any) -> dict[str, Any]:
        requests.append(kwargs)
        return {"id": "task-1", "status": "queued"}

    monkeypatch.setattr(model, "_request_json", fake_request_json)

    result = model._start_polling(
        model="seedance-2-0-260128",
        credentials=_credentials(),
        prompt_messages=[
            UserPromptMessage(
                content=[
                    TextPromptMessageContent(data="make a calm ocean video"),
                    ImagePromptMessageContent(
                        format="png",
                        mime_type="image/png",
                        url="https://example.com/frame.png",
                    ),
                ],
            )
        ],
        model_parameters={"duration": 5, "resolution": "720p", "web_search": True},
        stream=False,
        workflow_run_id="wr-1",
        node_id="llm-1",
    )

    assert result.status == LLMPollingStatus.RUNNING
    assert result.plugin_state == {
        "task_id": "task-1",
        "model": "seedance-2-0-260128",
        "platform": "byteplus",
    }
    assert requests[0]["method"] == "POST"
    assert requests[0]["path"] == "contents/generations/tasks"
    assert requests[0]["payload"] == {
        "model": "seedance-2-0-260128",
        "content": [
            {"type": "text", "text": "make a calm ocean video"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/frame.png"},
                "role": "first_frame",
            },
        ],
        "duration": 5,
        "resolution": "720p",
    }


def test_seedance_check_polling_returns_video_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    monkeypatch.setattr(
        model,
        "_usage_from_provider_payload",
        lambda **_: LLMUsage.empty_usage(),
    )

    def fake_request_json(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["method"] == "GET"
        assert kwargs["path"] == "contents/generations/tasks/task-1"
        return {
            "id": "task-1",
            "status": "succeeded",
            "content": {"video_url": "https://example.com/result.mp4"},
            "usage": {"completion_tokens": 12, "total_tokens": 12},
        }

    monkeypatch.setattr(model, "_request_json", fake_request_json)

    result = model._check_polling(
        model="seedance-2-0-260128",
        credentials=_credentials(),
        plugin_state={"task_id": "task-1"},
        workflow_run_id="wr-1",
        node_id="llm-1",
    )

    assert result.status == LLMPollingStatus.SUCCEEDED
    assert result.result is not None
    assert isinstance(result.result.message.content, list)
    video = result.result.message.content[0]
    assert isinstance(video, VideoPromptMessageContent)
    assert video.url == "https://example.com/result.mp4"
    assert video.mime_type == "video/mp4"


@pytest.mark.parametrize("error", [_ArkHTTPError(429, "rate limited"), URLError("timed out")])
def test_seedance_check_polling_keeps_running_on_retryable_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    model = _model()

    def fake_request_json(**_: Any) -> dict[str, Any]:
        raise error

    monkeypatch.setattr(model, "_request_json", fake_request_json)

    result = model._check_polling(
        model="seedance-2-0-260128",
        credentials=_credentials(),
        plugin_state={"task_id": "task-1"},
        workflow_run_id="wr-1",
        node_id="llm-1",
    )

    assert result.status == LLMPollingStatus.RUNNING
    assert result.plugin_state == {"task_id": "task-1"}
    assert result.next_check_after_seconds == 10


def test_seedance_15_rejects_video_input() -> None:
    model = _model()

    with pytest.raises(InvokeError, match="video and audio input is only supported"):
        model._start_polling(
            model="seedance-1-5-pro-251215",
            credentials=_credentials(),
            prompt_messages=[
                UserPromptMessage(
                    content=[
                        TextPromptMessageContent(data="make a video"),
                        VideoPromptMessageContent(
                            format="mp4",
                            mime_type="video/mp4",
                            url="https://example.com/input.mp4",
                        ),
                    ],
                )
            ],
            model_parameters={},
            stream=False,
            workflow_run_id="wr-1",
            node_id="llm-1",
        )


def test_seedance_2_accepts_reference_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    requests: list[dict[str, Any]] = []

    def fake_request_json(**kwargs: Any) -> dict[str, Any]:
        requests.append(kwargs)
        return {"id": "task-1", "status": "queued"}

    monkeypatch.setattr(model, "_request_json", fake_request_json)

    result = model._start_polling(
        model="seedance-2-0-260128",
        credentials=_credentials(),
        prompt_messages=[
            UserPromptMessage(
                content=[
                    TextPromptMessageContent(data="animate this scene with music"),
                    ImagePromptMessageContent(
                        format="png",
                        mime_type="image/png",
                        url="https://example.com/reference.png",
                    ),
                    AudioPromptMessageContent(
                        format="mp3",
                        mime_type="audio/mpeg",
                        url="https://example.com/reference.mp3",
                    ),
                ],
            )
        ],
        model_parameters={},
        stream=False,
        workflow_run_id="wr-1",
        node_id="llm-1",
    )

    assert result.status == LLMPollingStatus.RUNNING
    assert requests[0]["payload"]["content"][1]["role"] == "reference_image"
    assert requests[0]["payload"]["content"][2]["role"] == "reference_audio"


def test_seedream_start_polling_returns_b64_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    requests: list[dict[str, Any]] = []
    monkeypatch.setattr(
        model,
        "_usage_from_provider_payload",
        lambda **_: LLMUsage.empty_usage(),
    )

    def fake_request_json(**kwargs: Any) -> dict[str, Any]:
        requests.append(kwargs)
        return {
            "model": "seedream-5-0-260128",
            "data": [{"b64_json": "aW1hZ2U="}],
        }

    monkeypatch.setattr(model, "_request_json", fake_request_json)

    result = model._start_polling(
        model="seedream-5-0-260128",
        credentials=_credentials(),
        prompt_messages=[UserPromptMessage(content="draw a small cabin")],
        model_parameters={
            "response_format": "b64_json",
            "output_format": "png",
            "max_images": 3,
            "web_search": True,
        },
        stream=False,
        workflow_run_id="wr-1",
        node_id="llm-1",
    )

    assert result.status == LLMPollingStatus.SUCCEEDED
    assert requests[0]["payload"]["model"] == "seedream-5-0-260128"
    assert "tools" not in requests[0]["payload"]
    assert requests[0]["payload"]["sequential_image_generation_options"] == {
        "max_images": 3
    }
    assert result.result is not None
    assert isinstance(result.result.message.content, list)
    image = result.result.message.content[0]
    assert isinstance(image, ImagePromptMessageContent)
    assert image.base64_data == "aW1hZ2U="
    assert image.mime_type == "image/png"


@pytest.mark.parametrize(
    ("response", "expected_error"),
    [
        (
            {
                "error": {
                    "code": "SensitiveContentDetected",
                    "message": "content rejected",
                }
            },
            "SensitiveContentDetected: content rejected",
        ),
        (
            {
                "data": [
                    {
                        "error": {
                            "code": "ImageFailed",
                            "message": "one image failed",
                        }
                    }
                ]
            },
            "ImageFailed: one image failed",
        ),
    ],
)
def test_seedream_start_polling_failed_responses(
    monkeypatch: pytest.MonkeyPatch,
    response: dict[str, Any],
    expected_error: str,
) -> None:
    model = _model()

    def fake_request_json(**_: Any) -> dict[str, Any]:
        return response

    monkeypatch.setattr(model, "_request_json", fake_request_json)

    result = model._start_polling(
        model="seedream-5-0-260128",
        credentials=_credentials(),
        prompt_messages=[UserPromptMessage(content="draw a cabin")],
        model_parameters={},
        stream=False,
        workflow_run_id="wr-1",
        node_id="llm-1",
    )

    assert result.status == LLMPollingStatus.FAILED
    assert result.error == expected_error


@pytest.mark.parametrize(
    ("task_payload", "expected_error"),
    [
        (
            {
                "id": "task-1",
                "status": "failed",
                "error": {"code": "TaskFailed", "message": "provider failed"},
            },
            "TaskFailed: provider failed",
        ),
        ({"id": "task-1", "status": "mystery"}, "unknown task status"),
        (
            {"id": "task-1", "status": "succeeded", "content": {}},
            "without video_url",
        ),
    ],
)
def test_seedance_check_polling_failed_terminal_states(
    monkeypatch: pytest.MonkeyPatch,
    task_payload: dict[str, Any],
    expected_error: str,
) -> None:
    model = _model()

    def fake_request_json(**_: Any) -> dict[str, Any]:
        return task_payload

    monkeypatch.setattr(model, "_request_json", fake_request_json)

    result = model._check_polling(
        model="seedance-2-0-260128",
        credentials=_credentials(),
        plugin_state={"task_id": "task-1"},
        workflow_run_id="wr-1",
        node_id="llm-1",
    )

    assert result.status == LLMPollingStatus.FAILED
    assert result.error is not None
    assert expected_error in result.error


def test_plugin_registration_loads_byteplus_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(plugin_root)

    registration = PluginRegistration(DifyPluginEnv())

    assert "byteplus" in registration.models_mapping
    assert "volcengine" not in registration.models_mapping

    models = {
        model.model
        for model in registration.models_mapping["byteplus"][0].models
        if model.model_type == ModelType.LLM
    }

    assert "seedance-2-0-260128" in models
    assert "doubao-seedance-2-0-260128" not in models
