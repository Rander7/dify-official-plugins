import json
import logging
from collections.abc import Generator
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime._streaming import Stream
from volcenginesdkarkruntime.types.chat import ChatCompletion, ChatCompletionChunk

from dify_plugin.entities.model.llm import (
    LLMPollingResult,
    LLMPollingStatus,
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    AudioPromptMessageContent,
    DocumentPromptMessageContent,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageTool,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
    VideoPromptMessageContent,
)
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeError
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class _PlatformSpec:
    name: str
    seedance_prefix: str
    seedream_prefix: str
    supports_web_search: bool


_BYTEPLUS_PLATFORM = _PlatformSpec(
    name="byteplus",
    seedance_prefix="seedance-",
    seedream_prefix="seedream-",
    supports_web_search=False,
)
_PLATFORM_SPECS = (_BYTEPLUS_PLATFORM,)

_SEEDANCE_RUNNING_STATUSES = {"queued", "running"}
_SEEDANCE_FAILED_STATUSES = {"failed", "expired", "cancelled"}
_DEFAULT_POLLING_INTERVAL_SECONDS = 10
_DEFAULT_POLLING_EXPIRES_AFTER_SECONDS = 1800
_DEFAULT_POLLING_MAX_ATTEMPTS = 60
_DEFAULT_SEEDANCE_INPUT_MODE = "first_frame"
_SEEDANCE_INPUT_MODES = {"first_frame", "first_last_frame", "reference_image"}
_SEEDANCE_IMAGE_ROLES = {"first_frame", "last_frame", "reference_image"}

_SEEDANCE_MODEL_PARAMETER_NAMES = {
    "ratio",
    "duration",
    "frames",
    "resolution",
    "seed",
    "camera_fixed",
    "generate_audio",
    "watermark",
    "return_last_frame",
    "draft",
    "callback_url",
    "safety_identifier",
    "service_tier",
    "execution_expires_after",
    "priority",
}
_SEEDREAM_MODEL_PARAMETER_NAMES = {
    "size",
    "response_format",
    "watermark",
    "sequential_image_generation",
    "sequential_image_generation_options",
    "guidance_scale",
    "output_format",
    "optimize_prompt_options",
}


class _ArkHTTPError(Exception):
    def __init__(self, status_code: int, response_text: str) -> None:
        super().__init__(
            f"API request failed with status code {status_code}: {response_text}"
        )
        self.status_code = status_code
        self.response_text = response_text


def _convert_prompt_message_tool_to_dict(tool: PromptMessageTool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _convert_content_to_ark(content: Any) -> Any:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[dict[str, Any]] = []
    for message_content in content:
        if message_content.type == PromptMessageContentType.TEXT:
            message_content = cast(TextPromptMessageContent, message_content)
            parts.append({"type": "text", "text": message_content.data})
        elif message_content.type == PromptMessageContentType.IMAGE:
            message_content = cast(ImagePromptMessageContent, message_content)
            detail = "high" if message_content.detail == ImagePromptMessageContent.DETAIL.HIGH else "low"
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": message_content.data,
                        "detail": detail,
                    },
                }
            )
        elif message_content.type == PromptMessageContentType.VIDEO:
            message_content = cast(VideoPromptMessageContent, message_content)
            parts.append({"type": "video_url", "video_url": {"url": message_content.data}})
        elif message_content.type == PromptMessageContentType.AUDIO:
            message_content = cast(AudioPromptMessageContent, message_content)
            parts.append({"type": "text", "text": message_content.data})
        elif message_content.type == PromptMessageContentType.DOCUMENT:
            message_content = cast(DocumentPromptMessageContent, message_content)
            parts.append({"type": "text", "text": message_content.data})
        else:
            parts.append({"type": "text", "text": str(message_content)})

    return parts


def _convert_prompt_message_to_dict(message: PromptMessage) -> dict[str, Any]:
    if isinstance(message, SystemPromptMessage):
        return {"role": "system", "content": _convert_content_to_ark(message.content) or ""}

    if isinstance(message, UserPromptMessage):
        return {"role": "user", "content": _convert_content_to_ark(message.content) or ""}

    if isinstance(message, AssistantPromptMessage):
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": _convert_content_to_ark(message.content) or "",
        }

        if message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": call.id,
                    "type": call.type or "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]

        return msg

    if isinstance(message, ToolPromptMessage):
        return {
            "role": "tool",
            "content": _convert_content_to_ark(message.content) or "",
            "tool_call_id": message.tool_call_id,
        }

    role = getattr(getattr(message, "role", None), "value", None) or "user"
    return {"role": role, "content": _convert_content_to_ark(getattr(message, "content", "")) or ""}


def _wrap_thinking(content: str, reasoning_content: str | None, is_reasoning: bool) -> tuple[str, bool]:
    content = content or ""

    if reasoning_content:
        if not is_reasoning:
            return "<think>\n" + reasoning_content, True
        return reasoning_content, True

    if is_reasoning:
        return "\n</think>" + (content or ""), False

    return content, False


def _platform_spec_for_model(model: str) -> _PlatformSpec | None:
    for platform in _PLATFORM_SPECS:
        if model.startswith(platform.seedance_prefix) or model.startswith(
            platform.seedream_prefix
        ):
            return platform
    return None


def _platform_name_for_model(model: str) -> str:
    platform = _platform_spec_for_model(model)
    return platform.name if platform else "unknown"


def _is_seedance_2_model(model: str) -> bool:
    platform = _platform_spec_for_model(model)
    return bool(platform and model.startswith(f"{platform.seedance_prefix}2-"))


def _is_seedance_model(model: str) -> bool:
    platform = _platform_spec_for_model(model)
    return bool(platform and model.startswith(platform.seedance_prefix))


def _is_seedream_model(model: str) -> bool:
    platform = _platform_spec_for_model(model)
    return bool(platform and model.startswith(platform.seedream_prefix))


def _filter_model_parameters(
    model_parameters: dict[str, Any],
    allowed_parameter_names: set[str],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in model_parameters.items()
        if key in allowed_parameter_names and value is not None
    }


def _extract_prompt_text(prompt_messages: list[PromptMessage]) -> str:
    text_parts: list[str] = []
    for message in prompt_messages:
        text = message.get_text_content().strip()
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)


def _iter_message_contents(prompt_messages: list[PromptMessage]):
    for message in prompt_messages:
        if not isinstance(message.content, list):
            continue
        yield from message.content


def _content_role(content: object) -> str | None:
    opaque_body = getattr(content, "opaque_body", None)
    if isinstance(opaque_body, dict):
        role = opaque_body.get("role")
        if isinstance(role, str) and role:
            return role
    return None


def _normalize_seedance_input_mode(value: object) -> str:
    if value is None or value == "":
        return _DEFAULT_SEEDANCE_INPUT_MODE
    if not isinstance(value, str):
        raise InvokeError("Seedance input_mode must be a string.")
    input_mode = value.strip()
    if input_mode not in _SEEDANCE_INPUT_MODES:
        raise InvokeError(
            "Seedance input_mode must be one of: first_frame, first_last_frame, reference_image."
        )
    return input_mode


def _guess_format_from_url(url: str, default: str) -> str:
    suffix = PurePosixPath(url.split("?", 1)[0]).suffix.removeprefix(".").lower()
    return suffix or default


def _guess_image_mime_type(*, url: str = "", output_format: str | None = None) -> str:
    image_format = (output_format or _guess_format_from_url(url, "jpeg")).lower()
    if image_format == "jpg":
        image_format = "jpeg"
    return f"image/{image_format}"


def _format_provider_error(error: object) -> str:
    if isinstance(error, _ArkHTTPError):
        return str(error)
    if isinstance(error, Mapping):
        code = error.get("code")
        message = error.get("message")
        if code and message:
            return f"{code}: {message}"
        if message:
            return str(message)
        if code:
            return str(code)
    return str(error or "provider request failed")


def _is_retryable_http_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600


class BytePlusArkLargeLanguageModel(LargeLanguageModel):
    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {}

    def validate_credentials(self, model: str, credentials: Mapping[str, Any]) -> None:
        if _is_seedance_model(model) or _is_seedream_model(model):
            self._validate_polling_credentials(model, dict(credentials))
            return

        try:
            client = Ark(
                base_url=credentials["api_endpoint_host"],
                api_key=credentials["ark_api_key"],
            )
            # minimal non-stream call
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=8,
            )
        except Exception as e:
            raise CredentialsValidateFailedError(e)

    def _validate_polling_credentials(self, model: str, credentials: dict[str, Any]) -> None:
        path = (
            "contents/generations/tasks?page_num=1&page_size=1"
            if _is_seedance_model(model)
            else "images/generations"
        )
        try:
            self._request_json(
                credentials=credentials,
                method="GET",
                path=path,
            )
        except _ArkHTTPError as error:
            if error.status_code in {400, 405}:
                return
            raise CredentialsValidateFailedError(error) from error
        except Exception as error:
            raise CredentialsValidateFailedError(error) from error

    def get_num_tokens(
        self,
        model: str,
        credentials: dict[str, Any],
        prompt_messages: list[PromptMessage],
        tools: list[PromptMessageTool] | None = None,
    ) -> int:
        # No official token counter exposed here; fall back to rough estimate.
        # This is acceptable for plugin implementations that do not support token counting.
        text = _extract_prompt_text(prompt_messages)
        return max(1, len(text) // 4)

    def _request_json(
        self,
        *,
        credentials: dict[str, Any],
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        endpoint_url = credentials["api_endpoint_host"].rstrip("/") + "/"
        request_url = urljoin(endpoint_url, path.lstrip("/"))
        headers = {
            "Content-Type": "application/json",
            "Accept-Charset": "utf-8",
        }
        api_key = credentials.get("ark_api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(request_url, data=data, headers=headers, method=method)

        try:
            with urlopen(request, timeout=60) as response:
                response_text = response.read().decode("utf-8", errors="replace")
        except HTTPError as error:
            response_text = error.read().decode("utf-8", errors="replace")
            raise _ArkHTTPError(error.code, response_text) from error

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise InvokeError(f"API returned invalid JSON: {response_text}") from error
        if not isinstance(result, dict):
            raise InvokeError(f"API returned unexpected response: {response_text}")
        return result

    def _build_seedance_content(
        self,
        prompt_messages: list[PromptMessage],
        *,
        model: str,
        input_mode: object = None,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        prompt_text = _extract_prompt_text(prompt_messages)
        if prompt_text:
            content.append({"type": "text", "text": prompt_text})

        image_contents: list[ImagePromptMessageContent] = []
        video_contents: list[VideoPromptMessageContent] = []
        audio_contents: list[AudioPromptMessageContent] = []
        for message_content in _iter_message_contents(prompt_messages):
            if isinstance(message_content, ImagePromptMessageContent):
                image_contents.append(message_content)
            elif isinstance(message_content, VideoPromptMessageContent):
                video_contents.append(message_content)
            elif isinstance(message_content, AudioPromptMessageContent):
                audio_contents.append(message_content)

        if (video_contents or audio_contents) and not _is_seedance_2_model(model):
            raise InvokeError("Seedance video and audio input is only supported by Seedance 2.0 models.")
        if len(video_contents) > 3:
            raise InvokeError("Seedance supports at most three reference videos.")
        if len(audio_contents) > 3:
            raise InvokeError("Seedance supports at most three reference audios.")
        if audio_contents and not (image_contents or video_contents):
            raise InvokeError("Seedance audio input requires at least one reference image or video.")

        image_roles = self._build_seedance_image_roles(
            model=model,
            image_contents=image_contents,
            input_mode=input_mode,
            reference_mode=bool(video_contents or audio_contents),
        )
        for image_content, role in zip(image_contents, image_roles, strict=True):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_content.data},
                    "role": role,
                }
            )

        for video_content in video_contents:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": video_content.data},
                    "role": _content_role(video_content) or "reference_video",
                }
            )

        for audio_content in audio_contents:
            content.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": audio_content.data},
                    "role": _content_role(audio_content) or "reference_audio",
                }
            )

        if not content:
            raise InvokeError("Seedance requires prompt text or multimodal input.")
        return content

    def _build_seedance_image_roles(
        self,
        *,
        model: str,
        image_contents: list[ImagePromptMessageContent],
        input_mode: object,
        reference_mode: bool,
    ) -> list[str]:
        if not image_contents:
            return []

        explicit_roles = [_content_role(image_content) for image_content in image_contents]
        if any(role is not None for role in explicit_roles):
            if any(role is None for role in explicit_roles):
                raise InvokeError("Seedance image roles must be provided for every image or omitted entirely.")
            roles = [cast(str, role) for role in explicit_roles]
            if reference_mode and any(role != "reference_image" for role in roles):
                raise InvokeError("Seedance video and audio input cannot be mixed with frame image roles.")
            self._validate_seedance_image_roles(model=model, roles=roles)
            return roles

        if reference_mode:
            if len(image_contents) > 9:
                raise InvokeError("Seedance reference_image input_mode supports at most nine images.")
            return ["reference_image" for _ in image_contents]

        normalized_input_mode = _normalize_seedance_input_mode(input_mode)
        if normalized_input_mode == "first_frame":
            if len(image_contents) != 1:
                raise InvokeError("Seedance first_frame input_mode requires exactly one image.")
            return ["first_frame"]
        if normalized_input_mode == "first_last_frame":
            if len(image_contents) != 2:
                raise InvokeError("Seedance first_last_frame input_mode requires exactly two images.")
            return ["first_frame", "last_frame"]

        if not _is_seedance_2_model(model):
            raise InvokeError("Seedance reference_image input_mode is only supported by Seedance 2.0 models.")
        if len(image_contents) > 9:
            raise InvokeError("Seedance reference_image input_mode supports at most nine images.")
        return ["reference_image" for _ in image_contents]

    def _validate_seedance_image_roles(self, *, model: str, roles: list[str]) -> None:
        unsupported_roles = [role for role in roles if role not in _SEEDANCE_IMAGE_ROLES]
        if unsupported_roles:
            raise InvokeError(f"Unsupported Seedance image role: {unsupported_roles[0]}")
        if "reference_image" in roles:
            if any(role != "reference_image" for role in roles):
                raise InvokeError("Seedance reference_image cannot be mixed with frame roles.")
            if not _is_seedance_2_model(model):
                raise InvokeError("Seedance reference_image role is only supported by Seedance 2.0 models.")
            if len(roles) > 9:
                raise InvokeError("Seedance reference_image role supports at most nine images.")
            return
        if roles == ["first_frame"]:
            return
        if roles == ["first_frame", "last_frame"]:
            return
        raise InvokeError("Seedance frame roles must be first_frame or first_frame followed by last_frame.")

    def _extract_seedream_images(
        self,
        prompt_messages: list[PromptMessage],
    ) -> list[str]:
        return [
            message_content.data
            for message_content in _iter_message_contents(prompt_messages)
            if isinstance(message_content, ImagePromptMessageContent)
        ]

    def _usage_from_provider_payload(
        self,
        *,
        model: str,
        credentials: dict[str, Any],
        usage_payload: object,
        completion_token_keys: tuple[str, ...],
    ):
        usage = usage_payload if isinstance(usage_payload, Mapping) else {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = 0
        for key in completion_token_keys:
            value = usage.get(key)
            if value is not None:
                completion_tokens = int(value)
                break
        if completion_tokens == 0:
            completion_tokens = int(usage.get("total_tokens") or 0)
        return self._calc_response_usage(
            model=model,
            credentials=credentials,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _seedance_polling_result_from_task(
        self,
        *,
        model: str,
        credentials: dict[str, Any],
        prompt_messages: list[PromptMessage],
        task_payload: dict[str, Any],
    ) -> LLMPollingResult:
        task_id = task_payload.get("id")
        if not isinstance(task_id, str) or not task_id:
            return LLMPollingResult(
                status=LLMPollingStatus.FAILED,
                error="Seedance response did not include a task id.",
            )

        status = str(task_payload.get("status") or "running").lower()
        plugin_state = {
            "task_id": task_id,
            "model": model,
            "platform": _platform_name_for_model(model),
        }
        if status in _SEEDANCE_RUNNING_STATUSES:
            return LLMPollingResult(
                status=LLMPollingStatus.RUNNING,
                plugin_state=plugin_state,
                next_check_after_seconds=_DEFAULT_POLLING_INTERVAL_SECONDS,
                expires_after_seconds=_DEFAULT_POLLING_EXPIRES_AFTER_SECONDS,
                max_attempts=_DEFAULT_POLLING_MAX_ATTEMPTS,
            )

        if status in _SEEDANCE_FAILED_STATUSES:
            return LLMPollingResult(
                status=LLMPollingStatus.FAILED,
                plugin_state=plugin_state,
                error=_format_provider_error(task_payload.get("error")),
            )

        if status != "succeeded":
            return LLMPollingResult(
                status=LLMPollingStatus.FAILED,
                plugin_state=plugin_state,
                error=f"Seedance returned unknown task status: {status}",
            )

        output_content = task_payload.get("content")
        if not isinstance(output_content, Mapping):
            return LLMPollingResult(
                status=LLMPollingStatus.FAILED,
                plugin_state=plugin_state,
                error="Seedance task succeeded without output content.",
            )
        video_url = output_content.get("video_url")
        if not isinstance(video_url, str) or not video_url:
            return LLMPollingResult(
                status=LLMPollingStatus.FAILED,
                plugin_state=plugin_state,
                error="Seedance task succeeded without video_url.",
            )

        assistant_contents: list[Any] = [
            VideoPromptMessageContent(
                format=_guess_format_from_url(video_url, "mp4"),
                mime_type="video/mp4",
                url=video_url,
                filename=f"{task_id}.mp4",
            )
        ]
        last_frame_url = output_content.get("last_frame_url")
        if isinstance(last_frame_url, str) and last_frame_url:
            assistant_contents.append(
                ImagePromptMessageContent(
                    format=_guess_format_from_url(last_frame_url, "jpeg"),
                    mime_type=_guess_image_mime_type(url=last_frame_url),
                    url=last_frame_url,
                    filename=f"{task_id}-last-frame.jpg",
                )
            )

        return LLMPollingResult(
            status=LLMPollingStatus.SUCCEEDED,
            plugin_state=plugin_state,
            result=LLMResult(
                model=model,
                prompt_messages=prompt_messages,
                message=AssistantPromptMessage(content=assistant_contents),
                usage=self._usage_from_provider_payload(
                    model=model,
                    credentials=credentials,
                    usage_payload=task_payload.get("usage"),
                    completion_token_keys=("completion_tokens",),
                ),
            ),
        )

    def _start_polling(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: list[PromptMessageTool] | None = None,
        stop: list[str] | None = None,
        stream: bool = False,
        user: str | None = None,
        *,
        workflow_run_id: str,
        node_id: str,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMPollingResult:
        del tools, stop, stream, workflow_run_id, node_id, json_schema
        platform = _platform_spec_for_model(model)
        if _is_seedance_model(model):
            payload = {
                "model": model,
                "content": self._build_seedance_content(
                    prompt_messages,
                    model=model,
                    input_mode=model_parameters.get("input_mode"),
                ),
                **_filter_model_parameters(
                    model_parameters,
                    _SEEDANCE_MODEL_PARAMETER_NAMES,
                ),
            }
            if platform and platform.supports_web_search and model_parameters.get(
                "web_search"
            ):
                payload["tools"] = [{"type": "web_search"}]
            if user:
                payload["safety_identifier"] = user
            try:
                response = self._request_json(
                    credentials=credentials,
                    method="POST",
                    path="contents/generations/tasks",
                    payload=payload,
                )
            except _ArkHTTPError as error:
                return LLMPollingResult(
                    status=LLMPollingStatus.FAILED,
                    error=_format_provider_error(error),
                )
            return self._seedance_polling_result_from_task(
                model=model,
                credentials=credentials,
                prompt_messages=prompt_messages,
                task_payload=response,
            )

        if _is_seedream_model(model):
            prompt = _extract_prompt_text(prompt_messages)
            if not prompt:
                raise InvokeError("Seedream requires prompt text.")
            payload = {
                "model": model,
                "prompt": prompt,
                **_filter_model_parameters(
                    model_parameters,
                    _SEEDREAM_MODEL_PARAMETER_NAMES,
                ),
            }
            max_images = model_parameters.get("max_images")
            if max_images is not None:
                payload["sequential_image_generation_options"] = {
                    "max_images": int(max_images)
                }
            if platform and platform.supports_web_search and model_parameters.get(
                "web_search"
            ):
                payload["tools"] = [{"type": "web_search"}]
            images = self._extract_seedream_images(prompt_messages)
            if images:
                payload["image"] = images[0] if len(images) == 1 else images
            try:
                response = self._request_json(
                    credentials=credentials,
                    method="POST",
                    path="images/generations",
                    payload=payload,
                )
            except _ArkHTTPError as error:
                return LLMPollingResult(
                    status=LLMPollingStatus.FAILED,
                    error=_format_provider_error(error),
                )
            return self._seedream_polling_result_from_response(
                model=model,
                credentials=credentials,
                prompt_messages=prompt_messages,
                response=response,
                output_format=payload.get("output_format"),
            )

        raise NotImplementedError(f"Model `{model}` does not support polling.")

    def _check_polling(
        self,
        model: str,
        credentials: dict,
        plugin_state: dict[str, Any],
        user: str | None = None,
        *,
        workflow_run_id: str,
        node_id: str,
    ) -> LLMPollingResult:
        del user, workflow_run_id, node_id
        if _is_seedance_model(model):
            task_id = (
                plugin_state.get("task_id")
                or plugin_state.get("job_id")
                or plugin_state.get("id")
            )
            if not isinstance(task_id, str) or not task_id:
                return LLMPollingResult(
                    status=LLMPollingStatus.FAILED,
                    error="Seedance check requires task_id in plugin_state.",
                )
            try:
                response = self._request_json(
                    credentials=credentials,
                    method="GET",
                    path=f"contents/generations/tasks/{task_id}",
                )
            except _ArkHTTPError as error:
                if _is_retryable_http_status(error.status_code):
                    return LLMPollingResult(
                        status=LLMPollingStatus.RUNNING,
                        plugin_state=dict(plugin_state),
                        next_check_after_seconds=_DEFAULT_POLLING_INTERVAL_SECONDS,
                        expires_after_seconds=_DEFAULT_POLLING_EXPIRES_AFTER_SECONDS,
                        max_attempts=_DEFAULT_POLLING_MAX_ATTEMPTS,
                    )
                return LLMPollingResult(
                    status=LLMPollingStatus.FAILED,
                    plugin_state=dict(plugin_state),
                    error=_format_provider_error(error),
                )
            except (URLError, TimeoutError) as error:
                logger.warning("Seedance check request failed: %s", error)
                return LLMPollingResult(
                    status=LLMPollingStatus.RUNNING,
                    plugin_state=dict(plugin_state),
                    next_check_after_seconds=_DEFAULT_POLLING_INTERVAL_SECONDS,
                    expires_after_seconds=_DEFAULT_POLLING_EXPIRES_AFTER_SECONDS,
                    max_attempts=_DEFAULT_POLLING_MAX_ATTEMPTS,
                )
            return self._seedance_polling_result_from_task(
                model=model,
                credentials=credentials,
                prompt_messages=[],
                task_payload=response,
            )

        if _is_seedream_model(model):
            return LLMPollingResult(
                status=LLMPollingStatus.FAILED,
                error="Seedream returns a terminal polling result from start_polling.",
            )

        raise NotImplementedError(f"Model `{model}` does not support polling.")

    def _seedream_polling_result_from_response(
        self,
        *,
        model: str,
        credentials: dict[str, Any],
        prompt_messages: list[PromptMessage],
        response: dict[str, Any],
        output_format: object,
    ) -> LLMPollingResult:
        if response.get("error"):
            return LLMPollingResult(
                status=LLMPollingStatus.FAILED,
                error=_format_provider_error(response.get("error")),
            )

        images = response.get("data")
        if not isinstance(images, list):
            return LLMPollingResult(
                status=LLMPollingStatus.FAILED,
                error="Seedream response did not include image data.",
            )

        assistant_contents: list[Any] = []
        image_errors: list[str] = []
        for index, image in enumerate(images):
            if not isinstance(image, Mapping):
                continue
            if image.get("error"):
                image_errors.append(_format_provider_error(image.get("error")))
                continue
            image_url = image.get("url")
            if isinstance(image_url, str) and image_url:
                image_format = _guess_format_from_url(
                    image_url,
                    str(output_format or "jpeg"),
                )
                assistant_contents.append(
                    ImagePromptMessageContent(
                        format=image_format,
                        mime_type=_guess_image_mime_type(
                            url=image_url,
                            output_format=str(output_format) if output_format else None,
                        ),
                        url=image_url,
                        filename=f"{model}-{index + 1}.{image_format}",
                    )
                )
                continue

            b64_json = image.get("b64_json")
            if isinstance(b64_json, str) and b64_json:
                image_format = str(output_format or "jpeg")
                assistant_contents.append(
                    ImagePromptMessageContent(
                        format=image_format,
                        mime_type=_guess_image_mime_type(output_format=image_format),
                        base64_data=b64_json,
                        filename=f"{model}-{index + 1}.{image_format}",
                    )
                )

        if not assistant_contents:
            error = (
                image_errors[0]
                if image_errors
                else "Seedream response did not include generated images."
            )
            return LLMPollingResult(
                status=LLMPollingStatus.FAILED,
                error=error,
            )

        return LLMPollingResult(
            status=LLMPollingStatus.SUCCEEDED,
            result=LLMResult(
                model=model,
                prompt_messages=prompt_messages,
                message=AssistantPromptMessage(content=assistant_contents),
                usage=self._usage_from_provider_payload(
                    model=model,
                    credentials=credentials,
                    usage_payload=response.get("usage"),
                    completion_token_keys=("output_tokens",),
                ),
            ),
        )

    def _invoke(
        self,
        model: str,
        credentials: dict[str, Any],
        prompt_messages: list[PromptMessage],
        model_parameters: dict[str, Any],
        tools: list[PromptMessageTool] | None = None,
        stop: list[str] | None = None,
        stream: bool = True,
        user: str | None = None,
    ) -> LLMResult | Generator[LLMResultChunk, None, None]:
        client = Ark(
            base_url=credentials["api_endpoint_host"],
            api_key=credentials["ark_api_key"],
        )

        messages = [_convert_prompt_message_to_dict(m) for m in prompt_messages]

        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": model_parameters.get("temperature"),
            "top_p": model_parameters.get("top_p"),
            "max_tokens": model_parameters.get("max_tokens"),
            "stop": stop,
            "user": user,
        }

        if model_parameters.get("thinking"):
            thinking_type = model_parameters["thinking"]
            params["thinking"] = {"type": thinking_type}
            if thinking_type == "disabled":
                params["reasoning_effort"] = "minimal"

        if tools:
            params["tools"] = [_convert_prompt_message_tool_to_dict(t) for t in tools]

            if "tool_choice" in model_parameters:
                params["tool_choice"] = model_parameters.get("tool_choice")
            if "parallel_tool_calls" in model_parameters:
                params["parallel_tool_calls"] = model_parameters.get("parallel_tool_calls")

        def _handle_stream() -> Generator[LLMResultChunk, None, None]:
            try:
                stream_options = model_parameters.get("stream_options")
                if stream_options is None:
                    stream_options = {"include_usage": True}

                req = {k: v for k, v in params.items() if v is not None}
                req["stream_options"] = stream_options

                resp = cast(
                    Stream[ChatCompletionChunk],
                    client.chat.completions.create(**req, stream=True),
                )

                aggregated_tool_calls: dict[int, AssistantPromptMessage.ToolCall] = {}
                usage_obj = None
                is_reasoning_started = False

                chunk_index = 0

                final_chunk = LLMResultChunk(
                    model=model,
                    prompt_messages=prompt_messages,
                    delta=LLMResultChunkDelta(
                        index=0,
                        message=AssistantPromptMessage(content=""),
                    ),
                )

                for chunk in resp:
                    if len(chunk.choices) == 0:
                        if chunk.usage:
                            usage_obj = chunk.usage
                        continue

                    choice = chunk.choices[0]
                    delta = choice.delta

                    delta_content = delta.content or ""
                    delta_reasoning = delta.reasoning_content
                    processed_content, is_reasoning_started = _wrap_thinking(
                        delta_content, delta_reasoning, is_reasoning_started
                    )

                    if delta.tool_calls:
                        for tool_call_chunk in delta.tool_calls:
                            idx = tool_call_chunk.index
                            existing = aggregated_tool_calls.get(idx)
                            if existing is None:
                                fn_name = ""
                                fn_args = ""
                                if tool_call_chunk.function:
                                    fn_name = tool_call_chunk.function.name or ""
                                    fn_args = tool_call_chunk.function.arguments or ""
                                aggregated_tool_calls[idx] = AssistantPromptMessage.ToolCall(
                                    id=tool_call_chunk.id or "",
                                    type=tool_call_chunk.type or "function",
                                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                        name=fn_name,
                                        arguments=fn_args,
                                    ),
                                )
                            else:
                                if tool_call_chunk.id:
                                    existing.id = tool_call_chunk.id
                                if tool_call_chunk.type:
                                    existing.type = tool_call_chunk.type
                                if tool_call_chunk.function:
                                    if tool_call_chunk.function.name:
                                        existing.function.name = tool_call_chunk.function.name
                                    if tool_call_chunk.function.arguments:
                                        existing.function.arguments += tool_call_chunk.function.arguments

                    if choice.finish_reason == "tool_calls" and aggregated_tool_calls:
                        tool_calls = [
                            aggregated_tool_calls[i]
                            for i in sorted(aggregated_tool_calls)
                            if aggregated_tool_calls[i] is not None
                        ]
                        yield LLMResultChunk(
                            model=chunk.model,
                            prompt_messages=prompt_messages,
                            delta=LLMResultChunkDelta(
                                index=chunk_index,
                                message=AssistantPromptMessage(content="", tool_calls=tool_calls),
                                finish_reason="tool_calls",
                            ),
                        )
                        chunk_index += 1
                        continue

                    if processed_content:
                        yield LLMResultChunk(
                            model=chunk.model,
                            prompt_messages=prompt_messages,
                            delta=LLMResultChunkDelta(
                                index=chunk_index,
                                message=AssistantPromptMessage(content=processed_content),
                            ),
                        )
                        chunk_index += 1

                    if choice.finish_reason is not None and choice.finish_reason != "tool_calls":
                        final_chunk = LLMResultChunk(
                            model=chunk.model,
                            prompt_messages=prompt_messages,
                            delta=LLMResultChunkDelta(
                                index=chunk_index,
                                message=AssistantPromptMessage(content=""),
                                finish_reason=choice.finish_reason,
                            ),
                        )

                if usage_obj is not None:
                    try:
                        usage = self._calc_response_usage(
                            model=model,
                            credentials=credentials,
                            prompt_tokens=usage_obj.prompt_tokens,
                            completion_tokens=usage_obj.completion_tokens,
                        )
                        final_chunk.delta.usage = usage
                    except Exception:
                        pass

                yield final_chunk
            except Exception as e:
                raise InvokeError(str(e))

        def _handle_block() -> LLMResult:
            try:
                resp = cast(
                    ChatCompletion,
                    client.chat.completions.create(
                        **{k: v for k, v in params.items() if v is not None},
                        stream=False,
                    ),
                )
                choice = resp.choices[0]
                msg = choice.message

                tool_calls = []
                if msg.tool_calls:
                    for call in msg.tool_calls:
                        tool_calls.append(
                            AssistantPromptMessage.ToolCall(
                                id=call.id,
                                type=call.type,
                                function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                    name=call.function.name,
                                    arguments=call.function.arguments,
                                ),
                            )
                        )

                content = msg.content or ""
                reasoning_content = msg.reasoning_content
                if reasoning_content:
                    content = f"<think>\n{reasoning_content}\n</think>\n" + (content or "")

                usage_obj = resp.usage
                if usage_obj is None:
                    prompt_tokens = self.get_num_tokens(model, credentials, prompt_messages, tools)
                    completion_tokens = max(1, len(content) // 4)
                else:
                    prompt_tokens = usage_obj.prompt_tokens
                    completion_tokens = usage_obj.completion_tokens

                usage = self._calc_response_usage(
                    model=model,
                    credentials=credentials,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

                return LLMResult(
                    model=model,
                    prompt_messages=prompt_messages,
                    message=AssistantPromptMessage(content=content, tool_calls=tool_calls),
                    usage=usage,
                )
            except Exception as e:
                raise InvokeError(str(e))

        return _handle_stream() if stream else _handle_block()
