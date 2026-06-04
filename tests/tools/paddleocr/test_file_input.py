import base64
import os
import sys

import pytest
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
PLUGIN_DIR = os.path.join(REPO_ROOT, "tools", "paddleocr")
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

from dify_plugin.file.file import File, FileType  # noqa: E402
from tools.document_parsing import DocumentParsingTool  # noqa: E402
from tools.document_parsing_vl import DocumentParsingVlTool  # noqa: E402
from tools.text_recognition import TextRecognitionTool  # noqa: E402
from tools.utils import normalize_file_input  # noqa: E402


def make_file(
    content: bytes,
    *,
    filename: str,
    mime_type: str | None,
    extension: str | None,
    file_type: FileType = FileType.DOCUMENT,
) -> File:
    file = File(
        url=f"https://example.com/files/{filename}",
        mime_type=mime_type,
        filename=filename,
        extension=extension,
        type=file_type,
    )
    file._blob = content
    return file


def test_file_upload_is_base64_encoded():
    file = make_file(
        b"image-bytes",
        filename="receipt.png",
        mime_type="image/png",
        extension=".png",
        file_type=FileType.IMAGE,
    )

    input_value, is_temp_file, file_type_code = normalize_file_input(file, "auto")

    assert input_value == base64.b64encode(b"image-bytes").decode("utf-8")
    assert is_temp_file is True
    assert file_type_code == 1


def test_pdf_file_upload_infers_file_type():
    file = make_file(
        b"%PDF-1.7", filename="invoice.pdf", mime_type="application/pdf", extension=".pdf"
    )

    input_value, is_temp_file, file_type_code = normalize_file_input(file, "auto")

    assert input_value == base64.b64encode(b"%PDF-1.7").decode("utf-8")
    assert is_temp_file is True
    assert file_type_code == 0


def test_image_file_upload_infers_file_type_from_filename_when_mime_type_missing():
    file = make_file(
        b"image-bytes",
        filename="scan.webp",
        mime_type=None,
        extension=None,
        file_type=FileType.IMAGE,
    )

    input_value, is_temp_file, file_type_code = normalize_file_input(file, None)

    assert input_value == base64.b64encode(b"image-bytes").decode("utf-8")
    assert is_temp_file is True
    assert file_type_code == 1


def test_explicit_file_type_overrides_inference():
    file = make_file(
        b"image-bytes",
        filename="scan.png",
        mime_type="image/png",
        extension=".png",
        file_type=FileType.IMAGE,
    )

    input_value, is_temp_file, file_type_code = normalize_file_input(file, "pdf")

    assert input_value == base64.b64encode(b"image-bytes").decode("utf-8")
    assert is_temp_file is True
    assert file_type_code == 0


def test_legacy_file_string_is_passed_through():
    input_value, is_temp_file, file_type_code = normalize_file_input("https://example.com/scan.pdf", "auto")

    assert input_value == "https://example.com/scan.pdf"
    assert is_temp_file is False
    assert file_type_code is None


def test_missing_file_input_raises_clear_error():
    with pytest.raises(RuntimeError, match="File is not provided."):
        normalize_file_input(None, "auto")


def invoke_tool_with_mocked_api(monkeypatch, tool_cls, credentials, parameters):
    captured = {}
    module_name = tool_cls.__module__.split(".")[-1]

    def fake_sdk_call(**kwargs):
        captured["kwargs"] = kwargs
        # Return mock result in SDK format
        from paddleocr._api_client.results import OCRResult, DocParsingResult, DocParsingPage, OCRPage

        if tool_cls == TextRecognitionTool:
            return OCRResult(job_id="test-job", pages=[
                OCRPage(pruned_result={"rec_texts": ["hello", "world"]}, ocr_image_url=None)
            ])
        else:
            return DocParsingResult(job_id="test-job", pages=[
                DocParsingPage(markdown_text="# Parsed", markdown_images={}, output_images={})
            ])

    # Mock the SDK client
    from unittest.mock import MagicMock

    fake_client = MagicMock()
    fake_client.ocr = fake_sdk_call
    fake_client.parse_document = fake_sdk_call

    monkeypatch.setattr(f"tools.{module_name}.get_sdk_client", lambda *args: fake_client)
    monkeypatch.setattr(f"tools.{module_name}.base64_to_temp_file", lambda *args: "temp_file.png")
    monkeypatch.setattr(f"tools.{module_name}.cleanup_temp_file", lambda *args: None)

    tool = tool_cls.from_credentials(credentials)
    list(tool._invoke(parameters))
    return captured


def test_text_recognition_sends_normalized_file_to_api(monkeypatch):
    file = make_file(
        b"image-bytes",
        filename="receipt.png",
        mime_type="image/png",
        extension=".png",
        file_type=FileType.IMAGE,
    )

    captured = invoke_tool_with_mocked_api(
        monkeypatch,
        TextRecognitionTool,
        {
            "aistudio_access_token": "token",
            "text_recognition_api_url": "https://example.com/text-recognition",
        },
        {"file": file, "fileType": "auto", "visualize": False},
    )

    # SDK receives file_path (temp file), not base64 directly
    assert "file_path" in captured["kwargs"]
    assert captured["kwargs"]["file_path"] == "temp_file.png"
    assert captured["kwargs"]["options"] is not None
    assert captured["kwargs"]["options"].visualize is False


def test_document_parsing_sends_normalized_file_to_api(monkeypatch):
    file = make_file(
        b"%PDF-1.7", filename="report.pdf", mime_type="application/pdf", extension=".pdf"
    )

    captured = invoke_tool_with_mocked_api(
        monkeypatch,
        DocumentParsingTool,
        {
            "aistudio_access_token": "token",
            "document_parsing_api_url": "https://example.com/document-parsing",
        },
        {"file": file, "fileType": "auto", "markdownIgnoreLabels": "header, footer"},
    )

    assert "file_path" in captured["kwargs"]
    assert captured["kwargs"]["file_path"] == "temp_file.png"
    assert captured["kwargs"]["options"] is not None
    assert captured["kwargs"]["options"].markdown_ignore_labels == ["header", "footer"]


def test_document_parsing_vl_sends_normalized_file_to_api(monkeypatch):
    file = make_file(
        b"image-bytes",
        filename="scan.jpg",
        mime_type="image/jpeg",
        extension=".jpg",
        file_type=FileType.IMAGE,
    )

    captured = invoke_tool_with_mocked_api(
        monkeypatch,
        DocumentParsingVlTool,
        {
            "aistudio_access_token": "token",
            "document_parsing_vl_api_url": "https://example.com/document-parsing-vl",
        },
        {"file": file, "fileType": "auto", "promptLabel": "undefined"},
    )

    assert "file_path" in captured["kwargs"]
    assert captured["kwargs"]["file_path"] == "temp_file.png"
    assert captured["kwargs"]["options"] is not None
    # promptLabel should be filtered out when "undefined"
    assert captured["kwargs"]["options"].prompt_label is None


def load_tool_yaml(tool_name: str) -> dict:
    path = os.path.join(PLUGIN_DIR, "tools", f"{tool_name}.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize(
    "tool_name", ["text_recognition", "document_parsing", "document_parsing_vl"]
)
def test_yaml_exposes_single_file_parameter_as_uploaded_file(tool_name):
    tool_yaml = load_tool_yaml(tool_name)
    parameters = {parameter["name"]: parameter for parameter in tool_yaml["parameters"]}

    assert parameters["file"]["type"] == "file"
    assert parameters["file"]["required"] is True
    assert "file_upload" not in parameters
    assert "URL or base64" in parameters["file"]["human_description"]["en_US"]
    assert "Dify uploaded" in parameters["file"]["human_description"]["en_US"]
