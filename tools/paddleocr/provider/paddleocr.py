from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

from tools.document_parsing import DocumentParsingTool
from tools.document_parsing_vl import DocumentParsingVlTool
from tools.text_recognition import TextRecognitionTool
from tools.utils import get_sdk_client


class PaddleocrProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        if "aistudio_access_token" not in credentials:
            raise ToolProviderCredentialValidationError(
                "AI Studio access token must be provided"
            )

        # Get base_url (optional, uses SDK default if not provided)
        base_url = credentials.get("base_url")

        # Test with OCR (works for all models)
        test_file = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"

        try:
            client = get_sdk_client(
                access_token=credentials["aistudio_access_token"],
                base_url=base_url,
            )
            client.ocr(file_url=test_file)
        except Exception as e:
            # Check for specific PaddleOCR error types
            try:
                from paddleocr import AuthError, PaddleOCRAPIError

                if isinstance(e, AuthError):
                    raise ToolProviderCredentialValidationError(
                        f"Authentication failed: {e}"
                    ) from e
                if isinstance(e, PaddleOCRAPIError):
                    raise ToolProviderCredentialValidationError(
                        f"PaddleOCR API error: {e}"
                    ) from e
            except ImportError:
                pass
            raise ToolProviderCredentialValidationError(
                f"Validation failed: {e}"
            ) from e