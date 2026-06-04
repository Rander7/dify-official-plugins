from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
from paddleocr._api_client import PaddleOCRClient
from paddleocr._api_client.errors import AuthError, PaddleOCRAPIError

from tools.document_parsing import DocumentParsingTool
from tools.document_parsing_vl import DocumentParsingVlTool
from tools.text_recognition import TextRecognitionTool
from tools.utils import extract_base_url


class PaddleocrProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        if "aistudio_access_token" not in credentials:
            raise ToolProviderCredentialValidationError(
                "AI Studio access token must be provided"
            )

        api_url_keys = (
            "text_recognition_api_url",
            "document_parsing_api_url",
            "document_parsing_vl_api_url",
        )
        tool_classes = (
            TextRecognitionTool,
            DocumentParsingTool,
            DocumentParsingVlTool,
        )
        test_file = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"

        if not any(key in credentials for key in api_url_keys):
            raise ToolProviderCredentialValidationError(
                "You should provide at least one API URL"
            )

        for api_url_key in api_url_keys:
            if api_url_key in credentials:
                try:
                    self._test_tool_validation(
                        credentials, api_url_key, test_file
                    )
                except AuthError as e:
                    raise ToolProviderCredentialValidationError(
                        f"Authentication failed: {e}"
                    ) from e
                except PaddleOCRAPIError as e:
                    raise ToolProviderCredentialValidationError(
                        f"PaddleOCR API error: {e}"
                    ) from e
                except Exception as e:
                    raise ToolProviderCredentialValidationError(
                        f"Validation failed: {e}"
                    ) from e

    def _test_tool_validation(
        self, credentials: dict[str, Any], api_url_key: str, test_file: str
    ) -> None:
        """Test tool validation using SDK.

        Args:
            credentials: Provider credentials
            api_url_key: Key for the API URL in credentials
            test_file: Test file URL
        """
        access_token = credentials["aistudio_access_token"]
        api_url = credentials[api_url_key]

        # Extract base URL and create SDK client
        base_url = extract_base_url(api_url)
        client = PaddleOCRClient(
            token=access_token,
            base_url=base_url,
            client_platform="dify",
        )

        # Test with OCR (works for any API URL)
        try:
            client.ocr(file_url=test_file)
        except Exception as e:
            # Re-raise to be caught by _validate_credentials
            raise