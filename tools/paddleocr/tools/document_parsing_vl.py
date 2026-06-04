from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from tools.utils import (
    build_paddleocr_vl_options,
    cleanup_temp_file,
    doc_result_to_legacy_format,
    get_markdown_from_result,
    get_sdk_client,
    normalize_file_input,
    process_images_from_result,
)


class DocumentParsingVlTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """Invoke the PaddleOCR API to parse the document images using a VLM."""
        if "aistudio_access_token" not in self.runtime.credentials:
            raise RuntimeError(
                "The AI Studio access token is not configured or invalid. Please provide it in the plugin settings."
            )
        access_token = self.runtime.credentials["aistudio_access_token"]

        if "document_parsing_vl_api_url" not in self.runtime.credentials:
            raise RuntimeError(
                "The large model document parsing API URL is not configured or invalid. Please provide it in the plugin settings."
            )
        api_url = self.runtime.credentials["document_parsing_vl_api_url"]

        # Normalize file input - returns (input_value, is_temp_file, file_type_code)
        file_input, is_temp_file, file_type_code = normalize_file_input(
            tool_parameters.get("file"), tool_parameters.get("fileType")
        )

        try:
            # Build options from parameters
            options = build_paddleocr_vl_options(tool_parameters)

            # Get SDK client
            client = get_sdk_client(access_token, api_url)

            # Call SDK with PaddleOCR-VL-1.6 model (latest VL model)
            if file_input.startswith(("http://", "https://")):
                result = client.parse_document(
                    model="PaddleOCR-VL-1.6",
                    file_url=file_input,
                    options=options,
                )
            else:
                result = client.parse_document(
                    model="PaddleOCR-VL-1.6",
                    file_path=file_input,
                    options=options,
                )

            # Convert result to legacy format
            legacy_result = doc_result_to_legacy_format(result)

            # Process images
            images, image_path_map, failed_images, blob_messages = process_images_from_result(
                legacy_result, self
            )

            # Get markdown
            markdown = get_markdown_from_result(legacy_result, image_path_map, failed_images)

            for blob_data, blob_meta in blob_messages:
                yield self.create_blob_message(blob_data, meta=blob_meta)

            yield self.create_variable_message("images", images)
            yield self.create_text_message(markdown)
            yield self.create_json_message(legacy_result)

        finally:
            # Clean up temporary file if created
            cleanup_temp_file(file_input, is_temp_file)