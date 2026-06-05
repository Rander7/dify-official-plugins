from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from tools.utils import (
    build_pp_structure_v3_options,
    cleanup_temp_file,
    get_sdk_client,
    normalize_file_input,
)


class DocumentParsingTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """Invoke the PaddleOCR API to parse the document images."""
        if "aistudio_access_token" not in self.runtime.credentials:
            raise RuntimeError(
                "The AI Studio access token is not configured or invalid. Please provide it in the plugin settings."
            )
        access_token = self.runtime.credentials["aistudio_access_token"]

        # Get base_url (optional, uses SDK default if not provided)
        base_url = self.runtime.credentials.get("base_url")

        # Normalize file input - returns (input_value, is_temp_file, file_type_code)
        file_input, is_temp_file, file_type_code = normalize_file_input(
            tool_parameters.get("file"), tool_parameters.get("fileType")
        )

        try:
            # Build options from parameters
            options = build_pp_structure_v3_options(tool_parameters)

            # Get SDK client
            client = get_sdk_client(access_token, base_url)

            # Call SDK with PP-StructureV3 model
            if file_input.startswith(("http://", "https://")):
                result = client.parse_document(
                    model="PP-StructureV3",
                    file_url=file_input,
                    options=options,
                )
            else:
                result = client.parse_document(
                    model="PP-StructureV3",
                    file_path=file_input,
                    options=options,
                )

            # Process images from SDK result
            images = []
            image_path_map = {}
            failed_images = []

            for page in result.pages:
                if page.markdown_images:
                    image_dict = page.markdown_images
                    if image_dict:
                        for image_path, image_url in image_dict.items():
                            if image_path in image_path_map:
                                continue
                            try:
                                import requests
                                image_bytes = requests.get(image_url, timeout=(10, 600)).content
                                file_name = f"paddleocr_image_{len(images)}.jpg"
                                upload_response = self.session.file.upload(
                                    file_name, image_bytes, "image/jpeg"
                                )
                                images.append(upload_response)
                                image_path_map[image_path] = upload_response
                                if not upload_response.preview_url:
                                    failed_images.append(image_path)
                            except Exception as e:
                                self.runtime.logger.warning(f"Failed to process image {image_path}: {e}")
                                failed_images.append(image_path)

            # Build markdown with image replacement
            markdown_text_list = []
            for page in result.pages:
                markdown_text = page.markdown_text
                if markdown_text is not None:
                    # Replace image paths with uploaded URLs
                    for image_path, upload_response in image_path_map.items():
                        if upload_response.preview_url:
                            markdown_text = markdown_text.replace(
                                f'src="{image_path}"',
                                f'src="{upload_response.preview_url}"'
                            )
                        else:
                            markdown_text = markdown_text.replace(
                                f'src="{image_path}"',
                                'src="[Image unavailable]"'
                            )
                    markdown_text_list.append(markdown_text)

            # Return raw SDK result as JSON
            yield self.create_json_message({
                "job_id": result.job_id,
                "pages": [
                    {
                        "markdown_text": page.markdown_text,
                        "markdown_images": page.markdown_images,
                        "output_images": page.output_images,
                    }
                    for page in result.pages
                ]
            })

        finally:
            # Clean up temporary file if created
            cleanup_temp_file(file_input, is_temp_file)