import base64
import logging
import os
import re
import tempfile
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse

from dify_plugin.file.file import File
from dify_plugin.invocations.file import UploadFileResponse

# Pre-compiled regex patterns for performance
HTML_IMG_PATTERN = re.compile(r'(<img[^>]*src=")([^"]+)(")')

# Template for failed image replacement pattern
FAILED_IMG_TAG_TEMPLATE = r'<img[^>]*src="[^"]*{escaped_path}[^"]*"[^>]*>'


logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def extract_base_url(api_url: str) -> str:
    """Extract base URL from full API URL.

    The SDK requires a base URL (e.g., https://example.com)
    but users provide the full API URL (e.g., https://example.com/ocr).
    This function extracts the base URL by removing the endpoint path.

    Args:
        api_url: Full API URL

    Returns:
        Base URL without endpoint path
    """
    parsed = urlparse(api_url)
    # Remove common PaddleOCR endpoints
    path = parsed.path
    if path in ("/ocr", "/layout-parsing", "/paddleocr"):
        path = ""
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def convert_file_type(file_type: str | None) -> int | None:
    """Convert file type string to API parameter value.

    Args:
        file_type: "auto", "pdf", or "image"

    Returns:
        0 for PDF, 1 for image, None for auto
    """
    if file_type == "pdf":
        return 0
    elif file_type == "image":
        return 1
    else:  # "auto" or None
        return None


def normalize_file_input(file_value: Any, file_type: str | None) -> Tuple[str, bool, int | None]:
    """Normalize PaddleOCR file input.

    Returns:
        A tuple of (input_value, is_temp_file, file_type_code):
        - input_value: URL, file path (temp or regular), or base64 string
        - is_temp_file: True if the value is a temporary file path that should be deleted
        - file_type_code: 0 for PDF, 1 for image, None for auto
    """
    if file_value is None or (isinstance(file_value, str) and file_value == ""):
        raise RuntimeError("File is not provided.")

    explicit_file_type = convert_file_type(file_type)

    if isinstance(file_value, File):
        encoded_file = base64.b64encode(file_value.blob).decode("utf-8")
        temp_file = base64_to_temp_file(encoded_file, infer_file_extension(file_value))
        file_type_code = explicit_file_type if explicit_file_type is not None else infer_file_type(file_value)
        return temp_file, True, file_type_code

    if isinstance(file_value, str):
        # Check if it's a URL
        if file_value.startswith(("http://", "https://")):
            return file_value, False, explicit_file_type
        # Check if it's base64 (data URL or raw)
        if file_value.startswith("data:") or is_likely_base64(file_value):
            temp_file = base64_to_temp_file(extract_base64(file_value))
            return temp_file, True, explicit_file_type
        # It's a file path
        return file_value, False, explicit_file_type

    raise RuntimeError("File must be a Dify file, URL, or base64-encoded string.")


def infer_file_type(file_value: File) -> int | None:
    mime_type = (file_value.mime_type or "").lower()
    if mime_type == "application/pdf":
        return 0
    if mime_type.startswith("image/"):
        return 1

    extension = normalize_extension(file_value.extension)
    if extension is None:
        extension = normalize_extension(os.path.splitext(file_value.filename or "")[1])

    if extension == ".pdf":
        return 0
    if extension in IMAGE_EXTENSIONS:
        return 1

    return None


def infer_file_extension(file_value: File) -> str:
    mime_type = (file_value.mime_type or "").lower()
    if mime_type == "application/pdf":
        return ".pdf"
    if mime_type.startswith("image/"):
        ext = mime_type.split("/")[-1]
        return f".{ext}"

    extension = normalize_extension(file_value.extension)
    if extension is None:
        extension = normalize_extension(os.path.splitext(file_value.filename or "")[1])

    return extension if extension else ".png"


def normalize_extension(extension: str | None) -> str | None:
    if not extension:
        return None
    extension = extension.lower()
    return extension if extension.startswith(".") else f".{extension}"


def extract_base64(data_url: str) -> str:
    if data_url.startswith("data:"):
        return data_url.split(",", 1)[1]
    return data_url


def is_likely_base64(s: str) -> bool:
    if len(s) < 32:
        return False
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False


def base64_to_temp_file(base64_str: str, suffix: str = ".png") -> str:
    """Save base64 string to a temporary file.

    Args:
        base64_str: Base64 encoded string
        suffix: File extension suffix

    Returns:
        Path to the temporary file
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(base64.b64decode(base64_str))
        return f.name


def cleanup_temp_file(file_path: str, is_temp: bool) -> None:
    """Clean up temporary file if it exists and is marked as temporary.

    Args:
        file_path: Path to the file
        is_temp: True if the file is a temporary file that should be deleted
    """
    if is_temp and file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {file_path}: {e}")


def get_sdk_client(access_token: str, api_url: str) -> Any:
    """Get PaddleOCR SDK client.

    Args:
        access_token: AI Studio access token
        api_url: API URL (full endpoint URL or base URL)

    Returns:
        PaddleOCRClient instance
    """
    from paddleocr._api_client import PaddleOCRClient

    base_url = extract_base_url(api_url)
    return PaddleOCRClient(
        token=access_token,
        base_url=base_url,
        client_platform="dify",
    )


def ocr_result_to_legacy_format(result: Any) -> dict:
    """Convert SDK OCRResult to legacy API format.

    Args:
        result: SDK OCRResult

    Returns:
        Legacy format dict
    """
    return {
        "result": {
            "ocrResults": [
                {
                    "prunedResult": page.pruned_result,
                    "ocrImageUrl": page.ocr_image_url,
                }
                for page in result.pages
            ]
        }
    }


def doc_result_to_legacy_format(result: Any) -> dict:
    """Convert SDK DocParsingResult to legacy API format.

    Args:
        result: SDK DocParsingResult

    Returns:
        Legacy format dict
    """
    return {
        "result": {
            "layoutParsingResults": [
                {
                    "markdown": {
                        "text": page.markdown_text,
                        "images": page.markdown_images,
                    },
                    "outputImages": page.output_images,
                }
                for page in result.pages
            ]
        }
    }


def build_ocr_options(params: dict[str, Any]) -> Any:
    """Build OCROptions from parameters.

    Args:
        params: Tool parameters

    Returns:
        OCROptions instance or None
    """
    from paddleocr._api_client.models import OCROptions

    option_map = {
        "useDocOrientationClassify": "use_doc_orientation_classify",
        "useDocUnwarping": "use_doc_unwarping",
        "useTextlineOrientation": "use_textline_orientation",
        "textDetLimitSideLen": "text_det_limit_side_len",
        "textDetLimitType": "text_det_limit_type",
        "textDetThresh": "text_det_thresh",
        "textDetBoxThresh": "text_det_box_thresh",
        "textDetUnclipRatio": "text_det_unclip_ratio",
        "textRecScoreThresh": "text_rec_score_thresh",
        "visualize": "visualize",
    }

    options_dict = {}
    for api_name, option_name in option_map.items():
        if api_name in params and params[api_name] is not None:
            options_dict[option_name] = params[api_name]

    return OCROptions(**options_dict) if options_dict else None


def build_pp_structure_v3_options(params: dict[str, Any]) -> Any:
    """Build PPStructureV3Options from parameters.

    Args:
        params: Tool parameters

    Returns:
        PPStructureV3Options instance or None
    """
    from paddleocr._api_client.models import PPStructureV3Options

    option_map = {
        "useDocOrientationClassify": "use_doc_orientation_classify",
        "useDocUnwarping": "use_doc_unwarping",
        "useTextlineOrientation": "use_textline_orientation",
        "useSealRecognition": "use_seal_recognition",
        "useTableRecognition": "use_table_recognition",
        "useFormulaRecognition": "use_formula_recognition",
        "useChartRecognition": "use_chart_recognition",
        "useRegionDetection": "use_region_detection",
        "formatBlockContent": "format_block_content",
        "layoutThreshold": "layout_threshold",
        "layoutNms": "layout_nms",
        "layoutUnclipRatio": "layout_unclip_ratio",
        "layoutMergeBboxesMode": "layout_merge_bboxes_mode",
        "textDetLimitSideLen": "text_det_limit_side_len",
        "textDetLimitType": "text_det_limit_type",
        "textDetThresh": "text_det_thresh",
        "textDetBoxThresh": "text_det_box_thresh",
        "textDetUnclipRatio": "text_det_unclip_ratio",
        "textRecScoreThresh": "text_rec_score_thresh",
        "sealDetLimitSideLen": "seal_det_limit_side_len",
        "sealDetLimitType": "seal_det_limit_type",
        "sealDetThresh": "seal_det_thresh",
        "sealDetBoxThresh": "seal_det_box_thresh",
        "sealDetUnclipRatio": "seal_det_unclip_ratio",
        "sealRecScoreThresh": "seal_rec_score_thresh",
        "useWiredTableCellsTransToHtml": "use_wired_table_cells_trans_to_html",
        "useWirelessTableCellsTransToHtml": "use_wireless_table_cells_trans_to_html",
        "useTableOrientationClassify": "use_table_orientation_classify",
        "useOcrResultsWithTableCells": "use_ocr_results_with_table_cells",
        "useE2eWiredTableRecModel": "use_e2e_wired_table_rec_model",
        "useE2eWirelessTableRecModel": "use_e2e_wireless_table_rec_model",
        "markdownIgnoreLabels": "markdown_ignore_labels",
        "prettifyMarkdown": "prettify_markdown",
        "showFormulaNumber": "show_formula_number",
        "visualize": "visualize",
    }

    options_dict = {}
    for api_name, option_name in option_map.items():
        if api_name in params and params[api_name] is not None:
            value = params[api_name]
            # Handle markdownIgnoreLabels conversion
            if api_name == "markdownIgnoreLabels" and isinstance(value, str):
                value = [label.strip() for label in value.split(",") if label.strip()]
            options_dict[option_name] = value

    return PPStructureV3Options(**options_dict) if options_dict else None


def build_paddleocr_vl_options(params: dict[str, Any]) -> Any:
    """Build PaddleOCRVLOptions from parameters.

    Args:
        params: Tool parameters

    Returns:
        PaddleOCRVLOptions instance or None
    """
    from paddleocr._api_client.models import PaddleOCRVLOptions

    option_map = {
        "useDocOrientationClassify": "use_doc_orientation_classify",
        "useDocUnwarping": "use_doc_unwarping",
        "useLayoutDetection": "use_layout_detection",
        "useChartRecognition": "use_chart_recognition",
        "useSealRecognition": "use_seal_recognition",
        "formatBlockContent": "format_block_content",
        "layoutThreshold": "layout_threshold",
        "layoutNms": "layout_nms",
        "layoutUnclipRatio": "layout_unclip_ratio",
        "layoutMergeBboxesMode": "layout_merge_bboxes_mode",
        "layoutShapeMode": "layout_shape_mode",
        "promptLabel": "prompt_label",
        "repetitionPenalty": "repetition_penalty",
        "temperature": "temperature",
        "topP": "top_p",
        "minPixels": "min_pixels",
        "maxPixels": "max_pixels",
        "maxNewTokens": "max_new_tokens",
        "mergeLayoutBlocks": "merge_layout_blocks",
        "markdownIgnoreLabels": "markdown_ignore_labels",
        "prettifyMarkdown": "prettify_markdown",
        "showFormulaNumber": "show_formula_number",
        "restructurePages": "restructure_pages",
        "mergeTables": "merge_tables",
        "relevelTitles": "relevel_titles",
        "visualize": "visualize",
    }

    options_dict = {}
    for api_name, option_name in option_map.items():
        if api_name in params and params[api_name] is not None:
            value = params[api_name]
            # Handle promptLabel conversion
            if api_name == "promptLabel" and value == "undefined":
                continue
            # Handle markdownIgnoreLabels conversion
            if api_name == "markdownIgnoreLabels" and isinstance(value, str):
                value = [label.strip() for label in value.split(",") if label.strip()]
            options_dict[option_name] = value

    return PaddleOCRVLOptions(**options_dict) if options_dict else None


def extract_image_urls_from_markdown(markdown: str) -> List[str]:
    """Extract image URLs from markdown"""
    image_pattern = re.compile(r'<img[^>]*src="([^"]*)"[^>]*>', re.IGNORECASE)
    matches = image_pattern.findall(markdown)
    return matches


def replace_markdown_image_paths(
    markdown: str,
    image_path_map: dict[str, UploadFileResponse],
    failed_images: Optional[List[str]] = None,
) -> str:
    """Replace image paths in HTML img tags with uploaded URLs.

    Handles the PaddleOCR standard image format.
    For failed images (no preview URL available), replaces with placeholder text.
    """
    if failed_images is None:
        failed_images = []

    logger.debug(
        f"Replacing image paths in markdown - {len(image_path_map)} images available, {len(failed_images)} need placeholder"
    )

    # Replace successful images using pre-compiled regex
    replaced_count = 0
    for image_path, upload_response in image_path_map.items():
        if upload_response.preview_url:
            original_markdown = markdown
            markdown = HTML_IMG_PATTERN.sub(
                lambda m: (
                    f"{m.group(1)}{upload_response.preview_url}{m.group(3)}"
                    if m.group(2) == image_path
                    else m.group(0)
                ),
                markdown,
            )
            if markdown != original_markdown:
                replaced_count += 1
                logger.debug(f"Replaced image path {image_path} with {upload_response.preview_url}")

    # Handle images that couldn't get URLs - replace with placeholder
    placeholder_count = 0
    for failed_path in failed_images:
        escaped_path = re.escape(failed_path)
        pattern = FAILED_IMG_TAG_TEMPLATE.format(escaped_path=escaped_path)
        original_markdown = markdown
        markdown = re.sub(pattern, "[Image unavailable]", markdown)
        if markdown != original_markdown:
            placeholder_count += 1
            logger.debug(f"Replaced failed image {failed_path} with placeholder")

    logger.debug(
        f"Markdown replacement completed - {replaced_count} URL replacements, {placeholder_count} placeholders"
    )
    return markdown


def process_images_from_result(
    result: dict, tool_instance
) -> Tuple[
    List[UploadFileResponse], dict[str, UploadFileResponse], List[str], List[Tuple[bytes, dict]]
]:
    """Extract and process images from API result

    Args:
        result: API response result
        tool_instance: Tool instance for file operations
    """
    images = []
    image_path_map = {}  # key: image path, value: UploadFileResponse
    failed_images = []  # images that failed to process
    blob_messages = []  # blob messages to yield: [(data, meta), ...]
    image_counter = 0

    logger.debug("Processing images from API result")

    for item in result.get("result", {}).get("layoutParsingResults", []):
        markdown_data = item.get("markdown", {})
        if markdown_data:
            # Get image dictionary {path: url} from markdown
            image_dict = markdown_data.get("images", {})
            if image_dict:
                logger.debug(
                    f"Found {len(image_dict)} images to process: {list(image_dict.keys())}"
                )
            else:
                logger.debug("No images found in this markdown item")

            for image_path, image_url in image_dict.items():
                if image_path in image_path_map:
                    # Already processed this path
                    logger.debug(f"Skipping already processed image: {image_path}")
                    continue

                logger.debug(f"Processing image: {image_path} -> {image_url}")

                image_processed_successfully = False

                try:
                    # Download image first
                    try:
                        image_bytes = download_image_from_url(image_url)
                    except Exception as download_error:
                        logger.warning(
                            f"Failed to download image {image_path} from {image_url}: {download_error}"
                        )
                        # Cannot download - cannot create blob message, mark as failed for markdown
                        failed_images.append(image_path)
                        continue

                    # Upload image to dify with error handling
                    file_name = f"paddleocr_image_{image_counter}.jpg"
                    logger.debug(f"Uploading image {image_path} as {file_name}")

                    try:
                        upload_response = tool_instance.session.file.upload(
                            file_name, image_bytes, "image/jpeg"
                        )
                        images.append(upload_response)
                        image_path_map[image_path] = upload_response
                        image_counter += 1

                        logger.debug(
                            f"Successfully uploaded image {image_path}, preview_url: {upload_response.preview_url}"
                        )

                        # Check if upload was successful but no preview URL
                        if not upload_response.preview_url:
                            logger.warning(
                                f"No preview URL for uploaded image {image_path}, creating blob message as fallback"
                            )
                            blob_messages.append(
                                (image_bytes, {"filename": file_name, "mime_type": "image/jpeg"})
                            )
                            failed_images.append(image_path)
                        else:
                            image_processed_successfully = True

                    except Exception as upload_error:
                        logger.error(f"Failed to upload image {image_path} to dify: {upload_error}")
                        # Create blob message as fallback when upload fails
                        logger.info(
                            f"Creating blob message as fallback for failed upload of {image_path}"
                        )
                        blob_messages.append(
                            (image_bytes, {"filename": file_name, "mime_type": "image/jpeg"})
                        )
                        failed_images.append(image_path)

                except Exception as e:
                    logger.error(f"Unexpected error processing image {image_path}: {e}")
                    failed_images.append(image_path)
                    continue

                if image_processed_successfully:
                    logger.debug(f"Successfully processed image {image_path}")

    logger.info(
        f"Image processing completed - successful: {len(images)}, markdown-failed: {len(failed_images)}, blob-messages: {len(blob_messages)}"
    )
    if failed_images:
        logger.warning(f"Images that failed processing (no URL available): {failed_images}")

    return images, image_path_map, failed_images, blob_messages


def get_markdown_from_result(
    result: dict,
    image_path_map: Optional[dict[str, UploadFileResponse]] = None,
    failed_images: Optional[List[str]] = None,
) -> str:
    """Extract markdown text from result, replace image references if image path mapping is provided"""
    markdown_text_list = []
    for item in result.get("result", {}).get("layoutParsingResults", []):
        markdown_text = item.get("markdown", {}).get("text")
        if markdown_text is not None:
            if image_path_map or failed_images:
                markdown_text = replace_markdown_image_paths(
                    markdown_text, image_path_map or {}, failed_images
                )
            markdown_text_list.append(markdown_text)
    return "\n\n".join(markdown_text_list)


def download_image_from_url(image_url: str) -> bytes:
    """Download image from URL and return image data and MIME type"""
    import requests

    try:
        logger.debug(f"Downloading image from URL: {image_url}")
        resp = requests.get(image_url, timeout=(10, 600))
        resp.raise_for_status()

        logger.debug(
            f"Successfully downloaded image from {image_url}, size: {len(resp.content)} bytes"
        )
        return resp.content
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout downloading image from {image_url}: {e}")
        raise RuntimeError(f"Failed to download image from {image_url}: timeout") from e
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error downloading image from {image_url}: {e}")
        raise RuntimeError(f"Failed to download image from {image_url}: network error") from e
    except Exception as e:
        logger.error(f"Unexpected error downloading image from {image_url}: {e}")
        raise RuntimeError(f"Failed to download image from {image_url}: {e}") from e