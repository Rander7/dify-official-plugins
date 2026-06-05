#!/usr/bin/env python3
"""
PaddleOCR 手动测试脚本

用途: 在提交 PR 前，手动验证 PaddleOCR SDK 集成的功能

使用方法:
1. 设置环境变量:
   export PADDLEOCR_ACCESS_TOKEN="your_token_here"
   export PADDLEOCR_BASE_URL="your_base_url_here"

2. 运行测试:
   python3 test_manual.py

测试内容:
- OCR 文字识别功能
- 文档解析功能
- Base64 输入处理
"""

import base64
import os
import sys
import tempfile
import urllib.request
from typing import Any

# Test image URL from PaddleOCR
TEST_IMAGE_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
# Simple PDF with text
TEST_PDF_URL = "https://www.africau.edu/images/default/sample.pdf"


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def print_result(test_name: str, success: bool, message: str = "") -> None:
    """Print test result."""
    status = "✅ 通过" if success else "❌ 失败"
    print(f"{status}: {test_name}")
    if message:
        print(f"   {message}")


def download_file(url: str) -> bytes:
    """Download file from URL."""
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def file_to_base64(file_path: str) -> str:
    """Convert file to base64 string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_imports() -> bool:
    """Test that public API imports work."""
    print_section("测试 1: 公开 API 导入")

    try:
        from paddleocr import (
            PaddleOCRClient,
            OCROptions,
            PPStructureV3Options,
            PaddleOCRVLOptions,
            AuthError,
            PaddleOCRAPIError,
        )
        print_result("公开 API 导入", True)
        return True
    except ImportError as e:
        print_result("公开 API 导入", False, f"导入错误: {e}")
        return False


def test_sdk_initialization() -> bool:
    """Test SDK client initialization."""
    print_section("测试 2: SDK 客户端初始化")

    access_token = os.environ.get("PADDLEOCR_ACCESS_TOKEN")
    base_url = os.environ.get("PADDLEOCR_BASE_URL")

    if not access_token:
        print_result("SDK 客户端初始化", False, "请设置环境变量 PADDLEOCR_ACCESS_TOKEN")
        return False

    try:
        from paddleocr import PaddleOCRClient

        # Test with default base_url (None)
        client_default = PaddleOCRClient(
            token=access_token,
            client_platform="dify",
        )
        print_result("SDK 客户端初始化 (默认 base_url)", True)

        # Test with custom base_url if provided
        if base_url:
            client_custom = PaddleOCRClient(
                token=access_token,
                base_url=base_url,
                client_platform="dify",
            )
            print_result("SDK 客户端初始化 (自定义 base_url)", True, f"Base URL: {base_url}")

        return True
    except Exception as e:
        print_result("SDK 客户端初始化", False, f"错误: {e}")
        return False


def test_ocr_with_url() -> bool:
    """Test OCR with file URL."""
    print_section("测试 3: OCR 文字识别 (URL 输入)")

    access_token = os.environ.get("PADDLEOCR_ACCESS_TOKEN")
    base_url = os.environ.get("PADDLEOCR_BASE_URL")

    if not access_token:
        print_result("OCR URL 输入", False, "缺少环境变量")
        return False

    try:
        from paddleocr import PaddleOCRClient, OCROptions

        client = PaddleOCRClient(
            token=access_token,
            base_url=base_url,  # None uses SDK default
            client_platform="dify",
        )

        print(f"  下载测试图片: {TEST_IMAGE_URL}")
        result = client.ocr(
            file_url=TEST_IMAGE_URL,
            options=OCROptions(),
        )

        # Check result structure
        if not result.pages:
            print_result("OCR URL 输入", False, "返回结果中没有 pages")
            return False

        first_page = result.pages[0]
        if not first_page.pruned_result:
            print_result("OCR URL 输入", False, "pruned_result 为空")
            return False

        rec_texts = first_page.pruned_result.get("rec_texts", [])
        if not rec_texts:
            print_result("OCR URL 输入", False, "rec_texts 为空")
            return False

        text_sample = "\n".join(rec_texts[:3]) if len(rec_texts) > 3 else "\n".join(rec_texts)
        print_result("OCR URL 输入", True, f"识别到 {len(rec_texts)} 行文本\n   示例: {text_sample[:50]}...")
        return True

    except Exception as e:
        print_result("OCR URL 输入", False, f"错误: {e}")
        return False


def test_ocr_with_base64() -> bool:
    """Test OCR with base64 input."""
    print_section("测试 4: OCR 文字识别 (Base64 输入)")

    access_token = os.environ.get("PADDLEOCR_ACCESS_TOKEN")
    base_url = os.environ.get("PADDLEOCR_BASE_URL")

    if not access_token:
        print_result("OCR Base64 输入", False, "缺少环境变量")
        return False

    try:
        from paddleocr import PaddleOCRClient, OCROptions

        client = PaddleOCRClient(
            token=access_token,
            base_url=base_url,
            client_platform="dify",
        )

        print(f"  下载测试图片并转换为 Base64")
        image_bytes = download_file(TEST_IMAGE_URL)
        base64_str = base64.b64encode(image_bytes).decode("utf-8")

        # SDK requires file_path for base64, so save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            f.write(base64.b64decode(base64_str))
            temp_file = f.name

        try:
            result = client.ocr(
                file_path=temp_file,
                options=OCROptions(),
            )

            if not result.pages or not result.pages[0].pruned_result:
                print_result("OCR Base64 输入", False, "结果无效")
                return False

            rec_texts = result.pages[0].pruned_result.get("rec_texts", [])
            print_result("OCR Base64 输入", True, f"识别到 {len(rec_texts)} 行文本")
            return True
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    except Exception as e:
        print_result("OCR Base64 输入", False, f"错误: {e}")
        return False


def test_document_parsing() -> bool:
    """Test document parsing."""
    print_section("测试 5: 文档解析 (PDF)")

    access_token = os.environ.get("PADDLEOCR_ACCESS_TOKEN")
    base_url = os.environ.get("PADDLEOCR_BASE_URL")

    if not access_token:
        print_result("文档解析", False, "缺少环境变量")
        return False

    try:
        from paddleocr import PaddleOCRClient, PPStructureV3Options

        client = PaddleOCRClient(
            token=access_token,
            base_url=base_url,
            client_platform="dify",
        )

        # Download PDF to temp file first
        print(f"  下载 PDF 文档: {TEST_PDF_URL}")
        pdf_bytes = download_file(TEST_PDF_URL)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(pdf_bytes)
            temp_pdf = f.name

        try:
            print(f"  解析 PDF 文档...")
            result = client.parse_document(
                model="PP-StructureV3",
                file_path=temp_pdf,
                options=PPStructureV3Options(),
            )

            if not result.pages:
                print_result("文档解析", False, "返回结果中没有 pages")
                return False

            first_page = result.pages[0]
            if not first_page.markdown_text:
                print_result("文档解析", False, "markdown_text 为空")
                return False

            markdown_sample = first_page.markdown_text[:100]
            print_result("文档解析", True, f"解析了 {len(result.pages)} 页\n   示例: {markdown_sample}...")
            return True
        finally:
            if os.path.exists(temp_pdf):
                os.unlink(temp_pdf)

    except Exception as e:
        print_result("文档解析", False, f"错误: {e}")
        return False


def main() -> int:
    """Run all tests."""
    print("\n" + "="*60)
    print("  PaddleOCR 手动测试脚本")
    print("="*60)
    print("\n  环境变量:")
    print(f"    PADDLEOCR_ACCESS_TOKEN: {'已设置' if os.environ.get('PADDLEOCR_ACCESS_TOKEN') else '未设置'}")
    print(f"    PADDLEOCR_BASE_URL: {'已设置' if os.environ.get('PADDLEOCR_BASE_URL') else '未设置（将使用 SDK 默认值）'}")

    # Run tests
    results = []
    results.append(("公开 API 导入", test_imports()))

    # Only run SDK tests if environment is set up
    if os.environ.get("PADDLEOCR_ACCESS_TOKEN"):
        results.append(("SDK 客户端初始化", test_sdk_initialization()))
        results.append(("OCR URL 输入", test_ocr_with_url()))
        results.append(("OCR Base64 输入", test_ocr_with_base64()))
        results.append(("文档解析", test_document_parsing()))
    else:
        print_section("跳过 SDK 功能测试")
        print("  未设置环境变量，仅运行导入测试")

    # Summary
    print_section("测试总结")
    total = len(results)
    passed = sum(1 for _, result in results if result)

    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")

    print(f"\n  总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n  🎉 所有测试通过！可以提交 PR 了。")
        return 0
    else:
        print(f"\n  ⚠️  {total - passed} 个测试失败，请修复后重试。")
        return 1


if __name__ == "__main__":
    sys.exit(main())