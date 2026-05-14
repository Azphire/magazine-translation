from typing import Any, Dict, List, Optional, TypedDict


class TranslationState(TypedDict, total=False):
    image_path: str
    page_num: int

    # PDF-native text boxes from PyMuPDF, already scaled to rendered image pixels.
    pdf_text_lines: List[Dict[str, Any]]

    # OCR fallback lines from RapidOCR.
    raw_ocr: List[Dict[str, Any]]

    # CV fallback text-line boxes detected directly from image pixels.
    cv_text_lines: List[Dict[str, Any]]

    parsed_json: Optional[Dict[str, Any]]
    parser_errors: Optional[str]
    parser_retry_count: int

    memory_dict: Dict[str, str]

    translated_blocks: List[Dict[str, Any]]
    translator_errors: Optional[str]
    translator_retry_count: int

    # Optional compatibility state for old per-page rendering paths.
    flow_context: Dict[str, Any]

    # Document-level translation and layout diagnostics.
    document_translation: Dict[str, Any]
    layout_report: Dict[str, Any]
    layout_errors: Optional[str]
    layout_retry_count: int
    layout_suggestions: Dict[str, Any]

    output_image_path: Optional[str]
    output_image_paths: List[str]
    debug_paths: Dict[str, str]
