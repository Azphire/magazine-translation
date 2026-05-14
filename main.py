from __future__ import annotations

import os
from typing import Dict, List, Optional

import fitz
from PIL import Image

from agents.critics import parser_critic_node
from agents.document_translator import translate_document
from agents.layout_critic import layout_critic_node
from agents.memory import MemoryAgent
from agents.renderer import render_document_pages
from agents.vision_parser import vision_parser_node
from config import MAX_LAYOUT_RETRIES, MAX_RETRIES, OUTPUT_DIR, PDF_RENDER_ZOOM, TEMP_DIR
from core.state import TranslationState
from utils.logger import logger


def extract_pdf_text_lines(page: fitz.Page, zoom: float) -> List[Dict]:
    """
    Extract vector text boxes from the PDF and scale them to rendered image pixels.

    These boxes are used for erasing and coverage checks. Translation still uses
    parser blocks because vector text alone does not encode magazine semantics.
    """
    lines: List[Dict] = []
    data = page.get_text("dict")
    index = 0
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            bbox = line.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                lines.append({
                    "id": index,
                    "box": [int(round(x1 * zoom)), int(round(y1 * zoom)), int(round(x2 * zoom)), int(round(y2 * zoom))],
                    "text": text,
                    "source": "pdf_line",
                })
                index += 1
            for span in spans:
                span_text = str(span.get("text", "")).strip()
                span_box = span.get("bbox")
                if not span_text or not span_box:
                    continue
                x1, y1, x2, y2 = span_box
                lines.append({
                    "id": index,
                    "box": [int(round(x1 * zoom)), int(round(y1 * zoom)), int(round(x2 * zoom)), int(round(y2 * zoom))],
                    "text": span_text,
                    "source": "pdf_span",
                })
                index += 1
    logger.info(f"[PDF Text] Extracted {len(lines)} vector line/span boxes.")
    return lines


def render_pdf_page_to_image(page: fitz.Page, page_num: int, temp_dir: str, zoom: float) -> str:
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    os.makedirs(temp_dir, exist_ok=True)
    image_path = os.path.join(temp_dir, f"page_{page_num}.jpg")
    pix.save(image_path)
    return image_path


def run_parser_with_retries(state: TranslationState) -> TranslationState:
    while state.get("parser_retry_count", 0) <= MAX_RETRIES:
        state.update(vision_parser_node(state))
        state.update(parser_critic_node(state))
        if not state.get("parser_errors"):
            break
        logger.warning(
            f"[Retry] Parser retry {state.get('parser_retry_count')}/{MAX_RETRIES}: "
            f"{state.get('parser_errors')}"
        )
    return state


def build_pdf_from_images(image_paths: List[str], output_pdf_path: str) -> None:
    """Insert every rendered page image into a new PDF page explicitly."""
    if not image_paths:
        raise ValueError("No images to stitch into PDF.")
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    pdf = fitz.open()
    inserted = 0
    for image_path in image_paths:
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"[PDF Stitch] Missing output image skipped: {image_path}")
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        page = pdf.new_page(width=width, height=height)
        page.insert_image(fitz.Rect(0, 0, width, height), filename=image_path)
        inserted += 1
    if pdf.page_count == 0:
        raise ValueError("PDF stitch failed: no valid image pages were inserted.")
    pdf.save(output_pdf_path, deflate=True)
    pdf.close()
    logger.info(f"✅ Stitched {inserted} image pages into PDF: {output_pdf_path}")


def parse_pdf_pages(doc: fitz.Document, memory_agent: MemoryAgent) -> List[TranslationState]:
    page_states: List[TranslationState] = []
    for page_num in range(len(doc)):
        logger.info(f"\n--- Parse page {page_num + 1}/{len(doc)} ---")
        page = doc.load_page(page_num)
        image_path = render_pdf_page_to_image(page, page_num, TEMP_DIR, PDF_RENDER_ZOOM)
        pdf_text_lines = extract_pdf_text_lines(page, PDF_RENDER_ZOOM)

        state = TranslationState(
            image_path=image_path,
            page_num=page_num,
            pdf_text_lines=pdf_text_lines,
            memory_dict=memory_agent.get_memory_context(),
            parser_retry_count=0,
            translator_retry_count=0,
            flow_context={},
        )
        state = run_parser_with_retries(state)
        if state.get("parser_errors"):
            logger.error(f"[Page {page_num}] Parser failed but rendering will still be attempted: {state.get('parser_errors')}")
        page_states.append(state)
    return page_states


def render_with_layout_critic(page_states: List[TranslationState], document_translation: Dict) -> Dict:
    """
    Render the document and retry with adjusted typography if layout checks fail.
    """
    suggestions: Dict = {}
    last_result: Dict = {}
    for attempt in range(MAX_LAYOUT_RETRIES + 1):
        logger.info(f"\n--- Layout render attempt {attempt + 1}/{MAX_LAYOUT_RETRIES + 1} ---")
        last_result = render_document_pages(
            page_states=page_states,
            document_translation=document_translation,
            layout_suggestions=suggestions,
        )
        critic_result = layout_critic_node({
            "layout_report": last_result.get("layout_report", {}),
            "layout_retry_count": attempt,
        })
        if not critic_result.get("layout_errors"):
            logger.info("[Layout] Accepted by layout critic.")
            break
        logger.warning(f"[Layout] Rejected by critic: {critic_result.get('layout_errors')}")
        suggestions = critic_result.get("layout_suggestions", {}) or {}
    return last_result


def update_memory_from_document(memory_agent: MemoryAgent, page_states: List[TranslationState], document_translation: Dict) -> None:
    pairs = []
    for state in page_states:
        for block in state.get("translated_blocks", []) or []:
            if block.get("source_text") and block.get("target_text"):
                pairs.append(block)
    for elem in document_translation.get("global_body_flow") or []:
        pairs.append({
            "source_text": " / ".join(elem.get("source_refs", [])),
            "target_text": elem.get("text", ""),
        })
    if pairs:
        memory_agent.update_memory(pairs)


def process_pdf(pdf_path: str, output_pdf_path: str) -> None:
    logger.info(f"=== Starting Magazine Translation Pipeline v6: {pdf_path} ===")
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    memory_agent = MemoryAgent()
    doc = fitz.open(pdf_path)
    page_states = parse_pdf_pages(doc, memory_agent)

    logger.info("\n=== Document-level translation and cross-page continuity analysis ===")
    page_states, document_translation = translate_document(
        page_states,
        memory_agent.get_memory_context(),
        max_retries=MAX_RETRIES,
    )

    render_result = render_with_layout_critic(page_states, document_translation)
    output_images = render_result.get("output_image_paths", [])
    update_memory_from_document(memory_agent, page_states, document_translation)

    if output_images:
        build_pdf_from_images(output_images, output_pdf_path)
    else:
        logger.error("No output pages were generated.")


def process_sample_images(image_paths: List[str], output_dir: Optional[str] = None) -> None:
    """
    Debug helper for already-rendered page images.

    It cannot use PDF-native boxes, so erasing falls back to OCR and CV text-line detection.
    """
    _ = output_dir or OUTPUT_DIR
    memory_agent = MemoryAgent()
    page_states: List[TranslationState] = []
    for page_num, image_path in enumerate(image_paths):
        state = TranslationState(
            image_path=image_path,
            page_num=page_num,
            pdf_text_lines=[],
            memory_dict=memory_agent.get_memory_context(),
            parser_retry_count=0,
            translator_retry_count=0,
            flow_context={},
        )
        page_states.append(run_parser_with_retries(state))

    page_states, document_translation = translate_document(
        page_states,
        memory_agent.get_memory_context(),
        max_retries=MAX_RETRIES,
    )
    render_with_layout_critic(page_states, document_translation)


if __name__ == "__main__":
    input_pdf = "./data/input/magazine.pdf"
    output_pdf = "./data/output/magazine_zh.pdf"
    process_pdf(input_pdf, output_pdf)
